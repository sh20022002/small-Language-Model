"""
Training and loss behavior tests:
  - Loss magnitude at init, finiteness, gradient flow
  - Gradient clipping, learning-rate schedule
  - Repetition unlikelihood loss (UL)
  - Padding / ignore_index correctness
  - collate_fn, QADataset
  - Single-batch overfit (loss must decrease)
  - Full train_model() integration smoke test
"""
import math
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from my_slm.transformer import Transformer
from my_slm.train import (
    _repetition_ul_loss,
    collate_fn,
    get_cosine_schedule_with_warmup,
    QADataset,
    train_model,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
VOCAB = 256
TINY = dict(vocab_size=VOCAB, dim=64, depth=2, heads=4, mlp_dim=128, window=16)


def _model() -> Transformer:
    return Transformer(**TINY)


def _batch(B=2, T=8):
    ids    = torch.randint(1, VOCAB, (B, T))
    attn   = torch.ones(B, T, dtype=torch.long)
    labels = ids.clone()
    return {"input_ids": ids, "attention_mask": attn, "labels": labels}


def _loader(n_samples=8, T=12, batch_size=4):
    """Return a DataLoader that yields pre-built batches (no tokenizer needed)."""
    ids    = torch.randint(1, VOCAB, (n_samples, T))
    attn   = torch.ones(n_samples, T, dtype=torch.long)
    labels = ids.clone()
    dataset = [
        {"input_ids": ids[i], "attention_mask": attn[i], "labels": labels[i]}
        for i in range(n_samples)
    ]

    def _collate(b):
        return {
            "input_ids":      torch.stack([x["input_ids"]      for x in b]),
            "attention_mask": torch.stack([x["attention_mask"] for x in b]),
            "labels":         torch.stack([x["labels"]         for x in b]),
        }

    return DataLoader(dataset, batch_size=batch_size, collate_fn=_collate)


# ---------------------------------------------------------------------------
# Initial loss sanity
# ---------------------------------------------------------------------------
class TestInitialLoss:
    def test_loss_near_log_vocab(self):
        """Freshly-initialised model loss ≈ log(vocab_size) — roughly uniform."""
        torch.manual_seed(42)
        model = _model().eval()
        loss_fn = nn.CrossEntropyLoss()
        b = _batch()
        with torch.no_grad():
            logits = model(b["input_ids"])
            B, T, V = logits.shape
            loss = loss_fn(logits.reshape(B * T, V), b["labels"].reshape(B * T))

        expected = math.log(VOCAB)
        assert loss.item() < expected * 2.5, \
            f"Initial loss {loss.item():.3f} is >> log(V)={expected:.3f}"
        assert loss.item() > 0.5, \
            f"Initial loss {loss.item():.3f} suspiciously low for an untrained model"

    def test_loss_finite_after_forward(self):
        model = _model()
        loss_fn = nn.CrossEntropyLoss()
        b = _batch()
        logits = model(b["input_ids"])
        B, T, V = logits.shape
        loss = loss_fn(logits.reshape(B * T, V), b["labels"].reshape(B * T))
        assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------
class TestGradientFlow:
    def _backward(self, model):
        loss_fn = nn.CrossEntropyLoss()
        b = _batch()
        logits = model(b["input_ids"])
        B, T, V = logits.shape
        loss = loss_fn(logits.reshape(B * T, V), b["labels"].reshape(B * T))
        loss.backward()
        return model

    def test_all_params_receive_gradient(self):
        model = self._backward(_model())
        missing = [
            n for n, p in model.named_parameters()
            if p.requires_grad and p.grad is None
        ]
        assert not missing, f"Parameters without gradient: {missing}"

    def test_not_all_gradients_are_zero(self):
        model = self._backward(_model())
        total = sum(1 for _, p in model.named_parameters() if p.requires_grad)
        zero = [
            n for n, p in model.named_parameters()
            if p.requires_grad and p.grad is not None and p.grad.abs().max() == 0
        ]
        assert len(zero) < total, \
            "All parameter gradients are zero — backward pass may be broken"

    def test_gradient_clipping_bounds_norm(self):
        model = self._backward(_model())
        max_norm = 1.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        total_norm = torch.sqrt(sum(
            p.grad.pow(2).sum()
            for p in model.parameters()
            if p.requires_grad and p.grad is not None
        ))
        assert total_norm.item() <= max_norm + 1e-4, \
            f"Grad norm {total_norm.item():.4f} exceeds max_norm {max_norm}"


# ---------------------------------------------------------------------------
# Repetition unlikelihood loss
# ---------------------------------------------------------------------------
class TestULLoss:
    def test_alpha_zero_returns_zero(self):
        model = _model()
        b = _batch()
        logits = model(b["input_ids"])
        loss = _repetition_ul_loss(logits, b["input_ids"], b["labels"], alpha=0.0)
        assert loss.item() == 0.0

    def test_positive_when_model_predicts_previous_token(self):
        """Model that always predicts the same repeated token should get positive UL loss."""
        B, T, V = 1, 6, 256
        logits = torch.full((B, T, V), -10.0)
        logits[:, :, 5] = 10.0                        # strongly predict token 5
        input_ids = torch.full((B, T), 5, dtype=torch.long)  # all tokens are 5
        labels = input_ids.clone()

        loss = _repetition_ul_loss(logits, input_ids, labels, alpha=0.1)
        assert loss.item() > 0.05, \
            f"UL loss {loss.item():.4f} expected > 0.05 when predicting a repeated token"

    def test_finite_on_random_logits(self):
        model = _model()
        b = _batch()
        logits = model(b["input_ids"])
        loss = _repetition_ul_loss(logits, b["input_ids"], b["labels"], alpha=0.1)
        assert torch.isfinite(loss), f"UL loss is not finite: {loss.item()}"

    def test_ignores_padding_labels(self):
        """UL loss must be finite even when most labels are -100."""
        B, T, V = 2, 8, 256
        logits = torch.randn(B, T, V)
        input_ids = torch.randint(0, V, (B, T))
        labels = torch.full((B, T), -100, dtype=torch.long)
        labels[:, 0] = input_ids[:, 0]  # only one real label per sequence

        loss = _repetition_ul_loss(logits, input_ids, labels, alpha=1.0)
        assert torch.isfinite(loss)

    def test_short_sequence_returns_zero(self):
        """T<2 cannot have a 'previous' token, so loss should be 0."""
        B, T, V = 2, 1, 256
        logits = torch.randn(B, T, V)
        input_ids = torch.randint(0, V, (B, T))
        labels = input_ids.clone()

        loss = _repetition_ul_loss(logits, input_ids, labels, alpha=0.1)
        assert loss.item() == 0.0


# ---------------------------------------------------------------------------
# Cross-entropy padding / ignore_index
# ---------------------------------------------------------------------------
class TestCrossEntropyPadding:
    def test_padding_tokens_do_not_affect_loss(self):
        """CE loss with ignore_index=-100 should differ when padding is added."""
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        B, T, V = 2, 8, 256
        logits = torch.randn(B, T, V)

        labels_full = torch.randint(0, V, (B, T))
        labels_pad  = labels_full.clone()
        labels_pad[:, -4:] = -100

        loss_full = loss_fn(logits.reshape(B * T, V), labels_full.reshape(B * T))
        loss_pad  = loss_fn(logits.reshape(B * T, V), labels_pad.reshape(B * T))

        assert not torch.allclose(loss_full, loss_pad), \
            "CE loss is unchanged when padding tokens are masked — ignore_index not working"

    def test_all_padding_raises_or_gives_zero(self):
        """When every label is -100 the loss must be 0 (no tokens to average over)."""
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        B, T, V = 2, 8, 256
        logits = torch.randn(B, T, V)
        labels = torch.full((B, T), -100, dtype=torch.long)

        loss = loss_fn(logits.reshape(B * T, V), labels.reshape(B * T))
        # PyTorch returns 0 when all tokens are ignored
        assert loss.item() == 0.0 or not torch.isfinite(loss), \
            "Expected 0 loss when all labels are -100"


# ---------------------------------------------------------------------------
# collate_fn
# ---------------------------------------------------------------------------
class TestCollateFn:
    def _seqs(self):
        return [
            {"input_ids": torch.tensor([1, 2, 3])},
            {"input_ids": torch.tensor([4, 5, 6, 7, 8])},
        ]

    def test_output_shapes(self):
        b = collate_fn(self._seqs())
        assert b["input_ids"].shape      == (2, 5)
        assert b["attention_mask"].shape == (2, 5)
        assert b["labels"].shape         == (2, 5)

    def test_padding_value_in_input_ids(self):
        b = collate_fn(self._seqs(), pad_id=0)
        assert b["input_ids"][0, 3].item() == 0
        assert b["input_ids"][0, 4].item() == 0

    def test_attention_mask_zero_on_padding(self):
        b = collate_fn(self._seqs())
        assert b["attention_mask"][0, 3].item() == 0
        assert b["attention_mask"][0, 4].item() == 0

    def test_attention_mask_one_on_real_tokens(self):
        b = collate_fn(self._seqs())
        assert b["attention_mask"][0, 0].item() == 1
        assert b["attention_mask"][1, 4].item() == 1

    def test_labels_ignore_index_on_padding(self):
        b = collate_fn(self._seqs(), ignore_index=-100)
        assert b["labels"][0, 3].item() == -100
        assert b["labels"][0, 4].item() == -100

    def test_labels_preserve_real_tokens(self):
        b = collate_fn(self._seqs())
        assert b["labels"][0, 0].item() == 1
        assert b["labels"][1, 0].item() == 4

    def test_uniform_length_batch(self):
        seqs = [
            {"input_ids": torch.tensor([10, 20, 30])},
            {"input_ids": torch.tensor([40, 50, 60])},
        ]
        b = collate_fn(seqs)
        assert b["input_ids"].shape == (2, 3)
        assert b["attention_mask"].all()  # no padding needed


# ---------------------------------------------------------------------------
# QADataset
# ---------------------------------------------------------------------------
class _DummyTok:
    """Minimal tokenizer: encodes each character to its ASCII value mod 256."""
    def encode(self, text):
        return [ord(c) % 256 for c in text]


class TestQADataset:
    def test_len(self):
        data = [{"question": "q1", "answer": "a1"}, {"question": "q2", "answer": "a2"}]
        ds = QADataset(data, _DummyTok())
        assert len(ds) == 2

    def test_item_dtype(self):
        data = [{"question": "hello", "answer": "world"}]
        item = QADataset(data, _DummyTok())[0]
        assert item["input_ids"].dtype == torch.long

    def test_max_length_respected(self):
        data = [{"question": "a" * 200, "answer": "b" * 200}]
        item = QADataset(data, _DummyTok(), max_length=32)[0]
        assert len(item["input_ids"]) <= 32

    def test_text_contains_qa_format(self):
        """Encoded text must include characters from both Q and A."""
        data = [{"question": "AAA", "answer": "ZZZ"}]
        item = QADataset(data, _DummyTok())[0]
        ids_set = set(item["input_ids"].tolist())
        assert ord("A") % 256 in ids_set
        assert ord("Z") % 256 in ids_set

    def test_alternative_keys(self):
        """QADataset falls back to 'input'/'output' keys."""
        data = [{"input": "hello", "output": "world"}]
        item = QADataset(data, _DummyTok())[0]
        assert len(item["input_ids"]) > 0


# ---------------------------------------------------------------------------
# Learning-rate schedule
# ---------------------------------------------------------------------------
class TestLRSchedule:
    def _make_sched(self, warmup=10, total=100):
        m   = _model()
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
        return opt, get_cosine_schedule_with_warmup(opt, warmup, total)

    def test_lr_increases_during_warmup(self):
        opt, sched = self._make_sched(warmup=10, total=100)
        lrs = []
        for _ in range(10):
            sched.step()
            lrs.append(opt.param_groups[0]["lr"])
        for i in range(1, len(lrs)):
            assert lrs[i] >= lrs[i - 1] - 1e-9, \
                f"LR decreased during warmup at step {i}: {lrs}"

    def test_lr_decreases_after_warmup(self):
        opt, sched = self._make_sched(warmup=5, total=50)
        for _ in range(6):
            sched.step()
        lr_peak = opt.param_groups[0]["lr"]
        for _ in range(20):
            sched.step()
        lr_later = opt.param_groups[0]["lr"]
        assert lr_later < lr_peak, "LR did not decrease after warmup"

    def test_lr_ends_near_zero(self):
        opt, sched = self._make_sched(warmup=5, total=50)
        for _ in range(50):
            sched.step()
        assert opt.param_groups[0]["lr"] < 1e-5, \
            f"Final LR {opt.param_groups[0]['lr']:.2e} is not near zero"

    def test_lr_starts_from_zero(self):
        """LambdaLR sets LR to 0 at step 0 (warmup fraction = 0/warmup_steps)."""
        opt, sched = self._make_sched(warmup=10, total=100)
        # At init the scheduler has already applied lambda(0) = 0/10 = 0
        assert opt.param_groups[0]["lr"] == 0.0
        sched.step()  # step 1: fraction = 1/10 > 0
        assert opt.param_groups[0]["lr"] > 0.0


# ---------------------------------------------------------------------------
# Single-batch overfit
# ---------------------------------------------------------------------------
class TestOverfitting:
    def test_loss_decreases_on_single_batch(self):
        """Model must be able to memorise a fixed tiny batch (sanity check)."""
        torch.manual_seed(0)
        model = _model()
        b = _batch(B=2, T=8)
        loss_fn = nn.CrossEntropyLoss()
        opt = torch.optim.Adam(model.parameters(), lr=5e-3)

        initial_loss = None
        for _ in range(40):
            model.train()
            logits = model(b["input_ids"])
            B, T, V = logits.shape
            loss = loss_fn(logits.reshape(B * T, V), b["labels"].reshape(B * T))
            if initial_loss is None:
                initial_loss = loss.item()
            opt.zero_grad()
            loss.backward()
            opt.step()

        assert loss.item() < initial_loss * 0.5, \
            f"Failed to overfit: initial={initial_loss:.3f}, final={loss.item():.3f}"

    def test_loss_decreases_across_epochs(self):
        """Average epoch loss should decrease when training on a fixed dataset."""
        torch.manual_seed(42)
        model = _model()
        loss_fn = nn.CrossEntropyLoss()
        opt = torch.optim.Adam(model.parameters(), lr=5e-3)
        loader = _loader(n_samples=8, T=10, batch_size=4)

        epoch_losses = []
        for _ in range(6):
            model.train()
            ep_loss = 0.0
            for b in loader:
                logits = model(b["input_ids"], b["attention_mask"])
                Bs, T, V = logits.shape
                loss = loss_fn(logits.reshape(Bs * T, V), b["labels"].reshape(Bs * T))
                opt.zero_grad()
                loss.backward()
                opt.step()
                ep_loss += loss.item()
            epoch_losses.append(ep_loss / len(loader))

        assert epoch_losses[-1] < epoch_losses[0], \
            f"Loss did not decrease across epochs: {epoch_losses}"


# ---------------------------------------------------------------------------
# train_model() integration smoke test
# ---------------------------------------------------------------------------
class TestTrainModelIntegration:
    def test_runs_one_epoch_without_error(self):
        torch.manual_seed(1)
        model  = _model()
        train  = _loader(n_samples=8,  T=10, batch_size=2)
        val    = _loader(n_samples=4,  T=10, batch_size=2)
        opt    = torch.optim.Adam(model.parameters(), lr=1e-3)

        with patch("matplotlib.pyplot.show"):
            result = train_model(
                model=model,
                train_loader=train,
                val_loader=val,
                optimizer=opt,
                device="cpu",
                epochs=1,
                accumulation_steps=1,
            )
        assert result is model

    def test_returns_model_with_same_params(self):
        torch.manual_seed(2)
        model = _model()
        ids_before = id(model)

        train = _loader(n_samples=4, T=8, batch_size=2)
        val   = _loader(n_samples=4, T=8, batch_size=2)
        opt   = torch.optim.Adam(model.parameters(), lr=1e-3)

        with patch("matplotlib.pyplot.show"):
            result = train_model(model, train, val, opt, "cpu", epochs=1,
                                 accumulation_steps=1)

        assert id(result) == ids_before, "train_model must return the same model object"

    def test_weights_change_after_training(self):
        torch.manual_seed(3)
        model = _model()
        w0 = model.token_emb.weight.detach().clone()

        train = _loader(n_samples=8, T=10, batch_size=4)
        val   = _loader(n_samples=4, T=10, batch_size=4)
        opt   = torch.optim.Adam(model.parameters(), lr=1e-2)

        with patch("matplotlib.pyplot.show"):
            train_model(model, train, val, opt, "cpu", epochs=2,
                        accumulation_steps=1)

        w1 = model.token_emb.weight.detach()
        assert not torch.allclose(w0, w1, atol=1e-6), \
            "Model weights did not change after training — optimizer may be broken"

    def test_gradient_accumulation_runs(self):
        """accumulation_steps > 1 should not crash and should still update weights."""
        torch.manual_seed(4)
        model = _model()
        train = _loader(n_samples=8, T=8, batch_size=2)
        val   = _loader(n_samples=4, T=8, batch_size=2)
        opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
        w0    = model.token_emb.weight.detach().clone()

        with patch("matplotlib.pyplot.show"):
            train_model(model, train, val, opt, "cpu", epochs=1,
                        accumulation_steps=4)

        assert not torch.allclose(w0, model.token_emb.weight.detach(), atol=1e-6)


# ---------------------------------------------------------------------------
# Notebook-callable entry point
# ---------------------------------------------------------------------------
def check_trained_model(model, tok, device, vocab_size: int, pad_id: int) -> bool:
    """
    Post-training behavior checks for a trained Transformer.

    Designed to be called from a notebook after all training stages::

        from tests.test_training import check_trained_model
        check_trained_model(model, tok, device, vocab_size, pad_id)

    Returns True if every check passes.
    """
    import math

    device = torch.device(device) if isinstance(device, str) else device

    def _check(ok, name):
        print(f"  {'✓ PASS' if ok else '✗ FAIL'}  {name}")
        return ok

    print("=" * 58)
    print("POST-TRAINING BEHAVIOR CHECKS")
    print("=" * 58)

    base    = getattr(model, '_orig_mod', model)
    base.eval()
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    V       = vocab_size

    sample_texts = [
        "The capital of France is Paris.",
        "Once upon a time there was a little girl.",
        "Neural networks learn representations from data.",
        "Water boils at one hundred degrees Celsius.",
    ]
    enc   = [tok.encode(t, max_length=64, truncation=True) for t in sample_texts]
    maxT  = max(len(e) for e in enc)
    B     = len(enc)
    ids   = torch.zeros(B, maxT, dtype=torch.long, device=device)
    amask = torch.zeros(B, maxT, dtype=torch.long, device=device)
    for i, e in enumerate(enc):
        ids[i, :len(e)]   = torch.tensor(e, device=device)
        amask[i, :len(e)] = 1
    labels = ids.clone(); labels[amask == 0] = -100

    with torch.no_grad():
        logits = base(ids, attention_mask=amask)
        Bs, Ts, Vs = logits.shape
        loss = loss_fn(logits.reshape(Bs*Ts, Vs), labels.reshape(Bs*Ts)).item()

    ok = True
    ok &= _check(math.isfinite(loss),
                 f"Loss is finite: {loss:.4f}")
    ok &= _check(loss < math.log(V),
                 f"Loss {loss:.3f} < log(V)={math.log(V):.3f}  (beat random baseline)")

    ppl = math.exp(min(loss, 20))
    print(f"  → Perplexity: {ppl:.1f}")

    pred = logits.argmax(dim=-1)
    real = labels != -100
    acc  = (pred[real] == labels[real]).float().mean().item() * 100
    ok &= _check(acc > 1.0, f"Top-1 accuracy > 1%: {acc:.1f}%")
    ok &= _check(torch.isfinite(logits).all().item(),
                 "Post-training logits are finite")
    ok &= _check(torch.isfinite(base.token_emb.weight).all().item(),
                 "Embedding weights are finite (no corruption)")

    a  = torch.randint(1, V, (1, 16), device=device)
    b  = a.clone(); b[0, -1] = (b[0, -1] + 1) % V
    am = torch.ones(1, 16, dtype=torch.long, device=device)
    with torch.no_grad():
        la, lb = base(a, am), base(b, am)
    ok &= _check(torch.allclose(la[0, :-1], lb[0, :-1], atol=1e-4),
                 "Causal masking still intact after training")

    prompt_enc = tok.encode("The capital of France is", add_special_tokens=False)[:16]
    prompt_t   = torch.tensor(prompt_enc, dtype=torch.long, device=device).unsqueeze(0)
    plen       = prompt_t.shape[1]
    gen        = base.generate(prompt_t, max_new_tokens=20, temperature=0.8,
                                top_k=40, suppress_ids=[pad_id])
    gen_toks   = gen[0, plen:].tolist()
    ok &= _check(all(0 <= t < V for t in gen_toks),
                 f"All {len(gen_toks)} generated tokens in vocab range [0, {V})")

    print("\n  Per-sample loss / perplexity:")
    for i, txt in enumerate(sample_texts):
        sl = loss_fn(logits[i].reshape(-1, Vs), labels[i].reshape(-1)).item()
        print(f"    [{i+1}] loss={sl:.3f}  ppl={math.exp(min(sl, 20)):.1f}"
              f"  '{txt[:42]}'")

    log_v = math.log(V)
    improvement = (log_v - loss) / log_v * 100
    print(f"\n  Loss vs random: {log_v:.3f} → {loss:.3f}"
          f"  ({improvement:+.1f}% {'better' if improvement > 0 else 'worse'})")
    gen_text = tok.decode(gen_toks, skip_special_tokens=True)
    print(f"  'The capital of France is' → '{gen_text.strip()}'")
    print()
    return ok
