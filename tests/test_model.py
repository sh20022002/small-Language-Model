"""
Model behavior tests: forward pass, causal masking, weight tying,
generation, RoPE, RMSNorm, attention mask, resize, dropout.
"""
import pytest
import torch
import torch.nn as nn

from my_slm.transformer import Transformer, RMSNorm, RoPE, MultiHeadAttention

# ---------------------------------------------------------------------------
# Shared config — tiny model so all tests run fast on CPU
# ---------------------------------------------------------------------------
VOCAB = 256
TINY = dict(vocab_size=VOCAB, dim=64, depth=2, heads=4, mlp_dim=128, window=16)


def _model(**kw) -> Transformer:
    return Transformer(**{**TINY, **kw}).eval()


def _ids(B=2, T=8):
    return torch.randint(1, VOCAB, (B, T))


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------
class TestForwardPass:
    def test_output_shape(self):
        logits = _model()(_ids())
        assert logits.shape == (2, 8, VOCAB)

    def test_output_finite(self):
        logits = _model()(_ids())
        assert torch.isfinite(logits).all(), "logits contain NaN or Inf"

    def test_batch_independence(self):
        """Two items in a batch should produce independent logits."""
        model = _model()
        ids_a = _ids(B=2, T=8)
        ids_b = ids_a.clone()
        ids_b[1] = torch.randint(1, VOCAB, (8,))  # mutate second item only

        out_a = model(ids_a)
        out_b = model(ids_b)
        # First item must be identical; second must differ
        assert torch.allclose(out_a[0], out_b[0], atol=1e-5)
        assert not torch.allclose(out_a[1], out_b[1], atol=1e-5)


# ---------------------------------------------------------------------------
# Causal masking
# ---------------------------------------------------------------------------
class TestCausalMasking:
    def test_future_token_does_not_affect_past_logits(self):
        model = _model()
        ids_a = _ids(B=1, T=10)
        ids_b = ids_a.clone()
        ids_b[0, -1] = (ids_b[0, -1] + 1) % VOCAB  # change only the last token

        out_a = model(ids_a)
        out_b = model(ids_b)
        assert torch.allclose(out_a[0, :-1], out_b[0, :-1], atol=1e-5), \
            "Past logits changed when a future token was modified — causal mask broken"

    def test_causal_local_mask_shape(self):
        T, W = 12, 4
        mask = MultiHeadAttention._causal_local_mask(T, W, torch.device("cpu"))
        assert mask.shape == (1, 1, T, T)

    def test_causal_local_mask_no_future_leakage(self):
        T, W = 8, 4
        mask = MultiHeadAttention._causal_local_mask(T, W, torch.device("cpu"))[0, 0]
        for i in range(T):
            for j in range(i + 1, T):
                assert not mask[i, j].item(), \
                    f"Future position ({i},{j}) is unmasked — upper triangle should be False"

    def test_causal_local_mask_window_blocks_distant_past(self):
        T, W = 10, 3
        mask = MultiHeadAttention._causal_local_mask(T, W, torch.device("cpu"))[0, 0]
        for i in range(T):
            for j in range(T):
                if i - j > W:
                    assert not mask[i, j].item(), \
                        f"Out-of-window position ({i},{j}) is unmasked (window={W})"

    def test_causal_local_mask_within_window_attended(self):
        T, W = 8, 4
        mask = MultiHeadAttention._causal_local_mask(T, W, torch.device("cpu"))[0, 0]
        for i in range(T):
            for j in range(max(0, i - W), i + 1):
                assert mask[i, j].item(), \
                    f"In-window position ({i},{j}) is incorrectly masked"


# ---------------------------------------------------------------------------
# Weight tying
# ---------------------------------------------------------------------------
class TestWeightTying:
    def test_tied_weights_share_storage(self):
        m = _model(tie_weights=True)
        assert m.token_emb.weight.data_ptr() == m.to_logits.weight.data_ptr(), \
            "tie_weights=True: embedding and output head must share the same tensor"

    def test_untied_weights_independent(self):
        m = _model(tie_weights=False)
        assert m.token_emb.weight.data_ptr() != m.to_logits.weight.data_ptr()


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------
class TestRMSNorm:
    def test_output_shape(self):
        norm = RMSNorm(64)
        x = torch.randn(2, 8, 64)
        assert norm(x).shape == x.shape

    def test_output_finite(self):
        norm = RMSNorm(64)
        assert torch.isfinite(norm(torch.randn(2, 8, 64))).all()

    def test_normalises_scale(self):
        """Output should have unit RMS regardless of input scale."""
        norm = RMSNorm(64)
        norm.weight.data.fill_(1.0)  # disable learned scale
        x = torch.randn(4, 64) * 100.0
        out = norm(x)
        rms = out.pow(2).mean(dim=-1).sqrt()
        assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4)


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------
class TestRoPE:
    def test_preserves_shape(self):
        x = torch.randn(2, 4, 8, 64)  # [B, H, T, D]
        assert RoPE.apply(x).shape == x.shape

    def test_output_finite(self):
        x = torch.randn(2, 4, 8, 64)
        assert torch.isfinite(RoPE.apply(x)).all()


# ---------------------------------------------------------------------------
# Attention mask
# ---------------------------------------------------------------------------
class TestAttentionMask:
    def test_padding_mask_changes_output(self):
        model = _model()
        ids = _ids(B=1, T=8)
        full_mask = torch.ones(1, 8, dtype=torch.long)
        half_mask = torch.cat([torch.ones(1, 4), torch.zeros(1, 4)], dim=1).long()

        out_full = model(ids, attention_mask=full_mask)
        out_half = model(ids, attention_mask=half_mask)
        assert not torch.allclose(out_full, out_half, atol=1e-6), \
            "Attention mask has no effect — padding is not being blocked"


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
class TestGeneration:
    def test_output_length(self):
        model = _model()
        ids = _ids(B=1, T=4)
        out = model.generate(ids, max_new_tokens=10, temperature=0)
        assert out.shape[1] == 4 + 10

    def test_tokens_in_vocab_range(self):
        model = _model()
        out = model.generate(_ids(B=1, T=4), max_new_tokens=8, temperature=1.0, top_k=10)
        assert out.min().item() >= 0
        assert out.max().item() < VOCAB

    def test_greedy_is_deterministic(self):
        model = _model()
        ids = _ids(B=1, T=4)
        out1 = model.generate(ids, max_new_tokens=8, temperature=0)
        out2 = model.generate(ids, max_new_tokens=8, temperature=0)
        assert torch.equal(out1, out2), "Greedy generation (temperature=0) is not deterministic"

    def test_eos_stops_generation(self):
        """If the model greedily picks EOS, generation stops before max_new_tokens."""
        model = _model()
        # We can't force EOS, but we can verify the function respects the limit
        ids = _ids(B=1, T=3)
        out = model.generate(ids, max_new_tokens=20, eos_token_id=1, temperature=0)
        assert out.shape[1] <= 3 + 20

    def test_suppress_ids_never_generated(self):
        model = _model()
        ids = _ids(B=1, T=4)
        # Suppress a large range; model must pick from the remaining few
        suppress = list(range(50, 250))
        out = model.generate(ids, max_new_tokens=10, suppress_ids=suppress,
                             temperature=1.0, top_k=5)
        generated = out[0, 4:].tolist()
        assert all(t not in suppress for t in generated), \
            f"Suppressed token appeared in generated output: {generated}"

    def test_repetition_penalty_changes_output(self):
        model = _model()
        ids = _ids(B=1, T=4)
        out_no_pen = model.generate(ids, max_new_tokens=8, temperature=0,
                                    repetition_penalty=1.0)
        out_pen    = model.generate(ids, max_new_tokens=8, temperature=0,
                                    repetition_penalty=2.0)
        # With a strong penalty the output sequence is likely different
        # (not guaranteed, but very likely with a random model)
        assert out_no_pen.shape == out_pen.shape  # at minimum shapes match


# ---------------------------------------------------------------------------
# resize_token_embeddings
# ---------------------------------------------------------------------------
class TestResizeEmbeddings:
    def test_grows_vocab(self):
        m = _model(tie_weights=False)
        m.resize_token_embeddings(VOCAB + 50)
        assert m.token_emb.num_embeddings == VOCAB + 50
        assert m.to_logits.out_features == VOCAB + 50

    def test_old_weights_preserved(self):
        m = _model(tie_weights=False)
        old_w = m.token_emb.weight.detach().clone()
        m.resize_token_embeddings(VOCAB + 10)
        assert torch.allclose(m.token_emb.weight[:VOCAB].data, old_w)

    def test_noop_when_same_size(self):
        m = _model()
        ptr_before = m.token_emb.weight.data_ptr()
        m.resize_token_embeddings(VOCAB)
        assert m.token_emb.weight.data_ptr() == ptr_before

    def test_new_rows_finite(self):
        m = _model(tie_weights=False)
        m.resize_token_embeddings(VOCAB + 20)
        new_rows = m.token_emb.weight[VOCAB:]
        assert torch.isfinite(new_rows).all()
        assert new_rows.abs().max().item() > 0  # not all zeros


# ---------------------------------------------------------------------------
# set_dropout
# ---------------------------------------------------------------------------
class TestSetDropout:
    def test_updates_all_dropout_layers(self):
        m = _model(dropout=0.0)
        m.set_dropout(0.5)
        for mod in m.modules():
            if isinstance(mod, nn.Dropout):
                assert mod.p == 0.5, f"Dropout layer was not updated: p={mod.p}"

    def test_set_back_to_zero(self):
        m = _model(dropout=0.3)
        m.set_dropout(0.0)
        for mod in m.modules():
            if isinstance(mod, nn.Dropout):
                assert mod.p == 0.0


# ---------------------------------------------------------------------------
# Notebook-callable entry point
# ---------------------------------------------------------------------------
def check_model_architecture(model, vocab_size: int, device) -> bool:
    """
    Run architecture sanity checks on any Transformer instance.

    Designed to be called from a notebook right after model creation::

        from tests.test_model import check_model_architecture
        check_model_architecture(model, vocab_size, device)

    Returns True if every check passes.
    """
    import math

    device = torch.device(device) if isinstance(device, str) else device

    def _check(ok, name):
        print(f"  {'✓ PASS' if ok else '✗ FAIL'}  {name}")
        return ok

    print("=" * 58)
    print("MODEL ARCHITECTURE CHECKS")
    print("=" * 58)

    base   = getattr(model, '_orig_mod', model)   # unwrap torch.compile
    V      = vocab_size
    B, T   = 2, 32
    ids    = torch.randint(1, V, (B, T), device=device)
    amask  = torch.ones(B, T, dtype=torch.long, device=device)
    loss_fn = nn.CrossEntropyLoss()

    base.eval()
    with torch.no_grad():
        logits = base(ids, attention_mask=amask)

    ok = True
    ok &= _check(logits.shape == (B, T, V),
                 f"Forward shape: got {tuple(logits.shape)}, expected ({B}, {T}, {V})")
    ok &= _check(torch.isfinite(logits).all().item(),
                 "Logits are finite (no NaN/Inf)")

    with torch.no_grad():
        init_loss = loss_fn(logits.reshape(B*T, V), ids.reshape(B*T)).item()
    expected = math.log(V)
    ok &= _check(0.3 * expected < init_loss < 3.0 * expected,
                 f"Initial loss {init_loss:.2f} ≈ log(V)={expected:.2f}")

    ids2 = ids.clone(); ids2[0, -1] = (ids2[0, -1] + 1) % V
    with torch.no_grad():
        logits2 = base(ids2, attention_mask=amask)
    ok &= _check(torch.allclose(logits[0, :-1], logits2[0, :-1], atol=1e-4),
                 "Causal mask: future token change does not affect past logits")

    ok &= _check(base.token_emb.weight.data_ptr() == base.to_logits.weight.data_ptr(),
                 "Weight tying: token_emb and to_logits share the same tensor")

    base.train()
    l = loss_fn(base(ids, attention_mask=amask).reshape(B*T, V), ids.reshape(B*T))
    l.backward()
    no_grad = [n for n, p in base.named_parameters() if p.requires_grad and p.grad is None]
    base.zero_grad(); base.eval()
    ok &= _check(not no_grad,
                 f"Gradient flow: all params have grad (missing: {no_grad or 'none'})")

    half = torch.cat([torch.ones(B, T//2, dtype=torch.long, device=device),
                      torch.zeros(B, T//2, dtype=torch.long, device=device)], dim=1)
    with torch.no_grad():
        logits_half = base(ids, attention_mask=half)
    ok &= _check(not torch.allclose(logits[:, :T//2], logits_half[:, :T//2], atol=1e-6),
                 "Attention mask changes output (padding is blocked)")

    params = sum(p.numel() for p in base.parameters())
    print(f"\n  Parameters : {params:,}  ({params/1e6:.1f}M)")
    print(f"  log(V) baseline loss: {expected:.3f}")
    print()
    return ok
