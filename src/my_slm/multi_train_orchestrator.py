import itertools
from dataclasses import dataclass
from typing import Iterable, List, Optional
from pathlib import Path
import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from my_slm.train import train_model, train_model_accelerate
from my_slm.utils import encode, get_eos_token_id, get_pad_token_id

# -----------------------------
# Dataset helpers
# -----------------------------

def _alpaca_getter(ex):
    ins = ex.get("instruction") or ""
    inp = ex.get("input") or ""
    out = ex.get("output") or ""
    if inp:
        return f"### Instruction:\n{ins}\n\n### Input:\n{inp}\n\n### Response:\n{out}\n"
    return f"### Instruction:\n{ins}\n\n### Response:\n{out}\n"


def _dolly_getter(ex):
    ins = ex.get("instruction") or ""
    ctx = ex.get("context") or ""
    out = ex.get("response") or ""
    if ctx:
        return f"### Instruction:\n{ins}\n\n### Context:\n{ctx}\n\n### Response:\n{out}\n"
    return f"### Instruction:\n{ins}\n\n### Response:\n{out}\n"


def _gsm8k_getter(ex):
    """GSM8K: question + step-by-step solution (chain-of-thought)."""
    q = ex.get("question") or ""
    a = ex.get("answer") or ""
    return f"### Question:\n{q}\n\n### Solution:\n{a}\n"


def _openorca_getter(ex):
    sys_p = ex.get("system_prompt") or ""
    q     = ex.get("question") or ""
    r     = ex.get("response") or ""
    if sys_p:
        return f"### System:\n{sys_p}\n\n### Question:\n{q}\n\n### Answer:\n{r}\n"
    return f"### Question:\n{q}\n\n### Answer:\n{r}\n"


def _hh_rlhf_getter(ex):
    """Return the 'chosen' (good) response trajectory from Anthropic HH-RLHF."""
    return ex.get("chosen") or ""


def get_hf_stream_and_text_getter(name: str, split: str = "train", skip: int = 0):
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError("pip install datasets") from e
    name = name.lower()
    if name == "wikitext":
        ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split=split, streaming=True)
        getter = lambda ex: ex.get("text") or ""
    elif name == "tinystories":
        ds = load_dataset("roneneldan/TinyStories", split=split, streaming=True)
        getter = lambda ex: ex.get("text") or ""
    elif name == "openwebtext":
        ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
        getter = lambda ex: ex.get("text") or ""
    elif name == "c4":
        ds = load_dataset("allenai/c4", "en", split=split, streaming=True)
        getter = lambda ex: ex.get("text") or ""
    elif name == "alpaca":
        ds = load_dataset("yahma/alpaca-cleaned", split="train", streaming=True)
        getter = _alpaca_getter
    elif name == "dolly":
        ds = load_dataset("databricks/databricks-dolly-15k", split="train", streaming=True)
        getter = _dolly_getter
    elif name == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split=split, streaming=True)
        getter = _gsm8k_getter
    elif name == "openorca":
        ds = load_dataset("open-orca/OpenOrca", split="train", streaming=True)
        getter = _openorca_getter
    elif name in ("stories", "pg19", "bookcorpus"):
        ds = load_dataset("ajibawa-2023/Children-Stories-Collection", split="train", streaming=True)
        getter = lambda ex: ex.get("text") or ""
    elif name == "hh_rlhf":
        ds = load_dataset("Anthropic/hh-rlhf", split="train", streaming=True)
        getter = _hh_rlhf_getter
    else:
        raise ValueError(
            f"Unknown dataset {name}. "
            "Choose: wikitext, tinystories, openwebtext, c4, stories, "
            "alpaca, dolly, gsm8k, openorca, hh_rlhf."
        )
    if skip > 0:
        ds = ds.skip(skip)
    return ds, getter


# Datasets with an official "validation" split on HF Hub
_HAS_VAL_SPLIT = {"wikitext", "tinystories", "c4"}
# Datasets whose held-out split is called "test" instead of "validation"
_HAS_TEST_SPLIT = {"gsm8k", "hh_rlhf"}


def get_hf_val_stream_and_getter(name: str, skip_after: int = 0):
    """Return a validation stream that doesn't overlap with the training stream."""
    name_lower = name.lower()
    if name_lower in _HAS_VAL_SPLIT:
        return get_hf_stream_and_text_getter(name, split="validation")
    if name_lower in _HAS_TEST_SPLIT:
        return get_hf_stream_and_text_getter(name, split="test")
    # dolly: only ~15 k examples — use the last 2 k as validation
    if name_lower == "dolly":
        return get_hf_stream_and_text_getter(name, split="train", skip=13_000)
    # Large datasets with only a train split: skip past the training portion
    return get_hf_stream_and_text_getter(name, split="train", skip=skip_after)

# Use the unified encode from utils; provides backward compatibility
_encode = encode


class PackedTokenDataset(IterableDataset):
    """
    Lazy packed training: tokenizes on-the-fly from a streaming HF dataset.
    Concatenates token IDs (separated by EOS) into a rolling buffer, yielding
    non-overlapping blocks of exactly `block_size` tokens as they fill up.

    Benefits:
    - Zero padding waste — every position contributes to the loss.
    - No upfront tokenization — training starts immediately.
    - Multi-epoch safe — re-iterating restarts the HF stream from the beginning.
    """
    def __init__(self, hf_stream, get_text, tokenizer, block_size: int,
                 max_texts: Optional[int] = None):
        self._stream     = hf_stream
        self._get_text   = get_text
        self._tokenizer  = tokenizer
        self._block_size = block_size
        self._max_texts  = max_texts

    def __iter__(self):
        buffer: List[int] = []
        n = 0
        for ex in self._stream:
            text = self._get_text(ex)
            if not text or not text.strip():
                continue
            ids = _encode(self._tokenizer, text)
            if not ids:
                continue
            buffer.extend(ids)
            n += 1
            if self._max_texts and n >= self._max_texts:
                break
            while len(buffer) >= self._block_size:
                yield torch.tensor(buffer[:self._block_size], dtype=torch.long)
                buffer = buffer[self._block_size:]
        # flush any remaining complete block
        while len(buffer) >= self._block_size:
            yield torch.tensor(buffer[:self._block_size], dtype=torch.long)
            buffer = buffer[self._block_size:]


class CombinedIterableDataset(IterableDataset):
    """Round-robin interleave multiple PackedTokenDatasets."""
    def __init__(self, *datasets):
        self.datasets = list(datasets)

    def __iter__(self):
        iters = [iter(d) for d in self.datasets]
        active = list(range(len(iters)))
        while active:
            next_active = []
            for i in active:
                try:
                    yield next(iters[i])
                    next_active.append(i)
                except StopIteration:
                    pass
            active = next_active


def packed_collate(batch: List[torch.Tensor]):
    """All blocks have identical length — stack without padding."""
    ids  = torch.stack(batch)                            # [B, block_size]
    attn = torch.ones_like(ids, dtype=torch.long)        # no padding → all ones
    return {"input_ids": ids, "attention_mask": attn, "labels": ids.clone()}


class TextTokenDataset(Dataset):
    """Materializes a small list of token ID tensors for simple training on Colab."""
    def __init__(self, hf_stream, get_text, tokenizer, max_len: int, max_items: Optional[int] = None):
        self.samples: List[torch.Tensor] = []
        n = 0
        for ex in hf_stream:
            text = get_text(ex)
            if not text or not text.strip():
                continue          # skip blank / whitespace-only entries (common in wikitext)
            ids = _encode(tokenizer, text, max_len=max_len)
            if len(ids) > 1:     # need ≥2 tokens for next-token prediction (logits[:, :-1])
                self.samples.append(torch.tensor(ids, dtype=torch.long))
                n += 1
                if max_items and n >= max_items:
                    break
        if not self.samples:
            raise RuntimeError("No samples produced; check dataset and text getter.")

    def __len__(self): return len(self.samples)
    def __getitem__(self, i): return self.samples[i]

def make_collate(pad_id: int, ignore_index: int):
    def collate(batch: List[torch.Tensor]):
        ids = pad_sequence(batch, batch_first=True, padding_value=pad_id)  # [B, T]
        attn = (ids != pad_id).long()                                      # [B, T]
        labels = ids.clone()
        labels[ids == pad_id] = ignore_index                               # ignore pads
        return {"input_ids": ids, "attention_mask": attn, "labels": labels}
    return collate

class SliceLoader:
    """Wrap a DataLoader to expose only the first `max_batches` batches (for step-based training)."""
    def __init__(self, loader: DataLoader, max_batches: int):
        self.loader = loader
        self.max_batches = max_batches
    def __iter__(self):
        return itertools.islice(iter(self.loader), self.max_batches)
    def __len__(self):
        try:
            return min(self.max_batches, len(self.loader))
        except TypeError:
            return self.max_batches


class CombinedDataset(Dataset):
    """Concatenate multiple TextTokenDataset objects into a single shuffleable pool."""
    def __init__(self, *datasets):
        self.datasets = list(datasets)
        self._offsets: List[int] = []
        total = 0
        for d in self.datasets:
            self._offsets.append(total)
            total += len(d)
        self._total = total

    def __len__(self):
        return self._total

    def __getitem__(self, i):
        # Walk backwards so larger offsets are checked first (O(n) but n is tiny)
        for j in range(len(self.datasets) - 1, -1, -1):
            if i >= self._offsets[j]:
                return self.datasets[j][i - self._offsets[j]]
        raise IndexError(i)


# -----------------------------
# Orchestrator (TRAIN-ONLY)
# -----------------------------

@dataclass
class StageConfig:
    name: "str | list[str]"   # single dataset name OR list for a combined/interleaved stage
    epochs: int = 0
    steps: int = 0            # if >0, train exactly this many batches (one pseudo-epoch)
    # ── Accuracy gate (epoch-based training only) ──────────────────────────────
    # Keep running extra epochs until val accuracy ≥ min_val_accuracy  OR
    # loss plateaus for plateau_patience consecutive epochs.
    # max_epochs is the hard cap; set to 0 to disable the gate entirely.
    min_val_accuracy: float = 0.0
    max_epochs: int = 0
    plateau_patience: int = 3
    plateau_min_delta: float = 5e-4

# Backward compatibility aliases
_get_pad_id = get_pad_token_id
_get_eos_id = get_eos_token_id


def train_across_datasets(
    *,
    model,
    optimizer,
    tokenizer,
    accelerator=None,               # pass an Accelerator for DDP/multi-GPU (like the notebook)
    device: Optional[str] = None,   # used only when accelerator is None
    stages: Iterable[StageConfig] = (
        StageConfig("tinystories", steps=1000),
        StageConfig("wikitext",   steps=2000),
        StageConfig("openwebtext", steps=2000),
        StageConfig("alpaca",     steps=1000),
    ),
    max_len: int = 256,
    train_items: int = 50_000,
    val_items: int = 2_000,
    batch_size: int = 32,
    scheduler=None,
    max_grad_norm: float = 1.0,
    ul_alpha: float = 0.1,
    accumulation_steps: int = 4,    # used only in single-GPU path
    save_dir: str = "./out",
    save_figures: bool = True,      # save per-stage loss curves as PNG in save_dir
    use_packed: bool = True,        # packed training: no padding waste (20-40% faster)
):
    """
    Multi-stage curriculum training over HuggingFace datasets.

    Two modes (automatically selected):
      accelerator is not None → DDP path  (mirrors notebook _train_fn, uses train_model_accelerate)
      accelerator is None     → single-GPU path (uses train_model with AMP + grad accumulation)

    Per-stage choice: stage.steps > 0 → step-based; stage.epochs > 0 → epoch-based.
    """
    use_ddp = accelerator is not None
    is_main = (not use_ddp) or accelerator.is_main_process

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    if not use_ddp and device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Suppress HF tokenizer warnings about "sequence longer than model_max_length".
    # PackedTokenDataset intentionally tokenizes full documents before splitting
    # into block_size chunks — the model never sees sequences longer than max_len.
    if not hasattr(tokenizer, "token2id") and hasattr(tokenizer, "model_max_length"):
        tokenizer.model_max_length = int(1e9)

    pad_id = _get_pad_id(tokenizer)
    collate = make_collate(pad_id=pad_id, ignore_index=-100)

    for stage in stages:
        # ── Resolve name(s) ───────────────────────────────────────────────────
        names = stage.name if isinstance(stage.name, list) else [stage.name]
        names = [n.lower() for n in names]
        display_name = "+".join(names)

        if is_main:
            print(f"\n=== Stage: {display_name} | epochs={stage.epochs} | steps={stage.steps}"
                  f" | acc_target={stage.min_val_accuracy:.1f}% | max_ep={stage.max_epochs} ===")

        # ── Dataset loading (one sub-dataset per name, then combined) ─────────
        items_per = max(1, train_items // len(names))
        val_per   = max(1, val_items   // len(names))
        try:
            train_parts, val_parts = [], []
            for n in names:
                tr_stream, getter = get_hf_stream_and_text_getter(n)
                vl_stream, _      = get_hf_val_stream_and_getter(n, skip_after=train_items)
                if use_packed:
                    train_parts.append(
                        PackedTokenDataset(tr_stream, getter, tokenizer,
                                           block_size=max_len, max_texts=items_per))
                    val_parts.append(
                        PackedTokenDataset(vl_stream, getter, tokenizer,
                                           block_size=max_len, max_texts=val_per))
                else:
                    train_parts.append(
                        TextTokenDataset(tr_stream, getter, tokenizer,
                                         max_len=max_len, max_items=items_per))
                    val_parts.append(
                        TextTokenDataset(vl_stream, getter, tokenizer,
                                         max_len=max_len, max_items=val_per))
            if use_packed:
                train_ds = CombinedIterableDataset(*train_parts) if len(train_parts) > 1 else train_parts[0]
                val_ds   = CombinedIterableDataset(*val_parts)   if len(val_parts)   > 1 else val_parts[0]
            else:
                train_ds = CombinedDataset(*train_parts) if len(train_parts) > 1 else train_parts[0]
                val_ds   = CombinedDataset(*val_parts)   if len(val_parts)   > 1 else val_parts[0]
        except Exception as e:
            if is_main:
                print(f"[Skip] Stage '{display_name}' failed to load — {type(e).__name__}: {e}")
            if use_ddp:
                accelerator.wait_for_everyone()
            continue

        _collate = packed_collate if use_packed else collate
        # IterableDataset (packed mode) doesn't support shuffle in DataLoader;
        # data diversity comes from the token-stream interleaving instead.
        base_train_loader = DataLoader(train_ds, batch_size=batch_size,
                                       shuffle=not use_packed,
                                       collate_fn=_collate, num_workers=0)
        val_loader        = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                                       collate_fn=_collate, num_workers=0)

        if use_ddp:
            base_train_loader, val_loader = accelerator.prepare(base_train_loader, val_loader)

        n_epochs     = 1 if stage.steps > 0 else max(stage.epochs, 1)
        train_loader = SliceLoader(base_train_loader, stage.steps) if stage.steps > 0 else base_train_loader

        if n_epochs == 0:
            if is_main:
                print(f"[Skip] Stage '{display_name}' has neither epochs nor steps > 0.")
            continue

        # ── Training ──────────────────────────────────────────────────────────
        safe_name = display_name.replace("+", "_")
        fig_path  = (str(Path(save_dir) / f"{safe_name}_loss.png")
                     if save_figures and is_main else None)
        try:
            common_kw = dict(
                min_val_accuracy = stage.min_val_accuracy,
                max_epochs       = stage.max_epochs,
                plateau_patience = stage.plateau_patience,
                plateau_min_delta= stage.plateau_min_delta,
            )
            if use_ddp:
                model = train_model_accelerate(
                    model         = model,
                    train_loader  = train_loader,
                    val_loader    = val_loader,
                    optimizer     = optimizer,
                    accelerator   = accelerator,
                    epochs        = n_epochs,
                    max_grad_norm = max_grad_norm,
                    scheduler     = scheduler,
                    ul_alpha      = ul_alpha,
                    fig_path      = fig_path,
                    **common_kw,
                )
            else:
                model = train_model(
                    model              = model,
                    train_loader       = train_loader,
                    val_loader         = val_loader,
                    optimizer          = optimizer,
                    device             = device,
                    epochs             = n_epochs,
                    ignore_index       = -100,
                    max_grad_norm      = max_grad_norm,
                    scheduler          = scheduler,
                    ul_alpha           = ul_alpha,
                    accumulation_steps = accumulation_steps,
                    fig_path           = fig_path,
                    **common_kw,
                )
        except (TypeError, AttributeError, NameError):
            # These are programming errors (e.g. a bad len()/API call), not a
            # flaky dataset — don't mask them as a soft skip, surface the bug.
            raise
        except Exception as e:
            if is_main:
                print(f"[Skip] Stage '{display_name}' training failed — {type(e).__name__}: {e}")
            if use_ddp:
                accelerator.wait_for_everyone()
            continue

        if is_main:
            ckpt_path = Path(save_dir) / f"{safe_name}_stage.pt"
            unwrapped = accelerator.unwrap_model(model) if use_ddp else model

            # Save checkpoint with real config for reproducibility
            config = {
                "vocab_size": unwrapped.token_emb.num_embeddings,
                "dim": unwrapped.token_emb.embedding_dim,
                "depth": unwrapped.depth,
                "heads": getattr(unwrapped.blocks[0].attn, "heads", 8),
                "kv_heads": getattr(unwrapped.blocks[0].attn, "kv_heads", 8),
                "mlp_dim": getattr(unwrapped.blocks[0].ff, "gate", None).out_features if hasattr(unwrapped.blocks[0].ff, "gate") else 2048,
                "window": unwrapped.max_seq_len,
                "tie_weights": True,  # This is the default in Transformer
            }
            torch.save({"config": config, "model_state": unwrapped.state_dict()}, ckpt_path)
            print(f"[Checkpoint] Saved {ckpt_path}")

        if use_ddp:
            accelerator.wait_for_everyone()

    return model
