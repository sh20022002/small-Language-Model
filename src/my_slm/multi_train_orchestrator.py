from dataclasses import dataclass
from typing import Iterable, List, Optional
from pathlib import Path
import itertools
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from my_slm.train import train_model, train_model_accelerate

# -----------------------------
# Dataset helpers
# -----------------------------

def get_hf_stream_and_text_getter(name: str):
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError("pip install datasets") from e
    name = name.lower()
    if name == "wikitext":
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
        getter = lambda ex: ex.get("text") or ""
    elif name == "tinystories":
        ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
        getter = lambda ex: ex.get("text") or ""
    elif name == "openwebtext":
        ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
        getter = lambda ex: ex.get("text") or ""
    elif name == "alpaca":
        ds = load_dataset("yahma/alpaca-cleaned", split="train", streaming=True)
        def getter(ex):
            ins = ex.get("instruction") or ""
            inp = ex.get("input") or ""
            out = ex.get("output") or ""
            if inp:
                return f"### Instruction:\n{ins}\n\n### Input:\n{inp}\n\n### Response:\n{out}\n"
            else:
                return f"### Instruction:\n{ins}\n\n### Response:\n{out}\n"
        getter = getter
    else:
        raise ValueError(f"Unknown dataset {name}. Choose: wikitext, tinystories, openwebtext, alpaca.")
    return ds, getter

def _encode(tokenizer, text: str, max_len: int | None = None) -> list:
    """Encode text with either HybridTokenizer or a HuggingFace tokenizer."""
    if hasattr(tokenizer, "token2id"):
        ids = tokenizer.encode(text, mode="flat")           # HybridTokenizer
        return ids[:max_len] if max_len else ids
    else:
        # Pass truncation to the HF tokenizer so it never warns about length
        kwargs = {"add_special_tokens": False}
        if max_len:
            kwargs["truncation"] = True
            kwargs["max_length"] = max_len
        return tokenizer.encode(text, **kwargs)


class TextTokenDataset(Dataset):
    """Materializes a small list of token ID tensors for simple training on Colab."""
    def __init__(self, hf_stream, get_text, tokenizer, max_len: int, max_items: Optional[int] = None):
        self.samples: List[torch.Tensor] = []
        n = 0
        for ex in hf_stream:
            text = get_text(ex)
            ids = _encode(tokenizer, text)[:max_len]
            if ids:
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
            # some loaders may not have __len__
            return self.max_batches

# -----------------------------
# Orchestrator (TRAIN-ONLY)
# -----------------------------

@dataclass
class StageConfig:
    name: str
    # Either train for full epochs OR for a fixed number of steps (batches)
    epochs: int = 0
    steps: int = 0  # if >0, we train only this many steps for the stage

def _get_pad_id(tokenizer) -> int:
    """Return pad token id for HybridTokenizer or HuggingFace tokenizer."""
    if hasattr(tokenizer, "token2id"):
        return tokenizer.token2id.get("<PAD>", 0)
    if hasattr(tokenizer, "pad_token_id") and tokenizer.pad_token_id is not None:
        return tokenizer.pad_token_id
    return 0


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

    pad_id = _get_pad_id(tokenizer)
    collate = make_collate(pad_id=pad_id, ignore_index=-100)

    for stage in stages:
        name = stage.name.lower()
        if is_main:
            print(f"\n=== Stage: {name} | epochs={stage.epochs} | steps={stage.steps} ===")

        # ── Dataset loading ───────────────────────────────────────────────────
        try:
            train_stream, getter = get_hf_stream_and_text_getter(name)
            val_stream, _        = get_hf_stream_and_text_getter(name)
            train_ds = TextTokenDataset(train_stream, getter, tokenizer, max_len=max_len, max_items=train_items)
            val_ds   = TextTokenDataset(val_stream,   getter, tokenizer, max_len=max_len, max_items=val_items)
        except Exception as e:
            if is_main:
                print(f"[Skip] Stage '{name}' failed to load — {type(e).__name__}: {e}")
            if use_ddp:
                accelerator.wait_for_everyone()
            continue

        base_train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  collate_fn=collate, num_workers=0)
        val_loader        = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=0)

        # DDP path: let Accelerator distribute the loaders
        if use_ddp:
            base_train_loader, val_loader = accelerator.prepare(base_train_loader, val_loader)

        n_epochs = 1 if stage.steps > 0 else max(stage.epochs, 1)
        train_loader = SliceLoader(base_train_loader, stage.steps) if stage.steps > 0 else base_train_loader

        if n_epochs == 0:
            if is_main:
                print(f"[Skip] Stage '{name}' has neither epochs nor steps > 0.")
            continue

        # ── Training ──────────────────────────────────────────────────────────
        try:
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
                )
        except Exception as e:
            if is_main:
                print(f"[Skip] Stage '{name}' training failed — {type(e).__name__}: {e}")
            if use_ddp:
                accelerator.wait_for_everyone()
            continue

        if is_main:
            ckpt_path = Path(save_dir) / f"{stage.name}_stage.pt"
            unwrapped = accelerator.unwrap_model(model) if use_ddp else model
            torch.save({"config": {}, "model_state": unwrapped.state_dict()}, ckpt_path)
            print(f"[Checkpoint] Saved {ckpt_path}")

        if use_ddp:
            accelerator.wait_for_everyone()

    return model
