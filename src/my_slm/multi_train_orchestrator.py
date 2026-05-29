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
    else:
        raise ValueError(
            f"Unknown dataset {name}. "
            "Choose: wikitext, tinystories, openwebtext, c4, alpaca, dolly, gsm8k, openorca."
        )
    if skip > 0:
        ds = ds.skip(skip)
    return ds, getter


# Datasets with an official "validation" split on HF Hub
_HAS_VAL_SPLIT = {"wikitext", "tinystories", "c4"}
# Datasets whose held-out split is called "test" instead of "validation"
_HAS_TEST_SPLIT = {"gsm8k"}


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

def _encode(tokenizer, text: str, max_len: int | None = None) -> list:
    """Encode text with either HybridTokenizer or a HuggingFace tokenizer.

    Appends EOS to every sequence so the model learns when to stop generating.
    max_len reserves one slot for EOS, keeping total length within the window.
    """
    if hasattr(tokenizer, "token2id"):
        ids = tokenizer.encode(text, mode="flat")
        eos = tokenizer.token2id.get("<EOS>") or tokenizer.token2id.get("<PAD>")
        if max_len:
            ids = ids[: max_len - 1]
        if eos is not None:
            ids = ids + [eos]
        return ids
    else:
        kwargs = {"add_special_tokens": False}
        if max_len:
            kwargs["truncation"] = True
            kwargs["max_length"] = max_len - 1   # leave room for EOS
        ids = tokenizer.encode(text, **kwargs)
        eos_id = getattr(tokenizer, "eos_token_id", None)
        if eos_id is not None:
            ids = ids + [eos_id]
        return ids


class TextTokenDataset(Dataset):
    """Materializes a small list of token ID tensors for simple training on Colab."""
    def __init__(self, hf_stream, get_text, tokenizer, max_len: int, max_items: Optional[int] = None):
        self.samples: List[torch.Tensor] = []
        n = 0
        for ex in hf_stream:
            text = get_text(ex)
            ids = _encode(tokenizer, text, max_len=max_len)
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
    save_figures: bool = True,      # save per-stage loss curves as PNG in save_dir
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

    # Override the HF tokenizer's built-in model_max_length (GPT-2 defaults to 1024)
    # so it doesn't warn when sequences are longer than that but still within our window.
    if not hasattr(tokenizer, "token2id") and hasattr(tokenizer, "model_max_length"):
        tokenizer.model_max_length = max_len

    pad_id = _get_pad_id(tokenizer)
    collate = make_collate(pad_id=pad_id, ignore_index=-100)

    for stage in stages:
        name = stage.name.lower()
        if is_main:
            print(f"\n=== Stage: {name} | epochs={stage.epochs} | steps={stage.steps} ===")

        # ── Dataset loading ───────────────────────────────────────────────────
        try:
            train_stream, getter = get_hf_stream_and_text_getter(name)
            val_stream, _        = get_hf_val_stream_and_getter(name, skip_after=train_items)
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
        fig_path = (str(Path(save_dir) / f"{name}_loss.png")
                    if save_figures and is_main else None)
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
                    fig_path      = fig_path,
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
