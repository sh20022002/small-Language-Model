# multi_train_steps.py
from dataclasses import dataclass
from typing import Iterable, List, Optional, Dict
from pathlib import Path
import itertools
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# External datasets
try:
    from datasets import load_dataset
except Exception as e:
    raise SystemExit("Please: pip install datasets") from e

# Your package (we don't init model/tokenizer here)
from my_slm.hybrid_tokeniztion import HybridTokenizer
from my_slm.train import train_model  # your existing trainer (epoch-based)

# -----------------------------
# Dataset helpers
# -----------------------------

def get_hf_stream_and_text_getter(name: str):
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

class TextTokenDataset(Dataset):
    """Materializes a small list of token ID tensors for simple training on Colab."""
    def __init__(self, hf_stream, get_text, tokenizer: HybridTokenizer, max_len: int, max_items: Optional[int] = None):
        self.samples: List[torch.Tensor] = []
        n = 0
        for ex in hf_stream:
            text = get_text(ex)
            ids = tokenizer.encode(text, mode="flat")[:max_len]
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

def train_across_datasets(
    *,
    model,                          # <-- you pass an initialized model
    optimizer,                      # <-- you pass an initialized optimizer
    tokenizer: HybridTokenizer,     # <-- you pass an initialized tokenizer
    epochs: int = 3,
    stages: Iterable[StageConfig] = (
        StageConfig("tinystories", steps=1000),
        StageConfig("wikitext",   steps=2000),
        StageConfig("openwebtext", steps=2000),
        StageConfig("alpaca",     steps=1000),
    ),
    max_len: int = 256,
    train_items: int = 50_000,      # cap materialized items per stage (for Colab)
    val_items: int = 2_000,
    batch_size: int = 32,
    save_dir: str = "./out",
) -> None:
    """
    Train an existing model/optimizer over multiple datasets.
    You can choose epoch-based or step-based per stage:
      - If stage.steps > 0  -> train exactly that many batches (one "micro-epoch")
      - Else if stage.epochs > 0 -> use your `train_model` for full epochs

    This function DOES NOT initialize model/tokenizer/optimizer.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    pad_id = tokenizer.token2id["<PAD>"]

    for stage in stages:
        name = stage.name.lower()
        print(f"\n=== Stage: {name} | epochs={stage.epochs} | steps={stage.steps} ===")

        # Build datasets & loaders
        train_stream, getter = get_hf_stream_and_text_getter(name)
        val_stream, _        = get_hf_stream_and_text_getter(name)

        train_ds = TextTokenDataset(train_stream, getter, tokenizer, max_len=max_len, max_items=train_items)
        val_ds   = TextTokenDataset(val_stream,   getter, tokenizer, max_len=max_len, max_items=val_items)

        collate = make_collate(pad_id=pad_id, ignore_index=pad_id)
        base_train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  collate_fn=collate, num_workers=0)
        val_loader        = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=0)

        # ---- Option A: STEP-BASED training (limit number of batches) ----
        if stage.steps and stage.steps > 0:
            # Wrap the base loader to expose only `steps` batches
            train_loader = SliceLoader(base_train_loader, max_batches=stage.steps)
            # Call *your* epoch-based function with epochs=1,
            # but the loader only has `steps` batches so it trains that many steps.
            model = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                device=device,
                epochs=epochs,                  # one pass over our sliced loader
                ignore_index=pad_id,
            )
        # ---- Option B: EPOCH-BASED training (full passes over dataset) ----
        elif stage.epochs and stage.epochs > 0:
            model = train_model(
                model=model,
                train_loader=base_train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                device=device,
                epochs=stage.epochs,
                ignore_index=pad_id,
            )
        else:
            print(f"[Skip] Stage '{name}' has neither epochs nor steps > 0.")

        # Save per-stage checkpoint (optional)
        ckpt_path = Path(save_dir) / f"{stage.name}_stage.pt"
        torch.save({"model": model.state_dict()}, ckpt_path)
        print(f"[Checkpoint] Saved {ckpt_path}")
        return model
