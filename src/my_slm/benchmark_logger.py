"""
benchmark_logger.py
Benchmark a language model at any stage and log results (loss/acc/ppl, time, model name).
Works with your src/my_slm layout and HybridTokenizer.
"""

from __future__ import annotations
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# HF datasets (install if missing: pip install datasets)
try:
    from datasets import load_dataset
except Exception as e:
    raise SystemExit("Please install: pip install datasets") from e

# Your package
from my_slm.hybrid_tokeniztion import HybridTokenizer



def get_hf_stream_and_text_getter(name: str):
    name = name.lower()
    if name == "wikitext":
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation", streaming=True)
        getter = lambda ex: ex.get("text") or ""
    elif name == "tinystories":
        ds = load_dataset("roneneldan/TinyStories", split="validation", streaming=True)
        getter = lambda ex: ex.get("text") or ""
    elif name == "openwebtext":
        # no official valid; use train with a small cap for benchmarking
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
        raise ValueError(f"Unknown dataset {name}. Choose: wikitext | tinystories | openwebtext | alpaca")
    return ds, getter


class TextTokenDataset(Dataset):
    """Materialize N examples for simple, reproducible benchmarking."""
    def __init__(self, hf_stream, text_getter, tok: HybridTokenizer, max_len: int, max_items: int):
        self.samples: List[torch.Tensor] = []
        n = 0
        for ex in hf_stream:
            text = text_getter(ex)
            ids = tok.encode(text, mode="flat")[:max_len]
            if ids:
                self.samples.append(torch.tensor(ids, dtype=torch.long))
                n += 1
                if n >= max_items:
                    break
        if not self.samples:
            raise RuntimeError("No samples produced for benchmarking.")

    def __len__(self): return len(self.samples)
    def __getitem__(self, i): return self.samples[i]


def make_collate(pad_id: int, ignore_index: int):
    def collate(batch: List[torch.Tensor]):
        ids = pad_sequence(batch, batch_first=True, padding_value=pad_id)  # [B, T]
        attn = (ids != pad_id).long()
        labels = ids.clone()
        labels[ids == pad_id] = ignore_index
        return {"input_ids": ids, "attention_mask": attn, "labels": labels}
    return collate


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())




@dataclass
class BenchConfig:
    dataset: str = "wikitext"
    max_len: int = 256
    items: int = 2000
    batch_size: int = 32
    log_csv: str = "./benchmarks.csv"


@torch.no_grad()
def evaluate_once(model, loader: DataLoader, ignore_index: int, device: str) -> Dict[str, float]:
    model.eval()
    loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)
    total_loss = 0.0
    n_batches = 0
    n_tokens = 0
    n_correct = 0

    for batch in loader:
        ids  = batch["input_ids"].to(device).long()
        attn = batch["attention_mask"].to(device).bool()
        labs = batch["labels"].to(device).long()
        logits = model(ids, attention_mask=attn)           # [B, T, V]
        B, T, V = logits.shape
        loss = loss_fn(logits.reshape(B*T, V), labs.reshape(B*T))
        total_loss += loss.detach().item()
        n_batches += 1

        preds = logits.argmax(dim=-1)                      # [B, T]
        valid = labs != ignore_index
        n_tokens += int(valid.sum().item())
        n_correct += int((preds[valid] == labs[valid]).sum().item())

    avg_loss = total_loss / max(1, n_batches)
    ppl = float(torch.exp(torch.clamp(torch.tensor(avg_loss), max=20.0)).item())
    acc = (n_correct / max(1, n_tokens)) if n_tokens else 0.0
    return {"loss": avg_loss, "ppl": ppl, "acc": acc}


def benchmark_stage(
    *,
    model,
    tokenizer: HybridTokenizer,
    stage_name: str,
    model_base_name: Optional[str] = None,
    cfg: BenchConfig = BenchConfig(),
    device: Optional[str] = None,
) -> Dict[str, object]:
    """
    Run a quick evaluation on cfg.dataset and append a row to cfg.log_csv.
    Returns the metrics dict.
    """
    t_start = time.perf_counter()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    pad_id = tokenizer.token2id["<PAD>"]
    collate = make_collate(pad_id, pad_id)

    # Data
    stream, getter = get_hf_stream_and_text_getter(cfg.dataset)
    ds = TextTokenDataset(stream, getter, tokenizer, cfg.max_len, cfg.items)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate, num_workers=0)

    # Eval
    t0 = time.perf_counter()
    metrics = evaluate_once(model, loader, ignore_index=pad_id, device=device)
    elapsed_eval = time.perf_counter() - t0
    elapsed_total = time.perf_counter() - t_start

    # Model naming: base@stage
    base = model_base_name or model.__class__.__name__
    model_name = f"{base}@{stage_name}" if stage_name else base

    # Environment
    dev_name = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
    params = count_params(model)

    # Log to CSV
    out = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stage": stage_name,
        "model_name": model_name,
        "device": device,
        "device_name": dev_name,
        "params": params,
        "dataset": cfg.dataset,
        "max_len": cfg.max_len,
        "items": cfg.items,
        "batch_size": cfg.batch_size,
        "eval_s": round(elapsed_eval, 4),
        "wall_s": round(elapsed_total, 4),
        "loss": round(metrics["loss"], 6),
        "ppl": round(metrics["ppl"], 4),
        "acc": round(metrics["acc"], 4),
        "torch": torch.__version__,
    }

    log_path = Path(cfg.log_csv)
    write_header = not log_path.exists()
    with log_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out.keys()))
        if write_header:
            w.writeheader()
        w.writerow(out)

    # Pretty print
    print(f"[BENCH] {model_name} | {cfg.dataset} | loss={out['loss']:.4f} ppl={out['ppl']:.2f} "
          f"acc={out['acc']:.3f} | eval {out['eval_s']:.2f}s | dev={dev_name} | params={params/1e6:.1f}M")

    return out
