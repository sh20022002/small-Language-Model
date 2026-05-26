"""
benchmark_logger.py
Evaluate a language model at any stage and log results (loss/ppl/acc, timing).
"""
from __future__ import annotations

import csv
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Dataset helpers live in multi_train_orchestrator — imported here to avoid duplication
from my_slm.multi_train_orchestrator import (
    get_hf_stream_and_text_getter,
    TextTokenDataset,
    make_collate,
    _get_pad_id,
    _encode,
)


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


@dataclass
class BenchConfig:
    dataset:    str   = "wikitext"
    max_len:    int   = 256
    items:      int   = 2_000
    batch_size: int   = 32
    log_csv:    str   = "./benchmarks.csv"


@torch.no_grad()
def evaluate_once(
    model,
    loader: DataLoader,
    ignore_index: int,
    device: str,
) -> Dict[str, float]:
    """Run one full pass over `loader` and return loss, ppl, and top-1 accuracy."""
    model.eval()
    loss_fn     = nn.CrossEntropyLoss(ignore_index=ignore_index)
    total_loss  = 0.0
    n_batches   = 0
    n_tokens    = 0
    n_correct   = 0

    for batch in loader:
        ids   = batch["input_ids"].to(device).long()
        attn  = batch["attention_mask"].to(device).bool()
        labs  = batch["labels"].to(device).long()

        logits = model(ids, attention_mask=attn)           # [B, T, V]
        B, T, V = logits.shape
        loss   = loss_fn(logits.reshape(B * T, V), labs.reshape(B * T))
        total_loss += loss.detach().item()
        n_batches  += 1

        preds     = logits.argmax(dim=-1)
        valid     = labs != ignore_index
        n_tokens  += int(valid.sum().item())
        n_correct += int((preds[valid] == labs[valid]).sum().item())

    avg_loss = total_loss / max(1, n_batches)
    ppl      = float(torch.exp(torch.clamp(torch.tensor(avg_loss), max=20.0)).item())
    acc      = n_correct / max(1, n_tokens)
    return {"loss": avg_loss, "ppl": ppl, "acc": acc}


def benchmark_stage(
    *,
    model,
    tokenizer,
    stage_name: str,
    model_base_name: Optional[str] = None,
    cfg: BenchConfig = field(default_factory=BenchConfig),
    device: Optional[str] = None,
) -> Dict[str, object]:
    """
    Evaluate model on cfg.dataset and append one row to cfg.log_csv.
    tokenizer can be HybridTokenizer or a HuggingFace tokenizer.
    Returns the metrics dict.
    """
    if cfg is None or not isinstance(cfg, BenchConfig):
        cfg = BenchConfig()

    t_start = time.perf_counter()
    device  = device or ("cuda" if torch.cuda.is_available() else "cpu")
    pad_id  = _get_pad_id(tokenizer)
    collate = make_collate(pad_id, ignore_index=pad_id)

    stream, getter = get_hf_stream_and_text_getter(cfg.dataset)
    ds     = TextTokenDataset(stream, getter, tokenizer, cfg.max_len, cfg.items)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False,
                        collate_fn=collate, num_workers=0)

    t0      = time.perf_counter()
    metrics = evaluate_once(model, loader, ignore_index=pad_id, device=device)
    elapsed_eval  = time.perf_counter() - t0
    elapsed_total = time.perf_counter() - t_start

    base       = model_base_name or model.__class__.__name__
    model_name = f"{base}@{stage_name}" if stage_name else base
    dev_name   = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
    params     = count_params(model)

    out = {
        "timestamp":  time.strftime("%Y-%m-%d %H:%M:%S"),
        "stage":      stage_name,
        "model_name": model_name,
        "device":     device,
        "device_name":dev_name,
        "params":     params,
        "dataset":    cfg.dataset,
        "max_len":    cfg.max_len,
        "items":      cfg.items,
        "batch_size": cfg.batch_size,
        "eval_s":     round(elapsed_eval, 4),
        "wall_s":     round(elapsed_total, 4),
        "loss":       round(metrics["loss"], 6),
        "ppl":        round(metrics["ppl"], 4),
        "acc":        round(metrics["acc"], 4),
        "torch":      torch.__version__,
    }

    log_path   = Path(cfg.log_csv)
    write_hdr  = not log_path.exists()
    with log_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out.keys()))
        if write_hdr:
            w.writeheader()
        w.writerow(out)

    print(f"[BENCH] {model_name} | {cfg.dataset} | "
          f"loss={out['loss']:.4f}  ppl={out['ppl']:.2f}  acc={out['acc']:.3f} | "
          f"eval {out['eval_s']:.2f}s | {dev_name} | {params/1e6:.1f}M params")

    return out
