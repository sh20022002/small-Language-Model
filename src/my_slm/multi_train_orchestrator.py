"""
multi_train_orchestrator.py
Train a single Transformer sequentially across multiple datasets
using your existing `my_slm.train.train_model` function.
"""

from dataclasses import dataclass
from typing import Iterable, List, Tuple, Optional, Dict

import math
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW

# External datasets
try:
    from datasets import load_dataset
except Exception as e:
    raise SystemExit("Please `pip install datasets` in your environment.") from e

# Your package imports
from my_slm.hybrid_tokeniztion import HybridTokenizer
from my_slm.transformer import Transformer
from my_slm.train import train_model  # <-- USES YOUR TRAIN LOOP

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
        raise ValueError(f"Unknown dataset {name}. Choose from: wikitext, tinystories, openwebtext, alpaca.")
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
            raise RuntimeError("No samples produced for dataset; check loader/text getter.")

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

# -----------------------------
# Orchestrator
# -----------------------------

@dataclass
class StageConfig:
    name: str
    epochs: int

def train_all_datasets(
    stages: Iterable[StageConfig] = (StageConfig("tinystories", 1),
                                     StageConfig("wikitext", 1),
                                     StageConfig("openwebtext", 1),
                                     StageConfig("alpaca", 1)),
    tokenizer_path: Optional[str] = None,
    build_examples: int = 20_000,
    k_bases: int = 5_000,
    max_merges: int = 10_000,
    max_len: int = 256,
    train_items: int = 50_000,
    val_items: int = 2_000,
    batch_size: int = 32,
    lr: float = 3e-4,
    weight_decay: float = 0.01,
    dim: int = 256,
    heads: int = 8,
    depth: int = 6,
    device: Optional[str] = None,
    save_dir: str = "./out",
):
    """
    Sequentially train a single model across multiple datasets using your `train_model`.

    Returns: (model, tokenizer)
    """
    os.makedirs(save_dir, exist_ok=True)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # ----- Tokenizer: load or build from the FIRST stage's stream -----
    if tokenizer_path and Path(tokenizer_path).exists():
        print(f"[Tokenizer] Loading from {tokenizer_path}")
        tok = HybridTokenizer.load(tokenizer_path)
    else:
        first_stage = next(iter(stages)).name if hasattr(stages, "__iter__") else "wikitext"
        print(f"[Tokenizer] Building from first stage '{first_stage}' (examples={build_examples})")
        stream, getter = get_hf_stream_and_text_getter(first_stage)
        tok = HybridTokenizer()
        for i, ex in enumerate(stream):
            tok.add_text(getter(ex))
            if (i + 1) >= build_examples:
                break
        tok.freeze_vocab(k_bases=k_bases, max_merges=max_merges)
        if tokenizer_path:
            Path(tokenizer_path).parent.mkdir(parents=True, exist_ok=True)
            tok.save(tokenizer_path)
            print(f"[Tokenizer] Saved to {tokenizer_path}")

    pad_id = tok.token2id["<PAD>"]

    # ----- Model -----
    model = Transformer(vocab_size=tok.vocab_size, dim=dim, heads=heads, depth=depth, max_seq_len=max_len)
    # Ensure vocab alignment if Transformer ignores vocab_size ctor
    if getattr(model, "token_emb", None) is not None and getattr(model.token_emb, "num_embeddings", None) != tok.vocab_size:
        try:
            model.resize_token_embeddings(tok.vocab_size)
        except Exception:
            pass
    model.to(device)

    # ----- Optimizer -----
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # ----- Stage loop -----
    for stage in stages:
        name = stage.name.lower()
        epochs = stage.epochs
        print(f"\n=== Stage: {name} | epochs={epochs} ===")

        # Build datasets & loaders
        train_stream, getter = get_hf_stream_and_text_getter(name)
        val_stream, _ = get_hf_stream_and_text_getter(name)  # reuse split for demo

        train_ds = TextTokenDataset(train_stream, getter, tok, max_len=max_len, max_items=train_items)
        val_ds   = TextTokenDataset(val_stream,   getter, tok, max_len=max_len, max_items=val_items)

        collate = make_collate(pad_id=pad_id, ignore_index=pad_id)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  collate_fn=collate, num_workers=0)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=0)

        # -----> Use YOUR train_model here <-----
        train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            device=device,
            epochs=epochs,
            ignore_index=pad_id,
        )

        # Save a stage checkpoint
        ckpt_path = Path(save_dir) / f"{stage.name}_stage.pt"
        torch.save({"model": model.state_dict(), "vocab_size": tok.vocab_size}, ckpt_path)
        print(f"[Checkpoint] Saved {ckpt_path}")

        # Optional: sample generate if model has .generate
        try:
            prompt = "Hello" if name != "alpaca" else "### Instruction:\nSay hello politely.\n\n### Response:\n"
            ids = tok.encode(prompt, mode="flat")[:max_len]
            if ids:
                x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
                if hasattr(model, "generate"):
                    y = model.generate(x, max_new_tokens=50, eos_token_id=tok.token2id.get("<EOS>"))
                else:
                    with torch.no_grad():
                        logits = model(x)
                        y = torch.cat([x, logits.argmax(-1)[:, -1:]], dim=1)
                print("[Sample]", tok.decode(y[0].tolist()))
        except Exception as e:
            print(f"[Sample] skipped ({e})")

    # Final
    final_path = Path(save_dir) / "final.pt"
    torch.save({"model": model.state_dict(), "vocab_size": tok.vocab_size}, final_path)
    print(f"[Final] Saved {final_path}")
    return model, tok
