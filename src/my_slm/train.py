import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import pandas as pd


def get_cosine_schedule_with_warmup(optimizer, warmup_steps: int, total_steps: int):
    """Linear warmup then cosine decay to 0. Step once per batch."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _repetition_ul_loss(logits, input_ids, labels, alpha):
    """Unlikelihood loss: penalise predicting the immediately preceding token."""
    B, T, V = logits.shape
    if T < 2 or alpha == 0.0:
        return logits.new_tensor(0.0)
    probs  = torch.softmax(logits[:, 1:], dim=-1)                              # [B, T-1, V]
    p_prev = probs.gather(-1, input_ids[:, :-1].unsqueeze(-1)).squeeze(-1)     # [B, T-1]
    valid  = (labels[:, 1:] != -100).float()
    ul     = -torch.log(1 - p_prev.clamp(max=1 - 1e-7)) * valid
    return alpha * ul.sum() / valid.sum().clamp(min=1)


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    epochs=5,
    ignore_index=-100,
    max_grad_norm=1.0,
    scheduler=None,
    ul_alpha=0.1,
    accumulation_steps=4,
):
    print('started Training...')
    device = torch.device(device) if isinstance(device, str) else device
    model.to(device)
    loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)

    # TF32 — free speedup on Ampere GPUs (A100, RTX 30xx); no-op on older GPUs
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # bfloat16 requires Ampere (compute capability ≥ 8.0, e.g. A100).
    # T4 is 7.5 — is_bf16_supported() can return True there but hardware
    # doesn't accelerate it; always use float16 + GradScaler on T4/V100.
    use_amp = device.type == "cuda"
    _major  = torch.cuda.get_device_capability()[0] if use_amp else 0
    amp_dtype = torch.bfloat16 if _major >= 8 else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp and amp_dtype == torch.float16)

    train_losses, val_losses, val_accuracies = [], [], []

    # Resize embeddings once before training if vocab grew after freeze
    if hasattr(model, "token_emb") and hasattr(model, "resize_token_embeddings"):
        try:
            all_ids = []
            for batch in train_loader:
                all_ids.append(int(batch["input_ids"].max()))
            if all_ids:
                max_id = max(all_ids)
                if max_id >= model.token_emb.num_embeddings:
                    model.resize_token_embeddings(max_id + 1)
                    print(f"[Info] Resized embeddings to {max_id + 1}")
        except Exception:
            pass  # streaming loaders may not support this pre-scan

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader):
            ids    = batch["input_ids"].to(device, non_blocking=True)
            attn   = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
                logits = model(ids, attention_mask=attn)    # [B, T, V]
                B, T, V = logits.shape
                loss = loss_fn(logits.reshape(B * T, V), labels.reshape(B * T))
                loss = loss + _repetition_ul_loss(logits, ids, labels, ul_alpha)
                loss = loss / accumulation_steps  # scale before backward

            scaler.scale(loss).backward()
            total_loss += loss.detach().item() * accumulation_steps  # unscale for logging

            is_last = (step + 1 == len(train_loader))
            if (step + 1) % accumulation_steps == 0 or is_last:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()

        avg_train_loss = total_loss / max(1, len(train_loader))
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                ids    = batch["input_ids"].to(device, non_blocking=True)
                attn   = batch["attention_mask"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)

                with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
                    logits = model(ids, attention_mask=attn)
                    B, T, V = logits.shape
                    loss = loss_fn(logits.reshape(B * T, V), labels.reshape(B * T))

                val_loss += loss.detach().item()

                pred = logits.argmax(dim=-1)
                mask = labels != ignore_index
                correct += (pred[mask] == labels[mask]).sum().item()
                total   += mask.sum().item()

        avg_val_loss = val_loss / max(1, len(val_loader))
        accuracy = (correct / max(total, 1)) * 100.0
        val_losses.append(avg_val_loss)
        val_accuracies.append(accuracy)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, Acc: {accuracy:.2f}%")

    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses,   label='Val Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return model



class QADataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # adjust keys to your data: 'question'/'answer' or 'input'/'output'
        q = item.get("question", item.get("input", ""))
        a = item.get("answer",   item.get("output", ""))
        text = f"Q: {q}\nA: {a}"
        tokens = self.tokenizer.encode(text)[:self.max_length]
        input_ids = torch.tensor(tokens, dtype=torch.long)
        return {"input_ids": input_ids}

def collate_fn(batch, pad_id=0, ignore_index=-100):
    ids = [b["input_ids"] for b in batch]                    # each [T]
    attn = [torch.ones_like(t, dtype=torch.long) for t in ids]  # 1s before pad

    ids  = torch.nn.utils.rnn.pad_sequence(ids,  batch_first=True, padding_value=pad_id)  # [B, T]
    attn = torch.nn.utils.rnn.pad_sequence(attn, batch_first=True, padding_value=0)       # [B, T]

    labels = ids.clone()
    labels[attn == 0] = ignore_index  # ignore padding in the loss

    return {"input_ids": ids, "attention_mask": attn, "labels": labels}
