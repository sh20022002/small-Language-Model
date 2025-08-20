import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib as plt

from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import pandas as pd



def train_model(model, train_loader, val_loader, optimizer, device, epochs=5, ignore_index=-100):
    model.to(device)
    loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)

    train_losses, val_losses, val_accuracies = [], [], []

    # (optional) grab embedding to know vocab size
    emb = next((m for m in model.modules() if isinstance(m, nn.Embedding)), None)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            ids    = batch["input_ids"].to(device)          # [B, T]
            attn   = batch["attention_mask"].to(device)     # [B, T]
            labels = batch["labels"].to(device)             # [B, T] with -100 on pads

           
            if emb is not None:
                mx, mn = int(ids.max()), int(ids.min())
                assert mn >= 0, f"Negative token id: {mn}"
                if mx >= model.token_emb.num_embeddings:
                    model.resize_token_embeddings(mx + 1)


            optimizer.zero_grad()

            try:
                logits = model(ids, attention_mask=attn)    # [B, T, V]
            except TypeError:
                logits = model(ids)

            # labels vs logits vocab check (excluding ignore_index)
            if (labels != ignore_index).any():
                V = logits.size(-1)
                max_lab = int(labels[labels != ignore_index].max())
                assert max_lab < V, f"Label {max_lab} >= logits classes {V}"

            loss = loss_fn(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            loss.backward()
            optimizer.step()

            total_loss += float(loss)

        avg_train_loss = total_loss / max(1, len(train_loader))
        train_losses.append(avg_train_loss)

        
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                ids    = batch["input_ids"].to(device, non_blocking=True)
                attn   = batch["attention_mask"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)

                try:
                    logits = model(ids, attention_mask=attn)
                except TypeError:
                    logits = model(ids)

                loss = loss_fn(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
                val_loss += float(loss)

                pred = logits.argmax(dim=-1)               # [B, T]
                mask = labels != ignore_index              # ignore pads in accuracy
                correct += (pred[mask] == labels[mask]).sum().item()
                total   += mask.sum().item()

        avg_val_loss = val_loss / max(1, len(val_loader))
        accuracy = (correct / max(total, 1)) * 100.0
        val_losses.append(avg_val_loss)
        val_accuracies.append(accuracy)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, Acc: {accuracy:.2f}%")

    # plots + csv as you had
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1); plt.plot(train_losses, label='Train Loss'); plt.plot(val_losses, label='Val Loss')
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
    plt.subplot(1,2,2); plt.plot(val_accuracies, label='Val Accuracy')
    plt.xlabel("Epoch"); plt.ylabel("Accuracy (%)"); plt.legend()
    plt.tight_layout(); plt.show()

    pd.DataFrame({"train": train_losses, "val": val_losses}).to_csv("losses.csv", index=False)
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
