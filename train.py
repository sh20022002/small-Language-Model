import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib as plt

from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import pandas as pd



def train_model(model, train_loader, val_loader, optimizer, device, epochs=5):
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    train_losses, val_losses, val_accuracies = [], [], []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            

            logits = model(input_ids)
            loss = loss_fn(logits.view(-1, logits.size(-1)), input_ids.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                

                logits = model(input_ids)
                loss = loss_fn(logits.view(-1, logits.size(-1)), input_ids.view(-1))
                val_loss += loss.item()

                pred = torch.argmax(logits, dim=-1)
                correct += (pred == input_ids).sum().item()
                total += input_ids.numel()

        avg_val_loss = val_loss / len(val_loader)
        accuracy = correct / total * 100
        val_losses.append(avg_val_loss)
        val_accuracies.append(accuracy)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.2f}%")

    # Plotting results
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    loss_vals = pd.DataFrame(list(zip(train_losses, val_losses)), columns=['train', 'val'])
    loss_vals.to_csv('losses.csv')

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
        text = f"Q: {item['question']}\\nA: {item['answer']}"
        tokens = self.tokenizer.encode(text)[:self.max_length]
        input_ids = torch.tensor(tokens, dtype=torch.long)
        return {"input_ids": input_ids}

def collate_fn(batch):
    input_ids = [item["input_ids"] for item in batch]
    padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    return {"input_ids": padded}


def load_train(data, tokenizer):
    train_data, val_data = train_test_split(data, test_size=0.1, random_state=42)

    train_dataset = QADataset(train_data, tokenizer)
    val_dataset = QADataset(val_data, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)
    return train_loader, val_loader