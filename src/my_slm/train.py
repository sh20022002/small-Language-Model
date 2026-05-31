import math
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import pandas as pd


def _save_or_close_fig(fig_path, train_losses, val_losses, label="Loss"):
    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, label=f"Train {label}")
    plt.plot(val_losses,   label=f"Val {label}")
    plt.xlabel("Epoch"); plt.ylabel(label); plt.legend(); plt.tight_layout()
    if fig_path:
        Path(fig_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(fig_path, dpi=100, bbox_inches="tight")
        print(f"[Plot] Loss curve → {fig_path}")
    plt.close()


def get_cosine_schedule_with_warmup(optimizer, warmup_steps: int, total_steps: int):
    """Linear warmup then cosine decay to 0. Step once per batch."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _repetition_ul_loss(logits, input_ids, labels, alpha):
    """Unlikelihood loss on shifted logits: discourage predicting the immediately preceding token."""
    B, T, V = logits.shape
    if T < 2 or alpha == 0.0:
        return logits.new_tensor(0.0)
    # logits[:, t] predicts ids[:, t+1]; at each prediction step t, penalise P(ids[:, t])
    shift_logits = logits[:, :-1]                                               # [B, T-1, V]
    prev_ids     = input_ids[:, :-1]                                            # [B, T-1]
    shift_labels = labels[:, 1:]                                                # [B, T-1]
    probs  = torch.softmax(shift_logits, dim=-1)
    p_prev = probs.gather(-1, prev_ids.unsqueeze(-1)).squeeze(-1)              # [B, T-1]
    valid  = (shift_labels != -100).float()
    ul     = -torch.log(1 - p_prev.clamp(max=1 - 1e-7)) * valid
    return alpha * ul.sum() / valid.sum().clamp(min=1)


def make_optimizer(
    model,
    lr: float,
    weight_decay: float = 0.1,
    betas: tuple = (0.9, 0.95),
    use_8bit: bool = False,
    use_galore: bool = False,
    galore_rank: int = 128,
    galore_update_proj_gap: int = 200,
    galore_scale: float = 0.25,
):
    """
    Build AdamW with GPT-style weight-decay groups.

    use_8bit   : bitsandbytes AdamW8bit — shrinks optimizer state ~8×.
    use_galore : GaLore gradient projection on large weight matrices.
                 Reduces optimizer memory by projecting gradients to rank-r.
                 Pairs with use_8bit for maximum memory savings.
                 Requires: pip install galore-torch
    """
    decay, no_decay, galore_groups = [], [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_embedding = 'token_emb' in name
        is_matrix    = param.ndim >= 2

        if use_galore and is_matrix and not is_embedding:
            # GaLore is most effective on large projection matrices.
            galore_groups.append({
                'params':            [param],
                'rank':              galore_rank,
                'update_proj_gap':   galore_update_proj_gap,
                'scale':             galore_scale,
                'proj_type':         'std',
                'weight_decay':      weight_decay,
            })
        elif is_matrix and not is_embedding:
            decay.append(param)
        else:
            no_decay.append(param)

    base_groups = [
        {'params': decay,    'weight_decay': weight_decay},
        {'params': no_decay, 'weight_decay': 0.0},
    ]
    all_groups = base_groups + galore_groups

    if use_galore:
        try:
            from galore_torch import GaLoreAdamW8bit, GaLoreAdamW
        except ImportError as exc:
            raise ImportError("pip install galore-torch to use use_galore=True") from exc
        return (GaLoreAdamW8bit if use_8bit else GaLoreAdamW)(
            all_groups, lr=lr, betas=betas)

    if use_8bit:
        try:
            import bitsandbytes as bnb
            return bnb.optim.AdamW8bit(base_groups, lr=lr, betas=betas)
        except ImportError as exc:
            raise ImportError("pip install bitsandbytes to use use_8bit=True") from exc

    return torch.optim.AdamW(base_groups, lr=lr, betas=betas)


def load_latest_checkpoint(
    model,
    output_dir: str,
    models_dir: str | None = None,
    optimizer=None,
    accelerator=None,
) -> int:
    """
    Find and load the most recent checkpoint into model.

    Search order (first match wins):
      1. checkpoint-N/state.pt inside output_dir  — full state: weights + optimizer + step
      2. *_stage.pt files inside output_dir        — weights only (no optimizer state)
      3. *_stage.pt files inside models_dir        — weights only (pre-trained stages)

    Args:
        model:       the Transformer to load weights into.
        output_dir:  training output directory (e.g. "/kaggle/working/slm_run").
        models_dir:  optional directory of pre-trained stage .pt files
                     (e.g. "<repo_root>/models"). Used as fallback when
                     output_dir has no checkpoints at all.
        optimizer:   if provided, optimizer state is restored from step checkpoints.
        accelerator: Accelerator instance for DDP (used to unwrap model).

    Returns:
        global_step (int) — the saved step number, or 0 when loaded from a
        stage checkpoint (optimizer state is not available in that format).
    """
    raw_model = accelerator.unwrap_model(model) if accelerator is not None else model

    def _load_state(path):
        state = torch.load(path, map_location="cpu")
        raw_model.load_state_dict(state["model_state"], strict=False)
        return state

    # ── Priority 1: step checkpoint (checkpoint-N/state.pt) ───────────────────
    out = Path(output_dir)
    if out.is_dir():
        step_dirs = sorted(
            [d for d in out.iterdir()
             if d.is_dir() and d.name.startswith("checkpoint-")
             and (d / "state.pt").exists()],
            key=lambda d: int(d.name.split("-")[1]),
        )
        if step_dirs:
            state = _load_state(step_dirs[-1] / "state.pt")
            if optimizer is not None and "optimizer" in state:
                optimizer.load_state_dict(state["optimizer"])
            step = state.get("step", 0)
            print(f"[Checkpoint] Resumed from {step_dirs[-1].name}  (step={step})")
            return step

    # ── Priority 2: stage checkpoint in output_dir ────────────────────────────
    if out.is_dir():
        stage_pts = sorted(out.glob("*_stage.pt"), key=lambda p: p.stat().st_mtime)
        if stage_pts:
            _load_state(stage_pts[-1])
            print(f"[Checkpoint] Loaded stage weights: {stage_pts[-1].name}  (no optimizer state)")
            return 0

    # ── Priority 3: stage checkpoint in models_dir (pre-trained) ─────────────
    if models_dir:
        md = Path(models_dir)
        if not md.is_dir():
            print(f"[Checkpoint] models_dir not found: {md}  "
                  "(upload .pt files as a Kaggle dataset and set MODELS_DIR)")
        else:
            stage_pts = sorted(md.glob("*_stage.pt"), key=lambda p: p.stat().st_mtime)
            if stage_pts:
                _load_state(stage_pts[-1])
                print(f"[Checkpoint] Loaded pre-trained weights: {stage_pts[-1].name}")
                return 0
            else:
                print(f"[Checkpoint] models_dir exists but contains no *_stage.pt files: {md}")

    print("[Checkpoint] No checkpoint found — training from scratch.")
    return 0


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
    fig_path: str | None = None,
    min_val_accuracy: float = 0.0,
    max_epochs: int = 0,
    plateau_patience: int = 3,
    plateau_min_delta: float = 5e-4,
):
    print('started Training...')
    device = torch.device(device) if isinstance(device, str) else device
    model.to(device)
    loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)

    if device.type == "cuda":
        # TF32 — free speedup on Ampere GPUs (A100, RTX 30xx); no-op on older GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Enable memory-efficient SDPA backend (works on Turing/T4 and newer).
        # Flash-SDP is also enabled but only activates on Ampere+ hardware.
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)

    # bfloat16 requires Ampere (compute capability ≥ 8.0, e.g. A100).
    # T4 is 7.5 — is_bf16_supported() can return True there but hardware
    # doesn't accelerate it; always use float16 + GradScaler on T4/V100.
    use_amp = device.type == "cuda"
    _major  = torch.cuda.get_device_capability()[0] if use_amp else 0
    amp_dtype = torch.bfloat16 if _major >= 8 else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp and amp_dtype == torch.float16)

    train_losses, val_losses, val_accuracies = [], [], []
    # effective_max: run at least `epochs` epochs; extend up to `max_epochs` total
    # when accuracy gate is active.  If max_epochs==0, gate is disabled.
    effective_max = max(epochs, max_epochs) if max_epochs > 0 else epochs
    best_val_loss = float('inf')
    no_improve    = 0

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

    for epoch in range(effective_max):
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
                # Shift: logits[t] predicts ids[t+1]
                loss = loss_fn(logits[:, :-1].reshape(B * (T - 1), V),
                               labels[:, 1:].reshape(B * (T - 1)))
                loss = loss + _repetition_ul_loss(logits, ids, labels, ul_alpha)
                loss = loss / accumulation_steps  # scale before backward

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[Warning] NaN/Inf loss at step {step} — skipping batch")
                optimizer.zero_grad(set_to_none=True)
                continue

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
                    shift_logits = logits[:, :-1].reshape(B * (T - 1), V)
                    shift_labels = labels[:, 1:].reshape(B * (T - 1))
                    loss = loss_fn(shift_logits, shift_labels)

                val_loss += loss.detach().item()

                pred = logits[:, :-1].argmax(dim=-1)
                mask = labels[:, 1:] != ignore_index
                correct += (pred[mask] == labels[:, 1:][mask]).sum().item()
                total   += mask.sum().item()

        avg_val_loss = val_loss / max(1, len(val_loader))
        accuracy = (correct / max(total, 1)) * 100.0
        val_losses.append(avg_val_loss)
        val_accuracies.append(accuracy)

        # Plateau tracking
        if avg_val_loss < best_val_loss - plateau_min_delta:
            best_val_loss = avg_val_loss
            no_improve    = 0
        else:
            no_improve   += 1

        plateau_tag = f"  [plateau {no_improve}/{plateau_patience}]" if no_improve > 0 else ""
        print(f"Epoch {epoch+1}/{effective_max} — Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, Acc: {accuracy:.2f}%{plateau_tag}")

        # Accuracy gate: after minimum epochs, stop when target reached + plateau detected
        if max_epochs > 0 and epoch >= epochs - 1:
            if accuracy >= min_val_accuracy and no_improve >= plateau_patience:
                print(f"[EarlyStop] Acc {accuracy:.1f}% ≥ {min_val_accuracy:.1f}% target "
                      f"and loss plateaued — stopping at epoch {epoch+1}")
                break

    _save_or_close_fig(fig_path, train_losses, val_losses)
    return model


def train_model_accelerate(
    model,
    train_loader,
    val_loader,
    optimizer,
    accelerator,
    epochs=5,
    ignore_index=-100,
    max_grad_norm=1.0,
    scheduler=None,
    ul_alpha=0.1,
    fig_path: str | None = None,
    min_val_accuracy: float = 0.0,
    max_epochs: int = 0,
    plateau_patience: int = 3,
    plateau_min_delta: float = 5e-4,
):
    """
    Drop-in replacement for train_model that uses HuggingFace Accelerate.

    Prerequisites (caller's responsibility):
        accelerator = Accelerator(mixed_precision="fp16",
                                  gradient_accumulation_steps=N)
        model, optimizer, train_loader, val_loader = accelerator.prepare(
            model, optimizer, train_loader, val_loader)

    Differences from train_model:
      - No device arg; Accelerator handles placement.
      - No GradScaler; Accelerator handles mixed-precision.
      - accelerator.accumulate() drives gradient accumulation.
      - Only the main process prints and plots.
    """
    is_main = accelerator.is_main_process
    loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)

    if is_main:
        print("Started Training (Accelerate / DDP) …")

    train_losses, val_losses, val_accuracies = [], [], []
    effective_max = max(epochs, max_epochs) if max_epochs > 0 else epochs
    best_val_loss = float('inf')
    no_improve    = 0

    for epoch in range(effective_max):
        model.train()
        total_loss = torch.tensor(0.0, device=accelerator.device)

        for batch in train_loader:
            ids    = batch["input_ids"]
            attn   = batch["attention_mask"]
            labels = batch["labels"]

            with accelerator.accumulate(model):
                logits = model(ids, attention_mask=attn)
                B, T, V = logits.shape
                # Shift: logits[t] predicts ids[t+1]
                loss = loss_fn(logits[:, :-1].reshape(B * (T - 1), V),
                               labels[:, 1:].reshape(B * (T - 1)))
                loss = loss + _repetition_ul_loss(logits, ids, labels, ul_alpha)
                # nan_to_num preserves grad_fn (unlike zeros_like which detaches).
                # NaN/Inf → 0.0 gives zero gradients for that step (safe no-op).
                if not loss.isfinite():
                    loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)

                optimizer.step()
                if scheduler is not None and accelerator.sync_gradients:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            total_loss += loss.detach()

        # Aggregate loss across all processes
        total_loss = accelerator.reduce(total_loss, reduction="mean")
        avg_train_loss = total_loss.item() / max(1, len(train_loader))
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                ids    = batch["input_ids"]
                attn   = batch["attention_mask"]
                labels = batch["labels"]

                logits = model(ids, attention_mask=attn)
                B, T, V = logits.shape
                shift_logits = logits[:, :-1].reshape(B * (T - 1), V)
                shift_labels = labels[:, 1:].reshape(B * (T - 1))
                l = loss_fn(shift_logits, shift_labels)
                val_loss += l.detach().item()

                pred = logits[:, :-1].argmax(dim=-1)
                mask = labels[:, 1:] != ignore_index
                correct += (pred[mask] == labels[:, 1:][mask]).sum().item()
                total   += mask.sum().item()

        avg_val_loss = val_loss / max(1, len(val_loader))
        accuracy     = (correct / max(total, 1)) * 100.0
        val_losses.append(avg_val_loss)
        val_accuracies.append(accuracy)

        # Plateau tracking
        if avg_val_loss < best_val_loss - plateau_min_delta:
            best_val_loss = avg_val_loss
            no_improve    = 0
        else:
            no_improve   += 1

        if is_main:
            plateau_tag = f"  [plateau {no_improve}/{plateau_patience}]" if no_improve > 0 else ""
            print(f"Epoch {epoch+1}/{effective_max} — Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {avg_val_loss:.4f}, Acc: {accuracy:.2f}%{plateau_tag}")

        # Accuracy gate: after minimum epochs, stop when target reached + plateau detected
        if max_epochs > 0 and epoch >= epochs - 1:
            if accuracy >= min_val_accuracy and no_improve >= plateau_patience:
                if is_main:
                    print(f"[EarlyStop] Acc {accuracy:.1f}% ≥ {min_val_accuracy:.1f}% target "
                          f"and loss plateaued — stopping at epoch {epoch+1}")
                break

    if is_main:
        _save_or_close_fig(fig_path, train_losses, val_losses)

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
