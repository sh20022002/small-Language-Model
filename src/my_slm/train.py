"""Training utilities: loss functions, optimizers, learning rate schedules."""

from __future__ import annotations

import logging
import math
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from torch.utils.data import DataLoader, Dataset

_logger = logging.getLogger(__name__)


def _save_or_close_fig(
    fig_path: Optional[str],
    train_losses: List[float],
    val_losses: List[float],
    label: str = "Loss",
) -> None:
    """Save training curve plot to file.

    Uses object-oriented Matplotlib API (Figure + Agg canvas) instead of pyplot
    to avoid headless environment crashes and avoid disturbing other matplotlib
    backends (like %matplotlib inline in notebooks).

    Args:
        fig_path: Path to save figure. If None, does nothing.
        train_losses: Training losses per epoch.
        val_losses: Validation losses per epoch.
        label: Label for y-axis (default "Loss").
    """
    if not fig_path:
        return
    fig = Figure(figsize=(6, 4))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.plot(train_losses, label=f"Train {label}")
    ax.plot(val_losses, label=f"Val {label}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(label)
    ax.legend()
    fig.tight_layout()
    Path(fig_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=100, bbox_inches="tight")
    _logger.info(f"Saved loss curve → {fig_path}")


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int
) -> torch.optim.lr_scheduler.LambdaLR:
    """Create cosine annealing schedule with linear warmup.

    Learning rate increases linearly during warmup, then decays following
    cosine annealing to zero.

    Args:
        optimizer: Optimizer to schedule.
        warmup_steps: Number of steps to warm up over.
        total_steps: Total training steps.

    Returns:
        LambdaLR scheduler. Call step() once per batch.
    """
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _repetition_ul_loss(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Unlikelihood loss: discourage repeating the previous token.

    Applies a penalty to the model's logits for predicting the previous token
    in the sequence. This encourages diversity in generation.

    Args:
        logits: Unnormalized log probabilities [B, T, V].
        input_ids: Input token IDs [B, T].
        labels: Target token IDs [B, T]. Tokens marked as -100 are ignored.
        alpha: Loss weight. Set to 0 to disable.

    Returns:
        Scalar loss tensor. 0 if alpha=0 or T<2.
    """
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
    model: nn.Module,
    lr: float,
    weight_decay: float = 0.1,
    betas: Tuple[float, float] = (0.9, 0.95),
    use_8bit: bool = False,
    use_galore: bool = False,
    galore_rank: int = 128,
    galore_update_proj_gap: int = 200,
    galore_scale: float = 0.25,
) -> torch.optim.Optimizer:
    """Create AdamW optimizer with GPT-style weight decay groups.

    Args:
        model: Model to optimize.
        lr: Learning rate.
        weight_decay: Weight decay (L2 penalty). Applied to matrix params, not scalars.
        betas: Momentum coefficients (beta1, beta2).
        use_8bit: Use bitsandbytes AdamW8bit for ~8x memory reduction.
        use_galore: Use GaLore gradient projection on large matrices.
        galore_rank: Rank for GaLore projection.
        galore_update_proj_gap: Update interval for GaLore projections.
        galore_scale: Scale factor for GaLore gradients.

    Returns:
        Configured optimizer.

    Raises:
        ImportError: If use_8bit or use_galore requested but packages not installed.
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
    stage_order: list[str] | None = None,
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
        stage_order: optional curriculum order of stage names (the safe_names,
                     e.g. ["tinystories_stories", "wikitext_c4", ...]). When
                     given, the furthest-along stage checkpoint is chosen instead
                     of the newest by mtime — important for uploaded Kaggle
                     datasets where all files share the same upload timestamp.

    Returns:
        global_step (int) — the saved step number, or 0 when loaded from a
        stage checkpoint (optimizer state is not available in that format).
    """
    raw_model = accelerator.unwrap_model(model) if accelerator is not None else model

    # Only the main DDP rank logs, so messages aren't duplicated per-GPU.
    _is_main = accelerator is None or accelerator.is_main_process
    def _log(*a):
        if _is_main:
            print(*a)

    def _pick_stage(stage_pts):
        """Choose the furthest-along stage checkpoint (by curriculum order when
        known, else newest mtime)."""
        if stage_order:
            def _rank(p):
                name = p.stem[:-6] if p.stem.endswith("_stage") else p.stem
                return stage_order.index(name) if name in stage_order else -1
            # Tie-break unknown names (rank -1) by mtime so they don't win.
            return max(stage_pts, key=lambda p: (_rank(p), p.stat().st_mtime))
        return max(stage_pts, key=lambda p: p.stat().st_mtime)

    def _load_state(path):
        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
        except pickle.UnpicklingError as e:
            # weights_only=True refused to unpickle something in this file.
            # Do NOT silently retry with weights_only=False — on torch<2.6
            # that path (and even weights_only=True pre-2.6, see
            # CVE-2025-32434) can execute arbitrary code from a malicious
            # checkpoint. Checkpoints here may come from Kaggle datasets
            # attached by someone other than the notebook owner, so fail
            # loudly instead of quietly deserializing untrusted pickle data.
            raise RuntimeError(
                f"Refusing to load {path}: weights_only=True rejected its "
                f"contents ({e}). If you trust this file's origin and need "
                f"to load it anyway, do so explicitly with "
                f"torch.load(path, weights_only=False) — never as a silent "
                f"fallback. See SECURITY.md."
            ) from e
        except TypeError:
            # Only torch<2.1 (no weights_only kwarg at all) lands here — a
            # real API-compatibility case, not a masked deserialization
            # failure. torch>=2.6 is required by pyproject.toml, so this is
            # effectively dead code kept for defense in depth.
            _log(f"⚠️  torch.load() has no weights_only kwarg on this PyTorch "
                 f"version — upgrade to torch>=2.6 (see SECURITY.md). "
                 f"Loading {path} without weights_only protection.")
            state = torch.load(path, map_location="cpu")

        # Load with logging for mismatches
        missing, unexpected = raw_model.load_state_dict(state.get("model_state", state), strict=False)
        if missing or unexpected:
            if missing:
                _log(f"⚠️  Missing keys in checkpoint: {missing[:3]}{'...' if len(missing) > 3 else ''}")
            if unexpected:
                _log(f"⚠️  Unexpected keys in checkpoint: {unexpected[:3]}{'...' if len(unexpected) > 3 else ''}")
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
            _log(f"[Checkpoint] Resumed from {step_dirs[-1].name}  (step={step})")
            return step

    # ── Priority 2: stage checkpoint in output_dir ────────────────────────────
    if out.is_dir():
        stage_pts = list(out.glob("*_stage.pt"))
        if stage_pts:
            best = _pick_stage(stage_pts)
            _load_state(best)
            _log(f"[Checkpoint] Loaded stage weights: {best.name}  (no optimizer state)")
            return 0

    # ── Priority 3: stage checkpoint in models_dir (pre-trained) ─────────────
    if models_dir:
        md = Path(models_dir)
        if not md.is_dir():
            _log(f"[Checkpoint] models_dir not found: {md}  "
                  "(upload .pt files as a Kaggle dataset and set MODELS_DIR)")
        else:
            stage_pts = list(md.glob("*_stage.pt"))
            if stage_pts:
                best = _pick_stage(stage_pts)
                _load_state(best)
                _log(f"[Checkpoint] Loaded pre-trained weights: {best.name}")
                return 0
            else:
                _log(f"[Checkpoint] models_dir exists but contains no *_stage.pt files: {md}")

    _log("[Checkpoint] No checkpoint found — training from scratch.")
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
        n_train_batches = 0
        pending_grads   = False
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
            n_train_batches += 1
            pending_grads   = True

            if (step + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                pending_grads = False
                if scheduler is not None:
                    scheduler.step()

        # Flush gradients left over from a final partial accumulation window.
        # Replaces the old `is_last` check, which needed len(train_loader) and
        # so broke on IterableDataset (packed mode).
        if pending_grads:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()

        avg_train_loss = total_loss / max(1, n_train_batches)
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        n_val_batches = 0
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
                n_val_batches += 1

                pred = logits[:, :-1].argmax(dim=-1)
                mask = labels[:, 1:] != ignore_index
                correct += (pred[mask] == labels[:, 1:][mask]).sum().item()
                total   += mask.sum().item()

        avg_val_loss = val_loss / max(1, n_val_batches)
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
        n_train_batches = 0

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
            n_train_batches += 1

        # Aggregate loss across all processes. len(train_loader) is unavailable
        # for IterableDataset (packed mode), so divide by the counted batches.
        total_loss = accelerator.reduce(total_loss, reduction="mean")
        avg_train_loss = total_loss.item() / max(1, n_train_batches)
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        n_val_batches = 0
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
                n_val_batches += 1

                pred = logits[:, :-1].argmax(dim=-1)
                mask = labels[:, 1:] != ignore_index
                correct += (pred[mask] == labels[:, 1:][mask]).sum().item()
                total   += mask.sum().item()

        avg_val_loss = val_loss / max(1, n_val_batches)
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
    """Question-Answer dataset for training language models.

    Formats data as "Q: {question}\\nA: {answer}" and tokenizes.
    Supports alternative keys 'input'/'output' as fallback.
    """

    def __init__(
        self,
        data: List[Dict[str, str]],
        tokenizer: Any,
        max_length: int = 128,
    ) -> None:
        """Initialize QA dataset.

        Args:
            data: List of dicts with 'question' + 'answer' (or 'input' + 'output').
            tokenizer: Tokenizer with encode() method.
            max_length: Maximum sequence length after tokenization.
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a QA example.

        Args:
            idx: Dataset index.

        Returns:
            Dict with "input_ids" tensor of shape [max_length].
        """
        item = self.data[idx]
        # Support both 'question'/'answer' and 'input'/'output' keys
        q = item.get("question", item.get("input", ""))
        a = item.get("answer", item.get("output", ""))
        text = f"Q: {q}\nA: {a}"
        tokens = self.tokenizer.encode(text)[: self.max_length]
        input_ids = torch.tensor(tokens, dtype=torch.long)
        return {"input_ids": input_ids}


def collate_fn(
    batch: List[Dict[str, torch.Tensor]],
    pad_id: int = 0,
    ignore_index: int = -100,
) -> Dict[str, torch.Tensor]:
    """Collate batch of QA examples.

    Pads sequences to max length in batch and creates attention masks.

    Args:
        batch: List of dicts from QADataset with "input_ids" key.
        pad_id: Token ID to use for padding.
        ignore_index: Label value to mark padding for loss computation.

    Returns:
        Dict with batched tensors:
        - "input_ids": [B, T]
        - "attention_mask": [B, T] (1 for real tokens, 0 for padding)
        - "labels": [B, T] (copy of input_ids, with padding masked as ignore_index)
    """
    ids = [b["input_ids"] for b in batch]  # each [T]
    attn = [torch.ones_like(t, dtype=torch.long) for t in ids]  # 1s before pad

    ids = torch.nn.utils.rnn.pad_sequence(ids, batch_first=True, padding_value=pad_id)  # [B, T]
    attn = torch.nn.utils.rnn.pad_sequence(attn, batch_first=True, padding_value=0)  # [B, T]

    labels = ids.clone()
    labels[attn == 0] = ignore_index  # ignore padding in the loss

    return {"input_ids": ids, "attention_mask": attn, "labels": labels}
