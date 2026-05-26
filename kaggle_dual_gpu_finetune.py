# =============================================================================
# Custom SLM — Kaggle 2x Tesla T4 (2x16 GB VRAM) — Train from Scratch
#
# Kaggle runtime settings (Settings panel, right side):
#   Accelerator  → GPU T4 x2
#   Internet     → On
#
# Key optimisations over the Colab version:
#   • DDP (DistributedDataParallel) via accelerate → both GPUs contribute equally
#   • bitsandbytes 8-bit Adam → optimizer states shrink ~8x vs fp32 Adam
#   • Gradient checkpointing   → recompute activations on backward (~30% less VRAM)
#   • float16 AMP              → T4 has fast fp16 hardware, no bfloat16 support
#   • Gradient accumulation ×16 → effective batch stays large despite bs=1/device
#   • Step-based checkpointing  → Kaggle session ≤12 h; resume automatically
# =============================================================================


# %% [markdown]
# ## Cell 1 — Install / upgrade dependencies

# %%
# !pip install -q -U accelerate bitsandbytes datasets transformers


# %% [markdown]
# ## Cell 2 — GPU diagnostics (run outside the DDP launcher)

# %%
import torch
n_gpus = torch.cuda.device_count()
print(f"GPUs visible: {n_gpus}")
for i in range(n_gpus):
    p = torch.cuda.get_device_properties(i)
    print(f"  cuda:{i}  {p.name}  {p.total_memory/1e9:.1f} GB")

if n_gpus < 2:
    print("WARNING: only 1 GPU found. Enable 'GPU T4 x2' in Kaggle Settings.")


# %% [markdown]
# ## Cell 3 — Configuration
# Tune these before launching; everything else is automatic.

# %%
CFG = {
    # ── Output ──────────────────────────────────────────────────────────────
    "output_dir": "/kaggle/working/slm_run",

    # ── Model size presets ─────────────────────────────────────────────────
    # Uncomment one config; default is "large" (~1.3 B params on 2x T4).
    #
    # "tiny"   : dim=256,  depth=4,  heads=4,  mlp=1024  → ~30M  (quick test)
    # "small"  : dim=512,  depth=8,  heads=8,  mlp=2048  → ~110M
    # "medium" : dim=768,  depth=12, heads=12, mlp=3072  → ~250M
    # "large"  : dim=1024, depth=16, heads=16, mlp=4096  → ~650M
    # "xlarge" : dim=1536, depth=24, heads=24, mlp=6144  → ~1.3B ← default
    "dim":      1536,
    "depth":    24,
    "heads":    24,
    "mlp_dim":  6144,
    "window":   2048,           # local attention window = max seq length
    "dropout":  0.1,

    # ── Tokenizer (GPT-2 BPE, 50 257 tokens, no auth required) ─────────────
    "tokenizer_name": "gpt2",

    # ── Datasets (from multi_train_orchestrator; order = curriculum) ────────
    # Each entry: [dataset_name, steps]
    "stages": [
        ["tinystories",  2000],
        ["wikitext",     3000],
        ["openwebtext",  3000],
        ["alpaca",       1000],
    ],
    "max_items_train": 50_000,   # max examples materialized per stage
    "max_items_val":    2_000,
    "max_len":          2048,    # token sequence length (match "window" above)

    # ── Training hyper-parameters ─────────────────────────────────────────
    "batch_size":              1,    # per device (DDP: effective = 1×2×16 = 32)
    "gradient_accumulation":  16,
    "learning_rate":         3e-4,
    "weight_decay":          0.1,
    "max_grad_norm":         1.0,
    "warmup_steps":          200,
    "ul_alpha":              0.1,   # unlikelihood loss coefficient

    # ── Checkpointing ─────────────────────────────────────────────────────
    "save_steps":   200,            # save checkpoint every N *optimizer* steps
    "save_total_limit": 2,          # keep only the last 2 checkpoints on disk
}


# %% [markdown]
# ## Cell 4 — Model definition (copied from transformer.py)
# Defined at module level so spawned DDP processes can import it.

# %%
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as gradient_checkpoint


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = (x.pow(2).mean(-1, keepdim=True) + self.eps).sqrt()
        return self.weight * x / norm


class RoPE:
    @staticmethod
    def apply(x, seq_dim=2):
        dim = x.shape[-1] // 2
        sinusoid = RoPE._sinusoid(x.shape[seq_dim], dim, x.device)
        x1, x2 = x[..., ::2], x[..., 1::2]
        sin, cos = sinusoid[..., 0], sinusoid[..., 1]
        return torch.stack([x1*cos - x2*sin, x1*sin + x2*cos], dim=-1).flatten(-2)

    @staticmethod
    def _sinusoid(seq_len, dim, device):
        theta = 10000 ** (-torch.arange(0, dim, device=device) / dim)
        pos   = torch.arange(seq_len, device=device).unsqueeze(-1)
        rates = pos * theta
        return torch.stack([rates.sin(), rates.cos()], dim=-1)


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads, window, dropout=0.0):
        super().__init__()
        self.heads     = heads
        self.head_dim  = dim // heads
        self.window    = window
        self.dropout_p = dropout
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out = nn.Linear(dim, dim,     bias=False)

    def forward(self, x, attention_mask=None):
        B, T, C = x.shape
        H, D    = self.heads, self.head_dim
        qkv = self.qkv(x).view(B, T, H, 3*D).transpose(1, 2)
        q, k, v = qkv.chunk(3, dim=-1)
        q, k = RoPE.apply(q), RoPE.apply(k)

        mask = self._causal_local_mask(T, self.window, x.device)
        if attention_mask is not None:
            mask = mask & attention_mask[:, None, None, :].bool()

        dp = self.dropout_p if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dp)
        return self.out(out.transpose(1, 2).contiguous().view(B, T, C))

    @staticmethod
    def _causal_local_mask(T, W, device):
        m = torch.tril(torch.ones(T, T, device=device, dtype=torch.bool))
        return torch.triu(m, diagonal=-W)[None, None]


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        inner = int(hidden_dim * 2 / 3)          # SwiGLU scaling
        self.gate  = nn.Linear(dim, inner, bias=False)
        self.value = nn.Linear(dim, inner, bias=False)
        self.proj  = nn.Linear(inner, dim, bias=False)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.proj(F.silu(self.gate(x)) * self.value(x)))


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_dim, window, dropout=0.0):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn  = MultiHeadAttention(dim, heads, window, dropout)
        self.norm2 = RMSNorm(dim)
        self.ff    = FeedForward(dim, mlp_dim, dropout)

    def forward(self, x, attention_mask=None):
        x = x + self.attn(self.norm1(x), attention_mask=attention_mask)
        x = x + self.ff(self.norm2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, vocab_size, dim, depth, heads, mlp_dim, window,
                 dropout=0.0, tie_weights=True, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.max_seq_len    = window
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.drop      = nn.Dropout(dropout)
        self.blocks    = nn.ModuleList([
            TransformerBlock(dim, heads, mlp_dim, window, dropout)
            for _ in range(depth)
        ])
        self.norm      = RMSNorm(dim)
        self.to_logits = nn.Linear(dim, vocab_size, bias=False)
        if tie_weights:
            self.to_logits.weight = self.token_emb.weight
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, x, attention_mask=None):
        if x.dim() == 2 and x.dtype in (torch.long, torch.int64):
            x = self.token_emb(x)
        x = self.drop(x)
        for block in self.blocks:
            if self.use_checkpoint and self.training:
                x = gradient_checkpoint(block, x, attention_mask, use_reentrant=False)
            else:
                x = block(x, attention_mask=attention_mask)
        return self.to_logits(self.norm(x))

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# %% [markdown]
# ## Cell 5 — Training function (runs inside every DDP process)

# %%
def _train_fn():
    """
    Launched by accelerate.notebook_launcher with num_processes=2.
    Each call of this function is one independent GPU process.
    """
    import os, math, itertools
    from pathlib import Path

    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from torch.nn.utils.rnn import pad_sequence

    from accelerate import Accelerator
    from accelerate.utils import set_seed
    import bitsandbytes as bnb
    from datasets import load_dataset
    from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

    # ── Accelerator (handles device placement, mixed-precision, DDP) ────────
    accelerator = Accelerator(
        mixed_precision="fp16",                    # T4 → fp16 (not bf16)
        gradient_accumulation_steps=CFG["gradient_accumulation"],
        log_with=None,
    )
    set_seed(42)
    device = accelerator.device
    is_main = accelerator.is_main_process

    if is_main:
        Path(CFG["output_dir"]).mkdir(parents=True, exist_ok=True)
        print(f"[Main] device={device}  world_size={accelerator.num_processes}")

    # ── Tokenizer ────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(CFG["tokenizer_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id
    vocab_size = len(tokenizer)

    # ── Model ────────────────────────────────────────────────────────────────
    model = Transformer(
        vocab_size     = vocab_size,
        dim            = CFG["dim"],
        depth          = CFG["depth"],
        heads          = CFG["heads"],
        mlp_dim        = CFG["mlp_dim"],
        window         = CFG["window"],
        dropout        = CFG["dropout"],
        tie_weights    = True,
        use_checkpoint = True,          # gradient checkpointing ON
    )
    if is_main:
        n = model.count_params()
        print(f"[Main] Parameters: {n/1e6:.1f}M ({n/1e9:.3f}B)")

    # ── Optimizer (8-bit AdamW from bitsandbytes) ───────────────────────────
    # Reduces optimizer state memory ~8x vs fp32 Adam.
    # Embed / LayerNorm parameters get weight_decay=0 (standard practice).
    no_decay = {"bias", "weight"}   # RMSNorm .weight, any bias
    decay_params    = [p for n, p in model.named_parameters()
                       if not any(nd in n for nd in no_decay)]
    no_decay_params = [p for n, p in model.named_parameters()
                       if     any(nd in n for nd in no_decay)]
    optimizer = bnb.optim.AdamW8bit(
        [
            {"params": decay_params,    "weight_decay": CFG["weight_decay"]},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=CFG["learning_rate"],
    )

    # ── Dataset helpers ──────────────────────────────────────────────────────
    def get_stream(name):
        name = name.lower()
        if name == "tinystories":
            ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
            return ds, lambda ex: ex.get("text") or ""
        if name == "wikitext":
            ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
            return ds, lambda ex: ex.get("text") or ""
        if name == "openwebtext":
            ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
            return ds, lambda ex: ex.get("text") or ""
        if name == "alpaca":
            ds = load_dataset("yahma/alpaca-cleaned", split="train", streaming=True)
            def fmt(ex):
                ins, inp, out = ex.get("instruction",""), ex.get("input",""), ex.get("output","")
                return (f"### Instruction:\n{ins}\n\n### Input:\n{inp}\n\n### Response:\n{out}\n"
                        if inp else
                        f"### Instruction:\n{ins}\n\n### Response:\n{out}\n")
            return ds, fmt
        raise ValueError(f"Unknown dataset: {name}")

    class TextDataset(Dataset):
        def __init__(self, stream, get_text, max_len, max_items):
            self.samples = []
            for ex in stream:
                text = get_text(ex)
                ids  = tokenizer.encode(text, add_special_tokens=False)[:max_len]
                if ids:
                    self.samples.append(torch.tensor(ids, dtype=torch.long))
                if max_items and len(self.samples) >= max_items:
                    break

        def __len__(self):  return len(self.samples)
        def __getitem__(self, i): return self.samples[i]

    def collate(batch):
        ids  = pad_sequence(batch, batch_first=True, padding_value=pad_id)
        attn = (ids != pad_id).long()
        lbl  = ids.clone()
        lbl[ids == pad_id] = -100
        return {"input_ids": ids, "attention_mask": attn, "labels": lbl}

    class SliceLoader:
        def __init__(self, loader, max_batches):
            self.loader, self.max_batches = loader, max_batches
        def __iter__(self): return itertools.islice(iter(self.loader), self.max_batches)
        def __len__(self): return min(self.max_batches, len(self.loader))

    # ── Loss helpers ─────────────────────────────────────────────────────────
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def ul_loss(logits, input_ids, labels, alpha):
        """Unlikelihood: penalise predicting the immediately preceding token."""
        B, T, V = logits.shape
        if T < 2 or alpha == 0.0:
            return logits.new_tensor(0.0)
        probs  = torch.softmax(logits[:, 1:], dim=-1)
        p_prev = probs.gather(-1, input_ids[:, :-1].unsqueeze(-1)).squeeze(-1)
        valid  = (labels[:, 1:] != -100).float()
        return alpha * (-torch.log(1 - p_prev.clamp(max=1-1e-7)) * valid).sum() \
               / valid.sum().clamp(min=1)

    # ── Checkpoint helpers ───────────────────────────────────────────────────
    def save_ckpt(step, out_dir):
        if not is_main:
            return
        ckpt_dir = Path(out_dir) / f"checkpoint-{step}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        unwrapped = accelerator.unwrap_model(model)
        torch.save({
            "model":     unwrapped.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step":      step,
        }, ckpt_dir / "state.pt")

        # Enforce save_total_limit
        all_ckpts = sorted(
            [d for d in Path(out_dir).iterdir()
             if d.is_dir() and d.name.startswith("checkpoint-")],
            key=lambda d: int(d.name.split("-")[1]),
        )
        for old in all_ckpts[: -CFG["save_total_limit"]]:
            import shutil
            shutil.rmtree(old)
            print(f"[Checkpoint] Removed old: {old}")

        print(f"[Checkpoint] Saved → {ckpt_dir}")

    def load_last_ckpt(out_dir):
        ckpts = sorted(
            [d for d in Path(out_dir).iterdir()
             if d.is_dir() and d.name.startswith("checkpoint-")],
            key=lambda d: int(d.name.split("-")[1]),
        ) if Path(out_dir).is_dir() else []
        if not ckpts:
            return 0
        state = torch.load(ckpts[-1] / "state.pt", map_location="cpu")
        accelerator.unwrap_model(model).load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        step = state["step"]
        if is_main:
            print(f"[Checkpoint] Resumed from step {step} ({ckpts[-1]})")
        return step

    # ── Scheduler (total steps estimated from stages) ────────────────────────
    total_opt_steps = sum(
        math.ceil(s / CFG["gradient_accumulation"]) for _, s in CFG["stages"]
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps  = CFG["warmup_steps"],
        num_training_steps= total_opt_steps,
    )

    # ── Prepare model + optimizer with accelerate ────────────────────────────
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

    # Resume
    global_step = load_last_ckpt(CFG["output_dir"])

    # ── Multi-stage training loop ─────────────────────────────────────────────
    for stage_name, stage_steps in CFG["stages"]:
        if is_main:
            print(f"\n{'='*60}")
            print(f" Stage: {stage_name}  |  steps={stage_steps}")
            print(f"{'='*60}")

        train_stream, getter = get_stream(stage_name)
        val_stream,   _      = get_stream(stage_name)

        train_ds = TextDataset(train_stream, getter, CFG["max_len"], CFG["max_items_train"])
        val_ds   = TextDataset(val_stream,   getter, CFG["max_len"], CFG["max_items_val"])

        train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"],
                                  shuffle=True, collate_fn=collate, num_workers=0)
        val_loader   = DataLoader(val_ds,   batch_size=CFG["batch_size"],
                                  shuffle=False, collate_fn=collate, num_workers=0)

        train_loader, val_loader = accelerator.prepare(train_loader, val_loader)
        sliced = SliceLoader(train_loader, max_batches=stage_steps)

        # ── Training steps ───────────────────────────────────────────────────
        model.train()
        running_loss = 0.0
        for local_step, batch in enumerate(sliced, 1):
            ids   = batch["input_ids"]
            attn  = batch["attention_mask"]
            lbls  = batch["labels"]

            with accelerator.accumulate(model):
                logits = model(ids, attention_mask=attn)
                B, T, V = logits.shape
                loss = loss_fn(logits.reshape(B*T, V), lbls.reshape(B*T))
                loss = loss + ul_loss(logits, ids, lbls, CFG["ul_alpha"])
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), CFG["max_grad_norm"])

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            running_loss += loss.detach().float()

            if accelerator.sync_gradients:
                global_step += 1

                if global_step % 50 == 0 and is_main:
                    avg = running_loss.item() / 50
                    lr  = scheduler.get_last_lr()[0]
                    print(f"  step={global_step:6d}  loss={avg:.4f}  lr={lr:.2e}")
                    running_loss = 0.0

                if global_step % CFG["save_steps"] == 0:
                    accelerator.wait_for_everyone()
                    save_ckpt(global_step, CFG["output_dir"])

        # ── Validation ───────────────────────────────────────────────────────
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                ids, attn, lbls = batch["input_ids"], batch["attention_mask"], batch["labels"]
                logits = model(ids, attention_mask=attn)
                B, T, V = logits.shape
                l = loss_fn(logits.reshape(B*T, V), lbls.reshape(B*T))
                val_loss += l.item()
                pred = logits.argmax(-1)
                mask = lbls != -100
                correct += (pred[mask] == lbls[mask]).sum().item()
                total   += mask.sum().item()
        val_loss /= max(1, len(val_loader))
        acc = correct / max(1, total) * 100

        if is_main:
            print(f"  [Val] stage={stage_name}  loss={val_loss:.4f}  acc={acc:.2f}%")

        # Stage checkpoint
        accelerator.wait_for_everyone()
        save_ckpt(global_step, CFG["output_dir"])
        model.train()

    # ── Final save ────────────────────────────────────────────────────────────
    accelerator.wait_for_everyone()
    if is_main:
        final_path = Path(CFG["output_dir"]) / "final_model.pt"
        torch.save(accelerator.unwrap_model(model).state_dict(), final_path)
        print(f"\nTraining complete. Final weights → {final_path}")

    accelerator.end_training()


# %% [markdown]
# ## Cell 6 — Launch DDP (spawns 2 processes, one per T4)

# %%
from accelerate import notebook_launcher

notebook_launcher(
    _train_fn,
    num_processes=torch.cuda.device_count(),   # 2 on T4×2, 1 on single GPU
    mixed_precision="fp16",
)


# %% [markdown]
# ## Cell 7 — Quick generation test (run after training finishes)

# %%
def _load_and_generate(prompt: str, max_new_tokens: int = 100):
    from pathlib import Path
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(CFG["tokenizer_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = Transformer(
        vocab_size=len(tokenizer), dim=CFG["dim"], depth=CFG["depth"],
        heads=CFG["heads"], mlp_dim=CFG["mlp_dim"], window=CFG["window"],
        dropout=0.0, use_checkpoint=False,
    )
    weights = Path(CFG["output_dir"]) / "final_model.pt"
    model.load_state_dict(torch.load(weights, map_location="cpu"))
    model.eval()

    ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            logits = model(ids[:, -CFG["window"]:])[:, -1, :]
            next_id = torch.multinomial(torch.softmax(logits / 0.8, -1), 1)
            ids = torch.cat([ids, next_id], dim=1)
            if next_id.item() == tokenizer.eos_token_id:
                break
    return tokenizer.decode(ids[0], skip_special_tokens=True)

# print(_load_and_generate("Once upon a time"))   # uncomment to test
