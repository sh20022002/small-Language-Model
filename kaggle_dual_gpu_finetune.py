# =============================================================================
# Custom SLM — Kaggle 2x Tesla T4 (2x16 GB VRAM) — Train from Scratch
#
# Kaggle Settings (right-hand panel) before running:
#   Accelerator  → GPU T4 x2
#   Internet     → On
#
# Memory budget per GPU (16 GB each, total 32 GB):
#   Model weights  fp16   ~3 GB for 1.3 B params
#   Gradients      fp16   ~3 GB
#   8-bit AdamW           ~0.75 GB   (vs ~6 GB for fp32 Adam)
#   Activations  (w/ GC)  ~2–3 GB
#   ─────────────────────────────
#   Estimated total        ~9–10 GB / GPU  → safe on T4
#
# Techniques used (all from the project's own src/ package):
#   • DDP via accelerate.notebook_launcher     (2 GPUs, equal utilisation)
#   • bitsandbytes 8-bit AdamW                 (make_optimizer use_8bit=True)
#   • float16 AMP                              (T4 has fast fp16, no bf16)
#   • Gradient checkpointing                   (transformer use_checkpoint=True)
#   • Gradient accumulation × 16              (effective batch = 32)
#   • Depth-scaled residual init               (transformer._init_weights)
#   • RoPE with precomputed buffers            (MultiHeadAttention._apply_rope)
#   • SwiGLU feed-forward                      (FeedForward)
#   • Cosine LR schedule with linear warmup    (train.get_cosine_schedule_with_warmup)
#   • Unlikelihood loss                         (train._repetition_ul_loss)
#   • Step-based checkpointing                 (auto-resume across sessions)
# =============================================================================

# %% [markdown]
# ## Cell 1 — Install dependencies
#
# Run this cell once, then restart the kernel.

# %%
# !pip install -q -U accelerate bitsandbytes datasets transformers
# # Install the project package from GitHub (or set REPO_ROOT below instead):
# !pip install -q git+https://github.com/sh20022002/small-Language-Model.git


# %% [markdown]
# ## Cell 2 — sys.path setup + GPU + compute benchmark

# %%
import sys
from pathlib import Path

# ── If you uploaded the repo as a Kaggle dataset instead of pip-installing: ──
# Set REPO_ROOT to the dataset path, e.g. "/kaggle/input/small-language-model"
# Leave None when installed via pip.
REPO_ROOT = None    # e.g.  "/kaggle/input/small-language-model"

if REPO_ROOT:
    for sub in ("src", "tests"):
        p = str(Path(REPO_ROOT) / sub)
        if p not in sys.path:
            sys.path.insert(0, p)

import torch

n_gpus = torch.cuda.device_count()
print(f"GPUs available: {n_gpus}")
for i in range(n_gpus):
    p = torch.cuda.get_device_properties(i)
    print(f"  cuda:{i}  {p.name}  {p.total_memory / 1e9:.1f} GB")

if n_gpus < 2:
    print("\nWARNING: fewer than 2 GPUs found — "
          "enable 'GPU T4 x2' in Kaggle Settings → Accelerator.")

# Compute benchmark (fp32 TFLOPS + MFU) — uses tests/mfu.py from the project
try:
    from mfu import run_perf_and_mfu
    if n_gpus >= 1:
        run_perf_and_mfu()
except ImportError:
    print("[mfu] Skipped — set REPO_ROOT above to enable, or pip-install the package.")


# %% [markdown]
# ## Cell 3 — Configuration
# **Edit this cell only.** Everything else adapts automatically.

# %%
# ── Model configuration ────────────────────────────────────────────────────────
# Estimated parameter counts (GPT-2 tokenizer, vocab=50 257):
#   tiny   : dim=256,  depth=4,  heads=4,  mlp=1024   → ~30 M   (quick smoke test)
#   small  : dim=512,  depth=8,  heads=8,  mlp=2048   → ~110 M
#   medium : dim=768,  depth=12, heads=12, mlp=3072   → ~250 M
#   large  : dim=1024, depth=16, heads=16, mlp=4096   → ~650 M
#   xlarge : dim=1536, depth=24, heads=24, mlp=6144   → ~1.3 B  ← default
MODEL_CFG = dict(
    dim     = 1536,
    depth   = 24,
    heads   = 24,
    mlp_dim = 6144,
    window  = 2048,     # local-attention window = max sequence length
    dropout = 0.1,
)

# ── Tokenizer ──────────────────────────────────────────────────────────────────
# A) Load a saved HybridTokenizer (from a previous Colab run):
#      TOKENIZER_PATH = "/kaggle/input/my-tokenizer/tokenizer.pkl.gz"
# B) Build HybridTokenizer on-the-fly from TinyStories (~2 min):
#      BUILD_HYBRID = True
# C) Default — GPT-2 BPE (50 257 tokens, no auth, instant):
TOKENIZER_PATH = None       # str path to .pkl.gz, or None
BUILD_HYBRID   = False      # True = build from TinyStories at startup
HF_TOKENIZER   = "gpt2"    # used only when A and B are both disabled

# ── Training stages (curriculum order) ────────────────────────────────────────
# [dataset_name, steps]  —  names: tinystories | wikitext | openwebtext | alpaca
STAGES = [
    ["tinystories",  2000],
    ["wikitext",     3000],
    ["openwebtext",  3000],
    ["alpaca",       1000],
]
MAX_ITEMS_TRAIN = 50_000
MAX_ITEMS_VAL   =  2_000

# ── Training hyper-parameters ──────────────────────────────────────────────────
BATCH_SIZE    = 1       # per GPU; effective = 1 × 2 GPUs × 16 accum = 32
GRAD_ACCUM    = 16
LR            = 3e-4
WEIGHT_DECAY  = 0.1
MAX_GRAD_NORM = 1.0
WARMUP_STEPS  = 200
UL_ALPHA      = 0.1     # unlikelihood-loss coefficient

# ── Checkpointing ──────────────────────────────────────────────────────────────
OUTPUT_DIR       = "/kaggle/working/slm_run"
SAVE_STEPS       = 200   # save every N *optimizer* steps
SAVE_TOTAL_LIMIT = 2     # keep only the last 2 checkpoints


# %% [markdown]
# ## Cell 4 — Training function  (executed once per GPU via DDP)

# %%
def _train_fn():
    """
    Launched by notebook_launcher with num_processes=2.
    Each GPU process runs this function independently.
    All imports are local to avoid multiprocessing pickling issues.
    """
    import math, shutil
    from pathlib import Path

    import torch
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    from torch.utils.data import DataLoader

    # ── Project imports ───────────────────────────────────────────────────────
    from my_slm.transformer import Transformer
    from my_slm.train import (
        make_optimizer,
        get_cosine_schedule_with_warmup,
        train_model_accelerate,
    )
    from my_slm.multi_train_orchestrator import (
        get_hf_stream_and_text_getter,
        TextTokenDataset,
        make_collate,
        _get_pad_id,
    )
    from my_slm.hybrid_tokeniztion import HybridTokenizer

    # ── Accelerator ───────────────────────────────────────────────────────────
    accelerator = Accelerator(
        mixed_precision            = "fp16",   # T4: fp16 only (no bf16)
        gradient_accumulation_steps= GRAD_ACCUM,
    )
    set_seed(42)
    is_main = accelerator.is_main_process

    if is_main:
        Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        print(f"[Accelerator] world={accelerator.num_processes}  "
              f"device={accelerator.device}  fp16=True  grad_accum={GRAD_ACCUM}")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    if TOKENIZER_PATH and Path(TOKENIZER_PATH).exists():
        tokenizer = HybridTokenizer.load(TOKENIZER_PATH)
        if is_main:
            print(f"[Tokenizer] HybridTokenizer loaded — vocab={tokenizer.vocab_size}")

    elif BUILD_HYBRID:
        from datasets import load_dataset
        tokenizer = HybridTokenizer()
        if is_main:
            print("[Tokenizer] Building HybridTokenizer from TinyStories …")
            stream = load_dataset("roneneldan/TinyStories",
                                  split="train", streaming=True)
            for i, ex in enumerate(stream):
                tokenizer.add_text(ex.get("text", ""))
                if i >= 20_000:
                    break
            tokenizer.freeze_vocab(k_bases=2000, max_merges=20_000)
            tok_out = Path(OUTPUT_DIR) / "tokenizer.pkl.gz"
            tokenizer.save(tok_out)
            print(f"[Tokenizer] Built — vocab={tokenizer.vocab_size}  → {tok_out}")
        accelerator.wait_for_everyone()
        if not is_main:
            tokenizer = HybridTokenizer.load(Path(OUTPUT_DIR) / "tokenizer.pkl.gz")

    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(HF_TOKENIZER)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if is_main:
            print(f"[Tokenizer] GPT-2 BPE — vocab={len(tokenizer)}")

    vocab_size = (tokenizer.vocab_size
                  if hasattr(tokenizer, "vocab_size") else len(tokenizer))
    pad_id = _get_pad_id(tokenizer)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = Transformer(
        vocab_size     = vocab_size,
        use_checkpoint = True,          # gradient checkpointing enabled
        **MODEL_CFG,
    )
    if is_main:
        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[Model] {n/1e6:.1f} M params ({n/1e9:.3f} B)  "
              f"dim={MODEL_CFG['dim']}  depth={MODEL_CFG['depth']}  "
              f"heads={MODEL_CFG['heads']}")

    # ── Architecture sanity check (pre-training, main process only) ───────────
    if is_main:
        try:
            # tests/test_model.py — notebook-callable entry point
            from test_model import check_model_architecture
            ok = check_model_architecture(model, vocab_size, device="cpu")
            if not ok:
                raise RuntimeError("Architecture checks failed — fix before training!")
        except ImportError:
            print("[check] test_model.py not found — set REPO_ROOT to enable.")

    # ── Optimizer (8-bit AdamW from bitsandbytes) ─────────────────────────────
    optimizer = make_optimizer(
        model,
        lr           = LR,
        weight_decay = WEIGHT_DECAY,
        betas        = (0.9, 0.95),
        use_8bit     = True,            # ~8× smaller optimizer state on GPU
    )

    # ── LR scheduler ─────────────────────────────────────────────────────────
    total_opt_steps = sum(
        math.ceil(steps / GRAD_ACCUM) for _, steps in STAGES
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        warmup_steps = WARMUP_STEPS,
        total_steps  = total_opt_steps,
    )

    # ── Prepare with Accelerator ──────────────────────────────────────────────
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

    # ── Checkpoint helpers ────────────────────────────────────────────────────
    def _save(step: int):
        if not is_main:
            return
        ckpt_dir = Path(OUTPUT_DIR) / f"checkpoint-{step}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            "config": {**MODEL_CFG, "vocab_size": vocab_size},
            "model_state": accelerator.unwrap_model(model).state_dict(),
            "optimizer":   optimizer.state_dict(),
            "step":        step,
        }, ckpt_dir / "state.pt")
        # Enforce SAVE_TOTAL_LIMIT
        ckpts = sorted(
            [d for d in Path(OUTPUT_DIR).iterdir()
             if d.is_dir() and d.name.startswith("checkpoint-")],
            key=lambda d: int(d.name.split("-")[1]),
        )
        for old in ckpts[:-SAVE_TOTAL_LIMIT]:
            shutil.rmtree(old)
        print(f"[Checkpoint] step={step}  → {ckpt_dir}")

    def _resume() -> int:
        ckpts = sorted(
            [d for d in Path(OUTPUT_DIR).iterdir()
             if d.is_dir() and d.name.startswith("checkpoint-")],
            key=lambda d: int(d.name.split("-")[1]),
        ) if Path(OUTPUT_DIR).is_dir() else []
        if not ckpts:
            return 0
        state = torch.load(ckpts[-1] / "state.pt", map_location="cpu")
        accelerator.unwrap_model(model).load_state_dict(state["model_state"],
                                                        strict=False)
        optimizer.load_state_dict(state["optimizer"])
        if is_main:
            print(f"[Checkpoint] Resumed from step {state['step']}  ({ckpts[-1].name})")
        return state["step"]

    global_step = _resume()

    # ── Multi-stage training loop ─────────────────────────────────────────────
    collate = make_collate(pad_id=pad_id, ignore_index=-100)

    for stage_name, stage_steps in STAGES:
        if is_main:
            print(f"\n{'='*60}\n  Stage: {stage_name}  |  steps={stage_steps}\n{'='*60}")

        train_stream, getter = get_hf_stream_and_text_getter(stage_name)
        val_stream,   _      = get_hf_stream_and_text_getter(stage_name)

        train_ds = TextTokenDataset(
            train_stream, getter, tokenizer,
            MODEL_CFG["window"], MAX_ITEMS_TRAIN,
        )
        val_ds = TextTokenDataset(
            val_stream, getter, tokenizer,
            MODEL_CFG["window"], MAX_ITEMS_VAL,
        )

        train_loader = DataLoader(
            train_ds, batch_size=BATCH_SIZE, shuffle=True,
            collate_fn=collate, num_workers=0,
        )
        val_loader = DataLoader(
            val_ds, batch_size=BATCH_SIZE, shuffle=False,
            collate_fn=collate, num_workers=0,
        )
        train_loader, val_loader = accelerator.prepare(train_loader, val_loader)

        # Limit to stage_steps batches (step-based curriculum)
        from my_slm.multi_train_orchestrator import SliceLoader
        sliced = SliceLoader(train_loader, max_batches=stage_steps)

        model = train_model_accelerate(
            model         = model,
            train_loader  = sliced,
            val_loader    = val_loader,
            optimizer     = optimizer,
            accelerator   = accelerator,
            epochs        = 1,
            max_grad_norm = MAX_GRAD_NORM,
            scheduler     = scheduler,
            ul_alpha      = UL_ALPHA,
        )

        global_step += math.ceil(stage_steps / GRAD_ACCUM)
        accelerator.wait_for_everyone()
        _save(global_step)

    # ── Final weights ─────────────────────────────────────────────────────────
    accelerator.wait_for_everyone()
    if is_main:
        final_path = Path(OUTPUT_DIR) / "final_model.pt"
        torch.save({
            "config": {**MODEL_CFG, "vocab_size": vocab_size},
            "model_state": accelerator.unwrap_model(model).state_dict(),
        }, final_path)
        print(f"\n[Done] Final weights → {final_path}")

    accelerator.end_training()


# %% [markdown]
# ## Cell 5 — Launch DDP (2 processes, one per T4)

# %%
from accelerate import notebook_launcher

notebook_launcher(
    _train_fn,
    num_processes  = torch.cuda.device_count(),   # 2 on T4×2, 1 on single GPU
    mixed_precision= "fp16",
)


# %% [markdown]
# ## Cell 6 — Post-training checks + semantic evaluation
#
# Run after Cell 5 completes.  Requires REPO_ROOT set in Cell 2.

# %%
def _post_training_eval(quick: bool = True):
    """
    1. Architecture + training behaviour checks (test_model, test_training).
    2. Semantic benchmarks (perplexity, top-k, BLiMP, LAMBADA, analogy).
    """
    from pathlib import Path
    import torch

    final_path = Path(OUTPUT_DIR) / "final_model.pt"
    if not final_path.exists():
        print(f"[Eval] {final_path} not found — run Cell 5 first.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model + tokenizer via semantic_eval helper
    try:
        from semantic_eval import load_model_and_tok, run_all, print_report
    except ImportError:
        print("[Eval] semantic_eval.py not found — set REPO_ROOT in Cell 2.")
        return

    tok_src = TOKENIZER_PATH or HF_TOKENIZER
    model, tok = load_model_and_tok(str(final_path), tok_src, device)

    # ── Architecture checks ───────────────────────────────────────────────────
    try:
        from test_model import check_model_architecture
        check_model_architecture(model, model.token_emb.num_embeddings, device)
    except ImportError:
        pass

    # ── Post-training behaviour checks ────────────────────────────────────────
    try:
        from test_training import check_trained_model
        from my_slm.multi_train_orchestrator import _get_pad_id
        check_trained_model(model, tok, device,
                            vocab_size=model.token_emb.num_embeddings,
                            pad_id=_get_pad_id(tok))
    except ImportError:
        pass

    # ── Semantic benchmarks ───────────────────────────────────────────────────
    report = run_all(model, tok, device, quick=quick)
    print_report(report)
    return report


# Uncomment to run:
# _post_training_eval(quick=True)


# %% [markdown]
# ## Cell 7 — Generation test

# %%
def _generate(prompt: str, max_new_tokens: int = 120, temperature: float = 0.8):
    from pathlib import Path
    import torch
    from my_slm.transformer import Transformer
    from my_slm.multi_train_orchestrator import _encode, _get_pad_id

    final_path = Path(OUTPUT_DIR) / "final_model.pt"
    ckpt = torch.load(final_path, map_location="cpu")

    cfg        = ckpt["config"]
    state_dict = ckpt["model_state"]

    if TOKENIZER_PATH and Path(TOKENIZER_PATH).exists():
        from my_slm.hybrid_tokeniztion import HybridTokenizer
        tok = HybridTokenizer.load(TOKENIZER_PATH)
    else:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(HF_TOKENIZER)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

    model = Transformer(**{k: cfg[k] for k in
                           ("vocab_size", "dim", "depth", "heads", "mlp_dim", "window")},
                        dropout=0.0, use_checkpoint=False)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    eos_id = _get_pad_id(tok)
    ids    = torch.tensor([_encode(tok, prompt)], dtype=torch.long)

    with torch.inference_mode():
        out = model.generate(
            ids,
            max_new_tokens   = max_new_tokens,
            temperature      = temperature,
            top_k            = 50,
            suppress_ids     = [eos_id],
            repetition_penalty = 1.3,
        )

    gen_ids = out[0, len(_encode(tok, prompt)):].tolist()
    if hasattr(tok, "token2id"):
        return tok.decode(gen_ids)
    return tok.decode(gen_ids, skip_special_tokens=True)


# print(_generate("Once upon a time"))   # uncomment to test
