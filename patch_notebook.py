"""Patch kaggle_dual_gpu_finetune.ipynb with all fixes."""
import json

with open("kaggle_dual_gpu_finetune.ipynb", encoding="utf-8") as f:
    nb = json.load(f)

# ── Cell 7 (index 6): Configuration ──────────────────────────────────────────
new_cell7 = """\
# ── Model configuration ────────────────────────────────────────────────────────
# Estimated parameter counts (GPT-2 tokenizer, vocab=50 257):
#   tiny   : dim=256,  depth=4,  heads=4,  mlp=1024   →  ~30 M
#   small  : dim=512,  depth=8,  heads=8,  mlp=2048   → ~110 M  ← default
#   medium : dim=768,  depth=12, heads=12, mlp=3072   → ~250 M
#   large  : dim=1024, depth=16, heads=16, mlp=4096   → ~650 M
MODEL_CFG = dict(
    dim     = 512,
    depth   = 8,
    heads   = 8,
    mlp_dim = 2048,
    window  = 2048,     # local-attention window = max sequence length
    dropout = 0.1,
)

# ── Tokenizer ──────────────────────────────────────────────────────────────────
TOKENIZER_PATH = None       # str path to .pkl.gz, or None
BUILD_HYBRID   = False      # True = build HybridTokenizer from TinyStories
HF_TOKENIZER   = "gpt2"    # used only when TOKENIZER_PATH and BUILD_HYBRID are both off

# ── Training stages — curriculum order ────────────────────────────────────────
# [dataset_name, steps, epochs]
#   steps > 0  → train exactly that many batches (recommended — predictable time)
#   steps == 0 → train for `epochs` full passes over MAX_ITEMS_TRAIN samples
#
# Curriculum:
#   1. TinyStories   — simple language patterns, sentence structure
#   2. WikiText      — encyclopedic factual knowledge
#   3. OpenWebText   — diverse web prose
#   4. C4            — large-scale web (extra diversity & volume)
#   5. Alpaca        — instruction following (SFT)
#   6. Dolly         — instruction variety (SFT)
#   7. GSM8K         — grade-school math with step-by-step solutions (chain-of-thought)
STAGES = [
    ["tinystories",  2000, 0],
    ["wikitext",     3000, 0],
    ["openwebtext",  3000, 0],
    ["c4",           3000, 0],
    ["alpaca",       1500, 0],
    ["dolly",        1000, 0],
    ["gsm8k",        1000, 0],
]
MAX_ITEMS_TRAIN = 20_000   # items materialised per stage (step-based: only ~steps used)
MAX_ITEMS_VAL   =  2_000

# ── Training hyper-parameters ──────────────────────────────────────────────────
BATCH_SIZE    = 1       # per GPU; effective = 1 × 2 GPUs × 8 accum = 16
GRAD_ACCUM    = 8
LR            = 2e-4   # slightly conservative — prevents NaN on fresh random init
WEIGHT_DECAY  = 0.1
MAX_GRAD_NORM = 1.0
WARMUP_STEPS  = 300    # ~18 % of total optimizer steps
UL_ALPHA      = 0.1    # unlikelihood-loss weight (anti-repetition)

# ── Checkpointing ──────────────────────────────────────────────────────────────
OUTPUT_DIR       = "/kaggle/working/slm_run"
SAVE_STEPS       = 500
SAVE_TOTAL_LIMIT = 3
"""

nb["cells"][6]["source"] = [new_cell7]

# ── Write back ────────────────────────────────────────────────────────────────
with open("kaggle_dual_gpu_finetune.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Patched cell 7 (config) OK")
