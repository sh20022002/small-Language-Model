# Architecture

System map for `small-Language-Model`: a byte-level BPE tokenizer
(`HybridTokenizer`) plus a small causal Transformer (GQA + RoPE + SwiGLU),
trained via a multi-stage curriculum on Kaggle (2×T4, DDP). This document is
maintained by the System Engineer role and should be updated whenever a
module is added, renamed, or its public surface changes.

## 1. Module structure

The installed package root is `src/my_slm/` (import name `my_slm`; the
**distribution** name in `pyproject.toml` is `hybrid-tokenizer` — a
historical mismatch, harmless for `pip install -e .` but worth knowing if
you ever see `pip show hybrid-tokenizer` and wonder where `my_slm` went).

```
src/my_slm/
├── __init__.py                  # public API surface — see §2
├── exceptions.py                # SLMException hierarchy (leaf: no my_slm imports)
├── transformer.py               # Transformer, RMSNorm, RoPE, MultiHeadAttention (leaf)
├── hybrid_tokeniztion.py        # HybridTokenizer — byte-level BPE (leaf, imports .exceptions)
├── utils.py                     # encode/decode/checkpoint helpers (leaf)
├── train.py                     # train_model / train_model_accelerate loops (leaf)
├── multi_train_orchestrator.py  # dataset loaders + curriculum → imports train, utils
├── benchmark_logger.py          # eval-and-log-to-CSV → imports multi_train_orchestrator
├── semantic_eval.py             # perplexity/BLiMP/LAMBADA suite → imports utils, transformer, hybrid_tokeniztion
├── mfu.py                       # GPU TFLOPS / MFU probe (leaf, no my_slm imports)
└── create_t_f.py                # wikidump2txt CLI — standalone script (leaf)
```

### Dependency graph (verified by AST scan — no cycles)

```
exceptions  transformer  utils  train  mfu  create_t_f      ← leaves, zero my_slm imports
    ↑
hybrid_tokeniztion  (imports .exceptions only)
    ↑            ↑                    ↑
    │            │                    │
    │      multi_train_orchestrator (imports train, utils)
    │            ↑
    │      benchmark_logger (imports multi_train_orchestrator)
    │
semantic_eval (imports utils, transformer, hybrid_tokeniztion)

__init__ (imports transformer, hybrid_tokeniztion, utils, exceptions)
```

Nothing in `multi_train_orchestrator.py`, `benchmark_logger.py`, or
`semantic_eval.py` is imported by anything "below" it — the graph is a
strict DAG. If you add a new module, keep it that way: a module may only
import from modules strictly above it in this list, or the cycle-detection
premise of this document is void and needs re-verifying (`git grep "^from my_slm\|^import my_slm"` per file, or re-run the AST scan below).

```bash
python -c "
import ast, pathlib
for f in sorted(pathlib.Path('src/my_slm').glob('*.py')):
    tree = ast.parse(f.read_text(encoding='utf-8'))
    imps = [('.'*n.level)+(n.module or '') for n in ast.walk(tree)
            if isinstance(n, ast.ImportFrom) and ((n.module or '').startswith('my_slm') or n.level)]
    print(f.stem, '->', imps)
"
```

### `__init__.py` — public API surface

```python
from my_slm import (
    Transformer, HybridTokenizer,
    encode, decode, get_pad_token_id, get_eos_token_id,
    save_checkpoint, load_checkpoint,
    # exception hierarchy (see §4)
    SLMException, TokenizerError, TokenizationError,
    TokenizerNotFrozenError, TokenizerFrozenError, VocabSizeError,
    EncodingError, DecodingError, TrainingError, DatasetError,
    CheckpointError, CheckpointNotFoundError, ConfigMismatchError,
)
```

Everything else (`train.py`, `multi_train_orchestrator.py`,
`benchmark_logger.py`, `semantic_eval.py`, `mfu.py`) is reached via explicit
submodule imports, e.g. `from my_slm.train import make_optimizer`. This is
deliberate: those modules pull in heavier or optional dependencies
(`matplotlib`, and lazily `datasets`/`transformers`/`accelerate`/
`bitsandbytes`/`galore-torch`/`safetensors`), so `import my_slm` itself stays
cheap and never fails on a minimal install.

## 2. Data flow

```
raw text corpus
   │  HybridTokenizer.add_text() / add_file()
   ▼
word-frequency table  →  freeze_vocab()  →  frozen BPE vocab + merge ranks
   │
   │  encode() (my_slm.utils, dispatches on hasattr(tok, "token2id") to pick
   │            HybridTokenizer vs HuggingFace tokenizer path)
   ▼
token ID stream
   │
   │  PackedTokenDataset (zero-padding packing) or TextTokenDataset (padded)
   │  — chosen by multi_train_orchestrator.train_across_datasets(use_packed=...)
   ▼
DataLoader batches  {input_ids, attention_mask, labels}
   │
   │  train_model() / train_model_accelerate()  (my_slm.train)
   │    - AMP (fp16/bf16 by compute capability) + GradScaler
   │    - repetition-unlikelihood auxiliary loss
   │    - accuracy-gated adaptive epochs (min_val_accuracy / max_epochs / plateau)
   ▼
trained Transformer weights
   │
   │  save_checkpoint() (my_slm.utils, safetensors-preferred) or the legacy
   │  {"config": {...}, "model_state": state_dict} torch.save format written
   │  by multi_train_orchestrator's per-stage `*_stage.pt` files
   ▼
checkpoint on disk (models/*.pt or a checkpoint/ directory)
   │
   ├─ load_checkpoint() / load_model_safely() (my_slm.utils)     → resume training
   ├─ load_latest_checkpoint() (my_slm.train)                    → curriculum warm-start
   ├─ load_model_and_tok() (my_slm.semantic_eval)                → evaluation suite
   └─ Transformer.generate()                                     → inference
```

Curriculum stages are declared as a list of `StageConfig` (dataclass in
`multi_train_orchestrator.py`) and run in order by `train_across_datasets()`.
Each stage independently picks the DDP path (`accelerator is not None` →
`train_model_accelerate`) or the single-GPU path (`train_model`), and a
failure in one stage (bad dataset, `Exception` during training) is logged
and skipped — the curriculum continues — **except** `TypeError`/
`AttributeError`/`NameError`, which are treated as programming bugs and
re-raised instead of silently skipped.

## 3. Configuration

| Surface | Where | Notes |
|---|---|---|
| Model architecture | `Transformer(vocab_size, dim, depth, heads, mlp_dim, window, kv_heads=None, ...)` | `kv_heads` enables GQA; omit for standard MHA |
| Curriculum | `StageConfig` (name/epochs/steps/min_val_accuracy/max_epochs/plateau_*) | list passed to `train_across_datasets(stages=...)` |
| Optimizer | `make_optimizer(model, lr, use_8bit, use_galore, ...)` | GaLore/8-bit are lazy-imported, fail loudly with an actionable `ImportError` if missing |
| Checkpoint schema | `CHECKPOINT_SCHEMA_VERSION` in `utils.py` (currently `1`) | `save_checkpoint`/`load_checkpoint` write/read `config.json` + `model.safetensors` (or `.pt` fallback) + `trainer_state.json` + `optimizer.pt` |
| Benchmark run | `BenchConfig` (dataclass in `benchmark_logger.py`) | appends one row per run to a CSV (`log_csv`) |
| Package dependencies | `pyproject.toml [project.dependencies]` | **hard floor**: `torch>=2.6.0` (CVE-2025-32434 — see `SECURITY.md`), `numpy>=1.21`, `tqdm>=4.60`, `matplotlib>=3.5`. `requires-python = ">=3.9"` is pinned to match — torch 2.6 dropped 3.8 support, so these two must move together. |
| CI / lint / packaging | `.github/workflows/ci.yml`, `Makefile`, `.pre-commit-config.yaml` | `make ci` runs the same steps as the CI job locally |

Anything imported lazily inside a function body (`datasets`, `transformers`,
`accelerate`, `bitsandbytes`, `galore-torch`, `safetensors`) is an *optional*
dependency: the module that needs it still imports cleanly without it, and
the feature that uses it raises a clear `ImportError` with a `pip install`
hint at call time, not at import time.

## 4. Exception hierarchy (`my_slm/exceptions.py`)

```
SLMException
├── TokenizerError
│   ├── TokenizerNotFrozenError(TokenizerError, RuntimeError)   ← dual base, see below
│   ├── TokenizerFrozenError(TokenizerError, RuntimeError)      ← dual base, see below
│   ├── VocabSizeError
│   └── TokenizationError
│       ├── EncodingError
│       └── DecodingError
└── TrainingError
    ├── DatasetError
    └── CheckpointError
        ├── CheckpointNotFoundError
        └── ConfigMismatchError
```

`TokenizerNotFrozenError` and `TokenizerFrozenError` deliberately inherit
**both** `TokenizerError` and `RuntimeError`. Before this hierarchy existed,
`HybridTokenizer.encode()`/`.segment()`/`.add_text()` raised a plain
`RuntimeError` for "not frozen yet" / "already frozen" — any `except
RuntimeError:` written against the old API still works unchanged, while new
code can catch the specific subclass. Do not narrow this to single
inheritance without checking for external `except RuntimeError` call sites
first (see §5).

## 5. Backward compatibility

- **Checkpoint formats**: `semantic_eval.load_model_and_tok()` and
  `utils.load_model_safely()` accept three shapes — the current
  `{"config": {...}, "model_state": state_dict}` format, a legacy variant
  with an *empty* `config` dict (architecture inferred from tensor shapes,
  with a printed warning), and a bare `state_dict` with no wrapper at all.
  Keep all three paths working when touching checkpoint I/O.
- **`torch.load` safety**: every load site tries `weights_only=True` first.
  `train.load_latest_checkpoint._load_state` intentionally does **not**
  fall back to `weights_only=False` on failure — that fallback was closed
  deliberately (CVE-2025-32434; see `SECURITY.md`) because checkpoints here
  may originate from third-party Kaggle datasets. If you're touching
  checkpoint loading, preserve "fail loudly" over "silently deserialize
  untrusted pickle."
- **Tokenizer helper aliases**: `_encode`/`_decode`/`_get_pad_id`/`_get_eos_id`
  module-level aliases in `multi_train_orchestrator.py` and `semantic_eval.py`
  point at the unified `my_slm.utils` functions. `benchmark_logger.py`
  imports `_get_pad_id`/`_encode` *from* `multi_train_orchestrator`, not
  from `utils` directly — if you rename the aliases, update both call sites.
- **Exception types**: see §4.

## 6. Extension points

- **New dataset for curriculum training**: add a branch to
  `get_hf_stream_and_text_getter()` in `multi_train_orchestrator.py`
  (dataset name → `(streaming HF dataset, text-getter fn)`), and — if it has
  a validation/test split under a different name than usual — add it to
  `_HAS_VAL_SPLIT` or `_HAS_TEST_SPLIT`.
- **New tokenizer backend**: `my_slm.utils.encode/decode/get_pad_token_id/
  get_eos_token_id` dispatch on `hasattr(tokenizer, "token2id")` to choose
  the `HybridTokenizer` path vs. the HuggingFace-tokenizer path. A third
  backend needs a third branch in each of those four functions (single
  source of truth — don't reimplement encode/decode elsewhere).
- **New error type**: subclass the closest existing branch in §4, not
  `SLMException` directly, unless it's a genuinely new top-level category.
  Export it from `my_slm/__init__.py`'s `__all__` if it's part of the
  public API (i.e. raised by something already exported there).
- **New checkpoint field**: bump `CHECKPOINT_SCHEMA_VERSION` in `utils.py`
  only for breaking changes to the on-disk layout; purely additive fields
  in `config.json`/`trainer_state.json` don't need a bump since both are
  loaded as permissive dicts.
- **GQA / attention variants**: `MultiHeadAttention(dim, heads, window,
  kv_heads=...)` — `kv_heads < heads` enables grouped-query attention via
  `repeat_interleave`; `kv_heads is None` falls back to standard MHA.

## 7. Known inconsistencies (tracked, not yet resolved)

- Distribution name `hybrid-tokenizer` (pyproject.toml) vs. import name
  `my_slm` — cosmetic, not a functional bug, but confusing on `pip show`.
- `create_t_f.py`'s module docstring still calls itself `wikidump2txt.py`
  (its name before a rename); behavior is correct, only the docstring
  header is stale.
