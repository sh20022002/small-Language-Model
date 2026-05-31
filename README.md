# small-Language-Model

A custom small language model trained from scratch on Kaggle (2×Tesla T4, 16 GB each).  
The model is a causal decoder Transformer with GQA, SwiGLU, RoPE, and gradient checkpointing, trained via a 5-stage curriculum using DDP + GaLore + 8-bit AdamW.

---

## Architecture

| Component | Choice |
|---|---|
| Attention | Multi-head causal with **Grouped-Query Attention** (GQA) |
| Position encoding | **RoPE** (Rotary Position Embedding), precomputed buffers |
| Normalization | **RMSNorm** (pre-norm, every block) |
| Feed-forward | **SwiGLU** with `2/3 × mlp_dim` inner width |
| Memory | **Gradient checkpointing** (`use_reentrant=False`) |
| Window | Local sliding-window causal mask (`window` tokens) |
| Weight tying | Input embeddings ↔ output projection (LLaMA / GPT-2 style) |

Default size (~110 M parameters):

```python
MODEL_CFG = dict(
    dim     = 512,
    depth   = 8,
    heads   = 8,
    kv_heads = 2,   # GQA: 2 KV heads shared by 8 Q heads (~25% KV params)
    mlp_dim = 2048,
    window  = 2048,
    dropout = 0.1,
)
```

---

## Training

### Entry point

The primary training entry point is the Kaggle notebook:

```
kaggle_dual_gpu_finetune.ipynb
```

It runs `torchrun` for DDP across 2×T4 GPUs. The training loop uses:

- **Packed training** — token streams concatenated end-to-end, zero padding waste
- **GaLore** — gradient projection of large 2-D weight matrices (saves ~50% optimizer memory)
- **8-bit AdamW** (bitsandbytes) — ~8× smaller optimizer state
- **SDPA memory-efficient backend** — Flash-Attention-equivalent throughput on T4 (Turing)
- **Adaptive epochs** — each stage keeps training until validation accuracy meets a target and loss plateaus

### Curriculum stages

| Stage | Datasets | Goal |
|---|---|---|
| 1 | tinystories + bookcorpus | Simple sentences, narrative fluency |
| 2 | wikitext + c4 | Factual knowledge, clean web prose |
| 3 | openwebtext + c4 | Diverse web language |
| 4 | alpaca + dolly | Instruction following (SFT) |
| 5 | gsm8k + openorca | Chain-of-thought reasoning |

### Key hyper-parameters (Kaggle defaults)

```python
BATCH_SIZE    = 1          # per GPU; effective = 1×2 GPUs×8 accum = 16
GRAD_ACCUM    = 8
LR            = 2e-4
WEIGHT_DECAY  = 0.1
WARMUP_STEPS  = 300
USE_GALORE    = True
GALORE_RANK   = 128
MAX_GRAD_NORM = 1.0
```

---

## Repo layout

```
small-Language-Model/
├── kaggle_dual_gpu_finetune.ipynb   # primary training notebook (Kaggle 2×T4)
├── pyproject.toml
├── README.md
├── docs/design/
│   ├── dinoLM.md                    # future: DINO-style self-supervised LM
│   └── tellm.md                     # future: personalized-business chatbot
├── tests/
│   ├── conftest.py
│   ├── mfu.py
│   ├── semantic_eval.py
│   ├── test_model.py
│   └── test_training.py
└── src/my_slm/
    ├── __init__.py
    ├── transformer.py               # Transformer, GQA, RoPE, SwiGLU
    ├── train.py                     # training loop, GaLore optimizer, adaptive epochs
    ├── multi_train_orchestrator.py  # dataset loaders, PackedTokenDataset, curriculum
    ├── hybrid_tokeniztion.py        # HybridTokenizer (UTF-8 byte fallback)
    ├── benchmark_logger.py
    └── create_t_f.py
```

---

## Install (dev)

```bash
git clone https://github.com/sh20022002/small-Language-Model.git
cd small-Language-Model
pip install -e .
```

Required packages:

```bash
pip install torch accelerate bitsandbytes datasets transformers huggingface_hub galore-torch
```

---

## Programmatic usage

### Build and run the model

```python
from my_slm.transformer import Transformer

model = Transformer(
    vocab_size = 50257,
    dim        = 512,
    depth      = 8,
    heads      = 8,
    kv_heads   = 2,      # GQA (omit for standard MHA)
    mlp_dim    = 2048,
    window     = 2048,
    dropout    = 0.1,
    use_checkpoint = True,
)

# Forward (training)
logits = model(input_ids, attention_mask=mask)  # [B, T, vocab_size]

# Autoregressive generation
out = model.generate(
    input_ids,
    max_new_tokens    = 200,
    temperature       = 0.8,
    top_k             = 50,
    repetition_penalty = 1.3,
)
```

### Train with GaLore + 8-bit AdamW

```python
from my_slm.train import make_optimizer, train_model_accelerate

optimizer = make_optimizer(
    model,
    lr           = 2e-4,
    weight_decay = 0.1,
    betas        = (0.9, 0.95),
    use_8bit     = True,
    use_galore   = True,
    galore_rank  = 128,
    galore_update_proj_gap = 200,
)
```

### Multi-dataset curriculum training

```python
from my_slm.multi_train_orchestrator import train_across_datasets, StageConfig

stages = [
    StageConfig(name=["tinystories", "bookcorpus"], epochs=3, min_val_accuracy=28.0, max_epochs=12),
    StageConfig(name=["wikitext", "c4"],            epochs=3, min_val_accuracy=32.0, max_epochs=12),
    StageConfig(name=["alpaca", "dolly"],           epochs=3, min_val_accuracy=38.0, max_epochs=12),
]

train_across_datasets(
    tokenizer   = tokenizer,
    model       = model,
    stages      = stages,
    use_packed  = True,      # PackedTokenDataset (zero padding waste)
    output_dir  = "slm_run",
)
```

---

## Tests

```bash
pytest -q
```

---

## License

MIT
