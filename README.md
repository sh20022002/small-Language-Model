# 🧠 `my_slm` — tiny SLM + Hybrid Tokenizer (Colab‑friendly)

This project contains a **hybrid tokenizer** and a small Transformer training loop you can run in Colab.  
The tokenizer mixes classic subword ideas with a **UTF‑8 byte fallback** so it can encode *any* text, then exposes both **RLE** and **flat** output modes — perfect for demos and model training.  
The SLM bits include a minimal Transformer, dataloader, and training utilities.

> Tokenizer highlights (from the original design): encodes arbitrary Unicode, prefers short common words as base tokens, supports greedy merges, and falls back to raw bytes when needed. It can return **run‑length pairs** or a **flat id list**.


## Repo layout

```
your-repo/
├─ setup.py
├─ README.md
├─ notebooks/                        #  (not packaged)
├─ tests/                
   └─mfu.py                          # tests device flops
└─ src/
   └─ my_slm/                        # installable package
      ├─ __init__.py
      ├─ hybrid_tokeniztion.py       # HybridTokenizer
      ├─ transformer.py              # tiny Transformer
      ├─ train.py                    # dataloader + training loop
      ├─ multi_train_orchestrator.py # trains on datasets
      ├─ benchmark_logger.py         # tests model on benchmark and logs reoults
      └─ data/
         └─ tokenizer_state.pkl.gz   # copressed freezed tokenizer
```

---

## Installation

### From GitHub (non‑editable)
```bash
pip install "git+https://github.com/sh20022002/small-Language-Model.git@main"
```

### Editable dev install
```bash
git clone https://github.com/sh20022002/small-Language-Model/tree/main.git
pip install -e ./small-Language-Model
```

> Packaging uses a `src/` layout with the import package **`my_slm`**.


---

## Quickstart — Hybrid Tokenizer demo (keep this section in the README ✅)

```python
from my_slm.hybrid_tokeniztion import HybridTokenizer

# 1) Build the frequency DB
tok = HybridTokenizer()
tok.add_text("Hello world, welcome to tokenization.")
# or: tok.add_file("path/to/text.txt"); tok.add_files("data/**/*.txt")

# 2) Freeze vocab (choose how many base tokens and merges to keep)
tok.freeze_vocab(k_bases=5000, max_merges=10000)

# 3) Inspect DB status
tok.db_status(preview=10)

# 4) Encode/Decode in both modes
s = "Hello, world!\nשלום  🙂"
enc_rle  = tok.encode(s, mode="rle")   # -> list[(token_id, count)]
enc_flat = tok.encode(s, mode="flat")  # -> list[int]
assert tok.decode(enc_rle)  == s
assert tok.decode(enc_flat) == s

# 5) Explain / segment (nice for demos)
tok.top_merges(10)              # see learned merges
parts = tok.explain_token("welcome")   # recursive decomposition
seg   = tok.segment("Hello שלום")      # token kinds: base/merge/byte/sp/nl
print(parts, seg)
```

**Why flat vs RLE?**  
- Use **`mode="rle"`** for compact, human‑readable dumps and round‑trip tests.  
- Use **`mode="flat"`** for **model training** (IDs shape `[B, T]`).

> Design notes from the tokenizer source: RLE or flat outputs, greedy longest‑match encoding, UTF‑8 fallback, and vocabulary freezing before encode/decode.


---

## Train the tiny SLM

```python
import torch
from torch.optim import AdamW
from my_slm.transformer import Transformer
from my_slm.hybrid_tokeniztion import HybridTokenizer

# 1) Build + freeze tokenizer
tok = HybridTokenizer()
tok.add_files("data/**/*.txt")
tok.freeze_vocab(k_bases=5000, max_merges=10000)
pad_id = tok.token2id["<PAD>"]

# 2) Create model and sync vocab BEFORE moving to device
model = Transformer(...)
model.resize_token_embeddings(tok.vocab_size)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
opt = AdamW(model.parameters(), lr=3e-4)

# 3) Dataloader should yield:
#    - input_ids: LongTensor [B, T] (built with tok.encode(text, mode='flat'))
#    - attention_mask: [B, T] bool/long (ids!=pad_id)
#    - labels: [B, T] with pad positions == pad_id
#    CrossEntropyLoss(ignore_index=pad_id)

# 4) Forward
ids = ids.long().to(device)
mask = (ids != pad_id).to(device)
logits = model(ids, attention_mask=mask)  # [B, T, V]
```

**Training tips**
- Keep token IDs `int64`; you can run model compute in `bf16/fp16` via autocast.
- If you ever grow the tokenizer after model init, **resize** the embeddings **and** the output head to match the new vocab.
- Labels/padding: set `ignore_index = pad_id` so padding doesn’t contribute to loss.


---

## Colab usage

```python
# simple install
%pip install -q "git+https://github.com/sh20022002/small-Language-Model/tree/main.git@main"

# or live-edit workflow
!git clone https://github.com/sh20022002/small-Language-Model/tree/main.git
%pip install -e /content/small-Language-Model
```

Tips:
- Don’t put notebooks inside `src/` — keep them under `notebooks/`.
- If you don't see `setup.py` in Colab after a non‑editable install, that's expected (pip installs the built wheel).


---

## Tests
```bash
pytest -q
```

---

## Troubleshooting

- **Unexpected kwarg `attention_mask`** → ensure `Transformer.forward(self, x, attention_mask=None)` accepts (can ignore) the mask.
- **ModuleList has no forward** → iterate blocks: `for block in self.blocks: x = block(x)`.
- **Vocab/label mismatch** → align `tokenizer.vocab_size == embedding.num_embeddings == head.out_features` and set `ignore_index=pad_id`.
- **Device mismatch** → if you resize layers after `model.to(device)`, move to device again and recreate the optimizer.
- **Batch length mismatch** → make labels from the **padded ids** and flatten `(B*T)` for both logits and labels before `CrossEntropyLoss`.
- **Autograd warning adding loss** → use `loss.detach().item()` for running sums.

---

## License
MIT

