# 📦 `HybridTokenizer` – Tokenization Strategy Overview

This tokenizer combines classic subword tokenization with UTF-8 byte fallback. It is designed to:

- Encode arbitrary Unicode text
- Prefer short, common words (≤ 4 UTF-8 bytes) as base tokens
- Support recursive merges (light BPE-style)
- Fall back to raw byte-level encoding when no match is found

---
### init

```python
tok = HybridTokenizer()
```
###  Build Token Frequency DB

```python
tok.add_text("Hello world, welcome to tokenization.")
```
or 

```python
tok.add_file(filename)
```
### freeze the vocabelaery
```python
tok.freeze_vocab(k_bases=5000, max_merges=10000)
```

### database report of the tokeized words
```python
tok.db_status(preview=10)
```
### resoult of report

<p align="center">
  <img src="scrshots\Screenshot 2025-08-04 180616.png" alt="vocab_status" width="800">
</p>

| Feature    | Description                                                 |
| ---------- | ----------------------------------------------------------- |
| Base vocab | Top `k` most frequent short words (≤ 4 bytes)               |
| Merging    | Greedy 2-part merges to compress longer words               |
| Fallback   | Raw UTF-8 bytes as `byte_NN` tokens                         |
| Output     | RLE-compressed list of `(token_id, count)`                  |
| Robustness | Can encode *any* valid Unicode text, regardless of language |

## usage
### encode
<p align="center">
  <img src="scrshots\Screenshot 2025-08-04 180534.png" alt="code" width="800">
</p>

### encoded to tokens
<p align="center">
  <img src="scrshots\Screenshot 2025-08-04 180548.png" alt="encoded" width="800">
</p>

### decoded back
<p align="center">
  <img src="scrshots\Screenshot 2025-08-04 180559.png" alt="decoded" width="800">
</p>