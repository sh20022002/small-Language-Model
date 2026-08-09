"""
Byte-level BPE tokenizer (GPT-2 / LLaMA style).

Training API:
    tok = HybridTokenizer()
    tok.add_text(corpus)          # accumulate word frequencies
    tok.freeze_vocab(32_000)      # run BPE merges, build vocab
    tok.save("tok.pkl.gz")

Inference API:
    tok = HybridTokenizer.load("tok.pkl.gz")
    ids = tok.encode("Hello world")
    text = tok.decode(ids)
"""

from __future__ import annotations

import gzip
import pickle
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Byte ↔ Unicode bijection (GPT-2 §3.2)
# Maps every byte 0-255 to a unique printable Unicode character so that
# merged token strings can be decoded unambiguously char-by-char.
# ---------------------------------------------------------------------------

def _build_byte_unicode() -> Tuple[Dict[int, str], Dict[str, int]]:
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = list(bs)
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    b2u = {b: chr(c) for b, c in zip(bs, cs)}
    u2b = {v: k for k, v in b2u.items()}
    return b2u, u2b

_BYTE2CHAR, _CHAR2BYTE = _build_byte_unicode()

# GPT-2 pre-tokenization regex
_PRETOK = re.compile(
    r"'s|'t|'re|'ve|'m|'ll|'d"
    r"| ?\w+"
    r"| ?[^\s\w]+"
    r"|\s+",
    re.UNICODE,
)

_SPECIAL = ["<PAD>", "<BOS>", "<EOS>", "<UNK>", "<MASK>"]


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _word_to_chars(word: str) -> Tuple[str, ...]:
    return tuple(_BYTE2CHAR[b] for b in word.encode("utf-8"))


def _get_pairs(word: Tuple[str, ...]) -> List[Tuple[str, str]]:
    return [(word[i], word[i + 1]) for i in range(len(word) - 1)]


def _apply_merge(
    word: Tuple[str, ...],
    a: str,
    b: str,
    merged: str,
) -> Tuple[str, ...]:
    out: List[str] = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
            out.append(merged)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return tuple(out)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class HybridTokenizer:
    """
    Byte-level BPE tokenizer.  Drop-in replacement for the old PMI tokenizer.

    Legacy constructor params (k_bases, max_merges) are accepted but ignored.
    """

    def __init__(
        self,
        lowercase: bool = False,
        k_bases: int = 0,    # legacy, ignored
        max_merges: int = 0, # legacy, ignored
    ):
        self.lowercase = lowercase

        self.token2id: Dict[str, int] = {}
        self.id2token: List[str] = []
        self._frozen = False

        self._word_freq: Counter[Tuple[str, ...]] = Counter()

        self.merge_list: List[Tuple[str, str]] = []
        self.merge_rank: Dict[Tuple[str, str], int] = {}
        self._merged_by: Dict[str, Tuple[str, str]] = {}

        self._word_cache: Dict[str, Tuple[int, ...]] = {}

        self._init_base_vocab()

    # ------------------------------------------------------------------
    # Vocab initialisation
    # ------------------------------------------------------------------

    def _init_base_vocab(self) -> None:
        self.token2id = {}
        self.id2token = []
        for tok in _SPECIAL:
            self._add_token(tok)
        for b in range(256):
            self._add_token(_BYTE2CHAR[b])

    def _add_token(self, tok: str) -> int:
        if tok not in self.token2id:
            idx = len(self.id2token)
            self.token2id[tok] = idx
            self.id2token.append(tok)
        return self.token2id[tok]

    # ------------------------------------------------------------------
    # Training interface
    # ------------------------------------------------------------------

    def add_text(self, text: str) -> None:
        if self._frozen:
            raise RuntimeError("Tokenizer is frozen; cannot add training data.")
        if self.lowercase:
            text = text.lower()
        for word in _PRETOK.findall(text):
            self._word_freq[_word_to_chars(word)] += 1

    def add_file(self, path: str) -> None:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                self.add_text(line)

    def freeze_vocab(
        self,
        vocab_size: int = 32_000,
        k_bases: int = 0,    # legacy, ignored
        max_merges: int = 0, # legacy, ignored
    ) -> None:
        """Run BPE merges until vocab reaches vocab_size."""
        if self._frozen:
            return

        target = max(vocab_size, len(self.id2token))
        n_merges = target - len(self.id2token)

        pair_freq: Counter[Tuple[str, str]] = Counter()
        for word, freq in self._word_freq.items():
            for pair in _get_pairs(word):
                pair_freq[pair] += freq

        words = dict(self._word_freq)

        for _ in range(n_merges):
            if not pair_freq:
                break
            best = pair_freq.most_common(1)[0][0]
            a, b = best
            merged = a + b

            self._add_token(merged)
            rank = len(self.merge_list)
            self.merge_list.append(best)
            self.merge_rank[best] = rank
            self._merged_by[merged] = (a, b)

            new_words: Dict[Tuple[str, ...], int] = {}
            for word, freq in words.items():
                if a not in word:
                    new_words[word] = freq
                    continue
                new_word = _apply_merge(word, a, b, merged)
                if new_word == word:
                    new_words[word] = freq
                    continue
                new_words[new_word] = freq

                for p in _get_pairs(word):
                    pair_freq[p] -= freq
                    if pair_freq[p] <= 0:
                        del pair_freq[p]
                for p in _get_pairs(new_word):
                    pair_freq[p] += freq

            words = new_words

        self._frozen = True
        self._word_freq.clear()

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def _bpe_word(self, word: str) -> Tuple[int, ...]:
        cached = self._word_cache.get(word)
        if cached is not None:
            return cached

        chars = _word_to_chars(word)
        if len(chars) == 1:
            result = (self.token2id[chars[0]],)
            self._word_cache[word] = result
            return result

        pieces = list(chars)
        while len(pieces) > 1:
            best_rank = len(self.merge_list)
            best_idx = -1
            for i in range(len(pieces) - 1):
                rank = self.merge_rank.get((pieces[i], pieces[i + 1]), len(self.merge_list))
                if rank < best_rank:
                    best_rank = rank
                    best_idx = i
            if best_idx == -1:
                break
            a, b = pieces[best_idx], pieces[best_idx + 1]
            pieces = pieces[:best_idx] + [a + b] + pieces[best_idx + 2:]

        result = tuple(
            self.token2id.get(p, self.token2id["<UNK>"]) for p in pieces
        )
        self._word_cache[word] = result
        return result

    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
        mode: str = "flat",  # legacy param, ignored
    ) -> List[int]:
        if not self._frozen:
            raise RuntimeError("Call freeze_vocab() before encode().")
        if self.lowercase:
            text = text.lower()

        ids: List[int] = []
        if add_special_tokens:
            ids.append(self.token2id["<BOS>"])
        for word in _PRETOK.findall(text):
            ids.extend(self._bpe_word(word))
        if add_special_tokens:
            ids.append(self.token2id["<EOS>"])
        return ids

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        special = set(_SPECIAL)
        byte_list: List[int] = []
        for idx in ids:
            if idx < 0 or idx >= len(self.id2token):
                continue
            tok = self.id2token[idx]
            if skip_special and tok in special:
                continue
            for char in tok:
                if char in _CHAR2BYTE:
                    byte_list.append(_CHAR2BYTE[char])

        if not byte_list:
            return ""

        byte_seq = bytes(byte_list)
        return byte_seq.decode("utf-8", errors="replace")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        payload = {
            "lowercase":  self.lowercase,
            "token2id":   self.token2id,
            "id2token":   self.id2token,
            "merge_list": self.merge_list,
            "merge_rank": self.merge_rank,
            "_merged_by": self._merged_by,
        }
        with gzip.open(path, "wb") as f:
            pickle.dump(payload, f, protocol=4)
        print(f"[Tokenizer] Saved {len(self.id2token):,} tokens → {path}")

    @classmethod
    def load(cls, path: str) -> "HybridTokenizer":
        with gzip.open(path, "rb") as f:
            payload = pickle.load(f)
        tok = cls.__new__(cls)
        tok.lowercase   = payload["lowercase"]
        tok.token2id    = payload["token2id"]
        tok.id2token    = payload["id2token"]
        tok.merge_list  = payload["merge_list"]
        tok.merge_rank  = payload["merge_rank"]
        tok._merged_by  = payload["_merged_by"]
        tok._frozen     = True
        tok._word_freq  = Counter()
        tok._word_cache = {}
        print(f"[Tokenizer] Loaded {len(tok.id2token):,} tokens ← {path}")
        return tok

    # ------------------------------------------------------------------
    # Utility / diagnostics
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return len(self.id2token)

    def segment(self, text: str) -> List[str]:
        """Return token strings (not ids) for a text."""
        if not self._frozen:
            raise RuntimeError("Call freeze_vocab() first.")
        tokens: List[str] = []
        for word in _PRETOK.findall(text):
            for idx in self._bpe_word(word):
                tokens.append(self.id2token[idx])
        return tokens

    def explain_token(self, token: str) -> str:
        """Recursively show how a merged token was assembled."""
        if token not in self._merged_by:
            try:
                return repr(bytes([_CHAR2BYTE[token]]))
            except KeyError:
                return repr(token)
        a, b = self._merged_by[token]
        return f"({self.explain_token(a)} + {self.explain_token(b)})"

    def top_merges(self, n: int = 20) -> List[Tuple[Tuple[str, str], int]]:
        return [(pair, rank) for rank, pair in enumerate(self.merge_list[:n])]

    def db_status(self) -> Dict[str, object]:
        return {
            "vocab_size":     len(self.id2token),
            "n_merges":       len(self.merge_list),
            "frozen":         self._frozen,
            "cache_size":     len(self._word_cache),
            "training_words": len(self._word_freq),
        }

    def self_test(self) -> bool:
        """Basic round-trip smoke test."""
        assert self._frozen, "freeze_vocab() must be called first"
        samples = [
            "Hello, world!",
            "The quick brown fox jumps over the lazy dog.",
            "café naïve résumé",
            "12345 !@#$%",
            "   spaces   ",
        ]
        ok = True
        for text in samples:
            rt = self.decode(self.encode(text))
            if rt != text:
                print(f"[FAIL] {text!r} → {rt!r}")
                ok = False
        if ok:
            print(f"[OK] self_test passed — vocab {len(self.id2token):,}")
        return ok
