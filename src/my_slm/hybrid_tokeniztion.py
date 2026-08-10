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
import logging
import pickle
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .exceptions import (
    TokenizerError,
    TokenizerNotFrozenError,
    TokenizerFrozenError,
    VocabSizeError,
    EncodingError,
    DecodingError,
)

_logger = logging.getLogger(__name__)

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
    Byte-level BPE (Byte-Pair Encoding) tokenizer inspired by GPT-2 and LLaMA.

    Provides training and inference interfaces for tokenizing text at the subword level
    using iterative merging of the most frequent byte pairs.

    Attributes:
        lowercase (bool): Whether to lowercase text during tokenization.
        token2id (Dict[str, int]): Mapping from token strings to IDs.
        id2token (List[str]): Mapping from IDs to token strings.
        merge_list (List[Tuple[str, str]]): Ordered list of merge operations applied.
        _frozen (bool): Whether tokenizer is frozen (training complete).

    Legacy constructor params (k_bases, max_merges) are accepted but ignored.

    Example:
        >>> tok = HybridTokenizer()
        >>> tok.add_text("Hello world! " * 100)
        >>> tok.freeze_vocab(32_000)
        >>> ids = tok.encode("Hello world")
        >>> text = tok.decode(ids)
    """

    def __init__(
        self,
        lowercase: bool = False,
        k_bases: int = 0,    # legacy, ignored
        max_merges: int = 0, # legacy, ignored
    ) -> None:
        """
        Initialize a HybridTokenizer.

        Args:
            lowercase: Convert text to lowercase during encoding/training.
            k_bases: Ignored (legacy parameter).
            max_merges: Ignored (legacy parameter).
        """
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
        """Initialize base vocabulary: special tokens + all 256 bytes."""
        self.token2id = {}
        self.id2token = []
        for tok in _SPECIAL:
            self._add_token(tok)
        for b in range(256):
            self._add_token(_BYTE2CHAR[b])

    def _add_token(self, tok: str) -> int:
        """
        Add a token to the vocabulary or return its existing ID.

        Args:
            tok: Token string to add.

        Returns:
            Token ID (new or existing).
        """
        if tok not in self.token2id:
            idx = len(self.id2token)
            self.token2id[tok] = idx
            self.id2token.append(tok)
        return self.token2id[tok]

    # ------------------------------------------------------------------
    # Training interface
    # ------------------------------------------------------------------

    def add_text(self, text: str) -> None:
        """
        Add text to the training corpus.

        Args:
            text: Text to tokenize and accumulate frequencies.

        Raises:
            TokenizerFrozenError: If tokenizer is frozen.
            EncodingError: If text encoding fails.
        """
        if self._frozen:
            raise TokenizerFrozenError()

        if not isinstance(text, str):
            raise EncodingError(str(text), "text must be a string")

        if not text:
            _logger.debug("add_text called with empty text; skipping")
            return

        try:
            if self.lowercase:
                text = text.lower()
            for word in _PRETOK.findall(text):
                self._word_freq[_word_to_chars(word)] += 1
        except Exception as e:
            raise EncodingError(text, f"failed to process text: {e}") from e

    def add_file(self, path: str) -> None:
        """
        Add text from a file to the training corpus.

        Args:
            path: Path to text file.

        Raises:
            TokenizerFrozenError: If tokenizer is frozen.
            FileNotFoundError: If file doesn't exist.
            EncodingError: If file reading or processing fails.
        """
        if self._frozen:
            raise TokenizerFrozenError()

        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Training file not found: {file_path}")

        if not file_path.is_file():
            raise ValueError(f"Expected file, got directory: {file_path}")

        try:
            with open(file_path, encoding="utf-8", errors="replace") as f:
                lines_read = 0
                for line in f:
                    self.add_text(line)
                    lines_read += 1
                _logger.info(f"Added {lines_read} lines from {file_path.name}")
        except TokenizerFrozenError:
            raise
        except EncodingError:
            raise
        except Exception as e:
            raise EncodingError(str(file_path), f"failed to read file: {e}") from e

    def freeze_vocab(
        self,
        vocab_size: int = 32_000,
        k_bases: int = 0,    # legacy, ignored
        max_merges: int = 0, # legacy, ignored
    ) -> None:
        """
        Run BPE merges until vocabulary reaches target size.

        Args:
            vocab_size: Target vocabulary size. Minimum is 256 (base byte vocabulary).
            k_bases: Ignored (legacy parameter).
            max_merges: Ignored (legacy parameter).

        Raises:
            VocabSizeError: If vocab_size < 256.
        """
        if self._frozen:
            _logger.info("Tokenizer already frozen; skipping freeze_vocab()")
            return

        min_vocab = len(self.id2token)
        if vocab_size < min_vocab:
            raise VocabSizeError(vocab_size, min_vocab)

        target = max(vocab_size, len(self.id2token))
        n_merges = target - len(self.id2token)

        _logger.info(
            f"Starting BPE: {len(self._word_freq)} unique words, "
            f"targeting vocab_size={target} ({n_merges} merges)"
        )

        pair_freq: Counter[Tuple[str, str]] = Counter()
        for word, freq in self._word_freq.items():
            for pair in _get_pairs(word):
                pair_freq[pair] += freq

        words = dict(self._word_freq)

        for merge_step in range(n_merges):
            if not pair_freq:
                _logger.warning(
                    f"BPE halted early at {merge_step}/{n_merges}: "
                    f"no more pairs to merge"
                )
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

            if (merge_step + 1) % max(1, n_merges // 10) == 0:
                _logger.debug(
                    f"BPE progress: {merge_step + 1}/{n_merges} "
                    f"({100 * (merge_step + 1) / n_merges:.0f}%), "
                    f"vocab_size={len(self.id2token)}"
                )

        self._frozen = True
        self._word_freq.clear()
        _logger.info(f"Tokenizer frozen with vocab_size={len(self.id2token)}")

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def _bpe_word(self, word: str) -> Tuple[int, ...]:
        """
        Encode a single word using BPE.

        Uses cached results when available for performance.

        Args:
            word: Single word to encode.

        Returns:
            Tuple of token IDs for this word.
        """
        cached = self._word_cache.get(word)
        if cached is not None:
            return cached

        try:
            chars = _word_to_chars(word)
        except Exception as e:
            _logger.warning(f"Failed to convert word to chars: {word!r}: {e}")
            return (self.token2id.get("<UNK>", 3),)

        if len(chars) == 1:
            result = (self.token2id.get(chars[0], self.token2id.get("<UNK>", 3)),)
            self._word_cache[word] = result
            return result

        pieces = list(chars)
        while len(pieces) > 1:
            best_rank = len(self.merge_list)
            best_idx = -1
            for i in range(len(pieces) - 1):
                rank = self.merge_rank.get(
                    (pieces[i], pieces[i + 1]), len(self.merge_list)
                )
                if rank < best_rank:
                    best_rank = rank
                    best_idx = i
            if best_idx == -1:
                break
            a, b = pieces[best_idx], pieces[best_idx + 1]
            pieces = pieces[:best_idx] + [a + b] + pieces[best_idx + 2:]

        result = tuple(
            self.token2id.get(p, self.token2id.get("<UNK>", 3)) for p in pieces
        )
        self._word_cache[word] = result
        return result

    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
        mode: str = "flat",  # legacy param, ignored
    ) -> List[int]:
        """
        Encode text to token IDs.

        Args:
            text: Text to encode.
            add_special_tokens: If True, prepend BOS and append EOS tokens.
            mode: Ignored (legacy parameter).

        Returns:
            List of token IDs.

        Raises:
            TokenizerNotFrozenError: If tokenizer is not frozen.
            EncodingError: If text encoding fails.
        """
        if not self._frozen:
            raise TokenizerNotFrozenError("encode")

        if not isinstance(text, str):
            raise EncodingError(str(text), "text must be a string")

        if not text:
            ids: List[int] = []
            if add_special_tokens:
                bos = self.token2id.get("<BOS>", 1)
                eos = self.token2id.get("<EOS>", 2)
                ids = [bos, eos]
            return ids

        try:
            if self.lowercase:
                text = text.lower()

            ids = []
            if add_special_tokens:
                ids.append(self.token2id.get("<BOS>", 1))
            for word in _PRETOK.findall(text):
                ids.extend(self._bpe_word(word))
            if add_special_tokens:
                ids.append(self.token2id.get("<EOS>", 2))
            return ids
        except Exception as e:
            raise EncodingError(text, f"encoding failed: {e}") from e

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """
        Decode token IDs back to text.

        Args:
            ids: List of token IDs to decode.
            skip_special: If True, skip special tokens like BOS, EOS, PAD.

        Returns:
            Decoded text. Invalid characters are replaced with U+FFFD.

        Raises:
            DecodingError: If IDs are invalid or decoding fails.
        """
        if not isinstance(ids, (list, tuple)):
            raise DecodingError([], f"expected list/tuple, got {type(ids).__name__}")

        if not ids:
            return ""

        try:
            special = set(_SPECIAL)
            byte_list: List[int] = []

            for idx in ids:
                if not isinstance(idx, int):
                    _logger.warning(f"Skipping non-integer token ID: {idx!r}")
                    continue

                if idx < 0 or idx >= len(self.id2token):
                    _logger.debug(f"Skipping out-of-range token ID: {idx}")
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
        except Exception as e:
            raise DecodingError(ids, f"decoding failed: {e}") from e

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Save tokenizer to a gzip-compressed pickle file.

        Args:
            path: Path to save the tokenizer.

        Raises:
            TokenizerError: If save fails.
        """
        try:
            file_path = Path(path)
            file_path.parent.mkdir(parents=True, exist_ok=True)

            payload = {
                "lowercase": self.lowercase,
                "token2id": self.token2id,
                "id2token": self.id2token,
                "merge_list": self.merge_list,
                "merge_rank": self.merge_rank,
                "_merged_by": self._merged_by,
            }
            with gzip.open(file_path, "wb") as f:
                pickle.dump(payload, f, protocol=4)
            _logger.info(
                f"Saved {len(self.id2token):,} tokens → {file_path} "
                f"({file_path.stat().st_size / 1024 / 1024:.1f} MB)"
            )
        except Exception as e:
            raise TokenizerError(f"Failed to save tokenizer to {path!r}: {e}") from e

    @classmethod
    def load(cls, path: str) -> "HybridTokenizer":
        """
        Load tokenizer from a gzip-compressed pickle file.

        Args:
            path: Path to load the tokenizer from.

        Returns:
            Loaded HybridTokenizer instance.

        Raises:
            FileNotFoundError: If file doesn't exist.
            TokenizerError: If load fails.
        """
        try:
            file_path = Path(path)
            if not file_path.exists():
                raise FileNotFoundError(f"Tokenizer file not found: {file_path}")

            with gzip.open(file_path, "rb") as f:
                payload = pickle.load(f)

            tok = cls.__new__(cls)
            tok.lowercase = payload["lowercase"]
            tok.token2id = payload["token2id"]
            tok.id2token = payload["id2token"]
            tok.merge_list = payload["merge_list"]
            tok.merge_rank = payload["merge_rank"]
            tok._merged_by = payload["_merged_by"]
            tok._frozen = True
            tok._word_freq = Counter()
            tok._word_cache = {}

            _logger.info(
                f"Loaded {len(tok.id2token):,} tokens ← {file_path} "
                f"({file_path.stat().st_size / 1024 / 1024:.1f} MB)"
            )
            return tok
        except FileNotFoundError:
            raise
        except Exception as e:
            raise TokenizerError(f"Failed to load tokenizer from {path!r}: {e}") from e

    # ------------------------------------------------------------------
    # Utility / diagnostics
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return len(self.id2token)

    def segment(self, text: str) -> List[str]:
        """
        Return token strings (not IDs) for a text.

        Args:
            text: Text to segment into tokens.

        Returns:
            List of token strings.

        Raises:
            TokenizerNotFrozenError: If tokenizer is not frozen.
        """
        if not self._frozen:
            raise TokenizerNotFrozenError("segment")

        if not isinstance(text, str):
            raise TypeError(f"text must be a string, got {type(text).__name__}")

        tokens: List[str] = []
        for word in _PRETOK.findall(text):
            for idx in self._bpe_word(word):
                if 0 <= idx < len(self.id2token):
                    tokens.append(self.id2token[idx])
        return tokens

    def explain_token(self, token: str) -> str:
        """
        Recursively explain how a merged token was assembled.

        Args:
            token: Token to explain.

        Returns:
            String representation of token assembly.
        """
        if token not in self._merged_by:
            try:
                return repr(bytes([_CHAR2BYTE[token]]))
            except KeyError:
                return repr(token)
        a, b = self._merged_by[token]
        return f"({self.explain_token(a)} + {self.explain_token(b)})"

    def top_merges(self, n: int = 20) -> List[Tuple[Tuple[str, str], int]]:
        """
        Get the top-N most frequent merge operations.

        Args:
            n: Number of top merges to return.

        Returns:
            List of (pair, rank) tuples.
        """
        return [(pair, rank) for rank, pair in enumerate(self.merge_list[:n])]

    def db_status(self) -> Dict[str, object]:
        """Get diagnostic status of the tokenizer."""
        return {
            "vocab_size": len(self.id2token),
            "n_merges": len(self.merge_list),
            "frozen": self._frozen,
            "cache_size": len(self._word_cache),
            "training_words": len(self._word_freq),
        }

    def self_test(self) -> bool:
        """
        Run a round-trip encoding/decoding smoke test.

        Returns:
            True if all tests pass.

        Raises:
            TokenizerNotFrozenError: If tokenizer is not frozen.
        """
        if not self._frozen:
            raise TokenizerNotFrozenError("self_test")

        samples = [
            "Hello, world!",
            "The quick brown fox jumps over the lazy dog.",
            "café naïve résumé",
            "12345 !@#$%",
            "   spaces   ",
            "",  # edge case: empty string
            "a",  # edge case: single character
        ]
        ok = True
        for text in samples:
            try:
                encoded = self.encode(text)
                rt = self.decode(encoded)
                if rt != text:
                    _logger.error(f"Round-trip failed: {text!r} → {rt!r}")
                    ok = False
                else:
                    _logger.debug(f"Round-trip OK: {text!r} ({len(encoded)} tokens)")
            except Exception as e:
                _logger.error(f"Round-trip exception: {text!r}: {e}")
                ok = False

        if ok:
            _logger.info(f"self_test passed — vocab_size={len(self.id2token)}")
        else:
            _logger.warning(f"self_test had failures — vocab_size={len(self.id2token)}")
        return ok
