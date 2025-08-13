"""
HybridTokenizer – minimal, fast prototype with semantic-leaning merges

• add_text(text) / add_file(path) / add_files(glob) → build frequency DB
• freeze_vocab(k_bases, max_merges) → create base + merge tokens
• encode(text, mode) → list[(id,count)] (rle) or list[int] (flat)
• decode(encoded) → original UTF‑8 string
• top_merges(), explain_token(), segment() → inspect learned structure

Algorithm
---------
1) Base tokens = most common words with ≤ byte_limit UTF‑8 bytes
2) Greedy longest‑match encoding against learned tokens
3) Fallback to raw UTF‑8 bytes for anything unknown
4) Merge learning uses PMI×freq on adjacent base pieces (captures morphology)

Notes
-----
- Keep this file name distinct from Python's stdlib `tokenize` module.
- Call freeze_vocab() after populating the DB, before encode/decode.
- `byte_limit=4` is a practical default; raise to 8 for languages with longer short words.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import List, Tuple, Union, Literal, Sequence
import pickle, gzip


# Regex patterns
# WORD_RE: sequences of letters or digits (underscore excluded)
# SPECIAL_RE: punctuation, emojis, symbols (anything that's not word or whitespace)
# TOKEN_RE: tokenizes into newlines, whitespace runs, word runs, or single specials
WORD_RE   = re.compile(r"[^\W_]+", re.UNICODE)     # letters & digits, no underscore
SPECIAL_RE = re.compile(r"[^\w\s]", re.UNICODE)   # punctuation, emoji, symbols
TOKEN_RE   = re.compile(r"\n|\s+|\w+|[^\w\s]", re.UNICODE)


class HybridTokenizer:
    def __init__(
        self,
        special_tokens: List[str] = (
            "<PAD>", "<UNK>", "<CLS>", "<SEP>", "<MASK>", "<EOS>",
            "<RES>", "<SP>", "<NL>"
        ),
        lowercase: bool = True,
        byte_limit: int = 4,
    ) -> None:
        # Deduplicate specials while preserving order
        self.special_tokens = list(dict.fromkeys(special_tokens))
        self.lowercase = lowercase
        self.byte_limit = int(byte_limit)

        self.word_db: Counter[str] = Counter()
        self.token2id: dict[str, int] = {}
        self.id2token: list[str] = []
        # merge_rules maps the concatenated token -> (left, right) parts
        self.merge_rules: dict[str, tuple[str, str]] = {}
        self.frozen: bool = False

        for tok in self.special_tokens:
            self._add_token(tok)
        # Cache special ids
        try:
            self.sp_id = self.token2id["<SP>"]
            self.nl_id = self.token2id["<NL>"]
        except KeyError as e:
            raise ValueError("<SP> and <NL> must be present in special_tokens") from e

    def _norm(self, tok: str) -> str:
        """Normalize token for the word DB (casefold if enabled)."""
        return tok.casefold() if self.lowercase else tok

    def add_text(self, text: str) -> None:
        """Add words from a text string to the frequency DB."""
        for w in WORD_RE.findall(text):
            self.word_db[self._norm(w)] += 1

    def add_file(self, filename: str | Path) -> None:
        """Stream file contents into the DB."""
        p = Path(filename)
        with p.open("r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                self.add_text(line)

    def add_files(self, pattern: str) -> None:
        """Glob pattern to add many files, e.g., "data/**/*.txt"."""
        for p in Path().glob(pattern):
            if p.is_file():
                self.add_file(p)

   
    @staticmethod
    def _utf8_len(s: str) -> int:
        return len(s.encode("utf-8"))

    @property
    def vocab_size(self) -> int:
        return len(self.id2token)

    def db_status(self, preview: int = 10, k_bases: int = 100) -> None:
        """Print a concise report of DB coverage and readiness."""
        total_types  = len(self.word_db)
        total_tokens = sum(self.word_db.values())
        # candidates that meet byte-length criterion
        base_cands = {w: f for w, f in self.word_db.items() if self._utf8_len(w) <= self.byte_limit}
        base_occurs = sum(base_cands.values())
        base_ratio = (base_occurs / total_tokens * 100) if total_tokens else 0.0

        preview_list = sorted(base_cands, key=lambda w: (-self.word_db[w], self._utf8_len(w)))[:preview]

        print("=== HybridTokenizer DB status ===")
        print(f"• Unique word types collected : {total_types:,}")
        print(f"• Total word occurrences       : {total_tokens:,}")
        print(f"• Candidate bases (≤{self.byte_limit} bytes) : {len(base_cands):,}  [cover {base_ratio:4.1f}% of corpus]")
        print(f"• Special tokens reserved      : {len(self.special_tokens)}")
        print(f"• Current token2id size        : {len(self.token2id)} (includes specials + any byte_* added so far)")
        print(f"• Frozen?                      : {self.frozen}")
        print(f"• Preview of top {preview} base candidates:")
        for w in preview_list:
            print(f"   {w:10}  freq={self.word_db[w]:>6}")

        if base_ratio < 90 and total_tokens:
            print("\n  < 90 % of corpus occurrences fit base-token criterion.")
            print("   Consider raising byte_limit or adding more data before freezing.")
        elif total_types < k_bases * 2:
            print("\n  Very small DB — you may want a larger corpus before freezing.")
        else:
            print("\n  Looks healthy — you can likely call freeze_vocab() anytime.")

    # ----------------------------
    # Vocab freezing (base + merges)
    # ----------------------------
    def freeze_vocab(self, k_bases: int = 500, max_merges: int = 10_000, min_freq: int = 2) -> None:
        """
        1) Base tokens = top-k words with ≤ byte_limit bytes (by freq)
        2) Learn merges using PMI × log(1+freq) on adjacent base pieces
           across the observed word corpus. Store merged token as concatenation
           (e.g., 'foo'+'bar' -> 'foobar') and record merge_rules.
        """
        if self.frozen:
            return

        # 1) choose base tokens
        bases = [w for w, f in self.word_db.items() if self._utf8_len(w) <= self.byte_limit]
        bases.sort(key=lambda w: (-self.word_db[w], self._utf8_len(w)))
        bases = bases[:k_bases]
        for w in bases:
            if w not in self.token2id:
                self._add_token(w)

        # helper: split a word into current base pieces (greedy longest match)
        def split_bases(word: str) -> list[str]:
            i, pieces = 0, []
            while i < len(word):
                j = len(word)
                hit = None
                while j > i:
                    cand = word[i:j]
                    if cand in self.token2id:
                        hit = cand
                        break
                    j -= 1
                if hit is None:
                    # fallback: single character (won't be added as a token yet)
                    hit = word[i:j] if j > i else word[i:i+1]
                pieces.append(hit)
                i += len(hit)
            return pieces

        # 2) collect adjacent base pairs with frequencies
        left_freq: Counter[str] = Counter()
        right_freq: Counter[str] = Counter()
        pair_freq: Counter[tuple[str, str]] = Counter()

        for w, f in self.word_db.items():
            if f < min_freq:
                continue
            pieces = split_bases(w)
            for a, b in zip(pieces, pieces[1:]):
                left_freq[a] += f
                right_freq[b] += f
                pair_freq[(a, b)] += f

        # 3) score merges by PMI × log(1+freq)
        import math
        merges_scored: list[tuple[float, tuple[str, str]]] = []
        total_pairs = sum(pair_freq.values()) or 1
        total_left  = sum(left_freq.values()) or 1
        total_right = sum(right_freq.values()) or 1

        for (a, b), f_ab in pair_freq.items():
            if f_ab < min_freq:
                continue
            p_ab = f_ab / total_pairs
            p_a  = left_freq[a] / total_left
            p_b  = right_freq[b] / total_right
            pmi = math.log(p_ab / max(p_a * p_b, 1e-12))
            score = pmi * math.log1p(f_ab)
            merges_scored.append((score, (a, b)))

        merges_scored.sort(reverse=True)
        for _, (a, b) in merges_scored[:max_merges]:
            merged = a + b
            if merged in self.token2id:
                continue
            self._add_token(merged)
            self.merge_rules[merged] = (a, b)

        self.frozen = True

    
    def encode(
        self,
        text: str,
        mode: Literal["rle", "flat"] = "rle"
    ) -> Union[List[Tuple[int, int]], List[int]]:
        """Encode text into token ids.

        mode="rle": return run-length compressed list of (token_id, count)
        mode="flat": return flat list of token_ids (no counts)
        """
        if not self.frozen:
            raise RuntimeError("call freeze_vocab() first")

        ids: List[int] = []
        for tok in TOKEN_RE.findall(text):
            if tok == "\n":                 # newline → <NL>
                ids.append(self.nl_id)
            elif tok.isspace():            # any other whitespace → <SP> (preserve run length)
                ids.extend([self.sp_id] * len(tok))
            elif tok in self.token2id:     # exact token (includes learned merges)
                ids.append(self.token2id[tok])
            elif SPECIAL_RE.match(tok):    # punctuation / emoji → bytes
                ids.extend(self._bytes_to_ids(tok.encode("utf-8")))
            else:                          # word-like
                norm = self._norm(tok)
                if norm == tok:
                    # safe to use learned subword tokens
                    ids.extend(self._encode_word(norm))
                else:
                    # preserve original casing by falling back to raw bytes
                    ids.extend(self._bytes_to_ids(tok.encode("utf-8")))


        if mode == "flat":
            return ids

        # run-length compress
        compressed: List[Tuple[int, int]] = []
        for tid in ids:
            if compressed and compressed[-1][0] == tid:
                t, c = compressed[-1]
                compressed[-1] = (t, c + 1)
            else:
                compressed.append((tid, 1))
        return compressed

    def _flush_bytes(self, buffer: bytearray, out: list[str]) -> None:
        if buffer:
            out.append(buffer.decode("utf-8", errors="replace"))
            buffer.clear()

    def decode(self, encoded: Union[Sequence[int], Sequence[Tuple[int, int]]]) -> str:
        """Decode from flat ids or RLE pairs back to a UTF‑8 string."""
        if isinstance(encoded, (int, str, bytes, bytearray)):
            raise TypeError("decode expects a sequence of ids or (id,count) pairs")

        # Optional: tensor/ndarray friendliness
        try:
            import torch  # type: ignore
            if isinstance(encoded, torch.Tensor):
                encoded = encoded.detach().cpu().tolist()
        except Exception:
            pass
        try:
            import numpy as np  # type: ignore
            if isinstance(encoded, np.ndarray):
                encoded = encoded.tolist()
        except Exception:
            pass

        if not encoded:
            return ""

        first = encoded[0]
        if isinstance(first, int):
            pairs = [(int(t), 1) for t in encoded]  # flat ids
        else:
            pairs = [(int(t), int(c)) for (t, c) in encoded]  # rle pairs
            for _, c in pairs:
                if c <= 0:
                    raise ValueError("Non-positive count in encoded stream")

        V = len(self.id2token)
        for tid, _ in pairs:
            if not (0 <= tid < V):
                raise IndexError(f"Token id {tid} out of range [0,{V-1}]")

        out: list[str] = []
        buf = bytearray()
        for tid, cnt in pairs:
            tok = self.id2token[tid]
            if tok == "<SP>":
                self._flush_bytes(buf, out); out.append(" " * cnt)
            elif tok == "<NL>":
                self._flush_bytes(buf, out); out.append("\n" * cnt)
            elif tok.startswith("byte_"):
                byte_val = int(tok[5:]); buf.extend([byte_val] * cnt)
            else:
                self._flush_bytes(buf, out); out.extend([tok] * cnt)
        self._flush_bytes(buf, out)
        return "".join(out)

    
    def _add_token(self, token: str) -> None:
        self.token2id[token] = len(self.id2token)
        self.id2token.append(token)

    def _bytes_to_ids(self, b: bytes) -> List[int]:
        out: List[int] = []
        for byte in b:
            tok = f"byte_{byte}"
            if tok not in self.token2id:
                self._add_token(tok)
            out.append(self.token2id[tok])
        return out

    def _encode_word(self, word: str) -> List[int]:
        """Recursive greedy encoding of a single *normalized* word.
        Matches the longest known token at each step; falls back to UTF‑8 bytes.
        """
        if word in self.token2id:
            return [self.token2id[word]]

        i: int = 0
        out: List[int] = []
        while i < len(word):
            j = len(word)
            hit: str | None = None
            while j > i:
                cand = word[i:j]
                if cand in self.token2id:
                    hit = cand
                    break
                j -= 1
            if hit is None:
                # fallback: emit the next Unicode codepoint as bytes
                b = word[i].encode("utf-8")
                out.extend(self._bytes_to_ids(b))
                i += 1
            else:
                out.append(self.token2id[hit])
                i += len(hit)
        return out  # type: ignore[return-value]


    def top_merges(self, n: int = 20) -> None:
        items: list[tuple[str, str, str, int]] = []
        for merged, (a, b) in self.merge_rules.items():
            items.append((merged, a, b, self.word_db.get(merged, 0)))
        items.sort(key=lambda x: (-len(x[0]), -x[3]))  # long & frequent first
        for merged, a, b, f in items[:n]:
            print(f"{merged:20}  = {a} + {b}   (freq≈{f})")

    def explain_token(self, token: str) -> list[str]:
        """Return the base pieces that formed `token` (recursive decomposition)."""
        parts: list[str] = []
        def rec(t: str) -> None:
            if t in self.merge_rules:
                L, R = self.merge_rules[t]
                rec(L); rec(R)
            else:
                parts.append(t)
        rec(token)
        return parts

    def segment(self, text: str) -> list[tuple[str, str]]:
        """Return [(token_string, kind)] for a given text, where kind∈{base,merge,byte,sp,nl}."""
        ids = self.encode(text, mode="flat")
        out: list[tuple[str, str]] = []
        for tid in ids:
            tok = self.id2token[tid]
            if tok == "<SP>": kind = "sp"
            elif tok == "<NL>": kind = "nl"
            elif tok.startswith("byte_"): kind = "byte"
            elif tok in self.merge_rules: kind = "merge"
            else: kind = "base"
            out.append((tok, kind))
        return out

    def save(self, file: str | Path, compress: bool = True) -> None:
        state = dict(
            special_tokens=self.special_tokens,
            lowercase=self.lowercase,
            byte_limit=self.byte_limit,
            word_db=self.word_db,
            token2id=self.token2id,
            id2token=self.id2token,
            merge_rules=self.merge_rules,
            frozen=self.frozen,
            sp_id=getattr(self, "sp_id", None),
            nl_id=getattr(self, "nl_id", None),
        )
        file = Path(file)
        if compress and not str(file).endswith(".gz"):
            file = file.with_suffix(file.suffix + ".gz")
        opener = gzip.open if compress else open
        with opener(file, "wb") as fh:
            pickle.dump(state, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, file: str | Path, compress: bool | None = None) -> "HybridTokenizer":
        file = Path(file)
        if compress is None:
            compress = str(file).endswith(".gz")
        opener = gzip.open if compress else open
        with opener(file, "rb") as fh:
            state = pickle.load(fh)

        obj = cls(
            special_tokens=state["special_tokens"],
            lowercase=state.get("lowercase", True),
            byte_limit=state.get("byte_limit", 4),
        )
        obj.word_db     = state["word_db"]
        obj.token2id    = state["token2id"]
        obj.id2token    = state["id2token"]
        obj.merge_rules = state["merge_rules"]
        obj.frozen      = state["frozen"]
        obj.sp_id       = state.get("sp_id", obj.token2id.get("<SP>", 0))
        obj.nl_id       = state.get("nl_id", obj.token2id.get("<NL>", 0))
        return obj

    def self_test(self) -> None:
        s = "Hello, world!\nשלום  🙂"
        enc_rle  = self.encode(s, mode="rle")
        enc_flat = self.encode(s, mode="flat")
        assert self.decode(enc_rle)  == s
        assert self.decode(enc_flat) == s
        print("Self‑test OK (round‑trip rle/flat)")
