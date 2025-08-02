"""
tokenize.py    minimal, fast prototype


• build_db(text | files)      → populate length/frequency DB
• freeze_vocab(k_bases, max_merges) → create base + merge tokens
• encode(text)                → list[(id,count)]  (run-length compressed)
• decode(encoded)             → original utf-8 string

The algorithm:
  1.  base tokens = shortest tokens (≤4 bytes) + n most-common of those
  2.  recursive greedy merge search  (longest merge first)
  3.  fallback: raw utf-8 bytes
"""

import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Tuple
import pickle, gzip, os


WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)      # letters/digits
SPECIAL_RE = re.compile(r"[^\w\s]", re.UNICODE)  # punctuation etc.
TOKEN_RE = re.compile(r"\n|\s+|\w+|[^\w\s]", re.UNICODE)

BYTE_LIMIT = 8

class HybridTokenizer:
    def __init__(
        self,
        special_tokens: List[str] = (
            "<PAD>", "<UNK>", "<CLS>", "<SEP>", "<MASK>",
            "<RES>",          # ← your old specials
            "<SP>", "<NL>"    # ← make sure these two are included
        )
    ):
        self.special_tokens = list(special_tokens)
        self.word_db: Counter = Counter()
        self.token2id, self.id2token = {}, []
        self.merge_rules = {}
        self.frozen = False

        
        for tok in self.special_tokens:
            self._add_token(tok)

        
        self.sp_id = self.token2id["<SP>"]
        self.nl_id = self.token2id["<NL>"]                   # assigns id

    # public API 
    def add_file(self, filename: str|Path):
        with open(filename, "r", encoding="utf-8") as fh:
            for line in fh:
                self.add_text(line)

    def _norm(self, tok: str) -> str:
        # Lowercase Latin, leave RTL scripts untouched
        return tok.lower() if tok and "A" <= tok[0] <= "Z" else tok

    def add_text(self, text: str):
        for w in WORD_RE.findall(text):
            self.word_db[self._norm(w)] += 1

    def db_status(self,
                  preview: int = 10,
                  byte_limit: int = 4,
                  k_bases: int = 100):
        """
        Print a concise report:

        • total unique words collected so far
        • total observed word-tokens
        • % of corpus occurrences that qualify as base-token candidates
          (<= `byte_limit` UTF-8 bytes)
        • how many special tokens are already reserved
        • current size of token table (grows if you’ve encoded text
          before freezing)
        • preview of the `preview` most-common would-be base tokens
        """
        total_types  = len(self.word_db)
        total_tokens = sum(self.word_db.values())

        # candidates that meet byte-length criterion
        base_cands   = {w:f for w,f in self.word_db.items()
                        if len(w.encode()) <= byte_limit}
        base_occurs  = sum(base_cands.values())
        base_ratio   = (base_occurs / total_tokens * 100) if total_tokens else 0

        # top-k preview
        preview_list = sorted(base_cands,
                              key=lambda w:(-self.word_db[w], len(w)))[:preview]

        print("=== HybridTokenizer DB status ===")
        print(f"• Unique word types collected : {total_types:,}")
        print(f"• Total word occurrences       : {total_tokens:,}")
        print(f"• Candidate bases (≤{byte_limit} bytes) : "
              f"{len(base_cands):,}  "
              f"[cover {base_ratio:4.1f}% of corpus]")
        print(f"• Special tokens reserved      : {len(self.special_tokens)}")
        print(f"• Current token2id size        : {len(self.token2id)} "
              f"(includes specials + any byte_* added so far)")
        print(f"• Frozen?                      : {self.frozen}")
        print(f"• Preview of top {preview} base candidates:")
        for w in preview_list:
            print(f"   {w:10}  freq={self.word_db[w]:>6}")

        # quick suggestion
        if base_ratio < 90:
            print("\n⚠️  < 90 % of corpus occurrences fit base-token criterion.")
            print("   Consider raising byte_limit or adding more data before freezing.")
        elif total_types < k_bases * 2:
            print("\n⚠️  Very small DB – you may want a larger corpus before freezing.")
        else:
            print("\n👍  Looks healthy – you can likely call freeze_vocab() anytime.")

    def freeze_vocab(self,
                     k_bases: int = 500,
                     max_merges: int = 10_000):
        """
        • choose base tokens  (≤4-byte UTF-8) – top-k by freq
        • build merge rules greedily by frequency
        """
        if self.frozen:
            return
        
        #  1: choose base tokens
        bases = [w for w in self.word_db
         if len(w.encode()) <= BYTE_LIMIT]
        bases.sort(key=lambda w: (-self.word_db[w], len(w)))         # freq⟂len
        bases = bases[:k_bases]
        for w in bases:
            self._add_token(w)

       
        #  2: create merges from bases  (very light-weight BPE)
        merge_counter: Counter = Counter()
        for word, freq in self.word_db.items():
            # skip if already base
            if word in self.token2id:
                continue
            # find 2-part splits (prefix+suffix) that are base tokens
            for i in range(1, len(word)):
                left, right = word[:i], word[i:]
                if left in self.token2id and right in self.token2id:
                    merge_counter[f"{left}+{right}"] += freq

        for merge, freq in merge_counter.most_common(max_merges):
            if merge not in self.token2id:
                self._add_token(merge)
                left, right = merge.split("+",1)
                self.merge_rules[merge] = (left, right)
        self.frozen = True


 
    def save(self, file: str | Path, compress: bool = True) -> None:
        """
        Persist the tokenizer (DB + vocab + merge rules).

        Args
        ----
        file       : destination path
        compress   : if True → use gzip-compressed pickle
        """
        state = dict(
            special_tokens = self.special_tokens,
            word_db        = self.word_db,
            token2id       = self.token2id,
            id2token       = self.id2token,
            merge_rules    = self.merge_rules,
            frozen         = self.frozen,
            sp_id          = getattr(self, "sp_id", None),
            nl_id          = getattr(self, "nl_id", None),
        )
        file = Path(file)
        if compress and not file.suffix.endswith(".gz"):
            file = file.with_suffix(file.suffix + ".gz")

        opener = gzip.open if compress else open
        with opener(file, "wb") as fh:
            pickle.dump(state, fh, protocol=pickle.HIGHEST_PROTOCOL)

    def encode(self, text: str) -> List[Tuple[int,int]]:
        """
        return run-length–compressed list of (token_id, count)
        """
        if not self.frozen:
            raise RuntimeError("call freeze_vocab() first")

        ids: List[int] = []
        # split into words + specials
        for tok in TOKEN_RE.findall(text):
            if tok == "\n":                     # newline → <NL>
                ids.append(self.nl_id)
            elif tok.isspace():                 # any other whitespace → <SP>
                ids.append(self.sp_id)
            elif tok in self.token2id:                       # exact token
                ids.append(self.token2id[tok])
            elif SPECIAL_RE.match(tok):                    # punct / emoji
                utf = tok.encode("utf-8")
                ids.extend(self._bytes_to_ids(utf))
            else:
                ids.extend(self._encode_word(tok.lower()))

        # run-length compress
        compressed: List[Tuple[int,int]] = []
        for tid in ids:
            if compressed and compressed[-1][0] == tid:
                prev = compressed[-1]
                compressed[-1] = (prev[0], prev[1]+1)
            else:
                compressed.append((tid,1))
        return compressed

    def _flush_bytes(self,buffer, out):
        if buffer:
            out.append(buffer.decode('utf-8', errors='replace'))
            buffer.clear()

    def decode(self, encoded):
        out, buf = [], bytearray()
        for tid, cnt in encoded:
            tok = self.id2token[tid]

            if tok == "<SP>":
                self._flush_bytes(buf, out);   out.append(" " * cnt)
            elif tok == "<NL>":
                self._flush_bytes(buf, out);   out.append("\n" * cnt)
            elif tok.startswith("byte_"):
                byte_val = int(tok[5:])
                buf.extend([byte_val] * cnt)          # add repeated bytes
            else:
                self._flush_bytes(buf, out)
                out.extend([tok] * cnt)
        self._flush_bytes(buf, out)
        return "".join(out)


    # internal helpers 
    def _add_token(self, token: str):
        self.token2id[token] = len(self.id2token)
        self.id2token.append(token)

    def _bytes_to_ids(self, b: bytes) -> List[int]:
        out = []
        for byte in b:
            tok = f"byte_{byte}"
            if tok not in self.token2id:
                self._add_token(tok)
            out.append(self.token2id[tok])
        return out

    # recursive greedy encoding of single word
    def _encode_word(self, word: str) -> List[int]:
        # exact?
        if word in self.token2id:
            return [self.token2id[word]]

        # greedy: longest merge present in merge_rules/token2id
        for length in range(len(word), 0, -1):
            prefix = word[:length]
            if prefix in self.token2id:
                suffix_ids = self._encode_word(word[length:])
                return [self.token2id[prefix]] + suffix_ids
            # try merge: any rule where left part == prefix?
            for merge, (left,right) in self.merge_rules.items():
                if prefix == merge:
                    suffix_ids = self._encode_word(word[length:])
                    return [self.token2id[merge]] + suffix_ids
        # fallback to bytes
        return self._bytes_to_ids(word.encode("utf-8"))
    


@classmethod
def load(cls, file: str | Path, compress: bool | None = None):
    """
    Restore a tokenizer saved with `save()`.

    Args
    ----
    file       : path to the pickle (auto-detect .gz)
    compress   : override auto-detection (True/False)
    """
    file = Path(file)
    if compress is None:
        compress = file.suffix.endswith(".gz")
    opener = gzip.open if compress else open
    with opener(file, "rb") as fh:
        state = pickle.load(fh)

    obj = cls(special_tokens=state["special_tokens"])
    # overwrite internals
    obj.word_db   = state["word_db"]
    obj.token2id  = state["token2id"]
    obj.id2token  = state["id2token"]
    obj.merge_rules = state["merge_rules"]
    obj.frozen    = state["frozen"]
    obj.sp_id     = state["sp_id"]
    obj.nl_id     = state["nl_id"]
    return obj



# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Example usage

    corpus = """Anya loved the old bookstore. It was a haven of musty paper and the scent of forgotten stories. One afternoon, she found a peculiar book tucked away on a high shelf. It was leather-bound, its title barely visible under layers of dust. Intrigued, she bought it. Back home, she carefully opened the book. The pages were filled with intricate drawings of plants and animals, each labeled with elegant, unfamiliar script. As she traced the lines of a strange, luminous flower, the drawing seemed to shimmer. Then, the flower pulsed with light, and the scent of its petals filled the room. Anya realized the book wasn't just a collection of drawings; it was a portal. With each page, she stepped further into a world of vibrant flora and fantastical creatures, forever changed by the stories held within its pages"""
    tok = HybridTokenizer()
    for line in corpus.splitlines():
        tok.add_text(line)
    tok.db_status(preview=5, byte_limit=4, k_bases=50)
    tok.freeze_vocab(k_bases=5000, max_merges=10000)

    s = "helo     my name is shmuel and i am 23 yers old !"
    print(s)
    encoded = tok.encode(s)
    print(encoded)
    print(tok.decode(tok.encode(s)))
# →  מה קורה? hello שלום!