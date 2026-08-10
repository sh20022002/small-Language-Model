# Security Audit — feature/multi-agent-improvements

**Date:** 2026-08-10
**Scope:** Full codebase (`src/my_slm/`, `tests/`, `kaggle_dual_gpu_finetune.ipynb`,
`pyproject.toml`, dependency pins, `.gitignore`) as it stood on branch
`feature/multi-agent-improvements`.
**Reviewer:** Security Reviewer agent.
**Note:** This branch was under active, concurrent development by other
agents (Backend Developer, DevOps/MLOps Engineer) while this audit ran —
`hybrid_tokeniztion.py`, `pyproject.toml`, `train.py`, and the Kaggle
notebook all changed mid-review. Findings below reflect the state of each
file at the time it was read (line numbers cited accordingly); a couple of
fixes were applied directly where the file was stable and the change was
self-contained (see "Status" per finding). Findings touching files with
in-flight edits from other agents are flagged rather than patched, to avoid
clobbering concurrent work.

## Summary

| # | Finding | Severity | Status |
|---|---|---|---|
| 1 | `torch.load()` unsafe-deserialization fallback (RCE) | **Critical** | **Fixed** (train.py, semantic_eval.py) |
| 2 | `torch>=2.0` permits CVE-2025-32434-vulnerable PyTorch | **Critical** | **Fixed** (pyproject.toml) |
| 3 | Kaggle notebook `_generate()` loads checkpoints with no `weights_only` at all | **High** | Flagged — needs fix in notebook |
| 4 | `HybridTokenizer` uses raw `pickle` for vocab persistence | **Medium** | Flagged — needs fix, file mid-edit |
| 5 | Unpinned `pip install git+https://...` (supply chain) | **Medium** | Flagged — recommend commit/tag pin |
| 6 | `trust_remote_code=True` in BLiMP dataset load | **Medium** | Flagged — recommend removal/pin |
| 7 | Pinned notebook deps (`transformers==4.41.0` etc.) have since-patched CVEs | **Medium** | Flagged — recommend version bump |
| 8 | No dependency vulnerability scanning in CI | **Medium** | Flagged — recommend `pip-audit`/`bandit` job |
| 9 | `os.system()` used to launch torchrun | **Low** | Flagged — hardening suggestion |
| 10 | Wikipedia-dump temp files never cleaned up | **Low** | Flagged — hygiene/DoS-by-disk-exhaustion |
| 11 | `.gitignore` doesn't explicitly exclude `*.pt` / `kaggle.json` | **Low** | Flagged — hygiene, file mid-edit |
| 12 | `validate_path()` normalizes but doesn't sandbox to a base dir | **Info** | Accepted — no attacker-controlled path input exists today |
| 13 | No secrets found in repo or history; `.env` correctly gitignored | **Info** | Verified clean |

---

## 1. `torch.load()` silently falls back to unsafe deserialization — CRITICAL — FIXED

**Where:** `src/my_slm/train.py` (`load_latest_checkpoint._load_state`),
`src/my_slm/semantic_eval.py` (`load_model_and_tok`).

**Before:**
```python
try:
    state = torch.load(path, map_location="cpu", weights_only=True)
except Exception:
    # Fallback for older PyTorch versions
    state = torch.load(path, map_location="cpu")
```

`weights_only=True` is PyTorch's safety boundary against
[CWE-502](https://cwe.mitre.org/data/definitions/502.html) (deserialization
of untrusted data): it restricts unpickling to a small allow-list of
tensor-safe types. Catching a bare `Exception` and silently retrying
*without* that flag defeats the protection entirely — any error from the
safe path (including the file legitimately containing something
disallowed, i.e. a maliciously crafted object) falls straight through to a
full, unrestricted `pickle.load()`. Checkpoints in this project are loaded
from `MODELS_DIR` / `MODEL_PATH`, which routinely point into
`/kaggle/input/...` — datasets that may be attached from Kaggle listings
the notebook operator did not author. A crafted `.pt` file there would
execute arbitrary code the moment `_generate()` or `load_latest_checkpoint()`
loads it.

**Fix applied:** the fallback now only triggers on `TypeError` (the actual
old-PyTorch-has-no-`weights_only`-kwarg case) and logs a loud warning when
it does. A `pickle.UnpicklingError` from the safe path now raises instead
of retrying unsafely, with a message pointing at `SECURITY.md` for how to
override deliberately if the file is genuinely trusted.

**Residual risk:** `src/my_slm/utils.py` (`load_checkpoint`,
`load_model_safely`) already used `weights_only=True` with no unsafe
fallback — no change needed there.

---

## 2. `torch>=2.0` in `pyproject.toml` permits a critical `torch.load()` RCE — CRITICAL — FIXED

**Where:** `pyproject.toml`, `dependencies`.

[CVE-2025-32434](https://github.com/pytorch/pytorch/security/advisories/GHSA-53q9-r3pm-6pq6)
(CVSS 9.3) is a `torch.load()` RCE that affects **all PyTorch ≤2.5.1, even
with `weights_only=True` set** — the exact flag this codebase relies on as
its deserialization safety boundary (see Finding 1). Patched in torch
2.6.0. The package's own dependency floor, `torch>=2.0`, permitted
installing a vulnerable version; only the Kaggle notebook's separate,
manually maintained `pip install torch==2.6.0` pin happened to avoid it.
Anyone installing this package outside that notebook (`pip install
hybrid-tokenizer`, or `pip install -e .`) had no such protection.

**Fix applied:** bumped to `torch>=2.6.0` with an inline comment explaining
why this is a hard floor, not a style preference.

---

## 3. Kaggle notebook `_generate()` loads checkpoints with no `weights_only` protection at all — HIGH — NOT FIXED (flagged)

**Where:** `kaggle_dual_gpu_finetune.ipynb`, Cell "Cell 7 — Generation
test" (`_generate()`):
```python
ckpt = torch.load(path, map_location="cpu")
```

Unlike every other `torch.load()` call site in this codebase, this one
passes no `weights_only` argument at all — on any PyTorch this defaults to
the fully unsafe unpickle path (`weights_only=False`), independent of the
`CVE-2025-32434` fix in 2.6.0. `path` here resolves from `MODEL_PATH`
(user-set string) or an auto-detected file under `MODELS_DIR`, which — same
as Finding 1 — can point into `/kaggle/input/...`. This is the single
riskiest checkpoint-loading call site in the project because it has *zero*
of the mitigations applied elsewhere.

**Recommended fix:** `ckpt = torch.load(path, map_location="cpu",
weights_only=True)`, with the same "fail loudly, don't silently downgrade"
pattern applied to `train.py`/`semantic_eval.py` in Findings 1–2.

**Why not fixed directly:** this file (`kaggle_dual_gpu_finetune.ipynb`)
was being actively edited by another agent throughout this review (it
appeared in `git status` as modified from the start of the session, and
changed further while this audit was in progress). Hand-editing notebook
JSON concurrently with another agent's edits risks silently reverting or
corrupting their in-flight changes. **Action for Backend Dev/DevOps:**
apply the one-line change above to Cell 16 (`_generate`) before merging.

---

## 4. `HybridTokenizer` persists vocab via raw `pickle` — MEDIUM — NOT FIXED (flagged)

**Where:** `src/my_slm/hybrid_tokeniztion.py`, `HybridTokenizer.save()` /
`HybridTokenizer.load()` (currently around line 531 / line 561).

```python
with gzip.open(file_path, "rb") as f:
    payload = pickle.load(f)
```

The payload is `{"lowercase": bool, "token2id": dict[str,int], "id2token":
list[str], "merge_list": list[tuple[str,str]], "merge_rank":
dict[tuple[str,str],int], "_merged_by": dict[str,tuple[str,str]]}` — plain
strings, ints, lists, dicts, and tuples. **None of this requires pickle.**
Tokenizer files are loaded from the same `TOKENIZER_PATH` /
`/kaggle/input/...` surface as model checkpoints (Findings 1–3), so this is
the same CWE-502 exposure applied to tokenizer files instead of weight
files — a crafted `tokenizer.pkl.gz` achieves the same arbitrary code
execution as a crafted `.pt` file, with no `weights_only`-style guard rail
available for pickle the way there is for `torch.load`.

**Recommended fix:** switch `save`/`load` to `json` (gzip-compressed, same
as today) since every field is JSON-native except the `tuple` keys in
`merge_rank`/`merge_list`, which can be encoded as `"a b"`-joined
strings or 2-element lists and reconstructed on load. This is a
self-contained, low-risk change (the schema is simple and stable), but it
touches the exact class another agent was mid-refactor on (new
`TokenizerError`/`EncodingError` exception hierarchy, logging) when this
audit ran, so it's left as a flagged recommendation rather than an
in-place edit to avoid a collision.

**If pickle must be kept short-term:** at minimum, document in
`SECURITY.md` (done) that tokenizer files are as untrusted as checkpoints,
and consider `pickle.Unpickler` with a restricted `find_class()` allow-list
as a stop-gap.

---

## 5. Unpinned `git+https://...` install — MEDIUM (supply chain) — flagged

**Where:** `kaggle_dual_gpu_finetune.ipynb`, Cell 2:
```
!pip install -q --upgrade --force-reinstall --no-build-isolation --no-cache-dir git+https://github.com/sh20022002/small-Language-Model.git
```

No `@<commit-sha>` or `@<tag>` pin means every Kaggle session installs
whatever is on the default branch *at run time*, not what was reviewed.
Combined with `--force-reinstall --no-cache-dir`, this guarantees the
latest remote code always executes with full GPU-session privileges. If
the repo's default branch is ever pushed to by a compromised account (or a
bad PR merges), every subsequent Kaggle run — potentially unattended,
scheduled runs — executes it. This is a standard supply-chain hardening
gap, not an active compromise.

**Recommended fix:** pin to a commit SHA (`...git@<sha>`) or a released
tag, and bump deliberately as part of the release process already
documented in `DEPLOYMENT.md`.

---

## 6. `trust_remote_code=True` for BLiMP dataset — MEDIUM — flagged

**Where:** `src/my_slm/semantic_eval.py:205`, `eval_blimp()`:
```python
ds = load_dataset('nyu-mll/blimp', phenomenon, split='train', trust_remote_code=True)
```

`trust_remote_code=True` tells the `datasets` library to download and
execute a Python loading script from the Hugging Face Hub repo, rather
than just reading data. The identifier is hardcoded to a well-known public
benchmark (not attacker-influenced input), which meaningfully lowers real-
world risk — but it's still routine execution of remote code as a side
effect of running the evaluation suite, and it's the kind of flag security
scanners (and HF's own advisories) flag on sight.

**Recommended fix:** try loading without `trust_remote_code=True` first
(the call is already wrapped in a `try/except` that degrades to `'skip
(...)'` per phenomenon, so removing the flag fails safe rather than
crashing); if BLiMP genuinely requires a loading script on `datasets==2.20.0`,
pin a `revision=<commit-sha>` so the executed script can't change without a
deliberate bump.

**Not fixed directly:** functional behavior risk (BLiMP eval may start
silently skipping if the flag turns out to be required) means this needs a
one-time verification run against real Kaggle GPU time, which is outside
this review's scope — flagged for Backend Dev to verify and land.

---

## 7. Notebook-pinned dependencies have since-patched CVEs — MEDIUM — flagged

**Where:** `kaggle_dual_gpu_finetune.ipynb`, Cell 2.

Versions pinned: `torch==2.6.0` (good — see Finding 2), `accelerate==0.31.0`,
`bitsandbytes==0.43.1`, `datasets==2.20.0`, `transformers==4.41.0`,
`huggingface_hub==0.23.0`, `galore-torch==0.2.0`.

`transformers==4.41.0` predates fixes for multiple disclosed ReDoS issues
(e.g. [CVE-2025-2099](https://github.com/advisories/GHSA-qq3j-4f4f-9583),
fixed in 4.50.0) and an insecure-URL-validation issue in `image_utils.py`
(fixed by 4.49.0/later). This codebase's use of `transformers` is limited
to `AutoTokenizer`/`AutoConfig` loading (no image pipelines, and the
ReDoS-affected code is in `testing_utils`, not on this project's hot
path), so exploitability here specifically looks low — but "pin once,
never revisit" means genuinely-patched-upstream bugs stay live in this
project indefinitely. Note also `transformers` had a separate, more severe
RCE (CVE-2026-4372, config-injection via `_attn_implementation_internal`,
patched in 5.3.0) — that one affects 4.56.0+, so 4.41.0 predates and is
*not* exposed to it, but it's a good example of why the pin needs periodic
review rather than being "safe forever."

**Recommended fix:** bump `transformers` to the latest 4.x or 5.x release
compatible with the rest of the pinned stack, and re-run the pinned-package
validation cell (Cell 1 already checks import success) before merging.
Re-review all six pins on a recurring cadence (e.g. every release, or
quarterly) rather than only at initial pin time.

---

## 8. No dependency/SAST scanning in CI — MEDIUM — flagged

**Where:** `.github/workflows/ci.yml` (added this session by the
DevOps/MLOps agent).

The new CI pipeline runs tests, coverage, Black/isort/Flake8/Pylint, and
mypy — good baseline hygiene — but has no step that would have caught
Findings 1–2 automatically (e.g. `bandit` for the unsafe-deserialization
pattern, `pip-audit` or `safety` for known-CVE dependency versions).

**Recommended fix:** add a `security` job to `ci.yml` (or a separate
`security.yml` alongside the existing `ci.yml`/`notebook.yml`):
```yaml
- run: pip install pip-audit bandit
- run: pip-audit -r <(pip freeze)
- run: bandit -r src/ -ll
```
Not added in this review to avoid conflicting with the DevOps agent's
in-flight CI work — flagged for that agent/DevOps to land.

---

## 9. `os.system()` for launching torchrun — LOW — flagged

**Where:** `kaggle_dual_gpu_finetune.ipynb`, Cells 10 and 12:
```python
_cmd = f'{sys.executable} -m torch.distributed.run --standalone --nproc_per_node={n_gpus or 2} --master_port=29500 /tmp/slm_worker.py'
_ret = os.system(_cmd)
```

`os.system()` runs through a shell, so any variable interpolated into
`_cmd` is a potential shell-injection vector if it ever becomes attacker-
influenced. Today it isn't — `sys.executable` and `n_gpus` (an `int`
derived from counting `nvidia-smi` output lines) are not attacker-
controlled in the current design — so this is a hardening suggestion, not
an active vulnerability.

**Recommended fix:** `subprocess.run([sys.executable, "-m",
"torch.distributed.run", "--standalone", f"--nproc_per_node={n_gpus or 2}",
"--master_port=29500", "/tmp/slm_worker.py"], check=True)` — avoids the
shell entirely and gives a proper `CalledProcessError` instead of a manual
exit-code check.

---

## 10. Wikipedia dump temp files never cleaned up — LOW — flagged

**Where:** `src/my_slm/create_t_f.py:109`:
```python
# Uncomment next line to remove temp files automatically:
# shutil.rmtree(tmpdir)
```

`wikidump2txt.py` downloads a multi-gigabyte `.bz2` Wikipedia dump and
extracts it to a temp directory that is never deleted by default. Not a
classic vulnerability, but repeated runs silently accumulate multi-GB
files — a disk-exhaustion / self-inflicted-DoS risk on constrained
environments, and worth fixing as part of the "temp file cleanup" audit
requirement.

**Recommended fix:** clean up by default, with an opt-out flag
(`--keep-tmp`) for debugging, rather than opt-in cleanup via a commented-
out line.

---

## 11. `.gitignore` doesn't explicitly exclude `*.pt` or `kaggle.json` — LOW — flagged

**Where:** `.gitignore`.

`models/*` is ignored (checkpoints saved there are covered), and `*.ckpt`,
`*.pkl`, `*.pkl.gz`, `*.bin` are ignored, but a bare `*.pt` is not — a
checkpoint saved outside `models/` (e.g. accidentally in the repo root
during local experimentation) would not be caught. `kaggle.json` (the
Kaggle API credential file referenced in `DEPLOYMENT.md`) also has no
explicit ignore rule. Risk is partially mitigated by the new pre-commit
hook `check-added-large-files --maxkb=1000`, which would catch large
checkpoint files (but not a small `kaggle.json`, which is only a few
hundred bytes).

**Recommended fix:** add `*.pt`, `kaggle.json`, and `.kaggle/` to
`.gitignore`.

**Not fixed directly:** `.gitignore` was under active edit by another
agent throughout this review — flagged rather than patched to avoid a
collision on a small, frequently-touched file.

---

## 12. `validate_path()` normalizes but does not sandbox — INFO / accepted risk

**Where:** `src/my_slm/utils.py:221`, `validate_path()`:
```python
def validate_path(path_str: str, must_exist: bool = True) -> Path:
    """Validate and normalize file path (prevents path traversal)."""
    p = Path(path_str).resolve()
    ...
```

The docstring says "prevents path traversal," but `.resolve()` alone only
*normalizes* a path (collapses `..` segments) — it does not restrict the
result to living under any particular base directory, so a caller passing
`"../../etc/passwd"` would resolve successfully rather than being
rejected. This is not exploitable today because every call site
(`load_model_safely`, notebook `MODEL_PATH`/`TOKENIZER_PATH`/`MODELS_DIR`)
is a path the notebook operator sets themselves in their own config cell —
there is no API or external input that supplies these paths on the
operator's behalf. Recorded here so the docstring's claim doesn't get
trusted beyond what the function actually does if a network-facing use
case is added later (e.g. a future inference API that accepts a model
name/path from a request).

**Recommendation (no code change needed today):** either soften the
docstring to "normalizes a path" rather than "prevents path traversal," or
add an actual `base_dir` allow-list parameter before this function is ever
used with externally-supplied input.

---

## 13. Secrets / credential exposure — INFO — verified clean

- No hardcoded API keys, tokens, or passwords found in any `.py` or
  `.ipynb` file (`grep` for `api_key|secret|password|Bearer|HF_TOKEN|
  KAGGLE_KEY|WANDB` across `src/` returned only docstring/variable-name
  matches referring to "HuggingFace", not actual secrets).
- A local `.env` file exists in the working tree but is correctly excluded
  by `.gitignore` (`*.env`) and confirmed **not tracked** via `git
  ls-files` / `git check-ignore -v .env`. Its contents were not read as
  part of this audit (not necessary to confirm it isn't committed).
- `git log --all` for `*.env*`, `*credential*`, `*.pem`, `kaggle.json`
  returned no history — no evidence these were ever committed and later
  removed.

---

## Dependency versions reviewed

| Package | Pinned (notebook) | pyproject.toml floor | Notes |
|---|---|---|---|
| torch | 2.6.0 | `>=2.6.0` (bumped this review) | Patches CVE-2025-32434 |
| transformers | 4.41.0 | — | Predates ReDoS fixes in 4.50.0; recommend bump (Finding 7) |
| datasets | 2.20.0 | — | No known critical CVE found for this version |
| huggingface_hub | 0.23.0 | — | No known critical CVE found for this version |
| accelerate | 0.31.0 | — | No known critical CVE found for this version |
| bitsandbytes | 0.43.1 | — | No known critical CVE found for this version |
| galore-torch | 0.2.0 | — | Small, low-adoption package; no CVE database entry found — recommend `pip-audit` as part of CI (Finding 8) rather than relying on manual review |
| numpy | — | `>=1.21` (no upper bound) | Fine for now; add upper bound or lock file for reproducibility |
| tqdm | — | `>=4.60` (no upper bound) | Low risk |
| matplotlib | — | `>=3.5` (no upper bound) | Low risk |

*CVE checks above were done via targeted web search during this review,
not a live vulnerability database query — treat "no known critical CVE
found" as "not found in this review," not as a guarantee, and re-verify
with `pip-audit` before release (Finding 8).*

## Fixes applied this session

1. `pyproject.toml` — `torch>=2.0` → `torch>=2.6.0` (Finding 2).
2. `src/my_slm/train.py` — `load_latest_checkpoint._load_state()` no
   longer silently retries `torch.load()` without `weights_only=True` on
   any exception; only the genuine old-PyTorch `TypeError` case falls back,
   with a loud warning (Finding 1).
3. `src/my_slm/semantic_eval.py` — same fix applied to
   `load_model_and_tok()` (Finding 1).
4. `SECURITY.md` and this file added.

All edits were verified with `python -m py_compile` after changes.

## Open items for Backend Dev / DevOps (not fixed in this session)

- Finding 3 (Kaggle notebook `_generate()` — highest-priority open item,
  one-line fix).
- Finding 4 (tokenizer pickle → JSON).
- Finding 5 (pin `git+https://...` install to a commit/tag).
- Finding 6 (verify/remove `trust_remote_code=True` for BLiMP).
- Finding 7 (bump `transformers` pin).
- Finding 8 (add `pip-audit`/`bandit` CI job).
- Finding 9 (`os.system` → `subprocess.run` list form).
- Finding 10 (default-clean wikidump temp files).
- Finding 11 (`.gitignore`: `*.pt`, `kaggle.json`).
