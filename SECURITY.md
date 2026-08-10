# Security Policy

This document describes the security posture of the small-Language-Model /
`hybrid-tokenizer` project: what it does and does not protect against, how
dependencies are managed, and how to report a vulnerability.

For the full findings from the most recent audit (severities, evidence,
remediation status), see [SECURITY_AUDIT.md](SECURITY_AUDIT.md).

## Threat model / known limitations

This is a research/training codebase, not a hosted service. There is no
network-facing API, no authentication layer, and no multi-tenant database —
most classic web-app vulnerability classes (SQLi, CSRF, session hijacking,
broken auth) do not apply because there is nothing here that plays those
roles. The realistic attack surface is narrower and specific to ML tooling:

- **Untrusted checkpoints / tokenizer files.** `torch.load()` and
  `pickle.load()` deserialize Python objects. A `.pt` or `.pkl.gz` file is
  not inert data — a maliciously crafted one can execute arbitrary code the
  moment it's loaded. This project loads such files from Kaggle dataset
  paths (`/kaggle/input/...`) that the notebook operator chooses but does
  not necessarily control the contents of (e.g. a dataset shared by someone
  else, or one attached from a public Kaggle listing). Treat every `.pt` /
  `.pkl.gz` file from outside your own training run as untrusted input.
- **Untrusted HF Hub dataset/model identifiers.** `datasets.load_dataset()`
  and `AutoTokenizer.from_pretrained()` fetch code and data from the
  Hugging Face Hub. `trust_remote_code=True` (used once, for BLiMP
  evaluation) explicitly permits the Hub to run Python code on your
  machine. Only ever point these at identifiers you trust.
- **Unpinned installs.** The Kaggle notebook installs this package with
  `pip install git+https://github.com/sh20022002/small-Language-Model.git`
  with no commit/tag pin, so every run fetches whatever is on the default
  branch *right now*. Anyone with push access (or a compromised commit)
  changes what every subsequent Kaggle session executes.
- **Not sandboxed.** Kaggle notebooks run with the permissions of the
  Kaggle kernel (GPU access, outbound internet, `/kaggle/working` write
  access). Code here does not add its own sandboxing — it inherits
  whatever Kaggle provides.
- **No secrets are expected in this repo.** There is no code path that
  reads or embeds API keys, tokens, or credentials. `.env` is gitignored
  and was verified not to be tracked. If you add integrations that need
  secrets (W&B, HF Hub write tokens, Kaggle API keys), use environment
  variables or Kaggle Secrets — never hardcode them.

## Data privacy stance

- Training data comes from public datasets (TinyStories, WikiText, C4,
  OpenWebText, Alpaca, Dolly, GSM8K, OpenOrca, Anthropic HH-RLHF, etc.)
  streamed from the Hugging Face Hub. No user-submitted or PII data is
  collected, stored, or transmitted by this codebase.
- Checkpoints and logs (`benchmarks.csv`, loss-curve PNGs, `trainer_state.json`)
  contain only training metrics and model weights — no user input is
  captured or persisted.
- `create_t_f.py` (Wikipedia dump extractor) downloads public Wikipedia
  dumps over HTTPS and does not authenticate or transmit anything.
- Anyone attaching a private/custom dataset on Kaggle is responsible for
  that dataset's own privacy handling; this project does not add any
  additional collection, logging, or exfiltration of that data.

## Dependency management

- Library dependencies are declared in `pyproject.toml`. `torch>=2.6.0` is
  a hard floor (not just a preference) because of
  [CVE-2025-32434](https://github.com/pytorch/pytorch/security/advisories/GHSA-53q9-r3pm-6pq6),
  a critical `torch.load()` RCE that affects versions ≤2.5.1 even when
  `weights_only=True` is used.
- The Kaggle training notebook additionally pins exact versions for
  `accelerate`, `bitsandbytes`, `datasets`, `transformers`,
  `huggingface_hub`, and `galore-torch` in its install cell, for
  reproducibility. These pins should be reviewed periodically — pinning
  once and never revisiting a version means known CVEs patched upstream
  stay unpatched here. See SECURITY_AUDIT.md for the current recommended
  bumps.
- Run `pip-audit` (or `safety check`) against `pyproject.toml`
  periodically, and before any release, to catch newly disclosed CVEs in
  transitive dependencies:
  ```bash
  pip install pip-audit
  pip-audit -r <(pip freeze)
  ```
- No dependency lock file is currently checked in. For a training
  pipeline this is a lower-severity gap than it would be for a deployed
  service, but reproducibility and supply-chain integrity both improve
  with one (`pip-compile`, `poetry.lock`, or `uv.lock`).

## Safe deserialization rules for this codebase

1. **Never** call `torch.load(path)` without `weights_only=True` as a
   silent fallback when the safe load fails. If `weights_only=True`
   raises, that is the file telling you it contains something it
   shouldn't — stop and inspect it, don't retry unsafely.
2. Prefer `safetensors` over pickle-based formats for anything written by
   this project (`utils.save_checkpoint()` already does this when the
   `safetensors` package is available).
3. Tokenizer vocabularies (`HybridTokenizer.save/load`) currently use
   `pickle` even though the payload is plain `str`/`int`/`list`/`dict`
   data with no need for arbitrary object graphs. This is tracked as a
   finding in SECURITY_AUDIT.md — prefer `json` for new persistence code
   where the schema is this simple.

## Reporting a vulnerability

If you find a security issue in this project:

1. **Do not open a public GitHub issue for it.**
2. Email the maintainer directly: **shmuel.tor@gmail.com** with a
   description, reproduction steps, and (if applicable) a proof-of-concept.
3. Allow a reasonable window to investigate and patch before any public
   disclosure. This is a small, single-maintainer research project — please
   be patient, but reports are taken seriously and will be acted on.

## Scope note

This project trains and evaluates a small transformer language model. It
does not implement content moderation, output filtering, or safety
alignment beyond whatever the underlying training data and objective
provide. Generated text should not be treated as vetted or safe for
unsupervised end-user exposure.
