"""Shared utilities for tokenization, I/O, and checkpoint management."""

import json
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# Tokenizer helpers — unified for HybridTokenizer and HuggingFace tokenizers
# ─────────────────────────────────────────────────────────────────────────────

def encode(tokenizer, text: str, max_len: Optional[int] = None) -> List[int]:
    """
    Encode text with either HybridTokenizer or HuggingFace tokenizer.

    Appends EOS to every sequence so the model learns when to stop generating.
    max_len reserves one slot for EOS, keeping total length within the window.
    """
    if hasattr(tokenizer, "token2id"):
        # HybridTokenizer path
        ids = tokenizer.encode(text, add_special_tokens=False)
        eos = tokenizer.token2id.get("<EOS>")
        if max_len and ids:
            ids = ids[: max_len - 1]
        if eos is not None:
            ids = ids + [eos]
        return ids
    else:
        # HuggingFace tokenizer path
        kwargs = {"add_special_tokens": False}
        if max_len:
            kwargs["truncation"] = True
            kwargs["max_length"] = max_len - 1
        ids = tokenizer.encode(text, **kwargs)
        eos_id = getattr(tokenizer, "eos_token_id", None)
        if eos_id is not None:
            ids = ids + [eos_id]
        return ids


def decode(tokenizer, ids: List[int]) -> str:
    """Decode token IDs back to text."""
    if hasattr(tokenizer, "token2id"):
        return tokenizer.decode(ids)
    return tokenizer.decode(ids, skip_special_tokens=True)


def get_pad_token_id(tokenizer) -> int:
    """Return pad token ID for HybridTokenizer or HuggingFace tokenizer."""
    if hasattr(tokenizer, "token2id"):
        return tokenizer.token2id.get("<PAD>", 0)
    if hasattr(tokenizer, "pad_token_id") and tokenizer.pad_token_id is not None:
        return tokenizer.pad_token_id
    return 0


def get_eos_token_id(tokenizer) -> int:
    """Return EOS token ID for HybridTokenizer or HuggingFace tokenizer."""
    if hasattr(tokenizer, "token2id"):
        return tokenizer.token2id.get("<EOS>", 2)
    if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
        return tokenizer.eos_token_id
    return 2


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint management
# ─────────────────────────────────────────────────────────────────────────────

CHECKPOINT_SCHEMA_VERSION = 1


def compute_tokenizer_hash(tokenizer) -> str:
    """Compute SHA256 hash of tokenizer vocab for cross-checking."""
    if hasattr(tokenizer, "id2token"):
        # HybridTokenizer
        vocab_str = json.dumps(
            {str(k): v for k, v in enumerate(tokenizer.id2token)},
            sort_keys=True
        )
    else:
        # HuggingFace tokenizer
        vocab_str = json.dumps(tokenizer.vocab, sort_keys=True)
    return hashlib.sha256(vocab_str.encode()).hexdigest()[:16]


def save_checkpoint(
    model: nn.Module,
    config: Dict[str, Any],
    out_dir: Path,
    optimizer: Optional[torch.optim.Optimizer] = None,
    trainer_state: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save model checkpoint with config, weights, optimizer state, and metadata.

    Args:
        model: The model to save
        config: Model architecture config (vocab_size, dim, depth, etc.)
        out_dir: Output directory for checkpoint
        optimizer: Optional optimizer to save state
        trainer_state: Optional training metadata (step, epoch, losses, etc.)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config["schema_version"] = CHECKPOINT_SCHEMA_VERSION

    # Save config
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    # Save model weights (handle tied weights)
    state_dict = model.state_dict()
    if config.get("tie_weights", False):
        # Drop to_logits.weight if it's tied to token_emb.weight
        if "to_logits.weight" in state_dict and "token_emb.weight" in state_dict:
            if state_dict["to_logits.weight"].data_ptr() == state_dict["token_emb.weight"].data_ptr():
                state_dict.pop("to_logits.weight", None)

    try:
        import safetensors.torch
        safetensors.torch.save_file(state_dict, out_dir / "model.safetensors")
    except ImportError:
        torch.save(state_dict, out_dir / "model.pt")

    # Save trainer state
    if trainer_state is not None:
        (out_dir / "trainer_state.json").write_text(json.dumps(trainer_state, indent=2))

    # Save optimizer (pickle is ok here, never loaded untrusted)
    if optimizer is not None:
        torch.save(optimizer.state_dict(), out_dir / "optimizer.pt")


def load_checkpoint(
    ckpt_dir: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu",
    strict: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Load model checkpoint, verifying config matches architecture.

    Args:
        ckpt_dir: Checkpoint directory
        model: Model instance to load state into
        optimizer: Optional optimizer to restore state
        device: Device to load to
        strict: If True, fail on config/model mismatch

    Returns:
        (config, trainer_state) dicts
    """
    ckpt_dir = Path(ckpt_dir)

    # Load config
    config_path = ckpt_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"No config.json in {ckpt_dir}")

    config = json.loads(config_path.read_text())

    # Validate schema version
    if config.get("schema_version", 1) != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported checkpoint version: {config.get('schema_version')}")

    # Load weights
    weights_path = ckpt_dir / "model.safetensors"
    if not weights_path.exists():
        weights_path = ckpt_dir / "model.pt"

    if weights_path.suffix == ".safetensors":
        try:
            import safetensors.torch
            state_dict = safetensors.torch.load_file(weights_path)
        except ImportError:
            state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    else:
        state_dict = torch.load(weights_path, map_location=device, weights_only=True)

    # Load state dict with logging
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    # Re-tie weights if applicable
    if config.get("tie_weights", False):
        if hasattr(model, "to_logits") and hasattr(model, "token_emb"):
            model.to_logits.weight = model.token_emb.weight

    # Log any mismatches
    if missing or unexpected:
        print(f"⚠️  Checkpoint/model mismatch:")
        if missing:
            print(f"   Missing keys: {missing}")
        if unexpected:
            print(f"   Unexpected keys: {unexpected}")

    if strict and (missing or unexpected):
        raise RuntimeError(f"Checkpoint/model mismatch (strict=True)")

    # Load trainer state
    trainer_state = {}
    state_path = ckpt_dir / "trainer_state.json"
    if state_path.exists():
        trainer_state = json.loads(state_path.read_text())

    # Load optimizer
    opt_path = ckpt_dir / "optimizer.pt"
    if optimizer is not None and opt_path.exists():
        try:
            optimizer.load_state_dict(torch.load(opt_path, map_location="cpu", weights_only=True))
        except Exception as e:
            print(f"⚠️  Could not restore optimizer state: {e}")

    return config, trainer_state


def validate_path(path_str: str, must_exist: bool = True) -> Path:
    """Validate and normalize file path (prevents path traversal)."""
    p = Path(path_str).resolve()
    if must_exist and not p.exists():
        raise FileNotFoundError(f"Path does not exist: {p}")
    if p.is_dir():
        raise ValueError(f"Expected file, got directory: {p}")
    return p


def load_model_safely(model_path: str, model_class, device: str = "cpu", **model_kwargs):
    """
    Load model checkpoint safely with validation.

    Args:
        model_path: Path to checkpoint
        model_class: Model class to instantiate
        device: Device to load to
        **model_kwargs: Additional model constructor arguments

    Returns:
        model: Loaded model
    """
    ckpt_path = validate_path(model_path)

    # Handle directory (new format) or file (old format)
    if ckpt_path.is_dir():
        model = model_class(**model_kwargs).to(device)
        config, _ = load_checkpoint(ckpt_path, model, device=device)
        return model
    else:
        # Old format: bare state dict
        model = model_class(**model_kwargs).to(device)
        state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict, strict=False)
        return model
