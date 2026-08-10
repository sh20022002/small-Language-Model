"""
Custom exceptions for the SLM tokenizer and training pipeline.

Provides fine-grained error handling with context for debugging.
"""


class SLMException(Exception):
    """Base exception for all SLM errors."""

    pass


class TokenizerError(SLMException):
    """Raised when tokenizer encounters invalid state or operation."""

    pass


class TokenizationError(TokenizerError):
    """Raised when encoding/decoding fails."""

    pass


class TokenizerNotFrozenError(TokenizerError):
    """Raised when tokenizer operations require frozen state."""

    def __init__(self, operation: str = "encode"):
        msg = f"Cannot {operation}: tokenizer is not frozen. Call freeze_vocab() first."
        super().__init__(msg)


class TokenizerFrozenError(TokenizerError):
    """Raised when trying to modify a frozen tokenizer."""

    def __init__(self):
        super().__init__(
            "Tokenizer is frozen and cannot accept training data. "
            "Create a new tokenizer instance to train on different data."
        )


class VocabSizeError(TokenizerError):
    """Raised when vocab_size parameter is invalid."""

    def __init__(self, vocab_size: int, min_size: int = 256):
        msg = (
            f"Invalid vocab_size={vocab_size}. "
            f"Minimum is {min_size} (base vocabulary + merges)."
        )
        super().__init__(msg)


class EncodingError(TokenizationError):
    """Raised when text encoding fails."""

    def __init__(self, text: str, reason: str = "unknown"):
        snippet = text[:50] + "..." if len(text) > 50 else text
        super().__init__(f"Failed to encode text {snippet!r}: {reason}")


class DecodingError(TokenizationError):
    """Raised when token ID decoding fails."""

    def __init__(self, token_ids: list, reason: str = "unknown"):
        snippet = str(token_ids[:10]) + "..." if len(token_ids) > 10 else str(token_ids)
        super().__init__(f"Failed to decode token IDs {snippet}: {reason}")


class TrainingError(SLMException):
    """Raised when training encounters an error."""

    pass


class DatasetError(TrainingError):
    """Raised when dataset loading or processing fails."""

    def __init__(self, path: str, reason: str = "unknown"):
        super().__init__(f"Dataset error for {path!r}: {reason}")


class CheckpointError(TrainingError):
    """Raised when checkpoint operations fail."""

    def __init__(self, ckpt_path: str, reason: str = "unknown"):
        super().__init__(f"Checkpoint error at {ckpt_path!r}: {reason}")


class CheckpointNotFoundError(CheckpointError):
    """Raised when checkpoint file is missing."""

    def __init__(self, ckpt_path: str):
        super().__init__(ckpt_path, "checkpoint file not found")


class ConfigMismatchError(CheckpointError):
    """Raised when loaded config doesn't match model architecture."""

    def __init__(self, expected: dict, actual: dict, mismatches: list):
        msg = f"Config mismatch: {', '.join(mismatches)}"
        super().__init__("", msg)
