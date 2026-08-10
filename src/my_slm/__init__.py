"""Small Language Model package."""

from my_slm.transformer import Transformer
from my_slm.hybrid_tokeniztion import HybridTokenizer
from my_slm.utils import (
    encode,
    decode,
    get_pad_token_id,
    get_eos_token_id,
    save_checkpoint,
    load_checkpoint,
)
from my_slm.exceptions import (
    SLMException,
    TokenizerError,
    TokenizationError,
    TokenizerNotFrozenError,
    TokenizerFrozenError,
    VocabSizeError,
    EncodingError,
    DecodingError,
    TrainingError,
    DatasetError,
    CheckpointError,
    CheckpointNotFoundError,
    ConfigMismatchError,
)

__all__ = [
    "Transformer",
    "HybridTokenizer",
    "encode",
    "decode",
    "get_pad_token_id",
    "get_eos_token_id",
    "save_checkpoint",
    "load_checkpoint",
    "SLMException",
    "TokenizerError",
    "TokenizationError",
    "TokenizerNotFrozenError",
    "TokenizerFrozenError",
    "VocabSizeError",
    "EncodingError",
    "DecodingError",
    "TrainingError",
    "DatasetError",
    "CheckpointError",
    "CheckpointNotFoundError",
    "ConfigMismatchError",
]
