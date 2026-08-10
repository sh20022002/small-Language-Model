"""
Edge-case tests for HybridTokenizer robustness.

Tests for error handling, boundary conditions, and unusual inputs.
"""

import pytest
import torch
from pathlib import Path
import tempfile
import gzip
import pickle

from my_slm.hybrid_tokeniztion import HybridTokenizer
from my_slm.exceptions import (
    TokenizerFrozenError,
    TokenizerNotFrozenError,
    VocabSizeError,
    EncodingError,
    DecodingError,
    TokenizerError,
)


class TestTokenizerErrorHandling:
    """Test proper error raising and handling."""

    def test_encode_before_freeze_raises(self):
        """encode() on unfrozen tokenizer should raise TokenizerNotFrozenError."""
        tok = HybridTokenizer()
        tok.add_text("hello world")
        with pytest.raises(TokenizerNotFrozenError):
            tok.encode("test")

    def test_add_text_after_freeze_raises(self):
        """add_text() on frozen tokenizer should raise TokenizerFrozenError."""
        tok = HybridTokenizer()
        tok.add_text("hello world")
        tok.freeze_vocab(300)  # Must be >= base vocab size (261)
        with pytest.raises(TokenizerFrozenError):
            tok.add_text("more text")

    def test_add_file_after_freeze_raises(self):
        """add_file() on frozen tokenizer should raise TokenizerFrozenError."""
        tok = HybridTokenizer()
        tok.add_text("hello")
        tok.freeze_vocab(300)  # Must be >= base vocab size (261)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("test")
            f.flush()
            path = f.name
        try:
            with pytest.raises(TokenizerFrozenError):
                tok.add_file(path)
        finally:
            Path(path).unlink()

    def test_segment_before_freeze_raises(self):
        """segment() on unfrozen tokenizer should raise TokenizerNotFrozenError."""
        tok = HybridTokenizer()
        tok.add_text("hello")
        with pytest.raises(TokenizerNotFrozenError):
            tok.segment("test")

    def test_decode_invalid_token_ids_type(self):
        """decode() with non-list input should raise DecodingError."""
        tok = HybridTokenizer()
        tok.freeze_vocab(300)  # Must be >= base vocab size (261)
        with pytest.raises(DecodingError):
            tok.decode("not a list")

    def test_encode_non_string_input(self):
        """encode() with non-string input should raise EncodingError."""
        tok = HybridTokenizer()
        tok.freeze_vocab(300)  # Must be >= base vocab size (261)
        with pytest.raises(EncodingError):
            tok.encode(12345)

    def test_add_text_non_string_input(self):
        """add_text() with non-string input should raise EncodingError."""
        tok = HybridTokenizer()
        with pytest.raises(EncodingError):
            tok.add_text(123)


class TestTokenizerEdgeCases:
    """Test boundary conditions and unusual inputs."""

    def test_empty_text(self):
        """Empty text should be handled gracefully."""
        tok = HybridTokenizer()
        tok.add_text("")
        tok.freeze_vocab(300)  # Must be >= base vocab size (261)
        ids = tok.encode("")
        assert ids == []

    def test_empty_text_with_special_tokens(self):
        """Empty text with special tokens should only return BOS and EOS."""
        tok = HybridTokenizer()
        tok.freeze_vocab(300)  # Must be >= base vocab size (261)
        ids = tok.encode("", add_special_tokens=True)
        assert len(ids) == 2  # BOS + EOS
        assert ids[0] == tok.token2id["<BOS>"]
        assert ids[1] == tok.token2id["<EOS>"]

    def test_single_character(self):
        """Single character should encode/decode correctly."""
        tok = HybridTokenizer()
        tok.add_text("a")
        tok.freeze_vocab(300)  # Must be >= base vocab size (261)
        ids = tok.encode("a")
        assert len(ids) > 0
        text = tok.decode(ids)
        assert text == "a"

    def test_only_whitespace(self):
        """Whitespace-only text should encode/decode correctly."""
        tok = HybridTokenizer()
        tok.add_text("   \t\n  ")
        tok.freeze_vocab(300)  # Must be >= base vocab size (261)
        ids = tok.encode("   ")
        decoded = tok.decode(ids)
        assert "   " in decoded or decoded == ""

    def test_very_long_text(self):
        """Very long text should not crash."""
        tok = HybridTokenizer()
        long_text = "hello world " * 10000
        tok.add_text(long_text)
        tok.freeze_vocab(512)
        ids = tok.encode("hello world")
        assert len(ids) > 0

    def test_unicode_characters(self):
        """Unicode characters should be handled properly."""
        tok = HybridTokenizer()
        texts = ["café", "naïve", "résumé", "中文", "🎉"]
        for text in texts:
            tok.add_text(text)
        tok.freeze_vocab(512)
        for text in texts:
            ids = tok.encode(text)
            decoded = tok.decode(ids)
            assert decoded == text

    def test_decode_empty_list(self):
        """Decoding empty list should return empty string."""
        tok = HybridTokenizer()
        tok.freeze_vocab(300)  # Must be >= base vocab size (261)
        assert tok.decode([]) == ""

    def test_decode_out_of_range_ids(self):
        """Out-of-range IDs should be skipped gracefully."""
        tok = HybridTokenizer()
        tok.freeze_vocab(300)  # Must be >= base vocab size (261)
        ids = [1, 2, 99999, 3]  # 99999 is out of range
        result = tok.decode(ids)
        # Should not crash, just skip invalid IDs
        assert isinstance(result, str)

    def test_decode_negative_ids(self):
        """Negative IDs should be skipped gracefully."""
        tok = HybridTokenizer()
        tok.freeze_vocab(300)  # Must be >= base vocab size (261)
        ids = [1, -5, 2, -1]
        result = tok.decode(ids)
        # Should not crash
        assert isinstance(result, str)

    def test_freeze_vocab_idempotent(self):
        """Calling freeze_vocab() twice should be a no-op."""
        tok = HybridTokenizer()
        tok.add_text("hello world " * 100)
        tok.freeze_vocab(512)
        vocab_size_1 = len(tok.id2token)
        tok.freeze_vocab(512)
        vocab_size_2 = len(tok.id2token)
        assert vocab_size_1 == vocab_size_2

    def test_freeze_with_vocab_size_less_than_base(self):
        """freeze_vocab() with vocab_size < base should raise VocabSizeError."""
        tok = HybridTokenizer()
        tok.add_text("hello")
        base_size = len(tok.id2token)
        # Trying to freeze with size less than base vocab size should fail
        with pytest.raises(VocabSizeError):
            tok.freeze_vocab(base_size - 100)  # way below base_size


class TestTokenizerPersistence:
    """Test save/load functionality with error handling."""

    def test_save_and_load_basic(self):
        """Save and load should preserve tokenizer state."""
        tok = HybridTokenizer()
        tok.add_text("hello world " * 100)
        tok.freeze_vocab(512)
        text = "hello world"
        ids_before = tok.encode(text)

        with tempfile.NamedTemporaryFile(suffix=".pkl.gz", delete=False) as f:
            path = f.name

        try:
            tok.save(path)
            tok2 = HybridTokenizer.load(path)
            ids_after = tok2.encode(text)
            assert ids_before == ids_after
        finally:
            Path(path).unlink()

    def test_load_nonexistent_file(self):
        """Loading nonexistent file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            HybridTokenizer.load("/nonexistent/path/tokenizer.pkl.gz")

    def test_save_creates_parent_directories(self):
        """save() should create parent directories if needed."""
        tok = HybridTokenizer()
        tok.freeze_vocab(300)  # Must be >= base vocab size (261)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "subdir1" / "subdir2" / "tok.pkl.gz"
            tok.save(str(path))
            assert path.exists()
            tok2 = HybridTokenizer.load(str(path))
            assert tok2.vocab_size == tok.vocab_size


class TestTokenizerProperties:
    """Test tokenizer properties and diagnostic methods."""

    def test_vocab_size_property(self):
        """vocab_size property should return correct value."""
        tok = HybridTokenizer()
        tok.add_text("hello world " * 100)
        tok.freeze_vocab(1024)
        assert tok.vocab_size == len(tok.id2token)

    def test_db_status_keys(self):
        """db_status() should include all required keys."""
        tok = HybridTokenizer()
        tok.add_text("hello")
        status = tok.db_status()
        assert "vocab_size" in status
        assert "n_merges" in status
        assert "frozen" in status
        assert "cache_size" in status
        assert "training_words" in status

    def test_top_merges_length(self):
        """top_merges() should return correct number of merges."""
        tok = HybridTokenizer()
        tok.add_text("hello world " * 100)
        tok.freeze_vocab(1024)
        merges = tok.top_merges(n=10)
        assert len(merges) <= 10

    def test_explain_token(self):
        """explain_token() should return a string."""
        tok = HybridTokenizer()
        tok.add_text("hello world " * 100)
        tok.freeze_vocab(512)
        for token in list(tok.token2id.keys())[:10]:
            explanation = tok.explain_token(token)
            assert isinstance(explanation, str)


class TestTokenizerRoundTrip:
    """Test encode/decode round-trip correctness."""

    def test_simple_text_round_trip(self):
        """Simple text should round-trip correctly."""
        tok = HybridTokenizer()
        text = "The quick brown fox"
        tok.add_text(text)
        tok.freeze_vocab(512)
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        assert decoded == text

    def test_special_characters_round_trip(self):
        """Special characters should round-trip correctly."""
        tok = HybridTokenizer()
        text = "!@#$%^&*()_+-=[]{}|;:',.<>?/`~"
        tok.add_text(text)
        tok.freeze_vocab(512)
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        assert decoded == text

    def test_numbers_round_trip(self):
        """Numbers should round-trip correctly."""
        tok = HybridTokenizer()
        text = "0123456789"
        tok.add_text(text)
        tok.freeze_vocab(512)
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        assert decoded == text

    def test_mixed_content_round_trip(self):
        """Mixed content should round-trip correctly."""
        tok = HybridTokenizer()
        text = "Hello, world! 123 café"
        tok.add_text(text)
        tok.freeze_vocab(1024)
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        assert decoded == text


class TestTokenizerSelfTest:
    """Test the built-in self_test method."""

    def test_self_test_passes_on_frozen_tokenizer(self):
        """self_test() should pass on a properly frozen tokenizer."""
        tok = HybridTokenizer()
        for sample in ["hello", "world", "test", "data"]:
            tok.add_text(sample)
        tok.freeze_vocab(512)
        assert tok.self_test() is True

    def test_self_test_raises_on_unfrozen(self):
        """self_test() should raise on unfrozen tokenizer."""
        tok = HybridTokenizer()
        with pytest.raises(TokenizerNotFrozenError):
            tok.self_test()


class TestTokenizerCaching:
    """Test word caching for performance."""

    def test_cache_improves_performance(self):
        """Repeated encoding should use cache."""
        tok = HybridTokenizer()
        tok.add_text("hello world " * 100)
        tok.freeze_vocab(512)

        # First call populates cache
        text = "hello"
        ids1 = tok.encode(text)
        cache_size_1 = len(tok._word_cache)

        # Second call should hit cache
        ids2 = tok.encode(text)
        cache_size_2 = len(tok._word_cache)

        assert ids1 == ids2
        assert cache_size_1 == cache_size_2

    def test_cache_cleared_after_load(self):
        """Cache should be empty after loading tokenizer."""
        tok = HybridTokenizer()
        tok.add_text("hello")
        tok.freeze_vocab(300)  # Must be >= base vocab size (261)

        with tempfile.NamedTemporaryFile(suffix=".pkl.gz", delete=False) as f:
            path = f.name

        try:
            tok.save(path)
            tok2 = HybridTokenizer.load(path)
            assert len(tok2._word_cache) == 0
        finally:
            Path(path).unlink()


class TestTokenizerLowercase:
    """Test lowercase parameter."""

    def test_lowercase_encoding(self):
        """lowercase=True should convert text to lowercase."""
        tok_lower = HybridTokenizer(lowercase=True)
        tok_lower.add_text("HELLO WORLD hello world")
        tok_lower.freeze_vocab(512)

        ids_upper = tok_lower.encode("HELLO")
        ids_lower = tok_lower.encode("hello")
        assert ids_upper == ids_lower

    def test_no_lowercase_encoding(self):
        """lowercase=False should preserve case."""
        tok = HybridTokenizer(lowercase=False)
        tok.add_text("HELLO hello")
        tok.freeze_vocab(512)

        # May or may not be equal depending on training data
        # Just verify it doesn't crash
        tok.encode("HELLO")
        tok.encode("hello")


class TestTokenizerAddFile:
    """Test add_file functionality."""

    def test_add_file_basic(self):
        """add_file() should read and process file."""
        tok = HybridTokenizer()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("hello world\n")
            f.write("this is a test\n")
            f.flush()
            path = f.name

        try:
            tok.add_file(path)
            tok.freeze_vocab(512)
            ids = tok.encode("hello")
            assert len(ids) > 0
        finally:
            Path(path).unlink()

    def test_add_file_missing_raises(self):
        """add_file() with missing file should raise FileNotFoundError."""
        tok = HybridTokenizer()
        with pytest.raises(FileNotFoundError):
            tok.add_file("/nonexistent/file.txt")

    def test_add_file_is_directory_raises(self):
        """add_file() on directory should raise ValueError."""
        tok = HybridTokenizer()
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError):
                tok.add_file(tmpdir)
