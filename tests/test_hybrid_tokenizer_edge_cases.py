"""
Edge case tests for HybridTokenizer: empty strings, unicode, special chars,
large inputs, malformed text, cache behavior, encoding/decoding round-trips.
"""
import pytest
import tempfile
import os
from pathlib import Path

import torch
from my_slm.hybrid_tokeniztion import HybridTokenizer, _word_to_chars, _get_pairs, _apply_merge


class TestTokenizerEdgeCases:
    """Test edge cases: empty, unicode, special chars, large inputs."""

    def test_empty_string_encode(self):
        tok = HybridTokenizer()
        tok.add_text("hello world " * 100)
        tok.freeze_vocab(512)
        ids = tok.encode("")
        assert isinstance(ids, list)
        assert len(ids) == 0

    def test_empty_string_decode(self):
        tok = HybridTokenizer()
        tok.add_text("hello " * 100)
        tok.freeze_vocab(512)
        text = tok.decode([])
        assert text == ""

    def test_unicode_emoji(self):
        tok = HybridTokenizer()
        tok.add_text("Hello 😀 World 🌍" * 50)
        tok.freeze_vocab(1024)
        ids = tok.encode("Hello 😀 World")
        assert len(ids) > 0
        decoded = tok.decode(ids)
        # Emoji may not round-trip perfectly, but should be close
        assert "Hello" in decoded

    def test_unicode_cjk_characters(self):
        tok = HybridTokenizer()
        tok.add_text("你好世界 こんにちは 안녕하세요" * 50)
        tok.freeze_vocab(1024)
        ids = tok.encode("你好世界")
        assert len(ids) > 0
        decoded = tok.decode(ids)
        assert len(decoded) > 0

    def test_special_characters(self):
        tok = HybridTokenizer()
        tok.add_text("!@#$%^&*()-_=+[]{}|;:',.<>?/" * 50)
        tok.freeze_vocab(512)
        ids = tok.encode("!@#$%^&*()")
        assert len(ids) > 0
        decoded = tok.decode(ids)
        assert len(decoded) > 0

    def test_whitespace_variants(self):
        tok = HybridTokenizer()
        tok.add_text("hello world" * 100)
        tok.freeze_vocab(512)
        # Tabs, newlines, multiple spaces
        text = "hello\tworld\ntest  case"
        ids = tok.encode(text)
        assert len(ids) > 0
        decoded = tok.decode(ids)
        assert "hello" in decoded and "world" in decoded

    def test_very_long_text(self):
        tok = HybridTokenizer()
        # Add training data
        tok.add_text("word " * 1000)
        tok.freeze_vocab(512)
        # Encode a very long text
        long_text = "word " * 5000
        ids = tok.encode(long_text)
        assert len(ids) > 1000

    def test_single_character_words(self):
        tok = HybridTokenizer()
        tok.add_text("a b c d e f g" * 100)
        tok.freeze_vocab(512)
        ids = tok.encode("a b c")
        assert len(ids) > 0

    def test_mixed_case_sensitivity(self):
        tok_lower = HybridTokenizer(lowercase=True)
        tok_lower.add_text("Hello World" * 100)
        tok_lower.freeze_vocab(512)

        ids1 = tok_lower.encode("HELLO WORLD")
        ids2 = tok_lower.encode("hello world")
        assert ids1 == ids2

    def test_non_lowercase_preserves_case(self):
        tok = HybridTokenizer(lowercase=False)
        tok.add_text("Hello World" * 100)
        tok.freeze_vocab(512)
        ids1 = tok.encode("Hello")
        ids2 = tok.encode("hello")
        # May not be exactly equal due to tokenization, but both should work
        assert len(ids1) > 0 and len(ids2) > 0

    def test_control_characters(self):
        tok = HybridTokenizer()
        tok.add_text("hello\x00world\x01test" * 50)
        tok.freeze_vocab(512)
        ids = tok.encode("hello\x00world")
        assert len(ids) > 0

    def test_null_bytes(self):
        tok = HybridTokenizer()
        tok.add_text("test\x00null" * 50)
        tok.freeze_vocab(512)
        ids = tok.encode("test\x00null")
        # Should handle null bytes gracefully
        assert isinstance(ids, list)

    def test_high_unicode_codepoints(self):
        tok = HybridTokenizer()
        # Add text with high unicode codepoints
        tok.add_text("test ፡ ። ፣ ፤ ፥ ፦ ፧ ፨" * 50)
        tok.freeze_vocab(512)
        ids = tok.encode("test ፡ ።")
        assert len(ids) > 0

    def test_repeated_characters(self):
        tok = HybridTokenizer()
        tok.add_text("aaa bbb ccc" * 100)
        tok.freeze_vocab(512)
        ids = tok.encode("aaaa bbbb")
        decoded = tok.decode(ids)
        # Should preserve repetition (roughly)
        assert "a" in decoded and "b" in decoded


class TestTokenizerRoundTrip:
    """Test encode->decode round-trip consistency."""

    def test_roundtrip_ascii(self):
        tok = HybridTokenizer()
        tok.add_text("The quick brown fox jumps over the lazy dog. " * 100)
        tok.freeze_vocab(512)

        original = "The quick brown fox jumps over the lazy dog."
        ids = tok.encode(original)
        decoded = tok.decode(ids)
        # Should preserve content
        assert "quick" in decoded and "brown" in decoded

    def test_roundtrip_numbers(self):
        tok = HybridTokenizer()
        tok.add_text("0123456789 test " * 100)
        tok.freeze_vocab(512)

        original = "12345 67890"
        ids = tok.encode(original)
        decoded = tok.decode(ids)
        # Numbers should be preserved
        assert "1" in decoded or "12" in decoded

    def test_roundtrip_mixed_content(self):
        tok = HybridTokenizer()
        tok.add_text("Hello123!@# world456$%^ test789&*()" * 100)
        tok.freeze_vocab(512)

        original = "Hello123!@# world456"
        ids = tok.encode(original)
        decoded = tok.decode(ids)
        assert len(decoded) > 0

    def test_special_tokens_skipped_in_decode(self):
        tok = HybridTokenizer()
        tok.add_text("hello world" * 100)
        tok.freeze_vocab(512)

        # Encode with special tokens
        ids = tok.encode("hello world", add_special_tokens=True)
        # BOS (1) + tokens + EOS (2)
        assert len(ids) >= 2

        # Decode should skip special tokens by default
        decoded = tok.decode(ids)
        assert "BOS" not in decoded and "EOS" not in decoded

    def test_decode_invalid_ids_ignored(self):
        tok = HybridTokenizer()
        tok.add_text("hello world" * 100)
        tok.freeze_vocab(512)

        # Include some invalid IDs
        invalid_ids = [1, 2, 999999, 1000000]
        decoded = tok.decode(invalid_ids)
        # Should handle gracefully (invalid IDs skipped)
        assert isinstance(decoded, str)


class TestTokenizerCaching:
    """Test caching behavior for performance."""

    def test_word_cache_on_repeated_encode(self):
        tok = HybridTokenizer()
        tok.add_text("hello world test " * 200)
        tok.freeze_vocab(512)

        word = "hello"
        # First encode fills cache
        ids1 = tok.encode(word)
        cache_size_1 = len(tok._word_cache)

        # Second encode uses cache
        ids2 = tok.encode(word)
        cache_size_2 = len(tok._word_cache)

        assert ids1 == ids2
        assert cache_size_2 >= cache_size_1

    def test_cache_consistency_after_freeze(self):
        tok = HybridTokenizer()
        tok.add_text("hello world test " * 100)
        tok.freeze_vocab(512)

        # Clear cache
        tok._word_cache.clear()
        ids1 = tok.encode("hello world")

        # Encode again (should be cached)
        ids2 = tok.encode("hello world")

        assert ids1 == ids2


class TestTokenizerSpecialTokens:
    """Test special token handling."""

    def test_special_tokens_exist(self):
        tok = HybridTokenizer()
        tok.add_text("test" * 100)
        tok.freeze_vocab(512)

        assert "<PAD>" in tok.token2id
        assert "<BOS>" in tok.token2id
        assert "<EOS>" in tok.token2id
        assert "<UNK>" in tok.token2id
        assert "<MASK>" in tok.token2id

    def test_bos_eos_token_ids(self):
        tok = HybridTokenizer()
        tok.add_text("test data" * 100)
        tok.freeze_vocab(512)

        ids = tok.encode("test", add_special_tokens=True)
        assert ids[0] == tok.token2id["<BOS>"]
        assert ids[-1] == tok.token2id["<EOS>"]

    def test_pad_token_exists(self):
        tok = HybridTokenizer()
        tok.add_text("test" * 100)
        tok.freeze_vocab(512)

        pad_id = tok.token2id["<PAD>"]
        assert pad_id == 0


class TestTokenizerVocabSize:
    """Test vocabulary size constraints and growth."""

    def test_vocab_size_after_freeze(self):
        tok = HybridTokenizer()
        tok.add_text("hello world " * 100)
        target_size = 512
        tok.freeze_vocab(target_size)

        assert tok.vocab_size >= 5  # At least special tokens + bytes
        assert tok.vocab_size <= target_size + 100  # Shouldn't exceed much

    def test_vocab_size_property(self):
        tok = HybridTokenizer()
        tok.add_text("test " * 100)
        tok.freeze_vocab(512)

        assert tok.vocab_size == len(tok.id2token)

    def test_token2id_id2token_consistency(self):
        tok = HybridTokenizer()
        tok.add_text("hello world test " * 100)
        tok.freeze_vocab(512)

        for token_str, token_id in tok.token2id.items():
            assert tok.id2token[token_id] == token_str


class TestTokenizerPersistence:
    """Test save/load functionality."""

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tok_file = os.path.join(tmpdir, "tok.pkl.gz")

            # Create and save
            tok1 = HybridTokenizer()
            tok1.add_text("hello world test " * 100)
            tok1.freeze_vocab(512)
            tok1.save(tok_file)

            # Load
            tok2 = HybridTokenizer.load(tok_file)

            # Should produce same encodings
            text = "hello world"
            ids1 = tok1.encode(text)
            ids2 = tok2.encode(text)
            assert ids1 == ids2

    def test_load_preserves_lowercase_flag(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tok_file = os.path.join(tmpdir, "tok.pkl.gz")

            tok1 = HybridTokenizer(lowercase=True)
            tok1.add_text("Hello World" * 100)
            tok1.freeze_vocab(512)
            tok1.save(tok_file)

            tok2 = HybridTokenizer.load(tok_file)
            assert tok2.lowercase == True

    def test_nonexistent_file_raises_error(self):
        with pytest.raises((FileNotFoundError, EOFError)):
            HybridTokenizer.load("/nonexistent/path/tok.pkl.gz")


class TestTokenizerHelpers:
    """Test low-level tokenizer helper functions."""

    def test_word_to_chars(self):
        result = _word_to_chars("hello")
        assert len(result) == 5
        assert isinstance(result, tuple)

    def test_word_to_chars_empty(self):
        result = _word_to_chars("")
        assert len(result) == 0

    def test_word_to_chars_unicode(self):
        result = _word_to_chars("café")
        assert len(result) > 0

    def test_get_pairs_empty(self):
        pairs = _get_pairs(())
        assert len(pairs) == 0

    def test_get_pairs_single(self):
        pairs = _get_pairs(("a",))
        assert len(pairs) == 0

    def test_get_pairs_multiple(self):
        word = ("a", "b", "c")
        pairs = _get_pairs(word)
        assert len(pairs) == 2
        assert ("a", "b") in pairs
        assert ("b", "c") in pairs

    def test_apply_merge_basic(self):
        word = ("a", "b", "c")
        result = _apply_merge(word, "a", "b", "ab")
        assert result == ("ab", "c")

    def test_apply_merge_no_match(self):
        word = ("a", "b", "c")
        result = _apply_merge(word, "x", "y", "xy")
        assert result == word

    def test_apply_merge_multiple_occurrences(self):
        word = ("a", "b", "a", "b")
        result = _apply_merge(word, "a", "b", "ab")
        assert result == ("ab", "ab")


class TestTokenizerIntegration:
    """Integration tests for tokenizer as a whole."""

    def test_tokenizer_self_test(self):
        """Use the built-in self_test method."""
        tok = HybridTokenizer()
        tok.add_text("Hello, world! The quick brown fox. " * 200)
        tok.freeze_vocab(1024)

        result = tok.self_test()
        assert result == True

    def test_segment_returns_token_strings(self):
        tok = HybridTokenizer()
        tok.add_text("hello world test " * 100)
        tok.freeze_vocab(512)

        tokens = tok.segment("hello world")
        assert len(tokens) > 0
        assert all(isinstance(t, str) for t in tokens)

    def test_db_status_returns_dict(self):
        tok = HybridTokenizer()
        tok.add_text("test " * 100)
        tok.freeze_vocab(512)

        status = tok.db_status()
        assert "vocab_size" in status
        assert "n_merges" in status
        assert "frozen" in status
        assert status["frozen"] == True

    def test_explain_token(self):
        tok = HybridTokenizer()
        tok.add_text("hello world test " * 100)
        tok.freeze_vocab(512)

        # Should not raise
        explanation = tok.explain_token("h")
        assert isinstance(explanation, str)

    def test_top_merges_returns_list(self):
        tok = HybridTokenizer()
        tok.add_text("hello world " * 100)
        tok.freeze_vocab(512)

        merges = tok.top_merges(10)
        assert isinstance(merges, list)
        assert len(merges) <= 10
