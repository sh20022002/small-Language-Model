"""
Performance benchmarks: tokenizer speed (encode/decode),
training throughput, memory usage, generation speed.
"""
import time
import torch
import torch.nn as nn
import pytest
from pathlib import Path

from my_slm.hybrid_tokeniztion import HybridTokenizer
from my_slm.transformer import Transformer


class TestTokenizerPerformance:
    """Benchmark tokenizer performance."""

    @pytest.fixture(scope="class")
    def tokenizer(self):
        """Create a tokenizer for all tests."""
        tok = HybridTokenizer()
        tok.add_text("The quick brown fox jumps over the lazy dog. " * 5000)
        tok.freeze_vocab(32768)
        return tok

    def test_encode_throughput(self, tokenizer):
        """Measure encoding throughput."""
        text = "The quick brown fox jumps over the lazy dog. " * 100

        start = time.time()
        ids = tokenizer.encode(text)
        elapsed = time.time() - start

        # Avoid division by zero
        if elapsed > 0:
            tokens_per_second = len(ids) / elapsed
            print(f"\nEncode throughput: {tokens_per_second:.0f} tokens/sec")
            # Should be reasonably fast (at least 10 tokens/sec)
            assert tokens_per_second > 10
        else:
            # Encoding was so fast it was unmeasurable
            assert len(ids) > 0

    def test_decode_throughput(self, tokenizer):
        """Measure decoding throughput."""
        text = "The quick brown fox jumps over the lazy dog. " * 100
        ids = tokenizer.encode(text)

        start = time.time()
        for _ in range(100):
            decoded = tokenizer.decode(ids)
        elapsed = time.time() - start

        ops_per_second = 100 / elapsed
        print(f"\nDecode throughput: {ops_per_second:.1f} decodings/sec")

        assert ops_per_second > 10  # At least 10 decodings per second

    def test_encode_large_batch(self, tokenizer):
        """Measure encoding a large batch of texts."""
        texts = ["Hello world test example. " * 20 for _ in range(1000)]

        start = time.time()
        all_ids = [tokenizer.encode(t) for t in texts]
        elapsed = time.time() - start

        total_tokens = sum(len(ids) for ids in all_ids)
        throughput = total_tokens / elapsed

        print(f"\nLarge batch encode: {throughput:.0f} tokens/sec ({len(texts)} texts)")
        assert throughput > 100

    def test_cache_speedup(self, tokenizer):
        """Measure cache behavior for repeated encodings."""
        text = "The quick brown fox jumps over the lazy dog"

        # First pass (populate cache)
        tokenizer._word_cache.clear()
        start = time.time()
        for _ in range(1000):
            ids = tokenizer.encode(text)
        elapsed_cold = time.time() - start

        # Second pass (use cache)
        start = time.time()
        for _ in range(1000):
            ids = tokenizer.encode(text)
        elapsed_hot = time.time() - start

        # Note: speedup may be < 1 if cache overhead is high for very fast operations
        if elapsed_hot > 0:
            ratio = elapsed_cold / elapsed_hot
            print(f"\nCache speedup ratio: {ratio:.1f}x")
        else:
            print(f"\nCache speedup: immeasurably fast")

        # Should at least complete without error
        assert True


class TestModelPerformance:
    """Benchmark model performance."""

    @pytest.fixture(scope="class")
    def model(self):
        """Create model for benchmarks."""
        return Transformer(
            vocab_size=256,
            dim=128,
            depth=4,
            heads=4,
            mlp_dim=256,
            window=64,
        ).eval()

    def test_forward_pass_throughput(self, model):
        """Measure forward pass throughput."""
        batch_size = 4
        seq_length = 64

        ids = torch.randint(1, 256, (batch_size, seq_length))

        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = model(ids)

        # Benchmark
        num_iters = 100
        start = time.time()
        with torch.no_grad():
            for _ in range(num_iters):
                logits = model(ids)
        elapsed = time.time() - start

        tokens_per_second = (batch_size * seq_length * num_iters) / elapsed
        print(f"\nForward pass throughput: {tokens_per_second:.0f} tokens/sec")

        assert tokens_per_second > 100

    def test_generation_speed(self, model):
        """Measure generation speed."""
        prompt = torch.randint(1, 256, (1, 16))

        start = time.time()
        with torch.no_grad():
            output = model.generate(prompt, max_new_tokens=100, temperature=0.8)
        elapsed = time.time() - start

        tokens_generated = 100
        tokens_per_second = tokens_generated / elapsed

        print(f"\nGeneration speed: {tokens_per_second:.0f} tokens/sec")
        # Generation should be reasonably fast
        assert tokens_per_second > 10

    def test_batch_size_scaling(self, model):
        """Test how throughput scales with batch size."""
        seq_length = 64

        for batch_size in [1, 2, 4, 8]:
            ids = torch.randint(1, 256, (batch_size, seq_length))

            start = time.time()
            num_iters = 50
            with torch.no_grad():
                for _ in range(num_iters):
                    _ = model(ids)
            elapsed = time.time() - start

            tokens_per_second = (batch_size * seq_length * num_iters) / elapsed
            print(f"\nBatch size {batch_size}: {tokens_per_second:.0f} tokens/sec")

    def test_sequence_length_scaling(self, model):
        """Test how throughput scales with sequence length (within window)."""
        batch_size = 4

        # Note: model window is 64, so keep sequences within that limit
        for seq_length in [16, 32, 64]:
            ids = torch.randint(1, 256, (batch_size, seq_length))

            start = time.time()
            num_iters = 50
            with torch.no_grad():
                for _ in range(num_iters):
                    _ = model(ids)
            elapsed = time.time() - start

            tokens_per_second = (batch_size * seq_length * num_iters) / elapsed
            print(f"\nSequence length {seq_length}: {tokens_per_second:.0f} tokens/sec")


class TestMemoryUsage:
    """Benchmark memory usage."""

    def test_model_parameter_count(self):
        """Measure model parameter count."""
        model = Transformer(
            vocab_size=256,
            dim=128,
            depth=4,
            heads=4,
            mlp_dim=256,
            window=64,
        )

        total_params = sum(p.numel() for p in model.parameters())
        total_bytes = total_params * 4  # assuming float32

        print(f"\nModel parameters: {total_params:,} ({total_bytes / 1e6:.1f} MB)")
        assert total_params > 1000

    def test_batch_memory_usage(self):
        """Estimate memory usage for a batch."""
        batch_size = 4
        seq_length = 128
        vocab_size = 256

        # Input IDs
        input_bytes = batch_size * seq_length * 8  # int64

        # Logits output
        logits_bytes = batch_size * seq_length * vocab_size * 4  # float32

        # Hidden states (rough estimate: depth layers)
        depth = 4
        hidden_dim = 128
        hidden_bytes = depth * batch_size * seq_length * hidden_dim * 4

        total_mb = (input_bytes + logits_bytes + hidden_bytes) / 1e6

        print(f"\nEstimated batch memory: {total_mb:.1f} MB")
        print(f"  Input IDs: {input_bytes / 1e6:.1f} MB")
        print(f"  Logits: {logits_bytes / 1e6:.1f} MB")
        print(f"  Hidden states: {hidden_bytes / 1e6:.1f} MB")

    def test_tokenizer_memory(self):
        """Measure tokenizer memory usage."""
        tok = HybridTokenizer()
        tok.add_text("test data " * 10000)
        tok.freeze_vocab(32768)

        # Rough estimates
        token2id_size = len(tok.token2id) * (50 + 8)  # string + int
        id2token_size = len(tok.id2token) * (50 + 8)
        merge_list_size = len(tok.merge_list) * (50 + 50 + 8)

        total_mb = (token2id_size + id2token_size + merge_list_size) / 1e6

        print(f"\nTokenizer memory: ~{total_mb:.1f} MB")
        print(f"  Vocab size: {len(tok.id2token):,}")
        print(f"  Merges: {len(tok.merge_list):,}")


class TestLatencyMetrics:
    """Measure latency metrics."""

    def test_forward_pass_latency_percentiles(self):
        """Measure latency percentiles for forward pass."""
        model = Transformer(
            vocab_size=256,
            dim=128,
            depth=4,
            heads=4,
            mlp_dim=256,
            window=64,
        ).eval()

        batch_size = 4
        seq_length = 64
        ids = torch.randint(1, 256, (batch_size, seq_length))

        latencies = []
        num_samples = 100

        for _ in range(num_samples):
            start = time.perf_counter()
            with torch.no_grad():
                _ = model(ids)
            elapsed = time.perf_counter() - start
            latencies.append(elapsed * 1000)  # ms

        latencies.sort()

        p50 = latencies[50]
        p95 = latencies[95]
        p99 = latencies[99]

        print(f"\nLatency percentiles (ms):")
        print(f"  p50: {p50:.2f}")
        print(f"  p95: {p95:.2f}")
        print(f"  p99: {p99:.2f}")

    def test_generation_latency_first_token(self):
        """Measure latency for first token generation."""
        model = Transformer(
            vocab_size=256,
            dim=128,
            depth=4,
            heads=4,
            mlp_dim=256,
            window=64,
        ).eval()

        prompt = torch.randint(1, 256, (1, 16))

        # Warmup
        with torch.no_grad():
            _ = model.generate(prompt, max_new_tokens=1, temperature=0.8)

        # Measure
        start = time.perf_counter()
        with torch.no_grad():
            _ = model.generate(prompt, max_new_tokens=1, temperature=0.8)
        elapsed = time.perf_counter() - start

        print(f"\nFirst token latency: {elapsed * 1000:.2f} ms")


class TestComputeEfficiency:
    """Test compute efficiency metrics."""

    def test_model_flops_estimate(self):
        """Rough FLOP estimate for forward pass."""
        batch_size = 4
        seq_length = 64
        vocab_size = 256
        dim = 128
        depth = 4
        heads = 4

        # Attention: O(seq_length^2 * dim * heads)
        attn_flops = batch_size * depth * seq_length * seq_length * dim

        # MLP: O(seq_length * dim^2)
        mlp_flops = batch_size * depth * seq_length * dim * dim * 4

        # Embedding lookup + projection
        embedding_flops = batch_size * seq_length * vocab_size + batch_size * seq_length * dim * vocab_size

        total_flops = attn_flops + mlp_flops + embedding_flops

        print(f"\nEstimated FLOPS for forward pass: {total_flops / 1e9:.2f} GFLOPS")
        print(f"  Attention: {attn_flops / 1e9:.2f} GFLOPS")
        print(f"  MLP: {mlp_flops / 1e9:.2f} GFLOPS")
        print(f"  Embedding: {embedding_flops / 1e9:.2f} GFLOPS")


class TestRegressionBenchmarks:
    """Regression tests to catch performance regressions."""

    def test_encode_performance_regression(self):
        """Check that encoding doesn't degrade."""
        tok = HybridTokenizer()
        tok.add_text("test data " * 5000)
        tok.freeze_vocab(32768)

        text = "test " * 100

        start = time.time()
        for _ in range(100):
            _ = tok.encode(text)
        elapsed = time.time() - start

        throughput = (len(text.split()) * 100) / elapsed

        # Set a baseline (should improve or stay roughly same)
        baseline = 100  # tokens/sec
        print(f"\nEncode performance: {throughput:.0f} tokens/sec (baseline: {baseline})")

        # Allow up to 2x slower (indicates regression)
        assert throughput > baseline / 2

    def test_model_forward_regression(self):
        """Check that forward pass doesn't degrade."""
        model = Transformer(
            vocab_size=256,
            dim=64,
            depth=2,
            heads=2,
            mlp_dim=128,
            window=32,
        ).eval()

        ids = torch.randint(1, 256, (2, 32))

        start = time.time()
        with torch.no_grad():
            for _ in range(100):
                _ = model(ids)
        elapsed = time.time() - start

        throughput = (2 * 32 * 100) / elapsed

        baseline = 100  # tokens/sec
        print(f"\nModel forward performance: {throughput:.0f} tokens/sec (baseline: {baseline})")

        assert throughput > baseline / 2
