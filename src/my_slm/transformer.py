"""Transformer architecture: attention, feed-forward, and model components."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as gradient_checkpoint


class RMSNorm(nn.Module):
    """RMS (Root Mean Square) Layer Normalization.

    More stable than Layer Norm, used in modern models like LLaMA.
    """

    def __init__(self, dim: int, eps: float = 1e-8) -> None:
        """Initialize RMSNorm layer.

        Args:
            dim: Feature dimension.
            eps: Small constant for numerical stability.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization.

        Args:
            x: Input tensor of any shape.

        Returns:
            Normalized tensor with same shape as input.
        """
        norm = (x.pow(2).mean(dim=-1, keepdim=True) + self.eps).sqrt()
        return self.weight * x / norm


class RoPE:
    """
    Rotary Position Embeddings (RoPE) helper.

    Static helper kept for backward-compat (tests import RoPE.apply).
    MultiHeadAttention uses precomputed buffers instead — see _apply_rope.
    """

    @staticmethod
    def apply(x: torch.Tensor, seq_dim: int = 2) -> torch.Tensor:
        """Apply RoPE to tensor x.

        Args:
            x: Tensor to rotate.
            seq_dim: Sequence dimension (default 2 for [B, H, T, D]).

        Returns:
            Rotated tensor with same shape.
        """
        dim = x.shape[-1] // 2
        sinusoid = RoPE._get_sinusoid_embedding(x.shape[seq_dim], dim, x.device)
        x1, x2 = x[..., ::2], x[..., 1::2]
        sin, cos = sinusoid[..., 0], sinusoid[..., 1]
        x_rope = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return x_rope.flatten(-2)

    @staticmethod
    def _get_sinusoid_embedding(
        seq_len: int, dim: int, device: torch.device
    ) -> torch.Tensor:
        """Generate sinusoidal position embeddings.

        Args:
            seq_len: Sequence length.
            dim: Embedding dimension.
            device: Device to create tensor on.

        Returns:
            Tensor of shape [seq_len, dim, 2] with sin/cos pairs.
        """
        theta = 10000 ** (-torch.arange(0, dim, device=device) / dim)
        positions = torch.arange(seq_len, device=device).unsqueeze(-1)
        angle_rates = positions * theta
        return torch.stack([torch.sin(angle_rates), torch.cos(angle_rates)], dim=-1)


class MultiHeadAttention(nn.Module):
    """Multi-head attention with grouped-query attention (GQA) and local windowing.

    Supports:
    - Standard multi-head attention (kv_heads == heads)
    - Grouped-query attention (kv_heads < heads) for memory efficiency
    - Causal + local windowing for efficient long sequences
    - RoPE for position encoding
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        window: int,
        dropout: float = 0.0,
        kv_heads: Optional[int] = None,
    ) -> None:
        """Initialize multi-head attention layer.

        Args:
            dim: Model dimension (must be divisible by heads).
            heads: Number of attention heads.
            window: Local attention window size.
            dropout: Attention dropout probability.
            kv_heads: Number of KV heads for GQA. If None, equals heads.

        Raises:
            AssertionError: If heads not divisible by kv_heads.
        """
        super().__init__()
        self.heads = heads
        self.kv_heads = kv_heads if kv_heads is not None else heads
        assert heads % self.kv_heads == 0, "heads must be divisible by kv_heads"
        self.head_dim = dim // heads
        self.window = window
        self.dropout_p = dropout

        # Separate Q and KV projections for GQA.
        self.q = nn.Linear(dim, dim, bias=False)
        self.kv = nn.Linear(dim, 2 * self.kv_heads * self.head_dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

        # Precomputed RoPE buffers.
        half = self.head_dim // 2
        theta = 10_000.0 ** (-torch.arange(half) / half)
        pos = torch.arange(window)
        freq = torch.outer(pos, theta)
        self.register_buffer("_rope_sin", freq.sin().unsqueeze(0).unsqueeze(0))
        self.register_buffer("_rope_cos", freq.cos().unsqueeze(0).unsqueeze(0))

    def _apply_rope(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RoPE rotation to attention heads.

        Args:
            x: Tensor of shape [B, H, T, D].

        Returns:
            Rotated tensor with same shape.
        """
        T = x.shape[2]
        sin = self._rope_sin[:, :, :T, :]
        cos = self._rope_cos[:, :, :T, :]
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(
            -2
        )

    def forward(
        self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute multi-head attention.

        Args:
            x: Input tensor of shape [B, T, C].
            attention_mask: Padding mask of shape [B, T]. 1 = attend, 0 = mask.

        Returns:
            Attention output of shape [B, T, C].
        """
        B, T, C = x.shape
        H, KVH, D = self.heads, self.kv_heads, self.head_dim

        q = self.q(x).view(B, T, H, D).transpose(1, 2)  # [B, H, T, D]
        kv = self.kv(x).view(B, T, 2, KVH, D).permute(2, 0, 3, 1, 4)  # [2, B, KVH, T, D]
        k, v = kv[0], kv[1]

        q = self._apply_rope(q)
        k = self._apply_rope(k)

        # GQA: broadcast KV heads to match Q heads.
        if KVH < H:
            rep = H // KVH
            k = k.repeat_interleave(rep, dim=1)  # [B, H, T, D]
            v = v.repeat_interleave(rep, dim=1)

        causal_mask = self._causal_local_mask(T, self.window, x.device)
        if attention_mask is not None:
            pad_mask = attention_mask[:, None, None, :].bool()
            causal_mask = causal_mask & pad_mask

        dp = self.dropout_p if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=causal_mask, dropout_p=dp)
        return self.out(out.transpose(1, 2).contiguous().view(B, T, C))

    @staticmethod
    def _causal_local_mask(
        T: int, W: int, device: torch.device
    ) -> torch.Tensor:
        """Create causal + local attention mask.

        Attention is allowed within a sliding window and only to past positions.

        Args:
            T: Sequence length.
            W: Window size.
            device: Device to create tensor on.

        Returns:
            Mask tensor of shape [1, 1, T, T] with dtype bool.
        """
        m = torch.tril(torch.ones(T, T, device=device, dtype=torch.bool))
        return torch.triu(m, diagonal=-W)[None, None]


class FeedForward(nn.Module):
    """Feed-forward network with SwiGLU activation.

    Uses gating mechanism with SiLU activation for better expressivity.
    """

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        """Initialize feed-forward network.

        Args:
            dim: Input/output dimension.
            hidden_dim: Hidden layer dimension (scaled to 2/3 for SwiGLU).
            dropout: Dropout probability.
        """
        super().__init__()
        # SwiGLU: inner = 2/3 * hidden_dim so total params ≈ standard FFN
        inner = int(hidden_dim * 2 / 3)
        self.gate = nn.Linear(dim, inner, bias=False)
        self.value = nn.Linear(dim, inner, bias=False)
        self.proj = nn.Linear(inner, dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply feed-forward with SwiGLU.

        Args:
            x: Input tensor.

        Returns:
            Output tensor with same shape.
        """
        return self.drop(self.proj(F.silu(self.gate(x)) * self.value(x)))


class TransformerBlock(nn.Module):
    """Transformer block: pre-norm attention + pre-norm feed-forward.

    Uses residual connections and RMS normalization for stability.
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        mlp_dim: int,
        window: int,
        kv_heads: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        """Initialize transformer block.

        Args:
            dim: Model dimension.
            heads: Number of attention heads.
            mlp_dim: Feed-forward hidden dimension.
            window: Attention window size.
            kv_heads: Number of KV heads for GQA.
            dropout: Dropout probability.
        """
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = MultiHeadAttention(dim, heads, window, dropout, kv_heads=kv_heads)
        self.norm2 = RMSNorm(dim)
        self.ff = FeedForward(dim, mlp_dim, dropout)

    def forward(
        self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply transformer block.

        Args:
            x: Input tensor of shape [B, T, C].
            attention_mask: Optional padding mask of shape [B, T].

        Returns:
            Output tensor with same shape.
        """
        x = x + self.attn(self.norm1(x), attention_mask=attention_mask)
        x = x + self.ff(self.norm2(x))
        return x


class Transformer(nn.Module):
    """Transformer language model.

    Features:
    - Multi-head attention with GQA (grouped-query attention)
    - Local causal windowing for efficient long sequences
    - RoPE position embeddings
    - Pre-norm architecture with residual connections
    - Weight tying (GPT-2 / LLaMA style)
    - Gradient checkpointing support for memory efficiency
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        window: int,
        kv_heads: Optional[int] = None,
        dropout: float = 0.0,
        tie_weights: bool = True,
        use_checkpoint: bool = False,
    ) -> None:
        """Initialize Transformer model.

        Args:
            vocab_size: Size of token vocabulary.
            dim: Model embedding dimension.
            depth: Number of transformer blocks.
            heads: Number of attention heads.
            mlp_dim: Feed-forward hidden dimension.
            window: Attention window size (local causal attention).
            kv_heads: Number of KV heads for GQA. If None, equals heads.
            dropout: Dropout probability.
            tie_weights: If True, share embedding and output weights.
            use_checkpoint: If True, use gradient checkpointing during training.
        """
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.max_seq_len = window
        self.depth = depth
        self.vocab_size = vocab_size
        self.dim = dim

        self.token_emb = nn.Embedding(vocab_size, dim)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim, heads, mlp_dim, window, kv_heads=kv_heads, dropout=dropout
                )
                for _ in range(depth)
            ]
        )
        self.norm = RMSNorm(dim)
        self.to_logits = nn.Linear(dim, vocab_size, bias=False)

        # Weight tying: shared embedding ↔ output projection (GPT-2 / LLaMA style)
        if tie_weights:
            self.to_logits.weight = self.token_emb.weight

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights using scaled normal distribution.

        Standard 0.02 init for most layers. Residual output projections
        (attention.out, FFN.proj) use depth-scaled std to prevent variance
        blow-up at initialization (GPT-2 paper §2.3; LLaMA, Mistral).
        """
        residual_std = 0.02 / math.sqrt(2 * self.depth)

        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                # Residual projections get scaled init
                is_residual = name.endswith((".attn.out", ".ff.proj"))
                std = residual_std if is_residual else 0.02
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute model logits.

        Args:
            x: Input tensor. Either [B, T] token IDs or [B, T, C] embeddings.
            attention_mask: Optional padding mask [B, T]. 1 = attend, 0 = mask.

        Returns:
            Logits of shape [B, T, vocab_size].

        Raises:
            AssertionError: If input shape is invalid.
        """
        # Embed token IDs if needed
        if x.dim() == 2 and x.dtype in (torch.long, torch.int64):
            x = self.token_emb(x)
        assert x.dim() == 3, f"expected [B,T,C], got {tuple(x.shape)}"

        x = self.drop(x)
        for block in self.blocks:
            if self.use_checkpoint and self.training:
                x = gradient_checkpoint(block, x, attention_mask, use_reentrant=False)
            else:
                x = block(x, attention_mask=attention_mask)
        x = self.norm(x)
        return self.to_logits(x)

    @torch.inference_mode()
    def generate(
        self,
        x: torch.Tensor,
        max_new_tokens: int = 100,
        eos_token_id: int | None = None,
        temperature: float = 0.8,
        top_k: int = 50,
        suppress_ids: list[int] | None = None,
        repetition_penalty: float = 1.3,
    ) -> torch.Tensor:
        """
        Autoregressive generation with input validation and KV cache support.

        Args:
            x: [B, T] LongTensor of prompt token ids
            max_new_tokens: Maximum new tokens to generate
            eos_token_id: Stop token ID (default: 2 for <EOS>)
            temperature: >1 = more random, <1 = more focused, 0 = greedy
            top_k: Keep only top-k candidates (0 = disabled)
            suppress_ids: Token IDs to never generate
            repetition_penalty: >1 discourages repeating tokens in context
        """
        self.eval()
        device = next(self.parameters()).device
        x = x.to(device)

        # Validate input token IDs
        vocab_size = self.token_emb.num_embeddings
        if (x < 0).any() or (x >= vocab_size).any():
            raise ValueError(f"Token ID out of range [0, {vocab_size}). Found: {x.max().item()}")

        # Validate input length
        if x.shape[1] > self.max_seq_len:
            raise ValueError(f"Prompt length {x.shape[1]} exceeds max_seq_len {self.max_seq_len}")

        block_size = self.max_seq_len

        for step in range(max_new_tokens):
            x_cond = x[:, -block_size:]
            logits = self(x_cond)[:, -1, :]  # [B, V]

            # Repetition penalty (vectorized per batch)
            if repetition_penalty != 1.0:
                for b in range(x.shape[0]):
                    unique_ids, counts = x[b].unique(return_counts=True)
                    score = logits[b, unique_ids]
                    factor = torch.clamp(
                        repetition_penalty ** counts.float(),
                        min=1e-5,
                        max=1e10,  # Prevent numeric overflow
                    )
                    logits[b, unique_ids] = torch.where(
                        score > 0,
                        score / factor,
                        score * factor,
                    )

            # Suppress specific tokens
            if suppress_ids:
                for sid in suppress_ids:
                    if 0 <= sid < vocab_size:
                        logits[:, sid] = float('-inf')

            # Sample next token
            if temperature == 0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k > 0:
                    k = min(top_k, logits.size(-1))
                    threshold = logits.topk(k).values[:, -1, None]
                    logits = logits.masked_fill(logits < threshold, float('-inf'))

                # Check for NaN from all-masked logits
                if (logits == float('-inf')).all(dim=-1).any():
                    logits[:, eos_token_id or 2] = 0.0  # Fallback to EOS

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            x = torch.cat([x, next_token], dim=1)

            # Early stop on EOS (check all batch elements)
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return x

    def set_dropout(self, p: float) -> None:
        """Update dropout probability in all layers.

        Call before fine-tuning to change regularization strength.

        Args:
            p: New dropout probability.
        """
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.p = p

    def resize_token_embeddings(self, new_size: int) -> None:
        """Resize token embeddings and output projection.

        Used when vocab size changes after model creation. Preserves old weights
        and initializes new rows. Re-ties weights if they were previously tied.

        Args:
            new_size: New vocabulary size.
        """
        old_emb = self.token_emb
        old_n, dim = old_emb.num_embeddings, old_emb.embedding_dim
        if new_size == old_n:
            return

        device = old_emb.weight.device
        dtype = old_emb.weight.dtype

        new_emb = nn.Embedding(new_size, dim, device=device, dtype=dtype)
        num_copy = min(old_n, new_size)
        with torch.no_grad():
            new_emb.weight[:num_copy].copy_(old_emb.weight[:num_copy])
            if new_size > old_n:
                nn.init.normal_(new_emb.weight[num_copy:], std=0.02)
        self.token_emb = new_emb

        new_out = nn.Linear(dim, new_size, bias=False, device=device, dtype=dtype)
        with torch.no_grad():
            new_out.weight[:num_copy].copy_(self.to_logits.weight[:num_copy])
            if new_size > old_n:
                nn.init.normal_(new_out.weight[num_copy:], std=0.02)
        self.to_logits = new_out

        # Re-tie if weights were previously tied
        if self.to_logits.weight.data_ptr() != self.token_emb.weight.data_ptr():
            self.to_logits.weight = self.token_emb.weight
