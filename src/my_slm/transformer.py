import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as gradient_checkpoint


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = (x.pow(2).mean(dim=-1, keepdim=True) + self.eps).sqrt()
        return self.weight * x / norm


class RoPE:
    """
    Static helper kept for backward-compat (tests import RoPE.apply).
    MultiHeadAttention uses precomputed buffers instead — see _apply_rope.
    """
    @staticmethod
    def apply(x, seq_dim=2):
        dim = x.shape[-1] // 2
        sinusoid = RoPE._get_sinusoid_embedding(x.shape[seq_dim], dim, x.device)
        x1, x2 = x[..., ::2], x[..., 1::2]
        sin, cos = sinusoid[..., 0], sinusoid[..., 1]
        x_rope = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return x_rope.flatten(-2)

    @staticmethod
    def _get_sinusoid_embedding(seq_len, dim, device):
        theta = 10000 ** (-torch.arange(0, dim, device=device) / dim)
        positions = torch.arange(seq_len, device=device).unsqueeze(-1)
        angle_rates = positions * theta
        return torch.stack([torch.sin(angle_rates), torch.cos(angle_rates)], dim=-1)


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads, window, dropout=0.0):
        super().__init__()
        self.heads     = heads
        self.head_dim  = dim // heads
        self.window    = window
        self.dropout_p = dropout
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out = nn.Linear(dim, dim,     bias=False)

        # Precompute RoPE sin/cos up to window length — stored as non-trainable
        # buffers so they move to the correct device automatically with model.to().
        half  = self.head_dim // 2
        theta = 10_000.0 ** (-torch.arange(half) / half)          # [D/2]
        pos   = torch.arange(window)                               # [W]
        freq  = torch.outer(pos, theta)                            # [W, D/2]
        # shape [1, 1, W, D/2] — broadcast over batch and head dims
        self.register_buffer('_rope_sin', freq.sin().unsqueeze(0).unsqueeze(0))
        self.register_buffer('_rope_cos', freq.cos().unsqueeze(0).unsqueeze(0))

    def _apply_rope(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, H, T, D] → rotated [B, H, T, D] using cached tables."""
        T   = x.shape[2]
        sin = self._rope_sin[:, :, :T, :]   # [1, 1, T, D/2]
        cos = self._rope_cos[:, :, :T, :]
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.stack([x1 * cos - x2 * sin,
                            x1 * sin + x2 * cos], dim=-1).flatten(-2)

    def forward(self, x, attention_mask=None):
        B, T, C = x.shape
        H, D    = self.heads, self.head_dim

        qkv = self.qkv(x).view(B, T, H, 3 * D).transpose(1, 2)   # [B, H, T, 3D]
        q, k, v = qkv.chunk(3, dim=-1)
        q, k = self._apply_rope(q), self._apply_rope(k)

        # Local causal mask: each position sees at most `window` previous tokens.
        causal_mask = self._causal_local_mask(T, self.window, x.device)

        if attention_mask is not None:
            pad_mask    = attention_mask[:, None, None, :].bool()  # [B,1,1,T]
            causal_mask = causal_mask & pad_mask

        dp  = self.dropout_p if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v,
                                             attn_mask=causal_mask,
                                             dropout_p=dp)
        return self.out(out.transpose(1, 2).contiguous().view(B, T, C))

    @staticmethod
    def _causal_local_mask(T, W, device):
        m = torch.tril(torch.ones(T, T, device=device, dtype=torch.bool))
        return torch.triu(m, diagonal=-W)[None, None]


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        # SwiGLU: inner dim scaled to 2/3 so total params ≈ standard FFN
        inner = int(hidden_dim * 2 / 3)
        self.gate  = nn.Linear(dim, inner, bias=False)
        self.value = nn.Linear(dim, inner, bias=False)
        self.proj  = nn.Linear(inner, dim, bias=False)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.proj(F.silu(self.gate(x)) * self.value(x)))


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_dim, window, dropout=0.0):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn  = MultiHeadAttention(dim, heads, window, dropout)
        self.norm2 = RMSNorm(dim)
        self.ff    = FeedForward(dim, mlp_dim, dropout)

    def forward(self, x, attention_mask=None):
        x = x + self.attn(self.norm1(x), attention_mask=attention_mask)
        x = x + self.ff(self.norm2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, vocab_size, dim, depth, heads, mlp_dim, window,
                 dropout=0.0, tie_weights=True, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.max_seq_len    = window
        self.depth          = depth          # stored for depth-scaled init
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.drop      = nn.Dropout(dropout)
        self.blocks    = nn.ModuleList([
            TransformerBlock(dim, heads, mlp_dim, window, dropout=dropout)
            for _ in range(depth)
        ])
        self.norm      = RMSNorm(dim)
        self.to_logits = nn.Linear(dim, vocab_size, bias=False)

        # Weight tying: shared embedding ↔ output projection (GPT-2 / LLaMA style)
        if tie_weights:
            self.to_logits.weight = self.token_emb.weight

        self._init_weights()

    def _init_weights(self):
        # Standard 0.02 init for most layers.
        # Residual output projections (attention.out, FFN.proj) use depth-scaled
        # std = 0.02 / sqrt(2 * depth) to prevent variance blow-up at init
        # (GPT-2 paper §2.3; also used in LLaMA, Mistral).
        residual_std = 0.02 / math.sqrt(2 * self.depth)

        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                # out-projection of attention and final FF proj feed into residual stream
                is_residual = name.endswith(('.attn.out', '.ff.proj'))
                std = residual_std if is_residual else 0.02
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, attention_mask=None):
        if x.dim() == 2 and x.dtype in (torch.long, torch.int64):
            x = self.token_emb(x)
        assert x.dim() == 3, f"expected [B,T,C], got {tuple(x.shape)}"

        x = self.drop(x)
        for block in self.blocks:
            if self.use_checkpoint and self.training:
                x = gradient_checkpoint(block, x, attention_mask,
                                        use_reentrant=False)
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
        Autoregressive generation.
        x                  : [B, T] LongTensor of prompt token ids
        temperature        : >1 = more random, <1 = more focused, 0 = greedy
        top_k              : keep only top-k candidates (0 = disabled)
        suppress_ids       : token ids to never generate (e.g. PAD, UNK)
        repetition_penalty : >1 discourages repeating tokens in the context
        """
        self.eval()
        device     = next(self.parameters()).device
        x          = x.to(device)
        block_size = self.max_seq_len

        for _ in range(max_new_tokens):
            x_cond = x[:, -block_size:]
            logits = self(x_cond)[:, -1, :]       # [B, V]

            if repetition_penalty != 1.0:
                for b in range(x.shape[0]):
                    unique_ids, counts = x[b].unique(return_counts=True)
                    score = logits[b, unique_ids]
                    # Scale penalty exponentially with frequency so repeated tokens
                    # get progressively harder to regenerate
                    factor = repetition_penalty ** counts.float()
                    logits[b, unique_ids] = torch.where(
                        score > 0,
                        score / factor,
                        score * factor,
                    )

            if suppress_ids:
                for sid in suppress_ids:
                    logits[:, sid] = float('-inf')

            if temperature == 0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k > 0:
                    k         = min(top_k, logits.size(-1))
                    threshold = logits.topk(k).values[:, -1, None]
                    logits    = logits.masked_fill(logits < threshold, float('-inf'))
                probs      = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            x = torch.cat([x, next_token], dim=1)

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return x

    def set_dropout(self, p: float):
        """Update dropout probability in all layers; call before fine-tuning."""
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.p = p

    def resize_token_embeddings(self, new_size: int):
        old_emb = self.token_emb
        old_n, dim = old_emb.num_embeddings, old_emb.embedding_dim
        if new_size == old_n:
            return

        device = old_emb.weight.device
        dtype  = old_emb.weight.dtype

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
