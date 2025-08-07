import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Advanced Modules ---
# RMSNorm is a normalization technique that uses root-mean-square statistics instead of full mean and variance.
# It avoids subtracting the mean and dividing by standard deviation, making it more stable for large-scale models.
# Compared to LayerNorm, RMSNorm is simpler and has shown better performance in some transformer architectures.
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        # RMSNorm computes the root-mean-square of the input across the last dimension:
        # norm = sqrt(mean(x^2)). This differs from LayerNorm, which subtracts the mean
        # and divides by standard deviation (involving variance). RMSNorm is more efficient
        # and numerically stable for large models.
        norm = x.norm(dim=-1, keepdim=True)
        return self.weight * x / (norm + self.eps)

# RoPE (Rotary Positional Embeddings) encodes position information by rotating query and key vectors in complex space.
# This allows positional dependencies to be baked directly into the dot-product attention mechanism without needing additive embeddings.
class RoPE:
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

class MultiHeadLocalAttention(nn.Module):
    def __init__(self, dim, heads, window):
        super().__init__()
        self.heads = heads
        self.dim = dim
        self.window = window
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out = nn.Linear(dim, dim)

    def forward(self, x):
        B, T, C = x.shape  # B: batch size, T: sequence length, C: embedding dim
        H, D = self.heads, self.head_dim  # H: number of heads, D: head dim

        # Project input into query, key, and value representations
        # Reshape to (B, T, H, 3D) then transpose to (B, H, T, 3D) for multi-head parallelism
        qkv = self.qkv(x).view(B, T, H, 3 * D).transpose(1, 2)  # (B, H, T, 3D)
        q, k, v = qkv.chunk(3, dim=-1)  # Split last dim into 3: (B, H, T, D) each

        q, k = RoPE.apply(q), RoPE.apply(k)

        attn = torch.matmul(q, k.transpose(-2, -1)) / (D ** 0.5)  # (B, H, T, T)
        local_mask = self._causal_local_mask(T, self.window, x.device)
        attn = attn.masked_fill(local_mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)  # (B, H, T, D)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out(out)

    def _causal_local_mask(self, T, W, device):
        # This mask restricts each token's attention to only a local window of size W before it,
        # including itself. This enforces causality and limits attention to recent context.
        mask = torch.tril(torch.ones((T, T), device=device))
        mask = torch.triu(mask, diagonal=-W)
        return mask[None, None, :, :]

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),  # swish/GELU-like
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        return self.net(x)

# TransformerBlock combines local self-attention and a feedforward MLP,
# each preceded by RMSNorm and wrapped with residual connections. This structure
# mirrors the design used in decoder-only transformer models.
class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_dim, window):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = MultiHeadLocalAttention(dim, heads, window)
        self.norm2 = RMSNorm(dim)
        self.ff = FeedForward(dim, mlp_dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, vocab_size, dim, depth, heads, mlp_dim, window):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, heads, mlp_dim, window) for _ in range(depth)
        ])
        self.norm = RMSNorm(dim)
        self.to_logits = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, x):
        x = self.token_emb(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.to_logits(x)

    def generate(self, x, max_new_tokens=50, eos_token_id=None):
        # Autoregressive generation loop:
        # At each step, we feed the entire current token sequence to the model,
        # take the logits at the last position, sample or argmax the next token,
        # append it to the input, and repeat. This continues until max_new_tokens
        # is reached or the <EOS> token is generated (if provided).
        for _ in range(max_new_tokens):
            logits = self.forward(x)[:, -1, :]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            x = torch.cat([x, next_token], dim=1)
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
        return x
