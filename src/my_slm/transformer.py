import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = (x.pow(2).mean(dim=-1, keepdim=True) + self.eps).sqrt()
        return self.weight * x / norm


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


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads, window, dropout=0.0):
        super().__init__()
        self.heads    = heads
        self.head_dim = dim // heads
        self.window   = window
        self.dropout_p = dropout
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x, attention_mask=None):
        B, T, C = x.shape
        H, D = self.heads, self.head_dim

        qkv = self.qkv(x).view(B, T, H, 3 * D).transpose(1, 2)  # [B, H, T, 3D]
        q, k, v = qkv.chunk(3, dim=-1)
        q, k = RoPE.apply(q), RoPE.apply(k)

        # Local causal mask: each token sees up to `window` previous tokens.
        # When window >= T this is identical to full causal attention.
        causal_mask = self._causal_local_mask(T, self.window, x.device)  # [1,1,T,T] bool

        if attention_mask is not None:
            # True = real token, False = padding — mask padding keys
            pad_mask  = attention_mask[:, None, None, :].bool()  # [B,1,1,T]
            causal_mask = causal_mask & pad_mask                  # [B,1,T,T] broadcast

        # F.scaled_dot_product_attention uses FlashAttention when available (PyTorch ≥ 2.0).
        # Bool mask: True = attend, False = block.
        dropout_p = self.dropout_p if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v,
                                             attn_mask=causal_mask,
                                             dropout_p=dropout_p)  # [B,H,T,D]

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out(out)

    @staticmethod
    def _causal_local_mask(T, W, device):
        mask = torch.tril(torch.ones(T, T, device=device, dtype=torch.bool))
        # triu with diagonal=-W keeps only the last W positions (local window)
        mask = torch.triu(mask, diagonal=-W)
        return mask[None, None, :, :]  # [1,1,T,T]


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        # SwiGLU: scale inner to 2/3 so total params ≈ standard FF (Phi, LLaMA)
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
        self.attn  = MultiHeadAttention(dim, heads, window, dropout=dropout)
        self.norm2 = RMSNorm(dim)
        self.ff    = FeedForward(dim, mlp_dim, dropout=dropout)

    def forward(self, x, attention_mask=None):
        x = x + self.attn(self.norm1(x), attention_mask=attention_mask)
        x = x + self.ff(self.norm2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, vocab_size, dim, depth, heads, mlp_dim, window,
                 dropout=0.0, tie_weights=True, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.max_seq_len = window
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.drop      = nn.Dropout(dropout)
        self.blocks    = nn.ModuleList([
            TransformerBlock(dim, heads, mlp_dim, window, dropout=dropout)
            for _ in range(depth)
        ])
        self.norm      = RMSNorm(dim)
        self.to_logits = nn.Linear(dim, vocab_size, bias=False)

        # Weight tying: share embedding ↔ output projection weights.
        # Reduces params and improves convergence (used in GPT-2, LLaMA, etc.)
        if tie_weights:
            self.to_logits.weight = self.token_emb.weight

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
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
                x = checkpoint(block, x, attention_mask, use_reentrant=False)
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
        x                  : [B, T] LongTensor of prompt token ids
        temperature        : >1 = more random, <1 = more focused, 0 = greedy
        top_k              : keep only top-k candidates before sampling (0 = disabled)
        suppress_ids       : token ids to never generate (e.g. PAD, UNK)
        repetition_penalty : >1 discourages repeating tokens already in context
        """
        self.eval()
        device     = next(self.parameters()).device
        x          = x.to(device)
        block_size = getattr(self, 'max_seq_len', 512)

        for _ in range(max_new_tokens):
            x_cond = x[:, -block_size:]
            logits = self(x_cond)[:, -1, :]  # [B, V]

            # Penalise tokens already present in the context
            if repetition_penalty != 1.0:
                for b in range(x.shape[0]):
                    seen = x[b].unique()
                    score = logits[b, seen]
                    logits[b, seen] = torch.where(
                        score > 0, score / repetition_penalty, score * repetition_penalty
                    )

            # Block special tokens so they are never sampled
            if suppress_ids:
                for sid in suppress_ids:
                    logits[:, sid] = float('-inf')

            if temperature == 0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature

                if top_k > 0:
                    k = min(top_k, logits.size(-1))
                    threshold = logits.topk(k).values[:, -1, None]
                    logits = logits.masked_fill(logits < threshold, float('-inf'))

                probs      = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)  # [B, 1]

            x = torch.cat([x, next_token], dim=1)

            if eos_token_id is not None and torch.all(next_token == eos_token_id).item():
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

        # Re-tie weights after resize
        if self.to_logits.weight.data_ptr() != self.token_emb.weight.data_ptr():
            # Only re-tie if they were tied before (check by comparing shapes)
            self.to_logits.weight = self.token_emb.weight
