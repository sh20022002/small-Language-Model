# Personalized-Business Chatbot Architecture

## 1) System at a glance

- **LFE (Language Feature Extractor):**  
  A (mostly) frozen Transformer that maps input text → latent **Z** rather than logits.  
  - Outputs:
    - **Z_seq ∈ ℝ^{T×d}** (per-token features from the last K layers, projected)
    - **z_cls ∈ ℝ^{d}** (pooled summary for routing / policy)

- **TPH (Tenant Preference Head):**  
  Small model that consumes **Z** (+ optional retrieval snippets) and **generates tokens** or **selects actions**, trained on tenant preference data (DPO / RLAIF-style) + a little supervised data.

- **(Optional) Policy / Guard Head:**  
  Scores factuality, style, compliance; edits or escalates if needed.

Why this works:
- LFE is your “ViT-backbone” equivalent; TPHs are tiny, swappable “heads” conditioned on a **tenant embedding p**.  
- Per-tenant state is just **p** (+ a few MB of adapter weights at most), so serving is cheap.

---

## 2) LFE (replace token logits with a good latent)

**Backbone:**  
Start from a solid open LM (3–7B). Expose hidden states from the top _K_ layers (e.g. _K = 2-4_). Add a projection head:

- `Z_seq = LN(Concat(H_{L−K..L})) W_proj` where `W_proj: ℝ^{K·d} → ℝ^{d_z}`
- `z_cls = Pool(Z_seq)` (mean / attention-pool with a learned query)

**Training / Retrofitting LFE** so **Z** is sufficient:
- Keep the backbone frozen or lightly LoRA-tune the top 2-3 layers.
- Add **multi-objective heads** during retrofit:
  1. **Distill-to-Teacher**: run a larger teacher LM; minimize KL on next-token logits predicted from **Z** via a tiny decoder head (ensures **Z** contains what’s needed to generate).
  2. **Span Denoising** (BART/T5 style): small decoder reconstructs masked spans from **Z**.
  3. **Contrastive (InfoNCE)** on paraphrases: same meaning → close _z_cls_; different → far.
  4. **Tool / Routing labels** (if available) — linear probe from _z_cls_.
- After retrofit, drop training-only decoders; keep **W_proj** and pooling. Goal: backbone emits a useful **Z** in one forward pass.

---

## 3) Tenant Preference Head (TPH)

Two main variants — choose based on cost/risk tradeoffs:

### A) Generative TPH (small decoder)
- **Inputs:** `Z_seq`, `z_cls`, tenant embedding **p**, optional retrieved docs **R**.
- **Architecture:**
  - 1-2 Transformer blocks (width ~ 512-1024)
  - Cross-attention over `Z_seq ⊕ R`
  - Prefix / soft prompts or adapter conditioning with **p**
- **Training Losses:**
  - **SFT** on tenant examples (if available)
  - **DPO / RLAIF** using thumbs-up / thumbs-down (or pairwise preference) data
  - **KL penalty** to a reference policy (to prevent drift)
  - **Policy penalties** (e.g. for disallowed claims, hallucinations) as auxiliary BCE losses

### B) Selector / Editor TPH (cheap & stable)
- Generate **N candidates** with a generic decoder (or base LM).
- TPH is a **reranker / editor** that picks or refines the best candidate based on tenant preferences.
- Train with **DPO** on pairwise preference data; include style/format editing if desired.

**Tenant embedding `p` (personalization)**
- Learned vector `p ∈ ℝ^{d_p}` (e.g. 128-512 dims).  
- Derived from:
  - Style guides, tone preferences, length, formality, disclaimers
  - Example dialogues, past chats, feedback (👍/👎)
- Injection mechanisms:
  - Soft prompts / prefix tuning
  - Adapter / LoRA conditioning
  - Hypernetwork: small net `A(p)` → generates LoRA deltas for TPH (if using adapters)

---

## 4) Training Workflow

**Stage 0 — Data Preparation**
- General, multi-domain instruction/task data + synthetic tool traces
- Tenant onboarding data: 30-200 examples; thumbs-up/-down feedback when available

**Stage 1 — LFE Retrofitting (once)**
- Train projection `W_proj`, pooling, and **auxiliary heads** (distill, denoise, contrastive, routing).
- Freeze or lightly adapt backbone. Stop when reconstruction and proxy tasks plateau.

**Stage 2 — Universal TPH Training**
- Train shared TPH on broad data; ensure good base performance before tenant specialization.
- Use distillation from a stronger teacher (if available) to improve generation quality.

**Stage 3 — Tenant Personalization**
- Initialize or fit **p** from onboarding data.
- Optionally fine-tune TPH via small LoRA deltas (if risk acceptable).
- Run **DPO / RLHF-style** preference tuning (user likes/dislikes).
- Train or apply **policy head** to enforce compliance, style, factuality, no-hallucination rules.

**Stage 4 — Integration with Retrieval / RAG**
- Create per-tenant KB / document index.
- Train a **retrieval projection head** mapping `[z_cls ⊕ p] → embedding space` for retrieval.
- Include retrieved passages **R** in the TPH cross-attention during personalization to ground output.

---

## 5) Inference Pipeline (Low Latency)

1. Tokenize user input → pass through **LFE** → obtain `Z_seq`, `z_cls`.
2. Use `[z_cls ⊕ p]` to retrieve tenant-specific documents → get **R**.
3. TPH decodes, attending over `Z_seq ⊕ R`, conditioned on **p** → generate response.
4. **Policy head** checks for violations / hallucination / style problems → decide:
   - pass as-is,
   - apply micro-edit,
   - escalate to human or safe fallback.

**Serving optimizations:**
- Quantize LFE (4-bit); keep TPH in 8-bit / FP16.
- Cache `Z_seq / z_cls` per conversation turn to speed multi-turn dialogues.
- Batch requests across tenants; only **p** differs per request.

---

## 6) Minimal PyTorch Skeleton

```python
class LFEBackbone(nn.Module):
    def __init__(self, base_model, d_out=1024, top_k=2):
        super().__init__()
        self.base = base_model  # frozen or lightly LoRA-tuned
        self.top_k = top_k
        d = base_model.config.hidden_size
        self.proj = nn.Linear(d * top_k, d_out)
        self.pool_q = nn.Parameter(torch.randn(d_out))

    @torch.no_grad()
    def forward(self, input_ids, attention_mask):
        hs = self.base.transformer_outputs(
            input_ids,
            attention_mask,
            output_hidden_states=True
        ).hidden_states
        H = torch.cat(hs[-self.top_k:], dim=-1)  # [B, T, K*d]
        Z_seq = F.layer_norm(self.proj(H), (self.proj.out_features,))  # [B, T, d_out]

        # attention pooling for z_cls
        q = self.pool_q.expand(Z_seq.size(0), 1, -1)  # [B, 1, d_out]
        attn = torch.softmax(
            (q @ Z_seq.transpose(1, 2)) / (Z_seq.size(-1) ** 0.5),
            dim=-1
        )  # [B, 1, T]
        z_cls = (attn @ Z_seq).squeeze(1)  # [B, d_out]
        return Z_seq, z_cls


class GenerativeTPH(nn.Module):
    def __init__(self, d_z=1024, d_model=768, n_layers=2, d_p=256, vocab=32000):
        super().__init__()
        self.p_embed = nn.Linear(d_p, d_model)
        self.z_proj = nn.Linear(d_z, d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8, batch_first=True)
        self.cross_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        dec_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=8, batch_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=n_layers)

        self.lm_head = nn.Linear(d_model, vocab, bias=False)

    def forward(self, Z_seq, z_cls, p, tgt_ids, tgt_mask, retrieved=None):
        p_tok = self.p_embed(p).unsqueeze(1)  # [B, 1, d_model]
        K = self.z_proj(Z_seq)                # [B, Tz, d_model]
        if retrieved is not None:
            K = torch.cat([K, self.z_proj(retrieved)], dim=1)
        K = self.cross_encoder(K)

        dec_inp = embed_tokens(tgt_ids)       # assumes tied embeddings
        dec_inp[:, 0:1, :] = dec_inp[:, 0:1, :] + p_tok  # inject personalization at BOS

        out = self.decoder(tgt=dec_inp, memory=K, tgt_mask=tgt_mask)
        logits = self.lm_head(out)
        return logits
