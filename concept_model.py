"""
FLM V4 — Concept Autoencoder
==============================
Encoder-decoder that compresses text into a stack of concept vectors,
then reconstructs it. The bottleneck forces the concept stack to encode
meaning — including word order, binding, and structure — because the
decoder must recover the original text from only the concept vectors.

Architecture:
    Encoder: bidirectional transformer → learned query cross-attention → concept stack
    Bottleneck: K vectors of dim D (e.g., 8 x 128 = 1024 total dims)
    Decoder: causal transformer that cross-attends to concept stack → reconstructed tokens

Training:
    1. Reconstruction loss: cross-entropy on next-token prediction from concept stack only
    2. Paraphrase loss: paraphrase pairs → similar concept stacks (cosine on flattened)
    Combined: the model learns to strip surface variation while preserving everything
    that matters for meaning (including word order and structure).

~45M parameters total (encoder ~20M, decoder ~20M, bottleneck ~5M).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field


@dataclass
class ConceptConfig:
    vocab_size: int = 32000
    # Encoder
    enc_hidden: int = 384
    enc_layers: int = 6
    enc_heads: int = 6
    enc_intermediate: int = 1536
    # Bottleneck
    num_concepts: int = 32      # K concept vectors (slots)
    concept_dim: int = 32       # dim per concept vector
    # Decoder
    dec_hidden: int = 384
    dec_layers: int = 6
    dec_heads: int = 6
    dec_intermediate: int = 1536
    # Shared
    max_seq_len: int = 128
    dropout: float = 0.1
    rms_norm_eps: float = 1e-5

    @property
    def total_concept_dim(self):
        return self.num_concepts * self.concept_dim


# ═══════════════════════════════════════════════════════════════════════
# Building blocks (shared between encoder and decoder)
# ═══════════════════════════════════════════════════════════════════════

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


def precompute_rope(dim, max_seq_len, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len).float()
    freqs = torch.outer(t, freqs)
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):
    seq_len = x.shape[2]
    cos = cos[:seq_len].unsqueeze(0).unsqueeze(0)
    sin = sin[:seq_len].unsqueeze(0).unsqueeze(0)
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    rotated = torch.cat((-x2, x1), dim=-1)
    return x * cos.repeat(1, 1, 1, 2) + rotated * sin.repeat(1, 1, 1, 2)


class SwiGLUFFN(nn.Module):
    def __init__(self, hidden_size, intermediate_size, dropout=0.1):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class SelfAttention(nn.Module):
    """Multi-head self-attention with RoPE. Supports optional causal mask."""

    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.hidden_size = hidden_size
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, rope_cos, rope_sin, attention_mask=None):
        bsz, seq_len, _ = x.shape
        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, rope_cos, rope_sin)
        k = apply_rope(k, rope_cos, rope_sin)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        if attention_mask is not None:
            attn = attn + attention_mask
        attn = F.softmax(attn, dim=-1, dtype=torch.float32).type_as(q)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.hidden_size)
        return self.o_proj(out)


class CrossAttention(nn.Module):
    """Multi-head cross-attention. Query attends to key/value memory."""

    def __init__(self, q_dim, kv_dim, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = q_dim // num_heads
        self.q_dim = q_dim
        self.q_proj = nn.Linear(q_dim, q_dim, bias=False)
        self.k_proj = nn.Linear(kv_dim, q_dim, bias=False)
        self.v_proj = nn.Linear(kv_dim, q_dim, bias=False)
        self.o_proj = nn.Linear(q_dim, q_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, memory):
        """
        query: (batch, q_len, q_dim)
        memory: (batch, m_len, kv_dim)
        """
        bsz, q_len, _ = query.shape
        m_len = memory.shape[1]

        q = self.q_proj(query).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(memory).view(bsz, m_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(memory).view(bsz, m_len, self.num_heads, self.head_dim).transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1, dtype=torch.float32).type_as(q)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(bsz, q_len, self.q_dim)
        return self.o_proj(out)


# ═══════════════════════════════════════════════════════════════════════
# Encoder
# ═══════════════════════════════════════════════════════════════════════

class EncoderBlock(nn.Module):
    def __init__(self, config: ConceptConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.enc_hidden, eps=config.rms_norm_eps)
        self.attention = SelfAttention(config.enc_hidden, config.enc_heads, config.dropout)
        self.ffn_norm = RMSNorm(config.enc_hidden, eps=config.rms_norm_eps)
        self.ffn = SwiGLUFFN(config.enc_hidden, config.enc_intermediate, config.dropout)

    def forward(self, x, rope_cos, rope_sin, attention_mask=None):
        x = x + self.attention(self.attn_norm(x), rope_cos, rope_sin, attention_mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class ConceptBottleneck(nn.Module):
    """
    Learned query tokens cross-attend to encoder output to produce concept stack.

    K learned queries → cross-attention to encoder hidden states → K concept vectors.
    This is the information bottleneck: all meaning must pass through these K vectors.
    """

    def __init__(self, config: ConceptConfig):
        super().__init__()
        self.num_concepts = config.num_concepts
        self.concept_dim = config.concept_dim

        # Learned query tokens
        self.queries = nn.Parameter(torch.randn(config.num_concepts, config.enc_hidden) * 0.02)

        # Cross-attention: queries attend to encoder output
        self.cross_attn = CrossAttention(
            q_dim=config.enc_hidden,
            kv_dim=config.enc_hidden,
            num_heads=config.enc_heads,
            dropout=config.dropout,
        )
        self.norm = RMSNorm(config.enc_hidden, eps=config.rms_norm_eps)

        # Project to concept dim
        self.proj = nn.Linear(config.enc_hidden, config.concept_dim, bias=False)

    def forward(self, encoder_output):
        """
        encoder_output: (batch, seq_len, enc_hidden)
        Returns: (batch, num_concepts, concept_dim) — the concept stack
        """
        bsz = encoder_output.shape[0]

        # Expand queries for batch
        queries = self.queries.unsqueeze(0).expand(bsz, -1, -1)  # (B, K, enc_hidden)

        # Cross-attend to encoder output
        concepts = self.cross_attn(queries, encoder_output)  # (B, K, enc_hidden)
        concepts = self.norm(concepts)

        # Project to concept dim
        concepts = self.proj(concepts)  # (B, K, concept_dim)

        return concepts


# ═══════════════════════════════════════════════════════════════════════
# Decoder
# ═══════════════════════════════════════════════════════════════════════

class DecoderBlock(nn.Module):
    """Decoder block: causal self-attention + cross-attention to concept stack + FFN."""

    def __init__(self, config: ConceptConfig):
        super().__init__()
        # Causal self-attention
        self.self_attn_norm = RMSNorm(config.dec_hidden, eps=config.rms_norm_eps)
        self.self_attn = SelfAttention(config.dec_hidden, config.dec_heads, config.dropout)

        # Cross-attention to concept stack
        self.cross_attn_norm = RMSNorm(config.dec_hidden, eps=config.rms_norm_eps)
        self.cross_attn = CrossAttention(
            q_dim=config.dec_hidden,
            kv_dim=config.concept_dim,
            num_heads=config.dec_heads,
            dropout=config.dropout,
        )

        # FFN
        self.ffn_norm = RMSNorm(config.dec_hidden, eps=config.rms_norm_eps)
        self.ffn = SwiGLUFFN(config.dec_hidden, config.dec_intermediate, config.dropout)

    def forward(self, x, concepts, rope_cos, rope_sin, causal_mask=None):
        """
        x: (batch, seq_len, dec_hidden) — decoder hidden states
        concepts: (batch, num_concepts, concept_dim) — the concept stack
        """
        # Causal self-attention
        x = x + self.self_attn(self.self_attn_norm(x), rope_cos, rope_sin, causal_mask)

        # Cross-attention to concepts
        x = x + self.cross_attn(self.cross_attn_norm(x), concepts)

        # FFN
        x = x + self.ffn(self.ffn_norm(x))
        return x


# ═══════════════════════════════════════════════════════════════════════
# Full Model
# ═══════════════════════════════════════════════════════════════════════

class ConceptAutoencoder(nn.Module):
    """
    Text → concept stack → text.

    The concept stack is the ONLY information path between encoder and decoder.
    This forces it to encode everything needed to reconstruct the input:
    word order, binding, structure, meaning — all in K small vectors.

    For paraphrase training: two texts that mean the same thing should produce
    the same concept stack, because the decoder can reconstruct either surface
    form from the shared meaning.
    """

    def __init__(self, config: ConceptConfig):
        super().__init__()
        self.config = config

        # Shared token embedding (encoder and decoder use same vocab)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.enc_hidden)

        # Encoder (bidirectional)
        head_dim = config.enc_hidden // config.enc_heads
        enc_rope_cos, enc_rope_sin = precompute_rope(head_dim, config.max_seq_len)
        self.register_buffer("enc_rope_cos", enc_rope_cos, persistent=False)
        self.register_buffer("enc_rope_sin", enc_rope_sin, persistent=False)

        self.enc_layers = nn.ModuleList([
            EncoderBlock(config) for _ in range(config.enc_layers)
        ])
        self.enc_norm = RMSNorm(config.enc_hidden, eps=config.rms_norm_eps)

        # Bottleneck
        self.bottleneck = ConceptBottleneck(config)

        # Decoder (causal)
        # Decoder may have different hidden size, so separate embedding projection
        if config.dec_hidden != config.enc_hidden:
            self.dec_embed_proj = nn.Linear(config.enc_hidden, config.dec_hidden, bias=False)
        else:
            self.dec_embed_proj = None

        dec_head_dim = config.dec_hidden // config.dec_heads
        dec_rope_cos, dec_rope_sin = precompute_rope(dec_head_dim, config.max_seq_len)
        self.register_buffer("dec_rope_cos", dec_rope_cos, persistent=False)
        self.register_buffer("dec_rope_sin", dec_rope_sin, persistent=False)

        self.dec_layers = nn.ModuleList([
            DecoderBlock(config) for _ in range(config.dec_layers)
        ])
        self.dec_norm = RMSNorm(config.dec_hidden, eps=config.rms_norm_eps)

        # Output head (tied with embeddings)
        self.lm_head = nn.Linear(config.dec_hidden, config.vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)

    def encode(self, input_ids, attention_mask=None):
        """
        Encode text into concept stack.

        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len) — 1=real, 0=padding

        Returns:
            concepts: (batch, num_concepts, concept_dim) — the concept stack
        """
        bsz, seq_len = input_ids.shape
        x = self.embed_tokens(input_ids)

        # Bidirectional attention mask (pad only, no causal)
        attn_mask = None
        if attention_mask is not None:
            pad_mask = torch.zeros(bsz, 1, 1, seq_len, device=x.device, dtype=x.dtype)
            pad_mask.masked_fill_(attention_mask[:, None, None, :] == 0, float("-inf"))
            attn_mask = pad_mask

        for layer in self.enc_layers:
            x = layer(x, self.enc_rope_cos, self.enc_rope_sin, attn_mask)

        x = self.enc_norm(x)

        # Bottleneck: learned queries cross-attend to encoder output
        concepts = self.bottleneck(x)  # (B, K, concept_dim)

        return concepts

    def decode(self, input_ids, concepts):
        """
        Decode from concept stack. Teacher-forced reconstruction.

        Args:
            input_ids: (batch, seq_len) — target tokens (shifted right for teacher forcing)
            concepts: (batch, num_concepts, concept_dim) — concept stack

        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        bsz, seq_len = input_ids.shape
        x = self.embed_tokens(input_ids)

        if self.dec_embed_proj is not None:
            x = self.dec_embed_proj(x)

        # Causal mask for autoregressive decoding
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=x.device, dtype=x.dtype),
            diagonal=1,
        ).unsqueeze(0).unsqueeze(0)  # (1, 1, seq, seq)

        for layer in self.dec_layers:
            x = layer(x, concepts, self.dec_rope_cos, self.dec_rope_sin, causal_mask)

        x = self.dec_norm(x)
        logits = self.lm_head(x)  # (B, seq_len, vocab_size)

        return logits

    def forward(self, input_ids, attention_mask=None):
        """
        Full forward: encode then decode (teacher-forced).

        For reconstruction training:
            - Input to encoder: full sequence
            - Input to decoder: sequence shifted right (predict next token)
            - Target: original sequence

        Returns:
            logits: (batch, seq_len, vocab_size)
            concepts: (batch, num_concepts, concept_dim)
        """
        concepts = self.encode(input_ids, attention_mask)

        # Decoder input: shift right (prepend a zero/BOS token, drop last)
        # The decoder predicts token[i] from tokens[0..i-1] + concepts
        dec_input = input_ids[:, :-1]  # all but last token

        logits = self.decode(dec_input, concepts)

        return logits, concepts

    def concept_vector(self, input_ids, attention_mask=None):
        """
        Get the flattened, L2-normalized concept vector for similarity comparison.

        Returns: (batch, num_concepts * concept_dim) — unit-norm vector
        """
        concepts = self.encode(input_ids, attention_mask)  # (B, K, D)
        flat = concepts.view(concepts.shape[0], -1)  # (B, K*D)
        return F.normalize(flat, p=2, dim=-1)

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


# ═══════════════════════════════════════════════════════════════════════
# Losses
# ═══════════════════════════════════════════════════════════════════════

def reconstruction_loss(logits, targets):
    """
    Standard cross-entropy reconstruction loss.

    logits: (batch, seq_len, vocab_size)
    targets: (batch, seq_len) — token IDs to predict
    """
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        ignore_index=0,  # ignore padding
    )


def paraphrase_loss(concepts_a, concepts_b, target_sim=0.95):
    """
    Paraphrase pairs should produce similar concept stacks.
    (V2 margin-based, kept for compatibility)
    """
    flat_a = concepts_a.view(concepts_a.shape[0], -1)
    flat_b = concepts_b.view(concepts_b.shape[0], -1)
    flat_a = F.normalize(flat_a, p=2, dim=-1)
    flat_b = F.normalize(flat_b, p=2, dim=-1)

    sim = F.cosine_similarity(flat_a, flat_b)
    loss = F.relu(target_sim - sim).mean()
    return loss, sim.mean().item()


def negative_loss(concepts_a, concepts_b, target_sim=0.3):
    """
    Non-paraphrase pairs should produce distant concept stacks.
    (V2 margin-based, kept for compatibility)
    """
    flat_a = concepts_a.view(concepts_a.shape[0], -1)
    flat_b = concepts_b.view(concepts_b.shape[0], -1)
    flat_a = F.normalize(flat_a, p=2, dim=-1)
    flat_b = F.normalize(flat_b, p=2, dim=-1)

    sim = F.cosine_similarity(flat_a, flat_b)
    loss = F.relu(sim - target_sim).mean()
    return loss, sim.mean().item()


def word_order_loss(concepts_orig, concepts_shuffled, target_sim=0.3):
    """
    Original and word-shuffled versions should have different concept stacks.
    (V2 margin-based, kept for compatibility)
    """
    flat_a = concepts_orig.view(concepts_orig.shape[0], -1)
    flat_b = concepts_shuffled.view(concepts_shuffled.shape[0], -1)
    flat_a = F.normalize(flat_a, p=2, dim=-1)
    flat_b = F.normalize(flat_b, p=2, dim=-1)

    sim = F.cosine_similarity(flat_a, flat_b)
    loss = F.relu(sim - target_sim).mean()
    return loss, sim.mean().item()


# ═══════════════════════════════════════════════════════════════════════
# V3 Losses — InfoNCE + Decorrelation
# ═══════════════════════════════════════════════════════════════════════

def slot_similarity_matrix(concepts_a, concepts_b):
    """
    Compute mean-of-per-slot-cosines similarity matrix.

    Instead of flattening to 1024-dim and computing one cosine, compute
    cosine per slot and average. This gives each slot equal weight so
    a big change in ONE slot (1/32) matters, instead of being drowned
    in flat cosine where it's only 32/1024 = 3.1% of the norm.

    concepts_a: (B1, K, D)
    concepts_b: (B2, K, D)
    Returns: (B1, B2) similarity matrix
    """
    # Normalize each slot independently
    normed_a = F.normalize(concepts_a, p=2, dim=-1)  # (B1, K, D)
    normed_b = F.normalize(concepts_b, p=2, dim=-1)  # (B2, K, D)
    # Per-slot sim: (K, B1, B2) via batched matmul per slot
    # Reshape to (K, B1, D) and (K, B2, D)
    a_t = normed_a.permute(1, 0, 2)  # (K, B1, D)
    b_t = normed_b.permute(1, 0, 2)  # (K, B2, D)
    per_slot_sim = torch.bmm(a_t, b_t.transpose(1, 2))  # (K, B1, B2)
    # Mean across slots
    return per_slot_sim.mean(dim=0)  # (B1, B2)


def info_nce_loss(concepts_a, concepts_b, temperature=0.07):
    """
    Symmetric InfoNCE contrastive loss with in-batch negatives.
    Uses mean-of-per-slot-cosines so each slot contributes equally.

    Returns: (loss, pos_sim_mean, neg_sim_mean)
    """
    # Similarity matrix using slot-aware similarity: (B, B)
    sim_raw = slot_similarity_matrix(concepts_a, concepts_b)
    sim_matrix = sim_raw / temperature
    labels = torch.arange(sim_matrix.shape[0], device=sim_matrix.device)

    # Symmetric: a→b and b→a
    loss = (F.cross_entropy(sim_matrix, labels) +
            F.cross_entropy(sim_matrix.T, labels)) / 2

    with torch.no_grad():
        pos_sim = sim_raw.diag().mean().item()
        mask = ~torch.eye(sim_raw.shape[0], dtype=torch.bool,
                          device=sim_raw.device)
        neg_sim = sim_raw[mask].mean().item()

    return loss, pos_sim, neg_sim


def word_order_info_nce(concepts_orig, concepts_shuffled, temperature=0.07):
    """
    InfoNCE for word-order: original should be closer to itself than to
    any shuffled version. Uses slot-aware similarity.

    Returns: (loss, wo_sim_mean)
    """
    B = concepts_orig.shape[0]

    # Slot-aware similarity matrices
    sim_orig_raw = slot_similarity_matrix(concepts_orig, concepts_orig)  # (B, B)
    sim_shuf_raw = slot_similarity_matrix(concepts_orig, concepts_shuffled)  # (B, B)

    sim_orig = sim_orig_raw / temperature
    sim_shuf = sim_shuf_raw / temperature
    sim_all = torch.cat([sim_orig, sim_shuf], dim=1)  # (B, 2B)

    # Positives are on the diagonal of sim_orig (self-similarity)
    labels = torch.arange(B, device=sim_all.device)
    loss = F.cross_entropy(sim_all, labels)

    with torch.no_grad():
        wo_sim = sim_shuf_raw.diag().mean().item()

    return loss, wo_sim


def slot_decorrelation_loss(concepts):
    """
    Encourage the K concept slots to encode different information per sample.

    concepts: (batch, K, D)

    For each sample, computes cosine similarity between all slot pairs
    and penalizes high similarity. This forces each slot to capture a
    different aspect of the input, increasing effective rank.

    Returns: loss scalar
    """
    B, K, D = concepts.shape
    # Normalize each slot: (B, K, D)
    normed = F.normalize(concepts, p=2, dim=-1)

    # Per-sample slot correlation: (B, K, K)
    corr = torch.bmm(normed, normed.transpose(1, 2))

    # Penalize off-diagonal entries (want each slot different within each sample)
    mask = ~torch.eye(K, dtype=torch.bool, device=corr.device).unsqueeze(0)
    loss = corr[mask.expand(B, -1, -1)].pow(2).mean()

    return loss


def slot_isolation_loss(concepts_base, concepts_variant, target_slot):
    """
    When only one concept axis varies between base and variant,
    only the target slot should change. All other slots should stay the same.

    concepts_base: (B, K, D)
    concepts_variant: (B, K, D)
    target_slot: int — which slot SHOULD change

    Loss = high similarity on non-target slots + low similarity on target slot
    """
    B, K, D = concepts_base.shape
    normed_base = F.normalize(concepts_base, p=2, dim=-1)
    normed_var = F.normalize(concepts_variant, p=2, dim=-1)

    # Per-slot cosine similarity: (B, K)
    slot_sims = (normed_base * normed_var).sum(dim=-1)

    # Non-target slots should be identical (sim -> 1.0)
    mask = torch.ones(K, dtype=torch.bool, device=slot_sims.device)
    mask[target_slot] = False
    unchanged_loss = (1.0 - slot_sims[:, mask]).pow(2).mean()

    # Target slot should be different (sim -> low)
    changed_loss = F.relu(slot_sims[:, target_slot] - 0.5).pow(2).mean()

    return unchanged_loss + changed_loss


# ═══════════════════════════════════════════════════════════════════════
# V6 Losses — Slot Classifiers + Per-Slot Contrastive
# ═══════════════════════════════════════════════════════════════════════

class SlotClassifiers(nn.Module):
    """Auxiliary classification heads for each concept slot.

    Each slot gets a small linear classifier: concept_dim -> num_classes.
    During training, the slot vector must predict the concept_value label.
    Discarded at inference time.
    """
    def __init__(self, concept_dim, num_classes_per_slot):
        """
        num_classes_per_slot: dict[int, int] — slot_id -> num_classes
        """
        super().__init__()
        self.heads = nn.ModuleDict()
        for slot_id, num_classes in num_classes_per_slot.items():
            self.heads[str(slot_id)] = nn.Linear(concept_dim, num_classes)

    def forward(self, concepts, slot_ids, labels):
        """
        concepts: (B, K, D) — concept stack
        slot_ids: (B,) — which slot each example targets
        labels: (B,) — integer class labels for each example

        Returns: cross-entropy loss averaged across examples
        """
        total_loss = 0.0
        count = 0
        for s in slot_ids.unique():
            s_key = str(s.item())
            if s_key not in self.heads:
                continue
            mask = slot_ids == s
            slot_vecs = concepts[mask, s.item(), :]  # (n, D)
            slot_labels = labels[mask]  # (n,)
            logits = self.heads[s_key](slot_vecs)  # (n, C)
            total_loss = total_loss + F.cross_entropy(logits, slot_labels)
            count += 1
        return total_loss / max(count, 1)

    @torch.no_grad()
    def accuracy(self, concepts, slot_ids, labels):
        """Compute classification accuracy per slot."""
        slot_accs = {}
        for s in slot_ids.unique():
            s_key = str(s.item())
            if s_key not in self.heads:
                continue
            mask = slot_ids == s
            slot_vecs = concepts[mask, s.item(), :]
            slot_labels = labels[mask]
            logits = self.heads[s_key](slot_vecs)
            preds = logits.argmax(dim=-1)
            acc = (preds == slot_labels).float().mean().item()
            slot_accs[s.item()] = acc
        return slot_accs


def per_slot_contrastive_loss(concepts, slot_id, labels, temperature=0.07):
    """
    SupCon-style contrastive loss on a single slot's vectors.

    concepts: (B, K, D) — concept stack
    slot_id: int — which slot to compute contrastive on
    labels: (B,) — concept_value class labels (same label = positive pair)
    temperature: float

    Pulls same-label slot vectors together, pushes different-label apart.
    """
    slot_vecs = F.normalize(concepts[:, slot_id, :], p=2, dim=-1)  # (B, D)
    sim = torch.mm(slot_vecs, slot_vecs.T) / temperature  # (B, B)
    B = sim.shape[0]

    # Mask: same label = positive
    label_eq = labels.unsqueeze(0) == labels.unsqueeze(1)  # (B, B)
    # Exclude self
    self_mask = ~torch.eye(B, dtype=torch.bool, device=sim.device)
    pos_mask = label_eq & self_mask

    # Need at least some positives
    if pos_mask.sum() == 0:
        return torch.tensor(0.0, device=sim.device)

    # For numerical stability
    sim_max = sim.max(dim=1, keepdim=True).values.detach()
    sim = sim - sim_max

    # Log-sum-exp over all non-self entries (denominator)
    exp_sim = torch.exp(sim) * self_mask.float()
    log_denom = torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

    # Mean of log(positive / denom) over positive pairs
    log_prob = sim - log_denom
    loss = -(log_prob * pos_mask.float()).sum() / pos_mask.sum()

    return loss


# ═══════════════════════════════════════════════════════════════════════
# V7 Losses — Flat Cosine + Margin + Per-Slot Paraphrase
# ═══════════════════════════════════════════════════════════════════════

def flat_similarity_matrix(concepts_a, concepts_b):
    """
    Flat cosine similarity on concatenated 1024-dim vectors.

    Unlike slot_similarity_matrix (mean-of-per-slot-cosines), this treats
    the full concept stack as one vector. Dead slots contribute near-zero
    norm instead of +1.0 similarity, eliminating the similarity floor.

    concepts_a: (B1, K, D)
    concepts_b: (B2, K, D)
    Returns: (B1, B2) similarity matrix
    """
    flat_a = F.normalize(concepts_a.view(concepts_a.shape[0], -1), p=2, dim=-1)
    flat_b = F.normalize(concepts_b.view(concepts_b.shape[0], -1), p=2, dim=-1)
    return torch.mm(flat_a, flat_b.T)


def flat_info_nce_loss(concepts_a, concepts_b, temperature=0.07):
    """
    Symmetric InfoNCE using flat cosine similarity.
    Replaces V6's slot_similarity_matrix-based NCE.

    Returns: (loss, pos_sim_mean, neg_sim_mean)
    """
    sim_raw = flat_similarity_matrix(concepts_a, concepts_b)
    sim_matrix = sim_raw / temperature
    labels = torch.arange(sim_matrix.shape[0], device=sim_matrix.device)

    loss = (F.cross_entropy(sim_matrix, labels) +
            F.cross_entropy(sim_matrix.T, labels)) / 2

    with torch.no_grad():
        pos_sim = sim_raw.diag().mean().item()
        mask = ~torch.eye(sim_raw.shape[0], dtype=torch.bool,
                          device=sim_raw.device)
        neg_sim = sim_raw[mask].mean().item()

    return loss, pos_sim, neg_sim


def flat_word_order_info_nce(concepts_orig, concepts_shuffled, temperature=0.07):
    """
    InfoNCE for word-order using flat cosine similarity.

    Returns: (loss, wo_sim_mean)
    """
    B = concepts_orig.shape[0]

    sim_orig_raw = flat_similarity_matrix(concepts_orig, concepts_orig)
    sim_shuf_raw = flat_similarity_matrix(concepts_orig, concepts_shuffled)

    sim_orig = sim_orig_raw / temperature
    sim_shuf = sim_shuf_raw / temperature
    sim_all = torch.cat([sim_orig, sim_shuf], dim=1)

    labels = torch.arange(B, device=sim_all.device)
    loss = F.cross_entropy(sim_all, labels)

    with torch.no_grad():
        wo_sim = sim_shuf_raw.diag().mean().item()

    return loss, wo_sim


def margin_paraphrase_loss(concepts_a, concepts_b, target_sim=0.85):
    """
    Margin loss: paraphrase pairs should have flat cosine > target_sim.
    Explicit absolute target, not just relative ranking.

    Returns: (loss, actual_sim_mean)
    """
    flat_a = F.normalize(concepts_a.view(concepts_a.shape[0], -1), p=2, dim=-1)
    flat_b = F.normalize(concepts_b.view(concepts_b.shape[0], -1), p=2, dim=-1)
    sim = (flat_a * flat_b).sum(dim=-1)  # (B,)
    loss = F.relu(target_sim - sim).mean()
    return loss, sim.mean().item()


def margin_negative_loss(concepts_a, concepts_b, target_sim=0.3):
    """
    Margin loss: unrelated pairs should have flat cosine < target_sim.

    Returns: (loss, actual_sim_mean)
    """
    flat_a = F.normalize(concepts_a.view(concepts_a.shape[0], -1), p=2, dim=-1)
    flat_b = F.normalize(concepts_b.view(concepts_b.shape[0], -1), p=2, dim=-1)
    sim = (flat_a * flat_b).sum(dim=-1)
    loss = F.relu(sim - target_sim).mean()
    return loss, sim.mean().item()


def margin_word_order_loss(concepts_orig, concepts_shuffled, target_sim=0.5):
    """
    Margin loss: word-order swaps should have flat cosine < target_sim.

    Returns: (loss, actual_sim_mean)
    """
    flat_a = F.normalize(concepts_orig.view(concepts_orig.shape[0], -1), p=2, dim=-1)
    flat_b = F.normalize(concepts_shuffled.view(concepts_shuffled.shape[0], -1), p=2, dim=-1)
    sim = (flat_a * flat_b).sum(dim=-1)
    loss = F.relu(sim - target_sim).mean()
    return loss, sim.mean().item()


def batch_repulsion_loss(concepts, target_sim=0.3):
    """
    Push random text pairs apart. Takes a batch of concepts from random
    (unrelated) texts and penalizes pairwise similarity above target.

    This prevents the global similarity space from compressing — without it,
    all representations drift toward high mutual similarity.

    concepts: (B, K, D)
    Returns: (loss, mean_pairwise_sim)
    """
    flat = F.normalize(concepts.view(concepts.shape[0], -1), p=2, dim=-1)
    sim = torch.mm(flat, flat.T)  # (B, B)
    # Exclude self-similarity diagonal
    mask = ~torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
    off_diag = sim[mask]
    # Penalize similarities above target
    loss = F.relu(off_diag - target_sim).mean()
    return loss, off_diag.mean().item()


def hard_repulsion_loss(concepts, target_sim=0.05, top_k=8):
    """
    Hard repulsion: penalize the WORST offenders (highest similarity pairs).
    Unlike batch_repulsion which averages over all pairs (diluting gradient),
    this focuses on the top-k most similar non-self pairs per sample.

    This pushes harder on the clustering gap by targeting the pairs that are
    closest together in concept space.

    concepts: (B, K, D)
    Returns: (loss, max_sim)
    """
    flat = F.normalize(concepts.view(concepts.shape[0], -1), p=2, dim=-1)
    sim = torch.mm(flat, flat.T)  # (B, B)
    # Zero out diagonal
    sim = sim - torch.eye(sim.shape[0], device=sim.device) * 2.0
    # For each sample, get top-k most similar
    k = min(top_k, sim.shape[0] - 1)
    top_sims, _ = sim.topk(k, dim=1)  # (B, k)
    # Penalize those above target
    loss = F.relu(top_sims - target_sim).mean()
    max_sim = top_sims[:, 0].mean().item()
    return loss, max_sim


def per_slot_paraphrase_loss(concepts_a, concepts_b):
    """
    Per-slot paraphrase consistency on REAL text pairs.
    For each of 32 slots, paraphrase pairs should produce similar slot vectors.

    This bridges the gap between synthetic-only slot training and real text.

    concepts_a: (B, K, D)
    concepts_b: (B, K, D)
    Returns: (loss, mean_slot_sim)
    """
    normed_a = F.normalize(concepts_a, p=2, dim=-1)  # (B, K, D)
    normed_b = F.normalize(concepts_b, p=2, dim=-1)
    # Per-slot cosine: (B, K)
    slot_sims = (normed_a * normed_b).sum(dim=-1)
    # Each slot should be close for paraphrases — target sim ~0.9
    loss = F.relu(0.85 - slot_sims).mean()
    return loss, slot_sims.mean().item()
