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
    num_concepts: int = 8       # K concept vectors
    concept_dim: int = 128      # dim per concept vector
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

def info_nce_loss(concepts_a, concepts_b, temperature=0.07):
    """
    Symmetric InfoNCE contrastive loss with in-batch negatives.

    concepts_a[i] and concepts_b[i] are positive pairs.
    All other combinations are negatives. Symmetric: both a→b and b→a
    matching, doubling gradient signal.

    Returns: (loss, pos_sim_mean, neg_sim_mean)
    """
    flat_a = concepts_a.view(concepts_a.shape[0], -1)
    flat_b = concepts_b.view(concepts_b.shape[0], -1)
    flat_a = F.normalize(flat_a, p=2, dim=-1)
    flat_b = F.normalize(flat_b, p=2, dim=-1)

    # Similarity matrix: (B, B)
    sim_matrix = flat_a @ flat_b.T / temperature
    labels = torch.arange(sim_matrix.shape[0], device=sim_matrix.device)

    # Symmetric: a→b and b→a
    loss = (F.cross_entropy(sim_matrix, labels) +
            F.cross_entropy(sim_matrix.T, labels)) / 2

    with torch.no_grad():
        pos_sim = (flat_a * flat_b).sum(dim=-1).mean().item()
        mask = ~torch.eye(sim_matrix.shape[0], dtype=torch.bool,
                          device=sim_matrix.device)
        neg_sim = (flat_a @ flat_b.T)[mask].mean().item()

    return loss, pos_sim, neg_sim


def word_order_info_nce(concepts_orig, concepts_shuffled, temperature=0.07):
    """
    InfoNCE for word-order: original should be closer to itself than to
    any shuffled version. Uses the same in-batch negative structure.

    The "positive" for each original is itself (identity), and the shuffled
    versions of all sentences in the batch are negatives.

    Returns: (loss, wo_sim_mean)
    """
    flat_orig = concepts_orig.view(concepts_orig.shape[0], -1)
    flat_shuf = concepts_shuffled.view(concepts_shuffled.shape[0], -1)
    flat_orig = F.normalize(flat_orig, p=2, dim=-1)
    flat_shuf = F.normalize(flat_shuf, p=2, dim=-1)

    B = flat_orig.shape[0]

    # Build (B, 2B) similarity: [orig_vs_orig, orig_vs_shuffled]
    sim_orig = flat_orig @ flat_orig.T / temperature   # (B, B)
    sim_shuf = flat_orig @ flat_shuf.T / temperature   # (B, B)
    sim_all = torch.cat([sim_orig, sim_shuf], dim=1)   # (B, 2B)

    # Positives are on the diagonal of sim_orig (i.e., column i for row i)
    labels = torch.arange(B, device=sim_all.device)

    # Mask out self-similarity from sim_orig diagonal (would be trivial 1.0)
    # Replace with large negative so it doesn't count as the positive
    # Actually: the positive IS the self-similarity. The negatives are
    # everything else (other originals + all shuffled). This pushes the model
    # to make each sentence unique AND different from its shuffled version.
    loss = F.cross_entropy(sim_all, labels)

    with torch.no_grad():
        wo_sim = (flat_orig * flat_shuf).sum(dim=-1).mean().item()

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
