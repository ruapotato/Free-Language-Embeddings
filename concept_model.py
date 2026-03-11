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
    # FR decoder (V13+)
    fr_vocab_size: int = 32005   # CamemBERT vocab
    # ES decoder (V14)
    es_vocab_size: int = 31002   # BETO vocab
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

class ParallelDecoderBlock(nn.Module):
    """Non-autoregressive decoder block: bidirectional self-attention + cross-attention to concept stack + FFN.

    Unlike DecoderBlock, there is NO causal mask. Positions can see all other
    positions, but the only information source is concept vectors (no input tokens).
    This forces concept vectors to carry all meaning.
    """

    def __init__(self, config: ConceptConfig):
        super().__init__()
        # Bidirectional self-attention (no causal mask)
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

    def forward(self, x, concepts, rope_cos, rope_sin, attention_mask=None):
        """
        x: (batch, seq_len, dec_hidden) — position query hidden states
        concepts: (batch, num_concepts, concept_dim) — the concept stack
        attention_mask: optional (batch, 1, 1, seq_len) additive mask for padding
        No causal mask — all positions see each other, but padding is masked out.
        """
        # Bidirectional self-attention (positions coordinate with each other)
        x = x + self.self_attn(self.self_attn_norm(x), rope_cos, rope_sin, attention_mask=attention_mask)

        # Cross-attention to concepts
        x = x + self.cross_attn(self.cross_attn_norm(x), concepts)

        # FFN
        x = x + self.ffn(self.ffn_norm(x))
        return x


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
# V10 — Non-Autoregressive Concept Autoencoder
# ═══════════════════════════════════════════════════════════════════════

class ConceptAutoencoderV10(nn.Module):
    """
    Text → concept stack → text (parallel decode).

    V10 replaces the autoregressive decoder with a non-autoregressive parallel
    decoder. Learned position queries cross-attend to concept vectors, and each
    position independently predicts its token. No teacher forcing, no causal mask.

    This forces the concept stack to encode EVERYTHING — word order, binding,
    structure — because the decoder has no other information source.
    """

    def __init__(self, config: ConceptConfig):
        super().__init__()
        self.config = config

        # Shared token embedding (encoder only in V10)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.enc_hidden)

        # Encoder (bidirectional) — same as V5-V9
        head_dim = config.enc_hidden // config.enc_heads
        enc_rope_cos, enc_rope_sin = precompute_rope(head_dim, config.max_seq_len)
        self.register_buffer("enc_rope_cos", enc_rope_cos, persistent=False)
        self.register_buffer("enc_rope_sin", enc_rope_sin, persistent=False)

        self.enc_layers = nn.ModuleList([
            EncoderBlock(config) for _ in range(config.enc_layers)
        ])
        self.enc_norm = RMSNorm(config.enc_hidden, eps=config.rms_norm_eps)

        # Bottleneck — same as V5-V9
        self.bottleneck = ConceptBottleneck(config)

        # Parallel decoder (non-autoregressive)
        # Learned position queries — one per output position
        self.position_queries = nn.Parameter(
            torch.zeros(config.max_seq_len, config.dec_hidden))
        # Initialize with sinusoidal pattern for diversity
        self._init_position_queries()

        dec_head_dim = config.dec_hidden // config.dec_heads
        dec_rope_cos, dec_rope_sin = precompute_rope(dec_head_dim, config.max_seq_len)
        self.register_buffer("dec_rope_cos", dec_rope_cos, persistent=False)
        self.register_buffer("dec_rope_sin", dec_rope_sin, persistent=False)

        self.par_dec_layers = nn.ModuleList([
            ParallelDecoderBlock(config) for _ in range(config.dec_layers)
        ])
        self.dec_norm = RMSNorm(config.dec_hidden, eps=config.rms_norm_eps)

        # Output head
        self.lm_head = nn.Linear(config.dec_hidden, config.vocab_size, bias=False)

        self.apply(self._init_weights)
        # Re-init position queries after _init_weights would have overwritten
        self._init_position_queries()

    def _init_position_queries(self):
        """Initialize position queries with sinusoidal pattern for diversity."""
        d = self.config.dec_hidden
        pos = torch.arange(self.config.max_seq_len).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2).float() * -(math.log(10000.0) / d))
        self.position_queries.data[:, 0::2] = torch.sin(pos * div)
        self.position_queries.data[:, 1::2] = torch.cos(pos * div)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)

    def encode(self, input_ids, attention_mask=None):
        """Encode text into concept stack. Same as V5-V9."""
        bsz, seq_len = input_ids.shape
        x = self.embed_tokens(input_ids)

        attn_mask = None
        if attention_mask is not None:
            pad_mask = torch.zeros(bsz, 1, 1, seq_len, device=x.device, dtype=x.dtype)
            pad_mask.masked_fill_(attention_mask[:, None, None, :] == 0, float("-inf"))
            attn_mask = pad_mask

        for layer in self.enc_layers:
            x = layer(x, self.enc_rope_cos, self.enc_rope_sin, attn_mask)

        x = self.enc_norm(x)
        concepts = self.bottleneck(x)
        return concepts

    def decode_parallel(self, concepts, seq_len, attention_mask=None):
        """
        Non-autoregressive decode from concept stack.

        Args:
            concepts: (batch, num_concepts, concept_dim)
            seq_len: int — number of output positions to predict
            attention_mask: optional (batch, seq_len) with 1=real, 0=padding.
                Masks padding positions in decoder self-attention so the model
                doesn't depend on batch padding context.

        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        bsz = concepts.shape[0]

        # Slice position queries to actual sequence length
        x = self.position_queries[:seq_len].unsqueeze(0).expand(bsz, -1, -1)

        # Build additive attention mask for padding
        dec_attn_mask = None
        if attention_mask is not None:
            dec_attn_mask = torch.zeros(bsz, 1, 1, seq_len, device=x.device, dtype=x.dtype)
            dec_attn_mask.masked_fill_(attention_mask[:, None, None, :] == 0, float("-inf"))

        for layer in self.par_dec_layers:
            x = layer(x, concepts, self.dec_rope_cos, self.dec_rope_sin,
                      attention_mask=dec_attn_mask)

        x = self.dec_norm(x)
        logits = self.lm_head(x)
        return logits

    def forward(self, input_ids, attention_mask=None):
        """
        Full forward: encode then parallel decode.

        Returns:
            logits: (batch, seq_len, vocab_size) — one prediction per input position
            concepts: (batch, num_concepts, concept_dim)
        """
        concepts = self.encode(input_ids, attention_mask)
        logits = self.decode_parallel(concepts, seq_len=input_ids.shape[1],
                                      attention_mask=attention_mask)
        return logits, concepts

    def concept_vector(self, input_ids, attention_mask=None):
        """Get flattened, L2-normalized concept vector for similarity comparison."""
        concepts = self.encode(input_ids, attention_mask)
        flat = concepts.view(concepts.shape[0], -1)
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


def analogy_loss(concepts_a, concepts_b, concepts_c, concepts_d, target_sim=0.9):
    """
    Analogy loss: encourage a - b + c ≈ d in concept space.
    Given analogy quads (a, b, c, d), the predicted vector (a - b + c)
    should have high cosine similarity to d.

    concepts_*: (B, K, D) concept tensors
    Returns: (loss, mean_sim)
    """
    flat_a = F.normalize(concepts_a.view(concepts_a.shape[0], -1), p=2, dim=-1)
    flat_b = F.normalize(concepts_b.view(concepts_b.shape[0], -1), p=2, dim=-1)
    flat_c = F.normalize(concepts_c.view(concepts_c.shape[0], -1), p=2, dim=-1)
    flat_d = F.normalize(concepts_d.view(concepts_d.shape[0], -1), p=2, dim=-1)
    predicted = F.normalize(flat_a - flat_b + flat_c, p=2, dim=-1)
    sim = (predicted * flat_d).sum(dim=-1)
    # Penalize when similarity is below target
    loss = F.relu(target_sim - sim).mean()
    return loss, sim.mean().item()


def direction_consistency_loss(direction_groups, target_sim=0.8):
    """
    Direction consistency loss: for each semantic attribute (negation, tense, etc.),
    the direction vectors (a - b) across different example pairs should be consistent.

    direction_groups: list of tensors, each (N_pairs, flat_dim) representing
                      normalized direction vectors for one attribute.
    target_sim: minimum pairwise cosine similarity between direction vectors.
    Returns: (loss, mean_consistency)
    """
    total_loss = 0.0
    total_sim = 0.0
    n_groups = 0

    for directions in direction_groups:
        if directions.shape[0] < 2:
            continue
        # directions: (N, D) - already normalized
        sim = torch.mm(directions, directions.T)  # (N, N)
        mask = ~torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
        off_diag = sim[mask]
        # Penalize when pairwise similarity is below target
        total_loss = total_loss + F.relu(target_sim - off_diag).mean()
        total_sim = total_sim + off_diag.mean().item()
        n_groups += 1

    if n_groups == 0:
        return torch.tensor(0.0, requires_grad=True), 0.0

    return total_loss / n_groups, total_sim / n_groups


def cluster_separation_loss(within_concepts, between_concepts,
                            within_target=0.5, between_target=0.2):
    """
    Contrastive clustering loss: push same-group sentences together,
    different-group sentences apart.

    within_concepts: list of (N, K, D) tensors — sentences from same group
    between_concepts: list of ((K, D), (K, D)) pairs — sentences from different groups
    within_target: minimum pairwise sim for same-group pairs
    between_target: maximum pairwise sim for different-group pairs
    Returns: (loss, within_mean, between_mean)
    """
    within_loss = 0.0
    within_total = 0.0
    n_within = 0
    for concepts in within_concepts:
        flat = F.normalize(concepts.view(concepts.shape[0], -1), p=2, dim=-1)
        sim = torch.mm(flat, flat.T)
        mask = ~torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
        off_diag = sim[mask]
        within_loss = within_loss + F.relu(within_target - off_diag).mean()
        within_total += off_diag.mean().item()
        n_within += 1

    between_loss = 0.0
    between_total = 0.0
    n_between = 0
    for c_a, c_b in between_concepts:
        flat_a = F.normalize(c_a.view(1, -1), p=2, dim=-1)
        flat_b = F.normalize(c_b.view(1, -1), p=2, dim=-1)
        sim = (flat_a * flat_b).sum()
        between_loss = between_loss + F.relu(sim - between_target)
        between_total += sim.item()
        n_between += 1

    loss = torch.tensor(0.0, device=within_concepts[0].device, requires_grad=True)
    if n_within > 0:
        loss = loss + within_loss / n_within
    if n_between > 0:
        loss = loss + between_loss / n_between

    w_mean = within_total / max(n_within, 1)
    b_mean = between_total / max(n_between, 1)
    return loss, w_mean, b_mean


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


# ═══════════════════════════════════════════════════════════════════════
# V13 — Dual-Decoder Concept Autoencoder (EN→concepts→EN + EN→concepts→FR)
# ═══════════════════════════════════════════════════════════════════════

class ConceptAutoencoderV13(nn.Module):
    """
    Text → concept stack → (EN reconstruction + FR translation).

    V13 adds a second parallel decoder that translates from concept vectors
    to French. Both decoders share the SAME concept bottleneck. This forces
    language-independent meaning encoding because:
      - The FR decoder can't rely on English surface tokens
      - Word order differs between EN and FR
      - The encoder must capture meaning, not just token patterns

    Encoder + bottleneck + EN decoder: same as V10/V12
    FR decoder: separate parallel decoder with its own embeddings and output head
    """

    def __init__(self, config: ConceptConfig):
        super().__init__()
        self.config = config

        # --- Encoder (same as V10) ---
        self.embed_tokens = nn.Embedding(config.vocab_size, config.enc_hidden)

        head_dim = config.enc_hidden // config.enc_heads
        enc_rope_cos, enc_rope_sin = precompute_rope(head_dim, config.max_seq_len)
        self.register_buffer("enc_rope_cos", enc_rope_cos, persistent=False)
        self.register_buffer("enc_rope_sin", enc_rope_sin, persistent=False)

        self.enc_layers = nn.ModuleList([
            EncoderBlock(config) for _ in range(config.enc_layers)
        ])
        self.enc_norm = RMSNorm(config.enc_hidden, eps=config.rms_norm_eps)

        # --- Bottleneck (same as V10) ---
        self.bottleneck = ConceptBottleneck(config)

        # --- EN Parallel Decoder (same as V10) ---
        self.position_queries = nn.Parameter(
            torch.zeros(config.max_seq_len, config.dec_hidden))

        dec_head_dim = config.dec_hidden // config.dec_heads
        dec_rope_cos, dec_rope_sin = precompute_rope(dec_head_dim, config.max_seq_len)
        self.register_buffer("dec_rope_cos", dec_rope_cos, persistent=False)
        self.register_buffer("dec_rope_sin", dec_rope_sin, persistent=False)

        self.par_dec_layers = nn.ModuleList([
            ParallelDecoderBlock(config) for _ in range(config.dec_layers)
        ])
        self.dec_norm = RMSNorm(config.dec_hidden, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.dec_hidden, config.vocab_size, bias=False)

        # --- FR Parallel Decoder (NEW) ---
        self.fr_vocab_size = config.fr_vocab_size
        self.position_queries_fr = nn.Parameter(
            torch.zeros(config.max_seq_len, config.dec_hidden))

        self.fr_dec_layers = nn.ModuleList([
            ParallelDecoderBlock(config) for _ in range(config.dec_layers)
        ])
        self.fr_dec_norm = RMSNorm(config.dec_hidden, eps=config.rms_norm_eps)
        self.fr_lm_head = nn.Linear(config.dec_hidden, config.fr_vocab_size, bias=False)

        self.apply(self._init_weights)
        # Re-init position queries after _init_weights
        self._init_position_queries(self.position_queries)
        self._init_position_queries(self.position_queries_fr)

    def _init_position_queries(self, param):
        d = self.config.dec_hidden
        pos = torch.arange(self.config.max_seq_len).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2).float() * -(math.log(10000.0) / d))
        param.data[:, 0::2] = torch.sin(pos * div)
        param.data[:, 1::2] = torch.cos(pos * div)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)

    def encode(self, input_ids, attention_mask=None):
        """Encode EN text into concept stack."""
        bsz, seq_len = input_ids.shape
        x = self.embed_tokens(input_ids)

        attn_mask = None
        if attention_mask is not None:
            pad_mask = torch.zeros(bsz, 1, 1, seq_len, device=x.device, dtype=x.dtype)
            pad_mask.masked_fill_(attention_mask[:, None, None, :] == 0, float("-inf"))
            attn_mask = pad_mask

        for layer in self.enc_layers:
            x = layer(x, self.enc_rope_cos, self.enc_rope_sin, attn_mask)

        x = self.enc_norm(x)
        concepts = self.bottleneck(x)
        return concepts

    def decode_parallel(self, concepts, seq_len, attention_mask=None):
        """EN parallel decode from concept stack (same as V10)."""
        bsz = concepts.shape[0]
        x = self.position_queries[:seq_len].unsqueeze(0).expand(bsz, -1, -1)

        dec_attn_mask = None
        if attention_mask is not None:
            dec_attn_mask = torch.zeros(bsz, 1, 1, seq_len, device=x.device, dtype=x.dtype)
            dec_attn_mask.masked_fill_(attention_mask[:, None, None, :] == 0, float("-inf"))

        for layer in self.par_dec_layers:
            x = layer(x, concepts, self.dec_rope_cos, self.dec_rope_sin,
                      attention_mask=dec_attn_mask)

        x = self.dec_norm(x)
        logits = self.lm_head(x)
        return logits

    def decode_parallel_fr(self, concepts, seq_len, attention_mask=None):
        """FR parallel decode from concept stack."""
        bsz = concepts.shape[0]
        x = self.position_queries_fr[:seq_len].unsqueeze(0).expand(bsz, -1, -1)

        dec_attn_mask = None
        if attention_mask is not None:
            dec_attn_mask = torch.zeros(bsz, 1, 1, seq_len, device=x.device, dtype=x.dtype)
            dec_attn_mask.masked_fill_(attention_mask[:, None, None, :] == 0, float("-inf"))

        for layer in self.fr_dec_layers:
            x = layer(x, concepts, self.dec_rope_cos, self.dec_rope_sin,
                      attention_mask=dec_attn_mask)

        x = self.fr_dec_norm(x)
        logits = self.fr_lm_head(x)
        return logits

    def forward(self, input_ids, attention_mask=None,
                fr_seq_len=None, fr_attention_mask=None):
        """
        Full forward: encode EN, decode EN + optionally decode FR.

        Returns:
            en_logits: (batch, en_seq_len, en_vocab_size)
            fr_logits: (batch, fr_seq_len, fr_vocab_size) or None
            concepts: (batch, num_concepts, concept_dim)
        """
        concepts = self.encode(input_ids, attention_mask)
        en_logits = self.decode_parallel(concepts, seq_len=input_ids.shape[1],
                                         attention_mask=attention_mask)

        fr_logits = None
        if fr_seq_len is not None:
            fr_logits = self.decode_parallel_fr(concepts, seq_len=fr_seq_len,
                                                attention_mask=fr_attention_mask)

        return en_logits, fr_logits, concepts

    def concept_vector(self, input_ids, attention_mask=None):
        """Get flattened, L2-normalized concept vector for similarity comparison."""
        concepts = self.encode(input_ids, attention_mask)
        flat = concepts.view(concepts.shape[0], -1)
        return F.normalize(flat, p=2, dim=-1)

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


def translation_loss(logits, targets, pad_token_id=1):
    """Cross-entropy for FR decoder output vs FR target tokens."""
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        ignore_index=pad_token_id,
    )


class ConceptAutoencoderV14(nn.Module):
    """
    V14 Hydra: tight bottleneck (16×16=256) with 5 parallel decoder heads.

    One encoder compresses EN into a 256-dim concept space, then 5 decoders
    reconstruct different outputs from the same bottleneck:
      1. EN reconstruction (BERT vocab)
      2. FR translation (CamemBERT vocab)
      3. ES translation (BETO vocab)
      4. EN paraphrase (BERT vocab, different surface form)
      5. Semantic parse (BERT vocab, structured output)

    The tight bottleneck + diverse decoders force the concept space to encode
    language-independent compositional meaning — not surface tokens.
    """

    def __init__(self, config: ConceptConfig):
        super().__init__()
        self.config = config

        # --- Encoder (same architecture, reads concept_dim from config) ---
        self.embed_tokens = nn.Embedding(config.vocab_size, config.enc_hidden)

        head_dim = config.enc_hidden // config.enc_heads
        enc_rope_cos, enc_rope_sin = precompute_rope(head_dim, config.max_seq_len)
        self.register_buffer("enc_rope_cos", enc_rope_cos, persistent=False)
        self.register_buffer("enc_rope_sin", enc_rope_sin, persistent=False)

        self.enc_layers = nn.ModuleList([
            EncoderBlock(config) for _ in range(config.enc_layers)
        ])
        self.enc_norm = RMSNorm(config.enc_hidden, eps=config.rms_norm_eps)

        # --- Tight Bottleneck (uses config.num_concepts × config.concept_dim) ---
        self.bottleneck = ConceptBottleneck(config)

        # --- Shared decoder RoPE ---
        dec_head_dim = config.dec_hidden // config.dec_heads
        dec_rope_cos, dec_rope_sin = precompute_rope(dec_head_dim, config.max_seq_len)
        self.register_buffer("dec_rope_cos", dec_rope_cos, persistent=False)
        self.register_buffer("dec_rope_sin", dec_rope_sin, persistent=False)

        # --- Head 1: EN Reconstruction ---
        self.en_pos_queries = nn.Parameter(
            torch.zeros(config.max_seq_len, config.dec_hidden))
        self.en_dec_layers = nn.ModuleList([
            ParallelDecoderBlock(config) for _ in range(config.dec_layers)
        ])
        self.en_dec_norm = RMSNorm(config.dec_hidden, eps=config.rms_norm_eps)
        self.en_lm_head = nn.Linear(config.dec_hidden, config.vocab_size, bias=False)

        # --- Head 2: FR Translation ---
        self.fr_pos_queries = nn.Parameter(
            torch.zeros(config.max_seq_len, config.dec_hidden))
        self.fr_dec_layers = nn.ModuleList([
            ParallelDecoderBlock(config) for _ in range(config.dec_layers)
        ])
        self.fr_dec_norm = RMSNorm(config.dec_hidden, eps=config.rms_norm_eps)
        self.fr_lm_head = nn.Linear(config.dec_hidden, config.fr_vocab_size, bias=False)

        # --- Head 3: ES Translation ---
        self.es_pos_queries = nn.Parameter(
            torch.zeros(config.max_seq_len, config.dec_hidden))
        self.es_dec_layers = nn.ModuleList([
            ParallelDecoderBlock(config) for _ in range(config.dec_layers)
        ])
        self.es_dec_norm = RMSNorm(config.dec_hidden, eps=config.rms_norm_eps)
        self.es_lm_head = nn.Linear(config.dec_hidden, config.es_vocab_size, bias=False)

        # --- Head 4: EN Paraphrase ---
        self.para_pos_queries = nn.Parameter(
            torch.zeros(config.max_seq_len, config.dec_hidden))
        self.para_dec_layers = nn.ModuleList([
            ParallelDecoderBlock(config) for _ in range(config.dec_layers)
        ])
        self.para_dec_norm = RMSNorm(config.dec_hidden, eps=config.rms_norm_eps)
        self.para_lm_head = nn.Linear(config.dec_hidden, config.vocab_size, bias=False)

        # --- Head 5: Semantic Parse ---
        self.parse_pos_queries = nn.Parameter(
            torch.zeros(config.max_seq_len, config.dec_hidden))
        self.parse_dec_layers = nn.ModuleList([
            ParallelDecoderBlock(config) for _ in range(config.dec_layers)
        ])
        self.parse_dec_norm = RMSNorm(config.dec_hidden, eps=config.rms_norm_eps)
        self.parse_lm_head = nn.Linear(config.dec_hidden, config.vocab_size, bias=False)

        self.apply(self._init_weights)
        # Init position queries with sinusoidal patterns
        for pq in [self.en_pos_queries, self.fr_pos_queries, self.es_pos_queries,
                    self.para_pos_queries, self.parse_pos_queries]:
            self._init_position_queries(pq)

    def _init_position_queries(self, param):
        d = self.config.dec_hidden
        pos = torch.arange(self.config.max_seq_len).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2).float() * -(math.log(10000.0) / d))
        param.data[:, 0::2] = torch.sin(pos * div)
        param.data[:, 1::2] = torch.cos(pos * div)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)

    def encode(self, input_ids, attention_mask=None):
        """Encode EN text into concept stack [B, num_concepts, concept_dim]."""
        bsz, seq_len = input_ids.shape
        x = self.embed_tokens(input_ids)

        attn_mask = None
        if attention_mask is not None:
            pad_mask = torch.zeros(bsz, 1, 1, seq_len, device=x.device, dtype=x.dtype)
            pad_mask.masked_fill_(attention_mask[:, None, None, :] == 0, float("-inf"))
            attn_mask = pad_mask

        for layer in self.enc_layers:
            x = layer(x, self.enc_rope_cos, self.enc_rope_sin, attn_mask)

        x = self.enc_norm(x)
        concepts = self.bottleneck(x)
        return concepts

    def _decode_head(self, concepts, seq_len, pos_queries, dec_layers, dec_norm,
                     lm_head, attention_mask=None):
        """Generic parallel decode through one decoder head."""
        bsz = concepts.shape[0]
        x = pos_queries[:seq_len].unsqueeze(0).expand(bsz, -1, -1)

        dec_attn_mask = None
        if attention_mask is not None:
            dec_attn_mask = torch.zeros(bsz, 1, 1, seq_len, device=x.device, dtype=x.dtype)
            dec_attn_mask.masked_fill_(attention_mask[:, None, None, :] == 0, float("-inf"))

        for layer in dec_layers:
            x = layer(x, concepts, self.dec_rope_cos, self.dec_rope_sin,
                      attention_mask=dec_attn_mask)

        x = dec_norm(x)
        return lm_head(x)

    def decode_en(self, concepts, seq_len, attention_mask=None):
        return self._decode_head(concepts, seq_len, self.en_pos_queries,
                                 self.en_dec_layers, self.en_dec_norm,
                                 self.en_lm_head, attention_mask)

    def decode_fr(self, concepts, seq_len, attention_mask=None):
        return self._decode_head(concepts, seq_len, self.fr_pos_queries,
                                 self.fr_dec_layers, self.fr_dec_norm,
                                 self.fr_lm_head, attention_mask)

    def decode_es(self, concepts, seq_len, attention_mask=None):
        return self._decode_head(concepts, seq_len, self.es_pos_queries,
                                 self.es_dec_layers, self.es_dec_norm,
                                 self.es_lm_head, attention_mask)

    def decode_para(self, concepts, seq_len, attention_mask=None):
        return self._decode_head(concepts, seq_len, self.para_pos_queries,
                                 self.para_dec_layers, self.para_dec_norm,
                                 self.para_lm_head, attention_mask)

    def decode_parse(self, concepts, seq_len, attention_mask=None):
        return self._decode_head(concepts, seq_len, self.parse_pos_queries,
                                 self.parse_dec_layers, self.parse_dec_norm,
                                 self.parse_lm_head, attention_mask)

    # Alias for probe_geometry.py compatibility
    def decode_parallel(self, concepts, seq_len, attention_mask=None):
        return self.decode_en(concepts, seq_len, attention_mask)

    def forward(self, input_ids, attention_mask=None, targets=None):
        """
        Forward pass. targets is a dict with optional keys:
            'fr': (ids, mask), 'es': (ids, mask), 'para': (ids, mask), 'parse': (ids, mask)
        Returns dict of logits + concepts.
        """
        concepts = self.encode(input_ids, attention_mask)

        result = {'concepts': concepts}
        result['en_logits'] = self.decode_en(concepts, input_ids.shape[1], attention_mask)

        if targets:
            for key, decode_fn in [('fr', self.decode_fr), ('es', self.decode_es),
                                    ('para', self.decode_para), ('parse', self.decode_parse)]:
                if key in targets:
                    ids, mask = targets[key]
                    result[f'{key}_logits'] = decode_fn(concepts, ids.shape[1], mask)

        return result

    def concept_vector(self, input_ids, attention_mask=None):
        concepts = self.encode(input_ids, attention_mask)
        flat = concepts.view(concepts.shape[0], -1)
        return F.normalize(flat, p=2, dim=-1)

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


# ═══════════════════════════════════════════════════════════════════════
# V16 — Full Hydra: 5 heads (EN + FR + ES + Para + Parse), deeper decoders (6L)
#        + programmatic geometry data, no geo gate
# ═══════════════════════════════════════════════════════════════════════

class ConceptAutoencoderV16(nn.Module):
    """
    V16: Full Hydra with 5 decoder heads and deeper decoders.

    All 5 heads from V14, but with 6-layer decoders:
      1. EN reconstruction (BERT vocab)
      2. FR translation (CamemBERT vocab)
      3. ES translation (BETO vocab)
      4. EN paraphrase (BERT vocab, different surface form)
      5. Semantic parse (BERT vocab, structured output)

    Key changes from V14:
      - 6 decoder layers instead of 4 (deeper reasoning per head)
      - Programmatic geometry data (train/test vocab splits)
      - No geo gate — geometry warmup from step 0
    """

    def __init__(self, config: ConceptConfig):
        super().__init__()
        self.config = config

        # --- Encoder ---
        self.embed_tokens = nn.Embedding(config.vocab_size, config.enc_hidden)

        head_dim = config.enc_hidden // config.enc_heads
        enc_rope_cos, enc_rope_sin = precompute_rope(head_dim, config.max_seq_len)
        self.register_buffer("enc_rope_cos", enc_rope_cos, persistent=False)
        self.register_buffer("enc_rope_sin", enc_rope_sin, persistent=False)

        self.enc_layers = nn.ModuleList([
            EncoderBlock(config) for _ in range(config.enc_layers)
        ])
        self.enc_norm = RMSNorm(config.enc_hidden, eps=config.rms_norm_eps)

        # --- Tight Bottleneck ---
        self.bottleneck = ConceptBottleneck(config)

        # --- Shared decoder RoPE ---
        dec_head_dim = config.dec_hidden // config.dec_heads
        dec_rope_cos, dec_rope_sin = precompute_rope(dec_head_dim, config.max_seq_len)
        self.register_buffer("dec_rope_cos", dec_rope_cos, persistent=False)
        self.register_buffer("dec_rope_sin", dec_rope_sin, persistent=False)

        # --- Head 1: EN Reconstruction ---
        self.en_pos_queries = nn.Parameter(
            torch.zeros(config.max_seq_len, config.dec_hidden))
        self.en_dec_layers = nn.ModuleList([
            ParallelDecoderBlock(config) for _ in range(config.dec_layers)
        ])
        self.en_dec_norm = RMSNorm(config.dec_hidden, eps=config.rms_norm_eps)
        self.en_lm_head = nn.Linear(config.dec_hidden, config.vocab_size, bias=False)

        # --- Head 2: FR Translation ---
        self.fr_pos_queries = nn.Parameter(
            torch.zeros(config.max_seq_len, config.dec_hidden))
        self.fr_dec_layers = nn.ModuleList([
            ParallelDecoderBlock(config) for _ in range(config.dec_layers)
        ])
        self.fr_dec_norm = RMSNorm(config.dec_hidden, eps=config.rms_norm_eps)
        self.fr_lm_head = nn.Linear(config.dec_hidden, config.fr_vocab_size, bias=False)

        # --- Head 3: ES Translation ---
        self.es_pos_queries = nn.Parameter(
            torch.zeros(config.max_seq_len, config.dec_hidden))
        self.es_dec_layers = nn.ModuleList([
            ParallelDecoderBlock(config) for _ in range(config.dec_layers)
        ])
        self.es_dec_norm = RMSNorm(config.dec_hidden, eps=config.rms_norm_eps)
        self.es_lm_head = nn.Linear(config.dec_hidden, config.es_vocab_size, bias=False)

        # --- Head 4: EN Paraphrase ---
        self.para_pos_queries = nn.Parameter(
            torch.zeros(config.max_seq_len, config.dec_hidden))
        self.para_dec_layers = nn.ModuleList([
            ParallelDecoderBlock(config) for _ in range(config.dec_layers)
        ])
        self.para_dec_norm = RMSNorm(config.dec_hidden, eps=config.rms_norm_eps)
        self.para_lm_head = nn.Linear(config.dec_hidden, config.vocab_size, bias=False)

        # --- Head 5: Semantic Parse ---
        self.parse_pos_queries = nn.Parameter(
            torch.zeros(config.max_seq_len, config.dec_hidden))
        self.parse_dec_layers = nn.ModuleList([
            ParallelDecoderBlock(config) for _ in range(config.dec_layers)
        ])
        self.parse_dec_norm = RMSNorm(config.dec_hidden, eps=config.rms_norm_eps)
        self.parse_lm_head = nn.Linear(config.dec_hidden, config.vocab_size, bias=False)

        self.apply(self._init_weights)
        for pq in [self.en_pos_queries, self.fr_pos_queries, self.es_pos_queries,
                    self.para_pos_queries, self.parse_pos_queries]:
            self._init_position_queries(pq)

    def _init_position_queries(self, param):
        d = self.config.dec_hidden
        pos = torch.arange(self.config.max_seq_len).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2).float() * -(math.log(10000.0) / d))
        param.data[:, 0::2] = torch.sin(pos * div)
        param.data[:, 1::2] = torch.cos(pos * div)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)

    def encode(self, input_ids, attention_mask=None):
        bsz, seq_len = input_ids.shape
        x = self.embed_tokens(input_ids)

        attn_mask = None
        if attention_mask is not None:
            pad_mask = torch.zeros(bsz, 1, 1, seq_len, device=x.device, dtype=x.dtype)
            pad_mask.masked_fill_(attention_mask[:, None, None, :] == 0, float("-inf"))
            attn_mask = pad_mask

        for layer in self.enc_layers:
            x = layer(x, self.enc_rope_cos, self.enc_rope_sin, attn_mask)

        x = self.enc_norm(x)
        concepts = self.bottleneck(x)
        return concepts

    def _decode_head(self, concepts, seq_len, pos_queries, dec_layers, dec_norm,
                     lm_head, attention_mask=None):
        bsz = concepts.shape[0]
        x = pos_queries[:seq_len].unsqueeze(0).expand(bsz, -1, -1)

        dec_attn_mask = None
        if attention_mask is not None:
            dec_attn_mask = torch.zeros(bsz, 1, 1, seq_len, device=x.device, dtype=x.dtype)
            dec_attn_mask.masked_fill_(attention_mask[:, None, None, :] == 0, float("-inf"))

        for layer in dec_layers:
            x = layer(x, concepts, self.dec_rope_cos, self.dec_rope_sin,
                      attention_mask=dec_attn_mask)

        x = dec_norm(x)
        return lm_head(x)

    def decode_en(self, concepts, seq_len, attention_mask=None):
        return self._decode_head(concepts, seq_len, self.en_pos_queries,
                                 self.en_dec_layers, self.en_dec_norm,
                                 self.en_lm_head, attention_mask)

    def decode_fr(self, concepts, seq_len, attention_mask=None):
        return self._decode_head(concepts, seq_len, self.fr_pos_queries,
                                 self.fr_dec_layers, self.fr_dec_norm,
                                 self.fr_lm_head, attention_mask)

    def decode_es(self, concepts, seq_len, attention_mask=None):
        return self._decode_head(concepts, seq_len, self.es_pos_queries,
                                 self.es_dec_layers, self.es_dec_norm,
                                 self.es_lm_head, attention_mask)

    def decode_para(self, concepts, seq_len, attention_mask=None):
        return self._decode_head(concepts, seq_len, self.para_pos_queries,
                                 self.para_dec_layers, self.para_dec_norm,
                                 self.para_lm_head, attention_mask)

    def decode_parse(self, concepts, seq_len, attention_mask=None):
        return self._decode_head(concepts, seq_len, self.parse_pos_queries,
                                 self.parse_dec_layers, self.parse_dec_norm,
                                 self.parse_lm_head, attention_mask)

    # Alias for probe_geometry.py compatibility
    def decode_parallel(self, concepts, seq_len, attention_mask=None):
        return self.decode_en(concepts, seq_len, attention_mask)

    def forward(self, input_ids, attention_mask=None, targets=None):
        """
        Forward pass. targets is a dict with optional keys:
            'fr': (ids, mask), 'es': (ids, mask), 'para': (ids, mask), 'parse': (ids, mask)
        Returns dict of logits + concepts.
        """
        concepts = self.encode(input_ids, attention_mask)

        result = {'concepts': concepts}
        result['en_logits'] = self.decode_en(concepts, input_ids.shape[1], attention_mask)

        if targets:
            for key, decode_fn in [('fr', self.decode_fr), ('es', self.decode_es),
                                    ('para', self.decode_para), ('parse', self.decode_parse)]:
                if key in targets:
                    ids, mask = targets[key]
                    result[f'{key}_logits'] = decode_fn(concepts, ids.shape[1], mask)

        return result

    def concept_vector(self, input_ids, attention_mask=None):
        concepts = self.encode(input_ids, attention_mask)
        flat = concepts.view(concepts.shape[0], -1)
        return F.normalize(flat, p=2, dim=-1)

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


# ═══════════════════════════════════════════════════════════════════════
# V17 — No Bottleneck: decoders cross-attend directly to encoder output
# ═══════════════════════════════════════════════════════════════════════

class NoBnDecoderBlock(nn.Module):
    """Decoder block where cross-attention reads full encoder output (enc_hidden dim)
    instead of a narrow bottleneck. Otherwise identical to ParallelDecoderBlock."""

    def __init__(self, config: ConceptConfig):
        super().__init__()
        self.self_attn_norm = RMSNorm(config.dec_hidden, eps=config.rms_norm_eps)
        self.self_attn = SelfAttention(config.dec_hidden, config.dec_heads, config.dropout)

        # Cross-attention: KV comes from encoder output (enc_hidden), not bottleneck
        self.cross_attn_norm = RMSNorm(config.dec_hidden, eps=config.rms_norm_eps)
        self.cross_attn = CrossAttention(
            q_dim=config.dec_hidden,
            kv_dim=config.enc_hidden,   # <-- full encoder dim, not concept_dim
            num_heads=config.dec_heads,
            dropout=config.dropout,
        )

        self.ffn_norm = RMSNorm(config.dec_hidden, eps=config.rms_norm_eps)
        self.ffn = SwiGLUFFN(config.dec_hidden, config.dec_intermediate, config.dropout)

    def forward(self, x, encoder_output, rope_cos, rope_sin, attention_mask=None):
        x = x + self.self_attn(self.self_attn_norm(x), rope_cos, rope_sin,
                                attention_mask=attention_mask)
        x = x + self.cross_attn(self.cross_attn_norm(x), encoder_output)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class ConceptAutoencoderV17(nn.Module):
    """
    V17: Bottleneck + 3 heads + geo gate + programmatic geometry data.

    Like V15 (V14 model) but with only 3 heads (EN, Para, Parse) — no FR/ES.
    Bottleneck restored because geometry losses need a compact target to sculpt.
    Geo gate + frequency ramp: every step for first 2K after gate, then ramp to 1/5.
    Programmatic geometry data with strict train/test vocab splits.
    """

    def __init__(self, config: ConceptConfig):
        super().__init__()
        self.config = config

        # --- Encoder ---
        self.embed_tokens = nn.Embedding(config.vocab_size, config.enc_hidden)

        head_dim = config.enc_hidden // config.enc_heads
        enc_rope_cos, enc_rope_sin = precompute_rope(head_dim, config.max_seq_len)
        self.register_buffer("enc_rope_cos", enc_rope_cos, persistent=False)
        self.register_buffer("enc_rope_sin", enc_rope_sin, persistent=False)

        self.enc_layers = nn.ModuleList([
            EncoderBlock(config) for _ in range(config.enc_layers)
        ])
        self.enc_norm = RMSNorm(config.enc_hidden, eps=config.rms_norm_eps)

        # --- Bottleneck (num_concepts × concept_dim) ---
        self.bottleneck = ConceptBottleneck(config)

        # --- Shared decoder RoPE ---
        dec_head_dim = config.dec_hidden // config.dec_heads
        dec_rope_cos, dec_rope_sin = precompute_rope(dec_head_dim, config.max_seq_len)
        self.register_buffer("dec_rope_cos", dec_rope_cos, persistent=False)
        self.register_buffer("dec_rope_sin", dec_rope_sin, persistent=False)

        # --- Head 1: EN Reconstruction ---
        self.en_pos_queries = nn.Parameter(
            torch.zeros(config.max_seq_len, config.dec_hidden))
        self.en_dec_layers = nn.ModuleList([
            ParallelDecoderBlock(config) for _ in range(config.dec_layers)
        ])
        self.en_dec_norm = RMSNorm(config.dec_hidden, eps=config.rms_norm_eps)
        self.en_lm_head = nn.Linear(config.dec_hidden, config.vocab_size, bias=False)

        # --- Head 2: EN Paraphrase ---
        self.para_pos_queries = nn.Parameter(
            torch.zeros(config.max_seq_len, config.dec_hidden))
        self.para_dec_layers = nn.ModuleList([
            ParallelDecoderBlock(config) for _ in range(config.dec_layers)
        ])
        self.para_dec_norm = RMSNorm(config.dec_hidden, eps=config.rms_norm_eps)
        self.para_lm_head = nn.Linear(config.dec_hidden, config.vocab_size, bias=False)

        # --- Head 3: Semantic Parse ---
        self.parse_pos_queries = nn.Parameter(
            torch.zeros(config.max_seq_len, config.dec_hidden))
        self.parse_dec_layers = nn.ModuleList([
            ParallelDecoderBlock(config) for _ in range(config.dec_layers)
        ])
        self.parse_dec_norm = RMSNorm(config.dec_hidden, eps=config.rms_norm_eps)
        self.parse_lm_head = nn.Linear(config.dec_hidden, config.vocab_size, bias=False)

        self.apply(self._init_weights)
        for pq in [self.en_pos_queries, self.para_pos_queries, self.parse_pos_queries]:
            self._init_position_queries(pq)

    def _init_position_queries(self, param):
        d = self.config.dec_hidden
        pos = torch.arange(self.config.max_seq_len).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2).float() * -(math.log(10000.0) / d))
        param.data[:, 0::2] = torch.sin(pos * div)
        param.data[:, 1::2] = torch.cos(pos * div)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)

    def encode(self, input_ids, attention_mask=None):
        """Encode text → concept stack (B, num_concepts, concept_dim)."""
        bsz, seq_len = input_ids.shape
        x = self.embed_tokens(input_ids)

        attn_mask = None
        if attention_mask is not None:
            pad_mask = torch.zeros(bsz, 1, 1, seq_len, device=x.device, dtype=x.dtype)
            pad_mask.masked_fill_(attention_mask[:, None, None, :] == 0, float("-inf"))
            attn_mask = pad_mask

        for layer in self.enc_layers:
            x = layer(x, self.enc_rope_cos, self.enc_rope_sin, attn_mask)

        x = self.enc_norm(x)
        concepts = self.bottleneck(x)
        return concepts

    def _decode_head(self, concepts, seq_len, pos_queries, dec_layers, dec_norm,
                     lm_head, attention_mask=None):
        """Decode from concept stack."""
        bsz = concepts.shape[0]
        x = pos_queries[:seq_len].unsqueeze(0).expand(bsz, -1, -1)

        dec_attn_mask = None
        if attention_mask is not None:
            dec_attn_mask = torch.zeros(bsz, 1, 1, seq_len, device=x.device, dtype=x.dtype)
            dec_attn_mask.masked_fill_(attention_mask[:, None, None, :] == 0, float("-inf"))

        for layer in dec_layers:
            x = layer(x, concepts, self.dec_rope_cos, self.dec_rope_sin,
                      attention_mask=dec_attn_mask)

        x = dec_norm(x)
        return lm_head(x)

    def decode_en(self, concepts, seq_len, attention_mask=None):
        return self._decode_head(concepts, seq_len, self.en_pos_queries,
                                 self.en_dec_layers, self.en_dec_norm,
                                 self.en_lm_head, attention_mask)

    def decode_para(self, concepts, seq_len, attention_mask=None):
        return self._decode_head(concepts, seq_len, self.para_pos_queries,
                                 self.para_dec_layers, self.para_dec_norm,
                                 self.para_lm_head, attention_mask)

    def decode_parse(self, concepts, seq_len, attention_mask=None):
        return self._decode_head(concepts, seq_len, self.parse_pos_queries,
                                 self.parse_dec_layers, self.parse_dec_norm,
                                 self.parse_lm_head, attention_mask)

    # Alias for probe_geometry.py compatibility
    def decode_parallel(self, concepts, seq_len, attention_mask=None):
        return self.decode_en(concepts, seq_len, attention_mask)

    def forward(self, input_ids, attention_mask=None, targets=None):
        concepts = self.encode(input_ids, attention_mask)

        result = {'concepts': concepts}
        result['en_logits'] = self.decode_en(concepts, input_ids.shape[1], attention_mask)

        if targets:
            for key, decode_fn in [('para', self.decode_para),
                                    ('parse', self.decode_parse)]:
                if key in targets:
                    ids, mask = targets[key]
                    result[f'{key}_logits'] = decode_fn(concepts, ids.shape[1], mask)

        return result

    def concept_vector(self, input_ids, attention_mask=None):
        """Flat bottleneck, L2-normalized for similarity."""
        concepts = self.encode(input_ids, attention_mask)
        flat = concepts.view(concepts.shape[0], -1)
        return F.normalize(flat, p=2, dim=-1)

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable
