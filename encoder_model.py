"""
FLM V4 — Semantic Encoder (Model 1 of Concept Vector Architecture)
===================================================================
Bidirectional transformer encoder that maps text to concept vectors.
Paraphrases and translations of the same meaning produce identical vectors.

~30M parameters. Encoder-only, no causal mask. Contrastive training only.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class EncoderConfig:
    vocab_size: int = 32000
    hidden_size: int = 384
    num_layers: int = 8
    num_heads: int = 6
    intermediate_size: int = 1536
    max_seq_len: int = 128
    dropout: float = 0.1
    output_dim: int = 512
    rms_norm_eps: float = 1e-5

    def param_estimate(self):
        embed = self.vocab_size * self.hidden_size
        head_dim = self.hidden_size // self.num_heads
        attn = 4 * self.hidden_size * self.hidden_size  # Q, K, V, O
        ffn = 3 * self.hidden_size * self.intermediate_size  # gate, up, down (SwiGLU)
        norms = 2 * self.hidden_size  # 2 RMSNorm per layer
        per_layer = attn + ffn + norms
        total = embed + per_layer * self.num_layers + self.hidden_size  # final norm
        if self.output_dim != self.hidden_size:
            total += self.hidden_size * self.output_dim  # projection
        return total


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


class EncoderAttention(nn.Module):
    """Bidirectional multi-head attention with RoPE."""

    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.hidden_size = config.hidden_size

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)

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


class EncoderFFN(nn.Module):
    """SwiGLU feed-forward network."""

    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class EncoderBlock(nn.Module):
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention = EncoderAttention(config)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn = EncoderFFN(config)

    def forward(self, x, rope_cos, rope_sin, attention_mask=None):
        x = x + self.attention(self.attn_norm(x), rope_cos, rope_sin, attention_mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class SemanticEncoder(nn.Module):
    """
    Bidirectional transformer encoder → L2-normalized concept vector.

    Input: token IDs + attention mask
    Output: unit-norm vector in R^output_dim

    Training: contrastive loss (NT-Xent). Paraphrase pairs → same vector.
    """

    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        head_dim = config.hidden_size // config.num_heads
        rope_cos, rope_sin = precompute_rope(head_dim, config.max_seq_len)
        self.register_buffer("rope_cos", rope_cos, persistent=False)
        self.register_buffer("rope_sin", rope_sin, persistent=False)

        self.layers = nn.ModuleList([EncoderBlock(config) for _ in range(config.num_layers)])
        self.final_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Project to output dim if different from hidden
        if config.output_dim != config.hidden_size:
            self.proj = nn.Linear(config.hidden_size, config.output_dim, bias=False)
        else:
            self.proj = None

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, input_ids, attention_mask=None):
        """
        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len) — 1 for real tokens, 0 for padding

        Returns:
            vectors: (batch, output_dim) — L2-normalized concept vectors
        """
        bsz, seq_len = input_ids.shape
        x = self.embed_tokens(input_ids)

        # Build pad mask for bidirectional attention (no causal mask)
        attn_mask = None
        if attention_mask is not None:
            # (batch, 1, 1, seq_len) — broadcast over heads and query positions
            pad_mask = torch.zeros(bsz, 1, 1, seq_len, device=x.device, dtype=x.dtype)
            pad_mask.masked_fill_(attention_mask[:, None, None, :] == 0, float("-inf"))
            attn_mask = pad_mask

        for layer in self.layers:
            x = layer(x, self.rope_cos, self.rope_sin, attn_mask)

        x = self.final_norm(x)

        # Mean pooling over non-padding tokens
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()  # (batch, seq, 1)
            x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)

        # Optional projection
        if self.proj is not None:
            x = self.proj(x)

        # L2 normalize to unit sphere
        x = F.normalize(x, p=2, dim=-1)

        return x

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


def nt_xent_loss(z_a, z_b, temperature=0.07):
    """
    NT-Xent (SimCLR) contrastive loss.

    z_a, z_b: (batch, dim) — L2-normalized vectors for positive pairs.
    All other pairs in the batch are negatives.
    """
    batch_size = z_a.shape[0]
    z = torch.cat([z_a, z_b], dim=0)  # (2*batch, dim)

    # Cosine similarity matrix
    sim = torch.mm(z, z.t()) / temperature  # (2*batch, 2*batch)

    # Mask out self-similarity
    mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
    sim.masked_fill_(mask, float("-inf"))

    # Positive pairs: (i, i+batch) and (i+batch, i)
    labels = torch.cat([
        torch.arange(batch_size, 2 * batch_size, device=z.device),
        torch.arange(0, batch_size, device=z.device),
    ])

    loss = F.cross_entropy(sim, labels)
    return loss
