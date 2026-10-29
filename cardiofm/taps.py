"""Representation taps that turn encoder outputs into the shared embedding space."""
from __future__ import annotations

import torch
import torch.nn as nn


class AttentionPool3d(nn.Module):
    """Attention pooling over image tokens -> a single global embedding.

    A learned query attends over the ``[B, N, C]`` token grid (CLIP-style) and the
    result is projected to ``embed_dim`` (the shared image-text dim).
    """

    def __init__(self, in_dim: int, embed_dim: int, num_heads: int = 8) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, in_dim) * (in_dim ** -0.5))
        self.attn = nn.MultiheadAttention(in_dim, num_heads, batch_first=True)
        self.proj = nn.Linear(in_dim, embed_dim)
        self.norm = nn.LayerNorm(in_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        b = tokens.shape[0]
        q = self.query.expand(b, -1, -1)
        pooled, _ = self.attn(q, tokens, tokens)  # [B, 1, C]
        pooled = self.norm(pooled.squeeze(1))
        return self.proj(pooled)  # [B, embed_dim]


class TokenProjection(nn.Module):
    """Project bottleneck image tokens to the text-decoder memory dim ``d_txt``."""

    def __init__(self, in_dim: int, d_txt: int) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, d_txt)
        self.norm = nn.LayerNorm(d_txt)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(tokens))  # [B, N, d_txt]
