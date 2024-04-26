"""Small GPT-style decoder-only transformer with residual-stream access."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import T, VOCAB


@dataclass
class ModelConfig:
    d_model: int = 64
    n_layer: int = 2
    n_head: int = 4


class Block(nn.Module):
    def __init__(self, c: ModelConfig):
        super().__init__()
        self.n_head = c.n_head
        self.ln1, self.ln2 = nn.LayerNorm(c.d_model), nn.LayerNorm(c.d_model)
        self.qkv = nn.Linear(c.d_model, 3 * c.d_model)
        self.proj = nn.Linear(c.d_model, c.d_model)
        self.mlp = nn.Sequential(nn.Linear(c.d_model, 4 * c.d_model), nn.GELU(),
                                 nn.Linear(4 * c.d_model, c.d_model))

    def forward(self, x):
        B, L, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, L, 3, self.n_head, D // self.n_head).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(B, L, D))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, c: ModelConfig):
        super().__init__()
        self.cfg = c
        self.tok = nn.Embedding(VOCAB, c.d_model)
        self.pos = nn.Embedding(T, c.d_model)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layer))
        self.ln_f = nn.LayerNorm(c.d_model)
        self.head = nn.Linear(c.d_model, VOCAB, bias=False)

    def forward(self, idx, return_hidden: bool = False):
        """Logits (B, T, V). With ``return_hidden`` also the residual stream at every
        depth: a list of n_layer+1 tensors (B, T, D) (embeddings, then after each block)."""
        h = self.tok(idx) + self.pos(torch.arange(idx.shape[1]))
        hidden = [h]
        for b in self.blocks:
            h = b(h)
            hidden.append(h)
        logits = self.head(self.ln_f(h))
        return (logits, hidden) if return_hidden else logits
