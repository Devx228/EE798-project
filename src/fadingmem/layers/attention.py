"""Causal multi-head softmax attention with rotary position embeddings (the Transformer baseline)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def rope_cache(max_len: int, dim: int, base: float = 10000.0, device=None):
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(max_len, device=device).float()
    freqs = torch.outer(t, inv_freq)  # [T, dim/2]
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):
    """x: ``[b, H, T, d]``; cos/sin: ``[T, d/2]``."""
    x1, x2 = x[..., ::2], x[..., 1::2]
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return out.flatten(-2)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4, max_len: int = 8192, rope_base: float = 10000.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads, self.head_dim = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.max_len, self.rope_base = max_len, rope_base
        cos, sin = rope_cache(max_len, self.head_dim, rope_base)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def _rope(self, T: int, offset: int = 0):
        if offset + T > self.cos.shape[0]:  # grow the table for long benchmarks / length generalisation
            cos, sin = rope_cache(2 * (offset + T), self.head_dim, self.rope_base, self.cos.device)
            self.cos, self.sin = cos, sin
        return self.cos[offset : offset + T], self.sin[offset : offset + T]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, T, D = x.shape
        q, k, v = self.qkv(x).view(b, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        cos, sin = self._rope(T)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out_proj(y.transpose(1, 2).reshape(b, T, D))

    def attention_map(self, x: torch.Tensor) -> torch.Tensor:
        """Explicit attention probabilities ``[b, H, T, T]`` for visualisation."""
        b, T, _ = x.shape
        q, k, _ = self.qkv(x).view(b, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        cos, sin = self._rope(T)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        scores = (q @ k.transpose(-1, -2)) * self.head_dim**-0.5
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), 1)
        return scores.masked_fill(mask, float("-inf")).softmax(-1)

    # ---- decoding with a KV cache: memory grows linearly with context -----------------------------
    def init_cache(self, batch: int, device, dtype, max_len: int | None = None):
        max_len = max_len or self.max_len
        shape = (batch, self.n_heads, max_len, self.head_dim)
        return {"k": torch.zeros(shape, device=device, dtype=dtype), "v": torch.zeros(shape, device=device, dtype=dtype), "pos": 0}

    def step(self, x_t: torch.Tensor, cache: dict) -> torch.Tensor:
        b, D = x_t.shape
        pos = cache["pos"]
        q, k, v = self.qkv(x_t).view(b, 3, self.n_heads, 1, self.head_dim).unbind(1)
        cos, sin = self._rope(1, pos)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        cache["k"][:, :, pos : pos + 1] = k
        cache["v"][:, :, pos : pos + 1] = v
        cache["pos"] = pos + 1
        y = F.scaled_dot_product_attention(q, cache["k"][:, :, : pos + 1], cache["v"][:, :, : pos + 1])
        return self.out_proj(y.reshape(b, D))

    def state_bytes(self, batch: int = 1, context: int = 1, bytes_per_el: int = 2) -> int:
        return 2 * batch * self.n_heads * context * self.head_dim * bytes_per_el
