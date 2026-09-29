"""Small building blocks shared by the sequence mixers."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ShortConv(nn.Module):
    """Causal depthwise 1-D convolution over time (the "short conv" used by Mamba / DeltaNet).

    Input and output are ``[batch, time, channels]``. For token-by-token decoding the layer keeps
    the last ``kernel_size - 1`` inputs as its state.
    """

    def __init__(self, channels: int, kernel_size: int = 4):
        super().__init__()
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.empty(channels, kernel_size))
        self.bias = nn.Parameter(torch.zeros(channels))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)  # [B, C, T]
        x = F.pad(x, (self.kernel_size - 1, 0))
        y = F.conv1d(x, self.weight.unsqueeze(1), self.bias, groups=x.shape[1])
        return y.transpose(1, 2)

    def init_state(self, batch: int, device, dtype) -> torch.Tensor:
        return torch.zeros(batch, self.weight.shape[0], self.kernel_size - 1, device=device, dtype=dtype)

    def step(self, x_t: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x_t: ``[B, C]``; state: ``[B, C, K-1]`` (oldest input first)."""
        window = torch.cat([state, x_t.unsqueeze(-1)], dim=-1)  # [B, C, K]
        y = (window * self.weight).sum(-1) + self.bias
        return y, window[..., 1:]


def l2_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def pad_time(x: torch.Tensor, multiple: int, dim: int, value: float = 0.0) -> torch.Tensor:
    """Right-pad ``x`` along ``dim`` so its length is a multiple of ``multiple``."""
    length = x.shape[dim]
    extra = (-length) % multiple
    if extra == 0:
        return x
    pad_shape = list(x.shape)
    pad_shape[dim] = extra
    return torch.cat([x, x.new_full(pad_shape, value)], dim=dim)


def scan_dtype(dtype: torch.dtype) -> torch.dtype:
    """Recurrences are accumulated in at least float32, even when the model runs in bf16."""
    return torch.promote_types(dtype, torch.float32)
