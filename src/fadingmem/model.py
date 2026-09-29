"""A small language-model-style network whose token mixer is chosen per layer.

The ``pattern`` string picks the mixer of each layer:

    "A" -> causal softmax attention (Transformer)
    "M" -> selective SSM (Mamba-2 style)
    "D" -> Gated DeltaNet

e.g. ``"MM"`` is a pure 2-layer SSM, ``"MAMM"`` a hybrid with one attention layer out of four.
A pattern shorter than ``n_layers`` is repeated.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers.attention import CausalSelfAttention
from .layers.deltanet import GatedDeltaNet
from .layers.ssm import SelectiveSSM


@dataclass
class ModelConfig:
    vocab_size: int = 4096
    d_model: int = 128
    n_layers: int = 2
    pattern: str = "M"
    mlp_ratio: float = 0.0  # 0 disables the MLP (common for synthetic recall probes)
    neg_eigen: bool = False  # extend SSM / DeltaNet transition eigenvalues to (-1, 1)
    attn: dict = field(default_factory=lambda: {"n_heads": 4})
    ssm: dict = field(default_factory=lambda: {"d_state": 16, "expand": 2, "head_dim": 32})
    deltanet: dict = field(default_factory=lambda: {"n_heads": 4, "gated": True})

    def layer_types(self) -> str:
        return (self.pattern * self.n_layers)[: self.n_layers]

    def to_dict(self):
        return asdict(self)


def build_mixer(kind: str, cfg: ModelConfig) -> nn.Module:
    if kind == "A":
        return CausalSelfAttention(cfg.d_model, **cfg.attn)
    if kind == "M":
        return SelectiveSSM(cfg.d_model, neg_eigen=cfg.neg_eigen, **cfg.ssm)
    if kind == "D":
        return GatedDeltaNet(cfg.d_model, neg_eigen=cfg.neg_eigen, **cfg.deltanet)
    raise ValueError(f"unknown mixer type {kind!r}")


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, ratio: float):
        super().__init__()
        hidden = int(d_model * ratio)
        self.w_in = nn.Linear(d_model, 2 * hidden, bias=False)
        self.w_out = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x):
        a, b = self.w_in(x).chunk(2, dim=-1)
        return self.w_out(F.silu(a) * b)


class Block(nn.Module):
    def __init__(self, kind: str, cfg: ModelConfig):
        super().__init__()
        self.kind = kind
        self.norm1 = nn.RMSNorm(cfg.d_model)
        self.mixer = build_mixer(kind, cfg)
        self.mlp = SwiGLU(cfg.d_model, cfg.mlp_ratio) if cfg.mlp_ratio > 0 else None
        self.norm2 = nn.RMSNorm(cfg.d_model) if self.mlp is not None else None

    def forward(self, x):
        x = x + self.mixer(self.norm1(x))
        if self.mlp is not None:
            x = x + self.mlp(self.norm2(x))
        return x

    def step(self, x_t, cache):
        x_t = x_t + self.mixer.step(self.norm1(x_t), cache)
        if self.mlp is not None:
            x_t = x_t + self.mlp(self.norm2(x_t))
        return x_t


class SequenceModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(kind, cfg) for kind in cfg.layer_types())
        self.norm_f = nn.RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight  # weight tying
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None and m.bias.abs().sum() == 0:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        x = self.embed(idx)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.norm_f(x))
        if targets is None:
            return logits
        loss = F.cross_entropy(logits.flatten(0, 1).float(), targets.flatten(), ignore_index=-100)
        return logits, loss

    # ---- autoregressive decoding -------------------------------------------------------------
    def init_cache(self, batch: int, device, dtype=torch.float32, max_len: int = 2048):
        caches = []
        for block in self.blocks:
            if block.kind == "A":
                caches.append(block.mixer.init_cache(batch, device, dtype, max_len))
            else:
                caches.append(block.mixer.init_cache(batch, device, dtype))
        return caches

    def step(self, idx_t: torch.Tensor, caches) -> torch.Tensor:
        """idx_t: ``[b]`` -> logits ``[b, vocab]``."""
        x = self.embed(idx_t)
        for block, cache in zip(self.blocks, caches):
            x = block.step(x, cache)
        return self.lm_head(self.norm_f(x))

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def state_size(self) -> int | None:
        """Floats of recurrent memory per sequence (summed over layers). ``None`` if any layer is
        attention, whose KV cache grows with the context instead of being fixed."""
        if any(b.kind == "A" for b in self.blocks):
            return None
        return sum(b.mixer.state_bytes(1) // 4 for b in self.blocks)

    def mixer_params(self) -> int:
        return sum(p.numel() for b in self.blocks for p in b.mixer.parameters())
