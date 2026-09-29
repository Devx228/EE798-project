"""Selective state-space mixer in the style of Mamba-2 (scalar decay per head, "SSD").

Per head the layer runs the linear recurrence

    h_t = a_t * h_{t-1} + (dt_t * x_t) B_t^T        h_t in R^{P x N}
    y_t = h_t C_t + D * x_t

where ``a_t = exp(-dt_t * exp(A_log))`` is an input-dependent decay in (0, 1). With
``neg_eigen=True`` the decay is remapped to ``2 a_t - 1`` in (-1, 1), following Grazzi et al.
(ICLR 2025), which lets the state flip sign and therefore track parity.

Two mathematically identical scan implementations are provided:

* ``ssd_scan_recurrent`` -- a readable token-by-token loop (reference / decoding).
* ``ssd_scan_chunked``   -- the chunkwise-parallel form used for training: quadratic
  "attention-like" work inside each chunk plus a short recurrence across chunks.

A forward-only Triton kernel lives in ``triton_scan.py``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import ShortConv, pad_time, scan_dtype


def ssd_scan_recurrent(x, a, B, C, initial_state=None):
    """Reference scan.

    Args:
        x: ``[b, T, H, P]`` inputs (already scaled by dt).
        a: ``[b, T, H]`` per-step decay (may be negative).
        B, C: ``[b, T, N]`` input / output projections shared by all heads.
        initial_state: optional ``[b, H, P, N]``.

    Returns:
        y ``[b, T, H, P]`` and the final state ``[b, H, P, N]``.
    """
    b, T, H, P = x.shape
    N = B.shape[-1]
    h = x.new_zeros(b, H, P, N) if initial_state is None else initial_state
    ys = []
    for t in range(T):
        h = a[:, t, :, None, None] * h + x[:, t, :, :, None] * B[:, t, None, None, :]
        ys.append(torch.einsum("bhpn,bn->bhp", h, C[:, t]))
    return torch.stack(ys, dim=1), h


def _signed_log(a, eps=1e-12):
    """Split a (possibly negative) decay into log|a| and sign so products can be taken in log space."""
    return torch.log(a.abs().clamp_min(eps)), torch.where(a < 0, -1.0, 1.0).to(a.dtype)


def ssd_scan_chunked(x, a, B, C, chunk_size: int = 64, initial_state=None):
    """Chunkwise-parallel scan, numerically equivalent to :func:`ssd_scan_recurrent`.

    Inside a chunk the output is ``y_i = sum_{j<=i} (C_i . B_j) * prod_{k=j+1..i} a_k * x_j``,
    i.e. causal attention with a decay mask. Chunk states are then chained with one step of the
    recurrence per chunk, so the cost is O(T * chunk) instead of O(T^2).
    """
    b, T, H, P = x.shape
    L = chunk_size
    # Padding with a=1, x=0, B=C=0 leaves the state and outputs untouched.
    x, B, C = (pad_time(t, L, dim=1) for t in (x, B, C))
    a = pad_time(a, L, dim=1, value=1.0)
    nc = x.shape[1] // L

    x = x.view(b, nc, L, H, P)
    B = B.view(b, nc, L, -1)
    C = C.view(b, nc, L, -1)
    log_a, sign_a = _signed_log(a.view(b, nc, L, H))

    cum_log = log_a.cumsum(dim=2)  # [b, nc, L, H]: log|prod_{k<=i} a_k| within chunk
    cum_sign = sign_a.cumprod(dim=2)

    # decay[i, j] = prod_{k=j+1..i} a_k  for j <= i   -> [b, nc, H, L, L]
    diff = cum_log.unsqueeze(3) - cum_log.unsqueeze(2)  # [b, nc, L(i), L(j), H]
    causal = torch.tril(torch.ones(L, L, dtype=torch.bool, device=x.device))
    diff = diff.masked_fill(~causal[None, None, :, :, None], float("-inf"))
    sign_ij = cum_sign.unsqueeze(3) * cum_sign.unsqueeze(2)
    decay = (torch.exp(diff) * sign_ij).permute(0, 1, 4, 2, 3)

    # 1) intra-chunk ("diagonal block") outputs
    scores = torch.einsum("bcin,bcjn->bcij", C, B)  # [b, nc, L, L]
    y_diag = torch.einsum("bcij,bchij,bcjhp->bcihp", scores, decay, x)

    # 2) state contributed by each chunk, measured at the chunk's end
    decay_to_end = torch.exp(cum_log[:, :, -1:] - cum_log) * cum_sign[:, :, -1:] * cum_sign  # [b, nc, L, H]
    chunk_states = torch.einsum("bclh,bclhp,bcln->bchpn", decay_to_end, x, B)

    # 3) sequential pass over chunks (nc is small)
    chunk_decay = torch.exp(cum_log[:, :, -1]) * cum_sign[:, :, -1]  # [b, nc, H]
    h = x.new_zeros(b, H, P, B.shape[-1]) if initial_state is None else initial_state
    starts = []
    for c in range(nc):
        starts.append(h)
        h = chunk_decay[:, c, :, None, None] * h + chunk_states[:, c]
    start_states = torch.stack(starts, dim=1)  # [b, nc, H, P, N]

    # 4) contribution of the incoming state to every position in the chunk
    decay_from_start = torch.exp(cum_log) * cum_sign  # [b, nc, L, H]
    y_off = torch.einsum("bcin,bchpn,bcih->bcihp", C, start_states, decay_from_start)

    y = (y_diag + y_off).reshape(b, nc * L, H, P)[:, :T]
    return y, h


class SelectiveSSM(nn.Module):
    """Mamba-2 style mixer: in_proj -> short conv -> selective scan -> gated RMSNorm -> out_proj."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        expand: int = 2,
        head_dim: int = 32,
        d_conv: int = 4,
        neg_eigen: bool = False,
        chunk_size: int = 64,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
        backend: str = "chunked",
    ):
        super().__init__()
        d_inner = expand * d_model
        assert d_inner % head_dim == 0, "expand * d_model must be divisible by head_dim"
        self.d_inner, self.d_state, self.head_dim = d_inner, d_state, head_dim
        self.n_heads = d_inner // head_dim
        self.neg_eigen, self.chunk_size, self.backend = neg_eigen, chunk_size, backend

        # z (gate) | x | B | C | dt
        self.in_proj = nn.Linear(d_model, 2 * d_inner + 2 * d_state + self.n_heads, bias=False)
        self.conv = ShortConv(d_inner + 2 * d_state, d_conv)

        # A in [1, 16] as in Mamba-2; dt initialised log-uniformly in [dt_min, dt_max].
        self.A_log = nn.Parameter(torch.log(torch.empty(self.n_heads).uniform_(1, 16)))
        dt = torch.exp(torch.empty(self.n_heads).uniform_(math.log(dt_min), math.log(dt_max)))
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))  # inverse softplus
        self.D = nn.Parameter(torch.ones(self.n_heads))
        self.norm = nn.RMSNorm(d_inner)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def _split(self, zxbcdt):
        return torch.split(zxbcdt, [self.d_inner, self.d_inner + 2 * self.d_state, self.n_heads], dim=-1)

    def _discretize(self, dt_raw):
        dt = F.softplus(dt_raw + self.dt_bias)  # [..., H]
        a = torch.exp(-dt * torch.exp(self.A_log))
        if self.neg_eigen:
            a = 2.0 * a - 1.0
        return dt, a

    def forward(self, u: torch.Tensor, return_state: bool = False):
        b, T, _ = u.shape
        z, xBC, dt_raw = self._split(self.in_proj(u))
        xBC = F.silu(self.conv(xBC))
        x, B, C = torch.split(xBC, [self.d_inner, self.d_state, self.d_state], dim=-1)
        sd = scan_dtype(u.dtype)
        dt, a = self._discretize(dt_raw.to(sd))
        x = x.view(b, T, self.n_heads, self.head_dim)

        xs, Bf, Cf = (x * dt.unsqueeze(-1)).to(sd), B.to(sd), C.to(sd)
        if self.backend == "recurrent":
            y, state = ssd_scan_recurrent(xs, a, Bf, Cf)
        elif self.backend == "triton" and not (torch.is_grad_enabled() and u.requires_grad):
            from .triton_scan import ssd_scan_triton

            y, state = ssd_scan_triton(xs, a, Bf, Cf)
        else:  # chunked (also the training fallback for the forward-only Triton kernel)
            y, state = ssd_scan_chunked(xs, a, Bf, Cf, self.chunk_size)

        y = y.to(u.dtype) + x * self.D[:, None]
        y = self.norm(y.reshape(b, T, self.d_inner) * F.silu(z))
        out = self.out_proj(y)
        return (out, state) if return_state else out

    # ---- token-by-token decoding -------------------------------------------------------------
    def init_cache(self, batch: int, device, dtype):
        return {
            "conv": self.conv.init_state(batch, device, dtype),
            "ssm": torch.zeros(batch, self.n_heads, self.head_dim, self.d_state, device=device, dtype=scan_dtype(dtype)),
        }

    def step(self, u_t: torch.Tensor, cache: dict) -> torch.Tensor:
        """u_t: ``[b, d_model]``. Constant memory and compute per token, independent of context length."""
        z, xBC, dt_raw = self._split(self.in_proj(u_t))
        xBC, cache["conv"] = self.conv.step(xBC, cache["conv"])
        xBC = F.silu(xBC)
        x, B, C = torch.split(xBC, [self.d_inner, self.d_state, self.d_state], dim=-1)
        h = cache["ssm"]
        dt, a = self._discretize(dt_raw.to(h.dtype))
        x = x.view(-1, self.n_heads, self.head_dim)
        h = a[:, :, None, None] * h + (x * dt[..., None]).to(h.dtype)[..., None] * B.to(h.dtype)[:, None, None, :]
        cache["ssm"] = h
        y = torch.einsum("bhpn,bn->bhp", h, C.to(h.dtype)).to(u_t.dtype) + x * self.D[:, None]
        y = self.norm(y.reshape(-1, self.d_inner) * F.silu(z))
        return self.out_proj(y)

    def state_bytes(self, batch: int = 1) -> int:
        return batch * self.n_heads * self.head_dim * self.d_state * 4
