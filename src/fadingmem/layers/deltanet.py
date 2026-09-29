"""Gated DeltaNet mixer (Yang, Kautz & Hatamizadeh, ICLR 2025).

The state is a fast-weight matrix ``S_t in R^{d_k x d_v}`` per head, updated with the gated
delta rule

    S_t = alpha_t (I - beta_t k_t k_t^T) S_{t-1} + beta_t k_t v_t^T,      o_t = S_t^T q_t

Reading: first decay the memory by ``alpha_t``, then *erase* what is currently stored under key
``k_t`` and *write* ``v_t`` in its place (an online least-squares / delta-rule step). Unlike the
diagonal SSM, the transition ``I - beta k k^T`` is a non-diagonal (Householder-like) matrix. Its
eigenvalue along ``k_t`` is ``1 - beta_t``; with ``neg_eigen=True`` we let ``beta_t`` range over
(0, 2) so that eigenvalue can become negative (Grazzi et al., ICLR 2025).

``gated=False`` recovers plain DeltaNet (alpha_t = 1).

DeltaProduct (Siems et al. 2025, ``n_householder = n_h > 1``): every token applies ``n_h``
delta-rule steps with their own keys, values and betas,

    S_t = alpha_t * prod_{j=1..n_h} (I - beta_{t,j} k_{t,j} k_{t,j}^T) S_{t-1} + (write terms),

so the transition is a product of ``n_h`` generalised Householder matrices. With beta in (0, 2)
one factor can be a reflection; two reflections compose to a rotation, which is what counting
mod m and non-abelian groups like S_3 need. We implement it by *interleaving* the sub-steps into
the time axis (length T * n_h) and reusing the verified chunked kernel: the decay is applied at
the first sub-step of each token and the output is read after the last.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import ShortConv, l2_normalize, pad_time, scan_dtype


def delta_rule_recurrent(q, k, v, beta, g, initial_state=None):
    """Reference gated delta rule.

    Args:
        q, k: ``[b, H, T, dk]`` (k assumed L2-normalised), v: ``[b, H, T, dv]``.
        beta: ``[b, H, T]`` write strength; g: ``[b, H, T]`` log-decay (alpha = exp(g)).

    Returns:
        o ``[b, H, T, dv]`` and final state ``[b, H, dk, dv]``.
    """
    b, H, T, dk = q.shape
    S = q.new_zeros(b, H, dk, v.shape[-1]) if initial_state is None else initial_state
    outs = []
    for t in range(T):
        S = S * torch.exp(g[:, :, t])[..., None, None]
        k_t, v_t, beta_t = k[:, :, t], v[:, :, t], beta[:, :, t, None]
        v_old = torch.einsum("bhkv,bhk->bhv", S, k_t)  # what the memory currently returns for k_t
        S = S + torch.einsum("bhk,bhv->bhkv", k_t, beta_t * (v_t - v_old))
        outs.append(torch.einsum("bhkv,bhk->bhv", S, q[:, :, t]))
    return torch.stack(outs, dim=2), S


def delta_rule_chunked(q, k, v, beta, g, chunk_size: int = 64, initial_state=None):
    """Chunkwise-parallel gated delta rule (WY / UT-transform form), equivalent to the recurrence.

    Within a chunk with cumulative log-decay gamma_i, the vectors actually written to memory,
    ``u_i = beta_i (v_i - alpha_i S_{i-1}^T k_i)``, satisfy a unit lower-triangular system

        (I + M) U = diag(beta) V - diag(beta * Gamma) K S_0,
        M_ij = beta_i exp(gamma_i - gamma_j) (k_i . k_j)   for j < i,

    which we solve with one triangular solve per chunk. Outputs and the next chunk state then
    follow from masked matrix products, exactly like causal linear attention.
    """
    b, H, T, dk = q.shape
    dv = v.shape[-1]
    L = chunk_size
    # Padding with k=v=q=0, beta=0, g=0 means "no write, no decay": the state is unchanged.
    q, k, v, beta, g = (pad_time(t, L, dim=2) for t in (q, k, v, beta, g))
    nc = q.shape[2] // L
    q, k, v = (t.view(b, H, nc, L, -1) for t in (q, k, v))
    beta, g = beta.view(b, H, nc, L), g.view(b, H, nc, L)

    gamma = g.cumsum(dim=-1)  # [b, H, nc, L]
    diff = gamma.unsqueeze(-1) - gamma.unsqueeze(-2)  # [.., L(i), L(j)]
    incl = torch.tril(torch.ones(L, L, dtype=torch.bool, device=q.device))
    decay = torch.exp(diff.masked_fill(~incl, float("-inf")))  # exp(gamma_i - gamma_j), j <= i

    kk = k @ k.transpose(-1, -2)
    M = (beta.unsqueeze(-1) * kk * decay).tril(-1)
    eye = torch.eye(L, device=q.device, dtype=q.dtype)
    rhs = torch.cat([beta.unsqueeze(-1) * v, (beta * torch.exp(gamma)).unsqueeze(-1) * k], dim=-1)
    sol = torch.linalg.solve_triangular(eye + M, rhs, upper=False, unitriangular=True)
    u_base, w = sol[..., :dv], sol[..., dv:]  # U = u_base - w @ S_0

    qk = (q @ k.transpose(-1, -2)) * decay  # causal, includes the diagonal
    S = q.new_zeros(b, H, dk, dv) if initial_state is None else initial_state
    outs = []
    for c in range(nc):
        u = u_base[:, :, c] - w[:, :, c] @ S
        o = (q[:, :, c] * torch.exp(gamma[:, :, c]).unsqueeze(-1)) @ S + qk[:, :, c] @ u
        outs.append(o)
        g_end = gamma[:, :, c, -1:]  # [b, H, 1]
        k_scaled = k[:, :, c] * torch.exp(g_end - gamma[:, :, c]).unsqueeze(-1)
        S = S * torch.exp(g_end).unsqueeze(-1) + k_scaled.transpose(-1, -2) @ u
    o = torch.stack(outs, dim=2).reshape(b, H, nc * L, dv)[:, :, :T]
    return o, S


class GatedDeltaNet(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        head_dim: int | None = None,
        d_conv: int = 4,
        gated: bool = True,
        neg_eigen: bool = False,
        n_householder: int = 1,
        chunk_size: int = 64,
        backend: str = "chunked",
    ):
        super().__init__()
        self.n_heads, self.n_h = n_heads, n_householder
        self.head_dim = head_dim or d_model // n_heads
        inner = self.n_heads * self.head_dim
        self.gated, self.neg_eigen = gated, neg_eigen
        self.chunk_size, self.backend = chunk_size, backend

        # q once per token; one (k, v) pair per Householder sub-step
        self.qkv_proj = nn.Linear(d_model, (1 + 2 * n_householder) * inner, bias=False)
        self.conv = ShortConv((1 + 2 * n_householder) * inner, d_conv)
        self.beta_proj = nn.Linear(d_model, n_heads * n_householder)
        self.a_proj = nn.Linear(d_model, n_heads)  # data-dependent decay (Mamba-2 style)
        self.A_log = nn.Parameter(torch.log(torch.empty(n_heads).uniform_(1, 16)))
        self.dt_bias = nn.Parameter(torch.full((n_heads,), -4.0))
        self.out_gate = nn.Linear(d_model, inner, bias=False)
        self.o_norm = nn.RMSNorm(self.head_dim)
        self.out_proj = nn.Linear(inner, d_model, bias=False)

    def _gates(self, u):
        """beta: ``[..., n_h, H]`` write strengths; g: ``[..., H]`` log-decay per token."""
        sd = scan_dtype(u.dtype)
        beta = torch.sigmoid(self.beta_proj(u).to(sd)).unflatten(-1, (self.n_h, self.n_heads))
        if self.neg_eigen:
            beta = 2.0 * beta
        if self.gated:
            g = -torch.exp(self.A_log) * F.softplus(self.a_proj(u).to(sd) + self.dt_bias)
        else:
            g = beta.new_zeros(beta.shape[:-2] + (self.n_heads,))
        return beta, g

    def _qkv(self, qkv):
        """q: ``[..., H, d]``; k, v: ``[..., n_h, H, d]``."""
        q, kv = F.silu(qkv).split([self.n_heads * self.head_dim, 2 * self.n_h * self.n_heads * self.head_dim], dim=-1)
        k, v = kv.unflatten(-1, (2, self.n_h, self.n_heads, self.head_dim)).unbind(-4)
        q = q.unflatten(-1, (self.n_heads, self.head_dim))
        sd = scan_dtype(q.dtype)
        q = l2_normalize(q.to(sd)) * self.head_dim**-0.5
        return q, l2_normalize(k.to(sd)), v.to(sd)

    def _finish(self, o, u):
        o = self.o_norm(o.to(u.dtype))
        o = o.flatten(-2) * F.silu(self.out_gate(u))
        return self.out_proj(o)

    def _interleave(self, q, k, v, beta, g):
        """Flatten the n_h sub-steps into time: token t, sub-step j -> position t * n_h + j.
        Returns tensors shaped ``[b, H, T * n_h, ...]`` for the delta-rule kernels."""
        b, T, n_h, H, d = k.shape
        q_x = q.new_zeros(b, T, n_h, H, d)
        q_x[:, :, -1] = q  # read the memory only after the last sub-step
        g_x = g.new_zeros(b, T, n_h, H)
        g_x[:, :, 0] = g  # decay once per token
        flat = lambda t: t.flatten(1, 2).transpose(1, 2)  # [b, T*n_h, H, ...] -> [b, H, T*n_h, ...]
        return flat(q_x), flat(k), flat(v), flat(beta), flat(g_x)

    def forward(self, u: torch.Tensor, return_state: bool = False):
        q, k, v = self._qkv(self.conv(self.qkv_proj(u)))
        beta, g = self._gates(u)
        args = self._interleave(q, k, v, beta, g)
        if self.backend == "recurrent":
            o, S = delta_rule_recurrent(*args)
        else:
            o, S = delta_rule_chunked(*args, chunk_size=self.chunk_size)
        o = o[:, :, self.n_h - 1 :: self.n_h]  # outputs after each token's last sub-step
        out = self._finish(o.transpose(1, 2), u)
        return (out, S) if return_state else out

    def init_cache(self, batch: int, device, dtype):
        return {
            "conv": self.conv.init_state(batch, device, dtype),
            "S": torch.zeros(batch, self.n_heads, self.head_dim, self.head_dim, device=device, dtype=scan_dtype(dtype)),
        }

    def step(self, u_t: torch.Tensor, cache: dict) -> torch.Tensor:
        qkv, cache["conv"] = self.conv.step(self.qkv_proj(u_t), cache["conv"])
        q, k, v = self._qkv(qkv)  # q: [b, H, d]; k, v: [b, n_h, H, d]
        beta, g = self._gates(u_t)
        S = cache["S"] * torch.exp(g)[..., None, None]
        for j in range(self.n_h):
            v_old = torch.einsum("bhkv,bhk->bhv", S, k[:, j])
            S = S + torch.einsum("bhk,bhv->bhkv", k[:, j], beta[:, j, :, None] * (v[:, j] - v_old))
        cache["S"] = S
        o = torch.einsum("bhkv,bhk->bhv", S, q)
        return self._finish(o, u_t)

    def state_bytes(self, batch: int = 1) -> int:
        return batch * self.n_heads * self.head_dim * self.head_dim * 4
