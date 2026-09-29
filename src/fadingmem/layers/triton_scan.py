"""Forward-only Triton kernel for the selective (SSD) scan.

Parallelisation: one program per (batch, head, block of P channels). Each program keeps its
``[BLOCK_P, N]`` slice of the state in registers and walks over time, so the state never touches
global memory -- only the inputs (x, a, B, C) are read once and y is written once. The kernel is
therefore memory-bandwidth bound, which is exactly what the roofline analysis in the report
examines.

Run the unit tests without a GPU via ``TRITON_INTERPRET=1 pytest tests/test_triton_scan.py``.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _ssd_scan_fwd_kernel(
    x_ptr, a_ptr, b_ptr, c_ptr, y_ptr, h_ptr,
    T, H, P, N,
    s_xb, s_xt, s_xh, s_xp,
    s_ab, s_at, s_ah,
    s_bb, s_bt, s_bn,
    s_cb, s_ct, s_cn,
    s_yb, s_yt, s_yh, s_yp,
    s_hb, s_hh, s_hp, s_hn,
    BLOCK_P: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_p = tl.program_id(1)
    bi = pid_bh // H
    hi = pid_bh % H

    offs_p = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
    offs_n = tl.arange(0, BLOCK_N)
    mask_p = offs_p < P
    mask_n = offs_n < N

    x_base = x_ptr + bi * s_xb + hi * s_xh + offs_p * s_xp
    y_base = y_ptr + bi * s_yb + hi * s_yh + offs_p * s_yp
    a_base = a_ptr + bi * s_ab + hi * s_ah
    b_base = b_ptr + bi * s_bb + offs_n * s_bn
    c_base = c_ptr + bi * s_cb + offs_n * s_cn

    state = tl.zeros([BLOCK_P, BLOCK_N], dtype=tl.float32)
    for t in range(T):
        a_t = tl.load(a_base + t * s_at).to(tl.float32)
        x_t = tl.load(x_base + t * s_xt, mask=mask_p, other=0.0).to(tl.float32)
        b_t = tl.load(b_base + t * s_bt, mask=mask_n, other=0.0).to(tl.float32)
        c_t = tl.load(c_base + t * s_ct, mask=mask_n, other=0.0).to(tl.float32)
        state = a_t * state + x_t[:, None] * b_t[None, :]
        y_t = tl.sum(state * c_t[None, :], axis=1)
        tl.store(y_base + t * s_yt, y_t, mask=mask_p)

    h_ptrs = h_ptr + bi * s_hb + hi * s_hh + offs_p[:, None] * s_hp + offs_n[None, :] * s_hn
    tl.store(h_ptrs, state, mask=mask_p[:, None] & mask_n[None, :])


def ssd_scan_triton(x, a, B, C, block_p: int = 32):
    """Same contract as :func:`fadingmem.layers.ssm.ssd_scan_recurrent` (no initial state, no autograd)."""
    b, T, H, P = x.shape
    N = B.shape[-1]
    y = torch.empty(b, T, H, P, device=x.device, dtype=torch.float32)
    h = torch.empty(b, H, P, N, device=x.device, dtype=torch.float32)
    BLOCK_P = min(triton.next_power_of_2(P), block_p)
    BLOCK_N = triton.next_power_of_2(N)
    grid = (b * H, triton.cdiv(P, BLOCK_P))
    _ssd_scan_fwd_kernel[grid](
        x, a, B, C, y, h,
        T, H, P, N,
        *x.stride(), *a.stride(), *B.stride(), *C.stride(), *y.stride(), *h.stride(),
        BLOCK_P=BLOCK_P, BLOCK_N=BLOCK_N,
    )
    return y, h
