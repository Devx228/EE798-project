"""Triton kernel vs reference. Runs on a GPU, or on CPU with TRITON_INTERPRET=1."""

import os

import pytest
import torch

triton = pytest.importorskip("triton")
if not torch.cuda.is_available() and os.environ.get("TRITON_INTERPRET") != "1":
    pytest.skip("needs CUDA or TRITON_INTERPRET=1", allow_module_level=True)

from fadingmem.layers.ssm import ssd_scan_recurrent  # noqa: E402
from fadingmem.layers.triton_scan import ssd_scan_triton  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.mark.parametrize("T,P,N", [(1, 8, 4), (33, 16, 16), (50, 40, 12)])
@pytest.mark.parametrize("negative", [False, True])
def test_triton_matches_reference(T, P, N, negative):
    torch.manual_seed(0)
    b, H = 2, 3
    x = torch.randn(b, T, H, P, device=DEVICE)
    a = torch.rand(b, T, H, device=DEVICE)
    if negative:
        a = 2 * a - 1
    B, C = torch.randn(b, T, N, device=DEVICE), torch.randn(b, T, N, device=DEVICE)
    y_ref, h_ref = ssd_scan_recurrent(x, a, B, C)
    y, h = ssd_scan_triton(x, a, B, C, block_p=16)
    torch.testing.assert_close(y, y_ref, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(h, h_ref, atol=1e-4, rtol=1e-4)
