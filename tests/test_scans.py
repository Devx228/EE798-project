"""The chunked (training) scans must match the step-by-step recurrences exactly."""

import pytest
import torch
import torch.nn.functional as F

from fadingmem.layers.deltanet import delta_rule_chunked, delta_rule_recurrent
from fadingmem.layers.ssm import ssd_scan_chunked, ssd_scan_recurrent

D = torch.float64


@pytest.mark.parametrize("T,chunk", [(1, 8), (37, 8), (64, 16), (100, 32)])
@pytest.mark.parametrize("negative", [False, True])
def test_ssd_chunked_matches_recurrent(T, chunk, negative):
    torch.manual_seed(T)
    b, H, P, N = 2, 3, 4, 5
    x = torch.randn(b, T, H, P, dtype=D)
    a = torch.rand(b, T, H, dtype=D)
    if negative:
        a = 2 * a - 1
    B, C = torch.randn(b, T, N, dtype=D), torch.randn(b, T, N, dtype=D)
    h0 = torch.randn(b, H, P, N, dtype=D)
    y1, h1 = ssd_scan_recurrent(x, a, B, C, h0)
    y2, h2 = ssd_scan_chunked(x, a, B, C, chunk, h0)
    torch.testing.assert_close(y1, y2)
    torch.testing.assert_close(h1, h2)


@pytest.mark.parametrize("T,chunk", [(1, 8), (37, 8), (64, 16), (100, 32)])
@pytest.mark.parametrize("beta_max,gated", [(1.0, True), (2.0, True), (1.0, False)])
def test_delta_chunked_matches_recurrent(T, chunk, beta_max, gated):
    torch.manual_seed(T)
    b, H, dk, dv = 2, 3, 6, 5
    q = torch.randn(b, H, T, dk, dtype=D)
    k = F.normalize(torch.randn(b, H, T, dk, dtype=D), dim=-1)
    v = torch.randn(b, H, T, dv, dtype=D)
    beta = beta_max * torch.rand(b, H, T, dtype=D)
    g = -torch.rand(b, H, T, dtype=D) if gated else torch.zeros(b, H, T, dtype=D)
    S0 = torch.randn(b, H, dk, dv, dtype=D)
    o1, S1 = delta_rule_recurrent(q, k, v, beta, g, S0)
    o2, S2 = delta_rule_chunked(q, k, v, beta, g, chunk, S0)
    torch.testing.assert_close(o1, o2)
    torch.testing.assert_close(S1, S2)


def test_delta_rule_retrieves_stored_value():
    """With beta=1 and no decay, writing (k, v) and reading with q=k returns v exactly."""
    k = F.normalize(torch.randn(1, 1, 1, 8, dtype=D), dim=-1)
    v = torch.randn(1, 1, 1, 4, dtype=D)
    o, _ = delta_rule_recurrent(k, k, v, torch.ones(1, 1, 1, dtype=D), torch.zeros(1, 1, 1, dtype=D))
    torch.testing.assert_close(o, v)


def test_negative_decay_flips_sign():
    """The mechanism behind parity: a = -1 flips the state, a in (0, 1) never can."""
    x = torch.zeros(1, 3, 1, 1, dtype=D)
    x[0, 0] = 1.0
    ones = torch.ones(1, 3, 1, dtype=D)
    y, _ = ssd_scan_recurrent(x, a=-ones, B=ones, C=ones)
    assert y.flatten().tolist() == [1.0, -1.0, 1.0]
