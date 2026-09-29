"""The chunked (training) scans must match the step-by-step recurrences exactly."""

import math

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


@pytest.mark.parametrize("T,chunk", [(1, 8), (45, 16), (100, 32)])
def test_rotational_scan_matches_recurrent(T, chunk):
    from fadingmem.layers.ssm import rotational_scan, rotational_scan_recurrent

    torch.manual_seed(T)
    b, H, P, N = 2, 3, 4, 6
    x = torch.randn(b, T, H, P, dtype=D)
    a = torch.rand(b, T, H, dtype=D)
    theta = (torch.rand(b, T, N // 2, dtype=D) * 2 - 1) * torch.pi
    B, C = torch.randn(b, T, N, dtype=D), torch.randn(b, T, N, dtype=D)
    y1, _ = rotational_scan_recurrent(x, a, theta, B, C)
    y2, _ = rotational_scan(x, a, theta, B, C, chunk)
    torch.testing.assert_close(y1, y2)


def test_rotation_counts_mod_3():
    """The mechanism behind Z_3: rotating by 2*pi/3 per step returns the state after 3 steps.
    A real diagonal decay (even a negative one) has no transition with this property."""
    from fadingmem.layers.ssm import rotational_scan_recurrent

    T = 7
    x = torch.zeros(1, T, 1, 1, dtype=D)
    x[0, 0] = 1.0
    B = torch.tensor([1.0, 0.0], dtype=D).expand(1, T, 2)
    theta = torch.full((1, T, 1), 2 * torch.pi / 3, dtype=D)
    y, _ = rotational_scan_recurrent(x, torch.ones(1, T, 1, dtype=D), theta, B, B)
    y = y.flatten()
    torch.testing.assert_close(y[3], y[0])
    torch.testing.assert_close(y[6], y[0])
    assert (y[1:3] - y[0]).abs().min() > 0.5


def test_two_reflections_make_a_rotation():
    """DeltaProduct's premise: with beta = 2 each factor (I - 2kk^T) is a reflection, and the
    product of two reflections is a rotation (det +1, not symmetric)."""
    k1 = torch.tensor([1.0, 0.0], dtype=D)
    k2 = torch.tensor([math.cos(math.pi / 3), math.sin(math.pi / 3)], dtype=D)
    I = torch.eye(2, dtype=D)
    R = (I - 2 * torch.outer(k2, k2)) @ (I - 2 * torch.outer(k1, k1))
    torch.testing.assert_close(torch.linalg.det(R), torch.tensor(1.0, dtype=D))
    torch.testing.assert_close(R @ R.T, I)
    assert not torch.allclose(R, R.T)  # a genuine rotation (by 2 * 60 deg = 120 deg)
    torch.testing.assert_close(torch.linalg.matrix_power(R, 3), I)
