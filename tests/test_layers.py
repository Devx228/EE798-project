"""Module-level checks: shapes, gradients, and decoding (step) == parallel forward."""

import pytest
import torch

from fadingmem.model import ModelConfig, SequenceModel

PATTERNS = ["A", "M", "D", "MA", "DA"]


def small_model(pattern, neg_eigen=False, **kw):
    cfg = ModelConfig(
        vocab_size=50, d_model=32, n_layers=len(pattern), pattern=pattern, mlp_ratio=2, neg_eigen=neg_eigen,
        attn={"n_heads": 2}, ssm={"d_state": 8, "expand": 2, "head_dim": 16, "chunk_size": 8},
        deltanet={"n_heads": 2, "chunk_size": 8}, **kw,
    )
    return SequenceModel(cfg).double()


@pytest.mark.parametrize("pattern", PATTERNS)
def test_forward_backward(pattern):
    torch.manual_seed(0)
    model = small_model(pattern)
    x = torch.randint(0, 50, (2, 21))
    logits, loss = model(x, x)
    assert logits.shape == (2, 21, 50)
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name


@pytest.mark.parametrize("pattern", PATTERNS)
@pytest.mark.parametrize("neg_eigen", [False, True])
def test_step_matches_forward(pattern, neg_eigen):
    torch.manual_seed(1)
    model = small_model(pattern, neg_eigen).eval()
    x = torch.randint(0, 50, (2, 19))
    with torch.no_grad():
        full = model(x)
        caches = model.init_cache(2, "cpu", torch.float64, max_len=32)
        steps = torch.stack([model.step(x[:, t], caches) for t in range(x.shape[1])], dim=1)
    torch.testing.assert_close(full, steps, atol=1e-8, rtol=1e-6)


def test_recurrent_backend_matches_chunked():
    torch.manual_seed(2)
    model = small_model("MD").eval()
    x = torch.randint(0, 50, (2, 30))
    with torch.no_grad():
        chunked = model(x)
        for block in model.blocks:
            block.mixer.backend = "recurrent"
        recurrent = model(x)
    torch.testing.assert_close(chunked, recurrent)
