import torch

from fadingmem.tasks import make_task
from fadingmem.tasks.recall import mqar, selective_copy
from fadingmem.tasks.state_tracking import group_word, symmetric_group


def test_mqar_targets_are_the_stored_values():
    x, y = mqar(8, seq_len=64, num_kv_pairs=8, vocab_size=128, generator=torch.Generator().manual_seed(0))
    assert x.shape == y.shape == (8, 64)
    for i in range(8):
        kv = {x[i, j].item(): x[i, j + 1].item() for j in range(0, 16, 2)}
        assert len(kv) == 8  # keys are distinct
        pos = (y[i] != -100).nonzero().flatten()
        assert len(pos) == 8
        for p in pos.tolist():
            assert y[i, p].item() == kv[x[i, p].item()]
            assert p + 1 >= 64 or x[i, p + 1].item() == y[i, p].item()  # value follows the query


def test_selective_copy_targets_in_order():
    x, y = selective_copy(4, seq_len=40, num_tokens=5, vocab_size=10, generator=torch.Generator().manual_seed(0))
    for i in range(4):
        data = x[i, :35][x[i, :35] >= 2]
        assert torch.equal(y[i, 35:], data)
        assert (x[i, 35:] == 1).all()


def test_parity_labels():
    x, y = group_word(16, seq_len=30, group="Z2", generator=torch.Generator().manual_seed(0))
    torch.testing.assert_close(y, x.cumsum(1) % 2)


def test_symmetric_group_is_a_group():
    table = symmetric_group(3)
    assert table.shape == (6, 6)
    for row in table:
        assert sorted(row.tolist()) == list(range(6))  # Latin square
    a, b, c = 1, 3, 4
    assert table[table[a, b], c] == table[a, table[b, c]]  # associativity
    assert not torch.equal(table, table.T)  # non-abelian


def test_registry_vocab():
    assert make_task("group", group="S3", seq_len=8)[1] == 6
    assert make_task("mqar", vocab_size=256)[1] == 256
