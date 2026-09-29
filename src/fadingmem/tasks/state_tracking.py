"""State-tracking probes: running products in a finite group.

Given a stream of group elements ``g_1, g_2, ...`` the target at every position is the prefix
product ``g_1 * g_2 * ... * g_t``. The group decides how hard this is for a linear RNN
(Merrill et al. 2024; Grazzi et al. 2025):

* ``Z_2`` (parity)   -- needs a transition with a *negative* eigenvalue (sign flip).
* ``Z_m`` (m > 2)    -- a rotation; diagonal real transitions cannot express it, complex /
                        rotational (Mamba-3 style) or non-diagonal ones can.
* ``S_3`` / ``S_5``  -- non-abelian permutation groups; need non-diagonal transitions
                        (DeltaNet / DeltaProduct). ``S_5`` is NC^1-complete.

Evaluating on sequences *longer* than those seen in training separates models that learned the
algorithm from those that memorised short-range statistics.
"""

from __future__ import annotations

import itertools

import torch


def cyclic_group(m: int) -> torch.Tensor:
    """Multiplication (addition mod m) table ``[m, m]``."""
    idx = torch.arange(m)
    return (idx[:, None] + idx[None, :]) % m


def symmetric_group(k: int) -> torch.Tensor:
    """Composition table of S_k over all k! permutations; ``table[i, j] = index(perm_j o perm_i)``,
    i.e. apply element i first, then element j."""
    perms = list(itertools.permutations(range(k)))
    index = {p: i for i, p in enumerate(perms)}
    n = len(perms)
    table = torch.empty(n, n, dtype=torch.long)
    for i, p in enumerate(perms):
        for j, q in enumerate(perms):
            table[i, j] = index[tuple(q[p[x]] for x in range(k))]
    return table


def group_table(group: str) -> torch.Tensor:
    kind, order = group[0].upper(), int(group[1:])
    if kind == "Z":
        return cyclic_group(order)
    if kind == "S":
        return symmetric_group(order)
    raise ValueError(f"unknown group {group!r}; use e.g. 'Z2', 'Z5', 'S3', 'S5'")


def group_word(batch_size: int, seq_len: int = 64, group: str = "Z2",
               generator: torch.Generator | None = None, table: torch.Tensor | None = None):
    """Random group words and their prefix products. Vocabulary size = group order."""
    gen = generator or torch.Generator().manual_seed(0)
    table = group_table(group) if table is None else table
    n = table.shape[0]
    inputs = torch.randint(0, n, (batch_size, seq_len), generator=gen)
    targets = torch.empty_like(inputs)
    acc = inputs[:, 0]
    targets[:, 0] = acc
    for t in range(1, seq_len):
        acc = table[acc, inputs[:, t]]
        targets[:, t] = acc
    return inputs, targets


def group_order(group: str) -> int:
    return group_table(group).shape[0]
