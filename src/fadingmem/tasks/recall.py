"""Memory / recall probes.

* Multi-Query Associative Recall (MQAR, Arora et al. 2023 "Zoology"): the model sees key-value
  pairs, then must answer many queries ``key -> value``. Needs *content-based lookup* -- the thing
  softmax attention does natively and a fixed-size recurrent state has to squeeze in.
* Selective copying (Gu & Dao 2023): data tokens are scattered among noise; after the marker the
  model must reproduce them in order. Needs *content-aware filtering* of what to store.

Every sampler returns ``(inputs, targets)`` of shape ``[batch, seq_len]``; targets are ``-100``
where no prediction is scored.
"""

from __future__ import annotations

import torch

IGNORE = -100


def _sample_without_replacement(n_rows: int, population: int, k: int, gen: torch.Generator) -> torch.Tensor:
    return torch.rand(n_rows, population, generator=gen).argsort(dim=1)[:, :k]


def mqar(batch_size: int, seq_len: int = 256, num_kv_pairs: int = 16, vocab_size: int = 4096,
         generator: torch.Generator | None = None):
    """Token 0 is filler; keys come from ``[1, V/2)`` and values from ``[V/2, V)``.

    Layout: ``k1 v1 k2 v2 ... kn vn | ... q_a . q_b . ...`` where every key appears exactly once as
    a query somewhere in the remaining positions (followed by its value, as in Zoology). The loss is
    scored only at query positions, whose target is the associated value.
    """
    gen = generator or torch.Generator().manual_seed(0)
    n = num_kv_pairs
    context = 2 * n
    slots = (seq_len - context) // 2
    if slots < n:
        raise ValueError(f"seq_len={seq_len} too short for {n} kv pairs (need >= {4 * n})")
    half = vocab_size // 2

    keys = _sample_without_replacement(batch_size, half - 1, n, gen) + 1
    values = torch.randint(half, vocab_size, (batch_size, n), generator=gen)

    seq = torch.zeros(batch_size, seq_len + 1, dtype=torch.long)
    seq[:, 0:context:2] = keys
    seq[:, 1:context:2] = values

    # queries in a random order, placed at random (even-aligned) free slots after the context
    order = torch.rand(batch_size, n, generator=gen).argsort(dim=1)
    q_keys, q_vals = keys.gather(1, order), values.gather(1, order)
    slot_idx = _sample_without_replacement(batch_size, slots, n, gen).sort(dim=1).values
    q_pos = context + 2 * slot_idx
    seq.scatter_(1, q_pos, q_keys)
    seq.scatter_(1, q_pos + 1, q_vals)

    inputs = seq[:, :-1].clone()
    targets = torch.full((batch_size, seq_len), IGNORE, dtype=torch.long)
    targets.scatter_(1, q_pos, q_vals)
    return inputs, targets


def selective_copy(batch_size: int, seq_len: int = 256, num_tokens: int = 16, vocab_size: int = 16,
                   generator: torch.Generator | None = None):
    """Token 0 = noise/blank, token 1 = marker, data tokens in ``[2, V)``.

    ``seq_len`` counts the full input: ``(seq_len - num_tokens)`` region positions followed by
    ``num_tokens`` markers. At the i-th marker the target is the i-th data token.
    """
    gen = generator or torch.Generator().manual_seed(0)
    region = seq_len - num_tokens
    data = torch.randint(2, vocab_size, (batch_size, num_tokens), generator=gen)
    pos = _sample_without_replacement(batch_size, region, num_tokens, gen).sort(dim=1).values

    inputs = torch.zeros(batch_size, seq_len, dtype=torch.long)
    inputs.scatter_(1, pos, data)
    inputs[:, region:] = 1
    targets = torch.full((batch_size, seq_len), IGNORE, dtype=torch.long)
    targets[:, region:] = data
    return inputs, targets
