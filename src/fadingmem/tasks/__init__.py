"""Task registry: ``make_task(name, **params)`` returns ``(sampler, vocab_size)``.

``sampler(batch_size, generator)`` produces a fresh ``(inputs, targets)`` batch on the CPU, so
training data is effectively infinite and never repeats.
"""

from __future__ import annotations

from functools import partial

from .recall import mqar, selective_copy
from .state_tracking import group_order, group_table, group_word


def make_task(name: str, **params):
    if name == "mqar":
        vocab = params.setdefault("vocab_size", 4096)
        fn = partial(mqar, **params)
    elif name == "selective_copy":
        vocab = params.setdefault("vocab_size", 16)
        fn = partial(selective_copy, **params)
    elif name == "group":
        group = params.setdefault("group", "Z2")
        vocab = group_order(group)
        fn = partial(group_word, table=group_table(group), **params)
    else:
        raise ValueError(f"unknown task {name!r}")

    def sampler(batch_size, generator):
        return fn(batch_size, generator=generator)

    return sampler, vocab


__all__ = ["make_task", "mqar", "selective_copy", "group_word", "group_table"]
