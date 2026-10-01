# Recall capacity, first attempt (superseded)

RTX 4050, MQAR, length 512, vocabulary 4096, d_model 128, 4000 steps, 1 seed, 2 learning rates.

Not usable as a capacity measurement: even the attention baseline stays at 12-30% with 8 key-value
pairs, and accuracy is non-monotonic in the number of pairs (e.g. SSM with N=64: 41% at 8 pairs,
99% at 16). The budget was too small for the task's sudden "click", so results mostly reflect
whether a run clicked in time. CPU checks ruled out the short conv (attention + conv also failed)
and vocabulary size (vocabulary 256 also failed in 3000 steps).

Kept as a methodological example for the report. Replaced by `recall_calibration` (budget) and a
re-run of `recall_capacity`.
