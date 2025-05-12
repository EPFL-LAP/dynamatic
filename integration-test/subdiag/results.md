# Results `subdiag` (Code and CFG modified)

The II could not achieve 1, because of the two load operations on `d` array inside the loop (`d[i] + d[i + 1]`). I also created an alternative benchmark (`subdiag_fast`), replacing `d[i] + d[i + 1]` with `d1[i] + d2[i + 1]`, which successfully achieved an II of 1.

Additionally, the current implementation of `cmpf` seems to generate output within the same cycle, which is unrealistic. While the change of the `cmpf` implementation does not affect the result in the speculation case, it may worsen the II in the non-speculation case.

|                      | No Speculation   | Speculation       |
|----------------------|------------------|-------------------|
| II (Haoran’s thesis) | 15               | 1                 |
| II                   | 16               | 2                 |
| Cycles (Test Bench)  | 1623 (End: 1621) | 228 (End: 222)    |
