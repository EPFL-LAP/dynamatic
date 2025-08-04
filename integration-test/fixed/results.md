# Results `fixed` (Code and CFG modified)

Since the variables `x0` and `x1` depend on values from previous iterations, the speculation case did not achieve a good II. However, it successfully eliminated the latency of `subf` and `cmpf`.

|                      | No Speculation   | Speculation       |
|----------------------|------------------|-------------------|
| II (Haoran’s thesis) | 16               | 6                 |
| II                   | 14               | 5                 |
| Cycles (Test Bench)  | 705 (End: 704)   | 270 (End: 265)    |
