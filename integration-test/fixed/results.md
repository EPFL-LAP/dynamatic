# Results `fixed` (Code and CFG modified)

Since the variables `x0` and `x1` depend on values from previous iterations, the speculation case did not achieve a good II. However, it successfully eliminated the latency of `subf` and `cmpf`.

|                      | No Speculation   | Speculation       |
|----------------------|------------------|-------------------|
| II (Haoran’s thesis) | 16               | 6                 |
| II                   | 13               | 4                 |
| Cycles (Test Bench)  | 657   | 216    |
