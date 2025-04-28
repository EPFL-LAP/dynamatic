# Results `loop_path` (Code and CFG modified)

The speculation case achieved an II of 1, matching the results in Haoran's thesis.

The II is only 2, even in the non-speculation case. This is because the break condition `(1000 - temp) <= x * temp` is optimized in the current frontend, eliminating the multiplication.

|                      | No Speculation   | Speculation       |
|----------------------|------------------|-------------------|
| II (Haoran’s thesis) | 6                | 1                 |
| II                   | 2                | 1                 |
| Cycles (Test Bench)  | 341 (End: 339)   | 175 (End: 173)    |
