# Results `sparse` (Code and CFG modified)

Although Haoran's thesis attempted data speculation and achieved an II of 1, it seemed to be somewhat imprecise, and the total cycle count may not have improved in that case.

|                      | No Speculation   | Speculation       |
|----------------------|------------------|-------------------|
| II (Haoran’s thesis) | 16               | 1                 |
| II                   | 15               | 10                |
| Cycles (Test Bench)  | 1237 (End: 1235) | 835 (End: 831)    |
