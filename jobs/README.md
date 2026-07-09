# Experiment Scripts

This document describes the different experiment scripts used to evaluate Dynamatic's LSQ, including the existing LSQ and a Bloom-filter-based one.
More detailed information about the experiments and their results can be found in the semester project report of Max Wipfli (*Towards Resource-Efficient Spatial LSQ Architectures*).

## General Information

Unless mentioned otherwise, each experiment script runs [run_evaluation.py](../tools/evaluation/run_evaluation.py) one or more times with different LSQ configurations.
The outputs of each evaluation run (a single JSON file each) are then written to a directory structure in `eval_results/<experiment name>` in the repository root.
To modify the LSQ, environment variables are set, which are read by the LSQ generator (see [configs.py](../tools/backend/lsq-generator-python/vhdl_gen/configs.py)) and used to override the values set by the LSQ JSON configuration file (which is provided by the compiler).

Processing/plotting is done with scripts in a separate **artifacts repository**, where the result JSONs from experiment runs during the semester project is also committed.

## List of Experiments

### `01_dynamatic_target_clock_period.sh`

**Report Section:** 4.1.1

**Variables:**
- Dynamatic target clock period

**Fixed (notable):**
- LSQ size (16)
- pipelining (none)

### `02_synthesis_target_clock_period.sh`

**Report Section:** 4.1.2

**Variables:**
- synthesis target clock period
- LSQ size

**Fixed (notable):**
- Dynamatic target clock period
- pipelining (none)

### `03_pipeline_configurations.sh`

**Report Section:** 5.1 (and as part of 5.3, and as baseline for following sections)

**Variables:**
- LSQ size (4, 6, 8, 10, 12, 16, 20)
- pipelining (all 16 configurations possible with `pipeComp`, `pipe0`, `pipe1`, `headLag`)

**Fixed (notable):**
- n/a

### `04_no_bypass.sh`

Same experiment as `03_pipeline_configurations.sh`, but simply without bypass.

**Report Section:** 5.2 (and as part of 5.3, and as baseline for following sections)

**Variables:**
- LSQ size (4, 6, 8, 10, 12, 16, 20)
- pipelining (all 16 configurations possible with `pipeComp`, `pipe0`, `pipe1`, `headLag`)

**Fixed (notable):**
- `bypass = False`

### `05_in_order.sh`

Artificially constrained issue to be fully in program order (using `inOrder = True`), otherwise same experiment as `04_no_bypass.sh`.

**Report Section:** 5.4 (and as part of 5.5)

**Variables:**
- LSQ size (4, 6, 8, 10, 12, 16, 20)
- pipelining (all 16 configurations possible with `pipeComp`, `pipe0`, `pipe1`, `headLag`)

**Fixed (notable):**
- `bypass = False`
- `inOrder = True`

### `07_issue_n_oldest.sh`

This experiment was not used in the end.
It is the same experiment as `07b_issue_n_oldest_store_no_compare.sh`, but with regular store issue (rather than conservative store issue).

**Variables:**
- pipelining (`headLag` and `headLag + pipe0`)
- load issue window: N-issuable, N-contiguous, per-port-N-issuable, per-port-N-contiguous (set using `issueOldestLoads` and `issueOldestLoadsType`)

**Fixed (notable):**
- LSQ size (20); avoids backpressure into datapath
- `bypass = False`

### `07a_store_no_compare.sh`

Evaluation of *conservative store issue*.
In the code, this is called "store issue without comparisons" or `stIssueNoCompare`.

This the same experiment as `04_no_bypass.sh`, with `stIssueNoCompare = True` added.

**Report Section:** 7.1

**Variables:**
- LSQ size (4, 6, 8, 10, 12, 16, 20)
- pipelining (all 16 configurations possible with `pipeComp`, `pipe0`, `pipe1`, `headLag`)

**Fixed (notable):**
- `stIssueNoCompare = True`
- `bypass = False`

### `07_issue_n_oldest_store_no_compare.sh`

This is the same experiment as `07_issue_n_oldest.sh`, but with conservative store issue (`stIssueNoCompare`) added.

**Report Section:** 7.2

**Variables:**
- pipelining (`headLag` and `headLag + pipe0`)
- load issue window: N-issuable, N-contiguous, per-port-N-issuable, per-port-N-contiguous (set using `issueOldestLoads` and `issueOldestLoadsType`)

**Fixed (notable):**
- LSQ size (20); avoids backpressure into datapath
- `stIssueNoCompare = True`
- `bypass = False`

### `10_bloom_filter_hash_unit_resources.sh`

This experiment was not used in the end.
It synthesizes the `BloomFilterHash` module for different filter configuration and address widths.

It does not use the regular evaluation flow, but rather uses a custom flow to modify the LSQ's JSON configuration and run synthesis for only the Bloom filter module.

### `11_bloom_filter_histogram.sh`

This experiment was not used in the end.
It evaluates the LSQ with Bloom filters, for the `histogram` kernel only, and only at LSQ size 16.
The next experiment (`11a_bloom_filter_histogram_sizes.sh`) is identical, but also sweeps LSQ sizes.

### `11a_bloom_filter_histogram_sizes.sh`

This experiment evaluates the LSQ with Bloom filters on the `histogram` kernel, for different filter configurations and LSQ sizes.

**Report Section:** 8.4.3

**Variables:**
- Bloom filter configuration (different values of $m$ and $k$, see **Table 8.1** in the report, *Evaluated* column)
- LSQ size (4, 6, 8, 10, 12, 16, 20)
- pipelining (`headLag` and `headLag + pipe0`)

**Fixed (notable):**
- `bloomFilterLoad = True` (Bloom filters for load issue)
- `bloomFilterSequential = True` (implementation detail)
- `bloomFilterSeed = 1` (fixed seed for generating hash functions)
- `stIssueNoCompare = True` (conservative store issue)
- `bypass = False` (no bypass)

### `12_bloom_filter_all.sh`

This experiment evaluates the LSQ with Bloom filters on all kernels, for different filter configurations and a fixed LSQ size.
In constrast to the previous experiment, it tests fewer Bloom filter configuration (for each $m$, only the "best" $k$).

**Report Section:** 8.4.1, 8.4.2

**Variables:**
- Bloom filter configuration (different values of $m$ and $k$, see **Table 8.1** in the report, *Selected* column)
- LSQ size (16)
- pipelining (`headLag` and `headLag + pipe0`)

**Fixed (notable):**
- `bloomFilterLoad = True` (Bloom filters for load issue)
- `bloomFilterSequential = True` (implementation detail)
- `bloomFilterSeed = 1` (fixed seed for generating hash functions)
- `stIssueNoCompare = True` (conservative store issue)
- `bypass = False` (no bypass)

### `13_bloom_filter_sweep_addrwidth_lsqsize.sh`

This experiment evaluates `histogram`'s LSQ (with Bloom filter), for different filter configurations, and jointly sweeps LSQ size and address width.

**Report Section:** 8.5.1, 8.5.2, 8.5.3

**Variables:**
- Bloom filter configuration ($m = 16, k = 2$ and $m = 64, k = 3$)
- LSQ size (4 to 64)
- address width (4 to 64)

**Fixed (notable):**
- pipelining (`headLag` only)
- `bloomFilterLoad = True` (Bloom filters for load issue)
- `bloomFilterSequential = True` (implementation detail)
- `bloomFilterSeed = 1` (fixed seed for generating hash functions)
- `stIssueNoCompare = True` (conservative store issue)
- `bypass = False` (no bypass)
