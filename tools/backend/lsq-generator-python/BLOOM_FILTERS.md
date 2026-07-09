# Bloom Filters in Dynamatic's LSQ

This document describes the major design decisions that were made when implementing Bloom filters into Dynamatic's LSQ.
More detailed information about the final design can be found in the semester project report of Max Wipfli (*Towards Resource-Efficient Spatial LSQ Architectures*).

## Overview

The goal of this project was to use Bloom filters for dependency checking in the LSQ to replace the comparator matrix (`addr_same_pcomp[i][j]`).

The **final design** uses
- a **single Bloom filter** for dependency-checking the ***oldest waiting load*** (and issue it if there are none),
- **_conservative store issue_**, where a store is only issued once there are no older loads still outstanding.
The rationale for choosing this configuration is explained in detail in chapters 6–8 of the semester project report.

The LSQ generator also supports using a single Bloom filter for store issue (rather than conservative store issue), largely mirroring the situation for load issue.
This feature is not used in the final design that was evaluated during the semester project.

## Configuration

The following configuration options (available in `vhdl_gen/configs.py`) are available to configure the Bloom filters:
- `bloomFilterLoad`: Whether to use a Bloom filter for load issue.
- `bloomFilterStore`: Whether to use a Bloom filter for store issue.
- `bloomFilterSequential`: Whether to hash addresses and encode them into Bloom filters already in the store/load address dispatchers.
- `bloomFilterHashCount`: The number of hash fuctions to use in the Bloom filter ($= k$).
- `bloomFilterHashW`: The width of each hash function's output ($= log_2(m)$).
- `bloomFilterSeed`: Fixed seed for the randomized selection of hash functions.
Additionally, `bloomFilterW` is the width of the Bloom filter bit vector ($= m$), which is computed as `2 ** bloomFilterHashW`.

To enable conservative store issue, the following option is available:
- `stIssueNoCompare`: Whether to avoid any dependency checking for stores and only issue them once all older loads have completed.

There are also a number of other configuration options that were used for intermediate experiments.
However, these are not relevant for the final design and have only been implemented for latency measurements, i.e., physical design was not considered.
They are `fallbackIssueLoad`, `fallbackIssueStore`, `inOrder`, `issueOldestLoads`, and `issueOldestLoadsType`.

## Bloom Filters

### High-Level Description

#### Problem

To guarantee correctness, the load issue logic must ensure that a load is only issued if all stores in its _conflict set_ target different addresses, i.e., there are no read-after-write (RAW) dependencies between any of the stores in the conflict set and the load in question.

The _conflict set_ of a load consists of all stores which are older than the load (in program order) and have not yet received their response from memory, i.e., there is no guarantee that the store data has been written to memory.

#### Solution Approach

We can solve this problem using Bloom filters by constructing a filter vector that includes the addresses of all stores in the conflict set.
We then check whether the load's address is contained in the filter; if it is not, the load can be issued.

#### Stateful Bloom Filters

Typically, a Bloom filter is a stateful component where elements (i.e., addresses) can be added to the set cycle-by-cycle.
To add an address, it is hashed and encoded to create a temporary Bloom filter with only this single-entry, which can then be combined with the existing filter using a logical OR operation.
It is possible to add multiple addresses in the same cycle by duplicating the hashing-and-encoding logic.
To be able to remove addresses from an existing Bloom filter, a counting Bloom filter (CBF) would have to be used.

The challenge with such a stateful Bloom filter is that the conflict set can change in almost arbitrary ways cycle-to-cycle, and the Bloom filter would therefore need to track these quasi-arbitrary updates.
In particular, when a load is issued, the Bloom filter must update to track the conflict set of the next load candidate.
In general, there can be an arbitrary number of stores between these two loads (e.g., if there is a load, a loop of stores, and then the next load), limited only by the size of the store queue.

It should be possible to update a stateful (counting) Bloom filter in this way.
For example, a "delta update" consisting of all the stores between two loads could be computed ahead of time and then be applied in a single cycle.
However, there are some complications there when store addresses arrive out-of-order.

While we believe that it may be possible to implement a stateful Bloom that tracks the conflict set in a resource-efficient way, this would require significant engineering effort to ensure all possible changes to the conflict set are correctly handled.
For this reason, we decided to go another route, which is easier to implement but more resource-intensive.

#### Combinational Bloom Filters

The combinational approach to constructing a Bloom filter vector is very simple.
Each element (i.e., address) in the conflict set is converted into its single-entry Bloom filter representation (i.e., hashed and encoded), and all these representations are simply combined through logical-OR reduction.

In contrast to the stateful construction which must track _changes_ to the conflict set, this strategy simply needs to know the current conflict set.
However, it needs to be built to construct the largest possible conflict set, which would encompass the full store queue.

As described above, we decided to use this approach to avoid the engineering complexity of stateful Bloom filters, even if they might be more resource-efficient.

### Distribution of Components

> **Note:**
> We first describe the implementation of Bloom filters for load issue.
> The differences when using them for store issue (or both load and store issue) are described later.

#### Filter Construction

When building a Bloom filter combinationally, there are four main steps.

1. **Hashing:** Each element (i.e., address) is hashed using $k$ different hash functions, resulting in $k$ hash values with a width of $log_2(m)$ each.
2. **One-Hot Encoding:** The hash values are one-hot encoded individually, resulting in $k$ bit vectors of width $m$.
3. **Single-Entry Filter Construction:** The $k$ one-hot vectors are combined using logical OR (i.e., OR-reduced) into a single $m$-bit vector: the single-entry Bloom filter.
4. **Final Filter Construction:** The single-entry filters for all entries in the set are combined using OR-reduction, yielding the final $m$-bit Bloom filter vector.

The **first three steps** are performed separately for each address.
We implement them in the `BloomFilterHash` module in [bloom_filter.py](./vhdl_gen/generators/bloom_filter.py).
The module takes in the address, performs the first three steps, and outputs an $m$-bit single-entry Bloom filter vector.
It is fully combinational/stateless.

To avoid duplicating this logic for each store queue entry, we place the module within the store address port-to-queue dispatcher.
Thus, we need one hash module per store port, instead of one per store queue entry.
Since there are usually fewer store ports (typically 1 to 3) than store queue entries (typically 16), this can save significant resources.
The downside is that this requires storing an additional $m$-bit Bloom filter vector within each store queue entry.
We believe that this is mostly not an issue because FPGAs have many available flip-flops that can be used "almost for free".

> **Note:**
> It would also be possible to split these three steps.
> For example, we could pre-compute the hashes, store them in the store queue, and then do the one-hot-encoding and OR-reduction in parallel for all store queue entries.
> This could potentially save registers in the store queue if storing the hash values ($k \cdot \log_2(m)$ bits) requires less space than storing the filter ($m$ bits).
> 
> However, we believe the hashing is the most inexpensive step, and the logic we actually want to deduplicate are the one-hot encoding and OR-reduction, which this splitting would not achieve.

The **final step** can only be performed once we know which load we want to issue.
This is required so we can obtain the conflict set (from the order matrix).
Thus, this step is done as part of the load issue logic, and consists of three separate sub-steps:

First, the single-entry filter of each store is first pre-processed.
If the store entry is not occupied or has already completed from memory, the filter is set to all-zeros (effectively removing it from the set).
Otherwise, if the store entry's address is still unknown, the filter is set to all-ones (guaranteeing that any load address that is checked will result in a match/conflict).

> **Note:**
> The masking for unknown addresses could also be performed out-of-band using a separate computation: "Does any store in the conflict set have an unknown address?"
> The final conflict decision would then be:
> 
> conflict IF bloom_filter_match OR any_address_unknown
>
> We decided to do it this way since to reduce timing pressure, since the masking logic's critical input is the row/column from the order matrix (which requires first finding the load candidate) rather than the single-entry filters themselves.

Second, the order matrix is used to mask out all filters where the store is younger than the load (i.e., their Bloom filters are set to all-zeros).

Third, all masked filters are OR-reduced into a single combined filter.
It will represent the addresses of all stores in the conflict set.
If at least one store in the conflict set has an unknown address, the filter will be all-ones.

> **Note:**
> This describes the behavior with `bloomFilterSequential = True`, which is what was evaluated.
> If `bloomFilterSequential` is set to `False`, the single-entry filters for stores are created on the fly as part of the load issue logic, rather than being created in the dispatchers.
> This requires more `BloomFilterHash` modules but saves the registers in the store queue.
>
> The naming of this configuration flag is somewhat unfortunate:
> It does not change between using combinational and stateful Bloom filters as described above, but rather handles an implementation detail specific to combinational Bloom filters.

#### Filter Checking

Once the Bloom filter representing the set of conflicting stores is available, we need to check whether the load candidate's address is contained in that set.
There are two main steps to do this:

1. Obtain the single entry filter for the load candidate address.
   This is done as part of the load issue logic using another instance of the `BloomFilterHash` module.
2. Check whether all bits in the load addresses' single-entry filter are also set in the Bloom filter of conflicting stores. If yes, then there might be a match and thus conflict. If no, there cannot be one.

#### Pipelining

There are two pipeline stages relevant to Bloom filters: `pipeComp` and `pipe0`.
The final stage (`pipe1`), which registers the outputs of the load/store issue logic, is not affected by our changes.

The exact location of the pipeline stages is described in detail in **Figure 8.4** of the semester project report.

`pipeComp` is trivial to implement.

For `pipe0`, we had to somehow break the long critical path:
```
ldq_issue -> load candidate -> order matrix -> BF construction -> BF check -> issue decision -> ldq_issue
```

To do this, we implemented a "prediction strategy" for the load candidate, which takes a long time to compute as it requires `CyclicPriorityMasking`.
We do this by computing the load candidate, and then shifting it by one bit to generate what is likely to be the next load candidate.
Then, the right candidate is selected based on whether a load is issued in the current cycle or not, and the selected load candidate is stored in the `pipe0` stage.

> **Note:**
> To accurately compute the next load candidate, we would need to apply another round of cyclic priority masking, which inflates the critical path too much.
> Thus, we apply the simple shift-by-one strategy, which is accurate in most cases.
> The only case where this fails is if the next load has already been issued (which is technically possible if the current load candidate received its address later than the next load in the queue).
> To handle this, we gate the next load candidate with whether it has already been issued.
> If so, there is simply no load candidate in the next cycle, which creates a temporary 1-cycle stall.
> We are fine with this since this case is very rare in practice.

> **Note:**
> This expensive load candidate computation could be removed if we switched from issuing the *oldest waiting load* to issuing the *oldest unissued load*, i.e., if loads were issued fully in-order.
> This would allow using a load issue *pointer* (similar to how there is a store issue pointer currently) instead of load issue *bits*.
> The load issue pointer would then always point to the load candidate, so finding it would be trivial.
> This could then use the same prediction strategy that is currently used for store issue (`store_*_curr` vs. `store_*_next`), which is much simpler to compute.

### Bloom Filters for Store Issue

When using **Bloom filters only for store issue** instead of load issue, the situation is simply flipped.
In particular, load addresses are now hashed and encoded in the dispatcher and the single-entry filter vectors stored in the load queue.
The store candidate is selected by using the pre-existing store issue pointer (`stq_issue`) and subsequently hashed and encoded on the fly.
When `pipe0` is enabled, we use the pre-existing look-ahead infrastructure with `store_*_curr` and `store_*_next` and then multiplex based on `stq_issue_en`.

When using **Bloom filters for both load and store issue**, all addresses are hashed and encoded in the dispatcher.
This means both the load and store queues contain the single-entry filter vectors for all addresses, and no more on-the-fly hashing and encoding is required.
Rather, multiplexers are used to look up the filter vector for the load and store candidates from the queue entries.

## Conservative Store Issue

To implement conservative store issue, we largely re-use the existing store issue logic from the original LSQ.

With the original logic, a store-load conflict was present if all of the following conditions are met:
1. The load entry is valid/allocated.
2. The load entry has not completed yet (i.e., received data back from memory).
3. The load is older than the store.
4. The store and load addresses match OR the load address is not valid yet.

We adapt this by simply assuming the last condition to always be true, which removes any dependence on address comparisons.
The remaining logic only does a few simple Boolean operations to determine a conflict.

Due to the small amount of remaining logic, we did not thoroughly consider potential other options that achieve the same behavior.

