# LSQ

If the Dynamatic compiler cannot statically disambiguate memory accesses, it will instantiate spatial load-store queues (LSQs).
They sit between elastic datapath and memory array and are responsible for ordering memory accesses (i.e., loads and stores) sufficiently to guarantee program correctness.
If possible, they allow out-of-order execution of memory accesses to improve kernel performance by delivering load data to the datapath as quickly as possible.

## Example

Consider this `histogram` kernel as a simple example:
```c
void histogram(int[] hist, int[] idx, int n) {
  for (int i = 0; i < n ; i++)
    hist[idx[i]] += 1;
}
```

Each iteration of this kernel increments some entry of the `hist` array.
However, which entry is modified is not known statically, since it depends on the `idx` array, which is itself an input to the kernel.
This necessitates use of an LSQ to ensure correctness.

The simplest possible LSQ would simply enforce that all memory accesses to the `hist` array are performed in program order, as shown here:
1. `LD hist[idx[0]]`
1. `ST hist[idx[0]]`
1. `LD hist[idx[1]]`
1. `ST hist[idx[1]]`
1. and so on...
This would guarantee correctness, but sacrifices substantial performance.
The reason is that the in-order execution of memory accesses effectively means that consecutive loop iterations do not overlap (or rather, overlap very little).
Put differently, the loop initiation interval (II) is approximately the latency of the load-increment-store path.

Now consider the best case, where all `idx` values are different.
In that case, an ideal schedule would simply use an II of 1:
there is no need to order memory accesses from one loop iteration with respect to another loop iteration.
Clearly, executing memory accesses out-of-order can increase performance substantially.

This potential is realized by using more complex *out-of-order* spatial LSQs.

## Out-of-Order LSQs

Dynamatic's spatial LSQ is based on a design described in [this TECS'17 paper](https://doi.org/10.1145/3126525).

### Overview 

The LSQ uses two circular queue structures to hold load and store requests, respectively.
Each time a basic block is entered, all its memory operations (the *group*) are allocated into the two queues (in-order) by the *group allocator*.
The LSQ then interfaces with all load and store operators and accepts operands (load/store addresses and store data) out-of-order, placing them into the pre-allocated queue slots.
As soon as a memory request is ready (i.e., all its operands are available), it may be executed by the out-of-order issue logic of the LSQ.
For stores, the request completes with the arrival of the (empty) response from memory, and the store queue entry is deallocated.
For loads, data arrives back from memory and is placed back into the queue.
From there, it is returned to the circuit's load operators, and the load queue entry is deallocated.

### Out-of-Order Features

The LSQ supports out-of-order execution by comparing requests' memory addresses at runtime:
if a load and a store target different memory locations, they can (in principle) be executed out of order.

In the following, we describe the different out-of-order features supported by the LSQ and describe which workloads they benefit.

#### Allowing Loads to Skip Older Stores

If a load is younger than a store (i.e., the store comes first in program order) and they target different addresses, the load can safely be executed before the store.

This is the most commonly used out-of-order feature of the LSQ.
For example, it is what unlocks most performance in the above `histogram` example kernel, since it allows loads to "run ahead" of stores in the absence of conflicts.

#### Allowing Stores to Skip Older Loads

If a store is younger than a load (i.e., the load comes first in program order) and they target different addresses, the store can safely be executed before the load.

This feature is used less frequently due to the nature of most compute kernels.
They typically load some data, perform some computations, and then store it at the end.
For this reason, once a store's operands are available, all previous loads have typically completed.
In many cases, there is even a data dependency between these loads and the store.
One (constructed) example where this might be beneficial is:
```c
void store_before_load(int a[], int b[], int n) {
  for (int i = 0; i < n; i++) {
    int x, y;
    x = a[i]                     // load 1
    y = b[(x * x + x) / 3 + x];  // load 2
    b[x] = 1;                    // store 3
    b[x + 1] = y;                // store 4
  }
}
```
In this example, the index for load 2 requires a complex computation involving `x`, while store 3 only requires `x` itself and the constant 1 as its operands.
In this case, store 3 could execute ahead of load 2.

#### Allowing Loads to Skip Older Loads

Sometimes, the load that is "next in line" cannot be executed, either because its address is not known yet, or because the LSQ has determined that it conflicts with an older store that it needs to wait for.
In these cases, the LSQ allows younger loads to "skip ahead" and execute before the older load (given they are not blocked themselves).

This feature is rarely needed in practice.
However, it can significantly increase performance if address sequences have certain structures.
Consider the (constructed) example:
```c
void load_before_load(int x[], int idx1[], int idx2[], int idx3[]) {
  for (int i = 0; i < n; i++)
    int a, b;
    a = x[idx1[i]];      // load 1
    b = x[idx2[i]];      // load 2
    x[idx3[i]] = a + b;  // store 3
}
```
Depending on the contents of `idx1`, `idx2`, and `idx3`, this kernel may benefit from this feature.
If `idx1[i]` matches `idx3[i-1]`, then load 1 cannot execute because it has to wait for the previous iteration's store 3 to complete.
If `idx2[i]` does not have the same issue, then load 2 could execute ahead of load 1, which reduces the time until both `a` and `b` are available.
Among Dynamatic's integration tests, this feature is most beneficial for the `matrix_power` kernel, which has a similar (but more complex) structure to the above example.

#### Allowing Stores to Skip Older Stores

If the "next in line" store cannot be executed because it is blocked (unknown address, missing data, or conflict with older load), it would be possible for a younger store to "skip ahead" and execute first.
As with loads skipping ahead of older loads, the younger store would need to have its operands available.
However, with stores, there is an additional condition.
A store may not execute ahead of an older store with the same address (as this would constitute a write-after-write hazard).

To avoid comparing store addresses with one another, which comes with additional complexity and resource cost, **Dynamatic's LSQ does not support this feature**.
This means that stores (among themselves) are executed fully in program order.

We believe the benefits of this feature would be very minor for real-world workloads.

### Store-to-Load Forwarding (Bypass)

If a load targets the same address as a previous store, the LSQ can bypass the store data directly to the subsequent load, given that:
- the store address is known,
- the load address is known,
- the store and load addresses are equal,
- all stores ordered between the store and the load have known addresses which are different from the load address.

With this feature, loads which conflict with earlier stores can have their load data returned earlier since it avoids the latency of two memory roundtrips (one to write, one to read back).
Furthermore, the memory read can be omitted (the write operation is still needed), which saves memory read bandwidth.

Bypass is most useful for kernels with frequent conflicts, since they benefit more from the reduced latency.
One such example is a naive matrix-vector multiplication kernel like this:
```c
void matvec(int y[], int a[], int x[], int n) {
  for (int i = 0; i < n; i++) {
    y[i] = 0;
    for (int j = 0; j < n; j++) {
      y[i] += a[i * n + j] * x[j];
    }
  }
}
```
In this case, every load from `y[i]` will conflict with the previous store, so bypass would reduce the latency of each load.
However, this example is not ideal, as the accumulator could simply be stored in a temporary variable and only written to memory once at the end.
This would eliminate the need for an LSQ entirely.

Bypass is especially beneficial for workloads that are constrained by memory read bandwidth (e.g., because they perform multiple reads for each write) and have frequent conflicts (so that bypass path is used regularly).
By reducing the number of reads from memory, bypass can lower the II from (for example) 2 to 1, thereby doubling throughput and approximately halving kernel latency.

### Resource Cost and Timing Considerations

While out-of-order LSQs can help minimize kernel latencies, they consume significant resources and can impact the circuit's critical path.
We describe the trade-off between minimizing latency and minimizing resource cost and/or critical path below.

> **Note:**
> A configuration file is used to specify LSQ parameters such as load/store queue depth and whether the bypass network is enabled.
> The compiler currently hard-codes the queue depths to 16 and always enables the bypass network.
> These parameters are currently not exposed to users of Dynamatic.
> However, they could be manually overridden in [`dynamatic/tools/backend/lsq-generator-python/core_gen/configs.py`](https://github.com/EPFL-LAP/dynamatic/blob/main/tools/backend/lsq-generator-python/core_gen/configs.py).
> When doing that, care must be taken in case a kernel includes multiple LSQs.
> One option to expose the configuration of LSQs to the user would be through pragmas in the kernel source code.

#### Resource Cost

A fully-fledged out-of-order LSQ in Dynamatic can dominate total resource usage, especially when attached to small datapaths.

The main drivers of resource cost are (in this order) the bypass network and the out-of-order issue logic (including address comparators).
The bypass network could be disabled completely if it is not beneficial for a given workload, saving up to half of total LSQ resources.
In general, most logic scales with load/store queue depth or even with their product.

Thus, reducing queue sizes can significantly reduce total resource usage.
However, this can come at a latency cost, since smaller queues fill up more quickly, which can cause backpressure into the datapath and stall the entire circuit.
Nevertheless, this can be an effective lever if the LSQ's resource cost is prohibitive otherwise.
There is no potential for deadlocks or incorrect results even with very small queue depths.
The only constraint is that the queues are large enough to allocate the loads/stores from each basic block at once (with one additional entry free in each queue). 

#### Timing (Critical Path)

Next to high resource cost, the LSQ often includes long timing paths, which may be critical in the overall design.
If the LSQ is responsible for the critical path, it is usually either at its output (where load data is returned to the circuit) or within the bypass network.

All these paths typically include *cyclic priority masking* units, whose delay is linear in either the load or store queue depth (not their product).
