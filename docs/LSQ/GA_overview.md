# Group Allocator

A friendly tour of how groups (Basic Blocks) are allocated into the LSQ.



## 1. Overview and Purpose
In a dynamically scheduled dataflow circuit, there is no traditional program counter or sequential instruction fetch. The **Group Allocator** provides a mechanism to handle this. It is based on the concept of a "group," which is a sequence of memory accesses (loads and stores) within a single basic block whose internal dependency order is known at compile time.  

Instead of allocating one memory operation at a time, the Group Allocator receives a single signal to "activate" a group. In response, it allocates all loads and stores belonging to that group into the Load Queue (LDQ) and Store Queue (STQ) at once. This block-level allocation is a key point for high throughput in the dataflow circuit.  

The allocator's logic is designed around internal Read-Only Memories (ROMs) that store the pre-compiled properties of each group, such as its size, port assignments, and internal ordering.

## 2. Key Responsibilities
The Group Allocator has the following core responsibilities:

1. **Resource Checking**: It continuously checks if there is enough free space in both the Load Queue and Store Queue to accommodate all operations for any incoming group request.

2. **Handshaking and Arbitration**: It manages the valid/ready handshake with the dataflow circuit that initiates group requests. If configured to allow multiple simultaneous requests, it uses a round-robin arbitrator to select one group per cycle, ensuring fairness.

3. **Entry Allocation**: For the selected group, it generates write-enable masks to reserve the exact number of required slots in the LDQ and STQ, starting from the queues' current tail pointers.

4. **Configuration Loading**: It fetches the selected group's pre-compiled information from internal ROMs and writes this data into the newly allocated LSQ entries. This includes the designated port index for each operation and the crucial load-store dependency information needed by the LSQ's order matrix.



![Group Allocator](./figs/group_allocator_v2.png)



## 3. Dataflow Walkthrough

1. **Calculate Used Space**  
The process begins by determining how many slots are currently occupied in the LDQ and the STQ. This is done using circular subtraction (`WrapSub`) on the queue's head and tail pointers. The results, `loads_sub` and `store_sub`, represent the number of active entries in each queue.  


2. **Calculate Available Free Space**  
Using the occupied space calculated in the previous step, this stage determines the number of available free slots (`empty_loads`, `empty_stores`). If a queue's empty flag (`ldq_empty_i` or `stq_empty_i`) is asserted, the number of free slots is simply the maximum size of that queue (`numLdqEntries` or `numStqEntries`). Otherwise, it is calculated from the `loads_sub` and `stores_sub` values.


3. **Check Group Readiness**  
For each potential group, the allocator checks if there is sufficient space to enqueue all of its loads and stores. It compares the required number of loads and stores for the group (`gaNumLoads` and `gaNumStores`) against the available free space (`empty_loads` and `empty_stores`). If and only if there is enough room in both queues, the `group_init_ready` signal is asserted for that group, indicating it is possible to allocate the group.


4. **Perform Handshake and Select Group**  
A group is officially selected for allocation when the external circuit asserts its validity (`group_init_valid_i`) `AND` the allocator has signaled it is ready (`group_init_ready`). The result of this handshake is the `group_init_hs` signal, which is a one-hot vector identifying the single winning group to be allocated in the current cycle.


5. **Load and Align Port Indices**  
The winning group's ID (`group_init_hs`) is now used as an address to look up pre-compiled information from internal ROMs. First, the port indices for every load and store in the group are fetched from the `gaLdPortIdx` and `gaStPortIdx` ROMs. This raw data is then aligned to the LSQ's circular buffer by performing a cyclic left shift (`CyclicLeftShift`) based on the current tail pointers (`ldq_tail_i` and `stq_tail_i`). The final outputs, `ldq_port_idx_o` and `stq_port_idx_o`, are ready to be written into the new LSQ entries.


6. **Load and Align the Order Matrix**  
Similarly, the load-store ordering information for the group is fetched from the `gaLdOrder` ROM. Since this is a 2D matrix that must fit into the 2D circular space of the LDQ and STQ, it requires a two-stage alignment. The columns are first shifted based on the `stq_tail_i`, and then the rows are shifted based on the `ldq_tail_i`. This produces the correctly aligned `ga_ls_order_o` matrix.

7. **Load Group Size**  
The specific number of loads (`num_loads`) and stores (`num_stores`) for the chosen group is also fetched from the `gaNumLoads` and `gaNumStores` ROMs using `group_init_hs` as the select signal. This information is also required to generate the correctly sized write masks.

8. **Generate and Align Write-Enable Masks**  
Based on the group size fetched on the previous step, simple bitmasks (`ldq_wen_unshifted` and `stq_wen_unshifted`) are generated. For example, if a group has 3 loads with total of 8 LDQ slots, the mask will be `11100000`. This mask is then cyclically shifted left by the value of the corresponding queue's tail pointer (`ldq_tail_i` or `stq_tail_i`). The final output signals, `ldq_wen_o` and `stq_wen_o`, are aligned write-enable masks that will activate the correct number of consecutive slots in the LSQ's circular buffers.

## 4. Interface Signals

| Signal Name         | Description     |
| ------------------- | --------------- |
| `group_init_valid_i`| Handshake valid signal from the circuit for each group. |
| `group_init_ready_o`| Handshake ready signal indicating the LSQ can accept a group. |
| `ldq_tail_i`        | Load queue tail |
| `ldq_head_i`        | Load queue head |
| `ldq_empty_i`       | A flag indicating if the Load Queue is empty.|
| `stq_tail_i`        | Store queue tail|
| `stq_head_i`        | Store queue head|
| `stq_empty_i`       | A flag indicating if the Store Queue is empty.|
| `ldq_wen_o`         | A write-enable mask to allocate entries in the Load Queue.|
| `num_loads_o`       | The number of loads in the group being allocated. |
| `ldq_port_idx_o`    | The port indices to be written into the new Load Queue entries. |
| `stq_wen_o`         | A write-enable mask to allocate entries in the Store Queue. |
| `num_stores_o`      | The number of stores in the group being allocated.|
| `stq_port_idx_o`    | The port indices to be written into the new Store Queue entries. |
| `ga_ls_order_o`     | Group Allocator load-store order matrix |