# Port-to-Queue Dispatcher

How addresses and data enter from multiple access ports to the LSQ's internal load and store queues.


## 1. Overview and Purpose  


![Port-to-Queue Dispatcher Top-Level](./figs/LSQ_Top-level_ptq_dispatcher.png)

The Port-to-Queue Dispatcher is a submodule within the Load-Store Queue (LSQ) responsible for routing incoming memory requests (addresses or data) from the dataflow circuit's access ports to the correct queue entries of the load queue and the store queue. All incoming requests are directed into either the load queue or the store queue. These queues are essential for tracking every memory request until its completion. It ensures each load queue or store queue entry gets the correct address or data from the appropriate port.  

We need a total of three **Port-to-Queue Dispatchers**—one each for the load address, store address, and store data. Why? To load, you must first supply the address where the data is stored. Likewise, a store operation needs both the value to write and the address to write it at.  

In the LSQ architecture, memory operations arrive via dedicated access ports. The system can process simultaneous payload writes to the LSQ from multiple ports in parallel. An arbitration mechanism is required, however, to handle cases where multiple queue entries compete for access to the same single port. 

## 2. Port-to-Queue Dispatcher Internal Blocks

![Port-to-Queue Dispatcher High-Level](./figs/PTQ_high_level.png)

Let's assume the following generic parameters for dimensionality:
* `N_PORTS`: The total number of ports.
* `N_ENTRIES`: The total number of entries in the queue.
* `PAYLOAD_WIDTH`: The bit-width of the payload (e.g., 8 bits).
* `PORT_IDX_WIDTH`: The bit-width required to index a port (e.g., `ceil(log2(N_PORTS))`).

### Port Interface Signals

![Port Interface](./figs/PTQ_Port_Interface.png)

These signals are used for communication between the external modules and the dispatcher's ports.

| Signal Name | Direction | Dimensionality | Description |
| :--- | :--- | :--- | :--- |
| **Inputs** | | | |
| `port_bits_i` | Input | `N_PORTS` of `std_logic_vector(PAYLOAD_WIDTH-1:0)` | Array of payloads (address or data), one for each port. |
| `port_valid_i` | Input | `N_PORTS` of `std_logic` | Array of valid flags. When `port_valid_i[p]` is high, the payload on `port_bits_i[p]` is valid.|
| **Outputs** | | | |
| `port_ready_o` | Output | `N_PORTS` of `std_logic` | Array of ready flags. `port_ready_o[p]` goes high if the queue can accept the payload from port `p` this cycle. |



### Queue Interface Signals

These signals are used for communication between the dispatcher logic and the queue's memory entries.

![Queue Interface](./figs/PTQ_Queue_Interface.png)

| Signal Name | Direction | Dimensionality | Description |
| :--- | :--- | :--- | :--- |
| **Inputs** | | | |
| `entry_valid_i` | Input | `N_ENTRIES` of `std_logic` | Is queue entry `e` logically allocated? |
| `entry_bits_valid_i` | Input | `N_ENTRIES` of `std_logic` | Has the address or the data slot for entry `e` already been filled? |
| `entry_port_idx_i` | Input | `N_ENTRIES` of `std_logic_vector(PORT_IDX_WIDTH-1:0)`| Indicates to which port each entry is assigned. |
| `queue_head_oh_i` | Input | `std_logic_vector(N_ENTRIES-1:0)` | One-hot vector indicating the head entry in the queue. |
| **Outputs** | | | |
| `entry_bits_o` | Output | `N_ENTRIES` of `std_logic_vector(PAYLOAD_WIDTH-1:0)`| The data to be written into each queue entry. Think of it as the ink flowing into a row on the whiteboard. |
| `entry_wen_o` | Output | `N_ENTRIES` of `std_logic` | A write-enable signal for each entry. When `entry_wen_o` of entry `e` is high, `entry_bits_valid_i[e]` will become high by the logic outside of the dispatcher. |




The Port-to-Queue Dispatcher has the following responsibilities:

1. **Matching**  
    ![Matching](./figs/PTQ_matching_description.png)  
    The Matching block is responsible for identifying which queue entries are actively waiting to receive an address or data payload.
    - **Input**:  
        - `entry_valid_i`: Indicates if the entry is allocated by the group allocator.
        - `entry_bits_valid_i`: Indicates if the entry's payload slot is already filled. 
    - **Processing**: For each queue entry, this block performs the check: `entry_valid_i AND (NOT entry_bits_valid_i)`. An entry is considered waiting only if it has been allocated (`entry_valid_i = 1`) but its payload slot is still empty (`entry_bits_valid_i = 0`)
    - **Output**:  
        - `entry_request_valid`: A array of bits indicating the queue entry is ready to receive address or data.

2. **Port Index Decoder**  
    ![Port_Index_Decoder](./figs/PTQ_Port_Index_Decoder_description.png)  
    When the group allocator allocates a queue entry, it also assigns the queue entry to a specific port, storing this port assignment as an integer. The Port Index Decoder decodes the port assignment for each queue entry from an integer representation to a one-hot representation.
    - **Input**:   
        - `entry_port_idx_i`: Queue entry-port assignment information
    - **Processing**:  
        - It performs a binary-to-one-hot conversion on the port index associated with each entry. For example, if there are 3 ports, a binary index of `01` would be converted to a one-hot vector of `010`.
    - **Output**:  
        - `entry_port_valid`: A one-hot vector for each entry that directly corresponds to the port it is assigned to.

3. **Payload Mux**  
    ![Mux1H](./figs/PTQ_Payload_Mux_description.png)  
    This block routes the address or data payload from the appropriate input port to the correct queue entries. 
    - **Input**:  
         - `port_bits_i`: An array containing the address or data payload from all access ports.
         - `entry_port_valid`: The one-hot port assignment for each queue entry, used as the select signal.
    - **Processing**: For each queue entry, a multiplexer `Mux1H` uses the corresponding `entry_port_valid` one-hot vector to select one payload from `port_bits_i` array.
    - **Output**:  
        - `entry_bits_o`: The selected payload of each queue entry.

4. **Entry-Port Assignment Masking Logic**  
    ![Entry-Port Assignment Assignment Logic](./figs/PTQ_entry_port_assignment_masking_description.png)  
    Each entry is waiting for the payload from a certain port. This block masks out the port assignments for each queue entry if it is not ready to receive the payload. It propagates these entry-port assignment information only for entries that are available to receive the payload.
    
    - **Input**:  
         - `entry_port_valid`: A one-hot vector for each entry representing its assigned port.
         - `entry_request_valid`: A bit array indicating which entries are ready to receive.
    - **Processing**: Performs a bitwise AND operation between each entry's one-hot port assignment (`entry_port_valid`) and its readiness status (`entry_request_valid`). This masks out assignments for entries that are not ready.
    - **Output**:  
        - `entry_port_request`: A one-hot vector for each entry representing its assigned port, but zero when the queue entry is not ready.



5. **Handshake Logic**  
    ![PTQ_Handshake](./figs/PTQ_Handshake_description.png)  
    This block manages the `valid/ready` handshake protocol with the external access ports. It generates the outgoing `port_ready_o` signals and produces the final entry-port assignments that have completed a successful handshake (i.e. the internal request is ready and the external port is valid).

    - **Input**:  
        - `entry_port_request`: A one-hot vector for each entry representing its assigned port, but zero when the queue entry is not ready.
        - `port_valid_i`: The incoming port valid signals from each external port.
    - **Processing**:  
        - Ready Generation: It determines if any queue entry is waiting for data from a specific port. If so, it asserts the `port_ready_o` signal for that port to indicate it can accept data. 
        - Handshake: It then uses the external `port_valid_i` signals to mask out entries in `entry_port_request` if the corresponding port is not valid.
    - **Output**:
        - `port_ready_o`: The outgoing ready signal to each external port.
        - `entry_port_and`: Represents the set of handshaked entry-port assignments. This signal indicates a successful handshake and is sent to the **Arbitration Logic** to select the oldest one.

6. **Arbitration Logic**  
    ![PTQ_Handshake](./figs/PTQ_Arbitration_description.png)  
    The core decision making block of the dispatcher. When multiple handshaked entry-port assignments are ready to be written in the same cycle, it chooses the oldest queue entry among the valid ones for each port.
    - **Input**:  
        - `entry_port_and`: The set of all currently valid and ready entry-port assignments.
        - `queue_head_oh_i`: The queue's one-hot head vector.
    - **Processing**: It uses a `CyclicPriorityMasking` algorithm. This ensures that among all candidates for each port, the one corresponding to the oldest entry in the queue is granted for the current clock cycle.
    - **Output**: `entry_wen_o` signal, which acts as the enable for the queue entry. This signal ultimately causes the queue's `entry_bits_valid` signal to go high via logic outside of the dispatcher.




## 3. Dataflow Walkthrough

![Store Address Port-to-Queue Dispatcher](./figs/PTQ_store_address.png)

### Example of Store Address Port-to-Queue Dispatcher (3 Store Ports, 4 Store Queue Entries)

1. **Matching: Identifying which queue slots are empty**  
    ![Matching](./figs/Matching.png)  
    The first job of this block is to determine which entries in the store queue are waiting for a store address.  
    Based on the example diagram:  
    - **Entry 1** is darkened to indicate that it has not been allocated by the Group Allocator. Its `Store Queue Valid` signal (equivalent to `entry_valid_i`) is `0`.  
    - **Entries 0, 2, and 3** have been allocated, so their `entry_valid_i` signal are `1`. However, among these, Entry 2 already has a valid address (`Store Queue Addr Valid = 1`).
    - Therefore, only `Entries 0 and 3` are actively waiting for their store address, as they are allocated but their `Store Queue Addr Valid` bit is still `0`.  
  
    This logic is captured by the expression `entry_request_valid = entry_valid_i AND (NOT entry_bits_valid_i)`, which creates a list of entries that need attention from the dispatcher.

2. **Port Index Decoder: Queue entries port assignment in one-hot format**
    ![Port_Index_Decoder](./figs/Port_Index_Decoder.png)  
    This block's circuit is to decode the binary port index assigned to each queue entry into a one-hot format.  
    Based on the example diagram:  
    - The `Store Queue` shows that `Entry 0` is assigned to `Port 1` , `Entry 1` to `Port 0`, `Entry 2` to `Port 1` and `Entry 3` to `Port 2`. 
    - The `Port Index Decoder` takes these binary indices (`00`, `01`, `10`) as input.
    - It processes them and generates a corresponding one-hot vector for each entry. Since there are three access ports, the vector are three bits wide:
        - `Entry 0 (Port 1)`: `010`
        - `Entry 1 (Port 0)`: `000`
        - `Entry 2 (Port 1)`: `010`
        - `Entry 3 (Port 2)`: `100`

    The output of this block, an array of one-hot vectors, is a crucial input for the `Payload Mux`, where it acts as the select signal to choose the data from the correct port.    

3. **Payload Mux: Routing the correct address**  
    ![PTQ_Payload_MUX](./figs/PTQ_Payload_MUX.png)  
    Based on the example diagram:
    - The `Access Ports` table shows the current address payloads being presented by each port:
        - `Port 0`: `01101111`
        - `Port 1`: `11111000`
        - `Port 2`: `00100000`
    - The `Port Index Decoder` has already determined the port assignments for each entry
    - The `Payload Mux` uses these assignments to perform the selection:
        - `Entry 0`: `11111000` (Address from `Port 1`)
        - `Entry 1`: `01101111` (Address from `Port 0`)
        - `Entry 2`: `11111000` (Address from `Port 1`)
        - `Entry 3`: `00100000` (Address from `Port 2`)
    
    The output of this block, `entry_bits_o` is logically committed to the queue only when the `Arbitration Logic` asserts the `entry_wen_o` signal for that specific entry.


4. **Entry-Port Assignment Masking Logic**  
    ![Entry-Port Assignment Assignment Logic](./figs/Ready_Port_Selector.png)  
    Based on the example diagram:
    - `entry_request_valid`:
        - `Entry 0`: `1` (Entry 0 is waiting)    -> `111`
        - `Entry 1`: `0` (Entry 1 is not waiting)  -> `000`
        - `Entry 2`: `0` (Entry 2 is not waiting)  -> `000`
        - `Entry 3`: `1` (Entry 3 is waiting)    -> `111`
    - `entry_port_valid`:
        - `Entry 0`: `010` (Port 1)
        - `Entry 1`: `000` (Port 0)
        - `Entry 2`: `010` (Port 1)
        - `Entry 3`: `100` (Port 2)
    - Bitwise AND operation
        - `Entry 0`: `111` AND `010` = `010`
        - `Entry 1`: `000` AND `000` = `000`
        - `Entry 2`: `000` AND `010` = `000`
        - `Entry 3`: `111` AND `100` = `100`
        
        > `entry_port_request`: It now only contains one-hot vectors for entries that are both allocated and waiting for a payload.


5. **Handshake Logic: Managing port readiness and masking the port assigned with invalid ports**
    ![PTQ_Handshake](./figs/PTQ_Handshake.png)  
    This block is responsible for the `valid/ready` handshake protocol with the `Access Ports`. It performs two functions: providing back-pressure to the ports and identifying all currently active memory requests for the arbiter.  
    Based on the example diagram:
    - **Back-pressure control**: First, the block determines which ports are `ready`.
        - From the `Entry-Port Assignment Masking Logic` block, we know that `Entry 0` and `Entry 3` are waiting for an address from `Port 1` and `Port 2` respectively.
        - Therefore, it asserts `port_ready_o` to `1` for both `Port 1` and `Port 2`.
        - No entry is waiting for `Port 0`, so its ready signal is `0`.
    - **Active request filtering**: The block checks which ports are handshaked. The `Access Ports` table shows `port_valid_i` is `1` for both `Port 1` and `Port 2`. Since the waiting entries (`Entry 0` and `Entry 3`) correspond to the valid ports (`Port 1` and `Port 2`), both are considered active and are passed to the `Arbitration Logic`.
        

6. **Arbitration Logic: Selecting the oldest active entry**  
    ![PTQ_masking](./figs/PTQ_masking.png)  
    This block is responsible for selecting the oldest active memory request for each port and generating the write enable signal for such requests.  
    Based on the example diagram:
    - The `Handshake Logic` has identified two active requests: one for `Entry 0` from `Port 1` and another for `Entry 3` from `Port 2`.
    - The CyclicPriorityMasking algorithm operates independently on each port's request list.
        - For `Port 1`, the only active request is from `Entry 0` (`1000`, 1st column of `entry_port_and`). With no other competitors for this port, `Entry 0` is selected as the winner for `Port 1`.
        - For `Port 2`, the only active request is from `Entry 3` (`0001`, 0th column of `entry_port_and`). Similarly, it is selected as the winner for `Port 2`.
    
    As a result, the `entry_wen_o` signal is asserted for both `Entry 0` and `Entry 3`, allowing two writes to proceed in parallel in the same clock cycle.

    To illustrate this process, let's assume the `entry_port_and` matrix, which represents all "live" requests, is as follows:

                 P2 P1 P0
        E0:    [ 0, 1, 0 ]
        E1:    [ 1, 0, 0 ]
        E2:    [ 0, 0, 0 ]
        E3:    [ 1, 0, 0 ]

    * **Priority Determination**: The `queue_head_oh_i` signal indicates that the head of the queue is at **Entry 2**. This establishes a priority order of **`2 -> 3 -> 0 -> 1`** for all arbitrations in this cycle.

    The `CyclicPriorityMasking` algorithm is then applied independently to each port's column of requests:

    * **For Port 2** (leftmost column `[0, 1, 0, 1]`): The active requests are from **Entry 1** and **Entry 3**. According to the priority order (`...3 -> 0 -> 1`), Entry 3 is older (has higher priority) than Entry 1. Therefore, **Entry 3** wins the arbitration for Port 2.

    * **For Port 1** (middle column `[1, 0, 0, 0]`): The only active request is from **Entry 0**. With no other competitors for this port, **Entry 0** is automatically selected as the winner for Port 1.

    * **For Port 0** (rightmost column `[0, 0, 0, 0]`): There are no active requests, so there is no winner.

    After the priority masking is complete, the resulting `entry_port_hs` matrix, which indicates the winners, becomes:

                 P2 P1 P0
        E0:    [ 0, 1, 0 ]  // Winner for Port 1
        E1:    [ 0, 0, 0 ]
        E2:    [ 0, 0, 0 ]
        E3:    [ 1, 0, 0 ]  // Winner for Port 2
    

