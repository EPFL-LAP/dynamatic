# Port-to-Queue Dispatcher

A friendly tour of how addresses and data enter the Load-Store Queue



## 1. Overview and Purpose  

The Port-to-Queue Dispatcher is a submodule within the Load-Store Queue (LSQ) responsible for routing incoming memory requests (addresses or data) from the dataflow circuit's access ports to the correct entries within the LSQ. Those requests must land in the LSQ, a circular buffer that tracks every memory operation. It ensures each LSQ entry gets the correct address or data from the appropriate port.  

We need a total of three **Port-to-Queue Dispatchers**—one each for the load address, store address, and store data. Why? To load, you must first supply the address where the data is stored. Likewise, a store operation needs both the value to write and the address to write it at.  

In the LSQ architecture, memory operations from the main program arrive at dedicated access ports. Because multiple ports can try to send data simultaneously, a mechanism is needed to arbitrate these requests and write them into the LSQ.

## 2. Key Responsibilities  

The Port-to-Queue Dispatcher has the following responsibilities:
1. **Matching**: It identifies which queue entries are waiting for an argument from which specific access ports. An entry is considered to be waiting if it has been allocated by the Group Allocator, but is still missing its corresponding argument, meaning its address or data slot has not yet been filled.

2. **Handshaking & Back-pressure**: It manages the valid/ready handshake with each access port by asserting the port ready signal. Back-pressure is applied by de-asserting this ready signal. This occurs specifically when there are no allocated queue entries currently waiting for an argument from that port.

3. **Arbitration & Selection**: When multiple ports send valid data simultaneously, the dispatcher arbitrates among them. It uses a Cyclic Priority Masker to select the oldest waiting entry, relative to the queue's head pointer. This selection respects the order in which requests were enqueued.

4. **Write**: After arbitration, it writes the selected payload into the correct LSQ entry.


![Port-to-Queue Dispatcher](./figs/queue_to_port_v2.png)


## 3. Dataflow Walkthrough

1. **Identifying which LSQ slots are empty**  
The first job is to figure out which entries in the LSQ are actually waiting for an argument. The dispatcher scans the queue and identifies all entries that have been allocated (`entry_valid_i=1`) but have not yet received their required address or data (`entry_bits_valid_i=0`). This logic is captured by the expression `entry_request_valid = entry_valid_i AND (NOT entry_bits_valid_i)`, which creates a list of entries that need attention.


2. **Signaling readiness to the ports**  
Once the waiting entries are known, the dispatcher matches them to their designated access ports. It first decodes each entry's `entry_port_idx_i` into a one-hot signal, `entry_port_valid`, to create a clear mapping. This map is then filtered by the `entry_request_valid` list to generate the final `entry_port_request` matrix. A `'1'` in this matrix at `(row_e, column_p)` means "entry `e` is waiting for data from port `p`."  
To complete the handshake, the dispatcher asserts the `port_ready_o` signal back to an access port if any entry is waiting for it (i.e., if its column in the `entry_port_request` matrix has at least one `'1'`). This applies back-pressure if no entry is ready to receive the port's address or data.


3. **Arbitrating between competing ports**  
Multiple ports may send valid data simultaneously, but the LSQ can only write one item (more precisely, one for Load Queue and the other one for Store Queue) at a time. To resolve this contention, the dispatcher first filters the `entry_port_request` matrix by checking which ports are sending valid data this cycle (`port_valid_i = 1`). The result of this filtering is the `entry_port_and` matrix, which represents all currently active and valid requests.  
From this set of valid requests, the dispatcher must select the one destined for the **oldest** entry in the queue. This is handled by cyclic priority masking, which uses the queue's current head pointer (`queue_head_oh_i`) to find the highest-priority (oldest) request. The output, `entry_port_hs`, is a matrix with at most one `1`, indicating the single oldest request.


4. **Committing the data to the winning entry**  
In the final step, the dispatcher generates the one-hot `entry_wen_o` signal by reducing each row of the `entry_port_hs` matrix. The single `'1'` in this `entry_wen_o` array enables the write for the specific winning entry. This signal acts as the final trigger, allowing the address or data from the winning port to be written into the correct field of the selected LSQ entry and completing the process for that cycle.

<!--
1. **Convert entry port index to one-hot**   
Each LSQ entry already stores the port index it ultimately belongs to. The dispatcher converts those indices into a one-hot matrix `entry_port_valid` so it can see, column-wise, which entries are waiting for which port.  

2. **Multiplex payload**  
Address or data to be dispatched from the port to the queue comes from `port_bits_i`. Guided by `entry_port_valid`, the dispatcher holds each port's payload until the write-enable signal `entry_wen_o` is high.

3. **Choose entry with valid requests**   
Next, an LSQ entry only needs attention if it is valid **and** its addr/data slot is still empty. That gating yields `entry_request_valid`.  

4. **Filter out ports with invalid entry**  
From the above `entry_request_valid`, discard LSQ entries that cannot accept the current port ID. Among the surviving candidates, pick those port ID matches `entry_port_valid`.

5. **Port handshake ready signal**   
If **any** entry in column **_p_** is requesting service, port **_p_** gets `ready = 1`, completing the valid-ready handshake. 

6. **Enable the oldest entry using priority masking**  
Even when an LSQ entry is ready, the originating port might not assert valid (`port_valid_i`). We therefore filter the candidate set with `port_valid_i` before the priority masking. This leaves the final candidate set.
Finally, we rotate the request matrix by the queue-head pointer and grant the oldest remaining entry via priority masking, setting `entry_wen_o=1` for such slot.
 -->


## 4. Interface Signals

| Signal Name          | Description     |
| -------------------- |  --------------- |
| `port_bits_i[p]`        | _“Here is my 16-bit payload.”_ (address or data)  |
| `port_valid_i[p]`       | _“…and I really mean it.”_  High when the payload is ready.   |
| `port_ready_o[p]`       | Dispatcher replies: _“Sure, send it!”_  Goes high if the LSQ can take the request this cycle.  |
| `entry_valid_i[e]`      | Is LSQ entry **_e_** logically allocated?   |
| `entry_bits_valid_i[e]` | Has the addr/data slot already been filled? |
| `entry_port_idx_i[e]`   | Indicates to which port the entry is assigned|
| `entry_bits_o[e]`       | The data actually written into LSQ entry **_e_**. Think of it as the ink flowing into row **_e_** on the whiteboard.|
| `entry_wen_o[e]`        | A short pulse that says _“commit the write into entry **_e_** now.”_ |
| `queue_head_oh_i[e]`    | One-hot vector indicating the head entry in LSQ |
