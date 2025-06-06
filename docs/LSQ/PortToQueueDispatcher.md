# Port-to-Queue Dispatcher

Detailed documentation for the Port-to-Queue dispatcher generator, which emits a VHDL entity and architecture to route incoming data and address from access ports into LSQ entries.

![Port-to-Queue Dispatcher](./figs/port_to_queue.png)
> **Figure:** Port-to-Queue Dispatcher

## Interface Signals

| Signal Name          | type                | Description     |
| -------------------- | ------------------- | --------------- |
| `port_bits_i`        | `LogicVecArray`     | Payload bus for each access port. It carries address when generating the load address dispatcher and the store address dispatcher. It carries data when generating the store data dispatcher.|
| `port_valid_i`       | `LogicArray`        | Valid signal for each input port (Valid address/data) |
| `port_ready_o`       | `LogicArray`        | Ready signal indicating LSQ is ready to receive address/data |
| `entry_valid_i`      | `LogicArray`        | Indicates if a queue entry is valid  |
| `entry_bits_valid_i` | `LogicArray`        | Valid bit for the contents of a queue entry|
| `entry_port_idx_i`   | `LogicVecArray`     | Indicates to which port the entry is assigned|
| `entry_bits_o`       | `LogicVecArray`     | Output bits written to the entry   |
| `entry_wen_o`        | `LogicArray`        | Write enable for each entry   |
| `queue_head_oh_i`    | `LogicVec`          | One-hot vector indicating the head entry in LSQ |

## Operational Summary
1. **Convert entry port index to one-hot**  
    For each entry, convert `entry_port_idx_i` into one-hot expression `entry_port_valid`.
2. **Multiplex payload**  
    Select from `port_bits_i` into `entry_bits_o`. 
3. **Choose entry with valid requests**  
    `entry_request_valid = entry_valid_i ∧ ¬entry_bits_valid_i`.   
4. **Filter out ports with invalid entry**  
    `entry_port_request = entry_port_valid ∧ entry_request_valid`.  
5. **Port handshake ready signal**  
    OR columns of `entry_port_request` → `port_ready_o`.  
6. **Enable the oldest entry using priority masking**  
    Filter by `port_valid_i`, rotate by `queue_head_oh_i`, pick the oldest one and enable it. `entry_wen_o`.
