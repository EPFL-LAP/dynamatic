"""
Declarative signal definitions, 
defining their names, numbers, and bitwidths in a single place.
Also provides docstrings for each signal to help remember what it does.
"""

from LSQ.entity import Signal
from LSQ.config import Config
from LSQ.utils import QueueType, QueuePointerType

from LSQ.rtl_signal_names import *


class Reset(Signal):
    """
    Input.

    Generic RTL reset signal

    Bitwidth=1, Number=1
    """
    def __init__(self):
        Signal.__init__(
            self,
            base_name="rst",
            direction=Signal.Direction.INPUT,
            size=Signal.Size(
                bitwidth=1,
                number=1
            )
        )


class Clock(Signal):
    """
    Input.

    Generic RTL clock signal

    Bitwidth=1, Number=1
    """
    def __init__(self):
        Signal.__init__(
            self,
            base_name="clk",
            direction=Signal.Direction.INPUT,
            size=Signal.Size(
                bitwidth=1,
                number=1
            )
        )

class GroupInitValid(Signal):
    """
    Input to the Group Allocator,
    from the dataflow circuit.

    Bitwidth = 1, Number = N

    1-bit valid signals for the "group init" channels, from the dataflow circuit. 
    For N groups, there are N "group init" channels, which results in 

    group_init_valid_0_i : in std_logic;
    group_init_ready_1_i : in std_logic;
    .
    .
    .
    group_init_ready_N_i : in std_logic;
    """
    def __init__(self, config : Config):
        Signal.__init__(
            self,
            base_name=f"{GROUP_INIT_CHANNEL_NAME}_valid",
            direction=Signal.Direction.INPUT,
            size=Signal.Size(
                bitwidth=1,
                number=config.num_groups()
            ),
            always_number=True
        )
    

class GroupInitReady(Signal):
    """
    Input of the Group Allocator,
    to the dataflow circuit.
        
    Bitwidth = 1, Number = N

    1-bit ready signals for the "group init" channels, from the dataflow circuit. 
    For N groups, there are N "group init" channels, which results in

    group_init_ready_0_i : out std_logic;
    group_init_ready_1_i : out std_logic;
    .
    .
    .
    group_init_ready_N_i : out std_logic;
    """
    def __init__(self, config : Config):
        Signal.__init__(
            self,
            base_name=f"{GROUP_INIT_CHANNEL_NAME}_ready",
            direction=Signal.Direction.OUTPUT,
            size=Signal.Size(
                bitwidth=1,
                number=config.num_groups()
            ),
            always_number=True
        )



class NaiveStoreOrderPerEntry(Signal):
    """        
    Output of Group Allocator, Input to Y.

    Bitwidth = N, Number = N

    Whether the stores in the store queue and ahead or behind
    each specific entry in the load queue.
        
    There is one signal per entry in the load queue,
    and 1 bit per entry in the store queue.
    
    The order of the memory operations, read from the ROM,
    has been shifted to generate this.
    
    This is done based on the store queue and load queue pointers.

    It is naive, however, as 1s for already allocated stores are not present.
    """

    def __init__(self, 
                    config : Config,
                    direction : Signal.Direction
                    ):

        Signal.__init__(
            self,
            base_name=NAIVE_STORE_ORDER_PER_ENTRY_NAME,
            direction=direction,
            size=Signal.Size(
                bitwidth=config.store_queue_num_entries(), 
                number=config.load_queue_num_entries()
            )
        )


class PortIdxPerEntry(Signal):
    """
    Output of Group Allocator,
    input to either Load Queue or Store Queue
    
    Bitwidth = N, Number = M

    Which (load/store) port index to allocate into each (load/store) queue entry. 
    
    The group allocator uses the head pointer from the (load/store) queue 
    to place the (load/store) port indices in the correct signal, 
    so that they arrive in the correct (load/store) queue entries. 
    
    This is guarded by the (load/store) queue entry write enable, 
    so not all of these signals are used.

    There is one signal per load queue entry, with the bitwidth required to identify a load port.
    Not one-hot.

    Absent is there is only 1 (load/store) port.
    """
    def __init__(self, 
                    config : Config,
                    queue_type : QueueType,
                    direction : Signal.Direction
                    ):
        Signal.__init__(
            self,
            base_name=PORT_INDEX_PER_ENTRY_NAME(queue_type),
            direction=direction,
            size=Signal.Size(
                bitwidth=config.ports_idx_bitwidth(queue_type),
                number=config.queue_num_entries(queue_type)
            ),
            always_vector=True
        )

class NumNewQueueEntries(Signal):
    """       
    Output of Group Allocator,
    input to either Load Queue or Store Queue.

    Internally used by the Group Allocator to generate
    write enable signals, also sent to the Load Queue or
    Store Queue.

    Bitwidth = N

    Number = 1

    Number of new (load/store) queue entries to allocate,
    based on which group is currently being allocated.
    
    Used by the (load/store) queue to update its tail pointer, 
    using update logic appropriate to circular buffers.
    
    There is a single "number of (load/store) queue entries to allocate" signal,
    and its bitwidth is equal to the bitwidth of the (load/store) queue pointers, 
    to allow easy arithmetic between then.
    """
    def __init__(self, 
                    config : Config,
                    queue_type : QueueType,
                    # used as local signal in group allocator
                    # where it does not have direction
                    direction : Signal.Direction = None,
                    masked : bool = False
                    ):
        Signal.__init__(
            self,
            base_name=NUM_NEW_ENTRIES_NAME(queue_type, masked),
            direction=direction,
            size=Signal.Size(
                bitwidth=config.queue_idx_bitwidth(queue_type),
                number=1
            )
        )

class QueueWriteEnable(Signal):
    """
    Output of the Group Allocator, 
    input to either the Load Queue or the Store Queue.

    Bitwidth = 1

    Number = N

    Write enable signals to the (load/store) queue, 
    used to allocate entries in the (load/store) queue. 
    
    There are N 1-bit write enable signals.
    
    As expected for write enable signals to queue entries, 
    there is 1 write enable signal per queue entry.
    """
    def __init__(self, 
                config : Config,
                queue_type : QueueType,
                direction : Signal.Direction
                ):

        Signal.__init__(
            self,
            base_name=WRITE_ENABLE_NAME(queue_type),
            direction=direction,
            size=Signal.Size(
                bitwidth=1,
                number=config.queue_num_entries(queue_type)
            )
        )

class QueuePointer(Signal):
    """
    Output of the Load Queue or Store Queue,
    input to the Group Allocator

    Bitwidth = N, Number = 1

    Pointer to the (head/tail) entry of a queue.
    There is only 1 queue (head/tail) pointer. 
    Like all queue pointers, its bitwidth is equal to ceil(log2(num_queue_entries))
    """
    def __init__(self, 
                    config : Config,
                    queue_type : QueueType,
                    queue_pointer_type : QueuePointerType,
                    direction : Signal.Direction
                    ):
        Signal.__init__(
            self,
            base_name=QUEUE_POINTER_NAME(queue_type, queue_pointer_type),
            direction=direction,
            size=Signal.Size(
                bitwidth=config.queue_idx_bitwidth(queue_type),
                number=1
            )
        )



class QueueIsEmpty(Signal):
    """
    Output of the Load Queue or Store Queue,
    input to the Group Allocator.

    Bitwidth = 1, Number = 1

    isEmpty? signal for the (load/store) queue
    """
    def __init__(self, 
                    queue_type : QueueType,
                    direction = Signal.Direction
                ):
        Signal.__init__(
            self,
            base_name=IS_EMPTY_NAME(queue_type),
            direction=direction,
            size=Signal.Size(
                bitwidth=1,
                number=1
            )
        )

 

class GroupInitTransfer(Signal):
    """
    Local signal of the Group Allocator.

    Output of its Group Handshaking Unit.
    Input to 
    the Number of New Queue Entries Unit,
    the Port Index per Queue Entry Unit
    and the Naive Store Order per Load Queue Entry Unit,
    
    Bitwidth = 1

    Number = N

    Whether a particular group init channel transfers this cycle.
        
    1-bit signal, 1 signal per group of memory accesses
    """

    def __init__(self, 
                    config : Config,
                    direction : Signal.Direction = None
                    ):

        Signal.__init__(
            self,
            base_name=GROUP_INIT_TRANSFER_NAME,
            direction=direction,
            size=Signal.Size(
                bitwidth=1,
                number=config.num_groups()
            ),
            always_number=True
        )
        