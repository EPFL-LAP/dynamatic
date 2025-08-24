from LSQ.entity import SignalSize, EntitySignalType
from LSQ.config import Config

from LSQ.rtl_signal_names import *

class GroupAllocatorDeclarativeSignals():
    class Reset():
        """
        Input: generic RTL reset signal
        """
        def __init__(self):
            # Reset is a single 1-bit signal
            self.signal_size = SignalSize(bitwidth=1, number=1)

            self.rtl_name = "rst"
            
            self.direction = EntitySignalType.INPUT

            self.comment = None
    
    class Clock():
        """
        Input: generic RTL clock signal
        """
         
        def __init__(self):
            # Clock is a single 1-bit signal
            self.signal_size = SignalSize(bitwidth=1, number=1)

            self.rtl_name = "clk"
            
            self.direction = EntitySignalType.INPUT

            self.comment = None

    class GroupInitValid():
        """
        Input: 1-bit valid signals for the "group init" channels, from the dataflow circuit. For N groups, there are N "group init" channels, which results in 

        group_init_valid_0_i : in std_logic;
        group_init_ready_1_i : in std_logic;
        .
        .
        .
        group_init_ready_N_i : in std_logic;
        """
        def __init__(self, config : Config):

            # There are N 1-bit group init valid signals

            self.signal_size = SignalSize(
                                bitwidth=1, 
                                number=config.num_groups()
                                )

            self.rtl_name = f"{GROUP_INIT_CHANNEL_NAME}_valid"
            
            self.direction = EntitySignalType.INPUT

            self.comment = f"""

    -- Group init signals from the dataflow circuit
    -- {config.num_groups()} signals, one for each group of memory operations.
""".removeprefix("\n")

    class GroupInitReady():
        """
        Input: 1-bit ready signals for the "group init" channels, from the dataflow circuit. For N groups, there are N "group init" channels, which results in

        group_init_ready_0_i : in std_logic;
        group_init_ready_1_i : in std_logic;
        .
        .
        .
        group_init_ready_N_i : in std_logic;
        """
        def __init__(self, config : Config):

            # There are N 1-bit group init ready signals
            self.signal_size = SignalSize(
                                bitwidth=1, 
                                number=config.num_groups()
                                )

            self.rtl_name = f"{GROUP_INIT_CHANNEL_NAME}_ready"
            
            self.direction = EntitySignalType.INPUT

            self.comment = None

    class LoadQueueTailPointer():
        """
        Input: pointer to the tail entry of the load queue, which is an input to the group allocator directly from the load queue.
        There is only 1 load queue tail pointer. Like all queue pointers, its bitwidth is equal to ceil(log2(num_queue_entries))
        """
        def __init__(self, config : Config):

            # The load queue tail pointer is a single, N-bit signal.
            self.signal_size = SignalSize(
                                bitwidth=config.ldq_idx_w(), 
                                number=1
                                )

            self.rtl_name = LOAD_QUEUE_TAIL_POINTER_NAME
            
            self.direction = EntitySignalType.INPUT

            self.comment = f"""

    -- Input signals from the load queue
""".removeprefix("\n")

    class LoadQueueHeadPointer():
        """
        Input: pointer to the head entry of the load queue, which is an input to the group allocator directly from the load queue.
        There is only 1 load queue head pointer. Like all queue pointers, its bitwidth is equal to ceil(log2(num_queue_entries))
        """
        def __init__(self, config : Config):

            # The load queue head pointer is a single, N-bit signal.
            self.signal_size = SignalSize(
                                bitwidth=config.ldq_idx_w(), 
                                number=1
                                )

            self.rtl_name = LOAD_QUEUE_HEAD_POINTER_NAME
            
            self.direction = EntitySignalType.INPUT

            self.comment = None

    class LoadQueueIsEmpty():
        """
        Input: isEmpty? signal from the load queue. There is a single, 1-bit isEmpty? signal, which is an input directly from the load queue.
        """
        def __init__(self):

            # There is a single 1-bit isEmpty? signal
            self.signal_size = SignalSize(
                                bitwidth=1, 
                                number=1
                                )

            self.rtl_name = LOAD_QUEUE_IS_EMPTY_NAME
            
            self.direction = EntitySignalType.INPUT

            self.comment = None

    class StoreQueueTailPointer():
        """
        Input: pointer to the tail entry of the store queue, which is an input to the group allocator directly from the store queue.
        There is only 1 store queue tail pointer. Like all queue pointers, its bitwidth is equal to ceil(log2(num_queue_entries))
        """
        def __init__(self, config : Config):

            # The store queue tail pointer is a single, N-bit signal.
            self.signal_size = SignalSize(
                                bitwidth=config.stq_idx_w(), 
                                number=1
                                )

            self.rtl_name = STORE_QUEUE_TAIL_POINTER_NAME
            
            self.direction = EntitySignalType.INPUT


            self.comment = f"""

    -- Input signals from the store queue
""".removeprefix("\n")

    class StoreQueueHeadPointer():
        """
        Input: pointer to the head entry of the store queue, which is an input to the group allocator directly from the store queue.
        There is only 1 store queue head pointer. Like all queue pointers, its bitwidth is equal to ceil(log2(num_queue_entries))
        """
        def __init__(self, config : Config):

            # The store queue tail pointer is a single, N-bit signal.
            self.signal_size = SignalSize(
                                bitwidth=config.stq_idx_w(), 
                                number=1
                                )

            self.rtl_name = STORE_QUEUE_HEAD_POINTER_NAME
            
            self.direction = EntitySignalType.INPUT

            self.comment = None


    class StoreQueueIsEmpty():
        """
        Input: isEmpty? signal from the store queue. There is a single, 1-bit isEmpty? signal, which is an input directly from the store queue.
        """
        def __init__(self):

            # There is a single 1-bit isEmpty? signal
            self.signal_size = SignalSize(
                                bitwidth=1, 
                                number=1
                                )

            self.rtl_name = STORE_QUEUE_IS_EMPTY_NAME
            
            self.direction = EntitySignalType.INPUT

            self.comment = None

    class LoadQueueWriteEnable():
        """
        Output: Write enable signals to the load queue, used to allocate entries in the load queue. There are N 1-bit write enable signals, which are an output directly from the store queue. As expected for write enable signals to queue entries, there is 1 write enable signal per queue entry.
        """
        def __init__(self, config : Config):

            # There are N 1-bit write enable signals.
            # As expected for write enable signals to queue entries, there is 1 write enable signal per queue entry.
            self.signal_size = SignalSize(
                                bitwidth=1, 
                                number=config.ldq_num_entries()
                                )

            self.rtl_name = LOAD_QUEUE_WRITE_ENABLE_NAME
            
            self.direction = EntitySignalType.OUTPUT

            self.comment = f"""

    -- Load queue write enable signals
    -- {config.ldq_num_entries()} signals, one for each queue entry.
""".removeprefix("\n")

    class NumNewLoadQueueEntries():
        """
        Output: Number of load queue entries to allocate, which is output directly to the load queue.
        Non-handshaked signal. Used by the load queue to update its tail pointer, using update logic appropriate to circular buffers.
        There is a single "number of load queue entries to allocate" signal, and its bitwidth is equal to the bitwidth of the load queue pointers, to allow easy arithmetic between then.
        """
        def __init__(self, config : Config):

            # There is a single N-bit "number of load queue entries to allocate" signal
            # and its bitwidth is equal to the bitwidth of the load queue pointers, to allow easy arithmetic between then.
            self.signal_size = SignalSize(
                                bitwidth=config.ldq_idx_w(), 
                                number=1
                                )

            self.rtl_name = NUM_NEW_LOAD_QUEUE_ENTRIES_NAME
            
            self.direction = EntitySignalType.OUTPUT
            
            self.comment = f"""

    -- Number of new load queue entries to allocate.
    -- Used by the load queue to update its tail pointer.
    -- Bitwidth equal to the load queue pointer bitwidth.
""".removeprefix("\n")

    class LoadPortIndexPerLoadQueueEntry():
        """
        Output: Which load port index to allocate into each load queue entry. The group allocator uses the head pointer from the load queue to place the load port indices in the correct signal, so that they arrive in the correct load queue entries. This is guarded by the load queue entry write enable, so not all of these signals are used.

        If there is only a single load port into the LSQ, this signal is not present.
        """
        def __init__(self, config : Config):

            # There is 1 N-bit isEmpty? signal
            self.signal_size = SignalSize(
                                bitwidth=config.ldpAddrW, 
                                number=config.ldq_num_entries()
                                )

            self.comment = f"""

    -- Load port index to write into each load queue entry.
    -- {config.ldq_num_entries()} signals, each {config.ldpAddrW} bit(s).
    -- This signal is not present if there is a single load port into the LSQ.
""".removeprefix("\n")

            self.rtl_name = LOAD_PORT_INDEX_PER_LOAD_QUEUE_NAME
            
            self.direction = EntitySignalType.OUTPUT