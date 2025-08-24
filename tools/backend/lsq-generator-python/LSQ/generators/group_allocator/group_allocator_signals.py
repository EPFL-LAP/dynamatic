from LSQ.entity import SignalSize, EntitySignalType, Entity
from LSQ.config import Config

from LSQ.rtl_signal_names import *

class GroupAllocatorDeclarativeIOSignals():
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

    -- Group init channels from the dataflow circuit
    -- {config.num_groups()} channels, one for each group of memory operations.
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
                                bitwidth=config.load_queue_idx_bitwidth(), 
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
                                bitwidth=config.load_queue_idx_bitwidth(), 
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
                                bitwidth=config.store_queue_idx_bitwidth(), 
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
                                bitwidth=config.store_queue_idx_bitwidth(), 
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
        Output: Write enable signals to the load queue, used to allocate entries in the load queue. There are N 1-bit write enable signals, which are an output directly to the load queue. As expected for write enable signals to queue entries, there is 1 write enable signal per queue entry.
        """
        def __init__(self, config : Config):

            # There are N 1-bit write enable signals.
            # As expected for write enable signals to queue entries, there is 1 write enable signal per queue entry.
            self.signal_size = SignalSize(
                                bitwidth=1, 
                                number=config.load_queue_num_entries()
                                )

            self.rtl_name = LOAD_QUEUE_WRITE_ENABLE_NAME
            
            self.direction = EntitySignalType.OUTPUT

            self.comment = f"""

    -- Load queue write enable signals
    -- {config.load_queue_num_entries()} signals, one for each queue entry.
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
                                bitwidth=config.load_queue_idx_bitwidth(), 
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

        There is one signal per load queue entry, and with the bitwidth required to identify a load port.
        Not one-hot.

        There is inconsistant code implying this signal should not be present 
        if there are no load ports.
        But it is currently added regardless (with bitwidth 1)
        """
        def __init__(self, config : Config):

            # There are N M-bit signals. One per load queue entry
            # and with the bitdwith required to identify a load port
            self.signal_size = SignalSize(
                                bitwidth=config.load_ports_idx_bitwidth(), 
                                number=config.load_queue_num_entries()
                                )

            self.comment = f"""

    -- Load port index to write into each load queue entry.
    -- {config.load_queue_num_entries()} signals, each {config.load_ports_idx_bitwidth()} bit(s).
    -- Not one-hot.
    -- There is inconsistant code implying this signal should not be present 
    -- if there are no load ports.
    -- But it is currently added regardless (with bitwidth 1)
    -- Actual number of load ports: {config.load_ports_num()}
""".removeprefix("\n")

            self.rtl_name = LOAD_PORT_INDEX_PER_LOAD_QUEUE_NAME
            
            self.direction = EntitySignalType.OUTPUT


    class StoreQueueWriteEnable():
        """
        Output: Write enable signals to the store queue, used to allocate entries in the store queue. There are N 1-bit write enable signals, which are an output directly to the store queue. As expected for write enable signals to queue entries, there is 1 write enable signal per queue entry.
        """
        def __init__(self, config : Config):

            # There are N 1-bit write enable signals.
            # As expected for write enable signals to queue entries, there is 1 write enable signal per queue entry.
            self.signal_size = SignalSize(
                                bitwidth=1, 
                                number=config.store_queue_num_entries()
                                )

            self.rtl_name = STORE_QUEUE_WRITE_ENABLE_NAME
            
            self.direction = EntitySignalType.OUTPUT

            self.comment = f"""

    -- Store queue write enable signals
    -- {config.load_queue_num_entries()} signals, one for each queue entry.
""".removeprefix("\n")

    class NumNewStoreQueueEntries():
        """
        Output: Number of store queue entries to allocate, which is output directly to the store queue.
        Non-handshaked signal. Used by the store queue to update its tail pointer, using update logic appropriate to circular buffers.
        There is a single "number of store queue entries to allocate" signal, and its bitwidth is equal to the bitwidth of the store queue pointers, to allow easy arithmetic between then.
        """
        def __init__(self, config : Config):

            # There is a single N-bit "number of store queue entries to allocate" signal
            # and its bitwidth is equal to the bitwidth of the store queue pointers, to allow easy arithmetic between then.
            self.signal_size = SignalSize(
                                bitwidth=config.store_queue_idx_bitwidth(), 
                                number=1
                                )

            self.rtl_name = NUM_NEW_STORE_QUEUE_ENTRIES_NAME
            
            self.direction = EntitySignalType.OUTPUT
            
            self.comment = f"""

    -- Number of new store queue entries to allocate.
    -- Used by the store queue to update its tail pointer.
    -- Bitwidth equal to the store queue pointer bitwidth.
""".removeprefix("\n")

    class StorePortIndexPerStoreQueueEntry():
        """
        Output: Which store port index to allocate into each store queue entry. The group allocator uses the head pointer from the store queue to place the store port indices in the correct signal, so that they arrive in the correct store queue entries. This is guarded by the store queue entry write enable, so not all of these signals are used.

        There is one signal per store queue entry, and with the bitwidth required to identify a store port.
        Not one-hot.

        There is inconsistant code implying this signal should not be present 
        if there are no store ports.
        But it is currently added regardless (with bitwidth 1)
        """
        def __init__(self, config : Config):

            # There are N M-bit signals. One per store queue entry
            # and with the bitdwidth required to identify a store port
            self.signal_size = SignalSize(
                                bitwidth=config.store_ports_idx_bitwidth(), 
                                number=config.store_queue_num_entries()
                                )

            self.comment = f"""

    -- Store port index to write into each store queue entry.
    -- {config.store_queue_num_entries()} signals, each {config.store_ports_idx_bitwidth()} bit(s).
    -- Not one-hot.
    -- There is inconsistant code implying this signal should not be present 
    -- if there are no store ports.
    -- But it is currently added regardless (with bitwidth 1)
    -- Actual number of store ports: {config.store_ports_num()}
""".removeprefix("\n")

            self.rtl_name = STORE_PORT_INDEX_PER_STORE_QUEUE_NAME
            
            self.direction = EntitySignalType.OUTPUT


    class StorePositionPerLoad():
        """
        Output: Whether the stores in the store queue and ahead or behind
        each specific entry in the load queue.
         
        There is one signal per entry in the load queue,
        and 1 bit per entry in the store queue.
        
        The order of the memory operations, read from the ROM,
        has been shifted to generate this, 
        as well as 0s and 1s added correctly to fill out each signal.

        This is done based on the store queue and load queue pointers.
        """
        def __init__(self, config : Config):

            # There are N M-bit signals. One per load queue entry
            # and 1 bit per store queue entry
            self.signal_size = SignalSize(
                                bitwidth=config.store_queue_num_entries(), 
                                number=config.load_queue_num_entries()
                                )

            self.comment = f"""

    -- Store position per load
    -- {config.load_queue_num_entries()} signals, each {config.store_queue_num_entries()} bit(s).
    -- One per entry in the load queue, with 1 bit per entry in the store queue.
    -- The order of the memory operations, read from the ROM, 
    -- has been shifted to generate this,
    -- as well as 0s and 1s added correctly to fill out each signal.
""".removeprefix("\n")

            self.rtl_name = STORE_POSITION_PER_LOAD_NAME
            
            self.direction = EntitySignalType.OUTPUT

class GroupAllocatorDeclarativeLocalSignals():
    class NumNewLoadQueueEntries():
        """
        Intermediate local signal that is the same as the NumNewLoadQueueEntries output.
        Used to make the value internally readable, to shift the load queue write enables.
        """
        def __init__(self, config : Config):

            # There is a single N-bit "number of load queue entries to allocate" signal
            # and its bitwidth is equal to the bitwidth of the load queue pointers, to allow easy arithmetic between then.
            self.signal_size = SignalSize(
                                bitwidth=config.load_queue_idx_bitwidth(), 
                                number=1
                                )

            self.rtl_name = NUM_NEW_LOAD_QUEUE_ENTRIES_NAME
                        
            self.comment = f"""

    -- Intermediate local signal that is the same as the NumNewLoadQueueEntries output.
    -- Used to make the value internally readable
    -- to shift the load queue write enables.
""".removeprefix("\n")
            
    class NumNewStoreQueueEntries():
        """
        Intermediate local signal that is the same as the NumNewStoreQueueEntries output.
        Used to make the value internally readable, to shift the store queue write enables.
        """
        def __init__(self, config : Config):

            # There is a single N-bit "number of store queue entries to allocate" signal
            # and its bitwidth is equal to the bitwidth of the store queue pointers, to allow easy arithmetic between then.
            self.signal_size = SignalSize(
                                bitwidth=config.store_queue_idx_bitwidth(), 
                                number=1
                                )

            self.rtl_name = NUM_NEW_STORE_QUEUE_ENTRIES_NAME
            
            self.comment = f"""

    -- Intermediate local signal that is the same as the NumNewStoreQueueEntries output.
    -- Used to make the value internally readable
    -- to shift the store queue write enables.
""".removeprefix("\n")

class Empty():
    pass

class GroupAllocatorDeclarativeBody():
    class GroupHandshaking():
        def __init__(self, config : Config):

            group_handshaking_declarative = Empty()

            io = GroupAllocatorDeclarativeIOSignals()
            group_handshaking_declarative.io_signals = [
                io.LoadQueueHeadPointer(config),
                io.LoadQueueTailPointer(config),
                io.StoreQueueHeadPointer(config),
                io.StoreQueueTailPointer(config),
                io.GroupInitValid(config),
                io.GroupInitReady(config)
            ]


            entity = Entity(group_handshaking_declarative)

            self.statement = entity.instantiate(
                "GroupHandshaking",
                "GroupHandshaking"
            )