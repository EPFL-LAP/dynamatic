from LSQ.entity import SignalSize, EntitySignalType
from LSQ.config import Config

from LSQ.rtl_signal_names import *

class GroupAllocatorDeclarativeSignals():
    class Reset():
        """
        Generic RTL reset signal
        """
        def __init__(self):
            # Reset is a single 1-bit signal
            self.signal_size = SignalSize(bitwidth=1, number=1)

            self.rtl_name = "rst"
            
            self.direction = EntitySignalType.INPUT
    
    class Clock():
        def __init__(self):
            self.description = "Generic RTL clock signal"

            # Clock is a single 1-bit signal
            self.signal_size = SignalSize(bitwidth=1, number=1)

            self.rtl_name = "clk"
            
            self.direction = EntitySignalType.INPUT

    class GroupInitValid():
        """
        1-bit valid signals for the "group init" channels, from the dataflow circuit. For N groups, there are N "group init" channels, which results in 

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

    class GroupInitReady():
        """
        1-bit ready signals for the "group init" channels, from the dataflow circuit. For N groups, there are N "group init" channels, which results in

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

    class LoadQueueTailPointer():
        """
        Pointer to the tail entry of the load queue, which is an input to the group allocator directly from the load queue.
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

    class LoadQueueHeadPointer():
        """
        Pointer to the head entry of the load queue, which is an input to the group allocator directly from the load queue.
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

    class LoadQueueIsEmpty():
        """
        isEmpty? signal from the load queue. There is a single, 1-bit isEmpty? signal, which is an input directly from the load queue.
        """
        def __init__(self):

            # There is a single 1-bit isEmpty? signal
            self.signal_size = SignalSize(
                                bitwidth=1, 
                                number=1
                                )

            self.rtl_name = LOAD_QUEUE_IS_EMPTY_NAME
            
            self.direction = EntitySignalType.INPUT