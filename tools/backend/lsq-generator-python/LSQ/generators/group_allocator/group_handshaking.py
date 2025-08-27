from LSQ.config import Config
from LSQ.entity import Entity, EntitySignalType, SignalSize
from LSQ.rtl_signal_names import *

class GroupHandshakingDeclarativeIOSignals():
    class LoadQueueTailPointer():
        """
        Input: Pointer to the tail entry of the load queue, passed directly through the group allocator
        to the GroupHandshaking unit.
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

    -- The group allocator only allocates a group
    -- if there are sufficient empty entries in each queue.
    -- To calculate this requires the queue pointers.
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

            self.entity_comment = None

class GroupHandshakingDeclarative():
     def __init__(self, config : Config):
          
        io = GroupHandshakingDeclarativeIOSignals()
        self.io_signals = [
                io.LoadQueueTailPointer(config),
                io.LoadQueueHeadPointer(config),
                io.StoreQueueTailPointer(config),
                io.StoreQueueHeadPointer(config),
                io.GroupInitValid(config),
                io.GroupInitReady(config)
          ]

class GroupHandshaking():
        def __init__(self, config : Config):

            group_handshaking_declarative = Empty()

            group_handshaking_declarative.io_signals = [

            ]


            entity = Entity(group_handshaking_declarative)

            self.statement = entity.instantiate(
                "GroupHandshaking",
                "GroupHandshaking"
            )