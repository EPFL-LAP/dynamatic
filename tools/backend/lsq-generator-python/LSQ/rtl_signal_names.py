from LSQ.utils import QueueType, QueuePointerType

GROUP_INIT_CHANNEL_NAME = "group_init"
"""
RTL name for the channel from the dataflow circuit representing a request to allocate a group of memory accesses.
"""


def QUEUE_POINTER_NAME(
        queue_type : QueueType, 
        queue_pointer_type : QueuePointerType
        ):
    """
    RTL name for the pointer to the (head/tail) entry of the (load/store) queue.
    """

    return f"{queue_type.value}_queue_{queue_pointer_type.value}"

LOAD_QUEUE_HEAD_POINTER_NAME = "ldq_head"
"""
RTL name for the pointer to the head entry of the load queue.
"""

LOAD_QUEUE_IS_EMPTY_NAME = "ldq_empty"
"""
RTL name for the isEmpty? signal from the load queue.
"""

STORE_QUEUE_TAIL_POINTER_NAME = "stq_tail"
"""
RTL name for the pointer to the tail entry of the load queue.
"""

STORE_QUEUE_HEAD_POINTER_NAME = "stq_head"
"""
RTL name for the pointer to the head entry of the load queue.
"""

STORE_QUEUE_IS_EMPTY_NAME = "stq_empty"
"""
RTL name for the isEmpty? signal from the load queue.
"""


LOAD_QUEUE_WRITE_ENABLE_NAME = "ldq_wen"
"""
RTL name for the write enables signals of the load queue.
"""

NUM_NEW_LOAD_QUEUE_ENTRIES_NAME = "num_loads"
"""
RTL name for the "number of new load queue entries" signal. Output by the group allocator, and used by the load queue to update its tail pointer.
"""

LOAD_PORT_INDEX_PER_LOAD_QUEUE_NAME = "ldq_port_idx"
"""
RTL name for index to a load port, per load queue entry.
"""

STORE_QUEUE_WRITE_ENABLE_NAME = "stq_wen"
"""
RTL name for the write enables signals of the store queue.
"""

NUM_NEW_STORE_QUEUE_ENTRIES_NAME = "num_stores"
"""
RTL name for the "number of new store queue entries" signal. Output by the group allocator, and used by the store queue to update its tail pointer.
"""

STORE_PORT_INDEX_PER_STORE_QUEUE_NAME = "stq_port_idx"
"""
RTL name for index to a store port, per store queue entry. 
"""

STORE_POSITION_PER_LOAD_NAME = "ga_ls_order"
"""
RTL name for signals which identify whether each of the stores precedes a load.
There is one of these signals per load queue entry, and 1 bit per store queue entry.
"""