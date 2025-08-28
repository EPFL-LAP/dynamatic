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

    return f"{queue_type.value}_q_{queue_pointer_type.value}"


def IS_EMPTY_NAME(
        queue_type : QueueType, 
        ):
    """
    RTL name for the isEmpty? signal from the (load/store) queue.
    """

    return f"{queue_type.value}_empty"



def WRITE_ENABLE_NAME(
        queue_type : QueueType, 
        ):
    """
    RTL name for the write enables signals of the load queue.
    """

    return f"{queue_type.value}_write_en"

def NUM_NEW_QUEUE_ENTRIES_NAME(
        queue_type : QueueType, 
        ):
    """
    RTL name for the "number of new (load/store) queue entries" signal. 
    Output by the group allocator, and used by the load queue to update its tail pointer.
    """

    return f"num_new_{queue_type.value}_q_entries"

def PORT_INDEX_PER_ENTRY_NAME(
        queue_type : QueueType, 
        ):
    """
    RTL name for index to a (load/store) port, per (load/store) queue entry.
    """
    return f"{queue_type.value}_port_idx_per_entry"


STORE_POSITION_PER_LOAD_NAME = "ga_ls_order"
"""
RTL name for signals which identify whether each of the stores precedes a load.
There is one of these signals per load queue entry, and 1 bit per store queue entry.
"""

GROUP_INIT_TRANSFER_NAME = f"{GROUP_INIT_CHANNEL_NAME}_transfer"
"""
RTL name for signals which identify whether a 
particular group init channel is 'transfer' in this cycle.
There is one of these 1-bit signals per group of memory operations.
"""


def NUM_EMPTY_ENTRIES_NAIVE_NAME(
        queue_type : QueueType, 
        ):
    """
    RTL name for the number of empty queue entries.
    """

    return f"{queue_type.value}_q_num_empty_entries_naive"
