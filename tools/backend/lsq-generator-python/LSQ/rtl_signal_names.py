from LSQ.utils import QueueType, QueuePointerType

GROUP_ALLOCATOR_NAME = "group_allocator"
"""
RTL name for the group allocator
"""

GROUP_INIT_CHANNEL_NAME = "group_init"
"""
RTL name for the channel from the dataflow circuit representing a request to allocate a group of memory accesses.
"""


# def QUEUE_POINTER_NAME(
#         queue_type : QueueType, 
#         queue_pointer_type : QueuePointerType
#         ):
#     """
#     RTL name for the pointer to the (head/tail) entry of the (load/store) queue.
#     """

#     return f"{queue_type.value}_q_{queue_pointer_type.value}"

def QUEUE_POINTER_NAME(
        queue_type : QueueType, 
        queue_pointer_type : QueuePointerType
        ):
    """
    RTL name for the pointer to the (head/tail) entry of the (load/store) queue.
    """
    match queue_type:
        case QueueType.LOAD:
            return f"ldq_{queue_pointer_type.value}"
        case QueueType.STORE:
            return f"stq_{queue_pointer_type.value}"



# def IS_EMPTY_NAME(
#         queue_type : QueueType, 
#         ):
#     """
#     RTL name for the isEmpty? signal from the (load/store) queue.
#     """

#     return f"{queue_type.value}_empty"

def IS_EMPTY_NAME(
        queue_type : QueueType, 
        ):
    """
    RTL name for the isEmpty? signal from the (load/store) queue.
    """
    match queue_type:
        case QueueType.LOAD:
            return f"ldq_empty"
        case QueueType.STORE:
            return f"stq_empty"



# def WRITE_ENABLE_NAME(
#         queue_type : QueueType, 
#         ):
#     """
#     RTL name for the write enables signals of the (load/store) queue.
#     """

#     return f"{queue_type.value}_write_en"

def WRITE_ENABLE_NAME(
        queue_type : QueueType, 
        ):
    """
    RTL name for the write enables signals of the (load/store) queue.
    """
    match queue_type:
        case QueueType.LOAD:
            return f"ldq_wen"
        case QueueType.STORE:
            return f"stq_wen"



# def NUM_NEW_QUEUE_ENTRIES_NAME(
#         queue_type : QueueType, 
#         ):
#     """
#     RTL name for the "number of new (load/store) queue entries" signal. 
#     Output by the group allocator, and used by the load queue to update its tail pointer.
#     """

#     return f"num_new_{queue_type.value}_q_entries"

def NUM_NEW_QUEUE_ENTRIES_NAME(
        queue_type : QueueType, 
        ):
    """
    RTL name for the "number of new (load/store) queue entries" signal. 
    Output by the group allocator, and used by the load queue to update its tail pointer.
    """

    match queue_type:
        case QueueType.LOAD:
            return f"num_loads"
        case QueueType.STORE:
            return f"num_stores"

# def PORT_INDEX_PER_ENTRY_NAME(
#         queue_type : QueueType, 
#         ):
#     """
#     RTL name for index to a (load/store) port, per (load/store) queue entry.
#     """
#     return f"{queue_type.value}_port_idx_per_entry"

def PORT_INDEX_PER_ENTRY_NAME(
        queue_type : QueueType, 
        ):
    """
    RTL name for index to a (load/store) port, per (load/store) queue entry.
    """

    match queue_type:
        case QueueType.LOAD:
            return f"ldq_port_idx"
        case QueueType.STORE:
            return f"stq_port_idx"



# NAIVE_STORE_ORDER_PER_ENTRY_NAME = "naive_store_order"
# """
# RTL name for signals which identify whether each of the stores precedes a load.
# There is one of these signals per load queue entry, and 1 bit per store queue entry.
# It is naive as it only considers loads and stores currently being allocated, not previous stores.
# """

NAIVE_STORE_ORDER_PER_ENTRY_NAME = "ga_ls_order"
"""
RTL name for signals which identify whether each of the stores precedes a load.
There is one of these signals per load queue entry, and 1 bit per store queue entry.
It is naive as it only considers loads and stores currently being allocated, not previous stores.
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

GROUP_HANDSHAKING_NAME = "group_handshaking"

def UNSHIFTED_PORT_INDEX_PER_ENTRY_NAME(
        queue_type : QueueType, 
        ):
    """
    RTL name for the unshifted port idx per queue entry
    """
    return f"unshifted_{PORT_INDEX_PER_ENTRY_NAME(queue_type)}"


UNSHIFTED_NAIVE_STORE_ORDER_PER_ENTRY_NAME = f"unshifted_{NAIVE_STORE_ORDER_PER_ENTRY_NAME}"
"""
RTL name for the unshifted naive store position per queue entry.
It is naive as it only considers loads and stores currently being allocated, not previous stores.
"""
    
SHIFTED_STORES_NAIVE_STORE_ORDER_PER_ENTRY_NAME = f"shifted_stores_{NAIVE_STORE_ORDER_PER_ENTRY_NAME}"
"""
RTL name for the naive store position per queue entry, shifted only based on the store queue pointer.
It is naive as it only considers loads and stores currently being allocated, not previous stores.
"""
    

def UNSHIFTED_WRITE_ENABLE_NAME(
        queue_type : QueueType, 
        ):
    """
    RTL name for the unshifted write enables signals of the (load/store) queue.
    """

    return f"unshifted_{WRITE_ENABLE_NAME(queue_type)}"


def POINTER_SUB_NAME(
        queue_type : QueueType
    ):
    match queue_type:
        case QueueType.LOAD:
            return "load_sub"
        case QueueType.STORE:
            return "store_sub"