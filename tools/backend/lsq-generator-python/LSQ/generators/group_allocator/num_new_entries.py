from LSQ.config import Config
from LSQ.utils import QueueType, bin_string

from LSQ.rtl_signal_names import *

from LSQ.entity import Signal

import LSQ.declarative_signals as ds


class NumNewEntries():
    def __init__(self, config : Config, queue_type : QueueType, prefix):
        self.top_level_comment = f"""
-- Number of New Entries in the (Load/Store) Queue Unit
-- Sub-unit of the Group Allocator.
--
-- Generates the number of newly allocated {queue_type.value} queue entries.
--
-- This is used by the {queue_type.value} queue to update its tail pointer,
-- based on circular buffer pointer update logic.
--
-- It is also used to generate the write enable signals
-- for the {queue_type.value} queue.
""".strip()

        self.name = NUM_NEW_ENTRIES_NAME(queue_type)
        self.prefix = prefix


        d = Signal.Direction
        self.entity_port_items = [
            ds.GroupInitTransfer(
                config, 
                d.INPUT
            ),
            ds.NumNewQueueEntries(
                config, 
                queue_type, 
                direction=d.OUTPUT
            )
        ]

        self.local_items = [
            NumNewEntriesMasked(
                config, 
                queue_type, 
                )
        ]

        b = NumNewEntriesBody()
        self.body = [
            b.Body(config, queue_type)
        ]

class NumNewEntriesBody():
    class Body():
        def _set_params(self, config : Config, queue_type : QueueType):
            match queue_type:
                case QueueType.LOAD:
                    def new_entries(idx) : return config.group_num_loads(idx)
                    self.new_entries = new_entries

                    self.new_entries_bitwidth = config.load_queue_idx_bitwidth()

                    def has_items(group_idx): return config.group_num_loads(group_idx) > 0
                    self.has_items = has_items
                case QueueType.STORE:
                    def new_entries(idx): return config.group_num_stores(idx)
                    self.new_entries = new_entries

                    self.new_entries_bitwidth = config.store_queue_idx_bitwidth()

                    def has_items(group_idx): return config.group_num_stores(group_idx) > 0
                    self.has_items = has_items

        def __init__(self, config : Config, queue_type : QueueType):
            self._set_params(config, queue_type)

            self.item = ""
           
            zeros_binary = bin_string(0, self.new_entries_bitwidth)

            num_new_entries_masked = MASKED_NUM_NEW_ENTRIES_NAME(queue_type)
            num_new_entries= NUM_NEW_ENTRIES_NAME(queue_type)

            mask_id = -1
            for i in range(config.num_groups()):
                if self.has_items(i):  
                    mask_id = mask_id + 1

                    new_entries = self.new_entries(i)
                    new_entries_binary = bin_string(new_entries, self.new_entries_bitwidth)

                    self.item += f"""
  -- Group {i} has {new_entries} {queue_type.value}(s)
  {num_new_entries_masked}_{mask_id} <= {new_entries_binary} when {GROUP_INIT_TRANSFER_NAME}_{i}_i else {zeros_binary};

""".removeprefix("\n")
                    
                else:
                    self.item += f"""
  -- Group {i} has no {queue_type.value}(s)

""".removeprefix("\n")
                    
            # one hot with one input
            if mask_id == 1:
                self.item += f"""
    {num_new_entries}_o <= {f"{num_new_entries_masked}_{mask_id}"};
"""
            else:
                # generate the or of each masked signal
                # apart from the last one
                one_hot_ors = ""
                for i in range(mask_id):
                    one_hot_ors += f"""
        {f"{num_new_entries_masked}_{i}"}
        or            
    """.removeprefix("\n")
                    
                one_hot_ors = one_hot_ors.strip()

                # assignment and last input to the or
                # as well as the ending semi colon
                self.item += f"""
    -- Since the inputs are masked by one-hot valid signals
    -- The output is simply an OR of the inputs
    {num_new_entries}_o <= 
        {one_hot_ors}
        {f"{num_new_entries_masked}_{mask_id}"};
    """
          
        def get(self):
            return self.item



# Declarative local signal only used by the num loads unit
class NumNewEntriesMasked(Signal):
    """
    Bitwidth = N, Number = M

    Number of new entries allocated to the queue,
    each specific to a group

    Same bitwidth as queue pointer.
    1 per group (only if they have a non-zero value)
    """

    def __init__(self, 
                    config : Config,
                    queue_type : QueueType,
                    ):
        
        
        match queue_type:
            case QueueType.LOAD:
                number = config.num_groups_with_loads()
            case QueueType.STORE:
                number = config.num_groups_with_stores()

        Signal.__init__(
            self,
            base_name=MASKED_NUM_NEW_ENTRIES_NAME(queue_type),
            size=Signal.Size(
                bitwidth=config.queue_idx_bitwidth(queue_type),
                number=number
            )
        )