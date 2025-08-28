from LSQ.entity import Signal, EntityComment
from LSQ.config import Config

from LSQ.rtl_signal_names import *

from LSQ.utils import get_as_binary_string_padded, get_required_bitwidth


from LSQ.operators.arithmetic import WrapSub, WrapSubReturn

class GroupAllocatorDeclarativePortItems():
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

    class GroupInitChannelComment(EntityComment):
        """
        RTL comment:
        
        -- Group init channels from the dataflow circuit

        -- {config.num_groups()} control channels,

        -- one for each group of memory operations.
        """
        def __init__(self, config : Config):
            comment = f"""

    -- Group init channels from the dataflow circuit
    -- {config.num_groups()} control channels,
    -- one for each group of memory operations.
""".removeprefix("\n")
            
            EntityComment.__init__(
                self,
                comment
            )

    class GroupInitValid(Signal):
        """
        Input

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
                )
            )


    class GroupInitReady(Signal):
        """
        Output.
         
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
                )
            )


    class QueueInputsComment(EntityComment):
        """
        RTL comment:
        
        -- Input signals from the (load/store) queue
        """
        def __init__(self, queue_type : QueueType):
            comment = f"""

    -- Input signals from the {queue_type} queue
""".removeprefix("\n")
            
            EntityComment.__init__(
                self,
                comment
            )


    class QueuePointer(Signal):
        """
        Input

        Bitwidth = N

        Number = 1

        Pointer to the (head/tail) entry of a queue.
        There is only 1 queue (head/tail) pointer. 
        Like all queue pointers, its bitwidth is equal to ceil(log2(num_queue_entries))
        """
        def __init__(self, 
                     config : Config,
                     queue_type : QueueType,
                     queue_pointer_type : QueuePointerType
                     ):
            match queue_type:
                case QueueType.LOAD:
                    bitwidth = config.load_queue_idx_bitwidth()
                case QueueType.STORE:
                    bitwidth = config.store_queue_idx_bitwidth()

            Signal.__init__(
                self,
                base_name=QUEUE_POINTER_NAME(queue_type, queue_pointer_type),
                direction=Signal.Direction.INPUT,
                size=Signal.Size(
                    bitwidth=bitwidth,
                    number=1
                )
            )


    class QueueIsEmpty(Signal):
        """
        Input

        Bitwidth = 1

        Number = 1

        isEmpty? signal for the (load/store) queue
        """
        def __init__(self, 
                     queue_type : QueueType
                     ):
            Signal.__init__(
                self,
                base_name=IS_EMPTY_NAME(queue_type),
                direction=Signal.Direction.INPUT,
                size=Signal.Size(
                    bitwidth=1,
                    number=1
                )
            )

    class QueueWriteEnableComment(EntityComment):
        """
        RTL comment:
            
        -- {queue_type.value} queue write enable signals

        -- {number} signals, one for each queue entry.
        """
        def __init__(
                self, 
                config: Config, 
                queue_type : QueueType
                ):
            
            match queue_type:
                case QueueType.LOAD:
                    number = config.load_queue_num_entries()
                case QueueType.STORE:
                    number = config.store_queue_num_entries()

            comment = f"""

    -- {queue_type.value} queue write enable signals
    -- {number} signals, one for each queue entry.
""".removeprefix("\n")
            
            EntityComment.__init__(
                self,
                comment
            )

    class QueueWriteEnable(Signal):
        """
        Output.
        
        Bitwidth = 1

        Number = N

        Write enable signals to the (load/store) queue, used to allocate entries in the load queue. 
        There are N 1-bit write enable signals.
        As expected for write enable signals to queue entries, there is 1 write enable signal per queue entry.
        """
        def __init__(self, 
                     config : Config,
                     queue_type : QueueType
                     ):
            match queue_type:
                case QueueType.LOAD:
                    number = config.load_queue_num_entries()
                case QueueType.STORE:
                    number = config.store_queue_num_entries()

            Signal.__init__(
                self,
                base_name=WRITE_ENABLE_NAME(queue_type),
                direction=Signal.Direction.OUTPUT,
                size=Signal.Size(
                    bitwidth=1,
                    number=number
                )
            )

    class NumNewQueueEntriesComment(EntityComment):
        """
        RTL comment:
            
        -- Number of new {queue_type_str} queue entries to allocate.

        -- Used by the {queue_type_str} queue to update its tail pointer.
        
        -- Bitwidth equal to the {queue_type_str} queue pointer bitwidth.
        """
        def __init__(
                self, 
                queue_type : QueueType
                ):

            queue_type_str = queue_type.value
            comment = f"""

    -- Number of new {queue_type_str} queue entries to allocate.
    -- Used by the {queue_type_str} queue to update its tail pointer.
    -- Bitwidth equal to the {queue_type_str} queue pointer bitwidth.
""".removeprefix("\n")
            EntityComment.__init__(
                self,
                comment
            )


    class NumNewQueueEntries(Signal):
        """
        Output.
        
        Bitwidth = N

        Number = 1

        Number of (load/store) queue entries to allocate,
        which is output directly to the (load/store) queue.

        Non-handshaked signal. 
        
        Used by the load queue to update its tail pointer, 
        using update logic appropriate to circular buffers.
        
        There is a single "number of load queue entries to allocate" signal,
        and its bitwidth is equal to the bitwidth of the load queue pointers, 
        to allow easy arithmetic between then.
        """
        def __init__(self, 
                     config : Config,
                     queue_type : QueueType
                     ):
            match queue_type:
                case QueueType.LOAD:
                    bitwidth = config.load_ports_idx_bitwidth()
                case QueueType.STORE:
                    bitwidth = config.store_queue_idx_bitwidth()

            Signal.__init__(
                self,
                base_name=NUM_NEW_QUEUE_ENTRIES_NAME(queue_type),
                direction=Signal.Direction.OUTPUT,
                size=Signal.Size(
                    bitwidth=bitwidth,
                    number=1
                )
            )

    class PortIndexPerQueueEntryComment(EntityComment):
        """
        RTL comment:
            
        -- Load port index to write into each load queue entry.

        -- {number} signals, each {bitwidth} bit(s).

        -- Not one-hot.

        -- There is inconsistant code implying this signal should not be present 

        -- if there are no load ports.

        -- But it is currently added regardless (with bitwidth 1)

        -- Actual number of load ports: {actual_num_ports}
        """
        def __init__(
                self, 
                config : Config,
                queue_type : QueueType
                ):

            match queue_type:
                case QueueType.LOAD:
                    number = config.load_queue_num_entries(),
                    bitwidth = config.load_ports_idx_bitwidth()
                    actual_num_ports = config.load_ports_num()
                case QueueType.STORE:
                    number = config.store_queue_num_entries(),
                    bitwidth = config.store_ports_idx_bitwidth()
                    actual_num_ports = config.store_ports_num()

            comment = f"""

    -- Load port index to write into each load queue entry.
    -- {number} signals, each {bitwidth} bit(s).
    -- Not one-hot.
    -- There is inconsistant code implying this signal should not be present 
    -- if there are no load ports.
    -- But it is currently added regardless (with bitwidth 1)
    -- Actual number of load ports: {actual_num_ports}
""".removeprefix("\n")
            EntityComment.__init__(
                self,
                comment
            )


    class PortIndexPerQueueEntry(Signal):
        """
        Output 
        
        Bitwidth = N

        Number = M

        Which (load/store) port index to allocate into each (load/store) queue entry. 
        
        The group allocator uses the head pointer from the (load/store) queue 
        to place the (load/store) port indices in the correct signal, 
        so that they arrive in the correct (load/store) queue entries. 
        
        This is guarded by the (load/store) queue entry write enable, 
        so not all of these signals are used.

        There is one signal per load queue entry, with the bitwidth required to identify a load port.
        Not one-hot.

        There is inconsistant code implying this signal should not be present 
        if there are no load ports.
        But it is currently added regardless (with bitwidth 1)
        """
        def __init__(self, 
                     config : Config,
                     queue_type : QueueType
                     ):
            match queue_type:
                case QueueType.LOAD:
                    bitwidth = config.load_ports_idx_bitwidth()
                case QueueType.STORE:
                    bitwidth = config.store_queue_idx_bitwidth()

            Signal.__init__(
                self,
                base_name=PORT_INDEX_PER_ENTRY_NAME(queue_type),
                direction=Signal.Direction.OUTPUT,
                size=Signal.Size(
                    bitwidth=bitwidth,
                    number=1
                )
            )

    class StorePositionPerLoadComment(EntityComment):
        """
        RTL comment:
            
        -- Store position per load

        -- {config.load_queue_num_entries()} signals, each {config.store_queue_num_entries()} bit(s).

        -- One per entry in the load queue, with 1 bit per entry in the store queue.

        -- The order of the memory operations, read from the ROM, 

        -- has been shifted to generate this,

        -- as well as 0s and 1s added correctly to fill out each signal.
        """
        def __init__(
                self, 
                config : Config,
                ):


            comment = f"""

    -- Store position per load
    -- {config.load_queue_num_entries()} signals, each {config.store_queue_num_entries()} bit(s).
    -- One per entry in the load queue, with 1 bit per entry in the store queue.
    -- The order of the memory operations, read from the ROM, 
    -- has been shifted to generate this,
    -- as well as 0s and 1s added correctly to fill out each signal.
""".removeprefix("\n")
            EntityComment.__init__(
                self,
                comment
            )



    class StorePositionPerLoad(Signal):
        """
        Output
        
        Bitwidth = N

        Number = N

        Whether the stores in the store queue and ahead or behind
        each specific entry in the load queue.
         
        There is one signal per entry in the load queue,
        and 1 bit per entry in the store queue.
        
        The order of the memory operations, read from the ROM,
        has been shifted to generate this, 
        as well as 0s and 1s added correctly to fill out each signal.

        This is done based on the store queue and load queue pointers.
        """

        def __init__(self, 
                     config : Config,
                     ):

            Signal.__init__(
                self,
                base_name=STORE_POSITION_PER_LOAD_NAME,
                direction=Signal.Direction.OUTPUT,
                size=Signal.Size(
                    bitwidth=config.store_queue_num_entries(), 
                    number=config.load_queue_num_entries()
                )
            )

class GroupAllocatorDeclarativeLocalItems():
    pass


class GroupHandshakingDeclarativePortItems():
    class GroupInitTransfer(Signal):
        """
        Output
        
        Bitwidth = 1

        Number = N

        Whether a particular group init channel transfers this cycle.
         
        1-bit signal, 1 signal per group of memory accesses
        """

        def __init__(self, 
                     config : Config,
                     ):

            Signal.__init__(
                self,
                base_name=GROUP_INIT_TRANSFER_NAME,
                direction=Signal.Direction.OUTPUT,
                size=Signal.Size(
                    bitwidth=1,
                    number=config.num_groups()
                )
            )
class GroupHandshakingDeclarativeLocalItems():
    class NumEmptyEntries(Signal):
        """
        Bitwidth = N

        Number = 1

        Number of empty entries in a queue, naively calculated.
        Needs to be combined with isEmpty to calculate the real value.
        """

        def __init__(self, 
                     config : Config,
                     queue_type : QueueType,
                     ):
            
            match queue_type:
                case QueueType.LOAD:
                        bitwidth = config.load_queue_idx_bitwidth()
                case QueueType.STORE:
                        bitwidth = config.store_queue_idx_bitwidth()

            Signal.__init__(
                self,
                base_name=NUM_EMPTY_ENTRIES_NAIVE_NAME(queue_type),
                size=Signal.Size(
                    bitwidth=bitwidth,
                    number=1
                )
            )

class GroupHandshakingDeclarativeBodyItems():
    class Body(Signal):

        def get_empty_entries_assignment(
                self,
                config : Config,
                queue_type : QueueType
            ):
            empty_entries_naive = NUM_EMPTY_ENTRIES_NAIVE_NAME(queue_type)
        
            head_pointer = QUEUE_POINTER_NAME(queue_type, QueuePointerType.HEAD)
            tail_pointer = QUEUE_POINTER_NAME(queue_type, QueuePointerType.TAIL)


            match queue_type:
                case QueueType.LOAD:
                    num_entries = config.load_queue_num_entries()
                case QueueType.STORE:
                    num_entries = config.store_queue_num_entries()


            wrap_sub_return = WrapSub(empty_entries_naive, head_pointer, tail_pointer, num_entries)
            if wrap_sub_return.single_line:
                return f"""

  -- num empty entries naive is generated by a wrap-around subtraction
  -- it is naive since it will be incorrect when the queue is empty
  {wrap_sub_return.line1}
""".removeprefix("\n")
            else:
                return f"""

  -- num empty entries naive is generated by a wrap-around subtraction
  -- it is naive since it will be incorrect when the queue is empty
  {wrap_sub_return.line1}
    {wrap_sub_return.line2}
    {wrap_sub_return.line3}
    {wrap_sub_return.line4}
"""

        def __init__(self, config : Config):
            self.item = ""
            self.item += self.get_empty_entries_assignment(config, QueueType.LOAD)
            self.item += self.get_empty_entries_assignment(config, QueueType.STORE)

            load_is_empty_name = f"{IS_EMPTY_NAME(QueueType.LOAD)}_i"
            store_is_empty_name = f"{IS_EMPTY_NAME(QueueType.STORE)}_i"

            load_empty_entries_naive = NUM_EMPTY_ENTRIES_NAIVE_NAME(QueueType.LOAD)
            store_empty_entries_naive = NUM_EMPTY_ENTRIES_NAIVE_NAME(QueueType.STORE)




            for i in range(config.num_groups()):
                init_ready_name = f"{GROUP_INIT_CHANNEL_NAME}_ready_{i}"

                num_loads = config.group_num_loads(i)
                num_stores = config.group_num_stores(i)

                self.item += f"""

  -- process to generate the ready signals for group init channel {i}
  -- by checking the number of empty elements vs. 
  -- the number of loads and stores in that group of memory operations
  -- to see if there is space to allocate them.
  process(all)
  begin
    -- if both queues are empty, the group allocator is ready
    if {load_is_empty_name} = '1' and {store_is_empty_name} = '1' then
        {init_ready_name} <= '1';
    -- otherwise, we must compare the number of loads and stores in this group
    -- to the number of empty entries in each queue.
    -- if either queue does not have enough space, the group allocator is not ready
    --
    -- Group {i} has:
    --      {num_loads} load(s)
    --      {num_stores} store(s)
    elsif unsigned({load_empty_entries_naive}) < {num_loads} or 
          unsigned({store_empty_entries_naive}) < {num_stores}
        {init_ready_name} <= '0';
    else
        {init_ready_name} <= '1';
    end if;
  end process;
""".removeprefix("\n")