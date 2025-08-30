from LSQ.entity import Signal, EntityComment, Instantiation, SimpleInstantiation, InstCxnType, Signal2D
from LSQ.config import Config

from LSQ.rtl_signal_names import *

from LSQ.utils import get_as_binary_string_padded, get_required_bitwidth, one_hot


from LSQ.operators.arithmetic import WrapSub

class PortIdxPerEntryLocalItems():
    class PortIdxPerQueueEntry(Signal2D):
        """
        Bitwidth = N
        Number = M

        Local 2D input vector storing the 
        (unshifted/shifted) port index per queue entry
         
        Bitwidth is bitwidth required to present port_idx
        Number is equal to the number of queue entries
        """
        def __init__(self, 
                     config : Config,
                     queue_type : QueueType,
                     shifted = False
                     ):
            
            match queue_type:
                case QueueType.LOAD:
                    bitwidth = config.load_ports_idx_bitwidth()
                    number = config.load_queue_num_entries()
                case QueueType.STORE:
                    bitwidth = config.store_ports_idx_bitwidth()
                    number = config.store_queue_num_entries()

            if shifted:
                base_name = PORT_INDEX_PER_ENTRY_NAME(queue_type)
            else:
                base_name = UNSHIFTED_PORT_INDEX_PER_ENTRY_NAME(queue_type)

            Signal2D.__init__(
                self,
                base_name=base_name,
                direction=Signal.Direction.INPUT,
                size=Signal.Size(
                    bitwidth=bitwidth,
                    number=number
                )
            )

class PortIdxPerEntryBodyItems():
    class Body():

        def _get_default_value(self, queue_type, idx, bitwidth):
            return f"""
    {UNSHIFTED_PORT_INDEX_PER_ENTRY_NAME(queue_type)}_{idx} <= {get_as_binary_string_padded(0, bitwidth)};
""".removeprefix("\n")

        def __init__(self, config : Config, queue_type : QueueType):
            
            pointer_name = QUEUE_POINTER_NAME(queue_type, QueuePointerType.TAIL)

            match queue_type:
                case QueueType.LOAD:
                    idx_bitwidth = config.load_ports_idx_bitwidth()
                    def ports(group_idx) : return config.group_load_ports(group_idx)
                    def has_items(group_idx): return config.group_num_loads(group_idx) > 0
                    num_entries = config.load_queue_num_entries()
                    
                case QueueType.STORE:
                    idx_bitwidth = config.store_ports_idx_bitwidth()
                    def ports(group_idx) : return config.group_store_ports(group_idx)
                    def has_items(group_idx): return config.group_num_stores(group_idx) > 0
                    num_entries = config.store_queue_num_entries()

            default_assignments = ""

            default_assignments += f"""
    -- If a group has less than {num_entries} {queue_type.value}s
    -- set the other port indices to 0
    {UNSHIFTED_PORT_INDEX_PER_ENTRY_NAME(queue_type)} <= (others => (others => '0'));
"""

            default_assignments = default_assignments.strip()

            case_input = ""
            num_cases = 0
            for i in range(config.num_groups()):
                if has_items(i):      
                    num_cases = num_cases + 1  
                    case_input += f"""
      {GROUP_INIT_TRANSFER_NAME}_{i}_i &
""" .removeprefix("\n")
            case_input = case_input.strip()[:-1]

            cases = ""

            case_number = 0
            for i in range(config.num_groups()):
                if has_items(i):      
                    group_one_hot = one_hot(case_number, num_cases)
                    case_number = case_number + 1
                    cases += f"""
      when {group_one_hot} =>
""".removeprefix("\n")
                    for j, idx in enumerate(ports(i)):
                        cases += f"""
        -- {queue_type.value} {j} of group {i} is from {queue_type.value} port {idx}
        {UNSHIFTED_PORT_INDEX_PER_ENTRY_NAME(queue_type)}({j}) <= {get_as_binary_string_padded(idx, idx_bitwidth)};

""".removeprefix("\n")
                else:
                    cases += f"""
      -- Group {i} has no {queue_type.value}s

""".removeprefix("\n")

            cases = cases.strip()

            unshifted_assignments = f"""
    -- This LSQ was generated without multi-group allocation
    -- and so assumes the dataflow circuit will only ever 
    -- have 1 group valid signal in a given cycle

    -- Using case statement to help infer one-hot mux
    case
      {case_input}
    is
      {cases}

    end case;
""".removeprefix("\n").strip()


            shifted_assignments = f"""
    -- {queue_type.value} port indices must be mod left shifted based on queue tail
    for i in 0 to {num_entries} - 1 loop
""".removeprefix("\n")
            
            port_idx = PORT_INDEX_PER_ENTRY_NAME(queue_type)
            unsh_port_idx = UNSHIFTED_PORT_INDEX_PER_ENTRY_NAME
            shifted_assignments += f"""
      {port_idx}(i) <=
        {unsh_port_idx(queue_type)}(
          (i + {pointer_name}_int)) mod {num_entries}
        );
""".removeprefix("\n")

            shifted_assignments += f"""
    end loop;
""".removeprefix("\n")
            
            shifted_assignments = shifted_assignments.lstrip()

            output_assignments = ""

            for i in range(num_entries):
                output_name = f"{PORT_INDEX_PER_ENTRY_NAME(queue_type)}_{i}_o"

                # pad single digit output names
                if i < 10:
                    output_name += " "


                output_assignments += f"""
  {output_name} <= {PORT_INDEX_PER_ENTRY_NAME(queue_type)}({i});
""".removeprefix("\n")
            
            output_assignments = output_assignments.strip()

            self.item = f"""
  process(all)
    variable offset : natural;
  begin
    -- convert q tail pointer to integer
    {pointer_name}_int = integer(unsigned({pointer_name}_i)

    {default_assignments}

    {unshifted_assignments}

    {shifted_assignments}
  end process;

  {output_assignments}
""".removeprefix("\n").strip()

        def get(self):
            return self.item
        


class StoreOrderPerEntryLocalItems():
    class StoreOrderPerEntry(Signal2D):
        """
        Bitwidth = N
        Number = M

        Local 2D input vector storing the 
        (unshifted/shifted) store order per queue entry
         
        Bitwidth is equal to the number of store queue entriews
        Number is equal to the number of load queue entries
        """
        def __init__(self, 
                     config : Config,
                     shifted = False
                     ):
            
            bitwidth = config.store_queue_num_entries()
            number = config.load_queue_num_entries()

            if shifted:
                base_name = STORE_ORDER_PER_ENTRY_NAME
            else:
                base_name = UNSHIFTED_STORE_ORDER_PER_ENTRY_NAME

            Signal2D.__init__(
                self,
                base_name=base_name,
                direction=Signal.Direction.INPUT,
                size=Signal.Size(
                    bitwidth=bitwidth,
                    number=number
                )
            )

class StoreOrderPerEntryBodyItems():
    class Body():

        def __init__(self, config : Config):
            
            load_pointer_name = QUEUE_POINTER_NAME(QueueType.LOAD, QueuePointerType.TAIL)
            store_pointer_name = QUEUE_POINTER_NAME(QueueType.STORE, QueuePointerType.TAIL)




#             default_assignments = ""

#             default_assignments += f"""
#     -- If a group has less than {num_entries} {queue_type.value}s
#     -- set the other port indices to 0
#     {UNSHIFTED_PORT_INDEX_PER_ENTRY_NAME(queue_type)} <= (others => (others => '0'));
# """

#             default_assignments = default_assignments.strip()

#             case_input = ""
#             num_cases = 0
#             for i in range(config.num_groups()):
#                 if has_items(i):      
#                     num_cases = num_cases + 1  
#                     case_input += f"""
#       {GROUP_INIT_TRANSFER_NAME}_{i}_i &
# """ .removeprefix("\n")
#             case_input = case_input.strip()[:-1]

#             cases = ""

#             case_number = 0
#             for i in range(config.num_groups()):
#                 if has_items(i):      
#                     group_one_hot = one_hot(case_number, num_cases)
#                     case_number = case_number + 1
#                     cases += f"""
#       when {group_one_hot} =>
# """.removeprefix("\n")
#                     for j, idx in enumerate(ports(i)):
#                         cases += f"""
#         -- {queue_type.value} {j} of group {i} is from {queue_type.value} port {idx}
#         {UNSHIFTED_PORT_INDEX_PER_ENTRY_NAME(queue_type)}({j}) <= {get_as_binary_string_padded(idx, idx_bitwidth)};

# """.removeprefix("\n")
#                 else:
#                     cases += f"""
#       -- Group {i} has no {queue_type.value}s

# """.removeprefix("\n")

#             cases = cases.strip()

#             unshifted_assignments = f"""
#     -- This LSQ was generated without multi-group allocation
#     -- and so assumes the dataflow circuit will only ever 
#     -- have 1 group valid signal in a given cycle

#     -- Using case statement to help infer one-hot mux
#     case
#       {case_input}
#     is
#       {cases}

#     end case;
# """.removeprefix("\n").strip()


#             shifted_assignments = f"""
#     -- {queue_type.value} port indices must be mod left shifted based on queue tail
#     for i in {num_entries} - 1 downto 0 loop
# """.removeprefix("\n")
            
#             port_idx = PORT_INDEX_PER_ENTRY_NAME(queue_type)
#             unsh_port_idx = UNSHIFTED_PORT_INDEX_PER_ENTRY_NAME
#             shifted_assignments += f"""
#       {port_idx}(i) <=
#         {unsh_port_idx(queue_type)}(
#           (i + offset)) mod {num_entries}
#         );
# """.removeprefix("\n")

#             shifted_assignments += f"""
#     end loop;
# """.removeprefix("\n")
            
#             shifted_assignments = shifted_assignments.lstrip()

            shifted = STORE_ORDER_PER_ENTRY_NAME
            unshifted = UNSHIFTED_STORE_ORDER_PER_ENTRY_NAME
            shifted_assignments = f"""
      for i in 0 to {config.load_queue_num_entries()} loop
        for j in 0 to {config.store_queue_num_entries()} loop
          row_idx := (i + {load_pointer_name}_int) mod {config.load_queue_num_entries()}
          col_idx := (j + {store_pointer_name}_int) mod {config.store_queue_num_entries()}
          {shifted}(row_idx)(col_idx) <= {unshifted}(i)(j)
        end loop;
      end loop;
""".strip()

            output_assignments = ""

            for i in range(config.load_queue_num_entries()):
                output_name = f"{STORE_ORDER_PER_ENTRY_NAME}_{i}_o"

                # pad single digit output names
                if i < 10:
                    output_name += " "


                output_assignments += f"""
  {output_name} <= {STORE_ORDER_PER_ENTRY_NAME}({i});
""".removeprefix("\n")
            
            output_assignments = output_assignments.strip()

            self.item = f"""
  process(all)
    -- tail pointers as integers for indexing
    variable {load_pointer_name}_int, {store_pointer_name}  : natural;
    -- where to shift a value to
    variable row_idx, col_idx : natural;
  begin
    -- convert q tail pointers to integer
    {load_pointer_name}_int = integer(unsigned({load_pointer_name}_i)
    {store_pointer_name}_int = integer(unsigned({store_pointer_name}_i)

    
    {shifted_assignments}

  end process;


  {output_assignments}
""".removeprefix("\n").strip()

        def get(self):
            return self.item

class GroupAllocatorPortItems():
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

    class PortIdxPerQueueEntryComment(EntityComment):
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


    class PortIdxPerQueueEntry(Signal):
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
                    number = config.load_queue_num_entries()
                case QueueType.STORE:
                    bitwidth = config.store_queue_idx_bitwidth()
                    number = config.store_queue_num_entries()

            Signal.__init__(
                self,
                base_name=PORT_INDEX_PER_ENTRY_NAME(queue_type),
                direction=Signal.Direction.OUTPUT,
                size=Signal.Size(
                    bitwidth=bitwidth,
                    number=number
                )
            )

    class StoreOrderPerEntryComment(EntityComment):
        """
        RTL comment:
            
        -- Store order per load queue entry

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

    -- Store order per load queue entry
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



    class StoreOrderPerEntry(Signal):
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
                base_name=STORE_ORDER_PER_ENTRY_NAME,
                direction=Signal.Direction.OUTPUT,
                size=Signal.Size(
                    bitwidth=config.store_queue_num_entries(), 
                    number=config.load_queue_num_entries()
                )
            )


class GroupAllocatorBodyItems():
    class HandshakingInst(Instantiation):
        def __init__(self, config : Config):

            p = GroupAllocatorPortItems()
            l = GroupAllocatorLocalItems()
            c = InstCxnType

            d = Signal.Direction

            si = SimpleInstantiation
            port_items = [
                si(p.GroupInitValid(config), c.INPUT),
                si(p.GroupInitReady(config), c.OUTPUT),

                si(p.QueuePointer(config, QueueType.LOAD, QueuePointerType.TAIL), c.INPUT),
                si(p.QueuePointer(config, QueueType.LOAD, QueuePointerType.HEAD), c.INPUT),
                si(p.QueueIsEmpty(QueueType.LOAD), c.INPUT),

                si(p.QueuePointer(config, QueueType.STORE, QueuePointerType.TAIL), c.INPUT),
                si(p.QueuePointer(config, QueueType.STORE, QueuePointerType.HEAD), c.INPUT),
                si(p.QueueIsEmpty(QueueType.STORE), c.INPUT),

                si(p.QueuePointer(config, QueueType.STORE, QueuePointerType.TAIL), c.INPUT),
                si(p.QueuePointer(config, QueueType.STORE, QueuePointerType.HEAD), c.INPUT),
                si(p.QueueIsEmpty(QueueType.STORE), c.INPUT),

                si(l.GroupInitTransfer(config, d.OUTPUT), c.LOCAL)
            ]

            Instantiation.__init__(
                self,
                name=GROUP_HANDSHAKING_ENTITY_NAME,
                entity_name=GROUP_HANDSHAKING_ENTITY_NAME,
                port_items=port_items
            )

    class PortIdxPerEntryInst(Instantiation):
        def __init__(self, config : Config, queue_type : QueueType):

            ga_l = GroupAllocatorLocalItems()
            ga_p = GroupAllocatorPortItems()
            c = InstCxnType
            d = Signal.Direction

            si = SimpleInstantiation
            port_items = [
                si(ga_l.GroupInitTransfer(config, d.INPUT), c.LOCAL),

                si(ga_p.QueuePointer(
                    config, 
                    queue_type, 
                    QueuePointerType.TAIL),
                    c.INPUT
                ),

                si(ga_p.PortIdxPerQueueEntry(
                    config, 
                    queue_type),
                    c.OUTPUT
                )
            ]

            Instantiation.__init__(
                self,
                name=PORT_INDEX_PER_ENTRY_NAME(queue_type),
                entity_name=PORT_INDEX_PER_ENTRY_NAME(queue_type),
                port_items=port_items
            )

            
class GroupAllocatorLocalItems():
    class GroupInitTransfer(Signal):
        """
        Local signal
        
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
                )
            )


class GroupHandshakingLocalItems():
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


            wrap_sub_return = WrapSub(empty_entries_naive, f"{head_pointer}_i", f"{tail_pointer}_i", num_entries)
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
                init_valid_name = f"{GROUP_INIT_CHANNEL_NAME}_valid_{i}_i"
                init_transfer_name = f"{GROUP_INIT_TRANSFER_NAME}_{i}_o"

                num_loads = config.group_num_loads(i)
                num_stores = config.group_num_stores(i)

                load_pointer_bitwidth = config.load_queue_idx_bitwidth()
                store_pointer_bitwidth = config.store_queue_idx_bitwidth()

                num_loads_binary_bitwidth = get_required_bitwidth(num_loads)
                num_stores_binary_bitwidth = get_required_bitwidth(num_stores)

                group_num_loads_binary = get_as_binary_string_padded(num_loads, load_pointer_bitwidth)
                group_num_stores_binary = get_as_binary_string_padded(num_stores, store_pointer_bitwidth)


                load_empty_entries_naive_use = load_empty_entries_naive
                store_empty_entries_naive_use = store_empty_entries_naive


                # load_empty_entries is the size of the load queue pointers
                # which may be 1 bit too small to compare to the number of required loads
                if load_pointer_bitwidth + 1 == num_loads_binary_bitwidth:
                    load_empty_entries_naive_use = f"0 & {load_empty_entries_naive}"
                elif load_pointer_bitwidth < num_loads_binary_bitwidth:
                    raise RuntimeError(
                        f"Unexpected comparison bitwidths. Pointer is {load_pointer_bitwidth} bits, " + \
                        f" num stores bitwidth is {num_loads_binary_bitwidth}"
                        )

                # store_empty_entries is the size of the store queue pointers
                # which may be 1 bit too small to compare to the number of required stores
                if config.store_queue_idx_bitwidth() + 1 == num_stores_binary_bitwidth:
                    store_empty_entries_naive_use = f"0 & {store_empty_entries_naive}"
                elif store_pointer_bitwidth < num_stores_binary_bitwidth:
                    raise RuntimeError(
                        f"Unexpected comparison bitwidths. Pointer is {store_pointer_bitwidth} bits, " + \
                        f" num stores bitwidth is {num_stores_binary_bitwidth}"
                        )

                self.item += f"""

  -- process to generate the ready signals for group init channel {i}
  -- by checking the number of empty elements vs. 
  -- the number of loads and stores in that group of memory operations
  -- to see if there is space to allocate them.
  -- if either queue does not have enough space, the group allocator is not ready
  -- Group {i} has:
  --      {num_loads} load(s)
  --      {num_stores} store(s)
  process(all)
  begin
    -- if the load queue does not have space
    if {load_is_empty_name} = '0' and {load_empty_entries_naive_use} < {group_num_loads_binary} then
        {init_ready_name} <= '0';
    -- if the store queue does not have space
    elsif {store_is_empty_name} = '0' and {store_empty_entries_naive_use} < {group_num_stores_binary} then
        {init_ready_name} <= '0';
    else 
        {init_ready_name} <= '1';
    end if;
  end process;

 -- drive the ready output
  {init_ready_name}_o <= {init_ready_name};

 -- drive the transfer output
 {init_transfer_name} <= {init_valid_name} and {init_ready_name};
""".removeprefix("\n")