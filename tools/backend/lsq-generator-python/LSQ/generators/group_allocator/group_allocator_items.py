from LSQ.entity import Signal, EntityComment, Instantiation, SimpleInstantiation, InstCxnType, Signal2D
from LSQ.config import Config

from LSQ.rtl_signal_names import *

from LSQ.utils import get_as_binary_string_padded, get_required_bitwidth, one_hot, mask_until

import LSQ.declarative_signals as ds

class WriteEnableLocalItems():
    class WriteEnable(Signal):
        """
        Bitwidth = N
        Number = 1

        Single vector storing the 
        (unshifted/shifted) write enable per queue entry
         
        Bitwidth is equal to the number of queue entries
        Number is 1
        """
        def __init__(self, 
                     config : Config,
                     queue_type : QueueType,
                     shifted = False
                     ):
            
            match queue_type:
                case QueueType.LOAD:
                    bitwidth = config.load_queue_num_entries()
                case QueueType.STORE:
                    bitwidth = config.store_queue_num_entries()

            if shifted:
                base_name = WRITE_ENABLE_NAME(queue_type)
            else:
                base_name = UNSHIFTED_WRITE_ENABLE_NAME(queue_type)

            Signal2D.__init__(
                self,
                base_name=base_name,
                direction=Signal.Direction.INPUT,
                size=Signal.Size(
                    bitwidth=bitwidth,
                    number=1
                )
            )

class WriteEnableBodyItems():
    class Body():
        def _set_params(self, config : Config, queue_type : QueueType):
            self.pointer_name = QUEUE_POINTER_NAME(queue_type, QueuePointerType.TAIL)

            match queue_type:
                case QueueType.LOAD:
                    self.num_entries = config.load_queue_num_entries()
                case QueueType.STORE:
                    self.num_entries = config.store_queue_num_entries()

        def _unshifted_assignments(self, queue_type : QueueType):
            unshf_wen = UNSHIFTED_WRITE_ENABLE_NAME(queue_type)
            new_entries = NUM_NEW_QUEUE_ENTRIES_NAME(queue_type)

            self.unshifted_assignments = f"""
  process(all)
    variable {new_entries}_int : natural;
  begin
    {new_entries}_int := to_integer(unsigned({new_entries}_i));

    for i in 0 to {self.num_entries} - 1 loop
      {unshf_wen}(i) <= '1' when i < {new_entries}_int else '0';
    end loop;
  end process;
""".strip()

        def _shifted_assignments(self, queue_type : QueueType):
            wen = WRITE_ENABLE_NAME(queue_type)
            unsh_wen = UNSHIFTED_WRITE_ENABLE_NAME(queue_type)

            self.shifted_assignments = f"""
  process(all)
    variable {self.pointer_name}_int : natural;
  begin
    {self.pointer_name}_int := to_integer(unsigned({self.pointer_name}_i));

    -- {queue_type.value} write enables must be mod left-shifted based on queue tail
    for i in 0 to {self.num_entries} - 1 loop
      {wen}((i + {self.pointer_name}_int) mod {self.num_entries}) <=
        {unsh_wen}(i);
    end loop;
  end process;
""".strip()


        def _output_assignments(self, queue_type):
            self.output_assignments = ""

            for i in range(self.num_entries):
                assign_to = f"{WRITE_ENABLE_NAME(queue_type)}_{i}_o"

                if i < 10:
                    assign_to += " "

                self.output_assignments += f"""
  {assign_to} <= {WRITE_ENABLE_NAME(queue_type)}({i});
""".removeprefix("\n")
                
            self.output_assignments = self.output_assignments.strip()


        def __init__(self, config : Config, queue_type : QueueType):
            self._set_params(config, queue_type)

            self._unshifted_assignments(queue_type)
            self._shifted_assignments(queue_type)
            self._output_assignments(queue_type)


            self.item = f"""
  {self.unshifted_assignments}

  {self.shifted_assignments}

  {self.output_assignments}
    
""".strip()
            
        def get(self):
            return self.item

class NumNewQueueEntriesBody():
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

            ############################
            # Build mux inner pieces
            ############################

            # case input is the std_logic_vector of concatenated bits
            # we pass to the mux's case statement
            case_input = ""

            # not all groups have loads/store
            # if the group does not have any of 
            # the relevant memory op,
            # we do not pass its transfer signal to the mux
            #
            # num_cases tracks how many groups 
            # are passed to the mux
            
            num_cases = 0

            # add each group to the mux's case statement input
            # if it has he relevant memory op
            for i in range(config.num_groups()):
                if self.has_items(i):      
                    
                    case_input += f"""
    case_input({num_cases}) := {GROUP_INIT_TRANSFER_NAME}_{i}_i;
""" .removeprefix("\n")
                    
                    num_cases = num_cases + 1  
                    
            case_input = case_input.strip()
                    

            # cases are the mux's data inputs
            # each is associated with one of the inputs
            # by a 'when' statement
            # and then a set of assignments
            cases = ""

            # not all groups are in the mux, 
            # so we need to track how many 'when' statements
            # we have added
            case_number = 0
            for i in range(config.num_groups()):
                # if it has at least one of the relevant ops
                if self.has_items(i):      
                    # get the case number one-hot encoded
                    # (not the group number, since not all groups are in the mux)
                    group_one_hot = one_hot(case_number, num_cases)
                    case_number = case_number + 1
                    
                    new_entries = self.new_entries(i)

                    assign_to = f"{NUM_NEW_QUEUE_ENTRIES_NAME(queue_type)}_o"
                    new_entries_bin = get_as_binary_string_padded(new_entries, self.new_entries_bitwidth)
                    

                    # map assignments to a select input
                    cases += f"""
      when {group_one_hot} =>
        -- Group {i} has {new_entries} {queue_type.value}(s)
        {assign_to} <= {new_entries_bin};

""".removeprefix("\n")

                # if there are no loads/stores
                else:
                    cases += f"""
      -- Group {i} has no {queue_type.value}s

""".removeprefix("\n")
                
                    
            cases += f"""
      -- defaults handled at top of process
      when others =>
        null;
""".removeprefix("\n")

            # format correctly
            cases = cases.strip()


            ############################
            # Actual mux statement
            ############################

            self.item = f"""
  process(all)
    variable case_input : std_logic_vector({num_cases} - 1 downto 0);
  begin
    -- If no group is transferring,
    -- or the group has no {queue_type.value}s,
    -- then set to zero
    {NUM_NEW_QUEUE_ENTRIES_NAME(queue_type)}_o <= (others => '0');

    {case_input}

    -- This LSQ was generated without multi-group allocation
    -- and so assumes the dataflow circuit will only ever 
    -- have 1 group valid signal in a given cycle

    -- Using case statement to help infer one-hot mux
    case
      case_input
    is
      {cases}

    end case;
  end process;
""".strip()
            
        def get(self):
            return self.item


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
        def _set_parameters(self, config : Config, queue_type : QueueType):
            self.pointer_name = QUEUE_POINTER_NAME(queue_type, QueuePointerType.TAIL)

            match queue_type:
                case QueueType.LOAD:
                    self.idx_bitwidth = config.load_ports_idx_bitwidth()
                    def ports(group_idx) : return config.group_load_ports(group_idx)
                    self.ports = ports

                    def has_items(group_idx): return config.group_num_loads(group_idx) > 0
                    self.has_items = has_items

                    self.num_entries = config.load_queue_num_entries()
                    
                case QueueType.STORE:
                    self.idx_bitwidth = config.store_ports_idx_bitwidth()

                    def ports(group_idx) : return config.group_store_ports(group_idx)
                    self.ports = ports

                    def has_items(group_idx): return config.group_num_stores(group_idx) > 0
                    self.has_items = has_items

                    self.num_entries = config.store_queue_num_entries()

        def _mux_rom(self, config : Config, queue_type):
            """
            Sets the VHDL for the one-hot mux from a ROM
            into self.unshifted_assignments

            The ROM values are the port indices per group.
            """


            ############################
            # Build mux inner pieces
            ############################

            # case input is the std_logic_vector of concatenated bits
            # we pass to the mux's case statement
            case_inputs = ""

            # not all groups have loads/store
            # if the group does not have any of 
            # the relevant memory op,
            # we do not pass its transfer signal to the mux
            #
            # num_cases tracks how many groups 
            # are passed to the mux
            self.num_cases = 0

            # add each group to the mux's case statement input
            # if it has he relevant memory op
            for i in range(config.num_groups()):
                if self.has_items(i):      
                     
                    case_inputs += f"""
    case_input({self.num_cases}) := {GROUP_INIT_TRANSFER_NAME}_{i}_i;
""" .removeprefix("\n")
                    
                    self.num_cases = self.num_cases + 1 
                    
            case_inputs = case_inputs.strip()

            # example case input:
            #

            # cases are the mux's data inputs
            # each is associated with one of the inputs
            # by a 'when' statement
            # and then a set of assignments
            cases = ""

            # not all groups are in the mux, 
            # so we need to track how many 'when' statements
            # we have added
            case_number = 0
            for i in range(config.num_groups()):
                # if it has at least one of the relevant ops
                if self.has_items(i):      
                    # get the case number one-hot encoded
                    # (not the group number, since not all groups are in the mux)
                    group_one_hot = one_hot(case_number, self.num_cases)
                    case_number = case_number + 1
                    
                    # map assignments to a select input
                    cases += f"""
      when {group_one_hot} =>
""".removeprefix("\n")
                    
                    # each group can have many loads or stores
                    # up to the maximum queue size
                    # each store or load in the group must be
                    # correctly associated with a port
                    for j, idx in enumerate(self.ports(i)):
                        assign_to = f"{UNSHIFTED_PORT_INDEX_PER_ENTRY_NAME(queue_type)}({j})"
                        idx_bin = get_as_binary_string_padded(idx, self.idx_bitwidth)

                        cases += f"""
        -- {queue_type.value} {j} of group {i} is from {queue_type.value} port {idx}
        {assign_to} <= {idx_bin};

""".removeprefix("\n")
                        
                # if there are no loads/stores
                else:
                    cases += f"""
      -- Group {i} has no {queue_type.value}s

""".removeprefix("\n")

            cases += f"""
      -- defaults handled at top of process
      when others =>
        null;
""".removeprefix("\n")

            # format correctly
            cases = cases.strip()


            ############################
            # Actual mux statement
            ############################

            self.unshifted_assignments = f"""
    -- If a group has less than {self.num_entries} {queue_type.value}s
    -- set the other port indices to 0
    {UNSHIFTED_PORT_INDEX_PER_ENTRY_NAME(queue_type)} <= (others => (others => '0'));

    {case_inputs}

    -- This LSQ was generated without multi-group allocation
    -- and so assumes the dataflow circuit will only ever 
    -- have 1 group valid signal in a given cycle

    -- Using case statement to help infer one-hot mux
    case
      case_input
    is
      {cases}

    end case;
""".removeprefix("\n").strip()
            
        def _shift(self, queue_type : QueueType):
            """
            Use indexing into data_array to infer barrel shift
            based on queue tail.
            """
            port_idx = PORT_INDEX_PER_ENTRY_NAME(queue_type)
            unsh_port_idx = UNSHIFTED_PORT_INDEX_PER_ENTRY_NAME(queue_type)

            self.shifted_assignments = f"""
    -- {queue_type.value} port indices must be mod left-shifted based on queue tail
    for i in 0 to {self.num_entries} - 1 loop
      {port_idx}((i + {self.pointer_name}_int) mod {self.num_entries}) <=
        {unsh_port_idx}(i);
        
    end loop;
""".strip()

        def _output_assigments(self, queue_type : QueueType):
            """
            Convert back from the shifted outputs as a data array,
            to individual signals
            """
            self.output_assignments = ""

            for i in range(self.num_entries):
                output_name = f"{PORT_INDEX_PER_ENTRY_NAME(queue_type)}_{i}_o"

                # pad single digit output names
                if i < 10:
                    output_name += " "

                self.output_assignments += f"""
  {output_name} <= {PORT_INDEX_PER_ENTRY_NAME(queue_type)}({i});
""".removeprefix("\n")
            
            self.output_assignments = self.output_assignments.strip()

        def __init__(self, config : Config, queue_type : QueueType):

            self._set_parameters(config, queue_type)

            self._mux_rom(config, queue_type)
            self._shift(queue_type)
            self._output_assigments(queue_type)


            self.item = f"""
  process(all)
    variable {self.pointer_name}_int : natural;

    variable case_input : std_logic_vector({self.num_cases} - 1 downto 0);
  begin
    -- convert q tail pointer to integer
    {self.pointer_name}_int := to_integer(unsigned({self.pointer_name}_i));

    {self.unshifted_assignments}

    {self.shifted_assignments}
  end process;

  {self.output_assignments}
""".removeprefix("\n").strip()

        def get(self):
            return self.item
        


class NaiveStoreOrderPerEntryLocalItems():
    class NaiveStoreOrderPerEntry(Signal2D):
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
                     shifted_stores = False,
                     shifted_both = False,
                     unshifted = False
                     ):
            
            bitwidth = config.store_queue_num_entries()
            number = config.load_queue_num_entries()

            if shifted_both:
                base_name = NAIVE_STORE_ORDER_PER_ENTRY_NAME
            elif shifted_stores:
                base_name = SHIFTED_STORES_NAIVE_STORE_ORDER_PER_ENTRY_NAME
            elif unshifted:
                base_name = UNSHIFTED_NAIVE_STORE_ORDER_PER_ENTRY_NAME
            else:
                raise RuntimeError("unclear store order signal")

            Signal2D.__init__(
                self,
                base_name=base_name,
                direction=Signal.Direction.INPUT,
                size=Signal.Size(
                    bitwidth=bitwidth,
                    number=number
                )
            )

class NaiveStoreOrderPerEntryBodyItems():
    class Body():

        def __init__(self, config : Config):
            needs_order_shift = False
            for group_orders in range(config.num_groups()):
                for order in config.group_store_order(group_orders):
                    if order > 0:
                        needs_order_shift = True
            
            if needs_order_shift:

                load_pointer_name = QUEUE_POINTER_NAME(QueueType.LOAD, QueuePointerType.TAIL)
                store_pointer_name = QUEUE_POINTER_NAME(QueueType.STORE, QueuePointerType.TAIL)

                case_inputs = ""
                num_cases = 0
                for i in range(config.num_groups()):
                    if config.group_num_loads(i) > 0:
                        case_inputs += f"""
    case_input({num_cases}) := {GROUP_INIT_TRANSFER_NAME}_{i}_i;
""".removeprefix("\n")
                        num_cases = num_cases + 1

                case_inputs = case_inputs.strip()

                cases = ""

                case_number = 0
                for i in range(config.num_groups()):
                    if config.group_num_loads(i) > 0:      
                        group_one_hot = one_hot(case_number, num_cases)
                        case_number = case_number + 1
                        cases += f"""
      when {group_one_hot} =>
""".removeprefix("\n")
                        for j, store_order in enumerate(config.group_store_order(i)):
                            if store_order > 0:
                                cases += f"""
        -- Ld {j} of group {i}'s store order
        {UNSHIFTED_NAIVE_STORE_ORDER_PER_ENTRY_NAME}({j}) <= {mask_until(store_order, config.store_queue_num_entries())};

""".removeprefix("\n")
                            else:
                                cases += f"""
        -- Ld {j} of group {i} has no preceding stores, use default value

""".removeprefix("\n")
                    else:
                        cases += f"""
      -- Group {i} has no loads

""".removeprefix("\n")

                cases += f"""
      -- defaults handled at top of process
      when others =>
        null;
""".removeprefix("\n")


                cases = cases.strip()

                unshifted_assignments = f"""
  {UNSHIFTED_NAIVE_STORE_ORDER_PER_ENTRY_NAME} <= (others => (others => '0'));

    {case_inputs}

    case
      case_input
    is
      {cases}
    end case;
""".strip()

                shifted = NAIVE_STORE_ORDER_PER_ENTRY_NAME
                shifted_stores = SHIFTED_STORES_NAIVE_STORE_ORDER_PER_ENTRY_NAME
                unshifted = UNSHIFTED_NAIVE_STORE_ORDER_PER_ENTRY_NAME
                shifted_assignments = f"""

      -- shift all the store orders based on the store queue pointer
      -- From Hailin's design, the circuit is better shifting based on
      -- one pointer at a time 
      for i in 0 to {config.load_queue_num_entries()} - 1 loop
        for j in 0 to {config.store_queue_num_entries()} - 1 loop
          col_idx := (j + {store_pointer_name}_int) mod {config.store_queue_num_entries()};

          -- assign shifted value based on store queue
          {shifted_stores}(i)(j) <= {unshifted}(i)(col_idx);
        end loop;
      end loop;

      -- shift all the store orders based on the load queue pointer
      for i in 0 to {config.load_queue_num_entries()} - 1 loop
        row_idx := (i + {load_pointer_name}_int) mod {config.load_queue_num_entries()};

        -- assign shifted value based on load queue
        {shifted}(i) <= {shifted_stores}(row_idx);
      end loop;
""".strip()

                output_assignments = ""

                for i in range(config.load_queue_num_entries()):
                    output_name = f"{NAIVE_STORE_ORDER_PER_ENTRY_NAME}_{i}_o"

                    # pad single digit output names
                    if i < 10:
                        output_name += " "


                    output_assignments += f"""
  {output_name} <= {NAIVE_STORE_ORDER_PER_ENTRY_NAME}({i});
""".removeprefix("\n")
            
                output_assignments = output_assignments.strip()

                self.item = f"""


  process(all)
    -- tail pointers as integers for indexing
    variable {load_pointer_name}_int, {store_pointer_name}_int : natural;

    -- where a location in the shifted order should read from
    variable row_idx, col_idx : natural;

    variable case_input : std_logic_vector({num_cases} - 1 downto 0);
  begin
    -- convert q tail pointers to integer
    {load_pointer_name}_int := to_integer(unsigned({load_pointer_name}_i));
    {store_pointer_name}_int := to_integer(unsigned({store_pointer_name}_i));

    {unshifted_assignments}

    {shifted_assignments}

  end process;


  {output_assignments}
""".removeprefix("\n").strip()
            else:
                self.item = f"""  
  -- Naive store orders are all zeros
  -- Since within each BB, no store ever precedes a load

""".removeprefix("\n")

                zeros = mask_until(0, config.store_queue_num_entries())
                for i in range(config.load_queue_num_entries()):
                    name = f"{NAIVE_STORE_ORDER_PER_ENTRY_NAME}_{i}_o"

                    # pad for <= alignment
                    if i < 10:
                        name += " "

                    self.item += f"""
  {name} <= {zeros};
""".removeprefix("\n")
                    
                self.item = self.item.strip()

        def get(self):
            return self.item

class GroupAllocatorPortItems():



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
                ),
                always_number=True
            )


    class QueueInputsComment(EntityComment):
        """
        RTL comment:
        
        -- Input signals from the (load/store) queue
        """
        def __init__(self, queue_type : QueueType):
            comment = f"""

    -- Input signals from the {queue_type.value} queue

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

    class PortIdxPerQueueEntryComment(EntityComment):
        """
        RTL comment:
            
        -- (Load/store) port index to write into each load queue entry.

        -- {number} signals, each {bitwidth} bit(s).

        -- Not one-hot.

        -- Absent is there is only 1 (load/store) port
        """
        def __init__(
                self, 
                config : Config,
                queue_type : QueueType
                ):

            match queue_type:
                case QueueType.LOAD:
                    number = config.load_queue_num_entries()
                    bitwidth = config.load_ports_idx_bitwidth()
                case QueueType.STORE:
                    number = config.store_queue_num_entries()
                    bitwidth = config.store_ports_idx_bitwidth()

            comment = f"""

    -- {queue_type.value} port index to write into each {queue_type.value} queue entry.
    -- {number} signals, each {bitwidth} bit(s).
    -- Not one-hot.
    -- Absent if there is only one {queue_type.value} port

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

        Absent is there is only 1 (load/store) port
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
                    bitwidth = config.store_ports_idx_bitwidth()
                    number = config.store_queue_num_entries()

            Signal.__init__(
                self,
                base_name=PORT_INDEX_PER_ENTRY_NAME(queue_type),
                direction=Signal.Direction.OUTPUT,
                size=Signal.Size(
                    bitwidth=bitwidth,
                    number=number
                ),
                always_vector=True
            )

    class NaiveStoreOrderPerEntryComment(EntityComment):
        """
        RTL comment:
            
        -- Store order per load queue entry

        -- {config.load_queue_num_entries()} signals, each {config.store_queue_num_entries()} bit(s).

        -- One per entry in the load queue, with 1 bit per entry in the store queue.

        -- The order of the memory operations, read from the ROM, 

        -- has been shifted to generate this.

        -- It is naive, however, as 1s for already allocated stores are not present.
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
    -- has been shifted to generate this.
    -- It is naive, however, as 1s for already allocated stores are not present.

""".removeprefix("\n")
            EntityComment.__init__(
                self,
                comment
            )



    class NaiveStoreOrderPerEntry(Signal):
        """
        Output
        
        Bitwidth = N

        Number = N

        Whether the stores in the store queue and ahead or behind
        each specific entry in the load queue.
         
        There is one signal per entry in the load queue,
        and 1 bit per entry in the store queue.
        
        The order of the memory operations, read from the ROM,
        has been shifted to generate this.
        
        This is done based on the store queue and load queue pointers.

        It is naive, however, as 1s for already allocated stores are not present.
        """

        def __init__(self, 
                     config : Config,
                     ):

            Signal.__init__(
                self,
                base_name=NAIVE_STORE_ORDER_PER_ENTRY_NAME,
                direction=Signal.Direction.OUTPUT,
                size=Signal.Size(
                    bitwidth=config.store_queue_num_entries(), 
                    number=config.load_queue_num_entries()
                )
            )


class GroupAllocatorBodyItems():
    class GroupHandshakingInst(Instantiation):
        def __init__(self, config : Config, prefix):

            p = GroupAllocatorPortItems()
            l = GroupAllocatorLocalItems()
            c = InstCxnType

            d = Signal.Direction

            si = SimpleInstantiation
            port_items = [
                si(ds.GroupInitValid(config), c.INPUT),
                si(p.GroupInitReady(config), c.OUTPUT),

                si(p.QueuePointer(config, QueueType.LOAD, QueuePointerType.TAIL), c.INPUT),
                si(p.QueuePointer(config, QueueType.LOAD, QueuePointerType.HEAD), c.INPUT),
                si(p.QueueIsEmpty(QueueType.LOAD), c.INPUT),

                si(p.QueuePointer(config, QueueType.STORE, QueuePointerType.TAIL), c.INPUT),
                si(p.QueuePointer(config, QueueType.STORE, QueuePointerType.HEAD), c.INPUT),
                si(p.QueueIsEmpty(QueueType.STORE), c.INPUT),


                si(l.GroupInitTransfer(config, d.OUTPUT), c.LOCAL)
            ]


            Instantiation.__init__(
                self,
                name=GROUP_HANDSHAKING_NAME,
                prefix=prefix,
                port_items=port_items
            )

    class PortIdxPerEntryInst(Instantiation):
        def __init__(self, config : Config, queue_type : QueueType, prefix):

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
                prefix=prefix,
                port_items=port_items
            )

    class NaiveStoreOrderPerEntryInst(Instantiation):
        def __init__(self, config : Config, prefix):

            ga_l = GroupAllocatorLocalItems()
            ga_p = GroupAllocatorPortItems()
            c = InstCxnType
            d = Signal.Direction

            si = SimpleInstantiation
            port_items = [
                si(ga_l.GroupInitTransfer(config, d.INPUT), c.LOCAL),

                si(ga_p.QueuePointer(
                    config, 
                    QueueType.LOAD, 
                    QueuePointerType.TAIL),
                    c.INPUT
                ),

                si(ga_p.QueuePointer(
                    config, 
                    QueueType.STORE, 
                    QueuePointerType.TAIL),
                    c.INPUT
                ),

                si(ga_p.NaiveStoreOrderPerEntry(
                    config),
                    c.OUTPUT
                )
            ]

            Instantiation.__init__(
                self,
                name=NAIVE_STORE_ORDER_PER_ENTRY_NAME,
                prefix=prefix,
                port_items=port_items
            )

    class NumNewQueueEntriesInst(Instantiation):
        def __init__(self, config : Config, queue_type : QueueType, prefix):

            ga_l = GroupAllocatorLocalItems()
            c = InstCxnType
            d = Signal.Direction

            si = SimpleInstantiation
            port_items = [
                si(ga_l.GroupInitTransfer(config, d.INPUT), c.LOCAL),

                si(ga_l.NumNewQueueEntries(
                    config, 
                    queue_type,
                    d.OUTPUT),
                    c.LOCAL
                )
            ]

            Instantiation.__init__(
                self,
                name=NUM_NEW_QUEUE_ENTRIES_NAME(queue_type),
                prefix=prefix,
                port_items=port_items
            )

    class WriteEnableInst(Instantiation):
        def __init__(self, config : Config, queue_type : QueueType, prefix):

            ga_l = GroupAllocatorLocalItems()
            ga_p = GroupAllocatorPortItems()

            c = InstCxnType
            d = Signal.Direction

            si = SimpleInstantiation
            port_items = [
                si(ga_l.NumNewQueueEntries(config, queue_type, d.INPUT), c.LOCAL),

                si(ga_p.QueuePointer(config, queue_type, QueuePointerType.TAIL), c.INPUT),
                si(ga_p.QueueWriteEnable(
                    config, 
                    queue_type),
                    c.OUTPUT
                )
            ]

            Instantiation.__init__(
                self,
                name=WRITE_ENABLE_NAME(queue_type),
                prefix=prefix,
                port_items=port_items
            )
    
    class NumNewEntriesAssignment():
        def get(self):
            return f"""
    -- the "number of new entries" signals are local, 
    -- since they are used to generate the write enable signals
    --
    -- Here we drive the outputs with them
    {NUM_NEW_QUEUE_ENTRIES_NAME(QueueType.LOAD)}_o <= {NUM_NEW_QUEUE_ENTRIES_NAME(QueueType.LOAD)};
    {NUM_NEW_QUEUE_ENTRIES_NAME(QueueType.STORE)}_o <= {NUM_NEW_QUEUE_ENTRIES_NAME(QueueType.STORE)};

""".removeprefix("\n")
                    
class GroupAllocatorLocalItems():
    class NumNewQueueEntries(Signal):
        """       
        Bitwidth = N

        Number = 1

        Number of (load/store) queue entries to allocate,
        which is output directly to the (load/store) queue.

        Non-handshaked signal. 
        
        Used by the (load/store) queue to update its tail pointer, 
        using update logic appropriate to circular buffers.
        
        There is a single "number of (load/store) queue entries to allocate" signal,
        and its bitwidth is equal to the bitwidth of the (load/store) queue pointers, 
        to allow easy arithmetic between then.
        """
        def __init__(self, 
                     config : Config,
                     queue_type : QueueType,
                     direction : Signal.Direction = None
                     ):
            match queue_type:
                case QueueType.LOAD:
                    bitwidth = config.load_queue_idx_bitwidth()
                case QueueType.STORE:
                    bitwidth = config.store_queue_idx_bitwidth()

            Signal.__init__(
                self,
                base_name=NUM_NEW_QUEUE_ENTRIES_NAME(queue_type),
                direction=direction,
                size=Signal.Size(
                    bitwidth=bitwidth,
                    number=1
                )
            )

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
                ),
                always_number=True
            )
            
