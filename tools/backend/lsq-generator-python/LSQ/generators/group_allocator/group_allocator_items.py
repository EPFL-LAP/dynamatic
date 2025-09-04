from LSQ.entity import Signal, RTLComment, Instantiation, SimpleInstantiation, InstCxnType, Signal2D
from LSQ.config import Config

from LSQ.rtl_signal_names import *

from LSQ.utils import bin_string, one_hot, mask_until

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
            new_entries = NUM_NEW_ENTRIES_NAME(queue_type)

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
                        idx_bin = bin_string(idx, self.idx_bitwidth)

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


class GroupAllocatorBodyItems():
    class GroupHandshakingInst(Instantiation):
        def __init__(self, config : Config, prefix):

            c = InstCxnType

            d = Signal.Direction

            si = SimpleInstantiation
            port_items = [
                si(
                    ds.GroupInitValid(
                        config
                    ), 
                    c.INPUT
                ),
                si(
                    ds.GroupInitReady(
                        config,
                    ), 
                    c.OUTPUT
                ),

                si(
                    ds.QueuePointer(
                        config, 
                        QueueType.LOAD, 
                        QueuePointerType.TAIL,
                        d.INPUT
                    ), 
                    c.INPUT
                ),
                si(
                    ds.QueuePointer(
                        config, 
                        QueueType.LOAD, 
                        QueuePointerType.HEAD,
                        d.INPUT
                        ), 
                    c.INPUT
                ),
                si(
                    ds.QueueIsEmpty(
                        QueueType.LOAD,
                        d.INPUT
                    ), 
                c.INPUT
                ),

                si(
                    ds.QueuePointer(
                        config, 
                        QueueType.STORE, 
                        QueuePointerType.TAIL,
                        d.INPUT
                    ), 
                    c.INPUT
                ),

                si(
                    ds.QueuePointer(
                        config, 
                        QueueType.STORE, 
                        QueuePointerType.HEAD,
                        d.INPUT
                    ), 
                    c.INPUT
                ),
                
                si(
                    ds.QueueIsEmpty(
                        QueueType.STORE,
                        d.INPUT
                    ), 
                    c.INPUT
                ),



                si(
                    ds.GroupInitTransfer(
                        config, 
                        d.OUTPUT
                    ), 
                    c.LOCAL
                )
            ]


            Instantiation.__init__(
                self,
                name=GROUP_HANDSHAKING_NAME,
                prefix=prefix,
                port_items=port_items
            )

    class PortIdxPerEntryInst(Instantiation):
        def __init__(self, config : Config, queue_type : QueueType, prefix):

            c = InstCxnType
            d = Signal.Direction

            si = SimpleInstantiation
            port_items = [
                si(
                    ds.GroupInitTransfer(
                        config, 
                        d.INPUT
                    ), 
                    c.LOCAL
                ),

                si(
                    ds.QueuePointer(
                    config, 
                    queue_type, 
                    QueuePointerType.TAIL,
                    d.INPUT
                    ),
                    c.INPUT
                ),

                si(ds.PortIdxPerEntry(
                    config, 
                    queue_type,
                    d.OUTPUT
                    ),
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

            c = InstCxnType
            d = Signal.Direction

            si = SimpleInstantiation
            port_items = [
                si(ds.GroupInitTransfer(
                    config, 
                    d.INPUT
                    ), 
                    c.LOCAL
                ),

                si(ds.QueuePointer(
                    config, 
                    QueueType.LOAD, 
                    QueuePointerType.TAIL,
                    d.INPUT
                    ),
                    c.INPUT
                ),

                si(ds.QueuePointer(
                    config, 
                    QueueType.STORE, 
                    QueuePointerType.TAIL,
                    d.INPUT
                    ),
                    c.INPUT
                ),

                si(ds.NaiveStoreOrderPerEntry(
                    config,
                    d.OUTPUT
                    ),
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

            c = InstCxnType
            d = Signal.Direction

            si = SimpleInstantiation
            port_items = [
                si(
                    ds.GroupInitTransfer(
                        config, 
                        d.INPUT
                    ), 
                    c.LOCAL
                ),

                si(
                    ds.NumNewQueueEntries(
                        config, 
                        queue_type,
                        d.OUTPUT
                    ),
                    c.LOCAL
                )
            ]

            Instantiation.__init__(
                self,
                name=NUM_NEW_ENTRIES_NAME(queue_type),
                prefix=prefix,
                port_items=port_items
            )

    class WriteEnableInst(Instantiation):
        def __init__(self, config : Config, queue_type : QueueType, prefix):

            c = InstCxnType
            d = Signal.Direction

            si = SimpleInstantiation

            port_items = [
                si(
                    ds.NumNewQueueEntries(
                        config, 
                        queue_type, 
                        d.INPUT
                    ), 
                    c.LOCAL
                ),

                si(
                    ds.QueuePointer(
                        config, 
                        queue_type, 
                        QueuePointerType.TAIL,
                        d.INPUT
                    ), 
                    c.INPUT
                    ),
                si(
                    ds.QueueWriteEnable(
                        config, 
                        queue_type,
                        d.OUTPUT
                    ),
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
    {NUM_NEW_ENTRIES_NAME(QueueType.LOAD)}_o <= {NUM_NEW_ENTRIES_NAME(QueueType.LOAD)};
    {NUM_NEW_ENTRIES_NAME(QueueType.STORE)}_o <= {NUM_NEW_ENTRIES_NAME(QueueType.STORE)};

""".removeprefix("\n")
                    