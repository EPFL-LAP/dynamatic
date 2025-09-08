
from LSQ.config import Config

from LSQ.rtl_signal_names import *

from LSQ.entity import Signal, Signal2D, Instantiation, SimpleInstantiation, InstCxnType, DeclarativeUnit, Entity, Architecture, RTLComment
import LSQ.declarative_signals as ds

from LSQ.utils import bin_string

from collections import defaultdict

from LSQ.generators.barrel_shifter import get_barrel_shifter, ShiftDirection

def get_port_index_per_entry(config, queue_type : QueueType, parent_name):
    
    declaration = PortIndexPerEntryDecl(config, parent_name, queue_type)
    unit = Entity(declaration).get() + Architecture(declaration).get()

    barrel_shifter = _get_barrel_shifter(config, declaration, queue_type)

    return barrel_shifter + unit

class PortIndexPerEntryDecl(DeclarativeUnit):
    def __init__(self, config: Config, parent_name, queue_type : QueueType):
        self.top_level_comment = f"""
-- {queue_type.value.capitalize()} Port Index per {queue_type.value.capitalize()} Queue Entry
-- Sub-unit of the Group Allocator.
--
-- Generates the {queue_type.value} port index per {queue_type.value} entry.
--
-- Each {queue_type.value} queue entry must know which {queue_type.value} port
-- it exchanges data with.
--
-- First, the port indices are selected based on 
-- which group is currently being allocated.
--
-- Then they are shifted into the correct place for the internal circular buffer,
-- based on the {queue_type.value} tail pointer.
""".strip()

        self.initialize_name(
            parent_name=parent_name, 
            unit_name=PORT_INDEX_PER_ENTRY_NAME(queue_type)
            )

        d = Signal.Direction
    
        self.entity_port_items = [
            ds.GroupInitTransfer(
                config, 
                d.INPUT
            ),
            ds.QueuePointer(
                config, 
                queue_type, 
                QueuePointerType.TAIL,
                d.INPUT
            ),
            ds.PortIdxPerEntry(
                config, 
                queue_type,
                d.OUTPUT
            )
        ]


        self.local_items = [
            MaskedPortIndex(config, queue_type),
            PortIdxPerEntry(config, queue_type, shifted=False),
            PortIdxPerEntry(config, queue_type, shifted=True),
        ]


        self.body = [
            Muxes(config, queue_type),
            BarrelShiftInstantiation(config, self.name(), queue_type),
            OutputAssignments(config, queue_type)
        ]

class Muxes():
    def __init__(self, config : Config, queue_type : QueueType):
        unshifted = UNSHIFTED_PORT_INDEX_PER_ENTRY_NAME(queue_type)

        # First, we need to generate masked port indices
        # by combining the port indices with the transfer signals

        self.item = ""


        idx_bitwidth = config.ports_idx_bitwidth(queue_type)

        zero_bin = bin_string(0, idx_bitwidth)

        # to_mux is a dictionary of lists
        # each key represents a queue entry
        # the list is the inputs to the mux for that entry
        to_mux = defaultdict(list)

        # iterate over all the groups
        for i in range(config.num_groups()):
            match queue_type:
                case QueueType.LOAD:
                    number = config.group_num_loads(i)
                    ports = config.group_load_ports(i)
                case QueueType.STORE:
                    number = config.group_num_stores(i)
                    ports = config.group_store_ports(i)


            # transfer name is constant per group
            transfer_name = f"{GROUP_INIT_TRANSFER_NAME}_i({i})"

            # assignment name is constant per group
            assign_to = MASKED_PORT_INDEX_PER_ENTRY_NAME(i, queue_type)

            # for each (load/store) in the group
            # we don't enumerate the ports themselves
            # since currently there is dummy zeros in them
            for j in range(number):
                # store that group i has a load j
                to_mux[j].append((i, j))

                port_index_int = ports[j]
                port_index = bin_string(port_index_int, idx_bitwidth)

                self.item += f"""
  -- {queue_type.value.capitalize()} {j} of group {i} has port index {port_index_int}
  {assign_to}({j}) <= {port_index} when {transfer_name} else {zero_bin};

""".removeprefix("\n")

        # if the max number of (load/stores) in any basic block is N
        # and the number of queue entries is M
        # only loads up to N are printed in this for loop
        # so first we find N
        # (N + 1) to M are handled in the generate statement below
        max_num_in_one_group = max(to_mux.keys()) + 1

        for i in range(max_num_in_one_group):
            # unshifted port index variable
            assign_to = f"{unshifted}({i})"

            # pad name if less than 10
            # to maintain alignment
            if i < 10:
                assign_to = assign_to + " "
        
            # get mux inputs
            # or an empty list
            mux_inputs = to_mux.get(i, [])

            # port indices must be contiguous
            assert(len(mux_inputs) > 0)

            # No mux, since only 1 group has this many loads
            if len(mux_inputs) == 1:

                group, index = mux_inputs[0]

                # port indices must be contiguous
                assert(index == i)

                masked = MASKED_PORT_INDEX_PER_ENTRY_NAME(group, queue_type)
                self.item += f"""
-- Only group {group} has a {queue_type.value} {i}
{assign_to} <= {masked}({i});
""".removeprefix("\n")
                
            # Here we build an actual mux
            else:
                one_hots = ""
                # for every input except the last input
                # add a store order plus an OR
                for group, index in mux_inputs[:-1]:
                    # port indices must be contiguous
                    assert (index == i)

                    masked = MASKED_PORT_INDEX_PER_ENTRY_NAME(group, queue_type)
                    one_hots += f"""
    {masked}({i})
      or
""".removeprefix("\n")
                one_hots = one_hots.strip()

                # add the last assignment plus a semi colon
                final_group, final_index = mux_inputs[-1]
                final_masked = MASKED_PORT_INDEX_PER_ENTRY_NAME(final_group, queue_type)

                # port indices must be contiguous
                assert (final_index == i)

                final_assignment = f"""
    {final_masked}({final_index});
""".strip()

                # combine the port indices, the ORs,
                # and the final port index with the ;
                self.item += f"""
  -- More than one group has a {queue_type.value} {i}
  -- We mux their port indices using OR, as they have been one-hot-masked
  {assign_to} <= 
    {one_hots}
    {final_assignment}

""".removeprefix("\n")
                    
        queue_entries = config.queue_num_entries(queue_type)
        self.item += f"""
  -- No group has more than {max_num_in_one_group} {queue_type.value}(s)
  -- So we use a generate to set the port index for
  -- {queue_type.value} queue entry {max_num_in_one_group} to {queue_type.value} queue entry {queue_entries - 1} 
  -- to zero
  remaining_entries : for i in {max_num_in_one_group} to {queue_entries} - 1 generate
    -- No group has a {queue_type.value} i
    {unshifted}(i) <= (others => '0');
  end generate;

""".removeprefix("\n")

    def get(self):
        return self.item
    
class OutputAssignments():
    def __init__(self, config : Config, queue_type : QueueType):
        self.item = ""
        for i in range(config.queue_num_entries(queue_type)):
            assign_to = f"{PORT_INDEX_PER_ENTRY_NAME(queue_type)}_{i}_o"

            if i < 10:
                assign_to += " "

            self.item += f"""
  {assign_to} <= {PORT_INDEX_PER_ENTRY_NAME(queue_type)}({i});
""".removeprefix("\n")
            
    def get(self):
        return self.item

# Declarative local signal only used by the port index per entry unit
class MaskedPortIndex():
    """
    3D signal
    
    Local 2D vector, per group, storing the
    port index per queue entry.
        
    Bitwidth is the amount required to index a port
    Number is equal to the number of (loads/stores) in that group
    """

    def __init__(self, config : Config, queue_type : QueueType):
        self.config = config
        self.item  = ""

        bitwidth = self.config.ports_idx_bitwidth(queue_type)
        for i in range(self.config.num_groups()):
            match queue_type:
                case QueueType.LOAD:
                    number = self.config.group_num_loads(i)
                case QueueType.STORE:
                    number = self.config.group_num_stores(i)
            
            if number == 0:
                continue

            name = MASKED_PORT_INDEX_PER_ENTRY_NAME(i, queue_type).ljust(35)
            
            self.item += f"""
  signal {name} : data_array({number} - 1 downto 0)({bitwidth} - 1 downto 0);
""".removeprefix("\n")

    def get_local_item(self):
        return self.item


# Declarative local signal only used by the port index per entry unit
class PortIdxPerEntry(Signal2D):
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
                    shifted = False,
                    # input/output to the barrel shifters
                    direction : Signal.Direction = None
                    ):
        
        if shifted:
            base_name = PORT_INDEX_PER_ENTRY_NAME(queue_type)
        else:
            base_name = UNSHIFTED_PORT_INDEX_PER_ENTRY_NAME(queue_type)

        Signal2D.__init__(
            self,
            base_name=base_name,
            direction=direction,
            size=Signal.Size(
                bitwidth=config.ports_idx_bitwidth(queue_type),
                number=config.queue_num_entries(queue_type)
            )
        )


# Barrel shifter aligns port index per entry
# with the queue
class BarrelShiftInstantiation(Instantiation):
    def __init__(self, config : Config, parent, queue_type : QueueType):
        si = SimpleInstantiation
        d = Signal.Direction
        c = InstCxnType


        # Barrel shifters are generated uniquely
        # per use
        #
        # and so their input and output signals
        # match the signals pass to the instantiation
        port_items = [
            # shift the store orders vertically
            # based on the load queue tail
            si(
                ds.QueuePointer(
                    config, 
                    queue_type,
                    QueuePointerType.TAIL,
                    d.INPUT
                    ),
                c.INPUT
            ),
            # pass in the store orders
            # shifted for the store queue
            si(
                PortIdxPerEntry(
                    config,
                    queue_type,
                    shifted=False,
                    direction = d.INPUT
                ),
                c.LOCAL
            ),
            # pass out the store orders
            # shifted for both queues
            si(
                PortIdxPerEntry(
                    config,
                    queue_type,
                    shifted=True,
                    direction = d.OUTPUT
                ),
                c.LOCAL
            )
        ]

        Instantiation.__init__(
            self,
            "barrel_shift",
            parent,
            port_items,
            comment=f"""
  -- Shift the array items of the port index per entry
  -- Based on the {queue_type.value} queue tail pointer
  -- So that array item 0 moves to array item (tail pointer)
  -- Making the port indices aligned with the {queue_type.value} queue
""".strip()
        )


def _get_barrel_shifter(config, declaration, queue_type : QueueType):

    d = Signal.Direction

    v_barrel_shift = get_barrel_shifter(
        declaration.name(),
        "barrel_shift",
        ds.QueuePointer(
                config, 
                queue_type,
                QueuePointerType.TAIL,
                d.INPUT
        ),
        PortIdxPerEntry(
            config,
            queue_type,
            shifted=False,
            direction = d.INPUT
        ),
        PortIdxPerEntry(
            config,
            queue_type,
            shifted=True,
            direction = d.OUTPUT
        ),
        ShiftDirection.VERTICAL,
        comment=f"""
-- Barrel shifter for the port index per entry unit
-- Aligns port index with {queue_type.value} queue
""".strip()
    )

    return  v_barrel_shift
