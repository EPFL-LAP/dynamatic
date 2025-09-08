
from LSQ.config import Config

from LSQ.rtl_signal_names import *

from LSQ.entity import Signal, Signal2D, Instantiation, SimpleInstantiation, InstCxnType, DeclarativeUnit, Entity, Architecture, RTLComment
import LSQ.declarative_signals as ds

from LSQ.utils import one_hot, mask_until

from collections import defaultdict

from LSQ.generators.barrel_shifter import get_barrel_shifter, ShiftDirection

def get_naive_store_order_per_entry(config, parent):
    
    # if a store never precedes a load inside of a BB
    # the naive store orders are all zeros
    # so first check if we need anything at all
    trivial = True
    for group_orders in range(config.num_groups()):
        for order in config.group_store_order(group_orders):
            # if any store order is non-zero,
            # we need the roms + muxs + shifter
            if order > 0:
                trivial = False

    declaration = NaiveStoreOrderPerEntryDecl(config, parent, trivial)
    unit = Entity(declaration).get() + Architecture(declaration).get()

    # if orders are all zeros, there are no dependencies
    if trivial:
        return unit

    # else we need the barrel shifters
    barrel_shifters = _get_barrel_shifters(config, declaration)

    return barrel_shifters + unit

def _get_barrel_shifters(config, declaration):

    d = Signal.Direction

    h_barrel_shift = get_barrel_shifter(
        declaration.name(),
        "barrel_shift_hoz",
        ds.QueuePointer(
                config, 
                QueueType.STORE,
                QueuePointerType.TAIL,
                d.INPUT
        ),
        NaiveStoreOrderPerEntry(
            config,
            unshifted=True,
            direction = d.INPUT
        ),
        NaiveStoreOrderPerEntry(
            config,
            shifted_stores=True,
            direction = d.OUTPUT
        ),
        ShiftDirection.HORIZONTAL
    )

    v_barrel_shift = get_barrel_shifter(
        declaration.name(),
        "barrel_shift_vrt",
        ds.QueuePointer(
                config, 
                QueueType.LOAD,
                QueuePointerType.TAIL,
                d.INPUT
        ),
        NaiveStoreOrderPerEntry(
            config,
            shifted_stores=True,
            direction = d.INPUT
        ),
        NaiveStoreOrderPerEntry(
            config,
            shifted_both=True,
            direction = d.OUTPUT
        ),
        ShiftDirection.VERTICAL
    )

    return h_barrel_shift + v_barrel_shift

class NaiveStoreOrderPerEntryDecl(DeclarativeUnit):
    def __init__(self, config: Config, parent, trivial):
        self.top_level_comment = f"""
-- Naive Store Order Per Load Queue Entry Unit
-- Sub-unit of the Group Allocator.
--
-- Generates the naive store orders.
--
-- There is one naive store order per entry in the load queue.
-- Each store order has 1 bit per entry in the store queue.
--
-- For the a load entry's store order, if the value is 1
-- the store must happen before that load.
--
-- These are the naive store orders, and so only contain the order
-- between stores and loads in the group currently being allocated
--
-- Information on already allocated stores is added to this later.
""".strip()

        self.unit_name = NAIVE_STORE_ORDER_PER_ENTRY_NAME

        self.parent = parent


        d = Signal.Direction
    
        self.entity_port_items = [
            RTLComment(f"""
                       
  -- The store order to allocate into the load queue
  -- depends on which group is being allocated,
  -- so we pass in the "group init transfer" signals

""".removeprefix("\n")),
            ds.GroupInitTransfer(
                config, 
                d.INPUT
            ),
            RTLComment(f"""
                       
  -- The store order needs to be aligned with both
  -- the load queue and the store queue to allocate properly
  -- so both tail pointers are needed

""".removeprefix("\n")),
            ds.QueuePointer(
                config, 
                QueueType.LOAD, 
                QueuePointerType.TAIL,
                d.INPUT
            ),
            ds.QueuePointer(
                config, 
                QueueType.STORE, 
                QueuePointerType.TAIL,
                d.INPUT
            ),
            RTLComment(f"""
                       
  -- Output:
  -- Naive store orders, indicating which stores preced which loads
  -- but only for the loads and store in the group being allocated
                       
""".removeprefix("\n")),
            ds.NaiveStoreOrderPerEntry(
                config,
                d.OUTPUT
            )
        ]


        # if there are no stores before any loads
        # inside of a BB
        # the naive store orders are always zero
        if trivial:
            self.local_items = []

            self.body = [
                Trivial(config)
            ]
        else:
            self.local_items = [
                RTLComment(f"""

  -- Masked store orders, unmuxed and unshifted
  -- used as input to muxes controlled by group init transfers                       
""".removeprefix("\n")),
                MaskedStoreOrder(config),
                            RTLComment(f"""

  -- Outputs of the muxes: unshifted store orders                  
""".removeprefix("\n")),
                NaiveStoreOrderPerEntry(
                    config, 
                    shifted_both=True
                ),
                RTLComment(f"""

  -- Outputs of the first barrel shifter
  -- The bits themselves now align with the store queue           
""".removeprefix("\n")),
                NaiveStoreOrderPerEntry(
                    config, 
                    shifted_stores=True
                ),
                RTLComment(f"""

  -- Outputs of the second barrel shifter
  -- The array items now align with the load queue        
""".removeprefix("\n")),
                NaiveStoreOrderPerEntry(
                    config, 
                    unshifted=True
                )
            ]

            self.body = [
                Muxes(config),
                HorizontalBarrelShiftInstantiation(config, self.name()),
                VerticalBarrelShiftInstantiation(config, self.name())
            ]
    

def ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"

class Trivial():
    def __init__(self, config):
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

class Muxes():
    def __init__(self, config : Config):
        unshifted = UNSHIFTED_NAIVE_STORE_ORDER_PER_ENTRY_NAME

        # First, we need to generate masked store orders
        # by combining the non-zero orders with the transfer signals
        # if the store orders are zero, we don't need to do anything

        store_order_width = config.queue_num_entries(QueueType.STORE)
        zero_store_order = mask_until(0, store_order_width)

        self.item = ""

        # to_mux is a dictionary of lists
        # each key represents a load queue entry
        # the list is the inputs to the mux for that entry
        to_mux = defaultdict(list)

        # iterate over all the groups
        for i in range(config.num_groups()):

            # transfer name is constant per group
            transfer_name = f"{GROUP_INIT_TRANSFER_NAME}_{i}_i"

            # we only mask the non-zero store orders
            # so we need to remember where to write the masked signal to
            masked_store_order_index = 0

            # for each store order in the group
            for j, store_order_int in enumerate(config.group_store_order(i)):
                # if non-zero
                if store_order_int != 0:
                    # the store order corresponds to load queue j, so the key is j
                    # we store the group and the index to read from
                    to_mux[j].append((i, masked_store_order_index))

                    # convert the integer store order to a signal
                    # e.g. if the store order is 3 
                    # (meaning the first 3 stores in this BB are ahead of this load)
                    # and there are 5 entries in the store queue
                    # the signal is 00111
                    store_order = mask_until(store_order_int, store_order_width)

                    assign_to = f"group_{i}_masked_naive_store_order({masked_store_order_index})"

                    masked_store_order_index = masked_store_order_index + 1

                    self.item += f"""
  -- Load {j} of group {i} has preceding stores inside of its BB
  -- and therefore a non-zero store order
  -- Mask it so we can use it an OR mux
  {assign_to} <= {store_order} when {transfer_name} else {zero_store_order};

""".removeprefix("\n")

        max_num_loads_in_one_group = max(to_mux.keys)
        for i in range(max_num_loads_in_one_group):
            # unshifted store order variable
            assign_to = f"{unshifted}({i})"

            # pad name if less than 10
            # to maintain alignment
            if i < 10:
                assign_to = assign_to + " "
        
            # get mux inputs
            # or an empty list
            mux_inputs = to_mux.get(i, [])

            # No mux at all, since no non-zero store orders
            if len(mux_inputs) == 0:
                self.item += f"""
  -- No group has a non-zero store order for load {i}
  {assign_to} <= (others => '0');
""".removeprefix("\n")
            # No mux at all, since there is only 1 non-zero store order
            elif len(mux_inputs) == 1:
                group, index = mux_inputs[0]
                self.item += f"""
-- Only group {group} has a non-zero store order for load {i}
{assign_to} <= group_{group}_masked_naive_store_order({index});
""".removeprefix("\n")
                
            # Here we build an actual mux

            else:
                one_hots = ""
                # for every input except the last input
                # add a store order plus an OR
                for group, index in mux_inputs[:-1]:
                    one_hots += f"""
    group_{group}_masked_naive_store_order({index})
      or
""".removeprefix("\n")
                one_hots = one_hots.strip()

                # add the last assignment plus a semi colon
                final_group, final_index = mux_inputs[-1]
                final_assignment = f"""
    group_{final_group}_masked_naive_store_order({final_index});
""".strip()

                # combine store orders, the ORs,
                # and the final store order with the ;
                self.item += f"""
  -- More than one group has a non-zero store order for load {i}
  -- We mux them using OR, as they have been one-hot-masked
  {assign_to} <= 
    {one_hots}
    {final_assignment}

""".removeprefix("\n")
                    
        queue_entries =config.queue_num_entries(QueueType.LOAD)
        self.item += f"""
  remaining_entries : for i in {max_num_loads_in_one_group} to {queue_entries} - 1generate
    -- No group has a non-zero store order for load i
    {unshifted}(i) <= (others => '0');
  end generate
""".removeprefix("\n")

    def get(self):
        return self.item
        
# Declarative local signal only used by the naive store order unit
class MaskedStoreOrder():
    """
    3D signal
    
    Local 2D vector, per group, storing the
    naive store order per queue entry.
        
    Bitwidth is equal to the number of store queue entriews
    Number is equal to the number of load queue entries
    """

    def __init__(self, config : Config):
        self.config = config

    def get_local_item(self):
        item = ""

        bitwidth = self.config.queue_num_entries(QueueType.STORE)
        for i in range(self.config.num_groups()):
            non_zero_store_orders = 0
            for store_order in self.config.group_store_order(i):
                if store_order != 0:
                    non_zero_store_orders = non_zero_store_orders + 1
            
            if non_zero_store_orders > 0:
                name = f"group_{i}_masked_naive_store_order".ljust(35)
                item += f"""
  signal {name} : data_array({non_zero_store_orders} - 1 downto 0)({bitwidth} - 1 downto 0);
""".removeprefix("\n")
        
        return item

# Declarative local signal only used by the naive store order unit
class NaiveStoreOrderPerEntry(Signal2D):
    """
    Bitwidth = N
    Number = M

    Local 2D input vector storing the 
    (unshifted/shifted for store queue/shifted for both queues) 
    store order per queue entry
        
    Bitwidth is equal to the number of store queue entries
    Number is equal to the number of load queue entries
    """
    def __init__(self, 
                    config : Config,
                    shifted_stores = False,
                    shifted_both = False,
                    unshifted = False,
                    # used as input/output of barrel shifters
                    direction = None
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
            direction=direction,
            size=Signal.Size(
                bitwidth=bitwidth,
                number=number
            )
        )

# Horizontal barrel shifter aligns naive store order
# to Store Queue
class HorizontalBarrelShiftInstantiation(Instantiation):
    def __init__(self, config : Config, parent):
        si = SimpleInstantiation
        d = Signal.Direction
        c = InstCxnType

        # Barrel shifters are generated uniquely
        # per use
        #
        # and so their input and output signals
        # match the signals pass to the instantiation
        port_items = [
            # shift the store orders horizontally
            # based on the store queue tail
            si(
                ds.QueuePointer(
                    config, 
                    QueueType.STORE,
                    QueuePointerType.TAIL,
                    d.INPUT
                    ),
                c.INPUT
            ),
            # pass in the unshifted store orders
            si(
                NaiveStoreOrderPerEntry(
                    config,
                    unshifted=True,
                    direction = d.INPUT
                ),
                c.LOCAL
            ),
            # pass out the store orders
            # shifted for the store queue
            si(
                NaiveStoreOrderPerEntry(
                    config,
                    shifted_stores=True,
                    direction = d.OUTPUT
                ),
                c.LOCAL
            )
        ]

        Instantiation.__init__(
            self,
            "barrel_shift_hoz",
            parent,
            port_items
        )


# Vertical barrel shifter aligns naive store order
# to Load Queue
class VerticalBarrelShiftInstantiation(Instantiation):
    def __init__(self, config : Config, parent):
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
                    QueueType.LOAD,
                    QueuePointerType.TAIL,
                    d.INPUT
                    ),
                c.INPUT
            ),
            # pass in the store orders
            # shifted for the store queue
            si(
                NaiveStoreOrderPerEntry(
                    config,
                    shifted_stores=True,
                    direction = d.INPUT
                ),
                c.LOCAL
            ),
            # pass out the store orders
            # shifted for both queues
            si(
                NaiveStoreOrderPerEntry(
                    config,
                    shifted_both=True,
                    direction = d.OUTPUT
                ),
                c.LOCAL
            )
        ]

        Instantiation.__init__(
            self,
            "barrel_shift_vrt",
            parent,
            port_items
        )


