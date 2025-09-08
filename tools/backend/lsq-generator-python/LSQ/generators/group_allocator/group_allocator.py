from LSQ.context import VHDLContext
from LSQ.signals import Logic, LogicArray, LogicVec, LogicVecArray
from LSQ.utils import MaskLess
from LSQ.config import Config

from LSQ.entity import Entity, Architecture, Signal, RTLComment, DeclarativeUnit, Instantiation, SimpleInstantiation, InstCxnType

from LSQ.utils import QueueType, QueuePointerType

from LSQ.rtl_signal_names import *

import LSQ.declarative_signals as ds


from LSQ.generators.group_allocator.group_handshaking import get_group_handshaking
from LSQ.generators.group_allocator.num_new_entries import get_num_new_entries
from LSQ.generators.group_allocator.naive_store_order_per_entry import get_naive_store_order_per_entry

from LSQ.generators.group_allocator.write_enables import get_write_enables

from LSQ.generators.group_allocator.port_index_per_entry import get_port_index_per_entry


def get_group_allocator(config, parent):
    declaration = GroupAllocatorDeclarative(config, parent)

    unit = Entity(declaration).get() + Architecture(declaration).get()

    ga_name = declaration.name()

    dependencies = get_group_handshaking(config, ga_name)

    dependencies += get_num_new_entries(config, QueueType.LOAD, ga_name)
    dependencies += get_num_new_entries(config, QueueType.STORE, ga_name)

    dependencies += get_write_enables(config, QueueType.LOAD, ga_name)
    dependencies += get_write_enables(config, QueueType.STORE, ga_name)

    if config.load_ports_num() > 1:
        dependencies += get_port_index_per_entry(config, QueueType.LOAD, ga_name)

    if config.store_ports_num() > 1:
        dependencies += get_port_index_per_entry(config, QueueType.STORE, ga_name)

    dependencies += get_naive_store_order_per_entry(config, ga_name)

    return dependencies + unit

class GroupAllocatorDeclarative(DeclarativeUnit):
    def __init__(self, config : Config, parent_name):
        """
        Declarative definition of the Group Allocator.

        First all the signals in its entity port mapping are listed.

        Then all its local signals.

        Then finally a list of instantiations of sub-units,
        which contain actual RTL logic.

        The group allocator contains only 2 assignments of actual RTL:
        Driving the output "number of new queue entries" signals
        with the local "number of new queue entries" signals.

        Args:
            config(Config) : Config containing the parameterization of this LSQ: queue sizes, bitwidths, etc.
            unqiue_name(str) : Unique name from the netlist printer
        """


        self.top_level_comment = f"""
-- Group Allocator
""".strip()
        
        self.initialize_name(
            parent_name=parent_name,
            unit_name=GROUP_ALLOCATOR_NAME
        )

        # Specify port entity items
        self.entity_port_items = self.get_port_items()

        # Specify local items
        self.local_items = 


        self.body = [
            GroupHandshakingInst(config, self.name()),


            *([PortIdxPerEntryInst(config, QueueType.LOAD, self.name())] \
                  if config.load_ports_num() > 1 else []),

            *([PortIdxPerEntryInst(config, QueueType.STORE, self.name())] \
                  if config.store_ports_num() > 1 else []),

            NumNewQueueEntriesInst(config, QueueType.LOAD, self.name()),
            NumNewQueueEntriesInst(config, QueueType.STORE, self.name()),

            NaiveStoreOrderPerEntryInst(config, self.name()),

            WriteEnableInst(config, QueueType.LOAD, self.name()),
            WriteEnableInst(config, QueueType.STORE, self.name()),

            NumNewEntriesAssignment()
        ]

    def get_port_items(self, config):
        LOAD_QUEUE = QueueType.LOAD
        STORE_QUEUE = QueueType.STORE

        d = Signal.Direction

        return [
            ds.Reset(),
            ds.Clock(),


            RTLComment(f"""
                          
    -- Group init channels from the dataflow circuit
    -- {config.num_groups()} control channel(s),
    -- One for each group of memory operations.


""".removeprefix("\n").removesuffix("\n")),
            ds.GroupInitValid(config),
            ds.GroupInitReady(config),



            RTLComment(f"""
                          
    -- Input signals from the load queue

"""),
            ds.QueuePointer(
                config, 
                QueueType.LOAD, 
                QueuePointerType.HEAD,
                d.INPUT
                ),

            ds.QueuePointer(
                config, 
                QueueType.LOAD, 
                QueuePointerType.TAIL,
                d.INPUT
                ),
            ds.QueueIsEmpty(
                QueueType.LOAD,
                d.INPUT
                ),




            RTLComment(f"""
                          
    -- Input signals from the store queue

"""),            
            ds.QueuePointer(
                config, 
                QueueType.STORE, 
                QueuePointerType.HEAD,
                d.INPUT
                ),
            ds.QueuePointer(
                config, 
                QueueType.STORE, 
                QueuePointerType.TAIL,
                d.INPUT
                ),
            ds.QueueIsEmpty(
                QueueType.STORE,
                d.INPUT
            ),




            RTLComment(f"""
                          
    -- Load queue write enable signals
    -- {config.queue_num_entries(LOAD_QUEUE)} signals, one for each queue entry.

"""),
            ds.QueueWriteEnable(
                config, 
                QueueType.LOAD,
                d.OUTPUT
                ),




            RTLComment(f"""
                          
    -- Number of new load queue entries to allocate.
    -- Used by the load queue to update its tail pointer.
    -- Bitwidth equal to the load queue pointer bitwidth.

"""),
            ds.NumNewEntries(
                config, 
                QueueType.LOAD, 
                d.OUTPUT
                ),



            RTLComment(f"""
                          
    -- Load port index to write into each load queue entry.
    -- {config.queue_num_entries(LOAD_QUEUE)} signals, each {config.ports_idx_bitwidth(LOAD_QUEUE)} bit(s).
    -- Not one-hot.
    -- Absent if there is only one load port

"""),
            ds.PortIdxPerEntry(
                config, 
                QueueType.LOAD,
                d.OUTPUT
                ),




            RTLComment(f"""
                          
    -- Store queue write enable signals
    -- {config.queue_num_entries(STORE_QUEUE)} signals, one for each queue entry.

"""),
            ds.QueueWriteEnable(
                config, 
                QueueType.STORE,
                d.OUTPUT
                ),




            RTLComment(f"""
                          
    -- Number of new store queue entries to allocate.
    -- Used by the store queue to update its tail pointer.
    -- Bitwidth equal to the store queue pointer bitwidth.

"""),
            ds.NumNewEntries(
                config, 
                QueueType.STORE, 
                d.OUTPUT
                ),




            RTLComment(f"""
                          
    -- Store port index to write into each store queue entry.
    -- {config.queue_num_entries(STORE_QUEUE)} signals, each {config.ports_idx_bitwidth(STORE_QUEUE)} bit(s).
    -- Not one-hot.
    -- Absent if there is only one store port

"""),
            ds.PortIdxPerEntry(
                config, 
                QueueType.STORE, 
                d.OUTPUT
                ),
    

            RTLComment(f"""

    -- Store order per load queue entry
    -- {config.queue_num_entries(LOAD_QUEUE)} signals, each {config.queue_num_entries(STORE_QUEUE)} bit(s).
    -- One per entry in the load queue, with 1 bit per entry in the store queue.
    -- The order of the memory operations, read from the ROM, 
    -- has been shifted to generate this.
    -- It is naive, however, as 1s for already allocated stores are not present.

"""),
            ds.NaiveStoreOrderPerEntry(
                config, 
                d.OUTPUT
                )
        ]

    def get_local_items(self, config):
        return [
            RTLComment(f"""
  -- One-hot group allocation signals
  -- Used as inputs to muxes, 
  -- to set the rest of the values to allocate
""".removeprefix("\n")),
            ds.GroupInitTransfer(config),

            RTLComment(f"""
                       
  -- The number of entries to allocate into the load queue       
""".removeprefix("\n")),
            ds.NumNewEntries(
                config, 
                QueueType.LOAD
                ),
            RTLComment(f"""
  -- The number of entries to allocate into the store queue       
""".removeprefix("\n")),
            ds.NumNewEntries(
                config, 
                QueueType.STORE
                )
        ]

def instantiate(
    config:             Config,
    lsq_name:           str,
    ctx:                VHDLContext,
    group_init_valid_i: LogicArray,
    group_init_ready_o: LogicArray,
    ldq_tail_i:         LogicVec,
    ldq_head_i:         LogicVec,
    ldq_empty_i:        Logic,
    stq_tail_i:         LogicVec,
    stq_head_i:         LogicVec,
    stq_empty_i:        Logic,
    ldq_wen_o:          LogicArray,
    num_loads_o:        LogicVec,
    ldq_port_idx_o:     LogicVecArray,
    stq_wen_o:          LogicArray,
    num_stores_o:       LogicVec,
    stq_port_idx_o:     LogicVecArray,
    ga_ls_order_o:      LogicVecArray
) -> str:


    module_name = f"{lsq_name}_{GROUP_ALLOCATOR_NAME}_unit"

    arch = ctx.get_current_indent(
    ) + f'{module_name} : entity work.{module_name}\n'
    ctx.tabLevel += 1
    arch += ctx.get_current_indent() + f'port map(\n'
    ctx.tabLevel += 1

    arch += ctx.get_current_indent() + f'rst => rst,\n'
    arch += ctx.get_current_indent() + f'clk => clk,\n'

    for i in range(0, config.num_groups()):
        arch += ctx.get_current_indent() + \
            f'group_init_valid_{i}_i => {group_init_valid_i.getNameRead(i)},\n'
    for i in range(0, config.num_groups()):
        arch += ctx.get_current_indent() + \
            f'group_init_ready_{i}_o => {group_init_ready_o.getNameWrite(i)},\n'

    arch += ctx.get_current_indent() + \
        f'ldq_tail_i => {ldq_tail_i.getNameRead()},\n'
    arch += ctx.get_current_indent() + \
        f'ldq_head_i => {ldq_head_i.getNameRead()},\n'
    arch += ctx.get_current_indent() + \
        f'ldq_empty_i => {ldq_empty_i.getNameRead()},\n'

    arch += ctx.get_current_indent() + \
        f'stq_tail_i => {stq_tail_i.getNameRead()},\n'
    arch += ctx.get_current_indent() + \
        f'stq_head_i => {stq_head_i.getNameRead()},\n'
    arch += ctx.get_current_indent() + \
        f'stq_empty_i => {stq_empty_i.getNameRead()},\n'

    for i in range(0, config.numLdqEntries):
        arch += ctx.get_current_indent() + \
            f'ldq_wen_{i}_o => {ldq_wen_o.getNameWrite(i)},\n'
    arch += ctx.get_current_indent() + \
        f'num_loads_o => {num_loads_o.getNameWrite()},\n'
    if (config.ldpAddrW > 0):
        for i in range(0, config.numLdqEntries):
            arch += ctx.get_current_indent() + \
                f'ldq_port_idx_{i}_o => {ldq_port_idx_o.getNameWrite(i)},\n'

    for i in range(0, config.numStqEntries):
        arch += ctx.get_current_indent() + \
            f'stq_wen_{i}_o => {stq_wen_o.getNameWrite(i)},\n'
    if (config.stpAddrW > 0):
        for i in range(0, config.numStqEntries):
            arch += ctx.get_current_indent() + \
                f'stq_port_idx_{i}_o => {stq_port_idx_o.getNameWrite(i)},\n'

    for i in range(0, config.numLdqEntries):
        arch += ctx.get_current_indent() + \
            f'ga_ls_order_{i}_o => {ga_ls_order_o.getNameWrite(i)},\n'

    arch += ctx.get_current_indent() + \
        f'num_stores_o => {num_stores_o.getNameWrite()}\n'

    ctx.tabLevel -= 1
    arch += ctx.get_current_indent() + f');\n'
    ctx.tabLevel -= 1
    return arch


class GroupHandshakingInst(Instantiation):
    def __init__(self, config : Config, parent):

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
            unit_name=GROUP_HANDSHAKING_NAME,
            parent=parent,
            port_items=port_items,
            comment=f"""
  -- Generate the group init ready and group init transfer signals,
  -- used to allocated into the load queue and store queue.
""".strip()
        )


class PortIdxPerEntryInst(Instantiation):
    def __init__(self, config : Config, queue_type : QueueType, parent):

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
            unit_name=PORT_INDEX_PER_ENTRY_NAME(queue_type),
            parent=parent,
            port_items=port_items,
            comment=f"""
  -- Generate the {queue_type.value} port index per {queue_type.value} queue entry
  -- aligned with the {queue_type.value} queue.
""".strip()
        )

class NaiveStoreOrderPerEntryInst(Instantiation):
    def __init__(self, config : Config, parent):

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
            unit_name=NAIVE_STORE_ORDER_PER_ENTRY_NAME,
            parent=parent,
            port_items=port_items,
            comment=f"""
  -- Generate the naive store order per load queue entry, aligned with the load queue.
  -- Naive as it only contains information
  -- about the order of stores in the group being allocated
""".strip()
        )

class NumNewQueueEntriesInst(Instantiation):
    def __init__(self, config : Config, queue_type : QueueType, parent):

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
                ds.NumNewEntries(
                    config, 
                    queue_type,
                    d.OUTPUT
                ),
                c.LOCAL
            )
        ]

        Instantiation.__init__(
            self,
            unit_name=NUM_NEW_ENTRIES_NAME(queue_type),
            parent=parent,
            port_items=port_items,
            comment=f"""
  -- Generate the number of new {queue_type.value} entries to allocate into the {queue_type.value} queue
  -- Mux of ROMs based on group init transfer signals
""".strip()
        )

class WriteEnableInst(Instantiation):
    def __init__(self, config : Config, queue_type : QueueType, parent):

        c = InstCxnType
        d = Signal.Direction

        si = SimpleInstantiation

        port_items = [
            si(
                ds.NumNewEntries(
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
            unit_name=WRITE_ENABLE_NAME(queue_type),
            parent=parent,
            port_items=port_items,
            comment=f"""
  -- Generate write enable signals for the {queue_type.value} queue
  -- Shifted so they are aligned with its entries.
""".strip()
        )

class NumNewEntriesAssignment():
    def get(self):
        return f"""
    -- the "number of new entries" signals are local, 
    -- since they are used to generate the write enable signals

    -- Here we drive the outputs with them
    {NUM_NEW_ENTRIES_NAME(QueueType.LOAD)}_o <= {NUM_NEW_ENTRIES_NAME(QueueType.LOAD)};
    {NUM_NEW_ENTRIES_NAME(QueueType.STORE)}_o <= {NUM_NEW_ENTRIES_NAME(QueueType.STORE)};

""".removeprefix("\n")
                    