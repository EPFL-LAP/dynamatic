from LSQ.context import VHDLContext
from LSQ.signals import Logic, LogicArray, LogicVec, LogicVecArray
from LSQ.utils import MaskLess
from LSQ.config import Config

from LSQ.entity import Entity, Architecture, Signal, RTLComment, DeclarativeUnit

from LSQ.utils import QueueType, QueuePointerType
# from LSQ.architecture import Architecture

from LSQ.rtl_signal_names import *

import LSQ.declarative_signals as ds

from LSQ.generators.group_allocator.group_handshaking import GroupHandshaking
from LSQ.generators.group_allocator.num_new_entries import NumNewEntries
from LSQ.generators.group_allocator.naive_store_order_per_entry import get_naive_store_order_per_entry

from LSQ.generators.group_allocator.write_enables import get_write_enables

from LSQ.generators.group_allocator.group_allocator_items import \
    (
        GroupAllocatorBodyItems,
        PortIdxPerEntryBodyItems,
        PortIdxPerEntryLocalItems,
    )

class PortIdxPerEntryDecl(DeclarativeUnit):
    def __init__(self, config : Config, queue_type : QueueType, parent):
        self.top_level_comment = f"""
-- {queue_type.value} Port Index per {queue_type.value} Queue Entry
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

        self.unit_name = PORT_INDEX_PER_ENTRY_NAME(queue_type)
        self.parent = parent


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

        l = PortIdxPerEntryLocalItems()

        self.local_items = [
            l.PortIdxPerQueueEntry(config, queue_type, shifted=False),
            l.PortIdxPerQueueEntry(config, queue_type, shifted=True)
        ]

        b = PortIdxPerEntryBodyItems()
        self.body = [
            b.Body(config, queue_type)
        ]

class GroupAllocatorDeclarative(DeclarativeUnit):
    def __init__(self, config : Config, parent):
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
        
        self.parent = parent
        self.unit_name = GROUP_ALLOCATOR_NAME


        LOAD_QUEUE = QueueType.LOAD
        STORE_QUEUE = QueueType.STORE


        d = Signal.Direction

        #################################
        ## Declarative Description
        ## of Group Allocators 
        ## Entity Port Map Signals
        #################################

        self.entity_port_items = [
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
            ds.NumNewQueueEntries(
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
            ds.NumNewQueueEntries(
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


        self.local_items = [
            ds.GroupInitTransfer(config),
            ds.NumNewQueueEntries(
                config, 
                QueueType.LOAD
                ),
            ds.NumNewQueueEntries(
                config, 
                QueueType.STORE
                )
        ]

        b = GroupAllocatorBodyItems

        self.body = [
            b.GroupHandshakingInst(config, self.name()),


            *([b.PortIdxPerEntryInst(config, QueueType.LOAD, self.name())] \
                  if config.load_ports_num() > 1 else []),

            *([b.PortIdxPerEntryInst(config, QueueType.STORE, self.name())] \
                  if config.store_ports_num() > 1 else []),

            b.NumNewQueueEntriesInst(config, QueueType.LOAD, self.name()),
            b.NumNewQueueEntriesInst(config, QueueType.STORE, self.name()),

            b.NaiveStoreOrderPerEntryInst(config, self.name()),

            b.WriteEnableInst(config, QueueType.LOAD, self.name()),
            b.WriteEnableInst(config, QueueType.STORE, self.name()),

            b.NumNewEntriesAssignment()
        ]

class GroupAllocator:
    def print_dec(self, dec):
        entity = Entity(dec)
        arch = Architecture(dec)

        return entity.get() + arch.get()

    def __init__(
        self,
        name: str,
        configs: Config
    ):
        """
        Group Allocator

        Models a group allocator for a Load-Store Queue (LSQ) system.

        This class encapsulates the logic for generating a VHDL module that allocates
        space for groups of memory operations (loads and stores) in the load queue and 
        the store queue.

        Parameters:
            name    : Base name of the group allocator.
            suffix  : Suffix appended to the entity name.
            configs : configuration generated from JSON

        Instance Variable:
            self.module_name = name + suffix : Entity and architecture identifier

        Example:
            ga = GroupAllocator(
                    name="config_0_core", 
                    suffix="_ga", 
                    configs=configs
                )

            # You can later generate VHDL entity and architecture by
            #     ga.generate(...)
            # You can later instantiate VHDL entity by
            #     ga.instantiate(...)
        """

        self.name = name
        self.configs = configs
        self.lsq_name = name

        self.module_name = f"{self.lsq_name}_{GROUP_ALLOCATOR_NAME}_unit"

    def generate(self, path_rtl, config : Config) -> None:
        """
        Generates the VHDL 'entity' and 'architecture' sections for a group allocator.

        Parameters:
            path_rtl    : Output directory for VHDL files.

        Output:
            Appends the 'entity' and 'architecture' definitions
            to the .vhd file at <path_rtl>/<self.name>.vhd.
            Entity and architecture use the identifier: <self.module_name>

        Example (Group Allocator):
            ga.generate(path_rtl)

            produces in rtl/config_0_core.vhd:

            entity config_0_core_ga is
                port(
                    rst           : in  std_logic;
                    clk           : in  std_logic;
                    ...
                );
            end entity;

            architecture arch of config_0_core_ga is
                -- signals generated here
            begin
                -- group allocator logic here
            end architecture;

        """

        ga_decl = GroupAllocatorDeclarative(config, self.lsq_name)

        unit = self.print_dec(GroupHandshaking(config, ga_decl.name()))

        unit += self.print_dec(NumNewEntries(config, QueueType.LOAD, ga_decl.name()))
        unit += self.print_dec(NumNewEntries(config, QueueType.STORE, ga_decl.name()))

        unit += get_write_enables(config, QueueType.LOAD, ga_decl.name())
        unit += get_write_enables(config, QueueType.STORE, ga_decl.name())

        if config.load_ports_num() > 1:
            unit += self.print_dec(PortIdxPerEntryDecl(config, QueueType.LOAD, ga_decl.name()))

        if config.store_ports_num() > 1:
            unit += self.print_dec(PortIdxPerEntryDecl(config, QueueType.STORE, ga_decl.name()))

        unit += get_naive_store_order_per_entry(config, ga_decl.name())

        unit += self.print_dec(ga_decl)

        # Write to the file
        with open(f'{path_rtl}/{self.name}.vhd', 'a') as file:
            file.write(unit)


    def instantiate(
        self,
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
        """
        Group Allocator Instantiation

        Creates the VHDL port mapping for the group allocator entity.

        Parameters:
            ctx                  : VHDLContext for code generation state.
            group_init_valid_i   : Group Allocator handshake valid signal
            group_init_ready_o   : Group Allocator handshake ready signal
            ldq_tail_i           : Load queue tail
            ldq_head_i           : Load queue head
            ldq_empty_i          : (boolean) load queue empty
            stq_tail_i           : Store queue tail
            stq_head_i           : Store queue head
            stq_empty_i          : (boolean) store queue empty
            ldq_wen_o            : Load queue write enable
            num_loads_o          : The number of loads
            ldq_port_idx_o       : Load queue port index
            stq_wen_o            : Store queue write enable
            num_stores_o         : The number of stores
            stq_port_idx_o       : Store queue port index
            ga_ls_order_o        : Group Allocator load-store order matrix

        Returns:
            VHDL instantiation string for inclusion in the architecture body.

        Example:
            arch += ga.instantiate(
                ctx,
                group_init_valid_i = group_init_valid_i,
                group_init_ready_o = group_init_ready_o,
                ldq_tail_i         = ldq_tail,
                ldq_head_i         = ldq_head,
                ldq_empty_i        = ldq_empty,
                stq_tail_i         = stq_tail,
                stq_head_i         = stq_head,
                stq_empty_i        = stq_empty,
                ldq_wen_o          = ldq_wen,
                num_loads_o        = num_loads,
                ldq_port_idx_o     = ldq_port_idx,
                stq_wen_o          = stq_wen,
                num_stores_o       = num_stores,
                stq_port_idx_o     = stq_port_idx,
                ga_ls_order_o      = ga_ls_order
            )

            This generates, inside 'config_0_core.vhd' and under the 'architecture config_0_core', the following instantiation

            architecture arch of config_0_core is
                signal ...
            begin
                ...
                config_0_core_ga : entity work.config_0_core_ga
                    port map(
                        rst => rst,
                        clk => clk,
                        group_init_valid_0_i => group_init_valid_0_i,
                        group_init_ready_0_o => group_init_ready_0_o,
                        ldq_tail_i => ldq_tail_q,
                        ldq_head_i => ldq_head_q,
                        ldq_empty_i => ldq_empty,
                        stq_tail_i => stq_tail_q,
                        stq_head_i => stq_head_q,
                        stq_empty_i => stq_empty,
                        ldq_wen_0_o => ldq_wen_0,
                        ldq_wen_1_o => ldq_wen_1,
                        num_loads_o => num_loads,
                        ldq_port_idx_0_o => ldq_port_idx_0_d,
                        ldq_port_idx_1_o => ldq_port_idx_1_d,
                        stq_wen_0_o => stq_wen_0,
                        stq_wen_1_o => stq_wen_1,
                        stq_port_idx_0_o => stq_port_idx_0_d,
                        stq_port_idx_1_o => stq_port_idx_1_d,
                        ga_ls_order_0_o => ga_ls_order_0,
                        ga_ls_order_1_o => ga_ls_order_1,
                        num_stores_o => num_stores
                    );
                ...
            end architecture;
        """

        arch = ctx.get_current_indent(
        ) + f'{self.module_name} : entity work.{self.module_name}\n'
        ctx.tabLevel += 1
        arch += ctx.get_current_indent() + f'port map(\n'
        ctx.tabLevel += 1

        arch += ctx.get_current_indent() + f'rst => rst,\n'
        arch += ctx.get_current_indent() + f'clk => clk,\n'

        for i in range(0, self.configs.num_groups()):
            arch += ctx.get_current_indent() + \
                f'group_init_valid_{i}_i => {group_init_valid_i.getNameRead(i)},\n'
        for i in range(0, self.configs.num_groups()):
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

        for i in range(0, self.configs.numLdqEntries):
            arch += ctx.get_current_indent() + \
                f'ldq_wen_{i}_o => {ldq_wen_o.getNameWrite(i)},\n'
        arch += ctx.get_current_indent() + \
            f'num_loads_o => {num_loads_o.getNameWrite()},\n'
        if (self.configs.ldpAddrW > 0):
            for i in range(0, self.configs.numLdqEntries):
                arch += ctx.get_current_indent() + \
                    f'ldq_port_idx_{i}_o => {ldq_port_idx_o.getNameWrite(i)},\n'

        for i in range(0, self.configs.numStqEntries):
            arch += ctx.get_current_indent() + \
                f'stq_wen_{i}_o => {stq_wen_o.getNameWrite(i)},\n'
        if (self.configs.stpAddrW > 0):
            for i in range(0, self.configs.numStqEntries):
                arch += ctx.get_current_indent() + \
                    f'stq_port_idx_{i}_o => {stq_port_idx_o.getNameWrite(i)},\n'

        for i in range(0, self.configs.numLdqEntries):
            arch += ctx.get_current_indent() + \
                f'ga_ls_order_{i}_o => {ga_ls_order_o.getNameWrite(i)},\n'

        arch += ctx.get_current_indent() + \
            f'num_stores_o => {num_stores_o.getNameWrite()}\n'

        ctx.tabLevel -= 1
        arch += ctx.get_current_indent() + f');\n'
        ctx.tabLevel -= 1
        return arch
