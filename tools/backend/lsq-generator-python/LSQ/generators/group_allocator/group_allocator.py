from LSQ.context import VHDLContext
from LSQ.signals import Logic, LogicArray, LogicVec, LogicVecArray
from LSQ.utils import MaskLess
from LSQ.config import Config

from LSQ.entity import Entity, Architecture, Signal, RTLComment, DeclarativeUnit, Instantiation, SimpleInstantiation, InstCxnType

from LSQ.utils import QueueType, QueuePointerType
# from LSQ.architecture import Architecture

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

from LSQ.context import VHDLContext
from LSQ.signals import Logic, LogicArray, LogicVec, LogicVecArray
from LSQ.operators import Op, WrapSub_old, Mux1HROM, CyclicLeftShift, CyclicPriorityMasking
from LSQ.utils import MaskLess
from LSQ.config import Config


class GroupAllocator:
    def __init__(
        self,
        name: str,
        suffix: str,
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
        self.module_name = name + suffix

    def generate(self, path_rtl) -> None:
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

        # ctx: VHDLContext for code generation state.
        # When we generate VHDL entity and architecture, we can use this context as a local variable.
        # We only need to get the context as a parameter when we instantiate the module.
        # It saves all information we need when we generate VHDL entity and architecture code.
        ctx = VHDLContext()

        ctx.tabLevel = 1
        ctx.tempCount = 0
        ctx.signalInitString = ''
        ctx.portInitString = '\tport(\n\t\trst : in std_logic;\n\t\tclk : in std_logic'
        ctx.regInitString = '\tprocess (clk, rst) is\n' + '\tbegin\n'
        arch = ''

        # IOs
        group_init_valid_i = LogicArray(
            ctx, 'group_init_valid', 'i', self.configs.numGroups)
        group_init_ready_o = LogicArray(
            ctx, 'group_init_ready', 'o', self.configs.numGroups)

        ldq_tail_i = LogicVec(ctx, 'ldq_tail', 'i', self.configs.ldqAddrW)
        ldq_head_i = LogicVec(ctx, 'ldq_head', 'i', self.configs.ldqAddrW)
        ldq_empty_i = Logic(ctx, 'ldq_empty', 'i')

        stq_tail_i = LogicVec(ctx, 'stq_tail', 'i', self.configs.stqAddrW)
        stq_head_i = LogicVec(ctx, 'stq_head', 'i', self.configs.stqAddrW)
        stq_empty_i = Logic(ctx, 'stq_empty', 'i')

        ldq_wen_o = LogicArray(ctx, 'ldq_wen', 'o', self.configs.numLdqEntries)
        num_loads_o = LogicVec(ctx, 'num_loads', 'o', self.configs.ldqAddrW)
        num_loads = LogicVec(ctx, 'num_loads', 'w', self.configs.ldqAddrW)
        if (self.configs.ldpAddrW > 0):
            ldq_port_idx_o = LogicVecArray(
                ctx, 'ldq_port_idx', 'o', self.configs.numLdqEntries, self.configs.ldpAddrW)

        stq_wen_o = LogicArray(ctx, 'stq_wen', 'o', self.configs.numStqEntries)
        num_stores_o = LogicVec(ctx, 'num_stores', 'o', self.configs.stqAddrW)
        num_stores = LogicVec(ctx, 'num_stores', 'w', self.configs.stqAddrW)
        if (self.configs.stpAddrW > 0):
            stq_port_idx_o = LogicVecArray(
                ctx, 'stq_port_idx', 'o', self.configs.numStqEntries, self.configs.stpAddrW)

        ga_ls_order_o = LogicVecArray(
            ctx, 'ga_ls_order', 'o', self.configs.numLdqEntries, self.configs.numStqEntries)

        # The number of empty load and store is calculated with cyclic subtraction.
        # If the empty signal is high, then set the number to max value.
        # loads_sub = LogicVec(ctx, 'loads_sub', 'w', self.configs.ldqAddrW)
        # stores_sub = LogicVec(ctx, 'stores_sub', 'w', self.configs.stqAddrW)
        # empty_loads = LogicVec(ctx, 'empty_loads', 'w',
        #                        self.configs.emptyLdAddrW)
        # empty_stores = LogicVec(ctx, 'empty_stores', 'w',
        #                         self.configs.emptyStAddrW)

        # arch += WrapSub_old(ctx, loads_sub, ldq_head_i,
        #                 ldq_tail_i, self.configs.numLdqEntries)
        # arch += WrapSub_old(ctx, stores_sub, stq_head_i,
        #                 stq_tail_i, self.configs.numStqEntries)

        # arch += Op(ctx, empty_loads, self.configs.numLdqEntries, 'when', ldq_empty_i, 'else',
        #            '(', '\'0\'', '&', loads_sub, ')')
        # arch += Op(ctx, empty_stores, self.configs.numStqEntries, 'when', stq_empty_i, 'else',
        #            '(', '\'0\'', '&', stores_sub, ')')

        # Generate handshake signals
        # group_init_ready = LogicArray(
        #     ctx, 'group_init_ready', 'w', self.configs.numGroups)
        group_init_hs = LogicArray(
            ctx, 'group_init_hs', 'w', self.configs.numGroups)

        # for i in range(0, self.configs.numGroups):
        #     arch += Op(ctx, group_init_ready[i],
        #                '\'1\'', 'when',
        #                '(', empty_loads,  '>=', (
        #         self.configs.gaNumLoads[i], self.configs.emptyLdAddrW),  ')', 'and',
        #         '(', empty_stores, '>=', (
        #         self.configs.gaNumStores[i], self.configs.emptyStAddrW), ')',
        #         'else', '\'0\'')

        # if (self.configs.gaMulti):
        #     group_init_and = LogicArray(
        #         ctx, 'group_init_and', 'w', self.configs.numGroups)
        #     ga_rr_mask = LogicVec(ctx, 'ga_rr_mask', 'r',
        #                           self.configs.numGroups)
        #     ga_rr_mask.regInit()
        #     for i in range(0, self.configs.numGroups):
        #         arch += Op(ctx, group_init_and[i],
        #                    group_init_ready[i], 'and', group_init_valid_i[i])
        #         arch += Op(ctx, group_init_ready_o[i], group_init_hs[i])
        #     arch += CyclicPriorityMasking(ctx, group_init_hs,
        #                                   group_init_and, ga_rr_mask)
        #     for i in range(0, self.configs.numGroups):
        #         arch += Op(ctx, (ga_rr_mask, (i+1) %
        #                          self.configs.numGroups), (group_init_hs, i))
        # else:
        #     for i in range(0, self.configs.numGroups):
        #         arch += Op(ctx, group_init_ready_o[i], group_init_ready[i])
        #         arch += Op(ctx, group_init_hs[i],
        #                    group_init_ready[i], 'and', group_init_valid_i[i])

        # ROM value
        if (self.configs.ldpAddrW > 0):
            ldq_port_idx_rom = LogicVecArray(
                ctx, 'ldq_port_idx_rom', 'w', self.configs.numLdqEntries, self.configs.ldpAddrW)
        if (self.configs.stpAddrW > 0):
            stq_port_idx_rom = LogicVecArray(
                ctx, 'stq_port_idx_rom', 'w', self.configs.numStqEntries, self.configs.stpAddrW)
        ga_ls_order_rom = LogicVecArray(
            ctx, 'ga_ls_order_rom', 'w', self.configs.numLdqEntries, self.configs.numStqEntries)
        ga_ls_order_temp = LogicVecArray(
            ctx, 'ga_ls_order_temp', 'w', self.configs.numLdqEntries, self.configs.numStqEntries)
        # if (self.configs.ldpAddrW > 0):
        #     arch += Mux1HROM(ctx, ldq_port_idx_rom,
        #                      self.configs.gaLdPortIdx, group_init_hs)
        # if (self.configs.stpAddrW > 0):
        #     arch += Mux1HROM(ctx, stq_port_idx_rom,
        #                      self.configs.gaStPortIdx, group_init_hs)
        arch += Mux1HROM(ctx, ga_ls_order_rom, self.configs.gaLdOrder,
                         group_init_hs, MaskLess)
        # arch += Mux1HROM(ctx, num_loads,
                        #  self.configs.gaNumLoads, group_init_hs)
        # arch += Mux1HROM(ctx, num_stores,
                        #  self.configs.gaNumStores, group_init_hs)
        arch += Op(ctx, num_loads_o, num_loads)
        arch += Op(ctx, num_stores_o, num_stores)

        ldq_wen_unshifted = LogicArray(
            ctx, 'ldq_wen_unshifted', 'w', self.configs.numLdqEntries)
        stq_wen_unshifted = LogicArray(
            ctx, 'stq_wen_unshifted', 'w', self.configs.numStqEntries)
        # for i in range(0, self.configs.numLdqEntries):
        #     arch += Op(ctx, ldq_wen_unshifted[i],
        #                '\'1\'', 'when',
        #                num_loads, '>', (i, self.configs.ldqAddrW),
        #                'else', '\'0\''
        #                )
        for i in range(0, self.configs.numStqEntries):
            arch += Op(ctx, stq_wen_unshifted[i],
                       '\'1\'', 'when',
                       num_stores, '>', (i, self.configs.stqAddrW),
                       'else', '\'0\''
                       )

        # Shift the arrays
        # if (self.configs.ldpAddrW > 0):
        #     arch += CyclicLeftShift(ctx, ldq_port_idx_o,
        #                             ldq_port_idx_rom, ldq_tail_i)
        # if (self.configs.stpAddrW > 0):
        #     arch += CyclicLeftShift(ctx, stq_port_idx_o,
        #                             stq_port_idx_rom, stq_tail_i)
        # arch += CyclicLeftShift(ctx, ldq_wen_o, ldq_wen_unshifted, ldq_tail_i)
        arch += CyclicLeftShift(ctx, stq_wen_o, stq_wen_unshifted, stq_tail_i)
        for i in range(0, self.configs.numLdqEntries):
            arch += CyclicLeftShift(ctx,
                                    ga_ls_order_temp[i], ga_ls_order_rom[i], stq_tail_i)
        arch += CyclicLeftShift(ctx, ga_ls_order_o,
                                ga_ls_order_temp, ldq_tail_i)

        ######   Write To File  ######
        ctx.portInitString += '\n\t);'
        if (self.configs.gaMulti):
            ctx.regInitString += '\tend process;\n'
        else:
            ctx.regInitString = ''


        # Write to the file
        with open(f'{path_rtl}/{self.name}.vhd', 'a') as file:
            file.write(get_group_handshaking(self.configs, self.module_name))
            if (self.configs.ldpAddrW > 0):
                file.write(get_port_index_per_entry(self.configs, QueueType.LOAD, self.module_name))
            if (self.configs.stpAddrW > 0):
                file.write(get_port_index_per_entry(self.configs, QueueType.STORE, self.module_name))
            file.write(get_num_new_entries(self.configs, QueueType.LOAD, self.module_name))
            file.write(get_num_new_entries(self.configs, QueueType.STORE, self.module_name))

            file.write(get_write_enables(self.configs, QueueType.LOAD, self.module_name))

            file.write('\n\n')
            file.write(ctx.library)
            file.write(f'entity {self.module_name} is\n')
            file.write(ctx.portInitString)
            file.write('\nend entity;\n\n')
            file.write(f'architecture arch of {self.module_name} is\n')
            file.write(ctx.signalInitString)
            file.write('begin\n')
            file.write(GroupHandshakingInst(self.configs, self.module_name).get())
            if (self.configs.ldpAddrW > 0):
                file.write(PortIdxPerEntryInst(self.configs, QueueType.LOAD, self.module_name).get())
            if (self.configs.stpAddrW > 0):
                file.write(PortIdxPerEntryInst(self.configs, QueueType.STORE, self.module_name).get())
            file.write(NumNewQueueEntriesInst(self.configs, QueueType.LOAD, self.module_name).get())
            file.write(NumNewQueueEntriesInst(self.configs, QueueType.STORE, self.module_name).get())

            file.write(WriteEnableInst(self.configs, QueueType.LOAD, self.module_name).get())

            file.write(arch + '\n')
            file.write(ctx.regInitString + 'end architecture;\n')

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


        self.local_items = [
            ds.GroupInitTransfer(config),
            ds.NumNewEntries(
                config, 
                QueueType.LOAD
                ),
            ds.NumNewEntries(
                config, 
                QueueType.STORE
                )
        ]


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
                    