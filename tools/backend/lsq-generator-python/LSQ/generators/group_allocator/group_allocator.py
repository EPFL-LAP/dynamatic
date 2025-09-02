from LSQ.context import VHDLContext
from LSQ.signals import Logic, LogicArray, LogicVec, LogicVecArray
from LSQ.operators import Op, WrapSub_old, Mux1HROM, CyclicLeftShift, CyclicPriorityMasking
from LSQ.utils import MaskLess
from LSQ.config import Config

from LSQ.entity import Entity, Architecture, Signal

from LSQ.utils import QueueType, QueuePointerType
# from LSQ.architecture import Architecture

from LSQ.rtl_signal_names import *

from LSQ.generators.group_allocator.group_allocator_items import \
    (
        GroupAllocatorPortItems, 
        GroupAllocatorBodyItems,
        GroupAllocatorLocalItems,
        GroupHandshakingLocalItems,
        GroupHandshakingBodyItems,
        PortIdxPerEntryBodyItems,
        PortIdxPerEntryLocalItems,
        NaiveStoreOrderPerEntryLocalItems,
        NaiveStoreOrderPerEntryBodyItems,
        NumNewQueueEntriesBody,
        WriteEnableLocalItems,
        WriteEnableBodyItems
    )

class WriteEnableDecl():
    def __init__(self, config : Config, queue_type : QueueType, prefix):
        self.name = WRITE_ENABLE_NAME(queue_type)
        self.prefix = prefix

        ga_p = GroupAllocatorPortItems()
        ga_l = GroupAllocatorLocalItems()

        d = Signal.Direction
        self.entity_port_items = [
            ga_l.NumNewQueueEntries(config, queue_type, d.INPUT),
            ga_p.QueuePointer(config, queue_type, QueuePointerType.TAIL),
            ga_p.QueueWriteEnable(config, queue_type)
        ]


        l = WriteEnableLocalItems()

        self.local_items = [
            l.WriteEnable(config, queue_type, shifted=False),
            l.WriteEnable(config, queue_type, shifted=True)
        ]

        b = WriteEnableBodyItems()

        self.body = [
            b.Body(config, queue_type)
        ]

class NumNewQueueEntriesDecl():
    def __init__(self, config : Config, queue_type : QueueType, prefix):
        self.name = NUM_NEW_QUEUE_ENTRIES_NAME(queue_type)
        self.prefix = prefix

        ga_l = GroupAllocatorLocalItems()

        d = Signal.Direction
        self.entity_port_items = [
            ga_l.GroupInitTransfer(config, d.INPUT),
            ga_l.NumNewQueueEntries(config, queue_type, direction=d.OUTPUT)
        ]

        self.local_items = []

        b = NumNewQueueEntriesBody()
        self.body = [
            b.Body(config, queue_type)
        ]

class NaiveStoreOrderPerEntryDecl():
    def __init__(self, config: Config, prefix):
        self.name = NAIVE_STORE_ORDER_PER_ENTRY_NAME

        self.prefix = prefix

        ga_p = GroupAllocatorPortItems()
        ga_l = GroupAllocatorLocalItems()

        d = Signal.Direction
    
        self.entity_port_items = [
            ga_l.GroupInitTransfer(config, d.INPUT),
            ga_p.QueuePointer(config, QueueType.LOAD, QueuePointerType.TAIL),
            ga_p.QueuePointer(config, QueueType.STORE, QueuePointerType.TAIL),
            ga_p.NaiveStoreOrderPerEntry(config)
        ]

        l = NaiveStoreOrderPerEntryLocalItems()

        self.local_items = [
            l.NaiveStoreOrderPerEntry(config, shifted=False),
            l.NaiveStoreOrderPerEntry(config, shifted=True)
        ]

        b = NaiveStoreOrderPerEntryBodyItems()
        self.body = [
            b.Body(config)
        ]

class PortIdxPerEntryDecl():
    def __init__(self, config : Config, queue_type : QueueType, prefix):
        self.name = PORT_INDEX_PER_ENTRY_NAME(queue_type)
        self.prefix = prefix

        ga_p = GroupAllocatorPortItems()
        ga_l = GroupAllocatorLocalItems()

        d = Signal.Direction
    
        self.entity_port_items = [
            ga_l.GroupInitTransfer(config, d.INPUT),
            ga_p.QueuePointer(config, queue_type, QueuePointerType.TAIL),
            ga_p.PortIdxPerQueueEntry(config, queue_type)
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

class GroupHandshakingDecl():
    def __init__(self, config : Config, prefix):
        self.name = GROUP_HANDSHAKING_NAME
        self.prefix = prefix

        ga_p = GroupAllocatorPortItems()

        ga_l = GroupAllocatorLocalItems()

        d = Signal.Direction
        self.entity_port_items = [
            ga_p.GroupInitValid(config),
            ga_p.GroupInitReady(config),

            ga_p.QueuePointer(config, QueueType.LOAD, QueuePointerType.TAIL),
            ga_p.QueuePointer(config, QueueType.LOAD, QueuePointerType.HEAD),
            ga_p.QueueIsEmpty(QueueType.LOAD),

            ga_p.QueuePointer(config, QueueType.STORE, QueuePointerType.TAIL),
            ga_p.QueuePointer(config, QueueType.STORE, QueuePointerType.HEAD),
            ga_p.QueueIsEmpty(QueueType.STORE),

            ga_l.GroupInitTransfer(config, d.OUTPUT)
        ]

        l = GroupHandshakingLocalItems()
        self.local_items = [
            l.NaiveNumEmptyEntries(config, QueueType.LOAD),
            l.NaiveNumEmptyEntries(config, QueueType.STORE),

            ga_p.GroupInitReady(config)
        ]

        b = GroupHandshakingBodyItems()

        self.body = [
            b.Body(config)
        ]

class GroupAllocatorDecl():
    def __init__(self, config : Config, prefix, subunit_prefix):
        p = GroupAllocatorPortItems()
        l = GroupAllocatorLocalItems()

        self.name = GROUP_ALLOCATOR_NAME
        self.prefix = prefix

        d = Signal.Direction

        self.entity_port_items = [
            p.Reset(),
            p.Clock(),

            p.GroupInitChannelComment(config),

            p.GroupInitValid(config),
            p.GroupInitReady(config),

            p.QueueInputsComment(queue_type=QueueType.LOAD),
            p.QueuePointer(config, QueueType.LOAD, QueuePointerType.HEAD),
            p.QueuePointer(config, QueueType.LOAD, QueuePointerType.TAIL),
            p.QueueIsEmpty(QueueType.LOAD),

            p.QueueInputsComment(queue_type=QueueType.STORE),
            p.QueuePointer(config, QueueType.STORE, QueuePointerType.HEAD),
            p.QueuePointer(config, QueueType.STORE, QueuePointerType.TAIL),
            p.QueueIsEmpty(QueueType.STORE),

            p.QueueWriteEnableComment(config, QueueType.LOAD),
            p.QueueWriteEnable(config, QueueType.LOAD),

            p.NumNewQueueEntriesComment(QueueType.LOAD),
            # since this is both a local signal and an output
            # the class is placed in local signals
            l.NumNewQueueEntries(config, QueueType.LOAD, d.OUTPUT),

            p.PortIdxPerQueueEntryComment(config, QueueType.LOAD),
            p.PortIdxPerQueueEntry(config, QueueType.LOAD),

            p.QueueWriteEnableComment(config, QueueType.STORE),
            p.QueueWriteEnable(config, QueueType.STORE),

            p.NumNewQueueEntriesComment(QueueType.STORE),
            # since this is both a local signal and an output
            # the class is placed in local signals
            l.NumNewQueueEntries(config, QueueType.STORE, d.OUTPUT),

            p.PortIdxPerQueueEntryComment(config, QueueType.STORE),
            p.PortIdxPerQueueEntry(config, QueueType.STORE),
    
            p.NaiveStoreOrderPerEntryComment(config),
            p.NaiveStoreOrderPerEntry(config)
        ]


        self.local_items = [
            l.GroupInitTransfer(config),
            l.NumNewQueueEntries(config, QueueType.LOAD),
            l.NumNewQueueEntries(config, QueueType.STORE)
        ]

        b = GroupAllocatorBodyItems

        self.body = [
            b.GroupHandshakingInst(config, subunit_prefix),


            *([b.PortIdxPerEntryInst(config, QueueType.LOAD, subunit_prefix)] \
                  if config.load_ports_num() > 1 else []),

            *([b.PortIdxPerEntryInst(config, QueueType.STORE, subunit_prefix)] \
                  if config.store_ports_num() > 1 else []),

            b.NumNewQueueEntriesInst(config, QueueType.LOAD, subunit_prefix),
            b.NumNewQueueEntriesInst(config, QueueType.STORE, subunit_prefix),

            b.NaiveStoreOrderPerEntry(config, subunit_prefix),

            b.WriteEnableInst(config, QueueType.LOAD, subunit_prefix),
            b.WriteEnableInst(config, QueueType.STORE, subunit_prefix)

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
        self.prefix = name

        self.module_name = f"{self.prefix}_{GROUP_ALLOCATOR_NAME}_unit"

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

        # Write to the file
        with open(f'{path_rtl}/{self.name}.vhd', 'a') as file:
            file.write(self.print_dec(GroupHandshakingDecl(config, self.prefix + "_ga")))

            file.write(self.print_dec(WriteEnableDecl(config, QueueType.LOAD, self.prefix + "_ga")))
            file.write(self.print_dec(WriteEnableDecl(config, QueueType.STORE, self.prefix + "_ga")))


        # subunit_prefix = self.prefix + "_ga"

        # unit = self.print_dec(WriteEnableDecl(config, QueueType.LOAD, subunit_prefix))
        # unit += self.print_dec(WriteEnableDecl(config, QueueType.STORE, subunit_prefix))

        # unit += self.print_dec(NumNewQueueEntriesDecl(config, QueueType.LOAD, subunit_prefix))
        # unit += self.print_dec(NumNewQueueEntriesDecl(config, QueueType.STORE, subunit_prefix))

        # unit += self.print_dec(NaiveStoreOrderPerEntryDecl(config, subunit_prefix))

        # if config.load_ports_num() > 1:
        #     unit += self.print_dec(PortIdxPerEntryDecl(config, QueueType.LOAD, subunit_prefix))

        # if config.store_ports_num() > 1:
        #     unit += self.print_dec(PortIdxPerEntryDecl(config, QueueType.STORE, subunit_prefix))

        # unit += self.print_dec(GroupHandshakingDecl(config, subunit_prefix))

        # unit += self.print_dec(GroupAllocatorDecl(config, self.prefix, subunit_prefix))

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


        # Generate handshake signals
        # group_init_ready = LogicArray(
            # ctx, 'group_init_ready', 'w', self.configs.numGroups)
        group_init_hs = LogicArray(
            ctx, 'group_init_hs', 'w', self.configs.numGroups)

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
        if (self.configs.ldpAddrW > 0):
            arch += Mux1HROM(ctx, ldq_port_idx_rom,
                             self.configs.gaLdPortIdx, group_init_hs)
        if (self.configs.stpAddrW > 0):
            arch += Mux1HROM(ctx, stq_port_idx_rom,
                             self.configs.gaStPortIdx, group_init_hs)
        arch += Mux1HROM(ctx, ga_ls_order_rom, self.configs.gaLdOrder,
                         group_init_hs, MaskLess)
        arch += Mux1HROM(ctx, num_loads,
                         self.configs.gaNumLoads, group_init_hs)
        arch += Mux1HROM(ctx, num_stores,
                         self.configs.gaNumStores, group_init_hs)
        arch += Op(ctx, num_loads_o, num_loads)
        arch += Op(ctx, num_stores_o, num_stores)

        # ldq_wen_unshifted = LogicArray(
            # ctx, 'ldq_wen_unshifted', 'w', self.configs.numLdqEntries)
        # stq_wen_unshifted = LogicArray(
            # ctx, 'stq_wen_unshifted', 'w', self.configs.numStqEntries)
        
        # for i in range(0, self.configs.numLdqEntries):
            # arch += Op(ctx, ldq_wen_unshifted[i],
                    #    '\'1\'', 'when',
                    #    num_loads, '>', (i, self.configs.ldqAddrW),
                    #    'else', '\'0\''
                    #    )

        # for i in range(0, self.configs.numStqEntries):
        #     arch += Op(ctx, stq_wen_unshifted[i],
        #                '\'1\'', 'when',
        #                num_stores, '>', (i, self.configs.stqAddrW),
        #                'else', '\'0\''
        #                )

        # Shift the arrays
        if (self.configs.ldpAddrW > 0):
            arch += CyclicLeftShift(ctx, ldq_port_idx_o,
                                    ldq_port_idx_rom, ldq_tail_i)
        if (self.configs.stpAddrW > 0):
            arch += CyclicLeftShift(ctx, stq_port_idx_o,
                                    stq_port_idx_rom, stq_tail_i)
            
        # arch += CyclicLeftShift(ctx, ldq_wen_o, ldq_wen_unshifted, ldq_tail_i)
        # arch += CyclicLeftShift(ctx, stq_wen_o, stq_wen_unshifted, stq_tail_i)

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
            file.write('\n\n')
            file.write(ctx.library)
            file.write(f'entity {self.module_name} is\n')
            file.write(ctx.portInitString)
            file.write('\nend entity;\n\n')
            file.write(f'architecture arch of {self.module_name} is\n')
            file.write(ctx.signalInitString)
            file.write('begin\n')

            b = GroupAllocatorBodyItems

            hs = b.GroupHandshakingInst(config, self.prefix + "_ga")
            file.write(hs.get())

            write_enable = b.WriteEnableInst(config, QueueType.LOAD, self.prefix + "_ga")
            file.write(write_enable.get())

            write_enable = b.WriteEnableInst(config, QueueType.STORE, self.prefix + "_ga")
            file.write(write_enable.get())

            file.write(arch + '\n')

            file.write(ctx.regInitString + 'end architecture;\n')

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
