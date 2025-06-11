from vhdl_gen.generators.base import BaseVHDLGenerator

from vhdl_gen.context import VHDLContext
from vhdl_gen.signals import Logic, LogicArray, LogicVec, LogicVecArray
from vhdl_gen.operators import Op, WrapSub, Mux1HROM, CyclicLeftShift, CyclicPriorityMasking
from vhdl_gen.utils import MaskLess
from vhdl_gen.configs import Configs


class GroupAllocator(BaseVHDLGenerator):
    def __init__(
            self,
            ctx: VHDLContext,
            path_rtl: str,
            name: str,
            suffix: str,

        configs: Configs
    ):
        super().__init__(ctx, path_rtl, name, suffix)

        self.configs = configs
        self.module_name = name + suffix

    def generate(self) -> None:
        """
        Group Allocator

        Generates the VHDL 'entity' and 'architecture' sections for a group allocator.

        Parameters:
            ctx         : VHDLContext for code generation state.
            path_rtl    : Output directory for VHDL files.
            name        : Base name of the group allocator.
            suffix      : Suffix appended to the entity name.
            configs     : configuration generated from JSON

        Output:
            Appends the 'entity' and 'architecture' definitions
            to the .vhd file at <path_rtl>/<name>_core.vhd.
            Entity and architecture use the identifier: <name><suffix>

        Example (Group Allocator):
            GroupAllocator(ctx, path_rtl, 'config_0', '_core_ga', configs)

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
        # Initialize the global parameters
        self.ctx.tabLevel = 1
        self.ctx.tempCount = 0
        self.ctx.signalInitString = ''
        self.ctx.portInitString = '\tport(\n\t\trst : in std_logic;\n\t\tclk : in std_logic'
        self.ctx.regInitString = '\tprocess (clk, rst) is\n' + '\tbegin\n'
        arch = ''

        # IOs
        group_init_valid_i = LogicArray(
            self.ctx, 'group_init_valid', 'i', self.configs.numGroups)
        group_init_ready_o = LogicArray(
            self.ctx, 'group_init_ready', 'o', self.configs.numGroups)

        ldq_tail_i = LogicVec(self.ctx, 'ldq_tail', 'i', self.configs.ldqAddrW)
        ldq_head_i = LogicVec(self.ctx, 'ldq_head', 'i', self.configs.ldqAddrW)
        ldq_empty_i = Logic(self.ctx, 'ldq_empty', 'i')

        stq_tail_i = LogicVec(self.ctx, 'stq_tail', 'i', self.configs.stqAddrW)
        stq_head_i = LogicVec(self.ctx, 'stq_head', 'i', self.configs.stqAddrW)
        stq_empty_i = Logic(self.ctx, 'stq_empty', 'i')

        ldq_wen_o = LogicArray(self.ctx, 'ldq_wen', 'o',
                               self.configs.numLdqEntries)
        num_loads_o = LogicVec(self.ctx, 'num_loads',
                               'o', self.configs.ldqAddrW)
        num_loads = LogicVec(self.ctx, 'num_loads', 'w', self.configs.ldqAddrW)
        if (self.configs.ldpAddrW > 0):
            ldq_port_idx_o = LogicVecArray(
                self.ctx, 'ldq_port_idx', 'o', self.configs.numLdqEntries, self.configs.ldpAddrW)

        stq_wen_o = LogicArray(self.ctx, 'stq_wen', 'o',
                               self.configs.numStqEntries)
        num_stores_o = LogicVec(self.ctx, 'num_stores',
                                'o', self.configs.stqAddrW)
        num_stores = LogicVec(self.ctx, 'num_stores',
                              'w', self.configs.stqAddrW)
        if (self.configs.stpAddrW > 0):
            stq_port_idx_o = LogicVecArray(
                self.ctx, 'stq_port_idx', 'o', self.configs.numStqEntries, self.configs.stpAddrW)

        ga_ls_order_o = LogicVecArray(
            self.ctx, 'ga_ls_order', 'o', self.configs.numLdqEntries, self.configs.numStqEntries)

        # The number of empty load and store is calculated with cyclic subtraction.
        # If the empty signal is high, then set the number to max value.
        loads_sub = LogicVec(self.ctx, 'loads_sub', 'w', self.configs.ldqAddrW)
        stores_sub = LogicVec(self.ctx, 'stores_sub',
                              'w', self.configs.stqAddrW)
        empty_loads = LogicVec(self.ctx, 'empty_loads',
                               'w', self.configs.emptyLdAddrW)
        empty_stores = LogicVec(self.ctx, 'empty_stores',
                                'w', self.configs.emptyStAddrW)

        arch += WrapSub(self.ctx, loads_sub, ldq_head_i,
                        ldq_tail_i, self.configs.numLdqEntries)
        arch += WrapSub(self.ctx, stores_sub, stq_head_i,
                        stq_tail_i, self.configs.numStqEntries)

        arch += Op(self.ctx, empty_loads, self.configs.numLdqEntries, 'when', ldq_empty_i, 'else',
                   '(', '\'0\'', '&', loads_sub, ')')
        arch += Op(self.ctx, empty_stores, self.configs.numStqEntries, 'when', stq_empty_i, 'else',
                   '(', '\'0\'', '&', stores_sub, ')')

        # Generate handshake signals
        group_init_ready = LogicArray(
            self.ctx, 'group_init_ready', 'w', self.configs.numGroups)
        group_init_hs = LogicArray(
            self.ctx, 'group_init_hs', 'w', self.configs.numGroups)

        for i in range(0, self.configs.numGroups):
            arch += Op(self.ctx, group_init_ready[i],
                       '\'1\'', 'when',
                       '(', empty_loads,  '>=', (
                self.configs.gaNumLoads[i], self.configs.emptyLdAddrW),  ')', 'and',
                '(', empty_stores, '>=', (
                self.configs.gaNumStores[i], self.configs.emptyStAddrW), ')',
                'else', '\'0\'')

        if (self.configs.gaMulti):
            group_init_and = LogicArray(
                self.ctx, 'group_init_and', 'w', self.configs.numGroups)
            ga_rr_mask = LogicVec(self.ctx, 'ga_rr_mask',
                                  'r', self.configs.numGroups)
            ga_rr_mask.regInit()
            for i in range(0, self.configs.numGroups):
                arch += Op(self.ctx, group_init_and[i],
                           group_init_ready[i], 'and', group_init_valid_i[i])
                arch += Op(self.ctx, group_init_ready_o[i], group_init_hs[i])
            arch += CyclicPriorityMasking(self.ctx, group_init_hs,
                                          group_init_and, ga_rr_mask)
            for i in range(0, self.configs.numGroups):
                arch += Op(self.ctx, (ga_rr_mask, (i+1) %
                                      self.configs.numGroups), (group_init_hs, i))
        else:
            for i in range(0, self.configs.numGroups):
                arch += Op(self.ctx,
                           group_init_ready_o[i], group_init_ready[i])
                arch += Op(self.ctx, group_init_hs[i],
                           group_init_ready[i], 'and', group_init_valid_i[i])

        # ROM value
        if (self.configs.ldpAddrW > 0):
            ldq_port_idx_rom = LogicVecArray(
                self.ctx, 'ldq_port_idx_rom', 'w', self.configs.numLdqEntries, self.configs.ldpAddrW)
        if (self.configs.stpAddrW > 0):
            stq_port_idx_rom = LogicVecArray(
                self.ctx, 'stq_port_idx_rom', 'w', self.configs.numStqEntries, self.configs.stpAddrW)
        ga_ls_order_rom = LogicVecArray(
            self.ctx, 'ga_ls_order_rom', 'w', self.configs.numLdqEntries, self.configs.numStqEntries)
        ga_ls_order_temp = LogicVecArray(
            self.ctx, 'ga_ls_order_temp', 'w', self.configs.numLdqEntries, self.configs.numStqEntries)
        if (self.configs.ldpAddrW > 0):
            arch += Mux1HROM(self.ctx, ldq_port_idx_rom,
                             self.configs.gaLdPortIdx, group_init_hs)
        if (self.configs.stpAddrW > 0):
            arch += Mux1HROM(self.ctx, stq_port_idx_rom,
                             self.configs.gaStPortIdx, group_init_hs)
        arch += Mux1HROM(self.ctx, ga_ls_order_rom, self.configs.gaLdOrder,
                         group_init_hs, MaskLess)
        arch += Mux1HROM(self.ctx, num_loads,
                         self.configs.gaNumLoads, group_init_hs)
        arch += Mux1HROM(self.ctx, num_stores,
                         self.configs.gaNumStores, group_init_hs)
        arch += Op(self.ctx, num_loads_o, num_loads)
        arch += Op(self.ctx, num_stores_o, num_stores)

        ldq_wen_unshifted = LogicArray(
            self.ctx, 'ldq_wen_unshifted', 'w', self.configs.numLdqEntries)
        stq_wen_unshifted = LogicArray(
            self.ctx, 'stq_wen_unshifted', 'w', self.configs.numStqEntries)
        for i in range(0, self.configs.numLdqEntries):
            arch += Op(self.ctx, ldq_wen_unshifted[i],
                       '\'1\'', 'when',
                       num_loads, '>', (i, self.configs.ldqAddrW),
                       'else', '\'0\''
                       )
        for i in range(0, self.configs.numStqEntries):
            arch += Op(self.ctx, stq_wen_unshifted[i],
                       '\'1\'', 'when',
                       num_stores, '>', (i, self.configs.stqAddrW),
                       'else', '\'0\''
                       )

        # Shift the arrays
        if (self.configs.ldpAddrW > 0):
            arch += CyclicLeftShift(self.ctx, ldq_port_idx_o,
                                    ldq_port_idx_rom, ldq_tail_i)
        if (self.configs.stpAddrW > 0):
            arch += CyclicLeftShift(self.ctx, stq_port_idx_o,
                                    stq_port_idx_rom, stq_tail_i)
        arch += CyclicLeftShift(self.ctx, ldq_wen_o,
                                ldq_wen_unshifted, ldq_tail_i)
        arch += CyclicLeftShift(self.ctx, stq_wen_o,
                                stq_wen_unshifted, stq_tail_i)
        for i in range(0, self.configs.numLdqEntries):
            arch += CyclicLeftShift(self.ctx,
                                    ga_ls_order_temp[i], ga_ls_order_rom[i], stq_tail_i)
        arch += CyclicLeftShift(self.ctx, ga_ls_order_o,
                                ga_ls_order_temp, ldq_tail_i)

        ######   Write To File  ######
        self.ctx.portInitString += '\n\t);'
        if (self.configs.gaMulti):
            self.ctx.regInitString += '\tend process;\n'
        else:
            self.ctx.regInitString = ''

        # Write to the file
        with open(f'{self.path_rtl}/{self.name}.vhd', 'a') as file:
            file.write('\n\n')
            file.write(self.ctx.library)
            file.write(f'entity {self.module_name} is\n')
            file.write(self.ctx.portInitString)
            file.write('\nend entity;\n\n')
            file.write(f'architecture arch of {self.module_name} is\n')
            file.write(self.ctx.signalInitString)
            file.write('begin\n' + arch + '\n')
            file.write(self.ctx.regInitString + 'end architecture;\n')

    def instantiate(
        self,
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
            name                 : Base name of the group allocator entity.
            configs              : configuration generated from JSON
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
            # Base architecture: 'config_0_core'
            # suffix for GroupAllocator instantiation: '_ga'

            arch += GroupAllocatorInst(
                ctx,
                name               = 'config_0_core' + '_ga',
                configs            = configs,
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

        arch = self.ctx.get_current_indent(
        ) + f'{self.module_name} : entity work.{self.module_name}\n'
        self.ctx.tabLevel += 1
        arch += self.ctx.get_current_indent() + f'port map(\n'
        self.ctx.tabLevel += 1

        arch += self.ctx.get_current_indent() + f'rst => rst,\n'
        arch += self.ctx.get_current_indent() + f'clk => clk,\n'

        for i in range(0, self.configs.numGroups):
            arch += self.ctx.get_current_indent() + \
                f'group_init_valid_{i}_i => {group_init_valid_i.getNameRead(i)},\n'
        for i in range(0, self.configs.numGroups):
            arch += self.ctx.get_current_indent() + \
                f'group_init_ready_{i}_o => {group_init_ready_o.getNameWrite(i)},\n'

        arch += self.ctx.get_current_indent() + \
            f'ldq_tail_i => {ldq_tail_i.getNameRead()},\n'
        arch += self.ctx.get_current_indent() + \
            f'ldq_head_i => {ldq_head_i.getNameRead()},\n'
        arch += self.ctx.get_current_indent() + \
            f'ldq_empty_i => {ldq_empty_i.getNameRead()},\n'

        arch += self.ctx.get_current_indent() + \
            f'stq_tail_i => {stq_tail_i.getNameRead()},\n'
        arch += self.ctx.get_current_indent() + \
            f'stq_head_i => {stq_head_i.getNameRead()},\n'
        arch += self.ctx.get_current_indent() + \
            f'stq_empty_i => {stq_empty_i.getNameRead()},\n'

        for i in range(0, self.configs.numLdqEntries):
            arch += self.ctx.get_current_indent() + \
                f'ldq_wen_{i}_o => {ldq_wen_o.getNameWrite(i)},\n'
        arch += self.ctx.get_current_indent() + \
            f'num_loads_o => {num_loads_o.getNameWrite()},\n'
        if (self.configs.ldpAddrW > 0):
            for i in range(0, self.configs.numLdqEntries):
                arch += self.ctx.get_current_indent() + \
                    f'ldq_port_idx_{i}_o => {ldq_port_idx_o.getNameWrite(i)},\n'

        for i in range(0, self.configs.numStqEntries):
            arch += self.ctx.get_current_indent() + \
                f'stq_wen_{i}_o => {stq_wen_o.getNameWrite(i)},\n'
        if (self.configs.stpAddrW > 0):
            for i in range(0, self.configs.numStqEntries):
                arch += self.ctx.get_current_indent() + \
                    f'stq_port_idx_{i}_o => {stq_port_idx_o.getNameWrite(i)},\n'

        for i in range(0, self.configs.numLdqEntries):
            arch += self.ctx.get_current_indent() + \
                f'ga_ls_order_{i}_o => {ga_ls_order_o.getNameWrite(i)},\n'

        arch += self.ctx.get_current_indent() + \
            f'num_stores_o => {num_stores_o.getNameWrite()}\n'

        self.ctx.tabLevel -= 1
        arch += self.ctx.get_current_indent() + f');\n'
        self.ctx.tabLevel -= 1
        return arch
