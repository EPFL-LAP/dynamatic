from vhdl_gen.context import VHDLContext
from vhdl_gen.signals import *
from vhdl_gen.operators import *
from vhdl_gen.configs import *


def GroupAllocator(ctx: VHDLContext, path_rtl: str, name: str, suffix: str, configs: Configs) -> str:
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
    ctx.tabLevel = 1
    ctx.tempCount = 0
    ctx.signalInitString = ''
    ctx.portInitString = '\tport(\n\t\trst : in std_logic;\n\t\tclk : in std_logic'
    ctx.regInitString = '\tprocess (clk, rst) is\n' + '\tbegin\n'
    arch = ''

    # IOs
    group_init_valid_i = LogicArray(
        ctx, 'group_init_valid', 'i', configs.numGroups)
    group_init_ready_o = LogicArray(
        ctx, 'group_init_ready', 'o', configs.numGroups)

    ldq_tail_i = LogicVec(ctx, 'ldq_tail', 'i', configs.ldqAddrW)
    ldq_head_i = LogicVec(ctx, 'ldq_head', 'i', configs.ldqAddrW)
    ldq_empty_i = Logic(ctx, 'ldq_empty', 'i')

    stq_tail_i = LogicVec(ctx, 'stq_tail', 'i', configs.stqAddrW)
    stq_head_i = LogicVec(ctx, 'stq_head', 'i', configs.stqAddrW)
    stq_empty_i = Logic(ctx, 'stq_empty', 'i')

    ldq_wen_o = LogicArray(ctx, 'ldq_wen', 'o', configs.numLdqEntries)
    num_loads_o = LogicVec(ctx, 'num_loads', 'o', configs.ldqAddrW)
    num_loads = LogicVec(ctx, 'num_loads', 'w', configs.ldqAddrW)
    if (configs.ldpAddrW > 0):
        ldq_port_idx_o = LogicVecArray(
            ctx, 'ldq_port_idx', 'o', configs.numLdqEntries, configs.ldpAddrW)

    stq_wen_o = LogicArray(ctx, 'stq_wen', 'o', configs.numStqEntries)
    num_stores_o = LogicVec(ctx, 'num_stores', 'o', configs.stqAddrW)
    num_stores = LogicVec(ctx, 'num_stores', 'w', configs.stqAddrW)
    if (configs.stpAddrW > 0):
        stq_port_idx_o = LogicVecArray(
            ctx, 'stq_port_idx', 'o', configs.numStqEntries, configs.stpAddrW)

    ga_ls_order_o = LogicVecArray(
        ctx, 'ga_ls_order', 'o', configs.numLdqEntries, configs.numStqEntries)

    # The number of empty load and store is calculated with cyclic subtraction.
    # If the empty signal is high, then set the number to max value.
    loads_sub = LogicVec(ctx, 'loads_sub', 'w', configs.ldqAddrW)
    stores_sub = LogicVec(ctx, 'stores_sub', 'w', configs.stqAddrW)
    empty_loads = LogicVec(ctx, 'empty_loads', 'w', configs.emptyLdAddrW)
    empty_stores = LogicVec(ctx, 'empty_stores', 'w', configs.emptyStAddrW)

    arch += WrapSub(ctx, loads_sub, ldq_head_i,
                    ldq_tail_i, configs.numLdqEntries)
    arch += WrapSub(ctx, stores_sub, stq_head_i,
                    stq_tail_i, configs.numStqEntries)

    arch += Op(ctx, empty_loads, configs.numLdqEntries, 'when', ldq_empty_i, 'else',
               '(', '\'0\'', '&', loads_sub, ')')
    arch += Op(ctx, empty_stores, configs.numStqEntries, 'when', stq_empty_i, 'else',
               '(', '\'0\'', '&', stores_sub, ')')

    # Generate handshake signals
    group_init_ready = LogicArray(
        ctx, 'group_init_ready', 'w', configs.numGroups)
    group_init_hs = LogicArray(ctx, 'group_init_hs', 'w', configs.numGroups)

    for i in range(0, configs.numGroups):
        arch += Op(ctx, group_init_ready[i],
                   '\'1\'', 'when',
                   '(', empty_loads,  '>=', (
                       configs.gaNumLoads[i], configs.emptyLdAddrW),  ')', 'and',
                   '(', empty_stores, '>=', (
                       configs.gaNumStores[i], configs.emptyStAddrW), ')',
                   'else', '\'0\'')

    if (configs.gaMulti):
        group_init_and = LogicArray(
            ctx, 'group_init_and', 'w', configs.numGroups)
        ga_rr_mask = LogicVec(ctx, 'ga_rr_mask', 'r', configs.numGroups)
        ga_rr_mask.regInit()
        for i in range(0, configs.numGroups):
            arch += Op(ctx, group_init_and[i],
                       group_init_ready[i], 'and', group_init_valid_i[i])
            arch += Op(ctx, group_init_ready_o[i], group_init_hs[i])
        arch += CyclicPriorityMasking(ctx, group_init_hs,
                                      group_init_and, ga_rr_mask)
        for i in range(0, configs.numGroups):
            arch += Op(ctx, (ga_rr_mask, (i+1) %
                       configs.numGroups), (group_init_hs, i))
    else:
        for i in range(0, configs.numGroups):
            arch += Op(ctx, group_init_ready_o[i], group_init_ready[i])
            arch += Op(ctx, group_init_hs[i],
                       group_init_ready[i], 'and', group_init_valid_i[i])

    # ROM value
    if (configs.ldpAddrW > 0):
        ldq_port_idx_rom = LogicVecArray(
            ctx, 'ldq_port_idx_rom', 'w', configs.numLdqEntries, configs.ldpAddrW)
    if (configs.stpAddrW > 0):
        stq_port_idx_rom = LogicVecArray(
            ctx, 'stq_port_idx_rom', 'w', configs.numStqEntries, configs.stpAddrW)
    ga_ls_order_rom = LogicVecArray(
        ctx, 'ga_ls_order_rom', 'w', configs.numLdqEntries, configs.numStqEntries)
    ga_ls_order_temp = LogicVecArray(
        ctx, 'ga_ls_order_temp', 'w', configs.numLdqEntries, configs.numStqEntries)
    if (configs.ldpAddrW > 0):
        arch += Mux1HROM(ctx, ldq_port_idx_rom,
                         configs.gaLdPortIdx, group_init_hs)
    if (configs.stpAddrW > 0):
        arch += Mux1HROM(ctx, stq_port_idx_rom,
                         configs.gaStPortIdx, group_init_hs)
    arch += Mux1HROM(ctx, ga_ls_order_rom, configs.gaLdOrder,
                     group_init_hs, MaskLess)
    arch += Mux1HROM(ctx, num_loads, configs.gaNumLoads, group_init_hs)
    arch += Mux1HROM(ctx, num_stores, configs.gaNumStores, group_init_hs)
    arch += Op(ctx, num_loads_o, num_loads)
    arch += Op(ctx, num_stores_o, num_stores)

    ldq_wen_unshifted = LogicArray(
        ctx, 'ldq_wen_unshifted', 'w', configs.numLdqEntries)
    stq_wen_unshifted = LogicArray(
        ctx, 'stq_wen_unshifted', 'w', configs.numStqEntries)
    for i in range(0, configs.numLdqEntries):
        arch += Op(ctx, ldq_wen_unshifted[i],
                   '\'1\'', 'when',
                   num_loads, '>', (i, configs.ldqAddrW),
                   'else', '\'0\''
                   )
    for i in range(0, configs.numStqEntries):
        arch += Op(ctx, stq_wen_unshifted[i],
                   '\'1\'', 'when',
                   num_stores, '>', (i, configs.stqAddrW),
                   'else', '\'0\''
                   )

    # Shift the arrays
    if (configs.ldpAddrW > 0):
        arch += CyclicLeftShift(ctx, ldq_port_idx_o,
                                ldq_port_idx_rom, ldq_tail_i)
    if (configs.stpAddrW > 0):
        arch += CyclicLeftShift(ctx, stq_port_idx_o,
                                stq_port_idx_rom, stq_tail_i)
    arch += CyclicLeftShift(ctx, ldq_wen_o, ldq_wen_unshifted, ldq_tail_i)
    arch += CyclicLeftShift(ctx, stq_wen_o, stq_wen_unshifted, stq_tail_i)
    for i in range(0, configs.numLdqEntries):
        arch += CyclicLeftShift(ctx,
                                ga_ls_order_temp[i], ga_ls_order_rom[i], stq_tail_i)
    arch += CyclicLeftShift(ctx, ga_ls_order_o, ga_ls_order_temp, ldq_tail_i)

    ######   Write To File  ######
    ctx.portInitString += '\n\t);'
    if (configs.gaMulti):
        ctx.regInitString += '\tend process;\n'
    else:
        ctx.regInitString = ''

    # Write to the file
    with open(f'{path_rtl}/{name}_core.vhd', 'a') as file:
        file.write('\n\n')
        file.write(ctx.library)
        file.write(f'entity {name + suffix} is\n')
        file.write(ctx.portInitString)
        file.write('\nend entity;\n\n')
        file.write(f'architecture arch of {name + suffix} is\n')
        file.write(ctx.signalInitString)
        file.write('begin\n' + arch + '\n')
        file.write(ctx.regInitString + 'end architecture;\n')
    return


def GroupAllocatorInst(
    ctx:                VHDLContext,
    name:               str,
    configs:            Configs,
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

    arch = ctx.get_current_indent() + f'{name} : entity work.{name}\n'
    ctx.tabLevel += 1
    arch += ctx.get_current_indent() + f'port map(\n'
    ctx.tabLevel += 1

    arch += ctx.get_current_indent() + f'rst => rst,\n'
    arch += ctx.get_current_indent() + f'clk => clk,\n'

    for i in range(0, configs.numGroups):
        arch += ctx.get_current_indent() + \
            f'group_init_valid_{i}_i => {group_init_valid_i.getNameRead(i)},\n'
    for i in range(0, configs.numGroups):
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

    for i in range(0, configs.numLdqEntries):
        arch += ctx.get_current_indent() + \
            f'ldq_wen_{i}_o => {ldq_wen_o.getNameWrite(i)},\n'
    arch += ctx.get_current_indent() + \
        f'num_loads_o => {num_loads_o.getNameWrite()},\n'
    if (configs.ldpAddrW > 0):
        for i in range(0, configs.numLdqEntries):
            arch += ctx.get_current_indent() + \
                f'ldq_port_idx_{i}_o => {ldq_port_idx_o.getNameWrite(i)},\n'

    for i in range(0, configs.numStqEntries):
        arch += ctx.get_current_indent() + \
            f'stq_wen_{i}_o => {stq_wen_o.getNameWrite(i)},\n'
    if (configs.stpAddrW > 0):
        for i in range(0, configs.numStqEntries):
            arch += ctx.get_current_indent() + \
                f'stq_port_idx_{i}_o => {stq_port_idx_o.getNameWrite(i)},\n'

    for i in range(0, configs.numLdqEntries):
        arch += ctx.get_current_indent() + \
            f'ga_ls_order_{i}_o => {ga_ls_order_o.getNameWrite(i)},\n'

    arch += ctx.get_current_indent() + \
        f'num_stores_o => {num_stores_o.getNameWrite()}\n'

    ctx.tabLevel -= 1
    arch += ctx.get_current_indent() + f');\n'
    ctx.tabLevel -= 1
    return arch
