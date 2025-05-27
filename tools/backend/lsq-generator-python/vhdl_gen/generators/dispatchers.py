from vhdl_gen.context import VHDLContext
from vhdl_gen.signals import *
from vhdl_gen.operators import *


def PortToQueueDispatcher(
    ctx:                VHDLContext,
    path_rtl:           str,
    name:               str,
    suffix:             str,
    numPorts:           int,
    numEntries:         int,
    bitsW:              int,
    portAddrW:          int
) -> str:
    # Initialize the global parameters
    ctx.tabLevel = 1
    ctx.tempCount = 0
    ctx.signalInitString = ''
    ctx.portInitString = '\tport(\n\t\trst : in std_logic;\n\t\tclk : in std_logic'
    arch = ''

    # IOs
    port_bits_i = LogicVecArray(ctx, 'port_bits', 'i', numPorts, bitsW)
    port_valid_i = LogicArray(ctx, 'port_valid', 'i', numPorts)
    port_ready_o = LogicArray(ctx, 'port_ready', 'o', numPorts)
    entry_valid_i = LogicArray(ctx, 'entry_valid', 'i', numEntries)
    entry_bits_valid_i = LogicArray(ctx, 'entry_bits_valid', 'i', numEntries)
    if (numPorts != 1):
        entry_port_idx_i = LogicVecArray(ctx, 'entry_port_idx', 'i', numEntries, portAddrW)
    entry_bits_o = LogicVecArray(ctx, 'entry_bits', 'o', numEntries, bitsW)
    entry_wen_o = LogicArray(ctx, 'entry_wen', 'o', numEntries)
    queue_head_oh_i = LogicVec(ctx, 'queue_head_oh', 'i', numEntries)

    # one-hot port index
    entry_port_valid = LogicVecArray(ctx, 'entry_port_valid', 'w', numEntries, numPorts)
    for i in range(0, numEntries):
        if (numPorts == 1):
            arch += Op(ctx, entry_port_valid[i], 1)
        else:
            arch += BitsToOH(ctx, entry_port_valid[i], entry_port_idx_i[i])

    # Mux for the data/addr
    for i in range(0, numEntries):
        arch += Mux1H(ctx, entry_bits_o[i], port_bits_i, entry_port_valid[i])

    # Entries that request data/address from a any port
    entry_request_valid = LogicArray(ctx, 'entry_request_valid', 'w', numEntries)
    for i in range(0, numEntries):
        arch += Op(ctx, entry_request_valid[i], entry_valid_i[i], 'and', 'not', entry_bits_valid_i[i])

    # Entry-port pairs that the entry request the data/address from the port
    entry_port_request = LogicVecArray(ctx, 'entry_port_request', 'w', numEntries, numPorts)
    for i in range(0, numEntries):
        arch += Op(ctx, entry_port_request[i], entry_port_valid[i], 'when', entry_request_valid[i], 'else', 0)

    # Reduce the matrix for each entry to get the ready signal:
    # If one or more entries is requesting data/address from a certain port, ready is set high.
    port_ready_vec = LogicVec(ctx, 'port_ready_vec', 'w', numPorts)
    arch += Reduce(ctx, port_ready_vec, entry_port_request, 'or')
    arch += VecToArray(ctx, port_ready_o, port_ready_vec)

    # AND the request signal with valid, it shows entry-port pairs that are both valid and ready.
    entry_port_and = LogicVecArray(ctx, 'entry_port_and', 'w', numEntries, numPorts)
    for i in range(0, numEntries):
        for j in range(0, numPorts):
            arch += ctx.get_current_indent() + f'{entry_port_and.getNameWrite(i, j)} <= ' \
                f'{entry_port_request.getNameRead(i, j)} and {port_valid_i.getNameRead(j)};\n'

    # For each port, the oldest entry recieves bit this cycle. The priority masking per port(column)
    # generates entry-port pairs that will tranfer data/address this cycle.
    entry_port_hs = LogicVecArray(ctx, 'entry_port_hs', 'w', numEntries, numPorts)
    arch += CyclicPriorityMasking(ctx, entry_port_hs, entry_port_and, queue_head_oh_i)

    # Reduce for each entry(row), which generates write enable signal for entries
    for i in range(0, numEntries):
        arch += Reduce(ctx, entry_wen_o[i], entry_port_hs[i], 'or')

    ######   Write To File  ######
    ctx.portInitString += '\n\t);'

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
        file.write('end architecture;\n')
    return


def QueueToPortDispatcher(
    ctx:                VHDLContext,
    path_rtl:           str,
    name:               str,
    suffix:             str,
    numPorts:           int,
    numEntries:         int,
    bitsW:              int,
    portAddrW:          int
) -> str:
    # Initialize the global parameters
    ctx.tabLevel = 1
    ctx.tempCount = 0
    ctx.signalInitString = ''
    ctx.portInitString = '\tport(\n\t\trst : in std_logic;\n\t\tclk : in std_logic'
    arch = ''

    # IOs
    if (bitsW != 0):
        port_bits_o = LogicVecArray(ctx, 'port_bits', 'o', numPorts, bitsW)
    port_valid_o = LogicArray(ctx, 'port_valid', 'o', numPorts)
    port_ready_i = LogicArray(ctx, 'port_ready', 'i', numPorts)
    entry_valid_i = LogicArray(ctx, 'entry_valid', 'i', numEntries)
    entry_bits_valid_i = LogicArray(ctx, 'entry_bits_valid', 'i', numEntries)
    if (numPorts != 1):
        entry_port_idx_i = LogicVecArray(ctx, 'entry_port_idx', 'i', numEntries, portAddrW)
    if (bitsW != 0):
        entry_bits_i = LogicVecArray(ctx, 'entry_bits', 'i', numEntries, bitsW)
    entry_reset_o = LogicArray(ctx, 'entry_reset', 'o', numEntries)
    queue_head_oh_i = LogicVec(ctx, 'queue_head_oh', 'i', numEntries)

    # one-hot port index
    entry_port_valid = LogicVecArray(ctx, 'entry_port_valid', 'w', numEntries, numPorts)
    for i in range(0, numEntries):
        if (numPorts == 1):
            arch += Op(ctx, entry_port_valid[i], 1)
        else:
            arch += BitsToOH(ctx, entry_port_valid[i], entry_port_idx_i[i])

    # This matrix shows entry-port pairs that the entry is linked with the port
    entry_port_request = LogicVecArray(ctx, 'entry_port_request', 'w', numEntries, numPorts)
    for i in range(0, numEntries):
        arch += Op(ctx, entry_port_request[i], entry_port_valid[i], 'when', entry_valid_i[i], 'else', 0)

    # For each port, the oldest entry send bits this cycle. The priority masking per port(column)
    # generates entry-port pairs that will tranfer data/address this cycle.
    # It is also used as one-hot select signal for data Mux.
    entry_port_request_prio = LogicVecArray(ctx, 'entry_port_request_prio', 'w', numEntries, numPorts)
    arch += CyclicPriorityMasking(ctx, entry_port_request_prio, entry_port_request, queue_head_oh_i)

    if (bitsW != 0):
        for j in range(0, numPorts):
            arch += Mux1H(ctx, port_bits_o[j], entry_bits_i, entry_port_request_prio, j)

    # Mask the matrix with dataValid
    entry_port_request_valid = LogicVecArray(ctx, 'entry_port_request_valid', 'w', numEntries, numPorts)
    for i in range(0, numEntries):
        arch += Op(ctx, entry_port_request_valid[i], entry_port_request_prio[i],
                   'when', entry_bits_valid_i[i], 'else', 0)

    # Reduce the matrix for each port to get the valid signal:
    # If an entry is providing data/address from a certain port, valid is set high.
    port_valid_vec = LogicVec(ctx, 'port_valid_vec', 'w', numPorts)
    arch += Reduce(ctx, port_valid_vec, entry_port_request_valid, 'or')
    arch += VecToArray(ctx, port_valid_o, port_valid_vec)

    # AND the request signal with ready, it shows entry-port pairs that are both valid and ready.
    entry_port_hs = LogicVecArray(ctx, 'entry_port_hs', 'w', numEntries, numPorts)
    for i in range(0, numEntries):
        for j in range(0, numPorts):
            arch += ctx.get_current_indent() + f'{entry_port_hs.getNameWrite(i, j)} <= ' \
                f'{entry_port_request_valid.getNameRead(i, j)} and {port_ready_i.getNameRead(j)};\n'

    # Reduce for each entry(row), which generates reset signal for entries
    for i in range(0, numEntries):
        arch += Reduce(ctx, entry_reset_o[i], entry_port_hs[i], 'or')

    ######   Write To File  ######
    ctx.portInitString += '\n\t);'

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
        file.write('end architecture;\n')
    return


def PortToQueueDispatcherInit(
    ctx:                VHDLContext,
    name:               str,
    numPorts:           int,
    numEntries:         int,
    port_bits_i:        LogicVecArray,
    port_valid_i:       LogicArray,
    port_ready_o:       LogicArray,
    entry_valid_i:      LogicArray,
    entry_bits_valid_i: LogicArray,
    entry_port_idx_i:   LogicVecArray,
    entry_bits_o:       LogicVecArray,
    entry_wen_o:        LogicArray,
    queue_head_oh_i:    LogicVec
) -> str:

    arch = ctx.get_current_indent() + f'{name}_dispatcher : entity work.{name}\n'
    ctx.tabLevel += 1
    arch += ctx.get_current_indent() + f'port map(\n'
    ctx.tabLevel += 1
    arch += ctx.get_current_indent() + f'rst => rst,\n'
    arch += ctx.get_current_indent() + f'clk => clk,\n'
    for i in range(0, numPorts):
        arch += ctx.get_current_indent() + f'port_bits_{i}_i => {port_bits_i.getNameRead(i)},\n'
    for i in range(0, numPorts):
        arch += ctx.get_current_indent() + f'port_ready_{i}_o => {port_ready_o.getNameWrite(i)},\n'
    for i in range(0, numPorts):
        arch += ctx.get_current_indent() + f'port_valid_{i}_i => {port_valid_i.getNameRead(i)},\n'
    for i in range(0, numEntries):
        arch += ctx.get_current_indent() + f'entry_valid_{i}_i => {entry_valid_i.getNameRead(i)},\n'
    for i in range(0, numEntries):
        arch += ctx.get_current_indent() + f'entry_bits_valid_{i}_i => {entry_bits_valid_i.getNameRead(i)},\n'
    for i in range(0, numEntries):
        if (numPorts != 1):
            arch += ctx.get_current_indent() + f'entry_port_idx_{i}_i => {entry_port_idx_i.getNameRead(i)},\n'
    for i in range(0, numEntries):
        arch += ctx.get_current_indent() + f'entry_bits_{i}_o => {entry_bits_o.getNameWrite(i)},\n'
    for i in range(0, numEntries):
        arch += ctx.get_current_indent() + f'entry_wen_{i}_o => {entry_wen_o.getNameWrite(i)},\n'
    arch += ctx.get_current_indent() + f'queue_head_oh_i => {queue_head_oh_i.getNameRead()}\n'
    ctx.tabLevel -= 1
    arch += ctx.get_current_indent() + f');\n'
    ctx.tabLevel -= 1
    return arch


def QueueToPortDispatcherInit(
    ctx:                VHDLContext,
    name:               str,
    numPorts:           int,
    numEntries:         int,
    port_bits_o:        LogicVecArray,
    port_valid_o:       LogicArray,
    port_ready_i:       LogicArray,
    entry_valid_i:      LogicArray,
    entry_bits_valid_i: LogicArray,
    entry_port_idx_i:   LogicVecArray,
    entry_bits_i:       LogicVecArray,
    entry_reset_o:      LogicArray,
    queue_head_oh_i:    LogicVec
) -> str:

    arch = ctx.get_current_indent() + f'{name}_dispatcher : entity work.{name}\n'
    ctx.tabLevel += 1
    arch += ctx.get_current_indent() + f'port map(\n'
    ctx.tabLevel += 1
    arch += ctx.get_current_indent() + f'rst => rst,\n'
    arch += ctx.get_current_indent() + f'clk => clk,\n'
    for i in range(0, numPorts):
        if (port_bits_o != None):
            arch += ctx.get_current_indent() + f'port_bits_{i}_o => {port_bits_o.getNameWrite(i)},\n'
    for i in range(0, numPorts):
        arch += ctx.get_current_indent() + f'port_ready_{i}_i => {port_ready_i.getNameRead(i)},\n'
    for i in range(0, numPorts):
        arch += ctx.get_current_indent() + f'port_valid_{i}_o => {port_valid_o.getNameWrite(i)},\n'
    for i in range(0, numEntries):
        arch += ctx.get_current_indent() + f'entry_valid_{i}_i => {entry_valid_i.getNameRead(i)},\n'
    for i in range(0, numEntries):
        arch += ctx.get_current_indent() + f'entry_bits_valid_{i}_i => {entry_bits_valid_i.getNameRead(i)},\n'
    for i in range(0, numEntries):
        if (numPorts != 1):
            arch += ctx.get_current_indent() + f'entry_port_idx_{i}_i => {entry_port_idx_i.getNameRead(i)},\n'
    for i in range(0, numEntries):
        if (entry_bits_i != None):
            arch += ctx.get_current_indent() + f'entry_bits_{i}_i => {entry_bits_i.getNameRead(i)},\n'
    for i in range(0, numEntries):
        arch += ctx.get_current_indent() + f'entry_reset_{i}_o => {entry_reset_o.getNameWrite(i)},\n'
    arch += ctx.get_current_indent() + f'queue_head_oh_i => {queue_head_oh_i.getNameRead()}\n'
    ctx.tabLevel -= 1
    arch += ctx.get_current_indent() + f');\n'
    ctx.tabLevel -= 1
    return arch
