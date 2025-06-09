from vhdl_gen.context import VHDLContext
from vhdl_gen.signals import *
from vhdl_gen.operators import *
from vhdl_gen.configs import *
from vhdl_gen.generators.dispatchers import PortToQueueDispatcherInst, QueueToPortDispatcherInst
from vhdl_gen.generators.group_allocator import GroupAllocatorInst


def LSQ(ctx: VHDLContext, path_rtl: str, name: str, configs: Configs):
    """
    LSQ

    Generates the VHDL 'entity' and 'architecture' sections for an LSQ.

    This function appends the following to the file '<path_rtl>/<name>.vhd:
        1. 'entity <name>' declaration
        2. 'architecture arch of <name>' implementation

    The generated code also instantitates:
        - Group Allocator
        - Port-to-Queue Dispatcher
            - Load Address Port Dispatcher
            - Store Address Port Dispatcher
            - Store Data Port Dispatcher
        - Queue-to-Port Dispatcher
            - Load Data Port Dispatcher
            - (Optionally) Store Backward Port Dispatcher

    Parameters:
        ctx         : VHDLContext for code generation state.
        path_rtl    : Output directory for VHDL files.
        name        : Base name of the LSQ.
        configs     : configuration generated from JSON

    Output:
        Appends the 'entity' and 'architecture' definitions
        to the .vhd file at <path_rtl>/<name>.vhd.
        Entity and architecture use the identifier: <name>

    Example:
        LSQ(ctx, path_rtl, 'config_0' + '_core', configs)


    *Instantiation of LSQ is in lsq-generator.py.
    """

    # Initialize the global parameters
    ctx.tabLevel = 1
    ctx.tempCount = 0
    ctx.signalInitString = ''
    ctx.portInitString = '\tport(\n\t\trst : in std_logic;\n\t\tclk : in std_logic'
    ctx.regInitString = '\tprocess (clk, rst) is\n' + '\tbegin\n'
    arch = ''

    ###### LSQ Architecture ######
    ######        IOs       ######

    # group initialzation signals
    group_init_valid_i = LogicArray(
        ctx, 'group_init_valid', 'i', configs.numGroups)
    group_init_ready_o = LogicArray(
        ctx, 'group_init_ready', 'o', configs.numGroups)

    # Memory access ports, i.e., the connection "kernel -> LSQ"
    # Load address channel (addr, valid, ready) from kernel, contains signals:
    ldp_addr_i = LogicVecArray(
        ctx, 'ldp_addr', 'i', configs.numLdPorts, configs.addrW)
    ldp_addr_valid_i = LogicArray(
        ctx, 'ldp_addr_valid', 'i', configs.numLdPorts)
    ldp_addr_ready_o = LogicArray(
        ctx, 'ldp_addr_ready', 'o', configs.numLdPorts)

    # Load data channel (data, valid, ready) to kernel
    ldp_data_o = LogicVecArray(
        ctx, 'ldp_data', 'o', configs.numLdPorts, configs.dataW)
    ldp_data_valid_o = LogicArray(
        ctx, 'ldp_data_valid', 'o', configs.numLdPorts)
    ldp_data_ready_i = LogicArray(
        ctx, 'ldp_data_ready', 'i', configs.numLdPorts)

    # Store address channel (addr, valid, ready) from kernel
    stp_addr_i = LogicVecArray(
        ctx, 'stp_addr', 'i', configs.numStPorts, configs.addrW)
    stp_addr_valid_i = LogicArray(
        ctx, 'stp_addr_valid', 'i', configs.numStPorts)
    stp_addr_ready_o = LogicArray(
        ctx, 'stp_addr_ready', 'o', configs.numStPorts)

    # Store data channel (data, valid, ready) from kernel
    stp_data_i = LogicVecArray(
        ctx, 'stp_data', 'i', configs.numStPorts, configs.dataW)
    stp_data_valid_i = LogicArray(
        ctx, 'stp_data_valid', 'i', configs.numStPorts)
    stp_data_ready_o = LogicArray(
        ctx, 'stp_data_ready', 'o', configs.numStPorts)

    if configs.stResp:
        stp_exec_valid_o = LogicArray(
            ctx, 'stp_exec_valid', 'o', configs.numStPorts)
        stp_exec_ready_i = LogicArray(
            ctx, 'stp_exec_ready', 'i', configs.numStPorts)

    # queue empty signal
    empty_o = Logic(ctx, 'empty', 'o')

    # Memory interface: i.e., the connection LSQ -> AXI
    # We assume that the memory interface has
    # 1. A read request channel (rreq) and a read response channel (rresp).
    # 2. A write request channel (wreq) and a write response channel (wresp).
    rreq_valid_o = LogicArray(ctx, 'rreq_valid', 'o', configs.numLdMem)
    rreq_ready_i = LogicArray(ctx, 'rreq_ready', 'i', configs.numLdMem)
    rreq_id_o = LogicVecArray(
        ctx, 'rreq_id', 'o', configs.numLdMem, configs.idW)
    rreq_addr_o = LogicVecArray(
        ctx, 'rreq_addr', 'o', configs.numLdMem, configs.addrW)

    rresp_valid_i = LogicArray(ctx, 'rresp_valid', 'i', configs.numLdMem)
    rresp_ready_o = LogicArray(ctx, 'rresp_ready', 'o', configs.numLdMem)
    rresp_id_i = LogicVecArray(
        ctx, 'rresp_id', 'i', configs.numLdMem, configs.idW)
    rresp_data_i = LogicVecArray(
        ctx, 'rresp_data', 'i', configs.numLdMem, configs.dataW)

    wreq_valid_o = LogicArray(ctx, 'wreq_valid', 'o', configs.numStMem)
    wreq_ready_i = LogicArray(ctx, 'wreq_ready', 'i', configs.numStMem)
    wreq_id_o = LogicVecArray(
        ctx, 'wreq_id', 'o', configs.numStMem, configs.idW)
    wreq_addr_o = LogicVecArray(
        ctx, 'wreq_addr', 'o', configs.numStMem, configs.addrW)
    wreq_data_o = LogicVecArray(
        ctx, 'wreq_data', 'o', configs.numStMem, configs.dataW)

    wresp_valid_i = LogicArray(ctx, 'wresp_valid', 'i', configs.numStMem)
    wresp_ready_o = LogicArray(ctx, 'wresp_ready', 'o', configs.numStMem)
    wresp_id_i = LogicVecArray(
        ctx, 'wresp_id', 'i', configs.numStMem, configs.idW)

    #! If this is the lsq master, then we need the following logic
    #! Define new interfaces needed by dynamatic
    if (configs.master):
        memStart_ready = Logic(ctx, 'memStart_ready', 'o')
        memStart_valid = Logic(ctx, 'memStart_valid', 'i')
        ctrlEnd_ready = Logic(ctx, 'ctrlEnd_ready', 'o')
        ctrlEnd_valid = Logic(ctx, 'ctrlEnd_valid', 'i')
        memEnd_ready = Logic(ctx, 'memEnd_ready', 'i')
        memEnd_valid = Logic(ctx, 'memEnd_valid', 'o')

        #! Add extra signals required
        memStartReady = Logic(ctx, 'memStartReady', 'w')
        memEndValid = Logic(ctx, 'memEndValid', 'w')
        ctrlEndReady = Logic(ctx, 'ctrlEndReady', 'w')
        temp_gen_mem = Logic(ctx, 'TEMP_GEN_MEM', 'w')

        #! Define the needed logic
        arch += "\t-- Define the intermediate logic\n"
        arch += f"\tTEMP_GEN_MEM <= {ctrlEnd_valid.getNameRead()} and stq_empty and ldq_empty;\n"

        arch += "\t-- Define logic for the new interfaces needed by dynamatic\n"
        arch += "\tprocess (clk) is\n\tbegin\n"
        arch += '\t' * 2 + "if rising_edge(clk) then\n"
        arch += '\t' * 3 + "if rst = '1' then\n"
        arch += '\t' * 4 + "memStartReady <= '1';\n"
        arch += '\t' * 4 + "memEndValid <= '0';\n"
        arch += '\t' * 4 + "ctrlEndReady <= '0';\n"
        arch += '\t' * 3 + "else\n"
        arch += '\t' * 4 + \
            "memStartReady <= (memEndValid and memEnd_ready_i) or ((not (memStart_valid_i and memStartReady)) and memStartReady);\n"
        arch += '\t' * 4 + "memEndValid <= TEMP_GEN_MEM or memEndValid;\n"
        arch += '\t' * 4 + \
            "ctrlEndReady <= (not (ctrlEnd_valid_i and ctrlEndReady)) and (TEMP_GEN_MEM or ctrlEndReady);\n"
        arch += '\t' * 3 + "end if;\n"
        arch += '\t' * 2 + "end if;\n"
        arch += "\tend process;\n\n"

        #! Assign signals for the newly added ports
        arch += "\t-- Update new memory interfaces\n"
        arch += Op(ctx, memStart_ready, memStartReady)
        arch += Op(ctx, ctrlEnd_ready, ctrlEndReady)
        arch += Op(ctx, memEnd_valid, memEndValid)

    ######  Queue Registers ######
    # Load Queue Entries
    ldq_valid = LogicArray(ctx, 'ldq_valid', 'r', configs.numLdqEntries)
    ldq_issue = LogicArray(ctx, 'ldq_issue', 'r', configs.numLdqEntries)
    if (configs.ldpAddrW > 0):
        ldq_port_idx = LogicVecArray(
            ctx, 'ldq_port_idx', 'r', configs.numLdqEntries, configs.ldpAddrW)
    else:
        ldq_port_idx = None
    ldq_addr_valid = LogicArray(
        ctx, 'ldq_addr_valid', 'r', configs.numLdqEntries)
    ldq_addr = LogicVecArray(ctx, 'ldq_addr', 'r',
                             configs.numLdqEntries, configs.addrW)
    ldq_data_valid = LogicArray(
        ctx, 'ldq_data_valid', 'r', configs.numLdqEntries)
    ldq_data = LogicVecArray(ctx, 'ldq_data', 'r',
                             configs.numLdqEntries, configs.dataW)

    # Store Queue Entries
    stq_valid = LogicArray(ctx, 'stq_valid', 'r', configs.numStqEntries)
    if configs.stResp:
        stq_exec = LogicArray(ctx, 'stq_exec', 'r', configs.numStqEntries)
    if (configs.stpAddrW > 0):
        stq_port_idx = LogicVecArray(
            ctx, 'stq_port_idx', 'r', configs.numStqEntries, configs.stpAddrW)
    else:
        stq_port_idx = None
    stq_addr_valid = LogicArray(
        ctx, 'stq_addr_valid', 'r', configs.numStqEntries)
    stq_addr = LogicVecArray(ctx, 'stq_addr', 'r',
                             configs.numStqEntries, configs.addrW)
    stq_data_valid = LogicArray(
        ctx, 'stq_data_valid', 'r', configs.numStqEntries)
    stq_data = LogicVecArray(ctx, 'stq_data', 'r',
                             configs.numStqEntries, configs.dataW)

    # Order for load-store
    store_is_older = LogicVecArray(
        ctx, 'store_is_older', 'r', configs.numLdqEntries, configs.numStqEntries)

    # Pointers
    ldq_tail = LogicVec(ctx, 'ldq_tail', 'r', configs.ldqAddrW)
    ldq_head = LogicVec(ctx, 'ldq_head', 'r', configs.ldqAddrW)

    stq_tail = LogicVec(ctx, 'stq_tail', 'r', configs.stqAddrW)
    stq_head = LogicVec(ctx, 'stq_head', 'r', configs.stqAddrW)
    stq_issue = LogicVec(ctx, 'stq_issue', 'r', configs.stqAddrW)
    stq_resp = LogicVec(ctx, 'stq_resp', 'r', configs.stqAddrW)

    # Entry related signals
    # From port dispatchers
    ldq_wen = LogicArray(ctx, 'ldq_wen', 'w', configs.numLdqEntries)
    ldq_addr_wen = LogicArray(ctx, 'ldq_addr_wen', 'w', configs.numLdqEntries)
    ldq_reset = LogicArray(ctx, 'ldq_reset', 'w', configs.numLdqEntries)
    stq_wen = LogicArray(ctx, 'stq_wen', 'w', configs.numStqEntries)
    stq_addr_wen = LogicArray(ctx, 'stq_addr_wen', 'w', configs.numStqEntries)
    stq_data_wen = LogicArray(ctx, 'stq_data_wen', 'w', configs.numStqEntries)
    stq_reset = LogicArray(ctx, 'stq_reset', 'w', configs.numStqEntries)
    # From Read/Write Block
    ldq_data_wen = LogicArray(ctx, 'ldq_data_wen', 'w', configs.numLdqEntries)
    ldq_issue_set = LogicArray(
        ctx, 'ldq_issue_set', 'w', configs.numLdqEntries)
    if configs.stResp:
        stq_exec_set = LogicArray(
            ctx, 'stq_exec_set', 'w', configs.numStqEntries)
    # Form Group Allocator
    ga_ls_order = LogicVecArray(
        ctx, 'ga_ls_order', 'w', configs.numLdqEntries, configs.numStqEntries)

    # Pointer related signals
    # For updating pointers
    num_loads = LogicVec(ctx, 'num_loads', 'w', configs.ldqAddrW)
    num_stores = LogicVec(ctx, 'num_stores', 'w', configs.stqAddrW)
    stq_issue_en = Logic(ctx, 'stq_issue_en', 'w')
    stq_resp_en = Logic(ctx, 'stq_resp_en', 'w')
    # Generated by pointers
    ldq_empty = Logic(ctx, 'ldq_empty', 'w')
    stq_empty = Logic(ctx, 'stq_empty', 'w')
    ldq_head_oh = LogicVec(ctx, 'ldq_head_oh', 'w', configs.numLdqEntries)
    stq_head_oh = LogicVec(ctx, 'stq_head_oh', 'w', configs.numStqEntries)

    arch += BitsToOH(ctx, ldq_head_oh, ldq_head)
    arch += BitsToOH(ctx, stq_head_oh, stq_head)

    # update queue entries
    # load queue
    if configs.pipe0 or configs.pipeComp:
        ldq_wen_p0 = LogicArray(ctx, 'ldq_wen_p0', 'r', configs.numLdqEntries)
        ldq_wen_p0.regInit()
        if configs.pipe0 and configs.pipeComp:
            ldq_wen_p1 = LogicArray(
                ctx, 'ldq_wen_p1', 'r', configs.numLdqEntries)
            ldq_wen_p1.regInit()
    ldq_valid_next = LogicArray(
        ctx, 'ldq_valid_next', 'w', configs.numLdqEntries)
    for i in range(0, configs.numLdqEntries):
        arch += Op(ctx, ldq_valid_next[i],
                   'not', ldq_reset[i], 'and', ldq_valid[i]
                   )
        arch += Op(ctx, ldq_valid[i],
                   ldq_wen[i], 'or', ldq_valid_next[i]
                   )
        if configs.pipe0 or configs.pipeComp:
            arch += Op(ctx, ldq_wen_p0[i], ldq_wen[i])
            if configs.pipe0 and configs.pipeComp:
                arch += Op(ctx, ldq_wen_p1[i], ldq_wen[i])
                arch += Op(ctx, ldq_issue[i],
                           'not', ldq_wen_p1[i], 'and',
                           '(', ldq_issue_set[i], 'or', ldq_issue[i], ')'
                           )
            else:
                arch += Op(ctx, ldq_issue[i],
                           'not', ldq_wen_p0[i], 'and',
                           '(', ldq_issue_set[i], 'or', ldq_issue[i], ')'
                           )
        else:
            arch += Op(ctx, ldq_issue[i],
                       'not', ldq_wen[i], 'and',
                       '(', ldq_issue_set[i], 'or', ldq_issue[i], ')'
                       )
        arch += Op(ctx, ldq_addr_valid[i],
                   'not', ldq_wen[i], 'and',
                   '(', ldq_addr_wen[i], 'or', ldq_addr_valid[i], ')'
                   )
        arch += Op(ctx, ldq_data_valid[i],
                   'not', ldq_wen[i], 'and',
                   '(', ldq_data_wen[i], 'or', ldq_data_valid[i], ')'
                   )
    # store queue
    stq_valid_next = LogicArray(
        ctx, 'stq_valid_next', 'w', configs.numStqEntries)
    for i in range(0, configs.numStqEntries):
        arch += Op(ctx, stq_valid_next[i],
                   'not', stq_reset[i], 'and', stq_valid[i]
                   )
        arch += Op(ctx, stq_valid[i],
                   stq_wen[i], 'or', stq_valid_next[i]
                   )
        if configs.stResp:
            arch += Op(ctx, stq_exec[i],
                       'not', stq_wen[i], 'and',
                       '(', stq_exec_set[i], 'or', stq_exec[i], ')'
                       )
        arch += Op(ctx, stq_addr_valid[i],
                   'not', stq_wen[i], 'and',
                   '(', stq_addr_wen[i], 'or', stq_addr_valid[i], ')'
                   )
        arch += Op(ctx, stq_data_valid[i],
                   'not', stq_wen[i], 'and',
                   '(', stq_data_wen[i], 'or', stq_data_valid[i], ')'
                   )

    # order matrix
    # store_is_older(i,j) = (not stq_reset(j) and (stq_valid(j) or ga_ls_order(i, j)))
    #                  when ldq_wen(i)
    #                  else not stq_reset(j) and store_is_older(i, j)
    for i in range(0, configs.numLdqEntries):
        for j in range(0, configs.numStqEntries):
            arch += Op(ctx, (store_is_older, i, j),
                       '(', 'not', (stq_reset, j), 'and', '(', (stq_valid,
                                                                j), 'or', (ga_ls_order, i, j), ')', ')',
                       'when', (ldq_wen, i), 'else',
                       'not', (stq_reset, j), 'and', (store_is_older, i, j)
                       )

    # pointers update
    ldq_not_empty = Logic(ctx, 'ldq_not_empty', 'w')
    stq_not_empty = Logic(ctx, 'stq_not_empty', 'w')
    arch += Reduce(ctx, ldq_not_empty, ldq_valid, 'or')
    arch += Op(ctx, ldq_empty, 'not', ldq_not_empty)
    arch += MuxLookUp(ctx, stq_not_empty, stq_valid, stq_head)
    arch += Op(ctx, stq_empty, 'not', stq_not_empty)
    arch += Op(ctx, empty_o, ldq_empty, 'and', stq_empty)

    arch += WrapAdd(ctx, ldq_tail, ldq_tail, num_loads, configs.numLdqEntries)
    arch += WrapAdd(ctx, stq_tail, stq_tail, num_stores, configs.numStqEntries)
    arch += WrapAddConst(ctx, stq_issue, stq_issue, 1, configs.numStqEntries)
    arch += WrapAddConst(ctx, stq_resp, stq_resp, 1, configs.numStqEntries)

    ldq_tail_oh = LogicVec(ctx, 'ldq_tail_oh', 'w', configs.numLdqEntries)
    arch += BitsToOH(ctx, ldq_tail_oh, ldq_tail)
    ldq_head_next_oh = LogicVec(
        ctx, 'ldq_head_next_oh', 'w', configs.numLdqEntries)
    ldq_head_next = LogicVec(ctx, 'ldq_head_next', 'w', configs.ldqAddrW)
    ldq_head_sel = Logic(ctx, 'ldq_head_sel', 'w')
    if configs.headLag:
        # Update the head pointer according to the valid signal of last cycle
        arch += CyclicPriorityMasking(ctx,
                                      ldq_head_next_oh, ldq_valid, ldq_tail_oh)
        arch += Reduce(ctx, ldq_head_sel, ldq_valid, 'or')
    else:
        arch += CyclicPriorityMasking(ctx, ldq_head_next_oh,
                                      ldq_valid_next, ldq_tail_oh)
        arch += Reduce(ctx, ldq_head_sel, ldq_valid_next, 'or')
    arch += OHToBits(ctx, ldq_head_next, ldq_head_next_oh)
    arch += Op(ctx, ldq_head, ldq_head_next, 'when',
               ldq_head_sel, 'else', ldq_tail)

    stq_tail_oh = LogicVec(ctx, 'stq_tail_oh', 'w', configs.numStqEntries)
    arch += BitsToOH(ctx, stq_tail_oh, stq_tail)
    stq_head_next_oh = LogicVec(
        ctx, 'stq_head_next_oh', 'w', configs.numStqEntries)
    stq_head_next = LogicVec(ctx, 'stq_head_next', 'w', configs.stqAddrW)
    stq_head_sel = Logic(ctx, 'stq_head_sel', 'w')
    if configs.stResp:
        if configs.headLag:
            # Update the head pointer according to the valid signal of last cycle
            arch += CyclicPriorityMasking(ctx,
                                          stq_head_next_oh, stq_valid, stq_tail_oh)
            arch += Reduce(ctx, stq_head_sel, stq_valid, 'or')
        else:
            arch += CyclicPriorityMasking(ctx, stq_head_next_oh,
                                          stq_valid_next, stq_tail_oh)
            arch += Reduce(ctx, stq_head_sel, stq_valid_next, 'or')
        arch += OHToBits(ctx, stq_head_next, stq_head_next_oh)
        arch += Op(ctx, stq_head, stq_head_next, 'when',
                   stq_head_sel, 'else', stq_tail)
    else:
        arch += WrapAddConst(ctx, stq_head_next, stq_head,
                             1, configs.numStqEntries)
        arch += Op(ctx, stq_head_sel, wresp_valid_i[0])
        arch += Op(ctx, stq_head, stq_head_next, 'when',
                   stq_head_sel, 'else', stq_head)

    # Load Queue Entries
    ldq_valid.regInit(init=[0]*configs.numLdqEntries)
    ldq_issue.regInit()
    if (configs.ldpAddrW > 0):
        ldq_port_idx.regInit(ldq_wen)
    ldq_addr_valid.regInit()
    ldq_addr.regInit(ldq_addr_wen)
    ldq_data_valid.regInit()
    ldq_data.regInit(ldq_data_wen)

    # Store Queue Entries
    stq_valid.regInit(init=[0]*configs.numStqEntries)
    if configs.stResp:
        stq_exec.regInit()
    if (configs.stpAddrW > 0):
        stq_port_idx.regInit(stq_wen)
    stq_addr_valid.regInit()
    stq_addr.regInit(stq_addr_wen)
    stq_data_valid.regInit()
    stq_data.regInit(stq_data_wen)

    # Order for load-store
    store_is_older.regInit()

    # Pointers
    ldq_tail.regInit(init=0)
    ldq_head.regInit(init=0)

    stq_tail.regInit(init=0)
    stq_head.regInit(init=0)
    stq_issue.regInit(enable=stq_issue_en, init=0)
    stq_resp.regInit(enable=stq_resp_en, init=0)

    ######   Entity Instantiation   ######

    # Group Allocator
    arch += GroupAllocatorInst(ctx, name + '_ga', configs,
                               group_init_valid_i, group_init_ready_o,
                               ldq_tail, ldq_head, ldq_empty,
                               stq_tail, stq_head, stq_empty,
                               ldq_wen, num_loads, ldq_port_idx,
                               stq_wen, num_stores, stq_port_idx,
                               ga_ls_order
                               )

    # Load Address Port Dispatcher
    arch += PortToQueueDispatcherInst(ctx, name + '_lda',
                                      configs.numLdPorts, configs.numLdqEntries,
                                      ldp_addr_i, ldp_addr_valid_i, ldp_addr_ready_o,
                                      ldq_valid, ldq_addr_valid, ldq_port_idx, ldq_addr, ldq_addr_wen, ldq_head_oh
                                      )
    # Load Data Port Dispatcher
    arch += QueueToPortDispatcherInst(ctx, name + '_ldd',
                                      configs.numLdPorts, configs.numLdqEntries,
                                      ldp_data_o, ldp_data_valid_o, ldp_data_ready_i,
                                      ldq_valid, ldq_data_valid, ldq_port_idx, ldq_data, ldq_reset, ldq_head_oh
                                      )
    # Store Address Port Dispatcher
    arch += PortToQueueDispatcherInst(ctx, name + '_sta',
                                      configs.numStPorts, configs.numStqEntries,
                                      stp_addr_i, stp_addr_valid_i, stp_addr_ready_o,
                                      stq_valid, stq_addr_valid, stq_port_idx, stq_addr, stq_addr_wen, stq_head_oh
                                      )
    # Store Data Port Dispatcher
    arch += PortToQueueDispatcherInst(ctx, name + '_std',
                                      configs.numStPorts, configs.numStqEntries,
                                      stp_data_i, stp_data_valid_i, stp_data_ready_o,
                                      stq_valid, stq_data_valid, stq_port_idx, stq_data, stq_data_wen, stq_head_oh
                                      )
    # Store Backward Port Dispatcher
    if configs.stResp:
        arch += QueueToPortDispatcherInst(ctx, name + '_stb',
                                          configs.numStPorts, configs.numStqEntries,
                                          None, stp_exec_valid_o, stp_exec_ready_i,
                                          stq_valid, stq_exec, stq_port_idx, None, stq_reset, stq_head_oh
                                          )

    if configs.pipe0:
        ###### Dependency Check ######
        load_idx_oh = LogicVecArray(
            ctx, 'load_idx_oh', 'w', configs.numLdMem, configs.numLdqEntries)
        load_en = LogicArray(ctx, 'load_en', 'w', configs.numLdMem)

        # Multiple store channels not yet implemented
        assert (configs.numStMem == 1)
        store_idx = LogicVec(ctx, 'store_idx', 'w', configs.stqAddrW)
        store_en = Logic(ctx, 'store_en', 'w')

        bypass_idx_oh_p0 = LogicVecArray(
            ctx, 'bypass_idx_oh_p0', 'r', configs.numLdqEntries, configs.numStqEntries)
        bypass_idx_oh_p0.regInit()
        bypass_en = LogicArray(ctx, 'bypass_en', 'w', configs.numLdqEntries)

        # Matrix Generation
        ld_st_conflict = LogicVecArray(
            ctx, 'ld_st_conflict', 'w', configs.numLdqEntries, configs.numStqEntries)
        can_bypass = LogicVecArray(
            ctx, 'can_bypass', 'w', configs.numLdqEntries, configs.numStqEntries)
        can_bypass_p0 = LogicVecArray(
            ctx, 'can_bypass_p0', 'r', configs.numLdqEntries, configs.numStqEntries)
        can_bypass_p0.regInit(init=[0]*configs.numLdqEntries)

        if configs.pipeComp:
            ldq_valid_pcomp = LogicArray(
                ctx, 'ldq_valid_pcomp', 'r', configs.numLdqEntries)
            ldq_addr_valid_pcomp = LogicArray(
                ctx, 'ldq_addr_valid_pcomp', 'r', configs.numLdqEntries)
            stq_valid_pcomp = LogicArray(
                ctx, 'stq_valid_pcomp', 'r', configs.numStqEntries)
            stq_addr_valid_pcomp = LogicArray(
                ctx, 'stq_addr_valid_pcomp', 'r', configs.numStqEntries)
            stq_data_valid_pcomp = LogicArray(
                ctx, 'stq_data_valid_pcomp', 'r', configs.numStqEntries)
            addr_valid_pcomp = LogicVecArray(
                ctx, 'addr_valid_pcomp', 'w', configs.numLdqEntries, configs.numStqEntries)
            addr_same_pcomp = LogicVecArray(
                ctx, 'addr_same_pcomp', 'r', configs.numLdqEntries, configs.numStqEntries)
            store_is_older_pcomp = LogicVecArray(
                ctx, 'store_is_older_pcomp', 'r', configs.numLdqEntries, configs.numStqEntries)

            ldq_valid_pcomp.regInit(init=[0]*configs.numLdqEntries)
            ldq_addr_valid_pcomp.regInit()
            stq_valid_pcomp.regInit(init=[0]*configs.numStqEntries)
            stq_addr_valid_pcomp.regInit()
            stq_data_valid_pcomp.regInit()
            addr_same_pcomp.regInit()
            store_is_older_pcomp.regInit()

            for i in range(0, configs.numLdqEntries):
                arch += Op(ctx, (ldq_valid_pcomp, i), (ldq_valid, i))
                arch += Op(ctx, (ldq_addr_valid_pcomp, i), (ldq_addr_valid, i))
            for j in range(0, configs.numStqEntries):
                arch += Op(ctx, (stq_valid_pcomp, j), (stq_valid, j))
                arch += Op(ctx, (stq_addr_valid_pcomp, j), (stq_addr_valid, j))
                arch += Op(ctx, (stq_data_valid_pcomp, j), (stq_data_valid, j))
            for i in range(0, configs.numLdqEntries):
                for j in range(0, configs.numStqEntries):
                    arch += Op(ctx, (store_is_older_pcomp, i, j),
                               (store_is_older, i, j))
            for i in range(0, configs.numLdqEntries):
                for j in range(0, configs.numStqEntries):
                    arch += Op(ctx, (addr_valid_pcomp, i, j),
                               (ldq_addr_valid_pcomp, i), 'and', (stq_addr_valid_pcomp, j))
            for i in range(0, configs.numLdqEntries):
                for j in range(0, configs.numStqEntries):
                    arch += Op(ctx, (addr_same_pcomp, i, j), '\'1\'', 'when',
                               (ldq_addr, i), '=', (stq_addr, j), 'else', '\'0\'')

            # A load conflicts with a store when:
            # 1. The store entry is valid, and
            # 2. The store is older than the load, and
            # 3. The address conflicts(same or invalid store address).
            for i in range(0, configs.numLdqEntries):
                for j in range(0, configs.numStqEntries):
                    arch += Op(ctx,
                               (ld_st_conflict, i, j),
                               (stq_valid_pcomp, j),   'and',
                               (store_is_older_pcomp, i, j), 'and',
                               '(', (addr_same_pcomp, i,
                                     j), 'or', 'not', (stq_addr_valid_pcomp, j), ')'
                               )

            # A conflicting store entry can be bypassed to a load entry when:
            # 1. The load entry is valid, and
            # 2. The load entry is not issued yet, and
            # 3. The address of the load-store pair are both valid and values the same.
            for i in range(0, configs.numLdqEntries):
                for j in range(0, configs.numStqEntries):
                    arch += Op(ctx,
                               (can_bypass_p0, i, j),
                               (ldq_valid_pcomp, i),        'and',
                               (stq_data_valid_pcomp, j),   'and',
                               (addr_same_pcomp, i, j),     'and',
                               (addr_valid_pcomp, i, j)
                               )
            for i in range(0, configs.numLdqEntries):
                for j in range(0, configs.numStqEntries):
                    arch += Op(ctx,
                               (can_bypass, i, j),
                               'not', (ldq_issue, i), 'and',
                               (can_bypass_p0, i, j)
                               )

            # Load

            load_conflict = LogicArray(
                ctx, 'load_conflict', 'w', configs.numLdqEntries)
            load_req_valid = LogicArray(
                ctx, 'load_req_valid', 'w', configs.numLdqEntries)
            can_load = LogicArray(ctx, 'can_load', 'w', configs.numLdqEntries)
            can_load_p0 = LogicArray(
                ctx, 'can_load_p0', 'r', configs.numLdqEntries)
            can_load_p0.regInit(init=[0]*configs.numLdqEntries)

            # The load conflicts with any store
            for i in range(0, configs.numLdqEntries):
                arch += Reduce(ctx, load_conflict[i], ld_st_conflict[i], 'or')
            # The load is valid when the entry is valid and not yet issued, the load address should also be valid.
            # We do not need to check ldq_data_valid, since unissued load request cannot have valid data.
            for i in range(0, configs.numLdqEntries):
                arch += Op(ctx, load_req_valid[i], ldq_valid_pcomp[i],
                           'and', ldq_addr_valid_pcomp[i])
            # Generate list for loads that does not face dependency issue
            for i in range(0, configs.numLdqEntries):
                arch += Op(ctx, can_load_p0[i], 'not',
                           load_conflict[i], 'and', load_req_valid[i])
            for i in range(0, configs.numLdqEntries):
                arch += Op(ctx, can_load[i], 'not',
                           ldq_issue[i], 'and', can_load_p0[i])

            ldq_head_oh_p0 = LogicVec(
                ctx, 'ldq_head_oh_p0', 'r', configs.numLdqEntries)
            ldq_head_oh_p0.regInit()
            arch += Op(ctx, ldq_head_oh_p0, ldq_head_oh)

            can_load_list = []
            can_load_list.append(can_load)
            for w in range(0, configs.numLdMem):
                arch += CyclicPriorityMasking(
                    ctx, load_idx_oh[w], can_load_list[w], ldq_head_oh_p0)
                arch += Reduce(ctx, load_en[w], can_load_list[w], 'or')
                if (w+1 != configs.numLdMem):
                    load_idx_oh_LogicArray = LogicArray(
                        ctx, f'load_idx_oh_Array_{w+1}', 'w', configs.numLdqEntries)
                    arch += VecToArray(ctx,
                                       load_idx_oh_LogicArray, load_idx_oh[w])
                    can_load_list.append(LogicArray(
                        ctx, f'can_load_list_{w+1}', 'w', configs.numLdqEntries))
                    for i in range(0, configs.numLdqEntries):
                        arch += Op(ctx, can_load_list[w+1][i], 'not',
                                   load_idx_oh_LogicArray[i], 'and', can_load_list[w][i])

            # Store
            stq_issue_en_p0 = Logic(ctx, 'stq_issue_en_p0', 'r')
            stq_issue_next = LogicVec(
                ctx, 'stq_issue_next', 'w', configs.stqAddrW)

            store_conflict = Logic(ctx, 'store_conflict', 'w')

            can_store_curr = Logic(ctx, 'can_store_curr', 'w')
            st_ld_conflict_curr = LogicVec(
                ctx, 'st_ld_conflict_curr', 'w', configs.numLdqEntries)
            store_valid_curr = Logic(ctx, 'store_valid_curr', 'w')
            store_data_valid_curr = Logic(ctx, 'store_data_valid_curr', 'w')
            store_addr_valid_curr = Logic(ctx, 'store_addr_valid_curr', 'w')

            can_store_next = Logic(ctx, 'can_store_next', 'w')
            st_ld_conflict_next = LogicVec(
                ctx, 'st_ld_conflict_next', 'w', configs.numLdqEntries)
            store_valid_next = Logic(ctx, 'store_valid_next', 'w')
            store_data_valid_next = Logic(ctx, 'store_data_valid_next', 'w')
            store_addr_valid_next = Logic(ctx, 'store_addr_valid_next', 'w')

            can_store_p0 = Logic(ctx, 'can_store_p0', 'r')
            st_ld_conflict_p0 = LogicVec(
                ctx, 'st_ld_conflict_p0', 'r', configs.numLdqEntries)

            stq_issue_en_p0.regInit(init=0)
            can_store_p0.regInit(init=0)
            st_ld_conflict_p0.regInit()

            arch += Op(ctx, stq_issue_en_p0, stq_issue_en)
            arch += WrapAddConst(ctx, stq_issue_next,
                                 stq_issue, 1, configs.numStqEntries)

            # A store conflicts with a load when:
            # 1. The load entry is valid, and
            # 2. The load is older than the store, and
            # 3. The address conflicts(same or invalid store address).
            # Index order are reversed for store matrix.
            for i in range(0, configs.numLdqEntries):
                arch += Op(ctx,
                           (st_ld_conflict_curr, i),
                           (ldq_valid_pcomp, i), 'and',
                           'not', MuxIndex(
                               store_is_older_pcomp[i], stq_issue), 'and',
                           '(', MuxIndex(
                               addr_same_pcomp[i], stq_issue), 'or', 'not', (ldq_addr_valid_pcomp, i), ')'
                           )
            for i in range(0, configs.numLdqEntries):
                arch += Op(ctx,
                           (st_ld_conflict_next, i),
                           (ldq_valid_pcomp, i), 'and',
                           'not', MuxIndex(
                               store_is_older_pcomp[i], stq_issue_next), 'and',
                           '(', MuxIndex(
                               addr_same_pcomp[i], stq_issue_next), 'or', 'not', (ldq_addr_valid_pcomp, i), ')'
                           )
            # The store is valid whe the entry is valid and the data is also valid,
            # the store address should also be valid
            arch += MuxLookUp(ctx, store_valid_curr,
                              stq_valid_pcomp, stq_issue)
            arch += MuxLookUp(ctx, store_data_valid_curr,
                              stq_data_valid_pcomp, stq_issue)
            arch += MuxLookUp(ctx, store_addr_valid_curr,
                              stq_addr_valid_pcomp, stq_issue)
            arch += Op(ctx, can_store_curr,
                       store_valid_curr, 'and',
                       store_data_valid_curr, 'and',
                       store_addr_valid_curr
                       )
            arch += MuxLookUp(ctx, store_valid_next,
                              stq_valid_pcomp, stq_issue_next)
            arch += MuxLookUp(ctx, store_data_valid_next,
                              stq_data_valid_pcomp, stq_issue_next)
            arch += MuxLookUp(ctx, store_addr_valid_next,
                              stq_addr_valid_pcomp, stq_issue_next)
            arch += Op(ctx, can_store_next,
                       store_valid_next, 'and',
                       store_data_valid_next, 'and',
                       store_addr_valid_next
                       )
            # Multiplex from current and next
            arch += Op(ctx, st_ld_conflict_p0, st_ld_conflict_next,
                       'when', stq_issue_en, 'else', st_ld_conflict_curr)
            arch += Op(ctx, can_store_p0, can_store_next, 'when',
                       stq_issue_en, 'else', can_store_curr)
            # The store conflicts with any load
            arch += Reduce(ctx, store_conflict, st_ld_conflict_p0, 'or')
            arch += Op(ctx, store_en, 'not',
                       store_conflict, 'and', can_store_p0)

            arch += Op(ctx, store_idx, stq_issue)

            # Bypass
            stq_last_oh = LogicVec(
                ctx, 'stq_last_oh', 'w', configs.numStqEntries)
            arch += BitsToOHSub1(ctx, stq_last_oh, stq_tail)
            for i in range(0, configs.numLdqEntries):
                bypass_en_vec = LogicVec(
                    ctx, f'bypass_en_vec_{i}', 'w', configs.numStqEntries)
                # Search for the youngest store that is older than the load and conflicts
                arch += CyclicPriorityMasking(
                    ctx, bypass_idx_oh_p0[i], ld_st_conflict[i], stq_last_oh, True)
                # Check if the youngest conflict store can bypass with the load
                arch += Op(ctx, bypass_en_vec,
                           bypass_idx_oh_p0[i], 'and', can_bypass[i])
                arch += Reduce(ctx, bypass_en[i], bypass_en_vec, 'or')
        else:
            addr_valid = LogicVecArray(
                ctx, 'addr_valid', 'w', configs.numLdqEntries, configs.numStqEntries)
            addr_same = LogicVecArray(
                ctx, 'addr_same', 'w', configs.numLdqEntries, configs.numStqEntries)

            for i in range(0, configs.numLdqEntries):
                for j in range(0, configs.numStqEntries):
                    arch += Op(ctx, (addr_valid, i, j),
                               (ldq_addr_valid, i), 'and', (stq_addr_valid, j))
            for i in range(0, configs.numLdqEntries):
                for j in range(0, configs.numStqEntries):
                    arch += Op(ctx, (addr_same, i, j), '\'1\'', 'when',
                               (ldq_addr, i), '=', (stq_addr, j), 'else', '\'0\'')

            # A load conflicts with a store when:
            # 1. The store entry is valid, and
            # 2. The store is older than the load, and
            # 3. The address conflicts(same or invalid store address).
            for i in range(0, configs.numLdqEntries):
                for j in range(0, configs.numStqEntries):
                    arch += Op(ctx,
                               (ld_st_conflict, i, j),
                               (stq_valid, j),         'and',
                               (store_is_older, i, j), 'and',
                               '(', (addr_same, i, j), 'or', 'not', (stq_addr_valid, j), ')'
                               )

            # A conflicting store entry can be bypassed to a load entry when:
            # 1. The load entry is valid, and
            # 2. The load entry is not issued yet, and
            # 3. The address of the load-store pair are both valid and values the same.
            for i in range(0, configs.numLdqEntries):
                for j in range(0, configs.numStqEntries):
                    arch += Op(ctx,
                               (can_bypass_p0, i, j),
                               (ldq_valid, i),        'and',
                               (stq_data_valid, j),   'and',
                               (addr_same, i, j),     'and',
                               (addr_valid, i, j)
                               )
            for i in range(0, configs.numLdqEntries):
                for j in range(0, configs.numStqEntries):
                    arch += Op(ctx,
                               (can_bypass, i, j),
                               'not', (ldq_issue, i), 'and',
                               (can_bypass_p0, i, j)
                               )

            # Load

            load_conflict = LogicArray(
                ctx, 'load_conflict', 'w', configs.numLdqEntries)
            load_req_valid = LogicArray(
                ctx, 'load_req_valid', 'w', configs.numLdqEntries)
            can_load = LogicArray(ctx, 'can_load', 'w', configs.numLdqEntries)
            can_load_p0 = LogicArray(
                ctx, 'can_load_p0', 'r', configs.numLdqEntries)
            can_load_p0.regInit(init=[0]*configs.numLdqEntries)

            # The load conflicts with any store
            for i in range(0, configs.numLdqEntries):
                arch += Reduce(ctx, load_conflict[i], ld_st_conflict[i], 'or')
            # The load is valid when the entry is valid and not yet issued, the load address should also be valid.
            # We do not need to check ldq_data_valid, since unissued load request cannot have valid data.
            for i in range(0, configs.numLdqEntries):
                arch += Op(ctx, load_req_valid[i],
                           ldq_valid[i], 'and', ldq_addr_valid[i])
            # Generate list for loads that does not face dependency issue
            for i in range(0, configs.numLdqEntries):
                arch += Op(ctx, can_load_p0[i], 'not',
                           load_conflict[i], 'and', load_req_valid[i])
            for i in range(0, configs.numLdqEntries):
                arch += Op(ctx, can_load[i], 'not',
                           ldq_issue[i], 'and', can_load_p0[i])

            ldq_head_oh_p0 = LogicVec(
                ctx, 'ldq_head_oh_p0', 'r', configs.numLdqEntries)
            ldq_head_oh_p0.regInit()
            arch += Op(ctx, ldq_head_oh_p0, ldq_head_oh)

            can_load_list = []
            can_load_list.append(can_load)
            for w in range(0, configs.numLdMem):
                arch += CyclicPriorityMasking(
                    ctx, load_idx_oh[w], can_load_list[w], ldq_head_oh_p0)
                arch += Reduce(ctx, load_en[w], can_load_list[w], 'or')
                if (w+1 != configs.numLdMem):
                    load_idx_oh_LogicArray = LogicArray(
                        ctx, f'load_idx_oh_Array_{w+1}', 'w', configs.numLdqEntries)
                    arch += VecToArray(ctx,
                                       load_idx_oh_LogicArray, load_idx_oh[w])
                    can_load_list.append(LogicArray(
                        ctx, f'can_load_list_{w+1}', 'w', configs.numLdqEntries))
                    for i in range(0, configs.numLdqEntries):
                        arch += Op(ctx, can_load_list[w+1][i], 'not',
                                   load_idx_oh_LogicArray[i], 'and', can_load_list[w][i])

            # Store
            stq_issue_en_p0 = Logic(ctx, 'stq_issue_en_p0', 'r')
            stq_issue_next = LogicVec(
                ctx, 'stq_issue_next', 'w', configs.stqAddrW)

            store_conflict = Logic(ctx, 'store_conflict', 'w')

            can_store_curr = Logic(ctx, 'can_store_curr', 'w')
            st_ld_conflict_curr = LogicVec(
                ctx, 'st_ld_conflict_curr', 'w', configs.numLdqEntries)
            store_valid_curr = Logic(ctx, 'store_valid_curr', 'w')
            store_data_valid_curr = Logic(ctx, 'store_data_valid_curr', 'w')
            store_addr_valid_curr = Logic(ctx, 'store_addr_valid_curr', 'w')

            can_store_next = Logic(ctx, 'can_store_next', 'w')
            st_ld_conflict_next = LogicVec(
                ctx, 'st_ld_conflict_next', 'w', configs.numLdqEntries)
            store_valid_next = Logic(ctx, 'store_valid_next', 'w')
            store_data_valid_next = Logic(ctx, 'store_data_valid_next', 'w')
            store_addr_valid_next = Logic(ctx, 'store_addr_valid_next', 'w')

            can_store_p0 = Logic(ctx, 'can_store_p0', 'r')
            st_ld_conflict_p0 = LogicVec(
                ctx, 'st_ld_conflict_p0', 'r', configs.numLdqEntries)

            stq_issue_en_p0.regInit(init=0)
            can_store_p0.regInit(init=0)
            st_ld_conflict_p0.regInit()

            arch += Op(ctx, stq_issue_en_p0, stq_issue_en)
            arch += WrapAddConst(ctx, stq_issue_next,
                                 stq_issue, 1, configs.numStqEntries)

            # A store conflicts with a load when:
            # 1. The load entry is valid, and
            # 2. The load is older than the store, and
            # 3. The address conflicts(same or invalid store address).
            # Index order are reversed for store matrix.
            for i in range(0, configs.numLdqEntries):
                arch += Op(ctx,
                           (st_ld_conflict_curr, i),
                           (ldq_valid, i), 'and',
                           'not', MuxIndex(
                               store_is_older[i], stq_issue), 'and',
                           '(', MuxIndex(
                               addr_same[i], stq_issue), 'or', 'not', (ldq_addr_valid, i), ')'
                           )
            for i in range(0, configs.numLdqEntries):
                arch += Op(ctx,
                           (st_ld_conflict_next, i),
                           (ldq_valid, i), 'and',
                           'not', MuxIndex(
                               store_is_older[i], stq_issue_next), 'and',
                           '(', MuxIndex(
                               addr_same[i], stq_issue_next), 'or', 'not', (ldq_addr_valid, i), ')'
                           )
            # The store is valid whe the entry is valid and the data is also valid,
            # the store address should also be valid
            arch += MuxLookUp(ctx, store_valid_curr, stq_valid, stq_issue)
            arch += MuxLookUp(ctx, store_data_valid_curr,
                              stq_data_valid, stq_issue)
            arch += MuxLookUp(ctx, store_addr_valid_curr,
                              stq_addr_valid, stq_issue)
            arch += Op(ctx, can_store_curr,
                       store_valid_curr, 'and',
                       store_data_valid_curr, 'and',
                       store_addr_valid_curr
                       )
            arch += MuxLookUp(ctx, store_valid_next, stq_valid, stq_issue_next)
            arch += MuxLookUp(ctx, store_data_valid_next,
                              stq_data_valid, stq_issue_next)
            arch += MuxLookUp(ctx, store_addr_valid_next,
                              stq_addr_valid, stq_issue_next)
            arch += Op(ctx, can_store_next,
                       store_valid_next, 'and',
                       store_data_valid_next, 'and',
                       store_addr_valid_next
                       )
            # Multiplex from current and next
            arch += Op(ctx, st_ld_conflict_p0, st_ld_conflict_next,
                       'when', stq_issue_en, 'else', st_ld_conflict_curr)
            arch += Op(ctx, can_store_p0, can_store_next, 'when',
                       stq_issue_en, 'else', can_store_curr)
            # The store conflicts with any load
            arch += Reduce(ctx, store_conflict, st_ld_conflict_p0, 'or')
            arch += Op(ctx, store_en, 'not',
                       store_conflict, 'and', can_store_p0)

            arch += Op(ctx, store_idx, stq_issue)

            # Bypass
            stq_last_oh = LogicVec(
                ctx, 'stq_last_oh', 'w', configs.numStqEntries)
            arch += BitsToOHSub1(ctx, stq_last_oh, stq_tail)
            for i in range(0, configs.numLdqEntries):
                bypass_en_vec = LogicVec(
                    ctx, f'bypass_en_vec_{i}', 'w', configs.numStqEntries)
                # Search for the youngest store that is older than the load and conflicts
                arch += CyclicPriorityMasking(
                    ctx, bypass_idx_oh_p0[i], ld_st_conflict[i], stq_last_oh, True)
                # Check if the youngest conflict store can bypass with the load
                arch += Op(ctx, bypass_en_vec,
                           bypass_idx_oh_p0[i], 'and', can_bypass[i])
                arch += Reduce(ctx, bypass_en[i], bypass_en_vec, 'or')
    else:
        ###### Dependency Check ######

        load_idx_oh = LogicVecArray(
            ctx, 'load_idx_oh', 'w', configs.numLdMem, configs.numLdqEntries)
        load_en = LogicArray(ctx, 'load_en', 'w', configs.numLdMem)

        # Multiple store channels not yet implemented
        assert (configs.numStMem == 1)
        store_idx = LogicVec(ctx, 'store_idx', 'w', configs.stqAddrW)
        store_en = Logic(ctx, 'store_en', 'w')

        bypass_idx_oh = LogicVecArray(
            ctx, 'bypass_idx_oh', 'w', configs.numLdqEntries, configs.numStqEntries)
        bypass_en = LogicArray(ctx, 'bypass_en', 'w', configs.numLdqEntries)

        # Matrix Generation
        ld_st_conflict = LogicVecArray(
            ctx, 'ld_st_conflict', 'w', configs.numLdqEntries, configs.numStqEntries)
        can_bypass = LogicVecArray(
            ctx, 'can_bypass', 'w', configs.numLdqEntries, configs.numStqEntries)

        if configs.pipeComp:
            ldq_valid_pcomp = LogicArray(
                ctx, 'ldq_valid_pcomp', 'r', configs.numLdqEntries)
            ldq_addr_valid_pcomp = LogicArray(
                ctx, 'ldq_addr_valid_pcomp', 'r', configs.numLdqEntries)
            stq_valid_pcomp = LogicArray(
                ctx, 'stq_valid_pcomp', 'r', configs.numStqEntries)
            stq_addr_valid_pcomp = LogicArray(
                ctx, 'stq_addr_valid_pcomp', 'r', configs.numStqEntries)
            stq_data_valid_pcomp = LogicArray(
                ctx, 'stq_data_valid_pcomp', 'r', configs.numStqEntries)
            addr_valid_pcomp = LogicVecArray(
                ctx, 'addr_valid_pcomp', 'w', configs.numLdqEntries, configs.numStqEntries)
            addr_same_pcomp = LogicVecArray(
                ctx, 'addr_same_pcomp', 'r', configs.numLdqEntries, configs.numStqEntries)
            store_is_older_pcomp = LogicVecArray(
                ctx, 'store_is_older_pcomp', 'r', configs.numLdqEntries, configs.numStqEntries)

            ldq_valid_pcomp.regInit(init=[0]*configs.numLdqEntries)
            ldq_addr_valid_pcomp.regInit()
            stq_valid_pcomp.regInit(init=[0]*configs.numStqEntries)
            stq_addr_valid_pcomp.regInit()
            stq_data_valid_pcomp.regInit()
            addr_same_pcomp.regInit()
            store_is_older_pcomp.regInit()

            for i in range(0, configs.numLdqEntries):
                arch += Op(ctx, (ldq_valid_pcomp, i), (ldq_valid, i))
                arch += Op(ctx, (ldq_addr_valid_pcomp, i), (ldq_addr_valid, i))
            for j in range(0, configs.numStqEntries):
                arch += Op(ctx, (stq_valid_pcomp, j), (stq_valid, j))
                arch += Op(ctx, (stq_addr_valid_pcomp, j), (stq_addr_valid, j))
                arch += Op(ctx, (stq_data_valid_pcomp, j), (stq_data_valid, j))
            for i in range(0, configs.numLdqEntries):
                for j in range(0, configs.numStqEntries):
                    arch += Op(ctx, (store_is_older_pcomp, i, j),
                               (store_is_older, i, j))
            for i in range(0, configs.numLdqEntries):
                for j in range(0, configs.numStqEntries):
                    arch += Op(ctx, (addr_valid_pcomp, i, j),
                               (ldq_addr_valid_pcomp, i), 'and', (stq_addr_valid_pcomp, j))
            for i in range(0, configs.numLdqEntries):
                for j in range(0, configs.numStqEntries):
                    arch += Op(ctx, (addr_same_pcomp, i, j), '\'1\'', 'when',
                               (ldq_addr, i), '=', (stq_addr, j), 'else', '\'0\'')

            # A load conflicts with a store when:
            # 1. The store entry is valid, and
            # 2. The store is older than the load, and
            # 3. The address conflicts(same or invalid store address).
            for i in range(0, configs.numLdqEntries):
                for j in range(0, configs.numStqEntries):
                    arch += Op(ctx,
                               (ld_st_conflict, i, j),
                               (stq_valid_pcomp, j),         'and',
                               (store_is_older_pcomp, i, j), 'and',
                               '(', (addr_same_pcomp, i,
                                     j), 'or', 'not', (stq_addr_valid_pcomp, j), ')'
                               )

            # A conflicting store entry can be bypassed to a load entry when:
            # 1. The load entry is valid, and
            # 2. The load entry is not issued yet, and
            # 3. The address of the load-store pair are both valid and values the same.
            for i in range(0, configs.numLdqEntries):
                for j in range(0, configs.numStqEntries):
                    arch += Op(ctx,
                               (can_bypass, i, j),
                               (ldq_valid_pcomp, i),        'and',
                               'not', (ldq_issue, i),       'and',
                               (stq_data_valid_pcomp, j),   'and',
                               (addr_same_pcomp, i, j),     'and',
                               (addr_valid_pcomp, i, j)
                               )

            # Load

            load_conflict = LogicArray(
                ctx, 'load_conflict', 'w', configs.numLdqEntries)
            load_req_valid = LogicArray(
                ctx, 'load_req_valid', 'w', configs.numLdqEntries)
            can_load = LogicArray(ctx, 'can_load', 'w', configs.numLdqEntries)

            # The load conflicts with any store
            for i in range(0, configs.numLdqEntries):
                arch += Reduce(ctx, load_conflict[i], ld_st_conflict[i], 'or')
            # The load is valid when the entry is valid and not yet issued, the load address should also be valid.
            # We do not need to check ldq_data_valid, since unissued load request cannot have valid data.
            for i in range(0, configs.numLdqEntries):
                arch += Op(ctx, load_req_valid[i], ldq_valid_pcomp[i], 'and',
                           'not', ldq_issue[i], 'and', ldq_addr_valid_pcomp[i])
            # Generate list for loads that does not face dependency issue
            for i in range(0, configs.numLdqEntries):
                arch += Op(ctx, can_load[i], 'not',
                           load_conflict[i], 'and', load_req_valid[i])

            can_load_list = []
            can_load_list.append(can_load)
            for w in range(0, configs.numLdMem):
                arch += CyclicPriorityMasking(ctx,
                                              load_idx_oh[w], can_load_list[w], ldq_head_oh)
                arch += Reduce(ctx, load_en[w], can_load_list[w], 'or')
                if (w+1 != configs.numLdMem):
                    load_idx_oh_LogicArray = LogicArray(
                        ctx, f'load_idx_oh_Array_{w+1}', 'w', configs.numLdqEntries)
                    arch += VecToArray(ctx,
                                       load_idx_oh_LogicArray, load_idx_oh[w])
                    can_load_list.append(LogicArray(
                        ctx, f'can_load_list_{w+1}', 'w', configs.numLdqEntries))
                    for i in range(0, configs.numLdqEntries):
                        arch += Op(ctx, can_load_list[w+1][i], 'not',
                                   load_idx_oh_LogicArray[i], 'and', can_load_list[w][i])

            # Store

            st_ld_conflict = LogicVec(
                ctx, 'st_ld_conflict', 'w', configs.numLdqEntries)
            store_conflict = Logic(ctx, 'store_conflict', 'w')
            store_valid = Logic(ctx, 'store_valid', 'w')
            store_data_valid = Logic(ctx, 'store_data_valid', 'w')
            store_addr_valid = Logic(ctx, 'store_addr_valid', 'w')

            # A store conflicts with a load when:
            # 1. The load entry is valid, and
            # 2. The load is older than the store, and
            # 3. The address conflicts(same or invalid store address).
            # Index order are reversed for store matrix.
            for i in range(0, configs.numLdqEntries):
                arch += Op(ctx,
                           (st_ld_conflict, i),
                           (ldq_valid_pcomp, i), 'and',
                           'not', MuxIndex(
                               store_is_older_pcomp[i], stq_issue), 'and',
                           '(', MuxIndex(
                               addr_same_pcomp[i], stq_issue), 'or', 'not', (ldq_addr_valid_pcomp, i), ')'
                           )
            # The store conflicts with any load
            arch += Reduce(ctx, store_conflict, st_ld_conflict, 'or')
            # The store is valid whe the entry is valid and the data is also valid,
            # the store address should also be valid
            arch += MuxLookUp(ctx, store_valid, stq_valid_pcomp, stq_issue)
            arch += MuxLookUp(ctx, store_data_valid,
                              stq_data_valid_pcomp, stq_issue)
            arch += MuxLookUp(ctx, store_addr_valid,
                              stq_addr_valid_pcomp, stq_issue)
            arch += Op(ctx, store_en,
                       'not', store_conflict, 'and',
                       store_valid, 'and',
                       store_data_valid, 'and',
                       store_addr_valid
                       )
            arch += Op(ctx, store_idx, stq_issue)

            stq_last_oh = LogicVec(
                ctx, 'stq_last_oh', 'w', configs.numStqEntries)
            arch += BitsToOHSub1(ctx, stq_last_oh, stq_tail)
            for i in range(0, configs.numLdqEntries):
                bypass_en_vec = LogicVec(
                    ctx, f'bypass_en_vec_{i}', 'w', configs.numStqEntries)
                # Search for the youngest store that is older than the load and conflicts
                arch += CyclicPriorityMasking(
                    ctx, bypass_idx_oh[i], ld_st_conflict[i], stq_last_oh, True)
                # Check if the youngest conflict store can bypass with the load
                arch += Op(ctx, bypass_en_vec,
                           bypass_idx_oh[i], 'and', can_bypass[i])
                arch += Reduce(ctx, bypass_en[i], bypass_en_vec, 'or')
        else:
            addr_valid = LogicVecArray(
                ctx, 'addr_valid', 'w', configs.numLdqEntries, configs.numStqEntries)
            addr_same = LogicVecArray(
                ctx, 'addr_same', 'w', configs.numLdqEntries, configs.numStqEntries)

            for i in range(0, configs.numLdqEntries):
                for j in range(0, configs.numStqEntries):
                    arch += Op(ctx, (addr_valid, i, j),
                               (ldq_addr_valid, i), 'and', (stq_addr_valid, j))
            for i in range(0, configs.numLdqEntries):
                for j in range(0, configs.numStqEntries):
                    arch += Op(ctx, (addr_same, i, j), '\'1\'', 'when',
                               (ldq_addr, i), '=', (stq_addr, j), 'else', '\'0\'')

            # A load conflicts with a store when:
            # 1. The store entry is valid, and
            # 2. The store is older than the load, and
            # 3. The address conflicts(same or invalid store address).
            for i in range(0, configs.numLdqEntries):
                for j in range(0, configs.numStqEntries):
                    arch += Op(ctx,
                               (ld_st_conflict, i, j),
                               (stq_valid, j),         'and',
                               (store_is_older, i, j), 'and',
                               '(', (addr_same, i, j), 'or', 'not', (stq_addr_valid, j), ')'
                               )

            # A conflicting store entry can be bypassed to a load entry when:
            # 1. The load entry is valid, and
            # 2. The load entry is not issued yet, and
            # 3. The address of the load-store pair are both valid and values the same.
            for i in range(0, configs.numLdqEntries):
                for j in range(0, configs.numStqEntries):
                    arch += Op(ctx,
                               (can_bypass, i, j),
                               (ldq_valid, i),        'and',
                               'not', (ldq_issue, i), 'and',
                               (stq_data_valid, j),   'and',
                               (addr_same, i, j),     'and',
                               (addr_valid, i, j)
                               )

            # Load

            load_conflict = LogicArray(
                ctx, 'load_conflict', 'w', configs.numLdqEntries)
            load_req_valid = LogicArray(
                ctx, 'load_req_valid', 'w', configs.numLdqEntries)
            can_load = LogicArray(ctx, 'can_load', 'w', configs.numLdqEntries)

            # The load conflicts with any store
            for i in range(0, configs.numLdqEntries):
                arch += Reduce(ctx, load_conflict[i], ld_st_conflict[i], 'or')
            # The load is valid when the entry is valid and not yet issued, the load address should also be valid.
            # We do not need to check ldq_data_valid, since unissued load request cannot have valid data.
            for i in range(0, configs.numLdqEntries):
                arch += Op(ctx, load_req_valid[i], ldq_valid[i], 'and',
                           'not', ldq_issue[i], 'and', ldq_addr_valid[i])
            # Generate list for loads that does not face dependency issue
            for i in range(0, configs.numLdqEntries):
                arch += Op(ctx, can_load[i], 'not',
                           load_conflict[i], 'and', load_req_valid[i])

            can_load_list = []
            can_load_list.append(can_load)
            for w in range(0, configs.numLdMem):
                arch += CyclicPriorityMasking(ctx,
                                              load_idx_oh[w], can_load_list[w], ldq_head_oh)
                arch += Reduce(ctx, load_en[w], can_load_list[w], 'or')
                if (w+1 != configs.numLdMem):
                    load_idx_oh_LogicArray = LogicArray(
                        ctx, f'load_idx_oh_Array_{w+1}', 'w', configs.numLdqEntries)
                    arch += VecToArray(ctx,
                                       load_idx_oh_LogicArray, load_idx_oh[w])
                    can_load_list.append(LogicArray(
                        ctx, f'can_load_list_{w+1}', 'w', configs.numLdqEntries))
                    for i in range(0, configs.numLdqEntries):
                        arch += Op(ctx, can_load_list[w+1][i], 'not',
                                   load_idx_oh_LogicArray[i], 'and', can_load_list[w][i])
            # Store

            st_ld_conflict = LogicVec(
                ctx, 'st_ld_conflict', 'w', configs.numLdqEntries)
            store_conflict = Logic(ctx, 'store_conflict', 'w')
            store_valid = Logic(ctx, 'store_valid', 'w')
            store_data_valid = Logic(ctx, 'store_data_valid', 'w')
            store_addr_valid = Logic(ctx, 'store_addr_valid', 'w')

            # A store conflicts with a load when:
            # 1. The load entry is valid, and
            # 2. The load is older than the store, and
            # 3. The address conflicts(same or invalid store address).
            # Index order are reversed for store matrix.
            for i in range(0, configs.numLdqEntries):
                arch += Op(ctx,
                           (st_ld_conflict, i),
                           (ldq_valid, i), 'and',
                           'not', MuxIndex(
                               store_is_older[i], stq_issue), 'and',
                           '(', MuxIndex(
                               addr_same[i], stq_issue), 'or', 'not', (ldq_addr_valid, i), ')'
                           )
            # The store conflicts with any load
            arch += Reduce(ctx, store_conflict, st_ld_conflict, 'or')
            # The store is valid whe the entry is valid and the data is also valid,
            # the store address should also be valid
            arch += MuxLookUp(ctx, store_valid, stq_valid, stq_issue)
            arch += MuxLookUp(ctx, store_data_valid, stq_data_valid, stq_issue)
            arch += MuxLookUp(ctx, store_addr_valid, stq_addr_valid, stq_issue)
            arch += Op(ctx, store_en,
                       'not', store_conflict, 'and',
                       store_valid, 'and',
                       store_data_valid, 'and',
                       store_addr_valid
                       )
            arch += Op(ctx, store_idx, stq_issue)

            stq_last_oh = LogicVec(
                ctx, 'stq_last_oh', 'w', configs.numStqEntries)
            arch += BitsToOHSub1(ctx, stq_last_oh, stq_tail)
            for i in range(0, configs.numLdqEntries):
                bypass_en_vec = LogicVec(
                    ctx, f'bypass_en_vec_{i}', 'w', configs.numStqEntries)
                # Search for the youngest store that is older than the load and conflicts
                arch += CyclicPriorityMasking(
                    ctx, bypass_idx_oh[i], ld_st_conflict[i], stq_last_oh, True)
                # Check if the youngest conflict store can bypass with the load
                arch += Op(ctx, bypass_en_vec,
                           bypass_idx_oh[i], 'and', can_bypass[i])
                arch += Reduce(ctx, bypass_en[i], bypass_en_vec, 'or')

    if configs.pipe1:
        # Pipeline Stage 1
        load_idx_oh_p1 = LogicVecArray(
            ctx, 'load_idx_oh_p1', 'r', configs.numLdMem, configs.numLdqEntries)
        load_en_p1 = LogicArray(ctx, 'load_en_p1', 'r', configs.numLdMem)

        load_hs = LogicArray(ctx, 'load_hs', 'w', configs.numLdMem)
        load_p1_ready = LogicArray(ctx, 'load_p1_ready', 'w', configs.numLdMem)

        store_idx_p1 = LogicVec(ctx, 'store_idx_p1', 'r', configs.stqAddrW)
        store_en_p1 = Logic(ctx, 'store_en_p1', 'r')

        store_hs = Logic(ctx, 'store_hs', 'w')
        store_p1_ready = Logic(ctx, 'store_p1_ready', 'w')

        bypass_idx_oh_p1 = LogicVecArray(
            ctx, 'bypass_idx_oh_p1', 'r', configs.numLdqEntries, configs.numStqEntries)
        bypass_en_p1 = LogicArray(
            ctx, 'bypass_en_p1', 'r', configs.numLdqEntries)

        load_idx_oh_p1.regInit(enable=load_p1_ready)
        load_en_p1.regInit(init=[0]*configs.numLdMem, enable=load_p1_ready)

        store_idx_p1.regInit(enable=store_p1_ready)
        store_en_p1.regInit(init=0, enable=store_p1_ready)

        bypass_idx_oh_p1.regInit()
        bypass_en_p1.regInit(init=[0]*configs.numLdqEntries)

        for w in range(0, configs.numLdMem):
            arch += Op(ctx, load_hs[w], load_en_p1[w], 'and', rreq_ready_i[w])
            arch += Op(ctx, load_p1_ready[w],
                       load_hs[w], 'or', 'not', load_en_p1[w])

        for w in range(0, configs.numLdMem):
            arch += Op(ctx, load_idx_oh_p1[w], load_idx_oh[w])
            arch += Op(ctx, load_en_p1[w], load_en[w])

        arch += Op(ctx, store_hs, store_en_p1, 'and', wreq_ready_i[0])
        arch += Op(ctx, store_p1_ready, store_hs, 'or', 'not', store_en_p1)

        arch += Op(ctx, store_idx_p1, store_idx)
        arch += Op(ctx, store_en_p1, store_en)

        if configs.pipe0:
            for i in range(0, configs.numLdqEntries):
                arch += Op(ctx, bypass_idx_oh_p1[i], bypass_idx_oh_p0[i])
        else:
            for i in range(0, configs.numLdqEntries):
                arch += Op(ctx, bypass_idx_oh_p1[i], bypass_idx_oh[i])

        for i in range(0, configs.numLdqEntries):
            arch += Op(ctx, bypass_en_p1[i], bypass_en[i])

        ######    Read/Write    ######
        # Read Request
        for w in range(0, configs.numLdMem):
            arch += Op(ctx, rreq_valid_o[w], load_en_p1[w])
            arch += OHToBits(ctx, rreq_id_o[w], load_idx_oh_p1[w])
            arch += Mux1H(ctx, rreq_addr_o[w], ldq_addr, load_idx_oh_p1[w])

        for i in range(0, configs.numLdqEntries):
            ldq_issue_set_vec = LogicVec(
                ctx, f'ldq_issue_set_vec_{i}', 'w', configs.numLdMem)
            for w in range(0, configs.numLdMem):
                arch += Op(ctx, (ldq_issue_set_vec, w),
                           '(', (load_idx_oh, w, i), 'and',
                           (load_p1_ready, w), ')', 'or',
                           (bypass_en, i)
                           )
            arch += Reduce(ctx, ldq_issue_set[i], ldq_issue_set_vec, 'or')

        # Write Request
        arch += Op(ctx, wreq_valid_o[0], store_en_p1)
        arch += Op(ctx, wreq_id_o[0], 0)
        arch += MuxLookUp(ctx, wreq_addr_o[0], stq_addr, store_idx_p1)
        arch += MuxLookUp(ctx, wreq_data_o[0], stq_data, store_idx_p1)
        arch += Op(ctx, stq_issue_en, store_en, 'and', store_p1_ready)

        # Read Response and Bypass
        for i in range(0, configs.numLdqEntries):
            # check each read response channel for each load
            read_idx_oh = LogicArray(
                ctx, f'read_idx_oh_{i}', 'w', configs.numLdMem)
            read_valid = Logic(ctx, f'read_valid_{i}', 'w')
            read_data = LogicVec(ctx, f'read_data_{i}', 'w', configs.dataW)
            for w in range(0, configs.numLdMem):
                arch += Op(ctx, read_idx_oh[w], rresp_valid_i[w], 'when',
                           '(', rresp_id_i[w], '=', (i, configs.idW), ')', 'else', '\'0\'')
            arch += Mux1H(ctx, read_data, rresp_data_i, read_idx_oh)
            arch += Reduce(ctx, read_valid, read_idx_oh, 'or')
            # multiplex from store queue data
            bypass_data = LogicVec(ctx, f'bypass_data_{i}', 'w', configs.dataW)
            arch += Mux1H(ctx, bypass_data, stq_data, bypass_idx_oh_p1[i])
            # multiplex from read and bypass data
            arch += Op(ctx, ldq_data[i], read_data, 'or', bypass_data)
            arch += Op(ctx, ldq_data_wen[i], bypass_en_p1[i], 'or', read_valid)
        for w in range(0, configs.numLdMem):
            arch += Op(ctx, rresp_ready_o[w], '\'1\'')

        # Write Response
        if configs.stResp:
            for i in range(0, configs.numStqEntries):
                arch += Op(ctx, stq_exec_set[i],
                           wresp_valid_i[0], 'when',
                           '(', stq_resp, '=', (i, configs.stqAddrW), ')',
                           'else', '\'0\''
                           )
        else:
            for i in range(0, configs.numStqEntries):
                arch += Op(ctx, stq_reset[i],
                           wresp_valid_i[0], 'when',
                           '(', stq_resp, '=', (i, configs.stqAddrW), ')',
                           'else', '\'0\''
                           )
        arch += Op(ctx, stq_resp_en, wresp_valid_i[0])
        arch += Op(ctx, wresp_ready_o[0], '\'1\'')
    else:
        ######    Read/Write    ######
        # Read Request
        for w in range(0, configs.numLdMem):
            arch += Op(ctx, rreq_valid_o[w], load_en[w])
            arch += OHToBits(ctx, rreq_id_o[w], load_idx_oh[w])
            arch += Mux1H(ctx, rreq_addr_o[w], ldq_addr, load_idx_oh[w])

        for i in range(0, configs.numLdqEntries):
            ldq_issue_set_vec = LogicVec(
                ctx, f'ldq_issue_set_vec_{i}', 'w', configs.numLdMem)
            for w in range(0, configs.numLdMem):
                arch += Op(ctx, (ldq_issue_set_vec, w),
                           '(', (load_idx_oh, w, i), 'and',
                           (rreq_ready_i, w), 'and',
                           (load_en, w), ')', 'or',
                           (bypass_en, i)
                           )
            arch += Reduce(ctx, ldq_issue_set[i], ldq_issue_set_vec, 'or')

        # Write Request
        arch += Op(ctx, wreq_valid_o[0], store_en)
        arch += Op(ctx, wreq_id_o[0], 0)
        arch += MuxLookUp(ctx, wreq_addr_o[0], stq_addr, store_idx)
        arch += MuxLookUp(ctx, wreq_data_o[0], stq_data, store_idx)
        arch += Op(ctx, stq_issue_en, store_en, 'and', wreq_ready_i[0])

        # Read Response and Bypass
        for i in range(0, configs.numLdqEntries):
            # check each read response channel for each load
            read_idx_oh = LogicArray(
                ctx, f'read_idx_oh_{i}', 'w', configs.numLdMem)
            read_valid = Logic(ctx, f'read_valid_{i}', 'w')
            read_data = LogicVec(ctx, f'read_data_{i}', 'w', configs.dataW)
            for w in range(0, configs.numLdMem):
                arch += Op(ctx, read_idx_oh[w], rresp_valid_i[w], 'when',
                           '(', rresp_id_i[w], '=', (i, configs.idW), ')', 'else', '\'0\'')
            arch += Mux1H(ctx, read_data, rresp_data_i, read_idx_oh)
            arch += Reduce(ctx, read_valid, read_idx_oh, 'or')
            # multiplex from store queue data
            bypass_data = LogicVec(ctx, f'bypass_data_{i}', 'w', configs.dataW)
            if configs.pipe0:
                arch += Mux1H(ctx, bypass_data, stq_data, bypass_idx_oh_p0[i])
            else:
                arch += Mux1H(ctx, bypass_data, stq_data, bypass_idx_oh[i])
            # multiplex from read and bypass data
            arch += Op(ctx, ldq_data[i], read_data, 'or', bypass_data)
            arch += Op(ctx, ldq_data_wen[i], bypass_en[i], 'or', read_valid)
        for w in range(0, configs.numLdMem):
            arch += Op(ctx, rresp_ready_o[w], '\'1\'')

        # Write Response
        if configs.stResp:
            for i in range(0, configs.numStqEntries):
                arch += Op(ctx, stq_exec_set[i],
                           wresp_valid_i[0], 'when',
                           '(', stq_resp, '=', (i, configs.stqAddrW), ')',
                           'else', '\'0\''
                           )
        else:
            for i in range(0, configs.numStqEntries):
                arch += Op(ctx, stq_reset[i],
                           wresp_valid_i[0], 'when',
                           '(', stq_resp, '=', (i, configs.stqAddrW), ')',
                           'else', '\'0\''
                           )
        arch += Op(ctx, stq_resp_en, wresp_valid_i[0])
        arch += Op(ctx, wresp_ready_o[0], '\'1\'')

    ######   Write To File  ######
    ctx.portInitString += '\n\t);'
    ctx.regInitString += '\tend process;\n'

    # Write to the file
    with open(f'{path_rtl}/{name}.vhd', 'a') as file:
        # with open(name + '.vhd', 'w') as file:
        file.write(ctx.library)
        file.write(f'entity {name} is\n')
        file.write(ctx.portInitString)
        file.write('\nend entity;\n\n')
        file.write(f'architecture arch of {name} is\n')
        file.write(ctx.signalInitString)
        file.write('begin\n' + arch + '\n')
        file.write(ctx.regInitString + 'end architecture;\n')
