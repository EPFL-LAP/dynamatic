from vhdl_gen.context import VHDLContext
from vhdl_gen.signals import *
from vhdl_gen.operators import *
from vhdl_gen.configs import Configs

import vhdl_gen.generators.lsq_submodule_wrapper as lsq_submodule_wrapper


class LSQ:
    def __init__(
        self,
        name: str,
        suffix: str,
        configs: Configs
    ):
        """
        LSQ

        Models the top-level Load-Store Queue (LSQ) module.

        This class integrates all necessary sub-components to form a complete LSQ.
        It is responsible for generating the top-level VHDL entity that wires
        together the Group Allocator, various Port/Queue Dispatchers, and the core
        queue logic with dependency checking.

        Parameters:
            name    : Base name of the LSQ. "<name saved in configs>_core"
            suffix  : Suffix appended to the name to form the VHDL entity name.
                      Since LSQ is the top module, you do not need to add any suffix.
            configs : configuration generated from JSON


        Instance Variable:
            self.module_name = name + suffix : Entity and architecture identifier


        Example:
            lsq_core = LSQ("config_0_core", '', configs)

            # You can later generate VHDL entity and architecture by
            #     lsq_core.generate(...)

            # Instantiation of the LSQ module does not use this class.
            # It considers more conditions, and it is done in lsq-generator.py.

        """

        self.name = name
        self.module_name = name + suffix
        self.configs = configs

    def generate(self, lsq_submodules, path_rtl) -> None:
        """
        Generates the VHDL 'entity' and 'architecture' sections for an LSQ.

        This function appends the following to the file '<path_rtl>/<self.name>.vhd:
            1. 'entity <self.module_name>' declaration
            2. 'architecture arch of <self.module_name>' implementation

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
            lsq_submodules  : A collection of objects representing submodules whose VHDL entity
                              definitions are already generated. This parameter is used to
                              generate their port map instantiations.
            path_rtl        : Output directory for VHDL files.

        Output:
            Appends the 'entity' and 'architecture' definitions
            to the .vhd file at <path_rtl>/<self.name>.vhd.
            Entity and architecture use the identifier: <self.module_name>

        Example:
            lsq_core.generate(lsq_submodules, path_rtl)

        """

        ctx = VHDLContext()

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
            ctx, 'group_init_valid', 'i', self.configs.numGroups)
        group_init_ready_o = LogicArray(
            ctx, 'group_init_ready', 'o', self.configs.numGroups)

        # Memory access ports, i.e., the connection "kernel -> LSQ"
        # Load address channel (addr, valid, ready) from kernel, contains signals:
        ldp_addr_i = LogicVecArray(
            ctx, 'ldp_addr', 'i', self.configs.numLdPorts, self.configs.addrW)
        ldp_addr_valid_i = LogicArray(
            ctx, 'ldp_addr_valid', 'i', self.configs.numLdPorts)
        ldp_addr_ready_o = LogicArray(
            ctx, 'ldp_addr_ready', 'o', self.configs.numLdPorts)

        # Load data channel (data, valid, ready) to kernel
        ldp_data_o = LogicVecArray(
            ctx, 'ldp_data', 'o', self.configs.numLdPorts, self.configs.dataW)
        ldp_data_valid_o = LogicArray(
            ctx, 'ldp_data_valid', 'o', self.configs.numLdPorts)
        ldp_data_ready_i = LogicArray(
            ctx, 'ldp_data_ready', 'i', self.configs.numLdPorts)

        # Store address channel (addr, valid, ready) from kernel
        stp_addr_i = LogicVecArray(
            ctx, 'stp_addr', 'i', self.configs.numStPorts, self.configs.addrW)
        stp_addr_valid_i = LogicArray(
            ctx, 'stp_addr_valid', 'i', self.configs.numStPorts)
        stp_addr_ready_o = LogicArray(
            ctx, 'stp_addr_ready', 'o', self.configs.numStPorts)

        # Store data channel (data, valid, ready) from kernel
        stp_data_i = LogicVecArray(
            ctx, 'stp_data', 'i', self.configs.numStPorts, self.configs.dataW)
        stp_data_valid_i = LogicArray(
            ctx, 'stp_data_valid', 'i', self.configs.numStPorts)
        stp_data_ready_o = LogicArray(
            ctx, 'stp_data_ready', 'o', self.configs.numStPorts)

        if self.configs.stResp:
            stp_exec_valid_o = LogicArray(
                ctx, 'stp_exec_valid', 'o', self.configs.numStPorts)
            stp_exec_ready_i = LogicArray(
                ctx, 'stp_exec_ready', 'i', self.configs.numStPorts)

        # queue empty signal
        empty_o = Logic(ctx, 'empty', 'o')

        # Memory interface: i.e., the connection LSQ -> AXI
        # We assume that the memory interface has
        # 1. A read request channel (rreq) and a read response channel (rresp).
        # 2. A write request channel (wreq) and a write response channel (wresp).
        rreq_valid_o = LogicArray(
            ctx, 'rreq_valid', 'o', self.configs.numLdMem)
        rreq_ready_i = LogicArray(
            ctx, 'rreq_ready', 'i', self.configs.numLdMem)
        rreq_id_o = LogicVecArray(
            ctx, 'rreq_id', 'o', self.configs.numLdMem, self.configs.idW)
        rreq_addr_o = LogicVecArray(
            ctx, 'rreq_addr', 'o', self.configs.numLdMem, self.configs.addrW)

        rresp_valid_i = LogicArray(
            ctx, 'rresp_valid', 'i', self.configs.numLdMem)
        rresp_ready_o = LogicArray(
            ctx, 'rresp_ready', 'o', self.configs.numLdMem)
        rresp_id_i = LogicVecArray(
            ctx, 'rresp_id', 'i', self.configs.numLdMem, self.configs.idW)
        rresp_data_i = LogicVecArray(
            ctx, 'rresp_data', 'i', self.configs.numLdMem, self.configs.dataW)

        wreq_valid_o = LogicArray(
            ctx, 'wreq_valid', 'o', self.configs.numStMem)
        wreq_ready_i = LogicArray(
            ctx, 'wreq_ready', 'i', self.configs.numStMem)
        wreq_id_o = LogicVecArray(
            ctx, 'wreq_id', 'o', self.configs.numStMem, self.configs.idW)
        wreq_addr_o = LogicVecArray(
            ctx, 'wreq_addr', 'o', self.configs.numStMem, self.configs.addrW)
        wreq_data_o = LogicVecArray(
            ctx, 'wreq_data', 'o', self.configs.numStMem, self.configs.dataW)

        wresp_valid_i = LogicArray(
            ctx, 'wresp_valid', 'i', self.configs.numStMem)
        wresp_ready_o = LogicArray(
            ctx, 'wresp_ready', 'o', self.configs.numStMem)
        wresp_id_i = LogicVecArray(
            ctx, 'wresp_id', 'i', self.configs.numStMem, self.configs.idW)

        #! If this is the lsq master, then we need the following logic
        #! Define new interfaces needed by dynamatic
        if (self.configs.master):
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

            #! The memory completion signal cannot be set to 1 when any group is allocating:
            no_curr_ga = "(not (" + " or ".join(group_init_valid_i.getNameRead(i)
                                                for i in range(group_init_valid_i.length)) + "))"

            #! Define the needed logic
            arch += "\t-- This signal indicates that all mem. ops are completed and func. can return.\n"
            arch += "\t-- LSQ can return iff all the following conditions are true:\n"
            arch += "\t-- 1. No more upcoming BBs containing memory accesses.\n"
            arch += "\t-- 2. Both store and load queues are empty.\n"
            arch += "\t-- 3. No GA in the same cycle.\n"
            arch += f"\tTEMP_GEN_MEM <= {ctrlEnd_valid.getNameRead()} and stq_empty and ldq_empty and {no_curr_ga};\n"

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
        ldq_alloc = LogicArray(ctx, 'ldq_alloc', 'r',
                               self.configs.numLdqEntries)
        ldq_issue = LogicArray(ctx, 'ldq_issue', 'r',
                               self.configs.numLdqEntries)
        if (self.configs.ldpAddrW > 0):
            ldq_port_idx = LogicVecArray(
                ctx, 'ldq_port_idx', 'r', self.configs.numLdqEntries, self.configs.ldpAddrW)
        else:
            ldq_port_idx = None
        ldq_addr_valid = LogicArray(
            ctx, 'ldq_addr_valid', 'r', self.configs.numLdqEntries)
        ldq_addr = LogicVecArray(ctx, 'ldq_addr', 'r',
                                 self.configs.numLdqEntries, self.configs.addrW)
        ldq_data_valid = LogicArray(
            ctx, 'ldq_data_valid', 'r', self.configs.numLdqEntries)
        ldq_data = LogicVecArray(ctx, 'ldq_data', 'r',
                                 self.configs.numLdqEntries, self.configs.dataW)

        # Store Queue Entries
        stq_alloc = LogicArray(ctx, 'stq_alloc', 'r',
                               self.configs.numStqEntries)
        if self.configs.stResp:
            stq_exec = LogicArray(ctx, 'stq_exec', 'r',
                                  self.configs.numStqEntries)
        if (self.configs.stpAddrW > 0):
            stq_port_idx = LogicVecArray(
                ctx, 'stq_port_idx', 'r', self.configs.numStqEntries, self.configs.stpAddrW)
        else:
            stq_port_idx = None
        stq_addr_valid = LogicArray(
            ctx, 'stq_addr_valid', 'r', self.configs.numStqEntries)
        stq_addr = LogicVecArray(ctx, 'stq_addr', 'r',
                                 self.configs.numStqEntries, self.configs.addrW)
        stq_data_valid = LogicArray(
            ctx, 'stq_data_valid', 'r', self.configs.numStqEntries)
        stq_data = LogicVecArray(ctx, 'stq_data', 'r',
                                 self.configs.numStqEntries, self.configs.dataW)

        # Order for load-store
        store_is_older = LogicVecArray(
            ctx, 'store_is_older', 'r', self.configs.numLdqEntries, self.configs.numStqEntries)

        # Pointers
        ldq_tail = LogicVec(ctx, 'ldq_tail', 'r', self.configs.ldqAddrW)
        ldq_head = LogicVec(ctx, 'ldq_head', 'r', self.configs.ldqAddrW)

        stq_tail = LogicVec(ctx, 'stq_tail', 'r', self.configs.stqAddrW)
        stq_head = LogicVec(ctx, 'stq_head', 'r', self.configs.stqAddrW)
        stq_issue = LogicVec(ctx, 'stq_issue', 'r', self.configs.stqAddrW)
        stq_resp = LogicVec(ctx, 'stq_resp', 'r', self.configs.stqAddrW)

        # Entry related signals
        # From port dispatchers
        ldq_wen = LogicArray(ctx, 'ldq_wen', 'w', self.configs.numLdqEntries)
        ldq_addr_wen = LogicArray(
            ctx, 'ldq_addr_wen', 'w', self.configs.numLdqEntries)
        ldq_reset = LogicArray(ctx, 'ldq_reset', 'w',
                               self.configs.numLdqEntries)
        stq_wen = LogicArray(ctx, 'stq_wen', 'w', self.configs.numStqEntries)
        stq_addr_wen = LogicArray(
            ctx, 'stq_addr_wen', 'w', self.configs.numStqEntries)
        stq_data_wen = LogicArray(
            ctx, 'stq_data_wen', 'w', self.configs.numStqEntries)
        stq_reset = LogicArray(ctx, 'stq_reset', 'w',
                               self.configs.numStqEntries)
        # From Read/Write Block
        ldq_data_wen = LogicArray(
            ctx, 'ldq_data_wen', 'w', self.configs.numLdqEntries)
        ldq_issue_set = LogicArray(
            ctx, 'ldq_issue_set', 'w', self.configs.numLdqEntries)
        if self.configs.stResp:
            stq_exec_set = LogicArray(
                ctx, 'stq_exec_set', 'w', self.configs.numStqEntries)
        # Form Group Allocator
        ga_ls_order = LogicVecArray(
            ctx, 'ga_ls_order', 'w', self.configs.numLdqEntries, self.configs.numStqEntries)

        # Pointer related signals
        # For updating pointers
        num_loads = LogicVec(ctx, 'num_loads', 'w', self.configs.ldqAddrW)
        num_stores = LogicVec(ctx, 'num_stores', 'w', self.configs.stqAddrW)
        stq_issue_en = Logic(ctx, 'stq_issue_en', 'w')
        stq_resp_en = Logic(ctx, 'stq_resp_en', 'w')
        # Generated by pointers
        ldq_empty = Logic(ctx, 'ldq_empty', 'w')
        stq_empty = Logic(ctx, 'stq_empty', 'w')
        ldq_head_oh = LogicVec(ctx, 'ldq_head_oh', 'w',
                               self.configs.numLdqEntries)
        stq_head_oh = LogicVec(ctx, 'stq_head_oh', 'w',
                               self.configs.numStqEntries)

        arch += BitsToOH(ctx, ldq_head_oh, ldq_head)
        arch += BitsToOH(ctx, stq_head_oh, stq_head)

        # Pipelining Strategy:
        # The signals are always passed through the pipeline stages (*_pcomp,
        # *_p0, *_p1). If the pipeline stage is enabled, the signal will be
        # registered (the signal type is 'r' for register). Otherwise, the
        # signal type is 'w' for wire, and the pipeline stage is effectively
        # bypassed. If the signals are registers, we need to conditionally call
        # regInit().
        pipe_comp_type = 'r' if self.configs.pipeComp else 'w'
        pipe0_type = 'r' if self.configs.pipe0 else 'w'
        pipe1_type = 'r' if self.configs.pipe1 else 'w'

        # update queue entries
        # load queue
        ldq_wen_pcomp = LogicArray(
            ctx, 'ldq_wen_pcomp', pipe_comp_type, self.configs.numLdqEntries)
        ldq_wen_p0 = LogicArray(
            ctx, 'ldq_wen_p0', pipe0_type, self.configs.numLdqEntries)
        ldq_alloc_next = LogicArray(
            ctx, 'ldq_alloc_next', 'w', self.configs.numLdqEntries)
        if self.configs.pipeComp:
            ldq_wen_pcomp.regInit()
        if self.configs.pipe0:
            ldq_wen_p0.regInit()

        for i in range(0, self.configs.numLdqEntries):
            arch += Op(ctx, ldq_alloc_next[i],
                       'not', ldq_reset[i], 'and', ldq_alloc[i]
                       )
            arch += Op(ctx, ldq_alloc[i],
                       ldq_wen[i], 'or', ldq_alloc_next[i]
                       )
            arch += Op(ctx, ldq_wen_pcomp[i], ldq_wen[i])
            arch += Op(ctx, ldq_wen_p0[i], ldq_wen_pcomp[i])
            arch += Op(ctx, ldq_issue[i],
                       'not', ldq_wen_p0[i], 'and',
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
        stq_alloc_next = LogicArray(
            ctx, 'stq_alloc_next', 'w', self.configs.numStqEntries)
        for i in range(0, self.configs.numStqEntries):
            arch += Op(ctx, stq_alloc_next[i],
                       'not', stq_reset[i], 'and', stq_alloc[i]
                       )
            arch += Op(ctx, stq_alloc[i],
                       stq_wen[i], 'or', stq_alloc_next[i]
                       )
            if self.configs.stResp:
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
        # store_is_older(i,j) = (not stq_reset(j) and (stq_alloc(j) or ga_ls_order(i, j)))
        #                  when ldq_wen(i)
        #                  else not stq_reset(j) and store_is_older(i, j)
        for i in range(0, self.configs.numLdqEntries):
            for j in range(0, self.configs.numStqEntries):
                arch += Op(ctx, (store_is_older, i, j),
                           '(', 'not', (stq_reset, j), 'and', '(', (stq_alloc,
                                                                    j), 'or', (ga_ls_order, i, j), ')', ')',
                           'when', (ldq_wen, i), 'else',
                           'not', (stq_reset, j), 'and', (store_is_older, i, j)
                           )

        # pointers update
        ldq_not_empty = Logic(ctx, 'ldq_not_empty', 'w')
        stq_not_empty = Logic(ctx, 'stq_not_empty', 'w')
        arch += Reduce(ctx, ldq_not_empty, ldq_alloc, 'or')
        arch += Op(ctx, ldq_empty, 'not', ldq_not_empty)
        arch += MuxLookUp(ctx, stq_not_empty, stq_alloc, stq_head)
        arch += Op(ctx, stq_empty, 'not', stq_not_empty)
        arch += Op(ctx, empty_o, ldq_empty, 'and', stq_empty)

        arch += WrapAdd(ctx, ldq_tail, ldq_tail, num_loads,
                        self.configs.numLdqEntries)
        arch += WrapAdd(ctx, stq_tail, stq_tail, num_stores,
                        self.configs.numStqEntries)
        arch += WrapAddConst(ctx, stq_issue, stq_issue, 1,
                             self.configs.numStqEntries)
        arch += WrapAddConst(ctx, stq_resp, stq_resp, 1,
                             self.configs.numStqEntries)

        ldq_tail_oh = LogicVec(ctx, 'ldq_tail_oh', 'w',
                               self.configs.numLdqEntries)
        arch += BitsToOH(ctx, ldq_tail_oh, ldq_tail)
        ldq_head_next_oh = LogicVec(
            ctx, 'ldq_head_next_oh', 'w', self.configs.numLdqEntries)
        ldq_head_next = LogicVec(ctx, 'ldq_head_next',
                                 'w', self.configs.ldqAddrW)
        ldq_head_sel = Logic(ctx, 'ldq_head_sel', 'w')
        if self.configs.headLag:
            # Update the head pointer according to the valid signal of last cycle
            arch += CyclicPriorityMasking(ctx,
                                          ldq_head_next_oh, ldq_alloc, ldq_tail_oh)
            arch += Reduce(ctx, ldq_head_sel, ldq_alloc, 'or')
        else:
            arch += CyclicPriorityMasking(ctx, ldq_head_next_oh,
                                          ldq_alloc_next, ldq_tail_oh)
            arch += Reduce(ctx, ldq_head_sel, ldq_alloc_next, 'or')
        arch += OHToBits(ctx, ldq_head_next, ldq_head_next_oh)
        arch += Op(ctx, ldq_head, ldq_head_next, 'when',
                   ldq_head_sel, 'else', ldq_tail)

        stq_tail_oh = LogicVec(ctx, 'stq_tail_oh', 'w',
                               self.configs.numStqEntries)
        arch += BitsToOH(ctx, stq_tail_oh, stq_tail)
        stq_head_next_oh = LogicVec(
            ctx, 'stq_head_next_oh', 'w', self.configs.numStqEntries)
        stq_head_next = LogicVec(ctx, 'stq_head_next',
                                 'w', self.configs.stqAddrW)
        stq_head_sel = Logic(ctx, 'stq_head_sel', 'w')
        if self.configs.stResp:
            if self.configs.headLag:
                # Update the head pointer according to the valid signal of last cycle
                arch += CyclicPriorityMasking(ctx,
                                              stq_head_next_oh, stq_alloc, stq_tail_oh)
                arch += Reduce(ctx, stq_head_sel, stq_alloc, 'or')
            else:
                arch += CyclicPriorityMasking(ctx, stq_head_next_oh,
                                              stq_alloc_next, stq_tail_oh)
                arch += Reduce(ctx, stq_head_sel, stq_alloc_next, 'or')
            arch += OHToBits(ctx, stq_head_next, stq_head_next_oh)
            arch += Op(ctx, stq_head, stq_head_next, 'when',
                       stq_head_sel, 'else', stq_tail)
        else:
            arch += WrapAddConst(ctx, stq_head_next, stq_head,
                                 1, self.configs.numStqEntries)
            arch += Op(ctx, stq_head_sel, wresp_valid_i[0])
            arch += Op(ctx, stq_head, stq_head_next, 'when',
                       stq_head_sel, 'else', stq_head)

        # Load Queue Entries
        ldq_alloc.regInit(init=[0]*self.configs.numLdqEntries)
        ldq_issue.regInit(init=[0]*self.configs.numLdqEntries)
        if (self.configs.ldpAddrW > 0):
            ldq_port_idx.regInit(ldq_wen)
        ldq_addr_valid.regInit()
        ldq_addr.regInit(ldq_addr_wen)
        ldq_data_valid.regInit()
        ldq_data.regInit(ldq_data_wen)

        # Store Queue Entries
        stq_alloc.regInit(init=[0]*self.configs.numStqEntries)
        if self.configs.stResp:
            stq_exec.regInit()
        if (self.configs.stpAddrW > 0):
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
        arch += lsq_submodules.group_allocator.instantiate(
            ctx,
            group_init_valid_i, group_init_ready_o,
            ldq_tail, ldq_head, ldq_empty,
            stq_tail, stq_head, stq_empty,
            ldq_wen, num_loads, ldq_port_idx,
            stq_wen, num_stores, stq_port_idx,
            ga_ls_order
        )

        # When the condition "lsq_submodules.ptq_dispatcher_lda != None" is not true:
        # The dispatcher module will be set to None when there are zero load ports.
        # In this case, do not instantiate dispatching logic when there are zero load ports.
        # - WARNING: This logic needs more testing
        # - TODO: Also remove the load queue when there are zero load ports.
        if lsq_submodules.ptq_dispatcher_lda != None:
            # Load Address Port Dispatcher
            arch += lsq_submodules.ptq_dispatcher_lda.instantiate(
                ctx,
                ldp_addr_i, ldp_addr_valid_i, ldp_addr_ready_o,
                ldq_alloc, ldq_addr_valid, ldq_port_idx, ldq_addr, ldq_addr_wen, ldq_head_oh
            )

        # When the condition "lsq_submodules.qtp_dispatcher_ldd != None" is not true:
        # The dispatcher module will be set to None when there are zero load ports.
        # In this case, do not instantiate dispatching logic when there are zero load ports.
        # - WARNING: This logic needs more testing
        # - TODO: Also remove the load queue when there are zero load ports.
        if lsq_submodules.qtp_dispatcher_ldd != None:
            # Load Data Port Dispatcher
            arch += lsq_submodules.qtp_dispatcher_ldd.instantiate(
                ctx,
                ldp_data_o, ldp_data_valid_o, ldp_data_ready_i,
                ldq_alloc, ldq_data_valid, ldq_port_idx, ldq_data, ldq_reset, ldq_head_oh
            )

        # Store Address Port Dispatcher
        arch += lsq_submodules.ptq_dispatcher_sta.instantiate(
            ctx,
            stp_addr_i, stp_addr_valid_i, stp_addr_ready_o,
            stq_alloc, stq_addr_valid, stq_port_idx, stq_addr, stq_addr_wen, stq_head_oh
        )

        # Store Data Port Dispatcher
        arch += lsq_submodules.ptq_dispatcher_std.instantiate(
            ctx,
            stp_data_i, stp_data_valid_i, stp_data_ready_o,
            stq_alloc, stq_data_valid, stq_port_idx, stq_data, stq_data_wen, stq_head_oh
        )

        # Store Backward Port Dispatcher
        if self.configs.stResp:
            arch += lsq_submodules.qtp_dispatcher_stb.instantiate(
                ctx,
                None, stp_exec_valid_o, stp_exec_ready_i,
                stq_alloc, stq_exec, stq_port_idx, None, stq_reset, stq_head_oh
            )

        ###### Dependency Check ######
        load_idx_oh = LogicVecArray(
            ctx, 'load_idx_oh', 'w', self.configs.numLdMem, self.configs.numLdqEntries)
        load_en = LogicArray(ctx, 'load_en', 'w', self.configs.numLdMem)

        # Multiple store channels not yet implemented
        assert (self.configs.numStMem == 1)
        # current store request index
        store_idx = LogicVec(ctx, 'store_idx', 'w', self.configs.stqAddrW)
        # whether the current store request is valid, including address and data
        store_req_valid = Logic(ctx, 'store_req_valid', 'w')
        # whether the current store request has conflicts with any previous loads
        store_conflict = Logic(ctx, 'store_conflict', 'w')
        if self.configs.fallbackIssue:
            # whether the to-be-issued store entry is older than each of the load entries
            store_is_older_arr = LogicArray(ctx, 'store_is_older_arr', 'w', self.configs.numLdqEntries)
        # store request enable (after fallback logic)
        store_en = Logic(ctx, 'store_en', 'w')

        # Fallback load/store signals
        if self.configs.fallbackIssue:
            fallback_load_idx_oh = LogicVec(ctx, 'fallback_load_idx_oh', 'w', self.configs.numLdqEntries)
            fallback_load_en = Logic(ctx, 'fallback_load_en', 'w')
            fallback_store_en_if_valid = Logic(ctx, 'fallback_store_en_if_valid', 'w')

        # Matrix Generation
        ld_st_conflict = LogicVecArray(
            ctx, 'ld_st_conflict', 'w', self.configs.numLdqEntries, self.configs.numStqEntries)
        can_bypass = LogicVecArray(
            ctx, 'can_bypass', 'w', self.configs.numLdqEntries, self.configs.numStqEntries)
        can_bypass_p0 = LogicVecArray(
            ctx, 'can_bypass_p0', pipe0_type, self.configs.numLdqEntries, self.configs.numStqEntries)
        if self.configs.pipe0:
            can_bypass_p0.regInit(init=[0]*self.configs.numLdqEntries)

        ldq_alloc_pcomp = LogicArray(
            ctx, 'ldq_alloc_pcomp', pipe_comp_type, self.configs.numLdqEntries)
        ldq_addr_valid_pcomp = LogicArray(
            ctx, 'ldq_addr_valid_pcomp', pipe_comp_type, self.configs.numLdqEntries)
        stq_alloc_pcomp = LogicArray(
            ctx, 'stq_alloc_pcomp', pipe_comp_type, self.configs.numStqEntries)
        stq_addr_valid_pcomp = LogicArray(
            ctx, 'stq_addr_valid_pcomp', pipe_comp_type, self.configs.numStqEntries)
        stq_data_valid_pcomp = LogicArray(
            ctx, 'stq_data_valid_pcomp', pipe_comp_type, self.configs.numStqEntries)
        # addr_valid_pcomp is always a wire: combines other registers signals
        addr_valid_pcomp = LogicVecArray(
            ctx, 'addr_valid_pcomp', 'w', self.configs.numLdqEntries, self.configs.numStqEntries)
        addr_same_pcomp = LogicVecArray(
            ctx, 'addr_same_pcomp', pipe_comp_type, self.configs.numLdqEntries, self.configs.numStqEntries)
        store_is_older_pcomp = LogicVecArray(
            ctx, 'store_is_older_pcomp', pipe_comp_type, self.configs.numLdqEntries, self.configs.numStqEntries)

        # combinational signal indicating whether a load has already completed (assuming it is allocated), meaning the
        # data (= read response) from memory has been received
        load_completed = LogicArray(ctx, 'load_completed', 'w', self.configs.numLdqEntries)
        # combinational signal indicating whether a store has already completed (assuming it is allocated), meaning the
        # write response from memory has been received
        store_completed = LogicArray(ctx, 'store_completed', 'w', self.configs.numStqEntries)

        if self.configs.pipeComp:
            ldq_alloc_pcomp.regInit(init=[0]*self.configs.numLdqEntries)
            ldq_addr_valid_pcomp.regInit()
            stq_alloc_pcomp.regInit(init=[0]*self.configs.numStqEntries)
            stq_addr_valid_pcomp.regInit()
            stq_data_valid_pcomp.regInit()
            addr_same_pcomp.regInit()
            store_is_older_pcomp.regInit()

        for i in range(0, self.configs.numLdqEntries):
            arch += Op(ctx, (ldq_alloc_pcomp, i), (ldq_alloc, i))
            arch += Op(ctx, (ldq_addr_valid_pcomp, i),
                       (ldq_addr_valid, i))
        for j in range(0, self.configs.numStqEntries):
            arch += Op(ctx, (stq_alloc_pcomp, j), (stq_alloc, j))
            arch += Op(ctx, (stq_addr_valid_pcomp, j),
                       (stq_addr_valid, j))
            arch += Op(ctx, (stq_data_valid_pcomp, j),
                       (stq_data_valid, j))
        for i in range(0, self.configs.numLdqEntries):
            for j in range(0, self.configs.numStqEntries):
                arch += Op(ctx, (store_is_older_pcomp, i, j),
                           (store_is_older, i, j))
        for i in range(0, self.configs.numLdqEntries):
            for j in range(0, self.configs.numStqEntries):
                arch += Op(ctx, (addr_valid_pcomp, i, j),
                           (ldq_addr_valid_pcomp, i), 'and', (stq_addr_valid_pcomp, j))
        for i in range(0, self.configs.numLdqEntries):
            for j in range(0, self.configs.numStqEntries):
                arch += Op(ctx, (addr_same_pcomp, i, j), '\'1\'', 'when',
                           (ldq_addr, i), '=', (stq_addr, j), 'else', '\'0\'')

        for i in range(self.configs.numLdqEntries):
            # No need to use pipelined ldq_data_valid here: As soon as the load entry has valid data (in the queue
            # itself, not the pipeline), the load is considered completed.
            arch += Op(ctx, load_completed[i], ldq_data_valid[i])
        for i in range(self.configs.numStqEntries):
            if self.configs.stResp:
                # No need to use pipelined stq_exec here: As soon as the store response has been received from memory,
                # the store is considered completed.
                arch += Op(ctx, store_completed[i], stq_exec[i])
            else:
                # If the store queue entry is not valid (anymore), the store has completed.
                arch += Op(ctx, store_completed[i], 'not', stq_alloc[i])

        # A load conflicts with a store when:
        # 1. The store entry is valid, and
        # 2. The store entry hasn't completed (received write response from memory), and
        # 3. The store is older than the load, and
        # 4. The address conflicts(same or invalid store address).
        # NOTE: Because we only consider non-completed stores to conflict with a load, bypass will
        # not forward from any stores which are already completed (but still allocated). However,
        # such loads only exist if store responses or pipe0 are enabled, which is not the case by
        # default.
        for i in range(0, self.configs.numLdqEntries):
            for j in range(0, self.configs.numStqEntries):
                arch += Op(ctx,
                           (ld_st_conflict, i, j),
                           (stq_alloc_pcomp, j), 'and',
                           'not', (store_completed, j), 'and',
                           (store_is_older_pcomp, i, j), 'and',
                           '(', (addr_same_pcomp, i,
                                 j), 'or', 'not', (stq_addr_valid_pcomp, j), ')'
                           )

        # A conflicting store entry can be bypassed to a load entry when:
        # 1. The load entry is valid, and
        # 2. The load entry is not issued yet, and
        # 3. The address of the load-store pair are both valid and values the same.
        for i in range(0, self.configs.numLdqEntries):
            for j in range(0, self.configs.numStqEntries):
                arch += Op(ctx,
                           (can_bypass_p0, i, j),
                           (ldq_alloc_pcomp, i),        'and',
                           (stq_data_valid_pcomp, j),   'and',
                           (addr_same_pcomp, i, j),     'and',
                           (addr_valid_pcomp, i, j)
                           )
        for i in range(0, self.configs.numLdqEntries):
            for j in range(0, self.configs.numStqEntries):
                arch += Op(ctx,
                           (can_bypass, i, j),
                           'not', (ldq_issue, i), 'and',
                           (can_bypass_p0, i, j)
                           )

        # Load

        ldq_alloc_p0 = LogicArray(
            ctx, 'ldq_alloc_p0', pipe0_type, self.configs.numLdqEntries)
        ldq_addr_valid_p0 = LogicArray(
            ctx, 'ldq_addr_valid_p0', pipe0_type, self.configs.numLdqEntries)
        load_conflict = LogicArray(
            ctx, 'load_conflict', 'w', self.configs.numLdqEntries)
        load_req_valid = LogicArray(
            ctx, 'load_req_valid', 'w', self.configs.numLdqEntries)
        can_load = LogicArray(
            ctx, 'can_load', 'w', self.configs.numLdqEntries)
        can_load_p0 = LogicArray(
            ctx, 'can_load_p0', pipe0_type, self.configs.numLdqEntries)
        if self.configs.pipe0:
            ldq_alloc_p0.regInit(init=[0]*self.configs.numLdqEntries)
            ldq_addr_valid_p0.regInit(init=[0]*self.configs.numLdqEntries)
            can_load_p0.regInit(init=[0]*self.configs.numLdqEntries)

        # Pipeline
        for i in range(0, self.configs.numLdqEntries):
            arch += Op(ctx, ldq_alloc_p0[i], ldq_alloc_pcomp[i])
        for i in range(0, self.configs.numLdqEntries):
            arch += Op(ctx, ldq_addr_valid_p0[i], ldq_addr_valid_pcomp[i])
        # The load conflicts with any store
        for i in range(0, self.configs.numLdqEntries):
            arch += Reduce(ctx,
                           load_conflict[i], ld_st_conflict[i], 'or')
        # The load is valid when the entry is valid and not yet issued, the load address should also be valid.
        # We do not need to check ldq_data_valid, since unissued load request cannot have valid data.
        for i in range(0, self.configs.numLdqEntries):
            arch += Op(ctx, load_req_valid[i], ldq_alloc_pcomp[i],
                       'and', ldq_addr_valid_pcomp[i])
        # Generate list for loads that does not face dependency issue
        for i in range(0, self.configs.numLdqEntries):
            arch += Op(ctx, can_load_p0[i], 'not',
                       load_conflict[i], 'and', load_req_valid[i])
        for i in range(0, self.configs.numLdqEntries):
            arch += Op(ctx, can_load[i], 'not',
                       ldq_issue[i], 'and', can_load_p0[i])

        ldq_head_oh_p0 = LogicVec(
            ctx, 'ldq_head_oh_p0', pipe0_type, self.configs.numLdqEntries)
        if self.configs.pipe0:
            ldq_head_oh_p0.regInit(init=0)
        arch += Op(ctx, ldq_head_oh_p0, ldq_head_oh)

        can_load_list = []
        can_load_list.append(can_load)

        # temporary (pre-fallback) signals
        load_idx_tmp_oh = LogicVecArray(ctx, 'load_idx_tmp_oh', 'w', self.configs.numLdMem, self.configs.numLdqEntries)
        load_en_tmp = LogicArray(ctx, 'load_en_tmp', 'w', self.configs.numLdMem)

        for w in range(self.configs.numLdMem):
            arch += CyclicPriorityMasking(
                ctx, load_idx_tmp_oh[w], can_load_list[w], ldq_head_oh_p0)
            arch += Reduce(ctx, load_en_tmp[w], can_load_list[w], 'or')
            if (w+1 != self.configs.numLdMem):
                load_idx_oh_LogicArray = LogicArray(
                    ctx, f'load_idx_oh_Array_{w+1}', 'w', self.configs.numLdqEntries)
                arch += VecToArray(ctx,
                                   load_idx_oh_LogicArray, load_idx_oh[w])
                can_load_list.append(LogicArray(
                    ctx, f'can_load_list_{w+1}', 'w', self.configs.numLdqEntries))
                for i in range(0, self.configs.numLdqEntries):
                    arch += Op(ctx, can_load_list[w+1][i], 'not',
                               load_idx_oh_LogicArray[i], 'and', can_load_list[w][i])

        for w in range(self.configs.numLdMem):
            last_load_port = (w == self.configs.numLdMem - 1)
            if self.configs.fallbackIssue and last_load_port:
                # last load port: use fallback load (if any) as the first priority, then service other loads (from _tmp)
                arch += Op(ctx, load_idx_oh[w], fallback_load_idx_oh, 'when', fallback_load_en, 'else', load_idx_tmp_oh[w])
                arch += Op(ctx, load_en[w], fallback_load_en, 'or', load_en_tmp[w])
            else:
                # non-last load port: use _tmp signals directly
                arch += Op(ctx, load_idx_oh[w], load_idx_tmp_oh[w])
                arch += Op(ctx, load_en[w], load_en_tmp[w])

        if self.configs.fallbackIssue:
            # Fallback Load / Store

            # The fallback load candidate is the oldest allocated and un-issued load.
            ldq_alloc_no_issue = LogicArray(ctx, 'ldq_alloc_no_issue', 'w', self.configs.numLdqEntries)
            fallback_load_candidate_oh = LogicArray(ctx, 'fallback_load_candidate_oh', 'w', self.configs.numLdqEntries)
            for i in range(0, self.configs.numLdqEntries):
                arch += Op(ctx, ldq_alloc_no_issue[i], ldq_alloc_p0[i], 'and', 'not', ldq_issue[i])
            arch += CyclicPriorityMasking(ctx, fallback_load_candidate_oh, ldq_alloc_no_issue, ldq_head_oh_p0)

            # The fallback store canddiate is the oldest allocated and un-issued store. This is simply
            # the store at the store queue issue pointer (if allocated). We do not explicitly track the
            # fallback store candidate, but rather just keep the relevant row/column from the order
            # matrix.

            # If the oldest load is older than the oldest store, this contains a single bit set (at the oldest load entry).
            # Otherwise (oldest store is oldest overall), this is all zeros.
            fallback_load_is_oldest_oh = LogicArray(ctx, 'fallback_load_is_oldest_oh', 'w', self.configs.numLdqEntries)
            for i in range(self.configs.numLdqEntries):
                arch += Op(ctx, fallback_load_is_oldest_oh[i], fallback_load_candidate_oh[i], 'and', 'not', store_is_older_arr[i])

            # Whether the fallback load is the oldest.
            fallback_load_is_oldest = Logic(ctx, 'fallback_load_is_oldest', 'w')
            arch += Reduce(ctx, fallback_load_is_oldest, fallback_load_is_oldest_oh, 'or')

            # NOTE: For both the outstanding loads and stores, we only need to consider loads/store which were issued
            # previously. If a load (store) is issued through the regular path in the same cycle as the fallback store
            # (load), it cannot conflict with the fallback store (load). This is because:
            # 1. The regular load (store) must be younger than the fallback store (load) by construction.
            # 2. The regular load (store) has been dependency-checked against the fallback store (load) before being
            #    issued.
            # 3. Thus, the fallback store (load) and the regular load (store) must have different addresses.

            store_outstanding = Logic(ctx, 'store_outstanding', 'w')
            arch += Op(ctx, store_outstanding, "'1'", 'when', '(', stq_issue, '/=', stq_resp, ')', 'else ', "'0'")

            load_outstanding_arr = LogicArray(ctx, 'load_outstanding_arr', 'w', self.configs.numLdqEntries)
            load_outstanding = Logic(ctx, 'load_outstanding', 'w')
            for i in range(self.configs.numLdqEntries):
                arch += Op(ctx, load_outstanding_arr[i], "'1'", 'when', '(', ldq_issue[i], 'and', 'not', ldq_data_valid[i], ')', 'else', "'0'")
            arch += Reduce(ctx, load_outstanding, load_outstanding_arr, 'or')

            # We can issue the fallback load candidate if:
            # - It is older than the oldest store (implicit in fallback_load_is_oldest_oh[]).
            # - It has a valid address.
            # - There are no outstanding stores.
            for i in range(self.configs.numLdqEntries):
                arch += Op(ctx, (fallback_load_idx_oh, i), fallback_load_is_oldest_oh[i], 'and', ldq_addr_valid_p0[i], 'and', 'not', store_outstanding)
            arch += Reduce(ctx, fallback_load_en, fallback_load_idx_oh, 'or')

            # We can issue the fallback store candidate (if it is valid) if:
            # - It is older than the oldest store (NOT fallback_load_is_oldest).
            # - There are no outstanding loads.
            arch += Op(ctx, fallback_store_en_if_valid, 'not', fallback_load_is_oldest, 'and', 'not', load_outstanding)

        # Store
        # When pipelining (pipe0) is enabled, this uses look-ahead to the next store entry to reduce the critical path.
        # Both the current and next stores are checked for validity and conflicts, and the result is multiplexed "late
        # in the clock cycle" to reduce the critical path. When pipelining is disabled, only the current store entry is
        # checked, so there is no need for computing the signals for the next store entry, and for the multiplexing.

        # Store request is valid if the entry is allocated and has valid address+data.
        store_req_valid_arr = LogicArray(ctx, 'store_req_valid_arr', 'w', self.configs.numStqEntries)
        for i in range(self.configs.numStqEntries):
            arch += Op(ctx, store_req_valid_arr[i], stq_alloc_pcomp[i], 'and', stq_addr_valid_pcomp[i], 'and', stq_data_valid_pcomp[i])

        store_conflict = Logic(ctx, 'store_conflict', 'w')
        store_req_valid_p0 = Logic(ctx, 'store_req_valid_p0', pipe0_type)
        st_ld_conflict_p0 = LogicVec(ctx, 'st_ld_conflict_p0', pipe0_type, self.configs.numLdqEntries)
        if self.configs.pipe0:
            store_req_valid_p0.regInit(init=0)
            st_ld_conflict_p0.regInit()
        if self.configs.fallbackIssue:
            store_is_older_arr_p0 = LogicArray(ctx, 'store_is_older_arr_p0', pipe0_type, self.configs.numLdqEntries)
            if self.configs.pipe0:
                store_is_older_arr_p0.regInit()

        # next issue pointer (needed for look-ahead when pipelining is enabled)
        if self.configs.pipe0:
            stq_issue_next = LogicVec(ctx, 'stq_issue_next', 'w', self.configs.stqAddrW)
            arch += WrapAddConst(ctx, stq_issue_next, stq_issue, 1, self.configs.numStqEntries)

        # checks for current and next (if needed) store entry
        store_req_valid_curr = Logic(ctx, 'store_req_valid_curr', 'w')
        store_is_older_arr_curr = LogicArray(ctx, 'store_is_older_arr_curr', 'w', self.configs.numLdqEntries)
        st_ld_conflict_curr = LogicVec(ctx, 'st_ld_conflict_curr', 'w', self.configs.numLdqEntries)
        if self.configs.pipe0:
            # with pipelining: also compute for the next entry
            store_req_valid_next = Logic(ctx, 'store_req_valid_next', 'w')
            store_is_older_arr_next = LogicArray(ctx, 'store_is_older_arr_next', 'w', self.configs.numLdqEntries)
            st_ld_conflict_next = LogicVec(ctx, 'st_ld_conflict_next', 'w', self.configs.numLdqEntries)

        # validity lookup
        arch += MuxLookUp(ctx, store_req_valid_curr, store_req_valid_arr, stq_issue)
        if self.configs.pipe0:
            # with pipelining: also compute for the next entry
            arch += MuxLookUp(ctx, store_req_valid_next, store_req_valid_arr, stq_issue_next)

        # extract column from order matrix
        for i in range(self.configs.numLdqEntries):
            arch += Op(ctx, store_is_older_arr_curr[i], MuxIndex(store_is_older_pcomp[i], stq_issue))
        if self.configs.pipe0:
            for i in range(self.configs.numLdqEntries):
                # with pipelining: also compute for the next entry
                arch += Op(ctx, store_is_older_arr_next[i], MuxIndex(store_is_older_pcomp[i], stq_issue_next))

        # A store conflicts with a load when:
        # 1. The load entry is valid, and
        # 2. The load entry hasn't completed (received data from memory), and
        # 3. The load is older than the store, and
        # 4. The address conflicts(same or invalid store address).
        # Index order are reversed for store matrix.
        for i in range(self.configs.numLdqEntries):
            arch += Op(ctx,
                       (st_ld_conflict_curr, i),
                       (ldq_alloc_pcomp, i), 'and',
                       'not', (load_completed, i), 'and',
                       'not', store_is_older_arr_curr[i], 'and',
                       '(', MuxIndex(
                           addr_same_pcomp[i], stq_issue), 'or', 'not', (ldq_addr_valid_pcomp, i), ')'
                       )
        if self.configs.pipe0:
            # with pipelining: also compute for the next entry
            for i in range(self.configs.numLdqEntries):
                arch += Op(ctx,
                           (st_ld_conflict_next, i),
                           (ldq_alloc_pcomp, i), 'and',
                           'not', (load_completed, i), 'and',
                           'not', store_is_older_arr_next[i], 'and',
                           '(', MuxIndex(
                               addr_same_pcomp[i], stq_issue_next), 'or', 'not', (ldq_addr_valid_pcomp, i), ')'
                           )

        if self.configs.pipe0:
            # with pipelining: multiplex between current and next store entry
            # Multiplex from current and next
            arch += Op(ctx, st_ld_conflict_p0, st_ld_conflict_next,
                       'when', stq_issue_en, 'else', st_ld_conflict_curr)
            arch += Op(ctx, store_req_valid_p0, store_req_valid_next, 'when',
                       stq_issue_en, 'else', store_req_valid_curr)
            if self.configs.fallbackIssue:
                for i in range(self.configs.numLdqEntries):
                    arch += Op(ctx, store_is_older_arr_p0[i], store_is_older_arr_next[i], 'when',
                               stq_issue_en, 'else', store_is_older_arr_curr[i])
        else:
            # without pipelining: only consider current store entry
            arch += Op(ctx, st_ld_conflict_p0, st_ld_conflict_curr)
            arch += Op(ctx, store_req_valid_p0, store_req_valid_curr)
            if self.configs.fallbackIssue:
                for i in range(self.configs.numLdqEntries):
                    arch += Op(ctx, store_is_older_arr_p0[i], store_is_older_arr_curr[i])

        # The store conflicts with any load
        arch += Reduce(ctx, store_conflict, st_ld_conflict_p0, 'or')
        arch += Op(ctx, store_idx, stq_issue)
        # The store can be issued when it is valid AND (no conflict OR it is older than the fallback load).
        if self.configs.fallbackIssue:
            arch += Op(ctx, store_en, store_req_valid_p0, 'and', '(', 'not', store_conflict, 'or', fallback_store_en_if_valid, ')')
            # ordering information needed by fallback issue logic
            for i in range(self.configs.numLdqEntries):
                arch += Op(ctx, store_is_older_arr[i], store_is_older_arr_p0[i])
        else:
            arch += Op(ctx, store_en, store_req_valid_p0, 'and', 'not', store_conflict)

        # Bypass
        bypass_idx_oh_p0 = LogicVecArray(
            ctx, 'bypass_idx_oh_p0', pipe0_type, self.configs.numLdqEntries, self.configs.numStqEntries)
        bypass_en = LogicArray(ctx, 'bypass_en', 'w',
                               self.configs.numLdqEntries)
        if self.configs.pipe0:
            bypass_idx_oh_p0.regInit()
        if self.configs.bypass:
            stq_last_oh = LogicVec(
                ctx, 'stq_last_oh', 'w', self.configs.numStqEntries)
            arch += BitsToOHSub1(ctx, stq_last_oh, stq_tail)
            for i in range(0, self.configs.numLdqEntries):
                bypass_en_vec = LogicVec(
                    ctx, f'bypass_en_vec_{i}', 'w', self.configs.numStqEntries)
                # Search for the youngest store that is older than the load and conflicts
                arch += CyclicPriorityMasking(
                    ctx, bypass_idx_oh_p0[i], ld_st_conflict[i], stq_last_oh, True)
                # Check if the youngest conflict store can bypass with the load
                arch += Op(ctx, bypass_en_vec,
                           bypass_idx_oh_p0[i], 'and', can_bypass[i])
                arch += Reduce(ctx, bypass_en[i], bypass_en_vec, 'or')
        else:
            # bypass disabled: tie bypass signals low
            for i in range(0, self.configs.numLdqEntries):
                arch += Op(ctx, bypass_en[i], 0)
            for i in range(0, self.configs.numLdqEntries):
                arch += Op(ctx, bypass_idx_oh_p0[i], 0)

        # Pipeline Stage 1

        # load registers (if enabled, w/ backpressure)
        load_idx_oh_p1 = LogicVecArray(
            ctx, 'load_idx_oh_p1', pipe1_type, self.configs.numLdMem, self.configs.numLdqEntries)
        load_en_p1 = LogicArray(
            ctx, 'load_en_p1', pipe1_type, self.configs.numLdMem)
        # store registers (if enabled, w/ backpressure)
        store_idx_p1 = LogicVec(
            ctx, 'store_idx_p1', pipe1_type, self.configs.stqAddrW)
        store_en_p1 = Logic(
            ctx, 'store_en_p1', pipe1_type)
        # bypass registers (if enabled, w/o backpressure)
        bypass_idx_oh_p1 = LogicVecArray(
            ctx, 'bypass_idx_oh_p1', pipe1_type, self.configs.numLdqEntries, self.configs.numStqEntries)
        bypass_en_p1 = LogicArray(
            ctx, 'bypass_en_p1', pipe1_type, self.configs.numLdqEntries)

        load_p1_ready = LogicArray(ctx, 'load_p1_ready', 'w', self.configs.numLdMem)
        store_p1_ready = Logic(ctx, 'store_p1_ready', 'w')

        if self.configs.pipe1:
            # pipeline register control signals (load_*_p1, store_*_p1)
            # This implements a pipeline register stage with backpressure and
            # with a # combinational path from output ready to input ready. We
            # are ready # for new data if either there is a handshake at the
            # output (*_hs), # or the register is currently empty (not *_en_p1).
            load_hs = LogicArray(ctx, 'load_hs', 'w', self.configs.numLdMem)
            for w in range(0, self.configs.numLdMem):
                arch += Op(ctx, load_hs[w], load_en_p1[w], 'and', rreq_ready_i[w])
                arch += Op(ctx, load_p1_ready[w], load_hs[w], 'or', 'not', load_en_p1[w])
            store_hs = Logic(ctx, 'store_hs', 'w')
            arch += Op(ctx, store_hs, store_en_p1, 'and', wreq_ready_i[0])
            arch += Op(ctx, store_p1_ready, store_hs, 'or', 'not', store_en_p1)
            # register init
            load_idx_oh_p1.regInit(enable=load_p1_ready)
            load_en_p1.regInit(init=[0]*self.configs.numLdMem, enable=load_p1_ready)
            store_idx_p1.regInit(enable=store_p1_ready)
            store_en_p1.regInit(init=0, enable=store_p1_ready)
            bypass_idx_oh_p1.regInit()
            bypass_en_p1.regInit(init=[0]*self.configs.numLdqEntries)
        else:
            # non-pipelined "pseudo-control" signals
            for w in range(0, self.configs.numLdMem):
                arch += Op(ctx, load_p1_ready[w], rreq_ready_i[w], 'and', load_en[w])
            arch += Op(ctx, store_p1_ready, wreq_ready_i[0])

        # pipeline register assignments
        for w in range(0, self.configs.numLdMem):
            arch += Op(ctx, load_idx_oh_p1[w], load_idx_oh[w])
            arch += Op(ctx, load_en_p1[w], load_en[w])
        arch += Op(ctx, store_idx_p1, store_idx)
        arch += Op(ctx, store_en_p1, store_en)
        for i in range(0, self.configs.numLdqEntries):
            arch += Op(ctx, bypass_idx_oh_p1[i], bypass_idx_oh_p0[i])
            arch += Op(ctx, bypass_en_p1[i], bypass_en[i])

        ######    Read/Write    ######
        # Read Request
        for w in range(0, self.configs.numLdMem):
            arch += Op(ctx, rreq_valid_o[w], load_en_p1[w])
            arch += OHToBits(ctx, rreq_id_o[w], load_idx_oh_p1[w])
            arch += Mux1H(ctx, rreq_addr_o[w], ldq_addr, load_idx_oh_p1[w])

        for i in range(0, self.configs.numLdqEntries):
            ldq_issue_set_vec = LogicVec(
                ctx, f'ldq_issue_set_vec_{i}', 'w', self.configs.numLdMem)
            for w in range(0, self.configs.numLdMem):
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
        for i in range(0, self.configs.numLdqEntries):
            # check each read response channel for each load
            read_idx_oh = LogicArray(
                ctx, f'read_idx_oh_{i}', 'w', self.configs.numLdMem)
            read_valid = Logic(ctx, f'read_valid_{i}', 'w')
            read_data = LogicVec(
                ctx, f'read_data_{i}', 'w', self.configs.dataW)
            for w in range(0, self.configs.numLdMem):
                arch += Op(ctx, read_idx_oh[w], rresp_valid_i[w], 'when',
                           '(', rresp_id_i[w], '=', (i, self.configs.idW), ')', 'else', '\'0\'')
            arch += Mux1H(ctx, read_data, rresp_data_i, read_idx_oh)
            arch += Reduce(ctx, read_valid, read_idx_oh, 'or')
            # multiplex from store queue data
            bypass_data = LogicVec(
                ctx, f'bypass_data_{i}', 'w', self.configs.dataW)
            arch += Mux1H(ctx, bypass_data, stq_data, bypass_idx_oh_p1[i])
            # multiplex from read and bypass data
            arch += Op(ctx, ldq_data[i], read_data, 'or', bypass_data)
            arch += Op(ctx, ldq_data_wen[i],
                       bypass_en_p1[i], 'or', read_valid)
        for w in range(0, self.configs.numLdMem):
            arch += Op(ctx, rresp_ready_o[w], '\'1\'')

        # Write Response
        if self.configs.stResp:
            for i in range(0, self.configs.numStqEntries):
                arch += Op(ctx, stq_exec_set[i],
                           wresp_valid_i[0], 'when',
                           '(', stq_resp, '=', (i, self.configs.stqAddrW), ')',
                           'else', '\'0\''
                           )
        else:
            for i in range(0, self.configs.numStqEntries):
                arch += Op(ctx, stq_reset[i],
                           wresp_valid_i[0], 'when',
                           '(', stq_resp, '=', (i, self.configs.stqAddrW), ')',
                           'else', '\'0\''
                           )
        arch += Op(ctx, stq_resp_en, wresp_valid_i[0])
        arch += Op(ctx, wresp_ready_o[0], '\'1\'')

        ######   Write To File  ######
        ctx.portInitString += '\n\t);'
        ctx.regInitString += '\tend process;\n'

        # Write to the file
        with open(f'{path_rtl}/{self.name}.vhd', 'a') as file:
            # with open(name + '.vhd', 'w') as file:
            file.write(ctx.library)
            file.write(f'entity {self.module_name} is\n')
            file.write(ctx.portInitString)
            file.write('\nend entity;\n\n')
            file.write(f'architecture arch of {self.module_name} is\n')
            file.write(ctx.signalInitString)
            file.write('begin\n' + arch + '\n')
            file.write(ctx.regInitString + 'end architecture;\n')

    def instantiate(self, **kwargs) -> str:
        """
        *Instantiation of LSQ is in lsq-generator.py.
        """
        pass
