from core_gen.emitters import Emitter
from core_gen.signals import *
from core_gen.operators import *
from core_gen.configs import Configs
from core_gen.ir import BinOp, Bin, Val, Bit, CustomStatement

import core_gen.generators.lsq_submodule_wrapper as lsq_submodule_wrapper


class LSQ:
    def __init__(self, name: str, suffix: str, configs: Configs):
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

    def generate(self, em: Emitter, lsq_submodules, path_rtl) -> None:
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
            em              : an instance of the Emitter class used for code generation
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
        ###### LSQ Architecture ######
        ######        IOs       ######

        # group initialzation signals
        group_init_valid_i = LogicArray(
            em, "group_init_valid", "i", self.configs.numGroups
        )
        group_init_ready_o = LogicArray(
            em, "group_init_ready", "o", self.configs.numGroups
        )

        # Memory access ports, i.e., the connection "kernel -> LSQ"
        # Load address channel (addr, valid, ready) from kernel, contains signals:
        ldp_addr_i = LogicVecArray(
            em, "ldp_addr", "i", self.configs.numLdPorts, self.configs.addrW
        )
        ldp_addr_valid_i = LogicArray(
            em, "ldp_addr_valid", "i", self.configs.numLdPorts
        )
        ldp_addr_ready_o = LogicArray(
            em, "ldp_addr_ready", "o", self.configs.numLdPorts
        )

        # Load data channel (data, valid, ready) to kernel
        ldp_data_o = LogicVecArray(
            em, "ldp_data", "o", self.configs.numLdPorts, self.configs.dataW
        )
        ldp_data_valid_o = LogicArray(
            em, "ldp_data_valid", "o", self.configs.numLdPorts
        )
        ldp_data_ready_i = LogicArray(
            em, "ldp_data_ready", "i", self.configs.numLdPorts
        )

        # Store address channel (addr, valid, ready) from kernel
        stp_addr_i = LogicVecArray(
            em, "stp_addr", "i", self.configs.numStPorts, self.configs.addrW
        )
        stp_addr_valid_i = LogicArray(
            em, "stp_addr_valid", "i", self.configs.numStPorts
        )
        stp_addr_ready_o = LogicArray(
            em, "stp_addr_ready", "o", self.configs.numStPorts
        )

        # Store data channel (data, valid, ready) from kernel
        stp_data_i = LogicVecArray(
            em, "stp_data", "i", self.configs.numStPorts, self.configs.dataW
        )
        stp_data_valid_i = LogicArray(
            em, "stp_data_valid", "i", self.configs.numStPorts
        )
        stp_data_ready_o = LogicArray(
            em, "stp_data_ready", "o", self.configs.numStPorts
        )

        if self.configs.stResp:
            stp_exec_valid_o = LogicArray(
                em, "stp_exec_valid", "o", self.configs.numStPorts
            )
            stp_exec_ready_i = LogicArray(
                em, "stp_exec_ready", "i", self.configs.numStPorts
            )

        # queue empty signal
        empty_o = Logic(em, "empty", "o")

        # Memory interface: i.e., the connection LSQ -> AXI
        # We assume that the memory interface has
        # 1. A read request channel (rreq) and a read response channel (rresp).
        # 2. A write request channel (wreq) and a write response channel (wresp).
        rreq_valid_o = LogicArray(em, "rreq_valid", "o", self.configs.numLdMem)
        rreq_ready_i = LogicArray(em, "rreq_ready", "i", self.configs.numLdMem)
        rreq_id_o = LogicVecArray(
            em, "rreq_id", "o", self.configs.numLdMem, self.configs.idW
        )
        rreq_addr_o = LogicVecArray(
            em, "rreq_addr", "o", self.configs.numLdMem, self.configs.addrW
        )

        rresp_valid_i = LogicArray(em, "rresp_valid", "i", self.configs.numLdMem)
        rresp_ready_o = LogicArray(em, "rresp_ready", "o", self.configs.numLdMem)
        rresp_id_i = LogicVecArray(
            em, "rresp_id", "i", self.configs.numLdMem, self.configs.idW
        )
        rresp_data_i = LogicVecArray(
            em, "rresp_data", "i", self.configs.numLdMem, self.configs.dataW
        )

        wreq_valid_o = LogicArray(em, "wreq_valid", "o", self.configs.numStMem)
        wreq_ready_i = LogicArray(em, "wreq_ready", "i", self.configs.numStMem)
        wreq_id_o = LogicVecArray(
            em, "wreq_id", "o", self.configs.numStMem, self.configs.idW
        )
        wreq_addr_o = LogicVecArray(
            em, "wreq_addr", "o", self.configs.numStMem, self.configs.addrW
        )
        wreq_data_o = LogicVecArray(
            em, "wreq_data", "o", self.configs.numStMem, self.configs.dataW
        )

        wresp_valid_i = LogicArray(em, "wresp_valid", "i", self.configs.numStMem)
        wresp_ready_o = LogicArray(em, "wresp_ready", "o", self.configs.numStMem)
        wresp_id_i = LogicVecArray(
            em, "wresp_id", "i", self.configs.numStMem, self.configs.idW
        )

        # Pointer related signals
        # For updating pointers
        num_loads = LogicVec(em, "num_loads", "w", self.configs.ldqAddrW)
        num_stores = LogicVec(em, "num_stores", "w", self.configs.stqAddrW)
        stq_issue_en = Logic(em, "stq_issue_en", "w")
        stq_resp_en = Logic(em, "stq_resp_en", "w")
        # Generated by pointers
        ldq_empty = Logic(em, "ldq_empty", "w")
        stq_empty = Logic(em, "stq_empty", "w")
        ldq_head_oh = LogicVec(em, "ldq_head_oh", "w", self.configs.numLdqEntries)
        stq_head_oh = LogicVec(em, "stq_head_oh", "w", self.configs.numStqEntries)
        #! If this is the lsq master, then we need the following logic
        #! Define new interfaces needed by dynamatic
        if self.configs.master:
            memStart_ready = Logic(em, "memStart_ready", "o")
            memStart_valid = Logic(em, "memStart_valid", "i")
            ctrlEnd_ready = Logic(em, "ctrlEnd_ready", "o")
            ctrlEnd_valid = Logic(em, "ctrlEnd_valid", "i")
            memEnd_ready = Logic(em, "memEnd_ready", "i")
            memEnd_valid = Logic(em, "memEnd_valid", "o")

            #! Add extra signals required
            memStartReady = Logic(em, "memStartReady", "w", force_reg=True)
            memEndValid = Logic(em, "memEndValid", "w", force_reg=True)
            ctrlEndReady = Logic(em, "ctrlEndReady", "w", force_reg=True)
            temp_gen_mem = Logic(em, "TEMP_GEN_MEM", "w", force_reg=True)

            #! The memory completion signal cannot be set to 1 when any group is allocating:
            binops = [
                Bin(Val(group_init_valid_i, i), BinOp.OR, None)
                for i in range(group_init_valid_i.length - 1)
            ]
            binops.append(Val(group_init_valid_i, group_init_valid_i.length - 1))
            for i, op in enumerate(binops[:-1]):
                op.right = binops[i + 1]
            no_curr_ga = ~binops[0]

            #! Define the needed logic
            em.add_comment(
                "This signal indicates that all mem. ops are completed and func. can return."
            )
            em.add_comment("LSQ can return iff all the following conditions are true:")
            em.add_comment("1. No more upcoming BBs containing memory accesses.")
            em.add_comment("2. Both store and load queues are empty.")
            em.add_comment("3. No GA in the same cycle.")
            em.add_assignment(
                temp_gen_mem, ctrlEnd_valid & stq_empty & ldq_empty & no_curr_ga
            )

            em.add_comment("Define logic for the new interfaces needed by dynamatic")
            vhdl_str = ""
            # TODO: Add proper emitter functions in order to do this
            vhdl_str += "\tprocess (clk) is\n\tbegin\n"
            vhdl_str += "\t" * 2 + "if rising_edge(clk) then\n"
            vhdl_str += "\t" * 3 + "if rst = '1' then\n"
            vhdl_str += "\t" * 4 + "memStartReady <= '1';\n"
            vhdl_str += "\t" * 4 + "memEndValid <= '0';\n"
            vhdl_str += "\t" * 4 + "ctrlEndReady <= '0';\n"
            vhdl_str += "\t" * 3 + "else\n"
            vhdl_str += (
                "\t" * 4
                + "memStartReady <= (memEndValid and memEnd_ready_i) or ((not (memStart_valid_i and memStartReady)) and memStartReady);\n"
            )
            vhdl_str += "\t" * 4 + "memEndValid <= TEMP_GEN_MEM or memEndValid;\n"
            vhdl_str += (
                "\t" * 4
                + "ctrlEndReady <= (not (ctrlEnd_valid_i and ctrlEndReady)) and (TEMP_GEN_MEM or ctrlEndReady);\n"
            )
            vhdl_str += "\t" * 3 + "end if;\n"
            vhdl_str += "\t" * 2 + "end if;\n"
            vhdl_str += "\tend process;\n\n"

            verilog_str = """
always @(posedge clk) begin
    if (rst) begin
        memStartReady <= 1'b1;
        memEndValid   <= 1'b0;
        ctrlEndReady  <= 1'b0;
    end
    else begin
        memStartReady <= (memEndValid && memEnd_ready_i) ||
                         ((!(memStart_valid_i && memStartReady)) && memStartReady);

        memEndValid   <= TEMP_GEN_MEM || memEndValid;

        ctrlEndReady  <= (!(ctrlEnd_valid_i && ctrlEndReady)) &&
                         (TEMP_GEN_MEM || ctrlEndReady);
    end
end
            """

            em.add_custom_statement(CustomStatement(vhdl_str, verilog_str))

            #! Assign signals for the newly added ports
            em.add_comment("Update new memory interfaces")
            em.add_assignment(memStart_ready, memStartReady)
            em.add_assignment(ctrlEnd_ready, ctrlEndReady)
            em.add_assignment(memEnd_valid, memEndValid)

        ######  Queue Registers ######
        # Load Queue Entries
        ldq_alloc = LogicArray(em, "ldq_alloc", "r", self.configs.numLdqEntries)
        ldq_issue = LogicArray(em, "ldq_issue", "r", self.configs.numLdqEntries)
        if self.configs.ldpAddrW > 0:
            ldq_port_idx = LogicVecArray(
                em,
                "ldq_port_idx",
                "r",
                self.configs.numLdqEntries,
                self.configs.ldpAddrW,
            )
        else:
            ldq_port_idx = None
        ldq_addr_valid = LogicArray(
            em, "ldq_addr_valid", "r", self.configs.numLdqEntries
        )
        ldq_addr = LogicVecArray(
            em, "ldq_addr", "r", self.configs.numLdqEntries, self.configs.addrW
        )
        ldq_data_valid = LogicArray(
            em, "ldq_data_valid", "r", self.configs.numLdqEntries
        )
        ldq_data = LogicVecArray(
            em, "ldq_data", "r", self.configs.numLdqEntries, self.configs.dataW
        )

        # Store Queue Entries
        stq_alloc = LogicArray(em, "stq_alloc", "r", self.configs.numStqEntries)
        if self.configs.stResp:
            stq_exec = LogicArray(em, "stq_exec", "r", self.configs.numStqEntries)
        if self.configs.stpAddrW > 0:
            stq_port_idx = LogicVecArray(
                em,
                "stq_port_idx",
                "r",
                self.configs.numStqEntries,
                self.configs.stpAddrW,
            )
        else:
            stq_port_idx = None
        stq_addr_valid = LogicArray(
            em, "stq_addr_valid", "r", self.configs.numStqEntries
        )
        stq_addr = LogicVecArray(
            em, "stq_addr", "r", self.configs.numStqEntries, self.configs.addrW
        )
        stq_data_valid = LogicArray(
            em, "stq_data_valid", "r", self.configs.numStqEntries
        )
        stq_data = LogicVecArray(
            em, "stq_data", "r", self.configs.numStqEntries, self.configs.dataW
        )

        # Order for load-store
        store_is_older = LogicVecArray(
            em,
            "store_is_older",
            "r",
            self.configs.numLdqEntries,
            self.configs.numStqEntries,
        )

        # Pointers
        ldq_tail = LogicVec(em, "ldq_tail", "r", self.configs.ldqAddrW)
        ldq_head = LogicVec(em, "ldq_head", "r", self.configs.ldqAddrW)

        stq_tail = LogicVec(em, "stq_tail", "r", self.configs.stqAddrW)
        stq_head = LogicVec(em, "stq_head", "r", self.configs.stqAddrW)
        stq_issue = LogicVec(em, "stq_issue", "r", self.configs.stqAddrW)
        stq_resp = LogicVec(em, "stq_resp", "r", self.configs.stqAddrW)

        # Entry related signals
        # From port dispatchers
        ldq_wen = LogicArray(em, "ldq_wen", "w", self.configs.numLdqEntries)
        ldq_addr_wen = LogicArray(em, "ldq_addr_wen", "w", self.configs.numLdqEntries)
        ldq_reset = LogicArray(em, "ldq_reset", "w", self.configs.numLdqEntries)
        stq_wen = LogicArray(em, "stq_wen", "w", self.configs.numStqEntries)
        stq_addr_wen = LogicArray(em, "stq_addr_wen", "w", self.configs.numStqEntries)
        stq_data_wen = LogicArray(em, "stq_data_wen", "w", self.configs.numStqEntries)
        stq_reset = LogicArray(em, "stq_reset", "w", self.configs.numStqEntries)
        # From Read/Write Block
        ldq_data_wen = LogicArray(em, "ldq_data_wen", "w", self.configs.numLdqEntries)
        ldq_issue_set = LogicArray(em, "ldq_issue_set", "w", self.configs.numLdqEntries)
        if self.configs.stResp:
            stq_exec_set = LogicArray(
                em, "stq_exec_set", "w", self.configs.numStqEntries
            )
        # Form Group Allocator
        ga_ls_order = LogicVecArray(
            em,
            "ga_ls_order",
            "w",
            self.configs.numLdqEntries,
            self.configs.numStqEntries,
        )

        BitsToOH(em, ldq_head_oh, ldq_head)
        BitsToOH(em, stq_head_oh, stq_head)

        # update queue entries
        # load queue
        if self.configs.pipe0 or self.configs.pipeComp:
            ldq_wen_p0 = LogicArray(em, "ldq_wen_p0", "r", self.configs.numLdqEntries)
            ldq_wen_p0.regInit()
            if self.configs.pipe0 and self.configs.pipeComp:
                ldq_wen_p1 = LogicArray(
                    em, "ldq_wen_p1", "r", self.configs.numLdqEntries
                )
                ldq_wen_p1.regInit()
        ldq_alloc_next = LogicArray(
            em, "ldq_alloc_next", "w", self.configs.numLdqEntries
        )
        for i in range(0, self.configs.numLdqEntries):
            em.add_assignment(ldq_alloc_next[i], ~ldq_reset[i] & ldq_alloc[i])
            em.add_assignment(ldq_alloc[i], ldq_wen[i] | ldq_alloc_next[i])
            if self.configs.pipe0 or self.configs.pipeComp:
                em.add_assignment(ldq_wen_p0[i], ldq_wen[i])
                if self.configs.pipe0 and self.configs.pipeComp:
                    em.add_assignment(ldq_wen_p1[i], ldq_wen[i])
                    em.add_assignment(
                        ldq_issue[i], ~ldq_wen_p1[i] & (ldq_issue_set[i] | ldq_issue[i])
                    )
                else:
                    em.add_assignment(
                        ldq_issue[i], ~ldq_wen_p0[i] & (ldq_issue_set[i] | ldq_issue[i])
                    )
            else:
                em.add_assignment(
                    ldq_issue[i], ~ldq_wen[i] & (ldq_issue_set[i] | ldq_issue[i])
                )

            em.add_assignment(
                ldq_addr_valid[i], ~ldq_wen[i] & (ldq_addr_wen[i] | ldq_addr_valid[i])
            )
            em.add_assignment(
                ldq_data_valid[i], ~ldq_wen[i] & (ldq_data_wen[i] | ldq_data_valid[i])
            )
        # store queue
        stq_alloc_next = LogicArray(
            em, "stq_alloc_next", "w", self.configs.numStqEntries
        )
        for i in range(0, self.configs.numStqEntries):
            em.add_assignment(stq_alloc_next[i], ~stq_reset[i] & stq_alloc[i])
            em.add_assignment(stq_alloc[i], stq_wen[i] | stq_alloc_next[i])
            if self.configs.stResp:
                em.add_assignment(
                    stq_exec[i], ~stq_wen[i] & (stq_exec_set[i] | stq_exec[i])
                )
            em.add_assignment(
                stq_addr_valid[i], ~stq_wen[i] & (stq_addr_wen[i] | stq_addr_valid[i])
            )
            em.add_assignment(
                stq_data_valid[i], ~stq_wen[i] & (stq_data_wen[i] | stq_data_valid[i])
            )

        # order matrix
        # store_is_older(i,j) = (not stq_reset(j) and (stq_alloc(j) or ga_ls_order(i, j)))
        #                  when ldq_wen(i)
        #                  else not stq_reset(j) and store_is_older(i, j)
        for i in range(0, self.configs.numLdqEntries):
            for j in range(0, self.configs.numStqEntries):
                em.add_assignment(
                    (store_is_older, i, j),
                    (~Val(stq_reset, j) & (Val(stq_alloc, j) | Val(ga_ls_order, i, j)))
                    .when(Val(ldq_wen, i))
                    .else_(~Val(stq_reset, j) & Val(store_is_older, i, j)),
                )

        # pointers update
        ldq_not_empty = Logic(em, "ldq_not_empty", "w")
        stq_not_empty = Logic(em, "stq_not_empty", "w")
        Reduce(em, ldq_not_empty, ldq_alloc, BinOp.OR)
        em.add_assignment(ldq_empty, ~ldq_not_empty)
        MuxLookUp(em, stq_not_empty, stq_alloc, stq_head)
        em.add_assignment(stq_empty, ~stq_not_empty)
        em.add_assignment(empty_o, ldq_empty & stq_empty)

        WrapAdd(em, ldq_tail, ldq_tail, num_loads, self.configs.numLdqEntries)
        WrapAdd(em, stq_tail, stq_tail, num_stores, self.configs.numStqEntries)
        WrapAddConst(em, stq_issue, stq_issue, 1, self.configs.numStqEntries)
        WrapAddConst(em, stq_resp, stq_resp, 1, self.configs.numStqEntries)

        ldq_tail_oh = LogicVec(em, "ldq_tail_oh", "w", self.configs.numLdqEntries)
        BitsToOH(em, ldq_tail_oh, ldq_tail)
        ldq_head_next_oh = LogicVec(
            em, "ldq_head_next_oh", "w", self.configs.numLdqEntries
        )
        ldq_head_next = LogicVec(em, "ldq_head_next", "w", self.configs.ldqAddrW)
        ldq_head_sel = Logic(em, "ldq_head_sel", "w")
        if self.configs.headLag:
            # Update the head pointer according to the valid signal of last cycle
            CyclicPriorityMasking(em, ldq_head_next_oh, ldq_alloc, ldq_tail_oh)
            Reduce(em, ldq_head_sel, ldq_alloc, BinOp.OR)
        else:
            CyclicPriorityMasking(em, ldq_head_next_oh, ldq_alloc_next, ldq_tail_oh)
            Reduce(em, ldq_head_sel, ldq_alloc_next, BinOp.OR)
        OHToBits(em, ldq_head_next, ldq_head_next_oh)
        em.add_assignment(ldq_head, ldq_head_next.when(ldq_head_sel).else_(ldq_tail))

        stq_tail_oh = LogicVec(em, "stq_tail_oh", "w", self.configs.numStqEntries)
        BitsToOH(em, stq_tail_oh, stq_tail)
        stq_head_next_oh = LogicVec(
            em, "stq_head_next_oh", "w", self.configs.numStqEntries
        )
        stq_head_next = LogicVec(em, "stq_head_next", "w", self.configs.stqAddrW)
        stq_head_sel = Logic(em, "stq_head_sel", "w")
        if self.configs.stResp:
            if self.configs.headLag:
                # Update the head pointer according to the valid signal of last cycle
                CyclicPriorityMasking(em, stq_head_next_oh, stq_alloc, stq_tail_oh)
                Reduce(em, stq_head_sel, stq_alloc, BinOp.OR)
            else:
                CyclicPriorityMasking(em, stq_head_next_oh, stq_alloc_next, stq_tail_oh)
                Reduce(em, stq_head_sel, stq_alloc_next, BinOp.OR)
            OHToBits(em, stq_head_next, stq_head_next_oh)
            em.add_assignment(
                stq_head, stq_head_next.when(stq_head_sel).else_(stq_tail)
            )
        else:
            WrapAddConst(em, stq_head_next, stq_head, 1, self.configs.numStqEntries)
            em.add_assignment(stq_head_sel, wresp_valid_i[0])
            em.add_assignment(
                stq_head, stq_head_next.when(stq_head_sel).else_(stq_head)
            )

        # Load Queue Entries
        ldq_alloc.regInit(init=[0] * self.configs.numLdqEntries)
        ldq_issue.regInit()
        if self.configs.ldpAddrW > 0:
            ldq_port_idx.regInit(ldq_wen)
        ldq_addr_valid.regInit()
        ldq_addr.regInit(ldq_addr_wen)
        ldq_data_valid.regInit()
        ldq_data.regInit(ldq_data_wen)

        # Store Queue Entries
        stq_alloc.regInit(init=[0] * self.configs.numStqEntries)
        if self.configs.stResp:
            stq_exec.regInit()
        if self.configs.stpAddrW > 0:
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
        lsq_submodules.group_allocator.instantiate(
            em,
            group_init_valid_i,
            group_init_ready_o,
            ldq_tail,
            ldq_head,
            ldq_empty,
            stq_tail,
            stq_head,
            stq_empty,
            ldq_wen,
            num_loads,
            ldq_port_idx,
            stq_wen,
            num_stores,
            stq_port_idx,
            ga_ls_order,
        )

        # When the condition "lsq_submodules.ptq_dispatcher_lda != None" is not true:
        # The dispatcher module will be set to None when there are zero load ports.
        # In this case, do not instantiate dispatching logic when there are zero load ports.
        # - WARNING: This logic needs more testing
        # - TODO: Also remove the load queue when there are zero load ports.
        if lsq_submodules.ptq_dispatcher_lda != None:
            # Load Address Port Dispatcher
            lsq_submodules.ptq_dispatcher_lda.instantiate(
                em,
                ldp_addr_i,
                ldp_addr_valid_i,
                ldp_addr_ready_o,
                ldq_alloc,
                ldq_addr_valid,
                ldq_port_idx,
                ldq_addr,
                ldq_addr_wen,
                ldq_head_oh,
            )

        # When the condition "lsq_submodules.qtp_dispatcher_ldd != None" is not true:
        # The dispatcher module will be set to None when there are zero load ports.
        # In this case, do not instantiate dispatching logic when there are zero load ports.
        # - WARNING: This logic needs more testing
        # - TODO: Also remove the load queue when there are zero load ports.
        if lsq_submodules.qtp_dispatcher_ldd != None:
            # Load Data Port Dispatcher
            lsq_submodules.qtp_dispatcher_ldd.instantiate(
                em,
                ldp_data_o,
                ldp_data_valid_o,
                ldp_data_ready_i,
                ldq_alloc,
                ldq_data_valid,
                ldq_port_idx,
                ldq_data,
                ldq_reset,
                ldq_head_oh,
            )

        # Store Address Port Dispatcher
        lsq_submodules.ptq_dispatcher_sta.instantiate(
            em,
            stp_addr_i,
            stp_addr_valid_i,
            stp_addr_ready_o,
            stq_alloc,
            stq_addr_valid,
            stq_port_idx,
            stq_addr,
            stq_addr_wen,
            stq_head_oh,
        )

        # Store Data Port Dispatcher
        lsq_submodules.ptq_dispatcher_std.instantiate(
            em,
            stp_data_i,
            stp_data_valid_i,
            stp_data_ready_o,
            stq_alloc,
            stq_data_valid,
            stq_port_idx,
            stq_data,
            stq_data_wen,
            stq_head_oh,
        )

        # Store Backward Port Dispatcher
        if self.configs.stResp:
            lsq_submodules.qtp_dispatcher_stb.instantiate(
                em,
                None,
                stp_exec_valid_o,
                stp_exec_ready_i,
                stq_alloc,
                stq_exec,
                stq_port_idx,
                None,
                stq_reset,
                stq_head_oh,
            )

        if self.configs.pipe0:
            ###### Dependency Check ######
            load_idx_oh = LogicVecArray(
                em,
                "load_idx_oh",
                "w",
                self.configs.numLdMem,
                self.configs.numLdqEntries,
            )
            load_en = LogicArray(em, "load_en", "w", self.configs.numLdMem)

            # Multiple store channels not yet implemented
            assert self.configs.numStMem == 1
            store_idx = LogicVec(em, "store_idx", "w", self.configs.stqAddrW)
            store_en = Logic(em, "store_en", "w")

            bypass_idx_oh_p0 = LogicVecArray(
                em,
                "bypass_idx_oh_p0",
                "r",
                self.configs.numLdqEntries,
                self.configs.numStqEntries,
            )
            bypass_idx_oh_p0.regInit()
            bypass_en = LogicArray(em, "bypass_en", "w", self.configs.numLdqEntries)

            # Matrix Generation
            ld_st_conflict = LogicVecArray(
                em,
                "ld_st_conflict",
                "w",
                self.configs.numLdqEntries,
                self.configs.numStqEntries,
            )
            can_bypass = LogicVecArray(
                em,
                "can_bypass",
                "w",
                self.configs.numLdqEntries,
                self.configs.numStqEntries,
            )
            can_bypass_p0 = LogicVecArray(
                em,
                "can_bypass_p0",
                "r",
                self.configs.numLdqEntries,
                self.configs.numStqEntries,
            )
            can_bypass_p0.regInit(init=[0] * self.configs.numLdqEntries)

            if self.configs.pipeComp:
                ldq_alloc_pcomp = LogicArray(
                    em, "ldq_alloc_pcomp", "r", self.configs.numLdqEntries
                )
                ldq_addr_valid_pcomp = LogicArray(
                    em, "ldq_addr_valid_pcomp", "r", self.configs.numLdqEntries
                )
                stq_alloc_pcomp = LogicArray(
                    em, "stq_alloc_pcomp", "r", self.configs.numStqEntries
                )
                stq_addr_valid_pcomp = LogicArray(
                    em, "stq_addr_valid_pcomp", "r", self.configs.numStqEntries
                )
                stq_data_valid_pcomp = LogicArray(
                    em, "stq_data_valid_pcomp", "r", self.configs.numStqEntries
                )
                addr_valid_pcomp = LogicVecArray(
                    em,
                    "addr_valid_pcomp",
                    "w",
                    self.configs.numLdqEntries,
                    self.configs.numStqEntries,
                )
                addr_same_pcomp = LogicVecArray(
                    em,
                    "addr_same_pcomp",
                    "r",
                    self.configs.numLdqEntries,
                    self.configs.numStqEntries,
                )
                store_is_older_pcomp = LogicVecArray(
                    em,
                    "store_is_older_pcomp",
                    "r",
                    self.configs.numLdqEntries,
                    self.configs.numStqEntries,
                )

                ldq_alloc_pcomp.regInit(init=[0] * self.configs.numLdqEntries)
                ldq_addr_valid_pcomp.regInit()
                stq_alloc_pcomp.regInit(init=[0] * self.configs.numStqEntries)
                stq_addr_valid_pcomp.regInit()
                stq_data_valid_pcomp.regInit()
                addr_same_pcomp.regInit()
                store_is_older_pcomp.regInit()

                for i in range(0, self.configs.numLdqEntries):
                    em.add_assignment((ldq_alloc_pcomp, i), Val(ldq_alloc, i))
                    em.add_assignment((ldq_addr_valid_pcomp, i), Val(ldq_addr_valid, i))
                for j in range(0, self.configs.numStqEntries):
                    em.add_assignment((stq_alloc_pcomp, j), Val(stq_alloc, j))
                    em.add_assignment((stq_addr_valid_pcomp, j), Val(stq_addr_valid, j))
                    em.add_assignment((stq_data_valid_pcomp, j), Val(stq_data_valid, j))
                for i in range(0, self.configs.numLdqEntries):
                    for j in range(0, self.configs.numStqEntries):
                        em.add_assignment(
                            (store_is_older_pcomp, i, j), Val(store_is_older, i, j)
                        )
                for i in range(0, self.configs.numLdqEntries):
                    for j in range(0, self.configs.numStqEntries):
                        em.add_assignment(
                            (addr_valid_pcomp, i, j),
                            Val(ldq_addr_valid_pcomp, i) & Val(stq_addr_valid_pcomp, j),
                        )
                for i in range(0, self.configs.numLdqEntries):
                    for j in range(0, self.configs.numStqEntries):
                        em.add_assignment(
                            (addr_same_pcomp, i, j),
                            Bit(1)
                            .when(Val(ldq_addr, i) == Val(stq_addr, j))
                            .else_(Bit(0)),
                        )

                # A load conflicts with a store when:
                # 1. The store entry is valid, and
                # 2. The store is older than the load, and
                # 3. The address conflicts(same or invalid store address).
                for i in range(0, self.configs.numLdqEntries):
                    for j in range(0, self.configs.numStqEntries):
                        em.add_assignment(
                            (ld_st_conflict, i, j),
                            Val(stq_alloc_pcomp, j)
                            & Val(store_is_older_pcomp, i, j)
                            & (
                                Val(addr_same_pcomp, i, j)
                                | ~Val(stq_addr_valid_pcomp, j)
                            ),
                        )

                # A conflicting store entry can be bypassed to a load entry when:
                # 1. The load entry is valid, and
                # 2. The load entry is not issued yet, and
                # 3. The address of the load-store pair are both valid and values the same.
                for i in range(0, self.configs.numLdqEntries):
                    for j in range(0, self.configs.numStqEntries):
                        em.add_assignment(
                            (can_bypass_p0, i, j),
                            Val(ldq_alloc_pcomp, i)
                            & Val(stq_data_valid_pcomp, j)
                            & Val(addr_same_pcomp, i, j)
                            & Val(addr_valid_pcomp, i, j),
                        )
                for i in range(0, self.configs.numLdqEntries):
                    for j in range(0, self.configs.numStqEntries):
                        em.add_assignment(
                            (can_bypass, i, j),
                            ~Val(ldq_issue, i) & Val(can_bypass_p0, i, j),
                        )

                # Load

                load_conflict = LogicArray(
                    em, "load_conflict", "w", self.configs.numLdqEntries
                )
                load_req_valid = LogicArray(
                    em, "load_req_valid", "w", self.configs.numLdqEntries
                )
                can_load = LogicArray(em, "can_load", "w", self.configs.numLdqEntries)
                can_load_p0 = LogicArray(
                    em, "can_load_p0", "r", self.configs.numLdqEntries
                )
                can_load_p0.regInit(init=[0] * self.configs.numLdqEntries)

                # The load conflicts with any store
                for i in range(0, self.configs.numLdqEntries):
                    Reduce(em, load_conflict[i], ld_st_conflict[i], BinOp.OR)
                # The load is valid when the entry is valid and not yet issued, the load address should also be valid.
                # We do not need to check ldq_data_valid, since unissued load request cannot have valid data.
                for i in range(0, self.configs.numLdqEntries):
                    em.add_assignment(
                        load_req_valid[i], ldq_alloc_pcomp[i] & ldq_addr_valid_pcomp[i]
                    )
                # Generate list for loads that does not face dependency issue
                for i in range(0, self.configs.numLdqEntries):
                    em.add_assignment(
                        can_load_p0[i], ~load_conflict[i] & load_req_valid[i]
                    )
                for i in range(0, self.configs.numLdqEntries):
                    em.add_assignment(can_load[i], ~ldq_issue[i] & can_load_p0[i])

                ldq_head_oh_p0 = LogicVec(
                    em, "ldq_head_oh_p0", "r", self.configs.numLdqEntries
                )
                ldq_head_oh_p0.regInit()
                em.add_assignment(ldq_head_oh_p0, ldq_head_oh)

                can_load_list = []
                can_load_list.append(can_load)
                for w in range(0, self.configs.numLdMem):
                    CyclicPriorityMasking(
                        em, load_idx_oh[w], can_load_list[w], ldq_head_oh_p0
                    )
                    Reduce(em, load_en[w], can_load_list[w], BinOp.OR)
                    if w + 1 != self.configs.numLdMem:
                        load_idx_oh_LogicArray = LogicArray(
                            em,
                            f"load_idx_oh_Array_{w+1}",
                            "w",
                            self.configs.numLdqEntries,
                        )
                        VecToArray(em, load_idx_oh_LogicArray, load_idx_oh[w])
                        can_load_list.append(
                            LogicArray(
                                em,
                                f"can_load_list_{w+1}",
                                "w",
                                self.configs.numLdqEntries,
                            )
                        )
                        for i in range(0, self.configs.numLdqEntries):
                            em.add_assignment(
                                can_load_list[w + 1][i],
                                ~load_idx_oh_LogicArray[i] & can_load_list[w][i],
                            )

                # Store
                stq_issue_en_p0 = Logic(em, "stq_issue_en_p0", "r")
                stq_issue_next = LogicVec(
                    em, "stq_issue_next", "w", self.configs.stqAddrW
                )

                store_conflict = Logic(em, "store_conflict", "w")

                can_store_curr = Logic(em, "can_store_curr", "w")
                st_ld_conflict_curr = LogicVec(
                    em, "st_ld_conflict_curr", "w", self.configs.numLdqEntries
                )
                store_valid_curr = Logic(em, "store_valid_curr", "w")
                store_data_valid_curr = Logic(em, "store_data_valid_curr", "w")
                store_addr_valid_curr = Logic(em, "store_addr_valid_curr", "w")

                can_store_next = Logic(em, "can_store_next", "w")
                st_ld_conflict_next = LogicVec(
                    em, "st_ld_conflict_next", "w", self.configs.numLdqEntries
                )
                store_valid_next = Logic(em, "store_valid_next", "w")
                store_data_valid_next = Logic(em, "store_data_valid_next", "w")
                store_addr_valid_next = Logic(em, "store_addr_valid_next", "w")

                can_store_p0 = Logic(em, "can_store_p0", "r")
                st_ld_conflict_p0 = LogicVec(
                    em, "st_ld_conflict_p0", "r", self.configs.numLdqEntries
                )

                stq_issue_en_p0.regInit(init=0)
                can_store_p0.regInit(init=0)
                st_ld_conflict_p0.regInit()

                em.add_assignment(stq_issue_en_p0, stq_issue_en)
                WrapAddConst(
                    em, stq_issue_next, stq_issue, 1, self.configs.numStqEntries
                )

                # A store conflicts with a load when:
                # 1. The load entry is valid, and
                # 2. The load is older than the store, and
                # 3. The address conflicts(same or invalid store address).
                # Index order are reversed for store matrix.
                for i in range(0, self.configs.numLdqEntries):
                    em.add_assignment(
                        (st_ld_conflict_curr, i),
                        Val(ldq_alloc_pcomp, i)
                        & ~Val(em.mux_index(store_is_older_pcomp[i], stq_issue))
                        & (
                            Val(em.mux_index(addr_same_pcomp[i], stq_issue))
                            | ~Val(ldq_addr_valid_pcomp, i)
                        ),
                    )
                for i in range(0, self.configs.numLdqEntries):
                    em.add_assignment(
                        (st_ld_conflict_next, i),
                        Val(ldq_alloc_pcomp, i)
                        & ~Val(em.mux_index(store_is_older_pcomp[i], stq_issue_next))
                        & (
                            Val(em.mux_index(addr_same_pcomp[i], stq_issue_next))
                            | ~Val(ldq_addr_valid_pcomp, i)
                        ),
                    )
                # The store is valid whe the entry is valid and the data is also valid,
                # the store address should also be valid
                MuxLookUp(em, store_valid_curr, stq_alloc_pcomp, stq_issue)
                MuxLookUp(em, store_data_valid_curr, stq_data_valid_pcomp, stq_issue)
                MuxLookUp(em, store_addr_valid_curr, stq_addr_valid_pcomp, stq_issue)
                em.add_assignment(
                    can_store_curr,
                    store_valid_curr & store_data_valid_curr & store_addr_valid_curr,
                )
                MuxLookUp(em, store_valid_next, stq_alloc_pcomp, stq_issue_next)
                MuxLookUp(
                    em, store_data_valid_next, stq_data_valid_pcomp, stq_issue_next
                )
                MuxLookUp(
                    em, store_addr_valid_next, stq_addr_valid_pcomp, stq_issue_next
                )
                em.add_assignment(
                    can_store_next,
                    store_valid_next & store_data_valid_next & store_addr_valid_next,
                )
                # Multiplex from current and next
                em.add_assignment(
                    st_ld_conflict_p0,
                    st_ld_conflict_next.when(stq_issue_en).else_(st_ld_conflict_curr),
                )
                em.add_assignment(
                    can_store_p0,
                    can_store_next.when(stq_issue_en).else_(can_store_curr),
                )
                # The store conflicts with any load
                Reduce(em, store_conflict, st_ld_conflict_p0, BinOp.OR)
                em.add_assignment(store_en, ~store_conflict & can_store_p0)

                em.add_assignment(store_idx, stq_issue)

                # Bypass
                stq_last_oh = LogicVec(
                    em, "stq_last_oh", "w", self.configs.numStqEntries
                )
                BitsToOHSub1(em, stq_last_oh, stq_tail)
                for i in range(0, self.configs.numLdqEntries):
                    bypass_en_vec = LogicVec(
                        em, f"bypass_en_vec_{i}", "w", self.configs.numStqEntries
                    )
                    # Search for the youngest store that is older than the load and conflicts
                    CyclicPriorityMasking(
                        em, bypass_idx_oh_p0[i], ld_st_conflict[i], stq_last_oh, True
                    )
                    # Check if the youngest conflict store can bypass with the load
                    em.add_assignment(
                        bypass_en_vec, bypass_idx_oh_p0[i] & can_bypass[i]
                    )
                    Reduce(em, bypass_en[i], bypass_en_vec, BinOp.OR)
            else:
                addr_valid = LogicVecArray(
                    em,
                    "addr_valid",
                    "w",
                    self.configs.numLdqEntries,
                    self.configs.numStqEntries,
                )
                addr_same = LogicVecArray(
                    em,
                    "addr_same",
                    "w",
                    self.configs.numLdqEntries,
                    self.configs.numStqEntries,
                )

                for i in range(0, self.configs.numLdqEntries):
                    for j in range(0, self.configs.numStqEntries):
                        em_add_assignment(
                            (addr_valid, i, j),
                            Val(ldq_addr_valid, i) & Val(stq_addr_valid, j),
                        )
                for i in range(0, self.configs.numLdqEntries):
                    for j in range(0, self.configs.numStqEntries):
                        em_add_assignment(
                            (addr_same, i, j),
                            Bit(1)
                            .when(Val(ldq_addr, i) == Val(stq_addr, j))
                            .else_(Bit(0)),
                        )

                # A load conflicts with a store when:
                # 1. The store entry is valid, and
                # 2. The store is older than the load, and
                # 3. The address conflicts(same or invalid store address).
                for i in range(0, self.configs.numLdqEntries):
                    for j in range(0, self.configs.numStqEntries):
                        em.add_assignment(
                            (ld_st_conflict, i, j),
                            Val(stq_alloc, j)
                            & Val(store_is_older, i, j)
                            & (Val(addr_same, i, j) | ~Val(stq_addr_valid, j)),
                        )

                # A conflicting store entry can be bypassed to a load entry when:
                # 1. The load entry is valid, and
                # 2. The load entry is not issued yet, and
                # 3. The address of the load-store pair are both valid and values the same.
                for i in range(0, self.configs.numLdqEntries):
                    for j in range(0, self.configs.numStqEntries):
                        em.add_assignment(
                            (can_bypass_p0, i, j),
                            Val(ldq_alloc, i)
                            & Val(stq_data_valid, j)
                            & Val(addr_same, i, j)
                            & Val(addr_valid, i, j),
                        )
                for i in range(0, self.configs.numLdqEntries):
                    for j in range(0, self.configs.numStqEntries):
                        em.add_assignment(
                            (can_bypass, i, j),
                            ~Val(ldq_issue, i) & Val(can_bypass_p0, i, j),
                        )

                # Load

                load_conflict = LogicArray(
                    em, "load_conflict", "w", self.configs.numLdqEntries
                )
                load_req_valid = LogicArray(
                    em, "load_req_valid", "w", self.configs.numLdqEntries
                )
                can_load = LogicArray(em, "can_load", "w", self.configs.numLdqEntries)
                can_load_p0 = LogicArray(
                    em, "can_load_p0", "r", self.configs.numLdqEntries
                )
                can_load_p0.regInit(init=[0] * self.configs.numLdqEntries)

                # The load conflicts with any store
                for i in range(0, self.configs.numLdqEntries):
                    Reduce(em, load_conflict[i], ld_st_conflict[i], BinOp.OR)
                # The load is valid when the entry is valid and not yet issued, the load address should also be valid.
                # We do not need to check ldq_data_valid, since unissued load request cannot have valid data.
                for i in range(0, self.configs.numLdqEntries):
                    em.add_assignment(
                        load_req_valid[i], ldq_alloc[i] & ldq_addr_valid[i]
                    )
                # Generate list for loads that does not face dependency issue
                for i in range(0, self.configs.numLdqEntries):
                    em.add_assignment(
                        can_load_p0[i], ~load_conflict[i] & load_req_valid[i]
                    )
                for i in range(0, self.configs.numLdqEntries):
                    em.add_assignment(can_load[i], ~ldq_issue[i] & can_load_p0[i])
                ldq_head_oh_p0 = LogicVec(
                    em, "ldq_head_oh_p0", "r", self.configs.numLdqEntries
                )
                ldq_head_oh_p0.regInit()
                em.add_assignment(ldq_head_oh_p0, ldq_head_oh)

                can_load_list = []
                can_load_list.append(can_load)
                for w in range(0, self.configs.numLdMem):
                    CyclicPriorityMasking(
                        em, load_idx_oh[w], can_load_list[w], ldq_head_oh_p0
                    )
                    Reduce(em, load_en[w], can_load_list[w], BinOp.OR)
                    if w + 1 != self.configs.numLdMem:
                        load_idx_oh_LogicArray = LogicArray(
                            em,
                            f"load_idx_oh_Array_{w+1}",
                            "w",
                            self.configs.numLdqEntries,
                        )
                        arch += VecToArray(em, load_idx_oh_LogicArray, load_idx_oh[w])
                        can_load_list.append(
                            LogicArray(
                                em,
                                f"can_load_list_{w+1}",
                                "w",
                                self.configs.numLdqEntries,
                            )
                        )
                        for i in range(0, self.configs.numLdqEntries):
                            em.add_assignment(
                                can_load_list[w + 1][i],
                                ~load_idx_oh_LogicArray[i] & can_load_list[w][i],
                            )

                # Store
                stq_issue_en_p0 = Logic(em, "stq_issue_en_p0", "r")
                stq_issue_next = LogicVec(
                    em, "stq_issue_next", "w", self.configs.stqAddrW
                )

                store_conflict = Logic(em, "store_conflict", "w")

                can_store_curr = Logic(em, "can_store_curr", "w")
                st_ld_conflict_curr = LogicVec(
                    em, "st_ld_conflict_curr", "w", self.configs.numLdqEntries
                )
                store_valid_curr = Logic(em, "store_valid_curr", "w")
                store_data_valid_curr = Logic(em, "store_data_valid_curr", "w")
                store_addr_valid_curr = Logic(em, "store_addr_valid_curr", "w")

                can_store_next = Logic(em, "can_store_next", "w")
                st_ld_conflict_next = LogicVec(
                    em, "st_ld_conflict_next", "w", self.configs.numLdqEntries
                )
                store_valid_next = Logic(em, "store_valid_next", "w")
                store_data_valid_next = Logic(em, "store_data_valid_next", "w")
                store_addr_valid_next = Logic(em, "store_addr_valid_next", "w")

                can_store_p0 = Logic(em, "can_store_p0", "r")
                st_ld_conflict_p0 = LogicVec(
                    em, "st_ld_conflict_p0", "r", self.configs.numLdqEntries
                )

                stq_issue_en_p0.regInit(init=0)
                can_store_p0.regInit(init=0)
                st_ld_conflict_p0.regInit()

                em.add_assignment(stq_issue_en_p0, stq_issue_en)
                WrapAddConst(
                    em, stq_issue_next, stq_issue, 1, self.configs.numStqEntries
                )

                # A store conflicts with a load when:
                # 1. The load entry is valid, and
                # 2. The load is older than the store, and
                # 3. The address conflicts(same or invalid store address).
                # Index order are reversed for store matrix.
                for i in range(0, self.configs.numLdqEntries):
                    em.add_assignment(
                        (st_ld_conflict_curr, i),
                        Val(ldq_alloc, i)
                        & ~Val(em.mux_index(store_is_older[i], stq_issue))
                        & (
                            Val(em.mux_index(addr_same[i], stq_issue))
                            | ~Val(ldq_addr_valid, i)
                        ),
                    )
                for i in range(0, self.configs.numLdqEntries):
                    em.add_assignment(
                        (st_ld_conflict_next, i),
                        Val(ldq_alloc, i)
                        & ~Val(em.mux_index(store_is_older[i], stq_issue_next))
                        & (
                            Val(em.mux_index(addr_same[i], stq_issue_next))
                            | ~Val(ldq_addr_valid, i)
                        ),
                    )
                # The store is valid whe the entry is valid and the data is also valid,
                # the store address should also be valid
                MuxLookUp(em, store_valid_curr, stq_alloc, stq_issue)
                MuxLookUp(em, store_data_valid_curr, stq_data_valid, stq_issue)
                MuxLookUp(em, store_addr_valid_curr, stq_addr_valid, stq_issue)
                em.add_assignment(
                    can_store_curr,
                    store_valid_curr & store_data_valid_curr & store_addr_valid_curr,
                )
                MuxLookUp(em, store_valid_next, stq_alloc, stq_issue_next)
                MuxLookUp(em, store_data_valid_next, stq_data_valid, stq_issue_next)
                MuxLookUp(em, store_addr_valid_next, stq_addr_valid, stq_issue_next)
                em.add_assignment(
                    can_store_next,
                    store_valid_next & store_data_valid_next & store_addr_valid_next,
                )
                # Multiplex from current and next
                em.add_assignment(
                    st_ld_conflict_p0,
                    st_ld_conflict_next.when(stq_issue_en).else_(st_ld_conflict_curr),
                )
                em.add_assignment(
                    can_store_p0,
                    can_store_next.when(stq_issue_en).else_(can_store_curr),
                )
                # The store conflicts with any load
                Reduce(em, store_conflict, st_ld_conflict_p0, BinOp.OR)
                em.add_assignment(store_en, ~store_conflict & can_store_p0)

                em.add_assignment(store_idx, stq_issue)

                # Bypass
                stq_last_oh = LogicVec(
                    em, "stq_last_oh", "w", self.configs.numStqEntries
                )
                BitsToOHSub1(em, stq_last_oh, stq_tail)
                for i in range(0, self.configs.numLdqEntries):
                    bypass_en_vec = LogicVec(
                        em, f"bypass_en_vec_{i}", "w", self.configs.numStqEntries
                    )
                    # Search for the youngest store that is older than the load and conflicts
                    CyclicPriorityMasking(
                        em, bypass_idx_oh_p0[i], ld_st_conflict[i], stq_last_oh, True
                    )
                    # Check if the youngest conflict store can bypass with the load
                    em.add_assignment(
                        bypass_en_vec, bypass_idx_oh_p0[i] & can_bypass[i]
                    )
                    Reduce(em, bypass_en[i], bypass_en_vec, BinOp.OR)
        else:
            ###### Dependency Check ######

            load_idx_oh = LogicVecArray(
                em,
                "load_idx_oh",
                "w",
                self.configs.numLdMem,
                self.configs.numLdqEntries,
            )
            load_en = LogicArray(em, "load_en", "w", self.configs.numLdMem)

            # Multiple store channels not yet implemented
            assert self.configs.numStMem == 1
            store_idx = LogicVec(em, "store_idx", "w", self.configs.stqAddrW)
            store_en = Logic(em, "store_en", "w")

            bypass_idx_oh = LogicVecArray(
                em,
                "bypass_idx_oh",
                "w",
                self.configs.numLdqEntries,
                self.configs.numStqEntries,
            )
            bypass_en = LogicArray(em, "bypass_en", "w", self.configs.numLdqEntries)

            # Matrix Generation
            ld_st_conflict = LogicVecArray(
                em,
                "ld_st_conflict",
                "w",
                self.configs.numLdqEntries,
                self.configs.numStqEntries,
            )
            can_bypass = LogicVecArray(
                em,
                "can_bypass",
                "w",
                self.configs.numLdqEntries,
                self.configs.numStqEntries,
            )

            if self.configs.pipeComp:
                ldq_alloc_pcomp = LogicArray(
                    em, "ldq_alloc_pcomp", "r", self.configs.numLdqEntries
                )
                ldq_addr_valid_pcomp = LogicArray(
                    em, "ldq_addr_valid_pcomp", "r", self.configs.numLdqEntries
                )
                stq_alloc_pcomp = LogicArray(
                    em, "stq_alloc_pcomp", "r", self.configs.numStqEntries
                )
                stq_addr_valid_pcomp = LogicArray(
                    em, "stq_addr_valid_pcomp", "r", self.configs.numStqEntries
                )
                stq_data_valid_pcomp = LogicArray(
                    em, "stq_data_valid_pcomp", "r", self.configs.numStqEntries
                )
                addr_valid_pcomp = LogicVecArray(
                    em,
                    "addr_valid_pcomp",
                    "w",
                    self.configs.numLdqEntries,
                    self.configs.numStqEntries,
                )
                addr_same_pcomp = LogicVecArray(
                    em,
                    "addr_same_pcomp",
                    "r",
                    self.configs.numLdqEntries,
                    self.configs.numStqEntries,
                )
                store_is_older_pcomp = LogicVecArray(
                    em,
                    "store_is_older_pcomp",
                    "r",
                    self.configs.numLdqEntries,
                    self.configs.numStqEntries,
                )

                ldq_alloc_pcomp.regInit(init=[0] * self.configs.numLdqEntries)
                ldq_addr_valid_pcomp.regInit()
                stq_alloc_pcomp.regInit(init=[0] * self.configs.numStqEntries)
                stq_addr_valid_pcomp.regInit()
                stq_data_valid_pcomp.regInit()
                addr_same_pcomp.regInit()
                store_is_older_pcomp.regInit()

                for i in range(0, self.configs.numLdqEntries):
                    em.add_assignment((ldq_alloc_pcomp, i), Val(ldq_alloc, i))
                    em.add_assignment((ldq_addr_valid_pcomp, i), Val(ldq_addr_valid, i))
                for j in range(0, self.configs.numStqEntries):
                    em.add_assignment((stq_alloc_pcomp, j), Val(stq_alloc, j))
                    em.add_assignment((stq_addr_valid_pcomp, j), Val(stq_addr_valid, j))
                    em.add_assignment((stq_data_valid_pcomp, j), Val(stq_data_valid, j))
                for i in range(0, self.configs.numLdqEntries):
                    for j in range(0, self.configs.numStqEntries):
                        em.add_assignment(
                            (store_is_older_pcomp, i, j), Val(store_is_older, i, j)
                        )
                for i in range(0, self.configs.numLdqEntries):
                    for j in range(0, self.configs.numStqEntries):
                        em.add_assignment(
                            (addr_valid_pcomp, i, j),
                            Val(ldq_addr_valid_pcomp, i) & Val(stq_addr_valid_pcomp, j),
                        )
                for i in range(0, self.configs.numLdqEntries):
                    for j in range(0, self.configs.numStqEntries):
                        em.add_assignment(
                            (addr_same_pcomp, i, j),
                            Bit(1)
                            .when(Val(ldq_addr, i) == Val(stq_addr, j))
                            .else_(Bit(0)),
                        )

                # A load conflicts with a store when:
                # 1. The store entry is valid, and
                # 2. The store is older than the load, and
                # 3. The address conflicts(same or invalid store address).
                for i in range(0, self.configs.numLdqEntries):
                    for j in range(0, self.configs.numStqEntries):
                        em.add_assignment(
                            (ld_st_conflict, i, j),
                            Val(stq_alloc_pcomp, j)
                            & Val(store_is_older_pcomp, i, j)
                            & (
                                Val(addr_same_pcomp, i, j)
                                | ~Val(stq_addr_valid_pcomp, j)
                            ),
                        )

                # A conflicting store entry can be bypassed to a load entry when:
                # 1. The load entry is valid, and
                # 2. The load entry is not issued yet, and
                # 3. The address of the load-store pair are both valid and values the same.
                for i in range(0, self.configs.numLdqEntries):
                    for j in range(0, self.configs.numStqEntries):
                        em.add_assignment(
                            (can_bypass, i, j),
                            Val(ldq_alloc_pcomp, i)
                            & ~Val(ldq_issue, i)
                            & Val(stq_data_valid_pcomp, j)
                            & Val(addr_same_pcomp, i, j)
                            & Val(addr_valid_pcomp, i, j),
                        )

                # Load

                load_conflict = LogicArray(
                    em, "load_conflict", "w", self.configs.numLdqEntries
                )
                load_req_valid = LogicArray(
                    em, "load_req_valid", "w", self.configs.numLdqEntries
                )
                can_load = LogicArray(em, "can_load", "w", self.configs.numLdqEntries)

                # The load conflicts with any store
                for i in range(0, self.configs.numLdqEntries):
                    Reduce(em, load_conflict[i], ld_st_conflict[i], BinOp.OR)
                # The load is valid when the entry is valid and not yet issued, the load address should also be valid.
                # We do not need to check ldq_data_valid, since unissued load request cannot have valid data.
                for i in range(0, self.configs.numLdqEntries):
                    em.add_assignment(
                        load_req_valid[i],
                        ldq_alloc_pcomp[i] & ~ldq_issue[i] & ldq_addr_valid_pcomp[i],
                    )
                # Generate list for loads that does not face dependency issue
                for i in range(0, self.configs.numLdqEntries):
                    em.add_assignment(
                        can_load[i], ~load_conflict[i] & load_req_valid[i]
                    )

                can_load_list = []
                can_load_list.append(can_load)
                for w in range(0, self.configs.numLdMem):
                    CyclicPriorityMasking(
                        em, load_idx_oh[w], can_load_list[w], ldq_head_oh
                    )
                    Reduce(em, load_en[w], can_load_list[w], BinOp.OR)
                    if w + 1 != self.configs.numLdMem:
                        load_idx_oh_LogicArray = LogicArray(
                            em,
                            f"load_idx_oh_Array_{w+1}",
                            "w",
                            self.configs.numLdqEntries,
                        )
                        VecToArray(em, load_idx_oh_LogicArray, load_idx_oh[w])
                        can_load_list.append(
                            LogicArray(
                                em,
                                f"can_load_list_{w+1}",
                                "w",
                                self.configs.numLdqEntries,
                            )
                        )
                        for i in range(0, self.configs.numLdqEntries):
                            em.add_assignment(
                                can_load_list[w + 1][i],
                                ~load_idx_oh_LogicArray[i] & can_load_list[w][i],
                            )

                # Store

                st_ld_conflict = LogicVec(
                    em, "st_ld_conflict", "w", self.configs.numLdqEntries
                )
                store_conflict = Logic(em, "store_conflict", "w")
                store_valid = Logic(em, "store_valid", "w")
                store_data_valid = Logic(em, "store_data_valid", "w")
                store_addr_valid = Logic(em, "store_addr_valid", "w")

                # A store conflicts with a load when:
                # 1. The load entry is valid, and
                # 2. The load is older than the store, and
                # 3. The address conflicts(same or invalid store address).
                # Index order are reversed for store matrix.
                for i in range(0, self.configs.numLdqEntries):
                    em.add_assignment(
                        (st_ld_conflict, i),
                        Val(ldq_alloc_pcomp, i)
                        & ~Val(em.mux_index(store_is_older_pcomp[i], stq_issue))
                        & (
                            Val(em.mux_index(addr_same_pcomp[i], stq_issue))
                            | ~Val(ldq_addr_valid_pcomp, i)
                        ),
                    )
                # The store conflicts with any load
                Reduce(em, store_conflict, st_ld_conflict, BinOp.OR)
                # The store is valid whe the entry is valid and the data is also valid,
                # the store address should also be valid
                MuxLookUp(em, store_valid, stq_alloc_pcomp, stq_issue)
                MuxLookUp(em, store_data_valid, stq_data_valid_pcomp, stq_issue)
                MuxLookUp(em, store_addr_valid, stq_addr_valid_pcomp, stq_issue)
                em.add_assignment(
                    store_en,
                    ~store_conflict & store_valid & store_data_valid & store_addr_valid,
                )
                em.add_assignment(store_idx, stq_issue)

                stq_last_oh = LogicVec(
                    em, "stq_last_oh", "w", self.configs.numStqEntries
                )
                BitsToOHSub1(em, stq_last_oh, stq_tail)
                for i in range(0, self.configs.numLdqEntries):
                    bypass_en_vec = LogicVec(
                        em, f"bypass_en_vec_{i}", "w", self.configs.numStqEntries
                    )
                    # Search for the youngest store that is older than the load and conflicts
                    CyclicPriorityMasking(
                        em, bypass_idx_oh[i], ld_st_conflict[i], stq_last_oh, True
                    )
                    # Check if the youngest conflict store can bypass with the load
                    em.add_assignment(bypass_en_vec, bypass_idx_oh[i] & can_bypass[i])
                    Reduce(em, bypass_en[i], bypass_en_vec, BinOp.OR)
            else:
                addr_valid = LogicVecArray(
                    em,
                    "addr_valid",
                    "w",
                    self.configs.numLdqEntries,
                    self.configs.numStqEntries,
                )
                addr_same = LogicVecArray(
                    em,
                    "addr_same",
                    "w",
                    self.configs.numLdqEntries,
                    self.configs.numStqEntries,
                )

                for i in range(0, self.configs.numLdqEntries):
                    for j in range(0, self.configs.numStqEntries):
                        em.add_assignment(
                            (addr_valid, i, j),
                            Val(ldq_addr_valid, i) & Val(stq_addr_valid, j),
                        )
                for i in range(0, self.configs.numLdqEntries):
                    for j in range(0, self.configs.numStqEntries):
                        em.add_assignment(
                            (addr_same, i, j),
                            Bit(1)
                            .when(Val(ldq_addr, i) == Val(stq_addr, j))
                            .else_(Bit(0)),
                        )

                # A load conflicts with a store when:
                # 1. The store entry is valid, and
                # 2. The store is older than the load, and
                # 3. The address conflicts(same or invalid store address).
                for i in range(0, self.configs.numLdqEntries):
                    for j in range(0, self.configs.numStqEntries):
                        em.add_assignment(
                            (ld_st_conflict, i, j),
                            Val(stq_alloc, j)
                            & Val(store_is_older, i, j)
                            & (Val(addr_same, i, j) | ~Val(stq_addr_valid, j)),
                        )

                # A conflicting store entry can be bypassed to a load entry when:
                # 1. The load entry is valid, and
                # 2. The load entry is not issued yet, and
                # 3. The address of the load-store pair are both valid and values the same.
                for i in range(0, self.configs.numLdqEntries):
                    for j in range(0, self.configs.numStqEntries):
                        em.add_assignment(
                            (can_bypass, i, j),
                            Val(ldq_alloc, i)
                            & ~Val(ldq_issue, i)
                            & Val(stq_data_valid, j)
                            & Val(addr_same, i, j)
                            & Val(addr_valid, i, j),
                        )

                # Load

                load_conflict = LogicArray(
                    em, "load_conflict", "w", self.configs.numLdqEntries
                )
                load_req_valid = LogicArray(
                    em, "load_req_valid", "w", self.configs.numLdqEntries
                )
                can_load = LogicArray(em, "can_load", "w", self.configs.numLdqEntries)

                # The load conflicts with any store
                for i in range(0, self.configs.numLdqEntries):
                    Reduce(em, load_conflict[i], ld_st_conflict[i], BinOp.OR)
                # The load is valid when the entry is valid and not yet issued, the load address should also be valid.
                # We do not need to check ldq_data_valid, since unissued load request cannot have valid data.
                for i in range(0, self.configs.numLdqEntries):
                    em.add_assignment(
                        load_req_valid[i],
                        ldq_alloc[i] & ~ldq_issue[i] & ldq_addr_valid[i],
                    )
                # Generate list for loads that does not face dependency issue
                for i in range(0, self.configs.numLdqEntries):
                    em.add_assignment(
                        can_load[i], ~load_conflict[i] & load_req_valid[i]
                    )

                can_load_list = []
                can_load_list.append(can_load)
                for w in range(0, self.configs.numLdMem):
                    CyclicPriorityMasking(
                        em, load_idx_oh[w], can_load_list[w], ldq_head_oh
                    )
                    Reduce(em, load_en[w], can_load_list[w], BinOp.OR)
                    if w + 1 != self.configs.numLdMem:
                        load_idx_oh_LogicArray = LogicArray(
                            em,
                            f"load_idx_oh_Array_{w+1}",
                            "w",
                            self.configs.numLdqEntries,
                        )
                        VecToArray(em, load_idx_oh_LogicArray, load_idx_oh[w])
                        can_load_list.append(
                            LogicArray(
                                em,
                                f"can_load_list_{w+1}",
                                "w",
                                self.configs.numLdqEntries,
                            )
                        )
                        for i in range(0, self.configs.numLdqEntries):
                            em.add_assignment(
                                can_load_list[w + 1][i],
                                ~load_idx_oh_LogicArray[i] & can_load_list[w][i],
                            )
                # Store

                st_ld_conflict = LogicVec(
                    em, "st_ld_conflict", "w", self.configs.numLdqEntries
                )
                store_conflict = Logic(em, "store_conflict", "w")
                store_valid = Logic(em, "store_valid", "w")
                store_data_valid = Logic(em, "store_data_valid", "w")
                store_addr_valid = Logic(em, "store_addr_valid", "w")

                # A store conflicts with a load when:
                # 1. The load entry is valid, and
                # 2. The load is older than the store, and
                # 3. The address conflicts(same or invalid store address).
                # Index order are reversed for store matrix.
                for i in range(0, self.configs.numLdqEntries):
                    em.add_assignment(
                        (st_ld_conflict, i),
                        Val(ldq_alloc, i)
                        & ~Val(em.mux_index(store_is_older[i], stq_issue))
                        & (
                            Val(em.mux_index(addr_same[i], stq_issue))
                            | ~Val(ldq_addr_valid, i)
                        ),
                    )
                # The store conflicts with any load
                Reduce(em, store_conflict, st_ld_conflict, BinOp.OR)
                # The store is valid whe the entry is valid and the data is also valid,
                # the store address should also be valid
                MuxLookUp(em, store_valid, stq_alloc, stq_issue)
                MuxLookUp(em, store_data_valid, stq_data_valid, stq_issue)
                MuxLookUp(em, store_addr_valid, stq_addr_valid, stq_issue)
                em.add_assignment(
                    store_en,
                    ~store_conflict & store_valid & store_data_valid & store_addr_valid,
                )
                em.add_assignment(store_idx, stq_issue)

                stq_last_oh = LogicVec(
                    em, "stq_last_oh", "w", self.configs.numStqEntries
                )
                BitsToOHSub1(em, stq_last_oh, stq_tail)
                for i in range(0, self.configs.numLdqEntries):
                    bypass_en_vec = LogicVec(
                        em, f"bypass_en_vec_{i}", "w", self.configs.numStqEntries
                    )
                    # Search for the youngest store that is older than the load and conflicts
                    CyclicPriorityMasking(
                        em, bypass_idx_oh[i], ld_st_conflict[i], stq_last_oh, True
                    )
                    # Check if the youngest conflict store can bypass with the load
                    em.add_assignment(bypass_en_vec, bypass_idx_oh[i] & can_bypass[i])
                    Reduce(em, bypass_en[i], bypass_en_vec, BinOp.OR)

        if self.configs.pipe1:
            # Pipeline Stage 1
            load_idx_oh_p1 = LogicVecArray(
                em,
                "load_idx_oh_p1",
                "r",
                self.configs.numLdMem,
                self.configs.numLdqEntries,
            )
            load_en_p1 = LogicArray(em, "load_en_p1", "r", self.configs.numLdMem)

            load_hs = LogicArray(em, "load_hs", "w", self.configs.numLdMem)
            load_p1_ready = LogicArray(em, "load_p1_ready", "w", self.configs.numLdMem)

            store_idx_p1 = LogicVec(em, "store_idx_p1", "r", self.configs.stqAddrW)
            store_en_p1 = Logic(em, "store_en_p1", "r")

            store_hs = Logic(em, "store_hs", "w")
            store_p1_ready = Logic(em, "store_p1_ready", "w")

            bypass_idx_oh_p1 = LogicVecArray(
                em,
                "bypass_idx_oh_p1",
                "r",
                self.configs.numLdqEntries,
                self.configs.numStqEntries,
            )
            bypass_en_p1 = LogicArray(
                em, "bypass_en_p1", "r", self.configs.numLdqEntries
            )

            load_idx_oh_p1.regInit(enable=load_p1_ready)
            load_en_p1.regInit(init=[0] * self.configs.numLdMem, enable=load_p1_ready)

            store_idx_p1.regInit(enable=store_p1_ready)
            store_en_p1.regInit(init=0, enable=store_p1_ready)

            bypass_idx_oh_p1.regInit()
            bypass_en_p1.regInit(init=[0] * self.configs.numLdqEntries)

            for w in range(0, self.configs.numLdMem):
                em.add_assignment(load_hs[w], load_en_p1[w] & rreq_ready_i[w])
                em.add_assignment(load_p1_ready[w], load_hs[w] | ~load_en_p1[w])

            for w in range(0, self.configs.numLdMem):
                em.add_assignment(load_idx_oh_p1[w], load_idx_oh[w])
                em.add_assignment(load_en_p1[w], load_en[w])

            em.add_assignment(store_hs, store_en_p1 & wreq_ready_i[0])
            em.add_assignment(store_p1_ready, store_hs | ~store_en_p1)

            em.add_assignment(store_idx_p1, store_idx)
            em.add_assignment(store_en_p1, store_en)

            if self.configs.pipe0:
                for i in range(0, self.configs.numLdqEntries):
                    em.add_assignment(bypass_idx_oh_p1[i], bypass_idx_oh_p0[i])
            else:
                for i in range(0, self.configs.numLdqEntries):
                    em.add_assignment(bypass_idx_oh_p1[i], bypass_idx_oh[i])

            for i in range(0, self.configs.numLdqEntries):
                em.add_assignment(bypass_en_p1[i], bypass_en[i])

            ######    Read/Write    ######
            # Read Request
            for w in range(0, self.configs.numLdMem):
                em.add_assignment(rreq_valid_o[w], load_en_p1[w])
                OHToBits(em, rreq_id_o[w], load_idx_oh_p1[w])
                Mux1H(em, rreq_addr_o[w], ldq_addr, load_idx_oh_p1[w])

            for i in range(0, self.configs.numLdqEntries):
                ldq_issue_set_vec = LogicVec(
                    em, f"ldq_issue_set_vec_{i}", "w", self.configs.numLdMem
                )
                for w in range(0, self.configs.numLdMem):
                    em.add_assignment(
                        (ldq_issue_set_vec, w),
                        (Val(load_idx_oh, w, i) & Val(load_p1_ready, w))
                        | Val(bypass_en, i),
                    )
                Reduce(em, ldq_issue_set[i], ldq_issue_set_vec, BinOp.OR)

            # Write Request
            em.add_assignment(wreq_valid_o[0], store_en_p1)
            em.add_assignment(wreq_id_o[0], 0)
            MuxLookUp(em, wreq_addr_o[0], stq_addr, store_idx_p1)
            MuxLookUp(em, wreq_data_o[0], stq_data, store_idx_p1)
            em.add_assignment(stq_issue_en, store_en & store_p1_ready)

            # Read Response and Bypass
            for i in range(0, self.configs.numLdqEntries):
                # check each read response channel for each load
                read_idx_oh = LogicArray(
                    em, f"read_idx_oh_{i}", "w", self.configs.numLdMem
                )
                read_valid = Logic(em, f"read_valid_{i}", "w")
                read_data = LogicVec(em, f"read_data_{i}", "w", self.configs.dataW)
                for w in range(0, self.configs.numLdMem):
                    em.add_assignment(
                        read_idx_oh[w],
                        rresp_valid_i[w].when(
                            (rresp_id_i[w] == Val(i, self.configs.idW)).else_(Bit(0))
                        ),
                    )
                Mux1H(em, read_data, rresp_data_i, read_idx_oh)
                Reduce(em, read_valid, read_idx_oh, BinOp.OR)
                # multiplex from store queue data
                bypass_data = LogicVec(em, f"bypass_data_{i}", "w", self.configs.dataW)
                Mux1H(em, bypass_data, stq_data, bypass_idx_oh_p1[i])
                # multiplex from read and bypass data
                em.add_assignment(ldq_data[i], read_data | bypass_data)
                em.add_assignment(ldq_data_wen[i], bypass_en_p1[i] | read_valid)
            for w in range(0, self.configs.numLdMem):
                em.add_assignment(rresp_ready_o[w], Bit(1))

            # Write Response
            if self.configs.stResp:
                for i in range(0, self.configs.numStqEntries):
                    em.add_assignment(
                        stq_exec_set[i],
                        wresp_valid_i[0]
                        .when((stq_resp == Val(i, self.configs.stqAddrW)))
                        .else_(Bit(0)),
                    )
            else:
                for i in range(0, self.configs.numStqEntries):
                    em.add_assignment(
                        stq_reset[i],
                        wresp_valid_i[0]
                        .when((stq_resp == Val(i, self.configs.stqAddrW)))
                        .else_(Bit(0)),
                    )

            em.add_assignment(stq_resp_en, wresp_valid_i[0])
            em.add_assignment(wresp_ready_o[0], Bit(1))
        else:
            ######    Read/Write    ######
            # Read Request
            for w in range(0, self.configs.numLdMem):
                em.add_assignment(rreq_valid_o[w], load_en[w])
                OHToBits(em, rreq_id_o[w], load_idx_oh[w])
                Mux1H(em, rreq_addr_o[w], ldq_addr, load_idx_oh[w])

            for i in range(0, self.configs.numLdqEntries):
                ldq_issue_set_vec = LogicVec(
                    em, f"ldq_issue_set_vec_{i}", "w", self.configs.numLdMem
                )
                for w in range(0, self.configs.numLdMem):
                    em.add_assignment(
                        (ldq_issue_set_vec, w),
                        (
                            Val(load_idx_oh, w, i)
                            & Val(rreq_ready_i, w)
                            & Val(load_en, w)
                        )
                        | Val(bypass_en, i),
                    )
                Reduce(em, ldq_issue_set[i], ldq_issue_set_vec, BinOp.OR)

            # Write Request
            em.add_assignment(wreq_valid_o[0], store_en)
            em.add_assignment(wreq_id_o[0], Val(0))
            MuxLookUp(em, wreq_addr_o[0], stq_addr, store_idx)
            MuxLookUp(em, wreq_data_o[0], stq_data, store_idx)
            em.add_assignment(stq_issue_en, store_en & wreq_ready_i[0])

            # Read Response and Bypass
            for i in range(0, self.configs.numLdqEntries):
                # check each read response channel for each load
                read_idx_oh = LogicArray(
                    em, f"read_idx_oh_{i}", "w", self.configs.numLdMem
                )
                read_valid = Logic(em, f"read_valid_{i}", "w")
                read_data = LogicVec(em, f"read_data_{i}", "w", self.configs.dataW)
                for w in range(0, self.configs.numLdMem):
                    em.add_assignment(
                        read_idx_oh[w],
                        rresp_valid_i[w]
                        .when(rresp_id_i[w] == Val(i, self.configs.idW))
                        .else_(Bit(0)),
                    )
                Mux1H(em, read_data, rresp_data_i, read_idx_oh)
                Reduce(em, read_valid, read_idx_oh, BinOp.OR)
                # multiplex from store queue data
                bypass_data = LogicVec(em, f"bypass_data_{i}", "w", self.configs.dataW)
                if self.configs.pipe0:
                    Mux1H(em, bypass_data, stq_data, bypass_idx_oh_p0[i])
                else:
                    Mux1H(em, bypass_data, stq_data, bypass_idx_oh[i])
                # multiplex from read and bypass data
                em.add_assignment(ldq_data[i], read_data | bypass_data)
                em.add_assignment(ldq_data_wen[i], bypass_en[i] | read_valid)
            for w in range(0, self.configs.numLdMem):
                em.add_assignment(rresp_ready_o[w], Bit(1))

            # Write Response
            if self.configs.stResp:
                for i in range(0, self.configs.numStqEntries):
                    em.add_assignment(
                        stq_exec_set[i],
                        wresp_valid_i[0]
                        .when(stq_resp == Val(i, self.configs.stqAddrW))
                        .else_(Bit(0)),
                    )
            else:
                for i in range(0, self.configs.numStqEntries):
                    em.add_assignment(
                        stq_reset[i],
                        wresp_valid_i[0]
                        .when(stq_resp == Val(i, self.configs.stqAddrW))
                        .else_(Bit(0)),
                    )
            em.add_assignment(stq_resp_en, wresp_valid_i[0])
            em.add_assignment(wresp_ready_o[0], Bit(1))

        # Write to the file
        output_str = em.get_definition_str(self.module_name)
        with open(f"{path_rtl}/{self.name}.{em.get_file_suffix()}", "a") as file:
            file.write(output_str)

    def instantiate(self, **kwargs) -> str:
        """
        *Instantiation of LSQ is in lsq-generator.py.
        """
        pass
