from core_gen.signals import Logic, LogicArray, LogicVec, LogicVecArray
from core_gen.ir import Val, WhenElse, Bit
from core_gen.operators import WrapSub, Mux1HROM, CyclicLeftShift, CyclicPriorityMasking
from core_gen.configs import Configs
from core_gen.emitters import Emitter


class GroupAllocator:
    def __init__(self, name: str, suffix: str, configs: Configs):
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

    def generate(self, em: Emitter, path_rtl: str) -> None:
        """
        Generates the VHDL 'entity' and 'architecture' sections for a group allocator.

        Parameters:
            em          : Emitter used for code generation
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
        # IOs
        group_init_valid_i = LogicArray(
            em, "group_init_valid", "i", self.configs.numGroups
        )
        group_init_ready_o = LogicArray(
            em, "group_init_ready", "o", self.configs.numGroups
        )

        ldq_tail_i = LogicVec(em, "ldq_tail", "i", self.configs.ldqAddrW)
        ldq_head_i = LogicVec(em, "ldq_head", "i", self.configs.ldqAddrW)
        ldq_empty_i = Logic(em, "ldq_empty", "i")

        stq_tail_i = LogicVec(em, "stq_tail", "i", self.configs.stqAddrW)
        stq_head_i = LogicVec(em, "stq_head", "i", self.configs.stqAddrW)
        stq_empty_i = Logic(em, "stq_empty", "i")

        ldq_wen_o = LogicArray(em, "ldq_wen", "o", self.configs.numLdqEntries)
        num_loads_o = LogicVec(em, "num_loads", "o", self.configs.ldqAddrW)
        num_loads = LogicVec(em, "num_loads", "w", self.configs.ldqAddrW)
        if self.configs.ldpAddrW > 0:
            ldq_port_idx_o = LogicVecArray(
                em,
                "ldq_port_idx",
                "o",
                self.configs.numLdqEntries,
                self.configs.ldpAddrW,
            )

        stq_wen_o = LogicArray(em, "stq_wen", "o", self.configs.numStqEntries)
        num_stores_o = LogicVec(em, "num_stores", "o", self.configs.stqAddrW)
        num_stores = LogicVec(em, "num_stores", "w", self.configs.stqAddrW)
        if self.configs.stpAddrW > 0:
            stq_port_idx_o = LogicVecArray(
                em,
                "stq_port_idx",
                "o",
                self.configs.numStqEntries,
                self.configs.stpAddrW,
            )

        ga_ls_order_o = LogicVecArray(
            em,
            "ga_ls_order",
            "o",
            self.configs.numLdqEntries,
            self.configs.numStqEntries,
        )

        # The number of empty load and store is calculated with cyclic subtraction.
        # If the empty signal is high, then set the number to max value.
        loads_sub = LogicVec(em, "loads_sub", "w", self.configs.ldqAddrW)
        stores_sub = LogicVec(em, "stores_sub", "w", self.configs.stqAddrW)
        empty_loads = LogicVec(em, "empty_loads", "w", self.configs.emptyLdAddrW)
        empty_stores = LogicVec(em, "empty_stores", "w", self.configs.emptyStAddrW)

        WrapSub(em, loads_sub, ldq_head_i, ldq_tail_i, self.configs.numLdqEntries)
        WrapSub(em, stores_sub, stq_head_i, stq_tail_i, self.configs.numStqEntries)

        em.add_assignment(
            empty_loads,
            Val(self.configs.numLdqEntries)
            .when(ldq_empty_i)
            .else_(Bit(0).concat(loads_sub)),
        )
        em.add_assignment(
            empty_stores,
            Val(self.configs.numStqEntries)
            .when(stq_empty_i)
            .else_(Bit(0).concat(stores_sub)),
        )

        # Generate handshake signals
        group_init_ready = LogicArray(
            em, "group_init_ready", "w", self.configs.numGroups
        )
        group_init_hs = LogicArray(em, "group_init_hs", "w", self.configs.numGroups)

        for i in range(0, self.configs.numGroups):
            em.add_assignment(
                group_init_ready[i],
                Bit(1)
                .when(
                    (
                        empty_loads
                        >= Val(self.configs.gaNumLoads[i], self.configs.emptyLdAddrW)
                    )
                    & (
                        empty_stores
                        >= Val(self.configs.gaNumStores[i], self.configs.emptyStAddrW)
                    )
                )
                .else_(Bit(0)),
            )

        if self.configs.gaMulti:
            group_init_and = LogicArray(
                em, "group_init_and", "w", self.configs.numGroups
            )
            ga_rr_mask = LogicVec(em, "ga_rr_mask", "r", self.configs.numGroups)
            ga_rr_mask.regInit()
            for i in range(0, self.configs.numGroups):
                em.add_assignment(
                    group_init_and[i], group_init_ready[i] & group_init_valid_i[i]
                )
                em.add_assignment(group_init_ready_o[i], group_init_hs[i])
            CyclicPriorityMasking(em, group_init_hs, group_init_and, ga_rr_mask)
            for i in range(0, self.configs.numGroups):
                em.add_assignment(
                    (ga_rr_mask, (i + 1) % self.configs.numGroups),
                    Val(group_init_hs, i),
                )
        else:
            for i in range(0, self.configs.numGroups):
                em.add_assignment(group_init_ready_o[i], group_init_ready[i])
                em.add_assignment(
                    group_init_hs[i], group_init_ready[i] & group_init_valid_i[i]
                )

        # ROM value
        if self.configs.ldpAddrW > 0:
            ldq_port_idx_rom = LogicVecArray(
                em,
                "ldq_port_idx_rom",
                "w",
                self.configs.numLdqEntries,
                self.configs.ldpAddrW,
            )
        if self.configs.stpAddrW > 0:
            stq_port_idx_rom = LogicVecArray(
                em,
                "stq_port_idx_rom",
                "w",
                self.configs.numStqEntries,
                self.configs.stpAddrW,
            )
        ga_ls_order_rom = LogicVecArray(
            em,
            "ga_ls_order_rom",
            "w",
            self.configs.numLdqEntries,
            self.configs.numStqEntries,
        )
        ga_ls_order_temp = LogicVecArray(
            em,
            "ga_ls_order_temp",
            "w",
            self.configs.numLdqEntries,
            self.configs.numStqEntries,
        )
        if self.configs.ldpAddrW > 0:
            Mux1HROM(em, ldq_port_idx_rom, self.configs.gaLdPortIdx, group_init_hs)
        if self.configs.stpAddrW > 0:
            Mux1HROM(em, stq_port_idx_rom, self.configs.gaStPortIdx, group_init_hs)
        Mux1HROM(
            em, ga_ls_order_rom, self.configs.gaLdOrder, group_init_hs, em.mask_less
        )
        Mux1HROM(em, num_loads, self.configs.gaNumLoads, group_init_hs)
        Mux1HROM(em, num_stores, self.configs.gaNumStores, group_init_hs)
        em.add_assignment(num_loads_o, num_loads)
        em.add_assignment(num_stores_o, num_stores)

        ldq_wen_unshifted = LogicArray(
            em, "ldq_wen_unshifted", "w", self.configs.numLdqEntries
        )
        stq_wen_unshifted = LogicArray(
            em, "stq_wen_unshifted", "w", self.configs.numStqEntries
        )
        for i in range(0, self.configs.numLdqEntries):
            em.add_assignment(
                ldq_wen_unshifted[i],
                Bit(1)
                .when(Val(num_loads) > Val(i, self.configs.ldqAddrW))
                .else_(Bit(0)),
            )
        for i in range(0, self.configs.numStqEntries):
            em.add_assignment(
                stq_wen_unshifted[i],
                Bit(1)
                .when(Val(num_stores) > Val(i, self.configs.stqAddrW))
                .else_(Bit(0)),
            )

        # Shift the arrays
        if self.configs.ldpAddrW > 0:
            CyclicLeftShift(em, ldq_port_idx_o, ldq_port_idx_rom, ldq_tail_i)
        if self.configs.stpAddrW > 0:
            CyclicLeftShift(em, stq_port_idx_o, stq_port_idx_rom, stq_tail_i)
        CyclicLeftShift(em, ldq_wen_o, ldq_wen_unshifted, ldq_tail_i)
        CyclicLeftShift(em, stq_wen_o, stq_wen_unshifted, stq_tail_i)
        for i in range(0, self.configs.numLdqEntries):
            CyclicLeftShift(em, ga_ls_order_temp[i], ga_ls_order_rom[i], stq_tail_i)
        CyclicLeftShift(em, ga_ls_order_o, ga_ls_order_temp, ldq_tail_i)

        # Write to the file
        output_str = em.get_definition_str(
            self.module_name, write_regs=self.configs.gaMulti
        )
        with open(f"{path_rtl}/{self.name}.{em.get_file_suffix()}", "a") as file:
            file.write(output_str)

    def instantiate(
        self,
        em: Emitter,
        group_init_valid_i: LogicArray,
        group_init_ready_o: LogicArray,
        ldq_tail_i: LogicVec,
        ldq_head_i: LogicVec,
        ldq_empty_i: Logic,
        stq_tail_i: LogicVec,
        stq_head_i: LogicVec,
        stq_empty_i: Logic,
        ldq_wen_o: LogicArray,
        num_loads_o: LogicVec,
        ldq_port_idx_o: LogicVecArray,
        stq_wen_o: LogicArray,
        num_stores_o: LogicVec,
        stq_port_idx_o: LogicVecArray,
        ga_ls_order_o: LogicVecArray,
    ) -> str:
        """
        Group Allocator Instantiation

        Creates the VHDL port mapping for the group allocator entity.

        Parameters:
            em                   : Emitter for code generation
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

        em.start_instantiation(self.module_name)

        em.add_map("rst", "rst")
        em.add_map("clk", "clk")

        for i in range(0, self.configs.numGroups):
            em.add_map(f"group_init_valid_{i}_i", group_init_valid_i.getNameRead(i))

        for i in range(0, self.configs.numGroups):
            em.add_map(f"group_init_ready_{i}_o", group_init_ready_o.getNameWrite(i))

        em.add_map("ldq_tail_i", ldq_tail_i.getNameRead())
        em.add_map("ldq_head_i", ldq_head_i.getNameRead())
        em.add_map("ldq_empty_i", ldq_empty_i.getNameRead())

        em.add_map("stq_tail_i", stq_tail_i.getNameRead())
        em.add_map("stq_head_i", stq_head_i.getNameRead())
        em.add_map("stq_empty_i", stq_empty_i.getNameRead())

        for i in range(0, self.configs.numLdqEntries):
            em.add_map(f"ldq_wen_{i}_o", ldq_wen_o.getNameWrite(i))

        em.add_map(f"num_loads_o", num_loads_o.getNameWrite())

        if self.configs.ldpAddrW > 0:
            for i in range(0, self.configs.numLdqEntries):
                em.add_map(f"ldq_port_idx_{i}_o", ldq_port_idx_o.getNameWrite(i))

        for i in range(0, self.configs.numStqEntries):
            em.add_map(f"stq_wen_{i}_o", stq_wen_o.getNameWrite(i))
        if self.configs.stpAddrW > 0:
            for i in range(0, self.configs.numStqEntries):
                em.add_map(f"stq_port_idx_{i}_o", stq_port_idx_o.getNameWrite(i))

        for i in range(0, self.configs.numLdqEntries):
            em.add_map(f"ga_ls_order_{i}_o", ga_ls_order_o.getNameWrite(i))

        em.add_map("num_stores_o", num_stores_o.getNameWrite())
        em.complete_instantiation()
        return em
