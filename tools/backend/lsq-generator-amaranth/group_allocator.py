# pyright: reportInvalidTypeForm=false
from amaranth import Module, Mux, Signal
from amaranth.lib.data import ArrayLayout
from amaranth.lib.wiring import Component, In, Out

from config import LsqConfig
from utils import MuxOneHot, RotateLeft, WrapSubtract


class GroupAllocator(Component):
    """
    Group Allocator for a Load-Store Queue (LSQ) system.

    Allocates space for groups of memory operations (loads and stores) in the
    load queue and the store queue. For now this is a port-only stub; the
    allocation logic is not yet implemented.

    Parameters
    ----------
    config : LsqConfig
        Configuration object containing parameters for the LSQ.
    """

    def __init__(self, config: LsqConfig):
        self._config = config
        super().__init__({
            # group init handshake
            "group_init_valid_i": In(ArrayLayout(1, config.numGroups)),
            "group_init_ready_o": Out(ArrayLayout(1, config.numGroups)),
            # load queue data
            "ldq_tail_i":  In(config.ldqAddrW),
            "ldq_head_i":  In(config.ldqAddrW),
            "ldq_empty_i": In(1),
            # store queue data
            "stq_tail_i":  In(config.stqAddrW),
            "stq_head_i":  In(config.stqAddrW),
            "stq_empty_i": In(1),
            # load queue outputs
            "ldq_wen_o":      Out(ArrayLayout(1, config.numLdqEntries)),
            "num_loads_o":    Out(config.ldqAddrW),
            "ldq_port_idx_o": Out(ArrayLayout(config.ldpAddrW, config.numLdqEntries)),
            # store queue outputs
            "stq_wen_o":      Out(ArrayLayout(1, config.numStqEntries)),
            "num_stores_o":   Out(config.stqAddrW),
            "stq_port_idx_o": Out(ArrayLayout(config.stpAddrW, config.numStqEntries)),
            # order matrix outputs
            "ga_ls_order_o":  Out(ArrayLayout(config.numStqEntries, config.numLdqEntries)),
        })

    def elaborate(self, platform):
        config = self._config
        m = Module()

        # Get empty counts for load and store queues
        def empty_count(name, empty_count, head, tail, empty, addr_width, num_entries):
            # num_entries if empty, else wrap_subtract(head, tail)
            m.submodules[f"{name}_wrap_subtract"] = wrap_subtract = WrapSubtract(addr_width, num_entries)
            m.d.comb += wrap_subtract.in_i.eq(head)
            m.d.comb += wrap_subtract.sub_i.eq(tail)
            m.d.comb += empty_count.eq(Mux(empty, num_entries, wrap_subtract.out_o))

        empty_loads = Signal(config.ldqAddrW + 1)
        empty_stores = Signal(config.stqAddrW + 1)
        empty_count("ldq", empty_loads, self.ldq_head_i, self.ldq_tail_i, self.ldq_empty_i, config.ldqAddrW, config.numLdqEntries)
        empty_count("stq", empty_stores, self.stq_head_i, self.stq_tail_i, self.stq_empty_i, config.stqAddrW, config.numStqEntries)

        # Group allocation handshakes
        # We are ready to allocate a group if there are enough empty slots in the load/store queues.
        # TODO: This could be improved by using only two comparators which compare the empty counts to the num_loads/stores ROM outputs.
        group_init_hs = Signal(config.numGroups)
        for i in range(config.numGroups):
            m.d.comb += self.group_init_ready_o[i].eq(
                (empty_loads >= config.gaNumLoads[i]) &
                (empty_stores >= config.gaNumStores[i])
            )
            m.d.comb += group_init_hs[i].eq(
                self.group_init_valid_i[i] & self.group_init_ready_o[i]
            )
        
        # ROMs
        def ga_rom(name, data, output):
            assert len(data) == config.numGroups, f"ROM data length mismatch for {name}: expected {config.numGroups}, got {len(data)}"
            m.submodules[f"{name}_rom"] = rom = MuxOneHot(output.shape(), config.numGroups)
            m.d.comb += rom.input_i.eq(rom.input_i.shape().const(data))
            m.d.comb += rom.sel_oh_i.eq(group_init_hs)
            m.d.comb += output.eq(rom.output_o)

        def pad_zeros_inner(data: list[list[int]], inner_len: int) -> list[list[int]]:
            """Pad inner lists with zeros to a given length."""
            return [row + [0] * (inner_len - len(row)) for row in data]

        ga_rom("num_loads", config.gaNumLoads, self.num_loads_o)
        ga_rom("num_stores", config.gaNumStores, self.num_stores_o)

        ldq_port_idx_data = pad_zeros_inner(config.gaLdPortIdx, config.numLdqEntries)
        stq_port_idx_data = pad_zeros_inner(config.gaStPortIdx, config.numStqEntries)
        ldq_port_idx_rom_out = Signal(ArrayLayout(config.ldpAddrW, config.numLdqEntries))
        stq_port_idx_rom_out = Signal(ArrayLayout(config.stpAddrW, config.numStqEntries))
        ga_rom("ldq_port_idx", ldq_port_idx_data, ldq_port_idx_rom_out)
        ga_rom("stq_port_idx", stq_port_idx_data, stq_port_idx_rom_out)

        ga_ls_order_data = [[(1 << x) - 1 for x in group] for group in config.gaLdOrder]
        ga_ls_order_data = pad_zeros_inner(ga_ls_order_data, config.numLdqEntries)
        ga_ls_order_rom_out = Signal(ArrayLayout(config.numStqEntries, config.numLdqEntries))
        ga_rom("ga_ls_order", ga_ls_order_data, ga_ls_order_rom_out)

        # LDQ/STQ write-enable generation
        ldq_wen_tmp = Signal(config.numLdqEntries)
        m.d.comb += ldq_wen_tmp.eq((1 << self.num_loads_o) - 1)
        stq_wen_tmp = Signal(config.numStqEntries)
        m.d.comb += stq_wen_tmp.eq((1 << self.num_stores_o) - 1)

        m.submodules.ldq_wen_rotate = ldq_wen_rotate = RotateLeft(1, config.numLdqEntries, config.ldqAddrW)
        m.d.comb += ldq_wen_rotate.input_i.eq(ldq_wen_tmp)
        m.d.comb += ldq_wen_rotate.rotate_amount_i.eq(self.ldq_tail_i)
        m.d.comb += self.ldq_wen_o.eq(ldq_wen_rotate.output_o)

        m.submodules.stq_wen_rotate = stq_wen_rotate = RotateLeft(1, config.numStqEntries, config.stqAddrW)
        m.d.comb += stq_wen_rotate.input_i.eq(stq_wen_tmp)
        m.d.comb += stq_wen_rotate.rotate_amount_i.eq(self.stq_tail_i)
        m.d.comb += self.stq_wen_o.eq(stq_wen_rotate.output_o)

        # LDQ/STQ port index generation
        m.submodules.ldq_port_idx_rotate = ldq_port_idx_rotate = RotateLeft(config.ldpAddrW, config.numLdqEntries, config.ldqAddrW)
        m.d.comb += ldq_port_idx_rotate.input_i.eq(ldq_port_idx_rom_out)
        m.d.comb += ldq_port_idx_rotate.rotate_amount_i.eq(self.ldq_tail_i)
        m.d.comb += self.ldq_port_idx_o.eq(ldq_port_idx_rotate.output_o)

        m.submodules.stq_port_idx_rotate = stq_port_idx_rotate = RotateLeft(config.stpAddrW, config.numStqEntries, config.stqAddrW)
        m.d.comb += stq_port_idx_rotate.input_i.eq(stq_port_idx_rom_out)
        m.d.comb += stq_port_idx_rotate.rotate_amount_i.eq(self.stq_tail_i)
        m.d.comb += self.stq_port_idx_o.eq(stq_port_idx_rotate.output_o)

        # order matrix generation
        ga_ls_order_tmp = Signal(ArrayLayout(config.numStqEntries, config.numLdqEntries))
        for i in range(config.numLdqEntries):
            m.submodules[f"ga_ls_order_rotate_store_{i}"] = ga_ls_order_rotate = RotateLeft(1, config.numStqEntries, config.stqAddrW)
            m.d.comb += ga_ls_order_rotate.input_i.eq(ga_ls_order_rom_out[i])
            m.d.comb += ga_ls_order_rotate.rotate_amount_i.eq(self.stq_tail_i)
            m.d.comb += ga_ls_order_tmp[i].eq(ga_ls_order_rotate.output_o)

        m.submodules.ga_ls_order_rotate_load = ga_ls_order_rotate_load = RotateLeft(config.numStqEntries, config.numLdqEntries, config.ldqAddrW)
        m.d.comb += ga_ls_order_rotate_load.input_i.eq(ga_ls_order_tmp)
        m.d.comb += ga_ls_order_rotate_load.rotate_amount_i.eq(self.ldq_tail_i)
        m.d.comb += self.ga_ls_order_o.eq(ga_ls_order_rotate_load.output_o)

        return m
