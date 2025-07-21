module {
  hw.module @elastic_miter_ri_fork_lhs_ri_fork_rhs(in %A_in : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out EQ_B_out : !handshake.channel<i1>, out EQ_C_out : !handshake.channel<i1>) {
    %in_fork_A_in.outs_0, %in_fork_A_in.outs_1 = hw.instance "in_fork_A_in" @handshake_lazy_fork_0(ins: %A_in: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs_0: !handshake.channel<i1>, outs_1: !handshake.channel<i1>)
    %lhs_in_buf_A_in.outs = hw.instance "lhs_in_buf_A_in" @handshake_buffer_0(ins: %in_fork_A_in.outs_0: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %rhs_in_buf_A_in.outs = hw.instance "rhs_in_buf_A_in" @handshake_buffer_0(ins: %in_fork_A_in.outs_1: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %out_nds_B_out.result = hw.instance "out_nds_B_out" @handshake_ndsource_0(clk: %clk: i1, rst: %rst: i1) -> (result: !handshake.control<>)
    %out_lf_B_out.outs_0, %out_lf_B_out.outs_1 = hw.instance "out_lf_B_out" @handshake_lazy_fork_1(ins: %out_nds_B_out.result: !handshake.control<>, clk: %clk: i1, rst: %rst: i1) -> (outs_0: !handshake.control<>, outs_1: !handshake.control<>)
    %out_buf_lhs_nds_B_out.outs = hw.instance "out_buf_lhs_nds_B_out" @handshake_buffer_1(ins: %out_lf_B_out.outs_0: !handshake.control<>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.control<>)
    %out_buf_rhs_nds_B_out.outs = hw.instance "out_buf_rhs_nds_B_out" @handshake_buffer_1(ins: %out_lf_B_out.outs_1: !handshake.control<>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.control<>)
    %lhs_out_bl_B_out.outs = hw.instance "lhs_out_bl_B_out" @handshake_blocker_0(ins: %lhs_fork.outs_0: !handshake.channel<i1>, ctrl: %out_buf_lhs_nds_B_out.outs: !handshake.control<>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %rhs_out_bl_B_out.outs = hw.instance "rhs_out_bl_B_out" @handshake_blocker_0(ins: %rhs_ri1.outs: !handshake.channel<i1>, ctrl: %out_buf_rhs_nds_B_out.outs: !handshake.control<>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %out_eq_B_out.result = hw.instance "out_eq_B_out" @handshake_cmpi_0(lhs: %lhs_out_bl_B_out.outs: !handshake.channel<i1>, rhs: %rhs_out_bl_B_out.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (result: !handshake.channel<i1>)
    %out_nds_C_out.result = hw.instance "out_nds_C_out" @handshake_ndsource_0(clk: %clk: i1, rst: %rst: i1) -> (result: !handshake.control<>)
    %out_lf_C_out.outs_0, %out_lf_C_out.outs_1 = hw.instance "out_lf_C_out" @handshake_lazy_fork_1(ins: %out_nds_C_out.result: !handshake.control<>, clk: %clk: i1, rst: %rst: i1) -> (outs_0: !handshake.control<>, outs_1: !handshake.control<>)
    %out_buf_lhs_nds_C_out.outs = hw.instance "out_buf_lhs_nds_C_out" @handshake_buffer_1(ins: %out_lf_C_out.outs_0: !handshake.control<>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.control<>)
    %out_buf_rhs_nds_C_out.outs = hw.instance "out_buf_rhs_nds_C_out" @handshake_buffer_1(ins: %out_lf_C_out.outs_1: !handshake.control<>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.control<>)
    %lhs_out_bl_C_out.outs = hw.instance "lhs_out_bl_C_out" @handshake_blocker_0(ins: %lhs_fork.outs_1: !handshake.channel<i1>, ctrl: %out_buf_lhs_nds_C_out.outs: !handshake.control<>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %rhs_out_bl_C_out.outs = hw.instance "rhs_out_bl_C_out" @handshake_blocker_0(ins: %rhs_ri2.outs: !handshake.channel<i1>, ctrl: %out_buf_rhs_nds_C_out.outs: !handshake.control<>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %out_eq_C_out.result = hw.instance "out_eq_C_out" @handshake_cmpi_0(lhs: %lhs_out_bl_C_out.outs: !handshake.channel<i1>, rhs: %rhs_out_bl_C_out.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (result: !handshake.channel<i1>)
    %lhs_buffer.outs = hw.instance "lhs_buffer" @handshake_buffer_2(ins: %lhs_in_buf_A_in.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %lhs_ri.outs = hw.instance "lhs_ri" @handshake_spec_v2_repeating_init_0(ins: %lhs_buffer.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %lhs_fork.outs_0, %lhs_fork.outs_1 = hw.instance "lhs_fork" @handshake_fork_0(ins: %lhs_ri.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs_0: !handshake.channel<i1>, outs_1: !handshake.channel<i1>)
    %rhs_fork.outs_0, %rhs_fork.outs_1 = hw.instance "rhs_fork" @handshake_fork_0(ins: %rhs_in_buf_A_in.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs_0: !handshake.channel<i1>, outs_1: !handshake.channel<i1>)
    %rhs_buffer1.outs = hw.instance "rhs_buffer1" @handshake_buffer_2(ins: %rhs_fork.outs_0: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %rhs_ri1.outs = hw.instance "rhs_ri1" @handshake_spec_v2_repeating_init_0(ins: %rhs_buffer1.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %rhs_buffer2.outs = hw.instance "rhs_buffer2" @handshake_buffer_2(ins: %rhs_fork.outs_1: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %rhs_ri2.outs = hw.instance "rhs_ri2" @handshake_spec_v2_repeating_init_0(ins: %rhs_buffer2.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    hw.output %out_eq_B_out.result, %out_eq_C_out.result : !handshake.channel<i1>, !handshake.channel<i1>
  }
  hw.module.extern @handshake_lazy_fork_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs_0 : !handshake.channel<i1>, out outs_1 : !handshake.channel<i1>) attributes {hw.name = "handshake.lazy_fork", hw.parameters = {DATA_TYPE = !handshake.channel<i1>, SIZE = 2 : ui32}}
  hw.module.extern @handshake_buffer_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i1>) attributes {hw.name = "handshake.buffer", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_DV", DATA_TYPE = !handshake.channel<i1>, NUM_SLOTS = 1 : ui32}}
  hw.module.extern @handshake_ndsource_0(in %clk : i1, in %rst : i1, out result : !handshake.control<>) attributes {hw.name = "handshake.ndsource", hw.parameters = {}}
  hw.module.extern @handshake_lazy_fork_1(in %ins : !handshake.control<>, in %clk : i1, in %rst : i1, out outs_0 : !handshake.control<>, out outs_1 : !handshake.control<>) attributes {hw.name = "handshake.lazy_fork", hw.parameters = {DATA_TYPE = !handshake.control<>, SIZE = 2 : ui32}}
  hw.module.extern @handshake_buffer_1(in %ins : !handshake.control<>, in %clk : i1, in %rst : i1, out outs : !handshake.control<>) attributes {hw.name = "handshake.buffer", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_DV", DATA_TYPE = !handshake.control<>, NUM_SLOTS = 1 : ui32}}
  hw.module.extern @handshake_blocker_0(in %ins : !handshake.channel<i1>, in %ctrl : !handshake.control<>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i1>) attributes {hw.name = "handshake.blocker", hw.parameters = {SIZE = 2 : ui32}}
  hw.module.extern @handshake_cmpi_0(in %lhs : !handshake.channel<i1>, in %rhs : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out result : !handshake.channel<i1>) attributes {hw.name = "handshake.cmpi", hw.parameters = {DATA_TYPE = !handshake.channel<i1>, INTERNAL_DELAY = "0.0", PREDICATE = "eq"}}
  hw.module.extern @handshake_buffer_2(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i1>) attributes {hw.name = "handshake.buffer", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_NONE", DATA_TYPE = !handshake.channel<i1>, NUM_SLOTS = 1 : ui32}}
  hw.module.extern @handshake_spec_v2_repeating_init_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i1>) attributes {hw.name = "handshake.spec_v2_repeating_init", hw.parameters = {INIT_TOKEN = 1 : ui32}}
  hw.module.extern @handshake_fork_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs_0 : !handshake.channel<i1>, out outs_1 : !handshake.channel<i1>) attributes {hw.name = "handshake.fork", hw.parameters = {DATA_TYPE = !handshake.channel<i1>, SIZE = 2 : ui32}}
  hw.module @elastic_miter_ri_fork_lhs_ri_fork_rhs_wrapper(in %A_in : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out EQ_B_out : !handshake.channel<i1>, out EQ_C_out : !handshake.channel<i1>) {
    %elastic_miter_ri_fork_lhs_ri_fork_rhs_wrapped.EQ_B_out, %elastic_miter_ri_fork_lhs_ri_fork_rhs_wrapped.EQ_C_out = hw.instance "elastic_miter_ri_fork_lhs_ri_fork_rhs_wrapped" @elastic_miter_ri_fork_lhs_ri_fork_rhs(A_in: %A_in: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (EQ_B_out: !handshake.channel<i1>, EQ_C_out: !handshake.channel<i1>)
    hw.output %elastic_miter_ri_fork_lhs_ri_fork_rhs_wrapped.EQ_B_out, %elastic_miter_ri_fork_lhs_ri_fork_rhs_wrapped.EQ_C_out : !handshake.channel<i1>, !handshake.channel<i1>
  }
}

