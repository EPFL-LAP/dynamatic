module {
  hw.module @elastic_miter_introduceIdentInterpolator_lhs_introduceIdentInterpolator_lhs(in %A_in : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out EQ_B_out : !handshake.channel<i1>) {
    %in_fork_A_in.outs_0, %in_fork_A_in.outs_1 = hw.instance "in_fork_A_in" @handshake_lazy_fork_0(ins: %A_in: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs_0: !handshake.channel<i1>, outs_1: !handshake.channel<i1>)
    %lhs_in_buf_A_in.outs = hw.instance "lhs_in_buf_A_in" @handshake_buffer_0(ins: %in_fork_A_in.outs_0: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %rhs_in_buf_A_in.outs = hw.instance "rhs_in_buf_A_in" @handshake_buffer_0(ins: %in_fork_A_in.outs_1: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %out_src_B_out.outs = hw.instance "out_src_B_out" @handshake_source_0(clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.control<>)
    %out_lf_B_out.outs_0, %out_lf_B_out.outs_1 = hw.instance "out_lf_B_out" @handshake_lazy_fork_1(ins: %out_src_B_out.outs: !handshake.control<>, clk: %clk: i1, rst: %rst: i1) -> (outs_0: !handshake.control<>, outs_1: !handshake.control<>)
    %out_buf_lhs_nds_B_out.outs = hw.instance "out_buf_lhs_nds_B_out" @handshake_buffer_1(ins: %out_lf_B_out.outs_0: !handshake.control<>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.control<>)
    %out_buf_rhs_nds_B_out.outs = hw.instance "out_buf_rhs_nds_B_out" @handshake_buffer_1(ins: %out_lf_B_out.outs_1: !handshake.control<>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.control<>)
    %lhs_out_bl_B_out.outs = hw.instance "lhs_out_bl_B_out" @handshake_blocker_0(ins: %lhs_not2.outs: !handshake.channel<i1>, ctrl: %out_buf_lhs_nds_B_out.outs: !handshake.control<>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %rhs_out_bl_B_out.outs = hw.instance "rhs_out_bl_B_out" @handshake_blocker_0(ins: %rhs_interpolate.result: !handshake.channel<i1>, ctrl: %out_buf_rhs_nds_B_out.outs: !handshake.control<>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %out_eq_B_out.result = hw.instance "out_eq_B_out" @handshake_cmpi_0(lhs: %lhs_out_bl_B_out.outs: !handshake.channel<i1>, rhs: %rhs_out_bl_B_out.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (result: !handshake.channel<i1>)
    %lhs_not1.outs = hw.instance "lhs_not1" @handshake_not_0(ins: %lhs_in_buf_A_in.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %lhs_not2.outs = hw.instance "lhs_not2" @handshake_not_0(ins: %lhs_not1.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %rhs_a_fork.outs_0, %rhs_a_fork.outs_1 = hw.instance "rhs_a_fork" @handshake_fork_0(ins: %rhs_in_buf_A_in.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs_0: !handshake.channel<i1>, outs_1: !handshake.channel<i1>)
    %rhs_interpolate.result = hw.instance "rhs_interpolate" @handshake_spec_v2_interpolator_0(short: %rhs_a_fork.outs_0: !handshake.channel<i1>, long: %rhs_a_fork.outs_1: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (result: !handshake.channel<i1>)
    hw.output %out_eq_B_out.result : !handshake.channel<i1>
  }
  hw.module.extern @handshake_lazy_fork_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs_0 : !handshake.channel<i1>, out outs_1 : !handshake.channel<i1>) attributes {hw.name = "handshake.lazy_fork", hw.parameters = {DATA_TYPE = !handshake.channel<i1>, SIZE = 2 : ui32}}
  hw.module.extern @handshake_buffer_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i1>) attributes {hw.name = "handshake.buffer", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_DV", DATA_TYPE = !handshake.channel<i1>, DEBUG_COUNTER = true, NUM_SLOTS = 1 : ui32}}
  hw.module.extern @handshake_source_0(in %clk : i1, in %rst : i1, out outs : !handshake.control<>) attributes {hw.name = "handshake.source", hw.parameters = {}}
  hw.module.extern @handshake_lazy_fork_1(in %ins : !handshake.control<>, in %clk : i1, in %rst : i1, out outs_0 : !handshake.control<>, out outs_1 : !handshake.control<>) attributes {hw.name = "handshake.lazy_fork", hw.parameters = {DATA_TYPE = !handshake.control<>, SIZE = 2 : ui32}}
  hw.module.extern @handshake_buffer_1(in %ins : !handshake.control<>, in %clk : i1, in %rst : i1, out outs : !handshake.control<>) attributes {hw.name = "handshake.buffer", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_DV", DATA_TYPE = !handshake.control<>, DEBUG_COUNTER = false, NUM_SLOTS = 1 : ui32}}
  hw.module.extern @handshake_blocker_0(in %ins : !handshake.channel<i1>, in %ctrl : !handshake.control<>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i1>) attributes {hw.name = "handshake.blocker", hw.parameters = {SIZE = 2 : ui32}}
  hw.module.extern @handshake_cmpi_0(in %lhs : !handshake.channel<i1>, in %rhs : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out result : !handshake.channel<i1>) attributes {hw.name = "handshake.cmpi", hw.parameters = {DATA_TYPE = !handshake.channel<i1>, PREDICATE = "eq"}}
  hw.module.extern @handshake_not_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i1>) attributes {hw.name = "handshake.not", hw.parameters = {DATA_TYPE = !handshake.channel<i1>}}
  hw.module.extern @handshake_fork_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs_0 : !handshake.channel<i1>, out outs_1 : !handshake.channel<i1>) attributes {hw.name = "handshake.fork", hw.parameters = {DATA_TYPE = !handshake.channel<i1>, SIZE = 2 : ui32}}
  hw.module.extern @handshake_spec_v2_interpolator_0(in %short : !handshake.channel<i1>, in %long : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out result : !handshake.channel<i1>) attributes {hw.name = "handshake.spec_v2_interpolator", hw.parameters = {}}
  hw.module @elastic_miter_introduceIdentInterpolator_lhs_introduceIdentInterpolator_lhs_wrapper(in %A_in : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out EQ_B_out : !handshake.channel<i1>) {
    %elastic_miter_introduceIdentInterpolator_lhs_introduceIdentInterpolator_lhs_wrapped.EQ_B_out = hw.instance "elastic_miter_introduceIdentInterpolator_lhs_introduceIdentInterpolator_lhs_wrapped" @elastic_miter_introduceIdentInterpolator_lhs_introduceIdentInterpolator_lhs(A_in: %A_in: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (EQ_B_out: !handshake.channel<i1>)
    hw.output %elastic_miter_introduceIdentInterpolator_lhs_introduceIdentInterpolator_lhs_wrapped.EQ_B_out : !handshake.channel<i1>
  }
}

