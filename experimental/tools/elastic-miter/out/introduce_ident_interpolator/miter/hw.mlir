module {
  hw.module @elastic_miter_introduce_ident_interpolator_lhs_introduce_ident_interpolator_rhs(in %A_in : !handshake.channel<i1>, in %clk : i1, in %rst : i1) {
    %in_fork_A_in.outs_0, %in_fork_A_in.outs_1 = hw.instance "in_fork_A_in" @handshake_lazy_fork_0(ins: %A_in: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs_0: !handshake.channel<i1>, outs_1: !handshake.channel<i1>)
    %rhs_in_buf_A_in.outs = hw.instance "rhs_in_buf_A_in" @handshake_buffer_0(ins: %in_fork_A_in.outs_1: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %lhs_in_ndw_A_in.outs = hw.instance "lhs_in_ndw_A_in" @handshake_ndwire_0(ins: %in_fork_A_in.outs_0: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %rhs_in_ndw_A_in.outs = hw.instance "rhs_in_ndw_A_in" @handshake_ndwire_0(ins: %rhs_in_buf_A_in.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    hw.instance "lhs_vm_sink_0" @handshake_sink_0(ins: %lhs_in_ndw_A_in.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> ()
    %rhs_vm_fork_0.outs_0, %rhs_vm_fork_0.outs_1 = hw.instance "rhs_vm_fork_0" @handshake_fork_0(ins: %rhs_in_ndw_A_in.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs_0: !handshake.channel<i1>, outs_1: !handshake.channel<i1>)
    %rhs_interpolate.result = hw.instance "rhs_interpolate" @handshake_spec_v2_interpolator_0(short: %rhs_vm_fork_0.outs_0: !handshake.channel<i1>, long: %rhs_vm_fork_0.outs_1: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (result: !handshake.channel<i1>)
    hw.instance "rhs_sink" @handshake_sink_0(ins: %rhs_interpolate.result: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> ()
    hw.output
  }
  hw.module.extern @handshake_lazy_fork_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs_0 : !handshake.channel<i1>, out outs_1 : !handshake.channel<i1>) attributes {hw.name = "handshake.lazy_fork", hw.parameters = {DATA_TYPE = !handshake.channel<i1>, SIZE = 2 : ui32}}
  hw.module.extern @handshake_buffer_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i1>) attributes {hw.name = "handshake.buffer", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_DV", DATA_TYPE = !handshake.channel<i1>, NUM_SLOTS = 1 : ui32}}
  hw.module.extern @handshake_ndwire_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i1>) attributes {hw.name = "handshake.ndwire", hw.parameters = {DATA_TYPE = !handshake.channel<i1>}}
  hw.module.extern @handshake_sink_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1) attributes {hw.name = "handshake.sink", hw.parameters = {DATA_TYPE = !handshake.channel<i1>}}
  hw.module.extern @handshake_fork_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs_0 : !handshake.channel<i1>, out outs_1 : !handshake.channel<i1>) attributes {hw.name = "handshake.fork", hw.parameters = {DATA_TYPE = !handshake.channel<i1>, SIZE = 2 : ui32}}
  hw.module.extern @handshake_spec_v2_interpolator_0(in %short : !handshake.channel<i1>, in %long : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out result : !handshake.channel<i1>) attributes {hw.name = "handshake.spec_v2_interpolator", hw.parameters = {}}
  hw.module @elastic_miter_introduce_ident_interpolator_lhs_introduce_ident_interpolator_rhs_wrapper(in %A_in : !handshake.channel<i1>, in %clk : i1, in %rst : i1) {
    hw.instance "elastic_miter_introduce_ident_interpolator_lhs_introduce_ident_interpolator_rhs_wrapped" @elastic_miter_introduce_ident_interpolator_lhs_introduce_ident_interpolator_rhs(A_in: %A_in: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> ()
    hw.output
  }
}

