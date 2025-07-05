module {
  hw.module @fork_rhs(in %A_in : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out B_out : !handshake.channel<i1>, out C_out : !handshake.channel<i1>) {
    %ndw_in_A_in.outs = hw.instance "ndw_in_A_in" @handshake_ndwire_0(ins: %A_in: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %ndw_out_B_out.outs = hw.instance "ndw_out_B_out" @handshake_ndwire_0(ins: %fork.outs_0: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %ndw_out_C_out.outs = hw.instance "ndw_out_C_out" @handshake_ndwire_0(ins: %fork.outs_1: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %fork.outs_0, %fork.outs_1 = hw.instance "fork" @handshake_lazy_fork_0(ins: %ndw_in_A_in.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs_0: !handshake.channel<i1>, outs_1: !handshake.channel<i1>)
    hw.output %ndw_out_B_out.outs, %ndw_out_C_out.outs : !handshake.channel<i1>, !handshake.channel<i1>
  }
  hw.module.extern @handshake_ndwire_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i1>) attributes {hw.name = "handshake.ndwire", hw.parameters = {DATA_TYPE = !handshake.channel<i1>}}
  hw.module.extern @handshake_lazy_fork_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs_0 : !handshake.channel<i1>, out outs_1 : !handshake.channel<i1>) attributes {hw.name = "handshake.lazy_fork", hw.parameters = {DATA_TYPE = !handshake.channel<i1>, SIZE = 2 : ui32}}
  hw.module @fork_rhs_wrapper(in %A_in : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out B_out : !handshake.channel<i1>, out C_out : !handshake.channel<i1>) {
    %fork_rhs_wrapped.B_out, %fork_rhs_wrapped.C_out = hw.instance "fork_rhs_wrapped" @fork_rhs(A_in: %A_in: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (B_out: !handshake.channel<i1>, C_out: !handshake.channel<i1>)
    hw.output %fork_rhs_wrapped.B_out, %fork_rhs_wrapped.C_out : !handshake.channel<i1>, !handshake.channel<i1>
  }
}

