module {
  hw.module @interpolator_ident_lhs(in %A_in : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out B_out : !handshake.channel<i1>) {
    %ndw_in_A_in.outs = hw.instance "ndw_in_A_in" @handshake_ndwire_0(ins: %A_in: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %ndw_out_B_out.outs = hw.instance "ndw_out_B_out" @handshake_ndwire_0(ins: %ndw_in_A_in.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    hw.output %ndw_out_B_out.outs : !handshake.channel<i1>
  }
  hw.module.extern @handshake_ndwire_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i1>) attributes {hw.name = "handshake.ndwire", hw.parameters = {DATA_TYPE = !handshake.channel<i1>}}
  hw.module @interpolator_ident_lhs_wrapper(in %A_in : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out B_out : !handshake.channel<i1>) {
    %interpolator_ident_lhs_wrapped.B_out = hw.instance "interpolator_ident_lhs_wrapped" @interpolator_ident_lhs(A_in: %A_in: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (B_out: !handshake.channel<i1>)
    hw.output %interpolator_ident_lhs_wrapped.B_out : !handshake.channel<i1>
  }
}

