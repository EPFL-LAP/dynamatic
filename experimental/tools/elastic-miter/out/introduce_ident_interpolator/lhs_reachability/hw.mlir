module {
  hw.module @introduce_ident_interpolator_lhs(in %A_in : !handshake.channel<i1>, in %clk : i1, in %rst : i1) {
    %ndw_in_A_in.outs = hw.instance "ndw_in_A_in" @handshake_ndwire_0(ins: %A_in: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    hw.instance "vm_sink_0" @handshake_sink_0(ins: %ndw_in_A_in.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> ()
    hw.output
  }
  hw.module.extern @handshake_ndwire_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i1>) attributes {hw.name = "handshake.ndwire", hw.parameters = {DATA_TYPE = !handshake.channel<i1>}}
  hw.module.extern @handshake_sink_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1) attributes {hw.name = "handshake.sink", hw.parameters = {DATA_TYPE = !handshake.channel<i1>}}
  hw.module @introduce_ident_interpolator_lhs_wrapper(in %A_in : !handshake.channel<i1>, in %clk : i1, in %rst : i1) {
    hw.instance "introduce_ident_interpolator_lhs_wrapped" @introduce_ident_interpolator_lhs(A_in: %A_in: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> ()
    hw.output
  }
}

