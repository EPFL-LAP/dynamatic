module {
  hw.module @sup_gamma_rhs(in %a1 : !handshake.channel<i1>, in %a2 : !handshake.channel<i1>, in %c1 : !handshake.channel<i1>, in %c2 : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out b : !handshake.channel<i1>) {
    %ndw_in_a1.outs = hw.instance "ndw_in_a1" @handshake_ndwire_0(ins: %a1: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %ndw_in_a2.outs = hw.instance "ndw_in_a2" @handshake_ndwire_0(ins: %a2: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %ndw_in_c1.outs = hw.instance "ndw_in_c1" @handshake_ndwire_0(ins: %c1: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %ndw_in_c2.outs = hw.instance "ndw_in_c2" @handshake_ndwire_0(ins: %c2: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %ndw_out_b.outs = hw.instance "ndw_out_b" @handshake_ndwire_0(ins: %passer.result: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %mux.outs = hw.instance "mux" @handshake_mux_0(index: %ndw_in_c2.outs: !handshake.channel<i1>, ins_0: %ndw_in_a2.outs: !handshake.channel<i1>, ins_1: %ndw_in_a1.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %passer.result = hw.instance "passer" @handshake_passer_0(data: %mux.outs: !handshake.channel<i1>, ctrl: %ndw_in_c1.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (result: !handshake.channel<i1>)
    hw.output %ndw_out_b.outs : !handshake.channel<i1>
  }
  hw.module.extern @handshake_ndwire_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i1>) attributes {hw.name = "handshake.ndwire", hw.parameters = {DATA_TYPE = !handshake.channel<i1>}}
  hw.module.extern @handshake_mux_0(in %index : !handshake.channel<i1>, in %ins_0 : !handshake.channel<i1>, in %ins_1 : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i1>) attributes {hw.name = "handshake.mux", hw.parameters = {DATA_TYPE = !handshake.channel<i1>, SELECT_TYPE = !handshake.channel<i1>, SIZE = 2 : ui32}}
  hw.module.extern @handshake_passer_0(in %data : !handshake.channel<i1>, in %ctrl : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out result : !handshake.channel<i1>) attributes {hw.name = "handshake.passer", hw.parameters = {}}
  hw.module @sup_gamma_rhs_wrapper(in %a1 : !handshake.channel<i1>, in %a2 : !handshake.channel<i1>, in %c1 : !handshake.channel<i1>, in %c2 : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out b : !handshake.channel<i1>) {
    %sup_gamma_rhs_wrapped.b = hw.instance "sup_gamma_rhs_wrapped" @sup_gamma_rhs(a1: %a1: !handshake.channel<i1>, a2: %a2: !handshake.channel<i1>, c1: %c1: !handshake.channel<i1>, c2: %c2: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (b: !handshake.channel<i1>)
    hw.output %sup_gamma_rhs_wrapped.b : !handshake.channel<i1>
  }
}

