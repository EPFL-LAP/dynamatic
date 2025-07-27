module {
  hw.module @sup_source_rhs(in %ctrl : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out result : !handshake.control<>) {
    %ndw_in_ctrl.outs = hw.instance "ndw_in_ctrl" @handshake_ndwire_0(ins: %ctrl: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %ndw_out_result.outs = hw.instance "ndw_out_result" @handshake_ndwire_1(ins: %passer.result: !handshake.control<>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.control<>)
    %source.outs = hw.instance "source" @handshake_source_0(clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.control<>)
    %passer.result = hw.instance "passer" @handshake_passer_0(data: %source.outs: !handshake.control<>, ctrl: %ndw_in_ctrl.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (result: !handshake.control<>)
    hw.output %ndw_out_result.outs : !handshake.control<>
  }
  hw.module.extern @handshake_ndwire_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i1>) attributes {hw.name = "handshake.ndwire", hw.parameters = {DATA_TYPE = !handshake.channel<i1>}}
  hw.module.extern @handshake_ndwire_1(in %ins : !handshake.control<>, in %clk : i1, in %rst : i1, out outs : !handshake.control<>) attributes {hw.name = "handshake.ndwire", hw.parameters = {DATA_TYPE = !handshake.control<>}}
  hw.module.extern @handshake_source_0(in %clk : i1, in %rst : i1, out outs : !handshake.control<>) attributes {hw.name = "handshake.source", hw.parameters = {}}
  hw.module.extern @handshake_passer_0(in %data : !handshake.control<>, in %ctrl : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out result : !handshake.control<>) attributes {hw.name = "handshake.passer", hw.parameters = {}}
  hw.module @sup_source_rhs_wrapper(in %ctrl : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out result : !handshake.control<>) {
    %sup_source_rhs_wrapped.result = hw.instance "sup_source_rhs_wrapped" @sup_source_rhs(ctrl: %ctrl: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (result: !handshake.control<>)
    hw.output %sup_source_rhs_wrapped.result : !handshake.control<>
  }
}

