module {
  hw.module @sup_source_lhs(in %ctrl : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out result : !handshake.control<>) {
    %ndw_in_ctrl.outs = hw.instance "ndw_in_ctrl" @handshake_ndwire_0(ins: %ctrl: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %ndw_out_result.outs = hw.instance "ndw_out_result" @handshake_ndwire_1(ins: %source.outs: !handshake.control<>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.control<>)
    hw.instance "sink" @handshake_sink_0(ins: %ndw_in_ctrl.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> ()
    %source.outs = hw.instance "source" @handshake_source_0(clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.control<>)
    hw.output %ndw_out_result.outs : !handshake.control<>
  }
  hw.module.extern @handshake_ndwire_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i1>) attributes {hw.name = "handshake.ndwire", hw.parameters = {DATA_TYPE = !handshake.channel<i1>}}
  hw.module.extern @handshake_ndwire_1(in %ins : !handshake.control<>, in %clk : i1, in %rst : i1, out outs : !handshake.control<>) attributes {hw.name = "handshake.ndwire", hw.parameters = {DATA_TYPE = !handshake.control<>}}
  hw.module.extern @handshake_sink_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1) attributes {hw.name = "handshake.sink", hw.parameters = {DATA_TYPE = !handshake.channel<i1>}}
  hw.module.extern @handshake_source_0(in %clk : i1, in %rst : i1, out outs : !handshake.control<>) attributes {hw.name = "handshake.source", hw.parameters = {}}
  hw.module @sup_source_lhs_wrapper(in %ctrl : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out result : !handshake.control<>) {
    %sup_source_lhs_wrapped.result = hw.instance "sup_source_lhs_wrapped" @sup_source_lhs(ctrl: %ctrl: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (result: !handshake.control<>)
    hw.output %sup_source_lhs_wrapped.result : !handshake.control<>
  }
}

