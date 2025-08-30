module {
  hw.module @and_false_rhs(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out out : !handshake.channel<i1>) {
    %ndw_in_ins.outs = hw.instance "ndw_in_ins" @handshake_ndwire_0(ins: %ins: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %ndw_out_out.outs = hw.instance "ndw_out_out" @handshake_ndwire_0(ins: %constant.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    hw.instance "sink" @handshake_sink_0(ins: %ndw_in_ins.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> ()
    %source.outs = hw.instance "source" @handshake_source_0(clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.control<>)
    %constant.outs = hw.instance "constant" @handshake_constant_0(ctrl: %source.outs: !handshake.control<>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    hw.output %ndw_out_out.outs : !handshake.channel<i1>
  }
  hw.module.extern @handshake_ndwire_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i1>) attributes {hw.name = "handshake.ndwire", hw.parameters = {DATA_TYPE = !handshake.channel<i1>}}
  hw.module.extern @handshake_sink_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1) attributes {hw.name = "handshake.sink", hw.parameters = {DATA_TYPE = !handshake.channel<i1>}}
  hw.module.extern @handshake_source_0(in %clk : i1, in %rst : i1, out outs : !handshake.control<>) attributes {hw.name = "handshake.source", hw.parameters = {}}
  hw.module.extern @handshake_constant_0(in %ctrl : !handshake.control<>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i1>) attributes {hw.name = "handshake.constant", hw.parameters = {DATA_WIDTH = 1 : ui32, VALUE = "0"}}
  hw.module @and_false_rhs_wrapper(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out out : !handshake.channel<i1>) {
    %and_false_rhs_wrapped.out = hw.instance "and_false_rhs_wrapped" @and_false_rhs(ins: %ins: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (out: !handshake.channel<i1>)
    hw.output %and_false_rhs_wrapped.out : !handshake.channel<i1>
  }
}

