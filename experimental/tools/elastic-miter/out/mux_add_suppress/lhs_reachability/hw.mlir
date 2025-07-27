module {
  hw.module @mux_add_suppress_lhs(in %ins : !handshake.channel<i1>, in %sel : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out out : !handshake.channel<i1>) {
    %ndw_in_ins.outs = hw.instance "ndw_in_ins" @handshake_ndwire_0(ins: %ins: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %ndw_in_sel.outs = hw.instance "ndw_in_sel" @handshake_ndwire_0(ins: %sel: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %ndw_out_out.outs = hw.instance "ndw_out_out" @handshake_ndwire_0(ins: %mux.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %source.outs = hw.instance "source" @handshake_source_0(clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.control<>)
    %constant.outs = hw.instance "constant" @handshake_constant_0(ctrl: %source.outs: !handshake.control<>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %mux.outs = hw.instance "mux" @handshake_mux_0(index: %ndw_in_sel.outs: !handshake.channel<i1>, ins_0: %ndw_in_ins.outs: !handshake.channel<i1>, ins_1: %constant.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    hw.output %ndw_out_out.outs : !handshake.channel<i1>
  }
  hw.module.extern @handshake_ndwire_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i1>) attributes {hw.name = "handshake.ndwire", hw.parameters = {DATA_TYPE = !handshake.channel<i1>}}
  hw.module.extern @handshake_source_0(in %clk : i1, in %rst : i1, out outs : !handshake.control<>) attributes {hw.name = "handshake.source", hw.parameters = {}}
  hw.module.extern @handshake_constant_0(in %ctrl : !handshake.control<>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i1>) attributes {hw.name = "handshake.constant", hw.parameters = {DATA_WIDTH = 1 : ui32, INTERNAL_DELAY = "0.0", VALUE = "1"}}
  hw.module.extern @handshake_mux_0(in %index : !handshake.channel<i1>, in %ins_0 : !handshake.channel<i1>, in %ins_1 : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i1>) attributes {hw.name = "handshake.mux", hw.parameters = {DATA_TYPE = !handshake.channel<i1>, SELECT_TYPE = !handshake.channel<i1>, SIZE = 2 : ui32}}
  hw.module @mux_add_suppress_lhs_wrapper(in %ins : !handshake.channel<i1>, in %sel : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out out : !handshake.channel<i1>) {
    %mux_add_suppress_lhs_wrapped.out = hw.instance "mux_add_suppress_lhs_wrapped" @mux_add_suppress_lhs(ins: %ins: !handshake.channel<i1>, sel: %sel: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (out: !handshake.channel<i1>)
    hw.output %mux_add_suppress_lhs_wrapped.out : !handshake.channel<i1>
  }
}

