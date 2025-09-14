module {
  hw.module @muxForkSwap_rhs(in %sel_in : !handshake.channel<i1>, in %data_in : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out out1 : !handshake.channel<i1>, out out2 : !handshake.channel<i1>) {
    %source.outs = hw.instance "source" @handshake_source_0(clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.control<>)
    %constant.outs = hw.instance "constant" @handshake_constant_0(ctrl: %source.outs: !handshake.control<>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %mux0.outs = hw.instance "mux0" @handshake_mux_0(index: %sel_in: !handshake.channel<i1>, ins_0: %constant.outs: !handshake.channel<i1>, ins_1: %data_in: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %fork.outs_0, %fork.outs_1 = hw.instance "fork" @handshake_fork_0(ins: %mux0.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs_0: !handshake.channel<i1>, outs_1: !handshake.channel<i1>)
    hw.output %fork.outs_0, %fork.outs_1 : !handshake.channel<i1>, !handshake.channel<i1>
  }
  hw.module.extern @handshake_source_0(in %clk : i1, in %rst : i1, out outs : !handshake.control<>) attributes {hw.name = "handshake.source", hw.parameters = {}}
  hw.module.extern @handshake_constant_0(in %ctrl : !handshake.control<>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i1>) attributes {hw.name = "handshake.constant", hw.parameters = {DATA_WIDTH = 1 : ui32, VALUE = "0"}}
  hw.module.extern @handshake_mux_0(in %index : !handshake.channel<i1>, in %ins_0 : !handshake.channel<i1>, in %ins_1 : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i1>) attributes {hw.name = "handshake.mux", hw.parameters = {DATA_TYPE = !handshake.channel<i1>, SELECT_TYPE = !handshake.channel<i1>, SIZE = 2 : ui32}}
  hw.module.extern @handshake_fork_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs_0 : !handshake.channel<i1>, out outs_1 : !handshake.channel<i1>) attributes {hw.name = "handshake.fork", hw.parameters = {DATA_TYPE = !handshake.channel<i1>, SIZE = 2 : ui32}}
  hw.module @muxForkSwap_rhs_wrapper(in %sel_in : !handshake.channel<i1>, in %data_in : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out out1 : !handshake.channel<i1>, out out2 : !handshake.channel<i1>) {
    %muxForkSwap_rhs_wrapped.out1, %muxForkSwap_rhs_wrapped.out2 = hw.instance "muxForkSwap_rhs_wrapped" @muxForkSwap_rhs(sel_in: %sel_in: !handshake.channel<i1>, data_in: %data_in: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (out1: !handshake.channel<i1>, out2: !handshake.channel<i1>)
    hw.output %muxForkSwap_rhs_wrapped.out1, %muxForkSwap_rhs_wrapped.out2 : !handshake.channel<i1>, !handshake.channel<i1>
  }
}

