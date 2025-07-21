module {
  hw.module @andForkSwap_lhs(in %in2 : !handshake.channel<i1>, in %in1 : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out out1 : !handshake.channel<i1>, out out2 : !handshake.channel<i1>) {
    %ndw_in_in2.outs = hw.instance "ndw_in_in2" @handshake_ndwire_0(ins: %in2: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %ndw_in_in1.outs = hw.instance "ndw_in_in1" @handshake_ndwire_0(ins: %in1: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %ndw_out_out1.outs = hw.instance "ndw_out_out1" @handshake_ndwire_0(ins: %fork.outs_0: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %ndw_out_out2.outs = hw.instance "ndw_out_out2" @handshake_ndwire_0(ins: %fork.outs_1: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %and.result = hw.instance "and" @handshake_andi_0(lhs: %ndw_in_in1.outs: !handshake.channel<i1>, rhs: %ndw_in_in2.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (result: !handshake.channel<i1>)
    %fork.outs_0, %fork.outs_1 = hw.instance "fork" @handshake_fork_0(ins: %and.result: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs_0: !handshake.channel<i1>, outs_1: !handshake.channel<i1>)
    hw.output %ndw_out_out1.outs, %ndw_out_out2.outs : !handshake.channel<i1>, !handshake.channel<i1>
  }
  hw.module.extern @handshake_ndwire_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i1>) attributes {hw.name = "handshake.ndwire", hw.parameters = {DATA_TYPE = !handshake.channel<i1>}}
  hw.module.extern @handshake_andi_0(in %lhs : !handshake.channel<i1>, in %rhs : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out result : !handshake.channel<i1>) attributes {hw.name = "handshake.andi", hw.parameters = {DATA_TYPE = !handshake.channel<i1>, INTERNAL_DELAY = "0.0"}}
  hw.module.extern @handshake_fork_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs_0 : !handshake.channel<i1>, out outs_1 : !handshake.channel<i1>) attributes {hw.name = "handshake.fork", hw.parameters = {DATA_TYPE = !handshake.channel<i1>, SIZE = 2 : ui32}}
  hw.module @andForkSwap_lhs_wrapper(in %in2 : !handshake.channel<i1>, in %in1 : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out out1 : !handshake.channel<i1>, out out2 : !handshake.channel<i1>) {
    %andForkSwap_lhs_wrapped.out1, %andForkSwap_lhs_wrapped.out2 = hw.instance "andForkSwap_lhs_wrapped" @andForkSwap_lhs(in2: %in2: !handshake.channel<i1>, in1: %in1: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (out1: !handshake.channel<i1>, out2: !handshake.channel<i1>)
    hw.output %andForkSwap_lhs_wrapped.out1, %andForkSwap_lhs_wrapped.out2 : !handshake.channel<i1>, !handshake.channel<i1>
  }
}

