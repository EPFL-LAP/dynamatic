module {
  hw.module @interpolatorForkSwap_lhs(in %long : !handshake.channel<i1>, in %short : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out out1 : !handshake.channel<i1>, out out2 : !handshake.channel<i1>) {
    %ndw_in_long.outs = hw.instance "ndw_in_long" @handshake_ndwire_0(ins: %long: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %ndw_in_short.outs = hw.instance "ndw_in_short" @handshake_ndwire_0(ins: %short: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %ndw_out_out1.outs = hw.instance "ndw_out_out1" @handshake_ndwire_0(ins: %fork.outs_0: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %ndw_out_out2.outs = hw.instance "ndw_out_out2" @handshake_ndwire_0(ins: %fork.outs_1: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %interpolator.result = hw.instance "interpolator" @handshake_spec_v2_interpolator_0(short: %ndw_in_short.outs: !handshake.channel<i1>, long: %ndw_in_long.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (result: !handshake.channel<i1>)
    %fork.outs_0, %fork.outs_1 = hw.instance "fork" @handshake_fork_0(ins: %interpolator.result: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs_0: !handshake.channel<i1>, outs_1: !handshake.channel<i1>)
    hw.output %ndw_out_out1.outs, %ndw_out_out2.outs : !handshake.channel<i1>, !handshake.channel<i1>
  }
  hw.module.extern @handshake_ndwire_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i1>) attributes {hw.name = "handshake.ndwire", hw.parameters = {DATA_TYPE = !handshake.channel<i1>}}
  hw.module.extern @handshake_spec_v2_interpolator_0(in %short : !handshake.channel<i1>, in %long : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out result : !handshake.channel<i1>) attributes {hw.name = "handshake.spec_v2_interpolator", hw.parameters = {}}
  hw.module.extern @handshake_fork_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs_0 : !handshake.channel<i1>, out outs_1 : !handshake.channel<i1>) attributes {hw.name = "handshake.fork", hw.parameters = {DATA_TYPE = !handshake.channel<i1>, SIZE = 2 : ui32}}
  hw.module @interpolatorForkSwap_lhs_wrapper(in %long : !handshake.channel<i1>, in %short : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out out1 : !handshake.channel<i1>, out out2 : !handshake.channel<i1>) {
    %interpolatorForkSwap_lhs_wrapped.out1, %interpolatorForkSwap_lhs_wrapped.out2 = hw.instance "interpolatorForkSwap_lhs_wrapped" @interpolatorForkSwap_lhs(long: %long: !handshake.channel<i1>, short: %short: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (out1: !handshake.channel<i1>, out2: !handshake.channel<i1>)
    hw.output %interpolatorForkSwap_lhs_wrapped.out1, %interpolatorForkSwap_lhs_wrapped.out2 : !handshake.channel<i1>, !handshake.channel<i1>
  }
}

