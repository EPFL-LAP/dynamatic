module {
  hw.module @introduce_ident_interpolator_rhs(in %A_in : !handshake.channel<i1>, in %clk : i1, in %rst : i1) {
    %ndw_in_A_in.outs = hw.instance "ndw_in_A_in" @handshake_ndwire_0(ins: %A_in: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %vm_fork_0.outs_0, %vm_fork_0.outs_1 = hw.instance "vm_fork_0" @handshake_fork_0(ins: %ndw_in_A_in.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs_0: !handshake.channel<i1>, outs_1: !handshake.channel<i1>)
    %interpolate.result = hw.instance "interpolate" @handshake_spec_v2_interpolator_0(short: %vm_fork_0.outs_0: !handshake.channel<i1>, long: %vm_fork_0.outs_1: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (result: !handshake.channel<i1>)
    hw.instance "sink" @handshake_sink_0(ins: %interpolate.result: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> ()
    hw.output
  }
  hw.module.extern @handshake_ndwire_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i1>) attributes {hw.name = "handshake.ndwire", hw.parameters = {DATA_TYPE = !handshake.channel<i1>}}
  hw.module.extern @handshake_fork_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs_0 : !handshake.channel<i1>, out outs_1 : !handshake.channel<i1>) attributes {hw.name = "handshake.fork", hw.parameters = {DATA_TYPE = !handshake.channel<i1>, SIZE = 2 : ui32}}
  hw.module.extern @handshake_spec_v2_interpolator_0(in %short : !handshake.channel<i1>, in %long : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out result : !handshake.channel<i1>) attributes {hw.name = "handshake.spec_v2_interpolator", hw.parameters = {}}
  hw.module.extern @handshake_sink_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1) attributes {hw.name = "handshake.sink", hw.parameters = {DATA_TYPE = !handshake.channel<i1>}}
  hw.module @introduce_ident_interpolator_rhs_wrapper(in %A_in : !handshake.channel<i1>, in %clk : i1, in %rst : i1) {
    hw.instance "introduce_ident_interpolator_rhs_wrapped" @introduce_ident_interpolator_rhs(A_in: %A_in: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> ()
    hw.output
  }
}

