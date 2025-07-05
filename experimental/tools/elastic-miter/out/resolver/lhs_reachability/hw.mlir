module {
  hw.module @resolver_lhs(in %Actual : !handshake.channel<i1>, in %Generated : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out Confirm : !handshake.channel<i1>) {
    %ndw_in_Actual.outs = hw.instance "ndw_in_Actual" @handshake_ndwire_0(ins: %Actual: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %ndw_in_Generated.outs = hw.instance "ndw_in_Generated" @handshake_ndwire_0(ins: %Generated: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %ndw_out_Confirm.outs = hw.instance "ndw_out_Confirm" @handshake_ndwire_0(ins: %fork.outs_1: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %passer.result = hw.instance "passer" @handshake_passer_0(data: %ndw_in_Actual.outs: !handshake.channel<i1>, ctrl: %fork.outs_0: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (result: !handshake.channel<i1>)
    %buf1.outs = hw.instance "buf1" @handshake_buffer_0(ins: %passer.result: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %buf2.outs = hw.instance "buf2" @handshake_buffer_1(ins: %buf1.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %ri.outs = hw.instance "ri" @handshake_spec_v2_repeating_init_0(ins: %buf2.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %interpolator.result = hw.instance "interpolator" @handshake_spec_v2_interpolator_0(short: %ri.outs: !handshake.channel<i1>, long: %ndw_in_Generated.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (result: !handshake.channel<i1>)
    %fork.outs_0, %fork.outs_1 = hw.instance "fork" @handshake_fork_0(ins: %interpolator.result: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs_0: !handshake.channel<i1>, outs_1: !handshake.channel<i1>)
    hw.output %ndw_out_Confirm.outs : !handshake.channel<i1>
  }
  hw.module.extern @handshake_ndwire_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i1>) attributes {hw.name = "handshake.ndwire", hw.parameters = {DATA_TYPE = !handshake.channel<i1>}}
  hw.module.extern @handshake_passer_0(in %data : !handshake.channel<i1>, in %ctrl : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out result : !handshake.channel<i1>) attributes {hw.name = "handshake.passer", hw.parameters = {}}
  hw.module.extern @handshake_buffer_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i1>) attributes {hw.name = "handshake.buffer", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", DATA_TYPE = !handshake.channel<i1>, NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}}
  hw.module.extern @handshake_buffer_1(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i1>) attributes {hw.name = "handshake.buffer", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", DATA_TYPE = !handshake.channel<i1>, NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}}
  hw.module.extern @handshake_spec_v2_repeating_init_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i1>) attributes {hw.name = "handshake.spec_v2_repeating_init", hw.parameters = {}}
  hw.module.extern @handshake_spec_v2_interpolator_0(in %short : !handshake.channel<i1>, in %long : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out result : !handshake.channel<i1>) attributes {hw.name = "handshake.spec_v2_interpolator", hw.parameters = {}}
  hw.module.extern @handshake_fork_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs_0 : !handshake.channel<i1>, out outs_1 : !handshake.channel<i1>) attributes {hw.name = "handshake.fork", hw.parameters = {DATA_TYPE = !handshake.channel<i1>, SIZE = 2 : ui32}}
  hw.module @resolver_lhs_wrapper(in %Actual : !handshake.channel<i1>, in %Generated : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out Confirm : !handshake.channel<i1>) {
    %resolver_lhs_wrapped.Confirm = hw.instance "resolver_lhs_wrapped" @resolver_lhs(Actual: %Actual: !handshake.channel<i1>, Generated: %Generated: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (Confirm: !handshake.channel<i1>)
    hw.output %resolver_lhs_wrapped.Confirm : !handshake.channel<i1>
  }
}

