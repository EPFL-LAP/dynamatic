module {
  hw.module @resolver_rhs(in %Actual : !handshake.channel<i1>, in %Generated : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out Confirm : !handshake.channel<i1>) {
    %ndw_in_Actual.outs = hw.instance "ndw_in_Actual" @handshake_ndwire_0(ins: %Actual: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %ndw_in_Generated.outs = hw.instance "ndw_in_Generated" @handshake_ndwire_0(ins: %Generated: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %ndw_out_Confirm.outs = hw.instance "ndw_out_Confirm" @handshake_ndwire_0(ins: %resolver.confirmSpec: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %resolver.confirmSpec = hw.instance "resolver" @handshake_spec_v2_resolver_0(actualCondition: %ndw_in_Actual.outs: !handshake.channel<i1>, generatedCondition: %ndw_in_Generated.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (confirmSpec: !handshake.channel<i1>)
    hw.output %ndw_out_Confirm.outs : !handshake.channel<i1>
  }
  hw.module.extern @handshake_ndwire_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i1>) attributes {hw.name = "handshake.ndwire", hw.parameters = {DATA_TYPE = !handshake.channel<i1>}}
  hw.module.extern @handshake_spec_v2_resolver_0(in %actualCondition : !handshake.channel<i1>, in %generatedCondition : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out confirmSpec : !handshake.channel<i1>) attributes {hw.name = "handshake.spec_v2_resolver", hw.parameters = {}}
  hw.module @resolver_rhs_wrapper(in %Actual : !handshake.channel<i1>, in %Generated : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out Confirm : !handshake.channel<i1>) {
    %resolver_rhs_wrapped.Confirm = hw.instance "resolver_rhs_wrapped" @resolver_rhs(Actual: %Actual: !handshake.channel<i1>, Generated: %Generated: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (Confirm: !handshake.channel<i1>)
    hw.output %resolver_rhs_wrapped.Confirm : !handshake.channel<i1>
  }
}

