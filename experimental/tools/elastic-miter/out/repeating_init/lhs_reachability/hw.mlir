module {
  hw.module @repeating_init_lhs(in %A_in : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out B_out : !handshake.channel<i1>) {
    %spec_v2_repeating_init.outs = hw.instance "spec_v2_repeating_init" @handshake_spec_v2_repeating_init_0(ins: %A_in: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    hw.output %spec_v2_repeating_init.outs : !handshake.channel<i1>
  }
  hw.module.extern @handshake_spec_v2_repeating_init_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i1>) attributes {hw.name = "handshake.spec_v2_repeating_init", hw.parameters = {INIT_TOKEN = 1 : ui32}}
  hw.module @repeating_init_lhs_wrapper(in %A_in : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out B_out : !handshake.channel<i1>) {
    %repeating_init_lhs_wrapped.B_out = hw.instance "repeating_init_lhs_wrapped" @repeating_init_lhs(A_in: %A_in: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (B_out: !handshake.channel<i1>)
    hw.output %repeating_init_lhs_wrapped.B_out : !handshake.channel<i1>
  }
}

