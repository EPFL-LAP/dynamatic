module {
  hw.module @e_rhs(in %D : !handshake.channel<i32>, in %C : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out A : !handshake.channel<i32>) {
    hw.instance "sink" @handshake_sink_0(ins: %C: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> ()
    hw.output %D : !handshake.channel<i32>
  }
  hw.module.extern @handshake_sink_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1) attributes {hw.name = "handshake.sink", hw.parameters = {DATA_TYPE = !handshake.channel<i1>}}
  hw.module @e_rhs_wrapper(in %D : !handshake.channel<i32>, in %C : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out A : !handshake.channel<i32>) {
    %e_rhs_wrapped.A = hw.instance "e_rhs_wrapped" @e_rhs(D: %D: !handshake.channel<i32>, C: %C: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (A: !handshake.channel<i32>)
    hw.output %e_rhs_wrapped.A : !handshake.channel<i32>
  }
}

