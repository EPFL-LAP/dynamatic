module {
  hw.module @introduceIdentInterpolator_lhs(in %A_in : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out B_out : !handshake.channel<i1>) {
    %not1.outs = hw.instance "not1" @handshake_not_0(ins: %A_in: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %not2.outs = hw.instance "not2" @handshake_not_0(ins: %not1.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    hw.output %not2.outs : !handshake.channel<i1>
  }
  hw.module.extern @handshake_not_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i1>) attributes {hw.name = "handshake.not", hw.parameters = {DATA_TYPE = !handshake.channel<i1>}}
  hw.module @introduceIdentInterpolator_lhs_wrapper(in %A_in : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out B_out : !handshake.channel<i1>) {
    %introduceIdentInterpolator_lhs_wrapped.B_out = hw.instance "introduceIdentInterpolator_lhs_wrapped" @introduceIdentInterpolator_lhs(A_in: %A_in: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (B_out: !handshake.channel<i1>)
    hw.output %introduceIdentInterpolator_lhs_wrapped.B_out : !handshake.channel<i1>
  }
}

