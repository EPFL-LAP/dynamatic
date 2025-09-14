module {
  hw.module @simpleInduction_lhs(in %A_in : !handshake.channel<i1>, in %C_in : !handshake.channel<i1>, in %D_in : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out B_out : !handshake.channel<i1>) {
    %p1.result = hw.instance "p1" @handshake_passer_0(data: %A_in: !handshake.channel<i1>, ctrl: %C_in: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (result: !handshake.channel<i1>)
    %p2.result = hw.instance "p2" @handshake_passer_0(data: %p1.result: !handshake.channel<i1>, ctrl: %D_in: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (result: !handshake.channel<i1>)
    hw.output %p2.result : !handshake.channel<i1>
  }
  hw.module.extern @handshake_passer_0(in %data : !handshake.channel<i1>, in %ctrl : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out result : !handshake.channel<i1>) attributes {hw.name = "handshake.passer", hw.parameters = {}}
  hw.module @simpleInduction_lhs_wrapper(in %A_in : !handshake.channel<i1>, in %C_in : !handshake.channel<i1>, in %D_in : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out B_out : !handshake.channel<i1>) {
    %simpleInduction_lhs_wrapped.B_out = hw.instance "simpleInduction_lhs_wrapped" @simpleInduction_lhs(A_in: %A_in: !handshake.channel<i1>, C_in: %C_in: !handshake.channel<i1>, D_in: %D_in: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (B_out: !handshake.channel<i1>)
    hw.output %simpleInduction_lhs_wrapped.B_out : !handshake.channel<i1>
  }
}

