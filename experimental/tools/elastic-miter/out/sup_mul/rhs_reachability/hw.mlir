module {
  hw.module @sup_mul_rhs(in %A_in : !handshake.channel<i1>, in %B_in : !handshake.channel<i1>, in %Cond : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out C_out : !handshake.channel<i1>) {
    %muli.result = hw.instance "muli" @handshake_muli_0(lhs: %A_in: !handshake.channel<i1>, rhs: %B_in: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (result: !handshake.channel<i1>)
    %passer1.result = hw.instance "passer1" @handshake_passer_0(data: %muli.result: !handshake.channel<i1>, ctrl: %Cond: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (result: !handshake.channel<i1>)
    hw.output %passer1.result : !handshake.channel<i1>
  }
  hw.module.extern @handshake_muli_0(in %lhs : !handshake.channel<i1>, in %rhs : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out result : !handshake.channel<i1>) attributes {hw.name = "handshake.muli", hw.parameters = {DATA_TYPE = !handshake.channel<i1>}}
  hw.module.extern @handshake_passer_0(in %data : !handshake.channel<i1>, in %ctrl : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out result : !handshake.channel<i1>) attributes {hw.name = "handshake.passer", hw.parameters = {}}
  hw.module @sup_mul_rhs_wrapper(in %A_in : !handshake.channel<i1>, in %B_in : !handshake.channel<i1>, in %Cond : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out C_out : !handshake.channel<i1>) {
    %sup_mul_rhs_wrapped.C_out = hw.instance "sup_mul_rhs_wrapped" @sup_mul_rhs(A_in: %A_in: !handshake.channel<i1>, B_in: %B_in: !handshake.channel<i1>, Cond: %Cond: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (C_out: !handshake.channel<i1>)
    hw.output %sup_mul_rhs_wrapped.C_out : !handshake.channel<i1>
  }
}

