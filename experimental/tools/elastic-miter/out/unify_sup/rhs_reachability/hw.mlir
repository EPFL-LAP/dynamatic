module {
  hw.module @unify_sup_rhs(in %A_in : !handshake.channel<i1>, in %Cond1 : !handshake.channel<i1>, in %Cond2 : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out B_out : !handshake.channel<i1>) {
    %andi.result = hw.instance "andi" @handshake_andi_0(lhs: %Cond1: !handshake.channel<i1>, rhs: %Cond2: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (result: !handshake.channel<i1>)
    %passer.result = hw.instance "passer" @handshake_passer_0(data: %A_in: !handshake.channel<i1>, ctrl: %andi.result: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (result: !handshake.channel<i1>)
    hw.output %passer.result : !handshake.channel<i1>
  }
  hw.module.extern @handshake_andi_0(in %lhs : !handshake.channel<i1>, in %rhs : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out result : !handshake.channel<i1>) attributes {hw.name = "handshake.andi", hw.parameters = {DATA_TYPE = !handshake.channel<i1>}}
  hw.module.extern @handshake_passer_0(in %data : !handshake.channel<i1>, in %ctrl : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out result : !handshake.channel<i1>) attributes {hw.name = "handshake.passer", hw.parameters = {}}
  hw.module @unify_sup_rhs_wrapper(in %A_in : !handshake.channel<i1>, in %Cond1 : !handshake.channel<i1>, in %Cond2 : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out B_out : !handshake.channel<i1>) {
    %unify_sup_rhs_wrapped.B_out = hw.instance "unify_sup_rhs_wrapped" @unify_sup_rhs(A_in: %A_in: !handshake.channel<i1>, Cond1: %Cond1: !handshake.channel<i1>, Cond2: %Cond2: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (B_out: !handshake.channel<i1>)
    hw.output %unify_sup_rhs_wrapped.B_out : !handshake.channel<i1>
  }
}

