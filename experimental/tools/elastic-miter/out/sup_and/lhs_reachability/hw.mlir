module {
  hw.module @sup_and_lhs(in %A_in : !handshake.channel<i1>, in %B_in : !handshake.channel<i1>, in %Cond : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out C_out : !handshake.channel<i1>) {
    %fork_cond.outs_0, %fork_cond.outs_1 = hw.instance "fork_cond" @handshake_fork_0(ins: %Cond: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs_0: !handshake.channel<i1>, outs_1: !handshake.channel<i1>)
    %passer1.result = hw.instance "passer1" @handshake_passer_0(data: %A_in: !handshake.channel<i1>, ctrl: %fork_cond.outs_0: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (result: !handshake.channel<i1>)
    %passer2.result = hw.instance "passer2" @handshake_passer_0(data: %B_in: !handshake.channel<i1>, ctrl: %fork_cond.outs_1: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (result: !handshake.channel<i1>)
    %andi.result = hw.instance "andi" @handshake_andi_0(lhs: %passer1.result: !handshake.channel<i1>, rhs: %passer2.result: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (result: !handshake.channel<i1>)
    hw.output %andi.result : !handshake.channel<i1>
  }
  hw.module.extern @handshake_fork_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs_0 : !handshake.channel<i1>, out outs_1 : !handshake.channel<i1>) attributes {hw.name = "handshake.fork", hw.parameters = {DATA_TYPE = !handshake.channel<i1>, SIZE = 2 : ui32}}
  hw.module.extern @handshake_passer_0(in %data : !handshake.channel<i1>, in %ctrl : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out result : !handshake.channel<i1>) attributes {hw.name = "handshake.passer", hw.parameters = {}}
  hw.module.extern @handshake_andi_0(in %lhs : !handshake.channel<i1>, in %rhs : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out result : !handshake.channel<i1>) attributes {hw.name = "handshake.andi", hw.parameters = {DATA_TYPE = !handshake.channel<i1>}}
  hw.module @sup_and_lhs_wrapper(in %A_in : !handshake.channel<i1>, in %B_in : !handshake.channel<i1>, in %Cond : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out C_out : !handshake.channel<i1>) {
    %sup_and_lhs_wrapped.C_out = hw.instance "sup_and_lhs_wrapped" @sup_and_lhs(A_in: %A_in: !handshake.channel<i1>, B_in: %B_in: !handshake.channel<i1>, Cond: %Cond: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (C_out: !handshake.channel<i1>)
    hw.output %sup_and_lhs_wrapped.C_out : !handshake.channel<i1>
  }
}

