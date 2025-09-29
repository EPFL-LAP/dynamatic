module {
  hw.module @sup_sup_lhs(in %A_in : !handshake.channel<i1>, in %Cond1 : !handshake.channel<i1>, in %Cond2 : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out B_out : !handshake.channel<i1>) {
    %fork_cond2.outs_0, %fork_cond2.outs_1 = hw.instance "fork_cond2" @handshake_fork_0(ins: %Cond2: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs_0: !handshake.channel<i1>, outs_1: !handshake.channel<i1>)
    %passer1.result = hw.instance "passer1" @handshake_passer_0(data: %Cond1: !handshake.channel<i1>, ctrl: %fork_cond2.outs_0: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (result: !handshake.channel<i1>)
    %passer2.result = hw.instance "passer2" @handshake_passer_0(data: %A_in: !handshake.channel<i1>, ctrl: %fork_cond2.outs_1: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (result: !handshake.channel<i1>)
    %passer3.result = hw.instance "passer3" @handshake_passer_0(data: %passer2.result: !handshake.channel<i1>, ctrl: %passer1.result: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (result: !handshake.channel<i1>)
    hw.output %passer3.result : !handshake.channel<i1>
  }
  hw.module.extern @handshake_fork_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs_0 : !handshake.channel<i1>, out outs_1 : !handshake.channel<i1>) attributes {hw.name = "handshake.fork", hw.parameters = {DATA_TYPE = !handshake.channel<i1>, SIZE = 2 : ui32}}
  hw.module.extern @handshake_passer_0(in %data : !handshake.channel<i1>, in %ctrl : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out result : !handshake.channel<i1>) attributes {hw.name = "handshake.passer", hw.parameters = {}}
  hw.module @sup_sup_lhs_wrapper(in %A_in : !handshake.channel<i1>, in %Cond1 : !handshake.channel<i1>, in %Cond2 : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out B_out : !handshake.channel<i1>) {
    %sup_sup_lhs_wrapped.B_out = hw.instance "sup_sup_lhs_wrapped" @sup_sup_lhs(A_in: %A_in: !handshake.channel<i1>, Cond1: %Cond1: !handshake.channel<i1>, Cond2: %Cond2: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (B_out: !handshake.channel<i1>)
    hw.output %sup_sup_lhs_wrapped.B_out : !handshake.channel<i1>
  }
}

