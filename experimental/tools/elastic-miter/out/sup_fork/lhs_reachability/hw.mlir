module {
  hw.module @sup_fork_lhs(in %A_in : !handshake.channel<i1>, in %Cond : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out B_out : !handshake.channel<i1>, out C_out : !handshake.channel<i1>) {
    %passer1.result = hw.instance "passer1" @handshake_passer_0(data: %A_in: !handshake.channel<i1>, ctrl: %Cond: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (result: !handshake.channel<i1>)
    %fork.outs_0, %fork.outs_1 = hw.instance "fork" @handshake_fork_0(ins: %passer1.result: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs_0: !handshake.channel<i1>, outs_1: !handshake.channel<i1>)
    hw.output %fork.outs_0, %fork.outs_1 : !handshake.channel<i1>, !handshake.channel<i1>
  }
  hw.module.extern @handshake_passer_0(in %data : !handshake.channel<i1>, in %ctrl : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out result : !handshake.channel<i1>) attributes {hw.name = "handshake.passer", hw.parameters = {}}
  hw.module.extern @handshake_fork_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs_0 : !handshake.channel<i1>, out outs_1 : !handshake.channel<i1>) attributes {hw.name = "handshake.fork", hw.parameters = {DATA_TYPE = !handshake.channel<i1>, SIZE = 2 : ui32}}
  hw.module @sup_fork_lhs_wrapper(in %A_in : !handshake.channel<i1>, in %Cond : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out B_out : !handshake.channel<i1>, out C_out : !handshake.channel<i1>) {
    %sup_fork_lhs_wrapped.B_out, %sup_fork_lhs_wrapped.C_out = hw.instance "sup_fork_lhs_wrapped" @sup_fork_lhs(A_in: %A_in: !handshake.channel<i1>, Cond: %Cond: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (B_out: !handshake.channel<i1>, C_out: !handshake.channel<i1>)
    hw.output %sup_fork_lhs_wrapped.B_out, %sup_fork_lhs_wrapped.C_out : !handshake.channel<i1>, !handshake.channel<i1>
  }
}

