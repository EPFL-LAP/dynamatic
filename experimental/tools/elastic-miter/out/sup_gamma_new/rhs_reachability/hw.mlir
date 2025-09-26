module {
  hw.module @sup_gamma_new_rhs(in %a1 : !handshake.channel<i1>, in %a2 : !handshake.channel<i1>, in %c1 : !handshake.channel<i1>, in %c2 : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out b : !handshake.channel<i1>) {
    %c2_fork.outs_0, %c2_fork.outs_1, %c2_fork.outs_2 = hw.instance "c2_fork" @handshake_fork_0(ins: %c2: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs_0: !handshake.channel<i1>, outs_1: !handshake.channel<i1>, outs_2: !handshake.channel<i1>)
    %a1_passer.result = hw.instance "a1_passer" @handshake_passer_0(data: %a1: !handshake.channel<i1>, ctrl: %c2_fork.outs_0: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (result: !handshake.channel<i1>)
    %not.outs = hw.instance "not" @handshake_not_0(ins: %c2_fork.outs_1: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %a2_passer.result = hw.instance "a2_passer" @handshake_passer_0(data: %a2: !handshake.channel<i1>, ctrl: %not.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (result: !handshake.channel<i1>)
    %mux.outs = hw.instance "mux" @handshake_mux_0(index: %c2_fork.outs_2: !handshake.channel<i1>, ins_0: %a2_passer.result: !handshake.channel<i1>, ins_1: %a1_passer.result: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %b_passer.result = hw.instance "b_passer" @handshake_passer_0(data: %mux.outs: !handshake.channel<i1>, ctrl: %c1: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (result: !handshake.channel<i1>)
    hw.output %b_passer.result : !handshake.channel<i1>
  }
  hw.module.extern @handshake_fork_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs_0 : !handshake.channel<i1>, out outs_1 : !handshake.channel<i1>, out outs_2 : !handshake.channel<i1>) attributes {hw.name = "handshake.fork", hw.parameters = {DATA_TYPE = !handshake.channel<i1>, SIZE = 3 : ui32}}
  hw.module.extern @handshake_passer_0(in %data : !handshake.channel<i1>, in %ctrl : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out result : !handshake.channel<i1>) attributes {hw.name = "handshake.passer", hw.parameters = {}}
  hw.module.extern @handshake_not_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i1>) attributes {hw.name = "handshake.not", hw.parameters = {DATA_TYPE = !handshake.channel<i1>}}
  hw.module.extern @handshake_mux_0(in %index : !handshake.channel<i1>, in %ins_0 : !handshake.channel<i1>, in %ins_1 : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i1>) attributes {hw.name = "handshake.mux", hw.parameters = {DATA_TYPE = !handshake.channel<i1>, SELECT_TYPE = !handshake.channel<i1>, SIZE = 2 : ui32}}
  hw.module @sup_gamma_new_rhs_wrapper(in %a1 : !handshake.channel<i1>, in %a2 : !handshake.channel<i1>, in %c1 : !handshake.channel<i1>, in %c2 : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out b : !handshake.channel<i1>) {
    %sup_gamma_new_rhs_wrapped.b = hw.instance "sup_gamma_new_rhs_wrapped" @sup_gamma_new_rhs(a1: %a1: !handshake.channel<i1>, a2: %a2: !handshake.channel<i1>, c1: %c1: !handshake.channel<i1>, c2: %c2: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (b: !handshake.channel<i1>)
    hw.output %sup_gamma_new_rhs_wrapped.b : !handshake.channel<i1>
  }
}

