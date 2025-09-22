module {
  hw.module @sup_gamma_mux2_lhs(in %cond : !handshake.channel<i1>, in %dataT : !handshake.channel<i1>, in %dataF : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out res : !handshake.channel<i1>) {
    %cond_fork.outs_0, %cond_fork.outs_1, %cond_fork.outs_2 = hw.instance "cond_fork" @handshake_fork_0(ins: %cond: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs_0: !handshake.channel<i1>, outs_1: !handshake.channel<i1>, outs_2: !handshake.channel<i1>)
    %dataT_passer.result = hw.instance "dataT_passer" @handshake_passer_0(data: %dataT: !handshake.channel<i1>, ctrl: %cond_fork.outs_0: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (result: !handshake.channel<i1>)
    %not.outs = hw.instance "not" @handshake_not_0(ins: %cond_fork.outs_1: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %dataF_passer.result = hw.instance "dataF_passer" @handshake_passer_0(data: %dataF: !handshake.channel<i1>, ctrl: %not.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (result: !handshake.channel<i1>)
    %res_mux.outs = hw.instance "res_mux" @handshake_mux_0(index: %cond_fork.outs_2: !handshake.channel<i1>, ins_0: %dataF_passer.result: !handshake.channel<i1>, ins_1: %dataT_passer.result: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    hw.output %res_mux.outs : !handshake.channel<i1>
  }
  hw.module.extern @handshake_fork_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs_0 : !handshake.channel<i1>, out outs_1 : !handshake.channel<i1>, out outs_2 : !handshake.channel<i1>) attributes {hw.name = "handshake.fork", hw.parameters = {DATA_TYPE = !handshake.channel<i1>, SIZE = 3 : ui32}}
  hw.module.extern @handshake_passer_0(in %data : !handshake.channel<i1>, in %ctrl : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out result : !handshake.channel<i1>) attributes {hw.name = "handshake.passer", hw.parameters = {}}
  hw.module.extern @handshake_not_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i1>) attributes {hw.name = "handshake.not", hw.parameters = {DATA_TYPE = !handshake.channel<i1>}}
  hw.module.extern @handshake_mux_0(in %index : !handshake.channel<i1>, in %ins_0 : !handshake.channel<i1>, in %ins_1 : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i1>) attributes {hw.name = "handshake.mux", hw.parameters = {DATA_TYPE = !handshake.channel<i1>, SELECT_TYPE = !handshake.channel<i1>, SIZE = 2 : ui32}}
  hw.module @sup_gamma_mux2_lhs_wrapper(in %cond : !handshake.channel<i1>, in %dataT : !handshake.channel<i1>, in %dataF : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out res : !handshake.channel<i1>) {
    %sup_gamma_mux2_lhs_wrapped.res = hw.instance "sup_gamma_mux2_lhs_wrapped" @sup_gamma_mux2_lhs(cond: %cond: !handshake.channel<i1>, dataT: %dataT: !handshake.channel<i1>, dataF: %dataF: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (res: !handshake.channel<i1>)
    hw.output %sup_gamma_mux2_lhs_wrapped.res : !handshake.channel<i1>
  }
}

