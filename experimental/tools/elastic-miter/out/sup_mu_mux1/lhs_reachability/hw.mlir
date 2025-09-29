module {
  hw.module @sup_mu_mux1_lhs(in %cond : !handshake.channel<i1>, in %dataT : !handshake.channel<i1>, in %dataF : !handshake.channel<i1>, in %ctrl : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out res : !handshake.channel<i1>) {
    %dataF_passer.result = hw.instance "dataF_passer" @handshake_passer_0(data: %dataF: !handshake.channel<i1>, ctrl: %ctrl: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (result: !handshake.channel<i1>)
    %initop.outs = hw.instance "initop" @handshake_init_0(ins: %cond: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %res_mux.outs = hw.instance "res_mux" @handshake_mux_0(index: %initop.outs: !handshake.channel<i1>, ins_0: %dataF_passer.result: !handshake.channel<i1>, ins_1: %dataT: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    hw.output %res_mux.outs : !handshake.channel<i1>
  }
  hw.module.extern @handshake_passer_0(in %data : !handshake.channel<i1>, in %ctrl : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out result : !handshake.channel<i1>) attributes {hw.name = "handshake.passer", hw.parameters = {}}
  hw.module.extern @handshake_init_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i1>) attributes {hw.name = "handshake.init", hw.parameters = {INIT_TOKEN = 0 : ui32}}
  hw.module.extern @handshake_mux_0(in %index : !handshake.channel<i1>, in %ins_0 : !handshake.channel<i1>, in %ins_1 : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i1>) attributes {hw.name = "handshake.mux", hw.parameters = {DATA_TYPE = !handshake.channel<i1>, SELECT_TYPE = !handshake.channel<i1>, SIZE = 2 : ui32}}
  hw.module @sup_mu_mux1_lhs_wrapper(in %cond : !handshake.channel<i1>, in %dataT : !handshake.channel<i1>, in %dataF : !handshake.channel<i1>, in %ctrl : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out res : !handshake.channel<i1>) {
    %sup_mu_mux1_lhs_wrapped.res = hw.instance "sup_mu_mux1_lhs_wrapped" @sup_mu_mux1_lhs(cond: %cond: !handshake.channel<i1>, dataT: %dataT: !handshake.channel<i1>, dataF: %dataF: !handshake.channel<i1>, ctrl: %ctrl: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (res: !handshake.channel<i1>)
    hw.output %sup_mu_mux1_lhs_wrapped.res : !handshake.channel<i1>
  }
}

