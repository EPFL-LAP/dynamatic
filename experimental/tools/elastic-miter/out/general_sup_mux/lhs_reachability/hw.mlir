module {
  hw.module @general_sup_mux_lhs(in %sel : !handshake.channel<i1>, in %ctrl1 : !handshake.channel<i1>, in %ctrl2 : !handshake.channel<i1>, in %a_in : !handshake.channel<i1>, in %b_in : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out res : !handshake.channel<i1>) {
    %fork_sel.outs_0, %fork_sel.outs_1 = hw.instance "fork_sel" @handshake_fork_0(ins: %sel: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs_0: !handshake.channel<i1>, outs_1: !handshake.channel<i1>)
    %fork_ctrl1.outs_0, %fork_ctrl1.outs_1 = hw.instance "fork_ctrl1" @handshake_fork_0(ins: %ctrl1: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs_0: !handshake.channel<i1>, outs_1: !handshake.channel<i1>)
    %fork_ctrl2.outs_0, %fork_ctrl2.outs_1 = hw.instance "fork_ctrl2" @handshake_fork_0(ins: %ctrl2: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs_0: !handshake.channel<i1>, outs_1: !handshake.channel<i1>)
    %sel_mux.outs = hw.instance "sel_mux" @handshake_mux_0(index: %fork_sel.outs_0: !handshake.channel<i1>, ins_0: %fork_ctrl1.outs_0: !handshake.channel<i1>, ins_1: %fork_ctrl2.outs_0: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %sel_passer.result = hw.instance "sel_passer" @handshake_passer_0(data: %fork_sel.outs_1: !handshake.channel<i1>, ctrl: %sel_mux.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (result: !handshake.channel<i1>)
    %a_passer.result = hw.instance "a_passer" @handshake_passer_0(data: %a_in: !handshake.channel<i1>, ctrl: %fork_ctrl1.outs_1: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (result: !handshake.channel<i1>)
    %b_passer.result = hw.instance "b_passer" @handshake_passer_0(data: %b_in: !handshake.channel<i1>, ctrl: %fork_ctrl2.outs_1: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (result: !handshake.channel<i1>)
    %data_mux.outs = hw.instance "data_mux" @handshake_mux_0(index: %sel_passer.result: !handshake.channel<i1>, ins_0: %a_passer.result: !handshake.channel<i1>, ins_1: %b_passer.result: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    hw.output %data_mux.outs : !handshake.channel<i1>
  }
  hw.module.extern @handshake_fork_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs_0 : !handshake.channel<i1>, out outs_1 : !handshake.channel<i1>) attributes {hw.name = "handshake.fork", hw.parameters = {DATA_TYPE = !handshake.channel<i1>, SIZE = 2 : ui32}}
  hw.module.extern @handshake_mux_0(in %index : !handshake.channel<i1>, in %ins_0 : !handshake.channel<i1>, in %ins_1 : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i1>) attributes {hw.name = "handshake.mux", hw.parameters = {DATA_TYPE = !handshake.channel<i1>, SELECT_TYPE = !handshake.channel<i1>, SIZE = 2 : ui32}}
  hw.module.extern @handshake_passer_0(in %data : !handshake.channel<i1>, in %ctrl : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out result : !handshake.channel<i1>) attributes {hw.name = "handshake.passer", hw.parameters = {}}
  hw.module @general_sup_mux_lhs_wrapper(in %sel : !handshake.channel<i1>, in %ctrl1 : !handshake.channel<i1>, in %ctrl2 : !handshake.channel<i1>, in %a_in : !handshake.channel<i1>, in %b_in : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out res : !handshake.channel<i1>) {
    %general_sup_mux_lhs_wrapped.res = hw.instance "general_sup_mux_lhs_wrapped" @general_sup_mux_lhs(sel: %sel: !handshake.channel<i1>, ctrl1: %ctrl1: !handshake.channel<i1>, ctrl2: %ctrl2: !handshake.channel<i1>, a_in: %a_in: !handshake.channel<i1>, b_in: %b_in: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (res: !handshake.channel<i1>)
    hw.output %general_sup_mux_lhs_wrapped.res : !handshake.channel<i1>
  }
}

