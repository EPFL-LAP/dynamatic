module {
  hw.module @general_sup_mumux_lhs(in %dataT : !handshake.channel<i1>, in %dataF : !handshake.channel<i1>, in %cond : !handshake.channel<i1>, in %ctrl : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out ctrlRes : !handshake.channel<i1>, out dataRes : !handshake.channel<i1>) {
    %ctrl_fork.outs_0, %ctrl_fork.outs_1 = hw.instance "ctrl_fork" @handshake_fork_0(ins: %ctrl: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs_0: !handshake.channel<i1>, outs_1: !handshake.channel<i1>)
    %cond_passer.result = hw.instance "cond_passer" @handshake_passer_0(data: %cond: !handshake.channel<i1>, ctrl: %ctrl_fork.outs_0: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (result: !handshake.channel<i1>)
    %dataT_passer.result = hw.instance "dataT_passer" @handshake_passer_0(data: %dataT: !handshake.channel<i1>, ctrl: %ctrl_fork.outs_1: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (result: !handshake.channel<i1>)
    %ri_fork.outs_0, %ri_fork.outs_1 = hw.instance "ri_fork" @handshake_fork_0(ins: %ri.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs_0: !handshake.channel<i1>, outs_1: !handshake.channel<i1>)
    %ri_buff.outs = hw.instance "ri_buff" @handshake_buffer_0(ins: %ri_fork.outs_0: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %initop.outs = hw.instance "initop" @handshake_init_0(ins: %ri_buff.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %init_fork.outs_0, %init_fork.outs_1 = hw.instance "init_fork" @handshake_fork_0(ins: %initop.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs_0: !handshake.channel<i1>, outs_1: !handshake.channel<i1>)
    %source1.outs = hw.instance "source1" @handshake_source_0(clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.control<>)
    %constant1.outs = hw.instance "constant1" @handshake_constant_0(ctrl: %source1.outs: !handshake.control<>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %ri.outs = hw.instance "ri" @handshake_mux_0(index: %init_fork.outs_0: !handshake.channel<i1>, ins_0: %constant1.outs: !handshake.channel<i1>, ins_1: %cond_passer.result: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %data_mux.outs = hw.instance "data_mux" @handshake_mux_0(index: %init_fork.outs_1: !handshake.channel<i1>, ins_0: %dataF: !handshake.channel<i1>, ins_1: %dataT_passer.result: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    hw.output %ri_fork.outs_1, %data_mux.outs : !handshake.channel<i1>, !handshake.channel<i1>
  }
  hw.module.extern @handshake_fork_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs_0 : !handshake.channel<i1>, out outs_1 : !handshake.channel<i1>) attributes {hw.name = "handshake.fork", hw.parameters = {DATA_TYPE = !handshake.channel<i1>, SIZE = 2 : ui32}}
  hw.module.extern @handshake_passer_0(in %data : !handshake.channel<i1>, in %ctrl : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out result : !handshake.channel<i1>) attributes {hw.name = "handshake.passer", hw.parameters = {}}
  hw.module.extern @handshake_buffer_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i1>) attributes {hw.name = "handshake.buffer", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", DATA_TYPE = !handshake.channel<i1>, DEBUG_COUNTER = false, NUM_SLOTS = 1 : ui32}}
  hw.module.extern @handshake_init_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i1>) attributes {hw.name = "handshake.init", hw.parameters = {INIT_TOKEN = 0 : ui32}}
  hw.module.extern @handshake_source_0(in %clk : i1, in %rst : i1, out outs : !handshake.control<>) attributes {hw.name = "handshake.source", hw.parameters = {}}
  hw.module.extern @handshake_constant_0(in %ctrl : !handshake.control<>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i1>) attributes {hw.name = "handshake.constant", hw.parameters = {DATA_WIDTH = 1 : ui32, VALUE = "1"}}
  hw.module.extern @handshake_mux_0(in %index : !handshake.channel<i1>, in %ins_0 : !handshake.channel<i1>, in %ins_1 : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i1>) attributes {hw.name = "handshake.mux", hw.parameters = {DATA_TYPE = !handshake.channel<i1>, SELECT_TYPE = !handshake.channel<i1>, SIZE = 2 : ui32}}
  hw.module @general_sup_mumux_lhs_wrapper(in %dataT : !handshake.channel<i1>, in %dataF : !handshake.channel<i1>, in %cond : !handshake.channel<i1>, in %ctrl : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out ctrlRes : !handshake.channel<i1>, out dataRes : !handshake.channel<i1>) {
    %general_sup_mumux_lhs_wrapped.ctrlRes, %general_sup_mumux_lhs_wrapped.dataRes = hw.instance "general_sup_mumux_lhs_wrapped" @general_sup_mumux_lhs(dataT: %dataT: !handshake.channel<i1>, dataF: %dataF: !handshake.channel<i1>, cond: %cond: !handshake.channel<i1>, ctrl: %ctrl: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (ctrlRes: !handshake.channel<i1>, dataRes: !handshake.channel<i1>)
    hw.output %general_sup_mumux_lhs_wrapped.ctrlRes, %general_sup_mumux_lhs_wrapped.dataRes : !handshake.channel<i1>, !handshake.channel<i1>
  }
}

