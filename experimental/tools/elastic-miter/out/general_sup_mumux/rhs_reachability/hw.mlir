module {
  hw.module @general_sup_mumux_rhs(in %dataT : !handshake.channel<i1>, in %dataF : !handshake.channel<i1>, in %cond : !handshake.channel<i1>, in %ctrl : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out ctrlRes : !handshake.channel<i1>, out dataRes : !handshake.channel<i1>) {
    %ctrl_fork.outs_0, %ctrl_fork.outs_1 = hw.instance "ctrl_fork" @handshake_fork_0(ins: %ctrl: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs_0: !handshake.channel<i1>, outs_1: !handshake.channel<i1>)
    %ctrl_not.outs = hw.instance "ctrl_not" @handshake_not_0(ins: %ctrl_fork.outs_0: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %ori.result = hw.instance "ori" @handshake_ori_0(lhs: %ctrl_not.outs: !handshake.channel<i1>, rhs: %cond: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (result: !handshake.channel<i1>)
    %ri_fork.outs_0, %ri_fork.outs_1 = hw.instance "ri_fork" @handshake_fork_0(ins: %ri.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs_0: !handshake.channel<i1>, outs_1: !handshake.channel<i1>)
    %ri_buff.outs = hw.instance "ri_buff" @handshake_buffer_0(ins: %ri_fork.outs_0: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %initop.outs = hw.instance "initop" @handshake_init_0(ins: %ri_buff.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %init_fork.outs_0, %init_fork.outs_1, %init_fork.outs_2 = hw.instance "init_fork" @handshake_fork_1(ins: %initop.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs_0: !handshake.channel<i1>, outs_1: !handshake.channel<i1>, outs_2: !handshake.channel<i1>)
    %source1.outs = hw.instance "source1" @handshake_source_0(clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.control<>)
    %constant1.outs = hw.instance "constant1" @handshake_constant_0(ctrl: %source1.outs: !handshake.control<>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %ri.outs = hw.instance "ri" @handshake_mux_0(index: %init_fork.outs_0: !handshake.channel<i1>, ins_0: %constant1.outs: !handshake.channel<i1>, ins_1: %ori.result: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %data_mux.outs = hw.instance "data_mux" @handshake_mux_0(index: %init_fork.outs_1: !handshake.channel<i1>, ins_0: %dataF: !handshake.channel<i1>, ins_1: %dataT: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %source2.outs = hw.instance "source2" @handshake_source_0(clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.control<>)
    %constant2.outs = hw.instance "constant2" @handshake_constant_0(ctrl: %source2.outs: !handshake.control<>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %ctrl_mux.outs = hw.instance "ctrl_mux" @handshake_mux_0(index: %init_fork.outs_2: !handshake.channel<i1>, ins_0: %constant2.outs: !handshake.channel<i1>, ins_1: %ctrl_fork.outs_1: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %ctrl_muxed_fork.outs_0, %ctrl_muxed_fork.outs_1 = hw.instance "ctrl_muxed_fork" @handshake_fork_0(ins: %ctrl_mux.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs_0: !handshake.channel<i1>, outs_1: !handshake.channel<i1>)
    %ctrlRes_passer.result = hw.instance "ctrlRes_passer" @handshake_passer_0(data: %ri_fork.outs_1: !handshake.channel<i1>, ctrl: %ctrl_muxed_fork.outs_0: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (result: !handshake.channel<i1>)
    %dataRes_passer.result = hw.instance "dataRes_passer" @handshake_passer_0(data: %data_mux.outs: !handshake.channel<i1>, ctrl: %ctrl_muxed_fork.outs_1: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (result: !handshake.channel<i1>)
    hw.output %ctrlRes_passer.result, %dataRes_passer.result : !handshake.channel<i1>, !handshake.channel<i1>
  }
  hw.module.extern @handshake_fork_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs_0 : !handshake.channel<i1>, out outs_1 : !handshake.channel<i1>) attributes {hw.name = "handshake.fork", hw.parameters = {DATA_TYPE = !handshake.channel<i1>, SIZE = 2 : ui32}}
  hw.module.extern @handshake_not_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i1>) attributes {hw.name = "handshake.not", hw.parameters = {DATA_TYPE = !handshake.channel<i1>}}
  hw.module.extern @handshake_ori_0(in %lhs : !handshake.channel<i1>, in %rhs : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out result : !handshake.channel<i1>) attributes {hw.name = "handshake.ori", hw.parameters = {DATA_TYPE = !handshake.channel<i1>}}
  hw.module.extern @handshake_buffer_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i1>) attributes {hw.name = "handshake.buffer", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", DATA_TYPE = !handshake.channel<i1>, DEBUG_COUNTER = false, NUM_SLOTS = 1 : ui32}}
  hw.module.extern @handshake_init_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i1>) attributes {hw.name = "handshake.init", hw.parameters = {INIT_TOKEN = 0 : ui32}}
  hw.module.extern @handshake_fork_1(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs_0 : !handshake.channel<i1>, out outs_1 : !handshake.channel<i1>, out outs_2 : !handshake.channel<i1>) attributes {hw.name = "handshake.fork", hw.parameters = {DATA_TYPE = !handshake.channel<i1>, SIZE = 3 : ui32}}
  hw.module.extern @handshake_source_0(in %clk : i1, in %rst : i1, out outs : !handshake.control<>) attributes {hw.name = "handshake.source", hw.parameters = {}}
  hw.module.extern @handshake_constant_0(in %ctrl : !handshake.control<>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i1>) attributes {hw.name = "handshake.constant", hw.parameters = {DATA_WIDTH = 1 : ui32, VALUE = "1"}}
  hw.module.extern @handshake_mux_0(in %index : !handshake.channel<i1>, in %ins_0 : !handshake.channel<i1>, in %ins_1 : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i1>) attributes {hw.name = "handshake.mux", hw.parameters = {DATA_TYPE = !handshake.channel<i1>, SELECT_TYPE = !handshake.channel<i1>, SIZE = 2 : ui32}}
  hw.module.extern @handshake_passer_0(in %data : !handshake.channel<i1>, in %ctrl : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out result : !handshake.channel<i1>) attributes {hw.name = "handshake.passer", hw.parameters = {}}
  hw.module @general_sup_mumux_rhs_wrapper(in %dataT : !handshake.channel<i1>, in %dataF : !handshake.channel<i1>, in %cond : !handshake.channel<i1>, in %ctrl : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out ctrlRes : !handshake.channel<i1>, out dataRes : !handshake.channel<i1>) {
    %general_sup_mumux_rhs_wrapped.ctrlRes, %general_sup_mumux_rhs_wrapped.dataRes = hw.instance "general_sup_mumux_rhs_wrapped" @general_sup_mumux_rhs(dataT: %dataT: !handshake.channel<i1>, dataF: %dataF: !handshake.channel<i1>, cond: %cond: !handshake.channel<i1>, ctrl: %ctrl: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (ctrlRes: !handshake.channel<i1>, dataRes: !handshake.channel<i1>)
    hw.output %general_sup_mumux_rhs_wrapped.ctrlRes, %general_sup_mumux_rhs_wrapped.dataRes : !handshake.channel<i1>, !handshake.channel<i1>
  }
}

