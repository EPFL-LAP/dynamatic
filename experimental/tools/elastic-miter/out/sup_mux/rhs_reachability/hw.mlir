module {
  hw.module @sup_mux_rhs(in %loopLiveIn : !handshake.channel<i1>, in %iterLiveOut : !handshake.channel<i1>, in %oldContinue : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out iterLiveIn : !handshake.channel<i1>) {
    %source.outs = hw.instance "source" @handshake_source_0(clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.control<>)
    %constant.outs = hw.instance "constant" @handshake_constant_0(ctrl: %source.outs: !handshake.control<>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %ctrl_mux.outs = hw.instance "ctrl_mux" @handshake_mux_0(index: %newInit2.outs: !handshake.channel<i1>, ins_0: %constant.outs: !handshake.channel<i1>, ins_1: %oldContinue: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %fork_continue.outs_0, %fork_continue.outs_1, %fork_continue.outs_2 = hw.instance "fork_continue" @handshake_fork_0(ins: %ctrl_mux.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs_0: !handshake.channel<i1>, outs_1: !handshake.channel<i1>, outs_2: !handshake.channel<i1>)
    %newInit.outs = hw.instance "newInit" @handshake_init_0(ins: %fork_continue.outs_0: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %buff.outs = hw.instance "buff" @handshake_buffer_0(ins: %fork_continue.outs_1: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %newInit2.outs = hw.instance "newInit2" @handshake_init_0(ins: %buff.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %data_mux.outs = hw.instance "data_mux" @handshake_mux_0(index: %newInit.outs: !handshake.channel<i1>, ins_0: %loopLiveIn: !handshake.channel<i1>, ins_1: %iterLiveOut: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %passer.result = hw.instance "passer" @handshake_passer_0(data: %data_mux.outs: !handshake.channel<i1>, ctrl: %fork_continue.outs_2: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (result: !handshake.channel<i1>)
    hw.output %passer.result : !handshake.channel<i1>
  }
  hw.module.extern @handshake_source_0(in %clk : i1, in %rst : i1, out outs : !handshake.control<>) attributes {hw.name = "handshake.source", hw.parameters = {}}
  hw.module.extern @handshake_constant_0(in %ctrl : !handshake.control<>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i1>) attributes {hw.name = "handshake.constant", hw.parameters = {DATA_WIDTH = 1 : ui32, VALUE = "1"}}
  hw.module.extern @handshake_mux_0(in %index : !handshake.channel<i1>, in %ins_0 : !handshake.channel<i1>, in %ins_1 : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i1>) attributes {hw.name = "handshake.mux", hw.parameters = {DATA_TYPE = !handshake.channel<i1>, SELECT_TYPE = !handshake.channel<i1>, SIZE = 2 : ui32}}
  hw.module.extern @handshake_fork_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs_0 : !handshake.channel<i1>, out outs_1 : !handshake.channel<i1>, out outs_2 : !handshake.channel<i1>) attributes {hw.name = "handshake.fork", hw.parameters = {DATA_TYPE = !handshake.channel<i1>, SIZE = 3 : ui32}}
  hw.module.extern @handshake_init_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i1>) attributes {hw.name = "handshake.init", hw.parameters = {INIT_TOKEN = 0 : ui32}}
  hw.module.extern @handshake_buffer_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i1>) attributes {hw.name = "handshake.buffer", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", DATA_TYPE = !handshake.channel<i1>, DEBUG_COUNTER = false, NUM_SLOTS = 1 : ui32}}
  hw.module.extern @handshake_passer_0(in %data : !handshake.channel<i1>, in %ctrl : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out result : !handshake.channel<i1>) attributes {hw.name = "handshake.passer", hw.parameters = {}}
  hw.module @sup_mux_rhs_wrapper(in %loopLiveIn : !handshake.channel<i1>, in %iterLiveOut : !handshake.channel<i1>, in %oldContinue : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out iterLiveIn : !handshake.channel<i1>) {
    %sup_mux_rhs_wrapped.iterLiveIn = hw.instance "sup_mux_rhs_wrapped" @sup_mux_rhs(loopLiveIn: %loopLiveIn: !handshake.channel<i1>, iterLiveOut: %iterLiveOut: !handshake.channel<i1>, oldContinue: %oldContinue: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (iterLiveIn: !handshake.channel<i1>)
    hw.output %sup_mux_rhs_wrapped.iterLiveIn : !handshake.channel<i1>
  }
}

