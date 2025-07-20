module {
  hw.module @sup_mux_lhs(in %loopLiveIn : !handshake.channel<i1>, in %oldContinue : !handshake.channel<i1>, in %iterLiveOut : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out iterLiveIn : !handshake.channel<i1>) {
    %ndw_in_loopLiveIn.outs = hw.instance "ndw_in_loopLiveIn" @handshake_ndwire_0(ins: %loopLiveIn: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %ndw_in_oldContinue.outs = hw.instance "ndw_in_oldContinue" @handshake_ndwire_0(ins: %oldContinue: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %ndw_in_iterLiveOut.outs = hw.instance "ndw_in_iterLiveOut" @handshake_ndwire_0(ins: %iterLiveOut: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %ndw_out_iterLiveIn.outs = hw.instance "ndw_out_iterLiveIn" @handshake_ndwire_0(ins: %data_mux.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %vm_fork_1.outs_0, %vm_fork_1.outs_1 = hw.instance "vm_fork_1" @handshake_fork_0(ins: %ndw_in_oldContinue.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs_0: !handshake.channel<i1>, outs_1: !handshake.channel<i1>)
    %passer.result = hw.instance "passer" @handshake_passer_0(data: %ndw_in_iterLiveOut.outs: !handshake.channel<i1>, ctrl: %vm_fork_1.outs_0: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (result: !handshake.channel<i1>)
    %oldInit.outs = hw.instance "oldInit" @handshake_init_0(ins: %vm_fork_1.outs_1: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %data_mux.outs = hw.instance "data_mux" @handshake_mux_0(index: %oldInit.outs: !handshake.channel<i1>, ins_0: %ndw_in_loopLiveIn.outs: !handshake.channel<i1>, ins_1: %passer.result: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    hw.output %ndw_out_iterLiveIn.outs : !handshake.channel<i1>
  }
  hw.module.extern @handshake_ndwire_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i1>) attributes {hw.name = "handshake.ndwire", hw.parameters = {DATA_TYPE = !handshake.channel<i1>}}
  hw.module.extern @handshake_fork_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs_0 : !handshake.channel<i1>, out outs_1 : !handshake.channel<i1>) attributes {hw.name = "handshake.fork", hw.parameters = {DATA_TYPE = !handshake.channel<i1>, SIZE = 2 : ui32}}
  hw.module.extern @handshake_passer_0(in %data : !handshake.channel<i1>, in %ctrl : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out result : !handshake.channel<i1>) attributes {hw.name = "handshake.passer", hw.parameters = {}}
  hw.module.extern @handshake_init_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i1>) attributes {hw.name = "handshake.init", hw.parameters = {INIT_TOKEN = 0 : ui32}}
  hw.module.extern @handshake_mux_0(in %index : !handshake.channel<i1>, in %ins_0 : !handshake.channel<i1>, in %ins_1 : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i1>) attributes {hw.name = "handshake.mux", hw.parameters = {DATA_TYPE = !handshake.channel<i1>, SELECT_TYPE = !handshake.channel<i1>, SIZE = 2 : ui32}}
  hw.module @sup_mux_lhs_wrapper(in %loopLiveIn : !handshake.channel<i1>, in %oldContinue : !handshake.channel<i1>, in %iterLiveOut : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out iterLiveIn : !handshake.channel<i1>) {
    %sup_mux_lhs_wrapped.iterLiveIn = hw.instance "sup_mux_lhs_wrapped" @sup_mux_lhs(loopLiveIn: %loopLiveIn: !handshake.channel<i1>, oldContinue: %oldContinue: !handshake.channel<i1>, iterLiveOut: %iterLiveOut: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (iterLiveIn: !handshake.channel<i1>)
    hw.output %sup_mux_lhs_wrapped.iterLiveIn : !handshake.channel<i1>
  }
}

