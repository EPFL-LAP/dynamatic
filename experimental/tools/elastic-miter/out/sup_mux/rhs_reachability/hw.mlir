module {
  hw.module @j_rhs(in %In1 : !handshake.channel<i1>, in %In2 : !handshake.channel<i1>, in %Ctrl : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out Out1 : !handshake.channel<i1>) {
    %ndw_in_In1.outs = hw.instance "ndw_in_In1" @handshake_ndwire_0(ins: %In1: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %ndw_in_In2.outs = hw.instance "ndw_in_In2" @handshake_ndwire_0(ins: %In2: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %ndw_in_Ctrl.outs = hw.instance "ndw_in_Ctrl" @handshake_ndwire_0(ins: %Ctrl: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %ndw_out_Out1.outs = hw.instance "ndw_out_Out1" @handshake_ndwire_0(ins: %passer.result: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %ri.outs = hw.instance "ri" @handshake_spec_v2_repeating_init_0(ins: %ndw_in_Ctrl.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %fork_data_mux.outs_0, %fork_data_mux.outs_1 = hw.instance "fork_data_mux" @handshake_fork_0(ins: %ri.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs_0: !handshake.channel<i1>, outs_1: !handshake.channel<i1>)
    %init_buffer_ctrl.outs = hw.instance "init_buffer_ctrl" @handshake_init_0(ins: %fork_data_mux.outs_0: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %data_mux.outs = hw.instance "data_mux" @handshake_mux_0(index: %init_buffer_ctrl.outs: !handshake.channel<i1>, ins_0: %ndw_in_In2.outs: !handshake.channel<i1>, ins_1: %ndw_in_In1.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %passer.result = hw.instance "passer" @handshake_passer_0(data: %data_mux.outs: !handshake.channel<i1>, ctrl: %fork_data_mux.outs_1: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (result: !handshake.channel<i1>)
    hw.output %ndw_out_Out1.outs : !handshake.channel<i1>
  }
  hw.module.extern @handshake_ndwire_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i1>) attributes {hw.name = "handshake.ndwire", hw.parameters = {DATA_TYPE = !handshake.channel<i1>}}
  hw.module.extern @handshake_spec_v2_repeating_init_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i1>) attributes {hw.name = "handshake.spec_v2_repeating_init", hw.parameters = {}}
  hw.module.extern @handshake_fork_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs_0 : !handshake.channel<i1>, out outs_1 : !handshake.channel<i1>) attributes {hw.name = "handshake.fork", hw.parameters = {DATA_TYPE = !handshake.channel<i1>, SIZE = 2 : ui32}}
  hw.module.extern @handshake_init_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i1>) attributes {hw.name = "handshake.init", hw.parameters = {INITIAL_TOKEN = false, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}}
  hw.module.extern @handshake_mux_0(in %index : !handshake.channel<i1>, in %ins_0 : !handshake.channel<i1>, in %ins_1 : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i1>) attributes {hw.name = "handshake.mux", hw.parameters = {DATA_TYPE = !handshake.channel<i1>, SELECT_TYPE = !handshake.channel<i1>, SIZE = 2 : ui32}}
  hw.module.extern @handshake_passer_0(in %data : !handshake.channel<i1>, in %ctrl : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out result : !handshake.channel<i1>) attributes {hw.name = "handshake.passer", hw.parameters = {}}
  hw.module @j_rhs_wrapper(in %In1 : !handshake.channel<i1>, in %In2 : !handshake.channel<i1>, in %Ctrl : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out Out1 : !handshake.channel<i1>) {
    %j_rhs_wrapped.Out1 = hw.instance "j_rhs_wrapped" @j_rhs(In1: %In1: !handshake.channel<i1>, In2: %In2: !handshake.channel<i1>, Ctrl: %Ctrl: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (Out1: !handshake.channel<i1>)
    hw.output %j_rhs_wrapped.Out1 : !handshake.channel<i1>
  }
}

