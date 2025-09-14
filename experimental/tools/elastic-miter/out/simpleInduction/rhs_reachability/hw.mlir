module {
  hw.module @simpleInduction_rhs(in %A_in : !handshake.channel<i1>, in %C_in : !handshake.channel<i1>, in %D_in : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out B_out : !handshake.channel<i1>) {
    %source.outs = hw.instance "source" @handshake_source_0(clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.control<>)
    %constant.outs = hw.instance "constant" @handshake_constant_0(ctrl: %source.outs: !handshake.control<>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %mux.outs = hw.instance "mux" @handshake_mux_0(index: %C_in: !handshake.channel<i1>, ins_0: %constant.outs: !handshake.channel<i1>, ins_1: %D_in: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %p1.result = hw.instance "p1" @handshake_passer_0(data: %A_in: !handshake.channel<i1>, ctrl: %mux.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (result: !handshake.channel<i1>)
    hw.output %p1.result : !handshake.channel<i1>
  }
  hw.module.extern @handshake_source_0(in %clk : i1, in %rst : i1, out outs : !handshake.control<>) attributes {hw.name = "handshake.source", hw.parameters = {}}
  hw.module.extern @handshake_constant_0(in %ctrl : !handshake.control<>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i1>) attributes {hw.name = "handshake.constant", hw.parameters = {DATA_WIDTH = 1 : ui32, VALUE = "0"}}
  hw.module.extern @handshake_mux_0(in %index : !handshake.channel<i1>, in %ins_0 : !handshake.channel<i1>, in %ins_1 : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i1>) attributes {hw.name = "handshake.mux", hw.parameters = {DATA_TYPE = !handshake.channel<i1>, SELECT_TYPE = !handshake.channel<i1>, SIZE = 2 : ui32}}
  hw.module.extern @handshake_passer_0(in %data : !handshake.channel<i1>, in %ctrl : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out result : !handshake.channel<i1>) attributes {hw.name = "handshake.passer", hw.parameters = {}}
  hw.module @simpleInduction_rhs_wrapper(in %A_in : !handshake.channel<i1>, in %C_in : !handshake.channel<i1>, in %D_in : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out B_out : !handshake.channel<i1>) {
    %simpleInduction_rhs_wrapped.B_out = hw.instance "simpleInduction_rhs_wrapped" @simpleInduction_rhs(A_in: %A_in: !handshake.channel<i1>, C_in: %C_in: !handshake.channel<i1>, D_in: %D_in: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (B_out: !handshake.channel<i1>)
    hw.output %simpleInduction_rhs_wrapped.B_out : !handshake.channel<i1>
  }
}

