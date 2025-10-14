module {
  hw.module @mux_to_and_lhs(in %A_in : !handshake.channel<i1>, in %B_in : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out C_out : !handshake.channel<i1>) {
    %source.outs = hw.instance "source" @handshake_source_0(clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.control<>)
    %constant.outs = hw.instance "constant" @handshake_constant_0(ctrl: %source.outs: !handshake.control<>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %a_fork.outs_0, %a_fork.outs_1 = hw.instance "a_fork" @handshake_fork_0(ins: %A_in: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs_0: !handshake.channel<i1>, outs_1: !handshake.channel<i1>)
    %passer1.result = hw.instance "passer1" @handshake_passer_0(data: %B_in: !handshake.channel<i1>, ctrl: %a_fork.outs_0: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (result: !handshake.channel<i1>)
    %mux.outs = hw.instance "mux" @handshake_mux_0(index: %a_fork.outs_1: !handshake.channel<i1>, ins_0: %constant.outs: !handshake.channel<i1>, ins_1: %passer1.result: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    hw.output %mux.outs : !handshake.channel<i1>
  }
  hw.module.extern @handshake_source_0(in %clk : i1, in %rst : i1, out outs : !handshake.control<>) attributes {hw.name = "handshake.source", hw.parameters = {}}
  hw.module.extern @handshake_constant_0(in %ctrl : !handshake.control<>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i1>) attributes {hw.name = "handshake.constant", hw.parameters = {DATA_WIDTH = 1 : ui32, VALUE = "0"}}
  hw.module.extern @handshake_fork_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs_0 : !handshake.channel<i1>, out outs_1 : !handshake.channel<i1>) attributes {hw.name = "handshake.fork", hw.parameters = {DATA_TYPE = !handshake.channel<i1>, SIZE = 2 : ui32}}
  hw.module.extern @handshake_passer_0(in %data : !handshake.channel<i1>, in %ctrl : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out result : !handshake.channel<i1>) attributes {hw.name = "handshake.passer", hw.parameters = {}}
  hw.module.extern @handshake_mux_0(in %index : !handshake.channel<i1>, in %ins_0 : !handshake.channel<i1>, in %ins_1 : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i1>) attributes {hw.name = "handshake.mux", hw.parameters = {DATA_TYPE = !handshake.channel<i1>, SELECT_TYPE = !handshake.channel<i1>, SIZE = 2 : ui32}}
  hw.module @mux_to_and_lhs_wrapper(in %A_in : !handshake.channel<i1>, in %B_in : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out C_out : !handshake.channel<i1>) {
    %mux_to_and_lhs_wrapped.C_out = hw.instance "mux_to_and_lhs_wrapped" @mux_to_and_lhs(A_in: %A_in: !handshake.channel<i1>, B_in: %B_in: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (C_out: !handshake.channel<i1>)
    hw.output %mux_to_and_lhs_wrapped.C_out : !handshake.channel<i1>
  }
}

