module {
  hw.module @mux_add_suppress_rhs(in %ins : !handshake.channel<i1>, in %sel : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out out : !handshake.channel<i1>) {
    %ndw_in_ins.outs = hw.instance "ndw_in_ins" @handshake_ndwire_0(ins: %ins: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %ndw_in_sel.outs = hw.instance "ndw_in_sel" @handshake_ndwire_0(ins: %sel: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %ndw_out_out.outs = hw.instance "ndw_out_out" @handshake_ndwire_0(ins: %mux.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %vm_fork_1.outs_0, %vm_fork_1.outs_1 = hw.instance "vm_fork_1" @handshake_fork_0(ins: %ndw_in_sel.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs_0: !handshake.channel<i1>, outs_1: !handshake.channel<i1>)
    %source.outs = hw.instance "source" @handshake_source_0(clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.control<>)
    %constant.outs = hw.instance "constant" @handshake_constant_0(ctrl: %source.outs: !handshake.control<>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %passer.result = hw.instance "passer" @handshake_passer_0(data: %constant.outs: !handshake.channel<i1>, ctrl: %vm_fork_1.outs_0: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (result: !handshake.channel<i1>)
    %mux.outs = hw.instance "mux" @handshake_mux_0(index: %vm_fork_1.outs_1: !handshake.channel<i1>, ins_0: %ndw_in_ins.outs: !handshake.channel<i1>, ins_1: %passer.result: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    hw.output %ndw_out_out.outs : !handshake.channel<i1>
  }
  hw.module.extern @handshake_ndwire_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i1>) attributes {hw.name = "handshake.ndwire", hw.parameters = {DATA_TYPE = !handshake.channel<i1>}}
  hw.module.extern @handshake_fork_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs_0 : !handshake.channel<i1>, out outs_1 : !handshake.channel<i1>) attributes {hw.name = "handshake.fork", hw.parameters = {DATA_TYPE = !handshake.channel<i1>, SIZE = 2 : ui32}}
  hw.module.extern @handshake_source_0(in %clk : i1, in %rst : i1, out outs : !handshake.control<>) attributes {hw.name = "handshake.source", hw.parameters = {}}
  hw.module.extern @handshake_constant_0(in %ctrl : !handshake.control<>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i1>) attributes {hw.name = "handshake.constant", hw.parameters = {DATA_WIDTH = 1 : ui32, VALUE = "1"}}
  hw.module.extern @handshake_passer_0(in %data : !handshake.channel<i1>, in %ctrl : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out result : !handshake.channel<i1>) attributes {hw.name = "handshake.passer", hw.parameters = {}}
  hw.module.extern @handshake_mux_0(in %index : !handshake.channel<i1>, in %ins_0 : !handshake.channel<i1>, in %ins_1 : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i1>) attributes {hw.name = "handshake.mux", hw.parameters = {DATA_TYPE = !handshake.channel<i1>, SELECT_TYPE = !handshake.channel<i1>, SIZE = 2 : ui32}}
  hw.module @mux_add_suppress_rhs_wrapper(in %ins : !handshake.channel<i1>, in %sel : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out out : !handshake.channel<i1>) {
    %mux_add_suppress_rhs_wrapped.out = hw.instance "mux_add_suppress_rhs_wrapped" @mux_add_suppress_rhs(ins: %ins: !handshake.channel<i1>, sel: %sel: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (out: !handshake.channel<i1>)
    hw.output %mux_add_suppress_rhs_wrapped.out : !handshake.channel<i1>
  }
}

