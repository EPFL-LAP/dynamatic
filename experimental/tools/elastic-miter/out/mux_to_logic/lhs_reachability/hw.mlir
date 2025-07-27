module {
  hw.module @mux_to_logic_lhs(in %in1 : !handshake.channel<i1>, in %in2 : !handshake.channel<i1>, in %sel : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out out : !handshake.channel<i1>) {
    %ndw_in_in1.outs = hw.instance "ndw_in_in1" @handshake_ndwire_0(ins: %in1: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %ndw_in_in2.outs = hw.instance "ndw_in_in2" @handshake_ndwire_0(ins: %in2: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %ndw_in_sel.outs = hw.instance "ndw_in_sel" @handshake_ndwire_0(ins: %sel: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %ndw_out_out.outs = hw.instance "ndw_out_out" @handshake_ndwire_0(ins: %mux.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %vm_fork_2.outs_0, %vm_fork_2.outs_1, %vm_fork_2.outs_2 = hw.instance "vm_fork_2" @handshake_fork_0(ins: %ndw_in_sel.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs_0: !handshake.channel<i1>, outs_1: !handshake.channel<i1>, outs_2: !handshake.channel<i1>)
    %not.outs = hw.instance "not" @handshake_not_0(ins: %vm_fork_2.outs_0: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %passer1.result = hw.instance "passer1" @handshake_passer_0(data: %ndw_in_in1.outs: !handshake.channel<i1>, ctrl: %vm_fork_2.outs_1: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (result: !handshake.channel<i1>)
    %passer2.result = hw.instance "passer2" @handshake_passer_0(data: %ndw_in_in2.outs: !handshake.channel<i1>, ctrl: %not.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (result: !handshake.channel<i1>)
    %mux.outs = hw.instance "mux" @handshake_mux_0(index: %vm_fork_2.outs_2: !handshake.channel<i1>, ins_0: %passer2.result: !handshake.channel<i1>, ins_1: %passer1.result: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    hw.output %ndw_out_out.outs : !handshake.channel<i1>
  }
  hw.module.extern @handshake_ndwire_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i1>) attributes {hw.name = "handshake.ndwire", hw.parameters = {DATA_TYPE = !handshake.channel<i1>}}
  hw.module.extern @handshake_fork_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs_0 : !handshake.channel<i1>, out outs_1 : !handshake.channel<i1>, out outs_2 : !handshake.channel<i1>) attributes {hw.name = "handshake.fork", hw.parameters = {DATA_TYPE = !handshake.channel<i1>, SIZE = 3 : ui32}}
  hw.module.extern @handshake_not_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i1>) attributes {hw.name = "handshake.not", hw.parameters = {DATA_TYPE = !handshake.channel<i1>}}
  hw.module.extern @handshake_passer_0(in %data : !handshake.channel<i1>, in %ctrl : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out result : !handshake.channel<i1>) attributes {hw.name = "handshake.passer", hw.parameters = {}}
  hw.module.extern @handshake_mux_0(in %index : !handshake.channel<i1>, in %ins_0 : !handshake.channel<i1>, in %ins_1 : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i1>) attributes {hw.name = "handshake.mux", hw.parameters = {DATA_TYPE = !handshake.channel<i1>, SELECT_TYPE = !handshake.channel<i1>, SIZE = 2 : ui32}}
  hw.module @mux_to_logic_lhs_wrapper(in %in1 : !handshake.channel<i1>, in %in2 : !handshake.channel<i1>, in %sel : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out out : !handshake.channel<i1>) {
    %mux_to_logic_lhs_wrapped.out = hw.instance "mux_to_logic_lhs_wrapped" @mux_to_logic_lhs(in1: %in1: !handshake.channel<i1>, in2: %in2: !handshake.channel<i1>, sel: %sel: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (out: !handshake.channel<i1>)
    hw.output %mux_to_logic_lhs_wrapped.out : !handshake.channel<i1>
  }
}

