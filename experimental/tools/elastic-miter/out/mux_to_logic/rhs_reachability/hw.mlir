module {
  hw.module @mux_to_logic_rhs(in %in1 : !handshake.channel<i1>, in %in2 : !handshake.channel<i1>, in %sel : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out out : !handshake.channel<i1>) {
    %ndw_in_in1.outs = hw.instance "ndw_in_in1" @handshake_ndwire_0(ins: %in1: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %ndw_in_in2.outs = hw.instance "ndw_in_in2" @handshake_ndwire_0(ins: %in2: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %ndw_in_sel.outs = hw.instance "ndw_in_sel" @handshake_ndwire_0(ins: %sel: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %ndw_out_out.outs = hw.instance "ndw_out_out" @handshake_ndwire_0(ins: %or.result: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %vm_fork_2.outs_0, %vm_fork_2.outs_1 = hw.instance "vm_fork_2" @handshake_fork_0(ins: %ndw_in_sel.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs_0: !handshake.channel<i1>, outs_1: !handshake.channel<i1>)
    %not.outs = hw.instance "not" @handshake_not_0(ins: %vm_fork_2.outs_0: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %and1.result = hw.instance "and1" @handshake_andi_0(lhs: %ndw_in_in1.outs: !handshake.channel<i1>, rhs: %vm_fork_2.outs_1: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (result: !handshake.channel<i1>)
    %and2.result = hw.instance "and2" @handshake_andi_0(lhs: %ndw_in_in2.outs: !handshake.channel<i1>, rhs: %not.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (result: !handshake.channel<i1>)
    %or.result = hw.instance "or" @handshake_ori_0(lhs: %and1.result: !handshake.channel<i1>, rhs: %and2.result: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (result: !handshake.channel<i1>)
    hw.output %ndw_out_out.outs : !handshake.channel<i1>
  }
  hw.module.extern @handshake_ndwire_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i1>) attributes {hw.name = "handshake.ndwire", hw.parameters = {DATA_TYPE = !handshake.channel<i1>}}
  hw.module.extern @handshake_fork_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs_0 : !handshake.channel<i1>, out outs_1 : !handshake.channel<i1>) attributes {hw.name = "handshake.fork", hw.parameters = {DATA_TYPE = !handshake.channel<i1>, SIZE = 2 : ui32}}
  hw.module.extern @handshake_not_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i1>) attributes {hw.name = "handshake.not", hw.parameters = {DATA_TYPE = !handshake.channel<i1>}}
  hw.module.extern @handshake_andi_0(in %lhs : !handshake.channel<i1>, in %rhs : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out result : !handshake.channel<i1>) attributes {hw.name = "handshake.andi", hw.parameters = {DATA_TYPE = !handshake.channel<i1>}}
  hw.module.extern @handshake_ori_0(in %lhs : !handshake.channel<i1>, in %rhs : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out result : !handshake.channel<i1>) attributes {hw.name = "handshake.ori", hw.parameters = {DATA_TYPE = !handshake.channel<i1>}}
  hw.module @mux_to_logic_rhs_wrapper(in %in1 : !handshake.channel<i1>, in %in2 : !handshake.channel<i1>, in %sel : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out out : !handshake.channel<i1>) {
    %mux_to_logic_rhs_wrapped.out = hw.instance "mux_to_logic_rhs_wrapped" @mux_to_logic_rhs(in1: %in1: !handshake.channel<i1>, in2: %in2: !handshake.channel<i1>, sel: %sel: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (out: !handshake.channel<i1>)
    hw.output %mux_to_logic_rhs_wrapped.out : !handshake.channel<i1>
  }
}

