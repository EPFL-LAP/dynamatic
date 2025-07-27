module {
  hw.module @sup_gamma_lhs(in %a1 : !handshake.channel<i1>, in %a2 : !handshake.channel<i1>, in %c1 : !handshake.channel<i1>, in %c2 : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out b : !handshake.channel<i1>) {
    %ndw_in_a1.outs = hw.instance "ndw_in_a1" @handshake_ndwire_0(ins: %a1: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %ndw_in_a2.outs = hw.instance "ndw_in_a2" @handshake_ndwire_0(ins: %a2: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %ndw_in_c1.outs = hw.instance "ndw_in_c1" @handshake_ndwire_0(ins: %c1: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %ndw_in_c2.outs = hw.instance "ndw_in_c2" @handshake_ndwire_0(ins: %c2: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %ndw_out_b.outs = hw.instance "ndw_out_b" @handshake_ndwire_0(ins: %mux.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %vm_fork_2.outs_0, %vm_fork_2.outs_1, %vm_fork_2.outs_2 = hw.instance "vm_fork_2" @handshake_fork_0(ins: %ndw_in_c1.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs_0: !handshake.channel<i1>, outs_1: !handshake.channel<i1>, outs_2: !handshake.channel<i1>)
    %vm_fork_3.outs_0, %vm_fork_3.outs_1, %vm_fork_3.outs_2 = hw.instance "vm_fork_3" @handshake_fork_0(ins: %ndw_in_c2.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs_0: !handshake.channel<i1>, outs_1: !handshake.channel<i1>, outs_2: !handshake.channel<i1>)
    %passer1.result = hw.instance "passer1" @handshake_passer_0(data: %vm_fork_2.outs_0: !handshake.channel<i1>, ctrl: %vm_fork_3.outs_0: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (result: !handshake.channel<i1>)
    %not.outs = hw.instance "not" @handshake_not_0(ins: %vm_fork_3.outs_1: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %passer2.result = hw.instance "passer2" @handshake_passer_0(data: %vm_fork_2.outs_1: !handshake.channel<i1>, ctrl: %not.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (result: !handshake.channel<i1>)
    %passer3.result = hw.instance "passer3" @handshake_passer_0(data: %vm_fork_3.outs_2: !handshake.channel<i1>, ctrl: %vm_fork_2.outs_2: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (result: !handshake.channel<i1>)
    %passer_a1.result = hw.instance "passer_a1" @handshake_passer_0(data: %ndw_in_a1.outs: !handshake.channel<i1>, ctrl: %passer1.result: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (result: !handshake.channel<i1>)
    %passer_a2.result = hw.instance "passer_a2" @handshake_passer_0(data: %ndw_in_a2.outs: !handshake.channel<i1>, ctrl: %passer2.result: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (result: !handshake.channel<i1>)
    %mux.outs = hw.instance "mux" @handshake_mux_0(index: %passer3.result: !handshake.channel<i1>, ins_0: %passer_a2.result: !handshake.channel<i1>, ins_1: %passer_a1.result: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    hw.output %ndw_out_b.outs : !handshake.channel<i1>
  }
  hw.module.extern @handshake_ndwire_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i1>) attributes {hw.name = "handshake.ndwire", hw.parameters = {DATA_TYPE = !handshake.channel<i1>}}
  hw.module.extern @handshake_fork_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs_0 : !handshake.channel<i1>, out outs_1 : !handshake.channel<i1>, out outs_2 : !handshake.channel<i1>) attributes {hw.name = "handshake.fork", hw.parameters = {DATA_TYPE = !handshake.channel<i1>, SIZE = 3 : ui32}}
  hw.module.extern @handshake_passer_0(in %data : !handshake.channel<i1>, in %ctrl : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out result : !handshake.channel<i1>) attributes {hw.name = "handshake.passer", hw.parameters = {}}
  hw.module.extern @handshake_not_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i1>) attributes {hw.name = "handshake.not", hw.parameters = {DATA_TYPE = !handshake.channel<i1>}}
  hw.module.extern @handshake_mux_0(in %index : !handshake.channel<i1>, in %ins_0 : !handshake.channel<i1>, in %ins_1 : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i1>) attributes {hw.name = "handshake.mux", hw.parameters = {DATA_TYPE = !handshake.channel<i1>, SELECT_TYPE = !handshake.channel<i1>, SIZE = 2 : ui32}}
  hw.module @sup_gamma_lhs_wrapper(in %a1 : !handshake.channel<i1>, in %a2 : !handshake.channel<i1>, in %c1 : !handshake.channel<i1>, in %c2 : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out b : !handshake.channel<i1>) {
    %sup_gamma_lhs_wrapped.b = hw.instance "sup_gamma_lhs_wrapped" @sup_gamma_lhs(a1: %a1: !handshake.channel<i1>, a2: %a2: !handshake.channel<i1>, c1: %c1: !handshake.channel<i1>, c2: %c2: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (b: !handshake.channel<i1>)
    hw.output %sup_gamma_lhs_wrapped.b : !handshake.channel<i1>
  }
}

