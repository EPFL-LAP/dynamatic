module {
  hw.module @interpolator_ind_lhs(in %A_in : !handshake.channel<i1>, in %B_in : !handshake.channel<i1>, in %Sup_in : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out Sup_out : !handshake.channel<i1>) {
    %ndw_in_A_in.outs = hw.instance "ndw_in_A_in" @handshake_ndwire_0(ins: %A_in: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %ndw_in_B_in.outs = hw.instance "ndw_in_B_in" @handshake_ndwire_0(ins: %B_in: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %ndw_in_Sup_in.outs = hw.instance "ndw_in_Sup_in" @handshake_ndwire_0(ins: %Sup_in: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %ndw_out_Sup_out.outs = hw.instance "ndw_out_Sup_out" @handshake_ndwire_0(ins: %passer2.result: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %fork.outs_0, %fork.outs_1 = hw.instance "fork" @handshake_fork_0(ins: %ndw_in_B_in.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs_0: !handshake.channel<i1>, outs_1: !handshake.channel<i1>)
    %interpolate.result = hw.instance "interpolate" @handshake_spec_v2_interpolator_0(short: %ndw_in_A_in.outs: !handshake.channel<i1>, long: %fork.outs_0: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (result: !handshake.channel<i1>)
    %ri.outs = hw.instance "ri" @handshake_spec_v2_repeating_init_0(ins: %fork.outs_1: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %passer1.result = hw.instance "passer1" @handshake_passer_0(data: %ndw_in_Sup_in.outs: !handshake.channel<i1>, ctrl: %ri.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (result: !handshake.channel<i1>)
    %passer2.result = hw.instance "passer2" @handshake_passer_0(data: %passer1.result: !handshake.channel<i1>, ctrl: %interpolate.result: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (result: !handshake.channel<i1>)
    hw.output %ndw_out_Sup_out.outs : !handshake.channel<i1>
  }
  hw.module.extern @handshake_ndwire_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i1>) attributes {hw.name = "handshake.ndwire", hw.parameters = {DATA_TYPE = !handshake.channel<i1>}}
  hw.module.extern @handshake_fork_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs_0 : !handshake.channel<i1>, out outs_1 : !handshake.channel<i1>) attributes {hw.name = "handshake.fork", hw.parameters = {DATA_TYPE = !handshake.channel<i1>, SIZE = 2 : ui32}}
  hw.module.extern @handshake_spec_v2_interpolator_0(in %short : !handshake.channel<i1>, in %long : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out result : !handshake.channel<i1>) attributes {hw.name = "handshake.spec_v2_interpolator", hw.parameters = {}}
  hw.module.extern @handshake_spec_v2_repeating_init_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i1>) attributes {hw.name = "handshake.spec_v2_repeating_init", hw.parameters = {INIT_TOKEN = 1 : ui32}}
  hw.module.extern @handshake_passer_0(in %data : !handshake.channel<i1>, in %ctrl : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out result : !handshake.channel<i1>) attributes {hw.name = "handshake.passer", hw.parameters = {}}
  hw.module @interpolator_ind_lhs_wrapper(in %A_in : !handshake.channel<i1>, in %B_in : !handshake.channel<i1>, in %Sup_in : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out Sup_out : !handshake.channel<i1>) {
    %interpolator_ind_lhs_wrapped.Sup_out = hw.instance "interpolator_ind_lhs_wrapped" @interpolator_ind_lhs(A_in: %A_in: !handshake.channel<i1>, B_in: %B_in: !handshake.channel<i1>, Sup_in: %Sup_in: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (Sup_out: !handshake.channel<i1>)
    hw.output %interpolator_ind_lhs_wrapped.Sup_out : !handshake.channel<i1>
  }
}

