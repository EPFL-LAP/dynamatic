module {
  hw.module @elastic_miter_switchSuppressCtrl_lhs_switchSuppressCtrl_rhs(in %val : !handshake.channel<i1>, in %A_in : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out EQ_B_out : !handshake.channel<i1>) {
    %fork.outs_0, %fork.outs_1, %fork.outs_2 = hw.instance "fork" @handshake_fork_0(ins: %val: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs_0: !handshake.channel<i1>, outs_1: !handshake.channel<i1>, outs_2: !handshake.channel<i1>)
    %interpolator.result = hw.instance "interpolator" @handshake_spec_v2_interpolator_0(short: %fork.outs_0: !handshake.channel<i1>, long: %fork.outs_1: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (result: !handshake.channel<i1>)
    %in_fork_val.outs_0, %in_fork_val.outs_1 = hw.instance "in_fork_val" @handshake_lazy_fork_0(ins: %fork.outs_2: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs_0: !handshake.channel<i1>, outs_1: !handshake.channel<i1>)
    %lhs_in_buf_val.outs = hw.instance "lhs_in_buf_val" @handshake_buffer_0(ins: %in_fork_val.outs_0: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %lhs_in_ndw_val.outs = hw.instance "lhs_in_ndw_val" @handshake_ndwire_0(ins: %lhs_in_buf_val.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %rhs_in_ndw_val.outs = hw.instance "rhs_in_ndw_val" @handshake_ndwire_0(ins: %in_fork_val.outs_1: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %in_fork_A_in.outs_0, %in_fork_A_in.outs_1 = hw.instance "in_fork_A_in" @handshake_lazy_fork_0(ins: %A_in: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs_0: !handshake.channel<i1>, outs_1: !handshake.channel<i1>)
    %lhs_in_buf_A_in.outs = hw.instance "lhs_in_buf_A_in" @handshake_buffer_0(ins: %in_fork_A_in.outs_0: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %rhs_in_buf_A_in.outs = hw.instance "rhs_in_buf_A_in" @handshake_buffer_0(ins: %in_fork_A_in.outs_1: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %lhs_in_ndw_A_in.outs = hw.instance "lhs_in_ndw_A_in" @handshake_ndwire_0(ins: %lhs_in_buf_A_in.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %rhs_in_ndw_A_in.outs = hw.instance "rhs_in_ndw_A_in" @handshake_ndwire_0(ins: %rhs_in_buf_A_in.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %in_fork_val2.outs_0, %in_fork_val2.outs_1 = hw.instance "in_fork_val2" @handshake_lazy_fork_0(ins: %interpolator.result: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs_0: !handshake.channel<i1>, outs_1: !handshake.channel<i1>)
    %rhs_in_buf_val2.outs = hw.instance "rhs_in_buf_val2" @handshake_buffer_0(ins: %in_fork_val2.outs_1: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %lhs_in_ndw_val2.outs = hw.instance "lhs_in_ndw_val2" @handshake_ndwire_0(ins: %in_fork_val2.outs_0: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %rhs_in_ndw_val2.outs = hw.instance "rhs_in_ndw_val2" @handshake_ndwire_0(ins: %rhs_in_buf_val2.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    hw.instance "lhs_vm_sink_2" @handshake_sink_0(ins: %lhs_in_ndw_val2.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> ()
    %lhs_passer.result = hw.instance "lhs_passer" @handshake_passer_0(data: %lhs_in_ndw_A_in.outs: !handshake.channel<i1>, ctrl: %lhs_in_ndw_val.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (result: !handshake.channel<i1>)
    hw.instance "rhs_vm_sink_0" @handshake_sink_0(ins: %rhs_in_ndw_val.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> ()
    %rhs_passer.result = hw.instance "rhs_passer" @handshake_passer_0(data: %rhs_in_ndw_A_in.outs: !handshake.channel<i1>, ctrl: %rhs_in_ndw_val2.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (result: !handshake.channel<i1>)
    %out_nds_B_out.result = hw.instance "out_nds_B_out" @handshake_ndsource_0(clk: %clk: i1, rst: %rst: i1) -> (result: !handshake.control<>)
    %out_lf_B_out.outs_0, %out_lf_B_out.outs_1 = hw.instance "out_lf_B_out" @handshake_lazy_fork_1(ins: %out_nds_B_out.result: !handshake.control<>, clk: %clk: i1, rst: %rst: i1) -> (outs_0: !handshake.control<>, outs_1: !handshake.control<>)
    %out_buf_lhs_nds_B_out.outs = hw.instance "out_buf_lhs_nds_B_out" @handshake_buffer_1(ins: %out_lf_B_out.outs_0: !handshake.control<>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.control<>)
    %out_buf_rhs_nds_B_out.outs = hw.instance "out_buf_rhs_nds_B_out" @handshake_buffer_1(ins: %out_lf_B_out.outs_1: !handshake.control<>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.control<>)
    %out_lhs_bl_B_out.outs = hw.instance "out_lhs_bl_B_out" @handshake_blocker_0(ins: %lhs_passer.result: !handshake.channel<i1>, ctrl: %out_buf_lhs_nds_B_out.outs: !handshake.control<>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %out_rhs_bl_B_out.outs = hw.instance "out_rhs_bl_B_out" @handshake_blocker_0(ins: %rhs_passer.result: !handshake.channel<i1>, ctrl: %out_buf_rhs_nds_B_out.outs: !handshake.control<>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %lhs_out_ndw_B_out.outs = hw.instance "lhs_out_ndw_B_out" @handshake_ndwire_0(ins: %out_lhs_bl_B_out.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %rhs_out_ndw_B_out.outs = hw.instance "rhs_out_ndw_B_out" @handshake_ndwire_0(ins: %out_rhs_bl_B_out.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %lhs_out_buf_B_out.outs = hw.instance "lhs_out_buf_B_out" @handshake_buffer_0(ins: %lhs_out_ndw_B_out.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %rhs_out_buf_B_out.outs = hw.instance "rhs_out_buf_B_out" @handshake_buffer_0(ins: %rhs_out_ndw_B_out.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %out_eq_B_out.result = hw.instance "out_eq_B_out" @handshake_cmpi_0(lhs: %lhs_out_buf_B_out.outs: !handshake.channel<i1>, rhs: %rhs_out_buf_B_out.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (result: !handshake.channel<i1>)
    hw.output %out_eq_B_out.result : !handshake.channel<i1>
  }
  hw.module.extern @handshake_fork_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs_0 : !handshake.channel<i1>, out outs_1 : !handshake.channel<i1>, out outs_2 : !handshake.channel<i1>) attributes {hw.name = "handshake.fork", hw.parameters = {DATA_TYPE = !handshake.channel<i1>, SIZE = 3 : ui32}}
  hw.module.extern @handshake_spec_v2_interpolator_0(in %short : !handshake.channel<i1>, in %long : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out result : !handshake.channel<i1>) attributes {hw.name = "handshake.spec_v2_interpolator", hw.parameters = {}}
  hw.module.extern @handshake_lazy_fork_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs_0 : !handshake.channel<i1>, out outs_1 : !handshake.channel<i1>) attributes {hw.name = "handshake.lazy_fork", hw.parameters = {DATA_TYPE = !handshake.channel<i1>, SIZE = 2 : ui32}}
  hw.module.extern @handshake_buffer_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i1>) attributes {hw.name = "handshake.buffer", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_DV", DATA_TYPE = !handshake.channel<i1>, NUM_SLOTS = 1 : ui32}}
  hw.module.extern @handshake_ndwire_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i1>) attributes {hw.name = "handshake.ndwire", hw.parameters = {DATA_TYPE = !handshake.channel<i1>}}
  hw.module.extern @handshake_sink_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1) attributes {hw.name = "handshake.sink", hw.parameters = {DATA_TYPE = !handshake.channel<i1>}}
  hw.module.extern @handshake_passer_0(in %data : !handshake.channel<i1>, in %ctrl : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out result : !handshake.channel<i1>) attributes {hw.name = "handshake.passer", hw.parameters = {}}
  hw.module.extern @handshake_ndsource_0(in %clk : i1, in %rst : i1, out result : !handshake.control<>) attributes {hw.name = "handshake.ndsource", hw.parameters = {}}
  hw.module.extern @handshake_lazy_fork_1(in %ins : !handshake.control<>, in %clk : i1, in %rst : i1, out outs_0 : !handshake.control<>, out outs_1 : !handshake.control<>) attributes {hw.name = "handshake.lazy_fork", hw.parameters = {DATA_TYPE = !handshake.control<>, SIZE = 2 : ui32}}
  hw.module.extern @handshake_buffer_1(in %ins : !handshake.control<>, in %clk : i1, in %rst : i1, out outs : !handshake.control<>) attributes {hw.name = "handshake.buffer", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_DV", DATA_TYPE = !handshake.control<>, NUM_SLOTS = 1 : ui32}}
  hw.module.extern @handshake_blocker_0(in %ins : !handshake.channel<i1>, in %ctrl : !handshake.control<>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i1>) attributes {hw.name = "handshake.blocker", hw.parameters = {SIZE = 2 : ui32}}
  hw.module.extern @handshake_cmpi_0(in %lhs : !handshake.channel<i1>, in %rhs : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out result : !handshake.channel<i1>) attributes {hw.name = "handshake.cmpi", hw.parameters = {DATA_TYPE = !handshake.channel<i1>, INTERNAL_DELAY = "0.0", PREDICATE = "eq"}}
  hw.module @elastic_miter_switchSuppressCtrl_lhs_switchSuppressCtrl_rhs_wrapper(in %val : !handshake.channel<i1>, in %A_in : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out EQ_B_out : !handshake.channel<i1>) {
    %elastic_miter_switchSuppressCtrl_lhs_switchSuppressCtrl_rhs_wrapped.EQ_B_out = hw.instance "elastic_miter_switchSuppressCtrl_lhs_switchSuppressCtrl_rhs_wrapped" @elastic_miter_switchSuppressCtrl_lhs_switchSuppressCtrl_rhs(val: %val: !handshake.channel<i1>, A_in: %A_in: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (EQ_B_out: !handshake.channel<i1>)
    hw.output %elastic_miter_switchSuppressCtrl_lhs_switchSuppressCtrl_rhs_wrapped.EQ_B_out : !handshake.channel<i1>
  }
}

