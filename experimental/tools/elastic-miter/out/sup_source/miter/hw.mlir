module {
  hw.module @elastic_miter_sup_source_lhs_sup_source_rhs(in %ctrl : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out EQ_result : !handshake.control<>) {
    %in_fork_ctrl.outs_0, %in_fork_ctrl.outs_1 = hw.instance "in_fork_ctrl" @handshake_lazy_fork_0(ins: %ctrl: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs_0: !handshake.channel<i1>, outs_1: !handshake.channel<i1>)
    %rhs_in_buf_ctrl.outs = hw.instance "rhs_in_buf_ctrl" @handshake_buffer_0(ins: %in_fork_ctrl.outs_1: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %out_nds_result.result = hw.instance "out_nds_result" @handshake_ndsource_0(clk: %clk: i1, rst: %rst: i1) -> (result: !handshake.control<>)
    %out_lf_result.outs_0, %out_lf_result.outs_1 = hw.instance "out_lf_result" @handshake_lazy_fork_1(ins: %out_nds_result.result: !handshake.control<>, clk: %clk: i1, rst: %rst: i1) -> (outs_0: !handshake.control<>, outs_1: !handshake.control<>)
    %out_buf_lhs_nds_result.outs = hw.instance "out_buf_lhs_nds_result" @handshake_buffer_1(ins: %out_lf_result.outs_0: !handshake.control<>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.control<>)
    %out_buf_rhs_nds_result.outs = hw.instance "out_buf_rhs_nds_result" @handshake_buffer_1(ins: %out_lf_result.outs_1: !handshake.control<>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.control<>)
    %lhs_out_bl_result.outs = hw.instance "lhs_out_bl_result" @handshake_blocker_0(ins: %lhs_source.outs: !handshake.control<>, ctrl: %out_buf_lhs_nds_result.outs: !handshake.control<>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.control<>)
    %rhs_out_bl_result.outs = hw.instance "rhs_out_bl_result" @handshake_blocker_0(ins: %rhs_passer.result: !handshake.control<>, ctrl: %out_buf_rhs_nds_result.outs: !handshake.control<>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.control<>)
    %out_eq_result.outs = hw.instance "out_eq_result" @handshake_join_0(ins_0: %lhs_out_bl_result.outs: !handshake.control<>, ins_1: %rhs_out_bl_result.outs: !handshake.control<>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.control<>)
    hw.instance "lhs_sink" @handshake_sink_0(ins: %in_fork_ctrl.outs_0: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> ()
    %lhs_source.outs = hw.instance "lhs_source" @handshake_source_0(clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.control<>)
    %rhs_source.outs = hw.instance "rhs_source" @handshake_source_0(clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.control<>)
    %rhs_passer.result = hw.instance "rhs_passer" @handshake_passer_0(data: %rhs_source.outs: !handshake.control<>, ctrl: %rhs_in_buf_ctrl.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (result: !handshake.control<>)
    hw.output %out_eq_result.outs : !handshake.control<>
  }
  hw.module.extern @handshake_lazy_fork_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs_0 : !handshake.channel<i1>, out outs_1 : !handshake.channel<i1>) attributes {hw.name = "handshake.lazy_fork", hw.parameters = {DATA_TYPE = !handshake.channel<i1>, SIZE = 2 : ui32}}
  hw.module.extern @handshake_buffer_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i1>) attributes {hw.name = "handshake.buffer", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_DV", DATA_TYPE = !handshake.channel<i1>, NUM_SLOTS = 1 : ui32}}
  hw.module.extern @handshake_ndsource_0(in %clk : i1, in %rst : i1, out result : !handshake.control<>) attributes {hw.name = "handshake.ndsource", hw.parameters = {}}
  hw.module.extern @handshake_lazy_fork_1(in %ins : !handshake.control<>, in %clk : i1, in %rst : i1, out outs_0 : !handshake.control<>, out outs_1 : !handshake.control<>) attributes {hw.name = "handshake.lazy_fork", hw.parameters = {DATA_TYPE = !handshake.control<>, SIZE = 2 : ui32}}
  hw.module.extern @handshake_buffer_1(in %ins : !handshake.control<>, in %clk : i1, in %rst : i1, out outs : !handshake.control<>) attributes {hw.name = "handshake.buffer", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_DV", DATA_TYPE = !handshake.control<>, NUM_SLOTS = 1 : ui32}}
  hw.module.extern @handshake_blocker_0(in %ins : !handshake.control<>, in %ctrl : !handshake.control<>, in %clk : i1, in %rst : i1, out outs : !handshake.control<>) attributes {hw.name = "handshake.blocker", hw.parameters = {SIZE = 2 : ui32}}
  hw.module.extern @handshake_join_0(in %ins_0 : !handshake.control<>, in %ins_1 : !handshake.control<>, in %clk : i1, in %rst : i1, out outs : !handshake.control<>) attributes {hw.name = "handshake.join", hw.parameters = {SIZE = 2 : ui32}}
  hw.module.extern @handshake_sink_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1) attributes {hw.name = "handshake.sink", hw.parameters = {DATA_TYPE = !handshake.channel<i1>}}
  hw.module.extern @handshake_source_0(in %clk : i1, in %rst : i1, out outs : !handshake.control<>) attributes {hw.name = "handshake.source", hw.parameters = {}}
  hw.module.extern @handshake_passer_0(in %data : !handshake.control<>, in %ctrl : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out result : !handshake.control<>) attributes {hw.name = "handshake.passer", hw.parameters = {}}
  hw.module @elastic_miter_sup_source_lhs_sup_source_rhs_wrapper(in %ctrl : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out EQ_result : !handshake.control<>) {
    %elastic_miter_sup_source_lhs_sup_source_rhs_wrapped.EQ_result = hw.instance "elastic_miter_sup_source_lhs_sup_source_rhs_wrapped" @elastic_miter_sup_source_lhs_sup_source_rhs(ctrl: %ctrl: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (EQ_result: !handshake.control<>)
    hw.output %elastic_miter_sup_source_lhs_sup_source_rhs_wrapped.EQ_result : !handshake.control<>
  }
}

