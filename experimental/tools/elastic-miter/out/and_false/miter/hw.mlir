module {
  hw.module @elastic_miter_and_false_lhs_and_false_rhs(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out EQ_out : !handshake.channel<i1>) {
    %in_fork_ins.outs_0, %in_fork_ins.outs_1 = hw.instance "in_fork_ins" @handshake_lazy_fork_0(ins: %ins: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs_0: !handshake.channel<i1>, outs_1: !handshake.channel<i1>)
    %lhs_in_buf_ins.outs = hw.instance "lhs_in_buf_ins" @handshake_buffer_0(ins: %in_fork_ins.outs_0: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %out_nds_out.result = hw.instance "out_nds_out" @handshake_ndsource_0(clk: %clk: i1, rst: %rst: i1) -> (result: !handshake.control<>)
    %out_lf_out.outs_0, %out_lf_out.outs_1 = hw.instance "out_lf_out" @handshake_lazy_fork_1(ins: %out_nds_out.result: !handshake.control<>, clk: %clk: i1, rst: %rst: i1) -> (outs_0: !handshake.control<>, outs_1: !handshake.control<>)
    %out_buf_lhs_nds_out.outs = hw.instance "out_buf_lhs_nds_out" @handshake_buffer_1(ins: %out_lf_out.outs_0: !handshake.control<>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.control<>)
    %out_buf_rhs_nds_out.outs = hw.instance "out_buf_rhs_nds_out" @handshake_buffer_1(ins: %out_lf_out.outs_1: !handshake.control<>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.control<>)
    %lhs_out_bl_out.outs = hw.instance "lhs_out_bl_out" @handshake_blocker_0(ins: %lhs_and.result: !handshake.channel<i1>, ctrl: %out_buf_lhs_nds_out.outs: !handshake.control<>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %rhs_out_bl_out.outs = hw.instance "rhs_out_bl_out" @handshake_blocker_0(ins: %rhs_constant.outs: !handshake.channel<i1>, ctrl: %out_buf_rhs_nds_out.outs: !handshake.control<>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %out_eq_out.result = hw.instance "out_eq_out" @handshake_cmpi_0(lhs: %lhs_out_bl_out.outs: !handshake.channel<i1>, rhs: %rhs_out_bl_out.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (result: !handshake.channel<i1>)
    %lhs_source.outs = hw.instance "lhs_source" @handshake_source_0(clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.control<>)
    %lhs_constant.outs = hw.instance "lhs_constant" @handshake_constant_0(ctrl: %lhs_source.outs: !handshake.control<>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %lhs_and.result = hw.instance "lhs_and" @handshake_andi_0(lhs: %lhs_in_buf_ins.outs: !handshake.channel<i1>, rhs: %lhs_constant.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (result: !handshake.channel<i1>)
    hw.instance "rhs_sink" @handshake_sink_0(ins: %in_fork_ins.outs_1: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> ()
    %rhs_source.outs = hw.instance "rhs_source" @handshake_source_0(clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.control<>)
    %rhs_constant.outs = hw.instance "rhs_constant" @handshake_constant_0(ctrl: %rhs_source.outs: !handshake.control<>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    hw.output %out_eq_out.result : !handshake.channel<i1>
  }
  hw.module.extern @handshake_lazy_fork_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs_0 : !handshake.channel<i1>, out outs_1 : !handshake.channel<i1>) attributes {hw.name = "handshake.lazy_fork", hw.parameters = {DATA_TYPE = !handshake.channel<i1>, SIZE = 2 : ui32}}
  hw.module.extern @handshake_buffer_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i1>) attributes {hw.name = "handshake.buffer", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_DV", DATA_TYPE = !handshake.channel<i1>, DEBUG_COUNTER = true, NUM_SLOTS = 1 : ui32}}
  hw.module.extern @handshake_ndsource_0(in %clk : i1, in %rst : i1, out result : !handshake.control<>) attributes {hw.name = "handshake.ndsource", hw.parameters = {}}
  hw.module.extern @handshake_lazy_fork_1(in %ins : !handshake.control<>, in %clk : i1, in %rst : i1, out outs_0 : !handshake.control<>, out outs_1 : !handshake.control<>) attributes {hw.name = "handshake.lazy_fork", hw.parameters = {DATA_TYPE = !handshake.control<>, SIZE = 2 : ui32}}
  hw.module.extern @handshake_buffer_1(in %ins : !handshake.control<>, in %clk : i1, in %rst : i1, out outs : !handshake.control<>) attributes {hw.name = "handshake.buffer", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_DV", DATA_TYPE = !handshake.control<>, DEBUG_COUNTER = false, NUM_SLOTS = 1 : ui32}}
  hw.module.extern @handshake_blocker_0(in %ins : !handshake.channel<i1>, in %ctrl : !handshake.control<>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i1>) attributes {hw.name = "handshake.blocker", hw.parameters = {SIZE = 2 : ui32}}
  hw.module.extern @handshake_cmpi_0(in %lhs : !handshake.channel<i1>, in %rhs : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out result : !handshake.channel<i1>) attributes {hw.name = "handshake.cmpi", hw.parameters = {DATA_TYPE = !handshake.channel<i1>, PREDICATE = "eq"}}
  hw.module.extern @handshake_source_0(in %clk : i1, in %rst : i1, out outs : !handshake.control<>) attributes {hw.name = "handshake.source", hw.parameters = {}}
  hw.module.extern @handshake_constant_0(in %ctrl : !handshake.control<>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i1>) attributes {hw.name = "handshake.constant", hw.parameters = {DATA_WIDTH = 1 : ui32, VALUE = "0"}}
  hw.module.extern @handshake_andi_0(in %lhs : !handshake.channel<i1>, in %rhs : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out result : !handshake.channel<i1>) attributes {hw.name = "handshake.andi", hw.parameters = {DATA_TYPE = !handshake.channel<i1>}}
  hw.module.extern @handshake_sink_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1) attributes {hw.name = "handshake.sink", hw.parameters = {DATA_TYPE = !handshake.channel<i1>}}
  hw.module @elastic_miter_and_false_lhs_and_false_rhs_wrapper(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out EQ_out : !handshake.channel<i1>) {
    %elastic_miter_and_false_lhs_and_false_rhs_wrapped.EQ_out = hw.instance "elastic_miter_and_false_lhs_and_false_rhs_wrapped" @elastic_miter_and_false_lhs_and_false_rhs(ins: %ins: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (EQ_out: !handshake.channel<i1>)
    hw.output %elastic_miter_and_false_lhs_and_false_rhs_wrapped.EQ_out : !handshake.channel<i1>
  }
}

