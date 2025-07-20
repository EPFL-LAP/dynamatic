module {
  hw.module @elastic_miter_test_lhs_test_rhs(in %A_in : !handshake.control<>, in %clk : i1, in %rst : i1) {
    %in_fork_A_in.outs_0, %in_fork_A_in.outs_1 = hw.instance "in_fork_A_in" @handshake_lazy_fork_0(ins: %A_in: !handshake.control<>, clk: %clk: i1, rst: %rst: i1) -> (outs_0: !handshake.control<>, outs_1: !handshake.control<>)
    %lhs_in_buf_A_in.outs = hw.instance "lhs_in_buf_A_in" @handshake_buffer_0(ins: %in_fork_A_in.outs_0: !handshake.control<>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.control<>)
    %rhs_in_buf_A_in.outs = hw.instance "rhs_in_buf_A_in" @handshake_buffer_0(ins: %in_fork_A_in.outs_1: !handshake.control<>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.control<>)
    %lhs_nds.result = hw.instance "lhs_nds" @handshake_ndsource_0(clk: %clk: i1, rst: %rst: i1) -> (result: !handshake.control<>)
    %lhs_join.outs = hw.instance "lhs_join" @handshake_join_0(ins_0: %lhs_in_buf_A_in.outs: !handshake.control<>, ins_1: %lhs_nds.result: !handshake.control<>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.control<>)
    hw.instance "lhs_sink" @handshake_sink_0(ins: %lhs_join.outs: !handshake.control<>, clk: %clk: i1, rst: %rst: i1) -> ()
    %rhs_nds.result = hw.instance "rhs_nds" @handshake_ndsource_0(clk: %clk: i1, rst: %rst: i1) -> (result: !handshake.control<>)
    %rhs_join.outs = hw.instance "rhs_join" @handshake_join_0(ins_0: %rhs_in_buf_A_in.outs: !handshake.control<>, ins_1: %rhs_nds.result: !handshake.control<>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.control<>)
    hw.instance "rhs_sink" @handshake_sink_0(ins: %rhs_join.outs: !handshake.control<>, clk: %clk: i1, rst: %rst: i1) -> ()
    hw.output
  }
  hw.module.extern @handshake_lazy_fork_0(in %ins : !handshake.control<>, in %clk : i1, in %rst : i1, out outs_0 : !handshake.control<>, out outs_1 : !handshake.control<>) attributes {hw.name = "handshake.lazy_fork", hw.parameters = {DATA_TYPE = !handshake.control<>, SIZE = 2 : ui32}}
  hw.module.extern @handshake_buffer_0(in %ins : !handshake.control<>, in %clk : i1, in %rst : i1, out outs : !handshake.control<>) attributes {hw.name = "handshake.buffer", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_DV", DATA_TYPE = !handshake.control<>, NUM_SLOTS = 1 : ui32}}
  hw.module.extern @handshake_ndsource_0(in %clk : i1, in %rst : i1, out result : !handshake.control<>) attributes {hw.name = "handshake.ndsource", hw.parameters = {}}
  hw.module.extern @handshake_join_0(in %ins_0 : !handshake.control<>, in %ins_1 : !handshake.control<>, in %clk : i1, in %rst : i1, out outs : !handshake.control<>) attributes {hw.name = "handshake.join", hw.parameters = {SIZE = 2 : ui32}}
  hw.module.extern @handshake_sink_0(in %ins : !handshake.control<>, in %clk : i1, in %rst : i1) attributes {hw.name = "handshake.sink", hw.parameters = {DATA_TYPE = !handshake.control<>}}
  hw.module @elastic_miter_test_lhs_test_rhs_wrapper(in %A_in : !handshake.control<>, in %clk : i1, in %rst : i1) {
    hw.instance "elastic_miter_test_lhs_test_rhs_wrapped" @elastic_miter_test_lhs_test_rhs(A_in: %A_in: !handshake.control<>, clk: %clk: i1, rst: %rst: i1) -> ()
    hw.output
  }
}

