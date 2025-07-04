module {
  hw.module @elastic_miter_sup_fork_lhs_sup_fork_rhs(in %A_in : !handshake.channel<i1>, in %Cond : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out EQ_B_out : !handshake.channel<i1>, out EQ_C_out : !handshake.channel<i1>) {
    %in_fork_A_in.outs_0, %in_fork_A_in.outs_1 = hw.instance "in_fork_A_in" @handshake_lazy_fork_0(ins: %A_in: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs_0: !handshake.channel<i1>, outs_1: !handshake.channel<i1>)
    %lhs_in_buf_A_in.outs = hw.instance "lhs_in_buf_A_in" @handshake_buffer_0(ins: %in_fork_A_in.outs_0: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %rhs_in_buf_A_in.outs = hw.instance "rhs_in_buf_A_in" @handshake_buffer_0(ins: %in_fork_A_in.outs_1: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %lhs_in_ndw_A_in.outs = hw.instance "lhs_in_ndw_A_in" @handshake_ndwire_0(ins: %lhs_in_buf_A_in.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %rhs_in_ndw_A_in.outs = hw.instance "rhs_in_ndw_A_in" @handshake_ndwire_0(ins: %rhs_in_buf_A_in.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %in_fork_Cond.outs_0, %in_fork_Cond.outs_1 = hw.instance "in_fork_Cond" @handshake_lazy_fork_0(ins: %Cond: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs_0: !handshake.channel<i1>, outs_1: !handshake.channel<i1>)
    %lhs_in_buf_Cond.outs = hw.instance "lhs_in_buf_Cond" @handshake_buffer_0(ins: %in_fork_Cond.outs_0: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %rhs_in_buf_Cond.outs = hw.instance "rhs_in_buf_Cond" @handshake_buffer_0(ins: %in_fork_Cond.outs_1: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %lhs_in_ndw_Cond.outs = hw.instance "lhs_in_ndw_Cond" @handshake_ndwire_0(ins: %lhs_in_buf_Cond.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %rhs_in_ndw_Cond.outs = hw.instance "rhs_in_ndw_Cond" @handshake_ndwire_0(ins: %rhs_in_buf_Cond.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %lhs_passer1.result = hw.instance "lhs_passer1" @handshake_passer_0(data: %lhs_in_ndw_A_in.outs: !handshake.channel<i1>, ctrl: %lhs_in_ndw_Cond.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (result: !handshake.channel<i1>)
    %lhs_fork.outs_0, %lhs_fork.outs_1 = hw.instance "lhs_fork" @handshake_fork_0(ins: %lhs_passer1.result: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs_0: !handshake.channel<i1>, outs_1: !handshake.channel<i1>)
    %rhs_fork.outs_0, %rhs_fork.outs_1 = hw.instance "rhs_fork" @handshake_fork_0(ins: %rhs_in_ndw_A_in.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs_0: !handshake.channel<i1>, outs_1: !handshake.channel<i1>)
    %rhs_fork_cond.outs_0, %rhs_fork_cond.outs_1 = hw.instance "rhs_fork_cond" @handshake_fork_0(ins: %rhs_in_ndw_Cond.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs_0: !handshake.channel<i1>, outs_1: !handshake.channel<i1>)
    %rhs_passer1.result = hw.instance "rhs_passer1" @handshake_passer_0(data: %rhs_fork.outs_0: !handshake.channel<i1>, ctrl: %rhs_fork_cond.outs_0: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (result: !handshake.channel<i1>)
    %rhs_passer2.result = hw.instance "rhs_passer2" @handshake_passer_0(data: %rhs_fork.outs_1: !handshake.channel<i1>, ctrl: %rhs_fork_cond.outs_1: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (result: !handshake.channel<i1>)
    %lhs_out_ndw_B_out.outs = hw.instance "lhs_out_ndw_B_out" @handshake_ndwire_0(ins: %lhs_fork.outs_0: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %rhs_out_ndw_B_out.outs = hw.instance "rhs_out_ndw_B_out" @handshake_ndwire_0(ins: %rhs_passer1.result: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %lhs_out_buf_B_out.outs = hw.instance "lhs_out_buf_B_out" @handshake_buffer_0(ins: %lhs_out_ndw_B_out.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %rhs_out_buf_B_out.outs = hw.instance "rhs_out_buf_B_out" @handshake_buffer_0(ins: %rhs_out_ndw_B_out.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %out_eq_B_out.result = hw.instance "out_eq_B_out" @handshake_cmpi_0(lhs: %lhs_out_buf_B_out.outs: !handshake.channel<i1>, rhs: %rhs_out_buf_B_out.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (result: !handshake.channel<i1>)
    %lhs_out_ndw_C_out.outs = hw.instance "lhs_out_ndw_C_out" @handshake_ndwire_0(ins: %lhs_fork.outs_1: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %rhs_out_ndw_C_out.outs = hw.instance "rhs_out_ndw_C_out" @handshake_ndwire_0(ins: %rhs_passer2.result: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %lhs_out_buf_C_out.outs = hw.instance "lhs_out_buf_C_out" @handshake_buffer_0(ins: %lhs_out_ndw_C_out.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %rhs_out_buf_C_out.outs = hw.instance "rhs_out_buf_C_out" @handshake_buffer_0(ins: %rhs_out_ndw_C_out.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %out_eq_C_out.result = hw.instance "out_eq_C_out" @handshake_cmpi_0(lhs: %lhs_out_buf_C_out.outs: !handshake.channel<i1>, rhs: %rhs_out_buf_C_out.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (result: !handshake.channel<i1>)
    hw.output %out_eq_B_out.result, %out_eq_C_out.result : !handshake.channel<i1>, !handshake.channel<i1>
  }
  hw.module.extern @handshake_lazy_fork_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs_0 : !handshake.channel<i1>, out outs_1 : !handshake.channel<i1>) attributes {hw.name = "handshake.lazy_fork", hw.parameters = {DATA_TYPE = !handshake.channel<i1>, SIZE = 2 : ui32}}
  hw.module.extern @handshake_buffer_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i1>) attributes {hw.name = "handshake.buffer", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_DV", DATA_TYPE = !handshake.channel<i1>, NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}}
  hw.module.extern @handshake_ndwire_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i1>) attributes {hw.name = "handshake.ndwire", hw.parameters = {DATA_TYPE = !handshake.channel<i1>}}
  hw.module.extern @handshake_passer_0(in %data : !handshake.channel<i1>, in %ctrl : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out result : !handshake.channel<i1>) attributes {hw.name = "handshake.passer", hw.parameters = {}}
  hw.module.extern @handshake_fork_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs_0 : !handshake.channel<i1>, out outs_1 : !handshake.channel<i1>) attributes {hw.name = "handshake.fork", hw.parameters = {DATA_TYPE = !handshake.channel<i1>, SIZE = 2 : ui32}}
  hw.module.extern @handshake_cmpi_0(in %lhs : !handshake.channel<i1>, in %rhs : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out result : !handshake.channel<i1>) attributes {hw.name = "handshake.cmpi", hw.parameters = {DATA_TYPE = !handshake.channel<i1>, PREDICATE = "eq"}}
  hw.module @elastic_miter_sup_fork_lhs_sup_fork_rhs_wrapper(in %A_in : !handshake.channel<i1>, in %Cond : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out EQ_B_out : !handshake.channel<i1>, out EQ_C_out : !handshake.channel<i1>) {
    %elastic_miter_sup_fork_lhs_sup_fork_rhs_wrapped.EQ_B_out, %elastic_miter_sup_fork_lhs_sup_fork_rhs_wrapped.EQ_C_out = hw.instance "elastic_miter_sup_fork_lhs_sup_fork_rhs_wrapped" @elastic_miter_sup_fork_lhs_sup_fork_rhs(A_in: %A_in: !handshake.channel<i1>, Cond: %Cond: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (EQ_B_out: !handshake.channel<i1>, EQ_C_out: !handshake.channel<i1>)
    hw.output %elastic_miter_sup_fork_lhs_sup_fork_rhs_wrapped.EQ_B_out, %elastic_miter_sup_fork_lhs_sup_fork_rhs_wrapped.EQ_C_out : !handshake.channel<i1>, !handshake.channel<i1>
  }
}

