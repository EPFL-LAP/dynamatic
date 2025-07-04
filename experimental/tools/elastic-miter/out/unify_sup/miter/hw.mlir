module {
  hw.module @elastic_miter_unify_sup_lhs_unify_sup_rhs(in %A_in : !handshake.channel<i1>, in %Cond1 : !handshake.channel<i1>, in %Cond2 : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out EQ_B_out : !handshake.channel<i1>) {
    %in_fork_A_in.outs_0, %in_fork_A_in.outs_1 = hw.instance "in_fork_A_in" @handshake_lazy_fork_0(ins: %A_in: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs_0: !handshake.channel<i1>, outs_1: !handshake.channel<i1>)
    %lhs_in_buf_A_in.outs = hw.instance "lhs_in_buf_A_in" @handshake_buffer_0(ins: %in_fork_A_in.outs_0: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %rhs_in_buf_A_in.outs = hw.instance "rhs_in_buf_A_in" @handshake_buffer_0(ins: %in_fork_A_in.outs_1: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %lhs_in_ndw_A_in.outs = hw.instance "lhs_in_ndw_A_in" @handshake_ndwire_0(ins: %lhs_in_buf_A_in.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %rhs_in_ndw_A_in.outs = hw.instance "rhs_in_ndw_A_in" @handshake_ndwire_0(ins: %rhs_in_buf_A_in.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %in_fork_Cond1.outs_0, %in_fork_Cond1.outs_1 = hw.instance "in_fork_Cond1" @handshake_lazy_fork_0(ins: %Cond1: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs_0: !handshake.channel<i1>, outs_1: !handshake.channel<i1>)
    %lhs_in_buf_Cond1.outs = hw.instance "lhs_in_buf_Cond1" @handshake_buffer_0(ins: %in_fork_Cond1.outs_0: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %rhs_in_buf_Cond1.outs = hw.instance "rhs_in_buf_Cond1" @handshake_buffer_0(ins: %in_fork_Cond1.outs_1: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %lhs_in_ndw_Cond1.outs = hw.instance "lhs_in_ndw_Cond1" @handshake_ndwire_0(ins: %lhs_in_buf_Cond1.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %rhs_in_ndw_Cond1.outs = hw.instance "rhs_in_ndw_Cond1" @handshake_ndwire_0(ins: %rhs_in_buf_Cond1.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %in_fork_Cond2.outs_0, %in_fork_Cond2.outs_1 = hw.instance "in_fork_Cond2" @handshake_lazy_fork_0(ins: %Cond2: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs_0: !handshake.channel<i1>, outs_1: !handshake.channel<i1>)
    %lhs_in_buf_Cond2.outs = hw.instance "lhs_in_buf_Cond2" @handshake_buffer_0(ins: %in_fork_Cond2.outs_0: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %rhs_in_buf_Cond2.outs = hw.instance "rhs_in_buf_Cond2" @handshake_buffer_0(ins: %in_fork_Cond2.outs_1: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %lhs_in_ndw_Cond2.outs = hw.instance "lhs_in_ndw_Cond2" @handshake_ndwire_0(ins: %lhs_in_buf_Cond2.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %rhs_in_ndw_Cond2.outs = hw.instance "rhs_in_ndw_Cond2" @handshake_ndwire_0(ins: %rhs_in_buf_Cond2.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %lhs_fork_cond2.outs_0, %lhs_fork_cond2.outs_1 = hw.instance "lhs_fork_cond2" @handshake_fork_0(ins: %lhs_in_ndw_Cond2.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs_0: !handshake.channel<i1>, outs_1: !handshake.channel<i1>)
    %lhs_passer1.result = hw.instance "lhs_passer1" @handshake_passer_0(data: %lhs_in_ndw_Cond1.outs: !handshake.channel<i1>, ctrl: %lhs_fork_cond2.outs_0: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (result: !handshake.channel<i1>)
    %lhs_passer2.result = hw.instance "lhs_passer2" @handshake_passer_0(data: %lhs_in_ndw_A_in.outs: !handshake.channel<i1>, ctrl: %lhs_fork_cond2.outs_1: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (result: !handshake.channel<i1>)
    %lhs_passer3.result = hw.instance "lhs_passer3" @handshake_passer_0(data: %lhs_passer2.result: !handshake.channel<i1>, ctrl: %lhs_passer1.result: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (result: !handshake.channel<i1>)
    %rhs_andi.result = hw.instance "rhs_andi" @handshake_andi_0(lhs: %rhs_in_ndw_Cond1.outs: !handshake.channel<i1>, rhs: %rhs_in_ndw_Cond2.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (result: !handshake.channel<i1>)
    %rhs_passer.result = hw.instance "rhs_passer" @handshake_passer_0(data: %rhs_in_ndw_A_in.outs: !handshake.channel<i1>, ctrl: %rhs_andi.result: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (result: !handshake.channel<i1>)
    %lhs_out_ndw_B_out.outs = hw.instance "lhs_out_ndw_B_out" @handshake_ndwire_0(ins: %lhs_passer3.result: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %rhs_out_ndw_B_out.outs = hw.instance "rhs_out_ndw_B_out" @handshake_ndwire_0(ins: %rhs_passer.result: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %lhs_out_buf_B_out.outs = hw.instance "lhs_out_buf_B_out" @handshake_buffer_0(ins: %lhs_out_ndw_B_out.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %rhs_out_buf_B_out.outs = hw.instance "rhs_out_buf_B_out" @handshake_buffer_0(ins: %rhs_out_ndw_B_out.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %out_eq_B_out.result = hw.instance "out_eq_B_out" @handshake_cmpi_0(lhs: %lhs_out_buf_B_out.outs: !handshake.channel<i1>, rhs: %rhs_out_buf_B_out.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (result: !handshake.channel<i1>)
    hw.output %out_eq_B_out.result : !handshake.channel<i1>
  }
  hw.module.extern @handshake_lazy_fork_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs_0 : !handshake.channel<i1>, out outs_1 : !handshake.channel<i1>) attributes {hw.name = "handshake.lazy_fork", hw.parameters = {DATA_TYPE = !handshake.channel<i1>, SIZE = 2 : ui32}}
  hw.module.extern @handshake_buffer_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i1>) attributes {hw.name = "handshake.buffer", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_DV", DATA_TYPE = !handshake.channel<i1>, NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}}
  hw.module.extern @handshake_ndwire_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i1>) attributes {hw.name = "handshake.ndwire", hw.parameters = {DATA_TYPE = !handshake.channel<i1>}}
  hw.module.extern @handshake_fork_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs_0 : !handshake.channel<i1>, out outs_1 : !handshake.channel<i1>) attributes {hw.name = "handshake.fork", hw.parameters = {DATA_TYPE = !handshake.channel<i1>, SIZE = 2 : ui32}}
  hw.module.extern @handshake_passer_0(in %data : !handshake.channel<i1>, in %ctrl : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out result : !handshake.channel<i1>) attributes {hw.name = "handshake.passer", hw.parameters = {}}
  hw.module.extern @handshake_andi_0(in %lhs : !handshake.channel<i1>, in %rhs : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out result : !handshake.channel<i1>) attributes {hw.name = "handshake.andi", hw.parameters = {DATA_TYPE = !handshake.channel<i1>, INTERNAL_DELAY = "0.0"}}
  hw.module.extern @handshake_cmpi_0(in %lhs : !handshake.channel<i1>, in %rhs : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out result : !handshake.channel<i1>) attributes {hw.name = "handshake.cmpi", hw.parameters = {DATA_TYPE = !handshake.channel<i1>, PREDICATE = "eq"}}
  hw.module @elastic_miter_unify_sup_lhs_unify_sup_rhs_wrapper(in %A_in : !handshake.channel<i1>, in %Cond1 : !handshake.channel<i1>, in %Cond2 : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out EQ_B_out : !handshake.channel<i1>) {
    %elastic_miter_unify_sup_lhs_unify_sup_rhs_wrapped.EQ_B_out = hw.instance "elastic_miter_unify_sup_lhs_unify_sup_rhs_wrapped" @elastic_miter_unify_sup_lhs_unify_sup_rhs(A_in: %A_in: !handshake.channel<i1>, Cond1: %Cond1: !handshake.channel<i1>, Cond2: %Cond2: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (EQ_B_out: !handshake.channel<i1>)
    hw.output %elastic_miter_unify_sup_lhs_unify_sup_rhs_wrapped.EQ_B_out : !handshake.channel<i1>
  }
}

