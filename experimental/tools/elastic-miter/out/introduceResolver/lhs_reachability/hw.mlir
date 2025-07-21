module {
  hw.module @introduceResolver_lhs(in %loopContinue : !handshake.channel<i1>, in %confirmSpec_backedge : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out confirmSpec : !handshake.channel<i1>) {
    %backedge_source_confirmSpec.outs = hw.instance "backedge_source_confirmSpec" @handshake_source_0(clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.control<>)
    %backedge_constant_confirmSpec.result = hw.instance "backedge_constant_confirmSpec" @handshake_ndconstant_0(ctrl: %backedge_source_confirmSpec.outs: !handshake.control<>, clk: %clk: i1, rst: %rst: i1) -> (result: !handshake.channel<i1>)
    %backedge_lf_start_confirmSpec.outs_0, %backedge_lf_start_confirmSpec.outs_1 = hw.instance "backedge_lf_start_confirmSpec" @handshake_fork_0(ins: %backedge_constant_confirmSpec.result: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs_0: !handshake.channel<i1>, outs_1: !handshake.channel<i1>)
    hw.instance "backedge_sink_start_confirmSpec" @handshake_sink_0(ins: %confirmSpec_backedge: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> ()
    %backedge_lf_end_confirmSpec.outs_0, %backedge_lf_end_confirmSpec.outs_1 = hw.instance "backedge_lf_end_confirmSpec" @handshake_lazy_fork_0(ins: %ndw_out_confirmSpec.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs_0: !handshake.channel<i1>, outs_1: !handshake.channel<i1>)
    %backedge_eq_confirmSpec.result = hw.instance "backedge_eq_confirmSpec" @handshake_cmpi_0(lhs: %backedge_lf_start_confirmSpec.outs_1: !handshake.channel<i1>, rhs: %backedge_lf_end_confirmSpec.outs_1: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (result: !handshake.channel<i1>)
    hw.instance "backedge_sink_end_confirmSpec" @handshake_sink_0(ins: %backedge_eq_confirmSpec.result: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> ()
    %ndw_in_loopContinue.outs = hw.instance "ndw_in_loopContinue" @handshake_ndwire_0(ins: %loopContinue: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %ndw_in_confirmSpec_backedge.outs = hw.instance "ndw_in_confirmSpec_backedge" @handshake_ndwire_0(ins: %backedge_lf_start_confirmSpec.outs_0: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %ndw_out_confirmSpec.outs = hw.instance "ndw_out_confirmSpec" @handshake_ndwire_0(ins: %interpolate.result: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %ctx_fork.outs_0, %ctx_fork.outs_1 = hw.instance "ctx_fork" @handshake_fork_0(ins: %ndw_in_loopContinue.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs_0: !handshake.channel<i1>, outs_1: !handshake.channel<i1>)
    %ctx_fork_cs.outs_0, %ctx_fork_cs.outs_1 = hw.instance "ctx_fork_cs" @handshake_fork_0(ins: %ndw_in_confirmSpec_backedge.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs_0: !handshake.channel<i1>, outs_1: !handshake.channel<i1>)
    %ctx_passer.result = hw.instance "ctx_passer" @handshake_passer_0(data: %ctx_fork.outs_0: !handshake.channel<i1>, ctrl: %ctx_fork_cs.outs_0: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (result: !handshake.channel<i1>)
    %ndspec.outs = hw.instance "ndspec" @handshake_spec_v2_nd_speculator_0(ins: %ctx_passer.result: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %passer.result = hw.instance "passer" @handshake_passer_0(data: %ctx_fork.outs_1: !handshake.channel<i1>, ctrl: %ctx_fork_cs.outs_1: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (result: !handshake.channel<i1>)
    %ri.outs = hw.instance "ri" @handshake_spec_v2_repeating_init_0(ins: %passer.result: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %buffer.outs = hw.instance "buffer" @handshake_buffer_0(ins: %ri.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %interpolate.result = hw.instance "interpolate" @handshake_spec_v2_interpolator_0(short: %buffer.outs: !handshake.channel<i1>, long: %ndspec.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (result: !handshake.channel<i1>)
    hw.output %backedge_lf_end_confirmSpec.outs_0 : !handshake.channel<i1>
  }
  hw.module.extern @handshake_source_0(in %clk : i1, in %rst : i1, out outs : !handshake.control<>) attributes {hw.name = "handshake.source", hw.parameters = {}}
  hw.module.extern @handshake_ndconstant_0(in %ctrl : !handshake.control<>, in %clk : i1, in %rst : i1, out result : !handshake.channel<i1>) attributes {hw.name = "handshake.ndconstant", hw.parameters = {}}
  hw.module.extern @handshake_fork_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs_0 : !handshake.channel<i1>, out outs_1 : !handshake.channel<i1>) attributes {hw.name = "handshake.fork", hw.parameters = {DATA_TYPE = !handshake.channel<i1>, SIZE = 2 : ui32}}
  hw.module.extern @handshake_sink_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1) attributes {hw.name = "handshake.sink", hw.parameters = {DATA_TYPE = !handshake.channel<i1>}}
  hw.module.extern @handshake_lazy_fork_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs_0 : !handshake.channel<i1>, out outs_1 : !handshake.channel<i1>) attributes {hw.name = "handshake.lazy_fork", hw.parameters = {DATA_TYPE = !handshake.channel<i1>, SIZE = 2 : ui32}}
  hw.module.extern @handshake_cmpi_0(in %lhs : !handshake.channel<i1>, in %rhs : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out result : !handshake.channel<i1>) attributes {hw.name = "handshake.cmpi", hw.parameters = {DATA_TYPE = !handshake.channel<i1>, INTERNAL_DELAY = "0.0", PREDICATE = "eq"}}
  hw.module.extern @handshake_ndwire_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i1>) attributes {hw.name = "handshake.ndwire", hw.parameters = {DATA_TYPE = !handshake.channel<i1>}}
  hw.module.extern @handshake_passer_0(in %data : !handshake.channel<i1>, in %ctrl : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out result : !handshake.channel<i1>) attributes {hw.name = "handshake.passer", hw.parameters = {}}
  hw.module.extern @handshake_spec_v2_nd_speculator_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i1>) attributes {hw.name = "handshake.spec_v2_nd_speculator", hw.parameters = {}}
  hw.module.extern @handshake_spec_v2_repeating_init_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i1>) attributes {hw.name = "handshake.spec_v2_repeating_init", hw.parameters = {INIT_TOKEN = 1 : ui32}}
  hw.module.extern @handshake_buffer_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i1>) attributes {hw.name = "handshake.buffer", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_NONE", DATA_TYPE = !handshake.channel<i1>, NUM_SLOTS = 1 : ui32}}
  hw.module.extern @handshake_spec_v2_interpolator_0(in %short : !handshake.channel<i1>, in %long : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out result : !handshake.channel<i1>) attributes {hw.name = "handshake.spec_v2_interpolator", hw.parameters = {}}
  hw.module @introduceResolver_lhs_wrapper(in %loopContinue : !handshake.channel<i1>, in %confirmSpec_backedge : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out confirmSpec : !handshake.channel<i1>) {
    %introduceResolver_lhs_wrapped.confirmSpec = hw.instance "introduceResolver_lhs_wrapped" @introduceResolver_lhs(loopContinue: %loopContinue: !handshake.channel<i1>, confirmSpec_backedge: %confirmSpec_backedge: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (confirmSpec: !handshake.channel<i1>)
    hw.output %introduceResolver_lhs_wrapped.confirmSpec : !handshake.channel<i1>
  }
}

