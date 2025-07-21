module {
  hw.module @ri_fork_rhs(in %A_in : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out B_out : !handshake.channel<i1>, out C_out : !handshake.channel<i1>) {
    %ndw_in_A_in.outs = hw.instance "ndw_in_A_in" @handshake_ndwire_0(ins: %A_in: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %ndw_out_B_out.outs = hw.instance "ndw_out_B_out" @handshake_ndwire_0(ins: %buffer1.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %ndw_out_C_out.outs = hw.instance "ndw_out_C_out" @handshake_ndwire_0(ins: %buffer2.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %fork.outs_0, %fork.outs_1 = hw.instance "fork" @handshake_fork_0(ins: %ndw_in_A_in.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs_0: !handshake.channel<i1>, outs_1: !handshake.channel<i1>)
    %ri1.outs = hw.instance "ri1" @handshake_spec_v2_repeating_init_0(ins: %fork.outs_0: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %buffer1.outs = hw.instance "buffer1" @handshake_buffer_0(ins: %ri1.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %ri2.outs = hw.instance "ri2" @handshake_spec_v2_repeating_init_0(ins: %fork.outs_1: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %buffer2.outs = hw.instance "buffer2" @handshake_buffer_0(ins: %ri2.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    hw.output %ndw_out_B_out.outs, %ndw_out_C_out.outs : !handshake.channel<i1>, !handshake.channel<i1>
  }
  hw.module.extern @handshake_ndwire_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i1>) attributes {hw.name = "handshake.ndwire", hw.parameters = {DATA_TYPE = !handshake.channel<i1>}}
  hw.module.extern @handshake_fork_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs_0 : !handshake.channel<i1>, out outs_1 : !handshake.channel<i1>) attributes {hw.name = "handshake.fork", hw.parameters = {DATA_TYPE = !handshake.channel<i1>, SIZE = 2 : ui32}}
  hw.module.extern @handshake_spec_v2_repeating_init_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i1>) attributes {hw.name = "handshake.spec_v2_repeating_init", hw.parameters = {INIT_TOKEN = 1 : ui32}}
  hw.module.extern @handshake_buffer_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i1>) attributes {hw.name = "handshake.buffer", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_NONE", DATA_TYPE = !handshake.channel<i1>, NUM_SLOTS = 1 : ui32}}
  hw.module @ri_fork_rhs_wrapper(in %A_in : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out B_out : !handshake.channel<i1>, out C_out : !handshake.channel<i1>) {
    %ri_fork_rhs_wrapped.B_out, %ri_fork_rhs_wrapped.C_out = hw.instance "ri_fork_rhs_wrapped" @ri_fork_rhs(A_in: %A_in: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (B_out: !handshake.channel<i1>, C_out: !handshake.channel<i1>)
    hw.output %ri_fork_rhs_wrapped.B_out, %ri_fork_rhs_wrapped.C_out : !handshake.channel<i1>, !handshake.channel<i1>
  }
}

