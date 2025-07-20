module {
  hw.module @elastic_miter_add_init_lhs_add_init_lhs(in %val : !handshake.channel<i1>, in %clk : i1, in %rst : i1) {
    %in_fork_val.outs_0, %in_fork_val.outs_1 = hw.instance "in_fork_val" @handshake_lazy_fork_0(ins: %val: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs_0: !handshake.channel<i1>, outs_1: !handshake.channel<i1>)
    %lhs_in_buf_val.outs = hw.instance "lhs_in_buf_val" @handshake_buffer_0(ins: %in_fork_val.outs_0: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %rhs_in_buf_val.outs = hw.instance "rhs_in_buf_val" @handshake_buffer_0(ins: %in_fork_val.outs_1: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %lhs_in_ndw_val.outs = hw.instance "lhs_in_ndw_val" @handshake_ndwire_0(ins: %lhs_in_buf_val.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %rhs_in_ndw_val.outs = hw.instance "rhs_in_ndw_val" @handshake_ndwire_0(ins: %rhs_in_buf_val.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    hw.instance "lhs_vm_sink_0" @handshake_sink_0(ins: %lhs_in_ndw_val.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> ()
    %rhs_init0.outs = hw.instance "rhs_init0" @handshake_init_0(ins: %rhs_in_ndw_val.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    hw.instance "rhs_sink" @handshake_sink_0(ins: %rhs_init0.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> ()
    hw.output
  }
  hw.module.extern @handshake_lazy_fork_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs_0 : !handshake.channel<i1>, out outs_1 : !handshake.channel<i1>) attributes {hw.name = "handshake.lazy_fork", hw.parameters = {DATA_TYPE = !handshake.channel<i1>, SIZE = 2 : ui32}}
  hw.module.extern @handshake_buffer_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i1>) attributes {hw.name = "handshake.buffer", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_DV", DATA_TYPE = !handshake.channel<i1>, NUM_SLOTS = 1 : ui32}}
  hw.module.extern @handshake_ndwire_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i1>) attributes {hw.name = "handshake.ndwire", hw.parameters = {DATA_TYPE = !handshake.channel<i1>}}
  hw.module.extern @handshake_sink_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1) attributes {hw.name = "handshake.sink", hw.parameters = {DATA_TYPE = !handshake.channel<i1>}}
  hw.module.extern @handshake_init_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i1>) attributes {hw.name = "handshake.init", hw.parameters = {INIT_TOKEN = 0 : ui32}}
  hw.module @elastic_miter_add_init_lhs_add_init_lhs_wrapper(in %val : !handshake.channel<i1>, in %clk : i1, in %rst : i1) {
    hw.instance "elastic_miter_add_init_lhs_add_init_lhs_wrapped" @elastic_miter_add_init_lhs_add_init_lhs(val: %val: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> ()
    hw.output
  }
}

