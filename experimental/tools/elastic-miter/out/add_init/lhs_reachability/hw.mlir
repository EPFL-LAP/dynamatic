module {
  hw.module @add_init_lhs(in %val : !handshake.channel<i1>, in %clk : i1, in %rst : i1) {
    %ndw_in_val.outs = hw.instance "ndw_in_val" @handshake_ndwire_0(ins: %val: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    hw.instance "vm_sink_0" @handshake_sink_0(ins: %ndw_in_val.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> ()
    hw.output
  }
  hw.module.extern @handshake_ndwire_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i1>) attributes {hw.name = "handshake.ndwire", hw.parameters = {DATA_TYPE = !handshake.channel<i1>}}
  hw.module.extern @handshake_sink_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1) attributes {hw.name = "handshake.sink", hw.parameters = {DATA_TYPE = !handshake.channel<i1>}}
  hw.module @add_init_lhs_wrapper(in %val : !handshake.channel<i1>, in %clk : i1, in %rst : i1) {
    hw.instance "add_init_lhs_wrapped" @add_init_lhs(val: %val: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> ()
    hw.output
  }
}

