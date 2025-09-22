module {
  hw.module @extension3_lhs(in %arg : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out res : !handshake.channel<i1>) {
    %initop.outs = hw.instance "initop" @handshake_init_0(ins: %arg: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    hw.output %initop.outs : !handshake.channel<i1>
  }
  hw.module.extern @handshake_init_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i1>) attributes {hw.name = "handshake.init", hw.parameters = {INIT_TOKEN = 0 : ui32}}
  hw.module @extension3_lhs_wrapper(in %arg : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out res : !handshake.channel<i1>) {
    %extension3_lhs_wrapped.res = hw.instance "extension3_lhs_wrapped" @extension3_lhs(arg: %arg: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (res: !handshake.channel<i1>)
    hw.output %extension3_lhs_wrapped.res : !handshake.channel<i1>
  }
}

