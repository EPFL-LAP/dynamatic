module {
  hw.module @extension2_lhs(in %arg : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out res : !handshake.channel<i1>) {
    %not1.outs = hw.instance "not1" @handshake_not_0(ins: %arg: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %not2.outs = hw.instance "not2" @handshake_not_0(ins: %not1.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    hw.output %not2.outs : !handshake.channel<i1>
  }
  hw.module.extern @handshake_not_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i1>) attributes {hw.name = "handshake.not", hw.parameters = {DATA_TYPE = !handshake.channel<i1>}}
  hw.module @extension2_lhs_wrapper(in %arg : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out res : !handshake.channel<i1>) {
    %extension2_lhs_wrapped.res = hw.instance "extension2_lhs_wrapped" @extension2_lhs(arg: %arg: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (res: !handshake.channel<i1>)
    hw.output %extension2_lhs_wrapped.res : !handshake.channel<i1>
  }
}

