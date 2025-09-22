module {
  hw.module @extension4_lhs(in %arg : !handshake.channel<i1>, in %ctrl : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out res : !handshake.channel<i1>) {
    %passer.result = hw.instance "passer" @handshake_passer_0(data: %arg: !handshake.channel<i1>, ctrl: %ctrl: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (result: !handshake.channel<i1>)
    %initop.outs = hw.instance "initop" @handshake_init_0(ins: %passer.result: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    hw.output %initop.outs : !handshake.channel<i1>
  }
  hw.module.extern @handshake_passer_0(in %data : !handshake.channel<i1>, in %ctrl : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out result : !handshake.channel<i1>) attributes {hw.name = "handshake.passer", hw.parameters = {}}
  hw.module.extern @handshake_init_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i1>) attributes {hw.name = "handshake.init", hw.parameters = {INIT_TOKEN = 0 : ui32}}
  hw.module @extension4_lhs_wrapper(in %arg : !handshake.channel<i1>, in %ctrl : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out res : !handshake.channel<i1>) {
    %extension4_lhs_wrapped.res = hw.instance "extension4_lhs_wrapped" @extension4_lhs(arg: %arg: !handshake.channel<i1>, ctrl: %ctrl: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (res: !handshake.channel<i1>)
    hw.output %extension4_lhs_wrapped.res : !handshake.channel<i1>
  }
}

