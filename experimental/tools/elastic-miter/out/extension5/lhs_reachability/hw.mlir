module {
  hw.module @extension5_lhs(in %arg : !handshake.channel<i1>, in %ctrl : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out res : !handshake.channel<i1>) {
    %passer.result = hw.instance "passer" @handshake_passer_0(data: %arg: !handshake.channel<i1>, ctrl: %ctrl: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (result: !handshake.channel<i1>)
    hw.output %passer.result : !handshake.channel<i1>
  }
  hw.module.extern @handshake_passer_0(in %data : !handshake.channel<i1>, in %ctrl : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out result : !handshake.channel<i1>) attributes {hw.name = "handshake.passer", hw.parameters = {}}
  hw.module @extension5_lhs_wrapper(in %arg : !handshake.channel<i1>, in %ctrl : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out res : !handshake.channel<i1>) {
    %extension5_lhs_wrapped.res = hw.instance "extension5_lhs_wrapped" @extension5_lhs(arg: %arg: !handshake.channel<i1>, ctrl: %ctrl: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (res: !handshake.channel<i1>)
    hw.output %extension5_lhs_wrapped.res : !handshake.channel<i1>
  }
}

