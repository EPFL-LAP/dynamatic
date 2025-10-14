module {
  hw.module @mux_to_and_rhs(in %A_in : !handshake.channel<i1>, in %B_in : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out C_out : !handshake.channel<i1>) {
    %andi1.result = hw.instance "andi1" @handshake_andi_0(lhs: %A_in: !handshake.channel<i1>, rhs: %B_in: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (result: !handshake.channel<i1>)
    hw.output %andi1.result : !handshake.channel<i1>
  }
  hw.module.extern @handshake_andi_0(in %lhs : !handshake.channel<i1>, in %rhs : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out result : !handshake.channel<i1>) attributes {hw.name = "handshake.andi", hw.parameters = {DATA_TYPE = !handshake.channel<i1>}}
  hw.module @mux_to_and_rhs_wrapper(in %A_in : !handshake.channel<i1>, in %B_in : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out C_out : !handshake.channel<i1>) {
    %mux_to_and_rhs_wrapped.C_out = hw.instance "mux_to_and_rhs_wrapped" @mux_to_and_rhs(A_in: %A_in: !handshake.channel<i1>, B_in: %B_in: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (C_out: !handshake.channel<i1>)
    hw.output %mux_to_and_rhs_wrapped.C_out : !handshake.channel<i1>
  }
}

