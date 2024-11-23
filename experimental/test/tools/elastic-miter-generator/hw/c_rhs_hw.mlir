module {
  hw.module @c_rhs(in %D : !handshake.channel<i32>, in %C : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out A : !handshake.channel<i1>, out B : !handshake.channel<i32>) {
    hw.output %C, %D : !handshake.channel<i1>, !handshake.channel<i32>
  }
  hw.module @c_rhs_wrapper(in %D : !handshake.channel<i32>, in %C : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out A : !handshake.channel<i1>, out B : !handshake.channel<i32>) {
    %c_rhs_wrapped.A, %c_rhs_wrapped.B = hw.instance "c_rhs_wrapped" @c_rhs(D: %D: !handshake.channel<i32>, C: %C: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (A: !handshake.channel<i1>, B: !handshake.channel<i32>)
    hw.output %c_rhs_wrapped.A, %c_rhs_wrapped.B : !handshake.channel<i1>, !handshake.channel<i32>
  }
}

