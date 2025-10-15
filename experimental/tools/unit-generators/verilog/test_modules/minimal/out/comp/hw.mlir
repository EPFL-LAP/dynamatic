module {
  hw.module @minimal(in %x : !handshake.channel<i32>, in %start : !handshake.control<>, in %clk : i1, in %rst : i1, out out0 : !handshake.channel<i32>, out end : !handshake.control<>) {
    hw.output %x, %start : !handshake.channel<i32>, !handshake.control<>
  }
  hw.module @minimal_wrapper(in %x : !handshake.channel<i32>, in %start : !handshake.control<>, in %clk : i1, in %rst : i1, out out0 : !handshake.channel<i32>, out end : !handshake.control<>) {
    %minimal_wrapped.out0, %minimal_wrapped.end = hw.instance "minimal_wrapped" @minimal(x: %x: !handshake.channel<i32>, start: %start: !handshake.control<>, clk: %clk: i1, rst: %rst: i1) -> (out0: !handshake.channel<i32>, end: !handshake.control<>)
    hw.output %minimal_wrapped.out0, %minimal_wrapped.end : !handshake.channel<i32>, !handshake.control<>
  }
}

