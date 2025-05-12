module {
  hw.module @matching(in %num_edges : !handshake.channel<i32>, in %start : !handshake.control<>, in %clk : i1, in %rst : i1, out out0 : !handshake.channel<i32>, out end : !handshake.control<>) {
    hw.output %num_edges, %start : !handshake.channel<i32>, !handshake.control<>
  }
  hw.module @matching_wrapper(in %num_edges : !handshake.channel<i32>, in %start : !handshake.control<>, in %clk : i1, in %rst : i1, out out0 : !handshake.channel<i32>, out end : !handshake.control<>) {
    %matching_wrapped.out0, %matching_wrapped.end = hw.instance "matching_wrapped" @matching(num_edges: %num_edges: !handshake.channel<i32>, start: %start: !handshake.control<>, clk: %clk: i1, rst: %rst: i1) -> (out0: !handshake.channel<i32>, end: !handshake.control<>)
    hw.output %matching_wrapped.out0, %matching_wrapped.end : !handshake.channel<i32>, !handshake.control<>
  }
}

