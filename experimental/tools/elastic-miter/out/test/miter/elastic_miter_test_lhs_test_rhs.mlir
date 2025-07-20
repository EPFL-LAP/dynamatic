module {
  handshake.func @elastic_miter_test_lhs_test_rhs(%arg0: !handshake.control<>, ...) attributes {argNames = ["A_in"], resNames = []} {
    %0:2 = lazy_fork [2] %arg0 {handshake.bb = 0 : ui32, handshake.name = "in_fork_A_in"} : <>
    %1 = buffer %0#0, bufferType = FIFO_BREAK_DV, numSlots = 1 {handshake.bb = 0 : ui32, handshake.name = "lhs_in_buf_A_in"} : <>
    %2 = buffer %0#1, bufferType = FIFO_BREAK_DV, numSlots = 1 {handshake.bb = 0 : ui32, handshake.name = "rhs_in_buf_A_in"} : <>
    %3 = ndsource {handshake.bb = 1 : ui32, handshake.name = "lhs_nds"} : <>
    %4 = join %1, %3 {handshake.bb = 1 : ui32, handshake.name = "lhs_join"} : <>
    sink %4 {handshake.bb = 1 : ui32, handshake.name = "lhs_sink"} : <>
    %5 = ndsource {handshake.bb = 3 : ui32, handshake.name = "rhs_nds"} : <>
    %6 = join %2, %5 {handshake.bb = 3 : ui32, handshake.name = "rhs_join"} : <>
    sink %6 {handshake.bb = 3 : ui32, handshake.name = "rhs_sink"} : <>
    end {handshake.bb = 5 : ui32, handshake.name = "end"}
  }
}
