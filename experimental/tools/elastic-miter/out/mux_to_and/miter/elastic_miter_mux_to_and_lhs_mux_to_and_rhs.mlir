module {
  handshake.func @elastic_miter_mux_to_and_lhs_mux_to_and_rhs(%arg0: !handshake.channel<i1>, %arg1: !handshake.channel<i1>, ...) -> !handshake.channel<i1> attributes {argNames = ["A_in", "B_in"], resNames = ["EQ_C_out"]} {
    %0:2 = lazy_fork [2] %arg0 {handshake.bb = 0 : ui32, handshake.name = "in_fork_A_in"} : <i1>
    %1 = buffer %0#0, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 0 : ui32, handshake.name = "lhs_in_buf_A_in"} : <i1>
    %2 = buffer %0#1, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 0 : ui32, handshake.name = "rhs_in_buf_A_in"} : <i1>
    %3:2 = lazy_fork [2] %arg1 {handshake.bb = 0 : ui32, handshake.name = "in_fork_B_in"} : <i1>
    %4 = buffer %3#0, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 0 : ui32, handshake.name = "lhs_in_buf_B_in"} : <i1>
    %5 = buffer %3#1, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 0 : ui32, handshake.name = "rhs_in_buf_B_in"} : <i1>
    %6 = source {handshake.bb = 3 : ui32, handshake.name = "out_src_C_out"} : <>
    %7:2 = lazy_fork [2] %6 {handshake.bb = 3 : ui32, handshake.name = "out_lf_C_out"} : <>
    %8 = buffer %7#0, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 3 : ui32, handshake.name = "out_buf_lhs_nds_C_out"} : <>
    %9 = buffer %7#1, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 3 : ui32, handshake.name = "out_buf_rhs_nds_C_out"} : <>
    %10 = blocker %17[%8] {handshake.bb = 3 : ui32, handshake.name = "lhs_out_bl_C_out"} : <i1>, <>
    %11 = blocker %18[%9] {handshake.bb = 3 : ui32, handshake.name = "rhs_out_bl_C_out"} : <i1>, <>
    %12 = cmpi eq, %10, %11 {handshake.bb = 3 : ui32, handshake.name = "out_eq_C_out"} : <i1>
    %13 = source {handshake.bb = 1 : ui32, handshake.name = "lhs_source"} : <>
    %14 = constant %13 {handshake.bb = 1 : ui32, handshake.name = "lhs_constant", value = false} : <>, <i1>
    %15:2 = fork [2] %1 {handshake.bb = 1 : ui32, handshake.name = "lhs_a_fork"} : <i1>
    %16 = passer %4[%15#0] {handshake.bb = 1 : ui32, handshake.name = "lhs_passer1"} : <i1>, <i1>
    %17 = mux %15#1 [%14, %16] {handshake.bb = 1 : ui32, handshake.name = "lhs_mux"} : <i1>, [<i1>, <i1>] to <i1>
    %18 = andi %2, %5 {handshake.bb = 2 : ui32, handshake.name = "rhs_andi1"} : <i1>
    end {handshake.bb = 3 : ui32, handshake.name = "end"} %12 : <i1>
  }
}
