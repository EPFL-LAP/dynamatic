module {
  handshake.func @elastic_miter_simpleInduction_lhs_simpleInduction_rhs(%arg0: !handshake.channel<i1>, %arg1: !handshake.channel<i1>, %arg2: !handshake.channel<i1>, ...) -> !handshake.channel<i1> attributes {argNames = ["A_in", "C_in", "D_in"], resNames = ["EQ_B_out"]} {
    %0:2 = lazy_fork [2] %arg0 {handshake.bb = 0 : ui32, handshake.name = "in_fork_A_in"} : <i1>
    %1 = buffer %0#0, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 0 : ui32, handshake.name = "lhs_in_buf_A_in"} : <i1>
    %2 = buffer %0#1, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 0 : ui32, handshake.name = "rhs_in_buf_A_in"} : <i1>
    %3:2 = lazy_fork [2] %arg1 {handshake.bb = 0 : ui32, handshake.name = "in_fork_C_in"} : <i1>
    %4 = buffer %3#0, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 0 : ui32, handshake.name = "lhs_in_buf_C_in"} : <i1>
    %5 = buffer %3#1, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 0 : ui32, handshake.name = "rhs_in_buf_C_in"} : <i1>
    %6:2 = lazy_fork [2] %arg2 {handshake.bb = 0 : ui32, handshake.name = "in_fork_D_in"} : <i1>
    %7 = buffer %6#0, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 0 : ui32, handshake.name = "lhs_in_buf_D_in"} : <i1>
    %8 = buffer %6#1, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 0 : ui32, handshake.name = "rhs_in_buf_D_in"} : <i1>
    %9 = ndsource {handshake.bb = 3 : ui32, handshake.name = "out_nds_B_out"} : <>
    %10:2 = lazy_fork [2] %9 {handshake.bb = 3 : ui32, handshake.name = "out_lf_B_out"} : <>
    %11 = buffer %10#0, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 3 : ui32, handshake.name = "out_buf_lhs_nds_B_out"} : <>
    %12 = buffer %10#1, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 3 : ui32, handshake.name = "out_buf_rhs_nds_B_out"} : <>
    %13 = blocker %17[%11] {handshake.bb = 3 : ui32, handshake.name = "lhs_out_bl_B_out"} : <i1>, <>
    %14 = blocker %21[%12] {handshake.bb = 3 : ui32, handshake.name = "rhs_out_bl_B_out"} : <i1>, <>
    %15 = cmpi eq, %13, %14 {handshake.bb = 3 : ui32, handshake.name = "out_eq_B_out"} : <i1>
    %16 = passer %1[%4] {handshake.bb = 1 : ui32, handshake.name = "lhs_p1"} : <i1>, <i1>
    %17 = passer %16[%7] {handshake.bb = 1 : ui32, handshake.name = "lhs_p2"} : <i1>, <i1>
    %18 = source {handshake.bb = 2 : ui32, handshake.name = "rhs_source"} : <>
    %19 = constant %18 {handshake.bb = 2 : ui32, handshake.name = "rhs_constant", value = false} : <>, <i1>
    %20 = mux %5 [%19, %8] {handshake.bb = 2 : ui32, handshake.name = "rhs_mux"} : <i1>, [<i1>, <i1>] to <i1>
    %21 = passer %2[%20] {handshake.bb = 2 : ui32, handshake.name = "rhs_p1"} : <i1>, <i1>
    end {handshake.bb = 3 : ui32, handshake.name = "end"} %15 : <i1>
  }
}
