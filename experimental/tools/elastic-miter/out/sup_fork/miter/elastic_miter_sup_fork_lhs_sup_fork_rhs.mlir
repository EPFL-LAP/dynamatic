module {
  handshake.func @elastic_miter_sup_fork_lhs_sup_fork_rhs(%arg0: !handshake.channel<i1>, %arg1: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>, !handshake.channel<i1>) attributes {argNames = ["A_in", "Cond"], resNames = ["EQ_B_out", "EQ_C_out"]} {
    %0:2 = lazy_fork [2] %arg0 {handshake.bb = 0 : ui32, handshake.name = "in_fork_A_in"} : <i1>
    %1 = buffer %0#0, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 0 : ui32, handshake.name = "lhs_in_buf_A_in"} : <i1>
    %2 = buffer %0#1, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 0 : ui32, handshake.name = "rhs_in_buf_A_in"} : <i1>
    %3:2 = lazy_fork [2] %arg1 {handshake.bb = 0 : ui32, handshake.name = "in_fork_Cond"} : <i1>
    %4 = buffer %3#0, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 0 : ui32, handshake.name = "lhs_in_buf_Cond"} : <i1>
    %5 = buffer %3#1, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 0 : ui32, handshake.name = "rhs_in_buf_Cond"} : <i1>
    %6 = ndsource {handshake.bb = 3 : ui32, handshake.name = "out_nds_B_out"} : <>
    %7:2 = lazy_fork [2] %6 {handshake.bb = 3 : ui32, handshake.name = "out_lf_B_out"} : <>
    %8 = buffer %7#0, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 3 : ui32, handshake.name = "out_buf_lhs_nds_B_out"} : <>
    %9 = buffer %7#1, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 3 : ui32, handshake.name = "out_buf_rhs_nds_B_out"} : <>
    %10 = blocker %21#0[%8] {handshake.bb = 3 : ui32, handshake.name = "lhs_out_bl_B_out"} : <i1>, <>
    %11 = blocker %24[%9] {handshake.bb = 3 : ui32, handshake.name = "rhs_out_bl_B_out"} : <i1>, <>
    %12 = cmpi eq, %10, %11 {handshake.bb = 3 : ui32, handshake.name = "out_eq_B_out"} : <i1>
    %13 = ndsource {handshake.bb = 3 : ui32, handshake.name = "out_nds_C_out"} : <>
    %14:2 = lazy_fork [2] %13 {handshake.bb = 3 : ui32, handshake.name = "out_lf_C_out"} : <>
    %15 = buffer %14#0, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 3 : ui32, handshake.name = "out_buf_lhs_nds_C_out"} : <>
    %16 = buffer %14#1, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 3 : ui32, handshake.name = "out_buf_rhs_nds_C_out"} : <>
    %17 = blocker %21#1[%15] {handshake.bb = 3 : ui32, handshake.name = "lhs_out_bl_C_out"} : <i1>, <>
    %18 = blocker %25[%16] {handshake.bb = 3 : ui32, handshake.name = "rhs_out_bl_C_out"} : <i1>, <>
    %19 = cmpi eq, %17, %18 {handshake.bb = 3 : ui32, handshake.name = "out_eq_C_out"} : <i1>
    %20 = passer %1[%4] {handshake.bb = 1 : ui32, handshake.name = "lhs_passer1"} : <i1>, <i1>
    %21:2 = fork [2] %20 {handshake.bb = 1 : ui32, handshake.name = "lhs_fork"} : <i1>
    %22:2 = fork [2] %2 {handshake.bb = 2 : ui32, handshake.name = "rhs_fork"} : <i1>
    %23:2 = fork [2] %5 {handshake.bb = 2 : ui32, handshake.name = "rhs_fork_cond"} : <i1>
    %24 = passer %22#0[%23#0] {handshake.bb = 2 : ui32, handshake.name = "rhs_passer1"} : <i1>, <i1>
    %25 = passer %22#1[%23#1] {handshake.bb = 2 : ui32, handshake.name = "rhs_passer2"} : <i1>, <i1>
    end {handshake.bb = 3 : ui32, handshake.name = "end"} %12, %19 : <i1>, <i1>
  }
}
