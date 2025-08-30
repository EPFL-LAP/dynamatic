module {
  handshake.func @elastic_miter_unify_sup_lhs_unify_sup_rhs(%arg0: !handshake.channel<i1>, %arg1: !handshake.channel<i1>, %arg2: !handshake.channel<i1>, ...) -> !handshake.channel<i1> attributes {argNames = ["A_in", "Cond1", "Cond2"], resNames = ["EQ_B_out"]} {
    %0:2 = lazy_fork [2] %arg0 {handshake.bb = 0 : ui32, handshake.name = "in_fork_A_in"} : <i1>
    %1 = buffer %0#0, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 0 : ui32, handshake.name = "lhs_in_buf_A_in"} : <i1>
    %2 = buffer %0#1, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 0 : ui32, handshake.name = "rhs_in_buf_A_in"} : <i1>
    %3:2 = lazy_fork [2] %arg1 {handshake.bb = 0 : ui32, handshake.name = "in_fork_Cond1"} : <i1>
    %4 = buffer %3#0, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 0 : ui32, handshake.name = "lhs_in_buf_Cond1"} : <i1>
    %5 = buffer %3#1, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 0 : ui32, handshake.name = "rhs_in_buf_Cond1"} : <i1>
    %6:2 = lazy_fork [2] %arg2 {handshake.bb = 0 : ui32, handshake.name = "in_fork_Cond2"} : <i1>
    %7 = buffer %6#0, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 0 : ui32, handshake.name = "lhs_in_buf_Cond2"} : <i1>
    %8 = buffer %6#1, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 0 : ui32, handshake.name = "rhs_in_buf_Cond2"} : <i1>
    %9 = ndsource {handshake.bb = 5 : ui32, handshake.name = "out_nds_B_out"} : <>
    %10:2 = lazy_fork [2] %9 {handshake.bb = 5 : ui32, handshake.name = "out_lf_B_out"} : <>
    %11 = buffer %10#0, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 5 : ui32, handshake.name = "out_buf_lhs_nds_B_out"} : <>
    %12 = buffer %10#1, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 5 : ui32, handshake.name = "out_buf_rhs_nds_B_out"} : <>
    %13 = blocker %19[%11] {handshake.bb = 5 : ui32, handshake.name = "lhs_out_bl_B_out"} : <i1>, <>
    %14 = blocker %21[%12] {handshake.bb = 5 : ui32, handshake.name = "rhs_out_bl_B_out"} : <i1>, <>
    %15 = cmpi eq, %13, %14 {handshake.bb = 5 : ui32, handshake.name = "out_eq_B_out"} : <i1>
    %16:2 = fork [2] %7 {handshake.bb = 1 : ui32, handshake.name = "lhs_vm_fork_2"} : <i1>
    %17 = passer %4[%16#0] {handshake.bb = 2 : ui32, handshake.name = "lhs_passer1"} : <i1>, <i1>
    %18 = passer %1[%16#1] {handshake.bb = 2 : ui32, handshake.name = "lhs_passer2"} : <i1>, <i1>
    %19 = passer %18[%17] {handshake.bb = 2 : ui32, handshake.name = "lhs_passer3"} : <i1>, <i1>
    %20 = andi %5, %8 {handshake.bb = 4 : ui32, handshake.name = "rhs_andi"} : <i1>
    %21 = passer %2[%20] {handshake.bb = 4 : ui32, handshake.name = "rhs_passer"} : <i1>, <i1>
    end {handshake.bb = 5 : ui32, handshake.name = "end"} %15 : <i1>
  }
}
