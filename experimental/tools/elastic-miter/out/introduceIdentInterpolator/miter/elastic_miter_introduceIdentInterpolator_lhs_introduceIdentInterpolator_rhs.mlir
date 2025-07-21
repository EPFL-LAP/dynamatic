module {
  handshake.func @elastic_miter_introduceIdentInterpolator_lhs_introduceIdentInterpolator_rhs(%arg0: !handshake.channel<i1>, %arg1: !handshake.channel<i1>, ...) -> !handshake.channel<i1> attributes {argNames = ["A_in", "val"], resNames = ["EQ_B_out"]} {
    %0:2 = lazy_fork [2] %arg0 {handshake.bb = 0 : ui32, handshake.name = "in_fork_A_in"} : <i1>
    %1 = buffer %0#0, bufferType = FIFO_BREAK_DV, numSlots = 1 {handshake.bb = 0 : ui32, handshake.name = "lhs_in_buf_A_in"} : <i1>
    %2 = buffer %0#1, bufferType = FIFO_BREAK_DV, numSlots = 1 {handshake.bb = 0 : ui32, handshake.name = "rhs_in_buf_A_in"} : <i1>
    %3:2 = lazy_fork [2] %arg1 {handshake.bb = 0 : ui32, handshake.name = "in_fork_val"} : <i1>
    %4 = buffer %3#0, bufferType = FIFO_BREAK_DV, numSlots = 1 {handshake.bb = 0 : ui32, handshake.name = "lhs_in_buf_val"} : <i1>
    %5 = buffer %3#1, bufferType = FIFO_BREAK_DV, numSlots = 1 {handshake.bb = 0 : ui32, handshake.name = "rhs_in_buf_val"} : <i1>
    %6 = ndsource {handshake.bb = 3 : ui32, handshake.name = "out_nds_B_out"} : <>
    %7:2 = lazy_fork [2] %6 {handshake.bb = 3 : ui32, handshake.name = "out_lf_B_out"} : <>
    %8 = buffer %7#0, bufferType = FIFO_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "out_buf_lhs_nds_B_out"} : <>
    %9 = buffer %7#1, bufferType = FIFO_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "out_buf_rhs_nds_B_out"} : <>
    %10 = blocker %13[%8] {handshake.bb = 3 : ui32, handshake.name = "lhs_out_bl_B_out"} : <i1>, <>
    %11 = blocker %16[%9] {handshake.bb = 3 : ui32, handshake.name = "rhs_out_bl_B_out"} : <i1>, <>
    %12 = cmpi eq, %10, %11 {handshake.bb = 3 : ui32, handshake.name = "out_eq_B_out"} : <i1>
    %13 = passer %1[%4] {handshake.bb = 1 : ui32, handshake.name = "lhs_passer"} : <i1>, <i1>
    %14:2 = fork [2] %5 {handshake.bb = 2 : ui32, handshake.name = "rhs_vm_fork_1"} : <i1>
    %15 = spec_v2_interpolator %14#0, %14#1 {handshake.bb = 2 : ui32, handshake.name = "rhs_interpolate"} : <i1>
    %16 = passer %2[%15] {handshake.bb = 2 : ui32, handshake.name = "rhs_passer"} : <i1>, <i1>
    end {handshake.bb = 3 : ui32, handshake.name = "end"} %12 : <i1>
  }
}
