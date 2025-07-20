module {
  handshake.func @elastic_miter_introduceIdentInterpolator_lhs_introduceIdentInterpolator_rhs(%arg0: !handshake.channel<i1>, %arg1: !handshake.channel<i1>, ...) -> !handshake.channel<i1> attributes {argNames = ["A_in", "val"], resNames = ["EQ_B_out"]} {
    %0:2 = lazy_fork [2] %arg0 {handshake.bb = 0 : ui32, handshake.name = "in_fork_A_in"} : <i1>
    %1 = buffer %0#0, bufferType = FIFO_BREAK_DV, numSlots = 1 {handshake.bb = 0 : ui32, handshake.name = "lhs_in_buf_A_in"} : <i1>
    %2 = buffer %0#1, bufferType = FIFO_BREAK_DV, numSlots = 1 {handshake.bb = 0 : ui32, handshake.name = "rhs_in_buf_A_in"} : <i1>
    %3 = ndwire %1 {handshake.bb = 0 : ui32, handshake.name = "lhs_in_ndw_A_in"} : <i1>
    %4 = ndwire %2 {handshake.bb = 0 : ui32, handshake.name = "rhs_in_ndw_A_in"} : <i1>
    %5:2 = lazy_fork [2] %arg1 {handshake.bb = 0 : ui32, handshake.name = "in_fork_val"} : <i1>
    %6 = buffer %5#0, bufferType = FIFO_BREAK_DV, numSlots = 1 {handshake.bb = 0 : ui32, handshake.name = "lhs_in_buf_val"} : <i1>
    %7 = buffer %5#1, bufferType = FIFO_BREAK_DV, numSlots = 1 {handshake.bb = 0 : ui32, handshake.name = "rhs_in_buf_val"} : <i1>
    %8 = ndwire %6 {handshake.bb = 0 : ui32, handshake.name = "lhs_in_ndw_val"} : <i1>
    %9 = ndwire %7 {handshake.bb = 0 : ui32, handshake.name = "rhs_in_ndw_val"} : <i1>
    %10 = passer %3[%8] {handshake.bb = 1 : ui32, handshake.name = "lhs_passer"} : <i1>, <i1>
    %11:2 = fork [2] %9 {handshake.bb = 2 : ui32, handshake.name = "rhs_vm_fork_1"} : <i1>
    %12 = spec_v2_interpolator %11#0, %11#1 {handshake.bb = 2 : ui32, handshake.name = "rhs_interpolate"} : <i1>
    %13 = passer %4[%12] {handshake.bb = 2 : ui32, handshake.name = "rhs_passer"} : <i1>, <i1>
    %14 = ndsource {handshake.bb = 3 : ui32, handshake.name = "out_nds_B_out"} : <>
    %15:2 = lazy_fork [2] %14 {handshake.bb = 3 : ui32, handshake.name = "out_lf_B_out"} : <>
    %16 = buffer %15#0, bufferType = FIFO_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "out_buf_lhs_nds_B_out"} : <>
    %17 = buffer %15#1, bufferType = FIFO_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "out_buf_rhs_nds_B_out"} : <>
    %18 = blocker %10[%16] {handshake.bb = 3 : ui32, handshake.name = "out_lhs_bl_B_out"} : <i1>, <>
    %19 = blocker %13[%17] {handshake.bb = 3 : ui32, handshake.name = "out_rhs_bl_B_out"} : <i1>, <>
    %20 = ndwire %18 {handshake.bb = 3 : ui32, handshake.name = "lhs_out_ndw_B_out"} : <i1>
    %21 = ndwire %19 {handshake.bb = 3 : ui32, handshake.name = "rhs_out_ndw_B_out"} : <i1>
    %22 = buffer %20, bufferType = FIFO_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "lhs_out_buf_B_out"} : <i1>
    %23 = buffer %21, bufferType = FIFO_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "rhs_out_buf_B_out"} : <i1>
    %24 = cmpi eq, %22, %23 {handshake.bb = 3 : ui32, handshake.name = "out_eq_B_out"} : <i1>
    end {handshake.bb = 3 : ui32, handshake.name = "end"} %24 : <i1>
  }
}
