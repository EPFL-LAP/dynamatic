module {
  handshake.func @elastic_miter_switchSuppressCtrl_lhs_switchSuppressCtrl_rhs(%arg0: !handshake.channel<i1>, %arg1: !handshake.channel<i1>, ...) -> !handshake.channel<i1> attributes {argNames = ["val", "A_in"], resNames = ["EQ_B_out"]} {
    %0:3 = fork [3] %arg0 {handshake.bb = 1 : ui32, handshake.name = "fork"} : <i1>
    %1 = spec_v2_interpolator %0#0, %0#1 {handshake.bb = 1 : ui32, handshake.name = "interpolator"} : <i1>
    %2:2 = lazy_fork [2] %0#2 {handshake.bb = 1 : ui32, handshake.name = "in_fork_val"} : <i1>
    %3 = buffer %2#0, bufferType = FIFO_BREAK_DV, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "lhs_in_buf_val"} : <i1>
    %4 = ndwire %3 {handshake.bb = 1 : ui32, handshake.name = "lhs_in_ndw_val"} : <i1>
    %5 = ndwire %2#1 {handshake.bb = 1 : ui32, handshake.name = "rhs_in_ndw_val"} : <i1>
    %6:2 = lazy_fork [2] %arg1 {handshake.bb = 1 : ui32, handshake.name = "in_fork_A_in"} : <i1>
    %7 = buffer %6#0, bufferType = FIFO_BREAK_DV, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "lhs_in_buf_A_in"} : <i1>
    %8 = buffer %6#1, bufferType = FIFO_BREAK_DV, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "rhs_in_buf_A_in"} : <i1>
    %9 = ndwire %7 {handshake.bb = 1 : ui32, handshake.name = "lhs_in_ndw_A_in"} : <i1>
    %10 = ndwire %8 {handshake.bb = 1 : ui32, handshake.name = "rhs_in_ndw_A_in"} : <i1>
    %11:2 = lazy_fork [2] %1 {handshake.bb = 1 : ui32, handshake.name = "in_fork_val2"} : <i1>
    %12 = buffer %11#1, bufferType = FIFO_BREAK_DV, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "rhs_in_buf_val2"} : <i1>
    %13 = ndwire %11#0 {handshake.bb = 1 : ui32, handshake.name = "lhs_in_ndw_val2"} : <i1>
    %14 = ndwire %12 {handshake.bb = 1 : ui32, handshake.name = "rhs_in_ndw_val2"} : <i1>
    sink %13 {handshake.bb = 2 : ui32, handshake.name = "lhs_vm_sink_2"} : <i1>
    %15 = passer %9[%4] {handshake.bb = 2 : ui32, handshake.name = "lhs_passer"} : <i1>, <i1>
    sink %5 {handshake.bb = 3 : ui32, handshake.name = "rhs_vm_sink_0"} : <i1>
    %16 = passer %10[%14] {handshake.bb = 3 : ui32, handshake.name = "rhs_passer"} : <i1>, <i1>
    %17 = ndsource {handshake.bb = 4 : ui32, handshake.name = "out_nds_B_out"} : <>
    %18:2 = lazy_fork [2] %17 {handshake.bb = 4 : ui32, handshake.name = "out_lf_B_out"} : <>
    %19 = buffer %18#0, bufferType = FIFO_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "out_buf_lhs_nds_B_out"} : <>
    %20 = buffer %18#1, bufferType = FIFO_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "out_buf_rhs_nds_B_out"} : <>
    %21 = blocker %15[%19] {handshake.bb = 4 : ui32, handshake.name = "out_lhs_bl_B_out"} : <i1>, <>
    %22 = blocker %16[%20] {handshake.bb = 4 : ui32, handshake.name = "out_rhs_bl_B_out"} : <i1>, <>
    %23 = ndwire %21 {handshake.bb = 4 : ui32, handshake.name = "lhs_out_ndw_B_out"} : <i1>
    %24 = ndwire %22 {handshake.bb = 4 : ui32, handshake.name = "rhs_out_ndw_B_out"} : <i1>
    %25 = buffer %23, bufferType = FIFO_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "lhs_out_buf_B_out"} : <i1>
    %26 = buffer %24, bufferType = FIFO_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "rhs_out_buf_B_out"} : <i1>
    %27 = cmpi eq, %25, %26 {handshake.bb = 4 : ui32, handshake.name = "out_eq_B_out"} : <i1>
    end {handshake.bb = 4 : ui32, handshake.name = "end"} %27 : <i1>
  }
}
