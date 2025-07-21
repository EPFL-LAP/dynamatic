module {
  handshake.func @elastic_miter_interpolator_ind_lhs_interpolator_ind_rhs(%arg0: !handshake.channel<i1>, %arg1: !handshake.channel<i1>, %arg2: !handshake.channel<i1>, ...) -> !handshake.channel<i1> attributes {argNames = ["A_in", "B_in", "Sup_in"], resNames = ["EQ_Sup_out"]} {
    %0:2 = lazy_fork [2] %arg0 {handshake.bb = 0 : ui32, handshake.name = "in_fork_A_in"} : <i1>
    %1 = buffer %0#0, bufferType = FIFO_BREAK_DV, numSlots = 2 {handshake.bb = 0 : ui32, handshake.name = "lhs_in_buf_A_in"} : <i1>
    %2 = buffer %0#1, bufferType = FIFO_BREAK_DV, numSlots = 2 {handshake.bb = 0 : ui32, handshake.name = "rhs_in_buf_A_in"} : <i1>
    %3 = ndwire %1 {handshake.bb = 0 : ui32, handshake.name = "lhs_in_ndw_A_in"} : <i1>
    %4 = ndwire %2 {handshake.bb = 0 : ui32, handshake.name = "rhs_in_ndw_A_in"} : <i1>
    %5:2 = lazy_fork [2] %arg1 {handshake.bb = 0 : ui32, handshake.name = "in_fork_B_in"} : <i1>
    %6 = buffer %5#0, bufferType = FIFO_BREAK_DV, numSlots = 2 {handshake.bb = 0 : ui32, handshake.name = "lhs_in_buf_B_in"} : <i1>
    %7 = buffer %5#1, bufferType = FIFO_BREAK_DV, numSlots = 2 {handshake.bb = 0 : ui32, handshake.name = "rhs_in_buf_B_in"} : <i1>
    %8 = ndwire %6 {handshake.bb = 0 : ui32, handshake.name = "lhs_in_ndw_B_in"} : <i1>
    %9 = ndwire %7 {handshake.bb = 0 : ui32, handshake.name = "rhs_in_ndw_B_in"} : <i1>
    %10:2 = lazy_fork [2] %arg2 {handshake.bb = 0 : ui32, handshake.name = "in_fork_Sup_in"} : <i1>
    %11 = buffer %10#0, bufferType = FIFO_BREAK_DV, numSlots = 2 {handshake.bb = 0 : ui32, handshake.name = "lhs_in_buf_Sup_in"} : <i1>
    %12 = buffer %10#1, bufferType = FIFO_BREAK_DV, numSlots = 2 {handshake.bb = 0 : ui32, handshake.name = "rhs_in_buf_Sup_in"} : <i1>
    %13 = ndwire %11 {handshake.bb = 0 : ui32, handshake.name = "lhs_in_ndw_Sup_in"} : <i1>
    %14 = ndwire %12 {handshake.bb = 0 : ui32, handshake.name = "rhs_in_ndw_Sup_in"} : <i1>
    %15 = ndsource {handshake.bb = 5 : ui32, handshake.name = "out_nds_Sup_out"} : <>
    %16:2 = lazy_fork [2] %15 {handshake.bb = 5 : ui32, handshake.name = "out_lf_Sup_out"} : <>
    %17 = buffer %16#0, bufferType = FIFO_BREAK_DV, numSlots = 2 {handshake.bb = 5 : ui32, handshake.name = "out_buf_lhs_nds_Sup_out"} : <>
    %18 = buffer %16#1, bufferType = FIFO_BREAK_DV, numSlots = 2 {handshake.bb = 5 : ui32, handshake.name = "out_buf_rhs_nds_Sup_out"} : <>
    %19 = blocker %30[%17] {handshake.bb = 5 : ui32, handshake.name = "lhs_out_bl_Sup_out"} : <i1>, <>
    %20 = blocker %33[%18] {handshake.bb = 5 : ui32, handshake.name = "rhs_out_bl_Sup_out"} : <i1>, <>
    %21 = ndwire %19 {handshake.bb = 5 : ui32, handshake.name = "lhs_out_ndw_Sup_out"} : <i1>
    %22 = ndwire %20 {handshake.bb = 5 : ui32, handshake.name = "rhs_out_ndw_Sup_out"} : <i1>
    %23 = buffer %21, bufferType = FIFO_BREAK_DV, numSlots = 2 {handshake.bb = 5 : ui32, handshake.name = "lhs_out_buf_Sup_out"} : <i1>
    %24 = buffer %22, bufferType = FIFO_BREAK_DV, numSlots = 2 {handshake.bb = 5 : ui32, handshake.name = "rhs_out_buf_Sup_out"} : <i1>
    %25 = cmpi eq, %23, %24 {handshake.bb = 5 : ui32, handshake.name = "out_eq_Sup_out"} : <i1>
    %26:2 = fork [2] %8 {handshake.bb = 2 : ui32, handshake.name = "lhs_fork"} : <i1>
    %27 = spec_v2_interpolator %3, %26#0 {handshake.bb = 2 : ui32, handshake.name = "lhs_interpolate"} : <i1>
    %28 = spec_v2_repeating_init %26#1 {handshake.bb = 2 : ui32, handshake.name = "lhs_ri", initToken = 1 : ui1} : <i1>
    %29 = passer %13[%28] {handshake.bb = 2 : ui32, handshake.name = "lhs_passer1"} : <i1>, <i1>
    %30 = passer %29[%27] {handshake.bb = 2 : ui32, handshake.name = "lhs_passer2"} : <i1>, <i1>
    %31 = spec_v2_repeating_init %9 {handshake.bb = 4 : ui32, handshake.name = "rhs_ri", initToken = 1 : ui1} : <i1>
    %32 = spec_v2_interpolator %4, %31 {handshake.bb = 4 : ui32, handshake.name = "rhs_interpolate"} : <i1>
    %33 = passer %14[%32] {handshake.bb = 4 : ui32, handshake.name = "rhs_passer"} : <i1>, <i1>
    end {handshake.bb = 5 : ui32, handshake.name = "end"} %25 : <i1>
  }
}
