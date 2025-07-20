module {
  handshake.func @elastic_miter_ri_fork_lhs_ri_fork_rhs(%arg0: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>, !handshake.channel<i1>) attributes {argNames = ["A_in"], resNames = ["EQ_B_out", "EQ_C_out"]} {
    %0:2 = lazy_fork [2] %arg0 {handshake.bb = 0 : ui32, handshake.name = "in_fork_A_in"} : <i1>
    %1 = buffer %0#0, bufferType = FIFO_BREAK_DV, numSlots = 1 {handshake.bb = 0 : ui32, handshake.name = "lhs_in_buf_A_in"} : <i1>
    %2 = buffer %0#1, bufferType = FIFO_BREAK_DV, numSlots = 1 {handshake.bb = 0 : ui32, handshake.name = "rhs_in_buf_A_in"} : <i1>
    %3 = ndwire %1 {handshake.bb = 0 : ui32, handshake.name = "lhs_in_ndw_A_in"} : <i1>
    %4 = ndwire %2 {handshake.bb = 0 : ui32, handshake.name = "rhs_in_ndw_A_in"} : <i1>
    %5 = spec_v2_repeating_init %3 {handshake.bb = 1 : ui32, handshake.name = "lhs_ri", initToken = 1 : ui1} : <i1>
    %6:2 = fork [2] %5 {handshake.bb = 1 : ui32, handshake.name = "lhs_fork"} : <i1>
    %7:2 = fork [2] %4 {handshake.bb = 2 : ui32, handshake.name = "rhs_fork"} : <i1>
    %8 = spec_v2_repeating_init %7#0 {handshake.bb = 2 : ui32, handshake.name = "rhs_ri1", initToken = 1 : ui1} : <i1>
    %9 = spec_v2_repeating_init %7#1 {handshake.bb = 2 : ui32, handshake.name = "rhs_ri2", initToken = 1 : ui1} : <i1>
    %10 = ndsource {handshake.bb = 3 : ui32, handshake.name = "out_nds_B_out"} : <>
    %11:2 = lazy_fork [2] %10 {handshake.bb = 3 : ui32, handshake.name = "out_lf_B_out"} : <>
    %12 = buffer %11#0, bufferType = FIFO_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "out_buf_lhs_nds_B_out"} : <>
    %13 = buffer %11#1, bufferType = FIFO_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "out_buf_rhs_nds_B_out"} : <>
    %14 = blocker %6#0[%12] {handshake.bb = 3 : ui32, handshake.name = "out_lhs_bl_B_out"} : <i1>, <>
    %15 = blocker %8[%13] {handshake.bb = 3 : ui32, handshake.name = "out_rhs_bl_B_out"} : <i1>, <>
    %16 = ndwire %14 {handshake.bb = 3 : ui32, handshake.name = "lhs_out_ndw_B_out"} : <i1>
    %17 = ndwire %15 {handshake.bb = 3 : ui32, handshake.name = "rhs_out_ndw_B_out"} : <i1>
    %18 = buffer %16, bufferType = FIFO_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "lhs_out_buf_B_out"} : <i1>
    %19 = buffer %17, bufferType = FIFO_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "rhs_out_buf_B_out"} : <i1>
    %20 = cmpi eq, %18, %19 {handshake.bb = 3 : ui32, handshake.name = "out_eq_B_out"} : <i1>
    %21 = ndsource {handshake.bb = 3 : ui32, handshake.name = "out_nds_C_out"} : <>
    %22:2 = lazy_fork [2] %21 {handshake.bb = 3 : ui32, handshake.name = "out_lf_C_out"} : <>
    %23 = buffer %22#0, bufferType = FIFO_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "out_buf_lhs_nds_C_out"} : <>
    %24 = buffer %22#1, bufferType = FIFO_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "out_buf_rhs_nds_C_out"} : <>
    %25 = blocker %6#1[%23] {handshake.bb = 3 : ui32, handshake.name = "out_lhs_bl_C_out"} : <i1>, <>
    %26 = blocker %9[%24] {handshake.bb = 3 : ui32, handshake.name = "out_rhs_bl_C_out"} : <i1>, <>
    %27 = ndwire %25 {handshake.bb = 3 : ui32, handshake.name = "lhs_out_ndw_C_out"} : <i1>
    %28 = ndwire %26 {handshake.bb = 3 : ui32, handshake.name = "rhs_out_ndw_C_out"} : <i1>
    %29 = buffer %27, bufferType = FIFO_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "lhs_out_buf_C_out"} : <i1>
    %30 = buffer %28, bufferType = FIFO_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "rhs_out_buf_C_out"} : <i1>
    %31 = cmpi eq, %29, %30 {handshake.bb = 3 : ui32, handshake.name = "out_eq_C_out"} : <i1>
    end {handshake.bb = 3 : ui32, handshake.name = "end"} %20, %31 : <i1>, <i1>
  }
}
