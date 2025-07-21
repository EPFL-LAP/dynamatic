module {
  handshake.func @elastic_miter_ri_fork_lhs_ri_fork_rhs(%arg0: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>, !handshake.channel<i1>) attributes {argNames = ["A_in"], resNames = ["EQ_B_out", "EQ_C_out"]} {
    %0:2 = lazy_fork [2] %arg0 {handshake.bb = 0 : ui32, handshake.name = "in_fork_A_in"} : <i1>
    %1 = buffer %0#0, bufferType = FIFO_BREAK_DV, numSlots = 1 {handshake.bb = 0 : ui32, handshake.name = "lhs_in_buf_A_in"} : <i1>
    %2 = buffer %0#1, bufferType = FIFO_BREAK_DV, numSlots = 1 {handshake.bb = 0 : ui32, handshake.name = "rhs_in_buf_A_in"} : <i1>
    %3 = ndsource {handshake.bb = 3 : ui32, handshake.name = "out_nds_B_out"} : <>
    %4:2 = lazy_fork [2] %3 {handshake.bb = 3 : ui32, handshake.name = "out_lf_B_out"} : <>
    %5 = buffer %4#0, bufferType = FIFO_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "out_buf_lhs_nds_B_out"} : <>
    %6 = buffer %4#1, bufferType = FIFO_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "out_buf_rhs_nds_B_out"} : <>
    %7 = blocker %19#0[%5] {handshake.bb = 3 : ui32, handshake.name = "lhs_out_bl_B_out"} : <i1>, <>
    %8 = blocker %22[%6] {handshake.bb = 3 : ui32, handshake.name = "rhs_out_bl_B_out"} : <i1>, <>
    %9 = cmpi eq, %7, %8 {handshake.bb = 3 : ui32, handshake.name = "out_eq_B_out"} : <i1>
    %10 = ndsource {handshake.bb = 3 : ui32, handshake.name = "out_nds_C_out"} : <>
    %11:2 = lazy_fork [2] %10 {handshake.bb = 3 : ui32, handshake.name = "out_lf_C_out"} : <>
    %12 = buffer %11#0, bufferType = FIFO_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "out_buf_lhs_nds_C_out"} : <>
    %13 = buffer %11#1, bufferType = FIFO_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "out_buf_rhs_nds_C_out"} : <>
    %14 = blocker %19#1[%12] {handshake.bb = 3 : ui32, handshake.name = "lhs_out_bl_C_out"} : <i1>, <>
    %15 = blocker %24[%13] {handshake.bb = 3 : ui32, handshake.name = "rhs_out_bl_C_out"} : <i1>, <>
    %16 = cmpi eq, %14, %15 {handshake.bb = 3 : ui32, handshake.name = "out_eq_C_out"} : <i1>
    %17 = spec_v2_repeating_init %1 {handshake.bb = 1 : ui32, handshake.name = "lhs_ri", initToken = 1 : ui1} : <i1>
    %18 = buffer %17, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "lhs_buffer"} : <i1>
    %19:2 = fork [2] %18 {handshake.bb = 1 : ui32, handshake.name = "lhs_fork"} : <i1>
    %20:2 = fork [2] %2 {handshake.bb = 2 : ui32, handshake.name = "rhs_fork"} : <i1>
    %21 = spec_v2_repeating_init %20#0 {handshake.bb = 2 : ui32, handshake.name = "rhs_ri1", initToken = 1 : ui1} : <i1>
    %22 = buffer %21, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "rhs_buffer1"} : <i1>
    %23 = spec_v2_repeating_init %20#1 {handshake.bb = 2 : ui32, handshake.name = "rhs_ri2", initToken = 1 : ui1} : <i1>
    %24 = buffer %23, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "rhs_buffer2"} : <i1>
    end {handshake.bb = 3 : ui32, handshake.name = "end"} %9, %16 : <i1>, <i1>
  }
}
