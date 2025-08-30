module {
  handshake.func @elastic_miter_suppressorInduction_lhs_suppressorInduction_rhs(%arg0: !handshake.channel<i1>, %arg1: !handshake.channel<i1>, ...) -> !handshake.channel<i1> attributes {argNames = ["short", "A_in"], resNames = ["EQ_B_out"]} {
    %0:2 = fork [2] %arg0 {handshake.bb = 1 : ui32, handshake.name = "fork_short"} : <i1>
    %1 = spec_v2_nd_speculator %0#0 {handshake.bb = 1 : ui32, handshake.name = "nd_spec"} : <i1>
    %2:2 = lazy_fork [2] %0#1 {handshake.bb = 1 : ui32, handshake.name = "in_fork_short"} : <i1>
    %3 = buffer %2#0, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 1 : ui32, handshake.name = "lhs_in_buf_short"} : <i1>
    %4 = buffer %2#1, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 1 : ui32, handshake.name = "rhs_in_buf_short"} : <i1>
    %5:2 = lazy_fork [2] %1 {handshake.bb = 1 : ui32, handshake.name = "in_fork_oldLong"} : <i1>
    %6 = buffer %5#0, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 1 : ui32, handshake.name = "lhs_in_buf_oldLong"} : <i1>
    %7 = buffer %5#1, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 1 : ui32, handshake.name = "rhs_in_buf_oldLong"} : <i1>
    %8:2 = lazy_fork [2] %arg1 {handshake.bb = 1 : ui32, handshake.name = "in_fork_A_in"} : <i1>
    %9 = buffer %8#0, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 1 : ui32, handshake.name = "lhs_in_buf_A_in"} : <i1>
    %10 = buffer %8#1, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 1 : ui32, handshake.name = "rhs_in_buf_A_in"} : <i1>
    %11 = ndsource {handshake.bb = 4 : ui32, handshake.name = "out_nds_B_out"} : <>
    %12:2 = lazy_fork [2] %11 {handshake.bb = 4 : ui32, handshake.name = "out_lf_B_out"} : <>
    %13 = buffer %12#0, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 4 : ui32, handshake.name = "out_buf_lhs_nds_B_out"} : <>
    %14 = buffer %12#1, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 4 : ui32, handshake.name = "out_buf_rhs_nds_B_out"} : <>
    %15 = blocker %23[%13] {handshake.bb = 4 : ui32, handshake.name = "lhs_out_bl_B_out"} : <i1>, <>
    %16 = blocker %27[%14] {handshake.bb = 4 : ui32, handshake.name = "rhs_out_bl_B_out"} : <i1>, <>
    %17 = cmpi eq, %15, %16 {handshake.bb = 4 : ui32, handshake.name = "out_eq_B_out"} : <i1>
    %18:2 = fork [2] %6 {handshake.bb = 2 : ui32, handshake.name = "lhs_vm_fork_1"} : <i1>
    %19 = spec_v2_interpolator %3, %18#0 {handshake.bb = 2 : ui32, handshake.name = "lhs_interpolate"} : <i1>
    %20 = spec_v2_repeating_init %18#1 {handshake.bb = 2 : ui32, handshake.name = "lhs_ri", initToken = 1 : ui1} : <i1>
    %21 = buffer %20, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "lhs_buffer"} : <i1>
    %22 = passer %9[%21] {handshake.bb = 2 : ui32, handshake.name = "lhs_passer1"} : <i1>, <i1>
    %23 = passer %22[%19] {handshake.bb = 2 : ui32, handshake.name = "lhs_passer2"} : <i1>, <i1>
    %24 = spec_v2_repeating_init %7 {handshake.bb = 3 : ui32, handshake.name = "rhs_ri", initToken = 1 : ui1} : <i1>
    %25 = buffer %24, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 3 : ui32, handshake.name = "rhs_buffer"} : <i1>
    %26 = spec_v2_interpolator %4, %25 {handshake.bb = 3 : ui32, handshake.name = "rhs_interpolate"} : <i1>
    %27 = passer %10[%26] {handshake.bb = 3 : ui32, handshake.name = "rhs_passer"} : <i1>, <i1>
    end {handshake.bb = 4 : ui32, handshake.name = "end"} %17 : <i1>
  }
}
