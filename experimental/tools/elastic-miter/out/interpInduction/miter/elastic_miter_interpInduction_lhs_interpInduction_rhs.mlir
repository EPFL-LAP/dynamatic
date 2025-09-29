module {
  handshake.func @elastic_miter_interpInduction_lhs_interpInduction_rhs(%arg0: !handshake.channel<i1>, %arg1: !handshake.channel<i1>, ...) -> !handshake.channel<i1> attributes {argNames = ["short_in", "long_in"], resNames = ["EQ_B_out"]} {
    %0:2 = fork [2] %arg0 {handshake.bb = 0 : ui32, handshake.name = "ctx_short_fork"} : <i1>
    %1 = spec_v2_nd_speculator %0#1 {handshake.bb = 0 : ui32, handshake.name = "ctx_n_repeating_init"} : <i1>
    sink %arg1 {handshake.bb = 0 : ui32, handshake.name = "ctx_sink"} : <i1>
    %2:2 = lazy_fork [2] %0#0 {handshake.bb = 0 : ui32, handshake.name = "in_fork_short_in"} : <i1>
    %3 = buffer %2#0, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 0 : ui32, handshake.name = "lhs_in_buf_short_in"} : <i1>
    %4 = buffer %2#1, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 0 : ui32, handshake.name = "rhs_in_buf_short_in"} : <i1>
    %5:2 = lazy_fork [2] %1 {handshake.bb = 0 : ui32, handshake.name = "in_fork_long_in"} : <i1>
    %6 = buffer %5#0, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 0 : ui32, handshake.name = "lhs_in_buf_long_in"} : <i1>
    %7 = buffer %5#1, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 0 : ui32, handshake.name = "rhs_in_buf_long_in"} : <i1>
    %8 = source {handshake.bb = 3 : ui32, handshake.name = "out_src_B_out"} : <>
    %9:2 = lazy_fork [2] %8 {handshake.bb = 3 : ui32, handshake.name = "out_lf_B_out"} : <>
    %10 = buffer %9#0, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 3 : ui32, handshake.name = "out_buf_lhs_nds_B_out"} : <>
    %11 = buffer %9#1, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 3 : ui32, handshake.name = "out_buf_rhs_nds_B_out"} : <>
    %12 = blocker %21[%10] {handshake.bb = 3 : ui32, handshake.name = "lhs_out_bl_B_out"} : <i1>, <>
    %13 = blocker %24[%11] {handshake.bb = 3 : ui32, handshake.name = "rhs_out_bl_B_out"} : <i1>, <>
    %14 = cmpi eq, %12, %13 {handshake.bb = 3 : ui32, handshake.name = "out_eq_B_out"} : <i1>
    %15:2 = fork [2] %6 {handshake.bb = 1 : ui32, handshake.name = "lhs_long_fork"} : <i1>
    %16 = spec_v2_interpolator %3, %15#0 {handshake.bb = 1 : ui32, handshake.name = "lhs_interpolate"} : <i1>
    %17 = spec_v2_repeating_init %15#1 {handshake.bb = 1 : ui32, handshake.name = "lhs_spec_v2_repeating_init", initToken = 1 : ui1} : <i1>
    %18 = buffer %17, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "lhs_buffer"} : <i1>
    %19 = source {handshake.bb = 1 : ui32, handshake.name = "lhs_source"} : <>
    %20 = constant %19 {handshake.bb = 1 : ui32, handshake.name = "lhs_constant", value = false} : <>, <i1>
    %21 = mux %18 [%20, %16] {handshake.bb = 1 : ui32, handshake.name = "lhs_mux"} : <i1>, [<i1>, <i1>] to <i1>
    %22 = spec_v2_repeating_init %7 {handshake.bb = 2 : ui32, handshake.name = "rhs_spec_v2_repeating_init", initToken = 1 : ui1} : <i1>
    %23 = buffer %22, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "rhs_buffer"} : <i1>
    %24 = spec_v2_interpolator %4, %23 {handshake.bb = 2 : ui32, handshake.name = "rhs_interpolate"} : <i1>
    end {handshake.bb = 3 : ui32, handshake.name = "end"} %14 : <i1>
  }
}
