module {
  handshake.func @elastic_miter_repeating_init_lhs_repeating_init_rhs(%arg0: !handshake.channel<i1>, ...) -> !handshake.channel<i1> attributes {argNames = ["A_in"], resNames = ["EQ_B_out"]} {
    %0:2 = lazy_fork [2] %arg0 {handshake.bb = 0 : ui32, handshake.name = "in_fork_A_in"} : <i1>
    %1 = buffer %0#0, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 0 : ui32, handshake.name = "lhs_in_buf_A_in"} : <i1>
    %2 = buffer %0#1, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 0 : ui32, handshake.name = "rhs_in_buf_A_in"} : <i1>
    %3 = source {handshake.bb = 3 : ui32, handshake.name = "out_src_B_out"} : <>
    %4:2 = lazy_fork [2] %3 {handshake.bb = 3 : ui32, handshake.name = "out_lf_B_out"} : <>
    %5 = buffer %4#0, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 3 : ui32, handshake.name = "out_buf_lhs_nds_B_out"} : <>
    %6 = buffer %4#1, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 3 : ui32, handshake.name = "out_buf_rhs_nds_B_out"} : <>
    %7 = blocker %10[%5] {handshake.bb = 3 : ui32, handshake.name = "lhs_out_bl_B_out"} : <i1>, <>
    %8 = blocker %14#1[%6] {handshake.bb = 3 : ui32, handshake.name = "rhs_out_bl_B_out"} : <i1>, <>
    %9 = cmpi eq, %7, %8 {handshake.bb = 3 : ui32, handshake.name = "out_eq_B_out"} : <i1>
    %10 = spec_v2_repeating_init %1 {handshake.bb = 1 : ui32, handshake.name = "lhs_spec_v2_repeating_init", initToken = 1 : ui1} : <i1>
    %11 = source {handshake.bb = 2 : ui32, handshake.name = "rhs_source"} : <>
    %12 = constant %11 {handshake.bb = 2 : ui32, handshake.name = "rhs_constant", value = true} : <>, <i1>
    %13 = mux %16 [%12, %2] {handshake.bb = 2 : ui32, handshake.name = "rhs_mux"} : <i1>, [<i1>, <i1>] to <i1>
    %14:2 = fork [2] %13 {handshake.bb = 2 : ui32, handshake.name = "rhs_ri_fork"} : <i1>
    %15 = init %14#0 {handshake.bb = 2 : ui32, handshake.name = "rhs_newInit", initToken = 0 : ui1} : <i1>
    %16 = buffer %15, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "rhs_buff"} : <i1>
    end {handshake.bb = 3 : ui32, handshake.name = "end"} %9 : <i1>
  }
}
