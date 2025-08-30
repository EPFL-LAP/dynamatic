module {
  handshake.func @elastic_miter_mux_add_suppress_lhs_mux_add_suppress_rhs(%arg0: !handshake.channel<i1>, %arg1: !handshake.channel<i1>, ...) -> !handshake.channel<i1> attributes {argNames = ["ins", "sel"], resNames = ["EQ_out"]} {
    %0:2 = lazy_fork [2] %arg0 {handshake.bb = 0 : ui32, handshake.name = "in_fork_ins"} : <i1>
    %1 = buffer %0#0, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 0 : ui32, handshake.name = "lhs_in_buf_ins"} : <i1>
    %2 = buffer %0#1, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 0 : ui32, handshake.name = "rhs_in_buf_ins"} : <i1>
    %3:2 = lazy_fork [2] %arg1 {handshake.bb = 0 : ui32, handshake.name = "in_fork_sel"} : <i1>
    %4 = buffer %3#0, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 0 : ui32, handshake.name = "lhs_in_buf_sel"} : <i1>
    %5 = buffer %3#1, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 0 : ui32, handshake.name = "rhs_in_buf_sel"} : <i1>
    %6 = ndsource {handshake.bb = 3 : ui32, handshake.name = "out_nds_out"} : <>
    %7:2 = lazy_fork [2] %6 {handshake.bb = 3 : ui32, handshake.name = "out_lf_out"} : <>
    %8 = buffer %7#0, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 3 : ui32, handshake.name = "out_buf_lhs_nds_out"} : <>
    %9 = buffer %7#1, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 3 : ui32, handshake.name = "out_buf_rhs_nds_out"} : <>
    %10 = blocker %15[%8] {handshake.bb = 3 : ui32, handshake.name = "lhs_out_bl_out"} : <i1>, <>
    %11 = blocker %20[%9] {handshake.bb = 3 : ui32, handshake.name = "rhs_out_bl_out"} : <i1>, <>
    %12 = cmpi eq, %10, %11 {handshake.bb = 3 : ui32, handshake.name = "out_eq_out"} : <i1>
    %13 = source {handshake.bb = 1 : ui32, handshake.name = "lhs_source"} : <>
    %14 = constant %13 {handshake.bb = 1 : ui32, handshake.name = "lhs_constant", value = true} : <>, <i1>
    %15 = mux %4 [%1, %14] {handshake.bb = 1 : ui32, handshake.name = "lhs_mux"} : <i1>, [<i1>, <i1>] to <i1>
    %16:2 = fork [2] %5 {handshake.bb = 2 : ui32, handshake.name = "rhs_vm_fork_1"} : <i1>
    %17 = source {handshake.bb = 2 : ui32, handshake.name = "rhs_source"} : <>
    %18 = constant %17 {handshake.bb = 2 : ui32, handshake.name = "rhs_constant", value = true} : <>, <i1>
    %19 = passer %18[%16#0] {handshake.bb = 2 : ui32, handshake.name = "rhs_passer"} : <i1>, <i1>
    %20 = mux %16#1 [%2, %19] {handshake.bb = 2 : ui32, handshake.name = "rhs_mux"} : <i1>, [<i1>, <i1>] to <i1>
    end {handshake.bb = 3 : ui32, handshake.name = "end"} %12 : <i1>
  }
}
