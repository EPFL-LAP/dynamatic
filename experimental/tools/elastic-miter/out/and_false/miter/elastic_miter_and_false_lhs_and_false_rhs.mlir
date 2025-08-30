module {
  handshake.func @elastic_miter_and_false_lhs_and_false_rhs(%arg0: !handshake.channel<i1>, ...) -> !handshake.channel<i1> attributes {argNames = ["ins"], resNames = ["EQ_out"]} {
    %0:2 = lazy_fork [2] %arg0 {handshake.bb = 0 : ui32, handshake.name = "in_fork_ins"} : <i1>
    %1 = buffer %0#0, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 0 : ui32, handshake.name = "lhs_in_buf_ins"} : <i1>
    %2 = buffer %0#1, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 0 : ui32, handshake.name = "rhs_in_buf_ins"} : <i1>
    %3 = ndsource {handshake.bb = 3 : ui32, handshake.name = "out_nds_out"} : <>
    %4:2 = lazy_fork [2] %3 {handshake.bb = 3 : ui32, handshake.name = "out_lf_out"} : <>
    %5 = buffer %4#0, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 3 : ui32, handshake.name = "out_buf_lhs_nds_out"} : <>
    %6 = buffer %4#1, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 3 : ui32, handshake.name = "out_buf_rhs_nds_out"} : <>
    %7 = blocker %12[%5] {handshake.bb = 3 : ui32, handshake.name = "lhs_out_bl_out"} : <i1>, <>
    %8 = blocker %14[%6] {handshake.bb = 3 : ui32, handshake.name = "rhs_out_bl_out"} : <i1>, <>
    %9 = cmpi eq, %7, %8 {handshake.bb = 3 : ui32, handshake.name = "out_eq_out"} : <i1>
    %10 = source {handshake.bb = 1 : ui32, handshake.name = "lhs_source"} : <>
    %11 = constant %10 {handshake.bb = 1 : ui32, handshake.name = "lhs_constant", value = false} : <>, <i1>
    %12 = andi %1, %11 {handshake.bb = 1 : ui32, handshake.name = "lhs_and"} : <i1>
    sink %2 {handshake.bb = 2 : ui32, handshake.name = "rhs_sink"} : <i1>
    %13 = source {handshake.bb = 2 : ui32, handshake.name = "rhs_source"} : <>
    %14 = constant %13 {handshake.bb = 2 : ui32, handshake.name = "rhs_constant", value = false} : <>, <i1>
    end {handshake.bb = 3 : ui32, handshake.name = "end"} %9 : <i1>
  }
}
