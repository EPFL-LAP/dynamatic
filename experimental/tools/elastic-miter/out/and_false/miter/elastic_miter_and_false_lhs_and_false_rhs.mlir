module {
  handshake.func @elastic_miter_and_false_lhs_and_false_rhs(%arg0: !handshake.channel<i1>, ...) -> !handshake.channel<i1> attributes {argNames = ["ins"], resNames = ["EQ_out"]} {
    %0:2 = lazy_fork [2] %arg0 {handshake.bb = 0 : ui32, handshake.name = "in_fork_ins"} : <i1>
    %1 = buffer %0#0, bufferType = FIFO_BREAK_DV, numSlots = 1 {handshake.bb = 0 : ui32, handshake.name = "lhs_in_buf_ins"} : <i1>
    %2 = ndsource {handshake.bb = 3 : ui32, handshake.name = "out_nds_out"} : <>
    %3:2 = lazy_fork [2] %2 {handshake.bb = 3 : ui32, handshake.name = "out_lf_out"} : <>
    %4 = buffer %3#0, bufferType = FIFO_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "out_buf_lhs_nds_out"} : <>
    %5 = buffer %3#1, bufferType = FIFO_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "out_buf_rhs_nds_out"} : <>
    %6 = blocker %11[%4] {handshake.bb = 3 : ui32, handshake.name = "lhs_out_bl_out"} : <i1>, <>
    %7 = blocker %13[%5] {handshake.bb = 3 : ui32, handshake.name = "rhs_out_bl_out"} : <i1>, <>
    %8 = cmpi eq, %6, %7 {handshake.bb = 3 : ui32, handshake.name = "out_eq_out"} : <i1>
    %9 = source {handshake.bb = 1 : ui32, handshake.name = "lhs_source"} : <>
    %10 = constant %9 {handshake.bb = 1 : ui32, handshake.name = "lhs_constant", value = false} : <>, <i1>
    %11 = andi %1, %10 {handshake.bb = 1 : ui32, handshake.name = "lhs_and"} : <i1>
    sink %0#1 {handshake.bb = 2 : ui32, handshake.name = "rhs_sink"} : <i1>
    %12 = source {handshake.bb = 2 : ui32, handshake.name = "rhs_source"} : <>
    %13 = constant %12 {handshake.bb = 2 : ui32, handshake.name = "rhs_constant", value = false} : <>, <i1>
    end {handshake.bb = 3 : ui32, handshake.name = "end"} %8 : <i1>
  }
}
