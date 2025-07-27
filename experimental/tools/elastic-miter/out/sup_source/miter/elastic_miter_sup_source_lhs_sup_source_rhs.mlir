module {
  handshake.func @elastic_miter_sup_source_lhs_sup_source_rhs(%arg0: !handshake.channel<i1>, ...) -> !handshake.control<> attributes {argNames = ["ctrl"], resNames = ["EQ_result"]} {
    %0:2 = lazy_fork [2] %arg0 {handshake.bb = 0 : ui32, handshake.name = "in_fork_ctrl"} : <i1>
    %1 = buffer %0#1, bufferType = FIFO_BREAK_DV, numSlots = 1 {handshake.bb = 0 : ui32, handshake.name = "rhs_in_buf_ctrl"} : <i1>
    %2 = ndsource {handshake.bb = 3 : ui32, handshake.name = "out_nds_result"} : <>
    %3:2 = lazy_fork [2] %2 {handshake.bb = 3 : ui32, handshake.name = "out_lf_result"} : <>
    %4 = buffer %3#0, bufferType = FIFO_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "out_buf_lhs_nds_result"} : <>
    %5 = buffer %3#1, bufferType = FIFO_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "out_buf_rhs_nds_result"} : <>
    %6 = blocker %9[%4] {handshake.bb = 3 : ui32, handshake.name = "lhs_out_bl_result"} : <>, <>
    %7 = blocker %11[%5] {handshake.bb = 3 : ui32, handshake.name = "rhs_out_bl_result"} : <>, <>
    %8 = join %6, %7 {handshake.bb = 3 : ui32, handshake.name = "out_eq_result"} : <>
    sink %0#0 {handshake.bb = 1 : ui32, handshake.name = "lhs_sink"} : <i1>
    %9 = source {handshake.bb = 1 : ui32, handshake.name = "lhs_source"} : <>
    %10 = source {handshake.bb = 2 : ui32, handshake.name = "rhs_source"} : <>
    %11 = passer %10[%1] {handshake.bb = 2 : ui32, handshake.name = "rhs_passer"} : <>, <i1>
    end {handshake.bb = 3 : ui32, handshake.name = "end"} %8 : <>
  }
}
