module {
  handshake.func @elastic_miter_muxForkSwap_lhs_muxForkSwap_rhs(%arg0: !handshake.channel<i1>, %arg1: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>, !handshake.channel<i1>) attributes {argNames = ["sel_in", "data_in"], resNames = ["EQ_out1", "EQ_out2"]} {
    %0:2 = lazy_fork [2] %arg0 {handshake.bb = 0 : ui32, handshake.name = "in_fork_sel_in"} : <i1>
    %1 = buffer %0#0, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 0 : ui32, handshake.name = "lhs_in_buf_sel_in"} : <i1>
    %2 = buffer %0#1, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 0 : ui32, handshake.name = "rhs_in_buf_sel_in"} : <i1>
    %3:2 = lazy_fork [2] %arg1 {handshake.bb = 0 : ui32, handshake.name = "in_fork_data_in"} : <i1>
    %4 = buffer %3#0, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 0 : ui32, handshake.name = "lhs_in_buf_data_in"} : <i1>
    %5 = buffer %3#1, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 0 : ui32, handshake.name = "rhs_in_buf_data_in"} : <i1>
    %6 = ndsource {handshake.bb = 3 : ui32, handshake.name = "out_nds_out1"} : <>
    %7:2 = lazy_fork [2] %6 {handshake.bb = 3 : ui32, handshake.name = "out_lf_out1"} : <>
    %8 = buffer %7#0, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 3 : ui32, handshake.name = "out_buf_lhs_nds_out1"} : <>
    %9 = buffer %7#1, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 3 : ui32, handshake.name = "out_buf_rhs_nds_out1"} : <>
    %10 = blocker %24[%8] {handshake.bb = 3 : ui32, handshake.name = "lhs_out_bl_out1"} : <i1>, <>
    %11 = blocker %31#0[%9] {handshake.bb = 3 : ui32, handshake.name = "rhs_out_bl_out1"} : <i1>, <>
    %12 = cmpi eq, %10, %11 {handshake.bb = 3 : ui32, handshake.name = "out_eq_out1"} : <i1>
    %13 = ndsource {handshake.bb = 3 : ui32, handshake.name = "out_nds_out2"} : <>
    %14:2 = lazy_fork [2] %13 {handshake.bb = 3 : ui32, handshake.name = "out_lf_out2"} : <>
    %15 = buffer %14#0, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 3 : ui32, handshake.name = "out_buf_lhs_nds_out2"} : <>
    %16 = buffer %14#1, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 3 : ui32, handshake.name = "out_buf_rhs_nds_out2"} : <>
    %17 = blocker %27[%15] {handshake.bb = 3 : ui32, handshake.name = "lhs_out_bl_out2"} : <i1>, <>
    %18 = blocker %31#1[%16] {handshake.bb = 3 : ui32, handshake.name = "rhs_out_bl_out2"} : <i1>, <>
    %19 = cmpi eq, %17, %18 {handshake.bb = 3 : ui32, handshake.name = "out_eq_out2"} : <i1>
    %20:2 = fork [2] %1 {handshake.bb = 1 : ui32, handshake.name = "lhs_sel_fork"} : <i1>
    %21:2 = fork [2] %4 {handshake.bb = 1 : ui32, handshake.name = "lhs_data_fork"} : <i1>
    %22 = source {handshake.bb = 1 : ui32, handshake.name = "lhs_source"} : <>
    %23 = constant %22 {handshake.bb = 1 : ui32, handshake.name = "lhs_constant", value = false} : <>, <i1>
    %24 = mux %20#0 [%23, %21#0] {handshake.bb = 1 : ui32, handshake.name = "lhs_mux0"} : <i1>, [<i1>, <i1>] to <i1>
    %25 = source {handshake.bb = 1 : ui32, handshake.name = "lhs_source2"} : <>
    %26 = constant %25 {handshake.bb = 1 : ui32, handshake.name = "lhs_constant2", value = false} : <>, <i1>
    %27 = mux %20#1 [%26, %21#1] {handshake.bb = 1 : ui32, handshake.name = "lhs_mux1"} : <i1>, [<i1>, <i1>] to <i1>
    %28 = source {handshake.bb = 2 : ui32, handshake.name = "rhs_source"} : <>
    %29 = constant %28 {handshake.bb = 2 : ui32, handshake.name = "rhs_constant", value = false} : <>, <i1>
    %30 = mux %2 [%29, %5] {handshake.bb = 2 : ui32, handshake.name = "rhs_mux0"} : <i1>, [<i1>, <i1>] to <i1>
    %31:2 = fork [2] %30 {handshake.bb = 2 : ui32, handshake.name = "rhs_fork"} : <i1>
    end {handshake.bb = 3 : ui32, handshake.name = "end"} %12, %19 : <i1>, <i1>
  }
}
