module {
  handshake.func @elastic_miter_andForkSwap_lhs_andForkSwap_rhs(%arg0: !handshake.channel<i1>, %arg1: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>, !handshake.channel<i1>) attributes {argNames = ["in1", "in2"], resNames = ["EQ_out1", "EQ_out2"]} {
    %0:2 = lazy_fork [2] %arg0 {handshake.bb = 0 : ui32, handshake.name = "in_fork_in1"} : <i1>
    %1 = buffer %0#0, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 0 : ui32, handshake.name = "lhs_in_buf_in1"} : <i1>
    %2 = buffer %0#1, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 0 : ui32, handshake.name = "rhs_in_buf_in1"} : <i1>
    %3:2 = lazy_fork [2] %arg1 {handshake.bb = 0 : ui32, handshake.name = "in_fork_in2"} : <i1>
    %4 = buffer %3#0, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 0 : ui32, handshake.name = "lhs_in_buf_in2"} : <i1>
    %5 = buffer %3#1, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 0 : ui32, handshake.name = "rhs_in_buf_in2"} : <i1>
    %6 = ndsource {handshake.bb = 3 : ui32, handshake.name = "out_nds_out1"} : <>
    %7:2 = lazy_fork [2] %6 {handshake.bb = 3 : ui32, handshake.name = "out_lf_out1"} : <>
    %8 = buffer %7#0, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 3 : ui32, handshake.name = "out_buf_lhs_nds_out1"} : <>
    %9 = buffer %7#1, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 3 : ui32, handshake.name = "out_buf_rhs_nds_out1"} : <>
    %10 = blocker %21#0[%8] {handshake.bb = 3 : ui32, handshake.name = "lhs_out_bl_out1"} : <i1>, <>
    %11 = blocker %24[%9] {handshake.bb = 3 : ui32, handshake.name = "rhs_out_bl_out1"} : <i1>, <>
    %12 = cmpi eq, %10, %11 {handshake.bb = 3 : ui32, handshake.name = "out_eq_out1"} : <i1>
    %13 = ndsource {handshake.bb = 3 : ui32, handshake.name = "out_nds_out2"} : <>
    %14:2 = lazy_fork [2] %13 {handshake.bb = 3 : ui32, handshake.name = "out_lf_out2"} : <>
    %15 = buffer %14#0, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 3 : ui32, handshake.name = "out_buf_lhs_nds_out2"} : <>
    %16 = buffer %14#1, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 3 : ui32, handshake.name = "out_buf_rhs_nds_out2"} : <>
    %17 = blocker %21#1[%15] {handshake.bb = 3 : ui32, handshake.name = "lhs_out_bl_out2"} : <i1>, <>
    %18 = blocker %25[%16] {handshake.bb = 3 : ui32, handshake.name = "rhs_out_bl_out2"} : <i1>, <>
    %19 = cmpi eq, %17, %18 {handshake.bb = 3 : ui32, handshake.name = "out_eq_out2"} : <i1>
    %20 = andi %1, %4 {handshake.bb = 1 : ui32, handshake.name = "lhs_and"} : <i1>
    %21:2 = fork [2] %20 {handshake.bb = 1 : ui32, handshake.name = "lhs_fork"} : <i1>
    %22:2 = fork [2] %2 {handshake.bb = 2 : ui32, handshake.name = "rhs_fork1"} : <i1>
    %23:2 = fork [2] %5 {handshake.bb = 2 : ui32, handshake.name = "rhs_fork2"} : <i1>
    %24 = andi %22#0, %23#0 {handshake.bb = 2 : ui32, handshake.name = "rhs_and1"} : <i1>
    %25 = andi %22#1, %23#1 {handshake.bb = 2 : ui32, handshake.name = "rhs_and2"} : <i1>
    end {handshake.bb = 3 : ui32, handshake.name = "end"} %12, %19 : <i1>, <i1>
  }
}
