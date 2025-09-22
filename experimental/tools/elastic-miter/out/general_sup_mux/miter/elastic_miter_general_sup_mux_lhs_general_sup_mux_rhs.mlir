module {
  handshake.func @elastic_miter_general_sup_mux_lhs_general_sup_mux_rhs(%arg0: !handshake.channel<i1>, %arg1: !handshake.channel<i1>, %arg2: !handshake.channel<i1>, %arg3: !handshake.channel<i1>, %arg4: !handshake.channel<i1>, ...) -> !handshake.channel<i1> attributes {argNames = ["sel", "ctrl1", "ctrl2", "a_in", "b_in"], resNames = ["EQ_res"]} {
    %0:2 = lazy_fork [2] %arg0 {handshake.bb = 0 : ui32, handshake.name = "in_fork_sel"} : <i1>
    %1 = buffer %0#0, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 0 : ui32, handshake.name = "lhs_in_buf_sel"} : <i1>
    %2 = buffer %0#1, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 0 : ui32, handshake.name = "rhs_in_buf_sel"} : <i1>
    %3:2 = lazy_fork [2] %arg1 {handshake.bb = 0 : ui32, handshake.name = "in_fork_ctrl1"} : <i1>
    %4 = buffer %3#0, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 0 : ui32, handshake.name = "lhs_in_buf_ctrl1"} : <i1>
    %5 = buffer %3#1, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 0 : ui32, handshake.name = "rhs_in_buf_ctrl1"} : <i1>
    %6:2 = lazy_fork [2] %arg2 {handshake.bb = 0 : ui32, handshake.name = "in_fork_ctrl2"} : <i1>
    %7 = buffer %6#0, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 0 : ui32, handshake.name = "lhs_in_buf_ctrl2"} : <i1>
    %8 = buffer %6#1, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 0 : ui32, handshake.name = "rhs_in_buf_ctrl2"} : <i1>
    %9:2 = lazy_fork [2] %arg3 {handshake.bb = 0 : ui32, handshake.name = "in_fork_a_in"} : <i1>
    %10 = buffer %9#0, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 0 : ui32, handshake.name = "lhs_in_buf_a_in"} : <i1>
    %11 = buffer %9#1, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 0 : ui32, handshake.name = "rhs_in_buf_a_in"} : <i1>
    %12:2 = lazy_fork [2] %arg4 {handshake.bb = 0 : ui32, handshake.name = "in_fork_b_in"} : <i1>
    %13 = buffer %12#0, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 0 : ui32, handshake.name = "lhs_in_buf_b_in"} : <i1>
    %14 = buffer %12#1, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 0 : ui32, handshake.name = "rhs_in_buf_b_in"} : <i1>
    %15 = ndsource {handshake.bb = 3 : ui32, handshake.name = "out_nds_res"} : <>
    %16:2 = lazy_fork [2] %15 {handshake.bb = 3 : ui32, handshake.name = "out_lf_res"} : <>
    %17 = buffer %16#0, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 3 : ui32, handshake.name = "out_buf_lhs_nds_res"} : <>
    %18 = buffer %16#1, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 3 : ui32, handshake.name = "out_buf_rhs_nds_res"} : <>
    %19 = blocker %29[%17] {handshake.bb = 3 : ui32, handshake.name = "lhs_out_bl_res"} : <i1>, <>
    %20 = blocker %33[%18] {handshake.bb = 3 : ui32, handshake.name = "rhs_out_bl_res"} : <i1>, <>
    %21 = cmpi eq, %19, %20 {handshake.bb = 3 : ui32, handshake.name = "out_eq_res"} : <i1>
    %22:2 = fork [2] %1 {handshake.bb = 1 : ui32, handshake.name = "lhs_fork_sel"} : <i1>
    %23:2 = fork [2] %4 {handshake.bb = 1 : ui32, handshake.name = "lhs_fork_ctrl1"} : <i1>
    %24:2 = fork [2] %7 {handshake.bb = 1 : ui32, handshake.name = "lhs_fork_ctrl2"} : <i1>
    %25 = mux %22#0 [%23#0, %24#0] {handshake.bb = 1 : ui32, handshake.name = "lhs_sel_mux"} : <i1>, [<i1>, <i1>] to <i1>
    %26 = passer %22#1[%25] {handshake.bb = 1 : ui32, handshake.name = "lhs_sel_passer"} : <i1>, <i1>
    %27 = passer %10[%23#1] {handshake.bb = 1 : ui32, handshake.name = "lhs_a_passer"} : <i1>, <i1>
    %28 = passer %13[%24#1] {handshake.bb = 1 : ui32, handshake.name = "lhs_b_passer"} : <i1>, <i1>
    %29 = mux %26 [%27, %28] {handshake.bb = 1 : ui32, handshake.name = "lhs_data_mux"} : <i1>, [<i1>, <i1>] to <i1>
    %30:2 = fork [2] %2 {handshake.bb = 2 : ui32, handshake.name = "rhs_fork_sel"} : <i1>
    %31 = mux %30#0 [%5, %8] {handshake.bb = 2 : ui32, handshake.name = "rhs_sel_mux"} : <i1>, [<i1>, <i1>] to <i1>
    %32 = mux %30#1 [%11, %14] {handshake.bb = 2 : ui32, handshake.name = "rhs_data_mux"} : <i1>, [<i1>, <i1>] to <i1>
    %33 = passer %32[%31] {handshake.bb = 2 : ui32, handshake.name = "rhs_data_passer"} : <i1>, <i1>
    end {handshake.bb = 3 : ui32, handshake.name = "end"} %21 : <i1>
  }
}
