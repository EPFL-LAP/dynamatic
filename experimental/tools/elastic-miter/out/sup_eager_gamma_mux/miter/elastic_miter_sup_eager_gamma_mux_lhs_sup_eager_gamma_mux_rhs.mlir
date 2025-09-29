module {
  handshake.func @elastic_miter_sup_eager_gamma_mux_lhs_sup_eager_gamma_mux_rhs(%arg0: !handshake.channel<i1>, %arg1: !handshake.channel<i1>, %arg2: !handshake.channel<i1>, ...) -> !handshake.channel<i1> attributes {argNames = ["A_in", "B_in", "ctrl"], resNames = ["EQ_res"]} {
    %0:2 = lazy_fork [2] %arg0 {handshake.bb = 0 : ui32, handshake.name = "in_fork_A_in"} : <i1>
    %1 = buffer %0#0, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 0 : ui32, handshake.name = "lhs_in_buf_A_in"} : <i1>
    %2 = buffer %0#1, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 0 : ui32, handshake.name = "rhs_in_buf_A_in"} : <i1>
    %3:2 = lazy_fork [2] %arg1 {handshake.bb = 0 : ui32, handshake.name = "in_fork_B_in"} : <i1>
    %4 = buffer %3#0, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 0 : ui32, handshake.name = "lhs_in_buf_B_in"} : <i1>
    %5 = buffer %3#1, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 0 : ui32, handshake.name = "rhs_in_buf_B_in"} : <i1>
    %6:2 = lazy_fork [2] %arg2 {handshake.bb = 0 : ui32, handshake.name = "in_fork_ctrl"} : <i1>
    %7 = buffer %6#0, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 0 : ui32, handshake.name = "lhs_in_buf_ctrl"} : <i1>
    %8 = buffer %6#1, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 0 : ui32, handshake.name = "rhs_in_buf_ctrl"} : <i1>
    %9 = source {handshake.bb = 3 : ui32, handshake.name = "out_src_res"} : <>
    %10:2 = lazy_fork [2] %9 {handshake.bb = 3 : ui32, handshake.name = "out_lf_res"} : <>
    %11 = buffer %10#0, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 3 : ui32, handshake.name = "out_buf_lhs_nds_res"} : <>
    %12 = buffer %10#1, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 3 : ui32, handshake.name = "out_buf_rhs_nds_res"} : <>
    %13 = blocker %23[%11] {handshake.bb = 3 : ui32, handshake.name = "lhs_out_bl_res"} : <i1>, <>
    %14 = blocker %31[%12] {handshake.bb = 3 : ui32, handshake.name = "rhs_out_bl_res"} : <i1>, <>
    %15 = cmpi eq, %13, %14 {handshake.bb = 3 : ui32, handshake.name = "out_eq_res"} : <i1>
    %16 = init %19 {handshake.bb = 1 : ui32, handshake.name = "lhs_initop", initToken = 0 : ui1} : <i1>
    %17:2 = fork [2] %16 {handshake.bb = 1 : ui32, handshake.name = "lhs_cst_dup_fork"} : <i1>
    %18 = not %17#0 {handshake.bb = 1 : ui32, handshake.name = "lhs_not"} : <i1>
    %19 = buffer %18, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "lhs_buff"} : <i1>
    %20:2 = fork [2] %7 {handshake.bb = 1 : ui32, handshake.name = "lhs_ctrl_fork"} : <i1>
    %21 = passer %1[%20#0] {handshake.bb = 1 : ui32, handshake.name = "lhs_a_passer"} : <i1>, <i1>
    %22 = passer %4[%20#1] {handshake.bb = 1 : ui32, handshake.name = "lhs_b_passer"} : <i1>, <i1>
    %23 = mux %17#1 [%22, %21] {handshake.bb = 1 : ui32, handshake.name = "lhs_mux"} : <i1>, [<i1>, <i1>] to <i1>
    %24 = init %27 {handshake.bb = 2 : ui32, handshake.name = "rhs_initop", initToken = 0 : ui1} : <i1>
    %25:3 = fork [3] %24 {handshake.bb = 2 : ui32, handshake.name = "rhs_cst_dup_fork"} : <i1>
    %26 = not %25#0 {handshake.bb = 2 : ui32, handshake.name = "rhs_not"} : <i1>
    %27 = buffer %26, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "rhs_buff"} : <i1>
    %28:2 = fork [2] %8 {handshake.bb = 2 : ui32, handshake.name = "rhs_ctrl_fork"} : <i1>
    %29 = mux %25#1 [%28#1, %28#0] {handshake.bb = 2 : ui32, handshake.name = "rhs_ctrl_mux"} : <i1>, [<i1>, <i1>] to <i1>
    %30 = mux %25#2 [%5, %2] {handshake.bb = 2 : ui32, handshake.name = "rhs_mux"} : <i1>, [<i1>, <i1>] to <i1>
    %31 = passer %30[%29] {handshake.bb = 2 : ui32, handshake.name = "rhs_res_passer"} : <i1>, <i1>
    end {handshake.bb = 3 : ui32, handshake.name = "end"} %15 : <i1>
  }
}
