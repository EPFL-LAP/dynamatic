module {
  handshake.func @elastic_miter_sup_gamma_mux1_lhs_sup_gamma_mux1_rhs(%arg0: !handshake.channel<i1>, %arg1: !handshake.channel<i1>, %arg2: !handshake.channel<i1>, ...) -> !handshake.channel<i1> attributes {argNames = ["cond", "dataT", "dataF"], resNames = ["EQ_res"]} {
    %0:2 = lazy_fork [2] %arg0 {handshake.bb = 0 : ui32, handshake.name = "in_fork_cond"} : <i1>
    %1 = buffer %0#0, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 0 : ui32, handshake.name = "lhs_in_buf_cond"} : <i1>
    %2 = buffer %0#1, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 0 : ui32, handshake.name = "rhs_in_buf_cond"} : <i1>
    %3:2 = lazy_fork [2] %arg1 {handshake.bb = 0 : ui32, handshake.name = "in_fork_dataT"} : <i1>
    %4 = buffer %3#0, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 0 : ui32, handshake.name = "lhs_in_buf_dataT"} : <i1>
    %5 = buffer %3#1, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 0 : ui32, handshake.name = "rhs_in_buf_dataT"} : <i1>
    %6:2 = lazy_fork [2] %arg2 {handshake.bb = 0 : ui32, handshake.name = "in_fork_dataF"} : <i1>
    %7 = buffer %6#0, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 0 : ui32, handshake.name = "lhs_in_buf_dataF"} : <i1>
    %8 = buffer %6#1, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 0 : ui32, handshake.name = "rhs_in_buf_dataF"} : <i1>
    %9 = source {handshake.bb = 3 : ui32, handshake.name = "out_src_res"} : <>
    %10:2 = lazy_fork [2] %9 {handshake.bb = 3 : ui32, handshake.name = "out_lf_res"} : <>
    %11 = buffer %10#0, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 3 : ui32, handshake.name = "out_buf_lhs_nds_res"} : <>
    %12 = buffer %10#1, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 3 : ui32, handshake.name = "out_buf_rhs_nds_res"} : <>
    %13 = blocker %20[%11] {handshake.bb = 3 : ui32, handshake.name = "lhs_out_bl_res"} : <i1>, <>
    %14 = blocker %29[%12] {handshake.bb = 3 : ui32, handshake.name = "rhs_out_bl_res"} : <i1>, <>
    %15 = cmpi eq, %13, %14 {handshake.bb = 3 : ui32, handshake.name = "out_eq_res"} : <i1>
    %16:3 = fork [3] %1 {handshake.bb = 1 : ui32, handshake.name = "lhs_cond_fork"} : <i1>
    %17 = passer %4[%16#0] {handshake.bb = 1 : ui32, handshake.name = "lhs_dataT_passer"} : <i1>, <i1>
    %18 = not %16#1 {handshake.bb = 1 : ui32, handshake.name = "lhs_not"} : <i1>
    %19 = passer %7[%18] {handshake.bb = 1 : ui32, handshake.name = "lhs_dataF_passer"} : <i1>, <i1>
    %20 = mux %16#2 [%19, %17] {handshake.bb = 1 : ui32, handshake.name = "lhs_res_mux"} : <i1>, [<i1>, <i1>] to <i1>
    %21 = init %24 {handshake.bb = 2 : ui32, handshake.name = "rhs_initop", initToken = 0 : ui1} : <i1>
    %22:3 = fork [3] %21 {handshake.bb = 2 : ui32, handshake.name = "rhs_cst_dup_fork"} : <i1>
    %23 = not %22#0 {handshake.bb = 2 : ui32, handshake.name = "rhs_not"} : <i1>
    %24 = buffer %23, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "rhs_buff"} : <i1>
    %25:2 = fork [2] %2 {handshake.bb = 2 : ui32, handshake.name = "rhs_cond_fork"} : <i1>
    %26 = not %25#0 {handshake.bb = 2 : ui32, handshake.name = "rhs_cond_not"} : <i1>
    %27 = mux %22#1 [%26, %25#1] {handshake.bb = 2 : ui32, handshake.name = "rhs_cond_mux"} : <i1>, [<i1>, <i1>] to <i1>
    %28 = mux %22#2 [%8, %5] {handshake.bb = 2 : ui32, handshake.name = "rhs_data_mux"} : <i1>, [<i1>, <i1>] to <i1>
    %29 = passer %28[%27] {handshake.bb = 2 : ui32, handshake.name = "rhs_passer"} : <i1>, <i1>
    end {handshake.bb = 3 : ui32, handshake.name = "end"} %15 : <i1>
  }
}
