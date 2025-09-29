module {
  handshake.func @elastic_miter_sup_mu_mux1_lhs_sup_mu_mux1_rhs(%arg0: !handshake.channel<i1>, %arg1: !handshake.channel<i1>, %arg2: !handshake.channel<i1>, %arg3: !handshake.channel<i1>, ...) -> !handshake.channel<i1> attributes {argNames = ["cond", "dataT", "dataF", "ctrl"], resNames = ["EQ_res"]} {
    %0:2 = lazy_fork [2] %arg0 {handshake.bb = 0 : ui32, handshake.name = "in_fork_cond"} : <i1>
    %1 = buffer %0#0, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 0 : ui32, handshake.name = "lhs_in_buf_cond"} : <i1>
    %2 = buffer %0#1, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 0 : ui32, handshake.name = "rhs_in_buf_cond"} : <i1>
    %3:2 = lazy_fork [2] %arg1 {handshake.bb = 0 : ui32, handshake.name = "in_fork_dataT"} : <i1>
    %4 = buffer %3#0, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 0 : ui32, handshake.name = "lhs_in_buf_dataT"} : <i1>
    %5 = buffer %3#1, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 0 : ui32, handshake.name = "rhs_in_buf_dataT"} : <i1>
    %6:2 = lazy_fork [2] %arg2 {handshake.bb = 0 : ui32, handshake.name = "in_fork_dataF"} : <i1>
    %7 = buffer %6#0, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 0 : ui32, handshake.name = "lhs_in_buf_dataF"} : <i1>
    %8 = buffer %6#1, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 0 : ui32, handshake.name = "rhs_in_buf_dataF"} : <i1>
    %9:2 = lazy_fork [2] %arg3 {handshake.bb = 0 : ui32, handshake.name = "in_fork_ctrl"} : <i1>
    %10 = buffer %9#0, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 0 : ui32, handshake.name = "lhs_in_buf_ctrl"} : <i1>
    %11 = buffer %9#1, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 0 : ui32, handshake.name = "rhs_in_buf_ctrl"} : <i1>
    %12 = source {handshake.bb = 3 : ui32, handshake.name = "out_src_res"} : <>
    %13:2 = lazy_fork [2] %12 {handshake.bb = 3 : ui32, handshake.name = "out_lf_res"} : <>
    %14 = buffer %13#0, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 3 : ui32, handshake.name = "out_buf_lhs_nds_res"} : <>
    %15 = buffer %13#1, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 3 : ui32, handshake.name = "out_buf_rhs_nds_res"} : <>
    %16 = blocker %21[%14] {handshake.bb = 3 : ui32, handshake.name = "lhs_out_bl_res"} : <i1>, <>
    %17 = blocker %39[%15] {handshake.bb = 3 : ui32, handshake.name = "rhs_out_bl_res"} : <i1>, <>
    %18 = cmpi eq, %16, %17 {handshake.bb = 3 : ui32, handshake.name = "out_eq_res"} : <i1>
    %19 = passer %7[%10] {handshake.bb = 1 : ui32, handshake.name = "lhs_dataF_passer"} : <i1>, <i1>
    %20 = init %1 {handshake.bb = 1 : ui32, handshake.name = "lhs_initop", initToken = 0 : ui1} : <i1>
    %21 = mux %20 [%19, %4] {handshake.bb = 1 : ui32, handshake.name = "lhs_res_mux"} : <i1>, [<i1>, <i1>] to <i1>
    %22:3 = fork [3] %11 {handshake.bb = 2 : ui32, handshake.name = "rhs_ctrl_fork"} : <i1>
    %23 = mux %26 [%22#0, %2] {handshake.bb = 2 : ui32, handshake.name = "rhs_ctrl_mux"} : <i1>, [<i1>, <i1>] to <i1>
    %24:3 = fork [3] %23 {handshake.bb = 2 : ui32, handshake.name = "rhs_ctrl_muxed_fork"} : <i1>
    %25 = buffer %24#0, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "rhs_buff"} : <i1>
    %26 = init %25 {handshake.bb = 2 : ui32, handshake.name = "rhs_ctrl_init", initToken = 0 : ui1} : <i1>
    %27 = source {handshake.bb = 2 : ui32, handshake.name = "rhs_source2"} : <>
    %28 = constant %27 {handshake.bb = 2 : ui32, handshake.name = "rhs_constant2", value = true} : <>, <i1>
    %29 = init %24#1 {handshake.bb = 2 : ui32, handshake.name = "rhs_initop2", initToken = 0 : ui1} : <i1>
    %30 = not %22#2 {handshake.bb = 2 : ui32, handshake.name = "rhs_ctrl_not"} : <i1>
    %31 = mux %29 [%30, %28] {handshake.bb = 2 : ui32, handshake.name = "rhs_mux_passer_ctrl"} : <i1>, [<i1>, <i1>] to <i1>
    %32 = passer %24#2[%31] {handshake.bb = 2 : ui32, handshake.name = "rhs_ctrl_passer"} : <i1>, <i1>
    %33 = init %32 {handshake.bb = 2 : ui32, handshake.name = "rhs_initop", initToken = 0 : ui1} : <i1>
    %34:2 = fork [2] %33 {handshake.bb = 2 : ui32, handshake.name = "rhs_init_fork"} : <i1>
    %35 = source {handshake.bb = 2 : ui32, handshake.name = "rhs_source"} : <>
    %36 = constant %35 {handshake.bb = 2 : ui32, handshake.name = "rhs_constant", value = true} : <>, <i1>
    %37 = mux %34#0 [%22#1, %36] {handshake.bb = 2 : ui32, handshake.name = "rhs_ctrl_mux2"} : <i1>, [<i1>, <i1>] to <i1>
    %38 = mux %34#1 [%8, %5] {handshake.bb = 2 : ui32, handshake.name = "rhs_data_mux"} : <i1>, [<i1>, <i1>] to <i1>
    %39 = passer %38[%37] {handshake.bb = 2 : ui32, handshake.name = "rhs_passer"} : <i1>, <i1>
    end {handshake.bb = 3 : ui32, handshake.name = "end"} %18 : <i1>
  }
}
