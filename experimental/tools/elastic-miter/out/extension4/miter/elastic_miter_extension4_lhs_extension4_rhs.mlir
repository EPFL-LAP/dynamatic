module {
  handshake.func @elastic_miter_extension4_lhs_extension4_rhs(%arg0: !handshake.channel<i1>, %arg1: !handshake.channel<i1>, ...) -> !handshake.channel<i1> attributes {argNames = ["arg", "ctrl"], resNames = ["EQ_res"]} {
    %0:2 = lazy_fork [2] %arg0 {handshake.bb = 0 : ui32, handshake.name = "in_fork_arg"} : <i1>
    %1 = buffer %0#0, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 0 : ui32, handshake.name = "lhs_in_buf_arg"} : <i1>
    %2 = buffer %0#1, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 0 : ui32, handshake.name = "rhs_in_buf_arg"} : <i1>
    %3:2 = lazy_fork [2] %arg1 {handshake.bb = 0 : ui32, handshake.name = "in_fork_ctrl"} : <i1>
    %4 = buffer %3#0, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 0 : ui32, handshake.name = "lhs_in_buf_ctrl"} : <i1>
    %5 = buffer %3#1, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 0 : ui32, handshake.name = "rhs_in_buf_ctrl"} : <i1>
    %6:2 = lazy_fork [2] %13 {handshake.bb = 3 : ui32, handshake.name = "out_lf_res"} : <>
    %7 = buffer %6#0, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 3 : ui32, handshake.name = "out_buf_lhs_nds_res"} : <>
    %8 = buffer %6#1, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 3 : ui32, handshake.name = "out_buf_rhs_nds_res"} : <>
    %9 = blocker %15[%7] {handshake.bb = 3 : ui32, handshake.name = "lhs_out_bl_res"} : <i1>, <>
    %10 = blocker %32[%8] {handshake.bb = 3 : ui32, handshake.name = "rhs_out_bl_res"} : <i1>, <>
    %11 = cmpi eq, %9, %10 {handshake.bb = 3 : ui32, handshake.name = "out_eq_res"} : <i1>
    %12:2 = fork [2] %14 {handshake.bb = 1 : ui32, handshake.name = "ctx_passer_fork"} : <i1>
    %13 = unconstant %12#1 {handshake.bb = 3 : ui32, handshake.name = "ctx_unconstant"} : <i1>, <>
    %14 = passer %1[%4] {handshake.bb = 1 : ui32, handshake.name = "lhs_passer"} : <i1>, <i1>
    %15 = init %12#0 {handshake.bb = 1 : ui32, handshake.name = "lhs_initop", initToken = 0 : ui1} : <i1>
    %16:2 = fork [2] %2 {handshake.bb = 2 : ui32, handshake.name = "rhs_arg_fork"} : <i1>
    %17:2 = fork [2] %5 {handshake.bb = 2 : ui32, handshake.name = "rhs_ctrl_fork"} : <i1>
    %18 = andi %16#0, %17#0 {handshake.bb = 2 : ui32, handshake.name = "rhs_andi"} : <i1>
    %19 = buffer %18, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "rhs_and_buff"} : <i1>
    %20 = not %17#1 {handshake.bb = 2 : ui32, handshake.name = "rhs_ctrl_not"} : <i1>
    %21 = ori %16#1, %20 {handshake.bb = 2 : ui32, handshake.name = "rhs_ori"} : <i1>
    %22 = buffer %27, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "rhs_buff"} : <i1>
    %23 = init %22 {handshake.bb = 2 : ui32, handshake.name = "rhs_initop", initToken = 0 : ui1} : <i1>
    %24:3 = fork [3] %23 {handshake.bb = 2 : ui32, handshake.name = "rhs_ri_fork"} : <i1>
    %25 = source {handshake.bb = 2 : ui32, handshake.name = "rhs_source1"} : <>
    %26 = constant %25 {handshake.bb = 2 : ui32, handshake.name = "rhs_constant1", value = true} : <>, <i1>
    %27 = mux %24#0 [%26, %21] {handshake.bb = 2 : ui32, handshake.name = "rhs_ri_mux"} : <i1>, [<i1>, <i1>] to <i1>
    %28 = source {handshake.bb = 2 : ui32, handshake.name = "rhs_source2"} : <>
    %29 = constant %28 {handshake.bb = 2 : ui32, handshake.name = "rhs_constant2", value = true} : <>, <i1>
    %30 = mux %24#1 [%29, %19] {handshake.bb = 2 : ui32, handshake.name = "rhs_mux"} : <i1>, [<i1>, <i1>] to <i1>
    %31 = passer %24#2[%30] {handshake.bb = 2 : ui32, handshake.name = "rhs_passer"} : <i1>, <i1>
    %32 = buffer %31, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "rhs_p_buff"} : <i1>
    end {handshake.bb = 3 : ui32, handshake.name = "end"} %11 : <i1>
  }
}
