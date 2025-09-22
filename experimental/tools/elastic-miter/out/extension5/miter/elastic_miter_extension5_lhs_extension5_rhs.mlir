module {
  handshake.func @elastic_miter_extension5_lhs_extension5_rhs(%arg0: !handshake.channel<i1>, %arg1: !handshake.channel<i1>, ...) -> !handshake.channel<i1> attributes {argNames = ["arg", "ctrl"], resNames = ["EQ_res"]} {
    %0:2 = lazy_fork [2] %arg0 {handshake.bb = 0 : ui32, handshake.name = "in_fork_arg"} : <i1>
    %1 = buffer %0#0, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 0 : ui32, handshake.name = "lhs_in_buf_arg"} : <i1>
    %2 = buffer %0#1, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 0 : ui32, handshake.name = "rhs_in_buf_arg"} : <i1>
    %3:2 = lazy_fork [2] %arg1 {handshake.bb = 0 : ui32, handshake.name = "in_fork_ctrl"} : <i1>
    %4 = buffer %3#0, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 0 : ui32, handshake.name = "lhs_in_buf_ctrl"} : <i1>
    %5 = buffer %3#1, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 0 : ui32, handshake.name = "rhs_in_buf_ctrl"} : <i1>
    %6 = ndsource {handshake.bb = 3 : ui32, handshake.name = "out_nds_res"} : <>
    %7:2 = lazy_fork [2] %6 {handshake.bb = 3 : ui32, handshake.name = "out_lf_res"} : <>
    %8 = buffer %7#0, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 3 : ui32, handshake.name = "out_buf_lhs_nds_res"} : <>
    %9 = buffer %7#1, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 3 : ui32, handshake.name = "out_buf_rhs_nds_res"} : <>
    %10 = blocker %13[%8] {handshake.bb = 3 : ui32, handshake.name = "lhs_out_bl_res"} : <i1>, <>
    %11 = blocker %29[%9] {handshake.bb = 3 : ui32, handshake.name = "rhs_out_bl_res"} : <i1>, <>
    %12 = cmpi eq, %10, %11 {handshake.bb = 3 : ui32, handshake.name = "out_eq_res"} : <i1>
    %13 = passer %1[%4] {handshake.bb = 1 : ui32, handshake.name = "lhs_passer"} : <i1>, <i1>
    %14:2 = fork [2] %2 {handshake.bb = 2 : ui32, handshake.name = "rhs_arg_fork"} : <i1>
    %15:2 = fork [2] %5 {handshake.bb = 2 : ui32, handshake.name = "rhs_ctrl_fork"} : <i1>
    %16 = andi %14#0, %15#0 {handshake.bb = 2 : ui32, handshake.name = "rhs_andi"} : <i1>
    %17 = not %15#1 {handshake.bb = 2 : ui32, handshake.name = "rhs_ctrl_not"} : <i1>
    %18 = ori %14#1, %17 {handshake.bb = 2 : ui32, handshake.name = "rhs_ori"} : <i1>
    %19 = buffer %18, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "rhs_or_buff"} : <i1>
    %20:3 = fork [3] %25 {handshake.bb = 2 : ui32, handshake.name = "rhs_ri_fork"} : <i1>
    %21 = buffer %20#0, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "rhs_buff"} : <i1>
    %22 = init %21 {handshake.bb = 2 : ui32, handshake.name = "rhs_initop", initToken = 0 : ui1} : <i1>
    %23 = source {handshake.bb = 2 : ui32, handshake.name = "rhs_source1"} : <>
    %24 = constant %23 {handshake.bb = 2 : ui32, handshake.name = "rhs_constant1", value = true} : <>, <i1>
    %25 = mux %22 [%24, %19] {handshake.bb = 2 : ui32, handshake.name = "rhs_ri_mux"} : <i1>, [<i1>, <i1>] to <i1>
    %26 = source {handshake.bb = 2 : ui32, handshake.name = "rhs_source2"} : <>
    %27 = constant %26 {handshake.bb = 2 : ui32, handshake.name = "rhs_constant2", value = true} : <>, <i1>
    %28 = mux %20#1 [%27, %16] {handshake.bb = 2 : ui32, handshake.name = "rhs_mux"} : <i1>, [<i1>, <i1>] to <i1>
    %29 = passer %20#2[%28] {handshake.bb = 2 : ui32, handshake.name = "rhs_passer"} : <i1>, <i1>
    end {handshake.bb = 3 : ui32, handshake.name = "end"} %12 : <i1>
  }
}
