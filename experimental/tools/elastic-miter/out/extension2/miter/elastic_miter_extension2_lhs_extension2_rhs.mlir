module {
  handshake.func @elastic_miter_extension2_lhs_extension2_rhs(%arg0: !handshake.channel<i1>, ...) -> !handshake.channel<i1> attributes {argNames = ["arg"], resNames = ["EQ_res"]} {
    %0:2 = lazy_fork [2] %arg0 {handshake.bb = 0 : ui32, handshake.name = "in_fork_arg"} : <i1>
    %1 = buffer %0#0, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 0 : ui32, handshake.name = "lhs_in_buf_arg"} : <i1>
    %2 = buffer %0#1, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 0 : ui32, handshake.name = "rhs_in_buf_arg"} : <i1>
    %3 = ndsource {handshake.bb = 3 : ui32, handshake.name = "out_nds_res"} : <>
    %4:2 = lazy_fork [2] %3 {handshake.bb = 3 : ui32, handshake.name = "out_lf_res"} : <>
    %5 = buffer %4#0, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 3 : ui32, handshake.name = "out_buf_lhs_nds_res"} : <>
    %6 = buffer %4#1, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 3 : ui32, handshake.name = "out_buf_rhs_nds_res"} : <>
    %7 = blocker %11[%5] {handshake.bb = 3 : ui32, handshake.name = "lhs_out_bl_res"} : <i1>, <>
    %8 = blocker %23[%6] {handshake.bb = 3 : ui32, handshake.name = "rhs_out_bl_res"} : <i1>, <>
    %9 = cmpi eq, %7, %8 {handshake.bb = 3 : ui32, handshake.name = "out_eq_res"} : <i1>
    %10 = not %1 {handshake.bb = 1 : ui32, handshake.name = "lhs_not1"} : <i1>
    %11 = not %10 {handshake.bb = 1 : ui32, handshake.name = "lhs_not2"} : <i1>
    %12:2 = fork [2] %2 {handshake.bb = 2 : ui32, handshake.name = "rhs_arg_fork"} : <i1>
    %13 = buffer %12#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "rhs_arg_buff"} : <i1>
    %14 = buffer %19#0, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "rhs_buff"} : <i1>
    %15 = init %14 {handshake.bb = 2 : ui32, handshake.name = "rhs_initop", initToken = 0 : ui1} : <i1>
    %16 = source {handshake.bb = 2 : ui32, handshake.name = "rhs_source1"} : <>
    %17 = constant %16 {handshake.bb = 2 : ui32, handshake.name = "rhs_constant1", value = true} : <>, <i1>
    %18 = mux %15 [%17, %13] {handshake.bb = 2 : ui32, handshake.name = "rhs_ri_mux"} : <i1>, [<i1>, <i1>] to <i1>
    %19:3 = fork [3] %18 {handshake.bb = 2 : ui32, handshake.name = "rhs_ri_fork"} : <i1>
    %20 = source {handshake.bb = 2 : ui32, handshake.name = "rhs_source2"} : <>
    %21 = constant %20 {handshake.bb = 2 : ui32, handshake.name = "rhs_constant2", value = true} : <>, <i1>
    %22 = mux %19#1 [%21, %12#1] {handshake.bb = 2 : ui32, handshake.name = "rhs_mux"} : <i1>, [<i1>, <i1>] to <i1>
    %23 = passer %19#2[%22] {handshake.bb = 2 : ui32, handshake.name = "rhs_passer"} : <i1>, <i1>
    end {handshake.bb = 3 : ui32, handshake.name = "end"} %9 : <i1>
  }
}
