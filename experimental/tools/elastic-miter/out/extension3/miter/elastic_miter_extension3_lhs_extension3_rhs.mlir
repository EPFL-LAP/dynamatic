module {
  handshake.func @elastic_miter_extension3_lhs_extension3_rhs(%arg0: !handshake.channel<i1>, ...) -> !handshake.channel<i1> attributes {argNames = ["arg"], resNames = ["EQ_res"]} {
    %0:2 = lazy_fork [2] %arg0 {handshake.bb = 0 : ui32, handshake.name = "in_fork_arg"} : <i1>
    %1 = buffer %0#0, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 0 : ui32, handshake.name = "lhs_in_buf_arg"} : <i1>
    %2 = buffer %0#1, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 0 : ui32, handshake.name = "rhs_in_buf_arg"} : <i1>
    %3 = ndsource {handshake.bb = 3 : ui32, handshake.name = "out_nds_res"} : <>
    %4:2 = lazy_fork [2] %3 {handshake.bb = 3 : ui32, handshake.name = "out_lf_res"} : <>
    %5 = buffer %4#0, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 3 : ui32, handshake.name = "out_buf_lhs_nds_res"} : <>
    %6 = buffer %4#1, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 3 : ui32, handshake.name = "out_buf_rhs_nds_res"} : <>
    %7 = blocker %10[%5] {handshake.bb = 3 : ui32, handshake.name = "lhs_out_bl_res"} : <i1>, <>
    %8 = blocker %22[%6] {handshake.bb = 3 : ui32, handshake.name = "rhs_out_bl_res"} : <i1>, <>
    %9 = cmpi eq, %7, %8 {handshake.bb = 3 : ui32, handshake.name = "out_eq_res"} : <i1>
    %10 = init %1 {handshake.bb = 1 : ui32, handshake.name = "lhs_initop", initToken = 0 : ui1} : <i1>
    %11:2 = fork [2] %2 {handshake.bb = 2 : ui32, handshake.name = "rhs_arg_fork"} : <i1>
    %12 = buffer %11#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "rhs_arg_buff"} : <i1>
    %13 = buffer %18, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "rhs_buff"} : <i1>
    %14 = init %13 {handshake.bb = 2 : ui32, handshake.name = "rhs_initop", initToken = 0 : ui1} : <i1>
    %15:3 = fork [3] %14 {handshake.bb = 2 : ui32, handshake.name = "rhs_ri_fork"} : <i1>
    %16 = source {handshake.bb = 2 : ui32, handshake.name = "rhs_source1"} : <>
    %17 = constant %16 {handshake.bb = 2 : ui32, handshake.name = "rhs_constant1", value = true} : <>, <i1>
    %18 = mux %15#0 [%17, %11#0] {handshake.bb = 2 : ui32, handshake.name = "rhs_ri_mux"} : <i1>, [<i1>, <i1>] to <i1>
    %19 = source {handshake.bb = 2 : ui32, handshake.name = "rhs_source2"} : <>
    %20 = constant %19 {handshake.bb = 2 : ui32, handshake.name = "rhs_constant2", value = true} : <>, <i1>
    %21 = mux %15#1 [%20, %12] {handshake.bb = 2 : ui32, handshake.name = "rhs_mux"} : <i1>, [<i1>, <i1>] to <i1>
    %22 = passer %15#2[%21] {handshake.bb = 2 : ui32, handshake.name = "rhs_passer"} : <i1>, <i1>
    end {handshake.bb = 3 : ui32, handshake.name = "end"} %9 : <i1>
  }
}
