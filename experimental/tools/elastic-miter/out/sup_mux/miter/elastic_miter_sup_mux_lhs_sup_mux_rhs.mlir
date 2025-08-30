module {
  handshake.func @elastic_miter_sup_mux_lhs_sup_mux_rhs(%arg0: !handshake.channel<i1>, %arg1: !handshake.channel<i1>, %arg2: !handshake.channel<i1>, ...) -> !handshake.channel<i1> attributes {argNames = ["loopLiveIn", "iterLiveOut", "oldContinue"], resNames = ["EQ_iterLiveIn"]} {
    %0:2 = lazy_fork [2] %arg0 {handshake.bb = 0 : ui32, handshake.name = "in_fork_loopLiveIn"} : <i1>
    %1 = buffer %0#0, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 0 : ui32, handshake.name = "lhs_in_buf_loopLiveIn"} : <i1>
    %2 = buffer %0#1, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 0 : ui32, handshake.name = "rhs_in_buf_loopLiveIn"} : <i1>
    %3:2 = lazy_fork [2] %arg1 {handshake.bb = 0 : ui32, handshake.name = "in_fork_iterLiveOut"} : <i1>
    %4 = buffer %3#0, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 0 : ui32, handshake.name = "lhs_in_buf_iterLiveOut"} : <i1>
    %5 = buffer %3#1, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 0 : ui32, handshake.name = "rhs_in_buf_iterLiveOut"} : <i1>
    %6:2 = lazy_fork [2] %arg2 {handshake.bb = 0 : ui32, handshake.name = "in_fork_oldContinue"} : <i1>
    %7 = buffer %6#0, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 0 : ui32, handshake.name = "lhs_in_buf_oldContinue"} : <i1>
    %8 = buffer %6#1, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 0 : ui32, handshake.name = "rhs_in_buf_oldContinue"} : <i1>
    %9 = ndsource {handshake.bb = 3 : ui32, handshake.name = "out_nds_iterLiveIn"} : <>
    %10:2 = lazy_fork [2] %9 {handshake.bb = 3 : ui32, handshake.name = "out_lf_iterLiveIn"} : <>
    %11 = buffer %10#0, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 3 : ui32, handshake.name = "out_buf_lhs_nds_iterLiveIn"} : <>
    %12 = buffer %10#1, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 3 : ui32, handshake.name = "out_buf_rhs_nds_iterLiveIn"} : <>
    %13 = blocker %19[%11] {handshake.bb = 3 : ui32, handshake.name = "lhs_out_bl_iterLiveIn"} : <i1>, <>
    %14 = blocker %27[%12] {handshake.bb = 3 : ui32, handshake.name = "rhs_out_bl_iterLiveIn"} : <i1>, <>
    %15 = cmpi eq, %13, %14 {handshake.bb = 3 : ui32, handshake.name = "out_eq_iterLiveIn"} : <i1>
    %16:2 = fork [2] %7 {handshake.bb = 1 : ui32, handshake.name = "lhs_vm_fork_2"} : <i1>
    %17 = passer %4[%16#0] {handshake.bb = 1 : ui32, handshake.name = "lhs_passer"} : <i1>, <i1>
    %18 = init %16#1 {handshake.bb = 1 : ui32, handshake.name = "lhs_oldInit", initToken = 0 : ui1} : <i1>
    %19 = mux %18 [%1, %17] {handshake.bb = 1 : ui32, handshake.name = "lhs_data_mux"} : <i1>, [<i1>, <i1>] to <i1>
    %20:2 = fork [2] %8 {handshake.bb = 2 : ui32, handshake.name = "rhs_vm_fork_2"} : <i1>
    %21 = spec_v2_repeating_init %20#0 {handshake.bb = 2 : ui32, handshake.name = "rhs_ri1", initToken = 1 : ui1} : <i1>
    %22 = buffer %21, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "rhs_buffer1"} : <i1>
    %23 = init %22 {handshake.bb = 2 : ui32, handshake.name = "rhs_newInit", initToken = 0 : ui1} : <i1>
    %24 = mux %23 [%2, %5] {handshake.bb = 2 : ui32, handshake.name = "rhs_data_mux"} : <i1>, [<i1>, <i1>] to <i1>
    %25 = spec_v2_repeating_init %20#1 {handshake.bb = 2 : ui32, handshake.name = "rhs_ri2", initToken = 1 : ui1} : <i1>
    %26 = buffer %25, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "rhs_buffer2"} : <i1>
    %27 = passer %24[%26] {handshake.bb = 2 : ui32, handshake.name = "rhs_passer"} : <i1>, <i1>
    end {handshake.bb = 3 : ui32, handshake.name = "end"} %15 : <i1>
  }
}
