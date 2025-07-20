module {
  handshake.func @elastic_miter_sup_mux_lhs_sup_mux_rhs(%arg0: !handshake.channel<i1>, %arg1: !handshake.channel<i1>, %arg2: !handshake.channel<i1>, ...) -> !handshake.channel<i1> attributes {argNames = ["loopLiveIn", "oldContinue", "iterLiveOut"], resNames = ["EQ_iterLiveIn"]} {
    %0:2 = lazy_fork [2] %arg0 {handshake.bb = 0 : ui32, handshake.name = "in_fork_loopLiveIn"} : <i1>
    %1 = buffer %0#0, bufferType = FIFO_BREAK_DV, numSlots = 2 {handshake.bb = 0 : ui32, handshake.name = "lhs_in_buf_loopLiveIn"} : <i1>
    %2 = buffer %0#1, bufferType = FIFO_BREAK_DV, numSlots = 2 {handshake.bb = 0 : ui32, handshake.name = "rhs_in_buf_loopLiveIn"} : <i1>
    %3 = ndwire %1 {handshake.bb = 0 : ui32, handshake.name = "lhs_in_ndw_loopLiveIn"} : <i1>
    %4 = ndwire %2 {handshake.bb = 0 : ui32, handshake.name = "rhs_in_ndw_loopLiveIn"} : <i1>
    %5:2 = lazy_fork [2] %arg1 {handshake.bb = 0 : ui32, handshake.name = "in_fork_oldContinue"} : <i1>
    %6 = buffer %5#0, bufferType = FIFO_BREAK_DV, numSlots = 2 {handshake.bb = 0 : ui32, handshake.name = "lhs_in_buf_oldContinue"} : <i1>
    %7 = buffer %5#1, bufferType = FIFO_BREAK_DV, numSlots = 2 {handshake.bb = 0 : ui32, handshake.name = "rhs_in_buf_oldContinue"} : <i1>
    %8 = ndwire %6 {handshake.bb = 0 : ui32, handshake.name = "lhs_in_ndw_oldContinue"} : <i1>
    %9 = ndwire %7 {handshake.bb = 0 : ui32, handshake.name = "rhs_in_ndw_oldContinue"} : <i1>
    %10:2 = lazy_fork [2] %arg2 {handshake.bb = 0 : ui32, handshake.name = "in_fork_iterLiveOut"} : <i1>
    %11 = buffer %10#0, bufferType = FIFO_BREAK_DV, numSlots = 2 {handshake.bb = 0 : ui32, handshake.name = "lhs_in_buf_iterLiveOut"} : <i1>
    %12 = buffer %10#1, bufferType = FIFO_BREAK_DV, numSlots = 2 {handshake.bb = 0 : ui32, handshake.name = "rhs_in_buf_iterLiveOut"} : <i1>
    %13 = ndwire %11 {handshake.bb = 0 : ui32, handshake.name = "lhs_in_ndw_iterLiveOut"} : <i1>
    %14 = ndwire %12 {handshake.bb = 0 : ui32, handshake.name = "rhs_in_ndw_iterLiveOut"} : <i1>
    %15:2 = fork [2] %8 {handshake.bb = 1 : ui32, handshake.name = "lhs_vm_fork_1"} : <i1>
    %16 = passer %13[%15#0] {handshake.bb = 2 : ui32, handshake.name = "lhs_passer"} : <i1>, <i1>
    %17 = init %15#1 {handshake.bb = 2 : ui32, handshake.name = "lhs_oldInit", initToken = 0 : ui1} : <i1>
    %18 = mux %17 [%3, %16] {handshake.bb = 2 : ui32, handshake.name = "lhs_data_mux"} : <i1>, [<i1>, <i1>] to <i1>
    %19:2 = fork [2] %9 {handshake.bb = 3 : ui32, handshake.name = "rhs_vm_fork_1"} : <i1>
    %20 = spec_v2_repeating_init %19#0 {handshake.bb = 4 : ui32, handshake.name = "rhs_ri1", initToken = 1 : ui1} : <i1>
    %21 = init %20 {handshake.bb = 4 : ui32, handshake.name = "rhs_newInit", initToken = 0 : ui1} : <i1>
    %22 = mux %21 [%4, %14] {handshake.bb = 4 : ui32, handshake.name = "rhs_data_mux"} : <i1>, [<i1>, <i1>] to <i1>
    %23 = spec_v2_repeating_init %19#1 {handshake.bb = 4 : ui32, handshake.name = "rhs_ri2", initToken = 1 : ui1} : <i1>
    %24 = passer %22[%23] {handshake.bb = 4 : ui32, handshake.name = "rhs_passer"} : <i1>, <i1>
    %25 = ndsource {handshake.bb = 5 : ui32, handshake.name = "out_nds_iterLiveIn"} : <>
    %26:2 = lazy_fork [2] %25 {handshake.bb = 5 : ui32, handshake.name = "out_lf_iterLiveIn"} : <>
    %27 = buffer %26#0, bufferType = FIFO_BREAK_DV, numSlots = 2 {handshake.bb = 5 : ui32, handshake.name = "out_buf_lhs_nds_iterLiveIn"} : <>
    %28 = buffer %26#1, bufferType = FIFO_BREAK_DV, numSlots = 2 {handshake.bb = 5 : ui32, handshake.name = "out_buf_rhs_nds_iterLiveIn"} : <>
    %29 = blocker %18[%27] {handshake.bb = 5 : ui32, handshake.name = "out_lhs_bl_iterLiveIn"} : <i1>, <>
    %30 = blocker %24[%28] {handshake.bb = 5 : ui32, handshake.name = "out_rhs_bl_iterLiveIn"} : <i1>, <>
    %31 = ndwire %29 {handshake.bb = 5 : ui32, handshake.name = "lhs_out_ndw_iterLiveIn"} : <i1>
    %32 = ndwire %30 {handshake.bb = 5 : ui32, handshake.name = "rhs_out_ndw_iterLiveIn"} : <i1>
    %33 = buffer %31, bufferType = FIFO_BREAK_DV, numSlots = 2 {handshake.bb = 5 : ui32, handshake.name = "lhs_out_buf_iterLiveIn"} : <i1>
    %34 = buffer %32, bufferType = FIFO_BREAK_DV, numSlots = 2 {handshake.bb = 5 : ui32, handshake.name = "rhs_out_buf_iterLiveIn"} : <i1>
    %35 = cmpi eq, %33, %34 {handshake.bb = 5 : ui32, handshake.name = "out_eq_iterLiveIn"} : <i1>
    end {handshake.bb = 5 : ui32, handshake.name = "end"} %35 : <i1>
  }
}
