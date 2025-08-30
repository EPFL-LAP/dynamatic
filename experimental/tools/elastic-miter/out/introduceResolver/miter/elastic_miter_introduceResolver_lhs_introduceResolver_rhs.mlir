module {
  handshake.func @elastic_miter_introduceResolver_lhs_introduceResolver_rhs(%arg0: !handshake.channel<i1>, %arg1: !handshake.channel<i1>, ...) -> !handshake.channel<i1> attributes {argNames = ["loopContinue", "confirmSpec_backedge"], resNames = ["EQ_confirmSpec"]} {
    %0 = source {handshake.bb = 0 : ui32, handshake.name = "backedge_source_confirmSpec"} : <>
    %1 = ndconstant %0 {handshake.bb = 0 : ui32, handshake.name = "backedge_constant_confirmSpec"} : <>, <i1>
    %2:2 = fork [2] %1 {handshake.bb = 0 : ui32, handshake.name = "backedge_lf_start_confirmSpec"} : <i1>
    sink %arg1 {handshake.bb = 0 : ui32, handshake.name = "backedge_sink_start_confirmSpec"} : <i1>
    %3:2 = lazy_fork [2] %24 {handshake.bb = 4 : ui32, handshake.name = "backedge_lf_end_confirmSpec"} : <i1>
    %4 = cmpi eq, %2#1, %3#1 {handshake.bb = 4 : ui32, handshake.name = "backedge_eq_confirmSpec"} : <i1>
    sink %4 {handshake.bb = 4 : ui32, handshake.name = "backedge_sink_end_confirmSpec"} : <i1>
    %5:2 = fork [2] %arg0 {handshake.bb = 1 : ui32, handshake.name = "ctx_fork"} : <i1>
    %6:2 = fork [2] %2#0 {handshake.bb = 1 : ui32, handshake.name = "ctx_fork_cs"} : <i1>
    %7 = passer %5#0[%6#0] {handshake.bb = 1 : ui32, handshake.name = "ctx_passer"} : <i1>, <i1>
    %8 = spec_v2_nd_speculator %7 {handshake.bb = 1 : ui32, handshake.name = "ndspec"} : <i1>
    %9 = spec_v2_repeating_init %8 {handshake.bb = 1 : ui32, handshake.name = "ctx_ri", initToken = 1 : ui1} : <i1>
    %10 = buffer %9, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "ctx_buffer"} : <i1>
    %11:2 = lazy_fork [2] %5#1 {handshake.bb = 1 : ui32, handshake.name = "in_fork_loopContinue"} : <i1>
    %12 = buffer %11#0, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 1 : ui32, handshake.name = "lhs_in_buf_loopContinue"} : <i1>
    %13 = buffer %11#1, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 1 : ui32, handshake.name = "rhs_in_buf_loopContinue"} : <i1>
    %14:2 = lazy_fork [2] %10 {handshake.bb = 1 : ui32, handshake.name = "in_fork_specLoopContinue"} : <i1>
    %15 = buffer %14#0, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 1 : ui32, handshake.name = "lhs_in_buf_specLoopContinue"} : <i1>
    %16 = buffer %14#1, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 1 : ui32, handshake.name = "rhs_in_buf_specLoopContinue"} : <i1>
    %17:2 = lazy_fork [2] %6#1 {handshake.bb = 1 : ui32, handshake.name = "in_fork_confirmSpec_backedge"} : <i1>
    %18 = buffer %17#0, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 1 : ui32, handshake.name = "lhs_in_buf_confirmSpec_backedge"} : <i1>
    %19 = buffer %17#1, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 1 : ui32, handshake.name = "rhs_in_buf_confirmSpec_backedge"} : <i1>
    %20 = ndsource {handshake.bb = 4 : ui32, handshake.name = "out_nds_confirmSpec"} : <>
    %21:2 = lazy_fork [2] %20 {handshake.bb = 4 : ui32, handshake.name = "out_lf_confirmSpec"} : <>
    %22 = buffer %21#0, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 4 : ui32, handshake.name = "out_buf_lhs_nds_confirmSpec"} : <>
    %23 = buffer %21#1, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 4 : ui32, handshake.name = "out_buf_rhs_nds_confirmSpec"} : <>
    %24 = blocker %30[%22] {handshake.bb = 4 : ui32, handshake.name = "lhs_out_bl_confirmSpec"} : <i1>, <>
    %25 = blocker %31[%23] {handshake.bb = 4 : ui32, handshake.name = "rhs_out_bl_confirmSpec"} : <i1>, <>
    %26 = cmpi eq, %3#0, %25 {handshake.bb = 4 : ui32, handshake.name = "out_eq_confirmSpec"} : <i1>
    %27 = passer %12[%18] {handshake.bb = 2 : ui32, handshake.name = "lhs_passer"} : <i1>, <i1>
    %28 = spec_v2_repeating_init %27 {handshake.bb = 2 : ui32, handshake.name = "lhs_ri", initToken = 1 : ui1} : <i1>
    %29 = buffer %28, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "lhs_buffer"} : <i1>
    %30 = spec_v2_interpolator %29, %15 {handshake.bb = 2 : ui32, handshake.name = "lhs_interpolate"} : <i1>
    sink %19 {handshake.bb = 3 : ui32, handshake.name = "rhs_vm_sink_2"} : <i1>
    %31 = spec_v2_resolver %13, %16 {handshake.bb = 3 : ui32, handshake.name = "rhs_resolver"} : <i1>
    end {handshake.bb = 4 : ui32, handshake.name = "end"} %26 : <i1>
  }
}
