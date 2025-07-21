module {
  handshake.func @elastic_miter_introduceResolver_lhs_introduceResolver_rhs(%arg0: !handshake.channel<i1>, %arg1: !handshake.channel<i1>, ...) -> !handshake.channel<i1> attributes {argNames = ["loopContinue", "confirmSpec_backedge"], resNames = ["EQ_confirmSpec"]} {
    %0 = source {handshake.bb = 0 : ui32, handshake.name = "backedge_source_confirmSpec"} : <>
    %1 = ndconstant %0 {handshake.bb = 0 : ui32, handshake.name = "backedge_constant_confirmSpec"} : <>, <i1>
    %2:2 = fork [2] %1 {handshake.bb = 0 : ui32, handshake.name = "backedge_lf_start_confirmSpec"} : <i1>
    sink %arg1 {handshake.bb = 0 : ui32, handshake.name = "backedge_sink_start_confirmSpec"} : <i1>
    %3:2 = lazy_fork [2] %21 {handshake.bb = 4 : ui32, handshake.name = "backedge_lf_end_confirmSpec"} : <i1>
    %4 = cmpi eq, %2#1, %3#1 {handshake.bb = 4 : ui32, handshake.name = "backedge_eq_confirmSpec"} : <i1>
    sink %4 {handshake.bb = 4 : ui32, handshake.name = "backedge_sink_end_confirmSpec"} : <i1>
    %5:2 = fork [2] %arg0 {handshake.bb = 1 : ui32, handshake.name = "ctx_fork"} : <i1>
    %6:2 = fork [2] %2#0 {handshake.bb = 1 : ui32, handshake.name = "ctx_fork_cs"} : <i1>
    %7 = passer %5#0[%6#0] {handshake.bb = 1 : ui32, handshake.name = "ctx_passer"} : <i1>, <i1>
    %8 = spec_v2_nd_speculator %7 {handshake.bb = 1 : ui32, handshake.name = "ndspec"} : <i1>
    %9:2 = lazy_fork [2] %5#1 {handshake.bb = 1 : ui32, handshake.name = "in_fork_loopContinue"} : <i1>
    %10 = buffer %9#0, bufferType = FIFO_BREAK_DV, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "lhs_in_buf_loopContinue"} : <i1>
    %11 = buffer %9#1, bufferType = FIFO_BREAK_DV, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "rhs_in_buf_loopContinue"} : <i1>
    %12:2 = lazy_fork [2] %8 {handshake.bb = 1 : ui32, handshake.name = "in_fork_specLoopContinue"} : <i1>
    %13 = buffer %12#0, bufferType = FIFO_BREAK_DV, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "lhs_in_buf_specLoopContinue"} : <i1>
    %14 = buffer %12#1, bufferType = FIFO_BREAK_DV, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "rhs_in_buf_specLoopContinue"} : <i1>
    %15:2 = lazy_fork [2] %6#1 {handshake.bb = 1 : ui32, handshake.name = "in_fork_confirmSpec_backedge"} : <i1>
    %16 = buffer %15#0, bufferType = FIFO_BREAK_DV, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "lhs_in_buf_confirmSpec_backedge"} : <i1>
    %17 = ndsource {handshake.bb = 4 : ui32, handshake.name = "out_nds_confirmSpec"} : <>
    %18:2 = lazy_fork [2] %17 {handshake.bb = 4 : ui32, handshake.name = "out_lf_confirmSpec"} : <>
    %19 = buffer %18#0, bufferType = FIFO_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "out_buf_lhs_nds_confirmSpec"} : <>
    %20 = buffer %18#1, bufferType = FIFO_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "out_buf_rhs_nds_confirmSpec"} : <>
    %21 = blocker %27[%19] {handshake.bb = 4 : ui32, handshake.name = "lhs_out_bl_confirmSpec"} : <i1>, <>
    %22 = blocker %28[%20] {handshake.bb = 4 : ui32, handshake.name = "rhs_out_bl_confirmSpec"} : <i1>, <>
    %23 = cmpi eq, %3#0, %22 {handshake.bb = 4 : ui32, handshake.name = "out_eq_confirmSpec"} : <i1>
    %24 = passer %10[%16] {handshake.bb = 2 : ui32, handshake.name = "lhs_passer"} : <i1>, <i1>
    %25 = spec_v2_repeating_init %24 {handshake.bb = 2 : ui32, handshake.name = "lhs_ri", initToken = 1 : ui1} : <i1>
    %26 = buffer %25, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "lhs_buffer"} : <i1>
    %27 = spec_v2_interpolator %26, %13 {handshake.bb = 2 : ui32, handshake.name = "lhs_interpolate"} : <i1>
    sink %15#1 {handshake.bb = 3 : ui32, handshake.name = "rhs_vm_sink_2"} : <i1>
    %28 = spec_v2_resolver %11, %14 {handshake.bb = 3 : ui32, handshake.name = "rhs_resolver"} : <i1>
    end {handshake.bb = 4 : ui32, handshake.name = "end"} %23 : <i1>
  }
}
