module {
  handshake.func @elastic_miter_unify_sup_lhs_unify_sup_rhs(%arg0: !handshake.channel<i1>, %arg1: !handshake.channel<i1>, %arg2: !handshake.channel<i1>, ...) -> !handshake.channel<i1> attributes {argNames = ["A_in", "Cond1", "Cond2"], resNames = ["EQ_B_out"]} {
    %0:2 = lazy_fork [2] %arg0 {handshake.bb = 0 : ui32, handshake.name = "in_fork_A_in"} : <i1>
    %1 = buffer %0#0 {handshake.bb = 0 : ui32, handshake.name = "lhs_in_buf_A_in", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %2 = buffer %0#1 {handshake.bb = 0 : ui32, handshake.name = "rhs_in_buf_A_in", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %3 = ndwire %1 {handshake.bb = 0 : ui32, handshake.name = "lhs_in_ndw_A_in"} : <i1>
    %4 = ndwire %2 {handshake.bb = 0 : ui32, handshake.name = "rhs_in_ndw_A_in"} : <i1>
    %5:2 = lazy_fork [2] %arg1 {handshake.bb = 0 : ui32, handshake.name = "in_fork_Cond1"} : <i1>
    %6 = buffer %5#0 {handshake.bb = 0 : ui32, handshake.name = "lhs_in_buf_Cond1", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %7 = buffer %5#1 {handshake.bb = 0 : ui32, handshake.name = "rhs_in_buf_Cond1", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %8 = ndwire %6 {handshake.bb = 0 : ui32, handshake.name = "lhs_in_ndw_Cond1"} : <i1>
    %9 = ndwire %7 {handshake.bb = 0 : ui32, handshake.name = "rhs_in_ndw_Cond1"} : <i1>
    %10:2 = lazy_fork [2] %arg2 {handshake.bb = 0 : ui32, handshake.name = "in_fork_Cond2"} : <i1>
    %11 = buffer %10#0 {handshake.bb = 0 : ui32, handshake.name = "lhs_in_buf_Cond2", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %12 = buffer %10#1 {handshake.bb = 0 : ui32, handshake.name = "rhs_in_buf_Cond2", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %13 = ndwire %11 {handshake.bb = 0 : ui32, handshake.name = "lhs_in_ndw_Cond2"} : <i1>
    %14 = ndwire %12 {handshake.bb = 0 : ui32, handshake.name = "rhs_in_ndw_Cond2"} : <i1>
    %15:2 = fork [2] %13 {handshake.bb = 1 : ui32, handshake.name = "lhs_fork_cond2"} : <i1>
    %16 = passer %8[%15#0] {handshake.bb = 1 : ui32, handshake.name = "lhs_passer1"} : <i1>, <i1>
    %17 = passer %3[%15#1] {handshake.bb = 1 : ui32, handshake.name = "lhs_passer2"} : <i1>, <i1>
    %18 = passer %17[%16] {handshake.bb = 1 : ui32, handshake.name = "lhs_passer3"} : <i1>, <i1>
    %19 = andi %9, %14 {handshake.bb = 2 : ui32, handshake.name = "rhs_andi"} : <i1>
    %20 = passer %4[%19] {handshake.bb = 2 : ui32, handshake.name = "rhs_passer"} : <i1>, <i1>
    %21 = ndsource {handshake.bb = 3 : ui32, handshake.name = "out_nds_B_out"} : <>
    %22:2 = lazy_fork [2] %21 {handshake.bb = 3 : ui32, handshake.name = "out_lf_B_out"} : <>
    %23 = buffer %22#0 {handshake.bb = 3 : ui32, handshake.name = "out_buf_lhs_nds_B_out", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <>
    %24 = buffer %22#1 {handshake.bb = 3 : ui32, handshake.name = "out_buf_rhs_nds_B_out", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <>
    %25 = blocker %18[%23] {handshake.bb = 3 : ui32, handshake.name = "out_lhs_bl_B_out"} : <i1>, <>
    %26 = blocker %20[%24] {handshake.bb = 3 : ui32, handshake.name = "out_rhs_bl_B_out"} : <i1>, <>
    %27 = ndwire %25 {handshake.bb = 3 : ui32, handshake.name = "lhs_out_ndw_B_out"} : <i1>
    %28 = ndwire %26 {handshake.bb = 3 : ui32, handshake.name = "rhs_out_ndw_B_out"} : <i1>
    %29 = buffer %27 {handshake.bb = 3 : ui32, handshake.name = "lhs_out_buf_B_out", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %30 = buffer %28 {handshake.bb = 3 : ui32, handshake.name = "rhs_out_buf_B_out", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %31 = cmpi eq, %29, %30 {handshake.bb = 3 : ui32, handshake.name = "out_eq_B_out"} : <i1>
    end {handshake.bb = 3 : ui32, handshake.name = "end"} %31 : <i1>
  }
}
