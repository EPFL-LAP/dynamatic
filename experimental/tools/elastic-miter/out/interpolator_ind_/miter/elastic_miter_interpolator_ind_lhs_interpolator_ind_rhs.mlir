module {
  handshake.func @elastic_miter_interpolator_ind_lhs_interpolator_ind_rhs(%arg0: !handshake.channel<i1>, %arg1: !handshake.channel<i1>, %arg2: !handshake.channel<i1>, ...) -> !handshake.channel<i1> attributes {argNames = ["A_in", "B_in", "Sup_in"], resNames = ["EQ_Sup_out"]} {
    %0:2 = lazy_fork [2] %arg0 {handshake.bb = 0 : ui32, handshake.name = "in_fork_A_in"} : <i1>
    %1 = buffer %0#0 {handshake.bb = 0 : ui32, handshake.name = "lhs_in_buf_A_in", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %2 = buffer %0#1 {handshake.bb = 0 : ui32, handshake.name = "rhs_in_buf_A_in", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %3 = ndwire %1 {handshake.bb = 0 : ui32, handshake.name = "lhs_in_ndw_A_in"} : <i1>
    %4 = ndwire %2 {handshake.bb = 0 : ui32, handshake.name = "rhs_in_ndw_A_in"} : <i1>
    %5:2 = lazy_fork [2] %arg1 {handshake.bb = 0 : ui32, handshake.name = "in_fork_B_in"} : <i1>
    %6 = buffer %5#0 {handshake.bb = 0 : ui32, handshake.name = "lhs_in_buf_B_in", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %7 = buffer %5#1 {handshake.bb = 0 : ui32, handshake.name = "rhs_in_buf_B_in", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %8 = ndwire %6 {handshake.bb = 0 : ui32, handshake.name = "lhs_in_ndw_B_in"} : <i1>
    %9 = ndwire %7 {handshake.bb = 0 : ui32, handshake.name = "rhs_in_ndw_B_in"} : <i1>
    %10:2 = lazy_fork [2] %arg2 {handshake.bb = 0 : ui32, handshake.name = "in_fork_Sup_in"} : <i1>
    %11 = buffer %10#0 {handshake.bb = 0 : ui32, handshake.name = "lhs_in_buf_Sup_in", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %12 = buffer %10#1 {handshake.bb = 0 : ui32, handshake.name = "rhs_in_buf_Sup_in", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %13 = ndwire %11 {handshake.bb = 0 : ui32, handshake.name = "lhs_in_ndw_Sup_in"} : <i1>
    %14 = ndwire %12 {handshake.bb = 0 : ui32, handshake.name = "rhs_in_ndw_Sup_in"} : <i1>
    %15:2 = fork [2] %8 {handshake.bb = 1 : ui32, handshake.name = "lhs_fork"} : <i1>
    %16 = spec_v2_interpolator %3, %15#0 {handshake.bb = 1 : ui32, handshake.name = "lhs_interpolate"} : <i1>
    %17 = spec_v2_repeating_init %15#1 {handshake.bb = 1 : ui32, handshake.name = "lhs_ri"} : <i1>
    %18 = passer %13[%17] {handshake.bb = 1 : ui32, handshake.name = "lhs_passer1"} : <i1>, <i1>
    %19 = passer %18[%16] {handshake.bb = 1 : ui32, handshake.name = "lhs_passer2"} : <i1>, <i1>
    %20 = spec_v2_repeating_init %9 {handshake.bb = 2 : ui32, handshake.name = "rhs_ri"} : <i1>
    %21 = spec_v2_interpolator %4, %20 {handshake.bb = 2 : ui32, handshake.name = "rhs_interpolate"} : <i1>
    %22 = passer %14[%21] {handshake.bb = 2 : ui32, handshake.name = "rhs_passer"} : <i1>, <i1>
    %23 = ndsource {handshake.bb = 3 : ui32, handshake.name = "out_nds_Sup_out"} : <>
    %24:2 = lazy_fork [2] %23 {handshake.bb = 3 : ui32, handshake.name = "out_lf_Sup_out"} : <>
    %25 = buffer %24#0 {handshake.bb = 3 : ui32, handshake.name = "out_buf_lhs_nds_Sup_out", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <>
    %26 = buffer %24#1 {handshake.bb = 3 : ui32, handshake.name = "out_buf_rhs_nds_Sup_out", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <>
    %27 = blocker %19[%25] {handshake.bb = 3 : ui32, handshake.name = "out_lhs_bl_Sup_out"} : <i1>, <>
    %28 = blocker %22[%26] {handshake.bb = 3 : ui32, handshake.name = "out_rhs_bl_Sup_out"} : <i1>, <>
    %29 = ndwire %27 {handshake.bb = 3 : ui32, handshake.name = "lhs_out_ndw_Sup_out"} : <i1>
    %30 = ndwire %28 {handshake.bb = 3 : ui32, handshake.name = "rhs_out_ndw_Sup_out"} : <i1>
    %31 = buffer %29 {handshake.bb = 3 : ui32, handshake.name = "lhs_out_buf_Sup_out", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %32 = buffer %30 {handshake.bb = 3 : ui32, handshake.name = "rhs_out_buf_Sup_out", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %33 = cmpi eq, %31, %32 {handshake.bb = 3 : ui32, handshake.name = "out_eq_Sup_out"} : <i1>
    end {handshake.bb = 3 : ui32, handshake.name = "end"} %33 : <i1>
  }
}
