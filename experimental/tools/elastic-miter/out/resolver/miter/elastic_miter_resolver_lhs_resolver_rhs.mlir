module {
  handshake.func @elastic_miter_resolver_lhs_resolver_rhs(%arg0: !handshake.channel<i1>, %arg1: !handshake.channel<i1>, ...) -> !handshake.channel<i1> attributes {argNames = ["Actual", "Generated"], resNames = ["EQ_Confirm"]} {
    %0:2 = lazy_fork [2] %5#0 {handshake.bb = 0 : ui32, handshake.name = "in_fork_Actual"} : <i1>
    %1 = buffer %0#0 {handshake.bb = 0 : ui32, handshake.name = "lhs_in_buf_Actual", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_DV", NUM_SLOTS = 4 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %2 = buffer %0#1 {handshake.bb = 0 : ui32, handshake.name = "rhs_in_buf_Actual", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_DV", NUM_SLOTS = 4 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %3 = ndwire %1 {handshake.bb = 0 : ui32, handshake.name = "lhs_in_ndw_Actual"} : <i1>
    %4 = ndwire %2 {handshake.bb = 0 : ui32, handshake.name = "rhs_in_ndw_Actual"} : <i1>
    %5:2 = lazy_fork [2] %arg0 {handshake.bb = 0 : ui32, handshake.name = "in_nd_fork"} : <i1>
    %6 = buffer %5#1 {handshake.bb = 0 : ui32, handshake.name = "in_nd_spec_pre_buffer", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_DV", NUM_SLOTS = 4 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %7 = spec_v2_nd_speculator %6 {handshake.bb = 0 : ui32, handshake.name = "in_nd_speculator"} : <i1>
    %8 = spec_v2_repeating_init %7 {handshake.bb = 0 : ui32, handshake.name = "in_ri"} : <i1>
    sink %arg1 {handshake.bb = 0 : ui32, handshake.name = "in_sink"} : <i1>
    %9:2 = lazy_fork [2] %8 {handshake.bb = 0 : ui32, handshake.name = "in_fork_Generated"} : <i1>
    %10 = buffer %9#0 {handshake.bb = 0 : ui32, handshake.name = "lhs_in_buf_Generated", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_DV", NUM_SLOTS = 4 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %11 = buffer %9#1 {handshake.bb = 0 : ui32, handshake.name = "rhs_in_buf_Generated", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_DV", NUM_SLOTS = 4 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %12 = ndwire %10 {handshake.bb = 0 : ui32, handshake.name = "lhs_in_ndw_Generated"} : <i1>
    %13 = ndwire %11 {handshake.bb = 0 : ui32, handshake.name = "rhs_in_ndw_Generated"} : <i1>
    %14 = passer %3[%19#0] {handshake.bb = 1 : ui32, handshake.name = "lhs_passer"} : <i1>, <i1>
    %15 = buffer %14 {handshake.bb = 1 : ui32, handshake.name = "lhs_buf1", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %16 = buffer %15 {handshake.bb = 1 : ui32, handshake.name = "lhs_buf2", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i1>
    %17 = spec_v2_repeating_init %16 {handshake.bb = 1 : ui32, handshake.name = "lhs_ri"} : <i1>
    %18 = spec_v2_interpolator %17, %12 {handshake.bb = 1 : ui32, handshake.name = "lhs_interpolator"} : <i1>
    %19:2 = fork [2] %18 {handshake.bb = 1 : ui32, handshake.name = "lhs_fork"} : <i1>
    %20 = spec_v2_resolver %4, %13 {handshake.bb = 2 : ui32, handshake.name = "rhs_resolver"} : <i1>
    %21 = ndsource {handshake.bb = 3 : ui32, handshake.name = "out_nds_Confirm"} : <>
    %22:2 = lazy_fork [2] %21 {handshake.bb = 3 : ui32, handshake.name = "out_lf_Confirm"} : <>
    %23 = buffer %22#0 {handshake.bb = 3 : ui32, handshake.name = "out_buf_lhs_nds_Confirm", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_DV", NUM_SLOTS = 4 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <>
    %24 = buffer %22#1 {handshake.bb = 3 : ui32, handshake.name = "out_buf_rhs_nds_Confirm", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_DV", NUM_SLOTS = 4 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <>
    %25 = blocker %19#1[%23] {handshake.bb = 3 : ui32, handshake.name = "out_lhs_bl_Confirm"} : <i1>, <>
    %26 = blocker %20[%24] {handshake.bb = 3 : ui32, handshake.name = "out_rhs_bl_Confirm"} : <i1>, <>
    %27 = ndwire %25 {handshake.bb = 3 : ui32, handshake.name = "lhs_out_ndw_Confirm"} : <i1>
    %28 = ndwire %26 {handshake.bb = 3 : ui32, handshake.name = "rhs_out_ndw_Confirm"} : <i1>
    %29 = buffer %27 {handshake.bb = 3 : ui32, handshake.name = "lhs_out_buf_Confirm", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_DV", NUM_SLOTS = 4 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %30 = buffer %28 {handshake.bb = 3 : ui32, handshake.name = "rhs_out_buf_Confirm", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_DV", NUM_SLOTS = 4 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %31 = cmpi eq, %29, %30 {handshake.bb = 3 : ui32, handshake.name = "out_eq_Confirm"} : <i1>
    end {handshake.bb = 3 : ui32, handshake.name = "end"} %31 : <i1>
  }
}
