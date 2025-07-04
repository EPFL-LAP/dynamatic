module {
  handshake.func @elastic_miter_resolver_lhs_resolver_rhs(%arg0: !handshake.channel<i1>, %arg1: !handshake.channel<i1>, ...) -> !handshake.channel<i1> attributes {argNames = ["Actual", "Generated"], resNames = ["EQ_Confirm"]} {
    %0:2 = lazy_fork [2] %5#0 {handshake.bb = 0 : ui32, handshake.name = "in_fork_Actual"} : <i1>
    %1 = buffer %0#0 {handshake.bb = 0 : ui32, handshake.name = "lhs_in_buf_Actual", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_DV", NUM_SLOTS = 3 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %2 = buffer %0#1 {handshake.bb = 0 : ui32, handshake.name = "rhs_in_buf_Actual", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_DV", NUM_SLOTS = 3 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %3 = ndwire %1 {handshake.bb = 0 : ui32, handshake.name = "lhs_in_ndw_Actual"} : <i1>
    %4 = ndwire %2 {handshake.bb = 0 : ui32, handshake.name = "rhs_in_ndw_Actual"} : <i1>
    %5:2 = lazy_fork [2] %arg0 {handshake.bb = 0 : ui32, handshake.name = "in_nd_fork"} : <i1>
    %6 = spec_v2_nd_speculator %5#1 {handshake.bb = 0 : ui32, handshake.name = "in_nd_speculator"} : <i1>
    %7 = spec_v2_repeating_init %6 {handshake.bb = 0 : ui32, handshake.name = "in_ri"} : <i1>
    sink %arg1 {handshake.bb = 0 : ui32, handshake.name = "in_sink"} : <i1>
    %8:2 = lazy_fork [2] %7 {handshake.bb = 0 : ui32, handshake.name = "in_fork_Generated"} : <i1>
    %9 = buffer %8#0 {handshake.bb = 0 : ui32, handshake.name = "lhs_in_buf_Generated", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_DV", NUM_SLOTS = 3 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %10 = buffer %8#1 {handshake.bb = 0 : ui32, handshake.name = "rhs_in_buf_Generated", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_DV", NUM_SLOTS = 3 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %11 = ndwire %9 {handshake.bb = 0 : ui32, handshake.name = "lhs_in_ndw_Generated"} : <i1>
    %12 = ndwire %10 {handshake.bb = 0 : ui32, handshake.name = "rhs_in_ndw_Generated"} : <i1>
    %13 = passer %3[%18#0] {handshake.bb = 1 : ui32, handshake.name = "lhs_passer"} : <i1>, <i1>
    %14 = buffer %13 {handshake.bb = 1 : ui32, handshake.name = "lhs_buf1", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %15 = buffer %14 {handshake.bb = 1 : ui32, handshake.name = "lhs_buf2", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i1>
    %16 = spec_v2_repeating_init %15 {handshake.bb = 1 : ui32, handshake.name = "lhs_ri"} : <i1>
    %17 = spec_v2_interpolator %16, %11 {handshake.bb = 1 : ui32, handshake.name = "lhs_interpolator"} : <i1>
    %18:2 = lazy_fork [2] %17 {handshake.bb = 1 : ui32, handshake.name = "lhs_fork"} : <i1>
    %19 = spec_v2_resolver %4, %12 {handshake.bb = 2 : ui32, handshake.name = "rhs_resolver"} : <i1>
    %20 = ndwire %18#1 {handshake.bb = 3 : ui32, handshake.name = "lhs_out_ndw_Confirm"} : <i1>
    %21 = ndwire %19 {handshake.bb = 3 : ui32, handshake.name = "rhs_out_ndw_Confirm"} : <i1>
    %22 = buffer %20 {handshake.bb = 3 : ui32, handshake.name = "lhs_out_buf_Confirm", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_DV", NUM_SLOTS = 3 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %23 = buffer %21 {handshake.bb = 3 : ui32, handshake.name = "rhs_out_buf_Confirm", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_DV", NUM_SLOTS = 3 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %24 = cmpi eq, %22, %23 {handshake.bb = 3 : ui32, handshake.name = "out_eq_Confirm"} : <i1>
    end {handshake.bb = 3 : ui32, handshake.name = "end"} %24 : <i1>
  }
}
