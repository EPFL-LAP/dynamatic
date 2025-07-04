module {
  handshake.func @elastic_miter_interpolator_ident_lhs_interpolator_ident_rhs(%arg0: !handshake.channel<i1>, ...) -> !handshake.channel<i1> attributes {argNames = ["A_in"], resNames = ["EQ_B_out"]} {
    %0:2 = lazy_fork [2] %arg0 {handshake.bb = 0 : ui32, handshake.name = "in_fork_A_in"} : <i1>
    %1 = buffer %0#0 {handshake.bb = 0 : ui32, handshake.name = "lhs_in_buf_A_in", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %2 = buffer %0#1 {handshake.bb = 0 : ui32, handshake.name = "rhs_in_buf_A_in", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %3 = ndwire %1 {handshake.bb = 0 : ui32, handshake.name = "lhs_in_ndw_A_in"} : <i1>
    %4 = ndwire %2 {handshake.bb = 0 : ui32, handshake.name = "rhs_in_ndw_A_in"} : <i1>
    %5:2 = fork [2] %4 {handshake.bb = 2 : ui32, handshake.name = "rhs_fork"} : <i1>
    %6 = spec_v2_interpolator %5#0, %5#1 {handshake.bb = 2 : ui32, handshake.name = "rhs_interpolate"} : <i1>
    %7 = ndwire %3 {handshake.bb = 3 : ui32, handshake.name = "lhs_out_ndw_B_out"} : <i1>
    %8 = ndwire %6 {handshake.bb = 3 : ui32, handshake.name = "rhs_out_ndw_B_out"} : <i1>
    %9 = buffer %7 {handshake.bb = 3 : ui32, handshake.name = "lhs_out_buf_B_out", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %10 = buffer %8 {handshake.bb = 3 : ui32, handshake.name = "rhs_out_buf_B_out", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %11 = cmpi eq, %9, %10 {handshake.bb = 3 : ui32, handshake.name = "out_eq_B_out"} : <i1>
    end {handshake.bb = 3 : ui32, handshake.name = "end"} %11 : <i1>
  }
}
