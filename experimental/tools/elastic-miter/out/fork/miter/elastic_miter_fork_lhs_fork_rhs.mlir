module {
  handshake.func @elastic_miter_fork_lhs_fork_rhs(%arg0: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>, !handshake.channel<i1>) attributes {argNames = ["A_in"], resNames = ["EQ_B_out", "EQ_C_out"]} {
    %0:2 = lazy_fork [2] %arg0 {handshake.bb = 0 : ui32, handshake.name = "in_fork_A_in"} : <i1>
    %1 = buffer %0#0 {handshake.bb = 0 : ui32, handshake.name = "lhs_in_buf_A_in", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %2 = buffer %0#1 {handshake.bb = 0 : ui32, handshake.name = "rhs_in_buf_A_in", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %3 = ndwire %1 {handshake.bb = 0 : ui32, handshake.name = "lhs_in_ndw_A_in"} : <i1>
    %4 = ndwire %2 {handshake.bb = 0 : ui32, handshake.name = "rhs_in_ndw_A_in"} : <i1>
    %5:2 = fork [2] %3 {handshake.bb = 1 : ui32, handshake.name = "lhs_fork"} : <i1>
    %6:2 = lazy_fork [2] %4 {handshake.bb = 2 : ui32, handshake.name = "rhs_fork"} : <i1>
    %7 = ndsource {handshake.bb = 3 : ui32, handshake.name = "out_nds_B_out"} : <i1>
    %8:2 = lazy_fork [2] %7 {handshake.bb = 3 : ui32, handshake.name = "out_lf_B_out"} : <i1>
    %9 = buffer %8#0 {handshake.bb = 3 : ui32, handshake.name = "out_buf_lhs_nds_B_out", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %10 = buffer %8#1 {handshake.bb = 3 : ui32, handshake.name = "out_buf_rhs_nds_B_out", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %11 = transfer_control %5#0[%9] {handshake.bb = 3 : ui32, handshake.name = "out_lhs_tc_B_out"} : <i1>, <i1>
    %12 = transfer_control %6#0[%10] {handshake.bb = 3 : ui32, handshake.name = "out_rhs_tc_B_out"} : <i1>, <i1>
    %13 = ndwire %11 {handshake.bb = 3 : ui32, handshake.name = "lhs_out_ndw_B_out"} : <i1>
    %14 = ndwire %12 {handshake.bb = 3 : ui32, handshake.name = "rhs_out_ndw_B_out"} : <i1>
    %15 = buffer %13 {handshake.bb = 3 : ui32, handshake.name = "lhs_out_buf_B_out", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %16 = buffer %14 {handshake.bb = 3 : ui32, handshake.name = "rhs_out_buf_B_out", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %17 = cmpi eq, %15, %16 {handshake.bb = 3 : ui32, handshake.name = "out_eq_B_out"} : <i1>
    %18 = ndsource {handshake.bb = 3 : ui32, handshake.name = "out_nds_C_out"} : <i1>
    %19:2 = lazy_fork [2] %18 {handshake.bb = 3 : ui32, handshake.name = "out_lf_C_out"} : <i1>
    %20 = buffer %19#0 {handshake.bb = 3 : ui32, handshake.name = "out_buf_lhs_nds_C_out", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %21 = buffer %19#1 {handshake.bb = 3 : ui32, handshake.name = "out_buf_rhs_nds_C_out", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %22 = transfer_control %5#1[%20] {handshake.bb = 3 : ui32, handshake.name = "out_lhs_tc_C_out"} : <i1>, <i1>
    %23 = transfer_control %6#1[%21] {handshake.bb = 3 : ui32, handshake.name = "out_rhs_tc_C_out"} : <i1>, <i1>
    %24 = ndwire %22 {handshake.bb = 3 : ui32, handshake.name = "lhs_out_ndw_C_out"} : <i1>
    %25 = ndwire %23 {handshake.bb = 3 : ui32, handshake.name = "rhs_out_ndw_C_out"} : <i1>
    %26 = buffer %24 {handshake.bb = 3 : ui32, handshake.name = "lhs_out_buf_C_out", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %27 = buffer %25 {handshake.bb = 3 : ui32, handshake.name = "rhs_out_buf_C_out", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %28 = cmpi eq, %26, %27 {handshake.bb = 3 : ui32, handshake.name = "out_eq_C_out"} : <i1>
    end {handshake.bb = 3 : ui32, handshake.name = "end"} %17, %28 : <i1>, <i1>
  }
}
