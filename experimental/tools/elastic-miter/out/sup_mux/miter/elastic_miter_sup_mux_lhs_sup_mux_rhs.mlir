module {
  handshake.func @elastic_miter_sup_mux_lhs_sup_mux_rhs(%arg0: !handshake.channel<i1>, %arg1: !handshake.channel<i1>, %arg2: !handshake.channel<i1>, ...) -> !handshake.channel<i1> attributes {argNames = ["In1", "In2", "Ctrl"], resNames = ["EQ_Out1"]} {
    %0:2 = lazy_fork [2] %arg0 {handshake.bb = 0 : ui32, handshake.name = "in_fork_In1"} : <i1>
    %1 = buffer %0#0 {handshake.bb = 0 : ui32, handshake.name = "lhs_in_buf_In1", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_DV", NUM_SLOTS = 2 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %2 = buffer %0#1 {handshake.bb = 0 : ui32, handshake.name = "rhs_in_buf_In1", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_DV", NUM_SLOTS = 2 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %3 = ndwire %1 {handshake.bb = 0 : ui32, handshake.name = "lhs_in_ndw_In1"} : <i1>
    %4 = ndwire %2 {handshake.bb = 0 : ui32, handshake.name = "rhs_in_ndw_In1"} : <i1>
    %5:2 = lazy_fork [2] %arg1 {handshake.bb = 0 : ui32, handshake.name = "in_fork_In2"} : <i1>
    %6 = buffer %5#0 {handshake.bb = 0 : ui32, handshake.name = "lhs_in_buf_In2", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_DV", NUM_SLOTS = 2 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %7 = buffer %5#1 {handshake.bb = 0 : ui32, handshake.name = "rhs_in_buf_In2", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_DV", NUM_SLOTS = 2 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %8 = ndwire %6 {handshake.bb = 0 : ui32, handshake.name = "lhs_in_ndw_In2"} : <i1>
    %9 = ndwire %7 {handshake.bb = 0 : ui32, handshake.name = "rhs_in_ndw_In2"} : <i1>
    %10:2 = lazy_fork [2] %arg2 {handshake.bb = 0 : ui32, handshake.name = "in_fork_Ctrl"} : <i1>
    %11 = buffer %10#0 {handshake.bb = 0 : ui32, handshake.name = "lhs_in_buf_Ctrl", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_DV", NUM_SLOTS = 2 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %12 = buffer %10#1 {handshake.bb = 0 : ui32, handshake.name = "rhs_in_buf_Ctrl", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_DV", NUM_SLOTS = 2 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %13 = ndwire %11 {handshake.bb = 0 : ui32, handshake.name = "lhs_in_ndw_Ctrl"} : <i1>
    %14 = ndwire %12 {handshake.bb = 0 : ui32, handshake.name = "rhs_in_ndw_Ctrl"} : <i1>
    %15:2 = fork [2] %13 {handshake.bb = 1 : ui32, handshake.name = "lhs_fork_data_mux"} : <i1>
    %16 = init %15#0 {handshake.bb = 1 : ui32, handshake.name = "lhs_init_buffer_ctrl", hw.parameters = {INITIAL_TOKEN = false, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %17 = passer %3[%15#1] {handshake.bb = 1 : ui32, handshake.name = "lhs_passer"} : <i1>, <i1>
    %18 = mux %16 [%8, %17] {handshake.bb = 1 : ui32, handshake.name = "lhs_data_mux"} : <i1>, [<i1>, <i1>] to <i1>
    %19 = spec_v2_repeating_init %14 {handshake.bb = 2 : ui32, handshake.name = "rhs_ri"} : <i1>
    %20:2 = fork [2] %19 {handshake.bb = 2 : ui32, handshake.name = "rhs_fork_data_mux"} : <i1>
    %21 = init %20#0 {handshake.bb = 2 : ui32, handshake.name = "rhs_init_buffer_ctrl", hw.parameters = {INITIAL_TOKEN = false, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %22 = mux %21 [%9, %4] {handshake.bb = 2 : ui32, handshake.name = "rhs_data_mux"} : <i1>, [<i1>, <i1>] to <i1>
    %23 = passer %22[%20#1] {handshake.bb = 2 : ui32, handshake.name = "rhs_passer"} : <i1>, <i1>
    %24 = ndsource {handshake.bb = 3 : ui32, handshake.name = "out_nds_Out1"} : <>
    %25:2 = lazy_fork [2] %24 {handshake.bb = 3 : ui32, handshake.name = "out_lf_Out1"} : <>
    %26 = buffer %25#0 {handshake.bb = 3 : ui32, handshake.name = "out_buf_lhs_nds_Out1", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_DV", NUM_SLOTS = 2 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <>
    %27 = buffer %25#1 {handshake.bb = 3 : ui32, handshake.name = "out_buf_rhs_nds_Out1", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_DV", NUM_SLOTS = 2 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <>
    %28 = blocker %18[%26] {handshake.bb = 3 : ui32, handshake.name = "out_lhs_bl_Out1"} : <i1>, <>
    %29 = blocker %23[%27] {handshake.bb = 3 : ui32, handshake.name = "out_rhs_bl_Out1"} : <i1>, <>
    %30 = ndwire %28 {handshake.bb = 3 : ui32, handshake.name = "lhs_out_ndw_Out1"} : <i1>
    %31 = ndwire %29 {handshake.bb = 3 : ui32, handshake.name = "rhs_out_ndw_Out1"} : <i1>
    %32 = buffer %30 {handshake.bb = 3 : ui32, handshake.name = "lhs_out_buf_Out1", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_DV", NUM_SLOTS = 2 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %33 = buffer %31 {handshake.bb = 3 : ui32, handshake.name = "rhs_out_buf_Out1", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_DV", NUM_SLOTS = 2 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %34 = cmpi eq, %32, %33 {handshake.bb = 3 : ui32, handshake.name = "out_eq_Out1"} : <i1>
    end {handshake.bb = 3 : ui32, handshake.name = "end"} %34 : <i1>
  }
}
