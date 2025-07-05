module {
  handshake.func @elastic_miter_sup_fork_lhs_sup_fork_rhs(%arg0: !handshake.channel<i1>, %arg1: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>, !handshake.channel<i1>) attributes {argNames = ["A_in", "Cond"], resNames = ["EQ_B_out", "EQ_C_out"]} {
    %0:2 = lazy_fork [2] %arg0 {handshake.bb = 0 : ui32, handshake.name = "in_fork_A_in"} : <i1>
    %1 = buffer %0#0 {handshake.bb = 0 : ui32, handshake.name = "lhs_in_buf_A_in", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %2 = buffer %0#1 {handshake.bb = 0 : ui32, handshake.name = "rhs_in_buf_A_in", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %3 = ndwire %1 {handshake.bb = 0 : ui32, handshake.name = "lhs_in_ndw_A_in"} : <i1>
    %4 = ndwire %2 {handshake.bb = 0 : ui32, handshake.name = "rhs_in_ndw_A_in"} : <i1>
    %5:2 = lazy_fork [2] %arg1 {handshake.bb = 0 : ui32, handshake.name = "in_fork_Cond"} : <i1>
    %6 = buffer %5#0 {handshake.bb = 0 : ui32, handshake.name = "lhs_in_buf_Cond", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %7 = buffer %5#1 {handshake.bb = 0 : ui32, handshake.name = "rhs_in_buf_Cond", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %8 = ndwire %6 {handshake.bb = 0 : ui32, handshake.name = "lhs_in_ndw_Cond"} : <i1>
    %9 = ndwire %7 {handshake.bb = 0 : ui32, handshake.name = "rhs_in_ndw_Cond"} : <i1>
    %10 = passer %3[%8] {handshake.bb = 1 : ui32, handshake.name = "lhs_passer1"} : <i1>, <i1>
    %11:2 = fork [2] %10 {handshake.bb = 1 : ui32, handshake.name = "lhs_fork"} : <i1>
    %12:2 = fork [2] %4 {handshake.bb = 2 : ui32, handshake.name = "rhs_fork"} : <i1>
    %13:2 = fork [2] %9 {handshake.bb = 2 : ui32, handshake.name = "rhs_fork_cond"} : <i1>
    %14 = passer %12#0[%13#0] {handshake.bb = 2 : ui32, handshake.name = "rhs_passer1"} : <i1>, <i1>
    %15 = passer %12#1[%13#1] {handshake.bb = 2 : ui32, handshake.name = "rhs_passer2"} : <i1>, <i1>
    %16 = ndsource {handshake.bb = 3 : ui32, handshake.name = "out_nds_B_out"} : <i1>
    %17:2 = lazy_fork [2] %16 {handshake.bb = 3 : ui32, handshake.name = "out_lf_B_out"} : <i1>
    %18 = buffer %17#0 {handshake.bb = 3 : ui32, handshake.name = "out_buf_lhs_nds_B_out", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %19 = buffer %17#1 {handshake.bb = 3 : ui32, handshake.name = "out_buf_rhs_nds_B_out", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %20 = transfer_control %11#0[%18] {handshake.bb = 3 : ui32, handshake.name = "out_lhs_tc_B_out"} : <i1>, <i1>
    %21 = transfer_control %14[%19] {handshake.bb = 3 : ui32, handshake.name = "out_rhs_tc_B_out"} : <i1>, <i1>
    %22 = ndwire %20 {handshake.bb = 3 : ui32, handshake.name = "lhs_out_ndw_B_out"} : <i1>
    %23 = ndwire %21 {handshake.bb = 3 : ui32, handshake.name = "rhs_out_ndw_B_out"} : <i1>
    %24 = buffer %22 {handshake.bb = 3 : ui32, handshake.name = "lhs_out_buf_B_out", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %25 = buffer %23 {handshake.bb = 3 : ui32, handshake.name = "rhs_out_buf_B_out", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %26 = cmpi eq, %24, %25 {handshake.bb = 3 : ui32, handshake.name = "out_eq_B_out"} : <i1>
    %27 = ndsource {handshake.bb = 3 : ui32, handshake.name = "out_nds_C_out"} : <i1>
    %28:2 = lazy_fork [2] %27 {handshake.bb = 3 : ui32, handshake.name = "out_lf_C_out"} : <i1>
    %29 = buffer %28#0 {handshake.bb = 3 : ui32, handshake.name = "out_buf_lhs_nds_C_out", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %30 = buffer %28#1 {handshake.bb = 3 : ui32, handshake.name = "out_buf_rhs_nds_C_out", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %31 = transfer_control %11#1[%29] {handshake.bb = 3 : ui32, handshake.name = "out_lhs_tc_C_out"} : <i1>, <i1>
    %32 = transfer_control %15[%30] {handshake.bb = 3 : ui32, handshake.name = "out_rhs_tc_C_out"} : <i1>, <i1>
    %33 = ndwire %31 {handshake.bb = 3 : ui32, handshake.name = "lhs_out_ndw_C_out"} : <i1>
    %34 = ndwire %32 {handshake.bb = 3 : ui32, handshake.name = "rhs_out_ndw_C_out"} : <i1>
    %35 = buffer %33 {handshake.bb = 3 : ui32, handshake.name = "lhs_out_buf_C_out", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %36 = buffer %34 {handshake.bb = 3 : ui32, handshake.name = "rhs_out_buf_C_out", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %37 = cmpi eq, %35, %36 {handshake.bb = 3 : ui32, handshake.name = "out_eq_C_out"} : <i1>
    end {handshake.bb = 3 : ui32, handshake.name = "end"} %26, %37 : <i1>, <i1>
  }
}
