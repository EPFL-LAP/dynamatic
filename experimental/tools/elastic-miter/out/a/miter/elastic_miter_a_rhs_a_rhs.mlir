module {
  handshake.func @elastic_miter_a_lhs_a_rhs(%arg0: !handshake.channel<i32>, %arg1: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>, !handshake.channel<i1>) attributes {argNames = ["D", "C"], resNames = ["EQ_T", "EQ_F"]} {
    %0:2 = lazy_fork [2] %arg0 {handshake.bb = 0 : ui32, handshake.name = "in_fork_D"} : <i32>
    %1 = buffer %0#0 {handshake.bb = 0 : ui32, handshake.name = "lhs_in_buf_D", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i32>
    %2 = buffer %0#1 {handshake.bb = 0 : ui32, handshake.name = "rhs_in_buf_D", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i32>
    %3 = ndwire %1 {handshake.bb = 0 : ui32, handshake.name = "lhs_in_ndw_D"} : <i32>
    %4 = ndwire %2 {handshake.bb = 0 : ui32, handshake.name = "rhs_in_ndw_D"} : <i32>
    %5:2 = lazy_fork [2] %arg1 {handshake.bb = 0 : ui32, handshake.name = "in_fork_C"} : <i1>
    %6 = buffer %5#0 {handshake.bb = 0 : ui32, handshake.name = "lhs_in_buf_C", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %7 = buffer %5#1 {handshake.bb = 0 : ui32, handshake.name = "rhs_in_buf_C", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %8 = ndwire %6 {handshake.bb = 0 : ui32, handshake.name = "lhs_in_ndw_C"} : <i1>
    %9 = ndwire %7 {handshake.bb = 0 : ui32, handshake.name = "rhs_in_ndw_C"} : <i1>
    %trueResult, %falseResult = cond_br %8, %3 {handshake.bb = 1 : ui32, handshake.name = "lhs_branch"} : <i1>, <i32>
    %10:2 = fork [2] %4 {handshake.bb = 2 : ui32, handshake.name = "rhs_fork_data"} : <i32>
    %11:2 = fork [2] %9 {handshake.bb = 2 : ui32, handshake.name = "rhs_fork_control"} : <i1>
    %12 = not %11#0 {handshake.bb = 2 : ui32, handshake.name = "rhs_not"} : <i1>
    %trueResult_0, %falseResult_1 = cond_br %12, %10#0 {handshake.bb = 2 : ui32, handshake.name = "rhs_supp_br_T"} : <i1>, <i32>
    sink %trueResult_0 {handshake.bb = 2 : ui32, handshake.name = "rhs_supp_sink_0"} : <i32>
    %trueResult_2, %falseResult_3 = cond_br %11#1, %10#1 {handshake.bb = 2 : ui32, handshake.name = "rhs_supp_br_F"} : <i1>, <i32>
    sink %trueResult_2 {handshake.bb = 2 : ui32, handshake.name = "rhs_supp_sink_1"} : <i32>
    %13 = ndwire %trueResult {handshake.bb = 3 : ui32, handshake.name = "lhs_out_ndw_T"} : <i32>
    %14 = ndwire %falseResult_1 {handshake.bb = 3 : ui32, handshake.name = "rhs_out_ndw_T"} : <i32>
    %15 = buffer %13 {handshake.bb = 3 : ui32, handshake.name = "lhs_out_buf_T", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i32>
    %16 = buffer %14 {handshake.bb = 3 : ui32, handshake.name = "rhs_out_buf_T", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i32>
    %17 = cmpi eq, %15, %16 {handshake.bb = 3 : ui32, handshake.name = "out_eq_T"} : <i32>
    %18 = ndwire %falseResult {handshake.bb = 3 : ui32, handshake.name = "lhs_out_ndw_F"} : <i32>
    %19 = ndwire %falseResult_3 {handshake.bb = 3 : ui32, handshake.name = "rhs_out_ndw_F"} : <i32>
    %20 = buffer %18 {handshake.bb = 3 : ui32, handshake.name = "lhs_out_buf_F", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i32>
    %21 = buffer %19 {handshake.bb = 3 : ui32, handshake.name = "rhs_out_buf_F", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i32>
    %22 = cmpi eq, %20, %21 {handshake.bb = 3 : ui32, handshake.name = "out_eq_F"} : <i32>
    end {handshake.bb = 3 : ui32, handshake.name = "end"} %17, %22 : <i1>, <i1>
  }
}
