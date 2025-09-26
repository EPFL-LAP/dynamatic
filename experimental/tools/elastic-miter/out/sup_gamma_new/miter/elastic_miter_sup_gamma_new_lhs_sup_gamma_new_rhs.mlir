module {
  handshake.func @elastic_miter_sup_gamma_new_lhs_sup_gamma_new_rhs(%arg0: !handshake.channel<i1>, %arg1: !handshake.channel<i1>, %arg2: !handshake.channel<i1>, %arg3: !handshake.channel<i1>, ...) -> !handshake.channel<i1> attributes {argNames = ["a1", "a2", "c1", "c2"], resNames = ["EQ_b"]} {
    %0:2 = lazy_fork [2] %arg0 {handshake.bb = 0 : ui32, handshake.name = "in_fork_a1"} : <i1>
    %1 = buffer %0#0, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 0 : ui32, handshake.name = "lhs_in_buf_a1"} : <i1>
    %2 = buffer %0#1, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 0 : ui32, handshake.name = "rhs_in_buf_a1"} : <i1>
    %3:2 = lazy_fork [2] %arg1 {handshake.bb = 0 : ui32, handshake.name = "in_fork_a2"} : <i1>
    %4 = buffer %3#0, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 0 : ui32, handshake.name = "lhs_in_buf_a2"} : <i1>
    %5 = buffer %3#1, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 0 : ui32, handshake.name = "rhs_in_buf_a2"} : <i1>
    %6:2 = lazy_fork [2] %arg2 {handshake.bb = 0 : ui32, handshake.name = "in_fork_c1"} : <i1>
    %7 = buffer %6#0, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 0 : ui32, handshake.name = "lhs_in_buf_c1"} : <i1>
    %8 = buffer %6#1, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 0 : ui32, handshake.name = "rhs_in_buf_c1"} : <i1>
    %9:2 = lazy_fork [2] %arg3 {handshake.bb = 0 : ui32, handshake.name = "in_fork_c2"} : <i1>
    %10 = buffer %9#0, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 0 : ui32, handshake.name = "lhs_in_buf_c2"} : <i1>
    %11 = buffer %9#1, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 0 : ui32, handshake.name = "rhs_in_buf_c2"} : <i1>
    %12 = ndsource {handshake.bb = 3 : ui32, handshake.name = "out_nds_b"} : <>
    %13:2 = lazy_fork [2] %12 {handshake.bb = 3 : ui32, handshake.name = "out_lf_b"} : <>
    %14 = buffer %13#0, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 3 : ui32, handshake.name = "out_buf_lhs_nds_b"} : <>
    %15 = buffer %13#1, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 3 : ui32, handshake.name = "out_buf_rhs_nds_b"} : <>
    %16 = blocker %29[%14] {handshake.bb = 3 : ui32, handshake.name = "lhs_out_bl_b"} : <i1>, <>
    %17 = blocker %35[%15] {handshake.bb = 3 : ui32, handshake.name = "rhs_out_bl_b"} : <i1>, <>
    %18 = cmpi eq, %16, %17 {handshake.bb = 3 : ui32, handshake.name = "out_eq_b"} : <i1>
    %19:5 = fork [5] %7 {handshake.bb = 1 : ui32, handshake.name = "lhs_c1_fork"} : <i1>
    %20:3 = fork [3] %10 {handshake.bb = 1 : ui32, handshake.name = "lhs_c2_fork"} : <i1>
    %21 = passer %20#0[%19#0] {handshake.bb = 1 : ui32, handshake.name = "lhs_c21_1"} : <i1>, <i1>
    %22 = passer %20#1[%19#1] {handshake.bb = 1 : ui32, handshake.name = "lhs_c21_2"} : <i1>, <i1>
    %23 = not %20#2 {handshake.bb = 1 : ui32, handshake.name = "lhs_not"} : <i1>
    %24 = passer %23[%19#2] {handshake.bb = 1 : ui32, handshake.name = "lhs_c2inv1"} : <i1>, <i1>
    %25 = passer %1[%19#3] {handshake.bb = 1 : ui32, handshake.name = "lhs_a1_passer1"} : <i1>, <i1>
    %26 = passer %25[%21] {handshake.bb = 1 : ui32, handshake.name = "lhs_a1_passer2"} : <i1>, <i1>
    %27 = passer %4[%19#4] {handshake.bb = 1 : ui32, handshake.name = "lhs_a2_passer1"} : <i1>, <i1>
    %28 = passer %27[%24] {handshake.bb = 1 : ui32, handshake.name = "lhs_a2_passer2"} : <i1>, <i1>
    %29 = mux %22 [%28, %26] {handshake.bb = 1 : ui32, handshake.name = "lhs_mux"} : <i1>, [<i1>, <i1>] to <i1>
    %30:3 = fork [3] %11 {handshake.bb = 2 : ui32, handshake.name = "rhs_c2_fork"} : <i1>
    %31 = passer %2[%30#0] {handshake.bb = 2 : ui32, handshake.name = "rhs_a1_passer"} : <i1>, <i1>
    %32 = not %30#1 {handshake.bb = 2 : ui32, handshake.name = "rhs_not"} : <i1>
    %33 = passer %5[%32] {handshake.bb = 2 : ui32, handshake.name = "rhs_a2_passer"} : <i1>, <i1>
    %34 = mux %30#2 [%33, %31] {handshake.bb = 2 : ui32, handshake.name = "rhs_mux"} : <i1>, [<i1>, <i1>] to <i1>
    %35 = passer %34[%8] {handshake.bb = 2 : ui32, handshake.name = "rhs_b_passer"} : <i1>, <i1>
    end {handshake.bb = 3 : ui32, handshake.name = "end"} %18 : <i1>
  }
}
