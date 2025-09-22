module {
  handshake.func @elastic_miter_general_sup_mumux_lhs_general_sup_mumux_rhs(%arg0: !handshake.channel<i1>, %arg1: !handshake.channel<i1>, %arg2: !handshake.channel<i1>, %arg3: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>, !handshake.channel<i1>) attributes {argNames = ["dataT", "dataF", "cond", "ctrl"], resNames = ["EQ_ctrlRes", "EQ_dataRes"]} {
    %0:2 = fork [2] %arg2 {handshake.bb = 0 : ui32, handshake.name = "ctx_cond_fork"} : <i1>
    %1:2 = fork [2] %arg3 {handshake.bb = 0 : ui32, handshake.name = "ctx_ctrl_fork"} : <i1>
    %2 = passer %0#1[%1#1] {handshake.bb = 0 : ui32, handshake.name = "ctx_passer"} : <i1>, <i1>
    sink %2 {handshake.bb = 0 : ui32, handshake.name = "ctx_sink"} : <i1>
    %3:2 = lazy_fork [2] %arg0 {handshake.bb = 0 : ui32, handshake.name = "in_fork_dataT"} : <i1>
    %4 = buffer %3#0, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 0 : ui32, handshake.name = "lhs_in_buf_dataT"} : <i1>
    %5 = buffer %3#1, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 0 : ui32, handshake.name = "rhs_in_buf_dataT"} : <i1>
    %6:2 = lazy_fork [2] %arg1 {handshake.bb = 0 : ui32, handshake.name = "in_fork_dataF"} : <i1>
    %7 = buffer %6#0, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 0 : ui32, handshake.name = "lhs_in_buf_dataF"} : <i1>
    %8 = buffer %6#1, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 0 : ui32, handshake.name = "rhs_in_buf_dataF"} : <i1>
    %9:2 = lazy_fork [2] %0#0 {handshake.bb = 0 : ui32, handshake.name = "in_fork_cond"} : <i1>
    %10 = buffer %9#0, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 0 : ui32, handshake.name = "lhs_in_buf_cond"} : <i1>
    %11 = buffer %9#1, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 0 : ui32, handshake.name = "rhs_in_buf_cond"} : <i1>
    %12:2 = lazy_fork [2] %1#0 {handshake.bb = 0 : ui32, handshake.name = "in_fork_ctrl"} : <i1>
    %13 = buffer %12#0, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 0 : ui32, handshake.name = "lhs_in_buf_ctrl"} : <i1>
    %14 = buffer %12#1, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = true, handshake.bb = 0 : ui32, handshake.name = "rhs_in_buf_ctrl"} : <i1>
    %15 = source {handshake.bb = 3 : ui32, handshake.name = "out_src_ctrlRes"} : <>
    %16:2 = lazy_fork [2] %15 {handshake.bb = 3 : ui32, handshake.name = "out_lf_ctrlRes"} : <>
    %17 = buffer %16#0, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 3 : ui32, handshake.name = "out_buf_lhs_nds_ctrlRes"} : <>
    %18 = buffer %16#1, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 3 : ui32, handshake.name = "out_buf_rhs_nds_ctrlRes"} : <>
    %19 = blocker %32#1[%17] {handshake.bb = 3 : ui32, handshake.name = "lhs_out_bl_ctrlRes"} : <i1>, <>
    %20 = blocker %55[%18] {handshake.bb = 3 : ui32, handshake.name = "rhs_out_bl_ctrlRes"} : <i1>, <>
    %21 = cmpi eq, %19, %20 {handshake.bb = 3 : ui32, handshake.name = "out_eq_ctrlRes"} : <i1>
    %22 = source {handshake.bb = 3 : ui32, handshake.name = "out_src_dataRes"} : <>
    %23:2 = lazy_fork [2] %22 {handshake.bb = 3 : ui32, handshake.name = "out_lf_dataRes"} : <>
    %24 = buffer %23#0, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 3 : ui32, handshake.name = "out_buf_lhs_nds_dataRes"} : <>
    %25 = buffer %23#1, bufferType = FIFO_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 3 : ui32, handshake.name = "out_buf_rhs_nds_dataRes"} : <>
    %26 = blocker %39[%24] {handshake.bb = 3 : ui32, handshake.name = "lhs_out_bl_dataRes"} : <i1>, <>
    %27 = blocker %56[%25] {handshake.bb = 3 : ui32, handshake.name = "rhs_out_bl_dataRes"} : <i1>, <>
    %28 = cmpi eq, %26, %27 {handshake.bb = 3 : ui32, handshake.name = "out_eq_dataRes"} : <i1>
    %29:2 = fork [2] %13 {handshake.bb = 1 : ui32, handshake.name = "lhs_ctrl_fork"} : <i1>
    %30 = passer %10[%29#0] {handshake.bb = 1 : ui32, handshake.name = "lhs_cond_passer"} : <i1>, <i1>
    %31 = passer %4[%29#1] {handshake.bb = 1 : ui32, handshake.name = "lhs_dataT_passer"} : <i1>, <i1>
    %32:2 = fork [2] %38 {handshake.bb = 1 : ui32, handshake.name = "lhs_ri_fork"} : <i1>
    %33 = buffer %32#0, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "lhs_ri_buff"} : <i1>
    %34 = init %33 {handshake.bb = 1 : ui32, handshake.name = "lhs_initop", initToken = 0 : ui1} : <i1>
    %35:2 = fork [2] %34 {handshake.bb = 1 : ui32, handshake.name = "lhs_init_fork"} : <i1>
    %36 = source {handshake.bb = 1 : ui32, handshake.name = "lhs_source1"} : <>
    %37 = constant %36 {handshake.bb = 1 : ui32, handshake.name = "lhs_constant1", value = true} : <>, <i1>
    %38 = mux %35#0 [%37, %30] {handshake.bb = 1 : ui32, handshake.name = "lhs_ri"} : <i1>, [<i1>, <i1>] to <i1>
    %39 = mux %35#1 [%7, %31] {handshake.bb = 1 : ui32, handshake.name = "lhs_data_mux"} : <i1>, [<i1>, <i1>] to <i1>
    %40:2 = fork [2] %14 {handshake.bb = 2 : ui32, handshake.name = "rhs_ctrl_fork"} : <i1>
    %41 = not %40#0 {handshake.bb = 2 : ui32, handshake.name = "rhs_ctrl_not"} : <i1>
    %42 = ori %41, %11 {handshake.bb = 2 : ui32, handshake.name = "rhs_ori"} : <i1>
    %43:2 = fork [2] %49 {handshake.bb = 2 : ui32, handshake.name = "rhs_ri_fork"} : <i1>
    %44 = buffer %43#0, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "rhs_ri_buff"} : <i1>
    %45 = init %44 {handshake.bb = 2 : ui32, handshake.name = "rhs_initop", initToken = 0 : ui1} : <i1>
    %46:3 = fork [3] %45 {handshake.bb = 2 : ui32, handshake.name = "rhs_init_fork"} : <i1>
    %47 = source {handshake.bb = 2 : ui32, handshake.name = "rhs_source1"} : <>
    %48 = constant %47 {handshake.bb = 2 : ui32, handshake.name = "rhs_constant1", value = true} : <>, <i1>
    %49 = mux %46#0 [%48, %42] {handshake.bb = 2 : ui32, handshake.name = "rhs_ri"} : <i1>, [<i1>, <i1>] to <i1>
    %50 = mux %46#1 [%8, %5] {handshake.bb = 2 : ui32, handshake.name = "rhs_data_mux"} : <i1>, [<i1>, <i1>] to <i1>
    %51 = source {handshake.bb = 2 : ui32, handshake.name = "rhs_source2"} : <>
    %52 = constant %51 {handshake.bb = 2 : ui32, handshake.name = "rhs_constant2", value = true} : <>, <i1>
    %53 = mux %46#2 [%52, %40#1] {handshake.bb = 2 : ui32, handshake.name = "rhs_ctrl_mux"} : <i1>, [<i1>, <i1>] to <i1>
    %54:2 = fork [2] %53 {handshake.bb = 2 : ui32, handshake.name = "rhs_ctrl_muxed_fork"} : <i1>
    %55 = passer %43#1[%54#0] {handshake.bb = 2 : ui32, handshake.name = "rhs_ctrlRes_passer"} : <i1>, <i1>
    %56 = passer %50[%54#1] {handshake.bb = 2 : ui32, handshake.name = "rhs_dataRes_passer"} : <i1>, <i1>
    end {handshake.bb = 3 : ui32, handshake.name = "end"} %21, %28 : <i1>, <i1>
  }
}
