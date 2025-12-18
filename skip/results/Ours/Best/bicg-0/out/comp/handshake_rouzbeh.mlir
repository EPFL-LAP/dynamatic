module {
  handshake.func @bicg(%arg0: memref<900xi32>, %arg1: memref<30xi32>, %arg2: memref<30xi32>, %arg3: memref<30xi32>, %arg4: memref<30xi32>, %arg5: !handshake.control<>, %arg6: !handshake.control<>, %arg7: !handshake.control<>, %arg8: !handshake.control<>, %arg9: !handshake.control<>, %arg10: !handshake.control<>, ...) -> (!handshake.channel<i32>, !handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["a", "s", "q", "p", "r", "a_start", "s_start", "q_start", "p_start", "r_start", "start"], cfg.edges = "[0,1][2,2,3,cmpi0][1,2][3,1,4,cmpi1]", resNames = ["out0", "a_end", "s_end", "q_end", "p_end", "r_end", "end"]} {
    %outputs, %memEnd = mem_controller[%arg4 : memref<30xi32>] %arg9 (%addressResult_14) %result_41 {connectedBlocks = [2 : i32], handshake.name = "mem_controller3"} :    (!handshake.channel<i32>) -> !handshake.channel<i32>
    %outputs_0, %memEnd_1 = mem_controller[%arg3 : memref<30xi32>] %arg8 (%addressResult_18) %result_41 {connectedBlocks = [2 : i32], handshake.name = "mem_controller4"} :    (!handshake.channel<i32>) -> !handshake.channel<i32>
    %outputs_2:2, %memEnd_3 = mem_controller[%arg2 : memref<30xi32>] %arg7 (%addressResult, %35, %addressResult_32, %dataResult_33) %result_41 {connectedBlocks = [1 : i32, 3 : i32], handshake.name = "mem_controller5"} :    (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>) -> (!handshake.channel<i32>, !handshake.control<>)
    %outputs_4:2, %memEnd_5 = mem_controller[%arg1 : memref<30xi32>] %arg6 (%18, %addressResult_12, %addressResult_16, %dataResult_17) %result_41 {connectedBlocks = [2 : i32], handshake.name = "mem_controller6"} :    (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>) -> (!handshake.channel<i32>, !handshake.control<>)
    %outputs_6, %memEnd_7 = mem_controller[%arg0 : memref<900xi32>] %arg5 (%addressResult_10) %result_41 {connectedBlocks = [2 : i32], handshake.name = "mem_controller7"} :    (!handshake.channel<i32>) -> !handshake.channel<i32>
    %0 = constant %arg10 {handshake.bb = 0 : ui32, handshake.name = "constant2", value = 0 : i32} : <>, <i32>
    %1 = br %0 {handshake.bb = 0 : ui32, handshake.name = "br2"} : <i32>
    %2 = br %arg10 {handshake.bb = 0 : ui32, handshake.name = "br3"} : <>
    %3 = mux %4 [%arg10, %trueResult_28] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux4"} : <i1>, [<>, <>] to <>
    %4 = init %42 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "init0"} : <i1>
    %5:2 = unbundle %dataResult  {handshake.bb = 1 : ui32, handshake.name = "unbundle0"} : <i32> to _ 
    %6 = mux %index [%1, %trueResult_35] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i32>, <i32>] to <i32>
    %result, %index = control_merge [%2, %trueResult_37]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %7 = constant %result {handshake.bb = 1 : ui32, handshake.name = "constant3", value = 0 : i32} : <>, <i32>
    %8 = buffer %5#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer0"} : <>
    %addressResult, %dataResult = load[%6] %outputs_2#0 {handshake.bb = 1 : ui32, handshake.deps = #handshake<deps[["store1", 2, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i32>, <i32>, <i32>, <i32>
    %9 = br %7 {handshake.bb = 1 : ui32, handshake.name = "br4"} : <i32>
    %10 = br %dataResult {handshake.bb = 1 : ui32, handshake.name = "br5"} : <i32>
    %11 = br %6 {handshake.bb = 1 : ui32, handshake.name = "br6"} : <i32>
    %12 = br %result {handshake.bb = 1 : ui32, handshake.name = "br7"} : <>
    %trueResult, %falseResult = cond_br %32, %28 {handshake.bb = 2 : ui32, handshake.name = "cond_br11"} : <i1>, <>
    %13 = mux %14 [%3, %trueResult] {ftd.phi, handshake.bb = 2 : ui32, handshake.name = "mux6"} : <i1>, [<>, <>] to <>
    %14 = init %32 {ftd.imerge, handshake.bb = 2 : ui32, handshake.name = "init2"} : <i1>
    %15 = mux %index_9 [%9, %trueResult_20] {handshake.bb = 2 : ui32, handshake.name = "mux1"} : <i1>, [<i32>, <i32>] to <i32>
    %16 = mux %index_9 [%10, %trueResult_22] {handshake.bb = 2 : ui32, handshake.name = "mux2"} : <i1>, [<i32>, <i32>] to <i32>
    %17 = mux %index_9 [%11, %trueResult_24] {handshake.bb = 2 : ui32, handshake.name = "mux3"} : <i1>, [<i32>, <i32>] to <i32>
    %result_8, %index_9 = control_merge [%12, %trueResult_26]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>, <>] to <>, <i1>
    %18 = constant %result_8 {handshake.bb = 2 : ui32, handshake.name = "constant0", value = 1 : i32} : <>, <i32>
    %19 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %20 = constant %19 {handshake.bb = 2 : ui32, handshake.name = "constant4", value = 30 : i32} : <>, <i32>
    %21 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %22 = constant %21 {handshake.bb = 2 : ui32, handshake.name = "constant5", value = 1 : i32} : <>, <i32>
    %23 = muli %17, %20 {handshake.bb = 2 : ui32, handshake.name = "muli2"} : <i32>
    %24 = addi %15, %23 {handshake.bb = 2 : ui32, handshake.name = "addi2"} : <i32>
    %addressResult_10, %dataResult_11 = load[%24] %outputs_6 {handshake.bb = 2 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i32>, <i32>, <i32>, <i32>
    %25 = gate %15, %13 {handshake.bb = 2 : ui32, handshake.name = "gate0"} : <i32>, !handshake.control<> to <i32>
    %addressResult_12, %dataResult_13 = load[%25] %outputs_4#0 {handshake.bb = 2 : ui32, handshake.deps = #handshake<deps[["store0", 1, false], ["store0", 3, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load2"} : <i32>, <i32>, <i32>, <i32>
    %addressResult_14, %dataResult_15 = load[%17] %outputs {handshake.bb = 2 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load3"} : <i32>, <i32>, <i32>, <i32>
    %26 = muli %dataResult_15, %dataResult_11 {handshake.bb = 2 : ui32, handshake.name = "muli0"} : <i32>
    %27 = addi %dataResult_13, %26 {handshake.bb = 2 : ui32, handshake.name = "addi0"} : <i32>
    %28 = buffer %doneResult, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer1"} : <>
    %addressResult_16, %dataResult_17, %doneResult = store[%15] %27 %outputs_4#1 {handshake.bb = 2 : ui32, handshake.deps = #handshake<deps[["load2", 1, false], ["store0", 1, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store0"} : <i32>, <i32>, <>, <i32>, <i32>, <>
    %addressResult_18, %dataResult_19 = load[%15] %outputs_0 {handshake.bb = 2 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load4"} : <i32>, <i32>, <i32>, <i32>
    %29 = muli %dataResult_11, %dataResult_19 {handshake.bb = 2 : ui32, handshake.name = "muli1"} : <i32>
    %30 = addi %16, %29 {handshake.bb = 2 : ui32, handshake.name = "addi1"} : <i32>
    %31 = addi %15, %22 {handshake.bb = 2 : ui32, handshake.name = "addi3"} : <i32>
    %32 = cmpi ult, %31, %20 {handshake.bb = 2 : ui32, handshake.name = "cmpi0"} : <i32>
    %trueResult_20, %falseResult_21 = cond_br %32, %31 {handshake.bb = 2 : ui32, handshake.name = "cond_br2"} : <i1>, <i32>
    %trueResult_22, %falseResult_23 = cond_br %32, %30 {handshake.bb = 2 : ui32, handshake.name = "cond_br3"} : <i1>, <i32>
    %trueResult_24, %falseResult_25 = cond_br %32, %17 {handshake.bb = 2 : ui32, handshake.name = "cond_br4"} : <i1>, <i32>
    %trueResult_26, %falseResult_27 = cond_br %32, %result_8 {handshake.bb = 2 : ui32, handshake.name = "cond_br5"} : <i1>, <>
    %trueResult_28, %falseResult_29 = cond_br %42, %falseResult {handshake.bb = 3 : ui32, handshake.name = "cond_br12"} : <i1>, <>
    %33 = merge %falseResult_25 {handshake.bb = 3 : ui32, handshake.name = "merge0"} : <i32>
    %34 = merge %falseResult_23 {handshake.bb = 3 : ui32, handshake.name = "merge1"} : <i32>
    %result_30, %index_31 = control_merge [%falseResult_27]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>] to <>, <i1>
    %35 = constant %result_30 {handshake.bb = 3 : ui32, handshake.name = "constant1", value = 1 : i32} : <>, <i32>
    %36 = source {handshake.bb = 3 : ui32, handshake.name = "source2"} : <>
    %37 = constant %36 {handshake.bb = 3 : ui32, handshake.name = "constant6", value = 30 : i32} : <>, <i32>
    %38 = source {handshake.bb = 3 : ui32, handshake.name = "source3"} : <>
    %39 = constant %38 {handshake.bb = 3 : ui32, handshake.name = "constant7", value = 1 : i32} : <>, <i32>
    %40 = gate %33, %8 {handshake.bb = 3 : ui32, handshake.name = "gate1"} : <i32>, !handshake.control<> to <i32>
    %addressResult_32, %dataResult_33, %doneResult_34 = store[%40] %34 %outputs_2#1 {handshake.bb = 3 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store1"} : <i32>, <i32>, <>, <i32>, <i32>, <>
    %41 = addi %33, %39 {handshake.bb = 3 : ui32, handshake.name = "addi4"} : <i32>
    %42 = cmpi ult, %41, %37 {handshake.bb = 3 : ui32, handshake.name = "cmpi1"} : <i32>
    %trueResult_35, %falseResult_36 = cond_br %42, %41 {handshake.bb = 3 : ui32, handshake.name = "cond_br6"} : <i1>, <i32>
    %trueResult_37, %falseResult_38 = cond_br %42, %result_30 {handshake.bb = 3 : ui32, handshake.name = "cond_br7"} : <i1>, <>
    %trueResult_39, %falseResult_40 = cond_br %42, %34 {handshake.bb = 3 : ui32, handshake.name = "cond_br8"} : <i1>, <i32>
    %43 = merge %falseResult_40 {handshake.bb = 4 : ui32, handshake.name = "merge2"} : <i32>
    %result_41, %index_42 = control_merge [%falseResult_38]  {handshake.bb = 4 : ui32, handshake.name = "control_merge3"} : [<>] to <>, <i1>
    end {handshake.bb = 4 : ui32, handshake.name = "end0"} %43, %memEnd_7, %memEnd_5, %memEnd_3, %memEnd_1, %memEnd, %arg10 : <i32>, <>, <>, <>, <>, <>, <>
  }
}

