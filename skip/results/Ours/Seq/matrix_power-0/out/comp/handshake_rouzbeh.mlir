module {
  handshake.func @matrix_power(%arg0: memref<400xi32>, %arg1: memref<20xi32>, %arg2: memref<20xi32>, %arg3: memref<20xi32>, %arg4: !handshake.control<>, %arg5: !handshake.control<>, %arg6: !handshake.control<>, %arg7: !handshake.control<>, %arg8: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["mat", "row", "col", "a", "mat_start", "row_start", "col_start", "a_start", "start"], cfg.edges = "[0,1][2,2,3,cmpi0][1,2][3,1,4,cmpi1]", resNames = ["mat_end", "row_end", "col_end", "a_end", "end"]} {
    %outputs, %memEnd = mem_controller[%arg3 : memref<20xi32>] %arg7 (%addressResult_8) %result_34 {connectedBlocks = [2 : i32], handshake.name = "mem_controller3"} :    (!handshake.channel<i32>) -> !handshake.channel<i32>
    %outputs_0, %memEnd_1 = mem_controller[%arg2 : memref<20xi32>] %arg6 (%addressResult_10) %result_34 {connectedBlocks = [2 : i32], handshake.name = "mem_controller4"} :    (!handshake.channel<i32>) -> !handshake.channel<i32>
    %outputs_2, %memEnd_3 = mem_controller[%arg1 : memref<20xi32>] %arg5 (%addressResult) %result_34 {connectedBlocks = [2 : i32], handshake.name = "mem_controller5"} :    (!handshake.channel<i32>) -> !handshake.channel<i32>
    %outputs_4:3, %memEnd_5 = mem_controller[%arg0 : memref<400xi32>] %arg4 (%21, %addressResult_12, %addressResult_14, %addressResult_16, %dataResult_17) %result_34 {connectedBlocks = [2 : i32], handshake.name = "mem_controller6"} :    (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>) -> (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>)
    %0 = constant %arg8 {handshake.bb = 0 : ui32, handshake.name = "constant3", value = 1 : i32} : <>, <i32>
    %1 = br %0 {handshake.bb = 0 : ui32, handshake.name = "br2"} : <i32>
    %2 = br %arg8 {handshake.bb = 0 : ui32, handshake.name = "br3"} : <>
    %3 = mux %5 [%arg8, %trueResult_26] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux4"} : <i1>, [<>, <>] to <>
    %4 = mux %5 [%arg8, %trueResult_26] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux5"} : <i1>, [<>, <>] to <>
    %5 = init %55 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "init0"} : <i1>
    %6 = mux %index [%1, %trueResult_30] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i32>, <i32>] to <i32>
    %result, %index = control_merge [%2, %trueResult_32]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %7 = constant %result {handshake.bb = 1 : ui32, handshake.name = "constant6", value = 0 : i32} : <>, <i32>
    %8 = source {handshake.bb = 1 : ui32, handshake.name = "source0"} : <>
    %9 = constant %8 {handshake.bb = 1 : ui32, handshake.name = "constant7", value = -1 : i32} : <>, <i32>
    %10 = addi %6, %9 {handshake.bb = 1 : ui32, handshake.name = "addi0"} : <i32>
    %11 = br %7 {handshake.bb = 1 : ui32, handshake.name = "br4"} : <i32>
    %12 = br %6 {handshake.bb = 1 : ui32, handshake.name = "br5"} : <i32>
    %13 = br %10 {handshake.bb = 1 : ui32, handshake.name = "br6"} : <i32>
    %14 = br %result {handshake.bb = 1 : ui32, handshake.name = "br7"} : <>
    %trueResult, %falseResult = cond_br %48, %46 {handshake.bb = 2 : ui32, handshake.name = "cond_br12"} : <i1>, <>
    %15 = mux %17 [%3, %trueResult] {ftd.phi, handshake.bb = 2 : ui32, handshake.name = "mux6"} : <i1>, [<>, <>] to <>
    %16 = mux %17 [%4, %trueResult] {ftd.phi, handshake.bb = 2 : ui32, handshake.name = "mux7"} : <i1>, [<>, <>] to <>
    %17 = init %48 {ftd.imerge, handshake.bb = 2 : ui32, handshake.name = "init2"} : <i1>
    %18 = mux %index_7 [%11, %trueResult_18] {handshake.bb = 2 : ui32, handshake.name = "mux1"} : <i1>, [<i32>, <i32>] to <i32>
    %19 = mux %index_7 [%12, %trueResult_20] {handshake.bb = 2 : ui32, handshake.name = "mux2"} : <i1>, [<i32>, <i32>] to <i32>
    %20 = mux %index_7 [%13, %trueResult_22] {handshake.bb = 2 : ui32, handshake.name = "mux3"} : <i1>, [<i32>, <i32>] to <i32>
    %result_6, %index_7 = control_merge [%14, %trueResult_24]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>, <>] to <>, <i1>
    %21 = constant %result_6 {handshake.bb = 2 : ui32, handshake.name = "constant0", value = 1 : i32} : <>, <i32>
    %22 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %23 = constant %22 {handshake.bb = 2 : ui32, handshake.name = "constant8", value = 1 : i32} : <>, <i32>
    %24 = source {handshake.bb = 2 : ui32, handshake.name = "source2"} : <>
    %25 = constant %24 {handshake.bb = 2 : ui32, handshake.name = "constant9", value = 20 : i32} : <>, <i32>
    %26 = source {handshake.bb = 2 : ui32, handshake.name = "source3"} : <>
    %27 = constant %26 {handshake.bb = 2 : ui32, handshake.name = "constant10", value = 4 : i32} : <>, <i32>
    %28 = source {handshake.bb = 2 : ui32, handshake.name = "source4"} : <>
    %29 = constant %28 {handshake.bb = 2 : ui32, handshake.name = "constant11", value = 2 : i32} : <>, <i32>
    %addressResult, %dataResult = load[%18] %outputs_2 {handshake.bb = 2 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i32>, <i32>, <i32>, <i32>
    %addressResult_8, %dataResult_9 = load[%18] %outputs {handshake.bb = 2 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i32>, <i32>, <i32>, <i32>
    %addressResult_10, %dataResult_11 = load[%18] %outputs_0 {handshake.bb = 2 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load2"} : <i32>, <i32>, <i32>, <i32>
    %30 = shli %20, %29 {handshake.bb = 2 : ui32, handshake.name = "shli0"} : <i32>
    %31 = shli %20, %27 {handshake.bb = 2 : ui32, handshake.name = "shli1"} : <i32>
    %32 = addi %30, %31 {handshake.bb = 2 : ui32, handshake.name = "addi7"} : <i32>
    %33 = addi %dataResult_11, %32 {handshake.bb = 2 : ui32, handshake.name = "addi2"} : <i32>
    %34 = gate %33, %15 {handshake.bb = 2 : ui32, handshake.name = "gate0"} : <i32>, !handshake.control<> to <i32>
    %addressResult_12, %dataResult_13 = load[%34] %outputs_4#0 {handshake.bb = 2 : ui32, handshake.deps = #handshake<deps[["store0", 0, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load3"} : <i32>, <i32>, <i32>, <i32>
    %35 = muli %dataResult_9, %dataResult_13 {handshake.bb = 2 : ui32, handshake.name = "muli0"} : <i32>
    %36 = shli %19, %29 {handshake.bb = 2 : ui32, handshake.name = "shli2"} : <i32>
    %37 = shli %19, %27 {handshake.bb = 2 : ui32, handshake.name = "shli3"} : <i32>
    %38 = addi %36, %37 {handshake.bb = 2 : ui32, handshake.name = "addi8"} : <i32>
    %39 = addi %dataResult, %38 {handshake.bb = 2 : ui32, handshake.name = "addi3"} : <i32>
    %40 = gate %39, %16 {handshake.bb = 2 : ui32, handshake.name = "gate1"} : <i32>, !handshake.control<> to <i32>
    %addressResult_14, %dataResult_15 = load[%40] %outputs_4#1 {handshake.bb = 2 : ui32, handshake.deps = #handshake<deps[["store0", 0, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load4"} : <i32>, <i32>, <i32>, <i32>
    %41 = addi %dataResult_15, %35 {handshake.bb = 2 : ui32, handshake.name = "addi1"} : <i32>
    %42 = shli %19, %29 {handshake.bb = 2 : ui32, handshake.name = "shli4"} : <i32>
    %43 = shli %19, %27 {handshake.bb = 2 : ui32, handshake.name = "shli5"} : <i32>
    %44 = addi %42, %43 {handshake.bb = 2 : ui32, handshake.name = "addi9"} : <i32>
    %45 = addi %dataResult, %44 {handshake.bb = 2 : ui32, handshake.name = "addi4"} : <i32>
    %46 = buffer %doneResult, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer0"} : <>
    %addressResult_16, %dataResult_17, %doneResult = store[%45] %41 %outputs_4#2 {handshake.bb = 2 : ui32, handshake.deps = #handshake<deps[["load3", 0, false], ["load4", 0, false], ["store0", 0, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store0"} : <i32>, <i32>, <>, <i32>, <i32>, <>
    %47 = addi %18, %23 {handshake.bb = 2 : ui32, handshake.name = "addi5"} : <i32>
    %48 = cmpi ult, %47, %25 {handshake.bb = 2 : ui32, handshake.name = "cmpi0"} : <i32>
    %trueResult_18, %falseResult_19 = cond_br %48, %47 {handshake.bb = 2 : ui32, handshake.name = "cond_br2"} : <i1>, <i32>
    %trueResult_20, %falseResult_21 = cond_br %48, %19 {handshake.bb = 2 : ui32, handshake.name = "cond_br3"} : <i1>, <i32>
    %trueResult_22, %falseResult_23 = cond_br %48, %20 {handshake.bb = 2 : ui32, handshake.name = "cond_br4"} : <i1>, <i32>
    %trueResult_24, %falseResult_25 = cond_br %48, %result_6 {handshake.bb = 2 : ui32, handshake.name = "cond_br5"} : <i1>, <>
    %trueResult_26, %falseResult_27 = cond_br %55, %falseResult {handshake.bb = 3 : ui32, handshake.name = "cond_br13"} : <i1>, <>
    %49 = merge %falseResult_21 {handshake.bb = 3 : ui32, handshake.name = "merge0"} : <i32>
    %result_28, %index_29 = control_merge [%falseResult_25]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>] to <>, <i1>
    %50 = source {handshake.bb = 3 : ui32, handshake.name = "source5"} : <>
    %51 = constant %50 {handshake.bb = 3 : ui32, handshake.name = "constant12", value = 1 : i32} : <>, <i32>
    %52 = source {handshake.bb = 3 : ui32, handshake.name = "source6"} : <>
    %53 = constant %52 {handshake.bb = 3 : ui32, handshake.name = "constant13", value = 20 : i32} : <>, <i32>
    %54 = addi %49, %51 {handshake.bb = 3 : ui32, handshake.name = "addi6"} : <i32>
    %55 = cmpi ult, %54, %53 {handshake.bb = 3 : ui32, handshake.name = "cmpi1"} : <i32>
    %trueResult_30, %falseResult_31 = cond_br %55, %54 {handshake.bb = 3 : ui32, handshake.name = "cond_br6"} : <i1>, <i32>
    %trueResult_32, %falseResult_33 = cond_br %55, %result_28 {handshake.bb = 3 : ui32, handshake.name = "cond_br7"} : <i1>, <>
    %result_34, %index_35 = control_merge [%falseResult_33]  {handshake.bb = 4 : ui32, handshake.name = "control_merge3"} : [<>] to <>, <i1>
    end {handshake.bb = 4 : ui32, handshake.name = "end0"} %memEnd_5, %memEnd_3, %memEnd_1, %memEnd, %arg8 : <>, <>, <>, <>, <>
  }
}

