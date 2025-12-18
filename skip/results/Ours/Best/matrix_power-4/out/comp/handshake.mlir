module {
  handshake.func @matrix_power(%arg0: memref<400xi32>, %arg1: memref<20xi32>, %arg2: memref<20xi32>, %arg3: memref<20xi32>, %arg4: !handshake.control<>, %arg5: !handshake.control<>, %arg6: !handshake.control<>, %arg7: !handshake.control<>, %arg8: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["mat", "row", "col", "a", "mat_start", "row_start", "col_start", "a_start", "start"], cfg.edges = "[0,1][2,2,3,cmpi0][1,2][3,1,4,cmpi1]", resNames = ["mat_end", "row_end", "col_end", "a_end", "end"]} {
    %outputs, %memEnd = mem_controller[%arg3 : memref<20xi32>] %arg7 (%addressResult_6) %result_28 {connectedBlocks = [2 : i32], handshake.name = "mem_controller0"} :    (!handshake.channel<i32>) -> !handshake.channel<i32>
    %outputs_0, %memEnd_1 = mem_controller[%arg2 : memref<20xi32>] %arg6 (%addressResult_8) %result_28 {connectedBlocks = [2 : i32], handshake.name = "mem_controller1"} :    (!handshake.channel<i32>) -> !handshake.channel<i32>
    %outputs_2, %memEnd_3 = mem_controller[%arg1 : memref<20xi32>] %arg5 (%addressResult) %result_28 {connectedBlocks = [2 : i32], handshake.name = "mem_controller2"} :    (!handshake.channel<i32>) -> !handshake.channel<i32>
    %0:4 = lsq[%arg0 : memref<400xi32>] (%arg4, %result_4, %addressResult_10, %addressResult_12, %addressResult_14, %dataResult_15, %result_28)  {groupSizes = [3 : i32], handshake.name = "lsq0"} : (!handshake.control<>, !handshake.control<>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>) -> (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>, !handshake.control<>)
    %1 = constant %arg8 {handshake.bb = 0 : ui32, handshake.name = "constant3", value = 1 : i32} : <>, <i32>
    %2 = br %1 {handshake.bb = 0 : ui32, handshake.name = "br2"} : <i32>
    %3 = br %arg8 {handshake.bb = 0 : ui32, handshake.name = "br3"} : <>
    %4 = mux %index [%2, %trueResult_24] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i32>, <i32>] to <i32>
    %result, %index = control_merge [%3, %trueResult_26]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %5 = constant %result {handshake.bb = 1 : ui32, handshake.name = "constant6", value = 0 : i32} : <>, <i32>
    %6 = source {handshake.bb = 1 : ui32, handshake.name = "source0"} : <>
    %7 = constant %6 {handshake.bb = 1 : ui32, handshake.name = "constant7", value = -1 : i32} : <>, <i32>
    %8 = trunci %4 {handshake.bb = 1 : ui32, handshake.name = "index_cast0"} : <i32> to <i32>
    %9 = addi %8, %7 {handshake.bb = 1 : ui32, handshake.name = "addi0"} : <i32>
    %10 = trunci %9 {handshake.bb = 1 : ui32, handshake.name = "index_cast1"} : <i32> to <i32>
    %11 = br %5 {handshake.bb = 1 : ui32, handshake.name = "br4"} : <i32>
    %12 = br %4 {handshake.bb = 1 : ui32, handshake.name = "br5"} : <i32>
    %13 = br %10 {handshake.bb = 1 : ui32, handshake.name = "br6"} : <i32>
    %14 = br %result {handshake.bb = 1 : ui32, handshake.name = "br7"} : <>
    %15 = mux %index_5 [%11, %trueResult] {handshake.bb = 2 : ui32, handshake.name = "mux1"} : <i1>, [<i32>, <i32>] to <i32>
    %16 = mux %index_5 [%12, %trueResult_16] {handshake.bb = 2 : ui32, handshake.name = "mux2"} : <i1>, [<i32>, <i32>] to <i32>
    %17 = mux %index_5 [%13, %trueResult_18] {handshake.bb = 2 : ui32, handshake.name = "mux3"} : <i1>, [<i32>, <i32>] to <i32>
    %result_4, %index_5 = control_merge [%14, %trueResult_20]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>, <>] to <>, <i1>
    %18 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %19 = constant %18 {handshake.bb = 2 : ui32, handshake.name = "constant8", value = 1 : i32} : <>, <i32>
    %20 = source {handshake.bb = 2 : ui32, handshake.name = "source2"} : <>
    %21 = constant %20 {handshake.bb = 2 : ui32, handshake.name = "constant9", value = 20 : i32} : <>, <i32>
    %22 = source {handshake.bb = 2 : ui32, handshake.name = "source3"} : <>
    %23 = constant %22 {handshake.bb = 2 : ui32, handshake.name = "constant10", value = 4 : i32} : <>, <i32>
    %24 = source {handshake.bb = 2 : ui32, handshake.name = "source4"} : <>
    %25 = constant %24 {handshake.bb = 2 : ui32, handshake.name = "constant11", value = 2 : i32} : <>, <i32>
    %addressResult, %dataResult = load[%15] %outputs_2 {handshake.bb = 2 : ui32, handshake.name = "load0"} : <i32>, <i32>, <i32>, <i32>
    %26 = trunci %dataResult {handshake.bb = 2 : ui32, handshake.name = "index_cast2"} : <i32> to <i32>
    %addressResult_6, %dataResult_7 = load[%15] %outputs {handshake.bb = 2 : ui32, handshake.name = "load1"} : <i32>, <i32>, <i32>, <i32>
    %addressResult_8, %dataResult_9 = load[%15] %outputs_0 {handshake.bb = 2 : ui32, handshake.name = "load2"} : <i32>, <i32>, <i32>, <i32>
    %27 = trunci %dataResult_9 {handshake.bb = 2 : ui32, handshake.name = "index_cast3"} : <i32> to <i32>
    %28 = shli %17, %25 {handshake.bb = 2 : ui32, handshake.name = "shli0"} : <i32>
    %29 = shli %17, %23 {handshake.bb = 2 : ui32, handshake.name = "shli1"} : <i32>
    %30 = addi %28, %29 {handshake.bb = 2 : ui32, handshake.name = "addi7"} : <i32>
    %31 = addi %27, %30 {handshake.bb = 2 : ui32, handshake.name = "addi2"} : <i32>
    %addressResult_10, %dataResult_11 = load[%31] %0#0 {handshake.bb = 2 : ui32, handshake.deps = #handshake<deps[["store0", 0, true]]>, handshake.name = "load3"} : <i32>, <i32>, <i32>, <i32>
    %32 = muli %dataResult_7, %dataResult_11 {handshake.bb = 2 : ui32, handshake.name = "muli0"} : <i32>
    %33 = shli %16, %25 {handshake.bb = 2 : ui32, handshake.name = "shli2"} : <i32>
    %34 = shli %16, %23 {handshake.bb = 2 : ui32, handshake.name = "shli3"} : <i32>
    %35 = addi %33, %34 {handshake.bb = 2 : ui32, handshake.name = "addi8"} : <i32>
    %36 = addi %26, %35 {handshake.bb = 2 : ui32, handshake.name = "addi3"} : <i32>
    %addressResult_12, %dataResult_13 = load[%36] %0#1 {handshake.bb = 2 : ui32, handshake.deps = #handshake<deps[["store0", 0, true]]>, handshake.name = "load4"} : <i32>, <i32>, <i32>, <i32>
    %37 = addi %dataResult_13, %32 {handshake.bb = 2 : ui32, handshake.name = "addi1"} : <i32>
    %38 = shli %16, %25 {handshake.bb = 2 : ui32, handshake.name = "shli4"} : <i32>
    %39 = shli %16, %23 {handshake.bb = 2 : ui32, handshake.name = "shli5"} : <i32>
    %40 = addi %38, %39 {handshake.bb = 2 : ui32, handshake.name = "addi9"} : <i32>
    %41 = addi %26, %40 {handshake.bb = 2 : ui32, handshake.name = "addi4"} : <i32>
    %addressResult_14, %dataResult_15, %doneResult = store[%41] %37 %0#2 {handshake.bb = 2 : ui32, handshake.deps = #handshake<deps[["load3", 0, true], ["load4", 0, true], ["store0", 0, true]]>, handshake.name = "store0"} : <i32>, <i32>, <>, <i32>, <i32>, <>
    %42 = addi %15, %19 {handshake.bb = 2 : ui32, handshake.name = "addi5"} : <i32>
    %43 = cmpi ult, %42, %21 {handshake.bb = 2 : ui32, handshake.name = "cmpi0"} : <i32>
    %trueResult, %falseResult = cond_br %43, %42 {handshake.bb = 2 : ui32, handshake.name = "cond_br2"} : <i1>, <i32>
    %trueResult_16, %falseResult_17 = cond_br %43, %16 {handshake.bb = 2 : ui32, handshake.name = "cond_br3"} : <i1>, <i32>
    %trueResult_18, %falseResult_19 = cond_br %43, %17 {handshake.bb = 2 : ui32, handshake.name = "cond_br4"} : <i1>, <i32>
    %trueResult_20, %falseResult_21 = cond_br %43, %result_4 {handshake.bb = 2 : ui32, handshake.name = "cond_br5"} : <i1>, <>
    %44 = merge %falseResult_17 {handshake.bb = 3 : ui32, handshake.name = "merge0"} : <i32>
    %result_22, %index_23 = control_merge [%falseResult_21]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>] to <>, <i1>
    %45 = source {handshake.bb = 3 : ui32, handshake.name = "source5"} : <>
    %46 = constant %45 {handshake.bb = 3 : ui32, handshake.name = "constant12", value = 1 : i32} : <>, <i32>
    %47 = source {handshake.bb = 3 : ui32, handshake.name = "source6"} : <>
    %48 = constant %47 {handshake.bb = 3 : ui32, handshake.name = "constant13", value = 20 : i32} : <>, <i32>
    %49 = addi %44, %46 {handshake.bb = 3 : ui32, handshake.name = "addi6"} : <i32>
    %50 = cmpi ult, %49, %48 {handshake.bb = 3 : ui32, handshake.name = "cmpi1"} : <i32>
    %trueResult_24, %falseResult_25 = cond_br %50, %49 {handshake.bb = 3 : ui32, handshake.name = "cond_br6"} : <i1>, <i32>
    %trueResult_26, %falseResult_27 = cond_br %50, %result_22 {handshake.bb = 3 : ui32, handshake.name = "cond_br7"} : <i1>, <>
    %result_28, %index_29 = control_merge [%falseResult_27]  {handshake.bb = 4 : ui32, handshake.name = "control_merge3"} : [<>] to <>, <i1>
    end {handshake.bb = 4 : ui32, handshake.name = "end0"} %0#3, %memEnd_3, %memEnd_1, %memEnd, %arg8 : <>, <>, <>, <>, <>
  }
}

