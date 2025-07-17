module {
  handshake.func @iterative_sqrt(%arg0: memref<10xi32>, %arg1: !handshake.control<>, %arg2: !handshake.control<>, ...) -> (!handshake.channel<i32>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["A", "A_start", "start"], cfg.edges = "[0,1][2,3,4,cmpi1][4,1][1,2,5,cmpi0][3,1]", resNames = ["out0", "A_end", "end"]} {
    %outputs:3, %memEnd = mem_controller[%arg0 : memref<10xi32>] %arg1 (%addressResult, %addressResult_4, %33, %21, %0#1, %0#2, %0#3) %result_24 {connectedBlocks = [1 : i32, 2 : i32, 4 : i32, 3 : i32], handshake.name = "mem_controller0"} :    (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>) -> (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>)
    %0:4 = lsq[MC] (%result_12, %addressResult_14, %dataResult_15, %result_20, %addressResult_22, %dataResult_23, %result_24, %addressResult_26, %outputs#2)  {groupSizes = [1 : i32, 1 : i32, 1 : i32], handshake.name = "lsq0"} : (!handshake.control<>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>, !handshake.channel<i32>, !handshake.channel<i32>) -> (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>)
    %1 = br %arg2 {handshake.bb = 0 : ui32, handshake.name = "br2"} : <>
    %2 = constant %arg2 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "constant0", value = false} : <>, <i1>
    %3 = merge %2, %8 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "merge0"} : <i1>
    %4 = mux %3 [%arg2, %trueResult_0] {ftd.regen, handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<>, <>] to <>
    %result, %index = control_merge [%1, %26, %38]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>, <>] to <>, <i2>
    %5 = constant %4 {handshake.bb = 1 : ui32, handshake.name = "constant5", value = 0 : i32} : <>, <i32>
    %6 = source {handshake.bb = 1 : ui32, handshake.name = "source0"} : <>
    %7 = constant %6 {handshake.bb = 1 : ui32, handshake.name = "constant6", value = 10 : i32} : <>, <i32>
    %addressResult, %dataResult = load[%5] %outputs#0 {handshake.bb = 1 : ui32, handshake.name = "load0"} : <i32>, <i32>, <i32>, <i32>
    %8 = cmpi sgt, %dataResult, %7 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i32>
    %trueResult, %falseResult = cond_br %8, %result {handshake.bb = 1 : ui32, handshake.name = "cond_br6"} : <i1>, <>
    %trueResult_0, %falseResult_1 = cond_br %8, %4 {handshake.bb = 2 : ui32, handshake.name = "cond_br16"} : <i1>, <>
    %result_2, %index_3 = control_merge [%trueResult]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>] to <>, <i1>
    %9 = constant %trueResult_0 {handshake.bb = 2 : ui32, handshake.name = "constant7", value = 1 : i32} : <>, <i32>
    %10 = source {handshake.bb = 2 : ui32, handshake.name = "source2"} : <>
    %11 = constant %10 {handshake.bb = 2 : ui32, handshake.name = "constant8", value = 10 : i32} : <>, <i32>
    %addressResult_4, %dataResult_5 = load[%9] %outputs#1 {handshake.bb = 2 : ui32, handshake.name = "load1"} : <i32>, <i32>, <i32>, <i32>
    %12 = cmpi slt, %dataResult_5, %11 {handshake.bb = 2 : ui32, handshake.name = "cmpi1"} : <i32>
    %trueResult_6, %falseResult_7 = cond_br %12, %result_2 {handshake.bb = 2 : ui32, handshake.name = "cond_br11"} : <i1>, <>
    %trueResult_8, %falseResult_9 = cond_br %14, %dataResult {handshake.bb = 3 : ui32, handshake.name = "cond_br12"} : <i1>, <i32>
    %13 = not %12 {ftd.skip, handshake.bb = 3 : ui32, handshake.name = "not7"} : <i1>
    %14 = mux %8 [%16, %13] {ftd.skip, handshake.bb = 3 : ui32, handshake.name = "mux4"} : <i1>, [<i1>, <i1>] to <i1>
    %15 = source {handshake.bb = 3 : ui32, handshake.name = "source3"} : <>
    %16 = constant %15 {ftd.skip, handshake.bb = 3 : ui32, handshake.name = "constant14", value = true} : <>, <i1>
    %trueResult_10, %falseResult_11 = cond_br %18, %4 {handshake.bb = 3 : ui32, handshake.name = "cond_br13"} : <i1>, <>
    %17 = not %12 {ftd.skip, handshake.bb = 3 : ui32, handshake.name = "not8"} : <i1>
    %18 = mux %8 [%20, %17] {ftd.skip, handshake.bb = 3 : ui32, handshake.name = "mux5"} : <i1>, [<i1>, <i1>] to <i1>
    %19 = source {handshake.bb = 3 : ui32, handshake.name = "source4"} : <>
    %20 = constant %19 {ftd.skip, handshake.bb = 3 : ui32, handshake.name = "constant15", value = true} : <>, <i1>
    %21 = constant %result_12 {handshake.bb = 3 : ui32, handshake.name = "constant16", value = 1 : i32} : <>, <i32>
    %result_12, %index_13 = control_merge [%trueResult_6]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>] to <>, <i1>
    %22 = constant %falseResult_11 {handshake.bb = 3 : ui32, handshake.name = "constant9", value = 0 : i32} : <>, <i32>
    %23 = source {handshake.bb = 3 : ui32, handshake.name = "source5"} : <>
    %24 = constant %23 {handshake.bb = 3 : ui32, handshake.name = "constant10", value = -1 : i32} : <>, <i32>
    %25 = addi %falseResult_9, %24 {handshake.bb = 3 : ui32, handshake.name = "addi0"} : <i32>
    %addressResult_14, %dataResult_15 = store[%22] %25 {handshake.bb = 3 : ui32, handshake.deps = #handshake<deps[<"load2" (1)>]>, handshake.name = "store0"} : <i32>, <i32>, <i32>, <i32>
    %26 = br %result_12 {handshake.bb = 3 : ui32, handshake.name = "br4"} : <>
    %trueResult_16, %falseResult_17 = cond_br %29, %dataResult {handshake.bb = 4 : ui32, handshake.name = "cond_br14"} : <i1>, <i32>
    %27 = source {handshake.bb = 4 : ui32, handshake.name = "source6"} : <>
    %28 = constant %27 {ftd.skip, handshake.bb = 4 : ui32, handshake.name = "constant17", value = true} : <>, <i1>
    %29 = mux %8 [%28, %12] {ftd.skip, handshake.bb = 4 : ui32, handshake.name = "mux6"} : <i1>, [<i1>, <i1>] to <i1>
    %trueResult_18, %falseResult_19 = cond_br %32, %4 {handshake.bb = 4 : ui32, handshake.name = "cond_br15"} : <i1>, <>
    %30 = source {handshake.bb = 4 : ui32, handshake.name = "source7"} : <>
    %31 = constant %30 {ftd.skip, handshake.bb = 4 : ui32, handshake.name = "constant18", value = true} : <>, <i1>
    %32 = mux %8 [%31, %12] {ftd.skip, handshake.bb = 4 : ui32, handshake.name = "mux7"} : <i1>, [<i1>, <i1>] to <i1>
    %33 = constant %result_20 {handshake.bb = 4 : ui32, handshake.name = "constant19", value = 1 : i32} : <>, <i32>
    %result_20, %index_21 = control_merge [%falseResult_7]  {handshake.bb = 4 : ui32, handshake.name = "control_merge3"} : [<>] to <>, <i1>
    %34 = constant %falseResult_19 {handshake.bb = 4 : ui32, handshake.name = "constant11", value = 0 : i32} : <>, <i32>
    %35 = source {handshake.bb = 4 : ui32, handshake.name = "source8"} : <>
    %36 = constant %35 {handshake.bb = 4 : ui32, handshake.name = "constant12", value = 1 : i32} : <>, <i32>
    %37 = shrsi %falseResult_17, %36 {handshake.bb = 4 : ui32, handshake.name = "shrsi0"} : <i32>
    %addressResult_22, %dataResult_23 = store[%34] %37 {handshake.bb = 4 : ui32, handshake.deps = #handshake<deps[<"load2" (1)>]>, handshake.name = "store1"} : <i32>, <i32>, <i32>, <i32>
    %38 = br %result_20 {handshake.bb = 4 : ui32, handshake.name = "br5"} : <>
    %result_24, %index_25 = control_merge [%falseResult]  {handshake.bb = 5 : ui32, handshake.name = "control_merge4"} : [<>] to <>, <i1>
    %39 = constant %arg2 {handshake.bb = 5 : ui32, handshake.name = "constant13", value = 0 : i32} : <>, <i32>
    %addressResult_26, %dataResult_27 = load[%39] %0#0 {handshake.bb = 5 : ui32, handshake.name = "load2"} : <i32>, <i32>, <i32>, <i32>
    end {handshake.bb = 5 : ui32, handshake.name = "end0"} %dataResult_27, %memEnd, %arg2 : <i32>, <>, <>
  }
}

