module {
  handshake.func @multiple_exit(%arg0: memref<10xi32>, %arg1: !handshake.channel<i32>, %arg2: !handshake.control<>, %arg3: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>) attributes {argNames = ["arr", "size", "arr_start", "start"], cfg.edges = "[0,1][2,3,1,cmpi1][4,1][1,2,5,andi0][3,4,1,cmpi2]", resNames = ["arr_end", "end"]} {
    %0:4 = lsq[%arg0 : memref<10xi32>] (%arg2, %result_6, %addressResult, %result_10, %addressResult_12, %result_18, %addressResult_20, %addressResult_22, %dataResult_23, %result_24)  {groupSizes = [1 : i32, 1 : i32, 2 : i32], handshake.name = "lsq0"} : (!handshake.control<>, !handshake.control<>, !handshake.channel<i32>, !handshake.control<>, !handshake.channel<i32>, !handshake.control<>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>) -> (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>)
    %1 = source {handshake.bb = 0 : ui32, handshake.name = "source0"} : <>
    %2 = constant %1 {handshake.bb = 0 : ui32, handshake.name = "constant0", value = 0 : i32} : <>, <i32>
    %3 = source {handshake.bb = 0 : ui32, handshake.name = "source1"} : <>
    %4 = constant %3 {handshake.bb = 0 : ui32, handshake.name = "constant4", value = true} : <>, <i1>
    %5 = br %arg3 {handshake.bb = 0 : ui32, handshake.name = "br1"} : <>
    %trueResult, %falseResult = cond_br %16, %11 {handshake.bb = 1 : ui32, handshake.name = "cond_br14"} : <i1>, <i32>
    %trueResult_0, %falseResult_1 = cond_br %16, %9 {handshake.bb = 1 : ui32, handshake.name = "cond_br15"} : <i1>, <i1>
    %trueResult_2, %falseResult_3 = cond_br %16, %8 {handshake.bb = 1 : ui32, handshake.name = "cond_br16"} : <i1>, <i32>
    %6 = constant %arg3 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "constant1", value = false} : <>, <i1>
    %7 = merge %6, %16 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "merge0"} : <i1>
    %8 = mux %7 [%arg1, %trueResult_2] {ftd.regen, handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i32>, <i32>] to <i32>
    %result, %index = control_merge [%5, %falseResult_9, %falseResult_15, %31]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>, <>, <>] to <>, <i2>
    %9 = mux %21 [%18, %10] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i1>, [<i1>, <i1>] to <i1>
    %10 = mux %24 [%24, %26] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux2"} : <i1>, [<i1>, <i1>] to <i1>
    %11 = mux %21 [%14, %12] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux3"} : <i1>, [<i32>, <i32>] to <i32>
    %12 = mux %24 [%14, %30] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux4"} : <i1>, [<i32>, <i32>] to <i32>
    %13 = mux %7 [%4, %trueResult_0] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux5"} : <i1>, [<i1>, <i1>] to <i1>
    %14 = mux %7 [%2, %trueResult] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux6"} : <i1>, [<i32>, <i32>] to <i32>
    %15 = cmpi slt, %14, %8 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i32>
    %16 = andi %15, %13 {handshake.bb = 1 : ui32, handshake.name = "andi0"} : <i1>
    %trueResult_4, %falseResult_5 = cond_br %16, %result {handshake.bb = 1 : ui32, handshake.name = "cond_br6"} : <i1>, <>
    %result_6, %index_7 = control_merge [%trueResult_4]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>] to <>, <i1>
    %17 = source {handshake.bb = 2 : ui32, handshake.name = "source3"} : <>
    %18 = constant %17 {handshake.bb = 2 : ui32, handshake.name = "constant5", value = false} : <>, <i1>
    %19 = source {handshake.bb = 2 : ui32, handshake.name = "source4"} : <>
    %20 = constant %19 {handshake.bb = 2 : ui32, handshake.name = "constant6", value = -1 : i32} : <>, <i32>
    %addressResult, %dataResult = load[%trueResult_16] %0#0 {handshake.bb = 2 : ui32, handshake.deps = #handshake<deps[<"store1" (0)>]>, handshake.name = "load3"} : <i32>, <i32>, <i32>, <i32>
    %21 = cmpi ne, %dataResult, %20 {handshake.bb = 2 : ui32, handshake.name = "cmpi1"} : <i32>
    %trueResult_8, %falseResult_9 = cond_br %21, %result_6 {handshake.bb = 2 : ui32, handshake.name = "cond_br11"} : <i1>, <>
    %result_10, %index_11 = control_merge [%trueResult_8]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>] to <>, <i1>
    %22 = source {handshake.bb = 3 : ui32, handshake.name = "source5"} : <>
    %23 = constant %22 {handshake.bb = 3 : ui32, handshake.name = "constant7", value = 0 : i32} : <>, <i32>
    %addressResult_12, %dataResult_13 = load[%trueResult_16] %0#1 {handshake.bb = 3 : ui32, handshake.deps = #handshake<deps[<"store1" (0)>]>, handshake.name = "load4"} : <i32>, <i32>, <i32>, <i32>
    %24 = cmpi ne, %dataResult_13, %23 {handshake.bb = 3 : ui32, handshake.name = "cmpi2"} : <i32>
    %trueResult_14, %falseResult_15 = cond_br %24, %result_10 {handshake.bb = 3 : ui32, handshake.name = "cond_br12"} : <i1>, <>
    %trueResult_16, %falseResult_17 = cond_br %16, %14 {handshake.bb = 4 : ui32, handshake.name = "cond_br17"} : <i1>, <i32>
    %result_18, %index_19 = control_merge [%trueResult_14]  {handshake.bb = 4 : ui32, handshake.name = "control_merge3"} : [<>] to <>, <i1>
    %25 = source {handshake.bb = 4 : ui32, handshake.name = "source6"} : <>
    %26 = constant %25 {handshake.bb = 4 : ui32, handshake.name = "constant8", value = true} : <>, <i1>
    %27 = source {handshake.bb = 4 : ui32, handshake.name = "source7"} : <>
    %28 = constant %27 {handshake.bb = 4 : ui32, handshake.name = "constant9", value = 1 : i32} : <>, <i32>
    %addressResult_20, %dataResult_21 = load[%trueResult_16] %0#2 {handshake.bb = 4 : ui32, handshake.deps = #handshake<deps[<"store1" (0)>]>, handshake.name = "load5"} : <i32>, <i32>, <i32>, <i32>
    %29 = addi %dataResult_21, %28 {handshake.bb = 4 : ui32, handshake.name = "addi0"} : <i32>
    %addressResult_22, %dataResult_23 = store[%trueResult_16] %29 {handshake.bb = 4 : ui32, handshake.deps = #handshake<deps[<"load3" (0)>, <"load4" (0)>, <"load5" (0)>, <"store1" (0)>]>, handshake.name = "store1"} : <i32>, <i32>, <i32>, <i32>
    %30 = addi %trueResult_16, %28 {handshake.bb = 4 : ui32, handshake.name = "addi1"} : <i32>
    %31 = br %result_18 {handshake.bb = 4 : ui32, handshake.name = "br2"} : <>
    %result_24, %index_25 = control_merge [%falseResult_5]  {handshake.bb = 5 : ui32, handshake.name = "control_merge4"} : [<>] to <>, <i1>
    end {handshake.bb = 5 : ui32, handshake.name = "end0"} %0#3, %arg3 : <>, <>
  }
}

