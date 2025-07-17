module {
  handshake.func @iterative_sqrt(%arg0: memref<10xi32>, %arg1: !handshake.control<>, %arg2: !handshake.control<>, ...) -> (!handshake.channel<i32>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["A", "A_start", "start"], resNames = ["out0", "A_end", "end"]} {
    %outputs:3, %memEnd = mem_controller[%arg0 : memref<10xi32>] %arg1 (%addressResult, %addressResult_4, %18, %11, %0#1, %0#2, %0#3) %result_18 {connectedBlocks = [1 : i32, 2 : i32, 4 : i32, 3 : i32], handshake.name = "mem_controller0"} :    (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>) -> (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>)
    %0:4 = lsq[MC] (%result_10, %addressResult_12, %dataResult_13, %result_14, %addressResult_16, %dataResult_17, %result_18, %addressResult_20, %outputs#2)  {groupSizes = [1 : i32, 1 : i32, 1 : i32], handshake.name = "lsq0"} : (!handshake.control<>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>, !handshake.channel<i32>, !handshake.channel<i32>) -> (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>)
    %1 = br %arg2 {handshake.bb = 0 : ui32, handshake.name = "br2"} : <>
    %result, %index = control_merge [%1, %17, %24]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>, <>] to <>, <i2>
    %2 = constant %result {handshake.bb = 1 : ui32, handshake.name = "constant5", value = 0 : i32} : <>, <i32>
    %3 = source {handshake.bb = 1 : ui32, handshake.name = "source0"} : <>
    %4 = constant %3 {handshake.bb = 1 : ui32, handshake.name = "constant6", value = 10 : i32} : <>, <i32>
    %addressResult, %dataResult = load[%2] %outputs#0 {handshake.bb = 1 : ui32, handshake.name = "load0"} : <i32>, <i32>, <i32>, <i32>
    %5 = cmpi sgt, %dataResult, %4 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i32>
    %trueResult, %falseResult = cond_br %5, %dataResult {handshake.bb = 1 : ui32, handshake.name = "cond_br2"} : <i1>, <i32>
    %trueResult_0, %falseResult_1 = cond_br %5, %result {handshake.bb = 1 : ui32, handshake.name = "cond_br3"} : <i1>, <>
    %6 = merge %trueResult {handshake.bb = 2 : ui32, handshake.name = "merge0"} : <i32>
    %result_2, %index_3 = control_merge [%trueResult_0]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>] to <>, <i1>
    %7 = constant %result_2 {handshake.bb = 2 : ui32, handshake.name = "constant7", value = 1 : i32} : <>, <i32>
    %8 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %9 = constant %8 {handshake.bb = 2 : ui32, handshake.name = "constant8", value = 10 : i32} : <>, <i32>
    %addressResult_4, %dataResult_5 = load[%7] %outputs#1 {handshake.bb = 2 : ui32, handshake.name = "load1"} : <i32>, <i32>, <i32>, <i32>
    %10 = cmpi slt, %dataResult_5, %9 {handshake.bb = 2 : ui32, handshake.name = "cmpi1"} : <i32>
    %trueResult_6, %falseResult_7 = cond_br %10, %6 {handshake.bb = 2 : ui32, handshake.name = "cond_br4"} : <i1>, <i32>
    %trueResult_8, %falseResult_9 = cond_br %10, %result_2 {handshake.bb = 2 : ui32, handshake.name = "cond_br5"} : <i1>, <>
    %11 = constant %result_10 {handshake.bb = 3 : ui32, handshake.name = "constant0", value = 1 : i32} : <>, <i32>
    %12 = merge %trueResult_6 {handshake.bb = 3 : ui32, handshake.name = "merge1"} : <i32>
    %result_10, %index_11 = control_merge [%trueResult_8]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>] to <>, <i1>
    %13 = constant %result_10 {handshake.bb = 3 : ui32, handshake.name = "constant9", value = 0 : i32} : <>, <i32>
    %14 = source {handshake.bb = 3 : ui32, handshake.name = "source2"} : <>
    %15 = constant %14 {handshake.bb = 3 : ui32, handshake.name = "constant10", value = -1 : i32} : <>, <i32>
    %16 = addi %12, %15 {handshake.bb = 3 : ui32, handshake.name = "addi0"} : <i32>
    %addressResult_12, %dataResult_13 = store[%13] %16 {handshake.bb = 3 : ui32, handshake.deps = #handshake<deps[<"load2" (1)>]>, handshake.name = "store0"} : <i32>, <i32>, <i32>, <i32>
    %17 = br %result_10 {handshake.bb = 3 : ui32, handshake.name = "br4"} : <>
    %18 = constant %result_14 {handshake.bb = 4 : ui32, handshake.name = "constant1", value = 1 : i32} : <>, <i32>
    %19 = merge %falseResult_7 {handshake.bb = 4 : ui32, handshake.name = "merge2"} : <i32>
    %result_14, %index_15 = control_merge [%falseResult_9]  {handshake.bb = 4 : ui32, handshake.name = "control_merge3"} : [<>] to <>, <i1>
    %20 = constant %result_14 {handshake.bb = 4 : ui32, handshake.name = "constant11", value = 0 : i32} : <>, <i32>
    %21 = source {handshake.bb = 4 : ui32, handshake.name = "source3"} : <>
    %22 = constant %21 {handshake.bb = 4 : ui32, handshake.name = "constant12", value = 1 : i32} : <>, <i32>
    %23 = shrsi %19, %22 {handshake.bb = 4 : ui32, handshake.name = "shrsi0"} : <i32>
    %addressResult_16, %dataResult_17 = store[%20] %23 {handshake.bb = 4 : ui32, handshake.deps = #handshake<deps[<"load2" (1)>]>, handshake.name = "store1"} : <i32>, <i32>, <i32>, <i32>
    %24 = br %result_14 {handshake.bb = 4 : ui32, handshake.name = "br5"} : <>
    %result_18, %index_19 = control_merge [%falseResult_1]  {handshake.bb = 5 : ui32, handshake.name = "control_merge4"} : [<>] to <>, <i1>
    %25 = constant %result_18 {handshake.bb = 5 : ui32, handshake.name = "constant13", value = 0 : i32} : <>, <i32>
    %addressResult_20, %dataResult_21 = load[%25] %0#0 {handshake.bb = 5 : ui32, handshake.name = "load2"} : <i32>, <i32>, <i32>, <i32>
    end {handshake.bb = 5 : ui32, handshake.name = "end0"} %dataResult_21, %memEnd, %arg2 : <i32>, <>, <>
  }
}

