module {
  handshake.func @iterative_sqrt(%arg0: memref<10xi32>, %arg1: !handshake.control<>, %arg2: !handshake.control<>, ...) -> (!handshake.channel<i32>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["A", "A_start", "start"], resNames = ["out0", "A_end", "end"]} {
    %outputs:3, %memEnd = mem_controller[%arg0 : memref<10xi32>] %arg1 (%addressResult, %addressResult_4, %26, %17, %0#1, %0#2, %0#3) %result_18 {connectedBlocks = [1 : i32, 2 : i32, 4 : i32, 3 : i32], handshake.name = "mem_controller1"} :    (!handshake.channel<i4>, !handshake.channel<i4>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i4>, !handshake.channel<i4>, !handshake.channel<i32>) -> (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>)
    %0:4 = lsq[MC] (%result_10, %addressResult_12, %dataResult_13, %result_14, %addressResult_16, %dataResult_17, %result_18, %addressResult_20, %outputs#2)  {groupSizes = [1 : i32, 1 : i32, 1 : i32], handshake.name = "lsq1"} : (!handshake.control<>, !handshake.channel<i4>, !handshake.channel<i32>, !handshake.control<>, !handshake.channel<i4>, !handshake.channel<i32>, !handshake.control<>, !handshake.channel<i4>, !handshake.channel<i32>) -> (!handshake.channel<i32>, !handshake.channel<i4>, !handshake.channel<i4>, !handshake.channel<i32>)
    %1 = br %arg2 {handshake.bb = 0 : ui32, handshake.name = "br2"} : <>
    %result, %index = control_merge [%1, %23, %33]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>, <>] to <>, <i2>
    %2 = constant %result {handshake.name = "constant4", value = false} : <>, <i1>
    %3 = extui %2 {handshake.name = "extui0"} : <i1> to <i4>
    %4 = source {handshake.bb = 1 : ui32, handshake.name = "source0"} : <>
    %5 = constant %4 {handshake.name = "constant14", value = 10 : i5} : <>, <i5>
    %6 = extsi %5 {handshake.name = "extsi1"} : <i5> to <i32>
    %addressResult, %dataResult = load[%3] %outputs#0 {handshake.bb = 1 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i4>, <i32>, <i4>, <i32>
    %7 = cmpi sgt, %dataResult, %6 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i32>
    %trueResult, %falseResult = cond_br %7, %dataResult {handshake.bb = 1 : ui32, handshake.name = "cond_br2"} : <i1>, <i32>
    %trueResult_0, %falseResult_1 = cond_br %7, %result {handshake.bb = 1 : ui32, handshake.name = "cond_br3"} : <i1>, <>
    %8 = merge %trueResult {handshake.bb = 2 : ui32, handshake.name = "merge0"} : <i32>
    %result_2, %index_3 = control_merge [%trueResult_0]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>] to <>, <i1>
    %9 = constant %result_2 {handshake.name = "constant15", value = 1 : i2} : <>, <i2>
    %10 = extui %9 {handshake.name = "extui1"} : <i2> to <i4>
    %11 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %12 = constant %11 {handshake.name = "constant16", value = 10 : i5} : <>, <i5>
    %13 = extsi %12 {handshake.name = "extsi3"} : <i5> to <i32>
    %addressResult_4, %dataResult_5 = load[%10] %outputs#1 {handshake.bb = 2 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i4>, <i32>, <i4>, <i32>
    %14 = cmpi slt, %dataResult_5, %13 {handshake.bb = 2 : ui32, handshake.name = "cmpi1"} : <i32>
    %trueResult_6, %falseResult_7 = cond_br %14, %8 {handshake.bb = 2 : ui32, handshake.name = "cond_br4"} : <i1>, <i32>
    %trueResult_8, %falseResult_9 = cond_br %14, %result_2 {handshake.bb = 2 : ui32, handshake.name = "cond_br5"} : <i1>, <>
    %15 = merge %trueResult_6 {handshake.bb = 3 : ui32, handshake.name = "merge1"} : <i32>
    %result_10, %index_11 = control_merge [%trueResult_8]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>] to <>, <i1>
    %16 = constant %result_10 {handshake.name = "constant17", value = 1 : i2} : <>, <i2>
    %17 = extsi %16 {handshake.name = "extsi4"} : <i2> to <i32>
    %18 = constant %result_10 {handshake.name = "constant18", value = false} : <>, <i1>
    %19 = extui %18 {handshake.name = "extui2"} : <i1> to <i4>
    %20 = source {handshake.bb = 3 : ui32, handshake.name = "source2"} : <>
    %21 = constant %20 {handshake.bb = 3 : ui32, handshake.name = "constant10", value = -1 : i32} : <>, <i32>
    %22 = addi %15, %21 {handshake.bb = 3 : ui32, handshake.name = "addi0"} : <i32>
    %addressResult_12, %dataResult_13 = store[%19] %22 {handshake.bb = 3 : ui32, handshake.deps = #handshake<deps[<"load2" (1)>]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 0>, handshake.name = "store0"} : <i4>, <i32>, <i4>, <i32>
    %23 = br %result_10 {handshake.bb = 3 : ui32, handshake.name = "br4"} : <>
    %24 = merge %falseResult_7 {handshake.bb = 4 : ui32, handshake.name = "merge2"} : <i32>
    %result_14, %index_15 = control_merge [%falseResult_9]  {handshake.bb = 4 : ui32, handshake.name = "control_merge3"} : [<>] to <>, <i1>
    %25 = constant %result_14 {handshake.name = "constant19", value = 1 : i2} : <>, <i2>
    %26 = extsi %25 {handshake.name = "extsi6"} : <i2> to <i32>
    %27 = constant %result_14 {handshake.name = "constant20", value = false} : <>, <i1>
    %28 = extui %27 {handshake.name = "extui3"} : <i1> to <i4>
    %29 = source {handshake.bb = 4 : ui32, handshake.name = "source3"} : <>
    %30 = constant %29 {handshake.name = "constant21", value = 1 : i2} : <>, <i2>
    %31 = extsi %30 {handshake.name = "extsi8"} : <i2> to <i32>
    %32 = shrsi %24, %31 {handshake.bb = 4 : ui32, handshake.name = "shrsi0"} : <i32>
    %addressResult_16, %dataResult_17 = store[%28] %32 {handshake.bb = 4 : ui32, handshake.deps = #handshake<deps[<"load2" (1)>]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 1>, handshake.name = "store1"} : <i4>, <i32>, <i4>, <i32>
    %33 = br %result_14 {handshake.bb = 4 : ui32, handshake.name = "br5"} : <>
    %result_18, %index_19 = control_merge [%falseResult_1]  {handshake.bb = 5 : ui32, handshake.name = "control_merge4"} : [<>] to <>, <i1>
    %34 = constant %result_18 {handshake.name = "constant22", value = false} : <>, <i1>
    %35 = extui %34 {handshake.name = "extui4"} : <i1> to <i4>
    %addressResult_20, %dataResult_21 = load[%35] %0#0 {handshake.bb = 5 : ui32, handshake.mem_interface = #handshake.mem_interface<LSQ: 2>, handshake.name = "load2"} : <i4>, <i32>, <i4>, <i32>
    end {handshake.bb = 5 : ui32, handshake.name = "end0"} %dataResult_21, %memEnd, %arg2 : <i32>, <>, <>
  }
}

