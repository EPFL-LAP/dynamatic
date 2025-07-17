module {
  handshake.func @multiple_exit(%arg0: memref<10xi32>, %arg1: !handshake.channel<i32>, %arg2: !handshake.control<>, %arg3: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>) attributes {argNames = ["arr", "size", "arr_start", "start"], resNames = ["arr_end", "end"]} {
    %0:4 = lsq[%arg0 : memref<10xi32>] (%arg2, %result_4, %addressResult, %result_16, %addressResult_18, %result_30, %addressResult_32, %addressResult_34, %dataResult_35, %result_36)  {groupSizes = [1 : i32, 1 : i32, 2 : i32], handshake.name = "lsq1"} : (!handshake.control<>, !handshake.control<>, !handshake.channel<i4>, !handshake.control<>, !handshake.channel<i4>, !handshake.control<>, !handshake.channel<i4>, !handshake.channel<i4>, !handshake.channel<i32>, !handshake.control<>) -> (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>)
    %1 = constant %arg3 {handshake.name = "constant1", value = false} : <>, <i1>
    %2 = constant %arg3 {handshake.bb = 0 : ui32, handshake.name = "constant4", value = true} : <>, <i1>
    %3 = br %1 {handshake.bb = 0 : ui32, handshake.name = "br1"} : <i1>
    %4 = extsi %3 {handshake.bb = 0 : ui32, handshake.name = "extsi3"} : <i1> to <i32>
    %5 = br %2 {handshake.bb = 0 : ui32, handshake.name = "br2"} : <i1>
    %6 = br %arg1 {handshake.bb = 0 : ui32, handshake.name = "br3"} : <i32>
    %7 = br %arg3 {handshake.bb = 0 : ui32, handshake.name = "br4"} : <>
    %8 = mux %index [%4, %falseResult_9, %falseResult_23, %37] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i2>, [<i32>, <i32>, <i32>, <i32>] to <i32>
    %9 = mux %index [%5, %falseResult_15, %falseResult_29, %38] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i2>, [<i1>, <i1>, <i1>, <i1>] to <i1>
    %10 = mux %index [%6, %falseResult_7, %falseResult_21, %39] {handshake.bb = 1 : ui32, handshake.name = "mux2"} : <i2>, [<i32>, <i32>, <i32>, <i32>] to <i32>
    %result, %index = control_merge [%7, %falseResult_13, %falseResult_27, %40]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>, <>, <>] to <>, <i2>
    %11 = cmpi slt, %8, %10 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i32>
    %12 = andi %11, %9 {handshake.bb = 1 : ui32, handshake.name = "andi0"} : <i1>
    %trueResult, %falseResult = cond_br %12, %8 {handshake.bb = 1 : ui32, handshake.name = "cond_br3"} : <i1>, <i32>
    %trueResult_0, %falseResult_1 = cond_br %12, %10 {handshake.bb = 1 : ui32, handshake.name = "cond_br4"} : <i1>, <i32>
    %trueResult_2, %falseResult_3 = cond_br %12, %result {handshake.bb = 1 : ui32, handshake.name = "cond_br5"} : <i1>, <>
    %13 = merge %trueResult {handshake.bb = 2 : ui32, handshake.name = "merge0"} : <i32>
    %14 = trunci %13 {handshake.bb = 2 : ui32, handshake.name = "trunci0"} : <i32> to <i4>
    %15 = trunci %13 {handshake.bb = 2 : ui32, handshake.name = "trunci1"} : <i32> to <i4>
    %16 = merge %trueResult_0 {handshake.bb = 2 : ui32, handshake.name = "merge1"} : <i32>
    %result_4, %index_5 = control_merge [%trueResult_2]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>] to <>, <i1>
    %17 = constant %result_4 {handshake.bb = 2 : ui32, handshake.name = "constant5", value = false} : <>, <i1>
    %18 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %19 = constant %18 {handshake.bb = 2 : ui32, handshake.name = "constant6", value = -1 : i32} : <>, <i32>
    %addressResult, %dataResult = load[%15] %0#0 {handshake.bb = 2 : ui32, handshake.deps = #handshake<deps[<"store1" (0)>]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 0>, handshake.name = "load3"} : <i4>, <i32>, <i4>, <i32>
    %20 = cmpi ne, %dataResult, %19 {handshake.bb = 2 : ui32, handshake.name = "cmpi1"} : <i32>
    %trueResult_6, %falseResult_7 = cond_br %20, %16 {handshake.bb = 2 : ui32, handshake.name = "cond_br6"} : <i1>, <i32>
    %trueResult_8, %falseResult_9 = cond_br %20, %13 {handshake.bb = 2 : ui32, handshake.name = "cond_br7"} : <i1>, <i32>
    %trueResult_10, %falseResult_11 = cond_br %20, %14 {handshake.bb = 2 : ui32, handshake.name = "cond_br8"} : <i1>, <i4>
    %trueResult_12, %falseResult_13 = cond_br %20, %result_4 {handshake.bb = 2 : ui32, handshake.name = "cond_br9"} : <i1>, <>
    %trueResult_14, %falseResult_15 = cond_br %20, %17 {handshake.bb = 2 : ui32, handshake.name = "cond_br10"} : <i1>, <i1>
    %21 = merge %trueResult_6 {handshake.bb = 3 : ui32, handshake.name = "merge2"} : <i32>
    %22 = merge %trueResult_8 {handshake.bb = 3 : ui32, handshake.name = "merge3"} : <i32>
    %23 = merge %trueResult_10 {handshake.bb = 3 : ui32, handshake.name = "merge4"} : <i4>
    %result_16, %index_17 = control_merge [%trueResult_12]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>] to <>, <i1>
    %24 = source {handshake.bb = 3 : ui32, handshake.name = "source1"} : <>
    %25 = constant %24 {handshake.name = "constant2", value = false} : <>, <i1>
    %26 = extsi %25 {handshake.name = "extsi1"} : <i1> to <i32>
    %addressResult_18, %dataResult_19 = load[%23] %0#1 {handshake.bb = 3 : ui32, handshake.deps = #handshake<deps[<"store1" (0)>]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 1>, handshake.name = "load4"} : <i4>, <i32>, <i4>, <i32>
    %27 = cmpi ne, %dataResult_19, %26 {handshake.bb = 3 : ui32, handshake.name = "cmpi2"} : <i32>
    %trueResult_20, %falseResult_21 = cond_br %27, %21 {handshake.bb = 3 : ui32, handshake.name = "cond_br11"} : <i1>, <i32>
    %trueResult_22, %falseResult_23 = cond_br %27, %22 {handshake.bb = 3 : ui32, handshake.name = "cond_br12"} : <i1>, <i32>
    %trueResult_24, %falseResult_25 = cond_br %27, %23 {handshake.bb = 3 : ui32, handshake.name = "cond_br13"} : <i1>, <i4>
    %trueResult_26, %falseResult_27 = cond_br %27, %result_16 {handshake.bb = 3 : ui32, handshake.name = "cond_br14"} : <i1>, <>
    %trueResult_28, %falseResult_29 = cond_br %27, %27 {handshake.bb = 3 : ui32, handshake.name = "cond_br15"} : <i1>, <i1>
    %28 = merge %trueResult_20 {handshake.bb = 4 : ui32, handshake.name = "merge5"} : <i32>
    %29 = merge %trueResult_22 {handshake.bb = 4 : ui32, handshake.name = "merge6"} : <i32>
    %30 = merge %trueResult_24 {handshake.bb = 4 : ui32, handshake.name = "merge7"} : <i4>
    %result_30, %index_31 = control_merge [%trueResult_26]  {handshake.bb = 4 : ui32, handshake.name = "control_merge3"} : [<>] to <>, <i1>
    %31 = constant %result_30 {handshake.bb = 4 : ui32, handshake.name = "constant8", value = true} : <>, <i1>
    %32 = source {handshake.bb = 4 : ui32, handshake.name = "source2"} : <>
    %33 = constant %32 {handshake.name = "constant3", value = 1 : i2} : <>, <i2>
    %34 = extsi %33 {handshake.name = "extsi2"} : <i2> to <i32>
    %addressResult_32, %dataResult_33 = load[%30] %0#2 {handshake.bb = 4 : ui32, handshake.deps = #handshake<deps[<"store1" (0)>]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 2>, handshake.name = "load5"} : <i4>, <i32>, <i4>, <i32>
    %35 = addi %dataResult_33, %34 {handshake.bb = 4 : ui32, handshake.name = "addi0"} : <i32>
    %addressResult_34, %dataResult_35 = store[%30] %35 {handshake.bb = 4 : ui32, handshake.deps = #handshake<deps[<"load3" (0)>, <"load4" (0)>, <"load5" (0)>, <"store1" (0)>]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 2>, handshake.name = "store1"} : <i4>, <i32>, <i4>, <i32>
    %36 = addi %29, %34 {handshake.bb = 4 : ui32, handshake.name = "addi1"} : <i32>
    %37 = br %36 {handshake.bb = 4 : ui32, handshake.name = "br5"} : <i32>
    %38 = br %31 {handshake.bb = 4 : ui32, handshake.name = "br6"} : <i1>
    %39 = br %28 {handshake.bb = 4 : ui32, handshake.name = "br8"} : <i32>
    %40 = br %result_30 {handshake.bb = 4 : ui32, handshake.name = "br9"} : <>
    %result_36, %index_37 = control_merge [%falseResult_3]  {handshake.bb = 5 : ui32, handshake.name = "control_merge4"} : [<>] to <>, <i1>
    end {handshake.bb = 5 : ui32, handshake.name = "end0"} %0#3, %arg3 : <>, <>
  }
}

