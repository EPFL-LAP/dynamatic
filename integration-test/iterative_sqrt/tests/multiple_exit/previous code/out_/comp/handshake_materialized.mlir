module {
  handshake.func @multiple_exit(%arg0: memref<10xi32>, %arg1: !handshake.channel<i32>, %arg2: !handshake.control<>, %arg3: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>) attributes {argNames = ["arr", "size", "arr_start", "start"], resNames = ["arr_end", "end"]} {
    %0:4 = fork [4] %arg3 {handshake.bb = 0 : ui32, handshake.name = "fork0"} : <>
    %1:4 = lsq[%arg0 : memref<10xi32>] (%arg2, %23#1, %addressResult, %34#1, %addressResult_18, %44#1, %addressResult_32, %addressResult_34, %dataResult_35, %result_36)  {groupSizes = [1 : i32, 1 : i32, 2 : i32], handshake.name = "lsq1"} : (!handshake.control<>, !handshake.control<>, !handshake.channel<i4>, !handshake.control<>, !handshake.channel<i4>, !handshake.control<>, !handshake.channel<i4>, !handshake.channel<i4>, !handshake.channel<i32>, !handshake.control<>) -> (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>)
    %2 = constant %0#3 {handshake.bb = 0 : ui32, handshake.name = "constant1", value = false} : <>, <i1>
    %3 = constant %0#2 {handshake.bb = 0 : ui32, handshake.name = "constant4", value = true} : <>, <i1>
    %4 = br %2 {handshake.bb = 0 : ui32, handshake.name = "br1"} : <i1>
    %5 = extsi %4 {handshake.bb = 0 : ui32, handshake.name = "extsi3"} : <i1> to <i32>
    %6 = br %3 {handshake.bb = 0 : ui32, handshake.name = "br2"} : <i1>
    %7 = br %arg1 {handshake.bb = 0 : ui32, handshake.name = "br3"} : <i32>
    %8 = br %0#1 {handshake.bb = 0 : ui32, handshake.name = "br4"} : <>
    %9 = mux %14#0 [%5, %falseResult_9, %falseResult_23, %53] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i2>, [<i32>, <i32>, <i32>, <i32>] to <i32>
    %10:2 = fork [2] %9 {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <i32>
    %11 = mux %14#1 [%6, %falseResult_15, %falseResult_29, %54] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i2>, [<i1>, <i1>, <i1>, <i1>] to <i1>
    %12 = mux %14#2 [%7, %falseResult_7, %falseResult_21, %55] {handshake.bb = 1 : ui32, handshake.name = "mux2"} : <i2>, [<i32>, <i32>, <i32>, <i32>] to <i32>
    %13:2 = fork [2] %12 {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <i32>
    %result, %index = control_merge [%8, %falseResult_13, %falseResult_27, %56]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>, <>, <>] to <>, <i2>
    %14:3 = fork [3] %index {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <i2>
    %15 = cmpi slt, %10#1, %13#1 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i32>
    %16 = andi %15, %11 {handshake.bb = 1 : ui32, handshake.name = "andi0"} : <i1>
    %17:3 = fork [3] %16 {handshake.bb = 1 : ui32, handshake.name = "fork4"} : <i1>
    %trueResult, %falseResult = cond_br %17#2, %10#0 {handshake.bb = 1 : ui32, handshake.name = "cond_br3"} : <i1>, <i32>
    sink %falseResult {handshake.name = "sink0"} : <i32>
    %trueResult_0, %falseResult_1 = cond_br %17#1, %13#0 {handshake.bb = 1 : ui32, handshake.name = "cond_br4"} : <i1>, <i32>
    sink %falseResult_1 {handshake.name = "sink1"} : <i32>
    %trueResult_2, %falseResult_3 = cond_br %17#0, %result {handshake.bb = 1 : ui32, handshake.name = "cond_br5"} : <i1>, <>
    %18 = merge %trueResult {handshake.bb = 2 : ui32, handshake.name = "merge0"} : <i32>
    %19:3 = fork [3] %18 {handshake.bb = 2 : ui32, handshake.name = "fork5"} : <i32>
    %20 = trunci %19#2 {handshake.bb = 2 : ui32, handshake.name = "trunci0"} : <i32> to <i4>
    %21 = trunci %19#1 {handshake.bb = 2 : ui32, handshake.name = "trunci1"} : <i32> to <i4>
    %22 = merge %trueResult_0 {handshake.bb = 2 : ui32, handshake.name = "merge1"} : <i32>
    %result_4, %index_5 = control_merge [%trueResult_2]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>] to <>, <i1>
    sink %index_5 {handshake.name = "sink2"} : <i1>
    %23:3 = lazy_fork [3] %result_4 {handshake.bb = 2 : ui32, handshake.name = "lazy_fork0"} : <>
    %24 = fork [1] %23#2 {handshake.bb = 2 : ui32, handshake.name = "fork6"} : <>
    %25 = constant %24 {handshake.bb = 2 : ui32, handshake.name = "constant5", value = false} : <>, <i1>
    %26 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %27 = constant %26 {handshake.bb = 2 : ui32, handshake.name = "constant6", value = -1 : i32} : <>, <i32>
    %addressResult, %dataResult = load[%21] %1#0 {handshake.bb = 2 : ui32, handshake.deps = #handshake<deps[<"store1" (0)>]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 0>, handshake.name = "load3"} : <i4>, <i32>, <i4>, <i32>
    %28 = cmpi ne, %dataResult, %27 {handshake.bb = 2 : ui32, handshake.name = "cmpi1"} : <i32>
    %29:5 = fork [5] %28 {handshake.bb = 2 : ui32, handshake.name = "fork7"} : <i1>
    %trueResult_6, %falseResult_7 = cond_br %29#4, %22 {handshake.bb = 2 : ui32, handshake.name = "cond_br6"} : <i1>, <i32>
    %trueResult_8, %falseResult_9 = cond_br %29#3, %19#0 {handshake.bb = 2 : ui32, handshake.name = "cond_br7"} : <i1>, <i32>
    %trueResult_10, %falseResult_11 = cond_br %29#2, %20 {handshake.bb = 2 : ui32, handshake.name = "cond_br8"} : <i1>, <i4>
    sink %falseResult_11 {handshake.name = "sink3"} : <i4>
    %trueResult_12, %falseResult_13 = cond_br %29#1, %23#0 {handshake.bb = 2 : ui32, handshake.name = "cond_br9"} : <i1>, <>
    %trueResult_14, %falseResult_15 = cond_br %29#0, %25 {handshake.bb = 2 : ui32, handshake.name = "cond_br10"} : <i1>, <i1>
    sink %trueResult_14 {handshake.name = "sink4"} : <i1>
    %30 = merge %trueResult_6 {handshake.bb = 3 : ui32, handshake.name = "merge2"} : <i32>
    %31 = merge %trueResult_8 {handshake.bb = 3 : ui32, handshake.name = "merge3"} : <i32>
    %32 = merge %trueResult_10 {handshake.bb = 3 : ui32, handshake.name = "merge4"} : <i4>
    %33:2 = fork [2] %32 {handshake.bb = 3 : ui32, handshake.name = "fork8"} : <i4>
    %result_16, %index_17 = control_merge [%trueResult_12]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>] to <>, <i1>
    sink %index_17 {handshake.name = "sink5"} : <i1>
    %34:2 = lazy_fork [2] %result_16 {handshake.bb = 3 : ui32, handshake.name = "lazy_fork1"} : <>
    %35 = source {handshake.bb = 3 : ui32, handshake.name = "source1"} : <>
    %36 = constant %35 {handshake.bb = 3 : ui32, handshake.name = "constant2", value = false} : <>, <i1>
    %37 = extsi %36 {handshake.bb = 3 : ui32, handshake.name = "extsi1"} : <i1> to <i32>
    %addressResult_18, %dataResult_19 = load[%33#1] %1#1 {handshake.bb = 3 : ui32, handshake.deps = #handshake<deps[<"store1" (0)>]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 1>, handshake.name = "load4"} : <i4>, <i32>, <i4>, <i32>
    %38 = cmpi ne, %dataResult_19, %37 {handshake.bb = 3 : ui32, handshake.name = "cmpi2"} : <i32>
    %39:6 = fork [6] %38 {handshake.bb = 3 : ui32, handshake.name = "fork9"} : <i1>
    %trueResult_20, %falseResult_21 = cond_br %39#5, %30 {handshake.bb = 3 : ui32, handshake.name = "cond_br11"} : <i1>, <i32>
    %trueResult_22, %falseResult_23 = cond_br %39#4, %31 {handshake.bb = 3 : ui32, handshake.name = "cond_br12"} : <i1>, <i32>
    %trueResult_24, %falseResult_25 = cond_br %39#3, %33#0 {handshake.bb = 3 : ui32, handshake.name = "cond_br13"} : <i1>, <i4>
    sink %falseResult_25 {handshake.name = "sink6"} : <i4>
    %trueResult_26, %falseResult_27 = cond_br %39#2, %34#0 {handshake.bb = 3 : ui32, handshake.name = "cond_br14"} : <i1>, <>
    %trueResult_28, %falseResult_29 = cond_br %39#0, %39#1 {handshake.bb = 3 : ui32, handshake.name = "cond_br15"} : <i1>, <i1>
    sink %trueResult_28 {handshake.name = "sink7"} : <i1>
    %40 = merge %trueResult_20 {handshake.bb = 4 : ui32, handshake.name = "merge5"} : <i32>
    %41 = merge %trueResult_22 {handshake.bb = 4 : ui32, handshake.name = "merge6"} : <i32>
    %42 = merge %trueResult_24 {handshake.bb = 4 : ui32, handshake.name = "merge7"} : <i4>
    %43:2 = fork [2] %42 {handshake.bb = 4 : ui32, handshake.name = "fork10"} : <i4>
    %result_30, %index_31 = control_merge [%trueResult_26]  {handshake.bb = 4 : ui32, handshake.name = "control_merge3"} : [<>] to <>, <i1>
    sink %index_31 {handshake.name = "sink8"} : <i1>
    %44:3 = lazy_fork [3] %result_30 {handshake.bb = 4 : ui32, handshake.name = "lazy_fork2"} : <>
    %45 = fork [1] %44#2 {handshake.bb = 4 : ui32, handshake.name = "fork11"} : <>
    %46 = constant %45 {handshake.bb = 4 : ui32, handshake.name = "constant8", value = true} : <>, <i1>
    %47 = source {handshake.bb = 4 : ui32, handshake.name = "source2"} : <>
    %48 = constant %47 {handshake.bb = 4 : ui32, handshake.name = "constant3", value = 1 : i2} : <>, <i2>
    %49 = extsi %48 {handshake.bb = 4 : ui32, handshake.name = "extsi2"} : <i2> to <i32>
    %50:2 = fork [2] %49 {handshake.bb = 4 : ui32, handshake.name = "fork12"} : <i32>
    %addressResult_32, %dataResult_33 = load[%43#1] %1#2 {handshake.bb = 4 : ui32, handshake.deps = #handshake<deps[<"store1" (0)>]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 2>, handshake.name = "load5"} : <i4>, <i32>, <i4>, <i32>
    %51 = addi %dataResult_33, %50#1 {handshake.bb = 4 : ui32, handshake.name = "addi0"} : <i32>
    %addressResult_34, %dataResult_35 = store[%43#0] %51 {handshake.bb = 4 : ui32, handshake.deps = #handshake<deps[<"load3" (0)>, <"load4" (0)>, <"load5" (0)>, <"store1" (0)>]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 2>, handshake.name = "store1"} : <i4>, <i32>, <i4>, <i32>
    %52 = addi %41, %50#0 {handshake.bb = 4 : ui32, handshake.name = "addi1"} : <i32>
    %53 = br %52 {handshake.bb = 4 : ui32, handshake.name = "br5"} : <i32>
    %54 = br %46 {handshake.bb = 4 : ui32, handshake.name = "br6"} : <i1>
    %55 = br %40 {handshake.bb = 4 : ui32, handshake.name = "br8"} : <i32>
    %56 = br %44#0 {handshake.bb = 4 : ui32, handshake.name = "br9"} : <>
    %result_36, %index_37 = control_merge [%falseResult_3]  {handshake.bb = 5 : ui32, handshake.name = "control_merge4"} : [<>] to <>, <i1>
    sink %index_37 {handshake.name = "sink9"} : <i1>
    end {handshake.bb = 5 : ui32, handshake.name = "end0"} %1#3, %0#0 : <>, <>
  }
}

