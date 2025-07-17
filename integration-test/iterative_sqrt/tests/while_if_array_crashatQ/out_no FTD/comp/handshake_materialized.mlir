module {
  handshake.func @iterative_sqrt(%arg0: memref<10xi32>, %arg1: !handshake.control<>, %arg2: !handshake.control<>, ...) -> (!handshake.channel<i32>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["A", "A_start", "start"], resNames = ["out0", "A_end", "end"]} {
    %0:2 = fork [2] %arg2 {handshake.bb = 0 : ui32, handshake.name = "fork0"} : <>
    %outputs:3, %memEnd = mem_controller[%arg0 : memref<10xi32>] %arg1 (%addressResult, %addressResult_4, %36, %25, %1#1, %1#2, %1#3) %44#1 {connectedBlocks = [1 : i32, 2 : i32, 4 : i32, 3 : i32], handshake.name = "mem_controller1"} :    (!handshake.channel<i4>, !handshake.channel<i4>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i4>, !handshake.channel<i4>, !handshake.channel<i32>) -> (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>)
    %1:4 = lsq[MC] (%22#1, %addressResult_12, %dataResult_13, %33#1, %addressResult_16, %dataResult_17, %44#2, %addressResult_20, %outputs#2)  {groupSizes = [1 : i32, 1 : i32, 1 : i32], handshake.name = "lsq1"} : (!handshake.control<>, !handshake.channel<i4>, !handshake.channel<i32>, !handshake.control<>, !handshake.channel<i4>, !handshake.channel<i32>, !handshake.control<>, !handshake.channel<i4>, !handshake.channel<i32>) -> (!handshake.channel<i32>, !handshake.channel<i4>, !handshake.channel<i4>, !handshake.channel<i32>)
    %2 = br %0#1 {handshake.bb = 0 : ui32, handshake.name = "br2"} : <>
    %result, %index = control_merge [%2, %31, %43]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>, <>] to <>, <i2>
    sink %index {handshake.name = "sink0"} : <i2>
    %3:2 = fork [2] %result {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <>
    %4 = constant %3#1 {handshake.bb = 1 : ui32, handshake.name = "constant4", value = false} : <>, <i1>
    %5 = extui %4 {handshake.bb = 1 : ui32, handshake.name = "extui0"} : <i1> to <i4>
    %6 = source {handshake.bb = 1 : ui32, handshake.name = "source0"} : <>
    %7 = constant %6 {handshake.bb = 1 : ui32, handshake.name = "constant14", value = 10 : i5} : <>, <i5>
    %8 = extsi %7 {handshake.bb = 1 : ui32, handshake.name = "extsi1"} : <i5> to <i32>
    %addressResult, %dataResult = load[%5] %outputs#0 {handshake.bb = 1 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i4>, <i32>, <i4>, <i32>
    %9:2 = fork [2] %dataResult {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <i32>
    %10 = cmpi sgt, %9#1, %8 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i32>
    %11:2 = fork [2] %10 {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <i1>
    %trueResult, %falseResult = cond_br %11#1, %9#0 {handshake.bb = 1 : ui32, handshake.name = "cond_br2"} : <i1>, <i32>
    sink %falseResult {handshake.name = "sink1"} : <i32>
    %trueResult_0, %falseResult_1 = cond_br %11#0, %3#0 {handshake.bb = 1 : ui32, handshake.name = "cond_br3"} : <i1>, <>
    %12 = merge %trueResult {handshake.bb = 2 : ui32, handshake.name = "merge0"} : <i32>
    %result_2, %index_3 = control_merge [%trueResult_0]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>] to <>, <i1>
    sink %index_3 {handshake.name = "sink2"} : <i1>
    %13:2 = fork [2] %result_2 {handshake.bb = 2 : ui32, handshake.name = "fork4"} : <>
    %14 = constant %13#1 {handshake.bb = 2 : ui32, handshake.name = "constant15", value = 1 : i2} : <>, <i2>
    %15 = extui %14 {handshake.bb = 2 : ui32, handshake.name = "extui1"} : <i2> to <i4>
    %16 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %17 = constant %16 {handshake.bb = 2 : ui32, handshake.name = "constant16", value = 10 : i5} : <>, <i5>
    %18 = extsi %17 {handshake.bb = 2 : ui32, handshake.name = "extsi3"} : <i5> to <i32>
    %addressResult_4, %dataResult_5 = load[%15] %outputs#1 {handshake.bb = 2 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i4>, <i32>, <i4>, <i32>
    %19 = cmpi slt, %dataResult_5, %18 {handshake.bb = 2 : ui32, handshake.name = "cmpi1"} : <i32>
    %20:2 = fork [2] %19 {handshake.bb = 2 : ui32, handshake.name = "fork5"} : <i1>
    %trueResult_6, %falseResult_7 = cond_br %20#1, %12 {handshake.bb = 2 : ui32, handshake.name = "cond_br4"} : <i1>, <i32>
    %trueResult_8, %falseResult_9 = cond_br %20#0, %13#0 {handshake.bb = 2 : ui32, handshake.name = "cond_br5"} : <i1>, <>
    %21 = merge %trueResult_6 {handshake.bb = 3 : ui32, handshake.name = "merge1"} : <i32>
    %result_10, %index_11 = control_merge [%trueResult_8]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>] to <>, <i1>
    sink %index_11 {handshake.name = "sink3"} : <i1>
    %22:3 = lazy_fork [3] %result_10 {handshake.bb = 3 : ui32, handshake.name = "lazy_fork0"} : <>
    %23:2 = fork [2] %22#2 {handshake.bb = 3 : ui32, handshake.name = "fork6"} : <>
    %24 = constant %23#1 {handshake.bb = 3 : ui32, handshake.name = "constant17", value = 1 : i2} : <>, <i2>
    %25 = extsi %24 {handshake.bb = 3 : ui32, handshake.name = "extsi4"} : <i2> to <i32>
    %26 = constant %23#0 {handshake.bb = 3 : ui32, handshake.name = "constant18", value = false} : <>, <i1>
    %27 = extui %26 {handshake.bb = 3 : ui32, handshake.name = "extui2"} : <i1> to <i4>
    %28 = source {handshake.bb = 3 : ui32, handshake.name = "source2"} : <>
    %29 = constant %28 {handshake.bb = 3 : ui32, handshake.name = "constant10", value = -1 : i32} : <>, <i32>
    %30 = addi %21, %29 {handshake.bb = 3 : ui32, handshake.name = "addi0"} : <i32>
    %addressResult_12, %dataResult_13 = store[%27] %30 {handshake.bb = 3 : ui32, handshake.deps = #handshake<deps[<"load2" (1)>]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 0>, handshake.name = "store0"} : <i4>, <i32>, <i4>, <i32>
    %31 = br %22#0 {handshake.bb = 3 : ui32, handshake.name = "br4"} : <>
    %32 = merge %falseResult_7 {handshake.bb = 4 : ui32, handshake.name = "merge2"} : <i32>
    %result_14, %index_15 = control_merge [%falseResult_9]  {handshake.bb = 4 : ui32, handshake.name = "control_merge3"} : [<>] to <>, <i1>
    sink %index_15 {handshake.name = "sink4"} : <i1>
    %33:3 = lazy_fork [3] %result_14 {handshake.bb = 4 : ui32, handshake.name = "lazy_fork1"} : <>
    %34:2 = fork [2] %33#2 {handshake.bb = 4 : ui32, handshake.name = "fork7"} : <>
    %35 = constant %34#1 {handshake.bb = 4 : ui32, handshake.name = "constant19", value = 1 : i2} : <>, <i2>
    %36 = extsi %35 {handshake.bb = 4 : ui32, handshake.name = "extsi6"} : <i2> to <i32>
    %37 = constant %34#0 {handshake.bb = 4 : ui32, handshake.name = "constant20", value = false} : <>, <i1>
    %38 = extui %37 {handshake.bb = 4 : ui32, handshake.name = "extui3"} : <i1> to <i4>
    %39 = source {handshake.bb = 4 : ui32, handshake.name = "source3"} : <>
    %40 = constant %39 {handshake.bb = 4 : ui32, handshake.name = "constant21", value = 1 : i2} : <>, <i2>
    %41 = extsi %40 {handshake.bb = 4 : ui32, handshake.name = "extsi8"} : <i2> to <i32>
    %42 = shrsi %32, %41 {handshake.bb = 4 : ui32, handshake.name = "shrsi0"} : <i32>
    %addressResult_16, %dataResult_17 = store[%38] %42 {handshake.bb = 4 : ui32, handshake.deps = #handshake<deps[<"load2" (1)>]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 1>, handshake.name = "store1"} : <i4>, <i32>, <i4>, <i32>
    %43 = br %33#0 {handshake.bb = 4 : ui32, handshake.name = "br5"} : <>
    %result_18, %index_19 = control_merge [%falseResult_1]  {handshake.bb = 5 : ui32, handshake.name = "control_merge4"} : [<>] to <>, <i1>
    sink %index_19 {handshake.name = "sink5"} : <i1>
    %44:3 = fork [3] %result_18 {handshake.bb = 5 : ui32, handshake.name = "fork8"} : <>
    %45 = constant %44#0 {handshake.bb = 5 : ui32, handshake.name = "constant22", value = false} : <>, <i1>
    %46 = extui %45 {handshake.bb = 5 : ui32, handshake.name = "extui4"} : <i1> to <i4>
    %addressResult_20, %dataResult_21 = load[%46] %1#0 {handshake.bb = 5 : ui32, handshake.mem_interface = #handshake.mem_interface<LSQ: 2>, handshake.name = "load2"} : <i4>, <i32>, <i4>, <i32>
    end {handshake.bb = 5 : ui32, handshake.name = "end0"} %dataResult_21, %memEnd, %0#0 : <i32>, <>, <>
  }
}

