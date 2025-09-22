module {
  handshake.func @collision_donut(%arg0: memref<1000xi32>, %arg1: memref<1000xi32>, %arg2: !handshake.control<>, %arg3: !handshake.control<>, %arg4: !handshake.control<>, ...) -> (!handshake.channel<i32>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["x", "y", "x_start", "y_start", "start"], resNames = ["out0", "x_end", "y_end", "end"]} {
    %0:3 = fork [3] %arg4 {handshake.bb = 0 : ui32, handshake.name = "fork0"} : <>
    %outputs, %memEnd = mem_controller[%arg1 : memref<1000xi32>] %arg3 (%addressResult_2) %51#1 {connectedBlocks = [1 : i32], handshake.name = "mem_controller2"} :    (!handshake.channel<i10>) -> !handshake.channel<i32>
    %outputs_0, %memEnd_1 = mem_controller[%arg0 : memref<1000xi32>] %arg2 (%addressResult) %51#0 {connectedBlocks = [1 : i32], handshake.name = "mem_controller3"} :    (!handshake.channel<i10>) -> !handshake.channel<i32>
    %1 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant0", value = false} : <>, <i1>
    %2 = extsi %1 {handshake.bb = 0 : ui32, handshake.name = "extsi7"} : <i1> to <i11>
    %3 = mux %index [%2, %42] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i11>, <i11>] to <i11>
    %4:3 = fork [3] %3 {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <i11>
    %5 = trunci %4#0 {handshake.bb = 1 : ui32, handshake.name = "trunci0"} : <i11> to <i10>
    %6 = trunci %4#1 {handshake.bb = 1 : ui32, handshake.name = "trunci1"} : <i11> to <i10>
    %result, %index = control_merge [%0#2, %trueResult_18]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %7:2 = fork [2] %result {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <>
    %8 = source {handshake.bb = 1 : ui32, handshake.name = "source0"} : <>
    %9 = constant %8 {handshake.bb = 1 : ui32, handshake.name = "constant1", value = 4 : i4} : <>, <i4>
    %10 = extsi %9 {handshake.bb = 1 : ui32, handshake.name = "extsi1"} : <i4> to <i32>
    %11 = constant %7#1 {handshake.bb = 1 : ui32, handshake.name = "constant8", value = -1 : i32} : <>, <i32>
    %addressResult, %dataResult = load[%6] %outputs_0 {handshake.bb = 1 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i10>, <i32>, <i10>, <i32>
    %12:2 = fork [2] %dataResult {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <i32>
    %addressResult_2, %dataResult_3 = load[%5] %outputs {handshake.bb = 1 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i10>, <i32>, <i10>, <i32>
    %13:2 = fork [2] %dataResult_3 {handshake.bb = 1 : ui32, handshake.name = "fork4"} : <i32>
    %14 = muli %12#0, %12#1 {handshake.bb = 1 : ui32, handshake.name = "muli0"} : <i32>
    %15 = muli %13#0, %13#1 {handshake.bb = 1 : ui32, handshake.name = "muli1"} : <i32>
    %16 = addi %14, %15 {handshake.bb = 1 : ui32, handshake.name = "addi0"} : <i32>
    %17:2 = fork [2] %16 {handshake.bb = 1 : ui32, handshake.name = "fork5"} : <i32>
    %18 = cmpi ult, %17#1, %10 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i32>
    %19:4 = fork [4] %18 {handshake.bb = 1 : ui32, handshake.name = "fork6"} : <i1>
    %trueResult, %falseResult = cond_br %19#0, %4#2 {handshake.bb = 1 : ui32, handshake.name = "cond_br3"} : <i1>, <i11>
    %20 = extsi %trueResult {handshake.bb = 1 : ui32, handshake.name = "extsi8"} : <i11> to <i12>
    %trueResult_4, %falseResult_5 = cond_br %19#3, %11 {handshake.bb = 1 : ui32, handshake.name = "cond_br4"} : <i1>, <i32>
    sink %falseResult_5 {handshake.name = "sink0"} : <i32>
    %trueResult_6, %falseResult_7 = cond_br %19#2, %7#0 {handshake.bb = 1 : ui32, handshake.name = "cond_br5"} : <i1>, <>
    %trueResult_8, %falseResult_9 = cond_br %19#1, %17#0 {handshake.bb = 1 : ui32, handshake.name = "cond_br6"} : <i1>, <i32>
    sink %trueResult_8 {handshake.name = "sink1"} : <i32>
    %21:2 = fork [2] %falseResult_7 {handshake.bb = 2 : ui32, handshake.name = "fork7"} : <>
    %22 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %23 = constant %22 {handshake.bb = 2 : ui32, handshake.name = "constant2", value = 19000 : i16} : <>, <i16>
    %24 = extsi %23 {handshake.bb = 2 : ui32, handshake.name = "extsi2"} : <i16> to <i32>
    %25 = constant %21#1 {handshake.bb = 2 : ui32, handshake.name = "constant10", value = -2 : i32} : <>, <i32>
    %26 = cmpi ugt, %falseResult_9, %24 {handshake.bb = 2 : ui32, handshake.name = "cmpi1"} : <i32>
    %27:3 = fork [3] %26 {handshake.bb = 2 : ui32, handshake.name = "fork8"} : <i1>
    %trueResult_10, %falseResult_11 = cond_br %27#0, %falseResult {handshake.bb = 2 : ui32, handshake.name = "cond_br7"} : <i1>, <i11>
    %28 = extsi %trueResult_10 {handshake.bb = 2 : ui32, handshake.name = "extsi9"} : <i11> to <i12>
    %trueResult_12, %falseResult_13 = cond_br %27#2, %25 {handshake.bb = 2 : ui32, handshake.name = "cond_br8"} : <i1>, <i32>
    sink %falseResult_13 {handshake.name = "sink3"} : <i32>
    %trueResult_14, %falseResult_15 = cond_br %27#1, %21#0 {handshake.bb = 2 : ui32, handshake.name = "cond_br9"} : <i1>, <>
    %29 = extsi %falseResult_11 {handshake.bb = 3 : ui32, handshake.name = "extsi10"} : <i11> to <i12>
    %30:2 = fork [2] %falseResult_15 {handshake.bb = 3 : ui32, handshake.name = "fork9"} : <>
    %31 = constant %30#0 {handshake.bb = 3 : ui32, handshake.name = "constant3", value = false} : <>, <i1>
    %32 = source {handshake.bb = 3 : ui32, handshake.name = "source2"} : <>
    %33 = constant %32 {handshake.bb = 3 : ui32, handshake.name = "constant4", value = 1 : i2} : <>, <i2>
    %34 = extsi %33 {handshake.bb = 3 : ui32, handshake.name = "extsi11"} : <i2> to <i12>
    %35 = source {handshake.bb = 3 : ui32, handshake.name = "source3"} : <>
    %36 = constant %35 {handshake.bb = 3 : ui32, handshake.name = "constant5", value = 1000 : i11} : <>, <i11>
    %37 = extsi %36 {handshake.bb = 3 : ui32, handshake.name = "extsi12"} : <i11> to <i12>
    %38 = addi %29, %34 {handshake.bb = 3 : ui32, handshake.name = "addi1"} : <i12>
    %39:2 = fork [2] %38 {handshake.bb = 3 : ui32, handshake.name = "fork10"} : <i12>
    %40 = cmpi ult, %39#1, %37 {handshake.bb = 3 : ui32, handshake.name = "cmpi2"} : <i12>
    %41:3 = fork [3] %40 {handshake.bb = 3 : ui32, handshake.name = "fork11"} : <i1>
    %trueResult_16, %falseResult_17 = cond_br %41#0, %39#0 {handshake.bb = 3 : ui32, handshake.name = "cond_br10"} : <i1>, <i12>
    %42 = trunci %trueResult_16 {handshake.bb = 3 : ui32, handshake.name = "trunci2"} : <i12> to <i11>
    %trueResult_18, %falseResult_19 = cond_br %41#1, %30#1 {handshake.bb = 3 : ui32, handshake.name = "cond_br11"} : <i1>, <>
    %trueResult_20, %falseResult_21 = cond_br %41#2, %31 {handshake.bb = 3 : ui32, handshake.name = "cond_br12"} : <i1>, <i1>
    sink %trueResult_20 {handshake.name = "sink5"} : <i1>
    %43 = extsi %falseResult_21 {handshake.bb = 3 : ui32, handshake.name = "extsi13"} : <i1> to <i32>
    %44 = mux %46#0 [%20, %falseResult_17] {handshake.bb = 4 : ui32, handshake.name = "mux1"} : <i1>, [<i12>, <i12>] to <i12>
    %45 = mux %46#1 [%trueResult_4, %43] {handshake.bb = 4 : ui32, handshake.name = "mux2"} : <i1>, [<i32>, <i32>] to <i32>
    %result_22, %index_23 = control_merge [%trueResult_6, %falseResult_19]  {handshake.bb = 4 : ui32, handshake.name = "control_merge3"} : [<>, <>] to <>, <i1>
    %46:2 = fork [2] %index_23 {handshake.bb = 4 : ui32, handshake.name = "fork12"} : <i1>
    %47 = mux %50#0 [%28, %44] {handshake.bb = 5 : ui32, handshake.name = "mux3"} : <i1>, [<i12>, <i12>] to <i12>
    %48 = extsi %47 {handshake.bb = 5 : ui32, handshake.name = "extsi14"} : <i12> to <i32>
    %49 = mux %50#1 [%trueResult_12, %45] {handshake.bb = 5 : ui32, handshake.name = "mux4"} : <i1>, [<i32>, <i32>] to <i32>
    %result_24, %index_25 = control_merge [%trueResult_14, %result_22]  {handshake.bb = 5 : ui32, handshake.name = "control_merge4"} : [<>, <>] to <>, <i1>
    %50:2 = fork [2] %index_25 {handshake.bb = 5 : ui32, handshake.name = "fork13"} : <i1>
    %51:2 = fork [2] %result_24 {handshake.bb = 5 : ui32, handshake.name = "fork14"} : <>
    %52 = source {handshake.bb = 5 : ui32, handshake.name = "source4"} : <>
    %53 = constant %52 {handshake.bb = 5 : ui32, handshake.name = "constant15", value = 1 : i2} : <>, <i2>
    %54 = extsi %53 {handshake.bb = 5 : ui32, handshake.name = "extsi6"} : <i2> to <i32>
    %55 = shli %48, %54 {handshake.bb = 5 : ui32, handshake.name = "shli0"} : <i32>
    %56 = andi %55, %49 {handshake.bb = 5 : ui32, handshake.name = "andi0"} : <i32>
    end {handshake.bb = 5 : ui32, handshake.name = "end0"} %56, %memEnd_1, %memEnd, %0#1 : <i32>, <>, <>, <>
  }
}

