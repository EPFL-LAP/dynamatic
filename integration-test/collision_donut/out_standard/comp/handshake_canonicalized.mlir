module {
  handshake.func @collision_donut(%arg0: memref<1000xi32>, %arg1: memref<1000xi32>, %arg2: !handshake.control<>, %arg3: !handshake.control<>, %arg4: !handshake.control<>, ...) -> (!handshake.channel<i32>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["x", "y", "x_start", "y_start", "start"], handshake.cfdfcThroughput = #handshake<cfdfcThroughput {"0" = 0.16666666666666666 : f64}>, handshake.cfdfcToBBList = #handshake<cfdfcToBBList {"0" = [1 : ui32, 2 : ui32, 3 : ui32]}>, resNames = ["out0", "x_end", "y_end", "end"]} {
    %0:3 = fork [3] %arg4 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork0"} : <>
    %outputs, %memEnd = mem_controller[%arg1 : memref<1000xi32>] %arg3 (%addressResult_2) %60#1 {connectedBlocks = [1 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller2"} :    (!handshake.channel<i10>) -> !handshake.channel<i32>
    %outputs_0, %memEnd_1 = mem_controller[%arg0 : memref<1000xi32>] %arg2 (%addressResult) %60#0 {connectedBlocks = [1 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller3"} :    (!handshake.channel<i10>) -> !handshake.channel<i32>
    %1 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant0", value = false} : <>, <i1>
    %2 = extsi %1 {handshake.bb = 0 : ui32, handshake.name = "extsi7"} : <i1> to <i11>
    %3 = mux %index [%2, %48] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i11>, <i11>] to <i11>
    %4 = buffer %3, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer0"} : <i11>
    %5:3 = fork [3] %4 {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <i11>
    %6 = trunci %5#0 {handshake.bb = 1 : ui32, handshake.name = "trunci0"} : <i11> to <i10>
    %7 = trunci %5#1 {handshake.bb = 1 : ui32, handshake.name = "trunci1"} : <i11> to <i10>
    %result, %index = control_merge [%0#2, %trueResult_18]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %8 = buffer %result, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer2"} : <>
    %9:2 = fork [2] %8 {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <>
    %10 = source {handshake.bb = 1 : ui32, handshake.name = "source0"} : <>
    %11 = constant %10 {handshake.bb = 1 : ui32, handshake.name = "constant1", value = 4 : i4} : <>, <i4>
    %12 = extsi %11 {handshake.bb = 1 : ui32, handshake.name = "extsi1"} : <i4> to <i32>
    %13 = constant %9#1 {handshake.bb = 1 : ui32, handshake.name = "constant8", value = -1 : i32} : <>, <i32>
    %addressResult, %dataResult = load[%7] %outputs_0 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i10>, <i32>, <i10>, <i32>
    %14:2 = fork [2] %dataResult {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <i32>
    %addressResult_2, %dataResult_3 = load[%6] %outputs {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i10>, <i32>, <i10>, <i32>
    %15:2 = fork [2] %dataResult_3 {handshake.bb = 1 : ui32, handshake.name = "fork4"} : <i32>
    %16 = muli %14#0, %14#1 {handshake.bb = 1 : ui32, handshake.name = "muli0"} : <i32>
    %17 = muli %15#0, %15#1 {handshake.bb = 1 : ui32, handshake.name = "muli1"} : <i32>
    %18 = addi %16, %17 {handshake.bb = 1 : ui32, handshake.name = "addi0"} : <i32>
    %19:2 = fork [2] %18 {handshake.bb = 1 : ui32, handshake.name = "fork5"} : <i32>
    %20 = cmpi ult, %19#1, %12 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i32>
    %21:4 = fork [4] %20 {handshake.bb = 1 : ui32, handshake.name = "fork6"} : <i1>
    %22 = buffer %5#2, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer1"} : <i11>
    %trueResult, %falseResult = cond_br %21#0, %22 {handshake.bb = 1 : ui32, handshake.name = "cond_br3"} : <i1>, <i11>
    %23 = extsi %trueResult {handshake.bb = 1 : ui32, handshake.name = "extsi8"} : <i11> to <i12>
    %trueResult_4, %falseResult_5 = cond_br %21#3, %13 {handshake.bb = 1 : ui32, handshake.name = "cond_br4"} : <i1>, <i32>
    sink %falseResult_5 {handshake.name = "sink0"} : <i32>
    %trueResult_6, %falseResult_7 = cond_br %21#2, %9#0 {handshake.bb = 1 : ui32, handshake.name = "cond_br5"} : <i1>, <>
    %trueResult_8, %falseResult_9 = cond_br %21#1, %19#0 {handshake.bb = 1 : ui32, handshake.name = "cond_br6"} : <i1>, <i32>
    sink %trueResult_8 {handshake.name = "sink1"} : <i32>
    %24 = buffer %falseResult_7, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer4"} : <>
    %25:2 = fork [2] %24 {handshake.bb = 2 : ui32, handshake.name = "fork7"} : <>
    %26 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %27 = constant %26 {handshake.bb = 2 : ui32, handshake.name = "constant2", value = 19000 : i16} : <>, <i16>
    %28 = extsi %27 {handshake.bb = 2 : ui32, handshake.name = "extsi2"} : <i16> to <i32>
    %29 = constant %25#1 {handshake.bb = 2 : ui32, handshake.name = "constant10", value = -2 : i32} : <>, <i32>
    %30 = cmpi ugt, %falseResult_9, %28 {handshake.bb = 2 : ui32, handshake.name = "cmpi1"} : <i32>
    %31 = buffer %30, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer5"} : <i1>
    %32:3 = fork [3] %31 {handshake.bb = 2 : ui32, handshake.name = "fork8"} : <i1>
    %33 = buffer %falseResult, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer3"} : <i11>
    %trueResult_10, %falseResult_11 = cond_br %32#0, %33 {handshake.bb = 2 : ui32, handshake.name = "cond_br7"} : <i1>, <i11>
    %34 = extsi %trueResult_10 {handshake.bb = 2 : ui32, handshake.name = "extsi9"} : <i11> to <i12>
    %trueResult_12, %falseResult_13 = cond_br %32#2, %29 {handshake.bb = 2 : ui32, handshake.name = "cond_br8"} : <i1>, <i32>
    sink %falseResult_13 {handshake.name = "sink3"} : <i32>
    %trueResult_14, %falseResult_15 = cond_br %32#1, %25#0 {handshake.bb = 2 : ui32, handshake.name = "cond_br9"} : <i1>, <>
    %35 = extsi %falseResult_11 {handshake.bb = 3 : ui32, handshake.name = "extsi10"} : <i11> to <i12>
    %36:2 = fork [2] %falseResult_15 {handshake.bb = 3 : ui32, handshake.name = "fork9"} : <>
    %37 = constant %36#0 {handshake.bb = 3 : ui32, handshake.name = "constant3", value = false} : <>, <i1>
    %38 = source {handshake.bb = 3 : ui32, handshake.name = "source2"} : <>
    %39 = constant %38 {handshake.bb = 3 : ui32, handshake.name = "constant4", value = 1 : i2} : <>, <i2>
    %40 = extsi %39 {handshake.bb = 3 : ui32, handshake.name = "extsi11"} : <i2> to <i12>
    %41 = source {handshake.bb = 3 : ui32, handshake.name = "source3"} : <>
    %42 = constant %41 {handshake.bb = 3 : ui32, handshake.name = "constant5", value = 1000 : i11} : <>, <i11>
    %43 = extsi %42 {handshake.bb = 3 : ui32, handshake.name = "extsi12"} : <i11> to <i12>
    %44 = addi %35, %40 {handshake.bb = 3 : ui32, handshake.name = "addi1"} : <i12>
    %45:2 = fork [2] %44 {handshake.bb = 3 : ui32, handshake.name = "fork10"} : <i12>
    %46 = cmpi ult, %45#1, %43 {handshake.bb = 3 : ui32, handshake.name = "cmpi2"} : <i12>
    %47:3 = fork [3] %46 {handshake.bb = 3 : ui32, handshake.name = "fork11"} : <i1>
    %trueResult_16, %falseResult_17 = cond_br %47#0, %45#0 {handshake.bb = 3 : ui32, handshake.name = "cond_br10"} : <i1>, <i12>
    %48 = trunci %trueResult_16 {handshake.bb = 3 : ui32, handshake.name = "trunci2"} : <i12> to <i11>
    %trueResult_18, %falseResult_19 = cond_br %47#1, %36#1 {handshake.bb = 3 : ui32, handshake.name = "cond_br11"} : <i1>, <>
    %trueResult_20, %falseResult_21 = cond_br %47#2, %37 {handshake.bb = 3 : ui32, handshake.name = "cond_br12"} : <i1>, <i1>
    sink %trueResult_20 {handshake.name = "sink5"} : <i1>
    %49 = extsi %falseResult_21 {handshake.bb = 3 : ui32, handshake.name = "extsi13"} : <i1> to <i32>
    %50 = mux %52#0 [%23, %falseResult_17] {handshake.bb = 4 : ui32, handshake.name = "mux1"} : <i1>, [<i12>, <i12>] to <i12>
    %51 = mux %52#1 [%trueResult_4, %49] {handshake.bb = 4 : ui32, handshake.name = "mux2"} : <i1>, [<i32>, <i32>] to <i32>
    %result_22, %index_23 = control_merge [%trueResult_6, %falseResult_19]  {handshake.bb = 4 : ui32, handshake.name = "control_merge3"} : [<>, <>] to <>, <i1>
    %52:2 = fork [2] %index_23 {handshake.bb = 4 : ui32, handshake.name = "fork12"} : <i1>
    %53 = buffer %50, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 5 : ui32, handshake.name = "buffer6"} : <i12>
    %54 = mux %59#0 [%34, %53] {handshake.bb = 5 : ui32, handshake.name = "mux3"} : <i1>, [<i12>, <i12>] to <i12>
    %55 = extsi %54 {handshake.bb = 5 : ui32, handshake.name = "extsi14"} : <i12> to <i32>
    %56 = buffer %51, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 5 : ui32, handshake.name = "buffer7"} : <i32>
    %57 = mux %59#1 [%trueResult_12, %56] {handshake.bb = 5 : ui32, handshake.name = "mux4"} : <i1>, [<i32>, <i32>] to <i32>
    %result_24, %index_25 = control_merge [%trueResult_14, %result_22]  {handshake.bb = 5 : ui32, handshake.name = "control_merge4"} : [<>, <>] to <>, <i1>
    %58 = buffer %index_25, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 5 : ui32, handshake.name = "buffer8"} : <i1>
    %59:2 = fork [2] %58 {handshake.bb = 5 : ui32, handshake.name = "fork13"} : <i1>
    %60:2 = fork [2] %result_24 {handshake.bb = 5 : ui32, handshake.name = "fork14"} : <>
    %61 = source {handshake.bb = 5 : ui32, handshake.name = "source4"} : <>
    %62 = constant %61 {handshake.bb = 5 : ui32, handshake.name = "constant15", value = 1 : i2} : <>, <i2>
    %63 = extsi %62 {handshake.bb = 5 : ui32, handshake.name = "extsi6"} : <i2> to <i32>
    %64 = shli %55, %63 {handshake.bb = 5 : ui32, handshake.name = "shli0"} : <i32>
    %65 = andi %64, %57 {handshake.bb = 5 : ui32, handshake.name = "andi0"} : <i32>
    end {handshake.bb = 5 : ui32, handshake.name = "end0"} %65, %memEnd_1, %memEnd, %0#1 : <i32>, <>, <>, <>
  }
}

