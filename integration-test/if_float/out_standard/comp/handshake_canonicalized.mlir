module {
  handshake.func @if_float(%arg0: !handshake.channel<f32>, %arg1: memref<100xf32>, %arg2: memref<100xf32>, %arg3: !handshake.control<>, %arg4: !handshake.control<>, %arg5: !handshake.control<>, ...) -> (!handshake.channel<f32>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["x0", "a", "minus_trace", "a_start", "minus_trace_start", "start"], handshake.cfdfcThroughput = #handshake<cfdfcThroughput {"0" = 0.038461538461538464 : f64, "1" = 0.023809523809523808 : f64}>, handshake.cfdfcToBBList = #handshake<cfdfcToBBList {"0" = [4 : ui32, 1 : ui32, 2 : ui32], "1" = [4 : ui32, 1 : ui32, 3 : ui32]}>, resNames = ["out0", "a_end", "minus_trace_end", "end"]} {
    %0:3 = fork [3] %arg5 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork0"} : <>
    %memEnd = mem_controller[%arg2 : memref<100xf32>] %arg4 (%33, %addressResult_5, %dataResult_6) %58#1 {connectedBlocks = [3 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller2"} :    (!handshake.channel<i32>, !handshake.channel<i7>, !handshake.channel<f32>) -> ()
    %outputs, %memEnd_0 = mem_controller[%arg1 : memref<100xf32>] %arg3 (%addressResult) %58#0 {connectedBlocks = [1 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller3"} :    (!handshake.channel<i7>) -> !handshake.channel<f32>
    %1 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant2", value = false} : <>, <i1>
    %2 = extsi %1 {handshake.bb = 0 : ui32, handshake.name = "extsi4"} : <i1> to <i8>
    %3 = mux %11#0 [%2, %trueResult_9] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i8>, <i8>] to <i8>
    %4 = buffer %3, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer0"} : <i8>
    %5:2 = fork [2] %4 {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <i8>
    %6 = trunci %5#0 {handshake.bb = 1 : ui32, handshake.name = "trunci0"} : <i8> to <i7>
    %7 = buffer %11#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer6"} : <i1>
    %8 = mux %7 [%arg0, %trueResult_11] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i1>, [<f32>, <f32>] to <f32>
    %9 = buffer %8, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer3"} : <f32>
    %10:3 = fork [3] %9 {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <f32>
    %result, %index = control_merge [%0#2, %trueResult_13]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %11:2 = fork [2] %index {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <i1>
    %12 = source {handshake.bb = 1 : ui32, handshake.name = "source0"} : <>
    %13 = constant %12 {handshake.bb = 1 : ui32, handshake.name = "constant6", value = -0.899999976 : f32} : <>, <f32>
    %14 = source {handshake.bb = 1 : ui32, handshake.name = "source1"} : <>
    %15 = constant %14 {handshake.bb = 1 : ui32, handshake.name = "constant7", value = 0.000000e+00 : f32} : <>, <f32>
    %16 = buffer %6, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer2"} : <i7>
    %addressResult, %dataResult = load[%16] %outputs {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i7>, <f32>, <i7>, <f32>
    %17 = mulf %dataResult, %10#2 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "mulf0", internal_delay = "2_875333"} : <f32>
    %18 = mulf %10#1, %13 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "mulf1", internal_delay = "2_875333"} : <f32>
    %19 = addf %17, %18 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "addf0", internal_delay = "2_922000"} : <f32>
    %20 = cmpf ugt, %19, %15 {handshake.bb = 1 : ui32, handshake.name = "cmpf0", internal_delay = "0_000000"} : <f32>
    %21:3 = fork [3] %20 {handshake.bb = 1 : ui32, handshake.name = "fork4"} : <i1>
    %22 = buffer %5#1, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer1"} : <i8>
    %trueResult, %falseResult = cond_br %21#0, %22 {handshake.bb = 1 : ui32, handshake.name = "cond_br2"} : <i1>, <i8>
    %23 = buffer %10#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer4"} : <f32>
    %trueResult_1, %falseResult_2 = cond_br %21#2, %23 {handshake.bb = 1 : ui32, handshake.name = "cond_br3"} : <i1>, <f32>
    %24 = buffer %result, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer5"} : <>
    %trueResult_3, %falseResult_4 = cond_br %21#1, %24 {handshake.bb = 1 : ui32, handshake.name = "cond_br4"} : <i1>, <>
    %25 = source {handshake.bb = 2 : ui32, handshake.name = "source2"} : <>
    %26 = constant %25 {handshake.bb = 2 : ui32, handshake.name = "constant8", value = 1.100000e+00 : f32} : <>, <f32>
    %27 = mulf %falseResult_2, %26 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "mulf2", internal_delay = "2_875333"} : <f32>
    %28:2 = fork [2] %trueResult {handshake.bb = 3 : ui32, handshake.name = "fork5"} : <i8>
    %29 = trunci %28#0 {handshake.bb = 3 : ui32, handshake.name = "trunci1"} : <i8> to <i7>
    %30:2 = fork [2] %trueResult_1 {handshake.bb = 3 : ui32, handshake.name = "fork6"} : <f32>
    %31:2 = fork [2] %trueResult_3 {handshake.bb = 3 : ui32, handshake.name = "fork7"} : <>
    %32 = constant %31#0 {handshake.bb = 3 : ui32, handshake.name = "constant3", value = 1 : i2} : <>, <i2>
    %33 = extsi %32 {handshake.bb = 3 : ui32, handshake.name = "extsi1"} : <i2> to <i32>
    %34 = source {handshake.bb = 3 : ui32, handshake.name = "source3"} : <>
    %35 = constant %34 {handshake.bb = 3 : ui32, handshake.name = "constant9", value = 1.100000e+00 : f32} : <>, <f32>
    %addressResult_5, %dataResult_6 = store[%29] %30#0 {handshake.bb = 3 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store0"} : <i7>, <f32>, <i7>, <f32>
    %36 = divf %30#1, %35 {fastmath = #arith.fastmath<none>, handshake.bb = 3 : ui32, handshake.name = "divf0", internal_delay = "3_812000"} : <f32>
    %37 = buffer %44#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 4 : ui32, handshake.name = "buffer9"} : <i1>
    %38 = mux %37 [%27, %36] {handshake.bb = 4 : ui32, handshake.name = "mux2"} : <i1>, [<f32>, <f32>] to <f32>
    %39 = buffer %38, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 4 : ui32, handshake.name = "buffer7"} : <f32>
    %40:2 = fork [2] %39 {handshake.bb = 4 : ui32, handshake.name = "fork8"} : <f32>
    %41 = mux %44#0 [%falseResult, %28#1] {handshake.bb = 4 : ui32, handshake.name = "mux3"} : <i1>, [<i8>, <i8>] to <i8>
    %42 = buffer %41, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 4 : ui32, handshake.name = "buffer8"} : <i8>
    %43 = extsi %42 {handshake.bb = 4 : ui32, handshake.name = "extsi5"} : <i8> to <i9>
    %result_7, %index_8 = control_merge [%falseResult_4, %31#1]  {handshake.bb = 4 : ui32, handshake.name = "control_merge3"} : [<>, <>] to <>, <i1>
    %44:2 = fork [2] %index_8 {handshake.bb = 4 : ui32, handshake.name = "fork9"} : <i1>
    %45 = source {handshake.bb = 4 : ui32, handshake.name = "source4"} : <>
    %46 = constant %45 {handshake.bb = 4 : ui32, handshake.name = "constant4", value = 1 : i2} : <>, <i2>
    %47 = extsi %46 {handshake.bb = 4 : ui32, handshake.name = "extsi6"} : <i2> to <i9>
    %48 = source {handshake.bb = 4 : ui32, handshake.name = "source5"} : <>
    %49 = constant %48 {handshake.bb = 4 : ui32, handshake.name = "constant12", value = 100 : i8} : <>, <i8>
    %50 = extsi %49 {handshake.bb = 4 : ui32, handshake.name = "extsi7"} : <i8> to <i9>
    %51 = addf %40#0, %40#1 {fastmath = #arith.fastmath<none>, handshake.bb = 4 : ui32, handshake.name = "addf1", internal_delay = "2_922000"} : <f32>
    %52 = addi %43, %47 {handshake.bb = 4 : ui32, handshake.name = "addi0"} : <i9>
    %53:2 = fork [2] %52 {handshake.bb = 4 : ui32, handshake.name = "fork10"} : <i9>
    %54 = trunci %53#0 {handshake.bb = 4 : ui32, handshake.name = "trunci2"} : <i9> to <i8>
    %55 = cmpi ult, %53#1, %50 {handshake.bb = 4 : ui32, handshake.name = "cmpi0"} : <i9>
    %56:3 = fork [3] %55 {handshake.bb = 4 : ui32, handshake.name = "fork11"} : <i1>
    %trueResult_9, %falseResult_10 = cond_br %56#0, %54 {handshake.bb = 4 : ui32, handshake.name = "cond_br5"} : <i1>, <i8>
    sink %falseResult_10 {handshake.name = "sink2"} : <i8>
    %57 = buffer %56#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 4 : ui32, handshake.name = "buffer10"} : <i1>
    %trueResult_11, %falseResult_12 = cond_br %57, %51 {handshake.bb = 4 : ui32, handshake.name = "cond_br6"} : <i1>, <f32>
    %trueResult_13, %falseResult_14 = cond_br %56#2, %result_7 {handshake.bb = 4 : ui32, handshake.name = "cond_br7"} : <i1>, <>
    %58:2 = fork [2] %falseResult_14 {handshake.bb = 5 : ui32, handshake.name = "fork12"} : <>
    end {handshake.bb = 5 : ui32, handshake.name = "end0"} %falseResult_12, %memEnd_0, %memEnd, %0#1 : <f32>, <>, <>, <>
  }
}

