module {
  handshake.func @if_float2(%arg0: !handshake.channel<f32>, %arg1: memref<100xf32>, %arg2: memref<100xf32>, %arg3: !handshake.control<>, %arg4: !handshake.control<>, %arg5: !handshake.control<>, ...) -> (!handshake.channel<f32>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["x0", "a", "minus_trace", "a_start", "minus_trace_start", "start"], handshake.cfdfcThroughput = #handshake<cfdfcThroughput {"0" = 0.030303030303030311 : f64}>, handshake.cfdfcToBBList = #handshake<cfdfcToBBList {"0" = [1 : ui32]}>, resNames = ["out0", "a_end", "minus_trace_end", "end"]} {
    %0:3 = fork [3] %arg5 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork0"} : <>
    %memEnd = mem_controller[%arg2 : memref<100xf32>] %arg4 (%67, %addressResult_1, %dataResult_2) %91#1 {connectedBlocks = [1 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller2"} :    (!handshake.channel<i32>, !handshake.channel<i7>, !handshake.channel<f32>) -> ()
    %outputs, %memEnd_0 = mem_controller[%arg1 : memref<100xf32>] %arg3 (%addressResult) %91#0 {connectedBlocks = [1 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller3"} :    (!handshake.channel<i7>) -> !handshake.channel<f32>
    %1 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant2", value = false} : <>, <i1>
    %2 = extsi %1 {handshake.bb = 0 : ui32, handshake.name = "extsi4"} : <i1> to <i8>
    %3 = mux %14#0 [%2, %trueResult] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i8>, <i8>] to <i8>
    %4 = buffer %3, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer0"} : <i8>
    %5 = buffer %4, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer1"} : <i8>
    %6:3 = fork [3] %5 {handshake.bb = 1 : ui32, handshake.name = "fork12"} : <i8>
    %7 = trunci %6#0 {handshake.bb = 1 : ui32, handshake.name = "trunci0"} : <i8> to <i7>
    %8 = buffer %14#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer6"} : <i1>
    %9 = mux %8 [%arg0, %trueResult_3] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i1>, [<f32>, <f32>] to <f32>
    %10 = buffer %9, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer2"} : <f32>
    %11:4 = fork [4] %10 {handshake.bb = 1 : ui32, handshake.name = "fork13"} : <f32>
    %result, %index = control_merge [%0#2, %trueResult_5]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %12 = buffer %result, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer4"} : <>
    %13:2 = fork [2] %12 {handshake.bb = 1 : ui32, handshake.name = "fork14"} : <>
    %14:2 = fork [2] %index {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <i1>
    %15 = source {handshake.bb = 1 : ui32, handshake.name = "source0"} : <>
    %16 = constant %15 {handshake.bb = 1 : ui32, handshake.name = "constant7", value = -0.899999976 : f32} : <>, <f32>
    %17 = source {handshake.bb = 1 : ui32, handshake.name = "source1"} : <>
    %18 = constant %17 {handshake.bb = 1 : ui32, handshake.name = "constant8", value = 0.000000e+00 : f32} : <>, <f32>
    %addressResult, %dataResult = load[%7] %outputs {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i7>, <f32>, <i7>, <f32>
    %19 = mulf %dataResult, %11#1 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "mulf0", internal_delay = "2_875333"} : <f32>
    %20 = mulf %11#0, %16 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "mulf1", internal_delay = "2_875333"} : <f32>
    %21 = addf %19, %20 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "addf0", internal_delay = "2_922000"} : <f32>
    %22 = cmpf ugt, %21, %18 {handshake.bb = 1 : ui32, handshake.name = "cmpf0", internal_delay = "0_000000"} : <f32>
    %23:7 = fork [7] %22 {handshake.bb = 1 : ui32, handshake.name = "fork37"} : <i1>
    %24 = spec_v2_repeating_init %23#1 {handshake.bb = 1 : ui32, handshake.name = "spec_v2_repeating_init0", initToken = 1 : ui1} : <i1>
    %25 = buffer %24, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer8"} : <i1>
    %26:4 = fork [4] %25 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork38"} : <i1>
    %27 = source {handshake.bb = 1 : ui32, handshake.name = "source7"} : <>
    %28 = constant %27 {handshake.bb = 1 : ui32, handshake.name = "constant0", value = true} : <>, <i1>
    %29 = mux %26#0 [%28, %23#0] {handshake.bb = 1 : ui32, handshake.name = "mux5"} : <i1>, [<i1>, <i1>] to <i1>
    %30 = buffer %29, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer9"} : <i1>
    %31:3 = fork [3] %30 {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <i1>
    %32 = buffer %72, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer15"} : <f32>
    %33 = passer %32[%31#2] {handshake.bb = 1 : ui32, handshake.name = "passer6"} : <f32>, <i1>
    %34 = buffer %73, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer16"} : <i8>
    %35 = passer %34[%31#1] {handshake.bb = 1 : ui32, handshake.name = "passer7"} : <i8>, <i1>
    %36 = buffer %75, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer17"} : <>
    %37 = passer %36[%31#0] {handshake.bb = 1 : ui32, handshake.name = "passer8"} : <>, <i1>
    %38 = buffer %23#6, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer7"} : <i1>
    %39 = not %38 {handshake.bb = 1 : ui32, handshake.name = "not0"} : <i1>
    %40:4 = fork [4] %39 {handshake.bb = 1 : ui32, handshake.name = "fork39"} : <i1>
    %41 = buffer %11#2, bufferType = FIFO_BREAK_NONE, numSlots = 2 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer3"} : <f32>
    %42 = passer %45[%40#0] {handshake.bb = 1 : ui32, handshake.name = "passer9"} : <i1>, <i1>
    %43 = buffer %13#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer5"} : <>
    %44 = source {handshake.bb = 1 : ui32, handshake.name = "source8"} : <>
    %45 = constant %44 {handshake.bb = 1 : ui32, handshake.name = "constant1", value = false} : <>, <i1>
    sink %42 {handshake.name = "sink0"} : <i1>
    %46 = source {handshake.bb = 1 : ui32, handshake.name = "source2"} : <>
    %47 = constant %46 {handshake.bb = 1 : ui32, handshake.name = "constant9", value = 3.000000e+00 : f32} : <>, <f32>
    %48 = addf %41, %47 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "addf1", internal_delay = "2_922000"} : <f32>
    %49 = passer %48[%40#2] {handshake.bb = 1 : ui32, handshake.name = "passer10"} : <f32>, <i1>
    %50 = buffer %6#1, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer10"} : <i8>
    %51 = buffer %50, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer11"} : <i8>
    %52 = passer %51[%40#3] {handshake.bb = 1 : ui32, handshake.name = "passer11"} : <i8>, <i1>
    %53 = passer %43[%40#1] {handshake.bb = 1 : ui32, handshake.name = "passer12"} : <>, <i1>
    %54 = buffer %6#2, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer12"} : <i8>
    %55:2 = fork [2] %54 {handshake.bb = 1 : ui32, handshake.name = "fork34"} : <i8>
    %56 = passer %57[%23#5] {handshake.bb = 1 : ui32, handshake.name = "passer13"} : <i7>, <i1>
    %57 = trunci %55#0 {handshake.bb = 1 : ui32, handshake.name = "trunci1"} : <i8> to <i7>
    %58 = buffer %61#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer13"} : <f32>
    %59 = passer %58[%23#4] {handshake.bb = 1 : ui32, handshake.name = "passer14"} : <f32>, <i1>
    %60 = buffer %11#3, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer14"} : <f32>
    %61:2 = fork [2] %60 {handshake.bb = 1 : ui32, handshake.name = "fork35"} : <f32>
    %62:2 = fork [2] %13#1 {handshake.bb = 1 : ui32, handshake.name = "fork36"} : <>
    %63 = passer %65[%23#2] {handshake.bb = 1 : ui32, handshake.name = "passer15"} : <i1>, <i1>
    %64 = source {handshake.bb = 1 : ui32, handshake.name = "source9"} : <>
    %65 = constant %64 {handshake.bb = 1 : ui32, handshake.name = "constant6", value = false} : <>, <i1>
    sink %63 {handshake.name = "sink1"} : <i1>
    %66 = constant %62#0 {handshake.bb = 1 : ui32, handshake.name = "constant3", value = 1 : i2} : <>, <i2>
    %67 = passer %68[%23#3] {handshake.bb = 1 : ui32, handshake.name = "passer16"} : <i32>, <i1>
    %68 = extsi %66 {handshake.bb = 1 : ui32, handshake.name = "extsi1"} : <i2> to <i32>
    %69 = source {handshake.bb = 1 : ui32, handshake.name = "source3"} : <>
    %70 = constant %69 {handshake.bb = 1 : ui32, handshake.name = "constant10", value = 1.000000e+00 : f32} : <>, <f32>
    %addressResult_1, %dataResult_2 = store[%56] %59 {handshake.bb = 1 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store0"} : <i7>, <f32>, <i7>, <f32>
    %71 = addf %61#1, %70 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "addf2", internal_delay = "2_922000"} : <f32>
    %72 = mux %26#1 [%49, %71] {handshake.bb = 1 : ui32, handshake.name = "mux2"} : <i1>, [<f32>, <f32>] to <f32>
    %73 = mux %26#2 [%52, %55#1] {handshake.bb = 1 : ui32, handshake.name = "mux3"} : <i1>, [<i8>, <i8>] to <i8>
    %74 = extsi %35 {handshake.bb = 1 : ui32, handshake.name = "extsi5"} : <i8> to <i9>
    %75 = mux %26#3 [%53, %62#1] {handshake.bb = 1 : ui32, handshake.name = "mux4"} : <i1>, [<>, <>] to <>
    %76 = source {handshake.bb = 1 : ui32, handshake.name = "source4"} : <>
    %77 = constant %76 {handshake.bb = 1 : ui32, handshake.name = "constant11", value = 1.000000e+00 : f32} : <>, <f32>
    %78 = source {handshake.bb = 1 : ui32, handshake.name = "source5"} : <>
    %79 = constant %78 {handshake.bb = 1 : ui32, handshake.name = "constant4", value = 1 : i2} : <>, <i2>
    %80 = extsi %79 {handshake.bb = 1 : ui32, handshake.name = "extsi6"} : <i2> to <i9>
    %81 = source {handshake.bb = 1 : ui32, handshake.name = "source6"} : <>
    %82 = constant %81 {handshake.bb = 1 : ui32, handshake.name = "constant5", value = 100 : i8} : <>, <i8>
    %83 = extsi %82 {handshake.bb = 1 : ui32, handshake.name = "extsi7"} : <i8> to <i9>
    %84 = divf %77, %33 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "divf0", internal_delay = "3_812000"} : <f32>
    %85 = addi %74, %80 {handshake.bb = 1 : ui32, handshake.name = "addi0"} : <i9>
    %86:2 = fork [2] %85 {handshake.bb = 1 : ui32, handshake.name = "fork9"} : <i9>
    %87 = trunci %86#0 {handshake.bb = 1 : ui32, handshake.name = "trunci2"} : <i9> to <i8>
    %88 = cmpi ult, %86#1, %83 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i9>
    %89:3 = fork [3] %88 {handshake.bb = 1 : ui32, handshake.name = "fork10"} : <i1>
    %trueResult, %falseResult = cond_br %89#0, %87 {handshake.bb = 1 : ui32, handshake.name = "cond_br5"} : <i1>, <i8>
    sink %falseResult {handshake.name = "sink2"} : <i8>
    %90 = buffer %89#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer18"} : <i1>
    %trueResult_3, %falseResult_4 = cond_br %90, %84 {handshake.bb = 1 : ui32, handshake.name = "cond_br6"} : <i1>, <f32>
    %trueResult_5, %falseResult_6 = cond_br %89#2, %37 {handshake.bb = 1 : ui32, handshake.name = "cond_br7"} : <i1>, <>
    %91:2 = fork [2] %falseResult_6 {handshake.bb = 2 : ui32, handshake.name = "fork11"} : <>
    end {handshake.bb = 2 : ui32, handshake.name = "end0"} %falseResult_4, %memEnd_0, %memEnd, %0#1 : <f32>, <>, <>, <>
  }
}

