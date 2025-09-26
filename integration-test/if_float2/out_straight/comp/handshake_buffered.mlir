module {
  handshake.func @if_float2(%arg0: !handshake.channel<f32>, %arg1: memref<100xf32>, %arg2: memref<100xf32>, %arg3: !handshake.control<>, %arg4: !handshake.control<>, %arg5: !handshake.control<>, ...) -> (!handshake.channel<f32>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["x0", "a", "minus_trace", "a_start", "minus_trace_start", "start"], handshake.cfdfcThroughput = #handshake<cfdfcThroughput {"0" = 0.030303030303030311 : f64}>, handshake.cfdfcToBBList = #handshake<cfdfcToBBList {"0" = [1 : ui32]}>, resNames = ["out0", "a_end", "minus_trace_end", "end"]} {
    %0:3 = fork [3] %arg5 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork0"} : <>
    %memEnd = mem_controller[%arg2 : memref<100xf32>] %arg4 (%73, %addressResult_5, %dataResult_6) %101#1 {connectedBlocks = [1 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller2"} :    (!handshake.channel<i32>, !handshake.channel<i7>, !handshake.channel<f32>) -> ()
    %outputs, %memEnd_0 = mem_controller[%arg1 : memref<100xf32>] %arg3 (%addressResult) %101#0 {connectedBlocks = [1 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller3"} :    (!handshake.channel<i7>) -> !handshake.channel<f32>
    %1 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant2", value = false} : <>, <i1>
    %2 = br %1 {handshake.bb = 0 : ui32, handshake.name = "br3"} : <i1>
    %3 = extsi %2 {handshake.bb = 0 : ui32, handshake.name = "extsi4"} : <i1> to <i8>
    %4 = br %arg0 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "br4"} : <f32>
    %5 = br %0#2 {handshake.bb = 0 : ui32, handshake.name = "br5"} : <>
    %6 = mux %17#0 [%3, %trueResult] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i8>, <i8>] to <i8>
    %7 = buffer %6, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer0"} : <i8>
    %8 = buffer %7, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer1"} : <i8>
    %9:3 = fork [3] %8 {handshake.bb = 1 : ui32, handshake.name = "fork12"} : <i8>
    %10 = trunci %9#0 {handshake.bb = 1 : ui32, handshake.name = "trunci0"} : <i8> to <i7>
    %11 = buffer %17#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer6"} : <i1>
    %12 = mux %11 [%4, %trueResult_7] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i1>, [<f32>, <f32>] to <f32>
    %13 = buffer %12, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer2"} : <f32>
    %14:4 = fork [4] %13 {handshake.bb = 1 : ui32, handshake.name = "fork13"} : <f32>
    %result, %index = control_merge [%5, %trueResult_9]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %15 = buffer %result, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer4"} : <>
    %16:2 = fork [2] %15 {handshake.bb = 1 : ui32, handshake.name = "fork14"} : <>
    %17:2 = fork [2] %index {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <i1>
    %18 = source {handshake.bb = 1 : ui32, handshake.name = "source0"} : <>
    %19 = constant %18 {handshake.bb = 1 : ui32, handshake.name = "constant7", value = -0.899999976 : f32} : <>, <f32>
    %20 = source {handshake.bb = 1 : ui32, handshake.name = "source1"} : <>
    %21 = constant %20 {handshake.bb = 1 : ui32, handshake.name = "constant8", value = 0.000000e+00 : f32} : <>, <f32>
    %addressResult, %dataResult = load[%10] %outputs {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i7>, <f32>, <i7>, <f32>
    %22 = mulf %dataResult, %14#1 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "mulf0", internal_delay = "2_875333"} : <f32>
    %23 = mulf %14#0, %19 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "mulf1", internal_delay = "2_875333"} : <f32>
    %24 = addf %22, %23 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "addf0", internal_delay = "2_922000"} : <f32>
    %25 = cmpf ugt, %24, %21 {handshake.bb = 1 : ui32, handshake.name = "cmpf0", internal_delay = "0_000000"} : <f32>
    %26:7 = fork [7] %25 {handshake.bb = 1 : ui32, handshake.name = "fork37"} : <i1>
    %27 = spec_v2_repeating_init %26#1 {handshake.bb = 1 : ui32, handshake.name = "spec_v2_repeating_init0", initToken = 1 : ui1} : <i1>
    %28 = buffer %27, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer8"} : <i1>
    %29:4 = fork [4] %28 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork38"} : <i1>
    %30 = source {handshake.bb = 1 : ui32, handshake.name = "source7"} : <>
    %31 = constant %30 {handshake.bb = 1 : ui32, handshake.name = "constant0", value = true} : <>, <i1>
    %32 = mux %29#0 [%31, %26#0] {handshake.bb = 1 : ui32, handshake.name = "mux5"} : <i1>, [<i1>, <i1>] to <i1>
    %33 = buffer %32, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer9"} : <i1>
    %34:3 = fork [3] %33 {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <i1>
    %35 = buffer %81, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer15"} : <f32>
    %36 = passer %35[%34#2] {handshake.bb = 1 : ui32, handshake.name = "passer6"} : <f32>, <i1>
    %37 = buffer %82, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer16"} : <i8>
    %38 = passer %37[%34#1] {handshake.bb = 1 : ui32, handshake.name = "passer7"} : <i8>, <i1>
    %39 = buffer %84, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer17"} : <>
    %40 = passer %39[%34#0] {handshake.bb = 1 : ui32, handshake.name = "passer8"} : <>, <i1>
    %41 = buffer %26#6, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer7"} : <i1>
    %42 = not %41 {handshake.bb = 1 : ui32, handshake.name = "not0"} : <i1>
    %43:4 = fork [4] %42 {handshake.bb = 1 : ui32, handshake.name = "fork39"} : <i1>
    %44 = merge %9#1 {handshake.bb = 1 : ui32, handshake.name = "merge0"} : <i8>
    %45 = buffer %14#2, bufferType = FIFO_BREAK_NONE, numSlots = 2 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer3"} : <f32>
    %46 = merge %45 {handshake.bb = 1 : ui32, handshake.name = "merge1"} : <f32>
    %47 = passer %index_2[%43#0] {handshake.bb = 1 : ui32, handshake.name = "passer9"} : <i1>, <i1>
    %48 = buffer %16#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer5"} : <>
    %result_1, %index_2 = control_merge [%48]  {handshake.bb = 1 : ui32, handshake.name = "control_merge1"} : [<>] to <>, <i1>
    sink %47 {handshake.name = "sink0"} : <i1>
    %49 = source {handshake.bb = 1 : ui32, handshake.name = "source2"} : <>
    %50 = constant %49 {handshake.bb = 1 : ui32, handshake.name = "constant9", value = 3.000000e+00 : f32} : <>, <f32>
    %51 = addf %46, %50 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "addf1", internal_delay = "2_922000"} : <f32>
    %52 = passer %53[%43#2] {handshake.bb = 1 : ui32, handshake.name = "passer10"} : <f32>, <i1>
    %53 = br %51 {handshake.bb = 1 : ui32, handshake.name = "br6"} : <f32>
    %54 = buffer %57, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer10"} : <i8>
    %55 = buffer %54, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer11"} : <i8>
    %56 = passer %55[%43#3] {handshake.bb = 1 : ui32, handshake.name = "passer11"} : <i8>, <i1>
    %57 = br %44 {handshake.bb = 1 : ui32, handshake.name = "br7"} : <i8>
    %58 = passer %59[%43#1] {handshake.bb = 1 : ui32, handshake.name = "passer12"} : <>, <i1>
    %59 = br %result_1 {handshake.bb = 1 : ui32, handshake.name = "br8"} : <>
    %60 = buffer %62, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer12"} : <i8>
    %61:2 = fork [2] %60 {handshake.bb = 1 : ui32, handshake.name = "fork34"} : <i8>
    %62 = merge %9#2 {handshake.bb = 1 : ui32, handshake.name = "merge2"} : <i8>
    %63 = passer %64[%26#5] {handshake.bb = 1 : ui32, handshake.name = "passer13"} : <i7>, <i1>
    %64 = trunci %61#0 {handshake.bb = 1 : ui32, handshake.name = "trunci1"} : <i8> to <i7>
    %65 = buffer %68#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer13"} : <f32>
    %66 = passer %65[%26#4] {handshake.bb = 1 : ui32, handshake.name = "passer14"} : <f32>, <i1>
    %67 = buffer %69, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer14"} : <f32>
    %68:2 = fork [2] %67 {handshake.bb = 1 : ui32, handshake.name = "fork35"} : <f32>
    %69 = merge %14#3 {handshake.bb = 1 : ui32, handshake.name = "merge3"} : <f32>
    %70:2 = fork [2] %result_3 {handshake.bb = 1 : ui32, handshake.name = "fork36"} : <>
    %71 = passer %index_4[%26#2] {handshake.bb = 1 : ui32, handshake.name = "passer15"} : <i1>, <i1>
    %result_3, %index_4 = control_merge [%16#1]  {handshake.bb = 1 : ui32, handshake.name = "control_merge2"} : [<>] to <>, <i1>
    sink %71 {handshake.name = "sink1"} : <i1>
    %72 = constant %70#0 {handshake.bb = 1 : ui32, handshake.name = "constant3", value = 1 : i2} : <>, <i2>
    %73 = passer %74[%26#3] {handshake.bb = 1 : ui32, handshake.name = "passer16"} : <i32>, <i1>
    %74 = extsi %72 {handshake.bb = 1 : ui32, handshake.name = "extsi1"} : <i2> to <i32>
    %75 = source {handshake.bb = 1 : ui32, handshake.name = "source3"} : <>
    %76 = constant %75 {handshake.bb = 1 : ui32, handshake.name = "constant10", value = 1.000000e+00 : f32} : <>, <f32>
    %addressResult_5, %dataResult_6 = store[%63] %66 {handshake.bb = 1 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store0"} : <i7>, <f32>, <i7>, <f32>
    %77 = addf %68#1, %76 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "addf2", internal_delay = "2_922000"} : <f32>
    %78 = br %77 {handshake.bb = 1 : ui32, handshake.name = "br9"} : <f32>
    %79 = br %61#1 {handshake.bb = 1 : ui32, handshake.name = "br10"} : <i8>
    %80 = br %70#1 {handshake.bb = 1 : ui32, handshake.name = "br11"} : <>
    %81 = mux %29#1 [%52, %78] {handshake.bb = 1 : ui32, handshake.name = "mux2"} : <i1>, [<f32>, <f32>] to <f32>
    %82 = mux %29#2 [%56, %79] {handshake.bb = 1 : ui32, handshake.name = "mux3"} : <i1>, [<i8>, <i8>] to <i8>
    %83 = extsi %38 {handshake.bb = 1 : ui32, handshake.name = "extsi5"} : <i8> to <i9>
    %84 = mux %29#3 [%58, %80] {handshake.bb = 1 : ui32, handshake.name = "mux4"} : <i1>, [<>, <>] to <>
    %85 = source {handshake.bb = 1 : ui32, handshake.name = "source4"} : <>
    %86 = constant %85 {handshake.bb = 1 : ui32, handshake.name = "constant11", value = 1.000000e+00 : f32} : <>, <f32>
    %87 = source {handshake.bb = 1 : ui32, handshake.name = "source5"} : <>
    %88 = constant %87 {handshake.bb = 1 : ui32, handshake.name = "constant4", value = 1 : i2} : <>, <i2>
    %89 = extsi %88 {handshake.bb = 1 : ui32, handshake.name = "extsi6"} : <i2> to <i9>
    %90 = source {handshake.bb = 1 : ui32, handshake.name = "source6"} : <>
    %91 = constant %90 {handshake.bb = 1 : ui32, handshake.name = "constant5", value = 100 : i8} : <>, <i8>
    %92 = extsi %91 {handshake.bb = 1 : ui32, handshake.name = "extsi7"} : <i8> to <i9>
    %93 = divf %86, %36 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "divf0", internal_delay = "3_812000"} : <f32>
    %94 = addi %83, %89 {handshake.bb = 1 : ui32, handshake.name = "addi0"} : <i9>
    %95:2 = fork [2] %94 {handshake.bb = 1 : ui32, handshake.name = "fork9"} : <i9>
    %96 = trunci %95#0 {handshake.bb = 1 : ui32, handshake.name = "trunci2"} : <i9> to <i8>
    %97 = cmpi ult, %95#1, %92 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i9>
    %98:3 = fork [3] %97 {handshake.bb = 1 : ui32, handshake.name = "fork10"} : <i1>
    %trueResult, %falseResult = cond_br %98#0, %96 {handshake.bb = 1 : ui32, handshake.name = "cond_br5"} : <i1>, <i8>
    sink %falseResult {handshake.name = "sink2"} : <i8>
    %99 = buffer %98#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer18"} : <i1>
    %trueResult_7, %falseResult_8 = cond_br %99, %93 {handshake.bb = 1 : ui32, handshake.name = "cond_br6"} : <i1>, <f32>
    %trueResult_9, %falseResult_10 = cond_br %98#2, %40 {handshake.bb = 1 : ui32, handshake.name = "cond_br7"} : <i1>, <>
    %100 = merge %falseResult_8 {handshake.bb = 2 : ui32, handshake.name = "merge4"} : <f32>
    %result_11, %index_12 = control_merge [%falseResult_10]  {handshake.bb = 2 : ui32, handshake.name = "control_merge4"} : [<>] to <>, <i1>
    sink %index_12 {handshake.name = "sink3"} : <i1>
    %101:2 = fork [2] %result_11 {handshake.bb = 2 : ui32, handshake.name = "fork11"} : <>
    end {handshake.bb = 2 : ui32, handshake.name = "end0"} %100, %memEnd_0, %memEnd, %0#1 : <f32>, <>, <>, <>
  }
}

