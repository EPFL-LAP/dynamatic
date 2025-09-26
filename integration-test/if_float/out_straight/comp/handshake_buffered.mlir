module {
  handshake.func @if_float(%arg0: !handshake.channel<f32>, %arg1: memref<100xf32>, %arg2: memref<100xf32>, %arg3: !handshake.control<>, %arg4: !handshake.control<>, %arg5: !handshake.control<>, ...) -> (!handshake.channel<f32>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["x0", "a", "minus_trace", "a_start", "minus_trace_start", "start"], handshake.cfdfcThroughput = #handshake<cfdfcThroughput {"0" = 0.045454545454545456 : f64}>, handshake.cfdfcToBBList = #handshake<cfdfcToBBList {"0" = [1 : ui32]}>, resNames = ["out0", "a_end", "minus_trace_end", "end"]} {
    %0:3 = fork [3] %arg5 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork0"} : <>
    %memEnd = mem_controller[%arg2 : memref<100xf32>] %arg4 (%56, %addressResult_5, %dataResult_6) %93#1 {connectedBlocks = [1 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller2"} :    (!handshake.channel<i32>, !handshake.channel<i7>, !handshake.channel<f32>) -> ()
    %outputs, %memEnd_0 = mem_controller[%arg1 : memref<100xf32>] %arg3 (%addressResult) %93#0 {connectedBlocks = [1 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller3"} :    (!handshake.channel<i7>) -> !handshake.channel<f32>
    %1 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant2", value = false} : <>, <i1>
    %2 = br %1 {handshake.bb = 0 : ui32, handshake.name = "br3"} : <i1>
    %3 = extsi %2 {handshake.bb = 0 : ui32, handshake.name = "extsi4"} : <i1> to <i8>
    %4 = br %arg0 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "br4"} : <f32>
    %5 = br %0#2 {handshake.bb = 0 : ui32, handshake.name = "br5"} : <>
    %6 = mux %17#0 [%3, %trueResult] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i8>, <i8>] to <i8>
    %7 = buffer %6, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer0"} : <i8>
    %8 = buffer %7, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer1"} : <i8>
    %9:3 = fork [3] %8 {handshake.bb = 1 : ui32, handshake.name = "fork14"} : <i8>
    %10 = trunci %9#0 {handshake.bb = 1 : ui32, handshake.name = "trunci0"} : <i8> to <i7>
    %11 = buffer %17#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer6"} : <i1>
    %12 = mux %11 [%4, %trueResult_7] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i1>, [<f32>, <f32>] to <f32>
    %13 = buffer %12, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer4"} : <f32>
    %14:4 = fork [4] %13 {handshake.bb = 1 : ui32, handshake.name = "fork15"} : <f32>
    %result, %index = control_merge [%5, %trueResult_9]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %15 = buffer %result, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer5"} : <>
    %16:2 = fork [2] %15 {handshake.bb = 1 : ui32, handshake.name = "fork16"} : <>
    %17:2 = fork [2] %index {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <i1>
    %18 = source {handshake.bb = 1 : ui32, handshake.name = "source0"} : <>
    %19 = constant %18 {handshake.bb = 1 : ui32, handshake.name = "constant6", value = -0.899999976 : f32} : <>, <f32>
    %20 = source {handshake.bb = 1 : ui32, handshake.name = "source1"} : <>
    %21 = constant %20 {handshake.bb = 1 : ui32, handshake.name = "constant7", value = 0.000000e+00 : f32} : <>, <f32>
    %addressResult, %dataResult = load[%10] %outputs {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i7>, <f32>, <i7>, <f32>
    %22 = mulf %dataResult, %14#1 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "mulf0", internal_delay = "2_875333"} : <f32>
    %23 = mulf %14#0, %19 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "mulf1", internal_delay = "2_875333"} : <f32>
    %24 = addf %22, %23 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "addf0", internal_delay = "2_922000"} : <f32>
    %25 = cmpf ugt, %24, %21 {handshake.bb = 1 : ui32, handshake.name = "cmpf0", internal_delay = "0_000000"} : <f32>
    %26:10 = fork [10] %25 {handshake.bb = 1 : ui32, handshake.name = "fork37"} : <i1>
    %27 = buffer %26#9, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer8"} : <i1>
    %28 = not %27 {handshake.bb = 1 : ui32, handshake.name = "not0"} : <i1>
    %29:3 = fork [3] %28 {handshake.bb = 1 : ui32, handshake.name = "fork38"} : <i1>
    %30 = buffer %9#1, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer2"} : <i8>
    %31 = buffer %30, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer3"} : <i8>
    %32 = merge %31 {handshake.bb = 1 : ui32, handshake.name = "merge0"} : <i8>
    %33 = merge %14#2 {handshake.bb = 1 : ui32, handshake.name = "merge1"} : <f32>
    %result_1, %index_2 = control_merge [%16#0]  {handshake.bb = 1 : ui32, handshake.name = "control_merge1"} : [<>] to <>, <i1>
    sink %index_2 {handshake.name = "sink0"} : <i1>
    %34 = source {handshake.bb = 1 : ui32, handshake.name = "source2"} : <>
    %35 = constant %34 {handshake.bb = 1 : ui32, handshake.name = "constant8", value = 1.100000e+00 : f32} : <>, <f32>
    %36 = mulf %33, %35 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "mulf2", internal_delay = "2_875333"} : <f32>
    %37 = passer %39[%29#1] {handshake.bb = 1 : ui32, handshake.name = "passer6"} : <f32>, <i1>
    %38 = buffer %36, bufferType = FIFO_BREAK_NONE, numSlots = 2 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer9"} : <f32>
    %39 = br %38 {handshake.bb = 1 : ui32, handshake.name = "br6"} : <f32>
    %40 = passer %41[%29#2] {handshake.bb = 1 : ui32, handshake.name = "passer7"} : <i8>, <i1>
    %41 = br %32 {handshake.bb = 1 : ui32, handshake.name = "br7"} : <i8>
    %42 = buffer %44, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer10"} : <>
    %43 = passer %42[%29#0] {handshake.bb = 1 : ui32, handshake.name = "passer8"} : <>, <i1>
    %44 = br %result_1 {handshake.bb = 1 : ui32, handshake.name = "br8"} : <>
    %45 = buffer %47, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer11"} : <i8>
    %46:2 = fork [2] %45 {handshake.bb = 1 : ui32, handshake.name = "fork34"} : <i8>
    %47 = merge %9#2 {handshake.bb = 1 : ui32, handshake.name = "merge2"} : <i8>
    %48 = passer %49[%26#8] {handshake.bb = 1 : ui32, handshake.name = "passer9"} : <i7>, <i1>
    %49 = trunci %46#0 {handshake.bb = 1 : ui32, handshake.name = "trunci1"} : <i8> to <i7>
    %50 = buffer %52#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer12"} : <f32>
    %51 = passer %50[%26#6] {handshake.bb = 1 : ui32, handshake.name = "passer10"} : <f32>, <i1>
    %52:2 = fork [2] %53 {handshake.bb = 1 : ui32, handshake.name = "fork35"} : <f32>
    %53 = merge %14#3 {handshake.bb = 1 : ui32, handshake.name = "merge3"} : <f32>
    %54:2 = fork [2] %result_3 {handshake.bb = 1 : ui32, handshake.name = "fork36"} : <>
    %result_3, %index_4 = control_merge [%16#1]  {handshake.bb = 1 : ui32, handshake.name = "control_merge2"} : [<>] to <>, <i1>
    sink %index_4 {handshake.name = "sink1"} : <i1>
    %55 = constant %54#0 {handshake.bb = 1 : ui32, handshake.name = "constant3", value = 1 : i2} : <>, <i2>
    %56 = passer %57[%26#4] {handshake.bb = 1 : ui32, handshake.name = "passer11"} : <i32>, <i1>
    %57 = extsi %55 {handshake.bb = 1 : ui32, handshake.name = "extsi1"} : <i2> to <i32>
    %58 = source {handshake.bb = 1 : ui32, handshake.name = "source3"} : <>
    %59 = constant %58 {handshake.bb = 1 : ui32, handshake.name = "constant9", value = 1.100000e+00 : f32} : <>, <f32>
    %addressResult_5, %dataResult_6 = store[%48] %51 {handshake.bb = 1 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store0"} : <i7>, <f32>, <i7>, <f32>
    %60 = divf %52#1, %59 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "divf0", internal_delay = "3_812000"} : <f32>
    %61 = buffer %26#5, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer7"} : <i1>
    %62 = passer %63[%61] {handshake.bb = 1 : ui32, handshake.name = "passer12"} : <f32>, <i1>
    %63 = br %60 {handshake.bb = 1 : ui32, handshake.name = "br9"} : <f32>
    %64 = passer %65[%26#7] {handshake.bb = 1 : ui32, handshake.name = "passer13"} : <i8>, <i1>
    %65 = br %46#1 {handshake.bb = 1 : ui32, handshake.name = "br10"} : <i8>
    %66 = passer %67[%26#3] {handshake.bb = 1 : ui32, handshake.name = "passer14"} : <>, <i1>
    %67 = br %54#1 {handshake.bb = 1 : ui32, handshake.name = "br11"} : <>
    %68 = buffer %62, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer13"} : <f32>
    %69 = mux %26#1 [%37, %68] {handshake.bb = 1 : ui32, handshake.name = "mux2"} : <i1>, [<f32>, <f32>] to <f32>
    %70 = buffer %69, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer16"} : <f32>
    %71:2 = fork [2] %70 {handshake.bb = 1 : ui32, handshake.name = "fork8"} : <f32>
    %72 = buffer %64, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer14"} : <i8>
    %73 = mux %26#0 [%40, %72] {handshake.bb = 1 : ui32, handshake.name = "mux3"} : <i1>, [<i8>, <i8>] to <i8>
    %74 = buffer %73, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer17"} : <i8>
    %75 = extsi %74 {handshake.bb = 1 : ui32, handshake.name = "extsi5"} : <i8> to <i9>
    %76 = buffer %66, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer15"} : <>
    %77 = mux %26#2 [%43, %76] {handshake.bb = 1 : ui32, handshake.name = "mux4"} : <i1>, [<>, <>] to <>
    %78 = source {handshake.bb = 1 : ui32, handshake.name = "source4"} : <>
    %79 = constant %78 {handshake.bb = 1 : ui32, handshake.name = "constant4", value = 1 : i2} : <>, <i2>
    %80 = extsi %79 {handshake.bb = 1 : ui32, handshake.name = "extsi6"} : <i2> to <i9>
    %81 = source {handshake.bb = 1 : ui32, handshake.name = "source5"} : <>
    %82 = constant %81 {handshake.bb = 1 : ui32, handshake.name = "constant12", value = 100 : i8} : <>, <i8>
    %83 = extsi %82 {handshake.bb = 1 : ui32, handshake.name = "extsi7"} : <i8> to <i9>
    %84 = addf %71#0, %71#1 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "addf1", internal_delay = "2_922000"} : <f32>
    %85 = addi %75, %80 {handshake.bb = 1 : ui32, handshake.name = "addi0"} : <i9>
    %86:2 = fork [2] %85 {handshake.bb = 1 : ui32, handshake.name = "fork10"} : <i9>
    %87 = trunci %86#0 {handshake.bb = 1 : ui32, handshake.name = "trunci2"} : <i9> to <i8>
    %88 = cmpi ult, %86#1, %83 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i9>
    %89:3 = fork [3] %88 {handshake.bb = 1 : ui32, handshake.name = "fork11"} : <i1>
    %trueResult, %falseResult = cond_br %89#0, %87 {handshake.bb = 1 : ui32, handshake.name = "cond_br5"} : <i1>, <i8>
    sink %falseResult {handshake.name = "sink2"} : <i8>
    %90 = buffer %89#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer19"} : <i1>
    %trueResult_7, %falseResult_8 = cond_br %90, %84 {handshake.bb = 1 : ui32, handshake.name = "cond_br6"} : <i1>, <f32>
    %91 = buffer %77, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer18"} : <>
    %trueResult_9, %falseResult_10 = cond_br %89#2, %91 {handshake.bb = 1 : ui32, handshake.name = "cond_br7"} : <i1>, <>
    %92 = merge %falseResult_8 {handshake.bb = 2 : ui32, handshake.name = "merge4"} : <f32>
    %result_11, %index_12 = control_merge [%falseResult_10]  {handshake.bb = 2 : ui32, handshake.name = "control_merge4"} : [<>] to <>, <i1>
    sink %index_12 {handshake.name = "sink3"} : <i1>
    %93:2 = fork [2] %result_11 {handshake.bb = 2 : ui32, handshake.name = "fork12"} : <>
    end {handshake.bb = 2 : ui32, handshake.name = "end0"} %92, %memEnd_0, %memEnd, %0#1 : <f32>, <>, <>, <>
  }
}

