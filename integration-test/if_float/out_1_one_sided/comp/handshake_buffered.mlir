module {
  handshake.func @if_float(%arg0: !handshake.channel<f32>, %arg1: memref<100xf32>, %arg2: memref<100xf32>, %arg3: !handshake.control<>, %arg4: !handshake.control<>, %arg5: !handshake.control<>, ...) -> (!handshake.channel<f32>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["x0", "a", "minus_trace", "a_start", "minus_trace_start", "start"], handshake.cfdfcThroughput = #handshake<cfdfcThroughput {"0" = 0.034482758620689655 : f64}>, handshake.cfdfcToBBList = #handshake<cfdfcToBBList {"0" = [1 : ui32]}>, resNames = ["out0", "a_end", "minus_trace_end", "end"]} {
    %0:3 = fork [3] %arg5 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork0"} : <>
    %memEnd = mem_controller[%arg2 : memref<100xf32>] %arg4 (%67, %addressResult_5, %dataResult_6) %102#1 {connectedBlocks = [1 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller2"} :    (!handshake.channel<i32>, !handshake.channel<i7>, !handshake.channel<f32>) -> ()
    %outputs, %memEnd_0 = mem_controller[%arg1 : memref<100xf32>] %arg3 (%addressResult) %102#0 {connectedBlocks = [1 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller3"} :    (!handshake.channel<i7>) -> !handshake.channel<f32>
    %1 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant2", value = false} : <>, <i1>
    %2 = br %1 {handshake.bb = 0 : ui32, handshake.name = "br3"} : <i1>
    %3 = extsi %2 {handshake.bb = 0 : ui32, handshake.name = "extsi4"} : <i1> to <i8>
    %4 = br %arg0 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "br4"} : <f32>
    %5 = br %0#2 {handshake.bb = 0 : ui32, handshake.name = "br5"} : <>
    %6 = mux %17#0 [%3, %trueResult] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i8>, <i8>] to <i8>
    %7 = buffer %6, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer0"} : <i8>
    %8 = buffer %7, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer1"} : <i8>
    %9:3 = fork [3] %8 {handshake.bb = 1 : ui32, handshake.name = "fork14"} : <i8>
    %10 = trunci %9#0 {handshake.bb = 1 : ui32, handshake.name = "trunci0"} : <i8> to <i7>
    %11 = buffer %17#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer5"} : <i1>
    %12 = mux %11 [%4, %trueResult_7] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i1>, [<f32>, <f32>] to <f32>
    %13 = buffer %12, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer3"} : <f32>
    %14:4 = fork [4] %13 {handshake.bb = 1 : ui32, handshake.name = "fork15"} : <f32>
    %result, %index = control_merge [%5, %trueResult_9]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %15 = buffer %result, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer4"} : <>
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
    %25 = buffer %24, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer6"} : <f32>
    %26 = cmpf ugt, %25, %21 {handshake.bb = 1 : ui32, handshake.name = "cmpf0", internal_delay = "0_000000"} : <f32>
    %27:6 = fork [6] %26 {handshake.bb = 1 : ui32, handshake.name = "fork48"} : <i1>
    %28 = spec_v2_repeating_init %27#1 {handshake.bb = 1 : ui32, handshake.name = "spec_v2_repeating_init0", initToken = 1 : ui1} : <i1>
    %29 = buffer %28, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer7"} : <i1>
    %30:4 = fork [4] %29 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork49"} : <i1>
    %31 = source {handshake.bb = 1 : ui32, handshake.name = "source6"} : <>
    %32 = constant %31 {handshake.bb = 1 : ui32, handshake.name = "constant0", value = true} : <>, <i1>
    %33 = mux %30#0 [%32, %27#0] {handshake.bb = 1 : ui32, handshake.name = "mux5"} : <i1>, [<i1>, <i1>] to <i1>
    %34 = buffer %33, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer8"} : <i1>
    %35:6 = fork [6] %34 {handshake.bb = 1 : ui32, handshake.name = "fork50"} : <i1>
    %36 = buffer %81, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer18"} : <>
    %37 = passer %36[%35#5] {handshake.bb = 1 : ui32, handshake.name = "passer6"} : <>, <i1>
    %38 = not %27#5 {handshake.bb = 1 : ui32, handshake.name = "not0"} : <i1>
    %39 = buffer %38, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer10"} : <i1>
    %40:3 = fork [3] %39 {handshake.bb = 1 : ui32, handshake.name = "fork51"} : <i1>
    %41 = merge %9#1 {handshake.bb = 1 : ui32, handshake.name = "merge0"} : <i8>
    %42 = merge %14#2 {handshake.bb = 1 : ui32, handshake.name = "merge1"} : <f32>
    %result_1, %index_2 = control_merge [%16#0]  {handshake.bb = 1 : ui32, handshake.name = "control_merge1"} : [<>] to <>, <i1>
    sink %index_2 {handshake.name = "sink0"} : <i1>
    %43 = source {handshake.bb = 1 : ui32, handshake.name = "source2"} : <>
    %44 = constant %43 {handshake.bb = 1 : ui32, handshake.name = "constant8", value = 1.100000e+00 : f32} : <>, <f32>
    %45 = buffer %42, bufferType = FIFO_BREAK_NONE, numSlots = 2 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer13"} : <f32>
    %46 = mulf %45, %44 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "mulf2", internal_delay = "2_875333"} : <f32>
    %47 = passer %48[%40#1] {handshake.bb = 1 : ui32, handshake.name = "passer7"} : <f32>, <i1>
    %48 = br %46 {handshake.bb = 1 : ui32, handshake.name = "br6"} : <f32>
    %49 = passer %52[%40#2] {handshake.bb = 1 : ui32, handshake.name = "passer8"} : <i8>, <i1>
    %50 = buffer %41, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer11"} : <i8>
    %51 = buffer %50, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer12"} : <i8>
    %52 = br %51 {handshake.bb = 1 : ui32, handshake.name = "br7"} : <i8>
    %53 = buffer %55, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer14"} : <>
    %54 = passer %53[%40#0] {handshake.bb = 1 : ui32, handshake.name = "passer9"} : <>, <i1>
    %55 = br %result_1 {handshake.bb = 1 : ui32, handshake.name = "br8"} : <>
    %56:2 = fork [2] %58 {handshake.bb = 1 : ui32, handshake.name = "fork43"} : <i8>
    %57 = buffer %9#2, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer2"} : <i8>
    %58 = merge %57 {handshake.bb = 1 : ui32, handshake.name = "merge2"} : <i8>
    %59 = passer %60[%27#4] {handshake.bb = 1 : ui32, handshake.name = "passer10"} : <i7>, <i1>
    %60 = trunci %56#0 {handshake.bb = 1 : ui32, handshake.name = "trunci1"} : <i8> to <i7>
    %61 = buffer %63#0, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer15"} : <f32>
    %62 = passer %61[%27#3] {handshake.bb = 1 : ui32, handshake.name = "passer11"} : <f32>, <i1>
    %63:2 = fork [2] %64 {handshake.bb = 1 : ui32, handshake.name = "fork44"} : <f32>
    %64 = merge %14#3 {handshake.bb = 1 : ui32, handshake.name = "merge3"} : <f32>
    %65:2 = fork [2] %result_3 {handshake.bb = 1 : ui32, handshake.name = "fork45"} : <>
    %result_3, %index_4 = control_merge [%16#1]  {handshake.bb = 1 : ui32, handshake.name = "control_merge2"} : [<>] to <>, <i1>
    sink %index_4 {handshake.name = "sink1"} : <i1>
    %66 = constant %65#0 {handshake.bb = 1 : ui32, handshake.name = "constant3", value = 1 : i2} : <>, <i2>
    %67 = passer %68[%27#2] {handshake.bb = 1 : ui32, handshake.name = "passer12"} : <i32>, <i1>
    %68 = extsi %66 {handshake.bb = 1 : ui32, handshake.name = "extsi1"} : <i2> to <i32>
    %69 = source {handshake.bb = 1 : ui32, handshake.name = "source3"} : <>
    %70 = constant %69 {handshake.bb = 1 : ui32, handshake.name = "constant9", value = 1.100000e+00 : f32} : <>, <f32>
    %addressResult_5, %dataResult_6 = store[%59] %62 {handshake.bb = 1 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store0"} : <i7>, <f32>, <i7>, <f32>
    %71 = divf %63#1, %70 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "divf0", internal_delay = "3_812000"} : <f32>
    %72 = br %71 {handshake.bb = 1 : ui32, handshake.name = "br9"} : <f32>
    %73 = br %56#1 {handshake.bb = 1 : ui32, handshake.name = "br10"} : <i8>
    %74 = br %65#1 {handshake.bb = 1 : ui32, handshake.name = "br11"} : <>
    %75 = mux %30#2 [%47, %72] {handshake.bb = 1 : ui32, handshake.name = "mux2"} : <i1>, [<f32>, <f32>] to <f32>
    %76 = buffer %75, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer16"} : <f32>
    %77:2 = fork [2] %76 {handshake.bb = 1 : ui32, handshake.name = "fork8"} : <f32>
    %78 = mux %30#3 [%49, %73] {handshake.bb = 1 : ui32, handshake.name = "mux3"} : <i1>, [<i8>, <i8>] to <i8>
    %79 = buffer %78, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer17"} : <i8>
    %80 = extsi %79 {handshake.bb = 1 : ui32, handshake.name = "extsi5"} : <i8> to <i9>
    %81 = mux %30#1 [%54, %74] {handshake.bb = 1 : ui32, handshake.name = "mux4"} : <i1>, [<>, <>] to <>
    %82 = source {handshake.bb = 1 : ui32, handshake.name = "source4"} : <>
    %83 = constant %82 {handshake.bb = 1 : ui32, handshake.name = "constant4", value = 1 : i2} : <>, <i2>
    %84 = extsi %83 {handshake.bb = 1 : ui32, handshake.name = "extsi6"} : <i2> to <i9>
    %85 = source {handshake.bb = 1 : ui32, handshake.name = "source5"} : <>
    %86 = constant %85 {handshake.bb = 1 : ui32, handshake.name = "constant12", value = 100 : i8} : <>, <i8>
    %87 = extsi %86 {handshake.bb = 1 : ui32, handshake.name = "extsi7"} : <i8> to <i9>
    %88 = buffer %35#4, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer9"} : <i1>
    %89 = passer %90[%88] {handshake.bb = 1 : ui32, handshake.name = "passer13"} : <f32>, <i1>
    %90 = addf %77#0, %77#1 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "addf1", internal_delay = "2_922000"} : <f32>
    %91:2 = fork [2] %92 {handshake.bb = 1 : ui32, handshake.name = "fork46"} : <i9>
    %92 = addi %80, %84 {handshake.bb = 1 : ui32, handshake.name = "addi0"} : <i9>
    %93 = passer %94[%35#3] {handshake.bb = 1 : ui32, handshake.name = "passer14"} : <i8>, <i1>
    %94 = trunci %91#0 {handshake.bb = 1 : ui32, handshake.name = "trunci2"} : <i9> to <i8>
    %95 = passer %98#0[%35#2] {handshake.bb = 1 : ui32, handshake.name = "passer15"} : <i1>, <i1>
    %96 = passer %98#1[%35#1] {handshake.bb = 1 : ui32, handshake.name = "passer16"} : <i1>, <i1>
    %97 = passer %98#2[%35#0] {handshake.bb = 1 : ui32, handshake.name = "passer17"} : <i1>, <i1>
    %98:3 = fork [3] %99 {handshake.bb = 1 : ui32, handshake.name = "fork47"} : <i1>
    %99 = cmpi ult, %91#1, %87 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i9>
    %trueResult, %falseResult = cond_br %95, %93 {handshake.bb = 1 : ui32, handshake.name = "cond_br5"} : <i1>, <i8>
    sink %falseResult {handshake.name = "sink2"} : <i8>
    %100 = buffer %96, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer19"} : <i1>
    %trueResult_7, %falseResult_8 = cond_br %100, %89 {handshake.bb = 1 : ui32, handshake.name = "cond_br6"} : <i1>, <f32>
    %trueResult_9, %falseResult_10 = cond_br %97, %37 {handshake.bb = 1 : ui32, handshake.name = "cond_br7"} : <i1>, <>
    %101 = merge %falseResult_8 {handshake.bb = 2 : ui32, handshake.name = "merge4"} : <f32>
    %result_11, %index_12 = control_merge [%falseResult_10]  {handshake.bb = 2 : ui32, handshake.name = "control_merge4"} : [<>] to <>, <i1>
    sink %index_12 {handshake.name = "sink3"} : <i1>
    %102:2 = fork [2] %result_11 {handshake.bb = 2 : ui32, handshake.name = "fork12"} : <>
    end {handshake.bb = 2 : ui32, handshake.name = "end0"} %101, %memEnd_0, %memEnd, %0#1 : <f32>, <>, <>, <>
  }
}

