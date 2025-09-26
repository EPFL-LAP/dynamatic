module {
  handshake.func @if_float2(%arg0: !handshake.channel<f32>, %arg1: memref<100xf32>, %arg2: memref<100xf32>, %arg3: !handshake.control<>, %arg4: !handshake.control<>, %arg5: !handshake.control<>, ...) -> (!handshake.channel<f32>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["x0", "a", "minus_trace", "a_start", "minus_trace_start", "start"], handshake.cfdfcThroughput = #handshake<cfdfcThroughput {"0" = 0.034482758620689655 : f64}>, handshake.cfdfcToBBList = #handshake<cfdfcToBBList {"0" = [1 : ui32]}>, resNames = ["out0", "a_end", "minus_trace_end", "end"]} {
    %0:3 = fork [3] %arg5 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork0"} : <>
    %memEnd = mem_controller[%arg2 : memref<100xf32>] %arg4 (%65, %addressResult_5, %dataResult_6) %104#1 {connectedBlocks = [1 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller2"} :    (!handshake.channel<i32>, !handshake.channel<i7>, !handshake.channel<f32>) -> ()
    %outputs, %memEnd_0 = mem_controller[%arg1 : memref<100xf32>] %arg3 (%addressResult) %104#0 {connectedBlocks = [1 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller3"} :    (!handshake.channel<i7>) -> !handshake.channel<f32>
    %1 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant2", value = false} : <>, <i1>
    %2 = br %1 {handshake.bb = 0 : ui32, handshake.name = "br3"} : <i1>
    %3 = extsi %2 {handshake.bb = 0 : ui32, handshake.name = "extsi4"} : <i1> to <i8>
    %4 = br %arg0 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "br4"} : <f32>
    %5 = br %0#2 {handshake.bb = 0 : ui32, handshake.name = "br5"} : <>
    %6 = mux %15#0 [%3, %trueResult] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i8>, <i8>] to <i8>
    %7 = buffer %6, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer0"} : <i8>
    %8:3 = fork [3] %7 {handshake.bb = 1 : ui32, handshake.name = "fork12"} : <i8>
    %9 = trunci %8#0 {handshake.bb = 1 : ui32, handshake.name = "trunci0"} : <i8> to <i7>
    %10 = mux %15#1 [%4, %trueResult_7] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i1>, [<f32>, <f32>] to <f32>
    %11 = buffer %10, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer1"} : <f32>
    %12:4 = fork [4] %11 {handshake.bb = 1 : ui32, handshake.name = "fork13"} : <f32>
    %result, %index = control_merge [%5, %trueResult_9]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %13 = buffer %result, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer4"} : <>
    %14:2 = fork [2] %13 {handshake.bb = 1 : ui32, handshake.name = "fork14"} : <>
    %15:2 = fork [2] %index {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <i1>
    %16 = source {handshake.bb = 1 : ui32, handshake.name = "source0"} : <>
    %17 = constant %16 {handshake.bb = 1 : ui32, handshake.name = "constant7", value = -0.899999976 : f32} : <>, <f32>
    %18 = source {handshake.bb = 1 : ui32, handshake.name = "source1"} : <>
    %19 = constant %18 {handshake.bb = 1 : ui32, handshake.name = "constant8", value = 0.000000e+00 : f32} : <>, <f32>
    %addressResult, %dataResult = load[%9] %outputs {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i7>, <f32>, <i7>, <f32>
    %20 = buffer %12#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer3"} : <f32>
    %21 = mulf %dataResult, %20 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "mulf0", internal_delay = "2_875333"} : <f32>
    %22 = buffer %12#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer2"} : <f32>
    %23 = mulf %22, %17 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "mulf1", internal_delay = "2_875333"} : <f32>
    %24 = addf %21, %23 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "addf0", internal_delay = "2_922000"} : <f32>
    %25 = buffer %24, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer5"} : <f32>
    %26 = cmpf ugt, %25, %19 {handshake.bb = 1 : ui32, handshake.name = "cmpf0", internal_delay = "0_000000"} : <f32>
    %27:6 = fork [6] %26 {handshake.bb = 1 : ui32, handshake.name = "fork46"} : <i1>
    %28 = spec_v2_repeating_init %27#1 {handshake.bb = 1 : ui32, handshake.name = "spec_v2_repeating_init0", initToken = 1 : ui1} : <i1>
    %29 = buffer %28, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer6"} : <i1>
    %30:4 = fork [4] %29 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork47"} : <i1>
    %31 = source {handshake.bb = 1 : ui32, handshake.name = "source7"} : <>
    %32 = constant %31 {handshake.bb = 1 : ui32, handshake.name = "constant0", value = true} : <>, <i1>
    %33 = buffer %30#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer7"} : <i1>
    %34 = mux %33 [%32, %27#0] {handshake.bb = 1 : ui32, handshake.name = "mux5"} : <i1>, [<i1>, <i1>] to <i1>
    %35 = buffer %34, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer10"} : <i1>
    %36:6 = fork [6] %35 {handshake.bb = 1 : ui32, handshake.name = "fork48"} : <i1>
    %37 = buffer %81, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer20"} : <>
    %38 = passer %37[%36#0] {handshake.bb = 1 : ui32, handshake.name = "passer6"} : <>, <i1>
    %39 = not %27#5 {handshake.bb = 1 : ui32, handshake.name = "not0"} : <i1>
    %40:3 = fork [3] %39 {handshake.bb = 1 : ui32, handshake.name = "fork49"} : <i1>
    %41 = merge %8#1 {handshake.bb = 1 : ui32, handshake.name = "merge0"} : <i8>
    %42 = merge %12#2 {handshake.bb = 1 : ui32, handshake.name = "merge1"} : <f32>
    %result_1, %index_2 = control_merge [%14#0]  {handshake.bb = 1 : ui32, handshake.name = "control_merge1"} : [<>] to <>, <i1>
    sink %index_2 {handshake.name = "sink4"} : <i1>
    %43 = source {handshake.bb = 1 : ui32, handshake.name = "source2"} : <>
    %44 = constant %43 {handshake.bb = 1 : ui32, handshake.name = "constant9", value = 3.000000e+00 : f32} : <>, <f32>
    %45 = addf %42, %44 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "addf1", internal_delay = "2_922000"} : <f32>
    %46 = passer %48[%40#1] {handshake.bb = 1 : ui32, handshake.name = "passer7"} : <f32>, <i1>
    %47 = buffer %45, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer12"} : <f32>
    %48 = br %47 {handshake.bb = 1 : ui32, handshake.name = "br6"} : <f32>
    %49 = passer %51[%40#2] {handshake.bb = 1 : ui32, handshake.name = "passer8"} : <i8>, <i1>
    %50 = buffer %41, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer11"} : <i8>
    %51 = br %50 {handshake.bb = 1 : ui32, handshake.name = "br7"} : <i8>
    %52 = passer %53[%40#0] {handshake.bb = 1 : ui32, handshake.name = "passer9"} : <>, <i1>
    %53 = br %result_1 {handshake.bb = 1 : ui32, handshake.name = "br8"} : <>
    %54:2 = fork [2] %55 {handshake.bb = 1 : ui32, handshake.name = "fork41"} : <i8>
    %55 = merge %8#2 {handshake.bb = 1 : ui32, handshake.name = "merge2"} : <i8>
    %56 = buffer %58, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer15"} : <i7>
    %57 = passer %56[%27#4] {handshake.bb = 1 : ui32, handshake.name = "passer10"} : <i7>, <i1>
    %58 = trunci %54#0 {handshake.bb = 1 : ui32, handshake.name = "trunci1"} : <i8> to <i7>
    %59 = buffer %61#0, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer16"} : <f32>
    %60 = passer %59[%27#3] {handshake.bb = 1 : ui32, handshake.name = "passer11"} : <f32>, <i1>
    %61:2 = fork [2] %62 {handshake.bb = 1 : ui32, handshake.name = "fork42"} : <f32>
    %62 = merge %12#3 {handshake.bb = 1 : ui32, handshake.name = "merge3"} : <f32>
    %63:2 = fork [2] %result_3 {handshake.bb = 1 : ui32, handshake.name = "fork43"} : <>
    %result_3, %index_4 = control_merge [%14#1]  {handshake.bb = 1 : ui32, handshake.name = "control_merge2"} : [<>] to <>, <i1>
    sink %index_4 {handshake.name = "sink5"} : <i1>
    %64 = constant %63#0 {handshake.bb = 1 : ui32, handshake.name = "constant3", value = 1 : i2} : <>, <i2>
    %65 = passer %66[%27#2] {handshake.bb = 1 : ui32, handshake.name = "passer12"} : <i32>, <i1>
    %66 = extsi %64 {handshake.bb = 1 : ui32, handshake.name = "extsi1"} : <i2> to <i32>
    %67 = source {handshake.bb = 1 : ui32, handshake.name = "source3"} : <>
    %68 = constant %67 {handshake.bb = 1 : ui32, handshake.name = "constant10", value = 1.000000e+00 : f32} : <>, <f32>
    %addressResult_5, %dataResult_6 = store[%57] %60 {handshake.bb = 1 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store0"} : <i7>, <f32>, <i7>, <f32>
    %69 = addf %61#1, %68 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "addf2", internal_delay = "2_922000"} : <f32>
    %70 = br %69 {handshake.bb = 1 : ui32, handshake.name = "br9"} : <f32>
    %71 = br %54#1 {handshake.bb = 1 : ui32, handshake.name = "br10"} : <i8>
    %72 = br %63#1 {handshake.bb = 1 : ui32, handshake.name = "br11"} : <>
    %73 = buffer %30#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer8"} : <i1>
    %74 = buffer %46, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer13"} : <f32>
    %75 = mux %73 [%74, %70] {handshake.bb = 1 : ui32, handshake.name = "mux2"} : <i1>, [<f32>, <f32>] to <f32>
    %76 = mux %30#2 [%49, %71] {handshake.bb = 1 : ui32, handshake.name = "mux3"} : <i1>, [<i8>, <i8>] to <i8>
    %77 = buffer %76, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer18"} : <i8>
    %78 = extsi %77 {handshake.bb = 1 : ui32, handshake.name = "extsi5"} : <i8> to <i9>
    %79 = buffer %30#3, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer9"} : <i1>
    %80 = buffer %52, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer14"} : <>
    %81 = mux %79 [%80, %72] {handshake.bb = 1 : ui32, handshake.name = "mux4"} : <i1>, [<>, <>] to <>
    %82 = source {handshake.bb = 1 : ui32, handshake.name = "source4"} : <>
    %83 = constant %82 {handshake.bb = 1 : ui32, handshake.name = "constant11", value = 1.000000e+00 : f32} : <>, <f32>
    %84 = source {handshake.bb = 1 : ui32, handshake.name = "source5"} : <>
    %85 = constant %84 {handshake.bb = 1 : ui32, handshake.name = "constant4", value = 1 : i2} : <>, <i2>
    %86 = extsi %85 {handshake.bb = 1 : ui32, handshake.name = "extsi6"} : <i2> to <i9>
    %87 = source {handshake.bb = 1 : ui32, handshake.name = "source6"} : <>
    %88 = constant %87 {handshake.bb = 1 : ui32, handshake.name = "constant5", value = 100 : i8} : <>, <i8>
    %89 = extsi %88 {handshake.bb = 1 : ui32, handshake.name = "extsi7"} : <i8> to <i9>
    %90 = passer %92[%36#5] {handshake.bb = 1 : ui32, handshake.name = "passer13"} : <f32>, <i1>
    %91 = buffer %75, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer17"} : <f32>
    %92 = divf %83, %91 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "divf0", internal_delay = "3_812000"} : <f32>
    %93:2 = fork [2] %95 {handshake.bb = 1 : ui32, handshake.name = "fork44"} : <i9>
    %94 = buffer %78, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer19"} : <i9>
    %95 = addi %94, %86 {handshake.bb = 1 : ui32, handshake.name = "addi0"} : <i9>
    %96 = passer %97[%36#4] {handshake.bb = 1 : ui32, handshake.name = "passer14"} : <i8>, <i1>
    %97 = trunci %93#0 {handshake.bb = 1 : ui32, handshake.name = "trunci2"} : <i9> to <i8>
    %98 = passer %101#0[%36#3] {handshake.bb = 1 : ui32, handshake.name = "passer15"} : <i1>, <i1>
    %99 = passer %101#1[%36#2] {handshake.bb = 1 : ui32, handshake.name = "passer16"} : <i1>, <i1>
    %100 = passer %101#2[%36#1] {handshake.bb = 1 : ui32, handshake.name = "passer17"} : <i1>, <i1>
    %101:3 = fork [3] %102 {handshake.bb = 1 : ui32, handshake.name = "fork45"} : <i1>
    %102 = cmpi ult, %93#1, %89 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i9>
    %trueResult, %falseResult = cond_br %98, %96 {handshake.bb = 1 : ui32, handshake.name = "cond_br5"} : <i1>, <i8>
    sink %falseResult {handshake.name = "sink2"} : <i8>
    %trueResult_7, %falseResult_8 = cond_br %99, %90 {handshake.bb = 1 : ui32, handshake.name = "cond_br6"} : <i1>, <f32>
    %trueResult_9, %falseResult_10 = cond_br %100, %38 {handshake.bb = 1 : ui32, handshake.name = "cond_br7"} : <i1>, <>
    %103 = merge %falseResult_8 {handshake.bb = 2 : ui32, handshake.name = "merge4"} : <f32>
    %result_11, %index_12 = control_merge [%falseResult_10]  {handshake.bb = 2 : ui32, handshake.name = "control_merge4"} : [<>] to <>, <i1>
    sink %index_12 {handshake.name = "sink3"} : <i1>
    %104:2 = fork [2] %result_11 {handshake.bb = 2 : ui32, handshake.name = "fork11"} : <>
    end {handshake.bb = 2 : ui32, handshake.name = "end0"} %103, %memEnd_0, %memEnd, %0#1 : <f32>, <>, <>, <>
  }
}

