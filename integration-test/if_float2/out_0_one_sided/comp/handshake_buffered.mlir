module {
  handshake.func @if_float2(%arg0: !handshake.channel<f32>, %arg1: memref<100xf32>, %arg2: memref<100xf32>, %arg3: !handshake.control<>, %arg4: !handshake.control<>, %arg5: !handshake.control<>, ...) -> (!handshake.channel<f32>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["x0", "a", "minus_trace", "a_start", "minus_trace_start", "start"], handshake.cfdfcThroughput = #handshake<cfdfcThroughput {"0" = 0.034482758620689655 : f64}>, handshake.cfdfcToBBList = #handshake<cfdfcToBBList {"0" = [1 : ui32]}>, resNames = ["out0", "a_end", "minus_trace_end", "end"]} {
    %0:3 = fork [3] %arg5 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork0"} : <>
    %memEnd = mem_controller[%arg2 : memref<100xf32>] %arg4 (%62, %addressResult_5, %dataResult_6) %102#1 {connectedBlocks = [1 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller2"} :    (!handshake.channel<i32>, !handshake.channel<i7>, !handshake.channel<f32>) -> ()
    %outputs, %memEnd_0 = mem_controller[%arg1 : memref<100xf32>] %arg3 (%addressResult) %102#0 {connectedBlocks = [1 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller3"} :    (!handshake.channel<i7>) -> !handshake.channel<f32>
    %1 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant2", value = false} : <>, <i1>
    %2 = br %1 {handshake.bb = 0 : ui32, handshake.name = "br3"} : <i1>
    %3 = extsi %2 {handshake.bb = 0 : ui32, handshake.name = "extsi4"} : <i1> to <i8>
    %4 = br %arg0 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "br4"} : <f32>
    %5 = br %0#2 {handshake.bb = 0 : ui32, handshake.name = "br5"} : <>
    %6 = mux %14#0 [%3, %trueResult] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i8>, <i8>] to <i8>
    %7 = buffer %6, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer0"} : <i8>
    %8:3 = fork [3] %7 {handshake.bb = 1 : ui32, handshake.name = "fork12"} : <i8>
    %9 = trunci %8#0 {handshake.bb = 1 : ui32, handshake.name = "trunci0"} : <i8> to <i7>
    %10 = mux %14#1 [%4, %trueResult_7] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i1>, [<f32>, <f32>] to <f32>
    %11 = buffer %10, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer1"} : <f32>
    %12:4 = fork [4] %11 {handshake.bb = 1 : ui32, handshake.name = "fork13"} : <f32>
    %result, %index = control_merge [%5, %trueResult_9]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %13:2 = fork [2] %result {handshake.bb = 1 : ui32, handshake.name = "fork14"} : <>
    %14:2 = fork [2] %index {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <i1>
    %15 = source {handshake.bb = 1 : ui32, handshake.name = "source0"} : <>
    %16 = constant %15 {handshake.bb = 1 : ui32, handshake.name = "constant7", value = -0.899999976 : f32} : <>, <f32>
    %17 = source {handshake.bb = 1 : ui32, handshake.name = "source1"} : <>
    %18 = constant %17 {handshake.bb = 1 : ui32, handshake.name = "constant8", value = 0.000000e+00 : f32} : <>, <f32>
    %addressResult, %dataResult = load[%9] %outputs {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i7>, <f32>, <i7>, <f32>
    %19 = buffer %12#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer2"} : <f32>
    %20 = mulf %dataResult, %19 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "mulf0", internal_delay = "2_875333"} : <f32>
    %21 = mulf %12#0, %16 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "mulf1", internal_delay = "2_875333"} : <f32>
    %22 = buffer %21, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer4"} : <f32>
    %23 = addf %20, %22 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "addf0", internal_delay = "2_922000"} : <f32>
    %24 = buffer %23, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer5"} : <f32>
    %25 = cmpf ugt, %24, %18 {handshake.bb = 1 : ui32, handshake.name = "cmpf0", internal_delay = "0_000000"} : <f32>
    %26:7 = fork [7] %25 {handshake.bb = 1 : ui32, handshake.name = "fork46"} : <i1>
    %27 = not %26#6 {handshake.bb = 1 : ui32, handshake.name = "not0"} : <i1>
    %28:2 = fork [2] %27 {handshake.bb = 1 : ui32, handshake.name = "fork47"} : <i1>
    %29 = spec_v2_repeating_init %28#1 {handshake.bb = 1 : ui32, handshake.name = "spec_v2_repeating_init0", initToken = 1 : ui1} : <i1>
    %30 = buffer %29, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer7"} : <i1>
    %31 = not %30 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "not1"} : <i1>
    %32:4 = fork [4] %31 {handshake.bb = 1 : ui32, handshake.name = "fork48"} : <i1>
    %33 = source {handshake.bb = 1 : ui32, handshake.name = "source7"} : <>
    %34 = constant %33 {handshake.bb = 1 : ui32, handshake.name = "constant0", value = true} : <>, <i1>
    %35 = buffer %32#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer8"} : <i1>
    %36 = mux %35 [%28#0, %34] {handshake.bb = 1 : ui32, handshake.name = "mux5"} : <i1>, [<i1>, <i1>] to <i1>
    %37 = buffer %36, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer10"} : <i1>
    %38:6 = fork [6] %37 {handshake.bb = 1 : ui32, handshake.name = "fork49"} : <i1>
    %39 = buffer %79, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer16"} : <>
    %40 = buffer %39, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer17"} : <>
    %41 = passer %40[%38#0] {handshake.bb = 1 : ui32, handshake.name = "passer6"} : <>, <i1>
    %42 = merge %8#1 {handshake.bb = 1 : ui32, handshake.name = "merge0"} : <i8>
    %43 = merge %12#2 {handshake.bb = 1 : ui32, handshake.name = "merge1"} : <f32>
    %result_1, %index_2 = control_merge [%13#0]  {handshake.bb = 1 : ui32, handshake.name = "control_merge1"} : [<>] to <>, <i1>
    sink %index_2 {handshake.name = "sink4"} : <i1>
    %44 = source {handshake.bb = 1 : ui32, handshake.name = "source2"} : <>
    %45 = constant %44 {handshake.bb = 1 : ui32, handshake.name = "constant9", value = 3.000000e+00 : f32} : <>, <f32>
    %46 = addf %43, %45 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "addf1", internal_delay = "2_922000"} : <f32>
    %47 = br %46 {handshake.bb = 1 : ui32, handshake.name = "br6"} : <f32>
    %48 = br %42 {handshake.bb = 1 : ui32, handshake.name = "br7"} : <i8>
    %49 = br %result_1 {handshake.bb = 1 : ui32, handshake.name = "br8"} : <>
    %50 = buffer %52, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer11"} : <i8>
    %51:2 = fork [2] %50 {handshake.bb = 1 : ui32, handshake.name = "fork41"} : <i8>
    %52 = merge %8#2 {handshake.bb = 1 : ui32, handshake.name = "merge2"} : <i8>
    %53 = passer %54[%26#5] {handshake.bb = 1 : ui32, handshake.name = "passer7"} : <i7>, <i1>
    %54 = trunci %51#0 {handshake.bb = 1 : ui32, handshake.name = "trunci1"} : <i8> to <i7>
    %55 = passer %56#0[%26#3] {handshake.bb = 1 : ui32, handshake.name = "passer8"} : <f32>, <i1>
    %56:2 = fork [2] %58 {handshake.bb = 1 : ui32, handshake.name = "fork42"} : <f32>
    %57 = buffer %12#3, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer3"} : <f32>
    %58 = merge %57 {handshake.bb = 1 : ui32, handshake.name = "merge3"} : <f32>
    %59 = buffer %result_3, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer12"} : <>
    %60:2 = fork [2] %59 {handshake.bb = 1 : ui32, handshake.name = "fork43"} : <>
    %result_3, %index_4 = control_merge [%13#1]  {handshake.bb = 1 : ui32, handshake.name = "control_merge2"} : [<>] to <>, <i1>
    sink %index_4 {handshake.name = "sink5"} : <i1>
    %61 = constant %60#0 {handshake.bb = 1 : ui32, handshake.name = "constant3", value = 1 : i2} : <>, <i2>
    %62 = passer %63[%26#1] {handshake.bb = 1 : ui32, handshake.name = "passer9"} : <i32>, <i1>
    %63 = extsi %61 {handshake.bb = 1 : ui32, handshake.name = "extsi1"} : <i2> to <i32>
    %64 = source {handshake.bb = 1 : ui32, handshake.name = "source3"} : <>
    %65 = constant %64 {handshake.bb = 1 : ui32, handshake.name = "constant10", value = 1.000000e+00 : f32} : <>, <f32>
    %addressResult_5, %dataResult_6 = store[%53] %55 {handshake.bb = 1 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store0"} : <i7>, <f32>, <i7>, <f32>
    %66 = addf %56#1, %65 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "addf2", internal_delay = "2_922000"} : <f32>
    %67 = buffer %26#2, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer6"} : <i1>
    %68 = passer %69[%67] {handshake.bb = 1 : ui32, handshake.name = "passer10"} : <f32>, <i1>
    %69 = br %66 {handshake.bb = 1 : ui32, handshake.name = "br9"} : <f32>
    %70 = passer %71[%26#4] {handshake.bb = 1 : ui32, handshake.name = "passer11"} : <i8>, <i1>
    %71 = br %51#1 {handshake.bb = 1 : ui32, handshake.name = "br10"} : <i8>
    %72 = passer %73[%26#0] {handshake.bb = 1 : ui32, handshake.name = "passer12"} : <>, <i1>
    %73 = br %60#1 {handshake.bb = 1 : ui32, handshake.name = "br11"} : <>
    %74 = buffer %32#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer9"} : <i1>
    %75 = mux %74 [%47, %68] {handshake.bb = 1 : ui32, handshake.name = "mux2"} : <i1>, [<f32>, <f32>] to <f32>
    %76 = mux %32#2 [%48, %70] {handshake.bb = 1 : ui32, handshake.name = "mux3"} : <i1>, [<i8>, <i8>] to <i8>
    %77 = buffer %76, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer14"} : <i8>
    %78 = extsi %77 {handshake.bb = 1 : ui32, handshake.name = "extsi5"} : <i8> to <i9>
    %79 = mux %32#3 [%49, %72] {handshake.bb = 1 : ui32, handshake.name = "mux4"} : <i1>, [<>, <>] to <>
    %80 = source {handshake.bb = 1 : ui32, handshake.name = "source4"} : <>
    %81 = constant %80 {handshake.bb = 1 : ui32, handshake.name = "constant11", value = 1.000000e+00 : f32} : <>, <f32>
    %82 = source {handshake.bb = 1 : ui32, handshake.name = "source5"} : <>
    %83 = constant %82 {handshake.bb = 1 : ui32, handshake.name = "constant4", value = 1 : i2} : <>, <i2>
    %84 = extsi %83 {handshake.bb = 1 : ui32, handshake.name = "extsi6"} : <i2> to <i9>
    %85 = source {handshake.bb = 1 : ui32, handshake.name = "source6"} : <>
    %86 = constant %85 {handshake.bb = 1 : ui32, handshake.name = "constant5", value = 100 : i8} : <>, <i8>
    %87 = extsi %86 {handshake.bb = 1 : ui32, handshake.name = "extsi7"} : <i8> to <i9>
    %88 = passer %90[%38#5] {handshake.bb = 1 : ui32, handshake.name = "passer13"} : <f32>, <i1>
    %89 = buffer %75, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer13"} : <f32>
    %90 = divf %81, %89 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "divf0", internal_delay = "3_812000"} : <f32>
    %91:2 = fork [2] %93 {handshake.bb = 1 : ui32, handshake.name = "fork44"} : <i9>
    %92 = buffer %78, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer15"} : <i9>
    %93 = addi %92, %84 {handshake.bb = 1 : ui32, handshake.name = "addi0"} : <i9>
    %94 = passer %95[%38#4] {handshake.bb = 1 : ui32, handshake.name = "passer14"} : <i8>, <i1>
    %95 = trunci %91#0 {handshake.bb = 1 : ui32, handshake.name = "trunci2"} : <i9> to <i8>
    %96 = passer %99#0[%38#3] {handshake.bb = 1 : ui32, handshake.name = "passer15"} : <i1>, <i1>
    %97 = passer %99#1[%38#2] {handshake.bb = 1 : ui32, handshake.name = "passer16"} : <i1>, <i1>
    %98 = passer %99#2[%38#1] {handshake.bb = 1 : ui32, handshake.name = "passer17"} : <i1>, <i1>
    %99:3 = fork [3] %100 {handshake.bb = 1 : ui32, handshake.name = "fork45"} : <i1>
    %100 = cmpi ult, %91#1, %87 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i9>
    %trueResult, %falseResult = cond_br %96, %94 {handshake.bb = 1 : ui32, handshake.name = "cond_br5"} : <i1>, <i8>
    sink %falseResult {handshake.name = "sink2"} : <i8>
    %trueResult_7, %falseResult_8 = cond_br %97, %88 {handshake.bb = 1 : ui32, handshake.name = "cond_br6"} : <i1>, <f32>
    %trueResult_9, %falseResult_10 = cond_br %98, %41 {handshake.bb = 1 : ui32, handshake.name = "cond_br7"} : <i1>, <>
    %101 = merge %falseResult_8 {handshake.bb = 2 : ui32, handshake.name = "merge4"} : <f32>
    %result_11, %index_12 = control_merge [%falseResult_10]  {handshake.bb = 2 : ui32, handshake.name = "control_merge4"} : [<>] to <>, <i1>
    sink %index_12 {handshake.name = "sink3"} : <i1>
    %102:2 = fork [2] %result_11 {handshake.bb = 2 : ui32, handshake.name = "fork11"} : <>
    end {handshake.bb = 2 : ui32, handshake.name = "end0"} %101, %memEnd_0, %memEnd, %0#1 : <f32>, <>, <>, <>
  }
}

