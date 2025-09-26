module {
  handshake.func @if_float(%arg0: !handshake.channel<f32>, %arg1: memref<100xf32>, %arg2: memref<100xf32>, %arg3: !handshake.control<>, %arg4: !handshake.control<>, %arg5: !handshake.control<>, ...) -> (!handshake.channel<f32>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["x0", "a", "minus_trace", "a_start", "minus_trace_start", "start"], handshake.cfdfcThroughput = #handshake<cfdfcThroughput {"0" = 0.034482758620689655 : f64}>, handshake.cfdfcToBBList = #handshake<cfdfcToBBList {"0" = [1 : ui32]}>, resNames = ["out0", "a_end", "minus_trace_end", "end"]} {
    %0:3 = fork [3] %arg5 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork0"} : <>
    %memEnd = mem_controller[%arg2 : memref<100xf32>] %arg4 (%57, %addressResult_1, %dataResult_2) %88#1 {connectedBlocks = [1 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller2"} :    (!handshake.channel<i32>, !handshake.channel<i7>, !handshake.channel<f32>) -> ()
    %outputs, %memEnd_0 = mem_controller[%arg1 : memref<100xf32>] %arg3 (%addressResult) %88#0 {connectedBlocks = [1 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller3"} :    (!handshake.channel<i7>) -> !handshake.channel<f32>
    %1 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant2", value = false} : <>, <i1>
    %2 = extsi %1 {handshake.bb = 0 : ui32, handshake.name = "extsi4"} : <i1> to <i8>
    %3 = mux %14#0 [%2, %trueResult] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i8>, <i8>] to <i8>
    %4 = buffer %3, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer0"} : <i8>
    %5 = buffer %4, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer1"} : <i8>
    %6:3 = fork [3] %5 {handshake.bb = 1 : ui32, handshake.name = "fork14"} : <i8>
    %7 = trunci %6#0 {handshake.bb = 1 : ui32, handshake.name = "trunci0"} : <i8> to <i7>
    %8 = buffer %14#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer5"} : <i1>
    %9 = mux %8 [%arg0, %trueResult_3] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i1>, [<f32>, <f32>] to <f32>
    %10 = buffer %9, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer3"} : <f32>
    %11:4 = fork [4] %10 {handshake.bb = 1 : ui32, handshake.name = "fork15"} : <f32>
    %result, %index = control_merge [%0#2, %trueResult_5]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %12 = buffer %result, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer4"} : <>
    %13:2 = fork [2] %12 {handshake.bb = 1 : ui32, handshake.name = "fork16"} : <>
    %14:2 = fork [2] %index {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <i1>
    %15 = source {handshake.bb = 1 : ui32, handshake.name = "source0"} : <>
    %16 = constant %15 {handshake.bb = 1 : ui32, handshake.name = "constant6", value = -0.899999976 : f32} : <>, <f32>
    %17 = source {handshake.bb = 1 : ui32, handshake.name = "source1"} : <>
    %18 = constant %17 {handshake.bb = 1 : ui32, handshake.name = "constant7", value = 0.000000e+00 : f32} : <>, <f32>
    %addressResult, %dataResult = load[%7] %outputs {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i7>, <f32>, <i7>, <f32>
    %19 = mulf %dataResult, %11#1 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "mulf0", internal_delay = "2_875333"} : <f32>
    %20 = mulf %11#0, %16 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "mulf1", internal_delay = "2_875333"} : <f32>
    %21 = addf %19, %20 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "addf0", internal_delay = "2_922000"} : <f32>
    %22 = buffer %21, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer6"} : <f32>
    %23 = cmpf ugt, %22, %18 {handshake.bb = 1 : ui32, handshake.name = "cmpf0", internal_delay = "0_000000"} : <f32>
    %24:6 = fork [6] %23 {handshake.bb = 1 : ui32, handshake.name = "fork48"} : <i1>
    %25 = spec_v2_repeating_init %24#1 {handshake.bb = 1 : ui32, handshake.name = "spec_v2_repeating_init0", initToken = 1 : ui1} : <i1>
    %26 = buffer %25, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer7"} : <i1>
    %27:4 = fork [4] %26 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork49"} : <i1>
    %28 = source {handshake.bb = 1 : ui32, handshake.name = "source6"} : <>
    %29 = constant %28 {handshake.bb = 1 : ui32, handshake.name = "constant0", value = true} : <>, <i1>
    %30 = mux %27#0 [%29, %24#0] {handshake.bb = 1 : ui32, handshake.name = "mux5"} : <i1>, [<i1>, <i1>] to <i1>
    %31 = buffer %30, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer8"} : <i1>
    %32:6 = fork [6] %31 {handshake.bb = 1 : ui32, handshake.name = "fork50"} : <i1>
    %33 = buffer %68, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer18"} : <>
    %34 = passer %33[%32#5] {handshake.bb = 1 : ui32, handshake.name = "passer6"} : <>, <i1>
    %35 = not %24#5 {handshake.bb = 1 : ui32, handshake.name = "not0"} : <i1>
    %36 = buffer %35, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer10"} : <i1>
    %37:3 = fork [3] %36 {handshake.bb = 1 : ui32, handshake.name = "fork51"} : <i1>
    %38 = source {handshake.bb = 1 : ui32, handshake.name = "source2"} : <>
    %39 = constant %38 {handshake.bb = 1 : ui32, handshake.name = "constant8", value = 1.100000e+00 : f32} : <>, <f32>
    %40 = buffer %11#2, bufferType = FIFO_BREAK_NONE, numSlots = 2 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer13"} : <f32>
    %41 = mulf %40, %39 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "mulf2", internal_delay = "2_875333"} : <f32>
    %42 = passer %41[%37#1] {handshake.bb = 1 : ui32, handshake.name = "passer7"} : <f32>, <i1>
    %43 = passer %45[%37#2] {handshake.bb = 1 : ui32, handshake.name = "passer8"} : <i8>, <i1>
    %44 = buffer %6#1, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer11"} : <i8>
    %45 = buffer %44, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer12"} : <i8>
    %46 = buffer %13#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer14"} : <>
    %47 = passer %46[%37#0] {handshake.bb = 1 : ui32, handshake.name = "passer9"} : <>, <i1>
    %48:2 = fork [2] %49 {handshake.bb = 1 : ui32, handshake.name = "fork43"} : <i8>
    %49 = buffer %6#2, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer2"} : <i8>
    %50 = passer %51[%24#4] {handshake.bb = 1 : ui32, handshake.name = "passer10"} : <i7>, <i1>
    %51 = trunci %48#0 {handshake.bb = 1 : ui32, handshake.name = "trunci1"} : <i8> to <i7>
    %52 = buffer %54#0, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer15"} : <f32>
    %53 = passer %52[%24#3] {handshake.bb = 1 : ui32, handshake.name = "passer11"} : <f32>, <i1>
    %54:2 = fork [2] %11#3 {handshake.bb = 1 : ui32, handshake.name = "fork44"} : <f32>
    %55:2 = fork [2] %13#1 {handshake.bb = 1 : ui32, handshake.name = "fork45"} : <>
    %56 = constant %55#0 {handshake.bb = 1 : ui32, handshake.name = "constant3", value = 1 : i2} : <>, <i2>
    %57 = passer %58[%24#2] {handshake.bb = 1 : ui32, handshake.name = "passer12"} : <i32>, <i1>
    %58 = extsi %56 {handshake.bb = 1 : ui32, handshake.name = "extsi1"} : <i2> to <i32>
    %59 = source {handshake.bb = 1 : ui32, handshake.name = "source3"} : <>
    %60 = constant %59 {handshake.bb = 1 : ui32, handshake.name = "constant9", value = 1.100000e+00 : f32} : <>, <f32>
    %addressResult_1, %dataResult_2 = store[%50] %53 {handshake.bb = 1 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store0"} : <i7>, <f32>, <i7>, <f32>
    %61 = divf %54#1, %60 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "divf0", internal_delay = "3_812000"} : <f32>
    %62 = mux %27#2 [%42, %61] {handshake.bb = 1 : ui32, handshake.name = "mux2"} : <i1>, [<f32>, <f32>] to <f32>
    %63 = buffer %62, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer16"} : <f32>
    %64:2 = fork [2] %63 {handshake.bb = 1 : ui32, handshake.name = "fork8"} : <f32>
    %65 = mux %27#3 [%43, %48#1] {handshake.bb = 1 : ui32, handshake.name = "mux3"} : <i1>, [<i8>, <i8>] to <i8>
    %66 = buffer %65, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer17"} : <i8>
    %67 = extsi %66 {handshake.bb = 1 : ui32, handshake.name = "extsi5"} : <i8> to <i9>
    %68 = mux %27#1 [%47, %55#1] {handshake.bb = 1 : ui32, handshake.name = "mux4"} : <i1>, [<>, <>] to <>
    %69 = source {handshake.bb = 1 : ui32, handshake.name = "source4"} : <>
    %70 = constant %69 {handshake.bb = 1 : ui32, handshake.name = "constant4", value = 1 : i2} : <>, <i2>
    %71 = extsi %70 {handshake.bb = 1 : ui32, handshake.name = "extsi6"} : <i2> to <i9>
    %72 = source {handshake.bb = 1 : ui32, handshake.name = "source5"} : <>
    %73 = constant %72 {handshake.bb = 1 : ui32, handshake.name = "constant12", value = 100 : i8} : <>, <i8>
    %74 = extsi %73 {handshake.bb = 1 : ui32, handshake.name = "extsi7"} : <i8> to <i9>
    %75 = buffer %32#4, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer9"} : <i1>
    %76 = passer %77[%75] {handshake.bb = 1 : ui32, handshake.name = "passer13"} : <f32>, <i1>
    %77 = addf %64#0, %64#1 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "addf1", internal_delay = "2_922000"} : <f32>
    %78:2 = fork [2] %79 {handshake.bb = 1 : ui32, handshake.name = "fork46"} : <i9>
    %79 = addi %67, %71 {handshake.bb = 1 : ui32, handshake.name = "addi0"} : <i9>
    %80 = passer %81[%32#3] {handshake.bb = 1 : ui32, handshake.name = "passer14"} : <i8>, <i1>
    %81 = trunci %78#0 {handshake.bb = 1 : ui32, handshake.name = "trunci2"} : <i9> to <i8>
    %82 = passer %85#0[%32#2] {handshake.bb = 1 : ui32, handshake.name = "passer15"} : <i1>, <i1>
    %83 = passer %85#1[%32#1] {handshake.bb = 1 : ui32, handshake.name = "passer16"} : <i1>, <i1>
    %84 = passer %85#2[%32#0] {handshake.bb = 1 : ui32, handshake.name = "passer17"} : <i1>, <i1>
    %85:3 = fork [3] %86 {handshake.bb = 1 : ui32, handshake.name = "fork47"} : <i1>
    %86 = cmpi ult, %78#1, %74 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i9>
    %trueResult, %falseResult = cond_br %82, %80 {handshake.bb = 1 : ui32, handshake.name = "cond_br5"} : <i1>, <i8>
    sink %falseResult {handshake.name = "sink2"} : <i8>
    %87 = buffer %83, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer19"} : <i1>
    %trueResult_3, %falseResult_4 = cond_br %87, %76 {handshake.bb = 1 : ui32, handshake.name = "cond_br6"} : <i1>, <f32>
    %trueResult_5, %falseResult_6 = cond_br %84, %34 {handshake.bb = 1 : ui32, handshake.name = "cond_br7"} : <i1>, <>
    %88:2 = fork [2] %falseResult_6 {handshake.bb = 2 : ui32, handshake.name = "fork12"} : <>
    end {handshake.bb = 2 : ui32, handshake.name = "end0"} %falseResult_4, %memEnd_0, %memEnd, %0#1 : <f32>, <>, <>, <>
  }
}

