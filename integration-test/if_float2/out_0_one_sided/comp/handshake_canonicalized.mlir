module {
  handshake.func @if_float2(%arg0: !handshake.channel<f32>, %arg1: memref<100xf32>, %arg2: memref<100xf32>, %arg3: !handshake.control<>, %arg4: !handshake.control<>, %arg5: !handshake.control<>, ...) -> (!handshake.channel<f32>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["x0", "a", "minus_trace", "a_start", "minus_trace_start", "start"], handshake.cfdfcThroughput = #handshake<cfdfcThroughput {"0" = 0.034482758620689655 : f64}>, handshake.cfdfcToBBList = #handshake<cfdfcToBBList {"0" = [1 : ui32]}>, resNames = ["out0", "a_end", "minus_trace_end", "end"]} {
    %0:3 = fork [3] %arg5 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork0"} : <>
    %memEnd = mem_controller[%arg2 : memref<100xf32>] %arg4 (%52, %addressResult_1, %dataResult_2) %88#1 {connectedBlocks = [1 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller2"} :    (!handshake.channel<i32>, !handshake.channel<i7>, !handshake.channel<f32>) -> ()
    %outputs, %memEnd_0 = mem_controller[%arg1 : memref<100xf32>] %arg3 (%addressResult) %88#0 {connectedBlocks = [1 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller3"} :    (!handshake.channel<i7>) -> !handshake.channel<f32>
    %1 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant2", value = false} : <>, <i1>
    %2 = extsi %1 {handshake.bb = 0 : ui32, handshake.name = "extsi4"} : <i1> to <i8>
    %3 = mux %11#0 [%2, %trueResult] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i8>, <i8>] to <i8>
    %4 = buffer %3, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer0"} : <i8>
    %5:3 = fork [3] %4 {handshake.bb = 1 : ui32, handshake.name = "fork12"} : <i8>
    %6 = trunci %5#0 {handshake.bb = 1 : ui32, handshake.name = "trunci0"} : <i8> to <i7>
    %7 = mux %11#1 [%arg0, %trueResult_3] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i1>, [<f32>, <f32>] to <f32>
    %8 = buffer %7, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer1"} : <f32>
    %9:4 = fork [4] %8 {handshake.bb = 1 : ui32, handshake.name = "fork13"} : <f32>
    %result, %index = control_merge [%0#2, %trueResult_5]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %10:2 = fork [2] %result {handshake.bb = 1 : ui32, handshake.name = "fork14"} : <>
    %11:2 = fork [2] %index {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <i1>
    %12 = source {handshake.bb = 1 : ui32, handshake.name = "source0"} : <>
    %13 = constant %12 {handshake.bb = 1 : ui32, handshake.name = "constant7", value = -0.899999976 : f32} : <>, <f32>
    %14 = source {handshake.bb = 1 : ui32, handshake.name = "source1"} : <>
    %15 = constant %14 {handshake.bb = 1 : ui32, handshake.name = "constant8", value = 0.000000e+00 : f32} : <>, <f32>
    %addressResult, %dataResult = load[%6] %outputs {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i7>, <f32>, <i7>, <f32>
    %16 = buffer %9#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer2"} : <f32>
    %17 = mulf %dataResult, %16 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "mulf0", internal_delay = "2_875333"} : <f32>
    %18 = mulf %9#0, %13 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "mulf1", internal_delay = "2_875333"} : <f32>
    %19 = buffer %18, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer4"} : <f32>
    %20 = addf %17, %19 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "addf0", internal_delay = "2_922000"} : <f32>
    %21 = buffer %20, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer5"} : <f32>
    %22 = cmpf ugt, %21, %15 {handshake.bb = 1 : ui32, handshake.name = "cmpf0", internal_delay = "0_000000"} : <f32>
    %23:7 = fork [7] %22 {handshake.bb = 1 : ui32, handshake.name = "fork46"} : <i1>
    %24 = not %23#6 {handshake.bb = 1 : ui32, handshake.name = "not0"} : <i1>
    %25:2 = fork [2] %24 {handshake.bb = 1 : ui32, handshake.name = "fork47"} : <i1>
    %26 = spec_v2_repeating_init %25#1 {handshake.bb = 1 : ui32, handshake.name = "spec_v2_repeating_init0", initToken = 1 : ui1} : <i1>
    %27 = buffer %26, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer7"} : <i1>
    %28 = not %27 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "not1"} : <i1>
    %29:4 = fork [4] %28 {handshake.bb = 1 : ui32, handshake.name = "fork48"} : <i1>
    %30 = source {handshake.bb = 1 : ui32, handshake.name = "source7"} : <>
    %31 = constant %30 {handshake.bb = 1 : ui32, handshake.name = "constant0", value = true} : <>, <i1>
    %32 = buffer %29#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer8"} : <i1>
    %33 = mux %32 [%25#0, %31] {handshake.bb = 1 : ui32, handshake.name = "mux5"} : <i1>, [<i1>, <i1>] to <i1>
    %34 = buffer %33, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer10"} : <i1>
    %35:6 = fork [6] %34 {handshake.bb = 1 : ui32, handshake.name = "fork49"} : <i1>
    %36 = buffer %66, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer16"} : <>
    %37 = buffer %36, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer17"} : <>
    %38 = passer %37[%35#0] {handshake.bb = 1 : ui32, handshake.name = "passer6"} : <>, <i1>
    %39 = source {handshake.bb = 1 : ui32, handshake.name = "source2"} : <>
    %40 = constant %39 {handshake.bb = 1 : ui32, handshake.name = "constant9", value = 3.000000e+00 : f32} : <>, <f32>
    %41 = addf %9#2, %40 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "addf1", internal_delay = "2_922000"} : <f32>
    %42 = buffer %5#2, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer11"} : <i8>
    %43:2 = fork [2] %42 {handshake.bb = 1 : ui32, handshake.name = "fork41"} : <i8>
    %44 = passer %45[%23#5] {handshake.bb = 1 : ui32, handshake.name = "passer7"} : <i7>, <i1>
    %45 = trunci %43#0 {handshake.bb = 1 : ui32, handshake.name = "trunci1"} : <i8> to <i7>
    %46 = passer %47#0[%23#3] {handshake.bb = 1 : ui32, handshake.name = "passer8"} : <f32>, <i1>
    %47:2 = fork [2] %48 {handshake.bb = 1 : ui32, handshake.name = "fork42"} : <f32>
    %48 = buffer %9#3, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer3"} : <f32>
    %49 = buffer %10#1, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer12"} : <>
    %50:2 = fork [2] %49 {handshake.bb = 1 : ui32, handshake.name = "fork43"} : <>
    %51 = constant %50#0 {handshake.bb = 1 : ui32, handshake.name = "constant3", value = 1 : i2} : <>, <i2>
    %52 = passer %53[%23#1] {handshake.bb = 1 : ui32, handshake.name = "passer9"} : <i32>, <i1>
    %53 = extsi %51 {handshake.bb = 1 : ui32, handshake.name = "extsi1"} : <i2> to <i32>
    %54 = source {handshake.bb = 1 : ui32, handshake.name = "source3"} : <>
    %55 = constant %54 {handshake.bb = 1 : ui32, handshake.name = "constant10", value = 1.000000e+00 : f32} : <>, <f32>
    %addressResult_1, %dataResult_2 = store[%44] %46 {handshake.bb = 1 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store0"} : <i7>, <f32>, <i7>, <f32>
    %56 = addf %47#1, %55 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "addf2", internal_delay = "2_922000"} : <f32>
    %57 = buffer %23#2, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer6"} : <i1>
    %58 = passer %56[%57] {handshake.bb = 1 : ui32, handshake.name = "passer10"} : <f32>, <i1>
    %59 = passer %43#1[%23#4] {handshake.bb = 1 : ui32, handshake.name = "passer11"} : <i8>, <i1>
    %60 = passer %50#1[%23#0] {handshake.bb = 1 : ui32, handshake.name = "passer12"} : <>, <i1>
    %61 = buffer %29#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer9"} : <i1>
    %62 = mux %61 [%41, %58] {handshake.bb = 1 : ui32, handshake.name = "mux2"} : <i1>, [<f32>, <f32>] to <f32>
    %63 = mux %29#2 [%5#1, %59] {handshake.bb = 1 : ui32, handshake.name = "mux3"} : <i1>, [<i8>, <i8>] to <i8>
    %64 = buffer %63, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer14"} : <i8>
    %65 = extsi %64 {handshake.bb = 1 : ui32, handshake.name = "extsi5"} : <i8> to <i9>
    %66 = mux %29#3 [%10#0, %60] {handshake.bb = 1 : ui32, handshake.name = "mux4"} : <i1>, [<>, <>] to <>
    %67 = source {handshake.bb = 1 : ui32, handshake.name = "source4"} : <>
    %68 = constant %67 {handshake.bb = 1 : ui32, handshake.name = "constant11", value = 1.000000e+00 : f32} : <>, <f32>
    %69 = source {handshake.bb = 1 : ui32, handshake.name = "source5"} : <>
    %70 = constant %69 {handshake.bb = 1 : ui32, handshake.name = "constant4", value = 1 : i2} : <>, <i2>
    %71 = extsi %70 {handshake.bb = 1 : ui32, handshake.name = "extsi6"} : <i2> to <i9>
    %72 = source {handshake.bb = 1 : ui32, handshake.name = "source6"} : <>
    %73 = constant %72 {handshake.bb = 1 : ui32, handshake.name = "constant5", value = 100 : i8} : <>, <i8>
    %74 = extsi %73 {handshake.bb = 1 : ui32, handshake.name = "extsi7"} : <i8> to <i9>
    %75 = passer %77[%35#5] {handshake.bb = 1 : ui32, handshake.name = "passer13"} : <f32>, <i1>
    %76 = buffer %62, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer13"} : <f32>
    %77 = divf %68, %76 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "divf0", internal_delay = "3_812000"} : <f32>
    %78:2 = fork [2] %80 {handshake.bb = 1 : ui32, handshake.name = "fork44"} : <i9>
    %79 = buffer %65, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer15"} : <i9>
    %80 = addi %79, %71 {handshake.bb = 1 : ui32, handshake.name = "addi0"} : <i9>
    %81 = passer %82[%35#4] {handshake.bb = 1 : ui32, handshake.name = "passer14"} : <i8>, <i1>
    %82 = trunci %78#0 {handshake.bb = 1 : ui32, handshake.name = "trunci2"} : <i9> to <i8>
    %83 = passer %86#0[%35#3] {handshake.bb = 1 : ui32, handshake.name = "passer15"} : <i1>, <i1>
    %84 = passer %86#1[%35#2] {handshake.bb = 1 : ui32, handshake.name = "passer16"} : <i1>, <i1>
    %85 = passer %86#2[%35#1] {handshake.bb = 1 : ui32, handshake.name = "passer17"} : <i1>, <i1>
    %86:3 = fork [3] %87 {handshake.bb = 1 : ui32, handshake.name = "fork45"} : <i1>
    %87 = cmpi ult, %78#1, %74 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i9>
    %trueResult, %falseResult = cond_br %83, %81 {handshake.bb = 1 : ui32, handshake.name = "cond_br5"} : <i1>, <i8>
    sink %falseResult {handshake.name = "sink2"} : <i8>
    %trueResult_3, %falseResult_4 = cond_br %84, %75 {handshake.bb = 1 : ui32, handshake.name = "cond_br6"} : <i1>, <f32>
    %trueResult_5, %falseResult_6 = cond_br %85, %38 {handshake.bb = 1 : ui32, handshake.name = "cond_br7"} : <i1>, <>
    %88:2 = fork [2] %falseResult_6 {handshake.bb = 2 : ui32, handshake.name = "fork11"} : <>
    end {handshake.bb = 2 : ui32, handshake.name = "end0"} %falseResult_4, %memEnd_0, %memEnd, %0#1 : <f32>, <>, <>, <>
  }
}

