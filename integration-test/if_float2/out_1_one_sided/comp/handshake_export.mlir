module {
  handshake.func @if_float2(%arg0: !handshake.channel<f32>, %arg1: memref<100xf32>, %arg2: memref<100xf32>, %arg3: !handshake.control<>, %arg4: !handshake.control<>, %arg5: !handshake.control<>, ...) -> (!handshake.channel<f32>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["x0", "a", "minus_trace", "a_start", "minus_trace_start", "start"], handshake.cfdfcThroughput = #handshake<cfdfcThroughput {"0" = 0.034482758620689655 : f64}>, handshake.cfdfcToBBList = #handshake<cfdfcToBBList {"0" = [1 : ui32]}>, resNames = ["out0", "a_end", "minus_trace_end", "end"]} {
    %0:3 = fork [3] %arg5 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork0"} : <>
    %memEnd = mem_controller[%arg2 : memref<100xf32>] %arg4 (%55, %addressResult_1, %dataResult_2) %90#1 {connectedBlocks = [1 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller2"} :    (!handshake.channel<i32>, !handshake.channel<i7>, !handshake.channel<f32>) -> ()
    %outputs, %memEnd_0 = mem_controller[%arg1 : memref<100xf32>] %arg3 (%addressResult) %90#0 {connectedBlocks = [1 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller3"} :    (!handshake.channel<i7>) -> !handshake.channel<f32>
    %1 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant2", value = false} : <>, <i1>
    %2 = extsi %1 {handshake.bb = 0 : ui32, handshake.name = "extsi4"} : <i1> to <i8>
    %3 = mux %12#0 [%2, %trueResult] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i8>, <i8>] to <i8>
    %4 = buffer %3, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer0"} : <i8>
    %5:3 = fork [3] %4 {handshake.bb = 1 : ui32, handshake.name = "fork12"} : <i8>
    %6 = trunci %5#0 {handshake.bb = 1 : ui32, handshake.name = "trunci0"} : <i8> to <i7>
    %7 = mux %12#1 [%arg0, %trueResult_3] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i1>, [<f32>, <f32>] to <f32>
    %8 = buffer %7, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer1"} : <f32>
    %9:4 = fork [4] %8 {handshake.bb = 1 : ui32, handshake.name = "fork13"} : <f32>
    %result, %index = control_merge [%0#2, %trueResult_5]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %10 = buffer %result, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer4"} : <>
    %11:2 = fork [2] %10 {handshake.bb = 1 : ui32, handshake.name = "fork14"} : <>
    %12:2 = fork [2] %index {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <i1>
    %13 = source {handshake.bb = 1 : ui32, handshake.name = "source0"} : <>
    %14 = constant %13 {handshake.bb = 1 : ui32, handshake.name = "constant7", value = -0.899999976 : f32} : <>, <f32>
    %15 = source {handshake.bb = 1 : ui32, handshake.name = "source1"} : <>
    %16 = constant %15 {handshake.bb = 1 : ui32, handshake.name = "constant8", value = 0.000000e+00 : f32} : <>, <f32>
    %addressResult, %dataResult = load[%6] %outputs {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i7>, <f32>, <i7>, <f32>
    %17 = buffer %9#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer3"} : <f32>
    %18 = mulf %dataResult, %17 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "mulf0", internal_delay = "2_875333"} : <f32>
    %19 = buffer %9#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer2"} : <f32>
    %20 = mulf %19, %14 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "mulf1", internal_delay = "2_875333"} : <f32>
    %21 = addf %18, %20 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "addf0", internal_delay = "2_922000"} : <f32>
    %22 = buffer %21, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer5"} : <f32>
    %23 = cmpf ugt, %22, %16 {handshake.bb = 1 : ui32, handshake.name = "cmpf0", internal_delay = "0_000000"} : <f32>
    %24:6 = fork [6] %23 {handshake.bb = 1 : ui32, handshake.name = "fork46"} : <i1>
    %25 = spec_v2_repeating_init %24#1 {handshake.bb = 1 : ui32, handshake.name = "spec_v2_repeating_init0", initToken = 1 : ui1} : <i1>
    %26 = buffer %25, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer6"} : <i1>
    %27:4 = fork [4] %26 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork47"} : <i1>
    %28 = source {handshake.bb = 1 : ui32, handshake.name = "source7"} : <>
    %29 = constant %28 {handshake.bb = 1 : ui32, handshake.name = "constant0", value = true} : <>, <i1>
    %30 = buffer %27#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer7"} : <i1>
    %31 = mux %30 [%29, %24#0] {handshake.bb = 1 : ui32, handshake.name = "mux5"} : <i1>, [<i1>, <i1>] to <i1>
    %32 = buffer %31, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer10"} : <i1>
    %33:6 = fork [6] %32 {handshake.bb = 1 : ui32, handshake.name = "fork48"} : <i1>
    %34 = buffer %68, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer20"} : <>
    %35 = passer %34[%33#0] {handshake.bb = 1 : ui32, handshake.name = "passer6"} : <>, <i1>
    %36 = not %24#5 {handshake.bb = 1 : ui32, handshake.name = "not0"} : <i1>
    %37:3 = fork [3] %36 {handshake.bb = 1 : ui32, handshake.name = "fork49"} : <i1>
    %38 = source {handshake.bb = 1 : ui32, handshake.name = "source2"} : <>
    %39 = constant %38 {handshake.bb = 1 : ui32, handshake.name = "constant9", value = 3.000000e+00 : f32} : <>, <f32>
    %40 = addf %9#2, %39 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "addf1", internal_delay = "2_922000"} : <f32>
    %41 = passer %42[%37#1] {handshake.bb = 1 : ui32, handshake.name = "passer7"} : <f32>, <i1>
    %42 = buffer %40, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer12"} : <f32>
    %43 = passer %44[%37#2] {handshake.bb = 1 : ui32, handshake.name = "passer8"} : <i8>, <i1>
    %44 = buffer %5#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer11"} : <i8>
    %45 = passer %11#0[%37#0] {handshake.bb = 1 : ui32, handshake.name = "passer9"} : <>, <i1>
    %46:2 = fork [2] %5#2 {handshake.bb = 1 : ui32, handshake.name = "fork41"} : <i8>
    %47 = buffer %49, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer15"} : <i7>
    %48 = passer %47[%24#4] {handshake.bb = 1 : ui32, handshake.name = "passer10"} : <i7>, <i1>
    %49 = trunci %46#0 {handshake.bb = 1 : ui32, handshake.name = "trunci1"} : <i8> to <i7>
    %50 = buffer %52#0, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer16"} : <f32>
    %51 = passer %50[%24#3] {handshake.bb = 1 : ui32, handshake.name = "passer11"} : <f32>, <i1>
    %52:2 = fork [2] %9#3 {handshake.bb = 1 : ui32, handshake.name = "fork42"} : <f32>
    %53:2 = fork [2] %11#1 {handshake.bb = 1 : ui32, handshake.name = "fork43"} : <>
    %54 = constant %53#0 {handshake.bb = 1 : ui32, handshake.name = "constant3", value = 1 : i2} : <>, <i2>
    %55 = passer %56[%24#2] {handshake.bb = 1 : ui32, handshake.name = "passer12"} : <i32>, <i1>
    %56 = extsi %54 {handshake.bb = 1 : ui32, handshake.name = "extsi1"} : <i2> to <i32>
    %57 = source {handshake.bb = 1 : ui32, handshake.name = "source3"} : <>
    %58 = constant %57 {handshake.bb = 1 : ui32, handshake.name = "constant10", value = 1.000000e+00 : f32} : <>, <f32>
    %addressResult_1, %dataResult_2 = store[%48] %51 {handshake.bb = 1 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store0"} : <i7>, <f32>, <i7>, <f32>
    %59 = addf %52#1, %58 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "addf2", internal_delay = "2_922000"} : <f32>
    %60 = buffer %27#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer8"} : <i1>
    %61 = buffer %41, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer13"} : <f32>
    %62 = mux %60 [%61, %59] {handshake.bb = 1 : ui32, handshake.name = "mux2"} : <i1>, [<f32>, <f32>] to <f32>
    %63 = mux %27#2 [%43, %46#1] {handshake.bb = 1 : ui32, handshake.name = "mux3"} : <i1>, [<i8>, <i8>] to <i8>
    %64 = buffer %63, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer18"} : <i8>
    %65 = extsi %64 {handshake.bb = 1 : ui32, handshake.name = "extsi5"} : <i8> to <i9>
    %66 = buffer %27#3, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer9"} : <i1>
    %67 = buffer %45, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer14"} : <>
    %68 = mux %66 [%67, %53#1] {handshake.bb = 1 : ui32, handshake.name = "mux4"} : <i1>, [<>, <>] to <>
    %69 = source {handshake.bb = 1 : ui32, handshake.name = "source4"} : <>
    %70 = constant %69 {handshake.bb = 1 : ui32, handshake.name = "constant11", value = 1.000000e+00 : f32} : <>, <f32>
    %71 = source {handshake.bb = 1 : ui32, handshake.name = "source5"} : <>
    %72 = constant %71 {handshake.bb = 1 : ui32, handshake.name = "constant4", value = 1 : i2} : <>, <i2>
    %73 = extsi %72 {handshake.bb = 1 : ui32, handshake.name = "extsi6"} : <i2> to <i9>
    %74 = source {handshake.bb = 1 : ui32, handshake.name = "source6"} : <>
    %75 = constant %74 {handshake.bb = 1 : ui32, handshake.name = "constant5", value = 100 : i8} : <>, <i8>
    %76 = extsi %75 {handshake.bb = 1 : ui32, handshake.name = "extsi7"} : <i8> to <i9>
    %77 = passer %79[%33#5] {handshake.bb = 1 : ui32, handshake.name = "passer13"} : <f32>, <i1>
    %78 = buffer %62, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer17"} : <f32>
    %79 = divf %70, %78 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "divf0", internal_delay = "3_812000"} : <f32>
    %80:2 = fork [2] %82 {handshake.bb = 1 : ui32, handshake.name = "fork44"} : <i9>
    %81 = buffer %65, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer19"} : <i9>
    %82 = addi %81, %73 {handshake.bb = 1 : ui32, handshake.name = "addi0"} : <i9>
    %83 = passer %84[%33#4] {handshake.bb = 1 : ui32, handshake.name = "passer14"} : <i8>, <i1>
    %84 = trunci %80#0 {handshake.bb = 1 : ui32, handshake.name = "trunci2"} : <i9> to <i8>
    %85 = passer %88#0[%33#3] {handshake.bb = 1 : ui32, handshake.name = "passer15"} : <i1>, <i1>
    %86 = passer %88#1[%33#2] {handshake.bb = 1 : ui32, handshake.name = "passer16"} : <i1>, <i1>
    %87 = passer %88#2[%33#1] {handshake.bb = 1 : ui32, handshake.name = "passer17"} : <i1>, <i1>
    %88:3 = fork [3] %89 {handshake.bb = 1 : ui32, handshake.name = "fork45"} : <i1>
    %89 = cmpi ult, %80#1, %76 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i9>
    %trueResult, %falseResult = cond_br %85, %83 {handshake.bb = 1 : ui32, handshake.name = "cond_br5"} : <i1>, <i8>
    sink %falseResult {handshake.name = "sink2"} : <i8>
    %trueResult_3, %falseResult_4 = cond_br %86, %77 {handshake.bb = 1 : ui32, handshake.name = "cond_br6"} : <i1>, <f32>
    %trueResult_5, %falseResult_6 = cond_br %87, %35 {handshake.bb = 1 : ui32, handshake.name = "cond_br7"} : <i1>, <>
    %90:2 = fork [2] %falseResult_6 {handshake.bb = 2 : ui32, handshake.name = "fork11"} : <>
    end {handshake.bb = 2 : ui32, handshake.name = "end0"} %falseResult_4, %memEnd_0, %memEnd, %0#1 : <f32>, <>, <>, <>
  }
}

