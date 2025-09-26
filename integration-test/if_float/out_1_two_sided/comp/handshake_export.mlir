module {
  handshake.func @if_float(%arg0: !handshake.channel<f32>, %arg1: memref<100xf32>, %arg2: memref<100xf32>, %arg3: !handshake.control<>, %arg4: !handshake.control<>, %arg5: !handshake.control<>, ...) -> (!handshake.channel<f32>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["x0", "a", "minus_trace", "a_start", "minus_trace_start", "start"], handshake.cfdfcThroughput = #handshake<cfdfcThroughput {"0" = 0.034482758620689655 : f64}>, handshake.cfdfcToBBList = #handshake<cfdfcToBBList {"0" = [1 : ui32]}>, resNames = ["out0", "a_end", "minus_trace_end", "end"]} {
    %0:3 = fork [3] %arg5 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork0"} : <>
    %memEnd = mem_controller[%arg2 : memref<100xf32>] %arg4 (%48, %addressResult_1, %dataResult_2) %81#1 {connectedBlocks = [1 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller2"} :    (!handshake.channel<i32>, !handshake.channel<i7>, !handshake.channel<f32>) -> ()
    %outputs, %memEnd_0 = mem_controller[%arg1 : memref<100xf32>] %arg3 (%addressResult) %81#0 {connectedBlocks = [1 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller3"} :    (!handshake.channel<i7>) -> !handshake.channel<f32>
    %1 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant2", value = false} : <>, <i1>
    %2 = extsi %1 {handshake.bb = 0 : ui32, handshake.name = "extsi4"} : <i1> to <i8>
    %3 = mux %14#0 [%2, %trueResult] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i8>, <i8>] to <i8>
    %4 = buffer %3, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer1"} : <i8>
    %5 = buffer %4, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer2"} : <i8>
    %6:3 = fork [3] %5 {handshake.bb = 1 : ui32, handshake.name = "fork14"} : <i8>
    %7 = trunci %6#0 {handshake.bb = 1 : ui32, handshake.name = "trunci0"} : <i8> to <i7>
    %8 = buffer %14#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer7"} : <i1>
    %9 = mux %8 [%arg0, %trueResult_3] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i1>, [<f32>, <f32>] to <f32>
    %10 = buffer %9, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer5"} : <f32>
    %11:4 = fork [4] %10 {handshake.bb = 1 : ui32, handshake.name = "fork15"} : <f32>
    %result, %index = control_merge [%0#2, %trueResult_5]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %12 = buffer %result, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer6"} : <>
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
    %22 = cmpf ugt, %21, %18 {handshake.bb = 1 : ui32, handshake.name = "cmpf0", internal_delay = "0_000000"} : <f32>
    %23:5 = fork [5] %22 {handshake.bb = 1 : ui32, handshake.name = "fork48"} : <i1>
    %24 = buffer %28#4, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer0"} : <i1>
    %25 = buffer %24, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer8"} : <i1>
    %26 = init %25 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "init0", initToken = 0 : ui1} : <i1>
    %27 = not %26 {handshake.bb = 1 : ui32, handshake.name = "not1"} : <i1>
    %28:5 = fork [5] %27 {handshake.bb = 1 : ui32, handshake.name = "fork49"} : <i1>
    %29 = mux %28#3 [%34, %23#0] {handshake.bb = 1 : ui32, handshake.name = "mux5"} : <i1>, [<i1>, <i1>] to <i1>
    %30 = buffer %29, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer10"} : <i1>
    %31:6 = fork [6] %30 {handshake.bb = 1 : ui32, handshake.name = "fork50"} : <i1>
    %32 = buffer %61, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer16"} : <>
    %33 = passer %32[%31#0] {handshake.bb = 1 : ui32, handshake.name = "passer6"} : <>, <i1>
    %34 = not %23#4 {handshake.bb = 1 : ui32, handshake.name = "not0"} : <i1>
    %35 = buffer %6#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer3"} : <i8>
    %36 = source {handshake.bb = 1 : ui32, handshake.name = "source2"} : <>
    %37 = constant %36 {handshake.bb = 1 : ui32, handshake.name = "constant8", value = 1.100000e+00 : f32} : <>, <f32>
    %38 = mulf %11#2, %37 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "mulf2", internal_delay = "2_875333"} : <f32>
    %39:2 = fork [2] %40 {handshake.bb = 1 : ui32, handshake.name = "fork43"} : <i8>
    %40 = buffer %6#2, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer4"} : <i8>
    %41 = passer %42[%23#3] {handshake.bb = 1 : ui32, handshake.name = "passer7"} : <i7>, <i1>
    %42 = trunci %39#0 {handshake.bb = 1 : ui32, handshake.name = "trunci1"} : <i8> to <i7>
    %43 = buffer %45#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer13"} : <f32>
    %44 = passer %43[%23#2] {handshake.bb = 1 : ui32, handshake.name = "passer8"} : <f32>, <i1>
    %45:2 = fork [2] %11#3 {handshake.bb = 1 : ui32, handshake.name = "fork44"} : <f32>
    %46:2 = fork [2] %13#1 {handshake.bb = 1 : ui32, handshake.name = "fork45"} : <>
    %47 = constant %46#0 {handshake.bb = 1 : ui32, handshake.name = "constant3", value = 1 : i2} : <>, <i2>
    %48 = passer %49[%23#1] {handshake.bb = 1 : ui32, handshake.name = "passer9"} : <i32>, <i1>
    %49 = extsi %47 {handshake.bb = 1 : ui32, handshake.name = "extsi1"} : <i2> to <i32>
    %50 = source {handshake.bb = 1 : ui32, handshake.name = "source3"} : <>
    %51 = constant %50 {handshake.bb = 1 : ui32, handshake.name = "constant9", value = 1.100000e+00 : f32} : <>, <f32>
    %addressResult_1, %dataResult_2 = store[%41] %44 {handshake.bb = 1 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store0"} : <i7>, <f32>, <i7>, <f32>
    %52 = divf %45#1, %51 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "divf0", internal_delay = "3_812000"} : <f32>
    %53 = buffer %28#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer9"} : <i1>
    %54 = buffer %38, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer12"} : <f32>
    %55 = mux %53 [%54, %52] {handshake.bb = 1 : ui32, handshake.name = "mux2"} : <i1>, [<f32>, <f32>] to <f32>
    %56 = buffer %55, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer14"} : <f32>
    %57:2 = fork [2] %56 {handshake.bb = 1 : ui32, handshake.name = "fork8"} : <f32>
    %58 = mux %28#2 [%35, %39#1] {handshake.bb = 1 : ui32, handshake.name = "mux3"} : <i1>, [<i8>, <i8>] to <i8>
    %59 = buffer %58, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer15"} : <i8>
    %60 = extsi %59 {handshake.bb = 1 : ui32, handshake.name = "extsi5"} : <i8> to <i9>
    %61 = mux %28#0 [%13#0, %46#1] {handshake.bb = 1 : ui32, handshake.name = "mux4"} : <i1>, [<>, <>] to <>
    %62 = source {handshake.bb = 1 : ui32, handshake.name = "source4"} : <>
    %63 = constant %62 {handshake.bb = 1 : ui32, handshake.name = "constant4", value = 1 : i2} : <>, <i2>
    %64 = extsi %63 {handshake.bb = 1 : ui32, handshake.name = "extsi6"} : <i2> to <i9>
    %65 = source {handshake.bb = 1 : ui32, handshake.name = "source5"} : <>
    %66 = constant %65 {handshake.bb = 1 : ui32, handshake.name = "constant12", value = 100 : i8} : <>, <i8>
    %67 = extsi %66 {handshake.bb = 1 : ui32, handshake.name = "extsi7"} : <i8> to <i9>
    %68 = buffer %31#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer11"} : <i1>
    %69 = passer %70[%68] {handshake.bb = 1 : ui32, handshake.name = "passer10"} : <f32>, <i1>
    %70 = addf %57#0, %57#1 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "addf1", internal_delay = "2_922000"} : <f32>
    %71:2 = fork [2] %72 {handshake.bb = 1 : ui32, handshake.name = "fork46"} : <i9>
    %72 = addi %60, %64 {handshake.bb = 1 : ui32, handshake.name = "addi0"} : <i9>
    %73 = passer %74[%31#5] {handshake.bb = 1 : ui32, handshake.name = "passer11"} : <i8>, <i1>
    %74 = trunci %71#0 {handshake.bb = 1 : ui32, handshake.name = "trunci2"} : <i9> to <i8>
    %75 = passer %78#0[%31#4] {handshake.bb = 1 : ui32, handshake.name = "passer12"} : <i1>, <i1>
    %76 = passer %78#1[%31#3] {handshake.bb = 1 : ui32, handshake.name = "passer13"} : <i1>, <i1>
    %77 = passer %78#2[%31#2] {handshake.bb = 1 : ui32, handshake.name = "passer14"} : <i1>, <i1>
    %78:3 = fork [3] %79 {handshake.bb = 1 : ui32, handshake.name = "fork47"} : <i1>
    %79 = cmpi ult, %71#1, %67 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i9>
    %trueResult, %falseResult = cond_br %75, %73 {handshake.bb = 1 : ui32, handshake.name = "cond_br5"} : <i1>, <i8>
    sink %falseResult {handshake.name = "sink2"} : <i8>
    %80 = buffer %76, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer17"} : <i1>
    %trueResult_3, %falseResult_4 = cond_br %80, %69 {handshake.bb = 1 : ui32, handshake.name = "cond_br6"} : <i1>, <f32>
    %trueResult_5, %falseResult_6 = cond_br %77, %33 {handshake.bb = 1 : ui32, handshake.name = "cond_br7"} : <i1>, <>
    %81:2 = fork [2] %falseResult_6 {handshake.bb = 2 : ui32, handshake.name = "fork12"} : <>
    end {handshake.bb = 2 : ui32, handshake.name = "end0"} %falseResult_4, %memEnd_0, %memEnd, %0#1 : <f32>, <>, <>, <>
  }
}

