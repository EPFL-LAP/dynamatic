module {
  handshake.func @if_float(%arg0: !handshake.channel<f32>, %arg1: memref<100xf32>, %arg2: memref<100xf32>, %arg3: !handshake.control<>, %arg4: !handshake.control<>, %arg5: !handshake.control<>, ...) -> (!handshake.channel<f32>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["x0", "a", "minus_trace", "a_start", "minus_trace_start", "start"], handshake.cfdfcThroughput = #handshake<cfdfcThroughput {"0" = 0.045454545454545456 : f64}>, handshake.cfdfcToBBList = #handshake<cfdfcToBBList {"0" = [1 : ui32]}>, resNames = ["out0", "a_end", "minus_trace_end", "end"]} {
    %0:3 = fork [3] %arg5 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork0"} : <>
    %memEnd = mem_controller[%arg2 : memref<100xf32>] %arg4 (%46, %addressResult_1, %dataResult_2) %79#1 {connectedBlocks = [1 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller2"} :    (!handshake.channel<i32>, !handshake.channel<i7>, !handshake.channel<f32>) -> ()
    %outputs, %memEnd_0 = mem_controller[%arg1 : memref<100xf32>] %arg3 (%addressResult) %79#0 {connectedBlocks = [1 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller3"} :    (!handshake.channel<i7>) -> !handshake.channel<f32>
    %1 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant2", value = false} : <>, <i1>
    %2 = extsi %1 {handshake.bb = 0 : ui32, handshake.name = "extsi4"} : <i1> to <i8>
    %3 = mux %14#0 [%2, %trueResult] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i8>, <i8>] to <i8>
    %4 = buffer %3, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer0"} : <i8>
    %5 = buffer %4, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer1"} : <i8>
    %6:3 = fork [3] %5 {handshake.bb = 1 : ui32, handshake.name = "fork14"} : <i8>
    %7 = trunci %6#0 {handshake.bb = 1 : ui32, handshake.name = "trunci0"} : <i8> to <i7>
    %8 = buffer %14#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer6"} : <i1>
    %9 = mux %8 [%arg0, %trueResult_3] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i1>, [<f32>, <f32>] to <f32>
    %10 = buffer %9, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer4"} : <f32>
    %11:4 = fork [4] %10 {handshake.bb = 1 : ui32, handshake.name = "fork15"} : <f32>
    %result, %index = control_merge [%0#2, %trueResult_5]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %12 = buffer %result, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer5"} : <>
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
    %23:10 = fork [10] %22 {handshake.bb = 1 : ui32, handshake.name = "fork37"} : <i1>
    %24 = buffer %23#9, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer8"} : <i1>
    %25 = not %24 {handshake.bb = 1 : ui32, handshake.name = "not0"} : <i1>
    %26:3 = fork [3] %25 {handshake.bb = 1 : ui32, handshake.name = "fork38"} : <i1>
    %27 = buffer %6#1, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer2"} : <i8>
    %28 = buffer %27, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer3"} : <i8>
    %29 = source {handshake.bb = 1 : ui32, handshake.name = "source2"} : <>
    %30 = constant %29 {handshake.bb = 1 : ui32, handshake.name = "constant8", value = 1.100000e+00 : f32} : <>, <f32>
    %31 = mulf %11#2, %30 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "mulf2", internal_delay = "2_875333"} : <f32>
    %32 = passer %33[%26#1] {handshake.bb = 1 : ui32, handshake.name = "passer6"} : <f32>, <i1>
    %33 = buffer %31, bufferType = FIFO_BREAK_NONE, numSlots = 2 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer9"} : <f32>
    %34 = passer %28[%26#2] {handshake.bb = 1 : ui32, handshake.name = "passer7"} : <i8>, <i1>
    %35 = buffer %13#0, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer10"} : <>
    %36 = passer %35[%26#0] {handshake.bb = 1 : ui32, handshake.name = "passer8"} : <>, <i1>
    %37 = buffer %6#2, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer11"} : <i8>
    %38:2 = fork [2] %37 {handshake.bb = 1 : ui32, handshake.name = "fork34"} : <i8>
    %39 = passer %40[%23#8] {handshake.bb = 1 : ui32, handshake.name = "passer9"} : <i7>, <i1>
    %40 = trunci %38#0 {handshake.bb = 1 : ui32, handshake.name = "trunci1"} : <i8> to <i7>
    %41 = buffer %43#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer12"} : <f32>
    %42 = passer %41[%23#6] {handshake.bb = 1 : ui32, handshake.name = "passer10"} : <f32>, <i1>
    %43:2 = fork [2] %11#3 {handshake.bb = 1 : ui32, handshake.name = "fork35"} : <f32>
    %44:2 = fork [2] %13#1 {handshake.bb = 1 : ui32, handshake.name = "fork36"} : <>
    %45 = constant %44#0 {handshake.bb = 1 : ui32, handshake.name = "constant3", value = 1 : i2} : <>, <i2>
    %46 = passer %47[%23#4] {handshake.bb = 1 : ui32, handshake.name = "passer11"} : <i32>, <i1>
    %47 = extsi %45 {handshake.bb = 1 : ui32, handshake.name = "extsi1"} : <i2> to <i32>
    %48 = source {handshake.bb = 1 : ui32, handshake.name = "source3"} : <>
    %49 = constant %48 {handshake.bb = 1 : ui32, handshake.name = "constant9", value = 1.100000e+00 : f32} : <>, <f32>
    %addressResult_1, %dataResult_2 = store[%39] %42 {handshake.bb = 1 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store0"} : <i7>, <f32>, <i7>, <f32>
    %50 = divf %43#1, %49 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "divf0", internal_delay = "3_812000"} : <f32>
    %51 = buffer %23#5, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer7"} : <i1>
    %52 = passer %50[%51] {handshake.bb = 1 : ui32, handshake.name = "passer12"} : <f32>, <i1>
    %53 = passer %38#1[%23#7] {handshake.bb = 1 : ui32, handshake.name = "passer13"} : <i8>, <i1>
    %54 = passer %44#1[%23#3] {handshake.bb = 1 : ui32, handshake.name = "passer14"} : <>, <i1>
    %55 = buffer %52, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer13"} : <f32>
    %56 = mux %23#1 [%32, %55] {handshake.bb = 1 : ui32, handshake.name = "mux2"} : <i1>, [<f32>, <f32>] to <f32>
    %57 = buffer %56, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer16"} : <f32>
    %58:2 = fork [2] %57 {handshake.bb = 1 : ui32, handshake.name = "fork8"} : <f32>
    %59 = buffer %53, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer14"} : <i8>
    %60 = mux %23#0 [%34, %59] {handshake.bb = 1 : ui32, handshake.name = "mux3"} : <i1>, [<i8>, <i8>] to <i8>
    %61 = buffer %60, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer17"} : <i8>
    %62 = extsi %61 {handshake.bb = 1 : ui32, handshake.name = "extsi5"} : <i8> to <i9>
    %63 = buffer %54, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer15"} : <>
    %64 = mux %23#2 [%36, %63] {handshake.bb = 1 : ui32, handshake.name = "mux4"} : <i1>, [<>, <>] to <>
    %65 = source {handshake.bb = 1 : ui32, handshake.name = "source4"} : <>
    %66 = constant %65 {handshake.bb = 1 : ui32, handshake.name = "constant4", value = 1 : i2} : <>, <i2>
    %67 = extsi %66 {handshake.bb = 1 : ui32, handshake.name = "extsi6"} : <i2> to <i9>
    %68 = source {handshake.bb = 1 : ui32, handshake.name = "source5"} : <>
    %69 = constant %68 {handshake.bb = 1 : ui32, handshake.name = "constant12", value = 100 : i8} : <>, <i8>
    %70 = extsi %69 {handshake.bb = 1 : ui32, handshake.name = "extsi7"} : <i8> to <i9>
    %71 = addf %58#0, %58#1 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "addf1", internal_delay = "2_922000"} : <f32>
    %72 = addi %62, %67 {handshake.bb = 1 : ui32, handshake.name = "addi0"} : <i9>
    %73:2 = fork [2] %72 {handshake.bb = 1 : ui32, handshake.name = "fork10"} : <i9>
    %74 = trunci %73#0 {handshake.bb = 1 : ui32, handshake.name = "trunci2"} : <i9> to <i8>
    %75 = cmpi ult, %73#1, %70 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i9>
    %76:3 = fork [3] %75 {handshake.bb = 1 : ui32, handshake.name = "fork11"} : <i1>
    %trueResult, %falseResult = cond_br %76#0, %74 {handshake.bb = 1 : ui32, handshake.name = "cond_br5"} : <i1>, <i8>
    sink %falseResult {handshake.name = "sink2"} : <i8>
    %77 = buffer %76#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer19"} : <i1>
    %trueResult_3, %falseResult_4 = cond_br %77, %71 {handshake.bb = 1 : ui32, handshake.name = "cond_br6"} : <i1>, <f32>
    %78 = buffer %64, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer18"} : <>
    %trueResult_5, %falseResult_6 = cond_br %76#2, %78 {handshake.bb = 1 : ui32, handshake.name = "cond_br7"} : <i1>, <>
    %79:2 = fork [2] %falseResult_6 {handshake.bb = 2 : ui32, handshake.name = "fork12"} : <>
    end {handshake.bb = 2 : ui32, handshake.name = "end0"} %falseResult_4, %memEnd_0, %memEnd, %0#1 : <f32>, <>, <>, <>
  }
}

