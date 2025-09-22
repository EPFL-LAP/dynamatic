module {
  handshake.func @golden_ratio(%arg0: !handshake.channel<f32>, %arg1: !handshake.channel<f32>, %arg2: !handshake.control<>, ...) -> (!handshake.channel<f32>, !handshake.control<>) attributes {argNames = ["x0", "x1", "start"], handshake.cfdfcThroughput = #handshake<cfdfcThroughput {"0" = 0.038461538461538498 : f64, "1" = 0.018181818181818181 : f64}>, handshake.cfdfcToBBList = #handshake<cfdfcToBBList {"0" = [2 : ui32], "1" = [1 : ui32, 2 : ui32, 3 : ui32]}>, resNames = ["out0", "end"]} {
    %0:3 = fork [3] %arg2 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork0"} : <>
    %1 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant0", value = false} : <>, <i1>
    %2 = extsi %1 {handshake.bb = 0 : ui32, handshake.name = "extsi3"} : <i1> to <i8>
    %3 = mux %6#0 [%2, %trueResult] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i8>, <i8>] to <i8>
    %4 = mux %6#1 [%arg1, %trueResult_0] {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mux1"} : <i1>, [<f32>, <f32>] to <f32>
    %5 = mux %6#2 [%arg0, %trueResult_2] {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mux2"} : <i1>, [<f32>, <f32>] to <f32>
    %result, %index = control_merge [%0#2, %trueResult_4]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %6:3 = fork [3] %index {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <i1>
    %7 = buffer %5, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer2"} : <f32>
    %8 = mux %42#0 [%7, %50] {handshake.bb = 2 : ui32, handshake.name = "mux3"} : <i1>, [<f32>, <f32>] to <f32>
    %9 = buffer %8, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer3"} : <f32>
    %10:4 = fork [4] %9 {handshake.bb = 2 : ui32, handshake.name = "fork2"} : <f32>
    %11 = buffer %3, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer0"} : <i8>
    %12 = mux %42#1 [%11, %46] {handshake.bb = 2 : ui32, handshake.name = "mux4"} : <i1>, [<i8>, <i8>] to <i8>
    %13 = buffer %12, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer7"} : <i8>
    %14 = buffer %13, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer8"} : <i8>
    %15:2 = fork [2] %14 {handshake.bb = 2 : ui32, handshake.name = "fork11"} : <i8>
    %16 = buffer %4, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer1"} : <f32>
    %17 = mux %42#2 [%16, %52] {handshake.bb = 2 : ui32, handshake.name = "mux5"} : <i1>, [<f32>, <f32>] to <f32>
    %18 = buffer %17, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer9"} : <f32>
    %19:2 = fork [2] %18 {handshake.bb = 2 : ui32, handshake.name = "fork3"} : <f32>
    %20 = mux %42#3 [%result, %48] {handshake.bb = 2 : ui32, handshake.name = "mux6"} : <i1>, [<>, <>] to <>
    %21 = buffer %20, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer11"} : <>
    %22 = buffer %21, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer12"} : <>
    %23:2 = fork [2] %22 {handshake.bb = 2 : ui32, handshake.name = "fork12"} : <>
    %24 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %25 = constant %24 {handshake.bb = 2 : ui32, handshake.name = "constant6", value = 5.000000e-01 : f32} : <>, <f32>
    %26 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %27 = constant %26 {handshake.bb = 2 : ui32, handshake.name = "constant7", value = 1.000000e-01 : f32} : <>, <f32>
    %28 = mulf %10#3, %19#1 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "mulf0", internal_delay = "2_875333"} : <f32>
    %29 = buffer %10#2, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer6"} : <f32>
    %30 = addf %29, %28 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "addf0", internal_delay = "2_922000"} : <f32>
    %31 = mulf %30, %25 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "mulf1", internal_delay = "2_875333"} : <f32>
    %32:2 = fork [2] %31 {handshake.bb = 2 : ui32, handshake.name = "fork5"} : <f32>
    %33 = buffer %10#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer5"} : <f32>
    %34 = subf %32#1, %33 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "subf0", internal_delay = "2_000000"} : <f32>
    %35 = absf %34 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "absf0"} : <f32> to <f32>
    %36 = cmpf olt, %35, %27 {handshake.bb = 2 : ui32, handshake.name = "cmpf0", internal_delay = "0_000000"} : <f32>
    %37:4 = fork [4] %36 {handshake.bb = 2 : ui32, handshake.name = "fork13"} : <i1>
    %38 = not %37#3 {handshake.bb = 2 : ui32, handshake.name = "not0"} : <i1>
    %39:5 = fork [5] %38 {handshake.bb = 2 : ui32, handshake.name = "fork4"} : <i1>
    %40 = buffer %39#4, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer14"} : <i1>
    %41 = init %40 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "init4", initToken = 0 : ui1} : <i1>
    %42:4 = fork [4] %41 {handshake.bb = 2 : ui32, handshake.name = "fork6"} : <i1>
    %43 = buffer %10#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer4"} : <f32>
    %44 = passer %43[%37#2] {handshake.bb = 2 : ui32, handshake.name = "passer0"} : <f32>, <i1>
    %45 = passer %15#1[%37#1] {handshake.bb = 2 : ui32, handshake.name = "passer1"} : <i8>, <i1>
    %46 = passer %15#0[%39#0] {handshake.bb = 2 : ui32, handshake.name = "passer2"} : <i8>, <i1>
    %47 = passer %23#0[%37#0] {handshake.bb = 2 : ui32, handshake.name = "passer3"} : <>, <i1>
    %48 = passer %23#1[%39#1] {handshake.bb = 2 : ui32, handshake.name = "passer4"} : <>, <i1>
    %49 = buffer %32#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer13"} : <f32>
    %50 = passer %49[%39#2] {handshake.bb = 2 : ui32, handshake.name = "passer5"} : <f32>, <i1>
    %51 = buffer %19#0, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer10"} : <f32>
    %52 = passer %51[%39#3] {handshake.bb = 2 : ui32, handshake.name = "passer6"} : <f32>, <i1>
    %53 = buffer %45, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 3 : ui32, handshake.name = "buffer15"} : <i8>
    %54 = extsi %53 {handshake.bb = 3 : ui32, handshake.name = "extsi4"} : <i8> to <i9>
    %55 = source {handshake.bb = 3 : ui32, handshake.name = "source2"} : <>
    %56 = constant %55 {handshake.bb = 3 : ui32, handshake.name = "constant8", value = 1.000000e+00 : f32} : <>, <f32>
    %57:2 = fork [2] %56 {handshake.bb = 3 : ui32, handshake.name = "fork7"} : <f32>
    %58 = source {handshake.bb = 3 : ui32, handshake.name = "source3"} : <>
    %59 = constant %58 {handshake.bb = 3 : ui32, handshake.name = "constant1", value = 1 : i2} : <>, <i2>
    %60 = extsi %59 {handshake.bb = 3 : ui32, handshake.name = "extsi5"} : <i2> to <i9>
    %61 = source {handshake.bb = 3 : ui32, handshake.name = "source4"} : <>
    %62 = constant %61 {handshake.bb = 3 : ui32, handshake.name = "constant2", value = 100 : i8} : <>, <i8>
    %63 = extsi %62 {handshake.bb = 3 : ui32, handshake.name = "extsi6"} : <i8> to <i9>
    %64 = addf %44, %57#1 {fastmath = #arith.fastmath<none>, handshake.bb = 3 : ui32, handshake.name = "addf1", internal_delay = "2_922000"} : <f32>
    %65:2 = fork [2] %64 {handshake.bb = 3 : ui32, handshake.name = "fork8"} : <f32>
    %66 = buffer %57#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 3 : ui32, handshake.name = "buffer17"} : <f32>
    %67 = divf %66, %65#1 {fastmath = #arith.fastmath<none>, handshake.bb = 3 : ui32, handshake.name = "divf0", internal_delay = "3_812000"} : <f32>
    %68 = addi %54, %60 {handshake.bb = 3 : ui32, handshake.name = "addi0"} : <i9>
    %69:2 = fork [2] %68 {handshake.bb = 3 : ui32, handshake.name = "fork9"} : <i9>
    %70 = trunci %69#0 {handshake.bb = 3 : ui32, handshake.name = "trunci0"} : <i9> to <i8>
    %71 = cmpi ult, %69#1, %63 {handshake.bb = 3 : ui32, handshake.name = "cmpi0"} : <i9>
    %72:4 = fork [4] %71 {handshake.bb = 3 : ui32, handshake.name = "fork10"} : <i1>
    %trueResult, %falseResult = cond_br %72#0, %70 {handshake.bb = 3 : ui32, handshake.name = "cond_br7"} : <i1>, <i8>
    sink %falseResult {handshake.name = "sink4"} : <i8>
    %trueResult_0, %falseResult_1 = cond_br %72#1, %67 {handshake.bb = 3 : ui32, handshake.name = "cond_br8"} : <i1>, <f32>
    sink %falseResult_1 {handshake.name = "sink5"} : <f32>
    %73 = buffer %65#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 3 : ui32, handshake.name = "buffer18"} : <f32>
    %trueResult_2, %falseResult_3 = cond_br %72#2, %73 {handshake.bb = 3 : ui32, handshake.name = "cond_br9"} : <i1>, <f32>
    %74 = buffer %47, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 3 : ui32, handshake.name = "buffer16"} : <>
    %trueResult_4, %falseResult_5 = cond_br %72#3, %74 {handshake.bb = 3 : ui32, handshake.name = "cond_br10"} : <i1>, <>
    sink %falseResult_5 {handshake.name = "sink6"} : <>
    end {handshake.bb = 4 : ui32, handshake.name = "end0"} %falseResult_3, %0#1 : <f32>, <>
  }
}

