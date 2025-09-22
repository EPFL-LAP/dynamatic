module {
  handshake.func @golden_ratio(%arg0: !handshake.channel<f32>, %arg1: !handshake.control<>, ...) -> (!handshake.channel<f32>, !handshake.control<>) attributes {argNames = ["x0", "start"], handshake.cfdfcThroughput = #handshake<cfdfcThroughput {"0" = 0.038461538461538464 : f64, "1" = 0.018181818181818184 : f64}>, handshake.cfdfcToBBList = #handshake<cfdfcToBBList {"0" = [2 : ui32], "1" = [2 : ui32, 3 : ui32, 1 : ui32]}>, resNames = ["out0", "end"]} {
    %0:3 = fork [3] %arg1 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork0"} : <>
    %1:2 = fork [2] %arg0 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork1"} : <f32>
    %2 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant0", value = false} : <>, <i1>
    %3 = source {handshake.bb = 0 : ui32, handshake.name = "source0"} : <>
    %4 = constant %3 {handshake.bb = 0 : ui32, handshake.name = "constant5", value = 1.000000e+00 : f32} : <>, <f32>
    %5 = divf %4, %1#1 {fastmath = #arith.fastmath<none>, handshake.bb = 0 : ui32, handshake.name = "divf0", internal_delay = "3_812000"} : <f32>
    %6 = extsi %2 {handshake.bb = 0 : ui32, handshake.name = "extsi3"} : <i1> to <i8>
    %7 = mux %10#1 [%5, %trueResult_10] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<f32>, <f32>] to <f32>
    %8 = mux %10#0 [%6, %trueResult_12] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i1>, [<i8>, <i8>] to <i8>
    %9 = mux %10#2 [%1#0, %trueResult_14] {handshake.bb = 1 : ui32, handshake.name = "mux2"} : <i1>, [<f32>, <f32>] to <f32>
    %result, %index = control_merge [%0#2, %trueResult_16]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %10:3 = fork [3] %index {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <i1>
    %11 = buffer %9, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer2"} : <f32>
    %12 = mux %21#1 [%11, %falseResult_7] {handshake.bb = 2 : ui32, handshake.name = "mux3"} : <i1>, [<f32>, <f32>] to <f32>
    %13 = buffer %12, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer3"} : <f32>
    %14:4 = fork [4] %13 {handshake.bb = 2 : ui32, handshake.name = "fork3"} : <f32>
    %15 = buffer %7, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer0"} : <f32>
    %16 = mux %21#2 [%15, %falseResult_9] {handshake.bb = 2 : ui32, handshake.name = "mux4"} : <i1>, [<f32>, <f32>] to <f32>
    %17 = buffer %16, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer7"} : <f32>
    %18:2 = fork [2] %17 {handshake.bb = 2 : ui32, handshake.name = "fork4"} : <f32>
    %19 = buffer %8, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer1"} : <i8>
    %20 = mux %21#0 [%19, %falseResult_3] {handshake.bb = 2 : ui32, handshake.name = "mux5"} : <i1>, [<i8>, <i8>] to <i8>
    %result_0, %index_1 = control_merge [%result, %falseResult_5]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>, <>] to <>, <i1>
    %21:3 = fork [3] %index_1 {handshake.bb = 2 : ui32, handshake.name = "fork5"} : <i1>
    %22 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %23 = constant %22 {handshake.bb = 2 : ui32, handshake.name = "constant6", value = 5.000000e-01 : f32} : <>, <f32>
    %24 = source {handshake.bb = 2 : ui32, handshake.name = "source2"} : <>
    %25 = constant %24 {handshake.bb = 2 : ui32, handshake.name = "constant7", value = 1.000000e-01 : f32} : <>, <f32>
    %26 = mulf %14#3, %18#1 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "mulf0", internal_delay = "2_875333"} : <f32>
    %27 = buffer %14#2, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer6"} : <f32>
    %28 = addf %27, %26 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "addf0", internal_delay = "2_922000"} : <f32>
    %29 = mulf %28, %23 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "mulf1", internal_delay = "2_875333"} : <f32>
    %30:2 = fork [2] %29 {handshake.bb = 2 : ui32, handshake.name = "fork6"} : <f32>
    %31 = buffer %14#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer5"} : <f32>
    %32 = subf %30#1, %31 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "subf0", internal_delay = "2_000000"} : <f32>
    %33 = absf %32 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "absf0"} : <f32> to <f32>
    %34 = cmpf olt, %33, %25 {handshake.bb = 2 : ui32, handshake.name = "cmpf0", internal_delay = "0_000000"} : <f32>
    %35:5 = fork [5] %34 {handshake.bb = 2 : ui32, handshake.name = "fork7"} : <i1>
    %36 = buffer %14#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer4"} : <f32>
    %trueResult, %falseResult = cond_br %35#4, %36 {handshake.bb = 2 : ui32, handshake.name = "cond_br2"} : <i1>, <f32>
    sink %falseResult {handshake.name = "sink0"} : <f32>
    %37 = buffer %20, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer9"} : <i8>
    %38 = buffer %37, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer10"} : <i8>
    %trueResult_2, %falseResult_3 = cond_br %35#0, %38 {handshake.bb = 2 : ui32, handshake.name = "cond_br3"} : <i1>, <i8>
    %39 = buffer %result_0, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer11"} : <>
    %40 = buffer %35#3, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer13"} : <i1>
    %trueResult_4, %falseResult_5 = cond_br %40, %39 {handshake.bb = 2 : ui32, handshake.name = "cond_br4"} : <i1>, <>
    %41 = buffer %30#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer12"} : <f32>
    %trueResult_6, %falseResult_7 = cond_br %35#2, %41 {handshake.bb = 2 : ui32, handshake.name = "cond_br5"} : <i1>, <f32>
    sink %trueResult_6 {handshake.name = "sink1"} : <f32>
    %42 = buffer %18#0, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer8"} : <f32>
    %trueResult_8, %falseResult_9 = cond_br %35#1, %42 {handshake.bb = 2 : ui32, handshake.name = "cond_br6"} : <i1>, <f32>
    sink %trueResult_8 {handshake.name = "sink2"} : <f32>
    %43 = extsi %trueResult_2 {handshake.bb = 3 : ui32, handshake.name = "extsi4"} : <i8> to <i9>
    %44 = source {handshake.bb = 3 : ui32, handshake.name = "source3"} : <>
    %45 = constant %44 {handshake.bb = 3 : ui32, handshake.name = "constant8", value = 1.000000e+00 : f32} : <>, <f32>
    %46:2 = fork [2] %45 {handshake.bb = 3 : ui32, handshake.name = "fork8"} : <f32>
    %47 = source {handshake.bb = 3 : ui32, handshake.name = "source4"} : <>
    %48 = constant %47 {handshake.bb = 3 : ui32, handshake.name = "constant1", value = 1 : i2} : <>, <i2>
    %49 = extsi %48 {handshake.bb = 3 : ui32, handshake.name = "extsi5"} : <i2> to <i9>
    %50 = source {handshake.bb = 3 : ui32, handshake.name = "source5"} : <>
    %51 = constant %50 {handshake.bb = 3 : ui32, handshake.name = "constant2", value = 100 : i8} : <>, <i8>
    %52 = extsi %51 {handshake.bb = 3 : ui32, handshake.name = "extsi6"} : <i8> to <i9>
    %53 = addf %trueResult, %46#1 {fastmath = #arith.fastmath<none>, handshake.bb = 3 : ui32, handshake.name = "addf1", internal_delay = "2_922000"} : <f32>
    %54:2 = fork [2] %53 {handshake.bb = 3 : ui32, handshake.name = "fork9"} : <f32>
    %55 = addi %43, %49 {handshake.bb = 3 : ui32, handshake.name = "addi0"} : <i9>
    %56 = buffer %55, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 3 : ui32, handshake.name = "buffer16"} : <i9>
    %57:2 = fork [2] %56 {handshake.bb = 3 : ui32, handshake.name = "fork10"} : <i9>
    %58 = trunci %57#0 {handshake.bb = 3 : ui32, handshake.name = "trunci0"} : <i9> to <i8>
    %59 = buffer %46#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 3 : ui32, handshake.name = "buffer14"} : <f32>
    %60 = divf %59, %54#1 {fastmath = #arith.fastmath<none>, handshake.bb = 3 : ui32, handshake.name = "divf1", internal_delay = "3_812000"} : <f32>
    %61 = cmpi ult, %57#1, %52 {handshake.bb = 3 : ui32, handshake.name = "cmpi0"} : <i9>
    %62:4 = fork [4] %61 {handshake.bb = 3 : ui32, handshake.name = "fork11"} : <i1>
    %trueResult_10, %falseResult_11 = cond_br %62#1, %60 {handshake.bb = 3 : ui32, handshake.name = "cond_br7"} : <i1>, <f32>
    sink %falseResult_11 {handshake.name = "sink4"} : <f32>
    %trueResult_12, %falseResult_13 = cond_br %62#0, %58 {handshake.bb = 3 : ui32, handshake.name = "cond_br8"} : <i1>, <i8>
    sink %falseResult_13 {handshake.name = "sink5"} : <i8>
    %63 = buffer %54#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 3 : ui32, handshake.name = "buffer15"} : <f32>
    %trueResult_14, %falseResult_15 = cond_br %62#2, %63 {handshake.bb = 3 : ui32, handshake.name = "cond_br9"} : <i1>, <f32>
    %trueResult_16, %falseResult_17 = cond_br %62#3, %trueResult_4 {handshake.bb = 3 : ui32, handshake.name = "cond_br10"} : <i1>, <>
    sink %falseResult_17 {handshake.name = "sink6"} : <>
    end {handshake.bb = 4 : ui32, handshake.name = "end0"} %falseResult_15, %0#1 : <f32>, <>
  }
}

