module {
  handshake.func @golden_ratio(%arg0: !handshake.channel<f32>, %arg1: !handshake.channel<f32>, %arg2: !handshake.control<>, ...) -> (!handshake.channel<f32>, !handshake.control<>) attributes {argNames = ["x0", "x1", "start"], handshake.cfdfcThroughput = #handshake<cfdfcThroughput {"0" = 0.058823529411764705 : f64, "1" = 0.018181818181818181 : f64}>, handshake.cfdfcToBBList = #handshake<cfdfcToBBList {"0" = [2 : ui32], "1" = [3 : ui32, 1 : ui32, 2 : ui32]}>, resNames = ["out0", "end"]} {
    %0:3 = fork [3] %arg2 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork0"} : <>
    %1 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant0", value = false} : <>, <i1>
    %2 = extsi %1 {handshake.bb = 0 : ui32, handshake.name = "extsi3"} : <i1> to <i8>
    %3 = mux %6#0 [%2, %trueResult] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i8>, <i8>] to <i8>
    %4 = mux %6#1 [%arg1, %trueResult_0] {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mux1"} : <i1>, [<f32>, <f32>] to <f32>
    %5 = mux %6#2 [%arg0, %trueResult_2] {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mux2"} : <i1>, [<f32>, <f32>] to <f32>
    %result, %index = control_merge [%0#2, %trueResult_4]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %6:3 = fork [3] %index {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <i1>
    %7 = buffer %5, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer2"} : <f32>
    %8 = mux %38#3 [%7, %21] {handshake.bb = 2 : ui32, handshake.name = "mux3"} : <i1>, [<f32>, <f32>] to <f32>
    %9 = buffer %3, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer0"} : <i8>
    %10 = mux %38#2 [%9, %46] {handshake.bb = 2 : ui32, handshake.name = "mux4"} : <i1>, [<i8>, <i8>] to <i8>
    %11 = buffer %4, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer1"} : <f32>
    %12 = mux %38#1 [%11, %59] {handshake.bb = 2 : ui32, handshake.name = "mux5"} : <i1>, [<f32>, <f32>] to <f32>
    %13 = mux %38#0 [%result, %52] {handshake.bb = 2 : ui32, handshake.name = "mux6"} : <i1>, [<>, <>] to <>
    %14 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %15 = constant %14 {handshake.bb = 2 : ui32, handshake.name = "constant6", value = 5.000000e-01 : f32} : <>, <f32>
    %16 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %17 = constant %16 {handshake.bb = 2 : ui32, handshake.name = "constant7", value = 1.000000e-01 : f32} : <>, <f32>
    %18 = mulf %57#3, %61#1 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "mulf0", internal_delay = "2_875333"} : <f32>
    %19 = buffer %57#2, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer20"} : <f32>
    %20 = addf %19, %18 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "addf0", internal_delay = "2_922000"} : <f32>
    %21 = passer %22#0[%35#1] {handshake.bb = 2 : ui32, handshake.name = "passer7"} : <f32>, <i1>
    %22:2 = fork [2] %23 {handshake.bb = 2 : ui32, handshake.name = "fork4"} : <f32>
    %23 = mulf %20, %15 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "mulf1", internal_delay = "2_875333"} : <f32>
    %24 = buffer %57#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer19"} : <f32>
    %25 = subf %22#1, %24 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "subf0", internal_delay = "2_000000"} : <f32>
    %26 = absf %25 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "absf0"} : <f32> to <f32>
    %27 = cmpf olt, %26, %17 {handshake.bb = 2 : ui32, handshake.name = "cmpf0", internal_delay = "0_000000"} : <f32>
    %28:2 = fork [2] %27 {handshake.bb = 2 : ui32, handshake.name = "fork6"} : <i1>
    %29 = buffer %35#2, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer11"} : <i1>
    %30 = passer %31[%29] {handshake.bb = 2 : ui32, handshake.name = "passer8"} : <i1>, <i1>
    %31 = not %28#0 {handshake.bb = 2 : ui32, handshake.name = "not0"} : <i1>
    %32 = spec_v2_repeating_init %30 {handshake.bb = 2 : ui32, handshake.name = "spec_v2_repeating_init0", initToken = 1 : ui1, specv2_top_ri = true} : <i1>
    %33 = buffer %32, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer9"} : <i1>
    %34 = buffer %33, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer10"} : <i1>
    %35:7 = fork [7] %34 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork2"} : <i1>
    %36 = buffer %35#6, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer13"} : <i1>
    %37 = init %36 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "init4", initToken = 0 : ui1} : <i1>
    %38:4 = fork [4] %37 {handshake.bb = 2 : ui32, handshake.name = "fork3"} : <i1>
    %39 = buffer %35#5, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer12"} : <i1>
    %40 = andi %28#1, %39 {handshake.bb = 2 : ui32, handshake.name = "andi0"} : <i1>
    %41:3 = fork [3] %40 {handshake.bb = 2 : ui32, handshake.name = "fork16"} : <i1>
    %42 = buffer %57#0, bufferType = FIFO_BREAK_NONE, numSlots = 2 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer18"} : <f32>
    %43 = passer %42[%41#2] {handshake.bb = 2 : ui32, handshake.name = "passer0"} : <f32>, <i1>
    %44 = buffer %49#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer15"} : <i8>
    %45 = passer %44[%41#1] {handshake.bb = 2 : ui32, handshake.name = "passer1"} : <i8>, <i1>
    %46 = passer %49#0[%35#4] {handshake.bb = 2 : ui32, handshake.name = "passer9"} : <i8>, <i1>
    %47 = buffer %10, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer4"} : <i8>
    %48 = buffer %47, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer5"} : <i8>
    %49:2 = fork [2] %48 {handshake.bb = 2 : ui32, handshake.name = "fork17"} : <i8>
    %50 = buffer %55#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer17"} : <>
    %51 = passer %50[%41#0] {handshake.bb = 2 : ui32, handshake.name = "passer3"} : <>, <i1>
    %52 = passer %55#1[%35#3] {handshake.bb = 2 : ui32, handshake.name = "passer10"} : <>, <i1>
    %53 = buffer %13, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer7"} : <>
    %54 = buffer %53, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer8"} : <>
    %55:2 = fork [2] %54 {handshake.bb = 2 : ui32, handshake.name = "fork18"} : <>
    %56 = buffer %8, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer3"} : <f32>
    %57:4 = fork [4] %56 {handshake.bb = 2 : ui32, handshake.name = "fork19"} : <f32>
    %58 = buffer %61#0, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer21"} : <f32>
    %59 = passer %58[%35#0] {handshake.bb = 2 : ui32, handshake.name = "passer11"} : <f32>, <i1>
    %60 = buffer %12, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer6"} : <f32>
    %61:2 = fork [2] %60 {handshake.bb = 2 : ui32, handshake.name = "fork20"} : <f32>
    %62 = buffer %45, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 3 : ui32, handshake.name = "buffer14"} : <i8>
    %63 = extsi %62 {handshake.bb = 3 : ui32, handshake.name = "extsi4"} : <i8> to <i9>
    %64 = source {handshake.bb = 3 : ui32, handshake.name = "source2"} : <>
    %65 = constant %64 {handshake.bb = 3 : ui32, handshake.name = "constant8", value = 1.000000e+00 : f32} : <>, <f32>
    %66:2 = fork [2] %65 {handshake.bb = 3 : ui32, handshake.name = "fork7"} : <f32>
    %67 = source {handshake.bb = 3 : ui32, handshake.name = "source3"} : <>
    %68 = constant %67 {handshake.bb = 3 : ui32, handshake.name = "constant1", value = 1 : i2} : <>, <i2>
    %69 = extsi %68 {handshake.bb = 3 : ui32, handshake.name = "extsi5"} : <i2> to <i9>
    %70 = source {handshake.bb = 3 : ui32, handshake.name = "source4"} : <>
    %71 = constant %70 {handshake.bb = 3 : ui32, handshake.name = "constant2", value = 100 : i8} : <>, <i8>
    %72 = extsi %71 {handshake.bb = 3 : ui32, handshake.name = "extsi6"} : <i8> to <i9>
    %73 = addf %43, %66#1 {fastmath = #arith.fastmath<none>, handshake.bb = 3 : ui32, handshake.name = "addf1", internal_delay = "2_922000"} : <f32>
    %74:2 = fork [2] %73 {handshake.bb = 3 : ui32, handshake.name = "fork8"} : <f32>
    %75 = buffer %66#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 3 : ui32, handshake.name = "buffer22"} : <f32>
    %76 = divf %75, %74#1 {fastmath = #arith.fastmath<none>, handshake.bb = 3 : ui32, handshake.name = "divf0", internal_delay = "3_812000"} : <f32>
    %77 = addi %63, %69 {handshake.bb = 3 : ui32, handshake.name = "addi0"} : <i9>
    %78:2 = fork [2] %77 {handshake.bb = 3 : ui32, handshake.name = "fork9"} : <i9>
    %79 = trunci %78#0 {handshake.bb = 3 : ui32, handshake.name = "trunci0"} : <i9> to <i8>
    %80 = cmpi ult, %78#1, %72 {handshake.bb = 3 : ui32, handshake.name = "cmpi0"} : <i9>
    %81:4 = fork [4] %80 {handshake.bb = 3 : ui32, handshake.name = "fork10"} : <i1>
    %trueResult, %falseResult = cond_br %81#0, %79 {handshake.bb = 3 : ui32, handshake.name = "cond_br7"} : <i1>, <i8>
    sink %falseResult {handshake.name = "sink4"} : <i8>
    %trueResult_0, %falseResult_1 = cond_br %81#1, %76 {handshake.bb = 3 : ui32, handshake.name = "cond_br8"} : <i1>, <f32>
    sink %falseResult_1 {handshake.name = "sink5"} : <f32>
    %82 = buffer %74#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 3 : ui32, handshake.name = "buffer23"} : <f32>
    %trueResult_2, %falseResult_3 = cond_br %81#2, %82 {handshake.bb = 3 : ui32, handshake.name = "cond_br9"} : <i1>, <f32>
    %83 = buffer %51, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 3 : ui32, handshake.name = "buffer16"} : <>
    %trueResult_4, %falseResult_5 = cond_br %81#3, %83 {handshake.bb = 3 : ui32, handshake.name = "cond_br10"} : <i1>, <>
    sink %falseResult_5 {handshake.name = "sink6"} : <>
    end {handshake.bb = 4 : ui32, handshake.name = "end0"} %falseResult_3, %0#1 : <f32>, <>
  }
}

