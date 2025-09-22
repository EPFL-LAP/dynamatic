module {
  handshake.func @golden_ratio(%arg0: !handshake.channel<f32>, %arg1: !handshake.channel<f32>, %arg2: !handshake.control<>, ...) -> (!handshake.channel<f32>, !handshake.control<>) attributes {argNames = ["x0", "x1", "start"], handshake.cfdfcThroughput = #handshake<cfdfcThroughput {"0" = 0.058823529411764705 : f64, "1" = 0.036363636363636362 : f64}>, handshake.cfdfcToBBList = #handshake<cfdfcToBBList {"0" = [2 : ui32], "1" = [2 : ui32, 3 : ui32, 1 : ui32]}>, resNames = ["out0", "end"]} {
    %0:3 = fork [3] %arg2 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork0"} : <>
    %1 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant0", value = false} : <>, <i1>
    %2 = extsi %1 {handshake.bb = 0 : ui32, handshake.name = "extsi3"} : <i1> to <i8>
    %3 = mux %7#0 [%2, %trueResult] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i8>, <i8>] to <i8>
    %4 = buffer %7#1, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer3"} : <i1>
    %5 = mux %4 [%arg1, %trueResult_0] {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mux1"} : <i1>, [<f32>, <f32>] to <f32>
    %6 = mux %7#2 [%arg0, %trueResult_2] {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mux2"} : <i1>, [<f32>, <f32>] to <f32>
    %result, %index = control_merge [%0#2, %trueResult_4]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %7:3 = fork [3] %index {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <i1>
    %8 = buffer %6, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer2"} : <f32>
    %9 = mux %41#3 [%8, %24] {handshake.bb = 2 : ui32, handshake.name = "mux3"} : <i1>, [<f32>, <f32>] to <f32>
    %10 = buffer %3, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer0"} : <i8>
    %11 = mux %41#2 [%10, %46] {handshake.bb = 2 : ui32, handshake.name = "mux4"} : <i1>, [<i8>, <i8>] to <i8>
    %12 = buffer %5, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer1"} : <f32>
    %13 = buffer %41#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer16"} : <i1>
    %14 = mux %13 [%12, %59] {handshake.bb = 2 : ui32, handshake.name = "mux5"} : <i1>, [<f32>, <f32>] to <f32>
    %15 = mux %41#0 [%result, %52] {handshake.bb = 2 : ui32, handshake.name = "mux6"} : <i1>, [<>, <>] to <>
    %16 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %17 = constant %16 {handshake.bb = 2 : ui32, handshake.name = "constant6", value = 5.000000e-01 : f32} : <>, <f32>
    %18 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %19 = constant %18 {handshake.bb = 2 : ui32, handshake.name = "constant7", value = 1.000000e-01 : f32} : <>, <f32>
    %20 = buffer %57#3, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer21"} : <f32>
    %21 = mulf %20, %61#1 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "mulf0", internal_delay = "2_875333"} : <f32>
    %22 = buffer %57#2, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer20"} : <f32>
    %23 = addf %22, %21 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "addf0", internal_delay = "2_922000"} : <f32>
    %24 = passer %25#0[%38#1] {handshake.bb = 2 : ui32, handshake.name = "passer7"} : <f32>, <i1>
    %25:2 = fork [2] %26 {handshake.bb = 2 : ui32, handshake.name = "fork4"} : <f32>
    %26 = mulf %23, %17 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "mulf1", internal_delay = "2_875333"} : <f32>
    %27 = buffer %57#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer19"} : <f32>
    %28 = subf %25#1, %27 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "subf0", internal_delay = "2_000000"} : <f32>
    %29 = absf %28 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "absf0"} : <f32> to <f32>
    %30 = cmpf olt, %29, %19 {handshake.bb = 2 : ui32, handshake.name = "cmpf0", internal_delay = "0_000000"} : <f32>
    %31:2 = fork [2] %30 {handshake.bb = 2 : ui32, handshake.name = "fork6"} : <i1>
    %32 = buffer %38#2, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer12"} : <i1>
    %33 = passer %34[%32] {handshake.bb = 2 : ui32, handshake.name = "passer8"} : <i1>, <i1>
    %34 = not %31#0 {handshake.bb = 2 : ui32, handshake.name = "not0"} : <i1>
    %35 = spec_v2_repeating_init %33 {handshake.bb = 2 : ui32, handshake.name = "spec_v2_repeating_init0", initToken = 1 : ui1, specv2_top_ri = true} : <i1>
    %36 = buffer %35, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer10"} : <i1>
    %37 = buffer %36, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer11"} : <i1>
    %38:7 = fork [7] %37 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork2"} : <i1>
    %39 = buffer %38#6, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer15"} : <i1>
    %40 = init %39 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "init4", initToken = 0 : ui1} : <i1>
    %41:4 = fork [4] %40 {handshake.bb = 2 : ui32, handshake.name = "fork3"} : <i1>
    %42 = buffer %38#5, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer14"} : <i1>
    %43 = andi %31#1, %42 {handshake.bb = 2 : ui32, handshake.name = "andi0"} : <i1>
    %44:8 = fork [8] %43 {handshake.bb = 2 : ui32, handshake.name = "fork33"} : <i1>
    %45 = buffer %38#4, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer13"} : <i1>
    %46 = passer %49#0[%45] {handshake.bb = 2 : ui32, handshake.name = "passer9"} : <i8>, <i1>
    %47 = buffer %11, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer5"} : <i8>
    %48 = buffer %47, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer6"} : <i8>
    %49:2 = fork [2] %48 {handshake.bb = 2 : ui32, handshake.name = "fork26"} : <i8>
    %50 = buffer %55#0, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer18"} : <>
    %51 = passer %50[%44#0] {handshake.bb = 2 : ui32, handshake.name = "passer3"} : <>, <i1>
    %52 = passer %55#1[%38#3] {handshake.bb = 2 : ui32, handshake.name = "passer10"} : <>, <i1>
    %53 = buffer %15, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer8"} : <>
    %54 = buffer %53, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer9"} : <>
    %55:2 = fork [2] %54 {handshake.bb = 2 : ui32, handshake.name = "fork27"} : <>
    %56 = buffer %9, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer4"} : <f32>
    %57:4 = fork [4] %56 {handshake.bb = 2 : ui32, handshake.name = "fork28"} : <f32>
    %58 = buffer %61#0, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer22"} : <f32>
    %59 = passer %58[%38#0] {handshake.bb = 2 : ui32, handshake.name = "passer11"} : <f32>, <i1>
    %60 = buffer %14, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer7"} : <f32>
    %61:2 = fork [2] %60 {handshake.bb = 2 : ui32, handshake.name = "fork29"} : <f32>
    %62 = extsi %49#1 {handshake.bb = 3 : ui32, handshake.name = "extsi4"} : <i8> to <i9>
    %63 = source {handshake.bb = 3 : ui32, handshake.name = "source2"} : <>
    %64 = constant %63 {handshake.bb = 3 : ui32, handshake.name = "constant8", value = 1.000000e+00 : f32} : <>, <f32>
    %65:2 = fork [2] %64 {handshake.bb = 3 : ui32, handshake.name = "fork7"} : <f32>
    %66 = source {handshake.bb = 3 : ui32, handshake.name = "source3"} : <>
    %67 = constant %66 {handshake.bb = 3 : ui32, handshake.name = "constant1", value = 1 : i2} : <>, <i2>
    %68 = extsi %67 {handshake.bb = 3 : ui32, handshake.name = "extsi5"} : <i2> to <i9>
    %69 = source {handshake.bb = 3 : ui32, handshake.name = "source4"} : <>
    %70 = constant %69 {handshake.bb = 3 : ui32, handshake.name = "constant2", value = 100 : i8} : <>, <i8>
    %71 = extsi %70 {handshake.bb = 3 : ui32, handshake.name = "extsi6"} : <i8> to <i9>
    %72 = buffer %74#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 3 : ui32, handshake.name = "buffer24"} : <f32>
    %73 = passer %72[%44#7] {handshake.bb = 3 : ui32, handshake.name = "passer12"} : <f32>, <i1>
    %74:2 = fork [2] %75 {handshake.bb = 3 : ui32, handshake.name = "fork30"} : <f32>
    %75 = addf %57#0, %65#1 {fastmath = #arith.fastmath<none>, handshake.bb = 3 : ui32, handshake.name = "addf1", internal_delay = "2_922000"} : <f32>
    %76 = buffer %44#6, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 3 : ui32, handshake.name = "buffer17"} : <i1>
    %77 = passer %79[%76] {handshake.bb = 3 : ui32, handshake.name = "passer13"} : <f32>, <i1>
    %78 = buffer %65#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 3 : ui32, handshake.name = "buffer23"} : <f32>
    %79 = divf %78, %74#1 {fastmath = #arith.fastmath<none>, handshake.bb = 3 : ui32, handshake.name = "divf0", internal_delay = "3_812000"} : <f32>
    %80:2 = fork [2] %81 {handshake.bb = 3 : ui32, handshake.name = "fork31"} : <i9>
    %81 = addi %62, %68 {handshake.bb = 3 : ui32, handshake.name = "addi0"} : <i9>
    %82 = passer %83[%44#5] {handshake.bb = 3 : ui32, handshake.name = "passer14"} : <i8>, <i1>
    %83 = trunci %80#0 {handshake.bb = 3 : ui32, handshake.name = "trunci0"} : <i9> to <i8>
    %84 = passer %88#0[%44#4] {handshake.bb = 3 : ui32, handshake.name = "passer15"} : <i1>, <i1>
    %85 = passer %88#1[%44#3] {handshake.bb = 3 : ui32, handshake.name = "passer16"} : <i1>, <i1>
    %86 = passer %88#2[%44#2] {handshake.bb = 3 : ui32, handshake.name = "passer17"} : <i1>, <i1>
    %87 = passer %88#3[%44#1] {handshake.bb = 3 : ui32, handshake.name = "passer18"} : <i1>, <i1>
    %88:4 = fork [4] %89 {handshake.bb = 3 : ui32, handshake.name = "fork32"} : <i1>
    %89 = cmpi ult, %80#1, %71 {handshake.bb = 3 : ui32, handshake.name = "cmpi0"} : <i9>
    %trueResult, %falseResult = cond_br %84, %82 {handshake.bb = 3 : ui32, handshake.name = "cond_br7"} : <i1>, <i8>
    sink %falseResult {handshake.name = "sink4"} : <i8>
    %90 = buffer %85, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 3 : ui32, handshake.name = "buffer25"} : <i1>
    %trueResult_0, %falseResult_1 = cond_br %90, %77 {handshake.bb = 3 : ui32, handshake.name = "cond_br8"} : <i1>, <f32>
    sink %falseResult_1 {handshake.name = "sink5"} : <f32>
    %trueResult_2, %falseResult_3 = cond_br %86, %73 {handshake.bb = 3 : ui32, handshake.name = "cond_br9"} : <i1>, <f32>
    %trueResult_4, %falseResult_5 = cond_br %87, %51 {handshake.bb = 3 : ui32, handshake.name = "cond_br10"} : <i1>, <>
    sink %falseResult_5 {handshake.name = "sink6"} : <>
    end {handshake.bb = 4 : ui32, handshake.name = "end0"} %falseResult_3, %0#1 : <f32>, <>
  }
}

