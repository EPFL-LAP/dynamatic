module {
  handshake.func @golden_ratio(%arg0: !handshake.channel<f32>, %arg1: !handshake.channel<f32>, %arg2: !handshake.control<>, ...) -> (!handshake.channel<f32>, !handshake.control<>) attributes {argNames = ["x0", "x1", "start"], handshake.cfdfcThroughput = #handshake<cfdfcThroughput {"0" = 0.058823529411764705 : f64, "1" = 0.018181818181818184 : f64}>, handshake.cfdfcToBBList = #handshake<cfdfcToBBList {"0" = [2 : ui32], "1" = [1 : ui32, 2 : ui32, 3 : ui32]}>, resNames = ["out0", "end"]} {
    %0:3 = fork [3] %arg2 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork0"} : <>
    %1 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant0", value = false} : <>, <i1>
    %2 = extsi %1 {handshake.bb = 0 : ui32, handshake.name = "extsi3"} : <i1> to <i8>
    %3 = mux %6#0 [%2, %trueResult] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i8>, <i8>] to <i8>
    %4 = mux %6#1 [%arg1, %trueResult_0] {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mux1"} : <i1>, [<f32>, <f32>] to <f32>
    %5 = mux %6#2 [%arg0, %trueResult_2] {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mux2"} : <i1>, [<f32>, <f32>] to <f32>
    %result, %index = control_merge [%0#2, %trueResult_4]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %6:3 = fork [3] %index {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <i1>
    %7 = buffer %5, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer2"} : <f32>
    %8 = mux %42#0 [%7, %23] {handshake.bb = 2 : ui32, handshake.name = "mux3"} : <i1>, [<f32>, <f32>] to <f32>
    %9 = buffer %3, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer0"} : <i8>
    %10 = mux %42#1 [%9, %54] {handshake.bb = 2 : ui32, handshake.name = "mux4"} : <i1>, [<i8>, <i8>] to <i8>
    %11 = buffer %4, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer1"} : <f32>
    %12 = mux %42#2 [%11, %65] {handshake.bb = 2 : ui32, handshake.name = "mux5"} : <i1>, [<f32>, <f32>] to <f32>
    %13 = mux %42#3 [%result, %60] {handshake.bb = 2 : ui32, handshake.name = "mux6"} : <i1>, [<>, <>] to <>
    %14 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %15 = constant %14 {handshake.bb = 2 : ui32, handshake.name = "constant6", value = 5.000000e-01 : f32} : <>, <f32>
    %16 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %17 = constant %16 {handshake.bb = 2 : ui32, handshake.name = "constant7", value = 1.000000e-01 : f32} : <>, <f32>
    %18 = mulf %22#3, %67#1 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "mulf0", internal_delay = "2_875333"} : <f32>
    %19 = buffer %22#2, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer11"} : <f32>
    %20 = addf %19, %18 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "addf0", internal_delay = "2_922000"} : <f32>
    %21 = buffer %8, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer3"} : <f32>
    %22:4 = fork [4] %21 {handshake.bb = 2 : ui32, handshake.name = "fork4"} : <f32>
    %23 = passer %24#0[%39#3] {handshake.bb = 2 : ui32, handshake.name = "passer7"} : <f32>, <i1>
    %24:2 = fork [2] %25 {handshake.bb = 2 : ui32, handshake.name = "fork6"} : <f32>
    %25 = mulf %20, %15 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "mulf1", internal_delay = "2_875333"} : <f32>
    %26 = buffer %22#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer10"} : <f32>
    %27 = subf %24#1, %26 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "subf0", internal_delay = "2_000000"} : <f32>
    %28 = absf %27 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "absf0"} : <f32> to <f32>
    %29 = cmpf olt, %28, %17 {handshake.bb = 2 : ui32, handshake.name = "cmpf0", internal_delay = "0_000000"} : <f32>
    %30:2 = fork [2] %29 {handshake.bb = 2 : ui32, handshake.name = "fork15"} : <i1>
    %31 = passer %32[%47#1] {handshake.bb = 2 : ui32, handshake.name = "passer8"} : <i1>, <i1>
    %32 = not %30#0 {handshake.bb = 2 : ui32, handshake.name = "not0"} : <i1>
    %33 = spec_v2_repeating_init %31 {handshake.bb = 2 : ui32, handshake.name = "spec_v2_repeating_init0", initToken = 1 : ui1, specv2_top_ri = true} : <i1>
    %34 = buffer %33, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer12"} : <i1>
    %35 = buffer %34, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer13"} : <i1>
    %36:2 = fork [2] %35 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork16"} : <i1>
    %37 = spec_v2_repeating_init %36#0 {handshake.bb = 2 : ui32, handshake.name = "spec_v2_repeating_init1", initToken = 1 : ui1} : <i1>
    %38 = buffer %37, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer14"} : <i1>
    %39:6 = fork [6] %38 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork2"} : <i1>
    %40 = buffer %39#5, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer16"} : <i1>
    %41 = init %40 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "init4", initToken = 0 : ui1} : <i1>
    %42:4 = fork [4] %41 {handshake.bb = 2 : ui32, handshake.name = "fork3"} : <i1>
    %43 = buffer %39#4, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer15"} : <i1>
    %44 = source {handshake.bb = 2 : ui32, handshake.name = "source5"} : <>
    %45 = constant %44 {handshake.bb = 2 : ui32, handshake.name = "constant3", value = false} : <>, <i1>
    %46 = mux %43 [%45, %36#1] {handshake.bb = 2 : ui32, handshake.name = "mux7"} : <i1>, [<i1>, <i1>] to <i1>
    %47:2 = fork [2] %46 {handshake.bb = 2 : ui32, handshake.name = "fork18"} : <i1>
    %48 = andi %30#1, %47#0 {handshake.bb = 2 : ui32, handshake.name = "andi1"} : <i1>
    %49:3 = fork [3] %48 {handshake.bb = 2 : ui32, handshake.name = "fork19"} : <i1>
    %50 = buffer %22#0, bufferType = FIFO_BREAK_NONE, numSlots = 2 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer9"} : <f32>
    %51 = passer %50[%49#2] {handshake.bb = 2 : ui32, handshake.name = "passer0"} : <f32>, <i1>
    %52 = buffer %57#1, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer17"} : <i8>
    %53 = passer %52[%49#1] {handshake.bb = 2 : ui32, handshake.name = "passer1"} : <i8>, <i1>
    %54 = passer %57#0[%39#0] {handshake.bb = 2 : ui32, handshake.name = "passer9"} : <i8>, <i1>
    %55 = buffer %10, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer4"} : <i8>
    %56 = buffer %55, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer5"} : <i8>
    %57:2 = fork [2] %56 {handshake.bb = 2 : ui32, handshake.name = "fork20"} : <i8>
    %58 = buffer %63#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer19"} : <>
    %59 = passer %58[%49#0] {handshake.bb = 2 : ui32, handshake.name = "passer3"} : <>, <i1>
    %60 = passer %63#1[%39#1] {handshake.bb = 2 : ui32, handshake.name = "passer10"} : <>, <i1>
    %61 = buffer %13, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer7"} : <>
    %62 = buffer %61, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer8"} : <>
    %63:2 = fork [2] %62 {handshake.bb = 2 : ui32, handshake.name = "fork21"} : <>
    %64 = buffer %67#0, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer20"} : <f32>
    %65 = passer %64[%39#2] {handshake.bb = 2 : ui32, handshake.name = "passer11"} : <f32>, <i1>
    %66 = buffer %12, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer6"} : <f32>
    %67:2 = fork [2] %66 {handshake.bb = 2 : ui32, handshake.name = "fork22"} : <f32>
    %68 = extsi %53 {handshake.bb = 3 : ui32, handshake.name = "extsi4"} : <i8> to <i9>
    %69 = source {handshake.bb = 3 : ui32, handshake.name = "source2"} : <>
    %70 = constant %69 {handshake.bb = 3 : ui32, handshake.name = "constant8", value = 1.000000e+00 : f32} : <>, <f32>
    %71:2 = fork [2] %70 {handshake.bb = 3 : ui32, handshake.name = "fork7"} : <f32>
    %72 = source {handshake.bb = 3 : ui32, handshake.name = "source3"} : <>
    %73 = constant %72 {handshake.bb = 3 : ui32, handshake.name = "constant1", value = 1 : i2} : <>, <i2>
    %74 = extsi %73 {handshake.bb = 3 : ui32, handshake.name = "extsi5"} : <i2> to <i9>
    %75 = source {handshake.bb = 3 : ui32, handshake.name = "source4"} : <>
    %76 = constant %75 {handshake.bb = 3 : ui32, handshake.name = "constant2", value = 100 : i8} : <>, <i8>
    %77 = extsi %76 {handshake.bb = 3 : ui32, handshake.name = "extsi6"} : <i8> to <i9>
    %78 = addf %51, %71#1 {fastmath = #arith.fastmath<none>, handshake.bb = 3 : ui32, handshake.name = "addf1", internal_delay = "2_922000"} : <f32>
    %79:2 = fork [2] %78 {handshake.bb = 3 : ui32, handshake.name = "fork8"} : <f32>
    %80 = buffer %71#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 3 : ui32, handshake.name = "buffer21"} : <f32>
    %81 = divf %80, %79#1 {fastmath = #arith.fastmath<none>, handshake.bb = 3 : ui32, handshake.name = "divf0", internal_delay = "3_812000"} : <f32>
    %82 = addi %68, %74 {handshake.bb = 3 : ui32, handshake.name = "addi0"} : <i9>
    %83 = buffer %82, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 3 : ui32, handshake.name = "buffer23"} : <i9>
    %84:2 = fork [2] %83 {handshake.bb = 3 : ui32, handshake.name = "fork9"} : <i9>
    %85 = trunci %84#0 {handshake.bb = 3 : ui32, handshake.name = "trunci0"} : <i9> to <i8>
    %86 = cmpi ult, %84#1, %77 {handshake.bb = 3 : ui32, handshake.name = "cmpi0"} : <i9>
    %87:4 = fork [4] %86 {handshake.bb = 3 : ui32, handshake.name = "fork10"} : <i1>
    %trueResult, %falseResult = cond_br %87#0, %85 {handshake.bb = 3 : ui32, handshake.name = "cond_br7"} : <i1>, <i8>
    sink %falseResult {handshake.name = "sink4"} : <i8>
    %trueResult_0, %falseResult_1 = cond_br %87#1, %81 {handshake.bb = 3 : ui32, handshake.name = "cond_br8"} : <i1>, <f32>
    sink %falseResult_1 {handshake.name = "sink5"} : <f32>
    %88 = buffer %79#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 3 : ui32, handshake.name = "buffer22"} : <f32>
    %trueResult_2, %falseResult_3 = cond_br %87#2, %88 {handshake.bb = 3 : ui32, handshake.name = "cond_br9"} : <i1>, <f32>
    %89 = buffer %59, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 3 : ui32, handshake.name = "buffer18"} : <>
    %trueResult_4, %falseResult_5 = cond_br %87#3, %89 {handshake.bb = 3 : ui32, handshake.name = "cond_br10"} : <i1>, <>
    sink %falseResult_5 {handshake.name = "sink6"} : <>
    end {handshake.bb = 4 : ui32, handshake.name = "end0"} %falseResult_3, %0#1 : <f32>, <>
  }
}

