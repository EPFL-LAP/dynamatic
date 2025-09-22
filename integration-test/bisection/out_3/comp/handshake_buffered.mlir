module {
  handshake.func @bisection(%arg0: !handshake.channel<f32>, %arg1: !handshake.channel<f32>, %arg2: !handshake.channel<f32>, %arg3: !handshake.control<>, ...) -> (!handshake.channel<f32>, !handshake.control<>) attributes {argNames = ["a", "b", "tol", "start"], handshake.cfdfcThroughput = #handshake<cfdfcThroughput {"0" = 0.076923076923076927 : f64, "1" = 0.033333333333333333 : f64}>, handshake.cfdfcToBBList = #handshake<cfdfcToBBList {"0" = [2 : ui32], "1" = [2 : ui32, 3 : ui32, 1 : ui32]}>, resNames = ["out0", "end"]} {
    %0:3 = fork [3] %arg3 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork0"} : <>
    %1:3 = fork [3] %arg0 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork1"} : <f32>
    %2 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant0", value = false} : <>, <i1>
    %3 = source {handshake.bb = 0 : ui32, handshake.name = "source0"} : <>
    %4 = constant %3 {handshake.bb = 0 : ui32, handshake.name = "constant5", value = -2.000000e+00 : f32} : <>, <f32>
    %5 = mulf %1#1, %1#2 {fastmath = #arith.fastmath<none>, handshake.bb = 0 : ui32, handshake.name = "mulf0", internal_delay = "2_875333"} : <f32>
    %6 = addf %5, %4 {fastmath = #arith.fastmath<none>, handshake.bb = 0 : ui32, handshake.name = "addf0", internal_delay = "2_922000"} : <f32>
    %7 = extsi %2 {handshake.bb = 0 : ui32, handshake.name = "extsi3"} : <i1> to <i8>
    %8 = mux %13#1 [%1#0, %trueResult] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<f32>, <f32>] to <f32>
    %9 = mux %13#2 [%arg1, %trueResult_0] {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mux1"} : <i1>, [<f32>, <f32>] to <f32>
    %10 = mux %13#0 [%7, %trueResult_2] {handshake.bb = 1 : ui32, handshake.name = "mux2"} : <i1>, [<i8>, <i8>] to <i8>
    %11 = mux %13#3 [%6, %trueResult_4] {handshake.bb = 1 : ui32, handshake.name = "mux3"} : <i1>, [<f32>, <f32>] to <f32>
    %12 = mux %13#4 [%arg2, %trueResult_6] {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mux4"} : <i1>, [<f32>, <f32>] to <f32>
    %result, %index = control_merge [%0#2, %trueResult_8]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %13:5 = fork [5] %index {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <i1>
    %14 = buffer %8, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer0"} : <f32>
    %15 = mux %107#5 [%14, %136] {handshake.bb = 2 : ui32, handshake.name = "mux5"} : <i1>, [<f32>, <f32>] to <f32>
    %16 = buffer %9, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer1"} : <f32>
    %17 = mux %107#4 [%16, %34] {handshake.bb = 2 : ui32, handshake.name = "mux6"} : <i1>, [<f32>, <f32>] to <f32>
    %18 = buffer %10, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer2"} : <i8>
    %19 = buffer %107#3, bufferType = FIFO_BREAK_NONE, numSlots = 2 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer33"} : <i1>
    %20 = mux %19 [%18, %87] {handshake.bb = 2 : ui32, handshake.name = "mux7"} : <i1>, [<i8>, <i8>] to <i8>
    %21 = buffer %11, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer3"} : <f32>
    %22 = mux %107#2 [%21, %139] {handshake.bb = 2 : ui32, handshake.name = "mux8"} : <i1>, [<f32>, <f32>] to <f32>
    %23 = buffer %12, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer4"} : <f32>
    %24 = mux %107#1 [%23, %143] {handshake.bb = 2 : ui32, handshake.name = "mux9"} : <i1>, [<f32>, <f32>] to <f32>
    %25 = buffer %107#0, bufferType = FIFO_BREAK_NONE, numSlots = 2 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer32"} : <i1>
    %26 = mux %25 [%result, %148] {handshake.bb = 2 : ui32, handshake.name = "mux15"} : <i1>, [<>, <>] to <>
    %27 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %28 = constant %27 {handshake.bb = 2 : ui32, handshake.name = "constant6", value = -2.000000e+00 : f32} : <>, <f32>
    %29 = source {handshake.bb = 2 : ui32, handshake.name = "source2"} : <>
    %30 = constant %29 {handshake.bb = 2 : ui32, handshake.name = "constant7", value = 5.000000e-01 : f32} : <>, <f32>
    %31 = addf %138#2, %33#2 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "addf1", internal_delay = "2_922000"} : <f32>
    %32 = buffer %17, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer6"} : <f32>
    %33:3 = fork [3] %32 {handshake.bb = 2 : ui32, handshake.name = "fork6"} : <f32>
    %34 = passer %35#3[%104#3] {handshake.bb = 2 : ui32, handshake.name = "passer44"} : <f32>, <i1>
    %35:6 = fork [6] %36 {handshake.bb = 2 : ui32, handshake.name = "fork7"} : <f32>
    %36 = mulf %31, %30 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "mulf1", internal_delay = "2_875333"} : <f32>
    %37 = mulf %35#0, %35#1 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "mulf2", internal_delay = "2_875333"} : <f32>
    %38:3 = fork [3] %39 {handshake.bb = 2 : ui32, handshake.name = "fork9"} : <f32>
    %39 = addf %37, %28 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "addf2", internal_delay = "2_922000"} : <f32>
    %40 = absf %38#2 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "absf0"} : <f32> to <f32>
    %41 = buffer %146#3, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer38"} : <f32>
    %42 = cmpf olt, %40, %41 {handshake.bb = 2 : ui32, handshake.name = "cmpf0", internal_delay = "0_000000"} : <f32>
    %43 = buffer %42, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer20"} : <i1>
    %44:2 = fork [2] %43 {handshake.bb = 2 : ui32, handshake.name = "fork13"} : <i1>
    %45:2 = fork [2] %46 {handshake.bb = 2 : ui32, handshake.name = "fork16"} : <i1>
    %46 = not %44#0 {handshake.bb = 2 : ui32, handshake.name = "not0"} : <i1>
    %47 = andi %120, %62#0 {handshake.bb = 2 : ui32, handshake.name = "andi0"} : <i1>
    %48 = andi %134, %81#0 {handshake.bb = 2 : ui32, handshake.name = "andi1"} : <i1>
    %49 = andi %59#1, %45#0 {handshake.bb = 2 : ui32, handshake.name = "andi2"} : <i1>
    %50 = buffer %35#5, bufferType = FIFO_BREAK_NONE, numSlots = 2 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer18"} : <f32>
    %51 = passer %50[%119#1] {handshake.bb = 2 : ui32, handshake.name = "passer0"} : <f32>, <i1>
    %52 = passer %151#0[%119#0] {handshake.bb = 2 : ui32, handshake.name = "passer2"} : <>, <i1>
    %53 = source {handshake.bb = 2 : ui32, handshake.name = "source3"} : <>
    %54 = constant %53 {handshake.bb = 2 : ui32, handshake.name = "constant8", value = 5.000000e-01 : f32} : <>, <f32>
    %55 = subf %33#1, %138#1 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "subf0", internal_delay = "2_000000"} : <f32>
    %56 = mulf %55, %54 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "mulf3", internal_delay = "2_875333"} : <f32>
    %57 = buffer %60, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer21"} : <i1>
    %58 = buffer %57, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer22"} : <i1>
    %59:2 = fork [2] %58 {handshake.bb = 2 : ui32, handshake.name = "fork17"} : <i1>
    %60 = cmpf olt, %56, %146#2 {handshake.bb = 2 : ui32, handshake.name = "cmpf1", internal_delay = "0_000000"} : <f32>
    %61 = not %59#0 {handshake.bb = 2 : ui32, handshake.name = "not1"} : <i1>
    %62:2 = fork [2] %63 {handshake.bb = 2 : ui32, handshake.name = "fork18"} : <i1>
    %63 = andi %45#1, %61 {handshake.bb = 2 : ui32, handshake.name = "andi3"} : <i1>
    %64 = buffer %35#4, bufferType = FIFO_BREAK_NONE, numSlots = 2 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer17"} : <f32>
    %65 = passer %64[%113#0] {handshake.bb = 2 : ui32, handshake.name = "passer10"} : <f32>, <i1>
    %66 = passer %151#5[%113#1] {handshake.bb = 2 : ui32, handshake.name = "passer12"} : <>, <i1>
    %67 = buffer %20, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer7"} : <i8>
    %68 = extsi %67 {handshake.bb = 2 : ui32, handshake.name = "extsi4"} : <i8> to <i9>
    %69:3 = fork [3] %70 {handshake.bb = 2 : ui32, handshake.name = "fork20"} : <f32>
    %70 = constant %151#1 {handshake.bb = 2 : ui32, handshake.name = "constant9", value = 0.000000e+00 : f32} : <>, <f32>
    %71 = source {handshake.bb = 2 : ui32, handshake.name = "source4"} : <>
    %72 = constant %71 {handshake.bb = 2 : ui32, handshake.name = "constant1", value = 1 : i2} : <>, <i2>
    %73 = extsi %72 {handshake.bb = 2 : ui32, handshake.name = "extsi5"} : <i2> to <i9>
    %74 = source {handshake.bb = 2 : ui32, handshake.name = "source5"} : <>
    %75 = constant %74 {handshake.bb = 2 : ui32, handshake.name = "constant2", value = 100 : i8} : <>, <i8>
    %76 = extsi %75 {handshake.bb = 2 : ui32, handshake.name = "extsi6"} : <i8> to <i9>
    %77 = buffer %142#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer36"} : <f32>
    %78 = mulf %77, %38#1 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "mulf4", internal_delay = "2_875333"} : <f32>
    %79:2 = fork [2] %80 {handshake.bb = 2 : ui32, handshake.name = "fork42"} : <i1>
    %80 = cmpf olt, %78, %69#0 {handshake.bb = 2 : ui32, handshake.name = "cmpf2", internal_delay = "0_000000"} : <f32>
    %81:2 = fork [2] %82 {handshake.bb = 2 : ui32, handshake.name = "fork43"} : <i1>
    %82 = andi %62#1, %79#0 {handshake.bb = 2 : ui32, handshake.name = "andi4"} : <i1>
    %83:2 = fork [2] %85 {handshake.bb = 2 : ui32, handshake.name = "fork44"} : <i9>
    %84 = buffer %68, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer23"} : <i9>
    %85 = addi %84, %73 {handshake.bb = 2 : ui32, handshake.name = "addi0"} : <i9>
    %86 = buffer %104#2, bufferType = FIFO_BREAK_NONE, numSlots = 2 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer29"} : <i1>
    %87 = passer %88#1[%86] {handshake.bb = 2 : ui32, handshake.name = "passer45"} : <i8>, <i1>
    %88:2 = fork [2] %89 {handshake.bb = 2 : ui32, handshake.name = "fork45"} : <i8>
    %89 = trunci %83#0 {handshake.bb = 2 : ui32, handshake.name = "trunci0"} : <i9> to <i8>
    %90:3 = fork [3] %91 {handshake.bb = 2 : ui32, handshake.name = "fork46"} : <i1>
    %91 = cmpi ult, %83#1, %76 {handshake.bb = 2 : ui32, handshake.name = "cmpi0"} : <i9>
    %92 = passer %93[%111#4] {handshake.bb = 2 : ui32, handshake.name = "passer46"} : <i1>, <i1>
    %93 = andi %81#1, %90#0 {handshake.bb = 2 : ui32, handshake.name = "andi5"} : <i1>
    %94 = spec_v2_repeating_init %92 {handshake.bb = 2 : ui32, handshake.name = "spec_v2_repeating_init0", initToken = 1 : ui1, specv2_top_ri = true} : <i1>
    %95 = buffer %94, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer24"} : <i1>
    %96 = buffer %95, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer25"} : <i1>
    %97:2 = fork [2] %96 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork47"} : <i1>
    %98 = spec_v2_repeating_init %97#0 {handshake.bb = 2 : ui32, handshake.name = "spec_v2_repeating_init1", initToken = 1 : ui1} : <i1>
    %99 = buffer %98, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer26"} : <i1>
    %100:2 = fork [2] %99 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork48"} : <i1>
    %101 = andi %97#1, %100#0 {handshake.bb = 2 : ui32, handshake.name = "andi6", specv2_tmp_and = true} : <i1>
    %102 = spec_v2_repeating_init %100#1 {handshake.bb = 2 : ui32, handshake.name = "spec_v2_repeating_init2", initToken = 1 : ui1} : <i1>
    %103 = buffer %102, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer27"} : <i1>
    %104:8 = fork [8] %103 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork3"} : <i1>
    %105 = buffer %104#7, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer31"} : <i1>
    %106 = init %105 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "init6", initToken = 0 : ui1} : <i1>
    %107:6 = fork [6] %106 {handshake.bb = 2 : ui32, handshake.name = "fork4"} : <i1>
    %108 = buffer %104#6, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer30"} : <i1>
    %109 = andi %101, %108 {handshake.bb = 2 : ui32, handshake.name = "andi7", specv2_tmp_and = true} : <i1>
    %110 = buffer %109, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer34"} : <i1>
    %111:5 = fork [5] %110 {handshake.bb = 2 : ui32, handshake.name = "fork50"} : <i1>
    %112 = andi %49, %111#0 {handshake.bb = 2 : ui32, handshake.name = "andi8"} : <i1>
    %113:2 = fork [2] %112 {handshake.bb = 2 : ui32, handshake.name = "fork51"} : <i1>
    %114 = andi %48, %111#1 {handshake.bb = 2 : ui32, handshake.name = "andi9"} : <i1>
    %115:2 = fork [2] %114 {handshake.bb = 2 : ui32, handshake.name = "fork52"} : <i1>
    %116 = andi %47, %111#2 {handshake.bb = 2 : ui32, handshake.name = "andi10"} : <i1>
    %117:8 = fork [8] %116 {handshake.bb = 2 : ui32, handshake.name = "fork53"} : <i1>
    %118 = andi %44#1, %111#3 {handshake.bb = 2 : ui32, handshake.name = "andi11"} : <i1>
    %119:2 = fork [2] %118 {handshake.bb = 2 : ui32, handshake.name = "fork54"} : <i1>
    %120 = not %79#1 {handshake.bb = 2 : ui32, handshake.name = "not2"} : <i1>
    %121 = buffer %146#0, bufferType = FIFO_BREAK_NONE, numSlots = 2 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer37"} : <f32>
    %122 = passer %121[%117#7] {handshake.bb = 2 : ui32, handshake.name = "passer21"} : <f32>, <i1>
    %123 = buffer %35#2, bufferType = FIFO_BREAK_NONE, numSlots = 2 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer16"} : <f32>
    %124 = passer %123[%117#3] {handshake.bb = 2 : ui32, handshake.name = "passer25"} : <f32>, <i1>
    %125 = passer %69#1[%117#1] {handshake.bb = 2 : ui32, handshake.name = "passer27"} : <f32>, <i1>
    %126 = passer %88#0[%117#5] {handshake.bb = 2 : ui32, handshake.name = "passer29"} : <i8>, <i1>
    %127 = passer %90#1[%117#6] {handshake.bb = 2 : ui32, handshake.name = "passer31"} : <i1>, <i1>
    %128 = passer %151#2[%117#2] {handshake.bb = 2 : ui32, handshake.name = "passer33"} : <>, <i1>
    %129 = buffer %33#0, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer14"} : <f32>
    %130 = buffer %129, bufferType = FIFO_BREAK_NONE, numSlots = 2 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer15"} : <f32>
    %131 = passer %130[%117#0] {handshake.bb = 2 : ui32, handshake.name = "passer34"} : <f32>, <i1>
    %132 = buffer %38#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer19"} : <f32>
    %133 = passer %132[%117#4] {handshake.bb = 2 : ui32, handshake.name = "passer35"} : <f32>, <i1>
    %134 = not %90#2 {handshake.bb = 2 : ui32, handshake.name = "not3"} : <i1>
    %135 = buffer %138#0, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer35"} : <f32>
    %136 = passer %135[%104#5] {handshake.bb = 2 : ui32, handshake.name = "passer47"} : <f32>, <i1>
    %137 = buffer %15, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer5"} : <f32>
    %138:3 = fork [3] %137 {handshake.bb = 2 : ui32, handshake.name = "fork55"} : <f32>
    %139 = passer %142#0[%104#4] {handshake.bb = 2 : ui32, handshake.name = "passer48"} : <f32>, <i1>
    %140 = buffer %22, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer8"} : <f32>
    %141 = buffer %140, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer9"} : <f32>
    %142:2 = fork [2] %141 {handshake.bb = 2 : ui32, handshake.name = "fork56"} : <f32>
    %143 = passer %146#1[%104#0] {handshake.bb = 2 : ui32, handshake.name = "passer49"} : <f32>, <i1>
    %144 = buffer %24, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer10"} : <f32>
    %145 = buffer %144, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer11"} : <f32>
    %146:4 = fork [4] %145 {handshake.bb = 2 : ui32, handshake.name = "fork57"} : <f32>
    %147 = buffer %104#1, bufferType = FIFO_BREAK_NONE, numSlots = 2 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer28"} : <i1>
    %148 = passer %151#4[%147] {handshake.bb = 2 : ui32, handshake.name = "passer50"} : <>, <i1>
    %149 = buffer %26, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer12"} : <>
    %150 = buffer %149, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer13"} : <>
    %151:6 = fork [6] %150 {handshake.bb = 2 : ui32, handshake.name = "fork58"} : <>
    %152 = passer %151#3[%115#1] {handshake.bb = 2 : ui32, handshake.name = "passer42"} : <>, <i1>
    %153 = passer %69#2[%115#0] {handshake.bb = 2 : ui32, handshake.name = "passer43"} : <f32>, <i1>
    %154:7 = fork [7] %127 {handshake.bb = 3 : ui32, handshake.name = "fork21"} : <i1>
    %trueResult, %falseResult = cond_br %154#6, %124 {handshake.bb = 3 : ui32, handshake.name = "cond_br38"} : <i1>, <f32>
    sink %falseResult {handshake.name = "sink26"} : <f32>
    %trueResult_0, %falseResult_1 = cond_br %154#5, %131 {handshake.bb = 3 : ui32, handshake.name = "cond_br39"} : <i1>, <f32>
    sink %falseResult_1 {handshake.name = "sink27"} : <f32>
    %trueResult_2, %falseResult_3 = cond_br %154#0, %126 {handshake.bb = 3 : ui32, handshake.name = "cond_br40"} : <i1>, <i8>
    sink %falseResult_3 {handshake.name = "sink28"} : <i8>
    %trueResult_4, %falseResult_5 = cond_br %154#4, %133 {handshake.bb = 3 : ui32, handshake.name = "cond_br41"} : <i1>, <f32>
    sink %falseResult_5 {handshake.name = "sink29"} : <f32>
    %trueResult_6, %falseResult_7 = cond_br %154#3, %122 {handshake.bb = 3 : ui32, handshake.name = "cond_br42"} : <i1>, <f32>
    sink %falseResult_7 {handshake.name = "sink30"} : <f32>
    %trueResult_8, %falseResult_9 = cond_br %154#2, %128 {handshake.bb = 3 : ui32, handshake.name = "cond_br43"} : <i1>, <>
    %trueResult_10, %falseResult_11 = cond_br %154#1, %125 {handshake.bb = 3 : ui32, handshake.name = "cond_br44"} : <i1>, <f32>
    sink %trueResult_10 {handshake.name = "sink31"} : <f32>
    %155 = mux %index_13 [%51, %153, %falseResult_11] {handshake.bb = 4 : ui32, handshake.name = "mux10"} : <i2>, [<f32>, <f32>, <f32>] to <f32>
    %result_12, %index_13 = control_merge [%52, %152, %falseResult_9]  {handshake.bb = 4 : ui32, handshake.name = "control_merge6"} : [<>, <>, <>] to <>, <i2>
    %156 = mux %index_15 [%65, %155] {handshake.bb = 5 : ui32, handshake.name = "mux11"} : <i1>, [<f32>, <f32>] to <f32>
    %result_14, %index_15 = control_merge [%66, %result_12]  {handshake.bb = 5 : ui32, handshake.name = "control_merge7"} : [<>, <>] to <>, <i1>
    sink %result_14 {handshake.name = "sink32"} : <>
    end {handshake.bb = 5 : ui32, handshake.name = "end0"} %156, %0#1 : <f32>, <>
  }
}

