module {
  handshake.func @bisection(%arg0: !handshake.channel<f32>, %arg1: !handshake.channel<f32>, %arg2: !handshake.channel<f32>, %arg3: !handshake.control<>, ...) -> (!handshake.channel<f32>, !handshake.control<>) attributes {argNames = ["a", "b", "tol", "start"], handshake.cfdfcThroughput = #handshake<cfdfcThroughput {"0" = 0.066666666666666666 : f64, "1" = 0.033333333333333333 : f64}>, handshake.cfdfcToBBList = #handshake<cfdfcToBBList {"0" = [2 : ui32], "1" = [2 : ui32, 3 : ui32, 1 : ui32]}>, resNames = ["out0", "end"]} {
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
    %15 = mux %99#5 [%14, %127] {handshake.bb = 2 : ui32, handshake.name = "mux5"} : <i1>, [<f32>, <f32>] to <f32>
    %16 = buffer %9, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer1"} : <f32>
    %17 = mux %99#4 [%16, %34] {handshake.bb = 2 : ui32, handshake.name = "mux6"} : <i1>, [<f32>, <f32>] to <f32>
    %18 = buffer %10, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer2"} : <i8>
    %19 = buffer %99#3, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer35"} : <i1>
    %20 = mux %19 [%18, %85] {handshake.bb = 2 : ui32, handshake.name = "mux7"} : <i1>, [<i8>, <i8>] to <i8>
    %21 = buffer %11, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer3"} : <f32>
    %22 = mux %99#2 [%21, %132] {handshake.bb = 2 : ui32, handshake.name = "mux8"} : <i1>, [<f32>, <f32>] to <f32>
    %23 = buffer %12, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer4"} : <f32>
    %24 = buffer %99#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer34"} : <i1>
    %25 = mux %24 [%23, %137] {handshake.bb = 2 : ui32, handshake.name = "mux9"} : <i1>, [<f32>, <f32>] to <f32>
    %26 = buffer %99#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer33"} : <i1>
    %27 = mux %26 [%result, %142] {handshake.bb = 2 : ui32, handshake.name = "mux15"} : <i1>, [<>, <>] to <>
    %28 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %29 = constant %28 {handshake.bb = 2 : ui32, handshake.name = "constant6", value = -2.000000e+00 : f32} : <>, <f32>
    %30 = source {handshake.bb = 2 : ui32, handshake.name = "source2"} : <>
    %31 = constant %30 {handshake.bb = 2 : ui32, handshake.name = "constant7", value = 5.000000e-01 : f32} : <>, <f32>
    %32 = addf %129#2, %131#2 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "addf1", internal_delay = "2_922000"} : <f32>
    %33 = buffer %35#3, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer16"} : <f32>
    %34 = passer %33[%96#6] {handshake.bb = 2 : ui32, handshake.name = "passer44"} : <f32>, <i1>
    %35:6 = fork [6] %36 {handshake.bb = 2 : ui32, handshake.name = "fork6"} : <f32>
    %36 = mulf %32, %31 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "mulf1", internal_delay = "2_875333"} : <f32>
    %37 = mulf %35#0, %35#1 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "mulf2", internal_delay = "2_875333"} : <f32>
    %38:3 = fork [3] %39 {handshake.bb = 2 : ui32, handshake.name = "fork7"} : <f32>
    %39 = addf %37, %29 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "addf2", internal_delay = "2_922000"} : <f32>
    %40 = absf %38#2 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "absf0"} : <f32> to <f32>
    %41 = buffer %40, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer20"} : <f32>
    %42 = cmpf olt, %41, %140#3 {handshake.bb = 2 : ui32, handshake.name = "cmpf0", internal_delay = "0_000000"} : <f32>
    %43:2 = fork [2] %42 {handshake.bb = 2 : ui32, handshake.name = "fork9"} : <i1>
    %44:2 = fork [2] %45 {handshake.bb = 2 : ui32, handshake.name = "fork13"} : <i1>
    %45 = not %43#0 {handshake.bb = 2 : ui32, handshake.name = "not0"} : <i1>
    %46 = andi %125, %80#0 {handshake.bb = 2 : ui32, handshake.name = "andi0"} : <i1>
    %47 = andi %112, %60#0 {handshake.bb = 2 : ui32, handshake.name = "andi1"} : <i1>
    %48 = andi %56#1, %44#0 {handshake.bb = 2 : ui32, handshake.name = "andi2"} : <i1>
    %49 = buffer %35#5, bufferType = FIFO_BREAK_NONE, numSlots = 2 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer18"} : <f32>
    %50 = passer %49[%111#1] {handshake.bb = 2 : ui32, handshake.name = "passer0"} : <f32>, <i1>
    %51 = passer %145#0[%111#0] {handshake.bb = 2 : ui32, handshake.name = "passer2"} : <>, <i1>
    %52 = source {handshake.bb = 2 : ui32, handshake.name = "source3"} : <>
    %53 = constant %52 {handshake.bb = 2 : ui32, handshake.name = "constant8", value = 5.000000e-01 : f32} : <>, <f32>
    %54 = subf %131#1, %129#1 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "subf0", internal_delay = "2_000000"} : <f32>
    %55 = mulf %54, %53 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "mulf3", internal_delay = "2_875333"} : <f32>
    %56:2 = fork [2] %58 {handshake.bb = 2 : ui32, handshake.name = "fork16"} : <i1>
    %57 = buffer %55, bufferType = FIFO_BREAK_NONE, numSlots = 2 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer21"} : <f32>
    %58 = cmpf olt, %57, %140#2 {handshake.bb = 2 : ui32, handshake.name = "cmpf1", internal_delay = "0_000000"} : <f32>
    %59 = not %56#0 {handshake.bb = 2 : ui32, handshake.name = "not1"} : <i1>
    %60:2 = fork [2] %61 {handshake.bb = 2 : ui32, handshake.name = "fork17"} : <i1>
    %61 = andi %44#1, %59 {handshake.bb = 2 : ui32, handshake.name = "andi3"} : <i1>
    %62 = buffer %35#4, bufferType = FIFO_BREAK_NONE, numSlots = 2 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer17"} : <f32>
    %63 = passer %62[%102#1] {handshake.bb = 2 : ui32, handshake.name = "passer10"} : <f32>, <i1>
    %64 = passer %145#5[%102#0] {handshake.bb = 2 : ui32, handshake.name = "passer12"} : <>, <i1>
    %65 = buffer %20, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer7"} : <i8>
    %66 = buffer %65, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer8"} : <i8>
    %67 = extsi %66 {handshake.bb = 2 : ui32, handshake.name = "extsi4"} : <i8> to <i9>
    %68:3 = fork [3] %69 {handshake.bb = 2 : ui32, handshake.name = "fork18"} : <f32>
    %69 = constant %145#1 {handshake.bb = 2 : ui32, handshake.name = "constant9", value = 0.000000e+00 : f32} : <>, <f32>
    %70 = source {handshake.bb = 2 : ui32, handshake.name = "source4"} : <>
    %71 = constant %70 {handshake.bb = 2 : ui32, handshake.name = "constant1", value = 1 : i2} : <>, <i2>
    %72 = extsi %71 {handshake.bb = 2 : ui32, handshake.name = "extsi5"} : <i2> to <i9>
    %73 = source {handshake.bb = 2 : ui32, handshake.name = "source5"} : <>
    %74 = constant %73 {handshake.bb = 2 : ui32, handshake.name = "constant2", value = 100 : i8} : <>, <i8>
    %75 = extsi %74 {handshake.bb = 2 : ui32, handshake.name = "extsi6"} : <i8> to <i9>
    %76 = buffer %135#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer39"} : <f32>
    %77 = mulf %76, %38#1 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "mulf4", internal_delay = "2_875333"} : <f32>
    %78:2 = fork [2] %79 {handshake.bb = 2 : ui32, handshake.name = "fork20"} : <i1>
    %79 = cmpf olt, %77, %68#0 {handshake.bb = 2 : ui32, handshake.name = "cmpf2", internal_delay = "0_000000"} : <f32>
    %80:2 = fork [2] %81 {handshake.bb = 2 : ui32, handshake.name = "fork42"} : <i1>
    %81 = andi %60#1, %78#0 {handshake.bb = 2 : ui32, handshake.name = "andi4"} : <i1>
    %82:2 = fork [2] %83 {handshake.bb = 2 : ui32, handshake.name = "fork43"} : <i9>
    %83 = addi %67, %72 {handshake.bb = 2 : ui32, handshake.name = "addi0"} : <i9>
    %84 = buffer %96#3, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer26"} : <i1>
    %85 = passer %86#1[%84] {handshake.bb = 2 : ui32, handshake.name = "passer45"} : <i8>, <i1>
    %86:2 = fork [2] %87 {handshake.bb = 2 : ui32, handshake.name = "fork44"} : <i8>
    %87 = trunci %82#0 {handshake.bb = 2 : ui32, handshake.name = "trunci0"} : <i9> to <i8>
    %88:3 = fork [3] %89 {handshake.bb = 2 : ui32, handshake.name = "fork45"} : <i1>
    %89 = cmpi ult, %82#1, %75 {handshake.bb = 2 : ui32, handshake.name = "cmpi0"} : <i9>
    %90 = buffer %96#4, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer27"} : <i1>
    %91 = passer %92[%90] {handshake.bb = 2 : ui32, handshake.name = "passer46"} : <i1>, <i1>
    %92 = andi %80#1, %88#0 {handshake.bb = 2 : ui32, handshake.name = "andi5"} : <i1>
    %93 = spec_v2_repeating_init %91 {handshake.bb = 2 : ui32, handshake.name = "spec_v2_repeating_init0", initToken = 1 : ui1, specv2_top_ri = true} : <i1>
    %94 = buffer %93, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer22"} : <i1>
    %95 = buffer %94, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer23"} : <i1>
    %96:12 = fork [12] %95 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork3"} : <i1>
    %97 = buffer %96#11, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer32"} : <i1>
    %98 = init %97 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "init6", initToken = 0 : ui1} : <i1>
    %99:6 = fork [6] %98 {handshake.bb = 2 : ui32, handshake.name = "fork4"} : <i1>
    %100 = buffer %96#10, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer31"} : <i1>
    %101 = andi %48, %100 {handshake.bb = 2 : ui32, handshake.name = "andi6"} : <i1>
    %102:2 = fork [2] %101 {handshake.bb = 2 : ui32, handshake.name = "fork47"} : <i1>
    %103 = buffer %96#9, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer30"} : <i1>
    %104 = andi %47, %103 {handshake.bb = 2 : ui32, handshake.name = "andi7"} : <i1>
    %105:8 = fork [8] %104 {handshake.bb = 2 : ui32, handshake.name = "fork48"} : <i1>
    %106 = buffer %96#8, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer29"} : <i1>
    %107 = andi %46, %106 {handshake.bb = 2 : ui32, handshake.name = "andi8"} : <i1>
    %108:2 = fork [2] %107 {handshake.bb = 2 : ui32, handshake.name = "fork49"} : <i1>
    %109 = buffer %96#7, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer28"} : <i1>
    %110 = andi %43#1, %109 {handshake.bb = 2 : ui32, handshake.name = "andi9"} : <i1>
    %111:2 = fork [2] %110 {handshake.bb = 2 : ui32, handshake.name = "fork50"} : <i1>
    %112 = not %78#1 {handshake.bb = 2 : ui32, handshake.name = "not2"} : <i1>
    %113 = passer %140#0[%105#0] {handshake.bb = 2 : ui32, handshake.name = "passer21"} : <f32>, <i1>
    %114 = buffer %35#2, bufferType = FIFO_BREAK_NONE, numSlots = 2 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer15"} : <f32>
    %115 = passer %114[%105#1] {handshake.bb = 2 : ui32, handshake.name = "passer25"} : <f32>, <i1>
    %116 = passer %68#1[%105#7] {handshake.bb = 2 : ui32, handshake.name = "passer27"} : <f32>, <i1>
    %117 = passer %86#0[%105#6] {handshake.bb = 2 : ui32, handshake.name = "passer29"} : <i8>, <i1>
    %118 = passer %88#1[%105#4] {handshake.bb = 2 : ui32, handshake.name = "passer31"} : <i1>, <i1>
    %119 = passer %145#2[%105#5] {handshake.bb = 2 : ui32, handshake.name = "passer33"} : <>, <i1>
    %120 = buffer %131#0, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer37"} : <f32>
    %121 = buffer %120, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer38"} : <f32>
    %122 = passer %121[%105#2] {handshake.bb = 2 : ui32, handshake.name = "passer34"} : <f32>, <i1>
    %123 = buffer %38#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer19"} : <f32>
    %124 = passer %123[%105#3] {handshake.bb = 2 : ui32, handshake.name = "passer35"} : <f32>, <i1>
    %125 = not %88#2 {handshake.bb = 2 : ui32, handshake.name = "not3"} : <i1>
    %126 = buffer %129#0, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer36"} : <f32>
    %127 = passer %126[%96#5] {handshake.bb = 2 : ui32, handshake.name = "passer47"} : <f32>, <i1>
    %128 = buffer %15, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer5"} : <f32>
    %129:3 = fork [3] %128 {handshake.bb = 2 : ui32, handshake.name = "fork51"} : <f32>
    %130 = buffer %17, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer6"} : <f32>
    %131:3 = fork [3] %130 {handshake.bb = 2 : ui32, handshake.name = "fork52"} : <f32>
    %132 = passer %135#0[%96#2] {handshake.bb = 2 : ui32, handshake.name = "passer48"} : <f32>, <i1>
    %133 = buffer %22, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer9"} : <f32>
    %134 = buffer %133, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer10"} : <f32>
    %135:2 = fork [2] %134 {handshake.bb = 2 : ui32, handshake.name = "fork53"} : <f32>
    %136 = buffer %96#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer25"} : <i1>
    %137 = passer %140#1[%136] {handshake.bb = 2 : ui32, handshake.name = "passer49"} : <f32>, <i1>
    %138 = buffer %25, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer11"} : <f32>
    %139 = buffer %138, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer12"} : <f32>
    %140:4 = fork [4] %139 {handshake.bb = 2 : ui32, handshake.name = "fork54"} : <f32>
    %141 = buffer %96#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer24"} : <i1>
    %142 = passer %145#4[%141] {handshake.bb = 2 : ui32, handshake.name = "passer50"} : <>, <i1>
    %143 = buffer %27, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer13"} : <>
    %144 = buffer %143, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer14"} : <>
    %145:6 = fork [6] %144 {handshake.bb = 2 : ui32, handshake.name = "fork55"} : <>
    %146 = passer %145#3[%108#1] {handshake.bb = 2 : ui32, handshake.name = "passer42"} : <>, <i1>
    %147 = passer %68#2[%108#0] {handshake.bb = 2 : ui32, handshake.name = "passer43"} : <f32>, <i1>
    %148:7 = fork [7] %118 {handshake.bb = 3 : ui32, handshake.name = "fork21"} : <i1>
    %trueResult, %falseResult = cond_br %148#6, %115 {handshake.bb = 3 : ui32, handshake.name = "cond_br38"} : <i1>, <f32>
    sink %falseResult {handshake.name = "sink26"} : <f32>
    %trueResult_0, %falseResult_1 = cond_br %148#5, %122 {handshake.bb = 3 : ui32, handshake.name = "cond_br39"} : <i1>, <f32>
    sink %falseResult_1 {handshake.name = "sink27"} : <f32>
    %trueResult_2, %falseResult_3 = cond_br %148#0, %117 {handshake.bb = 3 : ui32, handshake.name = "cond_br40"} : <i1>, <i8>
    sink %falseResult_3 {handshake.name = "sink28"} : <i8>
    %trueResult_4, %falseResult_5 = cond_br %148#4, %124 {handshake.bb = 3 : ui32, handshake.name = "cond_br41"} : <i1>, <f32>
    sink %falseResult_5 {handshake.name = "sink29"} : <f32>
    %trueResult_6, %falseResult_7 = cond_br %148#3, %113 {handshake.bb = 3 : ui32, handshake.name = "cond_br42"} : <i1>, <f32>
    sink %falseResult_7 {handshake.name = "sink30"} : <f32>
    %trueResult_8, %falseResult_9 = cond_br %148#2, %119 {handshake.bb = 3 : ui32, handshake.name = "cond_br43"} : <i1>, <>
    %trueResult_10, %falseResult_11 = cond_br %148#1, %116 {handshake.bb = 3 : ui32, handshake.name = "cond_br44"} : <i1>, <f32>
    sink %trueResult_10 {handshake.name = "sink31"} : <f32>
    %149 = mux %index_13 [%50, %147, %falseResult_11] {handshake.bb = 4 : ui32, handshake.name = "mux10"} : <i2>, [<f32>, <f32>, <f32>] to <f32>
    %result_12, %index_13 = control_merge [%51, %146, %falseResult_9]  {handshake.bb = 4 : ui32, handshake.name = "control_merge6"} : [<>, <>, <>] to <>, <i2>
    %150 = mux %index_15 [%63, %149] {handshake.bb = 5 : ui32, handshake.name = "mux11"} : <i1>, [<f32>, <f32>] to <f32>
    %result_14, %index_15 = control_merge [%64, %result_12]  {handshake.bb = 5 : ui32, handshake.name = "control_merge7"} : [<>, <>] to <>, <i1>
    sink %result_14 {handshake.name = "sink32"} : <>
    end {handshake.bb = 5 : ui32, handshake.name = "end0"} %150, %0#1 : <f32>, <>
  }
}

