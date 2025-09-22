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
    %15 = mux %102#0 [%14, %132] {handshake.bb = 2 : ui32, handshake.name = "mux5"} : <i1>, [<f32>, <f32>] to <f32>
    %16 = buffer %9, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer1"} : <f32>
    %17 = mux %102#1 [%16, %33] {handshake.bb = 2 : ui32, handshake.name = "mux6"} : <i1>, [<f32>, <f32>] to <f32>
    %18 = buffer %10, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer2"} : <i8>
    %19 = mux %102#2 [%18, %84] {handshake.bb = 2 : ui32, handshake.name = "mux7"} : <i1>, [<i8>, <i8>] to <i8>
    %20 = buffer %11, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer3"} : <f32>
    %21 = mux %102#3 [%20, %135] {handshake.bb = 2 : ui32, handshake.name = "mux8"} : <i1>, [<f32>, <f32>] to <f32>
    %22 = buffer %12, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer4"} : <f32>
    %23 = mux %102#4 [%22, %139] {handshake.bb = 2 : ui32, handshake.name = "mux9"} : <i1>, [<f32>, <f32>] to <f32>
    %24 = buffer %102#5, bufferType = FIFO_BREAK_NONE, numSlots = 2 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer33"} : <i1>
    %25 = mux %24 [%result, %145] {handshake.bb = 2 : ui32, handshake.name = "mux15"} : <i1>, [<>, <>] to <>
    %26 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %27 = constant %26 {handshake.bb = 2 : ui32, handshake.name = "constant6", value = -2.000000e+00 : f32} : <>, <f32>
    %28 = source {handshake.bb = 2 : ui32, handshake.name = "source2"} : <>
    %29 = constant %28 {handshake.bb = 2 : ui32, handshake.name = "constant7", value = 5.000000e-01 : f32} : <>, <f32>
    %30 = addf %134#2, %32#2 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "addf1", internal_delay = "2_922000"} : <f32>
    %31 = buffer %17, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer6"} : <f32>
    %32:3 = fork [3] %31 {handshake.bb = 2 : ui32, handshake.name = "fork6"} : <f32>
    %33 = passer %34#3[%99#2] {handshake.bb = 2 : ui32, handshake.name = "passer44"} : <f32>, <i1>
    %34:6 = fork [6] %35 {handshake.bb = 2 : ui32, handshake.name = "fork7"} : <f32>
    %35 = mulf %30, %29 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "mulf1", internal_delay = "2_875333"} : <f32>
    %36 = mulf %34#0, %34#1 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "mulf2", internal_delay = "2_875333"} : <f32>
    %37:3 = fork [3] %38 {handshake.bb = 2 : ui32, handshake.name = "fork9"} : <f32>
    %38 = addf %36, %27 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "addf2", internal_delay = "2_922000"} : <f32>
    %39 = buffer %37#2, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer20"} : <f32>
    %40 = absf %39 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "absf0"} : <f32> to <f32>
    %41 = buffer %142#3, bufferType = FIFO_BREAK_NONE, numSlots = 2 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer38"} : <f32>
    %42 = cmpf olt, %40, %41 {handshake.bb = 2 : ui32, handshake.name = "cmpf0", internal_delay = "0_000000"} : <f32>
    %43:2 = fork [2] %42 {handshake.bb = 2 : ui32, handshake.name = "fork13"} : <i1>
    %44:2 = fork [2] %45 {handshake.bb = 2 : ui32, handshake.name = "fork16"} : <i1>
    %45 = not %43#0 {handshake.bb = 2 : ui32, handshake.name = "not0"} : <i1>
    %46 = andi %57#1, %44#0 {handshake.bb = 2 : ui32, handshake.name = "andi0"} : <i1>
    %47 = andi %130, %79#0 {handshake.bb = 2 : ui32, handshake.name = "andi1"} : <i1>
    %48 = andi %115, %60#0 {handshake.bb = 2 : ui32, handshake.name = "andi2"} : <i1>
    %49 = buffer %34#5, bufferType = FIFO_BREAK_NONE, numSlots = 2 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer18"} : <f32>
    %50 = passer %49[%114#1] {handshake.bb = 2 : ui32, handshake.name = "passer0"} : <f32>, <i1>
    %51 = passer %148#0[%114#0] {handshake.bb = 2 : ui32, handshake.name = "passer2"} : <>, <i1>
    %52 = source {handshake.bb = 2 : ui32, handshake.name = "source3"} : <>
    %53 = constant %52 {handshake.bb = 2 : ui32, handshake.name = "constant8", value = 5.000000e-01 : f32} : <>, <f32>
    %54 = subf %32#1, %134#1 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "subf0", internal_delay = "2_000000"} : <f32>
    %55 = mulf %54, %53 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "mulf3", internal_delay = "2_875333"} : <f32>
    %56 = buffer %58, bufferType = FIFO_BREAK_NONE, numSlots = 2 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer21"} : <i1>
    %57:2 = fork [2] %56 {handshake.bb = 2 : ui32, handshake.name = "fork17"} : <i1>
    %58 = cmpf olt, %55, %142#2 {handshake.bb = 2 : ui32, handshake.name = "cmpf1", internal_delay = "0_000000"} : <f32>
    %59 = not %57#0 {handshake.bb = 2 : ui32, handshake.name = "not1"} : <i1>
    %60:2 = fork [2] %61 {handshake.bb = 2 : ui32, handshake.name = "fork18"} : <i1>
    %61 = andi %44#1, %59 {handshake.bb = 2 : ui32, handshake.name = "andi3"} : <i1>
    %62 = buffer %34#4, bufferType = FIFO_BREAK_NONE, numSlots = 2 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer17"} : <f32>
    %63 = passer %62[%112#0] {handshake.bb = 2 : ui32, handshake.name = "passer10"} : <f32>, <i1>
    %64 = passer %148#5[%112#1] {handshake.bb = 2 : ui32, handshake.name = "passer12"} : <>, <i1>
    %65 = buffer %19, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer7"} : <i8>
    %66 = extsi %65 {handshake.bb = 2 : ui32, handshake.name = "extsi4"} : <i8> to <i9>
    %67:3 = fork [3] %68 {handshake.bb = 2 : ui32, handshake.name = "fork20"} : <f32>
    %68 = constant %148#1 {handshake.bb = 2 : ui32, handshake.name = "constant9", value = 0.000000e+00 : f32} : <>, <f32>
    %69 = source {handshake.bb = 2 : ui32, handshake.name = "source4"} : <>
    %70 = constant %69 {handshake.bb = 2 : ui32, handshake.name = "constant1", value = 1 : i2} : <>, <i2>
    %71 = extsi %70 {handshake.bb = 2 : ui32, handshake.name = "extsi5"} : <i2> to <i9>
    %72 = source {handshake.bb = 2 : ui32, handshake.name = "source5"} : <>
    %73 = constant %72 {handshake.bb = 2 : ui32, handshake.name = "constant2", value = 100 : i8} : <>, <i8>
    %74 = extsi %73 {handshake.bb = 2 : ui32, handshake.name = "extsi6"} : <i8> to <i9>
    %75 = buffer %138#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer36"} : <f32>
    %76 = mulf %75, %37#1 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "mulf4", internal_delay = "2_875333"} : <f32>
    %77:2 = fork [2] %78 {handshake.bb = 2 : ui32, handshake.name = "fork42"} : <i1>
    %78 = cmpf olt, %76, %67#0 {handshake.bb = 2 : ui32, handshake.name = "cmpf2", internal_delay = "0_000000"} : <f32>
    %79:2 = fork [2] %80 {handshake.bb = 2 : ui32, handshake.name = "fork43"} : <i1>
    %80 = andi %60#1, %77#0 {handshake.bb = 2 : ui32, handshake.name = "andi4"} : <i1>
    %81:2 = fork [2] %83 {handshake.bb = 2 : ui32, handshake.name = "fork44"} : <i9>
    %82 = buffer %66, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer22"} : <i9>
    %83 = addi %82, %71 {handshake.bb = 2 : ui32, handshake.name = "addi0"} : <i9>
    %84 = passer %85#1[%99#4] {handshake.bb = 2 : ui32, handshake.name = "passer45"} : <i8>, <i1>
    %85:2 = fork [2] %86 {handshake.bb = 2 : ui32, handshake.name = "fork45"} : <i8>
    %86 = trunci %81#0 {handshake.bb = 2 : ui32, handshake.name = "trunci0"} : <i9> to <i8>
    %87 = buffer %90, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer24"} : <i1>
    %88 = buffer %87, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer25"} : <i1>
    %89:3 = fork [3] %88 {handshake.bb = 2 : ui32, handshake.name = "fork46"} : <i1>
    %90 = cmpi ult, %81#1, %74 {handshake.bb = 2 : ui32, handshake.name = "cmpi0"} : <i9>
    %91 = passer %92[%106#4] {handshake.bb = 2 : ui32, handshake.name = "passer46"} : <i1>, <i1>
    %92 = andi %79#1, %89#0 {handshake.bb = 2 : ui32, handshake.name = "andi5"} : <i1>
    %93 = spec_v2_repeating_init %91 {handshake.bb = 2 : ui32, handshake.name = "spec_v2_repeating_init0", initToken = 1 : ui1, specv2_top_ri = true} : <i1>
    %94 = buffer %93, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer26"} : <i1>
    %95 = buffer %94, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer27"} : <i1>
    %96:2 = fork [2] %95 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork47"} : <i1>
    %97 = spec_v2_repeating_init %96#0 {handshake.bb = 2 : ui32, handshake.name = "spec_v2_repeating_init1", initToken = 1 : ui1} : <i1>
    %98 = buffer %97, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer28"} : <i1>
    %99:8 = fork [8] %98 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork3"} : <i1>
    %100 = buffer %99#7, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer32"} : <i1>
    %101 = init %100 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "init6", initToken = 0 : ui1} : <i1>
    %102:6 = fork [6] %101 {handshake.bb = 2 : ui32, handshake.name = "fork4"} : <i1>
    %103 = buffer %99#6, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer31"} : <i1>
    %104 = andi %96#1, %103 {handshake.bb = 2 : ui32, handshake.name = "andi6", specv2_tmp_and = true} : <i1>
    %105 = buffer %104, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer34"} : <i1>
    %106:5 = fork [5] %105 {handshake.bb = 2 : ui32, handshake.name = "fork49"} : <i1>
    %107 = andi %48, %106#0 {handshake.bb = 2 : ui32, handshake.name = "andi7"} : <i1>
    %108:8 = fork [8] %107 {handshake.bb = 2 : ui32, handshake.name = "fork50"} : <i1>
    %109 = andi %47, %106#1 {handshake.bb = 2 : ui32, handshake.name = "andi8"} : <i1>
    %110:2 = fork [2] %109 {handshake.bb = 2 : ui32, handshake.name = "fork51"} : <i1>
    %111 = andi %46, %106#2 {handshake.bb = 2 : ui32, handshake.name = "andi9"} : <i1>
    %112:2 = fork [2] %111 {handshake.bb = 2 : ui32, handshake.name = "fork52"} : <i1>
    %113 = andi %43#1, %106#3 {handshake.bb = 2 : ui32, handshake.name = "andi10"} : <i1>
    %114:2 = fork [2] %113 {handshake.bb = 2 : ui32, handshake.name = "fork53"} : <i1>
    %115 = not %77#1 {handshake.bb = 2 : ui32, handshake.name = "not2"} : <i1>
    %116 = buffer %142#0, bufferType = FIFO_BREAK_NONE, numSlots = 2 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer37"} : <f32>
    %117 = passer %116[%108#5] {handshake.bb = 2 : ui32, handshake.name = "passer21"} : <f32>, <i1>
    %118 = buffer %34#2, bufferType = FIFO_BREAK_NONE, numSlots = 2 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer16"} : <f32>
    %119 = passer %118[%108#6] {handshake.bb = 2 : ui32, handshake.name = "passer25"} : <f32>, <i1>
    %120 = passer %67#1[%108#4] {handshake.bb = 2 : ui32, handshake.name = "passer27"} : <f32>, <i1>
    %121 = buffer %85#0, bufferType = FIFO_BREAK_NONE, numSlots = 2 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer23"} : <i8>
    %122 = passer %121[%108#3] {handshake.bb = 2 : ui32, handshake.name = "passer29"} : <i8>, <i1>
    %123 = passer %89#1[%108#1] {handshake.bb = 2 : ui32, handshake.name = "passer31"} : <i1>, <i1>
    %124 = passer %148#2[%108#2] {handshake.bb = 2 : ui32, handshake.name = "passer33"} : <>, <i1>
    %125 = buffer %32#0, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer14"} : <f32>
    %126 = buffer %125, bufferType = FIFO_BREAK_NONE, numSlots = 2 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer15"} : <f32>
    %127 = passer %126[%108#7] {handshake.bb = 2 : ui32, handshake.name = "passer34"} : <f32>, <i1>
    %128 = buffer %37#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer19"} : <f32>
    %129 = passer %128[%108#0] {handshake.bb = 2 : ui32, handshake.name = "passer35"} : <f32>, <i1>
    %130 = not %89#2 {handshake.bb = 2 : ui32, handshake.name = "not3"} : <i1>
    %131 = buffer %134#0, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer35"} : <f32>
    %132 = passer %131[%99#0] {handshake.bb = 2 : ui32, handshake.name = "passer47"} : <f32>, <i1>
    %133 = buffer %15, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer5"} : <f32>
    %134:3 = fork [3] %133 {handshake.bb = 2 : ui32, handshake.name = "fork54"} : <f32>
    %135 = passer %138#0[%99#3] {handshake.bb = 2 : ui32, handshake.name = "passer48"} : <f32>, <i1>
    %136 = buffer %21, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer8"} : <f32>
    %137 = buffer %136, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer9"} : <f32>
    %138:2 = fork [2] %137 {handshake.bb = 2 : ui32, handshake.name = "fork55"} : <f32>
    %139 = passer %142#1[%99#1] {handshake.bb = 2 : ui32, handshake.name = "passer49"} : <f32>, <i1>
    %140 = buffer %23, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer10"} : <f32>
    %141 = buffer %140, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer11"} : <f32>
    %142:4 = fork [4] %141 {handshake.bb = 2 : ui32, handshake.name = "fork56"} : <f32>
    %143 = buffer %99#5, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer29"} : <i1>
    %144 = buffer %143, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer30"} : <i1>
    %145 = passer %148#4[%144] {handshake.bb = 2 : ui32, handshake.name = "passer50"} : <>, <i1>
    %146 = buffer %25, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer12"} : <>
    %147 = buffer %146, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer13"} : <>
    %148:6 = fork [6] %147 {handshake.bb = 2 : ui32, handshake.name = "fork57"} : <>
    %149 = passer %148#3[%110#1] {handshake.bb = 2 : ui32, handshake.name = "passer42"} : <>, <i1>
    %150 = passer %67#2[%110#0] {handshake.bb = 2 : ui32, handshake.name = "passer43"} : <f32>, <i1>
    %151:7 = fork [7] %123 {handshake.bb = 3 : ui32, handshake.name = "fork21"} : <i1>
    %trueResult, %falseResult = cond_br %151#6, %119 {handshake.bb = 3 : ui32, handshake.name = "cond_br38"} : <i1>, <f32>
    sink %falseResult {handshake.name = "sink26"} : <f32>
    %trueResult_0, %falseResult_1 = cond_br %151#5, %127 {handshake.bb = 3 : ui32, handshake.name = "cond_br39"} : <i1>, <f32>
    sink %falseResult_1 {handshake.name = "sink27"} : <f32>
    %trueResult_2, %falseResult_3 = cond_br %151#0, %122 {handshake.bb = 3 : ui32, handshake.name = "cond_br40"} : <i1>, <i8>
    sink %falseResult_3 {handshake.name = "sink28"} : <i8>
    %trueResult_4, %falseResult_5 = cond_br %151#4, %129 {handshake.bb = 3 : ui32, handshake.name = "cond_br41"} : <i1>, <f32>
    sink %falseResult_5 {handshake.name = "sink29"} : <f32>
    %trueResult_6, %falseResult_7 = cond_br %151#3, %117 {handshake.bb = 3 : ui32, handshake.name = "cond_br42"} : <i1>, <f32>
    sink %falseResult_7 {handshake.name = "sink30"} : <f32>
    %trueResult_8, %falseResult_9 = cond_br %151#2, %124 {handshake.bb = 3 : ui32, handshake.name = "cond_br43"} : <i1>, <>
    %trueResult_10, %falseResult_11 = cond_br %151#1, %120 {handshake.bb = 3 : ui32, handshake.name = "cond_br44"} : <i1>, <f32>
    sink %trueResult_10 {handshake.name = "sink31"} : <f32>
    %152 = mux %index_13 [%50, %150, %falseResult_11] {handshake.bb = 4 : ui32, handshake.name = "mux10"} : <i2>, [<f32>, <f32>, <f32>] to <f32>
    %result_12, %index_13 = control_merge [%51, %149, %falseResult_9]  {handshake.bb = 4 : ui32, handshake.name = "control_merge6"} : [<>, <>, <>] to <>, <i2>
    %153 = mux %index_15 [%63, %152] {handshake.bb = 5 : ui32, handshake.name = "mux11"} : <i1>, [<f32>, <f32>] to <f32>
    %result_14, %index_15 = control_merge [%64, %result_12]  {handshake.bb = 5 : ui32, handshake.name = "control_merge7"} : [<>, <>] to <>, <i1>
    sink %result_14 {handshake.name = "sink32"} : <>
    end {handshake.bb = 5 : ui32, handshake.name = "end0"} %153, %0#1 : <f32>, <>
  }
}

