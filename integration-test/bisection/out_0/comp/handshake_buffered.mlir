module {
  handshake.func @bisection(%arg0: !handshake.channel<f32>, %arg1: !handshake.channel<f32>, %arg2: !handshake.channel<f32>, %arg3: !handshake.control<>, ...) -> (!handshake.channel<f32>, !handshake.control<>) attributes {argNames = ["a", "b", "tol", "start"], handshake.cfdfcThroughput = #handshake<cfdfcThroughput {"0" = 0.033333333333333333 : f64, "1" = 0.033333333333333333 : f64}>, handshake.cfdfcToBBList = #handshake<cfdfcToBBList {"0" = [2 : ui32], "1" = [3 : ui32, 1 : ui32, 2 : ui32]}>, resNames = ["out0", "end"]} {
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
    %15 = mux %102#0 [%14, %117] {handshake.bb = 2 : ui32, handshake.name = "mux5"} : <i1>, [<f32>, <f32>] to <f32>
    %16 = buffer %15, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer5"} : <f32>
    %17:3 = fork [3] %16 {handshake.bb = 2 : ui32, handshake.name = "fork47"} : <f32>
    %18 = buffer %9, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer1"} : <f32>
    %19 = mux %102#1 [%18, %119] {handshake.bb = 2 : ui32, handshake.name = "mux6"} : <i1>, [<f32>, <f32>] to <f32>
    %20 = buffer %19, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer7"} : <f32>
    %21:3 = fork [3] %20 {handshake.bb = 2 : ui32, handshake.name = "fork48"} : <f32>
    %22 = buffer %10, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer2"} : <i8>
    %23 = mux %102#2 [%22, %120] {handshake.bb = 2 : ui32, handshake.name = "mux7"} : <i1>, [<i8>, <i8>] to <i8>
    %24 = buffer %11, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer3"} : <f32>
    %25 = mux %102#3 [%24, %122] {handshake.bb = 2 : ui32, handshake.name = "mux8"} : <i1>, [<f32>, <f32>] to <f32>
    %26 = buffer %12, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer4"} : <f32>
    %27 = mux %102#4 [%26, %123] {handshake.bb = 2 : ui32, handshake.name = "mux9"} : <i1>, [<f32>, <f32>] to <f32>
    %28 = buffer %27, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer11"} : <f32>
    %29 = buffer %28, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer12"} : <f32>
    %30:4 = fork [4] %29 {handshake.bb = 2 : ui32, handshake.name = "fork49"} : <f32>
    %31 = mux %102#5 [%result, %124] {handshake.bb = 2 : ui32, handshake.name = "mux15"} : <i1>, [<>, <>] to <>
    %32 = buffer %31, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer13"} : <>
    %33 = buffer %32, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer14"} : <>
    %34:6 = fork [6] %33 {handshake.bb = 2 : ui32, handshake.name = "fork50"} : <>
    %35 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %36 = constant %35 {handshake.bb = 2 : ui32, handshake.name = "constant6", value = -2.000000e+00 : f32} : <>, <f32>
    %37 = source {handshake.bb = 2 : ui32, handshake.name = "source2"} : <>
    %38 = constant %37 {handshake.bb = 2 : ui32, handshake.name = "constant7", value = 5.000000e-01 : f32} : <>, <f32>
    %39 = addf %17#0, %21#0 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "addf1", internal_delay = "2_922000"} : <f32>
    %40 = mulf %39, %38 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "mulf1", internal_delay = "2_875333"} : <f32>
    %41:6 = fork [6] %40 {handshake.bb = 2 : ui32, handshake.name = "fork51"} : <f32>
    %42 = mulf %41#0, %41#1 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "mulf2", internal_delay = "2_875333"} : <f32>
    %43 = addf %42, %36 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "addf2", internal_delay = "2_922000"} : <f32>
    %44:3 = fork [3] %43 {handshake.bb = 2 : ui32, handshake.name = "fork52"} : <f32>
    %45 = buffer %44#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer19"} : <f32>
    %46 = absf %45 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "absf0"} : <f32> to <f32>
    %47 = cmpf olt, %46, %30#0 {handshake.bb = 2 : ui32, handshake.name = "cmpf0", internal_delay = "0_000000"} : <f32>
    %48:3 = fork [3] %47 {handshake.bb = 2 : ui32, handshake.name = "fork25"} : <i1>
    %49 = not %48#2 {handshake.bb = 2 : ui32, handshake.name = "not0"} : <i1>
    %50:2 = fork [2] %49 {handshake.bb = 2 : ui32, handshake.name = "fork6"} : <i1>
    %51 = andi %103, %69#0 {handshake.bb = 2 : ui32, handshake.name = "andi0"} : <i1>
    %52:8 = fork [8] %51 {handshake.bb = 2 : ui32, handshake.name = "fork7"} : <i1>
    %53 = andi %115, %90#0 {handshake.bb = 2 : ui32, handshake.name = "andi1"} : <i1>
    %54:2 = fork [2] %53 {handshake.bb = 2 : ui32, handshake.name = "fork9"} : <i1>
    %55 = andi %66#1, %50#0 {handshake.bb = 2 : ui32, handshake.name = "andi2"} : <i1>
    %56:2 = fork [2] %55 {handshake.bb = 2 : ui32, handshake.name = "fork13"} : <i1>
    %57 = buffer %41#2, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer15"} : <f32>
    %58 = passer %57[%48#1] {handshake.bb = 2 : ui32, handshake.name = "passer0"} : <f32>, <i1>
    %59 = passer %34#0[%48#0] {handshake.bb = 2 : ui32, handshake.name = "passer2"} : <>, <i1>
    %60 = source {handshake.bb = 2 : ui32, handshake.name = "source3"} : <>
    %61 = constant %60 {handshake.bb = 2 : ui32, handshake.name = "constant8", value = 5.000000e-01 : f32} : <>, <f32>
    %62 = subf %21#2, %17#2 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "subf0", internal_delay = "2_000000"} : <f32>
    %63 = mulf %62, %61 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "mulf3", internal_delay = "2_875333"} : <f32>
    %64 = buffer %63, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer21"} : <f32>
    %65 = cmpf olt, %64, %30#1 {handshake.bb = 2 : ui32, handshake.name = "cmpf1", internal_delay = "0_000000"} : <f32>
    %66:2 = fork [2] %65 {handshake.bb = 2 : ui32, handshake.name = "fork16"} : <i1>
    %67 = not %66#0 {handshake.bb = 2 : ui32, handshake.name = "not1"} : <i1>
    %68 = andi %50#1, %67 {handshake.bb = 2 : ui32, handshake.name = "andi3"} : <i1>
    %69:2 = fork [2] %68 {handshake.bb = 2 : ui32, handshake.name = "fork17"} : <i1>
    %70 = buffer %41#3, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer16"} : <f32>
    %71 = passer %70[%56#1] {handshake.bb = 2 : ui32, handshake.name = "passer10"} : <f32>, <i1>
    %72 = passer %34#1[%56#0] {handshake.bb = 2 : ui32, handshake.name = "passer12"} : <>, <i1>
    %73 = buffer %23, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer9"} : <i8>
    %74 = extsi %73 {handshake.bb = 2 : ui32, handshake.name = "extsi4"} : <i8> to <i9>
    %75 = buffer %25, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer10"} : <f32>
    %76:2 = fork [2] %75 {handshake.bb = 2 : ui32, handshake.name = "fork14"} : <f32>
    %77:3 = fork [3] %78 {handshake.bb = 2 : ui32, handshake.name = "fork18"} : <f32>
    %78 = constant %34#2 {handshake.bb = 2 : ui32, handshake.name = "constant9", value = 0.000000e+00 : f32} : <>, <f32>
    %79 = source {handshake.bb = 2 : ui32, handshake.name = "source4"} : <>
    %80 = constant %79 {handshake.bb = 2 : ui32, handshake.name = "constant1", value = 1 : i2} : <>, <i2>
    %81 = extsi %80 {handshake.bb = 2 : ui32, handshake.name = "extsi5"} : <i2> to <i9>
    %82 = source {handshake.bb = 2 : ui32, handshake.name = "source5"} : <>
    %83 = constant %82 {handshake.bb = 2 : ui32, handshake.name = "constant2", value = 100 : i8} : <>, <i8>
    %84 = extsi %83 {handshake.bb = 2 : ui32, handshake.name = "extsi6"} : <i8> to <i9>
    %85 = buffer %76#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer23"} : <f32>
    %86 = mulf %85, %44#2 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "mulf4", internal_delay = "2_875333"} : <f32>
    %87 = cmpf olt, %86, %77#0 {handshake.bb = 2 : ui32, handshake.name = "cmpf2", internal_delay = "0_000000"} : <f32>
    %88:2 = fork [2] %87 {handshake.bb = 2 : ui32, handshake.name = "fork20"} : <i1>
    %89 = andi %69#1, %88#0 {handshake.bb = 2 : ui32, handshake.name = "andi4"} : <i1>
    %90:2 = fork [2] %89 {handshake.bb = 2 : ui32, handshake.name = "fork42"} : <i1>
    %91 = buffer %93, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer24"} : <i9>
    %92:2 = fork [2] %91 {handshake.bb = 2 : ui32, handshake.name = "fork43"} : <i9>
    %93 = addi %74, %81 {handshake.bb = 2 : ui32, handshake.name = "addi0"} : <i9>
    %94:2 = fork [2] %95 {handshake.bb = 2 : ui32, handshake.name = "fork44"} : <i8>
    %95 = trunci %92#0 {handshake.bb = 2 : ui32, handshake.name = "trunci0"} : <i9> to <i8>
    %96 = cmpi ult, %92#1, %84 {handshake.bb = 2 : ui32, handshake.name = "cmpi0"} : <i9>
    %97:3 = fork [3] %96 {handshake.bb = 2 : ui32, handshake.name = "fork45"} : <i1>
    %98 = andi %90#1, %97#0 {handshake.bb = 2 : ui32, handshake.name = "andi5"} : <i1>
    %99:7 = fork [7] %98 {handshake.bb = 2 : ui32, handshake.name = "fork3"} : <i1>
    %100 = buffer %99#6, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer25"} : <i1>
    %101 = init %100 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "init6", initToken = 0 : ui1} : <i1>
    %102:6 = fork [6] %101 {handshake.bb = 2 : ui32, handshake.name = "fork4"} : <i1>
    %103 = not %88#1 {handshake.bb = 2 : ui32, handshake.name = "not2"} : <i1>
    %104 = passer %30#2[%52#7] {handshake.bb = 2 : ui32, handshake.name = "passer21"} : <f32>, <i1>
    %105 = buffer %41#4, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer17"} : <f32>
    %106 = passer %105[%52#3] {handshake.bb = 2 : ui32, handshake.name = "passer25"} : <f32>, <i1>
    %107 = passer %77#1[%52#1] {handshake.bb = 2 : ui32, handshake.name = "passer27"} : <f32>, <i1>
    %108 = passer %94#0[%52#5] {handshake.bb = 2 : ui32, handshake.name = "passer29"} : <i8>, <i1>
    %109 = passer %97#1[%52#2] {handshake.bb = 2 : ui32, handshake.name = "passer31"} : <i1>, <i1>
    %110 = passer %34#3[%52#0] {handshake.bb = 2 : ui32, handshake.name = "passer33"} : <>, <i1>
    %111 = buffer %21#1, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer8"} : <f32>
    %112 = passer %111[%52#6] {handshake.bb = 2 : ui32, handshake.name = "passer34"} : <f32>, <i1>
    %113 = buffer %44#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer20"} : <f32>
    %114 = passer %113[%52#4] {handshake.bb = 2 : ui32, handshake.name = "passer35"} : <f32>, <i1>
    %115 = not %97#2 {handshake.bb = 2 : ui32, handshake.name = "not3"} : <i1>
    %116 = buffer %17#1, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer6"} : <f32>
    %117 = passer %116[%99#4] {handshake.bb = 2 : ui32, handshake.name = "passer36"} : <f32>, <i1>
    %118 = buffer %41#5, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer18"} : <f32>
    %119 = passer %118[%99#5] {handshake.bb = 2 : ui32, handshake.name = "passer37"} : <f32>, <i1>
    %120 = passer %94#1[%99#0] {handshake.bb = 2 : ui32, handshake.name = "passer38"} : <i8>, <i1>
    %121 = buffer %76#0, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer22"} : <f32>
    %122 = passer %121[%99#3] {handshake.bb = 2 : ui32, handshake.name = "passer39"} : <f32>, <i1>
    %123 = passer %30#3[%99#1] {handshake.bb = 2 : ui32, handshake.name = "passer40"} : <f32>, <i1>
    %124 = passer %34#5[%99#2] {handshake.bb = 2 : ui32, handshake.name = "passer41"} : <>, <i1>
    %125 = passer %34#4[%54#0] {handshake.bb = 2 : ui32, handshake.name = "passer42"} : <>, <i1>
    %126 = passer %77#2[%54#1] {handshake.bb = 2 : ui32, handshake.name = "passer43"} : <f32>, <i1>
    %127:7 = fork [7] %109 {handshake.bb = 3 : ui32, handshake.name = "fork21"} : <i1>
    %trueResult, %falseResult = cond_br %127#6, %106 {handshake.bb = 3 : ui32, handshake.name = "cond_br38"} : <i1>, <f32>
    sink %falseResult {handshake.name = "sink26"} : <f32>
    %trueResult_0, %falseResult_1 = cond_br %127#5, %112 {handshake.bb = 3 : ui32, handshake.name = "cond_br39"} : <i1>, <f32>
    sink %falseResult_1 {handshake.name = "sink27"} : <f32>
    %trueResult_2, %falseResult_3 = cond_br %127#0, %108 {handshake.bb = 3 : ui32, handshake.name = "cond_br40"} : <i1>, <i8>
    sink %falseResult_3 {handshake.name = "sink28"} : <i8>
    %trueResult_4, %falseResult_5 = cond_br %127#4, %114 {handshake.bb = 3 : ui32, handshake.name = "cond_br41"} : <i1>, <f32>
    sink %falseResult_5 {handshake.name = "sink29"} : <f32>
    %trueResult_6, %falseResult_7 = cond_br %127#3, %104 {handshake.bb = 3 : ui32, handshake.name = "cond_br42"} : <i1>, <f32>
    sink %falseResult_7 {handshake.name = "sink30"} : <f32>
    %trueResult_8, %falseResult_9 = cond_br %127#2, %110 {handshake.bb = 3 : ui32, handshake.name = "cond_br43"} : <i1>, <>
    %trueResult_10, %falseResult_11 = cond_br %127#1, %107 {handshake.bb = 3 : ui32, handshake.name = "cond_br44"} : <i1>, <f32>
    sink %trueResult_10 {handshake.name = "sink31"} : <f32>
    %128 = mux %index_13 [%58, %126, %falseResult_11] {handshake.bb = 4 : ui32, handshake.name = "mux10"} : <i2>, [<f32>, <f32>, <f32>] to <f32>
    %result_12, %index_13 = control_merge [%59, %125, %falseResult_9]  {handshake.bb = 4 : ui32, handshake.name = "control_merge6"} : [<>, <>, <>] to <>, <i2>
    %129 = mux %index_15 [%71, %128] {handshake.bb = 5 : ui32, handshake.name = "mux11"} : <i1>, [<f32>, <f32>] to <f32>
    %result_14, %index_15 = control_merge [%72, %result_12]  {handshake.bb = 5 : ui32, handshake.name = "control_merge7"} : [<>, <>] to <>, <i1>
    sink %result_14 {handshake.name = "sink32"} : <>
    end {handshake.bb = 5 : ui32, handshake.name = "end0"} %129, %0#1 : <f32>, <>
  }
}

