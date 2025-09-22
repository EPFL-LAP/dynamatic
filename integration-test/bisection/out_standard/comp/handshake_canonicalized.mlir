module {
  handshake.func @bisection(%arg0: !handshake.channel<f32>, %arg1: !handshake.channel<f32>, %arg2: !handshake.channel<f32>, %arg3: !handshake.control<>, ...) -> (!handshake.channel<f32>, !handshake.control<>) attributes {argNames = ["a", "b", "tol", "start"], handshake.cfdfcThroughput = #handshake<cfdfcThroughput {"0" = 0.023255813953484031 : f64}>, handshake.cfdfcToBBList = #handshake<cfdfcToBBList {"0" = [3 : ui32, 1 : ui32, 2 : ui32]}>, resNames = ["out0", "end"]} {
    %0:3 = fork [3] %arg3 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork0"} : <>
    %1:3 = fork [3] %arg0 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork1"} : <f32>
    %2 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant0", value = false} : <>, <i1>
    %3 = source {handshake.bb = 0 : ui32, handshake.name = "source0"} : <>
    %4 = constant %3 {handshake.bb = 0 : ui32, handshake.name = "constant5", value = -2.000000e+00 : f32} : <>, <f32>
    %5 = mulf %1#1, %1#2 {fastmath = #arith.fastmath<none>, handshake.bb = 0 : ui32, handshake.name = "mulf0", internal_delay = "2_875333"} : <f32>
    %6 = addf %5, %4 {fastmath = #arith.fastmath<none>, handshake.bb = 0 : ui32, handshake.name = "addf0", internal_delay = "2_922000"} : <f32>
    %7 = extsi %2 {handshake.bb = 0 : ui32, handshake.name = "extsi3"} : <i1> to <i8>
    %8 = mux %20#1 [%1#0, %trueResult_30] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<f32>, <f32>] to <f32>
    %9 = buffer %8, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer0"} : <f32>
    %10:2 = fork [2] %9 {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <f32>
    %11 = mux %20#2 [%arg1, %trueResult_32] {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mux1"} : <i1>, [<f32>, <f32>] to <f32>
    %12 = buffer %11, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer2"} : <f32>
    %13:2 = fork [2] %12 {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <f32>
    %14 = mux %20#0 [%7, %trueResult_34] {handshake.bb = 1 : ui32, handshake.name = "mux2"} : <i1>, [<i8>, <i8>] to <i8>
    %15 = mux %20#3 [%6, %trueResult_36] {handshake.bb = 1 : ui32, handshake.name = "mux3"} : <i1>, [<f32>, <f32>] to <f32>
    %16 = mux %20#4 [%arg2, %trueResult_38] {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mux4"} : <i1>, [<f32>, <f32>] to <f32>
    %17 = buffer %16, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer8"} : <f32>
    %18 = buffer %17, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer9"} : <f32>
    %19:2 = fork [2] %18 {handshake.bb = 1 : ui32, handshake.name = "fork4"} : <f32>
    %result, %index = control_merge [%0#2, %trueResult_40]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %20:5 = fork [5] %index {handshake.bb = 1 : ui32, handshake.name = "fork5"} : <i1>
    %21 = source {handshake.bb = 1 : ui32, handshake.name = "source1"} : <>
    %22 = constant %21 {handshake.bb = 1 : ui32, handshake.name = "constant6", value = -2.000000e+00 : f32} : <>, <f32>
    %23 = source {handshake.bb = 1 : ui32, handshake.name = "source2"} : <>
    %24 = constant %23 {handshake.bb = 1 : ui32, handshake.name = "constant7", value = 5.000000e-01 : f32} : <>, <f32>
    %25 = addf %10#1, %13#1 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "addf1", internal_delay = "2_922000"} : <f32>
    %26 = mulf %25, %24 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "mulf1", internal_delay = "2_875333"} : <f32>
    %27:3 = fork [3] %26 {handshake.bb = 1 : ui32, handshake.name = "fork6"} : <f32>
    %28 = mulf %27#1, %27#2 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "mulf2", internal_delay = "2_875333"} : <f32>
    %29 = addf %28, %22 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "addf2", internal_delay = "2_922000"} : <f32>
    %30:2 = fork [2] %29 {handshake.bb = 1 : ui32, handshake.name = "fork7"} : <f32>
    %31 = absf %30#1 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "absf0"} : <f32> to <f32>
    %32 = cmpf olt, %31, %19#1 {handshake.bb = 1 : ui32, handshake.name = "cmpf0", internal_delay = "0_000000"} : <f32>
    %33:8 = fork [8] %32 {handshake.bb = 1 : ui32, handshake.name = "fork8"} : <i1>
    %34 = buffer %27#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer11"} : <f32>
    %35 = buffer %33#7, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer14"} : <i1>
    %trueResult, %falseResult = cond_br %35, %34 {handshake.bb = 1 : ui32, handshake.name = "cond_br3"} : <i1>, <f32>
    %36 = buffer %result, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer10"} : <>
    %37 = buffer %33#6, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer13"} : <i1>
    %trueResult_0, %falseResult_1 = cond_br %37, %36 {handshake.bb = 1 : ui32, handshake.name = "cond_br4"} : <i1>, <>
    %trueResult_2, %falseResult_3 = cond_br %33#5, %19#0 {handshake.bb = 1 : ui32, handshake.name = "cond_br5"} : <i1>, <f32>
    sink %trueResult_2 {handshake.name = "sink0"} : <f32>
    %38 = buffer %10#0, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer1"} : <f32>
    %trueResult_4, %falseResult_5 = cond_br %33#4, %38 {handshake.bb = 1 : ui32, handshake.name = "cond_br6"} : <i1>, <f32>
    sink %trueResult_4 {handshake.name = "sink1"} : <f32>
    %39 = buffer %13#0, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer3"} : <f32>
    %trueResult_6, %falseResult_7 = cond_br %33#3, %39 {handshake.bb = 1 : ui32, handshake.name = "cond_br7"} : <i1>, <f32>
    sink %trueResult_6 {handshake.name = "sink2"} : <f32>
    %40 = buffer %14, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer4"} : <i8>
    %41 = buffer %40, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer5"} : <i8>
    %42 = buffer %33#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer12"} : <i1>
    %trueResult_8, %falseResult_9 = cond_br %42, %41 {handshake.bb = 1 : ui32, handshake.name = "cond_br8"} : <i1>, <i8>
    sink %trueResult_8 {handshake.name = "sink3"} : <i8>
    %43 = buffer %15, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer6"} : <f32>
    %44 = buffer %43, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer7"} : <f32>
    %trueResult_10, %falseResult_11 = cond_br %33#2, %44 {handshake.bb = 1 : ui32, handshake.name = "cond_br9"} : <i1>, <f32>
    sink %trueResult_10 {handshake.name = "sink4"} : <f32>
    %trueResult_12, %falseResult_13 = cond_br %33#1, %30#0 {handshake.bb = 1 : ui32, handshake.name = "cond_br10"} : <i1>, <f32>
    sink %trueResult_12 {handshake.name = "sink5"} : <f32>
    %45 = buffer %falseResult_3, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer15"} : <f32>
    %46:2 = fork [2] %45 {handshake.bb = 2 : ui32, handshake.name = "fork9"} : <f32>
    %47:2 = fork [2] %falseResult_5 {handshake.bb = 2 : ui32, handshake.name = "fork10"} : <f32>
    %48:2 = fork [2] %falseResult_7 {handshake.bb = 2 : ui32, handshake.name = "fork11"} : <f32>
    %49 = source {handshake.bb = 2 : ui32, handshake.name = "source3"} : <>
    %50 = constant %49 {handshake.bb = 2 : ui32, handshake.name = "constant8", value = 5.000000e-01 : f32} : <>, <f32>
    %51 = subf %48#1, %47#1 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "subf0", internal_delay = "2_000000"} : <f32>
    %52 = mulf %51, %50 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "mulf3", internal_delay = "2_875333"} : <f32>
    %53 = cmpf olt, %52, %46#1 {handshake.bb = 2 : ui32, handshake.name = "cmpf1", internal_delay = "0_000000"} : <f32>
    %54:8 = fork [8] %53 {handshake.bb = 2 : ui32, handshake.name = "fork12"} : <i1>
    %trueResult_14, %falseResult_15 = cond_br %54#7, %falseResult {handshake.bb = 2 : ui32, handshake.name = "cond_br11"} : <i1>, <f32>
    %55 = buffer %54#6, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer20"} : <i1>
    %trueResult_16, %falseResult_17 = cond_br %55, %falseResult_1 {handshake.bb = 2 : ui32, handshake.name = "cond_br12"} : <i1>, <>
    %trueResult_18, %falseResult_19 = cond_br %54#5, %46#0 {handshake.bb = 2 : ui32, handshake.name = "cond_br13"} : <i1>, <f32>
    sink %trueResult_18 {handshake.name = "sink7"} : <f32>
    %56 = buffer %47#0, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer18"} : <f32>
    %trueResult_20, %falseResult_21 = cond_br %54#4, %56 {handshake.bb = 2 : ui32, handshake.name = "cond_br14"} : <i1>, <f32>
    sink %trueResult_20 {handshake.name = "sink8"} : <f32>
    %57 = buffer %48#0, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer19"} : <f32>
    %trueResult_22, %falseResult_23 = cond_br %54#3, %57 {handshake.bb = 2 : ui32, handshake.name = "cond_br15"} : <i1>, <f32>
    sink %trueResult_22 {handshake.name = "sink9"} : <f32>
    %trueResult_24, %falseResult_25 = cond_br %54#0, %falseResult_9 {handshake.bb = 2 : ui32, handshake.name = "cond_br16"} : <i1>, <i8>
    sink %trueResult_24 {handshake.name = "sink10"} : <i8>
    %58 = buffer %falseResult_11, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer16"} : <f32>
    %trueResult_26, %falseResult_27 = cond_br %54#2, %58 {handshake.bb = 2 : ui32, handshake.name = "cond_br17"} : <i1>, <f32>
    sink %trueResult_26 {handshake.name = "sink11"} : <f32>
    %59 = buffer %falseResult_13, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer17"} : <f32>
    %trueResult_28, %falseResult_29 = cond_br %54#1, %59 {handshake.bb = 2 : ui32, handshake.name = "cond_br18"} : <i1>, <f32>
    sink %trueResult_28 {handshake.name = "sink12"} : <f32>
    %60 = extsi %falseResult_25 {handshake.bb = 3 : ui32, handshake.name = "extsi4"} : <i8> to <i9>
    %61:2 = fork [2] %falseResult_27 {handshake.bb = 3 : ui32, handshake.name = "fork13"} : <f32>
    %62 = buffer %falseResult_15, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 3 : ui32, handshake.name = "buffer21"} : <f32>
    %63:2 = fork [2] %62 {handshake.bb = 3 : ui32, handshake.name = "fork14"} : <f32>
    %64:2 = fork [2] %falseResult_29 {handshake.bb = 3 : ui32, handshake.name = "fork15"} : <f32>
    %65:2 = fork [2] %falseResult_17 {handshake.bb = 3 : ui32, handshake.name = "fork16"} : <>
    %66 = constant %65#1 {handshake.bb = 3 : ui32, handshake.name = "constant9", value = 0.000000e+00 : f32} : <>, <f32>
    %67:2 = fork [2] %66 {handshake.bb = 3 : ui32, handshake.name = "fork17"} : <f32>
    %68 = source {handshake.bb = 3 : ui32, handshake.name = "source4"} : <>
    %69 = constant %68 {handshake.bb = 3 : ui32, handshake.name = "constant1", value = 1 : i2} : <>, <i2>
    %70 = extsi %69 {handshake.bb = 3 : ui32, handshake.name = "extsi5"} : <i2> to <i9>
    %71 = source {handshake.bb = 3 : ui32, handshake.name = "source5"} : <>
    %72 = constant %71 {handshake.bb = 3 : ui32, handshake.name = "constant2", value = 100 : i8} : <>, <i8>
    %73 = extsi %72 {handshake.bb = 3 : ui32, handshake.name = "extsi6"} : <i8> to <i9>
    %74 = mulf %61#1, %64#1 {fastmath = #arith.fastmath<none>, handshake.bb = 3 : ui32, handshake.name = "mulf4", internal_delay = "2_875333"} : <f32>
    %75 = cmpf olt, %74, %67#1 {handshake.bb = 3 : ui32, handshake.name = "cmpf2", internal_delay = "0_000000"} : <f32>
    %76:3 = fork [3] %75 {handshake.bb = 3 : ui32, handshake.name = "fork18"} : <i1>
    %77 = buffer %falseResult_23, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 3 : ui32, handshake.name = "buffer24"} : <f32>
    %78 = select %76#2[%63#1, %77] {handshake.bb = 3 : ui32, handshake.name = "select0"} : <i1>, <f32>
    %79 = buffer %falseResult_21, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 3 : ui32, handshake.name = "buffer23"} : <f32>
    %80 = select %76#1[%79, %63#0] {handshake.bb = 3 : ui32, handshake.name = "select1"} : <i1>, <f32>
    %81 = buffer %61#0, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 3 : ui32, handshake.name = "buffer25"} : <f32>
    %82 = buffer %64#0, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 3 : ui32, handshake.name = "buffer26"} : <f32>
    %83 = select %76#0[%81, %82] {handshake.bb = 3 : ui32, handshake.name = "select2"} : <i1>, <f32>
    %84 = addi %60, %70 {handshake.bb = 3 : ui32, handshake.name = "addi0"} : <i9>
    %85 = buffer %84, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 3 : ui32, handshake.name = "buffer27"} : <i9>
    %86:2 = fork [2] %85 {handshake.bb = 3 : ui32, handshake.name = "fork19"} : <i9>
    %87 = trunci %86#0 {handshake.bb = 3 : ui32, handshake.name = "trunci0"} : <i9> to <i8>
    %88 = cmpi ult, %86#1, %73 {handshake.bb = 3 : ui32, handshake.name = "cmpi0"} : <i9>
    %89:7 = fork [7] %88 {handshake.bb = 3 : ui32, handshake.name = "fork20"} : <i1>
    %trueResult_30, %falseResult_31 = cond_br %89#1, %80 {handshake.bb = 3 : ui32, handshake.name = "cond_br19"} : <i1>, <f32>
    sink %falseResult_31 {handshake.name = "sink14"} : <f32>
    %trueResult_32, %falseResult_33 = cond_br %89#2, %78 {handshake.bb = 3 : ui32, handshake.name = "cond_br20"} : <i1>, <f32>
    sink %falseResult_33 {handshake.name = "sink15"} : <f32>
    %trueResult_34, %falseResult_35 = cond_br %89#0, %87 {handshake.bb = 3 : ui32, handshake.name = "cond_br21"} : <i1>, <i8>
    sink %falseResult_35 {handshake.name = "sink16"} : <i8>
    %trueResult_36, %falseResult_37 = cond_br %89#3, %83 {handshake.bb = 3 : ui32, handshake.name = "cond_br22"} : <i1>, <f32>
    sink %falseResult_37 {handshake.name = "sink17"} : <f32>
    %90 = buffer %falseResult_19, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 3 : ui32, handshake.name = "buffer22"} : <f32>
    %trueResult_38, %falseResult_39 = cond_br %89#4, %90 {handshake.bb = 3 : ui32, handshake.name = "cond_br23"} : <i1>, <f32>
    sink %falseResult_39 {handshake.name = "sink18"} : <f32>
    %trueResult_40, %falseResult_41 = cond_br %89#5, %65#0 {handshake.bb = 3 : ui32, handshake.name = "cond_br24"} : <i1>, <>
    %trueResult_42, %falseResult_43 = cond_br %89#6, %67#0 {handshake.bb = 3 : ui32, handshake.name = "cond_br25"} : <i1>, <f32>
    sink %trueResult_42 {handshake.name = "sink19"} : <f32>
    %91 = mux %index_45 [%trueResult, %falseResult_43] {handshake.bb = 4 : ui32, handshake.name = "mux5"} : <i1>, [<f32>, <f32>] to <f32>
    %result_44, %index_45 = control_merge [%trueResult_0, %falseResult_41]  {handshake.bb = 4 : ui32, handshake.name = "control_merge3"} : [<>, <>] to <>, <i1>
    %92 = mux %index_47 [%trueResult_14, %91] {handshake.bb = 5 : ui32, handshake.name = "mux6"} : <i1>, [<f32>, <f32>] to <f32>
    %result_46, %index_47 = control_merge [%trueResult_16, %result_44]  {handshake.bb = 5 : ui32, handshake.name = "control_merge4"} : [<>, <>] to <>, <i1>
    sink %result_46 {handshake.name = "sink20"} : <>
    end {handshake.bb = 5 : ui32, handshake.name = "end0"} %92, %0#1 : <f32>, <>
  }
}

