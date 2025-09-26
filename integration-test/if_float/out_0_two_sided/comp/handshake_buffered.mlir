module {
  handshake.func @if_float(%arg0: !handshake.channel<f32>, %arg1: memref<100xf32>, %arg2: memref<100xf32>, %arg3: !handshake.control<>, %arg4: !handshake.control<>, %arg5: !handshake.control<>, ...) -> (!handshake.channel<f32>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["x0", "a", "minus_trace", "a_start", "minus_trace_start", "start"], handshake.cfdfcThroughput = #handshake<cfdfcThroughput {"0" = 0.034482758620689655 : f64}>, handshake.cfdfcToBBList = #handshake<cfdfcToBBList {"0" = [1 : ui32]}>, resNames = ["out0", "a_end", "minus_trace_end", "end"]} {
    %0:3 = fork [3] %arg5 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork0"} : <>
    %memEnd = mem_controller[%arg2 : memref<100xf32>] %arg4 (%59, %addressResult_5, %dataResult_6) %95#1 {connectedBlocks = [1 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller2"} :    (!handshake.channel<i32>, !handshake.channel<i7>, !handshake.channel<f32>) -> ()
    %outputs, %memEnd_0 = mem_controller[%arg1 : memref<100xf32>] %arg3 (%addressResult) %95#0 {connectedBlocks = [1 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller3"} :    (!handshake.channel<i7>) -> !handshake.channel<f32>
    %1 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant2", value = false} : <>, <i1>
    %2 = br %1 {handshake.bb = 0 : ui32, handshake.name = "br3"} : <i1>
    %3 = extsi %2 {handshake.bb = 0 : ui32, handshake.name = "extsi4"} : <i1> to <i8>
    %4 = br %arg0 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "br4"} : <f32>
    %5 = br %0#2 {handshake.bb = 0 : ui32, handshake.name = "br5"} : <>
    %6 = mux %17#0 [%3, %trueResult] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i8>, <i8>] to <i8>
    %7 = buffer %6, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer1"} : <i8>
    %8:3 = fork [3] %7 {handshake.bb = 1 : ui32, handshake.name = "fork14"} : <i8>
    %9 = buffer %8#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer2"} : <i8>
    %10 = trunci %9 {handshake.bb = 1 : ui32, handshake.name = "trunci0"} : <i8> to <i7>
    %11 = buffer %17#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer6"} : <i1>
    %12 = mux %11 [%4, %trueResult_7] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i1>, [<f32>, <f32>] to <f32>
    %13 = buffer %12, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer4"} : <f32>
    %14:4 = fork [4] %13 {handshake.bb = 1 : ui32, handshake.name = "fork15"} : <f32>
    %result, %index = control_merge [%5, %trueResult_9]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %15 = buffer %result, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer5"} : <>
    %16:2 = fork [2] %15 {handshake.bb = 1 : ui32, handshake.name = "fork16"} : <>
    %17:2 = fork [2] %index {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <i1>
    %18 = source {handshake.bb = 1 : ui32, handshake.name = "source0"} : <>
    %19 = constant %18 {handshake.bb = 1 : ui32, handshake.name = "constant6", value = -0.899999976 : f32} : <>, <f32>
    %20 = source {handshake.bb = 1 : ui32, handshake.name = "source1"} : <>
    %21 = constant %20 {handshake.bb = 1 : ui32, handshake.name = "constant7", value = 0.000000e+00 : f32} : <>, <f32>
    %addressResult, %dataResult = load[%10] %outputs {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i7>, <f32>, <i7>, <f32>
    %22 = mulf %dataResult, %14#1 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "mulf0", internal_delay = "2_875333"} : <f32>
    %23 = mulf %14#0, %19 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "mulf1", internal_delay = "2_875333"} : <f32>
    %24 = addf %22, %23 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "addf0", internal_delay = "2_922000"} : <f32>
    %25 = buffer %24, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer7"} : <f32>
    %26 = cmpf ugt, %25, %21 {handshake.bb = 1 : ui32, handshake.name = "cmpf0", internal_delay = "0_000000"} : <f32>
    %27:5 = fork [5] %26 {handshake.bb = 1 : ui32, handshake.name = "fork48"} : <i1>
    %28 = buffer %32, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer0"} : <i1>
    %29 = buffer %28, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer8"} : <i1>
    %30 = init %29 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "init0", initToken = 0 : ui1} : <i1>
    %31:5 = fork [5] %30 {handshake.bb = 1 : ui32, handshake.name = "fork49"} : <i1>
    %32 = not %31#4 {handshake.bb = 1 : ui32, handshake.name = "not1"} : <i1>
    %33 = mux %31#3 [%38, %27#0] {handshake.bb = 1 : ui32, handshake.name = "mux5"} : <i1>, [<i1>, <i1>] to <i1>
    %34 = buffer %33, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer9"} : <i1>
    %35:6 = fork [6] %34 {handshake.bb = 1 : ui32, handshake.name = "fork50"} : <i1>
    %36 = buffer %74, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer16"} : <>
    %37 = passer %36[%35#0] {handshake.bb = 1 : ui32, handshake.name = "passer6"} : <>, <i1>
    %38 = not %27#4 {handshake.bb = 1 : ui32, handshake.name = "not0"} : <i1>
    %39 = merge %8#1 {handshake.bb = 1 : ui32, handshake.name = "merge0"} : <i8>
    %40 = merge %14#2 {handshake.bb = 1 : ui32, handshake.name = "merge1"} : <f32>
    %result_1, %index_2 = control_merge [%16#0]  {handshake.bb = 1 : ui32, handshake.name = "control_merge1"} : [<>] to <>, <i1>
    sink %index_2 {handshake.name = "sink0"} : <i1>
    %41 = source {handshake.bb = 1 : ui32, handshake.name = "source2"} : <>
    %42 = constant %41 {handshake.bb = 1 : ui32, handshake.name = "constant8", value = 1.100000e+00 : f32} : <>, <f32>
    %43 = mulf %40, %42 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "mulf2", internal_delay = "2_875333"} : <f32>
    %44 = buffer %43, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer11"} : <f32>
    %45 = br %44 {handshake.bb = 1 : ui32, handshake.name = "br6"} : <f32>
    %46 = br %39 {handshake.bb = 1 : ui32, handshake.name = "br7"} : <i8>
    %47 = br %result_1 {handshake.bb = 1 : ui32, handshake.name = "br8"} : <>
    %48:2 = fork [2] %50 {handshake.bb = 1 : ui32, handshake.name = "fork43"} : <i8>
    %49 = buffer %8#2, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer3"} : <i8>
    %50 = merge %49 {handshake.bb = 1 : ui32, handshake.name = "merge2"} : <i8>
    %51 = passer %52[%27#3] {handshake.bb = 1 : ui32, handshake.name = "passer7"} : <i7>, <i1>
    %52 = trunci %48#0 {handshake.bb = 1 : ui32, handshake.name = "trunci1"} : <i8> to <i7>
    %53 = buffer %55#0, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer13"} : <f32>
    %54 = passer %53[%27#2] {handshake.bb = 1 : ui32, handshake.name = "passer8"} : <f32>, <i1>
    %55:2 = fork [2] %56 {handshake.bb = 1 : ui32, handshake.name = "fork44"} : <f32>
    %56 = merge %14#3 {handshake.bb = 1 : ui32, handshake.name = "merge3"} : <f32>
    %57:2 = fork [2] %result_3 {handshake.bb = 1 : ui32, handshake.name = "fork45"} : <>
    %result_3, %index_4 = control_merge [%16#1]  {handshake.bb = 1 : ui32, handshake.name = "control_merge2"} : [<>] to <>, <i1>
    sink %index_4 {handshake.name = "sink1"} : <i1>
    %58 = constant %57#0 {handshake.bb = 1 : ui32, handshake.name = "constant3", value = 1 : i2} : <>, <i2>
    %59 = passer %60[%27#1] {handshake.bb = 1 : ui32, handshake.name = "passer9"} : <i32>, <i1>
    %60 = extsi %58 {handshake.bb = 1 : ui32, handshake.name = "extsi1"} : <i2> to <i32>
    %61 = source {handshake.bb = 1 : ui32, handshake.name = "source3"} : <>
    %62 = constant %61 {handshake.bb = 1 : ui32, handshake.name = "constant9", value = 1.100000e+00 : f32} : <>, <f32>
    %addressResult_5, %dataResult_6 = store[%51] %54 {handshake.bb = 1 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store0"} : <i7>, <f32>, <i7>, <f32>
    %63 = divf %55#1, %62 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "divf0", internal_delay = "3_812000"} : <f32>
    %64 = br %63 {handshake.bb = 1 : ui32, handshake.name = "br9"} : <f32>
    %65 = br %48#1 {handshake.bb = 1 : ui32, handshake.name = "br10"} : <i8>
    %66 = br %57#1 {handshake.bb = 1 : ui32, handshake.name = "br11"} : <>
    %67 = mux %31#1 [%45, %64] {handshake.bb = 1 : ui32, handshake.name = "mux2"} : <i1>, [<f32>, <f32>] to <f32>
    %68 = buffer %67, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer14"} : <f32>
    %69:2 = fork [2] %68 {handshake.bb = 1 : ui32, handshake.name = "fork8"} : <f32>
    %70 = buffer %46, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer12"} : <i8>
    %71 = mux %31#2 [%70, %65] {handshake.bb = 1 : ui32, handshake.name = "mux3"} : <i1>, [<i8>, <i8>] to <i8>
    %72 = buffer %71, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer15"} : <i8>
    %73 = extsi %72 {handshake.bb = 1 : ui32, handshake.name = "extsi5"} : <i8> to <i9>
    %74 = mux %31#0 [%47, %66] {handshake.bb = 1 : ui32, handshake.name = "mux4"} : <i1>, [<>, <>] to <>
    %75 = source {handshake.bb = 1 : ui32, handshake.name = "source4"} : <>
    %76 = constant %75 {handshake.bb = 1 : ui32, handshake.name = "constant4", value = 1 : i2} : <>, <i2>
    %77 = extsi %76 {handshake.bb = 1 : ui32, handshake.name = "extsi6"} : <i2> to <i9>
    %78 = source {handshake.bb = 1 : ui32, handshake.name = "source5"} : <>
    %79 = constant %78 {handshake.bb = 1 : ui32, handshake.name = "constant12", value = 100 : i8} : <>, <i8>
    %80 = extsi %79 {handshake.bb = 1 : ui32, handshake.name = "extsi7"} : <i8> to <i9>
    %81 = buffer %35#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer10"} : <i1>
    %82 = passer %83[%81] {handshake.bb = 1 : ui32, handshake.name = "passer10"} : <f32>, <i1>
    %83 = addf %69#0, %69#1 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "addf1", internal_delay = "2_922000"} : <f32>
    %84:2 = fork [2] %85 {handshake.bb = 1 : ui32, handshake.name = "fork46"} : <i9>
    %85 = addi %73, %77 {handshake.bb = 1 : ui32, handshake.name = "addi0"} : <i9>
    %86 = passer %87[%35#5] {handshake.bb = 1 : ui32, handshake.name = "passer11"} : <i8>, <i1>
    %87 = trunci %84#0 {handshake.bb = 1 : ui32, handshake.name = "trunci2"} : <i9> to <i8>
    %88 = passer %91#0[%35#4] {handshake.bb = 1 : ui32, handshake.name = "passer12"} : <i1>, <i1>
    %89 = passer %91#1[%35#3] {handshake.bb = 1 : ui32, handshake.name = "passer13"} : <i1>, <i1>
    %90 = passer %91#2[%35#2] {handshake.bb = 1 : ui32, handshake.name = "passer14"} : <i1>, <i1>
    %91:3 = fork [3] %92 {handshake.bb = 1 : ui32, handshake.name = "fork47"} : <i1>
    %92 = cmpi ult, %84#1, %80 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i9>
    %trueResult, %falseResult = cond_br %88, %86 {handshake.bb = 1 : ui32, handshake.name = "cond_br5"} : <i1>, <i8>
    sink %falseResult {handshake.name = "sink2"} : <i8>
    %93 = buffer %89, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer17"} : <i1>
    %trueResult_7, %falseResult_8 = cond_br %93, %82 {handshake.bb = 1 : ui32, handshake.name = "cond_br6"} : <i1>, <f32>
    %trueResult_9, %falseResult_10 = cond_br %90, %37 {handshake.bb = 1 : ui32, handshake.name = "cond_br7"} : <i1>, <>
    %94 = merge %falseResult_8 {handshake.bb = 2 : ui32, handshake.name = "merge4"} : <f32>
    %result_11, %index_12 = control_merge [%falseResult_10]  {handshake.bb = 2 : ui32, handshake.name = "control_merge4"} : [<>] to <>, <i1>
    sink %index_12 {handshake.name = "sink3"} : <i1>
    %95:2 = fork [2] %result_11 {handshake.bb = 2 : ui32, handshake.name = "fork12"} : <>
    end {handshake.bb = 2 : ui32, handshake.name = "end0"} %94, %memEnd_0, %memEnd, %0#1 : <f32>, <>, <>, <>
  }
}

