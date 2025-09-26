module {
  handshake.func @if_float2(%arg0: !handshake.channel<f32>, %arg1: memref<100xf32>, %arg2: memref<100xf32>, %arg3: !handshake.control<>, %arg4: !handshake.control<>, %arg5: !handshake.control<>, ...) -> (!handshake.channel<f32>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["x0", "a", "minus_trace", "a_start", "minus_trace_start", "start"], handshake.cfdfcThroughput = #handshake<cfdfcThroughput {"0" = 0.023809523809523808 : f64, "1" = 0.023809523809523808 : f64}>, handshake.cfdfcToBBList = #handshake<cfdfcToBBList {"0" = [1 : ui32, 2 : ui32, 4 : ui32], "1" = [3 : ui32, 4 : ui32, 1 : ui32]}>, resNames = ["out0", "a_end", "minus_trace_end", "end"]} {
    %0:3 = fork [3] %arg5 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork0"} : <>
    %memEnd = mem_controller[%arg2 : memref<100xf32>] %arg4 (%43, %addressResult_9, %dataResult_10) %73#1 {connectedBlocks = [3 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller2"} :    (!handshake.channel<i32>, !handshake.channel<i7>, !handshake.channel<f32>) -> ()
    %outputs, %memEnd_0 = mem_controller[%arg1 : memref<100xf32>] %arg3 (%addressResult) %73#0 {connectedBlocks = [1 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller3"} :    (!handshake.channel<i7>) -> !handshake.channel<f32>
    %1 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant2", value = false} : <>, <i1>
    %2 = br %1 {handshake.bb = 0 : ui32, handshake.name = "br3"} : <i1>
    %3 = extsi %2 {handshake.bb = 0 : ui32, handshake.name = "extsi4"} : <i1> to <i8>
    %4 = br %arg0 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "br4"} : <f32>
    %5 = br %0#2 {handshake.bb = 0 : ui32, handshake.name = "br5"} : <>
    %6 = mux %15#0 [%3, %trueResult_13] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i8>, <i8>] to <i8>
    %7 = buffer %6, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer0"} : <i8>
    %8:2 = fork [2] %7 {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <i8>
    %9 = buffer %8#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer1"} : <i8>
    %10 = trunci %9 {handshake.bb = 1 : ui32, handshake.name = "trunci0"} : <i8> to <i7>
    %11 = buffer %15#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer6"} : <i1>
    %12 = mux %11 [%4, %trueResult_15] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i1>, [<f32>, <f32>] to <f32>
    %13 = buffer %12, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer3"} : <f32>
    %14:3 = fork [3] %13 {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <f32>
    %result, %index = control_merge [%5, %trueResult_17]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %15:2 = fork [2] %index {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <i1>
    %16 = source {handshake.bb = 1 : ui32, handshake.name = "source0"} : <>
    %17 = constant %16 {handshake.bb = 1 : ui32, handshake.name = "constant7", value = -0.899999976 : f32} : <>, <f32>
    %18 = source {handshake.bb = 1 : ui32, handshake.name = "source1"} : <>
    %19 = constant %18 {handshake.bb = 1 : ui32, handshake.name = "constant8", value = 0.000000e+00 : f32} : <>, <f32>
    %addressResult, %dataResult = load[%10] %outputs {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i7>, <f32>, <i7>, <f32>
    %20 = mulf %dataResult, %14#2 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "mulf0", internal_delay = "2_875333"} : <f32>
    %21 = mulf %14#1, %17 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "mulf1", internal_delay = "2_875333"} : <f32>
    %22 = addf %20, %21 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "addf0", internal_delay = "2_922000"} : <f32>
    %23 = cmpf ugt, %22, %19 {handshake.bb = 1 : ui32, handshake.name = "cmpf0", internal_delay = "0_000000"} : <f32>
    %24:3 = fork [3] %23 {handshake.bb = 1 : ui32, handshake.name = "fork4"} : <i1>
    %25 = buffer %8#1, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer2"} : <i8>
    %trueResult, %falseResult = cond_br %24#0, %25 {handshake.bb = 1 : ui32, handshake.name = "cond_br2"} : <i1>, <i8>
    %26 = buffer %14#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer4"} : <f32>
    %trueResult_1, %falseResult_2 = cond_br %24#2, %26 {handshake.bb = 1 : ui32, handshake.name = "cond_br3"} : <i1>, <f32>
    %27 = buffer %result, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer5"} : <>
    %trueResult_3, %falseResult_4 = cond_br %24#1, %27 {handshake.bb = 1 : ui32, handshake.name = "cond_br4"} : <i1>, <>
    %28 = merge %falseResult {handshake.bb = 2 : ui32, handshake.name = "merge0"} : <i8>
    %29 = merge %falseResult_2 {handshake.bb = 2 : ui32, handshake.name = "merge1"} : <f32>
    %result_5, %index_6 = control_merge [%falseResult_4]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>] to <>, <i1>
    sink %index_6 {handshake.name = "sink0"} : <i1>
    %30 = source {handshake.bb = 2 : ui32, handshake.name = "source2"} : <>
    %31 = constant %30 {handshake.bb = 2 : ui32, handshake.name = "constant9", value = 3.000000e+00 : f32} : <>, <f32>
    %32 = addf %29, %31 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "addf1", internal_delay = "2_922000"} : <f32>
    %33 = br %32 {handshake.bb = 2 : ui32, handshake.name = "br6"} : <f32>
    %34 = br %28 {handshake.bb = 2 : ui32, handshake.name = "br7"} : <i8>
    %35 = br %result_5 {handshake.bb = 2 : ui32, handshake.name = "br8"} : <>
    %36 = merge %trueResult {handshake.bb = 3 : ui32, handshake.name = "merge2"} : <i8>
    %37:2 = fork [2] %36 {handshake.bb = 3 : ui32, handshake.name = "fork5"} : <i8>
    %38 = trunci %37#0 {handshake.bb = 3 : ui32, handshake.name = "trunci1"} : <i8> to <i7>
    %39 = merge %trueResult_1 {handshake.bb = 3 : ui32, handshake.name = "merge3"} : <f32>
    %40:2 = fork [2] %39 {handshake.bb = 3 : ui32, handshake.name = "fork6"} : <f32>
    %result_7, %index_8 = control_merge [%trueResult_3]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>] to <>, <i1>
    sink %index_8 {handshake.name = "sink1"} : <i1>
    %41:2 = fork [2] %result_7 {handshake.bb = 3 : ui32, handshake.name = "fork7"} : <>
    %42 = constant %41#0 {handshake.bb = 3 : ui32, handshake.name = "constant3", value = 1 : i2} : <>, <i2>
    %43 = extsi %42 {handshake.bb = 3 : ui32, handshake.name = "extsi1"} : <i2> to <i32>
    %44 = source {handshake.bb = 3 : ui32, handshake.name = "source3"} : <>
    %45 = constant %44 {handshake.bb = 3 : ui32, handshake.name = "constant10", value = 1.000000e+00 : f32} : <>, <f32>
    %addressResult_9, %dataResult_10 = store[%38] %40#0 {handshake.bb = 3 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store0"} : <i7>, <f32>, <i7>, <f32>
    %46 = addf %40#1, %45 {fastmath = #arith.fastmath<none>, handshake.bb = 3 : ui32, handshake.name = "addf2", internal_delay = "2_922000"} : <f32>
    %47 = br %46 {handshake.bb = 3 : ui32, handshake.name = "br9"} : <f32>
    %48 = br %37#1 {handshake.bb = 3 : ui32, handshake.name = "br10"} : <i8>
    %49 = br %41#1 {handshake.bb = 3 : ui32, handshake.name = "br11"} : <>
    %50 = buffer %55#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 4 : ui32, handshake.name = "buffer9"} : <i1>
    %51 = mux %50 [%33, %47] {handshake.bb = 4 : ui32, handshake.name = "mux2"} : <i1>, [<f32>, <f32>] to <f32>
    %52 = mux %55#0 [%34, %48] {handshake.bb = 4 : ui32, handshake.name = "mux3"} : <i1>, [<i8>, <i8>] to <i8>
    %53 = buffer %52, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 4 : ui32, handshake.name = "buffer8"} : <i8>
    %54 = extsi %53 {handshake.bb = 4 : ui32, handshake.name = "extsi5"} : <i8> to <i9>
    %result_11, %index_12 = control_merge [%35, %49]  {handshake.bb = 4 : ui32, handshake.name = "control_merge3"} : [<>, <>] to <>, <i1>
    %55:2 = fork [2] %index_12 {handshake.bb = 4 : ui32, handshake.name = "fork8"} : <i1>
    %56 = source {handshake.bb = 4 : ui32, handshake.name = "source4"} : <>
    %57 = constant %56 {handshake.bb = 4 : ui32, handshake.name = "constant11", value = 1.000000e+00 : f32} : <>, <f32>
    %58 = source {handshake.bb = 4 : ui32, handshake.name = "source5"} : <>
    %59 = constant %58 {handshake.bb = 4 : ui32, handshake.name = "constant4", value = 1 : i2} : <>, <i2>
    %60 = extsi %59 {handshake.bb = 4 : ui32, handshake.name = "extsi6"} : <i2> to <i9>
    %61 = source {handshake.bb = 4 : ui32, handshake.name = "source6"} : <>
    %62 = constant %61 {handshake.bb = 4 : ui32, handshake.name = "constant5", value = 100 : i8} : <>, <i8>
    %63 = extsi %62 {handshake.bb = 4 : ui32, handshake.name = "extsi7"} : <i8> to <i9>
    %64 = buffer %51, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 4 : ui32, handshake.name = "buffer7"} : <f32>
    %65 = divf %57, %64 {fastmath = #arith.fastmath<none>, handshake.bb = 4 : ui32, handshake.name = "divf0", internal_delay = "3_812000"} : <f32>
    %66 = addi %54, %60 {handshake.bb = 4 : ui32, handshake.name = "addi0"} : <i9>
    %67:2 = fork [2] %66 {handshake.bb = 4 : ui32, handshake.name = "fork9"} : <i9>
    %68 = trunci %67#0 {handshake.bb = 4 : ui32, handshake.name = "trunci2"} : <i9> to <i8>
    %69 = cmpi ult, %67#1, %63 {handshake.bb = 4 : ui32, handshake.name = "cmpi0"} : <i9>
    %70:3 = fork [3] %69 {handshake.bb = 4 : ui32, handshake.name = "fork10"} : <i1>
    %trueResult_13, %falseResult_14 = cond_br %70#0, %68 {handshake.bb = 4 : ui32, handshake.name = "cond_br5"} : <i1>, <i8>
    sink %falseResult_14 {handshake.name = "sink2"} : <i8>
    %71 = buffer %70#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 4 : ui32, handshake.name = "buffer10"} : <i1>
    %trueResult_15, %falseResult_16 = cond_br %71, %65 {handshake.bb = 4 : ui32, handshake.name = "cond_br6"} : <i1>, <f32>
    %trueResult_17, %falseResult_18 = cond_br %70#2, %result_11 {handshake.bb = 4 : ui32, handshake.name = "cond_br7"} : <i1>, <>
    %72 = merge %falseResult_16 {handshake.bb = 5 : ui32, handshake.name = "merge4"} : <f32>
    %result_19, %index_20 = control_merge [%falseResult_18]  {handshake.bb = 5 : ui32, handshake.name = "control_merge4"} : [<>] to <>, <i1>
    sink %index_20 {handshake.name = "sink3"} : <i1>
    %73:2 = fork [2] %result_19 {handshake.bb = 5 : ui32, handshake.name = "fork11"} : <>
    end {handshake.bb = 5 : ui32, handshake.name = "end0"} %72, %memEnd_0, %memEnd, %0#1 : <f32>, <>, <>, <>
  }
}

