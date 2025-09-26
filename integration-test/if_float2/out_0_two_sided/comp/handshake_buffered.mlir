module {
  handshake.func @if_float2(%arg0: !handshake.channel<f32>, %arg1: memref<100xf32>, %arg2: memref<100xf32>, %arg3: !handshake.control<>, %arg4: !handshake.control<>, %arg5: !handshake.control<>, ...) -> (!handshake.channel<f32>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["x0", "a", "minus_trace", "a_start", "minus_trace_start", "start"], handshake.cfdfcThroughput = #handshake<cfdfcThroughput {"0" = 0.034482758620689585 : f64}>, handshake.cfdfcToBBList = #handshake<cfdfcToBBList {"0" = [1 : ui32]}>, resNames = ["out0", "a_end", "minus_trace_end", "end"]} {
    %0:3 = fork [3] %arg5 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork0"} : <>
    %memEnd = mem_controller[%arg2 : memref<100xf32>] %arg4 (%60, %addressResult_5, %dataResult_6) %96#1 {connectedBlocks = [1 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller2"} :    (!handshake.channel<i32>, !handshake.channel<i7>, !handshake.channel<f32>) -> ()
    %outputs, %memEnd_0 = mem_controller[%arg1 : memref<100xf32>] %arg3 (%addressResult) %96#0 {connectedBlocks = [1 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller3"} :    (!handshake.channel<i7>) -> !handshake.channel<f32>
    %1 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant2", value = false} : <>, <i1>
    %2 = br %1 {handshake.bb = 0 : ui32, handshake.name = "br3"} : <i1>
    %3 = extsi %2 {handshake.bb = 0 : ui32, handshake.name = "extsi4"} : <i1> to <i8>
    %4 = br %arg0 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "br4"} : <f32>
    %5 = br %0#2 {handshake.bb = 0 : ui32, handshake.name = "br5"} : <>
    %6 = mux %16#0 [%3, %trueResult] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i8>, <i8>] to <i8>
    %7 = buffer %6, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer1"} : <i8>
    %8 = buffer %7, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer2"} : <i8>
    %9:3 = fork [3] %8 {handshake.bb = 1 : ui32, handshake.name = "fork12"} : <i8>
    %10 = trunci %9#0 {handshake.bb = 1 : ui32, handshake.name = "trunci0"} : <i8> to <i7>
    %11 = mux %16#1 [%4, %trueResult_7] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i1>, [<f32>, <f32>] to <f32>
    %12 = buffer %11, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer3"} : <f32>
    %13:4 = fork [4] %12 {handshake.bb = 1 : ui32, handshake.name = "fork13"} : <f32>
    %result, %index = control_merge [%5, %trueResult_9]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %14 = buffer %result, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer6"} : <>
    %15:2 = fork [2] %14 {handshake.bb = 1 : ui32, handshake.name = "fork14"} : <>
    %16:2 = fork [2] %index {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <i1>
    %17 = source {handshake.bb = 1 : ui32, handshake.name = "source0"} : <>
    %18 = constant %17 {handshake.bb = 1 : ui32, handshake.name = "constant7", value = -0.899999976 : f32} : <>, <f32>
    %19 = source {handshake.bb = 1 : ui32, handshake.name = "source1"} : <>
    %20 = constant %19 {handshake.bb = 1 : ui32, handshake.name = "constant8", value = 0.000000e+00 : f32} : <>, <f32>
    %addressResult, %dataResult = load[%10] %outputs {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i7>, <f32>, <i7>, <f32>
    %21 = buffer %13#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer5"} : <f32>
    %22 = mulf %dataResult, %21 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "mulf0", internal_delay = "2_875333"} : <f32>
    %23 = buffer %13#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer4"} : <f32>
    %24 = mulf %23, %18 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "mulf1", internal_delay = "2_875333"} : <f32>
    %25 = buffer %22, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer7"} : <f32>
    %26 = addf %25, %24 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "addf0", internal_delay = "2_922000"} : <f32>
    %27 = cmpf ugt, %26, %20 {handshake.bb = 1 : ui32, handshake.name = "cmpf0", internal_delay = "0_000000"} : <f32>
    %28:5 = fork [5] %27 {handshake.bb = 1 : ui32, handshake.name = "fork46"} : <i1>
    %29 = buffer %33, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer0"} : <i1>
    %30 = buffer %29, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer8"} : <i1>
    %31 = init %30 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "init0", initToken = 0 : ui1} : <i1>
    %32:5 = fork [5] %31 {handshake.bb = 1 : ui32, handshake.name = "fork47"} : <i1>
    %33 = not %32#4 {handshake.bb = 1 : ui32, handshake.name = "not1"} : <i1>
    %34 = buffer %32#3, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer10"} : <i1>
    %35 = mux %34 [%40, %28#0] {handshake.bb = 1 : ui32, handshake.name = "mux5"} : <i1>, [<i1>, <i1>] to <i1>
    %36 = buffer %35, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer11"} : <i1>
    %37:6 = fork [6] %36 {handshake.bb = 1 : ui32, handshake.name = "fork48"} : <i1>
    %38 = buffer %73, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer16"} : <>
    %39 = passer %38[%37#5] {handshake.bb = 1 : ui32, handshake.name = "passer6"} : <>, <i1>
    %40 = not %28#4 {handshake.bb = 1 : ui32, handshake.name = "not0"} : <i1>
    %41 = merge %9#1 {handshake.bb = 1 : ui32, handshake.name = "merge0"} : <i8>
    %42 = merge %13#2 {handshake.bb = 1 : ui32, handshake.name = "merge1"} : <f32>
    %result_1, %index_2 = control_merge [%15#0]  {handshake.bb = 1 : ui32, handshake.name = "control_merge1"} : [<>] to <>, <i1>
    sink %index_2 {handshake.name = "sink4"} : <i1>
    %43 = source {handshake.bb = 1 : ui32, handshake.name = "source2"} : <>
    %44 = constant %43 {handshake.bb = 1 : ui32, handshake.name = "constant9", value = 3.000000e+00 : f32} : <>, <f32>
    %45 = addf %42, %44 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "addf1", internal_delay = "2_922000"} : <f32>
    %46 = br %45 {handshake.bb = 1 : ui32, handshake.name = "br6"} : <f32>
    %47 = br %41 {handshake.bb = 1 : ui32, handshake.name = "br7"} : <i8>
    %48 = br %result_1 {handshake.bb = 1 : ui32, handshake.name = "br8"} : <>
    %49:2 = fork [2] %50 {handshake.bb = 1 : ui32, handshake.name = "fork41"} : <i8>
    %50 = merge %9#2 {handshake.bb = 1 : ui32, handshake.name = "merge2"} : <i8>
    %51 = passer %53[%28#3] {handshake.bb = 1 : ui32, handshake.name = "passer7"} : <i7>, <i1>
    %52 = buffer %49#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer12"} : <i8>
    %53 = trunci %52 {handshake.bb = 1 : ui32, handshake.name = "trunci1"} : <i8> to <i7>
    %54 = buffer %56#0, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer13"} : <f32>
    %55 = passer %54[%28#2] {handshake.bb = 1 : ui32, handshake.name = "passer8"} : <f32>, <i1>
    %56:2 = fork [2] %57 {handshake.bb = 1 : ui32, handshake.name = "fork42"} : <f32>
    %57 = merge %13#3 {handshake.bb = 1 : ui32, handshake.name = "merge3"} : <f32>
    %58:2 = fork [2] %result_3 {handshake.bb = 1 : ui32, handshake.name = "fork43"} : <>
    %result_3, %index_4 = control_merge [%15#1]  {handshake.bb = 1 : ui32, handshake.name = "control_merge2"} : [<>] to <>, <i1>
    sink %index_4 {handshake.name = "sink5"} : <i1>
    %59 = constant %58#0 {handshake.bb = 1 : ui32, handshake.name = "constant3", value = 1 : i2} : <>, <i2>
    %60 = passer %61[%28#1] {handshake.bb = 1 : ui32, handshake.name = "passer9"} : <i32>, <i1>
    %61 = extsi %59 {handshake.bb = 1 : ui32, handshake.name = "extsi1"} : <i2> to <i32>
    %62 = source {handshake.bb = 1 : ui32, handshake.name = "source3"} : <>
    %63 = constant %62 {handshake.bb = 1 : ui32, handshake.name = "constant10", value = 1.000000e+00 : f32} : <>, <f32>
    %addressResult_5, %dataResult_6 = store[%51] %55 {handshake.bb = 1 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store0"} : <i7>, <f32>, <i7>, <f32>
    %64 = addf %56#1, %63 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "addf2", internal_delay = "2_922000"} : <f32>
    %65 = br %64 {handshake.bb = 1 : ui32, handshake.name = "br9"} : <f32>
    %66 = br %49#1 {handshake.bb = 1 : ui32, handshake.name = "br10"} : <i8>
    %67 = br %58#1 {handshake.bb = 1 : ui32, handshake.name = "br11"} : <>
    %68 = mux %32#0 [%46, %65] {handshake.bb = 1 : ui32, handshake.name = "mux2"} : <i1>, [<f32>, <f32>] to <f32>
    %69 = mux %32#1 [%47, %66] {handshake.bb = 1 : ui32, handshake.name = "mux3"} : <i1>, [<i8>, <i8>] to <i8>
    %70 = buffer %69, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer15"} : <i8>
    %71 = extsi %70 {handshake.bb = 1 : ui32, handshake.name = "extsi5"} : <i8> to <i9>
    %72 = buffer %32#2, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer9"} : <i1>
    %73 = mux %72 [%48, %67] {handshake.bb = 1 : ui32, handshake.name = "mux4"} : <i1>, [<>, <>] to <>
    %74 = source {handshake.bb = 1 : ui32, handshake.name = "source4"} : <>
    %75 = constant %74 {handshake.bb = 1 : ui32, handshake.name = "constant11", value = 1.000000e+00 : f32} : <>, <f32>
    %76 = source {handshake.bb = 1 : ui32, handshake.name = "source5"} : <>
    %77 = constant %76 {handshake.bb = 1 : ui32, handshake.name = "constant4", value = 1 : i2} : <>, <i2>
    %78 = extsi %77 {handshake.bb = 1 : ui32, handshake.name = "extsi6"} : <i2> to <i9>
    %79 = source {handshake.bb = 1 : ui32, handshake.name = "source6"} : <>
    %80 = constant %79 {handshake.bb = 1 : ui32, handshake.name = "constant5", value = 100 : i8} : <>, <i8>
    %81 = extsi %80 {handshake.bb = 1 : ui32, handshake.name = "extsi7"} : <i8> to <i9>
    %82 = passer %84[%37#0] {handshake.bb = 1 : ui32, handshake.name = "passer10"} : <f32>, <i1>
    %83 = buffer %68, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer14"} : <f32>
    %84 = divf %75, %83 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "divf0", internal_delay = "3_812000"} : <f32>
    %85 = buffer %87, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer17"} : <i9>
    %86:2 = fork [2] %85 {handshake.bb = 1 : ui32, handshake.name = "fork44"} : <i9>
    %87 = addi %71, %78 {handshake.bb = 1 : ui32, handshake.name = "addi0"} : <i9>
    %88 = passer %89[%37#4] {handshake.bb = 1 : ui32, handshake.name = "passer11"} : <i8>, <i1>
    %89 = trunci %86#0 {handshake.bb = 1 : ui32, handshake.name = "trunci2"} : <i9> to <i8>
    %90 = passer %93#0[%37#3] {handshake.bb = 1 : ui32, handshake.name = "passer12"} : <i1>, <i1>
    %91 = passer %93#1[%37#2] {handshake.bb = 1 : ui32, handshake.name = "passer13"} : <i1>, <i1>
    %92 = passer %93#2[%37#1] {handshake.bb = 1 : ui32, handshake.name = "passer14"} : <i1>, <i1>
    %93:3 = fork [3] %94 {handshake.bb = 1 : ui32, handshake.name = "fork45"} : <i1>
    %94 = cmpi ult, %86#1, %81 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i9>
    %trueResult, %falseResult = cond_br %90, %88 {handshake.bb = 1 : ui32, handshake.name = "cond_br5"} : <i1>, <i8>
    sink %falseResult {handshake.name = "sink2"} : <i8>
    %trueResult_7, %falseResult_8 = cond_br %91, %82 {handshake.bb = 1 : ui32, handshake.name = "cond_br6"} : <i1>, <f32>
    %trueResult_9, %falseResult_10 = cond_br %92, %39 {handshake.bb = 1 : ui32, handshake.name = "cond_br7"} : <i1>, <>
    %95 = merge %falseResult_8 {handshake.bb = 2 : ui32, handshake.name = "merge4"} : <f32>
    %result_11, %index_12 = control_merge [%falseResult_10]  {handshake.bb = 2 : ui32, handshake.name = "control_merge4"} : [<>] to <>, <i1>
    sink %index_12 {handshake.name = "sink3"} : <i1>
    %96:2 = fork [2] %result_11 {handshake.bb = 2 : ui32, handshake.name = "fork11"} : <>
    end {handshake.bb = 2 : ui32, handshake.name = "end0"} %95, %memEnd_0, %memEnd, %0#1 : <f32>, <>, <>, <>
  }
}

