module {
  handshake.func @if_float(%arg0: !handshake.channel<f32>, %arg1: memref<100xf32>, %arg2: memref<100xf32>, %arg3: !handshake.control<>, %arg4: !handshake.control<>, %arg5: !handshake.control<>, ...) -> (!handshake.channel<f32>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["x0", "a", "minus_trace", "a_start", "minus_trace_start", "start"], resNames = ["out0", "a_end", "minus_trace_end", "end"]} {
    %0:3 = fork [3] %arg5 {handshake.bb = 0 : ui32, handshake.name = "fork0"} : <>
    %memEnd = mem_controller[%arg2 : memref<100xf32>] %arg4 (%36, %addressResult_9, %dataResult_10) %61#1 {connectedBlocks = [3 : i32], handshake.name = "mem_controller2"} :    (!handshake.channel<i32>, !handshake.channel<i7>, !handshake.channel<f32>) -> ()
    %outputs, %memEnd_0 = mem_controller[%arg1 : memref<100xf32>] %arg3 (%addressResult) %61#0 {connectedBlocks = [1 : i32], handshake.name = "mem_controller3"} :    (!handshake.channel<i7>) -> !handshake.channel<f32>
    %1 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant2", value = false} : <>, <i1>
    %2 = br %1 {handshake.bb = 0 : ui32, handshake.name = "br3"} : <i1>
    %3 = extsi %2 {handshake.bb = 0 : ui32, handshake.name = "extsi4"} : <i1> to <i8>
    %4 = br %arg0 {handshake.bb = 0 : ui32, handshake.name = "br4"} : <f32>
    %5 = br %0#2 {handshake.bb = 0 : ui32, handshake.name = "br5"} : <>
    %6 = mux %11#0 [%3, %trueResult_13] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i8>, <i8>] to <i8>
    %7:2 = fork [2] %6 {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <i8>
    %8 = trunci %7#0 {handshake.bb = 1 : ui32, handshake.name = "trunci0"} : <i8> to <i7>
    %9 = mux %11#1 [%4, %trueResult_15] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i1>, [<f32>, <f32>] to <f32>
    %10:3 = fork [3] %9 {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <f32>
    %result, %index = control_merge [%5, %trueResult_17]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %11:2 = fork [2] %index {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <i1>
    %12 = source {handshake.bb = 1 : ui32, handshake.name = "source0"} : <>
    %13 = constant %12 {handshake.bb = 1 : ui32, handshake.name = "constant6", value = -0.899999976 : f32} : <>, <f32>
    %14 = source {handshake.bb = 1 : ui32, handshake.name = "source1"} : <>
    %15 = constant %14 {handshake.bb = 1 : ui32, handshake.name = "constant7", value = 0.000000e+00 : f32} : <>, <f32>
    %addressResult, %dataResult = load[%8] %outputs {handshake.bb = 1 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i7>, <f32>, <i7>, <f32>
    %16 = mulf %dataResult, %10#2 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "mulf0"} : <f32>
    %17 = mulf %10#1, %13 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "mulf1"} : <f32>
    %18 = addf %16, %17 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "addf0"} : <f32>
    %19 = cmpf ugt, %18, %15 {handshake.bb = 1 : ui32, handshake.name = "cmpf0"} : <f32>
    %20:3 = fork [3] %19 {handshake.bb = 1 : ui32, handshake.name = "fork4"} : <i1>
    %trueResult, %falseResult = cond_br %20#0, %7#1 {handshake.bb = 1 : ui32, handshake.name = "cond_br2"} : <i1>, <i8>
    %trueResult_1, %falseResult_2 = cond_br %20#2, %10#0 {handshake.bb = 1 : ui32, handshake.name = "cond_br3"} : <i1>, <f32>
    %trueResult_3, %falseResult_4 = cond_br %20#1, %result {handshake.bb = 1 : ui32, handshake.name = "cond_br4"} : <i1>, <>
    %21 = merge %falseResult {handshake.bb = 2 : ui32, handshake.name = "merge0"} : <i8>
    %22 = merge %falseResult_2 {handshake.bb = 2 : ui32, handshake.name = "merge1"} : <f32>
    %result_5, %index_6 = control_merge [%falseResult_4]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>] to <>, <i1>
    sink %index_6 {handshake.name = "sink0"} : <i1>
    %23 = source {handshake.bb = 2 : ui32, handshake.name = "source2"} : <>
    %24 = constant %23 {handshake.bb = 2 : ui32, handshake.name = "constant8", value = 1.100000e+00 : f32} : <>, <f32>
    %25 = mulf %22, %24 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "mulf2"} : <f32>
    %26 = br %25 {handshake.bb = 2 : ui32, handshake.name = "br6"} : <f32>
    %27 = br %21 {handshake.bb = 2 : ui32, handshake.name = "br7"} : <i8>
    %28 = br %result_5 {handshake.bb = 2 : ui32, handshake.name = "br8"} : <>
    %29 = merge %trueResult {handshake.bb = 3 : ui32, handshake.name = "merge2"} : <i8>
    %30:2 = fork [2] %29 {handshake.bb = 3 : ui32, handshake.name = "fork5"} : <i8>
    %31 = trunci %30#0 {handshake.bb = 3 : ui32, handshake.name = "trunci1"} : <i8> to <i7>
    %32 = merge %trueResult_1 {handshake.bb = 3 : ui32, handshake.name = "merge3"} : <f32>
    %33:2 = fork [2] %32 {handshake.bb = 3 : ui32, handshake.name = "fork6"} : <f32>
    %result_7, %index_8 = control_merge [%trueResult_3]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>] to <>, <i1>
    sink %index_8 {handshake.name = "sink1"} : <i1>
    %34:2 = fork [2] %result_7 {handshake.bb = 3 : ui32, handshake.name = "fork7"} : <>
    %35 = constant %34#0 {handshake.bb = 3 : ui32, handshake.name = "constant3", value = 1 : i2} : <>, <i2>
    %36 = extsi %35 {handshake.bb = 3 : ui32, handshake.name = "extsi1"} : <i2> to <i32>
    %37 = source {handshake.bb = 3 : ui32, handshake.name = "source3"} : <>
    %38 = constant %37 {handshake.bb = 3 : ui32, handshake.name = "constant9", value = 1.100000e+00 : f32} : <>, <f32>
    %addressResult_9, %dataResult_10 = store[%31] %33#0 {handshake.bb = 3 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store0"} : <i7>, <f32>, <i7>, <f32>
    %39 = divf %33#1, %38 {fastmath = #arith.fastmath<none>, handshake.bb = 3 : ui32, handshake.name = "divf0"} : <f32>
    %40 = br %39 {handshake.bb = 3 : ui32, handshake.name = "br9"} : <f32>
    %41 = br %30#1 {handshake.bb = 3 : ui32, handshake.name = "br10"} : <i8>
    %42 = br %34#1 {handshake.bb = 3 : ui32, handshake.name = "br11"} : <>
    %43 = mux %47#1 [%26, %40] {handshake.bb = 4 : ui32, handshake.name = "mux2"} : <i1>, [<f32>, <f32>] to <f32>
    %44:2 = fork [2] %43 {handshake.bb = 4 : ui32, handshake.name = "fork8"} : <f32>
    %45 = mux %47#0 [%27, %41] {handshake.bb = 4 : ui32, handshake.name = "mux3"} : <i1>, [<i8>, <i8>] to <i8>
    %46 = extsi %45 {handshake.bb = 4 : ui32, handshake.name = "extsi5"} : <i8> to <i9>
    %result_11, %index_12 = control_merge [%28, %42]  {handshake.bb = 4 : ui32, handshake.name = "control_merge3"} : [<>, <>] to <>, <i1>
    %47:2 = fork [2] %index_12 {handshake.bb = 4 : ui32, handshake.name = "fork9"} : <i1>
    %48 = source {handshake.bb = 4 : ui32, handshake.name = "source4"} : <>
    %49 = constant %48 {handshake.bb = 4 : ui32, handshake.name = "constant4", value = 1 : i2} : <>, <i2>
    %50 = extsi %49 {handshake.bb = 4 : ui32, handshake.name = "extsi6"} : <i2> to <i9>
    %51 = source {handshake.bb = 4 : ui32, handshake.name = "source5"} : <>
    %52 = constant %51 {handshake.bb = 4 : ui32, handshake.name = "constant12", value = 100 : i8} : <>, <i8>
    %53 = extsi %52 {handshake.bb = 4 : ui32, handshake.name = "extsi7"} : <i8> to <i9>
    %54 = addf %44#0, %44#1 {fastmath = #arith.fastmath<none>, handshake.bb = 4 : ui32, handshake.name = "addf1"} : <f32>
    %55 = addi %46, %50 {handshake.bb = 4 : ui32, handshake.name = "addi0"} : <i9>
    %56:2 = fork [2] %55 {handshake.bb = 4 : ui32, handshake.name = "fork10"} : <i9>
    %57 = trunci %56#0 {handshake.bb = 4 : ui32, handshake.name = "trunci2"} : <i9> to <i8>
    %58 = cmpi ult, %56#1, %53 {handshake.bb = 4 : ui32, handshake.name = "cmpi0"} : <i9>
    %59:3 = fork [3] %58 {handshake.bb = 4 : ui32, handshake.name = "fork11"} : <i1>
    %trueResult_13, %falseResult_14 = cond_br %59#0, %57 {handshake.bb = 4 : ui32, handshake.name = "cond_br5"} : <i1>, <i8>
    sink %falseResult_14 {handshake.name = "sink2"} : <i8>
    %trueResult_15, %falseResult_16 = cond_br %59#1, %54 {handshake.bb = 4 : ui32, handshake.name = "cond_br6"} : <i1>, <f32>
    %trueResult_17, %falseResult_18 = cond_br %59#2, %result_11 {handshake.bb = 4 : ui32, handshake.name = "cond_br7"} : <i1>, <>
    %60 = merge %falseResult_16 {handshake.bb = 5 : ui32, handshake.name = "merge4"} : <f32>
    %result_19, %index_20 = control_merge [%falseResult_18]  {handshake.bb = 5 : ui32, handshake.name = "control_merge4"} : [<>] to <>, <i1>
    sink %index_20 {handshake.name = "sink3"} : <i1>
    %61:2 = fork [2] %result_19 {handshake.bb = 5 : ui32, handshake.name = "fork12"} : <>
    end {handshake.bb = 5 : ui32, handshake.name = "end0"} %60, %memEnd_0, %memEnd, %0#1 : <f32>, <>, <>, <>
  }
}

