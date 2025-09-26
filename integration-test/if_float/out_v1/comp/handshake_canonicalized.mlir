module {
  handshake.func @if_float(%arg0: !handshake.channel<f32>, %arg1: memref<100xf32>, %arg2: memref<100xf32>, %arg3: !handshake.control<>, %arg4: !handshake.control<>, %arg5: !handshake.control<>, ...) -> (!handshake.channel<f32>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["x0", "a", "minus_trace", "a_start", "minus_trace_start", "start"], resNames = ["out0", "a_end", "minus_trace_end", "end"]} {
    %0:3 = fork [3] %arg5 {handshake.bb = 0 : ui32, handshake.name = "fork0"} : <>
    %memEnd = mem_controller[%arg2 : memref<100xf32>] %arg4 (%26, %addressResult_5, %dataResult_6) %47#1 {connectedBlocks = [3 : i32], handshake.name = "mem_controller2"} :    (!handshake.channel<i32>, !handshake.channel<i7>, !handshake.channel<f32>) -> ()
    %outputs, %memEnd_0 = mem_controller[%arg1 : memref<100xf32>] %arg3 (%addressResult) %47#0 {connectedBlocks = [1 : i32], handshake.name = "mem_controller3"} :    (!handshake.channel<i7>) -> !handshake.channel<f32>
    %1 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant2", value = false} : <>, <i1>
    %2 = extsi %1 {handshake.bb = 0 : ui32, handshake.name = "extsi4"} : <i1> to <i8>
    %3 = mux %8#0 [%2, %trueResult_9] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i8>, <i8>] to <i8>
    %4:2 = fork [2] %3 {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <i8>
    %5 = trunci %4#0 {handshake.bb = 1 : ui32, handshake.name = "trunci0"} : <i8> to <i7>
    %6 = mux %8#1 [%arg0, %trueResult_11] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i1>, [<f32>, <f32>] to <f32>
    %7:3 = fork [3] %6 {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <f32>
    %result, %index = control_merge [%0#2, %trueResult_13]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %8:2 = fork [2] %index {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <i1>
    %9 = source {handshake.bb = 1 : ui32, handshake.name = "source0"} : <>
    %10 = constant %9 {handshake.bb = 1 : ui32, handshake.name = "constant6", value = -0.899999976 : f32} : <>, <f32>
    %11 = source {handshake.bb = 1 : ui32, handshake.name = "source1"} : <>
    %12 = constant %11 {handshake.bb = 1 : ui32, handshake.name = "constant7", value = 0.000000e+00 : f32} : <>, <f32>
    %addressResult, %dataResult = load[%5] %outputs {handshake.bb = 1 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i7>, <f32>, <i7>, <f32>
    %13 = mulf %dataResult, %7#2 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "mulf0"} : <f32>
    %14 = mulf %7#1, %10 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "mulf1"} : <f32>
    %15 = addf %13, %14 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "addf0"} : <f32>
    %16 = cmpf ugt, %15, %12 {handshake.bb = 1 : ui32, handshake.name = "cmpf0"} : <f32>
    %17:3 = fork [3] %16 {handshake.bb = 1 : ui32, handshake.name = "fork4"} : <i1>
    %trueResult, %falseResult = cond_br %17#0, %4#1 {handshake.bb = 1 : ui32, handshake.name = "cond_br2"} : <i1>, <i8>
    %trueResult_1, %falseResult_2 = cond_br %17#2, %7#0 {handshake.bb = 1 : ui32, handshake.name = "cond_br3"} : <i1>, <f32>
    %trueResult_3, %falseResult_4 = cond_br %17#1, %result {handshake.bb = 1 : ui32, handshake.name = "cond_br4"} : <i1>, <>
    %18 = source {handshake.bb = 2 : ui32, handshake.name = "source2"} : <>
    %19 = constant %18 {handshake.bb = 2 : ui32, handshake.name = "constant8", value = 1.100000e+00 : f32} : <>, <f32>
    %20 = mulf %falseResult_2, %19 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "mulf2"} : <f32>
    %21:2 = fork [2] %trueResult {handshake.bb = 3 : ui32, handshake.name = "fork5"} : <i8>
    %22 = trunci %21#0 {handshake.bb = 3 : ui32, handshake.name = "trunci1"} : <i8> to <i7>
    %23:2 = fork [2] %trueResult_1 {handshake.bb = 3 : ui32, handshake.name = "fork6"} : <f32>
    %24:2 = fork [2] %trueResult_3 {handshake.bb = 3 : ui32, handshake.name = "fork7"} : <>
    %25 = constant %24#0 {handshake.bb = 3 : ui32, handshake.name = "constant3", value = 1 : i2} : <>, <i2>
    %26 = extsi %25 {handshake.bb = 3 : ui32, handshake.name = "extsi1"} : <i2> to <i32>
    %27 = source {handshake.bb = 3 : ui32, handshake.name = "source3"} : <>
    %28 = constant %27 {handshake.bb = 3 : ui32, handshake.name = "constant9", value = 1.100000e+00 : f32} : <>, <f32>
    %addressResult_5, %dataResult_6 = store[%22] %23#0 {handshake.bb = 3 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store0"} : <i7>, <f32>, <i7>, <f32>
    %29 = divf %23#1, %28 {fastmath = #arith.fastmath<none>, handshake.bb = 3 : ui32, handshake.name = "divf0"} : <f32>
    %30 = mux %34#1 [%20, %29] {handshake.bb = 4 : ui32, handshake.name = "mux2"} : <i1>, [<f32>, <f32>] to <f32>
    %31:2 = fork [2] %30 {handshake.bb = 4 : ui32, handshake.name = "fork8"} : <f32>
    %32 = mux %34#0 [%falseResult, %21#1] {handshake.bb = 4 : ui32, handshake.name = "mux3"} : <i1>, [<i8>, <i8>] to <i8>
    %33 = extsi %32 {handshake.bb = 4 : ui32, handshake.name = "extsi5"} : <i8> to <i9>
    %result_7, %index_8 = control_merge [%falseResult_4, %24#1]  {handshake.bb = 4 : ui32, handshake.name = "control_merge3"} : [<>, <>] to <>, <i1>
    %34:2 = fork [2] %index_8 {handshake.bb = 4 : ui32, handshake.name = "fork9"} : <i1>
    %35 = source {handshake.bb = 4 : ui32, handshake.name = "source4"} : <>
    %36 = constant %35 {handshake.bb = 4 : ui32, handshake.name = "constant4", value = 1 : i2} : <>, <i2>
    %37 = extsi %36 {handshake.bb = 4 : ui32, handshake.name = "extsi6"} : <i2> to <i9>
    %38 = source {handshake.bb = 4 : ui32, handshake.name = "source5"} : <>
    %39 = constant %38 {handshake.bb = 4 : ui32, handshake.name = "constant12", value = 100 : i8} : <>, <i8>
    %40 = extsi %39 {handshake.bb = 4 : ui32, handshake.name = "extsi7"} : <i8> to <i9>
    %41 = addf %31#0, %31#1 {fastmath = #arith.fastmath<none>, handshake.bb = 4 : ui32, handshake.name = "addf1"} : <f32>
    %42 = addi %33, %37 {handshake.bb = 4 : ui32, handshake.name = "addi0"} : <i9>
    %43:2 = fork [2] %42 {handshake.bb = 4 : ui32, handshake.name = "fork10"} : <i9>
    %44 = trunci %43#0 {handshake.bb = 4 : ui32, handshake.name = "trunci2"} : <i9> to <i8>
    %45 = cmpi ult, %43#1, %40 {handshake.bb = 4 : ui32, handshake.name = "cmpi0"} : <i9>
    %46:3 = fork [3] %45 {handshake.bb = 4 : ui32, handshake.name = "fork11"} : <i1>
    %trueResult_9, %falseResult_10 = cond_br %46#0, %44 {handshake.bb = 4 : ui32, handshake.name = "cond_br5"} : <i1>, <i8>
    sink %falseResult_10 {handshake.name = "sink2"} : <i8>
    %trueResult_11, %falseResult_12 = cond_br %46#1, %41 {handshake.bb = 4 : ui32, handshake.name = "cond_br6"} : <i1>, <f32>
    %trueResult_13, %falseResult_14 = cond_br %46#2, %result_7 {handshake.bb = 4 : ui32, handshake.name = "cond_br7"} : <i1>, <>
    %47:2 = fork [2] %falseResult_14 {handshake.bb = 5 : ui32, handshake.name = "fork12"} : <>
    end {handshake.bb = 5 : ui32, handshake.name = "end0"} %falseResult_12, %memEnd_0, %memEnd, %0#1 : <f32>, <>, <>, <>
  }
}

