module {
  handshake.func @if_float2(%arg0: !handshake.channel<f32>, %arg1: memref<100xf32>, %arg2: memref<100xf32>, %arg3: !handshake.control<>, %arg4: !handshake.control<>, %arg5: !handshake.control<>, ...) -> (!handshake.channel<f32>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["x0", "a", "minus_trace", "a_start", "minus_trace_start", "start"], resNames = ["out0", "a_end", "minus_trace_end", "end"]} {
    %0:3 = fork [3] %arg5 {handshake.bb = 0 : ui32, handshake.name = "fork0"} : <>
    %memEnd = mem_controller[%arg2 : memref<100xf32>] %arg4 (%45, %addressResult_5, %dataResult_6) %71#1 {connectedBlocks = [3 : i32], handshake.name = "mem_controller2"} :    (!handshake.channel<i32>, !handshake.channel<i7>, !handshake.channel<f32>) -> ()
    %outputs, %memEnd_0 = mem_controller[%arg1 : memref<100xf32>] %arg3 (%addressResult) %71#0 {connectedBlocks = [1 : i32], handshake.name = "mem_controller3"} :    (!handshake.channel<i7>) -> !handshake.channel<f32>
    %1 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant2", value = false} : <>, <i1>
    %2 = br %1 {handshake.bb = 0 : ui32, handshake.name = "br3"} : <i1>
    %3 = extsi %2 {handshake.bb = 0 : ui32, handshake.name = "extsi4"} : <i1> to <i8>
    %4 = br %arg0 {handshake.bb = 0 : ui32, handshake.name = "br4"} : <f32>
    %5 = br %0#2 {handshake.bb = 0 : ui32, handshake.name = "br5"} : <>
    %6 = mux %12#0 [%3, %trueResult] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i8>, <i8>] to <i8>
    %7:3 = fork [3] %6 {handshake.bb = 1 : ui32, handshake.name = "fork12"} : <i8>
    %8 = trunci %7#0 {handshake.bb = 1 : ui32, handshake.name = "trunci0"} : <i8> to <i7>
    %9 = mux %12#1 [%4, %trueResult_7] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i1>, [<f32>, <f32>] to <f32>
    %10:4 = fork [4] %9 {handshake.bb = 1 : ui32, handshake.name = "fork13"} : <f32>
    %result, %index = control_merge [%5, %trueResult_9]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %11:2 = fork [2] %result {handshake.bb = 1 : ui32, handshake.name = "fork14"} : <>
    %12:2 = fork [2] %index {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <i1>
    %13 = source {handshake.bb = 1 : ui32, handshake.name = "source0"} : <>
    %14 = constant %13 {handshake.bb = 1 : ui32, handshake.name = "constant7", value = -0.899999976 : f32} : <>, <f32>
    %15 = source {handshake.bb = 1 : ui32, handshake.name = "source1"} : <>
    %16 = constant %15 {handshake.bb = 1 : ui32, handshake.name = "constant8", value = 0.000000e+00 : f32} : <>, <f32>
    %addressResult, %dataResult = load[%8] %outputs {handshake.bb = 1 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i7>, <f32>, <i7>, <f32>
    %17 = mulf %dataResult, %10#1 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "mulf0"} : <f32>
    %18 = mulf %10#0, %14 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "mulf1"} : <f32>
    %19 = addf %17, %18 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "addf0"} : <f32>
    %20 = cmpf ugt, %19, %16 {handshake.bb = 1 : ui32, handshake.name = "cmpf0"} : <f32>
    %21:7 = fork [7] %20 {handshake.bb = 1 : ui32, handshake.name = "fork15"} : <i1>
    %22 = not %21#4 {handshake.bb = 1 : ui32, handshake.name = "not0"} : <i1>
    %23:3 = fork [3] %22 {handshake.bb = 1 : ui32, handshake.name = "fork16"} : <i1>
    %24 = passer %7#2[%21#3] {handshake.bb = 1 : ui32, handshake.name = "passer0"} : <i8>, <i1>
    %25 = passer %7#1[%23#2] {handshake.bb = 1 : ui32, handshake.name = "passer1"} : <i8>, <i1>
    %26 = passer %10#3[%21#2] {handshake.bb = 1 : ui32, handshake.name = "passer2"} : <f32>, <i1>
    %27 = passer %10#2[%23#1] {handshake.bb = 1 : ui32, handshake.name = "passer3"} : <f32>, <i1>
    %28 = passer %11#1[%21#1] {handshake.bb = 1 : ui32, handshake.name = "passer4"} : <>, <i1>
    %29 = passer %11#0[%23#0] {handshake.bb = 1 : ui32, handshake.name = "passer5"} : <>, <i1>
    %30 = merge %25 {handshake.bb = 2 : ui32, handshake.name = "merge0"} : <i8>
    %31 = merge %27 {handshake.bb = 2 : ui32, handshake.name = "merge1"} : <f32>
    %result_1, %index_2 = control_merge [%29]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>] to <>, <i1>
    sink %index_2 {handshake.name = "sink0"} : <i1>
    %32 = source {handshake.bb = 2 : ui32, handshake.name = "source2"} : <>
    %33 = constant %32 {handshake.bb = 2 : ui32, handshake.name = "constant9", value = 3.000000e+00 : f32} : <>, <f32>
    %34 = addf %31, %33 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "addf1"} : <f32>
    %35 = br %34 {handshake.bb = 2 : ui32, handshake.name = "br6"} : <f32>
    %36 = br %30 {handshake.bb = 2 : ui32, handshake.name = "br7"} : <i8>
    %37 = br %result_1 {handshake.bb = 2 : ui32, handshake.name = "br8"} : <>
    %38 = merge %24 {handshake.bb = 3 : ui32, handshake.name = "merge2"} : <i8>
    %39:2 = fork [2] %38 {handshake.bb = 3 : ui32, handshake.name = "fork5"} : <i8>
    %40 = trunci %39#0 {handshake.bb = 3 : ui32, handshake.name = "trunci1"} : <i8> to <i7>
    %41 = merge %26 {handshake.bb = 3 : ui32, handshake.name = "merge3"} : <f32>
    %42:2 = fork [2] %41 {handshake.bb = 3 : ui32, handshake.name = "fork6"} : <f32>
    %result_3, %index_4 = control_merge [%28]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>] to <>, <i1>
    sink %index_4 {handshake.name = "sink1"} : <i1>
    %43:2 = fork [2] %result_3 {handshake.bb = 3 : ui32, handshake.name = "fork7"} : <>
    %44 = constant %43#0 {handshake.bb = 3 : ui32, handshake.name = "constant3", value = 1 : i2} : <>, <i2>
    %45 = extsi %44 {handshake.bb = 3 : ui32, handshake.name = "extsi1"} : <i2> to <i32>
    %46 = source {handshake.bb = 3 : ui32, handshake.name = "source3"} : <>
    %47 = constant %46 {handshake.bb = 3 : ui32, handshake.name = "constant10", value = 1.000000e+00 : f32} : <>, <f32>
    %addressResult_5, %dataResult_6 = store[%40] %42#0 {handshake.bb = 3 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store0"} : <i7>, <f32>, <i7>, <f32>
    %48 = addf %42#1, %47 {fastmath = #arith.fastmath<none>, handshake.bb = 3 : ui32, handshake.name = "addf2"} : <f32>
    %49 = br %48 {handshake.bb = 3 : ui32, handshake.name = "br9"} : <f32>
    %50 = br %39#1 {handshake.bb = 3 : ui32, handshake.name = "br10"} : <i8>
    %51 = br %43#1 {handshake.bb = 3 : ui32, handshake.name = "br11"} : <>
    %52 = mux %21#6 [%35, %49] {handshake.bb = 4 : ui32, handshake.name = "mux2"} : <i1>, [<f32>, <f32>] to <f32>
    %53 = mux %21#5 [%36, %50] {handshake.bb = 4 : ui32, handshake.name = "mux3"} : <i1>, [<i8>, <i8>] to <i8>
    %54 = extsi %53 {handshake.bb = 4 : ui32, handshake.name = "extsi5"} : <i8> to <i9>
    %55 = mux %21#0 [%37, %51] {handshake.bb = 4 : ui32, handshake.name = "mux4"} : <i1>, [<>, <>] to <>
    %56 = source {handshake.bb = 4 : ui32, handshake.name = "source4"} : <>
    %57 = constant %56 {handshake.bb = 4 : ui32, handshake.name = "constant11", value = 1.000000e+00 : f32} : <>, <f32>
    %58 = source {handshake.bb = 4 : ui32, handshake.name = "source5"} : <>
    %59 = constant %58 {handshake.bb = 4 : ui32, handshake.name = "constant4", value = 1 : i2} : <>, <i2>
    %60 = extsi %59 {handshake.bb = 4 : ui32, handshake.name = "extsi6"} : <i2> to <i9>
    %61 = source {handshake.bb = 4 : ui32, handshake.name = "source6"} : <>
    %62 = constant %61 {handshake.bb = 4 : ui32, handshake.name = "constant5", value = 100 : i8} : <>, <i8>
    %63 = extsi %62 {handshake.bb = 4 : ui32, handshake.name = "extsi7"} : <i8> to <i9>
    %64 = divf %57, %52 {fastmath = #arith.fastmath<none>, handshake.bb = 4 : ui32, handshake.name = "divf0"} : <f32>
    %65 = addi %54, %60 {handshake.bb = 4 : ui32, handshake.name = "addi0"} : <i9>
    %66:2 = fork [2] %65 {handshake.bb = 4 : ui32, handshake.name = "fork9"} : <i9>
    %67 = trunci %66#0 {handshake.bb = 4 : ui32, handshake.name = "trunci2"} : <i9> to <i8>
    %68 = cmpi ult, %66#1, %63 {handshake.bb = 4 : ui32, handshake.name = "cmpi0"} : <i9>
    %69:3 = fork [3] %68 {handshake.bb = 4 : ui32, handshake.name = "fork10"} : <i1>
    %trueResult, %falseResult = cond_br %69#0, %67 {handshake.bb = 4 : ui32, handshake.name = "cond_br5"} : <i1>, <i8>
    sink %falseResult {handshake.name = "sink2"} : <i8>
    %trueResult_7, %falseResult_8 = cond_br %69#1, %64 {handshake.bb = 4 : ui32, handshake.name = "cond_br6"} : <i1>, <f32>
    %trueResult_9, %falseResult_10 = cond_br %69#2, %55 {handshake.bb = 4 : ui32, handshake.name = "cond_br7"} : <i1>, <>
    %70 = merge %falseResult_8 {handshake.bb = 5 : ui32, handshake.name = "merge4"} : <f32>
    %result_11, %index_12 = control_merge [%falseResult_10]  {handshake.bb = 5 : ui32, handshake.name = "control_merge4"} : [<>] to <>, <i1>
    sink %index_12 {handshake.name = "sink3"} : <i1>
    %71:2 = fork [2] %result_11 {handshake.bb = 5 : ui32, handshake.name = "fork11"} : <>
    end {handshake.bb = 5 : ui32, handshake.name = "end0"} %70, %memEnd_0, %memEnd, %0#1 : <f32>, <>, <>, <>
  }
}

