module {
  handshake.func @if_float2(%arg0: !handshake.channel<f32>, %arg1: memref<100xf32>, %arg2: memref<100xf32>, %arg3: !handshake.control<>, %arg4: !handshake.control<>, %arg5: !handshake.control<>, ...) -> (!handshake.channel<f32>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["x0", "a", "minus_trace", "a_start", "minus_trace_start", "start"], resNames = ["out0", "a_end", "minus_trace_end", "end"]} {
    %0:3 = fork [3] %arg5 {handshake.bb = 0 : ui32, handshake.name = "fork0"} : <>
    %memEnd = mem_controller[%arg2 : memref<100xf32>] %arg4 (%47, %addressResult_5, %dataResult_6) %79#1 {connectedBlocks = [1 : i32], handshake.name = "mem_controller2"} :    (!handshake.channel<i32>, !handshake.channel<i7>, !handshake.channel<f32>) -> ()
    %outputs, %memEnd_0 = mem_controller[%arg1 : memref<100xf32>] %arg3 (%addressResult) %79#0 {connectedBlocks = [1 : i32], handshake.name = "mem_controller3"} :    (!handshake.channel<i7>) -> !handshake.channel<f32>
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
    %21:5 = fork [5] %20 {handshake.bb = 1 : ui32, handshake.name = "fork46"} : <i1>
    %22 = buffer %25#4, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer0"} : <i1>
    %23 = init %22 {handshake.bb = 1 : ui32, handshake.name = "init0", initToken = 0 : ui1} : <i1>
    %24 = not %23 {handshake.bb = 1 : ui32, handshake.name = "not1"} : <i1>
    %25:5 = fork [5] %24 {handshake.bb = 1 : ui32, handshake.name = "fork47"} : <i1>
    %26 = mux %25#3 [%29, %21#0] {handshake.bb = 1 : ui32, handshake.name = "mux5"} : <i1>, [<i1>, <i1>] to <i1>
    %27:6 = fork [6] %26 {handshake.bb = 1 : ui32, handshake.name = "fork48"} : <i1>
    %28 = passer %58[%27#5] {handshake.bb = 1 : ui32, handshake.name = "passer6"} : <>, <i1>
    %29 = not %21#4 {handshake.bb = 1 : ui32, handshake.name = "not0"} : <i1>
    %30 = merge %7#1 {handshake.bb = 1 : ui32, handshake.name = "merge0"} : <i8>
    %31 = merge %10#2 {handshake.bb = 1 : ui32, handshake.name = "merge1"} : <f32>
    %result_1, %index_2 = control_merge [%11#0]  {handshake.bb = 1 : ui32, handshake.name = "control_merge1"} : [<>] to <>, <i1>
    sink %index_2 {handshake.name = "sink4"} : <i1>
    %32 = source {handshake.bb = 1 : ui32, handshake.name = "source2"} : <>
    %33 = constant %32 {handshake.bb = 1 : ui32, handshake.name = "constant9", value = 3.000000e+00 : f32} : <>, <f32>
    %34 = addf %31, %33 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "addf1"} : <f32>
    %35 = br %34 {handshake.bb = 1 : ui32, handshake.name = "br6"} : <f32>
    %36 = br %30 {handshake.bb = 1 : ui32, handshake.name = "br7"} : <i8>
    %37 = br %result_1 {handshake.bb = 1 : ui32, handshake.name = "br8"} : <>
    %38:2 = fork [2] %39 {handshake.bb = 1 : ui32, handshake.name = "fork41"} : <i8>
    %39 = merge %7#2 {handshake.bb = 1 : ui32, handshake.name = "merge2"} : <i8>
    %40 = passer %41[%21#3] {handshake.bb = 1 : ui32, handshake.name = "passer7"} : <i7>, <i1>
    %41 = trunci %38#0 {handshake.bb = 1 : ui32, handshake.name = "trunci1"} : <i8> to <i7>
    %42 = passer %43#0[%21#2] {handshake.bb = 1 : ui32, handshake.name = "passer8"} : <f32>, <i1>
    %43:2 = fork [2] %44 {handshake.bb = 1 : ui32, handshake.name = "fork42"} : <f32>
    %44 = merge %10#3 {handshake.bb = 1 : ui32, handshake.name = "merge3"} : <f32>
    %45:2 = fork [2] %result_3 {handshake.bb = 1 : ui32, handshake.name = "fork43"} : <>
    %result_3, %index_4 = control_merge [%11#1]  {handshake.bb = 1 : ui32, handshake.name = "control_merge2"} : [<>] to <>, <i1>
    sink %index_4 {handshake.name = "sink5"} : <i1>
    %46 = constant %45#0 {handshake.bb = 1 : ui32, handshake.name = "constant3", value = 1 : i2} : <>, <i2>
    %47 = passer %48[%21#1] {handshake.bb = 1 : ui32, handshake.name = "passer9"} : <i32>, <i1>
    %48 = extsi %46 {handshake.bb = 1 : ui32, handshake.name = "extsi1"} : <i2> to <i32>
    %49 = source {handshake.bb = 1 : ui32, handshake.name = "source3"} : <>
    %50 = constant %49 {handshake.bb = 1 : ui32, handshake.name = "constant10", value = 1.000000e+00 : f32} : <>, <f32>
    %addressResult_5, %dataResult_6 = store[%40] %42 {handshake.bb = 1 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store0"} : <i7>, <f32>, <i7>, <f32>
    %51 = addf %43#1, %50 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "addf2"} : <f32>
    %52 = br %51 {handshake.bb = 1 : ui32, handshake.name = "br9"} : <f32>
    %53 = br %38#1 {handshake.bb = 1 : ui32, handshake.name = "br10"} : <i8>
    %54 = br %45#1 {handshake.bb = 1 : ui32, handshake.name = "br11"} : <>
    %55 = mux %25#0 [%35, %52] {handshake.bb = 1 : ui32, handshake.name = "mux2"} : <i1>, [<f32>, <f32>] to <f32>
    %56 = mux %25#1 [%36, %53] {handshake.bb = 1 : ui32, handshake.name = "mux3"} : <i1>, [<i8>, <i8>] to <i8>
    %57 = extsi %56 {handshake.bb = 1 : ui32, handshake.name = "extsi5"} : <i8> to <i9>
    %58 = mux %25#2 [%37, %54] {handshake.bb = 1 : ui32, handshake.name = "mux4"} : <i1>, [<>, <>] to <>
    %59 = source {handshake.bb = 1 : ui32, handshake.name = "source4"} : <>
    %60 = constant %59 {handshake.bb = 1 : ui32, handshake.name = "constant11", value = 1.000000e+00 : f32} : <>, <f32>
    %61 = source {handshake.bb = 1 : ui32, handshake.name = "source5"} : <>
    %62 = constant %61 {handshake.bb = 1 : ui32, handshake.name = "constant4", value = 1 : i2} : <>, <i2>
    %63 = extsi %62 {handshake.bb = 1 : ui32, handshake.name = "extsi6"} : <i2> to <i9>
    %64 = source {handshake.bb = 1 : ui32, handshake.name = "source6"} : <>
    %65 = constant %64 {handshake.bb = 1 : ui32, handshake.name = "constant5", value = 100 : i8} : <>, <i8>
    %66 = extsi %65 {handshake.bb = 1 : ui32, handshake.name = "extsi7"} : <i8> to <i9>
    %67 = passer %68[%27#0] {handshake.bb = 1 : ui32, handshake.name = "passer10"} : <f32>, <i1>
    %68 = divf %60, %55 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "divf0"} : <f32>
    %69:2 = fork [2] %70 {handshake.bb = 1 : ui32, handshake.name = "fork44"} : <i9>
    %70 = addi %57, %63 {handshake.bb = 1 : ui32, handshake.name = "addi0"} : <i9>
    %71 = passer %72[%27#4] {handshake.bb = 1 : ui32, handshake.name = "passer11"} : <i8>, <i1>
    %72 = trunci %69#0 {handshake.bb = 1 : ui32, handshake.name = "trunci2"} : <i9> to <i8>
    %73 = passer %76#0[%27#3] {handshake.bb = 1 : ui32, handshake.name = "passer12"} : <i1>, <i1>
    %74 = passer %76#1[%27#2] {handshake.bb = 1 : ui32, handshake.name = "passer13"} : <i1>, <i1>
    %75 = passer %76#2[%27#1] {handshake.bb = 1 : ui32, handshake.name = "passer14"} : <i1>, <i1>
    %76:3 = fork [3] %77 {handshake.bb = 1 : ui32, handshake.name = "fork45"} : <i1>
    %77 = cmpi ult, %69#1, %66 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i9>
    %trueResult, %falseResult = cond_br %73, %71 {handshake.bb = 1 : ui32, handshake.name = "cond_br5"} : <i1>, <i8>
    sink %falseResult {handshake.name = "sink2"} : <i8>
    %trueResult_7, %falseResult_8 = cond_br %74, %67 {handshake.bb = 1 : ui32, handshake.name = "cond_br6"} : <i1>, <f32>
    %trueResult_9, %falseResult_10 = cond_br %75, %28 {handshake.bb = 1 : ui32, handshake.name = "cond_br7"} : <i1>, <>
    %78 = merge %falseResult_8 {handshake.bb = 2 : ui32, handshake.name = "merge4"} : <f32>
    %result_11, %index_12 = control_merge [%falseResult_10]  {handshake.bb = 2 : ui32, handshake.name = "control_merge4"} : [<>] to <>, <i1>
    sink %index_12 {handshake.name = "sink3"} : <i1>
    %79:2 = fork [2] %result_11 {handshake.bb = 2 : ui32, handshake.name = "fork11"} : <>
    end {handshake.bb = 2 : ui32, handshake.name = "end0"} %78, %memEnd_0, %memEnd, %0#1 : <f32>, <>, <>, <>
  }
}

