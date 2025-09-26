module {
  handshake.func @if_float2(%arg0: !handshake.channel<f32>, %arg1: memref<100xf32>, %arg2: memref<100xf32>, %arg3: !handshake.control<>, %arg4: !handshake.control<>, %arg5: !handshake.control<>, ...) -> (!handshake.channel<f32>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["x0", "a", "minus_trace", "a_start", "minus_trace_start", "start"], resNames = ["out0", "a_end", "minus_trace_end", "end"]} {
    %0:3 = fork [3] %arg5 {handshake.bb = 0 : ui32, handshake.name = "fork0"} : <>
    %memEnd = mem_controller[%arg2 : memref<100xf32>] %arg4 (%49, %addressResult_5, %dataResult_6) %84#1 {connectedBlocks = [1 : i32], handshake.name = "mem_controller2"} :    (!handshake.channel<i32>, !handshake.channel<i7>, !handshake.channel<f32>) -> ()
    %outputs, %memEnd_0 = mem_controller[%arg1 : memref<100xf32>] %arg3 (%addressResult) %84#0 {connectedBlocks = [1 : i32], handshake.name = "mem_controller3"} :    (!handshake.channel<i7>) -> !handshake.channel<f32>
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
    %21:7 = fork [7] %20 {handshake.bb = 1 : ui32, handshake.name = "fork46"} : <i1>
    %22 = not %21#6 {handshake.bb = 1 : ui32, handshake.name = "not0"} : <i1>
    %23:2 = fork [2] %22 {handshake.bb = 1 : ui32, handshake.name = "fork47"} : <i1>
    %24 = spec_v2_repeating_init %23#1 {handshake.bb = 1 : ui32, handshake.name = "spec_v2_repeating_init0", initToken = 1 : ui1} : <i1>
    %25 = not %24 {handshake.bb = 1 : ui32, handshake.name = "not1"} : <i1>
    %26:4 = fork [4] %25 {handshake.bb = 1 : ui32, handshake.name = "fork48"} : <i1>
    %27 = source {handshake.bb = 1 : ui32, handshake.name = "source7"} : <>
    %28 = constant %27 {handshake.bb = 1 : ui32, handshake.name = "constant0", value = true} : <>, <i1>
    %29 = mux %26#0 [%23#0, %28] {handshake.bb = 1 : ui32, handshake.name = "mux5"} : <i1>, [<i1>, <i1>] to <i1>
    %30:6 = fork [6] %29 {handshake.bb = 1 : ui32, handshake.name = "fork49"} : <i1>
    %31 = passer %63[%30#0] {handshake.bb = 1 : ui32, handshake.name = "passer6"} : <>, <i1>
    %32 = merge %7#1 {handshake.bb = 1 : ui32, handshake.name = "merge0"} : <i8>
    %33 = merge %10#2 {handshake.bb = 1 : ui32, handshake.name = "merge1"} : <f32>
    %result_1, %index_2 = control_merge [%11#0]  {handshake.bb = 1 : ui32, handshake.name = "control_merge1"} : [<>] to <>, <i1>
    sink %index_2 {handshake.name = "sink4"} : <i1>
    %34 = source {handshake.bb = 1 : ui32, handshake.name = "source2"} : <>
    %35 = constant %34 {handshake.bb = 1 : ui32, handshake.name = "constant9", value = 3.000000e+00 : f32} : <>, <f32>
    %36 = addf %33, %35 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "addf1"} : <f32>
    %37 = br %36 {handshake.bb = 1 : ui32, handshake.name = "br6"} : <f32>
    %38 = br %32 {handshake.bb = 1 : ui32, handshake.name = "br7"} : <i8>
    %39 = br %result_1 {handshake.bb = 1 : ui32, handshake.name = "br8"} : <>
    %40:2 = fork [2] %41 {handshake.bb = 1 : ui32, handshake.name = "fork41"} : <i8>
    %41 = merge %7#2 {handshake.bb = 1 : ui32, handshake.name = "merge2"} : <i8>
    %42 = passer %43[%21#5] {handshake.bb = 1 : ui32, handshake.name = "passer7"} : <i7>, <i1>
    %43 = trunci %40#0 {handshake.bb = 1 : ui32, handshake.name = "trunci1"} : <i8> to <i7>
    %44 = passer %45#0[%21#3] {handshake.bb = 1 : ui32, handshake.name = "passer8"} : <f32>, <i1>
    %45:2 = fork [2] %46 {handshake.bb = 1 : ui32, handshake.name = "fork42"} : <f32>
    %46 = merge %10#3 {handshake.bb = 1 : ui32, handshake.name = "merge3"} : <f32>
    %47:2 = fork [2] %result_3 {handshake.bb = 1 : ui32, handshake.name = "fork43"} : <>
    %result_3, %index_4 = control_merge [%11#1]  {handshake.bb = 1 : ui32, handshake.name = "control_merge2"} : [<>] to <>, <i1>
    sink %index_4 {handshake.name = "sink5"} : <i1>
    %48 = constant %47#0 {handshake.bb = 1 : ui32, handshake.name = "constant3", value = 1 : i2} : <>, <i2>
    %49 = passer %50[%21#1] {handshake.bb = 1 : ui32, handshake.name = "passer9"} : <i32>, <i1>
    %50 = extsi %48 {handshake.bb = 1 : ui32, handshake.name = "extsi1"} : <i2> to <i32>
    %51 = source {handshake.bb = 1 : ui32, handshake.name = "source3"} : <>
    %52 = constant %51 {handshake.bb = 1 : ui32, handshake.name = "constant10", value = 1.000000e+00 : f32} : <>, <f32>
    %addressResult_5, %dataResult_6 = store[%42] %44 {handshake.bb = 1 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store0"} : <i7>, <f32>, <i7>, <f32>
    %53 = addf %45#1, %52 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "addf2"} : <f32>
    %54 = passer %55[%21#2] {handshake.bb = 1 : ui32, handshake.name = "passer10"} : <f32>, <i1>
    %55 = br %53 {handshake.bb = 1 : ui32, handshake.name = "br9"} : <f32>
    %56 = passer %57[%21#4] {handshake.bb = 1 : ui32, handshake.name = "passer11"} : <i8>, <i1>
    %57 = br %40#1 {handshake.bb = 1 : ui32, handshake.name = "br10"} : <i8>
    %58 = passer %59[%21#0] {handshake.bb = 1 : ui32, handshake.name = "passer12"} : <>, <i1>
    %59 = br %47#1 {handshake.bb = 1 : ui32, handshake.name = "br11"} : <>
    %60 = mux %26#1 [%37, %54] {handshake.bb = 1 : ui32, handshake.name = "mux2"} : <i1>, [<f32>, <f32>] to <f32>
    %61 = mux %26#2 [%38, %56] {handshake.bb = 1 : ui32, handshake.name = "mux3"} : <i1>, [<i8>, <i8>] to <i8>
    %62 = extsi %61 {handshake.bb = 1 : ui32, handshake.name = "extsi5"} : <i8> to <i9>
    %63 = mux %26#3 [%39, %58] {handshake.bb = 1 : ui32, handshake.name = "mux4"} : <i1>, [<>, <>] to <>
    %64 = source {handshake.bb = 1 : ui32, handshake.name = "source4"} : <>
    %65 = constant %64 {handshake.bb = 1 : ui32, handshake.name = "constant11", value = 1.000000e+00 : f32} : <>, <f32>
    %66 = source {handshake.bb = 1 : ui32, handshake.name = "source5"} : <>
    %67 = constant %66 {handshake.bb = 1 : ui32, handshake.name = "constant4", value = 1 : i2} : <>, <i2>
    %68 = extsi %67 {handshake.bb = 1 : ui32, handshake.name = "extsi6"} : <i2> to <i9>
    %69 = source {handshake.bb = 1 : ui32, handshake.name = "source6"} : <>
    %70 = constant %69 {handshake.bb = 1 : ui32, handshake.name = "constant5", value = 100 : i8} : <>, <i8>
    %71 = extsi %70 {handshake.bb = 1 : ui32, handshake.name = "extsi7"} : <i8> to <i9>
    %72 = passer %73[%30#5] {handshake.bb = 1 : ui32, handshake.name = "passer13"} : <f32>, <i1>
    %73 = divf %65, %60 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "divf0"} : <f32>
    %74:2 = fork [2] %75 {handshake.bb = 1 : ui32, handshake.name = "fork44"} : <i9>
    %75 = addi %62, %68 {handshake.bb = 1 : ui32, handshake.name = "addi0"} : <i9>
    %76 = passer %77[%30#4] {handshake.bb = 1 : ui32, handshake.name = "passer14"} : <i8>, <i1>
    %77 = trunci %74#0 {handshake.bb = 1 : ui32, handshake.name = "trunci2"} : <i9> to <i8>
    %78 = passer %81#0[%30#3] {handshake.bb = 1 : ui32, handshake.name = "passer15"} : <i1>, <i1>
    %79 = passer %81#1[%30#2] {handshake.bb = 1 : ui32, handshake.name = "passer16"} : <i1>, <i1>
    %80 = passer %81#2[%30#1] {handshake.bb = 1 : ui32, handshake.name = "passer17"} : <i1>, <i1>
    %81:3 = fork [3] %82 {handshake.bb = 1 : ui32, handshake.name = "fork45"} : <i1>
    %82 = cmpi ult, %74#1, %71 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i9>
    %trueResult, %falseResult = cond_br %78, %76 {handshake.bb = 1 : ui32, handshake.name = "cond_br5"} : <i1>, <i8>
    sink %falseResult {handshake.name = "sink2"} : <i8>
    %trueResult_7, %falseResult_8 = cond_br %79, %72 {handshake.bb = 1 : ui32, handshake.name = "cond_br6"} : <i1>, <f32>
    %trueResult_9, %falseResult_10 = cond_br %80, %31 {handshake.bb = 1 : ui32, handshake.name = "cond_br7"} : <i1>, <>
    %83 = merge %falseResult_8 {handshake.bb = 2 : ui32, handshake.name = "merge4"} : <f32>
    %result_11, %index_12 = control_merge [%falseResult_10]  {handshake.bb = 2 : ui32, handshake.name = "control_merge4"} : [<>] to <>, <i1>
    sink %index_12 {handshake.name = "sink3"} : <i1>
    %84:2 = fork [2] %result_11 {handshake.bb = 2 : ui32, handshake.name = "fork11"} : <>
    end {handshake.bb = 2 : ui32, handshake.name = "end0"} %83, %memEnd_0, %memEnd, %0#1 : <f32>, <>, <>, <>
  }
}

