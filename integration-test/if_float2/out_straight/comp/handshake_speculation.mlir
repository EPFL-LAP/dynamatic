module {
  handshake.func @if_float2(%arg0: !handshake.channel<f32>, %arg1: memref<100xf32>, %arg2: memref<100xf32>, %arg3: !handshake.control<>, %arg4: !handshake.control<>, %arg5: !handshake.control<>, ...) -> (!handshake.channel<f32>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["x0", "a", "minus_trace", "a_start", "minus_trace_start", "start"], resNames = ["out0", "a_end", "minus_trace_end", "end"]} {
    %0:3 = fork [3] %arg5 {handshake.bb = 0 : ui32, handshake.name = "fork0"} : <>
    %memEnd = mem_controller[%arg2 : memref<100xf32>] %arg4 (%55, %addressResult_5, %dataResult_6) %82#1 {connectedBlocks = [1 : i32], handshake.name = "mem_controller2"} :    (!handshake.channel<i32>, !handshake.channel<i7>, !handshake.channel<f32>) -> ()
    %outputs, %memEnd_0 = mem_controller[%arg1 : memref<100xf32>] %arg3 (%addressResult) %82#0 {connectedBlocks = [1 : i32], handshake.name = "mem_controller3"} :    (!handshake.channel<i7>) -> !handshake.channel<f32>
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
    %21:7 = fork [7] %20 {handshake.bb = 1 : ui32, handshake.name = "fork37"} : <i1>
    %22 = spec_v2_repeating_init %21#1 {handshake.bb = 1 : ui32, handshake.name = "spec_v2_repeating_init0", initToken = 1 : ui1} : <i1>
    %23:4 = fork [4] %22 {handshake.bb = 1 : ui32, handshake.name = "fork38"} : <i1>
    %24 = source {handshake.bb = 1 : ui32, handshake.name = "source7"} : <>
    %25 = constant %24 {handshake.bb = 1 : ui32, handshake.name = "constant0", value = true} : <>, <i1>
    %26 = mux %23#0 [%25, %21#0] {handshake.bb = 1 : ui32, handshake.name = "mux5"} : <i1>, [<i1>, <i1>] to <i1>
    %27:3 = fork [3] %26 {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <i1>
    %28 = passer %63[%27#2] {handshake.bb = 1 : ui32, handshake.name = "passer6"} : <f32>, <i1>
    %29 = passer %64[%27#1] {handshake.bb = 1 : ui32, handshake.name = "passer7"} : <i8>, <i1>
    %30 = passer %66[%27#0] {handshake.bb = 1 : ui32, handshake.name = "passer8"} : <>, <i1>
    %31 = not %21#6 {handshake.bb = 1 : ui32, handshake.name = "not0"} : <i1>
    %32:4 = fork [4] %31 {handshake.bb = 1 : ui32, handshake.name = "fork39"} : <i1>
    %33 = merge %7#1 {handshake.bb = 1 : ui32, handshake.name = "merge0"} : <i8>
    %34 = merge %10#2 {handshake.bb = 1 : ui32, handshake.name = "merge1"} : <f32>
    %35 = passer %index_2[%32#0] {handshake.bb = 1 : ui32, handshake.name = "passer9"} : <i1>, <i1>
    %result_1, %index_2 = control_merge [%11#0]  {handshake.bb = 1 : ui32, handshake.name = "control_merge1"} : [<>] to <>, <i1>
    sink %35 {handshake.name = "sink0"} : <i1>
    %36 = source {handshake.bb = 1 : ui32, handshake.name = "source2"} : <>
    %37 = constant %36 {handshake.bb = 1 : ui32, handshake.name = "constant9", value = 3.000000e+00 : f32} : <>, <f32>
    %38 = addf %34, %37 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "addf1"} : <f32>
    %39 = passer %40[%32#2] {handshake.bb = 1 : ui32, handshake.name = "passer10"} : <f32>, <i1>
    %40 = br %38 {handshake.bb = 1 : ui32, handshake.name = "br6"} : <f32>
    %41 = passer %42[%32#3] {handshake.bb = 1 : ui32, handshake.name = "passer11"} : <i8>, <i1>
    %42 = br %33 {handshake.bb = 1 : ui32, handshake.name = "br7"} : <i8>
    %43 = passer %44[%32#1] {handshake.bb = 1 : ui32, handshake.name = "passer12"} : <>, <i1>
    %44 = br %result_1 {handshake.bb = 1 : ui32, handshake.name = "br8"} : <>
    %45:2 = fork [2] %46 {handshake.bb = 1 : ui32, handshake.name = "fork34"} : <i8>
    %46 = merge %7#2 {handshake.bb = 1 : ui32, handshake.name = "merge2"} : <i8>
    %47 = passer %48[%21#5] {handshake.bb = 1 : ui32, handshake.name = "passer13"} : <i7>, <i1>
    %48 = trunci %45#0 {handshake.bb = 1 : ui32, handshake.name = "trunci1"} : <i8> to <i7>
    %49 = passer %50#0[%21#4] {handshake.bb = 1 : ui32, handshake.name = "passer14"} : <f32>, <i1>
    %50:2 = fork [2] %51 {handshake.bb = 1 : ui32, handshake.name = "fork35"} : <f32>
    %51 = merge %10#3 {handshake.bb = 1 : ui32, handshake.name = "merge3"} : <f32>
    %52:2 = fork [2] %result_3 {handshake.bb = 1 : ui32, handshake.name = "fork36"} : <>
    %53 = passer %index_4[%21#2] {handshake.bb = 1 : ui32, handshake.name = "passer15"} : <i1>, <i1>
    %result_3, %index_4 = control_merge [%11#1]  {handshake.bb = 1 : ui32, handshake.name = "control_merge2"} : [<>] to <>, <i1>
    sink %53 {handshake.name = "sink1"} : <i1>
    %54 = constant %52#0 {handshake.bb = 1 : ui32, handshake.name = "constant3", value = 1 : i2} : <>, <i2>
    %55 = passer %56[%21#3] {handshake.bb = 1 : ui32, handshake.name = "passer16"} : <i32>, <i1>
    %56 = extsi %54 {handshake.bb = 1 : ui32, handshake.name = "extsi1"} : <i2> to <i32>
    %57 = source {handshake.bb = 1 : ui32, handshake.name = "source3"} : <>
    %58 = constant %57 {handshake.bb = 1 : ui32, handshake.name = "constant10", value = 1.000000e+00 : f32} : <>, <f32>
    %addressResult_5, %dataResult_6 = store[%47] %49 {handshake.bb = 1 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store0"} : <i7>, <f32>, <i7>, <f32>
    %59 = addf %50#1, %58 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "addf2"} : <f32>
    %60 = br %59 {handshake.bb = 1 : ui32, handshake.name = "br9"} : <f32>
    %61 = br %45#1 {handshake.bb = 1 : ui32, handshake.name = "br10"} : <i8>
    %62 = br %52#1 {handshake.bb = 1 : ui32, handshake.name = "br11"} : <>
    %63 = mux %23#1 [%39, %60] {handshake.bb = 1 : ui32, handshake.name = "mux2"} : <i1>, [<f32>, <f32>] to <f32>
    %64 = mux %23#2 [%41, %61] {handshake.bb = 1 : ui32, handshake.name = "mux3"} : <i1>, [<i8>, <i8>] to <i8>
    %65 = extsi %29 {handshake.bb = 1 : ui32, handshake.name = "extsi5"} : <i8> to <i9>
    %66 = mux %23#3 [%43, %62] {handshake.bb = 1 : ui32, handshake.name = "mux4"} : <i1>, [<>, <>] to <>
    %67 = source {handshake.bb = 1 : ui32, handshake.name = "source4"} : <>
    %68 = constant %67 {handshake.bb = 1 : ui32, handshake.name = "constant11", value = 1.000000e+00 : f32} : <>, <f32>
    %69 = source {handshake.bb = 1 : ui32, handshake.name = "source5"} : <>
    %70 = constant %69 {handshake.bb = 1 : ui32, handshake.name = "constant4", value = 1 : i2} : <>, <i2>
    %71 = extsi %70 {handshake.bb = 1 : ui32, handshake.name = "extsi6"} : <i2> to <i9>
    %72 = source {handshake.bb = 1 : ui32, handshake.name = "source6"} : <>
    %73 = constant %72 {handshake.bb = 1 : ui32, handshake.name = "constant5", value = 100 : i8} : <>, <i8>
    %74 = extsi %73 {handshake.bb = 1 : ui32, handshake.name = "extsi7"} : <i8> to <i9>
    %75 = divf %68, %28 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "divf0"} : <f32>
    %76 = addi %65, %71 {handshake.bb = 1 : ui32, handshake.name = "addi0"} : <i9>
    %77:2 = fork [2] %76 {handshake.bb = 1 : ui32, handshake.name = "fork9"} : <i9>
    %78 = trunci %77#0 {handshake.bb = 1 : ui32, handshake.name = "trunci2"} : <i9> to <i8>
    %79 = cmpi ult, %77#1, %74 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i9>
    %80:3 = fork [3] %79 {handshake.bb = 1 : ui32, handshake.name = "fork10"} : <i1>
    %trueResult, %falseResult = cond_br %80#0, %78 {handshake.bb = 1 : ui32, handshake.name = "cond_br5"} : <i1>, <i8>
    sink %falseResult {handshake.name = "sink2"} : <i8>
    %trueResult_7, %falseResult_8 = cond_br %80#1, %75 {handshake.bb = 1 : ui32, handshake.name = "cond_br6"} : <i1>, <f32>
    %trueResult_9, %falseResult_10 = cond_br %80#2, %30 {handshake.bb = 1 : ui32, handshake.name = "cond_br7"} : <i1>, <>
    %81 = merge %falseResult_8 {handshake.bb = 2 : ui32, handshake.name = "merge4"} : <f32>
    %result_11, %index_12 = control_merge [%falseResult_10]  {handshake.bb = 2 : ui32, handshake.name = "control_merge4"} : [<>] to <>, <i1>
    sink %index_12 {handshake.name = "sink3"} : <i1>
    %82:2 = fork [2] %result_11 {handshake.bb = 2 : ui32, handshake.name = "fork11"} : <>
    end {handshake.bb = 2 : ui32, handshake.name = "end0"} %81, %memEnd_0, %memEnd, %0#1 : <f32>, <>, <>, <>
  }
}

