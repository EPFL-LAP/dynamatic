module {
  handshake.func @if_float(%arg0: !handshake.channel<f32>, %arg1: memref<100xf32>, %arg2: memref<100xf32>, %arg3: !handshake.control<>, %arg4: !handshake.control<>, %arg5: !handshake.control<>, ...) -> (!handshake.channel<f32>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["x0", "a", "minus_trace", "a_start", "minus_trace_start", "start"], resNames = ["out0", "a_end", "minus_trace_end", "end"]} {
    %0:3 = fork [3] %arg5 {handshake.bb = 0 : ui32, handshake.name = "fork0"} : <>
    %memEnd = mem_controller[%arg2 : memref<100xf32>] %arg4 (%44, %addressResult_5, %dataResult_6) %73#1 {connectedBlocks = [1 : i32], handshake.name = "mem_controller2"} :    (!handshake.channel<i32>, !handshake.channel<i7>, !handshake.channel<f32>) -> ()
    %outputs, %memEnd_0 = mem_controller[%arg1 : memref<100xf32>] %arg3 (%addressResult) %73#0 {connectedBlocks = [1 : i32], handshake.name = "mem_controller3"} :    (!handshake.channel<i7>) -> !handshake.channel<f32>
    %1 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant2", value = false} : <>, <i1>
    %2 = br %1 {handshake.bb = 0 : ui32, handshake.name = "br3"} : <i1>
    %3 = extsi %2 {handshake.bb = 0 : ui32, handshake.name = "extsi4"} : <i1> to <i8>
    %4 = br %arg0 {handshake.bb = 0 : ui32, handshake.name = "br4"} : <f32>
    %5 = br %0#2 {handshake.bb = 0 : ui32, handshake.name = "br5"} : <>
    %6 = mux %12#0 [%3, %trueResult] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i8>, <i8>] to <i8>
    %7:3 = fork [3] %6 {handshake.bb = 1 : ui32, handshake.name = "fork14"} : <i8>
    %8 = trunci %7#0 {handshake.bb = 1 : ui32, handshake.name = "trunci0"} : <i8> to <i7>
    %9 = mux %12#1 [%4, %trueResult_7] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i1>, [<f32>, <f32>] to <f32>
    %10:4 = fork [4] %9 {handshake.bb = 1 : ui32, handshake.name = "fork15"} : <f32>
    %result, %index = control_merge [%5, %trueResult_9]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %11:2 = fork [2] %result {handshake.bb = 1 : ui32, handshake.name = "fork16"} : <>
    %12:2 = fork [2] %index {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <i1>
    %13 = source {handshake.bb = 1 : ui32, handshake.name = "source0"} : <>
    %14 = constant %13 {handshake.bb = 1 : ui32, handshake.name = "constant6", value = -0.899999976 : f32} : <>, <f32>
    %15 = source {handshake.bb = 1 : ui32, handshake.name = "source1"} : <>
    %16 = constant %15 {handshake.bb = 1 : ui32, handshake.name = "constant7", value = 0.000000e+00 : f32} : <>, <f32>
    %addressResult, %dataResult = load[%8] %outputs {handshake.bb = 1 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i7>, <f32>, <i7>, <f32>
    %17 = mulf %dataResult, %10#1 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "mulf0"} : <f32>
    %18 = mulf %10#0, %14 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "mulf1"} : <f32>
    %19 = addf %17, %18 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "addf0"} : <f32>
    %20 = cmpf ugt, %19, %16 {handshake.bb = 1 : ui32, handshake.name = "cmpf0"} : <f32>
    %21:10 = fork [10] %20 {handshake.bb = 1 : ui32, handshake.name = "fork37"} : <i1>
    %22 = not %21#9 {handshake.bb = 1 : ui32, handshake.name = "not0"} : <i1>
    %23:3 = fork [3] %22 {handshake.bb = 1 : ui32, handshake.name = "fork38"} : <i1>
    %24 = merge %7#1 {handshake.bb = 1 : ui32, handshake.name = "merge0"} : <i8>
    %25 = merge %10#2 {handshake.bb = 1 : ui32, handshake.name = "merge1"} : <f32>
    %result_1, %index_2 = control_merge [%11#0]  {handshake.bb = 1 : ui32, handshake.name = "control_merge1"} : [<>] to <>, <i1>
    sink %index_2 {handshake.name = "sink0"} : <i1>
    %26 = source {handshake.bb = 1 : ui32, handshake.name = "source2"} : <>
    %27 = constant %26 {handshake.bb = 1 : ui32, handshake.name = "constant8", value = 1.100000e+00 : f32} : <>, <f32>
    %28 = mulf %25, %27 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "mulf2"} : <f32>
    %29 = passer %30[%23#1] {handshake.bb = 1 : ui32, handshake.name = "passer6"} : <f32>, <i1>
    %30 = br %28 {handshake.bb = 1 : ui32, handshake.name = "br6"} : <f32>
    %31 = passer %32[%23#2] {handshake.bb = 1 : ui32, handshake.name = "passer7"} : <i8>, <i1>
    %32 = br %24 {handshake.bb = 1 : ui32, handshake.name = "br7"} : <i8>
    %33 = passer %34[%23#0] {handshake.bb = 1 : ui32, handshake.name = "passer8"} : <>, <i1>
    %34 = br %result_1 {handshake.bb = 1 : ui32, handshake.name = "br8"} : <>
    %35:2 = fork [2] %36 {handshake.bb = 1 : ui32, handshake.name = "fork34"} : <i8>
    %36 = merge %7#2 {handshake.bb = 1 : ui32, handshake.name = "merge2"} : <i8>
    %37 = passer %38[%21#8] {handshake.bb = 1 : ui32, handshake.name = "passer9"} : <i7>, <i1>
    %38 = trunci %35#0 {handshake.bb = 1 : ui32, handshake.name = "trunci1"} : <i8> to <i7>
    %39 = passer %40#0[%21#6] {handshake.bb = 1 : ui32, handshake.name = "passer10"} : <f32>, <i1>
    %40:2 = fork [2] %41 {handshake.bb = 1 : ui32, handshake.name = "fork35"} : <f32>
    %41 = merge %10#3 {handshake.bb = 1 : ui32, handshake.name = "merge3"} : <f32>
    %42:2 = fork [2] %result_3 {handshake.bb = 1 : ui32, handshake.name = "fork36"} : <>
    %result_3, %index_4 = control_merge [%11#1]  {handshake.bb = 1 : ui32, handshake.name = "control_merge2"} : [<>] to <>, <i1>
    sink %index_4 {handshake.name = "sink1"} : <i1>
    %43 = constant %42#0 {handshake.bb = 1 : ui32, handshake.name = "constant3", value = 1 : i2} : <>, <i2>
    %44 = passer %45[%21#4] {handshake.bb = 1 : ui32, handshake.name = "passer11"} : <i32>, <i1>
    %45 = extsi %43 {handshake.bb = 1 : ui32, handshake.name = "extsi1"} : <i2> to <i32>
    %46 = source {handshake.bb = 1 : ui32, handshake.name = "source3"} : <>
    %47 = constant %46 {handshake.bb = 1 : ui32, handshake.name = "constant9", value = 1.100000e+00 : f32} : <>, <f32>
    %addressResult_5, %dataResult_6 = store[%37] %39 {handshake.bb = 1 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store0"} : <i7>, <f32>, <i7>, <f32>
    %48 = divf %40#1, %47 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "divf0"} : <f32>
    %49 = passer %50[%21#5] {handshake.bb = 1 : ui32, handshake.name = "passer12"} : <f32>, <i1>
    %50 = br %48 {handshake.bb = 1 : ui32, handshake.name = "br9"} : <f32>
    %51 = passer %52[%21#7] {handshake.bb = 1 : ui32, handshake.name = "passer13"} : <i8>, <i1>
    %52 = br %35#1 {handshake.bb = 1 : ui32, handshake.name = "br10"} : <i8>
    %53 = passer %54[%21#3] {handshake.bb = 1 : ui32, handshake.name = "passer14"} : <>, <i1>
    %54 = br %42#1 {handshake.bb = 1 : ui32, handshake.name = "br11"} : <>
    %55 = mux %21#1 [%29, %49] {handshake.bb = 1 : ui32, handshake.name = "mux2"} : <i1>, [<f32>, <f32>] to <f32>
    %56:2 = fork [2] %55 {handshake.bb = 1 : ui32, handshake.name = "fork8"} : <f32>
    %57 = mux %21#0 [%31, %51] {handshake.bb = 1 : ui32, handshake.name = "mux3"} : <i1>, [<i8>, <i8>] to <i8>
    %58 = extsi %57 {handshake.bb = 1 : ui32, handshake.name = "extsi5"} : <i8> to <i9>
    %59 = mux %21#2 [%33, %53] {handshake.bb = 1 : ui32, handshake.name = "mux4"} : <i1>, [<>, <>] to <>
    %60 = source {handshake.bb = 1 : ui32, handshake.name = "source4"} : <>
    %61 = constant %60 {handshake.bb = 1 : ui32, handshake.name = "constant4", value = 1 : i2} : <>, <i2>
    %62 = extsi %61 {handshake.bb = 1 : ui32, handshake.name = "extsi6"} : <i2> to <i9>
    %63 = source {handshake.bb = 1 : ui32, handshake.name = "source5"} : <>
    %64 = constant %63 {handshake.bb = 1 : ui32, handshake.name = "constant12", value = 100 : i8} : <>, <i8>
    %65 = extsi %64 {handshake.bb = 1 : ui32, handshake.name = "extsi7"} : <i8> to <i9>
    %66 = addf %56#0, %56#1 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "addf1"} : <f32>
    %67 = addi %58, %62 {handshake.bb = 1 : ui32, handshake.name = "addi0"} : <i9>
    %68:2 = fork [2] %67 {handshake.bb = 1 : ui32, handshake.name = "fork10"} : <i9>
    %69 = trunci %68#0 {handshake.bb = 1 : ui32, handshake.name = "trunci2"} : <i9> to <i8>
    %70 = cmpi ult, %68#1, %65 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i9>
    %71:3 = fork [3] %70 {handshake.bb = 1 : ui32, handshake.name = "fork11"} : <i1>
    %trueResult, %falseResult = cond_br %71#0, %69 {handshake.bb = 1 : ui32, handshake.name = "cond_br5"} : <i1>, <i8>
    sink %falseResult {handshake.name = "sink2"} : <i8>
    %trueResult_7, %falseResult_8 = cond_br %71#1, %66 {handshake.bb = 1 : ui32, handshake.name = "cond_br6"} : <i1>, <f32>
    %trueResult_9, %falseResult_10 = cond_br %71#2, %59 {handshake.bb = 1 : ui32, handshake.name = "cond_br7"} : <i1>, <>
    %72 = merge %falseResult_8 {handshake.bb = 2 : ui32, handshake.name = "merge4"} : <f32>
    %result_11, %index_12 = control_merge [%falseResult_10]  {handshake.bb = 2 : ui32, handshake.name = "control_merge4"} : [<>] to <>, <i1>
    sink %index_12 {handshake.name = "sink3"} : <i1>
    %73:2 = fork [2] %result_11 {handshake.bb = 2 : ui32, handshake.name = "fork12"} : <>
    end {handshake.bb = 2 : ui32, handshake.name = "end0"} %72, %memEnd_0, %memEnd, %0#1 : <f32>, <>, <>, <>
  }
}

