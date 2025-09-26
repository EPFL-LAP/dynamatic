module {
  handshake.func @if_float2(%arg0: !handshake.channel<f32>, %arg1: memref<100xf32>, %arg2: memref<100xf32>, %arg3: !handshake.control<>, %arg4: !handshake.control<>, %arg5: !handshake.control<>, ...) -> (!handshake.channel<f32>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["x0", "a", "minus_trace", "a_start", "minus_trace_start", "start"], resNames = ["out0", "a_end", "minus_trace_end", "end"]} {
    %memEnd = mem_controller[%arg2 : memref<100xf32>] %arg4 (%24, %addressResult_9, %dataResult_10) %result_19 {connectedBlocks = [3 : i32], handshake.name = "mem_controller0"} :    (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<f32>) -> ()
    %outputs, %memEnd_0 = mem_controller[%arg1 : memref<100xf32>] %arg3 (%addressResult) %result_19 {connectedBlocks = [1 : i32], handshake.name = "mem_controller1"} :    (!handshake.channel<i32>) -> !handshake.channel<f32>
    %0 = constant %arg5 {handshake.bb = 0 : ui32, handshake.name = "constant6", value = 0 : i32} : <>, <i32>
    %1 = br %0 {handshake.bb = 0 : ui32, handshake.name = "br3"} : <i32>
    %2 = br %arg0 {handshake.bb = 0 : ui32, handshake.name = "br4"} : <f32>
    %3 = br %arg5 {handshake.bb = 0 : ui32, handshake.name = "br5"} : <>
    %4 = mux %index [%1, %trueResult_13] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i32>, <i32>] to <i32>
    %5 = mux %index [%2, %trueResult_15] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i1>, [<f32>, <f32>] to <f32>
    %result, %index = control_merge [%3, %trueResult_17]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %6 = source {handshake.bb = 1 : ui32, handshake.name = "source0"} : <>
    %7 = constant %6 {handshake.bb = 1 : ui32, handshake.name = "constant7", value = -0.899999976 : f32} : <>, <f32>
    %8 = source {handshake.bb = 1 : ui32, handshake.name = "source1"} : <>
    %9 = constant %8 {handshake.bb = 1 : ui32, handshake.name = "constant8", value = 0.000000e+00 : f32} : <>, <f32>
    %10 = extui %4 {handshake.bb = 1 : ui32, handshake.name = "extui0"} : <i32> to <i64>
    %11 = trunci %10 {handshake.bb = 1 : ui32, handshake.name = "index_cast0"} : <i64> to <i32>
    %addressResult, %dataResult = load[%11] %outputs {handshake.bb = 1 : ui32, handshake.name = "load0"} : <i32>, <f32>, <i32>, <f32>
    %12 = mulf %dataResult, %5 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "mulf0"} : <f32>
    %13 = mulf %5, %7 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "mulf1"} : <f32>
    %14 = addf %12, %13 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "addf0"} : <f32>
    %15 = cmpf ugt, %14, %9 {handshake.bb = 1 : ui32, handshake.name = "cmpf0"} : <f32>
    %trueResult, %falseResult = cond_br %15, %4 {handshake.bb = 1 : ui32, handshake.name = "cond_br2"} : <i1>, <i32>
    %trueResult_1, %falseResult_2 = cond_br %15, %5 {handshake.bb = 1 : ui32, handshake.name = "cond_br3"} : <i1>, <f32>
    %trueResult_3, %falseResult_4 = cond_br %15, %result {handshake.bb = 1 : ui32, handshake.name = "cond_br4"} : <i1>, <>
    %16 = merge %falseResult {handshake.bb = 2 : ui32, handshake.name = "merge0"} : <i32>
    %17 = merge %falseResult_2 {handshake.bb = 2 : ui32, handshake.name = "merge1"} : <f32>
    %result_5, %index_6 = control_merge [%falseResult_4]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>] to <>, <i1>
    %18 = source {handshake.bb = 2 : ui32, handshake.name = "source2"} : <>
    %19 = constant %18 {handshake.bb = 2 : ui32, handshake.name = "constant9", value = 3.000000e+00 : f32} : <>, <f32>
    %20 = addf %17, %19 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "addf1"} : <f32>
    %21 = br %20 {handshake.bb = 2 : ui32, handshake.name = "br6"} : <f32>
    %22 = br %16 {handshake.bb = 2 : ui32, handshake.name = "br7"} : <i32>
    %23 = br %result_5 {handshake.bb = 2 : ui32, handshake.name = "br8"} : <>
    %24 = constant %result_7 {handshake.bb = 3 : ui32, handshake.name = "constant0", value = 1 : i32} : <>, <i32>
    %25 = merge %trueResult {handshake.bb = 3 : ui32, handshake.name = "merge2"} : <i32>
    %26 = merge %trueResult_1 {handshake.bb = 3 : ui32, handshake.name = "merge3"} : <f32>
    %result_7, %index_8 = control_merge [%trueResult_3]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>] to <>, <i1>
    %27 = source {handshake.bb = 3 : ui32, handshake.name = "source3"} : <>
    %28 = constant %27 {handshake.bb = 3 : ui32, handshake.name = "constant10", value = 1.000000e+00 : f32} : <>, <f32>
    %29 = extui %25 {handshake.bb = 3 : ui32, handshake.name = "extui1"} : <i32> to <i64>
    %30 = trunci %29 {handshake.bb = 3 : ui32, handshake.name = "index_cast1"} : <i64> to <i32>
    %addressResult_9, %dataResult_10 = store[%30] %26 {handshake.bb = 3 : ui32, handshake.name = "store0"} : <i32>, <f32>, <i32>, <f32>
    %31 = addf %26, %28 {fastmath = #arith.fastmath<none>, handshake.bb = 3 : ui32, handshake.name = "addf2"} : <f32>
    %32 = br %31 {handshake.bb = 3 : ui32, handshake.name = "br9"} : <f32>
    %33 = br %25 {handshake.bb = 3 : ui32, handshake.name = "br10"} : <i32>
    %34 = br %result_7 {handshake.bb = 3 : ui32, handshake.name = "br11"} : <>
    %35 = mux %index_12 [%21, %32] {handshake.bb = 4 : ui32, handshake.name = "mux2"} : <i1>, [<f32>, <f32>] to <f32>
    %36 = mux %index_12 [%22, %33] {handshake.bb = 4 : ui32, handshake.name = "mux3"} : <i1>, [<i32>, <i32>] to <i32>
    %result_11, %index_12 = control_merge [%23, %34]  {handshake.bb = 4 : ui32, handshake.name = "control_merge3"} : [<>, <>] to <>, <i1>
    %37 = source {handshake.bb = 4 : ui32, handshake.name = "source4"} : <>
    %38 = constant %37 {handshake.bb = 4 : ui32, handshake.name = "constant11", value = 1.000000e+00 : f32} : <>, <f32>
    %39 = source {handshake.bb = 4 : ui32, handshake.name = "source5"} : <>
    %40 = constant %39 {handshake.bb = 4 : ui32, handshake.name = "constant12", value = 1 : i32} : <>, <i32>
    %41 = source {handshake.bb = 4 : ui32, handshake.name = "source6"} : <>
    %42 = constant %41 {handshake.bb = 4 : ui32, handshake.name = "constant13", value = 100 : i32} : <>, <i32>
    %43 = divf %38, %35 {fastmath = #arith.fastmath<none>, handshake.bb = 4 : ui32, handshake.name = "divf0"} : <f32>
    %44 = addi %36, %40 {handshake.bb = 4 : ui32, handshake.name = "addi0"} : <i32>
    %45 = cmpi ult, %44, %42 {handshake.bb = 4 : ui32, handshake.name = "cmpi0"} : <i32>
    %trueResult_13, %falseResult_14 = cond_br %45, %44 {handshake.bb = 4 : ui32, handshake.name = "cond_br5"} : <i1>, <i32>
    %trueResult_15, %falseResult_16 = cond_br %45, %43 {handshake.bb = 4 : ui32, handshake.name = "cond_br6"} : <i1>, <f32>
    %trueResult_17, %falseResult_18 = cond_br %45, %result_11 {handshake.bb = 4 : ui32, handshake.name = "cond_br7"} : <i1>, <>
    %46 = merge %falseResult_16 {handshake.bb = 5 : ui32, handshake.name = "merge4"} : <f32>
    %result_19, %index_20 = control_merge [%falseResult_18]  {handshake.bb = 5 : ui32, handshake.name = "control_merge4"} : [<>] to <>, <i1>
    end {handshake.bb = 5 : ui32, handshake.name = "end0"} %46, %memEnd_0, %memEnd, %arg5 : <f32>, <>, <>, <>
  }
}

