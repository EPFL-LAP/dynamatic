module {
  handshake.func @golden_ratio(%arg0: !handshake.channel<f32>, %arg1: !handshake.channel<f32>, %arg2: !handshake.control<>, ...) -> (!handshake.channel<f32>, !handshake.control<>) attributes {argNames = ["x0", "x1", "start"], resNames = ["out0", "end"]} {
    %0 = constant %arg2 {handshake.bb = 0 : ui32, handshake.name = "constant5", value = 0 : i32} : <>, <i32>
    %1 = br %0 {handshake.bb = 0 : ui32, handshake.name = "br2"} : <i32>
    %2 = br %arg1 {handshake.bb = 0 : ui32, handshake.name = "br3"} : <f32>
    %3 = br %arg0 {handshake.bb = 0 : ui32, handshake.name = "br4"} : <f32>
    %4 = br %arg2 {handshake.bb = 0 : ui32, handshake.name = "br5"} : <>
    %5 = mux %index [%1, %trueResult_12] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i32>, <i32>] to <i32>
    %6 = mux %index [%2, %trueResult_14] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i1>, [<f32>, <f32>] to <f32>
    %7 = mux %index [%3, %trueResult_16] {handshake.bb = 1 : ui32, handshake.name = "mux2"} : <i1>, [<f32>, <f32>] to <f32>
    %result, %index = control_merge [%4, %trueResult_18]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %8 = br %7 {handshake.bb = 1 : ui32, handshake.name = "br6"} : <f32>
    %9 = br %5 {handshake.bb = 1 : ui32, handshake.name = "br7"} : <i32>
    %10 = br %6 {handshake.bb = 1 : ui32, handshake.name = "br8"} : <f32>
    %11 = br %result {handshake.bb = 1 : ui32, handshake.name = "br9"} : <>
    %12 = mux %index_1 [%8, %falseResult_7] {handshake.bb = 2 : ui32, handshake.name = "mux3"} : <i1>, [<f32>, <f32>] to <f32>
    %13 = mux %index_1 [%9, %falseResult_3] {handshake.bb = 2 : ui32, handshake.name = "mux4"} : <i1>, [<i32>, <i32>] to <i32>
    %14 = mux %index_1 [%10, %falseResult_9] {handshake.bb = 2 : ui32, handshake.name = "mux5"} : <i1>, [<f32>, <f32>] to <f32>
    %result_0, %index_1 = control_merge [%11, %falseResult_5]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>, <>] to <>, <i1>
    %15 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %16 = constant %15 {handshake.bb = 2 : ui32, handshake.name = "constant6", value = 5.000000e-01 : f32} : <>, <f32>
    %17 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %18 = constant %17 {handshake.bb = 2 : ui32, handshake.name = "constant7", value = 1.000000e-01 : f32} : <>, <f32>
    %19 = mulf %12, %14 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "mulf0"} : <f32>
    %20 = addf %12, %19 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "addf0"} : <f32>
    %21 = mulf %20, %16 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "mulf1"} : <f32>
    %22 = subf %21, %12 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "subf0"} : <f32>
    %23 = absf %22 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "absf0"} : <f32> to <f32>
    %24 = cmpf olt, %23, %18 {handshake.bb = 2 : ui32, handshake.name = "cmpf0"} : <f32>
    %trueResult, %falseResult = cond_br %24, %12 {handshake.bb = 2 : ui32, handshake.name = "cond_br2"} : <i1>, <f32>
    %trueResult_2, %falseResult_3 = cond_br %24, %13 {handshake.bb = 2 : ui32, handshake.name = "cond_br3"} : <i1>, <i32>
    %trueResult_4, %falseResult_5 = cond_br %24, %result_0 {handshake.bb = 2 : ui32, handshake.name = "cond_br4"} : <i1>, <>
    %trueResult_6, %falseResult_7 = cond_br %24, %21 {handshake.bb = 2 : ui32, handshake.name = "cond_br5"} : <i1>, <f32>
    %trueResult_8, %falseResult_9 = cond_br %24, %14 {handshake.bb = 2 : ui32, handshake.name = "cond_br6"} : <i1>, <f32>
    %25 = merge %trueResult {handshake.bb = 3 : ui32, handshake.name = "merge0"} : <f32>
    %26 = merge %trueResult_2 {handshake.bb = 3 : ui32, handshake.name = "merge1"} : <i32>
    %result_10, %index_11 = control_merge [%trueResult_4]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>] to <>, <i1>
    %27 = source {handshake.bb = 3 : ui32, handshake.name = "source2"} : <>
    %28 = constant %27 {handshake.bb = 3 : ui32, handshake.name = "constant8", value = 1.000000e+00 : f32} : <>, <f32>
    %29 = source {handshake.bb = 3 : ui32, handshake.name = "source3"} : <>
    %30 = constant %29 {handshake.bb = 3 : ui32, handshake.name = "constant9", value = 1 : i32} : <>, <i32>
    %31 = source {handshake.bb = 3 : ui32, handshake.name = "source4"} : <>
    %32 = constant %31 {handshake.bb = 3 : ui32, handshake.name = "constant10", value = 100 : i32} : <>, <i32>
    %33 = addf %25, %28 {fastmath = #arith.fastmath<none>, handshake.bb = 3 : ui32, handshake.name = "addf1"} : <f32>
    %34 = divf %28, %33 {fastmath = #arith.fastmath<none>, handshake.bb = 3 : ui32, handshake.name = "divf0"} : <f32>
    %35 = addi %26, %30 {handshake.bb = 3 : ui32, handshake.name = "addi0"} : <i32>
    %36 = cmpi ult, %35, %32 {handshake.bb = 3 : ui32, handshake.name = "cmpi0"} : <i32>
    %trueResult_12, %falseResult_13 = cond_br %36, %35 {handshake.bb = 3 : ui32, handshake.name = "cond_br7"} : <i1>, <i32>
    %trueResult_14, %falseResult_15 = cond_br %36, %34 {handshake.bb = 3 : ui32, handshake.name = "cond_br8"} : <i1>, <f32>
    %trueResult_16, %falseResult_17 = cond_br %36, %33 {handshake.bb = 3 : ui32, handshake.name = "cond_br9"} : <i1>, <f32>
    %trueResult_18, %falseResult_19 = cond_br %36, %result_10 {handshake.bb = 3 : ui32, handshake.name = "cond_br10"} : <i1>, <>
    %37 = merge %falseResult_17 {handshake.bb = 4 : ui32, handshake.name = "merge2"} : <f32>
    %result_20, %index_21 = control_merge [%falseResult_19]  {handshake.bb = 4 : ui32, handshake.name = "control_merge3"} : [<>] to <>, <i1>
    end {handshake.bb = 4 : ui32, handshake.name = "end0"} %37, %arg2 : <f32>, <>
  }
}

