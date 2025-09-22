module {
  handshake.func @golden_ratio(%arg0: !handshake.channel<f32>, %arg1: !handshake.control<>, ...) -> (!handshake.channel<f32>, !handshake.control<>) attributes {argNames = ["x0", "start"], resNames = ["out0", "end"]} {
    %0 = constant %arg1 {handshake.bb = 0 : ui32, handshake.name = "constant4", value = 0 : i32} : <>, <i32>
    %1 = source {handshake.bb = 0 : ui32, handshake.name = "source0"} : <>
    %2 = constant %1 {handshake.bb = 0 : ui32, handshake.name = "constant5", value = 1.000000e+00 : f32} : <>, <f32>
    %3 = divf %2, %arg0 {fastmath = #arith.fastmath<none>, handshake.bb = 0 : ui32, handshake.name = "divf0"} : <f32>
    %4 = br %3 {handshake.bb = 0 : ui32, handshake.name = "br2"} : <f32>
    %5 = br %0 {handshake.bb = 0 : ui32, handshake.name = "br3"} : <i32>
    %6 = br %arg0 {handshake.bb = 0 : ui32, handshake.name = "br4"} : <f32>
    %7 = br %arg1 {handshake.bb = 0 : ui32, handshake.name = "br5"} : <>
    %8 = mux %index [%4, %trueResult_12] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<f32>, <f32>] to <f32>
    %9 = mux %index [%5, %trueResult_14] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i1>, [<i32>, <i32>] to <i32>
    %10 = mux %index [%6, %trueResult_16] {handshake.bb = 1 : ui32, handshake.name = "mux2"} : <i1>, [<f32>, <f32>] to <f32>
    %result, %index = control_merge [%7, %trueResult_18]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %11 = br %10 {handshake.bb = 1 : ui32, handshake.name = "br6"} : <f32>
    %12 = br %8 {handshake.bb = 1 : ui32, handshake.name = "br7"} : <f32>
    %13 = br %9 {handshake.bb = 1 : ui32, handshake.name = "br8"} : <i32>
    %14 = br %result {handshake.bb = 1 : ui32, handshake.name = "br9"} : <>
    %15 = mux %index_1 [%11, %falseResult_7] {handshake.bb = 2 : ui32, handshake.name = "mux3"} : <i1>, [<f32>, <f32>] to <f32>
    %16 = mux %index_1 [%12, %falseResult_9] {handshake.bb = 2 : ui32, handshake.name = "mux4"} : <i1>, [<f32>, <f32>] to <f32>
    %17 = mux %index_1 [%13, %falseResult_3] {handshake.bb = 2 : ui32, handshake.name = "mux5"} : <i1>, [<i32>, <i32>] to <i32>
    %result_0, %index_1 = control_merge [%14, %falseResult_5]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>, <>] to <>, <i1>
    %18 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %19 = constant %18 {handshake.bb = 2 : ui32, handshake.name = "constant6", value = 5.000000e-01 : f32} : <>, <f32>
    %20 = source {handshake.bb = 2 : ui32, handshake.name = "source2"} : <>
    %21 = constant %20 {handshake.bb = 2 : ui32, handshake.name = "constant7", value = 1.000000e-01 : f32} : <>, <f32>
    %22 = mulf %15, %16 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "mulf0"} : <f32>
    %23 = addf %15, %22 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "addf0"} : <f32>
    %24 = mulf %23, %19 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "mulf1"} : <f32>
    %25 = subf %24, %15 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "subf0"} : <f32>
    %26 = absf %25 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "absf0"} : <f32> to <f32>
    %27 = cmpf olt, %26, %21 {handshake.bb = 2 : ui32, handshake.name = "cmpf0"} : <f32>
    %trueResult, %falseResult = cond_br %27, %15 {handshake.bb = 2 : ui32, handshake.name = "cond_br2"} : <i1>, <f32>
    %trueResult_2, %falseResult_3 = cond_br %27, %17 {handshake.bb = 2 : ui32, handshake.name = "cond_br3"} : <i1>, <i32>
    %trueResult_4, %falseResult_5 = cond_br %27, %result_0 {handshake.bb = 2 : ui32, handshake.name = "cond_br4"} : <i1>, <>
    %trueResult_6, %falseResult_7 = cond_br %27, %24 {handshake.bb = 2 : ui32, handshake.name = "cond_br5"} : <i1>, <f32>
    %trueResult_8, %falseResult_9 = cond_br %27, %16 {handshake.bb = 2 : ui32, handshake.name = "cond_br6"} : <i1>, <f32>
    %28 = merge %trueResult {handshake.bb = 3 : ui32, handshake.name = "merge0"} : <f32>
    %29 = merge %trueResult_2 {handshake.bb = 3 : ui32, handshake.name = "merge1"} : <i32>
    %result_10, %index_11 = control_merge [%trueResult_4]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>] to <>, <i1>
    %30 = source {handshake.bb = 3 : ui32, handshake.name = "source3"} : <>
    %31 = constant %30 {handshake.bb = 3 : ui32, handshake.name = "constant8", value = 1.000000e+00 : f32} : <>, <f32>
    %32 = source {handshake.bb = 3 : ui32, handshake.name = "source4"} : <>
    %33 = constant %32 {handshake.bb = 3 : ui32, handshake.name = "constant9", value = 1 : i32} : <>, <i32>
    %34 = source {handshake.bb = 3 : ui32, handshake.name = "source5"} : <>
    %35 = constant %34 {handshake.bb = 3 : ui32, handshake.name = "constant10", value = 100 : i32} : <>, <i32>
    %36 = addf %28, %31 {fastmath = #arith.fastmath<none>, handshake.bb = 3 : ui32, handshake.name = "addf1"} : <f32>
    %37 = addi %29, %33 {handshake.bb = 3 : ui32, handshake.name = "addi0"} : <i32>
    %38 = divf %31, %36 {fastmath = #arith.fastmath<none>, handshake.bb = 3 : ui32, handshake.name = "divf1"} : <f32>
    %39 = cmpi ult, %37, %35 {handshake.bb = 3 : ui32, handshake.name = "cmpi0"} : <i32>
    %trueResult_12, %falseResult_13 = cond_br %39, %38 {handshake.bb = 3 : ui32, handshake.name = "cond_br7"} : <i1>, <f32>
    %trueResult_14, %falseResult_15 = cond_br %39, %37 {handshake.bb = 3 : ui32, handshake.name = "cond_br8"} : <i1>, <i32>
    %trueResult_16, %falseResult_17 = cond_br %39, %36 {handshake.bb = 3 : ui32, handshake.name = "cond_br9"} : <i1>, <f32>
    %trueResult_18, %falseResult_19 = cond_br %39, %result_10 {handshake.bb = 3 : ui32, handshake.name = "cond_br10"} : <i1>, <>
    %40 = merge %falseResult_17 {handshake.bb = 4 : ui32, handshake.name = "merge2"} : <f32>
    %result_20, %index_21 = control_merge [%falseResult_19]  {handshake.bb = 4 : ui32, handshake.name = "control_merge3"} : [<>] to <>, <i1>
    end {handshake.bb = 4 : ui32, handshake.name = "end0"} %40, %arg1 : <f32>, <>
  }
}

