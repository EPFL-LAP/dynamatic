module {
  handshake.func @bisection(%arg0: !handshake.channel<f32>, %arg1: !handshake.channel<f32>, %arg2: !handshake.channel<f32>, %arg3: !handshake.control<>, ...) -> (!handshake.channel<f32>, !handshake.control<>) attributes {argNames = ["a", "b", "tol", "start"], resNames = ["out0", "end"]} {
    %0 = constant %arg3 {handshake.bb = 0 : ui32, handshake.name = "constant4", value = 0 : i32} : <>, <i32>
    %1 = source {handshake.bb = 0 : ui32, handshake.name = "source0"} : <>
    %2 = constant %1 {handshake.bb = 0 : ui32, handshake.name = "constant5", value = -2.000000e+00 : f32} : <>, <f32>
    %3 = mulf %arg0, %arg0 {fastmath = #arith.fastmath<none>, handshake.bb = 0 : ui32, handshake.name = "mulf0"} : <f32>
    %4 = addf %3, %2 {fastmath = #arith.fastmath<none>, handshake.bb = 0 : ui32, handshake.name = "addf0"} : <f32>
    %5 = br %arg0 {handshake.bb = 0 : ui32, handshake.name = "br2"} : <f32>
    %6 = br %arg1 {handshake.bb = 0 : ui32, handshake.name = "br3"} : <f32>
    %7 = br %0 {handshake.bb = 0 : ui32, handshake.name = "br4"} : <i32>
    %8 = br %4 {handshake.bb = 0 : ui32, handshake.name = "br5"} : <f32>
    %9 = br %arg2 {handshake.bb = 0 : ui32, handshake.name = "br6"} : <f32>
    %10 = br %arg3 {handshake.bb = 0 : ui32, handshake.name = "br7"} : <>
    %11 = mux %index [%5, %trueResult_34] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<f32>, <f32>] to <f32>
    %12 = mux %index [%6, %trueResult_36] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i1>, [<f32>, <f32>] to <f32>
    %13 = mux %index [%7, %trueResult_38] {handshake.bb = 1 : ui32, handshake.name = "mux2"} : <i1>, [<i32>, <i32>] to <i32>
    %14 = mux %index [%8, %trueResult_40] {handshake.bb = 1 : ui32, handshake.name = "mux3"} : <i1>, [<f32>, <f32>] to <f32>
    %15 = mux %index [%9, %trueResult_42] {handshake.bb = 1 : ui32, handshake.name = "mux4"} : <i1>, [<f32>, <f32>] to <f32>
    %result, %index = control_merge [%10, %trueResult_44]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %16 = source {handshake.bb = 1 : ui32, handshake.name = "source1"} : <>
    %17 = constant %16 {handshake.bb = 1 : ui32, handshake.name = "constant6", value = -2.000000e+00 : f32} : <>, <f32>
    %18 = source {handshake.bb = 1 : ui32, handshake.name = "source2"} : <>
    %19 = constant %18 {handshake.bb = 1 : ui32, handshake.name = "constant7", value = 5.000000e-01 : f32} : <>, <f32>
    %20 = addf %11, %12 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "addf1"} : <f32>
    %21 = mulf %20, %19 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "mulf1"} : <f32>
    %22 = mulf %21, %21 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "mulf2"} : <f32>
    %23 = addf %22, %17 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "addf2"} : <f32>
    %24 = absf %23 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "absf0"} : <f32> to <f32>
    %25 = cmpf olt, %24, %15 {handshake.bb = 1 : ui32, handshake.name = "cmpf0"} : <f32>
    %trueResult, %falseResult = cond_br %25, %21 {handshake.bb = 1 : ui32, handshake.name = "cond_br3"} : <i1>, <f32>
    %trueResult_0, %falseResult_1 = cond_br %25, %result {handshake.bb = 1 : ui32, handshake.name = "cond_br4"} : <i1>, <>
    %trueResult_2, %falseResult_3 = cond_br %25, %15 {handshake.bb = 1 : ui32, handshake.name = "cond_br5"} : <i1>, <f32>
    %trueResult_4, %falseResult_5 = cond_br %25, %11 {handshake.bb = 1 : ui32, handshake.name = "cond_br6"} : <i1>, <f32>
    %trueResult_6, %falseResult_7 = cond_br %25, %12 {handshake.bb = 1 : ui32, handshake.name = "cond_br7"} : <i1>, <f32>
    %trueResult_8, %falseResult_9 = cond_br %25, %13 {handshake.bb = 1 : ui32, handshake.name = "cond_br8"} : <i1>, <i32>
    %trueResult_10, %falseResult_11 = cond_br %25, %14 {handshake.bb = 1 : ui32, handshake.name = "cond_br9"} : <i1>, <f32>
    %trueResult_12, %falseResult_13 = cond_br %25, %23 {handshake.bb = 1 : ui32, handshake.name = "cond_br10"} : <i1>, <f32>
    %26 = merge %falseResult_3 {handshake.bb = 2 : ui32, handshake.name = "merge0"} : <f32>
    %27 = merge %falseResult_5 {handshake.bb = 2 : ui32, handshake.name = "merge1"} : <f32>
    %28 = merge %falseResult_7 {handshake.bb = 2 : ui32, handshake.name = "merge2"} : <f32>
    %29 = merge %falseResult_9 {handshake.bb = 2 : ui32, handshake.name = "merge3"} : <i32>
    %30 = merge %falseResult_11 {handshake.bb = 2 : ui32, handshake.name = "merge4"} : <f32>
    %31 = merge %falseResult {handshake.bb = 2 : ui32, handshake.name = "merge5"} : <f32>
    %32 = merge %falseResult_13 {handshake.bb = 2 : ui32, handshake.name = "merge6"} : <f32>
    %result_14, %index_15 = control_merge [%falseResult_1]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>] to <>, <i1>
    %33 = source {handshake.bb = 2 : ui32, handshake.name = "source3"} : <>
    %34 = constant %33 {handshake.bb = 2 : ui32, handshake.name = "constant8", value = 5.000000e-01 : f32} : <>, <f32>
    %35 = subf %28, %27 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "subf0"} : <f32>
    %36 = mulf %35, %34 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "mulf3"} : <f32>
    %37 = cmpf olt, %36, %26 {handshake.bb = 2 : ui32, handshake.name = "cmpf1"} : <f32>
    %trueResult_16, %falseResult_17 = cond_br %37, %31 {handshake.bb = 2 : ui32, handshake.name = "cond_br11"} : <i1>, <f32>
    %trueResult_18, %falseResult_19 = cond_br %37, %result_14 {handshake.bb = 2 : ui32, handshake.name = "cond_br12"} : <i1>, <>
    %trueResult_20, %falseResult_21 = cond_br %37, %26 {handshake.bb = 2 : ui32, handshake.name = "cond_br13"} : <i1>, <f32>
    %trueResult_22, %falseResult_23 = cond_br %37, %27 {handshake.bb = 2 : ui32, handshake.name = "cond_br14"} : <i1>, <f32>
    %trueResult_24, %falseResult_25 = cond_br %37, %28 {handshake.bb = 2 : ui32, handshake.name = "cond_br15"} : <i1>, <f32>
    %trueResult_26, %falseResult_27 = cond_br %37, %29 {handshake.bb = 2 : ui32, handshake.name = "cond_br16"} : <i1>, <i32>
    %trueResult_28, %falseResult_29 = cond_br %37, %30 {handshake.bb = 2 : ui32, handshake.name = "cond_br17"} : <i1>, <f32>
    %trueResult_30, %falseResult_31 = cond_br %37, %32 {handshake.bb = 2 : ui32, handshake.name = "cond_br18"} : <i1>, <f32>
    %38 = merge %falseResult_21 {handshake.bb = 3 : ui32, handshake.name = "merge7"} : <f32>
    %39 = merge %falseResult_23 {handshake.bb = 3 : ui32, handshake.name = "merge8"} : <f32>
    %40 = merge %falseResult_25 {handshake.bb = 3 : ui32, handshake.name = "merge9"} : <f32>
    %41 = merge %falseResult_27 {handshake.bb = 3 : ui32, handshake.name = "merge10"} : <i32>
    %42 = merge %falseResult_29 {handshake.bb = 3 : ui32, handshake.name = "merge11"} : <f32>
    %43 = merge %falseResult_17 {handshake.bb = 3 : ui32, handshake.name = "merge12"} : <f32>
    %44 = merge %falseResult_31 {handshake.bb = 3 : ui32, handshake.name = "merge13"} : <f32>
    %result_32, %index_33 = control_merge [%falseResult_19]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>] to <>, <i1>
    %45 = constant %result_32 {handshake.bb = 3 : ui32, handshake.name = "constant9", value = 0.000000e+00 : f32} : <>, <f32>
    %46 = source {handshake.bb = 3 : ui32, handshake.name = "source4"} : <>
    %47 = constant %46 {handshake.bb = 3 : ui32, handshake.name = "constant10", value = 1 : i32} : <>, <i32>
    %48 = source {handshake.bb = 3 : ui32, handshake.name = "source5"} : <>
    %49 = constant %48 {handshake.bb = 3 : ui32, handshake.name = "constant11", value = 100 : i32} : <>, <i32>
    %50 = mulf %42, %44 {fastmath = #arith.fastmath<none>, handshake.bb = 3 : ui32, handshake.name = "mulf4"} : <f32>
    %51 = cmpf olt, %50, %45 {handshake.bb = 3 : ui32, handshake.name = "cmpf2"} : <f32>
    %52 = select %51[%43, %40] {handshake.bb = 3 : ui32, handshake.name = "select0"} : <i1>, <f32>
    %53 = select %51[%39, %43] {handshake.bb = 3 : ui32, handshake.name = "select1"} : <i1>, <f32>
    %54 = select %51[%42, %44] {handshake.bb = 3 : ui32, handshake.name = "select2"} : <i1>, <f32>
    %55 = addi %41, %47 {handshake.bb = 3 : ui32, handshake.name = "addi0"} : <i32>
    %56 = cmpi ult, %55, %49 {handshake.bb = 3 : ui32, handshake.name = "cmpi0"} : <i32>
    %trueResult_34, %falseResult_35 = cond_br %56, %53 {handshake.bb = 3 : ui32, handshake.name = "cond_br19"} : <i1>, <f32>
    %trueResult_36, %falseResult_37 = cond_br %56, %52 {handshake.bb = 3 : ui32, handshake.name = "cond_br20"} : <i1>, <f32>
    %trueResult_38, %falseResult_39 = cond_br %56, %55 {handshake.bb = 3 : ui32, handshake.name = "cond_br21"} : <i1>, <i32>
    %trueResult_40, %falseResult_41 = cond_br %56, %54 {handshake.bb = 3 : ui32, handshake.name = "cond_br22"} : <i1>, <f32>
    %trueResult_42, %falseResult_43 = cond_br %56, %38 {handshake.bb = 3 : ui32, handshake.name = "cond_br23"} : <i1>, <f32>
    %trueResult_44, %falseResult_45 = cond_br %56, %result_32 {handshake.bb = 3 : ui32, handshake.name = "cond_br24"} : <i1>, <>
    %trueResult_46, %falseResult_47 = cond_br %56, %45 {handshake.bb = 3 : ui32, handshake.name = "cond_br25"} : <i1>, <f32>
    %57 = mux %index_49 [%trueResult, %falseResult_47] {handshake.bb = 4 : ui32, handshake.name = "mux5"} : <i1>, [<f32>, <f32>] to <f32>
    %result_48, %index_49 = control_merge [%trueResult_0, %falseResult_45]  {handshake.bb = 4 : ui32, handshake.name = "control_merge3"} : [<>, <>] to <>, <i1>
    %58 = br %57 {handshake.bb = 4 : ui32, handshake.name = "br8"} : <f32>
    %59 = br %result_48 {handshake.bb = 4 : ui32, handshake.name = "br9"} : <>
    %60 = mux %index_51 [%trueResult_16, %58] {handshake.bb = 5 : ui32, handshake.name = "mux6"} : <i1>, [<f32>, <f32>] to <f32>
    %result_50, %index_51 = control_merge [%trueResult_18, %59]  {handshake.bb = 5 : ui32, handshake.name = "control_merge4"} : [<>, <>] to <>, <i1>
    end {handshake.bb = 5 : ui32, handshake.name = "end0"} %60, %arg3 : <f32>, <>
  }
}

