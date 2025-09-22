module {
  handshake.func @bisection(%arg0: !handshake.channel<f32>, %arg1: !handshake.channel<f32>, %arg2: !handshake.channel<f32>, %arg3: !handshake.control<>, ...) -> (!handshake.channel<f32>, !handshake.control<>) attributes {argNames = ["a", "b", "tol", "start"], resNames = ["out0", "end"]} {
    %0 = constant %arg3 {handshake.bb = 0 : ui32, handshake.name = "constant4", value = 0 : i32} : <>, <i32>
    %1 = source {handshake.bb = 0 : ui32, handshake.name = "source0"} : <>
    %2 = constant %1 {handshake.bb = 0 : ui32, handshake.name = "constant5", value = -2.000000e+00 : f32} : <>, <f32>
    %3 = mulf %arg0, %arg0 {fastmath = #arith.fastmath<none>, handshake.bb = 0 : ui32, handshake.name = "mulf0"} : <f32>
    %4 = addf %3, %2 {fastmath = #arith.fastmath<none>, handshake.bb = 0 : ui32, handshake.name = "addf0"} : <f32>
    %5 = br %arg0 {handshake.bb = 0 : ui32, handshake.name = "br3"} : <f32>
    %6 = br %arg1 {handshake.bb = 0 : ui32, handshake.name = "br4"} : <f32>
    %7 = br %0 {handshake.bb = 0 : ui32, handshake.name = "br5"} : <i32>
    %8 = br %4 {handshake.bb = 0 : ui32, handshake.name = "br6"} : <f32>
    %9 = br %arg2 {handshake.bb = 0 : ui32, handshake.name = "br7"} : <f32>
    %10 = br %arg3 {handshake.bb = 0 : ui32, handshake.name = "br8"} : <>
    %11 = mux %index [%5, %trueResult_74] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<f32>, <f32>] to <f32>
    %12 = mux %index [%6, %trueResult_76] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i1>, [<f32>, <f32>] to <f32>
    %13 = mux %index [%7, %trueResult_78] {handshake.bb = 1 : ui32, handshake.name = "mux2"} : <i1>, [<i32>, <i32>] to <i32>
    %14 = mux %index [%8, %trueResult_80] {handshake.bb = 1 : ui32, handshake.name = "mux3"} : <i1>, [<f32>, <f32>] to <f32>
    %15 = mux %index [%9, %trueResult_82] {handshake.bb = 1 : ui32, handshake.name = "mux4"} : <i1>, [<f32>, <f32>] to <f32>
    %result, %index = control_merge [%10, %trueResult_84]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %16 = br %11 {handshake.bb = 1 : ui32, handshake.name = "br9"} : <f32>
    %17 = br %12 {handshake.bb = 1 : ui32, handshake.name = "br10"} : <f32>
    %18 = br %13 {handshake.bb = 1 : ui32, handshake.name = "br11"} : <i32>
    %19 = br %14 {handshake.bb = 1 : ui32, handshake.name = "br12"} : <f32>
    %20 = br %15 {handshake.bb = 1 : ui32, handshake.name = "br13"} : <f32>
    %21 = br %result {handshake.bb = 1 : ui32, handshake.name = "br14"} : <>
    %22 = mux %index_1 [%16, %trueResult_58] {handshake.bb = 2 : ui32, handshake.name = "mux5"} : <i1>, [<f32>, <f32>] to <f32>
    %23 = mux %index_1 [%17, %trueResult_60] {handshake.bb = 2 : ui32, handshake.name = "mux6"} : <i1>, [<f32>, <f32>] to <f32>
    %24 = mux %index_1 [%18, %trueResult_62] {handshake.bb = 2 : ui32, handshake.name = "mux7"} : <i1>, [<i32>, <i32>] to <i32>
    %25 = mux %index_1 [%19, %trueResult_64] {handshake.bb = 2 : ui32, handshake.name = "mux8"} : <i1>, [<f32>, <f32>] to <f32>
    %26 = mux %index_1 [%20, %trueResult_66] {handshake.bb = 2 : ui32, handshake.name = "mux9"} : <i1>, [<f32>, <f32>] to <f32>
    %result_0, %index_1 = control_merge [%21, %trueResult_68]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>, <>] to <>, <i1>
    %27 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %28 = constant %27 {handshake.bb = 2 : ui32, handshake.name = "constant6", value = -2.000000e+00 : f32} : <>, <f32>
    %29 = source {handshake.bb = 2 : ui32, handshake.name = "source2"} : <>
    %30 = constant %29 {handshake.bb = 2 : ui32, handshake.name = "constant7", value = 5.000000e-01 : f32} : <>, <f32>
    %31 = addf %22, %23 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "addf1"} : <f32>
    %32 = mulf %31, %30 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "mulf1"} : <f32>
    %33 = mulf %32, %32 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "mulf2"} : <f32>
    %34 = addf %33, %28 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "addf2"} : <f32>
    %35 = absf %34 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "absf0"} : <f32> to <f32>
    %36 = cmpf olt, %35, %26 {handshake.bb = 2 : ui32, handshake.name = "cmpf0"} : <f32>
    %trueResult, %falseResult = cond_br %36, %32 {handshake.bb = 2 : ui32, handshake.name = "cond_br5"} : <i1>, <f32>
    %trueResult_2, %falseResult_3 = cond_br %36, %result_0 {handshake.bb = 2 : ui32, handshake.name = "cond_br6"} : <i1>, <>
    %trueResult_4, %falseResult_5 = cond_br %36, %26 {handshake.bb = 2 : ui32, handshake.name = "cond_br7"} : <i1>, <f32>
    %trueResult_6, %falseResult_7 = cond_br %36, %22 {handshake.bb = 2 : ui32, handshake.name = "cond_br8"} : <i1>, <f32>
    %trueResult_8, %falseResult_9 = cond_br %36, %23 {handshake.bb = 2 : ui32, handshake.name = "cond_br9"} : <i1>, <f32>
    %trueResult_10, %falseResult_11 = cond_br %36, %24 {handshake.bb = 2 : ui32, handshake.name = "cond_br10"} : <i1>, <i32>
    %trueResult_12, %falseResult_13 = cond_br %36, %25 {handshake.bb = 2 : ui32, handshake.name = "cond_br11"} : <i1>, <f32>
    %trueResult_14, %falseResult_15 = cond_br %36, %34 {handshake.bb = 2 : ui32, handshake.name = "cond_br12"} : <i1>, <f32>
    %37 = merge %falseResult_5 {handshake.bb = 3 : ui32, handshake.name = "merge0"} : <f32>
    %38 = merge %falseResult_7 {handshake.bb = 3 : ui32, handshake.name = "merge1"} : <f32>
    %39 = merge %falseResult_9 {handshake.bb = 3 : ui32, handshake.name = "merge2"} : <f32>
    %40 = merge %falseResult_11 {handshake.bb = 3 : ui32, handshake.name = "merge3"} : <i32>
    %41 = merge %falseResult_13 {handshake.bb = 3 : ui32, handshake.name = "merge4"} : <f32>
    %42 = merge %falseResult {handshake.bb = 3 : ui32, handshake.name = "merge5"} : <f32>
    %43 = merge %falseResult_15 {handshake.bb = 3 : ui32, handshake.name = "merge6"} : <f32>
    %result_16, %index_17 = control_merge [%falseResult_3]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>] to <>, <i1>
    %44 = source {handshake.bb = 3 : ui32, handshake.name = "source3"} : <>
    %45 = constant %44 {handshake.bb = 3 : ui32, handshake.name = "constant8", value = 5.000000e-01 : f32} : <>, <f32>
    %46 = subf %39, %38 {fastmath = #arith.fastmath<none>, handshake.bb = 3 : ui32, handshake.name = "subf0"} : <f32>
    %47 = mulf %46, %45 {fastmath = #arith.fastmath<none>, handshake.bb = 3 : ui32, handshake.name = "mulf3"} : <f32>
    %48 = cmpf olt, %47, %37 {handshake.bb = 3 : ui32, handshake.name = "cmpf1"} : <f32>
    %trueResult_18, %falseResult_19 = cond_br %48, %42 {handshake.bb = 3 : ui32, handshake.name = "cond_br13"} : <i1>, <f32>
    %trueResult_20, %falseResult_21 = cond_br %48, %result_16 {handshake.bb = 3 : ui32, handshake.name = "cond_br14"} : <i1>, <>
    %trueResult_22, %falseResult_23 = cond_br %48, %37 {handshake.bb = 3 : ui32, handshake.name = "cond_br15"} : <i1>, <f32>
    %trueResult_24, %falseResult_25 = cond_br %48, %38 {handshake.bb = 3 : ui32, handshake.name = "cond_br16"} : <i1>, <f32>
    %trueResult_26, %falseResult_27 = cond_br %48, %39 {handshake.bb = 3 : ui32, handshake.name = "cond_br17"} : <i1>, <f32>
    %trueResult_28, %falseResult_29 = cond_br %48, %40 {handshake.bb = 3 : ui32, handshake.name = "cond_br18"} : <i1>, <i32>
    %trueResult_30, %falseResult_31 = cond_br %48, %41 {handshake.bb = 3 : ui32, handshake.name = "cond_br19"} : <i1>, <f32>
    %trueResult_32, %falseResult_33 = cond_br %48, %43 {handshake.bb = 3 : ui32, handshake.name = "cond_br20"} : <i1>, <f32>
    %49 = merge %falseResult_23 {handshake.bb = 4 : ui32, handshake.name = "merge7"} : <f32>
    %50 = merge %falseResult_25 {handshake.bb = 4 : ui32, handshake.name = "merge8"} : <f32>
    %51 = merge %falseResult_27 {handshake.bb = 4 : ui32, handshake.name = "merge9"} : <f32>
    %52 = merge %falseResult_29 {handshake.bb = 4 : ui32, handshake.name = "merge10"} : <i32>
    %53 = merge %falseResult_31 {handshake.bb = 4 : ui32, handshake.name = "merge11"} : <f32>
    %54 = merge %falseResult_19 {handshake.bb = 4 : ui32, handshake.name = "merge12"} : <f32>
    %55 = merge %falseResult_33 {handshake.bb = 4 : ui32, handshake.name = "merge13"} : <f32>
    %result_34, %index_35 = control_merge [%falseResult_21]  {handshake.bb = 4 : ui32, handshake.name = "control_merge3"} : [<>] to <>, <i1>
    %56 = constant %result_34 {handshake.bb = 4 : ui32, handshake.name = "constant9", value = 0.000000e+00 : f32} : <>, <f32>
    %57 = source {handshake.bb = 4 : ui32, handshake.name = "source4"} : <>
    %58 = constant %57 {handshake.bb = 4 : ui32, handshake.name = "constant10", value = 1 : i32} : <>, <i32>
    %59 = source {handshake.bb = 4 : ui32, handshake.name = "source5"} : <>
    %60 = constant %59 {handshake.bb = 4 : ui32, handshake.name = "constant11", value = 100 : i32} : <>, <i32>
    %61 = mulf %53, %55 {fastmath = #arith.fastmath<none>, handshake.bb = 4 : ui32, handshake.name = "mulf4"} : <f32>
    %62 = cmpf olt, %61, %56 {handshake.bb = 4 : ui32, handshake.name = "cmpf2"} : <f32>
    %63 = addi %52, %58 {handshake.bb = 4 : ui32, handshake.name = "addi0"} : <i32>
    %64 = cmpi ult, %63, %60 {handshake.bb = 4 : ui32, handshake.name = "cmpi0"} : <i32>
    %trueResult_36, %falseResult_37 = cond_br %62, %49 {handshake.bb = 4 : ui32, handshake.name = "cond_br21"} : <i1>, <f32>
    %trueResult_38, %falseResult_39 = cond_br %62, %50 {handshake.bb = 4 : ui32, handshake.name = "cond_br22"} : <i1>, <f32>
    %trueResult_40, %falseResult_41 = cond_br %62, %53 {handshake.bb = 4 : ui32, handshake.name = "cond_br23"} : <i1>, <f32>
    %trueResult_42, %falseResult_43 = cond_br %62, %54 {handshake.bb = 4 : ui32, handshake.name = "cond_br24"} : <i1>, <f32>
    %trueResult_44, %falseResult_45 = cond_br %62, %56 {handshake.bb = 4 : ui32, handshake.name = "cond_br25"} : <i1>, <f32>
    %trueResult_46, %falseResult_47 = cond_br %62, %63 {handshake.bb = 4 : ui32, handshake.name = "cond_br26"} : <i1>, <i32>
    %trueResult_48, %falseResult_49 = cond_br %62, %64 {handshake.bb = 4 : ui32, handshake.name = "cond_br27"} : <i1>, <i1>
    %trueResult_50, %falseResult_51 = cond_br %62, %result_34 {handshake.bb = 4 : ui32, handshake.name = "cond_br28"} : <i1>, <>
    %trueResult_52, %falseResult_53 = cond_br %62, %51 {handshake.bb = 4 : ui32, handshake.name = "cond_br29"} : <i1>, <f32>
    %trueResult_54, %falseResult_55 = cond_br %62, %55 {handshake.bb = 4 : ui32, handshake.name = "cond_br30"} : <i1>, <f32>
    %65 = merge %trueResult_36 {handshake.bb = 5 : ui32, handshake.name = "merge14"} : <f32>
    %66 = merge %trueResult_38 {handshake.bb = 5 : ui32, handshake.name = "merge15"} : <f32>
    %67 = merge %trueResult_40 {handshake.bb = 5 : ui32, handshake.name = "merge16"} : <f32>
    %68 = merge %trueResult_42 {handshake.bb = 5 : ui32, handshake.name = "merge17"} : <f32>
    %69 = merge %trueResult_44 {handshake.bb = 5 : ui32, handshake.name = "merge18"} : <f32>
    %70 = merge %trueResult_46 {handshake.bb = 5 : ui32, handshake.name = "merge19"} : <i32>
    %71 = merge %trueResult_48 {handshake.bb = 5 : ui32, handshake.name = "merge20"} : <i1>
    %result_56, %index_57 = control_merge [%trueResult_50]  {handshake.bb = 5 : ui32, handshake.name = "control_merge4"} : [<>] to <>, <i1>
    %trueResult_58, %falseResult_59 = cond_br %71, %66 {handshake.bb = 5 : ui32, handshake.name = "cond_br31"} : <i1>, <f32>
    %trueResult_60, %falseResult_61 = cond_br %71, %68 {handshake.bb = 5 : ui32, handshake.name = "cond_br32"} : <i1>, <f32>
    %trueResult_62, %falseResult_63 = cond_br %71, %70 {handshake.bb = 5 : ui32, handshake.name = "cond_br33"} : <i1>, <i32>
    %trueResult_64, %falseResult_65 = cond_br %71, %67 {handshake.bb = 5 : ui32, handshake.name = "cond_br34"} : <i1>, <f32>
    %trueResult_66, %falseResult_67 = cond_br %71, %65 {handshake.bb = 5 : ui32, handshake.name = "cond_br35"} : <i1>, <f32>
    %trueResult_68, %falseResult_69 = cond_br %71, %result_56 {handshake.bb = 5 : ui32, handshake.name = "cond_br36"} : <i1>, <>
    %trueResult_70, %falseResult_71 = cond_br %71, %69 {handshake.bb = 5 : ui32, handshake.name = "cond_br37"} : <i1>, <f32>
    %72 = merge %falseResult_37 {handshake.bb = 6 : ui32, handshake.name = "merge21"} : <f32>
    %73 = merge %falseResult_53 {handshake.bb = 6 : ui32, handshake.name = "merge22"} : <f32>
    %74 = merge %falseResult_43 {handshake.bb = 6 : ui32, handshake.name = "merge23"} : <f32>
    %75 = merge %falseResult_55 {handshake.bb = 6 : ui32, handshake.name = "merge24"} : <f32>
    %76 = merge %falseResult_45 {handshake.bb = 6 : ui32, handshake.name = "merge25"} : <f32>
    %77 = merge %falseResult_47 {handshake.bb = 6 : ui32, handshake.name = "merge26"} : <i32>
    %78 = merge %falseResult_49 {handshake.bb = 6 : ui32, handshake.name = "merge27"} : <i1>
    %result_72, %index_73 = control_merge [%falseResult_51]  {handshake.bb = 6 : ui32, handshake.name = "control_merge5"} : [<>] to <>, <i1>
    %trueResult_74, %falseResult_75 = cond_br %78, %74 {handshake.bb = 6 : ui32, handshake.name = "cond_br38"} : <i1>, <f32>
    %trueResult_76, %falseResult_77 = cond_br %78, %73 {handshake.bb = 6 : ui32, handshake.name = "cond_br39"} : <i1>, <f32>
    %trueResult_78, %falseResult_79 = cond_br %78, %77 {handshake.bb = 6 : ui32, handshake.name = "cond_br40"} : <i1>, <i32>
    %trueResult_80, %falseResult_81 = cond_br %78, %75 {handshake.bb = 6 : ui32, handshake.name = "cond_br41"} : <i1>, <f32>
    %trueResult_82, %falseResult_83 = cond_br %78, %72 {handshake.bb = 6 : ui32, handshake.name = "cond_br42"} : <i1>, <f32>
    %trueResult_84, %falseResult_85 = cond_br %78, %result_72 {handshake.bb = 6 : ui32, handshake.name = "cond_br43"} : <i1>, <>
    %trueResult_86, %falseResult_87 = cond_br %78, %76 {handshake.bb = 6 : ui32, handshake.name = "cond_br44"} : <i1>, <f32>
    %79 = mux %index_89 [%trueResult, %falseResult_71, %falseResult_87] {handshake.bb = 7 : ui32, handshake.name = "mux10"} : <i2>, [<f32>, <f32>, <f32>] to <f32>
    %result_88, %index_89 = control_merge [%trueResult_2, %falseResult_69, %falseResult_85]  {handshake.bb = 7 : ui32, handshake.name = "control_merge6"} : [<>, <>, <>] to <>, <i2>
    %80 = br %79 {handshake.bb = 7 : ui32, handshake.name = "br15"} : <f32>
    %81 = br %result_88 {handshake.bb = 7 : ui32, handshake.name = "br16"} : <>
    %82 = mux %index_91 [%trueResult_18, %80] {handshake.bb = 8 : ui32, handshake.name = "mux11"} : <i1>, [<f32>, <f32>] to <f32>
    %result_90, %index_91 = control_merge [%trueResult_20, %81]  {handshake.bb = 8 : ui32, handshake.name = "control_merge7"} : [<>, <>] to <>, <i1>
    end {handshake.bb = 8 : ui32, handshake.name = "end0"} %82, %arg3 : <f32>, <>
  }
}

