module {
  handshake.func @bisection(%arg0: !handshake.channel<f32>, %arg1: !handshake.channel<f32>, %arg2: !handshake.channel<f32>, %arg3: !handshake.control<>, ...) -> (!handshake.channel<f32>, !handshake.control<>) attributes {argNames = ["a", "b", "tol", "start"], resNames = ["out0", "end"]} {
    %0:3 = fork [3] %arg3 {handshake.bb = 0 : ui32, handshake.name = "fork0"} : <>
    %1:3 = fork [3] %arg0 {handshake.bb = 0 : ui32, handshake.name = "fork1"} : <f32>
    %2 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant0", value = false} : <>, <i1>
    %3 = source {handshake.bb = 0 : ui32, handshake.name = "source0"} : <>
    %4 = constant %3 {handshake.bb = 0 : ui32, handshake.name = "constant5", value = -2.000000e+00 : f32} : <>, <f32>
    %5 = mulf %1#1, %1#2 {fastmath = #arith.fastmath<none>, handshake.bb = 0 : ui32, handshake.name = "mulf0"} : <f32>
    %6 = addf %5, %4 {fastmath = #arith.fastmath<none>, handshake.bb = 0 : ui32, handshake.name = "addf0"} : <f32>
    %7 = extsi %2 {handshake.bb = 0 : ui32, handshake.name = "extsi3"} : <i1> to <i8>
    %8 = mux %13#1 [%1#0, %trueResult] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<f32>, <f32>] to <f32>
    %9 = mux %13#2 [%arg1, %trueResult_0] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i1>, [<f32>, <f32>] to <f32>
    %10 = mux %13#0 [%7, %trueResult_2] {handshake.bb = 1 : ui32, handshake.name = "mux2"} : <i1>, [<i8>, <i8>] to <i8>
    %11 = mux %13#3 [%6, %trueResult_4] {handshake.bb = 1 : ui32, handshake.name = "mux3"} : <i1>, [<f32>, <f32>] to <f32>
    %12 = mux %13#4 [%arg2, %trueResult_6] {handshake.bb = 1 : ui32, handshake.name = "mux4"} : <i1>, [<f32>, <f32>] to <f32>
    %result, %index = control_merge [%0#2, %trueResult_8]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %13:5 = fork [5] %index {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <i1>
    %14 = init %88#7 {handshake.bb = 2 : ui32, handshake.name = "init0", initToken = 0 : ui1} : <i1>
    %15 = mux %14 [%8, %109] {handshake.bb = 2 : ui32, handshake.name = "mux5"} : <i1>, [<f32>, <f32>] to <f32>
    %16 = init %88#8 {handshake.bb = 2 : ui32, handshake.name = "init1", initToken = 0 : ui1} : <i1>
    %17 = mux %16 [%9, %32] {handshake.bb = 2 : ui32, handshake.name = "mux6"} : <i1>, [<f32>, <f32>] to <f32>
    %18 = init %88#9 {handshake.bb = 2 : ui32, handshake.name = "init2", initToken = 0 : ui1} : <i1>
    %19 = mux %18 [%10, %75] {handshake.bb = 2 : ui32, handshake.name = "mux7"} : <i1>, [<i8>, <i8>] to <i8>
    %20 = init %88#10 {handshake.bb = 2 : ui32, handshake.name = "init3", initToken = 0 : ui1} : <i1>
    %21 = mux %20 [%11, %111] {handshake.bb = 2 : ui32, handshake.name = "mux8"} : <i1>, [<f32>, <f32>] to <f32>
    %22 = init %88#11 {handshake.bb = 2 : ui32, handshake.name = "init4", initToken = 0 : ui1} : <i1>
    %23 = init %88#12 {handshake.bb = 2 : ui32, handshake.name = "init5", initToken = 0 : ui1} : <i1>
    %24 = mux %22 [%12, %113] {handshake.bb = 2 : ui32, handshake.name = "mux9"} : <i1>, [<f32>, <f32>] to <f32>
    %25 = mux %23 [%result, %115] {handshake.bb = 2 : ui32, handshake.name = "mux15"} : <i1>, [<>, <>] to <>
    %26 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %27 = constant %26 {handshake.bb = 2 : ui32, handshake.name = "constant6", value = -2.000000e+00 : f32} : <>, <f32>
    %28 = source {handshake.bb = 2 : ui32, handshake.name = "source2"} : <>
    %29 = constant %28 {handshake.bb = 2 : ui32, handshake.name = "constant7", value = 5.000000e-01 : f32} : <>, <f32>
    %30 = addf %110#2, %31#2 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "addf1"} : <f32>
    %31:3 = fork [3] %17 {handshake.bb = 2 : ui32, handshake.name = "fork6"} : <f32>
    %32 = passer %33#3[%88#3] {handshake.bb = 2 : ui32, handshake.name = "passer44"} : <f32>, <i1>
    %33:6 = fork [6] %34 {handshake.bb = 2 : ui32, handshake.name = "fork7"} : <f32>
    %34 = mulf %30, %29 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "mulf1"} : <f32>
    %35 = mulf %33#0, %33#1 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "mulf2"} : <f32>
    %36:3 = fork [3] %37 {handshake.bb = 2 : ui32, handshake.name = "fork9"} : <f32>
    %37 = addf %35, %27 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "addf2"} : <f32>
    %38 = absf %36#2 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "absf0"} : <f32> to <f32>
    %39 = cmpf olt, %38, %114#3 {handshake.bb = 2 : ui32, handshake.name = "cmpf0"} : <f32>
    %40:2 = fork [2] %39 {handshake.bb = 2 : ui32, handshake.name = "fork13"} : <i1>
    %41:2 = fork [2] %42 {handshake.bb = 2 : ui32, handshake.name = "fork16"} : <i1>
    %42 = not %40#0 {handshake.bb = 2 : ui32, handshake.name = "not0"} : <i1>
    %43 = andi %99, %55#0 {handshake.bb = 2 : ui32, handshake.name = "andi0"} : <i1>
    %44 = andi %108, %71#0 {handshake.bb = 2 : ui32, handshake.name = "andi1"} : <i1>
    %45 = andi %52#1, %41#0 {handshake.bb = 2 : ui32, handshake.name = "andi2"} : <i1>
    %46 = passer %33#5[%98#1] {handshake.bb = 2 : ui32, handshake.name = "passer0"} : <f32>, <i1>
    %47 = passer %116#0[%98#0] {handshake.bb = 2 : ui32, handshake.name = "passer2"} : <>, <i1>
    %48 = source {handshake.bb = 2 : ui32, handshake.name = "source3"} : <>
    %49 = constant %48 {handshake.bb = 2 : ui32, handshake.name = "constant8", value = 5.000000e-01 : f32} : <>, <f32>
    %50 = subf %31#1, %110#1 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "subf0"} : <f32>
    %51 = mulf %50, %49 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "mulf3"} : <f32>
    %52:2 = fork [2] %53 {handshake.bb = 2 : ui32, handshake.name = "fork17"} : <i1>
    %53 = cmpf olt, %51, %114#2 {handshake.bb = 2 : ui32, handshake.name = "cmpf1"} : <f32>
    %54 = not %52#0 {handshake.bb = 2 : ui32, handshake.name = "not1"} : <i1>
    %55:2 = fork [2] %56 {handshake.bb = 2 : ui32, handshake.name = "fork18"} : <i1>
    %56 = andi %41#1, %54 {handshake.bb = 2 : ui32, handshake.name = "andi3"} : <i1>
    %57 = passer %33#4[%92#0] {handshake.bb = 2 : ui32, handshake.name = "passer10"} : <f32>, <i1>
    %58 = passer %116#5[%92#1] {handshake.bb = 2 : ui32, handshake.name = "passer12"} : <>, <i1>
    %59 = extsi %19 {handshake.bb = 2 : ui32, handshake.name = "extsi4"} : <i8> to <i9>
    %60:3 = fork [3] %61 {handshake.bb = 2 : ui32, handshake.name = "fork20"} : <f32>
    %61 = constant %116#1 {handshake.bb = 2 : ui32, handshake.name = "constant9", value = 0.000000e+00 : f32} : <>, <f32>
    %62 = source {handshake.bb = 2 : ui32, handshake.name = "source4"} : <>
    %63 = constant %62 {handshake.bb = 2 : ui32, handshake.name = "constant1", value = 1 : i2} : <>, <i2>
    %64 = extsi %63 {handshake.bb = 2 : ui32, handshake.name = "extsi5"} : <i2> to <i9>
    %65 = source {handshake.bb = 2 : ui32, handshake.name = "source5"} : <>
    %66 = constant %65 {handshake.bb = 2 : ui32, handshake.name = "constant2", value = 100 : i8} : <>, <i8>
    %67 = extsi %66 {handshake.bb = 2 : ui32, handshake.name = "extsi6"} : <i8> to <i9>
    %68 = mulf %112#1, %36#1 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "mulf4"} : <f32>
    %69:2 = fork [2] %70 {handshake.bb = 2 : ui32, handshake.name = "fork42"} : <i1>
    %70 = cmpf olt, %68, %60#0 {handshake.bb = 2 : ui32, handshake.name = "cmpf2"} : <f32>
    %71:2 = fork [2] %72 {handshake.bb = 2 : ui32, handshake.name = "fork43"} : <i1>
    %72 = andi %55#1, %69#0 {handshake.bb = 2 : ui32, handshake.name = "andi4"} : <i1>
    %73:2 = fork [2] %74 {handshake.bb = 2 : ui32, handshake.name = "fork44"} : <i9>
    %74 = addi %59, %64 {handshake.bb = 2 : ui32, handshake.name = "addi0"} : <i9>
    %75 = passer %76#1[%88#4] {handshake.bb = 2 : ui32, handshake.name = "passer45"} : <i8>, <i1>
    %76:2 = fork [2] %77 {handshake.bb = 2 : ui32, handshake.name = "fork45"} : <i8>
    %77 = trunci %73#0 {handshake.bb = 2 : ui32, handshake.name = "trunci0"} : <i9> to <i8>
    %78:3 = fork [3] %79 {handshake.bb = 2 : ui32, handshake.name = "fork46"} : <i1>
    %79 = cmpi ult, %73#1, %67 {handshake.bb = 2 : ui32, handshake.name = "cmpi0"} : <i9>
    %80 = passer %81[%90#4] {handshake.bb = 2 : ui32, handshake.name = "passer46"} : <i1>, <i1>
    %81 = andi %71#1, %78#0 {handshake.bb = 2 : ui32, handshake.name = "andi5"} : <i1>
    %82 = spec_v2_repeating_init %80 {handshake.bb = 2 : ui32, handshake.name = "spec_v2_repeating_init0", initToken = 1 : ui1, specv2_top_ri = true} : <i1>
    %83:2 = fork [2] %82 {handshake.bb = 2 : ui32, handshake.name = "fork47"} : <i1>
    %84 = spec_v2_repeating_init %83#0 {handshake.bb = 2 : ui32, handshake.name = "spec_v2_repeating_init1", initToken = 1 : ui1} : <i1>
    %85:2 = fork [2] %84 {handshake.bb = 2 : ui32, handshake.name = "fork48"} : <i1>
    %86 = andi %83#1, %85#0 {handshake.bb = 2 : ui32, handshake.name = "andi6", specv2_tmp_and = true} : <i1>
    %87 = spec_v2_repeating_init %85#1 {handshake.bb = 2 : ui32, handshake.name = "spec_v2_repeating_init2", initToken = 1 : ui1} : <i1>
    %88:13 = fork [13] %87 {handshake.bb = 2 : ui32, handshake.name = "fork49"} : <i1>
    %89 = andi %86, %88#0 {handshake.bb = 2 : ui32, handshake.name = "andi7", specv2_tmp_and = true} : <i1>
    %90:5 = fork [5] %89 {handshake.bb = 2 : ui32, handshake.name = "fork50"} : <i1>
    %91 = andi %45, %90#0 {handshake.bb = 2 : ui32, handshake.name = "andi8"} : <i1>
    %92:2 = fork [2] %91 {handshake.bb = 2 : ui32, handshake.name = "fork51"} : <i1>
    %93 = andi %44, %90#1 {handshake.bb = 2 : ui32, handshake.name = "andi9"} : <i1>
    %94:2 = fork [2] %93 {handshake.bb = 2 : ui32, handshake.name = "fork52"} : <i1>
    %95 = andi %43, %90#2 {handshake.bb = 2 : ui32, handshake.name = "andi10"} : <i1>
    %96:8 = fork [8] %95 {handshake.bb = 2 : ui32, handshake.name = "fork53"} : <i1>
    %97 = andi %40#1, %90#3 {handshake.bb = 2 : ui32, handshake.name = "andi11"} : <i1>
    %98:2 = fork [2] %97 {handshake.bb = 2 : ui32, handshake.name = "fork54"} : <i1>
    %99 = not %69#1 {handshake.bb = 2 : ui32, handshake.name = "not2"} : <i1>
    %100 = passer %114#0[%96#7] {handshake.bb = 2 : ui32, handshake.name = "passer21"} : <f32>, <i1>
    %101 = passer %33#2[%96#3] {handshake.bb = 2 : ui32, handshake.name = "passer25"} : <f32>, <i1>
    %102 = passer %60#1[%96#1] {handshake.bb = 2 : ui32, handshake.name = "passer27"} : <f32>, <i1>
    %103 = passer %76#0[%96#5] {handshake.bb = 2 : ui32, handshake.name = "passer29"} : <i8>, <i1>
    %104 = passer %78#1[%96#6] {handshake.bb = 2 : ui32, handshake.name = "passer31"} : <i1>, <i1>
    %105 = passer %116#2[%96#2] {handshake.bb = 2 : ui32, handshake.name = "passer33"} : <>, <i1>
    %106 = passer %31#0[%96#0] {handshake.bb = 2 : ui32, handshake.name = "passer34"} : <f32>, <i1>
    %107 = passer %36#0[%96#4] {handshake.bb = 2 : ui32, handshake.name = "passer35"} : <f32>, <i1>
    %108 = not %78#2 {handshake.bb = 2 : ui32, handshake.name = "not3"} : <i1>
    %109 = passer %110#0[%88#1] {handshake.bb = 2 : ui32, handshake.name = "passer47"} : <f32>, <i1>
    %110:3 = fork [3] %15 {handshake.bb = 2 : ui32, handshake.name = "fork55"} : <f32>
    %111 = passer %112#0[%88#2] {handshake.bb = 2 : ui32, handshake.name = "passer48"} : <f32>, <i1>
    %112:2 = fork [2] %21 {handshake.bb = 2 : ui32, handshake.name = "fork56"} : <f32>
    %113 = passer %114#1[%88#6] {handshake.bb = 2 : ui32, handshake.name = "passer49"} : <f32>, <i1>
    %114:4 = fork [4] %24 {handshake.bb = 2 : ui32, handshake.name = "fork57"} : <f32>
    %115 = passer %116#4[%88#5] {handshake.bb = 2 : ui32, handshake.name = "passer50"} : <>, <i1>
    %116:6 = fork [6] %25 {handshake.bb = 2 : ui32, handshake.name = "fork58"} : <>
    %117 = passer %116#3[%94#1] {handshake.bb = 2 : ui32, handshake.name = "passer42"} : <>, <i1>
    %118 = passer %60#2[%94#0] {handshake.bb = 2 : ui32, handshake.name = "passer43"} : <f32>, <i1>
    %119:7 = fork [7] %104 {handshake.bb = 3 : ui32, handshake.name = "fork21"} : <i1>
    %trueResult, %falseResult = cond_br %119#6, %101 {handshake.bb = 3 : ui32, handshake.name = "cond_br38"} : <i1>, <f32>
    sink %falseResult {handshake.name = "sink26"} : <f32>
    %trueResult_0, %falseResult_1 = cond_br %119#5, %106 {handshake.bb = 3 : ui32, handshake.name = "cond_br39"} : <i1>, <f32>
    sink %falseResult_1 {handshake.name = "sink27"} : <f32>
    %trueResult_2, %falseResult_3 = cond_br %119#0, %103 {handshake.bb = 3 : ui32, handshake.name = "cond_br40"} : <i1>, <i8>
    sink %falseResult_3 {handshake.name = "sink28"} : <i8>
    %trueResult_4, %falseResult_5 = cond_br %119#4, %107 {handshake.bb = 3 : ui32, handshake.name = "cond_br41"} : <i1>, <f32>
    sink %falseResult_5 {handshake.name = "sink29"} : <f32>
    %trueResult_6, %falseResult_7 = cond_br %119#3, %100 {handshake.bb = 3 : ui32, handshake.name = "cond_br42"} : <i1>, <f32>
    sink %falseResult_7 {handshake.name = "sink30"} : <f32>
    %trueResult_8, %falseResult_9 = cond_br %119#2, %105 {handshake.bb = 3 : ui32, handshake.name = "cond_br43"} : <i1>, <>
    %trueResult_10, %falseResult_11 = cond_br %119#1, %102 {handshake.bb = 3 : ui32, handshake.name = "cond_br44"} : <i1>, <f32>
    sink %trueResult_10 {handshake.name = "sink31"} : <f32>
    %120 = mux %index_13 [%46, %118, %falseResult_11] {handshake.bb = 4 : ui32, handshake.name = "mux10"} : <i2>, [<f32>, <f32>, <f32>] to <f32>
    %result_12, %index_13 = control_merge [%47, %117, %falseResult_9]  {handshake.bb = 4 : ui32, handshake.name = "control_merge6"} : [<>, <>, <>] to <>, <i2>
    %121 = mux %index_15 [%57, %120] {handshake.bb = 5 : ui32, handshake.name = "mux11"} : <i1>, [<f32>, <f32>] to <f32>
    %result_14, %index_15 = control_merge [%58, %result_12]  {handshake.bb = 5 : ui32, handshake.name = "control_merge7"} : [<>, <>] to <>, <i1>
    sink %result_14 {handshake.name = "sink32"} : <>
    end {handshake.bb = 5 : ui32, handshake.name = "end0"} %121, %0#1 : <f32>, <>
  }
}

