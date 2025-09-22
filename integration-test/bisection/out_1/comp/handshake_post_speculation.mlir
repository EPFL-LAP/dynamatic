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
    %14 = mux %78#5 [%8, %97] {handshake.bb = 2 : ui32, handshake.name = "mux5"} : <i1>, [<f32>, <f32>] to <f32>
    %15 = mux %78#4 [%9, %25] {handshake.bb = 2 : ui32, handshake.name = "mux6"} : <i1>, [<f32>, <f32>] to <f32>
    %16 = mux %78#3 [%10, %68] {handshake.bb = 2 : ui32, handshake.name = "mux7"} : <i1>, [<i8>, <i8>] to <i8>
    %17 = mux %78#2 [%11, %100] {handshake.bb = 2 : ui32, handshake.name = "mux8"} : <i1>, [<f32>, <f32>] to <f32>
    %18 = mux %78#1 [%12, %102] {handshake.bb = 2 : ui32, handshake.name = "mux9"} : <i1>, [<f32>, <f32>] to <f32>
    %19 = mux %78#0 [%result, %104] {handshake.bb = 2 : ui32, handshake.name = "mux15"} : <i1>, [<>, <>] to <>
    %20 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %21 = constant %20 {handshake.bb = 2 : ui32, handshake.name = "constant6", value = -2.000000e+00 : f32} : <>, <f32>
    %22 = source {handshake.bb = 2 : ui32, handshake.name = "source2"} : <>
    %23 = constant %22 {handshake.bb = 2 : ui32, handshake.name = "constant7", value = 5.000000e-01 : f32} : <>, <f32>
    %24 = addf %98#2, %99#2 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "addf1"} : <f32>
    %25 = passer %26#3[%76#6] {handshake.bb = 2 : ui32, handshake.name = "passer44"} : <f32>, <i1>
    %26:6 = fork [6] %27 {handshake.bb = 2 : ui32, handshake.name = "fork6"} : <f32>
    %27 = mulf %24, %23 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "mulf1"} : <f32>
    %28 = mulf %26#0, %26#1 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "mulf2"} : <f32>
    %29:3 = fork [3] %30 {handshake.bb = 2 : ui32, handshake.name = "fork7"} : <f32>
    %30 = addf %28, %21 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "addf2"} : <f32>
    %31 = absf %29#2 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "absf0"} : <f32> to <f32>
    %32 = cmpf olt, %31, %103#3 {handshake.bb = 2 : ui32, handshake.name = "cmpf0"} : <f32>
    %33:2 = fork [2] %32 {handshake.bb = 2 : ui32, handshake.name = "fork9"} : <i1>
    %34:2 = fork [2] %35 {handshake.bb = 2 : ui32, handshake.name = "fork13"} : <i1>
    %35 = not %33#0 {handshake.bb = 2 : ui32, handshake.name = "not0"} : <i1>
    %36 = andi %96, %64#0 {handshake.bb = 2 : ui32, handshake.name = "andi0"} : <i1>
    %37 = andi %87, %48#0 {handshake.bb = 2 : ui32, handshake.name = "andi1"} : <i1>
    %38 = andi %45#1, %34#0 {handshake.bb = 2 : ui32, handshake.name = "andi2"} : <i1>
    %39 = passer %26#5[%86#1] {handshake.bb = 2 : ui32, handshake.name = "passer0"} : <f32>, <i1>
    %40 = passer %105#0[%86#0] {handshake.bb = 2 : ui32, handshake.name = "passer2"} : <>, <i1>
    %41 = source {handshake.bb = 2 : ui32, handshake.name = "source3"} : <>
    %42 = constant %41 {handshake.bb = 2 : ui32, handshake.name = "constant8", value = 5.000000e-01 : f32} : <>, <f32>
    %43 = subf %99#1, %98#1 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "subf0"} : <f32>
    %44 = mulf %43, %42 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "mulf3"} : <f32>
    %45:2 = fork [2] %46 {handshake.bb = 2 : ui32, handshake.name = "fork16"} : <i1>
    %46 = cmpf olt, %44, %103#2 {handshake.bb = 2 : ui32, handshake.name = "cmpf1"} : <f32>
    %47 = not %45#0 {handshake.bb = 2 : ui32, handshake.name = "not1"} : <i1>
    %48:2 = fork [2] %49 {handshake.bb = 2 : ui32, handshake.name = "fork17"} : <i1>
    %49 = andi %34#1, %47 {handshake.bb = 2 : ui32, handshake.name = "andi3"} : <i1>
    %50 = passer %26#4[%80#1] {handshake.bb = 2 : ui32, handshake.name = "passer10"} : <f32>, <i1>
    %51 = passer %105#5[%80#0] {handshake.bb = 2 : ui32, handshake.name = "passer12"} : <>, <i1>
    %52 = extsi %16 {handshake.bb = 2 : ui32, handshake.name = "extsi4"} : <i8> to <i9>
    %53:3 = fork [3] %54 {handshake.bb = 2 : ui32, handshake.name = "fork18"} : <f32>
    %54 = constant %105#1 {handshake.bb = 2 : ui32, handshake.name = "constant9", value = 0.000000e+00 : f32} : <>, <f32>
    %55 = source {handshake.bb = 2 : ui32, handshake.name = "source4"} : <>
    %56 = constant %55 {handshake.bb = 2 : ui32, handshake.name = "constant1", value = 1 : i2} : <>, <i2>
    %57 = extsi %56 {handshake.bb = 2 : ui32, handshake.name = "extsi5"} : <i2> to <i9>
    %58 = source {handshake.bb = 2 : ui32, handshake.name = "source5"} : <>
    %59 = constant %58 {handshake.bb = 2 : ui32, handshake.name = "constant2", value = 100 : i8} : <>, <i8>
    %60 = extsi %59 {handshake.bb = 2 : ui32, handshake.name = "extsi6"} : <i8> to <i9>
    %61 = mulf %101#1, %29#1 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "mulf4"} : <f32>
    %62:2 = fork [2] %63 {handshake.bb = 2 : ui32, handshake.name = "fork20"} : <i1>
    %63 = cmpf olt, %61, %53#0 {handshake.bb = 2 : ui32, handshake.name = "cmpf2"} : <f32>
    %64:2 = fork [2] %65 {handshake.bb = 2 : ui32, handshake.name = "fork42"} : <i1>
    %65 = andi %48#1, %62#0 {handshake.bb = 2 : ui32, handshake.name = "andi4"} : <i1>
    %66:2 = fork [2] %67 {handshake.bb = 2 : ui32, handshake.name = "fork43"} : <i9>
    %67 = addi %52, %57 {handshake.bb = 2 : ui32, handshake.name = "addi0"} : <i9>
    %68 = passer %69#1[%76#3] {handshake.bb = 2 : ui32, handshake.name = "passer45"} : <i8>, <i1>
    %69:2 = fork [2] %70 {handshake.bb = 2 : ui32, handshake.name = "fork44"} : <i8>
    %70 = trunci %66#0 {handshake.bb = 2 : ui32, handshake.name = "trunci0"} : <i9> to <i8>
    %71:3 = fork [3] %72 {handshake.bb = 2 : ui32, handshake.name = "fork45"} : <i1>
    %72 = cmpi ult, %66#1, %60 {handshake.bb = 2 : ui32, handshake.name = "cmpi0"} : <i9>
    %73 = passer %74[%76#4] {handshake.bb = 2 : ui32, handshake.name = "passer46"} : <i1>, <i1>
    %74 = andi %64#1, %71#0 {handshake.bb = 2 : ui32, handshake.name = "andi5"} : <i1>
    %75 = spec_v2_repeating_init %73 {handshake.bb = 2 : ui32, handshake.name = "spec_v2_repeating_init0", initToken = 1 : ui1, specv2_top_ri = true} : <i1>
    %76:12 = fork [12] %75 {handshake.bb = 2 : ui32, handshake.name = "fork3"} : <i1>
    %77 = init %76#11 {handshake.bb = 2 : ui32, handshake.name = "init6", initToken = 0 : ui1} : <i1>
    %78:6 = fork [6] %77 {handshake.bb = 2 : ui32, handshake.name = "fork4"} : <i1>
    %79 = andi %38, %76#10 {handshake.bb = 2 : ui32, handshake.name = "andi6"} : <i1>
    %80:2 = fork [2] %79 {handshake.bb = 2 : ui32, handshake.name = "fork47"} : <i1>
    %81 = andi %37, %76#9 {handshake.bb = 2 : ui32, handshake.name = "andi7"} : <i1>
    %82:8 = fork [8] %81 {handshake.bb = 2 : ui32, handshake.name = "fork48"} : <i1>
    %83 = andi %36, %76#8 {handshake.bb = 2 : ui32, handshake.name = "andi8"} : <i1>
    %84:2 = fork [2] %83 {handshake.bb = 2 : ui32, handshake.name = "fork49"} : <i1>
    %85 = andi %33#1, %76#7 {handshake.bb = 2 : ui32, handshake.name = "andi9"} : <i1>
    %86:2 = fork [2] %85 {handshake.bb = 2 : ui32, handshake.name = "fork50"} : <i1>
    %87 = not %62#1 {handshake.bb = 2 : ui32, handshake.name = "not2"} : <i1>
    %88 = passer %103#0[%82#0] {handshake.bb = 2 : ui32, handshake.name = "passer21"} : <f32>, <i1>
    %89 = passer %26#2[%82#1] {handshake.bb = 2 : ui32, handshake.name = "passer25"} : <f32>, <i1>
    %90 = passer %53#1[%82#7] {handshake.bb = 2 : ui32, handshake.name = "passer27"} : <f32>, <i1>
    %91 = passer %69#0[%82#6] {handshake.bb = 2 : ui32, handshake.name = "passer29"} : <i8>, <i1>
    %92 = passer %71#1[%82#4] {handshake.bb = 2 : ui32, handshake.name = "passer31"} : <i1>, <i1>
    %93 = passer %105#2[%82#5] {handshake.bb = 2 : ui32, handshake.name = "passer33"} : <>, <i1>
    %94 = passer %99#0[%82#2] {handshake.bb = 2 : ui32, handshake.name = "passer34"} : <f32>, <i1>
    %95 = passer %29#0[%82#3] {handshake.bb = 2 : ui32, handshake.name = "passer35"} : <f32>, <i1>
    %96 = not %71#2 {handshake.bb = 2 : ui32, handshake.name = "not3"} : <i1>
    %97 = passer %98#0[%76#5] {handshake.bb = 2 : ui32, handshake.name = "passer47"} : <f32>, <i1>
    %98:3 = fork [3] %14 {handshake.bb = 2 : ui32, handshake.name = "fork51"} : <f32>
    %99:3 = fork [3] %15 {handshake.bb = 2 : ui32, handshake.name = "fork52"} : <f32>
    %100 = passer %101#0[%76#2] {handshake.bb = 2 : ui32, handshake.name = "passer48"} : <f32>, <i1>
    %101:2 = fork [2] %17 {handshake.bb = 2 : ui32, handshake.name = "fork53"} : <f32>
    %102 = passer %103#1[%76#1] {handshake.bb = 2 : ui32, handshake.name = "passer49"} : <f32>, <i1>
    %103:4 = fork [4] %18 {handshake.bb = 2 : ui32, handshake.name = "fork54"} : <f32>
    %104 = passer %105#4[%76#0] {handshake.bb = 2 : ui32, handshake.name = "passer50"} : <>, <i1>
    %105:6 = fork [6] %19 {handshake.bb = 2 : ui32, handshake.name = "fork55"} : <>
    %106 = passer %105#3[%84#1] {handshake.bb = 2 : ui32, handshake.name = "passer42"} : <>, <i1>
    %107 = passer %53#2[%84#0] {handshake.bb = 2 : ui32, handshake.name = "passer43"} : <f32>, <i1>
    %108:7 = fork [7] %92 {handshake.bb = 3 : ui32, handshake.name = "fork21"} : <i1>
    %trueResult, %falseResult = cond_br %108#6, %89 {handshake.bb = 3 : ui32, handshake.name = "cond_br38"} : <i1>, <f32>
    sink %falseResult {handshake.name = "sink26"} : <f32>
    %trueResult_0, %falseResult_1 = cond_br %108#5, %94 {handshake.bb = 3 : ui32, handshake.name = "cond_br39"} : <i1>, <f32>
    sink %falseResult_1 {handshake.name = "sink27"} : <f32>
    %trueResult_2, %falseResult_3 = cond_br %108#0, %91 {handshake.bb = 3 : ui32, handshake.name = "cond_br40"} : <i1>, <i8>
    sink %falseResult_3 {handshake.name = "sink28"} : <i8>
    %trueResult_4, %falseResult_5 = cond_br %108#4, %95 {handshake.bb = 3 : ui32, handshake.name = "cond_br41"} : <i1>, <f32>
    sink %falseResult_5 {handshake.name = "sink29"} : <f32>
    %trueResult_6, %falseResult_7 = cond_br %108#3, %88 {handshake.bb = 3 : ui32, handshake.name = "cond_br42"} : <i1>, <f32>
    sink %falseResult_7 {handshake.name = "sink30"} : <f32>
    %trueResult_8, %falseResult_9 = cond_br %108#2, %93 {handshake.bb = 3 : ui32, handshake.name = "cond_br43"} : <i1>, <>
    %trueResult_10, %falseResult_11 = cond_br %108#1, %90 {handshake.bb = 3 : ui32, handshake.name = "cond_br44"} : <i1>, <f32>
    sink %trueResult_10 {handshake.name = "sink31"} : <f32>
    %109 = mux %index_13 [%39, %107, %falseResult_11] {handshake.bb = 4 : ui32, handshake.name = "mux10"} : <i2>, [<f32>, <f32>, <f32>] to <f32>
    %result_12, %index_13 = control_merge [%40, %106, %falseResult_9]  {handshake.bb = 4 : ui32, handshake.name = "control_merge6"} : [<>, <>, <>] to <>, <i2>
    %110 = mux %index_15 [%50, %109] {handshake.bb = 5 : ui32, handshake.name = "mux11"} : <i1>, [<f32>, <f32>] to <f32>
    %result_14, %index_15 = control_merge [%51, %result_12]  {handshake.bb = 5 : ui32, handshake.name = "control_merge7"} : [<>, <>] to <>, <i1>
    sink %result_14 {handshake.name = "sink32"} : <>
    end {handshake.bb = 5 : ui32, handshake.name = "end0"} %110, %0#1 : <f32>, <>
  }
}

