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
    %14 = init %82#11 {handshake.bb = 2 : ui32, handshake.name = "init0", initToken = 0 : ui1} : <i1>
    %15 = mux %14 [%8, %101] {handshake.bb = 2 : ui32, handshake.name = "mux5"} : <i1>, [<f32>, <f32>] to <f32>
    %16 = init %82#12 {handshake.bb = 2 : ui32, handshake.name = "init1", initToken = 0 : ui1} : <i1>
    %17 = mux %16 [%9, %31] {handshake.bb = 2 : ui32, handshake.name = "mux6"} : <i1>, [<f32>, <f32>] to <f32>
    %18 = init %82#13 {handshake.bb = 2 : ui32, handshake.name = "init2", initToken = 0 : ui1} : <i1>
    %19 = mux %18 [%10, %74] {handshake.bb = 2 : ui32, handshake.name = "mux7"} : <i1>, [<i8>, <i8>] to <i8>
    %20 = init %82#14 {handshake.bb = 2 : ui32, handshake.name = "init3", initToken = 0 : ui1} : <i1>
    %21 = mux %20 [%11, %104] {handshake.bb = 2 : ui32, handshake.name = "mux8"} : <i1>, [<f32>, <f32>] to <f32>
    %22 = init %82#15 {handshake.bb = 2 : ui32, handshake.name = "init4", initToken = 0 : ui1} : <i1>
    %23 = init %82#16 {handshake.bb = 2 : ui32, handshake.name = "init5", initToken = 0 : ui1} : <i1>
    %24 = mux %22 [%12, %106] {handshake.bb = 2 : ui32, handshake.name = "mux9"} : <i1>, [<f32>, <f32>] to <f32>
    %25 = mux %23 [%result, %108] {handshake.bb = 2 : ui32, handshake.name = "mux15"} : <i1>, [<>, <>] to <>
    %26 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %27 = constant %26 {handshake.bb = 2 : ui32, handshake.name = "constant6", value = -2.000000e+00 : f32} : <>, <f32>
    %28 = source {handshake.bb = 2 : ui32, handshake.name = "source2"} : <>
    %29 = constant %28 {handshake.bb = 2 : ui32, handshake.name = "constant7", value = 5.000000e-01 : f32} : <>, <f32>
    %30 = addf %102#2, %103#2 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "addf1"} : <f32>
    %31 = passer %32#3[%82#4] {handshake.bb = 2 : ui32, handshake.name = "passer44"} : <f32>, <i1>
    %32:6 = fork [6] %33 {handshake.bb = 2 : ui32, handshake.name = "fork6"} : <f32>
    %33 = mulf %30, %29 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "mulf1"} : <f32>
    %34 = mulf %32#0, %32#1 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "mulf2"} : <f32>
    %35:3 = fork [3] %36 {handshake.bb = 2 : ui32, handshake.name = "fork7"} : <f32>
    %36 = addf %34, %27 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "addf2"} : <f32>
    %37 = absf %35#2 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "absf0"} : <f32> to <f32>
    %38 = cmpf olt, %37, %107#3 {handshake.bb = 2 : ui32, handshake.name = "cmpf0"} : <f32>
    %39:2 = fork [2] %38 {handshake.bb = 2 : ui32, handshake.name = "fork9"} : <i1>
    %40:2 = fork [2] %41 {handshake.bb = 2 : ui32, handshake.name = "fork13"} : <i1>
    %41 = not %39#0 {handshake.bb = 2 : ui32, handshake.name = "not0"} : <i1>
    %42 = andi %100, %70#0 {handshake.bb = 2 : ui32, handshake.name = "andi0"} : <i1>
    %43 = andi %91, %54#0 {handshake.bb = 2 : ui32, handshake.name = "andi1"} : <i1>
    %44 = andi %51#1, %40#0 {handshake.bb = 2 : ui32, handshake.name = "andi2"} : <i1>
    %45 = passer %32#5[%90#1] {handshake.bb = 2 : ui32, handshake.name = "passer0"} : <f32>, <i1>
    %46 = passer %109#0[%90#0] {handshake.bb = 2 : ui32, handshake.name = "passer2"} : <>, <i1>
    %47 = source {handshake.bb = 2 : ui32, handshake.name = "source3"} : <>
    %48 = constant %47 {handshake.bb = 2 : ui32, handshake.name = "constant8", value = 5.000000e-01 : f32} : <>, <f32>
    %49 = subf %103#1, %102#1 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "subf0"} : <f32>
    %50 = mulf %49, %48 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "mulf3"} : <f32>
    %51:2 = fork [2] %52 {handshake.bb = 2 : ui32, handshake.name = "fork16"} : <i1>
    %52 = cmpf olt, %50, %107#2 {handshake.bb = 2 : ui32, handshake.name = "cmpf1"} : <f32>
    %53 = not %51#0 {handshake.bb = 2 : ui32, handshake.name = "not1"} : <i1>
    %54:2 = fork [2] %55 {handshake.bb = 2 : ui32, handshake.name = "fork17"} : <i1>
    %55 = andi %40#1, %53 {handshake.bb = 2 : ui32, handshake.name = "andi3"} : <i1>
    %56 = passer %32#4[%84#1] {handshake.bb = 2 : ui32, handshake.name = "passer10"} : <f32>, <i1>
    %57 = passer %109#5[%84#0] {handshake.bb = 2 : ui32, handshake.name = "passer12"} : <>, <i1>
    %58 = extsi %19 {handshake.bb = 2 : ui32, handshake.name = "extsi4"} : <i8> to <i9>
    %59:3 = fork [3] %60 {handshake.bb = 2 : ui32, handshake.name = "fork18"} : <f32>
    %60 = constant %109#1 {handshake.bb = 2 : ui32, handshake.name = "constant9", value = 0.000000e+00 : f32} : <>, <f32>
    %61 = source {handshake.bb = 2 : ui32, handshake.name = "source4"} : <>
    %62 = constant %61 {handshake.bb = 2 : ui32, handshake.name = "constant1", value = 1 : i2} : <>, <i2>
    %63 = extsi %62 {handshake.bb = 2 : ui32, handshake.name = "extsi5"} : <i2> to <i9>
    %64 = source {handshake.bb = 2 : ui32, handshake.name = "source5"} : <>
    %65 = constant %64 {handshake.bb = 2 : ui32, handshake.name = "constant2", value = 100 : i8} : <>, <i8>
    %66 = extsi %65 {handshake.bb = 2 : ui32, handshake.name = "extsi6"} : <i8> to <i9>
    %67 = mulf %105#1, %35#1 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "mulf4"} : <f32>
    %68:2 = fork [2] %69 {handshake.bb = 2 : ui32, handshake.name = "fork20"} : <i1>
    %69 = cmpf olt, %67, %59#0 {handshake.bb = 2 : ui32, handshake.name = "cmpf2"} : <f32>
    %70:2 = fork [2] %71 {handshake.bb = 2 : ui32, handshake.name = "fork42"} : <i1>
    %71 = andi %54#1, %68#0 {handshake.bb = 2 : ui32, handshake.name = "andi4"} : <i1>
    %72:2 = fork [2] %73 {handshake.bb = 2 : ui32, handshake.name = "fork43"} : <i9>
    %73 = addi %58, %63 {handshake.bb = 2 : ui32, handshake.name = "addi0"} : <i9>
    %74 = passer %75#1[%82#7] {handshake.bb = 2 : ui32, handshake.name = "passer45"} : <i8>, <i1>
    %75:2 = fork [2] %76 {handshake.bb = 2 : ui32, handshake.name = "fork44"} : <i8>
    %76 = trunci %72#0 {handshake.bb = 2 : ui32, handshake.name = "trunci0"} : <i9> to <i8>
    %77:3 = fork [3] %78 {handshake.bb = 2 : ui32, handshake.name = "fork45"} : <i1>
    %78 = cmpi ult, %72#1, %66 {handshake.bb = 2 : ui32, handshake.name = "cmpi0"} : <i9>
    %79 = passer %80[%82#6] {handshake.bb = 2 : ui32, handshake.name = "passer46"} : <i1>, <i1>
    %80 = andi %70#1, %77#0 {handshake.bb = 2 : ui32, handshake.name = "andi5"} : <i1>
    %81 = spec_v2_repeating_init %79 {handshake.bb = 2 : ui32, handshake.name = "spec_v2_repeating_init0", initToken = 1 : ui1, specv2_top_ri = true} : <i1>
    %82:17 = fork [17] %81 {handshake.bb = 2 : ui32, handshake.name = "fork46"} : <i1>
    %83 = andi %44, %82#0 {handshake.bb = 2 : ui32, handshake.name = "andi6"} : <i1>
    %84:2 = fork [2] %83 {handshake.bb = 2 : ui32, handshake.name = "fork47"} : <i1>
    %85 = andi %43, %82#1 {handshake.bb = 2 : ui32, handshake.name = "andi7"} : <i1>
    %86:8 = fork [8] %85 {handshake.bb = 2 : ui32, handshake.name = "fork48"} : <i1>
    %87 = andi %42, %82#2 {handshake.bb = 2 : ui32, handshake.name = "andi8"} : <i1>
    %88:2 = fork [2] %87 {handshake.bb = 2 : ui32, handshake.name = "fork49"} : <i1>
    %89 = andi %39#1, %82#3 {handshake.bb = 2 : ui32, handshake.name = "andi9"} : <i1>
    %90:2 = fork [2] %89 {handshake.bb = 2 : ui32, handshake.name = "fork50"} : <i1>
    %91 = not %68#1 {handshake.bb = 2 : ui32, handshake.name = "not2"} : <i1>
    %92 = passer %107#0[%86#0] {handshake.bb = 2 : ui32, handshake.name = "passer21"} : <f32>, <i1>
    %93 = passer %32#2[%86#1] {handshake.bb = 2 : ui32, handshake.name = "passer25"} : <f32>, <i1>
    %94 = passer %59#1[%86#7] {handshake.bb = 2 : ui32, handshake.name = "passer27"} : <f32>, <i1>
    %95 = passer %75#0[%86#6] {handshake.bb = 2 : ui32, handshake.name = "passer29"} : <i8>, <i1>
    %96 = passer %77#1[%86#4] {handshake.bb = 2 : ui32, handshake.name = "passer31"} : <i1>, <i1>
    %97 = passer %109#2[%86#5] {handshake.bb = 2 : ui32, handshake.name = "passer33"} : <>, <i1>
    %98 = passer %103#0[%86#2] {handshake.bb = 2 : ui32, handshake.name = "passer34"} : <f32>, <i1>
    %99 = passer %35#0[%86#3] {handshake.bb = 2 : ui32, handshake.name = "passer35"} : <f32>, <i1>
    %100 = not %77#2 {handshake.bb = 2 : ui32, handshake.name = "not3"} : <i1>
    %101 = passer %102#0[%82#5] {handshake.bb = 2 : ui32, handshake.name = "passer47"} : <f32>, <i1>
    %102:3 = fork [3] %15 {handshake.bb = 2 : ui32, handshake.name = "fork51"} : <f32>
    %103:3 = fork [3] %17 {handshake.bb = 2 : ui32, handshake.name = "fork52"} : <f32>
    %104 = passer %105#0[%82#8] {handshake.bb = 2 : ui32, handshake.name = "passer48"} : <f32>, <i1>
    %105:2 = fork [2] %21 {handshake.bb = 2 : ui32, handshake.name = "fork53"} : <f32>
    %106 = passer %107#1[%82#9] {handshake.bb = 2 : ui32, handshake.name = "passer49"} : <f32>, <i1>
    %107:4 = fork [4] %24 {handshake.bb = 2 : ui32, handshake.name = "fork54"} : <f32>
    %108 = passer %109#4[%82#10] {handshake.bb = 2 : ui32, handshake.name = "passer50"} : <>, <i1>
    %109:6 = fork [6] %25 {handshake.bb = 2 : ui32, handshake.name = "fork55"} : <>
    %110 = passer %109#3[%88#1] {handshake.bb = 2 : ui32, handshake.name = "passer42"} : <>, <i1>
    %111 = passer %59#2[%88#0] {handshake.bb = 2 : ui32, handshake.name = "passer43"} : <f32>, <i1>
    %112:7 = fork [7] %96 {handshake.bb = 3 : ui32, handshake.name = "fork21"} : <i1>
    %trueResult, %falseResult = cond_br %112#6, %93 {handshake.bb = 3 : ui32, handshake.name = "cond_br38"} : <i1>, <f32>
    sink %falseResult {handshake.name = "sink26"} : <f32>
    %trueResult_0, %falseResult_1 = cond_br %112#5, %98 {handshake.bb = 3 : ui32, handshake.name = "cond_br39"} : <i1>, <f32>
    sink %falseResult_1 {handshake.name = "sink27"} : <f32>
    %trueResult_2, %falseResult_3 = cond_br %112#0, %95 {handshake.bb = 3 : ui32, handshake.name = "cond_br40"} : <i1>, <i8>
    sink %falseResult_3 {handshake.name = "sink28"} : <i8>
    %trueResult_4, %falseResult_5 = cond_br %112#4, %99 {handshake.bb = 3 : ui32, handshake.name = "cond_br41"} : <i1>, <f32>
    sink %falseResult_5 {handshake.name = "sink29"} : <f32>
    %trueResult_6, %falseResult_7 = cond_br %112#3, %92 {handshake.bb = 3 : ui32, handshake.name = "cond_br42"} : <i1>, <f32>
    sink %falseResult_7 {handshake.name = "sink30"} : <f32>
    %trueResult_8, %falseResult_9 = cond_br %112#2, %97 {handshake.bb = 3 : ui32, handshake.name = "cond_br43"} : <i1>, <>
    %trueResult_10, %falseResult_11 = cond_br %112#1, %94 {handshake.bb = 3 : ui32, handshake.name = "cond_br44"} : <i1>, <f32>
    sink %trueResult_10 {handshake.name = "sink31"} : <f32>
    %113 = mux %index_13 [%45, %111, %falseResult_11] {handshake.bb = 4 : ui32, handshake.name = "mux10"} : <i2>, [<f32>, <f32>, <f32>] to <f32>
    %result_12, %index_13 = control_merge [%46, %110, %falseResult_9]  {handshake.bb = 4 : ui32, handshake.name = "control_merge6"} : [<>, <>, <>] to <>, <i2>
    %114 = mux %index_15 [%56, %113] {handshake.bb = 5 : ui32, handshake.name = "mux11"} : <i1>, [<f32>, <f32>] to <f32>
    %result_14, %index_15 = control_merge [%57, %result_12]  {handshake.bb = 5 : ui32, handshake.name = "control_merge7"} : [<>, <>] to <>, <i1>
    sink %result_14 {handshake.name = "sink32"} : <>
    end {handshake.bb = 5 : ui32, handshake.name = "end0"} %114, %0#1 : <f32>, <>
  }
}

