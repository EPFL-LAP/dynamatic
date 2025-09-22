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
    %14 = mux %81#0 [%8, %102] {handshake.bb = 2 : ui32, handshake.name = "mux5"} : <i1>, [<f32>, <f32>] to <f32>
    %15 = mux %81#1 [%9, %26] {handshake.bb = 2 : ui32, handshake.name = "mux6"} : <i1>, [<f32>, <f32>] to <f32>
    %16 = mux %81#2 [%10, %69] {handshake.bb = 2 : ui32, handshake.name = "mux7"} : <i1>, [<i8>, <i8>] to <i8>
    %17 = mux %81#3 [%11, %104] {handshake.bb = 2 : ui32, handshake.name = "mux8"} : <i1>, [<f32>, <f32>] to <f32>
    %18 = mux %81#4 [%12, %106] {handshake.bb = 2 : ui32, handshake.name = "mux9"} : <i1>, [<f32>, <f32>] to <f32>
    %19 = mux %81#5 [%result, %108] {handshake.bb = 2 : ui32, handshake.name = "mux15"} : <i1>, [<>, <>] to <>
    %20 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %21 = constant %20 {handshake.bb = 2 : ui32, handshake.name = "constant6", value = -2.000000e+00 : f32} : <>, <f32>
    %22 = source {handshake.bb = 2 : ui32, handshake.name = "source2"} : <>
    %23 = constant %22 {handshake.bb = 2 : ui32, handshake.name = "constant7", value = 5.000000e-01 : f32} : <>, <f32>
    %24 = addf %103#2, %25#2 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "addf1"} : <f32>
    %25:3 = fork [3] %15 {handshake.bb = 2 : ui32, handshake.name = "fork6"} : <f32>
    %26 = passer %27#3[%79#2] {handshake.bb = 2 : ui32, handshake.name = "passer44"} : <f32>, <i1>
    %27:6 = fork [6] %28 {handshake.bb = 2 : ui32, handshake.name = "fork7"} : <f32>
    %28 = mulf %24, %23 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "mulf1"} : <f32>
    %29 = mulf %27#0, %27#1 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "mulf2"} : <f32>
    %30:3 = fork [3] %31 {handshake.bb = 2 : ui32, handshake.name = "fork9"} : <f32>
    %31 = addf %29, %21 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "addf2"} : <f32>
    %32 = absf %30#2 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "absf0"} : <f32> to <f32>
    %33 = cmpf olt, %32, %107#3 {handshake.bb = 2 : ui32, handshake.name = "cmpf0"} : <f32>
    %34:2 = fork [2] %33 {handshake.bb = 2 : ui32, handshake.name = "fork13"} : <i1>
    %35:2 = fork [2] %36 {handshake.bb = 2 : ui32, handshake.name = "fork16"} : <i1>
    %36 = not %34#0 {handshake.bb = 2 : ui32, handshake.name = "not0"} : <i1>
    %37 = andi %46#1, %35#0 {handshake.bb = 2 : ui32, handshake.name = "andi0"} : <i1>
    %38 = andi %101, %65#0 {handshake.bb = 2 : ui32, handshake.name = "andi1"} : <i1>
    %39 = andi %92, %49#0 {handshake.bb = 2 : ui32, handshake.name = "andi2"} : <i1>
    %40 = passer %27#5[%91#1] {handshake.bb = 2 : ui32, handshake.name = "passer0"} : <f32>, <i1>
    %41 = passer %109#0[%91#0] {handshake.bb = 2 : ui32, handshake.name = "passer2"} : <>, <i1>
    %42 = source {handshake.bb = 2 : ui32, handshake.name = "source3"} : <>
    %43 = constant %42 {handshake.bb = 2 : ui32, handshake.name = "constant8", value = 5.000000e-01 : f32} : <>, <f32>
    %44 = subf %25#1, %103#1 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "subf0"} : <f32>
    %45 = mulf %44, %43 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "mulf3"} : <f32>
    %46:2 = fork [2] %47 {handshake.bb = 2 : ui32, handshake.name = "fork17"} : <i1>
    %47 = cmpf olt, %45, %107#2 {handshake.bb = 2 : ui32, handshake.name = "cmpf1"} : <f32>
    %48 = not %46#0 {handshake.bb = 2 : ui32, handshake.name = "not1"} : <i1>
    %49:2 = fork [2] %50 {handshake.bb = 2 : ui32, handshake.name = "fork18"} : <i1>
    %50 = andi %35#1, %48 {handshake.bb = 2 : ui32, handshake.name = "andi3"} : <i1>
    %51 = passer %27#4[%89#0] {handshake.bb = 2 : ui32, handshake.name = "passer10"} : <f32>, <i1>
    %52 = passer %109#5[%89#1] {handshake.bb = 2 : ui32, handshake.name = "passer12"} : <>, <i1>
    %53 = extsi %16 {handshake.bb = 2 : ui32, handshake.name = "extsi4"} : <i8> to <i9>
    %54:3 = fork [3] %55 {handshake.bb = 2 : ui32, handshake.name = "fork20"} : <f32>
    %55 = constant %109#1 {handshake.bb = 2 : ui32, handshake.name = "constant9", value = 0.000000e+00 : f32} : <>, <f32>
    %56 = source {handshake.bb = 2 : ui32, handshake.name = "source4"} : <>
    %57 = constant %56 {handshake.bb = 2 : ui32, handshake.name = "constant1", value = 1 : i2} : <>, <i2>
    %58 = extsi %57 {handshake.bb = 2 : ui32, handshake.name = "extsi5"} : <i2> to <i9>
    %59 = source {handshake.bb = 2 : ui32, handshake.name = "source5"} : <>
    %60 = constant %59 {handshake.bb = 2 : ui32, handshake.name = "constant2", value = 100 : i8} : <>, <i8>
    %61 = extsi %60 {handshake.bb = 2 : ui32, handshake.name = "extsi6"} : <i8> to <i9>
    %62 = mulf %105#1, %30#1 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "mulf4"} : <f32>
    %63:2 = fork [2] %64 {handshake.bb = 2 : ui32, handshake.name = "fork42"} : <i1>
    %64 = cmpf olt, %62, %54#0 {handshake.bb = 2 : ui32, handshake.name = "cmpf2"} : <f32>
    %65:2 = fork [2] %66 {handshake.bb = 2 : ui32, handshake.name = "fork43"} : <i1>
    %66 = andi %49#1, %63#0 {handshake.bb = 2 : ui32, handshake.name = "andi4"} : <i1>
    %67:2 = fork [2] %68 {handshake.bb = 2 : ui32, handshake.name = "fork44"} : <i9>
    %68 = addi %53, %58 {handshake.bb = 2 : ui32, handshake.name = "addi0"} : <i9>
    %69 = passer %70#1[%79#4] {handshake.bb = 2 : ui32, handshake.name = "passer45"} : <i8>, <i1>
    %70:2 = fork [2] %71 {handshake.bb = 2 : ui32, handshake.name = "fork45"} : <i8>
    %71 = trunci %67#0 {handshake.bb = 2 : ui32, handshake.name = "trunci0"} : <i9> to <i8>
    %72:3 = fork [3] %73 {handshake.bb = 2 : ui32, handshake.name = "fork46"} : <i1>
    %73 = cmpi ult, %67#1, %61 {handshake.bb = 2 : ui32, handshake.name = "cmpi0"} : <i9>
    %74 = passer %75[%83#4] {handshake.bb = 2 : ui32, handshake.name = "passer46"} : <i1>, <i1>
    %75 = andi %65#1, %72#0 {handshake.bb = 2 : ui32, handshake.name = "andi5"} : <i1>
    %76 = spec_v2_repeating_init %74 {handshake.bb = 2 : ui32, handshake.name = "spec_v2_repeating_init0", initToken = 1 : ui1, specv2_top_ri = true} : <i1>
    %77:2 = fork [2] %76 {handshake.bb = 2 : ui32, handshake.name = "fork47"} : <i1>
    %78 = spec_v2_repeating_init %77#0 {handshake.bb = 2 : ui32, handshake.name = "spec_v2_repeating_init1", initToken = 1 : ui1} : <i1>
    %79:8 = fork [8] %78 {handshake.bb = 2 : ui32, handshake.name = "fork3"} : <i1>
    %80 = init %79#7 {handshake.bb = 2 : ui32, handshake.name = "init6", initToken = 0 : ui1} : <i1>
    %81:6 = fork [6] %80 {handshake.bb = 2 : ui32, handshake.name = "fork4"} : <i1>
    %82 = andi %77#1, %79#6 {handshake.bb = 2 : ui32, handshake.name = "andi6", specv2_tmp_and = true} : <i1>
    %83:5 = fork [5] %82 {handshake.bb = 2 : ui32, handshake.name = "fork49"} : <i1>
    %84 = andi %39, %83#0 {handshake.bb = 2 : ui32, handshake.name = "andi7"} : <i1>
    %85:8 = fork [8] %84 {handshake.bb = 2 : ui32, handshake.name = "fork50"} : <i1>
    %86 = andi %38, %83#1 {handshake.bb = 2 : ui32, handshake.name = "andi8"} : <i1>
    %87:2 = fork [2] %86 {handshake.bb = 2 : ui32, handshake.name = "fork51"} : <i1>
    %88 = andi %37, %83#2 {handshake.bb = 2 : ui32, handshake.name = "andi9"} : <i1>
    %89:2 = fork [2] %88 {handshake.bb = 2 : ui32, handshake.name = "fork52"} : <i1>
    %90 = andi %34#1, %83#3 {handshake.bb = 2 : ui32, handshake.name = "andi10"} : <i1>
    %91:2 = fork [2] %90 {handshake.bb = 2 : ui32, handshake.name = "fork53"} : <i1>
    %92 = not %63#1 {handshake.bb = 2 : ui32, handshake.name = "not2"} : <i1>
    %93 = passer %107#0[%85#5] {handshake.bb = 2 : ui32, handshake.name = "passer21"} : <f32>, <i1>
    %94 = passer %27#2[%85#6] {handshake.bb = 2 : ui32, handshake.name = "passer25"} : <f32>, <i1>
    %95 = passer %54#1[%85#4] {handshake.bb = 2 : ui32, handshake.name = "passer27"} : <f32>, <i1>
    %96 = passer %70#0[%85#3] {handshake.bb = 2 : ui32, handshake.name = "passer29"} : <i8>, <i1>
    %97 = passer %72#1[%85#1] {handshake.bb = 2 : ui32, handshake.name = "passer31"} : <i1>, <i1>
    %98 = passer %109#2[%85#2] {handshake.bb = 2 : ui32, handshake.name = "passer33"} : <>, <i1>
    %99 = passer %25#0[%85#7] {handshake.bb = 2 : ui32, handshake.name = "passer34"} : <f32>, <i1>
    %100 = passer %30#0[%85#0] {handshake.bb = 2 : ui32, handshake.name = "passer35"} : <f32>, <i1>
    %101 = not %72#2 {handshake.bb = 2 : ui32, handshake.name = "not3"} : <i1>
    %102 = passer %103#0[%79#0] {handshake.bb = 2 : ui32, handshake.name = "passer47"} : <f32>, <i1>
    %103:3 = fork [3] %14 {handshake.bb = 2 : ui32, handshake.name = "fork54"} : <f32>
    %104 = passer %105#0[%79#3] {handshake.bb = 2 : ui32, handshake.name = "passer48"} : <f32>, <i1>
    %105:2 = fork [2] %17 {handshake.bb = 2 : ui32, handshake.name = "fork55"} : <f32>
    %106 = passer %107#1[%79#1] {handshake.bb = 2 : ui32, handshake.name = "passer49"} : <f32>, <i1>
    %107:4 = fork [4] %18 {handshake.bb = 2 : ui32, handshake.name = "fork56"} : <f32>
    %108 = passer %109#4[%79#5] {handshake.bb = 2 : ui32, handshake.name = "passer50"} : <>, <i1>
    %109:6 = fork [6] %19 {handshake.bb = 2 : ui32, handshake.name = "fork57"} : <>
    %110 = passer %109#3[%87#1] {handshake.bb = 2 : ui32, handshake.name = "passer42"} : <>, <i1>
    %111 = passer %54#2[%87#0] {handshake.bb = 2 : ui32, handshake.name = "passer43"} : <f32>, <i1>
    %112:7 = fork [7] %97 {handshake.bb = 3 : ui32, handshake.name = "fork21"} : <i1>
    %trueResult, %falseResult = cond_br %112#6, %94 {handshake.bb = 3 : ui32, handshake.name = "cond_br38"} : <i1>, <f32>
    sink %falseResult {handshake.name = "sink26"} : <f32>
    %trueResult_0, %falseResult_1 = cond_br %112#5, %99 {handshake.bb = 3 : ui32, handshake.name = "cond_br39"} : <i1>, <f32>
    sink %falseResult_1 {handshake.name = "sink27"} : <f32>
    %trueResult_2, %falseResult_3 = cond_br %112#0, %96 {handshake.bb = 3 : ui32, handshake.name = "cond_br40"} : <i1>, <i8>
    sink %falseResult_3 {handshake.name = "sink28"} : <i8>
    %trueResult_4, %falseResult_5 = cond_br %112#4, %100 {handshake.bb = 3 : ui32, handshake.name = "cond_br41"} : <i1>, <f32>
    sink %falseResult_5 {handshake.name = "sink29"} : <f32>
    %trueResult_6, %falseResult_7 = cond_br %112#3, %93 {handshake.bb = 3 : ui32, handshake.name = "cond_br42"} : <i1>, <f32>
    sink %falseResult_7 {handshake.name = "sink30"} : <f32>
    %trueResult_8, %falseResult_9 = cond_br %112#2, %98 {handshake.bb = 3 : ui32, handshake.name = "cond_br43"} : <i1>, <>
    %trueResult_10, %falseResult_11 = cond_br %112#1, %95 {handshake.bb = 3 : ui32, handshake.name = "cond_br44"} : <i1>, <f32>
    sink %trueResult_10 {handshake.name = "sink31"} : <f32>
    %113 = mux %index_13 [%40, %111, %falseResult_11] {handshake.bb = 4 : ui32, handshake.name = "mux10"} : <i2>, [<f32>, <f32>, <f32>] to <f32>
    %result_12, %index_13 = control_merge [%41, %110, %falseResult_9]  {handshake.bb = 4 : ui32, handshake.name = "control_merge6"} : [<>, <>, <>] to <>, <i2>
    %114 = mux %index_15 [%51, %113] {handshake.bb = 5 : ui32, handshake.name = "mux11"} : <i1>, [<f32>, <f32>] to <f32>
    %result_14, %index_15 = control_merge [%52, %result_12]  {handshake.bb = 5 : ui32, handshake.name = "control_merge7"} : [<>, <>] to <>, <i1>
    sink %result_14 {handshake.name = "sink32"} : <>
    end {handshake.bb = 5 : ui32, handshake.name = "end0"} %114, %0#1 : <f32>, <>
  }
}

