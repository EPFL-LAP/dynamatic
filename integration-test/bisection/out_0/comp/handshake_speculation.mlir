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
    %14 = init %86#5 {handshake.bb = 2 : ui32, handshake.name = "init0", initToken = 0 : ui1} : <i1>
    %15 = mux %14 [%8, %97] {handshake.bb = 2 : ui32, handshake.name = "mux5"} : <i1>, [<f32>, <f32>] to <f32>
    %16:3 = fork [3] %15 {handshake.bb = 2 : ui32, handshake.name = "fork47"} : <f32>
    %17 = init %86#4 {handshake.bb = 2 : ui32, handshake.name = "init1", initToken = 0 : ui1} : <i1>
    %18 = mux %17 [%9, %98] {handshake.bb = 2 : ui32, handshake.name = "mux6"} : <i1>, [<f32>, <f32>] to <f32>
    %19:3 = fork [3] %18 {handshake.bb = 2 : ui32, handshake.name = "fork48"} : <f32>
    %20 = init %86#3 {handshake.bb = 2 : ui32, handshake.name = "init2", initToken = 0 : ui1} : <i1>
    %21 = mux %20 [%10, %99] {handshake.bb = 2 : ui32, handshake.name = "mux7"} : <i1>, [<i8>, <i8>] to <i8>
    %22 = init %86#2 {handshake.bb = 2 : ui32, handshake.name = "init3", initToken = 0 : ui1} : <i1>
    %23 = mux %22 [%11, %100] {handshake.bb = 2 : ui32, handshake.name = "mux8"} : <i1>, [<f32>, <f32>] to <f32>
    %24 = init %86#1 {handshake.bb = 2 : ui32, handshake.name = "init4", initToken = 0 : ui1} : <i1>
    %25 = init %86#0 {handshake.bb = 2 : ui32, handshake.name = "init5", initToken = 0 : ui1} : <i1>
    %26 = mux %24 [%12, %101] {handshake.bb = 2 : ui32, handshake.name = "mux9"} : <i1>, [<f32>, <f32>] to <f32>
    %27:4 = fork [4] %26 {handshake.bb = 2 : ui32, handshake.name = "fork49"} : <f32>
    %28 = mux %25 [%result, %102] {handshake.bb = 2 : ui32, handshake.name = "mux15"} : <i1>, [<>, <>] to <>
    %29:6 = fork [6] %28 {handshake.bb = 2 : ui32, handshake.name = "fork50"} : <>
    %30 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %31 = constant %30 {handshake.bb = 2 : ui32, handshake.name = "constant6", value = -2.000000e+00 : f32} : <>, <f32>
    %32 = source {handshake.bb = 2 : ui32, handshake.name = "source2"} : <>
    %33 = constant %32 {handshake.bb = 2 : ui32, handshake.name = "constant7", value = 5.000000e-01 : f32} : <>, <f32>
    %34 = addf %16#0, %19#0 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "addf1"} : <f32>
    %35 = mulf %34, %33 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "mulf1"} : <f32>
    %36:6 = fork [6] %35 {handshake.bb = 2 : ui32, handshake.name = "fork51"} : <f32>
    %37 = mulf %36#0, %36#1 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "mulf2"} : <f32>
    %38 = addf %37, %31 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "addf2"} : <f32>
    %39:3 = fork [3] %38 {handshake.bb = 2 : ui32, handshake.name = "fork52"} : <f32>
    %40 = absf %39#0 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "absf0"} : <f32> to <f32>
    %41 = cmpf olt, %40, %27#0 {handshake.bb = 2 : ui32, handshake.name = "cmpf0"} : <f32>
    %42:3 = fork [3] %41 {handshake.bb = 2 : ui32, handshake.name = "fork25"} : <i1>
    %43 = not %42#2 {handshake.bb = 2 : ui32, handshake.name = "not0"} : <i1>
    %44:2 = fork [2] %43 {handshake.bb = 2 : ui32, handshake.name = "fork6"} : <i1>
    %45 = andi %87, %61#0 {handshake.bb = 2 : ui32, handshake.name = "andi0"} : <i1>
    %46:8 = fork [8] %45 {handshake.bb = 2 : ui32, handshake.name = "fork7"} : <i1>
    %47 = andi %96, %78#0 {handshake.bb = 2 : ui32, handshake.name = "andi1"} : <i1>
    %48:2 = fork [2] %47 {handshake.bb = 2 : ui32, handshake.name = "fork9"} : <i1>
    %49 = andi %58#1, %44#0 {handshake.bb = 2 : ui32, handshake.name = "andi2"} : <i1>
    %50:2 = fork [2] %49 {handshake.bb = 2 : ui32, handshake.name = "fork13"} : <i1>
    %51 = passer %36#2[%42#1] {handshake.bb = 2 : ui32, handshake.name = "passer0"} : <f32>, <i1>
    %52 = passer %29#0[%42#0] {handshake.bb = 2 : ui32, handshake.name = "passer2"} : <>, <i1>
    %53 = source {handshake.bb = 2 : ui32, handshake.name = "source3"} : <>
    %54 = constant %53 {handshake.bb = 2 : ui32, handshake.name = "constant8", value = 5.000000e-01 : f32} : <>, <f32>
    %55 = subf %19#2, %16#2 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "subf0"} : <f32>
    %56 = mulf %55, %54 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "mulf3"} : <f32>
    %57 = cmpf olt, %56, %27#1 {handshake.bb = 2 : ui32, handshake.name = "cmpf1"} : <f32>
    %58:2 = fork [2] %57 {handshake.bb = 2 : ui32, handshake.name = "fork16"} : <i1>
    %59 = not %58#0 {handshake.bb = 2 : ui32, handshake.name = "not1"} : <i1>
    %60 = andi %44#1, %59 {handshake.bb = 2 : ui32, handshake.name = "andi3"} : <i1>
    %61:2 = fork [2] %60 {handshake.bb = 2 : ui32, handshake.name = "fork17"} : <i1>
    %62 = passer %36#3[%50#1] {handshake.bb = 2 : ui32, handshake.name = "passer10"} : <f32>, <i1>
    %63 = passer %29#1[%50#0] {handshake.bb = 2 : ui32, handshake.name = "passer12"} : <>, <i1>
    %64 = extsi %21 {handshake.bb = 2 : ui32, handshake.name = "extsi4"} : <i8> to <i9>
    %65:2 = fork [2] %23 {handshake.bb = 2 : ui32, handshake.name = "fork14"} : <f32>
    %66:3 = fork [3] %67 {handshake.bb = 2 : ui32, handshake.name = "fork18"} : <f32>
    %67 = constant %29#2 {handshake.bb = 2 : ui32, handshake.name = "constant9", value = 0.000000e+00 : f32} : <>, <f32>
    %68 = source {handshake.bb = 2 : ui32, handshake.name = "source4"} : <>
    %69 = constant %68 {handshake.bb = 2 : ui32, handshake.name = "constant1", value = 1 : i2} : <>, <i2>
    %70 = extsi %69 {handshake.bb = 2 : ui32, handshake.name = "extsi5"} : <i2> to <i9>
    %71 = source {handshake.bb = 2 : ui32, handshake.name = "source5"} : <>
    %72 = constant %71 {handshake.bb = 2 : ui32, handshake.name = "constant2", value = 100 : i8} : <>, <i8>
    %73 = extsi %72 {handshake.bb = 2 : ui32, handshake.name = "extsi6"} : <i8> to <i9>
    %74 = mulf %65#1, %39#2 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "mulf4"} : <f32>
    %75 = cmpf olt, %74, %66#0 {handshake.bb = 2 : ui32, handshake.name = "cmpf2"} : <f32>
    %76:2 = fork [2] %75 {handshake.bb = 2 : ui32, handshake.name = "fork20"} : <i1>
    %77 = andi %61#1, %76#0 {handshake.bb = 2 : ui32, handshake.name = "andi4"} : <i1>
    %78:2 = fork [2] %77 {handshake.bb = 2 : ui32, handshake.name = "fork42"} : <i1>
    %79:2 = fork [2] %80 {handshake.bb = 2 : ui32, handshake.name = "fork43"} : <i9>
    %80 = addi %64, %70 {handshake.bb = 2 : ui32, handshake.name = "addi0"} : <i9>
    %81:2 = fork [2] %82 {handshake.bb = 2 : ui32, handshake.name = "fork44"} : <i8>
    %82 = trunci %79#0 {handshake.bb = 2 : ui32, handshake.name = "trunci0"} : <i9> to <i8>
    %83 = cmpi ult, %79#1, %73 {handshake.bb = 2 : ui32, handshake.name = "cmpi0"} : <i9>
    %84:3 = fork [3] %83 {handshake.bb = 2 : ui32, handshake.name = "fork45"} : <i1>
    %85 = andi %78#1, %84#0 {handshake.bb = 2 : ui32, handshake.name = "andi5"} : <i1>
    %86:12 = fork [12] %85 {handshake.bb = 2 : ui32, handshake.name = "fork46"} : <i1>
    %87 = not %76#1 {handshake.bb = 2 : ui32, handshake.name = "not2"} : <i1>
    %88 = passer %27#2[%46#7] {handshake.bb = 2 : ui32, handshake.name = "passer21"} : <f32>, <i1>
    %89 = passer %36#4[%46#3] {handshake.bb = 2 : ui32, handshake.name = "passer25"} : <f32>, <i1>
    %90 = passer %66#1[%46#1] {handshake.bb = 2 : ui32, handshake.name = "passer27"} : <f32>, <i1>
    %91 = passer %81#0[%46#5] {handshake.bb = 2 : ui32, handshake.name = "passer29"} : <i8>, <i1>
    %92 = passer %84#1[%46#2] {handshake.bb = 2 : ui32, handshake.name = "passer31"} : <i1>, <i1>
    %93 = passer %29#3[%46#0] {handshake.bb = 2 : ui32, handshake.name = "passer33"} : <>, <i1>
    %94 = passer %19#1[%46#6] {handshake.bb = 2 : ui32, handshake.name = "passer34"} : <f32>, <i1>
    %95 = passer %39#1[%46#4] {handshake.bb = 2 : ui32, handshake.name = "passer35"} : <f32>, <i1>
    %96 = not %84#2 {handshake.bb = 2 : ui32, handshake.name = "not3"} : <i1>
    %97 = passer %16#1[%86#7] {handshake.bb = 2 : ui32, handshake.name = "passer36"} : <f32>, <i1>
    %98 = passer %36#5[%86#6] {handshake.bb = 2 : ui32, handshake.name = "passer37"} : <f32>, <i1>
    %99 = passer %81#1[%86#11] {handshake.bb = 2 : ui32, handshake.name = "passer38"} : <i8>, <i1>
    %100 = passer %65#0[%86#8] {handshake.bb = 2 : ui32, handshake.name = "passer39"} : <f32>, <i1>
    %101 = passer %27#3[%86#10] {handshake.bb = 2 : ui32, handshake.name = "passer40"} : <f32>, <i1>
    %102 = passer %29#5[%86#9] {handshake.bb = 2 : ui32, handshake.name = "passer41"} : <>, <i1>
    %103 = passer %29#4[%48#0] {handshake.bb = 2 : ui32, handshake.name = "passer42"} : <>, <i1>
    %104 = passer %66#2[%48#1] {handshake.bb = 2 : ui32, handshake.name = "passer43"} : <f32>, <i1>
    %105:7 = fork [7] %92 {handshake.bb = 3 : ui32, handshake.name = "fork21"} : <i1>
    %trueResult, %falseResult = cond_br %105#6, %89 {handshake.bb = 3 : ui32, handshake.name = "cond_br38"} : <i1>, <f32>
    sink %falseResult {handshake.name = "sink26"} : <f32>
    %trueResult_0, %falseResult_1 = cond_br %105#5, %94 {handshake.bb = 3 : ui32, handshake.name = "cond_br39"} : <i1>, <f32>
    sink %falseResult_1 {handshake.name = "sink27"} : <f32>
    %trueResult_2, %falseResult_3 = cond_br %105#0, %91 {handshake.bb = 3 : ui32, handshake.name = "cond_br40"} : <i1>, <i8>
    sink %falseResult_3 {handshake.name = "sink28"} : <i8>
    %trueResult_4, %falseResult_5 = cond_br %105#4, %95 {handshake.bb = 3 : ui32, handshake.name = "cond_br41"} : <i1>, <f32>
    sink %falseResult_5 {handshake.name = "sink29"} : <f32>
    %trueResult_6, %falseResult_7 = cond_br %105#3, %88 {handshake.bb = 3 : ui32, handshake.name = "cond_br42"} : <i1>, <f32>
    sink %falseResult_7 {handshake.name = "sink30"} : <f32>
    %trueResult_8, %falseResult_9 = cond_br %105#2, %93 {handshake.bb = 3 : ui32, handshake.name = "cond_br43"} : <i1>, <>
    %trueResult_10, %falseResult_11 = cond_br %105#1, %90 {handshake.bb = 3 : ui32, handshake.name = "cond_br44"} : <i1>, <f32>
    sink %trueResult_10 {handshake.name = "sink31"} : <f32>
    %106 = mux %index_13 [%51, %104, %falseResult_11] {handshake.bb = 4 : ui32, handshake.name = "mux10"} : <i2>, [<f32>, <f32>, <f32>] to <f32>
    %result_12, %index_13 = control_merge [%52, %103, %falseResult_9]  {handshake.bb = 4 : ui32, handshake.name = "control_merge6"} : [<>, <>, <>] to <>, <i2>
    %107 = mux %index_15 [%62, %106] {handshake.bb = 5 : ui32, handshake.name = "mux11"} : <i1>, [<f32>, <f32>] to <f32>
    %result_14, %index_15 = control_merge [%63, %result_12]  {handshake.bb = 5 : ui32, handshake.name = "control_merge7"} : [<>, <>] to <>, <i1>
    sink %result_14 {handshake.name = "sink32"} : <>
    end {handshake.bb = 5 : ui32, handshake.name = "end0"} %107, %0#1 : <f32>, <>
  }
}

