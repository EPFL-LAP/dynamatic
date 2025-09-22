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
    %14 = mux %82#0 [%8, %93] {handshake.bb = 2 : ui32, handshake.name = "mux5"} : <i1>, [<f32>, <f32>] to <f32>
    %15:3 = fork [3] %14 {handshake.bb = 2 : ui32, handshake.name = "fork47"} : <f32>
    %16 = mux %82#1 [%9, %94] {handshake.bb = 2 : ui32, handshake.name = "mux6"} : <i1>, [<f32>, <f32>] to <f32>
    %17:3 = fork [3] %16 {handshake.bb = 2 : ui32, handshake.name = "fork48"} : <f32>
    %18 = mux %82#2 [%10, %95] {handshake.bb = 2 : ui32, handshake.name = "mux7"} : <i1>, [<i8>, <i8>] to <i8>
    %19 = mux %82#3 [%11, %96] {handshake.bb = 2 : ui32, handshake.name = "mux8"} : <i1>, [<f32>, <f32>] to <f32>
    %20 = mux %82#4 [%12, %97] {handshake.bb = 2 : ui32, handshake.name = "mux9"} : <i1>, [<f32>, <f32>] to <f32>
    %21:4 = fork [4] %20 {handshake.bb = 2 : ui32, handshake.name = "fork49"} : <f32>
    %22 = mux %82#5 [%result, %98] {handshake.bb = 2 : ui32, handshake.name = "mux15"} : <i1>, [<>, <>] to <>
    %23:6 = fork [6] %22 {handshake.bb = 2 : ui32, handshake.name = "fork50"} : <>
    %24 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %25 = constant %24 {handshake.bb = 2 : ui32, handshake.name = "constant6", value = -2.000000e+00 : f32} : <>, <f32>
    %26 = source {handshake.bb = 2 : ui32, handshake.name = "source2"} : <>
    %27 = constant %26 {handshake.bb = 2 : ui32, handshake.name = "constant7", value = 5.000000e-01 : f32} : <>, <f32>
    %28 = addf %15#0, %17#0 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "addf1"} : <f32>
    %29 = mulf %28, %27 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "mulf1"} : <f32>
    %30:6 = fork [6] %29 {handshake.bb = 2 : ui32, handshake.name = "fork51"} : <f32>
    %31 = mulf %30#0, %30#1 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "mulf2"} : <f32>
    %32 = addf %31, %25 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "addf2"} : <f32>
    %33:3 = fork [3] %32 {handshake.bb = 2 : ui32, handshake.name = "fork52"} : <f32>
    %34 = absf %33#0 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "absf0"} : <f32> to <f32>
    %35 = cmpf olt, %34, %21#0 {handshake.bb = 2 : ui32, handshake.name = "cmpf0"} : <f32>
    %36:3 = fork [3] %35 {handshake.bb = 2 : ui32, handshake.name = "fork25"} : <i1>
    %37 = not %36#2 {handshake.bb = 2 : ui32, handshake.name = "not0"} : <i1>
    %38:2 = fork [2] %37 {handshake.bb = 2 : ui32, handshake.name = "fork6"} : <i1>
    %39 = andi %83, %55#0 {handshake.bb = 2 : ui32, handshake.name = "andi0"} : <i1>
    %40:8 = fork [8] %39 {handshake.bb = 2 : ui32, handshake.name = "fork7"} : <i1>
    %41 = andi %92, %72#0 {handshake.bb = 2 : ui32, handshake.name = "andi1"} : <i1>
    %42:2 = fork [2] %41 {handshake.bb = 2 : ui32, handshake.name = "fork9"} : <i1>
    %43 = andi %52#1, %38#0 {handshake.bb = 2 : ui32, handshake.name = "andi2"} : <i1>
    %44:2 = fork [2] %43 {handshake.bb = 2 : ui32, handshake.name = "fork13"} : <i1>
    %45 = passer %30#2[%36#1] {handshake.bb = 2 : ui32, handshake.name = "passer0"} : <f32>, <i1>
    %46 = passer %23#0[%36#0] {handshake.bb = 2 : ui32, handshake.name = "passer2"} : <>, <i1>
    %47 = source {handshake.bb = 2 : ui32, handshake.name = "source3"} : <>
    %48 = constant %47 {handshake.bb = 2 : ui32, handshake.name = "constant8", value = 5.000000e-01 : f32} : <>, <f32>
    %49 = subf %17#2, %15#2 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "subf0"} : <f32>
    %50 = mulf %49, %48 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "mulf3"} : <f32>
    %51 = cmpf olt, %50, %21#1 {handshake.bb = 2 : ui32, handshake.name = "cmpf1"} : <f32>
    %52:2 = fork [2] %51 {handshake.bb = 2 : ui32, handshake.name = "fork16"} : <i1>
    %53 = not %52#0 {handshake.bb = 2 : ui32, handshake.name = "not1"} : <i1>
    %54 = andi %38#1, %53 {handshake.bb = 2 : ui32, handshake.name = "andi3"} : <i1>
    %55:2 = fork [2] %54 {handshake.bb = 2 : ui32, handshake.name = "fork17"} : <i1>
    %56 = passer %30#3[%44#1] {handshake.bb = 2 : ui32, handshake.name = "passer10"} : <f32>, <i1>
    %57 = passer %23#1[%44#0] {handshake.bb = 2 : ui32, handshake.name = "passer12"} : <>, <i1>
    %58 = extsi %18 {handshake.bb = 2 : ui32, handshake.name = "extsi4"} : <i8> to <i9>
    %59:2 = fork [2] %19 {handshake.bb = 2 : ui32, handshake.name = "fork14"} : <f32>
    %60:3 = fork [3] %61 {handshake.bb = 2 : ui32, handshake.name = "fork18"} : <f32>
    %61 = constant %23#2 {handshake.bb = 2 : ui32, handshake.name = "constant9", value = 0.000000e+00 : f32} : <>, <f32>
    %62 = source {handshake.bb = 2 : ui32, handshake.name = "source4"} : <>
    %63 = constant %62 {handshake.bb = 2 : ui32, handshake.name = "constant1", value = 1 : i2} : <>, <i2>
    %64 = extsi %63 {handshake.bb = 2 : ui32, handshake.name = "extsi5"} : <i2> to <i9>
    %65 = source {handshake.bb = 2 : ui32, handshake.name = "source5"} : <>
    %66 = constant %65 {handshake.bb = 2 : ui32, handshake.name = "constant2", value = 100 : i8} : <>, <i8>
    %67 = extsi %66 {handshake.bb = 2 : ui32, handshake.name = "extsi6"} : <i8> to <i9>
    %68 = mulf %59#1, %33#2 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "mulf4"} : <f32>
    %69 = cmpf olt, %68, %60#0 {handshake.bb = 2 : ui32, handshake.name = "cmpf2"} : <f32>
    %70:2 = fork [2] %69 {handshake.bb = 2 : ui32, handshake.name = "fork20"} : <i1>
    %71 = andi %55#1, %70#0 {handshake.bb = 2 : ui32, handshake.name = "andi4"} : <i1>
    %72:2 = fork [2] %71 {handshake.bb = 2 : ui32, handshake.name = "fork42"} : <i1>
    %73:2 = fork [2] %74 {handshake.bb = 2 : ui32, handshake.name = "fork43"} : <i9>
    %74 = addi %58, %64 {handshake.bb = 2 : ui32, handshake.name = "addi0"} : <i9>
    %75:2 = fork [2] %76 {handshake.bb = 2 : ui32, handshake.name = "fork44"} : <i8>
    %76 = trunci %73#0 {handshake.bb = 2 : ui32, handshake.name = "trunci0"} : <i9> to <i8>
    %77 = cmpi ult, %73#1, %67 {handshake.bb = 2 : ui32, handshake.name = "cmpi0"} : <i9>
    %78:3 = fork [3] %77 {handshake.bb = 2 : ui32, handshake.name = "fork45"} : <i1>
    %79 = andi %72#1, %78#0 {handshake.bb = 2 : ui32, handshake.name = "andi5"} : <i1>
    %80:7 = fork [7] %79 {handshake.bb = 2 : ui32, handshake.name = "fork3"} : <i1>
    %81 = init %80#6 {handshake.bb = 2 : ui32, handshake.name = "init6", initToken = 0 : ui1} : <i1>
    %82:6 = fork [6] %81 {handshake.bb = 2 : ui32, handshake.name = "fork4"} : <i1>
    %83 = not %70#1 {handshake.bb = 2 : ui32, handshake.name = "not2"} : <i1>
    %84 = passer %21#2[%40#7] {handshake.bb = 2 : ui32, handshake.name = "passer21"} : <f32>, <i1>
    %85 = passer %30#4[%40#3] {handshake.bb = 2 : ui32, handshake.name = "passer25"} : <f32>, <i1>
    %86 = passer %60#1[%40#1] {handshake.bb = 2 : ui32, handshake.name = "passer27"} : <f32>, <i1>
    %87 = passer %75#0[%40#5] {handshake.bb = 2 : ui32, handshake.name = "passer29"} : <i8>, <i1>
    %88 = passer %78#1[%40#2] {handshake.bb = 2 : ui32, handshake.name = "passer31"} : <i1>, <i1>
    %89 = passer %23#3[%40#0] {handshake.bb = 2 : ui32, handshake.name = "passer33"} : <>, <i1>
    %90 = passer %17#1[%40#6] {handshake.bb = 2 : ui32, handshake.name = "passer34"} : <f32>, <i1>
    %91 = passer %33#1[%40#4] {handshake.bb = 2 : ui32, handshake.name = "passer35"} : <f32>, <i1>
    %92 = not %78#2 {handshake.bb = 2 : ui32, handshake.name = "not3"} : <i1>
    %93 = passer %15#1[%80#4] {handshake.bb = 2 : ui32, handshake.name = "passer36"} : <f32>, <i1>
    %94 = passer %30#5[%80#5] {handshake.bb = 2 : ui32, handshake.name = "passer37"} : <f32>, <i1>
    %95 = passer %75#1[%80#0] {handshake.bb = 2 : ui32, handshake.name = "passer38"} : <i8>, <i1>
    %96 = passer %59#0[%80#3] {handshake.bb = 2 : ui32, handshake.name = "passer39"} : <f32>, <i1>
    %97 = passer %21#3[%80#1] {handshake.bb = 2 : ui32, handshake.name = "passer40"} : <f32>, <i1>
    %98 = passer %23#5[%80#2] {handshake.bb = 2 : ui32, handshake.name = "passer41"} : <>, <i1>
    %99 = passer %23#4[%42#0] {handshake.bb = 2 : ui32, handshake.name = "passer42"} : <>, <i1>
    %100 = passer %60#2[%42#1] {handshake.bb = 2 : ui32, handshake.name = "passer43"} : <f32>, <i1>
    %101:7 = fork [7] %88 {handshake.bb = 3 : ui32, handshake.name = "fork21"} : <i1>
    %trueResult, %falseResult = cond_br %101#6, %85 {handshake.bb = 3 : ui32, handshake.name = "cond_br38"} : <i1>, <f32>
    sink %falseResult {handshake.name = "sink26"} : <f32>
    %trueResult_0, %falseResult_1 = cond_br %101#5, %90 {handshake.bb = 3 : ui32, handshake.name = "cond_br39"} : <i1>, <f32>
    sink %falseResult_1 {handshake.name = "sink27"} : <f32>
    %trueResult_2, %falseResult_3 = cond_br %101#0, %87 {handshake.bb = 3 : ui32, handshake.name = "cond_br40"} : <i1>, <i8>
    sink %falseResult_3 {handshake.name = "sink28"} : <i8>
    %trueResult_4, %falseResult_5 = cond_br %101#4, %91 {handshake.bb = 3 : ui32, handshake.name = "cond_br41"} : <i1>, <f32>
    sink %falseResult_5 {handshake.name = "sink29"} : <f32>
    %trueResult_6, %falseResult_7 = cond_br %101#3, %84 {handshake.bb = 3 : ui32, handshake.name = "cond_br42"} : <i1>, <f32>
    sink %falseResult_7 {handshake.name = "sink30"} : <f32>
    %trueResult_8, %falseResult_9 = cond_br %101#2, %89 {handshake.bb = 3 : ui32, handshake.name = "cond_br43"} : <i1>, <>
    %trueResult_10, %falseResult_11 = cond_br %101#1, %86 {handshake.bb = 3 : ui32, handshake.name = "cond_br44"} : <i1>, <f32>
    sink %trueResult_10 {handshake.name = "sink31"} : <f32>
    %102 = mux %index_13 [%45, %100, %falseResult_11] {handshake.bb = 4 : ui32, handshake.name = "mux10"} : <i2>, [<f32>, <f32>, <f32>] to <f32>
    %result_12, %index_13 = control_merge [%46, %99, %falseResult_9]  {handshake.bb = 4 : ui32, handshake.name = "control_merge6"} : [<>, <>, <>] to <>, <i2>
    %103 = mux %index_15 [%56, %102] {handshake.bb = 5 : ui32, handshake.name = "mux11"} : <i1>, [<f32>, <f32>] to <f32>
    %result_14, %index_15 = control_merge [%57, %result_12]  {handshake.bb = 5 : ui32, handshake.name = "control_merge7"} : [<>, <>] to <>, <i1>
    sink %result_14 {handshake.name = "sink32"} : <>
    end {handshake.bb = 5 : ui32, handshake.name = "end0"} %103, %0#1 : <f32>, <>
  }
}

