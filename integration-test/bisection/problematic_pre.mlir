module {
  handshake.func @bisection(%arg0: !handshake.channel<f32>, %arg1: !handshake.channel<f32>, %arg2: !handshake.channel<f32>, %arg3: !handshake.control<>, ...) -> (!handshake.channel<f32>, !handshake.control<>) attributes {argNames = ["a", "b", "tol", "start"], resNames = ["out0", "end"]} {
    %0 = source {handshake.bb = 3 : ui32, handshake.name = "source6"} : <>
    %1 = constant %0 {handshake.bb = 3 : ui32, handshake.name = "constant3", value = false} : <>, <i1>
    %2 = mux %54#0 [%1, %77#0] {handshake.bb = 3 : ui32, handshake.name = "mux12", specv2_loop_cond_mux = true} : <i1>, [<i1>, <i1>] to <i1>
    %3 = source {handshake.bb = 4 : ui32, handshake.name = "source7"} : <>
    %4 = constant %3 {handshake.bb = 4 : ui32, handshake.name = "constant4", value = false} : <>, <i1>
    %5 = mux %2 [%4, %104#0] {handshake.bb = 4 : ui32, handshake.name = "mux13", specv2_loop_cond_mux = true} : <i1>, [<i1>, <i1>] to <i1>
    %6 = source {handshake.bb = 5 : ui32, handshake.name = "source8"} : <>
    %7 = constant %6 {handshake.bb = 5 : ui32, handshake.name = "constant10", value = false} : <>, <i1>
    %8 = mux %5 [%7, %130#0] {handshake.bb = 5 : ui32, handshake.name = "mux14", specv2_loop_cond_mux = true} : <i1>, [<i1>, <i1>] to <i1>
    %9:6 = fork [6] %8 {handshake.bb = 5 : ui32, handshake.name = "fork22"} : <i1>
    %10:3 = fork [3] %arg3 {handshake.bb = 0 : ui32, handshake.name = "fork0"} : <>
    %11:3 = fork [3] %arg0 {handshake.bb = 0 : ui32, handshake.name = "fork1"} : <f32>
    %12 = constant %10#0 {handshake.bb = 0 : ui32, handshake.name = "constant0", value = false} : <>, <i1>
    %13 = source {handshake.bb = 0 : ui32, handshake.name = "source0"} : <>
    %14 = constant %13 {handshake.bb = 0 : ui32, handshake.name = "constant5", value = -2.000000e+00 : f32} : <>, <f32>
    %15 = mulf %11#1, %11#2 {fastmath = #arith.fastmath<none>, handshake.bb = 0 : ui32, handshake.name = "mulf0"} : <f32>
    %16 = addf %15, %14 {fastmath = #arith.fastmath<none>, handshake.bb = 0 : ui32, handshake.name = "addf0"} : <f32>
    %17 = extsi %12 {handshake.bb = 0 : ui32, handshake.name = "extsi3"} : <i1> to <i8>
    %18 = mux %23#1 [%11#0, %trueResult] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<f32>, <f32>] to <f32>
    %19 = mux %23#2 [%arg1, %trueResult_0] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i1>, [<f32>, <f32>] to <f32>
    %20 = mux %23#0 [%17, %trueResult_2] {handshake.bb = 1 : ui32, handshake.name = "mux2"} : <i1>, [<i8>, <i8>] to <i8>
    %21 = mux %23#3 [%16, %trueResult_4] {handshake.bb = 1 : ui32, handshake.name = "mux3"} : <i1>, [<f32>, <f32>] to <f32>
    %22 = mux %23#4 [%arg2, %trueResult_6] {handshake.bb = 1 : ui32, handshake.name = "mux4"} : <i1>, [<f32>, <f32>] to <f32>
    %result, %index = control_merge [%10#2, %trueResult_8]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %23:5 = fork [5] %index {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <i1>
    %24 = init %9#5 {handshake.bb = 2 : ui32, handshake.name = "init0", initToken = 0 : ui1} : <i1>
    %25 = mux %24 [%18, %133] {handshake.bb = 2 : ui32, handshake.name = "mux5"} : <i1>, [<f32>, <f32>] to <f32>
    %26:2 = fork [2] %25 {handshake.bb = 2 : ui32, handshake.name = "fork3"} : <f32>
    %27 = init %9#4 {handshake.bb = 2 : ui32, handshake.name = "init1", initToken = 0 : ui1} : <i1>
    %28 = mux %27 [%19, %134] {handshake.bb = 2 : ui32, handshake.name = "mux6"} : <i1>, [<f32>, <f32>] to <f32>
    %29:2 = fork [2] %28 {handshake.bb = 2 : ui32, handshake.name = "fork4"} : <f32>
    %30 = init %9#3 {handshake.bb = 2 : ui32, handshake.name = "init2", initToken = 0 : ui1} : <i1>
    %31 = mux %30 [%20, %135] {handshake.bb = 2 : ui32, handshake.name = "mux7"} : <i1>, [<i8>, <i8>] to <i8>
    %32 = init %9#2 {handshake.bb = 2 : ui32, handshake.name = "init3", initToken = 0 : ui1} : <i1>
    %33 = mux %32 [%21, %136] {handshake.bb = 2 : ui32, handshake.name = "mux8"} : <i1>, [<f32>, <f32>] to <f32>
    %34 = init %9#1 {handshake.bb = 2 : ui32, handshake.name = "init4", initToken = 0 : ui1} : <i1>
    %35 = init %9#0 {handshake.bb = 2 : ui32, handshake.name = "init5", initToken = 0 : ui1} : <i1>
    %36 = mux %34 [%22, %137] {handshake.bb = 2 : ui32, handshake.name = "mux9"} : <i1>, [<f32>, <f32>] to <f32>
    %37:2 = fork [2] %36 {handshake.bb = 2 : ui32, handshake.name = "fork5"} : <f32>
    %38 = mux %35 [%result, %138] {handshake.bb = 2 : ui32, handshake.name = "mux15"} : <i1>, [<>, <>] to <>
    %39:2 = fork [2] %38 {handshake.bb = 2 : ui32, handshake.name = "fork23"} : <>
    %40 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %41 = constant %40 {handshake.bb = 2 : ui32, handshake.name = "constant6", value = -2.000000e+00 : f32} : <>, <f32>
    %42 = source {handshake.bb = 2 : ui32, handshake.name = "source2"} : <>
    %43 = constant %42 {handshake.bb = 2 : ui32, handshake.name = "constant7", value = 5.000000e-01 : f32} : <>, <f32>
    %44 = addf %26#1, %29#1 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "addf1"} : <f32>
    %45 = mulf %44, %43 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "mulf1"} : <f32>
    %46:4 = fork [4] %45 {handshake.bb = 2 : ui32, handshake.name = "fork24"} : <f32>
    %47 = mulf %46#0, %46#1 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "mulf2"} : <f32>
    %48 = addf %47, %41 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "addf2"} : <f32>
    %49:2 = fork [2] %48 {handshake.bb = 2 : ui32, handshake.name = "fork8"} : <f32>
    %50 = absf %49#1 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "absf0"} : <f32> to <f32>
    %51 = cmpf olt, %50, %37#1 {handshake.bb = 2 : ui32, handshake.name = "cmpf0"} : <f32>
    %52:3 = fork [3] %51 {handshake.bb = 2 : ui32, handshake.name = "fork25"} : <i1>
    %53 = not %52#2 {handshake.bb = 2 : ui32, handshake.name = "not0"} : <i1>
    %54:9 = fork [9] %53 {handshake.bb = 2 : ui32, handshake.name = "fork26"} : <i1>
    %55 = passer %46#3[%52#1] {handshake.bb = 2 : ui32, handshake.name = "passer0"} : <f32>, <i1>
    %56 = passer %46#2[%54#8] {handshake.bb = 2 : ui32, handshake.name = "passer1"} : <f32>, <i1>
    %57:2 = fork [2] %56 {handshake.bb = 2 : ui32, handshake.name = "fork27"} : <f32>
    %58 = passer %39#0[%52#0] {handshake.bb = 2 : ui32, handshake.name = "passer2"} : <>, <i1>
    %59 = passer %39#1[%54#7] {handshake.bb = 2 : ui32, handshake.name = "passer3"} : <>, <i1>
    %60:2 = fork [2] %59 {handshake.bb = 2 : ui32, handshake.name = "fork28"} : <>
    %61 = passer %37#0[%54#6] {handshake.bb = 2 : ui32, handshake.name = "passer4"} : <f32>, <i1>
    %62 = passer %26#0[%54#5] {handshake.bb = 2 : ui32, handshake.name = "passer5"} : <f32>, <i1>
    %63 = passer %29#0[%54#4] {handshake.bb = 2 : ui32, handshake.name = "passer6"} : <f32>, <i1>
    %64 = passer %31[%54#3] {handshake.bb = 2 : ui32, handshake.name = "passer7"} : <i8>, <i1>
    %65 = passer %33[%54#2] {handshake.bb = 2 : ui32, handshake.name = "passer8"} : <f32>, <i1>
    %66 = passer %49#0[%54#1] {handshake.bb = 2 : ui32, handshake.name = "passer9"} : <f32>, <i1>
    %67:2 = fork [2] %61 {handshake.bb = 3 : ui32, handshake.name = "fork10"} : <f32>
    %68:2 = fork [2] %62 {handshake.bb = 3 : ui32, handshake.name = "fork11"} : <f32>
    %69:2 = fork [2] %63 {handshake.bb = 3 : ui32, handshake.name = "fork12"} : <f32>
    %70 = source {handshake.bb = 3 : ui32, handshake.name = "source3"} : <>
    %71 = constant %70 {handshake.bb = 3 : ui32, handshake.name = "constant8", value = 5.000000e-01 : f32} : <>, <f32>
    %72 = subf %69#1, %68#1 {fastmath = #arith.fastmath<none>, handshake.bb = 3 : ui32, handshake.name = "subf0"} : <f32>
    %73 = mulf %72, %71 {fastmath = #arith.fastmath<none>, handshake.bb = 3 : ui32, handshake.name = "mulf3"} : <f32>
    %74 = cmpf olt, %73, %67#1 {handshake.bb = 3 : ui32, handshake.name = "cmpf1"} : <f32>
    %75:3 = fork [3] %74 {handshake.bb = 3 : ui32, handshake.name = "fork29"} : <i1>
    %76 = not %75#2 {handshake.bb = 3 : ui32, handshake.name = "not1"} : <i1>
    %77:9 = fork [9] %76 {handshake.bb = 3 : ui32, handshake.name = "fork30"} : <i1>
    %78 = passer %57#1[%75#1] {handshake.bb = 3 : ui32, handshake.name = "passer10"} : <f32>, <i1>
    %79 = passer %57#0[%77#8] {handshake.bb = 3 : ui32, handshake.name = "passer11"} : <f32>, <i1>
    %80:2 = fork [2] %79 {handshake.bb = 3 : ui32, handshake.name = "fork31"} : <f32>
    %81 = passer %60#1[%75#0] {handshake.bb = 3 : ui32, handshake.name = "passer12"} : <>, <i1>
    %82 = passer %60#0[%77#7] {handshake.bb = 3 : ui32, handshake.name = "passer13"} : <>, <i1>
    %83 = passer %67#0[%77#6] {handshake.bb = 3 : ui32, handshake.name = "passer14"} : <f32>, <i1>
    %84:2 = fork [2] %83 {handshake.bb = 3 : ui32, handshake.name = "fork32"} : <f32>
    %85 = passer %68#0[%77#5] {handshake.bb = 3 : ui32, handshake.name = "passer15"} : <f32>, <i1>
    %86 = passer %69#0[%77#4] {handshake.bb = 3 : ui32, handshake.name = "passer16"} : <f32>, <i1>
    %87 = passer %64[%77#3] {handshake.bb = 3 : ui32, handshake.name = "passer17"} : <i8>, <i1>
    %88 = passer %65[%77#2] {handshake.bb = 3 : ui32, handshake.name = "passer18"} : <f32>, <i1>
    %89 = passer %66[%77#1] {handshake.bb = 3 : ui32, handshake.name = "passer19"} : <f32>, <i1>
    %90 = extsi %87 {handshake.bb = 4 : ui32, handshake.name = "extsi4"} : <i8> to <i9>
    %91:2 = fork [2] %88 {handshake.bb = 4 : ui32, handshake.name = "fork14"} : <f32>
    %92:2 = fork [2] %89 {handshake.bb = 4 : ui32, handshake.name = "fork15"} : <f32>
    %93:3 = fork [3] %82 {handshake.bb = 4 : ui32, handshake.name = "fork33"} : <>
    %94 = constant %93#0 {handshake.bb = 4 : ui32, handshake.name = "constant9", value = 0.000000e+00 : f32} : <>, <f32>
    %95:3 = fork [3] %94 {handshake.bb = 4 : ui32, handshake.name = "fork34"} : <f32>
    %96 = source {handshake.bb = 4 : ui32, handshake.name = "source4"} : <>
    %97 = constant %96 {handshake.bb = 4 : ui32, handshake.name = "constant1", value = 1 : i2} : <>, <i2>
    %98 = extsi %97 {handshake.bb = 4 : ui32, handshake.name = "extsi5"} : <i2> to <i9>
    %99 = source {handshake.bb = 4 : ui32, handshake.name = "source5"} : <>
    %100 = constant %99 {handshake.bb = 4 : ui32, handshake.name = "constant2", value = 100 : i8} : <>, <i8>
    %101 = extsi %100 {handshake.bb = 4 : ui32, handshake.name = "extsi6"} : <i8> to <i9>
    %102 = mulf %91#1, %92#1 {fastmath = #arith.fastmath<none>, handshake.bb = 4 : ui32, handshake.name = "mulf4"} : <f32>
    %103 = cmpf olt, %102, %95#0 {handshake.bb = 4 : ui32, handshake.name = "cmpf2"} : <f32>
    %104:10 = fork [10] %103 {handshake.bb = 4 : ui32, handshake.name = "fork35"} : <i1>
    %105 = addi %90, %98 {handshake.bb = 4 : ui32, handshake.name = "addi0"} : <i9>
    %106:2 = fork [2] %105 {handshake.bb = 4 : ui32, handshake.name = "fork19"} : <i9>
    %107 = trunci %106#0 {handshake.bb = 4 : ui32, handshake.name = "trunci0"} : <i9> to <i8>
    %108:2 = fork [2] %107 {handshake.bb = 4 : ui32, handshake.name = "fork36"} : <i8>
    %109 = cmpi ult, %106#1, %101 {handshake.bb = 4 : ui32, handshake.name = "cmpi0"} : <i9>
    %110:2 = fork [2] %109 {handshake.bb = 4 : ui32, handshake.name = "fork37"} : <i1>
    %111 = not %104#9 {handshake.bb = 4 : ui32, handshake.name = "not2"} : <i1>
    %112:8 = fork [8] %111 {handshake.bb = 4 : ui32, handshake.name = "fork38"} : <i1>
    %113 = passer %84#1[%104#8] {handshake.bb = 4 : ui32, handshake.name = "passer20"} : <f32>, <i1>
    %114 = passer %84#0[%112#7] {handshake.bb = 4 : ui32, handshake.name = "passer21"} : <f32>, <i1>
    %115 = passer %85[%104#7] {handshake.bb = 4 : ui32, handshake.name = "passer22"} : <f32>, <i1>
    %116 = passer %91#0[%104#6] {handshake.bb = 4 : ui32, handshake.name = "passer23"} : <f32>, <i1>
    %117 = passer %80#1[%104#5] {handshake.bb = 4 : ui32, handshake.name = "passer24"} : <f32>, <i1>
    %118 = passer %80#0[%112#6] {handshake.bb = 4 : ui32, handshake.name = "passer25"} : <f32>, <i1>
    %119 = passer %95#2[%104#4] {handshake.bb = 4 : ui32, handshake.name = "passer26"} : <f32>, <i1>
    %120 = passer %95#1[%112#5] {handshake.bb = 4 : ui32, handshake.name = "passer27"} : <f32>, <i1>
    %121 = passer %108#1[%104#3] {handshake.bb = 4 : ui32, handshake.name = "passer28"} : <i8>, <i1>
    %122 = passer %108#0[%112#4] {handshake.bb = 4 : ui32, handshake.name = "passer29"} : <i8>, <i1>
    %123 = passer %110#1[%104#2] {handshake.bb = 4 : ui32, handshake.name = "passer30"} : <i1>, <i1>
    %124 = passer %110#0[%112#3] {handshake.bb = 4 : ui32, handshake.name = "passer31"} : <i1>, <i1>
    %125 = passer %93#2[%104#1] {handshake.bb = 4 : ui32, handshake.name = "passer32"} : <>, <i1>
    %126:2 = fork [2] %125 {handshake.bb = 4 : ui32, handshake.name = "fork39"} : <>
    %127 = passer %93#1[%112#2] {handshake.bb = 4 : ui32, handshake.name = "passer33"} : <>, <i1>
    %128 = passer %86[%112#1] {handshake.bb = 4 : ui32, handshake.name = "passer34"} : <f32>, <i1>
    %129 = passer %92#0[%112#0] {handshake.bb = 4 : ui32, handshake.name = "passer35"} : <f32>, <i1>
    %130:8 = fork [8] %123 {handshake.bb = 5 : ui32, handshake.name = "fork40"} : <i1>
    %131 = not %130#7 {handshake.bb = 5 : ui32, handshake.name = "not3"} : <i1>
    %132:2 = fork [2] %131 {handshake.bb = 5 : ui32, handshake.name = "fork41"} : <i1>
    %133 = passer %115[%130#6] {handshake.bb = 5 : ui32, handshake.name = "passer36"} : <f32>, <i1>
    %134 = passer %117[%130#5] {handshake.bb = 5 : ui32, handshake.name = "passer37"} : <f32>, <i1>
    %135 = passer %121[%130#4] {handshake.bb = 5 : ui32, handshake.name = "passer38"} : <i8>, <i1>
    %136 = passer %116[%130#3] {handshake.bb = 5 : ui32, handshake.name = "passer39"} : <f32>, <i1>
    %137 = passer %113[%130#2] {handshake.bb = 5 : ui32, handshake.name = "passer40"} : <f32>, <i1>
    %138 = passer %126#1[%130#1] {handshake.bb = 5 : ui32, handshake.name = "passer41"} : <>, <i1>
    %139 = passer %126#0[%132#1] {handshake.bb = 5 : ui32, handshake.name = "passer42"} : <>, <i1>
    %140 = passer %119[%132#0] {handshake.bb = 5 : ui32, handshake.name = "passer43"} : <f32>, <i1>
    %141:7 = fork [7] %124 {handshake.bb = 6 : ui32, handshake.name = "fork21"} : <i1>
    %trueResult, %falseResult = cond_br %141#6, %118 {handshake.bb = 6 : ui32, handshake.name = "cond_br38"} : <i1>, <f32>
    sink %falseResult {handshake.name = "sink26"} : <f32>
    %trueResult_0, %falseResult_1 = cond_br %141#5, %128 {handshake.bb = 6 : ui32, handshake.name = "cond_br39"} : <i1>, <f32>
    sink %falseResult_1 {handshake.name = "sink27"} : <f32>
    %trueResult_2, %falseResult_3 = cond_br %141#0, %122 {handshake.bb = 6 : ui32, handshake.name = "cond_br40"} : <i1>, <i8>
    sink %falseResult_3 {handshake.name = "sink28"} : <i8>
    %trueResult_4, %falseResult_5 = cond_br %141#4, %129 {handshake.bb = 6 : ui32, handshake.name = "cond_br41"} : <i1>, <f32>
    sink %falseResult_5 {handshake.name = "sink29"} : <f32>
    %trueResult_6, %falseResult_7 = cond_br %141#3, %114 {handshake.bb = 6 : ui32, handshake.name = "cond_br42"} : <i1>, <f32>
    sink %falseResult_7 {handshake.name = "sink30"} : <f32>
    %trueResult_8, %falseResult_9 = cond_br %141#2, %127 {handshake.bb = 6 : ui32, handshake.name = "cond_br43"} : <i1>, <>
    %trueResult_10, %falseResult_11 = cond_br %141#1, %120 {handshake.bb = 6 : ui32, handshake.name = "cond_br44"} : <i1>, <f32>
    sink %trueResult_10 {handshake.name = "sink31"} : <f32>
    %142 = mux %index_13 [%55, %140, %falseResult_11] {handshake.bb = 7 : ui32, handshake.name = "mux10"} : <i2>, [<f32>, <f32>, <f32>] to <f32>
    %result_12, %index_13 = control_merge [%58, %139, %falseResult_9]  {handshake.bb = 7 : ui32, handshake.name = "control_merge6"} : [<>, <>, <>] to <>, <i2>
    %143 = mux %index_15 [%78, %142] {handshake.bb = 8 : ui32, handshake.name = "mux11"} : <i1>, [<f32>, <f32>] to <f32>
    %result_14, %index_15 = control_merge [%81, %result_12]  {handshake.bb = 8 : ui32, handshake.name = "control_merge7"} : [<>, <>] to <>, <i1>
    sink %result_14 {handshake.name = "sink32"} : <>
    end {handshake.bb = 8 : ui32, handshake.name = "end0"} %143, %10#1 : <f32>, <>
  }
}

