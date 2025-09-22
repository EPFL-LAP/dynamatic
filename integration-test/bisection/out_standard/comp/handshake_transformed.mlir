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
    %8 = mux %16#1 [%1#0, %trueResult_30] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<f32>, <f32>] to <f32>
    %9:2 = fork [2] %8 {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <f32>
    %10 = mux %16#2 [%arg1, %trueResult_32] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i1>, [<f32>, <f32>] to <f32>
    %11:2 = fork [2] %10 {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <f32>
    %12 = mux %16#0 [%7, %trueResult_34] {handshake.bb = 1 : ui32, handshake.name = "mux2"} : <i1>, [<i8>, <i8>] to <i8>
    %13 = mux %16#3 [%6, %trueResult_36] {handshake.bb = 1 : ui32, handshake.name = "mux3"} : <i1>, [<f32>, <f32>] to <f32>
    %14 = mux %16#4 [%arg2, %trueResult_38] {handshake.bb = 1 : ui32, handshake.name = "mux4"} : <i1>, [<f32>, <f32>] to <f32>
    %15:2 = fork [2] %14 {handshake.bb = 1 : ui32, handshake.name = "fork4"} : <f32>
    %result, %index = control_merge [%0#2, %trueResult_40]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %16:5 = fork [5] %index {handshake.bb = 1 : ui32, handshake.name = "fork5"} : <i1>
    %17 = source {handshake.bb = 1 : ui32, handshake.name = "source1"} : <>
    %18 = constant %17 {handshake.bb = 1 : ui32, handshake.name = "constant6", value = -2.000000e+00 : f32} : <>, <f32>
    %19 = source {handshake.bb = 1 : ui32, handshake.name = "source2"} : <>
    %20 = constant %19 {handshake.bb = 1 : ui32, handshake.name = "constant7", value = 5.000000e-01 : f32} : <>, <f32>
    %21 = addf %9#1, %11#1 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "addf1"} : <f32>
    %22 = mulf %21, %20 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "mulf1"} : <f32>
    %23:3 = fork [3] %22 {handshake.bb = 1 : ui32, handshake.name = "fork6"} : <f32>
    %24 = mulf %23#1, %23#2 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "mulf2"} : <f32>
    %25 = addf %24, %18 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "addf2"} : <f32>
    %26:2 = fork [2] %25 {handshake.bb = 1 : ui32, handshake.name = "fork7"} : <f32>
    %27 = absf %26#1 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "absf0"} : <f32> to <f32>
    %28 = cmpf olt, %27, %15#1 {handshake.bb = 1 : ui32, handshake.name = "cmpf0"} : <f32>
    %29:8 = fork [8] %28 {handshake.bb = 1 : ui32, handshake.name = "fork8"} : <i1>
    %trueResult, %falseResult = cond_br %29#7, %23#0 {handshake.bb = 1 : ui32, handshake.name = "cond_br3"} : <i1>, <f32>
    %trueResult_0, %falseResult_1 = cond_br %29#6, %result {handshake.bb = 1 : ui32, handshake.name = "cond_br4"} : <i1>, <>
    %trueResult_2, %falseResult_3 = cond_br %29#5, %15#0 {handshake.bb = 1 : ui32, handshake.name = "cond_br5"} : <i1>, <f32>
    sink %trueResult_2 {handshake.name = "sink0"} : <f32>
    %trueResult_4, %falseResult_5 = cond_br %29#4, %9#0 {handshake.bb = 1 : ui32, handshake.name = "cond_br6"} : <i1>, <f32>
    sink %trueResult_4 {handshake.name = "sink1"} : <f32>
    %trueResult_6, %falseResult_7 = cond_br %29#3, %11#0 {handshake.bb = 1 : ui32, handshake.name = "cond_br7"} : <i1>, <f32>
    sink %trueResult_6 {handshake.name = "sink2"} : <f32>
    %trueResult_8, %falseResult_9 = cond_br %29#0, %12 {handshake.bb = 1 : ui32, handshake.name = "cond_br8"} : <i1>, <i8>
    sink %trueResult_8 {handshake.name = "sink3"} : <i8>
    %trueResult_10, %falseResult_11 = cond_br %29#2, %13 {handshake.bb = 1 : ui32, handshake.name = "cond_br9"} : <i1>, <f32>
    sink %trueResult_10 {handshake.name = "sink4"} : <f32>
    %trueResult_12, %falseResult_13 = cond_br %29#1, %26#0 {handshake.bb = 1 : ui32, handshake.name = "cond_br10"} : <i1>, <f32>
    sink %trueResult_12 {handshake.name = "sink5"} : <f32>
    %30:2 = fork [2] %falseResult_3 {handshake.bb = 2 : ui32, handshake.name = "fork9"} : <f32>
    %31:2 = fork [2] %falseResult_5 {handshake.bb = 2 : ui32, handshake.name = "fork10"} : <f32>
    %32:2 = fork [2] %falseResult_7 {handshake.bb = 2 : ui32, handshake.name = "fork11"} : <f32>
    %33 = source {handshake.bb = 2 : ui32, handshake.name = "source3"} : <>
    %34 = constant %33 {handshake.bb = 2 : ui32, handshake.name = "constant8", value = 5.000000e-01 : f32} : <>, <f32>
    %35 = subf %32#1, %31#1 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "subf0"} : <f32>
    %36 = mulf %35, %34 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "mulf3"} : <f32>
    %37 = cmpf olt, %36, %30#1 {handshake.bb = 2 : ui32, handshake.name = "cmpf1"} : <f32>
    %38:8 = fork [8] %37 {handshake.bb = 2 : ui32, handshake.name = "fork12"} : <i1>
    %trueResult_14, %falseResult_15 = cond_br %38#7, %falseResult {handshake.bb = 2 : ui32, handshake.name = "cond_br11"} : <i1>, <f32>
    %trueResult_16, %falseResult_17 = cond_br %38#6, %falseResult_1 {handshake.bb = 2 : ui32, handshake.name = "cond_br12"} : <i1>, <>
    %trueResult_18, %falseResult_19 = cond_br %38#5, %30#0 {handshake.bb = 2 : ui32, handshake.name = "cond_br13"} : <i1>, <f32>
    sink %trueResult_18 {handshake.name = "sink7"} : <f32>
    %trueResult_20, %falseResult_21 = cond_br %38#4, %31#0 {handshake.bb = 2 : ui32, handshake.name = "cond_br14"} : <i1>, <f32>
    sink %trueResult_20 {handshake.name = "sink8"} : <f32>
    %trueResult_22, %falseResult_23 = cond_br %38#3, %32#0 {handshake.bb = 2 : ui32, handshake.name = "cond_br15"} : <i1>, <f32>
    sink %trueResult_22 {handshake.name = "sink9"} : <f32>
    %trueResult_24, %falseResult_25 = cond_br %38#0, %falseResult_9 {handshake.bb = 2 : ui32, handshake.name = "cond_br16"} : <i1>, <i8>
    sink %trueResult_24 {handshake.name = "sink10"} : <i8>
    %trueResult_26, %falseResult_27 = cond_br %38#2, %falseResult_11 {handshake.bb = 2 : ui32, handshake.name = "cond_br17"} : <i1>, <f32>
    sink %trueResult_26 {handshake.name = "sink11"} : <f32>
    %trueResult_28, %falseResult_29 = cond_br %38#1, %falseResult_13 {handshake.bb = 2 : ui32, handshake.name = "cond_br18"} : <i1>, <f32>
    sink %trueResult_28 {handshake.name = "sink12"} : <f32>
    %39 = extsi %falseResult_25 {handshake.bb = 3 : ui32, handshake.name = "extsi4"} : <i8> to <i9>
    %40:2 = fork [2] %falseResult_27 {handshake.bb = 3 : ui32, handshake.name = "fork13"} : <f32>
    %41:2 = fork [2] %falseResult_15 {handshake.bb = 3 : ui32, handshake.name = "fork14"} : <f32>
    %42:2 = fork [2] %falseResult_29 {handshake.bb = 3 : ui32, handshake.name = "fork15"} : <f32>
    %43:2 = fork [2] %falseResult_17 {handshake.bb = 3 : ui32, handshake.name = "fork16"} : <>
    %44 = constant %43#1 {handshake.bb = 3 : ui32, handshake.name = "constant9", value = 0.000000e+00 : f32} : <>, <f32>
    %45:2 = fork [2] %44 {handshake.bb = 3 : ui32, handshake.name = "fork17"} : <f32>
    %46 = source {handshake.bb = 3 : ui32, handshake.name = "source4"} : <>
    %47 = constant %46 {handshake.bb = 3 : ui32, handshake.name = "constant1", value = 1 : i2} : <>, <i2>
    %48 = extsi %47 {handshake.bb = 3 : ui32, handshake.name = "extsi5"} : <i2> to <i9>
    %49 = source {handshake.bb = 3 : ui32, handshake.name = "source5"} : <>
    %50 = constant %49 {handshake.bb = 3 : ui32, handshake.name = "constant2", value = 100 : i8} : <>, <i8>
    %51 = extsi %50 {handshake.bb = 3 : ui32, handshake.name = "extsi6"} : <i8> to <i9>
    %52 = mulf %40#1, %42#1 {fastmath = #arith.fastmath<none>, handshake.bb = 3 : ui32, handshake.name = "mulf4"} : <f32>
    %53 = cmpf olt, %52, %45#1 {handshake.bb = 3 : ui32, handshake.name = "cmpf2"} : <f32>
    %54:3 = fork [3] %53 {handshake.bb = 3 : ui32, handshake.name = "fork18"} : <i1>
    %55 = select %54#2[%41#1, %falseResult_23] {handshake.bb = 3 : ui32, handshake.name = "select0"} : <i1>, <f32>
    %56 = select %54#1[%falseResult_21, %41#0] {handshake.bb = 3 : ui32, handshake.name = "select1"} : <i1>, <f32>
    %57 = select %54#0[%40#0, %42#0] {handshake.bb = 3 : ui32, handshake.name = "select2"} : <i1>, <f32>
    %58 = addi %39, %48 {handshake.bb = 3 : ui32, handshake.name = "addi0"} : <i9>
    %59:2 = fork [2] %58 {handshake.bb = 3 : ui32, handshake.name = "fork19"} : <i9>
    %60 = trunci %59#0 {handshake.bb = 3 : ui32, handshake.name = "trunci0"} : <i9> to <i8>
    %61 = cmpi ult, %59#1, %51 {handshake.bb = 3 : ui32, handshake.name = "cmpi0"} : <i9>
    %62:7 = fork [7] %61 {handshake.bb = 3 : ui32, handshake.name = "fork20"} : <i1>
    %trueResult_30, %falseResult_31 = cond_br %62#1, %56 {handshake.bb = 3 : ui32, handshake.name = "cond_br19"} : <i1>, <f32>
    sink %falseResult_31 {handshake.name = "sink14"} : <f32>
    %trueResult_32, %falseResult_33 = cond_br %62#2, %55 {handshake.bb = 3 : ui32, handshake.name = "cond_br20"} : <i1>, <f32>
    sink %falseResult_33 {handshake.name = "sink15"} : <f32>
    %trueResult_34, %falseResult_35 = cond_br %62#0, %60 {handshake.bb = 3 : ui32, handshake.name = "cond_br21"} : <i1>, <i8>
    sink %falseResult_35 {handshake.name = "sink16"} : <i8>
    %trueResult_36, %falseResult_37 = cond_br %62#3, %57 {handshake.bb = 3 : ui32, handshake.name = "cond_br22"} : <i1>, <f32>
    sink %falseResult_37 {handshake.name = "sink17"} : <f32>
    %trueResult_38, %falseResult_39 = cond_br %62#4, %falseResult_19 {handshake.bb = 3 : ui32, handshake.name = "cond_br23"} : <i1>, <f32>
    sink %falseResult_39 {handshake.name = "sink18"} : <f32>
    %trueResult_40, %falseResult_41 = cond_br %62#5, %43#0 {handshake.bb = 3 : ui32, handshake.name = "cond_br24"} : <i1>, <>
    %trueResult_42, %falseResult_43 = cond_br %62#6, %45#0 {handshake.bb = 3 : ui32, handshake.name = "cond_br25"} : <i1>, <f32>
    sink %trueResult_42 {handshake.name = "sink19"} : <f32>
    %63 = mux %index_45 [%trueResult, %falseResult_43] {handshake.bb = 4 : ui32, handshake.name = "mux5"} : <i1>, [<f32>, <f32>] to <f32>
    %result_44, %index_45 = control_merge [%trueResult_0, %falseResult_41]  {handshake.bb = 4 : ui32, handshake.name = "control_merge3"} : [<>, <>] to <>, <i1>
    %64 = mux %index_47 [%trueResult_14, %63] {handshake.bb = 5 : ui32, handshake.name = "mux6"} : <i1>, [<f32>, <f32>] to <f32>
    %result_46, %index_47 = control_merge [%trueResult_16, %result_44]  {handshake.bb = 5 : ui32, handshake.name = "control_merge4"} : [<>, <>] to <>, <i1>
    sink %result_46 {handshake.name = "sink20"} : <>
    end {handshake.bb = 5 : ui32, handshake.name = "end0"} %64, %0#1 : <f32>, <>
  }
}

