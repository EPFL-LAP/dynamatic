module {
  handshake.func @golden_ratio(%arg0: !handshake.channel<f32>, %arg1: !handshake.control<>, ...) -> (!handshake.channel<f32>, !handshake.control<>) attributes {argNames = ["x0", "start"], resNames = ["out0", "end"]} {
    %0:3 = fork [3] %arg1 {handshake.bb = 0 : ui32, handshake.name = "fork0"} : <>
    %1:2 = fork [2] %arg0 {handshake.bb = 0 : ui32, handshake.name = "fork1"} : <f32>
    %2 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant0", value = false} : <>, <i1>
    %3 = source {handshake.bb = 0 : ui32, handshake.name = "source0"} : <>
    %4 = constant %3 {handshake.bb = 0 : ui32, handshake.name = "constant5", value = 1.000000e+00 : f32} : <>, <f32>
    %5 = divf %4, %1#1 {fastmath = #arith.fastmath<none>, handshake.bb = 0 : ui32, handshake.name = "divf0"} : <f32>
    %6 = extsi %2 {handshake.bb = 0 : ui32, handshake.name = "extsi3"} : <i1> to <i8>
    %7 = mux %10#1 [%5, %trueResult_10] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<f32>, <f32>] to <f32>
    %8 = mux %10#0 [%6, %trueResult_12] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i1>, [<i8>, <i8>] to <i8>
    %9 = mux %10#2 [%1#0, %trueResult_14] {handshake.bb = 1 : ui32, handshake.name = "mux2"} : <i1>, [<f32>, <f32>] to <f32>
    %result, %index = control_merge [%0#2, %trueResult_16]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %10:3 = fork [3] %index {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <i1>
    %11 = mux %16#1 [%9, %falseResult_7] {handshake.bb = 2 : ui32, handshake.name = "mux3"} : <i1>, [<f32>, <f32>] to <f32>
    %12:4 = fork [4] %11 {handshake.bb = 2 : ui32, handshake.name = "fork3"} : <f32>
    %13 = mux %16#2 [%7, %falseResult_9] {handshake.bb = 2 : ui32, handshake.name = "mux4"} : <i1>, [<f32>, <f32>] to <f32>
    %14:2 = fork [2] %13 {handshake.bb = 2 : ui32, handshake.name = "fork4"} : <f32>
    %15 = mux %16#0 [%8, %falseResult_3] {handshake.bb = 2 : ui32, handshake.name = "mux5"} : <i1>, [<i8>, <i8>] to <i8>
    %result_0, %index_1 = control_merge [%result, %falseResult_5]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>, <>] to <>, <i1>
    %16:3 = fork [3] %index_1 {handshake.bb = 2 : ui32, handshake.name = "fork5"} : <i1>
    %17 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %18 = constant %17 {handshake.bb = 2 : ui32, handshake.name = "constant6", value = 5.000000e-01 : f32} : <>, <f32>
    %19 = source {handshake.bb = 2 : ui32, handshake.name = "source2"} : <>
    %20 = constant %19 {handshake.bb = 2 : ui32, handshake.name = "constant7", value = 1.000000e-01 : f32} : <>, <f32>
    %21 = mulf %12#3, %14#1 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "mulf0"} : <f32>
    %22 = addf %12#2, %21 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "addf0"} : <f32>
    %23 = mulf %22, %18 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "mulf1"} : <f32>
    %24:2 = fork [2] %23 {handshake.bb = 2 : ui32, handshake.name = "fork6"} : <f32>
    %25 = subf %24#1, %12#1 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "subf0"} : <f32>
    %26 = absf %25 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "absf0"} : <f32> to <f32>
    %27 = cmpf olt, %26, %20 {handshake.bb = 2 : ui32, handshake.name = "cmpf0"} : <f32>
    %28:5 = fork [5] %27 {handshake.bb = 2 : ui32, handshake.name = "fork7"} : <i1>
    %trueResult, %falseResult = cond_br %28#4, %12#0 {handshake.bb = 2 : ui32, handshake.name = "cond_br2"} : <i1>, <f32>
    sink %falseResult {handshake.name = "sink0"} : <f32>
    %trueResult_2, %falseResult_3 = cond_br %28#0, %15 {handshake.bb = 2 : ui32, handshake.name = "cond_br3"} : <i1>, <i8>
    %trueResult_4, %falseResult_5 = cond_br %28#3, %result_0 {handshake.bb = 2 : ui32, handshake.name = "cond_br4"} : <i1>, <>
    %trueResult_6, %falseResult_7 = cond_br %28#2, %24#0 {handshake.bb = 2 : ui32, handshake.name = "cond_br5"} : <i1>, <f32>
    sink %trueResult_6 {handshake.name = "sink1"} : <f32>
    %trueResult_8, %falseResult_9 = cond_br %28#1, %14#0 {handshake.bb = 2 : ui32, handshake.name = "cond_br6"} : <i1>, <f32>
    sink %trueResult_8 {handshake.name = "sink2"} : <f32>
    %29 = extsi %trueResult_2 {handshake.bb = 3 : ui32, handshake.name = "extsi4"} : <i8> to <i9>
    %30 = source {handshake.bb = 3 : ui32, handshake.name = "source3"} : <>
    %31 = constant %30 {handshake.bb = 3 : ui32, handshake.name = "constant8", value = 1.000000e+00 : f32} : <>, <f32>
    %32:2 = fork [2] %31 {handshake.bb = 3 : ui32, handshake.name = "fork8"} : <f32>
    %33 = source {handshake.bb = 3 : ui32, handshake.name = "source4"} : <>
    %34 = constant %33 {handshake.bb = 3 : ui32, handshake.name = "constant1", value = 1 : i2} : <>, <i2>
    %35 = extsi %34 {handshake.bb = 3 : ui32, handshake.name = "extsi5"} : <i2> to <i9>
    %36 = source {handshake.bb = 3 : ui32, handshake.name = "source5"} : <>
    %37 = constant %36 {handshake.bb = 3 : ui32, handshake.name = "constant2", value = 100 : i8} : <>, <i8>
    %38 = extsi %37 {handshake.bb = 3 : ui32, handshake.name = "extsi6"} : <i8> to <i9>
    %39 = addf %trueResult, %32#1 {fastmath = #arith.fastmath<none>, handshake.bb = 3 : ui32, handshake.name = "addf1"} : <f32>
    %40:2 = fork [2] %39 {handshake.bb = 3 : ui32, handshake.name = "fork9"} : <f32>
    %41 = addi %29, %35 {handshake.bb = 3 : ui32, handshake.name = "addi0"} : <i9>
    %42:2 = fork [2] %41 {handshake.bb = 3 : ui32, handshake.name = "fork10"} : <i9>
    %43 = trunci %42#0 {handshake.bb = 3 : ui32, handshake.name = "trunci0"} : <i9> to <i8>
    %44 = divf %32#0, %40#1 {fastmath = #arith.fastmath<none>, handshake.bb = 3 : ui32, handshake.name = "divf1"} : <f32>
    %45 = cmpi ult, %42#1, %38 {handshake.bb = 3 : ui32, handshake.name = "cmpi0"} : <i9>
    %46:4 = fork [4] %45 {handshake.bb = 3 : ui32, handshake.name = "fork11"} : <i1>
    %trueResult_10, %falseResult_11 = cond_br %46#1, %44 {handshake.bb = 3 : ui32, handshake.name = "cond_br7"} : <i1>, <f32>
    sink %falseResult_11 {handshake.name = "sink4"} : <f32>
    %trueResult_12, %falseResult_13 = cond_br %46#0, %43 {handshake.bb = 3 : ui32, handshake.name = "cond_br8"} : <i1>, <i8>
    sink %falseResult_13 {handshake.name = "sink5"} : <i8>
    %trueResult_14, %falseResult_15 = cond_br %46#2, %40#0 {handshake.bb = 3 : ui32, handshake.name = "cond_br9"} : <i1>, <f32>
    %trueResult_16, %falseResult_17 = cond_br %46#3, %trueResult_4 {handshake.bb = 3 : ui32, handshake.name = "cond_br10"} : <i1>, <>
    sink %falseResult_17 {handshake.name = "sink6"} : <>
    end {handshake.bb = 4 : ui32, handshake.name = "end0"} %falseResult_15, %0#1 : <f32>, <>
  }
}

