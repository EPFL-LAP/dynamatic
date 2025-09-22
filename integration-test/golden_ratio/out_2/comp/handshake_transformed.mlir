module {
  handshake.func @golden_ratio(%arg0: !handshake.channel<f32>, %arg1: !handshake.channel<f32>, %arg2: !handshake.control<>, ...) -> (!handshake.channel<f32>, !handshake.control<>) attributes {argNames = ["x0", "x1", "start"], resNames = ["out0", "end"]} {
    %0:3 = fork [3] %arg2 {handshake.bb = 0 : ui32, handshake.name = "fork0"} : <>
    %1 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant0", value = false} : <>, <i1>
    %2 = extsi %1 {handshake.bb = 0 : ui32, handshake.name = "extsi3"} : <i1> to <i8>
    %3 = mux %6#0 [%2, %trueResult_10] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i8>, <i8>] to <i8>
    %4 = mux %6#1 [%arg1, %trueResult_12] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i1>, [<f32>, <f32>] to <f32>
    %5 = mux %6#2 [%arg0, %trueResult_14] {handshake.bb = 1 : ui32, handshake.name = "mux2"} : <i1>, [<f32>, <f32>] to <f32>
    %result, %index = control_merge [%0#2, %trueResult_16]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %6:3 = fork [3] %index {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <i1>
    %7 = mux %12#1 [%5, %falseResult_7] {handshake.bb = 2 : ui32, handshake.name = "mux3"} : <i1>, [<f32>, <f32>] to <f32>
    %8:4 = fork [4] %7 {handshake.bb = 2 : ui32, handshake.name = "fork2"} : <f32>
    %9 = mux %12#0 [%3, %falseResult_3] {handshake.bb = 2 : ui32, handshake.name = "mux4"} : <i1>, [<i8>, <i8>] to <i8>
    %10 = mux %12#2 [%4, %falseResult_9] {handshake.bb = 2 : ui32, handshake.name = "mux5"} : <i1>, [<f32>, <f32>] to <f32>
    %11:2 = fork [2] %10 {handshake.bb = 2 : ui32, handshake.name = "fork3"} : <f32>
    %result_0, %index_1 = control_merge [%result, %falseResult_5]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>, <>] to <>, <i1>
    %12:3 = fork [3] %index_1 {handshake.bb = 2 : ui32, handshake.name = "fork4"} : <i1>
    %13 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %14 = constant %13 {handshake.bb = 2 : ui32, handshake.name = "constant6", value = 5.000000e-01 : f32} : <>, <f32>
    %15 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %16 = constant %15 {handshake.bb = 2 : ui32, handshake.name = "constant7", value = 1.000000e-01 : f32} : <>, <f32>
    %17 = mulf %8#3, %11#1 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "mulf0"} : <f32>
    %18 = addf %8#2, %17 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "addf0"} : <f32>
    %19 = mulf %18, %14 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "mulf1"} : <f32>
    %20:2 = fork [2] %19 {handshake.bb = 2 : ui32, handshake.name = "fork5"} : <f32>
    %21 = subf %20#1, %8#1 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "subf0"} : <f32>
    %22 = absf %21 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "absf0"} : <f32> to <f32>
    %23 = cmpf olt, %22, %16 {handshake.bb = 2 : ui32, handshake.name = "cmpf0"} : <f32>
    %24:5 = fork [5] %23 {handshake.bb = 2 : ui32, handshake.name = "fork6"} : <i1>
    %trueResult, %falseResult = cond_br %24#4, %8#0 {handshake.bb = 2 : ui32, handshake.name = "cond_br2"} : <i1>, <f32>
    sink %falseResult {handshake.name = "sink0"} : <f32>
    %trueResult_2, %falseResult_3 = cond_br %24#0, %9 {handshake.bb = 2 : ui32, handshake.name = "cond_br3"} : <i1>, <i8>
    %trueResult_4, %falseResult_5 = cond_br %24#3, %result_0 {handshake.bb = 2 : ui32, handshake.name = "cond_br4"} : <i1>, <>
    %trueResult_6, %falseResult_7 = cond_br %24#2, %20#0 {handshake.bb = 2 : ui32, handshake.name = "cond_br5"} : <i1>, <f32>
    sink %trueResult_6 {handshake.name = "sink1"} : <f32>
    %trueResult_8, %falseResult_9 = cond_br %24#1, %11#0 {handshake.bb = 2 : ui32, handshake.name = "cond_br6"} : <i1>, <f32>
    sink %trueResult_8 {handshake.name = "sink2"} : <f32>
    %25 = extsi %trueResult_2 {handshake.bb = 3 : ui32, handshake.name = "extsi4"} : <i8> to <i9>
    %26 = source {handshake.bb = 3 : ui32, handshake.name = "source2"} : <>
    %27 = constant %26 {handshake.bb = 3 : ui32, handshake.name = "constant8", value = 1.000000e+00 : f32} : <>, <f32>
    %28:2 = fork [2] %27 {handshake.bb = 3 : ui32, handshake.name = "fork7"} : <f32>
    %29 = source {handshake.bb = 3 : ui32, handshake.name = "source3"} : <>
    %30 = constant %29 {handshake.bb = 3 : ui32, handshake.name = "constant1", value = 1 : i2} : <>, <i2>
    %31 = extsi %30 {handshake.bb = 3 : ui32, handshake.name = "extsi5"} : <i2> to <i9>
    %32 = source {handshake.bb = 3 : ui32, handshake.name = "source4"} : <>
    %33 = constant %32 {handshake.bb = 3 : ui32, handshake.name = "constant2", value = 100 : i8} : <>, <i8>
    %34 = extsi %33 {handshake.bb = 3 : ui32, handshake.name = "extsi6"} : <i8> to <i9>
    %35 = addf %trueResult, %28#1 {fastmath = #arith.fastmath<none>, handshake.bb = 3 : ui32, handshake.name = "addf1"} : <f32>
    %36:2 = fork [2] %35 {handshake.bb = 3 : ui32, handshake.name = "fork8"} : <f32>
    %37 = divf %28#0, %36#1 {fastmath = #arith.fastmath<none>, handshake.bb = 3 : ui32, handshake.name = "divf0"} : <f32>
    %38 = addi %25, %31 {handshake.bb = 3 : ui32, handshake.name = "addi0"} : <i9>
    %39:2 = fork [2] %38 {handshake.bb = 3 : ui32, handshake.name = "fork9"} : <i9>
    %40 = trunci %39#0 {handshake.bb = 3 : ui32, handshake.name = "trunci0"} : <i9> to <i8>
    %41 = cmpi ult, %39#1, %34 {handshake.bb = 3 : ui32, handshake.name = "cmpi0"} : <i9>
    %42:4 = fork [4] %41 {handshake.bb = 3 : ui32, handshake.name = "fork10"} : <i1>
    %trueResult_10, %falseResult_11 = cond_br %42#0, %40 {handshake.bb = 3 : ui32, handshake.name = "cond_br7"} : <i1>, <i8>
    sink %falseResult_11 {handshake.name = "sink4"} : <i8>
    %trueResult_12, %falseResult_13 = cond_br %42#1, %37 {handshake.bb = 3 : ui32, handshake.name = "cond_br8"} : <i1>, <f32>
    sink %falseResult_13 {handshake.name = "sink5"} : <f32>
    %trueResult_14, %falseResult_15 = cond_br %42#2, %36#0 {handshake.bb = 3 : ui32, handshake.name = "cond_br9"} : <i1>, <f32>
    %trueResult_16, %falseResult_17 = cond_br %42#3, %trueResult_4 {handshake.bb = 3 : ui32, handshake.name = "cond_br10"} : <i1>, <>
    sink %falseResult_17 {handshake.name = "sink6"} : <>
    end {handshake.bb = 4 : ui32, handshake.name = "end0"} %falseResult_15, %0#1 : <f32>, <>
  }
}

