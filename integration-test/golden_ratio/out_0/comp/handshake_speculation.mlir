module {
  handshake.func @golden_ratio(%arg0: !handshake.channel<f32>, %arg1: !handshake.channel<f32>, %arg2: !handshake.control<>, ...) -> (!handshake.channel<f32>, !handshake.control<>) attributes {argNames = ["x0", "x1", "start"], resNames = ["out0", "end"]} {
    %0:3 = fork [3] %arg2 {handshake.bb = 0 : ui32, handshake.name = "fork0"} : <>
    %1 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant0", value = false} : <>, <i1>
    %2 = extsi %1 {handshake.bb = 0 : ui32, handshake.name = "extsi3"} : <i1> to <i8>
    %3 = mux %6#0 [%2, %trueResult] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i8>, <i8>] to <i8>
    %4 = mux %6#1 [%arg1, %trueResult_0] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i1>, [<f32>, <f32>] to <f32>
    %5 = mux %6#2 [%arg0, %trueResult_2] {handshake.bb = 1 : ui32, handshake.name = "mux2"} : <i1>, [<f32>, <f32>] to <f32>
    %result, %index = control_merge [%0#2, %trueResult_4]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %6:3 = fork [3] %index {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <i1>
    %7 = init %32#3 {handshake.bb = 2 : ui32, handshake.name = "init0", initToken = 0 : ui1} : <i1>
    %8 = mux %7 [%5, %38] {handshake.bb = 2 : ui32, handshake.name = "mux3"} : <i1>, [<f32>, <f32>] to <f32>
    %9:4 = fork [4] %8 {handshake.bb = 2 : ui32, handshake.name = "fork2"} : <f32>
    %10 = init %32#2 {handshake.bb = 2 : ui32, handshake.name = "init1", initToken = 0 : ui1} : <i1>
    %11 = mux %10 [%3, %35] {handshake.bb = 2 : ui32, handshake.name = "mux4"} : <i1>, [<i8>, <i8>] to <i8>
    %12:2 = fork [2] %11 {handshake.bb = 2 : ui32, handshake.name = "fork11"} : <i8>
    %13 = init %32#1 {handshake.bb = 2 : ui32, handshake.name = "init2", initToken = 0 : ui1} : <i1>
    %14 = init %32#0 {handshake.bb = 2 : ui32, handshake.name = "init3", initToken = 0 : ui1} : <i1>
    %15 = mux %13 [%4, %39] {handshake.bb = 2 : ui32, handshake.name = "mux5"} : <i1>, [<f32>, <f32>] to <f32>
    %16:2 = fork [2] %15 {handshake.bb = 2 : ui32, handshake.name = "fork3"} : <f32>
    %17 = mux %14 [%result, %37] {handshake.bb = 2 : ui32, handshake.name = "mux6"} : <i1>, [<>, <>] to <>
    %18:2 = fork [2] %17 {handshake.bb = 2 : ui32, handshake.name = "fork12"} : <>
    %19 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %20 = constant %19 {handshake.bb = 2 : ui32, handshake.name = "constant6", value = 5.000000e-01 : f32} : <>, <f32>
    %21 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %22 = constant %21 {handshake.bb = 2 : ui32, handshake.name = "constant7", value = 1.000000e-01 : f32} : <>, <f32>
    %23 = mulf %9#3, %16#1 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "mulf0"} : <f32>
    %24 = addf %9#2, %23 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "addf0"} : <f32>
    %25 = mulf %24, %20 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "mulf1"} : <f32>
    %26:2 = fork [2] %25 {handshake.bb = 2 : ui32, handshake.name = "fork5"} : <f32>
    %27 = subf %26#1, %9#1 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "subf0"} : <f32>
    %28 = absf %27 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "absf0"} : <f32> to <f32>
    %29 = cmpf olt, %28, %22 {handshake.bb = 2 : ui32, handshake.name = "cmpf0"} : <f32>
    %30:4 = fork [4] %29 {handshake.bb = 2 : ui32, handshake.name = "fork13"} : <i1>
    %31 = not %30#3 {handshake.bb = 2 : ui32, handshake.name = "not0"} : <i1>
    %32:8 = fork [8] %31 {handshake.bb = 2 : ui32, handshake.name = "fork14"} : <i1>
    %33 = passer %9#0[%30#2] {handshake.bb = 2 : ui32, handshake.name = "passer0"} : <f32>, <i1>
    %34 = passer %12#1[%30#1] {handshake.bb = 2 : ui32, handshake.name = "passer1"} : <i8>, <i1>
    %35 = passer %12#0[%32#7] {handshake.bb = 2 : ui32, handshake.name = "passer2"} : <i8>, <i1>
    %36 = passer %18#0[%30#0] {handshake.bb = 2 : ui32, handshake.name = "passer3"} : <>, <i1>
    %37 = passer %18#1[%32#6] {handshake.bb = 2 : ui32, handshake.name = "passer4"} : <>, <i1>
    %38 = passer %26#0[%32#5] {handshake.bb = 2 : ui32, handshake.name = "passer5"} : <f32>, <i1>
    %39 = passer %16#0[%32#4] {handshake.bb = 2 : ui32, handshake.name = "passer6"} : <f32>, <i1>
    %40 = extsi %34 {handshake.bb = 3 : ui32, handshake.name = "extsi4"} : <i8> to <i9>
    %41 = source {handshake.bb = 3 : ui32, handshake.name = "source2"} : <>
    %42 = constant %41 {handshake.bb = 3 : ui32, handshake.name = "constant8", value = 1.000000e+00 : f32} : <>, <f32>
    %43:2 = fork [2] %42 {handshake.bb = 3 : ui32, handshake.name = "fork7"} : <f32>
    %44 = source {handshake.bb = 3 : ui32, handshake.name = "source3"} : <>
    %45 = constant %44 {handshake.bb = 3 : ui32, handshake.name = "constant1", value = 1 : i2} : <>, <i2>
    %46 = extsi %45 {handshake.bb = 3 : ui32, handshake.name = "extsi5"} : <i2> to <i9>
    %47 = source {handshake.bb = 3 : ui32, handshake.name = "source4"} : <>
    %48 = constant %47 {handshake.bb = 3 : ui32, handshake.name = "constant2", value = 100 : i8} : <>, <i8>
    %49 = extsi %48 {handshake.bb = 3 : ui32, handshake.name = "extsi6"} : <i8> to <i9>
    %50 = addf %33, %43#1 {fastmath = #arith.fastmath<none>, handshake.bb = 3 : ui32, handshake.name = "addf1"} : <f32>
    %51:2 = fork [2] %50 {handshake.bb = 3 : ui32, handshake.name = "fork8"} : <f32>
    %52 = divf %43#0, %51#1 {fastmath = #arith.fastmath<none>, handshake.bb = 3 : ui32, handshake.name = "divf0"} : <f32>
    %53 = addi %40, %46 {handshake.bb = 3 : ui32, handshake.name = "addi0"} : <i9>
    %54:2 = fork [2] %53 {handshake.bb = 3 : ui32, handshake.name = "fork9"} : <i9>
    %55 = trunci %54#0 {handshake.bb = 3 : ui32, handshake.name = "trunci0"} : <i9> to <i8>
    %56 = cmpi ult, %54#1, %49 {handshake.bb = 3 : ui32, handshake.name = "cmpi0"} : <i9>
    %57:4 = fork [4] %56 {handshake.bb = 3 : ui32, handshake.name = "fork10"} : <i1>
    %trueResult, %falseResult = cond_br %57#0, %55 {handshake.bb = 3 : ui32, handshake.name = "cond_br7"} : <i1>, <i8>
    sink %falseResult {handshake.name = "sink4"} : <i8>
    %trueResult_0, %falseResult_1 = cond_br %57#1, %52 {handshake.bb = 3 : ui32, handshake.name = "cond_br8"} : <i1>, <f32>
    sink %falseResult_1 {handshake.name = "sink5"} : <f32>
    %trueResult_2, %falseResult_3 = cond_br %57#2, %51#0 {handshake.bb = 3 : ui32, handshake.name = "cond_br9"} : <i1>, <f32>
    %trueResult_4, %falseResult_5 = cond_br %57#3, %36 {handshake.bb = 3 : ui32, handshake.name = "cond_br10"} : <i1>, <>
    sink %falseResult_5 {handshake.name = "sink6"} : <>
    end {handshake.bb = 4 : ui32, handshake.name = "end0"} %falseResult_3, %0#1 : <f32>, <>
  }
}

