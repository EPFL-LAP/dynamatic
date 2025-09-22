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
    %7 = mux %30#0 [%5, %36] {handshake.bb = 2 : ui32, handshake.name = "mux3"} : <i1>, [<f32>, <f32>] to <f32>
    %8:4 = fork [4] %7 {handshake.bb = 2 : ui32, handshake.name = "fork2"} : <f32>
    %9 = mux %30#1 [%3, %33] {handshake.bb = 2 : ui32, handshake.name = "mux4"} : <i1>, [<i8>, <i8>] to <i8>
    %10:2 = fork [2] %9 {handshake.bb = 2 : ui32, handshake.name = "fork11"} : <i8>
    %11 = mux %30#2 [%4, %37] {handshake.bb = 2 : ui32, handshake.name = "mux5"} : <i1>, [<f32>, <f32>] to <f32>
    %12:2 = fork [2] %11 {handshake.bb = 2 : ui32, handshake.name = "fork3"} : <f32>
    %13 = mux %30#3 [%result, %35] {handshake.bb = 2 : ui32, handshake.name = "mux6"} : <i1>, [<>, <>] to <>
    %14:2 = fork [2] %13 {handshake.bb = 2 : ui32, handshake.name = "fork12"} : <>
    %15 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %16 = constant %15 {handshake.bb = 2 : ui32, handshake.name = "constant6", value = 5.000000e-01 : f32} : <>, <f32>
    %17 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %18 = constant %17 {handshake.bb = 2 : ui32, handshake.name = "constant7", value = 1.000000e-01 : f32} : <>, <f32>
    %19 = mulf %8#3, %12#1 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "mulf0"} : <f32>
    %20 = addf %8#2, %19 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "addf0"} : <f32>
    %21 = mulf %20, %16 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "mulf1"} : <f32>
    %22:2 = fork [2] %21 {handshake.bb = 2 : ui32, handshake.name = "fork5"} : <f32>
    %23 = subf %22#1, %8#1 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "subf0"} : <f32>
    %24 = absf %23 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "absf0"} : <f32> to <f32>
    %25 = cmpf olt, %24, %18 {handshake.bb = 2 : ui32, handshake.name = "cmpf0"} : <f32>
    %26:4 = fork [4] %25 {handshake.bb = 2 : ui32, handshake.name = "fork13"} : <i1>
    %27 = not %26#3 {handshake.bb = 2 : ui32, handshake.name = "not0"} : <i1>
    %28:5 = fork [5] %27 {handshake.bb = 2 : ui32, handshake.name = "fork4"} : <i1>
    %29 = init %28#4 {handshake.bb = 2 : ui32, handshake.name = "init4", initToken = 0 : ui1} : <i1>
    %30:4 = fork [4] %29 {handshake.bb = 2 : ui32, handshake.name = "fork6"} : <i1>
    %31 = passer %8#0[%26#2] {handshake.bb = 2 : ui32, handshake.name = "passer0"} : <f32>, <i1>
    %32 = passer %10#1[%26#1] {handshake.bb = 2 : ui32, handshake.name = "passer1"} : <i8>, <i1>
    %33 = passer %10#0[%28#0] {handshake.bb = 2 : ui32, handshake.name = "passer2"} : <i8>, <i1>
    %34 = passer %14#0[%26#0] {handshake.bb = 2 : ui32, handshake.name = "passer3"} : <>, <i1>
    %35 = passer %14#1[%28#1] {handshake.bb = 2 : ui32, handshake.name = "passer4"} : <>, <i1>
    %36 = passer %22#0[%28#2] {handshake.bb = 2 : ui32, handshake.name = "passer5"} : <f32>, <i1>
    %37 = passer %12#0[%28#3] {handshake.bb = 2 : ui32, handshake.name = "passer6"} : <f32>, <i1>
    %38 = extsi %32 {handshake.bb = 3 : ui32, handshake.name = "extsi4"} : <i8> to <i9>
    %39 = source {handshake.bb = 3 : ui32, handshake.name = "source2"} : <>
    %40 = constant %39 {handshake.bb = 3 : ui32, handshake.name = "constant8", value = 1.000000e+00 : f32} : <>, <f32>
    %41:2 = fork [2] %40 {handshake.bb = 3 : ui32, handshake.name = "fork7"} : <f32>
    %42 = source {handshake.bb = 3 : ui32, handshake.name = "source3"} : <>
    %43 = constant %42 {handshake.bb = 3 : ui32, handshake.name = "constant1", value = 1 : i2} : <>, <i2>
    %44 = extsi %43 {handshake.bb = 3 : ui32, handshake.name = "extsi5"} : <i2> to <i9>
    %45 = source {handshake.bb = 3 : ui32, handshake.name = "source4"} : <>
    %46 = constant %45 {handshake.bb = 3 : ui32, handshake.name = "constant2", value = 100 : i8} : <>, <i8>
    %47 = extsi %46 {handshake.bb = 3 : ui32, handshake.name = "extsi6"} : <i8> to <i9>
    %48 = addf %31, %41#1 {fastmath = #arith.fastmath<none>, handshake.bb = 3 : ui32, handshake.name = "addf1"} : <f32>
    %49:2 = fork [2] %48 {handshake.bb = 3 : ui32, handshake.name = "fork8"} : <f32>
    %50 = divf %41#0, %49#1 {fastmath = #arith.fastmath<none>, handshake.bb = 3 : ui32, handshake.name = "divf0"} : <f32>
    %51 = addi %38, %44 {handshake.bb = 3 : ui32, handshake.name = "addi0"} : <i9>
    %52:2 = fork [2] %51 {handshake.bb = 3 : ui32, handshake.name = "fork9"} : <i9>
    %53 = trunci %52#0 {handshake.bb = 3 : ui32, handshake.name = "trunci0"} : <i9> to <i8>
    %54 = cmpi ult, %52#1, %47 {handshake.bb = 3 : ui32, handshake.name = "cmpi0"} : <i9>
    %55:4 = fork [4] %54 {handshake.bb = 3 : ui32, handshake.name = "fork10"} : <i1>
    %trueResult, %falseResult = cond_br %55#0, %53 {handshake.bb = 3 : ui32, handshake.name = "cond_br7"} : <i1>, <i8>
    sink %falseResult {handshake.name = "sink4"} : <i8>
    %trueResult_0, %falseResult_1 = cond_br %55#1, %50 {handshake.bb = 3 : ui32, handshake.name = "cond_br8"} : <i1>, <f32>
    sink %falseResult_1 {handshake.name = "sink5"} : <f32>
    %trueResult_2, %falseResult_3 = cond_br %55#2, %49#0 {handshake.bb = 3 : ui32, handshake.name = "cond_br9"} : <i1>, <f32>
    %trueResult_4, %falseResult_5 = cond_br %55#3, %34 {handshake.bb = 3 : ui32, handshake.name = "cond_br10"} : <i1>, <>
    sink %falseResult_5 {handshake.name = "sink6"} : <>
    end {handshake.bb = 4 : ui32, handshake.name = "end0"} %falseResult_3, %0#1 : <f32>, <>
  }
}

