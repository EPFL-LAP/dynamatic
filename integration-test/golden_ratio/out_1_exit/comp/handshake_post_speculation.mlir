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
    %7 = mux %29#3 [%5, %17] {handshake.bb = 2 : ui32, handshake.name = "mux3"} : <i1>, [<f32>, <f32>] to <f32>
    %8 = mux %29#2 [%3, %32] {handshake.bb = 2 : ui32, handshake.name = "mux4"} : <i1>, [<i8>, <i8>] to <i8>
    %9 = mux %29#1 [%4, %38] {handshake.bb = 2 : ui32, handshake.name = "mux5"} : <i1>, [<f32>, <f32>] to <f32>
    %10 = mux %29#0 [%result, %35] {handshake.bb = 2 : ui32, handshake.name = "mux6"} : <i1>, [<>, <>] to <>
    %11 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %12 = constant %11 {handshake.bb = 2 : ui32, handshake.name = "constant6", value = 5.000000e-01 : f32} : <>, <f32>
    %13 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %14 = constant %13 {handshake.bb = 2 : ui32, handshake.name = "constant7", value = 1.000000e-01 : f32} : <>, <f32>
    %15 = mulf %37#3, %39#1 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "mulf0"} : <f32>
    %16 = addf %37#2, %15 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "addf0"} : <f32>
    %17 = passer %18#0[%27#1] {handshake.bb = 2 : ui32, handshake.name = "passer7"} : <f32>, <i1>
    %18:2 = fork [2] %19 {handshake.bb = 2 : ui32, handshake.name = "fork4"} : <f32>
    %19 = mulf %16, %12 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "mulf1"} : <f32>
    %20 = subf %18#1, %37#1 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "subf0"} : <f32>
    %21 = absf %20 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "absf0"} : <f32> to <f32>
    %22 = cmpf olt, %21, %14 {handshake.bb = 2 : ui32, handshake.name = "cmpf0"} : <f32>
    %23:2 = fork [2] %22 {handshake.bb = 2 : ui32, handshake.name = "fork6"} : <i1>
    %24 = passer %25[%27#2] {handshake.bb = 2 : ui32, handshake.name = "passer8"} : <i1>, <i1>
    %25 = not %23#0 {handshake.bb = 2 : ui32, handshake.name = "not0"} : <i1>
    %26 = spec_v2_repeating_init %24 {handshake.bb = 2 : ui32, handshake.name = "spec_v2_repeating_init0", initToken = 1 : ui1, specv2_top_ri = true} : <i1>
    %27:7 = fork [7] %26 {handshake.bb = 2 : ui32, handshake.name = "fork2"} : <i1>
    %28 = init %27#6 {handshake.bb = 2 : ui32, handshake.name = "init4", initToken = 0 : ui1} : <i1>
    %29:4 = fork [4] %28 {handshake.bb = 2 : ui32, handshake.name = "fork3"} : <i1>
    %30 = andi %23#1, %27#5 {handshake.bb = 2 : ui32, handshake.name = "andi0"} : <i1>
    %31:8 = fork [8] %30 {handshake.bb = 2 : ui32, handshake.name = "fork33"} : <i1>
    %32 = passer %33#0[%27#4] {handshake.bb = 2 : ui32, handshake.name = "passer9"} : <i8>, <i1>
    %33:2 = fork [2] %8 {handshake.bb = 2 : ui32, handshake.name = "fork26"} : <i8>
    %34 = passer %36#0[%31#0] {handshake.bb = 2 : ui32, handshake.name = "passer3"} : <>, <i1>
    %35 = passer %36#1[%27#3] {handshake.bb = 2 : ui32, handshake.name = "passer10"} : <>, <i1>
    %36:2 = fork [2] %10 {handshake.bb = 2 : ui32, handshake.name = "fork27"} : <>
    %37:4 = fork [4] %7 {handshake.bb = 2 : ui32, handshake.name = "fork28"} : <f32>
    %38 = passer %39#0[%27#0] {handshake.bb = 2 : ui32, handshake.name = "passer11"} : <f32>, <i1>
    %39:2 = fork [2] %9 {handshake.bb = 2 : ui32, handshake.name = "fork29"} : <f32>
    %40 = extsi %33#1 {handshake.bb = 3 : ui32, handshake.name = "extsi4"} : <i8> to <i9>
    %41 = source {handshake.bb = 3 : ui32, handshake.name = "source2"} : <>
    %42 = constant %41 {handshake.bb = 3 : ui32, handshake.name = "constant8", value = 1.000000e+00 : f32} : <>, <f32>
    %43:2 = fork [2] %42 {handshake.bb = 3 : ui32, handshake.name = "fork7"} : <f32>
    %44 = source {handshake.bb = 3 : ui32, handshake.name = "source3"} : <>
    %45 = constant %44 {handshake.bb = 3 : ui32, handshake.name = "constant1", value = 1 : i2} : <>, <i2>
    %46 = extsi %45 {handshake.bb = 3 : ui32, handshake.name = "extsi5"} : <i2> to <i9>
    %47 = source {handshake.bb = 3 : ui32, handshake.name = "source4"} : <>
    %48 = constant %47 {handshake.bb = 3 : ui32, handshake.name = "constant2", value = 100 : i8} : <>, <i8>
    %49 = extsi %48 {handshake.bb = 3 : ui32, handshake.name = "extsi6"} : <i8> to <i9>
    %50 = passer %51#0[%31#7] {handshake.bb = 3 : ui32, handshake.name = "passer12"} : <f32>, <i1>
    %51:2 = fork [2] %52 {handshake.bb = 3 : ui32, handshake.name = "fork30"} : <f32>
    %52 = addf %37#0, %43#1 {fastmath = #arith.fastmath<none>, handshake.bb = 3 : ui32, handshake.name = "addf1"} : <f32>
    %53 = passer %54[%31#6] {handshake.bb = 3 : ui32, handshake.name = "passer13"} : <f32>, <i1>
    %54 = divf %43#0, %51#1 {fastmath = #arith.fastmath<none>, handshake.bb = 3 : ui32, handshake.name = "divf0"} : <f32>
    %55:2 = fork [2] %56 {handshake.bb = 3 : ui32, handshake.name = "fork31"} : <i9>
    %56 = addi %40, %46 {handshake.bb = 3 : ui32, handshake.name = "addi0"} : <i9>
    %57 = passer %58[%31#5] {handshake.bb = 3 : ui32, handshake.name = "passer14"} : <i8>, <i1>
    %58 = trunci %55#0 {handshake.bb = 3 : ui32, handshake.name = "trunci0"} : <i9> to <i8>
    %59 = passer %63#0[%31#4] {handshake.bb = 3 : ui32, handshake.name = "passer15"} : <i1>, <i1>
    %60 = passer %63#1[%31#3] {handshake.bb = 3 : ui32, handshake.name = "passer16"} : <i1>, <i1>
    %61 = passer %63#2[%31#2] {handshake.bb = 3 : ui32, handshake.name = "passer17"} : <i1>, <i1>
    %62 = passer %63#3[%31#1] {handshake.bb = 3 : ui32, handshake.name = "passer18"} : <i1>, <i1>
    %63:4 = fork [4] %64 {handshake.bb = 3 : ui32, handshake.name = "fork32"} : <i1>
    %64 = cmpi ult, %55#1, %49 {handshake.bb = 3 : ui32, handshake.name = "cmpi0"} : <i9>
    %trueResult, %falseResult = cond_br %59, %57 {handshake.bb = 3 : ui32, handshake.name = "cond_br7"} : <i1>, <i8>
    sink %falseResult {handshake.name = "sink4"} : <i8>
    %trueResult_0, %falseResult_1 = cond_br %60, %53 {handshake.bb = 3 : ui32, handshake.name = "cond_br8"} : <i1>, <f32>
    sink %falseResult_1 {handshake.name = "sink5"} : <f32>
    %trueResult_2, %falseResult_3 = cond_br %61, %50 {handshake.bb = 3 : ui32, handshake.name = "cond_br9"} : <i1>, <f32>
    %trueResult_4, %falseResult_5 = cond_br %62, %34 {handshake.bb = 3 : ui32, handshake.name = "cond_br10"} : <i1>, <>
    sink %falseResult_5 {handshake.name = "sink6"} : <>
    end {handshake.bb = 4 : ui32, handshake.name = "end0"} %falseResult_3, %0#1 : <f32>, <>
  }
}

