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
    %8 = mux %29#2 [%3, %34] {handshake.bb = 2 : ui32, handshake.name = "mux4"} : <i1>, [<i8>, <i8>] to <i8>
    %9 = mux %29#1 [%4, %40] {handshake.bb = 2 : ui32, handshake.name = "mux5"} : <i1>, [<f32>, <f32>] to <f32>
    %10 = mux %29#0 [%result, %37] {handshake.bb = 2 : ui32, handshake.name = "mux6"} : <i1>, [<>, <>] to <>
    %11 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %12 = constant %11 {handshake.bb = 2 : ui32, handshake.name = "constant6", value = 5.000000e-01 : f32} : <>, <f32>
    %13 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %14 = constant %13 {handshake.bb = 2 : ui32, handshake.name = "constant7", value = 1.000000e-01 : f32} : <>, <f32>
    %15 = mulf %39#3, %41#1 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "mulf0"} : <f32>
    %16 = addf %39#2, %15 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "addf0"} : <f32>
    %17 = passer %18#0[%27#1] {handshake.bb = 2 : ui32, handshake.name = "passer7"} : <f32>, <i1>
    %18:2 = fork [2] %19 {handshake.bb = 2 : ui32, handshake.name = "fork4"} : <f32>
    %19 = mulf %16, %12 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "mulf1"} : <f32>
    %20 = subf %18#1, %39#1 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "subf0"} : <f32>
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
    %31:3 = fork [3] %30 {handshake.bb = 2 : ui32, handshake.name = "fork16"} : <i1>
    %32 = passer %39#0[%31#2] {handshake.bb = 2 : ui32, handshake.name = "passer0"} : <f32>, <i1>
    %33 = passer %35#1[%31#1] {handshake.bb = 2 : ui32, handshake.name = "passer1"} : <i8>, <i1>
    %34 = passer %35#0[%27#4] {handshake.bb = 2 : ui32, handshake.name = "passer9"} : <i8>, <i1>
    %35:2 = fork [2] %8 {handshake.bb = 2 : ui32, handshake.name = "fork17"} : <i8>
    %36 = passer %38#0[%31#0] {handshake.bb = 2 : ui32, handshake.name = "passer3"} : <>, <i1>
    %37 = passer %38#1[%27#3] {handshake.bb = 2 : ui32, handshake.name = "passer10"} : <>, <i1>
    %38:2 = fork [2] %10 {handshake.bb = 2 : ui32, handshake.name = "fork18"} : <>
    %39:4 = fork [4] %7 {handshake.bb = 2 : ui32, handshake.name = "fork19"} : <f32>
    %40 = passer %41#0[%27#0] {handshake.bb = 2 : ui32, handshake.name = "passer11"} : <f32>, <i1>
    %41:2 = fork [2] %9 {handshake.bb = 2 : ui32, handshake.name = "fork20"} : <f32>
    %42 = extsi %33 {handshake.bb = 3 : ui32, handshake.name = "extsi4"} : <i8> to <i9>
    %43 = source {handshake.bb = 3 : ui32, handshake.name = "source2"} : <>
    %44 = constant %43 {handshake.bb = 3 : ui32, handshake.name = "constant8", value = 1.000000e+00 : f32} : <>, <f32>
    %45:2 = fork [2] %44 {handshake.bb = 3 : ui32, handshake.name = "fork7"} : <f32>
    %46 = source {handshake.bb = 3 : ui32, handshake.name = "source3"} : <>
    %47 = constant %46 {handshake.bb = 3 : ui32, handshake.name = "constant1", value = 1 : i2} : <>, <i2>
    %48 = extsi %47 {handshake.bb = 3 : ui32, handshake.name = "extsi5"} : <i2> to <i9>
    %49 = source {handshake.bb = 3 : ui32, handshake.name = "source4"} : <>
    %50 = constant %49 {handshake.bb = 3 : ui32, handshake.name = "constant2", value = 100 : i8} : <>, <i8>
    %51 = extsi %50 {handshake.bb = 3 : ui32, handshake.name = "extsi6"} : <i8> to <i9>
    %52 = addf %32, %45#1 {fastmath = #arith.fastmath<none>, handshake.bb = 3 : ui32, handshake.name = "addf1"} : <f32>
    %53:2 = fork [2] %52 {handshake.bb = 3 : ui32, handshake.name = "fork8"} : <f32>
    %54 = divf %45#0, %53#1 {fastmath = #arith.fastmath<none>, handshake.bb = 3 : ui32, handshake.name = "divf0"} : <f32>
    %55 = addi %42, %48 {handshake.bb = 3 : ui32, handshake.name = "addi0"} : <i9>
    %56:2 = fork [2] %55 {handshake.bb = 3 : ui32, handshake.name = "fork9"} : <i9>
    %57 = trunci %56#0 {handshake.bb = 3 : ui32, handshake.name = "trunci0"} : <i9> to <i8>
    %58 = cmpi ult, %56#1, %51 {handshake.bb = 3 : ui32, handshake.name = "cmpi0"} : <i9>
    %59:4 = fork [4] %58 {handshake.bb = 3 : ui32, handshake.name = "fork10"} : <i1>
    %trueResult, %falseResult = cond_br %59#0, %57 {handshake.bb = 3 : ui32, handshake.name = "cond_br7"} : <i1>, <i8>
    sink %falseResult {handshake.name = "sink4"} : <i8>
    %trueResult_0, %falseResult_1 = cond_br %59#1, %54 {handshake.bb = 3 : ui32, handshake.name = "cond_br8"} : <i1>, <f32>
    sink %falseResult_1 {handshake.name = "sink5"} : <f32>
    %trueResult_2, %falseResult_3 = cond_br %59#2, %53#0 {handshake.bb = 3 : ui32, handshake.name = "cond_br9"} : <i1>, <f32>
    %trueResult_4, %falseResult_5 = cond_br %59#3, %36 {handshake.bb = 3 : ui32, handshake.name = "cond_br10"} : <i1>, <>
    sink %falseResult_5 {handshake.name = "sink6"} : <>
    end {handshake.bb = 4 : ui32, handshake.name = "end0"} %falseResult_3, %0#1 : <f32>, <>
  }
}

