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
    %7 = init %31#6 {handshake.bb = 2 : ui32, handshake.name = "init0", initToken = 0 : ui1} : <i1>
    %8 = mux %7 [%5, %21] {handshake.bb = 2 : ui32, handshake.name = "mux3"} : <i1>, [<f32>, <f32>] to <f32>
    %9 = init %31#7 {handshake.bb = 2 : ui32, handshake.name = "init1", initToken = 0 : ui1} : <i1>
    %10 = mux %9 [%3, %34] {handshake.bb = 2 : ui32, handshake.name = "mux4"} : <i1>, [<i8>, <i8>] to <i8>
    %11 = init %31#8 {handshake.bb = 2 : ui32, handshake.name = "init2", initToken = 0 : ui1} : <i1>
    %12 = init %31#9 {handshake.bb = 2 : ui32, handshake.name = "init3", initToken = 0 : ui1} : <i1>
    %13 = mux %11 [%4, %40] {handshake.bb = 2 : ui32, handshake.name = "mux5"} : <i1>, [<f32>, <f32>] to <f32>
    %14 = mux %12 [%result, %37] {handshake.bb = 2 : ui32, handshake.name = "mux6"} : <i1>, [<>, <>] to <>
    %15 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %16 = constant %15 {handshake.bb = 2 : ui32, handshake.name = "constant6", value = 5.000000e-01 : f32} : <>, <f32>
    %17 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %18 = constant %17 {handshake.bb = 2 : ui32, handshake.name = "constant7", value = 1.000000e-01 : f32} : <>, <f32>
    %19 = mulf %39#3, %41#1 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "mulf0"} : <f32>
    %20 = addf %39#2, %19 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "addf0"} : <f32>
    %21 = passer %22#0[%31#4] {handshake.bb = 2 : ui32, handshake.name = "passer7"} : <f32>, <i1>
    %22:2 = fork [2] %23 {handshake.bb = 2 : ui32, handshake.name = "fork4"} : <f32>
    %23 = mulf %20, %16 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "mulf1"} : <f32>
    %24 = subf %22#1, %39#1 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "subf0"} : <f32>
    %25 = absf %24 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "absf0"} : <f32> to <f32>
    %26 = cmpf olt, %25, %18 {handshake.bb = 2 : ui32, handshake.name = "cmpf0"} : <f32>
    %27:2 = fork [2] %26 {handshake.bb = 2 : ui32, handshake.name = "fork6"} : <i1>
    %28 = passer %29[%31#3] {handshake.bb = 2 : ui32, handshake.name = "passer8"} : <i1>, <i1>
    %29 = not %27#0 {handshake.bb = 2 : ui32, handshake.name = "not0"} : <i1>
    %30 = spec_v2_repeating_init %28 {handshake.bb = 2 : ui32, handshake.name = "spec_v2_repeating_init0", initToken = 1 : ui1, specv2_top_ri = true} : <i1>
    %31:10 = fork [10] %30 {handshake.bb = 2 : ui32, handshake.name = "fork15"} : <i1>
    %32 = andi %27#1, %31#0 {handshake.bb = 2 : ui32, handshake.name = "andi0"} : <i1>
    %33:8 = fork [8] %32 {handshake.bb = 2 : ui32, handshake.name = "fork33"} : <i1>
    %34 = passer %35#0[%31#1] {handshake.bb = 2 : ui32, handshake.name = "passer9"} : <i8>, <i1>
    %35:2 = fork [2] %10 {handshake.bb = 2 : ui32, handshake.name = "fork26"} : <i8>
    %36 = passer %38#0[%33#0] {handshake.bb = 2 : ui32, handshake.name = "passer3"} : <>, <i1>
    %37 = passer %38#1[%31#2] {handshake.bb = 2 : ui32, handshake.name = "passer10"} : <>, <i1>
    %38:2 = fork [2] %14 {handshake.bb = 2 : ui32, handshake.name = "fork27"} : <>
    %39:4 = fork [4] %8 {handshake.bb = 2 : ui32, handshake.name = "fork28"} : <f32>
    %40 = passer %41#0[%31#5] {handshake.bb = 2 : ui32, handshake.name = "passer11"} : <f32>, <i1>
    %41:2 = fork [2] %13 {handshake.bb = 2 : ui32, handshake.name = "fork29"} : <f32>
    %42 = extsi %35#1 {handshake.bb = 3 : ui32, handshake.name = "extsi4"} : <i8> to <i9>
    %43 = source {handshake.bb = 3 : ui32, handshake.name = "source2"} : <>
    %44 = constant %43 {handshake.bb = 3 : ui32, handshake.name = "constant8", value = 1.000000e+00 : f32} : <>, <f32>
    %45:2 = fork [2] %44 {handshake.bb = 3 : ui32, handshake.name = "fork7"} : <f32>
    %46 = source {handshake.bb = 3 : ui32, handshake.name = "source3"} : <>
    %47 = constant %46 {handshake.bb = 3 : ui32, handshake.name = "constant1", value = 1 : i2} : <>, <i2>
    %48 = extsi %47 {handshake.bb = 3 : ui32, handshake.name = "extsi5"} : <i2> to <i9>
    %49 = source {handshake.bb = 3 : ui32, handshake.name = "source4"} : <>
    %50 = constant %49 {handshake.bb = 3 : ui32, handshake.name = "constant2", value = 100 : i8} : <>, <i8>
    %51 = extsi %50 {handshake.bb = 3 : ui32, handshake.name = "extsi6"} : <i8> to <i9>
    %52 = passer %53#0[%33#7] {handshake.bb = 3 : ui32, handshake.name = "passer12"} : <f32>, <i1>
    %53:2 = fork [2] %54 {handshake.bb = 3 : ui32, handshake.name = "fork30"} : <f32>
    %54 = addf %39#0, %45#1 {fastmath = #arith.fastmath<none>, handshake.bb = 3 : ui32, handshake.name = "addf1"} : <f32>
    %55 = passer %56[%33#6] {handshake.bb = 3 : ui32, handshake.name = "passer13"} : <f32>, <i1>
    %56 = divf %45#0, %53#1 {fastmath = #arith.fastmath<none>, handshake.bb = 3 : ui32, handshake.name = "divf0"} : <f32>
    %57:2 = fork [2] %58 {handshake.bb = 3 : ui32, handshake.name = "fork31"} : <i9>
    %58 = addi %42, %48 {handshake.bb = 3 : ui32, handshake.name = "addi0"} : <i9>
    %59 = passer %60[%33#5] {handshake.bb = 3 : ui32, handshake.name = "passer14"} : <i8>, <i1>
    %60 = trunci %57#0 {handshake.bb = 3 : ui32, handshake.name = "trunci0"} : <i9> to <i8>
    %61 = passer %65#0[%33#4] {handshake.bb = 3 : ui32, handshake.name = "passer15"} : <i1>, <i1>
    %62 = passer %65#1[%33#3] {handshake.bb = 3 : ui32, handshake.name = "passer16"} : <i1>, <i1>
    %63 = passer %65#2[%33#2] {handshake.bb = 3 : ui32, handshake.name = "passer17"} : <i1>, <i1>
    %64 = passer %65#3[%33#1] {handshake.bb = 3 : ui32, handshake.name = "passer18"} : <i1>, <i1>
    %65:4 = fork [4] %66 {handshake.bb = 3 : ui32, handshake.name = "fork32"} : <i1>
    %66 = cmpi ult, %57#1, %51 {handshake.bb = 3 : ui32, handshake.name = "cmpi0"} : <i9>
    %trueResult, %falseResult = cond_br %61, %59 {handshake.bb = 3 : ui32, handshake.name = "cond_br7"} : <i1>, <i8>
    sink %falseResult {handshake.name = "sink4"} : <i8>
    %trueResult_0, %falseResult_1 = cond_br %62, %55 {handshake.bb = 3 : ui32, handshake.name = "cond_br8"} : <i1>, <f32>
    sink %falseResult_1 {handshake.name = "sink5"} : <f32>
    %trueResult_2, %falseResult_3 = cond_br %63, %52 {handshake.bb = 3 : ui32, handshake.name = "cond_br9"} : <i1>, <f32>
    %trueResult_4, %falseResult_5 = cond_br %64, %36 {handshake.bb = 3 : ui32, handshake.name = "cond_br10"} : <i1>, <>
    sink %falseResult_5 {handshake.name = "sink6"} : <>
    end {handshake.bb = 4 : ui32, handshake.name = "end0"} %falseResult_3, %0#1 : <f32>, <>
  }
}

