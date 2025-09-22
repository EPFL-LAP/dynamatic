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
    %7 = mux %32#0 [%5, %18] {handshake.bb = 2 : ui32, handshake.name = "mux3"} : <i1>, [<f32>, <f32>] to <f32>
    %8 = mux %32#1 [%3, %39] {handshake.bb = 2 : ui32, handshake.name = "mux4"} : <i1>, [<i8>, <i8>] to <i8>
    %9 = mux %32#2 [%4, %44] {handshake.bb = 2 : ui32, handshake.name = "mux5"} : <i1>, [<f32>, <f32>] to <f32>
    %10 = mux %32#3 [%result, %42] {handshake.bb = 2 : ui32, handshake.name = "mux6"} : <i1>, [<>, <>] to <>
    %11 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %12 = constant %11 {handshake.bb = 2 : ui32, handshake.name = "constant6", value = 5.000000e-01 : f32} : <>, <f32>
    %13 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %14 = constant %13 {handshake.bb = 2 : ui32, handshake.name = "constant7", value = 1.000000e-01 : f32} : <>, <f32>
    %15 = mulf %17#3, %45#1 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "mulf0"} : <f32>
    %16 = addf %17#2, %15 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "addf0"} : <f32>
    %17:4 = fork [4] %7 {handshake.bb = 2 : ui32, handshake.name = "fork4"} : <f32>
    %18 = passer %19#0[%30#3] {handshake.bb = 2 : ui32, handshake.name = "passer7"} : <f32>, <i1>
    %19:2 = fork [2] %20 {handshake.bb = 2 : ui32, handshake.name = "fork6"} : <f32>
    %20 = mulf %16, %12 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "mulf1"} : <f32>
    %21 = subf %19#1, %17#1 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "subf0"} : <f32>
    %22 = absf %21 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "absf0"} : <f32> to <f32>
    %23 = cmpf olt, %22, %14 {handshake.bb = 2 : ui32, handshake.name = "cmpf0"} : <f32>
    %24:2 = fork [2] %23 {handshake.bb = 2 : ui32, handshake.name = "fork15"} : <i1>
    %25 = passer %26[%34#1] {handshake.bb = 2 : ui32, handshake.name = "passer8"} : <i1>, <i1>
    %26 = not %24#0 {handshake.bb = 2 : ui32, handshake.name = "not0"} : <i1>
    %27 = spec_v2_repeating_init %25 {handshake.bb = 2 : ui32, handshake.name = "spec_v2_repeating_init0", initToken = 1 : ui1, specv2_top_ri = true} : <i1>
    %28:2 = fork [2] %27 {handshake.bb = 2 : ui32, handshake.name = "fork16"} : <i1>
    %29 = spec_v2_repeating_init %28#0 {handshake.bb = 2 : ui32, handshake.name = "spec_v2_repeating_init1", initToken = 1 : ui1} : <i1>
    %30:6 = fork [6] %29 {handshake.bb = 2 : ui32, handshake.name = "fork2"} : <i1>
    %31 = init %30#5 {handshake.bb = 2 : ui32, handshake.name = "init4", initToken = 0 : ui1} : <i1>
    %32:4 = fork [4] %31 {handshake.bb = 2 : ui32, handshake.name = "fork3"} : <i1>
    %33 = andi %28#1, %30#4 {handshake.bb = 2 : ui32, handshake.name = "andi0", specv2_tmp_and = true} : <i1>
    %34:2 = fork [2] %33 {handshake.bb = 2 : ui32, handshake.name = "fork18"} : <i1>
    %35 = andi %24#1, %34#0 {handshake.bb = 2 : ui32, handshake.name = "andi1"} : <i1>
    %36:3 = fork [3] %35 {handshake.bb = 2 : ui32, handshake.name = "fork19"} : <i1>
    %37 = passer %17#0[%36#2] {handshake.bb = 2 : ui32, handshake.name = "passer0"} : <f32>, <i1>
    %38 = passer %40#1[%36#1] {handshake.bb = 2 : ui32, handshake.name = "passer1"} : <i8>, <i1>
    %39 = passer %40#0[%30#0] {handshake.bb = 2 : ui32, handshake.name = "passer9"} : <i8>, <i1>
    %40:2 = fork [2] %8 {handshake.bb = 2 : ui32, handshake.name = "fork20"} : <i8>
    %41 = passer %43#0[%36#0] {handshake.bb = 2 : ui32, handshake.name = "passer3"} : <>, <i1>
    %42 = passer %43#1[%30#1] {handshake.bb = 2 : ui32, handshake.name = "passer10"} : <>, <i1>
    %43:2 = fork [2] %10 {handshake.bb = 2 : ui32, handshake.name = "fork21"} : <>
    %44 = passer %45#0[%30#2] {handshake.bb = 2 : ui32, handshake.name = "passer11"} : <f32>, <i1>
    %45:2 = fork [2] %9 {handshake.bb = 2 : ui32, handshake.name = "fork22"} : <f32>
    %46 = extsi %38 {handshake.bb = 3 : ui32, handshake.name = "extsi4"} : <i8> to <i9>
    %47 = source {handshake.bb = 3 : ui32, handshake.name = "source2"} : <>
    %48 = constant %47 {handshake.bb = 3 : ui32, handshake.name = "constant8", value = 1.000000e+00 : f32} : <>, <f32>
    %49:2 = fork [2] %48 {handshake.bb = 3 : ui32, handshake.name = "fork7"} : <f32>
    %50 = source {handshake.bb = 3 : ui32, handshake.name = "source3"} : <>
    %51 = constant %50 {handshake.bb = 3 : ui32, handshake.name = "constant1", value = 1 : i2} : <>, <i2>
    %52 = extsi %51 {handshake.bb = 3 : ui32, handshake.name = "extsi5"} : <i2> to <i9>
    %53 = source {handshake.bb = 3 : ui32, handshake.name = "source4"} : <>
    %54 = constant %53 {handshake.bb = 3 : ui32, handshake.name = "constant2", value = 100 : i8} : <>, <i8>
    %55 = extsi %54 {handshake.bb = 3 : ui32, handshake.name = "extsi6"} : <i8> to <i9>
    %56 = addf %37, %49#1 {fastmath = #arith.fastmath<none>, handshake.bb = 3 : ui32, handshake.name = "addf1"} : <f32>
    %57:2 = fork [2] %56 {handshake.bb = 3 : ui32, handshake.name = "fork8"} : <f32>
    %58 = divf %49#0, %57#1 {fastmath = #arith.fastmath<none>, handshake.bb = 3 : ui32, handshake.name = "divf0"} : <f32>
    %59 = addi %46, %52 {handshake.bb = 3 : ui32, handshake.name = "addi0"} : <i9>
    %60:2 = fork [2] %59 {handshake.bb = 3 : ui32, handshake.name = "fork9"} : <i9>
    %61 = trunci %60#0 {handshake.bb = 3 : ui32, handshake.name = "trunci0"} : <i9> to <i8>
    %62 = cmpi ult, %60#1, %55 {handshake.bb = 3 : ui32, handshake.name = "cmpi0"} : <i9>
    %63:4 = fork [4] %62 {handshake.bb = 3 : ui32, handshake.name = "fork10"} : <i1>
    %trueResult, %falseResult = cond_br %63#0, %61 {handshake.bb = 3 : ui32, handshake.name = "cond_br7"} : <i1>, <i8>
    sink %falseResult {handshake.name = "sink4"} : <i8>
    %trueResult_0, %falseResult_1 = cond_br %63#1, %58 {handshake.bb = 3 : ui32, handshake.name = "cond_br8"} : <i1>, <f32>
    sink %falseResult_1 {handshake.name = "sink5"} : <f32>
    %trueResult_2, %falseResult_3 = cond_br %63#2, %57#0 {handshake.bb = 3 : ui32, handshake.name = "cond_br9"} : <i1>, <f32>
    %trueResult_4, %falseResult_5 = cond_br %63#3, %41 {handshake.bb = 3 : ui32, handshake.name = "cond_br10"} : <i1>, <>
    sink %falseResult_5 {handshake.name = "sink6"} : <>
    end {handshake.bb = 4 : ui32, handshake.name = "end0"} %falseResult_3, %0#1 : <f32>, <>
  }
}

