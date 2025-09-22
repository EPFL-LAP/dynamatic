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
    %7 = init %34#4 {handshake.bb = 2 : ui32, handshake.name = "init0", initToken = 0 : ui1} : <i1>
    %8 = mux %7 [%5, %22] {handshake.bb = 2 : ui32, handshake.name = "mux3"} : <i1>, [<f32>, <f32>] to <f32>
    %9 = init %34#3 {handshake.bb = 2 : ui32, handshake.name = "init1", initToken = 0 : ui1} : <i1>
    %10 = mux %9 [%3, %41] {handshake.bb = 2 : ui32, handshake.name = "mux4"} : <i1>, [<i8>, <i8>] to <i8>
    %11 = init %34#2 {handshake.bb = 2 : ui32, handshake.name = "init2", initToken = 0 : ui1} : <i1>
    %12 = init %34#1 {handshake.bb = 2 : ui32, handshake.name = "init3", initToken = 0 : ui1} : <i1>
    %13 = mux %11 [%4, %46] {handshake.bb = 2 : ui32, handshake.name = "mux5"} : <i1>, [<f32>, <f32>] to <f32>
    %14 = mux %12 [%result, %44] {handshake.bb = 2 : ui32, handshake.name = "mux6"} : <i1>, [<>, <>] to <>
    %15 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %16 = constant %15 {handshake.bb = 2 : ui32, handshake.name = "constant6", value = 5.000000e-01 : f32} : <>, <f32>
    %17 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %18 = constant %17 {handshake.bb = 2 : ui32, handshake.name = "constant7", value = 1.000000e-01 : f32} : <>, <f32>
    %19 = mulf %21#3, %47#1 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "mulf0"} : <f32>
    %20 = addf %21#2, %19 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "addf0"} : <f32>
    %21:4 = fork [4] %8 {handshake.bb = 2 : ui32, handshake.name = "fork4"} : <f32>
    %22 = passer %23#0[%34#5] {handshake.bb = 2 : ui32, handshake.name = "passer7"} : <f32>, <i1>
    %23:2 = fork [2] %24 {handshake.bb = 2 : ui32, handshake.name = "fork6"} : <f32>
    %24 = mulf %20, %16 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "mulf1"} : <f32>
    %25 = subf %23#1, %21#1 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "subf0"} : <f32>
    %26 = absf %25 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "absf0"} : <f32> to <f32>
    %27 = cmpf olt, %26, %18 {handshake.bb = 2 : ui32, handshake.name = "cmpf0"} : <f32>
    %28:2 = fork [2] %27 {handshake.bb = 2 : ui32, handshake.name = "fork15"} : <i1>
    %29 = passer %30[%36#1] {handshake.bb = 2 : ui32, handshake.name = "passer8"} : <i1>, <i1>
    %30 = not %28#0 {handshake.bb = 2 : ui32, handshake.name = "not0"} : <i1>
    %31 = spec_v2_repeating_init %29 {handshake.bb = 2 : ui32, handshake.name = "spec_v2_repeating_init0", initToken = 1 : ui1, specv2_top_ri = true} : <i1>
    %32:2 = fork [2] %31 {handshake.bb = 2 : ui32, handshake.name = "fork16"} : <i1>
    %33 = spec_v2_repeating_init %32#0 {handshake.bb = 2 : ui32, handshake.name = "spec_v2_repeating_init1", initToken = 1 : ui1} : <i1>
    %34:9 = fork [9] %33 {handshake.bb = 2 : ui32, handshake.name = "fork17"} : <i1>
    %35 = andi %32#1, %34#0 {handshake.bb = 2 : ui32, handshake.name = "andi0", specv2_tmp_and = true} : <i1>
    %36:2 = fork [2] %35 {handshake.bb = 2 : ui32, handshake.name = "fork18"} : <i1>
    %37 = andi %28#1, %36#0 {handshake.bb = 2 : ui32, handshake.name = "andi1"} : <i1>
    %38:3 = fork [3] %37 {handshake.bb = 2 : ui32, handshake.name = "fork19"} : <i1>
    %39 = passer %21#0[%38#2] {handshake.bb = 2 : ui32, handshake.name = "passer0"} : <f32>, <i1>
    %40 = passer %42#1[%38#1] {handshake.bb = 2 : ui32, handshake.name = "passer1"} : <i8>, <i1>
    %41 = passer %42#0[%34#8] {handshake.bb = 2 : ui32, handshake.name = "passer9"} : <i8>, <i1>
    %42:2 = fork [2] %10 {handshake.bb = 2 : ui32, handshake.name = "fork20"} : <i8>
    %43 = passer %45#0[%38#0] {handshake.bb = 2 : ui32, handshake.name = "passer3"} : <>, <i1>
    %44 = passer %45#1[%34#7] {handshake.bb = 2 : ui32, handshake.name = "passer10"} : <>, <i1>
    %45:2 = fork [2] %14 {handshake.bb = 2 : ui32, handshake.name = "fork21"} : <>
    %46 = passer %47#0[%34#6] {handshake.bb = 2 : ui32, handshake.name = "passer11"} : <f32>, <i1>
    %47:2 = fork [2] %13 {handshake.bb = 2 : ui32, handshake.name = "fork22"} : <f32>
    %48 = extsi %40 {handshake.bb = 3 : ui32, handshake.name = "extsi4"} : <i8> to <i9>
    %49 = source {handshake.bb = 3 : ui32, handshake.name = "source2"} : <>
    %50 = constant %49 {handshake.bb = 3 : ui32, handshake.name = "constant8", value = 1.000000e+00 : f32} : <>, <f32>
    %51:2 = fork [2] %50 {handshake.bb = 3 : ui32, handshake.name = "fork7"} : <f32>
    %52 = source {handshake.bb = 3 : ui32, handshake.name = "source3"} : <>
    %53 = constant %52 {handshake.bb = 3 : ui32, handshake.name = "constant1", value = 1 : i2} : <>, <i2>
    %54 = extsi %53 {handshake.bb = 3 : ui32, handshake.name = "extsi5"} : <i2> to <i9>
    %55 = source {handshake.bb = 3 : ui32, handshake.name = "source4"} : <>
    %56 = constant %55 {handshake.bb = 3 : ui32, handshake.name = "constant2", value = 100 : i8} : <>, <i8>
    %57 = extsi %56 {handshake.bb = 3 : ui32, handshake.name = "extsi6"} : <i8> to <i9>
    %58 = addf %39, %51#1 {fastmath = #arith.fastmath<none>, handshake.bb = 3 : ui32, handshake.name = "addf1"} : <f32>
    %59:2 = fork [2] %58 {handshake.bb = 3 : ui32, handshake.name = "fork8"} : <f32>
    %60 = divf %51#0, %59#1 {fastmath = #arith.fastmath<none>, handshake.bb = 3 : ui32, handshake.name = "divf0"} : <f32>
    %61 = addi %48, %54 {handshake.bb = 3 : ui32, handshake.name = "addi0"} : <i9>
    %62:2 = fork [2] %61 {handshake.bb = 3 : ui32, handshake.name = "fork9"} : <i9>
    %63 = trunci %62#0 {handshake.bb = 3 : ui32, handshake.name = "trunci0"} : <i9> to <i8>
    %64 = cmpi ult, %62#1, %57 {handshake.bb = 3 : ui32, handshake.name = "cmpi0"} : <i9>
    %65:4 = fork [4] %64 {handshake.bb = 3 : ui32, handshake.name = "fork10"} : <i1>
    %trueResult, %falseResult = cond_br %65#0, %63 {handshake.bb = 3 : ui32, handshake.name = "cond_br7"} : <i1>, <i8>
    sink %falseResult {handshake.name = "sink4"} : <i8>
    %trueResult_0, %falseResult_1 = cond_br %65#1, %60 {handshake.bb = 3 : ui32, handshake.name = "cond_br8"} : <i1>, <f32>
    sink %falseResult_1 {handshake.name = "sink5"} : <f32>
    %trueResult_2, %falseResult_3 = cond_br %65#2, %59#0 {handshake.bb = 3 : ui32, handshake.name = "cond_br9"} : <i1>, <f32>
    %trueResult_4, %falseResult_5 = cond_br %65#3, %43 {handshake.bb = 3 : ui32, handshake.name = "cond_br10"} : <i1>, <>
    sink %falseResult_5 {handshake.name = "sink6"} : <>
    end {handshake.bb = 4 : ui32, handshake.name = "end0"} %falseResult_3, %0#1 : <f32>, <>
  }
}

