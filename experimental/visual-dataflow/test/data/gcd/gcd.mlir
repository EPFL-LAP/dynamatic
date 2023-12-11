module {
  handshake.func @gcd(%arg0: i32, %arg1: i32, %arg2: none, ...) -> i32 attributes {argNames = ["a", "b", "start"], resNames = ["out0"]} {
    %0:2 = fork [2] %arg2 {bb = 0 : ui32, name = #handshake.name<"fork0">} : none
    %1 = constant %0#1 {bb = 0 : ui32, name = #handshake.name<"constant1">, value = false} : i1
    %2 = arith.extsi %1 {bb = 0 : ui32, name = #handshake.name<"extsi0">} : i1 to i32
    %3 = mux %9#2 [%trueResult_12, %2] {bb = 1 : ui32, name = #handshake.name<"mux0">} : i1, i32
    %4 = mux %9#1 [%trueResult_14, %arg1] {bb = 1 : ui32, name = #handshake.name<"mux1">} : i1, i32
    %5 = buffer [1] seq %4 {bb = 1 : ui32, name = #handshake.name<"buffer8">} : i32
    %6:2 = fork [2] %5 {bb = 1 : ui32, name = #handshake.name<"fork1">} : i32
    %7 = mux %9#0 [%trueResult_16, %arg0] {bb = 1 : ui32, name = #handshake.name<"mux2">} : i1, i32
    %8:3 = fork [3] %7 {bb = 1 : ui32, name = #handshake.name<"fork2">} : i32
    %result, %index = control_merge %trueResult_18, %0#0 {bb = 1 : ui32, name = #handshake.name<"control_merge0">} : none, i1
    %9:3 = fork [3] %index {bb = 1 : ui32, name = #handshake.name<"fork3">} : i1
    %10 = source {bb = 1 : ui32, name = #handshake.name<"source0">}
    %11 = constant %10 {bb = 1 : ui32, name = #handshake.name<"constant8">, value = 1 : i2} : i2
    %12 = source {bb = 1 : ui32, name = #handshake.name<"source1">}
    %13 = constant %12 {bb = 1 : ui32, name = #handshake.name<"constant9">, value = false} : i1
    %14 = arith.extsi %13 {bb = 1 : ui32, name = #handshake.name<"extsi8">} : i1 to i2
    %15 = buffer [1] seq %8#2 {bb = 1 : ui32, name = #handshake.name<"buffer28">} : i32
    %16 = arith.ori %15, %6#1 {bb = 1 : ui32, name = #handshake.name<"ori0">} : i32
    %17 = arith.trunci %16 {bb = 1 : ui32, name = #handshake.name<"trunci1">} : i32 to i2
    %18 = arith.andi %17, %11 {bb = 1 : ui32, name = #handshake.name<"andi0">} : i2
    %19 = buffer [1] seq %18 {bb = 1 : ui32, name = #handshake.name<"buffer16">} : i2
    %20 = arith.cmpi eq, %19, %14 {bb = 1 : ui32, name = #handshake.name<"cmpi0">} : i2
    %21:7 = fork [7] %20 {bb = 1 : ui32, name = #handshake.name<"fork4">} : i1
    %22 = buffer [1] seq %3 {bb = 1 : ui32, name = #handshake.name<"buffer5">} : i32
    %23 = buffer [1] seq %21#0 {bb = 1 : ui32, name = #handshake.name<"buffer22">} : i1
    %trueResult, %falseResult = cond_br %23, %22 {bb = 1 : ui32, name = #handshake.name<"cond_br3">} : i32
    %24 = buffer [1] seq %6#0 {bb = 1 : ui32, name = #handshake.name<"buffer29">} : i32
    %trueResult_0, %falseResult_1 = cond_br %21#1, %24 {bb = 1 : ui32, name = #handshake.name<"cond_br4">} : i32
    %25 = buffer [1] seq %8#1 {bb = 1 : ui32, name = #handshake.name<"buffer20">} : i32
    %trueResult_2, %falseResult_3 = cond_br %21#2, %25 {bb = 1 : ui32, name = #handshake.name<"cond_br5">} : i32
    %trueResult_4, %falseResult_5 = cond_br %21#3, %21#4 {bb = 1 : ui32, name = #handshake.name<"cond_br6">} : i1
    %26 = buffer [1] fifo %result {bb = 1 : ui32, name = #handshake.name<"buffer1">} : none
    %trueResult_6, %falseResult_7 = cond_br %21#5, %26 {bb = 1 : ui32, name = #handshake.name<"cond_br7">} : none
    %27 = buffer [1] seq %8#0 {bb = 1 : ui32, name = #handshake.name<"buffer7">} : i32
    %28 = buffer [1] seq %21#6 {bb = 1 : ui32, name = #handshake.name<"buffer15">} : i1
    %trueResult_8, %falseResult_9 = cond_br %28, %27 {bb = 1 : ui32, name = #handshake.name<"cond_br10">} : i32
    sink %trueResult_8 {name = #handshake.name<"sink0">} : i32
    %29 = buffer [1] seq %trueResult_6 {bb = 2 : ui32, name = #handshake.name<"buffer27">} : none
    %30:2 = fork [2] %29 {bb = 2 : ui32, name = #handshake.name<"fork5">} : none
    %31 = source {bb = 2 : ui32, name = #handshake.name<"source2">}
    %32 = constant %31 {bb = 2 : ui32, name = #handshake.name<"constant10">, value = 1 : i2} : i2
    %33 = arith.extsi %32 {bb = 2 : ui32, name = #handshake.name<"extsi3">} : i2 to i32
    %34:3 = fork [3] %33 {bb = 2 : ui32, name = #handshake.name<"fork6">} : i32
    %35 = buffer [1] seq %trueResult_2 {bb = 2 : ui32, name = #handshake.name<"buffer17">} : i32
    %36 = arith.shrsi %35, %34#0 {bb = 2 : ui32, name = #handshake.name<"shrsi0">} : i32
    %37 = buffer [1] seq %trueResult_0 {bb = 2 : ui32, name = #handshake.name<"buffer26">} : i32
    %38 = arith.shrsi %37, %34#1 {bb = 2 : ui32, name = #handshake.name<"shrsi1">} : i32
    %39 = arith.addi %trueResult, %34#2 {bb = 2 : ui32, name = #handshake.name<"addi0">} : i32
    %40 = constant %30#1 {bb = 2 : ui32, name = #handshake.name<"constant11">, value = false} : i1
    %41 = buffer [1] seq %39 {bb = 2 : ui32, name = #handshake.name<"buffer9">} : i32
    %42 = arith.extsi %40 {bb = 2 : ui32, name = #handshake.name<"extsi9">} : i1 to i32
    %43 = mux %56#4 [%41, %falseResult] {bb = 3 : ui32, name = #handshake.name<"mux3">} : i1, i32
    %44 = buffer [1] seq %38 {bb = 3 : ui32, name = #handshake.name<"buffer0">} : i32
    %45 = buffer [1] seq %falseResult_1 {bb = 3 : ui32, name = #handshake.name<"buffer6">} : i32
    %46 = mux %56#3 [%44, %45] {bb = 3 : ui32, name = #handshake.name<"mux4">} : i1, i32
    %47 = buffer [1] seq %36 {bb = 3 : ui32, name = #handshake.name<"buffer24">} : i32
    %48 = mux %56#2 [%47, %falseResult_9] {bb = 3 : ui32, name = #handshake.name<"mux5">} : i1, i32
    %49 = buffer [1] seq %falseResult_3 {bb = 3 : ui32, name = #handshake.name<"buffer4">} : i32
    %50 = mux %56#1 [%42, %49] {bb = 3 : ui32, name = #handshake.name<"mux6">} : i1, i32
    %51 = buffer [1] seq %trueResult_4 {bb = 3 : ui32, name = #handshake.name<"buffer11">} : i1
    %52 = buffer [1] seq %falseResult_5 {bb = 3 : ui32, name = #handshake.name<"buffer18">} : i1
    %53 = mux %56#0 [%51, %52] {bb = 3 : ui32, name = #handshake.name<"mux7">} : i1, i1
    %54:5 = fork [5] %53 {bb = 3 : ui32, name = #handshake.name<"fork7">} : i1
    %55 = buffer [1] seq %falseResult_7 {bb = 3 : ui32, name = #handshake.name<"buffer10">} : none
    %result_10, %index_11 = control_merge %30#0, %55 {bb = 3 : ui32, name = #handshake.name<"control_merge1">} : none, i1
    %56:5 = fork [5] %index_11 {bb = 3 : ui32, name = #handshake.name<"fork8">} : i1
    %trueResult_12, %falseResult_13 = cond_br %54#4, %43 {bb = 3 : ui32, name = #handshake.name<"cond_br14">} : i32
    %trueResult_14, %falseResult_15 = cond_br %54#3, %46 {bb = 3 : ui32, name = #handshake.name<"cond_br15">} : i32
    sink %falseResult_15 {name = #handshake.name<"sink1">} : i32
    %trueResult_16, %falseResult_17 = cond_br %54#2, %48 {bb = 3 : ui32, name = #handshake.name<"cond_br16">} : i32
    sink %falseResult_17 {name = #handshake.name<"sink2">} : i32
    %trueResult_18, %falseResult_19 = cond_br %54#1, %result_10 {bb = 3 : ui32, name = #handshake.name<"cond_br17">} : none
    %trueResult_20, %falseResult_21 = cond_br %54#0, %50 {bb = 3 : ui32, name = #handshake.name<"cond_br18">} : i32
    sink %trueResult_20 {name = #handshake.name<"sink3">} : i32
    %57 = buffer [1] seq %79 {bb = 4 : ui32, name = #handshake.name<"buffer13">} : i32
    %58 = mux %64#1 [%57, %falseResult_21] {bb = 4 : ui32, name = #handshake.name<"mux8">} : i1, i32
    %59 = buffer [1] seq %58 {bb = 4 : ui32, name = #handshake.name<"buffer3">} : i32
    %60:2 = fork [2] %59 {bb = 4 : ui32, name = #handshake.name<"fork9">} : i32
    %61 = arith.trunci %60#0 {bb = 4 : ui32, name = #handshake.name<"trunci0">} : i32 to i2
    %62 = mux %64#0 [%trueResult_26, %falseResult_13] {bb = 4 : ui32, name = #handshake.name<"mux9">} : i1, i32
    %63 = buffer [1] seq %trueResult_28 {bb = 4 : ui32, name = #handshake.name<"buffer2">} : none
    %result_22, %index_23 = control_merge %63, %falseResult_19 {bb = 4 : ui32, name = #handshake.name<"control_merge2">} : none, i1
    %64:2 = fork [2] %index_23 {bb = 4 : ui32, name = #handshake.name<"fork10">} : i1
    %65 = source {bb = 4 : ui32, name = #handshake.name<"source3">}
    %66 = constant %65 {bb = 4 : ui32, name = #handshake.name<"constant12">, value = 1 : i2} : i2
    %67 = source {bb = 4 : ui32, name = #handshake.name<"source4">}
    %68 = constant %67 {bb = 4 : ui32, name = #handshake.name<"constant13">, value = false} : i1
    %69 = arith.extsi %68 {bb = 4 : ui32, name = #handshake.name<"extsi10">} : i1 to i2
    %70 = arith.andi %61, %66 {bb = 4 : ui32, name = #handshake.name<"andi2">} : i2
    %71 = arith.cmpi eq, %70, %69 {bb = 4 : ui32, name = #handshake.name<"cmpi1">} : i2
    %72 = buffer [1] seq %71 {bb = 4 : ui32, name = #handshake.name<"buffer23">} : i1
    %73:3 = fork [3] %72 {bb = 4 : ui32, name = #handshake.name<"fork11">} : i1
    %trueResult_24, %falseResult_25 = cond_br %73#0, %60#1 {bb = 4 : ui32, name = #handshake.name<"cond_br21">} : i32
    %74 = buffer [1] seq %62 {bb = 4 : ui32, name = #handshake.name<"buffer25">} : i32
    %trueResult_26, %falseResult_27 = cond_br %73#1, %74 {bb = 4 : ui32, name = #handshake.name<"cond_br22">} : i32
    %trueResult_28, %falseResult_29 = cond_br %73#2, %result_22 {bb = 4 : ui32, name = #handshake.name<"cond_br23">} : none
    sink %falseResult_29 {name = #handshake.name<"sink4">} : none
    %75 = source {bb = 5 : ui32, name = #handshake.name<"source5">}
    %76 = constant %75 {bb = 5 : ui32, name = #handshake.name<"constant14">, value = 1 : i2} : i2
    %77 = arith.extsi %76 {bb = 5 : ui32, name = #handshake.name<"extsi7">} : i2 to i32
    %78 = buffer [1] seq %trueResult_24 {bb = 5 : ui32, name = #handshake.name<"buffer14">} : i32
    %79 = arith.shrsi %78, %77 {bb = 5 : ui32, name = #handshake.name<"shrsi2">} : i32
    %80 = buffer [1] seq %falseResult_27 {bb = 6 : ui32, name = #handshake.name<"buffer12">} : i32
    %81 = buffer [1] seq %falseResult_25 {bb = 6 : ui32, name = #handshake.name<"buffer19">} : i32
    %82 = arith.shli %81, %80 {bb = 6 : ui32, name = #handshake.name<"shli0">} : i32
    %83 = buffer [1] seq %82 {bb = 6 : ui32, name = #handshake.name<"buffer21">} : i32
    %84 = d_return {bb = 6 : ui32, name = #handshake.name<"d_return0">} %83 : i32
    end {bb = 6 : ui32, name = #handshake.name<"end0">} %84 : i32
  }
}

