module {
  handshake.func @binary_search(%arg0: i32, %arg1: memref<101xi32>, %arg2: none, ...) -> i32 attributes {argNames = ["search", "a", "start"], resNames = ["out0"]} {
    %memOutputs:2, %done = mem_controller[%arg1 : memref<101xi32>] (%addressResult, %addressResult_24) {bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00}>, connectedBlocks = [2 : i32, 4 : i32], name = #handshake.name<"mem_controller0">} : (i32, i32) -> (i32, i32, none)
    %0:4 = fork [4] %arg2 {bb = 0 : ui32, name = #handshake.name<"fork0">} : none
    %1 = constant %0#3 {bb = 0 : ui32, name = #handshake.name<"constant1">, value = false} : i1
    %2 = constant %0#2 {bb = 0 : ui32, name = #handshake.name<"constant2">, value = true} : i1
    %3 = constant %0#1 {bb = 0 : ui32, name = #handshake.name<"constant3">, value = -1 : i32} : i32
    %4 = arith.extsi %1 {bb = 0 : ui32, name = #handshake.name<"extsi0">} : i1 to i9
    %5 = buffer [1] fifo %15#0 {bb = 1 : ui32, name = #handshake.name<"buffer17">} : i1
    %6 = mux %5 [%54, %4] {bb = 1 : ui32, name = #handshake.name<"mux0">} : i1, i9
    %7 = buffer [1] seq %6 {bb = 1 : ui32, name = #handshake.name<"buffer2">} : i9
    %8:2 = fork [2] %7 {bb = 1 : ui32, name = #handshake.name<"fork1">} : i9
    %9 = arith.trunci %8#0 {bb = 1 : ui32, name = #handshake.name<"trunci0">} : i9 to i8
    %10 = buffer [1] fifo %15#3 {bb = 1 : ui32, name = #handshake.name<"buffer16">} : i1
    %11 = mux %10 [%55, %3] {bb = 1 : ui32, name = #handshake.name<"mux1">} : i1, i32
    %12 = mux %15#2 [%49, %2] {bb = 1 : ui32, name = #handshake.name<"mux2">} : i1, i1
    %13 = mux %15#1 [%39#0, %arg0] {bb = 1 : ui32, name = #handshake.name<"mux3">} : i1, i32
    %14 = buffer [1] fifo %trueResult_4 {bb = 1 : ui32, name = #handshake.name<"buffer32">} : none
    %result, %index = control_merge %14, %0#0 {bb = 1 : ui32, name = #handshake.name<"control_merge0">} : none, i1
    %15:4 = fork [4] %index {bb = 1 : ui32, name = #handshake.name<"fork2">} : i1
    %16 = buffer [1] seq %result {bb = 1 : ui32, name = #handshake.name<"buffer28">} : none
    %17:4 = fork [4] %16 {bb = 1 : ui32, name = #handshake.name<"fork3">} : none
    %18 = constant %17#0 {bb = 1 : ui32, name = #handshake.name<"constant13">, value = -1 : i32} : i32
    %19 = constant %17#1 {bb = 1 : ui32, name = #handshake.name<"constant14">, value = true} : i1
    %20 = constant %17#2 {bb = 1 : ui32, name = #handshake.name<"constant15">, value = 1 : i2} : i2
    %21 = source {bb = 1 : ui32, name = #handshake.name<"source0">}
    %22 = constant %21 {bb = 1 : ui32, name = #handshake.name<"constant16">, value = 101 : i8} : i8
    %23 = arith.extsi %22 {bb = 1 : ui32, name = #handshake.name<"extsi6">} : i8 to i9
    %24 = arith.cmpi ult, %8#1, %23 {bb = 1 : ui32, name = #handshake.name<"cmpi0">} : i9
    %25 = buffer [1] seq %12 {bb = 1 : ui32, name = #handshake.name<"buffer5">} : i1
    %26 = arith.andi %24, %25 {bb = 1 : ui32, name = #handshake.name<"andi0">} : i1
    %27 = buffer [1] seq %26 {bb = 1 : ui32, name = #handshake.name<"buffer8">} : i1
    %28:7 = fork [7] %27 {bb = 1 : ui32, name = #handshake.name<"fork4">} : i1
    %29 = buffer [1] fifo %28#6 {bb = 1 : ui32, name = #handshake.name<"buffer12">} : i1
    %30 = buffer [1] seq %11 {bb = 1 : ui32, name = #handshake.name<"buffer14">} : i32
    %trueResult, %falseResult = cond_br %29, %30 {bb = 1 : ui32, name = #handshake.name<"cond_br2">} : i32
    %31 = buffer [1] fifo %9 {bb = 1 : ui32, name = #handshake.name<"buffer26">} : i8
    %trueResult_0, %falseResult_1 = cond_br %28#0, %31 {bb = 1 : ui32, name = #handshake.name<"cond_br0">} : i8
    sink %falseResult_1 {name = #handshake.name<"sink0">} : i8
    %32 = buffer [1] seq %13 {bb = 1 : ui32, name = #handshake.name<"buffer20">} : i32
    %trueResult_2, %falseResult_3 = cond_br %28#5, %32 {bb = 1 : ui32, name = #handshake.name<"cond_br4">} : i32
    %trueResult_4, %falseResult_5 = cond_br %28#4, %17#3 {bb = 1 : ui32, name = #handshake.name<"cond_br5">} : none
    %trueResult_6, %falseResult_7 = cond_br %28#1, %20 {bb = 1 : ui32, name = #handshake.name<"cond_br1">} : i2
    sink %trueResult_6 {name = #handshake.name<"sink1">} : i2
    %33 = arith.extsi %falseResult_7 {bb = 1 : ui32, name = #handshake.name<"extsi1">} : i2 to i9
    %trueResult_8, %falseResult_9 = cond_br %28#3, %18 {bb = 1 : ui32, name = #handshake.name<"cond_br7">} : i32
    sink %trueResult_8 {name = #handshake.name<"sink2">} : i32
    %trueResult_10, %falseResult_11 = cond_br %28#2, %19 {bb = 1 : ui32, name = #handshake.name<"cond_br8">} : i1
    sink %trueResult_10 {name = #handshake.name<"sink3">} : i1
    %34:2 = fork [2] %trueResult_0 {bb = 2 : ui32, name = #handshake.name<"fork5">} : i8
    %35 = arith.extsi %34#0 {bb = 2 : ui32, name = #handshake.name<"extsi2">} : i8 to i9
    %36 = arith.extsi %34#1 {bb = 2 : ui32, name = #handshake.name<"extsi7">} : i8 to i32
    %37:2 = fork [2] %36 {bb = 2 : ui32, name = #handshake.name<"fork6">} : i32
    %38 = buffer [1] seq %trueResult_2 {bb = 2 : ui32, name = #handshake.name<"buffer29">} : i32
    %39:3 = fork [3] %38 {bb = 2 : ui32, name = #handshake.name<"fork7">} : i32
    %40 = source {bb = 2 : ui32, name = #handshake.name<"source1">}
    %41 = constant %40 {bb = 2 : ui32, name = #handshake.name<"constant17">, value = 2 : i3} : i3
    %42 = source {bb = 2 : ui32, name = #handshake.name<"source2">}
    %43 = constant %42 {bb = 2 : ui32, name = #handshake.name<"constant18">, value = false} : i1
    %44 = arith.extsi %43 {bb = 2 : ui32, name = #handshake.name<"extsi8">} : i1 to i3
    %addressResult, %dataResult = mc_load[%37#1] %memOutputs#0 {bb = 2 : ui32, bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00}>, name = #handshake.name<"mc_load0">} : i32, i32
    %45:2 = fork [2] %dataResult {bb = 2 : ui32, name = #handshake.name<"fork8">} : i32
    %46 = arith.cmpi eq, %45#1, %39#2 {bb = 2 : ui32, name = #handshake.name<"cmpi1">} : i32
    %47:2 = fork [2] %46 {bb = 2 : ui32, name = #handshake.name<"fork9">} : i1
    %48 = arith.select %47#1, %37#0, %trueResult {bb = 2 : ui32, name = #handshake.name<"select0">} : i32
    %49 = arith.cmpi ne, %45#0, %39#1 {bb = 2 : ui32, name = #handshake.name<"cmpi2">} : i32
    %50 = arith.select %47#0, %44, %41 {bb = 2 : ui32, name = #handshake.name<"select3">} : i3
    %51 = arith.extsi %50 {bb = 2 : ui32, name = #handshake.name<"extsi9">} : i3 to i9
    %52 = buffer [1] seq %35 {bb = 2 : ui32, name = #handshake.name<"buffer25">} : i9
    %53 = buffer [1] seq %51 {bb = 2 : ui32, name = #handshake.name<"buffer33">} : i9
    %54 = arith.addi %52, %53 {bb = 2 : ui32, name = #handshake.name<"addi0">} : i9
    %55 = buffer [1] seq %48 {bb = 2 : ui32, name = #handshake.name<"buffer22">} : i32
    %56 = buffer [1] fifo %66#0 {bb = 3 : ui32, name = #handshake.name<"buffer10">} : i1
    %57 = mux %56 [%102, %33] {bb = 3 : ui32, name = #handshake.name<"mux4">} : i1, i9
    %58:2 = fork [2] %57 {bb = 3 : ui32, name = #handshake.name<"fork10">} : i9
    %59 = buffer [1] seq %58#0 {bb = 3 : ui32, name = #handshake.name<"buffer30">} : i9
    %60 = arith.trunci %59 {bb = 3 : ui32, name = #handshake.name<"trunci1">} : i9 to i8
    %61 = buffer [1] fifo %66#4 {bb = 3 : ui32, name = #handshake.name<"buffer13">} : i1
    %62 = mux %61 [%103, %falseResult_9] {bb = 3 : ui32, name = #handshake.name<"mux5">} : i1, i32
    %63 = mux %66#3 [%97, %falseResult_11] {bb = 3 : ui32, name = #handshake.name<"mux6">} : i1, i1
    %64 = mux %66#2 [%85#0, %falseResult_3] {bb = 3 : ui32, name = #handshake.name<"mux7">} : i1, i32
    %65 = mux %66#1 [%104, %falseResult] {bb = 3 : ui32, name = #handshake.name<"mux8">} : i1, i32
    %result_12, %index_13 = control_merge %86, %falseResult_5 {bb = 3 : ui32, name = #handshake.name<"control_merge1">} : none, i1
    %66:5 = fork [5] %index_13 {bb = 3 : ui32, name = #handshake.name<"fork11">} : i1
    %67 = source {bb = 3 : ui32, name = #handshake.name<"source3">}
    %68 = constant %67 {bb = 3 : ui32, name = #handshake.name<"constant19">, value = 101 : i8} : i8
    %69 = arith.extsi %68 {bb = 3 : ui32, name = #handshake.name<"extsi10">} : i8 to i9
    %70 = buffer [1] seq %58#1 {bb = 3 : ui32, name = #handshake.name<"buffer18">} : i9
    %71 = arith.cmpi ult, %70, %69 {bb = 3 : ui32, name = #handshake.name<"cmpi3">} : i9
    %72 = buffer [1] seq %63 {bb = 3 : ui32, name = #handshake.name<"buffer1">} : i1
    %73 = arith.andi %71, %72 {bb = 3 : ui32, name = #handshake.name<"andi1">} : i1
    %74 = buffer [1] seq %73 {bb = 3 : ui32, name = #handshake.name<"buffer11">} : i1
    %75:5 = fork [5] %74 {bb = 3 : ui32, name = #handshake.name<"fork12">} : i1
    %76 = buffer [1] seq %62 {bb = 3 : ui32, name = #handshake.name<"buffer21">} : i32
    %trueResult_14, %falseResult_15 = cond_br %75#4, %76 {bb = 3 : ui32, name = #handshake.name<"cond_br12">} : i32
    %trueResult_16, %falseResult_17 = cond_br %75#0, %60 {bb = 3 : ui32, name = #handshake.name<"cond_br3">} : i8
    sink %falseResult_17 {name = #handshake.name<"sink4">} : i8
    %77 = buffer [1] seq %64 {bb = 3 : ui32, name = #handshake.name<"buffer15">} : i32
    %trueResult_18, %falseResult_19 = cond_br %75#3, %77 {bb = 3 : ui32, name = #handshake.name<"cond_br14">} : i32
    sink %falseResult_19 {name = #handshake.name<"sink5">} : i32
    %78 = buffer [1] seq %65 {bb = 3 : ui32, name = #handshake.name<"buffer19">} : i32
    %trueResult_20, %falseResult_21 = cond_br %75#2, %78 {bb = 3 : ui32, name = #handshake.name<"cond_br15">} : i32
    %79 = buffer [1] seq %result_12 {bb = 3 : ui32, name = #handshake.name<"buffer3">} : none
    %trueResult_22, %falseResult_23 = cond_br %75#1, %79 {bb = 3 : ui32, name = #handshake.name<"cond_br16">} : none
    sink %falseResult_23 {name = #handshake.name<"sink6">} : none
    %80:2 = fork [2] %trueResult_16 {bb = 4 : ui32, name = #handshake.name<"fork13">} : i8
    %81 = arith.extsi %80#0 {bb = 4 : ui32, name = #handshake.name<"extsi3">} : i8 to i9
    %82 = arith.extsi %80#1 {bb = 4 : ui32, name = #handshake.name<"extsi11">} : i8 to i32
    %83:2 = fork [2] %82 {bb = 4 : ui32, name = #handshake.name<"fork14">} : i32
    %84 = buffer [1] seq %trueResult_18 {bb = 4 : ui32, name = #handshake.name<"buffer31">} : i32
    %85:3 = fork [3] %84 {bb = 4 : ui32, name = #handshake.name<"fork15">} : i32
    %86 = buffer [1] fifo %trueResult_22 {bb = 4 : ui32, name = #handshake.name<"buffer27">} : none
    %87 = source {bb = 4 : ui32, name = #handshake.name<"source4">}
    %88 = constant %87 {bb = 4 : ui32, name = #handshake.name<"constant20">, value = 2 : i3} : i3
    %89 = source {bb = 4 : ui32, name = #handshake.name<"source5">}
    %90 = constant %89 {bb = 4 : ui32, name = #handshake.name<"constant21">, value = false} : i1
    %91 = arith.extsi %90 {bb = 4 : ui32, name = #handshake.name<"extsi12">} : i1 to i3
    %addressResult_24, %dataResult_25 = mc_load[%83#1] %memOutputs#1 {bb = 4 : ui32, bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00}>, name = #handshake.name<"mc_load1">} : i32, i32
    %92:2 = fork [2] %dataResult_25 {bb = 4 : ui32, name = #handshake.name<"fork16">} : i32
    %93 = arith.cmpi eq, %92#1, %85#2 {bb = 4 : ui32, name = #handshake.name<"cmpi4">} : i32
    %94:2 = fork [2] %93 {bb = 4 : ui32, name = #handshake.name<"fork17">} : i1
    %95 = buffer [1] seq %trueResult_14 {bb = 4 : ui32, name = #handshake.name<"buffer7">} : i32
    %96 = arith.select %94#1, %83#0, %95 {bb = 4 : ui32, name = #handshake.name<"select1">} : i32
    %97 = arith.cmpi ne, %92#0, %85#1 {bb = 4 : ui32, name = #handshake.name<"cmpi5">} : i32
    %98 = arith.select %94#0, %91, %88 {bb = 4 : ui32, name = #handshake.name<"select5">} : i3
    %99 = arith.extsi %98 {bb = 4 : ui32, name = #handshake.name<"extsi13">} : i3 to i9
    %100 = buffer [1] seq %99 {bb = 4 : ui32, name = #handshake.name<"buffer0">} : i9
    %101 = buffer [1] seq %81 {bb = 4 : ui32, name = #handshake.name<"buffer9">} : i9
    %102 = arith.addi %101, %100 {bb = 4 : ui32, name = #handshake.name<"addi1">} : i9
    %103 = buffer [1] seq %96 {bb = 4 : ui32, name = #handshake.name<"buffer23">} : i32
    %104 = buffer [1] fifo %trueResult_20 {bb = 4 : ui32, name = #handshake.name<"buffer6">} : i32
    %105 = buffer [1] seq %falseResult_21 {bb = 5 : ui32, name = #handshake.name<"buffer4">} : i32
    %106:2 = fork [2] %105 {bb = 5 : ui32, name = #handshake.name<"fork18">} : i32
    %107 = source {bb = 5 : ui32, name = #handshake.name<"source6">}
    %108 = constant %107 {bb = 5 : ui32, name = #handshake.name<"constant22">, value = -1 : i32} : i32
    %109 = arith.cmpi ne, %106#1, %108 {bb = 5 : ui32, name = #handshake.name<"cmpi6">} : i32
    %110 = arith.select %109, %106#0, %falseResult_15 {bb = 5 : ui32, name = #handshake.name<"select2">} : i32
    %111 = buffer [1] seq %110 {bb = 5 : ui32, name = #handshake.name<"buffer24">} : i32
    %112 = d_return {bb = 5 : ui32, name = #handshake.name<"d_return0">} %111 : i32
    end {bb = 5 : ui32, bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00}>, name = #handshake.name<"end0">} %112, %done : i32, none
  }
}

