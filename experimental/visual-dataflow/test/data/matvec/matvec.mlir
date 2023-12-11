module {
  handshake.func @matvec(%arg0: memref<10000xi32>, %arg1: memref<100xi32>, %arg2: memref<100xi32>, %arg3: none, ...) -> i32 attributes {argNames = ["m", "v", "out", "start"], resNames = ["out0"]} {
    %done = mem_controller[%arg2 : memref<100xi32>] (%58, %addressResult_13, %dataResult_14) {bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00}>, connectedBlocks = [3 : i32], name = #handshake.name<"mem_controller0">} : (i32, i32, i32) -> none
    %memOutputs, %done_0 = mem_controller[%arg1 : memref<100xi32>] (%addressResult) {bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00}>, connectedBlocks = [2 : i32], name = #handshake.name<"mem_controller1">} : (i32) -> (i32, none)
    %memOutputs_1, %done_2 = mem_controller[%arg0 : memref<10000xi32>] (%addressResult_5) {bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00}>, connectedBlocks = [2 : i32], name = #handshake.name<"mem_controller2">} : (i32) -> (i32, none)
    %0:2 = fork [2] %arg3 {bb = 0 : ui32, name = #handshake.name<"fork0">} : none
    %1 = constant %0#1 {bb = 0 : ui32, name = #handshake.name<"constant0">, value = false} : i1
    %2 = arith.extsi %1 {bb = 0 : ui32, name = #handshake.name<"extsi0">} : i1 to i8
    %3 = mux %index [%trueResult_15, %2] {bb = 1 : ui32, name = #handshake.name<"mux0">} : i1, i8
    %result, %index = control_merge %trueResult_17, %0#0 {bb = 1 : ui32, name = #handshake.name<"control_merge0">} : none, i1
    %4:2 = fork [2] %result {bb = 1 : ui32, name = #handshake.name<"fork1">} : none
    %5 = constant %4#0 {bb = 1 : ui32, name = #handshake.name<"constant1">, value = false} : i1
    %6:2 = fork [2] %5 {bb = 1 : ui32, name = #handshake.name<"fork2">} : i1
    %7 = arith.extsi %6#0 {bb = 1 : ui32, name = #handshake.name<"extsi1">} : i1 to i8
    %8 = arith.extsi %6#1 {bb = 1 : ui32, name = #handshake.name<"extsi6">} : i1 to i32
    %9 = mux %25#1 [%trueResult, %7] {bb = 2 : ui32, name = #handshake.name<"mux4">} : i1, i8
    %10 = buffer [1] seq %9 {bb = 2 : ui32, name = #handshake.name<"buffer13">} : i8
    %11:3 = fork [3] %10 {bb = 2 : ui32, name = #handshake.name<"fork3">} : i8
    %12 = buffer [3] fifo %11#0 {bb = 2 : ui32, name = #handshake.name<"buffer8">} : i8
    %13 = arith.extsi %12 {bb = 2 : ui32, name = #handshake.name<"extsi7">} : i8 to i17
    %14 = arith.extsi %11#1 {bb = 2 : ui32, name = #handshake.name<"extsi8">} : i8 to i9
    %15 = buffer [3] fifo %11#2 {bb = 2 : ui32, name = #handshake.name<"buffer10">} : i8
    %16 = arith.extsi %15 {bb = 2 : ui32, name = #handshake.name<"extsi9">} : i8 to i32
    %17 = buffer [5] fifo %25#2 {bb = 2 : ui32, name = #handshake.name<"buffer16">} : i1
    %18 = mux %17 [%trueResult_7, %8] {bb = 2 : ui32, name = #handshake.name<"mux2">} : i1, i32
    %19 = buffer [1] fifo %25#0 {bb = 2 : ui32, name = #handshake.name<"buffer0">} : i1
    %20 = buffer [1] seq %3 {bb = 2 : ui32, name = #handshake.name<"buffer11">} : i8
    %21 = mux %19 [%trueResult_9, %20] {bb = 2 : ui32, name = #handshake.name<"mux1">} : i1, i8
    %22 = buffer [2] seq %21 {bb = 2 : ui32, name = #handshake.name<"buffer12">} : i8
    %23:2 = fork [2] %22 {bb = 2 : ui32, name = #handshake.name<"fork4">} : i8
    %24 = arith.extsi %23#1 {bb = 2 : ui32, name = #handshake.name<"extsi10">} : i8 to i16
    %result_3, %index_4 = control_merge %trueResult_11, %4#1 {bb = 2 : ui32, name = #handshake.name<"control_merge1">} : none, i1
    %25:3 = fork [3] %index_4 {bb = 2 : ui32, name = #handshake.name<"fork5">} : i1
    %26 = source {bb = 2 : ui32, name = #handshake.name<"source0">}
    %27 = constant %26 {bb = 2 : ui32, name = #handshake.name<"constant8">, value = 100 : i8} : i8
    %28:2 = fork [2] %27 {bb = 2 : ui32, name = #handshake.name<"fork6">} : i8
    %29 = arith.extsi %28#0 {bb = 2 : ui32, name = #handshake.name<"extsi11">} : i8 to i16
    %30 = arith.extsi %28#1 {bb = 2 : ui32, name = #handshake.name<"extsi12">} : i8 to i9
    %31 = source {bb = 2 : ui32, name = #handshake.name<"source1">}
    %32 = constant %31 {bb = 2 : ui32, name = #handshake.name<"constant9">, value = 1 : i2} : i2
    %33 = arith.extsi %32 {bb = 2 : ui32, name = #handshake.name<"extsi13">} : i2 to i9
    %addressResult, %dataResult = mc_load[%16] %memOutputs {bb = 2 : ui32, bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00}>, name = #handshake.name<"mc_load0">} : i32, i32
    %34 = arith.muli %24, %29 {bb = 2 : ui32, name = #handshake.name<"muli1">} : i16
    %35 = arith.extsi %34 {bb = 2 : ui32, name = #handshake.name<"extsi14">} : i16 to i17
    %36 = arith.addi %13, %35 {bb = 2 : ui32, name = #handshake.name<"addi1">} : i17
    %37 = arith.extsi %36 {bb = 2 : ui32, name = #handshake.name<"extsi15">} : i17 to i32
    %addressResult_5, %dataResult_6 = mc_load[%37] %memOutputs_1 {bb = 2 : ui32, bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00}>, name = #handshake.name<"mc_load1">} : i32, i32
    %38 = arith.muli %dataResult, %dataResult_6 {bb = 2 : ui32, name = #handshake.name<"muli0">} : i32
    %39 = buffer [4] seq %18 {bb = 2 : ui32, name = #handshake.name<"buffer1">} : i32
    %40 = arith.addi %39, %38 {bb = 2 : ui32, name = #handshake.name<"addi0">} : i32
    %41 = arith.addi %14, %33 {bb = 2 : ui32, name = #handshake.name<"addi4">} : i9
    %42 = buffer [1] seq %41 {bb = 2 : ui32, name = #handshake.name<"buffer5">} : i9
    %43:2 = fork [2] %42 {bb = 2 : ui32, name = #handshake.name<"fork7">} : i9
    %44 = arith.trunci %43#0 {bb = 2 : ui32, name = #handshake.name<"trunci0">} : i9 to i8
    %45 = arith.cmpi ult, %43#1, %30 {bb = 2 : ui32, name = #handshake.name<"cmpi0">} : i9
    %46:4 = fork [4] %45 {bb = 2 : ui32, name = #handshake.name<"fork8">} : i1
    %trueResult, %falseResult = cond_br %46#0, %44 {bb = 2 : ui32, name = #handshake.name<"cond_br0">} : i8
    sink %falseResult {name = #handshake.name<"sink0">} : i8
    %47 = buffer [5] fifo %46#2 {bb = 2 : ui32, name = #handshake.name<"buffer17">} : i1
    %trueResult_7, %falseResult_8 = cond_br %47, %40 {bb = 2 : ui32, name = #handshake.name<"cond_br3">} : i32
    %trueResult_9, %falseResult_10 = cond_br %46#1, %23#0 {bb = 2 : ui32, name = #handshake.name<"cond_br1">} : i8
    %48 = buffer [2] seq %result_3 {bb = 2 : ui32, name = #handshake.name<"buffer3">} : none
    %trueResult_11, %falseResult_12 = cond_br %46#3, %48 {bb = 2 : ui32, name = #handshake.name<"cond_br5">} : none
    %49 = buffer [1] seq %falseResult_10 {bb = 3 : ui32, name = #handshake.name<"buffer4">} : i8
    %50:2 = fork [2] %49 {bb = 3 : ui32, name = #handshake.name<"fork9">} : i8
    %51 = arith.extsi %50#0 {bb = 3 : ui32, name = #handshake.name<"extsi2">} : i8 to i9
    %52 = buffer [3] fifo %50#1 {bb = 3 : ui32, name = #handshake.name<"buffer7">} : i8
    %53 = arith.extsi %52 {bb = 3 : ui32, name = #handshake.name<"extsi16">} : i8 to i32
    %54:2 = fork [2] %falseResult_8 {bb = 3 : ui32, name = #handshake.name<"fork10">} : i32
    %55 = buffer [1] seq %falseResult_12 {bb = 3 : ui32, name = #handshake.name<"buffer2">} : none
    %56:2 = fork [2] %55 {bb = 3 : ui32, name = #handshake.name<"fork11">} : none
    %57 = constant %56#1 {bb = 3 : ui32, name = #handshake.name<"constant10">, value = 1 : i2} : i2
    %58 = arith.extsi %57 {bb = 3 : ui32, name = #handshake.name<"extsi3">} : i2 to i32
    %59 = source {bb = 3 : ui32, name = #handshake.name<"source2">}
    %60 = constant %59 {bb = 3 : ui32, name = #handshake.name<"constant11">, value = 100 : i8} : i8
    %61 = arith.extsi %60 {bb = 3 : ui32, name = #handshake.name<"extsi4">} : i8 to i9
    %62 = source {bb = 3 : ui32, name = #handshake.name<"source3">}
    %63 = constant %62 {bb = 3 : ui32, name = #handshake.name<"constant12">, value = 1 : i2} : i2
    %64 = arith.extsi %63 {bb = 3 : ui32, name = #handshake.name<"extsi17">} : i2 to i9
    %addressResult_13, %dataResult_14 = mc_store[%53] %54#1 {bb = 3 : ui32, name = #handshake.name<"mc_store0">} : i32, i32
    %65 = arith.addi %51, %64 {bb = 3 : ui32, name = #handshake.name<"addi2">} : i9
    %66 = buffer [1] seq %65 {bb = 3 : ui32, name = #handshake.name<"buffer15">} : i9
    %67:2 = fork [2] %66 {bb = 3 : ui32, name = #handshake.name<"fork12">} : i9
    %68 = arith.trunci %67#0 {bb = 3 : ui32, name = #handshake.name<"trunci1">} : i9 to i8
    %69 = buffer [1] seq %61 {bb = 3 : ui32, name = #handshake.name<"buffer9">} : i9
    %70 = arith.cmpi ult, %67#1, %69 {bb = 3 : ui32, name = #handshake.name<"cmpi1">} : i9
    %71:3 = fork [3] %70 {bb = 3 : ui32, name = #handshake.name<"fork13">} : i1
    %trueResult_15, %falseResult_16 = cond_br %71#0, %68 {bb = 3 : ui32, name = #handshake.name<"cond_br2">} : i8
    sink %falseResult_16 {name = #handshake.name<"sink1">} : i8
    %trueResult_17, %falseResult_18 = cond_br %71#1, %56#0 {bb = 3 : ui32, name = #handshake.name<"cond_br10">} : none
    sink %falseResult_18 {name = #handshake.name<"sink2">} : none
    %72 = buffer [2] fifo %71#2 {bb = 3 : ui32, name = #handshake.name<"buffer6">} : i1
    %trueResult_19, %falseResult_20 = cond_br %72, %54#0 {bb = 3 : ui32, name = #handshake.name<"cond_br11">} : i32
    sink %trueResult_19 {name = #handshake.name<"sink3">} : i32
    %73 = buffer [1] seq %falseResult_20 {bb = 4 : ui32, name = #handshake.name<"buffer14">} : i32
    %74 = d_return {bb = 4 : ui32, name = #handshake.name<"d_return0">} %73 : i32
    end {bb = 4 : ui32, bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00}>, name = #handshake.name<"end0">} %74, %done, %done_0, %done_2 : i32, none, none, none
  }
}

