module {
  handshake.func @fir(%arg0: memref<1000xi32>, %arg1: memref<1000xi32>, %arg2: none, ...) -> i32 attributes {argNames = ["di", "idx", "start"], resNames = ["out0"]} {
    %memOutputs, %done = mem_controller[%arg1 : memref<1000xi32>] (%addressResult) {bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00}>, connectedBlocks = [1 : i32], name = #handshake.name<"mem_controller0">} : (i32) -> (i32, none)
    %memOutputs_0, %done_1 = mem_controller[%arg0 : memref<1000xi32>] (%addressResult_2) {bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00}>, connectedBlocks = [1 : i32], name = #handshake.name<"mem_controller1">} : (i32) -> (i32, none)
    %0:2 = fork [2] %arg2 {bb = 0 : ui32, name = #handshake.name<"fork0">} : none
    %1 = constant %0#1 {bb = 0 : ui32, name = #handshake.name<"constant1">, value = false} : i1
    %2:2 = fork [2] %1 {bb = 0 : ui32, name = #handshake.name<"fork1">} : i1
    %3 = arith.extsi %2#0 {bb = 0 : ui32, name = #handshake.name<"extsi0">} : i1 to i11
    %4 = arith.extsi %2#1 {bb = 0 : ui32, name = #handshake.name<"extsi1">} : i1 to i32
    %5 = mux %13#0 [%trueResult, %3] {bb = 1 : ui32, name = #handshake.name<"mux2">} : i1, i11
    %6 = buffer [1] seq %5 {bb = 1 : ui32, name = #handshake.name<"buffer3">} : i11
    %7:3 = fork [3] %6 {bb = 1 : ui32, name = #handshake.name<"fork2">} : i11
    %8 = arith.extsi %7#0 {bb = 1 : ui32, name = #handshake.name<"extsi2">} : i11 to i12
    %9 = arith.extsi %7#1 {bb = 1 : ui32, name = #handshake.name<"extsi3">} : i11 to i12
    %10 = arith.extsi %7#2 {bb = 1 : ui32, name = #handshake.name<"extsi4">} : i11 to i32
    %11 = buffer [3] fifo %13#1 {bb = 1 : ui32, name = #handshake.name<"buffer4">} : i1
    %12 = mux %11 [%trueResult_4, %4] {bb = 1 : ui32, name = #handshake.name<"mux1">} : i1, i32
    %result, %index = control_merge %trueResult_6, %0#0 {bb = 1 : ui32, name = #handshake.name<"control_merge0">} : none, i1
    %13:2 = fork [2] %index {bb = 1 : ui32, name = #handshake.name<"fork3">} : i1
    %14 = source {bb = 1 : ui32, name = #handshake.name<"source0">}
    %15 = constant %14 {bb = 1 : ui32, name = #handshake.name<"constant4">, value = 999 : i11} : i11
    %16 = arith.extsi %15 {bb = 1 : ui32, name = #handshake.name<"extsi5">} : i11 to i12
    %17 = source {bb = 1 : ui32, name = #handshake.name<"source1">}
    %18 = constant %17 {bb = 1 : ui32, name = #handshake.name<"constant6">, value = 1000 : i11} : i11
    %19 = arith.extsi %18 {bb = 1 : ui32, name = #handshake.name<"extsi8">} : i11 to i12
    %20 = source {bb = 1 : ui32, name = #handshake.name<"source2">}
    %21 = constant %20 {bb = 1 : ui32, name = #handshake.name<"constant7">, value = 1 : i2} : i2
    %22 = arith.extsi %21 {bb = 1 : ui32, name = #handshake.name<"extsi6">} : i2 to i12
    %addressResult, %dataResult = mc_load[%10] %memOutputs {bb = 1 : ui32, bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00}>, name = #handshake.name<"mc_load0">} : i32, i32
    %23 = arith.subi %16, %9 {bb = 1 : ui32, name = #handshake.name<"subi1">} : i12
    %24 = arith.extsi %23 {bb = 1 : ui32, name = #handshake.name<"extsi7">} : i12 to i32
    %addressResult_2, %dataResult_3 = mc_load[%24] %memOutputs_0 {bb = 1 : ui32, bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00}>, name = #handshake.name<"mc_load1">} : i32, i32
    %25 = arith.muli %dataResult, %dataResult_3 {bb = 1 : ui32, name = #handshake.name<"muli0">} : i32
    %26 = buffer [2] seq %12 {bb = 1 : ui32, name = #handshake.name<"buffer5">} : i32
    %27 = arith.addi %26, %25 {bb = 1 : ui32, name = #handshake.name<"addi0">} : i32
    %28 = arith.addi %8, %22 {bb = 1 : ui32, name = #handshake.name<"addi1">} : i12
    %29 = buffer [1] seq %28 {bb = 1 : ui32, name = #handshake.name<"buffer6">} : i12
    %30:2 = fork [2] %29 {bb = 1 : ui32, name = #handshake.name<"fork4">} : i12
    %31 = arith.trunci %30#0 {bb = 1 : ui32, name = #handshake.name<"trunci0">} : i12 to i11
    %32 = arith.cmpi ult, %30#1, %19 {bb = 1 : ui32, name = #handshake.name<"cmpi0">} : i12
    %33:3 = fork [3] %32 {bb = 1 : ui32, name = #handshake.name<"fork5">} : i1
    %trueResult, %falseResult = cond_br %33#0, %31 {bb = 1 : ui32, name = #handshake.name<"cond_br0">} : i11
    sink %falseResult {name = #handshake.name<"sink0">} : i11
    %34 = buffer [3] fifo %33#1 {bb = 1 : ui32, name = #handshake.name<"buffer0">} : i1
    %trueResult_4, %falseResult_5 = cond_br %34, %27 {bb = 1 : ui32, name = #handshake.name<"cond_br2">} : i32
    %35 = buffer [2] seq %result {bb = 1 : ui32, name = #handshake.name<"buffer2">} : none
    %trueResult_6, %falseResult_7 = cond_br %33#2, %35 {bb = 1 : ui32, name = #handshake.name<"cond_br3">} : none
    sink %falseResult_7 {name = #handshake.name<"sink1">} : none
    %36 = buffer [1] seq %falseResult_5 {bb = 2 : ui32, name = #handshake.name<"buffer1">} : i32
    %37 = d_return {bb = 2 : ui32, name = #handshake.name<"d_return0">} %36 : i32
    end {bb = 2 : ui32, bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00}>, name = #handshake.name<"end0">} %37, %done, %done_1 : i32, none, none
  }
}

