module {
  handshake.func @bicg(%arg0: memref<900xi32>, %arg1: memref<30xi32>, %arg2: memref<30xi32>, %arg3: memref<30xi32>, %arg4: memref<30xi32>, %arg5: none, ...) -> i32 attributes {argNames = ["a", "s", "q", "p", "r", "start"], resNames = ["out0"]} {
    %memOutputs, %done = mem_controller[%arg4 : memref<30xi32>] (%addressResult_14) {bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00}>, connectedBlocks = [2 : i32], name = #handshake.name<"mem_controller0">} : (i32) -> (i32, none)
    %memOutputs_0, %done_1 = mem_controller[%arg3 : memref<30xi32>] (%addressResult_18) {bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00}>, connectedBlocks = [2 : i32], name = #handshake.name<"mem_controller1">} : (i32) -> (i32, none)
    %memOutputs_2, %done_3 = lsq[%arg2 : memref<30xi32>] (%7#0, %addressResult, %addressResult_26, %dataResult_27)  {bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00}>, groupSizes = [2 : i32], name = #handshake.name<"lsq0">} : (none, i32, i32, i32) -> (i32, none)
    %memOutputs_4, %done_5 = lsq[%arg1 : memref<30xi32>] (%27#0, %addressResult_12, %addressResult_16, %dataResult_17)  {bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00}>, groupSizes = [2 : i32], name = #handshake.name<"lsq1">} : (none, i32, i32, i32) -> (i32, none)
    %memOutputs_6, %done_7 = mem_controller[%arg0 : memref<900xi32>] (%addressResult_10) {bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00}>, connectedBlocks = [2 : i32], name = #handshake.name<"mem_controller2">} : (i32) -> (i32, none)
    %0:2 = fork [2] %arg5 {bb = 0 : ui32, name = #handshake.name<"fork0">} : none
    %1 = constant %0#1 {bb = 0 : ui32, name = #handshake.name<"constant0">, value = false} : i1
    %2 = arith.extsi %1 {bb = 0 : ui32, name = #handshake.name<"extsi0">} : i1 to i6
    %3 = mux %index [%trueResult_28, %2] {bb = 1 : ui32, name = #handshake.name<"mux0">} : i1, i6
    %4 = buffer [1] seq %3 {bb = 1 : ui32, name = #handshake.name<"buffer12">} : i6
    %5:2 = fork [2] %4 {bb = 1 : ui32, name = #handshake.name<"fork1">} : i6
    %6 = arith.extsi %5#1 {bb = 1 : ui32, name = #handshake.name<"extsi6">} : i6 to i32
    %result, %index = control_merge %trueResult_30, %0#0 {bb = 1 : ui32, name = #handshake.name<"control_merge0">} : none, i1
    %7:3 = fork [3] %result {bb = 1 : ui32, name = #handshake.name<"fork2">} : none
    %8 = buffer [1] seq %7#1 {bb = 1 : ui32, name = #handshake.name<"buffer11">} : none
    %9 = constant %8 {bb = 1 : ui32, bufProps = #handshake<bufProps{"0": [0,inf], [1,inf], 0.000000e+00, 0.000000e+00, 0.000000e+00}>, name = #handshake.name<"constant1">, value = false} : i1
    %addressResult, %dataResult = lsq_load[%6] %memOutputs_2 {bb = 1 : ui32, bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00}>, name = #handshake.name<"lsq_load0">} : i32, i32
    %10 = arith.extsi %9 {bb = 1 : ui32, name = #handshake.name<"extsi1">} : i1 to i6
    %11 = buffer [1] seq %7#2 {bb = 1 : ui32, name = #handshake.name<"buffer0">} : none
    %12 = mux %26#1 [%trueResult, %10] {bb = 2 : ui32, name = #handshake.name<"mux4">} : i1, i6
    %13:3 = fork [3] %12 {bb = 2 : ui32, name = #handshake.name<"fork3">} : i6
    %14 = arith.extsi %13#0 {bb = 2 : ui32, name = #handshake.name<"extsi7">} : i6 to i13
    %15 = arith.extsi %13#1 {bb = 2 : ui32, name = #handshake.name<"extsi8">} : i6 to i7
    %16 = buffer [2] fifo %13#2 {bb = 2 : ui32, name = #handshake.name<"buffer4">} : i6
    %17 = arith.extsi %16 {bb = 2 : ui32, name = #handshake.name<"extsi9">} : i6 to i32
    %18:3 = fork [3] %17 {bb = 2 : ui32, name = #handshake.name<"fork4">} : i32
    %19 = buffer [4] fifo %26#2 {bb = 2 : ui32, name = #handshake.name<"buffer16">} : i1
    %20 = mux %19 [%trueResult_20, %dataResult] {bb = 2 : ui32, name = #handshake.name<"mux2">} : i1, i32
    %21 = mux %26#0 [%trueResult_22, %5#0] {bb = 2 : ui32, name = #handshake.name<"mux1">} : i1, i6
    %22:3 = fork [3] %21 {bb = 2 : ui32, name = #handshake.name<"fork5">} : i6
    %23 = arith.extsi %22#1 {bb = 2 : ui32, name = #handshake.name<"extsi10">} : i6 to i12
    %24 = buffer [2] fifo %22#2 {bb = 2 : ui32, name = #handshake.name<"buffer14">} : i6
    %25 = arith.extsi %24 {bb = 2 : ui32, name = #handshake.name<"extsi11">} : i6 to i32
    %result_8, %index_9 = control_merge %trueResult_24, %11 {bb = 2 : ui32, name = #handshake.name<"control_merge1">} : none, i1
    %26:3 = fork [3] %index_9 {bb = 2 : ui32, name = #handshake.name<"fork6">} : i1
    %27:2 = fork [2] %result_8 {bb = 2 : ui32, name = #handshake.name<"fork7">} : none
    %28 = source {bb = 2 : ui32, name = #handshake.name<"source0">}
    %29 = constant %28 {bb = 2 : ui32, name = #handshake.name<"constant6">, value = 30 : i6} : i6
    %30:2 = fork [2] %29 {bb = 2 : ui32, name = #handshake.name<"fork8">} : i6
    %31 = arith.extsi %30#0 {bb = 2 : ui32, name = #handshake.name<"extsi12">} : i6 to i12
    %32 = buffer [1] fifo %30#1 {bb = 2 : ui32, name = #handshake.name<"buffer5">} : i6
    %33 = arith.extsi %32 {bb = 2 : ui32, name = #handshake.name<"extsi13">} : i6 to i7
    %34 = source {bb = 2 : ui32, name = #handshake.name<"source1">}
    %35 = constant %34 {bb = 2 : ui32, name = #handshake.name<"constant7">, value = 1 : i2} : i2
    %36 = arith.extsi %35 {bb = 2 : ui32, name = #handshake.name<"extsi14">} : i2 to i7
    %37 = arith.muli %23, %31 {bb = 2 : ui32, name = #handshake.name<"muli2">} : i12
    %38 = arith.extsi %37 {bb = 2 : ui32, name = #handshake.name<"extsi15">} : i12 to i13
    %39 = buffer [3] seq %14 {bb = 2 : ui32, name = #handshake.name<"buffer21">} : i13
    %40 = arith.addi %39, %38 {bb = 2 : ui32, name = #handshake.name<"addi2">} : i13
    %41 = arith.extsi %40 {bb = 2 : ui32, name = #handshake.name<"extsi16">} : i13 to i32
    %addressResult_10, %dataResult_11 = mc_load[%41] %memOutputs_6 {bb = 2 : ui32, bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00}>, name = #handshake.name<"mc_load0">} : i32, i32
    %42:2 = fork [2] %dataResult_11 {bb = 2 : ui32, name = #handshake.name<"fork9">} : i32
    %43 = buffer [1] fifo %18#0 {bb = 2 : ui32, name = #handshake.name<"buffer13">} : i32
    %addressResult_12, %dataResult_13 = lsq_load[%43] %memOutputs_4 {bb = 2 : ui32, bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00}>, name = #handshake.name<"lsq_load1">} : i32, i32
    %addressResult_14, %dataResult_15 = mc_load[%25] %memOutputs {bb = 2 : ui32, bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00}>, name = #handshake.name<"mc_load1">} : i32, i32
    %44 = arith.muli %dataResult_15, %42#1 {bb = 2 : ui32, name = #handshake.name<"muli0">} : i32
    %45 = arith.addi %dataResult_13, %44 {bb = 2 : ui32, name = #handshake.name<"addi0">} : i32
    %addressResult_16, %dataResult_17 = lsq_store[%18#1] %45 {bb = 2 : ui32, name = #handshake.name<"lsq_store0">} : i32, i32
    %addressResult_18, %dataResult_19 = mc_load[%18#2] %memOutputs_0 {bb = 2 : ui32, bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00}>, name = #handshake.name<"mc_load2">} : i32, i32
    %46 = arith.muli %42#0, %dataResult_19 {bb = 2 : ui32, name = #handshake.name<"muli1">} : i32
    %47 = buffer [2] seq %20 {bb = 2 : ui32, name = #handshake.name<"buffer8">} : i32
    %48 = arith.addi %47, %46 {bb = 2 : ui32, name = #handshake.name<"addi1">} : i32
    %49 = buffer [1] seq %36 {bb = 2 : ui32, name = #handshake.name<"buffer1">} : i7
    %50 = buffer [1] seq %15 {bb = 2 : ui32, name = #handshake.name<"buffer2">} : i7
    %51 = arith.addi %50, %49 {bb = 2 : ui32, name = #handshake.name<"addi5">} : i7
    %52:2 = fork [2] %51 {bb = 2 : ui32, name = #handshake.name<"fork10">} : i7
    %53 = buffer [1] seq %52#0 {bb = 2 : ui32, name = #handshake.name<"buffer19">} : i7
    %54 = arith.trunci %53 {bb = 2 : ui32, name = #handshake.name<"trunci0">} : i7 to i6
    %55 = arith.cmpi ult, %52#1, %33 {bb = 2 : ui32, name = #handshake.name<"cmpi0">} : i7
    %56 = buffer [1] seq %55 {bb = 2 : ui32, name = #handshake.name<"buffer17">} : i1
    %57:4 = fork [4] %56 {bb = 2 : ui32, name = #handshake.name<"fork11">} : i1
    %trueResult, %falseResult = cond_br %57#0, %54 {bb = 2 : ui32, name = #handshake.name<"cond_br0">} : i6
    sink %falseResult {name = #handshake.name<"sink0">} : i6
    %58 = buffer [4] fifo %57#2 {bb = 2 : ui32, name = #handshake.name<"buffer9">} : i1
    %trueResult_20, %falseResult_21 = cond_br %58, %48 {bb = 2 : ui32, name = #handshake.name<"cond_br3">} : i32
    %59 = buffer [2] seq %22#0 {bb = 2 : ui32, name = #handshake.name<"buffer18">} : i6
    %trueResult_22, %falseResult_23 = cond_br %57#1, %59 {bb = 2 : ui32, name = #handshake.name<"cond_br1">} : i6
    %60 = buffer [2] seq %27#1 {bb = 2 : ui32, name = #handshake.name<"buffer3">} : none
    %61 = buffer [1] fifo %57#3 {bb = 2 : ui32, name = #handshake.name<"buffer15">} : i1
    %trueResult_24, %falseResult_25 = cond_br %61, %60 {bb = 2 : ui32, bufProps = #handshake<bufProps{"1": [0,inf], [1,inf], 0.000000e+00, 0.000000e+00, 0.000000e+00}>, name = #handshake.name<"cond_br5">} : none
    %62:2 = fork [2] %falseResult_23 {bb = 3 : ui32, name = #handshake.name<"fork12">} : i6
    %63 = arith.extsi %62#0 {bb = 3 : ui32, name = #handshake.name<"extsi2">} : i6 to i7
    %64 = arith.extsi %62#1 {bb = 3 : ui32, name = #handshake.name<"extsi17">} : i6 to i32
    %65:2 = fork [2] %falseResult_21 {bb = 3 : ui32, name = #handshake.name<"fork13">} : i32
    %66 = source {bb = 3 : ui32, name = #handshake.name<"source2">}
    %67 = constant %66 {bb = 3 : ui32, name = #handshake.name<"constant8">, value = 30 : i6} : i6
    %68 = arith.extsi %67 {bb = 3 : ui32, name = #handshake.name<"extsi3">} : i6 to i7
    %69 = source {bb = 3 : ui32, name = #handshake.name<"source3">}
    %70 = constant %69 {bb = 3 : ui32, name = #handshake.name<"constant9">, value = 1 : i2} : i2
    %71 = arith.extsi %70 {bb = 3 : ui32, name = #handshake.name<"extsi18">} : i2 to i7
    %addressResult_26, %dataResult_27 = lsq_store[%64] %65#1 {bb = 3 : ui32, name = #handshake.name<"lsq_store1">} : i32, i32
    %72 = arith.addi %63, %71 {bb = 3 : ui32, name = #handshake.name<"addi3">} : i7
    %73 = buffer [1] seq %72 {bb = 3 : ui32, name = #handshake.name<"buffer20">} : i7
    %74:2 = fork [2] %73 {bb = 3 : ui32, name = #handshake.name<"fork14">} : i7
    %75 = arith.trunci %74#0 {bb = 3 : ui32, name = #handshake.name<"trunci1">} : i7 to i6
    %76 = buffer [1] seq %68 {bb = 3 : ui32, name = #handshake.name<"buffer10">} : i7
    %77 = arith.cmpi ult, %74#1, %76 {bb = 3 : ui32, name = #handshake.name<"cmpi1">} : i7
    %78:3 = fork [3] %77 {bb = 3 : ui32, name = #handshake.name<"fork15">} : i1
    %trueResult_28, %falseResult_29 = cond_br %78#0, %75 {bb = 3 : ui32, name = #handshake.name<"cond_br2">} : i6
    sink %falseResult_29 {name = #handshake.name<"sink1">} : i6
    %trueResult_30, %falseResult_31 = cond_br %78#1, %falseResult_25 {bb = 3 : ui32, name = #handshake.name<"cond_br10">} : none
    sink %falseResult_31 {name = #handshake.name<"sink2">} : none
    %79 = buffer [2] fifo %78#2 {bb = 3 : ui32, name = #handshake.name<"buffer7">} : i1
    %trueResult_32, %falseResult_33 = cond_br %79, %65#0 {bb = 3 : ui32, name = #handshake.name<"cond_br11">} : i32
    sink %trueResult_32 {name = #handshake.name<"sink3">} : i32
    %80 = buffer [1] seq %falseResult_33 {bb = 4 : ui32, name = #handshake.name<"buffer6">} : i32
    %81 = d_return {bb = 4 : ui32, name = #handshake.name<"d_return0">} %80 : i32
    end {bb = 4 : ui32, bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00, "5": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00}>, name = #handshake.name<"end0">} %81, %done, %done_1, %done_3, %done_5, %done_7 : i32, none, none, none, none, none
  }
}

