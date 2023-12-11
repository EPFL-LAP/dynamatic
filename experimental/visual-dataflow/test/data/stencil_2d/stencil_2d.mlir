module {
  handshake.func @stencil_2d(%arg0: memref<900xi32>, %arg1: memref<10xi32>, %arg2: memref<900xi32>, %arg3: none, ...) -> i32 attributes {argNames = ["orig", "filter", "sol", "start"], resNames = ["out0"]} {
    %done = mem_controller[%arg2 : memref<900xi32>] (%108, %addressResult_25, %dataResult_26) {bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00}>, connectedBlocks = [5 : i32], name = #handshake.name<"mem_controller0">} : (i32, i32, i32) -> none
    %memOutputs, %done_0 = mem_controller[%arg1 : memref<10xi32>] (%addressResult) {bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00}>, connectedBlocks = [3 : i32], name = #handshake.name<"mem_controller1">} : (i32) -> (i32, none)
    %memOutputs_1, %done_2 = mem_controller[%arg0 : memref<900xi32>] (%addressResult_7) {bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00}>, connectedBlocks = [3 : i32], name = #handshake.name<"mem_controller2">} : (i32) -> (i32, none)
    %0:2 = fork [2] %arg3 {bb = 0 : ui32, name = #handshake.name<"fork0">} : none
    %1 = constant %0#1 {bb = 0 : ui32, name = #handshake.name<"constant0">, value = false} : i1
    %2 = arith.extsi %1 {bb = 0 : ui32, name = #handshake.name<"extsi0">} : i1 to i6
    %3 = mux %index [%trueResult_27, %2] {bb = 1 : ui32, name = #handshake.name<"mux0">} : i1, i6
    %result, %index = control_merge %trueResult_29, %0#0 {bb = 1 : ui32, name = #handshake.name<"control_merge0">} : none, i1
    %4 = buffer [1] seq %result {bb = 1 : ui32, name = #handshake.name<"buffer7">} : none
    %5:2 = fork [2] %4 {bb = 1 : ui32, name = #handshake.name<"fork1">} : none
    %6 = constant %5#0 {bb = 1 : ui32, name = #handshake.name<"constant1">, value = false} : i1
    %7:2 = fork [2] %6 {bb = 1 : ui32, name = #handshake.name<"fork2">} : i1
    %8 = arith.extsi %7#0 {bb = 1 : ui32, name = #handshake.name<"extsi1">} : i1 to i3
    %9 = arith.extsi %7#1 {bb = 1 : ui32, name = #handshake.name<"extsi10">} : i1 to i32
    %10 = mux %15#1 [%trueResult_17, %8] {bb = 2 : ui32, name = #handshake.name<"mux1">} : i1, i3
    %11 = buffer [2] fifo %15#2 {bb = 2 : ui32, name = #handshake.name<"buffer8">} : i1
    %12 = mux %11 [%trueResult_19, %9] {bb = 2 : ui32, name = #handshake.name<"mux2">} : i1, i32
    %13 = buffer [1] seq %3 {bb = 2 : ui32, name = #handshake.name<"buffer14">} : i6
    %14 = mux %15#0 [%trueResult_21, %13] {bb = 2 : ui32, name = #handshake.name<"mux3">} : i1, i6
    %result_3, %index_4 = control_merge %trueResult_23, %5#1 {bb = 2 : ui32, name = #handshake.name<"control_merge1">} : none, i1
    %15:3 = fork [3] %index_4 {bb = 2 : ui32, name = #handshake.name<"fork3">} : i1
    %16:2 = fork [2] %result_3 {bb = 2 : ui32, name = #handshake.name<"fork4">} : none
    %17 = constant %16#0 {bb = 2 : ui32, name = #handshake.name<"constant4">, value = false} : i1
    %18 = arith.extsi %17 {bb = 2 : ui32, name = #handshake.name<"extsi2">} : i1 to i3
    %19 = mux %39#2 [%trueResult, %18] {bb = 3 : ui32, name = #handshake.name<"mux8">} : i1, i3
    %20 = buffer [1] seq %19 {bb = 3 : ui32, name = #handshake.name<"buffer10">} : i3
    %21:3 = fork [3] %20 {bb = 3 : ui32, name = #handshake.name<"fork5">} : i3
    %22 = buffer [1] fifo %21#0 {bb = 3 : ui32, name = #handshake.name<"buffer0">} : i3
    %23 = arith.extsi %22 {bb = 3 : ui32, name = #handshake.name<"extsi11">} : i3 to i7
    %24 = buffer [2] fifo %21#1 {bb = 3 : ui32, name = #handshake.name<"buffer6">} : i3
    %25 = arith.extsi %24 {bb = 3 : ui32, name = #handshake.name<"extsi12">} : i3 to i6
    %26 = arith.extsi %21#2 {bb = 3 : ui32, name = #handshake.name<"extsi13">} : i3 to i4
    %27 = buffer [5] fifo %39#3 {bb = 3 : ui32, name = #handshake.name<"buffer32">} : i1
    %28 = mux %27 [%trueResult_9, %12] {bb = 3 : ui32, name = #handshake.name<"mux5">} : i1, i32
    %29 = mux %39#0 [%trueResult_11, %14] {bb = 3 : ui32, name = #handshake.name<"mux4">} : i1, i6
    %30 = buffer [2] seq %29 {bb = 3 : ui32, name = #handshake.name<"buffer20">} : i6
    %31:2 = fork [2] %30 {bb = 3 : ui32, name = #handshake.name<"fork6">} : i6
    %32 = arith.extsi %31#1 {bb = 3 : ui32, name = #handshake.name<"extsi14">} : i6 to i7
    %33 = mux %39#1 [%trueResult_13, %10] {bb = 3 : ui32, name = #handshake.name<"mux6">} : i1, i3
    %34 = buffer [1] seq %33 {bb = 3 : ui32, name = #handshake.name<"buffer22">} : i3
    %35:4 = fork [4] %34 {bb = 3 : ui32, name = #handshake.name<"fork7">} : i3
    %36 = arith.extsi %35#1 {bb = 3 : ui32, name = #handshake.name<"extsi15">} : i3 to i9
    %37 = arith.extsi %35#2 {bb = 3 : ui32, name = #handshake.name<"extsi3">} : i3 to i5
    %38 = arith.extsi %35#3 {bb = 3 : ui32, name = #handshake.name<"extsi16">} : i3 to i4
    %result_5, %index_6 = control_merge %trueResult_15, %16#1 {bb = 3 : ui32, name = #handshake.name<"control_merge2">} : none, i1
    %39:4 = fork [4] %index_6 {bb = 3 : ui32, name = #handshake.name<"fork8">} : i1
    %40 = source {bb = 3 : ui32, name = #handshake.name<"source0">}
    %41 = constant %40 {bb = 3 : ui32, name = #handshake.name<"constant5">, value = 30 : i6} : i6
    %42 = arith.extsi %41 {bb = 3 : ui32, name = #handshake.name<"extsi17">} : i6 to i9
    %43 = source {bb = 3 : ui32, name = #handshake.name<"source1">}
    %44 = constant %43 {bb = 3 : ui32, name = #handshake.name<"constant12">, value = 3 : i3} : i3
    %45 = arith.extsi %44 {bb = 3 : ui32, name = #handshake.name<"extsi18">} : i3 to i4
    %46 = source {bb = 3 : ui32, name = #handshake.name<"source2">}
    %47 = constant %46 {bb = 3 : ui32, name = #handshake.name<"constant13">, value = 1 : i2} : i2
    %48:2 = fork [2] %47 {bb = 3 : ui32, name = #handshake.name<"fork9">} : i2
    %49 = arith.extui %48#0 {bb = 3 : ui32, name = #handshake.name<"extui0">} : i2 to i4
    %50 = arith.extsi %48#1 {bb = 3 : ui32, name = #handshake.name<"extsi19">} : i2 to i4
    %51 = arith.shli %38, %49 {bb = 3 : ui32, name = #handshake.name<"shli0">} : i4
    %52 = buffer [1] seq %51 {bb = 3 : ui32, name = #handshake.name<"buffer19">} : i4
    %53 = arith.extsi %52 {bb = 3 : ui32, name = #handshake.name<"extsi20">} : i4 to i5
    %54 = buffer [1] fifo %37 {bb = 3 : ui32, name = #handshake.name<"buffer17">} : i5
    %55 = arith.addi %54, %53 {bb = 3 : ui32, name = #handshake.name<"addi1">} : i5
    %56 = arith.extsi %55 {bb = 3 : ui32, name = #handshake.name<"extsi21">} : i5 to i6
    %57 = buffer [2] seq %56 {bb = 3 : ui32, name = #handshake.name<"buffer15">} : i6
    %58 = arith.addi %25, %57 {bb = 3 : ui32, name = #handshake.name<"addi2">} : i6
    %59 = arith.extsi %58 {bb = 3 : ui32, name = #handshake.name<"extsi22">} : i6 to i32
    %addressResult, %dataResult = mc_load[%59] %memOutputs {bb = 3 : ui32, bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00}>, name = #handshake.name<"mc_load0">} : i32, i32
    %60 = arith.addi %23, %32 {bb = 3 : ui32, name = #handshake.name<"addi3">} : i7
    %61 = arith.extsi %60 {bb = 3 : ui32, name = #handshake.name<"extsi23">} : i7 to i10
    %62 = arith.muli %36, %42 {bb = 3 : ui32, name = #handshake.name<"muli1">} : i9
    %63 = arith.extsi %62 {bb = 3 : ui32, name = #handshake.name<"extsi24">} : i9 to i10
    %64 = buffer [2] seq %61 {bb = 3 : ui32, name = #handshake.name<"buffer28">} : i10
    %65 = arith.addi %64, %63 {bb = 3 : ui32, name = #handshake.name<"addi4">} : i10
    %66 = arith.extsi %65 {bb = 3 : ui32, name = #handshake.name<"extsi25">} : i10 to i32
    %addressResult_7, %dataResult_8 = mc_load[%66] %memOutputs_1 {bb = 3 : ui32, bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00}>, name = #handshake.name<"mc_load1">} : i32, i32
    %67 = arith.muli %dataResult, %dataResult_8 {bb = 3 : ui32, name = #handshake.name<"muli0">} : i32
    %68 = buffer [2] seq %28 {bb = 3 : ui32, name = #handshake.name<"buffer27">} : i32
    %69 = arith.addi %68, %67 {bb = 3 : ui32, name = #handshake.name<"addi0">} : i32
    %70 = arith.addi %26, %50 {bb = 3 : ui32, name = #handshake.name<"addi8">} : i4
    %71 = buffer [1] seq %70 {bb = 3 : ui32, name = #handshake.name<"buffer12">} : i4
    %72:2 = fork [2] %71 {bb = 3 : ui32, name = #handshake.name<"fork10">} : i4
    %73 = arith.trunci %72#0 {bb = 3 : ui32, name = #handshake.name<"trunci0">} : i4 to i3
    %74 = buffer [1] seq %45 {bb = 3 : ui32, name = #handshake.name<"buffer11">} : i4
    %75 = arith.cmpi ult, %72#1, %74 {bb = 3 : ui32, name = #handshake.name<"cmpi0">} : i4
    %76:5 = fork [5] %75 {bb = 3 : ui32, name = #handshake.name<"fork11">} : i1
    %trueResult, %falseResult = cond_br %76#0, %73 {bb = 3 : ui32, name = #handshake.name<"cond_br0">} : i3
    sink %falseResult {name = #handshake.name<"sink0">} : i3
    %77 = buffer [5] fifo %76#3 {bb = 3 : ui32, name = #handshake.name<"buffer13">} : i1
    %trueResult_9, %falseResult_10 = cond_br %77, %69 {bb = 3 : ui32, name = #handshake.name<"cond_br4">} : i32
    %trueResult_11, %falseResult_12 = cond_br %76#1, %31#0 {bb = 3 : ui32, name = #handshake.name<"cond_br1">} : i6
    %78 = buffer [1] fifo %35#0 {bb = 3 : ui32, name = #handshake.name<"buffer31">} : i3
    %trueResult_13, %falseResult_14 = cond_br %76#2, %78 {bb = 3 : ui32, name = #handshake.name<"cond_br2">} : i3
    %79 = buffer [2] seq %result_5 {bb = 3 : ui32, name = #handshake.name<"buffer25">} : none
    %trueResult_15, %falseResult_16 = cond_br %76#4, %79 {bb = 3 : ui32, name = #handshake.name<"cond_br7">} : none
    %80 = arith.extsi %falseResult_14 {bb = 4 : ui32, name = #handshake.name<"extsi4">} : i3 to i4
    %81 = source {bb = 4 : ui32, name = #handshake.name<"source3">}
    %82 = constant %81 {bb = 4 : ui32, name = #handshake.name<"constant14">, value = 3 : i3} : i3
    %83 = arith.extsi %82 {bb = 4 : ui32, name = #handshake.name<"extsi5">} : i3 to i4
    %84 = source {bb = 4 : ui32, name = #handshake.name<"source4">}
    %85 = constant %84 {bb = 4 : ui32, name = #handshake.name<"constant15">, value = 1 : i2} : i2
    %86 = arith.extsi %85 {bb = 4 : ui32, name = #handshake.name<"extsi26">} : i2 to i4
    %87 = buffer [1] seq %80 {bb = 4 : ui32, name = #handshake.name<"buffer16">} : i4
    %88 = buffer [1] seq %86 {bb = 4 : ui32, name = #handshake.name<"buffer18">} : i4
    %89 = arith.addi %87, %88 {bb = 4 : ui32, name = #handshake.name<"addi5">} : i4
    %90:2 = fork [2] %89 {bb = 4 : ui32, name = #handshake.name<"fork12">} : i4
    %91 = buffer [1] seq %90#0 {bb = 4 : ui32, name = #handshake.name<"buffer3">} : i4
    %92 = arith.trunci %91 {bb = 4 : ui32, name = #handshake.name<"trunci1">} : i4 to i3
    %93 = arith.cmpi ult, %90#1, %83 {bb = 4 : ui32, name = #handshake.name<"cmpi1">} : i4
    %94 = buffer [1] seq %93 {bb = 4 : ui32, name = #handshake.name<"buffer21">} : i1
    %95:4 = fork [4] %94 {bb = 4 : ui32, name = #handshake.name<"fork13">} : i1
    %trueResult_17, %falseResult_18 = cond_br %95#0, %92 {bb = 4 : ui32, name = #handshake.name<"cond_br3">} : i3
    sink %falseResult_18 {name = #handshake.name<"sink1">} : i3
    %96 = buffer [1] seq %falseResult_10 {bb = 4 : ui32, name = #handshake.name<"buffer23">} : i32
    %97 = buffer [2] fifo %95#2 {bb = 4 : ui32, name = #handshake.name<"buffer29">} : i1
    %trueResult_19, %falseResult_20 = cond_br %97, %96 {bb = 4 : ui32, name = #handshake.name<"cond_br13">} : i32
    %98 = buffer [1] seq %falseResult_12 {bb = 4 : ui32, name = #handshake.name<"buffer5">} : i6
    %trueResult_21, %falseResult_22 = cond_br %95#1, %98 {bb = 4 : ui32, name = #handshake.name<"cond_br5">} : i6
    %99 = buffer [1] seq %falseResult_16 {bb = 4 : ui32, name = #handshake.name<"buffer2">} : none
    %trueResult_23, %falseResult_24 = cond_br %95#3, %99 {bb = 4 : ui32, name = #handshake.name<"cond_br15">} : none
    %100:2 = fork [2] %falseResult_22 {bb = 5 : ui32, name = #handshake.name<"fork14">} : i6
    %101 = arith.extsi %100#0 {bb = 5 : ui32, name = #handshake.name<"extsi6">} : i6 to i7
    %102 = buffer [2] fifo %100#1 {bb = 5 : ui32, name = #handshake.name<"buffer24">} : i6
    %103 = arith.extsi %102 {bb = 5 : ui32, name = #handshake.name<"extsi27">} : i6 to i32
    %104:2 = fork [2] %falseResult_20 {bb = 5 : ui32, name = #handshake.name<"fork15">} : i32
    %105 = buffer [1] fifo %falseResult_24 {bb = 5 : ui32, name = #handshake.name<"buffer30">} : none
    %106:2 = fork [2] %105 {bb = 5 : ui32, name = #handshake.name<"fork16">} : none
    %107 = constant %106#1 {bb = 5 : ui32, name = #handshake.name<"constant16">, value = 1 : i2} : i2
    %108 = arith.extsi %107 {bb = 5 : ui32, name = #handshake.name<"extsi7">} : i2 to i32
    %109 = source {bb = 5 : ui32, name = #handshake.name<"source5">}
    %110 = constant %109 {bb = 5 : ui32, name = #handshake.name<"constant17">, value = 28 : i6} : i6
    %111 = arith.extsi %110 {bb = 5 : ui32, name = #handshake.name<"extsi8">} : i6 to i7
    %112 = source {bb = 5 : ui32, name = #handshake.name<"source6">}
    %113 = constant %112 {bb = 5 : ui32, name = #handshake.name<"constant18">, value = 1 : i2} : i2
    %114 = arith.extsi %113 {bb = 5 : ui32, name = #handshake.name<"extsi28">} : i2 to i7
    %addressResult_25, %dataResult_26 = mc_store[%103] %104#1 {bb = 5 : ui32, name = #handshake.name<"mc_store0">} : i32, i32
    %115 = arith.addi %101, %114 {bb = 5 : ui32, name = #handshake.name<"addi6">} : i7
    %116 = buffer [1] seq %115 {bb = 5 : ui32, name = #handshake.name<"buffer26">} : i7
    %117:2 = fork [2] %116 {bb = 5 : ui32, name = #handshake.name<"fork17">} : i7
    %118 = arith.trunci %117#0 {bb = 5 : ui32, name = #handshake.name<"trunci2">} : i7 to i6
    %119 = buffer [1] seq %111 {bb = 5 : ui32, name = #handshake.name<"buffer1">} : i7
    %120 = arith.cmpi ult, %117#1, %119 {bb = 5 : ui32, name = #handshake.name<"cmpi2">} : i7
    %121:3 = fork [3] %120 {bb = 5 : ui32, name = #handshake.name<"fork18">} : i1
    %trueResult_27, %falseResult_28 = cond_br %121#0, %118 {bb = 5 : ui32, name = #handshake.name<"cond_br6">} : i6
    sink %falseResult_28 {name = #handshake.name<"sink2">} : i6
    %trueResult_29, %falseResult_30 = cond_br %121#1, %106#0 {bb = 5 : ui32, name = #handshake.name<"cond_br20">} : none
    sink %falseResult_30 {name = #handshake.name<"sink3">} : none
    %122 = buffer [2] fifo %121#2 {bb = 5 : ui32, name = #handshake.name<"buffer4">} : i1
    %trueResult_31, %falseResult_32 = cond_br %122, %104#0 {bb = 5 : ui32, name = #handshake.name<"cond_br21">} : i32
    sink %trueResult_31 {name = #handshake.name<"sink4">} : i32
    %123 = buffer [1] seq %falseResult_32 {bb = 6 : ui32, name = #handshake.name<"buffer9">} : i32
    %124 = d_return {bb = 6 : ui32, name = #handshake.name<"d_return0">} %123 : i32
    end {bb = 6 : ui32, bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00}>, name = #handshake.name<"end0">} %124, %done, %done_0, %done_2 : i32, none, none, none
  }
}

