module {
  handshake.func @polyn_mult(%arg0: memref<100xi32>, %arg1: memref<100xi32>, %arg2: memref<100xi32>, %arg3: none, ...) -> i32 attributes {argNames = ["a", "b", "out", "start"], resNames = ["out0"]} {
    %memOutputs:2, %done = lsq[%arg2 : memref<100xi32>] (%9#0, %addressResult, %dataResult, %51#1, %addressResult_18, %addressResult_20, %dataResult_21, %109#0, %addressResult_28, %addressResult_30, %dataResult_31)  {bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00, "10": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00, "11": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00, "5": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00, "6": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00, "7": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00, "8": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00, "9": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00}>, groupSizes = [1 : i32, 2 : i32, 2 : i32], name = #handshake.name<"lsq0">} : (none, i32, i32, none, i32, i32, i32, none, i32, i32, i32) -> (i32, i32, none)
    %memOutputs_0:2, %done_1 = mem_controller[%arg1 : memref<100xi32>] (%addressResult_16, %addressResult_26) {bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00}>, connectedBlocks = [3 : i32, 5 : i32], name = #handshake.name<"mem_controller0">} : (i32, i32) -> (i32, i32, none)
    %memOutputs_2:2, %done_3 = mem_controller[%arg0 : memref<100xi32>] (%addressResult_14, %addressResult_24) {bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00}>, connectedBlocks = [3 : i32, 5 : i32], name = #handshake.name<"mem_controller1">} : (i32, i32) -> (i32, i32, none)
    %0:2 = fork [2] %arg3 {bb = 0 : ui32, name = #handshake.name<"fork0">} : none
    %1 = constant %0#1 {bb = 0 : ui32, name = #handshake.name<"constant0">, value = false} : i1
    %2 = arith.extsi %1 {bb = 0 : ui32, name = #handshake.name<"extsi0">} : i1 to i8
    %3 = mux %index [%trueResult_44, %2] {bb = 1 : ui32, name = #handshake.name<"mux10">} : i1, i8
    %4 = buffer [1] seq %3 {bb = 1 : ui32, name = #handshake.name<"buffer23">} : i8
    %5:4 = fork [4] %4 {bb = 1 : ui32, name = #handshake.name<"fork1">} : i8
    %6 = arith.extsi %5#2 {bb = 1 : ui32, name = #handshake.name<"extsi9">} : i8 to i9
    %7 = arith.extsi %5#3 {bb = 1 : ui32, name = #handshake.name<"extsi10">} : i8 to i32
    %result, %index = control_merge %trueResult_46, %0#0 {bb = 1 : ui32, name = #handshake.name<"control_merge0">} : none, i1
    %8 = buffer [1] seq %result {bb = 1 : ui32, name = #handshake.name<"buffer12">} : none
    %9:4 = fork [4] %8 {bb = 1 : ui32, name = #handshake.name<"fork2">} : none
    %10 = constant %9#1 {bb = 1 : ui32, name = #handshake.name<"constant1">, value = false} : i1
    %11 = arith.extsi %10 {bb = 1 : ui32, name = #handshake.name<"extsi1">} : i1 to i32
    %12 = source {bb = 1 : ui32, name = #handshake.name<"source0">}
    %13 = constant %12 {bb = 1 : ui32, name = #handshake.name<"constant3">, value = 100 : i8} : i8
    %14 = arith.extsi %13 {bb = 1 : ui32, name = #handshake.name<"extsi11">} : i8 to i9
    %15 = constant %9#2 {bb = 1 : ui32, name = #handshake.name<"constant4">, value = 1 : i2} : i2
    %addressResult, %dataResult = lsq_store[%7] %11 {bb = 1 : ui32, name = #handshake.name<"lsq_store0">} : i32, i32
    %16 = arith.subi %14, %6 {bb = 1 : ui32, name = #handshake.name<"subi3">} : i9
    %17 = arith.extsi %15 {bb = 1 : ui32, name = #handshake.name<"extsi12">} : i2 to i10
    %18 = mux %29#0 [%67, %17] {bb = 2 : ui32, name = #handshake.name<"mux0">} : i1, i10
    %19 = buffer [1] seq %18 {bb = 2 : ui32, name = #handshake.name<"buffer2">} : i10
    %20:2 = fork [2] %19 {bb = 2 : ui32, name = #handshake.name<"fork3">} : i10
    %21 = arith.trunci %20#0 {bb = 2 : ui32, name = #handshake.name<"trunci0">} : i10 to i9
    %22 = mux %29#2 [%39#1, %5#1] {bb = 2 : ui32, name = #handshake.name<"mux1">} : i1, i8
    %23 = mux %29#1 [%44, %5#0] {bb = 2 : ui32, name = #handshake.name<"mux2">} : i1, i8
    %24 = buffer [1] fifo %29#3 {bb = 2 : ui32, name = #handshake.name<"buffer5">} : i1
    %25 = buffer [1] seq %16 {bb = 2 : ui32, name = #handshake.name<"buffer37">} : i9
    %26 = mux %24 [%68, %25] {bb = 2 : ui32, name = #handshake.name<"mux3">} : i1, i9
    %27:2 = fork [2] %26 {bb = 2 : ui32, name = #handshake.name<"fork4">} : i9
    %28 = arith.extsi %27#0 {bb = 2 : ui32, name = #handshake.name<"extsi2">} : i9 to i10
    %result_4, %index_5 = control_merge %51#0, %9#3 {bb = 2 : ui32, name = #handshake.name<"control_merge1">} : none, i1
    %29:4 = fork [4] %index_5 {bb = 2 : ui32, name = #handshake.name<"fork5">} : i1
    %30 = buffer [1] seq %28 {bb = 2 : ui32, name = #handshake.name<"buffer21">} : i10
    %31 = arith.cmpi ult, %20#1, %30 {bb = 2 : ui32, name = #handshake.name<"cmpi0">} : i10
    %32:5 = fork [5] %31 {bb = 2 : ui32, name = #handshake.name<"fork6">} : i1
    %33 = buffer [1] seq %22 {bb = 2 : ui32, name = #handshake.name<"buffer19">} : i8
    %trueResult, %falseResult = cond_br %32#3, %33 {bb = 2 : ui32, name = #handshake.name<"cond_br0">} : i8
    %34 = buffer [1] seq %23 {bb = 2 : ui32, name = #handshake.name<"buffer35">} : i8
    %trueResult_6, %falseResult_7 = cond_br %32#4, %34 {bb = 2 : ui32, name = #handshake.name<"cond_br1">} : i8
    %35 = buffer [1] seq %27#1 {bb = 2 : ui32, name = #handshake.name<"buffer3">} : i9
    %trueResult_8, %falseResult_9 = cond_br %32#2, %35 {bb = 2 : ui32, name = #handshake.name<"cond_br2">} : i9
    sink %falseResult_9 {name = #handshake.name<"sink0">} : i9
    %trueResult_10, %falseResult_11 = cond_br %32#0, %21 {bb = 2 : ui32, name = #handshake.name<"cond_br3">} : i9
    sink %falseResult_11 {name = #handshake.name<"sink1">} : i9
    %36 = buffer [1] seq %32#1 {bb = 2 : ui32, name = #handshake.name<"buffer22">} : i1
    %37 = buffer [2] seq %result_4 {bb = 2 : ui32, name = #handshake.name<"buffer29">} : none
    %trueResult_12, %falseResult_13 = cond_br %36, %37 {bb = 2 : ui32, name = #handshake.name<"cond_br7">} : none
    %38 = buffer [1] seq %trueResult {bb = 3 : ui32, name = #handshake.name<"buffer1">} : i8
    %39:3 = fork [3] %38 {bb = 3 : ui32, name = #handshake.name<"fork7">} : i8
    %40 = arith.extsi %39#0 {bb = 3 : ui32, name = #handshake.name<"extsi3">} : i8 to i10
    %41 = buffer [1] fifo %39#2 {bb = 3 : ui32, name = #handshake.name<"buffer27">} : i8
    %42 = arith.extsi %41 {bb = 3 : ui32, name = #handshake.name<"extsi4">} : i8 to i32
    %43:2 = fork [2] %42 {bb = 3 : ui32, name = #handshake.name<"fork8">} : i32
    %44 = buffer [1] seq %trueResult_6 {bb = 3 : ui32, name = #handshake.name<"buffer31">} : i8
    %45:3 = fork [3] %trueResult_10 {bb = 3 : ui32, name = #handshake.name<"fork9">} : i9
    %46 = arith.extsi %45#0 {bb = 3 : ui32, name = #handshake.name<"extsi13">} : i9 to i10
    %47 = buffer [1] seq %45#1 {bb = 3 : ui32, name = #handshake.name<"buffer6">} : i9
    %48 = arith.extsi %47 {bb = 3 : ui32, name = #handshake.name<"extsi5">} : i9 to i10
    %49 = buffer [1] seq %45#2 {bb = 3 : ui32, name = #handshake.name<"buffer17">} : i9
    %50 = arith.extsi %49 {bb = 3 : ui32, name = #handshake.name<"extsi14">} : i9 to i10
    %51:2 = fork [2] %trueResult_12 {bb = 3 : ui32, name = #handshake.name<"fork10">} : none
    %52 = source {bb = 3 : ui32, name = #handshake.name<"source1">}
    %53 = constant %52 {bb = 3 : ui32, name = #handshake.name<"constant14">, value = 100 : i8} : i8
    %54 = arith.extsi %53 {bb = 3 : ui32, name = #handshake.name<"extsi15">} : i8 to i10
    %55 = source {bb = 3 : ui32, name = #handshake.name<"source2">}
    %56 = constant %55 {bb = 3 : ui32, name = #handshake.name<"constant15">, value = 1 : i2} : i2
    %57 = arith.extsi %56 {bb = 3 : ui32, name = #handshake.name<"extsi6">} : i2 to i10
    %58 = arith.addi %40, %50 {bb = 3 : ui32, name = #handshake.name<"addi1">} : i10
    %59 = arith.extsi %58 {bb = 3 : ui32, name = #handshake.name<"extsi16">} : i10 to i32
    %addressResult_14, %dataResult_15 = mc_load[%59] %memOutputs_2#0 {bb = 3 : ui32, bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00}>, name = #handshake.name<"mc_load0">} : i32, i32
    %60 = arith.subi %54, %48 {bb = 3 : ui32, name = #handshake.name<"subi0">} : i10
    %61 = arith.extsi %60 {bb = 3 : ui32, name = #handshake.name<"extsi17">} : i10 to i32
    %addressResult_16, %dataResult_17 = mc_load[%61] %memOutputs_0#0 {bb = 3 : ui32, bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00}>, name = #handshake.name<"mc_load1">} : i32, i32
    %62 = arith.muli %dataResult_15, %dataResult_17 {bb = 3 : ui32, name = #handshake.name<"muli0">} : i32
    %addressResult_18, %dataResult_19 = lsq_load[%43#0] %memOutputs#0 {bb = 3 : ui32, bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00}>, name = #handshake.name<"lsq_load0">} : i32, i32
    %63 = arith.addi %dataResult_19, %62 {bb = 3 : ui32, name = #handshake.name<"addi0">} : i32
    %64 = buffer [3] fifo %43#1 {bb = 3 : ui32, name = #handshake.name<"buffer24">} : i32
    %addressResult_20, %dataResult_21 = lsq_store[%64] %63 {bb = 3 : ui32, name = #handshake.name<"lsq_store1">} : i32, i32
    %65 = buffer [1] seq %46 {bb = 3 : ui32, name = #handshake.name<"buffer8">} : i10
    %66 = buffer [1] seq %57 {bb = 3 : ui32, name = #handshake.name<"buffer16">} : i10
    %67 = arith.addi %65, %66 {bb = 3 : ui32, name = #handshake.name<"addi3">} : i10
    %68 = buffer [1] seq %trueResult_8 {bb = 3 : ui32, name = #handshake.name<"buffer13">} : i9
    %69 = buffer [1] seq %falseResult {bb = 4 : ui32, name = #handshake.name<"buffer39">} : i8
    %70:2 = fork [2] %69 {bb = 4 : ui32, name = #handshake.name<"fork11">} : i8
    %71 = arith.extsi %70#1 {bb = 4 : ui32, name = #handshake.name<"extsi7">} : i8 to i9
    %72 = buffer [1] seq %falseResult_7 {bb = 4 : ui32, name = #handshake.name<"buffer40">} : i8
    %73:2 = fork [2] %72 {bb = 4 : ui32, name = #handshake.name<"fork12">} : i8
    %74 = arith.extsi %73#1 {bb = 4 : ui32, name = #handshake.name<"extsi8">} : i8 to i9
    %75:2 = fork [2] %falseResult_13 {bb = 4 : ui32, name = #handshake.name<"fork13">} : none
    %76 = source {bb = 4 : ui32, name = #handshake.name<"source3">}
    %77 = constant %76 {bb = 4 : ui32, name = #handshake.name<"constant16">, value = 1 : i2} : i2
    %78:2 = fork [2] %77 {bb = 4 : ui32, name = #handshake.name<"fork14">} : i2
    %79 = arith.extsi %78#0 {bb = 4 : ui32, name = #handshake.name<"extsi18">} : i2 to i9
    %80 = arith.extsi %78#1 {bb = 4 : ui32, name = #handshake.name<"extsi28">} : i2 to i9
    %81 = constant %75#1 {bb = 4 : ui32, name = #handshake.name<"constant17">, value = false} : i1
    %82 = arith.addi %74, %79 {bb = 4 : ui32, name = #handshake.name<"addi4">} : i9
    %83 = arith.addi %71, %80 {bb = 4 : ui32, name = #handshake.name<"addi5">} : i9
    %84 = arith.extsi %81 {bb = 4 : ui32, name = #handshake.name<"extsi19">} : i1 to i9
    %85 = buffer [1] seq %82 {bb = 4 : ui32, name = #handshake.name<"buffer28">} : i9
    %86 = buffer [1] seq %83 {bb = 4 : ui32, name = #handshake.name<"buffer7">} : i9
    %87 = mux %108#0 [%trueResult_32, %84] {bb = 5 : ui32, name = #handshake.name<"mux4">} : i1, i9
    %88 = buffer [1] seq %87 {bb = 5 : ui32, name = #handshake.name<"buffer33">} : i9
    %89:3 = fork [3] %88 {bb = 5 : ui32, name = #handshake.name<"fork15">} : i9
    %90 = arith.extsi %89#0 {bb = 5 : ui32, name = #handshake.name<"extsi20">} : i9 to i10
    %91 = arith.extsi %89#1 {bb = 5 : ui32, name = #handshake.name<"extsi21">} : i9 to i10
    %92 = arith.extsi %89#2 {bb = 5 : ui32, name = #handshake.name<"extsi22">} : i9 to i32
    %93 = mux %108#3 [%trueResult_34, %70#0] {bb = 5 : ui32, name = #handshake.name<"mux5">} : i1, i8
    %94 = buffer [1] seq %93 {bb = 5 : ui32, name = #handshake.name<"buffer18">} : i8
    %95:3 = fork [3] %94 {bb = 5 : ui32, name = #handshake.name<"fork16">} : i8
    %96 = arith.extsi %95#0 {bb = 5 : ui32, name = #handshake.name<"extsi23">} : i8 to i10
    %97 = buffer [1] fifo %95#2 {bb = 5 : ui32, name = #handshake.name<"buffer14">} : i8
    %98 = arith.extsi %97 {bb = 5 : ui32, name = #handshake.name<"extsi24">} : i8 to i32
    %99:2 = fork [2] %98 {bb = 5 : ui32, name = #handshake.name<"fork17">} : i32
    %100 = mux %108#1 [%trueResult_36, %73#0] {bb = 5 : ui32, name = #handshake.name<"mux6">} : i1, i8
    %101 = buffer [1] fifo %108#2 {bb = 5 : ui32, name = #handshake.name<"buffer25">} : i1
    %102 = mux %101 [%trueResult_38, %85] {bb = 5 : ui32, name = #handshake.name<"mux7">} : i1, i9
    %103 = buffer [1] fifo %108#4 {bb = 5 : ui32, name = #handshake.name<"buffer4">} : i1
    %104 = mux %103 [%trueResult_40, %86] {bb = 5 : ui32, name = #handshake.name<"mux8">} : i1, i9
    %105 = buffer [2] seq %104 {bb = 5 : ui32, name = #handshake.name<"buffer15">} : i9
    %106:2 = fork [2] %105 {bb = 5 : ui32, name = #handshake.name<"fork18">} : i9
    %107 = arith.extsi %106#0 {bb = 5 : ui32, name = #handshake.name<"extsi29">} : i9 to i10
    %result_22, %index_23 = control_merge %trueResult_42, %75#0 {bb = 5 : ui32, name = #handshake.name<"control_merge2">} : none, i1
    %108:5 = fork [5] %index_23 {bb = 5 : ui32, name = #handshake.name<"fork19">} : i1
    %109:2 = fork [2] %result_22 {bb = 5 : ui32, name = #handshake.name<"fork20">} : none
    %110 = source {bb = 5 : ui32, name = #handshake.name<"source5">}
    %111 = constant %110 {bb = 5 : ui32, name = #handshake.name<"constant19">, value = 1 : i2} : i2
    %112 = arith.extsi %111 {bb = 5 : ui32, name = #handshake.name<"extsi30">} : i2 to i10
    %113 = arith.subi %96, %91 {bb = 5 : ui32, name = #handshake.name<"subi1">} : i10
    %114 = arith.extsi %113 {bb = 5 : ui32, name = #handshake.name<"extsi31">} : i10 to i32
    %addressResult_24, %dataResult_25 = mc_load[%114] %memOutputs_2#1 {bb = 5 : ui32, bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00}>, name = #handshake.name<"mc_load2">} : i32, i32
    %addressResult_26, %dataResult_27 = mc_load[%92] %memOutputs_0#1 {bb = 5 : ui32, bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00}>, name = #handshake.name<"mc_load3">} : i32, i32
    %115 = arith.muli %dataResult_25, %dataResult_27 {bb = 5 : ui32, name = #handshake.name<"muli1">} : i32
    %addressResult_28, %dataResult_29 = lsq_load[%99#0] %memOutputs#1 {bb = 5 : ui32, bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00}>, name = #handshake.name<"lsq_load1">} : i32, i32
    %116 = arith.addi %dataResult_29, %115 {bb = 5 : ui32, name = #handshake.name<"addi2">} : i32
    %117 = buffer [3] fifo %99#1 {bb = 5 : ui32, name = #handshake.name<"buffer34">} : i32
    %addressResult_30, %dataResult_31 = lsq_store[%117] %116 {bb = 5 : ui32, name = #handshake.name<"lsq_store2">} : i32, i32
    %118 = arith.addi %90, %112 {bb = 5 : ui32, name = #handshake.name<"addi6">} : i10
    %119 = buffer [1] seq %118 {bb = 5 : ui32, name = #handshake.name<"buffer9">} : i10
    %120:2 = fork [2] %119 {bb = 5 : ui32, name = #handshake.name<"fork21">} : i10
    %121 = arith.trunci %120#0 {bb = 5 : ui32, name = #handshake.name<"trunci1">} : i10 to i9
    %122 = arith.cmpi ult, %120#1, %107 {bb = 5 : ui32, name = #handshake.name<"cmpi1">} : i10
    %123:6 = fork [6] %122 {bb = 5 : ui32, name = #handshake.name<"fork22">} : i1
    %trueResult_32, %falseResult_33 = cond_br %123#0, %121 {bb = 5 : ui32, name = #handshake.name<"cond_br4">} : i9
    sink %falseResult_33 {name = #handshake.name<"sink2">} : i9
    %124 = buffer [1] fifo %95#1 {bb = 5 : ui32, name = #handshake.name<"buffer0">} : i8
    %trueResult_34, %falseResult_35 = cond_br %123#3, %124 {bb = 5 : ui32, name = #handshake.name<"cond_br5">} : i8
    %125 = buffer [2] seq %100 {bb = 5 : ui32, name = #handshake.name<"buffer36">} : i8
    %trueResult_36, %falseResult_37 = cond_br %123#5, %125 {bb = 5 : ui32, name = #handshake.name<"cond_br6">} : i8
    %126 = buffer [2] seq %102 {bb = 5 : ui32, name = #handshake.name<"buffer10">} : i9
    %trueResult_38, %falseResult_39 = cond_br %123#4, %126 {bb = 5 : ui32, name = #handshake.name<"cond_br8">} : i9
    %trueResult_40, %falseResult_41 = cond_br %123#2, %106#1 {bb = 5 : ui32, name = #handshake.name<"cond_br9">} : i9
    sink %falseResult_41 {name = #handshake.name<"sink3">} : i9
    %127 = buffer [2] seq %109#1 {bb = 5 : ui32, name = #handshake.name<"buffer32">} : none
    %trueResult_42, %falseResult_43 = cond_br %123#1, %127 {bb = 5 : ui32, name = #handshake.name<"cond_br16">} : none
    %128 = buffer [1] seq %falseResult_35 {bb = 6 : ui32, name = #handshake.name<"buffer41">} : i8
    %129 = arith.extsi %128 {bb = 6 : ui32, name = #handshake.name<"extsi32">} : i8 to i9
    %130 = buffer [1] seq %falseResult_37 {bb = 6 : ui32, name = #handshake.name<"buffer38">} : i8
    %131 = arith.extsi %130 {bb = 6 : ui32, name = #handshake.name<"extsi25">} : i8 to i10
    %132 = buffer [1] seq %falseResult_39 {bb = 6 : ui32, name = #handshake.name<"buffer11">} : i9
    %133 = arith.extsi %132 {bb = 6 : ui32, name = #handshake.name<"extsi33">} : i9 to i10
    %134 = buffer [1] seq %falseResult_43 {bb = 6 : ui32, name = #handshake.name<"buffer30">} : none
    %135 = source {bb = 6 : ui32, name = #handshake.name<"source6">}
    %136 = constant %135 {bb = 6 : ui32, name = #handshake.name<"constant20">, value = 100 : i8} : i8
    %137 = arith.extsi %136 {bb = 6 : ui32, name = #handshake.name<"extsi34">} : i8 to i9
    %138 = source {bb = 6 : ui32, name = #handshake.name<"source7">}
    %139 = constant %138 {bb = 6 : ui32, name = #handshake.name<"constant21">, value = 1 : i2} : i2
    %140 = arith.extsi %139 {bb = 6 : ui32, name = #handshake.name<"extsi26">} : i2 to i9
    %141 = arith.addi %133, %131 {bb = 6 : ui32, name = #handshake.name<"addi7">} : i10
    %142 = arith.addi %129, %140 {bb = 6 : ui32, name = #handshake.name<"addi8">} : i9
    %143 = buffer [1] seq %142 {bb = 6 : ui32, name = #handshake.name<"buffer26">} : i9
    %144:2 = fork [2] %143 {bb = 6 : ui32, name = #handshake.name<"fork23">} : i9
    %145 = arith.trunci %144#0 {bb = 6 : ui32, name = #handshake.name<"trunci2">} : i9 to i8
    %146 = arith.cmpi ult, %144#1, %137 {bb = 6 : ui32, name = #handshake.name<"cmpi2">} : i9
    %147:3 = fork [3] %146 {bb = 6 : ui32, name = #handshake.name<"fork24">} : i1
    %trueResult_44, %falseResult_45 = cond_br %147#0, %145 {bb = 6 : ui32, name = #handshake.name<"cond_br10">} : i8
    sink %falseResult_45 {name = #handshake.name<"sink4">} : i8
    %trueResult_46, %falseResult_47 = cond_br %147#2, %134 {bb = 6 : ui32, name = #handshake.name<"cond_br22">} : none
    sink %falseResult_47 {name = #handshake.name<"sink5">} : none
    %trueResult_48, %falseResult_49 = cond_br %147#1, %141 {bb = 6 : ui32, name = #handshake.name<"cond_br11">} : i10
    sink %trueResult_48 {name = #handshake.name<"sink6">} : i10
    %148 = buffer [1] seq %falseResult_49 {bb = 7 : ui32, name = #handshake.name<"buffer20">} : i10
    %149 = arith.extsi %148 {bb = 7 : ui32, name = #handshake.name<"extsi27">} : i10 to i32
    %150 = d_return {bb = 7 : ui32, name = #handshake.name<"d_return0">} %149 : i32
    end {bb = 7 : ui32, bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00}>, name = #handshake.name<"end0">} %150, %done, %done_1, %done_3 : i32, none, none, none
  }
}

