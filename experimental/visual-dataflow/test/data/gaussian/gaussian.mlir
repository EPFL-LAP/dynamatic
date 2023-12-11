module {
  handshake.func @gaussian(%arg0: memref<20xi32>, %arg1: memref<400xi32>, %arg2: none, ...) -> i32 attributes {argNames = ["c", "a", "start"], resNames = ["out0"]} {
    %memOutputs:2, %done = lsq[%arg1 : memref<400xi32>] (%64#0, %addressResult, %addressResult_16, %addressResult_18, %dataResult_19)  {bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00, "5": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00}>, groupSizes = [3 : i32], name = #handshake.name<"lsq0">} : (none, i32, i32, i32, i32) -> (i32, i32, none)
    %memOutputs_0, %done_1 = mem_controller[%arg0 : memref<20xi32>] (%addressResult_14) {bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00}>, connectedBlocks = [3 : i32], name = #handshake.name<"mem_controller0">} : (i32) -> (i32, none)
    %0:3 = fork [3] %arg2 {bb = 0 : ui32, name = #handshake.name<"fork0">} : none
    %1 = constant %0#2 {bb = 0 : ui32, name = #handshake.name<"constant0">, value = 1 : i2} : i2
    %2 = constant %0#1 {bb = 0 : ui32, name = #handshake.name<"constant1">, value = false} : i1
    %3 = arith.extsi %1 {bb = 0 : ui32, name = #handshake.name<"extsi0">} : i2 to i6
    %4 = arith.extsi %2 {bb = 0 : ui32, name = #handshake.name<"extsi1">} : i1 to i32
    %5 = mux %10#0 [%trueResult_30, %3] {bb = 1 : ui32, name = #handshake.name<"mux0">} : i1, i6
    %6 = buffer [1] seq %5 {bb = 1 : ui32, name = #handshake.name<"buffer11">} : i6
    %7:2 = fork [2] %6 {bb = 1 : ui32, name = #handshake.name<"fork1">} : i6
    %8 = arith.extsi %7#1 {bb = 1 : ui32, name = #handshake.name<"extsi11">} : i6 to i7
    %9 = mux %10#1 [%trueResult_32, %4] {bb = 1 : ui32, name = #handshake.name<"mux1">} : i1, i32
    %result, %index = control_merge %trueResult_34, %0#0 {bb = 1 : ui32, name = #handshake.name<"control_merge0">} : none, i1
    %10:2 = fork [2] %index {bb = 1 : ui32, name = #handshake.name<"fork2">} : i1
    %11 = source {bb = 1 : ui32, name = #handshake.name<"source0">}
    %12 = constant %11 {bb = 1 : ui32, name = #handshake.name<"constant2">, value = 1 : i2} : i2
    %13 = arith.extsi %12 {bb = 1 : ui32, name = #handshake.name<"extsi12">} : i2 to i7
    %14 = arith.addi %8, %13 {bb = 1 : ui32, name = #handshake.name<"addi1">} : i7
    %15 = buffer [1] seq %14 {bb = 1 : ui32, name = #handshake.name<"buffer26">} : i7
    %16 = buffer [1] seq %9 {bb = 1 : ui32, name = #handshake.name<"buffer35">} : i32
    %17 = buffer [1] seq %26#1 {bb = 2 : ui32, name = #handshake.name<"buffer28">} : i1
    %18 = buffer [1] seq %138 {bb = 2 : ui32, name = #handshake.name<"buffer32">} : i7
    %19 = mux %17 [%18, %15] {bb = 2 : ui32, name = #handshake.name<"mux2">} : i1, i7
    %20:2 = fork [2] %19 {bb = 2 : ui32, name = #handshake.name<"fork3">} : i7
    %21 = buffer [1] fifo %20#0 {bb = 2 : ui32, name = #handshake.name<"buffer38">} : i7
    %22 = arith.trunci %21 {bb = 2 : ui32, name = #handshake.name<"trunci0">} : i7 to i6
    %23 = mux %26#2 [%falseResult_23, %16] {bb = 2 : ui32, name = #handshake.name<"mux3">} : i1, i32
    %24 = mux %26#0 [%falseResult_25, %7#0] {bb = 2 : ui32, name = #handshake.name<"mux4">} : i1, i6
    %25 = buffer [1] fifo %result {bb = 2 : ui32, name = #handshake.name<"buffer17">} : none
    %result_2, %index_3 = control_merge %falseResult_29, %25 {bb = 2 : ui32, name = #handshake.name<"control_merge1">} : none, i1
    %26:3 = fork [3] %index_3 {bb = 2 : ui32, name = #handshake.name<"fork4">} : i1
    %27 = buffer [1] seq %result_2 {bb = 2 : ui32, name = #handshake.name<"buffer19">} : none
    %28:2 = fork [2] %27 {bb = 2 : ui32, name = #handshake.name<"fork5">} : none
    %29 = constant %28#0 {bb = 2 : ui32, name = #handshake.name<"constant5">, value = 1 : i2} : i2
    %30 = source {bb = 2 : ui32, name = #handshake.name<"source1">}
    %31 = constant %30 {bb = 2 : ui32, name = #handshake.name<"constant13">, value = 19 : i6} : i6
    %32 = arith.extsi %31 {bb = 2 : ui32, name = #handshake.name<"extsi13">} : i6 to i7
    %33 = arith.cmpi ult, %20#1, %32 {bb = 2 : ui32, name = #handshake.name<"cmpi0">} : i7
    %34 = buffer [1] seq %33 {bb = 2 : ui32, name = #handshake.name<"buffer12">} : i1
    %35:5 = fork [5] %34 {bb = 2 : ui32, name = #handshake.name<"fork6">} : i1
    %trueResult, %falseResult = cond_br %35#4, %29 {bb = 2 : ui32, name = #handshake.name<"cond_br0">} : i2
    sink %falseResult {name = #handshake.name<"sink0">} : i2
    %36 = arith.extsi %trueResult {bb = 2 : ui32, name = #handshake.name<"extsi14">} : i2 to i6
    %37 = buffer [1] fifo %35#2 {bb = 2 : ui32, name = #handshake.name<"buffer2">} : i1
    %38 = buffer [1] seq %23 {bb = 2 : ui32, name = #handshake.name<"buffer7">} : i32
    %trueResult_4, %falseResult_5 = cond_br %37, %38 {bb = 2 : ui32, name = #handshake.name<"cond_br4">} : i32
    %39 = buffer [1] seq %24 {bb = 2 : ui32, name = #handshake.name<"buffer18">} : i6
    %trueResult_6, %falseResult_7 = cond_br %35#1, %39 {bb = 2 : ui32, name = #handshake.name<"cond_br1">} : i6
    %trueResult_8, %falseResult_9 = cond_br %35#0, %22 {bb = 2 : ui32, name = #handshake.name<"cond_br2">} : i6
    sink %falseResult_9 {name = #handshake.name<"sink1">} : i6
    %trueResult_10, %falseResult_11 = cond_br %35#3, %28#1 {bb = 2 : ui32, name = #handshake.name<"cond_br7">} : none
    %40 = mux %62#2 [%trueResult_20, %36] {bb = 3 : ui32, name = #handshake.name<"mux9">} : i1, i6
    %41:5 = fork [5] %40 {bb = 3 : ui32, name = #handshake.name<"fork7">} : i6
    %42 = buffer [3] seq %41#0 {bb = 3 : ui32, name = #handshake.name<"buffer13">} : i6
    %43 = arith.extsi %42 {bb = 3 : ui32, name = #handshake.name<"extsi15">} : i6 to i12
    %44 = arith.extsi %41#1 {bb = 3 : ui32, name = #handshake.name<"extsi16">} : i6 to i12
    %45 = arith.extsi %41#2 {bb = 3 : ui32, name = #handshake.name<"extsi17">} : i6 to i12
    %46 = arith.extsi %41#3 {bb = 3 : ui32, name = #handshake.name<"extsi18">} : i6 to i7
    %47 = arith.extsi %41#4 {bb = 3 : ui32, name = #handshake.name<"extsi19">} : i6 to i32
    %48 = mux %62#3 [%trueResult_22, %trueResult_4] {bb = 3 : ui32, name = #handshake.name<"mux6">} : i1, i32
    %49 = mux %62#0 [%trueResult_24, %trueResult_6] {bb = 3 : ui32, name = #handshake.name<"mux5">} : i1, i6
    %50 = buffer [2] seq %49 {bb = 3 : ui32, name = #handshake.name<"buffer15">} : i6
    %51:4 = fork [4] %50 {bb = 3 : ui32, name = #handshake.name<"fork8">} : i6
    %52 = arith.extsi %51#1 {bb = 3 : ui32, name = #handshake.name<"extsi2">} : i6 to i10
    %53 = arith.extsi %51#2 {bb = 3 : ui32, name = #handshake.name<"extsi3">} : i6 to i8
    %54 = arith.extsi %51#3 {bb = 3 : ui32, name = #handshake.name<"extsi20">} : i6 to i32
    %55 = mux %62#1 [%trueResult_26, %trueResult_8] {bb = 3 : ui32, name = #handshake.name<"mux7">} : i1, i6
    %56 = buffer [2] seq %55 {bb = 3 : ui32, name = #handshake.name<"buffer31">} : i6
    %57:5 = fork [5] %56 {bb = 3 : ui32, name = #handshake.name<"fork9">} : i6
    %58 = arith.extsi %57#1 {bb = 3 : ui32, name = #handshake.name<"extsi21">} : i6 to i10
    %59 = arith.extsi %57#2 {bb = 3 : ui32, name = #handshake.name<"extsi22">} : i6 to i8
    %60 = arith.extsi %57#3 {bb = 3 : ui32, name = #handshake.name<"extsi23">} : i6 to i10
    %61 = arith.extsi %57#4 {bb = 3 : ui32, name = #handshake.name<"extsi24">} : i6 to i8
    %result_12, %index_13 = control_merge %trueResult_28, %trueResult_10 {bb = 3 : ui32, name = #handshake.name<"control_merge2">} : none, i1
    %62:4 = fork [4] %index_13 {bb = 3 : ui32, name = #handshake.name<"fork10">} : i1
    %63 = buffer [2] seq %result_12 {bb = 3 : ui32, name = #handshake.name<"buffer10">} : none
    %64:2 = fork [2] %63 {bb = 3 : ui32, name = #handshake.name<"fork11">} : none
    %65 = source {bb = 3 : ui32, name = #handshake.name<"source2">}
    %66 = constant %65 {bb = 3 : ui32, name = #handshake.name<"constant14">, value = 20 : i6} : i6
    %67 = arith.extsi %66 {bb = 3 : ui32, name = #handshake.name<"extsi25">} : i6 to i7
    %68 = source {bb = 3 : ui32, name = #handshake.name<"source3">}
    %69 = constant %68 {bb = 3 : ui32, name = #handshake.name<"constant15">, value = 1 : i2} : i2
    %70 = arith.extsi %69 {bb = 3 : ui32, name = #handshake.name<"extsi26">} : i2 to i7
    %71 = source {bb = 3 : ui32, name = #handshake.name<"source4">}
    %72 = constant %71 {bb = 3 : ui32, name = #handshake.name<"constant16">, value = 4 : i4} : i4
    %73:3 = fork [3] %72 {bb = 3 : ui32, name = #handshake.name<"fork12">} : i4
    %74 = arith.extui %73#0 {bb = 3 : ui32, name = #handshake.name<"extui0">} : i4 to i10
    %75 = arith.extui %73#1 {bb = 3 : ui32, name = #handshake.name<"extui1">} : i4 to i10
    %76 = arith.extui %73#2 {bb = 3 : ui32, name = #handshake.name<"extui2">} : i4 to i10
    %77 = source {bb = 3 : ui32, name = #handshake.name<"source5">}
    %78 = constant %77 {bb = 3 : ui32, name = #handshake.name<"constant17">, value = 2 : i3} : i3
    %79:3 = fork [3] %78 {bb = 3 : ui32, name = #handshake.name<"fork13">} : i3
    %80 = arith.extui %79#0 {bb = 3 : ui32, name = #handshake.name<"extui3">} : i3 to i8
    %81 = arith.extui %79#1 {bb = 3 : ui32, name = #handshake.name<"extui4">} : i3 to i8
    %82 = arith.extui %79#2 {bb = 3 : ui32, name = #handshake.name<"extui5">} : i3 to i8
    %83 = arith.shli %61, %82 {bb = 3 : ui32, name = #handshake.name<"shli0">} : i8
    %84 = buffer [1] seq %83 {bb = 3 : ui32, name = #handshake.name<"buffer20">} : i8
    %85 = arith.extsi %84 {bb = 3 : ui32, name = #handshake.name<"extsi27">} : i8 to i11
    %86 = arith.shli %60, %76 {bb = 3 : ui32, name = #handshake.name<"shli1">} : i10
    %87 = buffer [1] seq %86 {bb = 3 : ui32, name = #handshake.name<"buffer4">} : i10
    %88 = arith.extsi %87 {bb = 3 : ui32, name = #handshake.name<"extsi28">} : i10 to i11
    %89 = arith.addi %85, %88 {bb = 3 : ui32, name = #handshake.name<"addi2">} : i11
    %90 = arith.extsi %89 {bb = 3 : ui32, name = #handshake.name<"extsi29">} : i11 to i12
    %91 = buffer [5] seq %45 {bb = 3 : ui32, name = #handshake.name<"buffer3">} : i12
    %92 = buffer [3] seq %90 {bb = 3 : ui32, name = #handshake.name<"buffer34">} : i12
    %93 = arith.addi %91, %92 {bb = 3 : ui32, name = #handshake.name<"addi3">} : i12
    %94 = arith.extsi %93 {bb = 3 : ui32, name = #handshake.name<"extsi30">} : i12 to i32
    %addressResult, %dataResult = lsq_load[%94] %memOutputs#0 {bb = 3 : ui32, bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00}>, name = #handshake.name<"lsq_load0">} : i32, i32
    %95 = buffer [3] fifo %54 {bb = 3 : ui32, name = #handshake.name<"buffer36">} : i32
    %addressResult_14, %dataResult_15 = mc_load[%95] %memOutputs_0 {bb = 3 : ui32, bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00}>, name = #handshake.name<"mc_load0">} : i32, i32
    %96 = arith.shli %53, %80 {bb = 3 : ui32, name = #handshake.name<"shli2">} : i8
    %97 = arith.extsi %96 {bb = 3 : ui32, name = #handshake.name<"extsi4">} : i8 to i11
    %98 = arith.shli %52, %74 {bb = 3 : ui32, name = #handshake.name<"shli3">} : i10
    %99 = buffer [1] seq %98 {bb = 3 : ui32, name = #handshake.name<"buffer37">} : i10
    %100 = arith.extsi %99 {bb = 3 : ui32, name = #handshake.name<"extsi31">} : i10 to i11
    %101 = buffer [1] seq %97 {bb = 3 : ui32, name = #handshake.name<"buffer21">} : i11
    %102 = arith.addi %101, %100 {bb = 3 : ui32, name = #handshake.name<"addi4">} : i11
    %103 = buffer [1] seq %102 {bb = 3 : ui32, name = #handshake.name<"buffer5">} : i11
    %104 = arith.extsi %103 {bb = 3 : ui32, name = #handshake.name<"extsi32">} : i11 to i12
    %105 = arith.addi %43, %104 {bb = 3 : ui32, name = #handshake.name<"addi5">} : i12
    %106 = arith.extsi %105 {bb = 3 : ui32, name = #handshake.name<"extsi33">} : i12 to i32
    %addressResult_16, %dataResult_17 = lsq_load[%106] %memOutputs#1 {bb = 3 : ui32, bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00}>, name = #handshake.name<"lsq_load1">} : i32, i32
    %107 = arith.muli %dataResult_15, %dataResult_17 {bb = 3 : ui32, name = #handshake.name<"muli0">} : i32
    %108 = arith.subi %dataResult, %107 {bb = 3 : ui32, name = #handshake.name<"subi0">} : i32
    %109 = arith.shli %59, %81 {bb = 3 : ui32, name = #handshake.name<"shli4">} : i8
    %110 = arith.extsi %109 {bb = 3 : ui32, name = #handshake.name<"extsi34">} : i8 to i11
    %111 = arith.shli %58, %75 {bb = 3 : ui32, name = #handshake.name<"shli5">} : i10
    %112 = buffer [1] seq %111 {bb = 3 : ui32, name = #handshake.name<"buffer6">} : i10
    %113 = arith.extsi %112 {bb = 3 : ui32, name = #handshake.name<"extsi35">} : i10 to i11
    %114 = buffer [1] seq %110 {bb = 3 : ui32, name = #handshake.name<"buffer0">} : i11
    %115 = arith.addi %114, %113 {bb = 3 : ui32, name = #handshake.name<"addi6">} : i11
    %116 = arith.extsi %115 {bb = 3 : ui32, name = #handshake.name<"extsi36">} : i11 to i12
    %117 = buffer [7] seq %44 {bb = 3 : ui32, name = #handshake.name<"buffer8">} : i12
    %118 = buffer [6] seq %116 {bb = 3 : ui32, name = #handshake.name<"buffer22">} : i12
    %119 = arith.addi %117, %118 {bb = 3 : ui32, name = #handshake.name<"addi7">} : i12
    %120 = arith.extsi %119 {bb = 3 : ui32, name = #handshake.name<"extsi37">} : i12 to i32
    %addressResult_18, %dataResult_19 = lsq_store[%120] %108 {bb = 3 : ui32, name = #handshake.name<"lsq_store0">} : i32, i32
    %121 = buffer [1] seq %47 {bb = 3 : ui32, name = #handshake.name<"buffer9">} : i32
    %122 = buffer [1] seq %48 {bb = 3 : ui32, name = #handshake.name<"buffer25">} : i32
    %123 = arith.addi %122, %121 {bb = 3 : ui32, name = #handshake.name<"addi0">} : i32
    %124 = buffer [1] seq %46 {bb = 3 : ui32, name = #handshake.name<"buffer14">} : i7
    %125 = buffer [1] seq %70 {bb = 3 : ui32, name = #handshake.name<"buffer24">} : i7
    %126 = arith.addi %124, %125 {bb = 3 : ui32, name = #handshake.name<"addi11">} : i7
    %127:2 = fork [2] %126 {bb = 3 : ui32, name = #handshake.name<"fork14">} : i7
    %128 = arith.trunci %127#0 {bb = 3 : ui32, name = #handshake.name<"trunci1">} : i7 to i6
    %129 = arith.cmpi ult, %127#1, %67 {bb = 3 : ui32, name = #handshake.name<"cmpi1">} : i7
    %130 = buffer [1] seq %129 {bb = 3 : ui32, name = #handshake.name<"buffer27">} : i1
    %131:5 = fork [5] %130 {bb = 3 : ui32, name = #handshake.name<"fork15">} : i1
    %132 = buffer [1] seq %128 {bb = 3 : ui32, name = #handshake.name<"buffer1">} : i6
    %trueResult_20, %falseResult_21 = cond_br %131#0, %132 {bb = 3 : ui32, name = #handshake.name<"cond_br3">} : i6
    sink %falseResult_21 {name = #handshake.name<"sink2">} : i6
    %133 = buffer [1] seq %123 {bb = 3 : ui32, name = #handshake.name<"buffer33">} : i32
    %trueResult_22, %falseResult_23 = cond_br %131#3, %133 {bb = 3 : ui32, name = #handshake.name<"cond_br12">} : i32
    %trueResult_24, %falseResult_25 = cond_br %131#1, %51#0 {bb = 3 : ui32, name = #handshake.name<"cond_br5">} : i6
    %trueResult_26, %falseResult_27 = cond_br %131#2, %57#0 {bb = 3 : ui32, name = #handshake.name<"cond_br6">} : i6
    %trueResult_28, %falseResult_29 = cond_br %131#4, %64#1 {bb = 3 : ui32, name = #handshake.name<"cond_br15">} : none
    %134 = arith.extsi %falseResult_27 {bb = 4 : ui32, name = #handshake.name<"extsi5">} : i6 to i7
    %135 = source {bb = 4 : ui32, name = #handshake.name<"source6">}
    %136 = constant %135 {bb = 4 : ui32, name = #handshake.name<"constant18">, value = 1 : i2} : i2
    %137 = arith.extsi %136 {bb = 4 : ui32, name = #handshake.name<"extsi38">} : i2 to i7
    %138 = arith.addi %134, %137 {bb = 4 : ui32, name = #handshake.name<"addi8">} : i7
    %139 = arith.extsi %falseResult_7 {bb = 5 : ui32, name = #handshake.name<"extsi6">} : i6 to i7
    %140 = source {bb = 5 : ui32, name = #handshake.name<"source7">}
    %141 = constant %140 {bb = 5 : ui32, name = #handshake.name<"constant19">, value = 1 : i2} : i2
    %142 = arith.extsi %141 {bb = 5 : ui32, name = #handshake.name<"extsi39">} : i2 to i7
    %143 = source {bb = 5 : ui32, name = #handshake.name<"source8">}
    %144 = constant %143 {bb = 5 : ui32, name = #handshake.name<"constant20">, value = 19 : i6} : i6
    %145 = arith.extsi %144 {bb = 5 : ui32, name = #handshake.name<"extsi7">} : i6 to i7
    %146 = arith.addi %139, %142 {bb = 5 : ui32, name = #handshake.name<"addi9">} : i7
    %147 = buffer [1] seq %146 {bb = 5 : ui32, name = #handshake.name<"buffer29">} : i7
    %148:2 = fork [2] %147 {bb = 5 : ui32, name = #handshake.name<"fork16">} : i7
    %149 = arith.trunci %148#0 {bb = 5 : ui32, name = #handshake.name<"trunci2">} : i7 to i6
    %150 = buffer [1] seq %145 {bb = 5 : ui32, name = #handshake.name<"buffer30">} : i7
    %151 = arith.cmpi ult, %148#1, %150 {bb = 5 : ui32, name = #handshake.name<"cmpi2">} : i7
    %152:3 = fork [3] %151 {bb = 5 : ui32, name = #handshake.name<"fork17">} : i1
    %trueResult_30, %falseResult_31 = cond_br %152#0, %149 {bb = 5 : ui32, name = #handshake.name<"cond_br8">} : i6
    sink %falseResult_31 {name = #handshake.name<"sink3">} : i6
    %trueResult_32, %falseResult_33 = cond_br %152#1, %falseResult_5 {bb = 5 : ui32, name = #handshake.name<"cond_br21">} : i32
    %153 = buffer [1] fifo %falseResult_11 {bb = 5 : ui32, name = #handshake.name<"buffer23">} : none
    %trueResult_34, %falseResult_35 = cond_br %152#2, %153 {bb = 5 : ui32, name = #handshake.name<"cond_br22">} : none
    sink %falseResult_35 {name = #handshake.name<"sink4">} : none
    %154 = buffer [1] seq %falseResult_33 {bb = 6 : ui32, name = #handshake.name<"buffer16">} : i32
    %155 = d_return {bb = 6 : ui32, name = #handshake.name<"d_return0">} %154 : i32
    end {bb = 6 : ui32, bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00}>, name = #handshake.name<"end0">} %155, %done, %done_1 : i32, none, none
  }
}

