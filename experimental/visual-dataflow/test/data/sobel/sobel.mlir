module {
  handshake.func @sobel(%arg0: memref<256xi32>, %arg1: memref<9xi32>, %arg2: memref<9xi32>, %arg3: memref<256xi32>, %arg4: none, ...) -> i32 attributes {argNames = ["in", "gX", "gY", "out", "start"], resNames = ["out0"]} {
    %done = lsq[%arg3 : memref<256xi32>] (%216#0, %addressResult_57, %dataResult_58)  {bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00}>, groupSizes = [1 : i32], name = #handshake.name<"lsq0">} : (none, i32, i32) -> none
    %memOutputs, %done_0 = mem_controller[%arg2 : memref<9xi32>] (%addressResult_19) {bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00}>, connectedBlocks = [5 : i32], name = #handshake.name<"mem_controller0">} : (i32) -> (i32, none)
    %memOutputs_1, %done_2 = mem_controller[%arg1 : memref<9xi32>] (%addressResult_17) {bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00}>, connectedBlocks = [5 : i32], name = #handshake.name<"mem_controller1">} : (i32) -> (i32, none)
    %memOutputs_3, %done_4 = mem_controller[%arg0 : memref<256xi32>] (%addressResult) {bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00}>, connectedBlocks = [3 : i32], name = #handshake.name<"mem_controller2">} : (i32) -> (i32, none)
    %0:2 = fork [2] %arg4 {bb = 0 : ui32, name = #handshake.name<"fork0">} : none
    %1 = constant %0#1 {bb = 0 : ui32, name = #handshake.name<"constant1">, value = false} : i1
    %2:2 = fork [2] %1 {bb = 0 : ui32, name = #handshake.name<"fork1">} : i1
    %3 = arith.extsi %2#0 {bb = 0 : ui32, name = #handshake.name<"extsi0">} : i1 to i5
    %4 = arith.extsi %2#1 {bb = 0 : ui32, name = #handshake.name<"extsi16">} : i1 to i32
    %5 = mux %7#0 [%trueResult_67, %3] {bb = 1 : ui32, name = #handshake.name<"mux0">} : i1, i5
    %6 = mux %7#1 [%trueResult_69, %4] {bb = 1 : ui32, name = #handshake.name<"mux1">} : i1, i32
    %result, %index = control_merge %trueResult_71, %0#0 {bb = 1 : ui32, name = #handshake.name<"control_merge0">} : none, i1
    %7:2 = fork [2] %index {bb = 1 : ui32, name = #handshake.name<"fork2">} : i1
    %8:2 = fork [2] %result {bb = 1 : ui32, name = #handshake.name<"fork3">} : none
    %9 = constant %8#0 {bb = 1 : ui32, name = #handshake.name<"constant4">, value = false} : i1
    %10 = arith.extsi %9 {bb = 1 : ui32, name = #handshake.name<"extsi1">} : i1 to i5
    %11 = buffer [1] seq %10 {bb = 2 : ui32, name = #handshake.name<"buffer56">} : i5
    %12 = buffer [1] seq %22#1 {bb = 2 : ui32, name = #handshake.name<"buffer70">} : i1
    %13 = buffer [1] seq %trueResult_59 {bb = 2 : ui32, name = #handshake.name<"buffer83">} : i5
    %14 = mux %12 [%13, %11] {bb = 2 : ui32, name = #handshake.name<"mux23">} : i1, i5
    %15:4 = fork [4] %14 {bb = 2 : ui32, name = #handshake.name<"fork4">} : i5
    %16 = buffer [2] seq %22#2 {bb = 2 : ui32, name = #handshake.name<"buffer42">} : i1
    %17 = buffer [1] seq %6 {bb = 2 : ui32, name = #handshake.name<"buffer44">} : i32
    %18 = mux %16 [%trueResult_61, %17] {bb = 2 : ui32, name = #handshake.name<"mux3">} : i1, i32
    %19 = buffer [1] seq %5 {bb = 2 : ui32, name = #handshake.name<"buffer49">} : i5
    %20 = buffer [1] seq %22#0 {bb = 2 : ui32, name = #handshake.name<"buffer54">} : i1
    %21 = mux %20 [%trueResult_63, %19] {bb = 2 : ui32, name = #handshake.name<"mux2">} : i1, i5
    %result_5, %index_6 = control_merge %trueResult_65, %8#1 {bb = 2 : ui32, name = #handshake.name<"control_merge1">} : none, i1
    %22:3 = fork [3] %index_6 {bb = 2 : ui32, name = #handshake.name<"fork5">} : i1
    %23 = source {bb = 2 : ui32, name = #handshake.name<"source0">}
    %24 = constant %23 {bb = 2 : ui32, name = #handshake.name<"constant5">, value = 5 : i4} : i4
    %25 = arith.extsi %24 {bb = 2 : ui32, name = #handshake.name<"extsi17">} : i4 to i5
    %26 = source {bb = 2 : ui32, name = #handshake.name<"source1">}
    %27 = constant %26 {bb = 2 : ui32, name = #handshake.name<"constant6">, value = false} : i1
    %28:3 = fork [3] %27 {bb = 2 : ui32, name = #handshake.name<"fork6">} : i1
    %29 = arith.extsi %28#0 {bb = 2 : ui32, name = #handshake.name<"extsi18">} : i1 to i5
    %30 = arith.extsi %28#1 {bb = 2 : ui32, name = #handshake.name<"extsi2">} : i1 to i5
    %31 = arith.cmpi eq, %15#0, %29 {bb = 2 : ui32, name = #handshake.name<"cmpi0">} : i5
    %32 = arith.cmpi eq, %15#1, %25 {bb = 2 : ui32, name = #handshake.name<"cmpi1">} : i5
    %33 = arith.cmpi ne, %15#2, %30 {bb = 2 : ui32, name = #handshake.name<"cmpi2">} : i5
    %34 = buffer [1] seq %32 {bb = 2 : ui32, name = #handshake.name<"buffer27">} : i1
    %35 = buffer [1] seq %33 {bb = 2 : ui32, name = #handshake.name<"buffer82">} : i1
    %36 = arith.andi %35, %34 {bb = 2 : ui32, name = #handshake.name<"andi0">} : i1
    %37 = buffer [1] seq %31 {bb = 2 : ui32, name = #handshake.name<"buffer51">} : i1
    %38 = arith.ori %37, %36 {bb = 2 : ui32, name = #handshake.name<"ori0">} : i1
    %39 = buffer [1] fifo %28#2 {bb = 2 : ui32, name = #handshake.name<"buffer2">} : i1
    %40 = buffer [1] seq %38 {bb = 2 : ui32, name = #handshake.name<"buffer92">} : i1
    %41 = arith.cmpi eq, %40, %39 {bb = 2 : ui32, name = #handshake.name<"cmpi3">} : i1
    %42:4 = fork [4] %41 {bb = 2 : ui32, name = #handshake.name<"fork7">} : i1
    %trueResult, %falseResult = cond_br %42#0, %21 {bb = 2 : ui32, name = #handshake.name<"cond_br0">} : i5
    %43 = buffer [1] fifo %15#3 {bb = 2 : ui32, name = #handshake.name<"buffer11">} : i5
    %trueResult_7, %falseResult_8 = cond_br %42#1, %43 {bb = 2 : ui32, name = #handshake.name<"cond_br1">} : i5
    %44 = buffer [1] fifo %42#2 {bb = 2 : ui32, name = #handshake.name<"buffer62">} : i1
    %trueResult_9, %falseResult_10 = cond_br %44, %18 {bb = 2 : ui32, name = #handshake.name<"cond_br7">} : i32
    %45 = buffer [1] seq %result_5 {bb = 2 : ui32, name = #handshake.name<"buffer12">} : none
    %trueResult_11, %falseResult_12 = cond_br %42#3, %45 {bb = 2 : ui32, name = #handshake.name<"cond_br8">} : none
    %46:2 = fork [2] %trueResult_7 {bb = 3 : ui32, name = #handshake.name<"fork8">} : i5
    %47 = arith.extsi %46#1 {bb = 3 : ui32, name = #handshake.name<"extsi3">} : i5 to i32
    %48 = buffer [1] seq %trueResult_11 {bb = 3 : ui32, name = #handshake.name<"buffer66">} : none
    %49:3 = fork [3] %48 {bb = 3 : ui32, name = #handshake.name<"fork9">} : none
    %50 = constant %49#2 {bb = 3 : ui32, name = #handshake.name<"constant7">, value = -1 : i32} : i32
    %51 = constant %49#1 {bb = 3 : ui32, name = #handshake.name<"constant8">, value = false} : i1
    %52:2 = fork [2] %51 {bb = 3 : ui32, name = #handshake.name<"fork10">} : i1
    %addressResult, %dataResult = mc_load[%47] %memOutputs_3 {bb = 3 : ui32, bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00}>, name = #handshake.name<"mc_load0">} : i32, i32
    %53 = arith.extsi %52#0 {bb = 3 : ui32, name = #handshake.name<"extsi19">} : i1 to i32
    %54 = arith.extsi %52#1 {bb = 3 : ui32, name = #handshake.name<"extsi20">} : i1 to i32
    %55 = buffer [1] seq %46#0 {bb = 3 : ui32, name = #handshake.name<"buffer41">} : i5
    %56 = mux %70#6 [%162, %50] {bb = 4 : ui32, name = #handshake.name<"mux5">} : i1, i32
    %57 = buffer [2] fifo %70#5 {bb = 4 : ui32, name = #handshake.name<"buffer78">} : i1
    %58 = mux %57 [%trueResult_41, %54] {bb = 4 : ui32, name = #handshake.name<"mux6">} : i1, i32
    %59 = buffer [2] fifo %70#4 {bb = 4 : ui32, name = #handshake.name<"buffer89">} : i1
    %60 = mux %59 [%trueResult_43, %53] {bb = 4 : ui32, name = #handshake.name<"mux7">} : i1, i32
    %61 = buffer [1] seq %trueResult {bb = 4 : ui32, name = #handshake.name<"buffer16">} : i5
    %62 = mux %70#0 [%trueResult_45, %61] {bb = 4 : ui32, name = #handshake.name<"mux4">} : i1, i5
    %63 = mux %70#1 [%trueResult_47, %55] {bb = 4 : ui32, name = #handshake.name<"mux8">} : i1, i5
    %64 = buffer [1] fifo %70#3 {bb = 4 : ui32, name = #handshake.name<"buffer71">} : i1
    %65 = buffer [1] seq %trueResult_9 {bb = 4 : ui32, name = #handshake.name<"buffer77">} : i32
    %66 = mux %64 [%trueResult_49, %65] {bb = 4 : ui32, name = #handshake.name<"mux10">} : i1, i32
    %67 = buffer [1] seq %trueResult_51 {bb = 4 : ui32, name = #handshake.name<"buffer7">} : i32
    %68 = buffer [1] fifo %70#2 {bb = 4 : ui32, name = #handshake.name<"buffer79">} : i1
    %69 = mux %68 [%67, %dataResult] {bb = 4 : ui32, name = #handshake.name<"mux11">} : i1, i32
    %result_13, %index_14 = control_merge %trueResult_53, %49#0 {bb = 4 : ui32, name = #handshake.name<"control_merge2">} : none, i1
    %70:7 = fork [7] %index_14 {bb = 4 : ui32, name = #handshake.name<"fork11">} : i1
    %71:2 = fork [2] %result_13 {bb = 4 : ui32, name = #handshake.name<"fork12">} : none
    %72 = constant %71#0 {bb = 4 : ui32, name = #handshake.name<"constant21">, value = -1 : i32} : i32
    %73 = buffer [1] seq %138 {bb = 5 : ui32, name = #handshake.name<"buffer28">} : i32
    %74 = mux %94#7 [%73, %72] {bb = 5 : ui32, name = #handshake.name<"mux12">} : i1, i32
    %75:3 = fork [3] %74 {bb = 5 : ui32, name = #handshake.name<"fork13">} : i32
    %76 = buffer [5] fifo %94#6 {bb = 5 : ui32, name = #handshake.name<"buffer81">} : i1
    %77 = mux %76 [%trueResult_23, %58] {bb = 5 : ui32, name = #handshake.name<"mux13">} : i1, i32
    %78 = buffer [5] fifo %94#5 {bb = 5 : ui32, name = #handshake.name<"buffer74">} : i1
    %79 = mux %78 [%trueResult_25, %60] {bb = 5 : ui32, name = #handshake.name<"mux14">} : i1, i32
    %80 = buffer [1] fifo %trueResult_27 {bb = 5 : ui32, name = #handshake.name<"buffer3">} : i5
    %81 = mux %94#0 [%80, %62] {bb = 5 : ui32, name = #handshake.name<"mux9">} : i1, i5
    %82 = buffer [1] fifo %trueResult_29 {bb = 5 : ui32, name = #handshake.name<"buffer68">} : i5
    %83 = mux %94#1 [%82, %63] {bb = 5 : ui32, name = #handshake.name<"mux15">} : i1, i5
    %84 = buffer [1] fifo %94#4 {bb = 5 : ui32, name = #handshake.name<"buffer45">} : i1
    %85 = mux %84 [%trueResult_31, %66] {bb = 5 : ui32, name = #handshake.name<"mux17">} : i1, i32
    %86 = buffer [3] fifo %94#3 {bb = 5 : ui32, name = #handshake.name<"buffer43">} : i1
    %87 = mux %86 [%trueResult_33, %69] {bb = 5 : ui32, name = #handshake.name<"mux18">} : i1, i32
    %88 = buffer [2] seq %87 {bb = 5 : ui32, name = #handshake.name<"buffer39">} : i32
    %89:3 = fork [3] %88 {bb = 5 : ui32, name = #handshake.name<"fork14">} : i32
    %90 = mux %94#2 [%trueResult_35, %56] {bb = 5 : ui32, name = #handshake.name<"mux19">} : i1, i32
    %91 = buffer [2] seq %90 {bb = 5 : ui32, name = #handshake.name<"buffer4">} : i32
    %92:5 = fork [5] %91 {bb = 5 : ui32, name = #handshake.name<"fork15">} : i32
    %93 = buffer [1] seq %trueResult_37 {bb = 5 : ui32, name = #handshake.name<"buffer59">} : none
    %result_15, %index_16 = control_merge %93, %71#1 {bb = 5 : ui32, name = #handshake.name<"control_merge3">} : none, i1
    %94:8 = fork [8] %index_16 {bb = 5 : ui32, name = #handshake.name<"fork16">} : i1
    %95 = source {bb = 5 : ui32, name = #handshake.name<"source2">}
    %96 = constant %95 {bb = 5 : ui32, name = #handshake.name<"constant22">, value = 4 : i4} : i4
    %97 = arith.extsi %96 {bb = 5 : ui32, name = #handshake.name<"extsi4">} : i4 to i32
    %98:2 = fork [2] %97 {bb = 5 : ui32, name = #handshake.name<"fork17">} : i32
    %99 = source {bb = 5 : ui32, name = #handshake.name<"source3">}
    %100 = constant %99 {bb = 5 : ui32, name = #handshake.name<"constant23">, value = 2 : i3} : i3
    %101 = arith.extsi %100 {bb = 5 : ui32, name = #handshake.name<"extsi5">} : i3 to i32
    %102 = source {bb = 5 : ui32, name = #handshake.name<"source4">}
    %103 = constant %102 {bb = 5 : ui32, name = #handshake.name<"constant24">, value = 1 : i2} : i2
    %104 = arith.extsi %103 {bb = 5 : ui32, name = #handshake.name<"extsi6">} : i2 to i32
    %105:3 = fork [3] %104 {bb = 5 : ui32, name = #handshake.name<"fork18">} : i32
    %106 = buffer [2] seq %105#0 {bb = 5 : ui32, name = #handshake.name<"buffer85">} : i32
    %107 = arith.shli %92#4, %106 {bb = 5 : ui32, name = #handshake.name<"shli0">} : i32
    %108 = buffer [1] seq %107 {bb = 5 : ui32, name = #handshake.name<"buffer5">} : i32
    %109 = buffer [1] fifo %92#3 {bb = 5 : ui32, name = #handshake.name<"buffer80">} : i32
    %110 = arith.addi %109, %108 {bb = 5 : ui32, name = #handshake.name<"addi13">} : i32
    %111 = buffer [2] fifo %75#2 {bb = 5 : ui32, name = #handshake.name<"buffer21">} : i32
    %112 = buffer [1] seq %110 {bb = 5 : ui32, name = #handshake.name<"buffer91">} : i32
    %113 = arith.addi %111, %112 {bb = 5 : ui32, name = #handshake.name<"addi4">} : i32
    %114 = buffer [1] seq %113 {bb = 5 : ui32, name = #handshake.name<"buffer87">} : i32
    %115 = arith.addi %114, %98#0 {bb = 5 : ui32, name = #handshake.name<"addi5">} : i32
    %addressResult_17, %dataResult_18 = mc_load[%115] %memOutputs_1 {bb = 5 : ui32, bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00}>, name = #handshake.name<"mc_load1">} : i32, i32
    %116 = arith.muli %89#2, %dataResult_18 {bb = 5 : ui32, name = #handshake.name<"muli0">} : i32
    %117 = buffer [2] seq %79 {bb = 5 : ui32, name = #handshake.name<"buffer19">} : i32
    %118 = arith.addi %117, %116 {bb = 5 : ui32, name = #handshake.name<"addi0">} : i32
    %119 = buffer [2] seq %105#1 {bb = 5 : ui32, name = #handshake.name<"buffer1">} : i32
    %120 = arith.shli %92#2, %119 {bb = 5 : ui32, name = #handshake.name<"shli1">} : i32
    %121 = buffer [1] fifo %92#1 {bb = 5 : ui32, name = #handshake.name<"buffer22">} : i32
    %122 = buffer [1] seq %120 {bb = 5 : ui32, name = #handshake.name<"buffer24">} : i32
    %123 = arith.addi %121, %122 {bb = 5 : ui32, name = #handshake.name<"addi14">} : i32
    %124 = buffer [2] fifo %75#1 {bb = 5 : ui32, name = #handshake.name<"buffer53">} : i32
    %125 = buffer [1] seq %123 {bb = 5 : ui32, name = #handshake.name<"buffer72">} : i32
    %126 = arith.addi %124, %125 {bb = 5 : ui32, name = #handshake.name<"addi6">} : i32
    %127 = buffer [1] seq %126 {bb = 5 : ui32, name = #handshake.name<"buffer30">} : i32
    %128 = arith.addi %127, %98#1 {bb = 5 : ui32, name = #handshake.name<"addi7">} : i32
    %addressResult_19, %dataResult_20 = mc_load[%128] %memOutputs {bb = 5 : ui32, bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00}>, name = #handshake.name<"mc_load2">} : i32, i32
    %129 = arith.muli %89#1, %dataResult_20 {bb = 5 : ui32, name = #handshake.name<"muli1">} : i32
    %130 = buffer [2] seq %77 {bb = 5 : ui32, name = #handshake.name<"buffer38">} : i32
    %131 = arith.addi %130, %129 {bb = 5 : ui32, name = #handshake.name<"addi1">} : i32
    %132 = arith.addi %75#0, %105#2 {bb = 5 : ui32, name = #handshake.name<"addi9">} : i32
    %133 = buffer [1] seq %132 {bb = 5 : ui32, name = #handshake.name<"buffer40">} : i32
    %134:2 = fork [2] %133 {bb = 5 : ui32, name = #handshake.name<"fork19">} : i32
    %135 = arith.trunci %134#0 {bb = 5 : ui32, name = #handshake.name<"trunci1">} : i32 to i2
    %136 = arith.cmpi ult, %134#1, %101 {bb = 5 : ui32, name = #handshake.name<"cmpi8">} : i32
    %137:9 = fork [9] %136 {bb = 5 : ui32, name = #handshake.name<"fork20">} : i1
    %trueResult_21, %falseResult_22 = cond_br %137#0, %135 {bb = 5 : ui32, name = #handshake.name<"cond_br2">} : i2
    sink %falseResult_22 {name = #handshake.name<"sink0">} : i2
    %138 = arith.extsi %trueResult_21 {bb = 5 : ui32, name = #handshake.name<"extsi21">} : i2 to i32
    %139 = buffer [5] fifo %137#8 {bb = 5 : ui32, name = #handshake.name<"buffer57">} : i1
    %trueResult_23, %falseResult_24 = cond_br %139, %131 {bb = 5 : ui32, name = #handshake.name<"cond_br14">} : i32
    %140 = buffer [5] fifo %137#7 {bb = 5 : ui32, name = #handshake.name<"buffer36">} : i1
    %trueResult_25, %falseResult_26 = cond_br %140, %118 {bb = 5 : ui32, name = #handshake.name<"cond_br15">} : i32
    %141 = buffer [1] seq %81 {bb = 5 : ui32, name = #handshake.name<"buffer55">} : i5
    %trueResult_27, %falseResult_28 = cond_br %137#1, %141 {bb = 5 : ui32, name = #handshake.name<"cond_br3">} : i5
    %142 = buffer [1] seq %83 {bb = 5 : ui32, name = #handshake.name<"buffer61">} : i5
    %trueResult_29, %falseResult_30 = cond_br %137#2, %142 {bb = 5 : ui32, name = #handshake.name<"cond_br4">} : i5
    %143 = buffer [1] fifo %137#6 {bb = 5 : ui32, name = #handshake.name<"buffer17">} : i1
    %144 = buffer [2] seq %85 {bb = 5 : ui32, name = #handshake.name<"buffer34">} : i32
    %trueResult_31, %falseResult_32 = cond_br %143, %144 {bb = 5 : ui32, name = #handshake.name<"cond_br18">} : i32
    %145 = buffer [3] fifo %137#5 {bb = 5 : ui32, name = #handshake.name<"buffer46">} : i1
    %trueResult_33, %falseResult_34 = cond_br %145, %89#0 {bb = 5 : ui32, name = #handshake.name<"cond_br19">} : i32
    %146 = buffer [1] seq %137#4 {bb = 5 : ui32, name = #handshake.name<"buffer67">} : i1
    %trueResult_35, %falseResult_36 = cond_br %146, %92#0 {bb = 5 : ui32, name = #handshake.name<"cond_br20">} : i32
    %147 = buffer [1] seq %result_15 {bb = 5 : ui32, name = #handshake.name<"buffer58">} : none
    %trueResult_37, %falseResult_38 = cond_br %137#3, %147 {bb = 5 : ui32, name = #handshake.name<"cond_br21">} : none
    %148 = source {bb = 6 : ui32, name = #handshake.name<"source5">}
    %149 = constant %148 {bb = 6 : ui32, name = #handshake.name<"constant25">, value = 2 : i3} : i3
    %150 = arith.extsi %149 {bb = 6 : ui32, name = #handshake.name<"extsi7">} : i3 to i32
    %151 = source {bb = 6 : ui32, name = #handshake.name<"source6">}
    %152 = constant %151 {bb = 6 : ui32, name = #handshake.name<"constant26">, value = 1 : i2} : i2
    %153 = arith.extsi %152 {bb = 6 : ui32, name = #handshake.name<"extsi8">} : i2 to i32
    %154 = arith.addi %falseResult_36, %153 {bb = 6 : ui32, name = #handshake.name<"addi10">} : i32
    %155 = buffer [1] seq %154 {bb = 6 : ui32, name = #handshake.name<"buffer69">} : i32
    %156:2 = fork [2] %155 {bb = 6 : ui32, name = #handshake.name<"fork21">} : i32
    %157 = buffer [1] fifo %156#0 {bb = 6 : ui32, name = #handshake.name<"buffer37">} : i32
    %158 = arith.trunci %157 {bb = 6 : ui32, name = #handshake.name<"trunci2">} : i32 to i2
    %159 = arith.cmpi ult, %156#1, %150 {bb = 6 : ui32, name = #handshake.name<"cmpi9">} : i32
    %160 = buffer [1] seq %159 {bb = 6 : ui32, name = #handshake.name<"buffer76">} : i1
    %161:8 = fork [8] %160 {bb = 6 : ui32, name = #handshake.name<"fork22">} : i1
    %trueResult_39, %falseResult_40 = cond_br %161#0, %158 {bb = 6 : ui32, name = #handshake.name<"cond_br5">} : i2
    sink %falseResult_40 {name = #handshake.name<"sink1">} : i2
    %162 = arith.extsi %trueResult_39 {bb = 6 : ui32, name = #handshake.name<"extsi11">} : i2 to i32
    %163 = buffer [2] fifo %161#7 {bb = 6 : ui32, name = #handshake.name<"buffer20">} : i1
    %164 = buffer [1] seq %falseResult_24 {bb = 6 : ui32, name = #handshake.name<"buffer64">} : i32
    %trueResult_41, %falseResult_42 = cond_br %163, %164 {bb = 6 : ui32, name = #handshake.name<"cond_br31">} : i32
    %165 = buffer [2] fifo %161#6 {bb = 6 : ui32, name = #handshake.name<"buffer18">} : i1
    %166 = buffer [1] seq %falseResult_26 {bb = 6 : ui32, name = #handshake.name<"buffer33">} : i32
    %trueResult_43, %falseResult_44 = cond_br %165, %166 {bb = 6 : ui32, name = #handshake.name<"cond_br32">} : i32
    %167 = buffer [1] seq %falseResult_28 {bb = 6 : ui32, name = #handshake.name<"buffer86">} : i5
    %trueResult_45, %falseResult_46 = cond_br %161#1, %167 {bb = 6 : ui32, name = #handshake.name<"cond_br6">} : i5
    %168 = buffer [1] seq %falseResult_30 {bb = 6 : ui32, name = #handshake.name<"buffer6">} : i5
    %trueResult_47, %falseResult_48 = cond_br %161#2, %168 {bb = 6 : ui32, name = #handshake.name<"cond_br9">} : i5
    %169 = buffer [1] fifo %161#5 {bb = 6 : ui32, name = #handshake.name<"buffer50">} : i1
    %170 = buffer [1] seq %falseResult_32 {bb = 6 : ui32, name = #handshake.name<"buffer88">} : i32
    %trueResult_49, %falseResult_50 = cond_br %169, %170 {bb = 6 : ui32, name = #handshake.name<"cond_br35">} : i32
    %171 = buffer [1] fifo %161#4 {bb = 6 : ui32, name = #handshake.name<"buffer25">} : i1
    %trueResult_51, %falseResult_52 = cond_br %171, %falseResult_34 {bb = 6 : ui32, name = #handshake.name<"cond_br36">} : i32
    sink %falseResult_52 {name = #handshake.name<"sink2">} : i32
    %172 = buffer [1] seq %falseResult_38 {bb = 6 : ui32, name = #handshake.name<"buffer73">} : none
    %trueResult_53, %falseResult_54 = cond_br %161#3, %172 {bb = 6 : ui32, name = #handshake.name<"cond_br37">} : none
    %173 = buffer [1] seq %falseResult_44 {bb = 7 : ui32, name = #handshake.name<"buffer52">} : i32
    %174:2 = fork [2] %173 {bb = 7 : ui32, name = #handshake.name<"fork23">} : i32
    %175 = buffer [1] seq %falseResult_42 {bb = 7 : ui32, name = #handshake.name<"buffer0">} : i32
    %176:2 = fork [2] %175 {bb = 7 : ui32, name = #handshake.name<"fork24">} : i32
    %177 = source {bb = 7 : ui32, name = #handshake.name<"source7">}
    %178 = constant %177 {bb = 7 : ui32, name = #handshake.name<"constant27">, value = 255 : i9} : i9
    %179 = arith.extsi %178 {bb = 7 : ui32, name = #handshake.name<"extsi9">} : i9 to i32
    %180:4 = fork [4] %179 {bb = 7 : ui32, name = #handshake.name<"fork25">} : i32
    %181 = source {bb = 7 : ui32, name = #handshake.name<"source8">}
    %182 = constant %181 {bb = 7 : ui32, name = #handshake.name<"constant28">, value = false} : i1
    %183 = arith.extsi %182 {bb = 7 : ui32, name = #handshake.name<"extsi10">} : i1 to i32
    %184:4 = fork [4] %183 {bb = 7 : ui32, name = #handshake.name<"fork26">} : i32
    %185 = arith.cmpi sgt, %176#1, %180#0 {bb = 7 : ui32, name = #handshake.name<"cmpi4">} : i32
    %186 = arith.cmpi sgt, %174#1, %180#1 {bb = 7 : ui32, name = #handshake.name<"cmpi5">} : i32
    %187 = arith.select %186, %180#2, %174#0 {bb = 7 : ui32, name = #handshake.name<"select1">} : i32
    %188 = buffer [1] seq %187 {bb = 7 : ui32, name = #handshake.name<"buffer47">} : i32
    %189:2 = fork [2] %188 {bb = 7 : ui32, name = #handshake.name<"fork27">} : i32
    %190 = arith.cmpi slt, %189#1, %184#0 {bb = 7 : ui32, name = #handshake.name<"cmpi6">} : i32
    %191 = arith.select %190, %184#1, %189#0 {bb = 7 : ui32, name = #handshake.name<"select2">} : i32
    %192 = arith.select %185, %180#3, %176#0 {bb = 7 : ui32, name = #handshake.name<"select3">} : i32
    %193 = buffer [1] seq %192 {bb = 7 : ui32, name = #handshake.name<"buffer26">} : i32
    %194:2 = fork [2] %193 {bb = 7 : ui32, name = #handshake.name<"fork28">} : i32
    %195 = arith.cmpi slt, %194#1, %184#2 {bb = 7 : ui32, name = #handshake.name<"cmpi7">} : i32
    %196 = arith.select %195, %184#3, %194#0 {bb = 7 : ui32, name = #handshake.name<"select4">} : i32
    %197 = buffer [1] seq %196 {bb = 7 : ui32, name = #handshake.name<"buffer15">} : i32
    %198 = buffer [1] seq %191 {bb = 7 : ui32, name = #handshake.name<"buffer60">} : i32
    %199 = arith.addi %198, %197 {bb = 7 : ui32, name = #handshake.name<"addi2">} : i32
    %200 = buffer [1] seq %falseResult_50 {bb = 7 : ui32, name = #handshake.name<"buffer8">} : i32
    %201 = buffer [1] seq %199 {bb = 7 : ui32, name = #handshake.name<"buffer29">} : i32
    %202 = arith.addi %200, %201 {bb = 7 : ui32, name = #handshake.name<"addi3">} : i32
    %203 = buffer [2] fifo %215#2 {bb = 8 : ui32, name = #handshake.name<"buffer10">} : i1
    %204 = mux %203 [%202, %falseResult_10] {bb = 8 : ui32, name = #handshake.name<"mux20">} : i1, i32
    %205 = buffer [2] seq %204 {bb = 8 : ui32, name = #handshake.name<"buffer84">} : i32
    %206:2 = fork [2] %205 {bb = 8 : ui32, name = #handshake.name<"fork29">} : i32
    %207 = mux %215#0 [%falseResult_46, %falseResult] {bb = 8 : ui32, name = #handshake.name<"mux16">} : i1, i5
    %208 = buffer [2] seq %207 {bb = 8 : ui32, name = #handshake.name<"buffer32">} : i5
    %209:2 = fork [2] %208 {bb = 8 : ui32, name = #handshake.name<"fork30">} : i5
    %210 = arith.extsi %209#1 {bb = 8 : ui32, name = #handshake.name<"extsi12">} : i5 to i6
    %211 = mux %215#1 [%falseResult_48, %falseResult_8] {bb = 8 : ui32, name = #handshake.name<"mux21">} : i1, i5
    %212:2 = fork [2] %211 {bb = 8 : ui32, name = #handshake.name<"fork31">} : i5
    %213 = arith.extsi %212#0 {bb = 8 : ui32, name = #handshake.name<"extsi13">} : i5 to i6
    %214 = arith.extsi %212#1 {bb = 8 : ui32, name = #handshake.name<"extsi22">} : i5 to i6
    %result_55, %index_56 = control_merge %falseResult_54, %falseResult_12 {bb = 8 : ui32, name = #handshake.name<"control_merge4">} : none, i1
    %215:3 = fork [3] %index_56 {bb = 8 : ui32, name = #handshake.name<"fork32">} : i1
    %216:2 = fork [2] %result_55 {bb = 8 : ui32, name = #handshake.name<"fork33">} : none
    %217 = source {bb = 8 : ui32, name = #handshake.name<"source9">}
    %218 = constant %217 {bb = 8 : ui32, name = #handshake.name<"constant29">, value = 1 : i2} : i2
    %219 = arith.extsi %218 {bb = 8 : ui32, name = #handshake.name<"extsi23">} : i2 to i6
    %220 = source {bb = 8 : ui32, name = #handshake.name<"source10">}
    %221 = constant %220 {bb = 8 : ui32, name = #handshake.name<"constant30">, value = 15 : i5} : i5
    %222 = arith.extsi %221 {bb = 8 : ui32, name = #handshake.name<"extsi14">} : i5 to i6
    %223 = source {bb = 8 : ui32, name = #handshake.name<"source11">}
    %224 = constant %223 {bb = 8 : ui32, name = #handshake.name<"constant31">, value = 255 : i9} : i9
    %225 = arith.extsi %224 {bb = 8 : ui32, name = #handshake.name<"extsi24">} : i9 to i10
    %226 = arith.trunci %206#1 {bb = 8 : ui32, name = #handshake.name<"trunci0">} : i32 to i8
    %227 = arith.extui %226 {bb = 8 : ui32, name = #handshake.name<"extui1">} : i8 to i10
    %228 = arith.subi %225, %227 {bb = 8 : ui32, name = #handshake.name<"subi1">} : i10
    %229 = arith.extsi %228 {bb = 8 : ui32, name = #handshake.name<"extsi25">} : i10 to i32
    %230 = buffer [2] seq %213 {bb = 8 : ui32, name = #handshake.name<"buffer35">} : i6
    %231 = arith.addi %230, %210 {bb = 8 : ui32, name = #handshake.name<"addi8">} : i6
    %232 = buffer [1] fifo %231 {bb = 8 : ui32, name = #handshake.name<"buffer13">} : i6
    %233 = arith.extsi %232 {bb = 8 : ui32, name = #handshake.name<"extsi26">} : i6 to i32
    %addressResult_57, %dataResult_58 = lsq_store[%233] %229 {bb = 8 : ui32, name = #handshake.name<"lsq_store0">} : i32, i32
    %234 = buffer [1] seq %219 {bb = 8 : ui32, name = #handshake.name<"buffer48">} : i6
    %235 = buffer [1] seq %214 {bb = 8 : ui32, name = #handshake.name<"buffer90">} : i6
    %236 = arith.addi %235, %234 {bb = 8 : ui32, name = #handshake.name<"addi11">} : i6
    %237:2 = fork [2] %236 {bb = 8 : ui32, name = #handshake.name<"fork34">} : i6
    %238 = arith.trunci %237#0 {bb = 8 : ui32, name = #handshake.name<"trunci4">} : i6 to i5
    %239 = arith.cmpi ult, %237#1, %222 {bb = 8 : ui32, name = #handshake.name<"cmpi10">} : i6
    %240:4 = fork [4] %239 {bb = 8 : ui32, name = #handshake.name<"fork35">} : i1
    %trueResult_59, %falseResult_60 = cond_br %240#0, %238 {bb = 8 : ui32, name = #handshake.name<"cond_br10">} : i5
    sink %falseResult_60 {name = #handshake.name<"sink3">} : i5
    %241 = buffer [2] seq %240#2 {bb = 8 : ui32, name = #handshake.name<"buffer75">} : i1
    %trueResult_61, %falseResult_62 = cond_br %241, %206#0 {bb = 8 : ui32, name = #handshake.name<"cond_br45">} : i32
    %242 = buffer [1] seq %240#1 {bb = 8 : ui32, name = #handshake.name<"buffer63">} : i1
    %trueResult_63, %falseResult_64 = cond_br %242, %209#0 {bb = 8 : ui32, name = #handshake.name<"cond_br11">} : i5
    %243 = buffer [1] fifo %216#1 {bb = 8 : ui32, name = #handshake.name<"buffer31">} : none
    %trueResult_65, %falseResult_66 = cond_br %240#3, %243 {bb = 8 : ui32, name = #handshake.name<"cond_br47">} : none
    %244 = arith.extsi %falseResult_64 {bb = 9 : ui32, name = #handshake.name<"extsi15">} : i5 to i6
    %245 = buffer [1] seq %falseResult_66 {bb = 9 : ui32, name = #handshake.name<"buffer23">} : none
    %246 = source {bb = 9 : ui32, name = #handshake.name<"source12">}
    %247 = constant %246 {bb = 9 : ui32, name = #handshake.name<"constant32">, value = 1 : i2} : i2
    %248 = arith.extsi %247 {bb = 9 : ui32, name = #handshake.name<"extsi27">} : i2 to i6
    %249 = source {bb = 9 : ui32, name = #handshake.name<"source13">}
    %250 = constant %249 {bb = 9 : ui32, name = #handshake.name<"constant33">, value = 15 : i5} : i5
    %251 = arith.extsi %250 {bb = 9 : ui32, name = #handshake.name<"extsi28">} : i5 to i6
    %252 = arith.addi %244, %248 {bb = 9 : ui32, name = #handshake.name<"addi12">} : i6
    %253 = buffer [1] seq %252 {bb = 9 : ui32, name = #handshake.name<"buffer65">} : i6
    %254:2 = fork [2] %253 {bb = 9 : ui32, name = #handshake.name<"fork36">} : i6
    %255 = arith.trunci %254#0 {bb = 9 : ui32, name = #handshake.name<"trunci3">} : i6 to i5
    %256 = buffer [1] seq %251 {bb = 9 : ui32, name = #handshake.name<"buffer14">} : i6
    %257 = arith.cmpi ult, %254#1, %256 {bb = 9 : ui32, name = #handshake.name<"cmpi11">} : i6
    %258:3 = fork [3] %257 {bb = 9 : ui32, name = #handshake.name<"fork37">} : i1
    %trueResult_67, %falseResult_68 = cond_br %258#0, %255 {bb = 9 : ui32, name = #handshake.name<"cond_br12">} : i5
    sink %falseResult_68 {name = #handshake.name<"sink4">} : i5
    %trueResult_69, %falseResult_70 = cond_br %258#1, %falseResult_62 {bb = 9 : ui32, name = #handshake.name<"cond_br52">} : i32
    %trueResult_71, %falseResult_72 = cond_br %258#2, %245 {bb = 9 : ui32, name = #handshake.name<"cond_br53">} : none
    sink %falseResult_72 {name = #handshake.name<"sink5">} : none
    %259 = buffer [1] seq %falseResult_70 {bb = 10 : ui32, name = #handshake.name<"buffer9">} : i32
    %260 = d_return {bb = 10 : ui32, name = #handshake.name<"d_return0">} %259 : i32
    end {bb = 10 : ui32, bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00}>, name = #handshake.name<"end0">} %260, %done, %done_0, %done_2, %done_4 : i32, none, none, none, none
  }
}

