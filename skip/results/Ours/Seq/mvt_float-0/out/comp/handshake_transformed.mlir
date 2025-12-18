module {
  handshake.func @mvt_float(%arg0: memref<900xf32>, %arg1: memref<30xf32>, %arg2: memref<30xf32>, %arg3: memref<30xf32>, %arg4: memref<30xf32>, %arg5: !handshake.control<>, %arg6: !handshake.control<>, %arg7: !handshake.control<>, %arg8: !handshake.control<>, %arg9: !handshake.control<>, %arg10: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["A", "x1", "x2", "y1", "y2", "A_start", "x1_start", "x2_start", "y1_start", "y2_start", "start"], cfg.edges = "[0,1][2,2,3,cmpi0][4,5][6,4,7,cmpi3][1,2][3,1,4,cmpi1][5,5,6,cmpi2]", resNames = ["A_end", "x1_end", "x2_end", "y1_end", "y2_end", "end"]} {
    %0:3 = fork [3] %arg10 {handshake.bb = 0 : ui32, handshake.name = "fork0"} : <>
    %outputs, %memEnd = mem_controller[%arg4 : memref<30xf32>] %arg9 (%addressResult_38) %188#4 {connectedBlocks = [5 : i32], handshake.name = "mem_controller3"} :    (!handshake.channel<i5>) -> !handshake.channel<f32>
    %outputs_0, %memEnd_1 = mem_controller[%arg3 : memref<30xf32>] %arg8 (%addressResult_12) %188#3 {connectedBlocks = [2 : i32], handshake.name = "mem_controller4"} :    (!handshake.channel<i5>) -> !handshake.channel<f32>
    %outputs_2:2, %memEnd_3 = mem_controller[%arg2 : memref<30xf32>] %arg7 (%addressResult_32, %170, %addressResult_50, %dataResult_51) %188#2 {connectedBlocks = [4 : i32, 6 : i32], handshake.name = "mem_controller5"} :    (!handshake.channel<i5>, !handshake.channel<i32>, !handshake.channel<i5>, !handshake.channel<f32>) -> (!handshake.channel<f32>, !handshake.control<>)
    %outputs_4:2, %memEnd_5 = mem_controller[%arg1 : memref<30xf32>] %arg6 (%addressResult, %77, %addressResult_22, %dataResult_23) %188#1 {connectedBlocks = [1 : i32, 3 : i32], handshake.name = "mem_controller6"} :    (!handshake.channel<i5>, !handshake.channel<i32>, !handshake.channel<i5>, !handshake.channel<f32>) -> (!handshake.channel<f32>, !handshake.control<>)
    %outputs_6:2, %memEnd_7 = mem_controller[%arg0 : memref<900xf32>] %arg5 (%addressResult_10, %addressResult_36) %188#0 {connectedBlocks = [2 : i32, 5 : i32], handshake.name = "mem_controller7"} :    (!handshake.channel<i10>, !handshake.channel<i10>) -> (!handshake.channel<f32>, !handshake.channel<f32>)
    %1 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant14", value = false} : <>, <i1>
    %2 = br %1 {handshake.bb = 0 : ui32, handshake.name = "br2"} : <i1>
    %3 = extsi %2 {handshake.bb = 0 : ui32, handshake.name = "extsi17"} : <i1> to <i6>
    %4 = br %0#2 {handshake.bb = 0 : ui32, handshake.name = "br4"} : <>
    %5 = init %93#1 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "init0"} : <i1>
    sink %5 {handshake.name = "sink0"} : <i1>
    %7:2 = unbundle %16#0  {handshake.bb = 1 : ui32, handshake.name = "unbundle0"} : <f32> to _ 
    %9 = mux %index [%3, %trueResult_24] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i6>, <i6>] to <i6>
    %10:2 = fork [2] %9 {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <i6>
    %11 = trunci %10#0 {handshake.bb = 1 : ui32, handshake.name = "trunci0"} : <i6> to <i5>
    %result, %index = control_merge [%4, %trueResult_26]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %13:2 = fork [2] %result {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <>
    %14 = constant %13#0 {handshake.bb = 1 : ui32, handshake.name = "constant15", value = false} : <>, <i1>
    %15 = buffer %7#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer0"} : <>
    %addressResult, %dataResult = load[%11] %outputs_4#0 {handshake.bb = 1 : ui32, handshake.deps = #handshake<deps[["store0", 2, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i5>, <f32>, <i5>, <f32>
    %16:2 = fork [2] %dataResult {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <f32>
    %17 = br %14 {handshake.bb = 1 : ui32, handshake.name = "br5"} : <i1>
    %18 = extsi %17 {handshake.bb = 1 : ui32, handshake.name = "extsi16"} : <i1> to <i6>
    %19 = br %16#1 {handshake.bb = 1 : ui32, handshake.name = "br6"} : <f32>
    %21 = br %10#1 {handshake.bb = 1 : ui32, handshake.name = "br7"} : <i6>
    %23 = br %13#1 {handshake.bb = 1 : ui32, handshake.name = "br8"} : <>
    %24 = mux %40#1 [%18, %trueResult] {handshake.bb = 2 : ui32, handshake.name = "mux1"} : <i1>, [<i6>, <i6>] to <i6>
    %26:3 = fork [3] %24 {handshake.bb = 2 : ui32, handshake.name = "fork4"} : <i6>
    %27 = extsi %28 {handshake.bb = 2 : ui32, handshake.name = "extsi18"} : <i6> to <i10>
    %28 = buffer %26#0, bufferType = FIFO_BREAK_NONE, numSlots = 9 {handshake.bb = 2 : ui32, handshake.name = "buffer8"} : <i6>
    %29 = extsi %26#2 {handshake.bb = 2 : ui32, handshake.name = "extsi19"} : <i6> to <i7>
    %31 = trunci %32 {handshake.bb = 2 : ui32, handshake.name = "trunci1"} : <i6> to <i5>
    %32 = buffer %26#1, bufferType = FIFO_BREAK_NONE, numSlots = 9 {handshake.bb = 2 : ui32, handshake.name = "buffer10"} : <i6>
    %33 = mux %34 [%19, %trueResult_14] {handshake.bb = 2 : ui32, handshake.name = "mux2"} : <i1>, [<f32>, <f32>] to <f32>
    %34 = buffer %40#2, bufferType = FIFO_BREAK_NONE, numSlots = 15 {handshake.bb = 2 : ui32, handshake.name = "buffer11"} : <i1>
    %35 = mux %40#0 [%21, %trueResult_16] {handshake.bb = 2 : ui32, handshake.name = "mux3"} : <i1>, [<i6>, <i6>] to <i6>
    %37:2 = fork [2] %35 {handshake.bb = 2 : ui32, handshake.name = "fork5"} : <i6>
    %38 = extsi %39 {handshake.bb = 2 : ui32, handshake.name = "extsi20"} : <i6> to <i12>
    %39 = buffer %37#1, bufferType = FIFO_BREAK_NONE, numSlots = 5 {handshake.bb = 2 : ui32, handshake.name = "buffer13"} : <i6>
    %result_8, %index_9 = control_merge [%23, %trueResult_18]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>, <>] to <>, <i1>
    %40:3 = fork [3] %index_9 {handshake.bb = 2 : ui32, handshake.name = "fork6"} : <i1>
    %41 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %42 = constant %41 {handshake.bb = 2 : ui32, handshake.name = "constant16", value = 30 : i6} : <>, <i6>
    %43:2 = fork [2] %42 {handshake.bb = 2 : ui32, handshake.name = "fork7"} : <i6>
    %44 = extsi %45 {handshake.bb = 2 : ui32, handshake.name = "extsi21"} : <i6> to <i12>
    %45 = buffer %43#0, bufferType = FIFO_BREAK_NONE, numSlots = 5 {handshake.bb = 2 : ui32, handshake.name = "buffer14"} : <i6>
    %46 = extsi %43#1 {handshake.bb = 2 : ui32, handshake.name = "extsi22"} : <i6> to <i7>
    %48 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %49 = constant %48 {handshake.bb = 2 : ui32, handshake.name = "constant17", value = 1 : i2} : <>, <i2>
    %50 = extsi %49 {handshake.bb = 2 : ui32, handshake.name = "extsi23"} : <i2> to <i7>
    %51 = muli %38, %44 {handshake.bb = 2 : ui32, handshake.name = "muli0"} : <i12>
    %52 = trunci %51 {handshake.bb = 2 : ui32, handshake.name = "trunci2"} : <i12> to <i10>
    %53 = addi %27, %52 {handshake.bb = 2 : ui32, handshake.name = "addi0"} : <i10>
    %addressResult_10, %dataResult_11 = load[%53] %outputs_6#0 {handshake.bb = 2 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i10>, <f32>, <i10>, <f32>
    %addressResult_12, %dataResult_13 = load[%31] %outputs_0 {handshake.bb = 2 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load2"} : <i5>, <f32>, <i5>, <f32>
    %54 = mulf %dataResult_11, %dataResult_13 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "mulf0"} : <f32>
    %55 = addf %33, %54 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "addf0"} : <f32>
    %56 = addi %29, %50 {handshake.bb = 2 : ui32, handshake.name = "addi2"} : <i7>
    %57:2 = fork [2] %56 {handshake.bb = 2 : ui32, handshake.name = "fork8"} : <i7>
    %58 = trunci %57#0 {handshake.bb = 2 : ui32, handshake.name = "trunci3"} : <i7> to <i6>
    %60 = cmpi ult, %57#1, %46 {handshake.bb = 2 : ui32, handshake.name = "cmpi0"} : <i7>
    %62:4 = fork [4] %60 {handshake.bb = 2 : ui32, handshake.name = "fork9"} : <i1>
    %trueResult, %falseResult = cond_br %62#0, %58 {handshake.bb = 2 : ui32, handshake.name = "cond_br4"} : <i1>, <i6>
    sink %falseResult {handshake.name = "sink1"} : <i6>
    %trueResult_14, %falseResult_15 = cond_br %64, %55 {handshake.bb = 2 : ui32, handshake.name = "cond_br5"} : <i1>, <f32>
    %64 = buffer %62#2, bufferType = FIFO_BREAK_NONE, numSlots = 15 {handshake.bb = 2 : ui32, handshake.name = "buffer19"} : <i1>
    %trueResult_16, %falseResult_17 = cond_br %62#1, %37#0 {handshake.bb = 2 : ui32, handshake.name = "cond_br6"} : <i1>, <i6>
    %trueResult_18, %falseResult_19 = cond_br %62#3, %result_8 {handshake.bb = 2 : ui32, handshake.name = "cond_br7"} : <i1>, <>
    %68 = merge %falseResult_17 {handshake.bb = 3 : ui32, handshake.name = "merge0"} : <i6>
    %69:2 = fork [2] %68 {handshake.bb = 3 : ui32, handshake.name = "fork10"} : <i6>
    %70 = extsi %69#0 {handshake.bb = 3 : ui32, handshake.name = "extsi24"} : <i6> to <i7>
    %72 = extsi %73 {handshake.bb = 3 : ui32, handshake.name = "extsi25"} : <i6> to <i32>
    %73 = buffer %69#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer24"} : <i6>
    %74 = merge %falseResult_15 {handshake.bb = 3 : ui32, handshake.name = "merge1"} : <f32>
    %result_20, %index_21 = control_merge [%falseResult_19]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>] to <>, <i1>
    sink %index_21 {handshake.name = "sink2"} : <i1>
    %75:3 = fork [3] %result_20 {handshake.bb = 3 : ui32, handshake.name = "fork11"} : <>
    %76 = constant %75#1 {handshake.bb = 3 : ui32, handshake.name = "constant18", value = 1 : i2} : <>, <i2>
    %77 = extsi %76 {handshake.bb = 3 : ui32, handshake.name = "extsi4"} : <i2> to <i32>
    %78 = constant %75#0 {handshake.bb = 3 : ui32, handshake.name = "constant19", value = false} : <>, <i1>
    %79 = source {handshake.bb = 3 : ui32, handshake.name = "source2"} : <>
    %80 = constant %79 {handshake.bb = 3 : ui32, handshake.name = "constant20", value = 30 : i6} : <>, <i6>
    %81 = extsi %80 {handshake.bb = 3 : ui32, handshake.name = "extsi26"} : <i6> to <i7>
    %82 = source {handshake.bb = 3 : ui32, handshake.name = "source3"} : <>
    %83 = constant %82 {handshake.bb = 3 : ui32, handshake.name = "constant21", value = 1 : i2} : <>, <i2>
    %84 = extsi %83 {handshake.bb = 3 : ui32, handshake.name = "extsi27"} : <i2> to <i7>
    %85 = gate %72, %15 {handshake.bb = 3 : ui32, handshake.name = "gate0"} : <i32>, !handshake.control<> to <i32>
    %86 = trunci %85 {handshake.bb = 3 : ui32, handshake.name = "trunci4"} : <i32> to <i5>
    %addressResult_22, %dataResult_23, %doneResult = store[%86] %74 %outputs_4#1 {handshake.bb = 3 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store0"} : <i5>, <f32>, <>, <i5>, <f32>, <>
    sink %doneResult {handshake.name = "sink3"} : <>
    %87 = addi %70, %84 {handshake.bb = 3 : ui32, handshake.name = "addi3"} : <i7>
    %88:2 = fork [2] %87 {handshake.bb = 3 : ui32, handshake.name = "fork12"} : <i7>
    %89 = trunci %88#0 {handshake.bb = 3 : ui32, handshake.name = "trunci5"} : <i7> to <i6>
    %91 = cmpi ult, %88#1, %81 {handshake.bb = 3 : ui32, handshake.name = "cmpi1"} : <i7>
    %93:4 = fork [4] %91 {handshake.bb = 3 : ui32, handshake.name = "fork13"} : <i1>
    %trueResult_24, %falseResult_25 = cond_br %93#0, %89 {handshake.bb = 3 : ui32, handshake.name = "cond_br8"} : <i1>, <i6>
    sink %falseResult_25 {handshake.name = "sink4"} : <i6>
    %trueResult_26, %falseResult_27 = cond_br %93#2, %75#2 {handshake.bb = 3 : ui32, handshake.name = "cond_br9"} : <i1>, <>
    %trueResult_28, %falseResult_29 = cond_br %93#3, %78 {handshake.bb = 3 : ui32, handshake.name = "cond_br10"} : <i1>, <i1>
    sink %trueResult_28 {handshake.name = "sink5"} : <i1>
    %97 = extsi %falseResult_29 {handshake.bb = 3 : ui32, handshake.name = "extsi15"} : <i1> to <i6>
    %98 = init %185#1 {ftd.imerge, handshake.bb = 4 : ui32, handshake.name = "init1"} : <i1>
    sink %98 {handshake.name = "sink6"} : <i1>
    %100:2 = unbundle %109#0  {handshake.bb = 4 : ui32, handshake.name = "unbundle2"} : <f32> to _ 
    %102 = mux %index_31 [%97, %trueResult_53] {handshake.bb = 4 : ui32, handshake.name = "mux4"} : <i1>, [<i6>, <i6>] to <i6>
    %103:2 = fork [2] %102 {handshake.bb = 4 : ui32, handshake.name = "fork14"} : <i6>
    %104 = trunci %103#0 {handshake.bb = 4 : ui32, handshake.name = "trunci6"} : <i6> to <i5>
    %result_30, %index_31 = control_merge [%falseResult_27, %trueResult_55]  {handshake.bb = 4 : ui32, handshake.name = "control_merge3"} : [<>, <>] to <>, <i1>
    %106:2 = fork [2] %result_30 {handshake.bb = 4 : ui32, handshake.name = "fork15"} : <>
    %107 = constant %106#0 {handshake.bb = 4 : ui32, handshake.name = "constant22", value = false} : <>, <i1>
    %108 = buffer %100#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer1"} : <>
    %addressResult_32, %dataResult_33 = load[%104] %outputs_2#0 {handshake.bb = 4 : ui32, handshake.deps = #handshake<deps[["store1", 2, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load3"} : <i5>, <f32>, <i5>, <f32>
    %109:2 = fork [2] %dataResult_33 {handshake.bb = 4 : ui32, handshake.name = "fork16"} : <f32>
    %110 = br %107 {handshake.bb = 4 : ui32, handshake.name = "br9"} : <i1>
    %111 = extsi %110 {handshake.bb = 4 : ui32, handshake.name = "extsi14"} : <i1> to <i6>
    %112 = br %109#1 {handshake.bb = 4 : ui32, handshake.name = "br10"} : <f32>
    %114 = br %103#1 {handshake.bb = 4 : ui32, handshake.name = "br11"} : <i6>
    %116 = br %106#1 {handshake.bb = 4 : ui32, handshake.name = "br12"} : <>
    %117 = mux %133#1 [%111, %trueResult_40] {handshake.bb = 5 : ui32, handshake.name = "mux5"} : <i1>, [<i6>, <i6>] to <i6>
    %119:3 = fork [3] %117 {handshake.bb = 5 : ui32, handshake.name = "fork17"} : <i6>
    %120 = extsi %119#1 {handshake.bb = 5 : ui32, handshake.name = "extsi28"} : <i6> to <i7>
    %122 = extsi %119#2 {handshake.bb = 5 : ui32, handshake.name = "extsi29"} : <i6> to <i12>
    %124 = trunci %119#0 {handshake.bb = 5 : ui32, handshake.name = "trunci7"} : <i6> to <i5>
    %126 = mux %127 [%112, %trueResult_42] {handshake.bb = 5 : ui32, handshake.name = "mux6"} : <i1>, [<f32>, <f32>] to <f32>
    %127 = buffer %133#2, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 5 : ui32, handshake.name = "buffer39"} : <i1>
    %128 = mux %133#0 [%114, %trueResult_44] {handshake.bb = 5 : ui32, handshake.name = "mux7"} : <i1>, [<i6>, <i6>] to <i6>
    %130:2 = fork [2] %128 {handshake.bb = 5 : ui32, handshake.name = "fork18"} : <i6>
    %131 = extsi %130#0 {handshake.bb = 5 : ui32, handshake.name = "extsi30"} : <i6> to <i10>
    %result_34, %index_35 = control_merge [%116, %trueResult_46]  {handshake.bb = 5 : ui32, handshake.name = "control_merge4"} : [<>, <>] to <>, <i1>
    %133:3 = fork [3] %index_35 {handshake.bb = 5 : ui32, handshake.name = "fork19"} : <i1>
    %134 = source {handshake.bb = 5 : ui32, handshake.name = "source4"} : <>
    %135 = constant %134 {handshake.bb = 5 : ui32, handshake.name = "constant23", value = 30 : i6} : <>, <i6>
    %136:2 = fork [2] %135 {handshake.bb = 5 : ui32, handshake.name = "fork20"} : <i6>
    %137 = extsi %136#0 {handshake.bb = 5 : ui32, handshake.name = "extsi31"} : <i6> to <i7>
    %139 = extsi %136#1 {handshake.bb = 5 : ui32, handshake.name = "extsi32"} : <i6> to <i12>
    %141 = source {handshake.bb = 5 : ui32, handshake.name = "source5"} : <>
    %142 = constant %141 {handshake.bb = 5 : ui32, handshake.name = "constant24", value = 1 : i2} : <>, <i2>
    %143 = extsi %142 {handshake.bb = 5 : ui32, handshake.name = "extsi33"} : <i2> to <i7>
    %144 = muli %122, %139 {handshake.bb = 5 : ui32, handshake.name = "muli1"} : <i12>
    %145 = trunci %144 {handshake.bb = 5 : ui32, handshake.name = "trunci8"} : <i12> to <i10>
    %146 = addi %131, %145 {handshake.bb = 5 : ui32, handshake.name = "addi1"} : <i10>
    %addressResult_36, %dataResult_37 = load[%146] %outputs_6#1 {handshake.bb = 5 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load4"} : <i10>, <f32>, <i10>, <f32>
    %addressResult_38, %dataResult_39 = load[%124] %outputs {handshake.bb = 5 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load5"} : <i5>, <f32>, <i5>, <f32>
    %147 = mulf %dataResult_37, %dataResult_39 {fastmath = #arith.fastmath<none>, handshake.bb = 5 : ui32, handshake.name = "mulf1"} : <f32>
    %148 = addf %126, %147 {fastmath = #arith.fastmath<none>, handshake.bb = 5 : ui32, handshake.name = "addf1"} : <f32>
    %149 = addi %120, %143 {handshake.bb = 5 : ui32, handshake.name = "addi4"} : <i7>
    %150:2 = fork [2] %149 {handshake.bb = 5 : ui32, handshake.name = "fork21"} : <i7>
    %151 = trunci %150#0 {handshake.bb = 5 : ui32, handshake.name = "trunci9"} : <i7> to <i6>
    %153 = cmpi ult, %150#1, %137 {handshake.bb = 5 : ui32, handshake.name = "cmpi2"} : <i7>
    %155:4 = fork [4] %153 {handshake.bb = 5 : ui32, handshake.name = "fork22"} : <i1>
    %trueResult_40, %falseResult_41 = cond_br %155#0, %151 {handshake.bb = 5 : ui32, handshake.name = "cond_br11"} : <i1>, <i6>
    sink %falseResult_41 {handshake.name = "sink7"} : <i6>
    %trueResult_42, %falseResult_43 = cond_br %157, %148 {handshake.bb = 5 : ui32, handshake.name = "cond_br12"} : <i1>, <f32>
    %157 = buffer %155#2, bufferType = FIFO_BREAK_NONE, numSlots = 3 {handshake.bb = 5 : ui32, handshake.name = "buffer47"} : <i1>
    %trueResult_44, %falseResult_45 = cond_br %155#1, %130#1 {handshake.bb = 5 : ui32, handshake.name = "cond_br13"} : <i1>, <i6>
    %trueResult_46, %falseResult_47 = cond_br %155#3, %result_34 {handshake.bb = 5 : ui32, handshake.name = "cond_br14"} : <i1>, <>
    %161 = merge %falseResult_45 {handshake.bb = 6 : ui32, handshake.name = "merge2"} : <i6>
    %162:2 = fork [2] %161 {handshake.bb = 6 : ui32, handshake.name = "fork23"} : <i6>
    %163 = extsi %162#0 {handshake.bb = 6 : ui32, handshake.name = "extsi34"} : <i6> to <i7>
    %165 = extsi %166 {handshake.bb = 6 : ui32, handshake.name = "extsi35"} : <i6> to <i32>
    %166 = buffer %162#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 6 : ui32, handshake.name = "buffer52"} : <i6>
    %167 = merge %falseResult_43 {handshake.bb = 6 : ui32, handshake.name = "merge3"} : <f32>
    %result_48, %index_49 = control_merge [%falseResult_47]  {handshake.bb = 6 : ui32, handshake.name = "control_merge5"} : [<>] to <>, <i1>
    sink %index_49 {handshake.name = "sink8"} : <i1>
    %168:2 = fork [2] %result_48 {handshake.bb = 6 : ui32, handshake.name = "fork24"} : <>
    %169 = constant %168#0 {handshake.bb = 6 : ui32, handshake.name = "constant25", value = 1 : i2} : <>, <i2>
    %170 = extsi %169 {handshake.bb = 6 : ui32, handshake.name = "extsi11"} : <i2> to <i32>
    %171 = source {handshake.bb = 6 : ui32, handshake.name = "source6"} : <>
    %172 = constant %171 {handshake.bb = 6 : ui32, handshake.name = "constant26", value = 30 : i6} : <>, <i6>
    %173 = extsi %172 {handshake.bb = 6 : ui32, handshake.name = "extsi36"} : <i6> to <i7>
    %174 = source {handshake.bb = 6 : ui32, handshake.name = "source7"} : <>
    %175 = constant %174 {handshake.bb = 6 : ui32, handshake.name = "constant27", value = 1 : i2} : <>, <i2>
    %176 = extsi %175 {handshake.bb = 6 : ui32, handshake.name = "extsi37"} : <i2> to <i7>
    %177 = gate %165, %108 {handshake.bb = 6 : ui32, handshake.name = "gate1"} : <i32>, !handshake.control<> to <i32>
    %178 = trunci %177 {handshake.bb = 6 : ui32, handshake.name = "trunci10"} : <i32> to <i5>
    %addressResult_50, %dataResult_51, %doneResult_52 = store[%178] %167 %outputs_2#1 {handshake.bb = 6 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store1"} : <i5>, <f32>, <>, <i5>, <f32>, <>
    sink %doneResult_52 {handshake.name = "sink9"} : <>
    %179 = addi %163, %176 {handshake.bb = 6 : ui32, handshake.name = "addi5"} : <i7>
    %180:2 = fork [2] %179 {handshake.bb = 6 : ui32, handshake.name = "fork25"} : <i7>
    %181 = trunci %180#0 {handshake.bb = 6 : ui32, handshake.name = "trunci11"} : <i7> to <i6>
    %183 = cmpi ult, %180#1, %173 {handshake.bb = 6 : ui32, handshake.name = "cmpi3"} : <i7>
    %185:3 = fork [3] %183 {handshake.bb = 6 : ui32, handshake.name = "fork26"} : <i1>
    %trueResult_53, %falseResult_54 = cond_br %185#0, %181 {handshake.bb = 6 : ui32, handshake.name = "cond_br15"} : <i1>, <i6>
    sink %falseResult_54 {handshake.name = "sink10"} : <i6>
    %trueResult_55, %falseResult_56 = cond_br %185#2, %168#1 {handshake.bb = 6 : ui32, handshake.name = "cond_br16"} : <i1>, <>
    %result_57, %index_58 = control_merge [%falseResult_56]  {handshake.bb = 7 : ui32, handshake.name = "control_merge6"} : [<>] to <>, <i1>
    sink %index_58 {handshake.name = "sink11"} : <i1>
    %188:5 = fork [5] %result_57 {handshake.bb = 7 : ui32, handshake.name = "fork27"} : <>
    end {handshake.bb = 7 : ui32, handshake.name = "end0"} %memEnd_7, %memEnd_5, %memEnd_3, %memEnd_1, %memEnd, %0#1 : <>, <>, <>, <>, <>, <>
  }
}

