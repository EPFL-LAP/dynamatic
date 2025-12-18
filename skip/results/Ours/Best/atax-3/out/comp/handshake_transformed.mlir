module {
  handshake.func @atax(%arg0: memref<400xf32>, %arg1: memref<20xf32>, %arg2: memref<20xf32>, %arg3: memref<20xf32>, %arg4: !handshake.control<>, %arg5: !handshake.control<>, %arg6: !handshake.control<>, %arg7: !handshake.control<>, %arg8: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["A", "x", "y", "tmp", "A_start", "x_start", "y_start", "tmp_start", "start"], cfg.edges = "[0,1][2,2,3,cmpi0][4,1,5,cmpi2][1,2][3,3,4,cmpi1]", resNames = ["A_end", "x_end", "y_end", "tmp_end", "end"]} {
    %0:8 = fork [8] %arg8 {handshake.bb = 0 : ui32, handshake.name = "fork0"} : <>
    %outputs:2, %memEnd = mem_controller[%arg3 : memref<20xf32>] %arg7 (%addressResult, %268, %addressResult_78, %dataResult_79) %314#3 {connectedBlocks = [1 : i32, 4 : i32], handshake.name = "mem_controller2"} :    (!handshake.channel<i5>, !handshake.channel<i32>, !handshake.channel<i5>, !handshake.channel<f32>) -> (!handshake.channel<f32>, !handshake.control<>)
    %outputs_0:2, %memEnd_1 = mem_controller[%arg2 : memref<20xf32>] %arg6 (%169, %addressResult_42, %addressResult_46, %dataResult_47) %314#2 {connectedBlocks = [3 : i32], handshake.name = "mem_controller3"} :    (!handshake.channel<i32>, !handshake.channel<i5>, !handshake.channel<i5>, !handshake.channel<f32>) -> (!handshake.channel<f32>, !handshake.control<>)
    %outputs_2, %memEnd_3 = mem_controller[%arg1 : memref<20xf32>] %arg5 (%addressResult_10) %314#1 {connectedBlocks = [2 : i32], handshake.name = "mem_controller4"} :    (!handshake.channel<i5>) -> !handshake.channel<f32>
    %outputs_4:2, %memEnd_5 = mem_controller[%arg0 : memref<400xf32>] %arg4 (%addressResult_8, %addressResult_44) %314#0 {connectedBlocks = [2 : i32, 3 : i32], handshake.name = "mem_controller5"} :    (!handshake.channel<i9>, !handshake.channel<i9>) -> (!handshake.channel<f32>, !handshake.channel<f32>)
    %1 = constant %0#1 {handshake.bb = 0 : ui32, handshake.name = "constant0", value = 1000 : i11} : <>, <i11>
    %2:3 = fork [3] %1 {handshake.bb = 0 : ui32, handshake.name = "fork1"} : <i11>
    %3 = extsi %2#1 {handshake.bb = 1 : ui32, handshake.name = "extsi0"} : <i11> to <i32>
    %5 = extsi %2#2 {handshake.bb = 1 : ui32, handshake.name = "extsi1"} : <i11> to <i32>
    %7 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant2", value = false} : <>, <i1>
    %8 = br %7 {handshake.bb = 0 : ui32, handshake.name = "br2"} : <i1>
    %9 = extsi %8 {handshake.bb = 0 : ui32, handshake.name = "extsi20"} : <i1> to <i6>
    %10 = br %0#7 {handshake.bb = 0 : ui32, handshake.name = "br3"} : <>
    %11 = mux %28#1 [%3, %trueResult_56] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux7"} : <i1>, [<i32>, <i32>] to <i32>
    %13 = mux %28#2 [%0#6, %trueResult_68] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux10"} : <i1>, [<>, <>] to <>
    %15 = mux %28#3 [%5, %trueResult_64] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux11"} : <i1>, [<i32>, <i32>] to <i32>
    %17 = mux %28#0 [%2#0, %256] {handshake.bb = 1 : ui32, handshake.name = "mux13"} : <i1>, [<i11>, <i11>] to <i11>
    %20 = mux %28#4 [%0#5, %trueResult_60] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux15"} : <i1>, [<>, <>] to <>
    %22 = mux %28#5 [%0#4, %trueResult_58] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux17"} : <i1>, [<>, <>] to <>
    %24 = mux %28#6 [%0#3, %trueResult_62] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux18"} : <i1>, [<>, <>] to <>
    %26 = init %311#7 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "init0"} : <i1>
    %28:7 = fork [7] %26 {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <i1>
    %29:2 = unbundle %53#0  {handshake.bb = 1 : ui32, handshake.name = "unbundle1"} : <f32> to _ 
    %31 = mux %index [%9, %trueResult_81] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i6>, <i6>] to <i6>
    %32:3 = fork [3] %31 {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <i6>
    %33 = trunci %32#0 {handshake.bb = 1 : ui32, handshake.name = "trunci0"} : <i6> to <i5>
    %result, %index = control_merge [%10, %trueResult_83]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %35:2 = fork [2] %result {handshake.bb = 1 : ui32, handshake.name = "fork4"} : <>
    %36 = constant %35#0 {handshake.bb = 1 : ui32, handshake.name = "constant17", value = false} : <>, <i1>
    %37 = buffer %32#2, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer0"} : <i6>
    %39 = extsi %37 {handshake.bb = 1 : ui32, handshake.name = "extsi21"} : <i6> to <i32>
    %40:2 = fork [2] %39 {handshake.bb = 1 : ui32, handshake.name = "fork5"} : <i32>
    %41 = init %40#0 {handshake.bb = 1 : ui32, handshake.name = "init14"} : <i32>
    %43:2 = fork [2] %41 {handshake.bb = 1 : ui32, handshake.name = "fork6"} : <i32>
    %44 = init %43#1 {handshake.bb = 1 : ui32, handshake.name = "init15"} : <i32>
    %46 = buffer %29#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 1 : ui32, handshake.name = "buffer1"} : <>
    %47:2 = fork [2] %46 {handshake.bb = 1 : ui32, handshake.name = "fork7"} : <>
    %48 = init %47#1 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 1 : ui32, handshake.name = "init16"} : <>
    %49:2 = fork [2] %48 {handshake.bb = 1 : ui32, handshake.name = "fork8"} : <>
    %50 = init %49#1 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 1 : ui32, handshake.name = "init17"} : <>
    %51:2 = fork [2] %50 {handshake.bb = 1 : ui32, handshake.name = "fork9"} : <>
    %52 = init %51#1 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 1 : ui32, handshake.name = "init18"} : <>
    %addressResult, %dataResult = load[%33] %outputs#0 {handshake.bb = 1 : ui32, handshake.deps = #handshake<deps[["store1", 2, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i5>, <f32>, <i5>, <f32>
    %53:2 = fork [2] %dataResult {handshake.bb = 1 : ui32, handshake.name = "fork10"} : <f32>
    %54 = br %36 {handshake.bb = 1 : ui32, handshake.name = "br4"} : <i1>
    %55 = extsi %54 {handshake.bb = 1 : ui32, handshake.name = "extsi19"} : <i1> to <i6>
    %56 = br %53#1 {handshake.bb = 1 : ui32, handshake.name = "br5"} : <f32>
    %58 = br %32#1 {handshake.bb = 1 : ui32, handshake.name = "br6"} : <i6>
    %60 = br %35#1 {handshake.bb = 1 : ui32, handshake.name = "br7"} : <>
    %61 = mux %78#1 [%55, %trueResult] {handshake.bb = 2 : ui32, handshake.name = "mux1"} : <i1>, [<i6>, <i6>] to <i6>
    %63:3 = fork [3] %61 {handshake.bb = 2 : ui32, handshake.name = "fork11"} : <i6>
    %64 = extsi %63#0 {handshake.bb = 2 : ui32, handshake.name = "extsi22"} : <i6> to <i9>
    %66 = extsi %63#2 {handshake.bb = 2 : ui32, handshake.name = "extsi23"} : <i6> to <i7>
    %68 = trunci %63#1 {handshake.bb = 2 : ui32, handshake.name = "trunci1"} : <i6> to <i5>
    %70 = mux %78#2 [%56, %trueResult_12] {handshake.bb = 2 : ui32, handshake.name = "mux2"} : <i1>, [<f32>, <f32>] to <f32>
    %72 = mux %78#0 [%58, %trueResult_14] {handshake.bb = 2 : ui32, handshake.name = "mux3"} : <i1>, [<i6>, <i6>] to <i6>
    %74:2 = fork [2] %72 {handshake.bb = 2 : ui32, handshake.name = "fork12"} : <i6>
    %75 = extsi %74#1 {handshake.bb = 2 : ui32, handshake.name = "extsi24"} : <i6> to <i32>
    %77:2 = fork [2] %75 {handshake.bb = 2 : ui32, handshake.name = "fork13"} : <i32>
    %result_6, %index_7 = control_merge [%60, %trueResult_16]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>, <>] to <>, <i1>
    %78:3 = fork [3] %index_7 {handshake.bb = 2 : ui32, handshake.name = "fork14"} : <i1>
    %79:2 = fork [2] %result_6 {handshake.bb = 2 : ui32, handshake.name = "fork15"} : <>
    %80 = constant %79#0 {handshake.bb = 2 : ui32, handshake.name = "constant21", value = false} : <>, <i1>
    %81 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %82 = constant %81 {handshake.bb = 2 : ui32, handshake.name = "constant22", value = 20 : i6} : <>, <i6>
    %83 = extsi %82 {handshake.bb = 2 : ui32, handshake.name = "extsi25"} : <i6> to <i7>
    %84 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %85 = constant %84 {handshake.bb = 2 : ui32, handshake.name = "constant23", value = 1 : i2} : <>, <i2>
    %86 = extsi %85 {handshake.bb = 2 : ui32, handshake.name = "extsi26"} : <i2> to <i7>
    %87 = source {handshake.bb = 2 : ui32, handshake.name = "source2"} : <>
    %88 = constant %87 {handshake.bb = 2 : ui32, handshake.name = "constant24", value = 4 : i4} : <>, <i4>
    %89 = extsi %88 {handshake.bb = 2 : ui32, handshake.name = "extsi8"} : <i4> to <i32>
    %90 = source {handshake.bb = 2 : ui32, handshake.name = "source3"} : <>
    %91 = constant %90 {handshake.bb = 2 : ui32, handshake.name = "constant25", value = 2 : i3} : <>, <i3>
    %92 = extsi %91 {handshake.bb = 2 : ui32, handshake.name = "extsi9"} : <i3> to <i32>
    %93 = shli %77#0, %92 {handshake.bb = 2 : ui32, handshake.name = "shli0"} : <i32>
    %95 = trunci %93 {handshake.bb = 2 : ui32, handshake.name = "trunci2"} : <i32> to <i9>
    %96 = shli %77#1, %89 {handshake.bb = 2 : ui32, handshake.name = "shli1"} : <i32>
    %98 = trunci %96 {handshake.bb = 2 : ui32, handshake.name = "trunci3"} : <i32> to <i9>
    %99 = addi %95, %98 {handshake.bb = 2 : ui32, handshake.name = "addi5"} : <i9>
    %100 = addi %64, %99 {handshake.bb = 2 : ui32, handshake.name = "addi0"} : <i9>
    %addressResult_8, %dataResult_9 = load[%100] %outputs_4#0 {handshake.bb = 2 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i9>, <f32>, <i9>, <f32>
    %addressResult_10, %dataResult_11 = load[%68] %outputs_2 {handshake.bb = 2 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load2"} : <i5>, <f32>, <i5>, <f32>
    %101 = mulf %dataResult_9, %dataResult_11 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "mulf0"} : <f32>
    %102 = addf %70, %101 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "addf0"} : <f32>
    %103 = addi %66, %86 {handshake.bb = 2 : ui32, handshake.name = "addi2"} : <i7>
    %104:2 = fork [2] %103 {handshake.bb = 2 : ui32, handshake.name = "fork16"} : <i7>
    %105 = trunci %104#0 {handshake.bb = 2 : ui32, handshake.name = "trunci4"} : <i7> to <i6>
    %107 = cmpi ult, %104#1, %83 {handshake.bb = 2 : ui32, handshake.name = "cmpi0"} : <i7>
    %109:5 = fork [5] %107 {handshake.bb = 2 : ui32, handshake.name = "fork17"} : <i1>
    %trueResult, %falseResult = cond_br %109#0, %105 {handshake.bb = 2 : ui32, handshake.name = "cond_br3"} : <i1>, <i6>
    sink %falseResult {handshake.name = "sink0"} : <i6>
    %trueResult_12, %falseResult_13 = cond_br %109#2, %102 {handshake.bb = 2 : ui32, handshake.name = "cond_br4"} : <i1>, <f32>
    %trueResult_14, %falseResult_15 = cond_br %109#1, %113 {handshake.bb = 2 : ui32, handshake.name = "cond_br5"} : <i1>, <i6>
    %113 = buffer %74#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer36"} : <i6>
    %trueResult_16, %falseResult_17 = cond_br %109#3, %79#1 {handshake.bb = 2 : ui32, handshake.name = "cond_br6"} : <i1>, <>
    %trueResult_18, %falseResult_19 = cond_br %109#4, %80 {handshake.bb = 2 : ui32, handshake.name = "cond_br7"} : <i1>, <i1>
    sink %trueResult_18 {handshake.name = "sink1"} : <i1>
    %116 = extsi %falseResult_19 {handshake.bb = 2 : ui32, handshake.name = "extsi18"} : <i1> to <i6>
    %trueResult_20, %falseResult_21 = cond_br %117, %228 {handshake.bb = 3 : ui32, handshake.name = "cond_br45"} : <i1>, <i32>
    %117 = buffer %243#9, bufferType = FIFO_BREAK_NONE, numSlots = 13 {handshake.bb = 3 : ui32, handshake.name = "buffer39"} : <i1>
    %trueResult_22, %falseResult_23 = cond_br %118, %233#1 {handshake.bb = 3 : ui32, handshake.name = "cond_br46"} : <i1>, <>
    %118 = buffer %243#8, bufferType = FIFO_BREAK_NONE, numSlots = 15 {handshake.bb = 3 : ui32, handshake.name = "buffer40"} : <i1>
    %trueResult_24, %falseResult_25 = cond_br %119, %120 {handshake.bb = 3 : ui32, handshake.name = "cond_br47"} : <i1>, <i32>
    %119 = buffer %243#7, bufferType = FIFO_BREAK_NONE, numSlots = 13 {handshake.bb = 3 : ui32, handshake.name = "buffer41"} : <i1>
    %120 = buffer %227#1, bufferType = FIFO_BREAK_NONE, numSlots = 13 {handshake.bb = 3 : ui32, handshake.name = "buffer42"} : <i32>
    %trueResult_26, %falseResult_27 = cond_br %121, %122 {handshake.bb = 3 : ui32, handshake.name = "cond_br48"} : <i1>, <i6>
    %121 = buffer %243#2, bufferType = FIFO_BREAK_NONE, numSlots = 13 {handshake.bb = 3 : ui32, handshake.name = "buffer43"} : <i1>
    %122 = buffer %223#0, bufferType = FIFO_BREAK_NONE, numSlots = 12 {handshake.bb = 3 : ui32, handshake.name = "buffer44"} : <i6>
    %123 = extsi %trueResult_26 {handshake.bb = 3 : ui32, handshake.name = "extsi27"} : <i6> to <i11>
    %trueResult_28, %falseResult_29 = cond_br %124, %231#1 {handshake.bb = 3 : ui32, handshake.name = "cond_br49"} : <i1>, <>
    %124 = buffer %243#6, bufferType = FIFO_BREAK_NONE, numSlots = 15 {handshake.bb = 3 : ui32, handshake.name = "buffer45"} : <i1>
    %trueResult_30, %falseResult_31 = cond_br %125, %235#1 {handshake.bb = 3 : ui32, handshake.name = "cond_br50"} : <i1>, <>
    %125 = buffer %243#5, bufferType = FIFO_BREAK_NONE, numSlots = 13 {handshake.bb = 3 : ui32, handshake.name = "buffer46"} : <i1>
    %trueResult_32, %falseResult_33 = cond_br %126, %236 {handshake.bb = 3 : ui32, handshake.name = "cond_br51"} : <i1>, <>
    %126 = buffer %243#4, bufferType = FIFO_BREAK_NONE, numSlots = 13 {handshake.bb = 3 : ui32, handshake.name = "buffer47"} : <i1>
    %127 = mux %128 [%11, %trueResult_20] {ftd.phi, handshake.bb = 3 : ui32, handshake.name = "mux21"} : <i1>, [<i32>, <i32>] to <i32>
    %128 = buffer %144#1, bufferType = FIFO_BREAK_NONE, numSlots = 12 {handshake.bb = 3 : ui32, handshake.name = "buffer48"} : <i1>
    %129 = mux %130 [%13, %trueResult_32] {ftd.phi, handshake.bb = 3 : ui32, handshake.name = "mux22"} : <i1>, [<>, <>] to <>
    %130 = buffer %144#2, bufferType = FIFO_BREAK_NONE, numSlots = 12 {handshake.bb = 3 : ui32, handshake.name = "buffer49"} : <i1>
    %131 = mux %132 [%15, %trueResult_24] {ftd.phi, handshake.bb = 3 : ui32, handshake.name = "mux23"} : <i1>, [<i32>, <i32>] to <i32>
    %132 = buffer %144#3, bufferType = FIFO_BREAK_NONE, numSlots = 13 {handshake.bb = 3 : ui32, handshake.name = "buffer50"} : <i1>
    %133 = mux %134 [%17, %123] {handshake.bb = 3 : ui32, handshake.name = "mux24"} : <i1>, [<i11>, <i11>] to <i11>
    %134 = buffer %144#0, bufferType = FIFO_BREAK_NONE, numSlots = 12 {handshake.bb = 3 : ui32, handshake.name = "buffer51"} : <i1>
    %135 = extsi %133 {handshake.bb = 3 : ui32, handshake.name = "extsi28"} : <i11> to <i32>
    %136 = mux %137 [%20, %trueResult_22] {ftd.phi, handshake.bb = 3 : ui32, handshake.name = "mux25"} : <i1>, [<>, <>] to <>
    %137 = buffer %144#4, bufferType = FIFO_BREAK_NONE, numSlots = 15 {handshake.bb = 3 : ui32, handshake.name = "buffer52"} : <i1>
    %138 = mux %139 [%22, %trueResult_28] {ftd.phi, handshake.bb = 3 : ui32, handshake.name = "mux26"} : <i1>, [<>, <>] to <>
    %139 = buffer %144#5, bufferType = FIFO_BREAK_NONE, numSlots = 15 {handshake.bb = 3 : ui32, handshake.name = "buffer53"} : <i1>
    %140 = mux %141 [%24, %trueResult_30] {ftd.phi, handshake.bb = 3 : ui32, handshake.name = "mux27"} : <i1>, [<>, <>] to <>
    %141 = buffer %144#6, bufferType = FIFO_BREAK_NONE, numSlots = 13 {handshake.bb = 3 : ui32, handshake.name = "buffer54"} : <i1>
    %142 = init %243#3 {ftd.imerge, handshake.bb = 3 : ui32, handshake.name = "init19"} : <i1>
    %144:7 = fork [7] %142 {handshake.bb = 3 : ui32, handshake.name = "fork18"} : <i1>
    %145 = mux %166#1 [%116, %trueResult_48] {handshake.bb = 3 : ui32, handshake.name = "mux4"} : <i1>, [<i6>, <i6>] to <i6>
    %147:5 = fork [5] %145 {handshake.bb = 3 : ui32, handshake.name = "fork19"} : <i6>
    %148 = extsi %149 {handshake.bb = 3 : ui32, handshake.name = "extsi29"} : <i6> to <i9>
    %149 = buffer %147#0, bufferType = FIFO_BREAK_NONE, numSlots = 10 {handshake.bb = 3 : ui32, handshake.name = "buffer57"} : <i6>
    %150 = extsi %147#2 {handshake.bb = 3 : ui32, handshake.name = "extsi30"} : <i6> to <i7>
    %152 = extsi %147#4 {handshake.bb = 3 : ui32, handshake.name = "extsi31"} : <i6> to <i32>
    %154:2 = fork [2] %152 {handshake.bb = 3 : ui32, handshake.name = "fork20"} : <i32>
    %155 = trunci %156 {handshake.bb = 3 : ui32, handshake.name = "trunci5"} : <i6> to <i5>
    %156 = buffer %147#1, bufferType = FIFO_BREAK_NONE, numSlots = 15 {handshake.bb = 3 : ui32, handshake.name = "buffer60"} : <i6>
    %157 = mux %166#0 [%falseResult_15, %trueResult_50] {handshake.bb = 3 : ui32, handshake.name = "mux5"} : <i1>, [<i6>, <i6>] to <i6>
    %159:2 = fork [2] %157 {handshake.bb = 3 : ui32, handshake.name = "fork21"} : <i6>
    %160 = extsi %159#1 {handshake.bb = 3 : ui32, handshake.name = "extsi32"} : <i6> to <i32>
    %162:2 = fork [2] %160 {handshake.bb = 3 : ui32, handshake.name = "fork22"} : <i32>
    %163 = mux %164 [%falseResult_13, %trueResult_52] {handshake.bb = 3 : ui32, handshake.name = "mux6"} : <i1>, [<f32>, <f32>] to <f32>
    %164 = buffer %166#2, bufferType = FIFO_BREAK_NONE, numSlots = 9 {handshake.bb = 3 : ui32, handshake.name = "buffer63"} : <i1>
    %165:2 = fork [2] %163 {handshake.bb = 3 : ui32, handshake.name = "fork23"} : <f32>
    %result_34, %index_35 = control_merge [%falseResult_17, %trueResult_54]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>, <>] to <>, <i1>
    %166:3 = fork [3] %index_35 {handshake.bb = 3 : ui32, handshake.name = "fork24"} : <i1>
    %167:2 = fork [2] %result_34 {handshake.bb = 3 : ui32, handshake.name = "fork25"} : <>
    %168 = constant %167#0 {handshake.bb = 3 : ui32, handshake.name = "constant26", value = 1 : i2} : <>, <i2>
    %169 = extsi %168 {handshake.bb = 3 : ui32, handshake.name = "extsi10"} : <i2> to <i32>
    %170 = source {handshake.bb = 3 : ui32, handshake.name = "source4"} : <>
    %171 = constant %170 {handshake.bb = 3 : ui32, handshake.name = "constant27", value = 20 : i6} : <>, <i6>
    %172 = extsi %171 {handshake.bb = 3 : ui32, handshake.name = "extsi33"} : <i6> to <i7>
    %173 = source {handshake.bb = 3 : ui32, handshake.name = "source5"} : <>
    %174 = constant %173 {handshake.bb = 3 : ui32, handshake.name = "constant28", value = 1 : i2} : <>, <i2>
    %175 = extsi %174 {handshake.bb = 3 : ui32, handshake.name = "extsi34"} : <i2> to <i7>
    %176 = source {handshake.bb = 3 : ui32, handshake.name = "source6"} : <>
    %177 = constant %176 {handshake.bb = 3 : ui32, handshake.name = "constant29", value = 4 : i4} : <>, <i4>
    %178 = extsi %177 {handshake.bb = 3 : ui32, handshake.name = "extsi13"} : <i4> to <i32>
    %179 = source {handshake.bb = 3 : ui32, handshake.name = "source7"} : <>
    %180 = constant %179 {handshake.bb = 3 : ui32, handshake.name = "constant30", value = 2 : i3} : <>, <i3>
    %181 = extsi %180 {handshake.bb = 3 : ui32, handshake.name = "extsi14"} : <i3> to <i32>
    %182 = gate %183, %129 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 3 : ui32, handshake.name = "gate0"} : <i32>, !handshake.control<> to <i32>
    %183 = buffer %154#0, bufferType = FIFO_BREAK_NONE, numSlots = 13 {handshake.bb = 3 : ui32, handshake.name = "buffer64"} : <i32>
    %184:3 = fork [3] %182 {handshake.bb = 3 : ui32, handshake.name = "fork26"} : <i32>
    %185 = cmpi ne, %184#2, %135 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 3 : ui32, handshake.name = "cmpi3"} : <i32>
    %187:2 = fork [2] %185 {handshake.bb = 3 : ui32, handshake.name = "fork27"} : <i1>
    %188 = cmpi ne, %184#1, %131 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 3 : ui32, handshake.name = "cmpi4"} : <i32>
    %190:2 = fork [2] %188 {handshake.bb = 3 : ui32, handshake.name = "fork28"} : <i1>
    %191 = cmpi ne, %184#0, %127 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 3 : ui32, handshake.name = "cmpi5"} : <i32>
    %193:2 = fork [2] %191 {handshake.bb = 3 : ui32, handshake.name = "fork29"} : <i1>
    %trueResult_36, %falseResult_37 = cond_br %194, %138 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 3 : ui32, handshake.name = "cond_br25"} : <i1>, <>
    %194 = buffer %187#1, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 3 : ui32, handshake.name = "buffer68"} : <i1>
    sink %trueResult_36 {handshake.name = "sink2"} : <>
    %trueResult_38, %falseResult_39 = cond_br %195, %136 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 3 : ui32, handshake.name = "cond_br26"} : <i1>, <>
    %195 = buffer %190#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer69"} : <i1>
    sink %trueResult_38 {handshake.name = "sink3"} : <>
    %trueResult_40, %falseResult_41 = cond_br %196, %140 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 3 : ui32, handshake.name = "cond_br27"} : <i1>, <>
    %196 = buffer %193#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer70"} : <i1>
    sink %trueResult_40 {handshake.name = "sink4"} : <>
    %197 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 3 : ui32, handshake.name = "source10"} : <>
    %198 = mux %187#0 [%falseResult_37, %197] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 3 : ui32, handshake.name = "mux28"} : <i1>, [<>, <>] to <>
    %200 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 3 : ui32, handshake.name = "source11"} : <>
    %201 = mux %190#0 [%falseResult_39, %200] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 3 : ui32, handshake.name = "mux29"} : <i1>, [<>, <>] to <>
    %203 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 3 : ui32, handshake.name = "source12"} : <>
    %204 = mux %193#0 [%falseResult_41, %203] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 3 : ui32, handshake.name = "mux30"} : <i1>, [<>, <>] to <>
    %206 = join %198, %201, %204 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 3 : ui32, handshake.name = "join0"} : <>
    %207 = gate %208, %206 {handshake.bb = 3 : ui32, handshake.name = "gate1"} : <i32>, !handshake.control<> to <i32>
    %208 = buffer %154#1, bufferType = FIFO_BREAK_NONE, numSlots = 12 {handshake.bb = 3 : ui32, handshake.name = "buffer74"} : <i32>
    %209 = trunci %207 {handshake.bb = 3 : ui32, handshake.name = "trunci6"} : <i32> to <i5>
    %addressResult_42, %dataResult_43 = load[%209] %outputs_0#0 {handshake.bb = 3 : ui32, handshake.deps = #handshake<deps[["store0", 1, false], ["store0", 3, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load3"} : <i5>, <f32>, <i5>, <f32>
    %210 = shli %211, %181 {handshake.bb = 3 : ui32, handshake.name = "shli2"} : <i32>
    %211 = buffer %162#0, bufferType = FIFO_BREAK_NONE, numSlots = 9 {handshake.bb = 3 : ui32, handshake.name = "buffer75"} : <i32>
    %212 = trunci %210 {handshake.bb = 3 : ui32, handshake.name = "trunci7"} : <i32> to <i9>
    %213 = shli %214, %178 {handshake.bb = 3 : ui32, handshake.name = "shli3"} : <i32>
    %214 = buffer %162#1, bufferType = FIFO_BREAK_NONE, numSlots = 9 {handshake.bb = 3 : ui32, handshake.name = "buffer76"} : <i32>
    %215 = trunci %213 {handshake.bb = 3 : ui32, handshake.name = "trunci8"} : <i32> to <i9>
    %216 = addi %212, %215 {handshake.bb = 3 : ui32, handshake.name = "addi6"} : <i9>
    %217 = addi %148, %216 {handshake.bb = 3 : ui32, handshake.name = "addi1"} : <i9>
    %addressResult_44, %dataResult_45 = load[%217] %outputs_4#1 {handshake.bb = 3 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load4"} : <i9>, <f32>, <i9>, <f32>
    %218 = mulf %dataResult_45, %219 {fastmath = #arith.fastmath<none>, handshake.bb = 3 : ui32, handshake.name = "mulf1"} : <f32>
    %219 = buffer %165#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer77"} : <f32>
    %220 = addf %dataResult_43, %218 {fastmath = #arith.fastmath<none>, handshake.bb = 3 : ui32, handshake.name = "addf1"} : <f32>
    %221 = buffer %147#3, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer2"} : <i6>
    %223:2 = fork [2] %221 {handshake.bb = 3 : ui32, handshake.name = "fork30"} : <i6>
    %224 = extsi %223#1 {handshake.bb = 3 : ui32, handshake.name = "extsi35"} : <i6> to <i32>
    %226 = init %224 {handshake.bb = 3 : ui32, handshake.name = "init26"} : <i32>
    %227:2 = fork [2] %226 {handshake.bb = 3 : ui32, handshake.name = "fork31"} : <i32>
    %228 = init %229 {handshake.bb = 3 : ui32, handshake.name = "init27"} : <i32>
    %229 = buffer %227#0, bufferType = FIFO_BREAK_NONE, numSlots = 15 {handshake.bb = 3 : ui32, handshake.name = "buffer80"} : <i32>
    %230 = buffer %doneResult, bufferType = FIFO_BREAK_NONE, numSlots = 1 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 3 : ui32, handshake.name = "buffer3"} : <>
    %231:2 = fork [2] %230 {handshake.bb = 3 : ui32, handshake.name = "fork32"} : <>
    %232 = init %231#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 3 : ui32, handshake.name = "init28"} : <>
    %233:2 = fork [2] %232 {handshake.bb = 3 : ui32, handshake.name = "fork33"} : <>
    %234 = init %233#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 3 : ui32, handshake.name = "init29"} : <>
    %235:2 = fork [2] %234 {handshake.bb = 3 : ui32, handshake.name = "fork34"} : <>
    %236 = init %235#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 3 : ui32, handshake.name = "init30"} : <>
    %addressResult_46, %dataResult_47, %doneResult = store[%155] %220 %outputs_0#1 {handshake.bb = 3 : ui32, handshake.deps = #handshake<deps[["load3", 1, false], ["store0", 1, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store0"} : <i5>, <f32>, <>, <i5>, <f32>, <>
    %237 = addi %150, %175 {handshake.bb = 3 : ui32, handshake.name = "addi3"} : <i7>
    %238:2 = fork [2] %237 {handshake.bb = 3 : ui32, handshake.name = "fork35"} : <i7>
    %239 = trunci %238#0 {handshake.bb = 3 : ui32, handshake.name = "trunci9"} : <i7> to <i6>
    %241 = cmpi ult, %238#1, %172 {handshake.bb = 3 : ui32, handshake.name = "cmpi1"} : <i7>
    %243:12 = fork [12] %241 {handshake.bb = 3 : ui32, handshake.name = "fork36"} : <i1>
    %trueResult_48, %falseResult_49 = cond_br %243#0, %239 {handshake.bb = 3 : ui32, handshake.name = "cond_br8"} : <i1>, <i6>
    sink %falseResult_49 {handshake.name = "sink5"} : <i6>
    %trueResult_50, %falseResult_51 = cond_br %243#1, %159#0 {handshake.bb = 3 : ui32, handshake.name = "cond_br9"} : <i1>, <i6>
    %trueResult_52, %falseResult_53 = cond_br %247, %165#0 {handshake.bb = 3 : ui32, handshake.name = "cond_br10"} : <i1>, <f32>
    %247 = buffer %243#10, bufferType = FIFO_BREAK_NONE, numSlots = 9 {handshake.bb = 3 : ui32, handshake.name = "buffer86"} : <i1>
    %trueResult_54, %falseResult_55 = cond_br %249, %167#1 {handshake.bb = 3 : ui32, handshake.name = "cond_br11"} : <i1>, <>
    %249 = buffer %243#11, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer88"} : <i1>
    %trueResult_56, %falseResult_57 = cond_br %250, %falseResult_21 {handshake.bb = 4 : ui32, handshake.name = "cond_br52"} : <i1>, <i32>
    %250 = buffer %311#6, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer89"} : <i1>
    sink %falseResult_57 {handshake.name = "sink6"} : <i32>
    %trueResult_58, %falseResult_59 = cond_br %251, %falseResult_29 {handshake.bb = 4 : ui32, handshake.name = "cond_br53"} : <i1>, <>
    %251 = buffer %311#5, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer90"} : <i1>
    sink %falseResult_59 {handshake.name = "sink7"} : <>
    %trueResult_60, %falseResult_61 = cond_br %252, %falseResult_23 {handshake.bb = 4 : ui32, handshake.name = "cond_br54"} : <i1>, <>
    %252 = buffer %311#4, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer91"} : <i1>
    sink %falseResult_61 {handshake.name = "sink8"} : <>
    %trueResult_62, %falseResult_63 = cond_br %253, %falseResult_31 {handshake.bb = 4 : ui32, handshake.name = "cond_br55"} : <i1>, <>
    %253 = buffer %311#3, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer92"} : <i1>
    sink %falseResult_63 {handshake.name = "sink9"} : <>
    %trueResult_64, %falseResult_65 = cond_br %254, %falseResult_25 {handshake.bb = 4 : ui32, handshake.name = "cond_br56"} : <i1>, <i32>
    %254 = buffer %311#2, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer93"} : <i1>
    sink %falseResult_65 {handshake.name = "sink10"} : <i32>
    %trueResult_66, %falseResult_67 = cond_br %255, %falseResult_27 {handshake.bb = 4 : ui32, handshake.name = "cond_br57"} : <i1>, <i6>
    %255 = buffer %311#9, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer94"} : <i1>
    sink %falseResult_67 {handshake.name = "sink11"} : <i6>
    %256 = extsi %trueResult_66 {handshake.bb = 4 : ui32, handshake.name = "extsi36"} : <i6> to <i11>
    %trueResult_68, %falseResult_69 = cond_br %257, %falseResult_33 {handshake.bb = 4 : ui32, handshake.name = "cond_br58"} : <i1>, <>
    %257 = buffer %311#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer95"} : <i1>
    sink %falseResult_69 {handshake.name = "sink12"} : <>
    %258 = merge %falseResult_51 {handshake.bb = 4 : ui32, handshake.name = "merge0"} : <i6>
    %259:2 = fork [2] %258 {handshake.bb = 4 : ui32, handshake.name = "fork37"} : <i6>
    %260 = extsi %259#0 {handshake.bb = 4 : ui32, handshake.name = "extsi37"} : <i6> to <i7>
    %262 = extsi %259#1 {handshake.bb = 4 : ui32, handshake.name = "extsi38"} : <i6> to <i32>
    %264:2 = fork [2] %262 {handshake.bb = 4 : ui32, handshake.name = "fork38"} : <i32>
    %265 = merge %falseResult_53 {handshake.bb = 4 : ui32, handshake.name = "merge1"} : <f32>
    %result_70, %index_71 = control_merge [%falseResult_55]  {handshake.bb = 4 : ui32, handshake.name = "control_merge3"} : [<>] to <>, <i1>
    sink %index_71 {handshake.name = "sink13"} : <i1>
    %266:2 = fork [2] %result_70 {handshake.bb = 4 : ui32, handshake.name = "fork39"} : <>
    %267 = constant %266#0 {handshake.bb = 4 : ui32, handshake.name = "constant31", value = 1 : i2} : <>, <i2>
    %268 = extsi %267 {handshake.bb = 4 : ui32, handshake.name = "extsi15"} : <i2> to <i32>
    %269 = source {handshake.bb = 4 : ui32, handshake.name = "source8"} : <>
    %270 = constant %269 {handshake.bb = 4 : ui32, handshake.name = "constant32", value = 20 : i6} : <>, <i6>
    %271 = extsi %270 {handshake.bb = 4 : ui32, handshake.name = "extsi39"} : <i6> to <i7>
    %272 = source {handshake.bb = 4 : ui32, handshake.name = "source9"} : <>
    %273 = constant %272 {handshake.bb = 4 : ui32, handshake.name = "constant33", value = 1 : i2} : <>, <i2>
    %274 = extsi %273 {handshake.bb = 4 : ui32, handshake.name = "extsi40"} : <i2> to <i7>
    %275 = gate %264#0, %52 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 4 : ui32, handshake.name = "gate2"} : <i32>, !handshake.control<> to <i32>
    %277:3 = fork [3] %275 {handshake.bb = 4 : ui32, handshake.name = "fork40"} : <i32>
    %278 = cmpi ne, %277#2, %40#1 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 4 : ui32, handshake.name = "cmpi6"} : <i32>
    %281:2 = fork [2] %278 {handshake.bb = 4 : ui32, handshake.name = "fork41"} : <i1>
    %282 = cmpi ne, %277#1, %43#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 4 : ui32, handshake.name = "cmpi7"} : <i32>
    %285:2 = fork [2] %282 {handshake.bb = 4 : ui32, handshake.name = "fork42"} : <i1>
    %286 = cmpi ne, %277#0, %44 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 4 : ui32, handshake.name = "cmpi8"} : <i32>
    %288:2 = fork [2] %286 {handshake.bb = 4 : ui32, handshake.name = "fork43"} : <i1>
    %trueResult_72, %falseResult_73 = cond_br %281#1, %47#0 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 4 : ui32, handshake.name = "cond_br42"} : <i1>, <>
    sink %trueResult_72 {handshake.name = "sink14"} : <>
    %trueResult_74, %falseResult_75 = cond_br %285#1, %49#0 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 4 : ui32, handshake.name = "cond_br43"} : <i1>, <>
    sink %trueResult_74 {handshake.name = "sink15"} : <>
    %trueResult_76, %falseResult_77 = cond_br %288#1, %51#0 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 4 : ui32, handshake.name = "cond_br44"} : <i1>, <>
    sink %trueResult_76 {handshake.name = "sink16"} : <>
    %292 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 4 : ui32, handshake.name = "source13"} : <>
    %293 = mux %281#0 [%falseResult_73, %292] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 4 : ui32, handshake.name = "mux31"} : <i1>, [<>, <>] to <>
    %295 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 4 : ui32, handshake.name = "source14"} : <>
    %296 = mux %285#0 [%falseResult_75, %295] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 4 : ui32, handshake.name = "mux32"} : <i1>, [<>, <>] to <>
    %298 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 4 : ui32, handshake.name = "source15"} : <>
    %299 = mux %300 [%falseResult_77, %298] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 4 : ui32, handshake.name = "mux33"} : <i1>, [<>, <>] to <>
    %300 = buffer %288#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer109"} : <i1>
    %301 = join %293, %296, %299 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 4 : ui32, handshake.name = "join1"} : <>
    %302 = gate %303, %301 {handshake.bb = 4 : ui32, handshake.name = "gate3"} : <i32>, !handshake.control<> to <i32>
    %303 = buffer %264#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer110"} : <i32>
    %304 = trunci %302 {handshake.bb = 4 : ui32, handshake.name = "trunci10"} : <i32> to <i5>
    %addressResult_78, %dataResult_79, %doneResult_80 = store[%304] %265 %outputs#1 {handshake.bb = 4 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store1"} : <i5>, <f32>, <>, <i5>, <f32>, <>
    sink %doneResult_80 {handshake.name = "sink17"} : <>
    %305 = addi %260, %274 {handshake.bb = 4 : ui32, handshake.name = "addi4"} : <i7>
    %306:2 = fork [2] %305 {handshake.bb = 4 : ui32, handshake.name = "fork44"} : <i7>
    %307 = trunci %306#0 {handshake.bb = 4 : ui32, handshake.name = "trunci11"} : <i7> to <i6>
    %309 = cmpi ult, %306#1, %271 {handshake.bb = 4 : ui32, handshake.name = "cmpi2"} : <i7>
    %311:10 = fork [10] %309 {handshake.bb = 4 : ui32, handshake.name = "fork45"} : <i1>
    %trueResult_81, %falseResult_82 = cond_br %311#0, %307 {handshake.bb = 4 : ui32, handshake.name = "cond_br12"} : <i1>, <i6>
    sink %falseResult_82 {handshake.name = "sink18"} : <i6>
    %trueResult_83, %falseResult_84 = cond_br %311#8, %266#1 {handshake.bb = 4 : ui32, handshake.name = "cond_br13"} : <i1>, <>
    %result_85, %index_86 = control_merge [%falseResult_84]  {handshake.bb = 5 : ui32, handshake.name = "control_merge4"} : [<>] to <>, <i1>
    sink %index_86 {handshake.name = "sink19"} : <i1>
    %314:4 = fork [4] %result_85 {handshake.bb = 5 : ui32, handshake.name = "fork46"} : <>
    end {handshake.bb = 5 : ui32, handshake.name = "end0"} %memEnd_5, %memEnd_3, %memEnd_1, %memEnd, %0#2 : <>, <>, <>, <>, <>
  }
}

