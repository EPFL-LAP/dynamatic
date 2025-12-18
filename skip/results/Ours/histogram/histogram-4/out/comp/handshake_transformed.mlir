module {
  handshake.func @histogram(%arg0: memref<1000xi32>, %arg1: memref<1000xf32>, %arg2: memref<1000xf32>, %arg3: !handshake.channel<i32>, %arg4: !handshake.control<>, %arg5: !handshake.control<>, %arg6: !handshake.control<>, %arg7: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["feature", "weight", "hist", "n", "feature_start", "weight_start", "hist_start", "start"], cfg.edges = "[0,1][2,1][1,2,3,cmpi0]", resNames = ["feature_end", "weight_end", "hist_end", "end"]} {
    %0:9 = fork [9] %arg7 {handshake.bb = 0 : ui32, handshake.name = "fork0"} : <>
    %outputs:2, %memEnd = mem_controller[%arg2 : memref<1000xf32>] %arg6 (%74, %addressResult_38, %addressResult_40, %dataResult_41) %142#2 {connectedBlocks = [2 : i32], handshake.name = "mem_controller2"} :    (!handshake.channel<i32>, !handshake.channel<i10>, !handshake.channel<i10>, !handshake.channel<f32>) -> (!handshake.channel<f32>, !handshake.control<>)
    %outputs_0, %memEnd_1 = mem_controller[%arg1 : memref<1000xf32>] %arg5 (%addressResult_28) %142#1 {connectedBlocks = [2 : i32], handshake.name = "mem_controller3"} :    (!handshake.channel<i10>) -> !handshake.channel<f32>
    %outputs_2, %memEnd_3 = mem_controller[%arg0 : memref<1000xi32>] %arg4 (%addressResult) %142#0 {connectedBlocks = [2 : i32], handshake.name = "mem_controller4"} :    (!handshake.channel<i10>) -> !handshake.channel<i32>
    %1 = constant %0#1 {handshake.bb = 0 : ui32, handshake.name = "constant7", value = 1000 : i11} : <>, <i11>
    %2:4 = fork [4] %1 {handshake.bb = 0 : ui32, handshake.name = "fork1"} : <i11>
    %3 = extsi %2#0 {handshake.bb = 1 : ui32, handshake.name = "extsi0"} : <i11> to <i32>
    %5 = extsi %2#1 {handshake.bb = 1 : ui32, handshake.name = "extsi1"} : <i11> to <i32>
    %7 = extsi %2#2 {handshake.bb = 1 : ui32, handshake.name = "extsi2"} : <i11> to <i32>
    %9 = extsi %2#3 {handshake.bb = 1 : ui32, handshake.name = "extsi3"} : <i11> to <i32>
    %11 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant8", value = false} : <>, <i1>
    %12 = br %11 {handshake.bb = 0 : ui32, handshake.name = "br2"} : <i1>
    %13 = extsi %12 {handshake.bb = 0 : ui32, handshake.name = "extsi7"} : <i1> to <i32>
    %14 = br %arg3 {handshake.bb = 0 : ui32, handshake.name = "br3"} : <i32>
    %15 = br %0#8 {handshake.bb = 0 : ui32, handshake.name = "br4"} : <>
    %16 = mux %39#0 [%3, %126] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux2"} : <i1>, [<i32>, <i32>] to <i32>
    %18 = mux %39#1 [%5, %119#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux3"} : <i1>, [<i32>, <i32>] to <i32>
    %21 = mux %39#2 [%7, %125#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux4"} : <i1>, [<i32>, <i32>] to <i32>
    %24 = mux %39#3 [%9, %25] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux5"} : <i1>, [<i32>, <i32>] to <i32>
    %25 = buffer %122#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer11"} : <i32>
    %27 = mux %39#4 [%0#7, %136] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux6"} : <i1>, [<>, <>] to <>
    %29 = mux %30 [%0#6, %133#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux7"} : <i1>, [<>, <>] to <>
    %30 = buffer %39#5, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 1 : ui32, handshake.name = "buffer14"} : <i1>
    %31 = mux %32 [%0#5, %129#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux8"} : <i1>, [<>, <>] to <>
    %32 = buffer %39#6, bufferType = FIFO_BREAK_NONE, numSlots = 4 {handshake.bb = 1 : ui32, handshake.name = "buffer15"} : <i1>
    %33 = mux %34 [%0#4, %135#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux9"} : <i1>, [<>, <>] to <>
    %34 = buffer %39#7, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer16"} : <i1>
    %35 = mux %36 [%0#3, %131#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux10"} : <i1>, [<>, <>] to <>
    %36 = buffer %39#8, bufferType = FIFO_BREAK_NONE, numSlots = 3 {handshake.bb = 1 : ui32, handshake.name = "buffer17"} : <i1>
    %37 = init %38 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "init0"} : <i1>
    %38 = buffer %50#12, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 1 : ui32, handshake.name = "buffer18"} : <i1>
    %39:9 = fork [9] %37 {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <i1>
    %40 = mux %46#0 [%13, %139] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i32>, <i32>] to <i32>
    %42:2 = fork [2] %40 {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <i32>
    %43 = mux %46#1 [%14, %140] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i1>, [<i32>, <i32>] to <i32>
    %45:2 = fork [2] %43 {handshake.bb = 1 : ui32, handshake.name = "fork4"} : <i32>
    %result, %index = control_merge [%15, %141]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %46:2 = fork [2] %index {handshake.bb = 1 : ui32, handshake.name = "fork5"} : <i1>
    %47 = cmpi slt, %42#1, %45#1 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i32>
    %50:13 = fork [13] %47 {handshake.bb = 1 : ui32, handshake.name = "fork6"} : <i1>
    %trueResult, %falseResult = cond_br %50#11, %45#0 {handshake.bb = 1 : ui32, handshake.name = "cond_br1"} : <i1>, <i32>
    sink %falseResult {handshake.name = "sink0"} : <i32>
    %trueResult_4, %falseResult_5 = cond_br %50#10, %42#0 {handshake.bb = 1 : ui32, handshake.name = "cond_br2"} : <i1>, <i32>
    sink %falseResult_5 {handshake.name = "sink1"} : <i32>
    %trueResult_6, %falseResult_7 = cond_br %50#9, %result {handshake.bb = 1 : ui32, handshake.name = "cond_br3"} : <i1>, <>
    %trueResult_8, %falseResult_9 = cond_br %56, %33 {handshake.bb = 2 : ui32, handshake.name = "cond_br16"} : <i1>, <>
    %56 = buffer %50#8, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 2 : ui32, handshake.name = "buffer28"} : <i1>
    sink %falseResult_9 {handshake.name = "sink2"} : <>
    %trueResult_10, %falseResult_11 = cond_br %57, %27 {handshake.bb = 2 : ui32, handshake.name = "cond_br17"} : <i1>, <>
    %57 = buffer %50#7, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer29"} : <i1>
    sink %falseResult_11 {handshake.name = "sink3"} : <>
    %trueResult_12, %falseResult_13 = cond_br %58, %35 {handshake.bb = 2 : ui32, handshake.name = "cond_br18"} : <i1>, <>
    %58 = buffer %50#6, bufferType = FIFO_BREAK_NONE, numSlots = 4 {handshake.bb = 2 : ui32, handshake.name = "buffer30"} : <i1>
    sink %falseResult_13 {handshake.name = "sink4"} : <>
    %trueResult_14, %falseResult_15 = cond_br %59, %18 {handshake.bb = 2 : ui32, handshake.name = "cond_br19"} : <i1>, <i32>
    %59 = buffer %50#5, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer31"} : <i1>
    sink %falseResult_15 {handshake.name = "sink5"} : <i32>
    %trueResult_16, %falseResult_17 = cond_br %60, %21 {handshake.bb = 2 : ui32, handshake.name = "cond_br20"} : <i1>, <i32>
    %60 = buffer %50#4, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer32"} : <i1>
    sink %falseResult_17 {handshake.name = "sink6"} : <i32>
    %trueResult_18, %falseResult_19 = cond_br %61, %31 {handshake.bb = 2 : ui32, handshake.name = "cond_br21"} : <i1>, <>
    %61 = buffer %50#3, bufferType = FIFO_BREAK_NONE, numSlots = 5 {handshake.bb = 2 : ui32, handshake.name = "buffer33"} : <i1>
    sink %falseResult_19 {handshake.name = "sink7"} : <>
    %trueResult_20, %falseResult_21 = cond_br %62, %24 {handshake.bb = 2 : ui32, handshake.name = "cond_br22"} : <i1>, <i32>
    %62 = buffer %50#2, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer34"} : <i1>
    sink %falseResult_21 {handshake.name = "sink8"} : <i32>
    %trueResult_22, %falseResult_23 = cond_br %63, %16 {handshake.bb = 2 : ui32, handshake.name = "cond_br23"} : <i1>, <i32>
    %63 = buffer %50#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer35"} : <i1>
    sink %falseResult_23 {handshake.name = "sink9"} : <i32>
    %trueResult_24, %falseResult_25 = cond_br %64, %29 {handshake.bb = 2 : ui32, handshake.name = "cond_br24"} : <i1>, <>
    %64 = buffer %50#0, bufferType = FIFO_BREAK_NONE, numSlots = 3 {handshake.bb = 2 : ui32, handshake.name = "buffer36"} : <i1>
    sink %falseResult_25 {handshake.name = "sink10"} : <>
    %65 = merge %trueResult {handshake.bb = 2 : ui32, handshake.name = "merge0"} : <i32>
    %66 = merge %trueResult_4 {handshake.bb = 2 : ui32, handshake.name = "merge1"} : <i32>
    %67:3 = fork [3] %66 {handshake.bb = 2 : ui32, handshake.name = "fork7"} : <i32>
    %68 = trunci %67#0 {handshake.bb = 2 : ui32, handshake.name = "trunci0"} : <i32> to <i10>
    %70 = trunci %67#1 {handshake.bb = 2 : ui32, handshake.name = "trunci1"} : <i32> to <i10>
    %result_26, %index_27 = control_merge [%trueResult_6]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>] to <>, <i1>
    sink %index_27 {handshake.name = "sink11"} : <i1>
    %72:2 = fork [2] %result_26 {handshake.bb = 2 : ui32, handshake.name = "fork8"} : <>
    %73 = constant %72#0 {handshake.bb = 2 : ui32, handshake.name = "constant9", value = 1 : i2} : <>, <i2>
    %74 = extsi %73 {handshake.bb = 2 : ui32, handshake.name = "extsi5"} : <i2> to <i32>
    %75 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %76 = constant %75 {handshake.bb = 2 : ui32, handshake.name = "constant10", value = 1 : i2} : <>, <i2>
    %77 = extsi %76 {handshake.bb = 2 : ui32, handshake.name = "extsi6"} : <i2> to <i32>
    %addressResult, %dataResult = load[%70] %outputs_2 {handshake.bb = 2 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i10>, <i32>, <i10>, <i32>
    %78:4 = fork [4] %dataResult {handshake.bb = 2 : ui32, handshake.name = "fork9"} : <i32>
    %79 = trunci %80 {handshake.bb = 2 : ui32, handshake.name = "trunci2"} : <i32> to <i10>
    %80 = buffer %78#0, bufferType = FIFO_BREAK_NONE, numSlots = 4 {handshake.bb = 2 : ui32, handshake.name = "buffer39"} : <i32>
    %addressResult_28, %dataResult_29 = load[%68] %outputs_0 {handshake.bb = 2 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i10>, <f32>, <i10>, <f32>
    %81 = gate %78#1, %trueResult_10 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "gate0"} : <i32>, !handshake.control<> to <i32>
    %83:4 = fork [4] %81 {handshake.bb = 2 : ui32, handshake.name = "fork10"} : <i32>
    %84 = cmpi ne, %83#3, %trueResult_14 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi1"} : <i32>
    %86:2 = fork [2] %84 {handshake.bb = 2 : ui32, handshake.name = "fork11"} : <i1>
    %87 = cmpi ne, %83#2, %trueResult_20 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi2"} : <i32>
    %89:2 = fork [2] %87 {handshake.bb = 2 : ui32, handshake.name = "fork12"} : <i1>
    %90 = cmpi ne, %83#1, %trueResult_16 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi3"} : <i32>
    %92:2 = fork [2] %90 {handshake.bb = 2 : ui32, handshake.name = "fork13"} : <i1>
    %93 = cmpi ne, %83#0, %trueResult_22 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi4"} : <i32>
    %95:2 = fork [2] %93 {handshake.bb = 2 : ui32, handshake.name = "fork14"} : <i1>
    %trueResult_30, %falseResult_31 = cond_br %96, %trueResult_18 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br12"} : <i1>, <>
    %96 = buffer %86#1, bufferType = FIFO_BREAK_NONE, numSlots = 4 {handshake.bb = 2 : ui32, handshake.name = "buffer45"} : <i1>
    sink %trueResult_30 {handshake.name = "sink12"} : <>
    %trueResult_32, %falseResult_33 = cond_br %97, %trueResult_12 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br13"} : <i1>, <>
    %97 = buffer %89#1, bufferType = FIFO_BREAK_NONE, numSlots = 3 {handshake.bb = 2 : ui32, handshake.name = "buffer46"} : <i1>
    sink %trueResult_32 {handshake.name = "sink13"} : <>
    %trueResult_34, %falseResult_35 = cond_br %98, %trueResult_24 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br14"} : <i1>, <>
    %98 = buffer %92#1, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 2 : ui32, handshake.name = "buffer47"} : <i1>
    sink %trueResult_34 {handshake.name = "sink14"} : <>
    %trueResult_36, %falseResult_37 = cond_br %99, %trueResult_8 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br15"} : <i1>, <>
    %99 = buffer %95#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer48"} : <i1>
    sink %trueResult_36 {handshake.name = "sink15"} : <>
    %100 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %101 = mux %86#0 [%falseResult_31, %100] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux11"} : <i1>, [<>, <>] to <>
    %103 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source2"} : <>
    %104 = mux %89#0 [%falseResult_33, %103] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux12"} : <i1>, [<>, <>] to <>
    %106 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source3"} : <>
    %107 = mux %92#0 [%falseResult_35, %106] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux13"} : <i1>, [<>, <>] to <>
    %109 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source4"} : <>
    %110 = mux %95#0 [%falseResult_37, %109] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux14"} : <i1>, [<>, <>] to <>
    %112 = join %101, %104, %107, %110 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "join0"} : <>
    %113 = gate %78#2, %112 {handshake.bb = 2 : ui32, handshake.name = "gate1"} : <i32>, !handshake.control<> to <i32>
    %115 = trunci %113 {handshake.bb = 2 : ui32, handshake.name = "trunci3"} : <i32> to <i10>
    %addressResult_38, %dataResult_39 = load[%115] %outputs#0 {handshake.bb = 2 : ui32, handshake.deps = #handshake<deps[["store1", 0, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load5"} : <i10>, <f32>, <i10>, <f32>
    %116 = addf %dataResult_39, %dataResult_29 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "addf0"} : <f32>
    %117 = buffer %78#3, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer0"} : <i32>
    %119:2 = fork [2] %117 {handshake.bb = 2 : ui32, handshake.name = "fork15"} : <i32>
    %120 = init %119#0 {handshake.bb = 2 : ui32, handshake.name = "init9"} : <i32>
    %122:2 = fork [2] %120 {handshake.bb = 2 : ui32, handshake.name = "fork16"} : <i32>
    %123 = init %124 {handshake.bb = 2 : ui32, handshake.name = "init10"} : <i32>
    %124 = buffer %122#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer56"} : <i32>
    %125:2 = fork [2] %123 {handshake.bb = 2 : ui32, handshake.name = "fork17"} : <i32>
    %126 = init %127 {handshake.bb = 2 : ui32, handshake.name = "init11"} : <i32>
    %127 = buffer %125#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer57"} : <i32>
    %128 = buffer %doneResult, bufferType = FIFO_BREAK_NONE, numSlots = 1 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "buffer1"} : <>
    %129:2 = fork [2] %128 {handshake.bb = 2 : ui32, handshake.name = "fork18"} : <>
    %130 = init %129#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init12"} : <>
    %131:2 = fork [2] %130 {handshake.bb = 2 : ui32, handshake.name = "fork19"} : <>
    %132 = init %131#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init13"} : <>
    %133:2 = fork [2] %132 {handshake.bb = 2 : ui32, handshake.name = "fork20"} : <>
    %134 = init %133#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init14"} : <>
    %135:2 = fork [2] %134 {handshake.bb = 2 : ui32, handshake.name = "fork21"} : <>
    %136 = init %135#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init15"} : <>
    %addressResult_40, %dataResult_41, %doneResult = store[%79] %116 %outputs#1 {handshake.bb = 2 : ui32, handshake.deps = #handshake<deps[["load5", 0, false], ["store1", 0, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store1"} : <i10>, <f32>, <>, <i10>, <f32>, <>
    %137 = addi %67#2, %77 {handshake.bb = 2 : ui32, handshake.name = "addi0"} : <i32>
    %139 = br %137 {handshake.bb = 2 : ui32, handshake.name = "br5"} : <i32>
    %140 = br %65 {handshake.bb = 2 : ui32, handshake.name = "br6"} : <i32>
    %141 = br %72#1 {handshake.bb = 2 : ui32, handshake.name = "br7"} : <>
    %result_42, %index_43 = control_merge [%falseResult_7]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>] to <>, <i1>
    sink %index_43 {handshake.name = "sink16"} : <i1>
    %142:3 = fork [3] %result_42 {handshake.bb = 3 : ui32, handshake.name = "fork22"} : <>
    end {handshake.bb = 3 : ui32, handshake.name = "end0"} %memEnd_3, %memEnd_1, %memEnd, %0#2 : <>, <>, <>, <>
  }
}

