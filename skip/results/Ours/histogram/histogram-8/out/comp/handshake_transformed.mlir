module {
  handshake.func @histogram(%arg0: memref<1000xi32>, %arg1: memref<1000xf32>, %arg2: memref<1000xf32>, %arg3: !handshake.channel<i32>, %arg4: !handshake.control<>, %arg5: !handshake.control<>, %arg6: !handshake.control<>, %arg7: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["feature", "weight", "hist", "n", "feature_start", "weight_start", "hist_start", "start"], cfg.edges = "[0,1][2,1][1,2,3,cmpi0]", resNames = ["feature_end", "weight_end", "hist_end", "end"]} {
    %0:13 = fork [13] %arg7 {handshake.bb = 0 : ui32, handshake.name = "fork0"} : <>
    %outputs:2, %memEnd = mem_controller[%arg2 : memref<1000xf32>] %arg6 (%110, %addressResult_62, %addressResult_64, %dataResult_65) %226#2 {connectedBlocks = [2 : i32], handshake.name = "mem_controller2"} :    (!handshake.channel<i32>, !handshake.channel<i10>, !handshake.channel<i10>, !handshake.channel<f32>) -> (!handshake.channel<f32>, !handshake.control<>)
    %outputs_0, %memEnd_1 = mem_controller[%arg1 : memref<1000xf32>] %arg5 (%addressResult_44) %226#1 {connectedBlocks = [2 : i32], handshake.name = "mem_controller3"} :    (!handshake.channel<i10>) -> !handshake.channel<f32>
    %outputs_2, %memEnd_3 = mem_controller[%arg0 : memref<1000xi32>] %arg4 (%addressResult) %226#0 {connectedBlocks = [2 : i32], handshake.name = "mem_controller4"} :    (!handshake.channel<i10>) -> !handshake.channel<i32>
    %1 = constant %0#1 {handshake.bb = 0 : ui32, handshake.name = "constant11", value = 1000 : i11} : <>, <i11>
    %2:8 = fork [8] %1 {handshake.bb = 0 : ui32, handshake.name = "fork1"} : <i11>
    %3 = extsi %2#0 {handshake.bb = 1 : ui32, handshake.name = "extsi0"} : <i11> to <i32>
    %5 = extsi %2#1 {handshake.bb = 1 : ui32, handshake.name = "extsi1"} : <i11> to <i32>
    %7 = extsi %2#2 {handshake.bb = 1 : ui32, handshake.name = "extsi2"} : <i11> to <i32>
    %9 = extsi %2#3 {handshake.bb = 1 : ui32, handshake.name = "extsi3"} : <i11> to <i32>
    %11 = extsi %2#4 {handshake.bb = 1 : ui32, handshake.name = "extsi4"} : <i11> to <i32>
    %13 = extsi %2#5 {handshake.bb = 1 : ui32, handshake.name = "extsi5"} : <i11> to <i32>
    %15 = extsi %2#6 {handshake.bb = 1 : ui32, handshake.name = "extsi6"} : <i11> to <i32>
    %17 = extsi %2#7 {handshake.bb = 1 : ui32, handshake.name = "extsi7"} : <i11> to <i32>
    %19 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant12", value = false} : <>, <i1>
    %20 = br %19 {handshake.bb = 0 : ui32, handshake.name = "br2"} : <i1>
    %21 = extsi %20 {handshake.bb = 0 : ui32, handshake.name = "extsi11"} : <i1> to <i32>
    %22 = br %arg3 {handshake.bb = 0 : ui32, handshake.name = "br3"} : <i32>
    %23 = br %0#12 {handshake.bb = 0 : ui32, handshake.name = "br4"} : <>
    %24 = mux %67#0 [%3, %192#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux2"} : <i1>, [<i32>, <i32>] to <i32>
    %27 = mux %67#1 [%5, %201#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux3"} : <i1>, [<i32>, <i32>] to <i32>
    %30 = mux %67#2 [%7, %202] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux4"} : <i1>, [<i32>, <i32>] to <i32>
    %32 = mux %67#3 [%9, %195#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux5"} : <i1>, [<i32>, <i32>] to <i32>
    %35 = mux %67#4 [%11, %198#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux6"} : <i1>, [<i32>, <i32>] to <i32>
    %38 = mux %67#5 [%13, %39] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux7"} : <i1>, [<i32>, <i32>] to <i32>
    %39 = buffer %186#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer19"} : <i32>
    %41 = mux %67#6 [%0#11, %220] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux8"} : <i1>, [<>, <>] to <>
    %43 = mux %67#7 [%15, %183#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux9"} : <i1>, [<i32>, <i32>] to <i32>
    %46 = mux %67#8 [%17, %189#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux10"} : <i1>, [<i32>, <i32>] to <i32>
    %49 = mux %50 [%0#10, %209#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux11"} : <i1>, [<>, <>] to <>
    %50 = buffer %67#9, bufferType = FIFO_BREAK_NONE, numSlots = 6 {handshake.bb = 1 : ui32, handshake.name = "buffer26"} : <i1>
    %51 = mux %52 [%0#9, %217#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux12"} : <i1>, [<>, <>] to <>
    %52 = buffer %67#10, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 1 : ui32, handshake.name = "buffer27"} : <i1>
    %53 = mux %54 [%0#8, %207#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux13"} : <i1>, [<>, <>] to <>
    %54 = buffer %67#11, bufferType = FIFO_BREAK_NONE, numSlots = 7 {handshake.bb = 1 : ui32, handshake.name = "buffer28"} : <i1>
    %55 = mux %56 [%0#7, %219#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux14"} : <i1>, [<>, <>] to <>
    %56 = buffer %67#12, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer29"} : <i1>
    %57 = mux %58 [%0#6, %211#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux15"} : <i1>, [<>, <>] to <>
    %58 = buffer %67#13, bufferType = FIFO_BREAK_NONE, numSlots = 5 {handshake.bb = 1 : ui32, handshake.name = "buffer30"} : <i1>
    %59 = mux %60 [%0#5, %213#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux16"} : <i1>, [<>, <>] to <>
    %60 = buffer %67#14, bufferType = FIFO_BREAK_NONE, numSlots = 4 {handshake.bb = 1 : ui32, handshake.name = "buffer31"} : <i1>
    %61 = mux %62 [%0#4, %205#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux17"} : <i1>, [<>, <>] to <>
    %62 = buffer %67#15, bufferType = FIFO_BREAK_NONE, numSlots = 8 {handshake.bb = 1 : ui32, handshake.name = "buffer32"} : <i1>
    %63 = mux %64 [%0#3, %215#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux18"} : <i1>, [<>, <>] to <>
    %64 = buffer %67#16, bufferType = FIFO_BREAK_NONE, numSlots = 3 {handshake.bb = 1 : ui32, handshake.name = "buffer33"} : <i1>
    %65 = init %66 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "init0"} : <i1>
    %66 = buffer %78#20, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 1 : ui32, handshake.name = "buffer34"} : <i1>
    %67:17 = fork [17] %65 {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <i1>
    %68 = mux %74#0 [%21, %223] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i32>, <i32>] to <i32>
    %70:2 = fork [2] %68 {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <i32>
    %71 = mux %74#1 [%22, %224] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i1>, [<i32>, <i32>] to <i32>
    %73:2 = fork [2] %71 {handshake.bb = 1 : ui32, handshake.name = "fork4"} : <i32>
    %result, %index = control_merge [%23, %225]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %74:2 = fork [2] %index {handshake.bb = 1 : ui32, handshake.name = "fork5"} : <i1>
    %75 = cmpi slt, %70#1, %73#1 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i32>
    %78:21 = fork [21] %75 {handshake.bb = 1 : ui32, handshake.name = "fork6"} : <i1>
    %trueResult, %falseResult = cond_br %78#19, %73#0 {handshake.bb = 1 : ui32, handshake.name = "cond_br1"} : <i1>, <i32>
    sink %falseResult {handshake.name = "sink0"} : <i32>
    %trueResult_4, %falseResult_5 = cond_br %78#18, %70#0 {handshake.bb = 1 : ui32, handshake.name = "cond_br2"} : <i1>, <i32>
    sink %falseResult_5 {handshake.name = "sink1"} : <i32>
    %trueResult_6, %falseResult_7 = cond_br %78#17, %result {handshake.bb = 1 : ui32, handshake.name = "cond_br3"} : <i1>, <>
    %trueResult_8, %falseResult_9 = cond_br %84, %24 {handshake.bb = 2 : ui32, handshake.name = "cond_br28"} : <i1>, <i32>
    %84 = buffer %78#16, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer44"} : <i1>
    sink %falseResult_9 {handshake.name = "sink2"} : <i32>
    %trueResult_10, %falseResult_11 = cond_br %85, %53 {handshake.bb = 2 : ui32, handshake.name = "cond_br29"} : <i1>, <>
    %85 = buffer %78#15, bufferType = FIFO_BREAK_NONE, numSlots = 8 {handshake.bb = 2 : ui32, handshake.name = "buffer45"} : <i1>
    sink %falseResult_11 {handshake.name = "sink3"} : <>
    %trueResult_12, %falseResult_13 = cond_br %86, %51 {handshake.bb = 2 : ui32, handshake.name = "cond_br30"} : <i1>, <>
    %86 = buffer %78#14, bufferType = FIFO_BREAK_NONE, numSlots = 3 {handshake.bb = 2 : ui32, handshake.name = "buffer46"} : <i1>
    sink %falseResult_13 {handshake.name = "sink4"} : <>
    %trueResult_14, %falseResult_15 = cond_br %87, %61 {handshake.bb = 2 : ui32, handshake.name = "cond_br31"} : <i1>, <>
    %87 = buffer %78#13, bufferType = FIFO_BREAK_NONE, numSlots = 9 {handshake.bb = 2 : ui32, handshake.name = "buffer47"} : <i1>
    sink %falseResult_15 {handshake.name = "sink5"} : <>
    %trueResult_16, %falseResult_17 = cond_br %88, %46 {handshake.bb = 2 : ui32, handshake.name = "cond_br32"} : <i1>, <i32>
    %88 = buffer %78#12, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer48"} : <i1>
    sink %falseResult_17 {handshake.name = "sink6"} : <i32>
    %trueResult_18, %falseResult_19 = cond_br %89, %38 {handshake.bb = 2 : ui32, handshake.name = "cond_br33"} : <i1>, <i32>
    %89 = buffer %78#11, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer49"} : <i1>
    sink %falseResult_19 {handshake.name = "sink7"} : <i32>
    %trueResult_20, %falseResult_21 = cond_br %90, %43 {handshake.bb = 2 : ui32, handshake.name = "cond_br34"} : <i1>, <i32>
    %90 = buffer %78#10, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer50"} : <i1>
    sink %falseResult_21 {handshake.name = "sink8"} : <i32>
    %trueResult_22, %falseResult_23 = cond_br %91, %41 {handshake.bb = 2 : ui32, handshake.name = "cond_br35"} : <i1>, <>
    %91 = buffer %78#9, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer51"} : <i1>
    sink %falseResult_23 {handshake.name = "sink9"} : <>
    %trueResult_24, %falseResult_25 = cond_br %92, %63 {handshake.bb = 2 : ui32, handshake.name = "cond_br36"} : <i1>, <>
    %92 = buffer %78#8, bufferType = FIFO_BREAK_NONE, numSlots = 4 {handshake.bb = 2 : ui32, handshake.name = "buffer52"} : <i1>
    sink %falseResult_25 {handshake.name = "sink10"} : <>
    %trueResult_26, %falseResult_27 = cond_br %93, %27 {handshake.bb = 2 : ui32, handshake.name = "cond_br37"} : <i1>, <i32>
    %93 = buffer %78#7, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer53"} : <i1>
    sink %falseResult_27 {handshake.name = "sink11"} : <i32>
    %trueResult_28, %falseResult_29 = cond_br %94, %57 {handshake.bb = 2 : ui32, handshake.name = "cond_br38"} : <i1>, <>
    %94 = buffer %78#6, bufferType = FIFO_BREAK_NONE, numSlots = 6 {handshake.bb = 2 : ui32, handshake.name = "buffer54"} : <i1>
    sink %falseResult_29 {handshake.name = "sink12"} : <>
    %trueResult_30, %falseResult_31 = cond_br %95, %55 {handshake.bb = 2 : ui32, handshake.name = "cond_br39"} : <i1>, <>
    %95 = buffer %78#5, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 2 : ui32, handshake.name = "buffer55"} : <i1>
    sink %falseResult_31 {handshake.name = "sink13"} : <>
    %trueResult_32, %falseResult_33 = cond_br %96, %59 {handshake.bb = 2 : ui32, handshake.name = "cond_br40"} : <i1>, <>
    %96 = buffer %78#4, bufferType = FIFO_BREAK_NONE, numSlots = 5 {handshake.bb = 2 : ui32, handshake.name = "buffer56"} : <i1>
    sink %falseResult_33 {handshake.name = "sink14"} : <>
    %trueResult_34, %falseResult_35 = cond_br %97, %35 {handshake.bb = 2 : ui32, handshake.name = "cond_br41"} : <i1>, <i32>
    %97 = buffer %78#3, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer57"} : <i1>
    sink %falseResult_35 {handshake.name = "sink15"} : <i32>
    %trueResult_36, %falseResult_37 = cond_br %98, %32 {handshake.bb = 2 : ui32, handshake.name = "cond_br42"} : <i1>, <i32>
    %98 = buffer %78#2, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer58"} : <i1>
    sink %falseResult_37 {handshake.name = "sink16"} : <i32>
    %trueResult_38, %falseResult_39 = cond_br %99, %49 {handshake.bb = 2 : ui32, handshake.name = "cond_br43"} : <i1>, <>
    %99 = buffer %78#1, bufferType = FIFO_BREAK_NONE, numSlots = 7 {handshake.bb = 2 : ui32, handshake.name = "buffer59"} : <i1>
    sink %falseResult_39 {handshake.name = "sink17"} : <>
    %trueResult_40, %falseResult_41 = cond_br %100, %30 {handshake.bb = 2 : ui32, handshake.name = "cond_br44"} : <i1>, <i32>
    %100 = buffer %78#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer60"} : <i1>
    sink %falseResult_41 {handshake.name = "sink18"} : <i32>
    %101 = merge %trueResult {handshake.bb = 2 : ui32, handshake.name = "merge0"} : <i32>
    %102 = merge %trueResult_4 {handshake.bb = 2 : ui32, handshake.name = "merge1"} : <i32>
    %103:3 = fork [3] %102 {handshake.bb = 2 : ui32, handshake.name = "fork7"} : <i32>
    %104 = trunci %105 {handshake.bb = 2 : ui32, handshake.name = "trunci0"} : <i32> to <i10>
    %105 = buffer %103#0, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 2 : ui32, handshake.name = "buffer61"} : <i32>
    %106 = trunci %103#1 {handshake.bb = 2 : ui32, handshake.name = "trunci1"} : <i32> to <i10>
    %result_42, %index_43 = control_merge [%trueResult_6]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>] to <>, <i1>
    sink %index_43 {handshake.name = "sink19"} : <i1>
    %108:2 = fork [2] %result_42 {handshake.bb = 2 : ui32, handshake.name = "fork8"} : <>
    %109 = constant %108#0 {handshake.bb = 2 : ui32, handshake.name = "constant13", value = 1 : i2} : <>, <i2>
    %110 = extsi %109 {handshake.bb = 2 : ui32, handshake.name = "extsi9"} : <i2> to <i32>
    %111 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %112 = constant %111 {handshake.bb = 2 : ui32, handshake.name = "constant14", value = 1 : i2} : <>, <i2>
    %113 = extsi %112 {handshake.bb = 2 : ui32, handshake.name = "extsi10"} : <i2> to <i32>
    %addressResult, %dataResult = load[%106] %outputs_2 {handshake.bb = 2 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i10>, <i32>, <i10>, <i32>
    %114:4 = fork [4] %dataResult {handshake.bb = 2 : ui32, handshake.name = "fork9"} : <i32>
    %115 = trunci %116 {handshake.bb = 2 : ui32, handshake.name = "trunci2"} : <i32> to <i10>
    %116 = buffer %114#0, bufferType = FIFO_BREAK_NONE, numSlots = 8 {handshake.bb = 2 : ui32, handshake.name = "buffer63"} : <i32>
    %addressResult_44, %dataResult_45 = load[%104] %outputs_0 {handshake.bb = 2 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i10>, <f32>, <i10>, <f32>
    %117 = gate %114#1, %trueResult_22 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "gate0"} : <i32>, !handshake.control<> to <i32>
    %119:8 = fork [8] %117 {handshake.bb = 2 : ui32, handshake.name = "fork10"} : <i32>
    %120 = cmpi ne, %119#7, %trueResult_20 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi1"} : <i32>
    %122:2 = fork [2] %120 {handshake.bb = 2 : ui32, handshake.name = "fork11"} : <i1>
    %123 = cmpi ne, %119#6, %trueResult_18 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi2"} : <i32>
    %125:2 = fork [2] %123 {handshake.bb = 2 : ui32, handshake.name = "fork12"} : <i1>
    %126 = cmpi ne, %119#5, %trueResult_16 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi3"} : <i32>
    %128:2 = fork [2] %126 {handshake.bb = 2 : ui32, handshake.name = "fork13"} : <i1>
    %129 = cmpi ne, %119#4, %trueResult_8 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi4"} : <i32>
    %131:2 = fork [2] %129 {handshake.bb = 2 : ui32, handshake.name = "fork14"} : <i1>
    %132 = cmpi ne, %119#3, %trueResult_36 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi5"} : <i32>
    %134:2 = fork [2] %132 {handshake.bb = 2 : ui32, handshake.name = "fork15"} : <i1>
    %135 = cmpi ne, %119#2, %trueResult_34 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi6"} : <i32>
    %137:2 = fork [2] %135 {handshake.bb = 2 : ui32, handshake.name = "fork16"} : <i1>
    %138 = cmpi ne, %119#1, %trueResult_26 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi7"} : <i32>
    %140:2 = fork [2] %138 {handshake.bb = 2 : ui32, handshake.name = "fork17"} : <i1>
    %141 = cmpi ne, %119#0, %trueResult_40 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi8"} : <i32>
    %143:2 = fork [2] %141 {handshake.bb = 2 : ui32, handshake.name = "fork18"} : <i1>
    %trueResult_46, %falseResult_47 = cond_br %144, %trueResult_14 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br20"} : <i1>, <>
    %144 = buffer %122#1, bufferType = FIFO_BREAK_NONE, numSlots = 8 {handshake.bb = 2 : ui32, handshake.name = "buffer73"} : <i1>
    sink %trueResult_46 {handshake.name = "sink20"} : <>
    %trueResult_48, %falseResult_49 = cond_br %145, %trueResult_10 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br21"} : <i1>, <>
    %145 = buffer %125#1, bufferType = FIFO_BREAK_NONE, numSlots = 7 {handshake.bb = 2 : ui32, handshake.name = "buffer74"} : <i1>
    sink %trueResult_48 {handshake.name = "sink21"} : <>
    %trueResult_50, %falseResult_51 = cond_br %146, %trueResult_38 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br22"} : <i1>, <>
    %146 = buffer %128#1, bufferType = FIFO_BREAK_NONE, numSlots = 6 {handshake.bb = 2 : ui32, handshake.name = "buffer75"} : <i1>
    sink %trueResult_50 {handshake.name = "sink22"} : <>
    %trueResult_52, %falseResult_53 = cond_br %147, %trueResult_28 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br23"} : <i1>, <>
    %147 = buffer %131#1, bufferType = FIFO_BREAK_NONE, numSlots = 5 {handshake.bb = 2 : ui32, handshake.name = "buffer76"} : <i1>
    sink %trueResult_52 {handshake.name = "sink23"} : <>
    %trueResult_54, %falseResult_55 = cond_br %148, %trueResult_32 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br24"} : <i1>, <>
    %148 = buffer %134#1, bufferType = FIFO_BREAK_NONE, numSlots = 4 {handshake.bb = 2 : ui32, handshake.name = "buffer77"} : <i1>
    sink %trueResult_54 {handshake.name = "sink24"} : <>
    %trueResult_56, %falseResult_57 = cond_br %149, %trueResult_24 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br25"} : <i1>, <>
    %149 = buffer %137#1, bufferType = FIFO_BREAK_NONE, numSlots = 3 {handshake.bb = 2 : ui32, handshake.name = "buffer78"} : <i1>
    sink %trueResult_56 {handshake.name = "sink25"} : <>
    %trueResult_58, %falseResult_59 = cond_br %150, %trueResult_12 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br26"} : <i1>, <>
    %150 = buffer %140#1, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 2 : ui32, handshake.name = "buffer79"} : <i1>
    sink %trueResult_58 {handshake.name = "sink26"} : <>
    %trueResult_60, %falseResult_61 = cond_br %151, %trueResult_30 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br27"} : <i1>, <>
    %151 = buffer %143#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer80"} : <i1>
    sink %trueResult_60 {handshake.name = "sink27"} : <>
    %152 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %153 = mux %154 [%falseResult_47, %152] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux19"} : <i1>, [<>, <>] to <>
    %154 = buffer %122#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer81"} : <i1>
    %155 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source2"} : <>
    %156 = mux %157 [%falseResult_49, %155] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux20"} : <i1>, [<>, <>] to <>
    %157 = buffer %125#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer82"} : <i1>
    %158 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source3"} : <>
    %159 = mux %160 [%falseResult_51, %158] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux21"} : <i1>, [<>, <>] to <>
    %160 = buffer %128#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer83"} : <i1>
    %161 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source4"} : <>
    %162 = mux %163 [%falseResult_53, %161] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux22"} : <i1>, [<>, <>] to <>
    %163 = buffer %131#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer84"} : <i1>
    %164 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source5"} : <>
    %165 = mux %166 [%falseResult_55, %164] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux23"} : <i1>, [<>, <>] to <>
    %166 = buffer %134#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer85"} : <i1>
    %167 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source6"} : <>
    %168 = mux %169 [%falseResult_57, %167] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux24"} : <i1>, [<>, <>] to <>
    %169 = buffer %137#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer86"} : <i1>
    %170 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source7"} : <>
    %171 = mux %172 [%falseResult_59, %170] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux25"} : <i1>, [<>, <>] to <>
    %172 = buffer %140#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer87"} : <i1>
    %173 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source8"} : <>
    %174 = mux %175 [%falseResult_61, %173] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux26"} : <i1>, [<>, <>] to <>
    %175 = buffer %143#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer88"} : <i1>
    %176 = join %153, %156, %159, %162, %165, %168, %171, %174 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "join0"} : <>
    %177 = gate %178, %176 {handshake.bb = 2 : ui32, handshake.name = "gate1"} : <i32>, !handshake.control<> to <i32>
    %178 = buffer %114#2, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer89"} : <i32>
    %179 = trunci %177 {handshake.bb = 2 : ui32, handshake.name = "trunci3"} : <i32> to <i10>
    %addressResult_62, %dataResult_63 = load[%179] %outputs#0 {handshake.bb = 2 : ui32, handshake.deps = #handshake<deps[["store1", 0, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load5"} : <i10>, <f32>, <i10>, <f32>
    %180 = addf %dataResult_63, %dataResult_45 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "addf0"} : <f32>
    %181 = buffer %114#3, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer0"} : <i32>
    %183:2 = fork [2] %181 {handshake.bb = 2 : ui32, handshake.name = "fork19"} : <i32>
    %184 = init %183#0 {handshake.bb = 2 : ui32, handshake.name = "init17"} : <i32>
    %186:2 = fork [2] %184 {handshake.bb = 2 : ui32, handshake.name = "fork20"} : <i32>
    %187 = init %188 {handshake.bb = 2 : ui32, handshake.name = "init18"} : <i32>
    %188 = buffer %186#0, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 2 : ui32, handshake.name = "buffer92"} : <i32>
    %189:2 = fork [2] %187 {handshake.bb = 2 : ui32, handshake.name = "fork21"} : <i32>
    %190 = init %191 {handshake.bb = 2 : ui32, handshake.name = "init19"} : <i32>
    %191 = buffer %189#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer93"} : <i32>
    %192:2 = fork [2] %190 {handshake.bb = 2 : ui32, handshake.name = "fork22"} : <i32>
    %193 = init %194 {handshake.bb = 2 : ui32, handshake.name = "init20"} : <i32>
    %194 = buffer %192#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer94"} : <i32>
    %195:2 = fork [2] %193 {handshake.bb = 2 : ui32, handshake.name = "fork23"} : <i32>
    %196 = init %197 {handshake.bb = 2 : ui32, handshake.name = "init21"} : <i32>
    %197 = buffer %195#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer95"} : <i32>
    %198:2 = fork [2] %196 {handshake.bb = 2 : ui32, handshake.name = "fork24"} : <i32>
    %199 = init %200 {handshake.bb = 2 : ui32, handshake.name = "init22"} : <i32>
    %200 = buffer %198#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer96"} : <i32>
    %201:2 = fork [2] %199 {handshake.bb = 2 : ui32, handshake.name = "fork25"} : <i32>
    %202 = init %203 {handshake.bb = 2 : ui32, handshake.name = "init23"} : <i32>
    %203 = buffer %201#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer97"} : <i32>
    %204 = buffer %doneResult, bufferType = FIFO_BREAK_NONE, numSlots = 1 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "buffer1"} : <>
    %205:2 = fork [2] %204 {handshake.bb = 2 : ui32, handshake.name = "fork26"} : <>
    %206 = init %205#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init24"} : <>
    %207:2 = fork [2] %206 {handshake.bb = 2 : ui32, handshake.name = "fork27"} : <>
    %208 = init %207#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init25"} : <>
    %209:2 = fork [2] %208 {handshake.bb = 2 : ui32, handshake.name = "fork28"} : <>
    %210 = init %209#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init26"} : <>
    %211:2 = fork [2] %210 {handshake.bb = 2 : ui32, handshake.name = "fork29"} : <>
    %212 = init %211#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init27"} : <>
    %213:2 = fork [2] %212 {handshake.bb = 2 : ui32, handshake.name = "fork30"} : <>
    %214 = init %213#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init28"} : <>
    %215:2 = fork [2] %214 {handshake.bb = 2 : ui32, handshake.name = "fork31"} : <>
    %216 = init %215#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init29"} : <>
    %217:2 = fork [2] %216 {handshake.bb = 2 : ui32, handshake.name = "fork32"} : <>
    %218 = init %217#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init30"} : <>
    %219:2 = fork [2] %218 {handshake.bb = 2 : ui32, handshake.name = "fork33"} : <>
    %220 = init %219#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init31"} : <>
    %addressResult_64, %dataResult_65, %doneResult = store[%115] %180 %outputs#1 {handshake.bb = 2 : ui32, handshake.deps = #handshake<deps[["load5", 0, false], ["store1", 0, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store1"} : <i10>, <f32>, <>, <i10>, <f32>, <>
    %221 = addi %103#2, %113 {handshake.bb = 2 : ui32, handshake.name = "addi0"} : <i32>
    %223 = br %221 {handshake.bb = 2 : ui32, handshake.name = "br5"} : <i32>
    %224 = br %101 {handshake.bb = 2 : ui32, handshake.name = "br6"} : <i32>
    %225 = br %108#1 {handshake.bb = 2 : ui32, handshake.name = "br7"} : <>
    %result_66, %index_67 = control_merge [%falseResult_7]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>] to <>, <i1>
    sink %index_67 {handshake.name = "sink28"} : <i1>
    %226:3 = fork [3] %result_66 {handshake.bb = 3 : ui32, handshake.name = "fork34"} : <>
    end {handshake.bb = 3 : ui32, handshake.name = "end0"} %memEnd_3, %memEnd_1, %memEnd, %0#2 : <>, <>, <>, <>
  }
}

