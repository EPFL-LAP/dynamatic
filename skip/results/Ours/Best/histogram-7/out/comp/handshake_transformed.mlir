module {
  handshake.func @histogram(%arg0: memref<1000xi32>, %arg1: memref<1000xf32>, %arg2: memref<1000xf32>, %arg3: !handshake.channel<i32>, %arg4: !handshake.control<>, %arg5: !handshake.control<>, %arg6: !handshake.control<>, %arg7: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["feature", "weight", "hist", "n", "feature_start", "weight_start", "hist_start", "start"], cfg.edges = "[0,1][2,1][1,2,3,cmpi0]", resNames = ["feature_end", "weight_end", "hist_end", "end"]} {
    %0:12 = fork [12] %arg7 {handshake.bb = 0 : ui32, handshake.name = "fork0"} : <>
    %outputs:2, %memEnd = mem_controller[%arg2 : memref<1000xf32>] %arg6 (%101, %addressResult_56, %addressResult_58, %dataResult_59) %205#2 {connectedBlocks = [2 : i32], handshake.name = "mem_controller2"} :    (!handshake.channel<i32>, !handshake.channel<i10>, !handshake.channel<i10>, !handshake.channel<f32>) -> (!handshake.channel<f32>, !handshake.control<>)
    %outputs_0, %memEnd_1 = mem_controller[%arg1 : memref<1000xf32>] %arg5 (%addressResult_40) %205#1 {connectedBlocks = [2 : i32], handshake.name = "mem_controller3"} :    (!handshake.channel<i10>) -> !handshake.channel<f32>
    %outputs_2, %memEnd_3 = mem_controller[%arg0 : memref<1000xi32>] %arg4 (%addressResult) %205#0 {connectedBlocks = [2 : i32], handshake.name = "mem_controller4"} :    (!handshake.channel<i10>) -> !handshake.channel<i32>
    %1 = constant %0#1 {handshake.bb = 0 : ui32, handshake.name = "constant10", value = 1000 : i11} : <>, <i11>
    %2:7 = fork [7] %1 {handshake.bb = 0 : ui32, handshake.name = "fork1"} : <i11>
    %3 = extsi %2#0 {handshake.bb = 1 : ui32, handshake.name = "extsi0"} : <i11> to <i32>
    %5 = extsi %2#1 {handshake.bb = 1 : ui32, handshake.name = "extsi1"} : <i11> to <i32>
    %7 = extsi %2#2 {handshake.bb = 1 : ui32, handshake.name = "extsi2"} : <i11> to <i32>
    %9 = extsi %2#3 {handshake.bb = 1 : ui32, handshake.name = "extsi3"} : <i11> to <i32>
    %11 = extsi %2#4 {handshake.bb = 1 : ui32, handshake.name = "extsi4"} : <i11> to <i32>
    %13 = extsi %2#5 {handshake.bb = 1 : ui32, handshake.name = "extsi5"} : <i11> to <i32>
    %15 = extsi %2#6 {handshake.bb = 1 : ui32, handshake.name = "extsi6"} : <i11> to <i32>
    %17 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant11", value = false} : <>, <i1>
    %18 = br %17 {handshake.bb = 0 : ui32, handshake.name = "br2"} : <i1>
    %19 = extsi %18 {handshake.bb = 0 : ui32, handshake.name = "extsi10"} : <i1> to <i32>
    %20 = br %arg3 {handshake.bb = 0 : ui32, handshake.name = "br3"} : <i32>
    %21 = br %0#11 {handshake.bb = 0 : ui32, handshake.name = "br4"} : <>
    %22 = mux %60#0 [%3, %23] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux2"} : <i1>, [<i32>, <i32>] to <i32>
    %23 = buffer %170#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer9"} : <i32>
    %25 = mux %60#1 [%5, %182#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux3"} : <i1>, [<i32>, <i32>] to <i32>
    %28 = mux %60#2 [%7, %183] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux4"} : <i1>, [<i32>, <i32>] to <i32>
    %30 = mux %60#3 [%0#10, %199] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux5"} : <i1>, [<>, <>] to <>
    %32 = mux %60#4 [%9, %179#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux6"} : <i1>, [<i32>, <i32>] to <i32>
    %35 = mux %60#5 [%11, %167#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux7"} : <i1>, [<i32>, <i32>] to <i32>
    %38 = mux %60#6 [%13, %173#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux8"} : <i1>, [<i32>, <i32>] to <i32>
    %41 = mux %60#7 [%15, %176#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux9"} : <i1>, [<i32>, <i32>] to <i32>
    %44 = mux %45 [%0#9, %196#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux10"} : <i1>, [<>, <>] to <>
    %45 = buffer %60#8, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 1 : ui32, handshake.name = "buffer23"} : <i1>
    %46 = mux %47 [%0#8, %188#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux11"} : <i1>, [<>, <>] to <>
    %47 = buffer %60#9, bufferType = FIFO_BREAK_NONE, numSlots = 6 {handshake.bb = 1 : ui32, handshake.name = "buffer24"} : <i1>
    %48 = mux %49 [%0#7, %194#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux12"} : <i1>, [<>, <>] to <>
    %49 = buffer %60#10, bufferType = FIFO_BREAK_NONE, numSlots = 3 {handshake.bb = 1 : ui32, handshake.name = "buffer25"} : <i1>
    %50 = mux %51 [%0#6, %190#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux13"} : <i1>, [<>, <>] to <>
    %51 = buffer %60#11, bufferType = FIFO_BREAK_NONE, numSlots = 5 {handshake.bb = 1 : ui32, handshake.name = "buffer26"} : <i1>
    %52 = mux %53 [%0#5, %186#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux14"} : <i1>, [<>, <>] to <>
    %53 = buffer %60#12, bufferType = FIFO_BREAK_NONE, numSlots = 7 {handshake.bb = 1 : ui32, handshake.name = "buffer27"} : <i1>
    %54 = mux %55 [%0#4, %192#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux15"} : <i1>, [<>, <>] to <>
    %55 = buffer %60#13, bufferType = FIFO_BREAK_NONE, numSlots = 4 {handshake.bb = 1 : ui32, handshake.name = "buffer28"} : <i1>
    %56 = mux %57 [%0#3, %198#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux16"} : <i1>, [<>, <>] to <>
    %57 = buffer %60#14, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer29"} : <i1>
    %58 = init %59 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "init0"} : <i1>
    %59 = buffer %71#18, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 1 : ui32, handshake.name = "buffer30"} : <i1>
    %60:15 = fork [15] %58 {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <i1>
    %61 = mux %67#0 [%19, %202] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i32>, <i32>] to <i32>
    %63:2 = fork [2] %61 {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <i32>
    %64 = mux %67#1 [%20, %203] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i1>, [<i32>, <i32>] to <i32>
    %66:2 = fork [2] %64 {handshake.bb = 1 : ui32, handshake.name = "fork4"} : <i32>
    %result, %index = control_merge [%21, %204]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %67:2 = fork [2] %index {handshake.bb = 1 : ui32, handshake.name = "fork5"} : <i1>
    %68 = cmpi slt, %63#1, %66#1 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i32>
    %71:19 = fork [19] %68 {handshake.bb = 1 : ui32, handshake.name = "fork6"} : <i1>
    %trueResult, %falseResult = cond_br %71#17, %66#0 {handshake.bb = 1 : ui32, handshake.name = "cond_br1"} : <i1>, <i32>
    sink %falseResult {handshake.name = "sink0"} : <i32>
    %trueResult_4, %falseResult_5 = cond_br %71#16, %63#0 {handshake.bb = 1 : ui32, handshake.name = "cond_br2"} : <i1>, <i32>
    sink %falseResult_5 {handshake.name = "sink1"} : <i32>
    %trueResult_6, %falseResult_7 = cond_br %71#15, %result {handshake.bb = 1 : ui32, handshake.name = "cond_br3"} : <i1>, <>
    %trueResult_8, %falseResult_9 = cond_br %77, %56 {handshake.bb = 2 : ui32, handshake.name = "cond_br25"} : <i1>, <>
    %77 = buffer %71#14, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 2 : ui32, handshake.name = "buffer40"} : <i1>
    sink %falseResult_9 {handshake.name = "sink2"} : <>
    %trueResult_10, %falseResult_11 = cond_br %78, %52 {handshake.bb = 2 : ui32, handshake.name = "cond_br26"} : <i1>, <>
    %78 = buffer %71#13, bufferType = FIFO_BREAK_NONE, numSlots = 8 {handshake.bb = 2 : ui32, handshake.name = "buffer41"} : <i1>
    sink %falseResult_11 {handshake.name = "sink3"} : <>
    %trueResult_12, %falseResult_13 = cond_br %79, %46 {handshake.bb = 2 : ui32, handshake.name = "cond_br27"} : <i1>, <>
    %79 = buffer %71#12, bufferType = FIFO_BREAK_NONE, numSlots = 7 {handshake.bb = 2 : ui32, handshake.name = "buffer42"} : <i1>
    sink %falseResult_13 {handshake.name = "sink4"} : <>
    %trueResult_14, %falseResult_15 = cond_br %80, %41 {handshake.bb = 2 : ui32, handshake.name = "cond_br28"} : <i1>, <i32>
    %80 = buffer %71#11, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer43"} : <i1>
    sink %falseResult_15 {handshake.name = "sink5"} : <i32>
    %trueResult_16, %falseResult_17 = cond_br %81, %32 {handshake.bb = 2 : ui32, handshake.name = "cond_br29"} : <i1>, <i32>
    %81 = buffer %71#10, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer44"} : <i1>
    sink %falseResult_17 {handshake.name = "sink6"} : <i32>
    %trueResult_18, %falseResult_19 = cond_br %82, %54 {handshake.bb = 2 : ui32, handshake.name = "cond_br30"} : <i1>, <>
    %82 = buffer %71#9, bufferType = FIFO_BREAK_NONE, numSlots = 5 {handshake.bb = 2 : ui32, handshake.name = "buffer45"} : <i1>
    sink %falseResult_19 {handshake.name = "sink7"} : <>
    %trueResult_20, %falseResult_21 = cond_br %83, %25 {handshake.bb = 2 : ui32, handshake.name = "cond_br31"} : <i1>, <i32>
    %83 = buffer %71#8, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer46"} : <i1>
    sink %falseResult_21 {handshake.name = "sink8"} : <i32>
    %trueResult_22, %falseResult_23 = cond_br %84, %48 {handshake.bb = 2 : ui32, handshake.name = "cond_br32"} : <i1>, <>
    %84 = buffer %71#7, bufferType = FIFO_BREAK_NONE, numSlots = 4 {handshake.bb = 2 : ui32, handshake.name = "buffer47"} : <i1>
    sink %falseResult_23 {handshake.name = "sink9"} : <>
    %trueResult_24, %falseResult_25 = cond_br %85, %35 {handshake.bb = 2 : ui32, handshake.name = "cond_br33"} : <i1>, <i32>
    %85 = buffer %71#6, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer48"} : <i1>
    sink %falseResult_25 {handshake.name = "sink10"} : <i32>
    %trueResult_26, %falseResult_27 = cond_br %86, %22 {handshake.bb = 2 : ui32, handshake.name = "cond_br34"} : <i1>, <i32>
    %86 = buffer %71#5, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer49"} : <i1>
    sink %falseResult_27 {handshake.name = "sink11"} : <i32>
    %trueResult_28, %falseResult_29 = cond_br %87, %30 {handshake.bb = 2 : ui32, handshake.name = "cond_br35"} : <i1>, <>
    %87 = buffer %71#4, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer50"} : <i1>
    sink %falseResult_29 {handshake.name = "sink12"} : <>
    %trueResult_30, %falseResult_31 = cond_br %88, %44 {handshake.bb = 2 : ui32, handshake.name = "cond_br36"} : <i1>, <>
    %88 = buffer %71#3, bufferType = FIFO_BREAK_NONE, numSlots = 3 {handshake.bb = 2 : ui32, handshake.name = "buffer51"} : <i1>
    sink %falseResult_31 {handshake.name = "sink13"} : <>
    %trueResult_32, %falseResult_33 = cond_br %89, %38 {handshake.bb = 2 : ui32, handshake.name = "cond_br37"} : <i1>, <i32>
    %89 = buffer %71#2, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer52"} : <i1>
    sink %falseResult_33 {handshake.name = "sink14"} : <i32>
    %trueResult_34, %falseResult_35 = cond_br %90, %28 {handshake.bb = 2 : ui32, handshake.name = "cond_br38"} : <i1>, <i32>
    %90 = buffer %71#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer53"} : <i1>
    sink %falseResult_35 {handshake.name = "sink15"} : <i32>
    %trueResult_36, %falseResult_37 = cond_br %91, %50 {handshake.bb = 2 : ui32, handshake.name = "cond_br39"} : <i1>, <>
    %91 = buffer %71#0, bufferType = FIFO_BREAK_NONE, numSlots = 6 {handshake.bb = 2 : ui32, handshake.name = "buffer54"} : <i1>
    sink %falseResult_37 {handshake.name = "sink16"} : <>
    %92 = merge %trueResult {handshake.bb = 2 : ui32, handshake.name = "merge0"} : <i32>
    %93 = merge %trueResult_4 {handshake.bb = 2 : ui32, handshake.name = "merge1"} : <i32>
    %94:3 = fork [3] %93 {handshake.bb = 2 : ui32, handshake.name = "fork7"} : <i32>
    %95 = trunci %96 {handshake.bb = 2 : ui32, handshake.name = "trunci0"} : <i32> to <i10>
    %96 = buffer %94#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer55"} : <i32>
    %97 = trunci %94#1 {handshake.bb = 2 : ui32, handshake.name = "trunci1"} : <i32> to <i10>
    %result_38, %index_39 = control_merge [%trueResult_6]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>] to <>, <i1>
    sink %index_39 {handshake.name = "sink17"} : <i1>
    %99:2 = fork [2] %result_38 {handshake.bb = 2 : ui32, handshake.name = "fork8"} : <>
    %100 = constant %99#0 {handshake.bb = 2 : ui32, handshake.name = "constant12", value = 1 : i2} : <>, <i2>
    %101 = extsi %100 {handshake.bb = 2 : ui32, handshake.name = "extsi8"} : <i2> to <i32>
    %102 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %103 = constant %102 {handshake.bb = 2 : ui32, handshake.name = "constant13", value = 1 : i2} : <>, <i2>
    %104 = extsi %103 {handshake.bb = 2 : ui32, handshake.name = "extsi9"} : <i2> to <i32>
    %addressResult, %dataResult = load[%97] %outputs_2 {handshake.bb = 2 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i10>, <i32>, <i10>, <i32>
    %105:4 = fork [4] %dataResult {handshake.bb = 2 : ui32, handshake.name = "fork9"} : <i32>
    %106 = trunci %107 {handshake.bb = 2 : ui32, handshake.name = "trunci2"} : <i32> to <i10>
    %107 = buffer %105#0, bufferType = FIFO_BREAK_NONE, numSlots = 7 {handshake.bb = 2 : ui32, handshake.name = "buffer57"} : <i32>
    %addressResult_40, %dataResult_41 = load[%95] %outputs_0 {handshake.bb = 2 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i10>, <f32>, <i10>, <f32>
    %108 = gate %105#1, %trueResult_28 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "gate0"} : <i32>, !handshake.control<> to <i32>
    %110:7 = fork [7] %108 {handshake.bb = 2 : ui32, handshake.name = "fork10"} : <i32>
    %111 = cmpi ne, %110#6, %trueResult_24 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi1"} : <i32>
    %113:2 = fork [2] %111 {handshake.bb = 2 : ui32, handshake.name = "fork11"} : <i1>
    %114 = cmpi ne, %110#5, %trueResult_26 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi2"} : <i32>
    %116:2 = fork [2] %114 {handshake.bb = 2 : ui32, handshake.name = "fork12"} : <i1>
    %117 = cmpi ne, %110#4, %trueResult_32 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi3"} : <i32>
    %119:2 = fork [2] %117 {handshake.bb = 2 : ui32, handshake.name = "fork13"} : <i1>
    %120 = cmpi ne, %110#3, %trueResult_14 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi4"} : <i32>
    %122:2 = fork [2] %120 {handshake.bb = 2 : ui32, handshake.name = "fork14"} : <i1>
    %123 = cmpi ne, %110#2, %trueResult_16 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi5"} : <i32>
    %125:2 = fork [2] %123 {handshake.bb = 2 : ui32, handshake.name = "fork15"} : <i1>
    %126 = cmpi ne, %110#1, %trueResult_20 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi6"} : <i32>
    %128:2 = fork [2] %126 {handshake.bb = 2 : ui32, handshake.name = "fork16"} : <i1>
    %129 = cmpi ne, %110#0, %trueResult_34 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi7"} : <i32>
    %131:2 = fork [2] %129 {handshake.bb = 2 : ui32, handshake.name = "fork17"} : <i1>
    %trueResult_42, %falseResult_43 = cond_br %132, %trueResult_10 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br18"} : <i1>, <>
    %132 = buffer %113#1, bufferType = FIFO_BREAK_NONE, numSlots = 7 {handshake.bb = 2 : ui32, handshake.name = "buffer66"} : <i1>
    sink %trueResult_42 {handshake.name = "sink18"} : <>
    %trueResult_44, %falseResult_45 = cond_br %133, %trueResult_12 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br19"} : <i1>, <>
    %133 = buffer %116#1, bufferType = FIFO_BREAK_NONE, numSlots = 6 {handshake.bb = 2 : ui32, handshake.name = "buffer67"} : <i1>
    sink %trueResult_44 {handshake.name = "sink19"} : <>
    %trueResult_46, %falseResult_47 = cond_br %134, %trueResult_36 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br20"} : <i1>, <>
    %134 = buffer %119#1, bufferType = FIFO_BREAK_NONE, numSlots = 5 {handshake.bb = 2 : ui32, handshake.name = "buffer68"} : <i1>
    sink %trueResult_46 {handshake.name = "sink20"} : <>
    %trueResult_48, %falseResult_49 = cond_br %135, %trueResult_18 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br21"} : <i1>, <>
    %135 = buffer %122#1, bufferType = FIFO_BREAK_NONE, numSlots = 4 {handshake.bb = 2 : ui32, handshake.name = "buffer69"} : <i1>
    sink %trueResult_48 {handshake.name = "sink21"} : <>
    %trueResult_50, %falseResult_51 = cond_br %136, %trueResult_22 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br22"} : <i1>, <>
    %136 = buffer %125#1, bufferType = FIFO_BREAK_NONE, numSlots = 3 {handshake.bb = 2 : ui32, handshake.name = "buffer70"} : <i1>
    sink %trueResult_50 {handshake.name = "sink22"} : <>
    %trueResult_52, %falseResult_53 = cond_br %137, %trueResult_30 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br23"} : <i1>, <>
    %137 = buffer %128#1, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 2 : ui32, handshake.name = "buffer71"} : <i1>
    sink %trueResult_52 {handshake.name = "sink23"} : <>
    %trueResult_54, %falseResult_55 = cond_br %138, %trueResult_8 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br24"} : <i1>, <>
    %138 = buffer %131#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer72"} : <i1>
    sink %trueResult_54 {handshake.name = "sink24"} : <>
    %139 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %140 = mux %113#0 [%falseResult_43, %139] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux17"} : <i1>, [<>, <>] to <>
    %142 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source2"} : <>
    %143 = mux %116#0 [%falseResult_45, %142] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux18"} : <i1>, [<>, <>] to <>
    %145 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source3"} : <>
    %146 = mux %119#0 [%falseResult_47, %145] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux19"} : <i1>, [<>, <>] to <>
    %148 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source4"} : <>
    %149 = mux %122#0 [%falseResult_49, %148] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux20"} : <i1>, [<>, <>] to <>
    %151 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source5"} : <>
    %152 = mux %125#0 [%falseResult_51, %151] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux21"} : <i1>, [<>, <>] to <>
    %154 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source6"} : <>
    %155 = mux %128#0 [%falseResult_53, %154] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux22"} : <i1>, [<>, <>] to <>
    %157 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source7"} : <>
    %158 = mux %131#0 [%falseResult_55, %157] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux23"} : <i1>, [<>, <>] to <>
    %160 = join %140, %143, %146, %149, %152, %155, %158 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "join0"} : <>
    %161 = gate %105#2, %160 {handshake.bb = 2 : ui32, handshake.name = "gate1"} : <i32>, !handshake.control<> to <i32>
    %163 = trunci %161 {handshake.bb = 2 : ui32, handshake.name = "trunci3"} : <i32> to <i10>
    %addressResult_56, %dataResult_57 = load[%163] %outputs#0 {handshake.bb = 2 : ui32, handshake.deps = #handshake<deps[["store1", 0, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load5"} : <i10>, <f32>, <i10>, <f32>
    %164 = addf %dataResult_57, %dataResult_41 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "addf0"} : <f32>
    %165 = buffer %105#3, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer0"} : <i32>
    %167:2 = fork [2] %165 {handshake.bb = 2 : ui32, handshake.name = "fork18"} : <i32>
    %168 = init %167#0 {handshake.bb = 2 : ui32, handshake.name = "init15"} : <i32>
    %170:2 = fork [2] %168 {handshake.bb = 2 : ui32, handshake.name = "fork19"} : <i32>
    %171 = init %172 {handshake.bb = 2 : ui32, handshake.name = "init16"} : <i32>
    %172 = buffer %170#0, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 2 : ui32, handshake.name = "buffer83"} : <i32>
    %173:2 = fork [2] %171 {handshake.bb = 2 : ui32, handshake.name = "fork20"} : <i32>
    %174 = init %175 {handshake.bb = 2 : ui32, handshake.name = "init17"} : <i32>
    %175 = buffer %173#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer84"} : <i32>
    %176:2 = fork [2] %174 {handshake.bb = 2 : ui32, handshake.name = "fork21"} : <i32>
    %177 = init %178 {handshake.bb = 2 : ui32, handshake.name = "init18"} : <i32>
    %178 = buffer %176#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer85"} : <i32>
    %179:2 = fork [2] %177 {handshake.bb = 2 : ui32, handshake.name = "fork22"} : <i32>
    %180 = init %181 {handshake.bb = 2 : ui32, handshake.name = "init19"} : <i32>
    %181 = buffer %179#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer86"} : <i32>
    %182:2 = fork [2] %180 {handshake.bb = 2 : ui32, handshake.name = "fork23"} : <i32>
    %183 = init %184 {handshake.bb = 2 : ui32, handshake.name = "init20"} : <i32>
    %184 = buffer %182#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer87"} : <i32>
    %185 = buffer %doneResult, bufferType = FIFO_BREAK_NONE, numSlots = 1 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "buffer1"} : <>
    %186:2 = fork [2] %185 {handshake.bb = 2 : ui32, handshake.name = "fork24"} : <>
    %187 = init %186#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init21"} : <>
    %188:2 = fork [2] %187 {handshake.bb = 2 : ui32, handshake.name = "fork25"} : <>
    %189 = init %188#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init22"} : <>
    %190:2 = fork [2] %189 {handshake.bb = 2 : ui32, handshake.name = "fork26"} : <>
    %191 = init %190#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init23"} : <>
    %192:2 = fork [2] %191 {handshake.bb = 2 : ui32, handshake.name = "fork27"} : <>
    %193 = init %192#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init24"} : <>
    %194:2 = fork [2] %193 {handshake.bb = 2 : ui32, handshake.name = "fork28"} : <>
    %195 = init %194#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init25"} : <>
    %196:2 = fork [2] %195 {handshake.bb = 2 : ui32, handshake.name = "fork29"} : <>
    %197 = init %196#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init26"} : <>
    %198:2 = fork [2] %197 {handshake.bb = 2 : ui32, handshake.name = "fork30"} : <>
    %199 = init %198#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init27"} : <>
    %addressResult_58, %dataResult_59, %doneResult = store[%106] %164 %outputs#1 {handshake.bb = 2 : ui32, handshake.deps = #handshake<deps[["load5", 0, false], ["store1", 0, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store1"} : <i10>, <f32>, <>, <i10>, <f32>, <>
    %200 = addi %94#2, %104 {handshake.bb = 2 : ui32, handshake.name = "addi0"} : <i32>
    %202 = br %200 {handshake.bb = 2 : ui32, handshake.name = "br5"} : <i32>
    %203 = br %92 {handshake.bb = 2 : ui32, handshake.name = "br6"} : <i32>
    %204 = br %99#1 {handshake.bb = 2 : ui32, handshake.name = "br7"} : <>
    %result_60, %index_61 = control_merge [%falseResult_7]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>] to <>, <i1>
    sink %index_61 {handshake.name = "sink25"} : <i1>
    %205:3 = fork [3] %result_60 {handshake.bb = 3 : ui32, handshake.name = "fork31"} : <>
    end {handshake.bb = 3 : ui32, handshake.name = "end0"} %memEnd_3, %memEnd_1, %memEnd, %0#2 : <>, <>, <>, <>
  }
}

