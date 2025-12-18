module {
  handshake.func @histogram(%arg0: memref<1000xi32>, %arg1: memref<1000xf32>, %arg2: memref<1000xf32>, %arg3: !handshake.channel<i32>, %arg4: !handshake.control<>, %arg5: !handshake.control<>, %arg6: !handshake.control<>, %arg7: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["feature", "weight", "hist", "n", "feature_start", "weight_start", "hist_start", "start"], cfg.edges = "[0,1][2,1][1,2,3,cmpi0]", resNames = ["feature_end", "weight_end", "hist_end", "end"]} {
    %0:11 = fork [11] %arg7 {handshake.bb = 0 : ui32, handshake.name = "fork0"} : <>
    %outputs:2, %memEnd = mem_controller[%arg2 : memref<1000xf32>] %arg6 (%92, %addressResult_50, %addressResult_52, %dataResult_53) %184#2 {connectedBlocks = [2 : i32], handshake.name = "mem_controller2"} :    (!handshake.channel<i32>, !handshake.channel<i10>, !handshake.channel<i10>, !handshake.channel<f32>) -> (!handshake.channel<f32>, !handshake.control<>)
    %outputs_0, %memEnd_1 = mem_controller[%arg1 : memref<1000xf32>] %arg5 (%addressResult_36) %184#1 {connectedBlocks = [2 : i32], handshake.name = "mem_controller3"} :    (!handshake.channel<i10>) -> !handshake.channel<f32>
    %outputs_2, %memEnd_3 = mem_controller[%arg0 : memref<1000xi32>] %arg4 (%addressResult) %184#0 {connectedBlocks = [2 : i32], handshake.name = "mem_controller4"} :    (!handshake.channel<i10>) -> !handshake.channel<i32>
    %1 = constant %0#1 {handshake.bb = 0 : ui32, handshake.name = "constant9", value = 1000 : i11} : <>, <i11>
    %2:6 = fork [6] %1 {handshake.bb = 0 : ui32, handshake.name = "fork1"} : <i11>
    %3 = extsi %2#0 {handshake.bb = 1 : ui32, handshake.name = "extsi0"} : <i11> to <i32>
    %5 = extsi %2#1 {handshake.bb = 1 : ui32, handshake.name = "extsi1"} : <i11> to <i32>
    %7 = extsi %2#2 {handshake.bb = 1 : ui32, handshake.name = "extsi2"} : <i11> to <i32>
    %9 = extsi %2#3 {handshake.bb = 1 : ui32, handshake.name = "extsi3"} : <i11> to <i32>
    %11 = extsi %2#4 {handshake.bb = 1 : ui32, handshake.name = "extsi4"} : <i11> to <i32>
    %13 = extsi %2#5 {handshake.bb = 1 : ui32, handshake.name = "extsi5"} : <i11> to <i32>
    %15 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant10", value = false} : <>, <i1>
    %16 = br %15 {handshake.bb = 0 : ui32, handshake.name = "br2"} : <i1>
    %17 = extsi %16 {handshake.bb = 0 : ui32, handshake.name = "extsi9"} : <i1> to <i32>
    %18 = br %arg3 {handshake.bb = 0 : ui32, handshake.name = "br3"} : <i32>
    %19 = br %0#10 {handshake.bb = 0 : ui32, handshake.name = "br4"} : <>
    %20 = mux %53#0 [%3, %21] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux2"} : <i1>, [<i32>, <i32>] to <i32>
    %21 = buffer %154#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer8"} : <i32>
    %23 = mux %53#1 [%5, %160#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux3"} : <i1>, [<i32>, <i32>] to <i32>
    %26 = mux %53#2 [%7, %157#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux4"} : <i1>, [<i32>, <i32>] to <i32>
    %29 = mux %53#3 [%9, %151#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux5"} : <i1>, [<i32>, <i32>] to <i32>
    %32 = mux %53#4 [%11, %163#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux6"} : <i1>, [<i32>, <i32>] to <i32>
    %35 = mux %53#5 [%0#9, %178] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux7"} : <i1>, [<>, <>] to <>
    %37 = mux %53#6 [%13, %164] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux8"} : <i1>, [<i32>, <i32>] to <i32>
    %39 = mux %40 [%0#8, %177#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux9"} : <i1>, [<>, <>] to <>
    %40 = buffer %53#7, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer20"} : <i1>
    %41 = mux %42 [%0#7, %169#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux10"} : <i1>, [<>, <>] to <>
    %42 = buffer %53#8, bufferType = FIFO_BREAK_NONE, numSlots = 5 {handshake.bb = 1 : ui32, handshake.name = "buffer21"} : <i1>
    %43 = mux %44 [%0#6, %175#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux11"} : <i1>, [<>, <>] to <>
    %44 = buffer %53#9, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 1 : ui32, handshake.name = "buffer22"} : <i1>
    %45 = mux %46 [%0#5, %171#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux12"} : <i1>, [<>, <>] to <>
    %46 = buffer %53#10, bufferType = FIFO_BREAK_NONE, numSlots = 4 {handshake.bb = 1 : ui32, handshake.name = "buffer23"} : <i1>
    %47 = mux %48 [%0#4, %167#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux13"} : <i1>, [<>, <>] to <>
    %48 = buffer %53#11, bufferType = FIFO_BREAK_NONE, numSlots = 6 {handshake.bb = 1 : ui32, handshake.name = "buffer24"} : <i1>
    %49 = mux %50 [%0#3, %173#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux14"} : <i1>, [<>, <>] to <>
    %50 = buffer %53#12, bufferType = FIFO_BREAK_NONE, numSlots = 3 {handshake.bb = 1 : ui32, handshake.name = "buffer25"} : <i1>
    %51 = init %52 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "init0"} : <i1>
    %52 = buffer %64#16, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 1 : ui32, handshake.name = "buffer26"} : <i1>
    %53:13 = fork [13] %51 {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <i1>
    %54 = mux %60#0 [%17, %181] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i32>, <i32>] to <i32>
    %56:2 = fork [2] %54 {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <i32>
    %57 = mux %60#1 [%18, %182] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i1>, [<i32>, <i32>] to <i32>
    %59:2 = fork [2] %57 {handshake.bb = 1 : ui32, handshake.name = "fork4"} : <i32>
    %result, %index = control_merge [%19, %183]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %60:2 = fork [2] %index {handshake.bb = 1 : ui32, handshake.name = "fork5"} : <i1>
    %61 = cmpi slt, %56#1, %59#1 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i32>
    %64:17 = fork [17] %61 {handshake.bb = 1 : ui32, handshake.name = "fork6"} : <i1>
    %trueResult, %falseResult = cond_br %64#15, %59#0 {handshake.bb = 1 : ui32, handshake.name = "cond_br1"} : <i1>, <i32>
    sink %falseResult {handshake.name = "sink0"} : <i32>
    %trueResult_4, %falseResult_5 = cond_br %64#14, %56#0 {handshake.bb = 1 : ui32, handshake.name = "cond_br2"} : <i1>, <i32>
    sink %falseResult_5 {handshake.name = "sink1"} : <i32>
    %trueResult_6, %falseResult_7 = cond_br %64#13, %result {handshake.bb = 1 : ui32, handshake.name = "cond_br3"} : <i1>, <>
    %trueResult_8, %falseResult_9 = cond_br %70, %20 {handshake.bb = 2 : ui32, handshake.name = "cond_br22"} : <i1>, <i32>
    %70 = buffer %64#12, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer36"} : <i1>
    sink %falseResult_9 {handshake.name = "sink2"} : <i32>
    %trueResult_10, %falseResult_11 = cond_br %71, %32 {handshake.bb = 2 : ui32, handshake.name = "cond_br23"} : <i1>, <i32>
    %71 = buffer %64#11, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer37"} : <i1>
    sink %falseResult_11 {handshake.name = "sink3"} : <i32>
    %trueResult_12, %falseResult_13 = cond_br %72, %49 {handshake.bb = 2 : ui32, handshake.name = "cond_br24"} : <i1>, <>
    %72 = buffer %64#10, bufferType = FIFO_BREAK_NONE, numSlots = 4 {handshake.bb = 2 : ui32, handshake.name = "buffer38"} : <i1>
    sink %falseResult_13 {handshake.name = "sink4"} : <>
    %trueResult_14, %falseResult_15 = cond_br %73, %47 {handshake.bb = 2 : ui32, handshake.name = "cond_br25"} : <i1>, <>
    %73 = buffer %64#9, bufferType = FIFO_BREAK_NONE, numSlots = 7 {handshake.bb = 2 : ui32, handshake.name = "buffer39"} : <i1>
    sink %falseResult_15 {handshake.name = "sink5"} : <>
    %trueResult_16, %falseResult_17 = cond_br %74, %29 {handshake.bb = 2 : ui32, handshake.name = "cond_br26"} : <i1>, <i32>
    %74 = buffer %64#8, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer40"} : <i1>
    sink %falseResult_17 {handshake.name = "sink6"} : <i32>
    %trueResult_18, %falseResult_19 = cond_br %75, %37 {handshake.bb = 2 : ui32, handshake.name = "cond_br27"} : <i1>, <i32>
    %75 = buffer %64#7, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer41"} : <i1>
    sink %falseResult_19 {handshake.name = "sink7"} : <i32>
    %trueResult_20, %falseResult_21 = cond_br %76, %45 {handshake.bb = 2 : ui32, handshake.name = "cond_br28"} : <i1>, <>
    %76 = buffer %64#6, bufferType = FIFO_BREAK_NONE, numSlots = 5 {handshake.bb = 2 : ui32, handshake.name = "buffer42"} : <i1>
    sink %falseResult_21 {handshake.name = "sink8"} : <>
    %trueResult_22, %falseResult_23 = cond_br %77, %23 {handshake.bb = 2 : ui32, handshake.name = "cond_br29"} : <i1>, <i32>
    %77 = buffer %64#5, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer43"} : <i1>
    sink %falseResult_23 {handshake.name = "sink9"} : <i32>
    %trueResult_24, %falseResult_25 = cond_br %78, %43 {handshake.bb = 2 : ui32, handshake.name = "cond_br30"} : <i1>, <>
    %78 = buffer %64#4, bufferType = FIFO_BREAK_NONE, numSlots = 3 {handshake.bb = 2 : ui32, handshake.name = "buffer44"} : <i1>
    sink %falseResult_25 {handshake.name = "sink10"} : <>
    %trueResult_26, %falseResult_27 = cond_br %79, %41 {handshake.bb = 2 : ui32, handshake.name = "cond_br31"} : <i1>, <>
    %79 = buffer %64#3, bufferType = FIFO_BREAK_NONE, numSlots = 6 {handshake.bb = 2 : ui32, handshake.name = "buffer45"} : <i1>
    sink %falseResult_27 {handshake.name = "sink11"} : <>
    %trueResult_28, %falseResult_29 = cond_br %80, %35 {handshake.bb = 2 : ui32, handshake.name = "cond_br32"} : <i1>, <>
    %80 = buffer %64#2, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer46"} : <i1>
    sink %falseResult_29 {handshake.name = "sink12"} : <>
    %trueResult_30, %falseResult_31 = cond_br %81, %39 {handshake.bb = 2 : ui32, handshake.name = "cond_br33"} : <i1>, <>
    %81 = buffer %64#1, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 2 : ui32, handshake.name = "buffer47"} : <i1>
    sink %falseResult_31 {handshake.name = "sink13"} : <>
    %trueResult_32, %falseResult_33 = cond_br %82, %26 {handshake.bb = 2 : ui32, handshake.name = "cond_br34"} : <i1>, <i32>
    %82 = buffer %64#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer48"} : <i1>
    sink %falseResult_33 {handshake.name = "sink14"} : <i32>
    %83 = merge %trueResult {handshake.bb = 2 : ui32, handshake.name = "merge0"} : <i32>
    %84 = merge %trueResult_4 {handshake.bb = 2 : ui32, handshake.name = "merge1"} : <i32>
    %85:3 = fork [3] %84 {handshake.bb = 2 : ui32, handshake.name = "fork7"} : <i32>
    %86 = trunci %87 {handshake.bb = 2 : ui32, handshake.name = "trunci0"} : <i32> to <i10>
    %87 = buffer %85#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer49"} : <i32>
    %88 = trunci %85#1 {handshake.bb = 2 : ui32, handshake.name = "trunci1"} : <i32> to <i10>
    %result_34, %index_35 = control_merge [%trueResult_6]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>] to <>, <i1>
    sink %index_35 {handshake.name = "sink15"} : <i1>
    %90:2 = fork [2] %result_34 {handshake.bb = 2 : ui32, handshake.name = "fork8"} : <>
    %91 = constant %90#0 {handshake.bb = 2 : ui32, handshake.name = "constant11", value = 1 : i2} : <>, <i2>
    %92 = extsi %91 {handshake.bb = 2 : ui32, handshake.name = "extsi7"} : <i2> to <i32>
    %93 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %94 = constant %93 {handshake.bb = 2 : ui32, handshake.name = "constant12", value = 1 : i2} : <>, <i2>
    %95 = extsi %94 {handshake.bb = 2 : ui32, handshake.name = "extsi8"} : <i2> to <i32>
    %addressResult, %dataResult = load[%88] %outputs_2 {handshake.bb = 2 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i10>, <i32>, <i10>, <i32>
    %96:4 = fork [4] %dataResult {handshake.bb = 2 : ui32, handshake.name = "fork9"} : <i32>
    %97 = trunci %98 {handshake.bb = 2 : ui32, handshake.name = "trunci2"} : <i32> to <i10>
    %98 = buffer %96#0, bufferType = FIFO_BREAK_NONE, numSlots = 6 {handshake.bb = 2 : ui32, handshake.name = "buffer51"} : <i32>
    %addressResult_36, %dataResult_37 = load[%86] %outputs_0 {handshake.bb = 2 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i10>, <f32>, <i10>, <f32>
    %99 = gate %96#1, %trueResult_28 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "gate0"} : <i32>, !handshake.control<> to <i32>
    %101:6 = fork [6] %99 {handshake.bb = 2 : ui32, handshake.name = "fork10"} : <i32>
    %102 = cmpi ne, %101#5, %trueResult_16 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi1"} : <i32>
    %104:2 = fork [2] %102 {handshake.bb = 2 : ui32, handshake.name = "fork11"} : <i1>
    %105 = cmpi ne, %101#4, %trueResult_8 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi2"} : <i32>
    %107:2 = fork [2] %105 {handshake.bb = 2 : ui32, handshake.name = "fork12"} : <i1>
    %108 = cmpi ne, %101#3, %trueResult_32 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi3"} : <i32>
    %110:2 = fork [2] %108 {handshake.bb = 2 : ui32, handshake.name = "fork13"} : <i1>
    %111 = cmpi ne, %101#2, %trueResult_22 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi4"} : <i32>
    %113:2 = fork [2] %111 {handshake.bb = 2 : ui32, handshake.name = "fork14"} : <i1>
    %114 = cmpi ne, %101#1, %trueResult_10 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi5"} : <i32>
    %116:2 = fork [2] %114 {handshake.bb = 2 : ui32, handshake.name = "fork15"} : <i1>
    %117 = cmpi ne, %101#0, %trueResult_18 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi6"} : <i32>
    %119:2 = fork [2] %117 {handshake.bb = 2 : ui32, handshake.name = "fork16"} : <i1>
    %trueResult_38, %falseResult_39 = cond_br %120, %trueResult_14 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br16"} : <i1>, <>
    %120 = buffer %104#1, bufferType = FIFO_BREAK_NONE, numSlots = 6 {handshake.bb = 2 : ui32, handshake.name = "buffer59"} : <i1>
    sink %trueResult_38 {handshake.name = "sink16"} : <>
    %trueResult_40, %falseResult_41 = cond_br %121, %trueResult_26 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br17"} : <i1>, <>
    %121 = buffer %107#1, bufferType = FIFO_BREAK_NONE, numSlots = 5 {handshake.bb = 2 : ui32, handshake.name = "buffer60"} : <i1>
    sink %trueResult_40 {handshake.name = "sink17"} : <>
    %trueResult_42, %falseResult_43 = cond_br %122, %trueResult_20 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br18"} : <i1>, <>
    %122 = buffer %110#1, bufferType = FIFO_BREAK_NONE, numSlots = 4 {handshake.bb = 2 : ui32, handshake.name = "buffer61"} : <i1>
    sink %trueResult_42 {handshake.name = "sink18"} : <>
    %trueResult_44, %falseResult_45 = cond_br %123, %trueResult_12 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br19"} : <i1>, <>
    %123 = buffer %113#1, bufferType = FIFO_BREAK_NONE, numSlots = 3 {handshake.bb = 2 : ui32, handshake.name = "buffer62"} : <i1>
    sink %trueResult_44 {handshake.name = "sink19"} : <>
    %trueResult_46, %falseResult_47 = cond_br %124, %trueResult_24 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br20"} : <i1>, <>
    %124 = buffer %116#1, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 2 : ui32, handshake.name = "buffer63"} : <i1>
    sink %trueResult_46 {handshake.name = "sink20"} : <>
    %trueResult_48, %falseResult_49 = cond_br %125, %trueResult_30 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br21"} : <i1>, <>
    %125 = buffer %119#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer64"} : <i1>
    sink %trueResult_48 {handshake.name = "sink21"} : <>
    %126 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %127 = mux %104#0 [%falseResult_39, %126] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux15"} : <i1>, [<>, <>] to <>
    %129 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source2"} : <>
    %130 = mux %107#0 [%falseResult_41, %129] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux16"} : <i1>, [<>, <>] to <>
    %132 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source3"} : <>
    %133 = mux %110#0 [%falseResult_43, %132] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux17"} : <i1>, [<>, <>] to <>
    %135 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source4"} : <>
    %136 = mux %113#0 [%falseResult_45, %135] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux18"} : <i1>, [<>, <>] to <>
    %138 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source5"} : <>
    %139 = mux %116#0 [%falseResult_47, %138] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux19"} : <i1>, [<>, <>] to <>
    %141 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source6"} : <>
    %142 = mux %119#0 [%falseResult_49, %141] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux20"} : <i1>, [<>, <>] to <>
    %144 = join %127, %130, %133, %136, %139, %142 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "join0"} : <>
    %145 = gate %96#2, %144 {handshake.bb = 2 : ui32, handshake.name = "gate1"} : <i32>, !handshake.control<> to <i32>
    %147 = trunci %145 {handshake.bb = 2 : ui32, handshake.name = "trunci3"} : <i32> to <i10>
    %addressResult_50, %dataResult_51 = load[%147] %outputs#0 {handshake.bb = 2 : ui32, handshake.deps = #handshake<deps[["store1", 0, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load5"} : <i10>, <f32>, <i10>, <f32>
    %148 = addf %dataResult_51, %dataResult_37 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "addf0"} : <f32>
    %149 = buffer %96#3, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer0"} : <i32>
    %151:2 = fork [2] %149 {handshake.bb = 2 : ui32, handshake.name = "fork17"} : <i32>
    %152 = init %151#0 {handshake.bb = 2 : ui32, handshake.name = "init13"} : <i32>
    %154:2 = fork [2] %152 {handshake.bb = 2 : ui32, handshake.name = "fork18"} : <i32>
    %155 = init %156 {handshake.bb = 2 : ui32, handshake.name = "init14"} : <i32>
    %156 = buffer %154#0, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 2 : ui32, handshake.name = "buffer74"} : <i32>
    %157:2 = fork [2] %155 {handshake.bb = 2 : ui32, handshake.name = "fork19"} : <i32>
    %158 = init %159 {handshake.bb = 2 : ui32, handshake.name = "init15"} : <i32>
    %159 = buffer %157#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer75"} : <i32>
    %160:2 = fork [2] %158 {handshake.bb = 2 : ui32, handshake.name = "fork20"} : <i32>
    %161 = init %162 {handshake.bb = 2 : ui32, handshake.name = "init16"} : <i32>
    %162 = buffer %160#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer76"} : <i32>
    %163:2 = fork [2] %161 {handshake.bb = 2 : ui32, handshake.name = "fork21"} : <i32>
    %164 = init %165 {handshake.bb = 2 : ui32, handshake.name = "init17"} : <i32>
    %165 = buffer %163#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer77"} : <i32>
    %166 = buffer %doneResult, bufferType = FIFO_BREAK_NONE, numSlots = 1 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "buffer1"} : <>
    %167:2 = fork [2] %166 {handshake.bb = 2 : ui32, handshake.name = "fork22"} : <>
    %168 = init %167#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init18"} : <>
    %169:2 = fork [2] %168 {handshake.bb = 2 : ui32, handshake.name = "fork23"} : <>
    %170 = init %169#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init19"} : <>
    %171:2 = fork [2] %170 {handshake.bb = 2 : ui32, handshake.name = "fork24"} : <>
    %172 = init %171#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init20"} : <>
    %173:2 = fork [2] %172 {handshake.bb = 2 : ui32, handshake.name = "fork25"} : <>
    %174 = init %173#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init21"} : <>
    %175:2 = fork [2] %174 {handshake.bb = 2 : ui32, handshake.name = "fork26"} : <>
    %176 = init %175#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init22"} : <>
    %177:2 = fork [2] %176 {handshake.bb = 2 : ui32, handshake.name = "fork27"} : <>
    %178 = init %177#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init23"} : <>
    %addressResult_52, %dataResult_53, %doneResult = store[%97] %148 %outputs#1 {handshake.bb = 2 : ui32, handshake.deps = #handshake<deps[["load5", 0, false], ["store1", 0, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store1"} : <i10>, <f32>, <>, <i10>, <f32>, <>
    %179 = addi %85#2, %95 {handshake.bb = 2 : ui32, handshake.name = "addi0"} : <i32>
    %181 = br %179 {handshake.bb = 2 : ui32, handshake.name = "br5"} : <i32>
    %182 = br %83 {handshake.bb = 2 : ui32, handshake.name = "br6"} : <i32>
    %183 = br %90#1 {handshake.bb = 2 : ui32, handshake.name = "br7"} : <>
    %result_54, %index_55 = control_merge [%falseResult_7]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>] to <>, <i1>
    sink %index_55 {handshake.name = "sink22"} : <i1>
    %184:3 = fork [3] %result_54 {handshake.bb = 3 : ui32, handshake.name = "fork28"} : <>
    end {handshake.bb = 3 : ui32, handshake.name = "end0"} %memEnd_3, %memEnd_1, %memEnd, %0#2 : <>, <>, <>, <>
  }
}

