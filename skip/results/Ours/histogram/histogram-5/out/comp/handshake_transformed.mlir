module {
  handshake.func @histogram(%arg0: memref<1000xi32>, %arg1: memref<1000xf32>, %arg2: memref<1000xf32>, %arg3: !handshake.channel<i32>, %arg4: !handshake.control<>, %arg5: !handshake.control<>, %arg6: !handshake.control<>, %arg7: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["feature", "weight", "hist", "n", "feature_start", "weight_start", "hist_start", "start"], cfg.edges = "[0,1][2,1][1,2,3,cmpi0]", resNames = ["feature_end", "weight_end", "hist_end", "end"]} {
    %0:10 = fork [10] %arg7 {handshake.bb = 0 : ui32, handshake.name = "fork0"} : <>
    %outputs:2, %memEnd = mem_controller[%arg2 : memref<1000xf32>] %arg6 (%83, %addressResult_44, %addressResult_46, %dataResult_47) %163#2 {connectedBlocks = [2 : i32], handshake.name = "mem_controller2"} :    (!handshake.channel<i32>, !handshake.channel<i10>, !handshake.channel<i10>, !handshake.channel<f32>) -> (!handshake.channel<f32>, !handshake.control<>)
    %outputs_0, %memEnd_1 = mem_controller[%arg1 : memref<1000xf32>] %arg5 (%addressResult_32) %163#1 {connectedBlocks = [2 : i32], handshake.name = "mem_controller3"} :    (!handshake.channel<i10>) -> !handshake.channel<f32>
    %outputs_2, %memEnd_3 = mem_controller[%arg0 : memref<1000xi32>] %arg4 (%addressResult) %163#0 {connectedBlocks = [2 : i32], handshake.name = "mem_controller4"} :    (!handshake.channel<i10>) -> !handshake.channel<i32>
    %1 = constant %0#1 {handshake.bb = 0 : ui32, handshake.name = "constant8", value = 1000 : i11} : <>, <i11>
    %2:5 = fork [5] %1 {handshake.bb = 0 : ui32, handshake.name = "fork1"} : <i11>
    %3 = extsi %2#0 {handshake.bb = 1 : ui32, handshake.name = "extsi0"} : <i11> to <i32>
    %5 = extsi %2#1 {handshake.bb = 1 : ui32, handshake.name = "extsi1"} : <i11> to <i32>
    %7 = extsi %2#2 {handshake.bb = 1 : ui32, handshake.name = "extsi2"} : <i11> to <i32>
    %9 = extsi %2#3 {handshake.bb = 1 : ui32, handshake.name = "extsi3"} : <i11> to <i32>
    %11 = extsi %2#4 {handshake.bb = 1 : ui32, handshake.name = "extsi4"} : <i11> to <i32>
    %13 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant9", value = false} : <>, <i1>
    %14 = br %13 {handshake.bb = 0 : ui32, handshake.name = "br2"} : <i1>
    %15 = extsi %14 {handshake.bb = 0 : ui32, handshake.name = "extsi8"} : <i1> to <i32>
    %16 = br %arg3 {handshake.bb = 0 : ui32, handshake.name = "br3"} : <i32>
    %17 = br %0#9 {handshake.bb = 0 : ui32, handshake.name = "br4"} : <>
    %18 = mux %46#0 [%3, %141#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux2"} : <i1>, [<i32>, <i32>] to <i32>
    %21 = mux %46#1 [%5, %22] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux3"} : <i1>, [<i32>, <i32>] to <i32>
    %22 = buffer %138#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer9"} : <i32>
    %24 = mux %46#2 [%7, %144#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux4"} : <i1>, [<i32>, <i32>] to <i32>
    %27 = mux %46#3 [%0#8, %157] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux5"} : <i1>, [<>, <>] to <>
    %29 = mux %46#4 [%9, %135#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux6"} : <i1>, [<i32>, <i32>] to <i32>
    %32 = mux %46#5 [%11, %145] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux7"} : <i1>, [<i32>, <i32>] to <i32>
    %34 = mux %35 [%0#7, %150#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux8"} : <i1>, [<>, <>] to <>
    %35 = buffer %46#6, bufferType = FIFO_BREAK_NONE, numSlots = 4 {handshake.bb = 1 : ui32, handshake.name = "buffer17"} : <i1>
    %36 = mux %37 [%0#6, %152#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux9"} : <i1>, [<>, <>] to <>
    %37 = buffer %46#7, bufferType = FIFO_BREAK_NONE, numSlots = 3 {handshake.bb = 1 : ui32, handshake.name = "buffer18"} : <i1>
    %38 = mux %39 [%0#5, %154#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux10"} : <i1>, [<>, <>] to <>
    %39 = buffer %46#8, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 1 : ui32, handshake.name = "buffer19"} : <i1>
    %40 = mux %41 [%0#4, %156#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux11"} : <i1>, [<>, <>] to <>
    %41 = buffer %46#9, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer20"} : <i1>
    %42 = mux %43 [%0#3, %148#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux12"} : <i1>, [<>, <>] to <>
    %43 = buffer %46#10, bufferType = FIFO_BREAK_NONE, numSlots = 5 {handshake.bb = 1 : ui32, handshake.name = "buffer21"} : <i1>
    %44 = init %45 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "init0"} : <i1>
    %45 = buffer %57#14, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 1 : ui32, handshake.name = "buffer22"} : <i1>
    %46:11 = fork [11] %44 {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <i1>
    %47 = mux %53#0 [%15, %160] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i32>, <i32>] to <i32>
    %49:2 = fork [2] %47 {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <i32>
    %50 = mux %53#1 [%16, %161] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i1>, [<i32>, <i32>] to <i32>
    %52:2 = fork [2] %50 {handshake.bb = 1 : ui32, handshake.name = "fork4"} : <i32>
    %result, %index = control_merge [%17, %162]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %53:2 = fork [2] %index {handshake.bb = 1 : ui32, handshake.name = "fork5"} : <i1>
    %54 = cmpi slt, %49#1, %52#1 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i32>
    %57:15 = fork [15] %54 {handshake.bb = 1 : ui32, handshake.name = "fork6"} : <i1>
    %trueResult, %falseResult = cond_br %57#13, %52#0 {handshake.bb = 1 : ui32, handshake.name = "cond_br1"} : <i1>, <i32>
    sink %falseResult {handshake.name = "sink0"} : <i32>
    %trueResult_4, %falseResult_5 = cond_br %57#12, %49#0 {handshake.bb = 1 : ui32, handshake.name = "cond_br2"} : <i1>, <i32>
    sink %falseResult_5 {handshake.name = "sink1"} : <i32>
    %trueResult_6, %falseResult_7 = cond_br %57#11, %result {handshake.bb = 1 : ui32, handshake.name = "cond_br3"} : <i1>, <>
    %trueResult_8, %falseResult_9 = cond_br %63, %38 {handshake.bb = 2 : ui32, handshake.name = "cond_br19"} : <i1>, <>
    %63 = buffer %57#10, bufferType = FIFO_BREAK_NONE, numSlots = 3 {handshake.bb = 2 : ui32, handshake.name = "buffer32"} : <i1>
    sink %falseResult_9 {handshake.name = "sink2"} : <>
    %trueResult_10, %falseResult_11 = cond_br %64, %42 {handshake.bb = 2 : ui32, handshake.name = "cond_br20"} : <i1>, <>
    %64 = buffer %57#9, bufferType = FIFO_BREAK_NONE, numSlots = 6 {handshake.bb = 2 : ui32, handshake.name = "buffer33"} : <i1>
    sink %falseResult_11 {handshake.name = "sink3"} : <>
    %trueResult_12, %falseResult_13 = cond_br %65, %29 {handshake.bb = 2 : ui32, handshake.name = "cond_br21"} : <i1>, <i32>
    %65 = buffer %57#8, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer34"} : <i1>
    sink %falseResult_13 {handshake.name = "sink4"} : <i32>
    %trueResult_14, %falseResult_15 = cond_br %66, %27 {handshake.bb = 2 : ui32, handshake.name = "cond_br22"} : <i1>, <>
    %66 = buffer %57#7, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer35"} : <i1>
    sink %falseResult_15 {handshake.name = "sink5"} : <>
    %trueResult_16, %falseResult_17 = cond_br %67, %24 {handshake.bb = 2 : ui32, handshake.name = "cond_br23"} : <i1>, <i32>
    %67 = buffer %57#6, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer36"} : <i1>
    sink %falseResult_17 {handshake.name = "sink6"} : <i32>
    %trueResult_18, %falseResult_19 = cond_br %68, %32 {handshake.bb = 2 : ui32, handshake.name = "cond_br24"} : <i1>, <i32>
    %68 = buffer %57#5, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer37"} : <i1>
    sink %falseResult_19 {handshake.name = "sink7"} : <i32>
    %trueResult_20, %falseResult_21 = cond_br %69, %21 {handshake.bb = 2 : ui32, handshake.name = "cond_br25"} : <i1>, <i32>
    %69 = buffer %57#4, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer38"} : <i1>
    sink %falseResult_21 {handshake.name = "sink8"} : <i32>
    %trueResult_22, %falseResult_23 = cond_br %70, %18 {handshake.bb = 2 : ui32, handshake.name = "cond_br26"} : <i1>, <i32>
    %70 = buffer %57#3, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer39"} : <i1>
    sink %falseResult_23 {handshake.name = "sink9"} : <i32>
    %trueResult_24, %falseResult_25 = cond_br %71, %36 {handshake.bb = 2 : ui32, handshake.name = "cond_br27"} : <i1>, <>
    %71 = buffer %57#2, bufferType = FIFO_BREAK_NONE, numSlots = 4 {handshake.bb = 2 : ui32, handshake.name = "buffer40"} : <i1>
    sink %falseResult_25 {handshake.name = "sink10"} : <>
    %trueResult_26, %falseResult_27 = cond_br %72, %34 {handshake.bb = 2 : ui32, handshake.name = "cond_br28"} : <i1>, <>
    %72 = buffer %57#1, bufferType = FIFO_BREAK_NONE, numSlots = 5 {handshake.bb = 2 : ui32, handshake.name = "buffer41"} : <i1>
    sink %falseResult_27 {handshake.name = "sink11"} : <>
    %trueResult_28, %falseResult_29 = cond_br %73, %40 {handshake.bb = 2 : ui32, handshake.name = "cond_br29"} : <i1>, <>
    %73 = buffer %57#0, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 2 : ui32, handshake.name = "buffer42"} : <i1>
    sink %falseResult_29 {handshake.name = "sink12"} : <>
    %74 = merge %trueResult {handshake.bb = 2 : ui32, handshake.name = "merge0"} : <i32>
    %75 = merge %trueResult_4 {handshake.bb = 2 : ui32, handshake.name = "merge1"} : <i32>
    %76:3 = fork [3] %75 {handshake.bb = 2 : ui32, handshake.name = "fork7"} : <i32>
    %77 = trunci %76#0 {handshake.bb = 2 : ui32, handshake.name = "trunci0"} : <i32> to <i10>
    %79 = trunci %76#1 {handshake.bb = 2 : ui32, handshake.name = "trunci1"} : <i32> to <i10>
    %result_30, %index_31 = control_merge [%trueResult_6]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>] to <>, <i1>
    sink %index_31 {handshake.name = "sink13"} : <i1>
    %81:2 = fork [2] %result_30 {handshake.bb = 2 : ui32, handshake.name = "fork8"} : <>
    %82 = constant %81#0 {handshake.bb = 2 : ui32, handshake.name = "constant10", value = 1 : i2} : <>, <i2>
    %83 = extsi %82 {handshake.bb = 2 : ui32, handshake.name = "extsi6"} : <i2> to <i32>
    %84 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %85 = constant %84 {handshake.bb = 2 : ui32, handshake.name = "constant11", value = 1 : i2} : <>, <i2>
    %86 = extsi %85 {handshake.bb = 2 : ui32, handshake.name = "extsi7"} : <i2> to <i32>
    %addressResult, %dataResult = load[%79] %outputs_2 {handshake.bb = 2 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i10>, <i32>, <i10>, <i32>
    %87:4 = fork [4] %dataResult {handshake.bb = 2 : ui32, handshake.name = "fork9"} : <i32>
    %88 = trunci %89 {handshake.bb = 2 : ui32, handshake.name = "trunci2"} : <i32> to <i10>
    %89 = buffer %87#0, bufferType = FIFO_BREAK_NONE, numSlots = 6 {handshake.bb = 2 : ui32, handshake.name = "buffer45"} : <i32>
    %addressResult_32, %dataResult_33 = load[%77] %outputs_0 {handshake.bb = 2 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i10>, <f32>, <i10>, <f32>
    %90 = gate %87#1, %trueResult_14 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "gate0"} : <i32>, !handshake.control<> to <i32>
    %92:5 = fork [5] %90 {handshake.bb = 2 : ui32, handshake.name = "fork10"} : <i32>
    %93 = cmpi ne, %92#4, %trueResult_12 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi1"} : <i32>
    %95:2 = fork [2] %93 {handshake.bb = 2 : ui32, handshake.name = "fork11"} : <i1>
    %96 = cmpi ne, %92#3, %trueResult_20 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi2"} : <i32>
    %98:2 = fork [2] %96 {handshake.bb = 2 : ui32, handshake.name = "fork12"} : <i1>
    %99 = cmpi ne, %92#2, %trueResult_22 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi3"} : <i32>
    %101:2 = fork [2] %99 {handshake.bb = 2 : ui32, handshake.name = "fork13"} : <i1>
    %102 = cmpi ne, %92#1, %trueResult_16 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi4"} : <i32>
    %104:2 = fork [2] %102 {handshake.bb = 2 : ui32, handshake.name = "fork14"} : <i1>
    %105 = cmpi ne, %92#0, %trueResult_18 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi5"} : <i32>
    %107:2 = fork [2] %105 {handshake.bb = 2 : ui32, handshake.name = "fork15"} : <i1>
    %trueResult_34, %falseResult_35 = cond_br %108, %trueResult_10 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br14"} : <i1>, <>
    %108 = buffer %95#1, bufferType = FIFO_BREAK_NONE, numSlots = 5 {handshake.bb = 2 : ui32, handshake.name = "buffer52"} : <i1>
    sink %trueResult_34 {handshake.name = "sink14"} : <>
    %trueResult_36, %falseResult_37 = cond_br %109, %trueResult_26 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br15"} : <i1>, <>
    %109 = buffer %98#1, bufferType = FIFO_BREAK_NONE, numSlots = 4 {handshake.bb = 2 : ui32, handshake.name = "buffer53"} : <i1>
    sink %trueResult_36 {handshake.name = "sink15"} : <>
    %trueResult_38, %falseResult_39 = cond_br %110, %trueResult_24 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br16"} : <i1>, <>
    %110 = buffer %101#1, bufferType = FIFO_BREAK_NONE, numSlots = 3 {handshake.bb = 2 : ui32, handshake.name = "buffer54"} : <i1>
    sink %trueResult_38 {handshake.name = "sink16"} : <>
    %trueResult_40, %falseResult_41 = cond_br %111, %trueResult_8 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br17"} : <i1>, <>
    %111 = buffer %104#1, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 2 : ui32, handshake.name = "buffer55"} : <i1>
    sink %trueResult_40 {handshake.name = "sink17"} : <>
    %trueResult_42, %falseResult_43 = cond_br %112, %trueResult_28 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br18"} : <i1>, <>
    %112 = buffer %107#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer56"} : <i1>
    sink %trueResult_42 {handshake.name = "sink18"} : <>
    %113 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %114 = mux %95#0 [%falseResult_35, %113] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux13"} : <i1>, [<>, <>] to <>
    %116 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source2"} : <>
    %117 = mux %98#0 [%falseResult_37, %116] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux14"} : <i1>, [<>, <>] to <>
    %119 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source3"} : <>
    %120 = mux %101#0 [%falseResult_39, %119] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux15"} : <i1>, [<>, <>] to <>
    %122 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source4"} : <>
    %123 = mux %104#0 [%falseResult_41, %122] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux16"} : <i1>, [<>, <>] to <>
    %125 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source5"} : <>
    %126 = mux %107#0 [%falseResult_43, %125] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux17"} : <i1>, [<>, <>] to <>
    %128 = join %114, %117, %120, %123, %126 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "join0"} : <>
    %129 = gate %87#2, %128 {handshake.bb = 2 : ui32, handshake.name = "gate1"} : <i32>, !handshake.control<> to <i32>
    %131 = trunci %129 {handshake.bb = 2 : ui32, handshake.name = "trunci3"} : <i32> to <i10>
    %addressResult_44, %dataResult_45 = load[%131] %outputs#0 {handshake.bb = 2 : ui32, handshake.deps = #handshake<deps[["store1", 0, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load5"} : <i10>, <f32>, <i10>, <f32>
    %132 = addf %dataResult_45, %dataResult_33 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "addf0"} : <f32>
    %133 = buffer %87#3, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer0"} : <i32>
    %135:2 = fork [2] %133 {handshake.bb = 2 : ui32, handshake.name = "fork16"} : <i32>
    %136 = init %135#0 {handshake.bb = 2 : ui32, handshake.name = "init11"} : <i32>
    %138:2 = fork [2] %136 {handshake.bb = 2 : ui32, handshake.name = "fork17"} : <i32>
    %139 = init %140 {handshake.bb = 2 : ui32, handshake.name = "init12"} : <i32>
    %140 = buffer %138#0, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 2 : ui32, handshake.name = "buffer65"} : <i32>
    %141:2 = fork [2] %139 {handshake.bb = 2 : ui32, handshake.name = "fork18"} : <i32>
    %142 = init %143 {handshake.bb = 2 : ui32, handshake.name = "init13"} : <i32>
    %143 = buffer %141#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer66"} : <i32>
    %144:2 = fork [2] %142 {handshake.bb = 2 : ui32, handshake.name = "fork19"} : <i32>
    %145 = init %146 {handshake.bb = 2 : ui32, handshake.name = "init14"} : <i32>
    %146 = buffer %144#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer67"} : <i32>
    %147 = buffer %doneResult, bufferType = FIFO_BREAK_NONE, numSlots = 1 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "buffer1"} : <>
    %148:2 = fork [2] %147 {handshake.bb = 2 : ui32, handshake.name = "fork20"} : <>
    %149 = init %148#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init15"} : <>
    %150:2 = fork [2] %149 {handshake.bb = 2 : ui32, handshake.name = "fork21"} : <>
    %151 = init %150#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init16"} : <>
    %152:2 = fork [2] %151 {handshake.bb = 2 : ui32, handshake.name = "fork22"} : <>
    %153 = init %152#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init17"} : <>
    %154:2 = fork [2] %153 {handshake.bb = 2 : ui32, handshake.name = "fork23"} : <>
    %155 = init %154#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init18"} : <>
    %156:2 = fork [2] %155 {handshake.bb = 2 : ui32, handshake.name = "fork24"} : <>
    %157 = init %156#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init19"} : <>
    %addressResult_46, %dataResult_47, %doneResult = store[%88] %132 %outputs#1 {handshake.bb = 2 : ui32, handshake.deps = #handshake<deps[["load5", 0, false], ["store1", 0, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store1"} : <i10>, <f32>, <>, <i10>, <f32>, <>
    %158 = addi %76#2, %86 {handshake.bb = 2 : ui32, handshake.name = "addi0"} : <i32>
    %160 = br %158 {handshake.bb = 2 : ui32, handshake.name = "br5"} : <i32>
    %161 = br %74 {handshake.bb = 2 : ui32, handshake.name = "br6"} : <i32>
    %162 = br %81#1 {handshake.bb = 2 : ui32, handshake.name = "br7"} : <>
    %result_48, %index_49 = control_merge [%falseResult_7]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>] to <>, <i1>
    sink %index_49 {handshake.name = "sink19"} : <i1>
    %163:3 = fork [3] %result_48 {handshake.bb = 3 : ui32, handshake.name = "fork25"} : <>
    end {handshake.bb = 3 : ui32, handshake.name = "end0"} %memEnd_3, %memEnd_1, %memEnd, %0#2 : <>, <>, <>, <>
  }
}

