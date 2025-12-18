module {
  handshake.func @histogram(%arg0: memref<1000xi32>, %arg1: memref<1000xf32>, %arg2: memref<1000xf32>, %arg3: !handshake.channel<i32>, %arg4: !handshake.control<>, %arg5: !handshake.control<>, %arg6: !handshake.control<>, %arg7: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["feature", "weight", "hist", "n", "feature_start", "weight_start", "hist_start", "start"], cfg.edges = "[0,1][2,1][1,2,3,cmpi0]", resNames = ["feature_end", "weight_end", "hist_end", "end"]} {
    %0:9 = fork [9] %arg7 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork0"} : <>
    %outputs:2, %memEnd = mem_controller[%arg2 : memref<1000xf32>] %arg6 (%64, %addressResult_36, %addressResult_38, %dataResult_39) %120#2 {connectedBlocks = [2 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "5": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller2"} :    (!handshake.channel<i32>, !handshake.channel<i10>, !handshake.channel<i10>, !handshake.channel<f32>) -> (!handshake.channel<f32>, !handshake.control<>)
    %outputs_0, %memEnd_1 = mem_controller[%arg1 : memref<1000xf32>] %arg5 (%addressResult_26) %120#1 {connectedBlocks = [2 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller3"} :    (!handshake.channel<i10>) -> !handshake.channel<f32>
    %outputs_2, %memEnd_3 = mem_controller[%arg0 : memref<1000xi32>] %arg4 (%addressResult) %120#0 {connectedBlocks = [2 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller4"} :    (!handshake.channel<i10>) -> !handshake.channel<i32>
    %1 = constant %0#1 {handshake.bb = 0 : ui32, handshake.name = "constant7", value = 1000 : i11} : <>, <i11>
    %2:4 = fork [4] %1 {handshake.bb = 0 : ui32, handshake.name = "fork1"} : <i11>
    %3 = extsi %2#0 {handshake.bb = 1 : ui32, handshake.name = "extsi0"} : <i11> to <i32>
    %4 = extsi %2#1 {handshake.bb = 1 : ui32, handshake.name = "extsi1"} : <i11> to <i32>
    %5 = extsi %2#2 {handshake.bb = 1 : ui32, handshake.name = "extsi2"} : <i11> to <i32>
    %6 = extsi %2#3 {handshake.bb = 1 : ui32, handshake.name = "extsi3"} : <i11> to <i32>
    %7 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant8", value = false} : <>, <i1>
    %8 = extsi %7 {handshake.bb = 0 : ui32, handshake.name = "extsi7"} : <i1> to <i32>
    %9 = mux %25#0 [%3, %108] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux2"} : <i1>, [<i32>, <i32>] to <i32>
    %10 = mux %25#1 [%4, %102#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux3"} : <i1>, [<i32>, <i32>] to <i32>
    %11 = mux %25#2 [%5, %107#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux4"} : <i1>, [<i32>, <i32>] to <i32>
    %12 = mux %25#3 [%6, %13] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux5"} : <i1>, [<i32>, <i32>] to <i32>
    %13 = buffer %104#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer11"} : <i32>
    %14 = mux %25#4 [%0#7, %118] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux6"} : <i1>, [<>, <>] to <>
    %15 = mux %16 [%0#6, %115#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux7"} : <i1>, [<>, <>] to <>
    %16 = buffer %25#5, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 1 : ui32, handshake.name = "buffer14"} : <i1>
    %17 = mux %18 [%0#5, %111#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux8"} : <i1>, [<>, <>] to <>
    %18 = buffer %25#6, bufferType = FIFO_BREAK_NONE, numSlots = 4 {handshake.bb = 1 : ui32, handshake.name = "buffer15"} : <i1>
    %19 = mux %20 [%0#4, %117#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux9"} : <i1>, [<>, <>] to <>
    %20 = buffer %25#7, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer16"} : <i1>
    %21 = mux %22 [%0#3, %113#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux10"} : <i1>, [<>, <>] to <>
    %22 = buffer %25#8, bufferType = FIFO_BREAK_NONE, numSlots = 3 {handshake.bb = 1 : ui32, handshake.name = "buffer17"} : <i1>
    %23 = init %24 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "init0"} : <i1>
    %24 = buffer %38#12, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 1 : ui32, handshake.name = "buffer18"} : <i1>
    %25:9 = fork [9] %23 {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <i1>
    %26 = buffer %119, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer37"} : <i32>
    %27 = mux %35#0 [%8, %26] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i32>, <i32>] to <i32>
    %28 = buffer %27, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer13"} : <i32>
    %29 = buffer %28, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer19"} : <i32>
    %30:2 = fork [2] %29 {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <i32>
    %31 = mux %35#1 [%arg3, %trueResult] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i1>, [<i32>, <i32>] to <i32>
    %32 = buffer %31, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer20"} : <i32>
    %33 = buffer %32, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer21"} : <i32>
    %34:2 = fork [2] %33 {handshake.bb = 1 : ui32, handshake.name = "fork4"} : <i32>
    %result, %index = control_merge [%0#8, %62#1]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %35:2 = fork [2] %index {handshake.bb = 1 : ui32, handshake.name = "fork5"} : <i1>
    %36 = cmpi slt, %30#1, %34#1 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i32>
    %37 = buffer %36, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer22"} : <i1>
    %38:13 = fork [13] %37 {handshake.bb = 1 : ui32, handshake.name = "fork6"} : <i1>
    %trueResult, %falseResult = cond_br %38#11, %34#0 {handshake.bb = 1 : ui32, handshake.name = "cond_br1"} : <i1>, <i32>
    sink %falseResult {handshake.name = "sink0"} : <i32>
    %trueResult_4, %falseResult_5 = cond_br %38#10, %30#0 {handshake.bb = 1 : ui32, handshake.name = "cond_br2"} : <i1>, <i32>
    sink %falseResult_5 {handshake.name = "sink1"} : <i32>
    %trueResult_6, %falseResult_7 = cond_br %38#9, %result {handshake.bb = 1 : ui32, handshake.name = "cond_br3"} : <i1>, <>
    %39 = buffer %19, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer10"} : <>
    %trueResult_8, %falseResult_9 = cond_br %40, %39 {handshake.bb = 2 : ui32, handshake.name = "cond_br16"} : <i1>, <>
    %40 = buffer %38#8, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 2 : ui32, handshake.name = "buffer28"} : <i1>
    sink %falseResult_9 {handshake.name = "sink2"} : <>
    %41 = buffer %14, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer6"} : <>
    %42 = buffer %41, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer7"} : <>
    %trueResult_10, %falseResult_11 = cond_br %43, %42 {handshake.bb = 2 : ui32, handshake.name = "cond_br17"} : <i1>, <>
    %43 = buffer %38#7, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer29"} : <i1>
    sink %falseResult_11 {handshake.name = "sink3"} : <>
    %44 = buffer %21, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer12"} : <>
    %trueResult_12, %falseResult_13 = cond_br %45, %44 {handshake.bb = 2 : ui32, handshake.name = "cond_br18"} : <i1>, <>
    %45 = buffer %38#6, bufferType = FIFO_BREAK_NONE, numSlots = 4 {handshake.bb = 2 : ui32, handshake.name = "buffer30"} : <i1>
    sink %falseResult_13 {handshake.name = "sink4"} : <>
    %46 = buffer %10, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer3"} : <i32>
    %trueResult_14, %falseResult_15 = cond_br %47, %46 {handshake.bb = 2 : ui32, handshake.name = "cond_br19"} : <i1>, <i32>
    %47 = buffer %38#5, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer31"} : <i1>
    sink %falseResult_15 {handshake.name = "sink5"} : <i32>
    %48 = buffer %11, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer4"} : <i32>
    %trueResult_16, %falseResult_17 = cond_br %49, %48 {handshake.bb = 2 : ui32, handshake.name = "cond_br20"} : <i1>, <i32>
    %49 = buffer %38#4, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer32"} : <i1>
    sink %falseResult_17 {handshake.name = "sink6"} : <i32>
    %50 = buffer %17, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer9"} : <>
    %trueResult_18, %falseResult_19 = cond_br %51, %50 {handshake.bb = 2 : ui32, handshake.name = "cond_br21"} : <i1>, <>
    %51 = buffer %38#3, bufferType = FIFO_BREAK_NONE, numSlots = 5 {handshake.bb = 2 : ui32, handshake.name = "buffer33"} : <i1>
    sink %falseResult_19 {handshake.name = "sink7"} : <>
    %52 = buffer %12, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer5"} : <i32>
    %trueResult_20, %falseResult_21 = cond_br %53, %52 {handshake.bb = 2 : ui32, handshake.name = "cond_br22"} : <i1>, <i32>
    %53 = buffer %38#2, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer34"} : <i1>
    sink %falseResult_21 {handshake.name = "sink8"} : <i32>
    %54 = buffer %9, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer2"} : <i32>
    %trueResult_22, %falseResult_23 = cond_br %55, %54 {handshake.bb = 2 : ui32, handshake.name = "cond_br23"} : <i1>, <i32>
    %55 = buffer %38#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer35"} : <i1>
    sink %falseResult_23 {handshake.name = "sink9"} : <i32>
    %56 = buffer %15, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer8"} : <>
    %trueResult_24, %falseResult_25 = cond_br %57, %56 {handshake.bb = 2 : ui32, handshake.name = "cond_br24"} : <i1>, <>
    %57 = buffer %38#0, bufferType = FIFO_BREAK_NONE, numSlots = 3 {handshake.bb = 2 : ui32, handshake.name = "buffer36"} : <i1>
    sink %falseResult_25 {handshake.name = "sink10"} : <>
    %58:3 = fork [3] %trueResult_4 {handshake.bb = 2 : ui32, handshake.name = "fork7"} : <i32>
    %59 = trunci %58#0 {handshake.bb = 2 : ui32, handshake.name = "trunci0"} : <i32> to <i10>
    %60 = trunci %58#1 {handshake.bb = 2 : ui32, handshake.name = "trunci1"} : <i32> to <i10>
    %61 = buffer %trueResult_6, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer23"} : <>
    %62:2 = fork [2] %61 {handshake.bb = 2 : ui32, handshake.name = "fork8"} : <>
    %63 = constant %62#0 {handshake.bb = 2 : ui32, handshake.name = "constant9", value = 1 : i2} : <>, <i2>
    %64 = extsi %63 {handshake.bb = 2 : ui32, handshake.name = "extsi5"} : <i2> to <i32>
    %65 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %66 = constant %65 {handshake.bb = 2 : ui32, handshake.name = "constant10", value = 1 : i2} : <>, <i2>
    %67 = extsi %66 {handshake.bb = 2 : ui32, handshake.name = "extsi6"} : <i2> to <i32>
    %addressResult, %dataResult = load[%60] %outputs_2 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i10>, <i32>, <i10>, <i32>
    %68:4 = fork [4] %dataResult {handshake.bb = 2 : ui32, handshake.name = "fork9"} : <i32>
    %69 = trunci %70 {handshake.bb = 2 : ui32, handshake.name = "trunci2"} : <i32> to <i10>
    %70 = buffer %68#0, bufferType = FIFO_BREAK_NONE, numSlots = 4 {handshake.bb = 2 : ui32, handshake.name = "buffer39"} : <i32>
    %addressResult_26, %dataResult_27 = load[%59] %outputs_0 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i10>, <f32>, <i10>, <f32>
    %71 = gate %68#1, %trueResult_10 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "gate0"} : <i32>, !handshake.control<> to <i32>
    %72:4 = fork [4] %71 {handshake.bb = 2 : ui32, handshake.name = "fork10"} : <i32>
    %73 = cmpi ne, %72#3, %trueResult_14 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi1"} : <i32>
    %74:2 = fork [2] %73 {handshake.bb = 2 : ui32, handshake.name = "fork11"} : <i1>
    %75 = cmpi ne, %72#2, %trueResult_20 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi2"} : <i32>
    %76:2 = fork [2] %75 {handshake.bb = 2 : ui32, handshake.name = "fork12"} : <i1>
    %77 = cmpi ne, %72#1, %trueResult_16 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi3"} : <i32>
    %78:2 = fork [2] %77 {handshake.bb = 2 : ui32, handshake.name = "fork13"} : <i1>
    %79 = cmpi ne, %72#0, %trueResult_22 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi4"} : <i32>
    %80:2 = fork [2] %79 {handshake.bb = 2 : ui32, handshake.name = "fork14"} : <i1>
    %trueResult_28, %falseResult_29 = cond_br %81, %trueResult_18 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br12"} : <i1>, <>
    %81 = buffer %74#1, bufferType = FIFO_BREAK_NONE, numSlots = 4 {handshake.bb = 2 : ui32, handshake.name = "buffer45"} : <i1>
    sink %trueResult_28 {handshake.name = "sink12"} : <>
    %trueResult_30, %falseResult_31 = cond_br %82, %trueResult_12 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br13"} : <i1>, <>
    %82 = buffer %76#1, bufferType = FIFO_BREAK_NONE, numSlots = 3 {handshake.bb = 2 : ui32, handshake.name = "buffer46"} : <i1>
    sink %trueResult_30 {handshake.name = "sink13"} : <>
    %trueResult_32, %falseResult_33 = cond_br %83, %trueResult_24 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br14"} : <i1>, <>
    %83 = buffer %78#1, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 2 : ui32, handshake.name = "buffer47"} : <i1>
    sink %trueResult_32 {handshake.name = "sink14"} : <>
    %trueResult_34, %falseResult_35 = cond_br %84, %trueResult_8 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br15"} : <i1>, <>
    %84 = buffer %80#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer48"} : <i1>
    sink %trueResult_34 {handshake.name = "sink15"} : <>
    %85 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %86 = mux %74#0 [%falseResult_29, %85] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux11"} : <i1>, [<>, <>] to <>
    %87 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source2"} : <>
    %88 = mux %76#0 [%falseResult_31, %87] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux12"} : <i1>, [<>, <>] to <>
    %89 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source3"} : <>
    %90 = mux %78#0 [%falseResult_33, %89] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux13"} : <i1>, [<>, <>] to <>
    %91 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source4"} : <>
    %92 = mux %80#0 [%falseResult_35, %91] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux14"} : <i1>, [<>, <>] to <>
    %93 = buffer %86, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer24"} : <>
    %94 = buffer %88, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer25"} : <>
    %95 = buffer %90, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer26"} : <>
    %96 = buffer %92, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer27"} : <>
    %97 = join %93, %94, %95, %96 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "join0"} : <>
    %98 = gate %68#2, %97 {handshake.bb = 2 : ui32, handshake.name = "gate1"} : <i32>, !handshake.control<> to <i32>
    %99 = trunci %98 {handshake.bb = 2 : ui32, handshake.name = "trunci3"} : <i32> to <i10>
    %addressResult_36, %dataResult_37 = load[%99] %outputs#0 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["store1", 0, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load5"} : <i10>, <f32>, <i10>, <f32>
    %100 = addf %dataResult_37, %dataResult_27 {fastmath = #arith.fastmath<none>, fpu_impl = #handshake<fpu_impl flopoco>, handshake.bb = 2 : ui32, handshake.name = "addf0", internal_delay = "2_922000"} : <f32>
    %101 = buffer %68#3, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer0"} : <i32>
    %102:2 = fork [2] %101 {handshake.bb = 2 : ui32, handshake.name = "fork15"} : <i32>
    %103 = init %102#0 {handshake.bb = 2 : ui32, handshake.name = "init9"} : <i32>
    %104:2 = fork [2] %103 {handshake.bb = 2 : ui32, handshake.name = "fork16"} : <i32>
    %105 = init %106 {handshake.bb = 2 : ui32, handshake.name = "init10"} : <i32>
    %106 = buffer %104#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer56"} : <i32>
    %107:2 = fork [2] %105 {handshake.bb = 2 : ui32, handshake.name = "fork17"} : <i32>
    %108 = init %109 {handshake.bb = 2 : ui32, handshake.name = "init11"} : <i32>
    %109 = buffer %107#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer57"} : <i32>
    %110 = buffer %doneResult, bufferType = FIFO_BREAK_NONE, numSlots = 1 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "buffer1"} : <>
    %111:2 = fork [2] %110 {handshake.bb = 2 : ui32, handshake.name = "fork18"} : <>
    %112 = init %111#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init12"} : <>
    %113:2 = fork [2] %112 {handshake.bb = 2 : ui32, handshake.name = "fork19"} : <>
    %114 = init %113#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init13"} : <>
    %115:2 = fork [2] %114 {handshake.bb = 2 : ui32, handshake.name = "fork20"} : <>
    %116 = init %115#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init14"} : <>
    %117:2 = fork [2] %116 {handshake.bb = 2 : ui32, handshake.name = "fork21"} : <>
    %118 = init %117#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init15"} : <>
    %addressResult_38, %dataResult_39, %doneResult = store[%69] %100 %outputs#1 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["load5", 0, false], ["store1", 0, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store1"} : <i10>, <f32>, <>, <i10>, <f32>, <>
    %119 = addi %58#2, %67 {handshake.bb = 2 : ui32, handshake.name = "addi0"} : <i32>
    %120:3 = fork [3] %falseResult_7 {handshake.bb = 3 : ui32, handshake.name = "fork22"} : <>
    end {handshake.bb = 3 : ui32, handshake.name = "end0"} %memEnd_3, %memEnd_1, %memEnd, %0#2 : <>, <>, <>, <>
  }
}

