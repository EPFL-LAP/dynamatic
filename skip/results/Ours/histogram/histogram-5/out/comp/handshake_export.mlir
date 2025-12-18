module {
  handshake.func @histogram(%arg0: memref<1000xi32>, %arg1: memref<1000xf32>, %arg2: memref<1000xf32>, %arg3: !handshake.channel<i32>, %arg4: !handshake.control<>, %arg5: !handshake.control<>, %arg6: !handshake.control<>, %arg7: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["feature", "weight", "hist", "n", "feature_start", "weight_start", "hist_start", "start"], cfg.edges = "[0,1][2,1][1,2,3,cmpi0]", resNames = ["feature_end", "weight_end", "hist_end", "end"]} {
    %0:10 = fork [10] %arg7 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork0"} : <>
    %outputs:2, %memEnd = mem_controller[%arg2 : memref<1000xf32>] %arg6 (%71, %addressResult_42, %addressResult_44, %dataResult_45) %139#2 {connectedBlocks = [2 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "5": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller2"} :    (!handshake.channel<i32>, !handshake.channel<i10>, !handshake.channel<i10>, !handshake.channel<f32>) -> (!handshake.channel<f32>, !handshake.control<>)
    %outputs_0, %memEnd_1 = mem_controller[%arg1 : memref<1000xf32>] %arg5 (%addressResult_30) %139#1 {connectedBlocks = [2 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller3"} :    (!handshake.channel<i10>) -> !handshake.channel<f32>
    %outputs_2, %memEnd_3 = mem_controller[%arg0 : memref<1000xi32>] %arg4 (%addressResult) %139#0 {connectedBlocks = [2 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller4"} :    (!handshake.channel<i10>) -> !handshake.channel<i32>
    %1 = constant %0#1 {handshake.bb = 0 : ui32, handshake.name = "constant8", value = 1000 : i11} : <>, <i11>
    %2:5 = fork [5] %1 {handshake.bb = 0 : ui32, handshake.name = "fork1"} : <i11>
    %3 = extsi %2#0 {handshake.bb = 1 : ui32, handshake.name = "extsi0"} : <i11> to <i32>
    %4 = extsi %2#1 {handshake.bb = 1 : ui32, handshake.name = "extsi1"} : <i11> to <i32>
    %5 = extsi %2#2 {handshake.bb = 1 : ui32, handshake.name = "extsi2"} : <i11> to <i32>
    %6 = extsi %2#3 {handshake.bb = 1 : ui32, handshake.name = "extsi3"} : <i11> to <i32>
    %7 = extsi %2#4 {handshake.bb = 1 : ui32, handshake.name = "extsi4"} : <i11> to <i32>
    %8 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant9", value = false} : <>, <i1>
    %9 = extsi %8 {handshake.bb = 0 : ui32, handshake.name = "extsi8"} : <i1> to <i32>
    %10 = mux %29#0 [%3, %120#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux2"} : <i1>, [<i32>, <i32>] to <i32>
    %11 = mux %29#1 [%4, %12] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux3"} : <i1>, [<i32>, <i32>] to <i32>
    %12 = buffer %117#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer9"} : <i32>
    %13 = mux %29#2 [%5, %123#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux4"} : <i1>, [<i32>, <i32>] to <i32>
    %14 = mux %29#3 [%0#8, %136] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux5"} : <i1>, [<>, <>] to <>
    %15 = mux %29#4 [%6, %115#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux6"} : <i1>, [<i32>, <i32>] to <i32>
    %16 = mux %29#5 [%7, %124] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux7"} : <i1>, [<i32>, <i32>] to <i32>
    %17 = mux %18 [%0#7, %129#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux8"} : <i1>, [<>, <>] to <>
    %18 = buffer %29#6, bufferType = FIFO_BREAK_NONE, numSlots = 4 {handshake.bb = 1 : ui32, handshake.name = "buffer17"} : <i1>
    %19 = mux %20 [%0#6, %131#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux9"} : <i1>, [<>, <>] to <>
    %20 = buffer %29#7, bufferType = FIFO_BREAK_NONE, numSlots = 3 {handshake.bb = 1 : ui32, handshake.name = "buffer18"} : <i1>
    %21 = mux %22 [%0#5, %133#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux10"} : <i1>, [<>, <>] to <>
    %22 = buffer %29#8, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 1 : ui32, handshake.name = "buffer19"} : <i1>
    %23 = mux %24 [%0#4, %135#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux11"} : <i1>, [<>, <>] to <>
    %24 = buffer %29#9, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer20"} : <i1>
    %25 = mux %26 [%0#3, %127#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux12"} : <i1>, [<>, <>] to <>
    %26 = buffer %29#10, bufferType = FIFO_BREAK_NONE, numSlots = 5 {handshake.bb = 1 : ui32, handshake.name = "buffer21"} : <i1>
    %27 = init %28 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "init0"} : <i1>
    %28 = buffer %41#14, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 1 : ui32, handshake.name = "buffer22"} : <i1>
    %29:11 = fork [11] %27 {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <i1>
    %30 = mux %38#0 [%9, %138] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i32>, <i32>] to <i32>
    %31 = buffer %30, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer15"} : <i32>
    %32 = buffer %31, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer16"} : <i32>
    %33:2 = fork [2] %32 {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <i32>
    %34 = mux %38#1 [%arg3, %trueResult] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i1>, [<i32>, <i32>] to <i32>
    %35 = buffer %34, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer23"} : <i32>
    %36 = buffer %35, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer24"} : <i32>
    %37:2 = fork [2] %36 {handshake.bb = 1 : ui32, handshake.name = "fork4"} : <i32>
    %result, %index = control_merge [%0#9, %69#1]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %38:2 = fork [2] %index {handshake.bb = 1 : ui32, handshake.name = "fork5"} : <i1>
    %39 = cmpi slt, %33#1, %37#1 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i32>
    %40 = buffer %39, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer25"} : <i1>
    %41:15 = fork [15] %40 {handshake.bb = 1 : ui32, handshake.name = "fork6"} : <i1>
    %trueResult, %falseResult = cond_br %41#13, %37#0 {handshake.bb = 1 : ui32, handshake.name = "cond_br1"} : <i1>, <i32>
    sink %falseResult {handshake.name = "sink0"} : <i32>
    %trueResult_4, %falseResult_5 = cond_br %41#12, %33#0 {handshake.bb = 1 : ui32, handshake.name = "cond_br2"} : <i1>, <i32>
    sink %falseResult_5 {handshake.name = "sink1"} : <i32>
    %trueResult_6, %falseResult_7 = cond_br %41#11, %result {handshake.bb = 1 : ui32, handshake.name = "cond_br3"} : <i1>, <>
    %42 = buffer %21, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer12"} : <>
    %trueResult_8, %falseResult_9 = cond_br %43, %42 {handshake.bb = 2 : ui32, handshake.name = "cond_br19"} : <i1>, <>
    %43 = buffer %41#10, bufferType = FIFO_BREAK_NONE, numSlots = 3 {handshake.bb = 2 : ui32, handshake.name = "buffer32"} : <i1>
    sink %falseResult_9 {handshake.name = "sink2"} : <>
    %44 = buffer %25, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer14"} : <>
    %trueResult_10, %falseResult_11 = cond_br %45, %44 {handshake.bb = 2 : ui32, handshake.name = "cond_br20"} : <i1>, <>
    %45 = buffer %41#9, bufferType = FIFO_BREAK_NONE, numSlots = 6 {handshake.bb = 2 : ui32, handshake.name = "buffer33"} : <i1>
    sink %falseResult_11 {handshake.name = "sink3"} : <>
    %46 = buffer %15, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer7"} : <i32>
    %trueResult_12, %falseResult_13 = cond_br %47, %46 {handshake.bb = 2 : ui32, handshake.name = "cond_br21"} : <i1>, <i32>
    %47 = buffer %41#8, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer34"} : <i1>
    sink %falseResult_13 {handshake.name = "sink4"} : <i32>
    %48 = buffer %14, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer5"} : <>
    %49 = buffer %48, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer6"} : <>
    %trueResult_14, %falseResult_15 = cond_br %50, %49 {handshake.bb = 2 : ui32, handshake.name = "cond_br22"} : <i1>, <>
    %50 = buffer %41#7, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer35"} : <i1>
    sink %falseResult_15 {handshake.name = "sink5"} : <>
    %51 = buffer %13, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer4"} : <i32>
    %trueResult_16, %falseResult_17 = cond_br %52, %51 {handshake.bb = 2 : ui32, handshake.name = "cond_br23"} : <i1>, <i32>
    %52 = buffer %41#6, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer36"} : <i1>
    sink %falseResult_17 {handshake.name = "sink6"} : <i32>
    %53 = buffer %16, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer8"} : <i32>
    %trueResult_18, %falseResult_19 = cond_br %54, %53 {handshake.bb = 2 : ui32, handshake.name = "cond_br24"} : <i1>, <i32>
    %54 = buffer %41#5, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer37"} : <i1>
    sink %falseResult_19 {handshake.name = "sink7"} : <i32>
    %55 = buffer %11, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer3"} : <i32>
    %trueResult_20, %falseResult_21 = cond_br %56, %55 {handshake.bb = 2 : ui32, handshake.name = "cond_br25"} : <i1>, <i32>
    %56 = buffer %41#4, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer38"} : <i1>
    sink %falseResult_21 {handshake.name = "sink8"} : <i32>
    %57 = buffer %10, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer2"} : <i32>
    %trueResult_22, %falseResult_23 = cond_br %58, %57 {handshake.bb = 2 : ui32, handshake.name = "cond_br26"} : <i1>, <i32>
    %58 = buffer %41#3, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer39"} : <i1>
    sink %falseResult_23 {handshake.name = "sink9"} : <i32>
    %59 = buffer %19, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer11"} : <>
    %trueResult_24, %falseResult_25 = cond_br %60, %59 {handshake.bb = 2 : ui32, handshake.name = "cond_br27"} : <i1>, <>
    %60 = buffer %41#2, bufferType = FIFO_BREAK_NONE, numSlots = 4 {handshake.bb = 2 : ui32, handshake.name = "buffer40"} : <i1>
    sink %falseResult_25 {handshake.name = "sink10"} : <>
    %61 = buffer %17, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer10"} : <>
    %trueResult_26, %falseResult_27 = cond_br %62, %61 {handshake.bb = 2 : ui32, handshake.name = "cond_br28"} : <i1>, <>
    %62 = buffer %41#1, bufferType = FIFO_BREAK_NONE, numSlots = 5 {handshake.bb = 2 : ui32, handshake.name = "buffer41"} : <i1>
    sink %falseResult_27 {handshake.name = "sink11"} : <>
    %63 = buffer %23, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer13"} : <>
    %trueResult_28, %falseResult_29 = cond_br %64, %63 {handshake.bb = 2 : ui32, handshake.name = "cond_br29"} : <i1>, <>
    %64 = buffer %41#0, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 2 : ui32, handshake.name = "buffer42"} : <i1>
    sink %falseResult_29 {handshake.name = "sink12"} : <>
    %65:3 = fork [3] %trueResult_4 {handshake.bb = 2 : ui32, handshake.name = "fork7"} : <i32>
    %66 = trunci %65#0 {handshake.bb = 2 : ui32, handshake.name = "trunci0"} : <i32> to <i10>
    %67 = trunci %65#1 {handshake.bb = 2 : ui32, handshake.name = "trunci1"} : <i32> to <i10>
    %68 = buffer %trueResult_6, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer26"} : <>
    %69:2 = fork [2] %68 {handshake.bb = 2 : ui32, handshake.name = "fork8"} : <>
    %70 = constant %69#0 {handshake.bb = 2 : ui32, handshake.name = "constant10", value = 1 : i2} : <>, <i2>
    %71 = extsi %70 {handshake.bb = 2 : ui32, handshake.name = "extsi6"} : <i2> to <i32>
    %72 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %73 = constant %72 {handshake.bb = 2 : ui32, handshake.name = "constant11", value = 1 : i2} : <>, <i2>
    %74 = extsi %73 {handshake.bb = 2 : ui32, handshake.name = "extsi7"} : <i2> to <i32>
    %addressResult, %dataResult = load[%67] %outputs_2 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i10>, <i32>, <i10>, <i32>
    %75:4 = fork [4] %dataResult {handshake.bb = 2 : ui32, handshake.name = "fork9"} : <i32>
    %76 = trunci %77 {handshake.bb = 2 : ui32, handshake.name = "trunci2"} : <i32> to <i10>
    %77 = buffer %75#0, bufferType = FIFO_BREAK_NONE, numSlots = 6 {handshake.bb = 2 : ui32, handshake.name = "buffer45"} : <i32>
    %addressResult_30, %dataResult_31 = load[%66] %outputs_0 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i10>, <f32>, <i10>, <f32>
    %78 = gate %75#1, %trueResult_14 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "gate0"} : <i32>, !handshake.control<> to <i32>
    %79:5 = fork [5] %78 {handshake.bb = 2 : ui32, handshake.name = "fork10"} : <i32>
    %80 = cmpi ne, %79#4, %trueResult_12 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi1"} : <i32>
    %81:2 = fork [2] %80 {handshake.bb = 2 : ui32, handshake.name = "fork11"} : <i1>
    %82 = cmpi ne, %79#3, %trueResult_20 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi2"} : <i32>
    %83:2 = fork [2] %82 {handshake.bb = 2 : ui32, handshake.name = "fork12"} : <i1>
    %84 = cmpi ne, %79#2, %trueResult_22 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi3"} : <i32>
    %85:2 = fork [2] %84 {handshake.bb = 2 : ui32, handshake.name = "fork13"} : <i1>
    %86 = cmpi ne, %79#1, %trueResult_16 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi4"} : <i32>
    %87:2 = fork [2] %86 {handshake.bb = 2 : ui32, handshake.name = "fork14"} : <i1>
    %88 = cmpi ne, %79#0, %trueResult_18 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi5"} : <i32>
    %89:2 = fork [2] %88 {handshake.bb = 2 : ui32, handshake.name = "fork15"} : <i1>
    %trueResult_32, %falseResult_33 = cond_br %90, %trueResult_10 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br14"} : <i1>, <>
    %90 = buffer %81#1, bufferType = FIFO_BREAK_NONE, numSlots = 5 {handshake.bb = 2 : ui32, handshake.name = "buffer52"} : <i1>
    sink %trueResult_32 {handshake.name = "sink14"} : <>
    %trueResult_34, %falseResult_35 = cond_br %91, %trueResult_26 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br15"} : <i1>, <>
    %91 = buffer %83#1, bufferType = FIFO_BREAK_NONE, numSlots = 4 {handshake.bb = 2 : ui32, handshake.name = "buffer53"} : <i1>
    sink %trueResult_34 {handshake.name = "sink15"} : <>
    %trueResult_36, %falseResult_37 = cond_br %92, %trueResult_24 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br16"} : <i1>, <>
    %92 = buffer %85#1, bufferType = FIFO_BREAK_NONE, numSlots = 3 {handshake.bb = 2 : ui32, handshake.name = "buffer54"} : <i1>
    sink %trueResult_36 {handshake.name = "sink16"} : <>
    %trueResult_38, %falseResult_39 = cond_br %93, %trueResult_8 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br17"} : <i1>, <>
    %93 = buffer %87#1, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 2 : ui32, handshake.name = "buffer55"} : <i1>
    sink %trueResult_38 {handshake.name = "sink17"} : <>
    %trueResult_40, %falseResult_41 = cond_br %94, %trueResult_28 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br18"} : <i1>, <>
    %94 = buffer %89#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer56"} : <i1>
    sink %trueResult_40 {handshake.name = "sink18"} : <>
    %95 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %96 = mux %81#0 [%falseResult_33, %95] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux13"} : <i1>, [<>, <>] to <>
    %97 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source2"} : <>
    %98 = mux %83#0 [%falseResult_35, %97] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux14"} : <i1>, [<>, <>] to <>
    %99 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source3"} : <>
    %100 = mux %85#0 [%falseResult_37, %99] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux15"} : <i1>, [<>, <>] to <>
    %101 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source4"} : <>
    %102 = mux %87#0 [%falseResult_39, %101] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux16"} : <i1>, [<>, <>] to <>
    %103 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source5"} : <>
    %104 = mux %89#0 [%falseResult_41, %103] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux17"} : <i1>, [<>, <>] to <>
    %105 = buffer %96, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer27"} : <>
    %106 = buffer %98, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer28"} : <>
    %107 = buffer %100, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer29"} : <>
    %108 = buffer %102, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer30"} : <>
    %109 = buffer %104, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer31"} : <>
    %110 = join %105, %106, %107, %108, %109 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "join0"} : <>
    %111 = gate %75#2, %110 {handshake.bb = 2 : ui32, handshake.name = "gate1"} : <i32>, !handshake.control<> to <i32>
    %112 = trunci %111 {handshake.bb = 2 : ui32, handshake.name = "trunci3"} : <i32> to <i10>
    %addressResult_42, %dataResult_43 = load[%112] %outputs#0 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["store1", 0, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load5"} : <i10>, <f32>, <i10>, <f32>
    %113 = addf %dataResult_43, %dataResult_31 {fastmath = #arith.fastmath<none>, fpu_impl = #handshake<fpu_impl flopoco>, handshake.bb = 2 : ui32, handshake.name = "addf0", internal_delay = "2_922000"} : <f32>
    %114 = buffer %75#3, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer0"} : <i32>
    %115:2 = fork [2] %114 {handshake.bb = 2 : ui32, handshake.name = "fork16"} : <i32>
    %116 = init %115#0 {handshake.bb = 2 : ui32, handshake.name = "init11"} : <i32>
    %117:2 = fork [2] %116 {handshake.bb = 2 : ui32, handshake.name = "fork17"} : <i32>
    %118 = init %119 {handshake.bb = 2 : ui32, handshake.name = "init12"} : <i32>
    %119 = buffer %117#0, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 2 : ui32, handshake.name = "buffer65"} : <i32>
    %120:2 = fork [2] %118 {handshake.bb = 2 : ui32, handshake.name = "fork18"} : <i32>
    %121 = init %122 {handshake.bb = 2 : ui32, handshake.name = "init13"} : <i32>
    %122 = buffer %120#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer66"} : <i32>
    %123:2 = fork [2] %121 {handshake.bb = 2 : ui32, handshake.name = "fork19"} : <i32>
    %124 = init %125 {handshake.bb = 2 : ui32, handshake.name = "init14"} : <i32>
    %125 = buffer %123#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer67"} : <i32>
    %126 = buffer %doneResult, bufferType = FIFO_BREAK_NONE, numSlots = 1 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "buffer1"} : <>
    %127:2 = fork [2] %126 {handshake.bb = 2 : ui32, handshake.name = "fork20"} : <>
    %128 = init %127#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init15"} : <>
    %129:2 = fork [2] %128 {handshake.bb = 2 : ui32, handshake.name = "fork21"} : <>
    %130 = init %129#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init16"} : <>
    %131:2 = fork [2] %130 {handshake.bb = 2 : ui32, handshake.name = "fork22"} : <>
    %132 = init %131#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init17"} : <>
    %133:2 = fork [2] %132 {handshake.bb = 2 : ui32, handshake.name = "fork23"} : <>
    %134 = init %133#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init18"} : <>
    %135:2 = fork [2] %134 {handshake.bb = 2 : ui32, handshake.name = "fork24"} : <>
    %136 = init %135#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init19"} : <>
    %addressResult_44, %dataResult_45, %doneResult = store[%76] %113 %outputs#1 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["load5", 0, false], ["store1", 0, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store1"} : <i10>, <f32>, <>, <i10>, <f32>, <>
    %137 = addi %65#2, %74 {handshake.bb = 2 : ui32, handshake.name = "addi0"} : <i32>
    %138 = buffer %137, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer43"} : <i32>
    %139:3 = fork [3] %falseResult_7 {handshake.bb = 3 : ui32, handshake.name = "fork25"} : <>
    end {handshake.bb = 3 : ui32, handshake.name = "end0"} %memEnd_3, %memEnd_1, %memEnd, %0#2 : <>, <>, <>, <>
  }
}

