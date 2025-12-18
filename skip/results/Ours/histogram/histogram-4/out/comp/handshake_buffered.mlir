module {
  handshake.func @histogram(%arg0: memref<1000xi32>, %arg1: memref<1000xf32>, %arg2: memref<1000xf32>, %arg3: !handshake.channel<i32>, %arg4: !handshake.control<>, %arg5: !handshake.control<>, %arg6: !handshake.control<>, %arg7: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["feature", "weight", "hist", "n", "feature_start", "weight_start", "hist_start", "start"], cfg.edges = "[0,1][2,1][1,2,3,cmpi0]", resNames = ["feature_end", "weight_end", "hist_end", "end"]} {
    %0:9 = fork [9] %arg7 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork0"} : <>
    %outputs:2, %memEnd = mem_controller[%arg2 : memref<1000xf32>] %arg6 (%69, %addressResult_38, %addressResult_40, %dataResult_41) %128#2 {connectedBlocks = [2 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "5": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller2"} :    (!handshake.channel<i32>, !handshake.channel<i10>, !handshake.channel<i10>, !handshake.channel<f32>) -> (!handshake.channel<f32>, !handshake.control<>)
    %outputs_0, %memEnd_1 = mem_controller[%arg1 : memref<1000xf32>] %arg5 (%addressResult_28) %128#1 {connectedBlocks = [2 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller3"} :    (!handshake.channel<i10>) -> !handshake.channel<f32>
    %outputs_2, %memEnd_3 = mem_controller[%arg0 : memref<1000xi32>] %arg4 (%addressResult) %128#0 {connectedBlocks = [2 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller4"} :    (!handshake.channel<i10>) -> !handshake.channel<i32>
    %1 = constant %0#1 {handshake.bb = 0 : ui32, handshake.name = "constant7", value = 1000 : i11} : <>, <i11>
    %2:4 = fork [4] %1 {handshake.bb = 0 : ui32, handshake.name = "fork1"} : <i11>
    %3 = extsi %2#0 {handshake.bb = 1 : ui32, handshake.name = "extsi0"} : <i11> to <i32>
    %4 = extsi %2#1 {handshake.bb = 1 : ui32, handshake.name = "extsi1"} : <i11> to <i32>
    %5 = extsi %2#2 {handshake.bb = 1 : ui32, handshake.name = "extsi2"} : <i11> to <i32>
    %6 = extsi %2#3 {handshake.bb = 1 : ui32, handshake.name = "extsi3"} : <i11> to <i32>
    %7 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant8", value = false} : <>, <i1>
    %8 = br %7 {handshake.bb = 0 : ui32, handshake.name = "br2"} : <i1>
    %9 = extsi %8 {handshake.bb = 0 : ui32, handshake.name = "extsi7"} : <i1> to <i32>
    %10 = br %arg3 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "br3"} : <i32>
    %11 = br %0#8 {handshake.bb = 0 : ui32, handshake.name = "br4"} : <>
    %12 = mux %28#0 [%3, %113] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux2"} : <i1>, [<i32>, <i32>] to <i32>
    %13 = mux %28#1 [%4, %107#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux3"} : <i1>, [<i32>, <i32>] to <i32>
    %14 = mux %28#2 [%5, %112#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux4"} : <i1>, [<i32>, <i32>] to <i32>
    %15 = mux %28#3 [%6, %16] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux5"} : <i1>, [<i32>, <i32>] to <i32>
    %16 = buffer %109#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer11"} : <i32>
    %17 = mux %28#4 [%0#7, %123] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux6"} : <i1>, [<>, <>] to <>
    %18 = mux %19 [%0#6, %120#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux7"} : <i1>, [<>, <>] to <>
    %19 = buffer %28#5, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 1 : ui32, handshake.name = "buffer14"} : <i1>
    %20 = mux %21 [%0#5, %116#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux8"} : <i1>, [<>, <>] to <>
    %21 = buffer %28#6, bufferType = FIFO_BREAK_NONE, numSlots = 4 {handshake.bb = 1 : ui32, handshake.name = "buffer15"} : <i1>
    %22 = mux %23 [%0#4, %122#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux9"} : <i1>, [<>, <>] to <>
    %23 = buffer %28#7, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer16"} : <i1>
    %24 = mux %25 [%0#3, %118#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux10"} : <i1>, [<>, <>] to <>
    %25 = buffer %28#8, bufferType = FIFO_BREAK_NONE, numSlots = 3 {handshake.bb = 1 : ui32, handshake.name = "buffer17"} : <i1>
    %26 = init %27 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "init0"} : <i1>
    %27 = buffer %41#12, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 1 : ui32, handshake.name = "buffer18"} : <i1>
    %28:9 = fork [9] %26 {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <i1>
    %29 = buffer %125, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer37"} : <i32>
    %30 = mux %38#0 [%9, %29] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i32>, <i32>] to <i32>
    %31 = buffer %30, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer13"} : <i32>
    %32 = buffer %31, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer19"} : <i32>
    %33:2 = fork [2] %32 {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <i32>
    %34 = mux %38#1 [%10, %126] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i1>, [<i32>, <i32>] to <i32>
    %35 = buffer %34, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer20"} : <i32>
    %36 = buffer %35, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer21"} : <i32>
    %37:2 = fork [2] %36 {handshake.bb = 1 : ui32, handshake.name = "fork4"} : <i32>
    %result, %index = control_merge [%11, %127]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %38:2 = fork [2] %index {handshake.bb = 1 : ui32, handshake.name = "fork5"} : <i1>
    %39 = cmpi slt, %33#1, %37#1 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i32>
    %40 = buffer %39, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer22"} : <i1>
    %41:13 = fork [13] %40 {handshake.bb = 1 : ui32, handshake.name = "fork6"} : <i1>
    %trueResult, %falseResult = cond_br %41#11, %37#0 {handshake.bb = 1 : ui32, handshake.name = "cond_br1"} : <i1>, <i32>
    sink %falseResult {handshake.name = "sink0"} : <i32>
    %trueResult_4, %falseResult_5 = cond_br %41#10, %33#0 {handshake.bb = 1 : ui32, handshake.name = "cond_br2"} : <i1>, <i32>
    sink %falseResult_5 {handshake.name = "sink1"} : <i32>
    %trueResult_6, %falseResult_7 = cond_br %41#9, %result {handshake.bb = 1 : ui32, handshake.name = "cond_br3"} : <i1>, <>
    %42 = buffer %22, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer10"} : <>
    %trueResult_8, %falseResult_9 = cond_br %43, %42 {handshake.bb = 2 : ui32, handshake.name = "cond_br16"} : <i1>, <>
    %43 = buffer %41#8, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 2 : ui32, handshake.name = "buffer28"} : <i1>
    sink %falseResult_9 {handshake.name = "sink2"} : <>
    %44 = buffer %17, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer6"} : <>
    %45 = buffer %44, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer7"} : <>
    %trueResult_10, %falseResult_11 = cond_br %46, %45 {handshake.bb = 2 : ui32, handshake.name = "cond_br17"} : <i1>, <>
    %46 = buffer %41#7, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer29"} : <i1>
    sink %falseResult_11 {handshake.name = "sink3"} : <>
    %47 = buffer %24, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer12"} : <>
    %trueResult_12, %falseResult_13 = cond_br %48, %47 {handshake.bb = 2 : ui32, handshake.name = "cond_br18"} : <i1>, <>
    %48 = buffer %41#6, bufferType = FIFO_BREAK_NONE, numSlots = 4 {handshake.bb = 2 : ui32, handshake.name = "buffer30"} : <i1>
    sink %falseResult_13 {handshake.name = "sink4"} : <>
    %49 = buffer %13, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer3"} : <i32>
    %trueResult_14, %falseResult_15 = cond_br %50, %49 {handshake.bb = 2 : ui32, handshake.name = "cond_br19"} : <i1>, <i32>
    %50 = buffer %41#5, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer31"} : <i1>
    sink %falseResult_15 {handshake.name = "sink5"} : <i32>
    %51 = buffer %14, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer4"} : <i32>
    %trueResult_16, %falseResult_17 = cond_br %52, %51 {handshake.bb = 2 : ui32, handshake.name = "cond_br20"} : <i1>, <i32>
    %52 = buffer %41#4, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer32"} : <i1>
    sink %falseResult_17 {handshake.name = "sink6"} : <i32>
    %53 = buffer %20, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer9"} : <>
    %trueResult_18, %falseResult_19 = cond_br %54, %53 {handshake.bb = 2 : ui32, handshake.name = "cond_br21"} : <i1>, <>
    %54 = buffer %41#3, bufferType = FIFO_BREAK_NONE, numSlots = 5 {handshake.bb = 2 : ui32, handshake.name = "buffer33"} : <i1>
    sink %falseResult_19 {handshake.name = "sink7"} : <>
    %55 = buffer %15, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer5"} : <i32>
    %trueResult_20, %falseResult_21 = cond_br %56, %55 {handshake.bb = 2 : ui32, handshake.name = "cond_br22"} : <i1>, <i32>
    %56 = buffer %41#2, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer34"} : <i1>
    sink %falseResult_21 {handshake.name = "sink8"} : <i32>
    %57 = buffer %12, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer2"} : <i32>
    %trueResult_22, %falseResult_23 = cond_br %58, %57 {handshake.bb = 2 : ui32, handshake.name = "cond_br23"} : <i1>, <i32>
    %58 = buffer %41#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer35"} : <i1>
    sink %falseResult_23 {handshake.name = "sink9"} : <i32>
    %59 = buffer %18, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer8"} : <>
    %trueResult_24, %falseResult_25 = cond_br %60, %59 {handshake.bb = 2 : ui32, handshake.name = "cond_br24"} : <i1>, <>
    %60 = buffer %41#0, bufferType = FIFO_BREAK_NONE, numSlots = 3 {handshake.bb = 2 : ui32, handshake.name = "buffer36"} : <i1>
    sink %falseResult_25 {handshake.name = "sink10"} : <>
    %61 = merge %trueResult {handshake.bb = 2 : ui32, handshake.name = "merge0"} : <i32>
    %62 = merge %trueResult_4 {handshake.bb = 2 : ui32, handshake.name = "merge1"} : <i32>
    %63:3 = fork [3] %62 {handshake.bb = 2 : ui32, handshake.name = "fork7"} : <i32>
    %64 = trunci %63#0 {handshake.bb = 2 : ui32, handshake.name = "trunci0"} : <i32> to <i10>
    %65 = trunci %63#1 {handshake.bb = 2 : ui32, handshake.name = "trunci1"} : <i32> to <i10>
    %66 = buffer %trueResult_6, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer23"} : <>
    %result_26, %index_27 = control_merge [%66]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>] to <>, <i1>
    sink %index_27 {handshake.name = "sink11"} : <i1>
    %67:2 = fork [2] %result_26 {handshake.bb = 2 : ui32, handshake.name = "fork8"} : <>
    %68 = constant %67#0 {handshake.bb = 2 : ui32, handshake.name = "constant9", value = 1 : i2} : <>, <i2>
    %69 = extsi %68 {handshake.bb = 2 : ui32, handshake.name = "extsi5"} : <i2> to <i32>
    %70 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %71 = constant %70 {handshake.bb = 2 : ui32, handshake.name = "constant10", value = 1 : i2} : <>, <i2>
    %72 = extsi %71 {handshake.bb = 2 : ui32, handshake.name = "extsi6"} : <i2> to <i32>
    %addressResult, %dataResult = load[%65] %outputs_2 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i10>, <i32>, <i10>, <i32>
    %73:4 = fork [4] %dataResult {handshake.bb = 2 : ui32, handshake.name = "fork9"} : <i32>
    %74 = trunci %75 {handshake.bb = 2 : ui32, handshake.name = "trunci2"} : <i32> to <i10>
    %75 = buffer %73#0, bufferType = FIFO_BREAK_NONE, numSlots = 4 {handshake.bb = 2 : ui32, handshake.name = "buffer39"} : <i32>
    %addressResult_28, %dataResult_29 = load[%64] %outputs_0 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i10>, <f32>, <i10>, <f32>
    %76 = gate %73#1, %trueResult_10 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "gate0"} : <i32>, !handshake.control<> to <i32>
    %77:4 = fork [4] %76 {handshake.bb = 2 : ui32, handshake.name = "fork10"} : <i32>
    %78 = cmpi ne, %77#3, %trueResult_14 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi1"} : <i32>
    %79:2 = fork [2] %78 {handshake.bb = 2 : ui32, handshake.name = "fork11"} : <i1>
    %80 = cmpi ne, %77#2, %trueResult_20 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi2"} : <i32>
    %81:2 = fork [2] %80 {handshake.bb = 2 : ui32, handshake.name = "fork12"} : <i1>
    %82 = cmpi ne, %77#1, %trueResult_16 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi3"} : <i32>
    %83:2 = fork [2] %82 {handshake.bb = 2 : ui32, handshake.name = "fork13"} : <i1>
    %84 = cmpi ne, %77#0, %trueResult_22 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi4"} : <i32>
    %85:2 = fork [2] %84 {handshake.bb = 2 : ui32, handshake.name = "fork14"} : <i1>
    %trueResult_30, %falseResult_31 = cond_br %86, %trueResult_18 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br12"} : <i1>, <>
    %86 = buffer %79#1, bufferType = FIFO_BREAK_NONE, numSlots = 4 {handshake.bb = 2 : ui32, handshake.name = "buffer45"} : <i1>
    sink %trueResult_30 {handshake.name = "sink12"} : <>
    %trueResult_32, %falseResult_33 = cond_br %87, %trueResult_12 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br13"} : <i1>, <>
    %87 = buffer %81#1, bufferType = FIFO_BREAK_NONE, numSlots = 3 {handshake.bb = 2 : ui32, handshake.name = "buffer46"} : <i1>
    sink %trueResult_32 {handshake.name = "sink13"} : <>
    %trueResult_34, %falseResult_35 = cond_br %88, %trueResult_24 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br14"} : <i1>, <>
    %88 = buffer %83#1, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 2 : ui32, handshake.name = "buffer47"} : <i1>
    sink %trueResult_34 {handshake.name = "sink14"} : <>
    %trueResult_36, %falseResult_37 = cond_br %89, %trueResult_8 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br15"} : <i1>, <>
    %89 = buffer %85#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer48"} : <i1>
    sink %trueResult_36 {handshake.name = "sink15"} : <>
    %90 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %91 = mux %79#0 [%falseResult_31, %90] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux11"} : <i1>, [<>, <>] to <>
    %92 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source2"} : <>
    %93 = mux %81#0 [%falseResult_33, %92] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux12"} : <i1>, [<>, <>] to <>
    %94 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source3"} : <>
    %95 = mux %83#0 [%falseResult_35, %94] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux13"} : <i1>, [<>, <>] to <>
    %96 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source4"} : <>
    %97 = mux %85#0 [%falseResult_37, %96] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux14"} : <i1>, [<>, <>] to <>
    %98 = buffer %91, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer24"} : <>
    %99 = buffer %93, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer25"} : <>
    %100 = buffer %95, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer26"} : <>
    %101 = buffer %97, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer27"} : <>
    %102 = join %98, %99, %100, %101 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "join0"} : <>
    %103 = gate %73#2, %102 {handshake.bb = 2 : ui32, handshake.name = "gate1"} : <i32>, !handshake.control<> to <i32>
    %104 = trunci %103 {handshake.bb = 2 : ui32, handshake.name = "trunci3"} : <i32> to <i10>
    %addressResult_38, %dataResult_39 = load[%104] %outputs#0 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["store1", 0, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load5"} : <i10>, <f32>, <i10>, <f32>
    %105 = addf %dataResult_39, %dataResult_29 {fastmath = #arith.fastmath<none>, fpu_impl = #handshake<fpu_impl flopoco>, handshake.bb = 2 : ui32, handshake.name = "addf0", internal_delay = "2_922000"} : <f32>
    %106 = buffer %73#3, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer0"} : <i32>
    %107:2 = fork [2] %106 {handshake.bb = 2 : ui32, handshake.name = "fork15"} : <i32>
    %108 = init %107#0 {handshake.bb = 2 : ui32, handshake.name = "init9"} : <i32>
    %109:2 = fork [2] %108 {handshake.bb = 2 : ui32, handshake.name = "fork16"} : <i32>
    %110 = init %111 {handshake.bb = 2 : ui32, handshake.name = "init10"} : <i32>
    %111 = buffer %109#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer56"} : <i32>
    %112:2 = fork [2] %110 {handshake.bb = 2 : ui32, handshake.name = "fork17"} : <i32>
    %113 = init %114 {handshake.bb = 2 : ui32, handshake.name = "init11"} : <i32>
    %114 = buffer %112#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer57"} : <i32>
    %115 = buffer %doneResult, bufferType = FIFO_BREAK_NONE, numSlots = 1 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "buffer1"} : <>
    %116:2 = fork [2] %115 {handshake.bb = 2 : ui32, handshake.name = "fork18"} : <>
    %117 = init %116#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init12"} : <>
    %118:2 = fork [2] %117 {handshake.bb = 2 : ui32, handshake.name = "fork19"} : <>
    %119 = init %118#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init13"} : <>
    %120:2 = fork [2] %119 {handshake.bb = 2 : ui32, handshake.name = "fork20"} : <>
    %121 = init %120#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init14"} : <>
    %122:2 = fork [2] %121 {handshake.bb = 2 : ui32, handshake.name = "fork21"} : <>
    %123 = init %122#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init15"} : <>
    %addressResult_40, %dataResult_41, %doneResult = store[%74] %105 %outputs#1 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["load5", 0, false], ["store1", 0, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store1"} : <i10>, <f32>, <>, <i10>, <f32>, <>
    %124 = addi %63#2, %72 {handshake.bb = 2 : ui32, handshake.name = "addi0"} : <i32>
    %125 = br %124 {handshake.bb = 2 : ui32, handshake.name = "br5"} : <i32>
    %126 = br %61 {handshake.bb = 2 : ui32, handshake.name = "br6"} : <i32>
    %127 = br %67#1 {handshake.bb = 2 : ui32, handshake.name = "br7"} : <>
    %result_42, %index_43 = control_merge [%falseResult_7]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>] to <>, <i1>
    sink %index_43 {handshake.name = "sink16"} : <i1>
    %128:3 = fork [3] %result_42 {handshake.bb = 3 : ui32, handshake.name = "fork22"} : <>
    end {handshake.bb = 3 : ui32, handshake.name = "end0"} %memEnd_3, %memEnd_1, %memEnd, %0#2 : <>, <>, <>, <>
  }
}

