module {
  handshake.func @histogram(%arg0: memref<1000xi32>, %arg1: memref<1000xf32>, %arg2: memref<1000xf32>, %arg3: !handshake.channel<i32>, %arg4: !handshake.control<>, %arg5: !handshake.control<>, %arg6: !handshake.control<>, %arg7: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["feature", "weight", "hist", "n", "feature_start", "weight_start", "hist_start", "start"], cfg.edges = "[0,1][2,1][1,2,3,cmpi0]", resNames = ["feature_end", "weight_end", "hist_end", "end"]} {
    %0:11 = fork [11] %arg7 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork0"} : <>
    %outputs:2, %memEnd = mem_controller[%arg2 : memref<1000xf32>] %arg6 (%85, %addressResult_50, %addressResult_52, %dataResult_53) %167#2 {connectedBlocks = [2 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "5": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller2"} :    (!handshake.channel<i32>, !handshake.channel<i10>, !handshake.channel<i10>, !handshake.channel<f32>) -> (!handshake.channel<f32>, !handshake.control<>)
    %outputs_0, %memEnd_1 = mem_controller[%arg1 : memref<1000xf32>] %arg5 (%addressResult_36) %167#1 {connectedBlocks = [2 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller3"} :    (!handshake.channel<i10>) -> !handshake.channel<f32>
    %outputs_2, %memEnd_3 = mem_controller[%arg0 : memref<1000xi32>] %arg4 (%addressResult) %167#0 {connectedBlocks = [2 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller4"} :    (!handshake.channel<i10>) -> !handshake.channel<i32>
    %1 = constant %0#1 {handshake.bb = 0 : ui32, handshake.name = "constant9", value = 1000 : i11} : <>, <i11>
    %2:6 = fork [6] %1 {handshake.bb = 0 : ui32, handshake.name = "fork1"} : <i11>
    %3 = extsi %2#0 {handshake.bb = 1 : ui32, handshake.name = "extsi0"} : <i11> to <i32>
    %4 = extsi %2#1 {handshake.bb = 1 : ui32, handshake.name = "extsi1"} : <i11> to <i32>
    %5 = extsi %2#2 {handshake.bb = 1 : ui32, handshake.name = "extsi2"} : <i11> to <i32>
    %6 = extsi %2#3 {handshake.bb = 1 : ui32, handshake.name = "extsi3"} : <i11> to <i32>
    %7 = extsi %2#4 {handshake.bb = 1 : ui32, handshake.name = "extsi4"} : <i11> to <i32>
    %8 = extsi %2#5 {handshake.bb = 1 : ui32, handshake.name = "extsi5"} : <i11> to <i32>
    %9 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant10", value = false} : <>, <i1>
    %10 = br %9 {handshake.bb = 0 : ui32, handshake.name = "br2"} : <i1>
    %11 = extsi %10 {handshake.bb = 0 : ui32, handshake.name = "extsi9"} : <i1> to <i32>
    %12 = br %arg3 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "br3"} : <i32>
    %13 = br %0#10 {handshake.bb = 0 : ui32, handshake.name = "br4"} : <>
    %14 = mux %36#0 [%3, %15] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux2"} : <i1>, [<i32>, <i32>] to <i32>
    %15 = buffer %137#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer8"} : <i32>
    %16 = mux %36#1 [%4, %143#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux3"} : <i1>, [<i32>, <i32>] to <i32>
    %17 = mux %36#2 [%5, %140#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux4"} : <i1>, [<i32>, <i32>] to <i32>
    %18 = mux %36#3 [%6, %135#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux5"} : <i1>, [<i32>, <i32>] to <i32>
    %19 = mux %36#4 [%7, %146#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux6"} : <i1>, [<i32>, <i32>] to <i32>
    %20 = mux %36#5 [%0#9, %161] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux7"} : <i1>, [<>, <>] to <>
    %21 = mux %36#6 [%8, %147] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux8"} : <i1>, [<i32>, <i32>] to <i32>
    %22 = mux %23 [%0#8, %160#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux9"} : <i1>, [<>, <>] to <>
    %23 = buffer %36#7, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer20"} : <i1>
    %24 = mux %25 [%0#7, %152#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux10"} : <i1>, [<>, <>] to <>
    %25 = buffer %36#8, bufferType = FIFO_BREAK_NONE, numSlots = 5 {handshake.bb = 1 : ui32, handshake.name = "buffer21"} : <i1>
    %26 = mux %27 [%0#6, %158#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux11"} : <i1>, [<>, <>] to <>
    %27 = buffer %36#9, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 1 : ui32, handshake.name = "buffer22"} : <i1>
    %28 = mux %29 [%0#5, %154#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux12"} : <i1>, [<>, <>] to <>
    %29 = buffer %36#10, bufferType = FIFO_BREAK_NONE, numSlots = 4 {handshake.bb = 1 : ui32, handshake.name = "buffer23"} : <i1>
    %30 = mux %31 [%0#4, %150#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux13"} : <i1>, [<>, <>] to <>
    %31 = buffer %36#11, bufferType = FIFO_BREAK_NONE, numSlots = 6 {handshake.bb = 1 : ui32, handshake.name = "buffer24"} : <i1>
    %32 = mux %33 [%0#3, %156#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux14"} : <i1>, [<>, <>] to <>
    %33 = buffer %36#12, bufferType = FIFO_BREAK_NONE, numSlots = 3 {handshake.bb = 1 : ui32, handshake.name = "buffer25"} : <i1>
    %34 = init %35 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "init0"} : <i1>
    %35 = buffer %49#16, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 1 : ui32, handshake.name = "buffer26"} : <i1>
    %36:13 = fork [13] %34 {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <i1>
    %37 = buffer %163, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer50"} : <i32>
    %38 = mux %46#0 [%11, %37] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i32>, <i32>] to <i32>
    %39 = buffer %38, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer17"} : <i32>
    %40 = buffer %39, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer18"} : <i32>
    %41:2 = fork [2] %40 {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <i32>
    %42 = mux %46#1 [%12, %164] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i1>, [<i32>, <i32>] to <i32>
    %43 = buffer %42, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer19"} : <i32>
    %44 = buffer %43, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer27"} : <i32>
    %45:2 = fork [2] %44 {handshake.bb = 1 : ui32, handshake.name = "fork4"} : <i32>
    %result, %index = control_merge [%13, %166]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %46:2 = fork [2] %index {handshake.bb = 1 : ui32, handshake.name = "fork5"} : <i1>
    %47 = cmpi slt, %41#1, %45#1 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i32>
    %48 = buffer %47, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer28"} : <i1>
    %49:17 = fork [17] %48 {handshake.bb = 1 : ui32, handshake.name = "fork6"} : <i1>
    %trueResult, %falseResult = cond_br %49#15, %45#0 {handshake.bb = 1 : ui32, handshake.name = "cond_br1"} : <i1>, <i32>
    sink %falseResult {handshake.name = "sink0"} : <i32>
    %trueResult_4, %falseResult_5 = cond_br %49#14, %41#0 {handshake.bb = 1 : ui32, handshake.name = "cond_br2"} : <i1>, <i32>
    sink %falseResult_5 {handshake.name = "sink1"} : <i32>
    %trueResult_6, %falseResult_7 = cond_br %49#13, %result {handshake.bb = 1 : ui32, handshake.name = "cond_br3"} : <i1>, <>
    %50 = buffer %14, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer2"} : <i32>
    %trueResult_8, %falseResult_9 = cond_br %51, %50 {handshake.bb = 2 : ui32, handshake.name = "cond_br22"} : <i1>, <i32>
    %51 = buffer %49#12, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer36"} : <i1>
    sink %falseResult_9 {handshake.name = "sink2"} : <i32>
    %52 = buffer %19, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer6"} : <i32>
    %trueResult_10, %falseResult_11 = cond_br %53, %52 {handshake.bb = 2 : ui32, handshake.name = "cond_br23"} : <i1>, <i32>
    %53 = buffer %49#11, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer37"} : <i1>
    sink %falseResult_11 {handshake.name = "sink3"} : <i32>
    %54 = buffer %32, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer16"} : <>
    %trueResult_12, %falseResult_13 = cond_br %55, %54 {handshake.bb = 2 : ui32, handshake.name = "cond_br24"} : <i1>, <>
    %55 = buffer %49#10, bufferType = FIFO_BREAK_NONE, numSlots = 4 {handshake.bb = 2 : ui32, handshake.name = "buffer38"} : <i1>
    sink %falseResult_13 {handshake.name = "sink4"} : <>
    %56 = buffer %30, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer15"} : <>
    %trueResult_14, %falseResult_15 = cond_br %57, %56 {handshake.bb = 2 : ui32, handshake.name = "cond_br25"} : <i1>, <>
    %57 = buffer %49#9, bufferType = FIFO_BREAK_NONE, numSlots = 7 {handshake.bb = 2 : ui32, handshake.name = "buffer39"} : <i1>
    sink %falseResult_15 {handshake.name = "sink5"} : <>
    %58 = buffer %18, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer5"} : <i32>
    %trueResult_16, %falseResult_17 = cond_br %59, %58 {handshake.bb = 2 : ui32, handshake.name = "cond_br26"} : <i1>, <i32>
    %59 = buffer %49#8, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer40"} : <i1>
    sink %falseResult_17 {handshake.name = "sink6"} : <i32>
    %60 = buffer %21, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer10"} : <i32>
    %trueResult_18, %falseResult_19 = cond_br %61, %60 {handshake.bb = 2 : ui32, handshake.name = "cond_br27"} : <i1>, <i32>
    %61 = buffer %49#7, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer41"} : <i1>
    sink %falseResult_19 {handshake.name = "sink7"} : <i32>
    %62 = buffer %28, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer14"} : <>
    %trueResult_20, %falseResult_21 = cond_br %63, %62 {handshake.bb = 2 : ui32, handshake.name = "cond_br28"} : <i1>, <>
    %63 = buffer %49#6, bufferType = FIFO_BREAK_NONE, numSlots = 5 {handshake.bb = 2 : ui32, handshake.name = "buffer42"} : <i1>
    sink %falseResult_21 {handshake.name = "sink8"} : <>
    %64 = buffer %16, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer3"} : <i32>
    %trueResult_22, %falseResult_23 = cond_br %65, %64 {handshake.bb = 2 : ui32, handshake.name = "cond_br29"} : <i1>, <i32>
    %65 = buffer %49#5, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer43"} : <i1>
    sink %falseResult_23 {handshake.name = "sink9"} : <i32>
    %66 = buffer %26, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer13"} : <>
    %trueResult_24, %falseResult_25 = cond_br %67, %66 {handshake.bb = 2 : ui32, handshake.name = "cond_br30"} : <i1>, <>
    %67 = buffer %49#4, bufferType = FIFO_BREAK_NONE, numSlots = 3 {handshake.bb = 2 : ui32, handshake.name = "buffer44"} : <i1>
    sink %falseResult_25 {handshake.name = "sink10"} : <>
    %68 = buffer %24, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer12"} : <>
    %trueResult_26, %falseResult_27 = cond_br %69, %68 {handshake.bb = 2 : ui32, handshake.name = "cond_br31"} : <i1>, <>
    %69 = buffer %49#3, bufferType = FIFO_BREAK_NONE, numSlots = 6 {handshake.bb = 2 : ui32, handshake.name = "buffer45"} : <i1>
    sink %falseResult_27 {handshake.name = "sink11"} : <>
    %70 = buffer %20, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer7"} : <>
    %71 = buffer %70, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer9"} : <>
    %trueResult_28, %falseResult_29 = cond_br %72, %71 {handshake.bb = 2 : ui32, handshake.name = "cond_br32"} : <i1>, <>
    %72 = buffer %49#2, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer46"} : <i1>
    sink %falseResult_29 {handshake.name = "sink12"} : <>
    %73 = buffer %22, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer11"} : <>
    %trueResult_30, %falseResult_31 = cond_br %74, %73 {handshake.bb = 2 : ui32, handshake.name = "cond_br33"} : <i1>, <>
    %74 = buffer %49#1, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 2 : ui32, handshake.name = "buffer47"} : <i1>
    sink %falseResult_31 {handshake.name = "sink13"} : <>
    %75 = buffer %17, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer4"} : <i32>
    %trueResult_32, %falseResult_33 = cond_br %76, %75 {handshake.bb = 2 : ui32, handshake.name = "cond_br34"} : <i1>, <i32>
    %76 = buffer %49#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer48"} : <i1>
    sink %falseResult_33 {handshake.name = "sink14"} : <i32>
    %77 = merge %trueResult {handshake.bb = 2 : ui32, handshake.name = "merge0"} : <i32>
    %78 = merge %trueResult_4 {handshake.bb = 2 : ui32, handshake.name = "merge1"} : <i32>
    %79:3 = fork [3] %78 {handshake.bb = 2 : ui32, handshake.name = "fork7"} : <i32>
    %80 = trunci %81 {handshake.bb = 2 : ui32, handshake.name = "trunci0"} : <i32> to <i10>
    %81 = buffer %79#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer49"} : <i32>
    %82 = trunci %79#1 {handshake.bb = 2 : ui32, handshake.name = "trunci1"} : <i32> to <i10>
    %result_34, %index_35 = control_merge [%trueResult_6]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>] to <>, <i1>
    sink %index_35 {handshake.name = "sink15"} : <i1>
    %83:2 = fork [2] %result_34 {handshake.bb = 2 : ui32, handshake.name = "fork8"} : <>
    %84 = constant %83#0 {handshake.bb = 2 : ui32, handshake.name = "constant11", value = 1 : i2} : <>, <i2>
    %85 = extsi %84 {handshake.bb = 2 : ui32, handshake.name = "extsi7"} : <i2> to <i32>
    %86 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %87 = constant %86 {handshake.bb = 2 : ui32, handshake.name = "constant12", value = 1 : i2} : <>, <i2>
    %88 = extsi %87 {handshake.bb = 2 : ui32, handshake.name = "extsi8"} : <i2> to <i32>
    %addressResult, %dataResult = load[%82] %outputs_2 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i10>, <i32>, <i10>, <i32>
    %89:4 = fork [4] %dataResult {handshake.bb = 2 : ui32, handshake.name = "fork9"} : <i32>
    %90 = trunci %91 {handshake.bb = 2 : ui32, handshake.name = "trunci2"} : <i32> to <i10>
    %91 = buffer %89#0, bufferType = FIFO_BREAK_NONE, numSlots = 6 {handshake.bb = 2 : ui32, handshake.name = "buffer51"} : <i32>
    %addressResult_36, %dataResult_37 = load[%80] %outputs_0 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i10>, <f32>, <i10>, <f32>
    %92 = gate %89#1, %trueResult_28 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "gate0"} : <i32>, !handshake.control<> to <i32>
    %93:6 = fork [6] %92 {handshake.bb = 2 : ui32, handshake.name = "fork10"} : <i32>
    %94 = cmpi ne, %93#5, %trueResult_16 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi1"} : <i32>
    %95:2 = fork [2] %94 {handshake.bb = 2 : ui32, handshake.name = "fork11"} : <i1>
    %96 = cmpi ne, %93#4, %trueResult_8 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi2"} : <i32>
    %97:2 = fork [2] %96 {handshake.bb = 2 : ui32, handshake.name = "fork12"} : <i1>
    %98 = cmpi ne, %93#3, %trueResult_32 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi3"} : <i32>
    %99:2 = fork [2] %98 {handshake.bb = 2 : ui32, handshake.name = "fork13"} : <i1>
    %100 = cmpi ne, %93#2, %trueResult_22 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi4"} : <i32>
    %101:2 = fork [2] %100 {handshake.bb = 2 : ui32, handshake.name = "fork14"} : <i1>
    %102 = cmpi ne, %93#1, %trueResult_10 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi5"} : <i32>
    %103:2 = fork [2] %102 {handshake.bb = 2 : ui32, handshake.name = "fork15"} : <i1>
    %104 = cmpi ne, %93#0, %trueResult_18 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi6"} : <i32>
    %105:2 = fork [2] %104 {handshake.bb = 2 : ui32, handshake.name = "fork16"} : <i1>
    %trueResult_38, %falseResult_39 = cond_br %106, %trueResult_14 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br16"} : <i1>, <>
    %106 = buffer %95#1, bufferType = FIFO_BREAK_NONE, numSlots = 6 {handshake.bb = 2 : ui32, handshake.name = "buffer59"} : <i1>
    sink %trueResult_38 {handshake.name = "sink16"} : <>
    %trueResult_40, %falseResult_41 = cond_br %107, %trueResult_26 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br17"} : <i1>, <>
    %107 = buffer %97#1, bufferType = FIFO_BREAK_NONE, numSlots = 5 {handshake.bb = 2 : ui32, handshake.name = "buffer60"} : <i1>
    sink %trueResult_40 {handshake.name = "sink17"} : <>
    %trueResult_42, %falseResult_43 = cond_br %108, %trueResult_20 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br18"} : <i1>, <>
    %108 = buffer %99#1, bufferType = FIFO_BREAK_NONE, numSlots = 4 {handshake.bb = 2 : ui32, handshake.name = "buffer61"} : <i1>
    sink %trueResult_42 {handshake.name = "sink18"} : <>
    %trueResult_44, %falseResult_45 = cond_br %109, %trueResult_12 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br19"} : <i1>, <>
    %109 = buffer %101#1, bufferType = FIFO_BREAK_NONE, numSlots = 3 {handshake.bb = 2 : ui32, handshake.name = "buffer62"} : <i1>
    sink %trueResult_44 {handshake.name = "sink19"} : <>
    %trueResult_46, %falseResult_47 = cond_br %110, %trueResult_24 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br20"} : <i1>, <>
    %110 = buffer %103#1, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 2 : ui32, handshake.name = "buffer63"} : <i1>
    sink %trueResult_46 {handshake.name = "sink20"} : <>
    %trueResult_48, %falseResult_49 = cond_br %111, %trueResult_30 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br21"} : <i1>, <>
    %111 = buffer %105#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer64"} : <i1>
    sink %trueResult_48 {handshake.name = "sink21"} : <>
    %112 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %113 = mux %95#0 [%falseResult_39, %112] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux15"} : <i1>, [<>, <>] to <>
    %114 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source2"} : <>
    %115 = mux %97#0 [%falseResult_41, %114] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux16"} : <i1>, [<>, <>] to <>
    %116 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source3"} : <>
    %117 = mux %99#0 [%falseResult_43, %116] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux17"} : <i1>, [<>, <>] to <>
    %118 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source4"} : <>
    %119 = mux %101#0 [%falseResult_45, %118] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux18"} : <i1>, [<>, <>] to <>
    %120 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source5"} : <>
    %121 = mux %103#0 [%falseResult_47, %120] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux19"} : <i1>, [<>, <>] to <>
    %122 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source6"} : <>
    %123 = mux %105#0 [%falseResult_49, %122] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux20"} : <i1>, [<>, <>] to <>
    %124 = buffer %113, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer30"} : <>
    %125 = buffer %115, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer31"} : <>
    %126 = buffer %117, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer32"} : <>
    %127 = buffer %119, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer33"} : <>
    %128 = buffer %121, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer34"} : <>
    %129 = buffer %123, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer35"} : <>
    %130 = join %124, %125, %126, %127, %128, %129 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "join0"} : <>
    %131 = gate %89#2, %130 {handshake.bb = 2 : ui32, handshake.name = "gate1"} : <i32>, !handshake.control<> to <i32>
    %132 = trunci %131 {handshake.bb = 2 : ui32, handshake.name = "trunci3"} : <i32> to <i10>
    %addressResult_50, %dataResult_51 = load[%132] %outputs#0 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["store1", 0, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load5"} : <i10>, <f32>, <i10>, <f32>
    %133 = addf %dataResult_51, %dataResult_37 {fastmath = #arith.fastmath<none>, fpu_impl = #handshake<fpu_impl flopoco>, handshake.bb = 2 : ui32, handshake.name = "addf0", internal_delay = "2_922000"} : <f32>
    %134 = buffer %89#3, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer0"} : <i32>
    %135:2 = fork [2] %134 {handshake.bb = 2 : ui32, handshake.name = "fork17"} : <i32>
    %136 = init %135#0 {handshake.bb = 2 : ui32, handshake.name = "init13"} : <i32>
    %137:2 = fork [2] %136 {handshake.bb = 2 : ui32, handshake.name = "fork18"} : <i32>
    %138 = init %139 {handshake.bb = 2 : ui32, handshake.name = "init14"} : <i32>
    %139 = buffer %137#0, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 2 : ui32, handshake.name = "buffer74"} : <i32>
    %140:2 = fork [2] %138 {handshake.bb = 2 : ui32, handshake.name = "fork19"} : <i32>
    %141 = init %142 {handshake.bb = 2 : ui32, handshake.name = "init15"} : <i32>
    %142 = buffer %140#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer75"} : <i32>
    %143:2 = fork [2] %141 {handshake.bb = 2 : ui32, handshake.name = "fork20"} : <i32>
    %144 = init %145 {handshake.bb = 2 : ui32, handshake.name = "init16"} : <i32>
    %145 = buffer %143#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer76"} : <i32>
    %146:2 = fork [2] %144 {handshake.bb = 2 : ui32, handshake.name = "fork21"} : <i32>
    %147 = init %148 {handshake.bb = 2 : ui32, handshake.name = "init17"} : <i32>
    %148 = buffer %146#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer77"} : <i32>
    %149 = buffer %doneResult, bufferType = FIFO_BREAK_NONE, numSlots = 1 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "buffer1"} : <>
    %150:2 = fork [2] %149 {handshake.bb = 2 : ui32, handshake.name = "fork22"} : <>
    %151 = init %150#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init18"} : <>
    %152:2 = fork [2] %151 {handshake.bb = 2 : ui32, handshake.name = "fork23"} : <>
    %153 = init %152#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init19"} : <>
    %154:2 = fork [2] %153 {handshake.bb = 2 : ui32, handshake.name = "fork24"} : <>
    %155 = init %154#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init20"} : <>
    %156:2 = fork [2] %155 {handshake.bb = 2 : ui32, handshake.name = "fork25"} : <>
    %157 = init %156#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init21"} : <>
    %158:2 = fork [2] %157 {handshake.bb = 2 : ui32, handshake.name = "fork26"} : <>
    %159 = init %158#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init22"} : <>
    %160:2 = fork [2] %159 {handshake.bb = 2 : ui32, handshake.name = "fork27"} : <>
    %161 = init %160#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init23"} : <>
    %addressResult_52, %dataResult_53, %doneResult = store[%90] %133 %outputs#1 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["load5", 0, false], ["store1", 0, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store1"} : <i10>, <f32>, <>, <i10>, <f32>, <>
    %162 = addi %79#2, %88 {handshake.bb = 2 : ui32, handshake.name = "addi0"} : <i32>
    %163 = br %162 {handshake.bb = 2 : ui32, handshake.name = "br5"} : <i32>
    %164 = br %77 {handshake.bb = 2 : ui32, handshake.name = "br6"} : <i32>
    %165 = buffer %83#1, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer29"} : <>
    %166 = br %165 {handshake.bb = 2 : ui32, handshake.name = "br7"} : <>
    %result_54, %index_55 = control_merge [%falseResult_7]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>] to <>, <i1>
    sink %index_55 {handshake.name = "sink22"} : <i1>
    %167:3 = fork [3] %result_54 {handshake.bb = 3 : ui32, handshake.name = "fork28"} : <>
    end {handshake.bb = 3 : ui32, handshake.name = "end0"} %memEnd_3, %memEnd_1, %memEnd, %0#2 : <>, <>, <>, <>
  }
}

