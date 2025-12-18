module {
  handshake.func @histogram(%arg0: memref<1000xi32>, %arg1: memref<1000xf32>, %arg2: memref<1000xf32>, %arg3: !handshake.channel<i32>, %arg4: !handshake.control<>, %arg5: !handshake.control<>, %arg6: !handshake.control<>, %arg7: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["feature", "weight", "hist", "n", "feature_start", "weight_start", "hist_start", "start"], cfg.edges = "[0,1][2,1][1,2,3,cmpi0]", resNames = ["feature_end", "weight_end", "hist_end", "end"]} {
    %0:13 = fork [13] %arg7 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork0"} : <>
    %outputs:2, %memEnd = mem_controller[%arg2 : memref<1000xf32>] %arg6 (%88, %addressResult_60, %addressResult_62, %dataResult_63) %206#2 {connectedBlocks = [2 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "5": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller2"} :    (!handshake.channel<i32>, !handshake.channel<i10>, !handshake.channel<i10>, !handshake.channel<f32>) -> (!handshake.channel<f32>, !handshake.control<>)
    %outputs_0, %memEnd_1 = mem_controller[%arg1 : memref<1000xf32>] %arg5 (%addressResult_42) %206#1 {connectedBlocks = [2 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller3"} :    (!handshake.channel<i10>) -> !handshake.channel<f32>
    %outputs_2, %memEnd_3 = mem_controller[%arg0 : memref<1000xi32>] %arg4 (%addressResult) %206#0 {connectedBlocks = [2 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller4"} :    (!handshake.channel<i10>) -> !handshake.channel<i32>
    %1 = constant %0#1 {handshake.bb = 0 : ui32, handshake.name = "constant11", value = 1000 : i11} : <>, <i11>
    %2:8 = fork [8] %1 {handshake.bb = 0 : ui32, handshake.name = "fork1"} : <i11>
    %3 = extsi %2#0 {handshake.bb = 1 : ui32, handshake.name = "extsi0"} : <i11> to <i32>
    %4 = extsi %2#1 {handshake.bb = 1 : ui32, handshake.name = "extsi1"} : <i11> to <i32>
    %5 = extsi %2#2 {handshake.bb = 1 : ui32, handshake.name = "extsi2"} : <i11> to <i32>
    %6 = extsi %2#3 {handshake.bb = 1 : ui32, handshake.name = "extsi3"} : <i11> to <i32>
    %7 = extsi %2#4 {handshake.bb = 1 : ui32, handshake.name = "extsi4"} : <i11> to <i32>
    %8 = extsi %2#5 {handshake.bb = 1 : ui32, handshake.name = "extsi5"} : <i11> to <i32>
    %9 = extsi %2#6 {handshake.bb = 1 : ui32, handshake.name = "extsi6"} : <i11> to <i32>
    %10 = extsi %2#7 {handshake.bb = 1 : ui32, handshake.name = "extsi7"} : <i11> to <i32>
    %11 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant12", value = false} : <>, <i1>
    %12 = extsi %11 {handshake.bb = 0 : ui32, handshake.name = "extsi11"} : <i1> to <i32>
    %13 = mux %41#0 [%3, %175#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux2"} : <i1>, [<i32>, <i32>] to <i32>
    %14 = mux %41#1 [%4, %184#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux3"} : <i1>, [<i32>, <i32>] to <i32>
    %15 = mux %41#2 [%5, %185] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux4"} : <i1>, [<i32>, <i32>] to <i32>
    %16 = mux %41#3 [%6, %178#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux5"} : <i1>, [<i32>, <i32>] to <i32>
    %17 = mux %41#4 [%7, %181#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux6"} : <i1>, [<i32>, <i32>] to <i32>
    %18 = mux %41#5 [%8, %19] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux7"} : <i1>, [<i32>, <i32>] to <i32>
    %19 = buffer %169#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer19"} : <i32>
    %20 = mux %41#6 [%0#11, %203] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux8"} : <i1>, [<>, <>] to <>
    %21 = mux %41#7 [%9, %167#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux9"} : <i1>, [<i32>, <i32>] to <i32>
    %22 = mux %41#8 [%10, %172#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux10"} : <i1>, [<i32>, <i32>] to <i32>
    %23 = mux %24 [%0#10, %192#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux11"} : <i1>, [<>, <>] to <>
    %24 = buffer %41#9, bufferType = FIFO_BREAK_NONE, numSlots = 6 {handshake.bb = 1 : ui32, handshake.name = "buffer26"} : <i1>
    %25 = mux %26 [%0#9, %200#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux12"} : <i1>, [<>, <>] to <>
    %26 = buffer %41#10, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 1 : ui32, handshake.name = "buffer27"} : <i1>
    %27 = mux %28 [%0#8, %190#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux13"} : <i1>, [<>, <>] to <>
    %28 = buffer %41#11, bufferType = FIFO_BREAK_NONE, numSlots = 7 {handshake.bb = 1 : ui32, handshake.name = "buffer28"} : <i1>
    %29 = mux %30 [%0#7, %202#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux14"} : <i1>, [<>, <>] to <>
    %30 = buffer %41#12, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer29"} : <i1>
    %31 = mux %32 [%0#6, %194#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux15"} : <i1>, [<>, <>] to <>
    %32 = buffer %41#13, bufferType = FIFO_BREAK_NONE, numSlots = 5 {handshake.bb = 1 : ui32, handshake.name = "buffer30"} : <i1>
    %33 = mux %34 [%0#5, %196#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux16"} : <i1>, [<>, <>] to <>
    %34 = buffer %41#14, bufferType = FIFO_BREAK_NONE, numSlots = 4 {handshake.bb = 1 : ui32, handshake.name = "buffer31"} : <i1>
    %35 = mux %36 [%0#4, %188#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux17"} : <i1>, [<>, <>] to <>
    %36 = buffer %41#15, bufferType = FIFO_BREAK_NONE, numSlots = 8 {handshake.bb = 1 : ui32, handshake.name = "buffer32"} : <i1>
    %37 = mux %38 [%0#3, %198#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux18"} : <i1>, [<>, <>] to <>
    %38 = buffer %41#16, bufferType = FIFO_BREAK_NONE, numSlots = 3 {handshake.bb = 1 : ui32, handshake.name = "buffer33"} : <i1>
    %39 = init %40 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "init0"} : <i1>
    %40 = buffer %53#20, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 1 : ui32, handshake.name = "buffer34"} : <i1>
    %41:17 = fork [17] %39 {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <i1>
    %42 = mux %50#0 [%12, %205] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i32>, <i32>] to <i32>
    %43 = buffer %42, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer12"} : <i32>
    %44:2 = fork [2] %43 {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <i32>
    %45 = buffer %trueResult, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer43"} : <i32>
    %46 = mux %50#1 [%arg3, %45] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i1>, [<i32>, <i32>] to <i32>
    %47 = buffer %46, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer14"} : <i32>
    %48:2 = fork [2] %47 {handshake.bb = 1 : ui32, handshake.name = "fork4"} : <i32>
    %49 = buffer %86#1, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer62"} : <>
    %result, %index = control_merge [%0#12, %49]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %50:2 = fork [2] %index {handshake.bb = 1 : ui32, handshake.name = "fork5"} : <i1>
    %51 = cmpi slt, %44#1, %48#1 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i32>
    %52 = buffer %51, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer15"} : <i1>
    %53:21 = fork [21] %52 {handshake.bb = 1 : ui32, handshake.name = "fork6"} : <i1>
    %trueResult, %falseResult = cond_br %53#19, %48#0 {handshake.bb = 1 : ui32, handshake.name = "cond_br1"} : <i1>, <i32>
    sink %falseResult {handshake.name = "sink0"} : <i32>
    %54 = buffer %44#0, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer13"} : <i32>
    %trueResult_4, %falseResult_5 = cond_br %53#18, %54 {handshake.bb = 1 : ui32, handshake.name = "cond_br2"} : <i1>, <i32>
    sink %falseResult_5 {handshake.name = "sink1"} : <i32>
    %trueResult_6, %falseResult_7 = cond_br %53#17, %result {handshake.bb = 1 : ui32, handshake.name = "cond_br3"} : <i1>, <>
    %trueResult_8, %falseResult_9 = cond_br %55, %13 {handshake.bb = 2 : ui32, handshake.name = "cond_br28"} : <i1>, <i32>
    %55 = buffer %53#16, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer44"} : <i1>
    sink %falseResult_9 {handshake.name = "sink2"} : <i32>
    %56 = buffer %27, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer6"} : <>
    %trueResult_10, %falseResult_11 = cond_br %57, %56 {handshake.bb = 2 : ui32, handshake.name = "cond_br29"} : <i1>, <>
    %57 = buffer %53#15, bufferType = FIFO_BREAK_NONE, numSlots = 8 {handshake.bb = 2 : ui32, handshake.name = "buffer45"} : <i1>
    sink %falseResult_11 {handshake.name = "sink3"} : <>
    %58 = buffer %25, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer5"} : <>
    %trueResult_12, %falseResult_13 = cond_br %59, %58 {handshake.bb = 2 : ui32, handshake.name = "cond_br30"} : <i1>, <>
    %59 = buffer %53#14, bufferType = FIFO_BREAK_NONE, numSlots = 3 {handshake.bb = 2 : ui32, handshake.name = "buffer46"} : <i1>
    sink %falseResult_13 {handshake.name = "sink4"} : <>
    %60 = buffer %35, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer10"} : <>
    %trueResult_14, %falseResult_15 = cond_br %61, %60 {handshake.bb = 2 : ui32, handshake.name = "cond_br31"} : <i1>, <>
    %61 = buffer %53#13, bufferType = FIFO_BREAK_NONE, numSlots = 9 {handshake.bb = 2 : ui32, handshake.name = "buffer47"} : <i1>
    sink %falseResult_15 {handshake.name = "sink5"} : <>
    %trueResult_16, %falseResult_17 = cond_br %62, %22 {handshake.bb = 2 : ui32, handshake.name = "cond_br32"} : <i1>, <i32>
    %62 = buffer %53#12, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer48"} : <i1>
    sink %falseResult_17 {handshake.name = "sink6"} : <i32>
    %trueResult_18, %falseResult_19 = cond_br %63, %18 {handshake.bb = 2 : ui32, handshake.name = "cond_br33"} : <i1>, <i32>
    %63 = buffer %53#11, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer49"} : <i1>
    sink %falseResult_19 {handshake.name = "sink7"} : <i32>
    %trueResult_20, %falseResult_21 = cond_br %64, %21 {handshake.bb = 2 : ui32, handshake.name = "cond_br34"} : <i1>, <i32>
    %64 = buffer %53#10, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer50"} : <i1>
    sink %falseResult_21 {handshake.name = "sink8"} : <i32>
    %65 = buffer %20, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer2"} : <>
    %66 = buffer %65, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer3"} : <>
    %trueResult_22, %falseResult_23 = cond_br %67, %66 {handshake.bb = 2 : ui32, handshake.name = "cond_br35"} : <i1>, <>
    %67 = buffer %53#9, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer51"} : <i1>
    sink %falseResult_23 {handshake.name = "sink9"} : <>
    %68 = buffer %37, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer11"} : <>
    %trueResult_24, %falseResult_25 = cond_br %69, %68 {handshake.bb = 2 : ui32, handshake.name = "cond_br36"} : <i1>, <>
    %69 = buffer %53#8, bufferType = FIFO_BREAK_NONE, numSlots = 4 {handshake.bb = 2 : ui32, handshake.name = "buffer52"} : <i1>
    sink %falseResult_25 {handshake.name = "sink10"} : <>
    %trueResult_26, %falseResult_27 = cond_br %70, %14 {handshake.bb = 2 : ui32, handshake.name = "cond_br37"} : <i1>, <i32>
    %70 = buffer %53#7, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer53"} : <i1>
    sink %falseResult_27 {handshake.name = "sink11"} : <i32>
    %71 = buffer %31, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer8"} : <>
    %trueResult_28, %falseResult_29 = cond_br %72, %71 {handshake.bb = 2 : ui32, handshake.name = "cond_br38"} : <i1>, <>
    %72 = buffer %53#6, bufferType = FIFO_BREAK_NONE, numSlots = 6 {handshake.bb = 2 : ui32, handshake.name = "buffer54"} : <i1>
    sink %falseResult_29 {handshake.name = "sink12"} : <>
    %73 = buffer %29, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer7"} : <>
    %trueResult_30, %falseResult_31 = cond_br %74, %73 {handshake.bb = 2 : ui32, handshake.name = "cond_br39"} : <i1>, <>
    %74 = buffer %53#5, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 2 : ui32, handshake.name = "buffer55"} : <i1>
    sink %falseResult_31 {handshake.name = "sink13"} : <>
    %75 = buffer %33, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer9"} : <>
    %trueResult_32, %falseResult_33 = cond_br %76, %75 {handshake.bb = 2 : ui32, handshake.name = "cond_br40"} : <i1>, <>
    %76 = buffer %53#4, bufferType = FIFO_BREAK_NONE, numSlots = 5 {handshake.bb = 2 : ui32, handshake.name = "buffer56"} : <i1>
    sink %falseResult_33 {handshake.name = "sink14"} : <>
    %trueResult_34, %falseResult_35 = cond_br %77, %17 {handshake.bb = 2 : ui32, handshake.name = "cond_br41"} : <i1>, <i32>
    %77 = buffer %53#3, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer57"} : <i1>
    sink %falseResult_35 {handshake.name = "sink15"} : <i32>
    %trueResult_36, %falseResult_37 = cond_br %78, %16 {handshake.bb = 2 : ui32, handshake.name = "cond_br42"} : <i1>, <i32>
    %78 = buffer %53#2, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer58"} : <i1>
    sink %falseResult_37 {handshake.name = "sink16"} : <i32>
    %79 = buffer %23, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer4"} : <>
    %trueResult_38, %falseResult_39 = cond_br %80, %79 {handshake.bb = 2 : ui32, handshake.name = "cond_br43"} : <i1>, <>
    %80 = buffer %53#1, bufferType = FIFO_BREAK_NONE, numSlots = 7 {handshake.bb = 2 : ui32, handshake.name = "buffer59"} : <i1>
    sink %falseResult_39 {handshake.name = "sink17"} : <>
    %trueResult_40, %falseResult_41 = cond_br %81, %15 {handshake.bb = 2 : ui32, handshake.name = "cond_br44"} : <i1>, <i32>
    %81 = buffer %53#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer60"} : <i1>
    sink %falseResult_41 {handshake.name = "sink18"} : <i32>
    %82:3 = fork [3] %trueResult_4 {handshake.bb = 2 : ui32, handshake.name = "fork7"} : <i32>
    %83 = trunci %84 {handshake.bb = 2 : ui32, handshake.name = "trunci0"} : <i32> to <i10>
    %84 = buffer %82#0, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 2 : ui32, handshake.name = "buffer61"} : <i32>
    %85 = trunci %82#1 {handshake.bb = 2 : ui32, handshake.name = "trunci1"} : <i32> to <i10>
    %86:2 = fork [2] %trueResult_6 {handshake.bb = 2 : ui32, handshake.name = "fork8"} : <>
    %87 = constant %86#0 {handshake.bb = 2 : ui32, handshake.name = "constant13", value = 1 : i2} : <>, <i2>
    %88 = extsi %87 {handshake.bb = 2 : ui32, handshake.name = "extsi9"} : <i2> to <i32>
    %89 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %90 = constant %89 {handshake.bb = 2 : ui32, handshake.name = "constant14", value = 1 : i2} : <>, <i2>
    %91 = extsi %90 {handshake.bb = 2 : ui32, handshake.name = "extsi10"} : <i2> to <i32>
    %addressResult, %dataResult = load[%85] %outputs_2 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i10>, <i32>, <i10>, <i32>
    %92:4 = fork [4] %dataResult {handshake.bb = 2 : ui32, handshake.name = "fork9"} : <i32>
    %93 = trunci %94 {handshake.bb = 2 : ui32, handshake.name = "trunci2"} : <i32> to <i10>
    %94 = buffer %92#0, bufferType = FIFO_BREAK_NONE, numSlots = 8 {handshake.bb = 2 : ui32, handshake.name = "buffer63"} : <i32>
    %addressResult_42, %dataResult_43 = load[%83] %outputs_0 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i10>, <f32>, <i10>, <f32>
    %95 = gate %92#1, %trueResult_22 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "gate0"} : <i32>, !handshake.control<> to <i32>
    %96:8 = fork [8] %95 {handshake.bb = 2 : ui32, handshake.name = "fork10"} : <i32>
    %97 = buffer %trueResult_20, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer20"} : <i32>
    %98 = cmpi ne, %96#7, %97 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi1"} : <i32>
    %99:2 = fork [2] %98 {handshake.bb = 2 : ui32, handshake.name = "fork11"} : <i1>
    %100 = buffer %trueResult_18, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer18"} : <i32>
    %101 = cmpi ne, %96#6, %100 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi2"} : <i32>
    %102:2 = fork [2] %101 {handshake.bb = 2 : ui32, handshake.name = "fork12"} : <i1>
    %103 = buffer %trueResult_16, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer17"} : <i32>
    %104 = cmpi ne, %96#5, %103 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi3"} : <i32>
    %105:2 = fork [2] %104 {handshake.bb = 2 : ui32, handshake.name = "fork13"} : <i1>
    %106 = buffer %trueResult_8, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer16"} : <i32>
    %107 = cmpi ne, %96#4, %106 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi4"} : <i32>
    %108:2 = fork [2] %107 {handshake.bb = 2 : ui32, handshake.name = "fork14"} : <i1>
    %109 = buffer %trueResult_36, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer23"} : <i32>
    %110 = cmpi ne, %96#3, %109 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi5"} : <i32>
    %111:2 = fork [2] %110 {handshake.bb = 2 : ui32, handshake.name = "fork15"} : <i1>
    %112 = buffer %trueResult_34, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer22"} : <i32>
    %113 = cmpi ne, %96#2, %112 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi6"} : <i32>
    %114:2 = fork [2] %113 {handshake.bb = 2 : ui32, handshake.name = "fork16"} : <i1>
    %115 = buffer %trueResult_26, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer21"} : <i32>
    %116 = cmpi ne, %96#1, %115 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi7"} : <i32>
    %117:2 = fork [2] %116 {handshake.bb = 2 : ui32, handshake.name = "fork17"} : <i1>
    %118 = buffer %trueResult_40, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer24"} : <i32>
    %119 = cmpi ne, %96#0, %118 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi8"} : <i32>
    %120:2 = fork [2] %119 {handshake.bb = 2 : ui32, handshake.name = "fork18"} : <i1>
    %trueResult_44, %falseResult_45 = cond_br %121, %trueResult_14 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br20"} : <i1>, <>
    %121 = buffer %99#1, bufferType = FIFO_BREAK_NONE, numSlots = 8 {handshake.bb = 2 : ui32, handshake.name = "buffer73"} : <i1>
    sink %trueResult_44 {handshake.name = "sink20"} : <>
    %trueResult_46, %falseResult_47 = cond_br %122, %trueResult_10 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br21"} : <i1>, <>
    %122 = buffer %102#1, bufferType = FIFO_BREAK_NONE, numSlots = 7 {handshake.bb = 2 : ui32, handshake.name = "buffer74"} : <i1>
    sink %trueResult_46 {handshake.name = "sink21"} : <>
    %trueResult_48, %falseResult_49 = cond_br %123, %trueResult_38 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br22"} : <i1>, <>
    %123 = buffer %105#1, bufferType = FIFO_BREAK_NONE, numSlots = 6 {handshake.bb = 2 : ui32, handshake.name = "buffer75"} : <i1>
    sink %trueResult_48 {handshake.name = "sink22"} : <>
    %trueResult_50, %falseResult_51 = cond_br %124, %trueResult_28 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br23"} : <i1>, <>
    %124 = buffer %108#1, bufferType = FIFO_BREAK_NONE, numSlots = 5 {handshake.bb = 2 : ui32, handshake.name = "buffer76"} : <i1>
    sink %trueResult_50 {handshake.name = "sink23"} : <>
    %trueResult_52, %falseResult_53 = cond_br %125, %trueResult_32 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br24"} : <i1>, <>
    %125 = buffer %111#1, bufferType = FIFO_BREAK_NONE, numSlots = 4 {handshake.bb = 2 : ui32, handshake.name = "buffer77"} : <i1>
    sink %trueResult_52 {handshake.name = "sink24"} : <>
    %trueResult_54, %falseResult_55 = cond_br %126, %trueResult_24 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br25"} : <i1>, <>
    %126 = buffer %114#1, bufferType = FIFO_BREAK_NONE, numSlots = 3 {handshake.bb = 2 : ui32, handshake.name = "buffer78"} : <i1>
    sink %trueResult_54 {handshake.name = "sink25"} : <>
    %trueResult_56, %falseResult_57 = cond_br %127, %trueResult_12 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br26"} : <i1>, <>
    %127 = buffer %117#1, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 2 : ui32, handshake.name = "buffer79"} : <i1>
    sink %trueResult_56 {handshake.name = "sink26"} : <>
    %trueResult_58, %falseResult_59 = cond_br %128, %trueResult_30 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br27"} : <i1>, <>
    %128 = buffer %120#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer80"} : <i1>
    sink %trueResult_58 {handshake.name = "sink27"} : <>
    %129 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %130 = mux %131 [%falseResult_45, %129] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux19"} : <i1>, [<>, <>] to <>
    %131 = buffer %99#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer81"} : <i1>
    %132 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source2"} : <>
    %133 = mux %134 [%falseResult_47, %132] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux20"} : <i1>, [<>, <>] to <>
    %134 = buffer %102#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer82"} : <i1>
    %135 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source3"} : <>
    %136 = mux %137 [%falseResult_49, %135] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux21"} : <i1>, [<>, <>] to <>
    %137 = buffer %105#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer83"} : <i1>
    %138 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source4"} : <>
    %139 = mux %140 [%falseResult_51, %138] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux22"} : <i1>, [<>, <>] to <>
    %140 = buffer %108#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer84"} : <i1>
    %141 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source5"} : <>
    %142 = mux %143 [%falseResult_53, %141] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux23"} : <i1>, [<>, <>] to <>
    %143 = buffer %111#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer85"} : <i1>
    %144 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source6"} : <>
    %145 = mux %146 [%falseResult_55, %144] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux24"} : <i1>, [<>, <>] to <>
    %146 = buffer %114#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer86"} : <i1>
    %147 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source7"} : <>
    %148 = mux %149 [%falseResult_57, %147] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux25"} : <i1>, [<>, <>] to <>
    %149 = buffer %117#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer87"} : <i1>
    %150 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source8"} : <>
    %151 = mux %152 [%falseResult_59, %150] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux26"} : <i1>, [<>, <>] to <>
    %152 = buffer %120#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer88"} : <i1>
    %153 = buffer %130, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer25"} : <>
    %154 = buffer %133, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer35"} : <>
    %155 = buffer %136, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer36"} : <>
    %156 = buffer %139, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer37"} : <>
    %157 = buffer %142, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer38"} : <>
    %158 = buffer %145, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer39"} : <>
    %159 = buffer %148, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer40"} : <>
    %160 = buffer %151, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer41"} : <>
    %161 = join %153, %154, %155, %156, %157, %158, %159, %160 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "join0"} : <>
    %162 = gate %163, %161 {handshake.bb = 2 : ui32, handshake.name = "gate1"} : <i32>, !handshake.control<> to <i32>
    %163 = buffer %92#2, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer89"} : <i32>
    %164 = trunci %162 {handshake.bb = 2 : ui32, handshake.name = "trunci3"} : <i32> to <i10>
    %addressResult_60, %dataResult_61 = load[%164] %outputs#0 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["store1", 0, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load5"} : <i10>, <f32>, <i10>, <f32>
    %165 = addf %dataResult_61, %dataResult_43 {fastmath = #arith.fastmath<none>, fpu_impl = #handshake<fpu_impl flopoco>, handshake.bb = 2 : ui32, handshake.name = "addf0", internal_delay = "2_922000"} : <f32>
    %166 = buffer %92#3, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer0"} : <i32>
    %167:2 = fork [2] %166 {handshake.bb = 2 : ui32, handshake.name = "fork19"} : <i32>
    %168 = init %167#0 {handshake.bb = 2 : ui32, handshake.name = "init17"} : <i32>
    %169:2 = fork [2] %168 {handshake.bb = 2 : ui32, handshake.name = "fork20"} : <i32>
    %170 = init %171 {handshake.bb = 2 : ui32, handshake.name = "init18"} : <i32>
    %171 = buffer %169#0, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 2 : ui32, handshake.name = "buffer92"} : <i32>
    %172:2 = fork [2] %170 {handshake.bb = 2 : ui32, handshake.name = "fork21"} : <i32>
    %173 = init %174 {handshake.bb = 2 : ui32, handshake.name = "init19"} : <i32>
    %174 = buffer %172#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer93"} : <i32>
    %175:2 = fork [2] %173 {handshake.bb = 2 : ui32, handshake.name = "fork22"} : <i32>
    %176 = init %177 {handshake.bb = 2 : ui32, handshake.name = "init20"} : <i32>
    %177 = buffer %175#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer94"} : <i32>
    %178:2 = fork [2] %176 {handshake.bb = 2 : ui32, handshake.name = "fork23"} : <i32>
    %179 = init %180 {handshake.bb = 2 : ui32, handshake.name = "init21"} : <i32>
    %180 = buffer %178#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer95"} : <i32>
    %181:2 = fork [2] %179 {handshake.bb = 2 : ui32, handshake.name = "fork24"} : <i32>
    %182 = init %183 {handshake.bb = 2 : ui32, handshake.name = "init22"} : <i32>
    %183 = buffer %181#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer96"} : <i32>
    %184:2 = fork [2] %182 {handshake.bb = 2 : ui32, handshake.name = "fork25"} : <i32>
    %185 = init %186 {handshake.bb = 2 : ui32, handshake.name = "init23"} : <i32>
    %186 = buffer %184#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer97"} : <i32>
    %187 = buffer %doneResult, bufferType = FIFO_BREAK_NONE, numSlots = 1 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "buffer1"} : <>
    %188:2 = fork [2] %187 {handshake.bb = 2 : ui32, handshake.name = "fork26"} : <>
    %189 = init %188#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init24"} : <>
    %190:2 = fork [2] %189 {handshake.bb = 2 : ui32, handshake.name = "fork27"} : <>
    %191 = init %190#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init25"} : <>
    %192:2 = fork [2] %191 {handshake.bb = 2 : ui32, handshake.name = "fork28"} : <>
    %193 = init %192#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init26"} : <>
    %194:2 = fork [2] %193 {handshake.bb = 2 : ui32, handshake.name = "fork29"} : <>
    %195 = init %194#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init27"} : <>
    %196:2 = fork [2] %195 {handshake.bb = 2 : ui32, handshake.name = "fork30"} : <>
    %197 = init %196#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init28"} : <>
    %198:2 = fork [2] %197 {handshake.bb = 2 : ui32, handshake.name = "fork31"} : <>
    %199 = init %198#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init29"} : <>
    %200:2 = fork [2] %199 {handshake.bb = 2 : ui32, handshake.name = "fork32"} : <>
    %201 = init %200#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init30"} : <>
    %202:2 = fork [2] %201 {handshake.bb = 2 : ui32, handshake.name = "fork33"} : <>
    %203 = init %202#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init31"} : <>
    %addressResult_62, %dataResult_63, %doneResult = store[%93] %165 %outputs#1 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["load5", 0, false], ["store1", 0, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store1"} : <i10>, <f32>, <>, <i10>, <f32>, <>
    %204 = addi %82#2, %91 {handshake.bb = 2 : ui32, handshake.name = "addi0"} : <i32>
    %205 = buffer %204, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer42"} : <i32>
    %206:3 = fork [3] %falseResult_7 {handshake.bb = 3 : ui32, handshake.name = "fork34"} : <>
    end {handshake.bb = 3 : ui32, handshake.name = "end0"} %memEnd_3, %memEnd_1, %memEnd, %0#2 : <>, <>, <>, <>
  }
}

