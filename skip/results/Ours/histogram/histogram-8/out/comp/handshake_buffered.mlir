module {
  handshake.func @histogram(%arg0: memref<1000xi32>, %arg1: memref<1000xf32>, %arg2: memref<1000xf32>, %arg3: !handshake.channel<i32>, %arg4: !handshake.control<>, %arg5: !handshake.control<>, %arg6: !handshake.control<>, %arg7: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["feature", "weight", "hist", "n", "feature_start", "weight_start", "hist_start", "start"], cfg.edges = "[0,1][2,1][1,2,3,cmpi0]", resNames = ["feature_end", "weight_end", "hist_end", "end"]} {
    %0:13 = fork [13] %arg7 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork0"} : <>
    %outputs:2, %memEnd = mem_controller[%arg2 : memref<1000xf32>] %arg6 (%93, %addressResult_62, %addressResult_64, %dataResult_65) %214#2 {connectedBlocks = [2 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "5": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller2"} :    (!handshake.channel<i32>, !handshake.channel<i10>, !handshake.channel<i10>, !handshake.channel<f32>) -> (!handshake.channel<f32>, !handshake.control<>)
    %outputs_0, %memEnd_1 = mem_controller[%arg1 : memref<1000xf32>] %arg5 (%addressResult_44) %214#1 {connectedBlocks = [2 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller3"} :    (!handshake.channel<i10>) -> !handshake.channel<f32>
    %outputs_2, %memEnd_3 = mem_controller[%arg0 : memref<1000xi32>] %arg4 (%addressResult) %214#0 {connectedBlocks = [2 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller4"} :    (!handshake.channel<i10>) -> !handshake.channel<i32>
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
    %12 = br %11 {handshake.bb = 0 : ui32, handshake.name = "br2"} : <i1>
    %13 = extsi %12 {handshake.bb = 0 : ui32, handshake.name = "extsi11"} : <i1> to <i32>
    %14 = br %arg3 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "br3"} : <i32>
    %15 = br %0#12 {handshake.bb = 0 : ui32, handshake.name = "br4"} : <>
    %16 = mux %44#0 [%3, %180#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux2"} : <i1>, [<i32>, <i32>] to <i32>
    %17 = mux %44#1 [%4, %189#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux3"} : <i1>, [<i32>, <i32>] to <i32>
    %18 = mux %44#2 [%5, %190] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux4"} : <i1>, [<i32>, <i32>] to <i32>
    %19 = mux %44#3 [%6, %183#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux5"} : <i1>, [<i32>, <i32>] to <i32>
    %20 = mux %44#4 [%7, %186#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux6"} : <i1>, [<i32>, <i32>] to <i32>
    %21 = mux %44#5 [%8, %22] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux7"} : <i1>, [<i32>, <i32>] to <i32>
    %22 = buffer %174#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer19"} : <i32>
    %23 = mux %44#6 [%0#11, %208] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux8"} : <i1>, [<>, <>] to <>
    %24 = mux %44#7 [%9, %172#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux9"} : <i1>, [<i32>, <i32>] to <i32>
    %25 = mux %44#8 [%10, %177#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux10"} : <i1>, [<i32>, <i32>] to <i32>
    %26 = mux %27 [%0#10, %197#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux11"} : <i1>, [<>, <>] to <>
    %27 = buffer %44#9, bufferType = FIFO_BREAK_NONE, numSlots = 6 {handshake.bb = 1 : ui32, handshake.name = "buffer26"} : <i1>
    %28 = mux %29 [%0#9, %205#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux12"} : <i1>, [<>, <>] to <>
    %29 = buffer %44#10, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 1 : ui32, handshake.name = "buffer27"} : <i1>
    %30 = mux %31 [%0#8, %195#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux13"} : <i1>, [<>, <>] to <>
    %31 = buffer %44#11, bufferType = FIFO_BREAK_NONE, numSlots = 7 {handshake.bb = 1 : ui32, handshake.name = "buffer28"} : <i1>
    %32 = mux %33 [%0#7, %207#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux14"} : <i1>, [<>, <>] to <>
    %33 = buffer %44#12, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer29"} : <i1>
    %34 = mux %35 [%0#6, %199#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux15"} : <i1>, [<>, <>] to <>
    %35 = buffer %44#13, bufferType = FIFO_BREAK_NONE, numSlots = 5 {handshake.bb = 1 : ui32, handshake.name = "buffer30"} : <i1>
    %36 = mux %37 [%0#5, %201#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux16"} : <i1>, [<>, <>] to <>
    %37 = buffer %44#14, bufferType = FIFO_BREAK_NONE, numSlots = 4 {handshake.bb = 1 : ui32, handshake.name = "buffer31"} : <i1>
    %38 = mux %39 [%0#4, %193#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux17"} : <i1>, [<>, <>] to <>
    %39 = buffer %44#15, bufferType = FIFO_BREAK_NONE, numSlots = 8 {handshake.bb = 1 : ui32, handshake.name = "buffer32"} : <i1>
    %40 = mux %41 [%0#3, %203#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux18"} : <i1>, [<>, <>] to <>
    %41 = buffer %44#16, bufferType = FIFO_BREAK_NONE, numSlots = 3 {handshake.bb = 1 : ui32, handshake.name = "buffer33"} : <i1>
    %42 = init %43 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "init0"} : <i1>
    %43 = buffer %56#20, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 1 : ui32, handshake.name = "buffer34"} : <i1>
    %44:17 = fork [17] %42 {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <i1>
    %45 = mux %53#0 [%13, %211] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i32>, <i32>] to <i32>
    %46 = buffer %45, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer12"} : <i32>
    %47:2 = fork [2] %46 {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <i32>
    %48 = buffer %212, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer43"} : <i32>
    %49 = mux %53#1 [%14, %48] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i1>, [<i32>, <i32>] to <i32>
    %50 = buffer %49, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer14"} : <i32>
    %51:2 = fork [2] %50 {handshake.bb = 1 : ui32, handshake.name = "fork4"} : <i32>
    %52 = buffer %213, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer62"} : <>
    %result, %index = control_merge [%15, %52]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %53:2 = fork [2] %index {handshake.bb = 1 : ui32, handshake.name = "fork5"} : <i1>
    %54 = cmpi slt, %47#1, %51#1 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i32>
    %55 = buffer %54, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer15"} : <i1>
    %56:21 = fork [21] %55 {handshake.bb = 1 : ui32, handshake.name = "fork6"} : <i1>
    %trueResult, %falseResult = cond_br %56#19, %51#0 {handshake.bb = 1 : ui32, handshake.name = "cond_br1"} : <i1>, <i32>
    sink %falseResult {handshake.name = "sink0"} : <i32>
    %57 = buffer %47#0, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer13"} : <i32>
    %trueResult_4, %falseResult_5 = cond_br %56#18, %57 {handshake.bb = 1 : ui32, handshake.name = "cond_br2"} : <i1>, <i32>
    sink %falseResult_5 {handshake.name = "sink1"} : <i32>
    %trueResult_6, %falseResult_7 = cond_br %56#17, %result {handshake.bb = 1 : ui32, handshake.name = "cond_br3"} : <i1>, <>
    %trueResult_8, %falseResult_9 = cond_br %58, %16 {handshake.bb = 2 : ui32, handshake.name = "cond_br28"} : <i1>, <i32>
    %58 = buffer %56#16, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer44"} : <i1>
    sink %falseResult_9 {handshake.name = "sink2"} : <i32>
    %59 = buffer %30, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer6"} : <>
    %trueResult_10, %falseResult_11 = cond_br %60, %59 {handshake.bb = 2 : ui32, handshake.name = "cond_br29"} : <i1>, <>
    %60 = buffer %56#15, bufferType = FIFO_BREAK_NONE, numSlots = 8 {handshake.bb = 2 : ui32, handshake.name = "buffer45"} : <i1>
    sink %falseResult_11 {handshake.name = "sink3"} : <>
    %61 = buffer %28, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer5"} : <>
    %trueResult_12, %falseResult_13 = cond_br %62, %61 {handshake.bb = 2 : ui32, handshake.name = "cond_br30"} : <i1>, <>
    %62 = buffer %56#14, bufferType = FIFO_BREAK_NONE, numSlots = 3 {handshake.bb = 2 : ui32, handshake.name = "buffer46"} : <i1>
    sink %falseResult_13 {handshake.name = "sink4"} : <>
    %63 = buffer %38, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer10"} : <>
    %trueResult_14, %falseResult_15 = cond_br %64, %63 {handshake.bb = 2 : ui32, handshake.name = "cond_br31"} : <i1>, <>
    %64 = buffer %56#13, bufferType = FIFO_BREAK_NONE, numSlots = 9 {handshake.bb = 2 : ui32, handshake.name = "buffer47"} : <i1>
    sink %falseResult_15 {handshake.name = "sink5"} : <>
    %trueResult_16, %falseResult_17 = cond_br %65, %25 {handshake.bb = 2 : ui32, handshake.name = "cond_br32"} : <i1>, <i32>
    %65 = buffer %56#12, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer48"} : <i1>
    sink %falseResult_17 {handshake.name = "sink6"} : <i32>
    %trueResult_18, %falseResult_19 = cond_br %66, %21 {handshake.bb = 2 : ui32, handshake.name = "cond_br33"} : <i1>, <i32>
    %66 = buffer %56#11, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer49"} : <i1>
    sink %falseResult_19 {handshake.name = "sink7"} : <i32>
    %trueResult_20, %falseResult_21 = cond_br %67, %24 {handshake.bb = 2 : ui32, handshake.name = "cond_br34"} : <i1>, <i32>
    %67 = buffer %56#10, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer50"} : <i1>
    sink %falseResult_21 {handshake.name = "sink8"} : <i32>
    %68 = buffer %23, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer2"} : <>
    %69 = buffer %68, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer3"} : <>
    %trueResult_22, %falseResult_23 = cond_br %70, %69 {handshake.bb = 2 : ui32, handshake.name = "cond_br35"} : <i1>, <>
    %70 = buffer %56#9, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer51"} : <i1>
    sink %falseResult_23 {handshake.name = "sink9"} : <>
    %71 = buffer %40, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer11"} : <>
    %trueResult_24, %falseResult_25 = cond_br %72, %71 {handshake.bb = 2 : ui32, handshake.name = "cond_br36"} : <i1>, <>
    %72 = buffer %56#8, bufferType = FIFO_BREAK_NONE, numSlots = 4 {handshake.bb = 2 : ui32, handshake.name = "buffer52"} : <i1>
    sink %falseResult_25 {handshake.name = "sink10"} : <>
    %trueResult_26, %falseResult_27 = cond_br %73, %17 {handshake.bb = 2 : ui32, handshake.name = "cond_br37"} : <i1>, <i32>
    %73 = buffer %56#7, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer53"} : <i1>
    sink %falseResult_27 {handshake.name = "sink11"} : <i32>
    %74 = buffer %34, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer8"} : <>
    %trueResult_28, %falseResult_29 = cond_br %75, %74 {handshake.bb = 2 : ui32, handshake.name = "cond_br38"} : <i1>, <>
    %75 = buffer %56#6, bufferType = FIFO_BREAK_NONE, numSlots = 6 {handshake.bb = 2 : ui32, handshake.name = "buffer54"} : <i1>
    sink %falseResult_29 {handshake.name = "sink12"} : <>
    %76 = buffer %32, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer7"} : <>
    %trueResult_30, %falseResult_31 = cond_br %77, %76 {handshake.bb = 2 : ui32, handshake.name = "cond_br39"} : <i1>, <>
    %77 = buffer %56#5, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 2 : ui32, handshake.name = "buffer55"} : <i1>
    sink %falseResult_31 {handshake.name = "sink13"} : <>
    %78 = buffer %36, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer9"} : <>
    %trueResult_32, %falseResult_33 = cond_br %79, %78 {handshake.bb = 2 : ui32, handshake.name = "cond_br40"} : <i1>, <>
    %79 = buffer %56#4, bufferType = FIFO_BREAK_NONE, numSlots = 5 {handshake.bb = 2 : ui32, handshake.name = "buffer56"} : <i1>
    sink %falseResult_33 {handshake.name = "sink14"} : <>
    %trueResult_34, %falseResult_35 = cond_br %80, %20 {handshake.bb = 2 : ui32, handshake.name = "cond_br41"} : <i1>, <i32>
    %80 = buffer %56#3, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer57"} : <i1>
    sink %falseResult_35 {handshake.name = "sink15"} : <i32>
    %trueResult_36, %falseResult_37 = cond_br %81, %19 {handshake.bb = 2 : ui32, handshake.name = "cond_br42"} : <i1>, <i32>
    %81 = buffer %56#2, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer58"} : <i1>
    sink %falseResult_37 {handshake.name = "sink16"} : <i32>
    %82 = buffer %26, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer4"} : <>
    %trueResult_38, %falseResult_39 = cond_br %83, %82 {handshake.bb = 2 : ui32, handshake.name = "cond_br43"} : <i1>, <>
    %83 = buffer %56#1, bufferType = FIFO_BREAK_NONE, numSlots = 7 {handshake.bb = 2 : ui32, handshake.name = "buffer59"} : <i1>
    sink %falseResult_39 {handshake.name = "sink17"} : <>
    %trueResult_40, %falseResult_41 = cond_br %84, %18 {handshake.bb = 2 : ui32, handshake.name = "cond_br44"} : <i1>, <i32>
    %84 = buffer %56#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer60"} : <i1>
    sink %falseResult_41 {handshake.name = "sink18"} : <i32>
    %85 = merge %trueResult {handshake.bb = 2 : ui32, handshake.name = "merge0"} : <i32>
    %86 = merge %trueResult_4 {handshake.bb = 2 : ui32, handshake.name = "merge1"} : <i32>
    %87:3 = fork [3] %86 {handshake.bb = 2 : ui32, handshake.name = "fork7"} : <i32>
    %88 = trunci %89 {handshake.bb = 2 : ui32, handshake.name = "trunci0"} : <i32> to <i10>
    %89 = buffer %87#0, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 2 : ui32, handshake.name = "buffer61"} : <i32>
    %90 = trunci %87#1 {handshake.bb = 2 : ui32, handshake.name = "trunci1"} : <i32> to <i10>
    %result_42, %index_43 = control_merge [%trueResult_6]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>] to <>, <i1>
    sink %index_43 {handshake.name = "sink19"} : <i1>
    %91:2 = fork [2] %result_42 {handshake.bb = 2 : ui32, handshake.name = "fork8"} : <>
    %92 = constant %91#0 {handshake.bb = 2 : ui32, handshake.name = "constant13", value = 1 : i2} : <>, <i2>
    %93 = extsi %92 {handshake.bb = 2 : ui32, handshake.name = "extsi9"} : <i2> to <i32>
    %94 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %95 = constant %94 {handshake.bb = 2 : ui32, handshake.name = "constant14", value = 1 : i2} : <>, <i2>
    %96 = extsi %95 {handshake.bb = 2 : ui32, handshake.name = "extsi10"} : <i2> to <i32>
    %addressResult, %dataResult = load[%90] %outputs_2 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i10>, <i32>, <i10>, <i32>
    %97:4 = fork [4] %dataResult {handshake.bb = 2 : ui32, handshake.name = "fork9"} : <i32>
    %98 = trunci %99 {handshake.bb = 2 : ui32, handshake.name = "trunci2"} : <i32> to <i10>
    %99 = buffer %97#0, bufferType = FIFO_BREAK_NONE, numSlots = 8 {handshake.bb = 2 : ui32, handshake.name = "buffer63"} : <i32>
    %addressResult_44, %dataResult_45 = load[%88] %outputs_0 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i10>, <f32>, <i10>, <f32>
    %100 = gate %97#1, %trueResult_22 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "gate0"} : <i32>, !handshake.control<> to <i32>
    %101:8 = fork [8] %100 {handshake.bb = 2 : ui32, handshake.name = "fork10"} : <i32>
    %102 = buffer %trueResult_20, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer20"} : <i32>
    %103 = cmpi ne, %101#7, %102 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi1"} : <i32>
    %104:2 = fork [2] %103 {handshake.bb = 2 : ui32, handshake.name = "fork11"} : <i1>
    %105 = buffer %trueResult_18, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer18"} : <i32>
    %106 = cmpi ne, %101#6, %105 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi2"} : <i32>
    %107:2 = fork [2] %106 {handshake.bb = 2 : ui32, handshake.name = "fork12"} : <i1>
    %108 = buffer %trueResult_16, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer17"} : <i32>
    %109 = cmpi ne, %101#5, %108 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi3"} : <i32>
    %110:2 = fork [2] %109 {handshake.bb = 2 : ui32, handshake.name = "fork13"} : <i1>
    %111 = buffer %trueResult_8, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer16"} : <i32>
    %112 = cmpi ne, %101#4, %111 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi4"} : <i32>
    %113:2 = fork [2] %112 {handshake.bb = 2 : ui32, handshake.name = "fork14"} : <i1>
    %114 = buffer %trueResult_36, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer23"} : <i32>
    %115 = cmpi ne, %101#3, %114 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi5"} : <i32>
    %116:2 = fork [2] %115 {handshake.bb = 2 : ui32, handshake.name = "fork15"} : <i1>
    %117 = buffer %trueResult_34, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer22"} : <i32>
    %118 = cmpi ne, %101#2, %117 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi6"} : <i32>
    %119:2 = fork [2] %118 {handshake.bb = 2 : ui32, handshake.name = "fork16"} : <i1>
    %120 = buffer %trueResult_26, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer21"} : <i32>
    %121 = cmpi ne, %101#1, %120 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi7"} : <i32>
    %122:2 = fork [2] %121 {handshake.bb = 2 : ui32, handshake.name = "fork17"} : <i1>
    %123 = buffer %trueResult_40, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer24"} : <i32>
    %124 = cmpi ne, %101#0, %123 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi8"} : <i32>
    %125:2 = fork [2] %124 {handshake.bb = 2 : ui32, handshake.name = "fork18"} : <i1>
    %trueResult_46, %falseResult_47 = cond_br %126, %trueResult_14 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br20"} : <i1>, <>
    %126 = buffer %104#1, bufferType = FIFO_BREAK_NONE, numSlots = 8 {handshake.bb = 2 : ui32, handshake.name = "buffer73"} : <i1>
    sink %trueResult_46 {handshake.name = "sink20"} : <>
    %trueResult_48, %falseResult_49 = cond_br %127, %trueResult_10 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br21"} : <i1>, <>
    %127 = buffer %107#1, bufferType = FIFO_BREAK_NONE, numSlots = 7 {handshake.bb = 2 : ui32, handshake.name = "buffer74"} : <i1>
    sink %trueResult_48 {handshake.name = "sink21"} : <>
    %trueResult_50, %falseResult_51 = cond_br %128, %trueResult_38 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br22"} : <i1>, <>
    %128 = buffer %110#1, bufferType = FIFO_BREAK_NONE, numSlots = 6 {handshake.bb = 2 : ui32, handshake.name = "buffer75"} : <i1>
    sink %trueResult_50 {handshake.name = "sink22"} : <>
    %trueResult_52, %falseResult_53 = cond_br %129, %trueResult_28 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br23"} : <i1>, <>
    %129 = buffer %113#1, bufferType = FIFO_BREAK_NONE, numSlots = 5 {handshake.bb = 2 : ui32, handshake.name = "buffer76"} : <i1>
    sink %trueResult_52 {handshake.name = "sink23"} : <>
    %trueResult_54, %falseResult_55 = cond_br %130, %trueResult_32 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br24"} : <i1>, <>
    %130 = buffer %116#1, bufferType = FIFO_BREAK_NONE, numSlots = 4 {handshake.bb = 2 : ui32, handshake.name = "buffer77"} : <i1>
    sink %trueResult_54 {handshake.name = "sink24"} : <>
    %trueResult_56, %falseResult_57 = cond_br %131, %trueResult_24 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br25"} : <i1>, <>
    %131 = buffer %119#1, bufferType = FIFO_BREAK_NONE, numSlots = 3 {handshake.bb = 2 : ui32, handshake.name = "buffer78"} : <i1>
    sink %trueResult_56 {handshake.name = "sink25"} : <>
    %trueResult_58, %falseResult_59 = cond_br %132, %trueResult_12 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br26"} : <i1>, <>
    %132 = buffer %122#1, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 2 : ui32, handshake.name = "buffer79"} : <i1>
    sink %trueResult_58 {handshake.name = "sink26"} : <>
    %trueResult_60, %falseResult_61 = cond_br %133, %trueResult_30 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br27"} : <i1>, <>
    %133 = buffer %125#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer80"} : <i1>
    sink %trueResult_60 {handshake.name = "sink27"} : <>
    %134 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %135 = mux %136 [%falseResult_47, %134] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux19"} : <i1>, [<>, <>] to <>
    %136 = buffer %104#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer81"} : <i1>
    %137 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source2"} : <>
    %138 = mux %139 [%falseResult_49, %137] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux20"} : <i1>, [<>, <>] to <>
    %139 = buffer %107#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer82"} : <i1>
    %140 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source3"} : <>
    %141 = mux %142 [%falseResult_51, %140] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux21"} : <i1>, [<>, <>] to <>
    %142 = buffer %110#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer83"} : <i1>
    %143 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source4"} : <>
    %144 = mux %145 [%falseResult_53, %143] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux22"} : <i1>, [<>, <>] to <>
    %145 = buffer %113#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer84"} : <i1>
    %146 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source5"} : <>
    %147 = mux %148 [%falseResult_55, %146] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux23"} : <i1>, [<>, <>] to <>
    %148 = buffer %116#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer85"} : <i1>
    %149 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source6"} : <>
    %150 = mux %151 [%falseResult_57, %149] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux24"} : <i1>, [<>, <>] to <>
    %151 = buffer %119#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer86"} : <i1>
    %152 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source7"} : <>
    %153 = mux %154 [%falseResult_59, %152] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux25"} : <i1>, [<>, <>] to <>
    %154 = buffer %122#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer87"} : <i1>
    %155 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source8"} : <>
    %156 = mux %157 [%falseResult_61, %155] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux26"} : <i1>, [<>, <>] to <>
    %157 = buffer %125#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer88"} : <i1>
    %158 = buffer %135, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer25"} : <>
    %159 = buffer %138, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer35"} : <>
    %160 = buffer %141, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer36"} : <>
    %161 = buffer %144, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer37"} : <>
    %162 = buffer %147, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer38"} : <>
    %163 = buffer %150, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer39"} : <>
    %164 = buffer %153, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer40"} : <>
    %165 = buffer %156, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer41"} : <>
    %166 = join %158, %159, %160, %161, %162, %163, %164, %165 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "join0"} : <>
    %167 = gate %168, %166 {handshake.bb = 2 : ui32, handshake.name = "gate1"} : <i32>, !handshake.control<> to <i32>
    %168 = buffer %97#2, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer89"} : <i32>
    %169 = trunci %167 {handshake.bb = 2 : ui32, handshake.name = "trunci3"} : <i32> to <i10>
    %addressResult_62, %dataResult_63 = load[%169] %outputs#0 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["store1", 0, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load5"} : <i10>, <f32>, <i10>, <f32>
    %170 = addf %dataResult_63, %dataResult_45 {fastmath = #arith.fastmath<none>, fpu_impl = #handshake<fpu_impl flopoco>, handshake.bb = 2 : ui32, handshake.name = "addf0", internal_delay = "2_922000"} : <f32>
    %171 = buffer %97#3, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer0"} : <i32>
    %172:2 = fork [2] %171 {handshake.bb = 2 : ui32, handshake.name = "fork19"} : <i32>
    %173 = init %172#0 {handshake.bb = 2 : ui32, handshake.name = "init17"} : <i32>
    %174:2 = fork [2] %173 {handshake.bb = 2 : ui32, handshake.name = "fork20"} : <i32>
    %175 = init %176 {handshake.bb = 2 : ui32, handshake.name = "init18"} : <i32>
    %176 = buffer %174#0, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 2 : ui32, handshake.name = "buffer92"} : <i32>
    %177:2 = fork [2] %175 {handshake.bb = 2 : ui32, handshake.name = "fork21"} : <i32>
    %178 = init %179 {handshake.bb = 2 : ui32, handshake.name = "init19"} : <i32>
    %179 = buffer %177#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer93"} : <i32>
    %180:2 = fork [2] %178 {handshake.bb = 2 : ui32, handshake.name = "fork22"} : <i32>
    %181 = init %182 {handshake.bb = 2 : ui32, handshake.name = "init20"} : <i32>
    %182 = buffer %180#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer94"} : <i32>
    %183:2 = fork [2] %181 {handshake.bb = 2 : ui32, handshake.name = "fork23"} : <i32>
    %184 = init %185 {handshake.bb = 2 : ui32, handshake.name = "init21"} : <i32>
    %185 = buffer %183#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer95"} : <i32>
    %186:2 = fork [2] %184 {handshake.bb = 2 : ui32, handshake.name = "fork24"} : <i32>
    %187 = init %188 {handshake.bb = 2 : ui32, handshake.name = "init22"} : <i32>
    %188 = buffer %186#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer96"} : <i32>
    %189:2 = fork [2] %187 {handshake.bb = 2 : ui32, handshake.name = "fork25"} : <i32>
    %190 = init %191 {handshake.bb = 2 : ui32, handshake.name = "init23"} : <i32>
    %191 = buffer %189#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer97"} : <i32>
    %192 = buffer %doneResult, bufferType = FIFO_BREAK_NONE, numSlots = 1 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "buffer1"} : <>
    %193:2 = fork [2] %192 {handshake.bb = 2 : ui32, handshake.name = "fork26"} : <>
    %194 = init %193#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init24"} : <>
    %195:2 = fork [2] %194 {handshake.bb = 2 : ui32, handshake.name = "fork27"} : <>
    %196 = init %195#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init25"} : <>
    %197:2 = fork [2] %196 {handshake.bb = 2 : ui32, handshake.name = "fork28"} : <>
    %198 = init %197#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init26"} : <>
    %199:2 = fork [2] %198 {handshake.bb = 2 : ui32, handshake.name = "fork29"} : <>
    %200 = init %199#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init27"} : <>
    %201:2 = fork [2] %200 {handshake.bb = 2 : ui32, handshake.name = "fork30"} : <>
    %202 = init %201#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init28"} : <>
    %203:2 = fork [2] %202 {handshake.bb = 2 : ui32, handshake.name = "fork31"} : <>
    %204 = init %203#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init29"} : <>
    %205:2 = fork [2] %204 {handshake.bb = 2 : ui32, handshake.name = "fork32"} : <>
    %206 = init %205#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init30"} : <>
    %207:2 = fork [2] %206 {handshake.bb = 2 : ui32, handshake.name = "fork33"} : <>
    %208 = init %207#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init31"} : <>
    %addressResult_64, %dataResult_65, %doneResult = store[%98] %170 %outputs#1 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["load5", 0, false], ["store1", 0, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store1"} : <i10>, <f32>, <>, <i10>, <f32>, <>
    %209 = addi %87#2, %96 {handshake.bb = 2 : ui32, handshake.name = "addi0"} : <i32>
    %210 = buffer %209, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer42"} : <i32>
    %211 = br %210 {handshake.bb = 2 : ui32, handshake.name = "br5"} : <i32>
    %212 = br %85 {handshake.bb = 2 : ui32, handshake.name = "br6"} : <i32>
    %213 = br %91#1 {handshake.bb = 2 : ui32, handshake.name = "br7"} : <>
    %result_66, %index_67 = control_merge [%falseResult_7]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>] to <>, <i1>
    sink %index_67 {handshake.name = "sink28"} : <i1>
    %214:3 = fork [3] %result_66 {handshake.bb = 3 : ui32, handshake.name = "fork34"} : <>
    end {handshake.bb = 3 : ui32, handshake.name = "end0"} %memEnd_3, %memEnd_1, %memEnd, %0#2 : <>, <>, <>, <>
  }
}

