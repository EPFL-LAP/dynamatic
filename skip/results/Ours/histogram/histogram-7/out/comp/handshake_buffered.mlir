module {
  handshake.func @histogram(%arg0: memref<1000xi32>, %arg1: memref<1000xf32>, %arg2: memref<1000xf32>, %arg3: !handshake.channel<i32>, %arg4: !handshake.control<>, %arg5: !handshake.control<>, %arg6: !handshake.control<>, %arg7: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["feature", "weight", "hist", "n", "feature_start", "weight_start", "hist_start", "start"], cfg.edges = "[0,1][2,1][1,2,3,cmpi0]", resNames = ["feature_end", "weight_end", "hist_end", "end"]} {
    %0:12 = fork [12] %arg7 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork0"} : <>
    %outputs:2, %memEnd = mem_controller[%arg2 : memref<1000xf32>] %arg6 (%94, %addressResult_56, %addressResult_58, %dataResult_59) %186#2 {connectedBlocks = [2 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "5": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller2"} :    (!handshake.channel<i32>, !handshake.channel<i10>, !handshake.channel<i10>, !handshake.channel<f32>) -> (!handshake.channel<f32>, !handshake.control<>)
    %outputs_0, %memEnd_1 = mem_controller[%arg1 : memref<1000xf32>] %arg5 (%addressResult_40) %186#1 {connectedBlocks = [2 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller3"} :    (!handshake.channel<i10>) -> !handshake.channel<f32>
    %outputs_2, %memEnd_3 = mem_controller[%arg0 : memref<1000xi32>] %arg4 (%addressResult) %186#0 {connectedBlocks = [2 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller4"} :    (!handshake.channel<i10>) -> !handshake.channel<i32>
    %1 = constant %0#1 {handshake.bb = 0 : ui32, handshake.name = "constant10", value = 1000 : i11} : <>, <i11>
    %2:7 = fork [7] %1 {handshake.bb = 0 : ui32, handshake.name = "fork1"} : <i11>
    %3 = extsi %2#0 {handshake.bb = 1 : ui32, handshake.name = "extsi0"} : <i11> to <i32>
    %4 = extsi %2#1 {handshake.bb = 1 : ui32, handshake.name = "extsi1"} : <i11> to <i32>
    %5 = extsi %2#2 {handshake.bb = 1 : ui32, handshake.name = "extsi2"} : <i11> to <i32>
    %6 = extsi %2#3 {handshake.bb = 1 : ui32, handshake.name = "extsi3"} : <i11> to <i32>
    %7 = extsi %2#4 {handshake.bb = 1 : ui32, handshake.name = "extsi4"} : <i11> to <i32>
    %8 = extsi %2#5 {handshake.bb = 1 : ui32, handshake.name = "extsi5"} : <i11> to <i32>
    %9 = extsi %2#6 {handshake.bb = 1 : ui32, handshake.name = "extsi6"} : <i11> to <i32>
    %10 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant11", value = false} : <>, <i1>
    %11 = br %10 {handshake.bb = 0 : ui32, handshake.name = "br2"} : <i1>
    %12 = extsi %11 {handshake.bb = 0 : ui32, handshake.name = "extsi10"} : <i1> to <i32>
    %13 = br %arg3 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "br3"} : <i32>
    %14 = br %0#11 {handshake.bb = 0 : ui32, handshake.name = "br4"} : <>
    %15 = mux %40#0 [%3, %16] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux2"} : <i1>, [<i32>, <i32>] to <i32>
    %16 = buffer %152#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer9"} : <i32>
    %17 = mux %40#1 [%4, %164#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux3"} : <i1>, [<i32>, <i32>] to <i32>
    %18 = mux %40#2 [%5, %165] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux4"} : <i1>, [<i32>, <i32>] to <i32>
    %19 = mux %40#3 [%0#10, %181] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux5"} : <i1>, [<>, <>] to <>
    %20 = mux %40#4 [%6, %161#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux6"} : <i1>, [<i32>, <i32>] to <i32>
    %21 = mux %40#5 [%7, %150#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux7"} : <i1>, [<i32>, <i32>] to <i32>
    %22 = mux %40#6 [%8, %155#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux8"} : <i1>, [<i32>, <i32>] to <i32>
    %23 = mux %40#7 [%9, %158#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux9"} : <i1>, [<i32>, <i32>] to <i32>
    %24 = mux %25 [%0#9, %178#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux10"} : <i1>, [<>, <>] to <>
    %25 = buffer %40#8, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 1 : ui32, handshake.name = "buffer23"} : <i1>
    %26 = mux %27 [%0#8, %170#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux11"} : <i1>, [<>, <>] to <>
    %27 = buffer %40#9, bufferType = FIFO_BREAK_NONE, numSlots = 6 {handshake.bb = 1 : ui32, handshake.name = "buffer24"} : <i1>
    %28 = mux %29 [%0#7, %176#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux12"} : <i1>, [<>, <>] to <>
    %29 = buffer %40#10, bufferType = FIFO_BREAK_NONE, numSlots = 3 {handshake.bb = 1 : ui32, handshake.name = "buffer25"} : <i1>
    %30 = mux %31 [%0#6, %172#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux13"} : <i1>, [<>, <>] to <>
    %31 = buffer %40#11, bufferType = FIFO_BREAK_NONE, numSlots = 5 {handshake.bb = 1 : ui32, handshake.name = "buffer26"} : <i1>
    %32 = mux %33 [%0#5, %168#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux14"} : <i1>, [<>, <>] to <>
    %33 = buffer %40#12, bufferType = FIFO_BREAK_NONE, numSlots = 7 {handshake.bb = 1 : ui32, handshake.name = "buffer27"} : <i1>
    %34 = mux %35 [%0#4, %174#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux15"} : <i1>, [<>, <>] to <>
    %35 = buffer %40#13, bufferType = FIFO_BREAK_NONE, numSlots = 4 {handshake.bb = 1 : ui32, handshake.name = "buffer28"} : <i1>
    %36 = mux %37 [%0#3, %180#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux16"} : <i1>, [<>, <>] to <>
    %37 = buffer %40#14, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer29"} : <i1>
    %38 = init %39 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "init0"} : <i1>
    %39 = buffer %53#18, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 1 : ui32, handshake.name = "buffer30"} : <i1>
    %40:15 = fork [15] %38 {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <i1>
    %41 = mux %50#0 [%12, %183] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i32>, <i32>] to <i32>
    %42 = buffer %41, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer19"} : <i32>
    %43 = buffer %42, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer20"} : <i32>
    %44:2 = fork [2] %43 {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <i32>
    %45 = mux %50#1 [%13, %184] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i1>, [<i32>, <i32>] to <i32>
    %46 = buffer %45, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer21"} : <i32>
    %47 = buffer %46, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer22"} : <i32>
    %48:2 = fork [2] %47 {handshake.bb = 1 : ui32, handshake.name = "fork4"} : <i32>
    %49 = buffer %185, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer56"} : <>
    %result, %index = control_merge [%14, %49]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %50:2 = fork [2] %index {handshake.bb = 1 : ui32, handshake.name = "fork5"} : <i1>
    %51 = cmpi slt, %44#1, %48#1 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i32>
    %52 = buffer %51, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer31"} : <i1>
    %53:19 = fork [19] %52 {handshake.bb = 1 : ui32, handshake.name = "fork6"} : <i1>
    %trueResult, %falseResult = cond_br %53#17, %48#0 {handshake.bb = 1 : ui32, handshake.name = "cond_br1"} : <i1>, <i32>
    sink %falseResult {handshake.name = "sink0"} : <i32>
    %trueResult_4, %falseResult_5 = cond_br %53#16, %44#0 {handshake.bb = 1 : ui32, handshake.name = "cond_br2"} : <i1>, <i32>
    sink %falseResult_5 {handshake.name = "sink1"} : <i32>
    %trueResult_6, %falseResult_7 = cond_br %53#15, %result {handshake.bb = 1 : ui32, handshake.name = "cond_br3"} : <i1>, <>
    %54 = buffer %36, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer18"} : <>
    %trueResult_8, %falseResult_9 = cond_br %55, %54 {handshake.bb = 2 : ui32, handshake.name = "cond_br25"} : <i1>, <>
    %55 = buffer %53#14, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 2 : ui32, handshake.name = "buffer40"} : <i1>
    sink %falseResult_9 {handshake.name = "sink2"} : <>
    %56 = buffer %32, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer16"} : <>
    %trueResult_10, %falseResult_11 = cond_br %57, %56 {handshake.bb = 2 : ui32, handshake.name = "cond_br26"} : <i1>, <>
    %57 = buffer %53#13, bufferType = FIFO_BREAK_NONE, numSlots = 8 {handshake.bb = 2 : ui32, handshake.name = "buffer41"} : <i1>
    sink %falseResult_11 {handshake.name = "sink3"} : <>
    %58 = buffer %26, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer13"} : <>
    %trueResult_12, %falseResult_13 = cond_br %59, %58 {handshake.bb = 2 : ui32, handshake.name = "cond_br27"} : <i1>, <>
    %59 = buffer %53#12, bufferType = FIFO_BREAK_NONE, numSlots = 7 {handshake.bb = 2 : ui32, handshake.name = "buffer42"} : <i1>
    sink %falseResult_13 {handshake.name = "sink4"} : <>
    %60 = buffer %23, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer11"} : <i32>
    %trueResult_14, %falseResult_15 = cond_br %61, %60 {handshake.bb = 2 : ui32, handshake.name = "cond_br28"} : <i1>, <i32>
    %61 = buffer %53#11, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer43"} : <i1>
    sink %falseResult_15 {handshake.name = "sink5"} : <i32>
    %62 = buffer %20, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer7"} : <i32>
    %trueResult_16, %falseResult_17 = cond_br %63, %62 {handshake.bb = 2 : ui32, handshake.name = "cond_br29"} : <i1>, <i32>
    %63 = buffer %53#10, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer44"} : <i1>
    sink %falseResult_17 {handshake.name = "sink6"} : <i32>
    %64 = buffer %34, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer17"} : <>
    %trueResult_18, %falseResult_19 = cond_br %65, %64 {handshake.bb = 2 : ui32, handshake.name = "cond_br30"} : <i1>, <>
    %65 = buffer %53#9, bufferType = FIFO_BREAK_NONE, numSlots = 5 {handshake.bb = 2 : ui32, handshake.name = "buffer45"} : <i1>
    sink %falseResult_19 {handshake.name = "sink7"} : <>
    %66 = buffer %17, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer3"} : <i32>
    %trueResult_20, %falseResult_21 = cond_br %67, %66 {handshake.bb = 2 : ui32, handshake.name = "cond_br31"} : <i1>, <i32>
    %67 = buffer %53#8, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer46"} : <i1>
    sink %falseResult_21 {handshake.name = "sink8"} : <i32>
    %68 = buffer %28, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer14"} : <>
    %trueResult_22, %falseResult_23 = cond_br %69, %68 {handshake.bb = 2 : ui32, handshake.name = "cond_br32"} : <i1>, <>
    %69 = buffer %53#7, bufferType = FIFO_BREAK_NONE, numSlots = 4 {handshake.bb = 2 : ui32, handshake.name = "buffer47"} : <i1>
    sink %falseResult_23 {handshake.name = "sink9"} : <>
    %70 = buffer %21, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer8"} : <i32>
    %trueResult_24, %falseResult_25 = cond_br %71, %70 {handshake.bb = 2 : ui32, handshake.name = "cond_br33"} : <i1>, <i32>
    %71 = buffer %53#6, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer48"} : <i1>
    sink %falseResult_25 {handshake.name = "sink10"} : <i32>
    %72 = buffer %15, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer2"} : <i32>
    %trueResult_26, %falseResult_27 = cond_br %73, %72 {handshake.bb = 2 : ui32, handshake.name = "cond_br34"} : <i1>, <i32>
    %73 = buffer %53#5, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer49"} : <i1>
    sink %falseResult_27 {handshake.name = "sink11"} : <i32>
    %74 = buffer %19, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer5"} : <>
    %75 = buffer %74, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer6"} : <>
    %trueResult_28, %falseResult_29 = cond_br %76, %75 {handshake.bb = 2 : ui32, handshake.name = "cond_br35"} : <i1>, <>
    %76 = buffer %53#4, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer50"} : <i1>
    sink %falseResult_29 {handshake.name = "sink12"} : <>
    %77 = buffer %24, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer12"} : <>
    %trueResult_30, %falseResult_31 = cond_br %78, %77 {handshake.bb = 2 : ui32, handshake.name = "cond_br36"} : <i1>, <>
    %78 = buffer %53#3, bufferType = FIFO_BREAK_NONE, numSlots = 3 {handshake.bb = 2 : ui32, handshake.name = "buffer51"} : <i1>
    sink %falseResult_31 {handshake.name = "sink13"} : <>
    %79 = buffer %22, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer10"} : <i32>
    %trueResult_32, %falseResult_33 = cond_br %80, %79 {handshake.bb = 2 : ui32, handshake.name = "cond_br37"} : <i1>, <i32>
    %80 = buffer %53#2, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer52"} : <i1>
    sink %falseResult_33 {handshake.name = "sink14"} : <i32>
    %81 = buffer %18, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer4"} : <i32>
    %trueResult_34, %falseResult_35 = cond_br %82, %81 {handshake.bb = 2 : ui32, handshake.name = "cond_br38"} : <i1>, <i32>
    %82 = buffer %53#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer53"} : <i1>
    sink %falseResult_35 {handshake.name = "sink15"} : <i32>
    %83 = buffer %30, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer15"} : <>
    %trueResult_36, %falseResult_37 = cond_br %84, %83 {handshake.bb = 2 : ui32, handshake.name = "cond_br39"} : <i1>, <>
    %84 = buffer %53#0, bufferType = FIFO_BREAK_NONE, numSlots = 6 {handshake.bb = 2 : ui32, handshake.name = "buffer54"} : <i1>
    sink %falseResult_37 {handshake.name = "sink16"} : <>
    %85 = merge %trueResult {handshake.bb = 2 : ui32, handshake.name = "merge0"} : <i32>
    %86 = merge %trueResult_4 {handshake.bb = 2 : ui32, handshake.name = "merge1"} : <i32>
    %87 = buffer %86, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer32"} : <i32>
    %88:3 = fork [3] %87 {handshake.bb = 2 : ui32, handshake.name = "fork7"} : <i32>
    %89 = trunci %90 {handshake.bb = 2 : ui32, handshake.name = "trunci0"} : <i32> to <i10>
    %90 = buffer %88#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer55"} : <i32>
    %91 = trunci %88#1 {handshake.bb = 2 : ui32, handshake.name = "trunci1"} : <i32> to <i10>
    %result_38, %index_39 = control_merge [%trueResult_6]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>] to <>, <i1>
    sink %index_39 {handshake.name = "sink17"} : <i1>
    %92:2 = fork [2] %result_38 {handshake.bb = 2 : ui32, handshake.name = "fork8"} : <>
    %93 = constant %92#0 {handshake.bb = 2 : ui32, handshake.name = "constant12", value = 1 : i2} : <>, <i2>
    %94 = extsi %93 {handshake.bb = 2 : ui32, handshake.name = "extsi8"} : <i2> to <i32>
    %95 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %96 = constant %95 {handshake.bb = 2 : ui32, handshake.name = "constant13", value = 1 : i2} : <>, <i2>
    %97 = extsi %96 {handshake.bb = 2 : ui32, handshake.name = "extsi9"} : <i2> to <i32>
    %addressResult, %dataResult = load[%91] %outputs_2 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i10>, <i32>, <i10>, <i32>
    %98:4 = fork [4] %dataResult {handshake.bb = 2 : ui32, handshake.name = "fork9"} : <i32>
    %99 = trunci %100 {handshake.bb = 2 : ui32, handshake.name = "trunci2"} : <i32> to <i10>
    %100 = buffer %98#0, bufferType = FIFO_BREAK_NONE, numSlots = 7 {handshake.bb = 2 : ui32, handshake.name = "buffer57"} : <i32>
    %addressResult_40, %dataResult_41 = load[%89] %outputs_0 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i10>, <f32>, <i10>, <f32>
    %101 = gate %98#1, %trueResult_28 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "gate0"} : <i32>, !handshake.control<> to <i32>
    %102:7 = fork [7] %101 {handshake.bb = 2 : ui32, handshake.name = "fork10"} : <i32>
    %103 = cmpi ne, %102#6, %trueResult_24 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi1"} : <i32>
    %104:2 = fork [2] %103 {handshake.bb = 2 : ui32, handshake.name = "fork11"} : <i1>
    %105 = cmpi ne, %102#5, %trueResult_26 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi2"} : <i32>
    %106:2 = fork [2] %105 {handshake.bb = 2 : ui32, handshake.name = "fork12"} : <i1>
    %107 = cmpi ne, %102#4, %trueResult_32 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi3"} : <i32>
    %108:2 = fork [2] %107 {handshake.bb = 2 : ui32, handshake.name = "fork13"} : <i1>
    %109 = cmpi ne, %102#3, %trueResult_14 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi4"} : <i32>
    %110:2 = fork [2] %109 {handshake.bb = 2 : ui32, handshake.name = "fork14"} : <i1>
    %111 = cmpi ne, %102#2, %trueResult_16 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi5"} : <i32>
    %112:2 = fork [2] %111 {handshake.bb = 2 : ui32, handshake.name = "fork15"} : <i1>
    %113 = cmpi ne, %102#1, %trueResult_20 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi6"} : <i32>
    %114:2 = fork [2] %113 {handshake.bb = 2 : ui32, handshake.name = "fork16"} : <i1>
    %115 = cmpi ne, %102#0, %trueResult_34 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi7"} : <i32>
    %116:2 = fork [2] %115 {handshake.bb = 2 : ui32, handshake.name = "fork17"} : <i1>
    %trueResult_42, %falseResult_43 = cond_br %117, %trueResult_10 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br18"} : <i1>, <>
    %117 = buffer %104#1, bufferType = FIFO_BREAK_NONE, numSlots = 7 {handshake.bb = 2 : ui32, handshake.name = "buffer66"} : <i1>
    sink %trueResult_42 {handshake.name = "sink18"} : <>
    %trueResult_44, %falseResult_45 = cond_br %118, %trueResult_12 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br19"} : <i1>, <>
    %118 = buffer %106#1, bufferType = FIFO_BREAK_NONE, numSlots = 6 {handshake.bb = 2 : ui32, handshake.name = "buffer67"} : <i1>
    sink %trueResult_44 {handshake.name = "sink19"} : <>
    %trueResult_46, %falseResult_47 = cond_br %119, %trueResult_36 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br20"} : <i1>, <>
    %119 = buffer %108#1, bufferType = FIFO_BREAK_NONE, numSlots = 5 {handshake.bb = 2 : ui32, handshake.name = "buffer68"} : <i1>
    sink %trueResult_46 {handshake.name = "sink20"} : <>
    %trueResult_48, %falseResult_49 = cond_br %120, %trueResult_18 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br21"} : <i1>, <>
    %120 = buffer %110#1, bufferType = FIFO_BREAK_NONE, numSlots = 4 {handshake.bb = 2 : ui32, handshake.name = "buffer69"} : <i1>
    sink %trueResult_48 {handshake.name = "sink21"} : <>
    %trueResult_50, %falseResult_51 = cond_br %121, %trueResult_22 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br22"} : <i1>, <>
    %121 = buffer %112#1, bufferType = FIFO_BREAK_NONE, numSlots = 3 {handshake.bb = 2 : ui32, handshake.name = "buffer70"} : <i1>
    sink %trueResult_50 {handshake.name = "sink22"} : <>
    %trueResult_52, %falseResult_53 = cond_br %122, %trueResult_30 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br23"} : <i1>, <>
    %122 = buffer %114#1, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 2 : ui32, handshake.name = "buffer71"} : <i1>
    sink %trueResult_52 {handshake.name = "sink23"} : <>
    %trueResult_54, %falseResult_55 = cond_br %123, %trueResult_8 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br24"} : <i1>, <>
    %123 = buffer %116#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer72"} : <i1>
    sink %trueResult_54 {handshake.name = "sink24"} : <>
    %124 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %125 = mux %104#0 [%falseResult_43, %124] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux17"} : <i1>, [<>, <>] to <>
    %126 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source2"} : <>
    %127 = mux %106#0 [%falseResult_45, %126] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux18"} : <i1>, [<>, <>] to <>
    %128 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source3"} : <>
    %129 = mux %108#0 [%falseResult_47, %128] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux19"} : <i1>, [<>, <>] to <>
    %130 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source4"} : <>
    %131 = mux %110#0 [%falseResult_49, %130] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux20"} : <i1>, [<>, <>] to <>
    %132 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source5"} : <>
    %133 = mux %112#0 [%falseResult_51, %132] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux21"} : <i1>, [<>, <>] to <>
    %134 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source6"} : <>
    %135 = mux %114#0 [%falseResult_53, %134] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux22"} : <i1>, [<>, <>] to <>
    %136 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source7"} : <>
    %137 = mux %116#0 [%falseResult_55, %136] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux23"} : <i1>, [<>, <>] to <>
    %138 = buffer %125, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer33"} : <>
    %139 = buffer %127, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer34"} : <>
    %140 = buffer %129, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer35"} : <>
    %141 = buffer %131, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer36"} : <>
    %142 = buffer %133, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer37"} : <>
    %143 = buffer %135, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer38"} : <>
    %144 = buffer %137, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer39"} : <>
    %145 = join %138, %139, %140, %141, %142, %143, %144 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "join0"} : <>
    %146 = gate %98#2, %145 {handshake.bb = 2 : ui32, handshake.name = "gate1"} : <i32>, !handshake.control<> to <i32>
    %147 = trunci %146 {handshake.bb = 2 : ui32, handshake.name = "trunci3"} : <i32> to <i10>
    %addressResult_56, %dataResult_57 = load[%147] %outputs#0 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["store1", 0, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load5"} : <i10>, <f32>, <i10>, <f32>
    %148 = addf %dataResult_57, %dataResult_41 {fastmath = #arith.fastmath<none>, fpu_impl = #handshake<fpu_impl flopoco>, handshake.bb = 2 : ui32, handshake.name = "addf0", internal_delay = "2_922000"} : <f32>
    %149 = buffer %98#3, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer0"} : <i32>
    %150:2 = fork [2] %149 {handshake.bb = 2 : ui32, handshake.name = "fork18"} : <i32>
    %151 = init %150#0 {handshake.bb = 2 : ui32, handshake.name = "init15"} : <i32>
    %152:2 = fork [2] %151 {handshake.bb = 2 : ui32, handshake.name = "fork19"} : <i32>
    %153 = init %154 {handshake.bb = 2 : ui32, handshake.name = "init16"} : <i32>
    %154 = buffer %152#0, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 2 : ui32, handshake.name = "buffer83"} : <i32>
    %155:2 = fork [2] %153 {handshake.bb = 2 : ui32, handshake.name = "fork20"} : <i32>
    %156 = init %157 {handshake.bb = 2 : ui32, handshake.name = "init17"} : <i32>
    %157 = buffer %155#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer84"} : <i32>
    %158:2 = fork [2] %156 {handshake.bb = 2 : ui32, handshake.name = "fork21"} : <i32>
    %159 = init %160 {handshake.bb = 2 : ui32, handshake.name = "init18"} : <i32>
    %160 = buffer %158#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer85"} : <i32>
    %161:2 = fork [2] %159 {handshake.bb = 2 : ui32, handshake.name = "fork22"} : <i32>
    %162 = init %163 {handshake.bb = 2 : ui32, handshake.name = "init19"} : <i32>
    %163 = buffer %161#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer86"} : <i32>
    %164:2 = fork [2] %162 {handshake.bb = 2 : ui32, handshake.name = "fork23"} : <i32>
    %165 = init %166 {handshake.bb = 2 : ui32, handshake.name = "init20"} : <i32>
    %166 = buffer %164#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer87"} : <i32>
    %167 = buffer %doneResult, bufferType = FIFO_BREAK_NONE, numSlots = 1 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "buffer1"} : <>
    %168:2 = fork [2] %167 {handshake.bb = 2 : ui32, handshake.name = "fork24"} : <>
    %169 = init %168#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init21"} : <>
    %170:2 = fork [2] %169 {handshake.bb = 2 : ui32, handshake.name = "fork25"} : <>
    %171 = init %170#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init22"} : <>
    %172:2 = fork [2] %171 {handshake.bb = 2 : ui32, handshake.name = "fork26"} : <>
    %173 = init %172#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init23"} : <>
    %174:2 = fork [2] %173 {handshake.bb = 2 : ui32, handshake.name = "fork27"} : <>
    %175 = init %174#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init24"} : <>
    %176:2 = fork [2] %175 {handshake.bb = 2 : ui32, handshake.name = "fork28"} : <>
    %177 = init %176#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init25"} : <>
    %178:2 = fork [2] %177 {handshake.bb = 2 : ui32, handshake.name = "fork29"} : <>
    %179 = init %178#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init26"} : <>
    %180:2 = fork [2] %179 {handshake.bb = 2 : ui32, handshake.name = "fork30"} : <>
    %181 = init %180#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init27"} : <>
    %addressResult_58, %dataResult_59, %doneResult = store[%99] %148 %outputs#1 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["load5", 0, false], ["store1", 0, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store1"} : <i10>, <f32>, <>, <i10>, <f32>, <>
    %182 = addi %88#2, %97 {handshake.bb = 2 : ui32, handshake.name = "addi0"} : <i32>
    %183 = br %182 {handshake.bb = 2 : ui32, handshake.name = "br5"} : <i32>
    %184 = br %85 {handshake.bb = 2 : ui32, handshake.name = "br6"} : <i32>
    %185 = br %92#1 {handshake.bb = 2 : ui32, handshake.name = "br7"} : <>
    %result_60, %index_61 = control_merge [%falseResult_7]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>] to <>, <i1>
    sink %index_61 {handshake.name = "sink25"} : <i1>
    %186:3 = fork [3] %result_60 {handshake.bb = 3 : ui32, handshake.name = "fork31"} : <>
    end {handshake.bb = 3 : ui32, handshake.name = "end0"} %memEnd_3, %memEnd_1, %memEnd, %0#2 : <>, <>, <>, <>
  }
}

