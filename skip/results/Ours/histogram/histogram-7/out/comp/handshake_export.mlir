module {
  handshake.func @histogram(%arg0: memref<1000xi32>, %arg1: memref<1000xf32>, %arg2: memref<1000xf32>, %arg3: !handshake.channel<i32>, %arg4: !handshake.control<>, %arg5: !handshake.control<>, %arg6: !handshake.control<>, %arg7: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["feature", "weight", "hist", "n", "feature_start", "weight_start", "hist_start", "start"], cfg.edges = "[0,1][2,1][1,2,3,cmpi0]", resNames = ["feature_end", "weight_end", "hist_end", "end"]} {
    %0:12 = fork [12] %arg7 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork0"} : <>
    %outputs:2, %memEnd = mem_controller[%arg2 : memref<1000xf32>] %arg6 (%89, %addressResult_54, %addressResult_56, %dataResult_57) %178#2 {connectedBlocks = [2 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "5": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller2"} :    (!handshake.channel<i32>, !handshake.channel<i10>, !handshake.channel<i10>, !handshake.channel<f32>) -> (!handshake.channel<f32>, !handshake.control<>)
    %outputs_0, %memEnd_1 = mem_controller[%arg1 : memref<1000xf32>] %arg5 (%addressResult_38) %178#1 {connectedBlocks = [2 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller3"} :    (!handshake.channel<i10>) -> !handshake.channel<f32>
    %outputs_2, %memEnd_3 = mem_controller[%arg0 : memref<1000xi32>] %arg4 (%addressResult) %178#0 {connectedBlocks = [2 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller4"} :    (!handshake.channel<i10>) -> !handshake.channel<i32>
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
    %11 = extsi %10 {handshake.bb = 0 : ui32, handshake.name = "extsi10"} : <i1> to <i32>
    %12 = mux %37#0 [%3, %13] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux2"} : <i1>, [<i32>, <i32>] to <i32>
    %13 = buffer %147#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer9"} : <i32>
    %14 = mux %37#1 [%4, %159#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux3"} : <i1>, [<i32>, <i32>] to <i32>
    %15 = mux %37#2 [%5, %160] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux4"} : <i1>, [<i32>, <i32>] to <i32>
    %16 = mux %37#3 [%0#10, %176] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux5"} : <i1>, [<>, <>] to <>
    %17 = mux %37#4 [%6, %156#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux6"} : <i1>, [<i32>, <i32>] to <i32>
    %18 = mux %37#5 [%7, %145#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux7"} : <i1>, [<i32>, <i32>] to <i32>
    %19 = mux %37#6 [%8, %150#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux8"} : <i1>, [<i32>, <i32>] to <i32>
    %20 = mux %37#7 [%9, %153#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux9"} : <i1>, [<i32>, <i32>] to <i32>
    %21 = mux %22 [%0#9, %173#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux10"} : <i1>, [<>, <>] to <>
    %22 = buffer %37#8, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 1 : ui32, handshake.name = "buffer23"} : <i1>
    %23 = mux %24 [%0#8, %165#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux11"} : <i1>, [<>, <>] to <>
    %24 = buffer %37#9, bufferType = FIFO_BREAK_NONE, numSlots = 6 {handshake.bb = 1 : ui32, handshake.name = "buffer24"} : <i1>
    %25 = mux %26 [%0#7, %171#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux12"} : <i1>, [<>, <>] to <>
    %26 = buffer %37#10, bufferType = FIFO_BREAK_NONE, numSlots = 3 {handshake.bb = 1 : ui32, handshake.name = "buffer25"} : <i1>
    %27 = mux %28 [%0#6, %167#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux13"} : <i1>, [<>, <>] to <>
    %28 = buffer %37#11, bufferType = FIFO_BREAK_NONE, numSlots = 5 {handshake.bb = 1 : ui32, handshake.name = "buffer26"} : <i1>
    %29 = mux %30 [%0#5, %163#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux14"} : <i1>, [<>, <>] to <>
    %30 = buffer %37#12, bufferType = FIFO_BREAK_NONE, numSlots = 7 {handshake.bb = 1 : ui32, handshake.name = "buffer27"} : <i1>
    %31 = mux %32 [%0#4, %169#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux15"} : <i1>, [<>, <>] to <>
    %32 = buffer %37#13, bufferType = FIFO_BREAK_NONE, numSlots = 4 {handshake.bb = 1 : ui32, handshake.name = "buffer28"} : <i1>
    %33 = mux %34 [%0#3, %175#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux16"} : <i1>, [<>, <>] to <>
    %34 = buffer %37#14, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer29"} : <i1>
    %35 = init %36 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "init0"} : <i1>
    %36 = buffer %50#18, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 1 : ui32, handshake.name = "buffer30"} : <i1>
    %37:15 = fork [15] %35 {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <i1>
    %38 = mux %47#0 [%11, %177] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i32>, <i32>] to <i32>
    %39 = buffer %38, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer19"} : <i32>
    %40 = buffer %39, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer20"} : <i32>
    %41:2 = fork [2] %40 {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <i32>
    %42 = mux %47#1 [%arg3, %trueResult] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i1>, [<i32>, <i32>] to <i32>
    %43 = buffer %42, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer21"} : <i32>
    %44 = buffer %43, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer22"} : <i32>
    %45:2 = fork [2] %44 {handshake.bb = 1 : ui32, handshake.name = "fork4"} : <i32>
    %46 = buffer %87#1, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer56"} : <>
    %result, %index = control_merge [%0#11, %46]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %47:2 = fork [2] %index {handshake.bb = 1 : ui32, handshake.name = "fork5"} : <i1>
    %48 = cmpi slt, %41#1, %45#1 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i32>
    %49 = buffer %48, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer31"} : <i1>
    %50:19 = fork [19] %49 {handshake.bb = 1 : ui32, handshake.name = "fork6"} : <i1>
    %trueResult, %falseResult = cond_br %50#17, %45#0 {handshake.bb = 1 : ui32, handshake.name = "cond_br1"} : <i1>, <i32>
    sink %falseResult {handshake.name = "sink0"} : <i32>
    %trueResult_4, %falseResult_5 = cond_br %50#16, %41#0 {handshake.bb = 1 : ui32, handshake.name = "cond_br2"} : <i1>, <i32>
    sink %falseResult_5 {handshake.name = "sink1"} : <i32>
    %trueResult_6, %falseResult_7 = cond_br %50#15, %result {handshake.bb = 1 : ui32, handshake.name = "cond_br3"} : <i1>, <>
    %51 = buffer %33, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer18"} : <>
    %trueResult_8, %falseResult_9 = cond_br %52, %51 {handshake.bb = 2 : ui32, handshake.name = "cond_br25"} : <i1>, <>
    %52 = buffer %50#14, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 2 : ui32, handshake.name = "buffer40"} : <i1>
    sink %falseResult_9 {handshake.name = "sink2"} : <>
    %53 = buffer %29, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer16"} : <>
    %trueResult_10, %falseResult_11 = cond_br %54, %53 {handshake.bb = 2 : ui32, handshake.name = "cond_br26"} : <i1>, <>
    %54 = buffer %50#13, bufferType = FIFO_BREAK_NONE, numSlots = 8 {handshake.bb = 2 : ui32, handshake.name = "buffer41"} : <i1>
    sink %falseResult_11 {handshake.name = "sink3"} : <>
    %55 = buffer %23, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer13"} : <>
    %trueResult_12, %falseResult_13 = cond_br %56, %55 {handshake.bb = 2 : ui32, handshake.name = "cond_br27"} : <i1>, <>
    %56 = buffer %50#12, bufferType = FIFO_BREAK_NONE, numSlots = 7 {handshake.bb = 2 : ui32, handshake.name = "buffer42"} : <i1>
    sink %falseResult_13 {handshake.name = "sink4"} : <>
    %57 = buffer %20, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer11"} : <i32>
    %trueResult_14, %falseResult_15 = cond_br %58, %57 {handshake.bb = 2 : ui32, handshake.name = "cond_br28"} : <i1>, <i32>
    %58 = buffer %50#11, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer43"} : <i1>
    sink %falseResult_15 {handshake.name = "sink5"} : <i32>
    %59 = buffer %17, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer7"} : <i32>
    %trueResult_16, %falseResult_17 = cond_br %60, %59 {handshake.bb = 2 : ui32, handshake.name = "cond_br29"} : <i1>, <i32>
    %60 = buffer %50#10, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer44"} : <i1>
    sink %falseResult_17 {handshake.name = "sink6"} : <i32>
    %61 = buffer %31, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer17"} : <>
    %trueResult_18, %falseResult_19 = cond_br %62, %61 {handshake.bb = 2 : ui32, handshake.name = "cond_br30"} : <i1>, <>
    %62 = buffer %50#9, bufferType = FIFO_BREAK_NONE, numSlots = 5 {handshake.bb = 2 : ui32, handshake.name = "buffer45"} : <i1>
    sink %falseResult_19 {handshake.name = "sink7"} : <>
    %63 = buffer %14, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer3"} : <i32>
    %trueResult_20, %falseResult_21 = cond_br %64, %63 {handshake.bb = 2 : ui32, handshake.name = "cond_br31"} : <i1>, <i32>
    %64 = buffer %50#8, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer46"} : <i1>
    sink %falseResult_21 {handshake.name = "sink8"} : <i32>
    %65 = buffer %25, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer14"} : <>
    %trueResult_22, %falseResult_23 = cond_br %66, %65 {handshake.bb = 2 : ui32, handshake.name = "cond_br32"} : <i1>, <>
    %66 = buffer %50#7, bufferType = FIFO_BREAK_NONE, numSlots = 4 {handshake.bb = 2 : ui32, handshake.name = "buffer47"} : <i1>
    sink %falseResult_23 {handshake.name = "sink9"} : <>
    %67 = buffer %18, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer8"} : <i32>
    %trueResult_24, %falseResult_25 = cond_br %68, %67 {handshake.bb = 2 : ui32, handshake.name = "cond_br33"} : <i1>, <i32>
    %68 = buffer %50#6, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer48"} : <i1>
    sink %falseResult_25 {handshake.name = "sink10"} : <i32>
    %69 = buffer %12, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer2"} : <i32>
    %trueResult_26, %falseResult_27 = cond_br %70, %69 {handshake.bb = 2 : ui32, handshake.name = "cond_br34"} : <i1>, <i32>
    %70 = buffer %50#5, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer49"} : <i1>
    sink %falseResult_27 {handshake.name = "sink11"} : <i32>
    %71 = buffer %16, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer5"} : <>
    %72 = buffer %71, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer6"} : <>
    %trueResult_28, %falseResult_29 = cond_br %73, %72 {handshake.bb = 2 : ui32, handshake.name = "cond_br35"} : <i1>, <>
    %73 = buffer %50#4, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer50"} : <i1>
    sink %falseResult_29 {handshake.name = "sink12"} : <>
    %74 = buffer %21, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer12"} : <>
    %trueResult_30, %falseResult_31 = cond_br %75, %74 {handshake.bb = 2 : ui32, handshake.name = "cond_br36"} : <i1>, <>
    %75 = buffer %50#3, bufferType = FIFO_BREAK_NONE, numSlots = 3 {handshake.bb = 2 : ui32, handshake.name = "buffer51"} : <i1>
    sink %falseResult_31 {handshake.name = "sink13"} : <>
    %76 = buffer %19, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer10"} : <i32>
    %trueResult_32, %falseResult_33 = cond_br %77, %76 {handshake.bb = 2 : ui32, handshake.name = "cond_br37"} : <i1>, <i32>
    %77 = buffer %50#2, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer52"} : <i1>
    sink %falseResult_33 {handshake.name = "sink14"} : <i32>
    %78 = buffer %15, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer4"} : <i32>
    %trueResult_34, %falseResult_35 = cond_br %79, %78 {handshake.bb = 2 : ui32, handshake.name = "cond_br38"} : <i1>, <i32>
    %79 = buffer %50#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer53"} : <i1>
    sink %falseResult_35 {handshake.name = "sink15"} : <i32>
    %80 = buffer %27, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer15"} : <>
    %trueResult_36, %falseResult_37 = cond_br %81, %80 {handshake.bb = 2 : ui32, handshake.name = "cond_br39"} : <i1>, <>
    %81 = buffer %50#0, bufferType = FIFO_BREAK_NONE, numSlots = 6 {handshake.bb = 2 : ui32, handshake.name = "buffer54"} : <i1>
    sink %falseResult_37 {handshake.name = "sink16"} : <>
    %82 = buffer %trueResult_4, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer32"} : <i32>
    %83:3 = fork [3] %82 {handshake.bb = 2 : ui32, handshake.name = "fork7"} : <i32>
    %84 = trunci %85 {handshake.bb = 2 : ui32, handshake.name = "trunci0"} : <i32> to <i10>
    %85 = buffer %83#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer55"} : <i32>
    %86 = trunci %83#1 {handshake.bb = 2 : ui32, handshake.name = "trunci1"} : <i32> to <i10>
    %87:2 = fork [2] %trueResult_6 {handshake.bb = 2 : ui32, handshake.name = "fork8"} : <>
    %88 = constant %87#0 {handshake.bb = 2 : ui32, handshake.name = "constant12", value = 1 : i2} : <>, <i2>
    %89 = extsi %88 {handshake.bb = 2 : ui32, handshake.name = "extsi8"} : <i2> to <i32>
    %90 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %91 = constant %90 {handshake.bb = 2 : ui32, handshake.name = "constant13", value = 1 : i2} : <>, <i2>
    %92 = extsi %91 {handshake.bb = 2 : ui32, handshake.name = "extsi9"} : <i2> to <i32>
    %addressResult, %dataResult = load[%86] %outputs_2 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i10>, <i32>, <i10>, <i32>
    %93:4 = fork [4] %dataResult {handshake.bb = 2 : ui32, handshake.name = "fork9"} : <i32>
    %94 = trunci %95 {handshake.bb = 2 : ui32, handshake.name = "trunci2"} : <i32> to <i10>
    %95 = buffer %93#0, bufferType = FIFO_BREAK_NONE, numSlots = 7 {handshake.bb = 2 : ui32, handshake.name = "buffer57"} : <i32>
    %addressResult_38, %dataResult_39 = load[%84] %outputs_0 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i10>, <f32>, <i10>, <f32>
    %96 = gate %93#1, %trueResult_28 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "gate0"} : <i32>, !handshake.control<> to <i32>
    %97:7 = fork [7] %96 {handshake.bb = 2 : ui32, handshake.name = "fork10"} : <i32>
    %98 = cmpi ne, %97#6, %trueResult_24 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi1"} : <i32>
    %99:2 = fork [2] %98 {handshake.bb = 2 : ui32, handshake.name = "fork11"} : <i1>
    %100 = cmpi ne, %97#5, %trueResult_26 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi2"} : <i32>
    %101:2 = fork [2] %100 {handshake.bb = 2 : ui32, handshake.name = "fork12"} : <i1>
    %102 = cmpi ne, %97#4, %trueResult_32 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi3"} : <i32>
    %103:2 = fork [2] %102 {handshake.bb = 2 : ui32, handshake.name = "fork13"} : <i1>
    %104 = cmpi ne, %97#3, %trueResult_14 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi4"} : <i32>
    %105:2 = fork [2] %104 {handshake.bb = 2 : ui32, handshake.name = "fork14"} : <i1>
    %106 = cmpi ne, %97#2, %trueResult_16 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi5"} : <i32>
    %107:2 = fork [2] %106 {handshake.bb = 2 : ui32, handshake.name = "fork15"} : <i1>
    %108 = cmpi ne, %97#1, %trueResult_20 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi6"} : <i32>
    %109:2 = fork [2] %108 {handshake.bb = 2 : ui32, handshake.name = "fork16"} : <i1>
    %110 = cmpi ne, %97#0, %trueResult_34 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi7"} : <i32>
    %111:2 = fork [2] %110 {handshake.bb = 2 : ui32, handshake.name = "fork17"} : <i1>
    %trueResult_40, %falseResult_41 = cond_br %112, %trueResult_10 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br18"} : <i1>, <>
    %112 = buffer %99#1, bufferType = FIFO_BREAK_NONE, numSlots = 7 {handshake.bb = 2 : ui32, handshake.name = "buffer66"} : <i1>
    sink %trueResult_40 {handshake.name = "sink18"} : <>
    %trueResult_42, %falseResult_43 = cond_br %113, %trueResult_12 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br19"} : <i1>, <>
    %113 = buffer %101#1, bufferType = FIFO_BREAK_NONE, numSlots = 6 {handshake.bb = 2 : ui32, handshake.name = "buffer67"} : <i1>
    sink %trueResult_42 {handshake.name = "sink19"} : <>
    %trueResult_44, %falseResult_45 = cond_br %114, %trueResult_36 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br20"} : <i1>, <>
    %114 = buffer %103#1, bufferType = FIFO_BREAK_NONE, numSlots = 5 {handshake.bb = 2 : ui32, handshake.name = "buffer68"} : <i1>
    sink %trueResult_44 {handshake.name = "sink20"} : <>
    %trueResult_46, %falseResult_47 = cond_br %115, %trueResult_18 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br21"} : <i1>, <>
    %115 = buffer %105#1, bufferType = FIFO_BREAK_NONE, numSlots = 4 {handshake.bb = 2 : ui32, handshake.name = "buffer69"} : <i1>
    sink %trueResult_46 {handshake.name = "sink21"} : <>
    %trueResult_48, %falseResult_49 = cond_br %116, %trueResult_22 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br22"} : <i1>, <>
    %116 = buffer %107#1, bufferType = FIFO_BREAK_NONE, numSlots = 3 {handshake.bb = 2 : ui32, handshake.name = "buffer70"} : <i1>
    sink %trueResult_48 {handshake.name = "sink22"} : <>
    %trueResult_50, %falseResult_51 = cond_br %117, %trueResult_30 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br23"} : <i1>, <>
    %117 = buffer %109#1, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 2 : ui32, handshake.name = "buffer71"} : <i1>
    sink %trueResult_50 {handshake.name = "sink23"} : <>
    %trueResult_52, %falseResult_53 = cond_br %118, %trueResult_8 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br24"} : <i1>, <>
    %118 = buffer %111#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer72"} : <i1>
    sink %trueResult_52 {handshake.name = "sink24"} : <>
    %119 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %120 = mux %99#0 [%falseResult_41, %119] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux17"} : <i1>, [<>, <>] to <>
    %121 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source2"} : <>
    %122 = mux %101#0 [%falseResult_43, %121] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux18"} : <i1>, [<>, <>] to <>
    %123 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source3"} : <>
    %124 = mux %103#0 [%falseResult_45, %123] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux19"} : <i1>, [<>, <>] to <>
    %125 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source4"} : <>
    %126 = mux %105#0 [%falseResult_47, %125] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux20"} : <i1>, [<>, <>] to <>
    %127 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source5"} : <>
    %128 = mux %107#0 [%falseResult_49, %127] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux21"} : <i1>, [<>, <>] to <>
    %129 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source6"} : <>
    %130 = mux %109#0 [%falseResult_51, %129] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux22"} : <i1>, [<>, <>] to <>
    %131 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source7"} : <>
    %132 = mux %111#0 [%falseResult_53, %131] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux23"} : <i1>, [<>, <>] to <>
    %133 = buffer %120, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer33"} : <>
    %134 = buffer %122, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer34"} : <>
    %135 = buffer %124, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer35"} : <>
    %136 = buffer %126, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer36"} : <>
    %137 = buffer %128, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer37"} : <>
    %138 = buffer %130, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer38"} : <>
    %139 = buffer %132, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer39"} : <>
    %140 = join %133, %134, %135, %136, %137, %138, %139 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "join0"} : <>
    %141 = gate %93#2, %140 {handshake.bb = 2 : ui32, handshake.name = "gate1"} : <i32>, !handshake.control<> to <i32>
    %142 = trunci %141 {handshake.bb = 2 : ui32, handshake.name = "trunci3"} : <i32> to <i10>
    %addressResult_54, %dataResult_55 = load[%142] %outputs#0 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["store1", 0, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load5"} : <i10>, <f32>, <i10>, <f32>
    %143 = addf %dataResult_55, %dataResult_39 {fastmath = #arith.fastmath<none>, fpu_impl = #handshake<fpu_impl flopoco>, handshake.bb = 2 : ui32, handshake.name = "addf0", internal_delay = "2_922000"} : <f32>
    %144 = buffer %93#3, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer0"} : <i32>
    %145:2 = fork [2] %144 {handshake.bb = 2 : ui32, handshake.name = "fork18"} : <i32>
    %146 = init %145#0 {handshake.bb = 2 : ui32, handshake.name = "init15"} : <i32>
    %147:2 = fork [2] %146 {handshake.bb = 2 : ui32, handshake.name = "fork19"} : <i32>
    %148 = init %149 {handshake.bb = 2 : ui32, handshake.name = "init16"} : <i32>
    %149 = buffer %147#0, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 2 : ui32, handshake.name = "buffer83"} : <i32>
    %150:2 = fork [2] %148 {handshake.bb = 2 : ui32, handshake.name = "fork20"} : <i32>
    %151 = init %152 {handshake.bb = 2 : ui32, handshake.name = "init17"} : <i32>
    %152 = buffer %150#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer84"} : <i32>
    %153:2 = fork [2] %151 {handshake.bb = 2 : ui32, handshake.name = "fork21"} : <i32>
    %154 = init %155 {handshake.bb = 2 : ui32, handshake.name = "init18"} : <i32>
    %155 = buffer %153#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer85"} : <i32>
    %156:2 = fork [2] %154 {handshake.bb = 2 : ui32, handshake.name = "fork22"} : <i32>
    %157 = init %158 {handshake.bb = 2 : ui32, handshake.name = "init19"} : <i32>
    %158 = buffer %156#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer86"} : <i32>
    %159:2 = fork [2] %157 {handshake.bb = 2 : ui32, handshake.name = "fork23"} : <i32>
    %160 = init %161 {handshake.bb = 2 : ui32, handshake.name = "init20"} : <i32>
    %161 = buffer %159#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer87"} : <i32>
    %162 = buffer %doneResult, bufferType = FIFO_BREAK_NONE, numSlots = 1 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "buffer1"} : <>
    %163:2 = fork [2] %162 {handshake.bb = 2 : ui32, handshake.name = "fork24"} : <>
    %164 = init %163#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init21"} : <>
    %165:2 = fork [2] %164 {handshake.bb = 2 : ui32, handshake.name = "fork25"} : <>
    %166 = init %165#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init22"} : <>
    %167:2 = fork [2] %166 {handshake.bb = 2 : ui32, handshake.name = "fork26"} : <>
    %168 = init %167#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init23"} : <>
    %169:2 = fork [2] %168 {handshake.bb = 2 : ui32, handshake.name = "fork27"} : <>
    %170 = init %169#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init24"} : <>
    %171:2 = fork [2] %170 {handshake.bb = 2 : ui32, handshake.name = "fork28"} : <>
    %172 = init %171#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init25"} : <>
    %173:2 = fork [2] %172 {handshake.bb = 2 : ui32, handshake.name = "fork29"} : <>
    %174 = init %173#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init26"} : <>
    %175:2 = fork [2] %174 {handshake.bb = 2 : ui32, handshake.name = "fork30"} : <>
    %176 = init %175#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init27"} : <>
    %addressResult_56, %dataResult_57, %doneResult = store[%94] %143 %outputs#1 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["load5", 0, false], ["store1", 0, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store1"} : <i10>, <f32>, <>, <i10>, <f32>, <>
    %177 = addi %83#2, %92 {handshake.bb = 2 : ui32, handshake.name = "addi0"} : <i32>
    %178:3 = fork [3] %falseResult_7 {handshake.bb = 3 : ui32, handshake.name = "fork31"} : <>
    end {handshake.bb = 3 : ui32, handshake.name = "end0"} %memEnd_3, %memEnd_1, %memEnd, %0#2 : <>, <>, <>, <>
  }
}

