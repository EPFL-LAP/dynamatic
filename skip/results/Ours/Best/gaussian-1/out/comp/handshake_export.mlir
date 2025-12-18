module {
  handshake.func @gaussian(%arg0: memref<20xi32>, %arg1: memref<400xi32>, %arg2: !handshake.control<>, %arg3: !handshake.control<>, %arg4: !handshake.control<>, ...) -> (!handshake.channel<i32>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["c", "a", "c_start", "a_start", "start"], cfg.edges = "[0,1][2,3,5,cmpi2][4,2][1,2][3,3,4,cmpi0][5,1,6,cmpi1]", resNames = ["out0", "c_end", "a_end", "end"]} {
    %0:9 = fork [9] %arg4 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork0"} : <>
    %outputs:3, %memEnd = mem_controller[%arg1 : memref<400xi32>] %arg3 (%125, %addressResult, %addressResult_40, %addressResult_42, %dataResult_43) %234#1 {connectedBlocks = [3 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "5": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "6": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller1"} :    (!handshake.channel<i32>, !handshake.channel<i9>, !handshake.channel<i9>, !handshake.channel<i9>, !handshake.channel<i32>) -> (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>)
    %outputs_0, %memEnd_1 = mem_controller[%arg0 : memref<20xi32>] %arg2 (%addressResult_36) %234#0 {connectedBlocks = [3 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller2"} :    (!handshake.channel<i5>) -> !handshake.channel<i32>
    %1 = constant %0#2 {handshake.bb = 0 : ui32, handshake.name = "constant5", value = 1000 : i11} : <>, <i11>
    %2:2 = fork [2] %1 {handshake.bb = 0 : ui32, handshake.name = "fork1"} : <i11>
    %3 = extsi %2#0 {handshake.bb = 1 : ui32, handshake.name = "extsi0"} : <i11> to <i32>
    %4 = extsi %2#1 {handshake.bb = 1 : ui32, handshake.name = "extsi1"} : <i11> to <i32>
    %5 = constant %0#1 {handshake.bb = 0 : ui32, handshake.name = "constant6", value = false} : <>, <i1>
    %6 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant19", value = 1 : i2} : <>, <i2>
    %7 = extsi %6 {handshake.bb = 0 : ui32, handshake.name = "extsi18"} : <i2> to <i6>
    %8 = extsi %5 {handshake.bb = 0 : ui32, handshake.name = "extsi19"} : <i1> to <i32>
    %9 = mux %16#0 [%0#7, %trueResult_60] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux10"} : <i1>, [<>, <>] to <>
    %10 = mux %16#1 [%3, %trueResult_56] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux11"} : <i1>, [<i32>, <i32>] to <i32>
    %11 = mux %16#2 [%4, %trueResult_58] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux12"} : <i1>, [<i32>, <i32>] to <i32>
    %12 = mux %16#3 [%0#6, %trueResult_66] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux13"} : <i1>, [<>, <>] to <>
    %13 = mux %16#4 [%0#5, %trueResult_64] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux14"} : <i1>, [<>, <>] to <>
    %14 = mux %16#5 [%0#4, %trueResult_62] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux15"} : <i1>, [<>, <>] to <>
    %15 = init %233#7 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "init0"} : <i1>
    %16:6 = fork [6] %15 {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <i1>
    %17 = mux %23#0 [%7, %trueResult_68] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i6>, <i6>] to <i6>
    %18 = buffer %17, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer8"} : <i6>
    %19:2 = fork [2] %18 {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <i6>
    %20 = extsi %19#1 {handshake.bb = 1 : ui32, handshake.name = "extsi20"} : <i6> to <i7>
    %21 = buffer %trueResult_70, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer73"} : <i32>
    %22 = mux %23#1 [%8, %21] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i1>, [<i32>, <i32>] to <i32>
    %result, %index = control_merge [%0#8, %trueResult_72]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %23:2 = fork [2] %index {handshake.bb = 1 : ui32, handshake.name = "fork4"} : <i1>
    %24 = source {handshake.bb = 1 : ui32, handshake.name = "source0"} : <>
    %25 = constant %24 {handshake.bb = 1 : ui32, handshake.name = "constant20", value = 1 : i2} : <>, <i2>
    %26 = extsi %25 {handshake.bb = 1 : ui32, handshake.name = "extsi21"} : <i2> to <i7>
    %27 = addi %20, %26 {handshake.bb = 1 : ui32, handshake.name = "addi2"} : <i7>
    %28 = buffer %27, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer10"} : <i7>
    %29 = buffer %22, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer9"} : <i32>
    %30 = buffer %9, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer2"} : <>
    %31 = mux %45#0 [%30, %88#0] {ftd.phi, handshake.bb = 2 : ui32, handshake.name = "mux16"} : <i1>, [<>, <>] to <>
    %32 = buffer %10, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer3"} : <i32>
    %33 = mux %45#1 [%32, %78#0] {ftd.phi, handshake.bb = 2 : ui32, handshake.name = "mux17"} : <i1>, [<i32>, <i32>] to <i32>
    %34 = buffer %11, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer4"} : <i32>
    %35 = mux %45#2 [%34, %78#1] {ftd.phi, handshake.bb = 2 : ui32, handshake.name = "mux18"} : <i1>, [<i32>, <i32>] to <i32>
    %36 = buffer %12, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer5"} : <>
    %37 = mux %45#3 [%36, %88#1] {ftd.phi, handshake.bb = 2 : ui32, handshake.name = "mux19"} : <i1>, [<>, <>] to <>
    %38 = buffer %13, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer6"} : <>
    %39 = mux %40 [%38, %83#0] {ftd.phi, handshake.bb = 2 : ui32, handshake.name = "mux20"} : <i1>, [<>, <>] to <>
    %40 = buffer %45#4, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer21"} : <i1>
    %41 = buffer %14, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer7"} : <>
    %42 = mux %43 [%41, %83#1] {ftd.phi, handshake.bb = 2 : ui32, handshake.name = "mux21"} : <i1>, [<>, <>] to <>
    %43 = buffer %45#5, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer22"} : <i1>
    %44 = init %62#2 {ftd.imerge, handshake.bb = 2 : ui32, handshake.name = "init6"} : <i1>
    %45:6 = fork [6] %44 {handshake.bb = 2 : ui32, handshake.name = "fork5"} : <i1>
    %46 = mux %52#1 [%28, %219] {handshake.bb = 2 : ui32, handshake.name = "mux2"} : <i1>, [<i7>, <i7>] to <i7>
    %47 = buffer %46, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer25"} : <i7>
    %48:2 = fork [2] %47 {handshake.bb = 2 : ui32, handshake.name = "fork6"} : <i7>
    %49 = trunci %48#0 {handshake.bb = 2 : ui32, handshake.name = "trunci0"} : <i7> to <i6>
    %50 = mux %52#2 [%29, %falseResult_49] {handshake.bb = 2 : ui32, handshake.name = "mux3"} : <i1>, [<i32>, <i32>] to <i32>
    %51 = mux %52#0 [%19#0, %falseResult_51] {handshake.bb = 2 : ui32, handshake.name = "mux4"} : <i1>, [<i6>, <i6>] to <i6>
    %result_2, %index_3 = control_merge [%result, %falseResult_55]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>, <>] to <>, <i1>
    %52:3 = fork [3] %index_3 {handshake.bb = 2 : ui32, handshake.name = "fork7"} : <i1>
    %53 = buffer %result_2, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer30"} : <>
    %54:2 = fork [2] %53 {handshake.bb = 2 : ui32, handshake.name = "fork8"} : <>
    %55 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %56 = constant %55 {handshake.bb = 2 : ui32, handshake.name = "constant21", value = 19 : i6} : <>, <i6>
    %57 = extsi %56 {handshake.bb = 2 : ui32, handshake.name = "extsi22"} : <i6> to <i7>
    %58 = constant %54#0 {handshake.bb = 2 : ui32, handshake.name = "constant22", value = 1 : i2} : <>, <i2>
    %59:2 = fork [2] %58 {handshake.bb = 2 : ui32, handshake.name = "fork9"} : <i2>
    %60 = cmpi ult, %48#1, %57 {handshake.bb = 2 : ui32, handshake.name = "cmpi2"} : <i7>
    %61 = buffer %60, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer31"} : <i1>
    %62:13 = fork [13] %61 {handshake.bb = 2 : ui32, handshake.name = "fork10"} : <i1>
    %trueResult, %falseResult = cond_br %62#12, %59#0 {handshake.bb = 2 : ui32, handshake.name = "cond_br3"} : <i1>, <i2>
    sink %falseResult {handshake.name = "sink0"} : <i2>
    %63 = extsi %trueResult {handshake.bb = 2 : ui32, handshake.name = "extsi17"} : <i2> to <i6>
    %trueResult_4, %falseResult_5 = cond_br %62#11, %59#1 {handshake.bb = 2 : ui32, handshake.name = "cond_br4"} : <i1>, <i2>
    sink %falseResult_5 {handshake.name = "sink1"} : <i2>
    %64 = extsi %trueResult_4 {handshake.bb = 2 : ui32, handshake.name = "extsi23"} : <i2> to <i32>
    %65 = buffer %50, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer26"} : <i32>
    %66 = buffer %65, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer27"} : <i32>
    %trueResult_6, %falseResult_7 = cond_br %62#3, %66 {handshake.bb = 2 : ui32, handshake.name = "cond_br5"} : <i1>, <i32>
    %67 = buffer %51, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer28"} : <i6>
    %68 = buffer %67, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer29"} : <i6>
    %trueResult_8, %falseResult_9 = cond_br %62#1, %68 {handshake.bb = 2 : ui32, handshake.name = "cond_br6"} : <i1>, <i6>
    %trueResult_10, %falseResult_11 = cond_br %62#0, %49 {handshake.bb = 2 : ui32, handshake.name = "cond_br7"} : <i1>, <i6>
    sink %falseResult_11 {handshake.name = "sink2"} : <i6>
    %trueResult_12, %falseResult_13 = cond_br %62#4, %54#1 {handshake.bb = 2 : ui32, handshake.name = "cond_br8"} : <i1>, <>
    %69 = buffer %33, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer13"} : <i32>
    %70 = buffer %69, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer14"} : <i32>
    %trueResult_14, %falseResult_15 = cond_br %62#5, %70 {handshake.bb = 3 : ui32, handshake.name = "cond_br47"} : <i1>, <i32>
    %71 = buffer %31, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer11"} : <>
    %72 = buffer %71, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer12"} : <>
    %trueResult_16, %falseResult_17 = cond_br %62#6, %72 {handshake.bb = 3 : ui32, handshake.name = "cond_br48"} : <i1>, <>
    %73 = buffer %35, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer15"} : <i32>
    %74 = buffer %73, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer16"} : <i32>
    %trueResult_18, %falseResult_19 = cond_br %62#7, %74 {handshake.bb = 3 : ui32, handshake.name = "cond_br49"} : <i1>, <i32>
    %75 = buffer %42, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer23"} : <>
    %76 = buffer %75, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer24"} : <>
    %trueResult_20, %falseResult_21 = cond_br %77, %76 {handshake.bb = 3 : ui32, handshake.name = "cond_br50"} : <i1>, <>
    %77 = buffer %62#8, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer40"} : <i1>
    %trueResult_22, %falseResult_23 = cond_br %212#6, %198 {handshake.bb = 3 : ui32, handshake.name = "cond_br51"} : <i1>, <i32>
    %78:2 = fork [2] %falseResult_23 {handshake.bb = 3 : ui32, handshake.name = "fork11"} : <i32>
    %79:2 = fork [2] %trueResult_22 {handshake.bb = 3 : ui32, handshake.name = "fork12"} : <i32>
    %80 = buffer %37, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer17"} : <>
    %81 = buffer %80, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer18"} : <>
    %trueResult_24, %falseResult_25 = cond_br %62#9, %81 {handshake.bb = 3 : ui32, handshake.name = "cond_br52"} : <i1>, <>
    %trueResult_26, %falseResult_27 = cond_br %82, %201#1 {handshake.bb = 3 : ui32, handshake.name = "cond_br53"} : <i1>, <>
    %82 = buffer %212#5, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer43"} : <i1>
    %83:2 = fork [2] %falseResult_27 {handshake.bb = 3 : ui32, handshake.name = "fork13"} : <>
    %84:2 = fork [2] %trueResult_26 {handshake.bb = 3 : ui32, handshake.name = "fork14"} : <>
    %85 = buffer %39, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer19"} : <>
    %86 = buffer %85, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer20"} : <>
    %trueResult_28, %falseResult_29 = cond_br %87, %86 {handshake.bb = 3 : ui32, handshake.name = "cond_br54"} : <i1>, <>
    %87 = buffer %62#10, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer44"} : <i1>
    %trueResult_30, %falseResult_31 = cond_br %212#4, %202 {handshake.bb = 3 : ui32, handshake.name = "cond_br55"} : <i1>, <>
    %88:2 = fork [2] %falseResult_31 {handshake.bb = 3 : ui32, handshake.name = "fork15"} : <>
    %89:2 = fork [2] %trueResult_30 {handshake.bb = 3 : ui32, handshake.name = "fork16"} : <>
    %90 = mux %99#0 [%trueResult_16, %89#1] {ftd.phi, handshake.bb = 3 : ui32, handshake.name = "mux22"} : <i1>, [<>, <>] to <>
    %91 = mux %99#1 [%trueResult_14, %79#1] {ftd.phi, handshake.bb = 3 : ui32, handshake.name = "mux23"} : <i1>, [<i32>, <i32>] to <i32>
    %92 = mux %99#2 [%trueResult_18, %79#0] {ftd.phi, handshake.bb = 3 : ui32, handshake.name = "mux24"} : <i1>, [<i32>, <i32>] to <i32>
    %93 = mux %99#3 [%trueResult_24, %89#0] {ftd.phi, handshake.bb = 3 : ui32, handshake.name = "mux25"} : <i1>, [<>, <>] to <>
    %94 = mux %95 [%trueResult_28, %84#1] {ftd.phi, handshake.bb = 3 : ui32, handshake.name = "mux26"} : <i1>, [<>, <>] to <>
    %95 = buffer %99#4, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer52"} : <i1>
    %96 = mux %97 [%trueResult_20, %84#0] {ftd.phi, handshake.bb = 3 : ui32, handshake.name = "mux27"} : <i1>, [<>, <>] to <>
    %97 = buffer %99#5, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer53"} : <i1>
    %98 = init %212#3 {ftd.imerge, handshake.bb = 3 : ui32, handshake.name = "init12"} : <i1>
    %99:6 = fork [6] %98 {handshake.bb = 3 : ui32, handshake.name = "fork17"} : <i1>
    %100 = mux %121#2 [%63, %trueResult_44] {handshake.bb = 3 : ui32, handshake.name = "mux5"} : <i1>, [<i6>, <i6>] to <i6>
    %101 = buffer %100, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer38"} : <i6>
    %102 = extsi %101 {handshake.bb = 3 : ui32, handshake.name = "extsi24"} : <i6> to <i7>
    %103 = mux %121#3 [%64, %trueResult_46] {handshake.bb = 3 : ui32, handshake.name = "mux6"} : <i1>, [<i32>, <i32>] to <i32>
    %104 = buffer %103, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer39"} : <i32>
    %105 = buffer %104, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer41"} : <i32>
    %106:5 = fork [5] %105 {handshake.bb = 3 : ui32, handshake.name = "fork18"} : <i32>
    %107 = mux %121#4 [%trueResult_6, %trueResult_48] {handshake.bb = 3 : ui32, handshake.name = "mux7"} : <i1>, [<i32>, <i32>] to <i32>
    %108 = mux %121#0 [%trueResult_8, %trueResult_50] {handshake.bb = 3 : ui32, handshake.name = "mux8"} : <i1>, [<i6>, <i6>] to <i6>
    %109 = buffer %108, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer45"} : <i6>
    %110 = buffer %109, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer46"} : <i6>
    %111:3 = fork [3] %110 {handshake.bb = 3 : ui32, handshake.name = "fork19"} : <i6>
    %112 = extsi %111#2 {handshake.bb = 3 : ui32, handshake.name = "extsi25"} : <i6> to <i32>
    %113:2 = fork [2] %112 {handshake.bb = 3 : ui32, handshake.name = "fork20"} : <i32>
    %114 = trunci %111#0 {handshake.bb = 3 : ui32, handshake.name = "trunci1"} : <i6> to <i5>
    %115 = mux %121#1 [%trueResult_10, %trueResult_52] {handshake.bb = 3 : ui32, handshake.name = "mux9"} : <i1>, [<i6>, <i6>] to <i6>
    %116 = buffer %115, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer47"} : <i6>
    %117 = buffer %116, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer48"} : <i6>
    %118:2 = fork [2] %117 {handshake.bb = 3 : ui32, handshake.name = "fork21"} : <i6>
    %119 = extsi %118#1 {handshake.bb = 3 : ui32, handshake.name = "extsi26"} : <i6> to <i32>
    %120:4 = fork [4] %119 {handshake.bb = 3 : ui32, handshake.name = "fork22"} : <i32>
    %result_32, %index_33 = control_merge [%trueResult_12, %trueResult_54]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>, <>] to <>, <i1>
    %121:5 = fork [5] %index_33 {handshake.bb = 3 : ui32, handshake.name = "fork23"} : <i1>
    %122 = buffer %result_32, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer49"} : <>
    %123:2 = fork [2] %122 {handshake.bb = 3 : ui32, handshake.name = "fork24"} : <>
    %124 = constant %123#0 {handshake.bb = 3 : ui32, handshake.name = "constant23", value = 1 : i2} : <>, <i2>
    %125 = extsi %124 {handshake.bb = 3 : ui32, handshake.name = "extsi8"} : <i2> to <i32>
    %126 = source {handshake.bb = 3 : ui32, handshake.name = "source2"} : <>
    %127 = constant %126 {handshake.bb = 3 : ui32, handshake.name = "constant24", value = 20 : i6} : <>, <i6>
    %128 = extsi %127 {handshake.bb = 3 : ui32, handshake.name = "extsi27"} : <i6> to <i7>
    %129 = source {handshake.bb = 3 : ui32, handshake.name = "source3"} : <>
    %130 = constant %129 {handshake.bb = 3 : ui32, handshake.name = "constant25", value = 1 : i2} : <>, <i2>
    %131:2 = fork [2] %130 {handshake.bb = 3 : ui32, handshake.name = "fork25"} : <i2>
    %132 = extsi %131#0 {handshake.bb = 3 : ui32, handshake.name = "extsi28"} : <i2> to <i7>
    %133 = extsi %131#1 {handshake.bb = 3 : ui32, handshake.name = "extsi10"} : <i2> to <i32>
    %134 = source {handshake.bb = 3 : ui32, handshake.name = "source5"} : <>
    %135 = constant %134 {handshake.bb = 3 : ui32, handshake.name = "constant26", value = 4 : i4} : <>, <i4>
    %136 = extsi %135 {handshake.bb = 3 : ui32, handshake.name = "extsi12"} : <i4> to <i32>
    %137:3 = fork [3] %136 {handshake.bb = 3 : ui32, handshake.name = "fork26"} : <i32>
    %138 = source {handshake.bb = 3 : ui32, handshake.name = "source6"} : <>
    %139 = constant %138 {handshake.bb = 3 : ui32, handshake.name = "constant27", value = 2 : i3} : <>, <i3>
    %140 = extsi %139 {handshake.bb = 3 : ui32, handshake.name = "extsi13"} : <i3> to <i32>
    %141:3 = fork [3] %140 {handshake.bb = 3 : ui32, handshake.name = "fork27"} : <i32>
    %142 = shli %120#0, %141#0 {handshake.bb = 3 : ui32, handshake.name = "shli0"} : <i32>
    %143 = shli %120#1, %137#0 {handshake.bb = 3 : ui32, handshake.name = "shli1"} : <i32>
    %144 = buffer %142, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer50"} : <i32>
    %145 = buffer %143, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer51"} : <i32>
    %146 = addi %144, %145 {handshake.bb = 3 : ui32, handshake.name = "addi9"} : <i32>
    %147 = buffer %146, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer54"} : <i32>
    %148 = addi %106#4, %147 {handshake.bb = 3 : ui32, handshake.name = "addi3"} : <i32>
    %149:2 = fork [2] %148 {handshake.bb = 3 : ui32, handshake.name = "fork28"} : <i32>
    %150 = buffer %90, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer32"} : <>
    %151 = gate %149#1, %150 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 3 : ui32, handshake.name = "gate0"} : <i32>, !handshake.control<> to <i32>
    %152 = buffer %91, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer33"} : <i32>
    %153 = buffer %151, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer55"} : <i32>
    %154 = cmpi ne, %153, %152 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 3 : ui32, handshake.name = "cmpi3"} : <i32>
    %155:2 = fork [2] %154 {handshake.bb = 3 : ui32, handshake.name = "fork29"} : <i1>
    %156 = buffer %94, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer36"} : <>
    %trueResult_34, %falseResult_35 = cond_br %157, %156 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 3 : ui32, handshake.name = "cond_br39"} : <i1>, <>
    %157 = buffer %155#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer71"} : <i1>
    sink %trueResult_34 {handshake.name = "sink3"} : <>
    %158 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 3 : ui32, handshake.name = "source10"} : <>
    %159 = mux %155#0 [%falseResult_35, %158] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 3 : ui32, handshake.name = "mux28"} : <i1>, [<>, <>] to <>
    %160 = buffer %159, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer56"} : <>
    %161 = join %160 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 3 : ui32, handshake.name = "join0"} : <>
    %162 = gate %149#0, %161 {handshake.bb = 3 : ui32, handshake.name = "gate1"} : <i32>, !handshake.control<> to <i32>
    %163 = trunci %162 {handshake.bb = 3 : ui32, handshake.name = "trunci2"} : <i32> to <i9>
    %addressResult, %dataResult = load[%163] %outputs#0 {handshake.bb = 3 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["store0", 0, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i9>, <i32>, <i9>, <i32>
    %addressResult_36, %dataResult_37 = load[%114] %outputs_0 {handshake.bb = 3 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i5>, <i32>, <i5>, <i32>
    %164 = shli %113#0, %141#1 {handshake.bb = 3 : ui32, handshake.name = "shli2"} : <i32>
    %165 = shli %113#1, %137#1 {handshake.bb = 3 : ui32, handshake.name = "shli3"} : <i32>
    %166 = buffer %164, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer57"} : <i32>
    %167 = buffer %165, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer58"} : <i32>
    %168 = addi %166, %167 {handshake.bb = 3 : ui32, handshake.name = "addi10"} : <i32>
    %169 = buffer %168, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer59"} : <i32>
    %170 = addi %106#3, %169 {handshake.bb = 3 : ui32, handshake.name = "addi4"} : <i32>
    %171:2 = fork [2] %170 {handshake.bb = 3 : ui32, handshake.name = "fork30"} : <i32>
    %172 = buffer %93, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer35"} : <>
    %173 = gate %171#1, %172 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 3 : ui32, handshake.name = "gate2"} : <i32>, !handshake.control<> to <i32>
    %174 = buffer %92, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer34"} : <i32>
    %175 = buffer %173, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer60"} : <i32>
    %176 = cmpi ne, %175, %174 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 3 : ui32, handshake.name = "cmpi4"} : <i32>
    %177:2 = fork [2] %176 {handshake.bb = 3 : ui32, handshake.name = "fork31"} : <i1>
    %178 = buffer %96, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer37"} : <>
    %trueResult_38, %falseResult_39 = cond_br %179, %178 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 3 : ui32, handshake.name = "cond_br40"} : <i1>, <>
    %179 = buffer %177#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer80"} : <i1>
    sink %trueResult_38 {handshake.name = "sink4"} : <>
    %180 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 3 : ui32, handshake.name = "source11"} : <>
    %181 = mux %177#0 [%falseResult_39, %180] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 3 : ui32, handshake.name = "mux29"} : <i1>, [<>, <>] to <>
    %182 = buffer %181, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer61"} : <>
    %183 = join %182 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 3 : ui32, handshake.name = "join1"} : <>
    %184 = gate %171#0, %183 {handshake.bb = 3 : ui32, handshake.name = "gate3"} : <i32>, !handshake.control<> to <i32>
    %185 = trunci %184 {handshake.bb = 3 : ui32, handshake.name = "trunci3"} : <i32> to <i9>
    %addressResult_40, %dataResult_41 = load[%185] %outputs#1 {handshake.bb = 3 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["store0", 0, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load2"} : <i9>, <i32>, <i9>, <i32>
    %186 = muli %dataResult_37, %dataResult_41 {handshake.bb = 3 : ui32, handshake.name = "muli0"} : <i32>
    %187 = subi %dataResult, %186 {handshake.bb = 3 : ui32, handshake.name = "subi0"} : <i32>
    %188 = shli %120#2, %141#2 {handshake.bb = 3 : ui32, handshake.name = "shli4"} : <i32>
    %189 = shli %120#3, %137#2 {handshake.bb = 3 : ui32, handshake.name = "shli5"} : <i32>
    %190 = buffer %188, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer62"} : <i32>
    %191 = buffer %189, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer63"} : <i32>
    %192 = addi %190, %191 {handshake.bb = 3 : ui32, handshake.name = "addi11"} : <i32>
    %193 = buffer %192, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer64"} : <i32>
    %194 = addi %106#2, %193 {handshake.bb = 3 : ui32, handshake.name = "addi5"} : <i32>
    %195:2 = fork [2] %194 {handshake.bb = 3 : ui32, handshake.name = "fork32"} : <i32>
    %196 = trunci %197 {handshake.bb = 3 : ui32, handshake.name = "trunci4"} : <i32> to <i9>
    %197 = buffer %195#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer88"} : <i32>
    %198 = buffer %195#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer0"} : <i32>
    %199 = buffer %doneResult, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer65"} : <>
    %200 = buffer %199, bufferType = FIFO_BREAK_NONE, numSlots = 1 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 3 : ui32, handshake.name = "buffer1"} : <>
    %201:2 = fork [2] %200 {handshake.bb = 3 : ui32, handshake.name = "fork33"} : <>
    %202 = init %201#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 3 : ui32, handshake.name = "init18"} : <>
    %addressResult_42, %dataResult_43, %doneResult = store[%196] %187 %outputs#2 {handshake.bb = 3 : ui32, handshake.bufProps = #handshake<bufProps{"2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["load0", 0, false], ["load2", 0, false], ["store0", 0, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store0"} : <i9>, <i32>, <>, <i9>, <i32>, <>
    %203 = buffer %107, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer42"} : <i32>
    %204 = addi %203, %106#1 {handshake.bb = 3 : ui32, handshake.name = "addi0"} : <i32>
    %205 = addi %106#0, %133 {handshake.bb = 3 : ui32, handshake.name = "addi1"} : <i32>
    %206 = addi %102, %132 {handshake.bb = 3 : ui32, handshake.name = "addi6"} : <i7>
    %207 = buffer %206, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer67"} : <i7>
    %208:2 = fork [2] %207 {handshake.bb = 3 : ui32, handshake.name = "fork34"} : <i7>
    %209 = trunci %208#0 {handshake.bb = 3 : ui32, handshake.name = "trunci5"} : <i7> to <i6>
    %210 = cmpi ult, %208#1, %128 {handshake.bb = 3 : ui32, handshake.name = "cmpi0"} : <i7>
    %211 = buffer %210, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer68"} : <i1>
    %212:10 = fork [10] %211 {handshake.bb = 3 : ui32, handshake.name = "fork35"} : <i1>
    %trueResult_44, %falseResult_45 = cond_br %212#0, %209 {handshake.bb = 3 : ui32, handshake.name = "cond_br9"} : <i1>, <i6>
    sink %falseResult_45 {handshake.name = "sink5"} : <i6>
    %trueResult_46, %falseResult_47 = cond_br %212#7, %205 {handshake.bb = 3 : ui32, handshake.name = "cond_br10"} : <i1>, <i32>
    sink %falseResult_47 {handshake.name = "sink6"} : <i32>
    %213 = buffer %204, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer66"} : <i32>
    %trueResult_48, %falseResult_49 = cond_br %212#8, %213 {handshake.bb = 3 : ui32, handshake.name = "cond_br11"} : <i1>, <i32>
    %trueResult_50, %falseResult_51 = cond_br %212#1, %111#1 {handshake.bb = 3 : ui32, handshake.name = "cond_br12"} : <i1>, <i6>
    %trueResult_52, %falseResult_53 = cond_br %212#2, %118#0 {handshake.bb = 3 : ui32, handshake.name = "cond_br13"} : <i1>, <i6>
    %trueResult_54, %falseResult_55 = cond_br %212#9, %123#1 {handshake.bb = 3 : ui32, handshake.name = "cond_br14"} : <i1>, <>
    %214 = extsi %falseResult_53 {handshake.bb = 4 : ui32, handshake.name = "extsi29"} : <i6> to <i7>
    %215 = source {handshake.bb = 4 : ui32, handshake.name = "source7"} : <>
    %216 = constant %215 {handshake.bb = 4 : ui32, handshake.name = "constant28", value = 1 : i2} : <>, <i2>
    %217 = extsi %216 {handshake.bb = 4 : ui32, handshake.name = "extsi30"} : <i2> to <i7>
    %218 = addi %214, %217 {handshake.bb = 4 : ui32, handshake.name = "addi8"} : <i7>
    %219 = buffer %218, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer69"} : <i7>
    %trueResult_56, %falseResult_57 = cond_br %233#6, %falseResult_15 {handshake.bb = 5 : ui32, handshake.name = "cond_br56"} : <i1>, <i32>
    sink %falseResult_57 {handshake.name = "sink8"} : <i32>
    %trueResult_58, %falseResult_59 = cond_br %233#5, %falseResult_19 {handshake.bb = 5 : ui32, handshake.name = "cond_br57"} : <i1>, <i32>
    sink %falseResult_59 {handshake.name = "sink9"} : <i32>
    %trueResult_60, %falseResult_61 = cond_br %233#4, %falseResult_17 {handshake.bb = 5 : ui32, handshake.name = "cond_br58"} : <i1>, <>
    sink %falseResult_61 {handshake.name = "sink10"} : <>
    %trueResult_62, %falseResult_63 = cond_br %233#3, %falseResult_21 {handshake.bb = 5 : ui32, handshake.name = "cond_br59"} : <i1>, <>
    sink %falseResult_63 {handshake.name = "sink11"} : <>
    %trueResult_64, %falseResult_65 = cond_br %233#2, %falseResult_29 {handshake.bb = 5 : ui32, handshake.name = "cond_br60"} : <i1>, <>
    sink %falseResult_65 {handshake.name = "sink12"} : <>
    %trueResult_66, %falseResult_67 = cond_br %233#1, %falseResult_25 {handshake.bb = 5 : ui32, handshake.name = "cond_br61"} : <i1>, <>
    sink %falseResult_67 {handshake.name = "sink13"} : <>
    %220 = extsi %falseResult_9 {handshake.bb = 5 : ui32, handshake.name = "extsi31"} : <i6> to <i7>
    %221 = source {handshake.bb = 5 : ui32, handshake.name = "source8"} : <>
    %222 = constant %221 {handshake.bb = 5 : ui32, handshake.name = "constant29", value = 19 : i6} : <>, <i6>
    %223 = extsi %222 {handshake.bb = 5 : ui32, handshake.name = "extsi32"} : <i6> to <i7>
    %224 = source {handshake.bb = 5 : ui32, handshake.name = "source9"} : <>
    %225 = constant %224 {handshake.bb = 5 : ui32, handshake.name = "constant30", value = 1 : i2} : <>, <i2>
    %226 = extsi %225 {handshake.bb = 5 : ui32, handshake.name = "extsi33"} : <i2> to <i7>
    %227 = addi %220, %226 {handshake.bb = 5 : ui32, handshake.name = "addi7"} : <i7>
    %228 = buffer %227, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 5 : ui32, handshake.name = "buffer70"} : <i7>
    %229:2 = fork [2] %228 {handshake.bb = 5 : ui32, handshake.name = "fork36"} : <i7>
    %230 = trunci %229#0 {handshake.bb = 5 : ui32, handshake.name = "trunci6"} : <i7> to <i6>
    %231 = cmpi ult, %229#1, %223 {handshake.bb = 5 : ui32, handshake.name = "cmpi1"} : <i7>
    %232 = buffer %231, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 5 : ui32, handshake.name = "buffer72"} : <i1>
    %233:10 = fork [10] %232 {handshake.bb = 5 : ui32, handshake.name = "fork37"} : <i1>
    %trueResult_68, %falseResult_69 = cond_br %233#0, %230 {handshake.bb = 5 : ui32, handshake.name = "cond_br15"} : <i1>, <i6>
    sink %falseResult_69 {handshake.name = "sink15"} : <i6>
    %trueResult_70, %falseResult_71 = cond_br %233#8, %falseResult_7 {handshake.bb = 5 : ui32, handshake.name = "cond_br16"} : <i1>, <i32>
    %trueResult_72, %falseResult_73 = cond_br %233#9, %falseResult_13 {handshake.bb = 5 : ui32, handshake.name = "cond_br17"} : <i1>, <>
    %234:2 = fork [2] %falseResult_73 {handshake.bb = 6 : ui32, handshake.name = "fork38"} : <>
    end {handshake.bb = 6 : ui32, handshake.name = "end0"} %falseResult_71, %memEnd_1, %memEnd, %0#3 : <i32>, <>, <>, <>
  }
}

