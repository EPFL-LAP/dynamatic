module {
  handshake.func @gaussian(%arg0: memref<20xi32>, %arg1: memref<400xi32>, %arg2: !handshake.control<>, %arg3: !handshake.control<>, %arg4: !handshake.control<>, ...) -> (!handshake.channel<i32>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["c", "a", "c_start", "a_start", "start"], cfg.edges = "[0,1][2,3,5,cmpi2][4,2][1,2][3,3,4,cmpi0][5,1,6,cmpi1]", resNames = ["out0", "c_end", "a_end", "end"]} {
    %0:6 = fork [6] %arg4 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork0"} : <>
    %outputs:3, %memEnd = mem_controller[%arg1 : memref<400xi32>] %arg3 (%103, %addressResult, %addressResult_24, %addressResult_26, %dataResult_27) %196#1 {connectedBlocks = [3 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "5": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "6": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller1"} :    (!handshake.channel<i32>, !handshake.channel<i9>, !handshake.channel<i9>, !handshake.channel<i9>, !handshake.channel<i32>) -> (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>)
    %outputs_0, %memEnd_1 = mem_controller[%arg0 : memref<20xi32>] %arg2 (%addressResult_22) %196#0 {connectedBlocks = [3 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller2"} :    (!handshake.channel<i5>) -> !handshake.channel<i32>
    %1 = constant %0#1 {handshake.bb = 0 : ui32, handshake.name = "constant1", value = false} : <>, <i1>
    %2 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant3", value = 1 : i2} : <>, <i2>
    %3 = extsi %2 {handshake.bb = 0 : ui32, handshake.name = "extsi16"} : <i2> to <i6>
    %4 = extsi %1 {handshake.bb = 0 : ui32, handshake.name = "extsi17"} : <i1> to <i32>
    %5 = mux %6 [%0#4, %trueResult_40] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux10"} : <i1>, [<>, <>] to <>
    %6 = buffer %9#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer1"} : <i1>
    %7 = mux %9#1 [%0#3, %trueResult_42] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux11"} : <i1>, [<>, <>] to <>
    %8 = init %193#3 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "init0"} : <i1>
    %9:2 = fork [2] %8 {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <i1>
    %10 = mux %16#0 [%3, %trueResult_44] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i6>, <i6>] to <i6>
    %11 = buffer %10, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer6"} : <i6>
    %12:2 = fork [2] %11 {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <i6>
    %13 = extsi %12#1 {handshake.bb = 1 : ui32, handshake.name = "extsi18"} : <i6> to <i7>
    %14 = mux %16#1 [%4, %trueResult_46] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i1>, [<i32>, <i32>] to <i32>
    %15 = buffer %trueResult_48, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer70"} : <>
    %result, %index = control_merge [%0#5, %15]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %16:2 = fork [2] %index {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <i1>
    %17 = source {handshake.bb = 1 : ui32, handshake.name = "source0"} : <>
    %18 = constant %17 {handshake.bb = 1 : ui32, handshake.name = "constant5", value = 1 : i2} : <>, <i2>
    %19 = extsi %18 {handshake.bb = 1 : ui32, handshake.name = "extsi19"} : <i2> to <i7>
    %20 = addi %13, %19 {handshake.bb = 1 : ui32, handshake.name = "addi2"} : <i7>
    %21 = buffer %20, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer10"} : <i7>
    %22 = buffer %14, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer7"} : <i32>
    %23 = buffer %5, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer2"} : <>
    %24 = buffer %23, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer3"} : <>
    %25 = mux %26 [%24, %68#0] {ftd.phi, handshake.bb = 2 : ui32, handshake.name = "mux12"} : <i1>, [<>, <>] to <>
    %26 = buffer %32#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer8"} : <i1>
    %27 = buffer %7, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer4"} : <>
    %28 = buffer %27, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer5"} : <>
    %29 = mux %30 [%28, %68#1] {ftd.phi, handshake.bb = 2 : ui32, handshake.name = "mux13"} : <i1>, [<>, <>] to <>
    %30 = buffer %32#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer9"} : <i1>
    %31 = init %51#2 {ftd.imerge, handshake.bb = 2 : ui32, handshake.name = "init2"} : <i1>
    %32:2 = fork [2] %31 {handshake.bb = 2 : ui32, handshake.name = "fork4"} : <i1>
    %33 = mux %34 [%21, %176] {handshake.bb = 2 : ui32, handshake.name = "mux2"} : <i1>, [<i7>, <i7>] to <i7>
    %34 = buffer %42#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer11"} : <i1>
    %35 = buffer %33, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer16"} : <i7>
    %36:2 = fork [2] %35 {handshake.bb = 2 : ui32, handshake.name = "fork5"} : <i7>
    %37 = trunci %36#0 {handshake.bb = 2 : ui32, handshake.name = "trunci0"} : <i7> to <i6>
    %38 = buffer %22, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer12"} : <i32>
    %39 = mux %42#2 [%38, %falseResult_33] {handshake.bb = 2 : ui32, handshake.name = "mux3"} : <i1>, [<i32>, <i32>] to <i32>
    %40 = mux %41 [%12#0, %falseResult_35] {handshake.bb = 2 : ui32, handshake.name = "mux4"} : <i1>, [<i6>, <i6>] to <i6>
    %41 = buffer %42#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer14"} : <i1>
    %result_2, %index_3 = control_merge [%result, %falseResult_39]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>, <>] to <>, <i1>
    %42:3 = fork [3] %index_3 {handshake.bb = 2 : ui32, handshake.name = "fork6"} : <i1>
    %43:2 = fork [2] %result_2 {handshake.bb = 2 : ui32, handshake.name = "fork7"} : <>
    %44 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %45 = constant %44 {handshake.bb = 2 : ui32, handshake.name = "constant6", value = 19 : i6} : <>, <i6>
    %46 = extsi %45 {handshake.bb = 2 : ui32, handshake.name = "extsi20"} : <i6> to <i7>
    %47 = constant %43#0 {handshake.bb = 2 : ui32, handshake.name = "constant19", value = 1 : i2} : <>, <i2>
    %48:2 = fork [2] %47 {handshake.bb = 2 : ui32, handshake.name = "fork8"} : <i2>
    %49 = cmpi ult, %36#1, %46 {handshake.bb = 2 : ui32, handshake.name = "cmpi2"} : <i7>
    %50 = buffer %49, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer33"} : <i1>
    %51:9 = fork [9] %50 {handshake.bb = 2 : ui32, handshake.name = "fork9"} : <i1>
    %trueResult, %falseResult = cond_br %51#8, %52 {handshake.bb = 2 : ui32, handshake.name = "cond_br3"} : <i1>, <i2>
    %52 = buffer %48#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer17"} : <i2>
    sink %falseResult {handshake.name = "sink0"} : <i2>
    %53 = extsi %trueResult {handshake.bb = 2 : ui32, handshake.name = "extsi15"} : <i2> to <i6>
    %trueResult_4, %falseResult_5 = cond_br %54, %55 {handshake.bb = 2 : ui32, handshake.name = "cond_br4"} : <i1>, <i2>
    %54 = buffer %51#7, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer18"} : <i1>
    %55 = buffer %48#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer19"} : <i2>
    sink %falseResult_5 {handshake.name = "sink1"} : <i2>
    %56 = extsi %trueResult_4 {handshake.bb = 2 : ui32, handshake.name = "extsi21"} : <i2> to <i32>
    %57 = buffer %39, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer21"} : <i32>
    %58 = buffer %57, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer22"} : <i32>
    %trueResult_6, %falseResult_7 = cond_br %59, %58 {handshake.bb = 2 : ui32, handshake.name = "cond_br5"} : <i1>, <i32>
    %59 = buffer %51#3, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer20"} : <i1>
    %60 = buffer %40, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer30"} : <i6>
    %61 = buffer %60, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer32"} : <i6>
    %trueResult_8, %falseResult_9 = cond_br %51#1, %61 {handshake.bb = 2 : ui32, handshake.name = "cond_br6"} : <i1>, <i6>
    %trueResult_10, %falseResult_11 = cond_br %51#0, %37 {handshake.bb = 2 : ui32, handshake.name = "cond_br7"} : <i1>, <i6>
    sink %falseResult_11 {handshake.name = "sink2"} : <i6>
    %trueResult_12, %falseResult_13 = cond_br %62, %43#1 {handshake.bb = 2 : ui32, handshake.name = "cond_br8"} : <i1>, <>
    %62 = buffer %51#4, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer23"} : <i1>
    %63 = buffer %29, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer15"} : <>
    %trueResult_14, %falseResult_15 = cond_br %64, %63 {handshake.bb = 3 : ui32, handshake.name = "cond_br25"} : <i1>, <>
    %64 = buffer %51#5, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer24"} : <i1>
    %65 = buffer %25, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer13"} : <>
    %trueResult_16, %falseResult_17 = cond_br %66, %65 {handshake.bb = 3 : ui32, handshake.name = "cond_br26"} : <i1>, <>
    %66 = buffer %51#6, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer25"} : <i1>
    %trueResult_18, %falseResult_19 = cond_br %67, %154 {handshake.bb = 3 : ui32, handshake.name = "cond_br27"} : <i1>, <>
    %67 = buffer %166#4, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer26"} : <i1>
    %68:2 = fork [2] %falseResult_19 {handshake.bb = 3 : ui32, handshake.name = "fork10"} : <>
    %69:2 = fork [2] %trueResult_18 {handshake.bb = 3 : ui32, handshake.name = "fork11"} : <>
    %70 = mux %71 [%trueResult_16, %69#1] {ftd.phi, handshake.bb = 3 : ui32, handshake.name = "mux14"} : <i1>, [<>, <>] to <>
    %71 = buffer %76#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer27"} : <i1>
    %72 = mux %73 [%trueResult_14, %69#0] {ftd.phi, handshake.bb = 3 : ui32, handshake.name = "mux15"} : <i1>, [<>, <>] to <>
    %73 = buffer %76#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer28"} : <i1>
    %74 = init %75 {ftd.imerge, handshake.bb = 3 : ui32, handshake.name = "init4"} : <i1>
    %75 = buffer %166#3, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer29"} : <i1>
    %76:2 = fork [2] %74 {handshake.bb = 3 : ui32, handshake.name = "fork12"} : <i1>
    %77 = mux %100#2 [%53, %trueResult_28] {handshake.bb = 3 : ui32, handshake.name = "mux5"} : <i1>, [<i6>, <i6>] to <i6>
    %78 = buffer %77, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer36"} : <i6>
    %79 = extsi %78 {handshake.bb = 3 : ui32, handshake.name = "extsi22"} : <i6> to <i7>
    %80 = mux %81 [%56, %trueResult_30] {handshake.bb = 3 : ui32, handshake.name = "mux6"} : <i1>, [<i32>, <i32>] to <i32>
    %81 = buffer %100#3, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer31"} : <i1>
    %82 = buffer %80, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer37"} : <i32>
    %83 = buffer %82, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer38"} : <i32>
    %84:5 = fork [5] %83 {handshake.bb = 3 : ui32, handshake.name = "fork13"} : <i32>
    %85 = trunci %84#0 {handshake.bb = 3 : ui32, handshake.name = "trunci1"} : <i32> to <i9>
    %86 = mux %100#4 [%trueResult_6, %trueResult_32] {handshake.bb = 3 : ui32, handshake.name = "mux7"} : <i1>, [<i32>, <i32>] to <i32>
    %87 = mux %100#0 [%trueResult_8, %trueResult_34] {handshake.bb = 3 : ui32, handshake.name = "mux8"} : <i1>, [<i6>, <i6>] to <i6>
    %88 = buffer %87, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer40"} : <i6>
    %89 = buffer %88, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer41"} : <i6>
    %90:3 = fork [3] %89 {handshake.bb = 3 : ui32, handshake.name = "fork14"} : <i6>
    %91 = extsi %90#2 {handshake.bb = 3 : ui32, handshake.name = "extsi23"} : <i6> to <i32>
    %92:2 = fork [2] %91 {handshake.bb = 3 : ui32, handshake.name = "fork15"} : <i32>
    %93 = trunci %90#0 {handshake.bb = 3 : ui32, handshake.name = "trunci2"} : <i6> to <i5>
    %94 = mux %100#1 [%trueResult_10, %trueResult_36] {handshake.bb = 3 : ui32, handshake.name = "mux9"} : <i1>, [<i6>, <i6>] to <i6>
    %95 = buffer %94, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer42"} : <i6>
    %96 = buffer %95, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer43"} : <i6>
    %97:2 = fork [2] %96 {handshake.bb = 3 : ui32, handshake.name = "fork16"} : <i6>
    %98 = extsi %97#1 {handshake.bb = 3 : ui32, handshake.name = "extsi24"} : <i6> to <i32>
    %99:4 = fork [4] %98 {handshake.bb = 3 : ui32, handshake.name = "fork17"} : <i32>
    %result_20, %index_21 = control_merge [%trueResult_12, %trueResult_38]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>, <>] to <>, <i1>
    %100:5 = fork [5] %index_21 {handshake.bb = 3 : ui32, handshake.name = "fork18"} : <i1>
    %101:2 = fork [2] %result_20 {handshake.bb = 3 : ui32, handshake.name = "fork19"} : <>
    %102 = constant %101#0 {handshake.bb = 3 : ui32, handshake.name = "constant20", value = 1 : i2} : <>, <i2>
    %103 = extsi %102 {handshake.bb = 3 : ui32, handshake.name = "extsi6"} : <i2> to <i32>
    %104 = source {handshake.bb = 3 : ui32, handshake.name = "source2"} : <>
    %105 = constant %104 {handshake.bb = 3 : ui32, handshake.name = "constant21", value = 20 : i6} : <>, <i6>
    %106 = extsi %105 {handshake.bb = 3 : ui32, handshake.name = "extsi25"} : <i6> to <i7>
    %107 = source {handshake.bb = 3 : ui32, handshake.name = "source3"} : <>
    %108 = constant %107 {handshake.bb = 3 : ui32, handshake.name = "constant22", value = 1 : i2} : <>, <i2>
    %109:2 = fork [2] %108 {handshake.bb = 3 : ui32, handshake.name = "fork20"} : <i2>
    %110 = extsi %109#0 {handshake.bb = 3 : ui32, handshake.name = "extsi26"} : <i2> to <i7>
    %111 = extsi %109#1 {handshake.bb = 3 : ui32, handshake.name = "extsi8"} : <i2> to <i32>
    %112 = source {handshake.bb = 3 : ui32, handshake.name = "source5"} : <>
    %113 = constant %112 {handshake.bb = 3 : ui32, handshake.name = "constant23", value = 4 : i4} : <>, <i4>
    %114 = extsi %113 {handshake.bb = 3 : ui32, handshake.name = "extsi10"} : <i4> to <i32>
    %115:3 = fork [3] %114 {handshake.bb = 3 : ui32, handshake.name = "fork21"} : <i32>
    %116 = source {handshake.bb = 3 : ui32, handshake.name = "source6"} : <>
    %117 = constant %116 {handshake.bb = 3 : ui32, handshake.name = "constant24", value = 2 : i3} : <>, <i3>
    %118 = extsi %117 {handshake.bb = 3 : ui32, handshake.name = "extsi11"} : <i3> to <i32>
    %119:3 = fork [3] %118 {handshake.bb = 3 : ui32, handshake.name = "fork22"} : <i32>
    %120 = shli %99#0, %119#0 {handshake.bb = 3 : ui32, handshake.name = "shli0"} : <i32>
    %121 = shli %122, %115#0 {handshake.bb = 3 : ui32, handshake.name = "shli1"} : <i32>
    %122 = buffer %99#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer44"} : <i32>
    %123 = buffer %120, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer47"} : <i32>
    %124 = buffer %121, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer48"} : <i32>
    %125 = addi %123, %124 {handshake.bb = 3 : ui32, handshake.name = "addi9"} : <i32>
    %126 = buffer %125, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer49"} : <i32>
    %127 = addi %128, %126 {handshake.bb = 3 : ui32, handshake.name = "addi3"} : <i32>
    %128 = buffer %84#4, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer45"} : <i32>
    %129 = buffer %72, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer35"} : <>
    %130 = gate %127, %129 {handshake.bb = 3 : ui32, handshake.name = "gate0"} : <i32>, !handshake.control<> to <i32>
    %131 = trunci %130 {handshake.bb = 3 : ui32, handshake.name = "trunci3"} : <i32> to <i9>
    %addressResult, %dataResult = load[%131] %outputs#0 {handshake.bb = 3 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["store0", 0, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i9>, <i32>, <i9>, <i32>
    %addressResult_22, %dataResult_23 = load[%93] %outputs_0 {handshake.bb = 3 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i5>, <i32>, <i5>, <i32>
    %132 = shli %92#0, %119#1 {handshake.bb = 3 : ui32, handshake.name = "shli2"} : <i32>
    %133 = shli %92#1, %115#1 {handshake.bb = 3 : ui32, handshake.name = "shli3"} : <i32>
    %134 = buffer %132, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer50"} : <i32>
    %135 = buffer %133, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer51"} : <i32>
    %136 = addi %134, %135 {handshake.bb = 3 : ui32, handshake.name = "addi10"} : <i32>
    %137 = buffer %136, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer52"} : <i32>
    %138 = addi %84#3, %137 {handshake.bb = 3 : ui32, handshake.name = "addi4"} : <i32>
    %139 = buffer %70, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer34"} : <>
    %140 = gate %138, %139 {handshake.bb = 3 : ui32, handshake.name = "gate1"} : <i32>, !handshake.control<> to <i32>
    %141 = trunci %140 {handshake.bb = 3 : ui32, handshake.name = "trunci4"} : <i32> to <i9>
    %addressResult_24, %dataResult_25 = load[%141] %outputs#1 {handshake.bb = 3 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["store0", 0, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load2"} : <i9>, <i32>, <i9>, <i32>
    %142 = muli %dataResult_23, %dataResult_25 {handshake.bb = 3 : ui32, handshake.name = "muli0"} : <i32>
    %143 = subi %dataResult, %142 {handshake.bb = 3 : ui32, handshake.name = "subi0"} : <i32>
    %144 = shli %99#2, %119#2 {handshake.bb = 3 : ui32, handshake.name = "shli4"} : <i32>
    %145 = buffer %144, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer53"} : <i32>
    %146 = trunci %145 {handshake.bb = 3 : ui32, handshake.name = "trunci5"} : <i32> to <i9>
    %147 = shli %99#3, %115#2 {handshake.bb = 3 : ui32, handshake.name = "shli5"} : <i32>
    %148 = buffer %147, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer54"} : <i32>
    %149 = trunci %148 {handshake.bb = 3 : ui32, handshake.name = "trunci6"} : <i32> to <i9>
    %150 = addi %146, %149 {handshake.bb = 3 : ui32, handshake.name = "addi11"} : <i9>
    %151 = buffer %150, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer55"} : <i9>
    %152 = addi %85, %151 {handshake.bb = 3 : ui32, handshake.name = "addi5"} : <i9>
    %153 = buffer %doneResult, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer56"} : <>
    %154 = buffer %153, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer0"} : <>
    %addressResult_26, %dataResult_27, %doneResult = store[%152] %143 %outputs#2 {handshake.bb = 3 : ui32, handshake.bufProps = #handshake<bufProps{"2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["load0", 0, false], ["load2", 0, false], ["store0", 0, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store0"} : <i9>, <i32>, <>, <i9>, <i32>, <>
    %155 = buffer %86, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer39"} : <i32>
    %156 = addi %155, %84#2 {handshake.bb = 3 : ui32, handshake.name = "addi0"} : <i32>
    %157 = addi %84#1, %111 {handshake.bb = 3 : ui32, handshake.name = "addi1"} : <i32>
    %158 = addi %79, %110 {handshake.bb = 3 : ui32, handshake.name = "addi6"} : <i7>
    %159 = buffer %158, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer61"} : <i7>
    %160:2 = fork [2] %159 {handshake.bb = 3 : ui32, handshake.name = "fork23"} : <i7>
    %161 = trunci %162 {handshake.bb = 3 : ui32, handshake.name = "trunci7"} : <i7> to <i6>
    %162 = buffer %160#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer57"} : <i7>
    %163 = cmpi ult, %164, %106 {handshake.bb = 3 : ui32, handshake.name = "cmpi0"} : <i7>
    %164 = buffer %160#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer58"} : <i7>
    %165 = buffer %163, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer62"} : <i1>
    %166:8 = fork [8] %165 {handshake.bb = 3 : ui32, handshake.name = "fork24"} : <i1>
    %trueResult_28, %falseResult_29 = cond_br %166#0, %161 {handshake.bb = 3 : ui32, handshake.name = "cond_br9"} : <i1>, <i6>
    sink %falseResult_29 {handshake.name = "sink3"} : <i6>
    %trueResult_30, %falseResult_31 = cond_br %167, %157 {handshake.bb = 3 : ui32, handshake.name = "cond_br10"} : <i1>, <i32>
    %167 = buffer %166#5, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer60"} : <i1>
    sink %falseResult_31 {handshake.name = "sink4"} : <i32>
    %168 = buffer %156, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer59"} : <i32>
    %trueResult_32, %falseResult_33 = cond_br %166#6, %168 {handshake.bb = 3 : ui32, handshake.name = "cond_br11"} : <i1>, <i32>
    %trueResult_34, %falseResult_35 = cond_br %166#1, %90#1 {handshake.bb = 3 : ui32, handshake.name = "cond_br12"} : <i1>, <i6>
    %trueResult_36, %falseResult_37 = cond_br %166#2, %169 {handshake.bb = 3 : ui32, handshake.name = "cond_br13"} : <i1>, <i6>
    %169 = buffer %97#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer65"} : <i6>
    %170 = buffer %101#1, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer46"} : <>
    %trueResult_38, %falseResult_39 = cond_br %166#7, %170 {handshake.bb = 3 : ui32, handshake.name = "cond_br14"} : <i1>, <>
    %171 = extsi %falseResult_37 {handshake.bb = 4 : ui32, handshake.name = "extsi27"} : <i6> to <i7>
    %172 = source {handshake.bb = 4 : ui32, handshake.name = "source7"} : <>
    %173 = constant %172 {handshake.bb = 4 : ui32, handshake.name = "constant25", value = 1 : i2} : <>, <i2>
    %174 = extsi %173 {handshake.bb = 4 : ui32, handshake.name = "extsi28"} : <i2> to <i7>
    %175 = addi %171, %174 {handshake.bb = 4 : ui32, handshake.name = "addi8"} : <i7>
    %176 = buffer %175, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer63"} : <i7>
    %trueResult_40, %falseResult_41 = cond_br %177, %falseResult_17 {handshake.bb = 5 : ui32, handshake.name = "cond_br28"} : <i1>, <>
    %177 = buffer %193#2, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 5 : ui32, handshake.name = "buffer67"} : <i1>
    sink %falseResult_41 {handshake.name = "sink6"} : <>
    %trueResult_42, %falseResult_43 = cond_br %178, %falseResult_15 {handshake.bb = 5 : ui32, handshake.name = "cond_br29"} : <i1>, <>
    %178 = buffer %193#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 5 : ui32, handshake.name = "buffer68"} : <i1>
    sink %falseResult_43 {handshake.name = "sink7"} : <>
    %179 = extsi %falseResult_9 {handshake.bb = 5 : ui32, handshake.name = "extsi29"} : <i6> to <i7>
    %180 = source {handshake.bb = 5 : ui32, handshake.name = "source8"} : <>
    %181 = constant %180 {handshake.bb = 5 : ui32, handshake.name = "constant26", value = 19 : i6} : <>, <i6>
    %182 = extsi %181 {handshake.bb = 5 : ui32, handshake.name = "extsi30"} : <i6> to <i7>
    %183 = source {handshake.bb = 5 : ui32, handshake.name = "source9"} : <>
    %184 = constant %183 {handshake.bb = 5 : ui32, handshake.name = "constant27", value = 1 : i2} : <>, <i2>
    %185 = extsi %184 {handshake.bb = 5 : ui32, handshake.name = "extsi31"} : <i2> to <i7>
    %186 = addi %179, %185 {handshake.bb = 5 : ui32, handshake.name = "addi7"} : <i7>
    %187 = buffer %186, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 5 : ui32, handshake.name = "buffer64"} : <i7>
    %188:2 = fork [2] %187 {handshake.bb = 5 : ui32, handshake.name = "fork25"} : <i7>
    %189 = trunci %190 {handshake.bb = 5 : ui32, handshake.name = "trunci8"} : <i7> to <i6>
    %190 = buffer %188#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 5 : ui32, handshake.name = "buffer69"} : <i7>
    %191 = cmpi ult, %188#1, %182 {handshake.bb = 5 : ui32, handshake.name = "cmpi1"} : <i7>
    %192 = buffer %191, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 5 : ui32, handshake.name = "buffer66"} : <i1>
    %193:6 = fork [6] %192 {handshake.bb = 5 : ui32, handshake.name = "fork26"} : <i1>
    %trueResult_44, %falseResult_45 = cond_br %193#0, %189 {handshake.bb = 5 : ui32, handshake.name = "cond_br15"} : <i1>, <i6>
    sink %falseResult_45 {handshake.name = "sink9"} : <i6>
    %trueResult_46, %falseResult_47 = cond_br %194, %falseResult_7 {handshake.bb = 5 : ui32, handshake.name = "cond_br16"} : <i1>, <i32>
    %194 = buffer %193#4, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 5 : ui32, handshake.name = "buffer72"} : <i1>
    %trueResult_48, %falseResult_49 = cond_br %195, %falseResult_13 {handshake.bb = 5 : ui32, handshake.name = "cond_br17"} : <i1>, <>
    %195 = buffer %193#5, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 5 : ui32, handshake.name = "buffer73"} : <i1>
    %196:2 = fork [2] %falseResult_49 {handshake.bb = 6 : ui32, handshake.name = "fork27"} : <>
    end {handshake.bb = 6 : ui32, handshake.name = "end0"} %falseResult_47, %memEnd_1, %memEnd, %0#2 : <i32>, <>, <>, <>
  }
}

