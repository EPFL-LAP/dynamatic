module {
  handshake.func @gaussian(%arg0: memref<20xi32>, %arg1: memref<400xi32>, %arg2: !handshake.control<>, %arg3: !handshake.control<>, %arg4: !handshake.control<>, ...) -> (!handshake.channel<i32>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["c", "a", "c_start", "a_start", "start"], cfg.edges = "[0,1][2,3,5,cmpi2][4,2][1,2][3,3,4,cmpi0][5,1,6,cmpi1]", resNames = ["out0", "c_end", "a_end", "end"]} {
    %0:6 = fork [6] %arg4 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork0"} : <>
    %outputs:3, %memEnd = mem_controller[%arg1 : memref<400xi32>] %arg3 (%110, %addressResult, %addressResult_24, %addressResult_26, %dataResult_27) %213#1 {connectedBlocks = [3 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "5": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "6": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller1"} :    (!handshake.channel<i32>, !handshake.channel<i9>, !handshake.channel<i9>, !handshake.channel<i9>, !handshake.channel<i32>) -> (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>)
    %outputs_0, %memEnd_1 = mem_controller[%arg0 : memref<20xi32>] %arg2 (%addressResult_22) %213#0 {connectedBlocks = [3 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller2"} :    (!handshake.channel<i5>) -> !handshake.channel<i32>
    %1 = constant %0#1 {handshake.bb = 0 : ui32, handshake.name = "constant1", value = false} : <>, <i1>
    %2 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant3", value = 1 : i2} : <>, <i2>
    %3 = br %2 {handshake.bb = 0 : ui32, handshake.name = "br2"} : <i2>
    %4 = extsi %3 {handshake.bb = 0 : ui32, handshake.name = "extsi16"} : <i2> to <i6>
    %5 = br %1 {handshake.bb = 0 : ui32, handshake.name = "br4"} : <i1>
    %6 = extsi %5 {handshake.bb = 0 : ui32, handshake.name = "extsi17"} : <i1> to <i32>
    %7 = br %0#5 {handshake.bb = 0 : ui32, handshake.name = "br5"} : <>
    %8 = mux %9 [%0#4, %trueResult_42] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux10"} : <i1>, [<>, <>] to <>
    %9 = buffer %12#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer1"} : <i1>
    %10 = mux %12#1 [%0#3, %trueResult_44] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux11"} : <i1>, [<>, <>] to <>
    %11 = init %209#3 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "init0"} : <i1>
    %12:2 = fork [2] %11 {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <i1>
    %13 = mux %19#0 [%4, %trueResult_48] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i6>, <i6>] to <i6>
    %14 = buffer %13, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer6"} : <i6>
    %15:2 = fork [2] %14 {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <i6>
    %16 = extsi %15#1 {handshake.bb = 1 : ui32, handshake.name = "extsi18"} : <i6> to <i7>
    %17 = mux %19#1 [%6, %trueResult_50] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i1>, [<i32>, <i32>] to <i32>
    %18 = buffer %trueResult_52, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer70"} : <>
    %result, %index = control_merge [%7, %18]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %19:2 = fork [2] %index {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <i1>
    %20 = source {handshake.bb = 1 : ui32, handshake.name = "source0"} : <>
    %21 = constant %20 {handshake.bb = 1 : ui32, handshake.name = "constant5", value = 1 : i2} : <>, <i2>
    %22 = extsi %21 {handshake.bb = 1 : ui32, handshake.name = "extsi19"} : <i2> to <i7>
    %23 = addi %16, %22 {handshake.bb = 1 : ui32, handshake.name = "addi2"} : <i7>
    %24 = buffer %23, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer10"} : <i7>
    %25 = br %24 {handshake.bb = 1 : ui32, handshake.name = "br6"} : <i7>
    %26 = buffer %17, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer7"} : <i32>
    %27 = br %26 {handshake.bb = 1 : ui32, handshake.name = "br7"} : <i32>
    %28 = br %15#0 {handshake.bb = 1 : ui32, handshake.name = "br8"} : <i6>
    %29 = br %result {handshake.bb = 1 : ui32, handshake.name = "br9"} : <>
    %30 = buffer %8, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer2"} : <>
    %31 = buffer %30, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer3"} : <>
    %32 = mux %33 [%31, %75#0] {ftd.phi, handshake.bb = 2 : ui32, handshake.name = "mux12"} : <i1>, [<>, <>] to <>
    %33 = buffer %39#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer8"} : <i1>
    %34 = buffer %10, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer4"} : <>
    %35 = buffer %34, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer5"} : <>
    %36 = mux %37 [%35, %75#1] {ftd.phi, handshake.bb = 2 : ui32, handshake.name = "mux13"} : <i1>, [<>, <>] to <>
    %37 = buffer %39#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer9"} : <i1>
    %38 = init %58#2 {ftd.imerge, handshake.bb = 2 : ui32, handshake.name = "init2"} : <i1>
    %39:2 = fork [2] %38 {handshake.bb = 2 : ui32, handshake.name = "fork4"} : <i1>
    %40 = mux %41 [%25, %187] {handshake.bb = 2 : ui32, handshake.name = "mux2"} : <i1>, [<i7>, <i7>] to <i7>
    %41 = buffer %49#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer11"} : <i1>
    %42 = buffer %40, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer16"} : <i7>
    %43:2 = fork [2] %42 {handshake.bb = 2 : ui32, handshake.name = "fork5"} : <i7>
    %44 = trunci %43#0 {handshake.bb = 2 : ui32, handshake.name = "trunci0"} : <i7> to <i6>
    %45 = buffer %27, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer12"} : <i32>
    %46 = mux %49#2 [%45, %188] {handshake.bb = 2 : ui32, handshake.name = "mux3"} : <i1>, [<i32>, <i32>] to <i32>
    %47 = mux %48 [%28, %189] {handshake.bb = 2 : ui32, handshake.name = "mux4"} : <i1>, [<i6>, <i6>] to <i6>
    %48 = buffer %49#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer14"} : <i1>
    %result_2, %index_3 = control_merge [%29, %190]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>, <>] to <>, <i1>
    %49:3 = fork [3] %index_3 {handshake.bb = 2 : ui32, handshake.name = "fork6"} : <i1>
    %50:2 = fork [2] %result_2 {handshake.bb = 2 : ui32, handshake.name = "fork7"} : <>
    %51 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %52 = constant %51 {handshake.bb = 2 : ui32, handshake.name = "constant6", value = 19 : i6} : <>, <i6>
    %53 = extsi %52 {handshake.bb = 2 : ui32, handshake.name = "extsi20"} : <i6> to <i7>
    %54 = constant %50#0 {handshake.bb = 2 : ui32, handshake.name = "constant19", value = 1 : i2} : <>, <i2>
    %55:2 = fork [2] %54 {handshake.bb = 2 : ui32, handshake.name = "fork8"} : <i2>
    %56 = cmpi ult, %43#1, %53 {handshake.bb = 2 : ui32, handshake.name = "cmpi2"} : <i7>
    %57 = buffer %56, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer33"} : <i1>
    %58:9 = fork [9] %57 {handshake.bb = 2 : ui32, handshake.name = "fork9"} : <i1>
    %trueResult, %falseResult = cond_br %58#8, %59 {handshake.bb = 2 : ui32, handshake.name = "cond_br3"} : <i1>, <i2>
    %59 = buffer %55#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer17"} : <i2>
    sink %falseResult {handshake.name = "sink0"} : <i2>
    %60 = extsi %trueResult {handshake.bb = 2 : ui32, handshake.name = "extsi15"} : <i2> to <i6>
    %trueResult_4, %falseResult_5 = cond_br %61, %62 {handshake.bb = 2 : ui32, handshake.name = "cond_br4"} : <i1>, <i2>
    %61 = buffer %58#7, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer18"} : <i1>
    %62 = buffer %55#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer19"} : <i2>
    sink %falseResult_5 {handshake.name = "sink1"} : <i2>
    %63 = extsi %trueResult_4 {handshake.bb = 2 : ui32, handshake.name = "extsi21"} : <i2> to <i32>
    %64 = buffer %46, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer21"} : <i32>
    %65 = buffer %64, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer22"} : <i32>
    %trueResult_6, %falseResult_7 = cond_br %66, %65 {handshake.bb = 2 : ui32, handshake.name = "cond_br5"} : <i1>, <i32>
    %66 = buffer %58#3, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer20"} : <i1>
    %67 = buffer %47, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer30"} : <i6>
    %68 = buffer %67, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer32"} : <i6>
    %trueResult_8, %falseResult_9 = cond_br %58#1, %68 {handshake.bb = 2 : ui32, handshake.name = "cond_br6"} : <i1>, <i6>
    %trueResult_10, %falseResult_11 = cond_br %58#0, %44 {handshake.bb = 2 : ui32, handshake.name = "cond_br7"} : <i1>, <i6>
    sink %falseResult_11 {handshake.name = "sink2"} : <i6>
    %trueResult_12, %falseResult_13 = cond_br %69, %50#1 {handshake.bb = 2 : ui32, handshake.name = "cond_br8"} : <i1>, <>
    %69 = buffer %58#4, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer23"} : <i1>
    %70 = buffer %36, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer15"} : <>
    %trueResult_14, %falseResult_15 = cond_br %71, %70 {handshake.bb = 3 : ui32, handshake.name = "cond_br25"} : <i1>, <>
    %71 = buffer %58#5, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer24"} : <i1>
    %72 = buffer %32, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer13"} : <>
    %trueResult_16, %falseResult_17 = cond_br %73, %72 {handshake.bb = 3 : ui32, handshake.name = "cond_br26"} : <i1>, <>
    %73 = buffer %58#6, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer25"} : <i1>
    %trueResult_18, %falseResult_19 = cond_br %74, %161 {handshake.bb = 3 : ui32, handshake.name = "cond_br27"} : <i1>, <>
    %74 = buffer %173#4, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer26"} : <i1>
    %75:2 = fork [2] %falseResult_19 {handshake.bb = 3 : ui32, handshake.name = "fork10"} : <>
    %76:2 = fork [2] %trueResult_18 {handshake.bb = 3 : ui32, handshake.name = "fork11"} : <>
    %77 = mux %78 [%trueResult_16, %76#1] {ftd.phi, handshake.bb = 3 : ui32, handshake.name = "mux14"} : <i1>, [<>, <>] to <>
    %78 = buffer %83#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer27"} : <i1>
    %79 = mux %80 [%trueResult_14, %76#0] {ftd.phi, handshake.bb = 3 : ui32, handshake.name = "mux15"} : <i1>, [<>, <>] to <>
    %80 = buffer %83#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer28"} : <i1>
    %81 = init %82 {ftd.imerge, handshake.bb = 3 : ui32, handshake.name = "init4"} : <i1>
    %82 = buffer %173#3, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer29"} : <i1>
    %83:2 = fork [2] %81 {handshake.bb = 3 : ui32, handshake.name = "fork12"} : <i1>
    %84 = mux %107#2 [%60, %trueResult_28] {handshake.bb = 3 : ui32, handshake.name = "mux5"} : <i1>, [<i6>, <i6>] to <i6>
    %85 = buffer %84, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer36"} : <i6>
    %86 = extsi %85 {handshake.bb = 3 : ui32, handshake.name = "extsi22"} : <i6> to <i7>
    %87 = mux %88 [%63, %trueResult_30] {handshake.bb = 3 : ui32, handshake.name = "mux6"} : <i1>, [<i32>, <i32>] to <i32>
    %88 = buffer %107#3, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer31"} : <i1>
    %89 = buffer %87, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer37"} : <i32>
    %90 = buffer %89, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer38"} : <i32>
    %91:5 = fork [5] %90 {handshake.bb = 3 : ui32, handshake.name = "fork13"} : <i32>
    %92 = trunci %91#0 {handshake.bb = 3 : ui32, handshake.name = "trunci1"} : <i32> to <i9>
    %93 = mux %107#4 [%trueResult_6, %trueResult_32] {handshake.bb = 3 : ui32, handshake.name = "mux7"} : <i1>, [<i32>, <i32>] to <i32>
    %94 = mux %107#0 [%trueResult_8, %trueResult_34] {handshake.bb = 3 : ui32, handshake.name = "mux8"} : <i1>, [<i6>, <i6>] to <i6>
    %95 = buffer %94, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer40"} : <i6>
    %96 = buffer %95, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer41"} : <i6>
    %97:3 = fork [3] %96 {handshake.bb = 3 : ui32, handshake.name = "fork14"} : <i6>
    %98 = extsi %97#2 {handshake.bb = 3 : ui32, handshake.name = "extsi23"} : <i6> to <i32>
    %99:2 = fork [2] %98 {handshake.bb = 3 : ui32, handshake.name = "fork15"} : <i32>
    %100 = trunci %97#0 {handshake.bb = 3 : ui32, handshake.name = "trunci2"} : <i6> to <i5>
    %101 = mux %107#1 [%trueResult_10, %trueResult_36] {handshake.bb = 3 : ui32, handshake.name = "mux9"} : <i1>, [<i6>, <i6>] to <i6>
    %102 = buffer %101, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer42"} : <i6>
    %103 = buffer %102, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer43"} : <i6>
    %104:2 = fork [2] %103 {handshake.bb = 3 : ui32, handshake.name = "fork16"} : <i6>
    %105 = extsi %104#1 {handshake.bb = 3 : ui32, handshake.name = "extsi24"} : <i6> to <i32>
    %106:4 = fork [4] %105 {handshake.bb = 3 : ui32, handshake.name = "fork17"} : <i32>
    %result_20, %index_21 = control_merge [%trueResult_12, %trueResult_38]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>, <>] to <>, <i1>
    %107:5 = fork [5] %index_21 {handshake.bb = 3 : ui32, handshake.name = "fork18"} : <i1>
    %108:2 = fork [2] %result_20 {handshake.bb = 3 : ui32, handshake.name = "fork19"} : <>
    %109 = constant %108#0 {handshake.bb = 3 : ui32, handshake.name = "constant20", value = 1 : i2} : <>, <i2>
    %110 = extsi %109 {handshake.bb = 3 : ui32, handshake.name = "extsi6"} : <i2> to <i32>
    %111 = source {handshake.bb = 3 : ui32, handshake.name = "source2"} : <>
    %112 = constant %111 {handshake.bb = 3 : ui32, handshake.name = "constant21", value = 20 : i6} : <>, <i6>
    %113 = extsi %112 {handshake.bb = 3 : ui32, handshake.name = "extsi25"} : <i6> to <i7>
    %114 = source {handshake.bb = 3 : ui32, handshake.name = "source3"} : <>
    %115 = constant %114 {handshake.bb = 3 : ui32, handshake.name = "constant22", value = 1 : i2} : <>, <i2>
    %116:2 = fork [2] %115 {handshake.bb = 3 : ui32, handshake.name = "fork20"} : <i2>
    %117 = extsi %116#0 {handshake.bb = 3 : ui32, handshake.name = "extsi26"} : <i2> to <i7>
    %118 = extsi %116#1 {handshake.bb = 3 : ui32, handshake.name = "extsi8"} : <i2> to <i32>
    %119 = source {handshake.bb = 3 : ui32, handshake.name = "source5"} : <>
    %120 = constant %119 {handshake.bb = 3 : ui32, handshake.name = "constant23", value = 4 : i4} : <>, <i4>
    %121 = extsi %120 {handshake.bb = 3 : ui32, handshake.name = "extsi10"} : <i4> to <i32>
    %122:3 = fork [3] %121 {handshake.bb = 3 : ui32, handshake.name = "fork21"} : <i32>
    %123 = source {handshake.bb = 3 : ui32, handshake.name = "source6"} : <>
    %124 = constant %123 {handshake.bb = 3 : ui32, handshake.name = "constant24", value = 2 : i3} : <>, <i3>
    %125 = extsi %124 {handshake.bb = 3 : ui32, handshake.name = "extsi11"} : <i3> to <i32>
    %126:3 = fork [3] %125 {handshake.bb = 3 : ui32, handshake.name = "fork22"} : <i32>
    %127 = shli %106#0, %126#0 {handshake.bb = 3 : ui32, handshake.name = "shli0"} : <i32>
    %128 = shli %129, %122#0 {handshake.bb = 3 : ui32, handshake.name = "shli1"} : <i32>
    %129 = buffer %106#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer44"} : <i32>
    %130 = buffer %127, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer47"} : <i32>
    %131 = buffer %128, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer48"} : <i32>
    %132 = addi %130, %131 {handshake.bb = 3 : ui32, handshake.name = "addi9"} : <i32>
    %133 = buffer %132, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer49"} : <i32>
    %134 = addi %135, %133 {handshake.bb = 3 : ui32, handshake.name = "addi3"} : <i32>
    %135 = buffer %91#4, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer45"} : <i32>
    %136 = buffer %79, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer35"} : <>
    %137 = gate %134, %136 {handshake.bb = 3 : ui32, handshake.name = "gate0"} : <i32>, !handshake.control<> to <i32>
    %138 = trunci %137 {handshake.bb = 3 : ui32, handshake.name = "trunci3"} : <i32> to <i9>
    %addressResult, %dataResult = load[%138] %outputs#0 {handshake.bb = 3 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["store0", 0, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i9>, <i32>, <i9>, <i32>
    %addressResult_22, %dataResult_23 = load[%100] %outputs_0 {handshake.bb = 3 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i5>, <i32>, <i5>, <i32>
    %139 = shli %99#0, %126#1 {handshake.bb = 3 : ui32, handshake.name = "shli2"} : <i32>
    %140 = shli %99#1, %122#1 {handshake.bb = 3 : ui32, handshake.name = "shli3"} : <i32>
    %141 = buffer %139, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer50"} : <i32>
    %142 = buffer %140, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer51"} : <i32>
    %143 = addi %141, %142 {handshake.bb = 3 : ui32, handshake.name = "addi10"} : <i32>
    %144 = buffer %143, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer52"} : <i32>
    %145 = addi %91#3, %144 {handshake.bb = 3 : ui32, handshake.name = "addi4"} : <i32>
    %146 = buffer %77, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer34"} : <>
    %147 = gate %145, %146 {handshake.bb = 3 : ui32, handshake.name = "gate1"} : <i32>, !handshake.control<> to <i32>
    %148 = trunci %147 {handshake.bb = 3 : ui32, handshake.name = "trunci4"} : <i32> to <i9>
    %addressResult_24, %dataResult_25 = load[%148] %outputs#1 {handshake.bb = 3 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["store0", 0, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load2"} : <i9>, <i32>, <i9>, <i32>
    %149 = muli %dataResult_23, %dataResult_25 {handshake.bb = 3 : ui32, handshake.name = "muli0"} : <i32>
    %150 = subi %dataResult, %149 {handshake.bb = 3 : ui32, handshake.name = "subi0"} : <i32>
    %151 = shli %106#2, %126#2 {handshake.bb = 3 : ui32, handshake.name = "shli4"} : <i32>
    %152 = buffer %151, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer53"} : <i32>
    %153 = trunci %152 {handshake.bb = 3 : ui32, handshake.name = "trunci5"} : <i32> to <i9>
    %154 = shli %106#3, %122#2 {handshake.bb = 3 : ui32, handshake.name = "shli5"} : <i32>
    %155 = buffer %154, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer54"} : <i32>
    %156 = trunci %155 {handshake.bb = 3 : ui32, handshake.name = "trunci6"} : <i32> to <i9>
    %157 = addi %153, %156 {handshake.bb = 3 : ui32, handshake.name = "addi11"} : <i9>
    %158 = buffer %157, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer55"} : <i9>
    %159 = addi %92, %158 {handshake.bb = 3 : ui32, handshake.name = "addi5"} : <i9>
    %160 = buffer %doneResult, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer56"} : <>
    %161 = buffer %160, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer0"} : <>
    %addressResult_26, %dataResult_27, %doneResult = store[%159] %150 %outputs#2 {handshake.bb = 3 : ui32, handshake.bufProps = #handshake<bufProps{"2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["load0", 0, false], ["load2", 0, false], ["store0", 0, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store0"} : <i9>, <i32>, <>, <i9>, <i32>, <>
    %162 = buffer %93, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer39"} : <i32>
    %163 = addi %162, %91#2 {handshake.bb = 3 : ui32, handshake.name = "addi0"} : <i32>
    %164 = addi %91#1, %118 {handshake.bb = 3 : ui32, handshake.name = "addi1"} : <i32>
    %165 = addi %86, %117 {handshake.bb = 3 : ui32, handshake.name = "addi6"} : <i7>
    %166 = buffer %165, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer61"} : <i7>
    %167:2 = fork [2] %166 {handshake.bb = 3 : ui32, handshake.name = "fork23"} : <i7>
    %168 = trunci %169 {handshake.bb = 3 : ui32, handshake.name = "trunci7"} : <i7> to <i6>
    %169 = buffer %167#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer57"} : <i7>
    %170 = cmpi ult, %171, %113 {handshake.bb = 3 : ui32, handshake.name = "cmpi0"} : <i7>
    %171 = buffer %167#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer58"} : <i7>
    %172 = buffer %170, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer62"} : <i1>
    %173:8 = fork [8] %172 {handshake.bb = 3 : ui32, handshake.name = "fork24"} : <i1>
    %trueResult_28, %falseResult_29 = cond_br %173#0, %168 {handshake.bb = 3 : ui32, handshake.name = "cond_br9"} : <i1>, <i6>
    sink %falseResult_29 {handshake.name = "sink3"} : <i6>
    %trueResult_30, %falseResult_31 = cond_br %174, %164 {handshake.bb = 3 : ui32, handshake.name = "cond_br10"} : <i1>, <i32>
    %174 = buffer %173#5, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer60"} : <i1>
    sink %falseResult_31 {handshake.name = "sink4"} : <i32>
    %175 = buffer %163, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer59"} : <i32>
    %trueResult_32, %falseResult_33 = cond_br %173#6, %175 {handshake.bb = 3 : ui32, handshake.name = "cond_br11"} : <i1>, <i32>
    %trueResult_34, %falseResult_35 = cond_br %173#1, %97#1 {handshake.bb = 3 : ui32, handshake.name = "cond_br12"} : <i1>, <i6>
    %trueResult_36, %falseResult_37 = cond_br %173#2, %176 {handshake.bb = 3 : ui32, handshake.name = "cond_br13"} : <i1>, <i6>
    %176 = buffer %104#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer65"} : <i6>
    %177 = buffer %108#1, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer46"} : <>
    %trueResult_38, %falseResult_39 = cond_br %173#7, %177 {handshake.bb = 3 : ui32, handshake.name = "cond_br14"} : <i1>, <>
    %178 = merge %falseResult_35 {handshake.bb = 4 : ui32, handshake.name = "merge0"} : <i6>
    %179 = merge %falseResult_37 {handshake.bb = 4 : ui32, handshake.name = "merge1"} : <i6>
    %180 = extsi %179 {handshake.bb = 4 : ui32, handshake.name = "extsi27"} : <i6> to <i7>
    %181 = merge %falseResult_33 {handshake.bb = 4 : ui32, handshake.name = "merge2"} : <i32>
    %result_40, %index_41 = control_merge [%falseResult_39]  {handshake.bb = 4 : ui32, handshake.name = "control_merge3"} : [<>] to <>, <i1>
    sink %index_41 {handshake.name = "sink5"} : <i1>
    %182 = source {handshake.bb = 4 : ui32, handshake.name = "source7"} : <>
    %183 = constant %182 {handshake.bb = 4 : ui32, handshake.name = "constant25", value = 1 : i2} : <>, <i2>
    %184 = extsi %183 {handshake.bb = 4 : ui32, handshake.name = "extsi28"} : <i2> to <i7>
    %185 = addi %180, %184 {handshake.bb = 4 : ui32, handshake.name = "addi8"} : <i7>
    %186 = buffer %185, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer63"} : <i7>
    %187 = br %186 {handshake.bb = 4 : ui32, handshake.name = "br10"} : <i7>
    %188 = br %181 {handshake.bb = 4 : ui32, handshake.name = "br11"} : <i32>
    %189 = br %178 {handshake.bb = 4 : ui32, handshake.name = "br12"} : <i6>
    %190 = br %result_40 {handshake.bb = 4 : ui32, handshake.name = "br13"} : <>
    %trueResult_42, %falseResult_43 = cond_br %191, %falseResult_17 {handshake.bb = 5 : ui32, handshake.name = "cond_br28"} : <i1>, <>
    %191 = buffer %209#2, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 5 : ui32, handshake.name = "buffer67"} : <i1>
    sink %falseResult_43 {handshake.name = "sink6"} : <>
    %trueResult_44, %falseResult_45 = cond_br %192, %falseResult_15 {handshake.bb = 5 : ui32, handshake.name = "cond_br29"} : <i1>, <>
    %192 = buffer %209#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 5 : ui32, handshake.name = "buffer68"} : <i1>
    sink %falseResult_45 {handshake.name = "sink7"} : <>
    %193 = merge %falseResult_9 {handshake.bb = 5 : ui32, handshake.name = "merge3"} : <i6>
    %194 = extsi %193 {handshake.bb = 5 : ui32, handshake.name = "extsi29"} : <i6> to <i7>
    %195 = merge %falseResult_7 {handshake.bb = 5 : ui32, handshake.name = "merge4"} : <i32>
    %result_46, %index_47 = control_merge [%falseResult_13]  {handshake.bb = 5 : ui32, handshake.name = "control_merge4"} : [<>] to <>, <i1>
    sink %index_47 {handshake.name = "sink8"} : <i1>
    %196 = source {handshake.bb = 5 : ui32, handshake.name = "source8"} : <>
    %197 = constant %196 {handshake.bb = 5 : ui32, handshake.name = "constant26", value = 19 : i6} : <>, <i6>
    %198 = extsi %197 {handshake.bb = 5 : ui32, handshake.name = "extsi30"} : <i6> to <i7>
    %199 = source {handshake.bb = 5 : ui32, handshake.name = "source9"} : <>
    %200 = constant %199 {handshake.bb = 5 : ui32, handshake.name = "constant27", value = 1 : i2} : <>, <i2>
    %201 = extsi %200 {handshake.bb = 5 : ui32, handshake.name = "extsi31"} : <i2> to <i7>
    %202 = addi %194, %201 {handshake.bb = 5 : ui32, handshake.name = "addi7"} : <i7>
    %203 = buffer %202, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 5 : ui32, handshake.name = "buffer64"} : <i7>
    %204:2 = fork [2] %203 {handshake.bb = 5 : ui32, handshake.name = "fork25"} : <i7>
    %205 = trunci %206 {handshake.bb = 5 : ui32, handshake.name = "trunci8"} : <i7> to <i6>
    %206 = buffer %204#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 5 : ui32, handshake.name = "buffer69"} : <i7>
    %207 = cmpi ult, %204#1, %198 {handshake.bb = 5 : ui32, handshake.name = "cmpi1"} : <i7>
    %208 = buffer %207, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 5 : ui32, handshake.name = "buffer66"} : <i1>
    %209:6 = fork [6] %208 {handshake.bb = 5 : ui32, handshake.name = "fork26"} : <i1>
    %trueResult_48, %falseResult_49 = cond_br %209#0, %205 {handshake.bb = 5 : ui32, handshake.name = "cond_br15"} : <i1>, <i6>
    sink %falseResult_49 {handshake.name = "sink9"} : <i6>
    %trueResult_50, %falseResult_51 = cond_br %210, %195 {handshake.bb = 5 : ui32, handshake.name = "cond_br16"} : <i1>, <i32>
    %210 = buffer %209#4, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 5 : ui32, handshake.name = "buffer72"} : <i1>
    %trueResult_52, %falseResult_53 = cond_br %211, %result_46 {handshake.bb = 5 : ui32, handshake.name = "cond_br17"} : <i1>, <>
    %211 = buffer %209#5, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 5 : ui32, handshake.name = "buffer73"} : <i1>
    %212 = merge %falseResult_51 {handshake.bb = 6 : ui32, handshake.name = "merge5"} : <i32>
    %result_54, %index_55 = control_merge [%falseResult_53]  {handshake.bb = 6 : ui32, handshake.name = "control_merge5"} : [<>] to <>, <i1>
    sink %index_55 {handshake.name = "sink10"} : <i1>
    %213:2 = fork [2] %result_54 {handshake.bb = 6 : ui32, handshake.name = "fork27"} : <>
    end {handshake.bb = 6 : ui32, handshake.name = "end0"} %212, %memEnd_1, %memEnd, %0#2 : <i32>, <>, <>, <>
  }
}

