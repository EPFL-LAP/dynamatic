module {
  handshake.func @triangular(%arg0: memref<10xi32>, %arg1: !handshake.channel<i32>, %arg2: memref<100xi32>, %arg3: !handshake.control<>, %arg4: !handshake.control<>, %arg5: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["x", "n", "a", "x_start", "a_start", "start"], resNames = ["x_end", "a_end", "end"]} {
    %0:3 = fork [3] %arg5 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork0"} : <>
    %outputs:2, %memEnd = mem_controller[%arg2 : memref<100xi32>] %arg4 (%81, %addressResult, %1#1, %1#2, %1#3) %181#1 {connectedBlocks = [4 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "5": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "6": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller2"} :    (!handshake.channel<i32>, !handshake.channel<i7>, !handshake.channel<i7>, !handshake.channel<i7>, !handshake.channel<i32>) -> (!handshake.channel<i32>, !handshake.channel<i32>)
    %1:4 = lsq[MC] (%78#0, %addressResult_26, %addressResult_28, %dataResult_29, %outputs#1)  {groupSizes = [2 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "lsq1"} : (!handshake.control<>, !handshake.channel<i7>, !handshake.channel<i7>, !handshake.channel<i32>, !handshake.channel<i32>) -> (!handshake.channel<i32>, !handshake.channel<i7>, !handshake.channel<i7>, !handshake.channel<i32>)
    %outputs_0, %memEnd_1 = mem_controller[%arg0 : memref<10xi32>] %arg3 (%addressResult_24) %181#0 {connectedBlocks = [4 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller3"} :    (!handshake.channel<i4>) -> !handshake.channel<i32>
    %2 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant3", value = false} : <>, <i1>
    %3 = br %2 {handshake.bb = 0 : ui32, handshake.name = "br4"} : <i1>
    %4 = extsi %3 {handshake.bb = 0 : ui32, handshake.name = "extsi6"} : <i1> to <i32>
    %5 = br %arg1 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "br5"} : <i32>
    %6 = br %0#2 {handshake.bb = 0 : ui32, handshake.name = "br6"} : <>
    %7 = buffer %178, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer50"} : <i32>
    %8 = mux %17#0 [%4, %7] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i32>, <i32>] to <i32>
    %9 = buffer %8, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer0"} : <i32>
    %10 = buffer %9, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer2"} : <i32>
    %11:2 = fork [2] %10 {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <i32>
    %12 = mux %13 [%5, %179] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i1>, [<i32>, <i32>] to <i32>
    %13 = buffer %17#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer1"} : <i1>
    %14 = buffer %12, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer4"} : <i32>
    %15 = buffer %14, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer5"} : <i32>
    %16:2 = fork [2] %15 {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <i32>
    %result, %index = control_merge [%6, %180]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %17:2 = fork [2] %index {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <i1>
    %18 = cmpi slt, %19, %16#1 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i32>
    %19 = buffer %11#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer3"} : <i32>
    %20 = buffer %18, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer6"} : <i1>
    %21:3 = fork [3] %20 {handshake.bb = 1 : ui32, handshake.name = "fork4"} : <i1>
    %trueResult, %falseResult = cond_br %21#2, %16#0 {handshake.bb = 1 : ui32, handshake.name = "cond_br2"} : <i1>, <i32>
    sink %falseResult {handshake.name = "sink0"} : <i32>
    %trueResult_2, %falseResult_3 = cond_br %21#1, %22 {handshake.bb = 1 : ui32, handshake.name = "cond_br3"} : <i1>, <i32>
    %22 = buffer %11#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer7"} : <i32>
    sink %falseResult_3 {handshake.name = "sink1"} : <i32>
    %trueResult_4, %falseResult_5 = cond_br %23, %result {handshake.bb = 1 : ui32, handshake.name = "cond_br4"} : <i1>, <>
    %23 = buffer %21#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer8"} : <i1>
    %24 = merge %trueResult {handshake.bb = 2 : ui32, handshake.name = "merge0"} : <i32>
    %25:2 = fork [2] %24 {handshake.bb = 2 : ui32, handshake.name = "fork5"} : <i32>
    %26 = merge %trueResult_2 {handshake.bb = 2 : ui32, handshake.name = "merge1"} : <i32>
    %27:2 = fork [2] %26 {handshake.bb = 2 : ui32, handshake.name = "fork6"} : <i32>
    %result_6, %index_7 = control_merge [%trueResult_4]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>] to <>, <i1>
    sink %index_7 {handshake.name = "sink2"} : <i1>
    %28:2 = fork [2] %result_6 {handshake.bb = 2 : ui32, handshake.name = "fork7"} : <>
    %29 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %30 = constant %29 {handshake.bb = 2 : ui32, handshake.name = "constant5", value = -1 : i32} : <>, <i32>
    %31 = constant %28#0 {handshake.bb = 2 : ui32, handshake.name = "constant4", value = false} : <>, <i1>
    %32 = subi %25#1, %27#1 {handshake.bb = 2 : ui32, handshake.name = "subi1"} : <i32>
    %33 = buffer %32, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer9"} : <i32>
    %34:2 = fork [2] %33 {handshake.bb = 2 : ui32, handshake.name = "fork8"} : <i32>
    %35 = addi %34#1, %30 {handshake.bb = 2 : ui32, handshake.name = "addi1"} : <i32>
    %36 = br %31 {handshake.bb = 2 : ui32, handshake.name = "br7"} : <i1>
    %37 = extsi %36 {handshake.bb = 2 : ui32, handshake.name = "extsi7"} : <i1> to <i32>
    %38 = br %25#0 {handshake.bb = 2 : ui32, handshake.name = "br8"} : <i32>
    %39 = br %27#0 {handshake.bb = 2 : ui32, handshake.name = "br9"} : <i32>
    %40 = br %34#0 {handshake.bb = 2 : ui32, handshake.name = "br10"} : <i32>
    %41 = br %35 {handshake.bb = 2 : ui32, handshake.name = "br11"} : <i32>
    %42 = br %28#1 {handshake.bb = 2 : ui32, handshake.name = "br12"} : <>
    %43 = mux %54#0 [%37, %165] {handshake.bb = 3 : ui32, handshake.name = "mux2"} : <i1>, [<i32>, <i32>] to <i32>
    %44 = buffer %43, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer10"} : <i32>
    %45 = buffer %44, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer11"} : <i32>
    %46:2 = fork [2] %45 {handshake.bb = 3 : ui32, handshake.name = "fork9"} : <i32>
    %47 = mux %54#1 [%38, %166] {handshake.bb = 3 : ui32, handshake.name = "mux3"} : <i1>, [<i32>, <i32>] to <i32>
    %48 = mux %54#2 [%39, %167] {handshake.bb = 3 : ui32, handshake.name = "mux4"} : <i1>, [<i32>, <i32>] to <i32>
    %49 = mux %54#3 [%40, %168] {handshake.bb = 3 : ui32, handshake.name = "mux5"} : <i1>, [<i32>, <i32>] to <i32>
    %50 = mux %54#4 [%41, %169] {handshake.bb = 3 : ui32, handshake.name = "mux6"} : <i1>, [<i32>, <i32>] to <i32>
    %51 = buffer %50, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer17"} : <i32>
    %52 = buffer %51, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer18"} : <i32>
    %53:2 = fork [2] %52 {handshake.bb = 3 : ui32, handshake.name = "fork10"} : <i32>
    %result_8, %index_9 = control_merge [%42, %171]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>, <>] to <>, <i1>
    %54:5 = fork [5] %index_9 {handshake.bb = 3 : ui32, handshake.name = "fork11"} : <i1>
    %55 = cmpi slt, %46#1, %53#1 {handshake.bb = 3 : ui32, handshake.name = "cmpi1"} : <i32>
    %56 = buffer %55, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer20"} : <i1>
    %57:6 = fork [6] %56 {handshake.bb = 3 : ui32, handshake.name = "fork12"} : <i1>
    %58 = buffer %47, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer12"} : <i32>
    %59 = buffer %58, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer13"} : <i32>
    %trueResult_10, %falseResult_11 = cond_br %57#5, %59 {handshake.bb = 3 : ui32, handshake.name = "cond_br5"} : <i1>, <i32>
    %60 = buffer %48, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer14"} : <i32>
    %61 = buffer %60, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer15"} : <i32>
    %trueResult_12, %falseResult_13 = cond_br %57#4, %61 {handshake.bb = 3 : ui32, handshake.name = "cond_br6"} : <i1>, <i32>
    %62 = buffer %49, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer16"} : <i32>
    %trueResult_14, %falseResult_15 = cond_br %57#3, %62 {handshake.bb = 3 : ui32, handshake.name = "cond_br7"} : <i1>, <i32>
    sink %falseResult_15 {handshake.name = "sink3"} : <i32>
    %trueResult_16, %falseResult_17 = cond_br %57#2, %53#0 {handshake.bb = 3 : ui32, handshake.name = "cond_br8"} : <i1>, <i32>
    sink %falseResult_17 {handshake.name = "sink4"} : <i32>
    %trueResult_18, %falseResult_19 = cond_br %57#1, %46#0 {handshake.bb = 3 : ui32, handshake.name = "cond_br9"} : <i1>, <i32>
    sink %falseResult_19 {handshake.name = "sink5"} : <i32>
    %63 = buffer %result_8, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer19"} : <>
    %trueResult_20, %falseResult_21 = cond_br %57#0, %63 {handshake.bb = 3 : ui32, handshake.name = "cond_br10"} : <i1>, <>
    %64 = merge %trueResult_10 {handshake.bb = 4 : ui32, handshake.name = "merge2"} : <i32>
    %65:6 = fork [6] %64 {handshake.bb = 4 : ui32, handshake.name = "fork13"} : <i32>
    %66 = trunci %65#0 {handshake.bb = 4 : ui32, handshake.name = "trunci0"} : <i32> to <i7>
    %67 = trunci %65#1 {handshake.bb = 4 : ui32, handshake.name = "trunci1"} : <i32> to <i7>
    %68 = merge %trueResult_12 {handshake.bb = 4 : ui32, handshake.name = "merge3"} : <i32>
    %69:4 = fork [4] %68 {handshake.bb = 4 : ui32, handshake.name = "fork14"} : <i32>
    %70 = buffer %trueResult_14, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer21"} : <i32>
    %71 = merge %70 {handshake.bb = 4 : ui32, handshake.name = "merge4"} : <i32>
    %72:3 = fork [3] %71 {handshake.bb = 4 : ui32, handshake.name = "fork15"} : <i32>
    %73 = trunci %72#0 {handshake.bb = 4 : ui32, handshake.name = "trunci2"} : <i32> to <i7>
    %74 = trunci %72#1 {handshake.bb = 4 : ui32, handshake.name = "trunci3"} : <i32> to <i4>
    %75 = merge %trueResult_16 {handshake.bb = 4 : ui32, handshake.name = "merge5"} : <i32>
    %76 = merge %trueResult_18 {handshake.bb = 4 : ui32, handshake.name = "merge6"} : <i32>
    %77:4 = fork [4] %76 {handshake.bb = 4 : ui32, handshake.name = "fork16"} : <i32>
    %result_22, %index_23 = control_merge [%trueResult_20]  {handshake.bb = 4 : ui32, handshake.name = "control_merge3"} : [<>] to <>, <i1>
    sink %index_23 {handshake.name = "sink6"} : <i1>
    %78:3 = fork [3] %result_22 {handshake.bb = 4 : ui32, handshake.name = "fork17"} : <>
    %79 = buffer %78#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer22"} : <>
    %80 = constant %79 {handshake.bb = 4 : ui32, handshake.bufProps = #handshake<bufProps{"0": [1,inf], [0,inf], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "constant12", value = 1 : i2} : <>, <i2>
    %81 = extsi %80 {handshake.bb = 4 : ui32, handshake.name = "extsi2"} : <i2> to <i32>
    %82 = source {handshake.bb = 4 : ui32, handshake.name = "source1"} : <>
    %83 = constant %82 {handshake.bb = 4 : ui32, handshake.name = "constant7", value = -2 : i32} : <>, <i32>
    %84:3 = fork [3] %83 {handshake.bb = 4 : ui32, handshake.name = "fork18"} : <i32>
    %85 = source {handshake.bb = 4 : ui32, handshake.name = "source2"} : <>
    %86 = constant %85 {handshake.bb = 4 : ui32, handshake.name = "constant8", value = -1 : i32} : <>, <i32>
    %87:5 = fork [5] %86 {handshake.bb = 4 : ui32, handshake.name = "fork19"} : <i32>
    %88 = trunci %87#0 {handshake.bb = 4 : ui32, handshake.name = "trunci4"} : <i32> to <i7>
    %89 = trunci %87#1 {handshake.bb = 4 : ui32, handshake.name = "trunci5"} : <i32> to <i4>
    %90 = source {handshake.bb = 4 : ui32, handshake.name = "source3"} : <>
    %91 = constant %90 {handshake.bb = 4 : ui32, handshake.name = "constant13", value = 1 : i2} : <>, <i2>
    %92 = extsi %91 {handshake.bb = 4 : ui32, handshake.name = "extsi3"} : <i2> to <i32>
    %93:7 = fork [7] %92 {handshake.bb = 4 : ui32, handshake.name = "fork20"} : <i32>
    %94 = source {handshake.bb = 4 : ui32, handshake.name = "source4"} : <>
    %95 = constant %94 {handshake.bb = 4 : ui32, handshake.name = "constant14", value = 3 : i3} : <>, <i3>
    %96 = extsi %95 {handshake.bb = 4 : ui32, handshake.name = "extsi4"} : <i3> to <i32>
    %97:3 = fork [3] %96 {handshake.bb = 4 : ui32, handshake.name = "fork21"} : <i32>
    %98 = addi %69#3, %77#3 {handshake.bb = 4 : ui32, handshake.name = "addi0"} : <i32>
    %99 = buffer %98, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer24"} : <i32>
    %100 = xori %99, %87#4 {handshake.bb = 4 : ui32, handshake.name = "xori0"} : <i32>
    %101 = addi %100, %93#0 {handshake.bb = 4 : ui32, handshake.name = "addi2"} : <i32>
    %102 = buffer %101, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer25"} : <i32>
    %103 = addi %102, %65#5 {handshake.bb = 4 : ui32, handshake.name = "addi3"} : <i32>
    %104 = buffer %103, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer26"} : <i32>
    %105 = addi %104, %84#2 {handshake.bb = 4 : ui32, handshake.name = "addi4"} : <i32>
    %106 = buffer %105, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer27"} : <i32>
    %107:2 = fork [2] %106 {handshake.bb = 4 : ui32, handshake.name = "fork22"} : <i32>
    %108 = addi %73, %88 {handshake.bb = 4 : ui32, handshake.name = "addi6"} : <i7>
    %109 = shli %107#1, %93#1 {handshake.bb = 4 : ui32, handshake.name = "shli0"} : <i32>
    %110 = buffer %109, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer29"} : <i32>
    %111 = trunci %110 {handshake.bb = 4 : ui32, handshake.name = "trunci6"} : <i32> to <i7>
    %112 = shli %107#0, %97#0 {handshake.bb = 4 : ui32, handshake.name = "shli1"} : <i32>
    %113 = buffer %112, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer30"} : <i32>
    %114 = trunci %113 {handshake.bb = 4 : ui32, handshake.name = "trunci7"} : <i32> to <i7>
    %115 = addi %111, %114 {handshake.bb = 4 : ui32, handshake.name = "addi5"} : <i7>
    %116 = buffer %108, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer28"} : <i7>
    %117 = buffer %115, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer31"} : <i7>
    %118 = addi %116, %117 {handshake.bb = 4 : ui32, handshake.name = "addi15"} : <i7>
    %addressResult, %dataResult = load[%118] %outputs#0 {handshake.bb = 4 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i7>, <i32>, <i7>, <i32>
    %119 = addi %74, %89 {handshake.bb = 4 : ui32, handshake.name = "addi8"} : <i4>
    %addressResult_24, %dataResult_25 = load[%119] %outputs_0 {handshake.bb = 4 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i4>, <i32>, <i4>, <i32>
    %120 = muli %dataResult, %dataResult_25 {handshake.bb = 4 : ui32, handshake.name = "muli0"} : <i32>
    %121 = addi %69#2, %77#2 {handshake.bb = 4 : ui32, handshake.name = "addi7"} : <i32>
    %122 = buffer %121, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer32"} : <i32>
    %123 = xori %122, %87#3 {handshake.bb = 4 : ui32, handshake.name = "xori1"} : <i32>
    %124 = addi %123, %93#2 {handshake.bb = 4 : ui32, handshake.name = "addi9"} : <i32>
    %125 = buffer %124, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer33"} : <i32>
    %126 = addi %125, %65#4 {handshake.bb = 4 : ui32, handshake.name = "addi10"} : <i32>
    %127 = buffer %126, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer34"} : <i32>
    %128 = addi %127, %84#1 {handshake.bb = 4 : ui32, handshake.name = "addi11"} : <i32>
    %129 = buffer %128, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer35"} : <i32>
    %130:2 = fork [2] %129 {handshake.bb = 4 : ui32, handshake.name = "fork23"} : <i32>
    %131 = shli %130#1, %93#3 {handshake.bb = 4 : ui32, handshake.name = "shli2"} : <i32>
    %132 = buffer %131, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer36"} : <i32>
    %133 = trunci %132 {handshake.bb = 4 : ui32, handshake.name = "trunci8"} : <i32> to <i7>
    %134 = shli %130#0, %97#1 {handshake.bb = 4 : ui32, handshake.name = "shli3"} : <i32>
    %135 = buffer %134, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer37"} : <i32>
    %136 = trunci %135 {handshake.bb = 4 : ui32, handshake.name = "trunci9"} : <i32> to <i7>
    %137 = addi %133, %136 {handshake.bb = 4 : ui32, handshake.name = "addi12"} : <i7>
    %138 = buffer %137, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer38"} : <i7>
    %139 = addi %66, %138 {handshake.bb = 4 : ui32, handshake.name = "addi16"} : <i7>
    %140 = buffer %139, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer39"} : <i7>
    %addressResult_26, %dataResult_27 = load[%140] %1#0 {handshake.bb = 4 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["store0", 1], ["store0", 3]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 0>, handshake.name = "load2"} : <i7>, <i32>, <i7>, <i32>
    %141 = subi %dataResult_27, %120 {handshake.bb = 4 : ui32, handshake.name = "subi0"} : <i32>
    %142 = addi %69#1, %77#1 {handshake.bb = 4 : ui32, handshake.name = "addi20"} : <i32>
    %143 = buffer %142, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer41"} : <i32>
    %144 = xori %143, %87#2 {handshake.bb = 4 : ui32, handshake.name = "xori2"} : <i32>
    %145 = addi %144, %93#4 {handshake.bb = 4 : ui32, handshake.name = "addi21"} : <i32>
    %146 = buffer %145, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer42"} : <i32>
    %147 = addi %146, %65#3 {handshake.bb = 4 : ui32, handshake.name = "addi13"} : <i32>
    %148 = buffer %147, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer43"} : <i32>
    %149 = addi %148, %84#0 {handshake.bb = 4 : ui32, handshake.name = "addi14"} : <i32>
    %150 = buffer %149, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer44"} : <i32>
    %151:2 = fork [2] %150 {handshake.bb = 4 : ui32, handshake.name = "fork24"} : <i32>
    %152 = shli %151#1, %93#5 {handshake.bb = 4 : ui32, handshake.name = "shli4"} : <i32>
    %153 = buffer %152, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer45"} : <i32>
    %154 = trunci %153 {handshake.bb = 4 : ui32, handshake.name = "trunci10"} : <i32> to <i7>
    %155 = shli %151#0, %97#2 {handshake.bb = 4 : ui32, handshake.name = "shli5"} : <i32>
    %156 = buffer %155, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer46"} : <i32>
    %157 = trunci %156 {handshake.bb = 4 : ui32, handshake.name = "trunci11"} : <i32> to <i7>
    %158 = addi %154, %157 {handshake.bb = 4 : ui32, handshake.name = "addi22"} : <i7>
    %159 = buffer %158, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer47"} : <i7>
    %160 = addi %67, %159 {handshake.bb = 4 : ui32, handshake.name = "addi17"} : <i7>
    %161 = buffer %141, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer40"} : <i32>
    %162 = buffer %160, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer48"} : <i7>
    %addressResult_28, %dataResult_29 = store[%162] %161 {handshake.bb = 4 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["load2", 1], ["store0", 1]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 0>, handshake.name = "store0"} : <i7>, <i32>, <i7>, <i32>
    %163 = addi %77#0, %93#6 {handshake.bb = 4 : ui32, handshake.name = "addi18"} : <i32>
    %164 = buffer %163, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer49"} : <i32>
    %165 = br %164 {handshake.bb = 4 : ui32, handshake.name = "br13"} : <i32>
    %166 = br %65#2 {handshake.bb = 4 : ui32, handshake.name = "br14"} : <i32>
    %167 = br %69#0 {handshake.bb = 4 : ui32, handshake.name = "br15"} : <i32>
    %168 = br %72#2 {handshake.bb = 4 : ui32, handshake.name = "br16"} : <i32>
    %169 = br %75 {handshake.bb = 4 : ui32, handshake.name = "br17"} : <i32>
    %170 = buffer %78#2, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer23"} : <>
    %171 = br %170 {handshake.bb = 4 : ui32, handshake.bufProps = #handshake<bufProps{"0": [1,inf], [0,inf], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "br18"} : <>
    %172 = merge %falseResult_11 {handshake.bb = 5 : ui32, handshake.name = "merge7"} : <i32>
    %173 = merge %falseResult_13 {handshake.bb = 5 : ui32, handshake.name = "merge8"} : <i32>
    %result_30, %index_31 = control_merge [%falseResult_21]  {handshake.bb = 5 : ui32, handshake.name = "control_merge4"} : [<>] to <>, <i1>
    sink %index_31 {handshake.name = "sink7"} : <i1>
    %174 = source {handshake.bb = 5 : ui32, handshake.name = "source5"} : <>
    %175 = constant %174 {handshake.bb = 5 : ui32, handshake.name = "constant15", value = 1 : i2} : <>, <i2>
    %176 = extsi %175 {handshake.bb = 5 : ui32, handshake.name = "extsi5"} : <i2> to <i32>
    %177 = addi %173, %176 {handshake.bb = 5 : ui32, handshake.name = "addi19"} : <i32>
    %178 = br %177 {handshake.bb = 5 : ui32, handshake.name = "br19"} : <i32>
    %179 = br %172 {handshake.bb = 5 : ui32, handshake.name = "br20"} : <i32>
    %180 = br %result_30 {handshake.bb = 5 : ui32, handshake.name = "br21"} : <>
    %result_32, %index_33 = control_merge [%falseResult_5]  {handshake.bb = 6 : ui32, handshake.name = "control_merge5"} : [<>] to <>, <i1>
    sink %index_33 {handshake.name = "sink8"} : <i1>
    %181:2 = fork [2] %result_32 {handshake.bb = 6 : ui32, handshake.name = "fork25"} : <>
    end {handshake.bb = 6 : ui32, handshake.name = "end0"} %memEnd_1, %memEnd, %0#1 : <>, <>, <>
  }
}

