module {
  handshake.func @triangular(%arg0: memref<10xi32>, %arg1: !handshake.channel<i32>, %arg2: memref<100xi32>, %arg3: !handshake.control<>, %arg4: !handshake.control<>, %arg5: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["x", "n", "a", "x_start", "a_start", "start"], cfg.edges = "[0,1][2,3][4,3][1,2,6,cmpi0][3,4,5,cmpi1][5,1]", resNames = ["x_end", "a_end", "end"]} {
    %0:4 = fork [4] %arg5 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork0"} : <>
    %outputs:3, %memEnd = mem_controller[%arg2 : memref<100xi32>] %arg4 (%91, %addressResult, %addressResult_30, %addressResult_32, %dataResult_33) %201#1 {connectedBlocks = [4 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "5": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "6": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller2"} :    (!handshake.channel<i32>, !handshake.channel<i7>, !handshake.channel<i7>, !handshake.channel<i7>, !handshake.channel<i32>) -> (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>)
    %outputs_0, %memEnd_1 = mem_controller[%arg0 : memref<10xi32>] %arg3 (%addressResult_28) %201#0 {connectedBlocks = [4 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller3"} :    (!handshake.channel<i4>) -> !handshake.channel<i32>
    %1 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant0", value = false} : <>, <i1>
    %2 = br %1 {handshake.bb = 0 : ui32, handshake.name = "br4"} : <i1>
    %3 = extsi %2 {handshake.bb = 0 : ui32, handshake.name = "extsi6"} : <i1> to <i32>
    %4 = br %arg1 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "br5"} : <i32>
    %5 = br %0#3 {handshake.bb = 0 : ui32, handshake.name = "br6"} : <>
    %6 = mux %7 [%0#2, %falseResult_25] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux7"} : <i1>, [<>, <>] to <>
    %7 = init %21#4 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "init0"} : <i1>
    %8 = buffer %198, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer62"} : <i32>
    %9 = mux %17#0 [%3, %8] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i32>, <i32>] to <i32>
    %10 = buffer %9, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer2"} : <i32>
    %11:2 = fork [2] %10 {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <i32>
    %12 = mux %13 [%4, %199] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i1>, [<i32>, <i32>] to <i32>
    %13 = buffer %17#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer3"} : <i1>
    %14 = buffer %12, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer4"} : <i32>
    %15 = buffer %14, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer6"} : <i32>
    %16:2 = fork [2] %15 {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <i32>
    %result, %index = control_merge [%5, %200]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %17:2 = fork [2] %index {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <i1>
    %18 = cmpi slt, %19, %16#1 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i32>
    %19 = buffer %11#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer5"} : <i32>
    %20 = buffer %18, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer7"} : <i1>
    %21:5 = fork [5] %20 {handshake.bb = 1 : ui32, handshake.name = "fork4"} : <i1>
    %trueResult, %falseResult = cond_br %21#3, %16#0 {handshake.bb = 1 : ui32, handshake.name = "cond_br2"} : <i1>, <i32>
    sink %falseResult {handshake.name = "sink0"} : <i32>
    %trueResult_2, %falseResult_3 = cond_br %21#2, %11#0 {handshake.bb = 1 : ui32, handshake.name = "cond_br3"} : <i1>, <i32>
    sink %falseResult_3 {handshake.name = "sink1"} : <i32>
    %trueResult_4, %falseResult_5 = cond_br %22, %result {handshake.bb = 1 : ui32, handshake.name = "cond_br4"} : <i1>, <>
    %22 = buffer %21#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer10"} : <i1>
    %23 = merge %trueResult {handshake.bb = 2 : ui32, handshake.name = "merge0"} : <i32>
    %24:2 = fork [2] %23 {handshake.bb = 2 : ui32, handshake.name = "fork5"} : <i32>
    %25 = merge %trueResult_2 {handshake.bb = 2 : ui32, handshake.name = "merge1"} : <i32>
    %26 = buffer %25, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer8"} : <i32>
    %27:2 = fork [2] %26 {handshake.bb = 2 : ui32, handshake.name = "fork6"} : <i32>
    %result_6, %index_7 = control_merge [%trueResult_4]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>] to <>, <i1>
    sink %index_7 {handshake.name = "sink2"} : <i1>
    %28:2 = fork [2] %result_6 {handshake.bb = 2 : ui32, handshake.name = "fork7"} : <>
    %29 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %30 = constant %29 {handshake.bb = 2 : ui32, handshake.name = "constant5", value = -1 : i32} : <>, <i32>
    %31 = constant %28#0 {handshake.bb = 2 : ui32, handshake.name = "constant3", value = false} : <>, <i1>
    %32 = subi %24#1, %27#1 {handshake.bb = 2 : ui32, handshake.name = "subi1"} : <i32>
    %33 = buffer %32, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer9"} : <i32>
    %34:2 = fork [2] %33 {handshake.bb = 2 : ui32, handshake.name = "fork8"} : <i32>
    %35 = addi %34#1, %30 {handshake.bb = 2 : ui32, handshake.name = "addi1"} : <i32>
    %36 = br %31 {handshake.bb = 2 : ui32, handshake.name = "br7"} : <i1>
    %37 = extsi %36 {handshake.bb = 2 : ui32, handshake.name = "extsi7"} : <i1> to <i32>
    %38 = br %24#0 {handshake.bb = 2 : ui32, handshake.name = "br8"} : <i32>
    %39 = br %27#0 {handshake.bb = 2 : ui32, handshake.name = "br9"} : <i32>
    %40 = br %41 {handshake.bb = 2 : ui32, handshake.name = "br10"} : <i32>
    %41 = buffer %34#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer16"} : <i32>
    %42 = br %35 {handshake.bb = 2 : ui32, handshake.name = "br11"} : <i32>
    %43 = br %28#1 {handshake.bb = 2 : ui32, handshake.name = "br12"} : <>
    %44 = buffer %6, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer1"} : <>
    %trueResult_8, %falseResult_9 = cond_br %21#0, %44 {handshake.bb = 3 : ui32, handshake.name = "cond_br12"} : <i1>, <>
    sink %falseResult_9 {handshake.name = "sink3"} : <>
    %45 = mux %46 [%trueResult_8, %182] {ftd.phi, handshake.bb = 3 : ui32, handshake.name = "mux8"} : <i1>, [<>, <>] to <>
    %46 = init %47 {ftd.imerge, handshake.bb = 3 : ui32, handshake.name = "init1"} : <i1>
    %47 = buffer %64#7, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 3 : ui32, handshake.name = "buffer18"} : <i1>
    %48 = mux %49 [%37, %186] {handshake.bb = 3 : ui32, handshake.name = "mux2"} : <i1>, [<i32>, <i32>] to <i32>
    %49 = buffer %60#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer19"} : <i1>
    %50 = buffer %48, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer13"} : <i32>
    %51 = buffer %50, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer14"} : <i32>
    %52:2 = fork [2] %51 {handshake.bb = 3 : ui32, handshake.name = "fork9"} : <i32>
    %53 = mux %60#1 [%38, %187] {handshake.bb = 3 : ui32, handshake.name = "mux3"} : <i1>, [<i32>, <i32>] to <i32>
    %54 = mux %60#2 [%39, %188] {handshake.bb = 3 : ui32, handshake.name = "mux4"} : <i1>, [<i32>, <i32>] to <i32>
    %55 = mux %60#3 [%40, %189] {handshake.bb = 3 : ui32, handshake.name = "mux5"} : <i1>, [<i32>, <i32>] to <i32>
    %56 = mux %60#4 [%42, %190] {handshake.bb = 3 : ui32, handshake.name = "mux6"} : <i1>, [<i32>, <i32>] to <i32>
    %57 = buffer %56, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer23"} : <i32>
    %58 = buffer %57, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer25"} : <i32>
    %59:2 = fork [2] %58 {handshake.bb = 3 : ui32, handshake.name = "fork10"} : <i32>
    %result_10, %index_11 = control_merge [%43, %191]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>, <>] to <>, <i1>
    %60:5 = fork [5] %index_11 {handshake.bb = 3 : ui32, handshake.name = "fork11"} : <i1>
    %61 = cmpi slt, %52#1, %62 {handshake.bb = 3 : ui32, handshake.name = "cmpi1"} : <i32>
    %62 = buffer %59#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer24"} : <i32>
    %63 = buffer %61, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer27"} : <i1>
    %64:8 = fork [8] %63 {handshake.bb = 3 : ui32, handshake.name = "fork12"} : <i1>
    %65 = buffer %53, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer15"} : <i32>
    %66 = buffer %65, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer17"} : <i32>
    %trueResult_12, %falseResult_13 = cond_br %64#6, %66 {handshake.bb = 3 : ui32, handshake.name = "cond_br5"} : <i1>, <i32>
    %67 = buffer %54, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer20"} : <i32>
    %68 = buffer %67, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer21"} : <i32>
    %trueResult_14, %falseResult_15 = cond_br %64#5, %68 {handshake.bb = 3 : ui32, handshake.name = "cond_br6"} : <i1>, <i32>
    %69 = buffer %55, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer22"} : <i32>
    %trueResult_16, %falseResult_17 = cond_br %64#4, %69 {handshake.bb = 3 : ui32, handshake.name = "cond_br7"} : <i1>, <i32>
    sink %falseResult_17 {handshake.name = "sink4"} : <i32>
    %trueResult_18, %falseResult_19 = cond_br %64#3, %59#0 {handshake.bb = 3 : ui32, handshake.name = "cond_br8"} : <i1>, <i32>
    sink %falseResult_19 {handshake.name = "sink5"} : <i32>
    %trueResult_20, %falseResult_21 = cond_br %64#2, %52#0 {handshake.bb = 3 : ui32, handshake.name = "cond_br9"} : <i1>, <i32>
    sink %falseResult_21 {handshake.name = "sink6"} : <i32>
    %70 = buffer %result_10, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer26"} : <>
    %trueResult_22, %falseResult_23 = cond_br %64#1, %70 {handshake.bb = 3 : ui32, handshake.name = "cond_br10"} : <i1>, <>
    %71 = buffer %45, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer11"} : <>
    %72 = buffer %71, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer12"} : <>
    %trueResult_24, %falseResult_25 = cond_br %73, %72 {handshake.bb = 4 : ui32, handshake.name = "cond_br13"} : <i1>, <>
    %73 = buffer %64#0, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 4 : ui32, handshake.name = "buffer34"} : <i1>
    %74 = merge %trueResult_12 {handshake.bb = 4 : ui32, handshake.name = "merge2"} : <i32>
    %75 = buffer %74, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer29"} : <i32>
    %76:6 = fork [6] %75 {handshake.bb = 4 : ui32, handshake.name = "fork13"} : <i32>
    %77 = trunci %78 {handshake.bb = 4 : ui32, handshake.name = "trunci0"} : <i32> to <i7>
    %78 = buffer %76#0, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 4 : ui32, handshake.name = "buffer35"} : <i32>
    %79 = merge %trueResult_14 {handshake.bb = 4 : ui32, handshake.name = "merge3"} : <i32>
    %80:4 = fork [4] %79 {handshake.bb = 4 : ui32, handshake.name = "fork14"} : <i32>
    %81 = buffer %trueResult_16, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer28"} : <i32>
    %82 = merge %81 {handshake.bb = 4 : ui32, handshake.name = "merge4"} : <i32>
    %83:3 = fork [3] %82 {handshake.bb = 4 : ui32, handshake.name = "fork15"} : <i32>
    %84 = trunci %83#0 {handshake.bb = 4 : ui32, handshake.name = "trunci1"} : <i32> to <i7>
    %85 = trunci %83#1 {handshake.bb = 4 : ui32, handshake.name = "trunci2"} : <i32> to <i4>
    %86 = merge %trueResult_18 {handshake.bb = 4 : ui32, handshake.name = "merge5"} : <i32>
    %87 = merge %trueResult_20 {handshake.bb = 4 : ui32, handshake.name = "merge6"} : <i32>
    %88:4 = fork [4] %87 {handshake.bb = 4 : ui32, handshake.name = "fork16"} : <i32>
    %result_26, %index_27 = control_merge [%trueResult_22]  {handshake.bb = 4 : ui32, handshake.name = "control_merge3"} : [<>] to <>, <i1>
    sink %index_27 {handshake.name = "sink7"} : <i1>
    %89:2 = fork [2] %result_26 {handshake.bb = 4 : ui32, handshake.name = "fork17"} : <>
    %90 = constant %89#0 {handshake.bb = 4 : ui32, handshake.name = "constant4", value = 1 : i2} : <>, <i2>
    %91 = extsi %90 {handshake.bb = 4 : ui32, handshake.name = "extsi2"} : <i2> to <i32>
    %92 = source {handshake.bb = 4 : ui32, handshake.name = "source1"} : <>
    %93 = constant %92 {handshake.bb = 4 : ui32, handshake.name = "constant7", value = -2 : i32} : <>, <i32>
    %94:3 = fork [3] %93 {handshake.bb = 4 : ui32, handshake.name = "fork18"} : <i32>
    %95 = source {handshake.bb = 4 : ui32, handshake.name = "source2"} : <>
    %96 = constant %95 {handshake.bb = 4 : ui32, handshake.name = "constant8", value = -1 : i32} : <>, <i32>
    %97:5 = fork [5] %96 {handshake.bb = 4 : ui32, handshake.name = "fork19"} : <i32>
    %98 = trunci %97#0 {handshake.bb = 4 : ui32, handshake.name = "trunci3"} : <i32> to <i7>
    %99 = trunci %97#1 {handshake.bb = 4 : ui32, handshake.name = "trunci4"} : <i32> to <i4>
    %100 = source {handshake.bb = 4 : ui32, handshake.name = "source3"} : <>
    %101 = constant %100 {handshake.bb = 4 : ui32, handshake.name = "constant12", value = 1 : i2} : <>, <i2>
    %102 = extsi %101 {handshake.bb = 4 : ui32, handshake.name = "extsi3"} : <i2> to <i32>
    %103:7 = fork [7] %102 {handshake.bb = 4 : ui32, handshake.name = "fork20"} : <i32>
    %104 = source {handshake.bb = 4 : ui32, handshake.name = "source4"} : <>
    %105 = constant %104 {handshake.bb = 4 : ui32, handshake.name = "constant13", value = 3 : i3} : <>, <i3>
    %106 = extsi %105 {handshake.bb = 4 : ui32, handshake.name = "extsi4"} : <i3> to <i32>
    %107:3 = fork [3] %106 {handshake.bb = 4 : ui32, handshake.name = "fork21"} : <i32>
    %108 = addi %80#3, %88#3 {handshake.bb = 4 : ui32, handshake.name = "addi0"} : <i32>
    %109 = buffer %108, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer30"} : <i32>
    %110 = xori %109, %97#4 {handshake.bb = 4 : ui32, handshake.name = "xori0"} : <i32>
    %111 = addi %110, %103#0 {handshake.bb = 4 : ui32, handshake.name = "addi2"} : <i32>
    %112 = buffer %111, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer31"} : <i32>
    %113 = addi %112, %76#5 {handshake.bb = 4 : ui32, handshake.name = "addi3"} : <i32>
    %114 = buffer %113, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer32"} : <i32>
    %115 = addi %114, %94#2 {handshake.bb = 4 : ui32, handshake.name = "addi4"} : <i32>
    %116 = buffer %115, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer33"} : <i32>
    %117:2 = fork [2] %116 {handshake.bb = 4 : ui32, handshake.name = "fork22"} : <i32>
    %118 = addi %84, %98 {handshake.bb = 4 : ui32, handshake.name = "addi6"} : <i7>
    %119 = shli %117#1, %103#1 {handshake.bb = 4 : ui32, handshake.name = "shli0"} : <i32>
    %120 = buffer %119, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer37"} : <i32>
    %121 = trunci %120 {handshake.bb = 4 : ui32, handshake.name = "trunci5"} : <i32> to <i7>
    %122 = shli %117#0, %107#0 {handshake.bb = 4 : ui32, handshake.name = "shli1"} : <i32>
    %123 = buffer %122, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer38"} : <i32>
    %124 = trunci %123 {handshake.bb = 4 : ui32, handshake.name = "trunci6"} : <i32> to <i7>
    %125 = addi %121, %124 {handshake.bb = 4 : ui32, handshake.name = "addi5"} : <i7>
    %126 = buffer %118, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer36"} : <i7>
    %127 = buffer %125, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer39"} : <i7>
    %128 = addi %126, %127 {handshake.bb = 4 : ui32, handshake.name = "addi15"} : <i7>
    %addressResult, %dataResult = load[%128] %outputs#0 {handshake.bb = 4 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i7>, <i32>, <i7>, <i32>
    %129 = addi %85, %99 {handshake.bb = 4 : ui32, handshake.name = "addi8"} : <i4>
    %addressResult_28, %dataResult_29 = load[%129] %outputs_0 {handshake.bb = 4 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i4>, <i32>, <i4>, <i32>
    %130 = muli %dataResult, %dataResult_29 {handshake.bb = 4 : ui32, handshake.name = "muli0"} : <i32>
    %131 = addi %80#2, %88#2 {handshake.bb = 4 : ui32, handshake.name = "addi7"} : <i32>
    %132 = buffer %131, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer40"} : <i32>
    %133 = xori %132, %97#3 {handshake.bb = 4 : ui32, handshake.name = "xori1"} : <i32>
    %134 = addi %133, %103#2 {handshake.bb = 4 : ui32, handshake.name = "addi9"} : <i32>
    %135 = buffer %134, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer41"} : <i32>
    %136 = addi %135, %137 {handshake.bb = 4 : ui32, handshake.name = "addi10"} : <i32>
    %137 = buffer %76#4, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer54"} : <i32>
    %138 = buffer %136, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer42"} : <i32>
    %139 = addi %138, %94#1 {handshake.bb = 4 : ui32, handshake.name = "addi11"} : <i32>
    %140 = buffer %139, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer43"} : <i32>
    %141:2 = fork [2] %140 {handshake.bb = 4 : ui32, handshake.name = "fork23"} : <i32>
    %142 = shli %143, %144 {handshake.bb = 4 : ui32, handshake.name = "shli2"} : <i32>
    %143 = buffer %141#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer56"} : <i32>
    %144 = buffer %103#3, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer57"} : <i32>
    %145 = shli %146, %147 {handshake.bb = 4 : ui32, handshake.name = "shli3"} : <i32>
    %146 = buffer %141#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer58"} : <i32>
    %147 = buffer %107#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer59"} : <i32>
    %148 = buffer %142, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer44"} : <i32>
    %149 = buffer %145, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer45"} : <i32>
    %150 = addi %148, %149 {handshake.bb = 4 : ui32, handshake.name = "addi12"} : <i32>
    %151 = buffer %150, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer46"} : <i32>
    %152 = addi %153, %151 {handshake.bb = 4 : ui32, handshake.name = "addi16"} : <i32>
    %153 = buffer %76#3, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer60"} : <i32>
    %154 = gate %152, %trueResult_24 {handshake.bb = 4 : ui32, handshake.name = "gate0"} : <i32>, !handshake.control<> to <i32>
    %155 = trunci %154 {handshake.bb = 4 : ui32, handshake.name = "trunci7"} : <i32> to <i7>
    %addressResult_30, %dataResult_31 = load[%155] %outputs#1 {handshake.bb = 4 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["store0", 1, false], ["store0", 3, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load2"} : <i7>, <i32>, <i7>, <i32>
    %156 = subi %dataResult_31, %130 {handshake.bb = 4 : ui32, handshake.name = "subi0"} : <i32>
    %157 = addi %80#1, %88#1 {handshake.bb = 4 : ui32, handshake.name = "addi20"} : <i32>
    %158 = buffer %157, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer47"} : <i32>
    %159 = xori %158, %97#2 {handshake.bb = 4 : ui32, handshake.name = "xori2"} : <i32>
    %160 = addi %159, %103#4 {handshake.bb = 4 : ui32, handshake.name = "addi21"} : <i32>
    %161 = buffer %160, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer48"} : <i32>
    %162 = addi %161, %76#2 {handshake.bb = 4 : ui32, handshake.name = "addi13"} : <i32>
    %163 = buffer %162, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer49"} : <i32>
    %164 = addi %163, %165 {handshake.bb = 4 : ui32, handshake.name = "addi14"} : <i32>
    %165 = buffer %94#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer66"} : <i32>
    %166 = buffer %164, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer50"} : <i32>
    %167:2 = fork [2] %166 {handshake.bb = 4 : ui32, handshake.name = "fork24"} : <i32>
    %168 = shli %169, %170 {handshake.bb = 4 : ui32, handshake.name = "shli4"} : <i32>
    %169 = buffer %167#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer67"} : <i32>
    %170 = buffer %103#5, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 4 : ui32, handshake.name = "buffer68"} : <i32>
    %171 = buffer %168, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer51"} : <i32>
    %172 = trunci %171 {handshake.bb = 4 : ui32, handshake.name = "trunci8"} : <i32> to <i7>
    %173 = shli %174, %175 {handshake.bb = 4 : ui32, handshake.name = "shli5"} : <i32>
    %174 = buffer %167#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer69"} : <i32>
    %175 = buffer %107#2, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 4 : ui32, handshake.name = "buffer70"} : <i32>
    %176 = buffer %173, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer52"} : <i32>
    %177 = trunci %176 {handshake.bb = 4 : ui32, handshake.name = "trunci9"} : <i32> to <i7>
    %178 = addi %172, %177 {handshake.bb = 4 : ui32, handshake.name = "addi22"} : <i7>
    %179 = buffer %178, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer53"} : <i7>
    %180 = addi %77, %179 {handshake.bb = 4 : ui32, handshake.name = "addi17"} : <i7>
    %181 = buffer %doneResult, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer55"} : <>
    %182 = buffer %181, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer0"} : <>
    %addressResult_32, %dataResult_33, %doneResult = store[%180] %156 %outputs#2 {handshake.bb = 4 : ui32, handshake.bufProps = #handshake<bufProps{"2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["load2", 1, false], ["store0", 1, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store0"} : <i7>, <i32>, <>, <i7>, <i32>, <>
    %183 = addi %88#0, %184 {handshake.bb = 4 : ui32, handshake.name = "addi18"} : <i32>
    %184 = buffer %103#6, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer71"} : <i32>
    %185 = buffer %183, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer61"} : <i32>
    %186 = br %185 {handshake.bb = 4 : ui32, handshake.name = "br13"} : <i32>
    %187 = br %76#1 {handshake.bb = 4 : ui32, handshake.name = "br14"} : <i32>
    %188 = br %80#0 {handshake.bb = 4 : ui32, handshake.name = "br15"} : <i32>
    %189 = br %83#2 {handshake.bb = 4 : ui32, handshake.name = "br16"} : <i32>
    %190 = br %86 {handshake.bb = 4 : ui32, handshake.name = "br17"} : <i32>
    %191 = br %89#1 {handshake.bb = 4 : ui32, handshake.name = "br18"} : <>
    %192 = merge %falseResult_13 {handshake.bb = 5 : ui32, handshake.name = "merge7"} : <i32>
    %193 = merge %falseResult_15 {handshake.bb = 5 : ui32, handshake.name = "merge8"} : <i32>
    %result_34, %index_35 = control_merge [%falseResult_23]  {handshake.bb = 5 : ui32, handshake.name = "control_merge4"} : [<>] to <>, <i1>
    sink %index_35 {handshake.name = "sink8"} : <i1>
    %194 = source {handshake.bb = 5 : ui32, handshake.name = "source5"} : <>
    %195 = constant %194 {handshake.bb = 5 : ui32, handshake.name = "constant14", value = 1 : i2} : <>, <i2>
    %196 = extsi %195 {handshake.bb = 5 : ui32, handshake.name = "extsi5"} : <i2> to <i32>
    %197 = addi %193, %196 {handshake.bb = 5 : ui32, handshake.name = "addi19"} : <i32>
    %198 = br %197 {handshake.bb = 5 : ui32, handshake.name = "br19"} : <i32>
    %199 = br %192 {handshake.bb = 5 : ui32, handshake.name = "br20"} : <i32>
    %200 = br %result_34 {handshake.bb = 5 : ui32, handshake.name = "br21"} : <>
    %result_36, %index_37 = control_merge [%falseResult_5]  {handshake.bb = 6 : ui32, handshake.name = "control_merge5"} : [<>] to <>, <i1>
    sink %index_37 {handshake.name = "sink9"} : <i1>
    %201:2 = fork [2] %result_36 {handshake.bb = 6 : ui32, handshake.name = "fork25"} : <>
    end {handshake.bb = 6 : ui32, handshake.name = "end0"} %memEnd_1, %memEnd, %0#1 : <>, <>, <>
  }
}

