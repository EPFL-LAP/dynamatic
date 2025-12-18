module {
  handshake.func @triangular(%arg0: memref<10xi32>, %arg1: !handshake.channel<i32>, %arg2: memref<100xi32>, %arg3: !handshake.control<>, %arg4: !handshake.control<>, %arg5: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["x", "n", "a", "x_start", "a_start", "start"], cfg.edges = "[0,1][2,3][4,3][1,2,6,cmpi0][3,4,5,cmpi1][5,1]", resNames = ["x_end", "a_end", "end"]} {
    %0:4 = fork [4] %arg5 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork0"} : <>
    %outputs:3, %memEnd = mem_controller[%arg2 : memref<100xi32>] %arg4 (%75, %addressResult, %addressResult_26, %addressResult_28, %dataResult_29) %174#1 {connectedBlocks = [4 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "5": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "6": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller2"} :    (!handshake.channel<i32>, !handshake.channel<i7>, !handshake.channel<i7>, !handshake.channel<i7>, !handshake.channel<i32>) -> (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>)
    %outputs_0, %memEnd_1 = mem_controller[%arg0 : memref<10xi32>] %arg3 (%addressResult_24) %174#0 {connectedBlocks = [4 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller3"} :    (!handshake.channel<i4>) -> !handshake.channel<i32>
    %1 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant0", value = false} : <>, <i1>
    %2 = extsi %1 {handshake.bb = 0 : ui32, handshake.name = "extsi6"} : <i1> to <i32>
    %3 = mux %4 [%0#2, %falseResult_23] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux7"} : <i1>, [<>, <>] to <>
    %4 = init %18#4 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "init0"} : <i1>
    %5 = buffer %173, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer62"} : <i32>
    %6 = mux %14#0 [%2, %5] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i32>, <i32>] to <i32>
    %7 = buffer %6, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer2"} : <i32>
    %8:2 = fork [2] %7 {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <i32>
    %9 = mux %10 [%arg1, %falseResult_11] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i1>, [<i32>, <i32>] to <i32>
    %10 = buffer %14#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer3"} : <i1>
    %11 = buffer %9, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer4"} : <i32>
    %12 = buffer %11, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer6"} : <i32>
    %13:2 = fork [2] %12 {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <i32>
    %result, %index = control_merge [%0#3, %falseResult_21]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %14:2 = fork [2] %index {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <i1>
    %15 = cmpi slt, %16, %13#1 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i32>
    %16 = buffer %8#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer5"} : <i32>
    %17 = buffer %15, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer7"} : <i1>
    %18:5 = fork [5] %17 {handshake.bb = 1 : ui32, handshake.name = "fork4"} : <i1>
    %trueResult, %falseResult = cond_br %18#3, %13#0 {handshake.bb = 1 : ui32, handshake.name = "cond_br2"} : <i1>, <i32>
    sink %falseResult {handshake.name = "sink0"} : <i32>
    %trueResult_2, %falseResult_3 = cond_br %18#2, %8#0 {handshake.bb = 1 : ui32, handshake.name = "cond_br3"} : <i1>, <i32>
    sink %falseResult_3 {handshake.name = "sink1"} : <i32>
    %trueResult_4, %falseResult_5 = cond_br %19, %result {handshake.bb = 1 : ui32, handshake.name = "cond_br4"} : <i1>, <>
    %19 = buffer %18#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer10"} : <i1>
    %20:2 = fork [2] %trueResult {handshake.bb = 2 : ui32, handshake.name = "fork5"} : <i32>
    %21 = buffer %trueResult_2, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer8"} : <i32>
    %22:2 = fork [2] %21 {handshake.bb = 2 : ui32, handshake.name = "fork6"} : <i32>
    %23:2 = fork [2] %trueResult_4 {handshake.bb = 2 : ui32, handshake.name = "fork7"} : <>
    %24 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %25 = constant %24 {handshake.bb = 2 : ui32, handshake.name = "constant5", value = -1 : i32} : <>, <i32>
    %26 = constant %23#0 {handshake.bb = 2 : ui32, handshake.name = "constant3", value = false} : <>, <i1>
    %27 = subi %20#1, %22#1 {handshake.bb = 2 : ui32, handshake.name = "subi1"} : <i32>
    %28 = buffer %27, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer9"} : <i32>
    %29:2 = fork [2] %28 {handshake.bb = 2 : ui32, handshake.name = "fork8"} : <i32>
    %30 = addi %29#1, %25 {handshake.bb = 2 : ui32, handshake.name = "addi1"} : <i32>
    %31 = extsi %26 {handshake.bb = 2 : ui32, handshake.name = "extsi7"} : <i1> to <i32>
    %32 = buffer %29#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer16"} : <i32>
    %33 = buffer %3, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer1"} : <>
    %trueResult_6, %falseResult_7 = cond_br %18#0, %33 {handshake.bb = 3 : ui32, handshake.name = "cond_br12"} : <i1>, <>
    sink %falseResult_7 {handshake.name = "sink3"} : <>
    %34 = mux %35 [%trueResult_6, %166] {ftd.phi, handshake.bb = 3 : ui32, handshake.name = "mux8"} : <i1>, [<>, <>] to <>
    %35 = init %36 {ftd.imerge, handshake.bb = 3 : ui32, handshake.name = "init1"} : <i1>
    %36 = buffer %53#7, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 3 : ui32, handshake.name = "buffer18"} : <i1>
    %37 = mux %38 [%31, %169] {handshake.bb = 3 : ui32, handshake.name = "mux2"} : <i1>, [<i32>, <i32>] to <i32>
    %38 = buffer %49#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer19"} : <i1>
    %39 = buffer %37, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer13"} : <i32>
    %40 = buffer %39, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer14"} : <i32>
    %41:2 = fork [2] %40 {handshake.bb = 3 : ui32, handshake.name = "fork9"} : <i32>
    %42 = mux %49#1 [%20#0, %64#1] {handshake.bb = 3 : ui32, handshake.name = "mux3"} : <i1>, [<i32>, <i32>] to <i32>
    %43 = mux %49#2 [%22#0, %67#0] {handshake.bb = 3 : ui32, handshake.name = "mux4"} : <i1>, [<i32>, <i32>] to <i32>
    %44 = mux %49#3 [%32, %69#2] {handshake.bb = 3 : ui32, handshake.name = "mux5"} : <i1>, [<i32>, <i32>] to <i32>
    %45 = mux %49#4 [%30, %trueResult_16] {handshake.bb = 3 : ui32, handshake.name = "mux6"} : <i1>, [<i32>, <i32>] to <i32>
    %46 = buffer %45, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer23"} : <i32>
    %47 = buffer %46, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer25"} : <i32>
    %48:2 = fork [2] %47 {handshake.bb = 3 : ui32, handshake.name = "fork10"} : <i32>
    %result_8, %index_9 = control_merge [%23#1, %73#1]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>, <>] to <>, <i1>
    %49:5 = fork [5] %index_9 {handshake.bb = 3 : ui32, handshake.name = "fork11"} : <i1>
    %50 = cmpi slt, %41#1, %51 {handshake.bb = 3 : ui32, handshake.name = "cmpi1"} : <i32>
    %51 = buffer %48#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer24"} : <i32>
    %52 = buffer %50, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer27"} : <i1>
    %53:8 = fork [8] %52 {handshake.bb = 3 : ui32, handshake.name = "fork12"} : <i1>
    %54 = buffer %42, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer15"} : <i32>
    %55 = buffer %54, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer17"} : <i32>
    %trueResult_10, %falseResult_11 = cond_br %53#6, %55 {handshake.bb = 3 : ui32, handshake.name = "cond_br5"} : <i1>, <i32>
    %56 = buffer %43, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer20"} : <i32>
    %57 = buffer %56, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer21"} : <i32>
    %trueResult_12, %falseResult_13 = cond_br %53#5, %57 {handshake.bb = 3 : ui32, handshake.name = "cond_br6"} : <i1>, <i32>
    %58 = buffer %44, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer22"} : <i32>
    %trueResult_14, %falseResult_15 = cond_br %53#4, %58 {handshake.bb = 3 : ui32, handshake.name = "cond_br7"} : <i1>, <i32>
    sink %falseResult_15 {handshake.name = "sink4"} : <i32>
    %trueResult_16, %falseResult_17 = cond_br %53#3, %48#0 {handshake.bb = 3 : ui32, handshake.name = "cond_br8"} : <i1>, <i32>
    sink %falseResult_17 {handshake.name = "sink5"} : <i32>
    %trueResult_18, %falseResult_19 = cond_br %53#2, %41#0 {handshake.bb = 3 : ui32, handshake.name = "cond_br9"} : <i1>, <i32>
    sink %falseResult_19 {handshake.name = "sink6"} : <i32>
    %59 = buffer %result_8, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer26"} : <>
    %trueResult_20, %falseResult_21 = cond_br %53#1, %59 {handshake.bb = 3 : ui32, handshake.name = "cond_br10"} : <i1>, <>
    %60 = buffer %34, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer11"} : <>
    %61 = buffer %60, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer12"} : <>
    %trueResult_22, %falseResult_23 = cond_br %62, %61 {handshake.bb = 4 : ui32, handshake.name = "cond_br13"} : <i1>, <>
    %62 = buffer %53#0, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 4 : ui32, handshake.name = "buffer34"} : <i1>
    %63 = buffer %trueResult_10, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer29"} : <i32>
    %64:6 = fork [6] %63 {handshake.bb = 4 : ui32, handshake.name = "fork13"} : <i32>
    %65 = trunci %66 {handshake.bb = 4 : ui32, handshake.name = "trunci0"} : <i32> to <i7>
    %66 = buffer %64#0, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 4 : ui32, handshake.name = "buffer35"} : <i32>
    %67:4 = fork [4] %trueResult_12 {handshake.bb = 4 : ui32, handshake.name = "fork14"} : <i32>
    %68 = buffer %trueResult_14, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer28"} : <i32>
    %69:3 = fork [3] %68 {handshake.bb = 4 : ui32, handshake.name = "fork15"} : <i32>
    %70 = trunci %69#0 {handshake.bb = 4 : ui32, handshake.name = "trunci1"} : <i32> to <i7>
    %71 = trunci %69#1 {handshake.bb = 4 : ui32, handshake.name = "trunci2"} : <i32> to <i4>
    %72:4 = fork [4] %trueResult_18 {handshake.bb = 4 : ui32, handshake.name = "fork16"} : <i32>
    %73:2 = fork [2] %trueResult_20 {handshake.bb = 4 : ui32, handshake.name = "fork17"} : <>
    %74 = constant %73#0 {handshake.bb = 4 : ui32, handshake.name = "constant4", value = 1 : i2} : <>, <i2>
    %75 = extsi %74 {handshake.bb = 4 : ui32, handshake.name = "extsi2"} : <i2> to <i32>
    %76 = source {handshake.bb = 4 : ui32, handshake.name = "source1"} : <>
    %77 = constant %76 {handshake.bb = 4 : ui32, handshake.name = "constant7", value = -2 : i32} : <>, <i32>
    %78:3 = fork [3] %77 {handshake.bb = 4 : ui32, handshake.name = "fork18"} : <i32>
    %79 = source {handshake.bb = 4 : ui32, handshake.name = "source2"} : <>
    %80 = constant %79 {handshake.bb = 4 : ui32, handshake.name = "constant8", value = -1 : i32} : <>, <i32>
    %81:5 = fork [5] %80 {handshake.bb = 4 : ui32, handshake.name = "fork19"} : <i32>
    %82 = trunci %81#0 {handshake.bb = 4 : ui32, handshake.name = "trunci3"} : <i32> to <i7>
    %83 = trunci %81#1 {handshake.bb = 4 : ui32, handshake.name = "trunci4"} : <i32> to <i4>
    %84 = source {handshake.bb = 4 : ui32, handshake.name = "source3"} : <>
    %85 = constant %84 {handshake.bb = 4 : ui32, handshake.name = "constant12", value = 1 : i2} : <>, <i2>
    %86 = extsi %85 {handshake.bb = 4 : ui32, handshake.name = "extsi3"} : <i2> to <i32>
    %87:7 = fork [7] %86 {handshake.bb = 4 : ui32, handshake.name = "fork20"} : <i32>
    %88 = source {handshake.bb = 4 : ui32, handshake.name = "source4"} : <>
    %89 = constant %88 {handshake.bb = 4 : ui32, handshake.name = "constant13", value = 3 : i3} : <>, <i3>
    %90 = extsi %89 {handshake.bb = 4 : ui32, handshake.name = "extsi4"} : <i3> to <i32>
    %91:3 = fork [3] %90 {handshake.bb = 4 : ui32, handshake.name = "fork21"} : <i32>
    %92 = addi %67#3, %72#3 {handshake.bb = 4 : ui32, handshake.name = "addi0"} : <i32>
    %93 = buffer %92, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer30"} : <i32>
    %94 = xori %93, %81#4 {handshake.bb = 4 : ui32, handshake.name = "xori0"} : <i32>
    %95 = addi %94, %87#0 {handshake.bb = 4 : ui32, handshake.name = "addi2"} : <i32>
    %96 = buffer %95, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer31"} : <i32>
    %97 = addi %96, %64#5 {handshake.bb = 4 : ui32, handshake.name = "addi3"} : <i32>
    %98 = buffer %97, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer32"} : <i32>
    %99 = addi %98, %78#2 {handshake.bb = 4 : ui32, handshake.name = "addi4"} : <i32>
    %100 = buffer %99, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer33"} : <i32>
    %101:2 = fork [2] %100 {handshake.bb = 4 : ui32, handshake.name = "fork22"} : <i32>
    %102 = addi %70, %82 {handshake.bb = 4 : ui32, handshake.name = "addi6"} : <i7>
    %103 = shli %101#1, %87#1 {handshake.bb = 4 : ui32, handshake.name = "shli0"} : <i32>
    %104 = buffer %103, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer37"} : <i32>
    %105 = trunci %104 {handshake.bb = 4 : ui32, handshake.name = "trunci5"} : <i32> to <i7>
    %106 = shli %101#0, %91#0 {handshake.bb = 4 : ui32, handshake.name = "shli1"} : <i32>
    %107 = buffer %106, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer38"} : <i32>
    %108 = trunci %107 {handshake.bb = 4 : ui32, handshake.name = "trunci6"} : <i32> to <i7>
    %109 = addi %105, %108 {handshake.bb = 4 : ui32, handshake.name = "addi5"} : <i7>
    %110 = buffer %102, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer36"} : <i7>
    %111 = buffer %109, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer39"} : <i7>
    %112 = addi %110, %111 {handshake.bb = 4 : ui32, handshake.name = "addi15"} : <i7>
    %addressResult, %dataResult = load[%112] %outputs#0 {handshake.bb = 4 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i7>, <i32>, <i7>, <i32>
    %113 = addi %71, %83 {handshake.bb = 4 : ui32, handshake.name = "addi8"} : <i4>
    %addressResult_24, %dataResult_25 = load[%113] %outputs_0 {handshake.bb = 4 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i4>, <i32>, <i4>, <i32>
    %114 = muli %dataResult, %dataResult_25 {handshake.bb = 4 : ui32, handshake.name = "muli0"} : <i32>
    %115 = addi %67#2, %72#2 {handshake.bb = 4 : ui32, handshake.name = "addi7"} : <i32>
    %116 = buffer %115, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer40"} : <i32>
    %117 = xori %116, %81#3 {handshake.bb = 4 : ui32, handshake.name = "xori1"} : <i32>
    %118 = addi %117, %87#2 {handshake.bb = 4 : ui32, handshake.name = "addi9"} : <i32>
    %119 = buffer %118, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer41"} : <i32>
    %120 = addi %119, %121 {handshake.bb = 4 : ui32, handshake.name = "addi10"} : <i32>
    %121 = buffer %64#4, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer54"} : <i32>
    %122 = buffer %120, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer42"} : <i32>
    %123 = addi %122, %78#1 {handshake.bb = 4 : ui32, handshake.name = "addi11"} : <i32>
    %124 = buffer %123, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer43"} : <i32>
    %125:2 = fork [2] %124 {handshake.bb = 4 : ui32, handshake.name = "fork23"} : <i32>
    %126 = shli %127, %128 {handshake.bb = 4 : ui32, handshake.name = "shli2"} : <i32>
    %127 = buffer %125#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer56"} : <i32>
    %128 = buffer %87#3, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer57"} : <i32>
    %129 = shli %130, %131 {handshake.bb = 4 : ui32, handshake.name = "shli3"} : <i32>
    %130 = buffer %125#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer58"} : <i32>
    %131 = buffer %91#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer59"} : <i32>
    %132 = buffer %126, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer44"} : <i32>
    %133 = buffer %129, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer45"} : <i32>
    %134 = addi %132, %133 {handshake.bb = 4 : ui32, handshake.name = "addi12"} : <i32>
    %135 = buffer %134, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer46"} : <i32>
    %136 = addi %137, %135 {handshake.bb = 4 : ui32, handshake.name = "addi16"} : <i32>
    %137 = buffer %64#3, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer60"} : <i32>
    %138 = gate %136, %trueResult_22 {handshake.bb = 4 : ui32, handshake.name = "gate0"} : <i32>, !handshake.control<> to <i32>
    %139 = trunci %138 {handshake.bb = 4 : ui32, handshake.name = "trunci7"} : <i32> to <i7>
    %addressResult_26, %dataResult_27 = load[%139] %outputs#1 {handshake.bb = 4 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["store0", 1, false], ["store0", 3, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load2"} : <i7>, <i32>, <i7>, <i32>
    %140 = subi %dataResult_27, %114 {handshake.bb = 4 : ui32, handshake.name = "subi0"} : <i32>
    %141 = addi %67#1, %72#1 {handshake.bb = 4 : ui32, handshake.name = "addi20"} : <i32>
    %142 = buffer %141, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer47"} : <i32>
    %143 = xori %142, %81#2 {handshake.bb = 4 : ui32, handshake.name = "xori2"} : <i32>
    %144 = addi %143, %87#4 {handshake.bb = 4 : ui32, handshake.name = "addi21"} : <i32>
    %145 = buffer %144, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer48"} : <i32>
    %146 = addi %145, %64#2 {handshake.bb = 4 : ui32, handshake.name = "addi13"} : <i32>
    %147 = buffer %146, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer49"} : <i32>
    %148 = addi %147, %149 {handshake.bb = 4 : ui32, handshake.name = "addi14"} : <i32>
    %149 = buffer %78#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer66"} : <i32>
    %150 = buffer %148, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer50"} : <i32>
    %151:2 = fork [2] %150 {handshake.bb = 4 : ui32, handshake.name = "fork24"} : <i32>
    %152 = shli %153, %154 {handshake.bb = 4 : ui32, handshake.name = "shli4"} : <i32>
    %153 = buffer %151#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer67"} : <i32>
    %154 = buffer %87#5, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 4 : ui32, handshake.name = "buffer68"} : <i32>
    %155 = buffer %152, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer51"} : <i32>
    %156 = trunci %155 {handshake.bb = 4 : ui32, handshake.name = "trunci8"} : <i32> to <i7>
    %157 = shli %158, %159 {handshake.bb = 4 : ui32, handshake.name = "shli5"} : <i32>
    %158 = buffer %151#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer69"} : <i32>
    %159 = buffer %91#2, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 4 : ui32, handshake.name = "buffer70"} : <i32>
    %160 = buffer %157, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer52"} : <i32>
    %161 = trunci %160 {handshake.bb = 4 : ui32, handshake.name = "trunci9"} : <i32> to <i7>
    %162 = addi %156, %161 {handshake.bb = 4 : ui32, handshake.name = "addi22"} : <i7>
    %163 = buffer %162, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer53"} : <i7>
    %164 = addi %65, %163 {handshake.bb = 4 : ui32, handshake.name = "addi17"} : <i7>
    %165 = buffer %doneResult, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer55"} : <>
    %166 = buffer %165, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer0"} : <>
    %addressResult_28, %dataResult_29, %doneResult = store[%164] %140 %outputs#2 {handshake.bb = 4 : ui32, handshake.bufProps = #handshake<bufProps{"2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["load2", 1, false], ["store0", 1, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store0"} : <i7>, <i32>, <>, <i7>, <i32>, <>
    %167 = addi %72#0, %168 {handshake.bb = 4 : ui32, handshake.name = "addi18"} : <i32>
    %168 = buffer %87#6, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer71"} : <i32>
    %169 = buffer %167, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer61"} : <i32>
    %170 = source {handshake.bb = 5 : ui32, handshake.name = "source5"} : <>
    %171 = constant %170 {handshake.bb = 5 : ui32, handshake.name = "constant14", value = 1 : i2} : <>, <i2>
    %172 = extsi %171 {handshake.bb = 5 : ui32, handshake.name = "extsi5"} : <i2> to <i32>
    %173 = addi %falseResult_13, %172 {handshake.bb = 5 : ui32, handshake.name = "addi19"} : <i32>
    %174:2 = fork [2] %falseResult_5 {handshake.bb = 6 : ui32, handshake.name = "fork25"} : <>
    end {handshake.bb = 6 : ui32, handshake.name = "end0"} %memEnd_1, %memEnd, %0#1 : <>, <>, <>
  }
}

