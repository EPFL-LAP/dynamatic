module {
  handshake.func @triangular(%arg0: memref<10xi32>, %arg1: !handshake.channel<i32>, %arg2: memref<100xi32>, %arg3: !handshake.control<>, %arg4: !handshake.control<>, %arg5: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["x", "n", "a", "x_start", "a_start", "start"], resNames = ["x_end", "a_end", "end"]} {
    %0:3 = fork [3] %arg5 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork0"} : <>
    %outputs:2, %memEnd = mem_controller[%arg2 : memref<100xi32>] %arg4 (%65, %addressResult, %1#1, %1#2, %1#3) %154#1 {connectedBlocks = [4 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "5": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "6": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller2"} :    (!handshake.channel<i32>, !handshake.channel<i7>, !handshake.channel<i7>, !handshake.channel<i7>, !handshake.channel<i32>) -> (!handshake.channel<i32>, !handshake.channel<i32>)
    %1:4 = lsq[MC] (%62#0, %addressResult_22, %addressResult_24, %dataResult_25, %outputs#1)  {groupSizes = [2 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "lsq1"} : (!handshake.control<>, !handshake.channel<i7>, !handshake.channel<i7>, !handshake.channel<i32>, !handshake.channel<i32>) -> (!handshake.channel<i32>, !handshake.channel<i7>, !handshake.channel<i7>, !handshake.channel<i32>)
    %outputs_0, %memEnd_1 = mem_controller[%arg0 : memref<10xi32>] %arg3 (%addressResult_20) %154#0 {connectedBlocks = [4 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller3"} :    (!handshake.channel<i4>) -> !handshake.channel<i32>
    %2 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant3", value = false} : <>, <i1>
    %3 = extsi %2 {handshake.bb = 0 : ui32, handshake.name = "extsi6"} : <i1> to <i32>
    %4 = buffer %153, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer50"} : <i32>
    %5 = mux %14#0 [%3, %4] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i32>, <i32>] to <i32>
    %6 = buffer %5, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer0"} : <i32>
    %7 = buffer %6, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer2"} : <i32>
    %8:2 = fork [2] %7 {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <i32>
    %9 = mux %10 [%arg1, %falseResult_9] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i1>, [<i32>, <i32>] to <i32>
    %10 = buffer %14#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer1"} : <i1>
    %11 = buffer %9, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer4"} : <i32>
    %12 = buffer %11, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer5"} : <i32>
    %13:2 = fork [2] %12 {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <i32>
    %result, %index = control_merge [%0#2, %falseResult_19]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %14:2 = fork [2] %index {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <i1>
    %15 = cmpi slt, %16, %13#1 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i32>
    %16 = buffer %8#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer3"} : <i32>
    %17 = buffer %15, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer6"} : <i1>
    %18:3 = fork [3] %17 {handshake.bb = 1 : ui32, handshake.name = "fork4"} : <i1>
    %trueResult, %falseResult = cond_br %18#2, %13#0 {handshake.bb = 1 : ui32, handshake.name = "cond_br2"} : <i1>, <i32>
    sink %falseResult {handshake.name = "sink0"} : <i32>
    %trueResult_2, %falseResult_3 = cond_br %18#1, %19 {handshake.bb = 1 : ui32, handshake.name = "cond_br3"} : <i1>, <i32>
    %19 = buffer %8#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer7"} : <i32>
    sink %falseResult_3 {handshake.name = "sink1"} : <i32>
    %trueResult_4, %falseResult_5 = cond_br %20, %result {handshake.bb = 1 : ui32, handshake.name = "cond_br4"} : <i1>, <>
    %20 = buffer %18#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer8"} : <i1>
    %21:2 = fork [2] %trueResult {handshake.bb = 2 : ui32, handshake.name = "fork5"} : <i32>
    %22:2 = fork [2] %trueResult_2 {handshake.bb = 2 : ui32, handshake.name = "fork6"} : <i32>
    %23:2 = fork [2] %trueResult_4 {handshake.bb = 2 : ui32, handshake.name = "fork7"} : <>
    %24 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %25 = constant %24 {handshake.bb = 2 : ui32, handshake.name = "constant5", value = -1 : i32} : <>, <i32>
    %26 = constant %23#0 {handshake.bb = 2 : ui32, handshake.name = "constant4", value = false} : <>, <i1>
    %27 = subi %21#1, %22#1 {handshake.bb = 2 : ui32, handshake.name = "subi1"} : <i32>
    %28 = buffer %27, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer9"} : <i32>
    %29:2 = fork [2] %28 {handshake.bb = 2 : ui32, handshake.name = "fork8"} : <i32>
    %30 = addi %29#1, %25 {handshake.bb = 2 : ui32, handshake.name = "addi1"} : <i32>
    %31 = extsi %26 {handshake.bb = 2 : ui32, handshake.name = "extsi7"} : <i1> to <i32>
    %32 = mux %43#0 [%31, %148] {handshake.bb = 3 : ui32, handshake.name = "mux2"} : <i1>, [<i32>, <i32>] to <i32>
    %33 = buffer %32, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer10"} : <i32>
    %34 = buffer %33, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer11"} : <i32>
    %35:2 = fork [2] %34 {handshake.bb = 3 : ui32, handshake.name = "fork9"} : <i32>
    %36 = mux %43#1 [%21#0, %53#2] {handshake.bb = 3 : ui32, handshake.name = "mux3"} : <i1>, [<i32>, <i32>] to <i32>
    %37 = mux %43#2 [%22#0, %56#0] {handshake.bb = 3 : ui32, handshake.name = "mux4"} : <i1>, [<i32>, <i32>] to <i32>
    %38 = mux %43#3 [%29#0, %58#2] {handshake.bb = 3 : ui32, handshake.name = "mux5"} : <i1>, [<i32>, <i32>] to <i32>
    %39 = mux %43#4 [%30, %trueResult_14] {handshake.bb = 3 : ui32, handshake.name = "mux6"} : <i1>, [<i32>, <i32>] to <i32>
    %40 = buffer %39, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer17"} : <i32>
    %41 = buffer %40, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer18"} : <i32>
    %42:2 = fork [2] %41 {handshake.bb = 3 : ui32, handshake.name = "fork10"} : <i32>
    %result_6, %index_7 = control_merge [%23#1, %149]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>, <>] to <>, <i1>
    %43:5 = fork [5] %index_7 {handshake.bb = 3 : ui32, handshake.name = "fork11"} : <i1>
    %44 = cmpi slt, %35#1, %42#1 {handshake.bb = 3 : ui32, handshake.name = "cmpi1"} : <i32>
    %45 = buffer %44, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer20"} : <i1>
    %46:6 = fork [6] %45 {handshake.bb = 3 : ui32, handshake.name = "fork12"} : <i1>
    %47 = buffer %36, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer12"} : <i32>
    %48 = buffer %47, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer13"} : <i32>
    %trueResult_8, %falseResult_9 = cond_br %46#5, %48 {handshake.bb = 3 : ui32, handshake.name = "cond_br5"} : <i1>, <i32>
    %49 = buffer %37, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer14"} : <i32>
    %50 = buffer %49, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer15"} : <i32>
    %trueResult_10, %falseResult_11 = cond_br %46#4, %50 {handshake.bb = 3 : ui32, handshake.name = "cond_br6"} : <i1>, <i32>
    %51 = buffer %38, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer16"} : <i32>
    %trueResult_12, %falseResult_13 = cond_br %46#3, %51 {handshake.bb = 3 : ui32, handshake.name = "cond_br7"} : <i1>, <i32>
    sink %falseResult_13 {handshake.name = "sink3"} : <i32>
    %trueResult_14, %falseResult_15 = cond_br %46#2, %42#0 {handshake.bb = 3 : ui32, handshake.name = "cond_br8"} : <i1>, <i32>
    sink %falseResult_15 {handshake.name = "sink4"} : <i32>
    %trueResult_16, %falseResult_17 = cond_br %46#1, %35#0 {handshake.bb = 3 : ui32, handshake.name = "cond_br9"} : <i1>, <i32>
    sink %falseResult_17 {handshake.name = "sink5"} : <i32>
    %52 = buffer %result_6, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer19"} : <>
    %trueResult_18, %falseResult_19 = cond_br %46#0, %52 {handshake.bb = 3 : ui32, handshake.name = "cond_br10"} : <i1>, <>
    %53:6 = fork [6] %trueResult_8 {handshake.bb = 4 : ui32, handshake.name = "fork13"} : <i32>
    %54 = trunci %53#0 {handshake.bb = 4 : ui32, handshake.name = "trunci0"} : <i32> to <i7>
    %55 = trunci %53#1 {handshake.bb = 4 : ui32, handshake.name = "trunci1"} : <i32> to <i7>
    %56:4 = fork [4] %trueResult_10 {handshake.bb = 4 : ui32, handshake.name = "fork14"} : <i32>
    %57 = buffer %trueResult_12, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer21"} : <i32>
    %58:3 = fork [3] %57 {handshake.bb = 4 : ui32, handshake.name = "fork15"} : <i32>
    %59 = trunci %58#0 {handshake.bb = 4 : ui32, handshake.name = "trunci2"} : <i32> to <i7>
    %60 = trunci %58#1 {handshake.bb = 4 : ui32, handshake.name = "trunci3"} : <i32> to <i4>
    %61:4 = fork [4] %trueResult_16 {handshake.bb = 4 : ui32, handshake.name = "fork16"} : <i32>
    %62:3 = fork [3] %trueResult_18 {handshake.bb = 4 : ui32, handshake.name = "fork17"} : <>
    %63 = buffer %62#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer22"} : <>
    %64 = constant %63 {handshake.bb = 4 : ui32, handshake.bufProps = #handshake<bufProps{"0": [1,inf], [0,inf], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "constant12", value = 1 : i2} : <>, <i2>
    %65 = extsi %64 {handshake.bb = 4 : ui32, handshake.name = "extsi2"} : <i2> to <i32>
    %66 = source {handshake.bb = 4 : ui32, handshake.name = "source1"} : <>
    %67 = constant %66 {handshake.bb = 4 : ui32, handshake.name = "constant7", value = -2 : i32} : <>, <i32>
    %68:3 = fork [3] %67 {handshake.bb = 4 : ui32, handshake.name = "fork18"} : <i32>
    %69 = source {handshake.bb = 4 : ui32, handshake.name = "source2"} : <>
    %70 = constant %69 {handshake.bb = 4 : ui32, handshake.name = "constant8", value = -1 : i32} : <>, <i32>
    %71:5 = fork [5] %70 {handshake.bb = 4 : ui32, handshake.name = "fork19"} : <i32>
    %72 = trunci %71#0 {handshake.bb = 4 : ui32, handshake.name = "trunci4"} : <i32> to <i7>
    %73 = trunci %71#1 {handshake.bb = 4 : ui32, handshake.name = "trunci5"} : <i32> to <i4>
    %74 = source {handshake.bb = 4 : ui32, handshake.name = "source3"} : <>
    %75 = constant %74 {handshake.bb = 4 : ui32, handshake.name = "constant13", value = 1 : i2} : <>, <i2>
    %76 = extsi %75 {handshake.bb = 4 : ui32, handshake.name = "extsi3"} : <i2> to <i32>
    %77:7 = fork [7] %76 {handshake.bb = 4 : ui32, handshake.name = "fork20"} : <i32>
    %78 = source {handshake.bb = 4 : ui32, handshake.name = "source4"} : <>
    %79 = constant %78 {handshake.bb = 4 : ui32, handshake.name = "constant14", value = 3 : i3} : <>, <i3>
    %80 = extsi %79 {handshake.bb = 4 : ui32, handshake.name = "extsi4"} : <i3> to <i32>
    %81:3 = fork [3] %80 {handshake.bb = 4 : ui32, handshake.name = "fork21"} : <i32>
    %82 = addi %56#3, %61#3 {handshake.bb = 4 : ui32, handshake.name = "addi0"} : <i32>
    %83 = buffer %82, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer24"} : <i32>
    %84 = xori %83, %71#4 {handshake.bb = 4 : ui32, handshake.name = "xori0"} : <i32>
    %85 = addi %84, %77#0 {handshake.bb = 4 : ui32, handshake.name = "addi2"} : <i32>
    %86 = buffer %85, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer25"} : <i32>
    %87 = addi %86, %53#5 {handshake.bb = 4 : ui32, handshake.name = "addi3"} : <i32>
    %88 = buffer %87, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer26"} : <i32>
    %89 = addi %88, %68#2 {handshake.bb = 4 : ui32, handshake.name = "addi4"} : <i32>
    %90 = buffer %89, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer27"} : <i32>
    %91:2 = fork [2] %90 {handshake.bb = 4 : ui32, handshake.name = "fork22"} : <i32>
    %92 = addi %59, %72 {handshake.bb = 4 : ui32, handshake.name = "addi6"} : <i7>
    %93 = shli %91#1, %77#1 {handshake.bb = 4 : ui32, handshake.name = "shli0"} : <i32>
    %94 = buffer %93, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer29"} : <i32>
    %95 = trunci %94 {handshake.bb = 4 : ui32, handshake.name = "trunci6"} : <i32> to <i7>
    %96 = shli %91#0, %81#0 {handshake.bb = 4 : ui32, handshake.name = "shli1"} : <i32>
    %97 = buffer %96, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer30"} : <i32>
    %98 = trunci %97 {handshake.bb = 4 : ui32, handshake.name = "trunci7"} : <i32> to <i7>
    %99 = addi %95, %98 {handshake.bb = 4 : ui32, handshake.name = "addi5"} : <i7>
    %100 = buffer %92, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer28"} : <i7>
    %101 = buffer %99, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer31"} : <i7>
    %102 = addi %100, %101 {handshake.bb = 4 : ui32, handshake.name = "addi15"} : <i7>
    %addressResult, %dataResult = load[%102] %outputs#0 {handshake.bb = 4 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i7>, <i32>, <i7>, <i32>
    %103 = addi %60, %73 {handshake.bb = 4 : ui32, handshake.name = "addi8"} : <i4>
    %addressResult_20, %dataResult_21 = load[%103] %outputs_0 {handshake.bb = 4 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i4>, <i32>, <i4>, <i32>
    %104 = muli %dataResult, %dataResult_21 {handshake.bb = 4 : ui32, handshake.name = "muli0"} : <i32>
    %105 = addi %56#2, %61#2 {handshake.bb = 4 : ui32, handshake.name = "addi7"} : <i32>
    %106 = buffer %105, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer32"} : <i32>
    %107 = xori %106, %71#3 {handshake.bb = 4 : ui32, handshake.name = "xori1"} : <i32>
    %108 = addi %107, %77#2 {handshake.bb = 4 : ui32, handshake.name = "addi9"} : <i32>
    %109 = buffer %108, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer33"} : <i32>
    %110 = addi %109, %53#4 {handshake.bb = 4 : ui32, handshake.name = "addi10"} : <i32>
    %111 = buffer %110, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer34"} : <i32>
    %112 = addi %111, %68#1 {handshake.bb = 4 : ui32, handshake.name = "addi11"} : <i32>
    %113 = buffer %112, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer35"} : <i32>
    %114:2 = fork [2] %113 {handshake.bb = 4 : ui32, handshake.name = "fork23"} : <i32>
    %115 = shli %114#1, %77#3 {handshake.bb = 4 : ui32, handshake.name = "shli2"} : <i32>
    %116 = buffer %115, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer36"} : <i32>
    %117 = trunci %116 {handshake.bb = 4 : ui32, handshake.name = "trunci8"} : <i32> to <i7>
    %118 = shli %114#0, %81#1 {handshake.bb = 4 : ui32, handshake.name = "shli3"} : <i32>
    %119 = buffer %118, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer37"} : <i32>
    %120 = trunci %119 {handshake.bb = 4 : ui32, handshake.name = "trunci9"} : <i32> to <i7>
    %121 = addi %117, %120 {handshake.bb = 4 : ui32, handshake.name = "addi12"} : <i7>
    %122 = buffer %121, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer38"} : <i7>
    %123 = addi %54, %122 {handshake.bb = 4 : ui32, handshake.name = "addi16"} : <i7>
    %124 = buffer %123, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer39"} : <i7>
    %addressResult_22, %dataResult_23 = load[%124] %1#0 {handshake.bb = 4 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["store0", 1], ["store0", 3]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 0>, handshake.name = "load2"} : <i7>, <i32>, <i7>, <i32>
    %125 = subi %dataResult_23, %104 {handshake.bb = 4 : ui32, handshake.name = "subi0"} : <i32>
    %126 = addi %56#1, %61#1 {handshake.bb = 4 : ui32, handshake.name = "addi20"} : <i32>
    %127 = buffer %126, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer41"} : <i32>
    %128 = xori %127, %71#2 {handshake.bb = 4 : ui32, handshake.name = "xori2"} : <i32>
    %129 = addi %128, %77#4 {handshake.bb = 4 : ui32, handshake.name = "addi21"} : <i32>
    %130 = buffer %129, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer42"} : <i32>
    %131 = addi %130, %53#3 {handshake.bb = 4 : ui32, handshake.name = "addi13"} : <i32>
    %132 = buffer %131, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer43"} : <i32>
    %133 = addi %132, %68#0 {handshake.bb = 4 : ui32, handshake.name = "addi14"} : <i32>
    %134 = buffer %133, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer44"} : <i32>
    %135:2 = fork [2] %134 {handshake.bb = 4 : ui32, handshake.name = "fork24"} : <i32>
    %136 = shli %135#1, %77#5 {handshake.bb = 4 : ui32, handshake.name = "shli4"} : <i32>
    %137 = buffer %136, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer45"} : <i32>
    %138 = trunci %137 {handshake.bb = 4 : ui32, handshake.name = "trunci10"} : <i32> to <i7>
    %139 = shli %135#0, %81#2 {handshake.bb = 4 : ui32, handshake.name = "shli5"} : <i32>
    %140 = buffer %139, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer46"} : <i32>
    %141 = trunci %140 {handshake.bb = 4 : ui32, handshake.name = "trunci11"} : <i32> to <i7>
    %142 = addi %138, %141 {handshake.bb = 4 : ui32, handshake.name = "addi22"} : <i7>
    %143 = buffer %142, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer47"} : <i7>
    %144 = addi %55, %143 {handshake.bb = 4 : ui32, handshake.name = "addi17"} : <i7>
    %145 = buffer %125, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer40"} : <i32>
    %146 = buffer %144, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer48"} : <i7>
    %addressResult_24, %dataResult_25 = store[%146] %145 {handshake.bb = 4 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["load2", 1], ["store0", 1]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 0>, handshake.name = "store0"} : <i7>, <i32>, <i7>, <i32>
    %147 = addi %61#0, %77#6 {handshake.bb = 4 : ui32, handshake.name = "addi18"} : <i32>
    %148 = buffer %147, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer49"} : <i32>
    %149 = buffer %62#2, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer23"} : <>
    %150 = source {handshake.bb = 5 : ui32, handshake.name = "source5"} : <>
    %151 = constant %150 {handshake.bb = 5 : ui32, handshake.name = "constant15", value = 1 : i2} : <>, <i2>
    %152 = extsi %151 {handshake.bb = 5 : ui32, handshake.name = "extsi5"} : <i2> to <i32>
    %153 = addi %falseResult_11, %152 {handshake.bb = 5 : ui32, handshake.name = "addi19"} : <i32>
    %154:2 = fork [2] %falseResult_5 {handshake.bb = 6 : ui32, handshake.name = "fork25"} : <>
    end {handshake.bb = 6 : ui32, handshake.name = "end0"} %memEnd_1, %memEnd, %0#1 : <>, <>, <>
  }
}

