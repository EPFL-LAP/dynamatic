module {
  handshake.func @mvt_float(%arg0: memref<900xf32>, %arg1: memref<30xf32>, %arg2: memref<30xf32>, %arg3: memref<30xf32>, %arg4: memref<30xf32>, %arg5: !handshake.control<>, %arg6: !handshake.control<>, %arg7: !handshake.control<>, %arg8: !handshake.control<>, %arg9: !handshake.control<>, %arg10: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["A", "x1", "x2", "y1", "y2", "A_start", "x1_start", "x2_start", "y1_start", "y2_start", "start"], cfg.edges = "[0,1][2,2,3,cmpi0][4,5][6,4,7,cmpi3][1,2][3,1,4,cmpi1][5,5,6,cmpi2]", resNames = ["A_end", "x1_end", "x2_end", "y1_end", "y2_end", "end"]} {
    %0:3 = fork [3] %arg10 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork0"} : <>
    %outputs, %memEnd = mem_controller[%arg4 : memref<30xf32>] %arg9 (%addressResult_36) %149#4 {connectedBlocks = [5 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller3"} :    (!handshake.channel<i5>) -> !handshake.channel<f32>
    %outputs_0, %memEnd_1 = mem_controller[%arg3 : memref<30xf32>] %arg8 (%addressResult_12) %149#3 {connectedBlocks = [2 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller4"} :    (!handshake.channel<i5>) -> !handshake.channel<f32>
    %outputs_2:2, %memEnd_3 = mem_controller[%arg2 : memref<30xf32>] %arg7 (%addressResult_30, %133, %addressResult_46, %dataResult_47) %149#2 {connectedBlocks = [4 : i32, 6 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "5": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller5"} :    (!handshake.channel<i5>, !handshake.channel<i32>, !handshake.channel<i5>, !handshake.channel<f32>) -> (!handshake.channel<f32>, !handshake.control<>)
    %outputs_4:2, %memEnd_5 = mem_controller[%arg1 : memref<30xf32>] %arg6 (%addressResult, %61, %addressResult_20, %dataResult_21) %149#1 {connectedBlocks = [1 : i32, 3 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "5": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller6"} :    (!handshake.channel<i5>, !handshake.channel<i32>, !handshake.channel<i5>, !handshake.channel<f32>) -> (!handshake.channel<f32>, !handshake.control<>)
    %outputs_6:2, %memEnd_7 = mem_controller[%arg0 : memref<900xf32>] %arg5 (%addressResult_10, %addressResult_34) %149#0 {connectedBlocks = [2 : i32, 5 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller7"} :    (!handshake.channel<i10>, !handshake.channel<i10>) -> (!handshake.channel<f32>, !handshake.channel<f32>)
    %1 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant14", value = false} : <>, <i1>
    %2 = extsi %1 {handshake.bb = 0 : ui32, handshake.name = "extsi17"} : <i1> to <i6>
    %3 = init %77#1 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "init0"} : <i1>
    sink %3 {handshake.name = "sink0"} : <i1>
    %4:2 = unbundle %12#0  {handshake.bb = 1 : ui32, handshake.name = "unbundle0"} : <f32> to _ 
    %5 = mux %index [%2, %trueResult_22] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i6>, <i6>] to <i6>
    %6 = buffer %5, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer2"} : <i6>
    %7:2 = fork [2] %6 {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <i6>
    %8 = trunci %7#0 {handshake.bb = 1 : ui32, handshake.name = "trunci0"} : <i6> to <i5>
    %result, %index = control_merge [%0#2, %trueResult_24]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %9:2 = fork [2] %result {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <>
    %10 = constant %9#0 {handshake.bb = 1 : ui32, handshake.name = "constant15", value = false} : <>, <i1>
    %11 = buffer %4#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer0"} : <>
    %addressResult, %dataResult = load[%8] %outputs_4#0 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["store0", 2, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i5>, <f32>, <i5>, <f32>
    %12:2 = fork [2] %dataResult {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <f32>
    %13 = extsi %10 {handshake.bb = 1 : ui32, handshake.name = "extsi16"} : <i1> to <i6>
    %14 = mux %30#1 [%13, %trueResult] {handshake.bb = 2 : ui32, handshake.name = "mux1"} : <i1>, [<i6>, <i6>] to <i6>
    %15 = buffer %14, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer3"} : <i6>
    %16:3 = fork [3] %15 {handshake.bb = 2 : ui32, handshake.name = "fork4"} : <i6>
    %17 = extsi %18 {handshake.bb = 2 : ui32, handshake.name = "extsi18"} : <i6> to <i10>
    %18 = buffer %16#0, bufferType = FIFO_BREAK_NONE, numSlots = 9 {handshake.bb = 2 : ui32, handshake.name = "buffer8"} : <i6>
    %19 = extsi %16#2 {handshake.bb = 2 : ui32, handshake.name = "extsi19"} : <i6> to <i7>
    %20 = trunci %21 {handshake.bb = 2 : ui32, handshake.name = "trunci1"} : <i6> to <i5>
    %21 = buffer %16#1, bufferType = FIFO_BREAK_NONE, numSlots = 9 {handshake.bb = 2 : ui32, handshake.name = "buffer10"} : <i6>
    %22 = mux %23 [%12#1, %trueResult_14] {handshake.bb = 2 : ui32, handshake.name = "mux2"} : <i1>, [<f32>, <f32>] to <f32>
    %23 = buffer %30#2, bufferType = FIFO_BREAK_NONE, numSlots = 15 {handshake.bb = 2 : ui32, handshake.name = "buffer11"} : <i1>
    %24 = mux %30#0 [%7#1, %trueResult_16] {handshake.bb = 2 : ui32, handshake.name = "mux3"} : <i1>, [<i6>, <i6>] to <i6>
    %25 = buffer %24, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer5"} : <i6>
    %26 = buffer %25, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer6"} : <i6>
    %27:2 = fork [2] %26 {handshake.bb = 2 : ui32, handshake.name = "fork5"} : <i6>
    %28 = extsi %29 {handshake.bb = 2 : ui32, handshake.name = "extsi20"} : <i6> to <i12>
    %29 = buffer %27#1, bufferType = FIFO_BREAK_NONE, numSlots = 5 {handshake.bb = 2 : ui32, handshake.name = "buffer13"} : <i6>
    %result_8, %index_9 = control_merge [%9#1, %trueResult_18]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>, <>] to <>, <i1>
    %30:3 = fork [3] %index_9 {handshake.bb = 2 : ui32, handshake.name = "fork6"} : <i1>
    %31 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %32 = constant %31 {handshake.bb = 2 : ui32, handshake.name = "constant16", value = 30 : i6} : <>, <i6>
    %33:2 = fork [2] %32 {handshake.bb = 2 : ui32, handshake.name = "fork7"} : <i6>
    %34 = extsi %35 {handshake.bb = 2 : ui32, handshake.name = "extsi21"} : <i6> to <i12>
    %35 = buffer %33#0, bufferType = FIFO_BREAK_NONE, numSlots = 5 {handshake.bb = 2 : ui32, handshake.name = "buffer14"} : <i6>
    %36 = extsi %33#1 {handshake.bb = 2 : ui32, handshake.name = "extsi22"} : <i6> to <i7>
    %37 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %38 = constant %37 {handshake.bb = 2 : ui32, handshake.name = "constant17", value = 1 : i2} : <>, <i2>
    %39 = extsi %38 {handshake.bb = 2 : ui32, handshake.name = "extsi23"} : <i2> to <i7>
    %40 = muli %28, %34 {handshake.bb = 2 : ui32, handshake.name = "muli0"} : <i12>
    %41 = trunci %40 {handshake.bb = 2 : ui32, handshake.name = "trunci2"} : <i12> to <i10>
    %42 = addi %17, %41 {handshake.bb = 2 : ui32, handshake.name = "addi0"} : <i10>
    %addressResult_10, %dataResult_11 = load[%42] %outputs_6#0 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i10>, <f32>, <i10>, <f32>
    %addressResult_12, %dataResult_13 = load[%20] %outputs_0 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load2"} : <i5>, <f32>, <i5>, <f32>
    %43 = mulf %dataResult_11, %dataResult_13 {fastmath = #arith.fastmath<none>, fpu_impl = #handshake<fpu_impl flopoco>, handshake.bb = 2 : ui32, handshake.name = "mulf0", internal_delay = "2_875333"} : <f32>
    %44 = buffer %22, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer4"} : <f32>
    %45 = addf %44, %43 {fastmath = #arith.fastmath<none>, fpu_impl = #handshake<fpu_impl flopoco>, handshake.bb = 2 : ui32, handshake.name = "addf0", internal_delay = "2_922000"} : <f32>
    %46 = addi %19, %39 {handshake.bb = 2 : ui32, handshake.name = "addi2"} : <i7>
    %47 = buffer %46, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer9"} : <i7>
    %48:2 = fork [2] %47 {handshake.bb = 2 : ui32, handshake.name = "fork8"} : <i7>
    %49 = trunci %48#0 {handshake.bb = 2 : ui32, handshake.name = "trunci3"} : <i7> to <i6>
    %50 = cmpi ult, %48#1, %36 {handshake.bb = 2 : ui32, handshake.name = "cmpi0"} : <i7>
    %51 = buffer %50, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer12"} : <i1>
    %52:4 = fork [4] %51 {handshake.bb = 2 : ui32, handshake.name = "fork9"} : <i1>
    %trueResult, %falseResult = cond_br %52#0, %49 {handshake.bb = 2 : ui32, handshake.name = "cond_br4"} : <i1>, <i6>
    sink %falseResult {handshake.name = "sink1"} : <i6>
    %trueResult_14, %falseResult_15 = cond_br %53, %45 {handshake.bb = 2 : ui32, handshake.name = "cond_br5"} : <i1>, <f32>
    %53 = buffer %52#2, bufferType = FIFO_BREAK_NONE, numSlots = 15 {handshake.bb = 2 : ui32, handshake.name = "buffer19"} : <i1>
    %trueResult_16, %falseResult_17 = cond_br %52#1, %27#0 {handshake.bb = 2 : ui32, handshake.name = "cond_br6"} : <i1>, <i6>
    %54 = buffer %result_8, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer7"} : <>
    %trueResult_18, %falseResult_19 = cond_br %52#3, %54 {handshake.bb = 2 : ui32, handshake.name = "cond_br7"} : <i1>, <>
    %55:2 = fork [2] %falseResult_17 {handshake.bb = 3 : ui32, handshake.name = "fork10"} : <i6>
    %56 = extsi %55#0 {handshake.bb = 3 : ui32, handshake.name = "extsi24"} : <i6> to <i7>
    %57 = extsi %58 {handshake.bb = 3 : ui32, handshake.name = "extsi25"} : <i6> to <i32>
    %58 = buffer %55#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer24"} : <i6>
    %59:3 = fork [3] %falseResult_19 {handshake.bb = 3 : ui32, handshake.name = "fork11"} : <>
    %60 = constant %59#1 {handshake.bb = 3 : ui32, handshake.name = "constant18", value = 1 : i2} : <>, <i2>
    %61 = extsi %60 {handshake.bb = 3 : ui32, handshake.name = "extsi4"} : <i2> to <i32>
    %62 = constant %59#0 {handshake.bb = 3 : ui32, handshake.name = "constant19", value = false} : <>, <i1>
    %63 = source {handshake.bb = 3 : ui32, handshake.name = "source2"} : <>
    %64 = constant %63 {handshake.bb = 3 : ui32, handshake.name = "constant20", value = 30 : i6} : <>, <i6>
    %65 = extsi %64 {handshake.bb = 3 : ui32, handshake.name = "extsi26"} : <i6> to <i7>
    %66 = source {handshake.bb = 3 : ui32, handshake.name = "source3"} : <>
    %67 = constant %66 {handshake.bb = 3 : ui32, handshake.name = "constant21", value = 1 : i2} : <>, <i2>
    %68 = extsi %67 {handshake.bb = 3 : ui32, handshake.name = "extsi27"} : <i2> to <i7>
    %69 = gate %57, %11 {handshake.bb = 3 : ui32, handshake.name = "gate0"} : <i32>, !handshake.control<> to <i32>
    %70 = trunci %69 {handshake.bb = 3 : ui32, handshake.name = "trunci4"} : <i32> to <i5>
    %addressResult_20, %dataResult_21, %doneResult = store[%70] %falseResult_15 %outputs_4#1 {handshake.bb = 3 : ui32, handshake.bufProps = #handshake<bufProps{"2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store0"} : <i5>, <f32>, <>, <i5>, <f32>, <>
    sink %doneResult {handshake.name = "sink3"} : <>
    %71 = addi %56, %68 {handshake.bb = 3 : ui32, handshake.name = "addi3"} : <i7>
    %72 = buffer %71, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer15"} : <i7>
    %73:2 = fork [2] %72 {handshake.bb = 3 : ui32, handshake.name = "fork12"} : <i7>
    %74 = trunci %73#0 {handshake.bb = 3 : ui32, handshake.name = "trunci5"} : <i7> to <i6>
    %75 = cmpi ult, %73#1, %65 {handshake.bb = 3 : ui32, handshake.name = "cmpi1"} : <i7>
    %76 = buffer %75, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer16"} : <i1>
    %77:4 = fork [4] %76 {handshake.bb = 3 : ui32, handshake.name = "fork13"} : <i1>
    %trueResult_22, %falseResult_23 = cond_br %77#0, %74 {handshake.bb = 3 : ui32, handshake.name = "cond_br8"} : <i1>, <i6>
    sink %falseResult_23 {handshake.name = "sink4"} : <i6>
    %trueResult_24, %falseResult_25 = cond_br %77#2, %59#2 {handshake.bb = 3 : ui32, handshake.name = "cond_br9"} : <i1>, <>
    %trueResult_26, %falseResult_27 = cond_br %77#3, %62 {handshake.bb = 3 : ui32, handshake.name = "cond_br10"} : <i1>, <i1>
    sink %trueResult_26 {handshake.name = "sink5"} : <i1>
    %78 = extsi %falseResult_27 {handshake.bb = 3 : ui32, handshake.name = "extsi15"} : <i1> to <i6>
    %79 = init %148#1 {ftd.imerge, handshake.bb = 4 : ui32, handshake.name = "init1"} : <i1>
    sink %79 {handshake.name = "sink6"} : <i1>
    %80:2 = unbundle %88#0  {handshake.bb = 4 : ui32, handshake.name = "unbundle2"} : <f32> to _ 
    %81 = mux %index_29 [%78, %trueResult_49] {handshake.bb = 4 : ui32, handshake.name = "mux4"} : <i1>, [<i6>, <i6>] to <i6>
    %82 = buffer %81, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer17"} : <i6>
    %83:2 = fork [2] %82 {handshake.bb = 4 : ui32, handshake.name = "fork14"} : <i6>
    %84 = trunci %83#0 {handshake.bb = 4 : ui32, handshake.name = "trunci6"} : <i6> to <i5>
    %result_28, %index_29 = control_merge [%falseResult_25, %trueResult_51]  {handshake.bb = 4 : ui32, handshake.name = "control_merge3"} : [<>, <>] to <>, <i1>
    %85:2 = fork [2] %result_28 {handshake.bb = 4 : ui32, handshake.name = "fork15"} : <>
    %86 = constant %85#0 {handshake.bb = 4 : ui32, handshake.name = "constant22", value = false} : <>, <i1>
    %87 = buffer %80#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer1"} : <>
    %addressResult_30, %dataResult_31 = load[%84] %outputs_2#0 {handshake.bb = 4 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["store1", 2, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load3"} : <i5>, <f32>, <i5>, <f32>
    %88:2 = fork [2] %dataResult_31 {handshake.bb = 4 : ui32, handshake.name = "fork16"} : <f32>
    %89 = extsi %86 {handshake.bb = 4 : ui32, handshake.name = "extsi14"} : <i1> to <i6>
    %90 = mux %103#1 [%89, %trueResult_38] {handshake.bb = 5 : ui32, handshake.name = "mux5"} : <i1>, [<i6>, <i6>] to <i6>
    %91 = buffer %90, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 5 : ui32, handshake.name = "buffer18"} : <i6>
    %92:3 = fork [3] %91 {handshake.bb = 5 : ui32, handshake.name = "fork17"} : <i6>
    %93 = extsi %92#1 {handshake.bb = 5 : ui32, handshake.name = "extsi28"} : <i6> to <i7>
    %94 = extsi %92#2 {handshake.bb = 5 : ui32, handshake.name = "extsi29"} : <i6> to <i12>
    %95 = trunci %92#0 {handshake.bb = 5 : ui32, handshake.name = "trunci7"} : <i6> to <i5>
    %96 = mux %97 [%88#1, %trueResult_40] {handshake.bb = 5 : ui32, handshake.name = "mux6"} : <i1>, [<f32>, <f32>] to <f32>
    %97 = buffer %103#2, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 5 : ui32, handshake.name = "buffer39"} : <i1>
    %98 = mux %103#0 [%83#1, %trueResult_42] {handshake.bb = 5 : ui32, handshake.name = "mux7"} : <i1>, [<i6>, <i6>] to <i6>
    %99 = buffer %98, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 5 : ui32, handshake.name = "buffer21"} : <i6>
    %100 = buffer %99, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 5 : ui32, handshake.name = "buffer22"} : <i6>
    %101:2 = fork [2] %100 {handshake.bb = 5 : ui32, handshake.name = "fork18"} : <i6>
    %102 = extsi %101#0 {handshake.bb = 5 : ui32, handshake.name = "extsi30"} : <i6> to <i10>
    %result_32, %index_33 = control_merge [%85#1, %trueResult_44]  {handshake.bb = 5 : ui32, handshake.name = "control_merge4"} : [<>, <>] to <>, <i1>
    %103:3 = fork [3] %index_33 {handshake.bb = 5 : ui32, handshake.name = "fork19"} : <i1>
    %104 = source {handshake.bb = 5 : ui32, handshake.name = "source4"} : <>
    %105 = constant %104 {handshake.bb = 5 : ui32, handshake.name = "constant23", value = 30 : i6} : <>, <i6>
    %106:2 = fork [2] %105 {handshake.bb = 5 : ui32, handshake.name = "fork20"} : <i6>
    %107 = extsi %106#0 {handshake.bb = 5 : ui32, handshake.name = "extsi31"} : <i6> to <i7>
    %108 = extsi %106#1 {handshake.bb = 5 : ui32, handshake.name = "extsi32"} : <i6> to <i12>
    %109 = source {handshake.bb = 5 : ui32, handshake.name = "source5"} : <>
    %110 = constant %109 {handshake.bb = 5 : ui32, handshake.name = "constant24", value = 1 : i2} : <>, <i2>
    %111 = extsi %110 {handshake.bb = 5 : ui32, handshake.name = "extsi33"} : <i2> to <i7>
    %112 = muli %94, %108 {handshake.bb = 5 : ui32, handshake.name = "muli1"} : <i12>
    %113 = trunci %112 {handshake.bb = 5 : ui32, handshake.name = "trunci8"} : <i12> to <i10>
    %114 = addi %102, %113 {handshake.bb = 5 : ui32, handshake.name = "addi1"} : <i10>
    %addressResult_34, %dataResult_35 = load[%114] %outputs_6#1 {handshake.bb = 5 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load4"} : <i10>, <f32>, <i10>, <f32>
    %addressResult_36, %dataResult_37 = load[%95] %outputs {handshake.bb = 5 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load5"} : <i5>, <f32>, <i5>, <f32>
    %115 = mulf %dataResult_35, %dataResult_37 {fastmath = #arith.fastmath<none>, fpu_impl = #handshake<fpu_impl flopoco>, handshake.bb = 5 : ui32, handshake.name = "mulf1", internal_delay = "2_875333"} : <f32>
    %116 = buffer %96, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 5 : ui32, handshake.name = "buffer20"} : <f32>
    %117 = addf %116, %115 {fastmath = #arith.fastmath<none>, fpu_impl = #handshake<fpu_impl flopoco>, handshake.bb = 5 : ui32, handshake.name = "addf1", internal_delay = "2_922000"} : <f32>
    %118 = addi %93, %111 {handshake.bb = 5 : ui32, handshake.name = "addi4"} : <i7>
    %119 = buffer %118, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 5 : ui32, handshake.name = "buffer25"} : <i7>
    %120:2 = fork [2] %119 {handshake.bb = 5 : ui32, handshake.name = "fork21"} : <i7>
    %121 = trunci %120#0 {handshake.bb = 5 : ui32, handshake.name = "trunci9"} : <i7> to <i6>
    %122 = cmpi ult, %120#1, %107 {handshake.bb = 5 : ui32, handshake.name = "cmpi2"} : <i7>
    %123 = buffer %122, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 5 : ui32, handshake.name = "buffer26"} : <i1>
    %124:4 = fork [4] %123 {handshake.bb = 5 : ui32, handshake.name = "fork22"} : <i1>
    %trueResult_38, %falseResult_39 = cond_br %124#0, %121 {handshake.bb = 5 : ui32, handshake.name = "cond_br11"} : <i1>, <i6>
    sink %falseResult_39 {handshake.name = "sink7"} : <i6>
    %trueResult_40, %falseResult_41 = cond_br %125, %117 {handshake.bb = 5 : ui32, handshake.name = "cond_br12"} : <i1>, <f32>
    %125 = buffer %124#2, bufferType = FIFO_BREAK_NONE, numSlots = 3 {handshake.bb = 5 : ui32, handshake.name = "buffer47"} : <i1>
    %trueResult_42, %falseResult_43 = cond_br %124#1, %101#1 {handshake.bb = 5 : ui32, handshake.name = "cond_br13"} : <i1>, <i6>
    %126 = buffer %result_32, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 5 : ui32, handshake.name = "buffer23"} : <>
    %trueResult_44, %falseResult_45 = cond_br %124#3, %126 {handshake.bb = 5 : ui32, handshake.name = "cond_br14"} : <i1>, <>
    %127:2 = fork [2] %falseResult_43 {handshake.bb = 6 : ui32, handshake.name = "fork23"} : <i6>
    %128 = extsi %127#0 {handshake.bb = 6 : ui32, handshake.name = "extsi34"} : <i6> to <i7>
    %129 = extsi %130 {handshake.bb = 6 : ui32, handshake.name = "extsi35"} : <i6> to <i32>
    %130 = buffer %127#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 6 : ui32, handshake.name = "buffer52"} : <i6>
    %131:2 = fork [2] %falseResult_45 {handshake.bb = 6 : ui32, handshake.name = "fork24"} : <>
    %132 = constant %131#0 {handshake.bb = 6 : ui32, handshake.name = "constant25", value = 1 : i2} : <>, <i2>
    %133 = extsi %132 {handshake.bb = 6 : ui32, handshake.name = "extsi11"} : <i2> to <i32>
    %134 = source {handshake.bb = 6 : ui32, handshake.name = "source6"} : <>
    %135 = constant %134 {handshake.bb = 6 : ui32, handshake.name = "constant26", value = 30 : i6} : <>, <i6>
    %136 = extsi %135 {handshake.bb = 6 : ui32, handshake.name = "extsi36"} : <i6> to <i7>
    %137 = source {handshake.bb = 6 : ui32, handshake.name = "source7"} : <>
    %138 = constant %137 {handshake.bb = 6 : ui32, handshake.name = "constant27", value = 1 : i2} : <>, <i2>
    %139 = extsi %138 {handshake.bb = 6 : ui32, handshake.name = "extsi37"} : <i2> to <i7>
    %140 = gate %129, %87 {handshake.bb = 6 : ui32, handshake.name = "gate1"} : <i32>, !handshake.control<> to <i32>
    %141 = trunci %140 {handshake.bb = 6 : ui32, handshake.name = "trunci10"} : <i32> to <i5>
    %addressResult_46, %dataResult_47, %doneResult_48 = store[%141] %falseResult_41 %outputs_2#1 {handshake.bb = 6 : ui32, handshake.bufProps = #handshake<bufProps{"2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store1"} : <i5>, <f32>, <>, <i5>, <f32>, <>
    sink %doneResult_48 {handshake.name = "sink9"} : <>
    %142 = addi %128, %139 {handshake.bb = 6 : ui32, handshake.name = "addi5"} : <i7>
    %143 = buffer %142, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 6 : ui32, handshake.name = "buffer27"} : <i7>
    %144:2 = fork [2] %143 {handshake.bb = 6 : ui32, handshake.name = "fork25"} : <i7>
    %145 = trunci %144#0 {handshake.bb = 6 : ui32, handshake.name = "trunci11"} : <i7> to <i6>
    %146 = cmpi ult, %144#1, %136 {handshake.bb = 6 : ui32, handshake.name = "cmpi3"} : <i7>
    %147 = buffer %146, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 6 : ui32, handshake.name = "buffer28"} : <i1>
    %148:3 = fork [3] %147 {handshake.bb = 6 : ui32, handshake.name = "fork26"} : <i1>
    %trueResult_49, %falseResult_50 = cond_br %148#0, %145 {handshake.bb = 6 : ui32, handshake.name = "cond_br15"} : <i1>, <i6>
    sink %falseResult_50 {handshake.name = "sink10"} : <i6>
    %trueResult_51, %falseResult_52 = cond_br %148#2, %131#1 {handshake.bb = 6 : ui32, handshake.name = "cond_br16"} : <i1>, <>
    %149:5 = fork [5] %falseResult_52 {handshake.bb = 7 : ui32, handshake.name = "fork27"} : <>
    end {handshake.bb = 7 : ui32, handshake.name = "end0"} %memEnd_7, %memEnd_5, %memEnd_3, %memEnd_1, %memEnd, %0#1 : <>, <>, <>, <>, <>, <>
  }
}

