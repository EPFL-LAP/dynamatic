module {
  handshake.func @mvt_float(%arg0: memref<900xf32>, %arg1: memref<30xf32>, %arg2: memref<30xf32>, %arg3: memref<30xf32>, %arg4: memref<30xf32>, %arg5: !handshake.control<>, %arg6: !handshake.control<>, %arg7: !handshake.control<>, %arg8: !handshake.control<>, %arg9: !handshake.control<>, %arg10: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["A", "x1", "x2", "y1", "y2", "A_start", "x1_start", "x2_start", "y1_start", "y2_start", "start"], cfg.edges = "[0,1][2,2,3,cmpi0][4,5][6,4,7,cmpi3][1,2][3,1,4,cmpi1][5,5,6,cmpi2]", resNames = ["A_end", "x1_end", "x2_end", "y1_end", "y2_end", "end"]} {
    %0:3 = fork [3] %arg10 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork0"} : <>
    %outputs, %memEnd = mem_controller[%arg4 : memref<30xf32>] %arg9 (%addressResult_38) %163#4 {connectedBlocks = [5 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller3"} :    (!handshake.channel<i5>) -> !handshake.channel<f32>
    %outputs_0, %memEnd_1 = mem_controller[%arg3 : memref<30xf32>] %arg8 (%addressResult_12) %163#3 {connectedBlocks = [2 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller4"} :    (!handshake.channel<i5>) -> !handshake.channel<f32>
    %outputs_2:2, %memEnd_3 = mem_controller[%arg2 : memref<30xf32>] %arg7 (%addressResult_32, %147, %addressResult_50, %dataResult_51) %163#2 {connectedBlocks = [4 : i32, 6 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "5": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller5"} :    (!handshake.channel<i5>, !handshake.channel<i32>, !handshake.channel<i5>, !handshake.channel<f32>) -> (!handshake.channel<f32>, !handshake.control<>)
    %outputs_4:2, %memEnd_5 = mem_controller[%arg1 : memref<30xf32>] %arg6 (%addressResult, %69, %addressResult_22, %dataResult_23) %163#1 {connectedBlocks = [1 : i32, 3 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "5": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller6"} :    (!handshake.channel<i5>, !handshake.channel<i32>, !handshake.channel<i5>, !handshake.channel<f32>) -> (!handshake.channel<f32>, !handshake.control<>)
    %outputs_6:2, %memEnd_7 = mem_controller[%arg0 : memref<900xf32>] %arg5 (%addressResult_10, %addressResult_36) %163#0 {connectedBlocks = [2 : i32, 5 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller7"} :    (!handshake.channel<i10>, !handshake.channel<i10>) -> (!handshake.channel<f32>, !handshake.channel<f32>)
    %1 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant14", value = false} : <>, <i1>
    %2 = br %1 {handshake.bb = 0 : ui32, handshake.name = "br2"} : <i1>
    %3 = extsi %2 {handshake.bb = 0 : ui32, handshake.name = "extsi17"} : <i1> to <i6>
    %4 = br %0#2 {handshake.bb = 0 : ui32, handshake.name = "br4"} : <>
    %5 = init %85#1 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "init0"} : <i1>
    sink %5 {handshake.name = "sink0"} : <i1>
    %6:2 = unbundle %14#0  {handshake.bb = 1 : ui32, handshake.name = "unbundle0"} : <f32> to _ 
    %7 = mux %index [%3, %trueResult_24] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i6>, <i6>] to <i6>
    %8 = buffer %7, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer2"} : <i6>
    %9:2 = fork [2] %8 {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <i6>
    %10 = trunci %9#0 {handshake.bb = 1 : ui32, handshake.name = "trunci0"} : <i6> to <i5>
    %result, %index = control_merge [%4, %trueResult_26]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %11:2 = fork [2] %result {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <>
    %12 = constant %11#0 {handshake.bb = 1 : ui32, handshake.name = "constant15", value = false} : <>, <i1>
    %13 = buffer %6#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer0"} : <>
    %addressResult, %dataResult = load[%10] %outputs_4#0 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["store0", 2, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i5>, <f32>, <i5>, <f32>
    %14:2 = fork [2] %dataResult {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <f32>
    %15 = br %12 {handshake.bb = 1 : ui32, handshake.name = "br5"} : <i1>
    %16 = extsi %15 {handshake.bb = 1 : ui32, handshake.name = "extsi16"} : <i1> to <i6>
    %17 = br %14#1 {handshake.bb = 1 : ui32, handshake.name = "br6"} : <f32>
    %18 = br %9#1 {handshake.bb = 1 : ui32, handshake.name = "br7"} : <i6>
    %19 = br %11#1 {handshake.bb = 1 : ui32, handshake.name = "br8"} : <>
    %20 = mux %36#1 [%16, %trueResult] {handshake.bb = 2 : ui32, handshake.name = "mux1"} : <i1>, [<i6>, <i6>] to <i6>
    %21 = buffer %20, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer3"} : <i6>
    %22:3 = fork [3] %21 {handshake.bb = 2 : ui32, handshake.name = "fork4"} : <i6>
    %23 = extsi %24 {handshake.bb = 2 : ui32, handshake.name = "extsi18"} : <i6> to <i10>
    %24 = buffer %22#0, bufferType = FIFO_BREAK_NONE, numSlots = 9 {handshake.bb = 2 : ui32, handshake.name = "buffer8"} : <i6>
    %25 = extsi %22#2 {handshake.bb = 2 : ui32, handshake.name = "extsi19"} : <i6> to <i7>
    %26 = trunci %27 {handshake.bb = 2 : ui32, handshake.name = "trunci1"} : <i6> to <i5>
    %27 = buffer %22#1, bufferType = FIFO_BREAK_NONE, numSlots = 9 {handshake.bb = 2 : ui32, handshake.name = "buffer10"} : <i6>
    %28 = mux %29 [%17, %trueResult_14] {handshake.bb = 2 : ui32, handshake.name = "mux2"} : <i1>, [<f32>, <f32>] to <f32>
    %29 = buffer %36#2, bufferType = FIFO_BREAK_NONE, numSlots = 15 {handshake.bb = 2 : ui32, handshake.name = "buffer11"} : <i1>
    %30 = mux %36#0 [%18, %trueResult_16] {handshake.bb = 2 : ui32, handshake.name = "mux3"} : <i1>, [<i6>, <i6>] to <i6>
    %31 = buffer %30, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer5"} : <i6>
    %32 = buffer %31, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer6"} : <i6>
    %33:2 = fork [2] %32 {handshake.bb = 2 : ui32, handshake.name = "fork5"} : <i6>
    %34 = extsi %35 {handshake.bb = 2 : ui32, handshake.name = "extsi20"} : <i6> to <i12>
    %35 = buffer %33#1, bufferType = FIFO_BREAK_NONE, numSlots = 5 {handshake.bb = 2 : ui32, handshake.name = "buffer13"} : <i6>
    %result_8, %index_9 = control_merge [%19, %trueResult_18]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>, <>] to <>, <i1>
    %36:3 = fork [3] %index_9 {handshake.bb = 2 : ui32, handshake.name = "fork6"} : <i1>
    %37 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %38 = constant %37 {handshake.bb = 2 : ui32, handshake.name = "constant16", value = 30 : i6} : <>, <i6>
    %39:2 = fork [2] %38 {handshake.bb = 2 : ui32, handshake.name = "fork7"} : <i6>
    %40 = extsi %41 {handshake.bb = 2 : ui32, handshake.name = "extsi21"} : <i6> to <i12>
    %41 = buffer %39#0, bufferType = FIFO_BREAK_NONE, numSlots = 5 {handshake.bb = 2 : ui32, handshake.name = "buffer14"} : <i6>
    %42 = extsi %39#1 {handshake.bb = 2 : ui32, handshake.name = "extsi22"} : <i6> to <i7>
    %43 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %44 = constant %43 {handshake.bb = 2 : ui32, handshake.name = "constant17", value = 1 : i2} : <>, <i2>
    %45 = extsi %44 {handshake.bb = 2 : ui32, handshake.name = "extsi23"} : <i2> to <i7>
    %46 = muli %34, %40 {handshake.bb = 2 : ui32, handshake.name = "muli0"} : <i12>
    %47 = trunci %46 {handshake.bb = 2 : ui32, handshake.name = "trunci2"} : <i12> to <i10>
    %48 = addi %23, %47 {handshake.bb = 2 : ui32, handshake.name = "addi0"} : <i10>
    %addressResult_10, %dataResult_11 = load[%48] %outputs_6#0 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i10>, <f32>, <i10>, <f32>
    %addressResult_12, %dataResult_13 = load[%26] %outputs_0 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load2"} : <i5>, <f32>, <i5>, <f32>
    %49 = mulf %dataResult_11, %dataResult_13 {fastmath = #arith.fastmath<none>, fpu_impl = #handshake<fpu_impl flopoco>, handshake.bb = 2 : ui32, handshake.name = "mulf0", internal_delay = "2_875333"} : <f32>
    %50 = buffer %28, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer4"} : <f32>
    %51 = addf %50, %49 {fastmath = #arith.fastmath<none>, fpu_impl = #handshake<fpu_impl flopoco>, handshake.bb = 2 : ui32, handshake.name = "addf0", internal_delay = "2_922000"} : <f32>
    %52 = addi %25, %45 {handshake.bb = 2 : ui32, handshake.name = "addi2"} : <i7>
    %53 = buffer %52, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer9"} : <i7>
    %54:2 = fork [2] %53 {handshake.bb = 2 : ui32, handshake.name = "fork8"} : <i7>
    %55 = trunci %54#0 {handshake.bb = 2 : ui32, handshake.name = "trunci3"} : <i7> to <i6>
    %56 = cmpi ult, %54#1, %42 {handshake.bb = 2 : ui32, handshake.name = "cmpi0"} : <i7>
    %57 = buffer %56, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer12"} : <i1>
    %58:4 = fork [4] %57 {handshake.bb = 2 : ui32, handshake.name = "fork9"} : <i1>
    %trueResult, %falseResult = cond_br %58#0, %55 {handshake.bb = 2 : ui32, handshake.name = "cond_br4"} : <i1>, <i6>
    sink %falseResult {handshake.name = "sink1"} : <i6>
    %trueResult_14, %falseResult_15 = cond_br %59, %51 {handshake.bb = 2 : ui32, handshake.name = "cond_br5"} : <i1>, <f32>
    %59 = buffer %58#2, bufferType = FIFO_BREAK_NONE, numSlots = 15 {handshake.bb = 2 : ui32, handshake.name = "buffer19"} : <i1>
    %trueResult_16, %falseResult_17 = cond_br %58#1, %33#0 {handshake.bb = 2 : ui32, handshake.name = "cond_br6"} : <i1>, <i6>
    %60 = buffer %result_8, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer7"} : <>
    %trueResult_18, %falseResult_19 = cond_br %58#3, %60 {handshake.bb = 2 : ui32, handshake.name = "cond_br7"} : <i1>, <>
    %61 = merge %falseResult_17 {handshake.bb = 3 : ui32, handshake.name = "merge0"} : <i6>
    %62:2 = fork [2] %61 {handshake.bb = 3 : ui32, handshake.name = "fork10"} : <i6>
    %63 = extsi %62#0 {handshake.bb = 3 : ui32, handshake.name = "extsi24"} : <i6> to <i7>
    %64 = extsi %65 {handshake.bb = 3 : ui32, handshake.name = "extsi25"} : <i6> to <i32>
    %65 = buffer %62#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer24"} : <i6>
    %66 = merge %falseResult_15 {handshake.bb = 3 : ui32, handshake.name = "merge1"} : <f32>
    %result_20, %index_21 = control_merge [%falseResult_19]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>] to <>, <i1>
    sink %index_21 {handshake.name = "sink2"} : <i1>
    %67:3 = fork [3] %result_20 {handshake.bb = 3 : ui32, handshake.name = "fork11"} : <>
    %68 = constant %67#1 {handshake.bb = 3 : ui32, handshake.name = "constant18", value = 1 : i2} : <>, <i2>
    %69 = extsi %68 {handshake.bb = 3 : ui32, handshake.name = "extsi4"} : <i2> to <i32>
    %70 = constant %67#0 {handshake.bb = 3 : ui32, handshake.name = "constant19", value = false} : <>, <i1>
    %71 = source {handshake.bb = 3 : ui32, handshake.name = "source2"} : <>
    %72 = constant %71 {handshake.bb = 3 : ui32, handshake.name = "constant20", value = 30 : i6} : <>, <i6>
    %73 = extsi %72 {handshake.bb = 3 : ui32, handshake.name = "extsi26"} : <i6> to <i7>
    %74 = source {handshake.bb = 3 : ui32, handshake.name = "source3"} : <>
    %75 = constant %74 {handshake.bb = 3 : ui32, handshake.name = "constant21", value = 1 : i2} : <>, <i2>
    %76 = extsi %75 {handshake.bb = 3 : ui32, handshake.name = "extsi27"} : <i2> to <i7>
    %77 = gate %64, %13 {handshake.bb = 3 : ui32, handshake.name = "gate0"} : <i32>, !handshake.control<> to <i32>
    %78 = trunci %77 {handshake.bb = 3 : ui32, handshake.name = "trunci4"} : <i32> to <i5>
    %addressResult_22, %dataResult_23, %doneResult = store[%78] %66 %outputs_4#1 {handshake.bb = 3 : ui32, handshake.bufProps = #handshake<bufProps{"2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store0"} : <i5>, <f32>, <>, <i5>, <f32>, <>
    sink %doneResult {handshake.name = "sink3"} : <>
    %79 = addi %63, %76 {handshake.bb = 3 : ui32, handshake.name = "addi3"} : <i7>
    %80 = buffer %79, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer15"} : <i7>
    %81:2 = fork [2] %80 {handshake.bb = 3 : ui32, handshake.name = "fork12"} : <i7>
    %82 = trunci %81#0 {handshake.bb = 3 : ui32, handshake.name = "trunci5"} : <i7> to <i6>
    %83 = cmpi ult, %81#1, %73 {handshake.bb = 3 : ui32, handshake.name = "cmpi1"} : <i7>
    %84 = buffer %83, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer16"} : <i1>
    %85:4 = fork [4] %84 {handshake.bb = 3 : ui32, handshake.name = "fork13"} : <i1>
    %trueResult_24, %falseResult_25 = cond_br %85#0, %82 {handshake.bb = 3 : ui32, handshake.name = "cond_br8"} : <i1>, <i6>
    sink %falseResult_25 {handshake.name = "sink4"} : <i6>
    %trueResult_26, %falseResult_27 = cond_br %85#2, %67#2 {handshake.bb = 3 : ui32, handshake.name = "cond_br9"} : <i1>, <>
    %trueResult_28, %falseResult_29 = cond_br %85#3, %70 {handshake.bb = 3 : ui32, handshake.name = "cond_br10"} : <i1>, <i1>
    sink %trueResult_28 {handshake.name = "sink5"} : <i1>
    %86 = extsi %falseResult_29 {handshake.bb = 3 : ui32, handshake.name = "extsi15"} : <i1> to <i6>
    %87 = init %162#1 {ftd.imerge, handshake.bb = 4 : ui32, handshake.name = "init1"} : <i1>
    sink %87 {handshake.name = "sink6"} : <i1>
    %88:2 = unbundle %96#0  {handshake.bb = 4 : ui32, handshake.name = "unbundle2"} : <f32> to _ 
    %89 = mux %index_31 [%86, %trueResult_53] {handshake.bb = 4 : ui32, handshake.name = "mux4"} : <i1>, [<i6>, <i6>] to <i6>
    %90 = buffer %89, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer17"} : <i6>
    %91:2 = fork [2] %90 {handshake.bb = 4 : ui32, handshake.name = "fork14"} : <i6>
    %92 = trunci %91#0 {handshake.bb = 4 : ui32, handshake.name = "trunci6"} : <i6> to <i5>
    %result_30, %index_31 = control_merge [%falseResult_27, %trueResult_55]  {handshake.bb = 4 : ui32, handshake.name = "control_merge3"} : [<>, <>] to <>, <i1>
    %93:2 = fork [2] %result_30 {handshake.bb = 4 : ui32, handshake.name = "fork15"} : <>
    %94 = constant %93#0 {handshake.bb = 4 : ui32, handshake.name = "constant22", value = false} : <>, <i1>
    %95 = buffer %88#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer1"} : <>
    %addressResult_32, %dataResult_33 = load[%92] %outputs_2#0 {handshake.bb = 4 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["store1", 2, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load3"} : <i5>, <f32>, <i5>, <f32>
    %96:2 = fork [2] %dataResult_33 {handshake.bb = 4 : ui32, handshake.name = "fork16"} : <f32>
    %97 = br %94 {handshake.bb = 4 : ui32, handshake.name = "br9"} : <i1>
    %98 = extsi %97 {handshake.bb = 4 : ui32, handshake.name = "extsi14"} : <i1> to <i6>
    %99 = br %96#1 {handshake.bb = 4 : ui32, handshake.name = "br10"} : <f32>
    %100 = br %91#1 {handshake.bb = 4 : ui32, handshake.name = "br11"} : <i6>
    %101 = br %93#1 {handshake.bb = 4 : ui32, handshake.name = "br12"} : <>
    %102 = mux %115#1 [%98, %trueResult_40] {handshake.bb = 5 : ui32, handshake.name = "mux5"} : <i1>, [<i6>, <i6>] to <i6>
    %103 = buffer %102, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 5 : ui32, handshake.name = "buffer18"} : <i6>
    %104:3 = fork [3] %103 {handshake.bb = 5 : ui32, handshake.name = "fork17"} : <i6>
    %105 = extsi %104#1 {handshake.bb = 5 : ui32, handshake.name = "extsi28"} : <i6> to <i7>
    %106 = extsi %104#2 {handshake.bb = 5 : ui32, handshake.name = "extsi29"} : <i6> to <i12>
    %107 = trunci %104#0 {handshake.bb = 5 : ui32, handshake.name = "trunci7"} : <i6> to <i5>
    %108 = mux %109 [%99, %trueResult_42] {handshake.bb = 5 : ui32, handshake.name = "mux6"} : <i1>, [<f32>, <f32>] to <f32>
    %109 = buffer %115#2, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 5 : ui32, handshake.name = "buffer39"} : <i1>
    %110 = mux %115#0 [%100, %trueResult_44] {handshake.bb = 5 : ui32, handshake.name = "mux7"} : <i1>, [<i6>, <i6>] to <i6>
    %111 = buffer %110, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 5 : ui32, handshake.name = "buffer21"} : <i6>
    %112 = buffer %111, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 5 : ui32, handshake.name = "buffer22"} : <i6>
    %113:2 = fork [2] %112 {handshake.bb = 5 : ui32, handshake.name = "fork18"} : <i6>
    %114 = extsi %113#0 {handshake.bb = 5 : ui32, handshake.name = "extsi30"} : <i6> to <i10>
    %result_34, %index_35 = control_merge [%101, %trueResult_46]  {handshake.bb = 5 : ui32, handshake.name = "control_merge4"} : [<>, <>] to <>, <i1>
    %115:3 = fork [3] %index_35 {handshake.bb = 5 : ui32, handshake.name = "fork19"} : <i1>
    %116 = source {handshake.bb = 5 : ui32, handshake.name = "source4"} : <>
    %117 = constant %116 {handshake.bb = 5 : ui32, handshake.name = "constant23", value = 30 : i6} : <>, <i6>
    %118:2 = fork [2] %117 {handshake.bb = 5 : ui32, handshake.name = "fork20"} : <i6>
    %119 = extsi %118#0 {handshake.bb = 5 : ui32, handshake.name = "extsi31"} : <i6> to <i7>
    %120 = extsi %118#1 {handshake.bb = 5 : ui32, handshake.name = "extsi32"} : <i6> to <i12>
    %121 = source {handshake.bb = 5 : ui32, handshake.name = "source5"} : <>
    %122 = constant %121 {handshake.bb = 5 : ui32, handshake.name = "constant24", value = 1 : i2} : <>, <i2>
    %123 = extsi %122 {handshake.bb = 5 : ui32, handshake.name = "extsi33"} : <i2> to <i7>
    %124 = muli %106, %120 {handshake.bb = 5 : ui32, handshake.name = "muli1"} : <i12>
    %125 = trunci %124 {handshake.bb = 5 : ui32, handshake.name = "trunci8"} : <i12> to <i10>
    %126 = addi %114, %125 {handshake.bb = 5 : ui32, handshake.name = "addi1"} : <i10>
    %addressResult_36, %dataResult_37 = load[%126] %outputs_6#1 {handshake.bb = 5 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load4"} : <i10>, <f32>, <i10>, <f32>
    %addressResult_38, %dataResult_39 = load[%107] %outputs {handshake.bb = 5 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load5"} : <i5>, <f32>, <i5>, <f32>
    %127 = mulf %dataResult_37, %dataResult_39 {fastmath = #arith.fastmath<none>, fpu_impl = #handshake<fpu_impl flopoco>, handshake.bb = 5 : ui32, handshake.name = "mulf1", internal_delay = "2_875333"} : <f32>
    %128 = buffer %108, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 5 : ui32, handshake.name = "buffer20"} : <f32>
    %129 = addf %128, %127 {fastmath = #arith.fastmath<none>, fpu_impl = #handshake<fpu_impl flopoco>, handshake.bb = 5 : ui32, handshake.name = "addf1", internal_delay = "2_922000"} : <f32>
    %130 = addi %105, %123 {handshake.bb = 5 : ui32, handshake.name = "addi4"} : <i7>
    %131 = buffer %130, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 5 : ui32, handshake.name = "buffer25"} : <i7>
    %132:2 = fork [2] %131 {handshake.bb = 5 : ui32, handshake.name = "fork21"} : <i7>
    %133 = trunci %132#0 {handshake.bb = 5 : ui32, handshake.name = "trunci9"} : <i7> to <i6>
    %134 = cmpi ult, %132#1, %119 {handshake.bb = 5 : ui32, handshake.name = "cmpi2"} : <i7>
    %135 = buffer %134, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 5 : ui32, handshake.name = "buffer26"} : <i1>
    %136:4 = fork [4] %135 {handshake.bb = 5 : ui32, handshake.name = "fork22"} : <i1>
    %trueResult_40, %falseResult_41 = cond_br %136#0, %133 {handshake.bb = 5 : ui32, handshake.name = "cond_br11"} : <i1>, <i6>
    sink %falseResult_41 {handshake.name = "sink7"} : <i6>
    %trueResult_42, %falseResult_43 = cond_br %137, %129 {handshake.bb = 5 : ui32, handshake.name = "cond_br12"} : <i1>, <f32>
    %137 = buffer %136#2, bufferType = FIFO_BREAK_NONE, numSlots = 3 {handshake.bb = 5 : ui32, handshake.name = "buffer47"} : <i1>
    %trueResult_44, %falseResult_45 = cond_br %136#1, %113#1 {handshake.bb = 5 : ui32, handshake.name = "cond_br13"} : <i1>, <i6>
    %138 = buffer %result_34, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 5 : ui32, handshake.name = "buffer23"} : <>
    %trueResult_46, %falseResult_47 = cond_br %136#3, %138 {handshake.bb = 5 : ui32, handshake.name = "cond_br14"} : <i1>, <>
    %139 = merge %falseResult_45 {handshake.bb = 6 : ui32, handshake.name = "merge2"} : <i6>
    %140:2 = fork [2] %139 {handshake.bb = 6 : ui32, handshake.name = "fork23"} : <i6>
    %141 = extsi %140#0 {handshake.bb = 6 : ui32, handshake.name = "extsi34"} : <i6> to <i7>
    %142 = extsi %143 {handshake.bb = 6 : ui32, handshake.name = "extsi35"} : <i6> to <i32>
    %143 = buffer %140#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 6 : ui32, handshake.name = "buffer52"} : <i6>
    %144 = merge %falseResult_43 {handshake.bb = 6 : ui32, handshake.name = "merge3"} : <f32>
    %result_48, %index_49 = control_merge [%falseResult_47]  {handshake.bb = 6 : ui32, handshake.name = "control_merge5"} : [<>] to <>, <i1>
    sink %index_49 {handshake.name = "sink8"} : <i1>
    %145:2 = fork [2] %result_48 {handshake.bb = 6 : ui32, handshake.name = "fork24"} : <>
    %146 = constant %145#0 {handshake.bb = 6 : ui32, handshake.name = "constant25", value = 1 : i2} : <>, <i2>
    %147 = extsi %146 {handshake.bb = 6 : ui32, handshake.name = "extsi11"} : <i2> to <i32>
    %148 = source {handshake.bb = 6 : ui32, handshake.name = "source6"} : <>
    %149 = constant %148 {handshake.bb = 6 : ui32, handshake.name = "constant26", value = 30 : i6} : <>, <i6>
    %150 = extsi %149 {handshake.bb = 6 : ui32, handshake.name = "extsi36"} : <i6> to <i7>
    %151 = source {handshake.bb = 6 : ui32, handshake.name = "source7"} : <>
    %152 = constant %151 {handshake.bb = 6 : ui32, handshake.name = "constant27", value = 1 : i2} : <>, <i2>
    %153 = extsi %152 {handshake.bb = 6 : ui32, handshake.name = "extsi37"} : <i2> to <i7>
    %154 = gate %142, %95 {handshake.bb = 6 : ui32, handshake.name = "gate1"} : <i32>, !handshake.control<> to <i32>
    %155 = trunci %154 {handshake.bb = 6 : ui32, handshake.name = "trunci10"} : <i32> to <i5>
    %addressResult_50, %dataResult_51, %doneResult_52 = store[%155] %144 %outputs_2#1 {handshake.bb = 6 : ui32, handshake.bufProps = #handshake<bufProps{"2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store1"} : <i5>, <f32>, <>, <i5>, <f32>, <>
    sink %doneResult_52 {handshake.name = "sink9"} : <>
    %156 = addi %141, %153 {handshake.bb = 6 : ui32, handshake.name = "addi5"} : <i7>
    %157 = buffer %156, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 6 : ui32, handshake.name = "buffer27"} : <i7>
    %158:2 = fork [2] %157 {handshake.bb = 6 : ui32, handshake.name = "fork25"} : <i7>
    %159 = trunci %158#0 {handshake.bb = 6 : ui32, handshake.name = "trunci11"} : <i7> to <i6>
    %160 = cmpi ult, %158#1, %150 {handshake.bb = 6 : ui32, handshake.name = "cmpi3"} : <i7>
    %161 = buffer %160, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 6 : ui32, handshake.name = "buffer28"} : <i1>
    %162:3 = fork [3] %161 {handshake.bb = 6 : ui32, handshake.name = "fork26"} : <i1>
    %trueResult_53, %falseResult_54 = cond_br %162#0, %159 {handshake.bb = 6 : ui32, handshake.name = "cond_br15"} : <i1>, <i6>
    sink %falseResult_54 {handshake.name = "sink10"} : <i6>
    %trueResult_55, %falseResult_56 = cond_br %162#2, %145#1 {handshake.bb = 6 : ui32, handshake.name = "cond_br16"} : <i1>, <>
    %result_57, %index_58 = control_merge [%falseResult_56]  {handshake.bb = 7 : ui32, handshake.name = "control_merge6"} : [<>] to <>, <i1>
    sink %index_58 {handshake.name = "sink11"} : <i1>
    %163:5 = fork [5] %result_57 {handshake.bb = 7 : ui32, handshake.name = "fork27"} : <>
    end {handshake.bb = 7 : ui32, handshake.name = "end0"} %memEnd_7, %memEnd_5, %memEnd_3, %memEnd_1, %memEnd, %0#1 : <>, <>, <>, <>, <>, <>
  }
}

