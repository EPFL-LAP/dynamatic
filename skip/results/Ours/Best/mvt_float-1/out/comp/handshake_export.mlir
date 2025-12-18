module {
  handshake.func @mvt_float(%arg0: memref<900xf32>, %arg1: memref<30xf32>, %arg2: memref<30xf32>, %arg3: memref<30xf32>, %arg4: memref<30xf32>, %arg5: !handshake.control<>, %arg6: !handshake.control<>, %arg7: !handshake.control<>, %arg8: !handshake.control<>, %arg9: !handshake.control<>, %arg10: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["A", "x1", "x2", "y1", "y2", "A_start", "x1_start", "x2_start", "y1_start", "y2_start", "start"], cfg.edges = "[0,1][2,2,3,cmpi0][4,5][6,4,7,cmpi3][1,2][3,1,4,cmpi1][5,5,6,cmpi2]", resNames = ["A_end", "x1_end", "x2_end", "y1_end", "y2_end", "end"]} {
    %0:3 = fork [3] %arg10 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork0"} : <>
    %outputs, %memEnd = mem_controller[%arg4 : memref<30xf32>] %arg9 (%addressResult_38) %177#4 {connectedBlocks = [5 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller3"} :    (!handshake.channel<i5>) -> !handshake.channel<f32>
    %outputs_0, %memEnd_1 = mem_controller[%arg3 : memref<30xf32>] %arg8 (%addressResult_12) %177#3 {connectedBlocks = [2 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller4"} :    (!handshake.channel<i5>) -> !handshake.channel<f32>
    %outputs_2:2, %memEnd_3 = mem_controller[%arg2 : memref<30xf32>] %arg7 (%addressResult_32, %151, %addressResult_50, %dataResult_51) %177#2 {connectedBlocks = [4 : i32, 6 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "5": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller5"} :    (!handshake.channel<i5>, !handshake.channel<i32>, !handshake.channel<i5>, !handshake.channel<f32>) -> (!handshake.channel<f32>, !handshake.control<>)
    %outputs_4:2, %memEnd_5 = mem_controller[%arg1 : memref<30xf32>] %arg6 (%addressResult, %65, %addressResult_22, %dataResult_23) %177#1 {connectedBlocks = [1 : i32, 3 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "5": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller6"} :    (!handshake.channel<i5>, !handshake.channel<i32>, !handshake.channel<i5>, !handshake.channel<f32>) -> (!handshake.channel<f32>, !handshake.control<>)
    %outputs_6:2, %memEnd_7 = mem_controller[%arg0 : memref<900xf32>] %arg5 (%addressResult_10, %addressResult_36) %177#0 {connectedBlocks = [2 : i32, 5 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller7"} :    (!handshake.channel<i10>, !handshake.channel<i10>) -> (!handshake.channel<f32>, !handshake.channel<f32>)
    %1 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant0", value = false} : <>, <i1>
    %2 = extsi %1 {handshake.bb = 0 : ui32, handshake.name = "extsi17"} : <i1> to <i6>
    %3 = init %91#1 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "init0"} : <i1>
    sink %3 {handshake.name = "sink0"} : <i1>
    %4:2 = unbundle %16#0  {handshake.bb = 1 : ui32, handshake.name = "unbundle1"} : <f32> to _ 
    %5 = mux %index [%2, %trueResult_24] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i6>, <i6>] to <i6>
    %6 = buffer %5, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer4"} : <i6>
    %7:3 = fork [3] %6 {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <i6>
    %8 = trunci %7#0 {handshake.bb = 1 : ui32, handshake.name = "trunci0"} : <i6> to <i5>
    %result, %index = control_merge [%0#2, %trueResult_26]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %9:2 = fork [2] %result {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <>
    %10 = constant %9#0 {handshake.bb = 1 : ui32, handshake.name = "constant1", value = false} : <>, <i1>
    %11 = buffer %7#2, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer0"} : <i6>
    %12 = extsi %11 {handshake.bb = 1 : ui32, handshake.name = "extsi18"} : <i6> to <i32>
    %13 = buffer %4#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 1 : ui32, handshake.name = "buffer1"} : <>
    %14:2 = fork [2] %13 {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <>
    %15 = init %14#1 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 1 : ui32, handshake.name = "init3"} : <>
    %addressResult, %dataResult = load[%8] %outputs_4#0 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["store0", 2, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i5>, <f32>, <i5>, <f32>
    %16:2 = fork [2] %dataResult {handshake.bb = 1 : ui32, handshake.name = "fork4"} : <f32>
    %17 = extsi %10 {handshake.bb = 1 : ui32, handshake.name = "extsi16"} : <i1> to <i6>
    %18 = mux %33#1 [%17, %trueResult] {handshake.bb = 2 : ui32, handshake.name = "mux1"} : <i1>, [<i6>, <i6>] to <i6>
    %19 = buffer %18, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer5"} : <i6>
    %20:3 = fork [3] %19 {handshake.bb = 2 : ui32, handshake.name = "fork5"} : <i6>
    %21 = extsi %22 {handshake.bb = 2 : ui32, handshake.name = "extsi19"} : <i6> to <i10>
    %22 = buffer %20#0, bufferType = FIFO_BREAK_NONE, numSlots = 9 {handshake.bb = 2 : ui32, handshake.name = "buffer11"} : <i6>
    %23 = extsi %20#2 {handshake.bb = 2 : ui32, handshake.name = "extsi20"} : <i6> to <i7>
    %24 = trunci %25 {handshake.bb = 2 : ui32, handshake.name = "trunci1"} : <i6> to <i5>
    %25 = buffer %20#1, bufferType = FIFO_BREAK_NONE, numSlots = 9 {handshake.bb = 2 : ui32, handshake.name = "buffer13"} : <i6>
    %26 = mux %27 [%16#1, %trueResult_14] {handshake.bb = 2 : ui32, handshake.name = "mux2"} : <i1>, [<f32>, <f32>] to <f32>
    %27 = buffer %33#2, bufferType = FIFO_BREAK_NONE, numSlots = 15 {handshake.bb = 2 : ui32, handshake.name = "buffer14"} : <i1>
    %28 = mux %33#0 [%7#1, %trueResult_16] {handshake.bb = 2 : ui32, handshake.name = "mux3"} : <i1>, [<i6>, <i6>] to <i6>
    %29 = buffer %28, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer7"} : <i6>
    %30:2 = fork [2] %29 {handshake.bb = 2 : ui32, handshake.name = "fork6"} : <i6>
    %31 = extsi %32 {handshake.bb = 2 : ui32, handshake.name = "extsi21"} : <i6> to <i12>
    %32 = buffer %30#1, bufferType = FIFO_BREAK_NONE, numSlots = 5 {handshake.bb = 2 : ui32, handshake.name = "buffer16"} : <i6>
    %result_8, %index_9 = control_merge [%9#1, %trueResult_18]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>, <>] to <>, <i1>
    %33:3 = fork [3] %index_9 {handshake.bb = 2 : ui32, handshake.name = "fork7"} : <i1>
    %34 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %35 = constant %34 {handshake.bb = 2 : ui32, handshake.name = "constant16", value = 30 : i6} : <>, <i6>
    %36:2 = fork [2] %35 {handshake.bb = 2 : ui32, handshake.name = "fork8"} : <i6>
    %37 = extsi %38 {handshake.bb = 2 : ui32, handshake.name = "extsi22"} : <i6> to <i12>
    %38 = buffer %36#0, bufferType = FIFO_BREAK_NONE, numSlots = 5 {handshake.bb = 2 : ui32, handshake.name = "buffer17"} : <i6>
    %39 = extsi %36#1 {handshake.bb = 2 : ui32, handshake.name = "extsi23"} : <i6> to <i7>
    %40 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %41 = constant %40 {handshake.bb = 2 : ui32, handshake.name = "constant17", value = 1 : i2} : <>, <i2>
    %42 = extsi %41 {handshake.bb = 2 : ui32, handshake.name = "extsi24"} : <i2> to <i7>
    %43 = muli %31, %37 {handshake.bb = 2 : ui32, handshake.name = "muli0"} : <i12>
    %44 = trunci %43 {handshake.bb = 2 : ui32, handshake.name = "trunci2"} : <i12> to <i10>
    %45 = addi %21, %44 {handshake.bb = 2 : ui32, handshake.name = "addi0"} : <i10>
    %addressResult_10, %dataResult_11 = load[%45] %outputs_6#0 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i10>, <f32>, <i10>, <f32>
    %addressResult_12, %dataResult_13 = load[%24] %outputs_0 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load2"} : <i5>, <f32>, <i5>, <f32>
    %46 = mulf %dataResult_11, %dataResult_13 {fastmath = #arith.fastmath<none>, fpu_impl = #handshake<fpu_impl flopoco>, handshake.bb = 2 : ui32, handshake.name = "mulf0", internal_delay = "2_875333"} : <f32>
    %47 = buffer %26, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer6"} : <f32>
    %48 = addf %47, %46 {fastmath = #arith.fastmath<none>, fpu_impl = #handshake<fpu_impl flopoco>, handshake.bb = 2 : ui32, handshake.name = "addf0", internal_delay = "2_922000"} : <f32>
    %49 = addi %23, %42 {handshake.bb = 2 : ui32, handshake.name = "addi2"} : <i7>
    %50 = buffer %49, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer10"} : <i7>
    %51:2 = fork [2] %50 {handshake.bb = 2 : ui32, handshake.name = "fork9"} : <i7>
    %52 = trunci %51#0 {handshake.bb = 2 : ui32, handshake.name = "trunci3"} : <i7> to <i6>
    %53 = cmpi ult, %51#1, %39 {handshake.bb = 2 : ui32, handshake.name = "cmpi0"} : <i7>
    %54 = buffer %53, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer12"} : <i1>
    %55:4 = fork [4] %54 {handshake.bb = 2 : ui32, handshake.name = "fork10"} : <i1>
    %trueResult, %falseResult = cond_br %55#0, %52 {handshake.bb = 2 : ui32, handshake.name = "cond_br4"} : <i1>, <i6>
    sink %falseResult {handshake.name = "sink1"} : <i6>
    %trueResult_14, %falseResult_15 = cond_br %56, %48 {handshake.bb = 2 : ui32, handshake.name = "cond_br5"} : <i1>, <f32>
    %56 = buffer %55#2, bufferType = FIFO_BREAK_NONE, numSlots = 15 {handshake.bb = 2 : ui32, handshake.name = "buffer22"} : <i1>
    %57 = buffer %30#0, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer8"} : <i6>
    %trueResult_16, %falseResult_17 = cond_br %55#1, %57 {handshake.bb = 2 : ui32, handshake.name = "cond_br6"} : <i1>, <i6>
    %58 = buffer %result_8, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer9"} : <>
    %trueResult_18, %falseResult_19 = cond_br %55#3, %58 {handshake.bb = 2 : ui32, handshake.name = "cond_br7"} : <i1>, <>
    %59:2 = fork [2] %falseResult_17 {handshake.bb = 3 : ui32, handshake.name = "fork11"} : <i6>
    %60 = extsi %59#0 {handshake.bb = 3 : ui32, handshake.name = "extsi25"} : <i6> to <i7>
    %61 = extsi %59#1 {handshake.bb = 3 : ui32, handshake.name = "extsi26"} : <i6> to <i32>
    %62:2 = fork [2] %61 {handshake.bb = 3 : ui32, handshake.name = "fork12"} : <i32>
    %63:3 = fork [3] %falseResult_19 {handshake.bb = 3 : ui32, handshake.name = "fork13"} : <>
    %64 = constant %63#1 {handshake.bb = 3 : ui32, handshake.name = "constant18", value = 1 : i2} : <>, <i2>
    %65 = extsi %64 {handshake.bb = 3 : ui32, handshake.name = "extsi4"} : <i2> to <i32>
    %66 = constant %63#0 {handshake.bb = 3 : ui32, handshake.name = "constant19", value = false} : <>, <i1>
    %67 = source {handshake.bb = 3 : ui32, handshake.name = "source2"} : <>
    %68 = constant %67 {handshake.bb = 3 : ui32, handshake.name = "constant20", value = 30 : i6} : <>, <i6>
    %69 = extsi %68 {handshake.bb = 3 : ui32, handshake.name = "extsi27"} : <i6> to <i7>
    %70 = source {handshake.bb = 3 : ui32, handshake.name = "source3"} : <>
    %71 = constant %70 {handshake.bb = 3 : ui32, handshake.name = "constant21", value = 1 : i2} : <>, <i2>
    %72 = extsi %71 {handshake.bb = 3 : ui32, handshake.name = "extsi28"} : <i2> to <i7>
    %73 = gate %62#0, %15 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 3 : ui32, handshake.name = "gate0"} : <i32>, !handshake.control<> to <i32>
    %74 = cmpi ne, %73, %12 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 3 : ui32, handshake.name = "cmpi4"} : <i32>
    %75 = buffer %74, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer15"} : <i1>
    %76:2 = fork [2] %75 {handshake.bb = 3 : ui32, handshake.name = "fork14"} : <i1>
    %trueResult_20, %falseResult_21 = cond_br %77, %14#0 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 3 : ui32, handshake.name = "cond_br3"} : <i1>, <>
    %77 = buffer %76#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer29"} : <i1>
    sink %trueResult_20 {handshake.name = "sink3"} : <>
    %78 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 3 : ui32, handshake.name = "source8"} : <>
    %79 = mux %80 [%falseResult_21, %78] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 3 : ui32, handshake.name = "mux11"} : <i1>, [<>, <>] to <>
    %80 = buffer %76#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer30"} : <i1>
    %81 = join %79 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 3 : ui32, handshake.name = "join0"} : <>
    %82 = gate %83, %81 {handshake.bb = 3 : ui32, handshake.name = "gate1"} : <i32>, !handshake.control<> to <i32>
    %83 = buffer %62#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer31"} : <i32>
    %84 = trunci %82 {handshake.bb = 3 : ui32, handshake.name = "trunci4"} : <i32> to <i5>
    %addressResult_22, %dataResult_23, %doneResult = store[%84] %falseResult_15 %outputs_4#1 {handshake.bb = 3 : ui32, handshake.bufProps = #handshake<bufProps{"2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store0"} : <i5>, <f32>, <>, <i5>, <f32>, <>
    sink %doneResult {handshake.name = "sink4"} : <>
    %85 = addi %60, %72 {handshake.bb = 3 : ui32, handshake.name = "addi3"} : <i7>
    %86 = buffer %85, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer18"} : <i7>
    %87:2 = fork [2] %86 {handshake.bb = 3 : ui32, handshake.name = "fork15"} : <i7>
    %88 = trunci %87#0 {handshake.bb = 3 : ui32, handshake.name = "trunci5"} : <i7> to <i6>
    %89 = cmpi ult, %87#1, %69 {handshake.bb = 3 : ui32, handshake.name = "cmpi1"} : <i7>
    %90 = buffer %89, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer19"} : <i1>
    %91:4 = fork [4] %90 {handshake.bb = 3 : ui32, handshake.name = "fork16"} : <i1>
    %trueResult_24, %falseResult_25 = cond_br %91#0, %88 {handshake.bb = 3 : ui32, handshake.name = "cond_br8"} : <i1>, <i6>
    sink %falseResult_25 {handshake.name = "sink5"} : <i6>
    %trueResult_26, %falseResult_27 = cond_br %91#2, %63#2 {handshake.bb = 3 : ui32, handshake.name = "cond_br9"} : <i1>, <>
    %trueResult_28, %falseResult_29 = cond_br %91#3, %66 {handshake.bb = 3 : ui32, handshake.name = "cond_br10"} : <i1>, <i1>
    sink %trueResult_28 {handshake.name = "sink6"} : <i1>
    %92 = extsi %falseResult_29 {handshake.bb = 3 : ui32, handshake.name = "extsi15"} : <i1> to <i6>
    %93 = init %176#1 {ftd.imerge, handshake.bb = 4 : ui32, handshake.name = "init4"} : <i1>
    sink %93 {handshake.name = "sink7"} : <i1>
    %94:2 = unbundle %106#0  {handshake.bb = 4 : ui32, handshake.name = "unbundle3"} : <f32> to _ 
    %95 = mux %index_31 [%92, %trueResult_53] {handshake.bb = 4 : ui32, handshake.name = "mux4"} : <i1>, [<i6>, <i6>] to <i6>
    %96 = buffer %95, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer20"} : <i6>
    %97:3 = fork [3] %96 {handshake.bb = 4 : ui32, handshake.name = "fork17"} : <i6>
    %98 = trunci %97#0 {handshake.bb = 4 : ui32, handshake.name = "trunci6"} : <i6> to <i5>
    %result_30, %index_31 = control_merge [%falseResult_27, %trueResult_55]  {handshake.bb = 4 : ui32, handshake.name = "control_merge3"} : [<>, <>] to <>, <i1>
    %99:2 = fork [2] %result_30 {handshake.bb = 4 : ui32, handshake.name = "fork18"} : <>
    %100 = constant %99#0 {handshake.bb = 4 : ui32, handshake.name = "constant22", value = false} : <>, <i1>
    %101 = buffer %97#2, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer2"} : <i6>
    %102 = extsi %101 {handshake.bb = 4 : ui32, handshake.name = "extsi29"} : <i6> to <i32>
    %103 = buffer %94#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 4 : ui32, handshake.name = "buffer3"} : <>
    %104:2 = fork [2] %103 {handshake.bb = 4 : ui32, handshake.name = "fork19"} : <>
    %105 = init %104#1 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 4 : ui32, handshake.name = "init7"} : <>
    %addressResult_32, %dataResult_33 = load[%98] %outputs_2#0 {handshake.bb = 4 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["store1", 2, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load3"} : <i5>, <f32>, <i5>, <f32>
    %106:2 = fork [2] %dataResult_33 {handshake.bb = 4 : ui32, handshake.name = "fork20"} : <f32>
    %107 = extsi %100 {handshake.bb = 4 : ui32, handshake.name = "extsi14"} : <i1> to <i6>
    %108 = mux %121#1 [%107, %trueResult_40] {handshake.bb = 5 : ui32, handshake.name = "mux5"} : <i1>, [<i6>, <i6>] to <i6>
    %109 = buffer %108, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 5 : ui32, handshake.name = "buffer21"} : <i6>
    %110:3 = fork [3] %109 {handshake.bb = 5 : ui32, handshake.name = "fork21"} : <i6>
    %111 = extsi %110#1 {handshake.bb = 5 : ui32, handshake.name = "extsi30"} : <i6> to <i7>
    %112 = extsi %110#2 {handshake.bb = 5 : ui32, handshake.name = "extsi31"} : <i6> to <i12>
    %113 = trunci %110#0 {handshake.bb = 5 : ui32, handshake.name = "trunci7"} : <i6> to <i5>
    %114 = mux %115 [%106#1, %trueResult_42] {handshake.bb = 5 : ui32, handshake.name = "mux6"} : <i1>, [<f32>, <f32>] to <f32>
    %115 = buffer %121#2, bufferType = FIFO_BREAK_NONE, numSlots = 3 {handshake.bb = 5 : ui32, handshake.name = "buffer47"} : <i1>
    %116 = mux %121#0 [%97#1, %trueResult_44] {handshake.bb = 5 : ui32, handshake.name = "mux7"} : <i1>, [<i6>, <i6>] to <i6>
    %117 = buffer %116, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 5 : ui32, handshake.name = "buffer24"} : <i6>
    %118 = buffer %117, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 5 : ui32, handshake.name = "buffer25"} : <i6>
    %119:2 = fork [2] %118 {handshake.bb = 5 : ui32, handshake.name = "fork22"} : <i6>
    %120 = extsi %119#0 {handshake.bb = 5 : ui32, handshake.name = "extsi32"} : <i6> to <i10>
    %result_34, %index_35 = control_merge [%99#1, %trueResult_46]  {handshake.bb = 5 : ui32, handshake.name = "control_merge4"} : [<>, <>] to <>, <i1>
    %121:3 = fork [3] %index_35 {handshake.bb = 5 : ui32, handshake.name = "fork23"} : <i1>
    %122 = source {handshake.bb = 5 : ui32, handshake.name = "source4"} : <>
    %123 = constant %122 {handshake.bb = 5 : ui32, handshake.name = "constant23", value = 30 : i6} : <>, <i6>
    %124:2 = fork [2] %123 {handshake.bb = 5 : ui32, handshake.name = "fork24"} : <i6>
    %125 = extsi %124#0 {handshake.bb = 5 : ui32, handshake.name = "extsi33"} : <i6> to <i7>
    %126 = extsi %124#1 {handshake.bb = 5 : ui32, handshake.name = "extsi34"} : <i6> to <i12>
    %127 = source {handshake.bb = 5 : ui32, handshake.name = "source5"} : <>
    %128 = constant %127 {handshake.bb = 5 : ui32, handshake.name = "constant24", value = 1 : i2} : <>, <i2>
    %129 = extsi %128 {handshake.bb = 5 : ui32, handshake.name = "extsi35"} : <i2> to <i7>
    %130 = muli %112, %126 {handshake.bb = 5 : ui32, handshake.name = "muli1"} : <i12>
    %131 = trunci %130 {handshake.bb = 5 : ui32, handshake.name = "trunci8"} : <i12> to <i10>
    %132 = addi %120, %131 {handshake.bb = 5 : ui32, handshake.name = "addi1"} : <i10>
    %addressResult_36, %dataResult_37 = load[%132] %outputs_6#1 {handshake.bb = 5 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load4"} : <i10>, <f32>, <i10>, <f32>
    %addressResult_38, %dataResult_39 = load[%113] %outputs {handshake.bb = 5 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load5"} : <i5>, <f32>, <i5>, <f32>
    %133 = mulf %dataResult_37, %dataResult_39 {fastmath = #arith.fastmath<none>, fpu_impl = #handshake<fpu_impl flopoco>, handshake.bb = 5 : ui32, handshake.name = "mulf1", internal_delay = "2_875333"} : <f32>
    %134 = buffer %114, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 5 : ui32, handshake.name = "buffer23"} : <f32>
    %135 = addf %134, %133 {fastmath = #arith.fastmath<none>, fpu_impl = #handshake<fpu_impl flopoco>, handshake.bb = 5 : ui32, handshake.name = "addf1", internal_delay = "2_922000"} : <f32>
    %136 = addi %111, %129 {handshake.bb = 5 : ui32, handshake.name = "addi4"} : <i7>
    %137 = buffer %136, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 5 : ui32, handshake.name = "buffer27"} : <i7>
    %138:2 = fork [2] %137 {handshake.bb = 5 : ui32, handshake.name = "fork25"} : <i7>
    %139 = trunci %138#0 {handshake.bb = 5 : ui32, handshake.name = "trunci9"} : <i7> to <i6>
    %140 = cmpi ult, %138#1, %125 {handshake.bb = 5 : ui32, handshake.name = "cmpi2"} : <i7>
    %141 = buffer %140, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 5 : ui32, handshake.name = "buffer28"} : <i1>
    %142:4 = fork [4] %141 {handshake.bb = 5 : ui32, handshake.name = "fork26"} : <i1>
    %trueResult_40, %falseResult_41 = cond_br %142#0, %139 {handshake.bb = 5 : ui32, handshake.name = "cond_br11"} : <i1>, <i6>
    sink %falseResult_41 {handshake.name = "sink8"} : <i6>
    %trueResult_42, %falseResult_43 = cond_br %143, %135 {handshake.bb = 5 : ui32, handshake.name = "cond_br12"} : <i1>, <f32>
    %143 = buffer %142#2, bufferType = FIFO_BREAK_NONE, numSlots = 4 {handshake.bb = 5 : ui32, handshake.name = "buffer55"} : <i1>
    %trueResult_44, %falseResult_45 = cond_br %142#1, %119#1 {handshake.bb = 5 : ui32, handshake.name = "cond_br13"} : <i1>, <i6>
    %144 = buffer %result_34, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 5 : ui32, handshake.name = "buffer26"} : <>
    %trueResult_46, %falseResult_47 = cond_br %142#3, %144 {handshake.bb = 5 : ui32, handshake.name = "cond_br14"} : <i1>, <>
    %145:2 = fork [2] %falseResult_45 {handshake.bb = 6 : ui32, handshake.name = "fork27"} : <i6>
    %146 = extsi %145#0 {handshake.bb = 6 : ui32, handshake.name = "extsi36"} : <i6> to <i7>
    %147 = extsi %145#1 {handshake.bb = 6 : ui32, handshake.name = "extsi37"} : <i6> to <i32>
    %148:2 = fork [2] %147 {handshake.bb = 6 : ui32, handshake.name = "fork28"} : <i32>
    %149:2 = fork [2] %falseResult_47 {handshake.bb = 6 : ui32, handshake.name = "fork29"} : <>
    %150 = constant %149#0 {handshake.bb = 6 : ui32, handshake.name = "constant25", value = 1 : i2} : <>, <i2>
    %151 = extsi %150 {handshake.bb = 6 : ui32, handshake.name = "extsi11"} : <i2> to <i32>
    %152 = source {handshake.bb = 6 : ui32, handshake.name = "source6"} : <>
    %153 = constant %152 {handshake.bb = 6 : ui32, handshake.name = "constant26", value = 30 : i6} : <>, <i6>
    %154 = extsi %153 {handshake.bb = 6 : ui32, handshake.name = "extsi38"} : <i6> to <i7>
    %155 = source {handshake.bb = 6 : ui32, handshake.name = "source7"} : <>
    %156 = constant %155 {handshake.bb = 6 : ui32, handshake.name = "constant27", value = 1 : i2} : <>, <i2>
    %157 = extsi %156 {handshake.bb = 6 : ui32, handshake.name = "extsi39"} : <i2> to <i7>
    %158 = gate %148#0, %105 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 6 : ui32, handshake.name = "gate2"} : <i32>, !handshake.control<> to <i32>
    %159 = cmpi ne, %158, %102 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 6 : ui32, handshake.name = "cmpi5"} : <i32>
    %160 = buffer %159, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 6 : ui32, handshake.name = "buffer32"} : <i1>
    %161:2 = fork [2] %160 {handshake.bb = 6 : ui32, handshake.name = "fork30"} : <i1>
    %trueResult_48, %falseResult_49 = cond_br %162, %104#0 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 6 : ui32, handshake.name = "cond_br20"} : <i1>, <>
    %162 = buffer %161#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 6 : ui32, handshake.name = "buffer62"} : <i1>
    sink %trueResult_48 {handshake.name = "sink10"} : <>
    %163 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 6 : ui32, handshake.name = "source9"} : <>
    %164 = mux %165 [%falseResult_49, %163] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 6 : ui32, handshake.name = "mux15"} : <i1>, [<>, <>] to <>
    %165 = buffer %161#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 6 : ui32, handshake.name = "buffer63"} : <i1>
    %166 = join %164 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 6 : ui32, handshake.name = "join1"} : <>
    %167 = gate %168, %166 {handshake.bb = 6 : ui32, handshake.name = "gate3"} : <i32>, !handshake.control<> to <i32>
    %168 = buffer %148#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 6 : ui32, handshake.name = "buffer64"} : <i32>
    %169 = trunci %167 {handshake.bb = 6 : ui32, handshake.name = "trunci10"} : <i32> to <i5>
    %addressResult_50, %dataResult_51, %doneResult_52 = store[%169] %falseResult_43 %outputs_2#1 {handshake.bb = 6 : ui32, handshake.bufProps = #handshake<bufProps{"2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store1"} : <i5>, <f32>, <>, <i5>, <f32>, <>
    sink %doneResult_52 {handshake.name = "sink11"} : <>
    %170 = addi %146, %157 {handshake.bb = 6 : ui32, handshake.name = "addi5"} : <i7>
    %171 = buffer %170, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 6 : ui32, handshake.name = "buffer33"} : <i7>
    %172:2 = fork [2] %171 {handshake.bb = 6 : ui32, handshake.name = "fork31"} : <i7>
    %173 = trunci %172#0 {handshake.bb = 6 : ui32, handshake.name = "trunci11"} : <i7> to <i6>
    %174 = cmpi ult, %172#1, %154 {handshake.bb = 6 : ui32, handshake.name = "cmpi3"} : <i7>
    %175 = buffer %174, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 6 : ui32, handshake.name = "buffer34"} : <i1>
    %176:3 = fork [3] %175 {handshake.bb = 6 : ui32, handshake.name = "fork32"} : <i1>
    %trueResult_53, %falseResult_54 = cond_br %176#0, %173 {handshake.bb = 6 : ui32, handshake.name = "cond_br15"} : <i1>, <i6>
    sink %falseResult_54 {handshake.name = "sink12"} : <i6>
    %trueResult_55, %falseResult_56 = cond_br %176#2, %149#1 {handshake.bb = 6 : ui32, handshake.name = "cond_br16"} : <i1>, <>
    %177:5 = fork [5] %falseResult_56 {handshake.bb = 7 : ui32, handshake.name = "fork33"} : <>
    end {handshake.bb = 7 : ui32, handshake.name = "end0"} %memEnd_7, %memEnd_5, %memEnd_3, %memEnd_1, %memEnd, %0#1 : <>, <>, <>, <>, <>, <>
  }
}

