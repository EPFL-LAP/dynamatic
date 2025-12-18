module {
  handshake.func @mvt_float(%arg0: memref<900xf32>, %arg1: memref<30xf32>, %arg2: memref<30xf32>, %arg3: memref<30xf32>, %arg4: memref<30xf32>, %arg5: !handshake.control<>, %arg6: !handshake.control<>, %arg7: !handshake.control<>, %arg8: !handshake.control<>, %arg9: !handshake.control<>, %arg10: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["A", "x1", "x2", "y1", "y2", "A_start", "x1_start", "x2_start", "y1_start", "y2_start", "start"], cfg.edges = "[0,1][2,2,3,cmpi0][4,5][6,4,7,cmpi3][1,2][3,1,4,cmpi1][5,5,6,cmpi2]", resNames = ["A_end", "x1_end", "x2_end", "y1_end", "y2_end", "end"]} {
    %0:3 = fork [3] %arg10 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork0"} : <>
    %outputs, %memEnd = mem_controller[%arg4 : memref<30xf32>] %arg9 (%addressResult_40) %191#4 {connectedBlocks = [5 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller3"} :    (!handshake.channel<i5>) -> !handshake.channel<f32>
    %outputs_0, %memEnd_1 = mem_controller[%arg3 : memref<30xf32>] %arg8 (%addressResult_12) %191#3 {connectedBlocks = [2 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller4"} :    (!handshake.channel<i5>) -> !handshake.channel<f32>
    %outputs_2:2, %memEnd_3 = mem_controller[%arg2 : memref<30xf32>] %arg7 (%addressResult_34, %165, %addressResult_54, %dataResult_55) %191#2 {connectedBlocks = [4 : i32, 6 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "5": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller5"} :    (!handshake.channel<i5>, !handshake.channel<i32>, !handshake.channel<i5>, !handshake.channel<f32>) -> (!handshake.channel<f32>, !handshake.control<>)
    %outputs_4:2, %memEnd_5 = mem_controller[%arg1 : memref<30xf32>] %arg6 (%addressResult, %73, %addressResult_24, %dataResult_25) %191#1 {connectedBlocks = [1 : i32, 3 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "5": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller6"} :    (!handshake.channel<i5>, !handshake.channel<i32>, !handshake.channel<i5>, !handshake.channel<f32>) -> (!handshake.channel<f32>, !handshake.control<>)
    %outputs_6:2, %memEnd_7 = mem_controller[%arg0 : memref<900xf32>] %arg5 (%addressResult_10, %addressResult_38) %191#0 {connectedBlocks = [2 : i32, 5 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller7"} :    (!handshake.channel<i10>, !handshake.channel<i10>) -> (!handshake.channel<f32>, !handshake.channel<f32>)
    %1 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant0", value = false} : <>, <i1>
    %2 = br %1 {handshake.bb = 0 : ui32, handshake.name = "br2"} : <i1>
    %3 = extsi %2 {handshake.bb = 0 : ui32, handshake.name = "extsi17"} : <i1> to <i6>
    %4 = br %0#2 {handshake.bb = 0 : ui32, handshake.name = "br4"} : <>
    %5 = init %99#1 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "init0"} : <i1>
    sink %5 {handshake.name = "sink0"} : <i1>
    %6:2 = unbundle %18#0  {handshake.bb = 1 : ui32, handshake.name = "unbundle1"} : <f32> to _ 
    %7 = mux %index [%3, %trueResult_26] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i6>, <i6>] to <i6>
    %8 = buffer %7, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer4"} : <i6>
    %9:3 = fork [3] %8 {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <i6>
    %10 = trunci %9#0 {handshake.bb = 1 : ui32, handshake.name = "trunci0"} : <i6> to <i5>
    %result, %index = control_merge [%4, %trueResult_28]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %11:2 = fork [2] %result {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <>
    %12 = constant %11#0 {handshake.bb = 1 : ui32, handshake.name = "constant1", value = false} : <>, <i1>
    %13 = buffer %9#2, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer0"} : <i6>
    %14 = extsi %13 {handshake.bb = 1 : ui32, handshake.name = "extsi18"} : <i6> to <i32>
    %15 = buffer %6#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 1 : ui32, handshake.name = "buffer1"} : <>
    %16:2 = fork [2] %15 {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <>
    %17 = init %16#1 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 1 : ui32, handshake.name = "init3"} : <>
    %addressResult, %dataResult = load[%10] %outputs_4#0 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["store0", 2, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i5>, <f32>, <i5>, <f32>
    %18:2 = fork [2] %dataResult {handshake.bb = 1 : ui32, handshake.name = "fork4"} : <f32>
    %19 = br %12 {handshake.bb = 1 : ui32, handshake.name = "br5"} : <i1>
    %20 = extsi %19 {handshake.bb = 1 : ui32, handshake.name = "extsi16"} : <i1> to <i6>
    %21 = br %18#1 {handshake.bb = 1 : ui32, handshake.name = "br6"} : <f32>
    %22 = br %9#1 {handshake.bb = 1 : ui32, handshake.name = "br7"} : <i6>
    %23 = br %11#1 {handshake.bb = 1 : ui32, handshake.name = "br8"} : <>
    %24 = mux %39#1 [%20, %trueResult] {handshake.bb = 2 : ui32, handshake.name = "mux1"} : <i1>, [<i6>, <i6>] to <i6>
    %25 = buffer %24, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer5"} : <i6>
    %26:3 = fork [3] %25 {handshake.bb = 2 : ui32, handshake.name = "fork5"} : <i6>
    %27 = extsi %28 {handshake.bb = 2 : ui32, handshake.name = "extsi19"} : <i6> to <i10>
    %28 = buffer %26#0, bufferType = FIFO_BREAK_NONE, numSlots = 9 {handshake.bb = 2 : ui32, handshake.name = "buffer11"} : <i6>
    %29 = extsi %26#2 {handshake.bb = 2 : ui32, handshake.name = "extsi20"} : <i6> to <i7>
    %30 = trunci %31 {handshake.bb = 2 : ui32, handshake.name = "trunci1"} : <i6> to <i5>
    %31 = buffer %26#1, bufferType = FIFO_BREAK_NONE, numSlots = 9 {handshake.bb = 2 : ui32, handshake.name = "buffer13"} : <i6>
    %32 = mux %33 [%21, %trueResult_14] {handshake.bb = 2 : ui32, handshake.name = "mux2"} : <i1>, [<f32>, <f32>] to <f32>
    %33 = buffer %39#2, bufferType = FIFO_BREAK_NONE, numSlots = 15 {handshake.bb = 2 : ui32, handshake.name = "buffer14"} : <i1>
    %34 = mux %39#0 [%22, %trueResult_16] {handshake.bb = 2 : ui32, handshake.name = "mux3"} : <i1>, [<i6>, <i6>] to <i6>
    %35 = buffer %34, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer7"} : <i6>
    %36:2 = fork [2] %35 {handshake.bb = 2 : ui32, handshake.name = "fork6"} : <i6>
    %37 = extsi %38 {handshake.bb = 2 : ui32, handshake.name = "extsi21"} : <i6> to <i12>
    %38 = buffer %36#1, bufferType = FIFO_BREAK_NONE, numSlots = 5 {handshake.bb = 2 : ui32, handshake.name = "buffer16"} : <i6>
    %result_8, %index_9 = control_merge [%23, %trueResult_18]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>, <>] to <>, <i1>
    %39:3 = fork [3] %index_9 {handshake.bb = 2 : ui32, handshake.name = "fork7"} : <i1>
    %40 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %41 = constant %40 {handshake.bb = 2 : ui32, handshake.name = "constant16", value = 30 : i6} : <>, <i6>
    %42:2 = fork [2] %41 {handshake.bb = 2 : ui32, handshake.name = "fork8"} : <i6>
    %43 = extsi %44 {handshake.bb = 2 : ui32, handshake.name = "extsi22"} : <i6> to <i12>
    %44 = buffer %42#0, bufferType = FIFO_BREAK_NONE, numSlots = 5 {handshake.bb = 2 : ui32, handshake.name = "buffer17"} : <i6>
    %45 = extsi %42#1 {handshake.bb = 2 : ui32, handshake.name = "extsi23"} : <i6> to <i7>
    %46 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %47 = constant %46 {handshake.bb = 2 : ui32, handshake.name = "constant17", value = 1 : i2} : <>, <i2>
    %48 = extsi %47 {handshake.bb = 2 : ui32, handshake.name = "extsi24"} : <i2> to <i7>
    %49 = muli %37, %43 {handshake.bb = 2 : ui32, handshake.name = "muli0"} : <i12>
    %50 = trunci %49 {handshake.bb = 2 : ui32, handshake.name = "trunci2"} : <i12> to <i10>
    %51 = addi %27, %50 {handshake.bb = 2 : ui32, handshake.name = "addi0"} : <i10>
    %addressResult_10, %dataResult_11 = load[%51] %outputs_6#0 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i10>, <f32>, <i10>, <f32>
    %addressResult_12, %dataResult_13 = load[%30] %outputs_0 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load2"} : <i5>, <f32>, <i5>, <f32>
    %52 = mulf %dataResult_11, %dataResult_13 {fastmath = #arith.fastmath<none>, fpu_impl = #handshake<fpu_impl flopoco>, handshake.bb = 2 : ui32, handshake.name = "mulf0", internal_delay = "2_875333"} : <f32>
    %53 = buffer %32, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer6"} : <f32>
    %54 = addf %53, %52 {fastmath = #arith.fastmath<none>, fpu_impl = #handshake<fpu_impl flopoco>, handshake.bb = 2 : ui32, handshake.name = "addf0", internal_delay = "2_922000"} : <f32>
    %55 = addi %29, %48 {handshake.bb = 2 : ui32, handshake.name = "addi2"} : <i7>
    %56 = buffer %55, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer10"} : <i7>
    %57:2 = fork [2] %56 {handshake.bb = 2 : ui32, handshake.name = "fork9"} : <i7>
    %58 = trunci %57#0 {handshake.bb = 2 : ui32, handshake.name = "trunci3"} : <i7> to <i6>
    %59 = cmpi ult, %57#1, %45 {handshake.bb = 2 : ui32, handshake.name = "cmpi0"} : <i7>
    %60 = buffer %59, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer12"} : <i1>
    %61:4 = fork [4] %60 {handshake.bb = 2 : ui32, handshake.name = "fork10"} : <i1>
    %trueResult, %falseResult = cond_br %61#0, %58 {handshake.bb = 2 : ui32, handshake.name = "cond_br4"} : <i1>, <i6>
    sink %falseResult {handshake.name = "sink1"} : <i6>
    %trueResult_14, %falseResult_15 = cond_br %62, %54 {handshake.bb = 2 : ui32, handshake.name = "cond_br5"} : <i1>, <f32>
    %62 = buffer %61#2, bufferType = FIFO_BREAK_NONE, numSlots = 15 {handshake.bb = 2 : ui32, handshake.name = "buffer22"} : <i1>
    %63 = buffer %36#0, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer8"} : <i6>
    %trueResult_16, %falseResult_17 = cond_br %61#1, %63 {handshake.bb = 2 : ui32, handshake.name = "cond_br6"} : <i1>, <i6>
    %64 = buffer %result_8, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer9"} : <>
    %trueResult_18, %falseResult_19 = cond_br %61#3, %64 {handshake.bb = 2 : ui32, handshake.name = "cond_br7"} : <i1>, <>
    %65 = merge %falseResult_17 {handshake.bb = 3 : ui32, handshake.name = "merge0"} : <i6>
    %66:2 = fork [2] %65 {handshake.bb = 3 : ui32, handshake.name = "fork11"} : <i6>
    %67 = extsi %66#0 {handshake.bb = 3 : ui32, handshake.name = "extsi25"} : <i6> to <i7>
    %68 = extsi %66#1 {handshake.bb = 3 : ui32, handshake.name = "extsi26"} : <i6> to <i32>
    %69:2 = fork [2] %68 {handshake.bb = 3 : ui32, handshake.name = "fork12"} : <i32>
    %70 = merge %falseResult_15 {handshake.bb = 3 : ui32, handshake.name = "merge1"} : <f32>
    %result_20, %index_21 = control_merge [%falseResult_19]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>] to <>, <i1>
    sink %index_21 {handshake.name = "sink2"} : <i1>
    %71:3 = fork [3] %result_20 {handshake.bb = 3 : ui32, handshake.name = "fork13"} : <>
    %72 = constant %71#1 {handshake.bb = 3 : ui32, handshake.name = "constant18", value = 1 : i2} : <>, <i2>
    %73 = extsi %72 {handshake.bb = 3 : ui32, handshake.name = "extsi4"} : <i2> to <i32>
    %74 = constant %71#0 {handshake.bb = 3 : ui32, handshake.name = "constant19", value = false} : <>, <i1>
    %75 = source {handshake.bb = 3 : ui32, handshake.name = "source2"} : <>
    %76 = constant %75 {handshake.bb = 3 : ui32, handshake.name = "constant20", value = 30 : i6} : <>, <i6>
    %77 = extsi %76 {handshake.bb = 3 : ui32, handshake.name = "extsi27"} : <i6> to <i7>
    %78 = source {handshake.bb = 3 : ui32, handshake.name = "source3"} : <>
    %79 = constant %78 {handshake.bb = 3 : ui32, handshake.name = "constant21", value = 1 : i2} : <>, <i2>
    %80 = extsi %79 {handshake.bb = 3 : ui32, handshake.name = "extsi28"} : <i2> to <i7>
    %81 = gate %69#0, %17 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 3 : ui32, handshake.name = "gate0"} : <i32>, !handshake.control<> to <i32>
    %82 = cmpi ne, %81, %14 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 3 : ui32, handshake.name = "cmpi4"} : <i32>
    %83 = buffer %82, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer15"} : <i1>
    %84:2 = fork [2] %83 {handshake.bb = 3 : ui32, handshake.name = "fork14"} : <i1>
    %trueResult_22, %falseResult_23 = cond_br %85, %16#0 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 3 : ui32, handshake.name = "cond_br3"} : <i1>, <>
    %85 = buffer %84#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer29"} : <i1>
    sink %trueResult_22 {handshake.name = "sink3"} : <>
    %86 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 3 : ui32, handshake.name = "source8"} : <>
    %87 = mux %88 [%falseResult_23, %86] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 3 : ui32, handshake.name = "mux11"} : <i1>, [<>, <>] to <>
    %88 = buffer %84#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer30"} : <i1>
    %89 = join %87 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 3 : ui32, handshake.name = "join0"} : <>
    %90 = gate %91, %89 {handshake.bb = 3 : ui32, handshake.name = "gate1"} : <i32>, !handshake.control<> to <i32>
    %91 = buffer %69#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer31"} : <i32>
    %92 = trunci %90 {handshake.bb = 3 : ui32, handshake.name = "trunci4"} : <i32> to <i5>
    %addressResult_24, %dataResult_25, %doneResult = store[%92] %70 %outputs_4#1 {handshake.bb = 3 : ui32, handshake.bufProps = #handshake<bufProps{"2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store0"} : <i5>, <f32>, <>, <i5>, <f32>, <>
    sink %doneResult {handshake.name = "sink4"} : <>
    %93 = addi %67, %80 {handshake.bb = 3 : ui32, handshake.name = "addi3"} : <i7>
    %94 = buffer %93, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer18"} : <i7>
    %95:2 = fork [2] %94 {handshake.bb = 3 : ui32, handshake.name = "fork15"} : <i7>
    %96 = trunci %95#0 {handshake.bb = 3 : ui32, handshake.name = "trunci5"} : <i7> to <i6>
    %97 = cmpi ult, %95#1, %77 {handshake.bb = 3 : ui32, handshake.name = "cmpi1"} : <i7>
    %98 = buffer %97, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer19"} : <i1>
    %99:4 = fork [4] %98 {handshake.bb = 3 : ui32, handshake.name = "fork16"} : <i1>
    %trueResult_26, %falseResult_27 = cond_br %99#0, %96 {handshake.bb = 3 : ui32, handshake.name = "cond_br8"} : <i1>, <i6>
    sink %falseResult_27 {handshake.name = "sink5"} : <i6>
    %trueResult_28, %falseResult_29 = cond_br %99#2, %71#2 {handshake.bb = 3 : ui32, handshake.name = "cond_br9"} : <i1>, <>
    %trueResult_30, %falseResult_31 = cond_br %99#3, %74 {handshake.bb = 3 : ui32, handshake.name = "cond_br10"} : <i1>, <i1>
    sink %trueResult_30 {handshake.name = "sink6"} : <i1>
    %100 = extsi %falseResult_31 {handshake.bb = 3 : ui32, handshake.name = "extsi15"} : <i1> to <i6>
    %101 = init %190#1 {ftd.imerge, handshake.bb = 4 : ui32, handshake.name = "init4"} : <i1>
    sink %101 {handshake.name = "sink7"} : <i1>
    %102:2 = unbundle %114#0  {handshake.bb = 4 : ui32, handshake.name = "unbundle3"} : <f32> to _ 
    %103 = mux %index_33 [%100, %trueResult_57] {handshake.bb = 4 : ui32, handshake.name = "mux4"} : <i1>, [<i6>, <i6>] to <i6>
    %104 = buffer %103, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer20"} : <i6>
    %105:3 = fork [3] %104 {handshake.bb = 4 : ui32, handshake.name = "fork17"} : <i6>
    %106 = trunci %105#0 {handshake.bb = 4 : ui32, handshake.name = "trunci6"} : <i6> to <i5>
    %result_32, %index_33 = control_merge [%falseResult_29, %trueResult_59]  {handshake.bb = 4 : ui32, handshake.name = "control_merge3"} : [<>, <>] to <>, <i1>
    %107:2 = fork [2] %result_32 {handshake.bb = 4 : ui32, handshake.name = "fork18"} : <>
    %108 = constant %107#0 {handshake.bb = 4 : ui32, handshake.name = "constant22", value = false} : <>, <i1>
    %109 = buffer %105#2, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer2"} : <i6>
    %110 = extsi %109 {handshake.bb = 4 : ui32, handshake.name = "extsi29"} : <i6> to <i32>
    %111 = buffer %102#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 4 : ui32, handshake.name = "buffer3"} : <>
    %112:2 = fork [2] %111 {handshake.bb = 4 : ui32, handshake.name = "fork19"} : <>
    %113 = init %112#1 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 4 : ui32, handshake.name = "init7"} : <>
    %addressResult_34, %dataResult_35 = load[%106] %outputs_2#0 {handshake.bb = 4 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["store1", 2, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load3"} : <i5>, <f32>, <i5>, <f32>
    %114:2 = fork [2] %dataResult_35 {handshake.bb = 4 : ui32, handshake.name = "fork20"} : <f32>
    %115 = br %108 {handshake.bb = 4 : ui32, handshake.name = "br9"} : <i1>
    %116 = extsi %115 {handshake.bb = 4 : ui32, handshake.name = "extsi14"} : <i1> to <i6>
    %117 = br %114#1 {handshake.bb = 4 : ui32, handshake.name = "br10"} : <f32>
    %118 = br %105#1 {handshake.bb = 4 : ui32, handshake.name = "br11"} : <i6>
    %119 = br %107#1 {handshake.bb = 4 : ui32, handshake.name = "br12"} : <>
    %120 = mux %133#1 [%116, %trueResult_42] {handshake.bb = 5 : ui32, handshake.name = "mux5"} : <i1>, [<i6>, <i6>] to <i6>
    %121 = buffer %120, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 5 : ui32, handshake.name = "buffer21"} : <i6>
    %122:3 = fork [3] %121 {handshake.bb = 5 : ui32, handshake.name = "fork21"} : <i6>
    %123 = extsi %122#1 {handshake.bb = 5 : ui32, handshake.name = "extsi30"} : <i6> to <i7>
    %124 = extsi %122#2 {handshake.bb = 5 : ui32, handshake.name = "extsi31"} : <i6> to <i12>
    %125 = trunci %122#0 {handshake.bb = 5 : ui32, handshake.name = "trunci7"} : <i6> to <i5>
    %126 = mux %127 [%117, %trueResult_44] {handshake.bb = 5 : ui32, handshake.name = "mux6"} : <i1>, [<f32>, <f32>] to <f32>
    %127 = buffer %133#2, bufferType = FIFO_BREAK_NONE, numSlots = 3 {handshake.bb = 5 : ui32, handshake.name = "buffer47"} : <i1>
    %128 = mux %133#0 [%118, %trueResult_46] {handshake.bb = 5 : ui32, handshake.name = "mux7"} : <i1>, [<i6>, <i6>] to <i6>
    %129 = buffer %128, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 5 : ui32, handshake.name = "buffer24"} : <i6>
    %130 = buffer %129, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 5 : ui32, handshake.name = "buffer25"} : <i6>
    %131:2 = fork [2] %130 {handshake.bb = 5 : ui32, handshake.name = "fork22"} : <i6>
    %132 = extsi %131#0 {handshake.bb = 5 : ui32, handshake.name = "extsi32"} : <i6> to <i10>
    %result_36, %index_37 = control_merge [%119, %trueResult_48]  {handshake.bb = 5 : ui32, handshake.name = "control_merge4"} : [<>, <>] to <>, <i1>
    %133:3 = fork [3] %index_37 {handshake.bb = 5 : ui32, handshake.name = "fork23"} : <i1>
    %134 = source {handshake.bb = 5 : ui32, handshake.name = "source4"} : <>
    %135 = constant %134 {handshake.bb = 5 : ui32, handshake.name = "constant23", value = 30 : i6} : <>, <i6>
    %136:2 = fork [2] %135 {handshake.bb = 5 : ui32, handshake.name = "fork24"} : <i6>
    %137 = extsi %136#0 {handshake.bb = 5 : ui32, handshake.name = "extsi33"} : <i6> to <i7>
    %138 = extsi %136#1 {handshake.bb = 5 : ui32, handshake.name = "extsi34"} : <i6> to <i12>
    %139 = source {handshake.bb = 5 : ui32, handshake.name = "source5"} : <>
    %140 = constant %139 {handshake.bb = 5 : ui32, handshake.name = "constant24", value = 1 : i2} : <>, <i2>
    %141 = extsi %140 {handshake.bb = 5 : ui32, handshake.name = "extsi35"} : <i2> to <i7>
    %142 = muli %124, %138 {handshake.bb = 5 : ui32, handshake.name = "muli1"} : <i12>
    %143 = trunci %142 {handshake.bb = 5 : ui32, handshake.name = "trunci8"} : <i12> to <i10>
    %144 = addi %132, %143 {handshake.bb = 5 : ui32, handshake.name = "addi1"} : <i10>
    %addressResult_38, %dataResult_39 = load[%144] %outputs_6#1 {handshake.bb = 5 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load4"} : <i10>, <f32>, <i10>, <f32>
    %addressResult_40, %dataResult_41 = load[%125] %outputs {handshake.bb = 5 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load5"} : <i5>, <f32>, <i5>, <f32>
    %145 = mulf %dataResult_39, %dataResult_41 {fastmath = #arith.fastmath<none>, fpu_impl = #handshake<fpu_impl flopoco>, handshake.bb = 5 : ui32, handshake.name = "mulf1", internal_delay = "2_875333"} : <f32>
    %146 = buffer %126, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 5 : ui32, handshake.name = "buffer23"} : <f32>
    %147 = addf %146, %145 {fastmath = #arith.fastmath<none>, fpu_impl = #handshake<fpu_impl flopoco>, handshake.bb = 5 : ui32, handshake.name = "addf1", internal_delay = "2_922000"} : <f32>
    %148 = addi %123, %141 {handshake.bb = 5 : ui32, handshake.name = "addi4"} : <i7>
    %149 = buffer %148, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 5 : ui32, handshake.name = "buffer27"} : <i7>
    %150:2 = fork [2] %149 {handshake.bb = 5 : ui32, handshake.name = "fork25"} : <i7>
    %151 = trunci %150#0 {handshake.bb = 5 : ui32, handshake.name = "trunci9"} : <i7> to <i6>
    %152 = cmpi ult, %150#1, %137 {handshake.bb = 5 : ui32, handshake.name = "cmpi2"} : <i7>
    %153 = buffer %152, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 5 : ui32, handshake.name = "buffer28"} : <i1>
    %154:4 = fork [4] %153 {handshake.bb = 5 : ui32, handshake.name = "fork26"} : <i1>
    %trueResult_42, %falseResult_43 = cond_br %154#0, %151 {handshake.bb = 5 : ui32, handshake.name = "cond_br11"} : <i1>, <i6>
    sink %falseResult_43 {handshake.name = "sink8"} : <i6>
    %trueResult_44, %falseResult_45 = cond_br %155, %147 {handshake.bb = 5 : ui32, handshake.name = "cond_br12"} : <i1>, <f32>
    %155 = buffer %154#2, bufferType = FIFO_BREAK_NONE, numSlots = 4 {handshake.bb = 5 : ui32, handshake.name = "buffer55"} : <i1>
    %trueResult_46, %falseResult_47 = cond_br %154#1, %131#1 {handshake.bb = 5 : ui32, handshake.name = "cond_br13"} : <i1>, <i6>
    %156 = buffer %result_36, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 5 : ui32, handshake.name = "buffer26"} : <>
    %trueResult_48, %falseResult_49 = cond_br %154#3, %156 {handshake.bb = 5 : ui32, handshake.name = "cond_br14"} : <i1>, <>
    %157 = merge %falseResult_47 {handshake.bb = 6 : ui32, handshake.name = "merge2"} : <i6>
    %158:2 = fork [2] %157 {handshake.bb = 6 : ui32, handshake.name = "fork27"} : <i6>
    %159 = extsi %158#0 {handshake.bb = 6 : ui32, handshake.name = "extsi36"} : <i6> to <i7>
    %160 = extsi %158#1 {handshake.bb = 6 : ui32, handshake.name = "extsi37"} : <i6> to <i32>
    %161:2 = fork [2] %160 {handshake.bb = 6 : ui32, handshake.name = "fork28"} : <i32>
    %162 = merge %falseResult_45 {handshake.bb = 6 : ui32, handshake.name = "merge3"} : <f32>
    %result_50, %index_51 = control_merge [%falseResult_49]  {handshake.bb = 6 : ui32, handshake.name = "control_merge5"} : [<>] to <>, <i1>
    sink %index_51 {handshake.name = "sink9"} : <i1>
    %163:2 = fork [2] %result_50 {handshake.bb = 6 : ui32, handshake.name = "fork29"} : <>
    %164 = constant %163#0 {handshake.bb = 6 : ui32, handshake.name = "constant25", value = 1 : i2} : <>, <i2>
    %165 = extsi %164 {handshake.bb = 6 : ui32, handshake.name = "extsi11"} : <i2> to <i32>
    %166 = source {handshake.bb = 6 : ui32, handshake.name = "source6"} : <>
    %167 = constant %166 {handshake.bb = 6 : ui32, handshake.name = "constant26", value = 30 : i6} : <>, <i6>
    %168 = extsi %167 {handshake.bb = 6 : ui32, handshake.name = "extsi38"} : <i6> to <i7>
    %169 = source {handshake.bb = 6 : ui32, handshake.name = "source7"} : <>
    %170 = constant %169 {handshake.bb = 6 : ui32, handshake.name = "constant27", value = 1 : i2} : <>, <i2>
    %171 = extsi %170 {handshake.bb = 6 : ui32, handshake.name = "extsi39"} : <i2> to <i7>
    %172 = gate %161#0, %113 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 6 : ui32, handshake.name = "gate2"} : <i32>, !handshake.control<> to <i32>
    %173 = cmpi ne, %172, %110 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 6 : ui32, handshake.name = "cmpi5"} : <i32>
    %174 = buffer %173, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 6 : ui32, handshake.name = "buffer32"} : <i1>
    %175:2 = fork [2] %174 {handshake.bb = 6 : ui32, handshake.name = "fork30"} : <i1>
    %trueResult_52, %falseResult_53 = cond_br %176, %112#0 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 6 : ui32, handshake.name = "cond_br20"} : <i1>, <>
    %176 = buffer %175#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 6 : ui32, handshake.name = "buffer62"} : <i1>
    sink %trueResult_52 {handshake.name = "sink10"} : <>
    %177 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 6 : ui32, handshake.name = "source9"} : <>
    %178 = mux %179 [%falseResult_53, %177] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 6 : ui32, handshake.name = "mux15"} : <i1>, [<>, <>] to <>
    %179 = buffer %175#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 6 : ui32, handshake.name = "buffer63"} : <i1>
    %180 = join %178 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 6 : ui32, handshake.name = "join1"} : <>
    %181 = gate %182, %180 {handshake.bb = 6 : ui32, handshake.name = "gate3"} : <i32>, !handshake.control<> to <i32>
    %182 = buffer %161#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 6 : ui32, handshake.name = "buffer64"} : <i32>
    %183 = trunci %181 {handshake.bb = 6 : ui32, handshake.name = "trunci10"} : <i32> to <i5>
    %addressResult_54, %dataResult_55, %doneResult_56 = store[%183] %162 %outputs_2#1 {handshake.bb = 6 : ui32, handshake.bufProps = #handshake<bufProps{"2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store1"} : <i5>, <f32>, <>, <i5>, <f32>, <>
    sink %doneResult_56 {handshake.name = "sink11"} : <>
    %184 = addi %159, %171 {handshake.bb = 6 : ui32, handshake.name = "addi5"} : <i7>
    %185 = buffer %184, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 6 : ui32, handshake.name = "buffer33"} : <i7>
    %186:2 = fork [2] %185 {handshake.bb = 6 : ui32, handshake.name = "fork31"} : <i7>
    %187 = trunci %186#0 {handshake.bb = 6 : ui32, handshake.name = "trunci11"} : <i7> to <i6>
    %188 = cmpi ult, %186#1, %168 {handshake.bb = 6 : ui32, handshake.name = "cmpi3"} : <i7>
    %189 = buffer %188, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 6 : ui32, handshake.name = "buffer34"} : <i1>
    %190:3 = fork [3] %189 {handshake.bb = 6 : ui32, handshake.name = "fork32"} : <i1>
    %trueResult_57, %falseResult_58 = cond_br %190#0, %187 {handshake.bb = 6 : ui32, handshake.name = "cond_br15"} : <i1>, <i6>
    sink %falseResult_58 {handshake.name = "sink12"} : <i6>
    %trueResult_59, %falseResult_60 = cond_br %190#2, %163#1 {handshake.bb = 6 : ui32, handshake.name = "cond_br16"} : <i1>, <>
    %result_61, %index_62 = control_merge [%falseResult_60]  {handshake.bb = 7 : ui32, handshake.name = "control_merge6"} : [<>] to <>, <i1>
    sink %index_62 {handshake.name = "sink13"} : <i1>
    %191:5 = fork [5] %result_61 {handshake.bb = 7 : ui32, handshake.name = "fork33"} : <>
    end {handshake.bb = 7 : ui32, handshake.name = "end0"} %memEnd_7, %memEnd_5, %memEnd_3, %memEnd_1, %memEnd, %0#1 : <>, <>, <>, <>, <>, <>
  }
}

