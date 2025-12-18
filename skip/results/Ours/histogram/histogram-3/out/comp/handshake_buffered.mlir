module {
  handshake.func @histogram(%arg0: memref<1000xi32>, %arg1: memref<1000xf32>, %arg2: memref<1000xf32>, %arg3: !handshake.channel<i32>, %arg4: !handshake.control<>, %arg5: !handshake.control<>, %arg6: !handshake.control<>, %arg7: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["feature", "weight", "hist", "n", "feature_start", "weight_start", "hist_start", "start"], cfg.edges = "[0,1][2,1][1,2,3,cmpi0]", resNames = ["feature_end", "weight_end", "hist_end", "end"]} {
    %0:8 = fork [8] %arg7 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork0"} : <>
    %outputs:2, %memEnd = mem_controller[%arg2 : memref<1000xf32>] %arg6 (%61, %addressResult_32, %addressResult_34, %dataResult_35) %109#2 {connectedBlocks = [2 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "5": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller2"} :    (!handshake.channel<i32>, !handshake.channel<i10>, !handshake.channel<i10>, !handshake.channel<f32>) -> (!handshake.channel<f32>, !handshake.control<>)
    %outputs_0, %memEnd_1 = mem_controller[%arg1 : memref<1000xf32>] %arg5 (%addressResult_24) %109#1 {connectedBlocks = [2 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller3"} :    (!handshake.channel<i10>) -> !handshake.channel<f32>
    %outputs_2, %memEnd_3 = mem_controller[%arg0 : memref<1000xi32>] %arg4 (%addressResult) %109#0 {connectedBlocks = [2 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller4"} :    (!handshake.channel<i10>) -> !handshake.channel<i32>
    %1 = constant %0#1 {handshake.bb = 0 : ui32, handshake.name = "constant6", value = 1000 : i11} : <>, <i11>
    %2:3 = fork [3] %1 {handshake.bb = 0 : ui32, handshake.name = "fork1"} : <i11>
    %3 = extsi %2#0 {handshake.bb = 1 : ui32, handshake.name = "extsi0"} : <i11> to <i32>
    %4 = extsi %2#1 {handshake.bb = 1 : ui32, handshake.name = "extsi1"} : <i11> to <i32>
    %5 = extsi %2#2 {handshake.bb = 1 : ui32, handshake.name = "extsi2"} : <i11> to <i32>
    %6 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant7", value = false} : <>, <i1>
    %7 = br %6 {handshake.bb = 0 : ui32, handshake.name = "br2"} : <i1>
    %8 = extsi %7 {handshake.bb = 0 : ui32, handshake.name = "extsi6"} : <i1> to <i32>
    %9 = br %arg3 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "br3"} : <i32>
    %10 = br %0#7 {handshake.bb = 0 : ui32, handshake.name = "br4"} : <>
    %11 = mux %24#0 [%0#6, %104] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux2"} : <i1>, [<>, <>] to <>
    %12 = mux %24#1 [%3, %13] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux3"} : <i1>, [<i32>, <i32>] to <i32>
    %13 = buffer %95#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer6"} : <i32>
    %14 = mux %24#2 [%4, %96] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux4"} : <i1>, [<i32>, <i32>] to <i32>
    %15 = mux %24#3 [%5, %93#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux5"} : <i1>, [<i32>, <i32>] to <i32>
    %16 = mux %17 [%0#5, %99#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux6"} : <i1>, [<>, <>] to <>
    %17 = buffer %24#4, bufferType = FIFO_BREAK_NONE, numSlots = 3 {handshake.bb = 1 : ui32, handshake.name = "buffer11"} : <i1>
    %18 = mux %19 [%0#4, %101#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux7"} : <i1>, [<>, <>] to <>
    %19 = buffer %24#5, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 1 : ui32, handshake.name = "buffer12"} : <i1>
    %20 = mux %21 [%0#3, %103#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux8"} : <i1>, [<>, <>] to <>
    %21 = buffer %24#6, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer13"} : <i1>
    %22 = init %23 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "init0"} : <i1>
    %23 = buffer %37#10, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer14"} : <i1>
    %24:7 = fork [7] %22 {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <i1>
    %25 = buffer %106, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer31"} : <i32>
    %26 = mux %34#0 [%8, %25] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i32>, <i32>] to <i32>
    %27 = buffer %26, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer15"} : <i32>
    %28 = buffer %27, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer16"} : <i32>
    %29:2 = fork [2] %28 {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <i32>
    %30 = mux %34#1 [%9, %107] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i1>, [<i32>, <i32>] to <i32>
    %31 = buffer %30, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer17"} : <i32>
    %32 = buffer %31, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer18"} : <i32>
    %33:2 = fork [2] %32 {handshake.bb = 1 : ui32, handshake.name = "fork4"} : <i32>
    %result, %index = control_merge [%10, %108]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %34:2 = fork [2] %index {handshake.bb = 1 : ui32, handshake.name = "fork5"} : <i1>
    %35 = cmpi slt, %29#1, %33#1 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i32>
    %36 = buffer %35, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer19"} : <i1>
    %37:11 = fork [11] %36 {handshake.bb = 1 : ui32, handshake.name = "fork6"} : <i1>
    %trueResult, %falseResult = cond_br %37#9, %33#0 {handshake.bb = 1 : ui32, handshake.name = "cond_br1"} : <i1>, <i32>
    sink %falseResult {handshake.name = "sink0"} : <i32>
    %trueResult_4, %falseResult_5 = cond_br %37#8, %29#0 {handshake.bb = 1 : ui32, handshake.name = "cond_br2"} : <i1>, <i32>
    sink %falseResult_5 {handshake.name = "sink1"} : <i32>
    %trueResult_6, %falseResult_7 = cond_br %37#7, %result {handshake.bb = 1 : ui32, handshake.name = "cond_br3"} : <i1>, <>
    %38 = buffer %14, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer5"} : <i32>
    %trueResult_8, %falseResult_9 = cond_br %39, %38 {handshake.bb = 2 : ui32, handshake.name = "cond_br13"} : <i1>, <i32>
    %39 = buffer %37#6, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer24"} : <i1>
    sink %falseResult_9 {handshake.name = "sink2"} : <i32>
    %40 = buffer %20, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer10"} : <>
    %trueResult_10, %falseResult_11 = cond_br %41, %40 {handshake.bb = 2 : ui32, handshake.name = "cond_br14"} : <i1>, <>
    %41 = buffer %37#5, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 2 : ui32, handshake.name = "buffer25"} : <i1>
    sink %falseResult_11 {handshake.name = "sink3"} : <>
    %42 = buffer %18, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer9"} : <>
    %trueResult_12, %falseResult_13 = cond_br %43, %42 {handshake.bb = 2 : ui32, handshake.name = "cond_br15"} : <i1>, <>
    %43 = buffer %37#4, bufferType = FIFO_BREAK_NONE, numSlots = 3 {handshake.bb = 2 : ui32, handshake.name = "buffer26"} : <i1>
    sink %falseResult_13 {handshake.name = "sink4"} : <>
    %44 = buffer %12, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer4"} : <i32>
    %trueResult_14, %falseResult_15 = cond_br %45, %44 {handshake.bb = 2 : ui32, handshake.name = "cond_br16"} : <i1>, <i32>
    %45 = buffer %37#3, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer27"} : <i1>
    sink %falseResult_15 {handshake.name = "sink5"} : <i32>
    %46 = buffer %11, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer2"} : <>
    %47 = buffer %46, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer3"} : <>
    %trueResult_16, %falseResult_17 = cond_br %48, %47 {handshake.bb = 2 : ui32, handshake.name = "cond_br17"} : <i1>, <>
    %48 = buffer %37#2, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer28"} : <i1>
    sink %falseResult_17 {handshake.name = "sink6"} : <>
    %49 = buffer %15, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer7"} : <i32>
    %trueResult_18, %falseResult_19 = cond_br %50, %49 {handshake.bb = 2 : ui32, handshake.name = "cond_br18"} : <i1>, <i32>
    %50 = buffer %37#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer29"} : <i1>
    sink %falseResult_19 {handshake.name = "sink7"} : <i32>
    %51 = buffer %16, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer8"} : <>
    %trueResult_20, %falseResult_21 = cond_br %52, %51 {handshake.bb = 2 : ui32, handshake.name = "cond_br19"} : <i1>, <>
    %52 = buffer %37#0, bufferType = FIFO_BREAK_NONE, numSlots = 4 {handshake.bb = 2 : ui32, handshake.name = "buffer30"} : <i1>
    sink %falseResult_21 {handshake.name = "sink8"} : <>
    %53 = merge %trueResult {handshake.bb = 2 : ui32, handshake.name = "merge0"} : <i32>
    %54 = merge %trueResult_4 {handshake.bb = 2 : ui32, handshake.name = "merge1"} : <i32>
    %55:3 = fork [3] %54 {handshake.bb = 2 : ui32, handshake.name = "fork7"} : <i32>
    %56 = trunci %55#0 {handshake.bb = 2 : ui32, handshake.name = "trunci0"} : <i32> to <i10>
    %57 = trunci %55#1 {handshake.bb = 2 : ui32, handshake.name = "trunci1"} : <i32> to <i10>
    %result_22, %index_23 = control_merge [%trueResult_6]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>] to <>, <i1>
    sink %index_23 {handshake.name = "sink9"} : <i1>
    %58 = buffer %result_22, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer20"} : <>
    %59:2 = fork [2] %58 {handshake.bb = 2 : ui32, handshake.name = "fork8"} : <>
    %60 = constant %59#0 {handshake.bb = 2 : ui32, handshake.name = "constant8", value = 1 : i2} : <>, <i2>
    %61 = extsi %60 {handshake.bb = 2 : ui32, handshake.name = "extsi4"} : <i2> to <i32>
    %62 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %63 = constant %62 {handshake.bb = 2 : ui32, handshake.name = "constant9", value = 1 : i2} : <>, <i2>
    %64 = extsi %63 {handshake.bb = 2 : ui32, handshake.name = "extsi5"} : <i2> to <i32>
    %addressResult, %dataResult = load[%57] %outputs_2 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i10>, <i32>, <i10>, <i32>
    %65:4 = fork [4] %dataResult {handshake.bb = 2 : ui32, handshake.name = "fork9"} : <i32>
    %66 = trunci %67 {handshake.bb = 2 : ui32, handshake.name = "trunci2"} : <i32> to <i10>
    %67 = buffer %65#0, bufferType = FIFO_BREAK_NONE, numSlots = 4 {handshake.bb = 2 : ui32, handshake.name = "buffer33"} : <i32>
    %addressResult_24, %dataResult_25 = load[%56] %outputs_0 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i10>, <f32>, <i10>, <f32>
    %68 = gate %65#1, %trueResult_16 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "gate0"} : <i32>, !handshake.control<> to <i32>
    %69:3 = fork [3] %68 {handshake.bb = 2 : ui32, handshake.name = "fork10"} : <i32>
    %70 = cmpi ne, %69#2, %trueResult_18 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi1"} : <i32>
    %71:2 = fork [2] %70 {handshake.bb = 2 : ui32, handshake.name = "fork11"} : <i1>
    %72 = cmpi ne, %69#1, %trueResult_14 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi2"} : <i32>
    %73:2 = fork [2] %72 {handshake.bb = 2 : ui32, handshake.name = "fork12"} : <i1>
    %74 = cmpi ne, %69#0, %trueResult_8 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi3"} : <i32>
    %75:2 = fork [2] %74 {handshake.bb = 2 : ui32, handshake.name = "fork13"} : <i1>
    %trueResult_26, %falseResult_27 = cond_br %76, %trueResult_20 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br10"} : <i1>, <>
    %76 = buffer %71#1, bufferType = FIFO_BREAK_NONE, numSlots = 3 {handshake.bb = 2 : ui32, handshake.name = "buffer38"} : <i1>
    sink %trueResult_26 {handshake.name = "sink10"} : <>
    %trueResult_28, %falseResult_29 = cond_br %77, %trueResult_12 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br11"} : <i1>, <>
    %77 = buffer %73#1, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 2 : ui32, handshake.name = "buffer39"} : <i1>
    sink %trueResult_28 {handshake.name = "sink11"} : <>
    %trueResult_30, %falseResult_31 = cond_br %78, %trueResult_10 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br12"} : <i1>, <>
    %78 = buffer %75#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer40"} : <i1>
    sink %trueResult_30 {handshake.name = "sink12"} : <>
    %79 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %80 = mux %71#0 [%falseResult_27, %79] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux9"} : <i1>, [<>, <>] to <>
    %81 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source2"} : <>
    %82 = mux %73#0 [%falseResult_29, %81] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux10"} : <i1>, [<>, <>] to <>
    %83 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source3"} : <>
    %84 = mux %75#0 [%falseResult_31, %83] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux11"} : <i1>, [<>, <>] to <>
    %85 = buffer %80, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer21"} : <>
    %86 = buffer %82, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer22"} : <>
    %87 = buffer %84, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer23"} : <>
    %88 = join %85, %86, %87 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "join0"} : <>
    %89 = gate %65#2, %88 {handshake.bb = 2 : ui32, handshake.name = "gate1"} : <i32>, !handshake.control<> to <i32>
    %90 = trunci %89 {handshake.bb = 2 : ui32, handshake.name = "trunci3"} : <i32> to <i10>
    %addressResult_32, %dataResult_33 = load[%90] %outputs#0 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["store1", 0, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load5"} : <i10>, <f32>, <i10>, <f32>
    %91 = addf %dataResult_33, %dataResult_25 {fastmath = #arith.fastmath<none>, fpu_impl = #handshake<fpu_impl flopoco>, handshake.bb = 2 : ui32, handshake.name = "addf0", internal_delay = "2_922000"} : <f32>
    %92 = buffer %65#3, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer0"} : <i32>
    %93:2 = fork [2] %92 {handshake.bb = 2 : ui32, handshake.name = "fork14"} : <i32>
    %94 = init %93#0 {handshake.bb = 2 : ui32, handshake.name = "init7"} : <i32>
    %95:2 = fork [2] %94 {handshake.bb = 2 : ui32, handshake.name = "fork15"} : <i32>
    %96 = init %97 {handshake.bb = 2 : ui32, handshake.name = "init8"} : <i32>
    %97 = buffer %95#0, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 2 : ui32, handshake.name = "buffer47"} : <i32>
    %98 = buffer %doneResult, bufferType = FIFO_BREAK_NONE, numSlots = 1 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "buffer1"} : <>
    %99:2 = fork [2] %98 {handshake.bb = 2 : ui32, handshake.name = "fork16"} : <>
    %100 = init %99#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init9"} : <>
    %101:2 = fork [2] %100 {handshake.bb = 2 : ui32, handshake.name = "fork17"} : <>
    %102 = init %101#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init10"} : <>
    %103:2 = fork [2] %102 {handshake.bb = 2 : ui32, handshake.name = "fork18"} : <>
    %104 = init %103#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init11"} : <>
    %addressResult_34, %dataResult_35, %doneResult = store[%66] %91 %outputs#1 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["load5", 0, false], ["store1", 0, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store1"} : <i10>, <f32>, <>, <i10>, <f32>, <>
    %105 = addi %55#2, %64 {handshake.bb = 2 : ui32, handshake.name = "addi0"} : <i32>
    %106 = br %105 {handshake.bb = 2 : ui32, handshake.name = "br5"} : <i32>
    %107 = br %53 {handshake.bb = 2 : ui32, handshake.name = "br6"} : <i32>
    %108 = br %59#1 {handshake.bb = 2 : ui32, handshake.name = "br7"} : <>
    %result_36, %index_37 = control_merge [%falseResult_7]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>] to <>, <i1>
    sink %index_37 {handshake.name = "sink13"} : <i1>
    %109:3 = fork [3] %result_36 {handshake.bb = 3 : ui32, handshake.name = "fork19"} : <>
    end {handshake.bb = 3 : ui32, handshake.name = "end0"} %memEnd_3, %memEnd_1, %memEnd, %0#2 : <>, <>, <>, <>
  }
}

