module {
  handshake.func @histogram(%arg0: memref<1000xi32>, %arg1: memref<1000xf32>, %arg2: memref<1000xf32>, %arg3: !handshake.channel<i32>, %arg4: !handshake.control<>, %arg5: !handshake.control<>, %arg6: !handshake.control<>, %arg7: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["feature", "weight", "hist", "n", "feature_start", "weight_start", "hist_start", "start"], cfg.edges = "[0,1][2,1][1,2,3,cmpi0]", resNames = ["feature_end", "weight_end", "hist_end", "end"]} {
    %0:7 = fork [7] %arg7 {handshake.bb = 0 : ui32, handshake.name = "fork0"} : <>
    %outputs:2, %memEnd = mem_controller[%arg2 : memref<1000xf32>] %arg6 (%56, %addressResult_26, %addressResult_28, %dataResult_29) %100#2 {connectedBlocks = [2 : i32], handshake.name = "mem_controller2"} :    (!handshake.channel<i32>, !handshake.channel<i10>, !handshake.channel<i10>, !handshake.channel<f32>) -> (!handshake.channel<f32>, !handshake.control<>)
    %outputs_0, %memEnd_1 = mem_controller[%arg1 : memref<1000xf32>] %arg5 (%addressResult_20) %100#1 {connectedBlocks = [2 : i32], handshake.name = "mem_controller3"} :    (!handshake.channel<i10>) -> !handshake.channel<f32>
    %outputs_2, %memEnd_3 = mem_controller[%arg0 : memref<1000xi32>] %arg4 (%addressResult) %100#0 {connectedBlocks = [2 : i32], handshake.name = "mem_controller4"} :    (!handshake.channel<i10>) -> !handshake.channel<i32>
    %1 = constant %0#1 {handshake.bb = 0 : ui32, handshake.name = "constant5", value = 1000 : i11} : <>, <i11>
    %2:2 = fork [2] %1 {handshake.bb = 0 : ui32, handshake.name = "fork1"} : <i11>
    %3 = extsi %2#0 {handshake.bb = 1 : ui32, handshake.name = "extsi0"} : <i11> to <i32>
    %5 = extsi %2#1 {handshake.bb = 1 : ui32, handshake.name = "extsi1"} : <i11> to <i32>
    %7 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant6", value = false} : <>, <i1>
    %8 = br %7 {handshake.bb = 0 : ui32, handshake.name = "br2"} : <i1>
    %9 = extsi %8 {handshake.bb = 0 : ui32, handshake.name = "extsi5"} : <i1> to <i32>
    %10 = br %arg3 {handshake.bb = 0 : ui32, handshake.name = "br3"} : <i32>
    %11 = br %0#6 {handshake.bb = 0 : ui32, handshake.name = "br4"} : <>
    %12 = mux %25#0 [%0#5, %94] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux2"} : <i1>, [<>, <>] to <>
    %14 = mux %25#1 [%3, %88] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux3"} : <i1>, [<i32>, <i32>] to <i32>
    %16 = mux %25#2 [%5, %87#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux4"} : <i1>, [<i32>, <i32>] to <i32>
    %19 = mux %20 [%0#4, %91#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux5"} : <i1>, [<>, <>] to <>
    %20 = buffer %25#3, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 1 : ui32, handshake.name = "buffer8"} : <i1>
    %21 = mux %22 [%0#3, %93#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux6"} : <i1>, [<>, <>] to <>
    %22 = buffer %25#4, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer9"} : <i1>
    %23 = init %24 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "init0"} : <i1>
    %24 = buffer %36#8, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer10"} : <i1>
    %25:5 = fork [5] %23 {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <i1>
    %26 = mux %32#0 [%9, %97] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i32>, <i32>] to <i32>
    %28:2 = fork [2] %26 {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <i32>
    %29 = mux %32#1 [%10, %98] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i1>, [<i32>, <i32>] to <i32>
    %31:2 = fork [2] %29 {handshake.bb = 1 : ui32, handshake.name = "fork4"} : <i32>
    %result, %index = control_merge [%11, %99]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %32:2 = fork [2] %index {handshake.bb = 1 : ui32, handshake.name = "fork5"} : <i1>
    %33 = cmpi slt, %28#1, %31#1 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i32>
    %36:9 = fork [9] %33 {handshake.bb = 1 : ui32, handshake.name = "fork6"} : <i1>
    %trueResult, %falseResult = cond_br %36#7, %31#0 {handshake.bb = 1 : ui32, handshake.name = "cond_br1"} : <i1>, <i32>
    sink %falseResult {handshake.name = "sink0"} : <i32>
    %trueResult_4, %falseResult_5 = cond_br %36#6, %28#0 {handshake.bb = 1 : ui32, handshake.name = "cond_br2"} : <i1>, <i32>
    sink %falseResult_5 {handshake.name = "sink1"} : <i32>
    %trueResult_6, %falseResult_7 = cond_br %36#5, %result {handshake.bb = 1 : ui32, handshake.name = "cond_br3"} : <i1>, <>
    %trueResult_8, %falseResult_9 = cond_br %42, %19 {handshake.bb = 2 : ui32, handshake.name = "cond_br10"} : <i1>, <>
    %42 = buffer %36#4, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 2 : ui32, handshake.name = "buffer20"} : <i1>
    sink %falseResult_9 {handshake.name = "sink2"} : <>
    %trueResult_10, %falseResult_11 = cond_br %36#3, %12 {handshake.bb = 2 : ui32, handshake.name = "cond_br11"} : <i1>, <>
    sink %falseResult_11 {handshake.name = "sink3"} : <>
    %trueResult_12, %falseResult_13 = cond_br %36#2, %14 {handshake.bb = 2 : ui32, handshake.name = "cond_br12"} : <i1>, <i32>
    sink %falseResult_13 {handshake.name = "sink4"} : <i32>
    %trueResult_14, %falseResult_15 = cond_br %36#1, %16 {handshake.bb = 2 : ui32, handshake.name = "cond_br13"} : <i1>, <i32>
    sink %falseResult_15 {handshake.name = "sink5"} : <i32>
    %trueResult_16, %falseResult_17 = cond_br %46, %21 {handshake.bb = 2 : ui32, handshake.name = "cond_br14"} : <i1>, <>
    %46 = buffer %36#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer24"} : <i1>
    sink %falseResult_17 {handshake.name = "sink6"} : <>
    %47 = merge %trueResult {handshake.bb = 2 : ui32, handshake.name = "merge0"} : <i32>
    %48 = merge %trueResult_4 {handshake.bb = 2 : ui32, handshake.name = "merge1"} : <i32>
    %49:3 = fork [3] %48 {handshake.bb = 2 : ui32, handshake.name = "fork7"} : <i32>
    %50 = trunci %49#0 {handshake.bb = 2 : ui32, handshake.name = "trunci0"} : <i32> to <i10>
    %52 = trunci %49#1 {handshake.bb = 2 : ui32, handshake.name = "trunci1"} : <i32> to <i10>
    %result_18, %index_19 = control_merge [%trueResult_6]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>] to <>, <i1>
    sink %index_19 {handshake.name = "sink7"} : <i1>
    %54:2 = fork [2] %result_18 {handshake.bb = 2 : ui32, handshake.name = "fork8"} : <>
    %55 = constant %54#0 {handshake.bb = 2 : ui32, handshake.name = "constant7", value = 1 : i2} : <>, <i2>
    %56 = extsi %55 {handshake.bb = 2 : ui32, handshake.name = "extsi3"} : <i2> to <i32>
    %57 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %58 = constant %57 {handshake.bb = 2 : ui32, handshake.name = "constant8", value = 1 : i2} : <>, <i2>
    %59 = extsi %58 {handshake.bb = 2 : ui32, handshake.name = "extsi4"} : <i2> to <i32>
    %addressResult, %dataResult = load[%52] %outputs_2 {handshake.bb = 2 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i10>, <i32>, <i10>, <i32>
    %60:4 = fork [4] %dataResult {handshake.bb = 2 : ui32, handshake.name = "fork9"} : <i32>
    %61 = trunci %62 {handshake.bb = 2 : ui32, handshake.name = "trunci2"} : <i32> to <i10>
    %62 = buffer %60#0, bufferType = FIFO_BREAK_NONE, numSlots = 3 {handshake.bb = 2 : ui32, handshake.name = "buffer27"} : <i32>
    %addressResult_20, %dataResult_21 = load[%50] %outputs_0 {handshake.bb = 2 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i10>, <f32>, <i10>, <f32>
    %63 = gate %60#1, %trueResult_10 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "gate0"} : <i32>, !handshake.control<> to <i32>
    %65:2 = fork [2] %63 {handshake.bb = 2 : ui32, handshake.name = "fork10"} : <i32>
    %66 = cmpi ne, %65#1, %trueResult_14 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi1"} : <i32>
    %68:2 = fork [2] %66 {handshake.bb = 2 : ui32, handshake.name = "fork11"} : <i1>
    %69 = cmpi ne, %65#0, %trueResult_12 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi2"} : <i32>
    %71:2 = fork [2] %69 {handshake.bb = 2 : ui32, handshake.name = "fork12"} : <i1>
    %trueResult_22, %falseResult_23 = cond_br %72, %trueResult_8 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br8"} : <i1>, <>
    %72 = buffer %68#1, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 2 : ui32, handshake.name = "buffer31"} : <i1>
    sink %trueResult_22 {handshake.name = "sink8"} : <>
    %trueResult_24, %falseResult_25 = cond_br %73, %trueResult_16 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br9"} : <i1>, <>
    %73 = buffer %71#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer32"} : <i1>
    sink %trueResult_24 {handshake.name = "sink9"} : <>
    %74 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %75 = mux %68#0 [%falseResult_23, %74] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux7"} : <i1>, [<>, <>] to <>
    %77 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source2"} : <>
    %78 = mux %71#0 [%falseResult_25, %77] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux8"} : <i1>, [<>, <>] to <>
    %80 = join %75, %78 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "join0"} : <>
    %81 = gate %60#2, %80 {handshake.bb = 2 : ui32, handshake.name = "gate1"} : <i32>, !handshake.control<> to <i32>
    %83 = trunci %81 {handshake.bb = 2 : ui32, handshake.name = "trunci3"} : <i32> to <i10>
    %addressResult_26, %dataResult_27 = load[%83] %outputs#0 {handshake.bb = 2 : ui32, handshake.deps = #handshake<deps[["store1", 0, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load5"} : <i10>, <f32>, <i10>, <f32>
    %84 = addf %dataResult_27, %dataResult_21 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "addf0"} : <f32>
    %85 = buffer %60#3, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer0"} : <i32>
    %87:2 = fork [2] %85 {handshake.bb = 2 : ui32, handshake.name = "fork13"} : <i32>
    %88 = init %89 {handshake.bb = 2 : ui32, handshake.name = "init5"} : <i32>
    %89 = buffer %87#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer37"} : <i32>
    %90 = buffer %doneResult, bufferType = FIFO_BREAK_NONE, numSlots = 1 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "buffer1"} : <>
    %91:2 = fork [2] %90 {handshake.bb = 2 : ui32, handshake.name = "fork14"} : <>
    %92 = init %91#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init6"} : <>
    %93:2 = fork [2] %92 {handshake.bb = 2 : ui32, handshake.name = "fork15"} : <>
    %94 = init %93#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init7"} : <>
    %addressResult_28, %dataResult_29, %doneResult = store[%61] %84 %outputs#1 {handshake.bb = 2 : ui32, handshake.deps = #handshake<deps[["load5", 0, false], ["store1", 0, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store1"} : <i10>, <f32>, <>, <i10>, <f32>, <>
    %95 = addi %49#2, %59 {handshake.bb = 2 : ui32, handshake.name = "addi0"} : <i32>
    %97 = br %95 {handshake.bb = 2 : ui32, handshake.name = "br5"} : <i32>
    %98 = br %47 {handshake.bb = 2 : ui32, handshake.name = "br6"} : <i32>
    %99 = br %54#1 {handshake.bb = 2 : ui32, handshake.name = "br7"} : <>
    %result_30, %index_31 = control_merge [%falseResult_7]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>] to <>, <i1>
    sink %index_31 {handshake.name = "sink10"} : <i1>
    %100:3 = fork [3] %result_30 {handshake.bb = 3 : ui32, handshake.name = "fork16"} : <>
    end {handshake.bb = 3 : ui32, handshake.name = "end0"} %memEnd_3, %memEnd_1, %memEnd, %0#2 : <>, <>, <>, <>
  }
}

