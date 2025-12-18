module {
  handshake.func @histogram(%arg0: memref<1000xi32>, %arg1: memref<1000xf32>, %arg2: memref<1000xf32>, %arg3: !handshake.channel<i32>, %arg4: !handshake.control<>, %arg5: !handshake.control<>, %arg6: !handshake.control<>, %arg7: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["feature", "weight", "hist", "n", "feature_start", "weight_start", "hist_start", "start"], cfg.edges = "[0,1][2,1][1,2,3,cmpi0]", resNames = ["feature_end", "weight_end", "hist_end", "end"]} {
    %outputs:2, %memEnd = mem_controller[%arg2 : memref<1000xf32>] %arg6 (%14, %addressResult_20, %addressResult_22, %dataResult_23) %result_24 {connectedBlocks = [2 : i32], handshake.name = "mem_controller2"} :    (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<f32>) -> (!handshake.channel<f32>, !handshake.control<>)
    %outputs_0, %memEnd_1 = mem_controller[%arg1 : memref<1000xf32>] %arg5 (%addressResult_16) %result_24 {connectedBlocks = [2 : i32], handshake.name = "mem_controller3"} :    (!handshake.channel<i32>) -> !handshake.channel<f32>
    %outputs_2, %memEnd_3 = mem_controller[%arg0 : memref<1000xi32>] %arg4 (%addressResult) %result_24 {connectedBlocks = [2 : i32], handshake.name = "mem_controller4"} :    (!handshake.channel<i32>) -> !handshake.channel<i32>
    %0 = constant %arg7 {handshake.bb = 0 : ui32, handshake.name = "constant1", value = 1000 : i32} : <>, <i32>
    %1 = constant %arg7 {handshake.bb = 0 : ui32, handshake.name = "constant0", value = 0 : i32} : <>, <i32>
    %2 = br %1 {handshake.bb = 0 : ui32, handshake.name = "br2"} : <i32>
    %3 = br %arg3 {handshake.bb = 0 : ui32, handshake.name = "br3"} : <i32>
    %4 = br %arg7 {handshake.bb = 0 : ui32, handshake.name = "br4"} : <>
    %5 = mux %8 [%arg7, %26] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux2"} : <i1>, [<>, <>] to <>
    %6 = mux %8 [%0, %24] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux3"} : <i1>, [<i32>, <i32>] to <i32>
    %7 = mux %8 [%arg7, %25] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux4"} : <i1>, [<>, <>] to <>
    %8 = init %11 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "init0"} : <i1>
    %9 = mux %index [%2, %28] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i32>, <i32>] to <i32>
    %10 = mux %index [%3, %29] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i1>, [<i32>, <i32>] to <i32>
    %result, %index = control_merge [%4, %30]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %11 = cmpi slt, %9, %10 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i32>
    %trueResult, %falseResult = cond_br %11, %10 {handshake.bb = 1 : ui32, handshake.name = "cond_br1"} : <i1>, <i32>
    %trueResult_4, %falseResult_5 = cond_br %11, %9 {handshake.bb = 1 : ui32, handshake.name = "cond_br2"} : <i1>, <i32>
    %trueResult_6, %falseResult_7 = cond_br %11, %result {handshake.bb = 1 : ui32, handshake.name = "cond_br3"} : <i1>, <>
    %trueResult_8, %falseResult_9 = cond_br %11, %7 {handshake.bb = 2 : ui32, handshake.name = "cond_br7"} : <i1>, <>
    %trueResult_10, %falseResult_11 = cond_br %11, %5 {handshake.bb = 2 : ui32, handshake.name = "cond_br8"} : <i1>, <>
    %trueResult_12, %falseResult_13 = cond_br %11, %6 {handshake.bb = 2 : ui32, handshake.name = "cond_br9"} : <i1>, <i32>
    %12 = merge %trueResult {handshake.bb = 2 : ui32, handshake.name = "merge0"} : <i32>
    %13 = merge %trueResult_4 {handshake.bb = 2 : ui32, handshake.name = "merge1"} : <i32>
    %result_14, %index_15 = control_merge [%trueResult_6]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>] to <>, <i1>
    %14 = constant %result_14 {handshake.bb = 2 : ui32, handshake.name = "constant3", value = 1 : i32} : <>, <i32>
    %15 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %16 = constant %15 {handshake.bb = 2 : ui32, handshake.name = "constant2", value = 1 : i32} : <>, <i32>
    %addressResult, %dataResult = load[%13] %outputs_2 {handshake.bb = 2 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i32>, <i32>, <i32>, <i32>
    %addressResult_16, %dataResult_17 = load[%13] %outputs_0 {handshake.bb = 2 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i32>, <f32>, <i32>, <f32>
    %17 = gate %dataResult, %trueResult_10 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "gate0"} : <i32>, !handshake.control<> to <i32>
    %18 = cmpi ne, %17, %trueResult_12 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi1"} : <i32>
    %trueResult_18, %falseResult_19 = cond_br %18, %trueResult_8 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br6"} : <i1>, <>
    %19 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %20 = mux %18 [%falseResult_19, %19] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux5"} : <i1>, [<>, <>] to <>
    %21 = join %20 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "join0"} : <>
    %22 = gate %dataResult, %21 {handshake.bb = 2 : ui32, handshake.name = "gate1"} : <i32>, !handshake.control<> to <i32>
    %addressResult_20, %dataResult_21 = load[%22] %outputs#0 {handshake.bb = 2 : ui32, handshake.deps = #handshake<deps[["store1", 0, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load5"} : <i32>, <f32>, <i32>, <f32>
    %23 = addf %dataResult_21, %dataResult_17 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "addf0"} : <f32>
    %24 = buffer %dataResult, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer0"} : <i32>
    %25 = buffer %doneResult, bufferType = FIFO_BREAK_NONE, numSlots = 1 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "buffer1"} : <>
    %26 = init %25 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init3"} : <>
    %addressResult_22, %dataResult_23, %doneResult = store[%dataResult] %23 %outputs#1 {handshake.bb = 2 : ui32, handshake.deps = #handshake<deps[["load5", 0, false], ["store1", 0, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store1"} : <i32>, <f32>, <>, <i32>, <f32>, <>
    %27 = addi %13, %16 {handshake.bb = 2 : ui32, handshake.name = "addi0"} : <i32>
    %28 = br %27 {handshake.bb = 2 : ui32, handshake.name = "br5"} : <i32>
    %29 = br %12 {handshake.bb = 2 : ui32, handshake.name = "br6"} : <i32>
    %30 = br %result_14 {handshake.bb = 2 : ui32, handshake.name = "br7"} : <>
    %result_24, %index_25 = control_merge [%falseResult_7]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>] to <>, <i1>
    end {handshake.bb = 3 : ui32, handshake.name = "end0"} %memEnd_3, %memEnd_1, %memEnd, %arg7 : <>, <>, <>, <>
  }
}

