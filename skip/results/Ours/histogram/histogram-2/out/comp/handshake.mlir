module {
  handshake.func @histogram(%arg0: memref<1000xi32>, %arg1: memref<1000xf32>, %arg2: memref<1000xf32>, %arg3: !handshake.channel<i32>, %arg4: !handshake.control<>, %arg5: !handshake.control<>, %arg6: !handshake.control<>, %arg7: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["feature", "weight", "hist", "n", "feature_start", "weight_start", "hist_start", "start"], cfg.edges = "[0,1][2,1][1,2,3,cmpi0]", resNames = ["feature_end", "weight_end", "hist_end", "end"]} {
    %0:3 = lsq[%arg2 : memref<1000xf32>] (%arg6, %result_6, %addressResult_10, %addressResult_12, %dataResult_13, %result_14)  {groupSizes = [2 : i32], handshake.name = "lsq0"} : (!handshake.control<>, !handshake.control<>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<f32>, !handshake.control<>) -> (!handshake.channel<f32>, !handshake.control<>, !handshake.control<>)
    %outputs, %memEnd = mem_controller[%arg1 : memref<1000xf32>] %arg5 (%addressResult_8) %result_14 {connectedBlocks = [2 : i32], handshake.name = "mem_controller0"} :    (!handshake.channel<i32>) -> !handshake.channel<f32>
    %outputs_0, %memEnd_1 = mem_controller[%arg0 : memref<1000xi32>] %arg4 (%addressResult) %result_14 {connectedBlocks = [2 : i32], handshake.name = "mem_controller1"} :    (!handshake.channel<i32>) -> !handshake.channel<i32>
    %1 = constant %arg7 {handshake.bb = 0 : ui32, handshake.name = "constant0", value = 0 : i32} : <>, <i32>
    %2 = trunci %arg3 {handshake.bb = 0 : ui32, handshake.name = "index_cast0"} : <i32> to <i32>
    %3 = br %1 {handshake.bb = 0 : ui32, handshake.name = "br2"} : <i32>
    %4 = br %2 {handshake.bb = 0 : ui32, handshake.name = "br3"} : <i32>
    %5 = br %arg7 {handshake.bb = 0 : ui32, handshake.name = "br4"} : <>
    %6 = mux %index [%3, %16] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i32>, <i32>] to <i32>
    %7 = mux %index [%4, %17] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i1>, [<i32>, <i32>] to <i32>
    %result, %index = control_merge [%5, %18]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %8 = cmpi slt, %6, %7 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i32>
    %trueResult, %falseResult = cond_br %8, %7 {handshake.bb = 1 : ui32, handshake.name = "cond_br1"} : <i1>, <i32>
    %trueResult_2, %falseResult_3 = cond_br %8, %6 {handshake.bb = 1 : ui32, handshake.name = "cond_br2"} : <i1>, <i32>
    %trueResult_4, %falseResult_5 = cond_br %8, %result {handshake.bb = 1 : ui32, handshake.name = "cond_br3"} : <i1>, <>
    %9 = merge %trueResult {handshake.bb = 2 : ui32, handshake.name = "merge0"} : <i32>
    %10 = merge %trueResult_2 {handshake.bb = 2 : ui32, handshake.name = "merge1"} : <i32>
    %result_6, %index_7 = control_merge [%trueResult_4]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>] to <>, <i1>
    %11 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %12 = constant %11 {handshake.bb = 2 : ui32, handshake.name = "constant2", value = 1 : i32} : <>, <i32>
    %addressResult, %dataResult = load[%10] %outputs_0 {handshake.bb = 2 : ui32, handshake.name = "load0"} : <i32>, <i32>, <i32>, <i32>
    %addressResult_8, %dataResult_9 = load[%10] %outputs {handshake.bb = 2 : ui32, handshake.name = "load1"} : <i32>, <f32>, <i32>, <f32>
    %13 = trunci %dataResult {handshake.bb = 2 : ui32, handshake.name = "index_cast1"} : <i32> to <i32>
    %addressResult_10, %dataResult_11 = load[%13] %0#0 {handshake.bb = 2 : ui32, handshake.deps = #handshake<deps[["store1", 0, true]]>, handshake.name = "load5"} : <i32>, <f32>, <i32>, <f32>
    %14 = addf %dataResult_11, %dataResult_9 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "addf0"} : <f32>
    %addressResult_12, %dataResult_13, %doneResult = store[%13] %14 %0#1 {handshake.bb = 2 : ui32, handshake.deps = #handshake<deps[["load5", 0, true], ["store1", 0, true]]>, handshake.name = "store1"} : <i32>, <f32>, <>, <i32>, <f32>, <>
    %15 = addi %10, %12 {handshake.bb = 2 : ui32, handshake.name = "addi0"} : <i32>
    %16 = br %15 {handshake.bb = 2 : ui32, handshake.name = "br5"} : <i32>
    %17 = br %9 {handshake.bb = 2 : ui32, handshake.name = "br6"} : <i32>
    %18 = br %result_6 {handshake.bb = 2 : ui32, handshake.name = "br7"} : <>
    %result_14, %index_15 = control_merge [%falseResult_5]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>] to <>, <i1>
    end {handshake.bb = 3 : ui32, handshake.name = "end0"} %memEnd_1, %memEnd, %0#2, %arg7 : <>, <>, <>, <>
  }
}

