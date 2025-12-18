module {
  handshake.func @histogram(%arg0: memref<1000xi32>, %arg1: memref<1000xf32>, %arg2: memref<1000xf32>, %arg3: !handshake.channel<i32>, %arg4: !handshake.control<>, %arg5: !handshake.control<>, %arg6: !handshake.control<>, %arg7: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["feature", "weight", "hist", "n", "feature_start", "weight_start", "hist_start", "start"], resNames = ["feature_end", "weight_end", "hist_end", "end"]} {
    %0:3 = fork [3] %arg7 {handshake.bb = 0 : ui32, handshake.name = "fork0"} : <>
    %1:2 = lsq[%arg2 : memref<1000xf32>] (%arg6, %30#0, %addressResult_10, %addressResult_12, %dataResult_13, %45#2)  {groupSizes = [2 : i32], handshake.name = "lsq1"} : (!handshake.control<>, !handshake.control<>, !handshake.channel<i10>, !handshake.channel<i10>, !handshake.channel<f32>, !handshake.control<>) -> (!handshake.channel<f32>, !handshake.control<>)
    %outputs, %memEnd = mem_controller[%arg1 : memref<1000xf32>] %arg5 (%addressResult_8) %45#1 {connectedBlocks = [2 : i32], handshake.name = "mem_controller2"} :    (!handshake.channel<i10>) -> !handshake.channel<f32>
    %outputs_0, %memEnd_1 = mem_controller[%arg0 : memref<1000xi32>] %arg4 (%addressResult) %45#0 {connectedBlocks = [2 : i32], handshake.name = "mem_controller3"} :    (!handshake.channel<i10>) -> !handshake.channel<i32>
    %2 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant1", value = false} : <>, <i1>
    %3 = br %2 {handshake.bb = 0 : ui32, handshake.name = "br2"} : <i1>
    %4 = extsi %3 {handshake.bb = 0 : ui32, handshake.name = "extsi2"} : <i1> to <i32>
    %5 = br %arg3 {handshake.bb = 0 : ui32, handshake.name = "br3"} : <i32>
    %6 = br %0#2 {handshake.bb = 0 : ui32, handshake.name = "br4"} : <>
    %7 = mux %13#0 [%4, %42] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i32>, <i32>] to <i32>
    %9:2 = fork [2] %7 {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <i32>
    %10 = mux %13#1 [%5, %43] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i1>, [<i32>, <i32>] to <i32>
    %12:2 = fork [2] %10 {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <i32>
    %result, %index = control_merge [%6, %44]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %13:2 = fork [2] %index {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <i1>
    %14 = cmpi slt, %9#1, %12#1 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i32>
    %17:3 = fork [3] %14 {handshake.bb = 1 : ui32, handshake.name = "fork4"} : <i1>
    %trueResult, %falseResult = cond_br %17#2, %12#0 {handshake.bb = 1 : ui32, handshake.name = "cond_br1"} : <i1>, <i32>
    sink %falseResult {handshake.name = "sink0"} : <i32>
    %trueResult_2, %falseResult_3 = cond_br %17#1, %9#0 {handshake.bb = 1 : ui32, handshake.name = "cond_br2"} : <i1>, <i32>
    sink %falseResult_3 {handshake.name = "sink1"} : <i32>
    %trueResult_4, %falseResult_5 = cond_br %17#0, %result {handshake.bb = 1 : ui32, handshake.name = "cond_br3"} : <i1>, <>
    %23 = merge %trueResult {handshake.bb = 2 : ui32, handshake.name = "merge0"} : <i32>
    %24 = merge %trueResult_2 {handshake.bb = 2 : ui32, handshake.name = "merge1"} : <i32>
    %25:3 = fork [3] %24 {handshake.bb = 2 : ui32, handshake.name = "fork5"} : <i32>
    %26 = trunci %27 {handshake.bb = 2 : ui32, handshake.name = "trunci0"} : <i32> to <i10>
    %27 = buffer %25#0, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 2 : ui32, handshake.name = "buffer9"} : <i32>
    %28 = trunci %25#1 {handshake.bb = 2 : ui32, handshake.name = "trunci1"} : <i32> to <i10>
    %result_6, %index_7 = control_merge [%trueResult_4]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>] to <>, <i1>
    sink %index_7 {handshake.name = "sink2"} : <i1>
    %30:2 = fork [2] %result_6 {handshake.bb = 2 : ui32, handshake.name = "fork6"} : <>
    %31 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %32 = constant %31 {handshake.bb = 2 : ui32, handshake.name = "constant3", value = 1 : i2} : <>, <i2>
    %33 = extsi %32 {handshake.bb = 2 : ui32, handshake.name = "extsi1"} : <i2> to <i32>
    %addressResult, %dataResult = load[%28] %outputs_0 {handshake.bb = 2 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i10>, <i32>, <i10>, <i32>
    %34:2 = fork [2] %dataResult {handshake.bb = 2 : ui32, handshake.name = "fork7"} : <i32>
    %35 = trunci %34#0 {handshake.bb = 2 : ui32, handshake.name = "trunci2"} : <i32> to <i10>
    %37 = trunci %34#1 {handshake.bb = 2 : ui32, handshake.name = "trunci3"} : <i32> to <i10>
    %addressResult_8, %dataResult_9 = load[%26] %outputs {handshake.bb = 2 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i10>, <f32>, <i10>, <f32>
    %addressResult_10, %dataResult_11 = load[%37] %1#0 {handshake.bb = 2 : ui32, handshake.deps = #handshake<deps[["store0", 0]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 0>, handshake.name = "load2"} : <i10>, <f32>, <i10>, <f32>
    %39 = addf %dataResult_11, %dataResult_9 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "addf0"} : <f32>
    %addressResult_12, %dataResult_13 = store[%35] %39 {handshake.bb = 2 : ui32, handshake.deps = #handshake<deps[["load2", 0], ["store0", 0]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 0>, handshake.name = "store0"} : <i10>, <f32>, <i10>, <f32>
    %40 = addi %41, %33 {handshake.bb = 2 : ui32, handshake.name = "addi0"} : <i32>
    %41 = buffer %25#2, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer13"} : <i32>
    %42 = br %40 {handshake.bb = 2 : ui32, handshake.name = "br5"} : <i32>
    %43 = br %23 {handshake.bb = 2 : ui32, handshake.name = "br6"} : <i32>
    %44 = br %30#1 {handshake.bb = 2 : ui32, handshake.name = "br7"} : <>
    %result_14, %index_15 = control_merge [%falseResult_5]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>] to <>, <i1>
    sink %index_15 {handshake.name = "sink3"} : <i1>
    %45:3 = fork [3] %result_14 {handshake.bb = 3 : ui32, handshake.name = "fork8"} : <>
    end {handshake.bb = 3 : ui32, handshake.name = "end0"} %memEnd_1, %memEnd, %1#1, %0#1 : <>, <>, <>, <>
  }
}

