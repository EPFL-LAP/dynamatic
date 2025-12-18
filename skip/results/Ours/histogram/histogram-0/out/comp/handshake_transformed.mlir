module {
  handshake.func @histogram(%arg0: memref<1000xi32>, %arg1: memref<1000xf32>, %arg2: memref<1000xf32>, %arg3: !handshake.channel<i32>, %arg4: !handshake.control<>, %arg5: !handshake.control<>, %arg6: !handshake.control<>, %arg7: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["feature", "weight", "hist", "n", "feature_start", "weight_start", "hist_start", "start"], cfg.edges = "[0,1][2,1][1,2,3,cmpi0]", resNames = ["feature_end", "weight_end", "hist_end", "end"]} {
    %0:4 = fork [4] %arg7 {handshake.bb = 0 : ui32, handshake.name = "fork0"} : <>
    %outputs:2, %memEnd = mem_controller[%arg2 : memref<1000xf32>] %arg6 (%35, %addressResult_14, %addressResult_16, %dataResult_17) %52#2 {connectedBlocks = [2 : i32], handshake.name = "mem_controller2"} :    (!handshake.channel<i32>, !handshake.channel<i10>, !handshake.channel<i10>, !handshake.channel<f32>) -> (!handshake.channel<f32>, !handshake.control<>)
    %outputs_0, %memEnd_1 = mem_controller[%arg1 : memref<1000xf32>] %arg5 (%addressResult_12) %52#1 {connectedBlocks = [2 : i32], handshake.name = "mem_controller3"} :    (!handshake.channel<i10>) -> !handshake.channel<f32>
    %outputs_2, %memEnd_3 = mem_controller[%arg0 : memref<1000xi32>] %arg4 (%addressResult) %52#0 {connectedBlocks = [2 : i32], handshake.name = "mem_controller4"} :    (!handshake.channel<i10>) -> !handshake.channel<i32>
    %1 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant3", value = false} : <>, <i1>
    %2 = br %1 {handshake.bb = 0 : ui32, handshake.name = "br2"} : <i1>
    %3 = extsi %2 {handshake.bb = 0 : ui32, handshake.name = "extsi3"} : <i1> to <i32>
    %4 = br %arg3 {handshake.bb = 0 : ui32, handshake.name = "br3"} : <i32>
    %5 = br %0#3 {handshake.bb = 0 : ui32, handshake.name = "br4"} : <>
    %6 = mux %7 [%0#2, %46] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux2"} : <i1>, [<>, <>] to <>
    %7 = init %19#4 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "init0"} : <i1>
    %9 = mux %15#0 [%3, %49] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i32>, <i32>] to <i32>
    %11:2 = fork [2] %9 {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <i32>
    %12 = mux %15#1 [%4, %50] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i1>, [<i32>, <i32>] to <i32>
    %14:2 = fork [2] %12 {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <i32>
    %result, %index = control_merge [%5, %51]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %15:2 = fork [2] %index {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <i1>
    %16 = cmpi slt, %11#1, %14#1 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i32>
    %19:5 = fork [5] %16 {handshake.bb = 1 : ui32, handshake.name = "fork4"} : <i1>
    %trueResult, %falseResult = cond_br %19#3, %14#0 {handshake.bb = 1 : ui32, handshake.name = "cond_br1"} : <i1>, <i32>
    sink %falseResult {handshake.name = "sink0"} : <i32>
    %trueResult_4, %falseResult_5 = cond_br %19#2, %11#0 {handshake.bb = 1 : ui32, handshake.name = "cond_br2"} : <i1>, <i32>
    sink %falseResult_5 {handshake.name = "sink1"} : <i32>
    %trueResult_6, %falseResult_7 = cond_br %19#1, %result {handshake.bb = 1 : ui32, handshake.name = "cond_br3"} : <i1>, <>
    %trueResult_8, %falseResult_9 = cond_br %19#0, %6 {handshake.bb = 2 : ui32, handshake.name = "cond_br4"} : <i1>, <>
    sink %falseResult_9 {handshake.name = "sink2"} : <>
    %26 = merge %trueResult {handshake.bb = 2 : ui32, handshake.name = "merge0"} : <i32>
    %27 = merge %trueResult_4 {handshake.bb = 2 : ui32, handshake.name = "merge1"} : <i32>
    %28:3 = fork [3] %27 {handshake.bb = 2 : ui32, handshake.name = "fork5"} : <i32>
    %29 = trunci %28#0 {handshake.bb = 2 : ui32, handshake.name = "trunci0"} : <i32> to <i10>
    %31 = trunci %28#1 {handshake.bb = 2 : ui32, handshake.name = "trunci1"} : <i32> to <i10>
    %result_10, %index_11 = control_merge [%trueResult_6]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>] to <>, <i1>
    sink %index_11 {handshake.name = "sink3"} : <i1>
    %33:2 = fork [2] %result_10 {handshake.bb = 2 : ui32, handshake.name = "fork6"} : <>
    %34 = constant %33#0 {handshake.bb = 2 : ui32, handshake.name = "constant4", value = 1 : i2} : <>, <i2>
    %35 = extsi %34 {handshake.bb = 2 : ui32, handshake.name = "extsi1"} : <i2> to <i32>
    %36 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %37 = constant %36 {handshake.bb = 2 : ui32, handshake.name = "constant5", value = 1 : i2} : <>, <i2>
    %38 = extsi %37 {handshake.bb = 2 : ui32, handshake.name = "extsi2"} : <i2> to <i32>
    %addressResult, %dataResult = load[%31] %outputs_2 {handshake.bb = 2 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i10>, <i32>, <i10>, <i32>
    %39:2 = fork [2] %dataResult {handshake.bb = 2 : ui32, handshake.name = "fork7"} : <i32>
    %40 = trunci %41 {handshake.bb = 2 : ui32, handshake.name = "trunci2"} : <i32> to <i10>
    %41 = buffer %39#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer14"} : <i32>
    %addressResult_12, %dataResult_13 = load[%29] %outputs_0 {handshake.bb = 2 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i10>, <f32>, <i10>, <f32>
    %42 = gate %39#1, %trueResult_8 {handshake.bb = 2 : ui32, handshake.name = "gate0"} : <i32>, !handshake.control<> to <i32>
    %44 = trunci %42 {handshake.bb = 2 : ui32, handshake.name = "trunci3"} : <i32> to <i10>
    %addressResult_14, %dataResult_15 = load[%44] %outputs#0 {handshake.bb = 2 : ui32, handshake.deps = #handshake<deps[["store1", 0, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load5"} : <i10>, <f32>, <i10>, <f32>
    %45 = addf %dataResult_15, %dataResult_13 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "addf0"} : <f32>
    %46 = buffer %doneResult, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer0"} : <>
    %addressResult_16, %dataResult_17, %doneResult = store[%40] %45 %outputs#1 {handshake.bb = 2 : ui32, handshake.deps = #handshake<deps[["load5", 0, false], ["store1", 0, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store1"} : <i10>, <f32>, <>, <i10>, <f32>, <>
    %47 = addi %28#2, %38 {handshake.bb = 2 : ui32, handshake.name = "addi0"} : <i32>
    %49 = br %47 {handshake.bb = 2 : ui32, handshake.name = "br5"} : <i32>
    %50 = br %26 {handshake.bb = 2 : ui32, handshake.name = "br6"} : <i32>
    %51 = br %33#1 {handshake.bb = 2 : ui32, handshake.name = "br7"} : <>
    %result_18, %index_19 = control_merge [%falseResult_7]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>] to <>, <i1>
    sink %index_19 {handshake.name = "sink4"} : <i1>
    %52:3 = fork [3] %result_18 {handshake.bb = 3 : ui32, handshake.name = "fork8"} : <>
    end {handshake.bb = 3 : ui32, handshake.name = "end0"} %memEnd_3, %memEnd_1, %memEnd, %0#1 : <>, <>, <>, <>
  }
}

