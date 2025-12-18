module {
  handshake.func @histogram(%arg0: memref<1000xi32>, %arg1: memref<1000xf32>, %arg2: memref<1000xf32>, %arg3: !handshake.channel<i32>, %arg4: !handshake.control<>, %arg5: !handshake.control<>, %arg6: !handshake.control<>, %arg7: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["feature", "weight", "hist", "n", "feature_start", "weight_start", "hist_start", "start"], cfg.edges = "[0,1][2,1][1,2,3,cmpi0]", resNames = ["feature_end", "weight_end", "hist_end", "end"]} {
    %0:6 = fork [6] %arg7 {handshake.bb = 0 : ui32, handshake.name = "fork0"} : <>
    %outputs:2, %memEnd = mem_controller[%arg2 : memref<1000xf32>] %arg6 (%45, %addressResult_20, %addressResult_22, %dataResult_23) %75#2 {connectedBlocks = [2 : i32], handshake.name = "mem_controller2"} :    (!handshake.channel<i32>, !handshake.channel<i10>, !handshake.channel<i10>, !handshake.channel<f32>) -> (!handshake.channel<f32>, !handshake.control<>)
    %outputs_0, %memEnd_1 = mem_controller[%arg1 : memref<1000xf32>] %arg5 (%addressResult_16) %75#1 {connectedBlocks = [2 : i32], handshake.name = "mem_controller3"} :    (!handshake.channel<i10>) -> !handshake.channel<f32>
    %outputs_2, %memEnd_3 = mem_controller[%arg0 : memref<1000xi32>] %arg4 (%addressResult) %75#0 {connectedBlocks = [2 : i32], handshake.name = "mem_controller4"} :    (!handshake.channel<i10>) -> !handshake.channel<i32>
    %1 = constant %0#1 {handshake.bb = 0 : ui32, handshake.name = "constant4", value = 1000 : i11} : <>, <i11>
    %2 = extsi %1 {handshake.bb = 1 : ui32, handshake.name = "extsi0"} : <i11> to <i32>
    %3 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant5", value = false} : <>, <i1>
    %4 = br %3 {handshake.bb = 0 : ui32, handshake.name = "br2"} : <i1>
    %5 = extsi %4 {handshake.bb = 0 : ui32, handshake.name = "extsi4"} : <i1> to <i32>
    %6 = br %arg3 {handshake.bb = 0 : ui32, handshake.name = "br3"} : <i32>
    %7 = br %0#5 {handshake.bb = 0 : ui32, handshake.name = "br4"} : <>
    %8 = mux %16#0 [%0#4, %69] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux2"} : <i1>, [<>, <>] to <>
    %10 = mux %16#1 [%2, %65] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux3"} : <i1>, [<i32>, <i32>] to <i32>
    %12 = mux %13 [%0#3, %68#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux4"} : <i1>, [<>, <>] to <>
    %13 = buffer %16#2, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer4"} : <i1>
    %14 = init %27#6 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "init0"} : <i1>
    %16:3 = fork [3] %14 {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <i1>
    %17 = mux %23#0 [%5, %72] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i32>, <i32>] to <i32>
    %19:2 = fork [2] %17 {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <i32>
    %20 = mux %23#1 [%6, %73] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i1>, [<i32>, <i32>] to <i32>
    %22:2 = fork [2] %20 {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <i32>
    %result, %index = control_merge [%7, %74]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %23:2 = fork [2] %index {handshake.bb = 1 : ui32, handshake.name = "fork4"} : <i1>
    %24 = cmpi slt, %19#1, %22#1 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i32>
    %27:7 = fork [7] %24 {handshake.bb = 1 : ui32, handshake.name = "fork5"} : <i1>
    %trueResult, %falseResult = cond_br %27#5, %22#0 {handshake.bb = 1 : ui32, handshake.name = "cond_br1"} : <i1>, <i32>
    sink %falseResult {handshake.name = "sink0"} : <i32>
    %trueResult_4, %falseResult_5 = cond_br %27#4, %19#0 {handshake.bb = 1 : ui32, handshake.name = "cond_br2"} : <i1>, <i32>
    sink %falseResult_5 {handshake.name = "sink1"} : <i32>
    %trueResult_6, %falseResult_7 = cond_br %27#3, %result {handshake.bb = 1 : ui32, handshake.name = "cond_br3"} : <i1>, <>
    %trueResult_8, %falseResult_9 = cond_br %33, %12 {handshake.bb = 2 : ui32, handshake.name = "cond_br7"} : <i1>, <>
    %33 = buffer %27#2, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer15"} : <i1>
    sink %falseResult_9 {handshake.name = "sink2"} : <>
    %trueResult_10, %falseResult_11 = cond_br %27#1, %8 {handshake.bb = 2 : ui32, handshake.name = "cond_br8"} : <i1>, <>
    sink %falseResult_11 {handshake.name = "sink3"} : <>
    %trueResult_12, %falseResult_13 = cond_br %27#0, %10 {handshake.bb = 2 : ui32, handshake.name = "cond_br9"} : <i1>, <i32>
    sink %falseResult_13 {handshake.name = "sink4"} : <i32>
    %36 = merge %trueResult {handshake.bb = 2 : ui32, handshake.name = "merge0"} : <i32>
    %37 = merge %trueResult_4 {handshake.bb = 2 : ui32, handshake.name = "merge1"} : <i32>
    %38:3 = fork [3] %37 {handshake.bb = 2 : ui32, handshake.name = "fork6"} : <i32>
    %39 = trunci %38#0 {handshake.bb = 2 : ui32, handshake.name = "trunci0"} : <i32> to <i10>
    %41 = trunci %38#1 {handshake.bb = 2 : ui32, handshake.name = "trunci1"} : <i32> to <i10>
    %result_14, %index_15 = control_merge [%trueResult_6]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>] to <>, <i1>
    sink %index_15 {handshake.name = "sink5"} : <i1>
    %43:2 = fork [2] %result_14 {handshake.bb = 2 : ui32, handshake.name = "fork7"} : <>
    %44 = constant %43#0 {handshake.bb = 2 : ui32, handshake.name = "constant6", value = 1 : i2} : <>, <i2>
    %45 = extsi %44 {handshake.bb = 2 : ui32, handshake.name = "extsi2"} : <i2> to <i32>
    %46 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %47 = constant %46 {handshake.bb = 2 : ui32, handshake.name = "constant7", value = 1 : i2} : <>, <i2>
    %48 = extsi %47 {handshake.bb = 2 : ui32, handshake.name = "extsi3"} : <i2> to <i32>
    %addressResult, %dataResult = load[%41] %outputs_2 {handshake.bb = 2 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i10>, <i32>, <i10>, <i32>
    %49:4 = fork [4] %dataResult {handshake.bb = 2 : ui32, handshake.name = "fork8"} : <i32>
    %50 = trunci %51 {handshake.bb = 2 : ui32, handshake.name = "trunci2"} : <i32> to <i10>
    %51 = buffer %49#0, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 2 : ui32, handshake.name = "buffer20"} : <i32>
    %addressResult_16, %dataResult_17 = load[%39] %outputs_0 {handshake.bb = 2 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i10>, <f32>, <i10>, <f32>
    %52 = gate %49#1, %trueResult_10 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "gate0"} : <i32>, !handshake.control<> to <i32>
    %54 = cmpi ne, %52, %trueResult_12 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi1"} : <i32>
    %55:2 = fork [2] %54 {handshake.bb = 2 : ui32, handshake.name = "fork9"} : <i1>
    %trueResult_18, %falseResult_19 = cond_br %56, %trueResult_8 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br6"} : <i1>, <>
    %56 = buffer %55#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer22"} : <i1>
    sink %trueResult_18 {handshake.name = "sink6"} : <>
    %57 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %58 = mux %55#0 [%falseResult_19, %57] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux5"} : <i1>, [<>, <>] to <>
    %60 = join %58 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "join0"} : <>
    %61 = gate %49#2, %60 {handshake.bb = 2 : ui32, handshake.name = "gate1"} : <i32>, !handshake.control<> to <i32>
    %63 = trunci %61 {handshake.bb = 2 : ui32, handshake.name = "trunci3"} : <i32> to <i10>
    %addressResult_20, %dataResult_21 = load[%63] %outputs#0 {handshake.bb = 2 : ui32, handshake.deps = #handshake<deps[["store1", 0, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load5"} : <i10>, <f32>, <i10>, <f32>
    %64 = addf %dataResult_21, %dataResult_17 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "addf0"} : <f32>
    %65 = buffer %49#3, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer0"} : <i32>
    %67 = buffer %doneResult, bufferType = FIFO_BREAK_NONE, numSlots = 1 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "buffer1"} : <>
    %68:2 = fork [2] %67 {handshake.bb = 2 : ui32, handshake.name = "fork10"} : <>
    %69 = init %68#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init3"} : <>
    %addressResult_22, %dataResult_23, %doneResult = store[%50] %64 %outputs#1 {handshake.bb = 2 : ui32, handshake.deps = #handshake<deps[["load5", 0, false], ["store1", 0, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store1"} : <i10>, <f32>, <>, <i10>, <f32>, <>
    %70 = addi %38#2, %48 {handshake.bb = 2 : ui32, handshake.name = "addi0"} : <i32>
    %72 = br %70 {handshake.bb = 2 : ui32, handshake.name = "br5"} : <i32>
    %73 = br %36 {handshake.bb = 2 : ui32, handshake.name = "br6"} : <i32>
    %74 = br %43#1 {handshake.bb = 2 : ui32, handshake.name = "br7"} : <>
    %result_24, %index_25 = control_merge [%falseResult_7]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>] to <>, <i1>
    sink %index_25 {handshake.name = "sink7"} : <i1>
    %75:3 = fork [3] %result_24 {handshake.bb = 3 : ui32, handshake.name = "fork11"} : <>
    end {handshake.bb = 3 : ui32, handshake.name = "end0"} %memEnd_3, %memEnd_1, %memEnd, %0#2 : <>, <>, <>, <>
  }
}

