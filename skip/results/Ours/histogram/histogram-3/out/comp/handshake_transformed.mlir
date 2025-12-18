module {
  handshake.func @histogram(%arg0: memref<1000xi32>, %arg1: memref<1000xf32>, %arg2: memref<1000xf32>, %arg3: !handshake.channel<i32>, %arg4: !handshake.control<>, %arg5: !handshake.control<>, %arg6: !handshake.control<>, %arg7: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["feature", "weight", "hist", "n", "feature_start", "weight_start", "hist_start", "start"], cfg.edges = "[0,1][2,1][1,2,3,cmpi0]", resNames = ["feature_end", "weight_end", "hist_end", "end"]} {
    %0:8 = fork [8] %arg7 {handshake.bb = 0 : ui32, handshake.name = "fork0"} : <>
    %outputs:2, %memEnd = mem_controller[%arg2 : memref<1000xf32>] %arg6 (%65, %addressResult_32, %addressResult_34, %dataResult_35) %121#2 {connectedBlocks = [2 : i32], handshake.name = "mem_controller2"} :    (!handshake.channel<i32>, !handshake.channel<i10>, !handshake.channel<i10>, !handshake.channel<f32>) -> (!handshake.channel<f32>, !handshake.control<>)
    %outputs_0, %memEnd_1 = mem_controller[%arg1 : memref<1000xf32>] %arg5 (%addressResult_24) %121#1 {connectedBlocks = [2 : i32], handshake.name = "mem_controller3"} :    (!handshake.channel<i10>) -> !handshake.channel<f32>
    %outputs_2, %memEnd_3 = mem_controller[%arg0 : memref<1000xi32>] %arg4 (%addressResult) %121#0 {connectedBlocks = [2 : i32], handshake.name = "mem_controller4"} :    (!handshake.channel<i10>) -> !handshake.channel<i32>
    %1 = constant %0#1 {handshake.bb = 0 : ui32, handshake.name = "constant6", value = 1000 : i11} : <>, <i11>
    %2:3 = fork [3] %1 {handshake.bb = 0 : ui32, handshake.name = "fork1"} : <i11>
    %3 = extsi %2#0 {handshake.bb = 1 : ui32, handshake.name = "extsi0"} : <i11> to <i32>
    %5 = extsi %2#1 {handshake.bb = 1 : ui32, handshake.name = "extsi1"} : <i11> to <i32>
    %7 = extsi %2#2 {handshake.bb = 1 : ui32, handshake.name = "extsi2"} : <i11> to <i32>
    %9 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant7", value = false} : <>, <i1>
    %10 = br %9 {handshake.bb = 0 : ui32, handshake.name = "br2"} : <i1>
    %11 = extsi %10 {handshake.bb = 0 : ui32, handshake.name = "extsi6"} : <i1> to <i32>
    %12 = br %arg3 {handshake.bb = 0 : ui32, handshake.name = "br3"} : <i32>
    %13 = br %0#7 {handshake.bb = 0 : ui32, handshake.name = "br4"} : <>
    %14 = mux %32#0 [%0#6, %115] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux2"} : <i1>, [<>, <>] to <>
    %16 = mux %32#1 [%3, %17] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux3"} : <i1>, [<i32>, <i32>] to <i32>
    %17 = buffer %106#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer6"} : <i32>
    %19 = mux %32#2 [%5, %107] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux4"} : <i1>, [<i32>, <i32>] to <i32>
    %21 = mux %32#3 [%7, %103#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux5"} : <i1>, [<i32>, <i32>] to <i32>
    %24 = mux %25 [%0#5, %110#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux6"} : <i1>, [<>, <>] to <>
    %25 = buffer %32#4, bufferType = FIFO_BREAK_NONE, numSlots = 3 {handshake.bb = 1 : ui32, handshake.name = "buffer11"} : <i1>
    %26 = mux %27 [%0#4, %112#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux7"} : <i1>, [<>, <>] to <>
    %27 = buffer %32#5, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 1 : ui32, handshake.name = "buffer12"} : <i1>
    %28 = mux %29 [%0#3, %114#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux8"} : <i1>, [<>, <>] to <>
    %29 = buffer %32#6, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer13"} : <i1>
    %30 = init %31 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "init0"} : <i1>
    %31 = buffer %43#10, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer14"} : <i1>
    %32:7 = fork [7] %30 {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <i1>
    %33 = mux %39#0 [%11, %118] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i32>, <i32>] to <i32>
    %35:2 = fork [2] %33 {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <i32>
    %36 = mux %39#1 [%12, %119] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i1>, [<i32>, <i32>] to <i32>
    %38:2 = fork [2] %36 {handshake.bb = 1 : ui32, handshake.name = "fork4"} : <i32>
    %result, %index = control_merge [%13, %120]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %39:2 = fork [2] %index {handshake.bb = 1 : ui32, handshake.name = "fork5"} : <i1>
    %40 = cmpi slt, %35#1, %38#1 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i32>
    %43:11 = fork [11] %40 {handshake.bb = 1 : ui32, handshake.name = "fork6"} : <i1>
    %trueResult, %falseResult = cond_br %43#9, %38#0 {handshake.bb = 1 : ui32, handshake.name = "cond_br1"} : <i1>, <i32>
    sink %falseResult {handshake.name = "sink0"} : <i32>
    %trueResult_4, %falseResult_5 = cond_br %43#8, %35#0 {handshake.bb = 1 : ui32, handshake.name = "cond_br2"} : <i1>, <i32>
    sink %falseResult_5 {handshake.name = "sink1"} : <i32>
    %trueResult_6, %falseResult_7 = cond_br %43#7, %result {handshake.bb = 1 : ui32, handshake.name = "cond_br3"} : <i1>, <>
    %trueResult_8, %falseResult_9 = cond_br %49, %19 {handshake.bb = 2 : ui32, handshake.name = "cond_br13"} : <i1>, <i32>
    %49 = buffer %43#6, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer24"} : <i1>
    sink %falseResult_9 {handshake.name = "sink2"} : <i32>
    %trueResult_10, %falseResult_11 = cond_br %50, %28 {handshake.bb = 2 : ui32, handshake.name = "cond_br14"} : <i1>, <>
    %50 = buffer %43#5, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 2 : ui32, handshake.name = "buffer25"} : <i1>
    sink %falseResult_11 {handshake.name = "sink3"} : <>
    %trueResult_12, %falseResult_13 = cond_br %51, %26 {handshake.bb = 2 : ui32, handshake.name = "cond_br15"} : <i1>, <>
    %51 = buffer %43#4, bufferType = FIFO_BREAK_NONE, numSlots = 3 {handshake.bb = 2 : ui32, handshake.name = "buffer26"} : <i1>
    sink %falseResult_13 {handshake.name = "sink4"} : <>
    %trueResult_14, %falseResult_15 = cond_br %52, %16 {handshake.bb = 2 : ui32, handshake.name = "cond_br16"} : <i1>, <i32>
    %52 = buffer %43#3, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer27"} : <i1>
    sink %falseResult_15 {handshake.name = "sink5"} : <i32>
    %trueResult_16, %falseResult_17 = cond_br %53, %14 {handshake.bb = 2 : ui32, handshake.name = "cond_br17"} : <i1>, <>
    %53 = buffer %43#2, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer28"} : <i1>
    sink %falseResult_17 {handshake.name = "sink6"} : <>
    %trueResult_18, %falseResult_19 = cond_br %54, %21 {handshake.bb = 2 : ui32, handshake.name = "cond_br18"} : <i1>, <i32>
    %54 = buffer %43#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer29"} : <i1>
    sink %falseResult_19 {handshake.name = "sink7"} : <i32>
    %trueResult_20, %falseResult_21 = cond_br %55, %24 {handshake.bb = 2 : ui32, handshake.name = "cond_br19"} : <i1>, <>
    %55 = buffer %43#0, bufferType = FIFO_BREAK_NONE, numSlots = 4 {handshake.bb = 2 : ui32, handshake.name = "buffer30"} : <i1>
    sink %falseResult_21 {handshake.name = "sink8"} : <>
    %56 = merge %trueResult {handshake.bb = 2 : ui32, handshake.name = "merge0"} : <i32>
    %57 = merge %trueResult_4 {handshake.bb = 2 : ui32, handshake.name = "merge1"} : <i32>
    %58:3 = fork [3] %57 {handshake.bb = 2 : ui32, handshake.name = "fork7"} : <i32>
    %59 = trunci %58#0 {handshake.bb = 2 : ui32, handshake.name = "trunci0"} : <i32> to <i10>
    %61 = trunci %58#1 {handshake.bb = 2 : ui32, handshake.name = "trunci1"} : <i32> to <i10>
    %result_22, %index_23 = control_merge [%trueResult_6]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>] to <>, <i1>
    sink %index_23 {handshake.name = "sink9"} : <i1>
    %63:2 = fork [2] %result_22 {handshake.bb = 2 : ui32, handshake.name = "fork8"} : <>
    %64 = constant %63#0 {handshake.bb = 2 : ui32, handshake.name = "constant8", value = 1 : i2} : <>, <i2>
    %65 = extsi %64 {handshake.bb = 2 : ui32, handshake.name = "extsi4"} : <i2> to <i32>
    %66 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %67 = constant %66 {handshake.bb = 2 : ui32, handshake.name = "constant9", value = 1 : i2} : <>, <i2>
    %68 = extsi %67 {handshake.bb = 2 : ui32, handshake.name = "extsi5"} : <i2> to <i32>
    %addressResult, %dataResult = load[%61] %outputs_2 {handshake.bb = 2 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i10>, <i32>, <i10>, <i32>
    %69:4 = fork [4] %dataResult {handshake.bb = 2 : ui32, handshake.name = "fork9"} : <i32>
    %70 = trunci %71 {handshake.bb = 2 : ui32, handshake.name = "trunci2"} : <i32> to <i10>
    %71 = buffer %69#0, bufferType = FIFO_BREAK_NONE, numSlots = 4 {handshake.bb = 2 : ui32, handshake.name = "buffer33"} : <i32>
    %addressResult_24, %dataResult_25 = load[%59] %outputs_0 {handshake.bb = 2 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i10>, <f32>, <i10>, <f32>
    %72 = gate %69#1, %trueResult_16 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "gate0"} : <i32>, !handshake.control<> to <i32>
    %74:3 = fork [3] %72 {handshake.bb = 2 : ui32, handshake.name = "fork10"} : <i32>
    %75 = cmpi ne, %74#2, %trueResult_18 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi1"} : <i32>
    %77:2 = fork [2] %75 {handshake.bb = 2 : ui32, handshake.name = "fork11"} : <i1>
    %78 = cmpi ne, %74#1, %trueResult_14 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi2"} : <i32>
    %80:2 = fork [2] %78 {handshake.bb = 2 : ui32, handshake.name = "fork12"} : <i1>
    %81 = cmpi ne, %74#0, %trueResult_8 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi3"} : <i32>
    %83:2 = fork [2] %81 {handshake.bb = 2 : ui32, handshake.name = "fork13"} : <i1>
    %trueResult_26, %falseResult_27 = cond_br %84, %trueResult_20 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br10"} : <i1>, <>
    %84 = buffer %77#1, bufferType = FIFO_BREAK_NONE, numSlots = 3 {handshake.bb = 2 : ui32, handshake.name = "buffer38"} : <i1>
    sink %trueResult_26 {handshake.name = "sink10"} : <>
    %trueResult_28, %falseResult_29 = cond_br %85, %trueResult_12 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br11"} : <i1>, <>
    %85 = buffer %80#1, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 2 : ui32, handshake.name = "buffer39"} : <i1>
    sink %trueResult_28 {handshake.name = "sink11"} : <>
    %trueResult_30, %falseResult_31 = cond_br %86, %trueResult_10 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br12"} : <i1>, <>
    %86 = buffer %83#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer40"} : <i1>
    sink %trueResult_30 {handshake.name = "sink12"} : <>
    %87 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %88 = mux %77#0 [%falseResult_27, %87] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux9"} : <i1>, [<>, <>] to <>
    %90 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source2"} : <>
    %91 = mux %80#0 [%falseResult_29, %90] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux10"} : <i1>, [<>, <>] to <>
    %93 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source3"} : <>
    %94 = mux %83#0 [%falseResult_31, %93] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux11"} : <i1>, [<>, <>] to <>
    %96 = join %88, %91, %94 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "join0"} : <>
    %97 = gate %69#2, %96 {handshake.bb = 2 : ui32, handshake.name = "gate1"} : <i32>, !handshake.control<> to <i32>
    %99 = trunci %97 {handshake.bb = 2 : ui32, handshake.name = "trunci3"} : <i32> to <i10>
    %addressResult_32, %dataResult_33 = load[%99] %outputs#0 {handshake.bb = 2 : ui32, handshake.deps = #handshake<deps[["store1", 0, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load5"} : <i10>, <f32>, <i10>, <f32>
    %100 = addf %dataResult_33, %dataResult_25 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "addf0"} : <f32>
    %101 = buffer %69#3, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer0"} : <i32>
    %103:2 = fork [2] %101 {handshake.bb = 2 : ui32, handshake.name = "fork14"} : <i32>
    %104 = init %103#0 {handshake.bb = 2 : ui32, handshake.name = "init7"} : <i32>
    %106:2 = fork [2] %104 {handshake.bb = 2 : ui32, handshake.name = "fork15"} : <i32>
    %107 = init %108 {handshake.bb = 2 : ui32, handshake.name = "init8"} : <i32>
    %108 = buffer %106#0, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 2 : ui32, handshake.name = "buffer47"} : <i32>
    %109 = buffer %doneResult, bufferType = FIFO_BREAK_NONE, numSlots = 1 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "buffer1"} : <>
    %110:2 = fork [2] %109 {handshake.bb = 2 : ui32, handshake.name = "fork16"} : <>
    %111 = init %110#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init9"} : <>
    %112:2 = fork [2] %111 {handshake.bb = 2 : ui32, handshake.name = "fork17"} : <>
    %113 = init %112#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init10"} : <>
    %114:2 = fork [2] %113 {handshake.bb = 2 : ui32, handshake.name = "fork18"} : <>
    %115 = init %114#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init11"} : <>
    %addressResult_34, %dataResult_35, %doneResult = store[%70] %100 %outputs#1 {handshake.bb = 2 : ui32, handshake.deps = #handshake<deps[["load5", 0, false], ["store1", 0, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store1"} : <i10>, <f32>, <>, <i10>, <f32>, <>
    %116 = addi %58#2, %68 {handshake.bb = 2 : ui32, handshake.name = "addi0"} : <i32>
    %118 = br %116 {handshake.bb = 2 : ui32, handshake.name = "br5"} : <i32>
    %119 = br %56 {handshake.bb = 2 : ui32, handshake.name = "br6"} : <i32>
    %120 = br %63#1 {handshake.bb = 2 : ui32, handshake.name = "br7"} : <>
    %result_36, %index_37 = control_merge [%falseResult_7]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>] to <>, <i1>
    sink %index_37 {handshake.name = "sink13"} : <i1>
    %121:3 = fork [3] %result_36 {handshake.bb = 3 : ui32, handshake.name = "fork19"} : <>
    end {handshake.bb = 3 : ui32, handshake.name = "end0"} %memEnd_3, %memEnd_1, %memEnd, %0#2 : <>, <>, <>, <>
  }
}

