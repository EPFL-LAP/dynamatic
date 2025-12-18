module {
  handshake.func @histogram(%arg0: memref<1000xi32>, %arg1: memref<1000xf32>, %arg2: memref<1000xf32>, %arg3: !handshake.channel<i32>, %arg4: !handshake.control<>, %arg5: !handshake.control<>, %arg6: !handshake.control<>, %arg7: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["feature", "weight", "hist", "n", "feature_start", "weight_start", "hist_start", "start"], cfg.edges = "[0,1][2,1][1,2,3,cmpi0]", resNames = ["feature_end", "weight_end", "hist_end", "end"]} {
    %0:7 = fork [7] %arg7 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork0"} : <>
    %outputs:2, %memEnd = mem_controller[%arg2 : memref<1000xf32>] %arg6 (%49, %addressResult_26, %addressResult_28, %dataResult_29) %87#2 {connectedBlocks = [2 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "5": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller2"} :    (!handshake.channel<i32>, !handshake.channel<i10>, !handshake.channel<i10>, !handshake.channel<f32>) -> (!handshake.channel<f32>, !handshake.control<>)
    %outputs_0, %memEnd_1 = mem_controller[%arg1 : memref<1000xf32>] %arg5 (%addressResult_20) %87#1 {connectedBlocks = [2 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller3"} :    (!handshake.channel<i10>) -> !handshake.channel<f32>
    %outputs_2, %memEnd_3 = mem_controller[%arg0 : memref<1000xi32>] %arg4 (%addressResult) %87#0 {connectedBlocks = [2 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller4"} :    (!handshake.channel<i10>) -> !handshake.channel<i32>
    %1 = constant %0#1 {handshake.bb = 0 : ui32, handshake.name = "constant5", value = 1000 : i11} : <>, <i11>
    %2:2 = fork [2] %1 {handshake.bb = 0 : ui32, handshake.name = "fork1"} : <i11>
    %3 = extsi %2#0 {handshake.bb = 1 : ui32, handshake.name = "extsi0"} : <i11> to <i32>
    %4 = extsi %2#1 {handshake.bb = 1 : ui32, handshake.name = "extsi1"} : <i11> to <i32>
    %5 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant6", value = false} : <>, <i1>
    %6 = br %5 {handshake.bb = 0 : ui32, handshake.name = "br2"} : <i1>
    %7 = extsi %6 {handshake.bb = 0 : ui32, handshake.name = "extsi5"} : <i1> to <i32>
    %8 = br %arg3 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "br3"} : <i32>
    %9 = br %0#6 {handshake.bb = 0 : ui32, handshake.name = "br4"} : <>
    %10 = mux %19#0 [%0#5, %82] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux2"} : <i1>, [<>, <>] to <>
    %11 = mux %19#1 [%3, %76] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux3"} : <i1>, [<i32>, <i32>] to <i32>
    %12 = mux %19#2 [%4, %75#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux4"} : <i1>, [<i32>, <i32>] to <i32>
    %13 = mux %14 [%0#4, %79#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux5"} : <i1>, [<>, <>] to <>
    %14 = buffer %19#3, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 1 : ui32, handshake.name = "buffer8"} : <i1>
    %15 = mux %16 [%0#3, %81#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux6"} : <i1>, [<>, <>] to <>
    %16 = buffer %19#4, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer9"} : <i1>
    %17 = init %18 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "init0"} : <i1>
    %18 = buffer %33#8, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer10"} : <i1>
    %19:5 = fork [5] %17 {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <i1>
    %20 = buffer %84, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer18"} : <i32>
    %21 = mux %30#0 [%7, %20] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i32>, <i32>] to <i32>
    %22 = buffer %21, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer11"} : <i32>
    %23 = buffer %22, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer12"} : <i32>
    %24:2 = fork [2] %23 {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <i32>
    %25 = mux %30#1 [%8, %85] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i1>, [<i32>, <i32>] to <i32>
    %26 = buffer %25, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer13"} : <i32>
    %27 = buffer %26, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer14"} : <i32>
    %28:2 = fork [2] %27 {handshake.bb = 1 : ui32, handshake.name = "fork4"} : <i32>
    %29 = buffer %86, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer19"} : <>
    %result, %index = control_merge [%9, %29]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %30:2 = fork [2] %index {handshake.bb = 1 : ui32, handshake.name = "fork5"} : <i1>
    %31 = cmpi slt, %24#1, %28#1 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i32>
    %32 = buffer %31, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer15"} : <i1>
    %33:9 = fork [9] %32 {handshake.bb = 1 : ui32, handshake.name = "fork6"} : <i1>
    %trueResult, %falseResult = cond_br %33#7, %28#0 {handshake.bb = 1 : ui32, handshake.name = "cond_br1"} : <i1>, <i32>
    sink %falseResult {handshake.name = "sink0"} : <i32>
    %trueResult_4, %falseResult_5 = cond_br %33#6, %24#0 {handshake.bb = 1 : ui32, handshake.name = "cond_br2"} : <i1>, <i32>
    sink %falseResult_5 {handshake.name = "sink1"} : <i32>
    %trueResult_6, %falseResult_7 = cond_br %33#5, %result {handshake.bb = 1 : ui32, handshake.name = "cond_br3"} : <i1>, <>
    %34 = buffer %13, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer6"} : <>
    %trueResult_8, %falseResult_9 = cond_br %35, %34 {handshake.bb = 2 : ui32, handshake.name = "cond_br10"} : <i1>, <>
    %35 = buffer %33#4, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 2 : ui32, handshake.name = "buffer20"} : <i1>
    sink %falseResult_9 {handshake.name = "sink2"} : <>
    %36 = buffer %10, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer2"} : <>
    %37 = buffer %36, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer3"} : <>
    %trueResult_10, %falseResult_11 = cond_br %33#3, %37 {handshake.bb = 2 : ui32, handshake.name = "cond_br11"} : <i1>, <>
    sink %falseResult_11 {handshake.name = "sink3"} : <>
    %38 = buffer %11, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer4"} : <i32>
    %trueResult_12, %falseResult_13 = cond_br %33#2, %38 {handshake.bb = 2 : ui32, handshake.name = "cond_br12"} : <i1>, <i32>
    sink %falseResult_13 {handshake.name = "sink4"} : <i32>
    %39 = buffer %12, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer5"} : <i32>
    %trueResult_14, %falseResult_15 = cond_br %33#1, %39 {handshake.bb = 2 : ui32, handshake.name = "cond_br13"} : <i1>, <i32>
    sink %falseResult_15 {handshake.name = "sink5"} : <i32>
    %40 = buffer %15, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer7"} : <>
    %trueResult_16, %falseResult_17 = cond_br %41, %40 {handshake.bb = 2 : ui32, handshake.name = "cond_br14"} : <i1>, <>
    %41 = buffer %33#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer24"} : <i1>
    sink %falseResult_17 {handshake.name = "sink6"} : <>
    %42 = merge %trueResult {handshake.bb = 2 : ui32, handshake.name = "merge0"} : <i32>
    %43 = merge %trueResult_4 {handshake.bb = 2 : ui32, handshake.name = "merge1"} : <i32>
    %44:3 = fork [3] %43 {handshake.bb = 2 : ui32, handshake.name = "fork7"} : <i32>
    %45 = trunci %44#0 {handshake.bb = 2 : ui32, handshake.name = "trunci0"} : <i32> to <i10>
    %46 = trunci %44#1 {handshake.bb = 2 : ui32, handshake.name = "trunci1"} : <i32> to <i10>
    %result_18, %index_19 = control_merge [%trueResult_6]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>] to <>, <i1>
    sink %index_19 {handshake.name = "sink7"} : <i1>
    %47:2 = fork [2] %result_18 {handshake.bb = 2 : ui32, handshake.name = "fork8"} : <>
    %48 = constant %47#0 {handshake.bb = 2 : ui32, handshake.name = "constant7", value = 1 : i2} : <>, <i2>
    %49 = extsi %48 {handshake.bb = 2 : ui32, handshake.name = "extsi3"} : <i2> to <i32>
    %50 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %51 = constant %50 {handshake.bb = 2 : ui32, handshake.name = "constant8", value = 1 : i2} : <>, <i2>
    %52 = extsi %51 {handshake.bb = 2 : ui32, handshake.name = "extsi4"} : <i2> to <i32>
    %addressResult, %dataResult = load[%46] %outputs_2 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i10>, <i32>, <i10>, <i32>
    %53:4 = fork [4] %dataResult {handshake.bb = 2 : ui32, handshake.name = "fork9"} : <i32>
    %54 = trunci %55 {handshake.bb = 2 : ui32, handshake.name = "trunci2"} : <i32> to <i10>
    %55 = buffer %53#0, bufferType = FIFO_BREAK_NONE, numSlots = 3 {handshake.bb = 2 : ui32, handshake.name = "buffer27"} : <i32>
    %addressResult_20, %dataResult_21 = load[%45] %outputs_0 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i10>, <f32>, <i10>, <f32>
    %56 = gate %53#1, %trueResult_10 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "gate0"} : <i32>, !handshake.control<> to <i32>
    %57:2 = fork [2] %56 {handshake.bb = 2 : ui32, handshake.name = "fork10"} : <i32>
    %58 = cmpi ne, %57#1, %trueResult_14 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi1"} : <i32>
    %59:2 = fork [2] %58 {handshake.bb = 2 : ui32, handshake.name = "fork11"} : <i1>
    %60 = cmpi ne, %57#0, %trueResult_12 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi2"} : <i32>
    %61:2 = fork [2] %60 {handshake.bb = 2 : ui32, handshake.name = "fork12"} : <i1>
    %trueResult_22, %falseResult_23 = cond_br %62, %trueResult_8 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br8"} : <i1>, <>
    %62 = buffer %59#1, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 2 : ui32, handshake.name = "buffer31"} : <i1>
    sink %trueResult_22 {handshake.name = "sink8"} : <>
    %trueResult_24, %falseResult_25 = cond_br %63, %trueResult_16 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br9"} : <i1>, <>
    %63 = buffer %61#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer32"} : <i1>
    sink %trueResult_24 {handshake.name = "sink9"} : <>
    %64 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %65 = mux %59#0 [%falseResult_23, %64] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux7"} : <i1>, [<>, <>] to <>
    %66 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source2"} : <>
    %67 = mux %61#0 [%falseResult_25, %66] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux8"} : <i1>, [<>, <>] to <>
    %68 = buffer %65, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer16"} : <>
    %69 = buffer %67, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer17"} : <>
    %70 = join %68, %69 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "join0"} : <>
    %71 = gate %53#2, %70 {handshake.bb = 2 : ui32, handshake.name = "gate1"} : <i32>, !handshake.control<> to <i32>
    %72 = trunci %71 {handshake.bb = 2 : ui32, handshake.name = "trunci3"} : <i32> to <i10>
    %addressResult_26, %dataResult_27 = load[%72] %outputs#0 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["store1", 0, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load5"} : <i10>, <f32>, <i10>, <f32>
    %73 = addf %dataResult_27, %dataResult_21 {fastmath = #arith.fastmath<none>, fpu_impl = #handshake<fpu_impl flopoco>, handshake.bb = 2 : ui32, handshake.name = "addf0", internal_delay = "2_922000"} : <f32>
    %74 = buffer %53#3, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer0"} : <i32>
    %75:2 = fork [2] %74 {handshake.bb = 2 : ui32, handshake.name = "fork13"} : <i32>
    %76 = init %77 {handshake.bb = 2 : ui32, handshake.name = "init5"} : <i32>
    %77 = buffer %75#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer37"} : <i32>
    %78 = buffer %doneResult, bufferType = FIFO_BREAK_NONE, numSlots = 1 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "buffer1"} : <>
    %79:2 = fork [2] %78 {handshake.bb = 2 : ui32, handshake.name = "fork14"} : <>
    %80 = init %79#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init6"} : <>
    %81:2 = fork [2] %80 {handshake.bb = 2 : ui32, handshake.name = "fork15"} : <>
    %82 = init %81#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init7"} : <>
    %addressResult_28, %dataResult_29, %doneResult = store[%54] %73 %outputs#1 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["load5", 0, false], ["store1", 0, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store1"} : <i10>, <f32>, <>, <i10>, <f32>, <>
    %83 = addi %44#2, %52 {handshake.bb = 2 : ui32, handshake.name = "addi0"} : <i32>
    %84 = br %83 {handshake.bb = 2 : ui32, handshake.name = "br5"} : <i32>
    %85 = br %42 {handshake.bb = 2 : ui32, handshake.name = "br6"} : <i32>
    %86 = br %47#1 {handshake.bb = 2 : ui32, handshake.name = "br7"} : <>
    %result_30, %index_31 = control_merge [%falseResult_7]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>] to <>, <i1>
    sink %index_31 {handshake.name = "sink10"} : <i1>
    %87:3 = fork [3] %result_30 {handshake.bb = 3 : ui32, handshake.name = "fork16"} : <>
    end {handshake.bb = 3 : ui32, handshake.name = "end0"} %memEnd_3, %memEnd_1, %memEnd, %0#2 : <>, <>, <>, <>
  }
}

