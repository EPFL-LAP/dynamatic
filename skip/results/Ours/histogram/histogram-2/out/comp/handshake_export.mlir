module {
  handshake.func @histogram(%arg0: memref<1000xi32>, %arg1: memref<1000xf32>, %arg2: memref<1000xf32>, %arg3: !handshake.channel<i32>, %arg4: !handshake.control<>, %arg5: !handshake.control<>, %arg6: !handshake.control<>, %arg7: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["feature", "weight", "hist", "n", "feature_start", "weight_start", "hist_start", "start"], cfg.edges = "[0,1][2,1][1,2,3,cmpi0]", resNames = ["feature_end", "weight_end", "hist_end", "end"]} {
    %0:7 = fork [7] %arg7 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork0"} : <>
    %outputs:2, %memEnd = mem_controller[%arg2 : memref<1000xf32>] %arg6 (%44, %addressResult_24, %addressResult_26, %dataResult_27) %79#2 {connectedBlocks = [2 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "5": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller2"} :    (!handshake.channel<i32>, !handshake.channel<i10>, !handshake.channel<i10>, !handshake.channel<f32>) -> (!handshake.channel<f32>, !handshake.control<>)
    %outputs_0, %memEnd_1 = mem_controller[%arg1 : memref<1000xf32>] %arg5 (%addressResult_18) %79#1 {connectedBlocks = [2 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller3"} :    (!handshake.channel<i10>) -> !handshake.channel<f32>
    %outputs_2, %memEnd_3 = mem_controller[%arg0 : memref<1000xi32>] %arg4 (%addressResult) %79#0 {connectedBlocks = [2 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller4"} :    (!handshake.channel<i10>) -> !handshake.channel<i32>
    %1 = constant %0#1 {handshake.bb = 0 : ui32, handshake.name = "constant5", value = 1000 : i11} : <>, <i11>
    %2:2 = fork [2] %1 {handshake.bb = 0 : ui32, handshake.name = "fork1"} : <i11>
    %3 = extsi %2#0 {handshake.bb = 1 : ui32, handshake.name = "extsi0"} : <i11> to <i32>
    %4 = extsi %2#1 {handshake.bb = 1 : ui32, handshake.name = "extsi1"} : <i11> to <i32>
    %5 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant6", value = false} : <>, <i1>
    %6 = extsi %5 {handshake.bb = 0 : ui32, handshake.name = "extsi5"} : <i1> to <i32>
    %7 = mux %16#0 [%0#5, %77] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux2"} : <i1>, [<>, <>] to <>
    %8 = mux %16#1 [%3, %71] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux3"} : <i1>, [<i32>, <i32>] to <i32>
    %9 = mux %16#2 [%4, %70#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux4"} : <i1>, [<i32>, <i32>] to <i32>
    %10 = mux %11 [%0#4, %74#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux5"} : <i1>, [<>, <>] to <>
    %11 = buffer %16#3, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 1 : ui32, handshake.name = "buffer8"} : <i1>
    %12 = mux %13 [%0#3, %76#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux6"} : <i1>, [<>, <>] to <>
    %13 = buffer %16#4, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer9"} : <i1>
    %14 = init %15 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "init0"} : <i1>
    %15 = buffer %30#8, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer10"} : <i1>
    %16:5 = fork [5] %14 {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <i1>
    %17 = buffer %78, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer18"} : <i32>
    %18 = mux %27#0 [%6, %17] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i32>, <i32>] to <i32>
    %19 = buffer %18, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer11"} : <i32>
    %20 = buffer %19, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer12"} : <i32>
    %21:2 = fork [2] %20 {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <i32>
    %22 = mux %27#1 [%arg3, %trueResult] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i1>, [<i32>, <i32>] to <i32>
    %23 = buffer %22, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer13"} : <i32>
    %24 = buffer %23, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer14"} : <i32>
    %25:2 = fork [2] %24 {handshake.bb = 1 : ui32, handshake.name = "fork4"} : <i32>
    %26 = buffer %42#1, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer19"} : <>
    %result, %index = control_merge [%0#6, %26]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %27:2 = fork [2] %index {handshake.bb = 1 : ui32, handshake.name = "fork5"} : <i1>
    %28 = cmpi slt, %21#1, %25#1 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i32>
    %29 = buffer %28, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer15"} : <i1>
    %30:9 = fork [9] %29 {handshake.bb = 1 : ui32, handshake.name = "fork6"} : <i1>
    %trueResult, %falseResult = cond_br %30#7, %25#0 {handshake.bb = 1 : ui32, handshake.name = "cond_br1"} : <i1>, <i32>
    sink %falseResult {handshake.name = "sink0"} : <i32>
    %trueResult_4, %falseResult_5 = cond_br %30#6, %21#0 {handshake.bb = 1 : ui32, handshake.name = "cond_br2"} : <i1>, <i32>
    sink %falseResult_5 {handshake.name = "sink1"} : <i32>
    %trueResult_6, %falseResult_7 = cond_br %30#5, %result {handshake.bb = 1 : ui32, handshake.name = "cond_br3"} : <i1>, <>
    %31 = buffer %10, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer6"} : <>
    %trueResult_8, %falseResult_9 = cond_br %32, %31 {handshake.bb = 2 : ui32, handshake.name = "cond_br10"} : <i1>, <>
    %32 = buffer %30#4, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 2 : ui32, handshake.name = "buffer20"} : <i1>
    sink %falseResult_9 {handshake.name = "sink2"} : <>
    %33 = buffer %7, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer2"} : <>
    %34 = buffer %33, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer3"} : <>
    %trueResult_10, %falseResult_11 = cond_br %30#3, %34 {handshake.bb = 2 : ui32, handshake.name = "cond_br11"} : <i1>, <>
    sink %falseResult_11 {handshake.name = "sink3"} : <>
    %35 = buffer %8, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer4"} : <i32>
    %trueResult_12, %falseResult_13 = cond_br %30#2, %35 {handshake.bb = 2 : ui32, handshake.name = "cond_br12"} : <i1>, <i32>
    sink %falseResult_13 {handshake.name = "sink4"} : <i32>
    %36 = buffer %9, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer5"} : <i32>
    %trueResult_14, %falseResult_15 = cond_br %30#1, %36 {handshake.bb = 2 : ui32, handshake.name = "cond_br13"} : <i1>, <i32>
    sink %falseResult_15 {handshake.name = "sink5"} : <i32>
    %37 = buffer %12, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer7"} : <>
    %trueResult_16, %falseResult_17 = cond_br %38, %37 {handshake.bb = 2 : ui32, handshake.name = "cond_br14"} : <i1>, <>
    %38 = buffer %30#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer24"} : <i1>
    sink %falseResult_17 {handshake.name = "sink6"} : <>
    %39:3 = fork [3] %trueResult_4 {handshake.bb = 2 : ui32, handshake.name = "fork7"} : <i32>
    %40 = trunci %39#0 {handshake.bb = 2 : ui32, handshake.name = "trunci0"} : <i32> to <i10>
    %41 = trunci %39#1 {handshake.bb = 2 : ui32, handshake.name = "trunci1"} : <i32> to <i10>
    %42:2 = fork [2] %trueResult_6 {handshake.bb = 2 : ui32, handshake.name = "fork8"} : <>
    %43 = constant %42#0 {handshake.bb = 2 : ui32, handshake.name = "constant7", value = 1 : i2} : <>, <i2>
    %44 = extsi %43 {handshake.bb = 2 : ui32, handshake.name = "extsi3"} : <i2> to <i32>
    %45 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %46 = constant %45 {handshake.bb = 2 : ui32, handshake.name = "constant8", value = 1 : i2} : <>, <i2>
    %47 = extsi %46 {handshake.bb = 2 : ui32, handshake.name = "extsi4"} : <i2> to <i32>
    %addressResult, %dataResult = load[%41] %outputs_2 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i10>, <i32>, <i10>, <i32>
    %48:4 = fork [4] %dataResult {handshake.bb = 2 : ui32, handshake.name = "fork9"} : <i32>
    %49 = trunci %50 {handshake.bb = 2 : ui32, handshake.name = "trunci2"} : <i32> to <i10>
    %50 = buffer %48#0, bufferType = FIFO_BREAK_NONE, numSlots = 3 {handshake.bb = 2 : ui32, handshake.name = "buffer27"} : <i32>
    %addressResult_18, %dataResult_19 = load[%40] %outputs_0 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i10>, <f32>, <i10>, <f32>
    %51 = gate %48#1, %trueResult_10 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "gate0"} : <i32>, !handshake.control<> to <i32>
    %52:2 = fork [2] %51 {handshake.bb = 2 : ui32, handshake.name = "fork10"} : <i32>
    %53 = cmpi ne, %52#1, %trueResult_14 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi1"} : <i32>
    %54:2 = fork [2] %53 {handshake.bb = 2 : ui32, handshake.name = "fork11"} : <i1>
    %55 = cmpi ne, %52#0, %trueResult_12 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi2"} : <i32>
    %56:2 = fork [2] %55 {handshake.bb = 2 : ui32, handshake.name = "fork12"} : <i1>
    %trueResult_20, %falseResult_21 = cond_br %57, %trueResult_8 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br8"} : <i1>, <>
    %57 = buffer %54#1, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 2 : ui32, handshake.name = "buffer31"} : <i1>
    sink %trueResult_20 {handshake.name = "sink8"} : <>
    %trueResult_22, %falseResult_23 = cond_br %58, %trueResult_16 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br9"} : <i1>, <>
    %58 = buffer %56#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer32"} : <i1>
    sink %trueResult_22 {handshake.name = "sink9"} : <>
    %59 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %60 = mux %54#0 [%falseResult_21, %59] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux7"} : <i1>, [<>, <>] to <>
    %61 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source2"} : <>
    %62 = mux %56#0 [%falseResult_23, %61] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux8"} : <i1>, [<>, <>] to <>
    %63 = buffer %60, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer16"} : <>
    %64 = buffer %62, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer17"} : <>
    %65 = join %63, %64 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "join0"} : <>
    %66 = gate %48#2, %65 {handshake.bb = 2 : ui32, handshake.name = "gate1"} : <i32>, !handshake.control<> to <i32>
    %67 = trunci %66 {handshake.bb = 2 : ui32, handshake.name = "trunci3"} : <i32> to <i10>
    %addressResult_24, %dataResult_25 = load[%67] %outputs#0 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["store1", 0, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load5"} : <i10>, <f32>, <i10>, <f32>
    %68 = addf %dataResult_25, %dataResult_19 {fastmath = #arith.fastmath<none>, fpu_impl = #handshake<fpu_impl flopoco>, handshake.bb = 2 : ui32, handshake.name = "addf0", internal_delay = "2_922000"} : <f32>
    %69 = buffer %48#3, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer0"} : <i32>
    %70:2 = fork [2] %69 {handshake.bb = 2 : ui32, handshake.name = "fork13"} : <i32>
    %71 = init %72 {handshake.bb = 2 : ui32, handshake.name = "init5"} : <i32>
    %72 = buffer %70#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer37"} : <i32>
    %73 = buffer %doneResult, bufferType = FIFO_BREAK_NONE, numSlots = 1 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "buffer1"} : <>
    %74:2 = fork [2] %73 {handshake.bb = 2 : ui32, handshake.name = "fork14"} : <>
    %75 = init %74#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init6"} : <>
    %76:2 = fork [2] %75 {handshake.bb = 2 : ui32, handshake.name = "fork15"} : <>
    %77 = init %76#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init7"} : <>
    %addressResult_26, %dataResult_27, %doneResult = store[%49] %68 %outputs#1 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["load5", 0, false], ["store1", 0, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store1"} : <i10>, <f32>, <>, <i10>, <f32>, <>
    %78 = addi %39#2, %47 {handshake.bb = 2 : ui32, handshake.name = "addi0"} : <i32>
    %79:3 = fork [3] %falseResult_7 {handshake.bb = 3 : ui32, handshake.name = "fork16"} : <>
    end {handshake.bb = 3 : ui32, handshake.name = "end0"} %memEnd_3, %memEnd_1, %memEnd, %0#2 : <>, <>, <>, <>
  }
}

