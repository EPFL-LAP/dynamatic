module {
  handshake.func @histogram(%arg0: memref<1000xi32>, %arg1: memref<1000xf32>, %arg2: memref<1000xf32>, %arg3: !handshake.channel<i32>, %arg4: !handshake.control<>, %arg5: !handshake.control<>, %arg6: !handshake.control<>, %arg7: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["feature", "weight", "hist", "n", "feature_start", "weight_start", "hist_start", "start"], resNames = ["feature_end", "weight_end", "hist_end", "end"]} {
    %0:3 = fork [3] %arg7 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork0"} : <>
    %1:2 = lsq[%arg2 : memref<1000xf32>] (%arg6, %26#0, %addressResult_10, %addressResult_12, %dataResult_13, %44#2)  {groupSizes = [2 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "5": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "lsq1"} : (!handshake.control<>, !handshake.control<>, !handshake.channel<i10>, !handshake.channel<i10>, !handshake.channel<f32>, !handshake.control<>) -> (!handshake.channel<f32>, !handshake.control<>)
    %outputs, %memEnd = mem_controller[%arg1 : memref<1000xf32>] %arg5 (%addressResult_8) %44#1 {connectedBlocks = [2 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller2"} :    (!handshake.channel<i10>) -> !handshake.channel<f32>
    %outputs_0, %memEnd_1 = mem_controller[%arg0 : memref<1000xi32>] %arg4 (%addressResult) %44#0 {connectedBlocks = [2 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller3"} :    (!handshake.channel<i10>) -> !handshake.channel<i32>
    %2 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant1", value = false} : <>, <i1>
    %3 = br %2 {handshake.bb = 0 : ui32, handshake.name = "br2"} : <i1>
    %4 = extsi %3 {handshake.bb = 0 : ui32, handshake.name = "extsi2"} : <i1> to <i32>
    %5 = br %arg3 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "br3"} : <i32>
    %6 = br %0#2 {handshake.bb = 0 : ui32, handshake.name = "br4"} : <>
    %7 = buffer %39, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer10"} : <i32>
    %8 = mux %16#0 [%4, %7] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i32>, <i32>] to <i32>
    %9 = buffer %8, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer0"} : <i32>
    %10 = buffer %9, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer1"} : <i32>
    %11:2 = fork [2] %10 {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <i32>
    %12 = buffer %40, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer11"} : <i32>
    %13 = mux %16#1 [%5, %12] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i1>, [<i32>, <i32>] to <i32>
    %14 = buffer %13, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer2"} : <i32>
    %15:2 = fork [2] %14 {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <i32>
    %result, %index = control_merge [%6, %43]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %16:2 = fork [2] %index {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <i1>
    %17 = cmpi slt, %11#1, %15#1 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i32>
    %18 = buffer %17, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer3"} : <i1>
    %19:3 = fork [3] %18 {handshake.bb = 1 : ui32, handshake.name = "fork4"} : <i1>
    %trueResult, %falseResult = cond_br %19#2, %15#0 {handshake.bb = 1 : ui32, handshake.name = "cond_br1"} : <i1>, <i32>
    sink %falseResult {handshake.name = "sink0"} : <i32>
    %trueResult_2, %falseResult_3 = cond_br %19#1, %11#0 {handshake.bb = 1 : ui32, handshake.name = "cond_br2"} : <i1>, <i32>
    sink %falseResult_3 {handshake.name = "sink1"} : <i32>
    %trueResult_4, %falseResult_5 = cond_br %19#0, %result {handshake.bb = 1 : ui32, handshake.name = "cond_br3"} : <i1>, <>
    %20 = merge %trueResult {handshake.bb = 2 : ui32, handshake.name = "merge0"} : <i32>
    %21 = merge %trueResult_2 {handshake.bb = 2 : ui32, handshake.name = "merge1"} : <i32>
    %22:3 = fork [3] %21 {handshake.bb = 2 : ui32, handshake.name = "fork5"} : <i32>
    %23 = trunci %24 {handshake.bb = 2 : ui32, handshake.name = "trunci0"} : <i32> to <i10>
    %24 = buffer %22#0, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 2 : ui32, handshake.name = "buffer9"} : <i32>
    %25 = trunci %22#1 {handshake.bb = 2 : ui32, handshake.name = "trunci1"} : <i32> to <i10>
    %result_6, %index_7 = control_merge [%trueResult_4]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>] to <>, <i1>
    sink %index_7 {handshake.name = "sink2"} : <i1>
    %26:2 = fork [2] %result_6 {handshake.bb = 2 : ui32, handshake.name = "fork6"} : <>
    %27 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %28 = constant %27 {handshake.bb = 2 : ui32, handshake.name = "constant3", value = 1 : i2} : <>, <i2>
    %29 = extsi %28 {handshake.bb = 2 : ui32, handshake.name = "extsi1"} : <i2> to <i32>
    %addressResult, %dataResult = load[%25] %outputs_0 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i10>, <i32>, <i10>, <i32>
    %30:2 = fork [2] %dataResult {handshake.bb = 2 : ui32, handshake.name = "fork7"} : <i32>
    %31 = trunci %30#0 {handshake.bb = 2 : ui32, handshake.name = "trunci2"} : <i32> to <i10>
    %32 = trunci %30#1 {handshake.bb = 2 : ui32, handshake.name = "trunci3"} : <i32> to <i10>
    %addressResult_8, %dataResult_9 = load[%23] %outputs {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i10>, <f32>, <i10>, <f32>
    %33 = buffer %32, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer7"} : <i10>
    %addressResult_10, %dataResult_11 = load[%33] %1#0 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["store0", 0]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 0>, handshake.name = "load2"} : <i10>, <f32>, <i10>, <f32>
    %34 = addf %dataResult_11, %dataResult_9 {fastmath = #arith.fastmath<none>, fpu_impl = #handshake<fpu_impl flopoco>, handshake.bb = 2 : ui32, handshake.name = "addf0", internal_delay = "2_922000"} : <f32>
    %35 = buffer %31, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer6"} : <i10>
    %36 = buffer %34, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer8"} : <f32>
    %addressResult_12, %dataResult_13 = store[%35] %36 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["load2", 0], ["store0", 0]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 0>, handshake.name = "store0"} : <i10>, <f32>, <i10>, <f32>
    %37 = addi %38, %29 {handshake.bb = 2 : ui32, handshake.name = "addi0"} : <i32>
    %38 = buffer %22#2, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer13"} : <i32>
    %39 = br %37 {handshake.bb = 2 : ui32, handshake.name = "br5"} : <i32>
    %40 = br %20 {handshake.bb = 2 : ui32, handshake.name = "br6"} : <i32>
    %41 = buffer %26#1, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer4"} : <>
    %42 = buffer %41, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer5"} : <>
    %43 = br %42 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"0": [1,inf], [0,inf], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "br7"} : <>
    %result_14, %index_15 = control_merge [%falseResult_5]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>] to <>, <i1>
    sink %index_15 {handshake.name = "sink3"} : <i1>
    %44:3 = fork [3] %result_14 {handshake.bb = 3 : ui32, handshake.name = "fork8"} : <>
    end {handshake.bb = 3 : ui32, handshake.name = "end0"} %memEnd_1, %memEnd, %1#1, %0#1 : <>, <>, <>, <>
  }
}

