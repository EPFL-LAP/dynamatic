module {
  handshake.func @subdiag_fast(%arg0: memref<1000xf32>, %arg1: memref<1000xf32>, %arg2: memref<1000xf32>, %arg3: !handshake.control<>, %arg4: !handshake.control<>, %arg5: !handshake.control<>, %arg6: !handshake.control<>, ...) -> (!handshake.channel<i32>, !handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["d1", "d2", "e", "d1_start", "d2_start", "e_start", "start"], handshake.cfdfcThroughput = #handshake<cfdfcThroughput {"0" = 0.071428571428571425 : f64}>, handshake.cfdfcToBBList = #handshake<cfdfcToBBList {"0" = [2 : ui32, 1 : ui32]}>, resNames = ["out0", "d1_end", "d2_end", "e_end", "end"]} {
    %0:3 = fork [3] %arg6 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork0"} : <>
    %outputs, %memEnd = mem_controller[%arg2 : memref<1000xf32>] %arg5 (%addressResult_6) %33#2 {connectedBlocks = [1 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller3"} :    (!handshake.channel<i10>) -> !handshake.channel<f32>
    %outputs_0, %memEnd_1 = mem_controller[%arg1 : memref<1000xf32>] %arg4 (%addressResult_4) %33#1 {connectedBlocks = [1 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller4"} :    (!handshake.channel<i10>) -> !handshake.channel<f32>
    %outputs_2, %memEnd_3 = mem_controller[%arg0 : memref<1000xf32>] %arg3 (%addressResult) %33#0 {connectedBlocks = [1 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller5"} :    (!handshake.channel<i10>) -> !handshake.channel<f32>
    %1 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant0", value = false} : <>, <i1>
    %2 = extsi %1 {handshake.bb = 0 : ui32, handshake.name = "extsi3"} : <i1> to <i11>
    %3 = mux %index [%2, %30] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i11>, <i11>] to <i11>
    %4 = buffer %3, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer0"} : <i11>
    %5:4 = fork [4] %4 {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <i11>
    %6 = trunci %5#0 {handshake.bb = 1 : ui32, handshake.name = "trunci0"} : <i11> to <i10>
    %7 = trunci %5#1 {handshake.bb = 1 : ui32, handshake.name = "trunci1"} : <i11> to <i10>
    %8 = trunci %5#2 {handshake.bb = 1 : ui32, handshake.name = "trunci2"} : <i11> to <i10>
    %result, %index = control_merge [%0#2, %trueResult_12]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %9 = source {handshake.bb = 1 : ui32, handshake.name = "source0"} : <>
    %10 = constant %9 {handshake.bb = 1 : ui32, handshake.name = "constant4", value = 1.000000e-03 : f32} : <>, <f32>
    %addressResult, %dataResult = load[%8] %outputs_2 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i10>, <f32>, <i10>, <f32>
    %addressResult_4, %dataResult_5 = load[%7] %outputs_0 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i10>, <f32>, <i10>, <f32>
    %11 = addf %dataResult, %dataResult_5 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "addf0", internal_delay = "2_922000"} : <f32>
    %addressResult_6, %dataResult_7 = load[%6] %outputs {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load2"} : <i10>, <f32>, <i10>, <f32>
    %12 = mulf %11, %10 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "mulf0", internal_delay = "2_875333"} : <f32>
    %13 = buffer %dataResult_7, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer3"} : <f32>
    %14 = cmpf ugt, %13, %12 {handshake.bb = 1 : ui32, handshake.name = "cmpf0", internal_delay = "0_000000"} : <f32>
    %15:2 = fork [2] %14 {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <i1>
    %16 = buffer %5#3, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer1"} : <i11>
    %trueResult, %falseResult = cond_br %15#0, %16 {handshake.bb = 1 : ui32, handshake.name = "cond_br2"} : <i1>, <i11>
    %17 = extsi %falseResult {handshake.bb = 1 : ui32, handshake.name = "extsi4"} : <i11> to <i12>
    %18 = buffer %result, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer2"} : <>
    %trueResult_8, %falseResult_9 = cond_br %15#1, %18 {handshake.bb = 1 : ui32, handshake.name = "cond_br3"} : <i1>, <>
    %19 = extsi %trueResult {handshake.bb = 2 : ui32, handshake.name = "extsi5"} : <i11> to <i12>
    %20 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %21 = constant %20 {handshake.bb = 2 : ui32, handshake.name = "constant1", value = 1 : i2} : <>, <i2>
    %22 = extsi %21 {handshake.bb = 2 : ui32, handshake.name = "extsi6"} : <i2> to <i12>
    %23 = source {handshake.bb = 2 : ui32, handshake.name = "source2"} : <>
    %24 = constant %23 {handshake.bb = 2 : ui32, handshake.name = "constant2", value = 999 : i11} : <>, <i11>
    %25 = extsi %24 {handshake.bb = 2 : ui32, handshake.name = "extsi7"} : <i11> to <i12>
    %26 = addi %19, %22 {handshake.bb = 2 : ui32, handshake.name = "addi0"} : <i12>
    %27:2 = fork [2] %26 {handshake.bb = 2 : ui32, handshake.name = "fork3"} : <i12>
    %28 = cmpi ult, %27#1, %25 {handshake.bb = 2 : ui32, handshake.name = "cmpi0"} : <i12>
    %29:2 = fork [2] %28 {handshake.bb = 2 : ui32, handshake.name = "fork4"} : <i1>
    %trueResult_10, %falseResult_11 = cond_br %29#0, %27#0 {handshake.bb = 2 : ui32, handshake.name = "cond_br4"} : <i1>, <i12>
    %30 = trunci %trueResult_10 {handshake.bb = 2 : ui32, handshake.name = "trunci3"} : <i12> to <i11>
    %trueResult_12, %falseResult_13 = cond_br %29#1, %trueResult_8 {handshake.bb = 2 : ui32, handshake.name = "cond_br5"} : <i1>, <>
    %31 = mux %index_15 [%17, %falseResult_11] {handshake.bb = 3 : ui32, handshake.name = "mux1"} : <i1>, [<i12>, <i12>] to <i12>
    %32 = extsi %31 {handshake.bb = 3 : ui32, handshake.name = "extsi8"} : <i12> to <i32>
    %result_14, %index_15 = control_merge [%falseResult_9, %falseResult_13]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>, <>] to <>, <i1>
    %33:3 = fork [3] %result_14 {handshake.bb = 3 : ui32, handshake.name = "fork5"} : <>
    end {handshake.bb = 3 : ui32, handshake.name = "end0"} %32, %memEnd_3, %memEnd_1, %memEnd, %0#1 : <i32>, <>, <>, <>, <>
  }
}

