module {
  handshake.func @subdiag_fast(%arg0: memref<1000xf32>, %arg1: memref<1000xf32>, %arg2: memref<1000xf32>, %arg3: !handshake.control<>, %arg4: !handshake.control<>, %arg5: !handshake.control<>, %arg6: !handshake.control<>, ...) -> (!handshake.channel<i32>, !handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["d1", "d2", "e", "d1_start", "d2_start", "e_start", "start"], handshake.cfdfcThroughput = #handshake<cfdfcThroughput {"0" = 1.000000e+00 : f64}>, handshake.cfdfcToBBList = #handshake<cfdfcToBBList {"0" = [1 : ui32]}>, resNames = ["out0", "d1_end", "d2_end", "e_end", "end"]} {
    %0:3 = fork [3] %arg6 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork0"} : <>
    %1 = non_spec %0#2 {handshake.bb = 1 : ui32, handshake.name = "non_spec0"} : !handshake.control<> to !handshake.control<[spec: i1]>
    %2 = spec_commit[%38#0] %44#2 {handshake.bb = 2 : ui32, handshake.name = "spec_commit0"} : !handshake.control<[spec: i1]>, !handshake.control<>, <i1>
    %outputs, %memEnd = mem_controller[%arg2 : memref<1000xf32>] %arg5 (%addressResult_6) %2 {connectedBlocks = [1 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller3"} :    (!handshake.channel<i10>) -> !handshake.channel<f32>
    %3 = spec_commit[%38#1] %44#1 {handshake.bb = 2 : ui32, handshake.name = "spec_commit1"} : !handshake.control<[spec: i1]>, !handshake.control<>, <i1>
    %outputs_0, %memEnd_1 = mem_controller[%arg1 : memref<1000xf32>] %arg4 (%addressResult_4) %3 {connectedBlocks = [1 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller4"} :    (!handshake.channel<i10>) -> !handshake.channel<f32>
    %4 = spec_commit[%38#2] %44#0 {handshake.bb = 2 : ui32, handshake.name = "spec_commit2"} : !handshake.control<[spec: i1]>, !handshake.control<>, <i1>
    %outputs_2, %memEnd_3 = mem_controller[%arg0 : memref<1000xf32>] %arg3 (%addressResult) %4 {connectedBlocks = [1 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller5"} :    (!handshake.channel<i10>) -> !handshake.channel<f32>
    %5 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant0", value = false} : <>, <i1>
    %6 = extsi %5 {handshake.bb = 0 : ui32, handshake.name = "extsi3"} : <i1> to <i32>
    %7 = non_spec %6 {handshake.bb = 1 : ui32, handshake.name = "non_spec1"} : !handshake.channel<i32> to !handshake.channel<i32, [spec: i1]>
    %8 = mux %index [%7, %trueResult_12] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1, [spec: i1]>, [<i32, [spec: i1]>, <i32, [spec: i1]>] to <i32, [spec: i1]>
    %9 = buffer %8, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer0"} : <i32, [spec: i1]>
    %10 = buffer %9, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer1"} : <i32, [spec: i1]>
    %11:6 = fork [6] %10 {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <i32, [spec: i1]>
    %12 = trunci %11#0 {handshake.bb = 1 : ui32, handshake.name = "trunci0"} : <i32, [spec: i1]> to <i10, [spec: i1]>
    %13 = trunci %11#1 {handshake.bb = 1 : ui32, handshake.name = "trunci1"} : <i32, [spec: i1]> to <i10, [spec: i1]>
    %14 = trunci %11#2 {handshake.bb = 1 : ui32, handshake.name = "trunci2"} : <i32, [spec: i1]> to <i10, [spec: i1]>
    %result, %index = control_merge [%1, %trueResult_14]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<[spec: i1]>, <[spec: i1]>] to <[spec: i1]>, <i1, [spec: i1]>
    %15 = buffer %result, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer3"} : <[spec: i1]>
    %16:2 = fork [2] %15 {handshake.bb = 1 : ui32, handshake.name = "fork5"} : <[spec: i1]>
    %17 = source {handshake.bb = 1 : ui32, handshake.name = "source0"} : <[spec: i1]>
    %18 = constant %17 {handshake.bb = 1 : ui32, handshake.name = "constant1", value = 1 : i2} : <[spec: i1]>, <i2, [spec: i1]>
    %19:2 = fork [2] %18 {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <i2, [spec: i1]>
    %20 = extsi %19#0 {handshake.bb = 1 : ui32, handshake.name = "extsi4"} : <i2, [spec: i1]> to <i10, [spec: i1]>
    %21 = extsi %19#1 {handshake.bb = 1 : ui32, handshake.name = "extsi1"} : <i2, [spec: i1]> to <i32, [spec: i1]>
    %22 = source {handshake.bb = 1 : ui32, handshake.name = "source1"} : <[spec: i1]>
    %23 = constant %22 {handshake.bb = 1 : ui32, handshake.name = "constant5", value = 1.000000e-03 : f32} : <[spec: i1]>, <f32, [spec: i1]>
    %24 = source {handshake.bb = 1 : ui32, handshake.name = "source2"} : <[spec: i1]>
    %25 = constant %24 {handshake.bb = 1 : ui32, handshake.name = "constant2", value = 998 : i11} : <[spec: i1]>, <i11, [spec: i1]>
    %26 = extsi %25 {handshake.bb = 1 : ui32, handshake.name = "extsi2"} : <i11, [spec: i1]> to <i32, [spec: i1]>
    %addressResult, %dataResult = load[%14] %outputs_2 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i10, [spec: i1]>, <f32>, <i10>, <f32, [spec: i1]>
    %27 = addi %12, %20 {handshake.bb = 1 : ui32, handshake.name = "addi0"} : <i10, [spec: i1]>
    %addressResult_4, %dataResult_5 = load[%27] %outputs_0 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i10, [spec: i1]>, <f32>, <i10>, <f32, [spec: i1]>
    %28 = addf %dataResult, %dataResult_5 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "addf0", internal_delay = "2_922000"} : <f32, [spec: i1]>
    %29 = addi %11#5, %21 {handshake.bb = 1 : ui32, handshake.name = "addi1"} : <i32, [spec: i1]>
    %addressResult_6, %dataResult_7 = load[%13] %outputs {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load2"} : <i10, [spec: i1]>, <f32>, <i10>, <f32, [spec: i1]>
    %30 = mulf %28, %23 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "mulf0", internal_delay = "2_875333"} : <f32, [spec: i1]>
    %31 = buffer %dataResult_7, bufferType = FIFO_BREAK_NONE, numSlots = 13 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer4"} : <f32, [spec: i1]>
    %32 = cmpf ugt, %31, %30 {handshake.bb = 1 : ui32, handshake.name = "cmpf0", internal_delay = "0_000000"} : <f32, [spec: i1]>
    %33 = buffer %11#4, bufferType = FIFO_BREAK_NONE, numSlots = 14 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer2"} : <i32, [spec: i1]>
    %34 = cmpi ult, %33, %26 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i32, [spec: i1]>
    %35 = andi %34, %32 {handshake.bb = 1 : ui32, handshake.name = "andi0"} : <i1, [spec: i1]>
    %dataOut, %SCSaveCtrl = spec_prebuffer1[%16#1] {fifoDepth = 16 : ui32, handshake.bb = 1 : ui32, handshake.name = "spec_prebuffer10"} : <[spec: i1]>, <i1, [spec: i1]>, <i3>
    %36:3 = fork [3] %SCSaveCtrl {handshake.bb = 1 : ui32, handshake.name = "fork6"} : <i3>
    %37:5 = fork [5] %dataOut {handshake.bb = 1 : ui32, handshake.name = "fork7"} : <i1, [spec: i1]>
    %saveCtrl, %commitCtrl, %SCIsMisspec = spec_prebuffer2 %35 {fifoDepth = 16 : ui32, handshake.bb = 1 : ui32, handshake.name = "spec_prebuffer20"} : <i1, [spec: i1]>, <i1>, <i1>, <i1>
    sink %saveCtrl {handshake.name = "sink2"} : <i1>
    %trueResult, %falseResult = cond_br %40#0, %commitCtrl {handshake.bb = 1 : ui32, handshake.name = "cond_br0"} : <i1>, <i1>
    %38:4 = fork [4] %falseResult {handshake.bb = 1 : ui32, handshake.name = "fork8"} : <i1>
    sink %trueResult {handshake.name = "sink3"} : <i1>
    %trueResult_8, %falseResult_9 = speculating_branch[%37#0] %37#3 {handshake.bb = 1 : ui32, handshake.name = "speculating_branch0"} : !handshake.channel<i1, [spec: i1]>, !handshake.channel<i1, [spec: i1]>, !handshake.channel<i1>, !handshake.channel<i1>
    sink %falseResult_9 {handshake.name = "sink4"} : <i1>
    %39 = buffer %trueResult_8, bufferType = FIFO_BREAK_NONE, numSlots = 14 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer5"} : <i1>
    %40:2 = fork [2] %39 {handshake.bb = 1 : ui32, handshake.name = "fork9"} : <i1>
    %trueResult_10, %falseResult_11 = cond_br %SCIsMisspec, %40#1 {handshake.bb = 1 : ui32, handshake.name = "cond_br4"} : <i1>, <i1>
    sink %falseResult_11 {handshake.name = "sink5"} : <i1>
    sink %trueResult_10 {handshake.name = "sink6"} : <i1>
    %41 = spec_save_commit[%36#2] %29 {fifoDepth = 16 : ui32, handshake.bb = 1 : ui32, handshake.name = "spec_save_commit0"} : !handshake.channel<i32, [spec: i1]>, <i3>
    %trueResult_12, %falseResult_13 = cond_br %37#4, %41 {handshake.bb = 1 : ui32, handshake.name = "cond_br1"} : <i1, [spec: i1]>, <i32, [spec: i1]>
    sink %falseResult_13 {handshake.name = "sink0"} : <i32, [spec: i1]>
    %42 = spec_save_commit[%36#1] %16#0 {fifoDepth = 16 : ui32, handshake.bb = 1 : ui32, handshake.name = "spec_save_commit1"} : !handshake.control<[spec: i1]>, <i3>
    %trueResult_14, %falseResult_15 = cond_br %37#2, %42 {handshake.bb = 1 : ui32, handshake.name = "cond_br2"} : <i1, [spec: i1]>, <[spec: i1]>
    %43 = spec_save_commit[%36#0] %11#3 {fifoDepth = 16 : ui32, handshake.bb = 1 : ui32, handshake.name = "spec_save_commit2"} : !handshake.channel<i32, [spec: i1]>, <i3>
    %trueResult_16, %falseResult_17 = cond_br %37#1, %43 {handshake.bb = 1 : ui32, handshake.name = "cond_br3"} : <i1, [spec: i1]>, <i32, [spec: i1]>
    sink %trueResult_16 {handshake.name = "sink1"} : <i32, [spec: i1]>
    %44:3 = fork [3] %falseResult_15 {handshake.bb = 2 : ui32, handshake.name = "fork4"} : <[spec: i1]>
    %45 = buffer %falseResult_17, bufferType = FIFO_BREAK_NONE, numSlots = 14 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer6"} : <i32, [spec: i1]>
    %46 = spec_commit[%38#3] %45 {handshake.bb = 1 : ui32, handshake.name = "spec_commit3"} : !handshake.channel<i32, [spec: i1]>, !handshake.channel<i32>, <i1>
    end {handshake.bb = 2 : ui32, handshake.name = "end0"} %46, %memEnd_3, %memEnd_1, %memEnd, %0#1 : <i32>, <>, <>, <>, <>
  }
}

