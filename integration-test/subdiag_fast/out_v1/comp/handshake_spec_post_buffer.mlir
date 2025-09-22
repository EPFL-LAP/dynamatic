module {
  handshake.func @subdiag_fast(%arg0: memref<1000xf32>, %arg1: memref<1000xf32>, %arg2: memref<1000xf32>, %arg3: !handshake.control<>, %arg4: !handshake.control<>, %arg5: !handshake.control<>, %arg6: !handshake.control<>, ...) -> (!handshake.channel<i32>, !handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["d1", "d2", "e", "d1_start", "d2_start", "e_start", "start"], handshake.cfdfcThroughput = #handshake<cfdfcThroughput {"0" = 1.000000e+00 : f64}>, handshake.cfdfcToBBList = #handshake<cfdfcToBBList {"0" = [1 : ui32]}>, resNames = ["out0", "d1_end", "d2_end", "e_end", "end"]} {
    %0:3 = fork [3] %arg6 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork0"} : <>
    %1 = non_spec %0#2 {handshake.bb = 1 : ui32, handshake.name = "non_spec0"} : !handshake.control<> to !handshake.control<[spec: i1]>
    %2 = buffer %46#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer7"} : <i1>
    %3 = buffer %53#2, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer8"} : <[spec: i1]>
    %4 = spec_commit[%2] %3 {handshake.bb = 2 : ui32, handshake.name = "spec_commit0"} : !handshake.control<[spec: i1]>, !handshake.control<>, <i1>
    %outputs, %memEnd = mem_controller[%arg2 : memref<1000xf32>] %arg5 (%addressResult_6) %4 {connectedBlocks = [1 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller3"} :    (!handshake.channel<i10>) -> !handshake.channel<f32>
    %5 = buffer %46#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer9"} : <i1>
    %6 = buffer %53#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer10"} : <[spec: i1]>
    %7 = spec_commit[%5] %6 {handshake.bb = 2 : ui32, handshake.name = "spec_commit1"} : !handshake.control<[spec: i1]>, !handshake.control<>, <i1>
    %outputs_0, %memEnd_1 = mem_controller[%arg1 : memref<1000xf32>] %arg4 (%addressResult_4) %7 {connectedBlocks = [1 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller4"} :    (!handshake.channel<i10>) -> !handshake.channel<f32>
    %8 = buffer %46#2, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer11"} : <i1>
    %9 = buffer %53#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer12"} : <[spec: i1]>
    %10 = spec_commit[%8] %9 {handshake.bb = 2 : ui32, handshake.name = "spec_commit2"} : !handshake.control<[spec: i1]>, !handshake.control<>, <i1>
    %outputs_2, %memEnd_3 = mem_controller[%arg0 : memref<1000xf32>] %arg3 (%addressResult) %10 {connectedBlocks = [1 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller5"} :    (!handshake.channel<i10>) -> !handshake.channel<f32>
    %11 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant0", value = false} : <>, <i1>
    %12 = extsi %11 {handshake.bb = 0 : ui32, handshake.name = "extsi3"} : <i1> to <i32>
    %13 = non_spec %12 {handshake.bb = 1 : ui32, handshake.name = "non_spec1"} : !handshake.channel<i32> to !handshake.channel<i32, [spec: i1]>
    %14 = mux %index [%13, %trueResult_14] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1, [spec: i1]>, [<i32, [spec: i1]>, <i32, [spec: i1]>] to <i32, [spec: i1]>
    %15 = buffer %14, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer0"} : <i32, [spec: i1]>
    %16 = buffer %15, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer1"} : <i32, [spec: i1]>
    %17:6 = fork [6] %16 {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <i32, [spec: i1]>
    %18 = trunci %17#0 {handshake.bb = 1 : ui32, handshake.name = "trunci0"} : <i32, [spec: i1]> to <i10, [spec: i1]>
    %19 = trunci %17#1 {handshake.bb = 1 : ui32, handshake.name = "trunci1"} : <i32, [spec: i1]> to <i10, [spec: i1]>
    %20 = trunci %17#2 {handshake.bb = 1 : ui32, handshake.name = "trunci2"} : <i32, [spec: i1]> to <i10, [spec: i1]>
    %result, %index = control_merge [%1, %trueResult_16]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<[spec: i1]>, <[spec: i1]>] to <[spec: i1]>, <i1, [spec: i1]>
    %21 = buffer %result, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer3"} : <[spec: i1]>
    %22:2 = fork [2] %21 {handshake.bb = 1 : ui32, handshake.name = "fork5"} : <[spec: i1]>
    %23 = source {handshake.bb = 1 : ui32, handshake.name = "source0"} : <[spec: i1]>
    %24 = constant %23 {handshake.bb = 1 : ui32, handshake.name = "constant1", value = 1 : i2} : <[spec: i1]>, <i2, [spec: i1]>
    %25:2 = fork [2] %24 {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <i2, [spec: i1]>
    %26 = extsi %25#0 {handshake.bb = 1 : ui32, handshake.name = "extsi4"} : <i2, [spec: i1]> to <i10, [spec: i1]>
    %27 = extsi %25#1 {handshake.bb = 1 : ui32, handshake.name = "extsi1"} : <i2, [spec: i1]> to <i32, [spec: i1]>
    %28 = source {handshake.bb = 1 : ui32, handshake.name = "source1"} : <[spec: i1]>
    %29 = constant %28 {handshake.bb = 1 : ui32, handshake.name = "constant5", value = 1.000000e-03 : f32} : <[spec: i1]>, <f32, [spec: i1]>
    %30 = source {handshake.bb = 1 : ui32, handshake.name = "source2"} : <[spec: i1]>
    %31 = constant %30 {handshake.bb = 1 : ui32, handshake.name = "constant2", value = 998 : i11} : <[spec: i1]>, <i11, [spec: i1]>
    %32 = extsi %31 {handshake.bb = 1 : ui32, handshake.name = "extsi2"} : <i11, [spec: i1]> to <i32, [spec: i1]>
    %addressResult, %dataResult = load[%20] %outputs_2 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i10, [spec: i1]>, <f32>, <i10>, <f32, [spec: i1]>
    %33 = addi %18, %26 {handshake.bb = 1 : ui32, handshake.name = "addi0"} : <i10, [spec: i1]>
    %addressResult_4, %dataResult_5 = load[%33] %outputs_0 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i10, [spec: i1]>, <f32>, <i10>, <f32, [spec: i1]>
    %34 = addf %dataResult, %dataResult_5 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "addf0", internal_delay = "2_922000"} : <f32, [spec: i1]>
    %35 = addi %17#5, %27 {handshake.bb = 1 : ui32, handshake.name = "addi1"} : <i32, [spec: i1]>
    %addressResult_6, %dataResult_7 = load[%19] %outputs {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load2"} : <i10, [spec: i1]>, <f32>, <i10>, <f32, [spec: i1]>
    %36 = mulf %34, %29 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "mulf0", internal_delay = "2_875333"} : <f32, [spec: i1]>
    %37 = buffer %dataResult_7, bufferType = FIFO_BREAK_NONE, numSlots = 13 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer4"} : <f32, [spec: i1]>
    %38 = cmpf ugt, %37, %36 {handshake.bb = 1 : ui32, handshake.name = "cmpf0", internal_delay = "0_000000"} : <f32, [spec: i1]>
    %39 = buffer %17#4, bufferType = FIFO_BREAK_NONE, numSlots = 14 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer2"} : <i32, [spec: i1]>
    %40 = cmpi ult, %39, %32 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i32, [spec: i1]>
    %41 = andi %40, %38 {handshake.bb = 1 : ui32, handshake.name = "andi0"} : <i1, [spec: i1]>
    %dataOut, %saveCtrl, %commitCtrl, %SCSaveCtrl, %SCCommitCtrl, %SCIsMisspec = speculator[%22#1] %41 {constant = true, defaultValue = 1 : ui32, fifoDepth = 16 : ui32, handshake.bb = 1 : ui32, handshake.name = "speculator0"} : <[spec: i1]>, <i1, [spec: i1]>, <i1, [spec: i1]>, <i1>, <i1>, <i3>, <i3>, <i1>
    %trueResult, %falseResult = cond_br %trueResult_12, %SCCommitCtrl {handshake.bb = 1 : ui32, handshake.name = "cond_br5"} : <i1>, <i3>
    sink %falseResult {handshake.name = "sink7"} : <i3>
    %42 = merge %SCSaveCtrl, %trueResult {handshake.bb = 1 : ui32, handshake.name = "merge0"} : <i3>
    %43:3 = fork [3] %42 {handshake.bb = 1 : ui32, handshake.name = "fork6"} : <i3>
    %44:5 = fork [5] %dataOut {handshake.bb = 1 : ui32, handshake.name = "fork7"} : <i1, [spec: i1]>
    sink %saveCtrl {handshake.name = "sink2"} : <i1>
    %45 = buffer %48#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer13"} : <i1>
    %trueResult_8, %falseResult_9 = cond_br %45, %commitCtrl {handshake.bb = 1 : ui32, handshake.name = "cond_br0"} : <i1>, <i1>
    %46:4 = fork [4] %falseResult_9 {handshake.bb = 1 : ui32, handshake.name = "fork8"} : <i1>
    sink %trueResult_8 {handshake.name = "sink3"} : <i1>
    %trueResult_10, %falseResult_11 = speculating_branch[%44#0] %44#3 {handshake.bb = 1 : ui32, handshake.name = "speculating_branch0"} : !handshake.channel<i1, [spec: i1]>, !handshake.channel<i1, [spec: i1]>, !handshake.channel<i1>, !handshake.channel<i1>
    sink %falseResult_11 {handshake.name = "sink4"} : <i1>
    %47 = buffer %trueResult_10, bufferType = FIFO_BREAK_NONE, numSlots = 14 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer5"} : <i1>
    %48:2 = fork [2] %47 {handshake.bb = 1 : ui32, handshake.name = "fork9"} : <i1>
    %49 = buffer %48#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer14"} : <i1>
    %trueResult_12, %falseResult_13 = cond_br %SCIsMisspec, %49 {handshake.bb = 1 : ui32, handshake.name = "cond_br4"} : <i1>, <i1>
    sink %falseResult_13 {handshake.name = "sink5"} : <i1>
    %50 = spec_save_commit[%43#2] %35 {fifoDepth = 16 : ui32, handshake.bb = 1 : ui32, handshake.name = "spec_save_commit0"} : !handshake.channel<i32, [spec: i1]>, <i3>
    %trueResult_14, %falseResult_15 = cond_br %44#4, %50 {handshake.bb = 1 : ui32, handshake.name = "cond_br1"} : <i1, [spec: i1]>, <i32, [spec: i1]>
    sink %falseResult_15 {handshake.name = "sink0"} : <i32, [spec: i1]>
    %51 = spec_save_commit[%43#1] %22#0 {fifoDepth = 16 : ui32, handshake.bb = 1 : ui32, handshake.name = "spec_save_commit1"} : !handshake.control<[spec: i1]>, <i3>
    %trueResult_16, %falseResult_17 = cond_br %44#2, %51 {handshake.bb = 1 : ui32, handshake.name = "cond_br2"} : <i1, [spec: i1]>, <[spec: i1]>
    %52 = spec_save_commit[%43#0] %17#3 {fifoDepth = 16 : ui32, handshake.bb = 1 : ui32, handshake.name = "spec_save_commit2"} : !handshake.channel<i32, [spec: i1]>, <i3>
    %trueResult_18, %falseResult_19 = cond_br %44#1, %52 {handshake.bb = 1 : ui32, handshake.name = "cond_br3"} : <i1, [spec: i1]>, <i32, [spec: i1]>
    sink %trueResult_18 {handshake.name = "sink1"} : <i32, [spec: i1]>
    %53:3 = fork [3] %falseResult_17 {handshake.bb = 2 : ui32, handshake.name = "fork4"} : <[spec: i1]>
    %54 = buffer %falseResult_19, bufferType = FIFO_BREAK_NONE, numSlots = 14 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer6"} : <i32, [spec: i1]>
    %55 = buffer %46#3, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer15"} : <i1>
    %56 = buffer %54, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer16"} : <i32, [spec: i1]>
    %57 = spec_commit[%55] %56 {handshake.bb = 1 : ui32, handshake.name = "spec_commit3"} : !handshake.channel<i32, [spec: i1]>, !handshake.channel<i32>, <i1>
    end {handshake.bb = 2 : ui32, handshake.name = "end0"} %57, %memEnd_3, %memEnd_1, %memEnd, %0#1 : <i32>, <>, <>, <>, <>
  }
}

