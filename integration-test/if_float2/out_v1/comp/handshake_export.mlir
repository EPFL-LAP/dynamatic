module {
  handshake.func @if_float2(%arg0: !handshake.channel<f32>, %arg1: memref<100xf32>, %arg2: memref<100xf32>, %arg3: !handshake.control<>, %arg4: !handshake.control<>, %arg5: !handshake.control<>, ...) -> (!handshake.channel<f32>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["x0", "a", "minus_trace", "a_start", "minus_trace_start", "start"], handshake.cfdfcThroughput = #handshake<cfdfcThroughput {"0" = 0.034482758620690127 : f64}>, handshake.cfdfcToBBList = #handshake<cfdfcToBBList {"0" = [1 : ui32]}>, resNames = ["out0", "a_end", "minus_trace_end", "end"]} {
    %0 = non_spec %arg0 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "non_spec0"} : !handshake.channel<f32> to !handshake.channel<f32, [spec: i1]>
    %1:3 = fork [3] %arg5 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork0"} : <>
    %2 = non_spec %1#2 {handshake.bb = 1 : ui32, handshake.name = "non_spec1"} : !handshake.control<> to !handshake.control<[spec: i1]>
    %3 = buffer %39#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer17"} : <i1>
    %4 = buffer %96#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer18"} : <[spec: i1]>
    %5 = spec_commit[%3] %4 {handshake.bb = 2 : ui32, handshake.name = "spec_commit0"} : !handshake.control<[spec: i1]>, !handshake.control<>, <i1>
    %6 = buffer %42#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer19"} : <i1>
    %7 = buffer %59, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer20"} : <i32, [spec: i1]>
    %8 = spec_commit[%6] %7 {handshake.bb = 1 : ui32, handshake.name = "spec_commit1"} : !handshake.channel<i32, [spec: i1]>, !handshake.channel<i32>, <i1>
    %memEnd = mem_controller[%arg2 : memref<100xf32>] %arg4 (%8, %addressResult_17, %dataResult_18) %5 {connectedBlocks = [1 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller2"} :    (!handshake.channel<i32>, !handshake.channel<i7>, !handshake.channel<f32>) -> ()
    %9 = buffer %39#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer21"} : <i1>
    %10 = buffer %96#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer22"} : <[spec: i1]>
    %11 = spec_commit[%9] %10 {handshake.bb = 2 : ui32, handshake.name = "spec_commit2"} : !handshake.control<[spec: i1]>, !handshake.control<>, <i1>
    %outputs, %memEnd_0 = mem_controller[%arg1 : memref<100xf32>] %arg3 (%addressResult) %11 {connectedBlocks = [1 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller3"} :    (!handshake.channel<i7>) -> !handshake.channel<f32>
    %12 = constant %1#0 {handshake.bb = 0 : ui32, handshake.name = "constant2", value = false} : <>, <i1>
    %13 = extsi %12 {handshake.bb = 0 : ui32, handshake.name = "extsi4"} : <i1> to <i8>
    %14 = non_spec %13 {handshake.bb = 1 : ui32, handshake.name = "non_spec2"} : !handshake.channel<i8> to !handshake.channel<i8, [spec: i1]>
    %15 = mux %23#0 [%14, %trueResult_19] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1, [spec: i1]>, [<i8, [spec: i1]>, <i8, [spec: i1]>] to <i8, [spec: i1]>
    %16 = buffer %15, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer0"} : <i8, [spec: i1]>
    %17:2 = fork [2] %16 {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <i8, [spec: i1]>
    %18 = trunci %17#0 {handshake.bb = 1 : ui32, handshake.name = "trunci0"} : <i8, [spec: i1]> to <i7, [spec: i1]>
    %19 = mux %23#1 [%0, %trueResult_21] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i1, [spec: i1]>, [<f32, [spec: i1]>, <f32, [spec: i1]>] to <f32, [spec: i1]>
    %20 = buffer %19, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer2"} : <f32, [spec: i1]>
    %21:3 = fork [3] %20 {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <f32, [spec: i1]>
    %result, %index = control_merge [%2, %trueResult_23]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<[spec: i1]>, <[spec: i1]>] to <[spec: i1]>, <i1, [spec: i1]>
    %22:2 = fork [2] %result {handshake.bb = 1 : ui32, handshake.name = "fork4"} : <[spec: i1]>
    %23:2 = fork [2] %index {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <i1, [spec: i1]>
    %24 = source {handshake.bb = 1 : ui32, handshake.name = "source0"} : <[spec: i1]>
    %25 = constant %24 {handshake.bb = 1 : ui32, handshake.name = "constant7", value = -6.000000e-01 : f32} : <[spec: i1]>, <f32, [spec: i1]>
    %26 = source {handshake.bb = 1 : ui32, handshake.name = "source1"} : <[spec: i1]>
    %27 = constant %26 {handshake.bb = 1 : ui32, handshake.name = "constant8", value = 0.000000e+00 : f32} : <[spec: i1]>, <f32, [spec: i1]>
    %28 = buffer %18, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer1"} : <i7, [spec: i1]>
    %addressResult, %dataResult = load[%28] %outputs {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i7, [spec: i1]>, <f32>, <i7>, <f32, [spec: i1]>
    %29 = buffer %21#2, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer3"} : <f32, [spec: i1]>
    %30 = mulf %dataResult, %29 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "mulf0", internal_delay = "2_875333"} : <f32, [spec: i1]>
    %31 = mulf %21#1, %25 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "mulf1", internal_delay = "2_875333"} : <f32, [spec: i1]>
    %32 = buffer %31, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer4"} : <f32, [spec: i1]>
    %33 = addf %30, %32 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "addf0", internal_delay = "2_922000"} : <f32, [spec: i1]>
    %34 = cmpf ugt, %33, %27 {handshake.bb = 1 : ui32, handshake.name = "cmpf0", internal_delay = "0_000000"} : <f32, [spec: i1]>
    %dataOut, %saveCtrl, %commitCtrl, %SCSaveCtrl, %SCCommitCtrl, %SCIsMisspec = speculator[%22#1] %34 {constant = true, defaultValue = 0 : ui32, fifoDepth = 30 : ui32, handshake.bb = 1 : ui32, handshake.name = "speculator0"} : <[spec: i1]>, <i1, [spec: i1]>, <i1, [spec: i1]>, <i1>, <i1>, <i3>, <i3>, <i1>
    %trueResult, %falseResult = cond_br %trueResult_9, %SCCommitCtrl {handshake.bb = 1 : ui32, handshake.name = "cond_br9"} : <i1>, <i3>
    sink %falseResult {handshake.name = "sink8"} : <i3>
    %35 = merge %SCSaveCtrl, %trueResult {handshake.bb = 1 : ui32, handshake.name = "merge0"} : <i3>
    %36:3 = fork [3] %35 {handshake.bb = 1 : ui32, handshake.name = "fork8"} : <i3>
    %37:2 = fork [2] %commitCtrl {handshake.bb = 1 : ui32, handshake.name = "fork12"} : <i1>
    sink %saveCtrl {handshake.name = "sink0"} : <i1>
    %38 = buffer %43#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer23"} : <i1>
    %trueResult_1, %falseResult_2 = cond_br %38, %37#0 {handshake.bb = 1 : ui32, handshake.name = "cond_br0"} : <i1>, <i1>
    %39:3 = fork [3] %falseResult_2 {handshake.bb = 1 : ui32, handshake.name = "fork14"} : <i1>
    sink %trueResult_1 {handshake.name = "sink1"} : <i1>
    %trueResult_3, %falseResult_4 = speculating_branch[%45#5] %45#6 {handshake.bb = 1 : ui32, handshake.name = "speculating_branch0"} : !handshake.channel<i1, [spec: i1]>, !handshake.channel<i1, [spec: i1]>, !handshake.channel<i1>, !handshake.channel<i1>
    sink %falseResult_4 {handshake.name = "sink3"} : <i1>
    %40 = buffer %trueResult_3, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer5"} : <i1>
    %41 = buffer %40, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer24"} : <i1>
    %trueResult_5, %falseResult_6 = cond_br %41, %37#1 {handshake.bb = 1 : ui32, handshake.name = "cond_br1"} : <i1>, <i1>
    sink %falseResult_6 {handshake.name = "sink4"} : <i1>
    %42:3 = fork [3] %trueResult_5 {handshake.bb = 1 : ui32, handshake.name = "fork15"} : <i1>
    %trueResult_7, %falseResult_8 = speculating_branch[%93#2] %93#3 {handshake.bb = 1 : ui32, handshake.name = "speculating_branch1"} : !handshake.channel<i1, [spec: i1]>, !handshake.channel<i1, [spec: i1]>, !handshake.channel<i1>, !handshake.channel<i1>
    sink %falseResult_8 {handshake.name = "sink5"} : <i1>
    %43:2 = fork [2] %trueResult_7 {handshake.bb = 1 : ui32, handshake.name = "fork16"} : <i1>
    %44 = buffer %43#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer25"} : <i1>
    %trueResult_9, %falseResult_10 = cond_br %SCIsMisspec, %44 {handshake.bb = 1 : ui32, handshake.name = "cond_br8"} : <i1>, <i1>
    sink %falseResult_10 {handshake.name = "sink6"} : <i1>
    %45:8 = fork [8] %dataOut {handshake.bb = 1 : ui32, handshake.name = "fork17"} : <i1, [spec: i1]>
    %46 = spec_save_commit[%36#1] %17#1 {fifoDepth = 30 : ui32, handshake.bb = 1 : ui32, handshake.name = "spec_save_commit0"} : !handshake.channel<i8, [spec: i1]>, <i3>
    %trueResult_11, %falseResult_12 = cond_br %45#7, %46 {handshake.bb = 1 : ui32, handshake.name = "cond_br2", specv1_adaptor_inner_loop = true} : <i1, [spec: i1]>, <i8, [spec: i1]>
    %47 = spec_save_commit[%36#0] %21#0 {fifoDepth = 30 : ui32, handshake.bb = 1 : ui32, handshake.name = "spec_save_commit1"} : !handshake.channel<f32, [spec: i1]>, <i3>
    %trueResult_13, %falseResult_14 = cond_br %45#4, %47 {handshake.bb = 1 : ui32, handshake.name = "cond_br3", specv1_adaptor_inner_loop = true} : <i1, [spec: i1]>, <f32, [spec: i1]>
    %48 = spec_save_commit[%36#2] %22#0 {fifoDepth = 30 : ui32, handshake.bb = 1 : ui32, handshake.name = "spec_save_commit2"} : !handshake.control<[spec: i1]>, <i3>
    %trueResult_15, %falseResult_16 = cond_br %45#3, %48 {handshake.bb = 1 : ui32, handshake.name = "cond_br4", specv1_adaptor_inner_loop = true} : <i1, [spec: i1]>, <[spec: i1]>
    %49 = source {handshake.bb = 1 : ui32, handshake.name = "source2"} : <[spec: i1]>
    %50 = constant %49 {handshake.bb = 1 : ui32, handshake.name = "constant9", value = 3.000000e+00 : f32} : <[spec: i1]>, <f32, [spec: i1]>
    %51 = addf %falseResult_14, %50 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "addf1", internal_delay = "2_922000"} : <f32, [spec: i1]>
    %52:2 = fork [2] %trueResult_11 {handshake.bb = 1 : ui32, handshake.name = "fork5"} : <i8, [spec: i1]>
    %53 = buffer %52#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer9"} : <i8, [spec: i1]>
    %54 = trunci %53 {handshake.bb = 1 : ui32, handshake.name = "trunci1"} : <i8, [spec: i1]> to <i7, [spec: i1]>
    %55:2 = fork [2] %trueResult_13 {handshake.bb = 1 : ui32, handshake.name = "fork6"} : <f32, [spec: i1]>
    %56:2 = fork [2] %trueResult_15 {handshake.bb = 1 : ui32, handshake.name = "fork7"} : <[spec: i1]>
    %57 = buffer %56#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer11"} : <[spec: i1]>
    %58 = constant %57 {handshake.bb = 1 : ui32, handshake.name = "constant3", value = 1 : i2} : <[spec: i1]>, <i2, [spec: i1]>
    %59 = extsi %58 {handshake.bb = 1 : ui32, handshake.name = "extsi1"} : <i2, [spec: i1]> to <i32, [spec: i1]>
    %60 = source {handshake.bb = 1 : ui32, handshake.name = "source3"} : <[spec: i1]>
    %61 = constant %60 {handshake.bb = 1 : ui32, handshake.name = "constant10", value = 1.000000e+00 : f32} : <[spec: i1]>, <f32, [spec: i1]>
    %62 = buffer %42#2, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer26"} : <i1>
    %63 = buffer %54, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer27"} : <i7, [spec: i1]>
    %64 = spec_commit[%62] %63 {handshake.bb = 1 : ui32, handshake.name = "spec_commit3"} : !handshake.channel<i7, [spec: i1]>, !handshake.channel<i7>, <i1>
    %65 = buffer %55#0, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer10"} : <f32, [spec: i1]>
    %66 = buffer %42#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer28"} : <i1>
    %67 = buffer %65, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer29"} : <f32, [spec: i1]>
    %68 = spec_commit[%66] %67 {handshake.bb = 1 : ui32, handshake.name = "spec_commit4"} : !handshake.channel<f32, [spec: i1]>, !handshake.channel<f32>, <i1>
    %addressResult_17, %dataResult_18 = store[%64] %68 {handshake.bb = 1 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store0"} : <i7>, <f32>, <i7>, <f32>
    %69 = addf %55#1, %61 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "addf2", internal_delay = "2_922000"} : <f32, [spec: i1]>
    %70 = buffer %45#1, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer6"} : <i1, [spec: i1]>
    %71 = mux %70 [%51, %69] {handshake.bb = 1 : ui32, handshake.name = "mux2", specv1_adaptor_inner_loop = true} : <i1, [spec: i1]>, [<f32, [spec: i1]>, <f32, [spec: i1]>] to <f32, [spec: i1]>
    %72 = buffer %falseResult_12, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer7"} : <i8, [spec: i1]>
    %73 = mux %45#0 [%72, %52#1] {handshake.bb = 1 : ui32, handshake.name = "mux3", specv1_adaptor_inner_loop = true} : <i1, [spec: i1]>, [<i8, [spec: i1]>, <i8, [spec: i1]>] to <i8, [spec: i1]>
    %74 = buffer %73, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer13"} : <i8, [spec: i1]>
    %75 = extsi %74 {handshake.bb = 1 : ui32, handshake.name = "extsi5"} : <i8, [spec: i1]> to <i9, [spec: i1]>
    %76 = buffer %falseResult_16, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer8"} : <[spec: i1]>
    %77 = mux %45#2 [%76, %56#1] {handshake.bb = 1 : ui32, handshake.name = "mux4", specv1_adaptor_inner_loop = true} : <i1, [spec: i1]>, [<[spec: i1]>, <[spec: i1]>] to <[spec: i1]>
    %78 = source {handshake.bb = 1 : ui32, handshake.name = "source4"} : <[spec: i1]>
    %79 = constant %78 {handshake.bb = 1 : ui32, handshake.name = "constant11", value = 1.000000e+00 : f32} : <[spec: i1]>, <f32, [spec: i1]>
    %80 = source {handshake.bb = 1 : ui32, handshake.name = "source5"} : <[spec: i1]>
    %81 = constant %80 {handshake.bb = 1 : ui32, handshake.name = "constant4", value = 1 : i2} : <[spec: i1]>, <i2, [spec: i1]>
    %82 = extsi %81 {handshake.bb = 1 : ui32, handshake.name = "extsi6"} : <i2, [spec: i1]> to <i9, [spec: i1]>
    %83 = source {handshake.bb = 1 : ui32, handshake.name = "source6"} : <[spec: i1]>
    %84 = constant %83 {handshake.bb = 1 : ui32, handshake.name = "constant5", value = 100 : i8} : <[spec: i1]>, <i8, [spec: i1]>
    %85 = extsi %84 {handshake.bb = 1 : ui32, handshake.name = "extsi7"} : <i8, [spec: i1]> to <i9, [spec: i1]>
    %86 = buffer %71, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer12"} : <f32, [spec: i1]>
    %87 = divf %79, %86 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "divf0", internal_delay = "3_812000"} : <f32, [spec: i1]>
    %88 = buffer %75, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer14"} : <i9, [spec: i1]>
    %89 = addi %88, %82 {handshake.bb = 1 : ui32, handshake.name = "addi0"} : <i9, [spec: i1]>
    %90:2 = fork [2] %89 {handshake.bb = 1 : ui32, handshake.name = "fork9"} : <i9, [spec: i1]>
    %91 = trunci %90#0 {handshake.bb = 1 : ui32, handshake.name = "trunci2"} : <i9, [spec: i1]> to <i8, [spec: i1]>
    %92 = cmpi ult, %90#1, %85 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i9, [spec: i1]>
    %93:5 = fork [5] %92 {handshake.bb = 1 : ui32, handshake.name = "fork18"} : <i1, [spec: i1]>
    %trueResult_19, %falseResult_20 = cond_br %93#4, %91 {handshake.bb = 1 : ui32, handshake.name = "cond_br5"} : <i1, [spec: i1]>, <i8, [spec: i1]>
    sink %falseResult_20 {handshake.name = "sink2"} : <i8, [spec: i1]>
    %trueResult_21, %falseResult_22 = cond_br %93#0, %87 {handshake.bb = 1 : ui32, handshake.name = "cond_br6"} : <i1, [spec: i1]>, <f32, [spec: i1]>
    %94 = buffer %77, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer15"} : <[spec: i1]>
    %95 = buffer %94, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer16"} : <[spec: i1]>
    %trueResult_23, %falseResult_24 = cond_br %93#1, %95 {handshake.bb = 1 : ui32, handshake.name = "cond_br7"} : <i1, [spec: i1]>, <[spec: i1]>
    %96:2 = fork [2] %falseResult_24 {handshake.bb = 2 : ui32, handshake.name = "fork11"} : <[spec: i1]>
    %97 = buffer %39#2, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer30"} : <i1>
    %98 = buffer %falseResult_22, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer31"} : <f32, [spec: i1]>
    %99 = spec_commit[%97] %98 {handshake.bb = 1 : ui32, handshake.name = "spec_commit5"} : !handshake.channel<f32, [spec: i1]>, !handshake.channel<f32>, <i1>
    end {handshake.bb = 2 : ui32, handshake.name = "end0"} %99, %memEnd_0, %memEnd, %1#1 : <f32>, <>, <>, <>
  }
}

