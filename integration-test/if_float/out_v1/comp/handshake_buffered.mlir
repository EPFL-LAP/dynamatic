module {
  handshake.func @if_float(%arg0: !handshake.channel<f32>, %arg1: memref<100xf32>, %arg2: memref<100xf32>, %arg3: !handshake.control<>, %arg4: !handshake.control<>, %arg5: !handshake.control<>, ...) -> (!handshake.channel<f32>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["x0", "a", "minus_trace", "a_start", "minus_trace_start", "start"], handshake.cfdfcThroughput = #handshake<cfdfcThroughput {"0" = 0.034482758620689655 : f64}>, handshake.cfdfcToBBList = #handshake<cfdfcToBBList {"0" = [1 : ui32]}>, resNames = ["out0", "a_end", "minus_trace_end", "end"]} {
    %0 = non_spec %arg0 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "non_spec0"} : !handshake.channel<f32> to !handshake.channel<f32, [spec: i1]>
    %1:3 = fork [3] %arg5 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork0"} : <>
    %2 = non_spec %1#2 {handshake.bb = 1 : ui32, handshake.name = "non_spec1"} : !handshake.control<> to !handshake.control<[spec: i1]>
    %3 = spec_commit[%31#0] %82#1 {handshake.bb = 2 : ui32, handshake.name = "spec_commit0"} : !handshake.control<[spec: i1]>, !handshake.control<>, <i1>
    %4 = spec_commit[%33#1] %50 {handshake.bb = 1 : ui32, handshake.name = "spec_commit1"} : !handshake.channel<i32, [spec: i1]>, !handshake.channel<i32>, <i1>
    %memEnd = mem_controller[%arg2 : memref<100xf32>] %arg4 (%4, %addressResult_15, %dataResult_16) %3 {connectedBlocks = [1 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller2"} :    (!handshake.channel<i32>, !handshake.channel<i7>, !handshake.channel<f32>) -> ()
    %5 = spec_commit[%31#1] %82#0 {handshake.bb = 2 : ui32, handshake.name = "spec_commit2"} : !handshake.control<[spec: i1]>, !handshake.control<>, <i1>
    %outputs, %memEnd_0 = mem_controller[%arg1 : memref<100xf32>] %arg3 (%addressResult) %5 {connectedBlocks = [1 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller3"} :    (!handshake.channel<i7>) -> !handshake.channel<f32>
    %6 = constant %1#0 {handshake.bb = 0 : ui32, handshake.name = "constant2", value = false} : <>, <i1>
    %7 = extsi %6 {handshake.bb = 0 : ui32, handshake.name = "extsi4"} : <i1> to <i8>
    %8 = non_spec %7 {handshake.bb = 1 : ui32, handshake.name = "non_spec2"} : !handshake.channel<i8> to !handshake.channel<i8, [spec: i1]>
    %9 = mux %17#0 [%8, %trueResult_17] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1, [spec: i1]>, [<i8, [spec: i1]>, <i8, [spec: i1]>] to <i8, [spec: i1]>
    %10 = buffer %9, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer0"} : <i8, [spec: i1]>
    %11:2 = fork [2] %10 {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <i8, [spec: i1]>
    %12 = trunci %11#0 {handshake.bb = 1 : ui32, handshake.name = "trunci0"} : <i8, [spec: i1]> to <i7, [spec: i1]>
    %13 = mux %17#1 [%0, %trueResult_19] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i1, [spec: i1]>, [<f32, [spec: i1]>, <f32, [spec: i1]>] to <f32, [spec: i1]>
    %14 = buffer %13, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer1"} : <f32, [spec: i1]>
    %15:3 = fork [3] %14 {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <f32, [spec: i1]>
    %result, %index = control_merge [%2, %trueResult_21]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<[spec: i1]>, <[spec: i1]>] to <[spec: i1]>, <i1, [spec: i1]>
    %16:2 = fork [2] %result {handshake.bb = 1 : ui32, handshake.name = "fork4"} : <[spec: i1]>
    %17:2 = fork [2] %index {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <i1, [spec: i1]>
    %18 = source {handshake.bb = 1 : ui32, handshake.name = "source0"} : <[spec: i1]>
    %19 = constant %18 {handshake.bb = 1 : ui32, handshake.name = "constant6", value = -0.899999976 : f32} : <[spec: i1]>, <f32, [spec: i1]>
    %20 = source {handshake.bb = 1 : ui32, handshake.name = "source1"} : <[spec: i1]>
    %21 = constant %20 {handshake.bb = 1 : ui32, handshake.name = "constant7", value = 0.000000e+00 : f32} : <[spec: i1]>, <f32, [spec: i1]>
    %addressResult, %dataResult = load[%12] %outputs {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i7, [spec: i1]>, <f32>, <i7>, <f32, [spec: i1]>
    %22 = buffer %15#2, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer2"} : <f32, [spec: i1]>
    %23 = buffer %dataResult, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer3"} : <f32, [spec: i1]>
    %24 = mulf %23, %22 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "mulf0", internal_delay = "2_875333"} : <f32, [spec: i1]>
    %25 = mulf %15#1, %19 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "mulf1", internal_delay = "2_875333"} : <f32, [spec: i1]>
    %26 = buffer %25, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer4"} : <f32, [spec: i1]>
    %27 = addf %24, %26 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "addf0", internal_delay = "2_922000"} : <f32, [spec: i1]>
    %28 = cmpf ugt, %27, %21 {handshake.bb = 1 : ui32, handshake.name = "cmpf0", internal_delay = "0_000000"} : <f32, [spec: i1]>
    %dataOut, %SCSaveCtrl = spec_prebuffer1[%16#1] {fifoDepth = 30 : ui32, handshake.bb = 1 : ui32, handshake.name = "spec_prebuffer10"} : <[spec: i1]>, <i1, [spec: i1]>, <i3>
    %29:3 = fork [3] %SCSaveCtrl {handshake.bb = 1 : ui32, handshake.name = "fork9"} : <i3>
    %saveCtrl, %commitCtrl, %SCIsMisspec = spec_prebuffer2 %28 {fifoDepth = 30 : ui32, handshake.bb = 1 : ui32, handshake.name = "spec_prebuffer20"} : <i1, [spec: i1]>, <i1>, <i1>, <i1>
    %30:2 = fork [2] %commitCtrl {handshake.bb = 1 : ui32, handshake.name = "fork13"} : <i1>
    sink %saveCtrl {handshake.name = "sink0"} : <i1>
    %trueResult, %falseResult = cond_br %34#0, %30#0 {handshake.bb = 1 : ui32, handshake.name = "cond_br0"} : <i1>, <i1>
    %31:3 = fork [3] %falseResult {handshake.bb = 1 : ui32, handshake.name = "fork15"} : <i1>
    sink %trueResult {handshake.name = "sink1"} : <i1>
    %trueResult_1, %falseResult_2 = speculating_branch[%35#5] %35#6 {handshake.bb = 1 : ui32, handshake.name = "speculating_branch0"} : !handshake.channel<i1, [spec: i1]>, !handshake.channel<i1, [spec: i1]>, !handshake.channel<i1>, !handshake.channel<i1>
    sink %falseResult_2 {handshake.name = "sink3"} : <i1>
    %32 = buffer %trueResult_1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer5"} : <i1>
    %trueResult_3, %falseResult_4 = cond_br %32, %30#1 {handshake.bb = 1 : ui32, handshake.name = "cond_br1"} : <i1>, <i1>
    sink %falseResult_4 {handshake.name = "sink4"} : <i1>
    %33:3 = fork [3] %trueResult_3 {handshake.bb = 1 : ui32, handshake.name = "fork16"} : <i1>
    %trueResult_5, %falseResult_6 = speculating_branch[%79#2] %79#3 {handshake.bb = 1 : ui32, handshake.name = "speculating_branch1"} : !handshake.channel<i1, [spec: i1]>, !handshake.channel<i1, [spec: i1]>, !handshake.channel<i1>, !handshake.channel<i1>
    sink %falseResult_6 {handshake.name = "sink5"} : <i1>
    %34:2 = fork [2] %trueResult_5 {handshake.bb = 1 : ui32, handshake.name = "fork17"} : <i1>
    %trueResult_7, %falseResult_8 = cond_br %SCIsMisspec, %34#1 {handshake.bb = 1 : ui32, handshake.name = "cond_br8"} : <i1>, <i1>
    sink %falseResult_8 {handshake.name = "sink6"} : <i1>
    sink %trueResult_7 {handshake.name = "sink7"} : <i1>
    %35:8 = fork [8] %dataOut {handshake.bb = 1 : ui32, handshake.name = "fork18"} : <i1, [spec: i1]>
    %36 = spec_save_commit[%29#0] %11#1 {fifoDepth = 30 : ui32, handshake.bb = 1 : ui32, handshake.name = "spec_save_commit0"} : !handshake.channel<i8, [spec: i1]>, <i3>
    %trueResult_9, %falseResult_10 = cond_br %35#7, %36 {handshake.bb = 1 : ui32, handshake.name = "cond_br2", specv1_adaptor_inner_loop = true} : <i1, [spec: i1]>, <i8, [spec: i1]>
    %37 = spec_save_commit[%29#1] %15#0 {fifoDepth = 30 : ui32, handshake.bb = 1 : ui32, handshake.name = "spec_save_commit1"} : !handshake.channel<f32, [spec: i1]>, <i3>
    %trueResult_11, %falseResult_12 = cond_br %35#4, %37 {handshake.bb = 1 : ui32, handshake.name = "cond_br3", specv1_adaptor_inner_loop = true} : <i1, [spec: i1]>, <f32, [spec: i1]>
    %38 = spec_save_commit[%29#2] %16#0 {fifoDepth = 30 : ui32, handshake.bb = 1 : ui32, handshake.name = "spec_save_commit2"} : !handshake.control<[spec: i1]>, <i3>
    %trueResult_13, %falseResult_14 = cond_br %35#3, %38 {handshake.bb = 1 : ui32, handshake.name = "cond_br4", specv1_adaptor_inner_loop = true} : <i1, [spec: i1]>, <[spec: i1]>
    %39 = source {handshake.bb = 1 : ui32, handshake.name = "source2"} : <[spec: i1]>
    %40 = constant %39 {handshake.bb = 1 : ui32, handshake.name = "constant8", value = 1.100000e+00 : f32} : <[spec: i1]>, <f32, [spec: i1]>
    %41 = buffer %falseResult_12, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer8"} : <f32, [spec: i1]>
    %42 = mulf %41, %40 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "mulf2", internal_delay = "2_875333"} : <f32, [spec: i1]>
    %43:2 = fork [2] %trueResult_9 {handshake.bb = 1 : ui32, handshake.name = "fork5"} : <i8, [spec: i1]>
    %44 = buffer %43#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer10"} : <i8, [spec: i1]>
    %45 = trunci %44 {handshake.bb = 1 : ui32, handshake.name = "trunci1"} : <i8, [spec: i1]> to <i7, [spec: i1]>
    %46:2 = fork [2] %trueResult_11 {handshake.bb = 1 : ui32, handshake.name = "fork6"} : <f32, [spec: i1]>
    %47:2 = fork [2] %trueResult_13 {handshake.bb = 1 : ui32, handshake.name = "fork7"} : <[spec: i1]>
    %48 = buffer %47#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer12"} : <[spec: i1]>
    %49 = constant %48 {handshake.bb = 1 : ui32, handshake.name = "constant3", value = 1 : i2} : <[spec: i1]>, <i2, [spec: i1]>
    %50 = extsi %49 {handshake.bb = 1 : ui32, handshake.name = "extsi1"} : <i2, [spec: i1]> to <i32, [spec: i1]>
    %51 = source {handshake.bb = 1 : ui32, handshake.name = "source3"} : <[spec: i1]>
    %52 = constant %51 {handshake.bb = 1 : ui32, handshake.name = "constant9", value = 1.100000e+00 : f32} : <[spec: i1]>, <f32, [spec: i1]>
    %53 = spec_commit[%33#2] %45 {handshake.bb = 1 : ui32, handshake.name = "spec_commit3"} : !handshake.channel<i7, [spec: i1]>, !handshake.channel<i7>, <i1>
    %54 = buffer %46#0, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer11"} : <f32, [spec: i1]>
    %55 = spec_commit[%33#0] %54 {handshake.bb = 1 : ui32, handshake.name = "spec_commit4"} : !handshake.channel<f32, [spec: i1]>, !handshake.channel<f32>, <i1>
    %addressResult_15, %dataResult_16 = store[%53] %55 {handshake.bb = 1 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store0"} : <i7>, <f32>, <i7>, <f32>
    %56 = divf %46#1, %52 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "divf0", internal_delay = "3_812000"} : <f32, [spec: i1]>
    %57 = buffer %35#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer6"} : <i1, [spec: i1]>
    %58 = mux %57 [%42, %56] {handshake.bb = 1 : ui32, handshake.name = "mux2", specv1_adaptor_inner_loop = true} : <i1, [spec: i1]>, [<f32, [spec: i1]>, <f32, [spec: i1]>] to <f32, [spec: i1]>
    %59 = buffer %58, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer13"} : <f32, [spec: i1]>
    %60:2 = fork [2] %59 {handshake.bb = 1 : ui32, handshake.name = "fork8"} : <f32, [spec: i1]>
    %61 = buffer %falseResult_10, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer7"} : <i8, [spec: i1]>
    %62 = mux %35#0 [%61, %43#1] {handshake.bb = 1 : ui32, handshake.name = "mux3", specv1_adaptor_inner_loop = true} : <i1, [spec: i1]>, [<i8, [spec: i1]>, <i8, [spec: i1]>] to <i8, [spec: i1]>
    %63 = buffer %62, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer14"} : <i8, [spec: i1]>
    %64 = extsi %63 {handshake.bb = 1 : ui32, handshake.name = "extsi5"} : <i8, [spec: i1]> to <i9, [spec: i1]>
    %65 = buffer %falseResult_14, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer9"} : <[spec: i1]>
    %66 = mux %35#2 [%65, %47#1] {handshake.bb = 1 : ui32, handshake.name = "mux4", specv1_adaptor_inner_loop = true} : <i1, [spec: i1]>, [<[spec: i1]>, <[spec: i1]>] to <[spec: i1]>
    %67 = source {handshake.bb = 1 : ui32, handshake.name = "source4"} : <[spec: i1]>
    %68 = constant %67 {handshake.bb = 1 : ui32, handshake.name = "constant4", value = 1 : i2} : <[spec: i1]>, <i2, [spec: i1]>
    %69 = extsi %68 {handshake.bb = 1 : ui32, handshake.name = "extsi6"} : <i2, [spec: i1]> to <i9, [spec: i1]>
    %70 = source {handshake.bb = 1 : ui32, handshake.name = "source5"} : <[spec: i1]>
    %71 = constant %70 {handshake.bb = 1 : ui32, handshake.name = "constant12", value = 100 : i8} : <[spec: i1]>, <i8, [spec: i1]>
    %72 = extsi %71 {handshake.bb = 1 : ui32, handshake.name = "extsi7"} : <i8, [spec: i1]> to <i9, [spec: i1]>
    %73 = addf %60#0, %60#1 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "addf1", internal_delay = "2_922000"} : <f32, [spec: i1]>
    %74 = buffer %64, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer15"} : <i9, [spec: i1]>
    %75 = addi %74, %69 {handshake.bb = 1 : ui32, handshake.name = "addi0"} : <i9, [spec: i1]>
    %76:2 = fork [2] %75 {handshake.bb = 1 : ui32, handshake.name = "fork10"} : <i9, [spec: i1]>
    %77 = trunci %76#0 {handshake.bb = 1 : ui32, handshake.name = "trunci2"} : <i9, [spec: i1]> to <i8, [spec: i1]>
    %78 = cmpi ult, %76#1, %72 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i9, [spec: i1]>
    %79:5 = fork [5] %78 {handshake.bb = 1 : ui32, handshake.name = "fork19"} : <i1, [spec: i1]>
    %trueResult_17, %falseResult_18 = cond_br %79#4, %77 {handshake.bb = 1 : ui32, handshake.name = "cond_br5"} : <i1, [spec: i1]>, <i8, [spec: i1]>
    sink %falseResult_18 {handshake.name = "sink2"} : <i8, [spec: i1]>
    %trueResult_19, %falseResult_20 = cond_br %79#0, %73 {handshake.bb = 1 : ui32, handshake.name = "cond_br6"} : <i1, [spec: i1]>, <f32, [spec: i1]>
    %80 = buffer %66, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer16"} : <[spec: i1]>
    %81 = buffer %80, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer17"} : <[spec: i1]>
    %trueResult_21, %falseResult_22 = cond_br %79#1, %81 {handshake.bb = 1 : ui32, handshake.name = "cond_br7"} : <i1, [spec: i1]>, <[spec: i1]>
    %82:2 = fork [2] %falseResult_22 {handshake.bb = 2 : ui32, handshake.name = "fork12"} : <[spec: i1]>
    %83 = spec_commit[%31#2] %falseResult_20 {handshake.bb = 1 : ui32, handshake.name = "spec_commit5"} : !handshake.channel<f32, [spec: i1]>, !handshake.channel<f32>, <i1>
    end {handshake.bb = 2 : ui32, handshake.name = "end0"} %83, %memEnd_0, %memEnd, %1#1 : <f32>, <>, <>, <>
  }
}

