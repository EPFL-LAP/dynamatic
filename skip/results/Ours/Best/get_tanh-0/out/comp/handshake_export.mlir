module {
  handshake.func @get_tanh(%arg0: memref<1000xf32>, %arg1: memref<1000xi32>, %arg2: !handshake.control<>, %arg3: !handshake.control<>, %arg4: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["A", "addr", "A_start", "addr_start", "start"], cfg.edges = "[0,1][2,3][1,3,2,cmpf0][3,1,4,cmpi0]", resNames = ["A_end", "addr_end", "end"]} {
    %0:4 = fork [4] %arg4 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork0"} : <>
    %outputs, %memEnd = mem_controller[%arg1 : memref<1000xi32>] %arg3 (%addressResult) %60#1 {connectedBlocks = [1 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller1"} :    (!handshake.channel<i10>) -> !handshake.channel<i32>
    %outputs_0:2, %memEnd_1 = mem_controller[%arg0 : memref<1000xf32>] %arg2 (%addressResult_2, %44, %addressResult_16, %dataResult_17) %60#0 {connectedBlocks = [1 : i32, 3 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "5": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller2"} :    (!handshake.channel<i10>, !handshake.channel<i32>, !handshake.channel<i10>, !handshake.channel<f32>) -> (!handshake.channel<f32>, !handshake.control<>)
    %1 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant1", value = false} : <>, <i1>
    %2 = extsi %1 {handshake.bb = 0 : ui32, handshake.name = "extsi4"} : <i1> to <i11>
    %3 = mux %4 [%0#2, %trueResult_12] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux4"} : <i1>, [<>, <>] to <>
    %4 = init %59#2 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "init0"} : <i1>
    %5 = mux %index [%2, %trueResult_18] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i11>, <i11>] to <i11>
    %6 = buffer %5, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer2"} : <i11>
    %7:2 = fork [2] %6 {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <i11>
    %8 = trunci %7#0 {handshake.bb = 1 : ui32, handshake.name = "trunci0"} : <i11> to <i10>
    %9 = buffer %trueResult_20, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer11"} : <>
    %result, %index = control_merge [%0#3, %9]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %10:2 = fork [2] %result {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <>
    %11 = constant %10#1 {handshake.bb = 1 : ui32, handshake.name = "constant6", value = 1.000000e+00 : f32} : <>, <f32>
    %12:2 = fork [2] %11 {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <f32>
    %addressResult, %dataResult = load[%8] %outputs {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i10>, <i32>, <i10>, <i32>
    %13:2 = fork [2] %dataResult {handshake.bb = 1 : ui32, handshake.name = "fork4"} : <i32>
    %14 = trunci %13#0 {handshake.bb = 1 : ui32, handshake.name = "trunci1"} : <i32> to <i10>
    %15 = buffer %3, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer1"} : <>
    %16 = gate %13#1, %15 {handshake.bb = 1 : ui32, handshake.name = "gate0"} : <i32>, !handshake.control<> to <i32>
    %17 = trunci %16 {handshake.bb = 1 : ui32, handshake.name = "trunci2"} : <i32> to <i10>
    %addressResult_2, %dataResult_3 = load[%17] %outputs_0#0 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["store1", 0, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load3"} : <i10>, <f32>, <i10>, <f32>
    %18:2 = fork [2] %dataResult_3 {handshake.bb = 1 : ui32, handshake.name = "fork5"} : <f32>
    %19 = cmpf oge, %18#0, %12#1 {fpu_impl = #handshake<fpu_impl flopoco>, handshake.bb = 1 : ui32, handshake.name = "cmpf0", internal_delay = "0_000000"} : <f32>
    %20 = buffer %19, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer4"} : <i1>
    %21:5 = fork [5] %20 {handshake.bb = 1 : ui32, handshake.name = "fork6"} : <i1>
    %trueResult, %falseResult = cond_br %21#4, %12#0 {handshake.bb = 1 : ui32, handshake.name = "cond_br2"} : <i1>, <f32>
    sink %falseResult {handshake.name = "sink0"} : <f32>
    %22 = buffer %7#1, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer3"} : <i11>
    %trueResult_4, %falseResult_5 = cond_br %21#1, %22 {handshake.bb = 1 : ui32, handshake.name = "cond_br3"} : <i1>, <i11>
    %trueResult_6, %falseResult_7 = cond_br %21#0, %14 {handshake.bb = 1 : ui32, handshake.name = "cond_br4"} : <i1>, <i10>
    %trueResult_8, %falseResult_9 = cond_br %21#3, %10#0 {handshake.bb = 1 : ui32, handshake.name = "cond_br5"} : <i1>, <>
    %trueResult_10, %falseResult_11 = cond_br %21#2, %18#1 {handshake.bb = 1 : ui32, handshake.name = "cond_br6"} : <i1>, <f32>
    sink %trueResult_10 {handshake.name = "sink1"} : <f32>
    %23:5 = fork [5] %falseResult_11 {handshake.bb = 2 : ui32, handshake.name = "fork7"} : <f32>
    %24 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %25 = constant %24 {handshake.bb = 2 : ui32, handshake.name = "constant7", value = 3.70476198 : f32} : <>, <f32>
    %26 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %27 = constant %26 {handshake.bb = 2 : ui32, handshake.name = "constant8", value = 19.5238094 : f32} : <>, <f32>
    %28 = mulf %23#3, %23#4 {fastmath = #arith.fastmath<none>, fpu_impl = #handshake<fpu_impl flopoco>, handshake.bb = 2 : ui32, handshake.name = "mulf0", internal_delay = "2_875333"} : <f32>
    %29 = addf %28, %27 {fastmath = #arith.fastmath<none>, fpu_impl = #handshake<fpu_impl flopoco>, handshake.bb = 2 : ui32, handshake.name = "addf0", internal_delay = "2_922000"} : <f32>
    %30 = mulf %29, %23#2 {fastmath = #arith.fastmath<none>, fpu_impl = #handshake<fpu_impl flopoco>, handshake.bb = 2 : ui32, handshake.name = "mulf1", internal_delay = "2_875333"} : <f32>
    %31 = mulf %30, %23#1 {fastmath = #arith.fastmath<none>, fpu_impl = #handshake<fpu_impl flopoco>, handshake.bb = 2 : ui32, handshake.name = "mulf2", internal_delay = "2_875333"} : <f32>
    %32 = addf %31, %25 {fastmath = #arith.fastmath<none>, fpu_impl = #handshake<fpu_impl flopoco>, handshake.bb = 2 : ui32, handshake.name = "addf1", internal_delay = "2_922000"} : <f32>
    %33 = mulf %32, %23#0 {fastmath = #arith.fastmath<none>, fpu_impl = #handshake<fpu_impl flopoco>, handshake.bb = 2 : ui32, handshake.name = "mulf3", internal_delay = "2_875333"} : <f32>
    %34 = buffer %falseResult_7, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer5"} : <i10>
    %trueResult_12, %falseResult_13 = cond_br %59#1, %51 {handshake.bb = 3 : ui32, handshake.name = "cond_br1"} : <i1>, <>
    sink %falseResult_13 {handshake.name = "sink3"} : <>
    %35 = mux %41#2 [%trueResult, %33] {handshake.bb = 3 : ui32, handshake.name = "mux1"} : <i1>, [<f32>, <f32>] to <f32>
    %36 = buffer %falseResult_5, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer6"} : <i11>
    %37 = mux %41#1 [%trueResult_4, %36] {handshake.bb = 3 : ui32, handshake.name = "mux2"} : <i1>, [<i11>, <i11>] to <i11>
    %38 = buffer %37, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer8"} : <i11>
    %39 = extsi %38 {handshake.bb = 3 : ui32, handshake.name = "extsi5"} : <i11> to <i12>
    %40 = mux %41#0 [%trueResult_6, %34] {handshake.bb = 3 : ui32, handshake.name = "mux3"} : <i1>, [<i10>, <i10>] to <i10>
    %result_14, %index_15 = control_merge [%trueResult_8, %falseResult_9]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>, <>] to <>, <i1>
    %41:3 = fork [3] %index_15 {handshake.bb = 3 : ui32, handshake.name = "fork8"} : <i1>
    %42:2 = fork [2] %result_14 {handshake.bb = 3 : ui32, handshake.name = "fork9"} : <>
    %43 = constant %42#0 {handshake.bb = 3 : ui32, handshake.name = "constant3", value = 1 : i2} : <>, <i2>
    %44 = extsi %43 {handshake.bb = 3 : ui32, handshake.name = "extsi1"} : <i2> to <i32>
    %45 = source {handshake.bb = 3 : ui32, handshake.name = "source2"} : <>
    %46 = constant %45 {handshake.bb = 3 : ui32, handshake.name = "constant4", value = 1000 : i11} : <>, <i11>
    %47 = extsi %46 {handshake.bb = 3 : ui32, handshake.name = "extsi6"} : <i11> to <i12>
    %48 = source {handshake.bb = 3 : ui32, handshake.name = "source3"} : <>
    %49 = constant %48 {handshake.bb = 3 : ui32, handshake.name = "constant5", value = 1 : i2} : <>, <i2>
    %50 = extsi %49 {handshake.bb = 3 : ui32, handshake.name = "extsi7"} : <i2> to <i12>
    %51 = buffer %doneResult, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer0"} : <>
    %52 = buffer %35, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer7"} : <f32>
    %53 = buffer %40, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer9"} : <i10>
    %addressResult_16, %dataResult_17, %doneResult = store[%53] %52 %outputs_0#1 {handshake.bb = 3 : ui32, handshake.bufProps = #handshake<bufProps{"2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["load3", 0, false], ["store1", 0, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store1"} : <i10>, <f32>, <>, <i10>, <f32>, <>
    %54 = addi %39, %50 {handshake.bb = 3 : ui32, handshake.name = "addi0"} : <i12>
    %55 = buffer %54, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer10"} : <i12>
    %56:2 = fork [2] %55 {handshake.bb = 3 : ui32, handshake.name = "fork10"} : <i12>
    %57 = trunci %56#0 {handshake.bb = 3 : ui32, handshake.name = "trunci3"} : <i12> to <i11>
    %58 = cmpi ult, %56#1, %47 {handshake.bb = 3 : ui32, handshake.name = "cmpi0"} : <i12>
    %59:4 = fork [4] %58 {handshake.bb = 3 : ui32, handshake.name = "fork11"} : <i1>
    %trueResult_18, %falseResult_19 = cond_br %59#0, %57 {handshake.bb = 3 : ui32, handshake.name = "cond_br7"} : <i1>, <i11>
    sink %falseResult_19 {handshake.name = "sink4"} : <i11>
    %trueResult_20, %falseResult_21 = cond_br %59#3, %42#1 {handshake.bb = 3 : ui32, handshake.name = "cond_br8"} : <i1>, <>
    %60:2 = fork [2] %falseResult_21 {handshake.bb = 4 : ui32, handshake.name = "fork12"} : <>
    end {handshake.bb = 4 : ui32, handshake.name = "end0"} %memEnd_1, %memEnd, %0#1 : <>, <>, <>
  }
}

