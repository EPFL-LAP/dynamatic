module {
  handshake.func @get_tanh(%arg0: memref<1000xf32>, %arg1: memref<1000xi32>, %arg2: !handshake.control<>, %arg3: !handshake.control<>, %arg4: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["A", "addr", "A_start", "addr_start", "start"], resNames = ["A_end", "addr_end", "end"]} {
    %0:3 = fork [3] %arg4 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork0"} : <>
    %outputs, %memEnd = mem_controller[%arg1 : memref<1000xi32>] %arg3 (%addressResult) %63#1 {connectedBlocks = [1 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller1"} :    (!handshake.channel<i10>) -> !handshake.channel<i32>
    %1:2 = lsq[%arg0 : memref<1000xf32>] (%arg2, %9#0, %addressResult_0, %46#0, %addressResult_12, %dataResult_13, %63#0)  {groupSizes = [1 : i32, 1 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "5": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "6": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "lsq1"} : (!handshake.control<>, !handshake.control<>, !handshake.channel<i10>, !handshake.control<>, !handshake.channel<i10>, !handshake.channel<f32>, !handshake.control<>) -> (!handshake.channel<f32>, !handshake.control<>)
    %2 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant0", value = false} : <>, <i1>
    %3 = extsi %2 {handshake.bb = 0 : ui32, handshake.name = "extsi3"} : <i1> to <i11>
    %4 = mux %index [%3, %trueResult_14] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i11>, <i11>] to <i11>
    %5 = buffer %4, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer0"} : <i11>
    %6 = buffer %5, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer1"} : <i11>
    %7:2 = fork [2] %6 {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <i11>
    %8 = trunci %7#0 {handshake.bb = 1 : ui32, handshake.name = "trunci0"} : <i11> to <i10>
    %result, %index = control_merge [%0#2, %trueResult_16]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %9:3 = lazy_fork [3] %result {handshake.bb = 1 : ui32, handshake.name = "lazy_fork0"} : <>
    %10 = buffer %9#2, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer3"} : <>
    %11 = buffer %10, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer4"} : <>
    %12 = constant %11 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"0": [1,inf], [0,inf], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "constant6", value = 1.000000e+00 : f32} : <>, <f32>
    %13:2 = fork [2] %12 {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <f32>
    %addressResult, %dataResult = load[%8] %outputs {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i10>, <i32>, <i10>, <i32>
    %14:2 = fork [2] %dataResult {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <i32>
    %15 = trunci %14#0 {handshake.bb = 1 : ui32, handshake.name = "trunci1"} : <i32> to <i10>
    %16 = trunci %14#1 {handshake.bb = 1 : ui32, handshake.name = "trunci2"} : <i32> to <i10>
    %17 = buffer %16, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer7"} : <i10>
    %addressResult_0, %dataResult_1 = load[%17] %1#0 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["store0", 0]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 0>, handshake.name = "load1"} : <i10>, <f32>, <i10>, <f32>
    %18:2 = fork [2] %dataResult_1 {handshake.bb = 1 : ui32, handshake.name = "fork4"} : <f32>
    %19 = cmpf oge, %18#0, %13#1 {fpu_impl = #handshake<fpu_impl flopoco>, handshake.bb = 1 : ui32, handshake.name = "cmpf0", internal_delay = "0_000000"} : <f32>
    %20 = buffer %19, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer8"} : <i1>
    %21:5 = fork [5] %20 {handshake.bb = 1 : ui32, handshake.name = "fork5"} : <i1>
    %trueResult, %falseResult = cond_br %22, %23 {handshake.bb = 1 : ui32, handshake.name = "cond_br2"} : <i1>, <f32>
    %22 = buffer %21#4, bufferType = FIFO_BREAK_NONE, numSlots = 3 {handshake.bb = 1 : ui32, handshake.name = "buffer5"} : <i1>
    %23 = buffer %13#0, bufferType = FIFO_BREAK_NONE, numSlots = 3 {handshake.bb = 1 : ui32, handshake.name = "buffer6"} : <f32>
    sink %falseResult {handshake.name = "sink0"} : <f32>
    %trueResult_2, %falseResult_3 = cond_br %21#1, %7#1 {handshake.bb = 1 : ui32, handshake.name = "cond_br3"} : <i1>, <i11>
    %trueResult_4, %falseResult_5 = cond_br %21#0, %15 {handshake.bb = 1 : ui32, handshake.name = "cond_br4"} : <i1>, <i10>
    %24 = buffer %9#1, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer2"} : <>
    %trueResult_6, %falseResult_7 = cond_br %21#3, %24 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,inf], [1,inf], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "cond_br5"} : <i1>, <>
    %trueResult_8, %falseResult_9 = cond_br %21#2, %18#1 {handshake.bb = 1 : ui32, handshake.name = "cond_br6"} : <i1>, <f32>
    sink %trueResult_8 {handshake.name = "sink1"} : <f32>
    %25:5 = fork [5] %falseResult_9 {handshake.bb = 2 : ui32, handshake.name = "fork6"} : <f32>
    %26 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %27 = constant %26 {handshake.bb = 2 : ui32, handshake.name = "constant7", value = 3.70476198 : f32} : <>, <f32>
    %28 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %29 = constant %28 {handshake.bb = 2 : ui32, handshake.name = "constant8", value = 19.5238094 : f32} : <>, <f32>
    %30 = mulf %25#3, %25#4 {fastmath = #arith.fastmath<none>, fpu_impl = #handshake<fpu_impl flopoco>, handshake.bb = 2 : ui32, handshake.name = "mulf0", internal_delay = "2_875333"} : <f32>
    %31 = addf %30, %29 {fastmath = #arith.fastmath<none>, fpu_impl = #handshake<fpu_impl flopoco>, handshake.bb = 2 : ui32, handshake.name = "addf0", internal_delay = "2_922000"} : <f32>
    %32 = mulf %31, %33 {fastmath = #arith.fastmath<none>, fpu_impl = #handshake<fpu_impl flopoco>, handshake.bb = 2 : ui32, handshake.name = "mulf1", internal_delay = "2_875333"} : <f32>
    %33 = buffer %25#2, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer15"} : <f32>
    %34 = mulf %32, %35 {fastmath = #arith.fastmath<none>, fpu_impl = #handshake<fpu_impl flopoco>, handshake.bb = 2 : ui32, handshake.name = "mulf2", internal_delay = "2_875333"} : <f32>
    %35 = buffer %25#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer16"} : <f32>
    %36 = addf %34, %27 {fastmath = #arith.fastmath<none>, fpu_impl = #handshake<fpu_impl flopoco>, handshake.bb = 2 : ui32, handshake.name = "addf1", internal_delay = "2_922000"} : <f32>
    %37 = mulf %36, %38 {fastmath = #arith.fastmath<none>, fpu_impl = #handshake<fpu_impl flopoco>, handshake.bb = 2 : ui32, handshake.name = "mulf3", internal_delay = "2_875333"} : <f32>
    %38 = buffer %25#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer17"} : <f32>
    %39 = mux %40 [%trueResult, %37] {handshake.bb = 3 : ui32, handshake.name = "mux1"} : <i1>, [<f32>, <f32>] to <f32>
    %40 = buffer %45#2, bufferType = FIFO_BREAK_NONE, numSlots = 4 {handshake.bb = 3 : ui32, handshake.name = "buffer18"} : <i1>
    %41 = mux %45#1 [%trueResult_2, %falseResult_3] {handshake.bb = 3 : ui32, handshake.name = "mux2"} : <i1>, [<i11>, <i11>] to <i11>
    %42 = buffer %41, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer10"} : <i11>
    %43 = extsi %42 {handshake.bb = 3 : ui32, handshake.name = "extsi4"} : <i11> to <i12>
    %44 = mux %45#0 [%trueResult_4, %falseResult_5] {handshake.bb = 3 : ui32, handshake.name = "mux3"} : <i1>, [<i10>, <i10>] to <i10>
    %result_10, %index_11 = control_merge [%trueResult_6, %falseResult_7]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>, <>] to <>, <i1>
    %45:3 = fork [3] %index_11 {handshake.bb = 3 : ui32, handshake.name = "fork7"} : <i1>
    %46:2 = lazy_fork [2] %result_10 {handshake.bb = 3 : ui32, handshake.name = "lazy_fork1"} : <>
    %47 = source {handshake.bb = 3 : ui32, handshake.name = "source2"} : <>
    %48 = constant %47 {handshake.bb = 3 : ui32, handshake.name = "constant1", value = 1000 : i11} : <>, <i11>
    %49 = extsi %48 {handshake.bb = 3 : ui32, handshake.name = "extsi5"} : <i11> to <i12>
    %50 = source {handshake.bb = 3 : ui32, handshake.name = "source3"} : <>
    %51 = constant %50 {handshake.bb = 3 : ui32, handshake.name = "constant3", value = 1 : i2} : <>, <i2>
    %52 = extsi %51 {handshake.bb = 3 : ui32, handshake.name = "extsi6"} : <i2> to <i12>
    %53 = buffer %39, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer9"} : <f32>
    %54 = buffer %44, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer12"} : <i10>
    %addressResult_12, %dataResult_13 = store[%54] %53 {handshake.bb = 3 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["load1", 0], ["store0", 0]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 1>, handshake.name = "store0"} : <i10>, <f32>, <i10>, <f32>
    %55 = buffer %43, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer11"} : <i12>
    %56 = addi %55, %52 {handshake.bb = 3 : ui32, handshake.name = "addi0"} : <i12>
    %57 = buffer %56, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer14"} : <i12>
    %58:2 = fork [2] %57 {handshake.bb = 3 : ui32, handshake.name = "fork8"} : <i12>
    %59 = trunci %58#0 {handshake.bb = 3 : ui32, handshake.name = "trunci3"} : <i12> to <i11>
    %60 = cmpi ult, %58#1, %49 {handshake.bb = 3 : ui32, handshake.name = "cmpi0"} : <i12>
    %61:2 = fork [2] %60 {handshake.bb = 3 : ui32, handshake.name = "fork9"} : <i1>
    %trueResult_14, %falseResult_15 = cond_br %61#0, %59 {handshake.bb = 3 : ui32, handshake.name = "cond_br7"} : <i1>, <i11>
    sink %falseResult_15 {handshake.name = "sink3"} : <i11>
    %62 = buffer %46#1, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer13"} : <>
    %trueResult_16, %falseResult_17 = cond_br %61#1, %62 {handshake.bb = 3 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,inf], [1,inf], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "cond_br8"} : <i1>, <>
    %63:2 = fork [2] %falseResult_17 {handshake.bb = 4 : ui32, handshake.name = "fork10"} : <>
    end {handshake.bb = 4 : ui32, handshake.name = "end0"} %1#1, %memEnd, %0#1 : <>, <>, <>
  }
}

