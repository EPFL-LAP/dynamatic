module {
  handshake.func @get_tanh(%arg0: memref<1000xf32>, %arg1: memref<1000xi32>, %arg2: !handshake.control<>, %arg3: !handshake.control<>, %arg4: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["A", "addr", "A_start", "addr_start", "start"], resNames = ["A_end", "addr_end", "end"]} {
    %0:3 = fork [3] %arg4 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork0"} : <>
    %outputs, %memEnd = mem_controller[%arg1 : memref<1000xi32>] %arg3 (%addressResult) %72#1 {connectedBlocks = [1 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller1"} :    (!handshake.channel<i10>) -> !handshake.channel<i32>
    %1:2 = lsq[%arg0 : memref<1000xf32>] (%arg2, %11#0, %addressResult_0, %55#0, %addressResult_14, %dataResult_15, %72#0)  {groupSizes = [1 : i32, 1 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "5": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "6": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "lsq1"} : (!handshake.control<>, !handshake.control<>, !handshake.channel<i10>, !handshake.control<>, !handshake.channel<i10>, !handshake.channel<f32>, !handshake.control<>) -> (!handshake.channel<f32>, !handshake.control<>)
    %2 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant0", value = false} : <>, <i1>
    %3 = br %2 {handshake.bb = 0 : ui32, handshake.name = "br1"} : <i1>
    %4 = extsi %3 {handshake.bb = 0 : ui32, handshake.name = "extsi3"} : <i1> to <i11>
    %5 = br %0#2 {handshake.bb = 0 : ui32, handshake.name = "br3"} : <>
    %6 = mux %index [%4, %trueResult_16] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i11>, <i11>] to <i11>
    %7 = buffer %6, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer0"} : <i11>
    %8 = buffer %7, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer1"} : <i11>
    %9:2 = fork [2] %8 {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <i11>
    %10 = trunci %9#0 {handshake.bb = 1 : ui32, handshake.name = "trunci0"} : <i11> to <i10>
    %result, %index = control_merge [%5, %trueResult_18]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %11:3 = lazy_fork [3] %result {handshake.bb = 1 : ui32, handshake.name = "lazy_fork0"} : <>
    %12 = buffer %11#2, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer3"} : <>
    %13 = buffer %12, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer4"} : <>
    %14 = constant %13 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"0": [1,inf], [0,inf], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "constant6", value = 1.000000e+00 : f32} : <>, <f32>
    %15:2 = fork [2] %14 {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <f32>
    %addressResult, %dataResult = load[%10] %outputs {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i10>, <i32>, <i10>, <i32>
    %16:2 = fork [2] %dataResult {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <i32>
    %17 = trunci %16#0 {handshake.bb = 1 : ui32, handshake.name = "trunci1"} : <i32> to <i10>
    %18 = trunci %16#1 {handshake.bb = 1 : ui32, handshake.name = "trunci2"} : <i32> to <i10>
    %19 = buffer %18, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer7"} : <i10>
    %addressResult_0, %dataResult_1 = load[%19] %1#0 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["store0", 0]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 0>, handshake.name = "load1"} : <i10>, <f32>, <i10>, <f32>
    %20:2 = fork [2] %dataResult_1 {handshake.bb = 1 : ui32, handshake.name = "fork4"} : <f32>
    %21 = cmpf oge, %20#0, %15#1 {fpu_impl = #handshake<fpu_impl flopoco>, handshake.bb = 1 : ui32, handshake.name = "cmpf0", internal_delay = "0_000000"} : <f32>
    %22 = buffer %21, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer8"} : <i1>
    %23:5 = fork [5] %22 {handshake.bb = 1 : ui32, handshake.name = "fork5"} : <i1>
    %trueResult, %falseResult = cond_br %24, %25 {handshake.bb = 1 : ui32, handshake.name = "cond_br2"} : <i1>, <f32>
    %24 = buffer %23#4, bufferType = FIFO_BREAK_NONE, numSlots = 3 {handshake.bb = 1 : ui32, handshake.name = "buffer5"} : <i1>
    %25 = buffer %15#0, bufferType = FIFO_BREAK_NONE, numSlots = 3 {handshake.bb = 1 : ui32, handshake.name = "buffer6"} : <f32>
    sink %falseResult {handshake.name = "sink0"} : <f32>
    %trueResult_2, %falseResult_3 = cond_br %23#1, %9#1 {handshake.bb = 1 : ui32, handshake.name = "cond_br3"} : <i1>, <i11>
    %trueResult_4, %falseResult_5 = cond_br %23#0, %17 {handshake.bb = 1 : ui32, handshake.name = "cond_br4"} : <i1>, <i10>
    %26 = buffer %11#1, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer2"} : <>
    %trueResult_6, %falseResult_7 = cond_br %23#3, %26 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,inf], [1,inf], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "cond_br5"} : <i1>, <>
    %trueResult_8, %falseResult_9 = cond_br %23#2, %20#1 {handshake.bb = 1 : ui32, handshake.name = "cond_br6"} : <i1>, <f32>
    sink %trueResult_8 {handshake.name = "sink1"} : <f32>
    %27 = merge %falseResult_3 {handshake.bb = 2 : ui32, handshake.name = "merge0"} : <i11>
    %28 = merge %falseResult_5 {handshake.bb = 2 : ui32, handshake.name = "merge1"} : <i10>
    %29 = merge %falseResult_9 {handshake.bb = 2 : ui32, handshake.name = "merge2"} : <f32>
    %30:5 = fork [5] %29 {handshake.bb = 2 : ui32, handshake.name = "fork6"} : <f32>
    %result_10, %index_11 = control_merge [%falseResult_7]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>] to <>, <i1>
    sink %index_11 {handshake.name = "sink2"} : <i1>
    %31 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %32 = constant %31 {handshake.bb = 2 : ui32, handshake.name = "constant7", value = 3.70476198 : f32} : <>, <f32>
    %33 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %34 = constant %33 {handshake.bb = 2 : ui32, handshake.name = "constant8", value = 19.5238094 : f32} : <>, <f32>
    %35 = mulf %30#3, %30#4 {fastmath = #arith.fastmath<none>, fpu_impl = #handshake<fpu_impl flopoco>, handshake.bb = 2 : ui32, handshake.name = "mulf0", internal_delay = "2_875333"} : <f32>
    %36 = addf %35, %34 {fastmath = #arith.fastmath<none>, fpu_impl = #handshake<fpu_impl flopoco>, handshake.bb = 2 : ui32, handshake.name = "addf0", internal_delay = "2_922000"} : <f32>
    %37 = mulf %36, %38 {fastmath = #arith.fastmath<none>, fpu_impl = #handshake<fpu_impl flopoco>, handshake.bb = 2 : ui32, handshake.name = "mulf1", internal_delay = "2_875333"} : <f32>
    %38 = buffer %30#2, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer15"} : <f32>
    %39 = mulf %37, %40 {fastmath = #arith.fastmath<none>, fpu_impl = #handshake<fpu_impl flopoco>, handshake.bb = 2 : ui32, handshake.name = "mulf2", internal_delay = "2_875333"} : <f32>
    %40 = buffer %30#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer16"} : <f32>
    %41 = addf %39, %32 {fastmath = #arith.fastmath<none>, fpu_impl = #handshake<fpu_impl flopoco>, handshake.bb = 2 : ui32, handshake.name = "addf1", internal_delay = "2_922000"} : <f32>
    %42 = mulf %41, %43 {fastmath = #arith.fastmath<none>, fpu_impl = #handshake<fpu_impl flopoco>, handshake.bb = 2 : ui32, handshake.name = "mulf3", internal_delay = "2_875333"} : <f32>
    %43 = buffer %30#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer17"} : <f32>
    %44 = br %42 {handshake.bb = 2 : ui32, handshake.name = "br4"} : <f32>
    %45 = br %27 {handshake.bb = 2 : ui32, handshake.name = "br5"} : <i11>
    %46 = br %28 {handshake.bb = 2 : ui32, handshake.name = "br6"} : <i10>
    %47 = br %result_10 {handshake.bb = 2 : ui32, handshake.name = "br7"} : <>
    %48 = mux %49 [%trueResult, %44] {handshake.bb = 3 : ui32, handshake.name = "mux1"} : <i1>, [<f32>, <f32>] to <f32>
    %49 = buffer %54#2, bufferType = FIFO_BREAK_NONE, numSlots = 4 {handshake.bb = 3 : ui32, handshake.name = "buffer18"} : <i1>
    %50 = mux %54#1 [%trueResult_2, %45] {handshake.bb = 3 : ui32, handshake.name = "mux2"} : <i1>, [<i11>, <i11>] to <i11>
    %51 = buffer %50, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer10"} : <i11>
    %52 = extsi %51 {handshake.bb = 3 : ui32, handshake.name = "extsi4"} : <i11> to <i12>
    %53 = mux %54#0 [%trueResult_4, %46] {handshake.bb = 3 : ui32, handshake.name = "mux3"} : <i1>, [<i10>, <i10>] to <i10>
    %result_12, %index_13 = control_merge [%trueResult_6, %47]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>, <>] to <>, <i1>
    %54:3 = fork [3] %index_13 {handshake.bb = 3 : ui32, handshake.name = "fork7"} : <i1>
    %55:2 = lazy_fork [2] %result_12 {handshake.bb = 3 : ui32, handshake.name = "lazy_fork1"} : <>
    %56 = source {handshake.bb = 3 : ui32, handshake.name = "source2"} : <>
    %57 = constant %56 {handshake.bb = 3 : ui32, handshake.name = "constant1", value = 1000 : i11} : <>, <i11>
    %58 = extsi %57 {handshake.bb = 3 : ui32, handshake.name = "extsi5"} : <i11> to <i12>
    %59 = source {handshake.bb = 3 : ui32, handshake.name = "source3"} : <>
    %60 = constant %59 {handshake.bb = 3 : ui32, handshake.name = "constant3", value = 1 : i2} : <>, <i2>
    %61 = extsi %60 {handshake.bb = 3 : ui32, handshake.name = "extsi6"} : <i2> to <i12>
    %62 = buffer %48, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer9"} : <f32>
    %63 = buffer %53, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer12"} : <i10>
    %addressResult_14, %dataResult_15 = store[%63] %62 {handshake.bb = 3 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["load1", 0], ["store0", 0]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 1>, handshake.name = "store0"} : <i10>, <f32>, <i10>, <f32>
    %64 = buffer %52, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer11"} : <i12>
    %65 = addi %64, %61 {handshake.bb = 3 : ui32, handshake.name = "addi0"} : <i12>
    %66 = buffer %65, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer14"} : <i12>
    %67:2 = fork [2] %66 {handshake.bb = 3 : ui32, handshake.name = "fork8"} : <i12>
    %68 = trunci %67#0 {handshake.bb = 3 : ui32, handshake.name = "trunci3"} : <i12> to <i11>
    %69 = cmpi ult, %67#1, %58 {handshake.bb = 3 : ui32, handshake.name = "cmpi0"} : <i12>
    %70:2 = fork [2] %69 {handshake.bb = 3 : ui32, handshake.name = "fork9"} : <i1>
    %trueResult_16, %falseResult_17 = cond_br %70#0, %68 {handshake.bb = 3 : ui32, handshake.name = "cond_br7"} : <i1>, <i11>
    sink %falseResult_17 {handshake.name = "sink3"} : <i11>
    %71 = buffer %55#1, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer13"} : <>
    %trueResult_18, %falseResult_19 = cond_br %70#1, %71 {handshake.bb = 3 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,inf], [1,inf], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "cond_br8"} : <i1>, <>
    %result_20, %index_21 = control_merge [%falseResult_19]  {handshake.bb = 4 : ui32, handshake.name = "control_merge3"} : [<>] to <>, <i1>
    sink %index_21 {handshake.name = "sink4"} : <i1>
    %72:2 = fork [2] %result_20 {handshake.bb = 4 : ui32, handshake.name = "fork10"} : <>
    end {handshake.bb = 4 : ui32, handshake.name = "end0"} %1#1, %memEnd, %0#1 : <>, <>, <>
  }
}

