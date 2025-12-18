module {
  handshake.func @get_tanh(%arg0: memref<1000xf32>, %arg1: memref<1000xi32>, %arg2: !handshake.control<>, %arg3: !handshake.control<>, %arg4: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["A", "addr", "A_start", "addr_start", "start"], resNames = ["A_end", "addr_end", "end"]} {
    %0:3 = fork [3] %arg4 {handshake.bb = 0 : ui32, handshake.name = "fork0"} : <>
    %outputs, %memEnd = mem_controller[%arg1 : memref<1000xi32>] %arg3 (%addressResult) %78#1 {connectedBlocks = [1 : i32], handshake.name = "mem_controller1"} :    (!handshake.channel<i10>) -> !handshake.channel<i32>
    %1:2 = lsq[%arg0 : memref<1000xf32>] (%arg2, %10#0, %addressResult_0, %62#0, %addressResult_14, %dataResult_15, %78#0)  {groupSizes = [1 : i32, 1 : i32], handshake.name = "lsq1"} : (!handshake.control<>, !handshake.control<>, !handshake.channel<i10>, !handshake.control<>, !handshake.channel<i10>, !handshake.channel<f32>, !handshake.control<>) -> (!handshake.channel<f32>, !handshake.control<>)
    %2 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant0", value = false} : <>, <i1>
    %3 = br %2 {handshake.bb = 0 : ui32, handshake.name = "br1"} : <i1>
    %4 = extsi %3 {handshake.bb = 0 : ui32, handshake.name = "extsi3"} : <i1> to <i11>
    %5 = br %0#2 {handshake.bb = 0 : ui32, handshake.name = "br3"} : <>
    %6 = mux %index [%4, %trueResult_16] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i11>, <i11>] to <i11>
    %7:2 = fork [2] %6 {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <i11>
    %8 = trunci %7#0 {handshake.bb = 1 : ui32, handshake.name = "trunci0"} : <i11> to <i10>
    %result, %index = control_merge [%5, %trueResult_18]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %10:3 = lazy_fork [3] %result {handshake.bb = 1 : ui32, handshake.name = "lazy_fork0"} : <>
    %11 = constant %10#2 {handshake.bb = 1 : ui32, handshake.name = "constant6", value = 1.000000e+00 : f32} : <>, <f32>
    %12:2 = fork [2] %11 {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <f32>
    %addressResult, %dataResult = load[%8] %outputs {handshake.bb = 1 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i10>, <i32>, <i10>, <i32>
    %13:2 = fork [2] %dataResult {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <i32>
    %14 = trunci %13#0 {handshake.bb = 1 : ui32, handshake.name = "trunci1"} : <i32> to <i10>
    %16 = trunci %13#1 {handshake.bb = 1 : ui32, handshake.name = "trunci2"} : <i32> to <i10>
    %addressResult_0, %dataResult_1 = load[%16] %1#0 {handshake.bb = 1 : ui32, handshake.deps = #handshake<deps[["store0", 0]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 0>, handshake.name = "load1"} : <i10>, <f32>, <i10>, <f32>
    %18:2 = fork [2] %dataResult_1 {handshake.bb = 1 : ui32, handshake.name = "fork4"} : <f32>
    %19 = cmpf oge, %18#0, %12#1 {handshake.bb = 1 : ui32, handshake.name = "cmpf0"} : <f32>
    %22:5 = fork [5] %19 {handshake.bb = 1 : ui32, handshake.name = "fork5"} : <i1>
    %trueResult, %falseResult = cond_br %23, %24 {handshake.bb = 1 : ui32, handshake.name = "cond_br2"} : <i1>, <f32>
    %23 = buffer %22#4, bufferType = FIFO_BREAK_NONE, numSlots = 3 {handshake.bb = 1 : ui32, handshake.name = "buffer5"} : <i1>
    %24 = buffer %12#0, bufferType = FIFO_BREAK_NONE, numSlots = 3 {handshake.bb = 1 : ui32, handshake.name = "buffer6"} : <f32>
    sink %falseResult {handshake.name = "sink0"} : <f32>
    %trueResult_2, %falseResult_3 = cond_br %22#1, %7#1 {handshake.bb = 1 : ui32, handshake.name = "cond_br3"} : <i1>, <i11>
    %trueResult_4, %falseResult_5 = cond_br %22#0, %14 {handshake.bb = 1 : ui32, handshake.name = "cond_br4"} : <i1>, <i10>
    %trueResult_6, %falseResult_7 = cond_br %22#3, %10#1 {handshake.bb = 1 : ui32, handshake.name = "cond_br5"} : <i1>, <>
    %trueResult_8, %falseResult_9 = cond_br %22#2, %18#1 {handshake.bb = 1 : ui32, handshake.name = "cond_br6"} : <i1>, <f32>
    sink %trueResult_8 {handshake.name = "sink1"} : <f32>
    %31 = merge %falseResult_3 {handshake.bb = 2 : ui32, handshake.name = "merge0"} : <i11>
    %32 = merge %falseResult_5 {handshake.bb = 2 : ui32, handshake.name = "merge1"} : <i10>
    %33 = merge %falseResult_9 {handshake.bb = 2 : ui32, handshake.name = "merge2"} : <f32>
    %34:5 = fork [5] %33 {handshake.bb = 2 : ui32, handshake.name = "fork6"} : <f32>
    %result_10, %index_11 = control_merge [%falseResult_7]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>] to <>, <i1>
    sink %index_11 {handshake.name = "sink2"} : <i1>
    %35 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %36 = constant %35 {handshake.bb = 2 : ui32, handshake.name = "constant7", value = 3.70476198 : f32} : <>, <f32>
    %37 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %38 = constant %37 {handshake.bb = 2 : ui32, handshake.name = "constant8", value = 19.5238094 : f32} : <>, <f32>
    %39 = mulf %34#3, %34#4 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "mulf0"} : <f32>
    %42 = addf %39, %38 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "addf0"} : <f32>
    %43 = mulf %42, %44 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "mulf1"} : <f32>
    %44 = buffer %34#2, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer15"} : <f32>
    %45 = mulf %43, %46 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "mulf2"} : <f32>
    %46 = buffer %34#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer16"} : <f32>
    %47 = addf %45, %36 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "addf1"} : <f32>
    %48 = mulf %47, %49 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "mulf3"} : <f32>
    %49 = buffer %34#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer17"} : <f32>
    %50 = br %48 {handshake.bb = 2 : ui32, handshake.name = "br4"} : <f32>
    %51 = br %31 {handshake.bb = 2 : ui32, handshake.name = "br5"} : <i11>
    %52 = br %32 {handshake.bb = 2 : ui32, handshake.name = "br6"} : <i10>
    %53 = br %result_10 {handshake.bb = 2 : ui32, handshake.name = "br7"} : <>
    %54 = mux %55 [%trueResult, %50] {handshake.bb = 3 : ui32, handshake.name = "mux1"} : <i1>, [<f32>, <f32>] to <f32>
    %55 = buffer %61#2, bufferType = FIFO_BREAK_NONE, numSlots = 4 {handshake.bb = 3 : ui32, handshake.name = "buffer18"} : <i1>
    %56 = mux %61#1 [%trueResult_2, %51] {handshake.bb = 3 : ui32, handshake.name = "mux2"} : <i1>, [<i11>, <i11>] to <i11>
    %58 = extsi %56 {handshake.bb = 3 : ui32, handshake.name = "extsi4"} : <i11> to <i12>
    %59 = mux %61#0 [%trueResult_4, %52] {handshake.bb = 3 : ui32, handshake.name = "mux3"} : <i1>, [<i10>, <i10>] to <i10>
    %result_12, %index_13 = control_merge [%trueResult_6, %53]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>, <>] to <>, <i1>
    %61:3 = fork [3] %index_13 {handshake.bb = 3 : ui32, handshake.name = "fork7"} : <i1>
    %62:2 = lazy_fork [2] %result_12 {handshake.bb = 3 : ui32, handshake.name = "lazy_fork1"} : <>
    %63 = source {handshake.bb = 3 : ui32, handshake.name = "source2"} : <>
    %64 = constant %63 {handshake.bb = 3 : ui32, handshake.name = "constant1", value = 1000 : i11} : <>, <i11>
    %65 = extsi %64 {handshake.bb = 3 : ui32, handshake.name = "extsi5"} : <i11> to <i12>
    %66 = source {handshake.bb = 3 : ui32, handshake.name = "source3"} : <>
    %67 = constant %66 {handshake.bb = 3 : ui32, handshake.name = "constant3", value = 1 : i2} : <>, <i2>
    %68 = extsi %67 {handshake.bb = 3 : ui32, handshake.name = "extsi6"} : <i2> to <i12>
    %addressResult_14, %dataResult_15 = store[%59] %54 {handshake.bb = 3 : ui32, handshake.deps = #handshake<deps[["load1", 0], ["store0", 0]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 1>, handshake.name = "store0"} : <i10>, <f32>, <i10>, <f32>
    %69 = addi %58, %68 {handshake.bb = 3 : ui32, handshake.name = "addi0"} : <i12>
    %70:2 = fork [2] %69 {handshake.bb = 3 : ui32, handshake.name = "fork8"} : <i12>
    %71 = trunci %70#0 {handshake.bb = 3 : ui32, handshake.name = "trunci3"} : <i12> to <i11>
    %73 = cmpi ult, %70#1, %65 {handshake.bb = 3 : ui32, handshake.name = "cmpi0"} : <i12>
    %75:2 = fork [2] %73 {handshake.bb = 3 : ui32, handshake.name = "fork9"} : <i1>
    %trueResult_16, %falseResult_17 = cond_br %75#0, %71 {handshake.bb = 3 : ui32, handshake.name = "cond_br7"} : <i1>, <i11>
    sink %falseResult_17 {handshake.name = "sink3"} : <i11>
    %trueResult_18, %falseResult_19 = cond_br %75#1, %62#1 {handshake.bb = 3 : ui32, handshake.name = "cond_br8"} : <i1>, <>
    %result_20, %index_21 = control_merge [%falseResult_19]  {handshake.bb = 4 : ui32, handshake.name = "control_merge3"} : [<>] to <>, <i1>
    sink %index_21 {handshake.name = "sink4"} : <i1>
    %78:2 = fork [2] %result_20 {handshake.bb = 4 : ui32, handshake.name = "fork10"} : <>
    end {handshake.bb = 4 : ui32, handshake.name = "end0"} %1#1, %memEnd, %0#1 : <>, <>, <>
  }
}

