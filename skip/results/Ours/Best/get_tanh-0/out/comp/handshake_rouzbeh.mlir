module {
  handshake.func @get_tanh(%arg0: memref<1000xf32>, %arg1: memref<1000xi32>, %arg2: !handshake.control<>, %arg3: !handshake.control<>, %arg4: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["A", "addr", "A_start", "addr_start", "start"], cfg.edges = "[0,1][2,3][1,3,2,cmpf0][3,1,4,cmpi0]", resNames = ["A_end", "addr_end", "end"]} {
    %outputs, %memEnd = mem_controller[%arg1 : memref<1000xi32>] %arg3 (%addressResult) %result_24 {connectedBlocks = [1 : i32], handshake.name = "mem_controller1"} :    (!handshake.channel<i32>) -> !handshake.channel<i32>
    %outputs_0:2, %memEnd_1 = mem_controller[%arg0 : memref<1000xf32>] %arg2 (%addressResult_2, %29, %addressResult_18, %dataResult_19) %result_24 {connectedBlocks = [1 : i32, 3 : i32], handshake.name = "mem_controller2"} :    (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<f32>) -> (!handshake.channel<f32>, !handshake.control<>)
    %0 = constant %arg4 {handshake.bb = 0 : ui32, handshake.name = "constant2", value = 0 : i32} : <>, <i32>
    %1 = br %0 {handshake.bb = 0 : ui32, handshake.name = "br1"} : <i32>
    %2 = br %arg4 {handshake.bb = 0 : ui32, handshake.name = "br3"} : <>
    %3 = mux %4 [%arg4, %trueResult_14] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux4"} : <i1>, [<>, <>] to <>
    %4 = init %36 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "init0"} : <i1>
    %5 = mux %index [%1, %trueResult_20] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i32>, <i32>] to <i32>
    %result, %index = control_merge [%2, %trueResult_22]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %6 = constant %result {handshake.bb = 1 : ui32, handshake.name = "constant6", value = 1.000000e+00 : f32} : <>, <f32>
    %addressResult, %dataResult = load[%5] %outputs {handshake.bb = 1 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i32>, <i32>, <i32>, <i32>
    %7 = gate %dataResult, %3 {handshake.bb = 1 : ui32, handshake.name = "gate0"} : <i32>, !handshake.control<> to <i32>
    %addressResult_2, %dataResult_3 = load[%7] %outputs_0#0 {handshake.bb = 1 : ui32, handshake.deps = #handshake<deps[["store1", 0, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load3"} : <i32>, <f32>, <i32>, <f32>
    %8 = cmpf oge, %dataResult_3, %6 {handshake.bb = 1 : ui32, handshake.name = "cmpf0"} : <f32>
    %trueResult, %falseResult = cond_br %8, %6 {handshake.bb = 1 : ui32, handshake.name = "cond_br2"} : <i1>, <f32>
    %trueResult_4, %falseResult_5 = cond_br %8, %5 {handshake.bb = 1 : ui32, handshake.name = "cond_br3"} : <i1>, <i32>
    %trueResult_6, %falseResult_7 = cond_br %8, %dataResult {handshake.bb = 1 : ui32, handshake.name = "cond_br4"} : <i1>, <i32>
    %trueResult_8, %falseResult_9 = cond_br %8, %result {handshake.bb = 1 : ui32, handshake.name = "cond_br5"} : <i1>, <>
    %trueResult_10, %falseResult_11 = cond_br %8, %dataResult_3 {handshake.bb = 1 : ui32, handshake.name = "cond_br6"} : <i1>, <f32>
    %9 = merge %falseResult_5 {handshake.bb = 2 : ui32, handshake.name = "merge0"} : <i32>
    %10 = merge %falseResult_7 {handshake.bb = 2 : ui32, handshake.name = "merge1"} : <i32>
    %11 = merge %falseResult_11 {handshake.bb = 2 : ui32, handshake.name = "merge2"} : <f32>
    %result_12, %index_13 = control_merge [%falseResult_9]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>] to <>, <i1>
    %12 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %13 = constant %12 {handshake.bb = 2 : ui32, handshake.name = "constant7", value = 3.70476198 : f32} : <>, <f32>
    %14 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %15 = constant %14 {handshake.bb = 2 : ui32, handshake.name = "constant8", value = 19.5238094 : f32} : <>, <f32>
    %16 = mulf %11, %11 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "mulf0"} : <f32>
    %17 = addf %16, %15 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "addf0"} : <f32>
    %18 = mulf %17, %11 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "mulf1"} : <f32>
    %19 = mulf %18, %11 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "mulf2"} : <f32>
    %20 = addf %19, %13 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "addf1"} : <f32>
    %21 = mulf %20, %11 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "mulf3"} : <f32>
    %22 = br %21 {handshake.bb = 2 : ui32, handshake.name = "br4"} : <f32>
    %23 = br %9 {handshake.bb = 2 : ui32, handshake.name = "br5"} : <i32>
    %24 = br %10 {handshake.bb = 2 : ui32, handshake.name = "br6"} : <i32>
    %25 = br %result_12 {handshake.bb = 2 : ui32, handshake.name = "br7"} : <>
    %trueResult_14, %falseResult_15 = cond_br %36, %34 {handshake.bb = 3 : ui32, handshake.name = "cond_br1"} : <i1>, <>
    %26 = mux %index_17 [%trueResult, %22] {handshake.bb = 3 : ui32, handshake.name = "mux1"} : <i1>, [<f32>, <f32>] to <f32>
    %27 = mux %index_17 [%trueResult_4, %23] {handshake.bb = 3 : ui32, handshake.name = "mux2"} : <i1>, [<i32>, <i32>] to <i32>
    %28 = mux %index_17 [%trueResult_6, %24] {handshake.bb = 3 : ui32, handshake.name = "mux3"} : <i1>, [<i32>, <i32>] to <i32>
    %result_16, %index_17 = control_merge [%trueResult_8, %25]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>, <>] to <>, <i1>
    %29 = constant %result_16 {handshake.bb = 3 : ui32, handshake.name = "constant0", value = 1 : i32} : <>, <i32>
    %30 = source {handshake.bb = 3 : ui32, handshake.name = "source2"} : <>
    %31 = constant %30 {handshake.bb = 3 : ui32, handshake.name = "constant9", value = 1000 : i32} : <>, <i32>
    %32 = source {handshake.bb = 3 : ui32, handshake.name = "source3"} : <>
    %33 = constant %32 {handshake.bb = 3 : ui32, handshake.name = "constant10", value = 1 : i32} : <>, <i32>
    %34 = buffer %doneResult, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer0"} : <>
    %addressResult_18, %dataResult_19, %doneResult = store[%28] %26 %outputs_0#1 {handshake.bb = 3 : ui32, handshake.deps = #handshake<deps[["load3", 0, false], ["store1", 0, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store1"} : <i32>, <f32>, <>, <i32>, <f32>, <>
    %35 = addi %27, %33 {handshake.bb = 3 : ui32, handshake.name = "addi0"} : <i32>
    %36 = cmpi ult, %35, %31 {handshake.bb = 3 : ui32, handshake.name = "cmpi0"} : <i32>
    %trueResult_20, %falseResult_21 = cond_br %36, %35 {handshake.bb = 3 : ui32, handshake.name = "cond_br7"} : <i1>, <i32>
    %trueResult_22, %falseResult_23 = cond_br %36, %result_16 {handshake.bb = 3 : ui32, handshake.name = "cond_br8"} : <i1>, <>
    %result_24, %index_25 = control_merge [%falseResult_23]  {handshake.bb = 4 : ui32, handshake.name = "control_merge3"} : [<>] to <>, <i1>
    end {handshake.bb = 4 : ui32, handshake.name = "end0"} %memEnd_1, %memEnd, %arg4 : <>, <>, <>
  }
}

