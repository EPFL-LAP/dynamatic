module {
  handshake.func @get_tanh(%arg0: memref<1000xf32>, %arg1: memref<1000xi32>, %arg2: !handshake.control<>, %arg3: !handshake.control<>, %arg4: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["A", "addr", "A_start", "addr_start", "start"], cfg.edges = "[0,1][2,3][1,3,2,cmpf0][3,1,4,cmpi0]", resNames = ["A_end", "addr_end", "end"]} {
    %0:4 = fork [4] %arg4 {handshake.bb = 0 : ui32, handshake.name = "fork0"} : <>
    %outputs, %memEnd = mem_controller[%arg1 : memref<1000xi32>] %arg3 (%addressResult) %85#1 {connectedBlocks = [1 : i32], handshake.name = "mem_controller1"} :    (!handshake.channel<i10>) -> !handshake.channel<i32>
    %outputs_0:2, %memEnd_1 = mem_controller[%arg0 : memref<1000xf32>] %arg2 (%addressResult_2, %68, %addressResult_18, %dataResult_19) %85#0 {connectedBlocks = [1 : i32, 3 : i32], handshake.name = "mem_controller2"} :    (!handshake.channel<i10>, !handshake.channel<i32>, !handshake.channel<i10>, !handshake.channel<f32>) -> (!handshake.channel<f32>, !handshake.control<>)
    %1 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant1", value = false} : <>, <i1>
    %2 = br %1 {handshake.bb = 0 : ui32, handshake.name = "br1"} : <i1>
    %3 = extsi %2 {handshake.bb = 0 : ui32, handshake.name = "extsi4"} : <i1> to <i11>
    %4 = br %0#3 {handshake.bb = 0 : ui32, handshake.name = "br3"} : <>
    %5 = mux %6 [%0#2, %trueResult_14] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux4"} : <i1>, [<>, <>] to <>
    %6 = init %82#2 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "init0"} : <i1>
    %8 = mux %index [%3, %trueResult_20] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i11>, <i11>] to <i11>
    %9:2 = fork [2] %8 {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <i11>
    %10 = trunci %9#0 {handshake.bb = 1 : ui32, handshake.name = "trunci0"} : <i11> to <i10>
    %result, %index = control_merge [%4, %trueResult_22]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %12:2 = fork [2] %result {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <>
    %13 = constant %12#1 {handshake.bb = 1 : ui32, handshake.name = "constant6", value = 1.000000e+00 : f32} : <>, <f32>
    %14:2 = fork [2] %13 {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <f32>
    %addressResult, %dataResult = load[%10] %outputs {handshake.bb = 1 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i10>, <i32>, <i10>, <i32>
    %15:2 = fork [2] %dataResult {handshake.bb = 1 : ui32, handshake.name = "fork4"} : <i32>
    %16 = trunci %15#0 {handshake.bb = 1 : ui32, handshake.name = "trunci1"} : <i32> to <i10>
    %18 = gate %15#1, %5 {handshake.bb = 1 : ui32, handshake.name = "gate0"} : <i32>, !handshake.control<> to <i32>
    %20 = trunci %18 {handshake.bb = 1 : ui32, handshake.name = "trunci2"} : <i32> to <i10>
    %addressResult_2, %dataResult_3 = load[%20] %outputs_0#0 {handshake.bb = 1 : ui32, handshake.deps = #handshake<deps[["store1", 0, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load3"} : <i10>, <f32>, <i10>, <f32>
    %21:2 = fork [2] %dataResult_3 {handshake.bb = 1 : ui32, handshake.name = "fork5"} : <f32>
    %22 = cmpf oge, %21#0, %14#1 {handshake.bb = 1 : ui32, handshake.name = "cmpf0"} : <f32>
    %25:5 = fork [5] %22 {handshake.bb = 1 : ui32, handshake.name = "fork6"} : <i1>
    %trueResult, %falseResult = cond_br %25#4, %14#0 {handshake.bb = 1 : ui32, handshake.name = "cond_br2"} : <i1>, <f32>
    sink %falseResult {handshake.name = "sink0"} : <f32>
    %trueResult_4, %falseResult_5 = cond_br %25#1, %9#1 {handshake.bb = 1 : ui32, handshake.name = "cond_br3"} : <i1>, <i11>
    %trueResult_6, %falseResult_7 = cond_br %25#0, %16 {handshake.bb = 1 : ui32, handshake.name = "cond_br4"} : <i1>, <i10>
    %trueResult_8, %falseResult_9 = cond_br %25#3, %12#0 {handshake.bb = 1 : ui32, handshake.name = "cond_br5"} : <i1>, <>
    %trueResult_10, %falseResult_11 = cond_br %25#2, %21#1 {handshake.bb = 1 : ui32, handshake.name = "cond_br6"} : <i1>, <f32>
    sink %trueResult_10 {handshake.name = "sink1"} : <f32>
    %34 = merge %falseResult_5 {handshake.bb = 2 : ui32, handshake.name = "merge0"} : <i11>
    %35 = merge %falseResult_7 {handshake.bb = 2 : ui32, handshake.name = "merge1"} : <i10>
    %36 = merge %falseResult_11 {handshake.bb = 2 : ui32, handshake.name = "merge2"} : <f32>
    %37:5 = fork [5] %36 {handshake.bb = 2 : ui32, handshake.name = "fork7"} : <f32>
    %result_12, %index_13 = control_merge [%falseResult_9]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>] to <>, <i1>
    sink %index_13 {handshake.name = "sink2"} : <i1>
    %38 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %39 = constant %38 {handshake.bb = 2 : ui32, handshake.name = "constant7", value = 3.70476198 : f32} : <>, <f32>
    %40 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %41 = constant %40 {handshake.bb = 2 : ui32, handshake.name = "constant8", value = 19.5238094 : f32} : <>, <f32>
    %42 = mulf %37#3, %37#4 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "mulf0"} : <f32>
    %45 = addf %42, %41 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "addf0"} : <f32>
    %46 = mulf %45, %37#2 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "mulf1"} : <f32>
    %48 = mulf %46, %37#1 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "mulf2"} : <f32>
    %50 = addf %48, %39 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "addf1"} : <f32>
    %51 = mulf %50, %37#0 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "mulf3"} : <f32>
    %53 = br %51 {handshake.bb = 2 : ui32, handshake.name = "br4"} : <f32>
    %54 = br %34 {handshake.bb = 2 : ui32, handshake.name = "br5"} : <i11>
    %55 = br %35 {handshake.bb = 2 : ui32, handshake.name = "br6"} : <i10>
    %56 = br %result_12 {handshake.bb = 2 : ui32, handshake.name = "br7"} : <>
    %trueResult_14, %falseResult_15 = cond_br %82#1, %75 {handshake.bb = 3 : ui32, handshake.name = "cond_br1"} : <i1>, <>
    sink %falseResult_15 {handshake.name = "sink3"} : <>
    %58 = mux %65#2 [%trueResult, %53] {handshake.bb = 3 : ui32, handshake.name = "mux1"} : <i1>, [<f32>, <f32>] to <f32>
    %60 = mux %65#1 [%trueResult_4, %54] {handshake.bb = 3 : ui32, handshake.name = "mux2"} : <i1>, [<i11>, <i11>] to <i11>
    %62 = extsi %60 {handshake.bb = 3 : ui32, handshake.name = "extsi5"} : <i11> to <i12>
    %63 = mux %65#0 [%trueResult_6, %55] {handshake.bb = 3 : ui32, handshake.name = "mux3"} : <i1>, [<i10>, <i10>] to <i10>
    %result_16, %index_17 = control_merge [%trueResult_8, %56]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>, <>] to <>, <i1>
    %65:3 = fork [3] %index_17 {handshake.bb = 3 : ui32, handshake.name = "fork8"} : <i1>
    %66:2 = fork [2] %result_16 {handshake.bb = 3 : ui32, handshake.name = "fork9"} : <>
    %67 = constant %66#0 {handshake.bb = 3 : ui32, handshake.name = "constant3", value = 1 : i2} : <>, <i2>
    %68 = extsi %67 {handshake.bb = 3 : ui32, handshake.name = "extsi1"} : <i2> to <i32>
    %69 = source {handshake.bb = 3 : ui32, handshake.name = "source2"} : <>
    %70 = constant %69 {handshake.bb = 3 : ui32, handshake.name = "constant4", value = 1000 : i11} : <>, <i11>
    %71 = extsi %70 {handshake.bb = 3 : ui32, handshake.name = "extsi6"} : <i11> to <i12>
    %72 = source {handshake.bb = 3 : ui32, handshake.name = "source3"} : <>
    %73 = constant %72 {handshake.bb = 3 : ui32, handshake.name = "constant5", value = 1 : i2} : <>, <i2>
    %74 = extsi %73 {handshake.bb = 3 : ui32, handshake.name = "extsi7"} : <i2> to <i12>
    %75 = buffer %doneResult, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer0"} : <>
    %addressResult_18, %dataResult_19, %doneResult = store[%63] %58 %outputs_0#1 {handshake.bb = 3 : ui32, handshake.deps = #handshake<deps[["load3", 0, false], ["store1", 0, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store1"} : <i10>, <f32>, <>, <i10>, <f32>, <>
    %76 = addi %62, %74 {handshake.bb = 3 : ui32, handshake.name = "addi0"} : <i12>
    %77:2 = fork [2] %76 {handshake.bb = 3 : ui32, handshake.name = "fork10"} : <i12>
    %78 = trunci %77#0 {handshake.bb = 3 : ui32, handshake.name = "trunci3"} : <i12> to <i11>
    %80 = cmpi ult, %77#1, %71 {handshake.bb = 3 : ui32, handshake.name = "cmpi0"} : <i12>
    %82:4 = fork [4] %80 {handshake.bb = 3 : ui32, handshake.name = "fork11"} : <i1>
    %trueResult_20, %falseResult_21 = cond_br %82#0, %78 {handshake.bb = 3 : ui32, handshake.name = "cond_br7"} : <i1>, <i11>
    sink %falseResult_21 {handshake.name = "sink4"} : <i11>
    %trueResult_22, %falseResult_23 = cond_br %82#3, %66#1 {handshake.bb = 3 : ui32, handshake.name = "cond_br8"} : <i1>, <>
    %result_24, %index_25 = control_merge [%falseResult_23]  {handshake.bb = 4 : ui32, handshake.name = "control_merge3"} : [<>] to <>, <i1>
    sink %index_25 {handshake.name = "sink5"} : <i1>
    %85:2 = fork [2] %result_24 {handshake.bb = 4 : ui32, handshake.name = "fork12"} : <>
    end {handshake.bb = 4 : ui32, handshake.name = "end0"} %memEnd_1, %memEnd, %0#1 : <>, <>, <>
  }
}

