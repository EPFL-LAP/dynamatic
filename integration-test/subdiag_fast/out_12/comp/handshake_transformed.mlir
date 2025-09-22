module {
  handshake.func @subdiag_fast(%arg0: memref<1000xf32>, %arg1: memref<1000xf32>, %arg2: memref<1000xf32>, %arg3: !handshake.control<>, %arg4: !handshake.control<>, %arg5: !handshake.control<>, %arg6: !handshake.control<>, ...) -> (!handshake.channel<i32>, !handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["d1", "d2", "e", "d1_start", "d2_start", "e_start", "start"], resNames = ["out0", "d1_end", "d2_end", "e_end", "end"]} {
    %0:3 = fork [3] %arg6 {handshake.bb = 0 : ui32, handshake.name = "fork0"} : <>
    %outputs, %memEnd = mem_controller[%arg2 : memref<1000xf32>] %arg5 (%addressResult_6) %29#2 {connectedBlocks = [1 : i32], handshake.name = "mem_controller3"} :    (!handshake.channel<i10>) -> !handshake.channel<f32>
    %outputs_0, %memEnd_1 = mem_controller[%arg1 : memref<1000xf32>] %arg4 (%addressResult_4) %29#1 {connectedBlocks = [1 : i32], handshake.name = "mem_controller4"} :    (!handshake.channel<i10>) -> !handshake.channel<f32>
    %outputs_2, %memEnd_3 = mem_controller[%arg0 : memref<1000xf32>] %arg3 (%addressResult) %29#0 {connectedBlocks = [1 : i32], handshake.name = "mem_controller5"} :    (!handshake.channel<i10>) -> !handshake.channel<f32>
    %1 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant0", value = false} : <>, <i1>
    %2 = extsi %1 {handshake.bb = 0 : ui32, handshake.name = "extsi3"} : <i1> to <i11>
    %3 = mux %index [%2, %26] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i11>, <i11>] to <i11>
    %4:4 = fork [4] %3 {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <i11>
    %5 = trunci %4#0 {handshake.bb = 1 : ui32, handshake.name = "trunci0"} : <i11> to <i10>
    %6 = trunci %4#1 {handshake.bb = 1 : ui32, handshake.name = "trunci1"} : <i11> to <i10>
    %7 = trunci %4#2 {handshake.bb = 1 : ui32, handshake.name = "trunci2"} : <i11> to <i10>
    %result, %index = control_merge [%0#2, %trueResult_12]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %8 = source {handshake.bb = 1 : ui32, handshake.name = "source0"} : <>
    %9 = constant %8 {handshake.bb = 1 : ui32, handshake.name = "constant4", value = 1.000000e-03 : f32} : <>, <f32>
    %addressResult, %dataResult = load[%7] %outputs_2 {handshake.bb = 1 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i10>, <f32>, <i10>, <f32>
    %addressResult_4, %dataResult_5 = load[%6] %outputs_0 {handshake.bb = 1 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i10>, <f32>, <i10>, <f32>
    %10 = addf %dataResult, %dataResult_5 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "addf0"} : <f32>
    %addressResult_6, %dataResult_7 = load[%5] %outputs {handshake.bb = 1 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load2"} : <i10>, <f32>, <i10>, <f32>
    %11 = mulf %10, %9 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "mulf0"} : <f32>
    %12 = cmpf ugt, %dataResult_7, %11 {handshake.bb = 1 : ui32, handshake.name = "cmpf0"} : <f32>
    %13:2 = fork [2] %12 {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <i1>
    %trueResult, %falseResult = cond_br %13#0, %4#3 {handshake.bb = 1 : ui32, handshake.name = "cond_br2"} : <i1>, <i11>
    %14 = extsi %falseResult {handshake.bb = 1 : ui32, handshake.name = "extsi4"} : <i11> to <i12>
    %trueResult_8, %falseResult_9 = cond_br %13#1, %result {handshake.bb = 1 : ui32, handshake.name = "cond_br3"} : <i1>, <>
    %15 = extsi %trueResult {handshake.bb = 2 : ui32, handshake.name = "extsi5"} : <i11> to <i12>
    %16 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %17 = constant %16 {handshake.bb = 2 : ui32, handshake.name = "constant1", value = 1 : i2} : <>, <i2>
    %18 = extsi %17 {handshake.bb = 2 : ui32, handshake.name = "extsi6"} : <i2> to <i12>
    %19 = source {handshake.bb = 2 : ui32, handshake.name = "source2"} : <>
    %20 = constant %19 {handshake.bb = 2 : ui32, handshake.name = "constant2", value = 999 : i11} : <>, <i11>
    %21 = extsi %20 {handshake.bb = 2 : ui32, handshake.name = "extsi7"} : <i11> to <i12>
    %22 = addi %15, %18 {handshake.bb = 2 : ui32, handshake.name = "addi0"} : <i12>
    %23:2 = fork [2] %22 {handshake.bb = 2 : ui32, handshake.name = "fork3"} : <i12>
    %24 = cmpi ult, %23#1, %21 {handshake.bb = 2 : ui32, handshake.name = "cmpi0"} : <i12>
    %25:2 = fork [2] %24 {handshake.bb = 2 : ui32, handshake.name = "fork4"} : <i1>
    %trueResult_10, %falseResult_11 = cond_br %25#0, %23#0 {handshake.bb = 2 : ui32, handshake.name = "cond_br4"} : <i1>, <i12>
    %26 = trunci %trueResult_10 {handshake.bb = 2 : ui32, handshake.name = "trunci3"} : <i12> to <i11>
    %trueResult_12, %falseResult_13 = cond_br %25#1, %trueResult_8 {handshake.bb = 2 : ui32, handshake.name = "cond_br5"} : <i1>, <>
    %27 = mux %index_15 [%14, %falseResult_11] {handshake.bb = 3 : ui32, handshake.name = "mux1"} : <i1>, [<i12>, <i12>] to <i12>
    %28 = extsi %27 {handshake.bb = 3 : ui32, handshake.name = "extsi8"} : <i12> to <i32>
    %result_14, %index_15 = control_merge [%falseResult_9, %falseResult_13]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>, <>] to <>, <i1>
    %29:3 = fork [3] %result_14 {handshake.bb = 3 : ui32, handshake.name = "fork5"} : <>
    end {handshake.bb = 3 : ui32, handshake.name = "end0"} %28, %memEnd_3, %memEnd_1, %memEnd, %0#1 : <i32>, <>, <>, <>, <>
  }
}

