module {
  handshake.func @subdiag_fast(%arg0: memref<1000xf32>, %arg1: memref<1000xf32>, %arg2: memref<1000xf32>, %arg3: !handshake.control<>, %arg4: !handshake.control<>, %arg5: !handshake.control<>, %arg6: !handshake.control<>, ...) -> (!handshake.channel<i32>, !handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["d1", "d2", "e", "d1_start", "d2_start", "e_start", "start"], resNames = ["out0", "d1_end", "d2_end", "e_end", "end"]} {
    %0:3 = fork [3] %arg6 {handshake.bb = 0 : ui32, handshake.name = "fork0"} : <>
    %outputs, %memEnd = mem_controller[%arg2 : memref<1000xf32>] %arg5 (%addressResult_6) %26#2 {connectedBlocks = [1 : i32], handshake.name = "mem_controller3"} :    (!handshake.channel<i10>) -> !handshake.channel<f32>
    %outputs_0, %memEnd_1 = mem_controller[%arg1 : memref<1000xf32>] %arg4 (%addressResult_4) %26#1 {connectedBlocks = [1 : i32], handshake.name = "mem_controller4"} :    (!handshake.channel<i10>) -> !handshake.channel<f32>
    %outputs_2, %memEnd_3 = mem_controller[%arg0 : memref<1000xf32>] %arg3 (%addressResult) %26#0 {connectedBlocks = [1 : i32], handshake.name = "mem_controller5"} :    (!handshake.channel<i10>) -> !handshake.channel<f32>
    %1 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant0", value = false} : <>, <i1>
    %2 = extsi %1 {handshake.bb = 0 : ui32, handshake.name = "extsi3"} : <i1> to <i32>
    %3 = mux %index [%2, %trueResult] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i32>, <i32>] to <i32>
    %4:6 = fork [6] %3 {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <i32>
    %5 = trunci %4#0 {handshake.bb = 1 : ui32, handshake.name = "trunci0"} : <i32> to <i10>
    %6 = trunci %4#1 {handshake.bb = 1 : ui32, handshake.name = "trunci1"} : <i32> to <i10>
    %7 = trunci %4#2 {handshake.bb = 1 : ui32, handshake.name = "trunci2"} : <i32> to <i10>
    %result, %index = control_merge [%0#2, %trueResult_8]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %8 = source {handshake.bb = 1 : ui32, handshake.name = "source0"} : <>
    %9 = constant %8 {handshake.bb = 1 : ui32, handshake.name = "constant1", value = 1 : i2} : <>, <i2>
    %10:2 = fork [2] %9 {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <i2>
    %11 = extsi %10#0 {handshake.bb = 1 : ui32, handshake.name = "extsi4"} : <i2> to <i10>
    %12 = extsi %10#1 {handshake.bb = 1 : ui32, handshake.name = "extsi1"} : <i2> to <i32>
    %13 = source {handshake.bb = 1 : ui32, handshake.name = "source1"} : <>
    %14 = constant %13 {handshake.bb = 1 : ui32, handshake.name = "constant5", value = 1.000000e-03 : f32} : <>, <f32>
    %15 = source {handshake.bb = 1 : ui32, handshake.name = "source2"} : <>
    %16 = constant %15 {handshake.bb = 1 : ui32, handshake.name = "constant2", value = 998 : i11} : <>, <i11>
    %17 = extsi %16 {handshake.bb = 1 : ui32, handshake.name = "extsi2"} : <i11> to <i32>
    %addressResult, %dataResult = load[%7] %outputs_2 {handshake.bb = 1 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i10>, <f32>, <i10>, <f32>
    %18 = addi %5, %11 {handshake.bb = 1 : ui32, handshake.name = "addi0"} : <i10>
    %addressResult_4, %dataResult_5 = load[%18] %outputs_0 {handshake.bb = 1 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i10>, <f32>, <i10>, <f32>
    %19 = addf %dataResult, %dataResult_5 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "addf0"} : <f32>
    %20 = addi %4#5, %12 {handshake.bb = 1 : ui32, handshake.name = "addi1"} : <i32>
    %addressResult_6, %dataResult_7 = load[%6] %outputs {handshake.bb = 1 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load2"} : <i10>, <f32>, <i10>, <f32>
    %21 = mulf %19, %14 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "mulf0"} : <f32>
    %22 = cmpf ugt, %dataResult_7, %21 {handshake.bb = 1 : ui32, handshake.name = "cmpf0"} : <f32>
    %23 = cmpi ult, %4#4, %17 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i32>
    %24 = andi %23, %22 {handshake.bb = 1 : ui32, handshake.name = "andi0"} : <i1>
    %25:3 = fork [3] %24 {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <i1>
    %trueResult, %falseResult = cond_br %25#2, %20 {handshake.bb = 1 : ui32, handshake.name = "cond_br1"} : <i1>, <i32>
    sink %falseResult {handshake.name = "sink0"} : <i32>
    %trueResult_8, %falseResult_9 = cond_br %25#1, %result {handshake.bb = 1 : ui32, handshake.name = "cond_br2"} : <i1>, <>
    %trueResult_10, %falseResult_11 = cond_br %25#0, %4#3 {handshake.bb = 1 : ui32, handshake.name = "cond_br3"} : <i1>, <i32>
    sink %trueResult_10 {handshake.name = "sink1"} : <i32>
    %26:3 = fork [3] %falseResult_9 {handshake.bb = 2 : ui32, handshake.name = "fork4"} : <>
    end {handshake.bb = 2 : ui32, handshake.name = "end0"} %falseResult_11, %memEnd_3, %memEnd_1, %memEnd, %0#1 : <i32>, <>, <>, <>, <>
  }
}

