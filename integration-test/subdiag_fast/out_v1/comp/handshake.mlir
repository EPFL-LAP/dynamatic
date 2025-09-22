module {
  handshake.func @subdiag_fast(%arg0: memref<1000xf32>, %arg1: memref<1000xf32>, %arg2: memref<1000xf32>, %arg3: !handshake.control<>, %arg4: !handshake.control<>, %arg5: !handshake.control<>, %arg6: !handshake.control<>, ...) -> (!handshake.channel<i32>, !handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["d1", "d2", "e", "d1_start", "d2_start", "e_start", "start"], resNames = ["out0", "d1_end", "d2_end", "e_end", "end"]} {
    %outputs, %memEnd = mem_controller[%arg2 : memref<1000xf32>] %arg5 (%addressResult_6) %result_12 {connectedBlocks = [1 : i32], handshake.name = "mem_controller0"} :    (!handshake.channel<i32>) -> !handshake.channel<f32>
    %outputs_0, %memEnd_1 = mem_controller[%arg1 : memref<1000xf32>] %arg4 (%addressResult_4) %result_12 {connectedBlocks = [1 : i32], handshake.name = "mem_controller1"} :    (!handshake.channel<i32>) -> !handshake.channel<f32>
    %outputs_2, %memEnd_3 = mem_controller[%arg0 : memref<1000xf32>] %arg3 (%addressResult) %result_12 {connectedBlocks = [1 : i32], handshake.name = "mem_controller2"} :    (!handshake.channel<i32>) -> !handshake.channel<f32>
    %0 = constant %arg6 {handshake.bb = 0 : ui32, handshake.name = "constant3", value = 0 : i32} : <>, <i32>
    %1 = br %0 {handshake.bb = 0 : ui32, handshake.name = "br1"} : <i32>
    %2 = br %arg6 {handshake.bb = 0 : ui32, handshake.name = "br2"} : <>
    %3 = mux %index [%1, %trueResult] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i32>, <i32>] to <i32>
    %result, %index = control_merge [%2, %trueResult_8]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %4 = source {handshake.bb = 1 : ui32, handshake.name = "source0"} : <>
    %5 = constant %4 {handshake.bb = 1 : ui32, handshake.name = "constant4", value = 1 : i32} : <>, <i32>
    %6 = source {handshake.bb = 1 : ui32, handshake.name = "source1"} : <>
    %7 = constant %6 {handshake.bb = 1 : ui32, handshake.name = "constant5", value = 1.000000e-03 : f32} : <>, <f32>
    %8 = source {handshake.bb = 1 : ui32, handshake.name = "source2"} : <>
    %9 = constant %8 {handshake.bb = 1 : ui32, handshake.name = "constant6", value = 998 : i32} : <>, <i32>
    %10 = extui %3 {handshake.bb = 1 : ui32, handshake.name = "extui0"} : <i32> to <i64>
    %11 = trunci %10 {handshake.bb = 1 : ui32, handshake.name = "index_cast0"} : <i64> to <i32>
    %addressResult, %dataResult = load[%11] %outputs_2 {handshake.bb = 1 : ui32, handshake.name = "load0"} : <i32>, <f32>, <i32>, <f32>
    %12 = addi %3, %5 {handshake.bb = 1 : ui32, handshake.name = "addi0"} : <i32>
    %13 = extui %12 {handshake.bb = 1 : ui32, handshake.name = "extui1"} : <i32> to <i64>
    %14 = trunci %13 {handshake.bb = 1 : ui32, handshake.name = "index_cast1"} : <i64> to <i32>
    %addressResult_4, %dataResult_5 = load[%14] %outputs_0 {handshake.bb = 1 : ui32, handshake.name = "load1"} : <i32>, <f32>, <i32>, <f32>
    %15 = addf %dataResult, %dataResult_5 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "addf0"} : <f32>
    %16 = addi %3, %5 {handshake.bb = 1 : ui32, handshake.name = "addi1"} : <i32>
    %17 = extui %3 {handshake.bb = 1 : ui32, handshake.name = "extui2"} : <i32> to <i64>
    %18 = trunci %17 {handshake.bb = 1 : ui32, handshake.name = "index_cast2"} : <i64> to <i32>
    %addressResult_6, %dataResult_7 = load[%18] %outputs {handshake.bb = 1 : ui32, handshake.name = "load2"} : <i32>, <f32>, <i32>, <f32>
    %19 = mulf %15, %7 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "mulf0"} : <f32>
    %20 = cmpf ugt, %dataResult_7, %19 {handshake.bb = 1 : ui32, handshake.name = "cmpf0"} : <f32>
    %21 = cmpi ult, %3, %9 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i32>
    %22 = andi %21, %20 {handshake.bb = 1 : ui32, handshake.name = "andi0"} : <i1>
    %trueResult, %falseResult = cond_br %22, %16 {handshake.bb = 1 : ui32, handshake.name = "cond_br1"} : <i1>, <i32>
    %trueResult_8, %falseResult_9 = cond_br %22, %result {handshake.bb = 1 : ui32, handshake.name = "cond_br2"} : <i1>, <>
    %trueResult_10, %falseResult_11 = cond_br %22, %3 {handshake.bb = 1 : ui32, handshake.name = "cond_br3"} : <i1>, <i32>
    %23 = merge %falseResult_11 {handshake.bb = 2 : ui32, handshake.name = "merge0"} : <i32>
    %result_12, %index_13 = control_merge [%falseResult_9]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>] to <>, <i1>
    end {handshake.bb = 2 : ui32, handshake.name = "end0"} %23, %memEnd_3, %memEnd_1, %memEnd, %arg6 : <i32>, <>, <>, <>, <>
  }
}

