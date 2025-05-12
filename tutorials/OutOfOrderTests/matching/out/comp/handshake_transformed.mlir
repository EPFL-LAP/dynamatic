module {
  handshake.func @matching(%arg0: !handshake.channel<i32>, %arg1: !handshake.control<>, ...) -> (!handshake.channel<f32>, !handshake.control<>) attributes {argNames = ["num_edges", "start"], cfg.edges = "[0,1][2,1][1,2,3,cmpi0]", resNames = ["out0", "end"]} {
    %0 = constant %arg1 {handshake.bb = 0 : ui32, handshake.name = "constant0", value = 0.000000e+00 : f32} : <>, <f32>
    %trueResult, %falseResult = cond_br %8, %5 {handshake.bb = 1 : ui32, handshake.name = "cond_br10"} : <i1>, <i32>
    %1 = constant %arg1 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "constant1", value = false} : <>, <i1>
    %2 = extsi %1 {handshake.bb = 1 : ui32, handshake.name = "extsi0"} : <i1> to <i32>
    %3 = merge %1, %8 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "merge0"} : <i1>
    %4 = mux %3 [%arg1, %trueResult_4] {ftd.regen, handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<>, <>] to <>
    %5 = mux %3 [%arg0, %trueResult] {ftd.regen, handshake.bb = 1 : ui32, handshake.name = "mux2"} : <i1>, [<i32>, <i32>] to <i32>
    %6 = mux %3 [%0, %12] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux3"} : <i1>, [<f32>, <f32>] to <f32>
    %7 = mux %3 [%2, %13] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux4"} : <i1>, [<i32>, <i32>] to <i32>
    %8 = cmpi slt, %7, %5 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i32>
    %trueResult_0, %falseResult_1 = cond_br %8, %7 {handshake.bb = 2 : ui32, handshake.name = "cond_br11"} : <i1>, <i32>
    %trueResult_2, %falseResult_3 = cond_br %8, %6 {handshake.bb = 2 : ui32, handshake.name = "cond_br12"} : <i1>, <f32>
    %trueResult_4, %falseResult_5 = cond_br %8, %4 {handshake.bb = 2 : ui32, handshake.name = "cond_br13"} : <i1>, <>
    %9 = constant %trueResult_4 {handshake.bb = 2 : ui32, handshake.name = "constant3", value = 1 : i2} : <>, <i2>
    %10 = extsi %9 {handshake.bb = 2 : ui32, handshake.name = "extsi1"} : <i2> to <i32>
    %11 = constant %trueResult_4 {handshake.bb = 2 : ui32, handshake.name = "constant5", value = 1.000000e+00 : f32} : <>, <f32>
    %12 = addf %trueResult_2, %11 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "addf0"} : <f32>
    %13 = addi %trueResult_0, %10 {handshake.bb = 2 : ui32, handshake.name = "addi0"} : <i32>
    end {handshake.bb = 3 : ui32, handshake.name = "end0"} %falseResult_3, %arg1 : <f32>, <>
  }
}
