module {
  handshake.func @loop_multiply(%arg0: memref<8xi32>, %arg1: !handshake.control<>, %arg2: !handshake.control<>, ...) -> (!handshake.channel<i32>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["a", "a_start", "start"], cfg.edges = "[0,1][1,1,2,cmpi0]", resNames = ["out0", "a_end", "end"]} {
    %outputs, %memEnd = mem_controller[%arg0 : memref<8xi32>] %arg1 (%addressResult) %arg2 {connectedBlocks = [2 : i32], handshake.name = "mem_controller1"} :    (!handshake.channel<i3>) -> !handshake.channel<i32>
    %0 = constant %arg2 {handshake.bb = 0 : ui32, handshake.name = "constant1", value = 2 : i3} : <>, <i3>
    %1 = extsi %0 {handshake.bb = 0 : ui32, handshake.name = "extsi0"} : <i3> to <i32>
    %trueResult, %falseResult = cond_br %16, %15 {handshake.bb = 1 : ui32, handshake.name = "cond_br7"} : <i1>, <i4>
    %trueResult_0, %falseResult_1 = cond_br %16, %13 {handshake.bb = 1 : ui32, handshake.name = "cond_br8"} : <i1>, <i32>
    %trueResult_2, %falseResult_3 = cond_br %16, %6 {handshake.bb = 1 : ui32, handshake.name = "cond_br9"} : <i1>, <>
    %2 = constant %arg2 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "constant0", value = false} : <>, <i1>
    %3 = extsi %2 {handshake.bb = 1 : ui32, handshake.name = "extsi5"} : <i1> to <i4>
    %4 = extui %2 {handshake.bb = 1 : ui32, handshake.name = "extui0"} : <i1> to <i3>
    %5 = merge %2, %16 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "merge0"} : <i1>
    %6 = mux %5 [%arg2, %trueResult_2] {ftd.regen, handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<>, <>] to <>
    %7 = mux %5 [%1, %trueResult_0] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux2"} : <i1>, [<i32>, <i32>] to <i32>
    %8 = mux %5 [%3, %trueResult] {handshake.bb = 1 : ui32, handshake.name = "mux3"} : <i1>, [<i4>, <i4>] to <i4>
    %9 = extsi %8 {handshake.bb = 1 : ui32, handshake.name = "extsi6"} : <i4> to <i5>
    %10 = constant %6 {handshake.bb = 1 : ui32, handshake.name = "constant7", value = 8 : i5} : <>, <i5>
    %11 = constant %6 {handshake.bb = 1 : ui32, handshake.name = "constant8", value = 1 : i2} : <>, <i2>
    %12 = extsi %11 {handshake.bb = 1 : ui32, handshake.name = "extsi7"} : <i2> to <i5>
    %13 = muli %7, %7 {handshake.bb = 1 : ui32, handshake.name = "muli0"} : <i32>
    %14 = addi %9, %12 {handshake.bb = 1 : ui32, handshake.name = "addi1"} : <i5>
    %15 = trunci %14 {handshake.bb = 1 : ui32, handshake.name = "trunci0"} : <i5> to <i4>
    %16 = cmpi ult, %14, %10 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i5>
    %addressResult, %dataResult = load[%4] %outputs {handshake.bb = 2 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i3>, <i32>, <i3>, <i32>
    %17 = addi %falseResult_1, %dataResult {handshake.bb = 2 : ui32, handshake.name = "addi0"} : <i32>
    end {handshake.bb = 2 : ui32, handshake.name = "end0"} %17, %memEnd, %arg2 : <i32>, <>, <>
  }
}

