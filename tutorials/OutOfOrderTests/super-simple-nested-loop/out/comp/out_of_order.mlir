module {
  handshake.func @bicg(%arg0: memref<30xi32>, %arg1: !handshake.control<>, %arg2: !handshake.control<>, ...) -> (!handshake.channel<i32>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["q", "q_start", "start"], cfg.edges = "[0,1][2,2,3,cmpi0][1,2][3,1,4,cmpi1]", resNames = ["out0", "q_end", "end"]} {
    %memEnd = mem_controller[%arg0 : memref<30xi32>] %arg1 (%23, %addressResult, %dataResult) %arg2 {connectedBlocks = [3 : i32], handshake.name = "mem_controller1"} :    (!handshake.channel<i32>, !handshake.channel<i5>, !handshake.channel<i32>) -> ()
    %0 = constant %arg2 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "constant0", value = false} : <>, <i1>
    %1 = extsi %0 {handshake.bb = 1 : ui32, handshake.name = "extsi7"} : <i1> to <i6>
    %2 = merge %0, %30 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "merge0"} : <i1>
    %3 = mux %2 [%arg2, %trueResult_8] {ftd.regen, handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<>, <>] to <>
    %4 = mux %2 [%1, %trueResult_6] {handshake.bb = 1 : ui32, handshake.name = "mux5"} : <i1>, [<i6>, <i6>] to <i6>
    %5 = extsi %4 {handshake.bb = 1 : ui32, handshake.name = "extsi8"} : <i6> to <i7>
    %6 = extsi %4 {handshake.bb = 1 : ui32, handshake.name = "extsi9"} : <i6> to <i32>
    %7 = trunci %4 {handshake.bb = 1 : ui32, handshake.name = "trunci0"} : <i6> to <i5>
    %8 = constant %3 {handshake.bb = 1 : ui32, handshake.name = "constant8", value = false} : <>, <i1>
    %9 = extsi %8 {handshake.bb = 1 : ui32, handshake.name = "extsi10"} : <i1> to <i6>
    %trueResult, %falseResult = cond_br %21, %18 {handshake.bb = 2 : ui32, handshake.name = "cond_br17"} : <i1>, <i32>
    %trueResult_0, %falseResult_1 = cond_br %21, %20 {handshake.bb = 2 : ui32, handshake.name = "cond_br18"} : <i1>, <i6>
    %trueResult_2, %falseResult_3 = cond_br %21, %result {handshake.bb = 2 : ui32, handshake.name = "cond_br19"} : <i1>, <>
    %result, %index = control_merge [%3, %trueResult_2]  {handshake.bb = 2 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %10 = mux %index [%6, %trueResult] {ftd.phi, handshake.bb = 2 : ui32, handshake.name = "mux8"} : <i1>, [<i32>, <i32>] to <i32>
    %11 = mux %index [%9, %trueResult_0] {handshake.bb = 2 : ui32, handshake.name = "mux9"} : <i1>, [<i6>, <i6>] to <i6>
    %12 = extsi %11 {handshake.bb = 2 : ui32, handshake.name = "extsi11"} : <i6> to <i7>
    %13 = extsi %11 {handshake.bb = 2 : ui32, handshake.name = "extsi12"} : <i6> to <i32>
    %14 = constant %result {handshake.bb = 2 : ui32, handshake.name = "constant9", value = 30 : i6} : <>, <i6>
    %15 = extsi %14 {handshake.bb = 2 : ui32, handshake.name = "extsi13"} : <i6> to <i7>
    %16 = constant %result {handshake.bb = 2 : ui32, handshake.name = "constant10", value = 1 : i2} : <>, <i2>
    %17 = extsi %16 {handshake.bb = 2 : ui32, handshake.name = "extsi14"} : <i2> to <i7>
    %18 = muli %10, %13 {handshake.bb = 2 : ui32, handshake.name = "muli0"} : <i32>
    %19 = addi %12, %17 {handshake.bb = 2 : ui32, handshake.name = "addi1"} : <i7>
    %20 = trunci %19 {handshake.bb = 2 : ui32, handshake.name = "trunci1"} : <i7> to <i6>
    %21 = cmpi ult, %19, %15 {handshake.bb = 2 : ui32, handshake.name = "cmpi0"} : <i7>
    %trueResult_4, %falseResult_5 = cond_br %30, %falseResult {handshake.bb = 3 : ui32, handshake.name = "cond_br20"} : <i1>, <i32>
    %trueResult_6, %falseResult_7 = cond_br %30, %29 {handshake.bb = 3 : ui32, handshake.name = "cond_br21"} : <i1>, <i6>
    %trueResult_8, %falseResult_9 = cond_br %30, %3 {handshake.bb = 3 : ui32, handshake.name = "cond_br22"} : <i1>, <>
    %22 = constant %arg2 {handshake.bb = 3 : ui32, handshake.name = "constant11", value = 1 : i2} : <>, <i2>
    %23 = extsi %22 {handshake.bb = 3 : ui32, handshake.name = "extsi4"} : <i2> to <i32>
    %24 = constant %3 {handshake.bb = 3 : ui32, handshake.name = "constant13", value = 30 : i6} : <>, <i6>
    %25 = extsi %24 {handshake.bb = 3 : ui32, handshake.name = "extsi15"} : <i6> to <i7>
    %26 = constant %3 {handshake.bb = 3 : ui32, handshake.name = "constant14", value = 1 : i2} : <>, <i2>
    %27 = extsi %26 {handshake.bb = 3 : ui32, handshake.name = "extsi16"} : <i2> to <i7>
    %addressResult, %dataResult = store[%7] %falseResult {handshake.bb = 3 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store0"} : <i5>, <i32>, <i5>, <i32>
    %28 = addi %5, %27 {handshake.bb = 3 : ui32, handshake.name = "addi2"} : <i7>
    %29 = trunci %28 {handshake.bb = 3 : ui32, handshake.name = "trunci2"} : <i7> to <i6>
    %30 = cmpi ult, %28, %25 {handshake.bb = 3 : ui32, handshake.name = "cmpi1"} : <i7>
    end {handshake.bb = 4 : ui32, handshake.name = "end0"} %falseResult_5, %memEnd, %arg2 : <i32>, <>, <>
  }
}

