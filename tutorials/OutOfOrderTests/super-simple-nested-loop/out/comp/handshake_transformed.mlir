module {
  handshake.func @bicg(%arg0: memref<30xi32>, %arg1: !handshake.control<>, %arg2: !handshake.control<>, ...) -> (!handshake.channel<i32>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["q", "q_start", "start"], cfg.edges = "[0,1][2,2,3,cmpi0][1,2][3,1,4,cmpi1]", resNames = ["out0", "q_end", "end"]} {
    %memEnd = mem_controller[%arg0 : memref<30xi32>] %arg1 (%26, %addressResult, %dataResult) %arg2 {connectedBlocks = [3 : i32], handshake.name = "mem_controller1"} :    (!handshake.channel<i32>, !handshake.channel<i5>, !handshake.channel<i32>) -> ()
    %0 = constant %arg2 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "constant0", value = false} : <>, <i1>
    %1 = extsi %0 {handshake.bb = 1 : ui32, handshake.name = "extsi7"} : <i1> to <i6>
    %2 = merge %0, %33 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "merge0"} : <i1>
    %3 = mux %2 [%arg2, %trueResult_8] {ftd.regen, handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<>, <>] to <>
    %4 = mux %2 [%1, %trueResult_6] {handshake.bb = 1 : ui32, handshake.name = "mux5"} : <i1>, [<i6>, <i6>] to <i6>
    %5 = extsi %4 {handshake.bb = 1 : ui32, handshake.name = "extsi8"} : <i6> to <i7>
    %6 = extsi %4 {handshake.bb = 1 : ui32, handshake.name = "extsi9"} : <i6> to <i32>
    %7 = trunci %4 {handshake.bb = 1 : ui32, handshake.name = "trunci0"} : <i6> to <i5>
    %8 = constant %3 {handshake.bb = 1 : ui32, handshake.name = "constant8", value = false} : <>, <i1>
    %9 = extsi %8 {handshake.bb = 1 : ui32, handshake.name = "extsi10"} : <i1> to <i6>
    %trueResult, %falseResult = cond_br %24, %21 {handshake.bb = 2 : ui32, handshake.name = "cond_br17"} : <i1>, <i32>
    %trueResult_0, %falseResult_1 = cond_br %24, %23 {handshake.bb = 2 : ui32, handshake.name = "cond_br18"} : <i1>, <i6>
    %trueResult_2, %falseResult_3 = cond_br %24, %12 {handshake.bb = 2 : ui32, handshake.name = "cond_br19"} : <i1>, <>
    %10 = constant %arg2 {ftd.imerge, handshake.bb = 2 : ui32, handshake.name = "constant12", value = false} : <>, <i1>
    %11 = merge %10, %24 {ftd.imerge, handshake.bb = 2 : ui32, handshake.name = "merge6"} : <i1>
    %12 = mux %11 [%3, %trueResult_2] {ftd.regen, handshake.bb = 2 : ui32, handshake.name = "mux6"} : <i1>, [<>, <>] to <>
    %13 = mux %11 [%6, %trueResult] {ftd.phi, handshake.bb = 2 : ui32, handshake.name = "mux8"} : <i1>, [<i32>, <i32>] to <i32>
    %14 = mux %11 [%9, %trueResult_0] {handshake.bb = 2 : ui32, handshake.name = "mux9"} : <i1>, [<i6>, <i6>] to <i6>
    %15 = extsi %14 {handshake.bb = 2 : ui32, handshake.name = "extsi11"} : <i6> to <i7>
    %16 = extsi %14 {handshake.bb = 2 : ui32, handshake.name = "extsi12"} : <i6> to <i32>
    %17 = constant %12 {handshake.bb = 2 : ui32, handshake.name = "constant9", value = 30 : i6} : <>, <i6>
    %18 = extsi %17 {handshake.bb = 2 : ui32, handshake.name = "extsi13"} : <i6> to <i7>
    %19 = constant %12 {handshake.bb = 2 : ui32, handshake.name = "constant10", value = 1 : i2} : <>, <i2>
    %20 = extsi %19 {handshake.bb = 2 : ui32, handshake.name = "extsi14"} : <i2> to <i7>
    %21 = muli %13, %16 {handshake.bb = 2 : ui32, handshake.name = "muli0"} : <i32>
    %22 = addi %15, %20 {handshake.bb = 2 : ui32, handshake.name = "addi1"} : <i7>
    %23 = trunci %22 {handshake.bb = 2 : ui32, handshake.name = "trunci1"} : <i7> to <i6>
    %24 = cmpi ult, %22, %18 {handshake.bb = 2 : ui32, handshake.name = "cmpi0"} : <i7>
    %trueResult_4, %falseResult_5 = cond_br %33, %falseResult {handshake.bb = 3 : ui32, handshake.name = "cond_br20"} : <i1>, <i32>
    %trueResult_6, %falseResult_7 = cond_br %33, %32 {handshake.bb = 3 : ui32, handshake.name = "cond_br21"} : <i1>, <i6>
    %trueResult_8, %falseResult_9 = cond_br %33, %3 {handshake.bb = 3 : ui32, handshake.name = "cond_br22"} : <i1>, <>
    %25 = constant %arg2 {handshake.bb = 3 : ui32, handshake.name = "constant11", value = 1 : i2} : <>, <i2>
    %26 = extsi %25 {handshake.bb = 3 : ui32, handshake.name = "extsi4"} : <i2> to <i32>
    %27 = constant %3 {handshake.bb = 3 : ui32, handshake.name = "constant13", value = 30 : i6} : <>, <i6>
    %28 = extsi %27 {handshake.bb = 3 : ui32, handshake.name = "extsi15"} : <i6> to <i7>
    %29 = constant %3 {handshake.bb = 3 : ui32, handshake.name = "constant14", value = 1 : i2} : <>, <i2>
    %30 = extsi %29 {handshake.bb = 3 : ui32, handshake.name = "extsi16"} : <i2> to <i7>
    %addressResult, %dataResult = store[%7] %falseResult {handshake.bb = 3 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store0"} : <i5>, <i32>, <i5>, <i32>
    %31 = addi %5, %30 {handshake.bb = 3 : ui32, handshake.name = "addi2"} : <i7>
    %32 = trunci %31 {handshake.bb = 3 : ui32, handshake.name = "trunci2"} : <i7> to <i6>
    %33 = cmpi ult, %31, %28 {handshake.bb = 3 : ui32, handshake.name = "cmpi1"} : <i7>
    end {handshake.bb = 4 : ui32, handshake.name = "end0"} %falseResult_5, %memEnd, %arg2 : <i32>, <>, <>
  }
}

