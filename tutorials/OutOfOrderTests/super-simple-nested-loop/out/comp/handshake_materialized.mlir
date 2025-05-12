module {
  handshake.func @bicg(%arg0: memref<30xi32>, %arg1: !handshake.control<>, %arg2: !handshake.control<>, ...) -> (!handshake.channel<i32>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["q", "q_start", "start"], cfg.edges = "[0,1][2,2,3,cmpi0][1,2][3,1,4,cmpi1]", resNames = ["out0", "q_end", "end"]} {
    %0:5 = fork [5] %arg2 {handshake.bb = 0 : ui32, handshake.name = "fork0"} : <>
    %memEnd = mem_controller[%arg0 : memref<30xi32>] %arg1 (%34, %addressResult, %dataResult) %0#4 {connectedBlocks = [3 : i32], handshake.name = "mem_controller1"} :    (!handshake.channel<i32>, !handshake.channel<i5>, !handshake.channel<i32>) -> ()
    %1 = constant %0#3 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "constant0", value = false} : <>, <i1>
    %2:2 = fork [2] %1 {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <i1>
    %3 = extsi %2#1 {handshake.bb = 1 : ui32, handshake.name = "extsi7"} : <i1> to <i6>
    %4 = merge %2#0, %43#0 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "merge0"} : <i1>
    %5:2 = fork [2] %4 {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <i1>
    %6 = mux %5#1 [%0#2, %trueResult_8] {ftd.regen, handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<>, <>] to <>
    %7:5 = fork [5] %6 {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <>
    %8 = mux %5#0 [%3, %trueResult_6] {handshake.bb = 1 : ui32, handshake.name = "mux5"} : <i1>, [<i6>, <i6>] to <i6>
    %9:3 = fork [3] %8 {handshake.bb = 1 : ui32, handshake.name = "fork4"} : <i6>
    %10 = extsi %9#2 {handshake.bb = 1 : ui32, handshake.name = "extsi8"} : <i6> to <i7>
    %11 = extsi %9#1 {handshake.bb = 1 : ui32, handshake.name = "extsi9"} : <i6> to <i32>
    %12 = trunci %9#0 {handshake.bb = 1 : ui32, handshake.name = "trunci0"} : <i6> to <i5>
    %13 = constant %7#4 {handshake.bb = 1 : ui32, handshake.name = "constant8", value = false} : <>, <i1>
    %14 = extsi %13 {handshake.bb = 1 : ui32, handshake.name = "extsi10"} : <i1> to <i6>
    %trueResult, %falseResult = cond_br %32#0, %27 {handshake.bb = 2 : ui32, handshake.name = "cond_br17"} : <i1>, <i32>
    %15:2 = fork [2] %falseResult {handshake.bb = 2 : ui32, handshake.name = "fork5"} : <i32>
    %trueResult_0, %falseResult_1 = cond_br %32#1, %30 {handshake.bb = 2 : ui32, handshake.name = "cond_br18"} : <i1>, <i6>
    sink %falseResult_1 {handshake.name = "sink0"} : <i6>
    %trueResult_2, %falseResult_3 = cond_br %32#2, %17#2 {handshake.bb = 2 : ui32, handshake.name = "cond_br19"} : <i1>, <>
    sink %falseResult_3 {handshake.name = "sink1"} : <>
    %result, %index = control_merge [%7#3, %trueResult_2]  {handshake.bb = 2 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %16:2 = fork [2] %index {handshake.bb = 2 : ui32, handshake.name = "fork6"} : <i1>
    %17:3 = fork [3] %result {handshake.bb = 2 : ui32, handshake.name = "fork7"} : <>
    %18 = mux %16#1 [%11, %trueResult] {ftd.phi, handshake.bb = 2 : ui32, handshake.name = "mux8"} : <i1>, [<i32>, <i32>] to <i32>
    %19 = mux %16#0 [%14, %trueResult_0] {handshake.bb = 2 : ui32, handshake.name = "mux9"} : <i1>, [<i6>, <i6>] to <i6>
    %20:2 = fork [2] %19 {handshake.bb = 2 : ui32, handshake.name = "fork8"} : <i6>
    %21 = extsi %20#1 {handshake.bb = 2 : ui32, handshake.name = "extsi11"} : <i6> to <i7>
    %22 = extsi %20#0 {handshake.bb = 2 : ui32, handshake.name = "extsi12"} : <i6> to <i32>
    %23 = constant %17#1 {handshake.bb = 2 : ui32, handshake.name = "constant9", value = 30 : i6} : <>, <i6>
    %24 = extsi %23 {handshake.bb = 2 : ui32, handshake.name = "extsi13"} : <i6> to <i7>
    %25 = constant %17#0 {handshake.bb = 2 : ui32, handshake.name = "constant10", value = 1 : i2} : <>, <i2>
    %26 = extsi %25 {handshake.bb = 2 : ui32, handshake.name = "extsi14"} : <i2> to <i7>
    %27 = muli %18, %22 {handshake.bb = 2 : ui32, handshake.name = "muli0"} : <i32>
    %28 = addi %21, %26 {handshake.bb = 2 : ui32, handshake.name = "addi1"} : <i7>
    %29:2 = fork [2] %28 {handshake.bb = 2 : ui32, handshake.name = "fork9"} : <i7>
    %30 = trunci %29#1 {handshake.bb = 2 : ui32, handshake.name = "trunci1"} : <i7> to <i6>
    %31 = cmpi ult, %29#0, %24 {handshake.bb = 2 : ui32, handshake.name = "cmpi0"} : <i7>
    %32:3 = fork [3] %31 {handshake.bb = 2 : ui32, handshake.name = "fork10"} : <i1>
    %trueResult_4, %falseResult_5 = cond_br %43#1, %15#1 {handshake.bb = 3 : ui32, handshake.name = "cond_br20"} : <i1>, <i32>
    sink %trueResult_4 {handshake.name = "sink2"} : <i32>
    %trueResult_6, %falseResult_7 = cond_br %43#2, %41 {handshake.bb = 3 : ui32, handshake.name = "cond_br21"} : <i1>, <i6>
    sink %falseResult_7 {handshake.name = "sink3"} : <i6>
    %trueResult_8, %falseResult_9 = cond_br %43#3, %7#2 {handshake.bb = 3 : ui32, handshake.name = "cond_br22"} : <i1>, <>
    sink %falseResult_9 {handshake.name = "sink4"} : <>
    %33 = constant %0#1 {handshake.bb = 3 : ui32, handshake.name = "constant11", value = 1 : i2} : <>, <i2>
    %34 = extsi %33 {handshake.bb = 3 : ui32, handshake.name = "extsi4"} : <i2> to <i32>
    %35 = constant %7#1 {handshake.bb = 3 : ui32, handshake.name = "constant13", value = 30 : i6} : <>, <i6>
    %36 = extsi %35 {handshake.bb = 3 : ui32, handshake.name = "extsi15"} : <i6> to <i7>
    %37 = constant %7#0 {handshake.bb = 3 : ui32, handshake.name = "constant14", value = 1 : i2} : <>, <i2>
    %38 = extsi %37 {handshake.bb = 3 : ui32, handshake.name = "extsi16"} : <i2> to <i7>
    %addressResult, %dataResult = store[%12] %15#0 {handshake.bb = 3 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store0"} : <i5>, <i32>, <i5>, <i32>
    %39 = addi %10, %38 {handshake.bb = 3 : ui32, handshake.name = "addi2"} : <i7>
    %40:2 = fork [2] %39 {handshake.bb = 3 : ui32, handshake.name = "fork11"} : <i7>
    %41 = trunci %40#1 {handshake.bb = 3 : ui32, handshake.name = "trunci2"} : <i7> to <i6>
    %42 = cmpi ult, %40#0, %36 {handshake.bb = 3 : ui32, handshake.name = "cmpi1"} : <i7>
    %43:4 = fork [4] %42 {handshake.bb = 3 : ui32, handshake.name = "fork12"} : <i1>
    end {handshake.bb = 4 : ui32, handshake.name = "end0"} %falseResult_5, %memEnd, %0#0 : <i32>, <>, <>
  }
}

