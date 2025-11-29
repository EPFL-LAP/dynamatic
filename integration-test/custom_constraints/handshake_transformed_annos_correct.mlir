module {
  handshake.func @custom_constraints(%arg0: memref<1000xi32>, %arg1: !handshake.control<>, %arg2: !handshake.control<>, ...) -> (!handshake.channel<i32>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["di", "di_start", "start"], resNames = ["out0", "di_end", "end"]} {
    %0:3 = fork [3] %arg2 {handshake.bb = 0 : ui32, handshake.name = "fork0"} : <>
    %memEnd = mem_controller[%arg0 : memref<1000xi32>] %arg1 (%14, %addressResult, %dataResult) %result_2 {connectedBlocks = [1 : i32], handshake.name = "mem_controller1"} :    (!handshake.channel<i32>, !handshake.channel<i10>, !handshake.channel<i32>) -> ()
    %1 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant6", value = false} : <>, <i1>
    %2 = br %1 {handshake.bb = 0 : ui32, handshake.name = "br1"} : <i1>
    %3 = extsi %2 {handshake.bb = 0 : ui32, handshake.name = "extsi5"} : <i1> to <i11>
    %4 = br %0#2 {handshake.bb = 0 : ui32, handshake.name = "br2"} : <>
    %5 = mux %index [%3, %trueResult] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i11>, <i11>] to <i11>
    %6:5 = fork [5] %5 {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <i11>
    %7 = extsi %6#1 {handshake.bb = 1 : ui32, handshake.name = "extsi6"} : <i11> to <i12>
    %8 = extsi %6#2 {handshake.bb = 1 : ui32, handshake.name = "extsi7"} : <i11> to <i22>
    %9 = extsi %6#3 {handshake.bb = 1 : ui32, handshake.name = "extsi8"} : <i11> to <i22>
    %10 = extsi %6#4 {handshake.bb = 1 : ui32, handshake.name = "extsi9", handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>} : <i11> to <i32>
    %11 = trunci %6#0 {handshake.bb = 1 : ui32, handshake.name = "trunci0", handshake.bufProps = #handshake<bufProps{"0": [12,12], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>} : <i11> to <i10>
    %result, %index = control_merge [%4, %trueResult_0]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %12:2 = fork [2] %result {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <>
    %13 = constant %12#0 {handshake.bb = 1 : ui32, handshake.name = "constant7", value = 1 : i2} : <>, <i2>
    %14 = extsi %13 {handshake.bb = 1 : ui32, handshake.name = "extsi1"} : <i2> to <i32>
    %15 = source {handshake.bb = 1 : ui32, handshake.name = "source0"} : <>
    %16 = constant %15 {handshake.bb = 1 : ui32, handshake.name = "constant8", value = 1000 : i11} : <>, <i11>
    %17 = extsi %16 {handshake.bb = 1 : ui32, handshake.name = "extsi10"} : <i11> to <i12>
    %18 = source {handshake.bb = 1 : ui32, handshake.name = "source1"} : <>
    %19 = constant %18 {handshake.bb = 1 : ui32, handshake.name = "constant9", value = 1 : i2} : <>, <i2>
    %20:2 = fork [2] %19 {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <i2>
    %21 = extsi %20#0 {handshake.bb = 1 : ui32, handshake.name = "extsi11"} : <i2> to <i12>
    %22 = extsi %20#1 {handshake.bb = 1 : ui32, handshake.name = "extsi3"} : <i2> to <i32>
    %23 = muli %9, %8 {handshake.bb = 1 : ui32, handshake.name = "muli0"} : <i22>
    %24 = extsi %23 {handshake.bb = 1 : ui32, handshake.name = "extsi12"} : <i22> to <i32>
    %25 = muli %24, %10 {handshake.bb = 1 : ui32, handshake.name = "muli1", handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>} : <i32>
    %26:2 = fork [2] %25 {handshake.bb = 1 : ui32, handshake.name = "fork4"} : <i32>
    %27 = muli %26#0, %26#1 {handshake.bb = 1 : ui32, handshake.name = "muli2"} : <i32>
    %28:2 = fork [2] %27 {handshake.bb = 1 : ui32, handshake.name = "fork5"} : <i32>
    %29 = muli %28#0, %28#1 {handshake.bb = 1 : ui32, handshake.name = "muli3"} : <i32>
    %30:2 = fork [2] %29 {handshake.bb = 1 : ui32, handshake.name = "fork6"} : <i32>
    %31 = muli %30#0, %30#1 {handshake.bb = 1 : ui32, handshake.name = "muli4"} : <i32>
    %32:2 = fork [2] %31 {handshake.bb = 1 : ui32, handshake.name = "fork7"} : <i32>
    %33 = muli %32#0, %32#1 {handshake.bb = 1 : ui32, handshake.name = "muli5"} : <i32>
    %34 = addi %33, %22 {handshake.bb = 1 : ui32, handshake.name = "addi0"} : <i32>
    %addressResult, %dataResult = store[%11] %34 {handshake.bb = 1 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store0"} : <i10>, <i32>, <i10>, <i32>
    %35 = addi %7, %21 {handshake.bb = 1 : ui32, handshake.name = "addi1"} : <i12>
    %36:2 = fork [2] %35 {handshake.bb = 1 : ui32, handshake.name = "fork8"} : <i12>
    %37 = trunci %36#0 {handshake.bb = 1 : ui32, handshake.name = "trunci1"} : <i12> to <i11>
    %38 = cmpi ult, %36#1, %17 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i12>
    %39:2 = fork [2] %38 {handshake.bb = 1 : ui32, handshake.name = "fork9"} : <i1>
    %trueResult, %falseResult = cond_br %39#0, %37 {handshake.bb = 1 : ui32, handshake.name = "cond_br1"} : <i1>, <i11>
    sink %falseResult {handshake.name = "sink0"} : <i11>
    %trueResult_0, %falseResult_1 = cond_br %39#1, %12#1 {handshake.bb = 1 : ui32, handshake.name = "cond_br2"} : <i1>, <>
    %result_2, %index_3 = control_merge [%falseResult_1]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>] to <>, <i1>
    sink %index_3 {handshake.name = "sink1"} : <i1>
    %40 = source {handshake.bb = 2 : ui32, handshake.name = "source2"} : <>
    %41 = constant %40 {handshake.bb = 2 : ui32, handshake.name = "constant10", value = false} : <>, <i1>
    %42 = extsi %41 {handshake.bb = 2 : ui32, handshake.name = "extsi4"} : <i1> to <i32>
    end {handshake.bb = 2 : ui32, handshake.name = "end0"} %42, %memEnd, %0#1 : <i32>, <>, <>
  }
}

