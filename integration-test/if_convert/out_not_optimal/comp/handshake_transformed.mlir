module {
  handshake.func @if_convert(%arg0: memref<200xi32>, %arg1: memref<200xi32>, %arg2: !handshake.control<>, %arg3: !handshake.control<>, %arg4: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["a", "b", "a_start", "b_start", "start"], resNames = ["a_end", "b_end", "end"]} {
    %0:3 = fork [3] %arg4 {handshake.bb = 0 : ui32, handshake.name = "fork0"} : <>
    %memEnd = mem_controller[%arg1 : memref<200xi32>] %arg3 (%27, %addressResult_7, %dataResult_8) %49#1 {connectedBlocks = [3 : i32], handshake.name = "mem_controller2"} :    (!handshake.channel<i32>, !handshake.channel<i8>, !handshake.channel<i32>) -> ()
    %outputs, %memEnd_0 = mem_controller[%arg0 : memref<200xi32>] %arg2 (%addressResult) %49#0 {connectedBlocks = [3 : i32], handshake.name = "mem_controller3"} :    (!handshake.channel<i8>) -> !handshake.channel<i32>
    %1 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant2", value = 1 : i2} : <>, <i2>
    %2 = extsi %1 {handshake.bb = 0 : ui32, handshake.name = "extsi7"} : <i2> to <i32>
    %3:2 = fork [2] %2 {handshake.bb = 0 : ui32, handshake.name = "fork1"} : <i32>
    %4 = mux %6#0 [%3#0, %48#0] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i32>, <i32>] to <i32>
    %5 = mux %6#1 [%3#1, %48#1] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i1>, [<i32>, <i32>] to <i32>
    %result, %index = control_merge [%0#2, %trueResult_11]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %6:2 = fork [2] %index {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <i1>
    %7 = mux %11#0 [%4, %39#1] {handshake.bb = 2 : ui32, handshake.name = "mux2"} : <i1>, [<i32>, <i32>] to <i32>
    %8:2 = fork [2] %7 {handshake.bb = 2 : ui32, handshake.name = "fork3"} : <i32>
    %9 = mux %11#1 [%5, %41] {handshake.bb = 2 : ui32, handshake.name = "mux3"} : <i1>, [<i32>, <i32>] to <i32>
    %10:3 = fork [3] %9 {handshake.bb = 2 : ui32, handshake.name = "fork4"} : <i32>
    %result_1, %index_2 = control_merge [%result, %22#1]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>, <>] to <>, <i1>
    %11:2 = fork [2] %index_2 {handshake.bb = 2 : ui32, handshake.name = "fork5"} : <i1>
    %12 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %13 = constant %12 {handshake.bb = 2 : ui32, handshake.name = "constant9", value = 199 : i9} : <>, <i9>
    %14 = extsi %13 {handshake.bb = 2 : ui32, handshake.name = "extsi1"} : <i9> to <i32>
    %15 = cmpi slt, %10#2, %14 {handshake.bb = 2 : ui32, handshake.name = "cmpi0"} : <i32>
    %16 = cmpi eq, %10#1, %8#1 {handshake.bb = 2 : ui32, handshake.name = "cmpi1"} : <i32>
    %17 = andi %15, %16 {handshake.bb = 2 : ui32, handshake.name = "andi0"} : <i1>
    %18:3 = fork [3] %17 {handshake.bb = 2 : ui32, handshake.name = "fork6"} : <i1>
    %trueResult, %falseResult = cond_br %18#2, %8#0 {handshake.bb = 2 : ui32, handshake.name = "cond_br2"} : <i1>, <i32>
    %trueResult_3, %falseResult_4 = cond_br %18#1, %10#0 {handshake.bb = 2 : ui32, handshake.name = "cond_br3"} : <i1>, <i32>
    sink %falseResult_4 {handshake.name = "sink0"} : <i32>
    %trueResult_5, %falseResult_6 = cond_br %18#0, %result_1 {handshake.bb = 2 : ui32, handshake.name = "cond_br4"} : <i1>, <>
    %19:2 = fork [2] %trueResult {handshake.bb = 3 : ui32, handshake.name = "fork7"} : <i32>
    %20 = trunci %19#0 {handshake.bb = 3 : ui32, handshake.name = "trunci0"} : <i32> to <i8>
    %21:3 = fork [3] %trueResult_3 {handshake.bb = 3 : ui32, handshake.name = "fork8"} : <i32>
    %22:2 = fork [2] %trueResult_5 {handshake.bb = 3 : ui32, handshake.name = "fork9"} : <>
    %23 = constant %22#0 {handshake.bb = 3 : ui32, handshake.name = "constant10", value = 1 : i2} : <>, <i2>
    %24:2 = fork [2] %23 {handshake.bb = 3 : ui32, handshake.name = "fork10"} : <i2>
    %25 = extsi %24#0 {handshake.bb = 3 : ui32, handshake.name = "extsi2"} : <i2> to <i32>
    %26:3 = fork [3] %25 {handshake.bb = 3 : ui32, handshake.name = "fork11"} : <i32>
    %27 = extsi %24#1 {handshake.bb = 3 : ui32, handshake.name = "extsi3"} : <i2> to <i32>
    %28 = source {handshake.bb = 3 : ui32, handshake.name = "source1"} : <>
    %29 = constant %28 {handshake.bb = 3 : ui32, handshake.name = "constant11", value = 10000 : i15} : <>, <i15>
    %30 = extsi %29 {handshake.bb = 3 : ui32, handshake.name = "extsi4"} : <i15> to <i32>
    %31 = source {handshake.bb = 3 : ui32, handshake.name = "source2"} : <>
    %32 = constant %31 {handshake.bb = 3 : ui32, handshake.name = "constant12", value = 2 : i3} : <>, <i3>
    %33 = extsi %32 {handshake.bb = 3 : ui32, handshake.name = "extsi5"} : <i3> to <i32>
    %addressResult, %dataResult = load[%20] %outputs {handshake.bb = 3 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i8>, <i32>, <i8>, <i32>
    %34 = muli %19#1, %dataResult {handshake.bb = 3 : ui32, handshake.name = "muli0"} : <i32>
    %35 = cmpi slt, %34, %30 {handshake.bb = 3 : ui32, handshake.name = "cmpi2"} : <i32>
    %36 = addi %21#2, %33 {handshake.bb = 3 : ui32, handshake.name = "addi0"} : <i32>
    %37 = addi %21#1, %26#1 {handshake.bb = 3 : ui32, handshake.name = "addi1"} : <i32>
    %38 = select %35[%36, %37] {handshake.bb = 3 : ui32, handshake.name = "select1"} : <i1>, <i32>
    %39:2 = fork [2] %38 {handshake.bb = 3 : ui32, handshake.name = "fork12"} : <i32>
    %40 = trunci %39#0 {handshake.bb = 3 : ui32, handshake.name = "trunci1"} : <i32> to <i8>
    %addressResult_7, %dataResult_8 = store[%40] %26#0 {handshake.bb = 3 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store1"} : <i8>, <i32>, <i8>, <i32>
    %41 = addi %21#0, %26#2 {handshake.bb = 3 : ui32, handshake.name = "addi2"} : <i32>
    %42:2 = fork [2] %falseResult {handshake.bb = 4 : ui32, handshake.name = "fork13"} : <i32>
    %43 = source {handshake.bb = 4 : ui32, handshake.name = "source3"} : <>
    %44 = constant %43 {handshake.bb = 4 : ui32, handshake.name = "constant13", value = 199 : i9} : <>, <i9>
    %45 = extsi %44 {handshake.bb = 4 : ui32, handshake.name = "extsi6"} : <i9> to <i32>
    %46 = cmpi slt, %42#1, %45 {handshake.bb = 4 : ui32, handshake.name = "cmpi3"} : <i32>
    %47:2 = fork [2] %46 {handshake.bb = 4 : ui32, handshake.name = "fork14"} : <i1>
    %trueResult_9, %falseResult_10 = cond_br %47#1, %42#0 {handshake.bb = 4 : ui32, handshake.name = "cond_br5"} : <i1>, <i32>
    sink %falseResult_10 {handshake.name = "sink3"} : <i32>
    %48:2 = fork [2] %trueResult_9 {handshake.bb = 4 : ui32, handshake.name = "fork15"} : <i32>
    %trueResult_11, %falseResult_12 = cond_br %47#0, %falseResult_6 {handshake.bb = 4 : ui32, handshake.name = "cond_br6"} : <i1>, <>
    %49:2 = fork [2] %falseResult_12 {handshake.bb = 5 : ui32, handshake.name = "fork16"} : <>
    end {handshake.bb = 5 : ui32, handshake.name = "end0"} %memEnd_0, %memEnd, %0#1 : <>, <>, <>
  }
}

