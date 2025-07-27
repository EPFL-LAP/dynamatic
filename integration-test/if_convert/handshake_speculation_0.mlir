module {
  handshake.func @if_convert(%arg0: memref<200xi32>, %arg1: memref<200xi32>, %arg2: !handshake.control<>, %arg3: !handshake.control<>, %arg4: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["a", "b", "a_start", "b_start", "start"], resNames = ["a_end", "b_end", "end"]} {
    %0 = init %21#6 {handshake.bb = 2 : ui32, handshake.name = "init0", initToken = 0 : ui1} : <i1>
    %1:3 = fork [3] %0 {handshake.bb = 2 : ui32, handshake.name = "fork16"} : <i1>
    %2:3 = fork [3] %arg4 {handshake.bb = 0 : ui32, handshake.name = "fork0"} : <>
    %memEnd = mem_controller[%arg1 : memref<200xi32>] %arg3 (%28, %addressResult_1, %dataResult_2) %59#1 {connectedBlocks = [2 : i32], handshake.name = "mem_controller2"} :    (!handshake.channel<i32>, !handshake.channel<i8>, !handshake.channel<i32>) -> ()
    %outputs, %memEnd_0 = mem_controller[%arg0 : memref<200xi32>] %arg2 (%addressResult) %59#0 {connectedBlocks = [2 : i32], handshake.name = "mem_controller3"} :    (!handshake.channel<i8>) -> !handshake.channel<i32>
    %3 = constant %2#0 {handshake.bb = 0 : ui32, handshake.name = "constant2", value = 1 : i2} : <>, <i2>
    %4 = extsi %3 {handshake.bb = 0 : ui32, handshake.name = "extsi7"} : <i2> to <i32>
    %5:2 = fork [2] %4 {handshake.bb = 0 : ui32, handshake.name = "fork1"} : <i32>
    %6 = mux %8#0 [%5#0, %58#0] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i32>, <i32>] to <i32>
    %7 = mux %8#1 [%5#1, %58#1] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i1>, [<i32>, <i32>] to <i32>
    %result, %index = control_merge [%2#2, %trueResult_3]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %8:2 = fork [2] %index {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <i1>
    %9 = mux %1#2 [%6, %26] {handshake.bb = 2 : ui32, handshake.name = "mux2"} : <i1>, [<i32>, <i32>] to <i32>
    %10:2 = fork [2] %9 {handshake.bb = 2 : ui32, handshake.name = "fork3"} : <i32>
    %11 = mux %1#1 [%7, %24] {handshake.bb = 2 : ui32, handshake.name = "mux3"} : <i1>, [<i32>, <i32>] to <i32>
    %12:7 = fork [7] %11 {handshake.bb = 2 : ui32, handshake.name = "fork17"} : <i32>
    %13 = mux %1#0 [%result, %29] {handshake.bb = 2 : ui32, handshake.name = "mux4"} : <i1>, [<>, <>] to <>
    %14:3 = fork [3] %13 {handshake.bb = 2 : ui32, handshake.name = "fork18"} : <>
    %15 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %16 = constant %15 {handshake.bb = 2 : ui32, handshake.name = "constant9", value = 199 : i9} : <>, <i9>
    %17 = extsi %16 {handshake.bb = 2 : ui32, handshake.name = "extsi1"} : <i9> to <i32>
    %18 = cmpi slt, %12#1, %17 {handshake.bb = 2 : ui32, handshake.name = "cmpi0"} : <i32>
    %19 = cmpi eq, %12#0, %10#1 {handshake.bb = 2 : ui32, handshake.name = "cmpi1"} : <i32>
    %20 = andi %18, %19 {handshake.bb = 2 : ui32, handshake.name = "andi0"} : <i1>
    %21:8 = fork [8] %20 {handshake.bb = 2 : ui32, handshake.name = "fork19"} : <i1>
    %22 = not %21#7 {handshake.bb = 2 : ui32, handshake.name = "not0"} : <i1>
    %23:2 = fork [2] %22 {handshake.bb = 2 : ui32, handshake.name = "fork20"} : <i1>
    %24 = passer %51[%21#2] {handshake.bb = 2 : ui32, handshake.name = "passer0"} : <i32>, <i1>
    %25 = passer %50[%21#0] {handshake.bb = 2 : ui32, handshake.name = "passer1"} : <i8>, <i1>
    %26 = passer %49#1[%21#1] {handshake.bb = 2 : ui32, handshake.name = "passer2"} : <i32>, <i1>
    %27 = passer %36#0[%21#3] {handshake.bb = 2 : ui32, handshake.name = "passer3"} : <i32>, <i1>
    %28 = passer %37[%21#4] {handshake.bb = 2 : ui32, handshake.name = "passer4"} : <i32>, <i1>
    %29 = passer %14#2[%21#5] {handshake.bb = 2 : ui32, handshake.name = "passer5"} : <>, <i1>
    %30 = passer %14#0[%23#1] {handshake.bb = 2 : ui32, handshake.name = "passer6"} : <>, <i1>
    %31 = passer %10#0[%23#0] {handshake.bb = 2 : ui32, handshake.name = "passer7"} : <i32>, <i1>
    %32 = trunci %12#2 {handshake.bb = 2 : ui32, handshake.name = "trunci0"} : <i32> to <i8>
    %33 = constant %14#1 {handshake.bb = 2 : ui32, handshake.name = "constant10", value = 1 : i2} : <>, <i2>
    %34:2 = fork [2] %33 {handshake.bb = 2 : ui32, handshake.name = "fork9"} : <i2>
    %35 = extsi %34#0 {handshake.bb = 2 : ui32, handshake.name = "extsi2"} : <i2> to <i32>
    %36:3 = fork [3] %35 {handshake.bb = 2 : ui32, handshake.name = "fork10"} : <i32>
    %37 = extsi %34#1 {handshake.bb = 2 : ui32, handshake.name = "extsi3"} : <i2> to <i32>
    %38 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %39 = constant %38 {handshake.bb = 2 : ui32, handshake.name = "constant11", value = 10000 : i15} : <>, <i15>
    %40 = extsi %39 {handshake.bb = 2 : ui32, handshake.name = "extsi4"} : <i15> to <i32>
    %41 = source {handshake.bb = 2 : ui32, handshake.name = "source2"} : <>
    %42 = constant %41 {handshake.bb = 2 : ui32, handshake.name = "constant12", value = 2 : i3} : <>, <i3>
    %43 = extsi %42 {handshake.bb = 2 : ui32, handshake.name = "extsi5"} : <i3> to <i32>
    %addressResult, %dataResult = load[%32] %outputs {handshake.bb = 2 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i8>, <i32>, <i8>, <i32>
    %44 = muli %12#6, %dataResult {handshake.bb = 2 : ui32, handshake.name = "muli0"} : <i32>
    %45 = cmpi slt, %44, %40 {handshake.bb = 2 : ui32, handshake.name = "cmpi2"} : <i32>
    %46 = addi %12#5, %43 {handshake.bb = 2 : ui32, handshake.name = "addi0"} : <i32>
    %47 = addi %12#4, %36#1 {handshake.bb = 2 : ui32, handshake.name = "addi1"} : <i32>
    %48 = select %45[%46, %47] {handshake.bb = 2 : ui32, handshake.name = "select1"} : <i1>, <i32>
    %49:2 = fork [2] %48 {handshake.bb = 2 : ui32, handshake.name = "fork11"} : <i32>
    %50 = trunci %49#0 {handshake.bb = 2 : ui32, handshake.name = "trunci1"} : <i32> to <i8>
    %addressResult_1, %dataResult_2 = store[%25] %27 {handshake.bb = 2 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store1"} : <i8>, <i32>, <i8>, <i32>
    %51 = addi %12#3, %36#2 {handshake.bb = 2 : ui32, handshake.name = "addi2"} : <i32>
    %52:2 = fork [2] %31 {handshake.bb = 3 : ui32, handshake.name = "fork12"} : <i32>
    %53 = source {handshake.bb = 3 : ui32, handshake.name = "source3"} : <>
    %54 = constant %53 {handshake.bb = 3 : ui32, handshake.name = "constant13", value = 199 : i9} : <>, <i9>
    %55 = extsi %54 {handshake.bb = 3 : ui32, handshake.name = "extsi6"} : <i9> to <i32>
    %56 = cmpi slt, %52#1, %55 {handshake.bb = 3 : ui32, handshake.name = "cmpi3"} : <i32>
    %57:2 = fork [2] %56 {handshake.bb = 3 : ui32, handshake.name = "fork13"} : <i1>
    %trueResult, %falseResult = cond_br %57#1, %52#0 {handshake.bb = 3 : ui32, handshake.name = "cond_br5"} : <i1>, <i32>
    sink %falseResult {handshake.name = "sink4"} : <i32>
    %58:2 = fork [2] %trueResult {handshake.bb = 3 : ui32, handshake.name = "fork14"} : <i32>
    %trueResult_3, %falseResult_4 = cond_br %57#0, %30 {handshake.bb = 3 : ui32, handshake.name = "cond_br6"} : <i1>, <>
    %59:2 = fork [2] %falseResult_4 {handshake.bb = 4 : ui32, handshake.name = "fork15"} : <>
    end {handshake.bb = 4 : ui32, handshake.name = "end0"} %memEnd_0, %memEnd, %2#1 : <>, <>, <>
  }
}

