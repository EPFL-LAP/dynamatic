module {
  handshake.func @if_convert(%arg0: memref<200xi32>, %arg1: memref<200xi32>, %arg2: !handshake.control<>, %arg3: !handshake.control<>, %arg4: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["a", "b", "a_start", "b_start", "start"], resNames = ["a_end", "b_end", "end"]} {
    %0:3 = fork [3] %arg4 {handshake.bb = 0 : ui32, handshake.name = "fork0"} : <>
    %memEnd = mem_controller[%arg1 : memref<200xi32>] %arg3 (%38, %addressResult_1, %dataResult_2) %75#1 {connectedBlocks = [2 : i32], handshake.name = "mem_controller2"} :    (!handshake.channel<i32>, !handshake.channel<i8>, !handshake.channel<i32>) -> ()
    %outputs, %memEnd_0 = mem_controller[%arg0 : memref<200xi32>] %arg2 (%addressResult) %75#0 {connectedBlocks = [2 : i32], handshake.name = "mem_controller3"} :    (!handshake.channel<i8>) -> !handshake.channel<i32>
    %1 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant2", value = 1 : i2} : <>, <i2>
    %2 = extsi %1 {handshake.bb = 0 : ui32, handshake.name = "extsi7"} : <i2> to <i32>
    %3:2 = fork [2] %2 {handshake.bb = 0 : ui32, handshake.name = "fork1"} : <i32>
    %4 = mux %6#0 [%3#0, %74#0] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i32>, <i32>] to <i32>
    %5 = mux %6#1 [%3#1, %74#1] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i1>, [<i32>, <i32>] to <i32>
    %result, %index = control_merge [%0#2, %trueResult_3]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %6:2 = fork [2] %index {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <i1>
    %7 = mux %19#2 [%4, %44] {handshake.bb = 2 : ui32, handshake.name = "mux2"} : <i1>, [<i32>, <i32>] to <i32>
    %8:3 = fork [3] %7 {handshake.bb = 2 : ui32, handshake.name = "fork22"} : <i32>
    %9 = mux %19#1 [%5, %43] {handshake.bb = 2 : ui32, handshake.name = "mux3"} : <i1>, [<i32>, <i32>] to <i32>
    %10:7 = fork [7] %9 {handshake.bb = 2 : ui32, handshake.name = "fork23"} : <i32>
    %11 = mux %19#0 [%result, %45] {handshake.bb = 2 : ui32, handshake.name = "mux4"} : <i1>, [<>, <>] to <>
    %12 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %13 = constant %12 {handshake.bb = 2 : ui32, handshake.name = "constant9", value = 199 : i9} : <>, <i9>
    %14 = extsi %13 {handshake.bb = 2 : ui32, handshake.name = "extsi1"} : <i9> to <i32>
    %15 = cmpi slt, %10#1, %14 {handshake.bb = 2 : ui32, handshake.name = "cmpi0"} : <i32>
    %16 = cmpi eq, %10#0, %8#0 {handshake.bb = 2 : ui32, handshake.name = "cmpi1"} : <i32>
    %17 = andi %15, %16 {handshake.bb = 2 : ui32, handshake.name = "andi0"} : <i1>
    %18 = init %21#4 {handshake.bb = 2 : ui32, handshake.name = "init0", initToken = 0 : ui1} : <i1>
    %19:3 = fork [3] %18 {handshake.bb = 2 : ui32, handshake.name = "fork24"} : <i1>
    %20 = spec_v2_repeating_init %22 {handshake.bb = 2 : ui32, handshake.name = "spec_v2_repeating_init0", initToken = 1 : ui1} : <i1>
    %21:5 = fork [5] %20 {handshake.bb = 2 : ui32, handshake.name = "fork25"} : <i1>
    %22 = spec_v2_repeating_init %27 {handshake.bb = 2 : ui32, handshake.name = "spec_v2_repeating_init1", initToken = 1 : ui1} : <i1>
    %23 = andi %30, %26#0 {handshake.bb = 2 : ui32, handshake.name = "andi1"} : <i1>
    %24:2 = fork [2] %23 {handshake.bb = 2 : ui32, handshake.name = "fork26"} : <i1>
    %25 = spec_v2_resolver %29#5, %21#0 {handshake.bb = 2 : ui32, handshake.name = "spec_v2_resolver0"} : <i1>
    %26:10 = fork [10] %25 {handshake.bb = 2 : ui32, handshake.name = "fork27"} : <i1>
    %27 = spec_v2_repeating_init %28 {handshake.bb = 2 : ui32, handshake.name = "spec_v2_repeating_init2", initToken = 1 : ui1} : <i1>
    %28 = passer %29#6[%26#9] {handshake.bb = 2 : ui32, handshake.name = "passer0"} : <i1>, <i1>
    %29:7 = fork [7] %17 {handshake.bb = 2 : ui32, handshake.name = "fork28"} : <i1>
    %30 = not %29#0 {handshake.bb = 2 : ui32, handshake.name = "not0"} : <i1>
    %31 = passer %29#1[%26#4] {handshake.bb = 2 : ui32, handshake.name = "passer1"} : <i1>, <i1>
    sink %31 {handshake.name = "sink2"} : <i1>
    %32 = passer %29#2[%26#3] {handshake.bb = 2 : ui32, handshake.name = "passer2"} : <i1>, <i1>
    %33 = passer %29#3[%26#2] {handshake.bb = 2 : ui32, handshake.name = "passer3"} : <i1>, <i1>
    %34 = passer %29#4[%26#1] {handshake.bb = 2 : ui32, handshake.name = "passer4"} : <i1>, <i1>
    %35 = passer %41[%33] {handshake.bb = 2 : ui32, handshake.name = "passer5"} : <i8>, <i1>
    %36 = passer %8#1[%26#6] {handshake.bb = 2 : ui32, handshake.name = "passer6"} : <i32>, <i1>
    sink %36 {handshake.name = "sink3"} : <i32>
    %37 = passer %40[%32] {handshake.bb = 2 : ui32, handshake.name = "passer7"} : <i32>, <i1>
    %38 = passer %42[%34] {handshake.bb = 2 : ui32, handshake.name = "passer8"} : <i32>, <i1>
    %39:3 = fork [3] %11 {handshake.bb = 2 : ui32, handshake.name = "fork21"} : <>
    %40 = passer %52#0[%26#7] {handshake.bb = 2 : ui32, handshake.name = "passer9"} : <i32>, <i1>
    %41 = passer %66[%26#5] {handshake.bb = 2 : ui32, handshake.name = "passer10"} : <i8>, <i1>
    %42 = passer %53[%26#8] {handshake.bb = 2 : ui32, handshake.name = "passer11"} : <i32>, <i1>
    %43 = passer %67[%21#2] {handshake.bb = 2 : ui32, handshake.name = "passer12"} : <i32>, <i1>
    %44 = passer %65#1[%21#1] {handshake.bb = 2 : ui32, handshake.name = "passer13"} : <i32>, <i1>
    %45 = passer %39#2[%21#3] {handshake.bb = 2 : ui32, handshake.name = "passer14"} : <>, <i1>
    %46 = passer %39#0[%24#1] {handshake.bb = 2 : ui32, handshake.name = "passer15"} : <>, <i1>
    %47 = passer %8#2[%24#0] {handshake.bb = 2 : ui32, handshake.name = "passer16"} : <i32>, <i1>
    %48 = trunci %10#2 {handshake.bb = 2 : ui32, handshake.name = "trunci0"} : <i32> to <i8>
    %49 = constant %39#1 {handshake.bb = 2 : ui32, handshake.name = "constant10", value = 1 : i2} : <>, <i2>
    %50:2 = fork [2] %49 {handshake.bb = 2 : ui32, handshake.name = "fork9"} : <i2>
    %51 = extsi %50#0 {handshake.bb = 2 : ui32, handshake.name = "extsi2"} : <i2> to <i32>
    %52:3 = fork [3] %51 {handshake.bb = 2 : ui32, handshake.name = "fork10"} : <i32>
    %53 = extsi %50#1 {handshake.bb = 2 : ui32, handshake.name = "extsi3"} : <i2> to <i32>
    %54 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %55 = constant %54 {handshake.bb = 2 : ui32, handshake.name = "constant11", value = 10000 : i15} : <>, <i15>
    %56 = extsi %55 {handshake.bb = 2 : ui32, handshake.name = "extsi4"} : <i15> to <i32>
    %57 = source {handshake.bb = 2 : ui32, handshake.name = "source2"} : <>
    %58 = constant %57 {handshake.bb = 2 : ui32, handshake.name = "constant12", value = 2 : i3} : <>, <i3>
    %59 = extsi %58 {handshake.bb = 2 : ui32, handshake.name = "extsi5"} : <i3> to <i32>
    %addressResult, %dataResult = load[%48] %outputs {handshake.bb = 2 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i8>, <i32>, <i8>, <i32>
    %60 = muli %10#6, %dataResult {handshake.bb = 2 : ui32, handshake.name = "muli0"} : <i32>
    %61 = cmpi slt, %60, %56 {handshake.bb = 2 : ui32, handshake.name = "cmpi2"} : <i32>
    %62 = addi %10#5, %59 {handshake.bb = 2 : ui32, handshake.name = "addi0"} : <i32>
    %63 = addi %10#4, %52#1 {handshake.bb = 2 : ui32, handshake.name = "addi1"} : <i32>
    %64 = select %61[%62, %63] {handshake.bb = 2 : ui32, handshake.name = "select1"} : <i1>, <i32>
    %65:2 = fork [2] %64 {handshake.bb = 2 : ui32, handshake.name = "fork11"} : <i32>
    %66 = trunci %65#0 {handshake.bb = 2 : ui32, handshake.name = "trunci1"} : <i32> to <i8>
    %addressResult_1, %dataResult_2 = store[%35] %37 {handshake.bb = 2 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store1"} : <i8>, <i32>, <i8>, <i32>
    %67 = addi %10#3, %52#2 {handshake.bb = 2 : ui32, handshake.name = "addi2"} : <i32>
    %68:2 = fork [2] %47 {handshake.bb = 3 : ui32, handshake.name = "fork12"} : <i32>
    %69 = source {handshake.bb = 3 : ui32, handshake.name = "source3"} : <>
    %70 = constant %69 {handshake.bb = 3 : ui32, handshake.name = "constant13", value = 199 : i9} : <>, <i9>
    %71 = extsi %70 {handshake.bb = 3 : ui32, handshake.name = "extsi6"} : <i9> to <i32>
    %72 = cmpi slt, %68#1, %71 {handshake.bb = 3 : ui32, handshake.name = "cmpi3"} : <i32>
    %73:2 = fork [2] %72 {handshake.bb = 3 : ui32, handshake.name = "fork13"} : <i1>
    %trueResult, %falseResult = cond_br %73#1, %68#0 {handshake.bb = 3 : ui32, handshake.name = "cond_br5"} : <i1>, <i32>
    sink %falseResult {handshake.name = "sink4"} : <i32>
    %74:2 = fork [2] %trueResult {handshake.bb = 3 : ui32, handshake.name = "fork14"} : <i32>
    %trueResult_3, %falseResult_4 = cond_br %73#0, %46 {handshake.bb = 3 : ui32, handshake.name = "cond_br6"} : <i1>, <>
    %75:2 = fork [2] %falseResult_4 {handshake.bb = 4 : ui32, handshake.name = "fork15"} : <>
    end {handshake.bb = 4 : ui32, handshake.name = "end0"} %memEnd_0, %memEnd, %0#1 : <>, <>, <>
  }
}

