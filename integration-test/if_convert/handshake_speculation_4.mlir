module {
  handshake.func @if_convert(%arg0: memref<200xi32>, %arg1: memref<200xi32>, %arg2: !handshake.control<>, %arg3: !handshake.control<>, %arg4: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["a", "b", "a_start", "b_start", "start"], resNames = ["a_end", "b_end", "end"]} {
    %0:3 = fork [3] %arg4 {handshake.bb = 0 : ui32, handshake.name = "fork0"} : <>
    %memEnd = mem_controller[%arg1 : memref<200xi32>] %arg3 (%40, %addressResult_1, %dataResult_2) %76#1 {connectedBlocks = [2 : i32], handshake.name = "mem_controller2"} :    (!handshake.channel<i32>, !handshake.channel<i8>, !handshake.channel<i32>) -> ()
    %outputs, %memEnd_0 = mem_controller[%arg0 : memref<200xi32>] %arg2 (%addressResult) %76#0 {connectedBlocks = [2 : i32], handshake.name = "mem_controller3"} :    (!handshake.channel<i8>) -> !handshake.channel<i32>
    %1 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant2", value = 1 : i2} : <>, <i2>
    %2 = extsi %1 {handshake.bb = 0 : ui32, handshake.name = "extsi7"} : <i2> to <i32>
    %3:2 = fork [2] %2 {handshake.bb = 0 : ui32, handshake.name = "fork1"} : <i32>
    %4 = mux %6#0 [%3#0, %75#0] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i32>, <i32>] to <i32>
    %5 = mux %6#1 [%3#1, %75#1] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i1>, [<i32>, <i32>] to <i32>
    %result, %index = control_merge [%0#2, %trueResult_3]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %6:2 = fork [2] %index {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <i1>
    %7 = mux %19#2 [%4, %43] {handshake.bb = 2 : ui32, handshake.name = "mux2"} : <i1>, [<i32>, <i32>] to <i32>
    %8:3 = fork [3] %7 {handshake.bb = 2 : ui32, handshake.name = "fork22"} : <i32>
    %9 = mux %19#1 [%5, %45] {handshake.bb = 2 : ui32, handshake.name = "mux3"} : <i1>, [<i32>, <i32>] to <i32>
    %10:7 = fork [7] %9 {handshake.bb = 2 : ui32, handshake.name = "fork23"} : <i32>
    %11 = mux %19#0 [%result, %46] {handshake.bb = 2 : ui32, handshake.name = "mux4"} : <i1>, [<>, <>] to <>
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
    %22 = spec_v2_repeating_init %23 {handshake.bb = 2 : ui32, handshake.name = "spec_v2_repeating_init1", initToken = 1 : ui1} : <i1>
    %23 = spec_v2_repeating_init %28 {handshake.bb = 2 : ui32, handshake.name = "spec_v2_repeating_init2", initToken = 1 : ui1} : <i1>
    %24 = andi %31, %27#0 {handshake.bb = 2 : ui32, handshake.name = "andi1"} : <i1>
    %25:2 = fork [2] %24 {handshake.bb = 2 : ui32, handshake.name = "fork26"} : <i1>
    %26 = spec_v2_resolver %30#5, %21#0 {handshake.bb = 2 : ui32, handshake.name = "spec_v2_resolver0"} : <i1>
    %27:10 = fork [10] %26 {handshake.bb = 2 : ui32, handshake.name = "fork27"} : <i1>
    %28 = spec_v2_repeating_init %29 {handshake.bb = 2 : ui32, handshake.name = "spec_v2_repeating_init3", initToken = 1 : ui1} : <i1>
    %29 = passer %30#6[%27#9] {handshake.bb = 2 : ui32, handshake.name = "passer0"} : <i1>, <i1>
    %30:7 = fork [7] %17 {handshake.bb = 2 : ui32, handshake.name = "fork28"} : <i1>
    %31 = not %30#0 {handshake.bb = 2 : ui32, handshake.name = "not0"} : <i1>
    %32 = passer %67[%27#7] {handshake.bb = 2 : ui32, handshake.name = "passer1"} : <i8>, <i1>
    %33 = passer %30#1[%27#1] {handshake.bb = 2 : ui32, handshake.name = "passer2"} : <i1>, <i1>
    sink %33 {handshake.name = "sink2"} : <i1>
    %34 = passer %30#2[%27#2] {handshake.bb = 2 : ui32, handshake.name = "passer3"} : <i1>, <i1>
    %35 = passer %30#3[%27#3] {handshake.bb = 2 : ui32, handshake.name = "passer4"} : <i1>, <i1>
    %36 = passer %30#4[%27#4] {handshake.bb = 2 : ui32, handshake.name = "passer5"} : <i1>, <i1>
    %37 = passer %42[%35] {handshake.bb = 2 : ui32, handshake.name = "passer6"} : <i32>, <i1>
    %38 = passer %32[%36] {handshake.bb = 2 : ui32, handshake.name = "passer7"} : <i8>, <i1>
    %39 = passer %8#1[%27#5] {handshake.bb = 2 : ui32, handshake.name = "passer8"} : <i32>, <i1>
    sink %39 {handshake.name = "sink3"} : <i32>
    %40 = passer %44[%34] {handshake.bb = 2 : ui32, handshake.name = "passer9"} : <i32>, <i1>
    %41:3 = fork [3] %11 {handshake.bb = 2 : ui32, handshake.name = "fork21"} : <>
    %42 = passer %53#0[%27#6] {handshake.bb = 2 : ui32, handshake.name = "passer10"} : <i32>, <i1>
    %43 = passer %66#1[%21#1] {handshake.bb = 2 : ui32, handshake.name = "passer11"} : <i32>, <i1>
    %44 = passer %54[%27#8] {handshake.bb = 2 : ui32, handshake.name = "passer12"} : <i32>, <i1>
    %45 = passer %68[%21#2] {handshake.bb = 2 : ui32, handshake.name = "passer13"} : <i32>, <i1>
    %46 = passer %41#2[%21#3] {handshake.bb = 2 : ui32, handshake.name = "passer14"} : <>, <i1>
    %47 = passer %41#0[%25#1] {handshake.bb = 2 : ui32, handshake.name = "passer15"} : <>, <i1>
    %48 = passer %8#2[%25#0] {handshake.bb = 2 : ui32, handshake.name = "passer16"} : <i32>, <i1>
    %49 = trunci %10#2 {handshake.bb = 2 : ui32, handshake.name = "trunci0"} : <i32> to <i8>
    %50 = constant %41#1 {handshake.bb = 2 : ui32, handshake.name = "constant10", value = 1 : i2} : <>, <i2>
    %51:2 = fork [2] %50 {handshake.bb = 2 : ui32, handshake.name = "fork9"} : <i2>
    %52 = extsi %51#0 {handshake.bb = 2 : ui32, handshake.name = "extsi2"} : <i2> to <i32>
    %53:3 = fork [3] %52 {handshake.bb = 2 : ui32, handshake.name = "fork10"} : <i32>
    %54 = extsi %51#1 {handshake.bb = 2 : ui32, handshake.name = "extsi3"} : <i2> to <i32>
    %55 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %56 = constant %55 {handshake.bb = 2 : ui32, handshake.name = "constant11", value = 10000 : i15} : <>, <i15>
    %57 = extsi %56 {handshake.bb = 2 : ui32, handshake.name = "extsi4"} : <i15> to <i32>
    %58 = source {handshake.bb = 2 : ui32, handshake.name = "source2"} : <>
    %59 = constant %58 {handshake.bb = 2 : ui32, handshake.name = "constant12", value = 2 : i3} : <>, <i3>
    %60 = extsi %59 {handshake.bb = 2 : ui32, handshake.name = "extsi5"} : <i3> to <i32>
    %addressResult, %dataResult = load[%49] %outputs {handshake.bb = 2 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i8>, <i32>, <i8>, <i32>
    %61 = muli %10#6, %dataResult {handshake.bb = 2 : ui32, handshake.name = "muli0"} : <i32>
    %62 = cmpi slt, %61, %57 {handshake.bb = 2 : ui32, handshake.name = "cmpi2"} : <i32>
    %63 = addi %10#5, %60 {handshake.bb = 2 : ui32, handshake.name = "addi0"} : <i32>
    %64 = addi %10#4, %53#1 {handshake.bb = 2 : ui32, handshake.name = "addi1"} : <i32>
    %65 = select %62[%63, %64] {handshake.bb = 2 : ui32, handshake.name = "select1"} : <i1>, <i32>
    %66:2 = fork [2] %65 {handshake.bb = 2 : ui32, handshake.name = "fork11"} : <i32>
    %67 = trunci %66#0 {handshake.bb = 2 : ui32, handshake.name = "trunci1"} : <i32> to <i8>
    %addressResult_1, %dataResult_2 = store[%38] %37 {handshake.bb = 2 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store1"} : <i8>, <i32>, <i8>, <i32>
    %68 = addi %10#3, %53#2 {handshake.bb = 2 : ui32, handshake.name = "addi2"} : <i32>
    %69:2 = fork [2] %48 {handshake.bb = 3 : ui32, handshake.name = "fork12"} : <i32>
    %70 = source {handshake.bb = 3 : ui32, handshake.name = "source3"} : <>
    %71 = constant %70 {handshake.bb = 3 : ui32, handshake.name = "constant13", value = 199 : i9} : <>, <i9>
    %72 = extsi %71 {handshake.bb = 3 : ui32, handshake.name = "extsi6"} : <i9> to <i32>
    %73 = cmpi slt, %69#1, %72 {handshake.bb = 3 : ui32, handshake.name = "cmpi3"} : <i32>
    %74:2 = fork [2] %73 {handshake.bb = 3 : ui32, handshake.name = "fork13"} : <i1>
    %trueResult, %falseResult = cond_br %74#1, %69#0 {handshake.bb = 3 : ui32, handshake.name = "cond_br5"} : <i1>, <i32>
    sink %falseResult {handshake.name = "sink4"} : <i32>
    %75:2 = fork [2] %trueResult {handshake.bb = 3 : ui32, handshake.name = "fork14"} : <i32>
    %trueResult_3, %falseResult_4 = cond_br %74#0, %47 {handshake.bb = 3 : ui32, handshake.name = "cond_br6"} : <i1>, <>
    %76:2 = fork [2] %falseResult_4 {handshake.bb = 4 : ui32, handshake.name = "fork15"} : <>
    end {handshake.bb = 4 : ui32, handshake.name = "end0"} %memEnd_0, %memEnd, %0#1 : <>, <>, <>
  }
}

