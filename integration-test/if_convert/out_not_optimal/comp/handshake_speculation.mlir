module {
  handshake.func @if_convert(%arg0: memref<200xi32>, %arg1: memref<200xi32>, %arg2: !handshake.control<>, %arg3: !handshake.control<>, %arg4: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["a", "b", "a_start", "b_start", "start"], resNames = ["a_end", "b_end", "end"]} {
    %0:3 = fork [3] %arg4 {handshake.bb = 0 : ui32, handshake.name = "fork0"} : <>
    %memEnd = mem_controller[%arg1 : memref<200xi32>] %arg3 (%40, %addressResult_1, %dataResult_2) %74#1 {connectedBlocks = [2 : i32], handshake.name = "mem_controller2"} :    (!handshake.channel<i32>, !handshake.channel<i8>, !handshake.channel<i32>) -> ()
    %outputs, %memEnd_0 = mem_controller[%arg0 : memref<200xi32>] %arg2 (%addressResult) %74#0 {connectedBlocks = [2 : i32], handshake.name = "mem_controller3"} :    (!handshake.channel<i8>) -> !handshake.channel<i32>
    %1 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant2", value = 1 : i2} : <>, <i2>
    %2 = extsi %1 {handshake.bb = 0 : ui32, handshake.name = "extsi7"} : <i2> to <i32>
    %3:2 = fork [2] %2 {handshake.bb = 0 : ui32, handshake.name = "fork1"} : <i32>
    %4 = mux %6#0 [%3#0, %73#0] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i32>, <i32>] to <i32>
    %5 = mux %6#1 [%3#1, %73#1] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i1>, [<i32>, <i32>] to <i32>
    %result, %index = control_merge [%0#2, %trueResult_3]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %6:2 = fork [2] %index {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <i1>
    %7 = mux %19#2 [%4, %33] {handshake.bb = 2 : ui32, handshake.name = "mux2"} : <i1>, [<i32>, <i32>] to <i32>
    %8:4 = fork [4] %7 {handshake.bb = 2 : ui32, handshake.name = "fork23"} : <i32>
    %9 = mux %19#1 [%5, %44] {handshake.bb = 2 : ui32, handshake.name = "mux3"} : <i1>, [<i32>, <i32>] to <i32>
    %10:5 = fork [5] %9 {handshake.bb = 2 : ui32, handshake.name = "fork24"} : <i32>
    %11 = mux %19#0 [%result, %45] {handshake.bb = 2 : ui32, handshake.name = "mux4"} : <i1>, [<>, <>] to <>
    %12 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %13 = constant %12 {handshake.bb = 2 : ui32, handshake.name = "constant9", value = 199 : i9} : <>, <i9>
    %14 = extsi %13 {handshake.bb = 2 : ui32, handshake.name = "extsi1"} : <i9> to <i32>
    %15 = cmpi slt, %10#1, %14 {handshake.bb = 2 : ui32, handshake.name = "cmpi0"} : <i32>
    %16 = cmpi eq, %10#0, %8#0 {handshake.bb = 2 : ui32, handshake.name = "cmpi1"} : <i32>
    %17 = andi %15, %16 {handshake.bb = 2 : ui32, handshake.name = "andi0"} : <i1>
    %18 = init %24#0 {handshake.bb = 2 : ui32, handshake.name = "init0", initToken = 0 : ui1} : <i1>
    %19:3 = fork [3] %18 {handshake.bb = 2 : ui32, handshake.name = "fork25"} : <i1>
    %20 = source {handshake.bb = 2 : ui32, handshake.name = "source4", specv2_ignore_buffer = true} : <>
    %21 = constant %20 {handshake.bb = 2 : ui32, handshake.name = "constant0", specv2_ignore_buffer = true, value = true} : <>, <i1>
    %22 = buffer %29, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer0", specv2_buffer_as_sink = true} : <i1>
    %23 = merge %22, %21 {handshake.bb = 2 : ui32, handshake.name = "merge0", specv2_buffer_as_source = true} : <i1>
    %24:5 = fork [5] %23 {handshake.bb = 2 : ui32, handshake.name = "fork26"} : <i1>
    %25 = andi %31, %28#0 {handshake.bb = 2 : ui32, handshake.name = "andi1"} : <i1>
    %26:3 = fork [3] %25 {handshake.bb = 2 : ui32, handshake.name = "fork27"} : <i1>
    %27 = spec_v2_resolver %30#4, %24#4 {handshake.bb = 2 : ui32, handshake.name = "spec_v2_resolver0"} : <i1>
    %28:7 = fork [7] %27 {handshake.bb = 2 : ui32, handshake.name = "fork28"} : <i1>
    %29 = passer %30#5[%26#0] {handshake.bb = 2 : ui32, handshake.name = "passer0"} : <i1>, <i1>
    %30:6 = fork [6] %17 {handshake.bb = 2 : ui32, handshake.name = "fork29"} : <i1>
    %31 = not %30#0 {handshake.bb = 2 : ui32, handshake.name = "not0"} : <i1>
    %32 = passer %8#1[%26#2] {handshake.bb = 2 : ui32, handshake.name = "passer1"} : <i32>, <i1>
    %33 = passer %64#1[%24#3] {handshake.bb = 2 : ui32, handshake.name = "passer2"} : <i32>, <i1>
    %34 = passer %65[%28#1] {handshake.bb = 2 : ui32, handshake.name = "passer3"} : <i8>, <i1>
    %35 = passer %34[%38] {handshake.bb = 2 : ui32, handshake.name = "passer4"} : <i8>, <i1>
    %36 = passer %30#1[%28#4] {handshake.bb = 2 : ui32, handshake.name = "passer5"} : <i1>, <i1>
    %37 = passer %30#2[%28#3] {handshake.bb = 2 : ui32, handshake.name = "passer6"} : <i1>, <i1>
    %38 = passer %30#3[%28#2] {handshake.bb = 2 : ui32, handshake.name = "passer7"} : <i1>, <i1>
    %39 = passer %42[%37] {handshake.bb = 2 : ui32, handshake.name = "passer8"} : <i32>, <i1>
    %40 = passer %43[%36] {handshake.bb = 2 : ui32, handshake.name = "passer9"} : <i32>, <i1>
    %41:3 = fork [3] %11 {handshake.bb = 2 : ui32, handshake.name = "fork22"} : <>
    %42 = passer %51#0[%28#5] {handshake.bb = 2 : ui32, handshake.name = "passer10"} : <i32>, <i1>
    %43 = passer %52[%28#6] {handshake.bb = 2 : ui32, handshake.name = "passer11"} : <i32>, <i1>
    %44 = passer %66[%24#2] {handshake.bb = 2 : ui32, handshake.name = "passer12"} : <i32>, <i1>
    %45 = passer %41#2[%24#1] {handshake.bb = 2 : ui32, handshake.name = "passer13"} : <>, <i1>
    %46 = passer %41#0[%26#1] {handshake.bb = 2 : ui32, handshake.name = "passer14"} : <>, <i1>
    %47 = trunci %8#2 {handshake.bb = 2 : ui32, handshake.name = "trunci0"} : <i32> to <i8>
    %48 = constant %41#1 {handshake.bb = 2 : ui32, handshake.name = "constant10", value = 1 : i2} : <>, <i2>
    %49:2 = fork [2] %48 {handshake.bb = 2 : ui32, handshake.name = "fork10"} : <i2>
    %50 = extsi %49#0 {handshake.bb = 2 : ui32, handshake.name = "extsi2"} : <i2> to <i32>
    %51:3 = fork [3] %50 {handshake.bb = 2 : ui32, handshake.name = "fork11"} : <i32>
    %52 = extsi %49#1 {handshake.bb = 2 : ui32, handshake.name = "extsi3"} : <i2> to <i32>
    %53 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %54 = constant %53 {handshake.bb = 2 : ui32, handshake.name = "constant11", value = 10000 : i15} : <>, <i15>
    %55 = extsi %54 {handshake.bb = 2 : ui32, handshake.name = "extsi4"} : <i15> to <i32>
    %56 = source {handshake.bb = 2 : ui32, handshake.name = "source2"} : <>
    %57 = constant %56 {handshake.bb = 2 : ui32, handshake.name = "constant12", value = 2 : i3} : <>, <i3>
    %58 = extsi %57 {handshake.bb = 2 : ui32, handshake.name = "extsi5"} : <i3> to <i32>
    %addressResult, %dataResult = load[%47] %outputs {handshake.bb = 2 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i8>, <i32>, <i8>, <i32>
    %59 = muli %8#3, %dataResult {handshake.bb = 2 : ui32, handshake.name = "muli0"} : <i32>
    %60 = cmpi slt, %59, %55 {handshake.bb = 2 : ui32, handshake.name = "cmpi2"} : <i32>
    %61 = addi %10#4, %58 {handshake.bb = 2 : ui32, handshake.name = "addi0"} : <i32>
    %62 = addi %10#3, %51#1 {handshake.bb = 2 : ui32, handshake.name = "addi1"} : <i32>
    %63 = select %60[%61, %62] {handshake.bb = 2 : ui32, handshake.name = "select1"} : <i1>, <i32>
    %64:2 = fork [2] %63 {handshake.bb = 2 : ui32, handshake.name = "fork12"} : <i32>
    %65 = trunci %64#0 {handshake.bb = 2 : ui32, handshake.name = "trunci1"} : <i32> to <i8>
    %addressResult_1, %dataResult_2 = store[%35] %39 {handshake.bb = 2 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store1"} : <i8>, <i32>, <i8>, <i32>
    %66 = addi %10#2, %51#2 {handshake.bb = 2 : ui32, handshake.name = "addi2"} : <i32>
    %67:2 = fork [2] %32 {handshake.bb = 3 : ui32, handshake.name = "fork13"} : <i32>
    %68 = source {handshake.bb = 3 : ui32, handshake.name = "source3"} : <>
    %69 = constant %68 {handshake.bb = 3 : ui32, handshake.name = "constant13", value = 199 : i9} : <>, <i9>
    %70 = extsi %69 {handshake.bb = 3 : ui32, handshake.name = "extsi6"} : <i9> to <i32>
    %71 = cmpi slt, %67#1, %70 {handshake.bb = 3 : ui32, handshake.name = "cmpi3"} : <i32>
    %72:2 = fork [2] %71 {handshake.bb = 3 : ui32, handshake.name = "fork14"} : <i1>
    %trueResult, %falseResult = cond_br %72#1, %67#0 {handshake.bb = 3 : ui32, handshake.name = "cond_br5"} : <i1>, <i32>
    sink %falseResult {handshake.name = "sink3"} : <i32>
    %73:2 = fork [2] %trueResult {handshake.bb = 3 : ui32, handshake.name = "fork15"} : <i32>
    %trueResult_3, %falseResult_4 = cond_br %72#0, %46 {handshake.bb = 3 : ui32, handshake.name = "cond_br6"} : <i1>, <>
    %74:2 = fork [2] %falseResult_4 {handshake.bb = 4 : ui32, handshake.name = "fork16"} : <>
    end {handshake.bb = 4 : ui32, handshake.name = "end0"} %memEnd_0, %memEnd, %0#1 : <>, <>, <>
  }
}

