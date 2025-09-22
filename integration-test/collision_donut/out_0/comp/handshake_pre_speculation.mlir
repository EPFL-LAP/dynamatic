module {
  handshake.func @collision_donut(%arg0: memref<1000xi32>, %arg1: memref<1000xi32>, %arg2: !handshake.control<>, %arg3: !handshake.control<>, %arg4: !handshake.control<>, ...) -> (!handshake.channel<i32>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["x", "y", "x_start", "y_start", "start"], resNames = ["out0", "x_end", "y_end", "end"]} {
    %0 = source {handshake.bb = 2 : ui32, handshake.name = "source5"} : <>
    %1 = constant %0 {handshake.bb = 2 : ui32, handshake.name = "constant6", value = false} : <>, <i1>
    %2 = mux %31#0 [%1, %48#0] {handshake.bb = 2 : ui32, handshake.name = "mux5", specv2_loop_cond_mux = true} : <i1>, [<i1>, <i1>] to <i1>
    %3 = source {handshake.bb = 3 : ui32, handshake.name = "source6"} : <>
    %4 = constant %3 {handshake.bb = 3 : ui32, handshake.name = "constant7", value = false} : <>, <i1>
    %5 = mux %2 [%4, %67#0] {handshake.bb = 3 : ui32, handshake.name = "mux6", specv2_loop_cond_mux = true} : <i1>, [<i1>, <i1>] to <i1>
    %6:2 = fork [2] %5 {handshake.bb = 3 : ui32, handshake.name = "fork15"} : <i1>
    %7:3 = fork [3] %arg4 {handshake.bb = 0 : ui32, handshake.name = "fork0"} : <>
    %outputs, %memEnd = mem_controller[%arg1 : memref<1000xi32>] %arg3 (%addressResult_2) %84#1 {connectedBlocks = [1 : i32], handshake.name = "mem_controller2"} :    (!handshake.channel<i10>) -> !handshake.channel<i32>
    %outputs_0, %memEnd_1 = mem_controller[%arg0 : memref<1000xi32>] %arg2 (%addressResult) %84#0 {connectedBlocks = [1 : i32], handshake.name = "mem_controller3"} :    (!handshake.channel<i10>) -> !handshake.channel<i32>
    %8 = constant %7#0 {handshake.bb = 0 : ui32, handshake.name = "constant0", value = false} : <>, <i1>
    %9 = extsi %8 {handshake.bb = 0 : ui32, handshake.name = "extsi7"} : <i1> to <i11>
    %10 = init %6#1 {handshake.bb = 1 : ui32, handshake.name = "init0", initToken = 0 : ui1} : <i1>
    %11 = init %6#0 {handshake.bb = 1 : ui32, handshake.name = "init1", initToken = 0 : ui1} : <i1>
    %12 = mux %10 [%9, %72] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i11>, <i11>] to <i11>
    %13:4 = fork [4] %12 {handshake.bb = 1 : ui32, handshake.name = "fork16"} : <i11>
    %14 = trunci %13#0 {handshake.bb = 1 : ui32, handshake.name = "trunci0"} : <i11> to <i10>
    %15 = trunci %13#1 {handshake.bb = 1 : ui32, handshake.name = "trunci1"} : <i11> to <i10>
    %16 = mux %11 [%7#2, %73] {handshake.bb = 1 : ui32, handshake.name = "mux7"} : <i1>, [<>, <>] to <>
    %17:3 = fork [3] %16 {handshake.bb = 1 : ui32, handshake.name = "fork17"} : <>
    %18 = source {handshake.bb = 1 : ui32, handshake.name = "source0"} : <>
    %19 = constant %18 {handshake.bb = 1 : ui32, handshake.name = "constant1", value = 4 : i4} : <>, <i4>
    %20 = extsi %19 {handshake.bb = 1 : ui32, handshake.name = "extsi1"} : <i4> to <i32>
    %21 = constant %17#0 {handshake.bb = 1 : ui32, handshake.name = "constant8", value = -1 : i32} : <>, <i32>
    %addressResult, %dataResult = load[%15] %outputs_0 {handshake.bb = 1 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i10>, <i32>, <i10>, <i32>
    %22:2 = fork [2] %dataResult {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <i32>
    %addressResult_2, %dataResult_3 = load[%14] %outputs {handshake.bb = 1 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i10>, <i32>, <i10>, <i32>
    %23:2 = fork [2] %dataResult_3 {handshake.bb = 1 : ui32, handshake.name = "fork4"} : <i32>
    %24 = muli %22#0, %22#1 {handshake.bb = 1 : ui32, handshake.name = "muli0"} : <i32>
    %25 = muli %23#0, %23#1 {handshake.bb = 1 : ui32, handshake.name = "muli1"} : <i32>
    %26 = addi %24, %25 {handshake.bb = 1 : ui32, handshake.name = "addi0"} : <i32>
    %27:2 = fork [2] %26 {handshake.bb = 1 : ui32, handshake.name = "fork5"} : <i32>
    %28 = cmpi ult, %27#1, %20 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i32>
    %29:4 = fork [4] %28 {handshake.bb = 1 : ui32, handshake.name = "fork18"} : <i1>
    %30 = not %29#3 {handshake.bb = 1 : ui32, handshake.name = "not0"} : <i1>
    %31:4 = fork [4] %30 {handshake.bb = 1 : ui32, handshake.name = "fork19"} : <i1>
    %32 = passer %13#3[%29#2] {handshake.bb = 1 : ui32, handshake.name = "passer0"} : <i11>, <i1>
    %33 = passer %13#2[%31#3] {handshake.bb = 1 : ui32, handshake.name = "passer1"} : <i11>, <i1>
    %34:2 = fork [2] %33 {handshake.bb = 1 : ui32, handshake.name = "fork20"} : <i11>
    %35 = extsi %32 {handshake.bb = 1 : ui32, handshake.name = "extsi8"} : <i11> to <i12>
    %36 = passer %21[%29#1] {handshake.bb = 1 : ui32, handshake.name = "passer2"} : <i32>, <i1>
    %37 = passer %17#2[%29#0] {handshake.bb = 1 : ui32, handshake.name = "passer3"} : <>, <i1>
    %38 = passer %17#1[%31#2] {handshake.bb = 1 : ui32, handshake.name = "passer4"} : <>, <i1>
    %39 = passer %27#0[%31#1] {handshake.bb = 1 : ui32, handshake.name = "passer5"} : <i32>, <i1>
    %40:3 = fork [3] %38 {handshake.bb = 2 : ui32, handshake.name = "fork21"} : <>
    %41 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %42 = constant %41 {handshake.bb = 2 : ui32, handshake.name = "constant2", value = 19000 : i16} : <>, <i16>
    %43 = extsi %42 {handshake.bb = 2 : ui32, handshake.name = "extsi2"} : <i16> to <i32>
    %44 = constant %40#0 {handshake.bb = 2 : ui32, handshake.name = "constant10", value = -2 : i32} : <>, <i32>
    %45 = cmpi ugt, %39, %43 {handshake.bb = 2 : ui32, handshake.name = "cmpi1"} : <i32>
    %46:4 = fork [4] %45 {handshake.bb = 2 : ui32, handshake.name = "fork22"} : <i1>
    %47 = not %46#3 {handshake.bb = 2 : ui32, handshake.name = "not1"} : <i1>
    %48:3 = fork [3] %47 {handshake.bb = 2 : ui32, handshake.name = "fork23"} : <i1>
    %49 = passer %34#1[%46#2] {handshake.bb = 2 : ui32, handshake.name = "passer6"} : <i11>, <i1>
    %50 = passer %34#0[%48#2] {handshake.bb = 2 : ui32, handshake.name = "passer7"} : <i11>, <i1>
    %51 = extsi %49 {handshake.bb = 2 : ui32, handshake.name = "extsi9"} : <i11> to <i12>
    %52 = passer %44[%46#1] {handshake.bb = 2 : ui32, handshake.name = "passer8"} : <i32>, <i1>
    %53 = passer %40#2[%46#0] {handshake.bb = 2 : ui32, handshake.name = "passer9"} : <>, <i1>
    %54 = passer %40#1[%48#1] {handshake.bb = 2 : ui32, handshake.name = "passer10"} : <>, <i1>
    %55 = extsi %50 {handshake.bb = 3 : ui32, handshake.name = "extsi10"} : <i11> to <i12>
    %56:3 = fork [3] %54 {handshake.bb = 3 : ui32, handshake.name = "fork24"} : <>
    %57 = constant %56#0 {handshake.bb = 3 : ui32, handshake.name = "constant3", value = false} : <>, <i1>
    %58 = source {handshake.bb = 3 : ui32, handshake.name = "source2"} : <>
    %59 = constant %58 {handshake.bb = 3 : ui32, handshake.name = "constant4", value = 1 : i2} : <>, <i2>
    %60 = extsi %59 {handshake.bb = 3 : ui32, handshake.name = "extsi11"} : <i2> to <i12>
    %61 = source {handshake.bb = 3 : ui32, handshake.name = "source3"} : <>
    %62 = constant %61 {handshake.bb = 3 : ui32, handshake.name = "constant5", value = 1000 : i11} : <>, <i11>
    %63 = extsi %62 {handshake.bb = 3 : ui32, handshake.name = "extsi12"} : <i11> to <i12>
    %64 = addi %55, %60 {handshake.bb = 3 : ui32, handshake.name = "addi1"} : <i12>
    %65:3 = fork [3] %64 {handshake.bb = 3 : ui32, handshake.name = "fork25"} : <i12>
    %66 = cmpi ult, %65#0, %63 {handshake.bb = 3 : ui32, handshake.name = "cmpi2"} : <i12>
    %67:4 = fork [4] %66 {handshake.bb = 3 : ui32, handshake.name = "fork26"} : <i1>
    %68 = not %67#3 {handshake.bb = 3 : ui32, handshake.name = "not2"} : <i1>
    %69:3 = fork [3] %68 {handshake.bb = 3 : ui32, handshake.name = "fork27"} : <i1>
    %70 = passer %65#2[%67#2] {handshake.bb = 3 : ui32, handshake.name = "passer11"} : <i12>, <i1>
    %71 = passer %65#1[%69#2] {handshake.bb = 3 : ui32, handshake.name = "passer12"} : <i12>, <i1>
    %72 = trunci %70 {handshake.bb = 3 : ui32, handshake.name = "trunci2"} : <i12> to <i11>
    %73 = passer %56#2[%67#1] {handshake.bb = 3 : ui32, handshake.name = "passer13"} : <>, <i1>
    %74 = passer %56#1[%69#1] {handshake.bb = 3 : ui32, handshake.name = "passer14"} : <>, <i1>
    %75 = passer %57[%69#0] {handshake.bb = 3 : ui32, handshake.name = "passer15"} : <i1>, <i1>
    %76 = extsi %75 {handshake.bb = 3 : ui32, handshake.name = "extsi13"} : <i1> to <i32>
    %77 = mux %79#0 [%35, %71] {handshake.bb = 4 : ui32, handshake.name = "mux1"} : <i1>, [<i12>, <i12>] to <i12>
    %78 = mux %79#1 [%36, %76] {handshake.bb = 4 : ui32, handshake.name = "mux2"} : <i1>, [<i32>, <i32>] to <i32>
    %result, %index = control_merge [%37, %74]  {handshake.bb = 4 : ui32, handshake.name = "control_merge3"} : [<>, <>] to <>, <i1>
    %79:2 = fork [2] %index {handshake.bb = 4 : ui32, handshake.name = "fork12"} : <i1>
    %80 = mux %83#0 [%51, %77] {handshake.bb = 5 : ui32, handshake.name = "mux3"} : <i1>, [<i12>, <i12>] to <i12>
    %81 = extsi %80 {handshake.bb = 5 : ui32, handshake.name = "extsi14"} : <i12> to <i32>
    %82 = mux %83#1 [%52, %78] {handshake.bb = 5 : ui32, handshake.name = "mux4"} : <i1>, [<i32>, <i32>] to <i32>
    %result_4, %index_5 = control_merge [%53, %result]  {handshake.bb = 5 : ui32, handshake.name = "control_merge4"} : [<>, <>] to <>, <i1>
    %83:2 = fork [2] %index_5 {handshake.bb = 5 : ui32, handshake.name = "fork13"} : <i1>
    %84:2 = fork [2] %result_4 {handshake.bb = 5 : ui32, handshake.name = "fork14"} : <>
    %85 = source {handshake.bb = 5 : ui32, handshake.name = "source4"} : <>
    %86 = constant %85 {handshake.bb = 5 : ui32, handshake.name = "constant15", value = 1 : i2} : <>, <i2>
    %87 = extsi %86 {handshake.bb = 5 : ui32, handshake.name = "extsi6"} : <i2> to <i32>
    %88 = shli %81, %87 {handshake.bb = 5 : ui32, handshake.name = "shli0"} : <i32>
    %89 = andi %88, %82 {handshake.bb = 5 : ui32, handshake.name = "andi0"} : <i32>
    end {handshake.bb = 5 : ui32, handshake.name = "end0"} %89, %memEnd_1, %memEnd, %7#1 : <i32>, <>, <>, <>
  }
}

