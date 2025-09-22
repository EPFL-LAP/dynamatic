module {
  handshake.func @collision_donut(%arg0: memref<1000xi32>, %arg1: memref<1000xi32>, %arg2: !handshake.control<>, %arg3: !handshake.control<>, %arg4: !handshake.control<>, ...) -> (!handshake.channel<i32>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["x", "y", "x_start", "y_start", "start"], resNames = ["out0", "x_end", "y_end", "end"]} {
    %0:3 = fork [3] %arg4 {handshake.bb = 0 : ui32, handshake.name = "fork0"} : <>
    %outputs, %memEnd = mem_controller[%arg1 : memref<1000xi32>] %arg3 (%addressResult_2) %94#1 {connectedBlocks = [1 : i32], handshake.name = "mem_controller2"} :    (!handshake.channel<i10>) -> !handshake.channel<i32>
    %outputs_0, %memEnd_1 = mem_controller[%arg0 : memref<1000xi32>] %arg2 (%addressResult) %94#0 {connectedBlocks = [1 : i32], handshake.name = "mem_controller3"} :    (!handshake.channel<i10>) -> !handshake.channel<i32>
    %1 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant0", value = false} : <>, <i1>
    %2 = extsi %1 {handshake.bb = 0 : ui32, handshake.name = "extsi7"} : <i1> to <i11>
    %3 = mux %68#1 [%2, %80] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i11>, <i11>] to <i11>
    %4 = trunci %79#0 {handshake.bb = 1 : ui32, handshake.name = "trunci0"} : <i11> to <i10>
    %5 = trunci %79#1 {handshake.bb = 1 : ui32, handshake.name = "trunci1"} : <i11> to <i10>
    %6 = mux %68#0 [%0#2, %82] {handshake.bb = 1 : ui32, handshake.name = "mux7"} : <i1>, [<>, <>] to <>
    %7 = source {handshake.bb = 1 : ui32, handshake.name = "source0"} : <>
    %8 = constant %7 {handshake.bb = 1 : ui32, handshake.name = "constant1", value = 4 : i4} : <>, <i4>
    %9 = extsi %8 {handshake.bb = 1 : ui32, handshake.name = "extsi1"} : <i4> to <i32>
    %10 = constant %83#0 {handshake.bb = 1 : ui32, handshake.name = "constant8", value = -1 : i32} : <>, <i32>
    %11:2 = fork [2] %dataResult {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <i32>
    %addressResult, %dataResult = load[%5] %outputs_0 {handshake.bb = 1 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i10>, <i32>, <i10>, <i32>
    %12:2 = fork [2] %dataResult_3 {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <i32>
    %addressResult_2, %dataResult_3 = load[%4] %outputs {handshake.bb = 1 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i10>, <i32>, <i10>, <i32>
    %13 = muli %11#0, %11#1 {handshake.bb = 1 : ui32, handshake.name = "muli0"} : <i32>
    %14 = muli %12#0, %12#1 {handshake.bb = 1 : ui32, handshake.name = "muli1"} : <i32>
    %15:2 = fork [2] %16 {handshake.bb = 1 : ui32, handshake.name = "fork6"} : <i32>
    %16 = addi %13, %14 {handshake.bb = 1 : ui32, handshake.name = "addi0"} : <i32>
    %17 = cmpi ult, %15#1, %9 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i32>
    %18:2 = fork [2] %17 {handshake.bb = 1 : ui32, handshake.name = "fork7"} : <i1>
    %19:2 = fork [2] %20 {handshake.bb = 1 : ui32, handshake.name = "fork8"} : <i1>
    %20 = not %18#0 {handshake.bb = 1 : ui32, handshake.name = "not0"} : <i1>
    %21 = andi %77, %34#0 {handshake.bb = 1 : ui32, handshake.name = "andi1"} : <i1>
    %22 = andi %31#1, %19#0 {handshake.bb = 1 : ui32, handshake.name = "andi2"} : <i1>
    %23 = passer %24[%76#2] {handshake.bb = 1 : ui32, handshake.name = "passer16"} : <i12>, <i1>
    %24 = extsi %79#4 {handshake.bb = 1 : ui32, handshake.name = "extsi8"} : <i11> to <i12>
    %25 = passer %10[%76#1] {handshake.bb = 1 : ui32, handshake.name = "passer2"} : <i32>, <i1>
    %26 = passer %83#6[%76#0] {handshake.bb = 1 : ui32, handshake.name = "passer3"} : <>, <i1>
    %27 = source {handshake.bb = 1 : ui32, handshake.name = "source1"} : <>
    %28 = constant %27 {handshake.bb = 1 : ui32, handshake.name = "constant2", value = 19000 : i16} : <>, <i16>
    %29 = extsi %28 {handshake.bb = 1 : ui32, handshake.name = "extsi2"} : <i16> to <i32>
    %30 = constant %83#1 {handshake.bb = 1 : ui32, handshake.name = "constant10", value = -2 : i32} : <>, <i32>
    %31:2 = fork [2] %32 {handshake.bb = 1 : ui32, handshake.name = "fork9"} : <i1>
    %32 = cmpi ugt, %15#0, %29 {handshake.bb = 1 : ui32, handshake.name = "cmpi1"} : <i32>
    %33 = not %31#0 {handshake.bb = 1 : ui32, handshake.name = "not1"} : <i1>
    %34:2 = fork [2] %35 {handshake.bb = 1 : ui32, handshake.name = "fork10"} : <i1>
    %35 = andi %19#1, %33 {handshake.bb = 1 : ui32, handshake.name = "andi3"} : <i1>
    %36 = passer %37[%72#0] {handshake.bb = 1 : ui32, handshake.name = "passer17"} : <i12>, <i1>
    %37 = extsi %79#3 {handshake.bb = 1 : ui32, handshake.name = "extsi9"} : <i11> to <i12>
    %38 = passer %30[%72#1] {handshake.bb = 1 : ui32, handshake.name = "passer8"} : <i32>, <i1>
    %39 = passer %83#5[%72#2] {handshake.bb = 1 : ui32, handshake.name = "passer9"} : <>, <i1>
    %40 = extsi %79#2 {handshake.bb = 1 : ui32, handshake.name = "extsi10"} : <i11> to <i12>
    %41 = constant %83#2 {handshake.bb = 1 : ui32, handshake.name = "constant3", value = false} : <>, <i1>
    %42 = source {handshake.bb = 1 : ui32, handshake.name = "source2"} : <>
    %43 = constant %42 {handshake.bb = 1 : ui32, handshake.name = "constant4", value = 1 : i2} : <>, <i2>
    %44 = extsi %43 {handshake.bb = 1 : ui32, handshake.name = "extsi11"} : <i2> to <i12>
    %45 = source {handshake.bb = 1 : ui32, handshake.name = "source3"} : <>
    %46 = constant %45 {handshake.bb = 1 : ui32, handshake.name = "constant5", value = 1000 : i11} : <>, <i11>
    %47 = extsi %46 {handshake.bb = 1 : ui32, handshake.name = "extsi12"} : <i11> to <i12>
    %48:3 = fork [3] %49 {handshake.bb = 1 : ui32, handshake.name = "fork11"} : <i12>
    %49 = addi %40, %44 {handshake.bb = 1 : ui32, handshake.name = "addi1"} : <i12>
    %50:2 = fork [2] %51 {handshake.bb = 1 : ui32, handshake.name = "fork28"} : <i1>
    %51 = cmpi ult, %48#0, %47 {handshake.bb = 1 : ui32, handshake.name = "cmpi2"} : <i12>
    %52 = passer %53[%70#3] {handshake.bb = 1 : ui32, handshake.name = "passer18"} : <i1>, <i1>
    %53 = andi %34#1, %50#0 {handshake.bb = 1 : ui32, handshake.name = "andi4"} : <i1>
    %54 = spec_v2_repeating_init %52 {handshake.bb = 1 : ui32, handshake.name = "spec_v2_repeating_init0", initToken = 1 : ui1} : <i1>
    %55:2 = fork [2] %54 {handshake.bb = 1 : ui32, handshake.name = "fork29"} : <i1>
    %56 = spec_v2_repeating_init %55#0 {handshake.bb = 1 : ui32, handshake.name = "spec_v2_repeating_init1", initToken = 1 : ui1} : <i1>
    %57:2 = fork [2] %56 {handshake.bb = 1 : ui32, handshake.name = "fork30"} : <i1>
    %58 = andi %55#1, %57#0 {handshake.bb = 1 : ui32, handshake.name = "andi5", specv2_tmp_and = true} : <i1>
    %59 = spec_v2_repeating_init %57#1 {handshake.bb = 1 : ui32, handshake.name = "spec_v2_repeating_init2", initToken = 1 : ui1} : <i1>
    %60:2 = fork [2] %59 {handshake.bb = 1 : ui32, handshake.name = "fork31"} : <i1>
    %61 = andi %58, %60#0 {handshake.bb = 1 : ui32, handshake.name = "andi6", specv2_tmp_and = true} : <i1>
    %62 = spec_v2_repeating_init %60#1 {handshake.bb = 1 : ui32, handshake.name = "spec_v2_repeating_init3", initToken = 1 : ui1} : <i1>
    %63:2 = fork [2] %62 {handshake.bb = 1 : ui32, handshake.name = "fork32"} : <i1>
    %64 = andi %61, %63#0 {handshake.bb = 1 : ui32, handshake.name = "andi7", specv2_tmp_and = true} : <i1>
    %65 = spec_v2_repeating_init %63#1 {handshake.bb = 1 : ui32, handshake.name = "spec_v2_repeating_init4", initToken = 1 : ui1} : <i1>
    %66:4 = fork [4] %65 {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <i1>
    %67 = init %66#3 {handshake.bb = 1 : ui32, handshake.name = "init2", initToken = 0 : ui1} : <i1>
    %68:2 = fork [2] %67 {handshake.bb = 1 : ui32, handshake.name = "fork4"} : <i1>
    %69 = andi %64, %66#2 {handshake.bb = 1 : ui32, handshake.name = "andi8", specv2_tmp_and = true} : <i1>
    %70:4 = fork [4] %69 {handshake.bb = 1 : ui32, handshake.name = "fork34"} : <i1>
    %71 = andi %22, %70#0 {handshake.bb = 1 : ui32, handshake.name = "andi9"} : <i1>
    %72:3 = fork [3] %71 {handshake.bb = 1 : ui32, handshake.name = "fork35"} : <i1>
    %73 = andi %21, %70#1 {handshake.bb = 1 : ui32, handshake.name = "andi10"} : <i1>
    %74:3 = fork [3] %73 {handshake.bb = 1 : ui32, handshake.name = "fork36"} : <i1>
    %75 = andi %18#1, %70#2 {handshake.bb = 1 : ui32, handshake.name = "andi11"} : <i1>
    %76:3 = fork [3] %75 {handshake.bb = 1 : ui32, handshake.name = "fork37"} : <i1>
    %77 = not %50#1 {handshake.bb = 1 : ui32, handshake.name = "not2"} : <i1>
    %78 = passer %48#1[%74#1] {handshake.bb = 1 : ui32, handshake.name = "passer12"} : <i12>, <i1>
    %79:5 = fork [5] %3 {handshake.bb = 1 : ui32, handshake.name = "fork38"} : <i11>
    %80 = passer %81[%66#0] {handshake.bb = 1 : ui32, handshake.name = "passer19"} : <i11>, <i1>
    %81 = trunci %48#2 {handshake.bb = 1 : ui32, handshake.name = "trunci2"} : <i12> to <i11>
    %82 = passer %83#4[%66#1] {handshake.bb = 1 : ui32, handshake.name = "passer20"} : <>, <i1>
    %83:7 = fork [7] %6 {handshake.bb = 1 : ui32, handshake.name = "fork39"} : <>
    %84 = passer %83#3[%74#2] {handshake.bb = 1 : ui32, handshake.name = "passer14"} : <>, <i1>
    %85 = passer %86[%74#0] {handshake.bb = 1 : ui32, handshake.name = "passer21"} : <i32>, <i1>
    %86 = extsi %41 {handshake.bb = 1 : ui32, handshake.name = "extsi13"} : <i1> to <i32>
    %87 = mux %89#0 [%23, %78] {handshake.bb = 2 : ui32, handshake.name = "mux1"} : <i1>, [<i12>, <i12>] to <i12>
    %88 = mux %89#1 [%25, %85] {handshake.bb = 2 : ui32, handshake.name = "mux2"} : <i1>, [<i32>, <i32>] to <i32>
    %result, %index = control_merge [%26, %84]  {handshake.bb = 2 : ui32, handshake.name = "control_merge3"} : [<>, <>] to <>, <i1>
    %89:2 = fork [2] %index {handshake.bb = 2 : ui32, handshake.name = "fork12"} : <i1>
    %90 = mux %93#0 [%36, %87] {handshake.bb = 3 : ui32, handshake.name = "mux3"} : <i1>, [<i12>, <i12>] to <i12>
    %91 = extsi %90 {handshake.bb = 3 : ui32, handshake.name = "extsi14"} : <i12> to <i32>
    %92 = mux %93#1 [%38, %88] {handshake.bb = 3 : ui32, handshake.name = "mux4"} : <i1>, [<i32>, <i32>] to <i32>
    %result_4, %index_5 = control_merge [%39, %result]  {handshake.bb = 3 : ui32, handshake.name = "control_merge4"} : [<>, <>] to <>, <i1>
    %93:2 = fork [2] %index_5 {handshake.bb = 3 : ui32, handshake.name = "fork13"} : <i1>
    %94:2 = fork [2] %result_4 {handshake.bb = 3 : ui32, handshake.name = "fork14"} : <>
    %95 = source {handshake.bb = 3 : ui32, handshake.name = "source4"} : <>
    %96 = constant %95 {handshake.bb = 3 : ui32, handshake.name = "constant15", value = 1 : i2} : <>, <i2>
    %97 = extsi %96 {handshake.bb = 3 : ui32, handshake.name = "extsi6"} : <i2> to <i32>
    %98 = shli %91, %97 {handshake.bb = 3 : ui32, handshake.name = "shli0"} : <i32>
    %99 = andi %98, %92 {handshake.bb = 3 : ui32, handshake.name = "andi0"} : <i32>
    end {handshake.bb = 3 : ui32, handshake.name = "end0"} %99, %memEnd_1, %memEnd, %0#1 : <i32>, <>, <>, <>
  }
}

