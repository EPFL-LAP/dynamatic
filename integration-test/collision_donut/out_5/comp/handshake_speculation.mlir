module {
  handshake.func @collision_donut(%arg0: memref<1000xi32>, %arg1: memref<1000xi32>, %arg2: !handshake.control<>, %arg3: !handshake.control<>, %arg4: !handshake.control<>, ...) -> (!handshake.channel<i32>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["x", "y", "x_start", "y_start", "start"], resNames = ["out0", "x_end", "y_end", "end"]} {
    %0:3 = fork [3] %arg4 {handshake.bb = 0 : ui32, handshake.name = "fork0"} : <>
    %outputs, %memEnd = mem_controller[%arg1 : memref<1000xi32>] %arg3 (%addressResult_2) %94#1 {connectedBlocks = [1 : i32], handshake.name = "mem_controller2"} :    (!handshake.channel<i10>) -> !handshake.channel<i32>
    %outputs_0, %memEnd_1 = mem_controller[%arg0 : memref<1000xi32>] %arg2 (%addressResult) %94#0 {connectedBlocks = [1 : i32], handshake.name = "mem_controller3"} :    (!handshake.channel<i10>) -> !handshake.channel<i32>
    %1 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant0", value = false} : <>, <i1>
    %2 = extsi %1 {handshake.bb = 0 : ui32, handshake.name = "extsi7"} : <i1> to <i11>
    %3 = init %68#3 {handshake.bb = 1 : ui32, handshake.name = "init0", initToken = 0 : ui1} : <i1>
    %4 = init %68#4 {handshake.bb = 1 : ui32, handshake.name = "init1", initToken = 0 : ui1} : <i1>
    %5 = mux %3 [%2, %80] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i11>, <i11>] to <i11>
    %6 = trunci %79#0 {handshake.bb = 1 : ui32, handshake.name = "trunci0"} : <i11> to <i10>
    %7 = trunci %79#1 {handshake.bb = 1 : ui32, handshake.name = "trunci1"} : <i11> to <i10>
    %8 = mux %4 [%0#2, %82] {handshake.bb = 1 : ui32, handshake.name = "mux7"} : <i1>, [<>, <>] to <>
    %9 = source {handshake.bb = 1 : ui32, handshake.name = "source0"} : <>
    %10 = constant %9 {handshake.bb = 1 : ui32, handshake.name = "constant1", value = 4 : i4} : <>, <i4>
    %11 = extsi %10 {handshake.bb = 1 : ui32, handshake.name = "extsi1"} : <i4> to <i32>
    %12 = constant %83#0 {handshake.bb = 1 : ui32, handshake.name = "constant8", value = -1 : i32} : <>, <i32>
    %13:2 = fork [2] %dataResult {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <i32>
    %addressResult, %dataResult = load[%7] %outputs_0 {handshake.bb = 1 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i10>, <i32>, <i10>, <i32>
    %14:2 = fork [2] %dataResult_3 {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <i32>
    %addressResult_2, %dataResult_3 = load[%6] %outputs {handshake.bb = 1 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i10>, <i32>, <i10>, <i32>
    %15 = muli %13#0, %13#1 {handshake.bb = 1 : ui32, handshake.name = "muli0"} : <i32>
    %16 = muli %14#0, %14#1 {handshake.bb = 1 : ui32, handshake.name = "muli1"} : <i32>
    %17:2 = fork [2] %18 {handshake.bb = 1 : ui32, handshake.name = "fork6"} : <i32>
    %18 = addi %15, %16 {handshake.bb = 1 : ui32, handshake.name = "addi0"} : <i32>
    %19 = cmpi ult, %17#1, %11 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i32>
    %20:2 = fork [2] %19 {handshake.bb = 1 : ui32, handshake.name = "fork7"} : <i1>
    %21:2 = fork [2] %22 {handshake.bb = 1 : ui32, handshake.name = "fork8"} : <i1>
    %22 = not %20#0 {handshake.bb = 1 : ui32, handshake.name = "not0"} : <i1>
    %23 = andi %77, %36#0 {handshake.bb = 1 : ui32, handshake.name = "andi1"} : <i1>
    %24 = andi %33#1, %21#0 {handshake.bb = 1 : ui32, handshake.name = "andi2"} : <i1>
    %25 = passer %26[%76#2] {handshake.bb = 1 : ui32, handshake.name = "passer16"} : <i12>, <i1>
    %26 = extsi %79#4 {handshake.bb = 1 : ui32, handshake.name = "extsi8"} : <i11> to <i12>
    %27 = passer %12[%76#1] {handshake.bb = 1 : ui32, handshake.name = "passer2"} : <i32>, <i1>
    %28 = passer %83#6[%76#0] {handshake.bb = 1 : ui32, handshake.name = "passer3"} : <>, <i1>
    %29 = source {handshake.bb = 1 : ui32, handshake.name = "source1"} : <>
    %30 = constant %29 {handshake.bb = 1 : ui32, handshake.name = "constant2", value = 19000 : i16} : <>, <i16>
    %31 = extsi %30 {handshake.bb = 1 : ui32, handshake.name = "extsi2"} : <i16> to <i32>
    %32 = constant %83#1 {handshake.bb = 1 : ui32, handshake.name = "constant10", value = -2 : i32} : <>, <i32>
    %33:2 = fork [2] %34 {handshake.bb = 1 : ui32, handshake.name = "fork9"} : <i1>
    %34 = cmpi ugt, %17#0, %31 {handshake.bb = 1 : ui32, handshake.name = "cmpi1"} : <i32>
    %35 = not %33#0 {handshake.bb = 1 : ui32, handshake.name = "not1"} : <i1>
    %36:2 = fork [2] %37 {handshake.bb = 1 : ui32, handshake.name = "fork10"} : <i1>
    %37 = andi %21#1, %35 {handshake.bb = 1 : ui32, handshake.name = "andi3"} : <i1>
    %38 = passer %39[%72#0] {handshake.bb = 1 : ui32, handshake.name = "passer17"} : <i12>, <i1>
    %39 = extsi %79#3 {handshake.bb = 1 : ui32, handshake.name = "extsi9"} : <i11> to <i12>
    %40 = passer %32[%72#1] {handshake.bb = 1 : ui32, handshake.name = "passer8"} : <i32>, <i1>
    %41 = passer %83#5[%72#2] {handshake.bb = 1 : ui32, handshake.name = "passer9"} : <>, <i1>
    %42 = extsi %79#2 {handshake.bb = 1 : ui32, handshake.name = "extsi10"} : <i11> to <i12>
    %43 = constant %83#2 {handshake.bb = 1 : ui32, handshake.name = "constant3", value = false} : <>, <i1>
    %44 = source {handshake.bb = 1 : ui32, handshake.name = "source2"} : <>
    %45 = constant %44 {handshake.bb = 1 : ui32, handshake.name = "constant4", value = 1 : i2} : <>, <i2>
    %46 = extsi %45 {handshake.bb = 1 : ui32, handshake.name = "extsi11"} : <i2> to <i12>
    %47 = source {handshake.bb = 1 : ui32, handshake.name = "source3"} : <>
    %48 = constant %47 {handshake.bb = 1 : ui32, handshake.name = "constant5", value = 1000 : i11} : <>, <i11>
    %49 = extsi %48 {handshake.bb = 1 : ui32, handshake.name = "extsi12"} : <i11> to <i12>
    %50:3 = fork [3] %51 {handshake.bb = 1 : ui32, handshake.name = "fork11"} : <i12>
    %51 = addi %42, %46 {handshake.bb = 1 : ui32, handshake.name = "addi1"} : <i12>
    %52:2 = fork [2] %53 {handshake.bb = 1 : ui32, handshake.name = "fork28"} : <i1>
    %53 = cmpi ult, %50#0, %49 {handshake.bb = 1 : ui32, handshake.name = "cmpi2"} : <i12>
    %54 = passer %55[%70#3] {handshake.bb = 1 : ui32, handshake.name = "passer18"} : <i1>, <i1>
    %55 = andi %36#1, %52#0 {handshake.bb = 1 : ui32, handshake.name = "andi4"} : <i1>
    %56 = spec_v2_repeating_init %54 {handshake.bb = 1 : ui32, handshake.name = "spec_v2_repeating_init0", initToken = 1 : ui1} : <i1>
    %57:2 = fork [2] %56 {handshake.bb = 1 : ui32, handshake.name = "fork29"} : <i1>
    %58 = spec_v2_repeating_init %57#0 {handshake.bb = 1 : ui32, handshake.name = "spec_v2_repeating_init1", initToken = 1 : ui1} : <i1>
    %59:2 = fork [2] %58 {handshake.bb = 1 : ui32, handshake.name = "fork30"} : <i1>
    %60 = andi %57#1, %59#0 {handshake.bb = 1 : ui32, handshake.name = "andi5", specv2_tmp_and = true} : <i1>
    %61 = spec_v2_repeating_init %59#1 {handshake.bb = 1 : ui32, handshake.name = "spec_v2_repeating_init2", initToken = 1 : ui1} : <i1>
    %62:2 = fork [2] %61 {handshake.bb = 1 : ui32, handshake.name = "fork31"} : <i1>
    %63 = andi %60, %62#0 {handshake.bb = 1 : ui32, handshake.name = "andi6", specv2_tmp_and = true} : <i1>
    %64 = spec_v2_repeating_init %62#1 {handshake.bb = 1 : ui32, handshake.name = "spec_v2_repeating_init3", initToken = 1 : ui1} : <i1>
    %65:2 = fork [2] %64 {handshake.bb = 1 : ui32, handshake.name = "fork32"} : <i1>
    %66 = andi %63, %65#0 {handshake.bb = 1 : ui32, handshake.name = "andi7", specv2_tmp_and = true} : <i1>
    %67 = spec_v2_repeating_init %65#1 {handshake.bb = 1 : ui32, handshake.name = "spec_v2_repeating_init4", initToken = 1 : ui1} : <i1>
    %68:5 = fork [5] %67 {handshake.bb = 1 : ui32, handshake.name = "fork33"} : <i1>
    %69 = andi %66, %68#0 {handshake.bb = 1 : ui32, handshake.name = "andi8", specv2_tmp_and = true} : <i1>
    %70:4 = fork [4] %69 {handshake.bb = 1 : ui32, handshake.name = "fork34"} : <i1>
    %71 = andi %24, %70#0 {handshake.bb = 1 : ui32, handshake.name = "andi9"} : <i1>
    %72:3 = fork [3] %71 {handshake.bb = 1 : ui32, handshake.name = "fork35"} : <i1>
    %73 = andi %23, %70#1 {handshake.bb = 1 : ui32, handshake.name = "andi10"} : <i1>
    %74:3 = fork [3] %73 {handshake.bb = 1 : ui32, handshake.name = "fork36"} : <i1>
    %75 = andi %20#1, %70#2 {handshake.bb = 1 : ui32, handshake.name = "andi11"} : <i1>
    %76:3 = fork [3] %75 {handshake.bb = 1 : ui32, handshake.name = "fork37"} : <i1>
    %77 = not %52#1 {handshake.bb = 1 : ui32, handshake.name = "not2"} : <i1>
    %78 = passer %50#1[%74#1] {handshake.bb = 1 : ui32, handshake.name = "passer12"} : <i12>, <i1>
    %79:5 = fork [5] %5 {handshake.bb = 1 : ui32, handshake.name = "fork38"} : <i11>
    %80 = passer %81[%68#2] {handshake.bb = 1 : ui32, handshake.name = "passer19"} : <i11>, <i1>
    %81 = trunci %50#2 {handshake.bb = 1 : ui32, handshake.name = "trunci2"} : <i12> to <i11>
    %82 = passer %83#4[%68#1] {handshake.bb = 1 : ui32, handshake.name = "passer20"} : <>, <i1>
    %83:7 = fork [7] %8 {handshake.bb = 1 : ui32, handshake.name = "fork39"} : <>
    %84 = passer %83#3[%74#2] {handshake.bb = 1 : ui32, handshake.name = "passer14"} : <>, <i1>
    %85 = passer %86[%74#0] {handshake.bb = 1 : ui32, handshake.name = "passer21"} : <i32>, <i1>
    %86 = extsi %43 {handshake.bb = 1 : ui32, handshake.name = "extsi13"} : <i1> to <i32>
    %87 = mux %89#0 [%25, %78] {handshake.bb = 2 : ui32, handshake.name = "mux1"} : <i1>, [<i12>, <i12>] to <i12>
    %88 = mux %89#1 [%27, %85] {handshake.bb = 2 : ui32, handshake.name = "mux2"} : <i1>, [<i32>, <i32>] to <i32>
    %result, %index = control_merge [%28, %84]  {handshake.bb = 2 : ui32, handshake.name = "control_merge3"} : [<>, <>] to <>, <i1>
    %89:2 = fork [2] %index {handshake.bb = 2 : ui32, handshake.name = "fork12"} : <i1>
    %90 = mux %93#0 [%38, %87] {handshake.bb = 3 : ui32, handshake.name = "mux3"} : <i1>, [<i12>, <i12>] to <i12>
    %91 = extsi %90 {handshake.bb = 3 : ui32, handshake.name = "extsi14"} : <i12> to <i32>
    %92 = mux %93#1 [%40, %88] {handshake.bb = 3 : ui32, handshake.name = "mux4"} : <i1>, [<i32>, <i32>] to <i32>
    %result_4, %index_5 = control_merge [%41, %result]  {handshake.bb = 3 : ui32, handshake.name = "control_merge4"} : [<>, <>] to <>, <i1>
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

