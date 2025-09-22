module {
  handshake.func @collision_donut(%arg0: memref<1000xi32>, %arg1: memref<1000xi32>, %arg2: !handshake.control<>, %arg3: !handshake.control<>, %arg4: !handshake.control<>, ...) -> (!handshake.channel<i32>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["x", "y", "x_start", "y_start", "start"], resNames = ["out0", "x_end", "y_end", "end"]} {
    %0:3 = fork [3] %arg4 {handshake.bb = 0 : ui32, handshake.name = "fork0"} : <>
    %outputs, %memEnd = mem_controller[%arg1 : memref<1000xi32>] %arg3 (%addressResult_2) %75#1 {connectedBlocks = [1 : i32], handshake.name = "mem_controller2"} :    (!handshake.channel<i10>) -> !handshake.channel<i32>
    %outputs_0, %memEnd_1 = mem_controller[%arg0 : memref<1000xi32>] %arg2 (%addressResult) %75#0 {connectedBlocks = [1 : i32], handshake.name = "mem_controller3"} :    (!handshake.channel<i10>) -> !handshake.channel<i32>
    %1 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant0", value = false} : <>, <i1>
    %2 = extsi %1 {handshake.bb = 0 : ui32, handshake.name = "extsi7"} : <i1> to <i11>
    %3 = init %59#1 {handshake.bb = 1 : ui32, handshake.name = "init0", initToken = 0 : ui1} : <i1>
    %4 = init %59#0 {handshake.bb = 1 : ui32, handshake.name = "init1", initToken = 0 : ui1} : <i1>
    %5 = mux %3 [%2, %62] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i11>, <i11>] to <i11>
    %6:5 = fork [5] %5 {handshake.bb = 1 : ui32, handshake.name = "fork31"} : <i11>
    %7 = trunci %6#0 {handshake.bb = 1 : ui32, handshake.name = "trunci0"} : <i11> to <i10>
    %8 = trunci %6#1 {handshake.bb = 1 : ui32, handshake.name = "trunci1"} : <i11> to <i10>
    %9 = mux %4 [%0#2, %64] {handshake.bb = 1 : ui32, handshake.name = "mux7"} : <i1>, [<>, <>] to <>
    %10:7 = fork [7] %9 {handshake.bb = 1 : ui32, handshake.name = "fork32"} : <>
    %11 = source {handshake.bb = 1 : ui32, handshake.name = "source0"} : <>
    %12 = constant %11 {handshake.bb = 1 : ui32, handshake.name = "constant1", value = 4 : i4} : <>, <i4>
    %13 = extsi %12 {handshake.bb = 1 : ui32, handshake.name = "extsi1"} : <i4> to <i32>
    %14 = constant %10#0 {handshake.bb = 1 : ui32, handshake.name = "constant8", value = -1 : i32} : <>, <i32>
    %addressResult, %dataResult = load[%8] %outputs_0 {handshake.bb = 1 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i10>, <i32>, <i10>, <i32>
    %15:2 = fork [2] %dataResult {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <i32>
    %addressResult_2, %dataResult_3 = load[%7] %outputs {handshake.bb = 1 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i10>, <i32>, <i10>, <i32>
    %16:2 = fork [2] %dataResult_3 {handshake.bb = 1 : ui32, handshake.name = "fork4"} : <i32>
    %17 = muli %15#0, %15#1 {handshake.bb = 1 : ui32, handshake.name = "muli0"} : <i32>
    %18 = muli %16#0, %16#1 {handshake.bb = 1 : ui32, handshake.name = "muli1"} : <i32>
    %19 = addi %17, %18 {handshake.bb = 1 : ui32, handshake.name = "addi0"} : <i32>
    %20:2 = fork [2] %19 {handshake.bb = 1 : ui32, handshake.name = "fork5"} : <i32>
    %21 = cmpi ult, %20#1, %13 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i32>
    %22:4 = fork [4] %21 {handshake.bb = 1 : ui32, handshake.name = "fork33"} : <i1>
    %23 = not %22#3 {handshake.bb = 1 : ui32, handshake.name = "not0"} : <i1>
    %24:2 = fork [2] %23 {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <i1>
    %25 = andi %38#1, %24#0 {handshake.bb = 1 : ui32, handshake.name = "andi1"} : <i1>
    %26:3 = fork [3] %25 {handshake.bb = 1 : ui32, handshake.name = "fork34"} : <i1>
    %27 = andi %60, %41#0 {handshake.bb = 1 : ui32, handshake.name = "andi2"} : <i1>
    %28:3 = fork [3] %27 {handshake.bb = 1 : ui32, handshake.name = "fork35"} : <i1>
    %29 = passer %30[%22#2] {handshake.bb = 1 : ui32, handshake.name = "passer16"} : <i12>, <i1>
    %30 = extsi %6#2 {handshake.bb = 1 : ui32, handshake.name = "extsi8"} : <i11> to <i12>
    %31 = passer %14[%22#1] {handshake.bb = 1 : ui32, handshake.name = "passer2"} : <i32>, <i1>
    %32 = passer %10#1[%22#0] {handshake.bb = 1 : ui32, handshake.name = "passer3"} : <>, <i1>
    %33 = source {handshake.bb = 1 : ui32, handshake.name = "source1"} : <>
    %34 = constant %33 {handshake.bb = 1 : ui32, handshake.name = "constant2", value = 19000 : i16} : <>, <i16>
    %35 = extsi %34 {handshake.bb = 1 : ui32, handshake.name = "extsi2"} : <i16> to <i32>
    %36 = constant %10#2 {handshake.bb = 1 : ui32, handshake.name = "constant10", value = -2 : i32} : <>, <i32>
    %37 = cmpi ugt, %20#0, %35 {handshake.bb = 1 : ui32, handshake.name = "cmpi1"} : <i32>
    %38:2 = fork [2] %37 {handshake.bb = 1 : ui32, handshake.name = "fork10"} : <i1>
    %39 = not %38#0 {handshake.bb = 1 : ui32, handshake.name = "not1"} : <i1>
    %40 = andi %24#1, %39 {handshake.bb = 1 : ui32, handshake.name = "andi3"} : <i1>
    %41:2 = fork [2] %40 {handshake.bb = 1 : ui32, handshake.name = "fork11"} : <i1>
    %42 = passer %43[%26#2] {handshake.bb = 1 : ui32, handshake.name = "passer17"} : <i12>, <i1>
    %43 = extsi %6#4 {handshake.bb = 1 : ui32, handshake.name = "extsi9"} : <i11> to <i12>
    %44 = passer %36[%26#0] {handshake.bb = 1 : ui32, handshake.name = "passer8"} : <i32>, <i1>
    %45 = passer %10#3[%26#1] {handshake.bb = 1 : ui32, handshake.name = "passer9"} : <>, <i1>
    %46 = extsi %6#3 {handshake.bb = 1 : ui32, handshake.name = "extsi10"} : <i11> to <i12>
    %47 = constant %10#4 {handshake.bb = 1 : ui32, handshake.name = "constant3", value = false} : <>, <i1>
    %48 = source {handshake.bb = 1 : ui32, handshake.name = "source2"} : <>
    %49 = constant %48 {handshake.bb = 1 : ui32, handshake.name = "constant4", value = 1 : i2} : <>, <i2>
    %50 = extsi %49 {handshake.bb = 1 : ui32, handshake.name = "extsi11"} : <i2> to <i12>
    %51 = source {handshake.bb = 1 : ui32, handshake.name = "source3"} : <>
    %52 = constant %51 {handshake.bb = 1 : ui32, handshake.name = "constant5", value = 1000 : i11} : <>, <i11>
    %53 = extsi %52 {handshake.bb = 1 : ui32, handshake.name = "extsi12"} : <i11> to <i12>
    %54:3 = fork [3] %55 {handshake.bb = 1 : ui32, handshake.name = "fork28"} : <i12>
    %55 = addi %46, %50 {handshake.bb = 1 : ui32, handshake.name = "addi1"} : <i12>
    %56 = cmpi ult, %54#0, %53 {handshake.bb = 1 : ui32, handshake.name = "cmpi2"} : <i12>
    %57:2 = fork [2] %56 {handshake.bb = 1 : ui32, handshake.name = "fork29"} : <i1>
    %58 = andi %41#1, %57#0 {handshake.bb = 1 : ui32, handshake.name = "andi4"} : <i1>
    %59:4 = fork [4] %58 {handshake.bb = 1 : ui32, handshake.name = "fork30"} : <i1>
    %60 = not %57#1 {handshake.bb = 1 : ui32, handshake.name = "not2"} : <i1>
    %61 = passer %54#1[%28#0] {handshake.bb = 1 : ui32, handshake.name = "passer12"} : <i12>, <i1>
    %62 = passer %63[%59#2] {handshake.bb = 1 : ui32, handshake.name = "passer18"} : <i11>, <i1>
    %63 = trunci %54#2 {handshake.bb = 1 : ui32, handshake.name = "trunci2"} : <i12> to <i11>
    %64 = passer %10#6[%59#3] {handshake.bb = 1 : ui32, handshake.name = "passer13"} : <>, <i1>
    %65 = passer %10#5[%28#2] {handshake.bb = 1 : ui32, handshake.name = "passer14"} : <>, <i1>
    %66 = passer %67[%28#1] {handshake.bb = 1 : ui32, handshake.name = "passer19"} : <i32>, <i1>
    %67 = extsi %47 {handshake.bb = 1 : ui32, handshake.name = "extsi13"} : <i1> to <i32>
    %68 = mux %70#0 [%29, %61] {handshake.bb = 2 : ui32, handshake.name = "mux1"} : <i1>, [<i12>, <i12>] to <i12>
    %69 = mux %70#1 [%31, %66] {handshake.bb = 2 : ui32, handshake.name = "mux2"} : <i1>, [<i32>, <i32>] to <i32>
    %result, %index = control_merge [%32, %65]  {handshake.bb = 2 : ui32, handshake.name = "control_merge3"} : [<>, <>] to <>, <i1>
    %70:2 = fork [2] %index {handshake.bb = 2 : ui32, handshake.name = "fork12"} : <i1>
    %71 = mux %74#0 [%42, %68] {handshake.bb = 3 : ui32, handshake.name = "mux3"} : <i1>, [<i12>, <i12>] to <i12>
    %72 = extsi %71 {handshake.bb = 3 : ui32, handshake.name = "extsi14"} : <i12> to <i32>
    %73 = mux %74#1 [%44, %69] {handshake.bb = 3 : ui32, handshake.name = "mux4"} : <i1>, [<i32>, <i32>] to <i32>
    %result_4, %index_5 = control_merge [%45, %result]  {handshake.bb = 3 : ui32, handshake.name = "control_merge4"} : [<>, <>] to <>, <i1>
    %74:2 = fork [2] %index_5 {handshake.bb = 3 : ui32, handshake.name = "fork13"} : <i1>
    %75:2 = fork [2] %result_4 {handshake.bb = 3 : ui32, handshake.name = "fork14"} : <>
    %76 = source {handshake.bb = 3 : ui32, handshake.name = "source4"} : <>
    %77 = constant %76 {handshake.bb = 3 : ui32, handshake.name = "constant15", value = 1 : i2} : <>, <i2>
    %78 = extsi %77 {handshake.bb = 3 : ui32, handshake.name = "extsi6"} : <i2> to <i32>
    %79 = shli %72, %78 {handshake.bb = 3 : ui32, handshake.name = "shli0"} : <i32>
    %80 = andi %79, %73 {handshake.bb = 3 : ui32, handshake.name = "andi0"} : <i32>
    end {handshake.bb = 3 : ui32, handshake.name = "end0"} %80, %memEnd_1, %memEnd, %0#1 : <i32>, <>, <>, <>
  }
}

