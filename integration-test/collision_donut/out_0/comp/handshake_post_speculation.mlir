module {
  handshake.func @collision_donut(%arg0: memref<1000xi32>, %arg1: memref<1000xi32>, %arg2: !handshake.control<>, %arg3: !handshake.control<>, %arg4: !handshake.control<>, ...) -> (!handshake.channel<i32>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["x", "y", "x_start", "y_start", "start"], resNames = ["out0", "x_end", "y_end", "end"]} {
    %0:3 = fork [3] %arg4 {handshake.bb = 0 : ui32, handshake.name = "fork0"} : <>
    %outputs, %memEnd = mem_controller[%arg1 : memref<1000xi32>] %arg3 (%addressResult_2) %75#1 {connectedBlocks = [1 : i32], handshake.name = "mem_controller2"} :    (!handshake.channel<i10>) -> !handshake.channel<i32>
    %outputs_0, %memEnd_1 = mem_controller[%arg0 : memref<1000xi32>] %arg2 (%addressResult) %75#0 {connectedBlocks = [1 : i32], handshake.name = "mem_controller3"} :    (!handshake.channel<i10>) -> !handshake.channel<i32>
    %1 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant0", value = false} : <>, <i1>
    %2 = extsi %1 {handshake.bb = 0 : ui32, handshake.name = "extsi7"} : <i1> to <i11>
    %3 = mux %59#0 [%2, %62] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i11>, <i11>] to <i11>
    %4:5 = fork [5] %3 {handshake.bb = 1 : ui32, handshake.name = "fork31"} : <i11>
    %5 = trunci %4#0 {handshake.bb = 1 : ui32, handshake.name = "trunci0"} : <i11> to <i10>
    %6 = trunci %4#1 {handshake.bb = 1 : ui32, handshake.name = "trunci1"} : <i11> to <i10>
    %7 = mux %59#1 [%0#2, %64] {handshake.bb = 1 : ui32, handshake.name = "mux7"} : <i1>, [<>, <>] to <>
    %8:7 = fork [7] %7 {handshake.bb = 1 : ui32, handshake.name = "fork32"} : <>
    %9 = source {handshake.bb = 1 : ui32, handshake.name = "source0"} : <>
    %10 = constant %9 {handshake.bb = 1 : ui32, handshake.name = "constant1", value = 4 : i4} : <>, <i4>
    %11 = extsi %10 {handshake.bb = 1 : ui32, handshake.name = "extsi1"} : <i4> to <i32>
    %12 = constant %8#0 {handshake.bb = 1 : ui32, handshake.name = "constant8", value = -1 : i32} : <>, <i32>
    %addressResult, %dataResult = load[%6] %outputs_0 {handshake.bb = 1 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i10>, <i32>, <i10>, <i32>
    %13:2 = fork [2] %dataResult {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <i32>
    %addressResult_2, %dataResult_3 = load[%5] %outputs {handshake.bb = 1 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i10>, <i32>, <i10>, <i32>
    %14:2 = fork [2] %dataResult_3 {handshake.bb = 1 : ui32, handshake.name = "fork4"} : <i32>
    %15 = muli %13#0, %13#1 {handshake.bb = 1 : ui32, handshake.name = "muli0"} : <i32>
    %16 = muli %14#0, %14#1 {handshake.bb = 1 : ui32, handshake.name = "muli1"} : <i32>
    %17 = addi %15, %16 {handshake.bb = 1 : ui32, handshake.name = "addi0"} : <i32>
    %18:2 = fork [2] %17 {handshake.bb = 1 : ui32, handshake.name = "fork5"} : <i32>
    %19 = cmpi ult, %18#1, %11 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i32>
    %20:4 = fork [4] %19 {handshake.bb = 1 : ui32, handshake.name = "fork33"} : <i1>
    %21 = not %20#3 {handshake.bb = 1 : ui32, handshake.name = "not0"} : <i1>
    %22:2 = fork [2] %21 {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <i1>
    %23 = andi %36#1, %22#0 {handshake.bb = 1 : ui32, handshake.name = "andi1"} : <i1>
    %24:3 = fork [3] %23 {handshake.bb = 1 : ui32, handshake.name = "fork34"} : <i1>
    %25 = andi %60, %39#0 {handshake.bb = 1 : ui32, handshake.name = "andi2"} : <i1>
    %26:3 = fork [3] %25 {handshake.bb = 1 : ui32, handshake.name = "fork35"} : <i1>
    %27 = passer %28[%20#2] {handshake.bb = 1 : ui32, handshake.name = "passer16"} : <i12>, <i1>
    %28 = extsi %4#2 {handshake.bb = 1 : ui32, handshake.name = "extsi8"} : <i11> to <i12>
    %29 = passer %12[%20#1] {handshake.bb = 1 : ui32, handshake.name = "passer2"} : <i32>, <i1>
    %30 = passer %8#1[%20#0] {handshake.bb = 1 : ui32, handshake.name = "passer3"} : <>, <i1>
    %31 = source {handshake.bb = 1 : ui32, handshake.name = "source1"} : <>
    %32 = constant %31 {handshake.bb = 1 : ui32, handshake.name = "constant2", value = 19000 : i16} : <>, <i16>
    %33 = extsi %32 {handshake.bb = 1 : ui32, handshake.name = "extsi2"} : <i16> to <i32>
    %34 = constant %8#2 {handshake.bb = 1 : ui32, handshake.name = "constant10", value = -2 : i32} : <>, <i32>
    %35 = cmpi ugt, %18#0, %33 {handshake.bb = 1 : ui32, handshake.name = "cmpi1"} : <i32>
    %36:2 = fork [2] %35 {handshake.bb = 1 : ui32, handshake.name = "fork10"} : <i1>
    %37 = not %36#0 {handshake.bb = 1 : ui32, handshake.name = "not1"} : <i1>
    %38 = andi %22#1, %37 {handshake.bb = 1 : ui32, handshake.name = "andi3"} : <i1>
    %39:2 = fork [2] %38 {handshake.bb = 1 : ui32, handshake.name = "fork11"} : <i1>
    %40 = passer %41[%24#2] {handshake.bb = 1 : ui32, handshake.name = "passer17"} : <i12>, <i1>
    %41 = extsi %4#4 {handshake.bb = 1 : ui32, handshake.name = "extsi9"} : <i11> to <i12>
    %42 = passer %34[%24#0] {handshake.bb = 1 : ui32, handshake.name = "passer8"} : <i32>, <i1>
    %43 = passer %8#3[%24#1] {handshake.bb = 1 : ui32, handshake.name = "passer9"} : <>, <i1>
    %44 = extsi %4#3 {handshake.bb = 1 : ui32, handshake.name = "extsi10"} : <i11> to <i12>
    %45 = constant %8#4 {handshake.bb = 1 : ui32, handshake.name = "constant3", value = false} : <>, <i1>
    %46 = source {handshake.bb = 1 : ui32, handshake.name = "source2"} : <>
    %47 = constant %46 {handshake.bb = 1 : ui32, handshake.name = "constant4", value = 1 : i2} : <>, <i2>
    %48 = extsi %47 {handshake.bb = 1 : ui32, handshake.name = "extsi11"} : <i2> to <i12>
    %49 = source {handshake.bb = 1 : ui32, handshake.name = "source3"} : <>
    %50 = constant %49 {handshake.bb = 1 : ui32, handshake.name = "constant5", value = 1000 : i11} : <>, <i11>
    %51 = extsi %50 {handshake.bb = 1 : ui32, handshake.name = "extsi12"} : <i11> to <i12>
    %52:3 = fork [3] %53 {handshake.bb = 1 : ui32, handshake.name = "fork28"} : <i12>
    %53 = addi %44, %48 {handshake.bb = 1 : ui32, handshake.name = "addi1"} : <i12>
    %54 = cmpi ult, %52#0, %51 {handshake.bb = 1 : ui32, handshake.name = "cmpi2"} : <i12>
    %55:2 = fork [2] %54 {handshake.bb = 1 : ui32, handshake.name = "fork29"} : <i1>
    %56 = andi %39#1, %55#0 {handshake.bb = 1 : ui32, handshake.name = "andi4"} : <i1>
    %57:3 = fork [3] %56 {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <i1>
    %58 = init %57#2 {handshake.bb = 1 : ui32, handshake.name = "init2", initToken = 0 : ui1} : <i1>
    %59:2 = fork [2] %58 {handshake.bb = 1 : ui32, handshake.name = "fork6"} : <i1>
    %60 = not %55#1 {handshake.bb = 1 : ui32, handshake.name = "not2"} : <i1>
    %61 = passer %52#1[%26#0] {handshake.bb = 1 : ui32, handshake.name = "passer12"} : <i12>, <i1>
    %62 = passer %63[%57#1] {handshake.bb = 1 : ui32, handshake.name = "passer18"} : <i11>, <i1>
    %63 = trunci %52#2 {handshake.bb = 1 : ui32, handshake.name = "trunci2"} : <i12> to <i11>
    %64 = passer %8#6[%57#0] {handshake.bb = 1 : ui32, handshake.name = "passer13"} : <>, <i1>
    %65 = passer %8#5[%26#2] {handshake.bb = 1 : ui32, handshake.name = "passer14"} : <>, <i1>
    %66 = passer %67[%26#1] {handshake.bb = 1 : ui32, handshake.name = "passer19"} : <i32>, <i1>
    %67 = extsi %45 {handshake.bb = 1 : ui32, handshake.name = "extsi13"} : <i1> to <i32>
    %68 = mux %70#0 [%27, %61] {handshake.bb = 2 : ui32, handshake.name = "mux1"} : <i1>, [<i12>, <i12>] to <i12>
    %69 = mux %70#1 [%29, %66] {handshake.bb = 2 : ui32, handshake.name = "mux2"} : <i1>, [<i32>, <i32>] to <i32>
    %result, %index = control_merge [%30, %65]  {handshake.bb = 2 : ui32, handshake.name = "control_merge3"} : [<>, <>] to <>, <i1>
    %70:2 = fork [2] %index {handshake.bb = 2 : ui32, handshake.name = "fork12"} : <i1>
    %71 = mux %74#0 [%40, %68] {handshake.bb = 3 : ui32, handshake.name = "mux3"} : <i1>, [<i12>, <i12>] to <i12>
    %72 = extsi %71 {handshake.bb = 3 : ui32, handshake.name = "extsi14"} : <i12> to <i32>
    %73 = mux %74#1 [%42, %69] {handshake.bb = 3 : ui32, handshake.name = "mux4"} : <i1>, [<i32>, <i32>] to <i32>
    %result_4, %index_5 = control_merge [%43, %result]  {handshake.bb = 3 : ui32, handshake.name = "control_merge4"} : [<>, <>] to <>, <i1>
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

