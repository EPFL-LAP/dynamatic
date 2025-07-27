module {
  handshake.func @nested_loop(%arg0: memref<1000xi32>, %arg1: memref<1000xi32>, %arg2: memref<1000xi32>, %arg3: !handshake.control<>, %arg4: !handshake.control<>, %arg5: !handshake.control<>, %arg6: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["a", "b", "c", "a_start", "b_start", "c_start", "start"], resNames = ["a_end", "b_end", "c_end", "end"]} {
    %0:3 = fork [3] %arg6 {handshake.bb = 0 : ui32, handshake.name = "fork0"} : <>
    %memEnd = mem_controller[%arg2 : memref<1000xi32>] %arg5 (%31, %addressResult_7, %dataResult_8) %56#2 {connectedBlocks = [2 : i32], handshake.name = "mem_controller3"} :    (!handshake.channel<i32>, !handshake.channel<i10>, !handshake.channel<i32>) -> ()
    %outputs, %memEnd_0 = mem_controller[%arg1 : memref<1000xi32>] %arg4 (%addressResult_5) %56#1 {connectedBlocks = [2 : i32], handshake.name = "mem_controller4"} :    (!handshake.channel<i10>) -> !handshake.channel<i32>
    %outputs_1, %memEnd_2 = mem_controller[%arg0 : memref<1000xi32>] %arg3 (%addressResult) %56#0 {connectedBlocks = [2 : i32], handshake.name = "mem_controller5"} :    (!handshake.channel<i10>) -> !handshake.channel<i32>
    %1 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant3", value = false} : <>, <i1>
    %2 = br %1 {handshake.bb = 0 : ui32, handshake.name = "br2"} : <i1>
    %3 = extsi %2 {handshake.bb = 0 : ui32, handshake.name = "extsi8"} : <i1> to <i2>
    %4 = br %0#2 {handshake.bb = 0 : ui32, handshake.name = "br3"} : <>
    %5 = mux %index [%3, %trueResult_17] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i2>, <i2>] to <i2>
    %6:2 = fork [2] %5 {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <i2>
    %7 = extsi %6#0 {handshake.bb = 1 : ui32, handshake.name = "extsi9"} : <i2> to <i12>
    %result, %index = control_merge [%4, %trueResult_19]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %8:2 = fork [2] %result {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <>
    %9 = source {handshake.bb = 1 : ui32, handshake.name = "source0"} : <>
    %10 = constant %9 {handshake.bb = 1 : ui32, handshake.name = "constant4", value = 400 : i10} : <>, <i10>
    %11 = extsi %10 {handshake.bb = 1 : ui32, handshake.name = "extsi10"} : <i10> to <i12>
    %12 = constant %8#0 {handshake.bb = 1 : ui32, handshake.name = "constant5", value = false} : <>, <i1>
    %13 = muli %7, %11 {handshake.bb = 1 : ui32, handshake.name = "muli0"} : <i12>
    %14 = br %12 {handshake.bb = 1 : ui32, handshake.name = "br4"} : <i1>
    %15 = extsi %14 {handshake.bb = 1 : ui32, handshake.name = "extsi11"} : <i1> to <i32>
    %16 = br %6#1 {handshake.bb = 1 : ui32, handshake.name = "br5"} : <i2>
    %17 = br %13 {handshake.bb = 1 : ui32, handshake.name = "br6"} : <i12>
    %18 = br %8#1 {handshake.bb = 1 : ui32, handshake.name = "br7"} : <>
    %19 = mux %28#2 [%15, %trueResult] {handshake.bb = 2 : ui32, handshake.name = "mux1"} : <i1>, [<i32>, <i32>] to <i32>
    %20:4 = fork [4] %19 {handshake.bb = 2 : ui32, handshake.name = "fork3"} : <i32>
    %21 = trunci %20#0 {handshake.bb = 2 : ui32, handshake.name = "trunci0"} : <i32> to <i10>
    %22 = trunci %20#1 {handshake.bb = 2 : ui32, handshake.name = "trunci1"} : <i32> to <i10>
    %23 = trunci %20#2 {handshake.bb = 2 : ui32, handshake.name = "trunci2"} : <i32> to <i10>
    %24 = mux %28#1 [%16, %trueResult_9] {handshake.bb = 2 : ui32, handshake.name = "mux2"} : <i1>, [<i2>, <i2>] to <i2>
    %25 = mux %28#0 [%17, %trueResult_11] {handshake.bb = 2 : ui32, handshake.name = "mux3"} : <i1>, [<i12>, <i12>] to <i12>
    %26:2 = fork [2] %25 {handshake.bb = 2 : ui32, handshake.name = "fork4"} : <i12>
    %27 = trunci %26#0 {handshake.bb = 2 : ui32, handshake.name = "trunci3"} : <i12> to <i10>
    %result_3, %index_4 = control_merge [%18, %trueResult_13]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>, <>] to <>, <i1>
    %28:3 = fork [3] %index_4 {handshake.bb = 2 : ui32, handshake.name = "fork5"} : <i1>
    %29:2 = fork [2] %result_3 {handshake.bb = 2 : ui32, handshake.name = "fork6"} : <>
    %30 = constant %29#0 {handshake.bb = 2 : ui32, handshake.name = "constant6", value = 1 : i2} : <>, <i2>
    %31 = extsi %30 {handshake.bb = 2 : ui32, handshake.name = "extsi3"} : <i2> to <i32>
    %32 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %33 = constant %32 {handshake.bb = 2 : ui32, handshake.name = "constant13", value = 1 : i2} : <>, <i2>
    %34 = extsi %33 {handshake.bb = 2 : ui32, handshake.name = "extsi4"} : <i2> to <i32>
    %35 = source {handshake.bb = 2 : ui32, handshake.name = "source2"} : <>
    %36 = constant %35 {handshake.bb = 2 : ui32, handshake.name = "constant14", value = 1000 : i11} : <>, <i11>
    %37 = extsi %36 {handshake.bb = 2 : ui32, handshake.name = "extsi5"} : <i11> to <i32>
    %addressResult, %dataResult = load[%23] %outputs_1 {handshake.bb = 2 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load2"} : <i10>, <i32>, <i10>, <i32>
    %addressResult_5, %dataResult_6 = load[%22] %outputs {handshake.bb = 2 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load3"} : <i10>, <i32>, <i10>, <i32>
    %38 = muli %dataResult, %dataResult_6 {handshake.bb = 2 : ui32, handshake.name = "muli1"} : <i32>
    %39:2 = fork [2] %38 {handshake.bb = 2 : ui32, handshake.name = "fork7"} : <i32>
    %40 = addi %21, %27 {handshake.bb = 2 : ui32, handshake.name = "addi0"} : <i10>
    %addressResult_7, %dataResult_8 = store[%40] %39#0 {handshake.bb = 2 : ui32, handshake.deps = #handshake<deps[<"store1" (0)>]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store1"} : <i10>, <i32>, <i10>, <i32>
    %41 = cmpi slt, %39#1, %37 {handshake.bb = 2 : ui32, handshake.name = "cmpi0"} : <i32>
    %42:4 = fork [4] %41 {handshake.bb = 2 : ui32, handshake.name = "fork8"} : <i1>
    %43 = addi %20#3, %34 {handshake.bb = 2 : ui32, handshake.name = "addi1"} : <i32>
    %trueResult, %falseResult = cond_br %42#3, %43 {handshake.bb = 2 : ui32, handshake.name = "cond_br2"} : <i1>, <i32>
    sink %falseResult {handshake.name = "sink0"} : <i32>
    %trueResult_9, %falseResult_10 = cond_br %42#1, %24 {handshake.bb = 2 : ui32, handshake.name = "cond_br3"} : <i1>, <i2>
    %trueResult_11, %falseResult_12 = cond_br %42#0, %26#1 {handshake.bb = 2 : ui32, handshake.name = "cond_br4"} : <i1>, <i12>
    sink %falseResult_12 {handshake.name = "sink1"} : <i12>
    %trueResult_13, %falseResult_14 = cond_br %42#2, %29#1 {handshake.bb = 2 : ui32, handshake.name = "cond_br5"} : <i1>, <>
    %44 = merge %falseResult_10 {handshake.bb = 3 : ui32, handshake.name = "merge0"} : <i2>
    %45 = extsi %44 {handshake.bb = 3 : ui32, handshake.name = "extsi12"} : <i2> to <i3>
    %result_15, %index_16 = control_merge [%falseResult_14]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>] to <>, <i1>
    sink %index_16 {handshake.name = "sink2"} : <i1>
    %46 = source {handshake.bb = 3 : ui32, handshake.name = "source3"} : <>
    %47 = constant %46 {handshake.bb = 3 : ui32, handshake.name = "constant15", value = 2 : i3} : <>, <i3>
    %48 = source {handshake.bb = 3 : ui32, handshake.name = "source4"} : <>
    %49 = constant %48 {handshake.bb = 3 : ui32, handshake.name = "constant16", value = 1 : i2} : <>, <i2>
    %50 = extsi %49 {handshake.bb = 3 : ui32, handshake.name = "extsi13"} : <i2> to <i3>
    %51 = addi %45, %50 {handshake.bb = 3 : ui32, handshake.name = "addi2"} : <i3>
    %52:2 = fork [2] %51 {handshake.bb = 3 : ui32, handshake.name = "fork9"} : <i3>
    %53 = trunci %52#0 {handshake.bb = 3 : ui32, handshake.name = "trunci4"} : <i3> to <i2>
    %54 = cmpi ult, %52#1, %47 {handshake.bb = 3 : ui32, handshake.name = "cmpi1"} : <i3>
    %55:2 = fork [2] %54 {handshake.bb = 3 : ui32, handshake.name = "fork10"} : <i1>
    %trueResult_17, %falseResult_18 = cond_br %55#0, %53 {handshake.bb = 3 : ui32, handshake.name = "cond_br6"} : <i1>, <i2>
    sink %falseResult_18 {handshake.name = "sink3"} : <i2>
    %trueResult_19, %falseResult_20 = cond_br %55#1, %result_15 {handshake.bb = 3 : ui32, handshake.name = "cond_br7"} : <i1>, <>
    %result_21, %index_22 = control_merge [%falseResult_20]  {handshake.bb = 4 : ui32, handshake.name = "control_merge3"} : [<>] to <>, <i1>
    sink %index_22 {handshake.name = "sink4"} : <i1>
    %56:3 = fork [3] %result_21 {handshake.bb = 4 : ui32, handshake.name = "fork11"} : <>
    end {handshake.bb = 4 : ui32, handshake.name = "end0"} %memEnd_2, %memEnd_0, %memEnd, %0#1 : <>, <>, <>, <>
  }
}

