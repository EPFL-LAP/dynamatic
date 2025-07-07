module {
  handshake.func @nested_loop(%arg0: memref<1000xi32>, %arg1: memref<1000xi32>, %arg2: memref<1000xi32>, %arg3: !handshake.control<>, %arg4: !handshake.control<>, %arg5: !handshake.control<>, %arg6: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["a", "b", "c", "a_start", "b_start", "c_start", "start"], resNames = ["a_end", "b_end", "c_end", "end"]} {
    %0:3 = fork [3] %arg6 {handshake.bb = 0 : ui32, handshake.name = "fork0"} : <>
    %memEnd = mem_controller[%arg2 : memref<1000xi32>] %arg5 (%37, %addressResult_5, %dataResult_6) %73#2 {connectedBlocks = [2 : i32], handshake.name = "mem_controller3"} :    (!handshake.channel<i32>, !handshake.channel<i10>, !handshake.channel<i32>) -> ()
    %outputs, %memEnd_0 = mem_controller[%arg1 : memref<1000xi32>] %arg4 (%addressResult_3) %73#1 {connectedBlocks = [2 : i32], handshake.name = "mem_controller4"} :    (!handshake.channel<i10>) -> !handshake.channel<i32>
    %outputs_1, %memEnd_2 = mem_controller[%arg0 : memref<1000xi32>] %arg3 (%addressResult) %73#0 {connectedBlocks = [2 : i32], handshake.name = "mem_controller5"} :    (!handshake.channel<i10>) -> !handshake.channel<i32>
    %1 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant3", value = false} : <>, <i1>
    %2 = extsi %1 {handshake.bb = 0 : ui32, handshake.name = "extsi8"} : <i1> to <i2>
    %3 = mux %index [%2, %trueResult] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i2>, <i2>] to <i2>
    %4:2 = fork [2] %3 {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <i2>
    %5 = extsi %4#0 {handshake.bb = 1 : ui32, handshake.name = "extsi9"} : <i2> to <i12>
    %result, %index = control_merge [%0#2, %trueResult_7]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %6:2 = fork [2] %result {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <>
    %7 = source {handshake.bb = 1 : ui32, handshake.name = "source0"} : <>
    %8 = constant %7 {handshake.bb = 1 : ui32, handshake.name = "constant4", value = 400 : i10} : <>, <i10>
    %9 = extsi %8 {handshake.bb = 1 : ui32, handshake.name = "extsi10"} : <i10> to <i12>
    %10 = constant %6#0 {handshake.bb = 1 : ui32, handshake.name = "constant5", value = false} : <>, <i1>
    %11 = muli %5, %9 {handshake.bb = 1 : ui32, handshake.name = "muli0"} : <i12>
    %12 = extsi %10 {handshake.bb = 1 : ui32, handshake.name = "extsi11"} : <i1> to <i32>
    %13 = mux %41#3 [%12, %44] {handshake.bb = 2 : ui32, handshake.name = "mux1"} : <i1>, [<i32>, <i32>] to <i32>
    %14:4 = fork [4] %13 {handshake.bb = 2 : ui32, handshake.name = "fork3"} : <i32>
    %15 = trunci %14#0 {handshake.bb = 2 : ui32, handshake.name = "trunci0"} : <i32> to <i10>
    %16 = trunci %14#1 {handshake.bb = 2 : ui32, handshake.name = "trunci1"} : <i32> to <i10>
    %17 = trunci %14#2 {handshake.bb = 2 : ui32, handshake.name = "trunci2"} : <i32> to <i10>
    %18 = mux %41#2 [%4#1, %42] {handshake.bb = 2 : ui32, handshake.name = "mux2"} : <i1>, [<i2>, <i2>] to <i2>
    %19 = mux %41#1 [%11, %45] {handshake.bb = 2 : ui32, handshake.name = "mux3"} : <i1>, [<i12>, <i12>] to <i12>
    %20:2 = fork [2] %19 {handshake.bb = 2 : ui32, handshake.name = "fork4"} : <i12>
    %21 = trunci %20#0 {handshake.bb = 2 : ui32, handshake.name = "trunci3"} : <i12> to <i10>
    %22 = mux %41#0 [%6#1, %43] {handshake.bb = 2 : ui32, handshake.name = "mux4"} : <i1>, [<>, <>] to <>
    %23:3 = fork [3] %22 {handshake.bb = 2 : ui32, handshake.name = "fork17"} : <>
    %24 = constant %23#0 {handshake.bb = 2 : ui32, handshake.name = "constant6", value = 1 : i2} : <>, <i2>
    %25 = extsi %24 {handshake.bb = 2 : ui32, handshake.name = "extsi3"} : <i2> to <i32>
    %26 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %27 = constant %26 {handshake.bb = 2 : ui32, handshake.name = "constant13", value = 1 : i2} : <>, <i2>
    %28 = extsi %27 {handshake.bb = 2 : ui32, handshake.name = "extsi4"} : <i2> to <i32>
    %29 = source {handshake.bb = 2 : ui32, handshake.name = "source2"} : <>
    %30 = constant %29 {handshake.bb = 2 : ui32, handshake.name = "constant14", value = 1000 : i11} : <>, <i11>
    %31 = extsi %30 {handshake.bb = 2 : ui32, handshake.name = "extsi5"} : <i11> to <i32>
    %addressResult, %dataResult = load[%17] %outputs_1 {handshake.bb = 2 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load2"} : <i10>, <i32>, <i10>, <i32>
    %addressResult_3, %dataResult_4 = load[%16] %outputs {handshake.bb = 2 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load3"} : <i10>, <i32>, <i10>, <i32>
    %32 = muli %dataResult, %dataResult_4 {handshake.bb = 2 : ui32, handshake.name = "muli1"} : <i32>
    %33:2 = fork [2] %32 {handshake.bb = 2 : ui32, handshake.name = "fork7"} : <i32>
    %34 = addi %15, %21 {handshake.bb = 2 : ui32, handshake.name = "addi0"} : <i10>
    %addressResult_5, %dataResult_6 = store[%38] %39 {handshake.bb = 2 : ui32, handshake.deps = #handshake<deps[<"store1" (0)>]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store1"} : <i10>, <i32>, <i10>, <i32>
    %35 = cmpi slt, %33#1, %31 {handshake.bb = 2 : ui32, handshake.name = "cmpi0"} : <i32>
    %36 = not %57#0 {handshake.bb = 2 : ui32, handshake.name = "not0"} : <i1>
    %37 = passer %25[%56#3] {handshake.bb = 2 : ui32, handshake.name = "passer0"} : <i32>, <i1>
    %38 = passer %34[%56#2] {handshake.bb = 2 : ui32, handshake.name = "passer1"} : <i10>, <i1>
    %39 = passer %33#0[%56#1] {handshake.bb = 2 : ui32, handshake.name = "passer2"} : <i32>, <i1>
    %40 = init %47#5 {handshake.bb = 2 : ui32, handshake.name = "init0"} : <i1>
    %41:4 = fork [4] %40 {handshake.bb = 2 : ui32, handshake.name = "fork18"} : <i1>
    %42 = passer %59#1[%47#4] {handshake.bb = 2 : ui32, handshake.name = "passer3"} : <i2>, <i1>
    %43 = passer %23#2[%47#3] {handshake.bb = 2 : ui32, handshake.name = "passer4"} : <>, <i1>
    %44 = passer %58[%47#2] {handshake.bb = 2 : ui32, handshake.name = "passer5"} : <i32>, <i1>
    %45 = passer %20#1[%47#1] {handshake.bb = 2 : ui32, handshake.name = "passer6"} : <i12>, <i1>
    %46 = spec_v2_repeating_init %48 {handshake.bb = 2 : ui32, handshake.name = "spec_v2_repeating_init0"} : <i1>
    %47:6 = fork [6] %46 {handshake.bb = 2 : ui32, handshake.name = "fork19"} : <i1>
    %48 = spec_v2_repeating_init %49 {handshake.bb = 2 : ui32, handshake.name = "spec_v2_repeating_init1"} : <i1>
    %49 = spec_v2_repeating_init %50 {handshake.bb = 2 : ui32, handshake.name = "spec_v2_repeating_init2"} : <i1>
    %50 = spec_v2_repeating_init %51 {handshake.bb = 2 : ui32, handshake.name = "spec_v2_repeating_init3"} : <i1>
    %51 = spec_v2_repeating_init %52 {handshake.bb = 2 : ui32, handshake.name = "spec_v2_repeating_init4"} : <i1>
    %52 = passer %57#2[%56#4] {handshake.bb = 2 : ui32, handshake.name = "passer7"} : <i1>, <i1>
    %53 = andi %56#0, %36 {handshake.bb = 2 : ui32, handshake.name = "andi0"} : <i1>
    %54:2 = fork [2] %53 {handshake.bb = 2 : ui32, handshake.name = "fork20"} : <i1>
    %55 = spec_v2_resolver %57#1, %47#0 {handshake.bb = 2 : ui32, handshake.name = "spec_v2_resolver0"} : <i1>
    %56:5 = fork [5] %55 {handshake.bb = 2 : ui32, handshake.name = "fork21"} : <i1>
    %57:3 = fork [3] %35 {handshake.bb = 2 : ui32, handshake.name = "fork22"} : <i1>
    %58 = addi %14#3, %28 {handshake.bb = 2 : ui32, handshake.name = "addi1"} : <i32>
    %59:2 = fork [2] %18 {handshake.bb = 2 : ui32, handshake.name = "fork16"} : <i2>
    %60 = passer %59#0[%54#1] {handshake.bb = 2 : ui32, handshake.name = "passer8"} : <i2>, <i1>
    %61 = passer %23#1[%54#0] {handshake.bb = 2 : ui32, handshake.name = "passer9"} : <>, <i1>
    %62 = extsi %60 {handshake.bb = 3 : ui32, handshake.name = "extsi12"} : <i2> to <i3>
    %63 = source {handshake.bb = 3 : ui32, handshake.name = "source3"} : <>
    %64 = constant %63 {handshake.bb = 3 : ui32, handshake.name = "constant15", value = 2 : i3} : <>, <i3>
    %65 = source {handshake.bb = 3 : ui32, handshake.name = "source4"} : <>
    %66 = constant %65 {handshake.bb = 3 : ui32, handshake.name = "constant16", value = 1 : i2} : <>, <i2>
    %67 = extsi %66 {handshake.bb = 3 : ui32, handshake.name = "extsi13"} : <i2> to <i3>
    %68 = addi %62, %67 {handshake.bb = 3 : ui32, handshake.name = "addi2"} : <i3>
    %69:2 = fork [2] %68 {handshake.bb = 3 : ui32, handshake.name = "fork9"} : <i3>
    %70 = trunci %69#0 {handshake.bb = 3 : ui32, handshake.name = "trunci4"} : <i3> to <i2>
    %71 = cmpi ult, %69#1, %64 {handshake.bb = 3 : ui32, handshake.name = "cmpi1"} : <i3>
    %72:2 = fork [2] %71 {handshake.bb = 3 : ui32, handshake.name = "fork10"} : <i1>
    %trueResult, %falseResult = cond_br %72#0, %70 {handshake.bb = 3 : ui32, handshake.name = "cond_br6"} : <i1>, <i2>
    sink %falseResult {handshake.name = "sink3"} : <i2>
    %trueResult_7, %falseResult_8 = cond_br %72#1, %61 {handshake.bb = 3 : ui32, handshake.name = "cond_br7"} : <i1>, <>
    %73:3 = fork [3] %falseResult_8 {handshake.bb = 4 : ui32, handshake.name = "fork11"} : <>
    end {handshake.bb = 4 : ui32, handshake.name = "end0"} %memEnd_2, %memEnd_0, %memEnd, %0#1 : <>, <>, <>, <>
  }
}

