module {
  handshake.func @nested_loop(%arg0: memref<1000xi32>, %arg1: memref<1000xi32>, %arg2: memref<1000xi32>, %arg3: !handshake.control<>, %arg4: !handshake.control<>, %arg5: !handshake.control<>, %arg6: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["a", "b", "c", "a_start", "b_start", "c_start", "start"], resNames = ["a_end", "b_end", "c_end", "end"]} {
    %0:3 = fork [3] %arg6 {handshake.bb = 0 : ui32, handshake.name = "fork0"} : <>
    %memEnd = mem_controller[%arg2 : memref<1000xi32>] %arg5 (%58, %addressResult_5, %dataResult_6) %72#2 {connectedBlocks = [2 : i32], handshake.name = "mem_controller3"} :    (!handshake.channel<i32>, !handshake.channel<i10>, !handshake.channel<i32>) -> ()
    %outputs, %memEnd_0 = mem_controller[%arg1 : memref<1000xi32>] %arg4 (%addressResult_3) %72#1 {connectedBlocks = [2 : i32], handshake.name = "mem_controller4"} :    (!handshake.channel<i10>) -> !handshake.channel<i32>
    %outputs_1, %memEnd_2 = mem_controller[%arg0 : memref<1000xi32>] %arg3 (%addressResult) %72#0 {connectedBlocks = [2 : i32], handshake.name = "mem_controller5"} :    (!handshake.channel<i10>) -> !handshake.channel<i32>
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
    %13 = mux %37#3 [%12, %52] {handshake.bb = 2 : ui32, handshake.name = "mux1"} : <i1>, [<i32>, <i32>] to <i32>
    %14:4 = fork [4] %13 {handshake.bb = 2 : ui32, handshake.name = "fork3"} : <i32>
    %15 = trunci %14#0 {handshake.bb = 2 : ui32, handshake.name = "trunci0"} : <i32> to <i10>
    %16 = trunci %14#1 {handshake.bb = 2 : ui32, handshake.name = "trunci1"} : <i32> to <i10>
    %17 = trunci %14#2 {handshake.bb = 2 : ui32, handshake.name = "trunci2"} : <i32> to <i10>
    %18 = mux %37#2 [%4#1, %54] {handshake.bb = 2 : ui32, handshake.name = "mux2"} : <i1>, [<i2>, <i2>] to <i2>
    %19 = mux %37#1 [%11, %57] {handshake.bb = 2 : ui32, handshake.name = "mux3"} : <i1>, [<i12>, <i12>] to <i12>
    %20:2 = fork [2] %19 {handshake.bb = 2 : ui32, handshake.name = "fork19"} : <i12>
    %21 = trunci %20#0 {handshake.bb = 2 : ui32, handshake.name = "trunci3"} : <i12> to <i10>
    %22 = mux %37#0 [%6#1, %59] {handshake.bb = 2 : ui32, handshake.name = "mux4"} : <i1>, [<>, <>] to <>
    %23:3 = fork [3] %22 {handshake.bb = 2 : ui32, handshake.name = "fork20"} : <>
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
    %addressResult_5, %dataResult_6 = store[%56] %51 {handshake.bb = 2 : ui32, handshake.deps = #handshake<deps[<"store1" (0)>]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store1"} : <i10>, <i32>, <i10>, <i32>
    %35 = cmpi slt, %33#1, %31 {handshake.bb = 2 : ui32, handshake.name = "cmpi0"} : <i32>
    %36 = init %42#0 {handshake.bb = 2 : ui32, handshake.name = "init0", initToken = 0 : ui1} : <i1>
    %37:4 = fork [4] %36 {handshake.bb = 2 : ui32, handshake.name = "fork21"} : <i1>
    %38 = source {handshake.bb = 2 : ui32, handshake.name = "source5", specv2_ignore_buffer = true} : <>
    %39 = constant %38 {handshake.bb = 2 : ui32, handshake.name = "constant0", specv2_ignore_buffer = true, value = true} : <>, <i1>
    %40 = buffer %47 {handshake.bb = 2 : ui32, handshake.name = "buffer0", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}, specv2_buffer_as_sink = true} : <i1>
    %41 = merge %40, %39 {handshake.bb = 2 : ui32, handshake.name = "merge1", specv2_buffer_as_source = true} : <i1>
    %42:6 = fork [6] %41 {handshake.bb = 2 : ui32, handshake.name = "fork22"} : <i1>
    %43 = andi %50, %46#0 {handshake.bb = 2 : ui32, handshake.name = "andi0"} : <i1>
    %44:3 = fork [3] %43 {handshake.bb = 2 : ui32, handshake.name = "fork23"} : <i1>
    %45 = spec_v2_resolver %48#1, %42#5 {handshake.bb = 2 : ui32, handshake.name = "spec_v2_resolver0"} : <i1>
    %46:4 = fork [4] %45 {handshake.bb = 2 : ui32, handshake.name = "fork24"} : <i1>
    %47 = passer %48#2[%44#0] {handshake.bb = 2 : ui32, handshake.name = "passer0"} : <i1>, <i1>
    %48:3 = fork [3] %35 {handshake.bb = 2 : ui32, handshake.name = "fork25"} : <i1>
    %49 = addi %14#3, %28 {handshake.bb = 2 : ui32, handshake.name = "addi1"} : <i32>
    %50 = not %48#0 {handshake.bb = 2 : ui32, handshake.name = "not0"} : <i1>
    %51 = passer %33#0[%46#2] {handshake.bb = 2 : ui32, handshake.name = "passer1"} : <i32>, <i1>
    %52 = passer %49[%42#3] {handshake.bb = 2 : ui32, handshake.name = "passer2"} : <i32>, <i1>
    %53:2 = fork [2] %18 {handshake.bb = 2 : ui32, handshake.name = "fork16"} : <i2>
    %54 = passer %53#1[%42#2] {handshake.bb = 2 : ui32, handshake.name = "passer3"} : <i2>, <i1>
    %55 = passer %53#0[%44#2] {handshake.bb = 2 : ui32, handshake.name = "passer4"} : <i2>, <i1>
    %56 = passer %34[%46#3] {handshake.bb = 2 : ui32, handshake.name = "passer5"} : <i10>, <i1>
    %57 = passer %20#1[%42#4] {handshake.bb = 2 : ui32, handshake.name = "passer6"} : <i12>, <i1>
    %58 = passer %25[%46#1] {handshake.bb = 2 : ui32, handshake.name = "passer7"} : <i32>, <i1>
    %59 = passer %23#2[%42#1] {handshake.bb = 2 : ui32, handshake.name = "passer8"} : <>, <i1>
    %60 = passer %23#1[%44#1] {handshake.bb = 2 : ui32, handshake.name = "passer9"} : <>, <i1>
    %61 = extsi %55 {handshake.bb = 3 : ui32, handshake.name = "extsi12"} : <i2> to <i3>
    %62 = source {handshake.bb = 3 : ui32, handshake.name = "source3"} : <>
    %63 = constant %62 {handshake.bb = 3 : ui32, handshake.name = "constant15", value = 2 : i3} : <>, <i3>
    %64 = source {handshake.bb = 3 : ui32, handshake.name = "source4"} : <>
    %65 = constant %64 {handshake.bb = 3 : ui32, handshake.name = "constant16", value = 1 : i2} : <>, <i2>
    %66 = extsi %65 {handshake.bb = 3 : ui32, handshake.name = "extsi13"} : <i2> to <i3>
    %67 = addi %61, %66 {handshake.bb = 3 : ui32, handshake.name = "addi2"} : <i3>
    %68:2 = fork [2] %67 {handshake.bb = 3 : ui32, handshake.name = "fork9"} : <i3>
    %69 = trunci %68#0 {handshake.bb = 3 : ui32, handshake.name = "trunci4"} : <i3> to <i2>
    %70 = cmpi ult, %68#1, %63 {handshake.bb = 3 : ui32, handshake.name = "cmpi1"} : <i3>
    %71:2 = fork [2] %70 {handshake.bb = 3 : ui32, handshake.name = "fork10"} : <i1>
    %trueResult, %falseResult = cond_br %71#0, %69 {handshake.bb = 3 : ui32, handshake.name = "cond_br6"} : <i1>, <i2>
    sink %falseResult {handshake.name = "sink3"} : <i2>
    %trueResult_7, %falseResult_8 = cond_br %71#1, %60 {handshake.bb = 3 : ui32, handshake.name = "cond_br7"} : <i1>, <>
    %72:3 = fork [3] %falseResult_8 {handshake.bb = 4 : ui32, handshake.name = "fork11"} : <>
    end {handshake.bb = 4 : ui32, handshake.name = "end0"} %memEnd_2, %memEnd_0, %memEnd, %0#1 : <>, <>, <>, <>
  }
}

