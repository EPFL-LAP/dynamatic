module {
  handshake.func @if_convert(%arg0: memref<200xi32>, %arg1: memref<200xi32>, %arg2: !handshake.control<>, %arg3: !handshake.control<>, %arg4: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["a", "b", "a_start", "b_start", "start"], handshake.cfdfcThroughput = #handshake<cfdfcThroughput {"0" = 2.000000e-01 : f64, "1" = 1.000000e+00 : f64}>, handshake.cfdfcToBBList = #handshake<cfdfcToBBList {"0" = [2 : ui32], "1" = [3 : ui32, 1 : ui32, 2 : ui32]}>, resNames = ["a_end", "b_end", "end"]} {
    %0:3 = fork [3] %arg4 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork0"} : <>
    %memEnd = mem_controller[%arg1 : memref<200xi32>] %arg3 (%53, %addressResult_1, %dataResult_2) %92#1 {connectedBlocks = [2 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller2"} :    (!handshake.channel<i32>, !handshake.channel<i8>, !handshake.channel<i32>) -> ()
    %outputs, %memEnd_0 = mem_controller[%arg0 : memref<200xi32>] %arg2 (%addressResult) %92#0 {connectedBlocks = [2 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller3"} :    (!handshake.channel<i8>) -> !handshake.channel<i32>
    %1 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant2", value = 1 : i2} : <>, <i2>
    %2 = extsi %1 {handshake.bb = 0 : ui32, handshake.name = "extsi7"} : <i2> to <i32>
    %3:2 = fork [2] %2 {handshake.bb = 0 : ui32, handshake.name = "fork1"} : <i32>
    %4 = mux %6#0 [%3#0, %91#0] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i32>, <i32>] to <i32>
    %5 = mux %6#1 [%3#1, %91#1] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i1>, [<i32>, <i32>] to <i32>
    %result, %index = control_merge [%0#2, %trueResult_3]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %6:2 = fork [2] %index {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <i1>
    %7 = buffer %4, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer1"} : <i32>
    %8 = mux %26#2 [%7, %44] {handshake.bb = 2 : ui32, handshake.name = "mux2"} : <i1>, [<i32>, <i32>] to <i32>
    %9 = buffer %8, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer3"} : <i32>
    %10:4 = fork [4] %9 {handshake.bb = 2 : ui32, handshake.name = "fork23"} : <i32>
    %11 = buffer %5, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer2"} : <i32>
    %12 = mux %26#1 [%11, %59] {handshake.bb = 2 : ui32, handshake.name = "mux3"} : <i1>, [<i32>, <i32>] to <i32>
    %13 = buffer %12, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer7"} : <i32>
    %14 = buffer %13, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer8"} : <i32>
    %15:5 = fork [5] %14 {handshake.bb = 2 : ui32, handshake.name = "fork24"} : <i32>
    %16 = mux %26#0 [%result, %60] {handshake.bb = 2 : ui32, handshake.name = "mux4"} : <i1>, [<>, <>] to <>
    %17 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %18 = constant %17 {handshake.bb = 2 : ui32, handshake.name = "constant9", value = 199 : i9} : <>, <i9>
    %19 = extsi %18 {handshake.bb = 2 : ui32, handshake.name = "extsi1"} : <i9> to <i32>
    %20 = cmpi slt, %15#1, %19 {handshake.bb = 2 : ui32, handshake.name = "cmpi0", internal_delay = "0_000000"} : <i32>
    %21 = buffer %10#0, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer4"} : <i32>
    %22 = cmpi eq, %15#0, %21 {handshake.bb = 2 : ui32, handshake.name = "cmpi1", internal_delay = "0_000000"} : <i32>
    %23 = andi %20, %22 {handshake.bb = 2 : ui32, handshake.name = "andi0", internal_delay = "0_000000"} : <i1>
    %24 = buffer %33#0, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 2 : ui32, handshake.name = "buffer13"} : <i1>
    %25 = init %24 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 2, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "init0", initToken = 0 : ui1} : <i1>
    %26:3 = fork [3] %25 {handshake.bb = 2 : ui32, handshake.name = "fork25"} : <i1>
    %27 = source {handshake.bb = 2 : ui32, handshake.name = "source4", specv2_ignore_buffer = true} : <>
    %28 = constant %27 {handshake.bb = 2 : ui32, handshake.name = "constant0", specv2_ignore_buffer = true, value = true} : <>, <i1>
    %29 = buffer %38, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer0", specv2_buffer_as_sink = true} : <i1>
    %30 = merge %29, %28 {handshake.bb = 2 : ui32, handshake.name = "merge0", specv2_buffer_as_source = true} : <i1>
    %31 = buffer %30, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer11"} : <i1>
    %32 = buffer %31, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer12"} : <i1>
    %33:5 = fork [5] %32 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork26"} : <i1>
    %34 = andi %40, %37#0 {handshake.bb = 2 : ui32, handshake.name = "andi1", internal_delay = "0_000000"} : <i1>
    %35:3 = fork [3] %34 {handshake.bb = 2 : ui32, handshake.name = "fork27"} : <i1>
    %36 = spec_v2_resolver %39#4, %33#4 {handshake.bb = 2 : ui32, handshake.name = "spec_v2_resolver0"} : <i1>
    %37:7 = fork [7] %36 {handshake.bb = 2 : ui32, handshake.name = "fork28"} : <i1>
    %38 = passer %39#5[%35#0] {handshake.bb = 2 : ui32, handshake.name = "passer0"} : <i1>, <i1>
    %39:6 = fork [6] %23 {handshake.bb = 2 : ui32, handshake.name = "fork29"} : <i1>
    %40 = not %39#0 {handshake.bb = 2 : ui32, handshake.name = "not0"} : <i1>
    %41 = buffer %10#1, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer5"} : <i32>
    %42 = passer %41[%35#2] {handshake.bb = 2 : ui32, handshake.name = "passer1"} : <i32>, <i1>
    %43 = buffer %33#3, bufferType = FIFO_BREAK_NONE, numSlots = 4 {handshake.bb = 2 : ui32, handshake.name = "buffer14"} : <i1>
    %44 = passer %81#1[%43] {handshake.bb = 2 : ui32, handshake.name = "passer2"} : <i32>, <i1>
    %45 = buffer %37#1, bufferType = FIFO_BREAK_NONE, numSlots = 4 {handshake.bb = 2 : ui32, handshake.name = "buffer15"} : <i1>
    %46 = passer %82[%45] {handshake.bb = 2 : ui32, handshake.name = "passer3"} : <i8>, <i1>
    %47 = buffer %51, bufferType = FIFO_BREAK_NONE, numSlots = 4 {handshake.bb = 2 : ui32, handshake.name = "buffer16"} : <i1>
    %48 = passer %46[%47] {handshake.bb = 2 : ui32, handshake.name = "passer4"} : <i8>, <i1>
    %49 = passer %39#1[%37#4] {handshake.bb = 2 : ui32, handshake.name = "passer5"} : <i1>, <i1>
    %50 = passer %39#2[%37#3] {handshake.bb = 2 : ui32, handshake.name = "passer6"} : <i1>, <i1>
    %51 = passer %39#3[%37#2] {handshake.bb = 2 : ui32, handshake.name = "passer7"} : <i1>, <i1>
    %52 = passer %57[%50] {handshake.bb = 2 : ui32, handshake.name = "passer8"} : <i32>, <i1>
    %53 = passer %58[%49] {handshake.bb = 2 : ui32, handshake.name = "passer9"} : <i32>, <i1>
    %54 = buffer %16, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer9"} : <>
    %55 = buffer %54, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer10"} : <>
    %56:3 = fork [3] %55 {handshake.bb = 2 : ui32, handshake.name = "fork22"} : <>
    %57 = passer %66#0[%37#5] {handshake.bb = 2 : ui32, handshake.name = "passer10"} : <i32>, <i1>
    %58 = passer %67[%37#6] {handshake.bb = 2 : ui32, handshake.name = "passer11"} : <i32>, <i1>
    %59 = passer %84[%33#2] {handshake.bb = 2 : ui32, handshake.name = "passer12"} : <i32>, <i1>
    %60 = passer %56#2[%33#1] {handshake.bb = 2 : ui32, handshake.name = "passer13"} : <>, <i1>
    %61 = passer %56#0[%35#1] {handshake.bb = 2 : ui32, handshake.name = "passer14"} : <>, <i1>
    %62 = trunci %10#2 {handshake.bb = 2 : ui32, handshake.name = "trunci0"} : <i32> to <i8>
    %63 = constant %56#1 {handshake.bb = 2 : ui32, handshake.name = "constant10", value = 1 : i2} : <>, <i2>
    %64:2 = fork [2] %63 {handshake.bb = 2 : ui32, handshake.name = "fork10"} : <i2>
    %65 = extsi %64#0 {handshake.bb = 2 : ui32, handshake.name = "extsi2"} : <i2> to <i32>
    %66:3 = fork [3] %65 {handshake.bb = 2 : ui32, handshake.name = "fork11"} : <i32>
    %67 = extsi %64#1 {handshake.bb = 2 : ui32, handshake.name = "extsi3"} : <i2> to <i32>
    %68 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %69 = constant %68 {handshake.bb = 2 : ui32, handshake.name = "constant11", value = 10000 : i15} : <>, <i15>
    %70 = extsi %69 {handshake.bb = 2 : ui32, handshake.name = "extsi4"} : <i15> to <i32>
    %71 = source {handshake.bb = 2 : ui32, handshake.name = "source2"} : <>
    %72 = constant %71 {handshake.bb = 2 : ui32, handshake.name = "constant12", value = 2 : i3} : <>, <i3>
    %73 = extsi %72 {handshake.bb = 2 : ui32, handshake.name = "extsi5"} : <i3> to <i32>
    %addressResult, %dataResult = load[%62] %outputs {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i8>, <i32>, <i8>, <i32>
    %74 = buffer %10#3, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer6"} : <i32>
    %75 = muli %74, %dataResult {handshake.bb = 2 : ui32, handshake.name = "muli0", internal_delay = "1_000000"} : <i32>
    %76 = cmpi slt, %75, %70 {handshake.bb = 2 : ui32, handshake.name = "cmpi2", internal_delay = "0_000000"} : <i32>
    %77 = addi %15#4, %73 {handshake.bb = 2 : ui32, handshake.name = "addi0", internal_delay = "0_000000"} : <i32>
    %78 = addi %15#3, %66#1 {handshake.bb = 2 : ui32, handshake.name = "addi1", internal_delay = "0_000000"} : <i32>
    %79 = buffer %78, bufferType = FIFO_BREAK_NONE, numSlots = 4 {handshake.bb = 2 : ui32, handshake.name = "buffer18"} : <i32>
    %80 = select %76[%77, %79] {handshake.bb = 2 : ui32, handshake.name = "select1", internal_delay = "0_000000"} : <i1>, <i32>
    %81:2 = fork [2] %80 {handshake.bb = 2 : ui32, handshake.name = "fork12"} : <i32>
    %82 = trunci %81#0 {handshake.bb = 2 : ui32, handshake.name = "trunci1"} : <i32> to <i8>
    %83 = buffer %52, bufferType = FIFO_BREAK_NONE, numSlots = 4 {handshake.bb = 2 : ui32, handshake.name = "buffer17"} : <i32>
    %addressResult_1, %dataResult_2 = store[%48] %83 {handshake.bb = 2 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store1"} : <i8>, <i32>, <i8>, <i32>
    %84 = addi %15#2, %66#2 {handshake.bb = 2 : ui32, handshake.name = "addi2", internal_delay = "0_000000"} : <i32>
    %85:2 = fork [2] %42 {handshake.bb = 3 : ui32, handshake.name = "fork13"} : <i32>
    %86 = source {handshake.bb = 3 : ui32, handshake.name = "source3"} : <>
    %87 = constant %86 {handshake.bb = 3 : ui32, handshake.name = "constant13", value = 199 : i9} : <>, <i9>
    %88 = extsi %87 {handshake.bb = 3 : ui32, handshake.name = "extsi6"} : <i9> to <i32>
    %89 = cmpi slt, %85#1, %88 {handshake.bb = 3 : ui32, handshake.name = "cmpi3", internal_delay = "0_000000"} : <i32>
    %90:2 = fork [2] %89 {handshake.bb = 3 : ui32, handshake.name = "fork14"} : <i1>
    %trueResult, %falseResult = cond_br %90#1, %85#0 {handshake.bb = 3 : ui32, handshake.name = "cond_br5"} : <i1>, <i32>
    sink %falseResult {handshake.name = "sink3"} : <i32>
    %91:2 = fork [2] %trueResult {handshake.bb = 3 : ui32, handshake.name = "fork15"} : <i32>
    %trueResult_3, %falseResult_4 = cond_br %90#0, %61 {handshake.bb = 3 : ui32, handshake.name = "cond_br6"} : <i1>, <>
    %92:2 = fork [2] %falseResult_4 {handshake.bb = 4 : ui32, handshake.name = "fork16"} : <>
    end {handshake.bb = 4 : ui32, handshake.name = "end0"} %memEnd_0, %memEnd, %0#1 : <>, <>, <>
  }
}

