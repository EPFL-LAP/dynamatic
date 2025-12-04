module {
  handshake.func @single_loop_unrolled(%arg0: memref<7xi32>, %arg1: memref<7xi32>, %arg2: memref<7xi32>, %arg3: memref<7xi32>, %arg4: !handshake.control<>, %arg5: !handshake.control<>, %arg6: !handshake.control<>, %arg7: !handshake.control<>, %arg8: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["a", "b", "c0", "c1", "a_start", "b_start", "c0_start", "c1_start", "start"], handshake.cfdfcThroughput = #handshake<cfdfcThroughput {"0" = 2.000000e-01 : f64, "1" = 2.000000e-01 : f64}>, handshake.cfdfcToBBList = #handshake<cfdfcToBBList {"0" = [1 : ui32], "1" = [2 : ui32]}>, resNames = ["a_end", "b_end", "c0_end", "c1_end", "end"]} {
    %0:3 = fork [3] %arg8 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork0"} : <>
    %memEnd = mem_controller[%arg3 : memref<7xi32>] %arg7 (%50, %addressResult_12, %dataResult_13) %71#3 {connectedBlocks = [2 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller4"} :    (!handshake.channel<i32>, !handshake.channel<i3>, !handshake.channel<i32>) -> ()
    %memEnd_0 = mem_controller[%arg2 : memref<7xi32>] %arg6 (%14, %addressResult_6, %dataResult_7) %71#2 {connectedBlocks = [1 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller5"} :    (!handshake.channel<i32>, !handshake.channel<i3>, !handshake.channel<i32>) -> ()
    %outputs:2, %memEnd_1 = mem_controller[%arg1 : memref<7xi32>] %arg5 (%addressResult_4, %addressResult_10) %71#1 {connectedBlocks = [1 : i32, 2 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller6"} :    (!handshake.channel<i3>, !handshake.channel<i3>) -> (!handshake.channel<i32>, !handshake.channel<i32>)
    %outputs_2:2, %memEnd_3 = mem_controller[%arg0 : memref<7xi32>] %arg4 (%addressResult, %addressResult_8) %71#0 {connectedBlocks = [1 : i32, 2 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller7"} :    (!handshake.channel<i3>, !handshake.channel<i3>) -> (!handshake.channel<i32>, !handshake.channel<i32>)
    %1 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant10", value = false} : <>, <i1>
    %2 = extsi %1 {handshake.bb = 0 : ui32, handshake.name = "extsi8"} : <i1> to <i32>
    %3 = mux %30#0 [%2, %34] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i32>, <i32>] to <i32>
    %4 = buffer %3, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer0"} : <i32>
    %5:4 = fork [4] %4 {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <i32>
    %6 = trunci %5#0 {handshake.bb = 1 : ui32, handshake.name = "trunci0"} : <i32> to <i3>
    %7 = trunci %5#1 {handshake.bb = 1 : ui32, handshake.name = "trunci1"} : <i32> to <i3>
    %8 = trunci %5#2 {handshake.bb = 1 : ui32, handshake.name = "trunci2"} : <i32> to <i3>
    %9 = mux %30#1 [%0#2, %35] {handshake.bb = 1 : ui32, handshake.name = "mux3"} : <i1>, [<>, <>] to <>
    %10 = buffer %9, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer2"} : <>
    %11 = buffer %10, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer3"} : <>
    %12:4 = fork [4] %11 {handshake.bb = 1 : ui32, handshake.name = "fork12"} : <>
    %13 = constant %12#1 {handshake.bb = 1 : ui32, handshake.name = "constant11", value = 1 : i2} : <>, <i2>
    %14 = extsi %13 {handshake.bb = 1 : ui32, handshake.name = "extsi1"} : <i2> to <i32>
    %15 = constant %12#0 {handshake.bb = 1 : ui32, handshake.name = "constant12", value = false} : <>, <i1>
    %16 = source {handshake.bb = 1 : ui32, handshake.name = "source0"} : <>
    %17 = constant %16 {handshake.bb = 1 : ui32, handshake.name = "constant13", value = 1 : i2} : <>, <i2>
    %18 = extsi %17 {handshake.bb = 1 : ui32, handshake.name = "extsi3"} : <i2> to <i32>
    %19 = source {handshake.bb = 1 : ui32, handshake.name = "source1"} : <>
    %20 = constant %19 {handshake.bb = 1 : ui32, handshake.name = "constant14", value = 7 : i4} : <>, <i4>
    %21 = extsi %20 {handshake.bb = 1 : ui32, handshake.name = "extsi4"} : <i4> to <i32>
    %addressResult, %dataResult = load[%8] %outputs_2#0 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i3>, <i32>, <i3>, <i32>
    %addressResult_4, %dataResult_5 = load[%7] %outputs#0 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i3>, <i32>, <i3>, <i32>
    %22 = muli %dataResult, %dataResult_5 {handshake.bb = 1 : ui32, handshake.name = "muli0"} : <i32>
    %23:2 = fork [2] %22 {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <i32>
    %24 = buffer %6, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer1"} : <i3>
    %addressResult_6, %dataResult_7 = store[%24] %23#0 {handshake.bb = 1 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store0"} : <i3>, <i32>, <i3>, <i32>
    %25 = addi %5#3, %18 {handshake.bb = 1 : ui32, handshake.name = "addi0"} : <i32>
    %26 = cmpi slt, %23#1, %21 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i32>
    %27:4 = fork [4] %26 {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <i1>
    %28 = buffer %27#3, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer5"} : <i1>
    %29 = init %28 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "init4", initToken = 0 : ui1} : <i1>
    %30:2 = fork [2] %29 {handshake.bb = 1 : ui32, handshake.name = "fork4"} : <i1>
    %31 = not %27#0 {handshake.bb = 1 : ui32, handshake.name = "not1"} : <i1>
    %32:2 = fork [2] %31 {handshake.bb = 1 : ui32, handshake.name = "fork14"} : <i1>
    %33 = buffer %25, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer4"} : <i32>
    %34 = passer %33[%27#1] {handshake.bb = 1 : ui32, handshake.name = "passer3"} : <i32>, <i1>
    %35 = passer %12#3[%27#2] {handshake.bb = 1 : ui32, handshake.name = "passer4"} : <>, <i1>
    %36 = passer %12#2[%32#1] {handshake.bb = 1 : ui32, handshake.name = "passer5"} : <>, <i1>
    %37 = passer %15[%32#0] {handshake.bb = 1 : ui32, handshake.name = "passer6"} : <i1>, <i1>
    %38 = extsi %37 {handshake.bb = 1 : ui32, handshake.name = "extsi9"} : <i1> to <i32>
    %break_dv_1 = buffer %38, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer_man1"} : <i32>
    %break_r_1 = buffer %break_dv_1, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer_man2"} : <i32>
    %39 = mux %65#0 [%break_r_1, %68] {handshake.bb = 2 : ui32, handshake.name = "mux1"} : <i1>, [<i32>, <i32>] to <i32>
    %40 = buffer %39, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer6"} : <i32>
    %41:4 = fork [4] %40 {handshake.bb = 2 : ui32, handshake.name = "fork5"} : <i32>
    %42 = trunci %41#0 {handshake.bb = 2 : ui32, handshake.name = "trunci3"} : <i32> to <i3>
    %43 = trunci %41#1 {handshake.bb = 2 : ui32, handshake.name = "trunci4"} : <i32> to <i3>
    %44 = trunci %41#2 {handshake.bb = 2 : ui32, handshake.name = "trunci5"} : <i32> to <i3>
    %break_dv_2 = buffer %36, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer_man3"} : <>
    %break_r_2 = buffer %break_dv_2, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer_man4"} : <>
    %45 = mux %65#1 [%break_r_2, %69] {handshake.bb = 2 : ui32, handshake.name = "mux2"} : <i1>, [<>, <>] to <>
    %46 = buffer %45, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer8"} : <>
    %47 = buffer %46, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer9"} : <>
    %48:3 = fork [3] %47 {handshake.bb = 2 : ui32, handshake.name = "fork10"} : <>
    %49 = constant %48#0 {handshake.bb = 2 : ui32, handshake.name = "constant15", value = 1 : i2} : <>, <i2>
    %50 = extsi %49 {handshake.bb = 2 : ui32, handshake.name = "extsi5"} : <i2> to <i32>
    %51 = source {handshake.bb = 2 : ui32, handshake.name = "source2"} : <>
    %52 = constant %51 {handshake.bb = 2 : ui32, handshake.name = "constant16", value = 1 : i2} : <>, <i2>
    %53 = extsi %52 {handshake.bb = 2 : ui32, handshake.name = "extsi6"} : <i2> to <i32>
    %54 = source {handshake.bb = 2 : ui32, handshake.name = "source3"} : <>
    %55 = constant %54 {handshake.bb = 2 : ui32, handshake.name = "constant17", value = 7 : i4} : <>, <i4>
    %56 = extsi %55 {handshake.bb = 2 : ui32, handshake.name = "extsi7"} : <i4> to <i32>
    %addressResult_8, %dataResult_9 = load[%44] %outputs_2#1 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load2"} : <i3>, <i32>, <i3>, <i32>
    %addressResult_10, %dataResult_11 = load[%43] %outputs#1 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load3"} : <i3>, <i32>, <i3>, <i32>
    %57 = muli %dataResult_9, %dataResult_11 {handshake.bb = 2 : ui32, handshake.name = "muli1"} : <i32>
    %58:2 = fork [2] %57 {handshake.bb = 2 : ui32, handshake.name = "fork7"} : <i32>
    %59 = buffer %42, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer7"} : <i3>
    %addressResult_12, %dataResult_13 = store[%59] %58#0 {handshake.bb = 2 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store1"} : <i3>, <i32>, <i3>, <i32>
    %60 = addi %41#3, %53 {handshake.bb = 2 : ui32, handshake.name = "addi1"} : <i32>
    %61 = cmpi slt, %58#1, %56 {handshake.bb = 2 : ui32, handshake.name = "cmpi1"} : <i32>
    %62:4 = fork [4] %61 {handshake.bb = 2 : ui32, handshake.name = "fork6"} : <i1>
    %63 = buffer %62#3, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer11"} : <i1>
    %64 = init %63 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "init5", initToken = 0 : ui1} : <i1>
    %65:2 = fork [2] %64 {handshake.bb = 2 : ui32, handshake.name = "fork8"} : <i1>
    %66 = not %62#0 {handshake.bb = 2 : ui32, handshake.name = "not0"} : <i1>
    %67 = buffer %60, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer10"} : <i32>
    %68 = passer %67[%62#1] {handshake.bb = 2 : ui32, handshake.name = "passer0"} : <i32>, <i1>
    %69 = passer %48#2[%62#2] {handshake.bb = 2 : ui32, handshake.name = "passer1"} : <>, <i1>
    %70 = passer %48#1[%66] {handshake.bb = 2 : ui32, handshake.name = "passer2"} : <>, <i1>
    %71:4 = fork [4] %70 {handshake.bb = 3 : ui32, handshake.name = "fork9"} : <>
    end {handshake.bb = 3 : ui32, handshake.name = "end0"} %memEnd_3, %memEnd_1, %memEnd_0, %memEnd, %0#1 : <>, <>, <>, <>, <>
  }
}

