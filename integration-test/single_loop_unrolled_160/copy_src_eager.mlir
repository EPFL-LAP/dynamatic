module {
  handshake.func @single_loop_unrolled(%arg0: memref<7xi32>, %arg1: memref<7xi32>, %arg2: memref<7xi32>, %arg3: memref<7xi32>, %arg4: !handshake.control<>, %arg5: !handshake.control<>, %arg6: !handshake.control<>, %arg7: !handshake.control<>, %arg8: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["a", "b", "c0", "c1", "a_start", "b_start", "c0_start", "c1_start", "start"], handshake.cfdfcThroughput = #handshake<cfdfcThroughput {"0" = 1.000000e+00 : f64, "1" = 1.000000e+00 : f64}>, handshake.cfdfcToBBList = #handshake<cfdfcToBBList {"0" = [2 : ui32], "1" = [1 : ui32]}>, resNames = ["a_end", "b_end", "c0_end", "c1_end", "end"]} {
    %0:3 = fork [3] %arg8 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork0"} : <>
    %memEnd = mem_controller[%arg3 : memref<7xi32>] %arg7 (%71, %addressResult_12, %dataResult_13) %116#3 {connectedBlocks = [2 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller4"} :    (!handshake.channel<i32>, !handshake.channel<i3>, !handshake.channel<i32>) -> ()
    %memEnd_0 = mem_controller[%arg2 : memref<7xi32>] %arg6 (%12, %addressResult_6, %dataResult_7) %116#2 {connectedBlocks = [1 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller5"} :    (!handshake.channel<i32>, !handshake.channel<i3>, !handshake.channel<i32>) -> ()
    %outputs:2, %memEnd_1 = mem_controller[%arg1 : memref<7xi32>] %arg5 (%addressResult_4, %addressResult_10) %116#1 {connectedBlocks = [1 : i32, 2 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller6"} :    (!handshake.channel<i3>, !handshake.channel<i3>) -> (!handshake.channel<i32>, !handshake.channel<i32>)
    %outputs_2:2, %memEnd_3 = mem_controller[%arg0 : memref<7xi32>] %arg4 (%addressResult, %addressResult_8) %116#0 {connectedBlocks = [1 : i32, 2 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller7"} :    (!handshake.channel<i3>, !handshake.channel<i3>) -> (!handshake.channel<i32>, !handshake.channel<i32>)
    %1 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant10", value = false} : <>, <i1>
    %2 = extsi %1 {handshake.bb = 0 : ui32, handshake.name = "extsi8"} : <i1> to <i32>
    %3 = mux %48#0 [%2, %27] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i32>, <i32>] to <i32>
    %4 = buffer %6, bufferType = FIFO_BREAK_NONE, numSlots = 5 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer1"} : <i3>
    %5 = passer %4[%50#3] {handshake.bb = 1 : ui32, handshake.name = "passer0", specv2_frontier = false} : <i3>, <i1>
    %6 = trunci %25#0 {handshake.bb = 1 : ui32, handshake.name = "trunci0"} : <i32> to <i3>
    %7 = trunci %25#1 {handshake.bb = 1 : ui32, handshake.name = "trunci1"} : <i32> to <i3>
    %8 = trunci %25#2 {handshake.bb = 1 : ui32, handshake.name = "trunci2"} : <i32> to <i3>
    %9 = buffer %48#1, bufferType = FIFO_BREAK_NONE, numSlots = 4 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer15"} : <i1>
    %10 = mux %9 [%0#2, %55] {handshake.bb = 1 : ui32, handshake.name = "mux3"} : <i1>, [<>, <>] to <>
    %11 = constant %58#1 {handshake.bb = 1 : ui32, handshake.name = "constant11", value = 1 : i2} : <>, <i2>
    %12 = passer %13[%50#4] {handshake.bb = 1 : ui32, handshake.name = "passer1", specv2_frontier = false} : <i32>, <i1>
    %13 = extsi %11 {handshake.bb = 1 : ui32, handshake.name = "extsi1"} : <i2> to <i32>
    %14 = constant %58#0 {handshake.bb = 1 : ui32, handshake.name = "constant12", value = false} : <>, <i1>
    %15 = source {handshake.bb = 1 : ui32, handshake.name = "source0"} : <>
    %16 = constant %15 {handshake.bb = 1 : ui32, handshake.name = "constant13", value = 1 : i2} : <>, <i2>
    %17 = extsi %16 {handshake.bb = 1 : ui32, handshake.name = "extsi3"} : <i2> to <i32>
    %18 = source {handshake.bb = 1 : ui32, handshake.name = "source1"} : <>
    %19 = constant %18 {handshake.bb = 1 : ui32, handshake.name = "constant14", value = 7 : i4} : <>, <i4>
    %20 = extsi %19 {handshake.bb = 1 : ui32, handshake.name = "extsi4"} : <i4> to <i32>
    %addressResult, %dataResult = load[%8] %outputs_2#0 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i3>, <i32>, <i3>, <i32>
    %addressResult_4, %dataResult_5 = load[%7] %outputs#0 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i3>, <i32>, <i3>, <i32>
    %21 = passer %22#0[%50#2] {handshake.bb = 1 : ui32, handshake.name = "passer13", specv2_frontier = false} : <i32>, <i1>
    %22:2 = fork [2] %23 {handshake.bb = 1 : ui32, handshake.name = "fork5"} : <i32>
    %23 = muli %dataResult, %dataResult_5 {handshake.bb = 1 : ui32, handshake.name = "muli0"} : <i32>
    %addressResult_6, %dataResult_7 = store[%5] %21 {handshake.bb = 1 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store0"} : <i3>, <i32>, <i3>, <i32>
    %24 = buffer %3, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer0"} : <i32>
    %25:4 = fork [4] %24 {handshake.bb = 1 : ui32, handshake.name = "fork7"} : <i32>
    %26 = buffer %28, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer4"} : <i32>
    %27 = passer %26[%45#0] {handshake.bb = 1 : ui32, handshake.name = "passer14", specv2_frontier = false} : <i32>, <i1>
    %28 = addi %25#3, %17 {handshake.bb = 1 : ui32, handshake.name = "addi0"} : <i32>
    %29 = passer %30#1[%50#1] {handshake.bb = 1 : ui32, handshake.name = "passer15", specv2_frontier = false} : <i1>, <i1>
    %30:2 = fork [2] %31 {handshake.bb = 1 : ui32, handshake.name = "fork8"} : <i1>
    %31 = cmpi slt, %22#1, %20 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i32>
    %32 = spec_v2_repeating_init %29 {handshake.bb = 1 : ui32, handshake.name = "spec_v2_repeating_init4", initToken = 1 : ui1, specv2_top_ri = true} : <i1>
    %33 = buffer %32, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer5"} : <i1>
    %34 = buffer %33, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer6"} : <i1>
    %35:2 = fork [2] %34 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork25"} : <i1>
    %36 = buffer %45#2, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer12"} : <i1>
    %37 = buffer %36, bufferType = FIFO_BREAK_NONE, numSlots = 2 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer13"} : <i1>
    %38 = spec_v2_interpolator %35#1, %37 {handshake.bb = 1 : ui32, handshake.name = "spec_v2_interpolator1"} : <i1>
    %39 = spec_v2_repeating_init %35#0 {handshake.bb = 1 : ui32, handshake.name = "spec_v2_repeating_init5", initToken = 1 : ui1} : <i1>
    %40 = buffer %39, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer8"} : <i1>
    %41 = spec_v2_repeating_init %40 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "spec_v2_repeating_init6", initToken = 1 : ui1} : <i1>
    %42 = buffer %41, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer9"} : <i1>
    %43 = spec_v2_repeating_init %42 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "spec_v2_repeating_init7", initToken = 1 : ui1} : <i1>
    %44 = buffer %43, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer10"} : <i1>
    %45:4 = fork [4] %44 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork1"} : <i1>
    %46 = buffer %45#3, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer14"} : <i1>
    %47 = init %46 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "init4", initToken = 0 : ui1} : <i1>
    %48:2 = fork [2] %47 {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <i1>
    %49 = buffer %38, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer7"} : <i1>
    %50:5 = fork [5] %49 {handshake.bb = 1 : ui32, handshake.name = "fork19"} : <i1>
    %51 = andi %53, %50#0 {handshake.bb = 1 : ui32, handshake.name = "andi1"} : <i1>
    %52:2 = fork [2] %51 {handshake.bb = 1 : ui32, handshake.name = "fork20"} : <i1>
    %53 = not %30#0 {handshake.bb = 1 : ui32, handshake.name = "not1"} : <i1>
    %54 = buffer %45#1, bufferType = FIFO_BREAK_NONE, numSlots = 4 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer11"} : <i1>
    %55 = passer %58#3[%54] {handshake.bb = 1 : ui32, handshake.name = "passer16", specv2_frontier = false} : <>, <i1>
    %56 = buffer %10, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer2"} : <>
    %57 = buffer %56, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer3"} : <>
    %58:4 = fork [4] %57 {handshake.bb = 1 : ui32, handshake.name = "fork24"} : <>
    %59 = passer %58#2[%52#1] {handshake.bb = 1 : ui32, handshake.name = "passer5", specv2_frontier = false} : <>, <i1>
    %60 = passer %61[%52#0] {handshake.bb = 1 : ui32, handshake.name = "passer17", specv2_frontier = false} : <i32>, <i1>
    %61 = extsi %14 {handshake.bb = 1 : ui32, handshake.name = "extsi9"} : <i1> to <i32>
    %break_dv_1 = buffer %60, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer_man1"} : <i32>
    %break_r_1 = buffer %break_dv_1, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer_man2"} : <i32>
    %62 = mux %105#0 [%break_r_1, %85] {handshake.bb = 2 : ui32, handshake.name = "mux1"} : <i1>, [<i32>, <i32>] to <i32>
    %63 = passer %65[%107#3] {handshake.bb = 2 : ui32, handshake.name = "passer7", specv2_frontier = false} : <i3>, <i1>
    %64 = buffer %83#0, bufferType = FIFO_BREAK_NONE, numSlots = 5 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer19"} : <i32>
    %65 = trunci %64 {handshake.bb = 2 : ui32, handshake.name = "trunci3"} : <i32> to <i3>
    %66 = trunci %83#1 {handshake.bb = 2 : ui32, handshake.name = "trunci4"} : <i32> to <i3>
    %67 = trunci %83#2 {handshake.bb = 2 : ui32, handshake.name = "trunci5"} : <i32> to <i3>
    %break_dv_2 = buffer %59, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer_man3"} : <>
    %break_r_2 = buffer %break_dv_2, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer_man4"} : <>
    %68 = mux %105#1 [%break_r_2, %110] {handshake.bb = 2 : ui32, handshake.name = "mux2"} : <i1>, [<>, <>] to <>
    %69 = buffer %113#0, bufferType = FIFO_BREAK_NONE, numSlots = 4 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer29"} : <>
    %70 = constant %69 {handshake.bb = 2 : ui32, handshake.name = "constant15", value = 1 : i2} : <>, <i2>
    %71 = passer %72[%107#4] {handshake.bb = 2 : ui32, handshake.name = "passer8", specv2_frontier = false} : <i32>, <i1>
    %72 = extsi %70 {handshake.bb = 2 : ui32, handshake.name = "extsi5"} : <i2> to <i32>
    %73 = source {handshake.bb = 2 : ui32, handshake.name = "source2"} : <>
    %74 = constant %73 {handshake.bb = 2 : ui32, handshake.name = "constant16", value = 1 : i2} : <>, <i2>
    %75 = extsi %74 {handshake.bb = 2 : ui32, handshake.name = "extsi6"} : <i2> to <i32>
    %76 = source {handshake.bb = 2 : ui32, handshake.name = "source3"} : <>
    %77 = constant %76 {handshake.bb = 2 : ui32, handshake.name = "constant17", value = 7 : i4} : <>, <i4>
    %78 = extsi %77 {handshake.bb = 2 : ui32, handshake.name = "extsi7"} : <i4> to <i32>
    %addressResult_8, %dataResult_9 = load[%67] %outputs_2#1 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load2"} : <i3>, <i32>, <i3>, <i32>
    %addressResult_10, %dataResult_11 = load[%66] %outputs#1 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load3"} : <i3>, <i32>, <i3>, <i32>
    %79 = passer %80#0[%107#2] {handshake.bb = 2 : ui32, handshake.name = "passer9", specv2_frontier = false} : <i32>, <i1>
    %80:2 = fork [2] %81 {handshake.bb = 2 : ui32, handshake.name = "fork2"} : <i32>
    %81 = muli %dataResult_9, %dataResult_11 {handshake.bb = 2 : ui32, handshake.name = "muli1"} : <i32>
    %addressResult_12, %dataResult_13 = store[%63] %79 {handshake.bb = 2 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store1"} : <i3>, <i32>, <i3>, <i32>
    %82 = buffer %62, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer16"} : <i32>
    %83:4 = fork [4] %82 {handshake.bb = 2 : ui32, handshake.name = "fork4"} : <i32>
    %84 = buffer %86, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer20"} : <i32>
    %85 = passer %84[%102#0] {handshake.bb = 2 : ui32, handshake.name = "passer10", specv2_frontier = false} : <i32>, <i1>
    %86 = addi %83#3, %75 {handshake.bb = 2 : ui32, handshake.name = "addi1"} : <i32>
    %87 = passer %88#1[%107#0] {handshake.bb = 2 : ui32, handshake.name = "passer11", specv2_frontier = false} : <i1>, <i1>
    %88:2 = fork [2] %89 {handshake.bb = 2 : ui32, handshake.name = "fork6"} : <i1>
    %89 = cmpi slt, %80#1, %78 {handshake.bb = 2 : ui32, handshake.name = "cmpi1"} : <i32>
    %90 = spec_v2_repeating_init %87 {handshake.bb = 2 : ui32, handshake.name = "spec_v2_repeating_init0", initToken = 1 : ui1, specv2_top_ri = true} : <i1>
    %91 = buffer %90, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer21"} : <i1>
    %92 = buffer %91, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer22"} : <i1>
    %93:2 = fork [2] %92 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork22"} : <i1>
    %94 = buffer %102#2, bufferType = FIFO_BREAK_NONE, numSlots = 3 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer27"} : <i1>
    %95 = spec_v2_interpolator %93#1, %94 {handshake.bb = 2 : ui32, handshake.name = "spec_v2_interpolator0"} : <i1>
    %96 = spec_v2_repeating_init %93#0 {handshake.bb = 2 : ui32, handshake.name = "spec_v2_repeating_init1", initToken = 1 : ui1} : <i1>
    %97 = buffer %96, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer24"} : <i1>
    %98 = spec_v2_repeating_init %97 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "spec_v2_repeating_init2", initToken = 1 : ui1} : <i1>
    %99 = buffer %98, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer25"} : <i1>
    %100 = spec_v2_repeating_init %99 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "spec_v2_repeating_init3", initToken = 1 : ui1} : <i1>
    %101 = buffer %100, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer26"} : <i1>
    %102:4 = fork [4] %101 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork10"} : <i1>
    %103 = buffer %102#3, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer28"} : <i1>
    %104 = init %103 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "init5", initToken = 0 : ui1} : <i1>
    %105:2 = fork [2] %104 {handshake.bb = 2 : ui32, handshake.name = "fork11"} : <i1>
    %106 = buffer %95, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer23"} : <i1>
    %107:5 = fork [5] %106 {handshake.bb = 2 : ui32, handshake.name = "fork23"} : <i1>
    %108 = andi %109, %107#1 {handshake.bb = 2 : ui32, handshake.name = "andi0"} : <i1>
    %109 = not %88#0 {handshake.bb = 2 : ui32, handshake.name = "not0"} : <i1>
    %110 = passer %113#2[%102#1] {handshake.bb = 2 : ui32, handshake.name = "passer12", specv2_frontier = false} : <>, <i1>
    %111 = buffer %68, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer17"} : <>
    %112 = buffer %111, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer18"} : <>
    %113:3 = fork [3] %112 {handshake.bb = 2 : ui32, handshake.name = "fork21"} : <>
    %114 = buffer %113#1, bufferType = FIFO_BREAK_NONE, numSlots = 4 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "buffer30"} : <>
    %115 = passer %114[%108] {handshake.bb = 2 : ui32, handshake.name = "passer2", specv2_frontier = false} : <>, <i1>
    %116:4 = fork [4] %115 {handshake.bb = 3 : ui32, handshake.name = "fork9"} : <>
    end {handshake.bb = 3 : ui32, handshake.name = "end0"} %memEnd_3, %memEnd_1, %memEnd_0, %memEnd, %0#1 : <>, <>, <>, <>, <>
  }
}

