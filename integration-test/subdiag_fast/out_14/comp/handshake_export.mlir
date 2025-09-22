module {
  handshake.func @subdiag_fast(%arg0: memref<1000xf32>, %arg1: memref<1000xf32>, %arg2: memref<1000xf32>, %arg3: !handshake.control<>, %arg4: !handshake.control<>, %arg5: !handshake.control<>, %arg6: !handshake.control<>, ...) -> (!handshake.channel<i32>, !handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["d1", "d2", "e", "d1_start", "d2_start", "e_start", "start"], handshake.cfdfcThroughput = #handshake<cfdfcThroughput {"0" = 1.000000e+00 : f64}>, handshake.cfdfcToBBList = #handshake<cfdfcToBBList {"0" = [1 : ui32]}>, resNames = ["out0", "d1_end", "d2_end", "e_end", "end"]} {
    %0:3 = fork [3] %arg6 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork0"} : <>
    %outputs, %memEnd = mem_controller[%arg2 : memref<1000xf32>] %arg5 (%addressResult_6) %154#2 {connectedBlocks = [1 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller3"} :    (!handshake.channel<i10>) -> !handshake.channel<f32>
    %outputs_0, %memEnd_1 = mem_controller[%arg1 : memref<1000xf32>] %arg4 (%addressResult_4) %154#1 {connectedBlocks = [1 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller4"} :    (!handshake.channel<i10>) -> !handshake.channel<f32>
    %outputs_2, %memEnd_3 = mem_controller[%arg0 : memref<1000xf32>] %arg3 (%addressResult) %154#0 {connectedBlocks = [1 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller5"} :    (!handshake.channel<i10>) -> !handshake.channel<f32>
    %1 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant0", value = false} : <>, <i1>
    %2 = extsi %1 {handshake.bb = 0 : ui32, handshake.name = "extsi3"} : <i1> to <i11>
    %3 = mux %129#0 [%2, %144] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i11>, <i11>] to <i11>
    %4 = trunci %143#0 {handshake.bb = 1 : ui32, handshake.name = "trunci0"} : <i11> to <i10>
    %5 = trunci %143#1 {handshake.bb = 1 : ui32, handshake.name = "trunci1"} : <i11> to <i10>
    %6 = trunci %143#2 {handshake.bb = 1 : ui32, handshake.name = "trunci2"} : <i11> to <i10>
    %7 = mux %129#1 [%0#2, %146] {handshake.bb = 1 : ui32, handshake.name = "mux3"} : <i1>, [<>, <>] to <>
    %8 = source {handshake.bb = 1 : ui32, handshake.name = "source0"} : <>
    %9 = constant %8 {handshake.bb = 1 : ui32, handshake.name = "constant4", value = 1.000000e-03 : f32} : <>, <f32>
    %addressResult, %dataResult = load[%6] %outputs_2 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i10>, <f32>, <i10>, <f32>
    %addressResult_4, %dataResult_5 = load[%5] %outputs_0 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i10>, <f32>, <i10>, <f32>
    %10 = addf %dataResult, %dataResult_5 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "addf0", internal_delay = "2_922000"} : <f32>
    %11 = buffer %4, bufferType = FIFO_BREAK_NONE, numSlots = 13 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer1"} : <i10>
    %addressResult_6, %dataResult_7 = load[%11] %outputs {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load2"} : <i10>, <f32>, <i10>, <f32>
    %12 = mulf %10, %9 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "mulf0", internal_delay = "2_875333"} : <f32>
    %13:3 = fork [3] %14 {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <i1>
    %14 = cmpf ugt, %dataResult_7, %12 {handshake.bb = 1 : ui32, handshake.name = "cmpf0", internal_delay = "0_000000"} : <f32>
    %15 = andi %139, %13#1 {handshake.bb = 1 : ui32, handshake.name = "andi0"} : <i1>
    %16 = not %13#0 {handshake.bb = 1 : ui32, handshake.name = "not0"} : <i1>
    %17 = passer %19[%136#1] {handshake.bb = 1 : ui32, handshake.name = "passer8"} : <i12>, <i1>
    %18 = buffer %143#3, bufferType = FIFO_BREAK_NONE, numSlots = 14 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer35"} : <i11>
    %19 = extsi %18 {handshake.bb = 1 : ui32, handshake.name = "extsi4"} : <i11> to <i12>
    %20 = buffer %149#2, bufferType = FIFO_BREAK_NONE, numSlots = 13 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer37"} : <>
    %21 = passer %20[%136#0] {handshake.bb = 1 : ui32, handshake.name = "passer3"} : <>, <i1>
    %22 = extsi %143#4 {handshake.bb = 1 : ui32, handshake.name = "extsi5"} : <i11> to <i12>
    %23 = source {handshake.bb = 1 : ui32, handshake.name = "source1"} : <>
    %24 = constant %23 {handshake.bb = 1 : ui32, handshake.name = "constant1", value = 1 : i2} : <>, <i2>
    %25 = extsi %24 {handshake.bb = 1 : ui32, handshake.name = "extsi6"} : <i2> to <i12>
    %26 = source {handshake.bb = 1 : ui32, handshake.name = "source2"} : <>
    %27 = constant %26 {handshake.bb = 1 : ui32, handshake.name = "constant2", value = 999 : i11} : <>, <i11>
    %28 = extsi %27 {handshake.bb = 1 : ui32, handshake.name = "extsi7"} : <i11> to <i12>
    %29 = buffer %31, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer5"} : <i12>
    %30:3 = fork [3] %29 {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <i12>
    %31 = addi %22, %25 {handshake.bb = 1 : ui32, handshake.name = "addi0"} : <i12>
    %32 = buffer %34, bufferType = FIFO_BREAK_NONE, numSlots = 13 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer6"} : <i1>
    %33:2 = fork [2] %32 {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <i1>
    %34 = cmpi ult, %30#0, %28 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i12>
    %35 = passer %36[%134#2] {handshake.bb = 1 : ui32, handshake.name = "passer9"} : <i1>, <i1>
    %36 = andi %13#2, %33#0 {handshake.bb = 1 : ui32, handshake.name = "andi1"} : <i1>
    %37 = spec_v2_repeating_init %35 {handshake.bb = 1 : ui32, handshake.name = "spec_v2_repeating_init0", initToken = 1 : ui1} : <i1>
    %38 = buffer %37, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer7"} : <i1>
    %39:2 = fork [2] %38 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork4"} : <i1>
    %40 = spec_v2_repeating_init %39#0 {handshake.bb = 1 : ui32, handshake.name = "spec_v2_repeating_init1", initToken = 1 : ui1} : <i1>
    %41 = buffer %40, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer8"} : <i1>
    %42:2 = fork [2] %41 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork15"} : <i1>
    %43 = source {handshake.bb = 1 : ui32, handshake.name = "source3"} : <>
    %44 = constant %43 {handshake.bb = 1 : ui32, handshake.name = "constant3", value = false} : <>, <i1>
    %45 = mux %42#0 [%44, %39#1] {handshake.bb = 1 : ui32, handshake.name = "mux2"} : <i1>, [<i1>, <i1>] to <i1>
    %46 = spec_v2_repeating_init %42#1 {handshake.bb = 1 : ui32, handshake.name = "spec_v2_repeating_init2", initToken = 1 : ui1} : <i1>
    %47 = buffer %46, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer9"} : <i1>
    %48:2 = fork [2] %47 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork16"} : <i1>
    %49 = buffer %48#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer10"} : <i1>
    %50 = source {handshake.bb = 1 : ui32, handshake.name = "source4"} : <>
    %51 = constant %50 {handshake.bb = 1 : ui32, handshake.name = "constant5", value = false} : <>, <i1>
    %52 = mux %49 [%51, %45] {handshake.bb = 1 : ui32, handshake.name = "mux4"} : <i1>, [<i1>, <i1>] to <i1>
    %53 = spec_v2_repeating_init %48#1 {handshake.bb = 1 : ui32, handshake.name = "spec_v2_repeating_init3", initToken = 1 : ui1} : <i1>
    %54 = buffer %53, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer11"} : <i1>
    %55:2 = fork [2] %54 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork17"} : <i1>
    %56 = buffer %55#0, bufferType = FIFO_BREAK_NONE, numSlots = 2 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer12"} : <i1>
    %57 = source {handshake.bb = 1 : ui32, handshake.name = "source5"} : <>
    %58 = constant %57 {handshake.bb = 1 : ui32, handshake.name = "constant6", value = false} : <>, <i1>
    %59 = mux %56 [%58, %52] {handshake.bb = 1 : ui32, handshake.name = "mux5"} : <i1>, [<i1>, <i1>] to <i1>
    %60 = spec_v2_repeating_init %55#1 {handshake.bb = 1 : ui32, handshake.name = "spec_v2_repeating_init4", initToken = 1 : ui1} : <i1>
    %61 = buffer %60, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer13"} : <i1>
    %62:2 = fork [2] %61 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork18"} : <i1>
    %63 = buffer %62#0, bufferType = FIFO_BREAK_NONE, numSlots = 3 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer14"} : <i1>
    %64 = source {handshake.bb = 1 : ui32, handshake.name = "source6"} : <>
    %65 = constant %64 {handshake.bb = 1 : ui32, handshake.name = "constant7", value = false} : <>, <i1>
    %66 = mux %63 [%65, %59] {handshake.bb = 1 : ui32, handshake.name = "mux6"} : <i1>, [<i1>, <i1>] to <i1>
    %67 = spec_v2_repeating_init %62#1 {handshake.bb = 1 : ui32, handshake.name = "spec_v2_repeating_init5", initToken = 1 : ui1} : <i1>
    %68 = buffer %67, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer15"} : <i1>
    %69:2 = fork [2] %68 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork19"} : <i1>
    %70 = buffer %69#0, bufferType = FIFO_BREAK_NONE, numSlots = 4 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer16"} : <i1>
    %71 = source {handshake.bb = 1 : ui32, handshake.name = "source7"} : <>
    %72 = constant %71 {handshake.bb = 1 : ui32, handshake.name = "constant8", value = false} : <>, <i1>
    %73 = mux %70 [%72, %66] {handshake.bb = 1 : ui32, handshake.name = "mux7"} : <i1>, [<i1>, <i1>] to <i1>
    %74 = spec_v2_repeating_init %69#1 {handshake.bb = 1 : ui32, handshake.name = "spec_v2_repeating_init6", initToken = 1 : ui1} : <i1>
    %75 = buffer %74, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer17"} : <i1>
    %76:2 = fork [2] %75 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork20"} : <i1>
    %77 = buffer %76#0, bufferType = FIFO_BREAK_NONE, numSlots = 5 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer18"} : <i1>
    %78 = source {handshake.bb = 1 : ui32, handshake.name = "source8"} : <>
    %79 = constant %78 {handshake.bb = 1 : ui32, handshake.name = "constant9", value = false} : <>, <i1>
    %80 = mux %77 [%79, %73] {handshake.bb = 1 : ui32, handshake.name = "mux8"} : <i1>, [<i1>, <i1>] to <i1>
    %81 = spec_v2_repeating_init %76#1 {handshake.bb = 1 : ui32, handshake.name = "spec_v2_repeating_init7", initToken = 1 : ui1} : <i1>
    %82 = buffer %81, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer19"} : <i1>
    %83:2 = fork [2] %82 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork21"} : <i1>
    %84 = buffer %83#0, bufferType = FIFO_BREAK_NONE, numSlots = 6 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer20"} : <i1>
    %85 = source {handshake.bb = 1 : ui32, handshake.name = "source9"} : <>
    %86 = constant %85 {handshake.bb = 1 : ui32, handshake.name = "constant10", value = false} : <>, <i1>
    %87 = mux %84 [%86, %80] {handshake.bb = 1 : ui32, handshake.name = "mux9"} : <i1>, [<i1>, <i1>] to <i1>
    %88 = spec_v2_repeating_init %83#1 {handshake.bb = 1 : ui32, handshake.name = "spec_v2_repeating_init8", initToken = 1 : ui1} : <i1>
    %89 = buffer %88, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer21"} : <i1>
    %90:2 = fork [2] %89 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork22"} : <i1>
    %91 = buffer %90#0, bufferType = FIFO_BREAK_NONE, numSlots = 7 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer22"} : <i1>
    %92 = source {handshake.bb = 1 : ui32, handshake.name = "source10"} : <>
    %93 = constant %92 {handshake.bb = 1 : ui32, handshake.name = "constant11", value = false} : <>, <i1>
    %94 = mux %91 [%93, %87] {handshake.bb = 1 : ui32, handshake.name = "mux10"} : <i1>, [<i1>, <i1>] to <i1>
    %95 = spec_v2_repeating_init %90#1 {handshake.bb = 1 : ui32, handshake.name = "spec_v2_repeating_init9", initToken = 1 : ui1} : <i1>
    %96 = buffer %95, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer23"} : <i1>
    %97:2 = fork [2] %96 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork23"} : <i1>
    %98 = buffer %97#0, bufferType = FIFO_BREAK_NONE, numSlots = 8 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer24"} : <i1>
    %99 = source {handshake.bb = 1 : ui32, handshake.name = "source11"} : <>
    %100 = constant %99 {handshake.bb = 1 : ui32, handshake.name = "constant12", value = false} : <>, <i1>
    %101 = mux %98 [%100, %94] {handshake.bb = 1 : ui32, handshake.name = "mux11"} : <i1>, [<i1>, <i1>] to <i1>
    %102 = spec_v2_repeating_init %97#1 {handshake.bb = 1 : ui32, handshake.name = "spec_v2_repeating_init10", initToken = 1 : ui1} : <i1>
    %103 = buffer %102, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer25"} : <i1>
    %104:2 = fork [2] %103 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork24"} : <i1>
    %105 = buffer %104#0, bufferType = FIFO_BREAK_NONE, numSlots = 9 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer26"} : <i1>
    %106 = source {handshake.bb = 1 : ui32, handshake.name = "source12"} : <>
    %107 = constant %106 {handshake.bb = 1 : ui32, handshake.name = "constant13", value = false} : <>, <i1>
    %108 = mux %105 [%107, %101] {handshake.bb = 1 : ui32, handshake.name = "mux12"} : <i1>, [<i1>, <i1>] to <i1>
    %109 = spec_v2_repeating_init %104#1 {handshake.bb = 1 : ui32, handshake.name = "spec_v2_repeating_init11", initToken = 1 : ui1} : <i1>
    %110 = buffer %109, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer28"} : <i1>
    %111:2 = fork [2] %110 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork25"} : <i1>
    %112 = buffer %108, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer27"} : <i1>
    %113 = buffer %111#0, bufferType = FIFO_BREAK_NONE, numSlots = 11 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer29"} : <i1>
    %114 = source {handshake.bb = 1 : ui32, handshake.name = "source13"} : <>
    %115 = constant %114 {handshake.bb = 1 : ui32, handshake.name = "constant14", value = false} : <>, <i1>
    %116 = mux %113 [%115, %112] {handshake.bb = 1 : ui32, handshake.name = "mux13"} : <i1>, [<i1>, <i1>] to <i1>
    %117 = spec_v2_repeating_init %111#1 {handshake.bb = 1 : ui32, handshake.name = "spec_v2_repeating_init12", initToken = 1 : ui1} : <i1>
    %118 = buffer %117, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer30"} : <i1>
    %119:2 = fork [2] %118 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork26"} : <i1>
    %120 = buffer %119#0, bufferType = FIFO_BREAK_NONE, numSlots = 12 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer31"} : <i1>
    %121 = source {handshake.bb = 1 : ui32, handshake.name = "source14"} : <>
    %122 = constant %121 {handshake.bb = 1 : ui32, handshake.name = "constant15", value = false} : <>, <i1>
    %123 = mux %120 [%122, %116] {handshake.bb = 1 : ui32, handshake.name = "mux14"} : <i1>, [<i1>, <i1>] to <i1>
    %124 = spec_v2_repeating_init %119#1 {handshake.bb = 1 : ui32, handshake.name = "spec_v2_repeating_init13", initToken = 1 : ui1} : <i1>
    %125 = buffer %124, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer32"} : <i1>
    %126:4 = fork [4] %125 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork6"} : <i1>
    %127 = buffer %126#3, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer34"} : <i1>
    %128 = init %127 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "init2", initToken = 0 : ui1} : <i1>
    %129:2 = fork [2] %128 {handshake.bb = 1 : ui32, handshake.name = "fork7"} : <i1>
    %130 = buffer %126#2, bufferType = FIFO_BREAK_NONE, numSlots = 13 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer33"} : <i1>
    %131 = source {handshake.bb = 1 : ui32, handshake.name = "source15"} : <>
    %132 = constant %131 {handshake.bb = 1 : ui32, handshake.name = "constant16", value = false} : <>, <i1>
    %133 = mux %130 [%132, %123] {handshake.bb = 1 : ui32, handshake.name = "mux15"} : <i1>, [<i1>, <i1>] to <i1>
    %134:3 = fork [3] %133 {handshake.bb = 1 : ui32, handshake.name = "fork28"} : <i1>
    %135 = andi %16, %134#0 {handshake.bb = 1 : ui32, handshake.name = "andi15"} : <i1>
    %136:2 = fork [2] %135 {handshake.bb = 1 : ui32, handshake.name = "fork29"} : <i1>
    %137 = andi %15, %134#1 {handshake.bb = 1 : ui32, handshake.name = "andi16"} : <i1>
    %138:2 = fork [2] %137 {handshake.bb = 1 : ui32, handshake.name = "fork30"} : <i1>
    %139 = not %33#1 {handshake.bb = 1 : ui32, handshake.name = "not1"} : <i1>
    %140 = buffer %30#1, bufferType = FIFO_BREAK_NONE, numSlots = 13 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer4"} : <i12>
    %141 = passer %140[%138#0] {handshake.bb = 1 : ui32, handshake.name = "passer5"} : <i12>, <i1>
    %142 = buffer %3, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer0"} : <i11>
    %143:5 = fork [5] %142 {handshake.bb = 1 : ui32, handshake.name = "fork31"} : <i11>
    %144 = passer %145[%126#1] {handshake.bb = 1 : ui32, handshake.name = "passer10"} : <i11>, <i1>
    %145 = trunci %30#2 {handshake.bb = 1 : ui32, handshake.name = "trunci3"} : <i12> to <i11>
    %146 = passer %149#1[%126#0] {handshake.bb = 1 : ui32, handshake.name = "passer11"} : <>, <i1>
    %147 = buffer %7, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer2"} : <>
    %148 = buffer %147, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer3"} : <>
    %149:3 = fork [3] %148 {handshake.bb = 1 : ui32, handshake.name = "fork32"} : <>
    %150 = buffer %149#0, bufferType = FIFO_BREAK_NONE, numSlots = 13 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer36"} : <>
    %151 = passer %150[%138#1] {handshake.bb = 1 : ui32, handshake.name = "passer7"} : <>, <i1>
    %152 = mux %index [%17, %141] {handshake.bb = 2 : ui32, handshake.name = "mux1"} : <i1>, [<i12>, <i12>] to <i12>
    %153 = extsi %152 {handshake.bb = 2 : ui32, handshake.name = "extsi8"} : <i12> to <i32>
    %result, %index = control_merge [%21, %151]  {handshake.bb = 2 : ui32, handshake.name = "control_merge2"} : [<>, <>] to <>, <i1>
    %154:3 = fork [3] %result {handshake.bb = 2 : ui32, handshake.name = "fork5"} : <>
    end {handshake.bb = 2 : ui32, handshake.name = "end0"} %153, %memEnd_3, %memEnd_1, %memEnd, %0#1 : <i32>, <>, <>, <>, <>
  }
}

