module {
  handshake.func @subdiag_fast(%arg0: memref<1000xf32>, %arg1: memref<1000xf32>, %arg2: memref<1000xf32>, %arg3: !handshake.control<>, %arg4: !handshake.control<>, %arg5: !handshake.control<>, %arg6: !handshake.control<>, ...) -> (!handshake.channel<i32>, !handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["d1", "d2", "e", "d1_start", "d2_start", "e_start", "start"], handshake.cfdfcThroughput = #handshake<cfdfcThroughput {"0" = 0.9285714285714286 : f64}>, handshake.cfdfcToBBList = #handshake<cfdfcToBBList {"0" = [1 : ui32]}>, resNames = ["out0", "d1_end", "d2_end", "e_end", "end"]} {
    %0:3 = fork [3] %arg6 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork0"} : <>
    %outputs, %memEnd = mem_controller[%arg2 : memref<1000xf32>] %arg5 (%addressResult_6) %121#2 {connectedBlocks = [1 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller3"} :    (!handshake.channel<i10>) -> !handshake.channel<f32>
    %outputs_0, %memEnd_1 = mem_controller[%arg1 : memref<1000xf32>] %arg4 (%addressResult_4) %121#1 {connectedBlocks = [1 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller4"} :    (!handshake.channel<i10>) -> !handshake.channel<f32>
    %outputs_2, %memEnd_3 = mem_controller[%arg0 : memref<1000xf32>] %arg3 (%addressResult) %121#0 {connectedBlocks = [1 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller5"} :    (!handshake.channel<i10>) -> !handshake.channel<f32>
    %1 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant0", value = false} : <>, <i1>
    %2 = extsi %1 {handshake.bb = 0 : ui32, handshake.name = "extsi3"} : <i1> to <i11>
    %3 = mux %96#0 [%2, %111] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i11>, <i11>] to <i11>
    %4 = buffer %110#0, bufferType = FIFO_BREAK_NONE, numSlots = 13 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer35"} : <i11>
    %5 = trunci %4 {handshake.bb = 1 : ui32, handshake.name = "trunci0"} : <i11> to <i10>
    %6 = trunci %110#1 {handshake.bb = 1 : ui32, handshake.name = "trunci1"} : <i11> to <i10>
    %7 = trunci %110#2 {handshake.bb = 1 : ui32, handshake.name = "trunci2"} : <i11> to <i10>
    %8 = buffer %96#1, bufferType = FIFO_BREAK_NONE, numSlots = 12 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer34"} : <i1>
    %9 = mux %8 [%0#2, %114] {handshake.bb = 1 : ui32, handshake.name = "mux3"} : <i1>, [<>, <>] to <>
    %10 = source {handshake.bb = 1 : ui32, handshake.name = "source0"} : <>
    %11 = constant %10 {handshake.bb = 1 : ui32, handshake.name = "constant4", value = 1.000000e-03 : f32} : <>, <f32>
    %addressResult, %dataResult = load[%7] %outputs_2 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i10>, <f32>, <i10>, <f32>
    %addressResult_4, %dataResult_5 = load[%6] %outputs_0 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i10>, <f32>, <i10>, <f32>
    %12 = addf %dataResult, %dataResult_5 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "addf0", internal_delay = "2_922000"} : <f32>
    %addressResult_6, %dataResult_7 = load[%5] %outputs {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load2"} : <i10>, <f32>, <i10>, <f32>
    %13 = mulf %12, %11 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "mulf0", internal_delay = "2_875333"} : <f32>
    %14:3 = fork [3] %15 {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <i1>
    %15 = cmpf ugt, %dataResult_7, %13 {handshake.bb = 1 : ui32, handshake.name = "cmpf0", internal_delay = "0_000000"} : <f32>
    %16 = andi %106, %14#1 {handshake.bb = 1 : ui32, handshake.name = "andi0"} : <i1>
    %17 = not %14#0 {handshake.bb = 1 : ui32, handshake.name = "not0"} : <i1>
    %18 = passer %20[%103#1] {handshake.bb = 1 : ui32, handshake.name = "passer8"} : <i12>, <i1>
    %19 = buffer %110#3, bufferType = FIFO_BREAK_NONE, numSlots = 13 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer36"} : <i11>
    %20 = extsi %19 {handshake.bb = 1 : ui32, handshake.name = "extsi4"} : <i11> to <i12>
    %21 = passer %117#2[%103#0] {handshake.bb = 1 : ui32, handshake.name = "passer3"} : <>, <i1>
    %22 = extsi %110#4 {handshake.bb = 1 : ui32, handshake.name = "extsi5"} : <i11> to <i12>
    %23 = source {handshake.bb = 1 : ui32, handshake.name = "source1"} : <>
    %24 = constant %23 {handshake.bb = 1 : ui32, handshake.name = "constant1", value = 1 : i2} : <>, <i2>
    %25 = extsi %24 {handshake.bb = 1 : ui32, handshake.name = "extsi6"} : <i2> to <i12>
    %26 = source {handshake.bb = 1 : ui32, handshake.name = "source2"} : <>
    %27 = constant %26 {handshake.bb = 1 : ui32, handshake.name = "constant2", value = 999 : i11} : <>, <i11>
    %28 = extsi %27 {handshake.bb = 1 : ui32, handshake.name = "extsi7"} : <i11> to <i12>
    %29 = buffer %31, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer4"} : <i12>
    %30:3 = fork [3] %29 {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <i12>
    %31 = addi %22, %25 {handshake.bb = 1 : ui32, handshake.name = "addi0"} : <i12>
    %32 = buffer %35, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer5"} : <i1>
    %33 = buffer %32, bufferType = FIFO_BREAK_NONE, numSlots = 11 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer6"} : <i1>
    %34:2 = fork [2] %33 {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <i1>
    %35 = cmpi ult, %30#0, %28 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i12>
    %36 = passer %37[%101#2] {handshake.bb = 1 : ui32, handshake.name = "passer9"} : <i1>, <i1>
    %37 = andi %14#2, %34#0 {handshake.bb = 1 : ui32, handshake.name = "andi1"} : <i1>
    %38 = spec_v2_repeating_init %36 {handshake.bb = 1 : ui32, handshake.name = "spec_v2_repeating_init0", initToken = 1 : ui1} : <i1>
    %39 = buffer %38, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer7"} : <i1>
    %40:2 = fork [2] %39 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork4"} : <i1>
    %41 = spec_v2_repeating_init %40#0 {handshake.bb = 1 : ui32, handshake.name = "spec_v2_repeating_init1", initToken = 1 : ui1} : <i1>
    %42 = buffer %41, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer8"} : <i1>
    %43:2 = fork [2] %42 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork15"} : <i1>
    %44 = buffer %43#0, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer9"} : <i1>
    %45 = andi %40#1, %44 {handshake.bb = 1 : ui32, handshake.name = "andi2", specv2_tmp_and = true} : <i1>
    %46 = spec_v2_repeating_init %43#1 {handshake.bb = 1 : ui32, handshake.name = "spec_v2_repeating_init2", initToken = 1 : ui1} : <i1>
    %47 = buffer %46, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer10"} : <i1>
    %48:2 = fork [2] %47 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork16"} : <i1>
    %49 = buffer %48#0, bufferType = FIFO_BREAK_NONE, numSlots = 2 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer11"} : <i1>
    %50 = andi %45, %49 {handshake.bb = 1 : ui32, handshake.name = "andi3", specv2_tmp_and = true} : <i1>
    %51 = spec_v2_repeating_init %48#1 {handshake.bb = 1 : ui32, handshake.name = "spec_v2_repeating_init3", initToken = 1 : ui1} : <i1>
    %52 = buffer %51, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer12"} : <i1>
    %53:2 = fork [2] %52 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork17"} : <i1>
    %54 = buffer %53#0, bufferType = FIFO_BREAK_NONE, numSlots = 3 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer13"} : <i1>
    %55 = andi %50, %54 {handshake.bb = 1 : ui32, handshake.name = "andi4", specv2_tmp_and = true} : <i1>
    %56 = spec_v2_repeating_init %53#1 {handshake.bb = 1 : ui32, handshake.name = "spec_v2_repeating_init4", initToken = 1 : ui1} : <i1>
    %57 = buffer %56, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer14"} : <i1>
    %58:2 = fork [2] %57 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork18"} : <i1>
    %59 = buffer %58#0, bufferType = FIFO_BREAK_NONE, numSlots = 4 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer15"} : <i1>
    %60 = andi %55, %59 {handshake.bb = 1 : ui32, handshake.name = "andi5", specv2_tmp_and = true} : <i1>
    %61 = spec_v2_repeating_init %58#1 {handshake.bb = 1 : ui32, handshake.name = "spec_v2_repeating_init5", initToken = 1 : ui1} : <i1>
    %62 = buffer %61, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer16"} : <i1>
    %63:2 = fork [2] %62 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork19"} : <i1>
    %64 = buffer %63#0, bufferType = FIFO_BREAK_NONE, numSlots = 5 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer17"} : <i1>
    %65 = andi %60, %64 {handshake.bb = 1 : ui32, handshake.name = "andi6", specv2_tmp_and = true} : <i1>
    %66 = spec_v2_repeating_init %63#1 {handshake.bb = 1 : ui32, handshake.name = "spec_v2_repeating_init6", initToken = 1 : ui1} : <i1>
    %67 = buffer %66, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer18"} : <i1>
    %68:2 = fork [2] %67 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork20"} : <i1>
    %69 = buffer %68#0, bufferType = FIFO_BREAK_NONE, numSlots = 6 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer19"} : <i1>
    %70 = andi %65, %69 {handshake.bb = 1 : ui32, handshake.name = "andi7", specv2_tmp_and = true} : <i1>
    %71 = spec_v2_repeating_init %68#1 {handshake.bb = 1 : ui32, handshake.name = "spec_v2_repeating_init7", initToken = 1 : ui1} : <i1>
    %72 = buffer %71, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer20"} : <i1>
    %73:2 = fork [2] %72 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork21"} : <i1>
    %74 = buffer %73#0, bufferType = FIFO_BREAK_NONE, numSlots = 7 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer21"} : <i1>
    %75 = andi %70, %74 {handshake.bb = 1 : ui32, handshake.name = "andi8", specv2_tmp_and = true} : <i1>
    %76 = spec_v2_repeating_init %73#1 {handshake.bb = 1 : ui32, handshake.name = "spec_v2_repeating_init8", initToken = 1 : ui1} : <i1>
    %77 = buffer %76, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer22"} : <i1>
    %78:2 = fork [2] %77 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork22"} : <i1>
    %79 = buffer %78#0, bufferType = FIFO_BREAK_NONE, numSlots = 8 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer23"} : <i1>
    %80 = andi %75, %79 {handshake.bb = 1 : ui32, handshake.name = "andi9", specv2_tmp_and = true} : <i1>
    %81 = spec_v2_repeating_init %78#1 {handshake.bb = 1 : ui32, handshake.name = "spec_v2_repeating_init9", initToken = 1 : ui1} : <i1>
    %82 = buffer %81, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer24"} : <i1>
    %83:2 = fork [2] %82 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork23"} : <i1>
    %84 = buffer %83#0, bufferType = FIFO_BREAK_NONE, numSlots = 9 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer25"} : <i1>
    %85 = andi %80, %84 {handshake.bb = 1 : ui32, handshake.name = "andi10", specv2_tmp_and = true} : <i1>
    %86 = spec_v2_repeating_init %83#1 {handshake.bb = 1 : ui32, handshake.name = "spec_v2_repeating_init10", initToken = 1 : ui1} : <i1>
    %87 = buffer %86, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer26"} : <i1>
    %88:2 = fork [2] %87 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork24"} : <i1>
    %89 = buffer %88#0, bufferType = FIFO_BREAK_NONE, numSlots = 10 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer27"} : <i1>
    %90 = andi %85, %89 {handshake.bb = 1 : ui32, handshake.name = "andi11", specv2_tmp_and = true} : <i1>
    %91 = spec_v2_repeating_init %88#1 {handshake.bb = 1 : ui32, handshake.name = "spec_v2_repeating_init11", initToken = 1 : ui1} : <i1>
    %92 = buffer %91, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer29"} : <i1>
    %93:4 = fork [4] %92 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork6"} : <i1>
    %94 = buffer %93#3, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer33"} : <i1>
    %95 = init %94 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "init2", initToken = 0 : ui1} : <i1>
    %96:2 = fork [2] %95 {handshake.bb = 1 : ui32, handshake.name = "fork7"} : <i1>
    %97 = buffer %90, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer28"} : <i1>
    %98 = buffer %93#2, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer31"} : <i1>
    %99 = buffer %98, bufferType = FIFO_BREAK_NONE, numSlots = 11 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer32"} : <i1>
    %100 = andi %97, %99 {handshake.bb = 1 : ui32, handshake.name = "andi12", specv2_tmp_and = true} : <i1>
    %101:3 = fork [3] %100 {handshake.bb = 1 : ui32, handshake.name = "fork26"} : <i1>
    %102 = andi %17, %101#0 {handshake.bb = 1 : ui32, handshake.name = "andi13"} : <i1>
    %103:2 = fork [2] %102 {handshake.bb = 1 : ui32, handshake.name = "fork27"} : <i1>
    %104 = andi %16, %101#1 {handshake.bb = 1 : ui32, handshake.name = "andi14"} : <i1>
    %105:2 = fork [2] %104 {handshake.bb = 1 : ui32, handshake.name = "fork28"} : <i1>
    %106 = not %34#1 {handshake.bb = 1 : ui32, handshake.name = "not1"} : <i1>
    %107 = buffer %30#1, bufferType = FIFO_BREAK_NONE, numSlots = 12 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer3"} : <i12>
    %108 = passer %107[%105#1] {handshake.bb = 1 : ui32, handshake.name = "passer5"} : <i12>, <i1>
    %109 = buffer %3, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer0"} : <i11>
    %110:5 = fork [5] %109 {handshake.bb = 1 : ui32, handshake.name = "fork29"} : <i11>
    %111 = passer %112[%93#1] {handshake.bb = 1 : ui32, handshake.name = "passer10"} : <i11>, <i1>
    %112 = trunci %30#2 {handshake.bb = 1 : ui32, handshake.name = "trunci3"} : <i12> to <i11>
    %113 = buffer %93#0, bufferType = FIFO_BREAK_NONE, numSlots = 12 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer30"} : <i1>
    %114 = passer %117#1[%113] {handshake.bb = 1 : ui32, handshake.name = "passer11"} : <>, <i1>
    %115 = buffer %9, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer1"} : <>
    %116 = buffer %115, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer2"} : <>
    %117:3 = fork [3] %116 {handshake.bb = 1 : ui32, handshake.name = "fork30"} : <>
    %118 = passer %117#0[%105#0] {handshake.bb = 1 : ui32, handshake.name = "passer7"} : <>, <i1>
    %119 = mux %index [%18, %108] {handshake.bb = 2 : ui32, handshake.name = "mux1"} : <i1>, [<i12>, <i12>] to <i12>
    %120 = extsi %119 {handshake.bb = 2 : ui32, handshake.name = "extsi8"} : <i12> to <i32>
    %result, %index = control_merge [%21, %118]  {handshake.bb = 2 : ui32, handshake.name = "control_merge2"} : [<>, <>] to <>, <i1>
    %121:3 = fork [3] %result {handshake.bb = 2 : ui32, handshake.name = "fork5"} : <>
    end {handshake.bb = 2 : ui32, handshake.name = "end0"} %120, %memEnd_3, %memEnd_1, %memEnd, %0#1 : <i32>, <>, <>, <>, <>
  }
}

