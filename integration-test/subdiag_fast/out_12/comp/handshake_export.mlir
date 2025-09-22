module {
  handshake.func @subdiag_fast(%arg0: memref<1000xf32>, %arg1: memref<1000xf32>, %arg2: memref<1000xf32>, %arg3: !handshake.control<>, %arg4: !handshake.control<>, %arg5: !handshake.control<>, %arg6: !handshake.control<>, ...) -> (!handshake.channel<i32>, !handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["d1", "d2", "e", "d1_start", "d2_start", "e_start", "start"], handshake.cfdfcThroughput = #handshake<cfdfcThroughput {"0" = 0.9285714285714286 : f64}>, handshake.cfdfcToBBList = #handshake<cfdfcToBBList {"0" = [1 : ui32]}>, resNames = ["out0", "d1_end", "d2_end", "e_end", "end"]} {
    %0:3 = fork [3] %arg6 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork0"} : <>
    %outputs, %memEnd = mem_controller[%arg2 : memref<1000xf32>] %arg5 (%addressResult_6) %143#2 {connectedBlocks = [1 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller3"} :    (!handshake.channel<i10>) -> !handshake.channel<f32>
    %outputs_0, %memEnd_1 = mem_controller[%arg1 : memref<1000xf32>] %arg4 (%addressResult_4) %143#1 {connectedBlocks = [1 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller4"} :    (!handshake.channel<i10>) -> !handshake.channel<f32>
    %outputs_2, %memEnd_3 = mem_controller[%arg0 : memref<1000xf32>] %arg3 (%addressResult) %143#0 {connectedBlocks = [1 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller5"} :    (!handshake.channel<i10>) -> !handshake.channel<f32>
    %1 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant0", value = false} : <>, <i1>
    %2 = extsi %1 {handshake.bb = 0 : ui32, handshake.name = "extsi3"} : <i1> to <i11>
    %3 = mux %116#0 [%2, %133] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i11>, <i11>] to <i11>
    %4 = buffer %132#0, bufferType = FIFO_BREAK_NONE, numSlots = 13 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer35"} : <i11>
    %5 = trunci %4 {handshake.bb = 1 : ui32, handshake.name = "trunci0"} : <i11> to <i10>
    %6 = trunci %132#1 {handshake.bb = 1 : ui32, handshake.name = "trunci1"} : <i11> to <i10>
    %7 = trunci %132#2 {handshake.bb = 1 : ui32, handshake.name = "trunci2"} : <i11> to <i10>
    %8 = buffer %116#1, bufferType = FIFO_BREAK_NONE, numSlots = 12 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer34"} : <i1>
    %9 = mux %8 [%0#2, %136] {handshake.bb = 1 : ui32, handshake.name = "mux3"} : <i1>, [<>, <>] to <>
    %10 = source {handshake.bb = 1 : ui32, handshake.name = "source0"} : <>
    %11 = constant %10 {handshake.bb = 1 : ui32, handshake.name = "constant4", value = 1.000000e-03 : f32} : <>, <f32>
    %addressResult, %dataResult = load[%7] %outputs_2 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i10>, <f32>, <i10>, <f32>
    %addressResult_4, %dataResult_5 = load[%6] %outputs_0 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i10>, <f32>, <i10>, <f32>
    %12 = addf %dataResult, %dataResult_5 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "addf0", internal_delay = "2_922000"} : <f32>
    %addressResult_6, %dataResult_7 = load[%5] %outputs {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load2"} : <i10>, <f32>, <i10>, <f32>
    %13 = mulf %12, %11 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "mulf0", internal_delay = "2_875333"} : <f32>
    %14:3 = fork [3] %15 {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <i1>
    %15 = cmpf ugt, %dataResult_7, %13 {handshake.bb = 1 : ui32, handshake.name = "cmpf0", internal_delay = "0_000000"} : <f32>
    %16 = andi %128, %14#1 {handshake.bb = 1 : ui32, handshake.name = "andi0"} : <i1>
    %17 = not %14#0 {handshake.bb = 1 : ui32, handshake.name = "not0"} : <i1>
    %18 = passer %20[%125#1] {handshake.bb = 1 : ui32, handshake.name = "passer8"} : <i12>, <i1>
    %19 = buffer %132#3, bufferType = FIFO_BREAK_NONE, numSlots = 13 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer36"} : <i11>
    %20 = extsi %19 {handshake.bb = 1 : ui32, handshake.name = "extsi4"} : <i11> to <i12>
    %21 = passer %139#2[%125#0] {handshake.bb = 1 : ui32, handshake.name = "passer3"} : <>, <i1>
    %22 = extsi %132#4 {handshake.bb = 1 : ui32, handshake.name = "extsi5"} : <i11> to <i12>
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
    %36 = passer %37[%123#2] {handshake.bb = 1 : ui32, handshake.name = "passer9"} : <i1>, <i1>
    %37 = andi %14#2, %34#0 {handshake.bb = 1 : ui32, handshake.name = "andi1"} : <i1>
    %38 = spec_v2_repeating_init %36 {handshake.bb = 1 : ui32, handshake.name = "spec_v2_repeating_init0", initToken = 1 : ui1} : <i1>
    %39 = buffer %38, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer7"} : <i1>
    %40:2 = fork [2] %39 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork4"} : <i1>
    %41 = spec_v2_repeating_init %40#0 {handshake.bb = 1 : ui32, handshake.name = "spec_v2_repeating_init1", initToken = 1 : ui1} : <i1>
    %42 = buffer %41, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer8"} : <i1>
    %43:2 = fork [2] %42 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork15"} : <i1>
    %44 = buffer %43#0, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer9"} : <i1>
    %45 = source {handshake.bb = 1 : ui32, handshake.name = "source3"} : <>
    %46 = constant %45 {handshake.bb = 1 : ui32, handshake.name = "constant3", value = false} : <>, <i1>
    %47 = mux %44 [%46, %40#1] {handshake.bb = 1 : ui32, handshake.name = "mux2"} : <i1>, [<i1>, <i1>] to <i1>
    %48 = spec_v2_repeating_init %43#1 {handshake.bb = 1 : ui32, handshake.name = "spec_v2_repeating_init2", initToken = 1 : ui1} : <i1>
    %49 = buffer %48, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer10"} : <i1>
    %50:2 = fork [2] %49 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork16"} : <i1>
    %51 = buffer %50#0, bufferType = FIFO_BREAK_NONE, numSlots = 2 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer11"} : <i1>
    %52 = source {handshake.bb = 1 : ui32, handshake.name = "source4"} : <>
    %53 = constant %52 {handshake.bb = 1 : ui32, handshake.name = "constant5", value = false} : <>, <i1>
    %54 = mux %51 [%53, %47] {handshake.bb = 1 : ui32, handshake.name = "mux4"} : <i1>, [<i1>, <i1>] to <i1>
    %55 = spec_v2_repeating_init %50#1 {handshake.bb = 1 : ui32, handshake.name = "spec_v2_repeating_init3", initToken = 1 : ui1} : <i1>
    %56 = buffer %55, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer12"} : <i1>
    %57:2 = fork [2] %56 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork17"} : <i1>
    %58 = buffer %57#0, bufferType = FIFO_BREAK_NONE, numSlots = 3 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer13"} : <i1>
    %59 = source {handshake.bb = 1 : ui32, handshake.name = "source5"} : <>
    %60 = constant %59 {handshake.bb = 1 : ui32, handshake.name = "constant6", value = false} : <>, <i1>
    %61 = mux %58 [%60, %54] {handshake.bb = 1 : ui32, handshake.name = "mux5"} : <i1>, [<i1>, <i1>] to <i1>
    %62 = spec_v2_repeating_init %57#1 {handshake.bb = 1 : ui32, handshake.name = "spec_v2_repeating_init4", initToken = 1 : ui1} : <i1>
    %63 = buffer %62, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer14"} : <i1>
    %64:2 = fork [2] %63 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork18"} : <i1>
    %65 = buffer %64#0, bufferType = FIFO_BREAK_NONE, numSlots = 4 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer15"} : <i1>
    %66 = source {handshake.bb = 1 : ui32, handshake.name = "source6"} : <>
    %67 = constant %66 {handshake.bb = 1 : ui32, handshake.name = "constant7", value = false} : <>, <i1>
    %68 = mux %65 [%67, %61] {handshake.bb = 1 : ui32, handshake.name = "mux6"} : <i1>, [<i1>, <i1>] to <i1>
    %69 = spec_v2_repeating_init %64#1 {handshake.bb = 1 : ui32, handshake.name = "spec_v2_repeating_init5", initToken = 1 : ui1} : <i1>
    %70 = buffer %69, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer16"} : <i1>
    %71:2 = fork [2] %70 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork19"} : <i1>
    %72 = buffer %71#0, bufferType = FIFO_BREAK_NONE, numSlots = 5 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer17"} : <i1>
    %73 = source {handshake.bb = 1 : ui32, handshake.name = "source7"} : <>
    %74 = constant %73 {handshake.bb = 1 : ui32, handshake.name = "constant8", value = false} : <>, <i1>
    %75 = mux %72 [%74, %68] {handshake.bb = 1 : ui32, handshake.name = "mux7"} : <i1>, [<i1>, <i1>] to <i1>
    %76 = spec_v2_repeating_init %71#1 {handshake.bb = 1 : ui32, handshake.name = "spec_v2_repeating_init6", initToken = 1 : ui1} : <i1>
    %77 = buffer %76, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer18"} : <i1>
    %78:2 = fork [2] %77 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork20"} : <i1>
    %79 = buffer %78#0, bufferType = FIFO_BREAK_NONE, numSlots = 6 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer19"} : <i1>
    %80 = source {handshake.bb = 1 : ui32, handshake.name = "source8"} : <>
    %81 = constant %80 {handshake.bb = 1 : ui32, handshake.name = "constant9", value = false} : <>, <i1>
    %82 = mux %79 [%81, %75] {handshake.bb = 1 : ui32, handshake.name = "mux8"} : <i1>, [<i1>, <i1>] to <i1>
    %83 = spec_v2_repeating_init %78#1 {handshake.bb = 1 : ui32, handshake.name = "spec_v2_repeating_init7", initToken = 1 : ui1} : <i1>
    %84 = buffer %83, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer20"} : <i1>
    %85:2 = fork [2] %84 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork21"} : <i1>
    %86 = buffer %85#0, bufferType = FIFO_BREAK_NONE, numSlots = 7 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer21"} : <i1>
    %87 = source {handshake.bb = 1 : ui32, handshake.name = "source9"} : <>
    %88 = constant %87 {handshake.bb = 1 : ui32, handshake.name = "constant10", value = false} : <>, <i1>
    %89 = mux %86 [%88, %82] {handshake.bb = 1 : ui32, handshake.name = "mux9"} : <i1>, [<i1>, <i1>] to <i1>
    %90 = spec_v2_repeating_init %85#1 {handshake.bb = 1 : ui32, handshake.name = "spec_v2_repeating_init8", initToken = 1 : ui1} : <i1>
    %91 = buffer %90, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer22"} : <i1>
    %92:2 = fork [2] %91 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork22"} : <i1>
    %93 = buffer %92#0, bufferType = FIFO_BREAK_NONE, numSlots = 8 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer23"} : <i1>
    %94 = source {handshake.bb = 1 : ui32, handshake.name = "source10"} : <>
    %95 = constant %94 {handshake.bb = 1 : ui32, handshake.name = "constant11", value = false} : <>, <i1>
    %96 = mux %93 [%95, %89] {handshake.bb = 1 : ui32, handshake.name = "mux10"} : <i1>, [<i1>, <i1>] to <i1>
    %97 = spec_v2_repeating_init %92#1 {handshake.bb = 1 : ui32, handshake.name = "spec_v2_repeating_init9", initToken = 1 : ui1} : <i1>
    %98 = buffer %97, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer24"} : <i1>
    %99:2 = fork [2] %98 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork23"} : <i1>
    %100 = buffer %99#0, bufferType = FIFO_BREAK_NONE, numSlots = 9 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer25"} : <i1>
    %101 = source {handshake.bb = 1 : ui32, handshake.name = "source11"} : <>
    %102 = constant %101 {handshake.bb = 1 : ui32, handshake.name = "constant12", value = false} : <>, <i1>
    %103 = mux %100 [%102, %96] {handshake.bb = 1 : ui32, handshake.name = "mux11"} : <i1>, [<i1>, <i1>] to <i1>
    %104 = spec_v2_repeating_init %99#1 {handshake.bb = 1 : ui32, handshake.name = "spec_v2_repeating_init10", initToken = 1 : ui1} : <i1>
    %105 = buffer %104, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer26"} : <i1>
    %106:2 = fork [2] %105 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork24"} : <i1>
    %107 = buffer %106#0, bufferType = FIFO_BREAK_NONE, numSlots = 10 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer27"} : <i1>
    %108 = source {handshake.bb = 1 : ui32, handshake.name = "source12"} : <>
    %109 = constant %108 {handshake.bb = 1 : ui32, handshake.name = "constant13", value = false} : <>, <i1>
    %110 = mux %107 [%109, %103] {handshake.bb = 1 : ui32, handshake.name = "mux12"} : <i1>, [<i1>, <i1>] to <i1>
    %111 = spec_v2_repeating_init %106#1 {handshake.bb = 1 : ui32, handshake.name = "spec_v2_repeating_init11", initToken = 1 : ui1} : <i1>
    %112 = buffer %111, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer29"} : <i1>
    %113:4 = fork [4] %112 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork6"} : <i1>
    %114 = buffer %113#3, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer33"} : <i1>
    %115 = init %114 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "init2", initToken = 0 : ui1} : <i1>
    %116:2 = fork [2] %115 {handshake.bb = 1 : ui32, handshake.name = "fork7"} : <i1>
    %117 = buffer %110, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer28"} : <i1>
    %118 = buffer %113#2, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer31"} : <i1>
    %119 = buffer %118, bufferType = FIFO_BREAK_NONE, numSlots = 11 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer32"} : <i1>
    %120 = source {handshake.bb = 1 : ui32, handshake.name = "source13"} : <>
    %121 = constant %120 {handshake.bb = 1 : ui32, handshake.name = "constant14", value = false} : <>, <i1>
    %122 = mux %119 [%121, %117] {handshake.bb = 1 : ui32, handshake.name = "mux13"} : <i1>, [<i1>, <i1>] to <i1>
    %123:3 = fork [3] %122 {handshake.bb = 1 : ui32, handshake.name = "fork26"} : <i1>
    %124 = andi %17, %123#0 {handshake.bb = 1 : ui32, handshake.name = "andi13"} : <i1>
    %125:2 = fork [2] %124 {handshake.bb = 1 : ui32, handshake.name = "fork27"} : <i1>
    %126 = andi %16, %123#1 {handshake.bb = 1 : ui32, handshake.name = "andi14"} : <i1>
    %127:2 = fork [2] %126 {handshake.bb = 1 : ui32, handshake.name = "fork28"} : <i1>
    %128 = not %34#1 {handshake.bb = 1 : ui32, handshake.name = "not1"} : <i1>
    %129 = buffer %30#1, bufferType = FIFO_BREAK_NONE, numSlots = 12 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer3"} : <i12>
    %130 = passer %129[%127#1] {handshake.bb = 1 : ui32, handshake.name = "passer5"} : <i12>, <i1>
    %131 = buffer %3, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer0"} : <i11>
    %132:5 = fork [5] %131 {handshake.bb = 1 : ui32, handshake.name = "fork29"} : <i11>
    %133 = passer %134[%113#1] {handshake.bb = 1 : ui32, handshake.name = "passer10"} : <i11>, <i1>
    %134 = trunci %30#2 {handshake.bb = 1 : ui32, handshake.name = "trunci3"} : <i12> to <i11>
    %135 = buffer %113#0, bufferType = FIFO_BREAK_NONE, numSlots = 12 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer30"} : <i1>
    %136 = passer %139#1[%135] {handshake.bb = 1 : ui32, handshake.name = "passer11"} : <>, <i1>
    %137 = buffer %9, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer1"} : <>
    %138 = buffer %137, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer2"} : <>
    %139:3 = fork [3] %138 {handshake.bb = 1 : ui32, handshake.name = "fork30"} : <>
    %140 = passer %139#0[%127#0] {handshake.bb = 1 : ui32, handshake.name = "passer7"} : <>, <i1>
    %141 = mux %index [%18, %130] {handshake.bb = 2 : ui32, handshake.name = "mux1"} : <i1>, [<i12>, <i12>] to <i12>
    %142 = extsi %141 {handshake.bb = 2 : ui32, handshake.name = "extsi8"} : <i12> to <i32>
    %result, %index = control_merge [%21, %140]  {handshake.bb = 2 : ui32, handshake.name = "control_merge2"} : [<>, <>] to <>, <i1>
    %143:3 = fork [3] %result {handshake.bb = 2 : ui32, handshake.name = "fork5"} : <>
    end {handshake.bb = 2 : ui32, handshake.name = "end0"} %142, %memEnd_3, %memEnd_1, %memEnd, %0#1 : <i32>, <>, <>, <>, <>
  }
}

