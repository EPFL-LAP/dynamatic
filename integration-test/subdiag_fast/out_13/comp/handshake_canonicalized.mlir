module {
  handshake.func @subdiag_fast(%arg0: memref<1000xf32>, %arg1: memref<1000xf32>, %arg2: memref<1000xf32>, %arg3: !handshake.control<>, %arg4: !handshake.control<>, %arg5: !handshake.control<>, %arg6: !handshake.control<>, ...) -> (!handshake.channel<i32>, !handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["d1", "d2", "e", "d1_start", "d2_start", "e_start", "start"], handshake.cfdfcThroughput = #handshake<cfdfcThroughput {"0" = 1.000000e+00 : f64}>, handshake.cfdfcToBBList = #handshake<cfdfcToBBList {"0" = [1 : ui32]}>, resNames = ["out0", "d1_end", "d2_end", "e_end", "end"]} {
    %0:3 = fork [3] %arg6 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork0"} : <>
    %outputs, %memEnd = mem_controller[%arg2 : memref<1000xf32>] %arg5 (%addressResult_6) %152#2 {connectedBlocks = [1 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller3"} :    (!handshake.channel<i10>) -> !handshake.channel<f32>
    %outputs_0, %memEnd_1 = mem_controller[%arg1 : memref<1000xf32>] %arg4 (%addressResult_4) %152#1 {connectedBlocks = [1 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller4"} :    (!handshake.channel<i10>) -> !handshake.channel<f32>
    %outputs_2, %memEnd_3 = mem_controller[%arg0 : memref<1000xf32>] %arg3 (%addressResult) %152#0 {connectedBlocks = [1 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller5"} :    (!handshake.channel<i10>) -> !handshake.channel<f32>
    %1 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant0", value = false} : <>, <i1>
    %2 = extsi %1 {handshake.bb = 0 : ui32, handshake.name = "extsi3"} : <i1> to <i11>
    %3 = mux %126#1 [%2, %142] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i11>, <i11>] to <i11>
    %4 = buffer %141#0, bufferType = FIFO_BREAK_NONE, numSlots = 13 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer37"} : <i11>
    %5 = trunci %4 {handshake.bb = 1 : ui32, handshake.name = "trunci0"} : <i11> to <i10>
    %6 = trunci %141#1 {handshake.bb = 1 : ui32, handshake.name = "trunci1"} : <i11> to <i10>
    %7 = trunci %141#2 {handshake.bb = 1 : ui32, handshake.name = "trunci2"} : <i11> to <i10>
    %8 = mux %126#0 [%0#2, %144] {handshake.bb = 1 : ui32, handshake.name = "mux3"} : <i1>, [<>, <>] to <>
    %9 = source {handshake.bb = 1 : ui32, handshake.name = "source0"} : <>
    %10 = constant %9 {handshake.bb = 1 : ui32, handshake.name = "constant4", value = 1.000000e-03 : f32} : <>, <f32>
    %addressResult, %dataResult = load[%7] %outputs_2 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i10>, <f32>, <i10>, <f32>
    %addressResult_4, %dataResult_5 = load[%6] %outputs_0 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i10>, <f32>, <i10>, <f32>
    %11 = addf %dataResult, %dataResult_5 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "addf0", internal_delay = "2_922000"} : <f32>
    %addressResult_6, %dataResult_7 = load[%5] %outputs {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load2"} : <i10>, <f32>, <i10>, <f32>
    %12 = mulf %11, %10 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "mulf0", internal_delay = "2_875333"} : <f32>
    %13:3 = fork [3] %14 {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <i1>
    %14 = cmpf ugt, %dataResult_7, %12 {handshake.bb = 1 : ui32, handshake.name = "cmpf0", internal_delay = "0_000000"} : <f32>
    %15 = andi %137, %13#1 {handshake.bb = 1 : ui32, handshake.name = "andi0"} : <i1>
    %16 = not %13#0 {handshake.bb = 1 : ui32, handshake.name = "not0"} : <i1>
    %17 = passer %19[%134#1] {handshake.bb = 1 : ui32, handshake.name = "passer8"} : <i12>, <i1>
    %18 = buffer %141#3, bufferType = FIFO_BREAK_NONE, numSlots = 14 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer38"} : <i11>
    %19 = extsi %18 {handshake.bb = 1 : ui32, handshake.name = "extsi4"} : <i11> to <i12>
    %20 = buffer %147#2, bufferType = FIFO_BREAK_NONE, numSlots = 13 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer40"} : <>
    %21 = passer %20[%134#0] {handshake.bb = 1 : ui32, handshake.name = "passer3"} : <>, <i1>
    %22 = extsi %141#4 {handshake.bb = 1 : ui32, handshake.name = "extsi5"} : <i11> to <i12>
    %23 = source {handshake.bb = 1 : ui32, handshake.name = "source1"} : <>
    %24 = constant %23 {handshake.bb = 1 : ui32, handshake.name = "constant1", value = 1 : i2} : <>, <i2>
    %25 = extsi %24 {handshake.bb = 1 : ui32, handshake.name = "extsi6"} : <i2> to <i12>
    %26 = source {handshake.bb = 1 : ui32, handshake.name = "source2"} : <>
    %27 = constant %26 {handshake.bb = 1 : ui32, handshake.name = "constant2", value = 999 : i11} : <>, <i11>
    %28 = extsi %27 {handshake.bb = 1 : ui32, handshake.name = "extsi7"} : <i11> to <i12>
    %29:3 = fork [3] %31 {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <i12>
    %30 = buffer %22, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer3"} : <i12>
    %31 = addi %30, %25 {handshake.bb = 1 : ui32, handshake.name = "addi0"} : <i12>
    %32:2 = fork [2] %35 {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <i1>
    %33 = buffer %29#0, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer4"} : <i12>
    %34 = buffer %33, bufferType = FIFO_BREAK_NONE, numSlots = 12 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer5"} : <i12>
    %35 = cmpi ult, %34, %28 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i12>
    %36 = passer %37[%132#2] {handshake.bb = 1 : ui32, handshake.name = "passer9"} : <i1>, <i1>
    %37 = andi %13#2, %32#0 {handshake.bb = 1 : ui32, handshake.name = "andi1"} : <i1>
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
    %105 = buffer %104, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer27"} : <i1>
    %106:2 = fork [2] %105 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork24"} : <i1>
    %107 = buffer %103, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer26"} : <i1>
    %108 = buffer %106#0, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer28"} : <i1>
    %109 = buffer %108, bufferType = FIFO_BREAK_NONE, numSlots = 10 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer29"} : <i1>
    %110 = source {handshake.bb = 1 : ui32, handshake.name = "source12"} : <>
    %111 = constant %110 {handshake.bb = 1 : ui32, handshake.name = "constant13", value = false} : <>, <i1>
    %112 = mux %109 [%111, %107] {handshake.bb = 1 : ui32, handshake.name = "mux12"} : <i1>, [<i1>, <i1>] to <i1>
    %113 = spec_v2_repeating_init %106#1 {handshake.bb = 1 : ui32, handshake.name = "spec_v2_repeating_init11", initToken = 1 : ui1} : <i1>
    %114 = buffer %113, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer30"} : <i1>
    %115:2 = fork [2] %114 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork25"} : <i1>
    %116 = buffer %115#0, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer31"} : <i1>
    %117 = buffer %116, bufferType = FIFO_BREAK_NONE, numSlots = 11 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer32"} : <i1>
    %118 = source {handshake.bb = 1 : ui32, handshake.name = "source13"} : <>
    %119 = constant %118 {handshake.bb = 1 : ui32, handshake.name = "constant14", value = false} : <>, <i1>
    %120 = mux %117 [%119, %112] {handshake.bb = 1 : ui32, handshake.name = "mux13"} : <i1>, [<i1>, <i1>] to <i1>
    %121 = spec_v2_repeating_init %115#1 {handshake.bb = 1 : ui32, handshake.name = "spec_v2_repeating_init12", initToken = 1 : ui1} : <i1>
    %122 = buffer %121, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer33"} : <i1>
    %123:4 = fork [4] %122 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork6"} : <i1>
    %124 = buffer %123#3, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer36"} : <i1>
    %125 = init %124 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "init2", initToken = 0 : ui1} : <i1>
    %126:2 = fork [2] %125 {handshake.bb = 1 : ui32, handshake.name = "fork7"} : <i1>
    %127 = buffer %123#2, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer34"} : <i1>
    %128 = buffer %127, bufferType = FIFO_BREAK_NONE, numSlots = 12 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer35"} : <i1>
    %129 = source {handshake.bb = 1 : ui32, handshake.name = "source14"} : <>
    %130 = constant %129 {handshake.bb = 1 : ui32, handshake.name = "constant15", value = false} : <>, <i1>
    %131 = mux %128 [%130, %120] {handshake.bb = 1 : ui32, handshake.name = "mux14"} : <i1>, [<i1>, <i1>] to <i1>
    %132:3 = fork [3] %131 {handshake.bb = 1 : ui32, handshake.name = "fork27"} : <i1>
    %133 = andi %16, %132#0 {handshake.bb = 1 : ui32, handshake.name = "andi14"} : <i1>
    %134:2 = fork [2] %133 {handshake.bb = 1 : ui32, handshake.name = "fork28"} : <i1>
    %135 = andi %15, %132#1 {handshake.bb = 1 : ui32, handshake.name = "andi15"} : <i1>
    %136:2 = fork [2] %135 {handshake.bb = 1 : ui32, handshake.name = "fork29"} : <i1>
    %137 = not %32#1 {handshake.bb = 1 : ui32, handshake.name = "not1"} : <i1>
    %138 = buffer %29#1, bufferType = FIFO_BREAK_NONE, numSlots = 13 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer6"} : <i12>
    %139 = passer %138[%136#0] {handshake.bb = 1 : ui32, handshake.name = "passer5"} : <i12>, <i1>
    %140 = buffer %3, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer0"} : <i11>
    %141:5 = fork [5] %140 {handshake.bb = 1 : ui32, handshake.name = "fork30"} : <i11>
    %142 = passer %143[%123#0] {handshake.bb = 1 : ui32, handshake.name = "passer10"} : <i11>, <i1>
    %143 = trunci %29#2 {handshake.bb = 1 : ui32, handshake.name = "trunci3"} : <i12> to <i11>
    %144 = passer %147#1[%123#1] {handshake.bb = 1 : ui32, handshake.name = "passer11"} : <>, <i1>
    %145 = buffer %8, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer1"} : <>
    %146 = buffer %145, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer2"} : <>
    %147:3 = fork [3] %146 {handshake.bb = 1 : ui32, handshake.name = "fork31"} : <>
    %148 = buffer %147#0, bufferType = FIFO_BREAK_NONE, numSlots = 13 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer39"} : <>
    %149 = passer %148[%136#1] {handshake.bb = 1 : ui32, handshake.name = "passer7"} : <>, <i1>
    %150 = mux %index [%17, %139] {handshake.bb = 2 : ui32, handshake.name = "mux1"} : <i1>, [<i12>, <i12>] to <i12>
    %151 = extsi %150 {handshake.bb = 2 : ui32, handshake.name = "extsi8"} : <i12> to <i32>
    %result, %index = control_merge [%21, %149]  {handshake.bb = 2 : ui32, handshake.name = "control_merge2"} : [<>, <>] to <>, <i1>
    %152:3 = fork [3] %result {handshake.bb = 2 : ui32, handshake.name = "fork5"} : <>
    end {handshake.bb = 2 : ui32, handshake.name = "end0"} %151, %memEnd_3, %memEnd_1, %memEnd, %0#1 : <i32>, <>, <>, <>, <>
  }
}

