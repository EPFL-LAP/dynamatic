module {
  handshake.func @collision_donut(%arg0: memref<1000xi32>, %arg1: memref<1000xi32>, %arg2: !handshake.control<>, %arg3: !handshake.control<>, %arg4: !handshake.control<>, ...) -> (!handshake.channel<i32>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["x", "y", "x_start", "y_start", "start"], handshake.cfdfcThroughput = #handshake<cfdfcThroughput {"0" = 1.000000e+00 : f64}>, handshake.cfdfcToBBList = #handshake<cfdfcToBBList {"0" = [1 : ui32]}>, resNames = ["out0", "x_end", "y_end", "end"]} {
    %0:3 = fork [3] %arg4 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork0"} : <>
    %outputs, %memEnd = mem_controller[%arg1 : memref<1000xi32>] %arg3 (%addressResult_2) %132#1 {connectedBlocks = [1 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller2"} :    (!handshake.channel<i10>) -> !handshake.channel<i32>
    %outputs_0, %memEnd_1 = mem_controller[%arg0 : memref<1000xi32>] %arg2 (%addressResult) %132#0 {connectedBlocks = [1 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller3"} :    (!handshake.channel<i10>) -> !handshake.channel<i32>
    %1 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant0", value = false} : <>, <i1>
    %2 = extsi %1 {handshake.bb = 0 : ui32, handshake.name = "extsi7"} : <i1> to <i11>
    %3 = buffer %93#3, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer23"} : <i1>
    %4 = init %3 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "init0", initToken = 0 : ui1} : <i1>
    %5 = buffer %93#4, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer24"} : <i1>
    %6 = buffer %5, bufferType = FIFO_BREAK_NONE, numSlots = 4 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer25"} : <i1>
    %7 = init %6 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "init1", initToken = 0 : ui1} : <i1>
    %8 = mux %4 [%2, %114] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i11>, <i11>] to <i11>
    %9 = trunci %113#0 {handshake.bb = 1 : ui32, handshake.name = "trunci0"} : <i11> to <i10>
    %10 = trunci %113#1 {handshake.bb = 1 : ui32, handshake.name = "trunci1"} : <i11> to <i10>
    %11 = mux %7 [%0#2, %117] {handshake.bb = 1 : ui32, handshake.name = "mux7"} : <i1>, [<>, <>] to <>
    %12 = source {handshake.bb = 1 : ui32, handshake.name = "source0"} : <>
    %13 = constant %12 {handshake.bb = 1 : ui32, handshake.name = "constant1", value = 4 : i4} : <>, <i4>
    %14 = extsi %13 {handshake.bb = 1 : ui32, handshake.name = "extsi1"} : <i4> to <i32>
    %15 = constant %120#0 {handshake.bb = 1 : ui32, handshake.name = "constant8", value = -1 : i32} : <>, <i32>
    %16:2 = fork [2] %dataResult {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <i32>
    %addressResult, %dataResult = load[%10] %outputs_0 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i10>, <i32>, <i10>, <i32>
    %17:2 = fork [2] %dataResult_3 {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <i32>
    %addressResult_2, %dataResult_3 = load[%9] %outputs {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i10>, <i32>, <i10>, <i32>
    %18 = muli %16#0, %16#1 {handshake.bb = 1 : ui32, handshake.name = "muli0"} : <i32>
    %19 = muli %17#0, %17#1 {handshake.bb = 1 : ui32, handshake.name = "muli1"} : <i32>
    %20:2 = fork [2] %21 {handshake.bb = 1 : ui32, handshake.name = "fork6"} : <i32>
    %21 = addi %18, %19 {handshake.bb = 1 : ui32, handshake.name = "addi0"} : <i32>
    %22 = cmpi ult, %20#1, %14 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i32>
    %23:2 = fork [2] %22 {handshake.bb = 1 : ui32, handshake.name = "fork7"} : <i1>
    %24:2 = fork [2] %25 {handshake.bb = 1 : ui32, handshake.name = "fork8"} : <i1>
    %25 = not %23#0 {handshake.bb = 1 : ui32, handshake.name = "not0"} : <i1>
    %26 = andi %109, %42#0 {handshake.bb = 1 : ui32, handshake.name = "andi1"} : <i1>
    %27 = andi %38#1, %24#0 {handshake.bb = 1 : ui32, handshake.name = "andi2"} : <i1>
    %28 = buffer %31, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer5"} : <i12>
    %29 = buffer %28, bufferType = FIFO_BREAK_NONE, numSlots = 5 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer6"} : <i12>
    %30 = passer %29[%108#2] {handshake.bb = 1 : ui32, handshake.name = "passer16"} : <i12>, <i1>
    %31 = extsi %113#4 {handshake.bb = 1 : ui32, handshake.name = "extsi8"} : <i11> to <i12>
    %32 = passer %15[%108#1] {handshake.bb = 1 : ui32, handshake.name = "passer2"} : <i32>, <i1>
    %33 = passer %120#6[%108#0] {handshake.bb = 1 : ui32, handshake.name = "passer3"} : <>, <i1>
    %34 = source {handshake.bb = 1 : ui32, handshake.name = "source1"} : <>
    %35 = constant %34 {handshake.bb = 1 : ui32, handshake.name = "constant2", value = 19000 : i16} : <>, <i16>
    %36 = extsi %35 {handshake.bb = 1 : ui32, handshake.name = "extsi2"} : <i16> to <i32>
    %37 = constant %120#1 {handshake.bb = 1 : ui32, handshake.name = "constant10", value = -2 : i32} : <>, <i32>
    %38:2 = fork [2] %39 {handshake.bb = 1 : ui32, handshake.name = "fork9"} : <i1>
    %39 = cmpi ugt, %20#0, %36 {handshake.bb = 1 : ui32, handshake.name = "cmpi1"} : <i32>
    %40 = not %38#0 {handshake.bb = 1 : ui32, handshake.name = "not1"} : <i1>
    %41 = buffer %43, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer7"} : <i1>
    %42:2 = fork [2] %41 {handshake.bb = 1 : ui32, handshake.name = "fork10"} : <i1>
    %43 = andi %24#1, %40 {handshake.bb = 1 : ui32, handshake.name = "andi3"} : <i1>
    %44 = passer %47[%103#0] {handshake.bb = 1 : ui32, handshake.name = "passer17"} : <i12>, <i1>
    %45 = buffer %113#3, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer27"} : <i11>
    %46 = buffer %45, bufferType = FIFO_BREAK_NONE, numSlots = 5 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer28"} : <i11>
    %47 = extsi %46 {handshake.bb = 1 : ui32, handshake.name = "extsi9"} : <i11> to <i12>
    %48 = passer %37[%103#1] {handshake.bb = 1 : ui32, handshake.name = "passer8"} : <i32>, <i1>
    %49 = passer %120#5[%103#2] {handshake.bb = 1 : ui32, handshake.name = "passer9"} : <>, <i1>
    %50 = buffer %113#2, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer26"} : <i11>
    %51 = extsi %50 {handshake.bb = 1 : ui32, handshake.name = "extsi10"} : <i11> to <i12>
    %52 = constant %120#2 {handshake.bb = 1 : ui32, handshake.name = "constant3", value = false} : <>, <i1>
    %53 = source {handshake.bb = 1 : ui32, handshake.name = "source2"} : <>
    %54 = constant %53 {handshake.bb = 1 : ui32, handshake.name = "constant4", value = 1 : i2} : <>, <i2>
    %55 = extsi %54 {handshake.bb = 1 : ui32, handshake.name = "extsi11"} : <i2> to <i12>
    %56 = source {handshake.bb = 1 : ui32, handshake.name = "source3"} : <>
    %57 = constant %56 {handshake.bb = 1 : ui32, handshake.name = "constant5", value = 1000 : i11} : <>, <i11>
    %58 = extsi %57 {handshake.bb = 1 : ui32, handshake.name = "extsi12"} : <i11> to <i12>
    %59:3 = fork [3] %60 {handshake.bb = 1 : ui32, handshake.name = "fork11"} : <i12>
    %60 = addi %51, %55 {handshake.bb = 1 : ui32, handshake.name = "addi1"} : <i12>
    %61 = buffer %64, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer9"} : <i1>
    %62 = buffer %61, bufferType = FIFO_BREAK_NONE, numSlots = 4 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer10"} : <i1>
    %63:2 = fork [2] %62 {handshake.bb = 1 : ui32, handshake.name = "fork28"} : <i1>
    %64 = cmpi ult, %59#0, %58 {handshake.bb = 1 : ui32, handshake.name = "cmpi2"} : <i12>
    %65 = passer %66[%100#3] {handshake.bb = 1 : ui32, handshake.name = "passer18"} : <i1>, <i1>
    %66 = andi %42#1, %63#0 {handshake.bb = 1 : ui32, handshake.name = "andi4"} : <i1>
    %67 = spec_v2_repeating_init %65 {handshake.bb = 1 : ui32, handshake.name = "spec_v2_repeating_init0", initToken = 1 : ui1} : <i1>
    %68 = buffer %67, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer11"} : <i1>
    %69:2 = fork [2] %68 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork29"} : <i1>
    %70 = spec_v2_repeating_init %69#0 {handshake.bb = 1 : ui32, handshake.name = "spec_v2_repeating_init1", initToken = 1 : ui1} : <i1>
    %71 = buffer %70, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer12"} : <i1>
    %72:2 = fork [2] %71 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork30"} : <i1>
    %73 = buffer %72#0, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer13"} : <i1>
    %74 = source {handshake.bb = 1 : ui32, handshake.name = "source5"} : <>
    %75 = constant %74 {handshake.bb = 1 : ui32, handshake.name = "constant6", value = false} : <>, <i1>
    %76 = mux %73 [%75, %69#1] {handshake.bb = 1 : ui32, handshake.name = "mux5"} : <i1>, [<i1>, <i1>] to <i1>
    %77 = spec_v2_repeating_init %72#1 {handshake.bb = 1 : ui32, handshake.name = "spec_v2_repeating_init2", initToken = 1 : ui1} : <i1>
    %78 = buffer %77, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer14"} : <i1>
    %79:2 = fork [2] %78 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork31"} : <i1>
    %80 = buffer %79#0, bufferType = FIFO_BREAK_NONE, numSlots = 2 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer15"} : <i1>
    %81 = source {handshake.bb = 1 : ui32, handshake.name = "source6"} : <>
    %82 = constant %81 {handshake.bb = 1 : ui32, handshake.name = "constant7", value = false} : <>, <i1>
    %83 = mux %80 [%82, %76] {handshake.bb = 1 : ui32, handshake.name = "mux6"} : <i1>, [<i1>, <i1>] to <i1>
    %84 = spec_v2_repeating_init %79#1 {handshake.bb = 1 : ui32, handshake.name = "spec_v2_repeating_init3", initToken = 1 : ui1} : <i1>
    %85 = buffer %84, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer16"} : <i1>
    %86:2 = fork [2] %85 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork32"} : <i1>
    %87 = buffer %86#0, bufferType = FIFO_BREAK_NONE, numSlots = 3 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer17"} : <i1>
    %88 = source {handshake.bb = 1 : ui32, handshake.name = "source7"} : <>
    %89 = constant %88 {handshake.bb = 1 : ui32, handshake.name = "constant9", value = false} : <>, <i1>
    %90 = mux %87 [%89, %83] {handshake.bb = 1 : ui32, handshake.name = "mux8"} : <i1>, [<i1>, <i1>] to <i1>
    %91 = spec_v2_repeating_init %86#1 {handshake.bb = 1 : ui32, handshake.name = "spec_v2_repeating_init4", initToken = 1 : ui1} : <i1>
    %92 = buffer %91, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer19"} : <i1>
    %93:5 = fork [5] %92 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork33"} : <i1>
    %94 = buffer %90, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer18"} : <i1>
    %95 = buffer %93#0, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer20"} : <i1>
    %96 = buffer %95, bufferType = FIFO_BREAK_NONE, numSlots = 4 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer21"} : <i1>
    %97 = source {handshake.bb = 1 : ui32, handshake.name = "source8"} : <>
    %98 = constant %97 {handshake.bb = 1 : ui32, handshake.name = "constant11", value = false} : <>, <i1>
    %99 = mux %96 [%98, %94] {handshake.bb = 1 : ui32, handshake.name = "mux9"} : <i1>, [<i1>, <i1>] to <i1>
    %100:4 = fork [4] %99 {handshake.bb = 1 : ui32, handshake.name = "fork34"} : <i1>
    %101 = buffer %27, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer4"} : <i1>
    %102 = andi %101, %100#0 {handshake.bb = 1 : ui32, handshake.name = "andi9"} : <i1>
    %103:3 = fork [3] %102 {handshake.bb = 1 : ui32, handshake.name = "fork35"} : <i1>
    %104 = andi %26, %100#1 {handshake.bb = 1 : ui32, handshake.name = "andi10"} : <i1>
    %105:3 = fork [3] %104 {handshake.bb = 1 : ui32, handshake.name = "fork36"} : <i1>
    %106 = buffer %23#1, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer3"} : <i1>
    %107 = andi %106, %100#2 {handshake.bb = 1 : ui32, handshake.name = "andi11"} : <i1>
    %108:3 = fork [3] %107 {handshake.bb = 1 : ui32, handshake.name = "fork37"} : <i1>
    %109 = not %63#1 {handshake.bb = 1 : ui32, handshake.name = "not2"} : <i1>
    %110 = buffer %59#1, bufferType = FIFO_BREAK_NONE, numSlots = 5 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer8"} : <i12>
    %111 = passer %110[%105#1] {handshake.bb = 1 : ui32, handshake.name = "passer12"} : <i12>, <i1>
    %112 = buffer %8, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer0"} : <i11>
    %113:5 = fork [5] %112 {handshake.bb = 1 : ui32, handshake.name = "fork38"} : <i11>
    %114 = passer %115[%93#2] {handshake.bb = 1 : ui32, handshake.name = "passer19"} : <i11>, <i1>
    %115 = trunci %59#2 {handshake.bb = 1 : ui32, handshake.name = "trunci2"} : <i12> to <i11>
    %116 = buffer %93#1, bufferType = FIFO_BREAK_NONE, numSlots = 5 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer22"} : <i1>
    %117 = passer %120#4[%116] {handshake.bb = 1 : ui32, handshake.name = "passer20"} : <>, <i1>
    %118 = buffer %11, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer1"} : <>
    %119 = buffer %118, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer2"} : <>
    %120:7 = fork [7] %119 {handshake.bb = 1 : ui32, handshake.name = "fork39"} : <>
    %121 = passer %120#3[%105#2] {handshake.bb = 1 : ui32, handshake.name = "passer14"} : <>, <i1>
    %122 = passer %123[%105#0] {handshake.bb = 1 : ui32, handshake.name = "passer21"} : <i32>, <i1>
    %123 = extsi %52 {handshake.bb = 1 : ui32, handshake.name = "extsi13"} : <i1> to <i32>
    %124 = mux %126#0 [%30, %111] {handshake.bb = 2 : ui32, handshake.name = "mux1"} : <i1>, [<i12>, <i12>] to <i12>
    %125 = mux %126#1 [%32, %122] {handshake.bb = 2 : ui32, handshake.name = "mux2"} : <i1>, [<i32>, <i32>] to <i32>
    %result, %index = control_merge [%33, %121]  {handshake.bb = 2 : ui32, handshake.name = "control_merge3"} : [<>, <>] to <>, <i1>
    %126:2 = fork [2] %index {handshake.bb = 2 : ui32, handshake.name = "fork12"} : <i1>
    %127 = mux %131#0 [%44, %124] {handshake.bb = 3 : ui32, handshake.name = "mux3"} : <i1>, [<i12>, <i12>] to <i12>
    %128 = buffer %127, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 3 : ui32, handshake.name = "buffer29"} : <i12>
    %129 = extsi %128 {handshake.bb = 3 : ui32, handshake.name = "extsi14"} : <i12> to <i32>
    %130 = mux %131#1 [%48, %125] {handshake.bb = 3 : ui32, handshake.name = "mux4"} : <i1>, [<i32>, <i32>] to <i32>
    %result_4, %index_5 = control_merge [%49, %result]  {handshake.bb = 3 : ui32, handshake.name = "control_merge4"} : [<>, <>] to <>, <i1>
    %131:2 = fork [2] %index_5 {handshake.bb = 3 : ui32, handshake.name = "fork13"} : <i1>
    %132:2 = fork [2] %result_4 {handshake.bb = 3 : ui32, handshake.name = "fork14"} : <>
    %133 = source {handshake.bb = 3 : ui32, handshake.name = "source4"} : <>
    %134 = constant %133 {handshake.bb = 3 : ui32, handshake.name = "constant15", value = 1 : i2} : <>, <i2>
    %135 = extsi %134 {handshake.bb = 3 : ui32, handshake.name = "extsi6"} : <i2> to <i32>
    %136 = shli %129, %135 {handshake.bb = 3 : ui32, handshake.name = "shli0"} : <i32>
    %137 = buffer %130, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 3 : ui32, handshake.name = "buffer30"} : <i32>
    %138 = andi %136, %137 {handshake.bb = 3 : ui32, handshake.name = "andi0"} : <i32>
    end {handshake.bb = 3 : ui32, handshake.name = "end0"} %138, %memEnd_1, %memEnd, %0#1 : <i32>, <>, <>, <>
  }
}

