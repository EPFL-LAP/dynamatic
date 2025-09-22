module {
  handshake.func @collision_donut(%arg0: memref<1000xi32>, %arg1: memref<1000xi32>, %arg2: !handshake.control<>, %arg3: !handshake.control<>, %arg4: !handshake.control<>, ...) -> (!handshake.channel<i32>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["x", "y", "x_start", "y_start", "start"], handshake.cfdfcThroughput = #handshake<cfdfcThroughput {"0" = 0.16666666666666666 : f64}>, handshake.cfdfcToBBList = #handshake<cfdfcToBBList {"0" = [1 : ui32]}>, resNames = ["out0", "x_end", "y_end", "end"]} {
    %0:3 = fork [3] %arg4 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork0"} : <>
    %outputs, %memEnd = mem_controller[%arg1 : memref<1000xi32>] %arg3 (%addressResult_2) %88#1 {connectedBlocks = [1 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller2"} :    (!handshake.channel<i10>) -> !handshake.channel<i32>
    %outputs_0, %memEnd_1 = mem_controller[%arg0 : memref<1000xi32>] %arg2 (%addressResult) %88#0 {connectedBlocks = [1 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller3"} :    (!handshake.channel<i10>) -> !handshake.channel<i32>
    %1 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant0", value = false} : <>, <i1>
    %2 = extsi %1 {handshake.bb = 0 : ui32, handshake.name = "extsi7"} : <i1> to <i11>
    %3 = buffer %70#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer10"} : <i1>
    %4 = init %3 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "init0", initToken = 0 : ui1} : <i1>
    %5 = buffer %70#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer9"} : <i1>
    %6 = init %5 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "init1", initToken = 0 : ui1} : <i1>
    %7 = mux %4 [%2, %73] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i11>, <i11>] to <i11>
    %8 = buffer %7, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer0"} : <i11>
    %9:5 = fork [5] %8 {handshake.bb = 1 : ui32, handshake.name = "fork31"} : <i11>
    %10 = trunci %9#0 {handshake.bb = 1 : ui32, handshake.name = "trunci0"} : <i11> to <i10>
    %11 = trunci %9#1 {handshake.bb = 1 : ui32, handshake.name = "trunci1"} : <i11> to <i10>
    %12 = mux %6 [%0#2, %75] {handshake.bb = 1 : ui32, handshake.name = "mux7"} : <i1>, [<>, <>] to <>
    %13 = buffer %12, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer3"} : <>
    %14 = buffer %13, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer4"} : <>
    %15:7 = fork [7] %14 {handshake.bb = 1 : ui32, handshake.name = "fork32"} : <>
    %16 = source {handshake.bb = 1 : ui32, handshake.name = "source0"} : <>
    %17 = constant %16 {handshake.bb = 1 : ui32, handshake.name = "constant1", value = 4 : i4} : <>, <i4>
    %18 = extsi %17 {handshake.bb = 1 : ui32, handshake.name = "extsi1"} : <i4> to <i32>
    %19 = constant %15#0 {handshake.bb = 1 : ui32, handshake.name = "constant8", value = -1 : i32} : <>, <i32>
    %addressResult, %dataResult = load[%11] %outputs_0 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i10>, <i32>, <i10>, <i32>
    %20:2 = fork [2] %dataResult {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <i32>
    %addressResult_2, %dataResult_3 = load[%10] %outputs {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i10>, <i32>, <i10>, <i32>
    %21:2 = fork [2] %dataResult_3 {handshake.bb = 1 : ui32, handshake.name = "fork4"} : <i32>
    %22 = muli %20#0, %20#1 {handshake.bb = 1 : ui32, handshake.name = "muli0"} : <i32>
    %23 = muli %21#0, %21#1 {handshake.bb = 1 : ui32, handshake.name = "muli1"} : <i32>
    %24 = addi %22, %23 {handshake.bb = 1 : ui32, handshake.name = "addi0"} : <i32>
    %25:2 = fork [2] %24 {handshake.bb = 1 : ui32, handshake.name = "fork5"} : <i32>
    %26 = cmpi ult, %25#1, %18 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i32>
    %27 = buffer %26, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer5"} : <i1>
    %28:4 = fork [4] %27 {handshake.bb = 1 : ui32, handshake.name = "fork33"} : <i1>
    %29 = not %28#3 {handshake.bb = 1 : ui32, handshake.name = "not0"} : <i1>
    %30:2 = fork [2] %29 {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <i1>
    %31 = andi %46#1, %30#0 {handshake.bb = 1 : ui32, handshake.name = "andi1"} : <i1>
    %32:3 = fork [3] %31 {handshake.bb = 1 : ui32, handshake.name = "fork34"} : <i1>
    %33 = andi %71, %49#0 {handshake.bb = 1 : ui32, handshake.name = "andi2"} : <i1>
    %34:3 = fork [3] %33 {handshake.bb = 1 : ui32, handshake.name = "fork35"} : <i1>
    %35 = buffer %37, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer6"} : <i12>
    %36 = passer %35[%28#2] {handshake.bb = 1 : ui32, handshake.name = "passer16"} : <i12>, <i1>
    %37 = extsi %9#2 {handshake.bb = 1 : ui32, handshake.name = "extsi8"} : <i11> to <i12>
    %38 = passer %19[%28#1] {handshake.bb = 1 : ui32, handshake.name = "passer2"} : <i32>, <i1>
    %39 = passer %15#1[%28#0] {handshake.bb = 1 : ui32, handshake.name = "passer3"} : <>, <i1>
    %40 = source {handshake.bb = 1 : ui32, handshake.name = "source1"} : <>
    %41 = constant %40 {handshake.bb = 1 : ui32, handshake.name = "constant2", value = 19000 : i16} : <>, <i16>
    %42 = extsi %41 {handshake.bb = 1 : ui32, handshake.name = "extsi2"} : <i16> to <i32>
    %43 = constant %15#2 {handshake.bb = 1 : ui32, handshake.name = "constant10", value = -2 : i32} : <>, <i32>
    %44 = cmpi ugt, %25#0, %42 {handshake.bb = 1 : ui32, handshake.name = "cmpi1"} : <i32>
    %45 = buffer %44, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer7"} : <i1>
    %46:2 = fork [2] %45 {handshake.bb = 1 : ui32, handshake.name = "fork10"} : <i1>
    %47 = not %46#0 {handshake.bb = 1 : ui32, handshake.name = "not1"} : <i1>
    %48 = andi %30#1, %47 {handshake.bb = 1 : ui32, handshake.name = "andi3"} : <i1>
    %49:2 = fork [2] %48 {handshake.bb = 1 : ui32, handshake.name = "fork11"} : <i1>
    %50 = passer %52[%32#2] {handshake.bb = 1 : ui32, handshake.name = "passer17"} : <i12>, <i1>
    %51 = buffer %9#4, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer2"} : <i11>
    %52 = extsi %51 {handshake.bb = 1 : ui32, handshake.name = "extsi9"} : <i11> to <i12>
    %53 = passer %43[%32#0] {handshake.bb = 1 : ui32, handshake.name = "passer8"} : <i32>, <i1>
    %54 = passer %15#3[%32#1] {handshake.bb = 1 : ui32, handshake.name = "passer9"} : <>, <i1>
    %55 = buffer %9#3, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer1"} : <i11>
    %56 = extsi %55 {handshake.bb = 1 : ui32, handshake.name = "extsi10"} : <i11> to <i12>
    %57 = constant %15#4 {handshake.bb = 1 : ui32, handshake.name = "constant3", value = false} : <>, <i1>
    %58 = source {handshake.bb = 1 : ui32, handshake.name = "source2"} : <>
    %59 = constant %58 {handshake.bb = 1 : ui32, handshake.name = "constant4", value = 1 : i2} : <>, <i2>
    %60 = extsi %59 {handshake.bb = 1 : ui32, handshake.name = "extsi11"} : <i2> to <i12>
    %61 = source {handshake.bb = 1 : ui32, handshake.name = "source3"} : <>
    %62 = constant %61 {handshake.bb = 1 : ui32, handshake.name = "constant5", value = 1000 : i11} : <>, <i11>
    %63 = extsi %62 {handshake.bb = 1 : ui32, handshake.name = "extsi12"} : <i11> to <i12>
    %64 = buffer %66, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "buffer8"} : <i12>
    %65:3 = fork [3] %64 {handshake.bb = 1 : ui32, handshake.name = "fork28"} : <i12>
    %66 = addi %56, %60 {handshake.bb = 1 : ui32, handshake.name = "addi1"} : <i12>
    %67 = cmpi ult, %65#0, %63 {handshake.bb = 1 : ui32, handshake.name = "cmpi2"} : <i12>
    %68:2 = fork [2] %67 {handshake.bb = 1 : ui32, handshake.name = "fork29"} : <i1>
    %69 = andi %49#1, %68#0 {handshake.bb = 1 : ui32, handshake.name = "andi4"} : <i1>
    %70:4 = fork [4] %69 {handshake.bb = 1 : ui32, handshake.name = "fork30"} : <i1>
    %71 = not %68#1 {handshake.bb = 1 : ui32, handshake.name = "not2"} : <i1>
    %72 = passer %65#1[%34#0] {handshake.bb = 1 : ui32, handshake.name = "passer12"} : <i12>, <i1>
    %73 = passer %74[%70#2] {handshake.bb = 1 : ui32, handshake.name = "passer18"} : <i11>, <i1>
    %74 = trunci %65#2 {handshake.bb = 1 : ui32, handshake.name = "trunci2"} : <i12> to <i11>
    %75 = passer %15#6[%70#3] {handshake.bb = 1 : ui32, handshake.name = "passer13"} : <>, <i1>
    %76 = passer %15#5[%34#2] {handshake.bb = 1 : ui32, handshake.name = "passer14"} : <>, <i1>
    %77 = passer %78[%34#1] {handshake.bb = 1 : ui32, handshake.name = "passer19"} : <i32>, <i1>
    %78 = extsi %57 {handshake.bb = 1 : ui32, handshake.name = "extsi13"} : <i1> to <i32>
    %79 = mux %81#0 [%36, %72] {handshake.bb = 2 : ui32, handshake.name = "mux1"} : <i1>, [<i12>, <i12>] to <i12>
    %80 = mux %81#1 [%38, %77] {handshake.bb = 2 : ui32, handshake.name = "mux2"} : <i1>, [<i32>, <i32>] to <i32>
    %result, %index = control_merge [%39, %76]  {handshake.bb = 2 : ui32, handshake.name = "control_merge3"} : [<>, <>] to <>, <i1>
    %81:2 = fork [2] %index {handshake.bb = 2 : ui32, handshake.name = "fork12"} : <i1>
    %82 = mux %87#0 [%50, %79] {handshake.bb = 3 : ui32, handshake.name = "mux3"} : <i1>, [<i12>, <i12>] to <i12>
    %83 = buffer %82, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 3 : ui32, handshake.name = "buffer12"} : <i12>
    %84 = extsi %83 {handshake.bb = 3 : ui32, handshake.name = "extsi14"} : <i12> to <i32>
    %85 = buffer %80, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 3 : ui32, handshake.name = "buffer11"} : <i32>
    %86 = mux %87#1 [%53, %85] {handshake.bb = 3 : ui32, handshake.name = "mux4"} : <i1>, [<i32>, <i32>] to <i32>
    %result_4, %index_5 = control_merge [%54, %result]  {handshake.bb = 3 : ui32, handshake.name = "control_merge4"} : [<>, <>] to <>, <i1>
    %87:2 = fork [2] %index_5 {handshake.bb = 3 : ui32, handshake.name = "fork13"} : <i1>
    %88:2 = fork [2] %result_4 {handshake.bb = 3 : ui32, handshake.name = "fork14"} : <>
    %89 = source {handshake.bb = 3 : ui32, handshake.name = "source4"} : <>
    %90 = constant %89 {handshake.bb = 3 : ui32, handshake.name = "constant15", value = 1 : i2} : <>, <i2>
    %91 = extsi %90 {handshake.bb = 3 : ui32, handshake.name = "extsi6"} : <i2> to <i32>
    %92 = shli %84, %91 {handshake.bb = 3 : ui32, handshake.name = "shli0"} : <i32>
    %93 = andi %92, %86 {handshake.bb = 3 : ui32, handshake.name = "andi0"} : <i32>
    end {handshake.bb = 3 : ui32, handshake.name = "end0"} %93, %memEnd_1, %memEnd, %0#1 : <i32>, <>, <>, <>
  }
}

