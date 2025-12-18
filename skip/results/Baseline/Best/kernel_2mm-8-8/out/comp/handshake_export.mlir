module {
  handshake.func @kernel_2mm(%arg0: !handshake.channel<i32>, %arg1: !handshake.channel<i32>, %arg2: memref<100xi32>, %arg3: memref<100xi32>, %arg4: memref<100xi32>, %arg5: memref<100xi32>, %arg6: memref<100xi32>, %arg7: !handshake.control<>, %arg8: !handshake.control<>, %arg9: !handshake.control<>, %arg10: !handshake.control<>, %arg11: !handshake.control<>, %arg12: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["alpha", "beta", "tmp", "A", "B", "C", "D", "tmp_start", "A_start", "B_start", "C_start", "D_start", "start"], resNames = ["tmp_end", "A_end", "B_end", "C_end", "D_end", "end"]} {
    %0:3 = fork [3] %arg12 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork0"} : <>
    %1:3 = lsq[%arg6 : memref<100xi32>] (%arg11, %207#0, %addressResult_50, %addressResult_52, %dataResult_53, %265#0, %addressResult_60, %addressResult_62, %dataResult_63, %362#4)  {groupSizes = [2 : i32, 2 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "5": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "6": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "7": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "8": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "9": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "lsq2"} : (!handshake.control<>, !handshake.control<>, !handshake.channel<i7>, !handshake.channel<i7>, !handshake.channel<i32>, !handshake.control<>, !handshake.channel<i7>, !handshake.channel<i7>, !handshake.channel<i32>, !handshake.control<>) -> (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>)
    %outputs, %memEnd = mem_controller[%arg5 : memref<100xi32>] %arg10 (%addressResult_58) %362#3 {connectedBlocks = [8 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller3"} :    (!handshake.channel<i7>) -> !handshake.channel<i32>
    %outputs_0, %memEnd_1 = mem_controller[%arg4 : memref<100xi32>] %arg9 (%addressResult_10) %362#2 {connectedBlocks = [3 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller4"} :    (!handshake.channel<i7>) -> !handshake.channel<i32>
    %outputs_2, %memEnd_3 = mem_controller[%arg3 : memref<100xi32>] %arg8 (%addressResult_8) %362#1 {connectedBlocks = [3 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller5"} :    (!handshake.channel<i7>) -> !handshake.channel<i32>
    %2:3 = lsq[%arg2 : memref<100xi32>] (%arg7, %29#0, %addressResult, %dataResult, %83#0, %addressResult_12, %addressResult_14, %dataResult_15, %265#2, %addressResult_56, %362#0)  {groupSizes = [1 : i32, 2 : i32, 1 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "10": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "5": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "6": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "7": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "8": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "9": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "lsq3"} : (!handshake.control<>, !handshake.control<>, !handshake.channel<i7>, !handshake.channel<i32>, !handshake.control<>, !handshake.channel<i7>, !handshake.channel<i7>, !handshake.channel<i32>, !handshake.control<>, !handshake.channel<i7>, !handshake.control<>) -> (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>)
    %3 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant0", value = false} : <>, <i1>
    %4 = extsi %3 {handshake.bb = 0 : ui32, handshake.name = "extsi30"} : <i1> to <i5>
    %5 = mux %9#0 [%4, %trueResult_36] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i5>, <i5>] to <i5>
    %6 = buffer %trueResult_38, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer51"} : <i32>
    %7 = mux %9#1 [%arg0, %6] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i1>, [<i32>, <i32>] to <i32>
    %8 = mux %9#2 [%arg1, %trueResult_40] {handshake.bb = 1 : ui32, handshake.name = "mux2"} : <i1>, [<i32>, <i32>] to <i32>
    %result, %index = control_merge [%0#2, %trueResult_42]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %9:3 = fork [3] %index {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <i1>
    %10:2 = fork [2] %result {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <>
    %11 = constant %10#0 {handshake.bb = 1 : ui32, handshake.name = "constant1", value = false} : <>, <i1>
    %12 = extsi %11 {handshake.bb = 1 : ui32, handshake.name = "extsi29"} : <i1> to <i5>
    %13 = buffer %7, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer1"} : <i32>
    %14 = buffer %8, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer2"} : <i32>
    %15 = buffer %5, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer0"} : <i5>
    %16 = mux %28#1 [%12, %trueResult_26] {handshake.bb = 2 : ui32, handshake.name = "mux3"} : <i1>, [<i5>, <i5>] to <i5>
    %17 = buffer %16, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer3"} : <i5>
    %18:2 = fork [2] %17 {handshake.bb = 2 : ui32, handshake.name = "fork3"} : <i5>
    %19 = extsi %18#0 {handshake.bb = 2 : ui32, handshake.name = "extsi31"} : <i5> to <i7>
    %20 = mux %28#2 [%13, %trueResult_28] {handshake.bb = 2 : ui32, handshake.name = "mux4"} : <i1>, [<i32>, <i32>] to <i32>
    %21 = mux %28#3 [%14, %trueResult_30] {handshake.bb = 2 : ui32, handshake.name = "mux5"} : <i1>, [<i32>, <i32>] to <i32>
    %22 = mux %28#0 [%15, %trueResult_32] {handshake.bb = 2 : ui32, handshake.name = "mux6"} : <i1>, [<i5>, <i5>] to <i5>
    %23 = buffer %22, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer7"} : <i5>
    %24 = buffer %23, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer8"} : <i5>
    %25:2 = fork [2] %24 {handshake.bb = 2 : ui32, handshake.name = "fork4"} : <i5>
    %26 = extsi %25#1 {handshake.bb = 2 : ui32, handshake.name = "extsi32"} : <i5> to <i32>
    %27:2 = fork [2] %26 {handshake.bb = 2 : ui32, handshake.name = "fork5"} : <i32>
    %result_4, %index_5 = control_merge [%10#1, %trueResult_34]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>, <>] to <>, <i1>
    %28:4 = fork [4] %index_5 {handshake.bb = 2 : ui32, handshake.name = "fork6"} : <i1>
    %29:3 = lazy_fork [3] %result_4 {handshake.bb = 2 : ui32, handshake.name = "lazy_fork0"} : <>
    %30 = buffer %29#2, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer10"} : <>
    %31 = constant %30 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"0": [1,inf], [0,inf], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "constant2", value = false} : <>, <i1>
    %32:2 = fork [2] %31 {handshake.bb = 2 : ui32, handshake.name = "fork7"} : <i1>
    %33 = extsi %32#1 {handshake.bb = 2 : ui32, handshake.name = "extsi3"} : <i1> to <i32>
    %34 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %35 = constant %34 {handshake.bb = 2 : ui32, handshake.name = "constant4", value = 1 : i2} : <>, <i2>
    %36 = extsi %35 {handshake.bb = 2 : ui32, handshake.name = "extsi4"} : <i2> to <i32>
    %37 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %38 = constant %37 {handshake.bb = 2 : ui32, handshake.name = "constant29", value = 3 : i3} : <>, <i3>
    %39 = extsi %38 {handshake.bb = 2 : ui32, handshake.name = "extsi5"} : <i3> to <i32>
    %40 = shli %27#0, %36 {handshake.bb = 2 : ui32, handshake.name = "shli0"} : <i32>
    %41 = buffer %40, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer12"} : <i32>
    %42 = trunci %41 {handshake.bb = 2 : ui32, handshake.name = "trunci0"} : <i32> to <i7>
    %43 = shli %27#1, %39 {handshake.bb = 2 : ui32, handshake.name = "shli1"} : <i32>
    %44 = buffer %43, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer13"} : <i32>
    %45 = trunci %44 {handshake.bb = 2 : ui32, handshake.name = "trunci1"} : <i32> to <i7>
    %46 = addi %42, %45 {handshake.bb = 2 : ui32, handshake.name = "addi19"} : <i7>
    %47 = buffer %46, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer14"} : <i7>
    %48 = addi %19, %47 {handshake.bb = 2 : ui32, handshake.name = "addi2"} : <i7>
    %49 = buffer %33, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer11"} : <i32>
    %50 = buffer %48, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer15"} : <i7>
    %addressResult, %dataResult = store[%50] %49 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["load2", 3], ["store1", 3], ["load4", 1]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 0>, handshake.name = "store0"} : <i7>, <i32>, <i7>, <i32>
    %51 = extsi %32#0 {handshake.bb = 2 : ui32, handshake.name = "extsi28"} : <i1> to <i5>
    %52 = buffer %20, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer4"} : <i32>
    %53 = buffer %52, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer5"} : <i32>
    %54 = buffer %21, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer6"} : <i32>
    %55 = buffer %29#1, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer9"} : <>
    %56 = mux %82#2 [%51, %trueResult] {handshake.bb = 3 : ui32, handshake.name = "mux7"} : <i1>, [<i5>, <i5>] to <i5>
    %57 = buffer %56, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer17"} : <i5>
    %58 = buffer %57, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer18"} : <i5>
    %59:3 = fork [3] %58 {handshake.bb = 3 : ui32, handshake.name = "fork8"} : <i5>
    %60 = extsi %59#0 {handshake.bb = 3 : ui32, handshake.name = "extsi33"} : <i5> to <i7>
    %61 = extsi %59#1 {handshake.bb = 3 : ui32, handshake.name = "extsi34"} : <i5> to <i6>
    %62 = extsi %59#2 {handshake.bb = 3 : ui32, handshake.name = "extsi35"} : <i5> to <i32>
    %63:2 = fork [2] %62 {handshake.bb = 3 : ui32, handshake.name = "fork9"} : <i32>
    %64 = mux %82#3 [%53, %trueResult_16] {handshake.bb = 3 : ui32, handshake.name = "mux8"} : <i1>, [<i32>, <i32>] to <i32>
    %65 = buffer %64, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer19"} : <i32>
    %66:2 = fork [2] %65 {handshake.bb = 3 : ui32, handshake.name = "fork10"} : <i32>
    %67 = buffer %54, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer16"} : <i32>
    %68 = mux %82#4 [%67, %trueResult_18] {handshake.bb = 3 : ui32, handshake.name = "mux9"} : <i1>, [<i32>, <i32>] to <i32>
    %69 = mux %82#0 [%25#0, %trueResult_20] {handshake.bb = 3 : ui32, handshake.name = "mux10"} : <i1>, [<i5>, <i5>] to <i5>
    %70 = buffer %69, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer23"} : <i5>
    %71 = buffer %70, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer24"} : <i5>
    %72:2 = fork [2] %71 {handshake.bb = 3 : ui32, handshake.name = "fork11"} : <i5>
    %73 = extsi %72#1 {handshake.bb = 3 : ui32, handshake.name = "extsi36"} : <i5> to <i32>
    %74:6 = fork [6] %73 {handshake.bb = 3 : ui32, handshake.name = "fork12"} : <i32>
    %75 = mux %82#1 [%18#1, %trueResult_22] {handshake.bb = 3 : ui32, handshake.name = "mux11"} : <i1>, [<i5>, <i5>] to <i5>
    %76 = buffer %75, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer25"} : <i5>
    %77 = buffer %76, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer26"} : <i5>
    %78:4 = fork [4] %77 {handshake.bb = 3 : ui32, handshake.name = "fork13"} : <i5>
    %79 = extsi %78#0 {handshake.bb = 3 : ui32, handshake.name = "extsi37"} : <i5> to <i7>
    %80 = extsi %78#1 {handshake.bb = 3 : ui32, handshake.name = "extsi38"} : <i5> to <i7>
    %81 = extsi %78#2 {handshake.bb = 3 : ui32, handshake.name = "extsi39"} : <i5> to <i7>
    %result_6, %index_7 = control_merge [%55, %trueResult_24]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>, <>] to <>, <i1>
    %82:5 = fork [5] %index_7 {handshake.bb = 3 : ui32, handshake.name = "fork14"} : <i1>
    %83:2 = lazy_fork [2] %result_6 {handshake.bb = 3 : ui32, handshake.name = "lazy_fork1"} : <>
    %84 = source {handshake.bb = 3 : ui32, handshake.name = "source2"} : <>
    %85 = constant %84 {handshake.bb = 3 : ui32, handshake.name = "constant30", value = 10 : i5} : <>, <i5>
    %86 = extsi %85 {handshake.bb = 3 : ui32, handshake.name = "extsi40"} : <i5> to <i6>
    %87 = source {handshake.bb = 3 : ui32, handshake.name = "source3"} : <>
    %88 = constant %87 {handshake.bb = 3 : ui32, handshake.name = "constant31", value = 1 : i2} : <>, <i2>
    %89:2 = fork [2] %88 {handshake.bb = 3 : ui32, handshake.name = "fork15"} : <i2>
    %90 = extsi %89#0 {handshake.bb = 3 : ui32, handshake.name = "extsi41"} : <i2> to <i6>
    %91 = extsi %89#1 {handshake.bb = 3 : ui32, handshake.name = "extsi7"} : <i2> to <i32>
    %92:4 = fork [4] %91 {handshake.bb = 3 : ui32, handshake.name = "fork16"} : <i32>
    %93 = source {handshake.bb = 3 : ui32, handshake.name = "source4"} : <>
    %94 = constant %93 {handshake.bb = 3 : ui32, handshake.name = "constant32", value = 3 : i3} : <>, <i3>
    %95 = extsi %94 {handshake.bb = 3 : ui32, handshake.name = "extsi8"} : <i3> to <i32>
    %96:4 = fork [4] %95 {handshake.bb = 3 : ui32, handshake.name = "fork17"} : <i32>
    %97 = shli %74#0, %92#0 {handshake.bb = 3 : ui32, handshake.name = "shli2"} : <i32>
    %98 = buffer %97, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer28"} : <i32>
    %99 = trunci %98 {handshake.bb = 3 : ui32, handshake.name = "trunci2"} : <i32> to <i7>
    %100 = shli %74#1, %96#0 {handshake.bb = 3 : ui32, handshake.name = "shli3"} : <i32>
    %101 = buffer %100, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer29"} : <i32>
    %102 = trunci %101 {handshake.bb = 3 : ui32, handshake.name = "trunci3"} : <i32> to <i7>
    %103 = addi %99, %102 {handshake.bb = 3 : ui32, handshake.name = "addi20"} : <i7>
    %104 = buffer %103, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer30"} : <i7>
    %105 = addi %60, %104 {handshake.bb = 3 : ui32, handshake.name = "addi3"} : <i7>
    %addressResult_8, %dataResult_9 = load[%105] %outputs_2 {handshake.bb = 3 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i7>, <i32>, <i7>, <i32>
    %106 = muli %66#1, %dataResult_9 {handshake.bb = 3 : ui32, handshake.name = "muli0"} : <i32>
    %107 = shli %63#0, %92#1 {handshake.bb = 3 : ui32, handshake.name = "shli4"} : <i32>
    %108 = buffer %107, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer31"} : <i32>
    %109 = trunci %108 {handshake.bb = 3 : ui32, handshake.name = "trunci4"} : <i32> to <i7>
    %110 = shli %63#1, %96#1 {handshake.bb = 3 : ui32, handshake.name = "shli5"} : <i32>
    %111 = buffer %110, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer32"} : <i32>
    %112 = trunci %111 {handshake.bb = 3 : ui32, handshake.name = "trunci5"} : <i32> to <i7>
    %113 = addi %109, %112 {handshake.bb = 3 : ui32, handshake.name = "addi21"} : <i7>
    %114 = buffer %113, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer33"} : <i7>
    %115 = addi %79, %114 {handshake.bb = 3 : ui32, handshake.name = "addi4"} : <i7>
    %addressResult_10, %dataResult_11 = load[%115] %outputs_0 {handshake.bb = 3 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i7>, <i32>, <i7>, <i32>
    %116 = muli %106, %dataResult_11 {handshake.bb = 3 : ui32, handshake.name = "muli1"} : <i32>
    %117 = shli %74#2, %92#2 {handshake.bb = 3 : ui32, handshake.name = "shli6"} : <i32>
    %118 = buffer %117, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer34"} : <i32>
    %119 = trunci %118 {handshake.bb = 3 : ui32, handshake.name = "trunci6"} : <i32> to <i7>
    %120 = shli %74#3, %96#2 {handshake.bb = 3 : ui32, handshake.name = "shli7"} : <i32>
    %121 = buffer %120, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer35"} : <i32>
    %122 = trunci %121 {handshake.bb = 3 : ui32, handshake.name = "trunci7"} : <i32> to <i7>
    %123 = addi %119, %122 {handshake.bb = 3 : ui32, handshake.name = "addi22"} : <i7>
    %124 = buffer %123, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer36"} : <i7>
    %125 = addi %80, %124 {handshake.bb = 3 : ui32, handshake.name = "addi5"} : <i7>
    %126 = buffer %125, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer37"} : <i7>
    %addressResult_12, %dataResult_13 = load[%126] %2#0 {handshake.bb = 3 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["store1", 3], ["store1", 4]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 1>, handshake.name = "load2"} : <i7>, <i32>, <i7>, <i32>
    %127 = addi %dataResult_13, %116 {handshake.bb = 3 : ui32, handshake.name = "addi0"} : <i32>
    %128 = shli %74#4, %92#3 {handshake.bb = 3 : ui32, handshake.name = "shli8"} : <i32>
    %129 = buffer %128, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer39"} : <i32>
    %130 = trunci %129 {handshake.bb = 3 : ui32, handshake.name = "trunci8"} : <i32> to <i7>
    %131 = shli %74#5, %96#3 {handshake.bb = 3 : ui32, handshake.name = "shli9"} : <i32>
    %132 = buffer %131, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer40"} : <i32>
    %133 = trunci %132 {handshake.bb = 3 : ui32, handshake.name = "trunci9"} : <i32> to <i7>
    %134 = addi %130, %133 {handshake.bb = 3 : ui32, handshake.name = "addi23"} : <i7>
    %135 = buffer %134, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer41"} : <i7>
    %136 = addi %81, %135 {handshake.bb = 3 : ui32, handshake.name = "addi6"} : <i7>
    %137 = buffer %127, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer38"} : <i32>
    %138 = buffer %136, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer42"} : <i7>
    %addressResult_14, %dataResult_15 = store[%138] %137 {handshake.bb = 3 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["load2", 3], ["store1", 3], ["load4", 1]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 1>, handshake.name = "store1"} : <i7>, <i32>, <i7>, <i32>
    %139 = addi %61, %90 {handshake.bb = 3 : ui32, handshake.name = "addi13"} : <i6>
    %140 = buffer %139, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer43"} : <i6>
    %141:2 = fork [2] %140 {handshake.bb = 3 : ui32, handshake.name = "fork18"} : <i6>
    %142 = trunci %141#0 {handshake.bb = 3 : ui32, handshake.name = "trunci10"} : <i6> to <i5>
    %143 = cmpi ult, %141#1, %86 {handshake.bb = 3 : ui32, handshake.name = "cmpi0"} : <i6>
    %144 = buffer %143, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer44"} : <i1>
    %145:6 = fork [6] %144 {handshake.bb = 3 : ui32, handshake.name = "fork19"} : <i1>
    %trueResult, %falseResult = cond_br %145#0, %142 {handshake.bb = 3 : ui32, handshake.name = "cond_br6"} : <i1>, <i5>
    sink %falseResult {handshake.name = "sink0"} : <i5>
    %146 = buffer %66#0, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer20"} : <i32>
    %trueResult_16, %falseResult_17 = cond_br %145#3, %146 {handshake.bb = 3 : ui32, handshake.name = "cond_br7"} : <i1>, <i32>
    %147 = buffer %68, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer21"} : <i32>
    %148 = buffer %147, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer22"} : <i32>
    %trueResult_18, %falseResult_19 = cond_br %145#4, %148 {handshake.bb = 3 : ui32, handshake.name = "cond_br8"} : <i1>, <i32>
    %trueResult_20, %falseResult_21 = cond_br %145#1, %72#0 {handshake.bb = 3 : ui32, handshake.name = "cond_br9"} : <i1>, <i5>
    %trueResult_22, %falseResult_23 = cond_br %145#2, %78#3 {handshake.bb = 3 : ui32, handshake.name = "cond_br10"} : <i1>, <i5>
    %149 = buffer %83#1, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer27"} : <>
    %trueResult_24, %falseResult_25 = cond_br %145#5, %149 {handshake.bb = 3 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,inf], [1,inf], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "cond_br11"} : <i1>, <>
    %150 = extsi %falseResult_23 {handshake.bb = 4 : ui32, handshake.name = "extsi42"} : <i5> to <i6>
    %151 = source {handshake.bb = 4 : ui32, handshake.name = "source5"} : <>
    %152 = constant %151 {handshake.bb = 4 : ui32, handshake.name = "constant33", value = 10 : i5} : <>, <i5>
    %153 = extsi %152 {handshake.bb = 4 : ui32, handshake.name = "extsi43"} : <i5> to <i6>
    %154 = source {handshake.bb = 4 : ui32, handshake.name = "source6"} : <>
    %155 = constant %154 {handshake.bb = 4 : ui32, handshake.name = "constant34", value = 1 : i2} : <>, <i2>
    %156 = extsi %155 {handshake.bb = 4 : ui32, handshake.name = "extsi44"} : <i2> to <i6>
    %157 = addi %150, %156 {handshake.bb = 4 : ui32, handshake.name = "addi14"} : <i6>
    %158 = buffer %157, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer45"} : <i6>
    %159:2 = fork [2] %158 {handshake.bb = 4 : ui32, handshake.name = "fork20"} : <i6>
    %160 = trunci %159#0 {handshake.bb = 4 : ui32, handshake.name = "trunci11"} : <i6> to <i5>
    %161 = cmpi ult, %159#1, %153 {handshake.bb = 4 : ui32, handshake.name = "cmpi1"} : <i6>
    %162 = buffer %161, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer46"} : <i1>
    %163:5 = fork [5] %162 {handshake.bb = 4 : ui32, handshake.name = "fork21"} : <i1>
    %trueResult_26, %falseResult_27 = cond_br %163#0, %160 {handshake.bb = 4 : ui32, handshake.name = "cond_br12"} : <i1>, <i5>
    sink %falseResult_27 {handshake.name = "sink2"} : <i5>
    %trueResult_28, %falseResult_29 = cond_br %163#2, %falseResult_17 {handshake.bb = 4 : ui32, handshake.name = "cond_br13"} : <i1>, <i32>
    %trueResult_30, %falseResult_31 = cond_br %163#3, %falseResult_19 {handshake.bb = 4 : ui32, handshake.name = "cond_br14"} : <i1>, <i32>
    %trueResult_32, %falseResult_33 = cond_br %163#1, %falseResult_21 {handshake.bb = 4 : ui32, handshake.name = "cond_br15"} : <i1>, <i5>
    %trueResult_34, %falseResult_35 = cond_br %163#4, %falseResult_25 {handshake.bb = 4 : ui32, handshake.name = "cond_br16"} : <i1>, <>
    %164 = extsi %falseResult_33 {handshake.bb = 5 : ui32, handshake.name = "extsi45"} : <i5> to <i6>
    %165:2 = fork [2] %falseResult_35 {handshake.bb = 5 : ui32, handshake.name = "fork22"} : <>
    %166 = constant %165#0 {handshake.bb = 5 : ui32, handshake.name = "constant35", value = false} : <>, <i1>
    %167 = source {handshake.bb = 5 : ui32, handshake.name = "source7"} : <>
    %168 = constant %167 {handshake.bb = 5 : ui32, handshake.name = "constant36", value = 10 : i5} : <>, <i5>
    %169 = extsi %168 {handshake.bb = 5 : ui32, handshake.name = "extsi46"} : <i5> to <i6>
    %170 = source {handshake.bb = 5 : ui32, handshake.name = "source8"} : <>
    %171 = constant %170 {handshake.bb = 5 : ui32, handshake.name = "constant37", value = 1 : i2} : <>, <i2>
    %172 = extsi %171 {handshake.bb = 5 : ui32, handshake.name = "extsi47"} : <i2> to <i6>
    %173 = buffer %164, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 5 : ui32, handshake.name = "buffer48"} : <i6>
    %174 = addi %173, %172 {handshake.bb = 5 : ui32, handshake.name = "addi15"} : <i6>
    %175 = buffer %174, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 5 : ui32, handshake.name = "buffer49"} : <i6>
    %176:2 = fork [2] %175 {handshake.bb = 5 : ui32, handshake.name = "fork23"} : <i6>
    %177 = trunci %176#0 {handshake.bb = 5 : ui32, handshake.name = "trunci12"} : <i6> to <i5>
    %178 = cmpi ult, %176#1, %169 {handshake.bb = 5 : ui32, handshake.name = "cmpi2"} : <i6>
    %179 = buffer %178, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 5 : ui32, handshake.name = "buffer50"} : <i1>
    %180:5 = fork [5] %179 {handshake.bb = 5 : ui32, handshake.name = "fork24"} : <i1>
    %trueResult_36, %falseResult_37 = cond_br %180#0, %177 {handshake.bb = 5 : ui32, handshake.name = "cond_br17"} : <i1>, <i5>
    sink %falseResult_37 {handshake.name = "sink4"} : <i5>
    %trueResult_38, %falseResult_39 = cond_br %180#1, %falseResult_29 {handshake.bb = 5 : ui32, handshake.name = "cond_br18"} : <i1>, <i32>
    sink %falseResult_39 {handshake.name = "sink5"} : <i32>
    %181 = buffer %falseResult_31, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 5 : ui32, handshake.name = "buffer47"} : <i32>
    %trueResult_40, %falseResult_41 = cond_br %180#2, %181 {handshake.bb = 5 : ui32, handshake.name = "cond_br19"} : <i1>, <i32>
    %trueResult_42, %falseResult_43 = cond_br %180#3, %165#1 {handshake.bb = 5 : ui32, handshake.name = "cond_br20"} : <i1>, <>
    %trueResult_44, %falseResult_45 = cond_br %180#4, %166 {handshake.bb = 5 : ui32, handshake.name = "cond_br21"} : <i1>, <i1>
    sink %trueResult_44 {handshake.name = "sink6"} : <i1>
    %182 = extsi %falseResult_45 {handshake.bb = 5 : ui32, handshake.name = "extsi27"} : <i1> to <i5>
    %183 = mux %185#0 [%182, %trueResult_82] {handshake.bb = 6 : ui32, handshake.name = "mux12"} : <i1>, [<i5>, <i5>] to <i5>
    %184 = mux %185#1 [%falseResult_41, %trueResult_84] {handshake.bb = 6 : ui32, handshake.name = "mux13"} : <i1>, [<i32>, <i32>] to <i32>
    %result_46, %index_47 = control_merge [%falseResult_43, %trueResult_86]  {handshake.bb = 6 : ui32, handshake.name = "control_merge5"} : [<>, <>] to <>, <i1>
    %185:2 = fork [2] %index_47 {handshake.bb = 6 : ui32, handshake.name = "fork25"} : <i1>
    %186:2 = fork [2] %result_46 {handshake.bb = 6 : ui32, handshake.name = "fork26"} : <>
    %187 = constant %186#0 {handshake.bb = 6 : ui32, handshake.name = "constant38", value = false} : <>, <i1>
    %188 = extsi %187 {handshake.bb = 6 : ui32, handshake.name = "extsi26"} : <i1> to <i5>
    %189 = buffer %184, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 6 : ui32, handshake.name = "buffer53"} : <i32>
    %190 = buffer %183, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 6 : ui32, handshake.name = "buffer52"} : <i5>
    %191 = mux %206#1 [%188, %trueResult_74] {handshake.bb = 7 : ui32, handshake.name = "mux14"} : <i1>, [<i5>, <i5>] to <i5>
    %192 = buffer %191, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 7 : ui32, handshake.name = "buffer54"} : <i5>
    %193:3 = fork [3] %192 {handshake.bb = 7 : ui32, handshake.name = "fork27"} : <i5>
    %194 = extsi %193#0 {handshake.bb = 7 : ui32, handshake.name = "extsi48"} : <i5> to <i7>
    %195 = extsi %193#1 {handshake.bb = 7 : ui32, handshake.name = "extsi49"} : <i5> to <i7>
    %196 = mux %206#2 [%189, %trueResult_76] {handshake.bb = 7 : ui32, handshake.name = "mux15"} : <i1>, [<i32>, <i32>] to <i32>
    %197 = buffer %196, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 7 : ui32, handshake.name = "buffer55"} : <i32>
    %198 = buffer %197, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 7 : ui32, handshake.name = "buffer56"} : <i32>
    %199:2 = fork [2] %198 {handshake.bb = 7 : ui32, handshake.name = "fork28"} : <i32>
    %200 = mux %206#0 [%190, %trueResult_78] {handshake.bb = 7 : ui32, handshake.name = "mux16"} : <i1>, [<i5>, <i5>] to <i5>
    %201 = buffer %200, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 7 : ui32, handshake.name = "buffer57"} : <i5>
    %202 = buffer %201, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 7 : ui32, handshake.name = "buffer58"} : <i5>
    %203:2 = fork [2] %202 {handshake.bb = 7 : ui32, handshake.name = "fork29"} : <i5>
    %204 = extsi %203#1 {handshake.bb = 7 : ui32, handshake.name = "extsi50"} : <i5> to <i32>
    %205:4 = fork [4] %204 {handshake.bb = 7 : ui32, handshake.name = "fork30"} : <i32>
    %result_48, %index_49 = control_merge [%186#1, %trueResult_80]  {handshake.bb = 7 : ui32, handshake.name = "control_merge6"} : [<>, <>] to <>, <i1>
    %206:3 = fork [3] %index_49 {handshake.bb = 7 : ui32, handshake.name = "fork31"} : <i1>
    %207:3 = lazy_fork [3] %result_48 {handshake.bb = 7 : ui32, handshake.name = "lazy_fork2"} : <>
    %208 = buffer %207#2, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 7 : ui32, handshake.name = "buffer60"} : <>
    %209 = constant %208 {handshake.bb = 7 : ui32, handshake.bufProps = #handshake<bufProps{"0": [1,inf], [0,inf], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "constant39", value = false} : <>, <i1>
    %210 = source {handshake.bb = 7 : ui32, handshake.name = "source9"} : <>
    %211 = constant %210 {handshake.bb = 7 : ui32, handshake.name = "constant40", value = 1 : i2} : <>, <i2>
    %212 = extsi %211 {handshake.bb = 7 : ui32, handshake.name = "extsi16"} : <i2> to <i32>
    %213:2 = fork [2] %212 {handshake.bb = 7 : ui32, handshake.name = "fork32"} : <i32>
    %214 = source {handshake.bb = 7 : ui32, handshake.name = "source10"} : <>
    %215 = constant %214 {handshake.bb = 7 : ui32, handshake.name = "constant41", value = 3 : i3} : <>, <i3>
    %216 = extsi %215 {handshake.bb = 7 : ui32, handshake.name = "extsi17"} : <i3> to <i32>
    %217:2 = fork [2] %216 {handshake.bb = 7 : ui32, handshake.name = "fork33"} : <i32>
    %218 = shli %205#0, %213#0 {handshake.bb = 7 : ui32, handshake.name = "shli10"} : <i32>
    %219 = buffer %218, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 7 : ui32, handshake.name = "buffer61"} : <i32>
    %220 = trunci %219 {handshake.bb = 7 : ui32, handshake.name = "trunci13"} : <i32> to <i7>
    %221 = shli %205#1, %217#0 {handshake.bb = 7 : ui32, handshake.name = "shli11"} : <i32>
    %222 = buffer %221, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 7 : ui32, handshake.name = "buffer62"} : <i32>
    %223 = trunci %222 {handshake.bb = 7 : ui32, handshake.name = "trunci14"} : <i32> to <i7>
    %224 = addi %220, %223 {handshake.bb = 7 : ui32, handshake.name = "addi24"} : <i7>
    %225 = buffer %224, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 7 : ui32, handshake.name = "buffer63"} : <i7>
    %226 = addi %194, %225 {handshake.bb = 7 : ui32, handshake.name = "addi7"} : <i7>
    %227 = buffer %226, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 7 : ui32, handshake.name = "buffer64"} : <i7>
    %addressResult_50, %dataResult_51 = load[%227] %1#0 {handshake.bb = 7 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["store2", 3], ["store3", 3]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 0>, handshake.name = "load3"} : <i7>, <i32>, <i7>, <i32>
    %228 = muli %dataResult_51, %199#1 {handshake.bb = 7 : ui32, handshake.name = "muli2"} : <i32>
    %229 = shli %205#2, %213#1 {handshake.bb = 7 : ui32, handshake.name = "shli12"} : <i32>
    %230 = buffer %229, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 7 : ui32, handshake.name = "buffer66"} : <i32>
    %231 = trunci %230 {handshake.bb = 7 : ui32, handshake.name = "trunci15"} : <i32> to <i7>
    %232 = shli %205#3, %217#1 {handshake.bb = 7 : ui32, handshake.name = "shli13"} : <i32>
    %233 = buffer %232, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 7 : ui32, handshake.name = "buffer67"} : <i32>
    %234 = trunci %233 {handshake.bb = 7 : ui32, handshake.name = "trunci16"} : <i32> to <i7>
    %235 = addi %231, %234 {handshake.bb = 7 : ui32, handshake.name = "addi25"} : <i7>
    %236 = buffer %235, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 7 : ui32, handshake.name = "buffer68"} : <i7>
    %237 = addi %195, %236 {handshake.bb = 7 : ui32, handshake.name = "addi8"} : <i7>
    %238 = buffer %228, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 7 : ui32, handshake.name = "buffer65"} : <i32>
    %239 = buffer %237, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 7 : ui32, handshake.name = "buffer69"} : <i7>
    %addressResult_52, %dataResult_53 = store[%239] %238 {handshake.bb = 7 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["load6", 3], ["store3", 3]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 0>, handshake.name = "store2"} : <i7>, <i32>, <i7>, <i32>
    %240 = extsi %209 {handshake.bb = 7 : ui32, handshake.name = "extsi25"} : <i1> to <i5>
    %241 = buffer %207#1, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 7 : ui32, handshake.name = "buffer59"} : <>
    %242 = mux %264#2 [%240, %trueResult_64] {handshake.bb = 8 : ui32, handshake.name = "mux17"} : <i1>, [<i5>, <i5>] to <i5>
    %243 = buffer %242, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 8 : ui32, handshake.name = "buffer70"} : <i5>
    %244 = buffer %243, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 8 : ui32, handshake.name = "buffer71"} : <i5>
    %245:3 = fork [3] %244 {handshake.bb = 8 : ui32, handshake.name = "fork34"} : <i5>
    %246 = extsi %245#0 {handshake.bb = 8 : ui32, handshake.name = "extsi51"} : <i5> to <i7>
    %247 = extsi %245#1 {handshake.bb = 8 : ui32, handshake.name = "extsi52"} : <i5> to <i6>
    %248 = extsi %245#2 {handshake.bb = 8 : ui32, handshake.name = "extsi53"} : <i5> to <i32>
    %249:2 = fork [2] %248 {handshake.bb = 8 : ui32, handshake.name = "fork35"} : <i32>
    %250 = mux %264#3 [%199#0, %trueResult_66] {handshake.bb = 8 : ui32, handshake.name = "mux18"} : <i1>, [<i32>, <i32>] to <i32>
    %251 = mux %264#0 [%203#0, %trueResult_68] {handshake.bb = 8 : ui32, handshake.name = "mux19"} : <i1>, [<i5>, <i5>] to <i5>
    %252 = buffer %251, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 8 : ui32, handshake.name = "buffer74"} : <i5>
    %253 = buffer %252, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 8 : ui32, handshake.name = "buffer75"} : <i5>
    %254:2 = fork [2] %253 {handshake.bb = 8 : ui32, handshake.name = "fork36"} : <i5>
    %255 = extsi %254#1 {handshake.bb = 8 : ui32, handshake.name = "extsi54"} : <i5> to <i32>
    %256:6 = fork [6] %255 {handshake.bb = 8 : ui32, handshake.name = "fork37"} : <i32>
    %257 = mux %264#1 [%193#2, %trueResult_70] {handshake.bb = 8 : ui32, handshake.name = "mux20"} : <i1>, [<i5>, <i5>] to <i5>
    %258 = buffer %257, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 8 : ui32, handshake.name = "buffer76"} : <i5>
    %259 = buffer %258, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 8 : ui32, handshake.name = "buffer77"} : <i5>
    %260:4 = fork [4] %259 {handshake.bb = 8 : ui32, handshake.name = "fork38"} : <i5>
    %261 = extsi %260#0 {handshake.bb = 8 : ui32, handshake.name = "extsi55"} : <i5> to <i7>
    %262 = extsi %260#1 {handshake.bb = 8 : ui32, handshake.name = "extsi56"} : <i5> to <i7>
    %263 = extsi %260#2 {handshake.bb = 8 : ui32, handshake.name = "extsi57"} : <i5> to <i7>
    %result_54, %index_55 = control_merge [%241, %trueResult_72]  {handshake.bb = 8 : ui32, handshake.name = "control_merge7"} : [<>, <>] to <>, <i1>
    %264:4 = fork [4] %index_55 {handshake.bb = 8 : ui32, handshake.name = "fork39"} : <i1>
    %265:3 = lazy_fork [3] %result_54 {handshake.bb = 8 : ui32, handshake.name = "lazy_fork3"} : <>
    %266 = source {handshake.bb = 8 : ui32, handshake.name = "source11"} : <>
    %267 = constant %266 {handshake.bb = 8 : ui32, handshake.name = "constant42", value = 10 : i5} : <>, <i5>
    %268 = extsi %267 {handshake.bb = 8 : ui32, handshake.name = "extsi58"} : <i5> to <i6>
    %269 = source {handshake.bb = 8 : ui32, handshake.name = "source12"} : <>
    %270 = constant %269 {handshake.bb = 8 : ui32, handshake.name = "constant43", value = 1 : i2} : <>, <i2>
    %271:2 = fork [2] %270 {handshake.bb = 8 : ui32, handshake.name = "fork40"} : <i2>
    %272 = extsi %271#0 {handshake.bb = 8 : ui32, handshake.name = "extsi59"} : <i2> to <i6>
    %273 = extsi %271#1 {handshake.bb = 8 : ui32, handshake.name = "extsi19"} : <i2> to <i32>
    %274:4 = fork [4] %273 {handshake.bb = 8 : ui32, handshake.name = "fork41"} : <i32>
    %275 = source {handshake.bb = 8 : ui32, handshake.name = "source13"} : <>
    %276 = constant %275 {handshake.bb = 8 : ui32, handshake.name = "constant44", value = 3 : i3} : <>, <i3>
    %277 = extsi %276 {handshake.bb = 8 : ui32, handshake.name = "extsi20"} : <i3> to <i32>
    %278:4 = fork [4] %277 {handshake.bb = 8 : ui32, handshake.name = "fork42"} : <i32>
    %279 = shli %256#0, %274#0 {handshake.bb = 8 : ui32, handshake.name = "shli14"} : <i32>
    %280 = buffer %279, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 8 : ui32, handshake.name = "buffer80"} : <i32>
    %281 = trunci %280 {handshake.bb = 8 : ui32, handshake.name = "trunci17"} : <i32> to <i7>
    %282 = shli %256#1, %278#0 {handshake.bb = 8 : ui32, handshake.name = "shli15"} : <i32>
    %283 = buffer %282, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 8 : ui32, handshake.name = "buffer81"} : <i32>
    %284 = trunci %283 {handshake.bb = 8 : ui32, handshake.name = "trunci18"} : <i32> to <i7>
    %285 = addi %281, %284 {handshake.bb = 8 : ui32, handshake.name = "addi26"} : <i7>
    %286 = buffer %285, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 8 : ui32, handshake.name = "buffer82"} : <i7>
    %287 = addi %246, %286 {handshake.bb = 8 : ui32, handshake.name = "addi9"} : <i7>
    %288 = buffer %287, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 8 : ui32, handshake.name = "buffer83"} : <i7>
    %addressResult_56, %dataResult_57 = load[%288] %2#1 {handshake.bb = 8 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<LSQ: 2>, handshake.name = "load4"} : <i7>, <i32>, <i7>, <i32>
    %289 = shli %249#0, %274#1 {handshake.bb = 8 : ui32, handshake.name = "shli16"} : <i32>
    %290 = buffer %289, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 8 : ui32, handshake.name = "buffer84"} : <i32>
    %291 = trunci %290 {handshake.bb = 8 : ui32, handshake.name = "trunci19"} : <i32> to <i7>
    %292 = shli %249#1, %278#1 {handshake.bb = 8 : ui32, handshake.name = "shli17"} : <i32>
    %293 = buffer %292, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 8 : ui32, handshake.name = "buffer85"} : <i32>
    %294 = trunci %293 {handshake.bb = 8 : ui32, handshake.name = "trunci20"} : <i32> to <i7>
    %295 = addi %291, %294 {handshake.bb = 8 : ui32, handshake.name = "addi27"} : <i7>
    %296 = buffer %295, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 8 : ui32, handshake.name = "buffer86"} : <i7>
    %297 = addi %261, %296 {handshake.bb = 8 : ui32, handshake.name = "addi10"} : <i7>
    %addressResult_58, %dataResult_59 = load[%297] %outputs {handshake.bb = 8 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load5"} : <i7>, <i32>, <i7>, <i32>
    %298 = muli %dataResult_57, %dataResult_59 {handshake.bb = 8 : ui32, handshake.name = "muli3"} : <i32>
    %299 = shli %256#2, %274#2 {handshake.bb = 8 : ui32, handshake.name = "shli18"} : <i32>
    %300 = buffer %299, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 8 : ui32, handshake.name = "buffer87"} : <i32>
    %301 = trunci %300 {handshake.bb = 8 : ui32, handshake.name = "trunci21"} : <i32> to <i7>
    %302 = shli %256#3, %278#2 {handshake.bb = 8 : ui32, handshake.name = "shli19"} : <i32>
    %303 = buffer %302, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 8 : ui32, handshake.name = "buffer88"} : <i32>
    %304 = trunci %303 {handshake.bb = 8 : ui32, handshake.name = "trunci22"} : <i32> to <i7>
    %305 = addi %301, %304 {handshake.bb = 8 : ui32, handshake.name = "addi28"} : <i7>
    %306 = buffer %305, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 8 : ui32, handshake.name = "buffer89"} : <i7>
    %307 = addi %262, %306 {handshake.bb = 8 : ui32, handshake.name = "addi11"} : <i7>
    %308 = buffer %307, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 8 : ui32, handshake.name = "buffer90"} : <i7>
    %addressResult_60, %dataResult_61 = load[%308] %1#1 {handshake.bb = 8 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["store3", 3], ["store3", 4]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 1>, handshake.name = "load6"} : <i7>, <i32>, <i7>, <i32>
    %309 = addi %dataResult_61, %298 {handshake.bb = 8 : ui32, handshake.name = "addi1"} : <i32>
    %310 = shli %256#4, %274#3 {handshake.bb = 8 : ui32, handshake.name = "shli20"} : <i32>
    %311 = buffer %310, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 8 : ui32, handshake.name = "buffer92"} : <i32>
    %312 = trunci %311 {handshake.bb = 8 : ui32, handshake.name = "trunci23"} : <i32> to <i7>
    %313 = shli %256#5, %278#3 {handshake.bb = 8 : ui32, handshake.name = "shli21"} : <i32>
    %314 = buffer %313, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 8 : ui32, handshake.name = "buffer93"} : <i32>
    %315 = trunci %314 {handshake.bb = 8 : ui32, handshake.name = "trunci24"} : <i32> to <i7>
    %316 = addi %312, %315 {handshake.bb = 8 : ui32, handshake.name = "addi29"} : <i7>
    %317 = buffer %316, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 8 : ui32, handshake.name = "buffer94"} : <i7>
    %318 = addi %263, %317 {handshake.bb = 8 : ui32, handshake.name = "addi12"} : <i7>
    %319 = buffer %309, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 8 : ui32, handshake.name = "buffer91"} : <i32>
    %320 = buffer %318, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 8 : ui32, handshake.name = "buffer95"} : <i7>
    %addressResult_62, %dataResult_63 = store[%320] %319 {handshake.bb = 8 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["load6", 3], ["store3", 3]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 1>, handshake.name = "store3"} : <i7>, <i32>, <i7>, <i32>
    %321 = addi %247, %272 {handshake.bb = 8 : ui32, handshake.name = "addi16"} : <i6>
    %322 = buffer %321, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 8 : ui32, handshake.name = "buffer96"} : <i6>
    %323:2 = fork [2] %322 {handshake.bb = 8 : ui32, handshake.name = "fork43"} : <i6>
    %324 = trunci %323#0 {handshake.bb = 8 : ui32, handshake.name = "trunci25"} : <i6> to <i5>
    %325 = cmpi ult, %323#1, %268 {handshake.bb = 8 : ui32, handshake.name = "cmpi3"} : <i6>
    %326 = buffer %325, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 8 : ui32, handshake.name = "buffer97"} : <i1>
    %327:5 = fork [5] %326 {handshake.bb = 8 : ui32, handshake.name = "fork44"} : <i1>
    %trueResult_64, %falseResult_65 = cond_br %327#0, %324 {handshake.bb = 8 : ui32, handshake.name = "cond_br22"} : <i1>, <i5>
    sink %falseResult_65 {handshake.name = "sink7"} : <i5>
    %328 = buffer %250, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 8 : ui32, handshake.name = "buffer72"} : <i32>
    %329 = buffer %328, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 8 : ui32, handshake.name = "buffer73"} : <i32>
    %trueResult_66, %falseResult_67 = cond_br %327#3, %329 {handshake.bb = 8 : ui32, handshake.name = "cond_br23"} : <i1>, <i32>
    %trueResult_68, %falseResult_69 = cond_br %327#1, %254#0 {handshake.bb = 8 : ui32, handshake.name = "cond_br24"} : <i1>, <i5>
    %trueResult_70, %falseResult_71 = cond_br %327#2, %260#3 {handshake.bb = 8 : ui32, handshake.name = "cond_br25"} : <i1>, <i5>
    %330 = buffer %265#1, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 8 : ui32, handshake.name = "buffer78"} : <>
    %331 = buffer %330, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 8 : ui32, handshake.name = "buffer79"} : <>
    %trueResult_72, %falseResult_73 = cond_br %327#4, %331 {handshake.bb = 8 : ui32, handshake.bufProps = #handshake<bufProps{"1": [1,inf], [1,inf], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "cond_br26"} : <i1>, <>
    %332 = extsi %falseResult_71 {handshake.bb = 9 : ui32, handshake.name = "extsi60"} : <i5> to <i6>
    %333 = source {handshake.bb = 9 : ui32, handshake.name = "source14"} : <>
    %334 = constant %333 {handshake.bb = 9 : ui32, handshake.name = "constant45", value = 10 : i5} : <>, <i5>
    %335 = extsi %334 {handshake.bb = 9 : ui32, handshake.name = "extsi61"} : <i5> to <i6>
    %336 = source {handshake.bb = 9 : ui32, handshake.name = "source15"} : <>
    %337 = constant %336 {handshake.bb = 9 : ui32, handshake.name = "constant46", value = 1 : i2} : <>, <i2>
    %338 = extsi %337 {handshake.bb = 9 : ui32, handshake.name = "extsi62"} : <i2> to <i6>
    %339 = addi %332, %338 {handshake.bb = 9 : ui32, handshake.name = "addi17"} : <i6>
    %340 = buffer %339, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 9 : ui32, handshake.name = "buffer99"} : <i6>
    %341:2 = fork [2] %340 {handshake.bb = 9 : ui32, handshake.name = "fork45"} : <i6>
    %342 = trunci %341#0 {handshake.bb = 9 : ui32, handshake.name = "trunci26"} : <i6> to <i5>
    %343 = cmpi ult, %341#1, %335 {handshake.bb = 9 : ui32, handshake.name = "cmpi4"} : <i6>
    %344 = buffer %343, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 9 : ui32, handshake.name = "buffer100"} : <i1>
    %345:4 = fork [4] %344 {handshake.bb = 9 : ui32, handshake.name = "fork46"} : <i1>
    %trueResult_74, %falseResult_75 = cond_br %345#0, %342 {handshake.bb = 9 : ui32, handshake.name = "cond_br27"} : <i1>, <i5>
    sink %falseResult_75 {handshake.name = "sink9"} : <i5>
    %trueResult_76, %falseResult_77 = cond_br %345#2, %falseResult_67 {handshake.bb = 9 : ui32, handshake.name = "cond_br28"} : <i1>, <i32>
    %346 = buffer %falseResult_69, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 9 : ui32, handshake.name = "buffer98"} : <i5>
    %trueResult_78, %falseResult_79 = cond_br %345#1, %346 {handshake.bb = 9 : ui32, handshake.name = "cond_br29"} : <i1>, <i5>
    %trueResult_80, %falseResult_81 = cond_br %345#3, %falseResult_73 {handshake.bb = 9 : ui32, handshake.name = "cond_br30"} : <i1>, <>
    %347 = extsi %falseResult_79 {handshake.bb = 10 : ui32, handshake.name = "extsi63"} : <i5> to <i6>
    %348 = source {handshake.bb = 10 : ui32, handshake.name = "source16"} : <>
    %349 = constant %348 {handshake.bb = 10 : ui32, handshake.name = "constant47", value = 10 : i5} : <>, <i5>
    %350 = extsi %349 {handshake.bb = 10 : ui32, handshake.name = "extsi64"} : <i5> to <i6>
    %351 = source {handshake.bb = 10 : ui32, handshake.name = "source17"} : <>
    %352 = constant %351 {handshake.bb = 10 : ui32, handshake.name = "constant48", value = 1 : i2} : <>, <i2>
    %353 = extsi %352 {handshake.bb = 10 : ui32, handshake.name = "extsi65"} : <i2> to <i6>
    %354 = addi %347, %353 {handshake.bb = 10 : ui32, handshake.name = "addi18"} : <i6>
    %355 = buffer %354, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 10 : ui32, handshake.name = "buffer102"} : <i6>
    %356:2 = fork [2] %355 {handshake.bb = 10 : ui32, handshake.name = "fork47"} : <i6>
    %357 = trunci %356#0 {handshake.bb = 10 : ui32, handshake.name = "trunci27"} : <i6> to <i5>
    %358 = cmpi ult, %356#1, %350 {handshake.bb = 10 : ui32, handshake.name = "cmpi5"} : <i6>
    %359 = buffer %358, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 10 : ui32, handshake.name = "buffer103"} : <i1>
    %360:3 = fork [3] %359 {handshake.bb = 10 : ui32, handshake.name = "fork48"} : <i1>
    %trueResult_82, %falseResult_83 = cond_br %360#0, %357 {handshake.bb = 10 : ui32, handshake.name = "cond_br31"} : <i1>, <i5>
    sink %falseResult_83 {handshake.name = "sink11"} : <i5>
    %361 = buffer %falseResult_77, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 10 : ui32, handshake.name = "buffer101"} : <i32>
    %trueResult_84, %falseResult_85 = cond_br %360#1, %361 {handshake.bb = 10 : ui32, handshake.name = "cond_br32"} : <i1>, <i32>
    sink %falseResult_85 {handshake.name = "sink12"} : <i32>
    %trueResult_86, %falseResult_87 = cond_br %360#2, %falseResult_81 {handshake.bb = 10 : ui32, handshake.name = "cond_br33"} : <i1>, <>
    %362:5 = fork [5] %falseResult_87 {handshake.bb = 11 : ui32, handshake.name = "fork49"} : <>
    end {handshake.bb = 11 : ui32, handshake.name = "end0"} %2#2, %memEnd_3, %memEnd_1, %memEnd, %1#2, %0#1 : <>, <>, <>, <>, <>, <>
  }
}

