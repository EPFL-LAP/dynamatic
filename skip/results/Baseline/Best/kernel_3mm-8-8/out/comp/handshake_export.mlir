module {
  handshake.func @kernel_3mm(%arg0: memref<100xi32>, %arg1: memref<100xi32>, %arg2: memref<100xi32>, %arg3: memref<100xi32>, %arg4: memref<100xi32>, %arg5: memref<100xi32>, %arg6: memref<100xi32>, %arg7: !handshake.control<>, %arg8: !handshake.control<>, %arg9: !handshake.control<>, %arg10: !handshake.control<>, %arg11: !handshake.control<>, %arg12: !handshake.control<>, %arg13: !handshake.control<>, %arg14: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["A", "B", "C", "D", "E", "F", "G", "A_start", "B_start", "C_start", "D_start", "E_start", "F_start", "G_start", "start"], resNames = ["A_end", "B_end", "C_end", "D_end", "E_end", "F_end", "G_end", "end"]} {
    %0:3 = fork [3] %arg14 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork0"} : <>
    %1:2 = lsq[%arg6 : memref<100xi32>] (%arg13, %336#0, %addressResult_76, %dataResult_77, %382#0, %addressResult_84, %addressResult_86, %dataResult_87, %481#6)  {groupSizes = [1 : i32, 2 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "5": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "6": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "7": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "8": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "lsq3"} : (!handshake.control<>, !handshake.control<>, !handshake.channel<i7>, !handshake.channel<i32>, !handshake.control<>, !handshake.channel<i7>, !handshake.channel<i7>, !handshake.channel<i32>, !handshake.control<>) -> (!handshake.channel<i32>, !handshake.control<>)
    %2:3 = lsq[%arg5 : memref<100xi32>] (%arg12, %179#0, %addressResult_40, %dataResult_41, %225#0, %addressResult_48, %addressResult_50, %dataResult_51, %384#1, %addressResult_82, %481#5)  {groupSizes = [1 : i32, 2 : i32, 1 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "10": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "5": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "6": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "7": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "8": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "9": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "lsq4"} : (!handshake.control<>, !handshake.control<>, !handshake.channel<i7>, !handshake.channel<i32>, !handshake.control<>, !handshake.channel<i7>, !handshake.channel<i7>, !handshake.channel<i32>, !handshake.control<>, !handshake.channel<i7>, !handshake.control<>) -> (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>)
    %3:3 = lsq[%arg4 : memref<100xi32>] (%arg11, %22#0, %addressResult, %dataResult, %68#0, %addressResult_14, %addressResult_16, %dataResult_17, %384#0, %addressResult_80, %481#4)  {groupSizes = [1 : i32, 2 : i32, 1 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "10": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "5": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "6": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "7": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "8": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "9": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "lsq5"} : (!handshake.control<>, !handshake.control<>, !handshake.channel<i7>, !handshake.channel<i32>, !handshake.control<>, !handshake.channel<i7>, !handshake.channel<i7>, !handshake.channel<i32>, !handshake.control<>, !handshake.channel<i7>, !handshake.control<>) -> (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>)
    %outputs, %memEnd = mem_controller[%arg3 : memref<100xi32>] %arg10 (%addressResult_46) %481#3 {connectedBlocks = [8 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller4"} :    (!handshake.channel<i7>) -> !handshake.channel<i32>
    %outputs_0, %memEnd_1 = mem_controller[%arg2 : memref<100xi32>] %arg9 (%addressResult_44) %481#2 {connectedBlocks = [8 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller5"} :    (!handshake.channel<i7>) -> !handshake.channel<i32>
    %outputs_2, %memEnd_3 = mem_controller[%arg1 : memref<100xi32>] %arg8 (%addressResult_12) %481#1 {connectedBlocks = [3 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller6"} :    (!handshake.channel<i7>) -> !handshake.channel<i32>
    %outputs_4, %memEnd_5 = mem_controller[%arg0 : memref<100xi32>] %arg7 (%addressResult_10) %481#0 {connectedBlocks = [3 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller7"} :    (!handshake.channel<i7>) -> !handshake.channel<i32>
    %4 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant0", value = false} : <>, <i1>
    %5 = extsi %4 {handshake.bb = 0 : ui32, handshake.name = "extsi47"} : <i1> to <i5>
    %6 = mux %index [%5, %trueResult_30] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i5>, <i5>] to <i5>
    %result, %index = control_merge [%0#2, %trueResult_32]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %7:2 = fork [2] %result {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <>
    %8 = constant %7#0 {handshake.bb = 1 : ui32, handshake.name = "constant1", value = false} : <>, <i1>
    %9 = extsi %8 {handshake.bb = 1 : ui32, handshake.name = "extsi46"} : <i1> to <i5>
    %10 = buffer %6, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer0"} : <i5>
    %11 = mux %21#1 [%9, %trueResult_24] {handshake.bb = 2 : ui32, handshake.name = "mux1"} : <i1>, [<i5>, <i5>] to <i5>
    %12 = buffer %11, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer1"} : <i5>
    %13:2 = fork [2] %12 {handshake.bb = 2 : ui32, handshake.name = "fork2"} : <i5>
    %14 = extsi %13#0 {handshake.bb = 2 : ui32, handshake.name = "extsi48"} : <i5> to <i7>
    %15 = mux %21#0 [%10, %trueResult_26] {handshake.bb = 2 : ui32, handshake.name = "mux2"} : <i1>, [<i5>, <i5>] to <i5>
    %16 = buffer %15, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer2"} : <i5>
    %17 = buffer %16, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer3"} : <i5>
    %18:2 = fork [2] %17 {handshake.bb = 2 : ui32, handshake.name = "fork3"} : <i5>
    %19 = extsi %18#1 {handshake.bb = 2 : ui32, handshake.name = "extsi49"} : <i5> to <i32>
    %20:2 = fork [2] %19 {handshake.bb = 2 : ui32, handshake.name = "fork4"} : <i32>
    %result_6, %index_7 = control_merge [%7#1, %trueResult_28]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>, <>] to <>, <i1>
    %21:2 = fork [2] %index_7 {handshake.bb = 2 : ui32, handshake.name = "fork5"} : <i1>
    %22:3 = lazy_fork [3] %result_6 {handshake.bb = 2 : ui32, handshake.name = "lazy_fork0"} : <>
    %23 = buffer %22#2, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer5"} : <>
    %24 = constant %23 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"0": [1,inf], [0,inf], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "constant2", value = false} : <>, <i1>
    %25:2 = fork [2] %24 {handshake.bb = 2 : ui32, handshake.name = "fork6"} : <i1>
    %26 = extsi %25#1 {handshake.bb = 2 : ui32, handshake.name = "extsi3"} : <i1> to <i32>
    %27 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %28 = constant %27 {handshake.bb = 2 : ui32, handshake.name = "constant4", value = 1 : i2} : <>, <i2>
    %29 = extsi %28 {handshake.bb = 2 : ui32, handshake.name = "extsi4"} : <i2> to <i32>
    %30 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %31 = constant %30 {handshake.bb = 2 : ui32, handshake.name = "constant43", value = 3 : i3} : <>, <i3>
    %32 = extsi %31 {handshake.bb = 2 : ui32, handshake.name = "extsi5"} : <i3> to <i32>
    %33 = shli %20#0, %29 {handshake.bb = 2 : ui32, handshake.name = "shli0"} : <i32>
    %34 = buffer %33, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer7"} : <i32>
    %35 = trunci %34 {handshake.bb = 2 : ui32, handshake.name = "trunci0"} : <i32> to <i7>
    %36 = shli %20#1, %32 {handshake.bb = 2 : ui32, handshake.name = "shli1"} : <i32>
    %37 = buffer %36, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer8"} : <i32>
    %38 = trunci %37 {handshake.bb = 2 : ui32, handshake.name = "trunci1"} : <i32> to <i7>
    %39 = addi %35, %38 {handshake.bb = 2 : ui32, handshake.name = "addi27"} : <i7>
    %40 = buffer %39, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer9"} : <i7>
    %41 = addi %14, %40 {handshake.bb = 2 : ui32, handshake.name = "addi3"} : <i7>
    %42 = buffer %26, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer6"} : <i32>
    %43 = buffer %41, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer10"} : <i7>
    %addressResult, %dataResult = store[%43] %42 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["load2", 3], ["store1", 3], ["load6", 1]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 0>, handshake.name = "store0"} : <i7>, <i32>, <i7>, <i32>
    %44 = extsi %25#0 {handshake.bb = 2 : ui32, handshake.name = "extsi45"} : <i1> to <i5>
    %45 = buffer %22#1, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer4"} : <>
    %46 = mux %67#2 [%44, %trueResult] {handshake.bb = 3 : ui32, handshake.name = "mux3"} : <i1>, [<i5>, <i5>] to <i5>
    %47 = buffer %46, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer11"} : <i5>
    %48 = buffer %47, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer12"} : <i5>
    %49:3 = fork [3] %48 {handshake.bb = 3 : ui32, handshake.name = "fork7"} : <i5>
    %50 = extsi %49#0 {handshake.bb = 3 : ui32, handshake.name = "extsi50"} : <i5> to <i7>
    %51 = extsi %49#1 {handshake.bb = 3 : ui32, handshake.name = "extsi51"} : <i5> to <i6>
    %52 = extsi %49#2 {handshake.bb = 3 : ui32, handshake.name = "extsi52"} : <i5> to <i32>
    %53:2 = fork [2] %52 {handshake.bb = 3 : ui32, handshake.name = "fork8"} : <i32>
    %54 = mux %67#0 [%18#0, %trueResult_18] {handshake.bb = 3 : ui32, handshake.name = "mux4"} : <i1>, [<i5>, <i5>] to <i5>
    %55 = buffer %54, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer13"} : <i5>
    %56 = buffer %55, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer14"} : <i5>
    %57:2 = fork [2] %56 {handshake.bb = 3 : ui32, handshake.name = "fork9"} : <i5>
    %58 = extsi %57#1 {handshake.bb = 3 : ui32, handshake.name = "extsi53"} : <i5> to <i32>
    %59:6 = fork [6] %58 {handshake.bb = 3 : ui32, handshake.name = "fork10"} : <i32>
    %60 = mux %67#1 [%13#1, %trueResult_20] {handshake.bb = 3 : ui32, handshake.name = "mux5"} : <i1>, [<i5>, <i5>] to <i5>
    %61 = buffer %60, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer15"} : <i5>
    %62 = buffer %61, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer16"} : <i5>
    %63:4 = fork [4] %62 {handshake.bb = 3 : ui32, handshake.name = "fork11"} : <i5>
    %64 = extsi %63#0 {handshake.bb = 3 : ui32, handshake.name = "extsi54"} : <i5> to <i7>
    %65 = extsi %63#1 {handshake.bb = 3 : ui32, handshake.name = "extsi55"} : <i5> to <i7>
    %66 = extsi %63#2 {handshake.bb = 3 : ui32, handshake.name = "extsi56"} : <i5> to <i7>
    %result_8, %index_9 = control_merge [%45, %trueResult_22]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>, <>] to <>, <i1>
    %67:3 = fork [3] %index_9 {handshake.bb = 3 : ui32, handshake.name = "fork12"} : <i1>
    %68:2 = lazy_fork [2] %result_8 {handshake.bb = 3 : ui32, handshake.name = "lazy_fork1"} : <>
    %69 = source {handshake.bb = 3 : ui32, handshake.name = "source2"} : <>
    %70 = constant %69 {handshake.bb = 3 : ui32, handshake.name = "constant44", value = 10 : i5} : <>, <i5>
    %71 = extsi %70 {handshake.bb = 3 : ui32, handshake.name = "extsi57"} : <i5> to <i6>
    %72 = source {handshake.bb = 3 : ui32, handshake.name = "source3"} : <>
    %73 = constant %72 {handshake.bb = 3 : ui32, handshake.name = "constant45", value = 1 : i2} : <>, <i2>
    %74:2 = fork [2] %73 {handshake.bb = 3 : ui32, handshake.name = "fork13"} : <i2>
    %75 = extsi %74#0 {handshake.bb = 3 : ui32, handshake.name = "extsi58"} : <i2> to <i6>
    %76 = extsi %74#1 {handshake.bb = 3 : ui32, handshake.name = "extsi7"} : <i2> to <i32>
    %77:4 = fork [4] %76 {handshake.bb = 3 : ui32, handshake.name = "fork14"} : <i32>
    %78 = source {handshake.bb = 3 : ui32, handshake.name = "source4"} : <>
    %79 = constant %78 {handshake.bb = 3 : ui32, handshake.name = "constant46", value = 3 : i3} : <>, <i3>
    %80 = extsi %79 {handshake.bb = 3 : ui32, handshake.name = "extsi8"} : <i3> to <i32>
    %81:4 = fork [4] %80 {handshake.bb = 3 : ui32, handshake.name = "fork15"} : <i32>
    %82 = shli %59#0, %77#0 {handshake.bb = 3 : ui32, handshake.name = "shli2"} : <i32>
    %83 = buffer %82, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer18"} : <i32>
    %84 = trunci %83 {handshake.bb = 3 : ui32, handshake.name = "trunci2"} : <i32> to <i7>
    %85 = shli %59#1, %81#0 {handshake.bb = 3 : ui32, handshake.name = "shli3"} : <i32>
    %86 = buffer %85, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer19"} : <i32>
    %87 = trunci %86 {handshake.bb = 3 : ui32, handshake.name = "trunci3"} : <i32> to <i7>
    %88 = addi %84, %87 {handshake.bb = 3 : ui32, handshake.name = "addi28"} : <i7>
    %89 = buffer %88, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer20"} : <i7>
    %90 = addi %50, %89 {handshake.bb = 3 : ui32, handshake.name = "addi4"} : <i7>
    %addressResult_10, %dataResult_11 = load[%90] %outputs_4 {handshake.bb = 3 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i7>, <i32>, <i7>, <i32>
    %91 = shli %53#0, %77#1 {handshake.bb = 3 : ui32, handshake.name = "shli4"} : <i32>
    %92 = buffer %91, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer21"} : <i32>
    %93 = trunci %92 {handshake.bb = 3 : ui32, handshake.name = "trunci4"} : <i32> to <i7>
    %94 = shli %53#1, %81#1 {handshake.bb = 3 : ui32, handshake.name = "shli5"} : <i32>
    %95 = buffer %94, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer22"} : <i32>
    %96 = trunci %95 {handshake.bb = 3 : ui32, handshake.name = "trunci5"} : <i32> to <i7>
    %97 = addi %93, %96 {handshake.bb = 3 : ui32, handshake.name = "addi29"} : <i7>
    %98 = buffer %97, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer23"} : <i7>
    %99 = addi %64, %98 {handshake.bb = 3 : ui32, handshake.name = "addi5"} : <i7>
    %addressResult_12, %dataResult_13 = load[%99] %outputs_2 {handshake.bb = 3 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i7>, <i32>, <i7>, <i32>
    %100 = muli %dataResult_11, %dataResult_13 {handshake.bb = 3 : ui32, handshake.name = "muli0"} : <i32>
    %101 = shli %59#2, %77#2 {handshake.bb = 3 : ui32, handshake.name = "shli6"} : <i32>
    %102 = buffer %101, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer24"} : <i32>
    %103 = trunci %102 {handshake.bb = 3 : ui32, handshake.name = "trunci6"} : <i32> to <i7>
    %104 = shli %59#3, %81#2 {handshake.bb = 3 : ui32, handshake.name = "shli7"} : <i32>
    %105 = buffer %104, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer25"} : <i32>
    %106 = trunci %105 {handshake.bb = 3 : ui32, handshake.name = "trunci7"} : <i32> to <i7>
    %107 = addi %103, %106 {handshake.bb = 3 : ui32, handshake.name = "addi30"} : <i7>
    %108 = buffer %107, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer26"} : <i7>
    %109 = addi %65, %108 {handshake.bb = 3 : ui32, handshake.name = "addi6"} : <i7>
    %110 = buffer %109, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer27"} : <i7>
    %addressResult_14, %dataResult_15 = load[%110] %3#0 {handshake.bb = 3 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["store1", 3], ["store1", 4]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 1>, handshake.name = "load2"} : <i7>, <i32>, <i7>, <i32>
    %111 = addi %dataResult_15, %100 {handshake.bb = 3 : ui32, handshake.name = "addi0"} : <i32>
    %112 = shli %59#4, %77#3 {handshake.bb = 3 : ui32, handshake.name = "shli8"} : <i32>
    %113 = buffer %112, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer29"} : <i32>
    %114 = trunci %113 {handshake.bb = 3 : ui32, handshake.name = "trunci8"} : <i32> to <i7>
    %115 = shli %59#5, %81#3 {handshake.bb = 3 : ui32, handshake.name = "shli9"} : <i32>
    %116 = buffer %115, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer30"} : <i32>
    %117 = trunci %116 {handshake.bb = 3 : ui32, handshake.name = "trunci9"} : <i32> to <i7>
    %118 = addi %114, %117 {handshake.bb = 3 : ui32, handshake.name = "addi31"} : <i7>
    %119 = buffer %118, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer31"} : <i7>
    %120 = addi %66, %119 {handshake.bb = 3 : ui32, handshake.name = "addi7"} : <i7>
    %121 = buffer %111, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer28"} : <i32>
    %122 = buffer %120, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer32"} : <i7>
    %addressResult_16, %dataResult_17 = store[%122] %121 {handshake.bb = 3 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["load2", 3], ["store1", 3], ["load6", 1]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 1>, handshake.name = "store1"} : <i7>, <i32>, <i7>, <i32>
    %123 = addi %51, %75 {handshake.bb = 3 : ui32, handshake.name = "addi18"} : <i6>
    %124 = buffer %123, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer33"} : <i6>
    %125:2 = fork [2] %124 {handshake.bb = 3 : ui32, handshake.name = "fork16"} : <i6>
    %126 = trunci %125#0 {handshake.bb = 3 : ui32, handshake.name = "trunci10"} : <i6> to <i5>
    %127 = cmpi ult, %125#1, %71 {handshake.bb = 3 : ui32, handshake.name = "cmpi0"} : <i6>
    %128 = buffer %127, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer34"} : <i1>
    %129:4 = fork [4] %128 {handshake.bb = 3 : ui32, handshake.name = "fork17"} : <i1>
    %trueResult, %falseResult = cond_br %129#0, %126 {handshake.bb = 3 : ui32, handshake.name = "cond_br9"} : <i1>, <i5>
    sink %falseResult {handshake.name = "sink0"} : <i5>
    %trueResult_18, %falseResult_19 = cond_br %129#1, %57#0 {handshake.bb = 3 : ui32, handshake.name = "cond_br10"} : <i1>, <i5>
    %trueResult_20, %falseResult_21 = cond_br %129#2, %63#3 {handshake.bb = 3 : ui32, handshake.name = "cond_br11"} : <i1>, <i5>
    %130 = buffer %68#1, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer17"} : <>
    %trueResult_22, %falseResult_23 = cond_br %129#3, %130 {handshake.bb = 3 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,inf], [1,inf], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "cond_br12"} : <i1>, <>
    %131 = extsi %falseResult_21 {handshake.bb = 4 : ui32, handshake.name = "extsi59"} : <i5> to <i6>
    %132 = source {handshake.bb = 4 : ui32, handshake.name = "source5"} : <>
    %133 = constant %132 {handshake.bb = 4 : ui32, handshake.name = "constant47", value = 10 : i5} : <>, <i5>
    %134 = extsi %133 {handshake.bb = 4 : ui32, handshake.name = "extsi60"} : <i5> to <i6>
    %135 = source {handshake.bb = 4 : ui32, handshake.name = "source6"} : <>
    %136 = constant %135 {handshake.bb = 4 : ui32, handshake.name = "constant48", value = 1 : i2} : <>, <i2>
    %137 = extsi %136 {handshake.bb = 4 : ui32, handshake.name = "extsi61"} : <i2> to <i6>
    %138 = addi %131, %137 {handshake.bb = 4 : ui32, handshake.name = "addi19"} : <i6>
    %139 = buffer %138, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer35"} : <i6>
    %140:2 = fork [2] %139 {handshake.bb = 4 : ui32, handshake.name = "fork18"} : <i6>
    %141 = trunci %140#0 {handshake.bb = 4 : ui32, handshake.name = "trunci11"} : <i6> to <i5>
    %142 = cmpi ult, %140#1, %134 {handshake.bb = 4 : ui32, handshake.name = "cmpi1"} : <i6>
    %143 = buffer %142, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer36"} : <i1>
    %144:3 = fork [3] %143 {handshake.bb = 4 : ui32, handshake.name = "fork19"} : <i1>
    %trueResult_24, %falseResult_25 = cond_br %144#0, %141 {handshake.bb = 4 : ui32, handshake.name = "cond_br13"} : <i1>, <i5>
    sink %falseResult_25 {handshake.name = "sink2"} : <i5>
    %trueResult_26, %falseResult_27 = cond_br %144#1, %falseResult_19 {handshake.bb = 4 : ui32, handshake.name = "cond_br14"} : <i1>, <i5>
    %trueResult_28, %falseResult_29 = cond_br %144#2, %falseResult_23 {handshake.bb = 4 : ui32, handshake.name = "cond_br15"} : <i1>, <>
    %145 = extsi %falseResult_27 {handshake.bb = 5 : ui32, handshake.name = "extsi62"} : <i5> to <i6>
    %146:2 = fork [2] %falseResult_29 {handshake.bb = 5 : ui32, handshake.name = "fork20"} : <>
    %147 = constant %146#0 {handshake.bb = 5 : ui32, handshake.name = "constant49", value = false} : <>, <i1>
    %148 = source {handshake.bb = 5 : ui32, handshake.name = "source7"} : <>
    %149 = constant %148 {handshake.bb = 5 : ui32, handshake.name = "constant50", value = 10 : i5} : <>, <i5>
    %150 = extsi %149 {handshake.bb = 5 : ui32, handshake.name = "extsi63"} : <i5> to <i6>
    %151 = source {handshake.bb = 5 : ui32, handshake.name = "source8"} : <>
    %152 = constant %151 {handshake.bb = 5 : ui32, handshake.name = "constant51", value = 1 : i2} : <>, <i2>
    %153 = extsi %152 {handshake.bb = 5 : ui32, handshake.name = "extsi64"} : <i2> to <i6>
    %154 = buffer %145, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 5 : ui32, handshake.name = "buffer37"} : <i6>
    %155 = addi %154, %153 {handshake.bb = 5 : ui32, handshake.name = "addi20"} : <i6>
    %156 = buffer %155, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 5 : ui32, handshake.name = "buffer38"} : <i6>
    %157:2 = fork [2] %156 {handshake.bb = 5 : ui32, handshake.name = "fork21"} : <i6>
    %158 = trunci %157#0 {handshake.bb = 5 : ui32, handshake.name = "trunci12"} : <i6> to <i5>
    %159 = cmpi ult, %157#1, %150 {handshake.bb = 5 : ui32, handshake.name = "cmpi2"} : <i6>
    %160 = buffer %159, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 5 : ui32, handshake.name = "buffer39"} : <i1>
    %161:3 = fork [3] %160 {handshake.bb = 5 : ui32, handshake.name = "fork22"} : <i1>
    %trueResult_30, %falseResult_31 = cond_br %161#0, %158 {handshake.bb = 5 : ui32, handshake.name = "cond_br16"} : <i1>, <i5>
    sink %falseResult_31 {handshake.name = "sink4"} : <i5>
    %trueResult_32, %falseResult_33 = cond_br %161#1, %146#1 {handshake.bb = 5 : ui32, handshake.name = "cond_br17"} : <i1>, <>
    %trueResult_34, %falseResult_35 = cond_br %161#2, %147 {handshake.bb = 5 : ui32, handshake.name = "cond_br18"} : <i1>, <i1>
    sink %trueResult_34 {handshake.name = "sink5"} : <i1>
    %162 = extsi %falseResult_35 {handshake.bb = 5 : ui32, handshake.name = "extsi44"} : <i1> to <i5>
    %163 = mux %index_37 [%162, %trueResult_66] {handshake.bb = 6 : ui32, handshake.name = "mux6"} : <i1>, [<i5>, <i5>] to <i5>
    %result_36, %index_37 = control_merge [%falseResult_33, %trueResult_68]  {handshake.bb = 6 : ui32, handshake.name = "control_merge5"} : [<>, <>] to <>, <i1>
    %164:2 = fork [2] %result_36 {handshake.bb = 6 : ui32, handshake.name = "fork23"} : <>
    %165 = constant %164#0 {handshake.bb = 6 : ui32, handshake.name = "constant52", value = false} : <>, <i1>
    %166 = extsi %165 {handshake.bb = 6 : ui32, handshake.name = "extsi43"} : <i1> to <i5>
    %167 = buffer %163, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 6 : ui32, handshake.name = "buffer40"} : <i5>
    %168 = mux %178#1 [%166, %trueResult_60] {handshake.bb = 7 : ui32, handshake.name = "mux7"} : <i1>, [<i5>, <i5>] to <i5>
    %169 = buffer %168, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 7 : ui32, handshake.name = "buffer41"} : <i5>
    %170:2 = fork [2] %169 {handshake.bb = 7 : ui32, handshake.name = "fork24"} : <i5>
    %171 = extsi %170#0 {handshake.bb = 7 : ui32, handshake.name = "extsi65"} : <i5> to <i7>
    %172 = mux %178#0 [%167, %trueResult_62] {handshake.bb = 7 : ui32, handshake.name = "mux8"} : <i1>, [<i5>, <i5>] to <i5>
    %173 = buffer %172, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 7 : ui32, handshake.name = "buffer42"} : <i5>
    %174 = buffer %173, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 7 : ui32, handshake.name = "buffer43"} : <i5>
    %175:2 = fork [2] %174 {handshake.bb = 7 : ui32, handshake.name = "fork25"} : <i5>
    %176 = extsi %175#1 {handshake.bb = 7 : ui32, handshake.name = "extsi66"} : <i5> to <i32>
    %177:2 = fork [2] %176 {handshake.bb = 7 : ui32, handshake.name = "fork26"} : <i32>
    %result_38, %index_39 = control_merge [%164#1, %trueResult_64]  {handshake.bb = 7 : ui32, handshake.name = "control_merge6"} : [<>, <>] to <>, <i1>
    %178:2 = fork [2] %index_39 {handshake.bb = 7 : ui32, handshake.name = "fork27"} : <i1>
    %179:3 = lazy_fork [3] %result_38 {handshake.bb = 7 : ui32, handshake.name = "lazy_fork2"} : <>
    %180 = buffer %179#2, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 7 : ui32, handshake.name = "buffer45"} : <>
    %181 = constant %180 {handshake.bb = 7 : ui32, handshake.bufProps = #handshake<bufProps{"0": [1,inf], [0,inf], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "constant53", value = false} : <>, <i1>
    %182:2 = fork [2] %181 {handshake.bb = 7 : ui32, handshake.name = "fork28"} : <i1>
    %183 = extsi %182#1 {handshake.bb = 7 : ui32, handshake.name = "extsi16"} : <i1> to <i32>
    %184 = source {handshake.bb = 7 : ui32, handshake.name = "source9"} : <>
    %185 = constant %184 {handshake.bb = 7 : ui32, handshake.name = "constant54", value = 1 : i2} : <>, <i2>
    %186 = extsi %185 {handshake.bb = 7 : ui32, handshake.name = "extsi17"} : <i2> to <i32>
    %187 = source {handshake.bb = 7 : ui32, handshake.name = "source10"} : <>
    %188 = constant %187 {handshake.bb = 7 : ui32, handshake.name = "constant55", value = 3 : i3} : <>, <i3>
    %189 = extsi %188 {handshake.bb = 7 : ui32, handshake.name = "extsi18"} : <i3> to <i32>
    %190 = shli %177#0, %186 {handshake.bb = 7 : ui32, handshake.name = "shli10"} : <i32>
    %191 = buffer %190, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 7 : ui32, handshake.name = "buffer47"} : <i32>
    %192 = trunci %191 {handshake.bb = 7 : ui32, handshake.name = "trunci13"} : <i32> to <i7>
    %193 = shli %177#1, %189 {handshake.bb = 7 : ui32, handshake.name = "shli11"} : <i32>
    %194 = buffer %193, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 7 : ui32, handshake.name = "buffer48"} : <i32>
    %195 = trunci %194 {handshake.bb = 7 : ui32, handshake.name = "trunci14"} : <i32> to <i7>
    %196 = addi %192, %195 {handshake.bb = 7 : ui32, handshake.name = "addi32"} : <i7>
    %197 = buffer %196, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 7 : ui32, handshake.name = "buffer49"} : <i7>
    %198 = addi %171, %197 {handshake.bb = 7 : ui32, handshake.name = "addi8"} : <i7>
    %199 = buffer %183, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 7 : ui32, handshake.name = "buffer46"} : <i32>
    %200 = buffer %198, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 7 : ui32, handshake.name = "buffer50"} : <i7>
    %addressResult_40, %dataResult_41 = store[%200] %199 {handshake.bb = 7 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["load5", 3], ["store3", 3], ["load7", 1]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 0>, handshake.name = "store2"} : <i7>, <i32>, <i7>, <i32>
    %201 = extsi %182#0 {handshake.bb = 7 : ui32, handshake.name = "extsi42"} : <i1> to <i5>
    %202 = buffer %179#1, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 7 : ui32, handshake.name = "buffer44"} : <>
    %203 = mux %224#2 [%201, %trueResult_52] {handshake.bb = 8 : ui32, handshake.name = "mux9"} : <i1>, [<i5>, <i5>] to <i5>
    %204 = buffer %203, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 8 : ui32, handshake.name = "buffer51"} : <i5>
    %205 = buffer %204, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 8 : ui32, handshake.name = "buffer52"} : <i5>
    %206:3 = fork [3] %205 {handshake.bb = 8 : ui32, handshake.name = "fork29"} : <i5>
    %207 = extsi %206#0 {handshake.bb = 8 : ui32, handshake.name = "extsi67"} : <i5> to <i7>
    %208 = extsi %206#1 {handshake.bb = 8 : ui32, handshake.name = "extsi68"} : <i5> to <i6>
    %209 = extsi %206#2 {handshake.bb = 8 : ui32, handshake.name = "extsi69"} : <i5> to <i32>
    %210:2 = fork [2] %209 {handshake.bb = 8 : ui32, handshake.name = "fork30"} : <i32>
    %211 = mux %224#0 [%175#0, %trueResult_54] {handshake.bb = 8 : ui32, handshake.name = "mux10"} : <i1>, [<i5>, <i5>] to <i5>
    %212 = buffer %211, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 8 : ui32, handshake.name = "buffer53"} : <i5>
    %213 = buffer %212, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 8 : ui32, handshake.name = "buffer54"} : <i5>
    %214:2 = fork [2] %213 {handshake.bb = 8 : ui32, handshake.name = "fork31"} : <i5>
    %215 = extsi %214#1 {handshake.bb = 8 : ui32, handshake.name = "extsi70"} : <i5> to <i32>
    %216:6 = fork [6] %215 {handshake.bb = 8 : ui32, handshake.name = "fork32"} : <i32>
    %217 = mux %224#1 [%170#1, %trueResult_56] {handshake.bb = 8 : ui32, handshake.name = "mux11"} : <i1>, [<i5>, <i5>] to <i5>
    %218 = buffer %217, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 8 : ui32, handshake.name = "buffer55"} : <i5>
    %219 = buffer %218, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 8 : ui32, handshake.name = "buffer56"} : <i5>
    %220:4 = fork [4] %219 {handshake.bb = 8 : ui32, handshake.name = "fork33"} : <i5>
    %221 = extsi %220#0 {handshake.bb = 8 : ui32, handshake.name = "extsi71"} : <i5> to <i7>
    %222 = extsi %220#1 {handshake.bb = 8 : ui32, handshake.name = "extsi72"} : <i5> to <i7>
    %223 = extsi %220#2 {handshake.bb = 8 : ui32, handshake.name = "extsi73"} : <i5> to <i7>
    %result_42, %index_43 = control_merge [%202, %trueResult_58]  {handshake.bb = 8 : ui32, handshake.name = "control_merge7"} : [<>, <>] to <>, <i1>
    %224:3 = fork [3] %index_43 {handshake.bb = 8 : ui32, handshake.name = "fork34"} : <i1>
    %225:2 = lazy_fork [2] %result_42 {handshake.bb = 8 : ui32, handshake.name = "lazy_fork3"} : <>
    %226 = source {handshake.bb = 8 : ui32, handshake.name = "source11"} : <>
    %227 = constant %226 {handshake.bb = 8 : ui32, handshake.name = "constant56", value = 10 : i5} : <>, <i5>
    %228 = extsi %227 {handshake.bb = 8 : ui32, handshake.name = "extsi74"} : <i5> to <i6>
    %229 = source {handshake.bb = 8 : ui32, handshake.name = "source12"} : <>
    %230 = constant %229 {handshake.bb = 8 : ui32, handshake.name = "constant57", value = 1 : i2} : <>, <i2>
    %231:2 = fork [2] %230 {handshake.bb = 8 : ui32, handshake.name = "fork35"} : <i2>
    %232 = extsi %231#0 {handshake.bb = 8 : ui32, handshake.name = "extsi75"} : <i2> to <i6>
    %233 = extsi %231#1 {handshake.bb = 8 : ui32, handshake.name = "extsi20"} : <i2> to <i32>
    %234:4 = fork [4] %233 {handshake.bb = 8 : ui32, handshake.name = "fork36"} : <i32>
    %235 = source {handshake.bb = 8 : ui32, handshake.name = "source13"} : <>
    %236 = constant %235 {handshake.bb = 8 : ui32, handshake.name = "constant58", value = 3 : i3} : <>, <i3>
    %237 = extsi %236 {handshake.bb = 8 : ui32, handshake.name = "extsi21"} : <i3> to <i32>
    %238:4 = fork [4] %237 {handshake.bb = 8 : ui32, handshake.name = "fork37"} : <i32>
    %239 = shli %216#0, %234#0 {handshake.bb = 8 : ui32, handshake.name = "shli12"} : <i32>
    %240 = buffer %239, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 8 : ui32, handshake.name = "buffer58"} : <i32>
    %241 = trunci %240 {handshake.bb = 8 : ui32, handshake.name = "trunci15"} : <i32> to <i7>
    %242 = shli %216#1, %238#0 {handshake.bb = 8 : ui32, handshake.name = "shli13"} : <i32>
    %243 = buffer %242, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 8 : ui32, handshake.name = "buffer59"} : <i32>
    %244 = trunci %243 {handshake.bb = 8 : ui32, handshake.name = "trunci16"} : <i32> to <i7>
    %245 = addi %241, %244 {handshake.bb = 8 : ui32, handshake.name = "addi33"} : <i7>
    %246 = buffer %245, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 8 : ui32, handshake.name = "buffer60"} : <i7>
    %247 = addi %207, %246 {handshake.bb = 8 : ui32, handshake.name = "addi9"} : <i7>
    %addressResult_44, %dataResult_45 = load[%247] %outputs_0 {handshake.bb = 8 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load3"} : <i7>, <i32>, <i7>, <i32>
    %248 = shli %210#0, %234#1 {handshake.bb = 8 : ui32, handshake.name = "shli14"} : <i32>
    %249 = buffer %248, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 8 : ui32, handshake.name = "buffer61"} : <i32>
    %250 = trunci %249 {handshake.bb = 8 : ui32, handshake.name = "trunci17"} : <i32> to <i7>
    %251 = shli %210#1, %238#1 {handshake.bb = 8 : ui32, handshake.name = "shli15"} : <i32>
    %252 = buffer %251, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 8 : ui32, handshake.name = "buffer62"} : <i32>
    %253 = trunci %252 {handshake.bb = 8 : ui32, handshake.name = "trunci18"} : <i32> to <i7>
    %254 = addi %250, %253 {handshake.bb = 8 : ui32, handshake.name = "addi34"} : <i7>
    %255 = buffer %254, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 8 : ui32, handshake.name = "buffer63"} : <i7>
    %256 = addi %221, %255 {handshake.bb = 8 : ui32, handshake.name = "addi10"} : <i7>
    %addressResult_46, %dataResult_47 = load[%256] %outputs {handshake.bb = 8 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load4"} : <i7>, <i32>, <i7>, <i32>
    %257 = muli %dataResult_45, %dataResult_47 {handshake.bb = 8 : ui32, handshake.name = "muli1"} : <i32>
    %258 = shli %216#2, %234#2 {handshake.bb = 8 : ui32, handshake.name = "shli16"} : <i32>
    %259 = buffer %258, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 8 : ui32, handshake.name = "buffer64"} : <i32>
    %260 = trunci %259 {handshake.bb = 8 : ui32, handshake.name = "trunci19"} : <i32> to <i7>
    %261 = shli %216#3, %238#2 {handshake.bb = 8 : ui32, handshake.name = "shli17"} : <i32>
    %262 = buffer %261, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 8 : ui32, handshake.name = "buffer65"} : <i32>
    %263 = trunci %262 {handshake.bb = 8 : ui32, handshake.name = "trunci20"} : <i32> to <i7>
    %264 = addi %260, %263 {handshake.bb = 8 : ui32, handshake.name = "addi35"} : <i7>
    %265 = buffer %264, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 8 : ui32, handshake.name = "buffer66"} : <i7>
    %266 = addi %222, %265 {handshake.bb = 8 : ui32, handshake.name = "addi11"} : <i7>
    %267 = buffer %266, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 8 : ui32, handshake.name = "buffer67"} : <i7>
    %addressResult_48, %dataResult_49 = load[%267] %2#0 {handshake.bb = 8 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["store3", 3], ["store3", 4]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 1>, handshake.name = "load5"} : <i7>, <i32>, <i7>, <i32>
    %268 = addi %dataResult_49, %257 {handshake.bb = 8 : ui32, handshake.name = "addi1"} : <i32>
    %269 = shli %216#4, %234#3 {handshake.bb = 8 : ui32, handshake.name = "shli18"} : <i32>
    %270 = buffer %269, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 8 : ui32, handshake.name = "buffer69"} : <i32>
    %271 = trunci %270 {handshake.bb = 8 : ui32, handshake.name = "trunci21"} : <i32> to <i7>
    %272 = shli %216#5, %238#3 {handshake.bb = 8 : ui32, handshake.name = "shli19"} : <i32>
    %273 = buffer %272, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 8 : ui32, handshake.name = "buffer70"} : <i32>
    %274 = trunci %273 {handshake.bb = 8 : ui32, handshake.name = "trunci22"} : <i32> to <i7>
    %275 = addi %271, %274 {handshake.bb = 8 : ui32, handshake.name = "addi36"} : <i7>
    %276 = buffer %275, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 8 : ui32, handshake.name = "buffer71"} : <i7>
    %277 = addi %223, %276 {handshake.bb = 8 : ui32, handshake.name = "addi12"} : <i7>
    %278 = buffer %268, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 8 : ui32, handshake.name = "buffer68"} : <i32>
    %279 = buffer %277, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 8 : ui32, handshake.name = "buffer72"} : <i7>
    %addressResult_50, %dataResult_51 = store[%279] %278 {handshake.bb = 8 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["load5", 3], ["store3", 3], ["load7", 1]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 1>, handshake.name = "store3"} : <i7>, <i32>, <i7>, <i32>
    %280 = addi %208, %232 {handshake.bb = 8 : ui32, handshake.name = "addi21"} : <i6>
    %281 = buffer %280, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 8 : ui32, handshake.name = "buffer73"} : <i6>
    %282:2 = fork [2] %281 {handshake.bb = 8 : ui32, handshake.name = "fork38"} : <i6>
    %283 = trunci %282#0 {handshake.bb = 8 : ui32, handshake.name = "trunci23"} : <i6> to <i5>
    %284 = cmpi ult, %282#1, %228 {handshake.bb = 8 : ui32, handshake.name = "cmpi3"} : <i6>
    %285 = buffer %284, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 8 : ui32, handshake.name = "buffer74"} : <i1>
    %286:4 = fork [4] %285 {handshake.bb = 8 : ui32, handshake.name = "fork39"} : <i1>
    %trueResult_52, %falseResult_53 = cond_br %286#0, %283 {handshake.bb = 8 : ui32, handshake.name = "cond_br19"} : <i1>, <i5>
    sink %falseResult_53 {handshake.name = "sink6"} : <i5>
    %trueResult_54, %falseResult_55 = cond_br %286#1, %214#0 {handshake.bb = 8 : ui32, handshake.name = "cond_br20"} : <i1>, <i5>
    %trueResult_56, %falseResult_57 = cond_br %286#2, %220#3 {handshake.bb = 8 : ui32, handshake.name = "cond_br21"} : <i1>, <i5>
    %287 = buffer %225#1, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 8 : ui32, handshake.name = "buffer57"} : <>
    %trueResult_58, %falseResult_59 = cond_br %286#3, %287 {handshake.bb = 8 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,inf], [1,inf], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "cond_br22"} : <i1>, <>
    %288 = extsi %falseResult_57 {handshake.bb = 9 : ui32, handshake.name = "extsi76"} : <i5> to <i6>
    %289 = source {handshake.bb = 9 : ui32, handshake.name = "source14"} : <>
    %290 = constant %289 {handshake.bb = 9 : ui32, handshake.name = "constant59", value = 10 : i5} : <>, <i5>
    %291 = extsi %290 {handshake.bb = 9 : ui32, handshake.name = "extsi77"} : <i5> to <i6>
    %292 = source {handshake.bb = 9 : ui32, handshake.name = "source15"} : <>
    %293 = constant %292 {handshake.bb = 9 : ui32, handshake.name = "constant60", value = 1 : i2} : <>, <i2>
    %294 = extsi %293 {handshake.bb = 9 : ui32, handshake.name = "extsi78"} : <i2> to <i6>
    %295 = addi %288, %294 {handshake.bb = 9 : ui32, handshake.name = "addi22"} : <i6>
    %296 = buffer %295, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 9 : ui32, handshake.name = "buffer75"} : <i6>
    %297:2 = fork [2] %296 {handshake.bb = 9 : ui32, handshake.name = "fork40"} : <i6>
    %298 = trunci %297#0 {handshake.bb = 9 : ui32, handshake.name = "trunci24"} : <i6> to <i5>
    %299 = cmpi ult, %297#1, %291 {handshake.bb = 9 : ui32, handshake.name = "cmpi4"} : <i6>
    %300 = buffer %299, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 9 : ui32, handshake.name = "buffer76"} : <i1>
    %301:3 = fork [3] %300 {handshake.bb = 9 : ui32, handshake.name = "fork41"} : <i1>
    %trueResult_60, %falseResult_61 = cond_br %301#0, %298 {handshake.bb = 9 : ui32, handshake.name = "cond_br23"} : <i1>, <i5>
    sink %falseResult_61 {handshake.name = "sink8"} : <i5>
    %trueResult_62, %falseResult_63 = cond_br %301#1, %falseResult_55 {handshake.bb = 9 : ui32, handshake.name = "cond_br24"} : <i1>, <i5>
    %trueResult_64, %falseResult_65 = cond_br %301#2, %falseResult_59 {handshake.bb = 9 : ui32, handshake.name = "cond_br25"} : <i1>, <>
    %302 = buffer %falseResult_63, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 10 : ui32, handshake.name = "buffer77"} : <i5>
    %303 = extsi %302 {handshake.bb = 10 : ui32, handshake.name = "extsi79"} : <i5> to <i6>
    %304:2 = fork [2] %falseResult_65 {handshake.bb = 10 : ui32, handshake.name = "fork42"} : <>
    %305 = constant %304#0 {handshake.bb = 10 : ui32, handshake.name = "constant61", value = false} : <>, <i1>
    %306 = source {handshake.bb = 10 : ui32, handshake.name = "source16"} : <>
    %307 = constant %306 {handshake.bb = 10 : ui32, handshake.name = "constant62", value = 10 : i5} : <>, <i5>
    %308 = extsi %307 {handshake.bb = 10 : ui32, handshake.name = "extsi80"} : <i5> to <i6>
    %309 = source {handshake.bb = 10 : ui32, handshake.name = "source17"} : <>
    %310 = constant %309 {handshake.bb = 10 : ui32, handshake.name = "constant63", value = 1 : i2} : <>, <i2>
    %311 = extsi %310 {handshake.bb = 10 : ui32, handshake.name = "extsi81"} : <i2> to <i6>
    %312 = addi %303, %311 {handshake.bb = 10 : ui32, handshake.name = "addi23"} : <i6>
    %313 = buffer %312, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 10 : ui32, handshake.name = "buffer78"} : <i6>
    %314:2 = fork [2] %313 {handshake.bb = 10 : ui32, handshake.name = "fork43"} : <i6>
    %315 = trunci %314#0 {handshake.bb = 10 : ui32, handshake.name = "trunci25"} : <i6> to <i5>
    %316 = cmpi ult, %314#1, %308 {handshake.bb = 10 : ui32, handshake.name = "cmpi5"} : <i6>
    %317 = buffer %316, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 10 : ui32, handshake.name = "buffer79"} : <i1>
    %318:3 = fork [3] %317 {handshake.bb = 10 : ui32, handshake.name = "fork44"} : <i1>
    %trueResult_66, %falseResult_67 = cond_br %318#0, %315 {handshake.bb = 10 : ui32, handshake.name = "cond_br26"} : <i1>, <i5>
    sink %falseResult_67 {handshake.name = "sink10"} : <i5>
    %trueResult_68, %falseResult_69 = cond_br %318#1, %304#1 {handshake.bb = 10 : ui32, handshake.name = "cond_br27"} : <i1>, <>
    %trueResult_70, %falseResult_71 = cond_br %318#2, %305 {handshake.bb = 10 : ui32, handshake.name = "cond_br28"} : <i1>, <i1>
    sink %trueResult_70 {handshake.name = "sink11"} : <i1>
    %319 = extsi %falseResult_71 {handshake.bb = 10 : ui32, handshake.name = "extsi41"} : <i1> to <i5>
    %320 = mux %index_73 [%319, %trueResult_102] {handshake.bb = 11 : ui32, handshake.name = "mux12"} : <i1>, [<i5>, <i5>] to <i5>
    %result_72, %index_73 = control_merge [%falseResult_69, %trueResult_104]  {handshake.bb = 11 : ui32, handshake.name = "control_merge10"} : [<>, <>] to <>, <i1>
    %321:2 = fork [2] %result_72 {handshake.bb = 11 : ui32, handshake.name = "fork45"} : <>
    %322 = constant %321#0 {handshake.bb = 11 : ui32, handshake.name = "constant64", value = false} : <>, <i1>
    %323 = extsi %322 {handshake.bb = 11 : ui32, handshake.name = "extsi40"} : <i1> to <i5>
    %324 = buffer %320, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 11 : ui32, handshake.name = "buffer80"} : <i5>
    %325 = mux %335#1 [%323, %trueResult_96] {handshake.bb = 12 : ui32, handshake.name = "mux13"} : <i1>, [<i5>, <i5>] to <i5>
    %326 = buffer %325, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 12 : ui32, handshake.name = "buffer81"} : <i5>
    %327:2 = fork [2] %326 {handshake.bb = 12 : ui32, handshake.name = "fork46"} : <i5>
    %328 = extsi %327#0 {handshake.bb = 12 : ui32, handshake.name = "extsi82"} : <i5> to <i7>
    %329 = mux %335#0 [%324, %trueResult_98] {handshake.bb = 12 : ui32, handshake.name = "mux14"} : <i1>, [<i5>, <i5>] to <i5>
    %330 = buffer %329, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 12 : ui32, handshake.name = "buffer82"} : <i5>
    %331 = buffer %330, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 12 : ui32, handshake.name = "buffer83"} : <i5>
    %332:2 = fork [2] %331 {handshake.bb = 12 : ui32, handshake.name = "fork47"} : <i5>
    %333 = extsi %332#1 {handshake.bb = 12 : ui32, handshake.name = "extsi83"} : <i5> to <i32>
    %334:2 = fork [2] %333 {handshake.bb = 12 : ui32, handshake.name = "fork48"} : <i32>
    %result_74, %index_75 = control_merge [%321#1, %trueResult_100]  {handshake.bb = 12 : ui32, handshake.name = "control_merge11"} : [<>, <>] to <>, <i1>
    %335:2 = fork [2] %index_75 {handshake.bb = 12 : ui32, handshake.name = "fork49"} : <i1>
    %336:3 = lazy_fork [3] %result_74 {handshake.bb = 12 : ui32, handshake.name = "lazy_fork4"} : <>
    %337 = buffer %336#2, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 12 : ui32, handshake.name = "buffer85"} : <>
    %338 = constant %337 {handshake.bb = 12 : ui32, handshake.bufProps = #handshake<bufProps{"0": [1,inf], [0,inf], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "constant65", value = false} : <>, <i1>
    %339:2 = fork [2] %338 {handshake.bb = 12 : ui32, handshake.name = "fork50"} : <i1>
    %340 = extsi %339#1 {handshake.bb = 12 : ui32, handshake.name = "extsi29"} : <i1> to <i32>
    %341 = source {handshake.bb = 12 : ui32, handshake.name = "source18"} : <>
    %342 = constant %341 {handshake.bb = 12 : ui32, handshake.name = "constant66", value = 1 : i2} : <>, <i2>
    %343 = extsi %342 {handshake.bb = 12 : ui32, handshake.name = "extsi30"} : <i2> to <i32>
    %344 = source {handshake.bb = 12 : ui32, handshake.name = "source19"} : <>
    %345 = constant %344 {handshake.bb = 12 : ui32, handshake.name = "constant67", value = 3 : i3} : <>, <i3>
    %346 = extsi %345 {handshake.bb = 12 : ui32, handshake.name = "extsi31"} : <i3> to <i32>
    %347 = shli %334#0, %343 {handshake.bb = 12 : ui32, handshake.name = "shli20"} : <i32>
    %348 = buffer %347, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 12 : ui32, handshake.name = "buffer87"} : <i32>
    %349 = trunci %348 {handshake.bb = 12 : ui32, handshake.name = "trunci26"} : <i32> to <i7>
    %350 = shli %334#1, %346 {handshake.bb = 12 : ui32, handshake.name = "shli21"} : <i32>
    %351 = buffer %350, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 12 : ui32, handshake.name = "buffer88"} : <i32>
    %352 = trunci %351 {handshake.bb = 12 : ui32, handshake.name = "trunci27"} : <i32> to <i7>
    %353 = addi %349, %352 {handshake.bb = 12 : ui32, handshake.name = "addi37"} : <i7>
    %354 = buffer %353, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 12 : ui32, handshake.name = "buffer89"} : <i7>
    %355 = addi %328, %354 {handshake.bb = 12 : ui32, handshake.name = "addi13"} : <i7>
    %356 = buffer %340, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 12 : ui32, handshake.name = "buffer86"} : <i32>
    %357 = buffer %355, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 12 : ui32, handshake.name = "buffer90"} : <i7>
    %addressResult_76, %dataResult_77 = store[%357] %356 {handshake.bb = 12 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["load8", 3], ["store5", 3]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 0>, handshake.name = "store4"} : <i7>, <i32>, <i7>, <i32>
    %358 = extsi %339#0 {handshake.bb = 12 : ui32, handshake.name = "extsi39"} : <i1> to <i5>
    %359 = buffer %336#1, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 12 : ui32, handshake.name = "buffer84"} : <>
    %360 = mux %381#2 [%358, %trueResult_88] {handshake.bb = 13 : ui32, handshake.name = "mux15"} : <i1>, [<i5>, <i5>] to <i5>
    %361 = buffer %360, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 13 : ui32, handshake.name = "buffer91"} : <i5>
    %362 = buffer %361, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 13 : ui32, handshake.name = "buffer92"} : <i5>
    %363:3 = fork [3] %362 {handshake.bb = 13 : ui32, handshake.name = "fork51"} : <i5>
    %364 = extsi %363#0 {handshake.bb = 13 : ui32, handshake.name = "extsi84"} : <i5> to <i7>
    %365 = extsi %363#1 {handshake.bb = 13 : ui32, handshake.name = "extsi85"} : <i5> to <i6>
    %366 = extsi %363#2 {handshake.bb = 13 : ui32, handshake.name = "extsi86"} : <i5> to <i32>
    %367:2 = fork [2] %366 {handshake.bb = 13 : ui32, handshake.name = "fork52"} : <i32>
    %368 = mux %381#0 [%332#0, %trueResult_90] {handshake.bb = 13 : ui32, handshake.name = "mux16"} : <i1>, [<i5>, <i5>] to <i5>
    %369 = buffer %368, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 13 : ui32, handshake.name = "buffer93"} : <i5>
    %370 = buffer %369, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 13 : ui32, handshake.name = "buffer94"} : <i5>
    %371:2 = fork [2] %370 {handshake.bb = 13 : ui32, handshake.name = "fork53"} : <i5>
    %372 = extsi %371#1 {handshake.bb = 13 : ui32, handshake.name = "extsi87"} : <i5> to <i32>
    %373:6 = fork [6] %372 {handshake.bb = 13 : ui32, handshake.name = "fork54"} : <i32>
    %374 = mux %381#1 [%327#1, %trueResult_92] {handshake.bb = 13 : ui32, handshake.name = "mux17"} : <i1>, [<i5>, <i5>] to <i5>
    %375 = buffer %374, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 13 : ui32, handshake.name = "buffer95"} : <i5>
    %376 = buffer %375, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 13 : ui32, handshake.name = "buffer96"} : <i5>
    %377:4 = fork [4] %376 {handshake.bb = 13 : ui32, handshake.name = "fork55"} : <i5>
    %378 = extsi %377#0 {handshake.bb = 13 : ui32, handshake.name = "extsi88"} : <i5> to <i7>
    %379 = extsi %377#1 {handshake.bb = 13 : ui32, handshake.name = "extsi89"} : <i5> to <i7>
    %380 = extsi %377#2 {handshake.bb = 13 : ui32, handshake.name = "extsi90"} : <i5> to <i7>
    %result_78, %index_79 = control_merge [%359, %trueResult_94]  {handshake.bb = 13 : ui32, handshake.name = "control_merge12"} : [<>, <>] to <>, <i1>
    %381:3 = fork [3] %index_79 {handshake.bb = 13 : ui32, handshake.name = "fork56"} : <i1>
    %382:3 = lazy_fork [3] %result_78 {handshake.bb = 13 : ui32, handshake.name = "lazy_fork5"} : <>
    %383 = buffer %382#2, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 13 : ui32, handshake.name = "buffer98"} : <>
    %384:2 = fork [2] %383 {handshake.bb = 13 : ui32, handshake.bufProps = #handshake<bufProps{"0": [1,inf], [0,inf], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork57"} : <>
    %385 = source {handshake.bb = 13 : ui32, handshake.name = "source20"} : <>
    %386 = constant %385 {handshake.bb = 13 : ui32, handshake.name = "constant68", value = 10 : i5} : <>, <i5>
    %387 = extsi %386 {handshake.bb = 13 : ui32, handshake.name = "extsi91"} : <i5> to <i6>
    %388 = source {handshake.bb = 13 : ui32, handshake.name = "source21"} : <>
    %389 = constant %388 {handshake.bb = 13 : ui32, handshake.name = "constant69", value = 1 : i2} : <>, <i2>
    %390:2 = fork [2] %389 {handshake.bb = 13 : ui32, handshake.name = "fork58"} : <i2>
    %391 = extsi %390#0 {handshake.bb = 13 : ui32, handshake.name = "extsi92"} : <i2> to <i6>
    %392 = extsi %390#1 {handshake.bb = 13 : ui32, handshake.name = "extsi33"} : <i2> to <i32>
    %393:4 = fork [4] %392 {handshake.bb = 13 : ui32, handshake.name = "fork59"} : <i32>
    %394 = source {handshake.bb = 13 : ui32, handshake.name = "source22"} : <>
    %395 = constant %394 {handshake.bb = 13 : ui32, handshake.name = "constant70", value = 3 : i3} : <>, <i3>
    %396 = extsi %395 {handshake.bb = 13 : ui32, handshake.name = "extsi34"} : <i3> to <i32>
    %397:4 = fork [4] %396 {handshake.bb = 13 : ui32, handshake.name = "fork60"} : <i32>
    %398 = shli %373#0, %393#0 {handshake.bb = 13 : ui32, handshake.name = "shli22"} : <i32>
    %399 = buffer %398, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 13 : ui32, handshake.name = "buffer99"} : <i32>
    %400 = trunci %399 {handshake.bb = 13 : ui32, handshake.name = "trunci28"} : <i32> to <i7>
    %401 = shli %373#1, %397#0 {handshake.bb = 13 : ui32, handshake.name = "shli23"} : <i32>
    %402 = buffer %401, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 13 : ui32, handshake.name = "buffer100"} : <i32>
    %403 = trunci %402 {handshake.bb = 13 : ui32, handshake.name = "trunci29"} : <i32> to <i7>
    %404 = addi %400, %403 {handshake.bb = 13 : ui32, handshake.name = "addi38"} : <i7>
    %405 = buffer %404, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 13 : ui32, handshake.name = "buffer101"} : <i7>
    %406 = addi %364, %405 {handshake.bb = 13 : ui32, handshake.name = "addi14"} : <i7>
    %407 = buffer %406, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 13 : ui32, handshake.name = "buffer102"} : <i7>
    %addressResult_80, %dataResult_81 = load[%407] %3#1 {handshake.bb = 13 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<LSQ: 2>, handshake.name = "load6"} : <i7>, <i32>, <i7>, <i32>
    %408 = shli %367#0, %393#1 {handshake.bb = 13 : ui32, handshake.name = "shli24"} : <i32>
    %409 = buffer %408, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 13 : ui32, handshake.name = "buffer103"} : <i32>
    %410 = trunci %409 {handshake.bb = 13 : ui32, handshake.name = "trunci30"} : <i32> to <i7>
    %411 = shli %367#1, %397#1 {handshake.bb = 13 : ui32, handshake.name = "shli25"} : <i32>
    %412 = buffer %411, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 13 : ui32, handshake.name = "buffer104"} : <i32>
    %413 = trunci %412 {handshake.bb = 13 : ui32, handshake.name = "trunci31"} : <i32> to <i7>
    %414 = addi %410, %413 {handshake.bb = 13 : ui32, handshake.name = "addi39"} : <i7>
    %415 = buffer %414, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 13 : ui32, handshake.name = "buffer105"} : <i7>
    %416 = addi %378, %415 {handshake.bb = 13 : ui32, handshake.name = "addi15"} : <i7>
    %417 = buffer %416, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 13 : ui32, handshake.name = "buffer106"} : <i7>
    %addressResult_82, %dataResult_83 = load[%417] %2#1 {handshake.bb = 13 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<LSQ: 2>, handshake.name = "load7"} : <i7>, <i32>, <i7>, <i32>
    %418 = muli %dataResult_81, %dataResult_83 {handshake.bb = 13 : ui32, handshake.name = "muli2"} : <i32>
    %419 = shli %373#2, %393#2 {handshake.bb = 13 : ui32, handshake.name = "shli26"} : <i32>
    %420 = buffer %419, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 13 : ui32, handshake.name = "buffer107"} : <i32>
    %421 = trunci %420 {handshake.bb = 13 : ui32, handshake.name = "trunci32"} : <i32> to <i7>
    %422 = shli %373#3, %397#2 {handshake.bb = 13 : ui32, handshake.name = "shli27"} : <i32>
    %423 = buffer %422, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 13 : ui32, handshake.name = "buffer108"} : <i32>
    %424 = trunci %423 {handshake.bb = 13 : ui32, handshake.name = "trunci33"} : <i32> to <i7>
    %425 = addi %421, %424 {handshake.bb = 13 : ui32, handshake.name = "addi40"} : <i7>
    %426 = buffer %425, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 13 : ui32, handshake.name = "buffer109"} : <i7>
    %427 = addi %379, %426 {handshake.bb = 13 : ui32, handshake.name = "addi16"} : <i7>
    %428 = buffer %427, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 13 : ui32, handshake.name = "buffer110"} : <i7>
    %addressResult_84, %dataResult_85 = load[%428] %1#0 {handshake.bb = 13 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["store5", 3], ["store5", 4]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 1>, handshake.name = "load8"} : <i7>, <i32>, <i7>, <i32>
    %429 = addi %dataResult_85, %418 {handshake.bb = 13 : ui32, handshake.name = "addi2"} : <i32>
    %430 = shli %432, %431 {handshake.bb = 13 : ui32, handshake.name = "shli28"} : <i32>
    %431 = buffer %393#3, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 13 : ui32, handshake.name = "buffer146"} : <i32>
    %432 = buffer %373#4, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 13 : ui32, handshake.name = "buffer147"} : <i32>
    %433 = buffer %430, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 13 : ui32, handshake.name = "buffer112"} : <i32>
    %434 = trunci %433 {handshake.bb = 13 : ui32, handshake.name = "trunci34"} : <i32> to <i7>
    %435 = shli %373#5, %397#3 {handshake.bb = 13 : ui32, handshake.name = "shli29"} : <i32>
    %436 = buffer %435, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 13 : ui32, handshake.name = "buffer113"} : <i32>
    %437 = trunci %436 {handshake.bb = 13 : ui32, handshake.name = "trunci35"} : <i32> to <i7>
    %438 = addi %434, %437 {handshake.bb = 13 : ui32, handshake.name = "addi41"} : <i7>
    %439 = buffer %438, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 13 : ui32, handshake.name = "buffer114"} : <i7>
    %440 = addi %380, %439 {handshake.bb = 13 : ui32, handshake.name = "addi17"} : <i7>
    %441 = buffer %429, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 13 : ui32, handshake.name = "buffer111"} : <i32>
    %442 = buffer %440, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 13 : ui32, handshake.name = "buffer115"} : <i7>
    %addressResult_86, %dataResult_87 = store[%442] %441 {handshake.bb = 13 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["load8", 3], ["store5", 3]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 1>, handshake.name = "store5"} : <i7>, <i32>, <i7>, <i32>
    %443 = addi %365, %391 {handshake.bb = 13 : ui32, handshake.name = "addi24"} : <i6>
    %444 = buffer %443, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 13 : ui32, handshake.name = "buffer116"} : <i6>
    %445:2 = fork [2] %444 {handshake.bb = 13 : ui32, handshake.name = "fork61"} : <i6>
    %446 = trunci %445#0 {handshake.bb = 13 : ui32, handshake.name = "trunci36"} : <i6> to <i5>
    %447 = cmpi ult, %445#1, %387 {handshake.bb = 13 : ui32, handshake.name = "cmpi6"} : <i6>
    %448 = buffer %447, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 13 : ui32, handshake.name = "buffer117"} : <i1>
    %449:4 = fork [4] %448 {handshake.bb = 13 : ui32, handshake.name = "fork62"} : <i1>
    %trueResult_88, %falseResult_89 = cond_br %449#0, %446 {handshake.bb = 13 : ui32, handshake.name = "cond_br29"} : <i1>, <i5>
    sink %falseResult_89 {handshake.name = "sink12"} : <i5>
    %trueResult_90, %falseResult_91 = cond_br %449#1, %371#0 {handshake.bb = 13 : ui32, handshake.name = "cond_br30"} : <i1>, <i5>
    %trueResult_92, %falseResult_93 = cond_br %449#2, %377#3 {handshake.bb = 13 : ui32, handshake.name = "cond_br31"} : <i1>, <i5>
    %450 = buffer %382#1, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 13 : ui32, handshake.name = "buffer97"} : <>
    %trueResult_94, %falseResult_95 = cond_br %449#3, %450 {handshake.bb = 13 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,inf], [1,inf], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "cond_br32"} : <i1>, <>
    %451 = extsi %falseResult_93 {handshake.bb = 14 : ui32, handshake.name = "extsi93"} : <i5> to <i6>
    %452 = source {handshake.bb = 14 : ui32, handshake.name = "source23"} : <>
    %453 = constant %452 {handshake.bb = 14 : ui32, handshake.name = "constant71", value = 10 : i5} : <>, <i5>
    %454 = extsi %453 {handshake.bb = 14 : ui32, handshake.name = "extsi94"} : <i5> to <i6>
    %455 = source {handshake.bb = 14 : ui32, handshake.name = "source24"} : <>
    %456 = constant %455 {handshake.bb = 14 : ui32, handshake.name = "constant72", value = 1 : i2} : <>, <i2>
    %457 = extsi %456 {handshake.bb = 14 : ui32, handshake.name = "extsi95"} : <i2> to <i6>
    %458 = addi %451, %457 {handshake.bb = 14 : ui32, handshake.name = "addi25"} : <i6>
    %459 = buffer %458, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 14 : ui32, handshake.name = "buffer118"} : <i6>
    %460:2 = fork [2] %459 {handshake.bb = 14 : ui32, handshake.name = "fork63"} : <i6>
    %461 = trunci %460#0 {handshake.bb = 14 : ui32, handshake.name = "trunci37"} : <i6> to <i5>
    %462 = cmpi ult, %460#1, %454 {handshake.bb = 14 : ui32, handshake.name = "cmpi7"} : <i6>
    %463 = buffer %462, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 14 : ui32, handshake.name = "buffer119"} : <i1>
    %464:3 = fork [3] %463 {handshake.bb = 14 : ui32, handshake.name = "fork64"} : <i1>
    %trueResult_96, %falseResult_97 = cond_br %464#0, %461 {handshake.bb = 14 : ui32, handshake.name = "cond_br33"} : <i1>, <i5>
    sink %falseResult_97 {handshake.name = "sink14"} : <i5>
    %trueResult_98, %falseResult_99 = cond_br %464#1, %falseResult_91 {handshake.bb = 14 : ui32, handshake.name = "cond_br34"} : <i1>, <i5>
    %trueResult_100, %falseResult_101 = cond_br %464#2, %falseResult_95 {handshake.bb = 14 : ui32, handshake.name = "cond_br35"} : <i1>, <>
    %465 = extsi %falseResult_99 {handshake.bb = 15 : ui32, handshake.name = "extsi96"} : <i5> to <i6>
    %466 = source {handshake.bb = 15 : ui32, handshake.name = "source25"} : <>
    %467 = constant %466 {handshake.bb = 15 : ui32, handshake.name = "constant73", value = 10 : i5} : <>, <i5>
    %468 = extsi %467 {handshake.bb = 15 : ui32, handshake.name = "extsi97"} : <i5> to <i6>
    %469 = source {handshake.bb = 15 : ui32, handshake.name = "source26"} : <>
    %470 = constant %469 {handshake.bb = 15 : ui32, handshake.name = "constant74", value = 1 : i2} : <>, <i2>
    %471 = extsi %470 {handshake.bb = 15 : ui32, handshake.name = "extsi98"} : <i2> to <i6>
    %472 = buffer %465, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 15 : ui32, handshake.name = "buffer120"} : <i6>
    %473 = addi %472, %471 {handshake.bb = 15 : ui32, handshake.name = "addi26"} : <i6>
    %474 = buffer %473, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 15 : ui32, handshake.name = "buffer121"} : <i6>
    %475:2 = fork [2] %474 {handshake.bb = 15 : ui32, handshake.name = "fork65"} : <i6>
    %476 = trunci %477 {handshake.bb = 15 : ui32, handshake.name = "trunci38"} : <i6> to <i5>
    %477 = buffer %475#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 15 : ui32, handshake.name = "buffer163"} : <i6>
    %478 = cmpi ult, %475#1, %468 {handshake.bb = 15 : ui32, handshake.name = "cmpi8"} : <i6>
    %479 = buffer %478, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 15 : ui32, handshake.name = "buffer122"} : <i1>
    %480:2 = fork [2] %479 {handshake.bb = 15 : ui32, handshake.name = "fork66"} : <i1>
    %trueResult_102, %falseResult_103 = cond_br %480#0, %476 {handshake.bb = 15 : ui32, handshake.name = "cond_br36"} : <i1>, <i5>
    sink %falseResult_103 {handshake.name = "sink16"} : <i5>
    %trueResult_104, %falseResult_105 = cond_br %480#1, %falseResult_101 {handshake.bb = 15 : ui32, handshake.name = "cond_br37"} : <i1>, <>
    %481:7 = fork [7] %falseResult_105 {handshake.bb = 16 : ui32, handshake.name = "fork67"} : <>
    end {handshake.bb = 16 : ui32, handshake.name = "end0"} %memEnd_5, %memEnd_3, %memEnd_1, %memEnd, %3#2, %2#2, %1#1, %0#1 : <>, <>, <>, <>, <>, <>, <>, <>
  }
}

