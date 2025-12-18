module {
  handshake.func @matrix_power(%arg0: memref<400xi32>, %arg1: memref<20xi32>, %arg2: memref<20xi32>, %arg3: memref<20xi32>, %arg4: !handshake.control<>, %arg5: !handshake.control<>, %arg6: !handshake.control<>, %arg7: !handshake.control<>, %arg8: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["mat", "row", "col", "a", "mat_start", "row_start", "col_start", "a_start", "start"], cfg.edges = "[0,1][2,2,3,cmpi0][1,2][3,1,4,cmpi1]", resNames = ["mat_end", "row_end", "col_end", "a_end", "end"]} {
    %0:14 = fork [14] %arg8 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork0"} : <>
    %outputs, %memEnd = mem_controller[%arg3 : memref<20xi32>] %arg7 (%addressResult_24) %300#3 {connectedBlocks = [2 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller3"} :    (!handshake.channel<i5>) -> !handshake.channel<i32>
    %outputs_0, %memEnd_1 = mem_controller[%arg2 : memref<20xi32>] %arg6 (%addressResult_26) %300#2 {connectedBlocks = [2 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller4"} :    (!handshake.channel<i5>) -> !handshake.channel<i32>
    %outputs_2, %memEnd_3 = mem_controller[%arg1 : memref<20xi32>] %arg5 (%addressResult) %300#1 {connectedBlocks = [2 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller5"} :    (!handshake.channel<i5>) -> !handshake.channel<i32>
    %outputs_4:3, %memEnd_5 = mem_controller[%arg0 : memref<400xi32>] %arg4 (%113, %addressResult_36, %addressResult_46, %addressResult_48, %dataResult_49) %300#0 {connectedBlocks = [2 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "5": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "6": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller6"} :    (!handshake.channel<i32>, !handshake.channel<i9>, !handshake.channel<i9>, !handshake.channel<i9>, !handshake.channel<i32>) -> (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>)
    %1 = constant %0#1 {handshake.bb = 0 : ui32, handshake.name = "constant18", value = 1000 : i11} : <>, <i11>
    %2:8 = fork [8] %1 {handshake.bb = 0 : ui32, handshake.name = "fork1"} : <i11>
    %3 = extsi %2#0 {handshake.bb = 1 : ui32, handshake.name = "extsi0"} : <i11> to <i32>
    %4 = extsi %2#1 {handshake.bb = 1 : ui32, handshake.name = "extsi1"} : <i11> to <i32>
    %5 = extsi %2#2 {handshake.bb = 1 : ui32, handshake.name = "extsi2"} : <i11> to <i32>
    %6 = extsi %2#3 {handshake.bb = 1 : ui32, handshake.name = "extsi3"} : <i11> to <i32>
    %7 = extsi %2#4 {handshake.bb = 1 : ui32, handshake.name = "extsi4"} : <i11> to <i32>
    %8 = extsi %2#5 {handshake.bb = 1 : ui32, handshake.name = "extsi5"} : <i11> to <i32>
    %9 = extsi %2#6 {handshake.bb = 1 : ui32, handshake.name = "extsi6"} : <i11> to <i32>
    %10 = extsi %2#7 {handshake.bb = 1 : ui32, handshake.name = "extsi7"} : <i11> to <i32>
    %11 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant19", value = 1 : i2} : <>, <i2>
    %12 = extsi %11 {handshake.bb = 0 : ui32, handshake.name = "extsi18"} : <i2> to <i6>
    %13 = mux %32#0 [%3, %283#0] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux4"} : <i1>, [<i32>, <i32>] to <i32>
    %14 = mux %32#1 [%0#12, %276#0] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux5"} : <i1>, [<>, <>] to <>
    %15 = mux %32#2 [%0#11, %276#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux6"} : <i1>, [<>, <>] to <>
    %16 = mux %32#3 [%4, %281#0] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux7"} : <i1>, [<i32>, <i32>] to <i32>
    %17 = mux %32#4 [%5, %285#0] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux8"} : <i1>, [<i32>, <i32>] to <i32>
    %18 = mux %32#5 [%6, %285#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux9"} : <i1>, [<i32>, <i32>] to <i32>
    %19 = mux %32#6 [%7, %281#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux10"} : <i1>, [<i32>, <i32>] to <i32>
    %20 = mux %32#7 [%8, %283#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux11"} : <i1>, [<i32>, <i32>] to <i32>
    %21 = mux %32#8 [%9, %282#0] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux12"} : <i1>, [<i32>, <i32>] to <i32>
    %22 = mux %32#9 [%10, %282#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux13"} : <i1>, [<i32>, <i32>] to <i32>
    %23 = mux %32#10 [%0#10, %284#0] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux14"} : <i1>, [<>, <>] to <>
    %24 = mux %32#11 [%0#9, %278#0] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux15"} : <i1>, [<>, <>] to <>
    %25 = mux %32#12 [%0#8, %280#0] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux16"} : <i1>, [<>, <>] to <>
    %26 = mux %32#13 [%0#7, %277#0] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux17"} : <i1>, [<>, <>] to <>
    %27 = mux %32#14 [%0#6, %278#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux18"} : <i1>, [<>, <>] to <>
    %28 = mux %32#15 [%0#5, %277#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux19"} : <i1>, [<>, <>] to <>
    %29 = mux %32#16 [%0#4, %284#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux20"} : <i1>, [<>, <>] to <>
    %30 = mux %32#17 [%0#3, %280#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux21"} : <i1>, [<>, <>] to <>
    %31 = init %299#10 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "init0"} : <i1>
    %32:18 = fork [18] %31 {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <i1>
    %33 = mux %index [%12, %trueResult_76] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i6>, <i6>] to <i6>
    %34 = buffer %33, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer13"} : <i6>
    %35:2 = fork [2] %34 {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <i6>
    %36 = extsi %35#1 {handshake.bb = 1 : ui32, handshake.name = "extsi19"} : <i6> to <i32>
    %result, %index = control_merge [%0#13, %trueResult_78]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %37:2 = fork [2] %result {handshake.bb = 1 : ui32, handshake.name = "fork4"} : <>
    %38 = constant %37#0 {handshake.bb = 1 : ui32, handshake.name = "constant20", value = false} : <>, <i1>
    %39 = source {handshake.bb = 1 : ui32, handshake.name = "source0"} : <>
    %40 = constant %39 {handshake.bb = 1 : ui32, handshake.name = "constant7", value = -1 : i32} : <>, <i32>
    %41 = addi %36, %40 {handshake.bb = 1 : ui32, handshake.name = "addi0"} : <i32>
    %42 = extsi %38 {handshake.bb = 1 : ui32, handshake.name = "extsi17"} : <i1> to <i6>
    %trueResult, %falseResult = cond_br %273#11, %265#1 {handshake.bb = 2 : ui32, handshake.name = "cond_br68"} : <i1>, <>
    %43:2 = fork [2] %trueResult {handshake.bb = 2 : ui32, handshake.name = "fork5"} : <>
    %trueResult_6, %falseResult_7 = cond_br %273#10, %266 {handshake.bb = 2 : ui32, handshake.name = "cond_br69"} : <i1>, <>
    %44:2 = fork [2] %trueResult_6 {handshake.bb = 2 : ui32, handshake.name = "fork6"} : <>
    %trueResult_8, %falseResult_9 = cond_br %273#9, %45 {handshake.bb = 2 : ui32, handshake.name = "cond_br70"} : <i1>, <i32>
    %45 = buffer %250#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer42"} : <i32>
    %46:2 = fork [2] %trueResult_8 {handshake.bb = 2 : ui32, handshake.name = "fork7"} : <i32>
    %trueResult_10, %falseResult_11 = cond_br %47, %263#1 {handshake.bb = 2 : ui32, handshake.name = "cond_br71"} : <i1>, <>
    %47 = buffer %273#8, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer43"} : <i1>
    %48:2 = fork [2] %trueResult_10 {handshake.bb = 2 : ui32, handshake.name = "fork8"} : <>
    %trueResult_12, %falseResult_13 = cond_br %273#7, %256 {handshake.bb = 2 : ui32, handshake.name = "cond_br72"} : <i1>, <i32>
    %49:2 = fork [2] %trueResult_12 {handshake.bb = 2 : ui32, handshake.name = "fork9"} : <i32>
    %trueResult_14, %falseResult_15 = cond_br %50, %259#1 {handshake.bb = 2 : ui32, handshake.name = "cond_br73"} : <i1>, <>
    %50 = buffer %273#6, bufferType = FIFO_BREAK_NONE, numSlots = 3 {handshake.bb = 2 : ui32, handshake.name = "buffer45"} : <i1>
    %51:2 = fork [2] %trueResult_14 {handshake.bb = 2 : ui32, handshake.name = "fork10"} : <>
    %trueResult_16, %falseResult_17 = cond_br %52, %261#1 {handshake.bb = 2 : ui32, handshake.name = "cond_br74"} : <i1>, <>
    %52 = buffer %273#5, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 2 : ui32, handshake.name = "buffer46"} : <i1>
    %53:2 = fork [2] %trueResult_16 {handshake.bb = 2 : ui32, handshake.name = "fork11"} : <>
    %trueResult_18, %falseResult_19 = cond_br %273#4, %255#1 {handshake.bb = 2 : ui32, handshake.name = "cond_br75"} : <i1>, <i32>
    %54:2 = fork [2] %trueResult_18 {handshake.bb = 2 : ui32, handshake.name = "fork12"} : <i32>
    %trueResult_20, %falseResult_21 = cond_br %273#3, %252#1 {handshake.bb = 2 : ui32, handshake.name = "cond_br76"} : <i1>, <i32>
    %55:2 = fork [2] %trueResult_20 {handshake.bb = 2 : ui32, handshake.name = "fork13"} : <i32>
    %56 = mux %90#0 [%13, %46#1] {ftd.phi, handshake.bb = 2 : ui32, handshake.name = "mux22"} : <i1>, [<i32>, <i32>] to <i32>
    %57 = buffer %14, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer2"} : <>
    %58 = mux %90#1 [%57, %44#1] {ftd.phi, handshake.bb = 2 : ui32, handshake.name = "mux23"} : <i1>, [<>, <>] to <>
    %59 = buffer %15, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer3"} : <>
    %60 = mux %90#2 [%59, %44#0] {ftd.phi, handshake.bb = 2 : ui32, handshake.name = "mux24"} : <i1>, [<>, <>] to <>
    %61 = mux %90#3 [%16, %54#1] {ftd.phi, handshake.bb = 2 : ui32, handshake.name = "mux25"} : <i1>, [<i32>, <i32>] to <i32>
    %62 = mux %90#4 [%17, %55#1] {ftd.phi, handshake.bb = 2 : ui32, handshake.name = "mux26"} : <i1>, [<i32>, <i32>] to <i32>
    %63 = mux %90#5 [%18, %55#0] {ftd.phi, handshake.bb = 2 : ui32, handshake.name = "mux27"} : <i1>, [<i32>, <i32>] to <i32>
    %64 = mux %90#6 [%19, %54#0] {ftd.phi, handshake.bb = 2 : ui32, handshake.name = "mux28"} : <i1>, [<i32>, <i32>] to <i32>
    %65 = mux %90#7 [%20, %46#0] {ftd.phi, handshake.bb = 2 : ui32, handshake.name = "mux29"} : <i1>, [<i32>, <i32>] to <i32>
    %66 = mux %90#8 [%21, %49#1] {ftd.phi, handshake.bb = 2 : ui32, handshake.name = "mux30"} : <i1>, [<i32>, <i32>] to <i32>
    %67 = mux %90#9 [%22, %49#0] {ftd.phi, handshake.bb = 2 : ui32, handshake.name = "mux31"} : <i1>, [<i32>, <i32>] to <i32>
    %68 = buffer %23, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer4"} : <>
    %69 = mux %70 [%68, %53#1] {ftd.phi, handshake.bb = 2 : ui32, handshake.name = "mux32"} : <i1>, [<>, <>] to <>
    %70 = buffer %90#10, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer69"} : <i1>
    %71 = buffer %24, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer5"} : <>
    %72 = mux %90#11 [%71, %43#1] {ftd.phi, handshake.bb = 2 : ui32, handshake.name = "mux33"} : <i1>, [<>, <>] to <>
    %73 = buffer %25, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer6"} : <>
    %74 = buffer %73, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer7"} : <>
    %75 = mux %76 [%74, %51#1] {ftd.phi, handshake.bb = 2 : ui32, handshake.name = "mux34"} : <i1>, [<>, <>] to <>
    %76 = buffer %90#12, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 2 : ui32, handshake.name = "buffer71"} : <i1>
    %77 = buffer %26, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer8"} : <>
    %78 = mux %90#13 [%77, %48#1] {ftd.phi, handshake.bb = 2 : ui32, handshake.name = "mux35"} : <i1>, [<>, <>] to <>
    %79 = buffer %27, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer9"} : <>
    %80 = mux %90#14 [%79, %43#0] {ftd.phi, handshake.bb = 2 : ui32, handshake.name = "mux36"} : <i1>, [<>, <>] to <>
    %81 = buffer %28, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer10"} : <>
    %82 = mux %90#15 [%81, %48#0] {ftd.phi, handshake.bb = 2 : ui32, handshake.name = "mux37"} : <i1>, [<>, <>] to <>
    %83 = buffer %29, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer11"} : <>
    %84 = mux %85 [%83, %53#0] {ftd.phi, handshake.bb = 2 : ui32, handshake.name = "mux38"} : <i1>, [<>, <>] to <>
    %85 = buffer %90#16, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer75"} : <i1>
    %86 = buffer %30, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer12"} : <>
    %87 = mux %88 [%86, %51#0] {ftd.phi, handshake.bb = 2 : ui32, handshake.name = "mux39"} : <i1>, [<>, <>] to <>
    %88 = buffer %90#17, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 2 : ui32, handshake.name = "buffer76"} : <i1>
    %89 = init %273#2 {ftd.imerge, handshake.bb = 2 : ui32, handshake.name = "init18"} : <i1>
    %90:18 = fork [18] %89 {handshake.bb = 2 : ui32, handshake.name = "fork14"} : <i1>
    %91 = mux %109#1 [%42, %trueResult_50] {handshake.bb = 2 : ui32, handshake.name = "mux1"} : <i1>, [<i6>, <i6>] to <i6>
    %92 = buffer %91, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer37"} : <i6>
    %93:4 = fork [4] %92 {handshake.bb = 2 : ui32, handshake.name = "fork15"} : <i6>
    %94 = extsi %93#3 {handshake.bb = 2 : ui32, handshake.name = "extsi20"} : <i6> to <i7>
    %95 = trunci %93#0 {handshake.bb = 2 : ui32, handshake.name = "trunci0"} : <i6> to <i5>
    %96 = trunci %93#1 {handshake.bb = 2 : ui32, handshake.name = "trunci1"} : <i6> to <i5>
    %97 = trunci %93#2 {handshake.bb = 2 : ui32, handshake.name = "trunci2"} : <i6> to <i5>
    %98 = mux %109#0 [%35#0, %trueResult_52] {handshake.bb = 2 : ui32, handshake.name = "mux2"} : <i1>, [<i6>, <i6>] to <i6>
    %99 = buffer %98, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer38"} : <i6>
    %100 = buffer %99, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer39"} : <i6>
    %101:2 = fork [2] %100 {handshake.bb = 2 : ui32, handshake.name = "fork16"} : <i6>
    %102 = extsi %101#1 {handshake.bb = 2 : ui32, handshake.name = "extsi21"} : <i6> to <i32>
    %103:4 = fork [4] %102 {handshake.bb = 2 : ui32, handshake.name = "fork17"} : <i32>
    %104 = buffer %41, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer14"} : <i32>
    %105 = mux %109#2 [%104, %trueResult_54] {handshake.bb = 2 : ui32, handshake.name = "mux3"} : <i1>, [<i32>, <i32>] to <i32>
    %106 = buffer %105, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer40"} : <i32>
    %107 = buffer %106, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer41"} : <i32>
    %108:3 = fork [3] %107 {handshake.bb = 2 : ui32, handshake.name = "fork18"} : <i32>
    %result_22, %index_23 = control_merge [%37#1, %trueResult_56]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>, <>] to <>, <i1>
    %109:3 = fork [3] %index_23 {handshake.bb = 2 : ui32, handshake.name = "fork19"} : <i1>
    %110 = buffer %result_22, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer44"} : <>
    %111:2 = fork [2] %110 {handshake.bb = 2 : ui32, handshake.name = "fork20"} : <>
    %112 = constant %111#0 {handshake.bb = 2 : ui32, handshake.name = "constant21", value = 1 : i2} : <>, <i2>
    %113 = extsi %112 {handshake.bb = 2 : ui32, handshake.name = "extsi10"} : <i2> to <i32>
    %114 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %115 = constant %114 {handshake.bb = 2 : ui32, handshake.name = "constant22", value = 1 : i2} : <>, <i2>
    %116 = extsi %115 {handshake.bb = 2 : ui32, handshake.name = "extsi22"} : <i2> to <i7>
    %117 = source {handshake.bb = 2 : ui32, handshake.name = "source2"} : <>
    %118 = constant %117 {handshake.bb = 2 : ui32, handshake.name = "constant23", value = 20 : i6} : <>, <i6>
    %119 = extsi %118 {handshake.bb = 2 : ui32, handshake.name = "extsi23"} : <i6> to <i7>
    %120 = source {handshake.bb = 2 : ui32, handshake.name = "source3"} : <>
    %121 = constant %120 {handshake.bb = 2 : ui32, handshake.name = "constant24", value = 4 : i4} : <>, <i4>
    %122 = extsi %121 {handshake.bb = 2 : ui32, handshake.name = "extsi13"} : <i4> to <i32>
    %123:3 = fork [3] %122 {handshake.bb = 2 : ui32, handshake.name = "fork21"} : <i32>
    %124 = source {handshake.bb = 2 : ui32, handshake.name = "source4"} : <>
    %125 = constant %124 {handshake.bb = 2 : ui32, handshake.name = "constant25", value = 2 : i3} : <>, <i3>
    %126 = extsi %125 {handshake.bb = 2 : ui32, handshake.name = "extsi14"} : <i3> to <i32>
    %127:3 = fork [3] %126 {handshake.bb = 2 : ui32, handshake.name = "fork22"} : <i32>
    %addressResult, %dataResult = load[%97] %outputs_2 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i5>, <i32>, <i5>, <i32>
    %128:2 = fork [2] %dataResult {handshake.bb = 2 : ui32, handshake.name = "fork23"} : <i32>
    %addressResult_24, %dataResult_25 = load[%96] %outputs {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i5>, <i32>, <i5>, <i32>
    %addressResult_26, %dataResult_27 = load[%95] %outputs_0 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load2"} : <i5>, <i32>, <i5>, <i32>
    %129 = shli %108#2, %127#0 {handshake.bb = 2 : ui32, handshake.name = "shli0"} : <i32>
    %130 = shli %108#1, %123#0 {handshake.bb = 2 : ui32, handshake.name = "shli1"} : <i32>
    %131 = buffer %129, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer47"} : <i32>
    %132 = buffer %130, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer48"} : <i32>
    %133 = addi %131, %132 {handshake.bb = 2 : ui32, handshake.name = "addi7"} : <i32>
    %134 = buffer %133, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer49"} : <i32>
    %135 = addi %dataResult_27, %134 {handshake.bb = 2 : ui32, handshake.name = "addi2"} : <i32>
    %136:2 = fork [2] %135 {handshake.bb = 2 : ui32, handshake.name = "fork24"} : <i32>
    %137 = buffer %60, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer17"} : <>
    %138 = gate %136#1, %137 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "gate0"} : <i32>, !handshake.control<> to <i32>
    %139 = buffer %138, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer50"} : <i32>
    %140:4 = fork [4] %139 {handshake.bb = 2 : ui32, handshake.name = "fork25"} : <i32>
    %141 = buffer %56, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer15"} : <i32>
    %142 = cmpi ne, %140#3, %141 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi2"} : <i32>
    %143:2 = fork [2] %142 {handshake.bb = 2 : ui32, handshake.name = "fork26"} : <i1>
    %144 = buffer %63, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer20"} : <i32>
    %145 = cmpi ne, %140#2, %144 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi3"} : <i32>
    %146:2 = fork [2] %145 {handshake.bb = 2 : ui32, handshake.name = "fork27"} : <i1>
    %147 = buffer %61, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer18"} : <i32>
    %148 = cmpi ne, %140#1, %147 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi4"} : <i32>
    %149:2 = fork [2] %148 {handshake.bb = 2 : ui32, handshake.name = "fork28"} : <i1>
    %150 = buffer %66, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer23"} : <i32>
    %151 = cmpi ne, %140#0, %150 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi5"} : <i32>
    %152:2 = fork [2] %151 {handshake.bb = 2 : ui32, handshake.name = "fork29"} : <i1>
    %153 = buffer %87, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer36"} : <>
    %trueResult_28, %falseResult_29 = cond_br %154, %153 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br42"} : <i1>, <>
    %154 = buffer %143#1, bufferType = FIFO_BREAK_NONE, numSlots = 3 {handshake.bb = 2 : ui32, handshake.name = "buffer95"} : <i1>
    sink %trueResult_28 {handshake.name = "sink0"} : <>
    %155 = buffer %84, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer34"} : <>
    %156 = buffer %155, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer35"} : <>
    %trueResult_30, %falseResult_31 = cond_br %157, %156 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br43"} : <i1>, <>
    %157 = buffer %146#1, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 2 : ui32, handshake.name = "buffer96"} : <i1>
    sink %trueResult_30 {handshake.name = "sink1"} : <>
    %158 = buffer %78, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer29"} : <>
    %159 = buffer %158, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer30"} : <>
    %trueResult_32, %falseResult_33 = cond_br %160, %159 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br44"} : <i1>, <>
    %160 = buffer %149#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer97"} : <i1>
    sink %trueResult_32 {handshake.name = "sink2"} : <>
    %161 = buffer %72, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer26"} : <>
    %162 = buffer %161, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer27"} : <>
    %trueResult_34, %falseResult_35 = cond_br %152#1, %162 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br45"} : <i1>, <>
    sink %trueResult_34 {handshake.name = "sink3"} : <>
    %163 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source7"} : <>
    %164 = buffer %falseResult_29, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer51"} : <>
    %165 = mux %143#0 [%164, %163] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux40"} : <i1>, [<>, <>] to <>
    %166 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source8"} : <>
    %167 = mux %146#0 [%falseResult_31, %166] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux41"} : <i1>, [<>, <>] to <>
    %168 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source9"} : <>
    %169 = mux %149#0 [%falseResult_33, %168] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux42"} : <i1>, [<>, <>] to <>
    %170 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source10"} : <>
    %171 = mux %152#0 [%falseResult_35, %170] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux43"} : <i1>, [<>, <>] to <>
    %172 = buffer %165, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer52"} : <>
    %173 = buffer %167, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer53"} : <>
    %174 = buffer %169, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer54"} : <>
    %175 = buffer %171, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer55"} : <>
    %176 = join %172, %173, %174, %175 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "join0"} : <>
    %177 = gate %136#0, %176 {handshake.bb = 2 : ui32, handshake.name = "gate1"} : <i32>, !handshake.control<> to <i32>
    %178 = trunci %177 {handshake.bb = 2 : ui32, handshake.name = "trunci3"} : <i32> to <i9>
    %addressResult_36, %dataResult_37 = load[%178] %outputs_4#0 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["store0", 0, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load3"} : <i9>, <i32>, <i9>, <i32>
    %179 = muli %dataResult_25, %dataResult_37 {handshake.bb = 2 : ui32, handshake.name = "muli0"} : <i32>
    %180 = shli %103#0, %127#1 {handshake.bb = 2 : ui32, handshake.name = "shli2"} : <i32>
    %181 = shli %103#1, %123#1 {handshake.bb = 2 : ui32, handshake.name = "shli3"} : <i32>
    %182 = buffer %180, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer56"} : <i32>
    %183 = buffer %181, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer57"} : <i32>
    %184 = addi %182, %183 {handshake.bb = 2 : ui32, handshake.name = "addi8"} : <i32>
    %185 = buffer %184, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer58"} : <i32>
    %186 = addi %128#0, %185 {handshake.bb = 2 : ui32, handshake.name = "addi3"} : <i32>
    %187:2 = fork [2] %186 {handshake.bb = 2 : ui32, handshake.name = "fork30"} : <i32>
    %188 = buffer %58, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer16"} : <>
    %189 = gate %187#1, %188 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "gate2"} : <i32>, !handshake.control<> to <i32>
    %190 = buffer %189, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer59"} : <i32>
    %191:4 = fork [4] %190 {handshake.bb = 2 : ui32, handshake.name = "fork31"} : <i32>
    %192 = buffer %65, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer22"} : <i32>
    %193 = cmpi ne, %191#3, %192 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi6"} : <i32>
    %194:2 = fork [2] %193 {handshake.bb = 2 : ui32, handshake.name = "fork32"} : <i1>
    %195 = buffer %62, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer19"} : <i32>
    %196 = cmpi ne, %191#2, %195 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi7"} : <i32>
    %197:2 = fork [2] %196 {handshake.bb = 2 : ui32, handshake.name = "fork33"} : <i1>
    %198 = buffer %64, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer21"} : <i32>
    %199 = cmpi ne, %191#1, %198 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi8"} : <i32>
    %200:2 = fork [2] %199 {handshake.bb = 2 : ui32, handshake.name = "fork34"} : <i1>
    %201 = buffer %67, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer24"} : <i32>
    %202 = cmpi ne, %191#0, %201 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi9"} : <i32>
    %203:2 = fork [2] %202 {handshake.bb = 2 : ui32, handshake.name = "fork35"} : <i1>
    %204 = buffer %75, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer28"} : <>
    %trueResult_38, %falseResult_39 = cond_br %205, %204 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br46"} : <i1>, <>
    %205 = buffer %194#1, bufferType = FIFO_BREAK_NONE, numSlots = 3 {handshake.bb = 2 : ui32, handshake.name = "buffer114"} : <i1>
    sink %trueResult_38 {handshake.name = "sink4"} : <>
    %206 = buffer %69, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer25"} : <>
    %trueResult_40, %falseResult_41 = cond_br %207, %206 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br47"} : <i1>, <>
    %207 = buffer %197#1, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 2 : ui32, handshake.name = "buffer115"} : <i1>
    sink %trueResult_40 {handshake.name = "sink5"} : <>
    %208 = buffer %82, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer33"} : <>
    %trueResult_42, %falseResult_43 = cond_br %209, %208 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br48"} : <i1>, <>
    %209 = buffer %200#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer116"} : <i1>
    sink %trueResult_42 {handshake.name = "sink6"} : <>
    %210 = buffer %80, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer31"} : <>
    %211 = buffer %210, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer32"} : <>
    %trueResult_44, %falseResult_45 = cond_br %212, %211 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br49"} : <i1>, <>
    %212 = buffer %203#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer117"} : <i1>
    sink %trueResult_44 {handshake.name = "sink7"} : <>
    %213 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source11"} : <>
    %214 = mux %215 [%falseResult_39, %213] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux44"} : <i1>, [<>, <>] to <>
    %215 = buffer %194#0, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 2 : ui32, handshake.name = "buffer118"} : <i1>
    %216 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source12"} : <>
    %217 = buffer %falseResult_41, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer60"} : <>
    %218 = mux %219 [%217, %216] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux45"} : <i1>, [<>, <>] to <>
    %219 = buffer %197#0, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 2 : ui32, handshake.name = "buffer119"} : <i1>
    %220 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source13"} : <>
    %221 = buffer %falseResult_43, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer61"} : <>
    %222 = mux %223 [%221, %220] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux46"} : <i1>, [<>, <>] to <>
    %223 = buffer %200#0, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 2 : ui32, handshake.name = "buffer120"} : <i1>
    %224 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source14"} : <>
    %225 = mux %226 [%falseResult_45, %224] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux47"} : <i1>, [<>, <>] to <>
    %226 = buffer %203#0, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 2 : ui32, handshake.name = "buffer121"} : <i1>
    %227 = buffer %214, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer62"} : <>
    %228 = buffer %218, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer63"} : <>
    %229 = buffer %222, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer64"} : <>
    %230 = buffer %225, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer65"} : <>
    %231 = join %227, %228, %229, %230 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "join1"} : <>
    %232 = gate %233, %231 {handshake.bb = 2 : ui32, handshake.name = "gate3"} : <i32>, !handshake.control<> to <i32>
    %233 = buffer %187#0, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 2 : ui32, handshake.name = "buffer122"} : <i32>
    %234 = trunci %232 {handshake.bb = 2 : ui32, handshake.name = "trunci4"} : <i32> to <i9>
    %addressResult_46, %dataResult_47 = load[%234] %outputs_4#1 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["store0", 0, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load4"} : <i9>, <i32>, <i9>, <i32>
    %235 = addi %dataResult_47, %179 {handshake.bb = 2 : ui32, handshake.name = "addi1"} : <i32>
    %236 = shli %103#2, %127#2 {handshake.bb = 2 : ui32, handshake.name = "shli4"} : <i32>
    %237 = shli %238, %123#2 {handshake.bb = 2 : ui32, handshake.name = "shli5"} : <i32>
    %238 = buffer %103#3, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer126"} : <i32>
    %239 = buffer %236, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer66"} : <i32>
    %240 = buffer %237, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer67"} : <i32>
    %241 = addi %239, %240 {handshake.bb = 2 : ui32, handshake.name = "addi9"} : <i32>
    %242 = buffer %241, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer68"} : <i32>
    %243 = addi %244, %242 {handshake.bb = 2 : ui32, handshake.name = "addi4"} : <i32>
    %244 = buffer %128#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer127"} : <i32>
    %245 = buffer %243, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer70"} : <i32>
    %246:2 = fork [2] %245 {handshake.bb = 2 : ui32, handshake.name = "fork36"} : <i32>
    %247 = trunci %248 {handshake.bb = 2 : ui32, handshake.name = "trunci5"} : <i32> to <i9>
    %248 = buffer %246#0, bufferType = FIFO_BREAK_NONE, numSlots = 3 {handshake.bb = 2 : ui32, handshake.name = "buffer128"} : <i32>
    %249 = buffer %246#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer0"} : <i32>
    %250:2 = fork [2] %249 {handshake.bb = 2 : ui32, handshake.name = "fork37"} : <i32>
    %251 = init %250#0 {handshake.bb = 2 : ui32, handshake.name = "init36"} : <i32>
    %252:2 = fork [2] %251 {handshake.bb = 2 : ui32, handshake.name = "fork38"} : <i32>
    %253 = init %254 {handshake.bb = 2 : ui32, handshake.name = "init37"} : <i32>
    %254 = buffer %252#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer131"} : <i32>
    %255:2 = fork [2] %253 {handshake.bb = 2 : ui32, handshake.name = "fork39"} : <i32>
    %256 = init %255#0 {handshake.bb = 2 : ui32, handshake.name = "init38"} : <i32>
    %257 = buffer %doneResult, bufferType = FIFO_BREAK_NONE, numSlots = 1 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "buffer1"} : <>
    %258 = buffer %257, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer72"} : <>
    %259:2 = fork [2] %258 {handshake.bb = 2 : ui32, handshake.name = "fork40"} : <>
    %260 = init %259#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init39"} : <>
    %261:2 = fork [2] %260 {handshake.bb = 2 : ui32, handshake.name = "fork41"} : <>
    %262 = init %261#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init40"} : <>
    %263:2 = fork [2] %262 {handshake.bb = 2 : ui32, handshake.name = "fork42"} : <>
    %264 = init %263#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init41"} : <>
    %265:2 = fork [2] %264 {handshake.bb = 2 : ui32, handshake.name = "fork43"} : <>
    %266 = init %265#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init42"} : <>
    %addressResult_48, %dataResult_49, %doneResult = store[%247] %235 %outputs_4#2 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["load3", 0, false], ["load4", 0, false], ["store0", 0, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store0"} : <i9>, <i32>, <>, <i9>, <i32>, <>
    %267 = addi %94, %116 {handshake.bb = 2 : ui32, handshake.name = "addi5"} : <i7>
    %268 = buffer %267, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer73"} : <i7>
    %269:2 = fork [2] %268 {handshake.bb = 2 : ui32, handshake.name = "fork44"} : <i7>
    %270 = trunci %269#0 {handshake.bb = 2 : ui32, handshake.name = "trunci6"} : <i7> to <i6>
    %271 = cmpi ult, %269#1, %119 {handshake.bb = 2 : ui32, handshake.name = "cmpi0"} : <i7>
    %272 = buffer %271, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer74"} : <i1>
    %273:14 = fork [14] %272 {handshake.bb = 2 : ui32, handshake.name = "fork45"} : <i1>
    %trueResult_50, %falseResult_51 = cond_br %273#0, %270 {handshake.bb = 2 : ui32, handshake.name = "cond_br2"} : <i1>, <i6>
    sink %falseResult_51 {handshake.name = "sink8"} : <i6>
    %trueResult_52, %falseResult_53 = cond_br %273#1, %274 {handshake.bb = 2 : ui32, handshake.name = "cond_br3"} : <i1>, <i6>
    %274 = buffer %101#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer137"} : <i6>
    %trueResult_54, %falseResult_55 = cond_br %273#12, %275 {handshake.bb = 2 : ui32, handshake.name = "cond_br4"} : <i1>, <i32>
    %275 = buffer %108#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer139"} : <i32>
    sink %falseResult_55 {handshake.name = "sink9"} : <i32>
    %trueResult_56, %falseResult_57 = cond_br %273#13, %111#1 {handshake.bb = 2 : ui32, handshake.name = "cond_br5"} : <i1>, <>
    %trueResult_58, %falseResult_59 = cond_br %299#9, %falseResult_7 {handshake.bb = 3 : ui32, handshake.name = "cond_br77"} : <i1>, <>
    sink %falseResult_59 {handshake.name = "sink10"} : <>
    %276:2 = fork [2] %trueResult_58 {handshake.bb = 3 : ui32, handshake.name = "fork46"} : <>
    %trueResult_60, %falseResult_61 = cond_br %299#8, %falseResult_11 {handshake.bb = 3 : ui32, handshake.name = "cond_br78"} : <i1>, <>
    sink %falseResult_61 {handshake.name = "sink11"} : <>
    %277:2 = fork [2] %trueResult_60 {handshake.bb = 3 : ui32, handshake.name = "fork47"} : <>
    %trueResult_62, %falseResult_63 = cond_br %299#7, %falseResult {handshake.bb = 3 : ui32, handshake.name = "cond_br79"} : <i1>, <>
    sink %falseResult_63 {handshake.name = "sink12"} : <>
    %278:2 = fork [2] %trueResult_62 {handshake.bb = 3 : ui32, handshake.name = "fork48"} : <>
    %trueResult_64, %falseResult_65 = cond_br %279, %falseResult_15 {handshake.bb = 3 : ui32, handshake.name = "cond_br80"} : <i1>, <>
    %279 = buffer %299#6, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer144"} : <i1>
    sink %falseResult_65 {handshake.name = "sink13"} : <>
    %280:2 = fork [2] %trueResult_64 {handshake.bb = 3 : ui32, handshake.name = "fork49"} : <>
    %trueResult_66, %falseResult_67 = cond_br %299#5, %falseResult_19 {handshake.bb = 3 : ui32, handshake.name = "cond_br81"} : <i1>, <i32>
    sink %falseResult_67 {handshake.name = "sink14"} : <i32>
    %281:2 = fork [2] %trueResult_66 {handshake.bb = 3 : ui32, handshake.name = "fork50"} : <i32>
    %trueResult_68, %falseResult_69 = cond_br %299#4, %falseResult_13 {handshake.bb = 3 : ui32, handshake.name = "cond_br82"} : <i1>, <i32>
    sink %falseResult_69 {handshake.name = "sink15"} : <i32>
    %282:2 = fork [2] %trueResult_68 {handshake.bb = 3 : ui32, handshake.name = "fork51"} : <i32>
    %trueResult_70, %falseResult_71 = cond_br %299#3, %falseResult_9 {handshake.bb = 3 : ui32, handshake.name = "cond_br83"} : <i1>, <i32>
    sink %falseResult_71 {handshake.name = "sink16"} : <i32>
    %283:2 = fork [2] %trueResult_70 {handshake.bb = 3 : ui32, handshake.name = "fork52"} : <i32>
    %trueResult_72, %falseResult_73 = cond_br %299#2, %falseResult_17 {handshake.bb = 3 : ui32, handshake.name = "cond_br84"} : <i1>, <>
    sink %falseResult_73 {handshake.name = "sink17"} : <>
    %284:2 = fork [2] %trueResult_72 {handshake.bb = 3 : ui32, handshake.name = "fork53"} : <>
    %trueResult_74, %falseResult_75 = cond_br %299#1, %falseResult_21 {handshake.bb = 3 : ui32, handshake.name = "cond_br85"} : <i1>, <i32>
    sink %falseResult_75 {handshake.name = "sink18"} : <i32>
    %285:2 = fork [2] %trueResult_74 {handshake.bb = 3 : ui32, handshake.name = "fork54"} : <i32>
    %286 = extsi %falseResult_53 {handshake.bb = 3 : ui32, handshake.name = "extsi24"} : <i6> to <i7>
    %287 = source {handshake.bb = 3 : ui32, handshake.name = "source5"} : <>
    %288 = constant %287 {handshake.bb = 3 : ui32, handshake.name = "constant26", value = 1 : i2} : <>, <i2>
    %289 = extsi %288 {handshake.bb = 3 : ui32, handshake.name = "extsi25"} : <i2> to <i7>
    %290 = source {handshake.bb = 3 : ui32, handshake.name = "source6"} : <>
    %291 = constant %290 {handshake.bb = 3 : ui32, handshake.name = "constant27", value = 20 : i6} : <>, <i6>
    %292 = extsi %291 {handshake.bb = 3 : ui32, handshake.name = "extsi26"} : <i6> to <i7>
    %293 = addi %286, %289 {handshake.bb = 3 : ui32, handshake.name = "addi6"} : <i7>
    %294 = buffer %293, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer77"} : <i7>
    %295:2 = fork [2] %294 {handshake.bb = 3 : ui32, handshake.name = "fork55"} : <i7>
    %296 = trunci %295#0 {handshake.bb = 3 : ui32, handshake.name = "trunci7"} : <i7> to <i6>
    %297 = cmpi ult, %295#1, %292 {handshake.bb = 3 : ui32, handshake.name = "cmpi1"} : <i7>
    %298 = buffer %297, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer78"} : <i1>
    %299:12 = fork [12] %298 {handshake.bb = 3 : ui32, handshake.name = "fork56"} : <i1>
    %trueResult_76, %falseResult_77 = cond_br %299#0, %296 {handshake.bb = 3 : ui32, handshake.name = "cond_br6"} : <i1>, <i6>
    sink %falseResult_77 {handshake.name = "sink20"} : <i6>
    %trueResult_78, %falseResult_79 = cond_br %299#11, %falseResult_57 {handshake.bb = 3 : ui32, handshake.name = "cond_br7"} : <i1>, <>
    %300:4 = fork [4] %falseResult_79 {handshake.bb = 4 : ui32, handshake.name = "fork57"} : <>
    end {handshake.bb = 4 : ui32, handshake.name = "end0"} %memEnd_5, %memEnd_3, %memEnd_1, %memEnd, %0#2 : <>, <>, <>, <>, <>
  }
}

