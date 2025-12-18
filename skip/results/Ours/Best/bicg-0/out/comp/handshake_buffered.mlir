module {
  handshake.func @bicg(%arg0: memref<900xi32>, %arg1: memref<30xi32>, %arg2: memref<30xi32>, %arg3: memref<30xi32>, %arg4: memref<30xi32>, %arg5: !handshake.control<>, %arg6: !handshake.control<>, %arg7: !handshake.control<>, %arg8: !handshake.control<>, %arg9: !handshake.control<>, %arg10: !handshake.control<>, ...) -> (!handshake.channel<i32>, !handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["a", "s", "q", "p", "r", "a_start", "s_start", "q_start", "p_start", "r_start", "start"], cfg.edges = "[0,1][2,2,3,cmpi0][1,2][3,1,4,cmpi1]", resNames = ["out0", "a_end", "s_end", "q_end", "p_end", "r_end", "end"]} {
    %0:4 = fork [4] %arg10 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork0"} : <>
    %outputs, %memEnd = mem_controller[%arg4 : memref<30xi32>] %arg9 (%addressResult_14) %111#4 {connectedBlocks = [2 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller3"} :    (!handshake.channel<i5>) -> !handshake.channel<i32>
    %outputs_0, %memEnd_1 = mem_controller[%arg3 : memref<30xi32>] %arg8 (%addressResult_18) %111#3 {connectedBlocks = [2 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller4"} :    (!handshake.channel<i5>) -> !handshake.channel<i32>
    %outputs_2:2, %memEnd_3 = mem_controller[%arg2 : memref<30xi32>] %arg7 (%addressResult, %93, %addressResult_32, %dataResult_33) %111#2 {connectedBlocks = [1 : i32, 3 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "5": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller5"} :    (!handshake.channel<i5>, !handshake.channel<i32>, !handshake.channel<i5>, !handshake.channel<i32>) -> (!handshake.channel<i32>, !handshake.control<>)
    %outputs_4:2, %memEnd_5 = mem_controller[%arg1 : memref<30xi32>] %arg6 (%48, %addressResult_12, %addressResult_16, %dataResult_17) %111#1 {connectedBlocks = [2 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "5": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller6"} :    (!handshake.channel<i32>, !handshake.channel<i5>, !handshake.channel<i5>, !handshake.channel<i32>) -> (!handshake.channel<i32>, !handshake.control<>)
    %outputs_6, %memEnd_7 = mem_controller[%arg0 : memref<900xi32>] %arg5 (%addressResult_10) %111#0 {connectedBlocks = [2 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller7"} :    (!handshake.channel<i10>) -> !handshake.channel<i32>
    %1 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant8", value = false} : <>, <i1>
    %2 = br %1 {handshake.bb = 0 : ui32, handshake.name = "br2"} : <i1>
    %3 = extsi %2 {handshake.bb = 0 : ui32, handshake.name = "extsi9"} : <i1> to <i6>
    %4 = br %0#3 {handshake.bb = 0 : ui32, handshake.name = "br3"} : <>
    %5 = mux %6 [%0#2, %trueResult_28] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux4"} : <i1>, [<>, <>] to <>
    %6 = init %108#2 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "init0"} : <i1>
    %7:2 = unbundle %15#0  {handshake.bb = 1 : ui32, handshake.name = "unbundle0"} : <i32> to _ 
    %8 = mux %index [%3, %trueResult_35] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i6>, <i6>] to <i6>
    %9 = buffer %8, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer3"} : <i6>
    %10:2 = fork [2] %9 {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <i6>
    %11 = trunci %10#0 {handshake.bb = 1 : ui32, handshake.name = "trunci0"} : <i6> to <i5>
    %result, %index = control_merge [%4, %trueResult_37]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %12:2 = fork [2] %result {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <>
    %13 = constant %12#0 {handshake.bb = 1 : ui32, handshake.name = "constant9", value = false} : <>, <i1>
    %14 = buffer %7#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer0"} : <>
    %addressResult, %dataResult = load[%11] %outputs_2#0 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["store1", 2, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i5>, <i32>, <i5>, <i32>
    %15:2 = fork [2] %dataResult {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <i32>
    %16 = br %13 {handshake.bb = 1 : ui32, handshake.name = "br4"} : <i1>
    %17 = extsi %16 {handshake.bb = 1 : ui32, handshake.name = "extsi8"} : <i1> to <i6>
    %18 = br %15#1 {handshake.bb = 1 : ui32, handshake.name = "br5"} : <i32>
    %19 = br %10#1 {handshake.bb = 1 : ui32, handshake.name = "br6"} : <i6>
    %20 = br %12#1 {handshake.bb = 1 : ui32, handshake.name = "br7"} : <>
    %trueResult, %falseResult = cond_br %21, %68 {handshake.bb = 2 : ui32, handshake.name = "cond_br11"} : <i1>, <>
    %21 = buffer %78#3, bufferType = FIFO_BREAK_NONE, numSlots = 4 {handshake.bb = 2 : ui32, handshake.name = "buffer7"} : <i1>
    %22 = buffer %5, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer2"} : <>
    %23 = mux %24 [%22, %trueResult] {ftd.phi, handshake.bb = 2 : ui32, handshake.name = "mux6"} : <i1>, [<>, <>] to <>
    %24 = init %25 {ftd.imerge, handshake.bb = 2 : ui32, handshake.name = "init2"} : <i1>
    %25 = buffer %78#2, bufferType = FIFO_BREAK_NONE, numSlots = 4 {handshake.bb = 2 : ui32, handshake.name = "buffer8"} : <i1>
    %26 = mux %45#1 [%17, %trueResult_20] {handshake.bb = 2 : ui32, handshake.name = "mux1"} : <i1>, [<i6>, <i6>] to <i6>
    %27 = buffer %26, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer5"} : <i6>
    %28:5 = fork [5] %27 {handshake.bb = 2 : ui32, handshake.name = "fork4"} : <i6>
    %29 = extsi %30 {handshake.bb = 2 : ui32, handshake.name = "extsi10"} : <i6> to <i10>
    %30 = buffer %28#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer10"} : <i6>
    %31 = extsi %28#3 {handshake.bb = 2 : ui32, handshake.name = "extsi11"} : <i6> to <i7>
    %32 = extsi %33 {handshake.bb = 2 : ui32, handshake.name = "extsi12"} : <i6> to <i32>
    %33 = buffer %28#4, bufferType = FIFO_BREAK_NONE, numSlots = 4 {handshake.bb = 2 : ui32, handshake.name = "buffer12"} : <i6>
    %34 = trunci %28#1 {handshake.bb = 2 : ui32, handshake.name = "trunci1"} : <i6> to <i5>
    %35 = trunci %36 {handshake.bb = 2 : ui32, handshake.name = "trunci2"} : <i6> to <i5>
    %36 = buffer %28#2, bufferType = FIFO_BREAK_NONE, numSlots = 3 {handshake.bb = 2 : ui32, handshake.name = "buffer14"} : <i6>
    %37 = mux %38 [%18, %trueResult_22] {handshake.bb = 2 : ui32, handshake.name = "mux2"} : <i1>, [<i32>, <i32>] to <i32>
    %38 = buffer %45#2, bufferType = FIFO_BREAK_NONE, numSlots = 4 {handshake.bb = 2 : ui32, handshake.name = "buffer15"} : <i1>
    %39 = mux %45#0 [%19, %trueResult_24] {handshake.bb = 2 : ui32, handshake.name = "mux3"} : <i1>, [<i6>, <i6>] to <i6>
    %40 = buffer %39, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer9"} : <i6>
    %41:3 = fork [3] %40 {handshake.bb = 2 : ui32, handshake.name = "fork5"} : <i6>
    %42 = extsi %41#2 {handshake.bb = 2 : ui32, handshake.name = "extsi13"} : <i6> to <i12>
    %43 = trunci %44 {handshake.bb = 2 : ui32, handshake.name = "trunci3"} : <i6> to <i5>
    %44 = buffer %41#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer18"} : <i6>
    %result_8, %index_9 = control_merge [%20, %trueResult_26]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>, <>] to <>, <i1>
    %45:3 = fork [3] %index_9 {handshake.bb = 2 : ui32, handshake.name = "fork6"} : <i1>
    %46:2 = fork [2] %result_8 {handshake.bb = 2 : ui32, handshake.name = "fork7"} : <>
    %47 = constant %46#0 {handshake.bb = 2 : ui32, handshake.name = "constant10", value = 1 : i2} : <>, <i2>
    %48 = extsi %47 {handshake.bb = 2 : ui32, handshake.name = "extsi2"} : <i2> to <i32>
    %49 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %50 = constant %49 {handshake.bb = 2 : ui32, handshake.name = "constant11", value = 30 : i6} : <>, <i6>
    %51:2 = fork [2] %50 {handshake.bb = 2 : ui32, handshake.name = "fork8"} : <i6>
    %52 = extsi %51#0 {handshake.bb = 2 : ui32, handshake.name = "extsi14"} : <i6> to <i12>
    %53 = extsi %54 {handshake.bb = 2 : ui32, handshake.name = "extsi15"} : <i6> to <i7>
    %54 = buffer %51#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer20"} : <i6>
    %55 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %56 = constant %55 {handshake.bb = 2 : ui32, handshake.name = "constant12", value = 1 : i2} : <>, <i2>
    %57 = extsi %56 {handshake.bb = 2 : ui32, handshake.name = "extsi16"} : <i2> to <i7>
    %58 = muli %42, %52 {handshake.bb = 2 : ui32, handshake.name = "muli2"} : <i12>
    %59 = trunci %58 {handshake.bb = 2 : ui32, handshake.name = "trunci4"} : <i12> to <i10>
    %60 = addi %29, %59 {handshake.bb = 2 : ui32, handshake.name = "addi2"} : <i10>
    %addressResult_10, %dataResult_11 = load[%60] %outputs_6 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i10>, <i32>, <i10>, <i32>
    %61:2 = fork [2] %dataResult_11 {handshake.bb = 2 : ui32, handshake.name = "fork9"} : <i32>
    %62 = buffer %23, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer4"} : <>
    %63 = gate %32, %62 {handshake.bb = 2 : ui32, handshake.name = "gate0"} : <i32>, !handshake.control<> to <i32>
    %64 = trunci %63 {handshake.bb = 2 : ui32, handshake.name = "trunci5"} : <i32> to <i5>
    %addressResult_12, %dataResult_13 = load[%64] %outputs_4#0 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["store0", 1, false], ["store0", 3, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load2"} : <i5>, <i32>, <i5>, <i32>
    %addressResult_14, %dataResult_15 = load[%43] %outputs {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load3"} : <i5>, <i32>, <i5>, <i32>
    %65 = muli %dataResult_15, %61#0 {handshake.bb = 2 : ui32, handshake.name = "muli0"} : <i32>
    %66 = addi %dataResult_13, %65 {handshake.bb = 2 : ui32, handshake.name = "addi0"} : <i32>
    %67 = buffer %doneResult, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer16"} : <>
    %68 = buffer %67, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer1"} : <>
    %addressResult_16, %dataResult_17, %doneResult = store[%35] %66 %outputs_4#1 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["load2", 1, false], ["store0", 1, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store0"} : <i5>, <i32>, <>, <i5>, <i32>, <>
    %addressResult_18, %dataResult_19 = load[%34] %outputs_0 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load4"} : <i5>, <i32>, <i5>, <i32>
    %69 = muli %61#1, %dataResult_19 {handshake.bb = 2 : ui32, handshake.name = "muli1"} : <i32>
    %70 = buffer %37, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer6"} : <i32>
    %71 = addi %70, %69 {handshake.bb = 2 : ui32, handshake.name = "addi1"} : <i32>
    %72 = addi %31, %57 {handshake.bb = 2 : ui32, handshake.name = "addi3"} : <i7>
    %73 = buffer %72, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer19"} : <i7>
    %74:2 = fork [2] %73 {handshake.bb = 2 : ui32, handshake.name = "fork10"} : <i7>
    %75 = trunci %74#0 {handshake.bb = 2 : ui32, handshake.name = "trunci6"} : <i7> to <i6>
    %76 = cmpi ult, %74#1, %53 {handshake.bb = 2 : ui32, handshake.name = "cmpi0"} : <i7>
    %77 = buffer %76, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer21"} : <i1>
    %78:6 = fork [6] %77 {handshake.bb = 2 : ui32, handshake.name = "fork11"} : <i1>
    %trueResult_20, %falseResult_21 = cond_br %78#0, %75 {handshake.bb = 2 : ui32, handshake.name = "cond_br2"} : <i1>, <i6>
    sink %falseResult_21 {handshake.name = "sink0"} : <i6>
    %79 = buffer %71, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer17"} : <i32>
    %trueResult_22, %falseResult_23 = cond_br %80, %79 {handshake.bb = 2 : ui32, handshake.name = "cond_br3"} : <i1>, <i32>
    %80 = buffer %78#4, bufferType = FIFO_BREAK_NONE, numSlots = 3 {handshake.bb = 2 : ui32, handshake.name = "buffer26"} : <i1>
    %81 = buffer %41#1, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer11"} : <i6>
    %trueResult_24, %falseResult_25 = cond_br %78#1, %81 {handshake.bb = 2 : ui32, handshake.name = "cond_br4"} : <i1>, <i6>
    %82 = buffer %46#1, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer13"} : <>
    %trueResult_26, %falseResult_27 = cond_br %78#5, %82 {handshake.bb = 2 : ui32, handshake.name = "cond_br5"} : <i1>, <>
    %trueResult_28, %falseResult_29 = cond_br %83, %falseResult {handshake.bb = 3 : ui32, handshake.name = "cond_br12"} : <i1>, <>
    %83 = buffer %108#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer30"} : <i1>
    sink %falseResult_29 {handshake.name = "sink1"} : <>
    %84 = merge %falseResult_25 {handshake.bb = 3 : ui32, handshake.name = "merge0"} : <i6>
    %85:2 = fork [2] %84 {handshake.bb = 3 : ui32, handshake.name = "fork12"} : <i6>
    %86 = extsi %85#0 {handshake.bb = 3 : ui32, handshake.name = "extsi17"} : <i6> to <i7>
    %87 = extsi %88 {handshake.bb = 3 : ui32, handshake.name = "extsi18"} : <i6> to <i32>
    %88 = buffer %85#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer32"} : <i6>
    %89 = merge %falseResult_23 {handshake.bb = 3 : ui32, handshake.name = "merge1"} : <i32>
    %90:2 = fork [2] %89 {handshake.bb = 3 : ui32, handshake.name = "fork13"} : <i32>
    %result_30, %index_31 = control_merge [%falseResult_27]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>] to <>, <i1>
    sink %index_31 {handshake.name = "sink2"} : <i1>
    %91:2 = fork [2] %result_30 {handshake.bb = 3 : ui32, handshake.name = "fork14"} : <>
    %92 = constant %91#0 {handshake.bb = 3 : ui32, handshake.name = "constant13", value = 1 : i2} : <>, <i2>
    %93 = extsi %92 {handshake.bb = 3 : ui32, handshake.name = "extsi5"} : <i2> to <i32>
    %94 = source {handshake.bb = 3 : ui32, handshake.name = "source2"} : <>
    %95 = constant %94 {handshake.bb = 3 : ui32, handshake.name = "constant14", value = 30 : i6} : <>, <i6>
    %96 = extsi %95 {handshake.bb = 3 : ui32, handshake.name = "extsi19"} : <i6> to <i7>
    %97 = source {handshake.bb = 3 : ui32, handshake.name = "source3"} : <>
    %98 = constant %97 {handshake.bb = 3 : ui32, handshake.name = "constant15", value = 1 : i2} : <>, <i2>
    %99 = extsi %98 {handshake.bb = 3 : ui32, handshake.name = "extsi20"} : <i2> to <i7>
    %100 = gate %87, %14 {handshake.bb = 3 : ui32, handshake.name = "gate1"} : <i32>, !handshake.control<> to <i32>
    %101 = trunci %100 {handshake.bb = 3 : ui32, handshake.name = "trunci7"} : <i32> to <i5>
    %addressResult_32, %dataResult_33, %doneResult_34 = store[%101] %90#0 %outputs_2#1 {handshake.bb = 3 : ui32, handshake.bufProps = #handshake<bufProps{"2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store1"} : <i5>, <i32>, <>, <i5>, <i32>, <>
    sink %doneResult_34 {handshake.name = "sink3"} : <>
    %102 = addi %86, %99 {handshake.bb = 3 : ui32, handshake.name = "addi4"} : <i7>
    %103 = buffer %102, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer22"} : <i7>
    %104:2 = fork [2] %103 {handshake.bb = 3 : ui32, handshake.name = "fork15"} : <i7>
    %105 = trunci %104#0 {handshake.bb = 3 : ui32, handshake.name = "trunci8"} : <i7> to <i6>
    %106 = cmpi ult, %104#1, %96 {handshake.bb = 3 : ui32, handshake.name = "cmpi1"} : <i7>
    %107 = buffer %106, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer23"} : <i1>
    %108:5 = fork [5] %107 {handshake.bb = 3 : ui32, handshake.name = "fork16"} : <i1>
    %trueResult_35, %falseResult_36 = cond_br %108#0, %105 {handshake.bb = 3 : ui32, handshake.name = "cond_br6"} : <i1>, <i6>
    sink %falseResult_36 {handshake.name = "sink4"} : <i6>
    %trueResult_37, %falseResult_38 = cond_br %108#3, %91#1 {handshake.bb = 3 : ui32, handshake.name = "cond_br7"} : <i1>, <>
    %trueResult_39, %falseResult_40 = cond_br %109, %90#1 {handshake.bb = 3 : ui32, handshake.name = "cond_br8"} : <i1>, <i32>
    %109 = buffer %108#4, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer38"} : <i1>
    sink %trueResult_39 {handshake.name = "sink5"} : <i32>
    %110 = merge %falseResult_40 {handshake.bb = 4 : ui32, handshake.name = "merge2"} : <i32>
    %result_41, %index_42 = control_merge [%falseResult_38]  {handshake.bb = 4 : ui32, handshake.name = "control_merge3"} : [<>] to <>, <i1>
    sink %index_42 {handshake.name = "sink6"} : <i1>
    %111:5 = fork [5] %result_41 {handshake.bb = 4 : ui32, handshake.name = "fork17"} : <>
    end {handshake.bb = 4 : ui32, handshake.name = "end0"} %110, %memEnd_7, %memEnd_5, %memEnd_3, %memEnd_1, %memEnd, %0#1 : <i32>, <>, <>, <>, <>, <>, <>
  }
}

