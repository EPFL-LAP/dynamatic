module {
  handshake.func @bicg(%arg0: memref<900xi32>, %arg1: memref<30xi32>, %arg2: memref<30xi32>, %arg3: memref<30xi32>, %arg4: memref<30xi32>, %arg5: !handshake.control<>, %arg6: !handshake.control<>, %arg7: !handshake.control<>, %arg8: !handshake.control<>, %arg9: !handshake.control<>, %arg10: !handshake.control<>, ...) -> (!handshake.channel<i32>, !handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["a", "s", "q", "p", "r", "a_start", "s_start", "q_start", "p_start", "r_start", "start"], cfg.edges = "[0,1][2,2,3,cmpi0][1,2][3,1,4,cmpi1]", resNames = ["out0", "a_end", "s_end", "q_end", "p_end", "r_end", "end"]} {
    %0:4 = fork [4] %arg10 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork0"} : <>
    %outputs, %memEnd = mem_controller[%arg4 : memref<30xi32>] %arg9 (%addressResult_14) %102#4 {connectedBlocks = [2 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller3"} :    (!handshake.channel<i5>) -> !handshake.channel<i32>
    %outputs_0, %memEnd_1 = mem_controller[%arg3 : memref<30xi32>] %arg8 (%addressResult_18) %102#3 {connectedBlocks = [2 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller4"} :    (!handshake.channel<i5>) -> !handshake.channel<i32>
    %outputs_2:2, %memEnd_3 = mem_controller[%arg2 : memref<30xi32>] %arg7 (%addressResult, %85, %addressResult_30, %dataResult_31) %102#2 {connectedBlocks = [1 : i32, 3 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "5": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller5"} :    (!handshake.channel<i5>, !handshake.channel<i32>, !handshake.channel<i5>, !handshake.channel<i32>) -> (!handshake.channel<i32>, !handshake.control<>)
    %outputs_4:2, %memEnd_5 = mem_controller[%arg1 : memref<30xi32>] %arg6 (%42, %addressResult_12, %addressResult_16, %dataResult_17) %102#1 {connectedBlocks = [2 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "5": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller6"} :    (!handshake.channel<i32>, !handshake.channel<i5>, !handshake.channel<i5>, !handshake.channel<i32>) -> (!handshake.channel<i32>, !handshake.control<>)
    %outputs_6, %memEnd_7 = mem_controller[%arg0 : memref<900xi32>] %arg5 (%addressResult_10) %102#0 {connectedBlocks = [2 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller7"} :    (!handshake.channel<i10>) -> !handshake.channel<i32>
    %1 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant8", value = false} : <>, <i1>
    %2 = extsi %1 {handshake.bb = 0 : ui32, handshake.name = "extsi9"} : <i1> to <i6>
    %3 = mux %4 [%0#2, %trueResult_28] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux4"} : <i1>, [<>, <>] to <>
    %4 = init %100#2 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "init0"} : <i1>
    %5:2 = unbundle %13#0  {handshake.bb = 1 : ui32, handshake.name = "unbundle0"} : <i32> to _ 
    %6 = mux %index [%2, %trueResult_33] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i6>, <i6>] to <i6>
    %7 = buffer %6, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer3"} : <i6>
    %8:2 = fork [2] %7 {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <i6>
    %9 = trunci %8#0 {handshake.bb = 1 : ui32, handshake.name = "trunci0"} : <i6> to <i5>
    %result, %index = control_merge [%0#3, %trueResult_35]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %10:2 = fork [2] %result {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <>
    %11 = constant %10#0 {handshake.bb = 1 : ui32, handshake.name = "constant9", value = false} : <>, <i1>
    %12 = buffer %5#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer0"} : <>
    %addressResult, %dataResult = load[%9] %outputs_2#0 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["store1", 2, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i5>, <i32>, <i5>, <i32>
    %13:2 = fork [2] %dataResult {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <i32>
    %14 = extsi %11 {handshake.bb = 1 : ui32, handshake.name = "extsi8"} : <i1> to <i6>
    %trueResult, %falseResult = cond_br %15, %62 {handshake.bb = 2 : ui32, handshake.name = "cond_br11"} : <i1>, <>
    %15 = buffer %72#3, bufferType = FIFO_BREAK_NONE, numSlots = 4 {handshake.bb = 2 : ui32, handshake.name = "buffer7"} : <i1>
    %16 = buffer %3, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer2"} : <>
    %17 = mux %18 [%16, %trueResult] {ftd.phi, handshake.bb = 2 : ui32, handshake.name = "mux6"} : <i1>, [<>, <>] to <>
    %18 = init %19 {ftd.imerge, handshake.bb = 2 : ui32, handshake.name = "init2"} : <i1>
    %19 = buffer %72#2, bufferType = FIFO_BREAK_NONE, numSlots = 4 {handshake.bb = 2 : ui32, handshake.name = "buffer8"} : <i1>
    %20 = mux %39#1 [%14, %trueResult_20] {handshake.bb = 2 : ui32, handshake.name = "mux1"} : <i1>, [<i6>, <i6>] to <i6>
    %21 = buffer %20, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer5"} : <i6>
    %22:5 = fork [5] %21 {handshake.bb = 2 : ui32, handshake.name = "fork4"} : <i6>
    %23 = extsi %24 {handshake.bb = 2 : ui32, handshake.name = "extsi10"} : <i6> to <i10>
    %24 = buffer %22#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer10"} : <i6>
    %25 = extsi %22#3 {handshake.bb = 2 : ui32, handshake.name = "extsi11"} : <i6> to <i7>
    %26 = extsi %27 {handshake.bb = 2 : ui32, handshake.name = "extsi12"} : <i6> to <i32>
    %27 = buffer %22#4, bufferType = FIFO_BREAK_NONE, numSlots = 4 {handshake.bb = 2 : ui32, handshake.name = "buffer12"} : <i6>
    %28 = trunci %22#1 {handshake.bb = 2 : ui32, handshake.name = "trunci1"} : <i6> to <i5>
    %29 = trunci %30 {handshake.bb = 2 : ui32, handshake.name = "trunci2"} : <i6> to <i5>
    %30 = buffer %22#2, bufferType = FIFO_BREAK_NONE, numSlots = 3 {handshake.bb = 2 : ui32, handshake.name = "buffer14"} : <i6>
    %31 = mux %32 [%13#1, %trueResult_22] {handshake.bb = 2 : ui32, handshake.name = "mux2"} : <i1>, [<i32>, <i32>] to <i32>
    %32 = buffer %39#2, bufferType = FIFO_BREAK_NONE, numSlots = 4 {handshake.bb = 2 : ui32, handshake.name = "buffer15"} : <i1>
    %33 = mux %39#0 [%8#1, %trueResult_24] {handshake.bb = 2 : ui32, handshake.name = "mux3"} : <i1>, [<i6>, <i6>] to <i6>
    %34 = buffer %33, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer9"} : <i6>
    %35:3 = fork [3] %34 {handshake.bb = 2 : ui32, handshake.name = "fork5"} : <i6>
    %36 = extsi %35#2 {handshake.bb = 2 : ui32, handshake.name = "extsi13"} : <i6> to <i12>
    %37 = trunci %38 {handshake.bb = 2 : ui32, handshake.name = "trunci3"} : <i6> to <i5>
    %38 = buffer %35#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer18"} : <i6>
    %result_8, %index_9 = control_merge [%10#1, %trueResult_26]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>, <>] to <>, <i1>
    %39:3 = fork [3] %index_9 {handshake.bb = 2 : ui32, handshake.name = "fork6"} : <i1>
    %40:2 = fork [2] %result_8 {handshake.bb = 2 : ui32, handshake.name = "fork7"} : <>
    %41 = constant %40#0 {handshake.bb = 2 : ui32, handshake.name = "constant10", value = 1 : i2} : <>, <i2>
    %42 = extsi %41 {handshake.bb = 2 : ui32, handshake.name = "extsi2"} : <i2> to <i32>
    %43 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %44 = constant %43 {handshake.bb = 2 : ui32, handshake.name = "constant11", value = 30 : i6} : <>, <i6>
    %45:2 = fork [2] %44 {handshake.bb = 2 : ui32, handshake.name = "fork8"} : <i6>
    %46 = extsi %45#0 {handshake.bb = 2 : ui32, handshake.name = "extsi14"} : <i6> to <i12>
    %47 = extsi %48 {handshake.bb = 2 : ui32, handshake.name = "extsi15"} : <i6> to <i7>
    %48 = buffer %45#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer20"} : <i6>
    %49 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %50 = constant %49 {handshake.bb = 2 : ui32, handshake.name = "constant12", value = 1 : i2} : <>, <i2>
    %51 = extsi %50 {handshake.bb = 2 : ui32, handshake.name = "extsi16"} : <i2> to <i7>
    %52 = muli %36, %46 {handshake.bb = 2 : ui32, handshake.name = "muli2"} : <i12>
    %53 = trunci %52 {handshake.bb = 2 : ui32, handshake.name = "trunci4"} : <i12> to <i10>
    %54 = addi %23, %53 {handshake.bb = 2 : ui32, handshake.name = "addi2"} : <i10>
    %addressResult_10, %dataResult_11 = load[%54] %outputs_6 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i10>, <i32>, <i10>, <i32>
    %55:2 = fork [2] %dataResult_11 {handshake.bb = 2 : ui32, handshake.name = "fork9"} : <i32>
    %56 = buffer %17, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer4"} : <>
    %57 = gate %26, %56 {handshake.bb = 2 : ui32, handshake.name = "gate0"} : <i32>, !handshake.control<> to <i32>
    %58 = trunci %57 {handshake.bb = 2 : ui32, handshake.name = "trunci5"} : <i32> to <i5>
    %addressResult_12, %dataResult_13 = load[%58] %outputs_4#0 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["store0", 1, false], ["store0", 3, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load2"} : <i5>, <i32>, <i5>, <i32>
    %addressResult_14, %dataResult_15 = load[%37] %outputs {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load3"} : <i5>, <i32>, <i5>, <i32>
    %59 = muli %dataResult_15, %55#0 {handshake.bb = 2 : ui32, handshake.name = "muli0"} : <i32>
    %60 = addi %dataResult_13, %59 {handshake.bb = 2 : ui32, handshake.name = "addi0"} : <i32>
    %61 = buffer %doneResult, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer16"} : <>
    %62 = buffer %61, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer1"} : <>
    %addressResult_16, %dataResult_17, %doneResult = store[%29] %60 %outputs_4#1 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["load2", 1, false], ["store0", 1, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store0"} : <i5>, <i32>, <>, <i5>, <i32>, <>
    %addressResult_18, %dataResult_19 = load[%28] %outputs_0 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load4"} : <i5>, <i32>, <i5>, <i32>
    %63 = muli %55#1, %dataResult_19 {handshake.bb = 2 : ui32, handshake.name = "muli1"} : <i32>
    %64 = buffer %31, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer6"} : <i32>
    %65 = addi %64, %63 {handshake.bb = 2 : ui32, handshake.name = "addi1"} : <i32>
    %66 = addi %25, %51 {handshake.bb = 2 : ui32, handshake.name = "addi3"} : <i7>
    %67 = buffer %66, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer19"} : <i7>
    %68:2 = fork [2] %67 {handshake.bb = 2 : ui32, handshake.name = "fork10"} : <i7>
    %69 = trunci %68#0 {handshake.bb = 2 : ui32, handshake.name = "trunci6"} : <i7> to <i6>
    %70 = cmpi ult, %68#1, %47 {handshake.bb = 2 : ui32, handshake.name = "cmpi0"} : <i7>
    %71 = buffer %70, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer21"} : <i1>
    %72:6 = fork [6] %71 {handshake.bb = 2 : ui32, handshake.name = "fork11"} : <i1>
    %trueResult_20, %falseResult_21 = cond_br %72#0, %69 {handshake.bb = 2 : ui32, handshake.name = "cond_br2"} : <i1>, <i6>
    sink %falseResult_21 {handshake.name = "sink0"} : <i6>
    %73 = buffer %65, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer17"} : <i32>
    %trueResult_22, %falseResult_23 = cond_br %74, %73 {handshake.bb = 2 : ui32, handshake.name = "cond_br3"} : <i1>, <i32>
    %74 = buffer %72#4, bufferType = FIFO_BREAK_NONE, numSlots = 3 {handshake.bb = 2 : ui32, handshake.name = "buffer26"} : <i1>
    %75 = buffer %35#1, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer11"} : <i6>
    %trueResult_24, %falseResult_25 = cond_br %72#1, %75 {handshake.bb = 2 : ui32, handshake.name = "cond_br4"} : <i1>, <i6>
    %76 = buffer %40#1, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer13"} : <>
    %trueResult_26, %falseResult_27 = cond_br %72#5, %76 {handshake.bb = 2 : ui32, handshake.name = "cond_br5"} : <i1>, <>
    %trueResult_28, %falseResult_29 = cond_br %77, %falseResult {handshake.bb = 3 : ui32, handshake.name = "cond_br12"} : <i1>, <>
    %77 = buffer %100#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer30"} : <i1>
    sink %falseResult_29 {handshake.name = "sink1"} : <>
    %78:2 = fork [2] %falseResult_25 {handshake.bb = 3 : ui32, handshake.name = "fork12"} : <i6>
    %79 = extsi %78#0 {handshake.bb = 3 : ui32, handshake.name = "extsi17"} : <i6> to <i7>
    %80 = extsi %81 {handshake.bb = 3 : ui32, handshake.name = "extsi18"} : <i6> to <i32>
    %81 = buffer %78#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer32"} : <i6>
    %82:2 = fork [2] %falseResult_23 {handshake.bb = 3 : ui32, handshake.name = "fork13"} : <i32>
    %83:2 = fork [2] %falseResult_27 {handshake.bb = 3 : ui32, handshake.name = "fork14"} : <>
    %84 = constant %83#0 {handshake.bb = 3 : ui32, handshake.name = "constant13", value = 1 : i2} : <>, <i2>
    %85 = extsi %84 {handshake.bb = 3 : ui32, handshake.name = "extsi5"} : <i2> to <i32>
    %86 = source {handshake.bb = 3 : ui32, handshake.name = "source2"} : <>
    %87 = constant %86 {handshake.bb = 3 : ui32, handshake.name = "constant14", value = 30 : i6} : <>, <i6>
    %88 = extsi %87 {handshake.bb = 3 : ui32, handshake.name = "extsi19"} : <i6> to <i7>
    %89 = source {handshake.bb = 3 : ui32, handshake.name = "source3"} : <>
    %90 = constant %89 {handshake.bb = 3 : ui32, handshake.name = "constant15", value = 1 : i2} : <>, <i2>
    %91 = extsi %90 {handshake.bb = 3 : ui32, handshake.name = "extsi20"} : <i2> to <i7>
    %92 = gate %80, %12 {handshake.bb = 3 : ui32, handshake.name = "gate1"} : <i32>, !handshake.control<> to <i32>
    %93 = trunci %92 {handshake.bb = 3 : ui32, handshake.name = "trunci7"} : <i32> to <i5>
    %addressResult_30, %dataResult_31, %doneResult_32 = store[%93] %82#0 %outputs_2#1 {handshake.bb = 3 : ui32, handshake.bufProps = #handshake<bufProps{"2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store1"} : <i5>, <i32>, <>, <i5>, <i32>, <>
    sink %doneResult_32 {handshake.name = "sink3"} : <>
    %94 = addi %79, %91 {handshake.bb = 3 : ui32, handshake.name = "addi4"} : <i7>
    %95 = buffer %94, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer22"} : <i7>
    %96:2 = fork [2] %95 {handshake.bb = 3 : ui32, handshake.name = "fork15"} : <i7>
    %97 = trunci %96#0 {handshake.bb = 3 : ui32, handshake.name = "trunci8"} : <i7> to <i6>
    %98 = cmpi ult, %96#1, %88 {handshake.bb = 3 : ui32, handshake.name = "cmpi1"} : <i7>
    %99 = buffer %98, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer23"} : <i1>
    %100:5 = fork [5] %99 {handshake.bb = 3 : ui32, handshake.name = "fork16"} : <i1>
    %trueResult_33, %falseResult_34 = cond_br %100#0, %97 {handshake.bb = 3 : ui32, handshake.name = "cond_br6"} : <i1>, <i6>
    sink %falseResult_34 {handshake.name = "sink4"} : <i6>
    %trueResult_35, %falseResult_36 = cond_br %100#3, %83#1 {handshake.bb = 3 : ui32, handshake.name = "cond_br7"} : <i1>, <>
    %trueResult_37, %falseResult_38 = cond_br %101, %82#1 {handshake.bb = 3 : ui32, handshake.name = "cond_br8"} : <i1>, <i32>
    %101 = buffer %100#4, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer38"} : <i1>
    sink %trueResult_37 {handshake.name = "sink5"} : <i32>
    %102:5 = fork [5] %falseResult_36 {handshake.bb = 4 : ui32, handshake.name = "fork17"} : <>
    end {handshake.bb = 4 : ui32, handshake.name = "end0"} %falseResult_38, %memEnd_7, %memEnd_5, %memEnd_3, %memEnd_1, %memEnd, %0#1 : <i32>, <>, <>, <>, <>, <>, <>
  }
}

