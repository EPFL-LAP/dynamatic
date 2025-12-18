module {
  handshake.func @bicg(%arg0: memref<900xi32>, %arg1: memref<30xi32>, %arg2: memref<30xi32>, %arg3: memref<30xi32>, %arg4: memref<30xi32>, %arg5: !handshake.control<>, %arg6: !handshake.control<>, %arg7: !handshake.control<>, %arg8: !handshake.control<>, %arg9: !handshake.control<>, %arg10: !handshake.control<>, ...) -> (!handshake.channel<i32>, !handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["a", "s", "q", "p", "r", "a_start", "s_start", "q_start", "p_start", "r_start", "start"], resNames = ["out0", "a_end", "s_end", "q_end", "p_end", "r_end", "end"]} {
    %0:3 = fork [3] %arg10 {handshake.bb = 0 : ui32, handshake.name = "fork0"} : <>
    %outputs, %memEnd = mem_controller[%arg4 : memref<30xi32>] %arg9 (%addressResult_10) %103#4 {connectedBlocks = [2 : i32], handshake.name = "mem_controller3"} :    (!handshake.channel<i5>) -> !handshake.channel<i32>
    %outputs_0, %memEnd_1 = mem_controller[%arg3 : memref<30xi32>] %arg8 (%addressResult_14) %103#3 {connectedBlocks = [2 : i32], handshake.name = "mem_controller4"} :    (!handshake.channel<i5>) -> !handshake.channel<i32>
    %1:2 = lsq[%arg2 : memref<30xi32>] (%arg7, %11#0, %addressResult, %83#0, %addressResult_24, %dataResult_25, %103#2)  {groupSizes = [1 : i32, 1 : i32], handshake.name = "lsq2"} : (!handshake.control<>, !handshake.control<>, !handshake.channel<i5>, !handshake.control<>, !handshake.channel<i5>, !handshake.channel<i32>, !handshake.control<>) -> (!handshake.channel<i32>, !handshake.control<>)
    %2:2 = lsq[%arg1 : memref<30xi32>] (%arg6, %42#0, %addressResult_8, %addressResult_12, %dataResult_13, %103#1)  {groupSizes = [2 : i32], handshake.name = "lsq3"} : (!handshake.control<>, !handshake.control<>, !handshake.channel<i5>, !handshake.channel<i5>, !handshake.channel<i32>, !handshake.control<>) -> (!handshake.channel<i32>, !handshake.control<>)
    %outputs_2, %memEnd_3 = mem_controller[%arg0 : memref<900xi32>] %arg5 (%addressResult_6) %103#0 {connectedBlocks = [2 : i32], handshake.name = "mem_controller5"} :    (!handshake.channel<i10>) -> !handshake.channel<i32>
    %3 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant0", value = false} : <>, <i1>
    %4 = br %3 {handshake.bb = 0 : ui32, handshake.name = "br2"} : <i1>
    %5 = extsi %4 {handshake.bb = 0 : ui32, handshake.name = "extsi7"} : <i1> to <i6>
    %6 = br %0#2 {handshake.bb = 0 : ui32, handshake.name = "br3"} : <>
    %7 = mux %index [%5, %trueResult_26] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i6>, <i6>] to <i6>
    %8:2 = fork [2] %7 {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <i6>
    %9 = trunci %8#0 {handshake.bb = 1 : ui32, handshake.name = "trunci0"} : <i6> to <i5>
    %result, %index = control_merge [%6, %trueResult_28]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %11:3 = lazy_fork [3] %result {handshake.bb = 1 : ui32, handshake.name = "lazy_fork0"} : <>
    %12 = constant %11#2 {handshake.bb = 1 : ui32, handshake.name = "constant1", value = false} : <>, <i1>
    %addressResult, %dataResult = load[%9] %1#0 {handshake.bb = 1 : ui32, handshake.deps = #handshake<deps[["store1", 2]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 0>, handshake.name = "load0"} : <i5>, <i32>, <i5>, <i32>
    %13 = br %12 {handshake.bb = 1 : ui32, handshake.name = "br4"} : <i1>
    %14 = extsi %13 {handshake.bb = 1 : ui32, handshake.name = "extsi6"} : <i1> to <i6>
    %15 = br %dataResult {handshake.bb = 1 : ui32, handshake.name = "br5"} : <i32>
    %16 = br %17 {handshake.bb = 1 : ui32, handshake.name = "br6"} : <i6>
    %17 = buffer %8#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer1"} : <i6>
    %18 = br %11#1 {handshake.bb = 1 : ui32, handshake.name = "br7"} : <>
    %19 = mux %41#1 [%14, %trueResult] {handshake.bb = 2 : ui32, handshake.name = "mux1"} : <i1>, [<i6>, <i6>] to <i6>
    %21:5 = fork [5] %19 {handshake.bb = 2 : ui32, handshake.name = "fork2"} : <i6>
    %22 = extsi %23 {handshake.bb = 2 : ui32, handshake.name = "extsi8"} : <i6> to <i10>
    %23 = buffer %21#0, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 2 : ui32, handshake.name = "buffer3"} : <i6>
    %24 = extsi %21#4 {handshake.bb = 2 : ui32, handshake.name = "extsi9"} : <i6> to <i7>
    %26 = trunci %21#1 {handshake.bb = 2 : ui32, handshake.name = "trunci1"} : <i6> to <i5>
    %28 = trunci %21#2 {handshake.bb = 2 : ui32, handshake.name = "trunci2"} : <i6> to <i5>
    %30 = trunci %21#3 {handshake.bb = 2 : ui32, handshake.name = "trunci3"} : <i6> to <i5>
    %32 = mux %33 [%15, %trueResult_16] {handshake.bb = 2 : ui32, handshake.name = "mux2"} : <i1>, [<i32>, <i32>] to <i32>
    %33 = buffer %41#2, bufferType = FIFO_BREAK_NONE, numSlots = 4 {handshake.bb = 2 : ui32, handshake.name = "buffer8"} : <i1>
    %34 = mux %41#0 [%16, %trueResult_18] {handshake.bb = 2 : ui32, handshake.name = "mux3"} : <i1>, [<i6>, <i6>] to <i6>
    %36:3 = fork [3] %34 {handshake.bb = 2 : ui32, handshake.name = "fork3"} : <i6>
    %37 = extsi %36#2 {handshake.bb = 2 : ui32, handshake.name = "extsi10"} : <i6> to <i12>
    %39 = trunci %40 {handshake.bb = 2 : ui32, handshake.name = "trunci4"} : <i6> to <i5>
    %40 = buffer %36#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer11"} : <i6>
    %result_4, %index_5 = control_merge [%18, %trueResult_20]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>, <>] to <>, <i1>
    %41:3 = fork [3] %index_5 {handshake.bb = 2 : ui32, handshake.name = "fork4"} : <i1>
    %42:2 = fork [2] %result_4 {handshake.bb = 2 : ui32, handshake.name = "fork5"} : <>
    %43 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %44 = constant %43 {handshake.bb = 2 : ui32, handshake.name = "constant8", value = 30 : i6} : <>, <i6>
    %45:2 = fork [2] %44 {handshake.bb = 2 : ui32, handshake.name = "fork6"} : <i6>
    %46 = extsi %45#0 {handshake.bb = 2 : ui32, handshake.name = "extsi11"} : <i6> to <i12>
    %48 = extsi %49 {handshake.bb = 2 : ui32, handshake.name = "extsi12"} : <i6> to <i7>
    %49 = buffer %45#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer13"} : <i6>
    %50 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %51 = constant %50 {handshake.bb = 2 : ui32, handshake.name = "constant9", value = 1 : i2} : <>, <i2>
    %52 = extsi %51 {handshake.bb = 2 : ui32, handshake.name = "extsi13"} : <i2> to <i7>
    %53 = muli %37, %46 {handshake.bb = 2 : ui32, handshake.name = "muli2"} : <i12>
    %54 = trunci %53 {handshake.bb = 2 : ui32, handshake.name = "trunci5"} : <i12> to <i10>
    %55 = addi %22, %54 {handshake.bb = 2 : ui32, handshake.name = "addi2"} : <i10>
    %addressResult_6, %dataResult_7 = load[%55] %outputs_2 {handshake.bb = 2 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i10>, <i32>, <i10>, <i32>
    %56:2 = fork [2] %dataResult_7 {handshake.bb = 2 : ui32, handshake.name = "fork7"} : <i32>
    %addressResult_8, %dataResult_9 = load[%30] %2#0 {handshake.bb = 2 : ui32, handshake.deps = #handshake<deps[["store0", 1], ["store0", 3]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 0>, handshake.name = "load2"} : <i5>, <i32>, <i5>, <i32>
    %addressResult_10, %dataResult_11 = load[%39] %outputs {handshake.bb = 2 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load3"} : <i5>, <i32>, <i5>, <i32>
    %57 = muli %dataResult_11, %56#0 {handshake.bb = 2 : ui32, handshake.name = "muli0"} : <i32>
    %59 = addi %dataResult_9, %57 {handshake.bb = 2 : ui32, handshake.name = "addi0"} : <i32>
    %addressResult_12, %dataResult_13 = store[%28] %59 {handshake.bb = 2 : ui32, handshake.deps = #handshake<deps[["load2", 1], ["store0", 1]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 0>, handshake.name = "store0"} : <i5>, <i32>, <i5>, <i32>
    %addressResult_14, %dataResult_15 = load[%26] %outputs_0 {handshake.bb = 2 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load4"} : <i5>, <i32>, <i5>, <i32>
    %60 = muli %56#1, %dataResult_15 {handshake.bb = 2 : ui32, handshake.name = "muli1"} : <i32>
    %62 = addi %32, %60 {handshake.bb = 2 : ui32, handshake.name = "addi1"} : <i32>
    %63 = addi %24, %52 {handshake.bb = 2 : ui32, handshake.name = "addi3"} : <i7>
    %64:2 = fork [2] %63 {handshake.bb = 2 : ui32, handshake.name = "fork8"} : <i7>
    %65 = trunci %64#0 {handshake.bb = 2 : ui32, handshake.name = "trunci6"} : <i7> to <i6>
    %67 = cmpi ult, %64#1, %48 {handshake.bb = 2 : ui32, handshake.name = "cmpi0"} : <i7>
    %69:4 = fork [4] %67 {handshake.bb = 2 : ui32, handshake.name = "fork9"} : <i1>
    %trueResult, %falseResult = cond_br %69#0, %65 {handshake.bb = 2 : ui32, handshake.name = "cond_br2"} : <i1>, <i6>
    sink %falseResult {handshake.name = "sink0"} : <i6>
    %trueResult_16, %falseResult_17 = cond_br %71, %62 {handshake.bb = 2 : ui32, handshake.name = "cond_br3"} : <i1>, <i32>
    %71 = buffer %69#2, bufferType = FIFO_BREAK_NONE, numSlots = 4 {handshake.bb = 2 : ui32, handshake.name = "buffer19"} : <i1>
    %trueResult_18, %falseResult_19 = cond_br %69#1, %36#1 {handshake.bb = 2 : ui32, handshake.name = "cond_br4"} : <i1>, <i6>
    %trueResult_20, %falseResult_21 = cond_br %74, %42#1 {handshake.bb = 2 : ui32, handshake.name = "cond_br5"} : <i1>, <>
    %74 = buffer %69#3, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer22"} : <i1>
    %75 = merge %falseResult_19 {handshake.bb = 3 : ui32, handshake.name = "merge0"} : <i6>
    %76:2 = fork [2] %75 {handshake.bb = 3 : ui32, handshake.name = "fork10"} : <i6>
    %77 = extsi %76#1 {handshake.bb = 3 : ui32, handshake.name = "extsi14"} : <i6> to <i7>
    %79 = trunci %76#0 {handshake.bb = 3 : ui32, handshake.name = "trunci7"} : <i6> to <i5>
    %81 = merge %falseResult_17 {handshake.bb = 3 : ui32, handshake.name = "merge1"} : <i32>
    %82:2 = fork [2] %81 {handshake.bb = 3 : ui32, handshake.name = "fork11"} : <i32>
    %result_22, %index_23 = control_merge [%falseResult_21]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>] to <>, <i1>
    sink %index_23 {handshake.name = "sink1"} : <i1>
    %83:2 = lazy_fork [2] %result_22 {handshake.bb = 3 : ui32, handshake.name = "lazy_fork1"} : <>
    %84 = source {handshake.bb = 3 : ui32, handshake.name = "source2"} : <>
    %85 = constant %84 {handshake.bb = 3 : ui32, handshake.name = "constant10", value = 30 : i6} : <>, <i6>
    %86 = extsi %85 {handshake.bb = 3 : ui32, handshake.name = "extsi15"} : <i6> to <i7>
    %87 = source {handshake.bb = 3 : ui32, handshake.name = "source3"} : <>
    %88 = constant %87 {handshake.bb = 3 : ui32, handshake.name = "constant11", value = 1 : i2} : <>, <i2>
    %89 = extsi %88 {handshake.bb = 3 : ui32, handshake.name = "extsi16"} : <i2> to <i7>
    %addressResult_24, %dataResult_25 = store[%79] %82#0 {handshake.bb = 3 : ui32, handshake.mem_interface = #handshake.mem_interface<LSQ: 1>, handshake.name = "store1"} : <i5>, <i32>, <i5>, <i32>
    %91 = addi %77, %89 {handshake.bb = 3 : ui32, handshake.name = "addi4"} : <i7>
    %92:2 = fork [2] %91 {handshake.bb = 3 : ui32, handshake.name = "fork12"} : <i7>
    %93 = trunci %94 {handshake.bb = 3 : ui32, handshake.name = "trunci8"} : <i7> to <i6>
    %94 = buffer %92#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer26"} : <i7>
    %95 = cmpi ult, %92#1, %86 {handshake.bb = 3 : ui32, handshake.name = "cmpi1"} : <i7>
    %97:3 = fork [3] %95 {handshake.bb = 3 : ui32, handshake.name = "fork13"} : <i1>
    %trueResult_26, %falseResult_27 = cond_br %98, %93 {handshake.bb = 3 : ui32, handshake.name = "cond_br6"} : <i1>, <i6>
    %98 = buffer %97#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer28"} : <i1>
    sink %falseResult_27 {handshake.name = "sink2"} : <i6>
    %trueResult_28, %falseResult_29 = cond_br %99, %83#1 {handshake.bb = 3 : ui32, handshake.name = "cond_br7"} : <i1>, <>
    %99 = buffer %97#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer29"} : <i1>
    %trueResult_30, %falseResult_31 = cond_br %100, %101 {handshake.bb = 3 : ui32, handshake.name = "cond_br8"} : <i1>, <i32>
    %100 = buffer %97#2, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer30"} : <i1>
    %101 = buffer %82#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer31"} : <i32>
    sink %trueResult_30 {handshake.name = "sink3"} : <i32>
    %102 = merge %falseResult_31 {handshake.bb = 4 : ui32, handshake.name = "merge2"} : <i32>
    %result_32, %index_33 = control_merge [%falseResult_29]  {handshake.bb = 4 : ui32, handshake.name = "control_merge3"} : [<>] to <>, <i1>
    sink %index_33 {handshake.name = "sink4"} : <i1>
    %103:5 = fork [5] %result_32 {handshake.bb = 4 : ui32, handshake.name = "fork14"} : <>
    end {handshake.bb = 4 : ui32, handshake.name = "end0"} %102, %memEnd_3, %2#1, %1#1, %memEnd_1, %memEnd, %0#1 : <i32>, <>, <>, <>, <>, <>, <>
  }
}

