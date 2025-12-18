module {
  handshake.func @bicg(%arg0: memref<900xi32>, %arg1: memref<30xi32>, %arg2: memref<30xi32>, %arg3: memref<30xi32>, %arg4: memref<30xi32>, %arg5: !handshake.control<>, %arg6: !handshake.control<>, %arg7: !handshake.control<>, %arg8: !handshake.control<>, %arg9: !handshake.control<>, %arg10: !handshake.control<>, ...) -> (!handshake.channel<i32>, !handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["a", "s", "q", "p", "r", "a_start", "s_start", "q_start", "p_start", "r_start", "start"], resNames = ["out0", "a_end", "s_end", "q_end", "p_end", "r_end", "end"]} {
    %0:3 = fork [3] %arg10 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork0"} : <>
    %outputs, %memEnd = mem_controller[%arg4 : memref<30xi32>] %arg9 (%addressResult_10) %104#4 {connectedBlocks = [2 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller3"} :    (!handshake.channel<i5>) -> !handshake.channel<i32>
    %outputs_0, %memEnd_1 = mem_controller[%arg3 : memref<30xi32>] %arg8 (%addressResult_14) %104#3 {connectedBlocks = [2 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller4"} :    (!handshake.channel<i5>) -> !handshake.channel<i32>
    %1:2 = lsq[%arg2 : memref<30xi32>] (%arg7, %11#0, %addressResult, %81#0, %addressResult_24, %dataResult_25, %104#2)  {groupSizes = [1 : i32, 1 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "5": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "6": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "lsq2"} : (!handshake.control<>, !handshake.control<>, !handshake.channel<i5>, !handshake.control<>, !handshake.channel<i5>, !handshake.channel<i32>, !handshake.control<>) -> (!handshake.channel<i32>, !handshake.control<>)
    %2:2 = lsq[%arg1 : memref<30xi32>] (%arg6, %41#0, %addressResult_8, %addressResult_12, %dataResult_13, %104#1)  {groupSizes = [2 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "5": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "lsq3"} : (!handshake.control<>, !handshake.control<>, !handshake.channel<i5>, !handshake.channel<i5>, !handshake.channel<i32>, !handshake.control<>) -> (!handshake.channel<i32>, !handshake.control<>)
    %outputs_2, %memEnd_3 = mem_controller[%arg0 : memref<900xi32>] %arg5 (%addressResult_6) %104#0 {connectedBlocks = [2 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller5"} :    (!handshake.channel<i10>) -> !handshake.channel<i32>
    %3 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant0", value = false} : <>, <i1>
    %4 = br %3 {handshake.bb = 0 : ui32, handshake.name = "br2"} : <i1>
    %5 = extsi %4 {handshake.bb = 0 : ui32, handshake.name = "extsi7"} : <i1> to <i6>
    %6 = br %0#2 {handshake.bb = 0 : ui32, handshake.name = "br3"} : <>
    %7 = mux %index [%5, %trueResult_26] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i6>, <i6>] to <i6>
    %8 = buffer %7, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer0"} : <i6>
    %9:2 = fork [2] %8 {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <i6>
    %10 = trunci %9#0 {handshake.bb = 1 : ui32, handshake.name = "trunci0"} : <i6> to <i5>
    %result, %index = control_merge [%6, %trueResult_28]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %11:3 = lazy_fork [3] %result {handshake.bb = 1 : ui32, handshake.name = "lazy_fork0"} : <>
    %12 = buffer %11#2, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer5"} : <>
    %13 = constant %12 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"0": [1,inf], [0,inf], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "constant1", value = false} : <>, <i1>
    %14 = buffer %10, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer2"} : <i5>
    %addressResult, %dataResult = load[%14] %1#0 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["store1", 2]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 0>, handshake.name = "load0"} : <i5>, <i32>, <i5>, <i32>
    %15 = br %13 {handshake.bb = 1 : ui32, handshake.name = "br4"} : <i1>
    %16 = extsi %15 {handshake.bb = 1 : ui32, handshake.name = "extsi6"} : <i1> to <i6>
    %17 = br %dataResult {handshake.bb = 1 : ui32, handshake.name = "br5"} : <i32>
    %18 = br %19 {handshake.bb = 1 : ui32, handshake.name = "br6"} : <i6>
    %19 = buffer %9#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer1"} : <i6>
    %20 = buffer %11#1, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer4"} : <>
    %21 = br %20 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [1,inf], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "br7"} : <>
    %22 = mux %40#1 [%16, %trueResult] {handshake.bb = 2 : ui32, handshake.name = "mux1"} : <i1>, [<i6>, <i6>] to <i6>
    %23 = buffer %22, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer6"} : <i6>
    %24:5 = fork [5] %23 {handshake.bb = 2 : ui32, handshake.name = "fork2"} : <i6>
    %25 = extsi %26 {handshake.bb = 2 : ui32, handshake.name = "extsi8"} : <i6> to <i10>
    %26 = buffer %24#0, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 2 : ui32, handshake.name = "buffer3"} : <i6>
    %27 = extsi %24#4 {handshake.bb = 2 : ui32, handshake.name = "extsi9"} : <i6> to <i7>
    %28 = trunci %24#1 {handshake.bb = 2 : ui32, handshake.name = "trunci1"} : <i6> to <i5>
    %29 = trunci %24#2 {handshake.bb = 2 : ui32, handshake.name = "trunci2"} : <i6> to <i5>
    %30 = trunci %24#3 {handshake.bb = 2 : ui32, handshake.name = "trunci3"} : <i6> to <i5>
    %31 = mux %32 [%17, %trueResult_16] {handshake.bb = 2 : ui32, handshake.name = "mux2"} : <i1>, [<i32>, <i32>] to <i32>
    %32 = buffer %40#2, bufferType = FIFO_BREAK_NONE, numSlots = 4 {handshake.bb = 2 : ui32, handshake.name = "buffer8"} : <i1>
    %33 = mux %40#0 [%18, %trueResult_18] {handshake.bb = 2 : ui32, handshake.name = "mux3"} : <i1>, [<i6>, <i6>] to <i6>
    %34 = buffer %33, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer12"} : <i6>
    %35 = buffer %34, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer14"} : <i6>
    %36:3 = fork [3] %35 {handshake.bb = 2 : ui32, handshake.name = "fork3"} : <i6>
    %37 = extsi %36#2 {handshake.bb = 2 : ui32, handshake.name = "extsi10"} : <i6> to <i12>
    %38 = trunci %39 {handshake.bb = 2 : ui32, handshake.name = "trunci4"} : <i6> to <i5>
    %39 = buffer %36#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer11"} : <i6>
    %result_4, %index_5 = control_merge [%21, %trueResult_20]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>, <>] to <>, <i1>
    %40:3 = fork [3] %index_5 {handshake.bb = 2 : ui32, handshake.name = "fork4"} : <i1>
    %41:2 = fork [2] %result_4 {handshake.bb = 2 : ui32, handshake.name = "fork5"} : <>
    %42 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %43 = constant %42 {handshake.bb = 2 : ui32, handshake.name = "constant8", value = 30 : i6} : <>, <i6>
    %44:2 = fork [2] %43 {handshake.bb = 2 : ui32, handshake.name = "fork6"} : <i6>
    %45 = extsi %44#0 {handshake.bb = 2 : ui32, handshake.name = "extsi11"} : <i6> to <i12>
    %46 = extsi %47 {handshake.bb = 2 : ui32, handshake.name = "extsi12"} : <i6> to <i7>
    %47 = buffer %44#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer13"} : <i6>
    %48 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %49 = constant %48 {handshake.bb = 2 : ui32, handshake.name = "constant9", value = 1 : i2} : <>, <i2>
    %50 = extsi %49 {handshake.bb = 2 : ui32, handshake.name = "extsi13"} : <i2> to <i7>
    %51 = muli %37, %45 {handshake.bb = 2 : ui32, handshake.name = "muli2"} : <i12>
    %52 = trunci %51 {handshake.bb = 2 : ui32, handshake.name = "trunci5"} : <i12> to <i10>
    %53 = addi %25, %52 {handshake.bb = 2 : ui32, handshake.name = "addi2"} : <i10>
    %addressResult_6, %dataResult_7 = load[%53] %outputs_2 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i10>, <i32>, <i10>, <i32>
    %54:2 = fork [2] %dataResult_7 {handshake.bb = 2 : ui32, handshake.name = "fork7"} : <i32>
    %55 = buffer %30, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer9"} : <i5>
    %addressResult_8, %dataResult_9 = load[%55] %2#0 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["store0", 1], ["store0", 3]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 0>, handshake.name = "load2"} : <i5>, <i32>, <i5>, <i32>
    %addressResult_10, %dataResult_11 = load[%38] %outputs {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load3"} : <i5>, <i32>, <i5>, <i32>
    %56 = muli %dataResult_11, %54#0 {handshake.bb = 2 : ui32, handshake.name = "muli0"} : <i32>
    %57 = addi %dataResult_9, %56 {handshake.bb = 2 : ui32, handshake.name = "addi0"} : <i32>
    %58 = buffer %29, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer7"} : <i5>
    %59 = buffer %57, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer17"} : <i32>
    %addressResult_12, %dataResult_13 = store[%58] %59 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["load2", 1], ["store0", 1]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 0>, handshake.name = "store0"} : <i5>, <i32>, <i5>, <i32>
    %addressResult_14, %dataResult_15 = load[%28] %outputs_0 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load4"} : <i5>, <i32>, <i5>, <i32>
    %60 = muli %54#1, %dataResult_15 {handshake.bb = 2 : ui32, handshake.name = "muli1"} : <i32>
    %61 = buffer %31, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer10"} : <i32>
    %62 = addi %61, %60 {handshake.bb = 2 : ui32, handshake.name = "addi1"} : <i32>
    %63 = addi %27, %50 {handshake.bb = 2 : ui32, handshake.name = "addi3"} : <i7>
    %64 = buffer %63, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer20"} : <i7>
    %65:2 = fork [2] %64 {handshake.bb = 2 : ui32, handshake.name = "fork8"} : <i7>
    %66 = trunci %65#0 {handshake.bb = 2 : ui32, handshake.name = "trunci6"} : <i7> to <i6>
    %67 = cmpi ult, %65#1, %46 {handshake.bb = 2 : ui32, handshake.name = "cmpi0"} : <i7>
    %68 = buffer %67, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer21"} : <i1>
    %69:4 = fork [4] %68 {handshake.bb = 2 : ui32, handshake.name = "fork9"} : <i1>
    %trueResult, %falseResult = cond_br %69#0, %66 {handshake.bb = 2 : ui32, handshake.name = "cond_br2"} : <i1>, <i6>
    sink %falseResult {handshake.name = "sink0"} : <i6>
    %70 = buffer %62, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer18"} : <i32>
    %trueResult_16, %falseResult_17 = cond_br %71, %70 {handshake.bb = 2 : ui32, handshake.name = "cond_br3"} : <i1>, <i32>
    %71 = buffer %69#2, bufferType = FIFO_BREAK_NONE, numSlots = 4 {handshake.bb = 2 : ui32, handshake.name = "buffer19"} : <i1>
    %trueResult_18, %falseResult_19 = cond_br %69#1, %36#1 {handshake.bb = 2 : ui32, handshake.name = "cond_br4"} : <i1>, <i6>
    %72 = buffer %41#1, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer15"} : <>
    %73 = buffer %72, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer16"} : <>
    %trueResult_20, %falseResult_21 = cond_br %74, %73 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [1,inf], [0,inf], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "cond_br5"} : <i1>, <>
    %74 = buffer %69#3, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer22"} : <i1>
    %75 = merge %falseResult_19 {handshake.bb = 3 : ui32, handshake.name = "merge0"} : <i6>
    %76:2 = fork [2] %75 {handshake.bb = 3 : ui32, handshake.name = "fork10"} : <i6>
    %77 = extsi %76#1 {handshake.bb = 3 : ui32, handshake.name = "extsi14"} : <i6> to <i7>
    %78 = trunci %76#0 {handshake.bb = 3 : ui32, handshake.name = "trunci7"} : <i6> to <i5>
    %79 = merge %falseResult_17 {handshake.bb = 3 : ui32, handshake.name = "merge1"} : <i32>
    %80:2 = fork [2] %79 {handshake.bb = 3 : ui32, handshake.name = "fork11"} : <i32>
    %result_22, %index_23 = control_merge [%falseResult_21]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>] to <>, <i1>
    sink %index_23 {handshake.name = "sink1"} : <i1>
    %81:2 = lazy_fork [2] %result_22 {handshake.bb = 3 : ui32, handshake.name = "lazy_fork1"} : <>
    %82 = source {handshake.bb = 3 : ui32, handshake.name = "source2"} : <>
    %83 = constant %82 {handshake.bb = 3 : ui32, handshake.name = "constant10", value = 30 : i6} : <>, <i6>
    %84 = extsi %83 {handshake.bb = 3 : ui32, handshake.name = "extsi15"} : <i6> to <i7>
    %85 = source {handshake.bb = 3 : ui32, handshake.name = "source3"} : <>
    %86 = constant %85 {handshake.bb = 3 : ui32, handshake.name = "constant11", value = 1 : i2} : <>, <i2>
    %87 = extsi %86 {handshake.bb = 3 : ui32, handshake.name = "extsi16"} : <i2> to <i7>
    %88 = buffer %78, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer23"} : <i5>
    %89 = buffer %80#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer24"} : <i32>
    %addressResult_24, %dataResult_25 = store[%88] %89 {handshake.bb = 3 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<LSQ: 1>, handshake.name = "store1"} : <i5>, <i32>, <i5>, <i32>
    %90 = addi %77, %87 {handshake.bb = 3 : ui32, handshake.name = "addi4"} : <i7>
    %91 = buffer %90, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer27"} : <i7>
    %92:2 = fork [2] %91 {handshake.bb = 3 : ui32, handshake.name = "fork12"} : <i7>
    %93 = trunci %94 {handshake.bb = 3 : ui32, handshake.name = "trunci8"} : <i7> to <i6>
    %94 = buffer %92#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer26"} : <i7>
    %95 = cmpi ult, %92#1, %84 {handshake.bb = 3 : ui32, handshake.name = "cmpi1"} : <i7>
    %96 = buffer %95, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer32"} : <i1>
    %97:3 = fork [3] %96 {handshake.bb = 3 : ui32, handshake.name = "fork13"} : <i1>
    %trueResult_26, %falseResult_27 = cond_br %98, %93 {handshake.bb = 3 : ui32, handshake.name = "cond_br6"} : <i1>, <i6>
    %98 = buffer %97#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer28"} : <i1>
    sink %falseResult_27 {handshake.name = "sink2"} : <i6>
    %99 = buffer %81#1, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer25"} : <>
    %trueResult_28, %falseResult_29 = cond_br %100, %99 {handshake.bb = 3 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,inf], [1,inf], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "cond_br7"} : <i1>, <>
    %100 = buffer %97#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer29"} : <i1>
    %trueResult_30, %falseResult_31 = cond_br %101, %102 {handshake.bb = 3 : ui32, handshake.name = "cond_br8"} : <i1>, <i32>
    %101 = buffer %97#2, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer30"} : <i1>
    %102 = buffer %80#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer31"} : <i32>
    sink %trueResult_30 {handshake.name = "sink3"} : <i32>
    %103 = merge %falseResult_31 {handshake.bb = 4 : ui32, handshake.name = "merge2"} : <i32>
    %result_32, %index_33 = control_merge [%falseResult_29]  {handshake.bb = 4 : ui32, handshake.name = "control_merge3"} : [<>] to <>, <i1>
    sink %index_33 {handshake.name = "sink4"} : <i1>
    %104:5 = fork [5] %result_32 {handshake.bb = 4 : ui32, handshake.name = "fork14"} : <>
    end {handshake.bb = 4 : ui32, handshake.name = "end0"} %103, %memEnd_3, %2#1, %1#1, %memEnd_1, %memEnd, %0#1 : <i32>, <>, <>, <>, <>, <>, <>
  }
}

