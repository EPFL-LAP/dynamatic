module {
  handshake.func @mvt_float(%arg0: memref<900xf32>, %arg1: memref<30xf32>, %arg2: memref<30xf32>, %arg3: memref<30xf32>, %arg4: memref<30xf32>, %arg5: !handshake.control<>, %arg6: !handshake.control<>, %arg7: !handshake.control<>, %arg8: !handshake.control<>, %arg9: !handshake.control<>, %arg10: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["A", "x1", "x2", "y1", "y2", "A_start", "x1_start", "x2_start", "y1_start", "y2_start", "start"], resNames = ["A_end", "x1_end", "x2_end", "y1_end", "y2_end", "end"]} {
    %0:3 = fork [3] %arg10 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork0"} : <>
    %outputs, %memEnd = mem_controller[%arg4 : memref<30xf32>] %arg9 (%addressResult_32) %147#4 {connectedBlocks = [5 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller3"} :    (!handshake.channel<i5>) -> !handshake.channel<f32>
    %outputs_0, %memEnd_1 = mem_controller[%arg3 : memref<30xf32>] %arg8 (%addressResult_8) %147#3 {connectedBlocks = [2 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller4"} :    (!handshake.channel<i5>) -> !handshake.channel<f32>
    %1:2 = lsq[%arg2 : memref<30xf32>] (%arg7, %83#0, %addressResult_26, %130#0, %addressResult_42, %dataResult_43, %147#2)  {groupSizes = [1 : i32, 1 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "5": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "6": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "lsq2"} : (!handshake.control<>, !handshake.control<>, !handshake.channel<i5>, !handshake.control<>, !handshake.channel<i5>, !handshake.channel<f32>, !handshake.control<>) -> (!handshake.channel<f32>, !handshake.control<>)
    %2:2 = lsq[%arg1 : memref<30xf32>] (%arg6, %9#0, %addressResult, %59#0, %addressResult_16, %dataResult_17, %147#1)  {groupSizes = [1 : i32, 1 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "5": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "6": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "lsq3"} : (!handshake.control<>, !handshake.control<>, !handshake.channel<i5>, !handshake.control<>, !handshake.channel<i5>, !handshake.channel<f32>, !handshake.control<>) -> (!handshake.channel<f32>, !handshake.control<>)
    %outputs_2:2, %memEnd_3 = mem_controller[%arg0 : memref<900xf32>] %arg5 (%addressResult_6, %addressResult_30) %147#0 {connectedBlocks = [2 : i32, 5 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller5"} :    (!handshake.channel<i10>, !handshake.channel<i10>) -> (!handshake.channel<f32>, !handshake.channel<f32>)
    %3 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant0", value = false} : <>, <i1>
    %4 = extsi %3 {handshake.bb = 0 : ui32, handshake.name = "extsi15"} : <i1> to <i6>
    %5 = mux %index [%4, %trueResult_18] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i6>, <i6>] to <i6>
    %6 = buffer %5, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer0"} : <i6>
    %7:2 = fork [2] %6 {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <i6>
    %8 = trunci %7#0 {handshake.bb = 1 : ui32, handshake.name = "trunci0"} : <i6> to <i5>
    %result, %index = control_merge [%0#2, %trueResult_20]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %9:3 = lazy_fork [3] %result {handshake.bb = 1 : ui32, handshake.name = "lazy_fork0"} : <>
    %10 = buffer %9#2, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer4"} : <>
    %11 = constant %10 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"0": [1,inf], [0,inf], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "constant1", value = false} : <>, <i1>
    %12 = buffer %8, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer1"} : <i5>
    %addressResult, %dataResult = load[%12] %2#0 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["store0", 2]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 0>, handshake.name = "load0"} : <i5>, <f32>, <i5>, <f32>
    %13 = extsi %11 {handshake.bb = 1 : ui32, handshake.name = "extsi14"} : <i1> to <i6>
    %14 = buffer %9#1, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer2"} : <>
    %15 = mux %32#1 [%13, %trueResult] {handshake.bb = 2 : ui32, handshake.name = "mux1"} : <i1>, [<i6>, <i6>] to <i6>
    %16 = buffer %15, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer7"} : <i6>
    %17:3 = fork [3] %16 {handshake.bb = 2 : ui32, handshake.name = "fork2"} : <i6>
    %18 = extsi %19 {handshake.bb = 2 : ui32, handshake.name = "extsi16"} : <i6> to <i10>
    %19 = buffer %17#0, bufferType = FIFO_BREAK_NONE, numSlots = 9 {handshake.bb = 2 : ui32, handshake.name = "buffer3"} : <i6>
    %20 = extsi %17#2 {handshake.bb = 2 : ui32, handshake.name = "extsi17"} : <i6> to <i7>
    %21 = trunci %22 {handshake.bb = 2 : ui32, handshake.name = "trunci1"} : <i6> to <i5>
    %22 = buffer %17#1, bufferType = FIFO_BREAK_NONE, numSlots = 9 {handshake.bb = 2 : ui32, handshake.name = "buffer5"} : <i6>
    %23 = mux %24 [%dataResult, %trueResult_10] {handshake.bb = 2 : ui32, handshake.name = "mux2"} : <i1>, [<f32>, <f32>] to <f32>
    %24 = buffer %32#2, bufferType = FIFO_BREAK_NONE, numSlots = 15 {handshake.bb = 2 : ui32, handshake.name = "buffer6"} : <i1>
    %25 = mux %32#0 [%7#1, %trueResult_12] {handshake.bb = 2 : ui32, handshake.name = "mux3"} : <i1>, [<i6>, <i6>] to <i6>
    %26 = buffer %25, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer11"} : <i6>
    %27 = buffer %26, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer12"} : <i6>
    %28:2 = fork [2] %27 {handshake.bb = 2 : ui32, handshake.name = "fork3"} : <i6>
    %29 = extsi %30 {handshake.bb = 2 : ui32, handshake.name = "extsi18"} : <i6> to <i12>
    %30 = buffer %28#1, bufferType = FIFO_BREAK_NONE, numSlots = 5 {handshake.bb = 2 : ui32, handshake.name = "buffer8"} : <i6>
    %31 = buffer %trueResult_14, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer16"} : <>
    %result_4, %index_5 = control_merge [%14, %31]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>, <>] to <>, <i1>
    %32:3 = fork [3] %index_5 {handshake.bb = 2 : ui32, handshake.name = "fork4"} : <i1>
    %33 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %34 = constant %33 {handshake.bb = 2 : ui32, handshake.name = "constant14", value = 30 : i6} : <>, <i6>
    %35:2 = fork [2] %34 {handshake.bb = 2 : ui32, handshake.name = "fork5"} : <i6>
    %36 = extsi %37 {handshake.bb = 2 : ui32, handshake.name = "extsi19"} : <i6> to <i12>
    %37 = buffer %35#0, bufferType = FIFO_BREAK_NONE, numSlots = 5 {handshake.bb = 2 : ui32, handshake.name = "buffer9"} : <i6>
    %38 = extsi %35#1 {handshake.bb = 2 : ui32, handshake.name = "extsi20"} : <i6> to <i7>
    %39 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %40 = constant %39 {handshake.bb = 2 : ui32, handshake.name = "constant15", value = 1 : i2} : <>, <i2>
    %41 = extsi %40 {handshake.bb = 2 : ui32, handshake.name = "extsi21"} : <i2> to <i7>
    %42 = muli %29, %36 {handshake.bb = 2 : ui32, handshake.name = "muli0"} : <i12>
    %43 = trunci %42 {handshake.bb = 2 : ui32, handshake.name = "trunci2"} : <i12> to <i10>
    %44 = addi %18, %43 {handshake.bb = 2 : ui32, handshake.name = "addi0"} : <i10>
    %addressResult_6, %dataResult_7 = load[%44] %outputs_2#0 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i10>, <f32>, <i10>, <f32>
    %addressResult_8, %dataResult_9 = load[%21] %outputs_0 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load2"} : <i5>, <f32>, <i5>, <f32>
    %45 = mulf %dataResult_7, %dataResult_9 {fastmath = #arith.fastmath<none>, fpu_impl = #handshake<fpu_impl flopoco>, handshake.bb = 2 : ui32, handshake.name = "mulf0", internal_delay = "2_875333"} : <f32>
    %46 = buffer %23, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer10"} : <f32>
    %47 = addf %46, %45 {fastmath = #arith.fastmath<none>, fpu_impl = #handshake<fpu_impl flopoco>, handshake.bb = 2 : ui32, handshake.name = "addf0", internal_delay = "2_922000"} : <f32>
    %48 = addi %20, %41 {handshake.bb = 2 : ui32, handshake.name = "addi2"} : <i7>
    %49 = buffer %48, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer13"} : <i7>
    %50:2 = fork [2] %49 {handshake.bb = 2 : ui32, handshake.name = "fork6"} : <i7>
    %51 = trunci %50#0 {handshake.bb = 2 : ui32, handshake.name = "trunci3"} : <i7> to <i6>
    %52 = cmpi ult, %50#1, %38 {handshake.bb = 2 : ui32, handshake.name = "cmpi0"} : <i7>
    %53 = buffer %52, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer15"} : <i1>
    %54:4 = fork [4] %53 {handshake.bb = 2 : ui32, handshake.name = "fork7"} : <i1>
    %trueResult, %falseResult = cond_br %54#0, %51 {handshake.bb = 2 : ui32, handshake.name = "cond_br4"} : <i1>, <i6>
    sink %falseResult {handshake.name = "sink0"} : <i6>
    %trueResult_10, %falseResult_11 = cond_br %55, %47 {handshake.bb = 2 : ui32, handshake.name = "cond_br5"} : <i1>, <f32>
    %55 = buffer %54#2, bufferType = FIFO_BREAK_NONE, numSlots = 15 {handshake.bb = 2 : ui32, handshake.name = "buffer14"} : <i1>
    %trueResult_12, %falseResult_13 = cond_br %54#1, %28#0 {handshake.bb = 2 : ui32, handshake.name = "cond_br6"} : <i1>, <i6>
    %trueResult_14, %falseResult_15 = cond_br %54#3, %result_4 {handshake.bb = 2 : ui32, handshake.name = "cond_br7"} : <i1>, <>
    %56:2 = fork [2] %falseResult_13 {handshake.bb = 3 : ui32, handshake.name = "fork8"} : <i6>
    %57 = extsi %56#1 {handshake.bb = 3 : ui32, handshake.name = "extsi22"} : <i6> to <i7>
    %58 = trunci %56#0 {handshake.bb = 3 : ui32, handshake.name = "trunci4"} : <i6> to <i5>
    %59:3 = lazy_fork [3] %falseResult_15 {handshake.bb = 3 : ui32, handshake.name = "lazy_fork1"} : <>
    %60 = buffer %59#2, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer20"} : <>
    %61 = constant %60 {handshake.bb = 3 : ui32, handshake.bufProps = #handshake<bufProps{"0": [1,inf], [0,inf], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "constant16", value = false} : <>, <i1>
    %62 = source {handshake.bb = 3 : ui32, handshake.name = "source2"} : <>
    %63 = constant %62 {handshake.bb = 3 : ui32, handshake.name = "constant17", value = 30 : i6} : <>, <i6>
    %64 = extsi %63 {handshake.bb = 3 : ui32, handshake.name = "extsi23"} : <i6> to <i7>
    %65 = source {handshake.bb = 3 : ui32, handshake.name = "source3"} : <>
    %66 = constant %65 {handshake.bb = 3 : ui32, handshake.name = "constant18", value = 1 : i2} : <>, <i2>
    %67 = extsi %66 {handshake.bb = 3 : ui32, handshake.name = "extsi24"} : <i2> to <i7>
    %68 = buffer %58, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer17"} : <i5>
    %69 = buffer %falseResult_11, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer18"} : <f32>
    %addressResult_16, %dataResult_17 = store[%68] %69 {handshake.bb = 3 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<LSQ: 1>, handshake.name = "store0"} : <i5>, <f32>, <i5>, <f32>
    %70 = addi %57, %67 {handshake.bb = 3 : ui32, handshake.name = "addi3"} : <i7>
    %71 = buffer %70, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer21"} : <i7>
    %72:2 = fork [2] %71 {handshake.bb = 3 : ui32, handshake.name = "fork9"} : <i7>
    %73 = trunci %72#0 {handshake.bb = 3 : ui32, handshake.name = "trunci5"} : <i7> to <i6>
    %74 = cmpi ult, %72#1, %64 {handshake.bb = 3 : ui32, handshake.name = "cmpi1"} : <i7>
    %75 = buffer %74, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer22"} : <i1>
    %76:3 = fork [3] %75 {handshake.bb = 3 : ui32, handshake.name = "fork10"} : <i1>
    %trueResult_18, %falseResult_19 = cond_br %76#0, %73 {handshake.bb = 3 : ui32, handshake.name = "cond_br8"} : <i1>, <i6>
    sink %falseResult_19 {handshake.name = "sink2"} : <i6>
    %77 = buffer %59#1, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer19"} : <>
    %trueResult_20, %falseResult_21 = cond_br %76#1, %77 {handshake.bb = 3 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,inf], [1,inf], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "cond_br9"} : <i1>, <>
    %trueResult_22, %falseResult_23 = cond_br %76#2, %61 {handshake.bb = 3 : ui32, handshake.name = "cond_br10"} : <i1>, <i1>
    sink %trueResult_22 {handshake.name = "sink3"} : <i1>
    %78 = extsi %falseResult_23 {handshake.bb = 3 : ui32, handshake.name = "extsi13"} : <i1> to <i6>
    %79 = mux %index_25 [%78, %trueResult_44] {handshake.bb = 4 : ui32, handshake.name = "mux4"} : <i1>, [<i6>, <i6>] to <i6>
    %80 = buffer %79, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer23"} : <i6>
    %81:2 = fork [2] %80 {handshake.bb = 4 : ui32, handshake.name = "fork11"} : <i6>
    %82 = trunci %81#0 {handshake.bb = 4 : ui32, handshake.name = "trunci6"} : <i6> to <i5>
    %result_24, %index_25 = control_merge [%falseResult_21, %trueResult_46]  {handshake.bb = 4 : ui32, handshake.name = "control_merge3"} : [<>, <>] to <>, <i1>
    %83:3 = lazy_fork [3] %result_24 {handshake.bb = 4 : ui32, handshake.name = "lazy_fork2"} : <>
    %84 = buffer %83#2, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer26"} : <>
    %85 = constant %84 {handshake.bb = 4 : ui32, handshake.bufProps = #handshake<bufProps{"0": [1,inf], [0,inf], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "constant19", value = false} : <>, <i1>
    %86 = buffer %82, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer24"} : <i5>
    %addressResult_26, %dataResult_27 = load[%86] %1#0 {handshake.bb = 4 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["store1", 2]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 0>, handshake.name = "load3"} : <i5>, <f32>, <i5>, <f32>
    %87 = extsi %85 {handshake.bb = 4 : ui32, handshake.name = "extsi12"} : <i1> to <i6>
    %88 = buffer %83#1, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer25"} : <>
    %89 = mux %102#1 [%87, %trueResult_34] {handshake.bb = 5 : ui32, handshake.name = "mux5"} : <i1>, [<i6>, <i6>] to <i6>
    %90 = buffer %89, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 5 : ui32, handshake.name = "buffer27"} : <i6>
    %91:3 = fork [3] %90 {handshake.bb = 5 : ui32, handshake.name = "fork12"} : <i6>
    %92 = extsi %91#1 {handshake.bb = 5 : ui32, handshake.name = "extsi25"} : <i6> to <i7>
    %93 = extsi %91#2 {handshake.bb = 5 : ui32, handshake.name = "extsi26"} : <i6> to <i12>
    %94 = trunci %91#0 {handshake.bb = 5 : ui32, handshake.name = "trunci7"} : <i6> to <i5>
    %95 = mux %96 [%dataResult_27, %trueResult_36] {handshake.bb = 5 : ui32, handshake.name = "mux6"} : <i1>, [<f32>, <f32>] to <f32>
    %96 = buffer %102#2, bufferType = FIFO_BREAK_NONE, numSlots = 3 {handshake.bb = 5 : ui32, handshake.name = "buffer31"} : <i1>
    %97 = mux %102#0 [%81#1, %trueResult_38] {handshake.bb = 5 : ui32, handshake.name = "mux7"} : <i1>, [<i6>, <i6>] to <i6>
    %98 = buffer %97, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 5 : ui32, handshake.name = "buffer29"} : <i6>
    %99 = buffer %98, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 5 : ui32, handshake.name = "buffer30"} : <i6>
    %100:2 = fork [2] %99 {handshake.bb = 5 : ui32, handshake.name = "fork13"} : <i6>
    %101 = extsi %100#0 {handshake.bb = 5 : ui32, handshake.name = "extsi27"} : <i6> to <i10>
    %result_28, %index_29 = control_merge [%88, %trueResult_40]  {handshake.bb = 5 : ui32, handshake.name = "control_merge4"} : [<>, <>] to <>, <i1>
    %102:3 = fork [3] %index_29 {handshake.bb = 5 : ui32, handshake.name = "fork14"} : <i1>
    %103 = source {handshake.bb = 5 : ui32, handshake.name = "source4"} : <>
    %104 = constant %103 {handshake.bb = 5 : ui32, handshake.name = "constant20", value = 30 : i6} : <>, <i6>
    %105:2 = fork [2] %104 {handshake.bb = 5 : ui32, handshake.name = "fork15"} : <i6>
    %106 = extsi %105#0 {handshake.bb = 5 : ui32, handshake.name = "extsi28"} : <i6> to <i7>
    %107 = extsi %105#1 {handshake.bb = 5 : ui32, handshake.name = "extsi29"} : <i6> to <i12>
    %108 = source {handshake.bb = 5 : ui32, handshake.name = "source5"} : <>
    %109 = constant %108 {handshake.bb = 5 : ui32, handshake.name = "constant21", value = 1 : i2} : <>, <i2>
    %110 = extsi %109 {handshake.bb = 5 : ui32, handshake.name = "extsi30"} : <i2> to <i7>
    %111 = muli %93, %107 {handshake.bb = 5 : ui32, handshake.name = "muli1"} : <i12>
    %112 = trunci %111 {handshake.bb = 5 : ui32, handshake.name = "trunci8"} : <i12> to <i10>
    %113 = addi %101, %112 {handshake.bb = 5 : ui32, handshake.name = "addi1"} : <i10>
    %addressResult_30, %dataResult_31 = load[%113] %outputs_2#1 {handshake.bb = 5 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load4"} : <i10>, <f32>, <i10>, <f32>
    %addressResult_32, %dataResult_33 = load[%94] %outputs {handshake.bb = 5 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load5"} : <i5>, <f32>, <i5>, <f32>
    %114 = mulf %dataResult_31, %dataResult_33 {fastmath = #arith.fastmath<none>, fpu_impl = #handshake<fpu_impl flopoco>, handshake.bb = 5 : ui32, handshake.name = "mulf1", internal_delay = "2_875333"} : <f32>
    %115 = buffer %95, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 5 : ui32, handshake.name = "buffer28"} : <f32>
    %116 = addf %115, %114 {fastmath = #arith.fastmath<none>, fpu_impl = #handshake<fpu_impl flopoco>, handshake.bb = 5 : ui32, handshake.name = "addf1", internal_delay = "2_922000"} : <f32>
    %117 = addi %92, %110 {handshake.bb = 5 : ui32, handshake.name = "addi4"} : <i7>
    %118 = buffer %117, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 5 : ui32, handshake.name = "buffer33"} : <i7>
    %119:2 = fork [2] %118 {handshake.bb = 5 : ui32, handshake.name = "fork16"} : <i7>
    %120 = trunci %119#0 {handshake.bb = 5 : ui32, handshake.name = "trunci9"} : <i7> to <i6>
    %121 = cmpi ult, %119#1, %106 {handshake.bb = 5 : ui32, handshake.name = "cmpi2"} : <i7>
    %122 = buffer %121, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 5 : ui32, handshake.name = "buffer34"} : <i1>
    %123:4 = fork [4] %122 {handshake.bb = 5 : ui32, handshake.name = "fork17"} : <i1>
    %trueResult_34, %falseResult_35 = cond_br %123#0, %120 {handshake.bb = 5 : ui32, handshake.name = "cond_br11"} : <i1>, <i6>
    sink %falseResult_35 {handshake.name = "sink4"} : <i6>
    %trueResult_36, %falseResult_37 = cond_br %124, %116 {handshake.bb = 5 : ui32, handshake.name = "cond_br12"} : <i1>, <f32>
    %124 = buffer %123#2, bufferType = FIFO_BREAK_NONE, numSlots = 4 {handshake.bb = 5 : ui32, handshake.name = "buffer39"} : <i1>
    %trueResult_38, %falseResult_39 = cond_br %125, %100#1 {handshake.bb = 5 : ui32, handshake.name = "cond_br13"} : <i1>, <i6>
    %125 = buffer %123#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 5 : ui32, handshake.name = "buffer40"} : <i1>
    %126 = buffer %result_28, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 5 : ui32, handshake.name = "buffer32"} : <>
    %trueResult_40, %falseResult_41 = cond_br %123#3, %126 {handshake.bb = 5 : ui32, handshake.name = "cond_br14"} : <i1>, <>
    %127:2 = fork [2] %falseResult_39 {handshake.bb = 6 : ui32, handshake.name = "fork18"} : <i6>
    %128 = extsi %127#1 {handshake.bb = 6 : ui32, handshake.name = "extsi31"} : <i6> to <i7>
    %129 = trunci %127#0 {handshake.bb = 6 : ui32, handshake.name = "trunci10"} : <i6> to <i5>
    %130:2 = lazy_fork [2] %falseResult_41 {handshake.bb = 6 : ui32, handshake.name = "lazy_fork3"} : <>
    %131 = source {handshake.bb = 6 : ui32, handshake.name = "source6"} : <>
    %132 = constant %131 {handshake.bb = 6 : ui32, handshake.name = "constant22", value = 30 : i6} : <>, <i6>
    %133 = extsi %132 {handshake.bb = 6 : ui32, handshake.name = "extsi32"} : <i6> to <i7>
    %134 = source {handshake.bb = 6 : ui32, handshake.name = "source7"} : <>
    %135 = constant %134 {handshake.bb = 6 : ui32, handshake.name = "constant23", value = 1 : i2} : <>, <i2>
    %136 = extsi %135 {handshake.bb = 6 : ui32, handshake.name = "extsi33"} : <i2> to <i7>
    %137 = buffer %129, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 6 : ui32, handshake.name = "buffer35"} : <i5>
    %138 = buffer %falseResult_37, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 6 : ui32, handshake.name = "buffer36"} : <f32>
    %addressResult_42, %dataResult_43 = store[%137] %138 {handshake.bb = 6 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<LSQ: 1>, handshake.name = "store1"} : <i5>, <f32>, <i5>, <f32>
    %139 = addi %128, %136 {handshake.bb = 6 : ui32, handshake.name = "addi5"} : <i7>
    %140 = buffer %139, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 6 : ui32, handshake.name = "buffer38"} : <i7>
    %141:2 = fork [2] %140 {handshake.bb = 6 : ui32, handshake.name = "fork19"} : <i7>
    %142 = trunci %141#0 {handshake.bb = 6 : ui32, handshake.name = "trunci11"} : <i7> to <i6>
    %143 = cmpi ult, %141#1, %133 {handshake.bb = 6 : ui32, handshake.name = "cmpi3"} : <i7>
    %144 = buffer %143, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 6 : ui32, handshake.name = "buffer41"} : <i1>
    %145:2 = fork [2] %144 {handshake.bb = 6 : ui32, handshake.name = "fork20"} : <i1>
    %trueResult_44, %falseResult_45 = cond_br %145#0, %142 {handshake.bb = 6 : ui32, handshake.name = "cond_br15"} : <i1>, <i6>
    sink %falseResult_45 {handshake.name = "sink6"} : <i6>
    %146 = buffer %130#1, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 6 : ui32, handshake.name = "buffer37"} : <>
    %trueResult_46, %falseResult_47 = cond_br %145#1, %146 {handshake.bb = 6 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,inf], [1,inf], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "cond_br16"} : <i1>, <>
    %147:5 = fork [5] %falseResult_47 {handshake.bb = 7 : ui32, handshake.name = "fork21"} : <>
    end {handshake.bb = 7 : ui32, handshake.name = "end0"} %memEnd_3, %2#1, %1#1, %memEnd_1, %memEnd, %0#1 : <>, <>, <>, <>, <>, <>
  }
}

