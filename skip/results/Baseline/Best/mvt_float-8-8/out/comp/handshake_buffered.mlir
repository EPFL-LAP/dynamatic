module {
  handshake.func @mvt_float(%arg0: memref<900xf32>, %arg1: memref<30xf32>, %arg2: memref<30xf32>, %arg3: memref<30xf32>, %arg4: memref<30xf32>, %arg5: !handshake.control<>, %arg6: !handshake.control<>, %arg7: !handshake.control<>, %arg8: !handshake.control<>, %arg9: !handshake.control<>, %arg10: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["A", "x1", "x2", "y1", "y2", "A_start", "x1_start", "x2_start", "y1_start", "y2_start", "start"], resNames = ["A_end", "x1_end", "x2_end", "y1_end", "y2_end", "end"]} {
    %0:3 = fork [3] %arg10 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork0"} : <>
    %outputs, %memEnd = mem_controller[%arg4 : memref<30xf32>] %arg9 (%addressResult_34) %161#4 {connectedBlocks = [5 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller3"} :    (!handshake.channel<i5>) -> !handshake.channel<f32>
    %outputs_0, %memEnd_1 = mem_controller[%arg3 : memref<30xf32>] %arg8 (%addressResult_8) %161#3 {connectedBlocks = [2 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller4"} :    (!handshake.channel<i5>) -> !handshake.channel<f32>
    %1:2 = lsq[%arg2 : memref<30xf32>] (%arg7, %91#0, %addressResult_28, %144#0, %addressResult_46, %dataResult_47, %161#2)  {groupSizes = [1 : i32, 1 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "5": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "6": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "lsq2"} : (!handshake.control<>, !handshake.control<>, !handshake.channel<i5>, !handshake.control<>, !handshake.channel<i5>, !handshake.channel<f32>, !handshake.control<>) -> (!handshake.channel<f32>, !handshake.control<>)
    %2:2 = lsq[%arg1 : memref<30xf32>] (%arg6, %11#0, %addressResult, %67#0, %addressResult_18, %dataResult_19, %161#1)  {groupSizes = [1 : i32, 1 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "5": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "6": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "lsq3"} : (!handshake.control<>, !handshake.control<>, !handshake.channel<i5>, !handshake.control<>, !handshake.channel<i5>, !handshake.channel<f32>, !handshake.control<>) -> (!handshake.channel<f32>, !handshake.control<>)
    %outputs_2:2, %memEnd_3 = mem_controller[%arg0 : memref<900xf32>] %arg5 (%addressResult_6, %addressResult_32) %161#0 {connectedBlocks = [2 : i32, 5 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller5"} :    (!handshake.channel<i10>, !handshake.channel<i10>) -> (!handshake.channel<f32>, !handshake.channel<f32>)
    %3 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant0", value = false} : <>, <i1>
    %4 = br %3 {handshake.bb = 0 : ui32, handshake.name = "br2"} : <i1>
    %5 = extsi %4 {handshake.bb = 0 : ui32, handshake.name = "extsi15"} : <i1> to <i6>
    %6 = br %0#2 {handshake.bb = 0 : ui32, handshake.name = "br4"} : <>
    %7 = mux %index [%5, %trueResult_20] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i6>, <i6>] to <i6>
    %8 = buffer %7, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer0"} : <i6>
    %9:2 = fork [2] %8 {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <i6>
    %10 = trunci %9#0 {handshake.bb = 1 : ui32, handshake.name = "trunci0"} : <i6> to <i5>
    %result, %index = control_merge [%6, %trueResult_22]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %11:3 = lazy_fork [3] %result {handshake.bb = 1 : ui32, handshake.name = "lazy_fork0"} : <>
    %12 = buffer %11#2, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer4"} : <>
    %13 = constant %12 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"0": [1,inf], [0,inf], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "constant1", value = false} : <>, <i1>
    %14 = buffer %10, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer1"} : <i5>
    %addressResult, %dataResult = load[%14] %2#0 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["store0", 2]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 0>, handshake.name = "load0"} : <i5>, <f32>, <i5>, <f32>
    %15 = br %13 {handshake.bb = 1 : ui32, handshake.name = "br5"} : <i1>
    %16 = extsi %15 {handshake.bb = 1 : ui32, handshake.name = "extsi14"} : <i1> to <i6>
    %17 = br %dataResult {handshake.bb = 1 : ui32, handshake.name = "br6"} : <f32>
    %18 = br %9#1 {handshake.bb = 1 : ui32, handshake.name = "br7"} : <i6>
    %19 = buffer %11#1, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer2"} : <>
    %20 = br %19 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [1,inf], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "br8"} : <>
    %21 = mux %38#1 [%16, %trueResult] {handshake.bb = 2 : ui32, handshake.name = "mux1"} : <i1>, [<i6>, <i6>] to <i6>
    %22 = buffer %21, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer7"} : <i6>
    %23:3 = fork [3] %22 {handshake.bb = 2 : ui32, handshake.name = "fork2"} : <i6>
    %24 = extsi %25 {handshake.bb = 2 : ui32, handshake.name = "extsi16"} : <i6> to <i10>
    %25 = buffer %23#0, bufferType = FIFO_BREAK_NONE, numSlots = 9 {handshake.bb = 2 : ui32, handshake.name = "buffer3"} : <i6>
    %26 = extsi %23#2 {handshake.bb = 2 : ui32, handshake.name = "extsi17"} : <i6> to <i7>
    %27 = trunci %28 {handshake.bb = 2 : ui32, handshake.name = "trunci1"} : <i6> to <i5>
    %28 = buffer %23#1, bufferType = FIFO_BREAK_NONE, numSlots = 9 {handshake.bb = 2 : ui32, handshake.name = "buffer5"} : <i6>
    %29 = mux %30 [%17, %trueResult_10] {handshake.bb = 2 : ui32, handshake.name = "mux2"} : <i1>, [<f32>, <f32>] to <f32>
    %30 = buffer %38#2, bufferType = FIFO_BREAK_NONE, numSlots = 15 {handshake.bb = 2 : ui32, handshake.name = "buffer6"} : <i1>
    %31 = mux %38#0 [%18, %trueResult_12] {handshake.bb = 2 : ui32, handshake.name = "mux3"} : <i1>, [<i6>, <i6>] to <i6>
    %32 = buffer %31, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer11"} : <i6>
    %33 = buffer %32, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer12"} : <i6>
    %34:2 = fork [2] %33 {handshake.bb = 2 : ui32, handshake.name = "fork3"} : <i6>
    %35 = extsi %36 {handshake.bb = 2 : ui32, handshake.name = "extsi18"} : <i6> to <i12>
    %36 = buffer %34#1, bufferType = FIFO_BREAK_NONE, numSlots = 5 {handshake.bb = 2 : ui32, handshake.name = "buffer8"} : <i6>
    %37 = buffer %trueResult_14, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer16"} : <>
    %result_4, %index_5 = control_merge [%20, %37]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>, <>] to <>, <i1>
    %38:3 = fork [3] %index_5 {handshake.bb = 2 : ui32, handshake.name = "fork4"} : <i1>
    %39 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %40 = constant %39 {handshake.bb = 2 : ui32, handshake.name = "constant14", value = 30 : i6} : <>, <i6>
    %41:2 = fork [2] %40 {handshake.bb = 2 : ui32, handshake.name = "fork5"} : <i6>
    %42 = extsi %43 {handshake.bb = 2 : ui32, handshake.name = "extsi19"} : <i6> to <i12>
    %43 = buffer %41#0, bufferType = FIFO_BREAK_NONE, numSlots = 5 {handshake.bb = 2 : ui32, handshake.name = "buffer9"} : <i6>
    %44 = extsi %41#1 {handshake.bb = 2 : ui32, handshake.name = "extsi20"} : <i6> to <i7>
    %45 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %46 = constant %45 {handshake.bb = 2 : ui32, handshake.name = "constant15", value = 1 : i2} : <>, <i2>
    %47 = extsi %46 {handshake.bb = 2 : ui32, handshake.name = "extsi21"} : <i2> to <i7>
    %48 = muli %35, %42 {handshake.bb = 2 : ui32, handshake.name = "muli0"} : <i12>
    %49 = trunci %48 {handshake.bb = 2 : ui32, handshake.name = "trunci2"} : <i12> to <i10>
    %50 = addi %24, %49 {handshake.bb = 2 : ui32, handshake.name = "addi0"} : <i10>
    %addressResult_6, %dataResult_7 = load[%50] %outputs_2#0 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i10>, <f32>, <i10>, <f32>
    %addressResult_8, %dataResult_9 = load[%27] %outputs_0 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load2"} : <i5>, <f32>, <i5>, <f32>
    %51 = mulf %dataResult_7, %dataResult_9 {fastmath = #arith.fastmath<none>, fpu_impl = #handshake<fpu_impl flopoco>, handshake.bb = 2 : ui32, handshake.name = "mulf0", internal_delay = "2_875333"} : <f32>
    %52 = buffer %29, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer10"} : <f32>
    %53 = addf %52, %51 {fastmath = #arith.fastmath<none>, fpu_impl = #handshake<fpu_impl flopoco>, handshake.bb = 2 : ui32, handshake.name = "addf0", internal_delay = "2_922000"} : <f32>
    %54 = addi %26, %47 {handshake.bb = 2 : ui32, handshake.name = "addi2"} : <i7>
    %55 = buffer %54, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer13"} : <i7>
    %56:2 = fork [2] %55 {handshake.bb = 2 : ui32, handshake.name = "fork6"} : <i7>
    %57 = trunci %56#0 {handshake.bb = 2 : ui32, handshake.name = "trunci3"} : <i7> to <i6>
    %58 = cmpi ult, %56#1, %44 {handshake.bb = 2 : ui32, handshake.name = "cmpi0"} : <i7>
    %59 = buffer %58, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer15"} : <i1>
    %60:4 = fork [4] %59 {handshake.bb = 2 : ui32, handshake.name = "fork7"} : <i1>
    %trueResult, %falseResult = cond_br %60#0, %57 {handshake.bb = 2 : ui32, handshake.name = "cond_br4"} : <i1>, <i6>
    sink %falseResult {handshake.name = "sink0"} : <i6>
    %trueResult_10, %falseResult_11 = cond_br %61, %53 {handshake.bb = 2 : ui32, handshake.name = "cond_br5"} : <i1>, <f32>
    %61 = buffer %60#2, bufferType = FIFO_BREAK_NONE, numSlots = 15 {handshake.bb = 2 : ui32, handshake.name = "buffer14"} : <i1>
    %trueResult_12, %falseResult_13 = cond_br %60#1, %34#0 {handshake.bb = 2 : ui32, handshake.name = "cond_br6"} : <i1>, <i6>
    %trueResult_14, %falseResult_15 = cond_br %60#3, %result_4 {handshake.bb = 2 : ui32, handshake.name = "cond_br7"} : <i1>, <>
    %62 = merge %falseResult_13 {handshake.bb = 3 : ui32, handshake.name = "merge0"} : <i6>
    %63:2 = fork [2] %62 {handshake.bb = 3 : ui32, handshake.name = "fork8"} : <i6>
    %64 = extsi %63#1 {handshake.bb = 3 : ui32, handshake.name = "extsi22"} : <i6> to <i7>
    %65 = trunci %63#0 {handshake.bb = 3 : ui32, handshake.name = "trunci4"} : <i6> to <i5>
    %66 = merge %falseResult_11 {handshake.bb = 3 : ui32, handshake.name = "merge1"} : <f32>
    %result_16, %index_17 = control_merge [%falseResult_15]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>] to <>, <i1>
    sink %index_17 {handshake.name = "sink1"} : <i1>
    %67:3 = lazy_fork [3] %result_16 {handshake.bb = 3 : ui32, handshake.name = "lazy_fork1"} : <>
    %68 = buffer %67#2, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer20"} : <>
    %69 = constant %68 {handshake.bb = 3 : ui32, handshake.bufProps = #handshake<bufProps{"0": [1,inf], [0,inf], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "constant16", value = false} : <>, <i1>
    %70 = source {handshake.bb = 3 : ui32, handshake.name = "source2"} : <>
    %71 = constant %70 {handshake.bb = 3 : ui32, handshake.name = "constant17", value = 30 : i6} : <>, <i6>
    %72 = extsi %71 {handshake.bb = 3 : ui32, handshake.name = "extsi23"} : <i6> to <i7>
    %73 = source {handshake.bb = 3 : ui32, handshake.name = "source3"} : <>
    %74 = constant %73 {handshake.bb = 3 : ui32, handshake.name = "constant18", value = 1 : i2} : <>, <i2>
    %75 = extsi %74 {handshake.bb = 3 : ui32, handshake.name = "extsi24"} : <i2> to <i7>
    %76 = buffer %65, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer17"} : <i5>
    %77 = buffer %66, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer18"} : <f32>
    %addressResult_18, %dataResult_19 = store[%76] %77 {handshake.bb = 3 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<LSQ: 1>, handshake.name = "store0"} : <i5>, <f32>, <i5>, <f32>
    %78 = addi %64, %75 {handshake.bb = 3 : ui32, handshake.name = "addi3"} : <i7>
    %79 = buffer %78, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer21"} : <i7>
    %80:2 = fork [2] %79 {handshake.bb = 3 : ui32, handshake.name = "fork9"} : <i7>
    %81 = trunci %80#0 {handshake.bb = 3 : ui32, handshake.name = "trunci5"} : <i7> to <i6>
    %82 = cmpi ult, %80#1, %72 {handshake.bb = 3 : ui32, handshake.name = "cmpi1"} : <i7>
    %83 = buffer %82, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer22"} : <i1>
    %84:3 = fork [3] %83 {handshake.bb = 3 : ui32, handshake.name = "fork10"} : <i1>
    %trueResult_20, %falseResult_21 = cond_br %84#0, %81 {handshake.bb = 3 : ui32, handshake.name = "cond_br8"} : <i1>, <i6>
    sink %falseResult_21 {handshake.name = "sink2"} : <i6>
    %85 = buffer %67#1, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer19"} : <>
    %trueResult_22, %falseResult_23 = cond_br %84#1, %85 {handshake.bb = 3 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,inf], [1,inf], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "cond_br9"} : <i1>, <>
    %trueResult_24, %falseResult_25 = cond_br %84#2, %69 {handshake.bb = 3 : ui32, handshake.name = "cond_br10"} : <i1>, <i1>
    sink %trueResult_24 {handshake.name = "sink3"} : <i1>
    %86 = extsi %falseResult_25 {handshake.bb = 3 : ui32, handshake.name = "extsi13"} : <i1> to <i6>
    %87 = mux %index_27 [%86, %trueResult_48] {handshake.bb = 4 : ui32, handshake.name = "mux4"} : <i1>, [<i6>, <i6>] to <i6>
    %88 = buffer %87, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer23"} : <i6>
    %89:2 = fork [2] %88 {handshake.bb = 4 : ui32, handshake.name = "fork11"} : <i6>
    %90 = trunci %89#0 {handshake.bb = 4 : ui32, handshake.name = "trunci6"} : <i6> to <i5>
    %result_26, %index_27 = control_merge [%falseResult_23, %trueResult_50]  {handshake.bb = 4 : ui32, handshake.name = "control_merge3"} : [<>, <>] to <>, <i1>
    %91:3 = lazy_fork [3] %result_26 {handshake.bb = 4 : ui32, handshake.name = "lazy_fork2"} : <>
    %92 = buffer %91#2, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer26"} : <>
    %93 = constant %92 {handshake.bb = 4 : ui32, handshake.bufProps = #handshake<bufProps{"0": [1,inf], [0,inf], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "constant19", value = false} : <>, <i1>
    %94 = buffer %90, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer24"} : <i5>
    %addressResult_28, %dataResult_29 = load[%94] %1#0 {handshake.bb = 4 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["store1", 2]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 0>, handshake.name = "load3"} : <i5>, <f32>, <i5>, <f32>
    %95 = br %93 {handshake.bb = 4 : ui32, handshake.name = "br9"} : <i1>
    %96 = extsi %95 {handshake.bb = 4 : ui32, handshake.name = "extsi12"} : <i1> to <i6>
    %97 = br %dataResult_29 {handshake.bb = 4 : ui32, handshake.name = "br10"} : <f32>
    %98 = br %89#1 {handshake.bb = 4 : ui32, handshake.name = "br11"} : <i6>
    %99 = buffer %91#1, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer25"} : <>
    %100 = br %99 {handshake.bb = 4 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [1,inf], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "br12"} : <>
    %101 = mux %114#1 [%96, %trueResult_36] {handshake.bb = 5 : ui32, handshake.name = "mux5"} : <i1>, [<i6>, <i6>] to <i6>
    %102 = buffer %101, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 5 : ui32, handshake.name = "buffer27"} : <i6>
    %103:3 = fork [3] %102 {handshake.bb = 5 : ui32, handshake.name = "fork12"} : <i6>
    %104 = extsi %103#1 {handshake.bb = 5 : ui32, handshake.name = "extsi25"} : <i6> to <i7>
    %105 = extsi %103#2 {handshake.bb = 5 : ui32, handshake.name = "extsi26"} : <i6> to <i12>
    %106 = trunci %103#0 {handshake.bb = 5 : ui32, handshake.name = "trunci7"} : <i6> to <i5>
    %107 = mux %108 [%97, %trueResult_38] {handshake.bb = 5 : ui32, handshake.name = "mux6"} : <i1>, [<f32>, <f32>] to <f32>
    %108 = buffer %114#2, bufferType = FIFO_BREAK_NONE, numSlots = 3 {handshake.bb = 5 : ui32, handshake.name = "buffer31"} : <i1>
    %109 = mux %114#0 [%98, %trueResult_40] {handshake.bb = 5 : ui32, handshake.name = "mux7"} : <i1>, [<i6>, <i6>] to <i6>
    %110 = buffer %109, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 5 : ui32, handshake.name = "buffer29"} : <i6>
    %111 = buffer %110, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 5 : ui32, handshake.name = "buffer30"} : <i6>
    %112:2 = fork [2] %111 {handshake.bb = 5 : ui32, handshake.name = "fork13"} : <i6>
    %113 = extsi %112#0 {handshake.bb = 5 : ui32, handshake.name = "extsi27"} : <i6> to <i10>
    %result_30, %index_31 = control_merge [%100, %trueResult_42]  {handshake.bb = 5 : ui32, handshake.name = "control_merge4"} : [<>, <>] to <>, <i1>
    %114:3 = fork [3] %index_31 {handshake.bb = 5 : ui32, handshake.name = "fork14"} : <i1>
    %115 = source {handshake.bb = 5 : ui32, handshake.name = "source4"} : <>
    %116 = constant %115 {handshake.bb = 5 : ui32, handshake.name = "constant20", value = 30 : i6} : <>, <i6>
    %117:2 = fork [2] %116 {handshake.bb = 5 : ui32, handshake.name = "fork15"} : <i6>
    %118 = extsi %117#0 {handshake.bb = 5 : ui32, handshake.name = "extsi28"} : <i6> to <i7>
    %119 = extsi %117#1 {handshake.bb = 5 : ui32, handshake.name = "extsi29"} : <i6> to <i12>
    %120 = source {handshake.bb = 5 : ui32, handshake.name = "source5"} : <>
    %121 = constant %120 {handshake.bb = 5 : ui32, handshake.name = "constant21", value = 1 : i2} : <>, <i2>
    %122 = extsi %121 {handshake.bb = 5 : ui32, handshake.name = "extsi30"} : <i2> to <i7>
    %123 = muli %105, %119 {handshake.bb = 5 : ui32, handshake.name = "muli1"} : <i12>
    %124 = trunci %123 {handshake.bb = 5 : ui32, handshake.name = "trunci8"} : <i12> to <i10>
    %125 = addi %113, %124 {handshake.bb = 5 : ui32, handshake.name = "addi1"} : <i10>
    %addressResult_32, %dataResult_33 = load[%125] %outputs_2#1 {handshake.bb = 5 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load4"} : <i10>, <f32>, <i10>, <f32>
    %addressResult_34, %dataResult_35 = load[%106] %outputs {handshake.bb = 5 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load5"} : <i5>, <f32>, <i5>, <f32>
    %126 = mulf %dataResult_33, %dataResult_35 {fastmath = #arith.fastmath<none>, fpu_impl = #handshake<fpu_impl flopoco>, handshake.bb = 5 : ui32, handshake.name = "mulf1", internal_delay = "2_875333"} : <f32>
    %127 = buffer %107, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 5 : ui32, handshake.name = "buffer28"} : <f32>
    %128 = addf %127, %126 {fastmath = #arith.fastmath<none>, fpu_impl = #handshake<fpu_impl flopoco>, handshake.bb = 5 : ui32, handshake.name = "addf1", internal_delay = "2_922000"} : <f32>
    %129 = addi %104, %122 {handshake.bb = 5 : ui32, handshake.name = "addi4"} : <i7>
    %130 = buffer %129, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 5 : ui32, handshake.name = "buffer33"} : <i7>
    %131:2 = fork [2] %130 {handshake.bb = 5 : ui32, handshake.name = "fork16"} : <i7>
    %132 = trunci %131#0 {handshake.bb = 5 : ui32, handshake.name = "trunci9"} : <i7> to <i6>
    %133 = cmpi ult, %131#1, %118 {handshake.bb = 5 : ui32, handshake.name = "cmpi2"} : <i7>
    %134 = buffer %133, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 5 : ui32, handshake.name = "buffer34"} : <i1>
    %135:4 = fork [4] %134 {handshake.bb = 5 : ui32, handshake.name = "fork17"} : <i1>
    %trueResult_36, %falseResult_37 = cond_br %135#0, %132 {handshake.bb = 5 : ui32, handshake.name = "cond_br11"} : <i1>, <i6>
    sink %falseResult_37 {handshake.name = "sink4"} : <i6>
    %trueResult_38, %falseResult_39 = cond_br %136, %128 {handshake.bb = 5 : ui32, handshake.name = "cond_br12"} : <i1>, <f32>
    %136 = buffer %135#2, bufferType = FIFO_BREAK_NONE, numSlots = 4 {handshake.bb = 5 : ui32, handshake.name = "buffer39"} : <i1>
    %trueResult_40, %falseResult_41 = cond_br %137, %112#1 {handshake.bb = 5 : ui32, handshake.name = "cond_br13"} : <i1>, <i6>
    %137 = buffer %135#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 5 : ui32, handshake.name = "buffer40"} : <i1>
    %138 = buffer %result_30, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 5 : ui32, handshake.name = "buffer32"} : <>
    %trueResult_42, %falseResult_43 = cond_br %135#3, %138 {handshake.bb = 5 : ui32, handshake.name = "cond_br14"} : <i1>, <>
    %139 = merge %falseResult_41 {handshake.bb = 6 : ui32, handshake.name = "merge2"} : <i6>
    %140:2 = fork [2] %139 {handshake.bb = 6 : ui32, handshake.name = "fork18"} : <i6>
    %141 = extsi %140#1 {handshake.bb = 6 : ui32, handshake.name = "extsi31"} : <i6> to <i7>
    %142 = trunci %140#0 {handshake.bb = 6 : ui32, handshake.name = "trunci10"} : <i6> to <i5>
    %143 = merge %falseResult_39 {handshake.bb = 6 : ui32, handshake.name = "merge3"} : <f32>
    %result_44, %index_45 = control_merge [%falseResult_43]  {handshake.bb = 6 : ui32, handshake.name = "control_merge5"} : [<>] to <>, <i1>
    sink %index_45 {handshake.name = "sink5"} : <i1>
    %144:2 = lazy_fork [2] %result_44 {handshake.bb = 6 : ui32, handshake.name = "lazy_fork3"} : <>
    %145 = source {handshake.bb = 6 : ui32, handshake.name = "source6"} : <>
    %146 = constant %145 {handshake.bb = 6 : ui32, handshake.name = "constant22", value = 30 : i6} : <>, <i6>
    %147 = extsi %146 {handshake.bb = 6 : ui32, handshake.name = "extsi32"} : <i6> to <i7>
    %148 = source {handshake.bb = 6 : ui32, handshake.name = "source7"} : <>
    %149 = constant %148 {handshake.bb = 6 : ui32, handshake.name = "constant23", value = 1 : i2} : <>, <i2>
    %150 = extsi %149 {handshake.bb = 6 : ui32, handshake.name = "extsi33"} : <i2> to <i7>
    %151 = buffer %142, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 6 : ui32, handshake.name = "buffer35"} : <i5>
    %152 = buffer %143, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 6 : ui32, handshake.name = "buffer36"} : <f32>
    %addressResult_46, %dataResult_47 = store[%151] %152 {handshake.bb = 6 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<LSQ: 1>, handshake.name = "store1"} : <i5>, <f32>, <i5>, <f32>
    %153 = addi %141, %150 {handshake.bb = 6 : ui32, handshake.name = "addi5"} : <i7>
    %154 = buffer %153, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 6 : ui32, handshake.name = "buffer38"} : <i7>
    %155:2 = fork [2] %154 {handshake.bb = 6 : ui32, handshake.name = "fork19"} : <i7>
    %156 = trunci %155#0 {handshake.bb = 6 : ui32, handshake.name = "trunci11"} : <i7> to <i6>
    %157 = cmpi ult, %155#1, %147 {handshake.bb = 6 : ui32, handshake.name = "cmpi3"} : <i7>
    %158 = buffer %157, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 6 : ui32, handshake.name = "buffer41"} : <i1>
    %159:2 = fork [2] %158 {handshake.bb = 6 : ui32, handshake.name = "fork20"} : <i1>
    %trueResult_48, %falseResult_49 = cond_br %159#0, %156 {handshake.bb = 6 : ui32, handshake.name = "cond_br15"} : <i1>, <i6>
    sink %falseResult_49 {handshake.name = "sink6"} : <i6>
    %160 = buffer %144#1, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 6 : ui32, handshake.name = "buffer37"} : <>
    %trueResult_50, %falseResult_51 = cond_br %159#1, %160 {handshake.bb = 6 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,inf], [1,inf], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "cond_br16"} : <i1>, <>
    %result_52, %index_53 = control_merge [%falseResult_51]  {handshake.bb = 7 : ui32, handshake.name = "control_merge6"} : [<>] to <>, <i1>
    sink %index_53 {handshake.name = "sink7"} : <i1>
    %161:5 = fork [5] %result_52 {handshake.bb = 7 : ui32, handshake.name = "fork21"} : <>
    end {handshake.bb = 7 : ui32, handshake.name = "end0"} %memEnd_3, %2#1, %1#1, %memEnd_1, %memEnd, %0#1 : <>, <>, <>, <>, <>, <>
  }
}

