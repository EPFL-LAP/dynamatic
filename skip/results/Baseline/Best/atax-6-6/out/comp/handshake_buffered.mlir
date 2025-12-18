module {
  handshake.func @atax(%arg0: memref<400xf32>, %arg1: memref<20xf32>, %arg2: memref<20xf32>, %arg3: memref<20xf32>, %arg4: !handshake.control<>, %arg5: !handshake.control<>, %arg6: !handshake.control<>, %arg7: !handshake.control<>, %arg8: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["A", "x", "y", "tmp", "A_start", "x_start", "y_start", "tmp_start", "start"], resNames = ["A_end", "x_end", "y_end", "tmp_end", "end"]} {
    %0:3 = fork [3] %arg8 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork0"} : <>
    %1:2 = lsq[%arg3 : memref<20xf32>] (%arg7, %11#0, %addressResult, %144#0, %addressResult_34, %dataResult_35, %164#3)  {groupSizes = [1 : i32, 1 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "5": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "6": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "lsq2"} : (!handshake.control<>, !handshake.control<>, !handshake.channel<i5>, !handshake.control<>, !handshake.channel<i5>, !handshake.channel<f32>, !handshake.control<>) -> (!handshake.channel<f32>, !handshake.control<>)
    %2:2 = lsq[%arg2 : memref<20xf32>] (%arg6, %94#0, %addressResult_18, %addressResult_22, %dataResult_23, %164#2)  {groupSizes = [2 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "5": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "lsq3"} : (!handshake.control<>, !handshake.control<>, !handshake.channel<i5>, !handshake.channel<i5>, !handshake.channel<f32>, !handshake.control<>) -> (!handshake.channel<f32>, !handshake.control<>)
    %outputs, %memEnd = mem_controller[%arg1 : memref<20xf32>] %arg5 (%addressResult_6) %164#1 {connectedBlocks = [2 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller2"} :    (!handshake.channel<i5>) -> !handshake.channel<f32>
    %outputs_0:2, %memEnd_1 = mem_controller[%arg0 : memref<400xf32>] %arg4 (%addressResult_4, %addressResult_20) %164#0 {connectedBlocks = [2 : i32, 3 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller3"} :    (!handshake.channel<i9>, !handshake.channel<i9>) -> (!handshake.channel<f32>, !handshake.channel<f32>)
    %3 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant0", value = false} : <>, <i1>
    %4 = br %3 {handshake.bb = 0 : ui32, handshake.name = "br2"} : <i1>
    %5 = extsi %4 {handshake.bb = 0 : ui32, handshake.name = "extsi15"} : <i1> to <i6>
    %6 = br %0#2 {handshake.bb = 0 : ui32, handshake.name = "br3"} : <>
    %7 = mux %index [%5, %trueResult_36] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i6>, <i6>] to <i6>
    %8 = buffer %7, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer0"} : <i6>
    %9:2 = fork [2] %8 {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <i6>
    %10 = trunci %9#0 {handshake.bb = 1 : ui32, handshake.name = "trunci0"} : <i6> to <i5>
    %result, %index = control_merge [%6, %trueResult_38]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %11:3 = lazy_fork [3] %result {handshake.bb = 1 : ui32, handshake.name = "lazy_fork0"} : <>
    %12 = buffer %11#2, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer4"} : <>
    %13 = constant %12 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"0": [1,inf], [0,inf], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "constant1", value = false} : <>, <i1>
    %14 = buffer %10, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer2"} : <i5>
    %addressResult, %dataResult = load[%14] %1#0 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["store1", 2]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 0>, handshake.name = "load0"} : <i5>, <f32>, <i5>, <f32>
    %15 = br %13 {handshake.bb = 1 : ui32, handshake.name = "br4"} : <i1>
    %16 = extsi %15 {handshake.bb = 1 : ui32, handshake.name = "extsi14"} : <i1> to <i6>
    %17 = br %dataResult {handshake.bb = 1 : ui32, handshake.name = "br5"} : <f32>
    %18 = br %19 {handshake.bb = 1 : ui32, handshake.name = "br6"} : <i6>
    %19 = buffer %9#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer1"} : <i6>
    %20 = buffer %11#1, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer3"} : <>
    %21 = br %20 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [1,inf], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "br7"} : <>
    %22 = mux %35#1 [%16, %trueResult] {handshake.bb = 2 : ui32, handshake.name = "mux1"} : <i1>, [<i6>, <i6>] to <i6>
    %23 = buffer %22, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer5"} : <i6>
    %24:3 = fork [3] %23 {handshake.bb = 2 : ui32, handshake.name = "fork2"} : <i6>
    %25 = extsi %24#0 {handshake.bb = 2 : ui32, handshake.name = "extsi16"} : <i6> to <i9>
    %26 = extsi %24#2 {handshake.bb = 2 : ui32, handshake.name = "extsi17"} : <i6> to <i7>
    %27 = trunci %24#1 {handshake.bb = 2 : ui32, handshake.name = "trunci1"} : <i6> to <i5>
    %28 = mux %35#2 [%17, %trueResult_8] {handshake.bb = 2 : ui32, handshake.name = "mux2"} : <i1>, [<f32>, <f32>] to <f32>
    %29 = mux %35#0 [%18, %trueResult_10] {handshake.bb = 2 : ui32, handshake.name = "mux3"} : <i1>, [<i6>, <i6>] to <i6>
    %30 = buffer %29, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer7"} : <i6>
    %31 = buffer %30, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer8"} : <i6>
    %32:2 = fork [2] %31 {handshake.bb = 2 : ui32, handshake.name = "fork3"} : <i6>
    %33 = extsi %32#1 {handshake.bb = 2 : ui32, handshake.name = "extsi18"} : <i6> to <i32>
    %34:2 = fork [2] %33 {handshake.bb = 2 : ui32, handshake.name = "fork4"} : <i32>
    %result_2, %index_3 = control_merge [%21, %trueResult_12]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>, <>] to <>, <i1>
    %35:3 = fork [3] %index_3 {handshake.bb = 2 : ui32, handshake.name = "fork5"} : <i1>
    %36:2 = fork [2] %result_2 {handshake.bb = 2 : ui32, handshake.name = "fork6"} : <>
    %37 = constant %36#0 {handshake.bb = 2 : ui32, handshake.name = "constant2", value = false} : <>, <i1>
    %38 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %39 = constant %38 {handshake.bb = 2 : ui32, handshake.name = "constant3", value = 20 : i6} : <>, <i6>
    %40 = extsi %39 {handshake.bb = 2 : ui32, handshake.name = "extsi19"} : <i6> to <i7>
    %41 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %42 = constant %41 {handshake.bb = 2 : ui32, handshake.name = "constant17", value = 1 : i2} : <>, <i2>
    %43 = extsi %42 {handshake.bb = 2 : ui32, handshake.name = "extsi20"} : <i2> to <i7>
    %44 = source {handshake.bb = 2 : ui32, handshake.name = "source2"} : <>
    %45 = constant %44 {handshake.bb = 2 : ui32, handshake.name = "constant18", value = 4 : i4} : <>, <i4>
    %46 = extsi %45 {handshake.bb = 2 : ui32, handshake.name = "extsi5"} : <i4> to <i32>
    %47 = source {handshake.bb = 2 : ui32, handshake.name = "source3"} : <>
    %48 = constant %47 {handshake.bb = 2 : ui32, handshake.name = "constant19", value = 2 : i3} : <>, <i3>
    %49 = extsi %48 {handshake.bb = 2 : ui32, handshake.name = "extsi6"} : <i3> to <i32>
    %50 = shli %51, %49 {handshake.bb = 2 : ui32, handshake.name = "shli0"} : <i32>
    %51 = buffer %34#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer9"} : <i32>
    %52 = buffer %50, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer15"} : <i32>
    %53 = trunci %52 {handshake.bb = 2 : ui32, handshake.name = "trunci2"} : <i32> to <i9>
    %54 = shli %55, %46 {handshake.bb = 2 : ui32, handshake.name = "shli1"} : <i32>
    %55 = buffer %34#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer10"} : <i32>
    %56 = buffer %54, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer16"} : <i32>
    %57 = trunci %56 {handshake.bb = 2 : ui32, handshake.name = "trunci3"} : <i32> to <i9>
    %58 = addi %53, %57 {handshake.bb = 2 : ui32, handshake.name = "addi5"} : <i9>
    %59 = buffer %58, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer17"} : <i9>
    %60 = addi %25, %59 {handshake.bb = 2 : ui32, handshake.name = "addi0"} : <i9>
    %addressResult_4, %dataResult_5 = load[%60] %outputs_0#0 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i9>, <f32>, <i9>, <f32>
    %addressResult_6, %dataResult_7 = load[%27] %outputs {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load2"} : <i5>, <f32>, <i5>, <f32>
    %61 = mulf %dataResult_5, %dataResult_7 {fastmath = #arith.fastmath<none>, fpu_impl = #handshake<fpu_impl flopoco>, handshake.bb = 2 : ui32, handshake.name = "mulf0", internal_delay = "2_875333"} : <f32>
    %62 = buffer %28, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer6"} : <f32>
    %63 = addf %62, %61 {fastmath = #arith.fastmath<none>, fpu_impl = #handshake<fpu_impl flopoco>, handshake.bb = 2 : ui32, handshake.name = "addf0", internal_delay = "2_922000"} : <f32>
    %64 = addi %26, %43 {handshake.bb = 2 : ui32, handshake.name = "addi2"} : <i7>
    %65 = buffer %64, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer18"} : <i7>
    %66:2 = fork [2] %65 {handshake.bb = 2 : ui32, handshake.name = "fork7"} : <i7>
    %67 = trunci %68 {handshake.bb = 2 : ui32, handshake.name = "trunci4"} : <i7> to <i6>
    %68 = buffer %66#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer11"} : <i7>
    %69 = cmpi ult, %66#1, %40 {handshake.bb = 2 : ui32, handshake.name = "cmpi0"} : <i7>
    %70 = buffer %69, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer19"} : <i1>
    %71:5 = fork [5] %70 {handshake.bb = 2 : ui32, handshake.name = "fork8"} : <i1>
    %trueResult, %falseResult = cond_br %72, %67 {handshake.bb = 2 : ui32, handshake.name = "cond_br3"} : <i1>, <i6>
    %72 = buffer %71#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer13"} : <i1>
    sink %falseResult {handshake.name = "sink0"} : <i6>
    %trueResult_8, %falseResult_9 = cond_br %73, %63 {handshake.bb = 2 : ui32, handshake.name = "cond_br4"} : <i1>, <f32>
    %73 = buffer %71#2, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer14"} : <i1>
    %trueResult_10, %falseResult_11 = cond_br %71#1, %32#0 {handshake.bb = 2 : ui32, handshake.name = "cond_br5"} : <i1>, <i6>
    %74 = buffer %36#1, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer12"} : <>
    %trueResult_12, %falseResult_13 = cond_br %71#3, %74 {handshake.bb = 2 : ui32, handshake.name = "cond_br6"} : <i1>, <>
    %trueResult_14, %falseResult_15 = cond_br %71#4, %37 {handshake.bb = 2 : ui32, handshake.name = "cond_br7"} : <i1>, <i1>
    sink %trueResult_14 {handshake.name = "sink1"} : <i1>
    %75 = extsi %falseResult_15 {handshake.bb = 2 : ui32, handshake.name = "extsi13"} : <i1> to <i6>
    %76 = mux %93#1 [%75, %trueResult_24] {handshake.bb = 3 : ui32, handshake.name = "mux4"} : <i1>, [<i6>, <i6>] to <i6>
    %77 = buffer %76, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer20"} : <i6>
    %78:4 = fork [4] %77 {handshake.bb = 3 : ui32, handshake.name = "fork9"} : <i6>
    %79 = extsi %78#0 {handshake.bb = 3 : ui32, handshake.name = "extsi21"} : <i6> to <i9>
    %80 = extsi %78#3 {handshake.bb = 3 : ui32, handshake.name = "extsi22"} : <i6> to <i7>
    %81 = trunci %78#1 {handshake.bb = 3 : ui32, handshake.name = "trunci5"} : <i6> to <i5>
    %82 = trunci %78#2 {handshake.bb = 3 : ui32, handshake.name = "trunci6"} : <i6> to <i5>
    %83 = mux %93#0 [%falseResult_11, %trueResult_26] {handshake.bb = 3 : ui32, handshake.name = "mux5"} : <i1>, [<i6>, <i6>] to <i6>
    %84 = buffer %83, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer23"} : <i6>
    %85 = buffer %84, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer24"} : <i6>
    %86:2 = fork [2] %85 {handshake.bb = 3 : ui32, handshake.name = "fork10"} : <i6>
    %87 = extsi %86#1 {handshake.bb = 3 : ui32, handshake.name = "extsi23"} : <i6> to <i32>
    %88:2 = fork [2] %87 {handshake.bb = 3 : ui32, handshake.name = "fork11"} : <i32>
    %89 = buffer %trueResult_28, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer46"} : <f32>
    %90 = mux %93#2 [%falseResult_9, %89] {handshake.bb = 3 : ui32, handshake.name = "mux6"} : <i1>, [<f32>, <f32>] to <f32>
    %91 = buffer %90, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer25"} : <f32>
    %92:2 = fork [2] %91 {handshake.bb = 3 : ui32, handshake.name = "fork12"} : <f32>
    %result_16, %index_17 = control_merge [%falseResult_13, %trueResult_30]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>, <>] to <>, <i1>
    %93:3 = fork [3] %index_17 {handshake.bb = 3 : ui32, handshake.name = "fork13"} : <i1>
    %94:2 = fork [2] %result_16 {handshake.bb = 3 : ui32, handshake.name = "fork14"} : <>
    %95 = source {handshake.bb = 3 : ui32, handshake.name = "source4"} : <>
    %96 = constant %95 {handshake.bb = 3 : ui32, handshake.name = "constant20", value = 20 : i6} : <>, <i6>
    %97 = extsi %96 {handshake.bb = 3 : ui32, handshake.name = "extsi24"} : <i6> to <i7>
    %98 = source {handshake.bb = 3 : ui32, handshake.name = "source5"} : <>
    %99 = constant %98 {handshake.bb = 3 : ui32, handshake.name = "constant21", value = 1 : i2} : <>, <i2>
    %100 = extsi %99 {handshake.bb = 3 : ui32, handshake.name = "extsi25"} : <i2> to <i7>
    %101 = source {handshake.bb = 3 : ui32, handshake.name = "source6"} : <>
    %102 = constant %101 {handshake.bb = 3 : ui32, handshake.name = "constant22", value = 4 : i4} : <>, <i4>
    %103 = extsi %102 {handshake.bb = 3 : ui32, handshake.name = "extsi9"} : <i4> to <i32>
    %104 = source {handshake.bb = 3 : ui32, handshake.name = "source7"} : <>
    %105 = constant %104 {handshake.bb = 3 : ui32, handshake.name = "constant23", value = 2 : i3} : <>, <i3>
    %106 = extsi %105 {handshake.bb = 3 : ui32, handshake.name = "extsi10"} : <i3> to <i32>
    %107 = buffer %82, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer22"} : <i5>
    %addressResult_18, %dataResult_19 = load[%107] %2#0 {handshake.bb = 3 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["store0", 1], ["store0", 3]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 0>, handshake.name = "load3"} : <i5>, <f32>, <i5>, <f32>
    %108 = shli %109, %106 {handshake.bb = 3 : ui32, handshake.name = "shli2"} : <i32>
    %109 = buffer %88#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer27"} : <i32>
    %110 = buffer %108, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer34"} : <i32>
    %111 = trunci %110 {handshake.bb = 3 : ui32, handshake.name = "trunci7"} : <i32> to <i9>
    %112 = shli %113, %103 {handshake.bb = 3 : ui32, handshake.name = "shli3"} : <i32>
    %113 = buffer %88#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer28"} : <i32>
    %114 = buffer %112, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer38"} : <i32>
    %115 = trunci %114 {handshake.bb = 3 : ui32, handshake.name = "trunci8"} : <i32> to <i9>
    %116 = addi %111, %115 {handshake.bb = 3 : ui32, handshake.name = "addi6"} : <i9>
    %117 = buffer %116, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer39"} : <i9>
    %118 = addi %79, %117 {handshake.bb = 3 : ui32, handshake.name = "addi1"} : <i9>
    %addressResult_20, %dataResult_21 = load[%118] %outputs_0#1 {handshake.bb = 3 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load4"} : <i9>, <f32>, <i9>, <f32>
    %119 = mulf %dataResult_21, %120 {fastmath = #arith.fastmath<none>, fpu_impl = #handshake<fpu_impl flopoco>, handshake.bb = 3 : ui32, handshake.name = "mulf1", internal_delay = "2_875333"} : <f32>
    %120 = buffer %92#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer29"} : <f32>
    %121 = addf %dataResult_19, %119 {fastmath = #arith.fastmath<none>, fpu_impl = #handshake<fpu_impl flopoco>, handshake.bb = 3 : ui32, handshake.name = "addf1", internal_delay = "2_922000"} : <f32>
    %122 = buffer %81, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer21"} : <i5>
    %123 = buffer %121, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer41"} : <f32>
    %addressResult_22, %dataResult_23 = store[%122] %123 {handshake.bb = 3 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["load3", 1], ["store0", 1]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 0>, handshake.name = "store0"} : <i5>, <f32>, <i5>, <f32>
    %124 = addi %80, %100 {handshake.bb = 3 : ui32, handshake.name = "addi3"} : <i7>
    %125 = buffer %124, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer44"} : <i7>
    %126:2 = fork [2] %125 {handshake.bb = 3 : ui32, handshake.name = "fork15"} : <i7>
    %127 = trunci %128 {handshake.bb = 3 : ui32, handshake.name = "trunci9"} : <i7> to <i6>
    %128 = buffer %126#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer30"} : <i7>
    %129 = cmpi ult, %126#1, %97 {handshake.bb = 3 : ui32, handshake.name = "cmpi1"} : <i7>
    %130 = buffer %129, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer45"} : <i1>
    %131:4 = fork [4] %130 {handshake.bb = 3 : ui32, handshake.name = "fork16"} : <i1>
    %trueResult_24, %falseResult_25 = cond_br %132, %127 {handshake.bb = 3 : ui32, handshake.name = "cond_br8"} : <i1>, <i6>
    %132 = buffer %131#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer32"} : <i1>
    sink %falseResult_25 {handshake.name = "sink2"} : <i6>
    %trueResult_26, %falseResult_27 = cond_br %133, %86#0 {handshake.bb = 3 : ui32, handshake.name = "cond_br9"} : <i1>, <i6>
    %133 = buffer %131#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer33"} : <i1>
    %trueResult_28, %falseResult_29 = cond_br %134, %135 {handshake.bb = 3 : ui32, handshake.name = "cond_br10"} : <i1>, <f32>
    %134 = buffer %131#2, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer35"} : <i1>
    %135 = buffer %92#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer36"} : <f32>
    %136 = buffer %94#1, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer26"} : <>
    %137 = buffer %136, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer31"} : <>
    %trueResult_30, %falseResult_31 = cond_br %138, %137 {handshake.bb = 3 : ui32, handshake.bufProps = #handshake<bufProps{"1": [1,inf], [0,inf], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "cond_br11"} : <i1>, <>
    %138 = buffer %131#3, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer37"} : <i1>
    %139 = merge %falseResult_27 {handshake.bb = 4 : ui32, handshake.name = "merge0"} : <i6>
    %140:2 = fork [2] %139 {handshake.bb = 4 : ui32, handshake.name = "fork17"} : <i6>
    %141 = extsi %140#1 {handshake.bb = 4 : ui32, handshake.name = "extsi26"} : <i6> to <i7>
    %142 = trunci %140#0 {handshake.bb = 4 : ui32, handshake.name = "trunci10"} : <i6> to <i5>
    %143 = merge %falseResult_29 {handshake.bb = 4 : ui32, handshake.name = "merge1"} : <f32>
    %result_32, %index_33 = control_merge [%falseResult_31]  {handshake.bb = 4 : ui32, handshake.name = "control_merge3"} : [<>] to <>, <i1>
    sink %index_33 {handshake.name = "sink3"} : <i1>
    %144:2 = lazy_fork [2] %result_32 {handshake.bb = 4 : ui32, handshake.name = "lazy_fork1"} : <>
    %145 = source {handshake.bb = 4 : ui32, handshake.name = "source8"} : <>
    %146 = constant %145 {handshake.bb = 4 : ui32, handshake.name = "constant24", value = 20 : i6} : <>, <i6>
    %147 = extsi %146 {handshake.bb = 4 : ui32, handshake.name = "extsi27"} : <i6> to <i7>
    %148 = source {handshake.bb = 4 : ui32, handshake.name = "source9"} : <>
    %149 = constant %148 {handshake.bb = 4 : ui32, handshake.name = "constant25", value = 1 : i2} : <>, <i2>
    %150 = extsi %149 {handshake.bb = 4 : ui32, handshake.name = "extsi28"} : <i2> to <i7>
    %151 = buffer %142, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer47"} : <i5>
    %152 = buffer %143, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer48"} : <f32>
    %addressResult_34, %dataResult_35 = store[%151] %152 {handshake.bb = 4 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<LSQ: 1>, handshake.name = "store1"} : <i5>, <f32>, <i5>, <f32>
    %153 = addi %141, %150 {handshake.bb = 4 : ui32, handshake.name = "addi4"} : <i7>
    %154 = buffer %153, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer50"} : <i7>
    %155:2 = fork [2] %154 {handshake.bb = 4 : ui32, handshake.name = "fork18"} : <i7>
    %156 = trunci %157 {handshake.bb = 4 : ui32, handshake.name = "trunci11"} : <i7> to <i6>
    %157 = buffer %155#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer40"} : <i7>
    %158 = cmpi ult, %155#1, %147 {handshake.bb = 4 : ui32, handshake.name = "cmpi2"} : <i7>
    %159 = buffer %158, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer51"} : <i1>
    %160:2 = fork [2] %159 {handshake.bb = 4 : ui32, handshake.name = "fork19"} : <i1>
    %trueResult_36, %falseResult_37 = cond_br %161, %156 {handshake.bb = 4 : ui32, handshake.name = "cond_br12"} : <i1>, <i6>
    %161 = buffer %160#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer42"} : <i1>
    sink %falseResult_37 {handshake.name = "sink4"} : <i6>
    %162 = buffer %144#1, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer49"} : <>
    %trueResult_38, %falseResult_39 = cond_br %163, %162 {handshake.bb = 4 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,inf], [1,inf], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "cond_br13"} : <i1>, <>
    %163 = buffer %160#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer43"} : <i1>
    %result_40, %index_41 = control_merge [%falseResult_39]  {handshake.bb = 5 : ui32, handshake.name = "control_merge4"} : [<>] to <>, <i1>
    sink %index_41 {handshake.name = "sink5"} : <i1>
    %164:4 = fork [4] %result_40 {handshake.bb = 5 : ui32, handshake.name = "fork20"} : <>
    end {handshake.bb = 5 : ui32, handshake.name = "end0"} %memEnd_1, %memEnd, %2#1, %1#1, %0#1 : <>, <>, <>, <>, <>
  }
}

