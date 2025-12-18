module {
  handshake.func @atax(%arg0: memref<400xf32>, %arg1: memref<20xf32>, %arg2: memref<20xf32>, %arg3: memref<20xf32>, %arg4: !handshake.control<>, %arg5: !handshake.control<>, %arg6: !handshake.control<>, %arg7: !handshake.control<>, %arg8: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["A", "x", "y", "tmp", "A_start", "x_start", "y_start", "tmp_start", "start"], resNames = ["A_end", "x_end", "y_end", "tmp_end", "end"]} {
    %0:3 = fork [3] %arg8 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork0"} : <>
    %1:2 = lsq[%arg3 : memref<20xf32>] (%arg7, %9#0, %addressResult, %136#0, %addressResult_32, %dataResult_33, %156#3)  {groupSizes = [1 : i32, 1 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "5": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "6": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "lsq2"} : (!handshake.control<>, !handshake.control<>, !handshake.channel<i5>, !handshake.control<>, !handshake.channel<i5>, !handshake.channel<f32>, !handshake.control<>) -> (!handshake.channel<f32>, !handshake.control<>)
    %2:2 = lsq[%arg2 : memref<20xf32>] (%arg6, %88#0, %addressResult_18, %addressResult_22, %dataResult_23, %156#2)  {groupSizes = [2 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "5": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "lsq3"} : (!handshake.control<>, !handshake.control<>, !handshake.channel<i5>, !handshake.channel<i5>, !handshake.channel<f32>, !handshake.control<>) -> (!handshake.channel<f32>, !handshake.control<>)
    %outputs, %memEnd = mem_controller[%arg1 : memref<20xf32>] %arg5 (%addressResult_6) %156#1 {connectedBlocks = [2 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller2"} :    (!handshake.channel<i5>) -> !handshake.channel<f32>
    %outputs_0:2, %memEnd_1 = mem_controller[%arg0 : memref<400xf32>] %arg4 (%addressResult_4, %addressResult_20) %156#0 {connectedBlocks = [2 : i32, 3 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller3"} :    (!handshake.channel<i9>, !handshake.channel<i9>) -> (!handshake.channel<f32>, !handshake.channel<f32>)
    %3 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant0", value = false} : <>, <i1>
    %4 = extsi %3 {handshake.bb = 0 : ui32, handshake.name = "extsi15"} : <i1> to <i6>
    %5 = mux %index [%4, %trueResult_34] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i6>, <i6>] to <i6>
    %6 = buffer %5, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer0"} : <i6>
    %7:2 = fork [2] %6 {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <i6>
    %8 = trunci %7#0 {handshake.bb = 1 : ui32, handshake.name = "trunci0"} : <i6> to <i5>
    %result, %index = control_merge [%0#2, %trueResult_36]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %9:3 = lazy_fork [3] %result {handshake.bb = 1 : ui32, handshake.name = "lazy_fork0"} : <>
    %10 = buffer %9#2, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer4"} : <>
    %11 = constant %10 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"0": [1,inf], [0,inf], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "constant1", value = false} : <>, <i1>
    %12 = buffer %8, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer2"} : <i5>
    %addressResult, %dataResult = load[%12] %1#0 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["store1", 2]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 0>, handshake.name = "load0"} : <i5>, <f32>, <i5>, <f32>
    %13 = extsi %11 {handshake.bb = 1 : ui32, handshake.name = "extsi14"} : <i1> to <i6>
    %14 = buffer %7#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer1"} : <i6>
    %15 = buffer %9#1, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer3"} : <>
    %16 = mux %29#1 [%13, %trueResult] {handshake.bb = 2 : ui32, handshake.name = "mux1"} : <i1>, [<i6>, <i6>] to <i6>
    %17 = buffer %16, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer5"} : <i6>
    %18:3 = fork [3] %17 {handshake.bb = 2 : ui32, handshake.name = "fork2"} : <i6>
    %19 = extsi %18#0 {handshake.bb = 2 : ui32, handshake.name = "extsi16"} : <i6> to <i9>
    %20 = extsi %18#2 {handshake.bb = 2 : ui32, handshake.name = "extsi17"} : <i6> to <i7>
    %21 = trunci %18#1 {handshake.bb = 2 : ui32, handshake.name = "trunci1"} : <i6> to <i5>
    %22 = mux %29#2 [%dataResult, %trueResult_8] {handshake.bb = 2 : ui32, handshake.name = "mux2"} : <i1>, [<f32>, <f32>] to <f32>
    %23 = mux %29#0 [%14, %trueResult_10] {handshake.bb = 2 : ui32, handshake.name = "mux3"} : <i1>, [<i6>, <i6>] to <i6>
    %24 = buffer %23, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer7"} : <i6>
    %25 = buffer %24, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer8"} : <i6>
    %26:2 = fork [2] %25 {handshake.bb = 2 : ui32, handshake.name = "fork3"} : <i6>
    %27 = extsi %26#1 {handshake.bb = 2 : ui32, handshake.name = "extsi18"} : <i6> to <i32>
    %28:2 = fork [2] %27 {handshake.bb = 2 : ui32, handshake.name = "fork4"} : <i32>
    %result_2, %index_3 = control_merge [%15, %trueResult_12]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>, <>] to <>, <i1>
    %29:3 = fork [3] %index_3 {handshake.bb = 2 : ui32, handshake.name = "fork5"} : <i1>
    %30:2 = fork [2] %result_2 {handshake.bb = 2 : ui32, handshake.name = "fork6"} : <>
    %31 = constant %30#0 {handshake.bb = 2 : ui32, handshake.name = "constant2", value = false} : <>, <i1>
    %32 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %33 = constant %32 {handshake.bb = 2 : ui32, handshake.name = "constant3", value = 20 : i6} : <>, <i6>
    %34 = extsi %33 {handshake.bb = 2 : ui32, handshake.name = "extsi19"} : <i6> to <i7>
    %35 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %36 = constant %35 {handshake.bb = 2 : ui32, handshake.name = "constant17", value = 1 : i2} : <>, <i2>
    %37 = extsi %36 {handshake.bb = 2 : ui32, handshake.name = "extsi20"} : <i2> to <i7>
    %38 = source {handshake.bb = 2 : ui32, handshake.name = "source2"} : <>
    %39 = constant %38 {handshake.bb = 2 : ui32, handshake.name = "constant18", value = 4 : i4} : <>, <i4>
    %40 = extsi %39 {handshake.bb = 2 : ui32, handshake.name = "extsi5"} : <i4> to <i32>
    %41 = source {handshake.bb = 2 : ui32, handshake.name = "source3"} : <>
    %42 = constant %41 {handshake.bb = 2 : ui32, handshake.name = "constant19", value = 2 : i3} : <>, <i3>
    %43 = extsi %42 {handshake.bb = 2 : ui32, handshake.name = "extsi6"} : <i3> to <i32>
    %44 = shli %45, %43 {handshake.bb = 2 : ui32, handshake.name = "shli0"} : <i32>
    %45 = buffer %28#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer9"} : <i32>
    %46 = buffer %44, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer15"} : <i32>
    %47 = trunci %46 {handshake.bb = 2 : ui32, handshake.name = "trunci2"} : <i32> to <i9>
    %48 = shli %49, %40 {handshake.bb = 2 : ui32, handshake.name = "shli1"} : <i32>
    %49 = buffer %28#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer10"} : <i32>
    %50 = buffer %48, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer16"} : <i32>
    %51 = trunci %50 {handshake.bb = 2 : ui32, handshake.name = "trunci3"} : <i32> to <i9>
    %52 = addi %47, %51 {handshake.bb = 2 : ui32, handshake.name = "addi5"} : <i9>
    %53 = buffer %52, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer17"} : <i9>
    %54 = addi %19, %53 {handshake.bb = 2 : ui32, handshake.name = "addi0"} : <i9>
    %addressResult_4, %dataResult_5 = load[%54] %outputs_0#0 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i9>, <f32>, <i9>, <f32>
    %addressResult_6, %dataResult_7 = load[%21] %outputs {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load2"} : <i5>, <f32>, <i5>, <f32>
    %55 = mulf %dataResult_5, %dataResult_7 {fastmath = #arith.fastmath<none>, fpu_impl = #handshake<fpu_impl flopoco>, handshake.bb = 2 : ui32, handshake.name = "mulf0", internal_delay = "2_875333"} : <f32>
    %56 = buffer %22, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer6"} : <f32>
    %57 = addf %56, %55 {fastmath = #arith.fastmath<none>, fpu_impl = #handshake<fpu_impl flopoco>, handshake.bb = 2 : ui32, handshake.name = "addf0", internal_delay = "2_922000"} : <f32>
    %58 = addi %20, %37 {handshake.bb = 2 : ui32, handshake.name = "addi2"} : <i7>
    %59 = buffer %58, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer18"} : <i7>
    %60:2 = fork [2] %59 {handshake.bb = 2 : ui32, handshake.name = "fork7"} : <i7>
    %61 = trunci %62 {handshake.bb = 2 : ui32, handshake.name = "trunci4"} : <i7> to <i6>
    %62 = buffer %60#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer11"} : <i7>
    %63 = cmpi ult, %60#1, %34 {handshake.bb = 2 : ui32, handshake.name = "cmpi0"} : <i7>
    %64 = buffer %63, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer19"} : <i1>
    %65:5 = fork [5] %64 {handshake.bb = 2 : ui32, handshake.name = "fork8"} : <i1>
    %trueResult, %falseResult = cond_br %66, %61 {handshake.bb = 2 : ui32, handshake.name = "cond_br3"} : <i1>, <i6>
    %66 = buffer %65#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer13"} : <i1>
    sink %falseResult {handshake.name = "sink0"} : <i6>
    %trueResult_8, %falseResult_9 = cond_br %67, %57 {handshake.bb = 2 : ui32, handshake.name = "cond_br4"} : <i1>, <f32>
    %67 = buffer %65#2, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer14"} : <i1>
    %trueResult_10, %falseResult_11 = cond_br %65#1, %26#0 {handshake.bb = 2 : ui32, handshake.name = "cond_br5"} : <i1>, <i6>
    %68 = buffer %30#1, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer12"} : <>
    %trueResult_12, %falseResult_13 = cond_br %65#3, %68 {handshake.bb = 2 : ui32, handshake.name = "cond_br6"} : <i1>, <>
    %trueResult_14, %falseResult_15 = cond_br %65#4, %31 {handshake.bb = 2 : ui32, handshake.name = "cond_br7"} : <i1>, <i1>
    sink %trueResult_14 {handshake.name = "sink1"} : <i1>
    %69 = extsi %falseResult_15 {handshake.bb = 2 : ui32, handshake.name = "extsi13"} : <i1> to <i6>
    %70 = mux %87#1 [%69, %trueResult_24] {handshake.bb = 3 : ui32, handshake.name = "mux4"} : <i1>, [<i6>, <i6>] to <i6>
    %71 = buffer %70, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer20"} : <i6>
    %72:4 = fork [4] %71 {handshake.bb = 3 : ui32, handshake.name = "fork9"} : <i6>
    %73 = extsi %72#0 {handshake.bb = 3 : ui32, handshake.name = "extsi21"} : <i6> to <i9>
    %74 = extsi %72#3 {handshake.bb = 3 : ui32, handshake.name = "extsi22"} : <i6> to <i7>
    %75 = trunci %72#1 {handshake.bb = 3 : ui32, handshake.name = "trunci5"} : <i6> to <i5>
    %76 = trunci %72#2 {handshake.bb = 3 : ui32, handshake.name = "trunci6"} : <i6> to <i5>
    %77 = mux %87#0 [%falseResult_11, %trueResult_26] {handshake.bb = 3 : ui32, handshake.name = "mux5"} : <i1>, [<i6>, <i6>] to <i6>
    %78 = buffer %77, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer23"} : <i6>
    %79 = buffer %78, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer24"} : <i6>
    %80:2 = fork [2] %79 {handshake.bb = 3 : ui32, handshake.name = "fork10"} : <i6>
    %81 = extsi %80#1 {handshake.bb = 3 : ui32, handshake.name = "extsi23"} : <i6> to <i32>
    %82:2 = fork [2] %81 {handshake.bb = 3 : ui32, handshake.name = "fork11"} : <i32>
    %83 = buffer %trueResult_28, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer46"} : <f32>
    %84 = mux %87#2 [%falseResult_9, %83] {handshake.bb = 3 : ui32, handshake.name = "mux6"} : <i1>, [<f32>, <f32>] to <f32>
    %85 = buffer %84, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer25"} : <f32>
    %86:2 = fork [2] %85 {handshake.bb = 3 : ui32, handshake.name = "fork12"} : <f32>
    %result_16, %index_17 = control_merge [%falseResult_13, %trueResult_30]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>, <>] to <>, <i1>
    %87:3 = fork [3] %index_17 {handshake.bb = 3 : ui32, handshake.name = "fork13"} : <i1>
    %88:2 = fork [2] %result_16 {handshake.bb = 3 : ui32, handshake.name = "fork14"} : <>
    %89 = source {handshake.bb = 3 : ui32, handshake.name = "source4"} : <>
    %90 = constant %89 {handshake.bb = 3 : ui32, handshake.name = "constant20", value = 20 : i6} : <>, <i6>
    %91 = extsi %90 {handshake.bb = 3 : ui32, handshake.name = "extsi24"} : <i6> to <i7>
    %92 = source {handshake.bb = 3 : ui32, handshake.name = "source5"} : <>
    %93 = constant %92 {handshake.bb = 3 : ui32, handshake.name = "constant21", value = 1 : i2} : <>, <i2>
    %94 = extsi %93 {handshake.bb = 3 : ui32, handshake.name = "extsi25"} : <i2> to <i7>
    %95 = source {handshake.bb = 3 : ui32, handshake.name = "source6"} : <>
    %96 = constant %95 {handshake.bb = 3 : ui32, handshake.name = "constant22", value = 4 : i4} : <>, <i4>
    %97 = extsi %96 {handshake.bb = 3 : ui32, handshake.name = "extsi9"} : <i4> to <i32>
    %98 = source {handshake.bb = 3 : ui32, handshake.name = "source7"} : <>
    %99 = constant %98 {handshake.bb = 3 : ui32, handshake.name = "constant23", value = 2 : i3} : <>, <i3>
    %100 = extsi %99 {handshake.bb = 3 : ui32, handshake.name = "extsi10"} : <i3> to <i32>
    %101 = buffer %76, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer22"} : <i5>
    %addressResult_18, %dataResult_19 = load[%101] %2#0 {handshake.bb = 3 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["store0", 1], ["store0", 3]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 0>, handshake.name = "load3"} : <i5>, <f32>, <i5>, <f32>
    %102 = shli %103, %100 {handshake.bb = 3 : ui32, handshake.name = "shli2"} : <i32>
    %103 = buffer %82#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer27"} : <i32>
    %104 = buffer %102, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer34"} : <i32>
    %105 = trunci %104 {handshake.bb = 3 : ui32, handshake.name = "trunci7"} : <i32> to <i9>
    %106 = shli %107, %97 {handshake.bb = 3 : ui32, handshake.name = "shli3"} : <i32>
    %107 = buffer %82#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer28"} : <i32>
    %108 = buffer %106, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer38"} : <i32>
    %109 = trunci %108 {handshake.bb = 3 : ui32, handshake.name = "trunci8"} : <i32> to <i9>
    %110 = addi %105, %109 {handshake.bb = 3 : ui32, handshake.name = "addi6"} : <i9>
    %111 = buffer %110, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer39"} : <i9>
    %112 = addi %73, %111 {handshake.bb = 3 : ui32, handshake.name = "addi1"} : <i9>
    %addressResult_20, %dataResult_21 = load[%112] %outputs_0#1 {handshake.bb = 3 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load4"} : <i9>, <f32>, <i9>, <f32>
    %113 = mulf %dataResult_21, %114 {fastmath = #arith.fastmath<none>, fpu_impl = #handshake<fpu_impl flopoco>, handshake.bb = 3 : ui32, handshake.name = "mulf1", internal_delay = "2_875333"} : <f32>
    %114 = buffer %86#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer29"} : <f32>
    %115 = addf %dataResult_19, %113 {fastmath = #arith.fastmath<none>, fpu_impl = #handshake<fpu_impl flopoco>, handshake.bb = 3 : ui32, handshake.name = "addf1", internal_delay = "2_922000"} : <f32>
    %116 = buffer %75, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer21"} : <i5>
    %117 = buffer %115, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer41"} : <f32>
    %addressResult_22, %dataResult_23 = store[%116] %117 {handshake.bb = 3 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["load3", 1], ["store0", 1]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 0>, handshake.name = "store0"} : <i5>, <f32>, <i5>, <f32>
    %118 = addi %74, %94 {handshake.bb = 3 : ui32, handshake.name = "addi3"} : <i7>
    %119 = buffer %118, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer44"} : <i7>
    %120:2 = fork [2] %119 {handshake.bb = 3 : ui32, handshake.name = "fork15"} : <i7>
    %121 = trunci %122 {handshake.bb = 3 : ui32, handshake.name = "trunci9"} : <i7> to <i6>
    %122 = buffer %120#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer30"} : <i7>
    %123 = cmpi ult, %120#1, %91 {handshake.bb = 3 : ui32, handshake.name = "cmpi1"} : <i7>
    %124 = buffer %123, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer45"} : <i1>
    %125:4 = fork [4] %124 {handshake.bb = 3 : ui32, handshake.name = "fork16"} : <i1>
    %trueResult_24, %falseResult_25 = cond_br %126, %121 {handshake.bb = 3 : ui32, handshake.name = "cond_br8"} : <i1>, <i6>
    %126 = buffer %125#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer32"} : <i1>
    sink %falseResult_25 {handshake.name = "sink2"} : <i6>
    %trueResult_26, %falseResult_27 = cond_br %127, %80#0 {handshake.bb = 3 : ui32, handshake.name = "cond_br9"} : <i1>, <i6>
    %127 = buffer %125#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer33"} : <i1>
    %trueResult_28, %falseResult_29 = cond_br %128, %129 {handshake.bb = 3 : ui32, handshake.name = "cond_br10"} : <i1>, <f32>
    %128 = buffer %125#2, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer35"} : <i1>
    %129 = buffer %86#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer36"} : <f32>
    %130 = buffer %88#1, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer26"} : <>
    %131 = buffer %130, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer31"} : <>
    %trueResult_30, %falseResult_31 = cond_br %132, %131 {handshake.bb = 3 : ui32, handshake.bufProps = #handshake<bufProps{"1": [1,inf], [0,inf], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "cond_br11"} : <i1>, <>
    %132 = buffer %125#3, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer37"} : <i1>
    %133:2 = fork [2] %falseResult_27 {handshake.bb = 4 : ui32, handshake.name = "fork17"} : <i6>
    %134 = extsi %133#1 {handshake.bb = 4 : ui32, handshake.name = "extsi26"} : <i6> to <i7>
    %135 = trunci %133#0 {handshake.bb = 4 : ui32, handshake.name = "trunci10"} : <i6> to <i5>
    %136:2 = lazy_fork [2] %falseResult_31 {handshake.bb = 4 : ui32, handshake.name = "lazy_fork1"} : <>
    %137 = source {handshake.bb = 4 : ui32, handshake.name = "source8"} : <>
    %138 = constant %137 {handshake.bb = 4 : ui32, handshake.name = "constant24", value = 20 : i6} : <>, <i6>
    %139 = extsi %138 {handshake.bb = 4 : ui32, handshake.name = "extsi27"} : <i6> to <i7>
    %140 = source {handshake.bb = 4 : ui32, handshake.name = "source9"} : <>
    %141 = constant %140 {handshake.bb = 4 : ui32, handshake.name = "constant25", value = 1 : i2} : <>, <i2>
    %142 = extsi %141 {handshake.bb = 4 : ui32, handshake.name = "extsi28"} : <i2> to <i7>
    %143 = buffer %135, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer47"} : <i5>
    %144 = buffer %falseResult_29, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer48"} : <f32>
    %addressResult_32, %dataResult_33 = store[%143] %144 {handshake.bb = 4 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<LSQ: 1>, handshake.name = "store1"} : <i5>, <f32>, <i5>, <f32>
    %145 = addi %134, %142 {handshake.bb = 4 : ui32, handshake.name = "addi4"} : <i7>
    %146 = buffer %145, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer50"} : <i7>
    %147:2 = fork [2] %146 {handshake.bb = 4 : ui32, handshake.name = "fork18"} : <i7>
    %148 = trunci %149 {handshake.bb = 4 : ui32, handshake.name = "trunci11"} : <i7> to <i6>
    %149 = buffer %147#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer40"} : <i7>
    %150 = cmpi ult, %147#1, %139 {handshake.bb = 4 : ui32, handshake.name = "cmpi2"} : <i7>
    %151 = buffer %150, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer51"} : <i1>
    %152:2 = fork [2] %151 {handshake.bb = 4 : ui32, handshake.name = "fork19"} : <i1>
    %trueResult_34, %falseResult_35 = cond_br %153, %148 {handshake.bb = 4 : ui32, handshake.name = "cond_br12"} : <i1>, <i6>
    %153 = buffer %152#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer42"} : <i1>
    sink %falseResult_35 {handshake.name = "sink4"} : <i6>
    %154 = buffer %136#1, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer49"} : <>
    %trueResult_36, %falseResult_37 = cond_br %155, %154 {handshake.bb = 4 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,inf], [1,inf], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "cond_br13"} : <i1>, <>
    %155 = buffer %152#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer43"} : <i1>
    %156:4 = fork [4] %falseResult_37 {handshake.bb = 5 : ui32, handshake.name = "fork20"} : <>
    end {handshake.bb = 5 : ui32, handshake.name = "end0"} %memEnd_1, %memEnd, %2#1, %1#1, %0#1 : <>, <>, <>, <>, <>
  }
}

