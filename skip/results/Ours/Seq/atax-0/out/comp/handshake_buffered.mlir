module {
  handshake.func @atax(%arg0: memref<400xf32>, %arg1: memref<20xf32>, %arg2: memref<20xf32>, %arg3: memref<20xf32>, %arg4: !handshake.control<>, %arg5: !handshake.control<>, %arg6: !handshake.control<>, %arg7: !handshake.control<>, %arg8: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["A", "x", "y", "tmp", "A_start", "x_start", "y_start", "tmp_start", "start"], cfg.edges = "[0,1][2,2,3,cmpi0][4,1,5,cmpi2][1,2][3,3,4,cmpi1]", resNames = ["A_end", "x_end", "y_end", "tmp_end", "end"]} {
    %0:4 = fork [4] %arg8 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork0"} : <>
    %outputs:2, %memEnd = mem_controller[%arg3 : memref<20xf32>] %arg7 (%addressResult, %148, %addressResult_42, %dataResult_43) %164#3 {connectedBlocks = [1 : i32, 4 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "5": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller2"} :    (!handshake.channel<i5>, !handshake.channel<i32>, !handshake.channel<i5>, !handshake.channel<f32>) -> (!handshake.channel<f32>, !handshake.control<>)
    %outputs_0:2, %memEnd_1 = mem_controller[%arg2 : memref<20xf32>] %arg6 (%101, %addressResult_24, %addressResult_28, %dataResult_29) %164#2 {connectedBlocks = [3 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "5": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller3"} :    (!handshake.channel<i32>, !handshake.channel<i5>, !handshake.channel<i5>, !handshake.channel<f32>) -> (!handshake.channel<f32>, !handshake.control<>)
    %outputs_2, %memEnd_3 = mem_controller[%arg1 : memref<20xf32>] %arg5 (%addressResult_10) %164#1 {connectedBlocks = [2 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller4"} :    (!handshake.channel<i5>) -> !handshake.channel<f32>
    %outputs_4:2, %memEnd_5 = mem_controller[%arg0 : memref<400xf32>] %arg4 (%addressResult_8, %addressResult_26) %164#0 {connectedBlocks = [2 : i32, 3 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller5"} :    (!handshake.channel<i9>, !handshake.channel<i9>) -> (!handshake.channel<f32>, !handshake.channel<f32>)
    %1 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant2", value = false} : <>, <i1>
    %2 = br %1 {handshake.bb = 0 : ui32, handshake.name = "br2"} : <i1>
    %3 = extsi %2 {handshake.bb = 0 : ui32, handshake.name = "extsi17"} : <i1> to <i6>
    %4 = br %0#3 {handshake.bb = 0 : ui32, handshake.name = "br3"} : <>
    %5 = mux %6 [%0#2, %trueResult_38] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux8"} : <i1>, [<>, <>] to <>
    %6 = init %163#2 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "init0"} : <i1>
    %7:2 = unbundle %15#0  {handshake.bb = 1 : ui32, handshake.name = "unbundle0"} : <f32> to _ 
    %8 = mux %index [%3, %trueResult_45] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i6>, <i6>] to <i6>
    %9 = buffer %8, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer3"} : <i6>
    %10:2 = fork [2] %9 {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <i6>
    %11 = trunci %10#0 {handshake.bb = 1 : ui32, handshake.name = "trunci0"} : <i6> to <i5>
    %result, %index = control_merge [%4, %trueResult_47]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %12:2 = fork [2] %result {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <>
    %13 = constant %12#0 {handshake.bb = 1 : ui32, handshake.name = "constant3", value = false} : <>, <i1>
    %14 = buffer %7#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer0"} : <>
    %addressResult, %dataResult = load[%11] %outputs#0 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["store1", 2, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i5>, <f32>, <i5>, <f32>
    %15:2 = fork [2] %dataResult {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <f32>
    %16 = br %13 {handshake.bb = 1 : ui32, handshake.name = "br4"} : <i1>
    %17 = extsi %16 {handshake.bb = 1 : ui32, handshake.name = "extsi16"} : <i1> to <i6>
    %18 = br %15#1 {handshake.bb = 1 : ui32, handshake.name = "br5"} : <f32>
    %19 = br %10#1 {handshake.bb = 1 : ui32, handshake.name = "br6"} : <i6>
    %20 = br %12#1 {handshake.bb = 1 : ui32, handshake.name = "br7"} : <>
    %21 = mux %34#1 [%17, %trueResult] {handshake.bb = 2 : ui32, handshake.name = "mux1"} : <i1>, [<i6>, <i6>] to <i6>
    %22 = buffer %21, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer4"} : <i6>
    %23:3 = fork [3] %22 {handshake.bb = 2 : ui32, handshake.name = "fork4"} : <i6>
    %24 = extsi %23#0 {handshake.bb = 2 : ui32, handshake.name = "extsi18"} : <i6> to <i9>
    %25 = extsi %23#2 {handshake.bb = 2 : ui32, handshake.name = "extsi19"} : <i6> to <i7>
    %26 = trunci %23#1 {handshake.bb = 2 : ui32, handshake.name = "trunci1"} : <i6> to <i5>
    %27 = mux %34#2 [%18, %trueResult_12] {handshake.bb = 2 : ui32, handshake.name = "mux2"} : <i1>, [<f32>, <f32>] to <f32>
    %28 = mux %34#0 [%19, %trueResult_14] {handshake.bb = 2 : ui32, handshake.name = "mux3"} : <i1>, [<i6>, <i6>] to <i6>
    %29 = buffer %28, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer6"} : <i6>
    %30 = buffer %29, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer7"} : <i6>
    %31:2 = fork [2] %30 {handshake.bb = 2 : ui32, handshake.name = "fork5"} : <i6>
    %32 = extsi %31#1 {handshake.bb = 2 : ui32, handshake.name = "extsi20"} : <i6> to <i32>
    %33:2 = fork [2] %32 {handshake.bb = 2 : ui32, handshake.name = "fork6"} : <i32>
    %result_6, %index_7 = control_merge [%20, %trueResult_16]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>, <>] to <>, <i1>
    %34:3 = fork [3] %index_7 {handshake.bb = 2 : ui32, handshake.name = "fork7"} : <i1>
    %35 = buffer %result_6, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer8"} : <>
    %36:2 = fork [2] %35 {handshake.bb = 2 : ui32, handshake.name = "fork8"} : <>
    %37 = constant %36#0 {handshake.bb = 2 : ui32, handshake.name = "constant17", value = false} : <>, <i1>
    %38 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %39 = constant %38 {handshake.bb = 2 : ui32, handshake.name = "constant18", value = 20 : i6} : <>, <i6>
    %40 = extsi %39 {handshake.bb = 2 : ui32, handshake.name = "extsi21"} : <i6> to <i7>
    %41 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %42 = constant %41 {handshake.bb = 2 : ui32, handshake.name = "constant19", value = 1 : i2} : <>, <i2>
    %43 = extsi %42 {handshake.bb = 2 : ui32, handshake.name = "extsi22"} : <i2> to <i7>
    %44 = source {handshake.bb = 2 : ui32, handshake.name = "source2"} : <>
    %45 = constant %44 {handshake.bb = 2 : ui32, handshake.name = "constant20", value = 4 : i4} : <>, <i4>
    %46 = extsi %45 {handshake.bb = 2 : ui32, handshake.name = "extsi5"} : <i4> to <i32>
    %47 = source {handshake.bb = 2 : ui32, handshake.name = "source3"} : <>
    %48 = constant %47 {handshake.bb = 2 : ui32, handshake.name = "constant21", value = 2 : i3} : <>, <i3>
    %49 = extsi %48 {handshake.bb = 2 : ui32, handshake.name = "extsi6"} : <i3> to <i32>
    %50 = shli %33#0, %49 {handshake.bb = 2 : ui32, handshake.name = "shli0"} : <i32>
    %51 = buffer %50, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer9"} : <i32>
    %52 = trunci %51 {handshake.bb = 2 : ui32, handshake.name = "trunci2"} : <i32> to <i9>
    %53 = shli %33#1, %46 {handshake.bb = 2 : ui32, handshake.name = "shli1"} : <i32>
    %54 = buffer %53, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer10"} : <i32>
    %55 = trunci %54 {handshake.bb = 2 : ui32, handshake.name = "trunci3"} : <i32> to <i9>
    %56 = addi %52, %55 {handshake.bb = 2 : ui32, handshake.name = "addi5"} : <i9>
    %57 = buffer %56, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer11"} : <i9>
    %58 = addi %24, %57 {handshake.bb = 2 : ui32, handshake.name = "addi0"} : <i9>
    %addressResult_8, %dataResult_9 = load[%58] %outputs_4#0 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i9>, <f32>, <i9>, <f32>
    %addressResult_10, %dataResult_11 = load[%26] %outputs_2 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load2"} : <i5>, <f32>, <i5>, <f32>
    %59 = mulf %dataResult_9, %dataResult_11 {fastmath = #arith.fastmath<none>, fpu_impl = #handshake<fpu_impl flopoco>, handshake.bb = 2 : ui32, handshake.name = "mulf0", internal_delay = "2_875333"} : <f32>
    %60 = buffer %27, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer5"} : <f32>
    %61 = addf %60, %59 {fastmath = #arith.fastmath<none>, fpu_impl = #handshake<fpu_impl flopoco>, handshake.bb = 2 : ui32, handshake.name = "addf0", internal_delay = "2_922000"} : <f32>
    %62 = addi %25, %43 {handshake.bb = 2 : ui32, handshake.name = "addi2"} : <i7>
    %63 = buffer %62, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer12"} : <i7>
    %64:2 = fork [2] %63 {handshake.bb = 2 : ui32, handshake.name = "fork9"} : <i7>
    %65 = trunci %64#0 {handshake.bb = 2 : ui32, handshake.name = "trunci4"} : <i7> to <i6>
    %66 = cmpi ult, %64#1, %40 {handshake.bb = 2 : ui32, handshake.name = "cmpi0"} : <i7>
    %67 = buffer %66, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer13"} : <i1>
    %68:5 = fork [5] %67 {handshake.bb = 2 : ui32, handshake.name = "fork10"} : <i1>
    %trueResult, %falseResult = cond_br %68#0, %65 {handshake.bb = 2 : ui32, handshake.name = "cond_br3"} : <i1>, <i6>
    sink %falseResult {handshake.name = "sink0"} : <i6>
    %trueResult_12, %falseResult_13 = cond_br %69, %61 {handshake.bb = 2 : ui32, handshake.name = "cond_br4"} : <i1>, <f32>
    %69 = buffer %68#2, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer19"} : <i1>
    %trueResult_14, %falseResult_15 = cond_br %68#1, %31#0 {handshake.bb = 2 : ui32, handshake.name = "cond_br5"} : <i1>, <i6>
    %trueResult_16, %falseResult_17 = cond_br %68#3, %36#1 {handshake.bb = 2 : ui32, handshake.name = "cond_br6"} : <i1>, <>
    %trueResult_18, %falseResult_19 = cond_br %68#4, %37 {handshake.bb = 2 : ui32, handshake.name = "cond_br7"} : <i1>, <i1>
    sink %trueResult_18 {handshake.name = "sink1"} : <i1>
    %70 = extsi %falseResult_19 {handshake.bb = 2 : ui32, handshake.name = "extsi15"} : <i1> to <i6>
    %trueResult_20, %falseResult_21 = cond_br %71, %130 {handshake.bb = 3 : ui32, handshake.name = "cond_br15"} : <i1>, <>
    %71 = buffer %137#3, bufferType = FIFO_BREAK_NONE, numSlots = 15 {handshake.bb = 3 : ui32, handshake.name = "buffer24"} : <i1>
    %72 = buffer %5, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer2"} : <>
    %73 = mux %74 [%72, %trueResult_20] {ftd.phi, handshake.bb = 3 : ui32, handshake.name = "mux9"} : <i1>, [<>, <>] to <>
    %74 = init %75 {ftd.imerge, handshake.bb = 3 : ui32, handshake.name = "init2"} : <i1>
    %75 = buffer %137#2, bufferType = FIFO_BREAK_NONE, numSlots = 15 {handshake.bb = 3 : ui32, handshake.name = "buffer25"} : <i1>
    %76 = mux %98#1 [%70, %trueResult_30] {handshake.bb = 3 : ui32, handshake.name = "mux4"} : <i1>, [<i6>, <i6>] to <i6>
    %77 = buffer %76, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer15"} : <i6>
    %78 = buffer %77, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer16"} : <i6>
    %79:4 = fork [4] %78 {handshake.bb = 3 : ui32, handshake.name = "fork11"} : <i6>
    %80 = extsi %81 {handshake.bb = 3 : ui32, handshake.name = "extsi23"} : <i6> to <i9>
    %81 = buffer %79#0, bufferType = FIFO_BREAK_NONE, numSlots = 10 {handshake.bb = 3 : ui32, handshake.name = "buffer27"} : <i6>
    %82 = extsi %79#2 {handshake.bb = 3 : ui32, handshake.name = "extsi24"} : <i6> to <i7>
    %83 = extsi %84 {handshake.bb = 3 : ui32, handshake.name = "extsi25"} : <i6> to <i32>
    %84 = buffer %79#3, bufferType = FIFO_BREAK_NONE, numSlots = 15 {handshake.bb = 3 : ui32, handshake.name = "buffer29"} : <i6>
    %85 = trunci %86 {handshake.bb = 3 : ui32, handshake.name = "trunci5"} : <i6> to <i5>
    %86 = buffer %79#1, bufferType = FIFO_BREAK_NONE, numSlots = 15 {handshake.bb = 3 : ui32, handshake.name = "buffer30"} : <i6>
    %87 = mux %98#0 [%falseResult_15, %trueResult_32] {handshake.bb = 3 : ui32, handshake.name = "mux5"} : <i1>, [<i6>, <i6>] to <i6>
    %88 = buffer %87, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer17"} : <i6>
    %89 = buffer %88, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer18"} : <i6>
    %90:2 = fork [2] %89 {handshake.bb = 3 : ui32, handshake.name = "fork12"} : <i6>
    %91 = extsi %90#1 {handshake.bb = 3 : ui32, handshake.name = "extsi26"} : <i6> to <i32>
    %92:2 = fork [2] %91 {handshake.bb = 3 : ui32, handshake.name = "fork13"} : <i32>
    %93 = mux %94 [%falseResult_13, %trueResult_34] {handshake.bb = 3 : ui32, handshake.name = "mux6"} : <i1>, [<f32>, <f32>] to <f32>
    %94 = buffer %98#2, bufferType = FIFO_BREAK_NONE, numSlots = 11 {handshake.bb = 3 : ui32, handshake.name = "buffer33"} : <i1>
    %95 = buffer %93, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer20"} : <f32>
    %96 = buffer %95, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer21"} : <f32>
    %97:2 = fork [2] %96 {handshake.bb = 3 : ui32, handshake.name = "fork14"} : <f32>
    %result_22, %index_23 = control_merge [%falseResult_17, %trueResult_36]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>, <>] to <>, <i1>
    %98:3 = fork [3] %index_23 {handshake.bb = 3 : ui32, handshake.name = "fork15"} : <i1>
    %99:2 = fork [2] %result_22 {handshake.bb = 3 : ui32, handshake.name = "fork16"} : <>
    %100 = constant %99#0 {handshake.bb = 3 : ui32, handshake.name = "constant22", value = 1 : i2} : <>, <i2>
    %101 = extsi %100 {handshake.bb = 3 : ui32, handshake.name = "extsi7"} : <i2> to <i32>
    %102 = source {handshake.bb = 3 : ui32, handshake.name = "source4"} : <>
    %103 = constant %102 {handshake.bb = 3 : ui32, handshake.name = "constant23", value = 20 : i6} : <>, <i6>
    %104 = extsi %103 {handshake.bb = 3 : ui32, handshake.name = "extsi27"} : <i6> to <i7>
    %105 = source {handshake.bb = 3 : ui32, handshake.name = "source5"} : <>
    %106 = constant %105 {handshake.bb = 3 : ui32, handshake.name = "constant24", value = 1 : i2} : <>, <i2>
    %107 = extsi %106 {handshake.bb = 3 : ui32, handshake.name = "extsi28"} : <i2> to <i7>
    %108 = source {handshake.bb = 3 : ui32, handshake.name = "source6"} : <>
    %109 = constant %108 {handshake.bb = 3 : ui32, handshake.name = "constant25", value = 4 : i4} : <>, <i4>
    %110 = extsi %109 {handshake.bb = 3 : ui32, handshake.name = "extsi10"} : <i4> to <i32>
    %111 = source {handshake.bb = 3 : ui32, handshake.name = "source7"} : <>
    %112 = constant %111 {handshake.bb = 3 : ui32, handshake.name = "constant26", value = 2 : i3} : <>, <i3>
    %113 = extsi %112 {handshake.bb = 3 : ui32, handshake.name = "extsi11"} : <i3> to <i32>
    %114 = buffer %73, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer14"} : <>
    %115 = gate %83, %114 {handshake.bb = 3 : ui32, handshake.name = "gate0"} : <i32>, !handshake.control<> to <i32>
    %116 = trunci %115 {handshake.bb = 3 : ui32, handshake.name = "trunci6"} : <i32> to <i5>
    %addressResult_24, %dataResult_25 = load[%116] %outputs_0#0 {handshake.bb = 3 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["store0", 1, false], ["store0", 3, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load3"} : <i5>, <f32>, <i5>, <f32>
    %117 = shli %118, %113 {handshake.bb = 3 : ui32, handshake.name = "shli2"} : <i32>
    %118 = buffer %92#0, bufferType = FIFO_BREAK_NONE, numSlots = 10 {handshake.bb = 3 : ui32, handshake.name = "buffer34"} : <i32>
    %119 = buffer %117, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer23"} : <i32>
    %120 = trunci %119 {handshake.bb = 3 : ui32, handshake.name = "trunci7"} : <i32> to <i9>
    %121 = shli %122, %110 {handshake.bb = 3 : ui32, handshake.name = "shli3"} : <i32>
    %122 = buffer %92#1, bufferType = FIFO_BREAK_NONE, numSlots = 10 {handshake.bb = 3 : ui32, handshake.name = "buffer35"} : <i32>
    %123 = buffer %121, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer26"} : <i32>
    %124 = trunci %123 {handshake.bb = 3 : ui32, handshake.name = "trunci8"} : <i32> to <i9>
    %125 = addi %120, %124 {handshake.bb = 3 : ui32, handshake.name = "addi6"} : <i9>
    %126 = buffer %125, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer28"} : <i9>
    %127 = addi %80, %126 {handshake.bb = 3 : ui32, handshake.name = "addi1"} : <i9>
    %addressResult_26, %dataResult_27 = load[%127] %outputs_4#1 {handshake.bb = 3 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load4"} : <i9>, <f32>, <i9>, <f32>
    %128 = mulf %dataResult_27, %97#1 {fastmath = #arith.fastmath<none>, fpu_impl = #handshake<fpu_impl flopoco>, handshake.bb = 3 : ui32, handshake.name = "mulf1", internal_delay = "2_875333"} : <f32>
    %129 = addf %dataResult_25, %128 {fastmath = #arith.fastmath<none>, fpu_impl = #handshake<fpu_impl flopoco>, handshake.bb = 3 : ui32, handshake.name = "addf1", internal_delay = "2_922000"} : <f32>
    %130 = buffer %doneResult, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer1"} : <>
    %addressResult_28, %dataResult_29, %doneResult = store[%85] %129 %outputs_0#1 {handshake.bb = 3 : ui32, handshake.bufProps = #handshake<bufProps{"2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["load3", 1, false], ["store0", 1, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store0"} : <i5>, <f32>, <>, <i5>, <f32>, <>
    %131 = addi %82, %107 {handshake.bb = 3 : ui32, handshake.name = "addi3"} : <i7>
    %132 = buffer %131, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer31"} : <i7>
    %133:2 = fork [2] %132 {handshake.bb = 3 : ui32, handshake.name = "fork17"} : <i7>
    %134 = trunci %133#0 {handshake.bb = 3 : ui32, handshake.name = "trunci9"} : <i7> to <i6>
    %135 = cmpi ult, %133#1, %104 {handshake.bb = 3 : ui32, handshake.name = "cmpi1"} : <i7>
    %136 = buffer %135, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer32"} : <i1>
    %137:6 = fork [6] %136 {handshake.bb = 3 : ui32, handshake.name = "fork18"} : <i1>
    %trueResult_30, %falseResult_31 = cond_br %137#0, %134 {handshake.bb = 3 : ui32, handshake.name = "cond_br8"} : <i1>, <i6>
    sink %falseResult_31 {handshake.name = "sink2"} : <i6>
    %trueResult_32, %falseResult_33 = cond_br %137#1, %90#0 {handshake.bb = 3 : ui32, handshake.name = "cond_br9"} : <i1>, <i6>
    %trueResult_34, %falseResult_35 = cond_br %138, %97#0 {handshake.bb = 3 : ui32, handshake.name = "cond_br10"} : <i1>, <f32>
    %138 = buffer %137#4, bufferType = FIFO_BREAK_NONE, numSlots = 12 {handshake.bb = 3 : ui32, handshake.name = "buffer42"} : <i1>
    %139 = buffer %99#1, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer22"} : <>
    %trueResult_36, %falseResult_37 = cond_br %137#5, %139 {handshake.bb = 3 : ui32, handshake.name = "cond_br11"} : <i1>, <>
    %trueResult_38, %falseResult_39 = cond_br %140, %falseResult_21 {handshake.bb = 4 : ui32, handshake.name = "cond_br16"} : <i1>, <>
    %140 = buffer %163#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer45"} : <i1>
    sink %falseResult_39 {handshake.name = "sink3"} : <>
    %141 = merge %falseResult_33 {handshake.bb = 4 : ui32, handshake.name = "merge0"} : <i6>
    %142:2 = fork [2] %141 {handshake.bb = 4 : ui32, handshake.name = "fork19"} : <i6>
    %143 = extsi %142#0 {handshake.bb = 4 : ui32, handshake.name = "extsi29"} : <i6> to <i7>
    %144 = extsi %142#1 {handshake.bb = 4 : ui32, handshake.name = "extsi30"} : <i6> to <i32>
    %145 = merge %falseResult_35 {handshake.bb = 4 : ui32, handshake.name = "merge1"} : <f32>
    %result_40, %index_41 = control_merge [%falseResult_37]  {handshake.bb = 4 : ui32, handshake.name = "control_merge3"} : [<>] to <>, <i1>
    sink %index_41 {handshake.name = "sink4"} : <i1>
    %146:2 = fork [2] %result_40 {handshake.bb = 4 : ui32, handshake.name = "fork20"} : <>
    %147 = constant %146#0 {handshake.bb = 4 : ui32, handshake.name = "constant27", value = 1 : i2} : <>, <i2>
    %148 = extsi %147 {handshake.bb = 4 : ui32, handshake.name = "extsi12"} : <i2> to <i32>
    %149 = source {handshake.bb = 4 : ui32, handshake.name = "source8"} : <>
    %150 = constant %149 {handshake.bb = 4 : ui32, handshake.name = "constant28", value = 20 : i6} : <>, <i6>
    %151 = extsi %150 {handshake.bb = 4 : ui32, handshake.name = "extsi31"} : <i6> to <i7>
    %152 = source {handshake.bb = 4 : ui32, handshake.name = "source9"} : <>
    %153 = constant %152 {handshake.bb = 4 : ui32, handshake.name = "constant29", value = 1 : i2} : <>, <i2>
    %154 = extsi %153 {handshake.bb = 4 : ui32, handshake.name = "extsi32"} : <i2> to <i7>
    %155 = gate %144, %14 {handshake.bb = 4 : ui32, handshake.name = "gate1"} : <i32>, !handshake.control<> to <i32>
    %156 = trunci %155 {handshake.bb = 4 : ui32, handshake.name = "trunci10"} : <i32> to <i5>
    %addressResult_42, %dataResult_43, %doneResult_44 = store[%156] %145 %outputs#1 {handshake.bb = 4 : ui32, handshake.bufProps = #handshake<bufProps{"2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store1"} : <i5>, <f32>, <>, <i5>, <f32>, <>
    sink %doneResult_44 {handshake.name = "sink5"} : <>
    %157 = addi %143, %154 {handshake.bb = 4 : ui32, handshake.name = "addi4"} : <i7>
    %158 = buffer %157, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer36"} : <i7>
    %159:2 = fork [2] %158 {handshake.bb = 4 : ui32, handshake.name = "fork21"} : <i7>
    %160 = trunci %159#0 {handshake.bb = 4 : ui32, handshake.name = "trunci11"} : <i7> to <i6>
    %161 = cmpi ult, %159#1, %151 {handshake.bb = 4 : ui32, handshake.name = "cmpi2"} : <i7>
    %162 = buffer %161, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer37"} : <i1>
    %163:4 = fork [4] %162 {handshake.bb = 4 : ui32, handshake.name = "fork22"} : <i1>
    %trueResult_45, %falseResult_46 = cond_br %163#0, %160 {handshake.bb = 4 : ui32, handshake.name = "cond_br12"} : <i1>, <i6>
    sink %falseResult_46 {handshake.name = "sink6"} : <i6>
    %trueResult_47, %falseResult_48 = cond_br %163#3, %146#1 {handshake.bb = 4 : ui32, handshake.name = "cond_br13"} : <i1>, <>
    %result_49, %index_50 = control_merge [%falseResult_48]  {handshake.bb = 5 : ui32, handshake.name = "control_merge4"} : [<>] to <>, <i1>
    sink %index_50 {handshake.name = "sink7"} : <i1>
    %164:4 = fork [4] %result_49 {handshake.bb = 5 : ui32, handshake.name = "fork23"} : <>
    end {handshake.bb = 5 : ui32, handshake.name = "end0"} %memEnd_5, %memEnd_3, %memEnd_1, %memEnd, %0#1 : <>, <>, <>, <>, <>
  }
}

