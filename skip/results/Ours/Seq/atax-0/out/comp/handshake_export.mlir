module {
  handshake.func @atax(%arg0: memref<400xf32>, %arg1: memref<20xf32>, %arg2: memref<20xf32>, %arg3: memref<20xf32>, %arg4: !handshake.control<>, %arg5: !handshake.control<>, %arg6: !handshake.control<>, %arg7: !handshake.control<>, %arg8: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["A", "x", "y", "tmp", "A_start", "x_start", "y_start", "tmp_start", "start"], cfg.edges = "[0,1][2,2,3,cmpi0][4,1,5,cmpi2][1,2][3,3,4,cmpi1]", resNames = ["A_end", "x_end", "y_end", "tmp_end", "end"]} {
    %0:4 = fork [4] %arg8 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork0"} : <>
    %outputs:2, %memEnd = mem_controller[%arg3 : memref<20xf32>] %arg7 (%addressResult, %140, %addressResult_40, %dataResult_41) %156#3 {connectedBlocks = [1 : i32, 4 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "5": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller2"} :    (!handshake.channel<i5>, !handshake.channel<i32>, !handshake.channel<i5>, !handshake.channel<f32>) -> (!handshake.channel<f32>, !handshake.control<>)
    %outputs_0:2, %memEnd_1 = mem_controller[%arg2 : memref<20xf32>] %arg6 (%95, %addressResult_24, %addressResult_28, %dataResult_29) %156#2 {connectedBlocks = [3 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "5": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller3"} :    (!handshake.channel<i32>, !handshake.channel<i5>, !handshake.channel<i5>, !handshake.channel<f32>) -> (!handshake.channel<f32>, !handshake.control<>)
    %outputs_2, %memEnd_3 = mem_controller[%arg1 : memref<20xf32>] %arg5 (%addressResult_10) %156#1 {connectedBlocks = [2 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller4"} :    (!handshake.channel<i5>) -> !handshake.channel<f32>
    %outputs_4:2, %memEnd_5 = mem_controller[%arg0 : memref<400xf32>] %arg4 (%addressResult_8, %addressResult_26) %156#0 {connectedBlocks = [2 : i32, 3 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller5"} :    (!handshake.channel<i9>, !handshake.channel<i9>) -> (!handshake.channel<f32>, !handshake.channel<f32>)
    %1 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant2", value = false} : <>, <i1>
    %2 = extsi %1 {handshake.bb = 0 : ui32, handshake.name = "extsi17"} : <i1> to <i6>
    %3 = mux %4 [%0#2, %trueResult_38] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux8"} : <i1>, [<>, <>] to <>
    %4 = init %155#2 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "init0"} : <i1>
    %5:2 = unbundle %13#0  {handshake.bb = 1 : ui32, handshake.name = "unbundle0"} : <f32> to _ 
    %6 = mux %index [%2, %trueResult_43] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i6>, <i6>] to <i6>
    %7 = buffer %6, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer3"} : <i6>
    %8:2 = fork [2] %7 {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <i6>
    %9 = trunci %8#0 {handshake.bb = 1 : ui32, handshake.name = "trunci0"} : <i6> to <i5>
    %result, %index = control_merge [%0#3, %trueResult_45]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %10:2 = fork [2] %result {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <>
    %11 = constant %10#0 {handshake.bb = 1 : ui32, handshake.name = "constant3", value = false} : <>, <i1>
    %12 = buffer %5#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer0"} : <>
    %addressResult, %dataResult = load[%9] %outputs#0 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["store1", 2, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i5>, <f32>, <i5>, <f32>
    %13:2 = fork [2] %dataResult {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <f32>
    %14 = extsi %11 {handshake.bb = 1 : ui32, handshake.name = "extsi16"} : <i1> to <i6>
    %15 = mux %28#1 [%14, %trueResult] {handshake.bb = 2 : ui32, handshake.name = "mux1"} : <i1>, [<i6>, <i6>] to <i6>
    %16 = buffer %15, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer4"} : <i6>
    %17:3 = fork [3] %16 {handshake.bb = 2 : ui32, handshake.name = "fork4"} : <i6>
    %18 = extsi %17#0 {handshake.bb = 2 : ui32, handshake.name = "extsi18"} : <i6> to <i9>
    %19 = extsi %17#2 {handshake.bb = 2 : ui32, handshake.name = "extsi19"} : <i6> to <i7>
    %20 = trunci %17#1 {handshake.bb = 2 : ui32, handshake.name = "trunci1"} : <i6> to <i5>
    %21 = mux %28#2 [%13#1, %trueResult_12] {handshake.bb = 2 : ui32, handshake.name = "mux2"} : <i1>, [<f32>, <f32>] to <f32>
    %22 = mux %28#0 [%8#1, %trueResult_14] {handshake.bb = 2 : ui32, handshake.name = "mux3"} : <i1>, [<i6>, <i6>] to <i6>
    %23 = buffer %22, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer6"} : <i6>
    %24 = buffer %23, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer7"} : <i6>
    %25:2 = fork [2] %24 {handshake.bb = 2 : ui32, handshake.name = "fork5"} : <i6>
    %26 = extsi %25#1 {handshake.bb = 2 : ui32, handshake.name = "extsi20"} : <i6> to <i32>
    %27:2 = fork [2] %26 {handshake.bb = 2 : ui32, handshake.name = "fork6"} : <i32>
    %result_6, %index_7 = control_merge [%10#1, %trueResult_16]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>, <>] to <>, <i1>
    %28:3 = fork [3] %index_7 {handshake.bb = 2 : ui32, handshake.name = "fork7"} : <i1>
    %29 = buffer %result_6, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer8"} : <>
    %30:2 = fork [2] %29 {handshake.bb = 2 : ui32, handshake.name = "fork8"} : <>
    %31 = constant %30#0 {handshake.bb = 2 : ui32, handshake.name = "constant17", value = false} : <>, <i1>
    %32 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %33 = constant %32 {handshake.bb = 2 : ui32, handshake.name = "constant18", value = 20 : i6} : <>, <i6>
    %34 = extsi %33 {handshake.bb = 2 : ui32, handshake.name = "extsi21"} : <i6> to <i7>
    %35 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %36 = constant %35 {handshake.bb = 2 : ui32, handshake.name = "constant19", value = 1 : i2} : <>, <i2>
    %37 = extsi %36 {handshake.bb = 2 : ui32, handshake.name = "extsi22"} : <i2> to <i7>
    %38 = source {handshake.bb = 2 : ui32, handshake.name = "source2"} : <>
    %39 = constant %38 {handshake.bb = 2 : ui32, handshake.name = "constant20", value = 4 : i4} : <>, <i4>
    %40 = extsi %39 {handshake.bb = 2 : ui32, handshake.name = "extsi5"} : <i4> to <i32>
    %41 = source {handshake.bb = 2 : ui32, handshake.name = "source3"} : <>
    %42 = constant %41 {handshake.bb = 2 : ui32, handshake.name = "constant21", value = 2 : i3} : <>, <i3>
    %43 = extsi %42 {handshake.bb = 2 : ui32, handshake.name = "extsi6"} : <i3> to <i32>
    %44 = shli %27#0, %43 {handshake.bb = 2 : ui32, handshake.name = "shli0"} : <i32>
    %45 = buffer %44, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer9"} : <i32>
    %46 = trunci %45 {handshake.bb = 2 : ui32, handshake.name = "trunci2"} : <i32> to <i9>
    %47 = shli %27#1, %40 {handshake.bb = 2 : ui32, handshake.name = "shli1"} : <i32>
    %48 = buffer %47, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer10"} : <i32>
    %49 = trunci %48 {handshake.bb = 2 : ui32, handshake.name = "trunci3"} : <i32> to <i9>
    %50 = addi %46, %49 {handshake.bb = 2 : ui32, handshake.name = "addi5"} : <i9>
    %51 = buffer %50, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer11"} : <i9>
    %52 = addi %18, %51 {handshake.bb = 2 : ui32, handshake.name = "addi0"} : <i9>
    %addressResult_8, %dataResult_9 = load[%52] %outputs_4#0 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i9>, <f32>, <i9>, <f32>
    %addressResult_10, %dataResult_11 = load[%20] %outputs_2 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load2"} : <i5>, <f32>, <i5>, <f32>
    %53 = mulf %dataResult_9, %dataResult_11 {fastmath = #arith.fastmath<none>, fpu_impl = #handshake<fpu_impl flopoco>, handshake.bb = 2 : ui32, handshake.name = "mulf0", internal_delay = "2_875333"} : <f32>
    %54 = buffer %21, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer5"} : <f32>
    %55 = addf %54, %53 {fastmath = #arith.fastmath<none>, fpu_impl = #handshake<fpu_impl flopoco>, handshake.bb = 2 : ui32, handshake.name = "addf0", internal_delay = "2_922000"} : <f32>
    %56 = addi %19, %37 {handshake.bb = 2 : ui32, handshake.name = "addi2"} : <i7>
    %57 = buffer %56, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer12"} : <i7>
    %58:2 = fork [2] %57 {handshake.bb = 2 : ui32, handshake.name = "fork9"} : <i7>
    %59 = trunci %58#0 {handshake.bb = 2 : ui32, handshake.name = "trunci4"} : <i7> to <i6>
    %60 = cmpi ult, %58#1, %34 {handshake.bb = 2 : ui32, handshake.name = "cmpi0"} : <i7>
    %61 = buffer %60, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer13"} : <i1>
    %62:5 = fork [5] %61 {handshake.bb = 2 : ui32, handshake.name = "fork10"} : <i1>
    %trueResult, %falseResult = cond_br %62#0, %59 {handshake.bb = 2 : ui32, handshake.name = "cond_br3"} : <i1>, <i6>
    sink %falseResult {handshake.name = "sink0"} : <i6>
    %trueResult_12, %falseResult_13 = cond_br %63, %55 {handshake.bb = 2 : ui32, handshake.name = "cond_br4"} : <i1>, <f32>
    %63 = buffer %62#2, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer19"} : <i1>
    %trueResult_14, %falseResult_15 = cond_br %62#1, %25#0 {handshake.bb = 2 : ui32, handshake.name = "cond_br5"} : <i1>, <i6>
    %trueResult_16, %falseResult_17 = cond_br %62#3, %30#1 {handshake.bb = 2 : ui32, handshake.name = "cond_br6"} : <i1>, <>
    %trueResult_18, %falseResult_19 = cond_br %62#4, %31 {handshake.bb = 2 : ui32, handshake.name = "cond_br7"} : <i1>, <i1>
    sink %trueResult_18 {handshake.name = "sink1"} : <i1>
    %64 = extsi %falseResult_19 {handshake.bb = 2 : ui32, handshake.name = "extsi15"} : <i1> to <i6>
    %trueResult_20, %falseResult_21 = cond_br %65, %124 {handshake.bb = 3 : ui32, handshake.name = "cond_br15"} : <i1>, <>
    %65 = buffer %131#3, bufferType = FIFO_BREAK_NONE, numSlots = 15 {handshake.bb = 3 : ui32, handshake.name = "buffer24"} : <i1>
    %66 = buffer %3, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer2"} : <>
    %67 = mux %68 [%66, %trueResult_20] {ftd.phi, handshake.bb = 3 : ui32, handshake.name = "mux9"} : <i1>, [<>, <>] to <>
    %68 = init %69 {ftd.imerge, handshake.bb = 3 : ui32, handshake.name = "init2"} : <i1>
    %69 = buffer %131#2, bufferType = FIFO_BREAK_NONE, numSlots = 15 {handshake.bb = 3 : ui32, handshake.name = "buffer25"} : <i1>
    %70 = mux %92#1 [%64, %trueResult_30] {handshake.bb = 3 : ui32, handshake.name = "mux4"} : <i1>, [<i6>, <i6>] to <i6>
    %71 = buffer %70, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer15"} : <i6>
    %72 = buffer %71, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer16"} : <i6>
    %73:4 = fork [4] %72 {handshake.bb = 3 : ui32, handshake.name = "fork11"} : <i6>
    %74 = extsi %75 {handshake.bb = 3 : ui32, handshake.name = "extsi23"} : <i6> to <i9>
    %75 = buffer %73#0, bufferType = FIFO_BREAK_NONE, numSlots = 10 {handshake.bb = 3 : ui32, handshake.name = "buffer27"} : <i6>
    %76 = extsi %73#2 {handshake.bb = 3 : ui32, handshake.name = "extsi24"} : <i6> to <i7>
    %77 = extsi %78 {handshake.bb = 3 : ui32, handshake.name = "extsi25"} : <i6> to <i32>
    %78 = buffer %73#3, bufferType = FIFO_BREAK_NONE, numSlots = 15 {handshake.bb = 3 : ui32, handshake.name = "buffer29"} : <i6>
    %79 = trunci %80 {handshake.bb = 3 : ui32, handshake.name = "trunci5"} : <i6> to <i5>
    %80 = buffer %73#1, bufferType = FIFO_BREAK_NONE, numSlots = 15 {handshake.bb = 3 : ui32, handshake.name = "buffer30"} : <i6>
    %81 = mux %92#0 [%falseResult_15, %trueResult_32] {handshake.bb = 3 : ui32, handshake.name = "mux5"} : <i1>, [<i6>, <i6>] to <i6>
    %82 = buffer %81, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer17"} : <i6>
    %83 = buffer %82, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer18"} : <i6>
    %84:2 = fork [2] %83 {handshake.bb = 3 : ui32, handshake.name = "fork12"} : <i6>
    %85 = extsi %84#1 {handshake.bb = 3 : ui32, handshake.name = "extsi26"} : <i6> to <i32>
    %86:2 = fork [2] %85 {handshake.bb = 3 : ui32, handshake.name = "fork13"} : <i32>
    %87 = mux %88 [%falseResult_13, %trueResult_34] {handshake.bb = 3 : ui32, handshake.name = "mux6"} : <i1>, [<f32>, <f32>] to <f32>
    %88 = buffer %92#2, bufferType = FIFO_BREAK_NONE, numSlots = 11 {handshake.bb = 3 : ui32, handshake.name = "buffer33"} : <i1>
    %89 = buffer %87, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer20"} : <f32>
    %90 = buffer %89, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer21"} : <f32>
    %91:2 = fork [2] %90 {handshake.bb = 3 : ui32, handshake.name = "fork14"} : <f32>
    %result_22, %index_23 = control_merge [%falseResult_17, %trueResult_36]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>, <>] to <>, <i1>
    %92:3 = fork [3] %index_23 {handshake.bb = 3 : ui32, handshake.name = "fork15"} : <i1>
    %93:2 = fork [2] %result_22 {handshake.bb = 3 : ui32, handshake.name = "fork16"} : <>
    %94 = constant %93#0 {handshake.bb = 3 : ui32, handshake.name = "constant22", value = 1 : i2} : <>, <i2>
    %95 = extsi %94 {handshake.bb = 3 : ui32, handshake.name = "extsi7"} : <i2> to <i32>
    %96 = source {handshake.bb = 3 : ui32, handshake.name = "source4"} : <>
    %97 = constant %96 {handshake.bb = 3 : ui32, handshake.name = "constant23", value = 20 : i6} : <>, <i6>
    %98 = extsi %97 {handshake.bb = 3 : ui32, handshake.name = "extsi27"} : <i6> to <i7>
    %99 = source {handshake.bb = 3 : ui32, handshake.name = "source5"} : <>
    %100 = constant %99 {handshake.bb = 3 : ui32, handshake.name = "constant24", value = 1 : i2} : <>, <i2>
    %101 = extsi %100 {handshake.bb = 3 : ui32, handshake.name = "extsi28"} : <i2> to <i7>
    %102 = source {handshake.bb = 3 : ui32, handshake.name = "source6"} : <>
    %103 = constant %102 {handshake.bb = 3 : ui32, handshake.name = "constant25", value = 4 : i4} : <>, <i4>
    %104 = extsi %103 {handshake.bb = 3 : ui32, handshake.name = "extsi10"} : <i4> to <i32>
    %105 = source {handshake.bb = 3 : ui32, handshake.name = "source7"} : <>
    %106 = constant %105 {handshake.bb = 3 : ui32, handshake.name = "constant26", value = 2 : i3} : <>, <i3>
    %107 = extsi %106 {handshake.bb = 3 : ui32, handshake.name = "extsi11"} : <i3> to <i32>
    %108 = buffer %67, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer14"} : <>
    %109 = gate %77, %108 {handshake.bb = 3 : ui32, handshake.name = "gate0"} : <i32>, !handshake.control<> to <i32>
    %110 = trunci %109 {handshake.bb = 3 : ui32, handshake.name = "trunci6"} : <i32> to <i5>
    %addressResult_24, %dataResult_25 = load[%110] %outputs_0#0 {handshake.bb = 3 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["store0", 1, false], ["store0", 3, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load3"} : <i5>, <f32>, <i5>, <f32>
    %111 = shli %112, %107 {handshake.bb = 3 : ui32, handshake.name = "shli2"} : <i32>
    %112 = buffer %86#0, bufferType = FIFO_BREAK_NONE, numSlots = 10 {handshake.bb = 3 : ui32, handshake.name = "buffer34"} : <i32>
    %113 = buffer %111, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer23"} : <i32>
    %114 = trunci %113 {handshake.bb = 3 : ui32, handshake.name = "trunci7"} : <i32> to <i9>
    %115 = shli %116, %104 {handshake.bb = 3 : ui32, handshake.name = "shli3"} : <i32>
    %116 = buffer %86#1, bufferType = FIFO_BREAK_NONE, numSlots = 10 {handshake.bb = 3 : ui32, handshake.name = "buffer35"} : <i32>
    %117 = buffer %115, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer26"} : <i32>
    %118 = trunci %117 {handshake.bb = 3 : ui32, handshake.name = "trunci8"} : <i32> to <i9>
    %119 = addi %114, %118 {handshake.bb = 3 : ui32, handshake.name = "addi6"} : <i9>
    %120 = buffer %119, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer28"} : <i9>
    %121 = addi %74, %120 {handshake.bb = 3 : ui32, handshake.name = "addi1"} : <i9>
    %addressResult_26, %dataResult_27 = load[%121] %outputs_4#1 {handshake.bb = 3 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load4"} : <i9>, <f32>, <i9>, <f32>
    %122 = mulf %dataResult_27, %91#1 {fastmath = #arith.fastmath<none>, fpu_impl = #handshake<fpu_impl flopoco>, handshake.bb = 3 : ui32, handshake.name = "mulf1", internal_delay = "2_875333"} : <f32>
    %123 = addf %dataResult_25, %122 {fastmath = #arith.fastmath<none>, fpu_impl = #handshake<fpu_impl flopoco>, handshake.bb = 3 : ui32, handshake.name = "addf1", internal_delay = "2_922000"} : <f32>
    %124 = buffer %doneResult, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer1"} : <>
    %addressResult_28, %dataResult_29, %doneResult = store[%79] %123 %outputs_0#1 {handshake.bb = 3 : ui32, handshake.bufProps = #handshake<bufProps{"2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["load3", 1, false], ["store0", 1, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store0"} : <i5>, <f32>, <>, <i5>, <f32>, <>
    %125 = addi %76, %101 {handshake.bb = 3 : ui32, handshake.name = "addi3"} : <i7>
    %126 = buffer %125, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer31"} : <i7>
    %127:2 = fork [2] %126 {handshake.bb = 3 : ui32, handshake.name = "fork17"} : <i7>
    %128 = trunci %127#0 {handshake.bb = 3 : ui32, handshake.name = "trunci9"} : <i7> to <i6>
    %129 = cmpi ult, %127#1, %98 {handshake.bb = 3 : ui32, handshake.name = "cmpi1"} : <i7>
    %130 = buffer %129, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer32"} : <i1>
    %131:6 = fork [6] %130 {handshake.bb = 3 : ui32, handshake.name = "fork18"} : <i1>
    %trueResult_30, %falseResult_31 = cond_br %131#0, %128 {handshake.bb = 3 : ui32, handshake.name = "cond_br8"} : <i1>, <i6>
    sink %falseResult_31 {handshake.name = "sink2"} : <i6>
    %trueResult_32, %falseResult_33 = cond_br %131#1, %84#0 {handshake.bb = 3 : ui32, handshake.name = "cond_br9"} : <i1>, <i6>
    %trueResult_34, %falseResult_35 = cond_br %132, %91#0 {handshake.bb = 3 : ui32, handshake.name = "cond_br10"} : <i1>, <f32>
    %132 = buffer %131#4, bufferType = FIFO_BREAK_NONE, numSlots = 12 {handshake.bb = 3 : ui32, handshake.name = "buffer42"} : <i1>
    %133 = buffer %93#1, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer22"} : <>
    %trueResult_36, %falseResult_37 = cond_br %131#5, %133 {handshake.bb = 3 : ui32, handshake.name = "cond_br11"} : <i1>, <>
    %trueResult_38, %falseResult_39 = cond_br %134, %falseResult_21 {handshake.bb = 4 : ui32, handshake.name = "cond_br16"} : <i1>, <>
    %134 = buffer %155#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer45"} : <i1>
    sink %falseResult_39 {handshake.name = "sink3"} : <>
    %135:2 = fork [2] %falseResult_33 {handshake.bb = 4 : ui32, handshake.name = "fork19"} : <i6>
    %136 = extsi %135#0 {handshake.bb = 4 : ui32, handshake.name = "extsi29"} : <i6> to <i7>
    %137 = extsi %135#1 {handshake.bb = 4 : ui32, handshake.name = "extsi30"} : <i6> to <i32>
    %138:2 = fork [2] %falseResult_37 {handshake.bb = 4 : ui32, handshake.name = "fork20"} : <>
    %139 = constant %138#0 {handshake.bb = 4 : ui32, handshake.name = "constant27", value = 1 : i2} : <>, <i2>
    %140 = extsi %139 {handshake.bb = 4 : ui32, handshake.name = "extsi12"} : <i2> to <i32>
    %141 = source {handshake.bb = 4 : ui32, handshake.name = "source8"} : <>
    %142 = constant %141 {handshake.bb = 4 : ui32, handshake.name = "constant28", value = 20 : i6} : <>, <i6>
    %143 = extsi %142 {handshake.bb = 4 : ui32, handshake.name = "extsi31"} : <i6> to <i7>
    %144 = source {handshake.bb = 4 : ui32, handshake.name = "source9"} : <>
    %145 = constant %144 {handshake.bb = 4 : ui32, handshake.name = "constant29", value = 1 : i2} : <>, <i2>
    %146 = extsi %145 {handshake.bb = 4 : ui32, handshake.name = "extsi32"} : <i2> to <i7>
    %147 = gate %137, %12 {handshake.bb = 4 : ui32, handshake.name = "gate1"} : <i32>, !handshake.control<> to <i32>
    %148 = trunci %147 {handshake.bb = 4 : ui32, handshake.name = "trunci10"} : <i32> to <i5>
    %addressResult_40, %dataResult_41, %doneResult_42 = store[%148] %falseResult_35 %outputs#1 {handshake.bb = 4 : ui32, handshake.bufProps = #handshake<bufProps{"2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store1"} : <i5>, <f32>, <>, <i5>, <f32>, <>
    sink %doneResult_42 {handshake.name = "sink5"} : <>
    %149 = addi %136, %146 {handshake.bb = 4 : ui32, handshake.name = "addi4"} : <i7>
    %150 = buffer %149, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer36"} : <i7>
    %151:2 = fork [2] %150 {handshake.bb = 4 : ui32, handshake.name = "fork21"} : <i7>
    %152 = trunci %151#0 {handshake.bb = 4 : ui32, handshake.name = "trunci11"} : <i7> to <i6>
    %153 = cmpi ult, %151#1, %143 {handshake.bb = 4 : ui32, handshake.name = "cmpi2"} : <i7>
    %154 = buffer %153, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer37"} : <i1>
    %155:4 = fork [4] %154 {handshake.bb = 4 : ui32, handshake.name = "fork22"} : <i1>
    %trueResult_43, %falseResult_44 = cond_br %155#0, %152 {handshake.bb = 4 : ui32, handshake.name = "cond_br12"} : <i1>, <i6>
    sink %falseResult_44 {handshake.name = "sink6"} : <i6>
    %trueResult_45, %falseResult_46 = cond_br %155#3, %138#1 {handshake.bb = 4 : ui32, handshake.name = "cond_br13"} : <i1>, <>
    %156:4 = fork [4] %falseResult_46 {handshake.bb = 5 : ui32, handshake.name = "fork23"} : <>
    end {handshake.bb = 5 : ui32, handshake.name = "end0"} %memEnd_5, %memEnd_3, %memEnd_1, %memEnd, %0#1 : <>, <>, <>, <>, <>
  }
}

