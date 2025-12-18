module {
  handshake.func @matrix_power(%arg0: memref<400xi32>, %arg1: memref<20xi32>, %arg2: memref<20xi32>, %arg3: memref<20xi32>, %arg4: !handshake.control<>, %arg5: !handshake.control<>, %arg6: !handshake.control<>, %arg7: !handshake.control<>, %arg8: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["mat", "row", "col", "a", "mat_start", "row_start", "col_start", "a_start", "start"], cfg.edges = "[0,1][2,2,3,cmpi0][1,2][3,1,4,cmpi1]", resNames = ["mat_end", "row_end", "col_end", "a_end", "end"]} {
    %0:5 = fork [5] %arg8 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork0"} : <>
    %outputs, %memEnd = mem_controller[%arg3 : memref<20xi32>] %arg7 (%addressResult_8) %118#3 {connectedBlocks = [2 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller3"} :    (!handshake.channel<i5>) -> !handshake.channel<i32>
    %outputs_0, %memEnd_1 = mem_controller[%arg2 : memref<20xi32>] %arg6 (%addressResult_10) %118#2 {connectedBlocks = [2 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller4"} :    (!handshake.channel<i5>) -> !handshake.channel<i32>
    %outputs_2, %memEnd_3 = mem_controller[%arg1 : memref<20xi32>] %arg5 (%addressResult) %118#1 {connectedBlocks = [2 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller5"} :    (!handshake.channel<i5>) -> !handshake.channel<i32>
    %outputs_4:3, %memEnd_5 = mem_controller[%arg0 : memref<400xi32>] %arg4 (%46, %addressResult_12, %addressResult_14, %addressResult_16, %dataResult_17) %118#0 {connectedBlocks = [2 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "5": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "6": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller6"} :    (!handshake.channel<i32>, !handshake.channel<i9>, !handshake.channel<i9>, !handshake.channel<i9>, !handshake.channel<i32>) -> (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>)
    %1 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant1", value = 1 : i2} : <>, <i2>
    %2 = extsi %1 {handshake.bb = 0 : ui32, handshake.name = "extsi10"} : <i2> to <i6>
    %3 = mux %6#0 [%0#3, %103#0] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux4"} : <i1>, [<>, <>] to <>
    %4 = mux %6#1 [%0#2, %103#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux5"} : <i1>, [<>, <>] to <>
    %5 = init %117#2 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "init0"} : <i1>
    %6:2 = fork [2] %5 {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <i1>
    %7 = mux %index [%2, %trueResult_28] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i6>, <i6>] to <i6>
    %8 = buffer %7, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer3"} : <i6>
    %9 = buffer %8, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer4"} : <i6>
    %10:2 = fork [2] %9 {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <i6>
    %11 = extsi %10#1 {handshake.bb = 1 : ui32, handshake.name = "extsi11"} : <i6> to <i32>
    %result, %index = control_merge [%0#4, %trueResult_30]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %12:2 = fork [2] %result {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <>
    %13 = constant %12#0 {handshake.bb = 1 : ui32, handshake.name = "constant2", value = false} : <>, <i1>
    %14 = source {handshake.bb = 1 : ui32, handshake.name = "source0"} : <>
    %15 = constant %14 {handshake.bb = 1 : ui32, handshake.name = "constant7", value = -1 : i32} : <>, <i32>
    %16 = addi %11, %15 {handshake.bb = 1 : ui32, handshake.name = "addi0"} : <i32>
    %17 = extsi %13 {handshake.bb = 1 : ui32, handshake.name = "extsi9"} : <i1> to <i6>
    %18 = buffer %94, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer22"} : <>
    %trueResult, %falseResult = cond_br %101#3, %18 {handshake.bb = 2 : ui32, handshake.name = "cond_br12"} : <i1>, <>
    %19:2 = fork [2] %trueResult {handshake.bb = 2 : ui32, handshake.name = "fork4"} : <>
    %20 = buffer %3, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer1"} : <>
    %21 = mux %25#0 [%20, %19#1] {ftd.phi, handshake.bb = 2 : ui32, handshake.name = "mux6"} : <i1>, [<>, <>] to <>
    %22 = buffer %4, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer2"} : <>
    %23 = mux %25#1 [%22, %19#0] {ftd.phi, handshake.bb = 2 : ui32, handshake.name = "mux7"} : <i1>, [<>, <>] to <>
    %24 = init %101#2 {ftd.imerge, handshake.bb = 2 : ui32, handshake.name = "init2"} : <i1>
    %25:2 = fork [2] %24 {handshake.bb = 2 : ui32, handshake.name = "fork5"} : <i1>
    %26 = mux %43#1 [%17, %trueResult_18] {handshake.bb = 2 : ui32, handshake.name = "mux1"} : <i1>, [<i6>, <i6>] to <i6>
    %27 = buffer %26, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer7"} : <i6>
    %28:4 = fork [4] %27 {handshake.bb = 2 : ui32, handshake.name = "fork6"} : <i6>
    %29 = extsi %28#3 {handshake.bb = 2 : ui32, handshake.name = "extsi12"} : <i6> to <i7>
    %30 = trunci %28#0 {handshake.bb = 2 : ui32, handshake.name = "trunci0"} : <i6> to <i5>
    %31 = trunci %28#1 {handshake.bb = 2 : ui32, handshake.name = "trunci1"} : <i6> to <i5>
    %32 = trunci %28#2 {handshake.bb = 2 : ui32, handshake.name = "trunci2"} : <i6> to <i5>
    %33 = mux %43#0 [%10#0, %trueResult_20] {handshake.bb = 2 : ui32, handshake.name = "mux2"} : <i1>, [<i6>, <i6>] to <i6>
    %34 = buffer %33, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer8"} : <i6>
    %35 = buffer %34, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer9"} : <i6>
    %36:2 = fork [2] %35 {handshake.bb = 2 : ui32, handshake.name = "fork7"} : <i6>
    %37 = extsi %36#1 {handshake.bb = 2 : ui32, handshake.name = "extsi13"} : <i6> to <i32>
    %38:4 = fork [4] %37 {handshake.bb = 2 : ui32, handshake.name = "fork8"} : <i32>
    %39 = mux %43#2 [%16, %trueResult_22] {handshake.bb = 2 : ui32, handshake.name = "mux3"} : <i1>, [<i32>, <i32>] to <i32>
    %40 = buffer %39, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer10"} : <i32>
    %41 = buffer %40, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer11"} : <i32>
    %42:3 = fork [3] %41 {handshake.bb = 2 : ui32, handshake.name = "fork9"} : <i32>
    %result_6, %index_7 = control_merge [%12#1, %trueResult_24]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>, <>] to <>, <i1>
    %43:3 = fork [3] %index_7 {handshake.bb = 2 : ui32, handshake.name = "fork10"} : <i1>
    %44:2 = fork [2] %result_6 {handshake.bb = 2 : ui32, handshake.name = "fork11"} : <>
    %45 = constant %44#0 {handshake.bb = 2 : ui32, handshake.name = "constant4", value = 1 : i2} : <>, <i2>
    %46 = extsi %45 {handshake.bb = 2 : ui32, handshake.name = "extsi2"} : <i2> to <i32>
    %47 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %48 = constant %47 {handshake.bb = 2 : ui32, handshake.name = "constant5", value = 1 : i2} : <>, <i2>
    %49 = extsi %48 {handshake.bb = 2 : ui32, handshake.name = "extsi14"} : <i2> to <i7>
    %50 = source {handshake.bb = 2 : ui32, handshake.name = "source2"} : <>
    %51 = constant %50 {handshake.bb = 2 : ui32, handshake.name = "constant14", value = 20 : i6} : <>, <i6>
    %52 = extsi %51 {handshake.bb = 2 : ui32, handshake.name = "extsi15"} : <i6> to <i7>
    %53 = source {handshake.bb = 2 : ui32, handshake.name = "source3"} : <>
    %54 = constant %53 {handshake.bb = 2 : ui32, handshake.name = "constant15", value = 4 : i4} : <>, <i4>
    %55 = extsi %54 {handshake.bb = 2 : ui32, handshake.name = "extsi5"} : <i4> to <i32>
    %56:3 = fork [3] %55 {handshake.bb = 2 : ui32, handshake.name = "fork12"} : <i32>
    %57 = source {handshake.bb = 2 : ui32, handshake.name = "source4"} : <>
    %58 = constant %57 {handshake.bb = 2 : ui32, handshake.name = "constant16", value = 2 : i3} : <>, <i3>
    %59 = extsi %58 {handshake.bb = 2 : ui32, handshake.name = "extsi6"} : <i3> to <i32>
    %60:3 = fork [3] %59 {handshake.bb = 2 : ui32, handshake.name = "fork13"} : <i32>
    %addressResult, %dataResult = load[%32] %outputs_2 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i5>, <i32>, <i5>, <i32>
    %61:2 = fork [2] %dataResult {handshake.bb = 2 : ui32, handshake.name = "fork14"} : <i32>
    %62 = trunci %61#0 {handshake.bb = 2 : ui32, handshake.name = "trunci3"} : <i32> to <i9>
    %addressResult_8, %dataResult_9 = load[%31] %outputs {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i5>, <i32>, <i5>, <i32>
    %addressResult_10, %dataResult_11 = load[%30] %outputs_0 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load2"} : <i5>, <i32>, <i5>, <i32>
    %63 = shli %42#2, %60#0 {handshake.bb = 2 : ui32, handshake.name = "shli0"} : <i32>
    %64 = shli %42#1, %56#0 {handshake.bb = 2 : ui32, handshake.name = "shli1"} : <i32>
    %65 = buffer %63, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer13"} : <i32>
    %66 = buffer %64, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer14"} : <i32>
    %67 = addi %65, %66 {handshake.bb = 2 : ui32, handshake.name = "addi7"} : <i32>
    %68 = buffer %67, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer15"} : <i32>
    %69 = addi %dataResult_11, %68 {handshake.bb = 2 : ui32, handshake.name = "addi2"} : <i32>
    %70 = buffer %21, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer5"} : <>
    %71 = gate %69, %70 {handshake.bb = 2 : ui32, handshake.name = "gate0"} : <i32>, !handshake.control<> to <i32>
    %72 = trunci %71 {handshake.bb = 2 : ui32, handshake.name = "trunci4"} : <i32> to <i9>
    %addressResult_12, %dataResult_13 = load[%72] %outputs_4#0 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["store0", 0, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load3"} : <i9>, <i32>, <i9>, <i32>
    %73 = muli %dataResult_9, %dataResult_13 {handshake.bb = 2 : ui32, handshake.name = "muli0"} : <i32>
    %74 = shli %38#0, %60#1 {handshake.bb = 2 : ui32, handshake.name = "shli2"} : <i32>
    %75 = shli %38#1, %56#1 {handshake.bb = 2 : ui32, handshake.name = "shli3"} : <i32>
    %76 = buffer %74, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer16"} : <i32>
    %77 = buffer %75, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer17"} : <i32>
    %78 = addi %76, %77 {handshake.bb = 2 : ui32, handshake.name = "addi8"} : <i32>
    %79 = buffer %78, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer18"} : <i32>
    %80 = addi %61#1, %79 {handshake.bb = 2 : ui32, handshake.name = "addi3"} : <i32>
    %81 = buffer %23, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer6"} : <>
    %82 = gate %80, %81 {handshake.bb = 2 : ui32, handshake.name = "gate1"} : <i32>, !handshake.control<> to <i32>
    %83 = trunci %82 {handshake.bb = 2 : ui32, handshake.name = "trunci5"} : <i32> to <i9>
    %addressResult_14, %dataResult_15 = load[%83] %outputs_4#1 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["store0", 0, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load4"} : <i9>, <i32>, <i9>, <i32>
    %84 = addi %dataResult_15, %73 {handshake.bb = 2 : ui32, handshake.name = "addi1"} : <i32>
    %85 = shli %38#2, %60#2 {handshake.bb = 2 : ui32, handshake.name = "shli4"} : <i32>
    %86 = buffer %85, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer19"} : <i32>
    %87 = trunci %86 {handshake.bb = 2 : ui32, handshake.name = "trunci6"} : <i32> to <i9>
    %88 = shli %38#3, %56#2 {handshake.bb = 2 : ui32, handshake.name = "shli5"} : <i32>
    %89 = buffer %88, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer20"} : <i32>
    %90 = trunci %89 {handshake.bb = 2 : ui32, handshake.name = "trunci7"} : <i32> to <i9>
    %91 = addi %87, %90 {handshake.bb = 2 : ui32, handshake.name = "addi9"} : <i9>
    %92 = buffer %91, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer21"} : <i9>
    %93 = addi %62, %92 {handshake.bb = 2 : ui32, handshake.name = "addi4"} : <i9>
    %94 = buffer %doneResult, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer0"} : <>
    %addressResult_16, %dataResult_17, %doneResult = store[%93] %84 %outputs_4#2 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["load3", 0, false], ["load4", 0, false], ["store0", 0, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store0"} : <i9>, <i32>, <>, <i9>, <i32>, <>
    %95 = addi %29, %49 {handshake.bb = 2 : ui32, handshake.name = "addi5"} : <i7>
    %96 = buffer %95, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer23"} : <i7>
    %97:2 = fork [2] %96 {handshake.bb = 2 : ui32, handshake.name = "fork15"} : <i7>
    %98 = trunci %97#0 {handshake.bb = 2 : ui32, handshake.name = "trunci8"} : <i7> to <i6>
    %99 = cmpi ult, %97#1, %52 {handshake.bb = 2 : ui32, handshake.name = "cmpi0"} : <i7>
    %100 = buffer %99, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer24"} : <i1>
    %101:6 = fork [6] %100 {handshake.bb = 2 : ui32, handshake.name = "fork16"} : <i1>
    %trueResult_18, %falseResult_19 = cond_br %101#0, %98 {handshake.bb = 2 : ui32, handshake.name = "cond_br2"} : <i1>, <i6>
    sink %falseResult_19 {handshake.name = "sink0"} : <i6>
    %trueResult_20, %falseResult_21 = cond_br %101#1, %36#0 {handshake.bb = 2 : ui32, handshake.name = "cond_br3"} : <i1>, <i6>
    %trueResult_22, %falseResult_23 = cond_br %101#4, %42#0 {handshake.bb = 2 : ui32, handshake.name = "cond_br4"} : <i1>, <i32>
    sink %falseResult_23 {handshake.name = "sink1"} : <i32>
    %102 = buffer %44#1, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer12"} : <>
    %trueResult_24, %falseResult_25 = cond_br %101#5, %102 {handshake.bb = 2 : ui32, handshake.name = "cond_br5"} : <i1>, <>
    %trueResult_26, %falseResult_27 = cond_br %117#1, %falseResult {handshake.bb = 3 : ui32, handshake.name = "cond_br13"} : <i1>, <>
    sink %falseResult_27 {handshake.name = "sink2"} : <>
    %103:2 = fork [2] %trueResult_26 {handshake.bb = 3 : ui32, handshake.name = "fork17"} : <>
    %104 = extsi %falseResult_21 {handshake.bb = 3 : ui32, handshake.name = "extsi16"} : <i6> to <i7>
    %105 = source {handshake.bb = 3 : ui32, handshake.name = "source5"} : <>
    %106 = constant %105 {handshake.bb = 3 : ui32, handshake.name = "constant17", value = 1 : i2} : <>, <i2>
    %107 = extsi %106 {handshake.bb = 3 : ui32, handshake.name = "extsi17"} : <i2> to <i7>
    %108 = source {handshake.bb = 3 : ui32, handshake.name = "source6"} : <>
    %109 = constant %108 {handshake.bb = 3 : ui32, handshake.name = "constant18", value = 20 : i6} : <>, <i6>
    %110 = extsi %109 {handshake.bb = 3 : ui32, handshake.name = "extsi18"} : <i6> to <i7>
    %111 = addi %104, %107 {handshake.bb = 3 : ui32, handshake.name = "addi6"} : <i7>
    %112 = buffer %111, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer25"} : <i7>
    %113:2 = fork [2] %112 {handshake.bb = 3 : ui32, handshake.name = "fork18"} : <i7>
    %114 = trunci %113#0 {handshake.bb = 3 : ui32, handshake.name = "trunci9"} : <i7> to <i6>
    %115 = cmpi ult, %113#1, %110 {handshake.bb = 3 : ui32, handshake.name = "cmpi1"} : <i7>
    %116 = buffer %115, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer26"} : <i1>
    %117:4 = fork [4] %116 {handshake.bb = 3 : ui32, handshake.name = "fork19"} : <i1>
    %trueResult_28, %falseResult_29 = cond_br %117#0, %114 {handshake.bb = 3 : ui32, handshake.name = "cond_br6"} : <i1>, <i6>
    sink %falseResult_29 {handshake.name = "sink4"} : <i6>
    %trueResult_30, %falseResult_31 = cond_br %117#3, %falseResult_25 {handshake.bb = 3 : ui32, handshake.name = "cond_br7"} : <i1>, <>
    %118:4 = fork [4] %falseResult_31 {handshake.bb = 4 : ui32, handshake.name = "fork20"} : <>
    end {handshake.bb = 4 : ui32, handshake.name = "end0"} %memEnd_5, %memEnd_3, %memEnd_1, %memEnd, %0#1 : <>, <>, <>, <>, <>
  }
}

