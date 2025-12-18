module {
  handshake.func @matrix_power(%arg0: memref<400xi32>, %arg1: memref<20xi32>, %arg2: memref<20xi32>, %arg3: memref<20xi32>, %arg4: !handshake.control<>, %arg5: !handshake.control<>, %arg6: !handshake.control<>, %arg7: !handshake.control<>, %arg8: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["mat", "row", "col", "a", "mat_start", "row_start", "col_start", "a_start", "start"], cfg.edges = "[0,1][2,2,3,cmpi0][1,2][3,1,4,cmpi1]", resNames = ["mat_end", "row_end", "col_end", "a_end", "end"]} {
    %0:5 = fork [5] %arg8 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork0"} : <>
    %outputs, %memEnd = mem_controller[%arg3 : memref<20xi32>] %arg7 (%addressResult_8) %125#3 {connectedBlocks = [2 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller3"} :    (!handshake.channel<i5>) -> !handshake.channel<i32>
    %outputs_0, %memEnd_1 = mem_controller[%arg2 : memref<20xi32>] %arg6 (%addressResult_10) %125#2 {connectedBlocks = [2 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller4"} :    (!handshake.channel<i5>) -> !handshake.channel<i32>
    %outputs_2, %memEnd_3 = mem_controller[%arg1 : memref<20xi32>] %arg5 (%addressResult) %125#1 {connectedBlocks = [2 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller5"} :    (!handshake.channel<i5>) -> !handshake.channel<i32>
    %outputs_4:3, %memEnd_5 = mem_controller[%arg0 : memref<400xi32>] %arg4 (%52, %addressResult_12, %addressResult_14, %addressResult_16, %dataResult_17) %125#0 {connectedBlocks = [2 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "5": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "6": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller6"} :    (!handshake.channel<i32>, !handshake.channel<i9>, !handshake.channel<i9>, !handshake.channel<i9>, !handshake.channel<i32>) -> (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>)
    %1 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant1", value = 1 : i2} : <>, <i2>
    %2 = br %1 {handshake.bb = 0 : ui32, handshake.name = "br2"} : <i2>
    %3 = extsi %2 {handshake.bb = 0 : ui32, handshake.name = "extsi10"} : <i2> to <i6>
    %4 = br %0#4 {handshake.bb = 0 : ui32, handshake.name = "br3"} : <>
    %5 = mux %8#0 [%0#3, %109#0] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux4"} : <i1>, [<>, <>] to <>
    %6 = mux %8#1 [%0#2, %109#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux5"} : <i1>, [<>, <>] to <>
    %7 = init %124#2 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "init0"} : <i1>
    %8:2 = fork [2] %7 {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <i1>
    %9 = mux %index [%3, %trueResult_30] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i6>, <i6>] to <i6>
    %10 = buffer %9, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer3"} : <i6>
    %11 = buffer %10, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer4"} : <i6>
    %12:2 = fork [2] %11 {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <i6>
    %13 = extsi %12#1 {handshake.bb = 1 : ui32, handshake.name = "extsi11"} : <i6> to <i32>
    %result, %index = control_merge [%4, %trueResult_32]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %14:2 = fork [2] %result {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <>
    %15 = constant %14#0 {handshake.bb = 1 : ui32, handshake.name = "constant2", value = false} : <>, <i1>
    %16 = source {handshake.bb = 1 : ui32, handshake.name = "source0"} : <>
    %17 = constant %16 {handshake.bb = 1 : ui32, handshake.name = "constant7", value = -1 : i32} : <>, <i32>
    %18 = addi %13, %17 {handshake.bb = 1 : ui32, handshake.name = "addi0"} : <i32>
    %19 = br %15 {handshake.bb = 1 : ui32, handshake.name = "br4"} : <i1>
    %20 = extsi %19 {handshake.bb = 1 : ui32, handshake.name = "extsi9"} : <i1> to <i6>
    %21 = br %12#0 {handshake.bb = 1 : ui32, handshake.name = "br5"} : <i6>
    %22 = br %18 {handshake.bb = 1 : ui32, handshake.name = "br6"} : <i32>
    %23 = br %14#1 {handshake.bb = 1 : ui32, handshake.name = "br7"} : <>
    %24 = buffer %100, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer22"} : <>
    %trueResult, %falseResult = cond_br %107#3, %24 {handshake.bb = 2 : ui32, handshake.name = "cond_br12"} : <i1>, <>
    %25:2 = fork [2] %trueResult {handshake.bb = 2 : ui32, handshake.name = "fork4"} : <>
    %26 = buffer %5, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer1"} : <>
    %27 = mux %31#0 [%26, %25#1] {ftd.phi, handshake.bb = 2 : ui32, handshake.name = "mux6"} : <i1>, [<>, <>] to <>
    %28 = buffer %6, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer2"} : <>
    %29 = mux %31#1 [%28, %25#0] {ftd.phi, handshake.bb = 2 : ui32, handshake.name = "mux7"} : <i1>, [<>, <>] to <>
    %30 = init %107#2 {ftd.imerge, handshake.bb = 2 : ui32, handshake.name = "init2"} : <i1>
    %31:2 = fork [2] %30 {handshake.bb = 2 : ui32, handshake.name = "fork5"} : <i1>
    %32 = mux %49#1 [%20, %trueResult_18] {handshake.bb = 2 : ui32, handshake.name = "mux1"} : <i1>, [<i6>, <i6>] to <i6>
    %33 = buffer %32, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer7"} : <i6>
    %34:4 = fork [4] %33 {handshake.bb = 2 : ui32, handshake.name = "fork6"} : <i6>
    %35 = extsi %34#3 {handshake.bb = 2 : ui32, handshake.name = "extsi12"} : <i6> to <i7>
    %36 = trunci %34#0 {handshake.bb = 2 : ui32, handshake.name = "trunci0"} : <i6> to <i5>
    %37 = trunci %34#1 {handshake.bb = 2 : ui32, handshake.name = "trunci1"} : <i6> to <i5>
    %38 = trunci %34#2 {handshake.bb = 2 : ui32, handshake.name = "trunci2"} : <i6> to <i5>
    %39 = mux %49#0 [%21, %trueResult_20] {handshake.bb = 2 : ui32, handshake.name = "mux2"} : <i1>, [<i6>, <i6>] to <i6>
    %40 = buffer %39, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer8"} : <i6>
    %41 = buffer %40, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer9"} : <i6>
    %42:2 = fork [2] %41 {handshake.bb = 2 : ui32, handshake.name = "fork7"} : <i6>
    %43 = extsi %42#1 {handshake.bb = 2 : ui32, handshake.name = "extsi13"} : <i6> to <i32>
    %44:4 = fork [4] %43 {handshake.bb = 2 : ui32, handshake.name = "fork8"} : <i32>
    %45 = mux %49#2 [%22, %trueResult_22] {handshake.bb = 2 : ui32, handshake.name = "mux3"} : <i1>, [<i32>, <i32>] to <i32>
    %46 = buffer %45, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer10"} : <i32>
    %47 = buffer %46, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer11"} : <i32>
    %48:3 = fork [3] %47 {handshake.bb = 2 : ui32, handshake.name = "fork9"} : <i32>
    %result_6, %index_7 = control_merge [%23, %trueResult_24]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>, <>] to <>, <i1>
    %49:3 = fork [3] %index_7 {handshake.bb = 2 : ui32, handshake.name = "fork10"} : <i1>
    %50:2 = fork [2] %result_6 {handshake.bb = 2 : ui32, handshake.name = "fork11"} : <>
    %51 = constant %50#0 {handshake.bb = 2 : ui32, handshake.name = "constant4", value = 1 : i2} : <>, <i2>
    %52 = extsi %51 {handshake.bb = 2 : ui32, handshake.name = "extsi2"} : <i2> to <i32>
    %53 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %54 = constant %53 {handshake.bb = 2 : ui32, handshake.name = "constant5", value = 1 : i2} : <>, <i2>
    %55 = extsi %54 {handshake.bb = 2 : ui32, handshake.name = "extsi14"} : <i2> to <i7>
    %56 = source {handshake.bb = 2 : ui32, handshake.name = "source2"} : <>
    %57 = constant %56 {handshake.bb = 2 : ui32, handshake.name = "constant14", value = 20 : i6} : <>, <i6>
    %58 = extsi %57 {handshake.bb = 2 : ui32, handshake.name = "extsi15"} : <i6> to <i7>
    %59 = source {handshake.bb = 2 : ui32, handshake.name = "source3"} : <>
    %60 = constant %59 {handshake.bb = 2 : ui32, handshake.name = "constant15", value = 4 : i4} : <>, <i4>
    %61 = extsi %60 {handshake.bb = 2 : ui32, handshake.name = "extsi5"} : <i4> to <i32>
    %62:3 = fork [3] %61 {handshake.bb = 2 : ui32, handshake.name = "fork12"} : <i32>
    %63 = source {handshake.bb = 2 : ui32, handshake.name = "source4"} : <>
    %64 = constant %63 {handshake.bb = 2 : ui32, handshake.name = "constant16", value = 2 : i3} : <>, <i3>
    %65 = extsi %64 {handshake.bb = 2 : ui32, handshake.name = "extsi6"} : <i3> to <i32>
    %66:3 = fork [3] %65 {handshake.bb = 2 : ui32, handshake.name = "fork13"} : <i32>
    %addressResult, %dataResult = load[%38] %outputs_2 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i5>, <i32>, <i5>, <i32>
    %67:2 = fork [2] %dataResult {handshake.bb = 2 : ui32, handshake.name = "fork14"} : <i32>
    %68 = trunci %67#0 {handshake.bb = 2 : ui32, handshake.name = "trunci3"} : <i32> to <i9>
    %addressResult_8, %dataResult_9 = load[%37] %outputs {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i5>, <i32>, <i5>, <i32>
    %addressResult_10, %dataResult_11 = load[%36] %outputs_0 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load2"} : <i5>, <i32>, <i5>, <i32>
    %69 = shli %48#2, %66#0 {handshake.bb = 2 : ui32, handshake.name = "shli0"} : <i32>
    %70 = shli %48#1, %62#0 {handshake.bb = 2 : ui32, handshake.name = "shli1"} : <i32>
    %71 = buffer %69, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer13"} : <i32>
    %72 = buffer %70, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer14"} : <i32>
    %73 = addi %71, %72 {handshake.bb = 2 : ui32, handshake.name = "addi7"} : <i32>
    %74 = buffer %73, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer15"} : <i32>
    %75 = addi %dataResult_11, %74 {handshake.bb = 2 : ui32, handshake.name = "addi2"} : <i32>
    %76 = buffer %27, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer5"} : <>
    %77 = gate %75, %76 {handshake.bb = 2 : ui32, handshake.name = "gate0"} : <i32>, !handshake.control<> to <i32>
    %78 = trunci %77 {handshake.bb = 2 : ui32, handshake.name = "trunci4"} : <i32> to <i9>
    %addressResult_12, %dataResult_13 = load[%78] %outputs_4#0 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["store0", 0, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load3"} : <i9>, <i32>, <i9>, <i32>
    %79 = muli %dataResult_9, %dataResult_13 {handshake.bb = 2 : ui32, handshake.name = "muli0"} : <i32>
    %80 = shli %44#0, %66#1 {handshake.bb = 2 : ui32, handshake.name = "shli2"} : <i32>
    %81 = shli %44#1, %62#1 {handshake.bb = 2 : ui32, handshake.name = "shli3"} : <i32>
    %82 = buffer %80, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer16"} : <i32>
    %83 = buffer %81, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer17"} : <i32>
    %84 = addi %82, %83 {handshake.bb = 2 : ui32, handshake.name = "addi8"} : <i32>
    %85 = buffer %84, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer18"} : <i32>
    %86 = addi %67#1, %85 {handshake.bb = 2 : ui32, handshake.name = "addi3"} : <i32>
    %87 = buffer %29, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer6"} : <>
    %88 = gate %86, %87 {handshake.bb = 2 : ui32, handshake.name = "gate1"} : <i32>, !handshake.control<> to <i32>
    %89 = trunci %88 {handshake.bb = 2 : ui32, handshake.name = "trunci5"} : <i32> to <i9>
    %addressResult_14, %dataResult_15 = load[%89] %outputs_4#1 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["store0", 0, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load4"} : <i9>, <i32>, <i9>, <i32>
    %90 = addi %dataResult_15, %79 {handshake.bb = 2 : ui32, handshake.name = "addi1"} : <i32>
    %91 = shli %44#2, %66#2 {handshake.bb = 2 : ui32, handshake.name = "shli4"} : <i32>
    %92 = buffer %91, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer19"} : <i32>
    %93 = trunci %92 {handshake.bb = 2 : ui32, handshake.name = "trunci6"} : <i32> to <i9>
    %94 = shli %44#3, %62#2 {handshake.bb = 2 : ui32, handshake.name = "shli5"} : <i32>
    %95 = buffer %94, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer20"} : <i32>
    %96 = trunci %95 {handshake.bb = 2 : ui32, handshake.name = "trunci7"} : <i32> to <i9>
    %97 = addi %93, %96 {handshake.bb = 2 : ui32, handshake.name = "addi9"} : <i9>
    %98 = buffer %97, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer21"} : <i9>
    %99 = addi %68, %98 {handshake.bb = 2 : ui32, handshake.name = "addi4"} : <i9>
    %100 = buffer %doneResult, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer0"} : <>
    %addressResult_16, %dataResult_17, %doneResult = store[%99] %90 %outputs_4#2 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["load3", 0, false], ["load4", 0, false], ["store0", 0, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store0"} : <i9>, <i32>, <>, <i9>, <i32>, <>
    %101 = addi %35, %55 {handshake.bb = 2 : ui32, handshake.name = "addi5"} : <i7>
    %102 = buffer %101, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer23"} : <i7>
    %103:2 = fork [2] %102 {handshake.bb = 2 : ui32, handshake.name = "fork15"} : <i7>
    %104 = trunci %103#0 {handshake.bb = 2 : ui32, handshake.name = "trunci8"} : <i7> to <i6>
    %105 = cmpi ult, %103#1, %58 {handshake.bb = 2 : ui32, handshake.name = "cmpi0"} : <i7>
    %106 = buffer %105, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer24"} : <i1>
    %107:6 = fork [6] %106 {handshake.bb = 2 : ui32, handshake.name = "fork16"} : <i1>
    %trueResult_18, %falseResult_19 = cond_br %107#0, %104 {handshake.bb = 2 : ui32, handshake.name = "cond_br2"} : <i1>, <i6>
    sink %falseResult_19 {handshake.name = "sink0"} : <i6>
    %trueResult_20, %falseResult_21 = cond_br %107#1, %42#0 {handshake.bb = 2 : ui32, handshake.name = "cond_br3"} : <i1>, <i6>
    %trueResult_22, %falseResult_23 = cond_br %107#4, %48#0 {handshake.bb = 2 : ui32, handshake.name = "cond_br4"} : <i1>, <i32>
    sink %falseResult_23 {handshake.name = "sink1"} : <i32>
    %108 = buffer %50#1, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer12"} : <>
    %trueResult_24, %falseResult_25 = cond_br %107#5, %108 {handshake.bb = 2 : ui32, handshake.name = "cond_br5"} : <i1>, <>
    %trueResult_26, %falseResult_27 = cond_br %124#1, %falseResult {handshake.bb = 3 : ui32, handshake.name = "cond_br13"} : <i1>, <>
    sink %falseResult_27 {handshake.name = "sink2"} : <>
    %109:2 = fork [2] %trueResult_26 {handshake.bb = 3 : ui32, handshake.name = "fork17"} : <>
    %110 = merge %falseResult_21 {handshake.bb = 3 : ui32, handshake.name = "merge0"} : <i6>
    %111 = extsi %110 {handshake.bb = 3 : ui32, handshake.name = "extsi16"} : <i6> to <i7>
    %result_28, %index_29 = control_merge [%falseResult_25]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>] to <>, <i1>
    sink %index_29 {handshake.name = "sink3"} : <i1>
    %112 = source {handshake.bb = 3 : ui32, handshake.name = "source5"} : <>
    %113 = constant %112 {handshake.bb = 3 : ui32, handshake.name = "constant17", value = 1 : i2} : <>, <i2>
    %114 = extsi %113 {handshake.bb = 3 : ui32, handshake.name = "extsi17"} : <i2> to <i7>
    %115 = source {handshake.bb = 3 : ui32, handshake.name = "source6"} : <>
    %116 = constant %115 {handshake.bb = 3 : ui32, handshake.name = "constant18", value = 20 : i6} : <>, <i6>
    %117 = extsi %116 {handshake.bb = 3 : ui32, handshake.name = "extsi18"} : <i6> to <i7>
    %118 = addi %111, %114 {handshake.bb = 3 : ui32, handshake.name = "addi6"} : <i7>
    %119 = buffer %118, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer25"} : <i7>
    %120:2 = fork [2] %119 {handshake.bb = 3 : ui32, handshake.name = "fork18"} : <i7>
    %121 = trunci %120#0 {handshake.bb = 3 : ui32, handshake.name = "trunci9"} : <i7> to <i6>
    %122 = cmpi ult, %120#1, %117 {handshake.bb = 3 : ui32, handshake.name = "cmpi1"} : <i7>
    %123 = buffer %122, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer26"} : <i1>
    %124:4 = fork [4] %123 {handshake.bb = 3 : ui32, handshake.name = "fork19"} : <i1>
    %trueResult_30, %falseResult_31 = cond_br %124#0, %121 {handshake.bb = 3 : ui32, handshake.name = "cond_br6"} : <i1>, <i6>
    sink %falseResult_31 {handshake.name = "sink4"} : <i6>
    %trueResult_32, %falseResult_33 = cond_br %124#3, %result_28 {handshake.bb = 3 : ui32, handshake.name = "cond_br7"} : <i1>, <>
    %result_34, %index_35 = control_merge [%falseResult_33]  {handshake.bb = 4 : ui32, handshake.name = "control_merge3"} : [<>] to <>, <i1>
    sink %index_35 {handshake.name = "sink5"} : <i1>
    %125:4 = fork [4] %result_34 {handshake.bb = 4 : ui32, handshake.name = "fork20"} : <>
    end {handshake.bb = 4 : ui32, handshake.name = "end0"} %memEnd_5, %memEnd_3, %memEnd_1, %memEnd, %0#1 : <>, <>, <>, <>, <>
  }
}

