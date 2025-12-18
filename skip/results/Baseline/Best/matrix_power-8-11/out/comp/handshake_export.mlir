module {
  handshake.func @matrix_power(%arg0: memref<400xi32>, %arg1: memref<20xi32>, %arg2: memref<20xi32>, %arg3: memref<20xi32>, %arg4: !handshake.control<>, %arg5: !handshake.control<>, %arg6: !handshake.control<>, %arg7: !handshake.control<>, %arg8: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["mat", "row", "col", "a", "mat_start", "row_start", "col_start", "a_start", "start"], resNames = ["mat_end", "row_end", "col_end", "a_end", "end"]} {
    %0:3 = fork [3] %arg8 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork0"} : <>
    %outputs, %memEnd = mem_controller[%arg3 : memref<20xi32>] %arg7 (%addressResult_6) %108#3 {connectedBlocks = [2 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller3"} :    (!handshake.channel<i5>) -> !handshake.channel<i32>
    %outputs_0, %memEnd_1 = mem_controller[%arg2 : memref<20xi32>] %arg6 (%addressResult_8) %108#2 {connectedBlocks = [2 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller4"} :    (!handshake.channel<i5>) -> !handshake.channel<i32>
    %outputs_2, %memEnd_3 = mem_controller[%arg1 : memref<20xi32>] %arg5 (%addressResult) %108#1 {connectedBlocks = [2 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller5"} :    (!handshake.channel<i5>) -> !handshake.channel<i32>
    %1:3 = lsq[%arg0 : memref<400xi32>] (%arg4, %33#0, %addressResult_10, %addressResult_12, %addressResult_14, %dataResult_15, %108#0)  {groupSizes = [3 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "5": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "6": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "lsq1"} : (!handshake.control<>, !handshake.control<>, !handshake.channel<i9>, !handshake.channel<i9>, !handshake.channel<i9>, !handshake.channel<i32>, !handshake.control<>) -> (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>)
    %2 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant0", value = 1 : i2} : <>, <i2>
    %3 = extsi %2 {handshake.bb = 0 : ui32, handshake.name = "extsi9"} : <i2> to <i6>
    %4 = mux %index [%3, %trueResult_22] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i6>, <i6>] to <i6>
    %5 = buffer %4, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer0"} : <i6>
    %6 = buffer %5, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer1"} : <i6>
    %7:2 = fork [2] %6 {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <i6>
    %8 = extsi %7#1 {handshake.bb = 1 : ui32, handshake.name = "extsi10"} : <i6> to <i32>
    %result, %index = control_merge [%0#2, %trueResult_24]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %9:2 = fork [2] %result {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <>
    %10 = constant %9#0 {handshake.bb = 1 : ui32, handshake.name = "constant1", value = false} : <>, <i1>
    %11 = source {handshake.bb = 1 : ui32, handshake.name = "source0"} : <>
    %12 = constant %11 {handshake.bb = 1 : ui32, handshake.name = "constant7", value = -1 : i32} : <>, <i32>
    %13 = addi %8, %12 {handshake.bb = 1 : ui32, handshake.name = "addi0"} : <i32>
    %14 = extsi %10 {handshake.bb = 1 : ui32, handshake.name = "extsi8"} : <i1> to <i6>
    %15 = mux %32#1 [%14, %trueResult] {handshake.bb = 2 : ui32, handshake.name = "mux1"} : <i1>, [<i6>, <i6>] to <i6>
    %16 = buffer %15, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer2"} : <i6>
    %17:4 = fork [4] %16 {handshake.bb = 2 : ui32, handshake.name = "fork3"} : <i6>
    %18 = extsi %17#3 {handshake.bb = 2 : ui32, handshake.name = "extsi11"} : <i6> to <i7>
    %19 = trunci %17#0 {handshake.bb = 2 : ui32, handshake.name = "trunci0"} : <i6> to <i5>
    %20 = trunci %17#1 {handshake.bb = 2 : ui32, handshake.name = "trunci1"} : <i6> to <i5>
    %21 = trunci %17#2 {handshake.bb = 2 : ui32, handshake.name = "trunci2"} : <i6> to <i5>
    %22 = mux %32#0 [%7#0, %trueResult_16] {handshake.bb = 2 : ui32, handshake.name = "mux2"} : <i1>, [<i6>, <i6>] to <i6>
    %23 = buffer %22, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer3"} : <i6>
    %24 = buffer %23, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer4"} : <i6>
    %25:2 = fork [2] %24 {handshake.bb = 2 : ui32, handshake.name = "fork4"} : <i6>
    %26 = extsi %25#1 {handshake.bb = 2 : ui32, handshake.name = "extsi12"} : <i6> to <i32>
    %27:4 = fork [4] %26 {handshake.bb = 2 : ui32, handshake.name = "fork5"} : <i32>
    %28 = mux %32#2 [%13, %trueResult_18] {handshake.bb = 2 : ui32, handshake.name = "mux3"} : <i1>, [<i32>, <i32>] to <i32>
    %29 = buffer %28, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer5"} : <i32>
    %30 = buffer %29, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer6"} : <i32>
    %31:3 = fork [3] %30 {handshake.bb = 2 : ui32, handshake.name = "fork6"} : <i32>
    %result_4, %index_5 = control_merge [%9#1, %trueResult_20]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>, <>] to <>, <i1>
    %32:3 = fork [3] %index_5 {handshake.bb = 2 : ui32, handshake.name = "fork7"} : <i1>
    %33:2 = fork [2] %result_4 {handshake.bb = 2 : ui32, handshake.name = "fork8"} : <>
    %34 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %35 = constant %34 {handshake.bb = 2 : ui32, handshake.name = "constant2", value = 1 : i2} : <>, <i2>
    %36 = extsi %35 {handshake.bb = 2 : ui32, handshake.name = "extsi13"} : <i2> to <i7>
    %37 = source {handshake.bb = 2 : ui32, handshake.name = "source2"} : <>
    %38 = constant %37 {handshake.bb = 2 : ui32, handshake.name = "constant4", value = 20 : i6} : <>, <i6>
    %39 = extsi %38 {handshake.bb = 2 : ui32, handshake.name = "extsi14"} : <i6> to <i7>
    %40 = source {handshake.bb = 2 : ui32, handshake.name = "source3"} : <>
    %41 = constant %40 {handshake.bb = 2 : ui32, handshake.name = "constant5", value = 4 : i4} : <>, <i4>
    %42 = extsi %41 {handshake.bb = 2 : ui32, handshake.name = "extsi4"} : <i4> to <i32>
    %43:3 = fork [3] %42 {handshake.bb = 2 : ui32, handshake.name = "fork9"} : <i32>
    %44 = source {handshake.bb = 2 : ui32, handshake.name = "source4"} : <>
    %45 = constant %44 {handshake.bb = 2 : ui32, handshake.name = "constant14", value = 2 : i3} : <>, <i3>
    %46 = extsi %45 {handshake.bb = 2 : ui32, handshake.name = "extsi5"} : <i3> to <i32>
    %47:3 = fork [3] %46 {handshake.bb = 2 : ui32, handshake.name = "fork10"} : <i32>
    %addressResult, %dataResult = load[%21] %outputs_2 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i5>, <i32>, <i5>, <i32>
    %48:2 = fork [2] %dataResult {handshake.bb = 2 : ui32, handshake.name = "fork11"} : <i32>
    %49 = trunci %48#0 {handshake.bb = 2 : ui32, handshake.name = "trunci3"} : <i32> to <i9>
    %50 = trunci %48#1 {handshake.bb = 2 : ui32, handshake.name = "trunci4"} : <i32> to <i9>
    %addressResult_6, %dataResult_7 = load[%20] %outputs {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i5>, <i32>, <i5>, <i32>
    %addressResult_8, %dataResult_9 = load[%19] %outputs_0 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load2"} : <i5>, <i32>, <i5>, <i32>
    %51 = trunci %dataResult_9 {handshake.bb = 2 : ui32, handshake.name = "trunci5"} : <i32> to <i9>
    %52 = shli %31#2, %47#0 {handshake.bb = 2 : ui32, handshake.name = "shli0"} : <i32>
    %53 = buffer %52, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer9"} : <i32>
    %54 = trunci %53 {handshake.bb = 2 : ui32, handshake.name = "trunci6"} : <i32> to <i9>
    %55 = shli %31#1, %43#0 {handshake.bb = 2 : ui32, handshake.name = "shli1"} : <i32>
    %56 = buffer %55, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer10"} : <i32>
    %57 = trunci %56 {handshake.bb = 2 : ui32, handshake.name = "trunci7"} : <i32> to <i9>
    %58 = addi %54, %57 {handshake.bb = 2 : ui32, handshake.name = "addi7"} : <i9>
    %59 = buffer %58, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer11"} : <i9>
    %60 = addi %51, %59 {handshake.bb = 2 : ui32, handshake.name = "addi2"} : <i9>
    %61 = buffer %60, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer12"} : <i9>
    %addressResult_10, %dataResult_11 = load[%61] %1#0 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["store0", 0]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 0>, handshake.name = "load3"} : <i9>, <i32>, <i9>, <i32>
    %62 = muli %dataResult_7, %dataResult_11 {handshake.bb = 2 : ui32, handshake.name = "muli0"} : <i32>
    %63 = shli %27#0, %47#1 {handshake.bb = 2 : ui32, handshake.name = "shli2"} : <i32>
    %64 = buffer %63, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer13"} : <i32>
    %65 = trunci %64 {handshake.bb = 2 : ui32, handshake.name = "trunci8"} : <i32> to <i9>
    %66 = shli %27#1, %43#1 {handshake.bb = 2 : ui32, handshake.name = "shli3"} : <i32>
    %67 = buffer %66, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer14"} : <i32>
    %68 = trunci %67 {handshake.bb = 2 : ui32, handshake.name = "trunci9"} : <i32> to <i9>
    %69 = addi %65, %68 {handshake.bb = 2 : ui32, handshake.name = "addi8"} : <i9>
    %70 = buffer %69, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer15"} : <i9>
    %71 = addi %49, %70 {handshake.bb = 2 : ui32, handshake.name = "addi3"} : <i9>
    %72 = buffer %71, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer16"} : <i9>
    %addressResult_12, %dataResult_13 = load[%72] %1#1 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["store0", 0]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 0>, handshake.name = "load4"} : <i9>, <i32>, <i9>, <i32>
    %73 = addi %dataResult_13, %62 {handshake.bb = 2 : ui32, handshake.name = "addi1"} : <i32>
    %74 = shli %27#2, %47#2 {handshake.bb = 2 : ui32, handshake.name = "shli4"} : <i32>
    %75 = buffer %74, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer18"} : <i32>
    %76 = trunci %75 {handshake.bb = 2 : ui32, handshake.name = "trunci10"} : <i32> to <i9>
    %77 = shli %27#3, %43#2 {handshake.bb = 2 : ui32, handshake.name = "shli5"} : <i32>
    %78 = buffer %77, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer19"} : <i32>
    %79 = trunci %78 {handshake.bb = 2 : ui32, handshake.name = "trunci11"} : <i32> to <i9>
    %80 = addi %76, %79 {handshake.bb = 2 : ui32, handshake.name = "addi9"} : <i9>
    %81 = buffer %80, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer20"} : <i9>
    %82 = addi %50, %81 {handshake.bb = 2 : ui32, handshake.name = "addi4"} : <i9>
    %83 = buffer %73, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer17"} : <i32>
    %84 = buffer %82, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer21"} : <i9>
    %addressResult_14, %dataResult_15 = store[%84] %83 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["load3", 0], ["load4", 0], ["store0", 0]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 0>, handshake.name = "store0"} : <i9>, <i32>, <i9>, <i32>
    %85 = addi %18, %36 {handshake.bb = 2 : ui32, handshake.name = "addi5"} : <i7>
    %86 = buffer %85, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer22"} : <i7>
    %87:2 = fork [2] %86 {handshake.bb = 2 : ui32, handshake.name = "fork12"} : <i7>
    %88 = trunci %87#0 {handshake.bb = 2 : ui32, handshake.name = "trunci12"} : <i7> to <i6>
    %89 = cmpi ult, %87#1, %39 {handshake.bb = 2 : ui32, handshake.name = "cmpi0"} : <i7>
    %90 = buffer %89, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer23"} : <i1>
    %91:4 = fork [4] %90 {handshake.bb = 2 : ui32, handshake.name = "fork13"} : <i1>
    %trueResult, %falseResult = cond_br %91#0, %88 {handshake.bb = 2 : ui32, handshake.name = "cond_br2"} : <i1>, <i6>
    sink %falseResult {handshake.name = "sink0"} : <i6>
    %trueResult_16, %falseResult_17 = cond_br %91#1, %25#0 {handshake.bb = 2 : ui32, handshake.name = "cond_br3"} : <i1>, <i6>
    %trueResult_18, %falseResult_19 = cond_br %91#2, %31#0 {handshake.bb = 2 : ui32, handshake.name = "cond_br4"} : <i1>, <i32>
    sink %falseResult_19 {handshake.name = "sink1"} : <i32>
    %92 = buffer %33#1, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer7"} : <>
    %93 = buffer %92, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer8"} : <>
    %trueResult_20, %falseResult_21 = cond_br %91#3, %93 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [1,inf], [0,inf], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "cond_br5"} : <i1>, <>
    %94 = extsi %falseResult_17 {handshake.bb = 3 : ui32, handshake.name = "extsi15"} : <i6> to <i7>
    %95 = source {handshake.bb = 3 : ui32, handshake.name = "source5"} : <>
    %96 = constant %95 {handshake.bb = 3 : ui32, handshake.name = "constant15", value = 1 : i2} : <>, <i2>
    %97 = extsi %96 {handshake.bb = 3 : ui32, handshake.name = "extsi16"} : <i2> to <i7>
    %98 = source {handshake.bb = 3 : ui32, handshake.name = "source6"} : <>
    %99 = constant %98 {handshake.bb = 3 : ui32, handshake.name = "constant16", value = 20 : i6} : <>, <i6>
    %100 = extsi %99 {handshake.bb = 3 : ui32, handshake.name = "extsi17"} : <i6> to <i7>
    %101 = addi %94, %97 {handshake.bb = 3 : ui32, handshake.name = "addi6"} : <i7>
    %102 = buffer %101, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer24"} : <i7>
    %103:2 = fork [2] %102 {handshake.bb = 3 : ui32, handshake.name = "fork14"} : <i7>
    %104 = trunci %103#0 {handshake.bb = 3 : ui32, handshake.name = "trunci13"} : <i7> to <i6>
    %105 = cmpi ult, %103#1, %100 {handshake.bb = 3 : ui32, handshake.name = "cmpi1"} : <i7>
    %106 = buffer %105, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer25"} : <i1>
    %107:2 = fork [2] %106 {handshake.bb = 3 : ui32, handshake.name = "fork15"} : <i1>
    %trueResult_22, %falseResult_23 = cond_br %107#0, %104 {handshake.bb = 3 : ui32, handshake.name = "cond_br6"} : <i1>, <i6>
    sink %falseResult_23 {handshake.name = "sink3"} : <i6>
    %trueResult_24, %falseResult_25 = cond_br %107#1, %falseResult_21 {handshake.bb = 3 : ui32, handshake.name = "cond_br7"} : <i1>, <>
    %108:4 = fork [4] %falseResult_25 {handshake.bb = 4 : ui32, handshake.name = "fork16"} : <>
    end {handshake.bb = 4 : ui32, handshake.name = "end0"} %1#2, %memEnd_3, %memEnd_1, %memEnd, %0#1 : <>, <>, <>, <>, <>
  }
}

