module {
  handshake.func @jacobi_1d_imper(%arg0: memref<100xi32>, %arg1: memref<100xi32>, %arg2: !handshake.control<>, %arg3: !handshake.control<>, %arg4: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["A", "B", "A_start", "B_start", "start"], resNames = ["A_end", "B_end", "end"]} {
    %0:3 = fork [3] %arg4 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork0"} : <>
    %1:2 = lsq[%arg1 : memref<100xi32>] (%arg3, %25#0, %addressResult_6, %dataResult_7, %74#0, %addressResult_16, %109#1)  {groupSizes = [1 : i32, 1 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "5": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "6": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "lsq2"} : (!handshake.control<>, !handshake.control<>, !handshake.channel<i7>, !handshake.channel<i32>, !handshake.control<>, !handshake.channel<i7>, !handshake.control<>) -> (!handshake.channel<i32>, !handshake.control<>)
    %2:4 = lsq[%arg0 : memref<100xi32>] (%arg2, %25#2, %addressResult, %addressResult_2, %addressResult_4, %74#2, %addressResult_18, %dataResult_19, %109#0)  {groupSizes = [3 : i32, 1 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "5": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "6": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "7": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "8": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "lsq3"} : (!handshake.control<>, !handshake.control<>, !handshake.channel<i7>, !handshake.channel<i7>, !handshake.channel<i7>, !handshake.control<>, !handshake.channel<i7>, !handshake.channel<i32>, !handshake.control<>) -> (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>)
    %3 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant0", value = false} : <>, <i1>
    %4 = br %3 {handshake.bb = 0 : ui32, handshake.name = "br2"} : <i1>
    %5 = extsi %4 {handshake.bb = 0 : ui32, handshake.name = "extsi11"} : <i1> to <i3>
    %6 = br %0#2 {handshake.bb = 0 : ui32, handshake.name = "br3"} : <>
    %7 = mux %index [%5, %trueResult_28] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i3>, <i3>] to <i3>
    %result, %index = control_merge [%6, %trueResult_30]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %8:2 = fork [2] %result {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <>
    %9 = constant %8#0 {handshake.bb = 1 : ui32, handshake.name = "constant1", value = 1 : i2} : <>, <i2>
    %10 = br %9 {handshake.bb = 1 : ui32, handshake.name = "br4"} : <i2>
    %11 = extsi %10 {handshake.bb = 1 : ui32, handshake.name = "extsi10"} : <i2> to <i8>
    %12 = buffer %7, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer0"} : <i3>
    %13 = br %12 {handshake.bb = 1 : ui32, handshake.name = "br5"} : <i3>
    %14 = br %8#1 {handshake.bb = 1 : ui32, handshake.name = "br6"} : <>
    %15 = mux %24#1 [%11, %trueResult] {handshake.bb = 2 : ui32, handshake.name = "mux1"} : <i1>, [<i8>, <i8>] to <i8>
    %16 = buffer %15, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer1"} : <i8>
    %17:5 = fork [5] %16 {handshake.bb = 2 : ui32, handshake.name = "fork2"} : <i8>
    %18 = trunci %17#0 {handshake.bb = 2 : ui32, handshake.name = "trunci0"} : <i8> to <i7>
    %19 = trunci %17#1 {handshake.bb = 2 : ui32, handshake.name = "trunci1"} : <i8> to <i7>
    %20 = extsi %17#4 {handshake.bb = 2 : ui32, handshake.name = "extsi12"} : <i8> to <i9>
    %21 = trunci %17#2 {handshake.bb = 2 : ui32, handshake.name = "trunci2"} : <i8> to <i7>
    %22 = trunci %17#3 {handshake.bb = 2 : ui32, handshake.name = "trunci3"} : <i8> to <i7>
    %23 = mux %24#0 [%13, %trueResult_8] {handshake.bb = 2 : ui32, handshake.name = "mux2"} : <i1>, [<i3>, <i3>] to <i3>
    %result_0, %index_1 = control_merge [%14, %trueResult_10]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>, <>] to <>, <i1>
    %24:2 = fork [2] %index_1 {handshake.bb = 2 : ui32, handshake.name = "fork3"} : <i1>
    %25:4 = lazy_fork [4] %result_0 {handshake.bb = 2 : ui32, handshake.name = "lazy_fork0"} : <>
    %26 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %27 = constant %26 {handshake.bb = 2 : ui32, handshake.name = "constant7", value = -1 : i32} : <>, <i32>
    %28 = trunci %27 {handshake.bb = 2 : ui32, handshake.name = "trunci4"} : <i32> to <i7>
    %29 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %30 = constant %29 {handshake.bb = 2 : ui32, handshake.name = "constant2", value = 99 : i8} : <>, <i8>
    %31 = extsi %30 {handshake.bb = 2 : ui32, handshake.name = "extsi13"} : <i8> to <i9>
    %32 = buffer %25#3, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer7"} : <>
    %33 = constant %32 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"0": [1,inf], [0,inf], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "constant4", value = 1 : i2} : <>, <i2>
    %34:3 = fork [3] %33 {handshake.bb = 2 : ui32, handshake.name = "fork4"} : <i2>
    %35 = extsi %34#0 {handshake.bb = 2 : ui32, handshake.name = "extsi14"} : <i2> to <i7>
    %36 = extsi %34#1 {handshake.bb = 2 : ui32, handshake.name = "extsi15"} : <i2> to <i9>
    %37 = source {handshake.bb = 2 : ui32, handshake.name = "source2"} : <>
    %38 = constant %37 {handshake.bb = 2 : ui32, handshake.name = "constant5", value = 1 : i2} : <>, <i2>
    %39 = extsi %38 {handshake.bb = 2 : ui32, handshake.name = "extsi4"} : <i2> to <i32>
    %40 = addi %18, %28 {handshake.bb = 2 : ui32, handshake.name = "addi2"} : <i7>
    %41 = buffer %40, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer8"} : <i7>
    %addressResult, %dataResult = load[%41] %2#0 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["store1", 1], ["store1", 2]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 0>, handshake.name = "load0"} : <i7>, <i32>, <i7>, <i32>
    %42 = buffer %22, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer3"} : <i7>
    %addressResult_2, %dataResult_3 = load[%42] %2#1 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["store1", 1], ["store1", 2]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 0>, handshake.name = "load1"} : <i7>, <i32>, <i7>, <i32>
    %43 = addi %dataResult, %dataResult_3 {handshake.bb = 2 : ui32, handshake.name = "addi0"} : <i32>
    %44 = addi %19, %35 {handshake.bb = 2 : ui32, handshake.name = "addi3"} : <i7>
    %45 = buffer %44, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer10"} : <i7>
    %addressResult_4, %dataResult_5 = load[%45] %2#2 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["store1", 1], ["store1", 2]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 0>, handshake.name = "load2"} : <i7>, <i32>, <i7>, <i32>
    %46 = buffer %43, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer9"} : <i32>
    %47 = addi %46, %dataResult_5 {handshake.bb = 2 : ui32, handshake.name = "addi1"} : <i32>
    %48 = buffer %47, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer11"} : <i32>
    %49:2 = fork [2] %48 {handshake.bb = 2 : ui32, handshake.name = "fork5"} : <i32>
    %50 = shli %49#1, %39 {handshake.bb = 2 : ui32, handshake.name = "shli0"} : <i32>
    %51 = buffer %50, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer12"} : <i32>
    %52 = addi %49#0, %51 {handshake.bb = 2 : ui32, handshake.name = "addi7"} : <i32>
    %53 = buffer %21, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer2"} : <i7>
    %54 = buffer %52, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer13"} : <i32>
    %addressResult_6, %dataResult_7 = store[%53] %54 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["store0", 1], ["load3", 1], ["load3", 2]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 0>, handshake.name = "store0"} : <i7>, <i32>, <i7>, <i32>
    %55 = addi %20, %36 {handshake.bb = 2 : ui32, handshake.name = "addi4"} : <i9>
    %56 = buffer %55, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer14"} : <i9>
    %57:2 = fork [2] %56 {handshake.bb = 2 : ui32, handshake.name = "fork6"} : <i9>
    %58 = trunci %57#0 {handshake.bb = 2 : ui32, handshake.name = "trunci5"} : <i9> to <i8>
    %59 = cmpi ult, %57#1, %31 {handshake.bb = 2 : ui32, handshake.name = "cmpi0"} : <i9>
    %60 = buffer %59, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer15"} : <i1>
    %61:4 = fork [4] %60 {handshake.bb = 2 : ui32, handshake.name = "fork7"} : <i1>
    %trueResult, %falseResult = cond_br %61#0, %58 {handshake.bb = 2 : ui32, handshake.name = "cond_br3"} : <i1>, <i8>
    sink %falseResult {handshake.name = "sink0"} : <i8>
    %62 = buffer %23, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer4"} : <i3>
    %63 = buffer %62, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer5"} : <i3>
    %trueResult_8, %falseResult_9 = cond_br %61#1, %63 {handshake.bb = 2 : ui32, handshake.name = "cond_br4"} : <i1>, <i3>
    %64 = buffer %25#1, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer6"} : <>
    %trueResult_10, %falseResult_11 = cond_br %61#2, %64 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,inf], [1,inf], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "cond_br5"} : <i1>, <>
    %trueResult_12, %falseResult_13 = cond_br %61#3, %34#2 {handshake.bb = 2 : ui32, handshake.name = "cond_br6"} : <i1>, <i2>
    sink %trueResult_12 {handshake.name = "sink1"} : <i2>
    %65 = extsi %falseResult_13 {handshake.bb = 2 : ui32, handshake.name = "extsi9"} : <i2> to <i8>
    %66 = mux %73#1 [%65, %trueResult_20] {handshake.bb = 3 : ui32, handshake.name = "mux3"} : <i1>, [<i8>, <i8>] to <i8>
    %67 = buffer %66, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer16"} : <i8>
    %68:3 = fork [3] %67 {handshake.bb = 3 : ui32, handshake.name = "fork8"} : <i8>
    %69 = extsi %68#2 {handshake.bb = 3 : ui32, handshake.name = "extsi16"} : <i8> to <i9>
    %70 = trunci %68#0 {handshake.bb = 3 : ui32, handshake.name = "trunci6"} : <i8> to <i7>
    %71 = trunci %68#1 {handshake.bb = 3 : ui32, handshake.name = "trunci7"} : <i8> to <i7>
    %72 = mux %73#0 [%falseResult_9, %trueResult_22] {handshake.bb = 3 : ui32, handshake.name = "mux4"} : <i1>, [<i3>, <i3>] to <i3>
    %result_14, %index_15 = control_merge [%falseResult_11, %trueResult_24]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>, <>] to <>, <i1>
    %73:2 = fork [2] %index_15 {handshake.bb = 3 : ui32, handshake.name = "fork9"} : <i1>
    %74:3 = lazy_fork [3] %result_14 {handshake.bb = 3 : ui32, handshake.name = "lazy_fork1"} : <>
    %75 = source {handshake.bb = 3 : ui32, handshake.name = "source3"} : <>
    %76 = constant %75 {handshake.bb = 3 : ui32, handshake.name = "constant15", value = 99 : i8} : <>, <i8>
    %77 = extsi %76 {handshake.bb = 3 : ui32, handshake.name = "extsi17"} : <i8> to <i9>
    %78 = source {handshake.bb = 3 : ui32, handshake.name = "source4"} : <>
    %79 = constant %78 {handshake.bb = 3 : ui32, handshake.name = "constant16", value = 1 : i2} : <>, <i2>
    %80 = extsi %79 {handshake.bb = 3 : ui32, handshake.name = "extsi18"} : <i2> to <i9>
    %81 = buffer %71, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer18"} : <i7>
    %addressResult_16, %dataResult_17 = load[%81] %1#0 {handshake.bb = 3 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["store0", 1]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 1>, handshake.name = "load3"} : <i7>, <i32>, <i7>, <i32>
    %82 = buffer %70, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer17"} : <i7>
    %83 = buffer %dataResult_17, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer22"} : <i32>
    %addressResult_18, %dataResult_19 = store[%82] %83 {handshake.bb = 3 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["load0", 1], ["load1", 1], ["load2", 1], ["store1", 1]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 1>, handshake.name = "store1"} : <i7>, <i32>, <i7>, <i32>
    %84 = addi %69, %80 {handshake.bb = 3 : ui32, handshake.name = "addi5"} : <i9>
    %85 = buffer %84, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer23"} : <i9>
    %86:2 = fork [2] %85 {handshake.bb = 3 : ui32, handshake.name = "fork10"} : <i9>
    %87 = trunci %86#0 {handshake.bb = 3 : ui32, handshake.name = "trunci8"} : <i9> to <i8>
    %88 = cmpi ult, %86#1, %77 {handshake.bb = 3 : ui32, handshake.name = "cmpi1"} : <i9>
    %89 = buffer %88, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer24"} : <i1>
    %90:3 = fork [3] %89 {handshake.bb = 3 : ui32, handshake.name = "fork11"} : <i1>
    %trueResult_20, %falseResult_21 = cond_br %90#0, %87 {handshake.bb = 3 : ui32, handshake.name = "cond_br7"} : <i1>, <i8>
    sink %falseResult_21 {handshake.name = "sink2"} : <i8>
    %91 = buffer %72, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer19"} : <i3>
    %92 = buffer %91, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer20"} : <i3>
    %trueResult_22, %falseResult_23 = cond_br %90#1, %92 {handshake.bb = 3 : ui32, handshake.name = "cond_br8"} : <i1>, <i3>
    %93 = buffer %74#1, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer21"} : <>
    %trueResult_24, %falseResult_25 = cond_br %90#2, %93 {handshake.bb = 3 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,inf], [1,inf], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "cond_br9"} : <i1>, <>
    %94 = merge %falseResult_23 {handshake.bb = 4 : ui32, handshake.name = "merge0"} : <i3>
    %95 = extsi %94 {handshake.bb = 4 : ui32, handshake.name = "extsi19"} : <i3> to <i4>
    %result_26, %index_27 = control_merge [%falseResult_25]  {handshake.bb = 4 : ui32, handshake.name = "control_merge3"} : [<>] to <>, <i1>
    sink %index_27 {handshake.name = "sink3"} : <i1>
    %96 = source {handshake.bb = 4 : ui32, handshake.name = "source5"} : <>
    %97 = constant %96 {handshake.bb = 4 : ui32, handshake.name = "constant17", value = 3 : i3} : <>, <i3>
    %98 = extsi %97 {handshake.bb = 4 : ui32, handshake.name = "extsi20"} : <i3> to <i4>
    %99 = source {handshake.bb = 4 : ui32, handshake.name = "source6"} : <>
    %100 = constant %99 {handshake.bb = 4 : ui32, handshake.name = "constant18", value = 1 : i2} : <>, <i2>
    %101 = extsi %100 {handshake.bb = 4 : ui32, handshake.name = "extsi21"} : <i2> to <i4>
    %102 = addi %95, %101 {handshake.bb = 4 : ui32, handshake.name = "addi6"} : <i4>
    %103 = buffer %102, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer25"} : <i4>
    %104:2 = fork [2] %103 {handshake.bb = 4 : ui32, handshake.name = "fork12"} : <i4>
    %105 = trunci %104#0 {handshake.bb = 4 : ui32, handshake.name = "trunci9"} : <i4> to <i3>
    %106 = cmpi ult, %104#1, %98 {handshake.bb = 4 : ui32, handshake.name = "cmpi2"} : <i4>
    %107 = buffer %106, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer26"} : <i1>
    %108:2 = fork [2] %107 {handshake.bb = 4 : ui32, handshake.name = "fork13"} : <i1>
    %trueResult_28, %falseResult_29 = cond_br %108#0, %105 {handshake.bb = 4 : ui32, handshake.name = "cond_br10"} : <i1>, <i3>
    sink %falseResult_29 {handshake.name = "sink4"} : <i3>
    %trueResult_30, %falseResult_31 = cond_br %108#1, %result_26 {handshake.bb = 4 : ui32, handshake.name = "cond_br11"} : <i1>, <>
    %result_32, %index_33 = control_merge [%falseResult_31]  {handshake.bb = 5 : ui32, handshake.name = "control_merge4"} : [<>] to <>, <i1>
    sink %index_33 {handshake.name = "sink5"} : <i1>
    %109:2 = fork [2] %result_32 {handshake.bb = 5 : ui32, handshake.name = "fork14"} : <>
    end {handshake.bb = 5 : ui32, handshake.name = "end0"} %2#3, %1#1, %0#1 : <>, <>, <>
  }
}

