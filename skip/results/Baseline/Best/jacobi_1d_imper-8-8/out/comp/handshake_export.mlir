module {
  handshake.func @jacobi_1d_imper(%arg0: memref<100xi32>, %arg1: memref<100xi32>, %arg2: !handshake.control<>, %arg3: !handshake.control<>, %arg4: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["A", "B", "A_start", "B_start", "start"], resNames = ["A_end", "B_end", "end"]} {
    %0:3 = fork [3] %arg4 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork0"} : <>
    %1:2 = lsq[%arg1 : memref<100xi32>] (%arg3, %20#0, %addressResult_6, %dataResult_7, %69#0, %addressResult_16, %103#1)  {groupSizes = [1 : i32, 1 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "5": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "6": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "lsq2"} : (!handshake.control<>, !handshake.control<>, !handshake.channel<i7>, !handshake.channel<i32>, !handshake.control<>, !handshake.channel<i7>, !handshake.control<>) -> (!handshake.channel<i32>, !handshake.control<>)
    %2:4 = lsq[%arg0 : memref<100xi32>] (%arg2, %20#2, %addressResult, %addressResult_2, %addressResult_4, %69#2, %addressResult_18, %dataResult_19, %103#0)  {groupSizes = [3 : i32, 1 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "5": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "6": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "7": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "8": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "lsq3"} : (!handshake.control<>, !handshake.control<>, !handshake.channel<i7>, !handshake.channel<i7>, !handshake.channel<i7>, !handshake.control<>, !handshake.channel<i7>, !handshake.channel<i32>, !handshake.control<>) -> (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>)
    %3 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant0", value = false} : <>, <i1>
    %4 = extsi %3 {handshake.bb = 0 : ui32, handshake.name = "extsi11"} : <i1> to <i3>
    %5 = mux %index [%4, %trueResult_26] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i3>, <i3>] to <i3>
    %result, %index = control_merge [%0#2, %trueResult_28]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %6:2 = fork [2] %result {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <>
    %7 = constant %6#0 {handshake.bb = 1 : ui32, handshake.name = "constant1", value = 1 : i2} : <>, <i2>
    %8 = extsi %7 {handshake.bb = 1 : ui32, handshake.name = "extsi10"} : <i2> to <i8>
    %9 = buffer %5, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer0"} : <i3>
    %10 = mux %19#1 [%8, %trueResult] {handshake.bb = 2 : ui32, handshake.name = "mux1"} : <i1>, [<i8>, <i8>] to <i8>
    %11 = buffer %10, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer1"} : <i8>
    %12:5 = fork [5] %11 {handshake.bb = 2 : ui32, handshake.name = "fork2"} : <i8>
    %13 = trunci %12#0 {handshake.bb = 2 : ui32, handshake.name = "trunci0"} : <i8> to <i7>
    %14 = trunci %12#1 {handshake.bb = 2 : ui32, handshake.name = "trunci1"} : <i8> to <i7>
    %15 = extsi %12#4 {handshake.bb = 2 : ui32, handshake.name = "extsi12"} : <i8> to <i9>
    %16 = trunci %12#2 {handshake.bb = 2 : ui32, handshake.name = "trunci2"} : <i8> to <i7>
    %17 = trunci %12#3 {handshake.bb = 2 : ui32, handshake.name = "trunci3"} : <i8> to <i7>
    %18 = mux %19#0 [%9, %trueResult_8] {handshake.bb = 2 : ui32, handshake.name = "mux2"} : <i1>, [<i3>, <i3>] to <i3>
    %result_0, %index_1 = control_merge [%6#1, %trueResult_10]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>, <>] to <>, <i1>
    %19:2 = fork [2] %index_1 {handshake.bb = 2 : ui32, handshake.name = "fork3"} : <i1>
    %20:4 = lazy_fork [4] %result_0 {handshake.bb = 2 : ui32, handshake.name = "lazy_fork0"} : <>
    %21 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %22 = constant %21 {handshake.bb = 2 : ui32, handshake.name = "constant7", value = -1 : i32} : <>, <i32>
    %23 = trunci %22 {handshake.bb = 2 : ui32, handshake.name = "trunci4"} : <i32> to <i7>
    %24 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %25 = constant %24 {handshake.bb = 2 : ui32, handshake.name = "constant2", value = 99 : i8} : <>, <i8>
    %26 = extsi %25 {handshake.bb = 2 : ui32, handshake.name = "extsi13"} : <i8> to <i9>
    %27 = buffer %20#3, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer7"} : <>
    %28 = constant %27 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"0": [1,inf], [0,inf], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "constant4", value = 1 : i2} : <>, <i2>
    %29:3 = fork [3] %28 {handshake.bb = 2 : ui32, handshake.name = "fork4"} : <i2>
    %30 = extsi %29#0 {handshake.bb = 2 : ui32, handshake.name = "extsi14"} : <i2> to <i7>
    %31 = extsi %29#1 {handshake.bb = 2 : ui32, handshake.name = "extsi15"} : <i2> to <i9>
    %32 = source {handshake.bb = 2 : ui32, handshake.name = "source2"} : <>
    %33 = constant %32 {handshake.bb = 2 : ui32, handshake.name = "constant5", value = 1 : i2} : <>, <i2>
    %34 = extsi %33 {handshake.bb = 2 : ui32, handshake.name = "extsi4"} : <i2> to <i32>
    %35 = addi %13, %23 {handshake.bb = 2 : ui32, handshake.name = "addi2"} : <i7>
    %36 = buffer %35, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer8"} : <i7>
    %addressResult, %dataResult = load[%36] %2#0 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["store1", 1], ["store1", 2]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 0>, handshake.name = "load0"} : <i7>, <i32>, <i7>, <i32>
    %37 = buffer %17, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer3"} : <i7>
    %addressResult_2, %dataResult_3 = load[%37] %2#1 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["store1", 1], ["store1", 2]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 0>, handshake.name = "load1"} : <i7>, <i32>, <i7>, <i32>
    %38 = addi %dataResult, %dataResult_3 {handshake.bb = 2 : ui32, handshake.name = "addi0"} : <i32>
    %39 = addi %14, %30 {handshake.bb = 2 : ui32, handshake.name = "addi3"} : <i7>
    %40 = buffer %39, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer10"} : <i7>
    %addressResult_4, %dataResult_5 = load[%40] %2#2 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["store1", 1], ["store1", 2]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 0>, handshake.name = "load2"} : <i7>, <i32>, <i7>, <i32>
    %41 = buffer %38, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer9"} : <i32>
    %42 = addi %41, %dataResult_5 {handshake.bb = 2 : ui32, handshake.name = "addi1"} : <i32>
    %43 = buffer %42, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer11"} : <i32>
    %44:2 = fork [2] %43 {handshake.bb = 2 : ui32, handshake.name = "fork5"} : <i32>
    %45 = shli %44#1, %34 {handshake.bb = 2 : ui32, handshake.name = "shli0"} : <i32>
    %46 = buffer %45, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer12"} : <i32>
    %47 = addi %44#0, %46 {handshake.bb = 2 : ui32, handshake.name = "addi7"} : <i32>
    %48 = buffer %16, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer2"} : <i7>
    %49 = buffer %47, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer13"} : <i32>
    %addressResult_6, %dataResult_7 = store[%48] %49 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["store0", 1], ["load3", 1], ["load3", 2]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 0>, handshake.name = "store0"} : <i7>, <i32>, <i7>, <i32>
    %50 = addi %15, %31 {handshake.bb = 2 : ui32, handshake.name = "addi4"} : <i9>
    %51 = buffer %50, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer14"} : <i9>
    %52:2 = fork [2] %51 {handshake.bb = 2 : ui32, handshake.name = "fork6"} : <i9>
    %53 = trunci %52#0 {handshake.bb = 2 : ui32, handshake.name = "trunci5"} : <i9> to <i8>
    %54 = cmpi ult, %52#1, %26 {handshake.bb = 2 : ui32, handshake.name = "cmpi0"} : <i9>
    %55 = buffer %54, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer15"} : <i1>
    %56:4 = fork [4] %55 {handshake.bb = 2 : ui32, handshake.name = "fork7"} : <i1>
    %trueResult, %falseResult = cond_br %56#0, %53 {handshake.bb = 2 : ui32, handshake.name = "cond_br3"} : <i1>, <i8>
    sink %falseResult {handshake.name = "sink0"} : <i8>
    %57 = buffer %18, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer4"} : <i3>
    %58 = buffer %57, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer5"} : <i3>
    %trueResult_8, %falseResult_9 = cond_br %56#1, %58 {handshake.bb = 2 : ui32, handshake.name = "cond_br4"} : <i1>, <i3>
    %59 = buffer %20#1, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer6"} : <>
    %trueResult_10, %falseResult_11 = cond_br %56#2, %59 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,inf], [1,inf], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "cond_br5"} : <i1>, <>
    %trueResult_12, %falseResult_13 = cond_br %56#3, %29#2 {handshake.bb = 2 : ui32, handshake.name = "cond_br6"} : <i1>, <i2>
    sink %trueResult_12 {handshake.name = "sink1"} : <i2>
    %60 = extsi %falseResult_13 {handshake.bb = 2 : ui32, handshake.name = "extsi9"} : <i2> to <i8>
    %61 = mux %68#1 [%60, %trueResult_20] {handshake.bb = 3 : ui32, handshake.name = "mux3"} : <i1>, [<i8>, <i8>] to <i8>
    %62 = buffer %61, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer16"} : <i8>
    %63:3 = fork [3] %62 {handshake.bb = 3 : ui32, handshake.name = "fork8"} : <i8>
    %64 = extsi %63#2 {handshake.bb = 3 : ui32, handshake.name = "extsi16"} : <i8> to <i9>
    %65 = trunci %63#0 {handshake.bb = 3 : ui32, handshake.name = "trunci6"} : <i8> to <i7>
    %66 = trunci %63#1 {handshake.bb = 3 : ui32, handshake.name = "trunci7"} : <i8> to <i7>
    %67 = mux %68#0 [%falseResult_9, %trueResult_22] {handshake.bb = 3 : ui32, handshake.name = "mux4"} : <i1>, [<i3>, <i3>] to <i3>
    %result_14, %index_15 = control_merge [%falseResult_11, %trueResult_24]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>, <>] to <>, <i1>
    %68:2 = fork [2] %index_15 {handshake.bb = 3 : ui32, handshake.name = "fork9"} : <i1>
    %69:3 = lazy_fork [3] %result_14 {handshake.bb = 3 : ui32, handshake.name = "lazy_fork1"} : <>
    %70 = source {handshake.bb = 3 : ui32, handshake.name = "source3"} : <>
    %71 = constant %70 {handshake.bb = 3 : ui32, handshake.name = "constant15", value = 99 : i8} : <>, <i8>
    %72 = extsi %71 {handshake.bb = 3 : ui32, handshake.name = "extsi17"} : <i8> to <i9>
    %73 = source {handshake.bb = 3 : ui32, handshake.name = "source4"} : <>
    %74 = constant %73 {handshake.bb = 3 : ui32, handshake.name = "constant16", value = 1 : i2} : <>, <i2>
    %75 = extsi %74 {handshake.bb = 3 : ui32, handshake.name = "extsi18"} : <i2> to <i9>
    %76 = buffer %66, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer18"} : <i7>
    %addressResult_16, %dataResult_17 = load[%76] %1#0 {handshake.bb = 3 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["store0", 1]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 1>, handshake.name = "load3"} : <i7>, <i32>, <i7>, <i32>
    %77 = buffer %65, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer17"} : <i7>
    %78 = buffer %dataResult_17, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer22"} : <i32>
    %addressResult_18, %dataResult_19 = store[%77] %78 {handshake.bb = 3 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["load0", 1], ["load1", 1], ["load2", 1], ["store1", 1]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 1>, handshake.name = "store1"} : <i7>, <i32>, <i7>, <i32>
    %79 = addi %64, %75 {handshake.bb = 3 : ui32, handshake.name = "addi5"} : <i9>
    %80 = buffer %79, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer23"} : <i9>
    %81:2 = fork [2] %80 {handshake.bb = 3 : ui32, handshake.name = "fork10"} : <i9>
    %82 = trunci %81#0 {handshake.bb = 3 : ui32, handshake.name = "trunci8"} : <i9> to <i8>
    %83 = cmpi ult, %81#1, %72 {handshake.bb = 3 : ui32, handshake.name = "cmpi1"} : <i9>
    %84 = buffer %83, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer24"} : <i1>
    %85:3 = fork [3] %84 {handshake.bb = 3 : ui32, handshake.name = "fork11"} : <i1>
    %trueResult_20, %falseResult_21 = cond_br %85#0, %82 {handshake.bb = 3 : ui32, handshake.name = "cond_br7"} : <i1>, <i8>
    sink %falseResult_21 {handshake.name = "sink2"} : <i8>
    %86 = buffer %67, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer19"} : <i3>
    %87 = buffer %86, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer20"} : <i3>
    %trueResult_22, %falseResult_23 = cond_br %85#1, %87 {handshake.bb = 3 : ui32, handshake.name = "cond_br8"} : <i1>, <i3>
    %88 = buffer %69#1, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer21"} : <>
    %trueResult_24, %falseResult_25 = cond_br %85#2, %88 {handshake.bb = 3 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,inf], [1,inf], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "cond_br9"} : <i1>, <>
    %89 = extsi %falseResult_23 {handshake.bb = 4 : ui32, handshake.name = "extsi19"} : <i3> to <i4>
    %90 = source {handshake.bb = 4 : ui32, handshake.name = "source5"} : <>
    %91 = constant %90 {handshake.bb = 4 : ui32, handshake.name = "constant17", value = 3 : i3} : <>, <i3>
    %92 = extsi %91 {handshake.bb = 4 : ui32, handshake.name = "extsi20"} : <i3> to <i4>
    %93 = source {handshake.bb = 4 : ui32, handshake.name = "source6"} : <>
    %94 = constant %93 {handshake.bb = 4 : ui32, handshake.name = "constant18", value = 1 : i2} : <>, <i2>
    %95 = extsi %94 {handshake.bb = 4 : ui32, handshake.name = "extsi21"} : <i2> to <i4>
    %96 = addi %89, %95 {handshake.bb = 4 : ui32, handshake.name = "addi6"} : <i4>
    %97 = buffer %96, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer25"} : <i4>
    %98:2 = fork [2] %97 {handshake.bb = 4 : ui32, handshake.name = "fork12"} : <i4>
    %99 = trunci %98#0 {handshake.bb = 4 : ui32, handshake.name = "trunci9"} : <i4> to <i3>
    %100 = cmpi ult, %98#1, %92 {handshake.bb = 4 : ui32, handshake.name = "cmpi2"} : <i4>
    %101 = buffer %100, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer26"} : <i1>
    %102:2 = fork [2] %101 {handshake.bb = 4 : ui32, handshake.name = "fork13"} : <i1>
    %trueResult_26, %falseResult_27 = cond_br %102#0, %99 {handshake.bb = 4 : ui32, handshake.name = "cond_br10"} : <i1>, <i3>
    sink %falseResult_27 {handshake.name = "sink4"} : <i3>
    %trueResult_28, %falseResult_29 = cond_br %102#1, %falseResult_25 {handshake.bb = 4 : ui32, handshake.name = "cond_br11"} : <i1>, <>
    %103:2 = fork [2] %falseResult_29 {handshake.bb = 5 : ui32, handshake.name = "fork14"} : <>
    end {handshake.bb = 5 : ui32, handshake.name = "end0"} %2#3, %1#1, %0#1 : <>, <>, <>
  }
}

