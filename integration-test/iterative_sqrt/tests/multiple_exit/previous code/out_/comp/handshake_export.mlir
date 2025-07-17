module {
  handshake.func @multiple_exit(%arg0: memref<10xi32>, %arg1: !handshake.channel<i32>, %arg2: !handshake.control<>, %arg3: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>) attributes {argNames = ["arr", "size", "arr_start", "start"], resNames = ["arr_end", "end"]} {
    %0:4 = fork [4] %arg3 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork0"} : <>
    %1 = buffer %falseResult_3 {handshake.name = "buffer35", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <>
    %2 = buffer %1 {handshake.name = "buffer36", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %3:4 = lsq[%arg0 : memref<10xi32>] (%arg2, %31#1, %addressResult, %47#1, %addressResult_14, %63#1, %addressResult_26, %addressResult_28, %dataResult_29, %2)  {groupSizes = [1 : i32, 1 : i32, 2 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "5": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "6": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "7": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "8": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "9": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "lsq1"} : (!handshake.control<>, !handshake.control<>, !handshake.channel<i4>, !handshake.control<>, !handshake.channel<i4>, !handshake.control<>, !handshake.channel<i4>, !handshake.channel<i4>, !handshake.channel<i32>, !handshake.control<>) -> (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>)
    %4 = constant %0#3 {handshake.bb = 0 : ui32, handshake.name = "constant1", value = false} : <>, <i1>
    %5 = constant %0#2 {handshake.bb = 0 : ui32, handshake.name = "constant4", value = true} : <>, <i1>
    %6 = extsi %4 {handshake.bb = 0 : ui32, handshake.name = "extsi3"} : <i1> to <i32>
    %7 = mux %16#0 [%6, %falseResult_7, %falseResult_19, %74] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i2>, [<i32>, <i32>, <i32>, <i32>] to <i32>
    %8 = buffer %7 {handshake.bb = 1 : ui32, handshake.name = "buffer0", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i32>
    %9 = buffer %8 {handshake.bb = 1 : ui32, handshake.name = "buffer1", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i32>
    %10:2 = fork [2] %9 {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <i32>
    %11 = mux %16#1 [%5, %falseResult_13, %falseResult_25, %66] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i2>, [<i1>, <i1>, <i1>, <i1>] to <i1>
    %12 = mux %16#2 [%arg1, %falseResult_5, %falseResult_17, %76] {handshake.bb = 1 : ui32, handshake.name = "mux2"} : <i2>, [<i32>, <i32>, <i32>, <i32>] to <i32>
    %13 = buffer %12 {handshake.bb = 1 : ui32, handshake.name = "buffer4", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i32>
    %14 = buffer %13 {handshake.bb = 1 : ui32, handshake.name = "buffer5", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i32>
    %15:2 = fork [2] %14 {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <i32>
    %result, %index = control_merge [%0#1, %falseResult_11, %falseResult_23, %77]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>, <>, <>] to <>, <i2>
    %16:3 = fork [3] %index {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <i2>
    %17 = cmpi slt, %10#1, %15#1 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i32>
    %18 = buffer %11 {handshake.bb = 1 : ui32, handshake.name = "buffer2", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %19 = buffer %18 {handshake.bb = 1 : ui32, handshake.name = "buffer3", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i1>
    %20 = andi %17, %19 {handshake.bb = 1 : ui32, handshake.name = "andi0"} : <i1>
    %21:3 = fork [3] %20 {handshake.bb = 1 : ui32, handshake.name = "fork4"} : <i1>
    %trueResult, %falseResult = cond_br %21#2, %10#0 {handshake.bb = 1 : ui32, handshake.name = "cond_br3"} : <i1>, <i32>
    sink %falseResult {handshake.name = "sink0"} : <i32>
    %trueResult_0, %falseResult_1 = cond_br %21#1, %15#0 {handshake.bb = 1 : ui32, handshake.name = "cond_br4"} : <i1>, <i32>
    sink %falseResult_1 {handshake.name = "sink1"} : <i32>
    %22 = buffer %result {handshake.bb = 1 : ui32, handshake.name = "buffer6", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <>
    %23 = buffer %22 {handshake.bb = 1 : ui32, handshake.name = "buffer7", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %trueResult_2, %falseResult_3 = cond_br %21#0, %23 {handshake.bb = 1 : ui32, handshake.name = "cond_br5"} : <i1>, <>
    %24 = buffer %trueResult {handshake.bb = 2 : ui32, handshake.name = "buffer8", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i32>
    %25 = buffer %24 {handshake.bb = 2 : ui32, handshake.name = "buffer9", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i32>
    %26:3 = fork [3] %25 {handshake.bb = 2 : ui32, handshake.name = "fork5"} : <i32>
    %27 = trunci %26#2 {handshake.bb = 2 : ui32, handshake.name = "trunci0"} : <i32> to <i4>
    %28 = trunci %26#1 {handshake.bb = 2 : ui32, handshake.name = "trunci1"} : <i32> to <i4>
    %29 = buffer %trueResult_2 {handshake.bb = 2 : ui32, handshake.name = "buffer12", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <>
    %30 = buffer %29 {handshake.bb = 2 : ui32, handshake.name = "buffer13", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %31:3 = lazy_fork [3] %30 {handshake.bb = 2 : ui32, handshake.name = "lazy_fork0"} : <>
    %32 = buffer %31#2 {handshake.bb = 2 : ui32, handshake.name = "buffer15", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %33 = fork [1] %32 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"0": [1,inf], [0,inf], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork6"} : <>
    %34 = constant %33 {handshake.bb = 2 : ui32, handshake.name = "constant5", value = false} : <>, <i1>
    %35 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %36 = constant %35 {handshake.bb = 2 : ui32, handshake.name = "constant6", value = -1 : i32} : <>, <i32>
    %addressResult, %dataResult = load[%28] %3#0 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[<"store1" (0)>]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 0>, handshake.name = "load3"} : <i4>, <i32>, <i4>, <i32>
    %37 = cmpi ne, %dataResult, %36 {handshake.bb = 2 : ui32, handshake.name = "cmpi1"} : <i32>
    %38:5 = fork [5] %37 {handshake.bb = 2 : ui32, handshake.name = "fork7"} : <i1>
    %39 = buffer %trueResult_0 {handshake.bb = 2 : ui32, handshake.name = "buffer10", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i32>
    %40 = buffer %39 {handshake.bb = 2 : ui32, handshake.name = "buffer11", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i32>
    %trueResult_4, %falseResult_5 = cond_br %38#4, %40 {handshake.bb = 2 : ui32, handshake.name = "cond_br6"} : <i1>, <i32>
    %trueResult_6, %falseResult_7 = cond_br %38#3, %26#0 {handshake.bb = 2 : ui32, handshake.name = "cond_br7"} : <i1>, <i32>
    %trueResult_8, %falseResult_9 = cond_br %38#2, %27 {handshake.bb = 2 : ui32, handshake.name = "cond_br8"} : <i1>, <i4>
    sink %falseResult_9 {handshake.name = "sink3"} : <i4>
    %41 = buffer %31#0 {handshake.bb = 2 : ui32, handshake.name = "buffer14", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <>
    %trueResult_10, %falseResult_11 = cond_br %38#1, %41 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,inf], [1,inf], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "cond_br9"} : <i1>, <>
    %trueResult_12, %falseResult_13 = cond_br %38#0, %34 {handshake.bb = 2 : ui32, handshake.name = "cond_br10"} : <i1>, <i1>
    sink %trueResult_12 {handshake.name = "sink4"} : <i1>
    %42 = buffer %trueResult_8 {handshake.bb = 3 : ui32, handshake.name = "buffer20", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i4>
    %43 = buffer %42 {handshake.bb = 3 : ui32, handshake.name = "buffer21", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i4>
    %44:2 = fork [2] %43 {handshake.bb = 3 : ui32, handshake.name = "fork8"} : <i4>
    %45 = buffer %trueResult_10 {handshake.bb = 3 : ui32, handshake.name = "buffer22", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <>
    %46 = buffer %45 {handshake.bb = 3 : ui32, handshake.name = "buffer23", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %47:2 = lazy_fork [2] %46 {handshake.bb = 3 : ui32, handshake.name = "lazy_fork1"} : <>
    %48 = source {handshake.bb = 3 : ui32, handshake.name = "source1"} : <>
    %49 = constant %48 {handshake.bb = 3 : ui32, handshake.name = "constant2", value = false} : <>, <i1>
    %50 = extsi %49 {handshake.bb = 3 : ui32, handshake.name = "extsi1"} : <i1> to <i32>
    %addressResult_14, %dataResult_15 = load[%44#1] %3#1 {handshake.bb = 3 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[<"store1" (0)>]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 1>, handshake.name = "load4"} : <i4>, <i32>, <i4>, <i32>
    %51 = cmpi ne, %dataResult_15, %50 {handshake.bb = 3 : ui32, handshake.name = "cmpi2"} : <i32>
    %52:6 = fork [6] %51 {handshake.bb = 3 : ui32, handshake.name = "fork9"} : <i1>
    %53 = buffer %trueResult_4 {handshake.bb = 3 : ui32, handshake.name = "buffer16", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i32>
    %54 = buffer %53 {handshake.bb = 3 : ui32, handshake.name = "buffer17", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i32>
    %trueResult_16, %falseResult_17 = cond_br %52#5, %54 {handshake.bb = 3 : ui32, handshake.name = "cond_br11"} : <i1>, <i32>
    %55 = buffer %trueResult_6 {handshake.bb = 3 : ui32, handshake.name = "buffer18", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i32>
    %56 = buffer %55 {handshake.bb = 3 : ui32, handshake.name = "buffer19", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i32>
    %trueResult_18, %falseResult_19 = cond_br %52#4, %56 {handshake.bb = 3 : ui32, handshake.name = "cond_br12"} : <i1>, <i32>
    %trueResult_20, %falseResult_21 = cond_br %52#3, %44#0 {handshake.bb = 3 : ui32, handshake.name = "cond_br13"} : <i1>, <i4>
    sink %falseResult_21 {handshake.name = "sink6"} : <i4>
    %57 = buffer %47#0 {handshake.bb = 3 : ui32, handshake.name = "buffer24", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <>
    %trueResult_22, %falseResult_23 = cond_br %52#2, %57 {handshake.bb = 3 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,inf], [1,inf], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "cond_br14"} : <i1>, <>
    %trueResult_24, %falseResult_25 = cond_br %52#0, %52#1 {handshake.bb = 3 : ui32, handshake.name = "cond_br15"} : <i1>, <i1>
    sink %trueResult_24 {handshake.name = "sink7"} : <i1>
    %58 = buffer %trueResult_20 {handshake.bb = 4 : ui32, handshake.name = "buffer29", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i4>
    %59 = buffer %58 {handshake.bb = 4 : ui32, handshake.name = "buffer30", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i4>
    %60:2 = fork [2] %59 {handshake.bb = 4 : ui32, handshake.name = "fork10"} : <i4>
    %61 = buffer %trueResult_22 {handshake.bb = 4 : ui32, handshake.name = "buffer31", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <>
    %62 = buffer %61 {handshake.bb = 4 : ui32, handshake.name = "buffer32", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %63:3 = lazy_fork [3] %62 {handshake.bb = 4 : ui32, handshake.name = "lazy_fork2"} : <>
    %64 = buffer %63#2 {handshake.bb = 4 : ui32, handshake.name = "buffer34", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %65 = fork [1] %64 {handshake.bb = 4 : ui32, handshake.bufProps = #handshake<bufProps{"0": [1,inf], [0,inf], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork11"} : <>
    %66 = constant %65 {handshake.bb = 4 : ui32, handshake.name = "constant8", value = true} : <>, <i1>
    %67 = source {handshake.bb = 4 : ui32, handshake.name = "source2"} : <>
    %68 = constant %67 {handshake.bb = 4 : ui32, handshake.name = "constant3", value = 1 : i2} : <>, <i2>
    %69 = extsi %68 {handshake.bb = 4 : ui32, handshake.name = "extsi2"} : <i2> to <i32>
    %70:2 = fork [2] %69 {handshake.bb = 4 : ui32, handshake.name = "fork12"} : <i32>
    %addressResult_26, %dataResult_27 = load[%60#1] %3#2 {handshake.bb = 4 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[<"store1" (0)>]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 2>, handshake.name = "load5"} : <i4>, <i32>, <i4>, <i32>
    %71 = addi %dataResult_27, %70#1 {handshake.bb = 4 : ui32, handshake.name = "addi0"} : <i32>
    %addressResult_28, %dataResult_29 = store[%60#0] %71 {handshake.bb = 4 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[<"load3" (0)>, <"load4" (0)>, <"load5" (0)>, <"store1" (0)>]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 2>, handshake.name = "store1"} : <i4>, <i32>, <i4>, <i32>
    %72 = buffer %trueResult_18 {handshake.bb = 4 : ui32, handshake.name = "buffer27", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i32>
    %73 = buffer %72 {handshake.bb = 4 : ui32, handshake.name = "buffer28", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i32>
    %74 = addi %73, %70#0 {handshake.bb = 4 : ui32, handshake.name = "addi1"} : <i32>
    %75 = buffer %trueResult_16 {handshake.bb = 4 : ui32, handshake.name = "buffer25", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i32>
    %76 = buffer %75 {handshake.bb = 4 : ui32, handshake.name = "buffer26", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i32>
    %77 = buffer %63#0 {handshake.bb = 4 : ui32, handshake.name = "buffer33", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <>
    end {handshake.bb = 5 : ui32, handshake.name = "end0"} %3#3, %0#0 : <>, <>
  }
}

