module {
  handshake.func @multiple_exit(%arg0: memref<10xi32>, %arg1: !handshake.control<>, %arg2: !handshake.control<>, ...) -> (!handshake.channel<i32>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["arr", "arr_start", "start"], resNames = ["out0", "arr_end", "end"]} {
    %0:4 = fork [4] %arg2 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork0"} : <>
    %1 = buffer %falseResult_5 {handshake.name = "buffer47", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <>
    %2 = buffer %1 {handshake.name = "buffer48", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %3:4 = lsq[%arg0 : memref<10xi32>] (%arg1, %39#1, %addressResult, %57#1, %addressResult_18, %82#1, %addressResult_32, %addressResult_34, %dataResult_35, %2)  {groupSizes = [1 : i32, 1 : i32, 2 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "5": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "6": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "7": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "8": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "9": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "lsq1"} : (!handshake.control<>, !handshake.control<>, !handshake.channel<i4>, !handshake.control<>, !handshake.channel<i4>, !handshake.control<>, !handshake.channel<i4>, !handshake.channel<i4>, !handshake.channel<i32>, !handshake.control<>) -> (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>)
    %4 = constant %0#3 {handshake.bb = 0 : ui32, handshake.name = "constant1", value = false} : <>, <i1>
    %5:2 = fork [2] %4 {handshake.bb = 0 : ui32, handshake.name = "fork1"} : <i1>
    %6 = constant %0#2 {handshake.bb = 0 : ui32, handshake.name = "constant6", value = true} : <>, <i1>
    %7 = extsi %5#1 {handshake.bb = 0 : ui32, handshake.name = "extsi8"} : <i1> to <i32>
    %8:2 = fork [2] %6 {handshake.bb = 0 : ui32, handshake.name = "fork2"} : <i1>
    %9 = extsi %5#0 {handshake.bb = 0 : ui32, handshake.name = "extsi7"} : <i1> to <i2>
    %10 = mux %17#0 [%7, %falseResult_11, %falseResult_21, %93] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i2>, [<i32>, <i32>, <i32>, <i32>] to <i32>
    %11 = buffer %10 {handshake.bb = 1 : ui32, handshake.name = "buffer0", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i32>
    %12 = buffer %11 {handshake.bb = 1 : ui32, handshake.name = "buffer1", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i32>
    %13:2 = fork [2] %12 {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <i32>
    %14 = mux %17#1 [%8#1, %falseResult_7, %falseResult_25, %95] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i2>, [<i1>, <i1>, <i1>, <i1>] to <i1>
    %15 = mux %17#2 [%9, %falseResult_9, %falseResult_27, %97] {handshake.bb = 1 : ui32, handshake.name = "mux2"} : <i2>, [<i2>, <i2>, <i2>, <i2>] to <i2>
    %16 = mux %17#3 [%8#0, %falseResult_17, %falseResult_31, %85] {handshake.bb = 1 : ui32, handshake.name = "mux3"} : <i2>, [<i1>, <i1>, <i1>, <i1>] to <i1>
    %result, %index = control_merge [%0#1, %falseResult_15, %falseResult_29, %98]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>, <>, <>] to <>, <i2>
    %17:4 = fork [4] %index {handshake.bb = 1 : ui32, handshake.name = "fork4"} : <i2>
    %18 = source {handshake.bb = 1 : ui32, handshake.name = "source0"} : <>
    %19 = constant %18 {handshake.bb = 1 : ui32, handshake.name = "constant2", value = 10 : i5} : <>, <i5>
    %20 = extsi %19 {handshake.bb = 1 : ui32, handshake.name = "extsi2"} : <i5> to <i32>
    %21 = cmpi slt, %13#1, %20 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i32>
    %22 = buffer %16 {handshake.bb = 1 : ui32, handshake.name = "buffer6", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %23 = buffer %22 {handshake.bb = 1 : ui32, handshake.name = "buffer7", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i1>
    %24 = andi %21, %23 {handshake.bb = 1 : ui32, handshake.name = "andi0"} : <i1>
    %25:4 = fork [4] %24 {handshake.bb = 1 : ui32, handshake.name = "fork5"} : <i1>
    %26 = buffer %14 {handshake.bb = 1 : ui32, handshake.name = "buffer2", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %27 = buffer %26 {handshake.bb = 1 : ui32, handshake.name = "buffer3", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i1>
    %trueResult, %falseResult = cond_br %25#3, %27 {handshake.bb = 1 : ui32, handshake.name = "cond_br3"} : <i1>, <i1>
    %28 = buffer %15 {handshake.bb = 1 : ui32, handshake.name = "buffer4", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i2>
    %29 = buffer %28 {handshake.bb = 1 : ui32, handshake.name = "buffer5", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i2>
    %trueResult_0, %falseResult_1 = cond_br %25#2, %29 {handshake.bb = 1 : ui32, handshake.name = "cond_br4"} : <i1>, <i2>
    %trueResult_2, %falseResult_3 = cond_br %25#1, %13#0 {handshake.bb = 1 : ui32, handshake.name = "cond_br5"} : <i1>, <i32>
    sink %falseResult_3 {handshake.name = "sink0"} : <i32>
    %30 = buffer %result {handshake.bb = 1 : ui32, handshake.name = "buffer8", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <>
    %31 = buffer %30 {handshake.bb = 1 : ui32, handshake.name = "buffer9", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %trueResult_4, %falseResult_5 = cond_br %25#0, %31 {handshake.bb = 1 : ui32, handshake.name = "cond_br6"} : <i1>, <>
    %32 = buffer %trueResult_2 {handshake.bb = 2 : ui32, handshake.name = "buffer14", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i32>
    %33 = buffer %32 {handshake.bb = 2 : ui32, handshake.name = "buffer15", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i32>
    %34:3 = fork [3] %33 {handshake.bb = 2 : ui32, handshake.name = "fork6"} : <i32>
    %35 = trunci %34#2 {handshake.bb = 2 : ui32, handshake.name = "trunci0"} : <i32> to <i4>
    %36 = trunci %34#1 {handshake.bb = 2 : ui32, handshake.name = "trunci1"} : <i32> to <i4>
    %37 = buffer %trueResult_4 {handshake.bb = 2 : ui32, handshake.name = "buffer16", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <>
    %38 = buffer %37 {handshake.bb = 2 : ui32, handshake.name = "buffer17", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %39:3 = lazy_fork [3] %38 {handshake.bb = 2 : ui32, handshake.name = "lazy_fork0"} : <>
    %40 = buffer %39#2 {handshake.bb = 2 : ui32, handshake.name = "buffer19", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %41 = fork [1] %40 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"0": [1,inf], [0,inf], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork7"} : <>
    %42 = constant %41 {handshake.bb = 2 : ui32, handshake.name = "constant8", value = false} : <>, <i1>
    %43 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %44 = constant %43 {handshake.bb = 2 : ui32, handshake.name = "constant9", value = -1 : i32} : <>, <i32>
    %addressResult, %dataResult = load[%36] %3#0 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[<"store1" (0)>]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 0>, handshake.name = "load3"} : <i4>, <i32>, <i4>, <i32>
    %45 = cmpi ne, %dataResult, %44 {handshake.bb = 2 : ui32, handshake.name = "cmpi1"} : <i32>
    %46:6 = fork [6] %45 {handshake.bb = 2 : ui32, handshake.name = "fork8"} : <i1>
    %47 = buffer %trueResult {handshake.bb = 2 : ui32, handshake.name = "buffer10", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %48 = buffer %47 {handshake.bb = 2 : ui32, handshake.name = "buffer11", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i1>
    %trueResult_6, %falseResult_7 = cond_br %46#5, %48 {handshake.bb = 2 : ui32, handshake.name = "cond_br7"} : <i1>, <i1>
    %49 = buffer %trueResult_0 {handshake.bb = 2 : ui32, handshake.name = "buffer12", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i2>
    %50 = buffer %49 {handshake.bb = 2 : ui32, handshake.name = "buffer13", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i2>
    %trueResult_8, %falseResult_9 = cond_br %46#4, %50 {handshake.bb = 2 : ui32, handshake.name = "cond_br8"} : <i1>, <i2>
    %trueResult_10, %falseResult_11 = cond_br %46#3, %34#0 {handshake.bb = 2 : ui32, handshake.name = "cond_br9"} : <i1>, <i32>
    %trueResult_12, %falseResult_13 = cond_br %46#2, %35 {handshake.bb = 2 : ui32, handshake.name = "cond_br10"} : <i1>, <i4>
    sink %falseResult_13 {handshake.name = "sink2"} : <i4>
    %51 = buffer %39#0 {handshake.bb = 2 : ui32, handshake.name = "buffer18", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <>
    %trueResult_14, %falseResult_15 = cond_br %46#1, %51 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,inf], [1,inf], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "cond_br11"} : <i1>, <>
    %trueResult_16, %falseResult_17 = cond_br %46#0, %42 {handshake.bb = 2 : ui32, handshake.name = "cond_br12"} : <i1>, <i1>
    sink %trueResult_16 {handshake.name = "sink3"} : <i1>
    %52 = buffer %trueResult_12 {handshake.bb = 3 : ui32, handshake.name = "buffer26", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i4>
    %53 = buffer %52 {handshake.bb = 3 : ui32, handshake.name = "buffer27", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i4>
    %54:2 = fork [2] %53 {handshake.bb = 3 : ui32, handshake.name = "fork9"} : <i4>
    %55 = buffer %trueResult_14 {handshake.bb = 3 : ui32, handshake.name = "buffer28", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <>
    %56 = buffer %55 {handshake.bb = 3 : ui32, handshake.name = "buffer29", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %57:2 = lazy_fork [2] %56 {handshake.bb = 3 : ui32, handshake.name = "lazy_fork1"} : <>
    %58 = source {handshake.bb = 3 : ui32, handshake.name = "source2"} : <>
    %59 = constant %58 {handshake.bb = 3 : ui32, handshake.name = "constant3", value = 1 : i2} : <>, <i2>
    %60 = source {handshake.bb = 3 : ui32, handshake.name = "source3"} : <>
    %61 = constant %60 {handshake.bb = 3 : ui32, handshake.name = "constant4", value = false} : <>, <i1>
    %62 = extsi %61 {handshake.bb = 3 : ui32, handshake.name = "extsi4"} : <i1> to <i32>
    %63:2 = fork [2] %62 {handshake.bb = 3 : ui32, handshake.name = "fork10"} : <i32>
    %addressResult_18, %dataResult_19 = load[%54#1] %3#1 {handshake.bb = 3 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[<"store1" (0)>]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 1>, handshake.name = "load4"} : <i4>, <i32>, <i4>, <i32>
    %64:2 = fork [2] %dataResult_19 {handshake.bb = 3 : ui32, handshake.name = "fork11"} : <i32>
    %65 = cmpi eq, %64#1, %63#1 {handshake.bb = 3 : ui32, handshake.name = "cmpi2"} : <i32>
    %66 = cmpi ne, %64#0, %63#0 {handshake.bb = 3 : ui32, handshake.name = "cmpi3"} : <i32>
    %67:8 = fork [8] %66 {handshake.bb = 3 : ui32, handshake.name = "fork12"} : <i1>
    %68 = buffer %trueResult_6 {handshake.bb = 3 : ui32, handshake.name = "buffer20", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %69 = buffer %68 {handshake.bb = 3 : ui32, handshake.name = "buffer21", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i1>
    %70 = andi %67#7, %69 {handshake.bb = 3 : ui32, handshake.name = "andi1"} : <i1>
    %71 = buffer %trueResult_8 {handshake.bb = 3 : ui32, handshake.name = "buffer22", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i2>
    %72 = buffer %71 {handshake.bb = 3 : ui32, handshake.name = "buffer23", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i2>
    %73 = select %65[%59, %72] {handshake.bb = 3 : ui32, handshake.name = "select0"} : <i1>, <i2>
    %74 = buffer %trueResult_10 {handshake.bb = 3 : ui32, handshake.name = "buffer24", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i32>
    %75 = buffer %74 {handshake.bb = 3 : ui32, handshake.name = "buffer25", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i32>
    %trueResult_20, %falseResult_21 = cond_br %67#6, %75 {handshake.bb = 3 : ui32, handshake.name = "cond_br13"} : <i1>, <i32>
    %trueResult_22, %falseResult_23 = cond_br %67#5, %54#0 {handshake.bb = 3 : ui32, handshake.name = "cond_br14"} : <i1>, <i4>
    sink %falseResult_23 {handshake.name = "sink5"} : <i4>
    %trueResult_24, %falseResult_25 = cond_br %67#4, %70 {handshake.bb = 3 : ui32, handshake.name = "cond_br15"} : <i1>, <i1>
    %trueResult_26, %falseResult_27 = cond_br %67#3, %73 {handshake.bb = 3 : ui32, handshake.name = "cond_br16"} : <i1>, <i2>
    %76 = buffer %57#0 {handshake.bb = 3 : ui32, handshake.name = "buffer30", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <>
    %trueResult_28, %falseResult_29 = cond_br %67#2, %76 {handshake.bb = 3 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,inf], [1,inf], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "cond_br17"} : <i1>, <>
    %trueResult_30, %falseResult_31 = cond_br %67#0, %67#1 {handshake.bb = 3 : ui32, handshake.name = "cond_br18"} : <i1>, <i1>
    sink %trueResult_30 {handshake.name = "sink6"} : <i1>
    %77 = buffer %trueResult_22 {handshake.bb = 4 : ui32, handshake.name = "buffer33", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i4>
    %78 = buffer %77 {handshake.bb = 4 : ui32, handshake.name = "buffer34", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i4>
    %79:2 = fork [2] %78 {handshake.bb = 4 : ui32, handshake.name = "fork13"} : <i4>
    %80 = buffer %trueResult_28 {handshake.bb = 4 : ui32, handshake.name = "buffer39", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <>
    %81 = buffer %80 {handshake.bb = 4 : ui32, handshake.name = "buffer40", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %82:3 = lazy_fork [3] %81 {handshake.bb = 4 : ui32, handshake.name = "lazy_fork2"} : <>
    %83 = buffer %82#2 {handshake.bb = 4 : ui32, handshake.name = "buffer42", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %84 = fork [1] %83 {handshake.bb = 4 : ui32, handshake.bufProps = #handshake<bufProps{"0": [1,inf], [0,inf], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork14"} : <>
    %85 = constant %84 {handshake.bb = 4 : ui32, handshake.name = "constant12", value = true} : <>, <i1>
    %86 = source {handshake.bb = 4 : ui32, handshake.name = "source4"} : <>
    %87 = constant %86 {handshake.bb = 4 : ui32, handshake.name = "constant5", value = 1 : i2} : <>, <i2>
    %88 = extsi %87 {handshake.bb = 4 : ui32, handshake.name = "extsi5"} : <i2> to <i32>
    %89:2 = fork [2] %88 {handshake.bb = 4 : ui32, handshake.name = "fork15"} : <i32>
    %addressResult_32, %dataResult_33 = load[%79#1] %3#2 {handshake.bb = 4 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[<"store1" (0)>]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 2>, handshake.name = "load5"} : <i4>, <i32>, <i4>, <i32>
    %90 = addi %dataResult_33, %89#1 {handshake.bb = 4 : ui32, handshake.name = "addi0"} : <i32>
    %addressResult_34, %dataResult_35 = store[%79#0] %90 {handshake.bb = 4 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[<"load3" (0)>, <"load4" (0)>, <"load5" (0)>, <"store1" (0)>]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 2>, handshake.name = "store1"} : <i4>, <i32>, <i4>, <i32>
    %91 = buffer %trueResult_20 {handshake.bb = 4 : ui32, handshake.name = "buffer31", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i32>
    %92 = buffer %91 {handshake.bb = 4 : ui32, handshake.name = "buffer32", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i32>
    %93 = addi %92, %89#0 {handshake.bb = 4 : ui32, handshake.name = "addi1"} : <i32>
    %94 = buffer %trueResult_24 {handshake.bb = 4 : ui32, handshake.name = "buffer35", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %95 = buffer %94 {handshake.bb = 4 : ui32, handshake.name = "buffer36", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i1>
    %96 = buffer %trueResult_26 {handshake.bb = 4 : ui32, handshake.name = "buffer37", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i2>
    %97 = buffer %96 {handshake.bb = 4 : ui32, handshake.name = "buffer38", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i2>
    %98 = buffer %82#0 {handshake.bb = 4 : ui32, handshake.name = "buffer41", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <>
    %99 = buffer %falseResult_1 {handshake.bb = 5 : ui32, handshake.name = "buffer45", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i2>
    %100 = buffer %99 {handshake.bb = 5 : ui32, handshake.name = "buffer46", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i2>
    %101 = extsi %100 {handshake.bb = 5 : ui32, handshake.name = "extsi9"} : <i2> to <i3>
    %102 = source {handshake.bb = 5 : ui32, handshake.name = "source5"} : <>
    %103 = constant %102 {handshake.bb = 5 : ui32, handshake.name = "constant15", value = 2 : i3} : <>, <i3>
    %104 = buffer %falseResult {handshake.bb = 5 : ui32, handshake.name = "buffer43", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %105 = buffer %104 {handshake.bb = 5 : ui32, handshake.name = "buffer44", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i1>
    %106 = select %105[%103, %101] {handshake.bb = 5 : ui32, handshake.name = "select1"} : <i1>, <i3>
    %107 = extsi %106 {handshake.bb = 5 : ui32, handshake.name = "extsi10"} : <i3> to <i32>
    end {handshake.bb = 5 : ui32, handshake.name = "end0"} %107, %3#3, %0#0 : <i32>, <>, <>
  }
}

