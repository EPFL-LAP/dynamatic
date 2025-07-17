module {
  handshake.func @multiple_exit(%arg0: memref<10xi32>, %arg1: !handshake.channel<i32>, %arg2: !handshake.control<>, %arg3: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>) attributes {argNames = ["arr", "size", "arr_start", "start"], resNames = ["arr_end", "end"]} {
    %0:4 = fork [4] %arg3 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork0"} : <>
    %1 = buffer %result_36 {handshake.name = "buffer35", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <>
    %2 = buffer %1 {handshake.name = "buffer36", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %3:4 = lsq[%arg0 : memref<10xi32>] (%arg2, %37#1, %addressResult, %56#1, %addressResult_18, %75#1, %addressResult_32, %addressResult_34, %dataResult_35, %2)  {groupSizes = [1 : i32, 1 : i32, 2 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "5": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "6": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "7": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "8": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "9": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "lsq1"} : (!handshake.control<>, !handshake.control<>, !handshake.channel<i4>, !handshake.control<>, !handshake.channel<i4>, !handshake.control<>, !handshake.channel<i4>, !handshake.channel<i4>, !handshake.channel<i32>, !handshake.control<>) -> (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>)
    %4 = constant %0#3 {handshake.bb = 0 : ui32, handshake.name = "constant1", value = false} : <>, <i1>
    %5 = constant %0#2 {handshake.bb = 0 : ui32, handshake.name = "constant4", value = true} : <>, <i1>
    %6 = br %4 {handshake.bb = 0 : ui32, handshake.name = "br1"} : <i1>
    %7 = extsi %6 {handshake.bb = 0 : ui32, handshake.name = "extsi3"} : <i1> to <i32>
    %8 = br %5 {handshake.bb = 0 : ui32, handshake.name = "br2"} : <i1>
    %9 = br %arg1 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "br3"} : <i32>
    %10 = br %0#1 {handshake.bb = 0 : ui32, handshake.name = "br4"} : <>
    %11 = mux %20#0 [%7, %falseResult_9, %falseResult_23, %87] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i2>, [<i32>, <i32>, <i32>, <i32>] to <i32>
    %12 = buffer %11 {handshake.bb = 1 : ui32, handshake.name = "buffer0", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i32>
    %13 = buffer %12 {handshake.bb = 1 : ui32, handshake.name = "buffer1", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i32>
    %14:2 = fork [2] %13 {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <i32>
    %15 = mux %20#1 [%8, %falseResult_15, %falseResult_29, %88] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i2>, [<i1>, <i1>, <i1>, <i1>] to <i1>
    %16 = mux %20#2 [%9, %falseResult_7, %falseResult_21, %91] {handshake.bb = 1 : ui32, handshake.name = "mux2"} : <i2>, [<i32>, <i32>, <i32>, <i32>] to <i32>
    %17 = buffer %16 {handshake.bb = 1 : ui32, handshake.name = "buffer4", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i32>
    %18 = buffer %17 {handshake.bb = 1 : ui32, handshake.name = "buffer5", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i32>
    %19:2 = fork [2] %18 {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <i32>
    %result, %index = control_merge [%10, %falseResult_13, %falseResult_27, %93]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>, <>, <>] to <>, <i2>
    %20:3 = fork [3] %index {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <i2>
    %21 = cmpi slt, %14#1, %19#1 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i32>
    %22 = buffer %15 {handshake.bb = 1 : ui32, handshake.name = "buffer2", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %23 = buffer %22 {handshake.bb = 1 : ui32, handshake.name = "buffer3", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i1>
    %24 = andi %21, %23 {handshake.bb = 1 : ui32, handshake.name = "andi0"} : <i1>
    %25:3 = fork [3] %24 {handshake.bb = 1 : ui32, handshake.name = "fork4"} : <i1>
    %trueResult, %falseResult = cond_br %25#2, %14#0 {handshake.bb = 1 : ui32, handshake.name = "cond_br3"} : <i1>, <i32>
    sink %falseResult {handshake.name = "sink0"} : <i32>
    %trueResult_0, %falseResult_1 = cond_br %25#1, %19#0 {handshake.bb = 1 : ui32, handshake.name = "cond_br4"} : <i1>, <i32>
    sink %falseResult_1 {handshake.name = "sink1"} : <i32>
    %26 = buffer %result {handshake.bb = 1 : ui32, handshake.name = "buffer6", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <>
    %27 = buffer %26 {handshake.bb = 1 : ui32, handshake.name = "buffer7", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %trueResult_2, %falseResult_3 = cond_br %25#0, %27 {handshake.bb = 1 : ui32, handshake.name = "cond_br5"} : <i1>, <>
    %28 = merge %trueResult {handshake.bb = 2 : ui32, handshake.name = "merge0"} : <i32>
    %29 = buffer %28 {handshake.bb = 2 : ui32, handshake.name = "buffer8", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i32>
    %30 = buffer %29 {handshake.bb = 2 : ui32, handshake.name = "buffer9", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i32>
    %31:3 = fork [3] %30 {handshake.bb = 2 : ui32, handshake.name = "fork5"} : <i32>
    %32 = trunci %31#2 {handshake.bb = 2 : ui32, handshake.name = "trunci0"} : <i32> to <i4>
    %33 = trunci %31#1 {handshake.bb = 2 : ui32, handshake.name = "trunci1"} : <i32> to <i4>
    %34 = merge %trueResult_0 {handshake.bb = 2 : ui32, handshake.name = "merge1"} : <i32>
    %result_4, %index_5 = control_merge [%trueResult_2]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>] to <>, <i1>
    sink %index_5 {handshake.name = "sink2"} : <i1>
    %35 = buffer %result_4 {handshake.bb = 2 : ui32, handshake.name = "buffer12", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <>
    %36 = buffer %35 {handshake.bb = 2 : ui32, handshake.name = "buffer13", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %37:3 = lazy_fork [3] %36 {handshake.bb = 2 : ui32, handshake.name = "lazy_fork0"} : <>
    %38 = buffer %37#2 {handshake.bb = 2 : ui32, handshake.name = "buffer15", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %39 = fork [1] %38 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"0": [1,inf], [0,inf], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork6"} : <>
    %40 = constant %39 {handshake.bb = 2 : ui32, handshake.name = "constant5", value = false} : <>, <i1>
    %41 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %42 = constant %41 {handshake.bb = 2 : ui32, handshake.name = "constant6", value = -1 : i32} : <>, <i32>
    %addressResult, %dataResult = load[%33] %3#0 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[<"store1" (0)>]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 0>, handshake.name = "load3"} : <i4>, <i32>, <i4>, <i32>
    %43 = cmpi ne, %dataResult, %42 {handshake.bb = 2 : ui32, handshake.name = "cmpi1"} : <i32>
    %44:5 = fork [5] %43 {handshake.bb = 2 : ui32, handshake.name = "fork7"} : <i1>
    %45 = buffer %34 {handshake.bb = 2 : ui32, handshake.name = "buffer10", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i32>
    %46 = buffer %45 {handshake.bb = 2 : ui32, handshake.name = "buffer11", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i32>
    %trueResult_6, %falseResult_7 = cond_br %44#4, %46 {handshake.bb = 2 : ui32, handshake.name = "cond_br6"} : <i1>, <i32>
    %trueResult_8, %falseResult_9 = cond_br %44#3, %31#0 {handshake.bb = 2 : ui32, handshake.name = "cond_br7"} : <i1>, <i32>
    %trueResult_10, %falseResult_11 = cond_br %44#2, %32 {handshake.bb = 2 : ui32, handshake.name = "cond_br8"} : <i1>, <i4>
    sink %falseResult_11 {handshake.name = "sink3"} : <i4>
    %47 = buffer %37#0 {handshake.bb = 2 : ui32, handshake.name = "buffer14", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <>
    %trueResult_12, %falseResult_13 = cond_br %44#1, %47 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,inf], [1,inf], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "cond_br9"} : <i1>, <>
    %trueResult_14, %falseResult_15 = cond_br %44#0, %40 {handshake.bb = 2 : ui32, handshake.name = "cond_br10"} : <i1>, <i1>
    sink %trueResult_14 {handshake.name = "sink4"} : <i1>
    %48 = merge %trueResult_6 {handshake.bb = 3 : ui32, handshake.name = "merge2"} : <i32>
    %49 = merge %trueResult_8 {handshake.bb = 3 : ui32, handshake.name = "merge3"} : <i32>
    %50 = merge %trueResult_10 {handshake.bb = 3 : ui32, handshake.name = "merge4"} : <i4>
    %51 = buffer %50 {handshake.bb = 3 : ui32, handshake.name = "buffer20", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i4>
    %52 = buffer %51 {handshake.bb = 3 : ui32, handshake.name = "buffer21", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i4>
    %53:2 = fork [2] %52 {handshake.bb = 3 : ui32, handshake.name = "fork8"} : <i4>
    %result_16, %index_17 = control_merge [%trueResult_12]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>] to <>, <i1>
    sink %index_17 {handshake.name = "sink5"} : <i1>
    %54 = buffer %result_16 {handshake.bb = 3 : ui32, handshake.name = "buffer22", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <>
    %55 = buffer %54 {handshake.bb = 3 : ui32, handshake.name = "buffer23", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %56:2 = lazy_fork [2] %55 {handshake.bb = 3 : ui32, handshake.name = "lazy_fork1"} : <>
    %57 = source {handshake.bb = 3 : ui32, handshake.name = "source1"} : <>
    %58 = constant %57 {handshake.bb = 3 : ui32, handshake.name = "constant2", value = false} : <>, <i1>
    %59 = extsi %58 {handshake.bb = 3 : ui32, handshake.name = "extsi1"} : <i1> to <i32>
    %addressResult_18, %dataResult_19 = load[%53#1] %3#1 {handshake.bb = 3 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[<"store1" (0)>]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 1>, handshake.name = "load4"} : <i4>, <i32>, <i4>, <i32>
    %60 = cmpi ne, %dataResult_19, %59 {handshake.bb = 3 : ui32, handshake.name = "cmpi2"} : <i32>
    %61:6 = fork [6] %60 {handshake.bb = 3 : ui32, handshake.name = "fork9"} : <i1>
    %62 = buffer %48 {handshake.bb = 3 : ui32, handshake.name = "buffer16", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i32>
    %63 = buffer %62 {handshake.bb = 3 : ui32, handshake.name = "buffer17", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i32>
    %trueResult_20, %falseResult_21 = cond_br %61#5, %63 {handshake.bb = 3 : ui32, handshake.name = "cond_br11"} : <i1>, <i32>
    %64 = buffer %49 {handshake.bb = 3 : ui32, handshake.name = "buffer18", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i32>
    %65 = buffer %64 {handshake.bb = 3 : ui32, handshake.name = "buffer19", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i32>
    %trueResult_22, %falseResult_23 = cond_br %61#4, %65 {handshake.bb = 3 : ui32, handshake.name = "cond_br12"} : <i1>, <i32>
    %trueResult_24, %falseResult_25 = cond_br %61#3, %53#0 {handshake.bb = 3 : ui32, handshake.name = "cond_br13"} : <i1>, <i4>
    sink %falseResult_25 {handshake.name = "sink6"} : <i4>
    %66 = buffer %56#0 {handshake.bb = 3 : ui32, handshake.name = "buffer24", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <>
    %trueResult_26, %falseResult_27 = cond_br %61#2, %66 {handshake.bb = 3 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,inf], [1,inf], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "cond_br14"} : <i1>, <>
    %trueResult_28, %falseResult_29 = cond_br %61#0, %61#1 {handshake.bb = 3 : ui32, handshake.name = "cond_br15"} : <i1>, <i1>
    sink %trueResult_28 {handshake.name = "sink7"} : <i1>
    %67 = merge %trueResult_20 {handshake.bb = 4 : ui32, handshake.name = "merge5"} : <i32>
    %68 = merge %trueResult_22 {handshake.bb = 4 : ui32, handshake.name = "merge6"} : <i32>
    %69 = merge %trueResult_24 {handshake.bb = 4 : ui32, handshake.name = "merge7"} : <i4>
    %70 = buffer %69 {handshake.bb = 4 : ui32, handshake.name = "buffer29", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i4>
    %71 = buffer %70 {handshake.bb = 4 : ui32, handshake.name = "buffer30", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i4>
    %72:2 = fork [2] %71 {handshake.bb = 4 : ui32, handshake.name = "fork10"} : <i4>
    %result_30, %index_31 = control_merge [%trueResult_26]  {handshake.bb = 4 : ui32, handshake.name = "control_merge3"} : [<>] to <>, <i1>
    sink %index_31 {handshake.name = "sink8"} : <i1>
    %73 = buffer %result_30 {handshake.bb = 4 : ui32, handshake.name = "buffer31", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <>
    %74 = buffer %73 {handshake.bb = 4 : ui32, handshake.name = "buffer32", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %75:3 = lazy_fork [3] %74 {handshake.bb = 4 : ui32, handshake.name = "lazy_fork2"} : <>
    %76 = buffer %75#2 {handshake.bb = 4 : ui32, handshake.name = "buffer34", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %77 = fork [1] %76 {handshake.bb = 4 : ui32, handshake.bufProps = #handshake<bufProps{"0": [1,inf], [0,inf], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork11"} : <>
    %78 = constant %77 {handshake.bb = 4 : ui32, handshake.name = "constant8", value = true} : <>, <i1>
    %79 = source {handshake.bb = 4 : ui32, handshake.name = "source2"} : <>
    %80 = constant %79 {handshake.bb = 4 : ui32, handshake.name = "constant3", value = 1 : i2} : <>, <i2>
    %81 = extsi %80 {handshake.bb = 4 : ui32, handshake.name = "extsi2"} : <i2> to <i32>
    %82:2 = fork [2] %81 {handshake.bb = 4 : ui32, handshake.name = "fork12"} : <i32>
    %addressResult_32, %dataResult_33 = load[%72#1] %3#2 {handshake.bb = 4 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[<"store1" (0)>]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 2>, handshake.name = "load5"} : <i4>, <i32>, <i4>, <i32>
    %83 = addi %dataResult_33, %82#1 {handshake.bb = 4 : ui32, handshake.name = "addi0"} : <i32>
    %addressResult_34, %dataResult_35 = store[%72#0] %83 {handshake.bb = 4 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[<"load3" (0)>, <"load4" (0)>, <"load5" (0)>, <"store1" (0)>]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 2>, handshake.name = "store1"} : <i4>, <i32>, <i4>, <i32>
    %84 = buffer %68 {handshake.bb = 4 : ui32, handshake.name = "buffer27", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i32>
    %85 = buffer %84 {handshake.bb = 4 : ui32, handshake.name = "buffer28", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i32>
    %86 = addi %85, %82#0 {handshake.bb = 4 : ui32, handshake.name = "addi1"} : <i32>
    %87 = br %86 {handshake.bb = 4 : ui32, handshake.name = "br5"} : <i32>
    %88 = br %78 {handshake.bb = 4 : ui32, handshake.name = "br6"} : <i1>
    %89 = buffer %67 {handshake.bb = 4 : ui32, handshake.name = "buffer25", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i32>
    %90 = buffer %89 {handshake.bb = 4 : ui32, handshake.name = "buffer26", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i32>
    %91 = br %90 {handshake.bb = 4 : ui32, handshake.name = "br8"} : <i32>
    %92 = buffer %75#0 {handshake.bb = 4 : ui32, handshake.name = "buffer33", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <>
    %93 = br %92 {handshake.bb = 4 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [1,inf], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "br9"} : <>
    %result_36, %index_37 = control_merge [%falseResult_3]  {handshake.bb = 5 : ui32, handshake.name = "control_merge4"} : [<>] to <>, <i1>
    sink %index_37 {handshake.name = "sink9"} : <i1>
    end {handshake.bb = 5 : ui32, handshake.name = "end0"} %3#3, %0#0 : <>, <>
  }
}

