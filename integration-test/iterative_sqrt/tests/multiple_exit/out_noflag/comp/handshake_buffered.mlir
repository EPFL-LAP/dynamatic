module {
  handshake.func @multiple_exit(%arg0: memref<10xi32>, %arg1: !handshake.control<>, %arg2: !handshake.control<>, ...) -> (!handshake.channel<i32>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["arr", "arr_start", "start"], resNames = ["out0", "arr_end", "end"]} {
    %0:4 = fork [4] %arg2 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork0"} : <>
    %1 = buffer %result_42 {handshake.name = "buffer47", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <>
    %2 = buffer %1 {handshake.name = "buffer48", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %3:4 = lsq[%arg0 : memref<10xi32>] (%arg1, %46#1, %addressResult, %68#1, %addressResult_22, %97#1, %addressResult_38, %addressResult_40, %dataResult_41, %2)  {groupSizes = [1 : i32, 1 : i32, 2 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "5": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "6": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "7": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "8": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "9": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "lsq1"} : (!handshake.control<>, !handshake.control<>, !handshake.channel<i4>, !handshake.control<>, !handshake.channel<i4>, !handshake.control<>, !handshake.channel<i4>, !handshake.channel<i4>, !handshake.channel<i32>, !handshake.control<>) -> (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>)
    %4 = constant %0#3 {handshake.bb = 0 : ui32, handshake.name = "constant1", value = false} : <>, <i1>
    %5:2 = fork [2] %4 {handshake.bb = 0 : ui32, handshake.name = "fork1"} : <i1>
    %6 = constant %0#2 {handshake.bb = 0 : ui32, handshake.name = "constant6", value = true} : <>, <i1>
    %7 = br %5#1 {handshake.bb = 0 : ui32, handshake.name = "br1"} : <i1>
    %8 = extsi %7 {handshake.bb = 0 : ui32, handshake.name = "extsi8"} : <i1> to <i32>
    %9 = br %6 {handshake.bb = 0 : ui32, handshake.name = "br2"} : <i1>
    %10:2 = fork [2] %9 {handshake.bb = 0 : ui32, handshake.name = "fork2"} : <i1>
    %11 = br %5#0 {handshake.bb = 0 : ui32, handshake.name = "br3"} : <i1>
    %12 = extsi %11 {handshake.bb = 0 : ui32, handshake.name = "extsi7"} : <i1> to <i2>
    %13 = br %0#1 {handshake.bb = 0 : ui32, handshake.name = "br4"} : <>
    %14 = mux %21#0 [%8, %falseResult_13, %falseResult_25, %109] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i2>, [<i32>, <i32>, <i32>, <i32>] to <i32>
    %15 = buffer %14 {handshake.bb = 1 : ui32, handshake.name = "buffer0", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i32>
    %16 = buffer %15 {handshake.bb = 1 : ui32, handshake.name = "buffer1", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i32>
    %17:2 = fork [2] %16 {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <i32>
    %18 = mux %21#1 [%10#1, %falseResult_9, %falseResult_29, %112] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i2>, [<i1>, <i1>, <i1>, <i1>] to <i1>
    %19 = mux %21#2 [%12, %falseResult_11, %falseResult_31, %115] {handshake.bb = 1 : ui32, handshake.name = "mux2"} : <i2>, [<i2>, <i2>, <i2>, <i2>] to <i2>
    %20 = mux %21#3 [%10#0, %falseResult_19, %falseResult_35, %116] {handshake.bb = 1 : ui32, handshake.name = "mux3"} : <i2>, [<i1>, <i1>, <i1>, <i1>] to <i1>
    %result, %index = control_merge [%13, %falseResult_17, %falseResult_33, %118]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>, <>, <>] to <>, <i2>
    %21:4 = fork [4] %index {handshake.bb = 1 : ui32, handshake.name = "fork4"} : <i2>
    %22 = source {handshake.bb = 1 : ui32, handshake.name = "source0"} : <>
    %23 = constant %22 {handshake.bb = 1 : ui32, handshake.name = "constant2", value = 10 : i5} : <>, <i5>
    %24 = extsi %23 {handshake.bb = 1 : ui32, handshake.name = "extsi2"} : <i5> to <i32>
    %25 = cmpi slt, %17#1, %24 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i32>
    %26 = buffer %20 {handshake.bb = 1 : ui32, handshake.name = "buffer6", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %27 = buffer %26 {handshake.bb = 1 : ui32, handshake.name = "buffer7", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i1>
    %28 = andi %25, %27 {handshake.bb = 1 : ui32, handshake.name = "andi0"} : <i1>
    %29:4 = fork [4] %28 {handshake.bb = 1 : ui32, handshake.name = "fork5"} : <i1>
    %30 = buffer %18 {handshake.bb = 1 : ui32, handshake.name = "buffer2", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %31 = buffer %30 {handshake.bb = 1 : ui32, handshake.name = "buffer3", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i1>
    %trueResult, %falseResult = cond_br %29#3, %31 {handshake.bb = 1 : ui32, handshake.name = "cond_br3"} : <i1>, <i1>
    %32 = buffer %19 {handshake.bb = 1 : ui32, handshake.name = "buffer4", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i2>
    %33 = buffer %32 {handshake.bb = 1 : ui32, handshake.name = "buffer5", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i2>
    %trueResult_0, %falseResult_1 = cond_br %29#2, %33 {handshake.bb = 1 : ui32, handshake.name = "cond_br4"} : <i1>, <i2>
    %trueResult_2, %falseResult_3 = cond_br %29#1, %17#0 {handshake.bb = 1 : ui32, handshake.name = "cond_br5"} : <i1>, <i32>
    sink %falseResult_3 {handshake.name = "sink0"} : <i32>
    %34 = buffer %result {handshake.bb = 1 : ui32, handshake.name = "buffer8", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <>
    %35 = buffer %34 {handshake.bb = 1 : ui32, handshake.name = "buffer9", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %trueResult_4, %falseResult_5 = cond_br %29#0, %35 {handshake.bb = 1 : ui32, handshake.name = "cond_br6"} : <i1>, <>
    %36 = merge %trueResult {handshake.bb = 2 : ui32, handshake.name = "merge0"} : <i1>
    %37 = merge %trueResult_0 {handshake.bb = 2 : ui32, handshake.name = "merge1"} : <i2>
    %38 = merge %trueResult_2 {handshake.bb = 2 : ui32, handshake.name = "merge2"} : <i32>
    %39 = buffer %38 {handshake.bb = 2 : ui32, handshake.name = "buffer14", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i32>
    %40 = buffer %39 {handshake.bb = 2 : ui32, handshake.name = "buffer15", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i32>
    %41:3 = fork [3] %40 {handshake.bb = 2 : ui32, handshake.name = "fork6"} : <i32>
    %42 = trunci %41#2 {handshake.bb = 2 : ui32, handshake.name = "trunci0"} : <i32> to <i4>
    %43 = trunci %41#1 {handshake.bb = 2 : ui32, handshake.name = "trunci1"} : <i32> to <i4>
    %result_6, %index_7 = control_merge [%trueResult_4]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>] to <>, <i1>
    sink %index_7 {handshake.name = "sink1"} : <i1>
    %44 = buffer %result_6 {handshake.bb = 2 : ui32, handshake.name = "buffer16", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <>
    %45 = buffer %44 {handshake.bb = 2 : ui32, handshake.name = "buffer17", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %46:3 = lazy_fork [3] %45 {handshake.bb = 2 : ui32, handshake.name = "lazy_fork0"} : <>
    %47 = buffer %46#2 {handshake.bb = 2 : ui32, handshake.name = "buffer19", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %48 = fork [1] %47 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"0": [1,inf], [0,inf], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork7"} : <>
    %49 = constant %48 {handshake.bb = 2 : ui32, handshake.name = "constant8", value = false} : <>, <i1>
    %50 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %51 = constant %50 {handshake.bb = 2 : ui32, handshake.name = "constant9", value = -1 : i32} : <>, <i32>
    %addressResult, %dataResult = load[%43] %3#0 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[<"store1" (0)>]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 0>, handshake.name = "load3"} : <i4>, <i32>, <i4>, <i32>
    %52 = cmpi ne, %dataResult, %51 {handshake.bb = 2 : ui32, handshake.name = "cmpi1"} : <i32>
    %53:6 = fork [6] %52 {handshake.bb = 2 : ui32, handshake.name = "fork8"} : <i1>
    %54 = buffer %36 {handshake.bb = 2 : ui32, handshake.name = "buffer10", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %55 = buffer %54 {handshake.bb = 2 : ui32, handshake.name = "buffer11", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i1>
    %trueResult_8, %falseResult_9 = cond_br %53#5, %55 {handshake.bb = 2 : ui32, handshake.name = "cond_br7"} : <i1>, <i1>
    %56 = buffer %37 {handshake.bb = 2 : ui32, handshake.name = "buffer12", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i2>
    %57 = buffer %56 {handshake.bb = 2 : ui32, handshake.name = "buffer13", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i2>
    %trueResult_10, %falseResult_11 = cond_br %53#4, %57 {handshake.bb = 2 : ui32, handshake.name = "cond_br8"} : <i1>, <i2>
    %trueResult_12, %falseResult_13 = cond_br %53#3, %41#0 {handshake.bb = 2 : ui32, handshake.name = "cond_br9"} : <i1>, <i32>
    %trueResult_14, %falseResult_15 = cond_br %53#2, %42 {handshake.bb = 2 : ui32, handshake.name = "cond_br10"} : <i1>, <i4>
    sink %falseResult_15 {handshake.name = "sink2"} : <i4>
    %58 = buffer %46#0 {handshake.bb = 2 : ui32, handshake.name = "buffer18", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <>
    %trueResult_16, %falseResult_17 = cond_br %53#1, %58 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,inf], [1,inf], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "cond_br11"} : <i1>, <>
    %trueResult_18, %falseResult_19 = cond_br %53#0, %49 {handshake.bb = 2 : ui32, handshake.name = "cond_br12"} : <i1>, <i1>
    sink %trueResult_18 {handshake.name = "sink3"} : <i1>
    %59 = merge %trueResult_8 {handshake.bb = 3 : ui32, handshake.name = "merge3"} : <i1>
    %60 = merge %trueResult_10 {handshake.bb = 3 : ui32, handshake.name = "merge4"} : <i2>
    %61 = merge %trueResult_12 {handshake.bb = 3 : ui32, handshake.name = "merge5"} : <i32>
    %62 = merge %trueResult_14 {handshake.bb = 3 : ui32, handshake.name = "merge6"} : <i4>
    %63 = buffer %62 {handshake.bb = 3 : ui32, handshake.name = "buffer26", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i4>
    %64 = buffer %63 {handshake.bb = 3 : ui32, handshake.name = "buffer27", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i4>
    %65:2 = fork [2] %64 {handshake.bb = 3 : ui32, handshake.name = "fork9"} : <i4>
    %result_20, %index_21 = control_merge [%trueResult_16]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>] to <>, <i1>
    sink %index_21 {handshake.name = "sink4"} : <i1>
    %66 = buffer %result_20 {handshake.bb = 3 : ui32, handshake.name = "buffer28", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <>
    %67 = buffer %66 {handshake.bb = 3 : ui32, handshake.name = "buffer29", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %68:2 = lazy_fork [2] %67 {handshake.bb = 3 : ui32, handshake.name = "lazy_fork1"} : <>
    %69 = source {handshake.bb = 3 : ui32, handshake.name = "source2"} : <>
    %70 = constant %69 {handshake.bb = 3 : ui32, handshake.name = "constant3", value = 1 : i2} : <>, <i2>
    %71 = source {handshake.bb = 3 : ui32, handshake.name = "source3"} : <>
    %72 = constant %71 {handshake.bb = 3 : ui32, handshake.name = "constant4", value = false} : <>, <i1>
    %73 = extsi %72 {handshake.bb = 3 : ui32, handshake.name = "extsi4"} : <i1> to <i32>
    %74:2 = fork [2] %73 {handshake.bb = 3 : ui32, handshake.name = "fork10"} : <i32>
    %addressResult_22, %dataResult_23 = load[%65#1] %3#1 {handshake.bb = 3 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[<"store1" (0)>]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 1>, handshake.name = "load4"} : <i4>, <i32>, <i4>, <i32>
    %75:2 = fork [2] %dataResult_23 {handshake.bb = 3 : ui32, handshake.name = "fork11"} : <i32>
    %76 = cmpi eq, %75#1, %74#1 {handshake.bb = 3 : ui32, handshake.name = "cmpi2"} : <i32>
    %77 = cmpi ne, %75#0, %74#0 {handshake.bb = 3 : ui32, handshake.name = "cmpi3"} : <i32>
    %78:8 = fork [8] %77 {handshake.bb = 3 : ui32, handshake.name = "fork12"} : <i1>
    %79 = buffer %59 {handshake.bb = 3 : ui32, handshake.name = "buffer20", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %80 = buffer %79 {handshake.bb = 3 : ui32, handshake.name = "buffer21", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i1>
    %81 = andi %78#7, %80 {handshake.bb = 3 : ui32, handshake.name = "andi1"} : <i1>
    %82 = buffer %60 {handshake.bb = 3 : ui32, handshake.name = "buffer22", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i2>
    %83 = buffer %82 {handshake.bb = 3 : ui32, handshake.name = "buffer23", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i2>
    %84 = select %76[%70, %83] {handshake.bb = 3 : ui32, handshake.name = "select0"} : <i1>, <i2>
    %85 = buffer %61 {handshake.bb = 3 : ui32, handshake.name = "buffer24", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i32>
    %86 = buffer %85 {handshake.bb = 3 : ui32, handshake.name = "buffer25", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i32>
    %trueResult_24, %falseResult_25 = cond_br %78#6, %86 {handshake.bb = 3 : ui32, handshake.name = "cond_br13"} : <i1>, <i32>
    %trueResult_26, %falseResult_27 = cond_br %78#5, %65#0 {handshake.bb = 3 : ui32, handshake.name = "cond_br14"} : <i1>, <i4>
    sink %falseResult_27 {handshake.name = "sink5"} : <i4>
    %trueResult_28, %falseResult_29 = cond_br %78#4, %81 {handshake.bb = 3 : ui32, handshake.name = "cond_br15"} : <i1>, <i1>
    %trueResult_30, %falseResult_31 = cond_br %78#3, %84 {handshake.bb = 3 : ui32, handshake.name = "cond_br16"} : <i1>, <i2>
    %87 = buffer %68#0 {handshake.bb = 3 : ui32, handshake.name = "buffer30", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <>
    %trueResult_32, %falseResult_33 = cond_br %78#2, %87 {handshake.bb = 3 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,inf], [1,inf], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "cond_br17"} : <i1>, <>
    %trueResult_34, %falseResult_35 = cond_br %78#0, %78#1 {handshake.bb = 3 : ui32, handshake.name = "cond_br18"} : <i1>, <i1>
    sink %trueResult_34 {handshake.name = "sink6"} : <i1>
    %88 = merge %trueResult_24 {handshake.bb = 4 : ui32, handshake.name = "merge7"} : <i32>
    %89 = merge %trueResult_26 {handshake.bb = 4 : ui32, handshake.name = "merge8"} : <i4>
    %90 = buffer %89 {handshake.bb = 4 : ui32, handshake.name = "buffer33", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i4>
    %91 = buffer %90 {handshake.bb = 4 : ui32, handshake.name = "buffer34", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i4>
    %92:2 = fork [2] %91 {handshake.bb = 4 : ui32, handshake.name = "fork13"} : <i4>
    %93 = merge %trueResult_28 {handshake.bb = 4 : ui32, handshake.name = "merge9"} : <i1>
    %94 = merge %trueResult_30 {handshake.bb = 4 : ui32, handshake.name = "merge10"} : <i2>
    %result_36, %index_37 = control_merge [%trueResult_32]  {handshake.bb = 4 : ui32, handshake.name = "control_merge3"} : [<>] to <>, <i1>
    sink %index_37 {handshake.name = "sink7"} : <i1>
    %95 = buffer %result_36 {handshake.bb = 4 : ui32, handshake.name = "buffer39", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <>
    %96 = buffer %95 {handshake.bb = 4 : ui32, handshake.name = "buffer40", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %97:3 = lazy_fork [3] %96 {handshake.bb = 4 : ui32, handshake.name = "lazy_fork2"} : <>
    %98 = buffer %97#2 {handshake.bb = 4 : ui32, handshake.name = "buffer42", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %99 = fork [1] %98 {handshake.bb = 4 : ui32, handshake.bufProps = #handshake<bufProps{"0": [1,inf], [0,inf], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork14"} : <>
    %100 = constant %99 {handshake.bb = 4 : ui32, handshake.name = "constant12", value = true} : <>, <i1>
    %101 = source {handshake.bb = 4 : ui32, handshake.name = "source4"} : <>
    %102 = constant %101 {handshake.bb = 4 : ui32, handshake.name = "constant5", value = 1 : i2} : <>, <i2>
    %103 = extsi %102 {handshake.bb = 4 : ui32, handshake.name = "extsi5"} : <i2> to <i32>
    %104:2 = fork [2] %103 {handshake.bb = 4 : ui32, handshake.name = "fork15"} : <i32>
    %addressResult_38, %dataResult_39 = load[%92#1] %3#2 {handshake.bb = 4 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[<"store1" (0)>]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 2>, handshake.name = "load5"} : <i4>, <i32>, <i4>, <i32>
    %105 = addi %dataResult_39, %104#1 {handshake.bb = 4 : ui32, handshake.name = "addi0"} : <i32>
    %addressResult_40, %dataResult_41 = store[%92#0] %105 {handshake.bb = 4 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[<"load3" (0)>, <"load4" (0)>, <"load5" (0)>, <"store1" (0)>]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 2>, handshake.name = "store1"} : <i4>, <i32>, <i4>, <i32>
    %106 = buffer %88 {handshake.bb = 4 : ui32, handshake.name = "buffer31", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i32>
    %107 = buffer %106 {handshake.bb = 4 : ui32, handshake.name = "buffer32", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i32>
    %108 = addi %107, %104#0 {handshake.bb = 4 : ui32, handshake.name = "addi1"} : <i32>
    %109 = br %108 {handshake.bb = 4 : ui32, handshake.name = "br5"} : <i32>
    %110 = buffer %93 {handshake.bb = 4 : ui32, handshake.name = "buffer35", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %111 = buffer %110 {handshake.bb = 4 : ui32, handshake.name = "buffer36", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i1>
    %112 = br %111 {handshake.bb = 4 : ui32, handshake.name = "br6"} : <i1>
    %113 = buffer %94 {handshake.bb = 4 : ui32, handshake.name = "buffer37", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i2>
    %114 = buffer %113 {handshake.bb = 4 : ui32, handshake.name = "buffer38", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i2>
    %115 = br %114 {handshake.bb = 4 : ui32, handshake.name = "br8"} : <i2>
    %116 = br %100 {handshake.bb = 4 : ui32, handshake.name = "br9"} : <i1>
    %117 = buffer %97#0 {handshake.bb = 4 : ui32, handshake.name = "buffer41", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <>
    %118 = br %117 {handshake.bb = 4 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [1,inf], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "br10"} : <>
    %119 = merge %falseResult {handshake.bb = 5 : ui32, handshake.name = "merge11"} : <i1>
    %120 = merge %falseResult_1 {handshake.bb = 5 : ui32, handshake.name = "merge12"} : <i2>
    %121 = buffer %120 {handshake.bb = 5 : ui32, handshake.name = "buffer45", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i2>
    %122 = buffer %121 {handshake.bb = 5 : ui32, handshake.name = "buffer46", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i2>
    %123 = extsi %122 {handshake.bb = 5 : ui32, handshake.name = "extsi9"} : <i2> to <i3>
    %result_42, %index_43 = control_merge [%falseResult_5]  {handshake.bb = 5 : ui32, handshake.name = "control_merge4"} : [<>] to <>, <i1>
    sink %index_43 {handshake.name = "sink8"} : <i1>
    %124 = source {handshake.bb = 5 : ui32, handshake.name = "source5"} : <>
    %125 = constant %124 {handshake.bb = 5 : ui32, handshake.name = "constant15", value = 2 : i3} : <>, <i3>
    %126 = buffer %119 {handshake.bb = 5 : ui32, handshake.name = "buffer43", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %127 = buffer %126 {handshake.bb = 5 : ui32, handshake.name = "buffer44", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i1>
    %128 = select %127[%125, %123] {handshake.bb = 5 : ui32, handshake.name = "select1"} : <i1>, <i3>
    %129 = extsi %128 {handshake.bb = 5 : ui32, handshake.name = "extsi10"} : <i3> to <i32>
    end {handshake.bb = 5 : ui32, handshake.name = "end0"} %129, %3#3, %0#0 : <i32>, <>, <>
  }
}

