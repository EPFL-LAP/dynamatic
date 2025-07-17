module {
  handshake.func @iterative_sqrt(%arg0: memref<10xi32>, %arg1: !handshake.control<>, %arg2: !handshake.control<>, ...) -> (!handshake.channel<i32>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["A", "A_start", "start"], resNames = ["out0", "A_end", "end"]} {
    %0:2 = fork [2] %arg2 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork0"} : <>
    %1 = buffer %65#1 {handshake.name = "buffer21", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %outputs:3, %memEnd = mem_controller[%arg0 : memref<10xi32>] %arg1 (%addressResult, %addressResult_4, %52, %35, %2#1, %2#2, %2#3) %1 {connectedBlocks = [1 : i32, 2 : i32, 4 : i32, 3 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "5": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "6": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "7": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "8": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "9": [1,inf], [0,inf], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller1"} :    (!handshake.channel<i4>, !handshake.channel<i4>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i4>, !handshake.channel<i4>, !handshake.channel<i32>) -> (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>)
    %2:4 = lsq[MC] (%31#1, %addressResult_12, %dataResult_13, %48#1, %addressResult_16, %dataResult_17, %65#2, %addressResult_20, %outputs#2)  {groupSizes = [1 : i32, 1 : i32, 1 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "5": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "6": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "7": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "8": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "lsq1"} : (!handshake.control<>, !handshake.channel<i4>, !handshake.channel<i32>, !handshake.control<>, !handshake.channel<i4>, !handshake.channel<i32>, !handshake.control<>, !handshake.channel<i4>, !handshake.channel<i32>) -> (!handshake.channel<i32>, !handshake.channel<i4>, !handshake.channel<i4>, !handshake.channel<i32>)
    %3 = br %0#1 {handshake.bb = 0 : ui32, handshake.name = "br2"} : <>
    %result, %index = control_merge [%3, %44, %62]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>, <>] to <>, <i2>
    sink %index {handshake.name = "sink0"} : <i2>
    %4 = buffer %result {handshake.bb = 1 : ui32, handshake.name = "buffer0", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <>
    %5 = buffer %4 {handshake.bb = 1 : ui32, handshake.name = "buffer1", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %6:2 = fork [2] %5 {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <>
    %7 = constant %6#1 {handshake.bb = 1 : ui32, handshake.name = "constant4", value = false} : <>, <i1>
    %8 = extui %7 {handshake.bb = 1 : ui32, handshake.name = "extui0"} : <i1> to <i4>
    %9 = source {handshake.bb = 1 : ui32, handshake.name = "source0"} : <>
    %10 = constant %9 {handshake.bb = 1 : ui32, handshake.name = "constant14", value = 10 : i5} : <>, <i5>
    %11 = extsi %10 {handshake.bb = 1 : ui32, handshake.name = "extsi1"} : <i5> to <i32>
    %addressResult, %dataResult = load[%8] %outputs#0 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i4>, <i32>, <i4>, <i32>
    %12:2 = fork [2] %dataResult {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <i32>
    %13 = cmpi sgt, %12#1, %11 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i32>
    %14:2 = fork [2] %13 {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <i1>
    %trueResult, %falseResult = cond_br %14#1, %12#0 {handshake.bb = 1 : ui32, handshake.name = "cond_br2"} : <i1>, <i32>
    sink %falseResult {handshake.name = "sink1"} : <i32>
    %trueResult_0, %falseResult_1 = cond_br %14#0, %6#0 {handshake.bb = 1 : ui32, handshake.name = "cond_br3"} : <i1>, <>
    %15 = merge %trueResult {handshake.bb = 2 : ui32, handshake.name = "merge0"} : <i32>
    %result_2, %index_3 = control_merge [%trueResult_0]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>] to <>, <i1>
    sink %index_3 {handshake.name = "sink2"} : <i1>
    %16 = buffer %result_2 {handshake.bb = 2 : ui32, handshake.name = "buffer4", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <>
    %17 = buffer %16 {handshake.bb = 2 : ui32, handshake.name = "buffer5", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %18:2 = fork [2] %17 {handshake.bb = 2 : ui32, handshake.name = "fork4"} : <>
    %19 = constant %18#1 {handshake.bb = 2 : ui32, handshake.name = "constant15", value = 1 : i2} : <>, <i2>
    %20 = extui %19 {handshake.bb = 2 : ui32, handshake.name = "extui1"} : <i2> to <i4>
    %21 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %22 = constant %21 {handshake.bb = 2 : ui32, handshake.name = "constant16", value = 10 : i5} : <>, <i5>
    %23 = extsi %22 {handshake.bb = 2 : ui32, handshake.name = "extsi3"} : <i5> to <i32>
    %addressResult_4, %dataResult_5 = load[%20] %outputs#1 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i4>, <i32>, <i4>, <i32>
    %24 = cmpi slt, %dataResult_5, %23 {handshake.bb = 2 : ui32, handshake.name = "cmpi1"} : <i32>
    %25:2 = fork [2] %24 {handshake.bb = 2 : ui32, handshake.name = "fork5"} : <i1>
    %26 = buffer %15 {handshake.bb = 2 : ui32, handshake.name = "buffer2", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i32>
    %27 = buffer %26 {handshake.bb = 2 : ui32, handshake.name = "buffer3", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i32>
    %trueResult_6, %falseResult_7 = cond_br %25#1, %27 {handshake.bb = 2 : ui32, handshake.name = "cond_br4"} : <i1>, <i32>
    %trueResult_8, %falseResult_9 = cond_br %25#0, %18#0 {handshake.bb = 2 : ui32, handshake.name = "cond_br5"} : <i1>, <>
    %28 = merge %trueResult_6 {handshake.bb = 3 : ui32, handshake.name = "merge1"} : <i32>
    %result_10, %index_11 = control_merge [%trueResult_8]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>] to <>, <i1>
    sink %index_11 {handshake.name = "sink3"} : <i1>
    %29 = buffer %result_10 {handshake.bb = 3 : ui32, handshake.name = "buffer8", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <>
    %30 = buffer %29 {handshake.bb = 3 : ui32, handshake.name = "buffer9", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %31:3 = lazy_fork [3] %30 {handshake.bb = 3 : ui32, handshake.name = "lazy_fork0"} : <>
    %32 = buffer %31#2 {handshake.bb = 3 : ui32, handshake.name = "buffer11", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %33:2 = fork [2] %32 {handshake.bb = 3 : ui32, handshake.bufProps = #handshake<bufProps{"0": [1,inf], [0,inf], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork6"} : <>
    %34 = constant %33#1 {handshake.bb = 3 : ui32, handshake.name = "constant17", value = 1 : i2} : <>, <i2>
    %35 = extsi %34 {handshake.bb = 3 : ui32, handshake.name = "extsi4"} : <i2> to <i32>
    %36 = constant %33#0 {handshake.bb = 3 : ui32, handshake.name = "constant18", value = false} : <>, <i1>
    %37 = extui %36 {handshake.bb = 3 : ui32, handshake.name = "extui2"} : <i1> to <i4>
    %38 = source {handshake.bb = 3 : ui32, handshake.name = "source2"} : <>
    %39 = constant %38 {handshake.bb = 3 : ui32, handshake.name = "constant10", value = -1 : i32} : <>, <i32>
    %40 = buffer %28 {handshake.bb = 3 : ui32, handshake.name = "buffer6", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i32>
    %41 = buffer %40 {handshake.bb = 3 : ui32, handshake.name = "buffer7", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i32>
    %42 = addi %41, %39 {handshake.bb = 3 : ui32, handshake.name = "addi0"} : <i32>
    %addressResult_12, %dataResult_13 = store[%37] %42 {handshake.bb = 3 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[<"load2" (1)>]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 0>, handshake.name = "store0"} : <i4>, <i32>, <i4>, <i32>
    %43 = buffer %31#0 {handshake.bb = 3 : ui32, handshake.name = "buffer10", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <>
    %44 = br %43 {handshake.bb = 3 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [1,inf], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "br4"} : <>
    %45 = merge %falseResult_7 {handshake.bb = 4 : ui32, handshake.name = "merge2"} : <i32>
    %result_14, %index_15 = control_merge [%falseResult_9]  {handshake.bb = 4 : ui32, handshake.name = "control_merge3"} : [<>] to <>, <i1>
    sink %index_15 {handshake.name = "sink4"} : <i1>
    %46 = buffer %result_14 {handshake.bb = 4 : ui32, handshake.name = "buffer14", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <>
    %47 = buffer %46 {handshake.bb = 4 : ui32, handshake.name = "buffer15", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %48:3 = lazy_fork [3] %47 {handshake.bb = 4 : ui32, handshake.name = "lazy_fork1"} : <>
    %49 = buffer %48#2 {handshake.bb = 4 : ui32, handshake.name = "buffer17", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %50:2 = fork [2] %49 {handshake.bb = 4 : ui32, handshake.bufProps = #handshake<bufProps{"0": [1,inf], [0,inf], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork7"} : <>
    %51 = constant %50#1 {handshake.bb = 4 : ui32, handshake.name = "constant19", value = 1 : i2} : <>, <i2>
    %52 = extsi %51 {handshake.bb = 4 : ui32, handshake.name = "extsi6"} : <i2> to <i32>
    %53 = constant %50#0 {handshake.bb = 4 : ui32, handshake.name = "constant20", value = false} : <>, <i1>
    %54 = extui %53 {handshake.bb = 4 : ui32, handshake.name = "extui3"} : <i1> to <i4>
    %55 = source {handshake.bb = 4 : ui32, handshake.name = "source3"} : <>
    %56 = constant %55 {handshake.bb = 4 : ui32, handshake.name = "constant21", value = 1 : i2} : <>, <i2>
    %57 = extsi %56 {handshake.bb = 4 : ui32, handshake.name = "extsi8"} : <i2> to <i32>
    %58 = buffer %45 {handshake.bb = 4 : ui32, handshake.name = "buffer12", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i32>
    %59 = buffer %58 {handshake.bb = 4 : ui32, handshake.name = "buffer13", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i32>
    %60 = shrsi %59, %57 {handshake.bb = 4 : ui32, handshake.name = "shrsi0"} : <i32>
    %addressResult_16, %dataResult_17 = store[%54] %60 {handshake.bb = 4 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[<"load2" (1)>]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 1>, handshake.name = "store1"} : <i4>, <i32>, <i4>, <i32>
    %61 = buffer %48#0 {handshake.bb = 4 : ui32, handshake.name = "buffer16", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <>
    %62 = br %61 {handshake.bb = 4 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [1,inf], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "br5"} : <>
    %result_18, %index_19 = control_merge [%falseResult_1]  {handshake.bb = 5 : ui32, handshake.name = "control_merge4"} : [<>] to <>, <i1>
    sink %index_19 {handshake.name = "sink5"} : <i1>
    %63 = buffer %result_18 {handshake.bb = 5 : ui32, handshake.name = "buffer18", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <>
    %64 = buffer %63 {handshake.bb = 5 : ui32, handshake.name = "buffer19", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %65:3 = fork [3] %64 {handshake.bb = 5 : ui32, handshake.name = "fork8"} : <>
    %66 = buffer %65#0 {handshake.bb = 5 : ui32, handshake.name = "buffer20", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %67 = constant %66 {handshake.bb = 5 : ui32, handshake.bufProps = #handshake<bufProps{"0": [1,inf], [0,inf], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "constant22", value = false} : <>, <i1>
    %68 = extui %67 {handshake.bb = 5 : ui32, handshake.name = "extui4"} : <i1> to <i4>
    %addressResult_20, %dataResult_21 = load[%68] %2#0 {handshake.bb = 5 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<LSQ: 2>, handshake.name = "load2"} : <i4>, <i32>, <i4>, <i32>
    end {handshake.bb = 5 : ui32, handshake.name = "end0"} %dataResult_21, %memEnd, %0#0 : <i32>, <>, <>
  }
}

