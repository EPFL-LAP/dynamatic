module {
  handshake.func @iterative_sqrt(%arg0: memref<10xi32>, %arg1: !handshake.control<>, %arg2: !handshake.control<>, ...) -> (!handshake.channel<i32>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["A", "A_start", "start"], resNames = ["out0", "A_end", "end"]} {
    %0:2 = fork [2] %arg2 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork0"} : <>
    %1 = buffer %60#1 {handshake.name = "buffer21", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %outputs:3, %memEnd = mem_controller[%arg0 : memref<10xi32>] %arg1 (%addressResult, %addressResult_2, %48, %33, %2#1, %2#2, %2#3) %1 {connectedBlocks = [1 : i32, 2 : i32, 4 : i32, 3 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "5": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "6": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "7": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "8": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "9": [1,inf], [0,inf], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller1"} :    (!handshake.channel<i4>, !handshake.channel<i4>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i4>, !handshake.channel<i4>, !handshake.channel<i32>) -> (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>)
    %2:4 = lsq[MC] (%29#1, %addressResult_8, %dataResult_9, %44#1, %addressResult_10, %dataResult_11, %60#2, %addressResult_12, %outputs#2)  {groupSizes = [1 : i32, 1 : i32, 1 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "5": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "6": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "7": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "8": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "lsq1"} : (!handshake.control<>, !handshake.channel<i4>, !handshake.channel<i32>, !handshake.control<>, !handshake.channel<i4>, !handshake.channel<i32>, !handshake.control<>, !handshake.channel<i4>, !handshake.channel<i32>) -> (!handshake.channel<i32>, !handshake.channel<i4>, !handshake.channel<i4>, !handshake.channel<i32>)
    %3 = merge %0#1, %41, %57 {handshake.bb = 1 : ui32, handshake.name = "merge3"} : <>
    %4 = buffer %3 {handshake.bb = 1 : ui32, handshake.name = "buffer0", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <>
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
    %15 = buffer %trueResult_0 {handshake.bb = 2 : ui32, handshake.name = "buffer4", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <>
    %16 = buffer %15 {handshake.bb = 2 : ui32, handshake.name = "buffer5", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %17:2 = fork [2] %16 {handshake.bb = 2 : ui32, handshake.name = "fork4"} : <>
    %18 = constant %17#1 {handshake.bb = 2 : ui32, handshake.name = "constant15", value = 1 : i2} : <>, <i2>
    %19 = extui %18 {handshake.bb = 2 : ui32, handshake.name = "extui1"} : <i2> to <i4>
    %20 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %21 = constant %20 {handshake.bb = 2 : ui32, handshake.name = "constant16", value = 10 : i5} : <>, <i5>
    %22 = extsi %21 {handshake.bb = 2 : ui32, handshake.name = "extsi3"} : <i5> to <i32>
    %addressResult_2, %dataResult_3 = load[%19] %outputs#1 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i4>, <i32>, <i4>, <i32>
    %23 = cmpi slt, %dataResult_3, %22 {handshake.bb = 2 : ui32, handshake.name = "cmpi1"} : <i32>
    %24:2 = fork [2] %23 {handshake.bb = 2 : ui32, handshake.name = "fork5"} : <i1>
    %25 = buffer %trueResult {handshake.bb = 2 : ui32, handshake.name = "buffer2", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i32>
    %26 = buffer %25 {handshake.bb = 2 : ui32, handshake.name = "buffer3", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i32>
    %trueResult_4, %falseResult_5 = cond_br %24#1, %26 {handshake.bb = 2 : ui32, handshake.name = "cond_br4"} : <i1>, <i32>
    %trueResult_6, %falseResult_7 = cond_br %24#0, %17#0 {handshake.bb = 2 : ui32, handshake.name = "cond_br5"} : <i1>, <>
    %27 = buffer %trueResult_6 {handshake.bb = 3 : ui32, handshake.name = "buffer8", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <>
    %28 = buffer %27 {handshake.bb = 3 : ui32, handshake.name = "buffer9", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %29:3 = lazy_fork [3] %28 {handshake.bb = 3 : ui32, handshake.name = "lazy_fork0"} : <>
    %30 = buffer %29#2 {handshake.bb = 3 : ui32, handshake.name = "buffer11", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %31:2 = fork [2] %30 {handshake.bb = 3 : ui32, handshake.bufProps = #handshake<bufProps{"0": [1,inf], [0,inf], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork6"} : <>
    %32 = constant %31#1 {handshake.bb = 3 : ui32, handshake.name = "constant17", value = 1 : i2} : <>, <i2>
    %33 = extsi %32 {handshake.bb = 3 : ui32, handshake.name = "extsi4"} : <i2> to <i32>
    %34 = constant %31#0 {handshake.bb = 3 : ui32, handshake.name = "constant18", value = false} : <>, <i1>
    %35 = extui %34 {handshake.bb = 3 : ui32, handshake.name = "extui2"} : <i1> to <i4>
    %36 = source {handshake.bb = 3 : ui32, handshake.name = "source2"} : <>
    %37 = constant %36 {handshake.bb = 3 : ui32, handshake.name = "constant10", value = -1 : i32} : <>, <i32>
    %38 = buffer %trueResult_4 {handshake.bb = 3 : ui32, handshake.name = "buffer6", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i32>
    %39 = buffer %38 {handshake.bb = 3 : ui32, handshake.name = "buffer7", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i32>
    %40 = addi %39, %37 {handshake.bb = 3 : ui32, handshake.name = "addi0"} : <i32>
    %addressResult_8, %dataResult_9 = store[%35] %40 {handshake.bb = 3 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[<"load2" (1)>]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 0>, handshake.name = "store0"} : <i4>, <i32>, <i4>, <i32>
    %41 = buffer %29#0 {handshake.bb = 3 : ui32, handshake.name = "buffer10", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <>
    %42 = buffer %falseResult_7 {handshake.bb = 4 : ui32, handshake.name = "buffer14", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <>
    %43 = buffer %42 {handshake.bb = 4 : ui32, handshake.name = "buffer15", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %44:3 = lazy_fork [3] %43 {handshake.bb = 4 : ui32, handshake.name = "lazy_fork1"} : <>
    %45 = buffer %44#2 {handshake.bb = 4 : ui32, handshake.name = "buffer17", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %46:2 = fork [2] %45 {handshake.bb = 4 : ui32, handshake.bufProps = #handshake<bufProps{"0": [1,inf], [0,inf], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork7"} : <>
    %47 = constant %46#1 {handshake.bb = 4 : ui32, handshake.name = "constant19", value = 1 : i2} : <>, <i2>
    %48 = extsi %47 {handshake.bb = 4 : ui32, handshake.name = "extsi6"} : <i2> to <i32>
    %49 = constant %46#0 {handshake.bb = 4 : ui32, handshake.name = "constant20", value = false} : <>, <i1>
    %50 = extui %49 {handshake.bb = 4 : ui32, handshake.name = "extui3"} : <i1> to <i4>
    %51 = source {handshake.bb = 4 : ui32, handshake.name = "source3"} : <>
    %52 = constant %51 {handshake.bb = 4 : ui32, handshake.name = "constant21", value = 1 : i2} : <>, <i2>
    %53 = extsi %52 {handshake.bb = 4 : ui32, handshake.name = "extsi8"} : <i2> to <i32>
    %54 = buffer %falseResult_5 {handshake.bb = 4 : ui32, handshake.name = "buffer12", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i32>
    %55 = buffer %54 {handshake.bb = 4 : ui32, handshake.name = "buffer13", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i32>
    %56 = shrsi %55, %53 {handshake.bb = 4 : ui32, handshake.name = "shrsi0"} : <i32>
    %addressResult_10, %dataResult_11 = store[%50] %56 {handshake.bb = 4 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[<"load2" (1)>]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 1>, handshake.name = "store1"} : <i4>, <i32>, <i4>, <i32>
    %57 = buffer %44#0 {handshake.bb = 4 : ui32, handshake.name = "buffer16", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <>
    %58 = buffer %falseResult_1 {handshake.bb = 5 : ui32, handshake.name = "buffer18", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <>
    %59 = buffer %58 {handshake.bb = 5 : ui32, handshake.name = "buffer19", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %60:3 = fork [3] %59 {handshake.bb = 5 : ui32, handshake.name = "fork8"} : <>
    %61 = buffer %60#0 {handshake.bb = 5 : ui32, handshake.name = "buffer20", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %62 = constant %61 {handshake.bb = 5 : ui32, handshake.bufProps = #handshake<bufProps{"0": [1,inf], [0,inf], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "constant22", value = false} : <>, <i1>
    %63 = extui %62 {handshake.bb = 5 : ui32, handshake.name = "extui4"} : <i1> to <i4>
    %addressResult_12, %dataResult_13 = load[%63] %2#0 {handshake.bb = 5 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<LSQ: 2>, handshake.name = "load2"} : <i4>, <i32>, <i4>, <i32>
    end {handshake.bb = 5 : ui32, handshake.name = "end0"} %dataResult_13, %memEnd, %0#0 : <i32>, <>, <>
  }
}

