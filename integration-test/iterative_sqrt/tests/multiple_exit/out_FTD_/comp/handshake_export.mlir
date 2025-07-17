module {
  handshake.func @multiple_exit(%arg0: memref<10xi32>, %arg1: !handshake.control<>, %arg2: !handshake.control<>, ...) -> (!handshake.channel<i32>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["arr", "arr_start", "start"], cfg.edges = "[0,1][2,3,1,cmpi1][4,1][1,2,5,andi0][3,4,1,cmpi3]", resNames = ["out0", "arr_end", "end"]} {
    %0:3 = fork [3] %arg2 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork0"} : <>
    %1 = buffer %falseResult_7 {handshake.name = "buffer33", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <>
    %2 = buffer %1 {handshake.name = "buffer34", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %3:4 = lsq[%arg0 : memref<10xi32>] (%arg1, %61#1, %addressResult, %72#1, %addressResult_14, %93#1, %addressResult_20, %addressResult_22, %dataResult_23, %2)  {groupSizes = [1 : i32, 1 : i32, 2 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "5": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "6": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "7": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "8": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "9": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "lsq1"} : (!handshake.control<>, !handshake.control<>, !handshake.channel<i4>, !handshake.control<>, !handshake.channel<i4>, !handshake.control<>, !handshake.channel<i4>, !handshake.channel<i4>, !handshake.channel<i32>, !handshake.control<>) -> (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>)
    %4 = source {handshake.bb = 0 : ui32, handshake.name = "source0"} : <>
    %5 = constant %4 {handshake.bb = 0 : ui32, handshake.name = "constant2", value = false} : <>, <i1>
    %6 = extsi %5 {handshake.bb = 1 : ui32, handshake.name = "extsi0"} : <i1> to <i32>
    %7 = source {handshake.bb = 0 : ui32, handshake.name = "source1"} : <>
    %8 = constant %7 {handshake.bb = 0 : ui32, handshake.name = "constant6", value = true} : <>, <i1>
    %9:2 = fork [2] %8 {handshake.bb = 0 : ui32, handshake.name = "fork1"} : <i1>
    %10 = buffer %27 {handshake.bb = 1 : ui32, handshake.name = "buffer10", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i32>
    %11 = buffer %10 {handshake.bb = 1 : ui32, handshake.name = "buffer11", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i32>
    %trueResult, %falseResult = cond_br %56#4, %11 {handshake.bb = 1 : ui32, handshake.name = "cond_br25"} : <i1>, <i32>
    sink %falseResult {handshake.name = "sink0"} : <i32>
    %12 = buffer %24 {handshake.bb = 1 : ui32, handshake.name = "buffer8", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %13 = buffer %12 {handshake.bb = 1 : ui32, handshake.name = "buffer9", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i1>
    %trueResult_0, %falseResult_1 = cond_br %56#5, %13 {handshake.bb = 1 : ui32, handshake.name = "cond_br26"} : <i1>, <i1>
    sink %falseResult_1 {handshake.name = "sink1"} : <i1>
    %14 = buffer %23 {handshake.bb = 1 : ui32, handshake.name = "buffer6", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i2>
    %15 = buffer %14 {handshake.bb = 1 : ui32, handshake.name = "buffer7", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i2>
    %trueResult_2, %falseResult_3 = cond_br %56#6, %15 {handshake.bb = 1 : ui32, handshake.name = "cond_br27"} : <i1>, <i2>
    sink %falseResult_3 {handshake.name = "sink2"} : <i2>
    %16 = buffer %21 {handshake.bb = 1 : ui32, handshake.name = "buffer2", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %17 = buffer %16 {handshake.bb = 1 : ui32, handshake.name = "buffer3", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i1>
    %trueResult_4, %falseResult_5 = cond_br %56#7, %17 {handshake.bb = 1 : ui32, handshake.name = "cond_br28"} : <i1>, <i1>
    sink %falseResult_5 {handshake.name = "sink3"} : <i1>
    %18 = merge %0#2, %falseResult_9, %falseResult_17, %102 {handshake.bb = 1 : ui32, handshake.name = "merge1"} : <>
    %19 = buffer %22 {handshake.bb = 1 : ui32, handshake.name = "buffer4", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %20 = buffer %19 {handshake.bb = 1 : ui32, handshake.name = "buffer5", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i1>
    %21 = mux %67#1 [%63, %20] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i1>, <i1>] to <i1>
    %22 = mux %82#2 [%82#3, %95] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i1>, [<i1>, <i1>] to <i1>
    %23 = mux %67#2 [%40#1, %84] {handshake.bb = 1 : ui32, handshake.name = "mux2"} : <i1>, [<i2>, <i2>] to <i2>
    %24 = mux %67#3 [%44#1, %83] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux3"} : <i1>, [<i1>, <i1>] to <i1>
    %25 = buffer %28 {handshake.bb = 1 : ui32, handshake.name = "buffer12", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i32>
    %26 = buffer %25 {handshake.bb = 1 : ui32, handshake.name = "buffer13", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i32>
    %27 = mux %67#4 [%48#2, %26] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux4"} : <i1>, [<i32>, <i32>] to <i32>
    %28 = mux %82#4 [%48#3, %101] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux5"} : <i1>, [<i32>, <i32>] to <i32>
    %29 = constant %0#1 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "constant1", value = false} : <>, <i1>
    %30:2 = fork [2] %29 {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <i1>
    %31 = extsi %30#1 {handshake.bb = 1 : ui32, handshake.name = "extsi7"} : <i1> to <i2>
    %32 = mux %36#3 [%9#1, %trueResult_4] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux6"} : <i1>, [<i1>, <i1>] to <i1>
    %33 = merge %30#0, %56#8 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "merge0"} : <i1>
    %34 = buffer %33 {handshake.bb = 1 : ui32, handshake.name = "buffer16", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %35 = buffer %34 {handshake.bb = 1 : ui32, handshake.name = "buffer17", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i1>
    %36:4 = fork [4] %35 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork3"} : <i1>
    %37 = mux %36#2 [%31, %trueResult_2] {handshake.bb = 1 : ui32, handshake.name = "mux7"} : <i1>, [<i2>, <i2>] to <i2>
    %38 = buffer %37 {handshake.bb = 1 : ui32, handshake.name = "buffer18", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i2>
    %39 = buffer %38 {handshake.bb = 1 : ui32, handshake.name = "buffer19", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i2>
    %40:2 = fork [2] %39 {handshake.bb = 1 : ui32, handshake.name = "fork4"} : <i2>
    %41 = mux %36#1 [%9#0, %trueResult_0] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux8"} : <i1>, [<i1>, <i1>] to <i1>
    %42 = buffer %41 {handshake.bb = 1 : ui32, handshake.name = "buffer20", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %43 = buffer %42 {handshake.bb = 1 : ui32, handshake.name = "buffer21", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i1>
    %44:2 = fork [2] %43 {handshake.bb = 1 : ui32, handshake.name = "fork5"} : <i1>
    %45 = mux %36#0 [%6, %trueResult] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux9"} : <i1>, [<i32>, <i32>] to <i32>
    %46 = buffer %45 {handshake.bb = 1 : ui32, handshake.name = "buffer22", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i32>
    %47 = buffer %46 {handshake.bb = 1 : ui32, handshake.name = "buffer23", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i32>
    %48:4 = fork [4] %47 {handshake.bb = 1 : ui32, handshake.name = "fork6"} : <i32>
    %49 = source {handshake.bb = 1 : ui32, handshake.name = "source2"} : <>
    %50 = constant %49 {handshake.bb = 1 : ui32, handshake.name = "constant3", value = 10 : i5} : <>, <i5>
    %51 = extsi %50 {handshake.bb = 1 : ui32, handshake.name = "extsi2"} : <i5> to <i32>
    %52 = cmpi slt, %48#1, %51 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i32>
    %53 = buffer %32 {handshake.bb = 1 : ui32, handshake.name = "buffer14", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %54 = buffer %53 {handshake.bb = 1 : ui32, handshake.name = "buffer15", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i1>
    %55 = andi %52, %54 {handshake.bb = 1 : ui32, handshake.name = "andi0"} : <i1>
    %56:9 = fork [9] %55 {handshake.bb = 1 : ui32, handshake.name = "fork7"} : <i1>
    %57 = buffer %18 {handshake.bb = 1 : ui32, handshake.name = "buffer0", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <>
    %58 = buffer %57 {handshake.bb = 1 : ui32, handshake.name = "buffer1", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %trueResult_6, %falseResult_7 = cond_br %56#3, %58 {handshake.bb = 1 : ui32, handshake.name = "cond_br9"} : <i1>, <>
    %59 = buffer %trueResult_6 {handshake.bb = 2 : ui32, handshake.name = "buffer24", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <>
    %60 = buffer %59 {handshake.bb = 2 : ui32, handshake.name = "buffer25", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %61:2 = lazy_fork [2] %60 {handshake.bb = 2 : ui32, handshake.name = "lazy_fork0"} : <>
    %62 = source {handshake.bb = 2 : ui32, handshake.name = "source6"} : <>
    %63 = constant %62 {handshake.bb = 2 : ui32, handshake.name = "constant8", value = false} : <>, <i1>
    %64 = source {handshake.bb = 2 : ui32, handshake.name = "source7"} : <>
    %65 = constant %64 {handshake.bb = 2 : ui32, handshake.name = "constant9", value = -1 : i32} : <>, <i32>
    %addressResult, %dataResult = load[%90] %3#0 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[<"store1" (0)>]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 0>, handshake.name = "load3"} : <i4>, <i32>, <i4>, <i32>
    %66 = cmpi ne, %dataResult, %65 {handshake.bb = 2 : ui32, handshake.name = "cmpi1"} : <i32>
    %67:5 = fork [5] %66 {handshake.bb = 2 : ui32, handshake.name = "fork8"} : <i1>
    %68 = buffer %61#0 {handshake.bb = 2 : ui32, handshake.name = "buffer26", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <>
    %trueResult_8, %falseResult_9 = cond_br %67#0, %68 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,inf], [1,inf], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "cond_br20"} : <i1>, <>
    %trueResult_10, %falseResult_11 = cond_br %56#2, %40#0 {handshake.bb = 3 : ui32, handshake.name = "cond_br29"} : <i1>, <i2>
    %69 = extsi %falseResult_11 {handshake.bb = 3 : ui32, handshake.name = "extsi8"} : <i2> to <i3>
    %trueResult_12, %falseResult_13 = cond_br %56#1, %44#0 {handshake.bb = 3 : ui32, handshake.name = "cond_br30"} : <i1>, <i1>
    %70 = buffer %trueResult_8 {handshake.bb = 3 : ui32, handshake.name = "buffer27", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <>
    %71 = buffer %70 {handshake.bb = 3 : ui32, handshake.name = "buffer28", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %72:2 = lazy_fork [2] %71 {handshake.bb = 3 : ui32, handshake.name = "lazy_fork1"} : <>
    %73 = source {handshake.bb = 3 : ui32, handshake.name = "source8"} : <>
    %74 = constant %73 {handshake.bb = 3 : ui32, handshake.name = "constant4", value = 1 : i2} : <>, <i2>
    %75 = source {handshake.bb = 3 : ui32, handshake.name = "source9"} : <>
    %76 = constant %75 {handshake.bb = 3 : ui32, handshake.name = "constant5", value = false} : <>, <i1>
    %77 = extsi %76 {handshake.bb = 3 : ui32, handshake.name = "extsi4"} : <i1> to <i32>
    %78:2 = fork [2] %77 {handshake.bb = 3 : ui32, handshake.name = "fork9"} : <i32>
    %addressResult_14, %dataResult_15 = load[%89] %3#1 {handshake.bb = 3 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[<"store1" (0)>]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 1>, handshake.name = "load4"} : <i4>, <i32>, <i4>, <i32>
    %79:2 = fork [2] %dataResult_15 {handshake.bb = 3 : ui32, handshake.name = "fork10"} : <i32>
    %80 = cmpi eq, %79#1, %78#1 {handshake.bb = 3 : ui32, handshake.name = "cmpi2"} : <i32>
    %81 = cmpi ne, %79#0, %78#0 {handshake.bb = 3 : ui32, handshake.name = "cmpi3"} : <i32>
    %82:5 = fork [5] %81 {handshake.bb = 3 : ui32, handshake.name = "fork11"} : <i1>
    %83 = andi %82#1, %trueResult_12 {handshake.bb = 3 : ui32, handshake.name = "andi1"} : <i1>
    %84 = select %80[%74, %trueResult_10] {handshake.bb = 3 : ui32, handshake.name = "select0"} : <i1>, <i2>
    %85 = buffer %72#0 {handshake.bb = 3 : ui32, handshake.name = "buffer29", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <>
    %trueResult_16, %falseResult_17 = cond_br %82#0, %85 {handshake.bb = 3 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,inf], [1,inf], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "cond_br23"} : <i1>, <>
    %trueResult_18, %falseResult_19 = cond_br %56#0, %48#0 {handshake.bb = 4 : ui32, handshake.name = "cond_br31"} : <i1>, <i32>
    sink %falseResult_19 {handshake.name = "sink7"} : <i32>
    %86:5 = fork [5] %trueResult_18 {handshake.bb = 4 : ui32, handshake.name = "fork12"} : <i32>
    %87 = trunci %86#4 {handshake.bb = 4 : ui32, handshake.name = "trunci0"} : <i32> to <i4>
    %88 = trunci %86#3 {handshake.bb = 4 : ui32, handshake.name = "trunci1"} : <i32> to <i4>
    %89 = trunci %86#2 {handshake.bb = 4 : ui32, handshake.name = "trunci2"} : <i32> to <i4>
    %90 = trunci %86#1 {handshake.bb = 4 : ui32, handshake.name = "trunci3"} : <i32> to <i4>
    %91 = buffer %trueResult_16 {handshake.bb = 4 : ui32, handshake.name = "buffer30", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <>
    %92 = buffer %91 {handshake.bb = 4 : ui32, handshake.name = "buffer31", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %93:2 = lazy_fork [2] %92 {handshake.bb = 4 : ui32, handshake.name = "lazy_fork2"} : <>
    %94 = source {handshake.bb = 4 : ui32, handshake.name = "source10"} : <>
    %95 = constant %94 {handshake.bb = 4 : ui32, handshake.name = "constant12", value = true} : <>, <i1>
    %96 = source {handshake.bb = 4 : ui32, handshake.name = "source11"} : <>
    %97 = constant %96 {handshake.bb = 4 : ui32, handshake.name = "constant15", value = 1 : i2} : <>, <i2>
    %98 = extsi %97 {handshake.bb = 4 : ui32, handshake.name = "extsi5"} : <i2> to <i32>
    %99:2 = fork [2] %98 {handshake.bb = 4 : ui32, handshake.name = "fork13"} : <i32>
    %addressResult_20, %dataResult_21 = load[%88] %3#2 {handshake.bb = 4 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[<"store1" (0)>]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 2>, handshake.name = "load5"} : <i4>, <i32>, <i4>, <i32>
    %100 = addi %dataResult_21, %99#1 {handshake.bb = 4 : ui32, handshake.name = "addi0"} : <i32>
    %addressResult_22, %dataResult_23 = store[%87] %100 {handshake.bb = 4 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[<"load3" (0)>, <"load4" (0)>, <"load5" (0)>, <"store1" (0)>]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 2>, handshake.name = "store1"} : <i4>, <i32>, <i4>, <i32>
    %101 = addi %86#0, %99#0 {handshake.bb = 4 : ui32, handshake.name = "addi1"} : <i32>
    %102 = buffer %93#0 {handshake.bb = 4 : ui32, handshake.name = "buffer32", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <>
    %103 = source {handshake.bb = 5 : ui32, handshake.name = "source12"} : <>
    %104 = constant %103 {handshake.bb = 5 : ui32, handshake.name = "constant16", value = 2 : i3} : <>, <i3>
    %105 = select %falseResult_13[%104, %69] {handshake.bb = 5 : ui32, handshake.name = "select1"} : <i1>, <i3>
    %106 = extsi %105 {handshake.bb = 5 : ui32, handshake.name = "extsi9"} : <i3> to <i32>
    end {handshake.bb = 5 : ui32, handshake.name = "end0"} %106, %3#3, %0#0 : <i32>, <>, <>
  }
}

