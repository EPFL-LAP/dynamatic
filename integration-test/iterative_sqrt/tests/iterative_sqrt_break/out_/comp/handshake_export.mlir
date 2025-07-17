module {
  handshake.func @iterative_sqrt(%arg0: !handshake.channel<i32>, %arg1: !handshake.control<>, ...) -> (!handshake.channel<i32>, !handshake.control<>) attributes {argNames = ["n", "start"], resNames = ["out0", "end"]} {
    %0:4 = fork [4] %arg1 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork0"} : <>
    %1 = constant %0#3 {handshake.bb = 0 : ui32, handshake.name = "constant1", value = false} : <>, <i1>
    %2 = constant %0#2 {handshake.bb = 0 : ui32, handshake.name = "constant3", value = true} : <>, <i1>
    %3:2 = fork [2] %arg0 {handshake.bb = 0 : ui32, handshake.name = "fork1"} : <i32>
    %4 = extsi %1 {handshake.bb = 0 : ui32, handshake.name = "extsi3"} : <i1> to <i32>
    %5 = mux %15#0 [%3#1, %trueResult_6, %75, %86] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i2>, [<i32>, <i32>, <i32>, <i32>] to <i32>
    %6 = buffer %5 {handshake.bb = 1 : ui32, handshake.name = "buffer0", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i32>
    %7 = buffer %6 {handshake.bb = 1 : ui32, handshake.name = "buffer1", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i32>
    %8:2 = fork [2] %7 {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <i32>
    %9 = mux %15#1 [%4, %trueResult_8, %73, %88] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i2>, [<i32>, <i32>, <i32>, <i32>] to <i32>
    %10 = buffer %9 {handshake.bb = 1 : ui32, handshake.name = "buffer2", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i32>
    %11 = buffer %10 {handshake.bb = 1 : ui32, handshake.name = "buffer3", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i32>
    %12:2 = fork [2] %11 {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <i32>
    %13 = mux %15#2 [%2, %trueResult_10, %77, %90] {handshake.bb = 1 : ui32, handshake.name = "mux2"} : <i2>, [<i1>, <i1>, <i1>, <i1>] to <i1>
    %14 = mux %15#3 [%3#0, %trueResult_12, %79, %92] {handshake.bb = 1 : ui32, handshake.name = "mux3"} : <i2>, [<i32>, <i32>, <i32>, <i32>] to <i32>
    %result, %index = control_merge [%0#1, %trueResult_14, %81, %94]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>, <>, <>] to <>, <i2>
    %15:4 = fork [4] %index {handshake.bb = 1 : ui32, handshake.name = "fork4"} : <i2>
    %16 = cmpi sle, %12#1, %8#1 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i32>
    %17 = buffer %13 {handshake.bb = 1 : ui32, handshake.name = "buffer4", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %18 = buffer %17 {handshake.bb = 1 : ui32, handshake.name = "buffer5", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i1>
    %19 = andi %16, %18 {handshake.bb = 1 : ui32, handshake.name = "andi0"} : <i1>
    %20:4 = fork [4] %19 {handshake.bb = 1 : ui32, handshake.name = "fork5"} : <i1>
    %trueResult, %falseResult = cond_br %20#3, %8#0 {handshake.bb = 1 : ui32, handshake.name = "cond_br3"} : <i1>, <i32>
    %trueResult_0, %falseResult_1 = cond_br %20#2, %12#0 {handshake.bb = 1 : ui32, handshake.name = "cond_br4"} : <i1>, <i32>
    sink %falseResult_1 {handshake.name = "sink0"} : <i32>
    %21 = buffer %14 {handshake.bb = 1 : ui32, handshake.name = "buffer6", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i32>
    %22 = buffer %21 {handshake.bb = 1 : ui32, handshake.name = "buffer7", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i32>
    %trueResult_2, %falseResult_3 = cond_br %20#1, %22 {handshake.bb = 1 : ui32, handshake.name = "cond_br5"} : <i1>, <i32>
    sink %falseResult_3 {handshake.name = "sink1"} : <i32>
    %23 = buffer %result {handshake.bb = 1 : ui32, handshake.name = "buffer8", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <>
    %24 = buffer %23 {handshake.bb = 1 : ui32, handshake.name = "buffer9", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %trueResult_4, %falseResult_5 = cond_br %20#0, %24 {handshake.bb = 1 : ui32, handshake.name = "cond_br6"} : <i1>, <>
    sink %falseResult_5 {handshake.name = "sink2"} : <>
    %25 = buffer %trueResult {handshake.bb = 2 : ui32, handshake.name = "buffer10", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i32>
    %26 = buffer %25 {handshake.bb = 2 : ui32, handshake.name = "buffer11", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i32>
    %27:4 = fork [4] %26 {handshake.bb = 2 : ui32, handshake.name = "fork6"} : <i32>
    %28 = buffer %trueResult_0 {handshake.bb = 2 : ui32, handshake.name = "buffer12", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i32>
    %29 = buffer %28 {handshake.bb = 2 : ui32, handshake.name = "buffer13", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i32>
    %30:4 = fork [4] %29 {handshake.bb = 2 : ui32, handshake.name = "fork7"} : <i32>
    %31 = buffer %trueResult_2 {handshake.bb = 2 : ui32, handshake.name = "buffer14", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i32>
    %32 = buffer %31 {handshake.bb = 2 : ui32, handshake.name = "buffer15", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i32>
    %33:3 = fork [3] %32 {handshake.bb = 2 : ui32, handshake.name = "fork8"} : <i32>
    %34 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %35 = constant %34 {handshake.bb = 2 : ui32, handshake.name = "constant2", value = 1 : i2} : <>, <i2>
    %36 = extsi %35 {handshake.bb = 2 : ui32, handshake.name = "extsi1"} : <i2> to <i32>
    %37 = addi %30#3, %27#3 {handshake.bb = 2 : ui32, handshake.name = "addi0"} : <i32>
    %38 = shrsi %37, %36 {handshake.bb = 2 : ui32, handshake.name = "shrsi0"} : <i32>
    %39:3 = fork [3] %38 {handshake.bb = 2 : ui32, handshake.name = "fork9"} : <i32>
    %40 = muli %39#1, %39#2 {handshake.bb = 2 : ui32, handshake.name = "muli0"} : <i32>
    %41:3 = fork [3] %40 {handshake.bb = 2 : ui32, handshake.name = "fork10"} : <i32>
    %42 = cmpi ne, %41#2, %33#2 {handshake.bb = 2 : ui32, handshake.name = "cmpi1"} : <i32>
    %43 = cmpi sle, %30#2, %27#2 {handshake.bb = 2 : ui32, handshake.name = "cmpi2"} : <i32>
    %44 = cmpi sgt, %30#1, %27#1 {handshake.bb = 2 : ui32, handshake.name = "cmpi3"} : <i32>
    %45 = andi %43, %42 {handshake.bb = 2 : ui32, handshake.name = "andi1"} : <i1>
    %46 = cmpi eq, %41#1, %33#1 {handshake.bb = 2 : ui32, handshake.name = "cmpi4"} : <i32>
    %47:7 = fork [7] %46 {handshake.bb = 2 : ui32, handshake.name = "fork11"} : <i1>
    %48 = ori %45, %44 {handshake.bb = 2 : ui32, handshake.name = "ori0"} : <i1>
    %trueResult_6, %falseResult_7 = cond_br %47#6, %39#0 {handshake.bb = 2 : ui32, handshake.name = "cond_br7"} : <i1>, <i32>
    %trueResult_8, %falseResult_9 = cond_br %47#5, %30#0 {handshake.bb = 2 : ui32, handshake.name = "cond_br8"} : <i1>, <i32>
    %trueResult_10, %falseResult_11 = cond_br %47#4, %48 {handshake.bb = 2 : ui32, handshake.name = "cond_br9"} : <i1>, <i1>
    %trueResult_12, %falseResult_13 = cond_br %47#3, %33#0 {handshake.bb = 2 : ui32, handshake.name = "cond_br10"} : <i1>, <i32>
    %49 = buffer %trueResult_4 {handshake.bb = 2 : ui32, handshake.name = "buffer16", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <>
    %50 = buffer %49 {handshake.bb = 2 : ui32, handshake.name = "buffer17", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %trueResult_14, %falseResult_15 = cond_br %47#2, %50 {handshake.bb = 2 : ui32, handshake.name = "cond_br11"} : <i1>, <>
    %trueResult_16, %falseResult_17 = cond_br %47#1, %27#0 {handshake.bb = 2 : ui32, handshake.name = "cond_br12"} : <i1>, <i32>
    sink %trueResult_16 {handshake.name = "sink4"} : <i32>
    %trueResult_18, %falseResult_19 = cond_br %47#0, %41#0 {handshake.bb = 2 : ui32, handshake.name = "cond_br13"} : <i1>, <i32>
    sink %trueResult_18 {handshake.name = "sink5"} : <i32>
    %51 = buffer %falseResult_13 {handshake.bb = 3 : ui32, handshake.name = "buffer18", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i32>
    %52 = buffer %51 {handshake.bb = 3 : ui32, handshake.name = "buffer19", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i32>
    %53:2 = fork [2] %52 {handshake.bb = 3 : ui32, handshake.name = "fork12"} : <i32>
    %54 = buffer %falseResult_19 {handshake.bb = 3 : ui32, handshake.name = "buffer26", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i32>
    %55 = buffer %54 {handshake.bb = 3 : ui32, handshake.name = "buffer27", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i32>
    %56 = cmpi slt, %55, %53#1 {handshake.bb = 3 : ui32, handshake.name = "cmpi5"} : <i32>
    %57:6 = fork [6] %56 {handshake.bb = 3 : ui32, handshake.name = "fork13"} : <i1>
    %trueResult_20, %falseResult_21 = cond_br %57#5, %53#0 {handshake.bb = 3 : ui32, handshake.name = "cond_br14"} : <i1>, <i32>
    %58 = buffer %falseResult_17 {handshake.bb = 3 : ui32, handshake.name = "buffer20", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i32>
    %59 = buffer %58 {handshake.bb = 3 : ui32, handshake.name = "buffer21", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i32>
    %trueResult_22, %falseResult_23 = cond_br %57#4, %59 {handshake.bb = 3 : ui32, handshake.name = "cond_br15"} : <i1>, <i32>
    sink %falseResult_23 {handshake.name = "sink7"} : <i32>
    %60 = buffer %falseResult_7 {handshake.bb = 3 : ui32, handshake.name = "buffer24", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i32>
    %61 = buffer %60 {handshake.bb = 3 : ui32, handshake.name = "buffer25", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i32>
    %trueResult_24, %falseResult_25 = cond_br %57#3, %61 {handshake.bb = 3 : ui32, handshake.name = "cond_br16"} : <i1>, <i32>
    %62 = buffer %falseResult_11 {handshake.bb = 3 : ui32, handshake.name = "buffer28", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %63 = buffer %62 {handshake.bb = 3 : ui32, handshake.name = "buffer29", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i1>
    %trueResult_26, %falseResult_27 = cond_br %57#2, %63 {handshake.bb = 3 : ui32, handshake.name = "cond_br17"} : <i1>, <i1>
    %64 = buffer %falseResult_15 {handshake.bb = 3 : ui32, handshake.name = "buffer30", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <>
    %65 = buffer %64 {handshake.bb = 3 : ui32, handshake.name = "buffer31", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %trueResult_28, %falseResult_29 = cond_br %57#1, %65 {handshake.bb = 3 : ui32, handshake.name = "cond_br18"} : <i1>, <>
    %66 = buffer %falseResult_9 {handshake.bb = 3 : ui32, handshake.name = "buffer22", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i32>
    %67 = buffer %66 {handshake.bb = 3 : ui32, handshake.name = "buffer23", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i32>
    %trueResult_30, %falseResult_31 = cond_br %57#0, %67 {handshake.bb = 3 : ui32, handshake.name = "cond_br19"} : <i1>, <i32>
    sink %trueResult_30 {handshake.name = "sink8"} : <i32>
    %68 = source {handshake.bb = 4 : ui32, handshake.name = "source1"} : <>
    %69 = constant %68 {handshake.bb = 4 : ui32, handshake.name = "constant7", value = 1 : i2} : <>, <i2>
    %70 = extsi %69 {handshake.bb = 4 : ui32, handshake.name = "extsi2"} : <i2> to <i32>
    %71 = buffer %trueResult_24 {handshake.bb = 4 : ui32, handshake.name = "buffer36", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i32>
    %72 = buffer %71 {handshake.bb = 4 : ui32, handshake.name = "buffer37", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i32>
    %73 = addi %72, %70 {handshake.bb = 4 : ui32, handshake.name = "addi1"} : <i32>
    %74 = buffer %trueResult_22 {handshake.bb = 4 : ui32, handshake.name = "buffer34", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i32>
    %75 = buffer %74 {handshake.bb = 4 : ui32, handshake.name = "buffer35", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i32>
    %76 = buffer %trueResult_26 {handshake.bb = 4 : ui32, handshake.name = "buffer38", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %77 = buffer %76 {handshake.bb = 4 : ui32, handshake.name = "buffer39", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i1>
    %78 = buffer %trueResult_20 {handshake.bb = 4 : ui32, handshake.name = "buffer32", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i32>
    %79 = buffer %78 {handshake.bb = 4 : ui32, handshake.name = "buffer33", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i32>
    %80 = buffer %trueResult_28 {handshake.bb = 4 : ui32, handshake.name = "buffer40", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <>
    %81 = buffer %80 {handshake.bb = 4 : ui32, handshake.name = "buffer41", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %82 = source {handshake.bb = 5 : ui32, handshake.name = "source2"} : <>
    %83 = constant %82 {handshake.bb = 5 : ui32, handshake.name = "constant6", value = -1 : i32} : <>, <i32>
    %84 = buffer %falseResult_25 {handshake.bb = 5 : ui32, handshake.name = "buffer46", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i32>
    %85 = buffer %84 {handshake.bb = 5 : ui32, handshake.name = "buffer47", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i32>
    %86 = addi %85, %83 {handshake.bb = 5 : ui32, handshake.name = "addi2"} : <i32>
    %87 = buffer %falseResult_31 {handshake.bb = 5 : ui32, handshake.name = "buffer44", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i32>
    %88 = buffer %87 {handshake.bb = 5 : ui32, handshake.name = "buffer45", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i32>
    %89 = buffer %falseResult_27 {handshake.bb = 5 : ui32, handshake.name = "buffer48", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %90 = buffer %89 {handshake.bb = 5 : ui32, handshake.name = "buffer49", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i1>
    %91 = buffer %falseResult_21 {handshake.bb = 5 : ui32, handshake.name = "buffer42", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i32>
    %92 = buffer %91 {handshake.bb = 5 : ui32, handshake.name = "buffer43", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i32>
    %93 = buffer %falseResult_29 {handshake.bb = 5 : ui32, handshake.name = "buffer50", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <>
    %94 = buffer %93 {handshake.bb = 5 : ui32, handshake.name = "buffer51", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %95 = buffer %falseResult {handshake.bb = 6 : ui32, handshake.name = "buffer52", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i32>
    %96 = buffer %95 {handshake.bb = 6 : ui32, handshake.name = "buffer53", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i32>
    end {handshake.bb = 6 : ui32, handshake.name = "end0"} %96, %0#0 : <i32>, <>
  }
}

