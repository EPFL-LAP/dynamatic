module {
  handshake.func @iterative_sqrt(%arg0: !handshake.channel<i32>, %arg1: !handshake.control<>, ...) -> (!handshake.channel<i32>, !handshake.control<>) attributes {argNames = ["n", "start"], resNames = ["out0", "end"]} {
    %0:4 = fork [4] %arg1 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork0"} : <>
    %1 = constant %0#3 {handshake.bb = 0 : ui32, handshake.name = "constant1", value = false} : <>, <i1>
    %2 = constant %0#2 {handshake.bb = 0 : ui32, handshake.name = "constant3", value = true} : <>, <i1>
    %3 = br %arg0 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "br2"} : <i32>
    %4:2 = fork [2] %3 {handshake.bb = 0 : ui32, handshake.name = "fork1"} : <i32>
    %5 = br %1 {handshake.bb = 0 : ui32, handshake.name = "br3"} : <i1>
    %6 = extsi %5 {handshake.bb = 0 : ui32, handshake.name = "extsi3"} : <i1> to <i32>
    %7 = br %2 {handshake.bb = 0 : ui32, handshake.name = "br4"} : <i1>
    %8 = br %0#1 {handshake.bb = 0 : ui32, handshake.name = "br5"} : <>
    %9 = mux %19#0 [%4#1, %trueResult_8, %93, %113] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i2>, [<i32>, <i32>, <i32>, <i32>] to <i32>
    %10 = buffer %9 {handshake.bb = 1 : ui32, handshake.name = "buffer0", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i32>
    %11 = buffer %10 {handshake.bb = 1 : ui32, handshake.name = "buffer1", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i32>
    %12:2 = fork [2] %11 {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <i32>
    %13 = mux %19#1 [%6, %trueResult_10, %94, %116] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i2>, [<i32>, <i32>, <i32>, <i32>] to <i32>
    %14 = buffer %13 {handshake.bb = 1 : ui32, handshake.name = "buffer2", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i32>
    %15 = buffer %14 {handshake.bb = 1 : ui32, handshake.name = "buffer3", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i32>
    %16:2 = fork [2] %15 {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <i32>
    %17 = mux %19#2 [%7, %trueResult_12, %97, %119] {handshake.bb = 1 : ui32, handshake.name = "mux2"} : <i2>, [<i1>, <i1>, <i1>, <i1>] to <i1>
    %18 = mux %19#3 [%4#0, %trueResult_14, %100, %122] {handshake.bb = 1 : ui32, handshake.name = "mux3"} : <i2>, [<i32>, <i32>, <i32>, <i32>] to <i32>
    %result, %index = control_merge [%8, %trueResult_16, %103, %125]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>, <>, <>] to <>, <i2>
    %19:4 = fork [4] %index {handshake.bb = 1 : ui32, handshake.name = "fork4"} : <i2>
    %20 = cmpi sle, %16#1, %12#1 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i32>
    %21 = buffer %17 {handshake.bb = 1 : ui32, handshake.name = "buffer4", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %22 = buffer %21 {handshake.bb = 1 : ui32, handshake.name = "buffer5", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i1>
    %23 = andi %20, %22 {handshake.bb = 1 : ui32, handshake.name = "andi0"} : <i1>
    %24:4 = fork [4] %23 {handshake.bb = 1 : ui32, handshake.name = "fork5"} : <i1>
    %trueResult, %falseResult = cond_br %24#3, %12#0 {handshake.bb = 1 : ui32, handshake.name = "cond_br3"} : <i1>, <i32>
    %trueResult_0, %falseResult_1 = cond_br %24#2, %16#0 {handshake.bb = 1 : ui32, handshake.name = "cond_br4"} : <i1>, <i32>
    sink %falseResult_1 {handshake.name = "sink0"} : <i32>
    %25 = buffer %18 {handshake.bb = 1 : ui32, handshake.name = "buffer6", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i32>
    %26 = buffer %25 {handshake.bb = 1 : ui32, handshake.name = "buffer7", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i32>
    %trueResult_2, %falseResult_3 = cond_br %24#1, %26 {handshake.bb = 1 : ui32, handshake.name = "cond_br5"} : <i1>, <i32>
    sink %falseResult_3 {handshake.name = "sink1"} : <i32>
    %27 = buffer %result {handshake.bb = 1 : ui32, handshake.name = "buffer8", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <>
    %28 = buffer %27 {handshake.bb = 1 : ui32, handshake.name = "buffer9", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %trueResult_4, %falseResult_5 = cond_br %24#0, %28 {handshake.bb = 1 : ui32, handshake.name = "cond_br6"} : <i1>, <>
    sink %falseResult_5 {handshake.name = "sink2"} : <>
    %29 = merge %trueResult {handshake.bb = 2 : ui32, handshake.name = "merge0"} : <i32>
    %30 = buffer %29 {handshake.bb = 2 : ui32, handshake.name = "buffer10", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i32>
    %31 = buffer %30 {handshake.bb = 2 : ui32, handshake.name = "buffer11", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i32>
    %32:4 = fork [4] %31 {handshake.bb = 2 : ui32, handshake.name = "fork6"} : <i32>
    %33 = merge %trueResult_0 {handshake.bb = 2 : ui32, handshake.name = "merge1"} : <i32>
    %34 = buffer %33 {handshake.bb = 2 : ui32, handshake.name = "buffer12", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i32>
    %35 = buffer %34 {handshake.bb = 2 : ui32, handshake.name = "buffer13", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i32>
    %36:4 = fork [4] %35 {handshake.bb = 2 : ui32, handshake.name = "fork7"} : <i32>
    %37 = merge %trueResult_2 {handshake.bb = 2 : ui32, handshake.name = "merge2"} : <i32>
    %38 = buffer %37 {handshake.bb = 2 : ui32, handshake.name = "buffer14", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i32>
    %39 = buffer %38 {handshake.bb = 2 : ui32, handshake.name = "buffer15", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i32>
    %40:3 = fork [3] %39 {handshake.bb = 2 : ui32, handshake.name = "fork8"} : <i32>
    %result_6, %index_7 = control_merge [%trueResult_4]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>] to <>, <i1>
    sink %index_7 {handshake.name = "sink3"} : <i1>
    %41 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %42 = constant %41 {handshake.bb = 2 : ui32, handshake.name = "constant2", value = 1 : i2} : <>, <i2>
    %43 = extsi %42 {handshake.bb = 2 : ui32, handshake.name = "extsi1"} : <i2> to <i32>
    %44 = addi %36#3, %32#3 {handshake.bb = 2 : ui32, handshake.name = "addi0"} : <i32>
    %45 = shrsi %44, %43 {handshake.bb = 2 : ui32, handshake.name = "shrsi0"} : <i32>
    %46:3 = fork [3] %45 {handshake.bb = 2 : ui32, handshake.name = "fork9"} : <i32>
    %47 = muli %46#1, %46#2 {handshake.bb = 2 : ui32, handshake.name = "muli0"} : <i32>
    %48:3 = fork [3] %47 {handshake.bb = 2 : ui32, handshake.name = "fork10"} : <i32>
    %49 = cmpi ne, %48#2, %40#2 {handshake.bb = 2 : ui32, handshake.name = "cmpi1"} : <i32>
    %50 = cmpi sle, %36#2, %32#2 {handshake.bb = 2 : ui32, handshake.name = "cmpi2"} : <i32>
    %51 = cmpi sgt, %36#1, %32#1 {handshake.bb = 2 : ui32, handshake.name = "cmpi3"} : <i32>
    %52 = andi %50, %49 {handshake.bb = 2 : ui32, handshake.name = "andi1"} : <i1>
    %53 = cmpi eq, %48#1, %40#1 {handshake.bb = 2 : ui32, handshake.name = "cmpi4"} : <i32>
    %54:7 = fork [7] %53 {handshake.bb = 2 : ui32, handshake.name = "fork11"} : <i1>
    %55 = ori %52, %51 {handshake.bb = 2 : ui32, handshake.name = "ori0"} : <i1>
    %trueResult_8, %falseResult_9 = cond_br %54#6, %46#0 {handshake.bb = 2 : ui32, handshake.name = "cond_br7"} : <i1>, <i32>
    %trueResult_10, %falseResult_11 = cond_br %54#5, %36#0 {handshake.bb = 2 : ui32, handshake.name = "cond_br8"} : <i1>, <i32>
    %trueResult_12, %falseResult_13 = cond_br %54#4, %55 {handshake.bb = 2 : ui32, handshake.name = "cond_br9"} : <i1>, <i1>
    %trueResult_14, %falseResult_15 = cond_br %54#3, %40#0 {handshake.bb = 2 : ui32, handshake.name = "cond_br10"} : <i1>, <i32>
    %56 = buffer %result_6 {handshake.bb = 2 : ui32, handshake.name = "buffer16", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <>
    %57 = buffer %56 {handshake.bb = 2 : ui32, handshake.name = "buffer17", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %trueResult_16, %falseResult_17 = cond_br %54#2, %57 {handshake.bb = 2 : ui32, handshake.name = "cond_br11"} : <i1>, <>
    %trueResult_18, %falseResult_19 = cond_br %54#1, %32#0 {handshake.bb = 2 : ui32, handshake.name = "cond_br12"} : <i1>, <i32>
    sink %trueResult_18 {handshake.name = "sink4"} : <i32>
    %trueResult_20, %falseResult_21 = cond_br %54#0, %48#0 {handshake.bb = 2 : ui32, handshake.name = "cond_br13"} : <i1>, <i32>
    sink %trueResult_20 {handshake.name = "sink5"} : <i32>
    %58 = merge %falseResult_15 {handshake.bb = 3 : ui32, handshake.name = "merge3"} : <i32>
    %59 = buffer %58 {handshake.bb = 3 : ui32, handshake.name = "buffer18", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i32>
    %60 = buffer %59 {handshake.bb = 3 : ui32, handshake.name = "buffer19", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i32>
    %61:2 = fork [2] %60 {handshake.bb = 3 : ui32, handshake.name = "fork12"} : <i32>
    %62 = merge %falseResult_19 {handshake.bb = 3 : ui32, handshake.name = "merge4"} : <i32>
    %63 = merge %falseResult_11 {handshake.bb = 3 : ui32, handshake.name = "merge5"} : <i32>
    %64 = merge %falseResult_9 {handshake.bb = 3 : ui32, handshake.name = "merge6"} : <i32>
    %65 = merge %falseResult_21 {handshake.bb = 3 : ui32, handshake.name = "merge7"} : <i32>
    %66 = merge %falseResult_13 {handshake.bb = 3 : ui32, handshake.name = "merge8"} : <i1>
    %result_22, %index_23 = control_merge [%falseResult_17]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>] to <>, <i1>
    sink %index_23 {handshake.name = "sink6"} : <i1>
    %67 = buffer %65 {handshake.bb = 3 : ui32, handshake.name = "buffer26", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i32>
    %68 = buffer %67 {handshake.bb = 3 : ui32, handshake.name = "buffer27", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i32>
    %69 = cmpi slt, %68, %61#1 {handshake.bb = 3 : ui32, handshake.name = "cmpi5"} : <i32>
    %70:6 = fork [6] %69 {handshake.bb = 3 : ui32, handshake.name = "fork13"} : <i1>
    %trueResult_24, %falseResult_25 = cond_br %70#5, %61#0 {handshake.bb = 3 : ui32, handshake.name = "cond_br14"} : <i1>, <i32>
    %71 = buffer %62 {handshake.bb = 3 : ui32, handshake.name = "buffer20", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i32>
    %72 = buffer %71 {handshake.bb = 3 : ui32, handshake.name = "buffer21", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i32>
    %trueResult_26, %falseResult_27 = cond_br %70#4, %72 {handshake.bb = 3 : ui32, handshake.name = "cond_br15"} : <i1>, <i32>
    sink %falseResult_27 {handshake.name = "sink7"} : <i32>
    %73 = buffer %64 {handshake.bb = 3 : ui32, handshake.name = "buffer24", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i32>
    %74 = buffer %73 {handshake.bb = 3 : ui32, handshake.name = "buffer25", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i32>
    %trueResult_28, %falseResult_29 = cond_br %70#3, %74 {handshake.bb = 3 : ui32, handshake.name = "cond_br16"} : <i1>, <i32>
    %75 = buffer %66 {handshake.bb = 3 : ui32, handshake.name = "buffer28", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %76 = buffer %75 {handshake.bb = 3 : ui32, handshake.name = "buffer29", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i1>
    %trueResult_30, %falseResult_31 = cond_br %70#2, %76 {handshake.bb = 3 : ui32, handshake.name = "cond_br17"} : <i1>, <i1>
    %77 = buffer %result_22 {handshake.bb = 3 : ui32, handshake.name = "buffer30", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <>
    %78 = buffer %77 {handshake.bb = 3 : ui32, handshake.name = "buffer31", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %trueResult_32, %falseResult_33 = cond_br %70#1, %78 {handshake.bb = 3 : ui32, handshake.name = "cond_br18"} : <i1>, <>
    %79 = buffer %63 {handshake.bb = 3 : ui32, handshake.name = "buffer22", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i32>
    %80 = buffer %79 {handshake.bb = 3 : ui32, handshake.name = "buffer23", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i32>
    %trueResult_34, %falseResult_35 = cond_br %70#0, %80 {handshake.bb = 3 : ui32, handshake.name = "cond_br19"} : <i1>, <i32>
    sink %trueResult_34 {handshake.name = "sink8"} : <i32>
    %81 = merge %trueResult_24 {handshake.bb = 4 : ui32, handshake.name = "merge9"} : <i32>
    %82 = merge %trueResult_26 {handshake.bb = 4 : ui32, handshake.name = "merge10"} : <i32>
    %83 = merge %trueResult_28 {handshake.bb = 4 : ui32, handshake.name = "merge11"} : <i32>
    %84 = merge %trueResult_30 {handshake.bb = 4 : ui32, handshake.name = "merge12"} : <i1>
    %result_36, %index_37 = control_merge [%trueResult_32]  {handshake.bb = 4 : ui32, handshake.name = "control_merge3"} : [<>] to <>, <i1>
    sink %index_37 {handshake.name = "sink9"} : <i1>
    %85 = source {handshake.bb = 4 : ui32, handshake.name = "source1"} : <>
    %86 = constant %85 {handshake.bb = 4 : ui32, handshake.name = "constant7", value = 1 : i2} : <>, <i2>
    %87 = extsi %86 {handshake.bb = 4 : ui32, handshake.name = "extsi2"} : <i2> to <i32>
    %88 = buffer %83 {handshake.bb = 4 : ui32, handshake.name = "buffer36", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i32>
    %89 = buffer %88 {handshake.bb = 4 : ui32, handshake.name = "buffer37", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i32>
    %90 = addi %89, %87 {handshake.bb = 4 : ui32, handshake.name = "addi1"} : <i32>
    %91 = buffer %82 {handshake.bb = 4 : ui32, handshake.name = "buffer34", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i32>
    %92 = buffer %91 {handshake.bb = 4 : ui32, handshake.name = "buffer35", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i32>
    %93 = br %92 {handshake.bb = 4 : ui32, handshake.name = "br6"} : <i32>
    %94 = br %90 {handshake.bb = 4 : ui32, handshake.name = "br8"} : <i32>
    %95 = buffer %84 {handshake.bb = 4 : ui32, handshake.name = "buffer38", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %96 = buffer %95 {handshake.bb = 4 : ui32, handshake.name = "buffer39", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i1>
    %97 = br %96 {handshake.bb = 4 : ui32, handshake.name = "br9"} : <i1>
    %98 = buffer %81 {handshake.bb = 4 : ui32, handshake.name = "buffer32", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i32>
    %99 = buffer %98 {handshake.bb = 4 : ui32, handshake.name = "buffer33", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i32>
    %100 = br %99 {handshake.bb = 4 : ui32, handshake.name = "br10"} : <i32>
    %101 = buffer %result_36 {handshake.bb = 4 : ui32, handshake.name = "buffer40", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <>
    %102 = buffer %101 {handshake.bb = 4 : ui32, handshake.name = "buffer41", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %103 = br %102 {handshake.bb = 4 : ui32, handshake.name = "br11"} : <>
    %104 = merge %falseResult_25 {handshake.bb = 5 : ui32, handshake.name = "merge13"} : <i32>
    %105 = merge %falseResult_35 {handshake.bb = 5 : ui32, handshake.name = "merge14"} : <i32>
    %106 = merge %falseResult_29 {handshake.bb = 5 : ui32, handshake.name = "merge15"} : <i32>
    %107 = merge %falseResult_31 {handshake.bb = 5 : ui32, handshake.name = "merge16"} : <i1>
    %result_38, %index_39 = control_merge [%falseResult_33]  {handshake.bb = 5 : ui32, handshake.name = "control_merge4"} : [<>] to <>, <i1>
    sink %index_39 {handshake.name = "sink10"} : <i1>
    %108 = source {handshake.bb = 5 : ui32, handshake.name = "source2"} : <>
    %109 = constant %108 {handshake.bb = 5 : ui32, handshake.name = "constant6", value = -1 : i32} : <>, <i32>
    %110 = buffer %106 {handshake.bb = 5 : ui32, handshake.name = "buffer46", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i32>
    %111 = buffer %110 {handshake.bb = 5 : ui32, handshake.name = "buffer47", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i32>
    %112 = addi %111, %109 {handshake.bb = 5 : ui32, handshake.name = "addi2"} : <i32>
    %113 = br %112 {handshake.bb = 5 : ui32, handshake.name = "br12"} : <i32>
    %114 = buffer %105 {handshake.bb = 5 : ui32, handshake.name = "buffer44", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i32>
    %115 = buffer %114 {handshake.bb = 5 : ui32, handshake.name = "buffer45", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i32>
    %116 = br %115 {handshake.bb = 5 : ui32, handshake.name = "br13"} : <i32>
    %117 = buffer %107 {handshake.bb = 5 : ui32, handshake.name = "buffer48", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %118 = buffer %117 {handshake.bb = 5 : ui32, handshake.name = "buffer49", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i1>
    %119 = br %118 {handshake.bb = 5 : ui32, handshake.name = "br14"} : <i1>
    %120 = buffer %104 {handshake.bb = 5 : ui32, handshake.name = "buffer42", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i32>
    %121 = buffer %120 {handshake.bb = 5 : ui32, handshake.name = "buffer43", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i32>
    %122 = br %121 {handshake.bb = 5 : ui32, handshake.name = "br15"} : <i32>
    %123 = buffer %result_38 {handshake.bb = 5 : ui32, handshake.name = "buffer50", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <>
    %124 = buffer %123 {handshake.bb = 5 : ui32, handshake.name = "buffer51", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %125 = br %124 {handshake.bb = 5 : ui32, handshake.name = "br16"} : <>
    %126 = merge %falseResult {handshake.bb = 6 : ui32, handshake.name = "merge17"} : <i32>
    %127 = buffer %126 {handshake.bb = 6 : ui32, handshake.name = "buffer52", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i32>
    %128 = buffer %127 {handshake.bb = 6 : ui32, handshake.name = "buffer53", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i32>
    end {handshake.bb = 6 : ui32, handshake.name = "end0"} %128, %0#0 : <i32>, <>
  }
}

