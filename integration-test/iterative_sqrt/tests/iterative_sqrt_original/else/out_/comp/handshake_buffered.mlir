module {
  handshake.func @iterative_sqrt(%arg0: !handshake.channel<i32>, %arg1: !handshake.control<>, ...) -> (!handshake.channel<i32>, !handshake.control<>) attributes {argNames = ["n", "start"], cfg.edges = "[0,1][2,1,3,cmpi5][4,1][1,2,6,andi0][3,4,5,cmpi6][5,1]", resNames = ["out0", "end"]} {
    %0:2 = fork [2] %arg1 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork0"} : <>
    %1:2 = fork [2] %arg0 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork1"} : <i32>
    %2 = source {handshake.bb = 0 : ui32, handshake.name = "source0"} : <>
    %3 = constant %2 {handshake.bb = 0 : ui32, handshake.name = "constant2", value = false} : <>, <i1>
    %4 = extsi %3 {handshake.bb = 1 : ui32, handshake.name = "extsi0"} : <i1> to <i32>
    %5 = source {handshake.bb = 0 : ui32, handshake.name = "source1"} : <>
    %6 = constant %5 {handshake.bb = 0 : ui32, handshake.name = "constant3", value = true} : <>, <i1>
    %7:2 = fork [2] %6 {handshake.bb = 0 : ui32, handshake.name = "fork2"} : <i1>
    %8 = buffer %29 {handshake.bb = 1 : ui32, handshake.name = "buffer8", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i32>
    %9 = buffer %8 {handshake.bb = 1 : ui32, handshake.name = "buffer9", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i32>
    %trueResult, %falseResult = cond_br %56#6, %9 {handshake.bb = 1 : ui32, handshake.name = "cond_br39"} : <i1>, <i32>
    sink %falseResult {handshake.name = "sink0"} : <i32>
    %10 = buffer %25 {handshake.bb = 1 : ui32, handshake.name = "buffer4", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i32>
    %11 = buffer %10 {handshake.bb = 1 : ui32, handshake.name = "buffer5", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i32>
    %trueResult_0, %falseResult_1 = cond_br %56#7, %11 {handshake.bb = 1 : ui32, handshake.name = "cond_br40"} : <i1>, <i32>
    sink %falseResult_1 {handshake.name = "sink1"} : <i32>
    %trueResult_2, %falseResult_3 = cond_br %56#8, %70 {handshake.bb = 1 : ui32, handshake.name = "cond_br41"} : <i1>, <i1>
    %12 = constant %0#1 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "constant1", value = false} : <>, <i1>
    %13:2 = fork [2] %12 {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <i1>
    %14 = extsi %13#1 {handshake.bb = 1 : ui32, handshake.name = "extsi1"} : <i1> to <i32>
    %15 = merge %13#0, %56#9 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "merge0"} : <i1>
    %16 = buffer %15 {handshake.bb = 1 : ui32, handshake.name = "buffer0", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %17 = buffer %16 {handshake.bb = 1 : ui32, handshake.name = "buffer1", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i1>
    %18:6 = fork [6] %17 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork4"} : <i1>
    %19 = mux %18#5 [%1#1, %91#1] {ftd.regen, handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i32>, <i32>] to <i32>
    %20 = buffer %19 {handshake.bb = 1 : ui32, handshake.name = "buffer2", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i32>
    %21 = buffer %20 {handshake.bb = 1 : ui32, handshake.name = "buffer3", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i32>
    %22:2 = fork [2] %21 {handshake.bb = 1 : ui32, handshake.name = "fork5"} : <i32>
    %23 = buffer %26 {handshake.bb = 1 : ui32, handshake.name = "buffer6", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i32>
    %24 = buffer %23 {handshake.bb = 1 : ui32, handshake.name = "buffer7", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i32>
    %25 = mux %87#1 [%24, %43#3] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux4"} : <i1>, [<i32>, <i32>] to <i32>
    %26 = mux %93#1 [%43#4, %97] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux5"} : <i1>, [<i32>, <i32>] to <i32>
    %27 = buffer %30 {handshake.bb = 1 : ui32, handshake.name = "buffer10", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i32>
    %28 = buffer %27 {handshake.bb = 1 : ui32, handshake.name = "buffer11", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i32>
    %29 = mux %87#2 [%28, %47#3] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux6"} : <i1>, [<i32>, <i32>] to <i32>
    %30 = mux %93#2 [%100, %47#4] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux7"} : <i1>, [<i32>, <i32>] to <i32>
    %31 = mux %18#4 [%7#1, %90] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux8"} : <i1>, [<i1>, <i1>] to <i1>
    %32 = buffer %31 {handshake.bb = 1 : ui32, handshake.name = "buffer12", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %33 = buffer %32 {handshake.bb = 1 : ui32, handshake.name = "buffer13", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i1>
    %34:3 = fork [3] %33 {handshake.bb = 1 : ui32, handshake.name = "fork6"} : <i1>
    %35 = mux %18#3 [%14, %89] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux9"} : <i1>, [<i32>, <i32>] to <i32>
    %36 = mux %18#2 [%7#0, %trueResult_2] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux10"} : <i1>, [<i1>, <i1>] to <i1>
    %37 = buffer %36 {handshake.bb = 1 : ui32, handshake.name = "buffer16", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %38 = buffer %37 {handshake.bb = 1 : ui32, handshake.name = "buffer17", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i1>
    %39:3 = fork [3] %38 {handshake.bb = 1 : ui32, handshake.name = "fork7"} : <i1>
    %40 = mux %18#1 [%4, %trueResult_0] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux11"} : <i1>, [<i32>, <i32>] to <i32>
    %41 = buffer %40 {handshake.bb = 1 : ui32, handshake.name = "buffer18", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i32>
    %42 = buffer %41 {handshake.bb = 1 : ui32, handshake.name = "buffer19", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i32>
    %43:5 = fork [5] %42 {handshake.bb = 1 : ui32, handshake.name = "fork8"} : <i32>
    %44 = mux %18#0 [%1#0, %trueResult] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux12"} : <i1>, [<i32>, <i32>] to <i32>
    %45 = buffer %44 {handshake.bb = 1 : ui32, handshake.name = "buffer20", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i32>
    %46 = buffer %45 {handshake.bb = 1 : ui32, handshake.name = "buffer21", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i32>
    %47:5 = fork [5] %46 {handshake.bb = 1 : ui32, handshake.name = "fork9"} : <i32>
    %48 = source {handshake.bb = 1 : ui32, handshake.name = "source2"} : <>
    %49 = constant %48 {handshake.bb = 1 : ui32, handshake.name = "constant4", value = true} : <>, <i1>
    %50:2 = fork [2] %49 {handshake.bb = 1 : ui32, handshake.name = "fork10"} : <i1>
    %51 = source {handshake.bb = 1 : ui32, handshake.name = "source3"} : <>
    %52 = constant %51 {handshake.bb = 1 : ui32, handshake.name = "constant9", value = 1 : i2} : <>, <i2>
    %53 = extsi %52 {handshake.bb = 1 : ui32, handshake.name = "extsi2"} : <i2> to <i32>
    %54 = cmpi sle, %43#2, %47#2 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i32>
    %55 = andi %54, %34#2 {handshake.bb = 1 : ui32, handshake.name = "andi0"} : <i1>
    %56:10 = fork [10] %55 {handshake.bb = 1 : ui32, handshake.name = "fork11"} : <i1>
    %57 = addi %43#1, %47#1 {handshake.bb = 1 : ui32, handshake.name = "addi0"} : <i32>
    %58 = shrsi %57, %53 {handshake.bb = 1 : ui32, handshake.name = "shrsi0"} : <i32>
    %59:2 = fork [2] %58 {handshake.bb = 1 : ui32, handshake.name = "fork12"} : <i32>
    %60 = muli %59#0, %59#1 {handshake.bb = 1 : ui32, handshake.name = "muli0"} : <i32>
    %61 = cmpi ne, %60, %22#1 {handshake.bb = 1 : ui32, handshake.name = "cmpi1"} : <i32>
    %62 = andi %61, %39#2 {handshake.bb = 1 : ui32, handshake.name = "andi1"} : <i1>
    %63 = xori %34#1, %50#1 {handshake.bb = 1 : ui32, handshake.name = "xori0"} : <i1>
    %64 = andi %34#0, %62 {handshake.bb = 1 : ui32, handshake.name = "andi2"} : <i1>
    %65 = andi %63, %39#1 {handshake.bb = 1 : ui32, handshake.name = "andi3"} : <i1>
    %66 = ori %64, %65 {handshake.bb = 1 : ui32, handshake.name = "ori0"} : <i1>
    %67 = xori %56#5, %50#0 {handshake.bb = 1 : ui32, handshake.name = "xori1"} : <i1>
    %68 = andi %56#4, %66 {handshake.bb = 1 : ui32, handshake.name = "andi4"} : <i1>
    %69 = andi %67, %39#0 {handshake.bb = 1 : ui32, handshake.name = "andi5"} : <i1>
    %70 = ori %68, %69 {handshake.bb = 1 : ui32, handshake.name = "ori1"} : <i1>
    %71 = buffer %35 {handshake.bb = 2 : ui32, handshake.name = "buffer14", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i32>
    %72 = buffer %71 {handshake.bb = 2 : ui32, handshake.name = "buffer15", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i32>
    %trueResult_4, %falseResult_5 = cond_br %56#3, %72 {handshake.bb = 2 : ui32, handshake.name = "cond_br42"} : <i1>, <i32>
    %trueResult_6, %falseResult_7 = cond_br %56#2, %43#0 {handshake.bb = 2 : ui32, handshake.name = "cond_br43"} : <i1>, <i32>
    sink %falseResult_7 {handshake.name = "sink2"} : <i32>
    %73:3 = fork [3] %trueResult_6 {handshake.bb = 2 : ui32, handshake.name = "fork13"} : <i32>
    %trueResult_8, %falseResult_9 = cond_br %56#1, %47#0 {handshake.bb = 2 : ui32, handshake.name = "cond_br44"} : <i1>, <i32>
    %74:3 = fork [3] %trueResult_8 {handshake.bb = 2 : ui32, handshake.name = "fork14"} : <i32>
    %75 = source {handshake.bb = 2 : ui32, handshake.name = "source8"} : <>
    %76 = constant %75 {handshake.bb = 2 : ui32, handshake.name = "constant10", value = 1 : i2} : <>, <i2>
    %77 = extsi %76 {handshake.bb = 2 : ui32, handshake.name = "extsi3"} : <i2> to <i32>
    %78 = addi %73#2, %74#2 {handshake.bb = 2 : ui32, handshake.name = "addi1"} : <i32>
    %79 = shrsi %78, %77 {handshake.bb = 2 : ui32, handshake.name = "shrsi1"} : <i32>
    %80:4 = fork [4] %79 {handshake.bb = 2 : ui32, handshake.name = "fork15"} : <i32>
    %81 = muli %80#2, %80#3 {handshake.bb = 2 : ui32, handshake.name = "muli1"} : <i32>
    %82:3 = fork [3] %81 {handshake.bb = 2 : ui32, handshake.name = "fork16"} : <i32>
    %83 = cmpi ne, %82#2, %91#2 {handshake.bb = 2 : ui32, handshake.name = "cmpi2"} : <i32>
    %84 = cmpi sle, %73#1, %74#1 {handshake.bb = 2 : ui32, handshake.name = "cmpi3"} : <i32>
    %85 = cmpi sgt, %73#0, %74#0 {handshake.bb = 2 : ui32, handshake.name = "cmpi4"} : <i32>
    %86 = cmpi eq, %82#1, %91#3 {handshake.bb = 2 : ui32, handshake.name = "cmpi5"} : <i32>
    %87:3 = fork [3] %86 {handshake.bb = 2 : ui32, handshake.name = "fork17"} : <i1>
    %88 = andi %84, %83 {handshake.bb = 2 : ui32, handshake.name = "andi6"} : <i1>
    %89 = select %87#0[%80#1, %trueResult_4] {handshake.bb = 2 : ui32, handshake.name = "select0"} : <i1>, <i32>
    %90 = ori %88, %85 {handshake.bb = 2 : ui32, handshake.name = "ori2"} : <i1>
    %trueResult_10, %falseResult_11 = cond_br %56#0, %22#0 {handshake.bb = 3 : ui32, handshake.name = "cond_br45"} : <i1>, <i32>
    sink %falseResult_11 {handshake.name = "sink3"} : <i32>
    %91:4 = fork [4] %trueResult_10 {handshake.bb = 3 : ui32, handshake.name = "fork18"} : <i32>
    %92 = cmpi slt, %82#0, %91#0 {handshake.bb = 3 : ui32, handshake.name = "cmpi6"} : <i32>
    %93:3 = fork [3] %92 {handshake.bb = 3 : ui32, handshake.name = "fork19"} : <i1>
    %trueResult_12, %falseResult_13 = cond_br %93#0, %80#0 {handshake.bb = 4 : ui32, handshake.name = "cond_br46"} : <i1>, <i32>
    %94 = source {handshake.bb = 4 : ui32, handshake.name = "source9"} : <>
    %95 = constant %94 {handshake.bb = 4 : ui32, handshake.name = "constant11", value = 1 : i2} : <>, <i2>
    %96 = extsi %95 {handshake.bb = 4 : ui32, handshake.name = "extsi4"} : <i2> to <i32>
    %97 = addi %trueResult_12, %96 {handshake.bb = 4 : ui32, handshake.name = "addi2"} : <i32>
    %98 = source {handshake.bb = 5 : ui32, handshake.name = "source10"} : <>
    %99 = constant %98 {handshake.bb = 5 : ui32, handshake.name = "constant8", value = -1 : i32} : <>, <i32>
    %100 = addi %falseResult_13, %99 {handshake.bb = 5 : ui32, handshake.name = "addi3"} : <i32>
    %101 = select %falseResult_3[%falseResult_9, %falseResult_5] {handshake.bb = 6 : ui32, handshake.name = "select1"} : <i1>, <i32>
    end {handshake.bb = 6 : ui32, handshake.name = "end0"} %101, %0#0 : <i32>, <>
  }
}

