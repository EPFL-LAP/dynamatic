module {
  handshake.func @iterative_sqrt(%arg0: !handshake.channel<i32>, %arg1: !handshake.control<>, ...) -> (!handshake.channel<i32>, !handshake.control<>) attributes {argNames = ["n", "start"], cfg.edges = "[0,1][2,3,1,cmpi2][4,1][1,2,6,andi0][3,4,5,cmpi3][5,1]", resNames = ["out0", "end"]} {
    %0:2 = fork [2] %arg1 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork0"} : <>
    %1:2 = fork [2] %arg0 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork1"} : <i32>
    %2 = source {handshake.bb = 0 : ui32, handshake.name = "source0"} : <>
    %3 = constant %2 {handshake.bb = 0 : ui32, handshake.name = "constant2", value = false} : <>, <i1>
    %4 = extsi %3 {handshake.bb = 1 : ui32, handshake.name = "extsi0"} : <i1> to <i32>
    %5 = source {handshake.bb = 0 : ui32, handshake.name = "source1"} : <>
    %6 = constant %5 {handshake.bb = 0 : ui32, handshake.name = "constant3", value = true} : <>, <i1>
    %7:2 = fork [2] %6 {handshake.bb = 0 : ui32, handshake.name = "fork2"} : <i1>
    %8 = buffer %26 {handshake.bb = 1 : ui32, handshake.name = "buffer8", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i32>
    %9 = buffer %8 {handshake.bb = 1 : ui32, handshake.name = "buffer9", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i32>
    %trueResult, %falseResult = cond_br %43#5, %9 {handshake.bb = 1 : ui32, handshake.name = "cond_br35"} : <i1>, <i32>
    sink %falseResult {handshake.name = "sink0"} : <i32>
    %10 = buffer %22 {handshake.bb = 1 : ui32, handshake.name = "buffer4", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i32>
    %11 = buffer %10 {handshake.bb = 1 : ui32, handshake.name = "buffer5", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i32>
    %trueResult_0, %falseResult_1 = cond_br %43#6, %11 {handshake.bb = 1 : ui32, handshake.name = "cond_br36"} : <i1>, <i32>
    sink %falseResult_1 {handshake.name = "sink1"} : <i32>
    %12 = constant %0#1 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "constant1", value = false} : <>, <i1>
    %13:2 = fork [2] %12 {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <i1>
    %14 = extsi %13#1 {handshake.bb = 1 : ui32, handshake.name = "extsi1"} : <i1> to <i32>
    %15 = merge %13#0, %43#7 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "merge0"} : <i1>
    %16 = buffer %15 {handshake.bb = 1 : ui32, handshake.name = "buffer0", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %17 = buffer %16 {handshake.bb = 1 : ui32, handshake.name = "buffer1", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i1>
    %18:6 = fork [6] %17 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork4"} : <i1>
    %19 = mux %18#5 [%1#1, %50#3] {ftd.regen, handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i32>, <i32>] to <i32>
    %20 = buffer %23 {handshake.bb = 1 : ui32, handshake.name = "buffer6", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i32>
    %21 = buffer %20 {handshake.bb = 1 : ui32, handshake.name = "buffer7", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i32>
    %22 = mux %61#1 [%34#2, %21] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux3"} : <i1>, [<i32>, <i32>] to <i32>
    %23 = mux %65#1 [%34#3, %69] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux4"} : <i1>, [<i32>, <i32>] to <i32>
    %24 = buffer %27 {handshake.bb = 1 : ui32, handshake.name = "buffer10", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i32>
    %25 = buffer %24 {handshake.bb = 1 : ui32, handshake.name = "buffer11", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i32>
    %26 = mux %61#2 [%38#2, %25] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux5"} : <i1>, [<i32>, <i32>] to <i32>
    %27 = mux %65#2 [%72, %38#3] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux6"} : <i1>, [<i32>, <i32>] to <i32>
    %28 = mux %18#4 [%7#1, %61#3] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux7"} : <i1>, [<i1>, <i1>] to <i1>
    %29 = mux %18#3 [%14, %63] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux8"} : <i1>, [<i32>, <i32>] to <i32>
    %30 = mux %18#2 [%7#0, %62] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux9"} : <i1>, [<i1>, <i1>] to <i1>
    %31 = mux %18#1 [%4, %trueResult_0] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux10"} : <i1>, [<i32>, <i32>] to <i32>
    %32 = buffer %31 {handshake.bb = 1 : ui32, handshake.name = "buffer18", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i32>
    %33 = buffer %32 {handshake.bb = 1 : ui32, handshake.name = "buffer19", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i32>
    %34:4 = fork [4] %33 {handshake.bb = 1 : ui32, handshake.name = "fork5"} : <i32>
    %35 = mux %18#0 [%1#0, %trueResult] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux11"} : <i1>, [<i32>, <i32>] to <i32>
    %36 = buffer %35 {handshake.bb = 1 : ui32, handshake.name = "buffer20", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i32>
    %37 = buffer %36 {handshake.bb = 1 : ui32, handshake.name = "buffer21", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i32>
    %38:4 = fork [4] %37 {handshake.bb = 1 : ui32, handshake.name = "fork6"} : <i32>
    %39 = cmpi sle, %34#1, %38#1 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i32>
    %40 = buffer %28 {handshake.bb = 1 : ui32, handshake.name = "buffer12", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %41 = buffer %40 {handshake.bb = 1 : ui32, handshake.name = "buffer13", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i1>
    %42 = andi %39, %41 {handshake.bb = 1 : ui32, handshake.name = "andi0"} : <i1>
    %43:8 = fork [8] %42 {handshake.bb = 1 : ui32, handshake.name = "fork7"} : <i1>
    %44 = buffer %29 {handshake.bb = 2 : ui32, handshake.name = "buffer14", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i32>
    %45 = buffer %44 {handshake.bb = 2 : ui32, handshake.name = "buffer15", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i32>
    %trueResult_2, %falseResult_3 = cond_br %43#4, %45 {handshake.bb = 2 : ui32, handshake.name = "cond_br37"} : <i1>, <i32>
    %46 = buffer %30 {handshake.bb = 2 : ui32, handshake.name = "buffer16", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %47 = buffer %46 {handshake.bb = 2 : ui32, handshake.name = "buffer17", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i1>
    %trueResult_4, %falseResult_5 = cond_br %43#3, %47 {handshake.bb = 2 : ui32, handshake.name = "cond_br38"} : <i1>, <i1>
    %48 = buffer %19 {handshake.bb = 2 : ui32, handshake.name = "buffer2", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i32>
    %49 = buffer %48 {handshake.bb = 2 : ui32, handshake.name = "buffer3", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i32>
    %trueResult_6, %falseResult_7 = cond_br %43#2, %49 {handshake.bb = 2 : ui32, handshake.name = "cond_br39"} : <i1>, <i32>
    sink %falseResult_7 {handshake.name = "sink2"} : <i32>
    %50:4 = fork [4] %trueResult_6 {handshake.bb = 2 : ui32, handshake.name = "fork8"} : <i32>
    %trueResult_8, %falseResult_9 = cond_br %43#1, %38#0 {handshake.bb = 2 : ui32, handshake.name = "cond_br40"} : <i1>, <i32>
    %trueResult_10, %falseResult_11 = cond_br %43#0, %34#0 {handshake.bb = 2 : ui32, handshake.name = "cond_br41"} : <i1>, <i32>
    sink %falseResult_11 {handshake.name = "sink3"} : <i32>
    %51 = source {handshake.bb = 2 : ui32, handshake.name = "source6"} : <>
    %52 = constant %51 {handshake.bb = 2 : ui32, handshake.name = "constant7", value = 1 : i2} : <>, <i2>
    %53 = extsi %52 {handshake.bb = 2 : ui32, handshake.name = "extsi2"} : <i2> to <i32>
    %54 = addi %trueResult_10, %trueResult_8 {handshake.bb = 2 : ui32, handshake.name = "addi0"} : <i32>
    %55 = shrsi %54, %53 {handshake.bb = 2 : ui32, handshake.name = "shrsi0"} : <i32>
    %56:4 = fork [4] %55 {handshake.bb = 2 : ui32, handshake.name = "fork9"} : <i32>
    %57 = muli %56#2, %56#3 {handshake.bb = 2 : ui32, handshake.name = "muli0"} : <i32>
    %58:3 = fork [3] %57 {handshake.bb = 2 : ui32, handshake.name = "fork10"} : <i32>
    %59 = cmpi eq, %58#2, %50#2 {handshake.bb = 2 : ui32, handshake.name = "cmpi1"} : <i32>
    %60 = cmpi ne, %58#1, %50#1 {handshake.bb = 2 : ui32, handshake.name = "cmpi2"} : <i32>
    %61:4 = fork [4] %60 {handshake.bb = 2 : ui32, handshake.name = "fork11"} : <i1>
    %62 = andi %61#0, %trueResult_4 {handshake.bb = 2 : ui32, handshake.name = "andi1"} : <i1>
    %63 = select %59[%56#1, %trueResult_2] {handshake.bb = 2 : ui32, handshake.name = "select0"} : <i1>, <i32>
    %64 = cmpi slt, %58#0, %50#0 {handshake.bb = 2 : ui32, handshake.name = "cmpi3"} : <i32>
    %65:3 = fork [3] %64 {handshake.bb = 2 : ui32, handshake.name = "fork12"} : <i1>
    %trueResult_12, %falseResult_13 = cond_br %65#0, %56#0 {handshake.bb = 4 : ui32, handshake.name = "cond_br42"} : <i1>, <i32>
    %66 = source {handshake.bb = 4 : ui32, handshake.name = "source7"} : <>
    %67 = constant %66 {handshake.bb = 4 : ui32, handshake.name = "constant8", value = 1 : i2} : <>, <i2>
    %68 = extsi %67 {handshake.bb = 4 : ui32, handshake.name = "extsi3"} : <i2> to <i32>
    %69 = addi %trueResult_12, %68 {handshake.bb = 4 : ui32, handshake.name = "addi1"} : <i32>
    %70 = source {handshake.bb = 5 : ui32, handshake.name = "source8"} : <>
    %71 = constant %70 {handshake.bb = 5 : ui32, handshake.name = "constant6", value = -1 : i32} : <>, <i32>
    %72 = addi %falseResult_13, %71 {handshake.bb = 5 : ui32, handshake.name = "addi2"} : <i32>
    %73 = select %falseResult_5[%falseResult_9, %falseResult_3] {handshake.bb = 6 : ui32, handshake.name = "select1"} : <i1>, <i32>
    end {handshake.bb = 6 : ui32, handshake.name = "end0"} %73, %0#0 : <i32>, <>
  }
}

