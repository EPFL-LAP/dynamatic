module {
  handshake.func @matching(%arg0: !handshake.channel<i32>, %arg1: !handshake.control<>, ...) -> (!handshake.channel<f32>, !handshake.control<>) attributes {argNames = ["num_edges", "start"], cfg.edges = "[0,1][2,1][1,2,3,cmpi0]", resNames = ["out0", "end"]} {
    %0:4 = fork [4] %arg1 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork0"} : <>
    %1 = constant %0#3 {handshake.bb = 0 : ui32, handshake.name = "constant0", value = 0.000000e+00 : f32} : <>, <f32>
    %trueResult, %falseResult = cond_br %20#3, %13#1 {handshake.bb = 1 : ui32, handshake.name = "cond_br10"} : <i1>, <i32>
    sink %falseResult {handshake.name = "sink0"} : <i32>
    %2 = constant %0#2 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "constant1", value = false} : <>, <i1>
    %3:2 = fork [2] %2 {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <i1>
    %4 = extsi %3#1 {handshake.bb = 1 : ui32, handshake.name = "extsi0"} : <i1> to <i32>
    %5 = merge %3#0, %20#4 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "merge0"} : <i1>
    %6 = buffer %5 {handshake.bb = 1 : ui32, handshake.name = "buffer0", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %7 = buffer %6 {handshake.bb = 1 : ui32, handshake.name = "buffer1", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i1>
    %8:4 = fork [4] %7 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"0": [1,inf], [0,inf], 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork2"} : <i1>
    %9 = mux %8#3 [%0#1, %25#2] {ftd.regen, handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<>, <>] to <>
    %10 = mux %8#2 [%arg0, %trueResult] {ftd.regen, handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mux2"} : <i1>, [<i32>, <i32>] to <i32>
    %11 = buffer %10 {handshake.bb = 1 : ui32, handshake.name = "buffer4", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i32>
    %12 = buffer %11 {handshake.bb = 1 : ui32, handshake.name = "buffer5", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i32>
    %13:2 = fork [2] %12 {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <i32>
    %14 = mux %8#1 [%1, %52] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux3"} : <i1>, [<f32>, <f32>] to <f32>
    %15 = mux %8#0 [%4, %55] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux4"} : <i1>, [<i32>, <i32>] to <i32>
    %16 = buffer %15 {handshake.bb = 1 : ui32, handshake.name = "buffer8", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i32>
    %17 = buffer %16 {handshake.bb = 1 : ui32, handshake.name = "buffer9", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i32>
    %18:2 = fork [2] %17 {handshake.bb = 1 : ui32, handshake.name = "fork4"} : <i32>
    %19 = cmpi slt, %18#1, %13#0 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i32>
    %20:5 = fork [5] %19 {handshake.bb = 1 : ui32, handshake.name = "fork5"} : <i1>
    %trueResult_0, %falseResult_1 = cond_br %20#2, %18#0 {handshake.bb = 2 : ui32, handshake.name = "cond_br11"} : <i1>, <i32>
    sink %falseResult_1 {handshake.name = "sink1"} : <i32>
    %21 = buffer %14 {handshake.bb = 2 : ui32, handshake.name = "buffer6", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <f32>
    %22 = buffer %21 {handshake.bb = 2 : ui32, handshake.name = "buffer7", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <f32>
    %trueResult_2, %falseResult_3 = cond_br %20#1, %22 {handshake.bb = 2 : ui32, handshake.name = "cond_br12"} : <i1>, <f32>
    %23 = buffer %9 {handshake.bb = 2 : ui32, handshake.name = "buffer2", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <>
    %24 = buffer %23 {handshake.bb = 2 : ui32, handshake.name = "buffer3", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %trueResult_4, %falseResult_5 = cond_br %20#0, %24 {handshake.bb = 2 : ui32, handshake.name = "cond_br13"} : <i1>, <>
    sink %falseResult_5 {handshake.name = "sink2"} : <>
    %25:3 = fork [3] %trueResult_4 {handshake.bb = 2 : ui32, handshake.name = "fork6"} : <>
    %26 = free_tags_fifo %45 {handshake.bb = 2 : ui32, handshake.name = "free_tags_fifo0"} : <i2>, <i2>
    %27:2 = fork [2] %26 {handshake.bb = 2 : ui32, handshake.name = "fork7"} : <i2>
    %28 = tagger[%27#1] %25#1 {handshake.bb = 2 : ui32, handshake.name = "tagger0"} : (!handshake.control<>, !handshake.channel<i2>) -> !handshake.control<[tag0: i2]>
    %29 = tagger[%27#0] %trueResult_0 {handshake.bb = 2 : ui32, handshake.name = "tagger1"} : (!handshake.channel<i32>, !handshake.channel<i2>) -> !handshake.channel<i32, [tag0: i2]>
    %dataOut, %tagOut = untagger %47#0 {handshake.bb = 2 : ui32, handshake.name = "untagger0"} : (!handshake.channel<i2, [tag0: i2]>) -> (!handshake.channel<i2>, !handshake.channel<i2>)
    %30:2 = fork [2] %tagOut {handshake.bb = 2 : ui32, handshake.name = "fork8"} : <i2>
    sink %dataOut {handshake.name = "sink3"} : <i2>
    %dataOut_6, %tagOut_7 = untagger %29 {handshake.bb = 2 : ui32, handshake.name = "untagger1"} : (!handshake.channel<i32, [tag0: i2]>) -> (!handshake.channel<i32>, !handshake.channel<i2>)
    %31:2 = fork [2] %tagOut_7 {handshake.bb = 2 : ui32, handshake.name = "fork9"} : <i2>
    %32:4 = demux[%31#1] %dataOut_6 {handshake.bb = 2 : ui32, handshake.name = "demux0"} : <i2>, (!handshake.channel<i32>) -> (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>)
    %33 = buffer %32#0 {handshake.bb = 2 : ui32, handshake.name = "buffer10", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i32>
    %34 = buffer %32#1 {handshake.bb = 2 : ui32, handshake.name = "buffer11", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i32>
    %35 = buffer %32#2 {handshake.bb = 2 : ui32, handshake.name = "buffer12", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i32>
    %36 = buffer %32#3 {handshake.bb = 2 : ui32, handshake.name = "buffer13", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i32>
    %37 = mux %30#1 [%33, %34, %35, %36] {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [1,inf], [0,inf], 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [1,inf], [0,inf], 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [1,inf], [0,inf], 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [1,inf], [0,inf], 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mux1"} : <i2>, [<i32>, <i32>, <i32>, <i32>] to <i32>
    %dataOut_8, %tagOut_9 = untagger %47#1 {handshake.bb = 2 : ui32, handshake.name = "untagger2"} : (!handshake.channel<i2, [tag0: i2]>) -> (!handshake.channel<i2>, !handshake.channel<i2>)
    %38:2 = fork [2] %tagOut_9 {handshake.bb = 2 : ui32, handshake.name = "fork10"} : <i2>
    %39:4 = demux[%38#1] %dataOut_8 {handshake.bb = 2 : ui32, handshake.name = "demux1"} : <i2>, (!handshake.channel<i2>) -> (!handshake.channel<i2>, !handshake.channel<i2>, !handshake.channel<i2>, !handshake.channel<i2>)
    %40 = buffer %39#0 {handshake.bb = 2 : ui32, handshake.name = "buffer16", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i2>
    %41 = buffer %39#1 {handshake.bb = 2 : ui32, handshake.name = "buffer17", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i2>
    %42 = buffer %39#2 {handshake.bb = 2 : ui32, handshake.name = "buffer18", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i2>
    %43 = buffer %39#3 {handshake.bb = 2 : ui32, handshake.name = "buffer19", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i2>
    %44 = mux %30#0 [%40, %41, %42, %43] {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [1,inf], [0,inf], 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [1,inf], [0,inf], 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [1,inf], [0,inf], 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [1,inf], [0,inf], 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mux5"} : <i2>, [<i2>, <i2>, <i2>, <i2>] to <i2>
    %45 = join %31#0, %38#0 {handshake.bb = 2 : ui32, handshake.name = "join0"} : <i2>
    %46 = constant %28 {handshake.bb = 2 : ui32, handshake.name = "constant3", value = 1 : i2} : <[tag0: i2]>, <i2, [tag0: i2]>
    %47:2 = fork [2] %46 {handshake.bb = 2 : ui32, handshake.name = "fork11"} : <i2, [tag0: i2]>
    %48 = buffer %44 {handshake.bb = 2 : ui32, handshake.name = "buffer20", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i2>
    %49 = buffer %48 {handshake.bb = 2 : ui32, handshake.name = "buffer21", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i2>
    %50 = extsi %49 {handshake.bb = 2 : ui32, handshake.name = "extsi1"} : <i2> to <i32>
    %51 = constant %25#0 {handshake.bb = 2 : ui32, handshake.name = "constant5", value = 1.000000e+00 : f32} : <>, <f32>
    %52 = addf %trueResult_2, %51 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "addf0"} : <f32>
    %53 = buffer %37 {handshake.bb = 2 : ui32, handshake.name = "buffer14", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i32>
    %54 = buffer %53 {handshake.bb = 2 : ui32, handshake.name = "buffer15", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i32>
    %55 = addi %54, %50 {handshake.bb = 2 : ui32, handshake.name = "addi0"} : <i32>
    end {handshake.bb = 3 : ui32, handshake.name = "end0"} %falseResult_3, %0#0 : <f32>, <>
  }
}

