module {
  handshake.func @matching(%arg0: !handshake.channel<i32>, %arg1: !handshake.control<>, ...) -> (!handshake.channel<f32>, !handshake.control<>) attributes {argNames = ["num_edges", "start"], cfg.edges = "[0,1][2,1][1,2,3,cmpi0]", resNames = ["out0", "end"]} {
    %0:4 = fork [4] %arg1 {handshake.bb = 0 : ui32, handshake.name = "fork0"} : <>
    %1 = constant %0#3 {handshake.bb = 0 : ui32, handshake.name = "constant0", value = 0.000000e+00 : f32} : <>, <f32>
    %trueResult, %falseResult = cond_br %14#3, %9#1 {handshake.bb = 1 : ui32, handshake.name = "cond_br10"} : <i1>, <i32>
    sink %falseResult {handshake.name = "sink0"} : <i32>
    %2 = constant %0#2 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "constant1", value = false} : <>, <i1>
    %3:2 = fork [2] %2 {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <i1>
    %4 = extsi %3#1 {handshake.bb = 1 : ui32, handshake.name = "extsi0"} : <i1> to <i32>
    %5 = merge %3#0, %14#4 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "merge0"} : <i1>
    %6:4 = fork [4] %5 {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <i1>
    %7 = mux %6#3 [%0#1, %15#2] {ftd.regen, handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<>, <>] to <>
    %8 = mux %6#2 [%arg0, %trueResult] {ftd.regen, handshake.bb = 1 : ui32, handshake.name = "mux2"} : <i1>, [<i32>, <i32>] to <i32>
    %9:2 = fork [2] %8 {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <i32>
    %10 = mux %6#1 [%1, %32] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux3"} : <i1>, [<f32>, <f32>] to <f32>
    %11 = mux %6#0 [%4, %33] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux4"} : <i1>, [<i32>, <i32>] to <i32>
    %12:2 = fork [2] %11 {handshake.bb = 1 : ui32, handshake.name = "fork4"} : <i32>
    %13 = cmpi slt, %12#1, %9#0 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i32>
    %14:5 = fork [5] %13 {handshake.bb = 1 : ui32, handshake.name = "fork5"} : <i1>
    %trueResult_0, %falseResult_1 = cond_br %14#2, %12#0 {handshake.bb = 2 : ui32, handshake.name = "cond_br11"} : <i1>, <i32>
    sink %falseResult_1 {handshake.name = "sink1"} : <i32>
    %trueResult_2, %falseResult_3 = cond_br %14#1, %10 {handshake.bb = 2 : ui32, handshake.name = "cond_br12"} : <i1>, <f32>
    %trueResult_4, %falseResult_5 = cond_br %14#0, %7 {handshake.bb = 2 : ui32, handshake.name = "cond_br13"} : <i1>, <>
    sink %falseResult_5 {handshake.name = "sink2"} : <>
    %15:3 = fork [3] %trueResult_4 {handshake.bb = 2 : ui32, handshake.name = "fork6"} : <>
    %16 = free_tags_fifo %27 {handshake.bb = 2 : ui32, handshake.name = "free_tags_fifo0"} : <i2>, <i2>
    %17:2 = fork [2] %16 {handshake.bb = 2 : ui32, handshake.name = "fork7"} : <i2>
    %18 = tagger[%17#1] %15#1 {handshake.bb = 2 : ui32, handshake.name = "tagger0"} : (!handshake.control<>, !handshake.channel<i2>) -> !handshake.control<[tag0: i2]>
    %19 = tagger[%17#0] %trueResult_0 {handshake.bb = 2 : ui32, handshake.name = "tagger1"} : (!handshake.channel<i32>, !handshake.channel<i2>) -> !handshake.channel<i32, [tag0: i2]>
    %dataOut, %tagOut = untagger %29#0 {handshake.bb = 2 : ui32, handshake.name = "untagger0"} : (!handshake.channel<i2, [tag0: i2]>) -> (!handshake.channel<i2>, !handshake.channel<i2>)
    %20:2 = fork [2] %tagOut {handshake.bb = 2 : ui32, handshake.name = "fork8"} : <i2>
    sink %dataOut {handshake.name = "sink3"} : <i2>
    %dataOut_6, %tagOut_7 = untagger %19 {handshake.bb = 2 : ui32, handshake.name = "untagger1"} : (!handshake.channel<i32, [tag0: i2]>) -> (!handshake.channel<i32>, !handshake.channel<i2>)
    %21:2 = fork [2] %tagOut_7 {handshake.bb = 2 : ui32, handshake.name = "fork9"} : <i2>
    %22:4 = demux[%21#1] %dataOut_6 {handshake.bb = 2 : ui32, handshake.name = "demux0"} : <i2>, (!handshake.channel<i32>) -> (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>)
    %23 = mux %20#1 [%22#0, %22#1, %22#2, %22#3] {handshake.bb = 2 : ui32, handshake.name = "mux1"} : <i2>, [<i32>, <i32>, <i32>, <i32>] to <i32>
    %dataOut_8, %tagOut_9 = untagger %29#1 {handshake.bb = 2 : ui32, handshake.name = "untagger2"} : (!handshake.channel<i2, [tag0: i2]>) -> (!handshake.channel<i2>, !handshake.channel<i2>)
    %24:2 = fork [2] %tagOut_9 {handshake.bb = 2 : ui32, handshake.name = "fork10"} : <i2>
    %25:4 = demux[%24#1] %dataOut_8 {handshake.bb = 2 : ui32, handshake.name = "demux1"} : <i2>, (!handshake.channel<i2>) -> (!handshake.channel<i2>, !handshake.channel<i2>, !handshake.channel<i2>, !handshake.channel<i2>)
    %26 = mux %20#0 [%25#0, %25#1, %25#2, %25#3] {handshake.bb = 2 : ui32, handshake.name = "mux5"} : <i2>, [<i2>, <i2>, <i2>, <i2>] to <i2>
    %27 = join %21#0, %24#0 {handshake.bb = 2 : ui32, handshake.name = "join0"} : <i2>
    %28 = constant %18 {handshake.bb = 2 : ui32, handshake.name = "constant3", value = 1 : i2} : <[tag0: i2]>, <i2, [tag0: i2]>
    %29:2 = fork [2] %28 {handshake.bb = 2 : ui32, handshake.name = "fork11"} : <i2, [tag0: i2]>
    %30 = extsi %26 {handshake.bb = 2 : ui32, handshake.name = "extsi1"} : <i2> to <i32>
    %31 = constant %15#0 {handshake.bb = 2 : ui32, handshake.name = "constant5", value = 1.000000e+00 : f32} : <>, <f32>
    %32 = addf %trueResult_2, %31 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "addf0"} : <f32>
    %33 = addi %23, %30 {handshake.bb = 2 : ui32, handshake.name = "addi0"} : <i32>
    end {handshake.bb = 3 : ui32, handshake.name = "end0"} %falseResult_3, %0#0 : <f32>, <>
  }
}

