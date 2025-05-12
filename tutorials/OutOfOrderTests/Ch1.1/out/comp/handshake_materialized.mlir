module {
  handshake.func @loop_multiply(%arg0: memref<8xi32>, %arg1: !handshake.control<>, %arg2: !handshake.control<>, ...) -> (!handshake.channel<i32>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["a", "a_start", "start"], cfg.edges = "[0,1][1,1,2,cmpi0]", resNames = ["out0", "a_end", "end"]} {
    %0:5 = fork [5] %arg2 {handshake.bb = 0 : ui32, handshake.name = "fork0"} : <>
    %outputs, %memEnd = mem_controller[%arg0 : memref<8xi32>] %arg1 (%addressResult) %0#4 {connectedBlocks = [2 : i32], handshake.name = "mem_controller1"} :    (!handshake.channel<i3>) -> !handshake.channel<i32>
    %1 = constant %0#3 {handshake.bb = 0 : ui32, handshake.name = "constant1", value = 2 : i3} : <>, <i3>
    %2 = extsi %1 {handshake.bb = 0 : ui32, handshake.name = "extsi0"} : <i3> to <i32>
    %trueResult, %falseResult = cond_br %80#0, %78 {handshake.bb = 1 : ui32, handshake.name = "cond_br7"} : <i1, [tag1: i3]>, <i4, [tag1: i3]>
    %trueResult_0, %falseResult_1 = cond_br %47, %50 {handshake.bb = 1 : ui32, handshake.name = "cond_br8"} : <i1, [tag1: i3]>, <i32, [tag1: i3]>
    %trueResult_2, %falseResult_3 = cond_br %80#1, %67 {handshake.bb = 1 : ui32, handshake.name = "cond_br9"} : <i1, [tag1: i3]>, <[tag1: i3]>
    %3:2 = fork [2] %falseResult_3 {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <[tag1: i3]>
    %4 = constant %0#2 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "constant0", value = false} : <>, <i1>
    %5:3 = fork [3] %4 {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <i1>
    %6 = extsi %5#2 {handshake.bb = 1 : ui32, handshake.name = "extsi5"} : <i1> to <i4>
    %7 = extui %5#1 {handshake.bb = 1 : ui32, handshake.name = "extui0"} : <i1> to <i3>
    %8 = merge %14, %80#2 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "merge0"} : <i1, [tag1: i3]>
    %9:3 = fork [3] %8 {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <i1, [tag1: i3]>
    %10 = free_tags_fifo %30 {handshake.bb = 2 : ui32, handshake.name = "free_tags_fifo0"} : <i3>, <i3>
    %11:5 = fork [5] %10 {handshake.bb = 2 : ui32, handshake.name = "fork4"} : <i3>
    %12 = tagger[%11#4] %dataResult {handshake.bb = 2 : ui32, handshake.name = "tagger0"} : (!handshake.channel<i32>, !handshake.channel<i3>) -> !handshake.channel<i32, [tag1: i3]>
    %13 = tagger[%11#3] %6 {handshake.bb = 1 : ui32, handshake.name = "tagger1"} : (!handshake.channel<i4>, !handshake.channel<i3>) -> !handshake.channel<i4, [tag1: i3]>
    %14 = tagger[%11#2] %5#0 {handshake.bb = 1 : ui32, handshake.name = "tagger2"} : (!handshake.channel<i1>, !handshake.channel<i3>) -> !handshake.channel<i1, [tag1: i3]>
    %15 = tagger[%11#1] %2 {handshake.bb = 0 : ui32, handshake.name = "tagger3"} : (!handshake.channel<i32>, !handshake.channel<i3>) -> !handshake.channel<i32, [tag1: i3]>
    %16 = tagger[%11#0] %0#1 {handshake.bb = 0 : ui32, handshake.name = "tagger4"} : (!handshake.control<>, !handshake.channel<i3>) -> !handshake.control<[tag1: i3]>
    %dataOut, %tagOut = untagger %3#1 {handshake.bb = 2 : ui32, handshake.name = "untagger0"} : (!handshake.control<[tag1: i3]>) -> (!handshake.control<>, !handshake.channel<i3>)
    %17:4 = fork [4] %tagOut {handshake.bb = 2 : ui32, handshake.name = "fork5"} : <i3>
    sink %dataOut {handshake.name = "sink0"} : <>
    %dataOut_4, %tagOut_5 = untagger %3#0 {handshake.bb = 1 : ui32, handshake.name = "untagger1"} : (!handshake.control<[tag1: i3]>) -> (!handshake.control<>, !handshake.channel<i3>)
    %18:2 = fork [2] %tagOut_5 {handshake.bb = 1 : ui32, handshake.name = "fork6"} : <i3>
    %19:8 = demux[%18#1] %dataOut_4 {handshake.bb = 1 : ui32, handshake.name = "demux0"} : <i3>, (!handshake.control<>) -> (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>)
    %20 = mux %17#3 [%19#0, %19#1, %19#2, %19#3, %19#4, %19#5, %19#6, %19#7] {handshake.name = "mux1"} : <i3>, [<>, <>, <>, <>, <>, <>, <>, <>] to <>
    sink %20 {handshake.name = "sink1"} : <>
    %dataOut_6, %tagOut_7 = untagger %falseResult {handshake.bb = 1 : ui32, handshake.name = "untagger2"} : (!handshake.channel<i4, [tag1: i3]>) -> (!handshake.channel<i4>, !handshake.channel<i3>)
    %21:2 = fork [2] %tagOut_7 {handshake.bb = 1 : ui32, handshake.name = "fork7"} : <i3>
    %22:8 = demux[%21#1] %dataOut_6 {handshake.bb = 1 : ui32, handshake.name = "demux1"} : <i3>, (!handshake.channel<i4>) -> (!handshake.channel<i4>, !handshake.channel<i4>, !handshake.channel<i4>, !handshake.channel<i4>, !handshake.channel<i4>, !handshake.channel<i4>, !handshake.channel<i4>, !handshake.channel<i4>)
    %23 = mux %17#2 [%22#0, %22#1, %22#2, %22#3, %22#4, %22#5, %22#6, %22#7] {handshake.name = "mux4"} : <i3>, [<i4>, <i4>, <i4>, <i4>, <i4>, <i4>, <i4>, <i4>] to <i4>
    sink %23 {handshake.name = "sink2"} : <i4>
    %dataOut_8, %tagOut_9 = untagger %falseResult_1 {handshake.bb = 2 : ui32, handshake.name = "untagger3"} : (!handshake.channel<i32, [tag1: i3]>) -> (!handshake.channel<i32>, !handshake.channel<i3>)
    %24:2 = fork [2] %tagOut_9 {handshake.bb = 2 : ui32, handshake.name = "fork8"} : <i3>
    %25:8 = demux[%24#1] %dataOut_8 {handshake.bb = 2 : ui32, handshake.name = "demux2"} : <i3>, (!handshake.channel<i32>) -> (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>)
    %26 = mux %17#1 [%25#0, %25#1, %25#2, %25#3, %25#4, %25#5, %25#6, %25#7] {handshake.bb = 2 : ui32, handshake.name = "mux5"} : <i3>, [<i32>, <i32>, <i32>, <i32>, <i32>, <i32>, <i32>, <i32>] to <i32>
    %dataOut_10, %tagOut_11 = untagger %12 {handshake.bb = 2 : ui32, handshake.name = "untagger4"} : (!handshake.channel<i32, [tag1: i3]>) -> (!handshake.channel<i32>, !handshake.channel<i3>)
    %27:2 = fork [2] %tagOut_11 {handshake.bb = 2 : ui32, handshake.name = "fork9"} : <i3>
    %28:8 = demux[%27#1] %dataOut_10 {handshake.bb = 2 : ui32, handshake.name = "demux3"} : <i3>, (!handshake.channel<i32>) -> (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>)
    %29 = mux %17#0 [%28#0, %28#1, %28#2, %28#3, %28#4, %28#5, %28#6, %28#7] {handshake.bb = 2 : ui32, handshake.name = "mux6"} : <i3>, [<i32>, <i32>, <i32>, <i32>, <i32>, <i32>, <i32>, <i32>] to <i32>
    %30 = join %18#0, %21#0, %24#0, %27#0 {handshake.bb = 2 : ui32, handshake.name = "join0"} : <i3>
    %31 = mux %9#2 [%16, %trueResult_2] {ftd.regen, handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1, [tag1: i3]>, [<[tag1: i3]>, <[tag1: i3]>] to <[tag1: i3]>
    %32:3 = fork [3] %31 {handshake.bb = 1 : ui32, handshake.name = "fork10"} : <[tag1: i3]>
    %33 = mux %9#1 [%15, %trueResult_0] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux2"} : <i1, [tag1: i3]>, [<i32, [tag1: i3]>, <i32, [tag1: i3]>] to <i32, [tag1: i3]>
    %34 = mux %9#0 [%13, %trueResult] {handshake.bb = 1 : ui32, handshake.name = "mux3"} : <i1, [tag1: i3]>, [<i4, [tag1: i3]>, <i4, [tag1: i3]>] to <i4, [tag1: i3]>
    %35 = extsi %34 {handshake.bb = 1 : ui32, handshake.name = "extsi6"} : <i4, [tag1: i3]> to <i5, [tag1: i3]>
    %36 = constant %32#2 {handshake.bb = 1 : ui32, handshake.name = "constant7", value = 8 : i5} : <[tag1: i3]>, <i5, [tag1: i3]>
    %37 = constant %32#1 {handshake.bb = 1 : ui32, handshake.name = "constant8", value = 1 : i2} : <[tag1: i3]>, <i2, [tag1: i3]>
    %38 = extsi %37 {handshake.bb = 1 : ui32, handshake.name = "extsi7"} : <i2, [tag1: i3]> to <i5, [tag1: i3]>
    %39 = free_tags_fifo %51 {handshake.bb = 1 : ui32, handshake.name = "free_tags_fifo1"} : <i3>, <i3>
    %40:2 = fork [2] %39 {handshake.bb = 1 : ui32, handshake.name = "fork11"} : <i3>
    %41 = tagger[%40#1] %74 {handshake.bb = 1 : ui32, handshake.name = "tagger5"} : (!handshake.channel<i32, [tag1: i3]>, !handshake.channel<i3>) -> !handshake.channel<i32, [tag0: i3, tag1: i3]>
    %42:2 = fork [2] %41 {handshake.bb = 1 : ui32, handshake.name = "fork12"} : <i32, [tag0: i3, tag1: i3]>
    %43 = tagger[%40#0] %80#3 {handshake.bb = 1 : ui32, handshake.name = "tagger6"} : (!handshake.channel<i1, [tag1: i3]>, !handshake.channel<i3>) -> !handshake.channel<i1, [tag0: i3, tag1: i3]>
    %dataOut_12, %tagOut_13 = untagger %53#0 {handshake.bb = 1 : ui32, handshake.name = "untagger5"} : (!handshake.channel<i32, [tag0: i3, tag1: i3]>) -> (!handshake.channel<i32, [tag1: i3]>, !handshake.channel<i3>)
    %44:2 = fork [2] %tagOut_13 {handshake.bb = 1 : ui32, handshake.name = "fork13"} : <i3>
    sink %dataOut_12 {handshake.name = "sink3"} : <i32, [tag1: i3]>
    %dataOut_14, %tagOut_15 = untagger %43 {handshake.bb = 1 : ui32, handshake.name = "untagger6"} : (!handshake.channel<i1, [tag0: i3, tag1: i3]>) -> (!handshake.channel<i1, [tag1: i3]>, !handshake.channel<i3>)
    %45:2 = fork [2] %tagOut_15 {handshake.bb = 1 : ui32, handshake.name = "fork14"} : <i3>
    %46:8 = demux[%45#1] %dataOut_14 {handshake.bb = 1 : ui32, handshake.name = "demux4"} : <i3>, (!handshake.channel<i1, [tag1: i3]>) -> (!handshake.channel<i1, [tag1: i3]>, !handshake.channel<i1, [tag1: i3]>, !handshake.channel<i1, [tag1: i3]>, !handshake.channel<i1, [tag1: i3]>, !handshake.channel<i1, [tag1: i3]>, !handshake.channel<i1, [tag1: i3]>, !handshake.channel<i1, [tag1: i3]>, !handshake.channel<i1, [tag1: i3]>)
    %47 = mux %44#1 [%46#0, %46#1, %46#2, %46#3, %46#4, %46#5, %46#6, %46#7] {handshake.bb = 1 : ui32, handshake.name = "mux7"} : <i3>, [<i1, [tag1: i3]>, <i1, [tag1: i3]>, <i1, [tag1: i3]>, <i1, [tag1: i3]>, <i1, [tag1: i3]>, <i1, [tag1: i3]>, <i1, [tag1: i3]>, <i1, [tag1: i3]>] to <i1, [tag1: i3]>
    %dataOut_16, %tagOut_17 = untagger %53#1 {handshake.bb = 1 : ui32, handshake.name = "untagger7"} : (!handshake.channel<i32, [tag0: i3, tag1: i3]>) -> (!handshake.channel<i32, [tag1: i3]>, !handshake.channel<i3>)
    %48:2 = fork [2] %tagOut_17 {handshake.bb = 1 : ui32, handshake.name = "fork15"} : <i3>
    %49:8 = demux[%48#1] %dataOut_16 {handshake.bb = 1 : ui32, handshake.name = "demux5"} : <i3>, (!handshake.channel<i32, [tag1: i3]>) -> (!handshake.channel<i32, [tag1: i3]>, !handshake.channel<i32, [tag1: i3]>, !handshake.channel<i32, [tag1: i3]>, !handshake.channel<i32, [tag1: i3]>, !handshake.channel<i32, [tag1: i3]>, !handshake.channel<i32, [tag1: i3]>, !handshake.channel<i32, [tag1: i3]>, !handshake.channel<i32, [tag1: i3]>)
    %50 = mux %44#0 [%49#0, %49#1, %49#2, %49#3, %49#4, %49#5, %49#6, %49#7] {handshake.bb = 1 : ui32, handshake.name = "mux8"} : <i3>, [<i32, [tag1: i3]>, <i32, [tag1: i3]>, <i32, [tag1: i3]>, <i32, [tag1: i3]>, <i32, [tag1: i3]>, <i32, [tag1: i3]>, <i32, [tag1: i3]>, <i32, [tag1: i3]>] to <i32, [tag1: i3]>
    %51 = join %45#0, %48#0 {handshake.bb = 1 : ui32, handshake.name = "join1"} : <i3>
    %52 = muli %42#0, %42#1 {handshake.bb = 1 : ui32, handshake.name = "muli0"} : <i32, [tag0: i3, tag1: i3]>
    %53:2 = fork [2] %52 {handshake.bb = 1 : ui32, handshake.name = "fork16"} : <i32, [tag0: i3, tag1: i3]>
    %54 = free_tags_fifo %75 {handshake.bb = 1 : ui32, handshake.name = "free_tags_fifo2"} : <i2>, <i2>
    %55:5 = fork [5] %54 {handshake.bb = 1 : ui32, handshake.name = "fork17"} : <i2>
    %56 = tagger[%55#4] %38 {handshake.bb = 1 : ui32, handshake.name = "tagger7"} : (!handshake.channel<i5, [tag1: i3]>, !handshake.channel<i2>) -> !handshake.channel<i5, [tag1: i3, tag2: i2]>
    %57 = tagger[%55#3] %36 {handshake.bb = 1 : ui32, handshake.name = "tagger8"} : (!handshake.channel<i5, [tag1: i3]>, !handshake.channel<i2>) -> !handshake.channel<i5, [tag1: i3, tag2: i2]>
    %58 = tagger[%55#2] %35 {handshake.bb = 1 : ui32, handshake.name = "tagger9"} : (!handshake.channel<i5, [tag1: i3]>, !handshake.channel<i2>) -> !handshake.channel<i5, [tag1: i3, tag2: i2]>
    %59 = tagger[%55#1] %33 {handshake.bb = 1 : ui32, handshake.name = "tagger10"} : (!handshake.channel<i32, [tag1: i3]>, !handshake.channel<i2>) -> !handshake.channel<i32, [tag1: i3, tag2: i2]>
    %60 = tagger[%55#0] %32#0 {handshake.bb = 1 : ui32, handshake.name = "tagger11"} : (!handshake.control<[tag1: i3]>, !handshake.channel<i2>) -> !handshake.control<[tag1: i3, tag2: i2]>
    %dataOut_18, %tagOut_19 = untagger %77#0 {handshake.bb = 1 : ui32, handshake.name = "untagger8"} : (!handshake.channel<i5, [tag1: i3, tag2: i2]>) -> (!handshake.channel<i5, [tag1: i3]>, !handshake.channel<i2>)
    %61:4 = fork [4] %tagOut_19 {handshake.bb = 1 : ui32, handshake.name = "fork18"} : <i2>
    sink %dataOut_18 {handshake.name = "sink4"} : <i5, [tag1: i3]>
    %dataOut_20, %tagOut_21 = untagger %57 {handshake.bb = 1 : ui32, handshake.name = "untagger9"} : (!handshake.channel<i5, [tag1: i3, tag2: i2]>) -> (!handshake.channel<i5, [tag1: i3]>, !handshake.channel<i2>)
    %62:2 = fork [2] %tagOut_21 {handshake.bb = 1 : ui32, handshake.name = "fork19"} : <i2>
    %63:4 = demux[%62#1] %dataOut_20 {handshake.bb = 1 : ui32, handshake.name = "demux6"} : <i2>, (!handshake.channel<i5, [tag1: i3]>) -> (!handshake.channel<i5, [tag1: i3]>, !handshake.channel<i5, [tag1: i3]>, !handshake.channel<i5, [tag1: i3]>, !handshake.channel<i5, [tag1: i3]>)
    %64 = mux %61#3 [%63#0, %63#1, %63#2, %63#3] {handshake.bb = 1 : ui32, handshake.name = "mux9"} : <i2>, [<i5, [tag1: i3]>, <i5, [tag1: i3]>, <i5, [tag1: i3]>, <i5, [tag1: i3]>] to <i5, [tag1: i3]>
    %dataOut_22, %tagOut_23 = untagger %60 {handshake.bb = 1 : ui32, handshake.name = "untagger10"} : (!handshake.control<[tag1: i3, tag2: i2]>) -> (!handshake.control<[tag1: i3]>, !handshake.channel<i2>)
    %65:2 = fork [2] %tagOut_23 {handshake.bb = 1 : ui32, handshake.name = "fork20"} : <i2>
    %66:4 = demux[%65#1] %dataOut_22 {handshake.bb = 1 : ui32, handshake.name = "demux7"} : <i2>, (!handshake.control<[tag1: i3]>) -> (!handshake.control<[tag1: i3]>, !handshake.control<[tag1: i3]>, !handshake.control<[tag1: i3]>, !handshake.control<[tag1: i3]>)
    %67 = mux %61#2 [%66#0, %66#1, %66#2, %66#3] {handshake.bb = 1 : ui32, handshake.name = "mux10"} : <i2>, [<[tag1: i3]>, <[tag1: i3]>, <[tag1: i3]>, <[tag1: i3]>] to <[tag1: i3]>
    %dataOut_24, %tagOut_25 = untagger %77#1 {handshake.bb = 1 : ui32, handshake.name = "untagger11"} : (!handshake.channel<i5, [tag1: i3, tag2: i2]>) -> (!handshake.channel<i5, [tag1: i3]>, !handshake.channel<i2>)
    %68:2 = fork [2] %tagOut_25 {handshake.bb = 1 : ui32, handshake.name = "fork21"} : <i2>
    %69:4 = demux[%68#1] %dataOut_24 {handshake.bb = 1 : ui32, handshake.name = "demux8"} : <i2>, (!handshake.channel<i5, [tag1: i3]>) -> (!handshake.channel<i5, [tag1: i3]>, !handshake.channel<i5, [tag1: i3]>, !handshake.channel<i5, [tag1: i3]>, !handshake.channel<i5, [tag1: i3]>)
    %70 = mux %61#1 [%69#0, %69#1, %69#2, %69#3] {handshake.bb = 1 : ui32, handshake.name = "mux11"} : <i2>, [<i5, [tag1: i3]>, <i5, [tag1: i3]>, <i5, [tag1: i3]>, <i5, [tag1: i3]>] to <i5, [tag1: i3]>
    %71:2 = fork [2] %70 {handshake.bb = 1 : ui32, handshake.name = "fork22"} : <i5, [tag1: i3]>
    %dataOut_26, %tagOut_27 = untagger %59 {handshake.bb = 1 : ui32, handshake.name = "untagger12"} : (!handshake.channel<i32, [tag1: i3, tag2: i2]>) -> (!handshake.channel<i32, [tag1: i3]>, !handshake.channel<i2>)
    %72:2 = fork [2] %tagOut_27 {handshake.bb = 1 : ui32, handshake.name = "fork23"} : <i2>
    %73:4 = demux[%72#1] %dataOut_26 {handshake.bb = 1 : ui32, handshake.name = "demux9"} : <i2>, (!handshake.channel<i32, [tag1: i3]>) -> (!handshake.channel<i32, [tag1: i3]>, !handshake.channel<i32, [tag1: i3]>, !handshake.channel<i32, [tag1: i3]>, !handshake.channel<i32, [tag1: i3]>)
    %74 = mux %61#0 [%73#0, %73#1, %73#2, %73#3] {handshake.bb = 1 : ui32, handshake.name = "mux12"} : <i2>, [<i32, [tag1: i3]>, <i32, [tag1: i3]>, <i32, [tag1: i3]>, <i32, [tag1: i3]>] to <i32, [tag1: i3]>
    %75 = join %62#0, %65#0, %68#0, %72#0 {handshake.bb = 1 : ui32, handshake.name = "join2"} : <i2>
    %76 = addi %58, %56 {handshake.bb = 1 : ui32, handshake.name = "addi1"} : <i5, [tag1: i3, tag2: i2]>
    %77:2 = fork [2] %76 {handshake.bb = 1 : ui32, handshake.name = "fork24"} : <i5, [tag1: i3, tag2: i2]>
    %78 = trunci %71#1 {handshake.bb = 1 : ui32, handshake.name = "trunci0"} : <i5, [tag1: i3]> to <i4, [tag1: i3]>
    %79 = cmpi ult, %71#0, %64 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i5, [tag1: i3]>
    %80:4 = fork [4] %79 {handshake.bb = 1 : ui32, handshake.name = "fork25"} : <i1, [tag1: i3]>
    %addressResult, %dataResult = load[%7] %outputs {handshake.bb = 2 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i3>, <i32>, <i3>, <i32>
    %81 = addi %26, %29 {handshake.bb = 2 : ui32, handshake.name = "addi0"} : <i32>
    end {handshake.bb = 2 : ui32, handshake.name = "end0"} %81, %memEnd, %0#0 : <i32>, <>, <>
  }
}

