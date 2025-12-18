module {
  handshake.func @bicg(%arg0: memref<900xi32>, %arg1: memref<30xi32>, %arg2: memref<30xi32>, %arg3: memref<30xi32>, %arg4: memref<30xi32>, %arg5: !handshake.control<>, %arg6: !handshake.control<>, %arg7: !handshake.control<>, %arg8: !handshake.control<>, %arg9: !handshake.control<>, %arg10: !handshake.control<>, ...) -> (!handshake.channel<i32>, !handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["a", "s", "q", "p", "r", "a_start", "s_start", "q_start", "p_start", "r_start", "start"], cfg.edges = "[0,1][2,2,3,cmpi0][1,2][3,1,4,cmpi1]", resNames = ["out0", "a_end", "s_end", "q_end", "p_end", "r_end", "end"]} {
    %0:4 = fork [4] %arg10 {handshake.bb = 0 : ui32, handshake.name = "fork0"} : <>
    %outputs, %memEnd = mem_controller[%arg4 : memref<30xi32>] %arg9 (%addressResult_14) %123#4 {connectedBlocks = [2 : i32], handshake.name = "mem_controller3"} :    (!handshake.channel<i5>) -> !handshake.channel<i32>
    %outputs_0, %memEnd_1 = mem_controller[%arg3 : memref<30xi32>] %arg8 (%addressResult_18) %123#3 {connectedBlocks = [2 : i32], handshake.name = "mem_controller4"} :    (!handshake.channel<i5>) -> !handshake.channel<i32>
    %outputs_2:2, %memEnd_3 = mem_controller[%arg2 : memref<30xi32>] %arg7 (%addressResult, %101, %addressResult_32, %dataResult_33) %123#2 {connectedBlocks = [1 : i32, 3 : i32], handshake.name = "mem_controller5"} :    (!handshake.channel<i5>, !handshake.channel<i32>, !handshake.channel<i5>, !handshake.channel<i32>) -> (!handshake.channel<i32>, !handshake.control<>)
    %outputs_4:2, %memEnd_5 = mem_controller[%arg1 : memref<30xi32>] %arg6 (%54, %addressResult_12, %addressResult_16, %dataResult_17) %123#1 {connectedBlocks = [2 : i32], handshake.name = "mem_controller6"} :    (!handshake.channel<i32>, !handshake.channel<i5>, !handshake.channel<i5>, !handshake.channel<i32>) -> (!handshake.channel<i32>, !handshake.control<>)
    %outputs_6, %memEnd_7 = mem_controller[%arg0 : memref<900xi32>] %arg5 (%addressResult_10) %123#0 {connectedBlocks = [2 : i32], handshake.name = "mem_controller7"} :    (!handshake.channel<i10>) -> !handshake.channel<i32>
    %1 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant8", value = false} : <>, <i1>
    %2 = br %1 {handshake.bb = 0 : ui32, handshake.name = "br2"} : <i1>
    %3 = extsi %2 {handshake.bb = 0 : ui32, handshake.name = "extsi9"} : <i1> to <i6>
    %4 = br %0#3 {handshake.bb = 0 : ui32, handshake.name = "br3"} : <>
    %5 = mux %6 [%0#2, %trueResult_28] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux4"} : <i1>, [<>, <>] to <>
    %6 = init %117#2 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "init0"} : <i1>
    %8:2 = unbundle %17#0  {handshake.bb = 1 : ui32, handshake.name = "unbundle0"} : <i32> to _ 
    %10 = mux %index [%3, %trueResult_35] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i6>, <i6>] to <i6>
    %11:2 = fork [2] %10 {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <i6>
    %12 = trunci %11#0 {handshake.bb = 1 : ui32, handshake.name = "trunci0"} : <i6> to <i5>
    %result, %index = control_merge [%4, %trueResult_37]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %14:2 = fork [2] %result {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <>
    %15 = constant %14#0 {handshake.bb = 1 : ui32, handshake.name = "constant9", value = false} : <>, <i1>
    %16 = buffer %8#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer0"} : <>
    %addressResult, %dataResult = load[%12] %outputs_2#0 {handshake.bb = 1 : ui32, handshake.deps = #handshake<deps[["store1", 2, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i5>, <i32>, <i5>, <i32>
    %17:2 = fork [2] %dataResult {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <i32>
    %18 = br %15 {handshake.bb = 1 : ui32, handshake.name = "br4"} : <i1>
    %19 = extsi %18 {handshake.bb = 1 : ui32, handshake.name = "extsi8"} : <i1> to <i6>
    %20 = br %17#1 {handshake.bb = 1 : ui32, handshake.name = "br5"} : <i32>
    %22 = br %11#1 {handshake.bb = 1 : ui32, handshake.name = "br6"} : <i6>
    %24 = br %14#1 {handshake.bb = 1 : ui32, handshake.name = "br7"} : <>
    %trueResult, %falseResult = cond_br %25, %74 {handshake.bb = 2 : ui32, handshake.name = "cond_br11"} : <i1>, <>
    %25 = buffer %84#3, bufferType = FIFO_BREAK_NONE, numSlots = 4 {handshake.bb = 2 : ui32, handshake.name = "buffer7"} : <i1>
    %26 = mux %27 [%5, %trueResult] {ftd.phi, handshake.bb = 2 : ui32, handshake.name = "mux6"} : <i1>, [<>, <>] to <>
    %27 = init %28 {ftd.imerge, handshake.bb = 2 : ui32, handshake.name = "init2"} : <i1>
    %28 = buffer %84#2, bufferType = FIFO_BREAK_NONE, numSlots = 4 {handshake.bb = 2 : ui32, handshake.name = "buffer8"} : <i1>
    %29 = mux %51#1 [%19, %trueResult_20] {handshake.bb = 2 : ui32, handshake.name = "mux1"} : <i1>, [<i6>, <i6>] to <i6>
    %31:5 = fork [5] %29 {handshake.bb = 2 : ui32, handshake.name = "fork4"} : <i6>
    %32 = extsi %33 {handshake.bb = 2 : ui32, handshake.name = "extsi10"} : <i6> to <i10>
    %33 = buffer %31#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer10"} : <i6>
    %34 = extsi %31#3 {handshake.bb = 2 : ui32, handshake.name = "extsi11"} : <i6> to <i7>
    %36 = extsi %37 {handshake.bb = 2 : ui32, handshake.name = "extsi12"} : <i6> to <i32>
    %37 = buffer %31#4, bufferType = FIFO_BREAK_NONE, numSlots = 4 {handshake.bb = 2 : ui32, handshake.name = "buffer12"} : <i6>
    %38 = trunci %31#1 {handshake.bb = 2 : ui32, handshake.name = "trunci1"} : <i6> to <i5>
    %40 = trunci %41 {handshake.bb = 2 : ui32, handshake.name = "trunci2"} : <i6> to <i5>
    %41 = buffer %31#2, bufferType = FIFO_BREAK_NONE, numSlots = 3 {handshake.bb = 2 : ui32, handshake.name = "buffer14"} : <i6>
    %42 = mux %43 [%20, %trueResult_22] {handshake.bb = 2 : ui32, handshake.name = "mux2"} : <i1>, [<i32>, <i32>] to <i32>
    %43 = buffer %51#2, bufferType = FIFO_BREAK_NONE, numSlots = 4 {handshake.bb = 2 : ui32, handshake.name = "buffer15"} : <i1>
    %44 = mux %51#0 [%22, %trueResult_24] {handshake.bb = 2 : ui32, handshake.name = "mux3"} : <i1>, [<i6>, <i6>] to <i6>
    %46:3 = fork [3] %44 {handshake.bb = 2 : ui32, handshake.name = "fork5"} : <i6>
    %47 = extsi %46#2 {handshake.bb = 2 : ui32, handshake.name = "extsi13"} : <i6> to <i12>
    %49 = trunci %50 {handshake.bb = 2 : ui32, handshake.name = "trunci3"} : <i6> to <i5>
    %50 = buffer %46#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer18"} : <i6>
    %result_8, %index_9 = control_merge [%24, %trueResult_26]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>, <>] to <>, <i1>
    %51:3 = fork [3] %index_9 {handshake.bb = 2 : ui32, handshake.name = "fork6"} : <i1>
    %52:2 = fork [2] %result_8 {handshake.bb = 2 : ui32, handshake.name = "fork7"} : <>
    %53 = constant %52#0 {handshake.bb = 2 : ui32, handshake.name = "constant10", value = 1 : i2} : <>, <i2>
    %54 = extsi %53 {handshake.bb = 2 : ui32, handshake.name = "extsi2"} : <i2> to <i32>
    %55 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %56 = constant %55 {handshake.bb = 2 : ui32, handshake.name = "constant11", value = 30 : i6} : <>, <i6>
    %57:2 = fork [2] %56 {handshake.bb = 2 : ui32, handshake.name = "fork8"} : <i6>
    %58 = extsi %57#0 {handshake.bb = 2 : ui32, handshake.name = "extsi14"} : <i6> to <i12>
    %60 = extsi %61 {handshake.bb = 2 : ui32, handshake.name = "extsi15"} : <i6> to <i7>
    %61 = buffer %57#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer20"} : <i6>
    %62 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %63 = constant %62 {handshake.bb = 2 : ui32, handshake.name = "constant12", value = 1 : i2} : <>, <i2>
    %64 = extsi %63 {handshake.bb = 2 : ui32, handshake.name = "extsi16"} : <i2> to <i7>
    %65 = muli %47, %58 {handshake.bb = 2 : ui32, handshake.name = "muli2"} : <i12>
    %66 = trunci %65 {handshake.bb = 2 : ui32, handshake.name = "trunci4"} : <i12> to <i10>
    %67 = addi %32, %66 {handshake.bb = 2 : ui32, handshake.name = "addi2"} : <i10>
    %addressResult_10, %dataResult_11 = load[%67] %outputs_6 {handshake.bb = 2 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i10>, <i32>, <i10>, <i32>
    %68:2 = fork [2] %dataResult_11 {handshake.bb = 2 : ui32, handshake.name = "fork9"} : <i32>
    %69 = gate %36, %26 {handshake.bb = 2 : ui32, handshake.name = "gate0"} : <i32>, !handshake.control<> to <i32>
    %70 = trunci %69 {handshake.bb = 2 : ui32, handshake.name = "trunci5"} : <i32> to <i5>
    %addressResult_12, %dataResult_13 = load[%70] %outputs_4#0 {handshake.bb = 2 : ui32, handshake.deps = #handshake<deps[["store0", 1, false], ["store0", 3, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load2"} : <i5>, <i32>, <i5>, <i32>
    %addressResult_14, %dataResult_15 = load[%49] %outputs {handshake.bb = 2 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load3"} : <i5>, <i32>, <i5>, <i32>
    %71 = muli %dataResult_15, %68#0 {handshake.bb = 2 : ui32, handshake.name = "muli0"} : <i32>
    %73 = addi %dataResult_13, %71 {handshake.bb = 2 : ui32, handshake.name = "addi0"} : <i32>
    %74 = buffer %doneResult, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer1"} : <>
    %addressResult_16, %dataResult_17, %doneResult = store[%40] %73 %outputs_4#1 {handshake.bb = 2 : ui32, handshake.deps = #handshake<deps[["load2", 1, false], ["store0", 1, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store0"} : <i5>, <i32>, <>, <i5>, <i32>, <>
    %addressResult_18, %dataResult_19 = load[%38] %outputs_0 {handshake.bb = 2 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load4"} : <i5>, <i32>, <i5>, <i32>
    %75 = muli %68#1, %dataResult_19 {handshake.bb = 2 : ui32, handshake.name = "muli1"} : <i32>
    %77 = addi %42, %75 {handshake.bb = 2 : ui32, handshake.name = "addi1"} : <i32>
    %78 = addi %34, %64 {handshake.bb = 2 : ui32, handshake.name = "addi3"} : <i7>
    %79:2 = fork [2] %78 {handshake.bb = 2 : ui32, handshake.name = "fork10"} : <i7>
    %80 = trunci %79#0 {handshake.bb = 2 : ui32, handshake.name = "trunci6"} : <i7> to <i6>
    %82 = cmpi ult, %79#1, %60 {handshake.bb = 2 : ui32, handshake.name = "cmpi0"} : <i7>
    %84:6 = fork [6] %82 {handshake.bb = 2 : ui32, handshake.name = "fork11"} : <i1>
    %trueResult_20, %falseResult_21 = cond_br %84#0, %80 {handshake.bb = 2 : ui32, handshake.name = "cond_br2"} : <i1>, <i6>
    sink %falseResult_21 {handshake.name = "sink0"} : <i6>
    %trueResult_22, %falseResult_23 = cond_br %86, %77 {handshake.bb = 2 : ui32, handshake.name = "cond_br3"} : <i1>, <i32>
    %86 = buffer %84#4, bufferType = FIFO_BREAK_NONE, numSlots = 3 {handshake.bb = 2 : ui32, handshake.name = "buffer26"} : <i1>
    %trueResult_24, %falseResult_25 = cond_br %84#1, %46#1 {handshake.bb = 2 : ui32, handshake.name = "cond_br4"} : <i1>, <i6>
    %trueResult_26, %falseResult_27 = cond_br %84#5, %52#1 {handshake.bb = 2 : ui32, handshake.name = "cond_br5"} : <i1>, <>
    %trueResult_28, %falseResult_29 = cond_br %90, %falseResult {handshake.bb = 3 : ui32, handshake.name = "cond_br12"} : <i1>, <>
    %90 = buffer %117#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer30"} : <i1>
    sink %falseResult_29 {handshake.name = "sink1"} : <>
    %91 = merge %falseResult_25 {handshake.bb = 3 : ui32, handshake.name = "merge0"} : <i6>
    %92:2 = fork [2] %91 {handshake.bb = 3 : ui32, handshake.name = "fork12"} : <i6>
    %93 = extsi %92#0 {handshake.bb = 3 : ui32, handshake.name = "extsi17"} : <i6> to <i7>
    %95 = extsi %96 {handshake.bb = 3 : ui32, handshake.name = "extsi18"} : <i6> to <i32>
    %96 = buffer %92#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer32"} : <i6>
    %97 = merge %falseResult_23 {handshake.bb = 3 : ui32, handshake.name = "merge1"} : <i32>
    %98:2 = fork [2] %97 {handshake.bb = 3 : ui32, handshake.name = "fork13"} : <i32>
    %result_30, %index_31 = control_merge [%falseResult_27]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>] to <>, <i1>
    sink %index_31 {handshake.name = "sink2"} : <i1>
    %99:2 = fork [2] %result_30 {handshake.bb = 3 : ui32, handshake.name = "fork14"} : <>
    %100 = constant %99#0 {handshake.bb = 3 : ui32, handshake.name = "constant13", value = 1 : i2} : <>, <i2>
    %101 = extsi %100 {handshake.bb = 3 : ui32, handshake.name = "extsi5"} : <i2> to <i32>
    %102 = source {handshake.bb = 3 : ui32, handshake.name = "source2"} : <>
    %103 = constant %102 {handshake.bb = 3 : ui32, handshake.name = "constant14", value = 30 : i6} : <>, <i6>
    %104 = extsi %103 {handshake.bb = 3 : ui32, handshake.name = "extsi19"} : <i6> to <i7>
    %105 = source {handshake.bb = 3 : ui32, handshake.name = "source3"} : <>
    %106 = constant %105 {handshake.bb = 3 : ui32, handshake.name = "constant15", value = 1 : i2} : <>, <i2>
    %107 = extsi %106 {handshake.bb = 3 : ui32, handshake.name = "extsi20"} : <i2> to <i7>
    %108 = gate %95, %16 {handshake.bb = 3 : ui32, handshake.name = "gate1"} : <i32>, !handshake.control<> to <i32>
    %109 = trunci %108 {handshake.bb = 3 : ui32, handshake.name = "trunci7"} : <i32> to <i5>
    %addressResult_32, %dataResult_33, %doneResult_34 = store[%109] %98#0 %outputs_2#1 {handshake.bb = 3 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store1"} : <i5>, <i32>, <>, <i5>, <i32>, <>
    sink %doneResult_34 {handshake.name = "sink3"} : <>
    %111 = addi %93, %107 {handshake.bb = 3 : ui32, handshake.name = "addi4"} : <i7>
    %112:2 = fork [2] %111 {handshake.bb = 3 : ui32, handshake.name = "fork15"} : <i7>
    %113 = trunci %112#0 {handshake.bb = 3 : ui32, handshake.name = "trunci8"} : <i7> to <i6>
    %115 = cmpi ult, %112#1, %104 {handshake.bb = 3 : ui32, handshake.name = "cmpi1"} : <i7>
    %117:5 = fork [5] %115 {handshake.bb = 3 : ui32, handshake.name = "fork16"} : <i1>
    %trueResult_35, %falseResult_36 = cond_br %117#0, %113 {handshake.bb = 3 : ui32, handshake.name = "cond_br6"} : <i1>, <i6>
    sink %falseResult_36 {handshake.name = "sink4"} : <i6>
    %trueResult_37, %falseResult_38 = cond_br %117#3, %99#1 {handshake.bb = 3 : ui32, handshake.name = "cond_br7"} : <i1>, <>
    %trueResult_39, %falseResult_40 = cond_br %120, %98#1 {handshake.bb = 3 : ui32, handshake.name = "cond_br8"} : <i1>, <i32>
    %120 = buffer %117#4, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer38"} : <i1>
    sink %trueResult_39 {handshake.name = "sink5"} : <i32>
    %122 = merge %falseResult_40 {handshake.bb = 4 : ui32, handshake.name = "merge2"} : <i32>
    %result_41, %index_42 = control_merge [%falseResult_38]  {handshake.bb = 4 : ui32, handshake.name = "control_merge3"} : [<>] to <>, <i1>
    sink %index_42 {handshake.name = "sink6"} : <i1>
    %123:5 = fork [5] %result_41 {handshake.bb = 4 : ui32, handshake.name = "fork17"} : <>
    end {handshake.bb = 4 : ui32, handshake.name = "end0"} %122, %memEnd_7, %memEnd_5, %memEnd_3, %memEnd_1, %memEnd, %0#1 : <i32>, <>, <>, <>, <>, <>, <>
  }
}

