module {
  handshake.func @matrix_power(%arg0: memref<400xi32>, %arg1: memref<20xi32>, %arg2: memref<20xi32>, %arg3: memref<20xi32>, %arg4: !handshake.control<>, %arg5: !handshake.control<>, %arg6: !handshake.control<>, %arg7: !handshake.control<>, %arg8: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["mat", "row", "col", "a", "mat_start", "row_start", "col_start", "a_start", "start"], cfg.edges = "[0,1][2,2,3,cmpi0][1,2][3,1,4,cmpi1]", resNames = ["mat_end", "row_end", "col_end", "a_end", "end"]} {
    %0:5 = fork [5] %arg8 {handshake.bb = 0 : ui32, handshake.name = "fork0"} : <>
    %outputs, %memEnd = mem_controller[%arg3 : memref<20xi32>] %arg7 (%addressResult_8) %143#3 {connectedBlocks = [2 : i32], handshake.name = "mem_controller3"} :    (!handshake.channel<i5>) -> !handshake.channel<i32>
    %outputs_0, %memEnd_1 = mem_controller[%arg2 : memref<20xi32>] %arg6 (%addressResult_10) %143#2 {connectedBlocks = [2 : i32], handshake.name = "mem_controller4"} :    (!handshake.channel<i5>) -> !handshake.channel<i32>
    %outputs_2, %memEnd_3 = mem_controller[%arg1 : memref<20xi32>] %arg5 (%addressResult) %143#1 {connectedBlocks = [2 : i32], handshake.name = "mem_controller5"} :    (!handshake.channel<i5>) -> !handshake.channel<i32>
    %outputs_4:3, %memEnd_5 = mem_controller[%arg0 : memref<400xi32>] %arg4 (%59, %addressResult_12, %addressResult_14, %addressResult_16, %dataResult_17) %143#0 {connectedBlocks = [2 : i32], handshake.name = "mem_controller6"} :    (!handshake.channel<i32>, !handshake.channel<i9>, !handshake.channel<i9>, !handshake.channel<i9>, !handshake.channel<i32>) -> (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>)
    %1 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant1", value = 1 : i2} : <>, <i2>
    %2 = br %1 {handshake.bb = 0 : ui32, handshake.name = "br2"} : <i2>
    %3 = extsi %2 {handshake.bb = 0 : ui32, handshake.name = "extsi10"} : <i2> to <i6>
    %4 = br %0#4 {handshake.bb = 0 : ui32, handshake.name = "br3"} : <>
    %5 = mux %11#0 [%0#3, %125#0] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux4"} : <i1>, [<>, <>] to <>
    %7 = mux %11#1 [%0#2, %125#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux5"} : <i1>, [<>, <>] to <>
    %9 = init %140#2 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "init0"} : <i1>
    %11:2 = fork [2] %9 {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <i1>
    %12 = mux %index [%3, %trueResult_30] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i6>, <i6>] to <i6>
    %13:2 = fork [2] %12 {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <i6>
    %14 = extsi %13#1 {handshake.bb = 1 : ui32, handshake.name = "extsi11"} : <i6> to <i32>
    %result, %index = control_merge [%4, %trueResult_32]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %16:2 = fork [2] %result {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <>
    %17 = constant %16#0 {handshake.bb = 1 : ui32, handshake.name = "constant2", value = false} : <>, <i1>
    %18 = source {handshake.bb = 1 : ui32, handshake.name = "source0"} : <>
    %19 = constant %18 {handshake.bb = 1 : ui32, handshake.name = "constant7", value = -1 : i32} : <>, <i32>
    %20 = addi %14, %19 {handshake.bb = 1 : ui32, handshake.name = "addi0"} : <i32>
    %21 = br %17 {handshake.bb = 1 : ui32, handshake.name = "br4"} : <i1>
    %22 = extsi %21 {handshake.bb = 1 : ui32, handshake.name = "extsi9"} : <i1> to <i6>
    %23 = br %13#0 {handshake.bb = 1 : ui32, handshake.name = "br5"} : <i6>
    %25 = br %20 {handshake.bb = 1 : ui32, handshake.name = "br6"} : <i32>
    %26 = br %16#1 {handshake.bb = 1 : ui32, handshake.name = "br7"} : <>
    %trueResult, %falseResult = cond_br %117#3, %110 {handshake.bb = 2 : ui32, handshake.name = "cond_br12"} : <i1>, <>
    %28:2 = fork [2] %trueResult {handshake.bb = 2 : ui32, handshake.name = "fork4"} : <>
    %29 = mux %35#0 [%5, %28#1] {ftd.phi, handshake.bb = 2 : ui32, handshake.name = "mux6"} : <i1>, [<>, <>] to <>
    %31 = mux %35#1 [%7, %28#0] {ftd.phi, handshake.bb = 2 : ui32, handshake.name = "mux7"} : <i1>, [<>, <>] to <>
    %33 = init %117#2 {ftd.imerge, handshake.bb = 2 : ui32, handshake.name = "init2"} : <i1>
    %35:2 = fork [2] %33 {handshake.bb = 2 : ui32, handshake.name = "fork5"} : <i1>
    %36 = mux %56#1 [%22, %trueResult_18] {handshake.bb = 2 : ui32, handshake.name = "mux1"} : <i1>, [<i6>, <i6>] to <i6>
    %38:4 = fork [4] %36 {handshake.bb = 2 : ui32, handshake.name = "fork6"} : <i6>
    %39 = extsi %38#3 {handshake.bb = 2 : ui32, handshake.name = "extsi12"} : <i6> to <i7>
    %41 = trunci %38#0 {handshake.bb = 2 : ui32, handshake.name = "trunci0"} : <i6> to <i5>
    %43 = trunci %38#1 {handshake.bb = 2 : ui32, handshake.name = "trunci1"} : <i6> to <i5>
    %45 = trunci %38#2 {handshake.bb = 2 : ui32, handshake.name = "trunci2"} : <i6> to <i5>
    %47 = mux %56#0 [%23, %trueResult_20] {handshake.bb = 2 : ui32, handshake.name = "mux2"} : <i1>, [<i6>, <i6>] to <i6>
    %49:2 = fork [2] %47 {handshake.bb = 2 : ui32, handshake.name = "fork7"} : <i6>
    %50 = extsi %49#1 {handshake.bb = 2 : ui32, handshake.name = "extsi13"} : <i6> to <i32>
    %52:4 = fork [4] %50 {handshake.bb = 2 : ui32, handshake.name = "fork8"} : <i32>
    %53 = mux %56#2 [%25, %trueResult_22] {handshake.bb = 2 : ui32, handshake.name = "mux3"} : <i1>, [<i32>, <i32>] to <i32>
    %55:3 = fork [3] %53 {handshake.bb = 2 : ui32, handshake.name = "fork9"} : <i32>
    %result_6, %index_7 = control_merge [%26, %trueResult_24]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>, <>] to <>, <i1>
    %56:3 = fork [3] %index_7 {handshake.bb = 2 : ui32, handshake.name = "fork10"} : <i1>
    %57:2 = fork [2] %result_6 {handshake.bb = 2 : ui32, handshake.name = "fork11"} : <>
    %58 = constant %57#0 {handshake.bb = 2 : ui32, handshake.name = "constant4", value = 1 : i2} : <>, <i2>
    %59 = extsi %58 {handshake.bb = 2 : ui32, handshake.name = "extsi2"} : <i2> to <i32>
    %60 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %61 = constant %60 {handshake.bb = 2 : ui32, handshake.name = "constant5", value = 1 : i2} : <>, <i2>
    %62 = extsi %61 {handshake.bb = 2 : ui32, handshake.name = "extsi14"} : <i2> to <i7>
    %63 = source {handshake.bb = 2 : ui32, handshake.name = "source2"} : <>
    %64 = constant %63 {handshake.bb = 2 : ui32, handshake.name = "constant14", value = 20 : i6} : <>, <i6>
    %65 = extsi %64 {handshake.bb = 2 : ui32, handshake.name = "extsi15"} : <i6> to <i7>
    %66 = source {handshake.bb = 2 : ui32, handshake.name = "source3"} : <>
    %67 = constant %66 {handshake.bb = 2 : ui32, handshake.name = "constant15", value = 4 : i4} : <>, <i4>
    %68 = extsi %67 {handshake.bb = 2 : ui32, handshake.name = "extsi5"} : <i4> to <i32>
    %69:3 = fork [3] %68 {handshake.bb = 2 : ui32, handshake.name = "fork12"} : <i32>
    %70 = source {handshake.bb = 2 : ui32, handshake.name = "source4"} : <>
    %71 = constant %70 {handshake.bb = 2 : ui32, handshake.name = "constant16", value = 2 : i3} : <>, <i3>
    %72 = extsi %71 {handshake.bb = 2 : ui32, handshake.name = "extsi6"} : <i3> to <i32>
    %73:3 = fork [3] %72 {handshake.bb = 2 : ui32, handshake.name = "fork13"} : <i32>
    %addressResult, %dataResult = load[%45] %outputs_2 {handshake.bb = 2 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i5>, <i32>, <i5>, <i32>
    %74:2 = fork [2] %dataResult {handshake.bb = 2 : ui32, handshake.name = "fork14"} : <i32>
    %75 = trunci %74#0 {handshake.bb = 2 : ui32, handshake.name = "trunci3"} : <i32> to <i9>
    %addressResult_8, %dataResult_9 = load[%43] %outputs {handshake.bb = 2 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i5>, <i32>, <i5>, <i32>
    %addressResult_10, %dataResult_11 = load[%41] %outputs_0 {handshake.bb = 2 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load2"} : <i5>, <i32>, <i5>, <i32>
    %77 = shli %55#2, %73#0 {handshake.bb = 2 : ui32, handshake.name = "shli0"} : <i32>
    %80 = shli %55#1, %69#0 {handshake.bb = 2 : ui32, handshake.name = "shli1"} : <i32>
    %83 = addi %77, %80 {handshake.bb = 2 : ui32, handshake.name = "addi7"} : <i32>
    %84 = addi %dataResult_11, %83 {handshake.bb = 2 : ui32, handshake.name = "addi2"} : <i32>
    %85 = gate %84, %29 {handshake.bb = 2 : ui32, handshake.name = "gate0"} : <i32>, !handshake.control<> to <i32>
    %86 = trunci %85 {handshake.bb = 2 : ui32, handshake.name = "trunci4"} : <i32> to <i9>
    %addressResult_12, %dataResult_13 = load[%86] %outputs_4#0 {handshake.bb = 2 : ui32, handshake.deps = #handshake<deps[["store0", 0, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load3"} : <i9>, <i32>, <i9>, <i32>
    %87 = muli %dataResult_9, %dataResult_13 {handshake.bb = 2 : ui32, handshake.name = "muli0"} : <i32>
    %88 = shli %52#0, %73#1 {handshake.bb = 2 : ui32, handshake.name = "shli2"} : <i32>
    %91 = shli %52#1, %69#1 {handshake.bb = 2 : ui32, handshake.name = "shli3"} : <i32>
    %94 = addi %88, %91 {handshake.bb = 2 : ui32, handshake.name = "addi8"} : <i32>
    %95 = addi %74#1, %94 {handshake.bb = 2 : ui32, handshake.name = "addi3"} : <i32>
    %97 = gate %95, %31 {handshake.bb = 2 : ui32, handshake.name = "gate1"} : <i32>, !handshake.control<> to <i32>
    %98 = trunci %97 {handshake.bb = 2 : ui32, handshake.name = "trunci5"} : <i32> to <i9>
    %addressResult_14, %dataResult_15 = load[%98] %outputs_4#1 {handshake.bb = 2 : ui32, handshake.deps = #handshake<deps[["store0", 0, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load4"} : <i9>, <i32>, <i9>, <i32>
    %99 = addi %dataResult_15, %87 {handshake.bb = 2 : ui32, handshake.name = "addi1"} : <i32>
    %100 = shli %52#2, %73#2 {handshake.bb = 2 : ui32, handshake.name = "shli4"} : <i32>
    %103 = trunci %100 {handshake.bb = 2 : ui32, handshake.name = "trunci6"} : <i32> to <i9>
    %104 = shli %52#3, %69#2 {handshake.bb = 2 : ui32, handshake.name = "shli5"} : <i32>
    %107 = trunci %104 {handshake.bb = 2 : ui32, handshake.name = "trunci7"} : <i32> to <i9>
    %108 = addi %103, %107 {handshake.bb = 2 : ui32, handshake.name = "addi9"} : <i9>
    %109 = addi %75, %108 {handshake.bb = 2 : ui32, handshake.name = "addi4"} : <i9>
    %110 = buffer %doneResult, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer0"} : <>
    %addressResult_16, %dataResult_17, %doneResult = store[%109] %99 %outputs_4#2 {handshake.bb = 2 : ui32, handshake.deps = #handshake<deps[["load3", 0, false], ["load4", 0, false], ["store0", 0, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store0"} : <i9>, <i32>, <>, <i9>, <i32>, <>
    %111 = addi %39, %62 {handshake.bb = 2 : ui32, handshake.name = "addi5"} : <i7>
    %112:2 = fork [2] %111 {handshake.bb = 2 : ui32, handshake.name = "fork15"} : <i7>
    %113 = trunci %112#0 {handshake.bb = 2 : ui32, handshake.name = "trunci8"} : <i7> to <i6>
    %115 = cmpi ult, %112#1, %65 {handshake.bb = 2 : ui32, handshake.name = "cmpi0"} : <i7>
    %117:6 = fork [6] %115 {handshake.bb = 2 : ui32, handshake.name = "fork16"} : <i1>
    %trueResult_18, %falseResult_19 = cond_br %117#0, %113 {handshake.bb = 2 : ui32, handshake.name = "cond_br2"} : <i1>, <i6>
    sink %falseResult_19 {handshake.name = "sink0"} : <i6>
    %trueResult_20, %falseResult_21 = cond_br %117#1, %49#0 {handshake.bb = 2 : ui32, handshake.name = "cond_br3"} : <i1>, <i6>
    %trueResult_22, %falseResult_23 = cond_br %117#4, %55#0 {handshake.bb = 2 : ui32, handshake.name = "cond_br4"} : <i1>, <i32>
    sink %falseResult_23 {handshake.name = "sink1"} : <i32>
    %trueResult_24, %falseResult_25 = cond_br %117#5, %57#1 {handshake.bb = 2 : ui32, handshake.name = "cond_br5"} : <i1>, <>
    %trueResult_26, %falseResult_27 = cond_br %140#1, %falseResult {handshake.bb = 3 : ui32, handshake.name = "cond_br13"} : <i1>, <>
    sink %falseResult_27 {handshake.name = "sink2"} : <>
    %125:2 = fork [2] %trueResult_26 {handshake.bb = 3 : ui32, handshake.name = "fork17"} : <>
    %126 = merge %falseResult_21 {handshake.bb = 3 : ui32, handshake.name = "merge0"} : <i6>
    %127 = extsi %126 {handshake.bb = 3 : ui32, handshake.name = "extsi16"} : <i6> to <i7>
    %result_28, %index_29 = control_merge [%falseResult_25]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>] to <>, <i1>
    sink %index_29 {handshake.name = "sink3"} : <i1>
    %128 = source {handshake.bb = 3 : ui32, handshake.name = "source5"} : <>
    %129 = constant %128 {handshake.bb = 3 : ui32, handshake.name = "constant17", value = 1 : i2} : <>, <i2>
    %130 = extsi %129 {handshake.bb = 3 : ui32, handshake.name = "extsi17"} : <i2> to <i7>
    %131 = source {handshake.bb = 3 : ui32, handshake.name = "source6"} : <>
    %132 = constant %131 {handshake.bb = 3 : ui32, handshake.name = "constant18", value = 20 : i6} : <>, <i6>
    %133 = extsi %132 {handshake.bb = 3 : ui32, handshake.name = "extsi18"} : <i6> to <i7>
    %134 = addi %127, %130 {handshake.bb = 3 : ui32, handshake.name = "addi6"} : <i7>
    %135:2 = fork [2] %134 {handshake.bb = 3 : ui32, handshake.name = "fork18"} : <i7>
    %136 = trunci %135#0 {handshake.bb = 3 : ui32, handshake.name = "trunci9"} : <i7> to <i6>
    %138 = cmpi ult, %135#1, %133 {handshake.bb = 3 : ui32, handshake.name = "cmpi1"} : <i7>
    %140:4 = fork [4] %138 {handshake.bb = 3 : ui32, handshake.name = "fork19"} : <i1>
    %trueResult_30, %falseResult_31 = cond_br %140#0, %136 {handshake.bb = 3 : ui32, handshake.name = "cond_br6"} : <i1>, <i6>
    sink %falseResult_31 {handshake.name = "sink4"} : <i6>
    %trueResult_32, %falseResult_33 = cond_br %140#3, %result_28 {handshake.bb = 3 : ui32, handshake.name = "cond_br7"} : <i1>, <>
    %result_34, %index_35 = control_merge [%falseResult_33]  {handshake.bb = 4 : ui32, handshake.name = "control_merge3"} : [<>] to <>, <i1>
    sink %index_35 {handshake.name = "sink5"} : <i1>
    %143:4 = fork [4] %result_34 {handshake.bb = 4 : ui32, handshake.name = "fork20"} : <>
    end {handshake.bb = 4 : ui32, handshake.name = "end0"} %memEnd_5, %memEnd_3, %memEnd_1, %memEnd, %0#1 : <>, <>, <>, <>, <>
  }
}

