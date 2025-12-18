module {
  handshake.func @matrix_power(%arg0: memref<400xi32>, %arg1: memref<20xi32>, %arg2: memref<20xi32>, %arg3: memref<20xi32>, %arg4: !handshake.control<>, %arg5: !handshake.control<>, %arg6: !handshake.control<>, %arg7: !handshake.control<>, %arg8: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["mat", "row", "col", "a", "mat_start", "row_start", "col_start", "a_start", "start"], resNames = ["mat_end", "row_end", "col_end", "a_end", "end"]} {
    %0:3 = fork [3] %arg8 {handshake.bb = 0 : ui32, handshake.name = "fork0"} : <>
    %outputs, %memEnd = mem_controller[%arg3 : memref<20xi32>] %arg7 (%addressResult_6) %125#3 {connectedBlocks = [2 : i32], handshake.name = "mem_controller3"} :    (!handshake.channel<i5>) -> !handshake.channel<i32>
    %outputs_0, %memEnd_1 = mem_controller[%arg2 : memref<20xi32>] %arg6 (%addressResult_8) %125#2 {connectedBlocks = [2 : i32], handshake.name = "mem_controller4"} :    (!handshake.channel<i5>) -> !handshake.channel<i32>
    %outputs_2, %memEnd_3 = mem_controller[%arg1 : memref<20xi32>] %arg5 (%addressResult) %125#1 {connectedBlocks = [2 : i32], handshake.name = "mem_controller5"} :    (!handshake.channel<i5>) -> !handshake.channel<i32>
    %1:3 = lsq[%arg0 : memref<400xi32>] (%arg4, %42#0, %addressResult_10, %addressResult_12, %addressResult_14, %dataResult_15, %125#0)  {groupSizes = [3 : i32], handshake.name = "lsq1"} : (!handshake.control<>, !handshake.control<>, !handshake.channel<i9>, !handshake.channel<i9>, !handshake.channel<i9>, !handshake.channel<i32>, !handshake.control<>) -> (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>)
    %2 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant0", value = 1 : i2} : <>, <i2>
    %3 = br %2 {handshake.bb = 0 : ui32, handshake.name = "br2"} : <i2>
    %4 = extsi %3 {handshake.bb = 0 : ui32, handshake.name = "extsi9"} : <i2> to <i6>
    %5 = br %0#2 {handshake.bb = 0 : ui32, handshake.name = "br3"} : <>
    %6 = mux %index [%4, %trueResult_24] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i6>, <i6>] to <i6>
    %7:2 = fork [2] %6 {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <i6>
    %8 = extsi %7#1 {handshake.bb = 1 : ui32, handshake.name = "extsi10"} : <i6> to <i32>
    %result, %index = control_merge [%5, %trueResult_26]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %10:2 = fork [2] %result {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <>
    %11 = constant %10#0 {handshake.bb = 1 : ui32, handshake.name = "constant1", value = false} : <>, <i1>
    %12 = source {handshake.bb = 1 : ui32, handshake.name = "source0"} : <>
    %13 = constant %12 {handshake.bb = 1 : ui32, handshake.name = "constant7", value = -1 : i32} : <>, <i32>
    %14 = addi %8, %13 {handshake.bb = 1 : ui32, handshake.name = "addi0"} : <i32>
    %15 = br %11 {handshake.bb = 1 : ui32, handshake.name = "br4"} : <i1>
    %16 = extsi %15 {handshake.bb = 1 : ui32, handshake.name = "extsi8"} : <i1> to <i6>
    %17 = br %7#0 {handshake.bb = 1 : ui32, handshake.name = "br5"} : <i6>
    %19 = br %14 {handshake.bb = 1 : ui32, handshake.name = "br6"} : <i32>
    %20 = br %10#1 {handshake.bb = 1 : ui32, handshake.name = "br7"} : <>
    %21 = mux %41#1 [%16, %trueResult] {handshake.bb = 2 : ui32, handshake.name = "mux1"} : <i1>, [<i6>, <i6>] to <i6>
    %23:4 = fork [4] %21 {handshake.bb = 2 : ui32, handshake.name = "fork3"} : <i6>
    %24 = extsi %23#3 {handshake.bb = 2 : ui32, handshake.name = "extsi11"} : <i6> to <i7>
    %26 = trunci %23#0 {handshake.bb = 2 : ui32, handshake.name = "trunci0"} : <i6> to <i5>
    %28 = trunci %23#1 {handshake.bb = 2 : ui32, handshake.name = "trunci1"} : <i6> to <i5>
    %30 = trunci %23#2 {handshake.bb = 2 : ui32, handshake.name = "trunci2"} : <i6> to <i5>
    %32 = mux %41#0 [%17, %trueResult_16] {handshake.bb = 2 : ui32, handshake.name = "mux2"} : <i1>, [<i6>, <i6>] to <i6>
    %34:2 = fork [2] %32 {handshake.bb = 2 : ui32, handshake.name = "fork4"} : <i6>
    %35 = extsi %34#1 {handshake.bb = 2 : ui32, handshake.name = "extsi12"} : <i6> to <i32>
    %37:4 = fork [4] %35 {handshake.bb = 2 : ui32, handshake.name = "fork5"} : <i32>
    %38 = mux %41#2 [%19, %trueResult_18] {handshake.bb = 2 : ui32, handshake.name = "mux3"} : <i1>, [<i32>, <i32>] to <i32>
    %40:3 = fork [3] %38 {handshake.bb = 2 : ui32, handshake.name = "fork6"} : <i32>
    %result_4, %index_5 = control_merge [%20, %trueResult_20]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>, <>] to <>, <i1>
    %41:3 = fork [3] %index_5 {handshake.bb = 2 : ui32, handshake.name = "fork7"} : <i1>
    %42:2 = fork [2] %result_4 {handshake.bb = 2 : ui32, handshake.name = "fork8"} : <>
    %43 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %44 = constant %43 {handshake.bb = 2 : ui32, handshake.name = "constant2", value = 1 : i2} : <>, <i2>
    %45 = extsi %44 {handshake.bb = 2 : ui32, handshake.name = "extsi13"} : <i2> to <i7>
    %46 = source {handshake.bb = 2 : ui32, handshake.name = "source2"} : <>
    %47 = constant %46 {handshake.bb = 2 : ui32, handshake.name = "constant4", value = 20 : i6} : <>, <i6>
    %48 = extsi %47 {handshake.bb = 2 : ui32, handshake.name = "extsi14"} : <i6> to <i7>
    %49 = source {handshake.bb = 2 : ui32, handshake.name = "source3"} : <>
    %50 = constant %49 {handshake.bb = 2 : ui32, handshake.name = "constant5", value = 4 : i4} : <>, <i4>
    %51 = extsi %50 {handshake.bb = 2 : ui32, handshake.name = "extsi4"} : <i4> to <i32>
    %52:3 = fork [3] %51 {handshake.bb = 2 : ui32, handshake.name = "fork9"} : <i32>
    %53 = source {handshake.bb = 2 : ui32, handshake.name = "source4"} : <>
    %54 = constant %53 {handshake.bb = 2 : ui32, handshake.name = "constant14", value = 2 : i3} : <>, <i3>
    %55 = extsi %54 {handshake.bb = 2 : ui32, handshake.name = "extsi5"} : <i3> to <i32>
    %56:3 = fork [3] %55 {handshake.bb = 2 : ui32, handshake.name = "fork10"} : <i32>
    %addressResult, %dataResult = load[%30] %outputs_2 {handshake.bb = 2 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i5>, <i32>, <i5>, <i32>
    %57:2 = fork [2] %dataResult {handshake.bb = 2 : ui32, handshake.name = "fork11"} : <i32>
    %58 = trunci %57#0 {handshake.bb = 2 : ui32, handshake.name = "trunci3"} : <i32> to <i9>
    %60 = trunci %57#1 {handshake.bb = 2 : ui32, handshake.name = "trunci4"} : <i32> to <i9>
    %addressResult_6, %dataResult_7 = load[%28] %outputs {handshake.bb = 2 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i5>, <i32>, <i5>, <i32>
    %addressResult_8, %dataResult_9 = load[%26] %outputs_0 {handshake.bb = 2 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load2"} : <i5>, <i32>, <i5>, <i32>
    %62 = trunci %dataResult_9 {handshake.bb = 2 : ui32, handshake.name = "trunci5"} : <i32> to <i9>
    %63 = shli %40#2, %56#0 {handshake.bb = 2 : ui32, handshake.name = "shli0"} : <i32>
    %66 = trunci %63 {handshake.bb = 2 : ui32, handshake.name = "trunci6"} : <i32> to <i9>
    %67 = shli %40#1, %52#0 {handshake.bb = 2 : ui32, handshake.name = "shli1"} : <i32>
    %70 = trunci %67 {handshake.bb = 2 : ui32, handshake.name = "trunci7"} : <i32> to <i9>
    %71 = addi %66, %70 {handshake.bb = 2 : ui32, handshake.name = "addi7"} : <i9>
    %72 = addi %62, %71 {handshake.bb = 2 : ui32, handshake.name = "addi2"} : <i9>
    %addressResult_10, %dataResult_11 = load[%72] %1#0 {handshake.bb = 2 : ui32, handshake.deps = #handshake<deps[["store0", 0]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 0>, handshake.name = "load3"} : <i9>, <i32>, <i9>, <i32>
    %73 = muli %dataResult_7, %dataResult_11 {handshake.bb = 2 : ui32, handshake.name = "muli0"} : <i32>
    %74 = shli %37#0, %56#1 {handshake.bb = 2 : ui32, handshake.name = "shli2"} : <i32>
    %77 = trunci %74 {handshake.bb = 2 : ui32, handshake.name = "trunci8"} : <i32> to <i9>
    %78 = shli %37#1, %52#1 {handshake.bb = 2 : ui32, handshake.name = "shli3"} : <i32>
    %81 = trunci %78 {handshake.bb = 2 : ui32, handshake.name = "trunci9"} : <i32> to <i9>
    %82 = addi %77, %81 {handshake.bb = 2 : ui32, handshake.name = "addi8"} : <i9>
    %83 = addi %58, %82 {handshake.bb = 2 : ui32, handshake.name = "addi3"} : <i9>
    %addressResult_12, %dataResult_13 = load[%83] %1#1 {handshake.bb = 2 : ui32, handshake.deps = #handshake<deps[["store0", 0]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 0>, handshake.name = "load4"} : <i9>, <i32>, <i9>, <i32>
    %84 = addi %dataResult_13, %73 {handshake.bb = 2 : ui32, handshake.name = "addi1"} : <i32>
    %85 = shli %37#2, %56#2 {handshake.bb = 2 : ui32, handshake.name = "shli4"} : <i32>
    %88 = trunci %85 {handshake.bb = 2 : ui32, handshake.name = "trunci10"} : <i32> to <i9>
    %89 = shli %37#3, %52#2 {handshake.bb = 2 : ui32, handshake.name = "shli5"} : <i32>
    %92 = trunci %89 {handshake.bb = 2 : ui32, handshake.name = "trunci11"} : <i32> to <i9>
    %93 = addi %88, %92 {handshake.bb = 2 : ui32, handshake.name = "addi9"} : <i9>
    %94 = addi %60, %93 {handshake.bb = 2 : ui32, handshake.name = "addi4"} : <i9>
    %addressResult_14, %dataResult_15 = store[%94] %84 {handshake.bb = 2 : ui32, handshake.deps = #handshake<deps[["load3", 0], ["load4", 0], ["store0", 0]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 0>, handshake.name = "store0"} : <i9>, <i32>, <i9>, <i32>
    %95 = addi %24, %45 {handshake.bb = 2 : ui32, handshake.name = "addi5"} : <i7>
    %96:2 = fork [2] %95 {handshake.bb = 2 : ui32, handshake.name = "fork12"} : <i7>
    %97 = trunci %96#0 {handshake.bb = 2 : ui32, handshake.name = "trunci12"} : <i7> to <i6>
    %99 = cmpi ult, %96#1, %48 {handshake.bb = 2 : ui32, handshake.name = "cmpi0"} : <i7>
    %101:4 = fork [4] %99 {handshake.bb = 2 : ui32, handshake.name = "fork13"} : <i1>
    %trueResult, %falseResult = cond_br %101#0, %97 {handshake.bb = 2 : ui32, handshake.name = "cond_br2"} : <i1>, <i6>
    sink %falseResult {handshake.name = "sink0"} : <i6>
    %trueResult_16, %falseResult_17 = cond_br %101#1, %34#0 {handshake.bb = 2 : ui32, handshake.name = "cond_br3"} : <i1>, <i6>
    %trueResult_18, %falseResult_19 = cond_br %101#2, %40#0 {handshake.bb = 2 : ui32, handshake.name = "cond_br4"} : <i1>, <i32>
    sink %falseResult_19 {handshake.name = "sink1"} : <i32>
    %trueResult_20, %falseResult_21 = cond_br %101#3, %42#1 {handshake.bb = 2 : ui32, handshake.name = "cond_br5"} : <i1>, <>
    %108 = merge %falseResult_17 {handshake.bb = 3 : ui32, handshake.name = "merge0"} : <i6>
    %109 = extsi %108 {handshake.bb = 3 : ui32, handshake.name = "extsi15"} : <i6> to <i7>
    %result_22, %index_23 = control_merge [%falseResult_21]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>] to <>, <i1>
    sink %index_23 {handshake.name = "sink2"} : <i1>
    %110 = source {handshake.bb = 3 : ui32, handshake.name = "source5"} : <>
    %111 = constant %110 {handshake.bb = 3 : ui32, handshake.name = "constant15", value = 1 : i2} : <>, <i2>
    %112 = extsi %111 {handshake.bb = 3 : ui32, handshake.name = "extsi16"} : <i2> to <i7>
    %113 = source {handshake.bb = 3 : ui32, handshake.name = "source6"} : <>
    %114 = constant %113 {handshake.bb = 3 : ui32, handshake.name = "constant16", value = 20 : i6} : <>, <i6>
    %115 = extsi %114 {handshake.bb = 3 : ui32, handshake.name = "extsi17"} : <i6> to <i7>
    %116 = addi %109, %112 {handshake.bb = 3 : ui32, handshake.name = "addi6"} : <i7>
    %117:2 = fork [2] %116 {handshake.bb = 3 : ui32, handshake.name = "fork14"} : <i7>
    %118 = trunci %117#0 {handshake.bb = 3 : ui32, handshake.name = "trunci13"} : <i7> to <i6>
    %120 = cmpi ult, %117#1, %115 {handshake.bb = 3 : ui32, handshake.name = "cmpi1"} : <i7>
    %122:2 = fork [2] %120 {handshake.bb = 3 : ui32, handshake.name = "fork15"} : <i1>
    %trueResult_24, %falseResult_25 = cond_br %122#0, %118 {handshake.bb = 3 : ui32, handshake.name = "cond_br6"} : <i1>, <i6>
    sink %falseResult_25 {handshake.name = "sink3"} : <i6>
    %trueResult_26, %falseResult_27 = cond_br %122#1, %result_22 {handshake.bb = 3 : ui32, handshake.name = "cond_br7"} : <i1>, <>
    %result_28, %index_29 = control_merge [%falseResult_27]  {handshake.bb = 4 : ui32, handshake.name = "control_merge3"} : [<>] to <>, <i1>
    sink %index_29 {handshake.name = "sink4"} : <i1>
    %125:4 = fork [4] %result_28 {handshake.bb = 4 : ui32, handshake.name = "fork16"} : <>
    end {handshake.bb = 4 : ui32, handshake.name = "end0"} %1#2, %memEnd_3, %memEnd_1, %memEnd, %0#1 : <>, <>, <>, <>, <>
  }
}

