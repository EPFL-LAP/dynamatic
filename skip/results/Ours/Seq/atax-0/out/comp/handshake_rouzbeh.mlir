module {
  handshake.func @atax(%arg0: memref<400xf32>, %arg1: memref<20xf32>, %arg2: memref<20xf32>, %arg3: memref<20xf32>, %arg4: !handshake.control<>, %arg5: !handshake.control<>, %arg6: !handshake.control<>, %arg7: !handshake.control<>, %arg8: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["A", "x", "y", "tmp", "A_start", "x_start", "y_start", "tmp_start", "start"], cfg.edges = "[0,1][2,2,3,cmpi0][4,1,5,cmpi2][1,2][3,3,4,cmpi1]", resNames = ["A_end", "x_end", "y_end", "tmp_end", "end"]} {
    %outputs:2, %memEnd = mem_controller[%arg3 : memref<20xf32>] %arg7 (%addressResult, %59, %addressResult_42, %dataResult_43) %result_49 {connectedBlocks = [1 : i32, 4 : i32], handshake.name = "mem_controller2"} :    (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<f32>) -> (!handshake.channel<f32>, !handshake.control<>)
    %outputs_0:2, %memEnd_1 = mem_controller[%arg2 : memref<20xf32>] %arg6 (%38, %addressResult_24, %addressResult_28, %dataResult_29) %result_49 {connectedBlocks = [3 : i32], handshake.name = "mem_controller3"} :    (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<f32>) -> (!handshake.channel<f32>, !handshake.control<>)
    %outputs_2, %memEnd_3 = mem_controller[%arg1 : memref<20xf32>] %arg5 (%addressResult_10) %result_49 {connectedBlocks = [2 : i32], handshake.name = "mem_controller4"} :    (!handshake.channel<i32>) -> !handshake.channel<f32>
    %outputs_4:2, %memEnd_5 = mem_controller[%arg0 : memref<400xf32>] %arg4 (%addressResult_8, %addressResult_26) %result_49 {connectedBlocks = [2 : i32, 3 : i32], handshake.name = "mem_controller5"} :    (!handshake.channel<i32>, !handshake.channel<i32>) -> (!handshake.channel<f32>, !handshake.channel<f32>)
    %0 = constant %arg8 {handshake.bb = 0 : ui32, handshake.name = "constant4", value = 0 : i32} : <>, <i32>
    %1 = br %0 {handshake.bb = 0 : ui32, handshake.name = "br2"} : <i32>
    %2 = br %arg8 {handshake.bb = 0 : ui32, handshake.name = "br3"} : <>
    %3 = mux %4 [%arg8, %trueResult_38] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux8"} : <i1>, [<>, <>] to <>
    %4 = init %66 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "init0"} : <i1>
    %5:2 = unbundle %dataResult  {handshake.bb = 1 : ui32, handshake.name = "unbundle0"} : <f32> to _ 
    %6 = mux %index [%1, %trueResult_45] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i32>, <i32>] to <i32>
    %result, %index = control_merge [%2, %trueResult_47]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %7 = constant %result {handshake.bb = 1 : ui32, handshake.name = "constant5", value = 0 : i32} : <>, <i32>
    %8 = buffer %5#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer0"} : <>
    %addressResult, %dataResult = load[%6] %outputs#0 {handshake.bb = 1 : ui32, handshake.deps = #handshake<deps[["store1", 2, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i32>, <f32>, <i32>, <f32>
    %9 = br %7 {handshake.bb = 1 : ui32, handshake.name = "br4"} : <i32>
    %10 = br %dataResult {handshake.bb = 1 : ui32, handshake.name = "br5"} : <f32>
    %11 = br %6 {handshake.bb = 1 : ui32, handshake.name = "br6"} : <i32>
    %12 = br %result {handshake.bb = 1 : ui32, handshake.name = "br7"} : <>
    %13 = mux %index_7 [%9, %trueResult] {handshake.bb = 2 : ui32, handshake.name = "mux1"} : <i1>, [<i32>, <i32>] to <i32>
    %14 = mux %index_7 [%10, %trueResult_12] {handshake.bb = 2 : ui32, handshake.name = "mux2"} : <i1>, [<f32>, <f32>] to <f32>
    %15 = mux %index_7 [%11, %trueResult_14] {handshake.bb = 2 : ui32, handshake.name = "mux3"} : <i1>, [<i32>, <i32>] to <i32>
    %result_6, %index_7 = control_merge [%12, %trueResult_16]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>, <>] to <>, <i1>
    %16 = constant %result_6 {handshake.bb = 2 : ui32, handshake.name = "constant6", value = 0 : i32} : <>, <i32>
    %17 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %18 = constant %17 {handshake.bb = 2 : ui32, handshake.name = "constant7", value = 20 : i32} : <>, <i32>
    %19 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %20 = constant %19 {handshake.bb = 2 : ui32, handshake.name = "constant8", value = 1 : i32} : <>, <i32>
    %21 = source {handshake.bb = 2 : ui32, handshake.name = "source2"} : <>
    %22 = constant %21 {handshake.bb = 2 : ui32, handshake.name = "constant9", value = 4 : i32} : <>, <i32>
    %23 = source {handshake.bb = 2 : ui32, handshake.name = "source3"} : <>
    %24 = constant %23 {handshake.bb = 2 : ui32, handshake.name = "constant10", value = 2 : i32} : <>, <i32>
    %25 = shli %15, %24 {handshake.bb = 2 : ui32, handshake.name = "shli0"} : <i32>
    %26 = shli %15, %22 {handshake.bb = 2 : ui32, handshake.name = "shli1"} : <i32>
    %27 = addi %25, %26 {handshake.bb = 2 : ui32, handshake.name = "addi5"} : <i32>
    %28 = addi %13, %27 {handshake.bb = 2 : ui32, handshake.name = "addi0"} : <i32>
    %addressResult_8, %dataResult_9 = load[%28] %outputs_4#0 {handshake.bb = 2 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i32>, <f32>, <i32>, <f32>
    %addressResult_10, %dataResult_11 = load[%13] %outputs_2 {handshake.bb = 2 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load2"} : <i32>, <f32>, <i32>, <f32>
    %29 = mulf %dataResult_9, %dataResult_11 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "mulf0"} : <f32>
    %30 = addf %14, %29 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "addf0"} : <f32>
    %31 = addi %13, %20 {handshake.bb = 2 : ui32, handshake.name = "addi2"} : <i32>
    %32 = cmpi ult, %31, %18 {handshake.bb = 2 : ui32, handshake.name = "cmpi0"} : <i32>
    %trueResult, %falseResult = cond_br %32, %31 {handshake.bb = 2 : ui32, handshake.name = "cond_br3"} : <i1>, <i32>
    %trueResult_12, %falseResult_13 = cond_br %32, %30 {handshake.bb = 2 : ui32, handshake.name = "cond_br4"} : <i1>, <f32>
    %trueResult_14, %falseResult_15 = cond_br %32, %15 {handshake.bb = 2 : ui32, handshake.name = "cond_br5"} : <i1>, <i32>
    %trueResult_16, %falseResult_17 = cond_br %32, %result_6 {handshake.bb = 2 : ui32, handshake.name = "cond_br6"} : <i1>, <>
    %trueResult_18, %falseResult_19 = cond_br %32, %16 {handshake.bb = 2 : ui32, handshake.name = "cond_br7"} : <i1>, <i32>
    %trueResult_20, %falseResult_21 = cond_br %56, %54 {handshake.bb = 3 : ui32, handshake.name = "cond_br15"} : <i1>, <>
    %33 = mux %34 [%3, %trueResult_20] {ftd.phi, handshake.bb = 3 : ui32, handshake.name = "mux9"} : <i1>, [<>, <>] to <>
    %34 = init %56 {ftd.imerge, handshake.bb = 3 : ui32, handshake.name = "init2"} : <i1>
    %35 = mux %index_23 [%falseResult_19, %trueResult_30] {handshake.bb = 3 : ui32, handshake.name = "mux4"} : <i1>, [<i32>, <i32>] to <i32>
    %36 = mux %index_23 [%falseResult_15, %trueResult_32] {handshake.bb = 3 : ui32, handshake.name = "mux5"} : <i1>, [<i32>, <i32>] to <i32>
    %37 = mux %index_23 [%falseResult_13, %trueResult_34] {handshake.bb = 3 : ui32, handshake.name = "mux6"} : <i1>, [<f32>, <f32>] to <f32>
    %result_22, %index_23 = control_merge [%falseResult_17, %trueResult_36]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>, <>] to <>, <i1>
    %38 = constant %result_22 {handshake.bb = 3 : ui32, handshake.name = "constant0", value = 1 : i32} : <>, <i32>
    %39 = source {handshake.bb = 3 : ui32, handshake.name = "source4"} : <>
    %40 = constant %39 {handshake.bb = 3 : ui32, handshake.name = "constant11", value = 20 : i32} : <>, <i32>
    %41 = source {handshake.bb = 3 : ui32, handshake.name = "source5"} : <>
    %42 = constant %41 {handshake.bb = 3 : ui32, handshake.name = "constant12", value = 1 : i32} : <>, <i32>
    %43 = source {handshake.bb = 3 : ui32, handshake.name = "source6"} : <>
    %44 = constant %43 {handshake.bb = 3 : ui32, handshake.name = "constant13", value = 4 : i32} : <>, <i32>
    %45 = source {handshake.bb = 3 : ui32, handshake.name = "source7"} : <>
    %46 = constant %45 {handshake.bb = 3 : ui32, handshake.name = "constant14", value = 2 : i32} : <>, <i32>
    %47 = gate %35, %33 {handshake.bb = 3 : ui32, handshake.name = "gate0"} : <i32>, !handshake.control<> to <i32>
    %addressResult_24, %dataResult_25 = load[%47] %outputs_0#0 {handshake.bb = 3 : ui32, handshake.deps = #handshake<deps[["store0", 1, false], ["store0", 3, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load3"} : <i32>, <f32>, <i32>, <f32>
    %48 = shli %36, %46 {handshake.bb = 3 : ui32, handshake.name = "shli2"} : <i32>
    %49 = shli %36, %44 {handshake.bb = 3 : ui32, handshake.name = "shli3"} : <i32>
    %50 = addi %48, %49 {handshake.bb = 3 : ui32, handshake.name = "addi6"} : <i32>
    %51 = addi %35, %50 {handshake.bb = 3 : ui32, handshake.name = "addi1"} : <i32>
    %addressResult_26, %dataResult_27 = load[%51] %outputs_4#1 {handshake.bb = 3 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load4"} : <i32>, <f32>, <i32>, <f32>
    %52 = mulf %dataResult_27, %37 {fastmath = #arith.fastmath<none>, handshake.bb = 3 : ui32, handshake.name = "mulf1"} : <f32>
    %53 = addf %dataResult_25, %52 {fastmath = #arith.fastmath<none>, handshake.bb = 3 : ui32, handshake.name = "addf1"} : <f32>
    %54 = buffer %doneResult, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer1"} : <>
    %addressResult_28, %dataResult_29, %doneResult = store[%35] %53 %outputs_0#1 {handshake.bb = 3 : ui32, handshake.deps = #handshake<deps[["load3", 1, false], ["store0", 1, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store0"} : <i32>, <f32>, <>, <i32>, <f32>, <>
    %55 = addi %35, %42 {handshake.bb = 3 : ui32, handshake.name = "addi3"} : <i32>
    %56 = cmpi ult, %55, %40 {handshake.bb = 3 : ui32, handshake.name = "cmpi1"} : <i32>
    %trueResult_30, %falseResult_31 = cond_br %56, %55 {handshake.bb = 3 : ui32, handshake.name = "cond_br8"} : <i1>, <i32>
    %trueResult_32, %falseResult_33 = cond_br %56, %36 {handshake.bb = 3 : ui32, handshake.name = "cond_br9"} : <i1>, <i32>
    %trueResult_34, %falseResult_35 = cond_br %56, %37 {handshake.bb = 3 : ui32, handshake.name = "cond_br10"} : <i1>, <f32>
    %trueResult_36, %falseResult_37 = cond_br %56, %result_22 {handshake.bb = 3 : ui32, handshake.name = "cond_br11"} : <i1>, <>
    %trueResult_38, %falseResult_39 = cond_br %66, %falseResult_21 {handshake.bb = 4 : ui32, handshake.name = "cond_br16"} : <i1>, <>
    %57 = merge %falseResult_33 {handshake.bb = 4 : ui32, handshake.name = "merge0"} : <i32>
    %58 = merge %falseResult_35 {handshake.bb = 4 : ui32, handshake.name = "merge1"} : <f32>
    %result_40, %index_41 = control_merge [%falseResult_37]  {handshake.bb = 4 : ui32, handshake.name = "control_merge3"} : [<>] to <>, <i1>
    %59 = constant %result_40 {handshake.bb = 4 : ui32, handshake.name = "constant1", value = 1 : i32} : <>, <i32>
    %60 = source {handshake.bb = 4 : ui32, handshake.name = "source8"} : <>
    %61 = constant %60 {handshake.bb = 4 : ui32, handshake.name = "constant15", value = 20 : i32} : <>, <i32>
    %62 = source {handshake.bb = 4 : ui32, handshake.name = "source9"} : <>
    %63 = constant %62 {handshake.bb = 4 : ui32, handshake.name = "constant16", value = 1 : i32} : <>, <i32>
    %64 = gate %57, %8 {handshake.bb = 4 : ui32, handshake.name = "gate1"} : <i32>, !handshake.control<> to <i32>
    %addressResult_42, %dataResult_43, %doneResult_44 = store[%64] %58 %outputs#1 {handshake.bb = 4 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store1"} : <i32>, <f32>, <>, <i32>, <f32>, <>
    %65 = addi %57, %63 {handshake.bb = 4 : ui32, handshake.name = "addi4"} : <i32>
    %66 = cmpi ult, %65, %61 {handshake.bb = 4 : ui32, handshake.name = "cmpi2"} : <i32>
    %trueResult_45, %falseResult_46 = cond_br %66, %65 {handshake.bb = 4 : ui32, handshake.name = "cond_br12"} : <i1>, <i32>
    %trueResult_47, %falseResult_48 = cond_br %66, %result_40 {handshake.bb = 4 : ui32, handshake.name = "cond_br13"} : <i1>, <>
    %result_49, %index_50 = control_merge [%falseResult_48]  {handshake.bb = 5 : ui32, handshake.name = "control_merge4"} : [<>] to <>, <i1>
    end {handshake.bb = 5 : ui32, handshake.name = "end0"} %memEnd_5, %memEnd_3, %memEnd_1, %memEnd, %arg8 : <>, <>, <>, <>, <>
  }
}

