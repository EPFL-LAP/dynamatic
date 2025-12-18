module {
  handshake.func @mvt_float(%arg0: memref<900xf32>, %arg1: memref<30xf32>, %arg2: memref<30xf32>, %arg3: memref<30xf32>, %arg4: memref<30xf32>, %arg5: !handshake.control<>, %arg6: !handshake.control<>, %arg7: !handshake.control<>, %arg8: !handshake.control<>, %arg9: !handshake.control<>, %arg10: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["A", "x1", "x2", "y1", "y2", "A_start", "x1_start", "x2_start", "y1_start", "y2_start", "start"], cfg.edges = "[0,1][2,2,3,cmpi0][4,5][6,4,7,cmpi3][1,2][3,1,4,cmpi1][5,5,6,cmpi2]", resNames = ["A_end", "x1_end", "x2_end", "y1_end", "y2_end", "end"]} {
    %outputs, %memEnd = mem_controller[%arg4 : memref<30xf32>] %arg9 (%addressResult_38) %result_57 {connectedBlocks = [5 : i32], handshake.name = "mem_controller3"} :    (!handshake.channel<i32>) -> !handshake.channel<f32>
    %outputs_0, %memEnd_1 = mem_controller[%arg3 : memref<30xf32>] %arg8 (%addressResult_12) %result_57 {connectedBlocks = [2 : i32], handshake.name = "mem_controller4"} :    (!handshake.channel<i32>) -> !handshake.channel<f32>
    %outputs_2:2, %memEnd_3 = mem_controller[%arg2 : memref<30xf32>] %arg7 (%addressResult_32, %60, %addressResult_50, %dataResult_51) %result_57 {connectedBlocks = [4 : i32, 6 : i32], handshake.name = "mem_controller5"} :    (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<f32>) -> (!handshake.channel<f32>, !handshake.control<>)
    %outputs_4:2, %memEnd_5 = mem_controller[%arg1 : memref<30xf32>] %arg6 (%addressResult, %27, %addressResult_22, %dataResult_23) %result_57 {connectedBlocks = [1 : i32, 3 : i32], handshake.name = "mem_controller6"} :    (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<f32>) -> (!handshake.channel<f32>, !handshake.control<>)
    %outputs_6:2, %memEnd_7 = mem_controller[%arg0 : memref<900xf32>] %arg5 (%addressResult_10, %addressResult_36) %result_57 {connectedBlocks = [2 : i32, 5 : i32], handshake.name = "mem_controller7"} :    (!handshake.channel<i32>, !handshake.channel<i32>) -> (!handshake.channel<f32>, !handshake.channel<f32>)
    %0 = constant %arg10 {handshake.bb = 0 : ui32, handshake.name = "constant2", value = 0 : i32} : <>, <i32>
    %1 = br %0 {handshake.bb = 0 : ui32, handshake.name = "br2"} : <i32>
    %2 = br %arg10 {handshake.bb = 0 : ui32, handshake.name = "br4"} : <>
    %3 = init %35 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "init0"} : <i1>
    %4:2 = unbundle %dataResult  {handshake.bb = 1 : ui32, handshake.name = "unbundle0"} : <f32> to _ 
    %5 = mux %index [%1, %trueResult_24] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i32>, <i32>] to <i32>
    %result, %index = control_merge [%2, %trueResult_26]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %6 = constant %result {handshake.bb = 1 : ui32, handshake.name = "constant3", value = 0 : i32} : <>, <i32>
    %7 = buffer %4#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer0"} : <>
    %addressResult, %dataResult = load[%5] %outputs_4#0 {handshake.bb = 1 : ui32, handshake.deps = #handshake<deps[["store0", 2, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i32>, <f32>, <i32>, <f32>
    %8 = br %6 {handshake.bb = 1 : ui32, handshake.name = "br5"} : <i32>
    %9 = br %dataResult {handshake.bb = 1 : ui32, handshake.name = "br6"} : <f32>
    %10 = br %5 {handshake.bb = 1 : ui32, handshake.name = "br7"} : <i32>
    %11 = br %result {handshake.bb = 1 : ui32, handshake.name = "br8"} : <>
    %12 = mux %index_9 [%8, %trueResult] {handshake.bb = 2 : ui32, handshake.name = "mux1"} : <i1>, [<i32>, <i32>] to <i32>
    %13 = mux %index_9 [%9, %trueResult_14] {handshake.bb = 2 : ui32, handshake.name = "mux2"} : <i1>, [<f32>, <f32>] to <f32>
    %14 = mux %index_9 [%10, %trueResult_16] {handshake.bb = 2 : ui32, handshake.name = "mux3"} : <i1>, [<i32>, <i32>] to <i32>
    %result_8, %index_9 = control_merge [%11, %trueResult_18]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>, <>] to <>, <i1>
    %15 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %16 = constant %15 {handshake.bb = 2 : ui32, handshake.name = "constant4", value = 30 : i32} : <>, <i32>
    %17 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %18 = constant %17 {handshake.bb = 2 : ui32, handshake.name = "constant5", value = 1 : i32} : <>, <i32>
    %19 = muli %14, %16 {handshake.bb = 2 : ui32, handshake.name = "muli0"} : <i32>
    %20 = addi %12, %19 {handshake.bb = 2 : ui32, handshake.name = "addi0"} : <i32>
    %addressResult_10, %dataResult_11 = load[%20] %outputs_6#0 {handshake.bb = 2 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i32>, <f32>, <i32>, <f32>
    %addressResult_12, %dataResult_13 = load[%12] %outputs_0 {handshake.bb = 2 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load2"} : <i32>, <f32>, <i32>, <f32>
    %21 = mulf %dataResult_11, %dataResult_13 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "mulf0"} : <f32>
    %22 = addf %13, %21 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "addf0"} : <f32>
    %23 = addi %12, %18 {handshake.bb = 2 : ui32, handshake.name = "addi2"} : <i32>
    %24 = cmpi ult, %23, %16 {handshake.bb = 2 : ui32, handshake.name = "cmpi0"} : <i32>
    %trueResult, %falseResult = cond_br %24, %23 {handshake.bb = 2 : ui32, handshake.name = "cond_br4"} : <i1>, <i32>
    %trueResult_14, %falseResult_15 = cond_br %24, %22 {handshake.bb = 2 : ui32, handshake.name = "cond_br5"} : <i1>, <f32>
    %trueResult_16, %falseResult_17 = cond_br %24, %14 {handshake.bb = 2 : ui32, handshake.name = "cond_br6"} : <i1>, <i32>
    %trueResult_18, %falseResult_19 = cond_br %24, %result_8 {handshake.bb = 2 : ui32, handshake.name = "cond_br7"} : <i1>, <>
    %25 = merge %falseResult_17 {handshake.bb = 3 : ui32, handshake.name = "merge0"} : <i32>
    %26 = merge %falseResult_15 {handshake.bb = 3 : ui32, handshake.name = "merge1"} : <f32>
    %result_20, %index_21 = control_merge [%falseResult_19]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>] to <>, <i1>
    %27 = constant %result_20 {handshake.bb = 3 : ui32, handshake.name = "constant0", value = 1 : i32} : <>, <i32>
    %28 = constant %result_20 {handshake.bb = 3 : ui32, handshake.name = "constant6", value = 0 : i32} : <>, <i32>
    %29 = source {handshake.bb = 3 : ui32, handshake.name = "source2"} : <>
    %30 = constant %29 {handshake.bb = 3 : ui32, handshake.name = "constant7", value = 30 : i32} : <>, <i32>
    %31 = source {handshake.bb = 3 : ui32, handshake.name = "source3"} : <>
    %32 = constant %31 {handshake.bb = 3 : ui32, handshake.name = "constant8", value = 1 : i32} : <>, <i32>
    %33 = gate %25, %7 {handshake.bb = 3 : ui32, handshake.name = "gate0"} : <i32>, !handshake.control<> to <i32>
    %addressResult_22, %dataResult_23, %doneResult = store[%33] %26 %outputs_4#1 {handshake.bb = 3 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store0"} : <i32>, <f32>, <>, <i32>, <f32>, <>
    %34 = addi %25, %32 {handshake.bb = 3 : ui32, handshake.name = "addi3"} : <i32>
    %35 = cmpi ult, %34, %30 {handshake.bb = 3 : ui32, handshake.name = "cmpi1"} : <i32>
    %trueResult_24, %falseResult_25 = cond_br %35, %34 {handshake.bb = 3 : ui32, handshake.name = "cond_br8"} : <i1>, <i32>
    %trueResult_26, %falseResult_27 = cond_br %35, %result_20 {handshake.bb = 3 : ui32, handshake.name = "cond_br9"} : <i1>, <>
    %trueResult_28, %falseResult_29 = cond_br %35, %28 {handshake.bb = 3 : ui32, handshake.name = "cond_br10"} : <i1>, <i32>
    %36 = init %67 {ftd.imerge, handshake.bb = 4 : ui32, handshake.name = "init1"} : <i1>
    %37:2 = unbundle %dataResult_33  {handshake.bb = 4 : ui32, handshake.name = "unbundle2"} : <f32> to _ 
    %38 = mux %index_31 [%falseResult_29, %trueResult_53] {handshake.bb = 4 : ui32, handshake.name = "mux4"} : <i1>, [<i32>, <i32>] to <i32>
    %result_30, %index_31 = control_merge [%falseResult_27, %trueResult_55]  {handshake.bb = 4 : ui32, handshake.name = "control_merge3"} : [<>, <>] to <>, <i1>
    %39 = constant %result_30 {handshake.bb = 4 : ui32, handshake.name = "constant9", value = 0 : i32} : <>, <i32>
    %40 = buffer %37#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer1"} : <>
    %addressResult_32, %dataResult_33 = load[%38] %outputs_2#0 {handshake.bb = 4 : ui32, handshake.deps = #handshake<deps[["store1", 2, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load3"} : <i32>, <f32>, <i32>, <f32>
    %41 = br %39 {handshake.bb = 4 : ui32, handshake.name = "br9"} : <i32>
    %42 = br %dataResult_33 {handshake.bb = 4 : ui32, handshake.name = "br10"} : <f32>
    %43 = br %38 {handshake.bb = 4 : ui32, handshake.name = "br11"} : <i32>
    %44 = br %result_30 {handshake.bb = 4 : ui32, handshake.name = "br12"} : <>
    %45 = mux %index_35 [%41, %trueResult_40] {handshake.bb = 5 : ui32, handshake.name = "mux5"} : <i1>, [<i32>, <i32>] to <i32>
    %46 = mux %index_35 [%42, %trueResult_42] {handshake.bb = 5 : ui32, handshake.name = "mux6"} : <i1>, [<f32>, <f32>] to <f32>
    %47 = mux %index_35 [%43, %trueResult_44] {handshake.bb = 5 : ui32, handshake.name = "mux7"} : <i1>, [<i32>, <i32>] to <i32>
    %result_34, %index_35 = control_merge [%44, %trueResult_46]  {handshake.bb = 5 : ui32, handshake.name = "control_merge4"} : [<>, <>] to <>, <i1>
    %48 = source {handshake.bb = 5 : ui32, handshake.name = "source4"} : <>
    %49 = constant %48 {handshake.bb = 5 : ui32, handshake.name = "constant10", value = 30 : i32} : <>, <i32>
    %50 = source {handshake.bb = 5 : ui32, handshake.name = "source5"} : <>
    %51 = constant %50 {handshake.bb = 5 : ui32, handshake.name = "constant11", value = 1 : i32} : <>, <i32>
    %52 = muli %45, %49 {handshake.bb = 5 : ui32, handshake.name = "muli1"} : <i32>
    %53 = addi %47, %52 {handshake.bb = 5 : ui32, handshake.name = "addi1"} : <i32>
    %addressResult_36, %dataResult_37 = load[%53] %outputs_6#1 {handshake.bb = 5 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load4"} : <i32>, <f32>, <i32>, <f32>
    %addressResult_38, %dataResult_39 = load[%45] %outputs {handshake.bb = 5 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load5"} : <i32>, <f32>, <i32>, <f32>
    %54 = mulf %dataResult_37, %dataResult_39 {fastmath = #arith.fastmath<none>, handshake.bb = 5 : ui32, handshake.name = "mulf1"} : <f32>
    %55 = addf %46, %54 {fastmath = #arith.fastmath<none>, handshake.bb = 5 : ui32, handshake.name = "addf1"} : <f32>
    %56 = addi %45, %51 {handshake.bb = 5 : ui32, handshake.name = "addi4"} : <i32>
    %57 = cmpi ult, %56, %49 {handshake.bb = 5 : ui32, handshake.name = "cmpi2"} : <i32>
    %trueResult_40, %falseResult_41 = cond_br %57, %56 {handshake.bb = 5 : ui32, handshake.name = "cond_br11"} : <i1>, <i32>
    %trueResult_42, %falseResult_43 = cond_br %57, %55 {handshake.bb = 5 : ui32, handshake.name = "cond_br12"} : <i1>, <f32>
    %trueResult_44, %falseResult_45 = cond_br %57, %47 {handshake.bb = 5 : ui32, handshake.name = "cond_br13"} : <i1>, <i32>
    %trueResult_46, %falseResult_47 = cond_br %57, %result_34 {handshake.bb = 5 : ui32, handshake.name = "cond_br14"} : <i1>, <>
    %58 = merge %falseResult_45 {handshake.bb = 6 : ui32, handshake.name = "merge2"} : <i32>
    %59 = merge %falseResult_43 {handshake.bb = 6 : ui32, handshake.name = "merge3"} : <f32>
    %result_48, %index_49 = control_merge [%falseResult_47]  {handshake.bb = 6 : ui32, handshake.name = "control_merge5"} : [<>] to <>, <i1>
    %60 = constant %result_48 {handshake.bb = 6 : ui32, handshake.name = "constant1", value = 1 : i32} : <>, <i32>
    %61 = source {handshake.bb = 6 : ui32, handshake.name = "source6"} : <>
    %62 = constant %61 {handshake.bb = 6 : ui32, handshake.name = "constant12", value = 30 : i32} : <>, <i32>
    %63 = source {handshake.bb = 6 : ui32, handshake.name = "source7"} : <>
    %64 = constant %63 {handshake.bb = 6 : ui32, handshake.name = "constant13", value = 1 : i32} : <>, <i32>
    %65 = gate %58, %40 {handshake.bb = 6 : ui32, handshake.name = "gate1"} : <i32>, !handshake.control<> to <i32>
    %addressResult_50, %dataResult_51, %doneResult_52 = store[%65] %59 %outputs_2#1 {handshake.bb = 6 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store1"} : <i32>, <f32>, <>, <i32>, <f32>, <>
    %66 = addi %58, %64 {handshake.bb = 6 : ui32, handshake.name = "addi5"} : <i32>
    %67 = cmpi ult, %66, %62 {handshake.bb = 6 : ui32, handshake.name = "cmpi3"} : <i32>
    %trueResult_53, %falseResult_54 = cond_br %67, %66 {handshake.bb = 6 : ui32, handshake.name = "cond_br15"} : <i1>, <i32>
    %trueResult_55, %falseResult_56 = cond_br %67, %result_48 {handshake.bb = 6 : ui32, handshake.name = "cond_br16"} : <i1>, <>
    %result_57, %index_58 = control_merge [%falseResult_56]  {handshake.bb = 7 : ui32, handshake.name = "control_merge6"} : [<>] to <>, <i1>
    end {handshake.bb = 7 : ui32, handshake.name = "end0"} %memEnd_7, %memEnd_5, %memEnd_3, %memEnd_1, %memEnd, %arg10 : <>, <>, <>, <>, <>, <>
  }
}

