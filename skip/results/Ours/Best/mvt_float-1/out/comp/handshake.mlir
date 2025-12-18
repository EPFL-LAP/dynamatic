module {
  handshake.func @mvt_float(%arg0: memref<900xf32>, %arg1: memref<30xf32>, %arg2: memref<30xf32>, %arg3: memref<30xf32>, %arg4: memref<30xf32>, %arg5: !handshake.control<>, %arg6: !handshake.control<>, %arg7: !handshake.control<>, %arg8: !handshake.control<>, %arg9: !handshake.control<>, %arg10: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["A", "x1", "x2", "y1", "y2", "A_start", "x1_start", "x2_start", "y1_start", "y2_start", "start"], cfg.edges = "[0,1][2,2,3,cmpi0][4,5][6,4,7,cmpi3][1,2][3,1,4,cmpi1][5,5,6,cmpi2]", resNames = ["A_end", "x1_end", "x2_end", "y1_end", "y2_end", "end"]} {
    %outputs, %memEnd = mem_controller[%arg4 : memref<30xf32>] %arg9 (%addressResult_34) %result_53 {connectedBlocks = [5 : i32], handshake.name = "mem_controller0"} :    (!handshake.channel<i32>) -> !handshake.channel<f32>
    %outputs_0, %memEnd_1 = mem_controller[%arg3 : memref<30xf32>] %arg8 (%addressResult_8) %result_53 {connectedBlocks = [2 : i32], handshake.name = "mem_controller1"} :    (!handshake.channel<i32>) -> !handshake.channel<f32>
    %0:3 = lsq[%arg2 : memref<30xf32>] (%arg7, %result_26, %addressResult_28, %result_44, %addressResult_46, %dataResult_47, %result_53)  {groupSizes = [1 : i32, 1 : i32], handshake.name = "lsq0"} : (!handshake.control<>, !handshake.control<>, !handshake.channel<i32>, !handshake.control<>, !handshake.channel<i32>, !handshake.channel<f32>, !handshake.control<>) -> (!handshake.channel<f32>, !handshake.control<>, !handshake.control<>)
    %1:3 = lsq[%arg1 : memref<30xf32>] (%arg6, %result, %addressResult, %result_16, %addressResult_18, %dataResult_19, %result_53)  {groupSizes = [1 : i32, 1 : i32], handshake.name = "lsq1"} : (!handshake.control<>, !handshake.control<>, !handshake.channel<i32>, !handshake.control<>, !handshake.channel<i32>, !handshake.channel<f32>, !handshake.control<>) -> (!handshake.channel<f32>, !handshake.control<>, !handshake.control<>)
    %outputs_2:2, %memEnd_3 = mem_controller[%arg0 : memref<900xf32>] %arg5 (%addressResult_6, %addressResult_32) %result_53 {connectedBlocks = [2 : i32, 5 : i32], handshake.name = "mem_controller2"} :    (!handshake.channel<i32>, !handshake.channel<i32>) -> (!handshake.channel<f32>, !handshake.channel<f32>)
    %2 = constant %arg10 {handshake.bb = 0 : ui32, handshake.name = "constant2", value = 0 : i32} : <>, <i32>
    %3 = br %2 {handshake.bb = 0 : ui32, handshake.name = "br2"} : <i32>
    %4 = br %arg10 {handshake.bb = 0 : ui32, handshake.name = "br4"} : <>
    %5 = mux %index [%3, %trueResult_20] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i32>, <i32>] to <i32>
    %result, %index = control_merge [%4, %trueResult_22]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %6 = constant %result {handshake.bb = 1 : ui32, handshake.name = "constant3", value = 0 : i32} : <>, <i32>
    %addressResult, %dataResult = load[%5] %1#0 {handshake.bb = 1 : ui32, handshake.deps = #handshake<deps[["store0", 2, true]]>, handshake.name = "load0"} : <i32>, <f32>, <i32>, <f32>
    %7 = br %6 {handshake.bb = 1 : ui32, handshake.name = "br5"} : <i32>
    %8 = br %dataResult {handshake.bb = 1 : ui32, handshake.name = "br6"} : <f32>
    %9 = br %5 {handshake.bb = 1 : ui32, handshake.name = "br7"} : <i32>
    %10 = br %result {handshake.bb = 1 : ui32, handshake.name = "br8"} : <>
    %11 = mux %index_5 [%7, %trueResult] {handshake.bb = 2 : ui32, handshake.name = "mux1"} : <i1>, [<i32>, <i32>] to <i32>
    %12 = mux %index_5 [%8, %trueResult_10] {handshake.bb = 2 : ui32, handshake.name = "mux2"} : <i1>, [<f32>, <f32>] to <f32>
    %13 = mux %index_5 [%9, %trueResult_12] {handshake.bb = 2 : ui32, handshake.name = "mux3"} : <i1>, [<i32>, <i32>] to <i32>
    %result_4, %index_5 = control_merge [%10, %trueResult_14]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>, <>] to <>, <i1>
    %14 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %15 = constant %14 {handshake.bb = 2 : ui32, handshake.name = "constant4", value = 30 : i32} : <>, <i32>
    %16 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %17 = constant %16 {handshake.bb = 2 : ui32, handshake.name = "constant5", value = 1 : i32} : <>, <i32>
    %18 = muli %13, %15 {handshake.bb = 2 : ui32, handshake.name = "muli0"} : <i32>
    %19 = addi %11, %18 {handshake.bb = 2 : ui32, handshake.name = "addi0"} : <i32>
    %addressResult_6, %dataResult_7 = load[%19] %outputs_2#0 {handshake.bb = 2 : ui32, handshake.name = "load1"} : <i32>, <f32>, <i32>, <f32>
    %addressResult_8, %dataResult_9 = load[%11] %outputs_0 {handshake.bb = 2 : ui32, handshake.name = "load2"} : <i32>, <f32>, <i32>, <f32>
    %20 = mulf %dataResult_7, %dataResult_9 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "mulf0"} : <f32>
    %21 = addf %12, %20 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "addf0"} : <f32>
    %22 = addi %11, %17 {handshake.bb = 2 : ui32, handshake.name = "addi2"} : <i32>
    %23 = cmpi ult, %22, %15 {handshake.bb = 2 : ui32, handshake.name = "cmpi0"} : <i32>
    %trueResult, %falseResult = cond_br %23, %22 {handshake.bb = 2 : ui32, handshake.name = "cond_br4"} : <i1>, <i32>
    %trueResult_10, %falseResult_11 = cond_br %23, %21 {handshake.bb = 2 : ui32, handshake.name = "cond_br5"} : <i1>, <f32>
    %trueResult_12, %falseResult_13 = cond_br %23, %13 {handshake.bb = 2 : ui32, handshake.name = "cond_br6"} : <i1>, <i32>
    %trueResult_14, %falseResult_15 = cond_br %23, %result_4 {handshake.bb = 2 : ui32, handshake.name = "cond_br7"} : <i1>, <>
    %24 = merge %falseResult_13 {handshake.bb = 3 : ui32, handshake.name = "merge0"} : <i32>
    %25 = merge %falseResult_11 {handshake.bb = 3 : ui32, handshake.name = "merge1"} : <f32>
    %result_16, %index_17 = control_merge [%falseResult_15]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>] to <>, <i1>
    %26 = constant %result_16 {handshake.bb = 3 : ui32, handshake.name = "constant6", value = 0 : i32} : <>, <i32>
    %27 = source {handshake.bb = 3 : ui32, handshake.name = "source2"} : <>
    %28 = constant %27 {handshake.bb = 3 : ui32, handshake.name = "constant7", value = 30 : i32} : <>, <i32>
    %29 = source {handshake.bb = 3 : ui32, handshake.name = "source3"} : <>
    %30 = constant %29 {handshake.bb = 3 : ui32, handshake.name = "constant8", value = 1 : i32} : <>, <i32>
    %addressResult_18, %dataResult_19, %doneResult = store[%24] %25 %1#1 {handshake.bb = 3 : ui32, handshake.name = "store0"} : <i32>, <f32>, <>, <i32>, <f32>, <>
    %31 = addi %24, %30 {handshake.bb = 3 : ui32, handshake.name = "addi3"} : <i32>
    %32 = cmpi ult, %31, %28 {handshake.bb = 3 : ui32, handshake.name = "cmpi1"} : <i32>
    %trueResult_20, %falseResult_21 = cond_br %32, %31 {handshake.bb = 3 : ui32, handshake.name = "cond_br8"} : <i1>, <i32>
    %trueResult_22, %falseResult_23 = cond_br %32, %result_16 {handshake.bb = 3 : ui32, handshake.name = "cond_br9"} : <i1>, <>
    %trueResult_24, %falseResult_25 = cond_br %32, %26 {handshake.bb = 3 : ui32, handshake.name = "cond_br10"} : <i1>, <i32>
    %33 = mux %index_27 [%falseResult_25, %trueResult_49] {handshake.bb = 4 : ui32, handshake.name = "mux4"} : <i1>, [<i32>, <i32>] to <i32>
    %result_26, %index_27 = control_merge [%falseResult_23, %trueResult_51]  {handshake.bb = 4 : ui32, handshake.name = "control_merge3"} : [<>, <>] to <>, <i1>
    %34 = constant %result_26 {handshake.bb = 4 : ui32, handshake.name = "constant9", value = 0 : i32} : <>, <i32>
    %addressResult_28, %dataResult_29 = load[%33] %0#0 {handshake.bb = 4 : ui32, handshake.deps = #handshake<deps[["store1", 2, true]]>, handshake.name = "load3"} : <i32>, <f32>, <i32>, <f32>
    %35 = br %34 {handshake.bb = 4 : ui32, handshake.name = "br9"} : <i32>
    %36 = br %dataResult_29 {handshake.bb = 4 : ui32, handshake.name = "br10"} : <f32>
    %37 = br %33 {handshake.bb = 4 : ui32, handshake.name = "br11"} : <i32>
    %38 = br %result_26 {handshake.bb = 4 : ui32, handshake.name = "br12"} : <>
    %39 = mux %index_31 [%35, %trueResult_36] {handshake.bb = 5 : ui32, handshake.name = "mux5"} : <i1>, [<i32>, <i32>] to <i32>
    %40 = mux %index_31 [%36, %trueResult_38] {handshake.bb = 5 : ui32, handshake.name = "mux6"} : <i1>, [<f32>, <f32>] to <f32>
    %41 = mux %index_31 [%37, %trueResult_40] {handshake.bb = 5 : ui32, handshake.name = "mux7"} : <i1>, [<i32>, <i32>] to <i32>
    %result_30, %index_31 = control_merge [%38, %trueResult_42]  {handshake.bb = 5 : ui32, handshake.name = "control_merge4"} : [<>, <>] to <>, <i1>
    %42 = source {handshake.bb = 5 : ui32, handshake.name = "source4"} : <>
    %43 = constant %42 {handshake.bb = 5 : ui32, handshake.name = "constant10", value = 30 : i32} : <>, <i32>
    %44 = source {handshake.bb = 5 : ui32, handshake.name = "source5"} : <>
    %45 = constant %44 {handshake.bb = 5 : ui32, handshake.name = "constant11", value = 1 : i32} : <>, <i32>
    %46 = muli %39, %43 {handshake.bb = 5 : ui32, handshake.name = "muli1"} : <i32>
    %47 = addi %41, %46 {handshake.bb = 5 : ui32, handshake.name = "addi1"} : <i32>
    %addressResult_32, %dataResult_33 = load[%47] %outputs_2#1 {handshake.bb = 5 : ui32, handshake.name = "load4"} : <i32>, <f32>, <i32>, <f32>
    %addressResult_34, %dataResult_35 = load[%39] %outputs {handshake.bb = 5 : ui32, handshake.name = "load5"} : <i32>, <f32>, <i32>, <f32>
    %48 = mulf %dataResult_33, %dataResult_35 {fastmath = #arith.fastmath<none>, handshake.bb = 5 : ui32, handshake.name = "mulf1"} : <f32>
    %49 = addf %40, %48 {fastmath = #arith.fastmath<none>, handshake.bb = 5 : ui32, handshake.name = "addf1"} : <f32>
    %50 = addi %39, %45 {handshake.bb = 5 : ui32, handshake.name = "addi4"} : <i32>
    %51 = cmpi ult, %50, %43 {handshake.bb = 5 : ui32, handshake.name = "cmpi2"} : <i32>
    %trueResult_36, %falseResult_37 = cond_br %51, %50 {handshake.bb = 5 : ui32, handshake.name = "cond_br11"} : <i1>, <i32>
    %trueResult_38, %falseResult_39 = cond_br %51, %49 {handshake.bb = 5 : ui32, handshake.name = "cond_br12"} : <i1>, <f32>
    %trueResult_40, %falseResult_41 = cond_br %51, %41 {handshake.bb = 5 : ui32, handshake.name = "cond_br13"} : <i1>, <i32>
    %trueResult_42, %falseResult_43 = cond_br %51, %result_30 {handshake.bb = 5 : ui32, handshake.name = "cond_br14"} : <i1>, <>
    %52 = merge %falseResult_41 {handshake.bb = 6 : ui32, handshake.name = "merge2"} : <i32>
    %53 = merge %falseResult_39 {handshake.bb = 6 : ui32, handshake.name = "merge3"} : <f32>
    %result_44, %index_45 = control_merge [%falseResult_43]  {handshake.bb = 6 : ui32, handshake.name = "control_merge5"} : [<>] to <>, <i1>
    %54 = source {handshake.bb = 6 : ui32, handshake.name = "source6"} : <>
    %55 = constant %54 {handshake.bb = 6 : ui32, handshake.name = "constant12", value = 30 : i32} : <>, <i32>
    %56 = source {handshake.bb = 6 : ui32, handshake.name = "source7"} : <>
    %57 = constant %56 {handshake.bb = 6 : ui32, handshake.name = "constant13", value = 1 : i32} : <>, <i32>
    %addressResult_46, %dataResult_47, %doneResult_48 = store[%52] %53 %0#1 {handshake.bb = 6 : ui32, handshake.name = "store1"} : <i32>, <f32>, <>, <i32>, <f32>, <>
    %58 = addi %52, %57 {handshake.bb = 6 : ui32, handshake.name = "addi5"} : <i32>
    %59 = cmpi ult, %58, %55 {handshake.bb = 6 : ui32, handshake.name = "cmpi3"} : <i32>
    %trueResult_49, %falseResult_50 = cond_br %59, %58 {handshake.bb = 6 : ui32, handshake.name = "cond_br15"} : <i1>, <i32>
    %trueResult_51, %falseResult_52 = cond_br %59, %result_44 {handshake.bb = 6 : ui32, handshake.name = "cond_br16"} : <i1>, <>
    %result_53, %index_54 = control_merge [%falseResult_52]  {handshake.bb = 7 : ui32, handshake.name = "control_merge6"} : [<>] to <>, <i1>
    end {handshake.bb = 7 : ui32, handshake.name = "end0"} %memEnd_3, %1#2, %0#2, %memEnd_1, %memEnd, %arg10 : <>, <>, <>, <>, <>, <>
  }
}

