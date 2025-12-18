module {
  handshake.func @kernel_3mm(%arg0: memref<100xi32>, %arg1: memref<100xi32>, %arg2: memref<100xi32>, %arg3: memref<100xi32>, %arg4: memref<100xi32>, %arg5: memref<100xi32>, %arg6: memref<100xi32>, %arg7: !handshake.control<>, %arg8: !handshake.control<>, %arg9: !handshake.control<>, %arg10: !handshake.control<>, %arg11: !handshake.control<>, %arg12: !handshake.control<>, %arg13: !handshake.control<>, %arg14: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["A", "B", "C", "D", "E", "F", "G", "A_start", "B_start", "C_start", "D_start", "E_start", "F_start", "G_start", "start"], cfg.edges = "[0,1][7,8][14,12,15,cmpi7][2,3][9,7,10,cmpi4][4,2,5,cmpi1][11,12][6,7][13,13,14,cmpi6][1,2][8,8,9,cmpi3][15,11,16,cmpi8][3,3,4,cmpi0][10,6,11,cmpi5][5,1,6,cmpi2][12,13]", resNames = ["A_end", "B_end", "C_end", "D_end", "E_end", "F_end", "G_end", "end"]} {
    %0:4 = lsq[%arg6 : memref<100xi32>] (%arg13, %result_85, %addressResult_87, %dataResult_88, %result_90, %addressResult_96, %addressResult_98, %dataResult_99, %result_123)  {groupSizes = [1 : i32, 2 : i32], handshake.name = "lsq0"} : (!handshake.control<>, !handshake.control<>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>) -> (!handshake.control<>, !handshake.channel<i32>, !handshake.control<>, !handshake.control<>)
    %1:5 = lsq[%arg5 : memref<100xi32>] (%arg12, %result_43, %addressResult_45, %dataResult_46, %result_48, %addressResult_54, %addressResult_56, %dataResult_57, %result_90, %addressResult_94, %result_123)  {groupSizes = [1 : i32, 2 : i32, 1 : i32], handshake.name = "lsq1"} : (!handshake.control<>, !handshake.control<>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>, !handshake.channel<i32>, !handshake.control<>) -> (!handshake.control<>, !handshake.channel<i32>, !handshake.control<>, !handshake.channel<i32>, !handshake.control<>)
    %2:5 = lsq[%arg4 : memref<100xi32>] (%arg11, %result_6, %addressResult, %dataResult, %result_8, %addressResult_14, %addressResult_16, %dataResult_17, %result_90, %addressResult_92, %result_123)  {groupSizes = [1 : i32, 2 : i32, 1 : i32], handshake.name = "lsq2"} : (!handshake.control<>, !handshake.control<>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>, !handshake.channel<i32>, !handshake.control<>) -> (!handshake.control<>, !handshake.channel<i32>, !handshake.control<>, !handshake.channel<i32>, !handshake.control<>)
    %outputs, %memEnd = mem_controller[%arg3 : memref<100xi32>] %arg10 (%addressResult_52) %result_123 {connectedBlocks = [8 : i32], handshake.name = "mem_controller0"} :    (!handshake.channel<i32>) -> !handshake.channel<i32>
    %outputs_0, %memEnd_1 = mem_controller[%arg2 : memref<100xi32>] %arg9 (%addressResult_50) %result_123 {connectedBlocks = [8 : i32], handshake.name = "mem_controller1"} :    (!handshake.channel<i32>) -> !handshake.channel<i32>
    %outputs_2, %memEnd_3 = mem_controller[%arg1 : memref<100xi32>] %arg8 (%addressResult_12) %result_123 {connectedBlocks = [3 : i32], handshake.name = "mem_controller2"} :    (!handshake.channel<i32>) -> !handshake.channel<i32>
    %outputs_4, %memEnd_5 = mem_controller[%arg0 : memref<100xi32>] %arg7 (%addressResult_10) %result_123 {connectedBlocks = [3 : i32], handshake.name = "mem_controller3"} :    (!handshake.channel<i32>) -> !handshake.channel<i32>
    %3 = constant %arg14 {handshake.bb = 0 : ui32, handshake.name = "constant3", value = 0 : i32} : <>, <i32>
    %4 = br %3 {handshake.bb = 0 : ui32, handshake.name = "br3"} : <i32>
    %5 = br %arg14 {handshake.bb = 0 : ui32, handshake.name = "br6"} : <>
    %6 = mux %index [%4, %trueResult_35] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i32>, <i32>] to <i32>
    %result, %index = control_merge [%5, %trueResult_37]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %7 = constant %result {handshake.bb = 1 : ui32, handshake.name = "constant5", value = 0 : i32} : <>, <i32>
    %8 = br %7 {handshake.bb = 1 : ui32, handshake.name = "br9"} : <i32>
    %9 = br %6 {handshake.bb = 1 : ui32, handshake.name = "br10"} : <i32>
    %10 = br %result {handshake.bb = 1 : ui32, handshake.name = "br11"} : <>
    %11 = mux %index_7 [%8, %trueResult_27] {handshake.bb = 2 : ui32, handshake.name = "mux1"} : <i1>, [<i32>, <i32>] to <i32>
    %12 = mux %index_7 [%9, %trueResult_29] {handshake.bb = 2 : ui32, handshake.name = "mux2"} : <i1>, [<i32>, <i32>] to <i32>
    %result_6, %index_7 = control_merge [%10, %trueResult_31]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>, <>] to <>, <i1>
    %13 = constant %result_6 {handshake.bb = 2 : ui32, handshake.name = "constant6", value = 0 : i32} : <>, <i32>
    %14 = constant %result_6 {handshake.bb = 2 : ui32, handshake.name = "constant7", value = 0 : i32} : <>, <i32>
    %15 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %16 = constant %15 {handshake.bb = 2 : ui32, handshake.name = "constant8", value = 1 : i32} : <>, <i32>
    %17 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %18 = constant %17 {handshake.bb = 2 : ui32, handshake.name = "constant9", value = 3 : i32} : <>, <i32>
    %19 = shli %12, %16 {handshake.bb = 2 : ui32, handshake.name = "shli0"} : <i32>
    %20 = shli %12, %18 {handshake.bb = 2 : ui32, handshake.name = "shli1"} : <i32>
    %21 = addi %19, %20 {handshake.bb = 2 : ui32, handshake.name = "addi27"} : <i32>
    %22 = addi %11, %21 {handshake.bb = 2 : ui32, handshake.name = "addi3"} : <i32>
    %addressResult, %dataResult, %doneResult = store[%22] %13 %2#0 {handshake.bb = 2 : ui32, handshake.deps = #handshake<deps[["load2", 3, true], ["store1", 3, true], ["load6", 1, true]]>, handshake.name = "store0"} : <i32>, <i32>, <>, <i32>, <i32>, <>
    %23 = br %14 {handshake.bb = 2 : ui32, handshake.name = "br12"} : <i32>
    %24 = br %12 {handshake.bb = 2 : ui32, handshake.name = "br13"} : <i32>
    %25 = br %11 {handshake.bb = 2 : ui32, handshake.name = "br14"} : <i32>
    %26 = br %result_6 {handshake.bb = 2 : ui32, handshake.name = "br15"} : <>
    %27 = mux %index_9 [%23, %trueResult] {handshake.bb = 3 : ui32, handshake.name = "mux3"} : <i1>, [<i32>, <i32>] to <i32>
    %28 = mux %index_9 [%24, %trueResult_19] {handshake.bb = 3 : ui32, handshake.name = "mux4"} : <i1>, [<i32>, <i32>] to <i32>
    %29 = mux %index_9 [%25, %trueResult_21] {handshake.bb = 3 : ui32, handshake.name = "mux5"} : <i1>, [<i32>, <i32>] to <i32>
    %result_8, %index_9 = control_merge [%26, %trueResult_23]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>, <>] to <>, <i1>
    %30 = source {handshake.bb = 3 : ui32, handshake.name = "source2"} : <>
    %31 = constant %30 {handshake.bb = 3 : ui32, handshake.name = "constant10", value = 10 : i32} : <>, <i32>
    %32 = source {handshake.bb = 3 : ui32, handshake.name = "source3"} : <>
    %33 = constant %32 {handshake.bb = 3 : ui32, handshake.name = "constant11", value = 1 : i32} : <>, <i32>
    %34 = source {handshake.bb = 3 : ui32, handshake.name = "source4"} : <>
    %35 = constant %34 {handshake.bb = 3 : ui32, handshake.name = "constant12", value = 3 : i32} : <>, <i32>
    %36 = shli %28, %33 {handshake.bb = 3 : ui32, handshake.name = "shli2"} : <i32>
    %37 = shli %28, %35 {handshake.bb = 3 : ui32, handshake.name = "shli3"} : <i32>
    %38 = addi %36, %37 {handshake.bb = 3 : ui32, handshake.name = "addi28"} : <i32>
    %39 = addi %27, %38 {handshake.bb = 3 : ui32, handshake.name = "addi4"} : <i32>
    %addressResult_10, %dataResult_11 = load[%39] %outputs_4 {handshake.bb = 3 : ui32, handshake.name = "load0"} : <i32>, <i32>, <i32>, <i32>
    %40 = shli %27, %33 {handshake.bb = 3 : ui32, handshake.name = "shli4"} : <i32>
    %41 = shli %27, %35 {handshake.bb = 3 : ui32, handshake.name = "shli5"} : <i32>
    %42 = addi %40, %41 {handshake.bb = 3 : ui32, handshake.name = "addi29"} : <i32>
    %43 = addi %29, %42 {handshake.bb = 3 : ui32, handshake.name = "addi5"} : <i32>
    %addressResult_12, %dataResult_13 = load[%43] %outputs_2 {handshake.bb = 3 : ui32, handshake.name = "load1"} : <i32>, <i32>, <i32>, <i32>
    %44 = muli %dataResult_11, %dataResult_13 {handshake.bb = 3 : ui32, handshake.name = "muli0"} : <i32>
    %45 = shli %28, %33 {handshake.bb = 3 : ui32, handshake.name = "shli6"} : <i32>
    %46 = shli %28, %35 {handshake.bb = 3 : ui32, handshake.name = "shli7"} : <i32>
    %47 = addi %45, %46 {handshake.bb = 3 : ui32, handshake.name = "addi30"} : <i32>
    %48 = addi %29, %47 {handshake.bb = 3 : ui32, handshake.name = "addi6"} : <i32>
    %addressResult_14, %dataResult_15 = load[%48] %2#1 {handshake.bb = 3 : ui32, handshake.deps = #handshake<deps[["store1", 3, true], ["store1", 4, true]]>, handshake.name = "load2"} : <i32>, <i32>, <i32>, <i32>
    %49 = addi %dataResult_15, %44 {handshake.bb = 3 : ui32, handshake.name = "addi0"} : <i32>
    %50 = shli %28, %33 {handshake.bb = 3 : ui32, handshake.name = "shli8"} : <i32>
    %51 = shli %28, %35 {handshake.bb = 3 : ui32, handshake.name = "shli9"} : <i32>
    %52 = addi %50, %51 {handshake.bb = 3 : ui32, handshake.name = "addi31"} : <i32>
    %53 = addi %29, %52 {handshake.bb = 3 : ui32, handshake.name = "addi7"} : <i32>
    %addressResult_16, %dataResult_17, %doneResult_18 = store[%53] %49 %2#2 {handshake.bb = 3 : ui32, handshake.deps = #handshake<deps[["load2", 3, true], ["store1", 3, true], ["load6", 1, true]]>, handshake.name = "store1"} : <i32>, <i32>, <>, <i32>, <i32>, <>
    %54 = addi %27, %33 {handshake.bb = 3 : ui32, handshake.name = "addi18"} : <i32>
    %55 = cmpi ult, %54, %31 {handshake.bb = 3 : ui32, handshake.name = "cmpi0"} : <i32>
    %trueResult, %falseResult = cond_br %55, %54 {handshake.bb = 3 : ui32, handshake.name = "cond_br9"} : <i1>, <i32>
    %trueResult_19, %falseResult_20 = cond_br %55, %28 {handshake.bb = 3 : ui32, handshake.name = "cond_br10"} : <i1>, <i32>
    %trueResult_21, %falseResult_22 = cond_br %55, %29 {handshake.bb = 3 : ui32, handshake.name = "cond_br11"} : <i1>, <i32>
    %trueResult_23, %falseResult_24 = cond_br %55, %result_8 {handshake.bb = 3 : ui32, handshake.name = "cond_br12"} : <i1>, <>
    %56 = merge %falseResult_20 {handshake.bb = 4 : ui32, handshake.name = "merge0"} : <i32>
    %57 = merge %falseResult_22 {handshake.bb = 4 : ui32, handshake.name = "merge1"} : <i32>
    %result_25, %index_26 = control_merge [%falseResult_24]  {handshake.bb = 4 : ui32, handshake.name = "control_merge3"} : [<>] to <>, <i1>
    %58 = source {handshake.bb = 4 : ui32, handshake.name = "source5"} : <>
    %59 = constant %58 {handshake.bb = 4 : ui32, handshake.name = "constant13", value = 10 : i32} : <>, <i32>
    %60 = source {handshake.bb = 4 : ui32, handshake.name = "source6"} : <>
    %61 = constant %60 {handshake.bb = 4 : ui32, handshake.name = "constant14", value = 1 : i32} : <>, <i32>
    %62 = addi %57, %61 {handshake.bb = 4 : ui32, handshake.name = "addi19"} : <i32>
    %63 = cmpi ult, %62, %59 {handshake.bb = 4 : ui32, handshake.name = "cmpi1"} : <i32>
    %trueResult_27, %falseResult_28 = cond_br %63, %62 {handshake.bb = 4 : ui32, handshake.name = "cond_br13"} : <i1>, <i32>
    %trueResult_29, %falseResult_30 = cond_br %63, %56 {handshake.bb = 4 : ui32, handshake.name = "cond_br14"} : <i1>, <i32>
    %trueResult_31, %falseResult_32 = cond_br %63, %result_25 {handshake.bb = 4 : ui32, handshake.name = "cond_br15"} : <i1>, <>
    %64 = merge %falseResult_30 {handshake.bb = 5 : ui32, handshake.name = "merge2"} : <i32>
    %result_33, %index_34 = control_merge [%falseResult_32]  {handshake.bb = 5 : ui32, handshake.name = "control_merge4"} : [<>] to <>, <i1>
    %65 = constant %result_33 {handshake.bb = 5 : ui32, handshake.name = "constant15", value = 0 : i32} : <>, <i32>
    %66 = source {handshake.bb = 5 : ui32, handshake.name = "source7"} : <>
    %67 = constant %66 {handshake.bb = 5 : ui32, handshake.name = "constant16", value = 10 : i32} : <>, <i32>
    %68 = source {handshake.bb = 5 : ui32, handshake.name = "source8"} : <>
    %69 = constant %68 {handshake.bb = 5 : ui32, handshake.name = "constant17", value = 1 : i32} : <>, <i32>
    %70 = addi %64, %69 {handshake.bb = 5 : ui32, handshake.name = "addi20"} : <i32>
    %71 = cmpi ult, %70, %67 {handshake.bb = 5 : ui32, handshake.name = "cmpi2"} : <i32>
    %trueResult_35, %falseResult_36 = cond_br %71, %70 {handshake.bb = 5 : ui32, handshake.name = "cond_br16"} : <i1>, <i32>
    %trueResult_37, %falseResult_38 = cond_br %71, %result_33 {handshake.bb = 5 : ui32, handshake.name = "cond_br17"} : <i1>, <>
    %trueResult_39, %falseResult_40 = cond_br %71, %65 {handshake.bb = 5 : ui32, handshake.name = "cond_br18"} : <i1>, <i32>
    %72 = mux %index_42 [%falseResult_40, %trueResult_77] {handshake.bb = 6 : ui32, handshake.name = "mux6"} : <i1>, [<i32>, <i32>] to <i32>
    %result_41, %index_42 = control_merge [%falseResult_38, %trueResult_79]  {handshake.bb = 6 : ui32, handshake.name = "control_merge5"} : [<>, <>] to <>, <i1>
    %73 = constant %result_41 {handshake.bb = 6 : ui32, handshake.name = "constant18", value = 0 : i32} : <>, <i32>
    %74 = br %73 {handshake.bb = 6 : ui32, handshake.name = "br16"} : <i32>
    %75 = br %72 {handshake.bb = 6 : ui32, handshake.name = "br17"} : <i32>
    %76 = br %result_41 {handshake.bb = 6 : ui32, handshake.name = "br18"} : <>
    %77 = mux %index_44 [%74, %trueResult_69] {handshake.bb = 7 : ui32, handshake.name = "mux7"} : <i1>, [<i32>, <i32>] to <i32>
    %78 = mux %index_44 [%75, %trueResult_71] {handshake.bb = 7 : ui32, handshake.name = "mux8"} : <i1>, [<i32>, <i32>] to <i32>
    %result_43, %index_44 = control_merge [%76, %trueResult_73]  {handshake.bb = 7 : ui32, handshake.name = "control_merge6"} : [<>, <>] to <>, <i1>
    %79 = constant %result_43 {handshake.bb = 7 : ui32, handshake.name = "constant19", value = 0 : i32} : <>, <i32>
    %80 = constant %result_43 {handshake.bb = 7 : ui32, handshake.name = "constant20", value = 0 : i32} : <>, <i32>
    %81 = source {handshake.bb = 7 : ui32, handshake.name = "source9"} : <>
    %82 = constant %81 {handshake.bb = 7 : ui32, handshake.name = "constant21", value = 1 : i32} : <>, <i32>
    %83 = source {handshake.bb = 7 : ui32, handshake.name = "source10"} : <>
    %84 = constant %83 {handshake.bb = 7 : ui32, handshake.name = "constant22", value = 3 : i32} : <>, <i32>
    %85 = shli %78, %82 {handshake.bb = 7 : ui32, handshake.name = "shli10"} : <i32>
    %86 = shli %78, %84 {handshake.bb = 7 : ui32, handshake.name = "shli11"} : <i32>
    %87 = addi %85, %86 {handshake.bb = 7 : ui32, handshake.name = "addi32"} : <i32>
    %88 = addi %77, %87 {handshake.bb = 7 : ui32, handshake.name = "addi8"} : <i32>
    %addressResult_45, %dataResult_46, %doneResult_47 = store[%88] %79 %1#0 {handshake.bb = 7 : ui32, handshake.deps = #handshake<deps[["load5", 3, true], ["store3", 3, true], ["load7", 1, true]]>, handshake.name = "store2"} : <i32>, <i32>, <>, <i32>, <i32>, <>
    %89 = br %80 {handshake.bb = 7 : ui32, handshake.name = "br19"} : <i32>
    %90 = br %78 {handshake.bb = 7 : ui32, handshake.name = "br20"} : <i32>
    %91 = br %77 {handshake.bb = 7 : ui32, handshake.name = "br21"} : <i32>
    %92 = br %result_43 {handshake.bb = 7 : ui32, handshake.name = "br22"} : <>
    %93 = mux %index_49 [%89, %trueResult_59] {handshake.bb = 8 : ui32, handshake.name = "mux9"} : <i1>, [<i32>, <i32>] to <i32>
    %94 = mux %index_49 [%90, %trueResult_61] {handshake.bb = 8 : ui32, handshake.name = "mux10"} : <i1>, [<i32>, <i32>] to <i32>
    %95 = mux %index_49 [%91, %trueResult_63] {handshake.bb = 8 : ui32, handshake.name = "mux11"} : <i1>, [<i32>, <i32>] to <i32>
    %result_48, %index_49 = control_merge [%92, %trueResult_65]  {handshake.bb = 8 : ui32, handshake.name = "control_merge7"} : [<>, <>] to <>, <i1>
    %96 = source {handshake.bb = 8 : ui32, handshake.name = "source11"} : <>
    %97 = constant %96 {handshake.bb = 8 : ui32, handshake.name = "constant23", value = 10 : i32} : <>, <i32>
    %98 = source {handshake.bb = 8 : ui32, handshake.name = "source12"} : <>
    %99 = constant %98 {handshake.bb = 8 : ui32, handshake.name = "constant24", value = 1 : i32} : <>, <i32>
    %100 = source {handshake.bb = 8 : ui32, handshake.name = "source13"} : <>
    %101 = constant %100 {handshake.bb = 8 : ui32, handshake.name = "constant25", value = 3 : i32} : <>, <i32>
    %102 = shli %94, %99 {handshake.bb = 8 : ui32, handshake.name = "shli12"} : <i32>
    %103 = shli %94, %101 {handshake.bb = 8 : ui32, handshake.name = "shli13"} : <i32>
    %104 = addi %102, %103 {handshake.bb = 8 : ui32, handshake.name = "addi33"} : <i32>
    %105 = addi %93, %104 {handshake.bb = 8 : ui32, handshake.name = "addi9"} : <i32>
    %addressResult_50, %dataResult_51 = load[%105] %outputs_0 {handshake.bb = 8 : ui32, handshake.name = "load3"} : <i32>, <i32>, <i32>, <i32>
    %106 = shli %93, %99 {handshake.bb = 8 : ui32, handshake.name = "shli14"} : <i32>
    %107 = shli %93, %101 {handshake.bb = 8 : ui32, handshake.name = "shli15"} : <i32>
    %108 = addi %106, %107 {handshake.bb = 8 : ui32, handshake.name = "addi34"} : <i32>
    %109 = addi %95, %108 {handshake.bb = 8 : ui32, handshake.name = "addi10"} : <i32>
    %addressResult_52, %dataResult_53 = load[%109] %outputs {handshake.bb = 8 : ui32, handshake.name = "load4"} : <i32>, <i32>, <i32>, <i32>
    %110 = muli %dataResult_51, %dataResult_53 {handshake.bb = 8 : ui32, handshake.name = "muli1"} : <i32>
    %111 = shli %94, %99 {handshake.bb = 8 : ui32, handshake.name = "shli16"} : <i32>
    %112 = shli %94, %101 {handshake.bb = 8 : ui32, handshake.name = "shli17"} : <i32>
    %113 = addi %111, %112 {handshake.bb = 8 : ui32, handshake.name = "addi35"} : <i32>
    %114 = addi %95, %113 {handshake.bb = 8 : ui32, handshake.name = "addi11"} : <i32>
    %addressResult_54, %dataResult_55 = load[%114] %1#1 {handshake.bb = 8 : ui32, handshake.deps = #handshake<deps[["store3", 3, true], ["store3", 4, true]]>, handshake.name = "load5"} : <i32>, <i32>, <i32>, <i32>
    %115 = addi %dataResult_55, %110 {handshake.bb = 8 : ui32, handshake.name = "addi1"} : <i32>
    %116 = shli %94, %99 {handshake.bb = 8 : ui32, handshake.name = "shli18"} : <i32>
    %117 = shli %94, %101 {handshake.bb = 8 : ui32, handshake.name = "shli19"} : <i32>
    %118 = addi %116, %117 {handshake.bb = 8 : ui32, handshake.name = "addi36"} : <i32>
    %119 = addi %95, %118 {handshake.bb = 8 : ui32, handshake.name = "addi12"} : <i32>
    %addressResult_56, %dataResult_57, %doneResult_58 = store[%119] %115 %1#2 {handshake.bb = 8 : ui32, handshake.deps = #handshake<deps[["load5", 3, true], ["store3", 3, true], ["load7", 1, true]]>, handshake.name = "store3"} : <i32>, <i32>, <>, <i32>, <i32>, <>
    %120 = addi %93, %99 {handshake.bb = 8 : ui32, handshake.name = "addi21"} : <i32>
    %121 = cmpi ult, %120, %97 {handshake.bb = 8 : ui32, handshake.name = "cmpi3"} : <i32>
    %trueResult_59, %falseResult_60 = cond_br %121, %120 {handshake.bb = 8 : ui32, handshake.name = "cond_br19"} : <i1>, <i32>
    %trueResult_61, %falseResult_62 = cond_br %121, %94 {handshake.bb = 8 : ui32, handshake.name = "cond_br20"} : <i1>, <i32>
    %trueResult_63, %falseResult_64 = cond_br %121, %95 {handshake.bb = 8 : ui32, handshake.name = "cond_br21"} : <i1>, <i32>
    %trueResult_65, %falseResult_66 = cond_br %121, %result_48 {handshake.bb = 8 : ui32, handshake.name = "cond_br22"} : <i1>, <>
    %122 = merge %falseResult_62 {handshake.bb = 9 : ui32, handshake.name = "merge3"} : <i32>
    %123 = merge %falseResult_64 {handshake.bb = 9 : ui32, handshake.name = "merge4"} : <i32>
    %result_67, %index_68 = control_merge [%falseResult_66]  {handshake.bb = 9 : ui32, handshake.name = "control_merge8"} : [<>] to <>, <i1>
    %124 = source {handshake.bb = 9 : ui32, handshake.name = "source14"} : <>
    %125 = constant %124 {handshake.bb = 9 : ui32, handshake.name = "constant26", value = 10 : i32} : <>, <i32>
    %126 = source {handshake.bb = 9 : ui32, handshake.name = "source15"} : <>
    %127 = constant %126 {handshake.bb = 9 : ui32, handshake.name = "constant27", value = 1 : i32} : <>, <i32>
    %128 = addi %123, %127 {handshake.bb = 9 : ui32, handshake.name = "addi22"} : <i32>
    %129 = cmpi ult, %128, %125 {handshake.bb = 9 : ui32, handshake.name = "cmpi4"} : <i32>
    %trueResult_69, %falseResult_70 = cond_br %129, %128 {handshake.bb = 9 : ui32, handshake.name = "cond_br23"} : <i1>, <i32>
    %trueResult_71, %falseResult_72 = cond_br %129, %122 {handshake.bb = 9 : ui32, handshake.name = "cond_br24"} : <i1>, <i32>
    %trueResult_73, %falseResult_74 = cond_br %129, %result_67 {handshake.bb = 9 : ui32, handshake.name = "cond_br25"} : <i1>, <>
    %130 = merge %falseResult_72 {handshake.bb = 10 : ui32, handshake.name = "merge5"} : <i32>
    %result_75, %index_76 = control_merge [%falseResult_74]  {handshake.bb = 10 : ui32, handshake.name = "control_merge9"} : [<>] to <>, <i1>
    %131 = constant %result_75 {handshake.bb = 10 : ui32, handshake.name = "constant28", value = 0 : i32} : <>, <i32>
    %132 = source {handshake.bb = 10 : ui32, handshake.name = "source16"} : <>
    %133 = constant %132 {handshake.bb = 10 : ui32, handshake.name = "constant29", value = 10 : i32} : <>, <i32>
    %134 = source {handshake.bb = 10 : ui32, handshake.name = "source17"} : <>
    %135 = constant %134 {handshake.bb = 10 : ui32, handshake.name = "constant30", value = 1 : i32} : <>, <i32>
    %136 = addi %130, %135 {handshake.bb = 10 : ui32, handshake.name = "addi23"} : <i32>
    %137 = cmpi ult, %136, %133 {handshake.bb = 10 : ui32, handshake.name = "cmpi5"} : <i32>
    %trueResult_77, %falseResult_78 = cond_br %137, %136 {handshake.bb = 10 : ui32, handshake.name = "cond_br26"} : <i1>, <i32>
    %trueResult_79, %falseResult_80 = cond_br %137, %result_75 {handshake.bb = 10 : ui32, handshake.name = "cond_br27"} : <i1>, <>
    %trueResult_81, %falseResult_82 = cond_br %137, %131 {handshake.bb = 10 : ui32, handshake.name = "cond_br28"} : <i1>, <i32>
    %138 = mux %index_84 [%falseResult_82, %trueResult_119] {handshake.bb = 11 : ui32, handshake.name = "mux12"} : <i1>, [<i32>, <i32>] to <i32>
    %result_83, %index_84 = control_merge [%falseResult_80, %trueResult_121]  {handshake.bb = 11 : ui32, handshake.name = "control_merge10"} : [<>, <>] to <>, <i1>
    %139 = constant %result_83 {handshake.bb = 11 : ui32, handshake.name = "constant31", value = 0 : i32} : <>, <i32>
    %140 = br %139 {handshake.bb = 11 : ui32, handshake.name = "br23"} : <i32>
    %141 = br %138 {handshake.bb = 11 : ui32, handshake.name = "br24"} : <i32>
    %142 = br %result_83 {handshake.bb = 11 : ui32, handshake.name = "br25"} : <>
    %143 = mux %index_86 [%140, %trueResult_111] {handshake.bb = 12 : ui32, handshake.name = "mux13"} : <i1>, [<i32>, <i32>] to <i32>
    %144 = mux %index_86 [%141, %trueResult_113] {handshake.bb = 12 : ui32, handshake.name = "mux14"} : <i1>, [<i32>, <i32>] to <i32>
    %result_85, %index_86 = control_merge [%142, %trueResult_115]  {handshake.bb = 12 : ui32, handshake.name = "control_merge11"} : [<>, <>] to <>, <i1>
    %145 = constant %result_85 {handshake.bb = 12 : ui32, handshake.name = "constant32", value = 0 : i32} : <>, <i32>
    %146 = constant %result_85 {handshake.bb = 12 : ui32, handshake.name = "constant33", value = 0 : i32} : <>, <i32>
    %147 = source {handshake.bb = 12 : ui32, handshake.name = "source18"} : <>
    %148 = constant %147 {handshake.bb = 12 : ui32, handshake.name = "constant34", value = 1 : i32} : <>, <i32>
    %149 = source {handshake.bb = 12 : ui32, handshake.name = "source19"} : <>
    %150 = constant %149 {handshake.bb = 12 : ui32, handshake.name = "constant35", value = 3 : i32} : <>, <i32>
    %151 = shli %144, %148 {handshake.bb = 12 : ui32, handshake.name = "shli20"} : <i32>
    %152 = shli %144, %150 {handshake.bb = 12 : ui32, handshake.name = "shli21"} : <i32>
    %153 = addi %151, %152 {handshake.bb = 12 : ui32, handshake.name = "addi37"} : <i32>
    %154 = addi %143, %153 {handshake.bb = 12 : ui32, handshake.name = "addi13"} : <i32>
    %addressResult_87, %dataResult_88, %doneResult_89 = store[%154] %145 %0#0 {handshake.bb = 12 : ui32, handshake.deps = #handshake<deps[["load8", 3, true], ["store5", 3, true]]>, handshake.name = "store4"} : <i32>, <i32>, <>, <i32>, <i32>, <>
    %155 = br %146 {handshake.bb = 12 : ui32, handshake.name = "br26"} : <i32>
    %156 = br %144 {handshake.bb = 12 : ui32, handshake.name = "br27"} : <i32>
    %157 = br %143 {handshake.bb = 12 : ui32, handshake.name = "br28"} : <i32>
    %158 = br %result_85 {handshake.bb = 12 : ui32, handshake.name = "br29"} : <>
    %159 = mux %index_91 [%155, %trueResult_101] {handshake.bb = 13 : ui32, handshake.name = "mux15"} : <i1>, [<i32>, <i32>] to <i32>
    %160 = mux %index_91 [%156, %trueResult_103] {handshake.bb = 13 : ui32, handshake.name = "mux16"} : <i1>, [<i32>, <i32>] to <i32>
    %161 = mux %index_91 [%157, %trueResult_105] {handshake.bb = 13 : ui32, handshake.name = "mux17"} : <i1>, [<i32>, <i32>] to <i32>
    %result_90, %index_91 = control_merge [%158, %trueResult_107]  {handshake.bb = 13 : ui32, handshake.name = "control_merge12"} : [<>, <>] to <>, <i1>
    %162 = source {handshake.bb = 13 : ui32, handshake.name = "source20"} : <>
    %163 = constant %162 {handshake.bb = 13 : ui32, handshake.name = "constant36", value = 10 : i32} : <>, <i32>
    %164 = source {handshake.bb = 13 : ui32, handshake.name = "source21"} : <>
    %165 = constant %164 {handshake.bb = 13 : ui32, handshake.name = "constant37", value = 1 : i32} : <>, <i32>
    %166 = source {handshake.bb = 13 : ui32, handshake.name = "source22"} : <>
    %167 = constant %166 {handshake.bb = 13 : ui32, handshake.name = "constant38", value = 3 : i32} : <>, <i32>
    %168 = shli %160, %165 {handshake.bb = 13 : ui32, handshake.name = "shli22"} : <i32>
    %169 = shli %160, %167 {handshake.bb = 13 : ui32, handshake.name = "shli23"} : <i32>
    %170 = addi %168, %169 {handshake.bb = 13 : ui32, handshake.name = "addi38"} : <i32>
    %171 = addi %159, %170 {handshake.bb = 13 : ui32, handshake.name = "addi14"} : <i32>
    %addressResult_92, %dataResult_93 = load[%171] %2#3 {handshake.bb = 13 : ui32, handshake.name = "load6"} : <i32>, <i32>, <i32>, <i32>
    %172 = shli %159, %165 {handshake.bb = 13 : ui32, handshake.name = "shli24"} : <i32>
    %173 = shli %159, %167 {handshake.bb = 13 : ui32, handshake.name = "shli25"} : <i32>
    %174 = addi %172, %173 {handshake.bb = 13 : ui32, handshake.name = "addi39"} : <i32>
    %175 = addi %161, %174 {handshake.bb = 13 : ui32, handshake.name = "addi15"} : <i32>
    %addressResult_94, %dataResult_95 = load[%175] %1#3 {handshake.bb = 13 : ui32, handshake.name = "load7"} : <i32>, <i32>, <i32>, <i32>
    %176 = muli %dataResult_93, %dataResult_95 {handshake.bb = 13 : ui32, handshake.name = "muli2"} : <i32>
    %177 = shli %160, %165 {handshake.bb = 13 : ui32, handshake.name = "shli26"} : <i32>
    %178 = shli %160, %167 {handshake.bb = 13 : ui32, handshake.name = "shli27"} : <i32>
    %179 = addi %177, %178 {handshake.bb = 13 : ui32, handshake.name = "addi40"} : <i32>
    %180 = addi %161, %179 {handshake.bb = 13 : ui32, handshake.name = "addi16"} : <i32>
    %addressResult_96, %dataResult_97 = load[%180] %0#1 {handshake.bb = 13 : ui32, handshake.deps = #handshake<deps[["store5", 3, true], ["store5", 4, true]]>, handshake.name = "load8"} : <i32>, <i32>, <i32>, <i32>
    %181 = addi %dataResult_97, %176 {handshake.bb = 13 : ui32, handshake.name = "addi2"} : <i32>
    %182 = shli %160, %165 {handshake.bb = 13 : ui32, handshake.name = "shli28"} : <i32>
    %183 = shli %160, %167 {handshake.bb = 13 : ui32, handshake.name = "shli29"} : <i32>
    %184 = addi %182, %183 {handshake.bb = 13 : ui32, handshake.name = "addi41"} : <i32>
    %185 = addi %161, %184 {handshake.bb = 13 : ui32, handshake.name = "addi17"} : <i32>
    %addressResult_98, %dataResult_99, %doneResult_100 = store[%185] %181 %0#2 {handshake.bb = 13 : ui32, handshake.deps = #handshake<deps[["load8", 3, true], ["store5", 3, true]]>, handshake.name = "store5"} : <i32>, <i32>, <>, <i32>, <i32>, <>
    %186 = addi %159, %165 {handshake.bb = 13 : ui32, handshake.name = "addi24"} : <i32>
    %187 = cmpi ult, %186, %163 {handshake.bb = 13 : ui32, handshake.name = "cmpi6"} : <i32>
    %trueResult_101, %falseResult_102 = cond_br %187, %186 {handshake.bb = 13 : ui32, handshake.name = "cond_br29"} : <i1>, <i32>
    %trueResult_103, %falseResult_104 = cond_br %187, %160 {handshake.bb = 13 : ui32, handshake.name = "cond_br30"} : <i1>, <i32>
    %trueResult_105, %falseResult_106 = cond_br %187, %161 {handshake.bb = 13 : ui32, handshake.name = "cond_br31"} : <i1>, <i32>
    %trueResult_107, %falseResult_108 = cond_br %187, %result_90 {handshake.bb = 13 : ui32, handshake.name = "cond_br32"} : <i1>, <>
    %188 = merge %falseResult_104 {handshake.bb = 14 : ui32, handshake.name = "merge6"} : <i32>
    %189 = merge %falseResult_106 {handshake.bb = 14 : ui32, handshake.name = "merge7"} : <i32>
    %result_109, %index_110 = control_merge [%falseResult_108]  {handshake.bb = 14 : ui32, handshake.name = "control_merge13"} : [<>] to <>, <i1>
    %190 = source {handshake.bb = 14 : ui32, handshake.name = "source23"} : <>
    %191 = constant %190 {handshake.bb = 14 : ui32, handshake.name = "constant39", value = 10 : i32} : <>, <i32>
    %192 = source {handshake.bb = 14 : ui32, handshake.name = "source24"} : <>
    %193 = constant %192 {handshake.bb = 14 : ui32, handshake.name = "constant40", value = 1 : i32} : <>, <i32>
    %194 = addi %189, %193 {handshake.bb = 14 : ui32, handshake.name = "addi25"} : <i32>
    %195 = cmpi ult, %194, %191 {handshake.bb = 14 : ui32, handshake.name = "cmpi7"} : <i32>
    %trueResult_111, %falseResult_112 = cond_br %195, %194 {handshake.bb = 14 : ui32, handshake.name = "cond_br33"} : <i1>, <i32>
    %trueResult_113, %falseResult_114 = cond_br %195, %188 {handshake.bb = 14 : ui32, handshake.name = "cond_br34"} : <i1>, <i32>
    %trueResult_115, %falseResult_116 = cond_br %195, %result_109 {handshake.bb = 14 : ui32, handshake.name = "cond_br35"} : <i1>, <>
    %196 = merge %falseResult_114 {handshake.bb = 15 : ui32, handshake.name = "merge8"} : <i32>
    %result_117, %index_118 = control_merge [%falseResult_116]  {handshake.bb = 15 : ui32, handshake.name = "control_merge14"} : [<>] to <>, <i1>
    %197 = source {handshake.bb = 15 : ui32, handshake.name = "source25"} : <>
    %198 = constant %197 {handshake.bb = 15 : ui32, handshake.name = "constant41", value = 10 : i32} : <>, <i32>
    %199 = source {handshake.bb = 15 : ui32, handshake.name = "source26"} : <>
    %200 = constant %199 {handshake.bb = 15 : ui32, handshake.name = "constant42", value = 1 : i32} : <>, <i32>
    %201 = addi %196, %200 {handshake.bb = 15 : ui32, handshake.name = "addi26"} : <i32>
    %202 = cmpi ult, %201, %198 {handshake.bb = 15 : ui32, handshake.name = "cmpi8"} : <i32>
    %trueResult_119, %falseResult_120 = cond_br %202, %201 {handshake.bb = 15 : ui32, handshake.name = "cond_br36"} : <i1>, <i32>
    %trueResult_121, %falseResult_122 = cond_br %202, %result_117 {handshake.bb = 15 : ui32, handshake.name = "cond_br37"} : <i1>, <>
    %result_123, %index_124 = control_merge [%falseResult_122]  {handshake.bb = 16 : ui32, handshake.name = "control_merge15"} : [<>] to <>, <i1>
    end {handshake.bb = 16 : ui32, handshake.name = "end0"} %memEnd_5, %memEnd_3, %memEnd_1, %memEnd, %2#4, %1#4, %0#3, %arg14 : <>, <>, <>, <>, <>, <>, <>, <>
  }
}

