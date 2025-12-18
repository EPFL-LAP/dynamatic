module {
  handshake.func @kernel_2mm(%arg0: !handshake.channel<i32>, %arg1: !handshake.channel<i32>, %arg2: memref<100xi32>, %arg3: memref<100xi32>, %arg4: memref<100xi32>, %arg5: memref<100xi32>, %arg6: memref<100xi32>, %arg7: !handshake.control<>, %arg8: !handshake.control<>, %arg9: !handshake.control<>, %arg10: !handshake.control<>, %arg11: !handshake.control<>, %arg12: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["alpha", "beta", "tmp", "A", "B", "C", "D", "tmp_start", "A_start", "B_start", "C_start", "D_start", "start"], cfg.edges = "[0,1][7,8][2,3][9,7,10,cmpi4][4,2,5,cmpi1][6,7][1,2][8,8,9,cmpi3][3,3,4,cmpi0][10,6,11,cmpi5][5,1,6,cmpi2]", resNames = ["tmp_end", "A_end", "B_end", "C_end", "D_end", "end"]} {
    %0:5 = lsq[%arg6 : memref<100xi32>] (%arg11, %result_53, %addressResult_55, %addressResult_57, %dataResult_58, %result_60, %addressResult_66, %addressResult_68, %dataResult_69, %result_99)  {groupSizes = [2 : i32, 2 : i32], handshake.name = "lsq0"} : (!handshake.control<>, !handshake.control<>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>) -> (!handshake.channel<i32>, !handshake.control<>, !handshake.channel<i32>, !handshake.control<>, !handshake.control<>)
    %outputs, %memEnd = mem_controller[%arg5 : memref<100xi32>] %arg10 (%addressResult_64) %result_99 {connectedBlocks = [8 : i32], handshake.name = "mem_controller0"} :    (!handshake.channel<i32>) -> !handshake.channel<i32>
    %outputs_0, %memEnd_1 = mem_controller[%arg4 : memref<100xi32>] %arg9 (%addressResult_10) %result_99 {connectedBlocks = [3 : i32], handshake.name = "mem_controller1"} :    (!handshake.channel<i32>) -> !handshake.channel<i32>
    %outputs_2, %memEnd_3 = mem_controller[%arg3 : memref<100xi32>] %arg8 (%addressResult_8) %result_99 {connectedBlocks = [3 : i32], handshake.name = "mem_controller2"} :    (!handshake.channel<i32>) -> !handshake.channel<i32>
    %1:5 = lsq[%arg2 : memref<100xi32>] (%arg7, %result_4, %addressResult, %dataResult, %result_6, %addressResult_12, %addressResult_14, %dataResult_15, %result_60, %addressResult_62, %result_99)  {groupSizes = [1 : i32, 2 : i32, 1 : i32], handshake.name = "lsq1"} : (!handshake.control<>, !handshake.control<>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>, !handshake.channel<i32>, !handshake.control<>) -> (!handshake.control<>, !handshake.channel<i32>, !handshake.control<>, !handshake.channel<i32>, !handshake.control<>)
    %2 = constant %arg12 {handshake.bb = 0 : ui32, handshake.name = "constant3", value = 0 : i32} : <>, <i32>
    %3 = br %2 {handshake.bb = 0 : ui32, handshake.name = "br3"} : <i32>
    %4 = br %arg0 {handshake.bb = 0 : ui32, handshake.name = "br6"} : <i32>
    %5 = br %arg1 {handshake.bb = 0 : ui32, handshake.name = "br7"} : <i32>
    %6 = br %arg12 {handshake.bb = 0 : ui32, handshake.name = "br8"} : <>
    %7 = mux %index [%3, %trueResult_41] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i32>, <i32>] to <i32>
    %8 = mux %index [%4, %trueResult_43] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i1>, [<i32>, <i32>] to <i32>
    %9 = mux %index [%5, %trueResult_45] {handshake.bb = 1 : ui32, handshake.name = "mux2"} : <i1>, [<i32>, <i32>] to <i32>
    %result, %index = control_merge [%6, %trueResult_47]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %10 = constant %result {handshake.bb = 1 : ui32, handshake.name = "constant5", value = 0 : i32} : <>, <i32>
    %11 = br %10 {handshake.bb = 1 : ui32, handshake.name = "br9"} : <i32>
    %12 = br %8 {handshake.bb = 1 : ui32, handshake.name = "br10"} : <i32>
    %13 = br %9 {handshake.bb = 1 : ui32, handshake.name = "br11"} : <i32>
    %14 = br %7 {handshake.bb = 1 : ui32, handshake.name = "br12"} : <i32>
    %15 = br %result {handshake.bb = 1 : ui32, handshake.name = "br13"} : <>
    %16 = mux %index_5 [%11, %trueResult_29] {handshake.bb = 2 : ui32, handshake.name = "mux3"} : <i1>, [<i32>, <i32>] to <i32>
    %17 = mux %index_5 [%12, %trueResult_31] {handshake.bb = 2 : ui32, handshake.name = "mux4"} : <i1>, [<i32>, <i32>] to <i32>
    %18 = mux %index_5 [%13, %trueResult_33] {handshake.bb = 2 : ui32, handshake.name = "mux5"} : <i1>, [<i32>, <i32>] to <i32>
    %19 = mux %index_5 [%14, %trueResult_35] {handshake.bb = 2 : ui32, handshake.name = "mux6"} : <i1>, [<i32>, <i32>] to <i32>
    %result_4, %index_5 = control_merge [%15, %trueResult_37]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>, <>] to <>, <i1>
    %20 = constant %result_4 {handshake.bb = 2 : ui32, handshake.name = "constant6", value = 0 : i32} : <>, <i32>
    %21 = constant %result_4 {handshake.bb = 2 : ui32, handshake.name = "constant7", value = 0 : i32} : <>, <i32>
    %22 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %23 = constant %22 {handshake.bb = 2 : ui32, handshake.name = "constant8", value = 1 : i32} : <>, <i32>
    %24 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %25 = constant %24 {handshake.bb = 2 : ui32, handshake.name = "constant9", value = 3 : i32} : <>, <i32>
    %26 = shli %19, %23 {handshake.bb = 2 : ui32, handshake.name = "shli0"} : <i32>
    %27 = shli %19, %25 {handshake.bb = 2 : ui32, handshake.name = "shli1"} : <i32>
    %28 = addi %26, %27 {handshake.bb = 2 : ui32, handshake.name = "addi19"} : <i32>
    %29 = addi %16, %28 {handshake.bb = 2 : ui32, handshake.name = "addi2"} : <i32>
    %addressResult, %dataResult, %doneResult = store[%29] %20 %1#0 {handshake.bb = 2 : ui32, handshake.deps = #handshake<deps[["load2", 3, true], ["store1", 3, true], ["load4", 1, true]]>, handshake.name = "store0"} : <i32>, <i32>, <>, <i32>, <i32>, <>
    %30 = br %21 {handshake.bb = 2 : ui32, handshake.name = "br14"} : <i32>
    %31 = br %17 {handshake.bb = 2 : ui32, handshake.name = "br15"} : <i32>
    %32 = br %18 {handshake.bb = 2 : ui32, handshake.name = "br16"} : <i32>
    %33 = br %19 {handshake.bb = 2 : ui32, handshake.name = "br17"} : <i32>
    %34 = br %16 {handshake.bb = 2 : ui32, handshake.name = "br18"} : <i32>
    %35 = br %result_4 {handshake.bb = 2 : ui32, handshake.name = "br19"} : <>
    %36 = mux %index_7 [%30, %trueResult] {handshake.bb = 3 : ui32, handshake.name = "mux7"} : <i1>, [<i32>, <i32>] to <i32>
    %37 = mux %index_7 [%31, %trueResult_17] {handshake.bb = 3 : ui32, handshake.name = "mux8"} : <i1>, [<i32>, <i32>] to <i32>
    %38 = mux %index_7 [%32, %trueResult_19] {handshake.bb = 3 : ui32, handshake.name = "mux9"} : <i1>, [<i32>, <i32>] to <i32>
    %39 = mux %index_7 [%33, %trueResult_21] {handshake.bb = 3 : ui32, handshake.name = "mux10"} : <i1>, [<i32>, <i32>] to <i32>
    %40 = mux %index_7 [%34, %trueResult_23] {handshake.bb = 3 : ui32, handshake.name = "mux11"} : <i1>, [<i32>, <i32>] to <i32>
    %result_6, %index_7 = control_merge [%35, %trueResult_25]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>, <>] to <>, <i1>
    %41 = source {handshake.bb = 3 : ui32, handshake.name = "source2"} : <>
    %42 = constant %41 {handshake.bb = 3 : ui32, handshake.name = "constant10", value = 10 : i32} : <>, <i32>
    %43 = source {handshake.bb = 3 : ui32, handshake.name = "source3"} : <>
    %44 = constant %43 {handshake.bb = 3 : ui32, handshake.name = "constant11", value = 1 : i32} : <>, <i32>
    %45 = source {handshake.bb = 3 : ui32, handshake.name = "source4"} : <>
    %46 = constant %45 {handshake.bb = 3 : ui32, handshake.name = "constant12", value = 3 : i32} : <>, <i32>
    %47 = shli %39, %44 {handshake.bb = 3 : ui32, handshake.name = "shli2"} : <i32>
    %48 = shli %39, %46 {handshake.bb = 3 : ui32, handshake.name = "shli3"} : <i32>
    %49 = addi %47, %48 {handshake.bb = 3 : ui32, handshake.name = "addi20"} : <i32>
    %50 = addi %36, %49 {handshake.bb = 3 : ui32, handshake.name = "addi3"} : <i32>
    %addressResult_8, %dataResult_9 = load[%50] %outputs_2 {handshake.bb = 3 : ui32, handshake.name = "load0"} : <i32>, <i32>, <i32>, <i32>
    %51 = muli %37, %dataResult_9 {handshake.bb = 3 : ui32, handshake.name = "muli0"} : <i32>
    %52 = shli %36, %44 {handshake.bb = 3 : ui32, handshake.name = "shli4"} : <i32>
    %53 = shli %36, %46 {handshake.bb = 3 : ui32, handshake.name = "shli5"} : <i32>
    %54 = addi %52, %53 {handshake.bb = 3 : ui32, handshake.name = "addi21"} : <i32>
    %55 = addi %40, %54 {handshake.bb = 3 : ui32, handshake.name = "addi4"} : <i32>
    %addressResult_10, %dataResult_11 = load[%55] %outputs_0 {handshake.bb = 3 : ui32, handshake.name = "load1"} : <i32>, <i32>, <i32>, <i32>
    %56 = muli %51, %dataResult_11 {handshake.bb = 3 : ui32, handshake.name = "muli1"} : <i32>
    %57 = shli %39, %44 {handshake.bb = 3 : ui32, handshake.name = "shli6"} : <i32>
    %58 = shli %39, %46 {handshake.bb = 3 : ui32, handshake.name = "shli7"} : <i32>
    %59 = addi %57, %58 {handshake.bb = 3 : ui32, handshake.name = "addi22"} : <i32>
    %60 = addi %40, %59 {handshake.bb = 3 : ui32, handshake.name = "addi5"} : <i32>
    %addressResult_12, %dataResult_13 = load[%60] %1#1 {handshake.bb = 3 : ui32, handshake.deps = #handshake<deps[["store1", 3, true], ["store1", 4, true]]>, handshake.name = "load2"} : <i32>, <i32>, <i32>, <i32>
    %61 = addi %dataResult_13, %56 {handshake.bb = 3 : ui32, handshake.name = "addi0"} : <i32>
    %62 = shli %39, %44 {handshake.bb = 3 : ui32, handshake.name = "shli8"} : <i32>
    %63 = shli %39, %46 {handshake.bb = 3 : ui32, handshake.name = "shli9"} : <i32>
    %64 = addi %62, %63 {handshake.bb = 3 : ui32, handshake.name = "addi23"} : <i32>
    %65 = addi %40, %64 {handshake.bb = 3 : ui32, handshake.name = "addi6"} : <i32>
    %addressResult_14, %dataResult_15, %doneResult_16 = store[%65] %61 %1#2 {handshake.bb = 3 : ui32, handshake.deps = #handshake<deps[["load2", 3, true], ["store1", 3, true], ["load4", 1, true]]>, handshake.name = "store1"} : <i32>, <i32>, <>, <i32>, <i32>, <>
    %66 = addi %36, %44 {handshake.bb = 3 : ui32, handshake.name = "addi13"} : <i32>
    %67 = cmpi ult, %66, %42 {handshake.bb = 3 : ui32, handshake.name = "cmpi0"} : <i32>
    %trueResult, %falseResult = cond_br %67, %66 {handshake.bb = 3 : ui32, handshake.name = "cond_br6"} : <i1>, <i32>
    %trueResult_17, %falseResult_18 = cond_br %67, %37 {handshake.bb = 3 : ui32, handshake.name = "cond_br7"} : <i1>, <i32>
    %trueResult_19, %falseResult_20 = cond_br %67, %38 {handshake.bb = 3 : ui32, handshake.name = "cond_br8"} : <i1>, <i32>
    %trueResult_21, %falseResult_22 = cond_br %67, %39 {handshake.bb = 3 : ui32, handshake.name = "cond_br9"} : <i1>, <i32>
    %trueResult_23, %falseResult_24 = cond_br %67, %40 {handshake.bb = 3 : ui32, handshake.name = "cond_br10"} : <i1>, <i32>
    %trueResult_25, %falseResult_26 = cond_br %67, %result_6 {handshake.bb = 3 : ui32, handshake.name = "cond_br11"} : <i1>, <>
    %68 = merge %falseResult_18 {handshake.bb = 4 : ui32, handshake.name = "merge0"} : <i32>
    %69 = merge %falseResult_20 {handshake.bb = 4 : ui32, handshake.name = "merge1"} : <i32>
    %70 = merge %falseResult_22 {handshake.bb = 4 : ui32, handshake.name = "merge2"} : <i32>
    %71 = merge %falseResult_24 {handshake.bb = 4 : ui32, handshake.name = "merge3"} : <i32>
    %result_27, %index_28 = control_merge [%falseResult_26]  {handshake.bb = 4 : ui32, handshake.name = "control_merge3"} : [<>] to <>, <i1>
    %72 = source {handshake.bb = 4 : ui32, handshake.name = "source5"} : <>
    %73 = constant %72 {handshake.bb = 4 : ui32, handshake.name = "constant13", value = 10 : i32} : <>, <i32>
    %74 = source {handshake.bb = 4 : ui32, handshake.name = "source6"} : <>
    %75 = constant %74 {handshake.bb = 4 : ui32, handshake.name = "constant14", value = 1 : i32} : <>, <i32>
    %76 = addi %71, %75 {handshake.bb = 4 : ui32, handshake.name = "addi14"} : <i32>
    %77 = cmpi ult, %76, %73 {handshake.bb = 4 : ui32, handshake.name = "cmpi1"} : <i32>
    %trueResult_29, %falseResult_30 = cond_br %77, %76 {handshake.bb = 4 : ui32, handshake.name = "cond_br12"} : <i1>, <i32>
    %trueResult_31, %falseResult_32 = cond_br %77, %68 {handshake.bb = 4 : ui32, handshake.name = "cond_br13"} : <i1>, <i32>
    %trueResult_33, %falseResult_34 = cond_br %77, %69 {handshake.bb = 4 : ui32, handshake.name = "cond_br14"} : <i1>, <i32>
    %trueResult_35, %falseResult_36 = cond_br %77, %70 {handshake.bb = 4 : ui32, handshake.name = "cond_br15"} : <i1>, <i32>
    %trueResult_37, %falseResult_38 = cond_br %77, %result_27 {handshake.bb = 4 : ui32, handshake.name = "cond_br16"} : <i1>, <>
    %78 = merge %falseResult_32 {handshake.bb = 5 : ui32, handshake.name = "merge4"} : <i32>
    %79 = merge %falseResult_34 {handshake.bb = 5 : ui32, handshake.name = "merge5"} : <i32>
    %80 = merge %falseResult_36 {handshake.bb = 5 : ui32, handshake.name = "merge6"} : <i32>
    %result_39, %index_40 = control_merge [%falseResult_38]  {handshake.bb = 5 : ui32, handshake.name = "control_merge4"} : [<>] to <>, <i1>
    %81 = constant %result_39 {handshake.bb = 5 : ui32, handshake.name = "constant15", value = 0 : i32} : <>, <i32>
    %82 = source {handshake.bb = 5 : ui32, handshake.name = "source7"} : <>
    %83 = constant %82 {handshake.bb = 5 : ui32, handshake.name = "constant16", value = 10 : i32} : <>, <i32>
    %84 = source {handshake.bb = 5 : ui32, handshake.name = "source8"} : <>
    %85 = constant %84 {handshake.bb = 5 : ui32, handshake.name = "constant17", value = 1 : i32} : <>, <i32>
    %86 = addi %80, %85 {handshake.bb = 5 : ui32, handshake.name = "addi15"} : <i32>
    %87 = cmpi ult, %86, %83 {handshake.bb = 5 : ui32, handshake.name = "cmpi2"} : <i32>
    %trueResult_41, %falseResult_42 = cond_br %87, %86 {handshake.bb = 5 : ui32, handshake.name = "cond_br17"} : <i1>, <i32>
    %trueResult_43, %falseResult_44 = cond_br %87, %78 {handshake.bb = 5 : ui32, handshake.name = "cond_br18"} : <i1>, <i32>
    %trueResult_45, %falseResult_46 = cond_br %87, %79 {handshake.bb = 5 : ui32, handshake.name = "cond_br19"} : <i1>, <i32>
    %trueResult_47, %falseResult_48 = cond_br %87, %result_39 {handshake.bb = 5 : ui32, handshake.name = "cond_br20"} : <i1>, <>
    %trueResult_49, %falseResult_50 = cond_br %87, %81 {handshake.bb = 5 : ui32, handshake.name = "cond_br21"} : <i1>, <i32>
    %88 = mux %index_52 [%falseResult_50, %trueResult_93] {handshake.bb = 6 : ui32, handshake.name = "mux12"} : <i1>, [<i32>, <i32>] to <i32>
    %89 = mux %index_52 [%falseResult_46, %trueResult_95] {handshake.bb = 6 : ui32, handshake.name = "mux13"} : <i1>, [<i32>, <i32>] to <i32>
    %result_51, %index_52 = control_merge [%falseResult_48, %trueResult_97]  {handshake.bb = 6 : ui32, handshake.name = "control_merge5"} : [<>, <>] to <>, <i1>
    %90 = constant %result_51 {handshake.bb = 6 : ui32, handshake.name = "constant18", value = 0 : i32} : <>, <i32>
    %91 = br %90 {handshake.bb = 6 : ui32, handshake.name = "br20"} : <i32>
    %92 = br %89 {handshake.bb = 6 : ui32, handshake.name = "br21"} : <i32>
    %93 = br %88 {handshake.bb = 6 : ui32, handshake.name = "br22"} : <i32>
    %94 = br %result_51 {handshake.bb = 6 : ui32, handshake.name = "br23"} : <>
    %95 = mux %index_54 [%91, %trueResult_83] {handshake.bb = 7 : ui32, handshake.name = "mux14"} : <i1>, [<i32>, <i32>] to <i32>
    %96 = mux %index_54 [%92, %trueResult_85] {handshake.bb = 7 : ui32, handshake.name = "mux15"} : <i1>, [<i32>, <i32>] to <i32>
    %97 = mux %index_54 [%93, %trueResult_87] {handshake.bb = 7 : ui32, handshake.name = "mux16"} : <i1>, [<i32>, <i32>] to <i32>
    %result_53, %index_54 = control_merge [%94, %trueResult_89]  {handshake.bb = 7 : ui32, handshake.name = "control_merge6"} : [<>, <>] to <>, <i1>
    %98 = constant %result_53 {handshake.bb = 7 : ui32, handshake.name = "constant19", value = 0 : i32} : <>, <i32>
    %99 = source {handshake.bb = 7 : ui32, handshake.name = "source9"} : <>
    %100 = constant %99 {handshake.bb = 7 : ui32, handshake.name = "constant20", value = 1 : i32} : <>, <i32>
    %101 = source {handshake.bb = 7 : ui32, handshake.name = "source10"} : <>
    %102 = constant %101 {handshake.bb = 7 : ui32, handshake.name = "constant21", value = 3 : i32} : <>, <i32>
    %103 = shli %97, %100 {handshake.bb = 7 : ui32, handshake.name = "shli10"} : <i32>
    %104 = shli %97, %102 {handshake.bb = 7 : ui32, handshake.name = "shli11"} : <i32>
    %105 = addi %103, %104 {handshake.bb = 7 : ui32, handshake.name = "addi24"} : <i32>
    %106 = addi %95, %105 {handshake.bb = 7 : ui32, handshake.name = "addi7"} : <i32>
    %addressResult_55, %dataResult_56 = load[%106] %0#0 {handshake.bb = 7 : ui32, handshake.deps = #handshake<deps[["store2", 3, true], ["store3", 3, true]]>, handshake.name = "load3"} : <i32>, <i32>, <i32>, <i32>
    %107 = muli %dataResult_56, %96 {handshake.bb = 7 : ui32, handshake.name = "muli2"} : <i32>
    %108 = shli %97, %100 {handshake.bb = 7 : ui32, handshake.name = "shli12"} : <i32>
    %109 = shli %97, %102 {handshake.bb = 7 : ui32, handshake.name = "shli13"} : <i32>
    %110 = addi %108, %109 {handshake.bb = 7 : ui32, handshake.name = "addi25"} : <i32>
    %111 = addi %95, %110 {handshake.bb = 7 : ui32, handshake.name = "addi8"} : <i32>
    %addressResult_57, %dataResult_58, %doneResult_59 = store[%111] %107 %0#1 {handshake.bb = 7 : ui32, handshake.deps = #handshake<deps[["load6", 3, true], ["store3", 3, true]]>, handshake.name = "store2"} : <i32>, <i32>, <>, <i32>, <i32>, <>
    %112 = br %98 {handshake.bb = 7 : ui32, handshake.name = "br24"} : <i32>
    %113 = br %96 {handshake.bb = 7 : ui32, handshake.name = "br25"} : <i32>
    %114 = br %97 {handshake.bb = 7 : ui32, handshake.name = "br26"} : <i32>
    %115 = br %95 {handshake.bb = 7 : ui32, handshake.name = "br27"} : <i32>
    %116 = br %result_53 {handshake.bb = 7 : ui32, handshake.name = "br28"} : <>
    %117 = mux %index_61 [%112, %trueResult_71] {handshake.bb = 8 : ui32, handshake.name = "mux17"} : <i1>, [<i32>, <i32>] to <i32>
    %118 = mux %index_61 [%113, %trueResult_73] {handshake.bb = 8 : ui32, handshake.name = "mux18"} : <i1>, [<i32>, <i32>] to <i32>
    %119 = mux %index_61 [%114, %trueResult_75] {handshake.bb = 8 : ui32, handshake.name = "mux19"} : <i1>, [<i32>, <i32>] to <i32>
    %120 = mux %index_61 [%115, %trueResult_77] {handshake.bb = 8 : ui32, handshake.name = "mux20"} : <i1>, [<i32>, <i32>] to <i32>
    %result_60, %index_61 = control_merge [%116, %trueResult_79]  {handshake.bb = 8 : ui32, handshake.name = "control_merge7"} : [<>, <>] to <>, <i1>
    %121 = source {handshake.bb = 8 : ui32, handshake.name = "source11"} : <>
    %122 = constant %121 {handshake.bb = 8 : ui32, handshake.name = "constant22", value = 10 : i32} : <>, <i32>
    %123 = source {handshake.bb = 8 : ui32, handshake.name = "source12"} : <>
    %124 = constant %123 {handshake.bb = 8 : ui32, handshake.name = "constant23", value = 1 : i32} : <>, <i32>
    %125 = source {handshake.bb = 8 : ui32, handshake.name = "source13"} : <>
    %126 = constant %125 {handshake.bb = 8 : ui32, handshake.name = "constant24", value = 3 : i32} : <>, <i32>
    %127 = shli %119, %124 {handshake.bb = 8 : ui32, handshake.name = "shli14"} : <i32>
    %128 = shli %119, %126 {handshake.bb = 8 : ui32, handshake.name = "shli15"} : <i32>
    %129 = addi %127, %128 {handshake.bb = 8 : ui32, handshake.name = "addi26"} : <i32>
    %130 = addi %117, %129 {handshake.bb = 8 : ui32, handshake.name = "addi9"} : <i32>
    %addressResult_62, %dataResult_63 = load[%130] %1#3 {handshake.bb = 8 : ui32, handshake.name = "load4"} : <i32>, <i32>, <i32>, <i32>
    %131 = shli %117, %124 {handshake.bb = 8 : ui32, handshake.name = "shli16"} : <i32>
    %132 = shli %117, %126 {handshake.bb = 8 : ui32, handshake.name = "shli17"} : <i32>
    %133 = addi %131, %132 {handshake.bb = 8 : ui32, handshake.name = "addi27"} : <i32>
    %134 = addi %120, %133 {handshake.bb = 8 : ui32, handshake.name = "addi10"} : <i32>
    %addressResult_64, %dataResult_65 = load[%134] %outputs {handshake.bb = 8 : ui32, handshake.name = "load5"} : <i32>, <i32>, <i32>, <i32>
    %135 = muli %dataResult_63, %dataResult_65 {handshake.bb = 8 : ui32, handshake.name = "muli3"} : <i32>
    %136 = shli %119, %124 {handshake.bb = 8 : ui32, handshake.name = "shli18"} : <i32>
    %137 = shli %119, %126 {handshake.bb = 8 : ui32, handshake.name = "shli19"} : <i32>
    %138 = addi %136, %137 {handshake.bb = 8 : ui32, handshake.name = "addi28"} : <i32>
    %139 = addi %120, %138 {handshake.bb = 8 : ui32, handshake.name = "addi11"} : <i32>
    %addressResult_66, %dataResult_67 = load[%139] %0#2 {handshake.bb = 8 : ui32, handshake.deps = #handshake<deps[["store3", 3, true], ["store3", 4, true]]>, handshake.name = "load6"} : <i32>, <i32>, <i32>, <i32>
    %140 = addi %dataResult_67, %135 {handshake.bb = 8 : ui32, handshake.name = "addi1"} : <i32>
    %141 = shli %119, %124 {handshake.bb = 8 : ui32, handshake.name = "shli20"} : <i32>
    %142 = shli %119, %126 {handshake.bb = 8 : ui32, handshake.name = "shli21"} : <i32>
    %143 = addi %141, %142 {handshake.bb = 8 : ui32, handshake.name = "addi29"} : <i32>
    %144 = addi %120, %143 {handshake.bb = 8 : ui32, handshake.name = "addi12"} : <i32>
    %addressResult_68, %dataResult_69, %doneResult_70 = store[%144] %140 %0#3 {handshake.bb = 8 : ui32, handshake.deps = #handshake<deps[["load6", 3, true], ["store3", 3, true]]>, handshake.name = "store3"} : <i32>, <i32>, <>, <i32>, <i32>, <>
    %145 = addi %117, %124 {handshake.bb = 8 : ui32, handshake.name = "addi16"} : <i32>
    %146 = cmpi ult, %145, %122 {handshake.bb = 8 : ui32, handshake.name = "cmpi3"} : <i32>
    %trueResult_71, %falseResult_72 = cond_br %146, %145 {handshake.bb = 8 : ui32, handshake.name = "cond_br22"} : <i1>, <i32>
    %trueResult_73, %falseResult_74 = cond_br %146, %118 {handshake.bb = 8 : ui32, handshake.name = "cond_br23"} : <i1>, <i32>
    %trueResult_75, %falseResult_76 = cond_br %146, %119 {handshake.bb = 8 : ui32, handshake.name = "cond_br24"} : <i1>, <i32>
    %trueResult_77, %falseResult_78 = cond_br %146, %120 {handshake.bb = 8 : ui32, handshake.name = "cond_br25"} : <i1>, <i32>
    %trueResult_79, %falseResult_80 = cond_br %146, %result_60 {handshake.bb = 8 : ui32, handshake.name = "cond_br26"} : <i1>, <>
    %147 = merge %falseResult_74 {handshake.bb = 9 : ui32, handshake.name = "merge7"} : <i32>
    %148 = merge %falseResult_76 {handshake.bb = 9 : ui32, handshake.name = "merge8"} : <i32>
    %149 = merge %falseResult_78 {handshake.bb = 9 : ui32, handshake.name = "merge9"} : <i32>
    %result_81, %index_82 = control_merge [%falseResult_80]  {handshake.bb = 9 : ui32, handshake.name = "control_merge8"} : [<>] to <>, <i1>
    %150 = source {handshake.bb = 9 : ui32, handshake.name = "source14"} : <>
    %151 = constant %150 {handshake.bb = 9 : ui32, handshake.name = "constant25", value = 10 : i32} : <>, <i32>
    %152 = source {handshake.bb = 9 : ui32, handshake.name = "source15"} : <>
    %153 = constant %152 {handshake.bb = 9 : ui32, handshake.name = "constant26", value = 1 : i32} : <>, <i32>
    %154 = addi %149, %153 {handshake.bb = 9 : ui32, handshake.name = "addi17"} : <i32>
    %155 = cmpi ult, %154, %151 {handshake.bb = 9 : ui32, handshake.name = "cmpi4"} : <i32>
    %trueResult_83, %falseResult_84 = cond_br %155, %154 {handshake.bb = 9 : ui32, handshake.name = "cond_br27"} : <i1>, <i32>
    %trueResult_85, %falseResult_86 = cond_br %155, %147 {handshake.bb = 9 : ui32, handshake.name = "cond_br28"} : <i1>, <i32>
    %trueResult_87, %falseResult_88 = cond_br %155, %148 {handshake.bb = 9 : ui32, handshake.name = "cond_br29"} : <i1>, <i32>
    %trueResult_89, %falseResult_90 = cond_br %155, %result_81 {handshake.bb = 9 : ui32, handshake.name = "cond_br30"} : <i1>, <>
    %156 = merge %falseResult_86 {handshake.bb = 10 : ui32, handshake.name = "merge10"} : <i32>
    %157 = merge %falseResult_88 {handshake.bb = 10 : ui32, handshake.name = "merge11"} : <i32>
    %result_91, %index_92 = control_merge [%falseResult_90]  {handshake.bb = 10 : ui32, handshake.name = "control_merge9"} : [<>] to <>, <i1>
    %158 = source {handshake.bb = 10 : ui32, handshake.name = "source16"} : <>
    %159 = constant %158 {handshake.bb = 10 : ui32, handshake.name = "constant27", value = 10 : i32} : <>, <i32>
    %160 = source {handshake.bb = 10 : ui32, handshake.name = "source17"} : <>
    %161 = constant %160 {handshake.bb = 10 : ui32, handshake.name = "constant28", value = 1 : i32} : <>, <i32>
    %162 = addi %157, %161 {handshake.bb = 10 : ui32, handshake.name = "addi18"} : <i32>
    %163 = cmpi ult, %162, %159 {handshake.bb = 10 : ui32, handshake.name = "cmpi5"} : <i32>
    %trueResult_93, %falseResult_94 = cond_br %163, %162 {handshake.bb = 10 : ui32, handshake.name = "cond_br31"} : <i1>, <i32>
    %trueResult_95, %falseResult_96 = cond_br %163, %156 {handshake.bb = 10 : ui32, handshake.name = "cond_br32"} : <i1>, <i32>
    %trueResult_97, %falseResult_98 = cond_br %163, %result_91 {handshake.bb = 10 : ui32, handshake.name = "cond_br33"} : <i1>, <>
    %result_99, %index_100 = control_merge [%falseResult_98]  {handshake.bb = 11 : ui32, handshake.name = "control_merge10"} : [<>] to <>, <i1>
    end {handshake.bb = 11 : ui32, handshake.name = "end0"} %1#4, %memEnd_3, %memEnd_1, %memEnd, %0#4, %arg12 : <>, <>, <>, <>, <>, <>
  }
}

