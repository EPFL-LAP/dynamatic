module {
  handshake.func @kernel_2mm(%arg0: !handshake.channel<i32>, %arg1: !handshake.channel<i32>, %arg2: memref<100xi32>, %arg3: memref<100xi32>, %arg4: memref<100xi32>, %arg5: memref<100xi32>, %arg6: memref<100xi32>, %arg7: !handshake.control<>, %arg8: !handshake.control<>, %arg9: !handshake.control<>, %arg10: !handshake.control<>, %arg11: !handshake.control<>, %arg12: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["alpha", "beta", "tmp", "A", "B", "C", "D", "tmp_start", "A_start", "B_start", "C_start", "D_start", "start"], resNames = ["tmp_end", "A_end", "B_end", "C_end", "D_end", "end"]} {
    %0:3 = lsq[%arg6 : memref<100xi32>] (%arg11, %result_52, %addressResult_54, %addressResult_56, %dataResult_57, %result_58, %addressResult_64, %addressResult_66, %dataResult_67, %result_96)  {groupSizes = [2 : i32, 2 : i32], handshake.name = "lsq0"} : (!handshake.control<>, !handshake.control<>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>) -> (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>)
    %outputs, %memEnd = mem_controller[%arg5 : memref<100xi32>] %arg10 (%addressResult_62) %result_96 {connectedBlocks = [8 : i32], handshake.name = "mem_controller0"} :    (!handshake.channel<i32>) -> !handshake.channel<i32>
    %outputs_0, %memEnd_1 = mem_controller[%arg4 : memref<100xi32>] %arg9 (%addressResult_10) %result_96 {connectedBlocks = [3 : i32], handshake.name = "mem_controller1"} :    (!handshake.channel<i32>) -> !handshake.channel<i32>
    %outputs_2, %memEnd_3 = mem_controller[%arg3 : memref<100xi32>] %arg8 (%addressResult_8) %result_96 {connectedBlocks = [3 : i32], handshake.name = "mem_controller2"} :    (!handshake.channel<i32>) -> !handshake.channel<i32>
    %1:3 = lsq[%arg2 : memref<100xi32>] (%arg7, %result_4, %addressResult, %dataResult, %result_6, %addressResult_12, %addressResult_14, %dataResult_15, %result_58, %addressResult_60, %result_96)  {groupSizes = [1 : i32, 2 : i32, 1 : i32], handshake.name = "lsq1"} : (!handshake.control<>, !handshake.control<>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>, !handshake.channel<i32>, !handshake.control<>) -> (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>)
    %2 = constant %arg12 {handshake.bb = 0 : ui32, handshake.name = "constant3", value = 0 : i32} : <>, <i32>
    %3 = br %2 {handshake.bb = 0 : ui32, handshake.name = "br3"} : <i32>
    %4 = br %arg0 {handshake.bb = 0 : ui32, handshake.name = "br6"} : <i32>
    %5 = br %arg1 {handshake.bb = 0 : ui32, handshake.name = "br7"} : <i32>
    %6 = br %arg12 {handshake.bb = 0 : ui32, handshake.name = "br8"} : <>
    %7 = mux %index [%3, %trueResult_40] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i32>, <i32>] to <i32>
    %8 = mux %index [%4, %trueResult_42] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i1>, [<i32>, <i32>] to <i32>
    %9 = mux %index [%5, %trueResult_44] {handshake.bb = 1 : ui32, handshake.name = "mux2"} : <i1>, [<i32>, <i32>] to <i32>
    %result, %index = control_merge [%6, %trueResult_46]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %10 = constant %result {handshake.bb = 1 : ui32, handshake.name = "constant5", value = 0 : i32} : <>, <i32>
    %11 = br %10 {handshake.bb = 1 : ui32, handshake.name = "br9"} : <i32>
    %12 = br %8 {handshake.bb = 1 : ui32, handshake.name = "br10"} : <i32>
    %13 = br %9 {handshake.bb = 1 : ui32, handshake.name = "br11"} : <i32>
    %14 = br %7 {handshake.bb = 1 : ui32, handshake.name = "br12"} : <i32>
    %15 = br %result {handshake.bb = 1 : ui32, handshake.name = "br13"} : <>
    %16 = mux %index_5 [%11, %trueResult_28] {handshake.bb = 2 : ui32, handshake.name = "mux3"} : <i1>, [<i32>, <i32>] to <i32>
    %17 = mux %index_5 [%12, %trueResult_30] {handshake.bb = 2 : ui32, handshake.name = "mux4"} : <i1>, [<i32>, <i32>] to <i32>
    %18 = mux %index_5 [%13, %trueResult_32] {handshake.bb = 2 : ui32, handshake.name = "mux5"} : <i1>, [<i32>, <i32>] to <i32>
    %19 = mux %index_5 [%14, %trueResult_34] {handshake.bb = 2 : ui32, handshake.name = "mux6"} : <i1>, [<i32>, <i32>] to <i32>
    %result_4, %index_5 = control_merge [%15, %trueResult_36]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>, <>] to <>, <i1>
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
    %addressResult, %dataResult = store[%29] %20 {handshake.bb = 2 : ui32, handshake.deps = #handshake<deps[["load2", 3], ["store1", 3], ["load4", 1]]>, handshake.name = "store0"} : <i32>, <i32>, <i32>, <i32>
    %30 = br %21 {handshake.bb = 2 : ui32, handshake.name = "br14"} : <i32>
    %31 = br %17 {handshake.bb = 2 : ui32, handshake.name = "br15"} : <i32>
    %32 = br %18 {handshake.bb = 2 : ui32, handshake.name = "br16"} : <i32>
    %33 = br %19 {handshake.bb = 2 : ui32, handshake.name = "br17"} : <i32>
    %34 = br %16 {handshake.bb = 2 : ui32, handshake.name = "br18"} : <i32>
    %35 = br %result_4 {handshake.bb = 2 : ui32, handshake.name = "br19"} : <>
    %36 = mux %index_7 [%30, %trueResult] {handshake.bb = 3 : ui32, handshake.name = "mux7"} : <i1>, [<i32>, <i32>] to <i32>
    %37 = mux %index_7 [%31, %trueResult_16] {handshake.bb = 3 : ui32, handshake.name = "mux8"} : <i1>, [<i32>, <i32>] to <i32>
    %38 = mux %index_7 [%32, %trueResult_18] {handshake.bb = 3 : ui32, handshake.name = "mux9"} : <i1>, [<i32>, <i32>] to <i32>
    %39 = mux %index_7 [%33, %trueResult_20] {handshake.bb = 3 : ui32, handshake.name = "mux10"} : <i1>, [<i32>, <i32>] to <i32>
    %40 = mux %index_7 [%34, %trueResult_22] {handshake.bb = 3 : ui32, handshake.name = "mux11"} : <i1>, [<i32>, <i32>] to <i32>
    %result_6, %index_7 = control_merge [%35, %trueResult_24]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>, <>] to <>, <i1>
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
    %addressResult_12, %dataResult_13 = load[%60] %1#0 {handshake.bb = 3 : ui32, handshake.deps = #handshake<deps[["store1", 3], ["store1", 4]]>, handshake.name = "load2"} : <i32>, <i32>, <i32>, <i32>
    %61 = addi %dataResult_13, %56 {handshake.bb = 3 : ui32, handshake.name = "addi0"} : <i32>
    %62 = shli %39, %44 {handshake.bb = 3 : ui32, handshake.name = "shli8"} : <i32>
    %63 = shli %39, %46 {handshake.bb = 3 : ui32, handshake.name = "shli9"} : <i32>
    %64 = addi %62, %63 {handshake.bb = 3 : ui32, handshake.name = "addi23"} : <i32>
    %65 = addi %40, %64 {handshake.bb = 3 : ui32, handshake.name = "addi6"} : <i32>
    %addressResult_14, %dataResult_15 = store[%65] %61 {handshake.bb = 3 : ui32, handshake.deps = #handshake<deps[["load2", 3], ["store1", 3], ["load4", 1]]>, handshake.name = "store1"} : <i32>, <i32>, <i32>, <i32>
    %66 = addi %36, %44 {handshake.bb = 3 : ui32, handshake.name = "addi13"} : <i32>
    %67 = cmpi ult, %66, %42 {handshake.bb = 3 : ui32, handshake.name = "cmpi0"} : <i32>
    %trueResult, %falseResult = cond_br %67, %66 {handshake.bb = 3 : ui32, handshake.name = "cond_br6"} : <i1>, <i32>
    %trueResult_16, %falseResult_17 = cond_br %67, %37 {handshake.bb = 3 : ui32, handshake.name = "cond_br7"} : <i1>, <i32>
    %trueResult_18, %falseResult_19 = cond_br %67, %38 {handshake.bb = 3 : ui32, handshake.name = "cond_br8"} : <i1>, <i32>
    %trueResult_20, %falseResult_21 = cond_br %67, %39 {handshake.bb = 3 : ui32, handshake.name = "cond_br9"} : <i1>, <i32>
    %trueResult_22, %falseResult_23 = cond_br %67, %40 {handshake.bb = 3 : ui32, handshake.name = "cond_br10"} : <i1>, <i32>
    %trueResult_24, %falseResult_25 = cond_br %67, %result_6 {handshake.bb = 3 : ui32, handshake.name = "cond_br11"} : <i1>, <>
    %68 = merge %falseResult_17 {handshake.bb = 4 : ui32, handshake.name = "merge0"} : <i32>
    %69 = merge %falseResult_19 {handshake.bb = 4 : ui32, handshake.name = "merge1"} : <i32>
    %70 = merge %falseResult_21 {handshake.bb = 4 : ui32, handshake.name = "merge2"} : <i32>
    %71 = merge %falseResult_23 {handshake.bb = 4 : ui32, handshake.name = "merge3"} : <i32>
    %result_26, %index_27 = control_merge [%falseResult_25]  {handshake.bb = 4 : ui32, handshake.name = "control_merge3"} : [<>] to <>, <i1>
    %72 = source {handshake.bb = 4 : ui32, handshake.name = "source5"} : <>
    %73 = constant %72 {handshake.bb = 4 : ui32, handshake.name = "constant13", value = 10 : i32} : <>, <i32>
    %74 = source {handshake.bb = 4 : ui32, handshake.name = "source6"} : <>
    %75 = constant %74 {handshake.bb = 4 : ui32, handshake.name = "constant14", value = 1 : i32} : <>, <i32>
    %76 = addi %71, %75 {handshake.bb = 4 : ui32, handshake.name = "addi14"} : <i32>
    %77 = cmpi ult, %76, %73 {handshake.bb = 4 : ui32, handshake.name = "cmpi1"} : <i32>
    %trueResult_28, %falseResult_29 = cond_br %77, %76 {handshake.bb = 4 : ui32, handshake.name = "cond_br12"} : <i1>, <i32>
    %trueResult_30, %falseResult_31 = cond_br %77, %68 {handshake.bb = 4 : ui32, handshake.name = "cond_br13"} : <i1>, <i32>
    %trueResult_32, %falseResult_33 = cond_br %77, %69 {handshake.bb = 4 : ui32, handshake.name = "cond_br14"} : <i1>, <i32>
    %trueResult_34, %falseResult_35 = cond_br %77, %70 {handshake.bb = 4 : ui32, handshake.name = "cond_br15"} : <i1>, <i32>
    %trueResult_36, %falseResult_37 = cond_br %77, %result_26 {handshake.bb = 4 : ui32, handshake.name = "cond_br16"} : <i1>, <>
    %78 = merge %falseResult_31 {handshake.bb = 5 : ui32, handshake.name = "merge4"} : <i32>
    %79 = merge %falseResult_33 {handshake.bb = 5 : ui32, handshake.name = "merge5"} : <i32>
    %80 = merge %falseResult_35 {handshake.bb = 5 : ui32, handshake.name = "merge6"} : <i32>
    %result_38, %index_39 = control_merge [%falseResult_37]  {handshake.bb = 5 : ui32, handshake.name = "control_merge4"} : [<>] to <>, <i1>
    %81 = constant %result_38 {handshake.bb = 5 : ui32, handshake.name = "constant15", value = 0 : i32} : <>, <i32>
    %82 = source {handshake.bb = 5 : ui32, handshake.name = "source7"} : <>
    %83 = constant %82 {handshake.bb = 5 : ui32, handshake.name = "constant16", value = 10 : i32} : <>, <i32>
    %84 = source {handshake.bb = 5 : ui32, handshake.name = "source8"} : <>
    %85 = constant %84 {handshake.bb = 5 : ui32, handshake.name = "constant17", value = 1 : i32} : <>, <i32>
    %86 = addi %80, %85 {handshake.bb = 5 : ui32, handshake.name = "addi15"} : <i32>
    %87 = cmpi ult, %86, %83 {handshake.bb = 5 : ui32, handshake.name = "cmpi2"} : <i32>
    %trueResult_40, %falseResult_41 = cond_br %87, %86 {handshake.bb = 5 : ui32, handshake.name = "cond_br17"} : <i1>, <i32>
    %trueResult_42, %falseResult_43 = cond_br %87, %78 {handshake.bb = 5 : ui32, handshake.name = "cond_br18"} : <i1>, <i32>
    %trueResult_44, %falseResult_45 = cond_br %87, %79 {handshake.bb = 5 : ui32, handshake.name = "cond_br19"} : <i1>, <i32>
    %trueResult_46, %falseResult_47 = cond_br %87, %result_38 {handshake.bb = 5 : ui32, handshake.name = "cond_br20"} : <i1>, <>
    %trueResult_48, %falseResult_49 = cond_br %87, %81 {handshake.bb = 5 : ui32, handshake.name = "cond_br21"} : <i1>, <i32>
    %88 = mux %index_51 [%falseResult_49, %trueResult_90] {handshake.bb = 6 : ui32, handshake.name = "mux12"} : <i1>, [<i32>, <i32>] to <i32>
    %89 = mux %index_51 [%falseResult_45, %trueResult_92] {handshake.bb = 6 : ui32, handshake.name = "mux13"} : <i1>, [<i32>, <i32>] to <i32>
    %result_50, %index_51 = control_merge [%falseResult_47, %trueResult_94]  {handshake.bb = 6 : ui32, handshake.name = "control_merge5"} : [<>, <>] to <>, <i1>
    %90 = constant %result_50 {handshake.bb = 6 : ui32, handshake.name = "constant18", value = 0 : i32} : <>, <i32>
    %91 = br %90 {handshake.bb = 6 : ui32, handshake.name = "br20"} : <i32>
    %92 = br %89 {handshake.bb = 6 : ui32, handshake.name = "br21"} : <i32>
    %93 = br %88 {handshake.bb = 6 : ui32, handshake.name = "br22"} : <i32>
    %94 = br %result_50 {handshake.bb = 6 : ui32, handshake.name = "br23"} : <>
    %95 = mux %index_53 [%91, %trueResult_80] {handshake.bb = 7 : ui32, handshake.name = "mux14"} : <i1>, [<i32>, <i32>] to <i32>
    %96 = mux %index_53 [%92, %trueResult_82] {handshake.bb = 7 : ui32, handshake.name = "mux15"} : <i1>, [<i32>, <i32>] to <i32>
    %97 = mux %index_53 [%93, %trueResult_84] {handshake.bb = 7 : ui32, handshake.name = "mux16"} : <i1>, [<i32>, <i32>] to <i32>
    %result_52, %index_53 = control_merge [%94, %trueResult_86]  {handshake.bb = 7 : ui32, handshake.name = "control_merge6"} : [<>, <>] to <>, <i1>
    %98 = constant %result_52 {handshake.bb = 7 : ui32, handshake.name = "constant19", value = 0 : i32} : <>, <i32>
    %99 = source {handshake.bb = 7 : ui32, handshake.name = "source9"} : <>
    %100 = constant %99 {handshake.bb = 7 : ui32, handshake.name = "constant20", value = 1 : i32} : <>, <i32>
    %101 = source {handshake.bb = 7 : ui32, handshake.name = "source10"} : <>
    %102 = constant %101 {handshake.bb = 7 : ui32, handshake.name = "constant21", value = 3 : i32} : <>, <i32>
    %103 = shli %97, %100 {handshake.bb = 7 : ui32, handshake.name = "shli10"} : <i32>
    %104 = shli %97, %102 {handshake.bb = 7 : ui32, handshake.name = "shli11"} : <i32>
    %105 = addi %103, %104 {handshake.bb = 7 : ui32, handshake.name = "addi24"} : <i32>
    %106 = addi %95, %105 {handshake.bb = 7 : ui32, handshake.name = "addi7"} : <i32>
    %addressResult_54, %dataResult_55 = load[%106] %0#0 {handshake.bb = 7 : ui32, handshake.deps = #handshake<deps[["store2", 3], ["store3", 3]]>, handshake.name = "load3"} : <i32>, <i32>, <i32>, <i32>
    %107 = muli %dataResult_55, %96 {handshake.bb = 7 : ui32, handshake.name = "muli2"} : <i32>
    %108 = shli %97, %100 {handshake.bb = 7 : ui32, handshake.name = "shli12"} : <i32>
    %109 = shli %97, %102 {handshake.bb = 7 : ui32, handshake.name = "shli13"} : <i32>
    %110 = addi %108, %109 {handshake.bb = 7 : ui32, handshake.name = "addi25"} : <i32>
    %111 = addi %95, %110 {handshake.bb = 7 : ui32, handshake.name = "addi8"} : <i32>
    %addressResult_56, %dataResult_57 = store[%111] %107 {handshake.bb = 7 : ui32, handshake.deps = #handshake<deps[["load6", 3], ["store3", 3]]>, handshake.name = "store2"} : <i32>, <i32>, <i32>, <i32>
    %112 = br %98 {handshake.bb = 7 : ui32, handshake.name = "br24"} : <i32>
    %113 = br %96 {handshake.bb = 7 : ui32, handshake.name = "br25"} : <i32>
    %114 = br %97 {handshake.bb = 7 : ui32, handshake.name = "br26"} : <i32>
    %115 = br %95 {handshake.bb = 7 : ui32, handshake.name = "br27"} : <i32>
    %116 = br %result_52 {handshake.bb = 7 : ui32, handshake.name = "br28"} : <>
    %117 = mux %index_59 [%112, %trueResult_68] {handshake.bb = 8 : ui32, handshake.name = "mux17"} : <i1>, [<i32>, <i32>] to <i32>
    %118 = mux %index_59 [%113, %trueResult_70] {handshake.bb = 8 : ui32, handshake.name = "mux18"} : <i1>, [<i32>, <i32>] to <i32>
    %119 = mux %index_59 [%114, %trueResult_72] {handshake.bb = 8 : ui32, handshake.name = "mux19"} : <i1>, [<i32>, <i32>] to <i32>
    %120 = mux %index_59 [%115, %trueResult_74] {handshake.bb = 8 : ui32, handshake.name = "mux20"} : <i1>, [<i32>, <i32>] to <i32>
    %result_58, %index_59 = control_merge [%116, %trueResult_76]  {handshake.bb = 8 : ui32, handshake.name = "control_merge7"} : [<>, <>] to <>, <i1>
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
    %addressResult_60, %dataResult_61 = load[%130] %1#1 {handshake.bb = 8 : ui32, handshake.name = "load4"} : <i32>, <i32>, <i32>, <i32>
    %131 = shli %117, %124 {handshake.bb = 8 : ui32, handshake.name = "shli16"} : <i32>
    %132 = shli %117, %126 {handshake.bb = 8 : ui32, handshake.name = "shli17"} : <i32>
    %133 = addi %131, %132 {handshake.bb = 8 : ui32, handshake.name = "addi27"} : <i32>
    %134 = addi %120, %133 {handshake.bb = 8 : ui32, handshake.name = "addi10"} : <i32>
    %addressResult_62, %dataResult_63 = load[%134] %outputs {handshake.bb = 8 : ui32, handshake.name = "load5"} : <i32>, <i32>, <i32>, <i32>
    %135 = muli %dataResult_61, %dataResult_63 {handshake.bb = 8 : ui32, handshake.name = "muli3"} : <i32>
    %136 = shli %119, %124 {handshake.bb = 8 : ui32, handshake.name = "shli18"} : <i32>
    %137 = shli %119, %126 {handshake.bb = 8 : ui32, handshake.name = "shli19"} : <i32>
    %138 = addi %136, %137 {handshake.bb = 8 : ui32, handshake.name = "addi28"} : <i32>
    %139 = addi %120, %138 {handshake.bb = 8 : ui32, handshake.name = "addi11"} : <i32>
    %addressResult_64, %dataResult_65 = load[%139] %0#1 {handshake.bb = 8 : ui32, handshake.deps = #handshake<deps[["store3", 3], ["store3", 4]]>, handshake.name = "load6"} : <i32>, <i32>, <i32>, <i32>
    %140 = addi %dataResult_65, %135 {handshake.bb = 8 : ui32, handshake.name = "addi1"} : <i32>
    %141 = shli %119, %124 {handshake.bb = 8 : ui32, handshake.name = "shli20"} : <i32>
    %142 = shli %119, %126 {handshake.bb = 8 : ui32, handshake.name = "shli21"} : <i32>
    %143 = addi %141, %142 {handshake.bb = 8 : ui32, handshake.name = "addi29"} : <i32>
    %144 = addi %120, %143 {handshake.bb = 8 : ui32, handshake.name = "addi12"} : <i32>
    %addressResult_66, %dataResult_67 = store[%144] %140 {handshake.bb = 8 : ui32, handshake.deps = #handshake<deps[["load6", 3], ["store3", 3]]>, handshake.name = "store3"} : <i32>, <i32>, <i32>, <i32>
    %145 = addi %117, %124 {handshake.bb = 8 : ui32, handshake.name = "addi16"} : <i32>
    %146 = cmpi ult, %145, %122 {handshake.bb = 8 : ui32, handshake.name = "cmpi3"} : <i32>
    %trueResult_68, %falseResult_69 = cond_br %146, %145 {handshake.bb = 8 : ui32, handshake.name = "cond_br22"} : <i1>, <i32>
    %trueResult_70, %falseResult_71 = cond_br %146, %118 {handshake.bb = 8 : ui32, handshake.name = "cond_br23"} : <i1>, <i32>
    %trueResult_72, %falseResult_73 = cond_br %146, %119 {handshake.bb = 8 : ui32, handshake.name = "cond_br24"} : <i1>, <i32>
    %trueResult_74, %falseResult_75 = cond_br %146, %120 {handshake.bb = 8 : ui32, handshake.name = "cond_br25"} : <i1>, <i32>
    %trueResult_76, %falseResult_77 = cond_br %146, %result_58 {handshake.bb = 8 : ui32, handshake.name = "cond_br26"} : <i1>, <>
    %147 = merge %falseResult_71 {handshake.bb = 9 : ui32, handshake.name = "merge7"} : <i32>
    %148 = merge %falseResult_73 {handshake.bb = 9 : ui32, handshake.name = "merge8"} : <i32>
    %149 = merge %falseResult_75 {handshake.bb = 9 : ui32, handshake.name = "merge9"} : <i32>
    %result_78, %index_79 = control_merge [%falseResult_77]  {handshake.bb = 9 : ui32, handshake.name = "control_merge8"} : [<>] to <>, <i1>
    %150 = source {handshake.bb = 9 : ui32, handshake.name = "source14"} : <>
    %151 = constant %150 {handshake.bb = 9 : ui32, handshake.name = "constant25", value = 10 : i32} : <>, <i32>
    %152 = source {handshake.bb = 9 : ui32, handshake.name = "source15"} : <>
    %153 = constant %152 {handshake.bb = 9 : ui32, handshake.name = "constant26", value = 1 : i32} : <>, <i32>
    %154 = addi %149, %153 {handshake.bb = 9 : ui32, handshake.name = "addi17"} : <i32>
    %155 = cmpi ult, %154, %151 {handshake.bb = 9 : ui32, handshake.name = "cmpi4"} : <i32>
    %trueResult_80, %falseResult_81 = cond_br %155, %154 {handshake.bb = 9 : ui32, handshake.name = "cond_br27"} : <i1>, <i32>
    %trueResult_82, %falseResult_83 = cond_br %155, %147 {handshake.bb = 9 : ui32, handshake.name = "cond_br28"} : <i1>, <i32>
    %trueResult_84, %falseResult_85 = cond_br %155, %148 {handshake.bb = 9 : ui32, handshake.name = "cond_br29"} : <i1>, <i32>
    %trueResult_86, %falseResult_87 = cond_br %155, %result_78 {handshake.bb = 9 : ui32, handshake.name = "cond_br30"} : <i1>, <>
    %156 = merge %falseResult_83 {handshake.bb = 10 : ui32, handshake.name = "merge10"} : <i32>
    %157 = merge %falseResult_85 {handshake.bb = 10 : ui32, handshake.name = "merge11"} : <i32>
    %result_88, %index_89 = control_merge [%falseResult_87]  {handshake.bb = 10 : ui32, handshake.name = "control_merge9"} : [<>] to <>, <i1>
    %158 = source {handshake.bb = 10 : ui32, handshake.name = "source16"} : <>
    %159 = constant %158 {handshake.bb = 10 : ui32, handshake.name = "constant27", value = 10 : i32} : <>, <i32>
    %160 = source {handshake.bb = 10 : ui32, handshake.name = "source17"} : <>
    %161 = constant %160 {handshake.bb = 10 : ui32, handshake.name = "constant28", value = 1 : i32} : <>, <i32>
    %162 = addi %157, %161 {handshake.bb = 10 : ui32, handshake.name = "addi18"} : <i32>
    %163 = cmpi ult, %162, %159 {handshake.bb = 10 : ui32, handshake.name = "cmpi5"} : <i32>
    %trueResult_90, %falseResult_91 = cond_br %163, %162 {handshake.bb = 10 : ui32, handshake.name = "cond_br31"} : <i1>, <i32>
    %trueResult_92, %falseResult_93 = cond_br %163, %156 {handshake.bb = 10 : ui32, handshake.name = "cond_br32"} : <i1>, <i32>
    %trueResult_94, %falseResult_95 = cond_br %163, %result_88 {handshake.bb = 10 : ui32, handshake.name = "cond_br33"} : <i1>, <>
    %result_96, %index_97 = control_merge [%falseResult_95]  {handshake.bb = 11 : ui32, handshake.name = "control_merge10"} : [<>] to <>, <i1>
    end {handshake.bb = 11 : ui32, handshake.name = "end0"} %1#2, %memEnd_3, %memEnd_1, %memEnd, %0#2, %arg12 : <>, <>, <>, <>, <>, <>
  }
}

