module {
  handshake.func @kernel_2mm(%arg0: !handshake.channel<i32>, %arg1: !handshake.channel<i32>, %arg2: memref<100xi32>, %arg3: memref<100xi32>, %arg4: memref<100xi32>, %arg5: memref<100xi32>, %arg6: memref<100xi32>, %arg7: !handshake.control<>, %arg8: !handshake.control<>, %arg9: !handshake.control<>, %arg10: !handshake.control<>, %arg11: !handshake.control<>, %arg12: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["alpha", "beta", "tmp", "A", "B", "C", "D", "tmp_start", "A_start", "B_start", "C_start", "D_start", "start"], cfg.edges = "[0,1][7,8][2,3][9,7,10,cmpi4][4,2,5,cmpi1][6,7][1,2][8,8,9,cmpi3][3,3,4,cmpi0][10,6,11,cmpi5][5,1,6,cmpi2]", resNames = ["tmp_end", "A_end", "B_end", "C_end", "D_end", "end"]} {
    %outputs:4, %memEnd = mem_controller[%arg6 : memref<100xi32>] %arg11 (%118, %addressResult_71, %addressResult_73, %dataResult_74, %150, %addressResult_92, %addressResult_94, %dataResult_95) %result_137 {connectedBlocks = [7 : i32, 8 : i32], handshake.name = "mem_controller3"} :    (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>) -> (!handshake.channel<i32>, !handshake.control<>, !handshake.channel<i32>, !handshake.control<>)
    %outputs_0, %memEnd_1 = mem_controller[%arg5 : memref<100xi32>] %arg10 (%addressResult_90) %result_137 {connectedBlocks = [8 : i32], handshake.name = "mem_controller4"} :    (!handshake.channel<i32>) -> !handshake.channel<i32>
    %outputs_2, %memEnd_3 = mem_controller[%arg4 : memref<100xi32>] %arg9 (%addressResult_16) %result_137 {connectedBlocks = [3 : i32], handshake.name = "mem_controller5"} :    (!handshake.channel<i32>) -> !handshake.channel<i32>
    %outputs_4, %memEnd_5 = mem_controller[%arg3 : memref<100xi32>] %arg8 (%addressResult_14) %result_137 {connectedBlocks = [3 : i32], handshake.name = "mem_controller6"} :    (!handshake.channel<i32>) -> !handshake.channel<i32>
    %outputs_6:4, %memEnd_7 = mem_controller[%arg2 : memref<100xi32>] %arg7 (%22, %addressResult, %dataResult, %48, %addressResult_18, %addressResult_20, %dataResult_21, %addressResult_88) %result_137 {connectedBlocks = [2 : i32, 3 : i32, 8 : i32], handshake.name = "mem_controller7"} :    (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>) -> (!handshake.control<>, !handshake.channel<i32>, !handshake.control<>, !handshake.channel<i32>)
    %0 = constant %arg12 {handshake.bb = 0 : ui32, handshake.name = "constant3", value = 0 : i32} : <>, <i32>
    %1 = br %0 {handshake.bb = 0 : ui32, handshake.name = "br3"} : <i32>
    %2 = br %arg0 {handshake.bb = 0 : ui32, handshake.name = "br6"} : <i32>
    %3 = br %arg1 {handshake.bb = 0 : ui32, handshake.name = "br7"} : <i32>
    %4 = br %arg12 {handshake.bb = 0 : ui32, handshake.name = "br8"} : <>
    %5 = mux %6 [%arg12, %trueResult_53] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux25"} : <i1>, [<>, <>] to <>
    %6 = init %98 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "init0"} : <i1>
    %7 = mux %index [%1, %trueResult_57] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i32>, <i32>] to <i32>
    %8 = mux %index [%2, %trueResult_59] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i1>, [<i32>, <i32>] to <i32>
    %9 = mux %index [%3, %trueResult_61] {handshake.bb = 1 : ui32, handshake.name = "mux2"} : <i1>, [<i32>, <i32>] to <i32>
    %result, %index = control_merge [%4, %trueResult_63]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %10 = constant %result {handshake.bb = 1 : ui32, handshake.name = "constant5", value = 0 : i32} : <>, <i32>
    %11 = br %10 {handshake.bb = 1 : ui32, handshake.name = "br9"} : <i32>
    %12 = br %8 {handshake.bb = 1 : ui32, handshake.name = "br10"} : <i32>
    %13 = br %9 {handshake.bb = 1 : ui32, handshake.name = "br11"} : <i32>
    %14 = br %7 {handshake.bb = 1 : ui32, handshake.name = "br12"} : <i32>
    %15 = br %result {handshake.bb = 1 : ui32, handshake.name = "br13"} : <>
    %16 = mux %17 [%5, %trueResult_35] {ftd.phi, handshake.bb = 2 : ui32, handshake.name = "mux30"} : <i1>, [<>, <>] to <>
    %17 = init %88 {ftd.imerge, handshake.bb = 2 : ui32, handshake.name = "init5"} : <i1>
    %18 = mux %index_9 [%11, %trueResult_41] {handshake.bb = 2 : ui32, handshake.name = "mux3"} : <i1>, [<i32>, <i32>] to <i32>
    %19 = mux %index_9 [%12, %trueResult_43] {handshake.bb = 2 : ui32, handshake.name = "mux4"} : <i1>, [<i32>, <i32>] to <i32>
    %20 = mux %index_9 [%13, %trueResult_45] {handshake.bb = 2 : ui32, handshake.name = "mux5"} : <i1>, [<i32>, <i32>] to <i32>
    %21 = mux %index_9 [%14, %trueResult_47] {handshake.bb = 2 : ui32, handshake.name = "mux6"} : <i1>, [<i32>, <i32>] to <i32>
    %result_8, %index_9 = control_merge [%15, %trueResult_49]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>, <>] to <>, <i1>
    %22 = constant %result_8 {handshake.bb = 2 : ui32, handshake.name = "constant0", value = 1 : i32} : <>, <i32>
    %23 = constant %result_8 {handshake.bb = 2 : ui32, handshake.name = "constant6", value = 0 : i32} : <>, <i32>
    %24 = constant %result_8 {handshake.bb = 2 : ui32, handshake.name = "constant7", value = 0 : i32} : <>, <i32>
    %25 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %26 = constant %25 {handshake.bb = 2 : ui32, handshake.name = "constant8", value = 1 : i32} : <>, <i32>
    %27 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %28 = constant %27 {handshake.bb = 2 : ui32, handshake.name = "constant9", value = 3 : i32} : <>, <i32>
    %29 = shli %21, %26 {handshake.bb = 2 : ui32, handshake.name = "shli0"} : <i32>
    %30 = shli %21, %28 {handshake.bb = 2 : ui32, handshake.name = "shli1"} : <i32>
    %31 = addi %29, %30 {handshake.bb = 2 : ui32, handshake.name = "addi19"} : <i32>
    %32 = addi %18, %31 {handshake.bb = 2 : ui32, handshake.name = "addi2"} : <i32>
    %33 = buffer %doneResult, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer0"} : <>
    %addressResult, %dataResult, %doneResult = store[%32] %23 %outputs_6#0 {handshake.bb = 2 : ui32, handshake.deps = #handshake<deps[["load2", 3, false], ["store1", 3, false], ["load4", 1, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store0"} : <i32>, <i32>, <>, <i32>, <i32>, <>
    %34 = br %24 {handshake.bb = 2 : ui32, handshake.name = "br14"} : <i32>
    %35 = br %19 {handshake.bb = 2 : ui32, handshake.name = "br15"} : <i32>
    %36 = br %20 {handshake.bb = 2 : ui32, handshake.name = "br16"} : <i32>
    %37 = br %21 {handshake.bb = 2 : ui32, handshake.name = "br17"} : <i32>
    %38 = br %18 {handshake.bb = 2 : ui32, handshake.name = "br18"} : <i32>
    %39 = br %result_8 {handshake.bb = 2 : ui32, handshake.name = "br19"} : <>
    %trueResult, %falseResult = cond_br %78, %41 {handshake.bb = 3 : ui32, handshake.name = "cond_br83"} : <i1>, <>
    %trueResult_10, %falseResult_11 = cond_br %78, %75 {handshake.bb = 3 : ui32, handshake.name = "cond_br84"} : <i1>, <>
    %40 = init %78 {ftd.imerge, handshake.bb = 3 : ui32, handshake.name = "init10"} : <i1>
    %41 = mux %40 [%33, %trueResult] {ftd.regen, handshake.bb = 3 : ui32, handshake.name = "mux31"} : <i1>, [<>, <>] to <>
    %42 = mux %40 [%16, %trueResult_10] {ftd.phi, handshake.bb = 3 : ui32, handshake.name = "mux34"} : <i1>, [<>, <>] to <>
    %43 = mux %index_13 [%34, %trueResult_23] {handshake.bb = 3 : ui32, handshake.name = "mux7"} : <i1>, [<i32>, <i32>] to <i32>
    %44 = mux %index_13 [%35, %trueResult_25] {handshake.bb = 3 : ui32, handshake.name = "mux8"} : <i1>, [<i32>, <i32>] to <i32>
    %45 = mux %index_13 [%36, %trueResult_27] {handshake.bb = 3 : ui32, handshake.name = "mux9"} : <i1>, [<i32>, <i32>] to <i32>
    %46 = mux %index_13 [%37, %trueResult_29] {handshake.bb = 3 : ui32, handshake.name = "mux10"} : <i1>, [<i32>, <i32>] to <i32>
    %47 = mux %index_13 [%38, %trueResult_31] {handshake.bb = 3 : ui32, handshake.name = "mux11"} : <i1>, [<i32>, <i32>] to <i32>
    %result_12, %index_13 = control_merge [%39, %trueResult_33]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>, <>] to <>, <i1>
    %48 = constant %result_12 {handshake.bb = 3 : ui32, handshake.name = "constant1", value = 1 : i32} : <>, <i32>
    %49 = source {handshake.bb = 3 : ui32, handshake.name = "source2"} : <>
    %50 = constant %49 {handshake.bb = 3 : ui32, handshake.name = "constant10", value = 10 : i32} : <>, <i32>
    %51 = source {handshake.bb = 3 : ui32, handshake.name = "source3"} : <>
    %52 = constant %51 {handshake.bb = 3 : ui32, handshake.name = "constant11", value = 1 : i32} : <>, <i32>
    %53 = source {handshake.bb = 3 : ui32, handshake.name = "source4"} : <>
    %54 = constant %53 {handshake.bb = 3 : ui32, handshake.name = "constant12", value = 3 : i32} : <>, <i32>
    %55 = shli %46, %52 {handshake.bb = 3 : ui32, handshake.name = "shli2"} : <i32>
    %56 = shli %46, %54 {handshake.bb = 3 : ui32, handshake.name = "shli3"} : <i32>
    %57 = addi %55, %56 {handshake.bb = 3 : ui32, handshake.name = "addi20"} : <i32>
    %58 = addi %43, %57 {handshake.bb = 3 : ui32, handshake.name = "addi3"} : <i32>
    %addressResult_14, %dataResult_15 = load[%58] %outputs_4 {handshake.bb = 3 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i32>, <i32>, <i32>, <i32>
    %59 = muli %44, %dataResult_15 {handshake.bb = 3 : ui32, handshake.name = "muli0"} : <i32>
    %60 = shli %43, %52 {handshake.bb = 3 : ui32, handshake.name = "shli4"} : <i32>
    %61 = shli %43, %54 {handshake.bb = 3 : ui32, handshake.name = "shli5"} : <i32>
    %62 = addi %60, %61 {handshake.bb = 3 : ui32, handshake.name = "addi21"} : <i32>
    %63 = addi %47, %62 {handshake.bb = 3 : ui32, handshake.name = "addi4"} : <i32>
    %addressResult_16, %dataResult_17 = load[%63] %outputs_2 {handshake.bb = 3 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i32>, <i32>, <i32>, <i32>
    %64 = muli %59, %dataResult_17 {handshake.bb = 3 : ui32, handshake.name = "muli1"} : <i32>
    %65 = shli %46, %52 {handshake.bb = 3 : ui32, handshake.name = "shli6"} : <i32>
    %66 = shli %46, %54 {handshake.bb = 3 : ui32, handshake.name = "shli7"} : <i32>
    %67 = addi %65, %66 {handshake.bb = 3 : ui32, handshake.name = "addi22"} : <i32>
    %68 = addi %47, %67 {handshake.bb = 3 : ui32, handshake.name = "addi5"} : <i32>
    %69 = gate %68, %41, %42 {handshake.bb = 3 : ui32, handshake.name = "gate0"} : <i32>, !handshake.control<>, !handshake.control<> to <i32>
    %addressResult_18, %dataResult_19 = load[%69] %outputs_6#1 {handshake.bb = 3 : ui32, handshake.deps = #handshake<deps[["store1", 3, false], ["store1", 4, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load2"} : <i32>, <i32>, <i32>, <i32>
    %70 = addi %dataResult_19, %64 {handshake.bb = 3 : ui32, handshake.name = "addi0"} : <i32>
    %71 = shli %46, %52 {handshake.bb = 3 : ui32, handshake.name = "shli8"} : <i32>
    %72 = shli %46, %54 {handshake.bb = 3 : ui32, handshake.name = "shli9"} : <i32>
    %73 = addi %71, %72 {handshake.bb = 3 : ui32, handshake.name = "addi23"} : <i32>
    %74 = addi %47, %73 {handshake.bb = 3 : ui32, handshake.name = "addi6"} : <i32>
    %75 = buffer %doneResult_22, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer1"} : <>
    %76 = gate %74, %41 {handshake.bb = 3 : ui32, handshake.name = "gate1"} : <i32>, !handshake.control<> to <i32>
    %addressResult_20, %dataResult_21, %doneResult_22 = store[%76] %70 %outputs_6#2 {handshake.bb = 3 : ui32, handshake.deps = #handshake<deps[["load2", 3, false], ["store1", 3, false], ["load4", 1, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store1"} : <i32>, <i32>, <>, <i32>, <i32>, <>
    %77 = addi %43, %52 {handshake.bb = 3 : ui32, handshake.name = "addi13"} : <i32>
    %78 = cmpi ult, %77, %50 {handshake.bb = 3 : ui32, handshake.name = "cmpi0"} : <i32>
    %trueResult_23, %falseResult_24 = cond_br %78, %77 {handshake.bb = 3 : ui32, handshake.name = "cond_br6"} : <i1>, <i32>
    %trueResult_25, %falseResult_26 = cond_br %78, %44 {handshake.bb = 3 : ui32, handshake.name = "cond_br7"} : <i1>, <i32>
    %trueResult_27, %falseResult_28 = cond_br %78, %45 {handshake.bb = 3 : ui32, handshake.name = "cond_br8"} : <i1>, <i32>
    %trueResult_29, %falseResult_30 = cond_br %78, %46 {handshake.bb = 3 : ui32, handshake.name = "cond_br9"} : <i1>, <i32>
    %trueResult_31, %falseResult_32 = cond_br %78, %47 {handshake.bb = 3 : ui32, handshake.name = "cond_br10"} : <i1>, <i32>
    %trueResult_33, %falseResult_34 = cond_br %78, %result_12 {handshake.bb = 3 : ui32, handshake.name = "cond_br11"} : <i1>, <>
    %trueResult_35, %falseResult_36 = cond_br %88, %falseResult_11 {handshake.bb = 4 : ui32, handshake.name = "cond_br85"} : <i1>, <>
    %trueResult_37, %falseResult_38 = cond_br %88, %33 {handshake.bb = 4 : ui32, handshake.name = "cond_br86"} : <i1>, <>
    %79 = merge %falseResult_26 {handshake.bb = 4 : ui32, handshake.name = "merge0"} : <i32>
    %80 = merge %falseResult_28 {handshake.bb = 4 : ui32, handshake.name = "merge1"} : <i32>
    %81 = merge %falseResult_30 {handshake.bb = 4 : ui32, handshake.name = "merge2"} : <i32>
    %82 = merge %falseResult_32 {handshake.bb = 4 : ui32, handshake.name = "merge3"} : <i32>
    %result_39, %index_40 = control_merge [%falseResult_34]  {handshake.bb = 4 : ui32, handshake.name = "control_merge3"} : [<>] to <>, <i1>
    %83 = source {handshake.bb = 4 : ui32, handshake.name = "source5"} : <>
    %84 = constant %83 {handshake.bb = 4 : ui32, handshake.name = "constant13", value = 10 : i32} : <>, <i32>
    %85 = source {handshake.bb = 4 : ui32, handshake.name = "source6"} : <>
    %86 = constant %85 {handshake.bb = 4 : ui32, handshake.name = "constant14", value = 1 : i32} : <>, <i32>
    %87 = addi %82, %86 {handshake.bb = 4 : ui32, handshake.name = "addi14"} : <i32>
    %88 = cmpi ult, %87, %84 {handshake.bb = 4 : ui32, handshake.name = "cmpi1"} : <i32>
    %trueResult_41, %falseResult_42 = cond_br %88, %87 {handshake.bb = 4 : ui32, handshake.name = "cond_br12"} : <i1>, <i32>
    %trueResult_43, %falseResult_44 = cond_br %88, %79 {handshake.bb = 4 : ui32, handshake.name = "cond_br13"} : <i1>, <i32>
    %trueResult_45, %falseResult_46 = cond_br %88, %80 {handshake.bb = 4 : ui32, handshake.name = "cond_br14"} : <i1>, <i32>
    %trueResult_47, %falseResult_48 = cond_br %88, %81 {handshake.bb = 4 : ui32, handshake.name = "cond_br15"} : <i1>, <i32>
    %trueResult_49, %falseResult_50 = cond_br %88, %result_39 {handshake.bb = 4 : ui32, handshake.name = "cond_br16"} : <i1>, <>
    %trueResult_51, %falseResult_52 = cond_br %98, %falseResult_38 {handshake.bb = 5 : ui32, handshake.name = "cond_br87"} : <i1>, <>
    %trueResult_53, %falseResult_54 = cond_br %98, %falseResult_36 {handshake.bb = 5 : ui32, handshake.name = "cond_br88"} : <i1>, <>
    %89 = merge %falseResult_44 {handshake.bb = 5 : ui32, handshake.name = "merge4"} : <i32>
    %90 = merge %falseResult_46 {handshake.bb = 5 : ui32, handshake.name = "merge5"} : <i32>
    %91 = merge %falseResult_48 {handshake.bb = 5 : ui32, handshake.name = "merge6"} : <i32>
    %result_55, %index_56 = control_merge [%falseResult_50]  {handshake.bb = 5 : ui32, handshake.name = "control_merge4"} : [<>] to <>, <i1>
    %92 = constant %result_55 {handshake.bb = 5 : ui32, handshake.name = "constant15", value = 0 : i32} : <>, <i32>
    %93 = source {handshake.bb = 5 : ui32, handshake.name = "source7"} : <>
    %94 = constant %93 {handshake.bb = 5 : ui32, handshake.name = "constant16", value = 10 : i32} : <>, <i32>
    %95 = source {handshake.bb = 5 : ui32, handshake.name = "source8"} : <>
    %96 = constant %95 {handshake.bb = 5 : ui32, handshake.name = "constant17", value = 1 : i32} : <>, <i32>
    %97 = addi %91, %96 {handshake.bb = 5 : ui32, handshake.name = "addi15"} : <i32>
    %98 = cmpi ult, %97, %94 {handshake.bb = 5 : ui32, handshake.name = "cmpi2"} : <i32>
    %trueResult_57, %falseResult_58 = cond_br %98, %97 {handshake.bb = 5 : ui32, handshake.name = "cond_br17"} : <i1>, <i32>
    %trueResult_59, %falseResult_60 = cond_br %98, %89 {handshake.bb = 5 : ui32, handshake.name = "cond_br18"} : <i1>, <i32>
    %trueResult_61, %falseResult_62 = cond_br %98, %90 {handshake.bb = 5 : ui32, handshake.name = "cond_br19"} : <i1>, <i32>
    %trueResult_63, %falseResult_64 = cond_br %98, %result_55 {handshake.bb = 5 : ui32, handshake.name = "cond_br20"} : <i1>, <>
    %trueResult_65, %falseResult_66 = cond_br %98, %92 {handshake.bb = 5 : ui32, handshake.name = "cond_br21"} : <i1>, <i32>
    %99 = init %197 {ftd.imerge, handshake.bb = 6 : ui32, handshake.name = "init14"} : <i1>
    %100 = mux %99 [%falseResult_54, %trueResult_127] {ftd.regen, handshake.bb = 6 : ui32, handshake.name = "mux35"} : <i1>, [<>, <>] to <>
    %101 = mux %99 [%falseResult_52, %trueResult_125] {ftd.regen, handshake.bb = 6 : ui32, handshake.name = "mux36"} : <i1>, [<>, <>] to <>
    %102 = mux %99 [%arg12, %trueResult_123] {ftd.phi, handshake.bb = 6 : ui32, handshake.name = "mux40"} : <i1>, [<>, <>] to <>
    %103 = mux %index_68 [%falseResult_66, %trueResult_131] {handshake.bb = 6 : ui32, handshake.name = "mux12"} : <i1>, [<i32>, <i32>] to <i32>
    %104 = mux %index_68 [%falseResult_62, %trueResult_133] {handshake.bb = 6 : ui32, handshake.name = "mux13"} : <i1>, [<i32>, <i32>] to <i32>
    %result_67, %index_68 = control_merge [%falseResult_64, %trueResult_135]  {handshake.bb = 6 : ui32, handshake.name = "control_merge5"} : [<>, <>] to <>, <i1>
    %105 = constant %result_67 {handshake.bb = 6 : ui32, handshake.name = "constant18", value = 0 : i32} : <>, <i32>
    %106 = br %105 {handshake.bb = 6 : ui32, handshake.name = "br20"} : <i32>
    %107 = br %104 {handshake.bb = 6 : ui32, handshake.name = "br21"} : <i32>
    %108 = br %103 {handshake.bb = 6 : ui32, handshake.name = "br22"} : <i32>
    %109 = br %result_67 {handshake.bb = 6 : ui32, handshake.name = "br23"} : <>
    %110 = init %189 {ftd.imerge, handshake.bb = 7 : ui32, handshake.name = "init20"} : <i1>
    %111 = mux %110 [%100, %trueResult_109] {ftd.regen, handshake.bb = 7 : ui32, handshake.name = "mux41"} : <i1>, [<>, <>] to <>
    %112 = mux %110 [%101, %trueResult_107] {ftd.regen, handshake.bb = 7 : ui32, handshake.name = "mux42"} : <i1>, [<>, <>] to <>
    %113 = mux %110 [%102, %trueResult_111] {ftd.phi, handshake.bb = 7 : ui32, handshake.name = "mux46"} : <i1>, [<>, <>] to <>
    %114:2 = unbundle %dataResult_72  {handshake.bb = 7 : ui32, handshake.name = "unbundle2"} : <i32> to _ 
    %115 = mux %index_70 [%106, %trueResult_115] {handshake.bb = 7 : ui32, handshake.name = "mux14"} : <i1>, [<i32>, <i32>] to <i32>
    %116 = mux %index_70 [%107, %trueResult_117] {handshake.bb = 7 : ui32, handshake.name = "mux15"} : <i1>, [<i32>, <i32>] to <i32>
    %117 = mux %index_70 [%108, %trueResult_119] {handshake.bb = 7 : ui32, handshake.name = "mux16"} : <i1>, [<i32>, <i32>] to <i32>
    %result_69, %index_70 = control_merge [%109, %trueResult_121]  {handshake.bb = 7 : ui32, handshake.name = "control_merge6"} : [<>, <>] to <>, <i1>
    %118 = constant %result_69 {handshake.bb = 7 : ui32, handshake.name = "constant2", value = 1 : i32} : <>, <i32>
    %119 = constant %result_69 {handshake.bb = 7 : ui32, handshake.name = "constant19", value = 0 : i32} : <>, <i32>
    %120 = source {handshake.bb = 7 : ui32, handshake.name = "source9"} : <>
    %121 = constant %120 {handshake.bb = 7 : ui32, handshake.name = "constant20", value = 1 : i32} : <>, <i32>
    %122 = source {handshake.bb = 7 : ui32, handshake.name = "source10"} : <>
    %123 = constant %122 {handshake.bb = 7 : ui32, handshake.name = "constant21", value = 3 : i32} : <>, <i32>
    %124 = shli %117, %121 {handshake.bb = 7 : ui32, handshake.name = "shli10"} : <i32>
    %125 = shli %117, %123 {handshake.bb = 7 : ui32, handshake.name = "shli11"} : <i32>
    %126 = addi %124, %125 {handshake.bb = 7 : ui32, handshake.name = "addi24"} : <i32>
    %127 = addi %115, %126 {handshake.bb = 7 : ui32, handshake.name = "addi7"} : <i32>
    %128 = buffer %114#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 7 : ui32, handshake.name = "buffer2"} : <>
    %addressResult_71, %dataResult_72 = load[%127] %outputs#0 {handshake.bb = 7 : ui32, handshake.deps = #handshake<deps[["store2", 3, false], ["store3", 3, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load3"} : <i32>, <i32>, <i32>, <i32>
    %129 = muli %dataResult_72, %116 {handshake.bb = 7 : ui32, handshake.name = "muli2"} : <i32>
    %130 = shli %117, %121 {handshake.bb = 7 : ui32, handshake.name = "shli12"} : <i32>
    %131 = shli %117, %123 {handshake.bb = 7 : ui32, handshake.name = "shli13"} : <i32>
    %132 = addi %130, %131 {handshake.bb = 7 : ui32, handshake.name = "addi25"} : <i32>
    %133 = addi %115, %132 {handshake.bb = 7 : ui32, handshake.name = "addi8"} : <i32>
    %134 = buffer %doneResult_75, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 7 : ui32, handshake.name = "buffer3"} : <>
    %addressResult_73, %dataResult_74, %doneResult_75 = store[%133] %129 %outputs#1 {handshake.bb = 7 : ui32, handshake.deps = #handshake<deps[["load6", 3, false], ["store3", 3, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store2"} : <i32>, <i32>, <>, <i32>, <i32>, <>
    %135 = br %119 {handshake.bb = 7 : ui32, handshake.name = "br24"} : <i32>
    %136 = br %116 {handshake.bb = 7 : ui32, handshake.name = "br25"} : <i32>
    %137 = br %117 {handshake.bb = 7 : ui32, handshake.name = "br26"} : <i32>
    %138 = br %115 {handshake.bb = 7 : ui32, handshake.name = "br27"} : <i32>
    %139 = br %result_69 {handshake.bb = 7 : ui32, handshake.name = "br28"} : <>
    %trueResult_76, %falseResult_77 = cond_br %180, %142 {handshake.bb = 8 : ui32, handshake.name = "cond_br89"} : <i1>, <>
    %trueResult_78, %falseResult_79 = cond_br %180, %144 {handshake.bb = 8 : ui32, handshake.name = "cond_br90"} : <i1>, <>
    %trueResult_80, %falseResult_81 = cond_br %180, %141 {handshake.bb = 8 : ui32, handshake.name = "cond_br91"} : <i1>, <>
    %trueResult_82, %falseResult_83 = cond_br %180, %143 {handshake.bb = 8 : ui32, handshake.name = "cond_br92"} : <i1>, <>
    %trueResult_84, %falseResult_85 = cond_br %180, %177 {handshake.bb = 8 : ui32, handshake.name = "cond_br93"} : <i1>, <>
    %140 = init %180 {ftd.imerge, handshake.bb = 8 : ui32, handshake.name = "init26"} : <i1>
    %141 = mux %140 [%134, %trueResult_80] {ftd.regen, handshake.bb = 8 : ui32, handshake.name = "mux47"} : <i1>, [<>, <>] to <>
    %142 = mux %140 [%111, %trueResult_76] {ftd.regen, handshake.bb = 8 : ui32, handshake.name = "mux48"} : <i1>, [<>, <>] to <>
    %143 = mux %140 [%112, %trueResult_82] {ftd.regen, handshake.bb = 8 : ui32, handshake.name = "mux49"} : <i1>, [<>, <>] to <>
    %144 = mux %140 [%128, %trueResult_78] {ftd.regen, handshake.bb = 8 : ui32, handshake.name = "mux51"} : <i1>, [<>, <>] to <>
    %145 = mux %140 [%113, %trueResult_84] {ftd.phi, handshake.bb = 8 : ui32, handshake.name = "mux52"} : <i1>, [<>, <>] to <>
    %146 = mux %index_87 [%135, %trueResult_97] {handshake.bb = 8 : ui32, handshake.name = "mux17"} : <i1>, [<i32>, <i32>] to <i32>
    %147 = mux %index_87 [%136, %trueResult_99] {handshake.bb = 8 : ui32, handshake.name = "mux18"} : <i1>, [<i32>, <i32>] to <i32>
    %148 = mux %index_87 [%137, %trueResult_101] {handshake.bb = 8 : ui32, handshake.name = "mux19"} : <i1>, [<i32>, <i32>] to <i32>
    %149 = mux %index_87 [%138, %trueResult_103] {handshake.bb = 8 : ui32, handshake.name = "mux20"} : <i1>, [<i32>, <i32>] to <i32>
    %result_86, %index_87 = control_merge [%139, %trueResult_105]  {handshake.bb = 8 : ui32, handshake.name = "control_merge7"} : [<>, <>] to <>, <i1>
    %150 = constant %result_86 {handshake.bb = 8 : ui32, handshake.name = "constant4", value = 1 : i32} : <>, <i32>
    %151 = source {handshake.bb = 8 : ui32, handshake.name = "source11"} : <>
    %152 = constant %151 {handshake.bb = 8 : ui32, handshake.name = "constant22", value = 10 : i32} : <>, <i32>
    %153 = source {handshake.bb = 8 : ui32, handshake.name = "source12"} : <>
    %154 = constant %153 {handshake.bb = 8 : ui32, handshake.name = "constant23", value = 1 : i32} : <>, <i32>
    %155 = source {handshake.bb = 8 : ui32, handshake.name = "source13"} : <>
    %156 = constant %155 {handshake.bb = 8 : ui32, handshake.name = "constant24", value = 3 : i32} : <>, <i32>
    %157 = shli %148, %154 {handshake.bb = 8 : ui32, handshake.name = "shli14"} : <i32>
    %158 = shli %148, %156 {handshake.bb = 8 : ui32, handshake.name = "shli15"} : <i32>
    %159 = addi %157, %158 {handshake.bb = 8 : ui32, handshake.name = "addi26"} : <i32>
    %160 = addi %146, %159 {handshake.bb = 8 : ui32, handshake.name = "addi9"} : <i32>
    %161 = gate %160, %143, %142 {handshake.bb = 8 : ui32, handshake.name = "gate2"} : <i32>, !handshake.control<>, !handshake.control<> to <i32>
    %addressResult_88, %dataResult_89 = load[%161] %outputs_6#3 {handshake.bb = 8 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load4"} : <i32>, <i32>, <i32>, <i32>
    %162 = shli %146, %154 {handshake.bb = 8 : ui32, handshake.name = "shli16"} : <i32>
    %163 = shli %146, %156 {handshake.bb = 8 : ui32, handshake.name = "shli17"} : <i32>
    %164 = addi %162, %163 {handshake.bb = 8 : ui32, handshake.name = "addi27"} : <i32>
    %165 = addi %149, %164 {handshake.bb = 8 : ui32, handshake.name = "addi10"} : <i32>
    %addressResult_90, %dataResult_91 = load[%165] %outputs_0 {handshake.bb = 8 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load5"} : <i32>, <i32>, <i32>, <i32>
    %166 = muli %dataResult_89, %dataResult_91 {handshake.bb = 8 : ui32, handshake.name = "muli3"} : <i32>
    %167 = shli %148, %154 {handshake.bb = 8 : ui32, handshake.name = "shli18"} : <i32>
    %168 = shli %148, %156 {handshake.bb = 8 : ui32, handshake.name = "shli19"} : <i32>
    %169 = addi %167, %168 {handshake.bb = 8 : ui32, handshake.name = "addi28"} : <i32>
    %170 = addi %149, %169 {handshake.bb = 8 : ui32, handshake.name = "addi11"} : <i32>
    %171 = gate %170, %145, %141 {handshake.bb = 8 : ui32, handshake.name = "gate3"} : <i32>, !handshake.control<>, !handshake.control<> to <i32>
    %addressResult_92, %dataResult_93 = load[%171] %outputs#2 {handshake.bb = 8 : ui32, handshake.deps = #handshake<deps[["store3", 3, false], ["store3", 4, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load6"} : <i32>, <i32>, <i32>, <i32>
    %172 = addi %dataResult_93, %166 {handshake.bb = 8 : ui32, handshake.name = "addi1"} : <i32>
    %173 = shli %148, %154 {handshake.bb = 8 : ui32, handshake.name = "shli20"} : <i32>
    %174 = shli %148, %156 {handshake.bb = 8 : ui32, handshake.name = "shli21"} : <i32>
    %175 = addi %173, %174 {handshake.bb = 8 : ui32, handshake.name = "addi29"} : <i32>
    %176 = addi %149, %175 {handshake.bb = 8 : ui32, handshake.name = "addi12"} : <i32>
    %177 = buffer %doneResult_96, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 8 : ui32, handshake.name = "buffer4"} : <>
    %178 = gate %176, %144, %141 {handshake.bb = 8 : ui32, handshake.name = "gate4"} : <i32>, !handshake.control<>, !handshake.control<> to <i32>
    %addressResult_94, %dataResult_95, %doneResult_96 = store[%178] %172 %outputs#3 {handshake.bb = 8 : ui32, handshake.deps = #handshake<deps[["load6", 3, false], ["store3", 3, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store3"} : <i32>, <i32>, <>, <i32>, <i32>, <>
    %179 = addi %146, %154 {handshake.bb = 8 : ui32, handshake.name = "addi16"} : <i32>
    %180 = cmpi ult, %179, %152 {handshake.bb = 8 : ui32, handshake.name = "cmpi3"} : <i32>
    %trueResult_97, %falseResult_98 = cond_br %180, %179 {handshake.bb = 8 : ui32, handshake.name = "cond_br22"} : <i1>, <i32>
    %trueResult_99, %falseResult_100 = cond_br %180, %147 {handshake.bb = 8 : ui32, handshake.name = "cond_br23"} : <i1>, <i32>
    %trueResult_101, %falseResult_102 = cond_br %180, %148 {handshake.bb = 8 : ui32, handshake.name = "cond_br24"} : <i1>, <i32>
    %trueResult_103, %falseResult_104 = cond_br %180, %149 {handshake.bb = 8 : ui32, handshake.name = "cond_br25"} : <i1>, <i32>
    %trueResult_105, %falseResult_106 = cond_br %180, %result_86 {handshake.bb = 8 : ui32, handshake.name = "cond_br26"} : <i1>, <>
    %trueResult_107, %falseResult_108 = cond_br %189, %112 {handshake.bb = 9 : ui32, handshake.name = "cond_br94"} : <i1>, <>
    %trueResult_109, %falseResult_110 = cond_br %189, %111 {handshake.bb = 9 : ui32, handshake.name = "cond_br95"} : <i1>, <>
    %trueResult_111, %falseResult_112 = cond_br %189, %falseResult_85 {handshake.bb = 9 : ui32, handshake.name = "cond_br96"} : <i1>, <>
    %181 = merge %falseResult_100 {handshake.bb = 9 : ui32, handshake.name = "merge7"} : <i32>
    %182 = merge %falseResult_102 {handshake.bb = 9 : ui32, handshake.name = "merge8"} : <i32>
    %183 = merge %falseResult_104 {handshake.bb = 9 : ui32, handshake.name = "merge9"} : <i32>
    %result_113, %index_114 = control_merge [%falseResult_106]  {handshake.bb = 9 : ui32, handshake.name = "control_merge8"} : [<>] to <>, <i1>
    %184 = source {handshake.bb = 9 : ui32, handshake.name = "source14"} : <>
    %185 = constant %184 {handshake.bb = 9 : ui32, handshake.name = "constant25", value = 10 : i32} : <>, <i32>
    %186 = source {handshake.bb = 9 : ui32, handshake.name = "source15"} : <>
    %187 = constant %186 {handshake.bb = 9 : ui32, handshake.name = "constant26", value = 1 : i32} : <>, <i32>
    %188 = addi %183, %187 {handshake.bb = 9 : ui32, handshake.name = "addi17"} : <i32>
    %189 = cmpi ult, %188, %185 {handshake.bb = 9 : ui32, handshake.name = "cmpi4"} : <i32>
    %trueResult_115, %falseResult_116 = cond_br %189, %188 {handshake.bb = 9 : ui32, handshake.name = "cond_br27"} : <i1>, <i32>
    %trueResult_117, %falseResult_118 = cond_br %189, %181 {handshake.bb = 9 : ui32, handshake.name = "cond_br28"} : <i1>, <i32>
    %trueResult_119, %falseResult_120 = cond_br %189, %182 {handshake.bb = 9 : ui32, handshake.name = "cond_br29"} : <i1>, <i32>
    %trueResult_121, %falseResult_122 = cond_br %189, %result_113 {handshake.bb = 9 : ui32, handshake.name = "cond_br30"} : <i1>, <>
    %trueResult_123, %falseResult_124 = cond_br %197, %falseResult_112 {handshake.bb = 10 : ui32, handshake.name = "cond_br97"} : <i1>, <>
    %trueResult_125, %falseResult_126 = cond_br %197, %101 {handshake.bb = 10 : ui32, handshake.name = "cond_br98"} : <i1>, <>
    %trueResult_127, %falseResult_128 = cond_br %197, %100 {handshake.bb = 10 : ui32, handshake.name = "cond_br99"} : <i1>, <>
    %190 = merge %falseResult_118 {handshake.bb = 10 : ui32, handshake.name = "merge10"} : <i32>
    %191 = merge %falseResult_120 {handshake.bb = 10 : ui32, handshake.name = "merge11"} : <i32>
    %result_129, %index_130 = control_merge [%falseResult_122]  {handshake.bb = 10 : ui32, handshake.name = "control_merge9"} : [<>] to <>, <i1>
    %192 = source {handshake.bb = 10 : ui32, handshake.name = "source16"} : <>
    %193 = constant %192 {handshake.bb = 10 : ui32, handshake.name = "constant27", value = 10 : i32} : <>, <i32>
    %194 = source {handshake.bb = 10 : ui32, handshake.name = "source17"} : <>
    %195 = constant %194 {handshake.bb = 10 : ui32, handshake.name = "constant28", value = 1 : i32} : <>, <i32>
    %196 = addi %191, %195 {handshake.bb = 10 : ui32, handshake.name = "addi18"} : <i32>
    %197 = cmpi ult, %196, %193 {handshake.bb = 10 : ui32, handshake.name = "cmpi5"} : <i32>
    %trueResult_131, %falseResult_132 = cond_br %197, %196 {handshake.bb = 10 : ui32, handshake.name = "cond_br31"} : <i1>, <i32>
    %trueResult_133, %falseResult_134 = cond_br %197, %190 {handshake.bb = 10 : ui32, handshake.name = "cond_br32"} : <i1>, <i32>
    %trueResult_135, %falseResult_136 = cond_br %197, %result_129 {handshake.bb = 10 : ui32, handshake.name = "cond_br33"} : <i1>, <>
    %result_137, %index_138 = control_merge [%falseResult_136]  {handshake.bb = 11 : ui32, handshake.name = "control_merge10"} : [<>] to <>, <i1>
    end {handshake.bb = 11 : ui32, handshake.name = "end0"} %memEnd_7, %memEnd_5, %memEnd_3, %memEnd_1, %memEnd, %arg12 : <>, <>, <>, <>, <>, <>
  }
}

