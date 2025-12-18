module {
  handshake.func @jacobi_1d_imper(%arg0: memref<100xi32>, %arg1: memref<100xi32>, %arg2: !handshake.control<>, %arg3: !handshake.control<>, %arg4: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["A", "B", "A_start", "B_start", "start"], cfg.edges = "[0,1][2,2,3,cmpi0][4,1,5,cmpi2][1,2][3,3,4,cmpi1]", resNames = ["A_end", "B_end", "end"]} {
    %outputs:2, %memEnd = mem_controller[%arg1 : memref<100xi32>] %arg3 (%17, %addressResult_16, %dataResult_17, %addressResult_38) %result_57 {connectedBlocks = [2 : i32, 3 : i32], handshake.name = "mem_controller0"} :    (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>) -> (!handshake.control<>, !handshake.channel<i32>)
    %outputs_0:4, %memEnd_1 = mem_controller[%arg0 : memref<100xi32>] %arg2 (%addressResult, %addressResult_12, %addressResult_14, %49, %addressResult_40, %dataResult_41) %result_57 {connectedBlocks = [2 : i32, 3 : i32], handshake.name = "mem_controller1"} :    (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>) -> (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>)
    %0 = constant %arg4 {handshake.bb = 0 : ui32, handshake.name = "constant3", value = 0 : i32} : <>, <i32>
    %1 = br %0 {handshake.bb = 0 : ui32, handshake.name = "br2"} : <i32>
    %2 = br %arg4 {handshake.bb = 0 : ui32, handshake.name = "br3"} : <>
    %3 = mux %4 [%arg4, %trueResult_49] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux8"} : <i1>, [<>, <>] to <>
    %4 = init %66 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "init0"} : <i1>
    %5 = mux %index [%1, %trueResult_53] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i32>, <i32>] to <i32>
    %result, %index = control_merge [%2, %trueResult_55]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %6 = constant %result {handshake.bb = 1 : ui32, handshake.name = "constant6", value = 1 : i32} : <>, <i32>
    %7 = br %6 {handshake.bb = 1 : ui32, handshake.name = "br4"} : <i32>
    %8 = br %5 {handshake.bb = 1 : ui32, handshake.name = "br5"} : <i32>
    %9 = br %result {handshake.bb = 1 : ui32, handshake.name = "br6"} : <>
    %trueResult, %falseResult = cond_br %40, %37 {handshake.bb = 2 : ui32, handshake.name = "cond_br45"} : <i1>, <>
    %trueResult_2, %falseResult_3 = cond_br %40, %28 {handshake.bb = 2 : ui32, handshake.name = "cond_br46"} : <i1>, <>
    %trueResult_4, %falseResult_5 = cond_br %40, %32 {handshake.bb = 2 : ui32, handshake.name = "cond_br47"} : <i1>, <>
    %trueResult_6, %falseResult_7 = cond_br %40, %26 {handshake.bb = 2 : ui32, handshake.name = "cond_br48"} : <i1>, <>
    %trueResult_8, %falseResult_9 = cond_br %40, %11 {handshake.bb = 2 : ui32, handshake.name = "cond_br49"} : <i1>, <>
    %10 = init %40 {ftd.imerge, handshake.bb = 2 : ui32, handshake.name = "init8"} : <i1>
    %11 = mux %10 [%3, %trueResult_8] {ftd.regen, handshake.bb = 2 : ui32, handshake.name = "mux13"} : <i1>, [<>, <>] to <>
    %12:2 = unbundle %dataResult_13  {handshake.bb = 2 : ui32, handshake.name = "unbundle0"} : <i32> to _ 
    %13:2 = unbundle %dataResult  {handshake.bb = 2 : ui32, handshake.name = "unbundle1"} : <i32> to _ 
    %14:2 = unbundle %dataResult_15  {handshake.bb = 2 : ui32, handshake.name = "unbundle2"} : <i32> to _ 
    %15 = mux %index_11 [%7, %trueResult_18] {handshake.bb = 2 : ui32, handshake.name = "mux1"} : <i1>, [<i32>, <i32>] to <i32>
    %16 = mux %index_11 [%8, %trueResult_20] {handshake.bb = 2 : ui32, handshake.name = "mux2"} : <i1>, [<i32>, <i32>] to <i32>
    %result_10, %index_11 = control_merge [%9, %trueResult_22]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>, <>] to <>, <i1>
    %17 = constant %result_10 {handshake.bb = 2 : ui32, handshake.name = "constant0", value = 1 : i32} : <>, <i32>
    %18 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %19 = constant %18 {handshake.bb = 2 : ui32, handshake.name = "constant7", value = -1 : i32} : <>, <i32>
    %20 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %21 = constant %20 {handshake.bb = 2 : ui32, handshake.name = "constant8", value = 99 : i32} : <>, <i32>
    %22 = constant %result_10 {handshake.bb = 2 : ui32, handshake.name = "constant9", value = 1 : i32} : <>, <i32>
    %23 = source {handshake.bb = 2 : ui32, handshake.name = "source2"} : <>
    %24 = constant %23 {handshake.bb = 2 : ui32, handshake.name = "constant10", value = 1 : i32} : <>, <i32>
    %25 = addi %15, %19 {handshake.bb = 2 : ui32, handshake.name = "addi2"} : <i32>
    %26 = buffer %13#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer0"} : <>
    %27 = gate %25, %11 {handshake.bb = 2 : ui32, handshake.name = "gate0"} : <i32>, !handshake.control<> to <i32>
    %addressResult, %dataResult = load[%27] %outputs_0#0 {handshake.bb = 2 : ui32, handshake.deps = #handshake<deps[["store1", 1, false], ["store1", 2, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i32>, <i32>, <i32>, <i32>
    %28 = buffer %12#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer1"} : <>
    %29 = gate %15, %11 {handshake.bb = 2 : ui32, handshake.name = "gate1"} : <i32>, !handshake.control<> to <i32>
    %addressResult_12, %dataResult_13 = load[%29] %outputs_0#1 {handshake.bb = 2 : ui32, handshake.deps = #handshake<deps[["store1", 1, false], ["store1", 2, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i32>, <i32>, <i32>, <i32>
    %30 = addi %dataResult, %dataResult_13 {handshake.bb = 2 : ui32, handshake.name = "addi0"} : <i32>
    %31 = addi %15, %22 {handshake.bb = 2 : ui32, handshake.name = "addi3"} : <i32>
    %32 = buffer %14#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer2"} : <>
    %33 = gate %31, %11 {handshake.bb = 2 : ui32, handshake.name = "gate2"} : <i32>, !handshake.control<> to <i32>
    %addressResult_14, %dataResult_15 = load[%33] %outputs_0#2 {handshake.bb = 2 : ui32, handshake.deps = #handshake<deps[["store1", 1, false], ["store1", 2, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load2"} : <i32>, <i32>, <i32>, <i32>
    %34 = addi %30, %dataResult_15 {handshake.bb = 2 : ui32, handshake.name = "addi1"} : <i32>
    %35 = shli %34, %24 {handshake.bb = 2 : ui32, handshake.name = "shli0"} : <i32>
    %36 = addi %34, %35 {handshake.bb = 2 : ui32, handshake.name = "addi7"} : <i32>
    %37 = buffer %doneResult, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer3"} : <>
    %38 = gate %15, %11 {handshake.bb = 2 : ui32, handshake.name = "gate3"} : <i32>, !handshake.control<> to <i32>
    %addressResult_16, %dataResult_17, %doneResult = store[%38] %36 %outputs#0 {handshake.bb = 2 : ui32, handshake.deps = #handshake<deps[["store0", 1, false], ["load3", 1, false], ["load3", 2, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store0"} : <i32>, <i32>, <>, <i32>, <i32>, <>
    %39 = addi %15, %22 {handshake.bb = 2 : ui32, handshake.name = "addi4"} : <i32>
    %40 = cmpi ult, %39, %21 {handshake.bb = 2 : ui32, handshake.name = "cmpi0"} : <i32>
    %trueResult_18, %falseResult_19 = cond_br %40, %39 {handshake.bb = 2 : ui32, handshake.name = "cond_br3"} : <i1>, <i32>
    %trueResult_20, %falseResult_21 = cond_br %40, %16 {handshake.bb = 2 : ui32, handshake.name = "cond_br4"} : <i1>, <i32>
    %trueResult_22, %falseResult_23 = cond_br %40, %result_10 {handshake.bb = 2 : ui32, handshake.name = "cond_br5"} : <i1>, <>
    %trueResult_24, %falseResult_25 = cond_br %40, %22 {handshake.bb = 2 : ui32, handshake.name = "cond_br6"} : <i1>, <i32>
    %trueResult_26, %falseResult_27 = cond_br %59, %45 {handshake.bb = 3 : ui32, handshake.name = "cond_br50"} : <i1>, <>
    %trueResult_28, %falseResult_29 = cond_br %59, %56 {handshake.bb = 3 : ui32, handshake.name = "cond_br51"} : <i1>, <>
    %trueResult_30, %falseResult_31 = cond_br %59, %43 {handshake.bb = 3 : ui32, handshake.name = "cond_br52"} : <i1>, <>
    %trueResult_32, %falseResult_33 = cond_br %59, %44 {handshake.bb = 3 : ui32, handshake.name = "cond_br53"} : <i1>, <>
    %trueResult_34, %falseResult_35 = cond_br %59, %42 {handshake.bb = 3 : ui32, handshake.name = "cond_br54"} : <i1>, <>
    %41 = init %59 {ftd.imerge, handshake.bb = 3 : ui32, handshake.name = "init16"} : <i1>
    %42 = mux %41 [%falseResult_3, %trueResult_34] {ftd.regen, handshake.bb = 3 : ui32, handshake.name = "mux21"} : <i1>, [<>, <>] to <>
    %43 = mux %41 [%falseResult_7, %trueResult_30] {ftd.regen, handshake.bb = 3 : ui32, handshake.name = "mux22"} : <i1>, [<>, <>] to <>
    %44 = mux %41 [%falseResult_5, %trueResult_32] {ftd.regen, handshake.bb = 3 : ui32, handshake.name = "mux23"} : <i1>, [<>, <>] to <>
    %45 = mux %41 [%falseResult, %trueResult_26] {ftd.regen, handshake.bb = 3 : ui32, handshake.name = "mux24"} : <i1>, [<>, <>] to <>
    %46:2 = unbundle %dataResult_39  {handshake.bb = 3 : ui32, handshake.name = "unbundle6"} : <i32> to _ 
    %47 = mux %index_37 [%falseResult_25, %trueResult_43] {handshake.bb = 3 : ui32, handshake.name = "mux3"} : <i1>, [<i32>, <i32>] to <i32>
    %48 = mux %index_37 [%falseResult_21, %trueResult_45] {handshake.bb = 3 : ui32, handshake.name = "mux4"} : <i1>, [<i32>, <i32>] to <i32>
    %result_36, %index_37 = control_merge [%falseResult_23, %trueResult_47]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>, <>] to <>, <i1>
    %49 = constant %result_36 {handshake.bb = 3 : ui32, handshake.name = "constant1", value = 1 : i32} : <>, <i32>
    %50 = source {handshake.bb = 3 : ui32, handshake.name = "source3"} : <>
    %51 = constant %50 {handshake.bb = 3 : ui32, handshake.name = "constant11", value = 99 : i32} : <>, <i32>
    %52 = source {handshake.bb = 3 : ui32, handshake.name = "source4"} : <>
    %53 = constant %52 {handshake.bb = 3 : ui32, handshake.name = "constant12", value = 1 : i32} : <>, <i32>
    %54 = buffer %46#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer4"} : <>
    %55 = gate %47, %45 {handshake.bb = 3 : ui32, handshake.name = "gate4"} : <i32>, !handshake.control<> to <i32>
    %addressResult_38, %dataResult_39 = load[%55] %outputs#1 {handshake.bb = 3 : ui32, handshake.deps = #handshake<deps[["store0", 1, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load3"} : <i32>, <i32>, <i32>, <i32>
    %56 = buffer %doneResult_42, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer5"} : <>
    %57 = gate %47, %44, %43, %42 {handshake.bb = 3 : ui32, handshake.name = "gate5"} : <i32>, !handshake.control<>, !handshake.control<>, !handshake.control<> to <i32>
    %addressResult_40, %dataResult_41, %doneResult_42 = store[%57] %dataResult_39 %outputs_0#3 {handshake.bb = 3 : ui32, handshake.deps = #handshake<deps[["load0", 1, false], ["load1", 1, false], ["load2", 1, false], ["store1", 1, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store1"} : <i32>, <i32>, <>, <i32>, <i32>, <>
    %58 = addi %47, %53 {handshake.bb = 3 : ui32, handshake.name = "addi5"} : <i32>
    %59 = cmpi ult, %58, %51 {handshake.bb = 3 : ui32, handshake.name = "cmpi1"} : <i32>
    %trueResult_43, %falseResult_44 = cond_br %59, %58 {handshake.bb = 3 : ui32, handshake.name = "cond_br7"} : <i1>, <i32>
    %trueResult_45, %falseResult_46 = cond_br %59, %48 {handshake.bb = 3 : ui32, handshake.name = "cond_br8"} : <i1>, <i32>
    %trueResult_47, %falseResult_48 = cond_br %59, %result_36 {handshake.bb = 3 : ui32, handshake.name = "cond_br9"} : <i1>, <>
    %trueResult_49, %falseResult_50 = cond_br %66, %falseResult_29 {handshake.bb = 4 : ui32, handshake.name = "cond_br55"} : <i1>, <>
    %60 = merge %falseResult_46 {handshake.bb = 4 : ui32, handshake.name = "merge0"} : <i32>
    %result_51, %index_52 = control_merge [%falseResult_48]  {handshake.bb = 4 : ui32, handshake.name = "control_merge3"} : [<>] to <>, <i1>
    %61 = source {handshake.bb = 4 : ui32, handshake.name = "source5"} : <>
    %62 = constant %61 {handshake.bb = 4 : ui32, handshake.name = "constant13", value = 3 : i32} : <>, <i32>
    %63 = source {handshake.bb = 4 : ui32, handshake.name = "source6"} : <>
    %64 = constant %63 {handshake.bb = 4 : ui32, handshake.name = "constant14", value = 1 : i32} : <>, <i32>
    %65 = addi %60, %64 {handshake.bb = 4 : ui32, handshake.name = "addi6"} : <i32>
    %66 = cmpi ult, %65, %62 {handshake.bb = 4 : ui32, handshake.name = "cmpi2"} : <i32>
    %trueResult_53, %falseResult_54 = cond_br %66, %65 {handshake.bb = 4 : ui32, handshake.name = "cond_br10"} : <i1>, <i32>
    %trueResult_55, %falseResult_56 = cond_br %66, %result_51 {handshake.bb = 4 : ui32, handshake.name = "cond_br11"} : <i1>, <>
    %result_57, %index_58 = control_merge [%falseResult_56]  {handshake.bb = 5 : ui32, handshake.name = "control_merge4"} : [<>] to <>, <i1>
    end {handshake.bb = 5 : ui32, handshake.name = "end0"} %memEnd_1, %memEnd, %arg4 : <>, <>, <>
  }
}

