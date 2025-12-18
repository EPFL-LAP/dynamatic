module {
  handshake.func @gaussian(%arg0: memref<20xi32>, %arg1: memref<400xi32>, %arg2: !handshake.control<>, %arg3: !handshake.control<>, %arg4: !handshake.control<>, ...) -> (!handshake.channel<i32>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["c", "a", "c_start", "a_start", "start"], cfg.edges = "[0,1][2,3,5,cmpi2][4,2][1,2][3,3,4,cmpi0][5,1,6,cmpi1]", resNames = ["out0", "c_end", "a_end", "end"]} {
    %outputs:3, %memEnd = mem_controller[%arg1 : memref<400xi32>] %arg3 (%50, %addressResult, %addressResult_40, %addressResult_42, %dataResult_43) %result_78 {connectedBlocks = [3 : i32], handshake.name = "mem_controller1"} :    (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>) -> (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>)
    %outputs_0, %memEnd_1 = mem_controller[%arg0 : memref<20xi32>] %arg2 (%addressResult_36) %result_78 {connectedBlocks = [3 : i32], handshake.name = "mem_controller2"} :    (!handshake.channel<i32>) -> !handshake.channel<i32>
    %0 = constant %arg4 {handshake.bb = 0 : ui32, handshake.name = "constant0", value = 1000 : i32} : <>, <i32>
    %1 = constant %arg4 {handshake.bb = 0 : ui32, handshake.name = "constant1", value = 1000 : i32} : <>, <i32>
    %2 = constant %arg4 {handshake.bb = 0 : ui32, handshake.name = "constant2", value = 0 : i32} : <>, <i32>
    %3 = constant %arg4 {handshake.bb = 0 : ui32, handshake.name = "constant4", value = 1 : i32} : <>, <i32>
    %4 = br %3 {handshake.bb = 0 : ui32, handshake.name = "br2"} : <i32>
    %5 = br %2 {handshake.bb = 0 : ui32, handshake.name = "br4"} : <i32>
    %6 = br %arg4 {handshake.bb = 0 : ui32, handshake.name = "br5"} : <>
    %7 = mux %13 [%arg4, %trueResult_62] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux10"} : <i1>, [<>, <>] to <>
    %8 = mux %13 [%1, %trueResult_58] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux11"} : <i1>, [<i32>, <i32>] to <i32>
    %9 = mux %13 [%0, %trueResult_60] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux12"} : <i1>, [<i32>, <i32>] to <i32>
    %10 = mux %13 [%arg4, %trueResult_68] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux13"} : <i1>, [<>, <>] to <>
    %11 = mux %13 [%arg4, %trueResult_66] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux14"} : <i1>, [<>, <>] to <>
    %12 = mux %13 [%arg4, %trueResult_64] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux15"} : <i1>, [<>, <>] to <>
    %13 = init %111 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "init0"} : <i1>
    %14 = mux %index [%4, %trueResult_72] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i32>, <i32>] to <i32>
    %15 = mux %index [%5, %trueResult_74] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i1>, [<i32>, <i32>] to <i32>
    %result, %index = control_merge [%6, %trueResult_76]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %16 = source {handshake.bb = 1 : ui32, handshake.name = "source0"} : <>
    %17 = constant %16 {handshake.bb = 1 : ui32, handshake.name = "constant7", value = 1 : i32} : <>, <i32>
    %18 = addi %14, %17 {handshake.bb = 1 : ui32, handshake.name = "addi2"} : <i32>
    %19 = br %18 {handshake.bb = 1 : ui32, handshake.name = "br6"} : <i32>
    %20 = br %15 {handshake.bb = 1 : ui32, handshake.name = "br7"} : <i32>
    %21 = br %14 {handshake.bb = 1 : ui32, handshake.name = "br8"} : <i32>
    %22 = br %result {handshake.bb = 1 : ui32, handshake.name = "br9"} : <>
    %23 = mux %29 [%7, %falseResult_31] {ftd.phi, handshake.bb = 2 : ui32, handshake.name = "mux16"} : <i1>, [<>, <>] to <>
    %24 = mux %29 [%8, %falseResult_23] {ftd.phi, handshake.bb = 2 : ui32, handshake.name = "mux17"} : <i1>, [<i32>, <i32>] to <i32>
    %25 = mux %29 [%9, %falseResult_23] {ftd.phi, handshake.bb = 2 : ui32, handshake.name = "mux18"} : <i1>, [<i32>, <i32>] to <i32>
    %26 = mux %29 [%10, %falseResult_31] {ftd.phi, handshake.bb = 2 : ui32, handshake.name = "mux19"} : <i1>, [<>, <>] to <>
    %27 = mux %29 [%11, %falseResult_27] {ftd.phi, handshake.bb = 2 : ui32, handshake.name = "mux20"} : <i1>, [<>, <>] to <>
    %28 = mux %29 [%12, %falseResult_27] {ftd.phi, handshake.bb = 2 : ui32, handshake.name = "mux21"} : <i1>, [<>, <>] to <>
    %29 = init %37 {ftd.imerge, handshake.bb = 2 : ui32, handshake.name = "init6"} : <i1>
    %30 = mux %index_3 [%19, %100] {handshake.bb = 2 : ui32, handshake.name = "mux2"} : <i1>, [<i32>, <i32>] to <i32>
    %31 = mux %index_3 [%20, %101] {handshake.bb = 2 : ui32, handshake.name = "mux3"} : <i1>, [<i32>, <i32>] to <i32>
    %32 = mux %index_3 [%21, %102] {handshake.bb = 2 : ui32, handshake.name = "mux4"} : <i1>, [<i32>, <i32>] to <i32>
    %result_2, %index_3 = control_merge [%22, %103]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>, <>] to <>, <i1>
    %33 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %34 = constant %33 {handshake.bb = 2 : ui32, handshake.name = "constant8", value = 19 : i32} : <>, <i32>
    %35 = constant %result_2 {handshake.bb = 2 : ui32, handshake.name = "constant9", value = 1 : i32} : <>, <i32>
    %36 = constant %result_2 {handshake.bb = 2 : ui32, handshake.name = "constant10", value = 1 : i32} : <>, <i32>
    %37 = cmpi ult, %30, %34 {handshake.bb = 2 : ui32, handshake.name = "cmpi2"} : <i32>
    %trueResult, %falseResult = cond_br %37, %35 {handshake.bb = 2 : ui32, handshake.name = "cond_br3"} : <i1>, <i32>
    %trueResult_4, %falseResult_5 = cond_br %37, %36 {handshake.bb = 2 : ui32, handshake.name = "cond_br4"} : <i1>, <i32>
    %trueResult_6, %falseResult_7 = cond_br %37, %31 {handshake.bb = 2 : ui32, handshake.name = "cond_br5"} : <i1>, <i32>
    %trueResult_8, %falseResult_9 = cond_br %37, %32 {handshake.bb = 2 : ui32, handshake.name = "cond_br6"} : <i1>, <i32>
    %trueResult_10, %falseResult_11 = cond_br %37, %30 {handshake.bb = 2 : ui32, handshake.name = "cond_br7"} : <i1>, <i32>
    %trueResult_12, %falseResult_13 = cond_br %37, %result_2 {handshake.bb = 2 : ui32, handshake.name = "cond_br8"} : <i1>, <>
    %trueResult_14, %falseResult_15 = cond_br %37, %24 {handshake.bb = 3 : ui32, handshake.name = "cond_br47"} : <i1>, <i32>
    %trueResult_16, %falseResult_17 = cond_br %37, %23 {handshake.bb = 3 : ui32, handshake.name = "cond_br48"} : <i1>, <>
    %trueResult_18, %falseResult_19 = cond_br %37, %25 {handshake.bb = 3 : ui32, handshake.name = "cond_br49"} : <i1>, <i32>
    %trueResult_20, %falseResult_21 = cond_br %37, %28 {handshake.bb = 3 : ui32, handshake.name = "cond_br50"} : <i1>, <>
    %trueResult_22, %falseResult_23 = cond_br %93, %87 {handshake.bb = 3 : ui32, handshake.name = "cond_br51"} : <i1>, <i32>
    %trueResult_24, %falseResult_25 = cond_br %37, %26 {handshake.bb = 3 : ui32, handshake.name = "cond_br52"} : <i1>, <>
    %trueResult_26, %falseResult_27 = cond_br %93, %88 {handshake.bb = 3 : ui32, handshake.name = "cond_br53"} : <i1>, <>
    %trueResult_28, %falseResult_29 = cond_br %37, %27 {handshake.bb = 3 : ui32, handshake.name = "cond_br54"} : <i1>, <>
    %trueResult_30, %falseResult_31 = cond_br %93, %89 {handshake.bb = 3 : ui32, handshake.name = "cond_br55"} : <i1>, <>
    %38 = mux %44 [%trueResult_16, %trueResult_30] {ftd.phi, handshake.bb = 3 : ui32, handshake.name = "mux22"} : <i1>, [<>, <>] to <>
    %39 = mux %44 [%trueResult_14, %trueResult_22] {ftd.phi, handshake.bb = 3 : ui32, handshake.name = "mux23"} : <i1>, [<i32>, <i32>] to <i32>
    %40 = mux %44 [%trueResult_18, %trueResult_22] {ftd.phi, handshake.bb = 3 : ui32, handshake.name = "mux24"} : <i1>, [<i32>, <i32>] to <i32>
    %41 = mux %44 [%trueResult_24, %trueResult_30] {ftd.phi, handshake.bb = 3 : ui32, handshake.name = "mux25"} : <i1>, [<>, <>] to <>
    %42 = mux %44 [%trueResult_28, %trueResult_26] {ftd.phi, handshake.bb = 3 : ui32, handshake.name = "mux26"} : <i1>, [<>, <>] to <>
    %43 = mux %44 [%trueResult_20, %trueResult_26] {ftd.phi, handshake.bb = 3 : ui32, handshake.name = "mux27"} : <i1>, [<>, <>] to <>
    %44 = init %93 {ftd.imerge, handshake.bb = 3 : ui32, handshake.name = "init12"} : <i1>
    %45 = mux %index_33 [%trueResult, %trueResult_44] {handshake.bb = 3 : ui32, handshake.name = "mux5"} : <i1>, [<i32>, <i32>] to <i32>
    %46 = mux %index_33 [%trueResult_4, %trueResult_46] {handshake.bb = 3 : ui32, handshake.name = "mux6"} : <i1>, [<i32>, <i32>] to <i32>
    %47 = mux %index_33 [%trueResult_6, %trueResult_48] {handshake.bb = 3 : ui32, handshake.name = "mux7"} : <i1>, [<i32>, <i32>] to <i32>
    %48 = mux %index_33 [%trueResult_8, %trueResult_50] {handshake.bb = 3 : ui32, handshake.name = "mux8"} : <i1>, [<i32>, <i32>] to <i32>
    %49 = mux %index_33 [%trueResult_10, %trueResult_52] {handshake.bb = 3 : ui32, handshake.name = "mux9"} : <i1>, [<i32>, <i32>] to <i32>
    %result_32, %index_33 = control_merge [%trueResult_12, %trueResult_54]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>, <>] to <>, <i1>
    %50 = constant %result_32 {handshake.bb = 3 : ui32, handshake.name = "constant3", value = 1 : i32} : <>, <i32>
    %51 = source {handshake.bb = 3 : ui32, handshake.name = "source2"} : <>
    %52 = constant %51 {handshake.bb = 3 : ui32, handshake.name = "constant11", value = 20 : i32} : <>, <i32>
    %53 = source {handshake.bb = 3 : ui32, handshake.name = "source3"} : <>
    %54 = constant %53 {handshake.bb = 3 : ui32, handshake.name = "constant12", value = 1 : i32} : <>, <i32>
    %55 = source {handshake.bb = 3 : ui32, handshake.name = "source4"} : <>
    %56 = constant %55 {handshake.bb = 3 : ui32, handshake.name = "constant13", value = 1 : i32} : <>, <i32>
    %57 = source {handshake.bb = 3 : ui32, handshake.name = "source5"} : <>
    %58 = constant %57 {handshake.bb = 3 : ui32, handshake.name = "constant14", value = 4 : i32} : <>, <i32>
    %59 = source {handshake.bb = 3 : ui32, handshake.name = "source6"} : <>
    %60 = constant %59 {handshake.bb = 3 : ui32, handshake.name = "constant15", value = 2 : i32} : <>, <i32>
    %61 = shli %49, %60 {handshake.bb = 3 : ui32, handshake.name = "shli0"} : <i32>
    %62 = shli %49, %58 {handshake.bb = 3 : ui32, handshake.name = "shli1"} : <i32>
    %63 = addi %61, %62 {handshake.bb = 3 : ui32, handshake.name = "addi9"} : <i32>
    %64 = addi %46, %63 {handshake.bb = 3 : ui32, handshake.name = "addi3"} : <i32>
    %65 = gate %64, %38 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 3 : ui32, handshake.name = "gate0"} : <i32>, !handshake.control<> to <i32>
    %66 = cmpi ne, %65, %39 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 3 : ui32, handshake.name = "cmpi3"} : <i32>
    %trueResult_34, %falseResult_35 = cond_br %66, %42 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 3 : ui32, handshake.name = "cond_br39"} : <i1>, <>
    %67 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 3 : ui32, handshake.name = "source10"} : <>
    %68 = mux %66 [%falseResult_35, %67] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 3 : ui32, handshake.name = "mux28"} : <i1>, [<>, <>] to <>
    %69 = join %68 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 3 : ui32, handshake.name = "join0"} : <>
    %70 = gate %64, %69 {handshake.bb = 3 : ui32, handshake.name = "gate1"} : <i32>, !handshake.control<> to <i32>
    %addressResult, %dataResult = load[%70] %outputs#0 {handshake.bb = 3 : ui32, handshake.deps = #handshake<deps[["store0", 0, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i32>, <i32>, <i32>, <i32>
    %addressResult_36, %dataResult_37 = load[%48] %outputs_0 {handshake.bb = 3 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i32>, <i32>, <i32>, <i32>
    %71 = shli %48, %60 {handshake.bb = 3 : ui32, handshake.name = "shli2"} : <i32>
    %72 = shli %48, %58 {handshake.bb = 3 : ui32, handshake.name = "shli3"} : <i32>
    %73 = addi %71, %72 {handshake.bb = 3 : ui32, handshake.name = "addi10"} : <i32>
    %74 = addi %46, %73 {handshake.bb = 3 : ui32, handshake.name = "addi4"} : <i32>
    %75 = gate %74, %41 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 3 : ui32, handshake.name = "gate2"} : <i32>, !handshake.control<> to <i32>
    %76 = cmpi ne, %75, %40 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 3 : ui32, handshake.name = "cmpi4"} : <i32>
    %trueResult_38, %falseResult_39 = cond_br %76, %43 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 3 : ui32, handshake.name = "cond_br40"} : <i1>, <>
    %77 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 3 : ui32, handshake.name = "source11"} : <>
    %78 = mux %76 [%falseResult_39, %77] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 3 : ui32, handshake.name = "mux29"} : <i1>, [<>, <>] to <>
    %79 = join %78 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 3 : ui32, handshake.name = "join1"} : <>
    %80 = gate %74, %79 {handshake.bb = 3 : ui32, handshake.name = "gate3"} : <i32>, !handshake.control<> to <i32>
    %addressResult_40, %dataResult_41 = load[%80] %outputs#1 {handshake.bb = 3 : ui32, handshake.deps = #handshake<deps[["store0", 0, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load2"} : <i32>, <i32>, <i32>, <i32>
    %81 = muli %dataResult_37, %dataResult_41 {handshake.bb = 3 : ui32, handshake.name = "muli0"} : <i32>
    %82 = subi %dataResult, %81 {handshake.bb = 3 : ui32, handshake.name = "subi0"} : <i32>
    %83 = shli %49, %60 {handshake.bb = 3 : ui32, handshake.name = "shli4"} : <i32>
    %84 = shli %49, %58 {handshake.bb = 3 : ui32, handshake.name = "shli5"} : <i32>
    %85 = addi %83, %84 {handshake.bb = 3 : ui32, handshake.name = "addi11"} : <i32>
    %86 = addi %46, %85 {handshake.bb = 3 : ui32, handshake.name = "addi5"} : <i32>
    %87 = buffer %86, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer0"} : <i32>
    %88 = buffer %doneResult, bufferType = FIFO_BREAK_NONE, numSlots = 1 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 3 : ui32, handshake.name = "buffer1"} : <>
    %89 = init %88 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 3 : ui32, handshake.name = "init18"} : <>
    %addressResult_42, %dataResult_43, %doneResult = store[%86] %82 %outputs#2 {handshake.bb = 3 : ui32, handshake.deps = #handshake<deps[["load0", 0, false], ["load2", 0, false], ["store0", 0, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store0"} : <i32>, <i32>, <>, <i32>, <i32>, <>
    %90 = addi %47, %46 {handshake.bb = 3 : ui32, handshake.name = "addi0"} : <i32>
    %91 = addi %46, %56 {handshake.bb = 3 : ui32, handshake.name = "addi1"} : <i32>
    %92 = addi %45, %54 {handshake.bb = 3 : ui32, handshake.name = "addi6"} : <i32>
    %93 = cmpi ult, %92, %52 {handshake.bb = 3 : ui32, handshake.name = "cmpi0"} : <i32>
    %trueResult_44, %falseResult_45 = cond_br %93, %92 {handshake.bb = 3 : ui32, handshake.name = "cond_br9"} : <i1>, <i32>
    %trueResult_46, %falseResult_47 = cond_br %93, %91 {handshake.bb = 3 : ui32, handshake.name = "cond_br10"} : <i1>, <i32>
    %trueResult_48, %falseResult_49 = cond_br %93, %90 {handshake.bb = 3 : ui32, handshake.name = "cond_br11"} : <i1>, <i32>
    %trueResult_50, %falseResult_51 = cond_br %93, %48 {handshake.bb = 3 : ui32, handshake.name = "cond_br12"} : <i1>, <i32>
    %trueResult_52, %falseResult_53 = cond_br %93, %49 {handshake.bb = 3 : ui32, handshake.name = "cond_br13"} : <i1>, <i32>
    %trueResult_54, %falseResult_55 = cond_br %93, %result_32 {handshake.bb = 3 : ui32, handshake.name = "cond_br14"} : <i1>, <>
    %94 = merge %falseResult_51 {handshake.bb = 4 : ui32, handshake.name = "merge0"} : <i32>
    %95 = merge %falseResult_53 {handshake.bb = 4 : ui32, handshake.name = "merge1"} : <i32>
    %96 = merge %falseResult_49 {handshake.bb = 4 : ui32, handshake.name = "merge2"} : <i32>
    %result_56, %index_57 = control_merge [%falseResult_55]  {handshake.bb = 4 : ui32, handshake.name = "control_merge3"} : [<>] to <>, <i1>
    %97 = source {handshake.bb = 4 : ui32, handshake.name = "source7"} : <>
    %98 = constant %97 {handshake.bb = 4 : ui32, handshake.name = "constant16", value = 1 : i32} : <>, <i32>
    %99 = addi %95, %98 {handshake.bb = 4 : ui32, handshake.name = "addi8"} : <i32>
    %100 = br %99 {handshake.bb = 4 : ui32, handshake.name = "br10"} : <i32>
    %101 = br %96 {handshake.bb = 4 : ui32, handshake.name = "br11"} : <i32>
    %102 = br %94 {handshake.bb = 4 : ui32, handshake.name = "br12"} : <i32>
    %103 = br %result_56 {handshake.bb = 4 : ui32, handshake.name = "br13"} : <>
    %trueResult_58, %falseResult_59 = cond_br %111, %falseResult_15 {handshake.bb = 5 : ui32, handshake.name = "cond_br56"} : <i1>, <i32>
    %trueResult_60, %falseResult_61 = cond_br %111, %falseResult_19 {handshake.bb = 5 : ui32, handshake.name = "cond_br57"} : <i1>, <i32>
    %trueResult_62, %falseResult_63 = cond_br %111, %falseResult_17 {handshake.bb = 5 : ui32, handshake.name = "cond_br58"} : <i1>, <>
    %trueResult_64, %falseResult_65 = cond_br %111, %falseResult_21 {handshake.bb = 5 : ui32, handshake.name = "cond_br59"} : <i1>, <>
    %trueResult_66, %falseResult_67 = cond_br %111, %falseResult_29 {handshake.bb = 5 : ui32, handshake.name = "cond_br60"} : <i1>, <>
    %trueResult_68, %falseResult_69 = cond_br %111, %falseResult_25 {handshake.bb = 5 : ui32, handshake.name = "cond_br61"} : <i1>, <>
    %104 = merge %falseResult_9 {handshake.bb = 5 : ui32, handshake.name = "merge3"} : <i32>
    %105 = merge %falseResult_7 {handshake.bb = 5 : ui32, handshake.name = "merge4"} : <i32>
    %result_70, %index_71 = control_merge [%falseResult_13]  {handshake.bb = 5 : ui32, handshake.name = "control_merge4"} : [<>] to <>, <i1>
    %106 = source {handshake.bb = 5 : ui32, handshake.name = "source8"} : <>
    %107 = constant %106 {handshake.bb = 5 : ui32, handshake.name = "constant17", value = 19 : i32} : <>, <i32>
    %108 = source {handshake.bb = 5 : ui32, handshake.name = "source9"} : <>
    %109 = constant %108 {handshake.bb = 5 : ui32, handshake.name = "constant18", value = 1 : i32} : <>, <i32>
    %110 = addi %104, %109 {handshake.bb = 5 : ui32, handshake.name = "addi7"} : <i32>
    %111 = cmpi ult, %110, %107 {handshake.bb = 5 : ui32, handshake.name = "cmpi1"} : <i32>
    %trueResult_72, %falseResult_73 = cond_br %111, %110 {handshake.bb = 5 : ui32, handshake.name = "cond_br15"} : <i1>, <i32>
    %trueResult_74, %falseResult_75 = cond_br %111, %105 {handshake.bb = 5 : ui32, handshake.name = "cond_br16"} : <i1>, <i32>
    %trueResult_76, %falseResult_77 = cond_br %111, %result_70 {handshake.bb = 5 : ui32, handshake.name = "cond_br17"} : <i1>, <>
    %112 = merge %falseResult_75 {handshake.bb = 6 : ui32, handshake.name = "merge5"} : <i32>
    %result_78, %index_79 = control_merge [%falseResult_77]  {handshake.bb = 6 : ui32, handshake.name = "control_merge5"} : [<>] to <>, <i1>
    end {handshake.bb = 6 : ui32, handshake.name = "end0"} %112, %memEnd_1, %memEnd, %arg4 : <i32>, <>, <>, <>
  }
}

