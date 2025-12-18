module {
  handshake.func @histogram(%arg0: memref<1000xi32>, %arg1: memref<1000xf32>, %arg2: memref<1000xf32>, %arg3: !handshake.channel<i32>, %arg4: !handshake.control<>, %arg5: !handshake.control<>, %arg6: !handshake.control<>, %arg7: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["feature", "weight", "hist", "n", "feature_start", "weight_start", "hist_start", "start"], cfg.edges = "[0,1][2,1][1,2,3,cmpi0]", resNames = ["feature_end", "weight_end", "hist_end", "end"]} {
    %outputs:2, %memEnd = mem_controller[%arg2 : memref<1000xf32>] %arg6 (%29, %addressResult_50, %addressResult_52, %dataResult_53) %result_54 {connectedBlocks = [2 : i32], handshake.name = "mem_controller2"} :    (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<f32>) -> (!handshake.channel<f32>, !handshake.control<>)
    %outputs_0, %memEnd_1 = mem_controller[%arg1 : memref<1000xf32>] %arg5 (%addressResult_36) %result_54 {connectedBlocks = [2 : i32], handshake.name = "mem_controller3"} :    (!handshake.channel<i32>) -> !handshake.channel<f32>
    %outputs_2, %memEnd_3 = mem_controller[%arg0 : memref<1000xi32>] %arg4 (%addressResult) %result_54 {connectedBlocks = [2 : i32], handshake.name = "mem_controller4"} :    (!handshake.channel<i32>) -> !handshake.channel<i32>
    %0 = constant %arg7 {handshake.bb = 0 : ui32, handshake.name = "constant1", value = 1000 : i32} : <>, <i32>
    %1 = constant %arg7 {handshake.bb = 0 : ui32, handshake.name = "constant3", value = 1000 : i32} : <>, <i32>
    %2 = constant %arg7 {handshake.bb = 0 : ui32, handshake.name = "constant4", value = 1000 : i32} : <>, <i32>
    %3 = constant %arg7 {handshake.bb = 0 : ui32, handshake.name = "constant5", value = 1000 : i32} : <>, <i32>
    %4 = constant %arg7 {handshake.bb = 0 : ui32, handshake.name = "constant6", value = 1000 : i32} : <>, <i32>
    %5 = constant %arg7 {handshake.bb = 0 : ui32, handshake.name = "constant7", value = 1000 : i32} : <>, <i32>
    %6 = constant %arg7 {handshake.bb = 0 : ui32, handshake.name = "constant0", value = 0 : i32} : <>, <i32>
    %7 = br %6 {handshake.bb = 0 : ui32, handshake.name = "br2"} : <i32>
    %8 = br %arg3 {handshake.bb = 0 : ui32, handshake.name = "br3"} : <i32>
    %9 = br %arg7 {handshake.bb = 0 : ui32, handshake.name = "br4"} : <>
    %10 = mux %23 [%5, %55] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux2"} : <i1>, [<i32>, <i32>] to <i32>
    %11 = mux %23 [%4, %57] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux3"} : <i1>, [<i32>, <i32>] to <i32>
    %12 = mux %23 [%3, %56] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux4"} : <i1>, [<i32>, <i32>] to <i32>
    %13 = mux %23 [%2, %54] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux5"} : <i1>, [<i32>, <i32>] to <i32>
    %14 = mux %23 [%1, %58] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux6"} : <i1>, [<i32>, <i32>] to <i32>
    %15 = mux %23 [%arg7, %66] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux7"} : <i1>, [<>, <>] to <>
    %16 = mux %23 [%0, %59] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux8"} : <i1>, [<i32>, <i32>] to <i32>
    %17 = mux %23 [%arg7, %65] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux9"} : <i1>, [<>, <>] to <>
    %18 = mux %23 [%arg7, %61] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux10"} : <i1>, [<>, <>] to <>
    %19 = mux %23 [%arg7, %64] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux11"} : <i1>, [<>, <>] to <>
    %20 = mux %23 [%arg7, %62] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux12"} : <i1>, [<>, <>] to <>
    %21 = mux %23 [%arg7, %60] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux13"} : <i1>, [<>, <>] to <>
    %22 = mux %23 [%arg7, %63] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux14"} : <i1>, [<>, <>] to <>
    %23 = init %26 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "init0"} : <i1>
    %24 = mux %index [%7, %68] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i32>, <i32>] to <i32>
    %25 = mux %index [%8, %69] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i1>, [<i32>, <i32>] to <i32>
    %result, %index = control_merge [%9, %70]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %26 = cmpi slt, %24, %25 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i32>
    %trueResult, %falseResult = cond_br %26, %25 {handshake.bb = 1 : ui32, handshake.name = "cond_br1"} : <i1>, <i32>
    %trueResult_4, %falseResult_5 = cond_br %26, %24 {handshake.bb = 1 : ui32, handshake.name = "cond_br2"} : <i1>, <i32>
    %trueResult_6, %falseResult_7 = cond_br %26, %result {handshake.bb = 1 : ui32, handshake.name = "cond_br3"} : <i1>, <>
    %trueResult_8, %falseResult_9 = cond_br %26, %10 {handshake.bb = 2 : ui32, handshake.name = "cond_br22"} : <i1>, <i32>
    %trueResult_10, %falseResult_11 = cond_br %26, %14 {handshake.bb = 2 : ui32, handshake.name = "cond_br23"} : <i1>, <i32>
    %trueResult_12, %falseResult_13 = cond_br %26, %22 {handshake.bb = 2 : ui32, handshake.name = "cond_br24"} : <i1>, <>
    %trueResult_14, %falseResult_15 = cond_br %26, %21 {handshake.bb = 2 : ui32, handshake.name = "cond_br25"} : <i1>, <>
    %trueResult_16, %falseResult_17 = cond_br %26, %13 {handshake.bb = 2 : ui32, handshake.name = "cond_br26"} : <i1>, <i32>
    %trueResult_18, %falseResult_19 = cond_br %26, %16 {handshake.bb = 2 : ui32, handshake.name = "cond_br27"} : <i1>, <i32>
    %trueResult_20, %falseResult_21 = cond_br %26, %20 {handshake.bb = 2 : ui32, handshake.name = "cond_br28"} : <i1>, <>
    %trueResult_22, %falseResult_23 = cond_br %26, %11 {handshake.bb = 2 : ui32, handshake.name = "cond_br29"} : <i1>, <i32>
    %trueResult_24, %falseResult_25 = cond_br %26, %19 {handshake.bb = 2 : ui32, handshake.name = "cond_br30"} : <i1>, <>
    %trueResult_26, %falseResult_27 = cond_br %26, %18 {handshake.bb = 2 : ui32, handshake.name = "cond_br31"} : <i1>, <>
    %trueResult_28, %falseResult_29 = cond_br %26, %15 {handshake.bb = 2 : ui32, handshake.name = "cond_br32"} : <i1>, <>
    %trueResult_30, %falseResult_31 = cond_br %26, %17 {handshake.bb = 2 : ui32, handshake.name = "cond_br33"} : <i1>, <>
    %trueResult_32, %falseResult_33 = cond_br %26, %12 {handshake.bb = 2 : ui32, handshake.name = "cond_br34"} : <i1>, <i32>
    %27 = merge %trueResult {handshake.bb = 2 : ui32, handshake.name = "merge0"} : <i32>
    %28 = merge %trueResult_4 {handshake.bb = 2 : ui32, handshake.name = "merge1"} : <i32>
    %result_34, %index_35 = control_merge [%trueResult_6]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>] to <>, <i1>
    %29 = constant %result_34 {handshake.bb = 2 : ui32, handshake.name = "constant8", value = 1 : i32} : <>, <i32>
    %30 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %31 = constant %30 {handshake.bb = 2 : ui32, handshake.name = "constant2", value = 1 : i32} : <>, <i32>
    %addressResult, %dataResult = load[%28] %outputs_2 {handshake.bb = 2 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i32>, <i32>, <i32>, <i32>
    %addressResult_36, %dataResult_37 = load[%28] %outputs_0 {handshake.bb = 2 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i32>, <f32>, <i32>, <f32>
    %32 = gate %dataResult, %trueResult_28 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "gate0"} : <i32>, !handshake.control<> to <i32>
    %33 = cmpi ne, %32, %trueResult_16 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi1"} : <i32>
    %34 = cmpi ne, %32, %trueResult_8 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi2"} : <i32>
    %35 = cmpi ne, %32, %trueResult_32 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi3"} : <i32>
    %36 = cmpi ne, %32, %trueResult_22 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi4"} : <i32>
    %37 = cmpi ne, %32, %trueResult_10 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi5"} : <i32>
    %38 = cmpi ne, %32, %trueResult_18 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi6"} : <i32>
    %trueResult_38, %falseResult_39 = cond_br %33, %trueResult_14 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br16"} : <i1>, <>
    %trueResult_40, %falseResult_41 = cond_br %34, %trueResult_26 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br17"} : <i1>, <>
    %trueResult_42, %falseResult_43 = cond_br %35, %trueResult_20 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br18"} : <i1>, <>
    %trueResult_44, %falseResult_45 = cond_br %36, %trueResult_12 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br19"} : <i1>, <>
    %trueResult_46, %falseResult_47 = cond_br %37, %trueResult_24 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br20"} : <i1>, <>
    %trueResult_48, %falseResult_49 = cond_br %38, %trueResult_30 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br21"} : <i1>, <>
    %39 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %40 = mux %33 [%falseResult_39, %39] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux15"} : <i1>, [<>, <>] to <>
    %41 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source2"} : <>
    %42 = mux %34 [%falseResult_41, %41] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux16"} : <i1>, [<>, <>] to <>
    %43 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source3"} : <>
    %44 = mux %35 [%falseResult_43, %43] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux17"} : <i1>, [<>, <>] to <>
    %45 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source4"} : <>
    %46 = mux %36 [%falseResult_45, %45] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux18"} : <i1>, [<>, <>] to <>
    %47 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source5"} : <>
    %48 = mux %37 [%falseResult_47, %47] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux19"} : <i1>, [<>, <>] to <>
    %49 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source6"} : <>
    %50 = mux %38 [%falseResult_49, %49] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux20"} : <i1>, [<>, <>] to <>
    %51 = join %40, %42, %44, %46, %48, %50 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "join0"} : <>
    %52 = gate %dataResult, %51 {handshake.bb = 2 : ui32, handshake.name = "gate1"} : <i32>, !handshake.control<> to <i32>
    %addressResult_50, %dataResult_51 = load[%52] %outputs#0 {handshake.bb = 2 : ui32, handshake.deps = #handshake<deps[["store1", 0, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load5"} : <i32>, <f32>, <i32>, <f32>
    %53 = addf %dataResult_51, %dataResult_37 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "addf0"} : <f32>
    %54 = buffer %dataResult, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer0"} : <i32>
    %55 = init %54 {handshake.bb = 2 : ui32, handshake.name = "init13"} : <i32>
    %56 = init %55 {handshake.bb = 2 : ui32, handshake.name = "init14"} : <i32>
    %57 = init %56 {handshake.bb = 2 : ui32, handshake.name = "init15"} : <i32>
    %58 = init %57 {handshake.bb = 2 : ui32, handshake.name = "init16"} : <i32>
    %59 = init %58 {handshake.bb = 2 : ui32, handshake.name = "init17"} : <i32>
    %60 = buffer %doneResult, bufferType = FIFO_BREAK_NONE, numSlots = 1 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "buffer1"} : <>
    %61 = init %60 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init18"} : <>
    %62 = init %61 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init19"} : <>
    %63 = init %62 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init20"} : <>
    %64 = init %63 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init21"} : <>
    %65 = init %64 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init22"} : <>
    %66 = init %65 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init23"} : <>
    %addressResult_52, %dataResult_53, %doneResult = store[%dataResult] %53 %outputs#1 {handshake.bb = 2 : ui32, handshake.deps = #handshake<deps[["load5", 0, false], ["store1", 0, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store1"} : <i32>, <f32>, <>, <i32>, <f32>, <>
    %67 = addi %28, %31 {handshake.bb = 2 : ui32, handshake.name = "addi0"} : <i32>
    %68 = br %67 {handshake.bb = 2 : ui32, handshake.name = "br5"} : <i32>
    %69 = br %27 {handshake.bb = 2 : ui32, handshake.name = "br6"} : <i32>
    %70 = br %result_34 {handshake.bb = 2 : ui32, handshake.name = "br7"} : <>
    %result_54, %index_55 = control_merge [%falseResult_7]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>] to <>, <i1>
    end {handshake.bb = 3 : ui32, handshake.name = "end0"} %memEnd_3, %memEnd_1, %memEnd, %arg7 : <>, <>, <>, <>
  }
}

