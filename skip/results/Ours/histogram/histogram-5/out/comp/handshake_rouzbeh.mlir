module {
  handshake.func @histogram(%arg0: memref<1000xi32>, %arg1: memref<1000xf32>, %arg2: memref<1000xf32>, %arg3: !handshake.channel<i32>, %arg4: !handshake.control<>, %arg5: !handshake.control<>, %arg6: !handshake.control<>, %arg7: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["feature", "weight", "hist", "n", "feature_start", "weight_start", "hist_start", "start"], cfg.edges = "[0,1][2,1][1,2,3,cmpi0]", resNames = ["feature_end", "weight_end", "hist_end", "end"]} {
    %outputs:2, %memEnd = mem_controller[%arg2 : memref<1000xf32>] %arg6 (%26, %addressResult_44, %addressResult_46, %dataResult_47) %result_48 {connectedBlocks = [2 : i32], handshake.name = "mem_controller2"} :    (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<f32>) -> (!handshake.channel<f32>, !handshake.control<>)
    %outputs_0, %memEnd_1 = mem_controller[%arg1 : memref<1000xf32>] %arg5 (%addressResult_32) %result_48 {connectedBlocks = [2 : i32], handshake.name = "mem_controller3"} :    (!handshake.channel<i32>) -> !handshake.channel<f32>
    %outputs_2, %memEnd_3 = mem_controller[%arg0 : memref<1000xi32>] %arg4 (%addressResult) %result_48 {connectedBlocks = [2 : i32], handshake.name = "mem_controller4"} :    (!handshake.channel<i32>) -> !handshake.channel<i32>
    %0 = constant %arg7 {handshake.bb = 0 : ui32, handshake.name = "constant1", value = 1000 : i32} : <>, <i32>
    %1 = constant %arg7 {handshake.bb = 0 : ui32, handshake.name = "constant3", value = 1000 : i32} : <>, <i32>
    %2 = constant %arg7 {handshake.bb = 0 : ui32, handshake.name = "constant4", value = 1000 : i32} : <>, <i32>
    %3 = constant %arg7 {handshake.bb = 0 : ui32, handshake.name = "constant5", value = 1000 : i32} : <>, <i32>
    %4 = constant %arg7 {handshake.bb = 0 : ui32, handshake.name = "constant6", value = 1000 : i32} : <>, <i32>
    %5 = constant %arg7 {handshake.bb = 0 : ui32, handshake.name = "constant0", value = 0 : i32} : <>, <i32>
    %6 = br %5 {handshake.bb = 0 : ui32, handshake.name = "br2"} : <i32>
    %7 = br %arg3 {handshake.bb = 0 : ui32, handshake.name = "br3"} : <i32>
    %8 = br %arg7 {handshake.bb = 0 : ui32, handshake.name = "br4"} : <>
    %9 = mux %20 [%4, %50] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux2"} : <i1>, [<i32>, <i32>] to <i32>
    %10 = mux %20 [%3, %49] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux3"} : <i1>, [<i32>, <i32>] to <i32>
    %11 = mux %20 [%2, %51] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux4"} : <i1>, [<i32>, <i32>] to <i32>
    %12 = mux %20 [%arg7, %58] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux5"} : <i1>, [<>, <>] to <>
    %13 = mux %20 [%1, %48] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux6"} : <i1>, [<i32>, <i32>] to <i32>
    %14 = mux %20 [%0, %52] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux7"} : <i1>, [<i32>, <i32>] to <i32>
    %15 = mux %20 [%arg7, %54] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux8"} : <i1>, [<>, <>] to <>
    %16 = mux %20 [%arg7, %55] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux9"} : <i1>, [<>, <>] to <>
    %17 = mux %20 [%arg7, %56] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux10"} : <i1>, [<>, <>] to <>
    %18 = mux %20 [%arg7, %57] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux11"} : <i1>, [<>, <>] to <>
    %19 = mux %20 [%arg7, %53] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux12"} : <i1>, [<>, <>] to <>
    %20 = init %23 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "init0"} : <i1>
    %21 = mux %index [%6, %60] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i32>, <i32>] to <i32>
    %22 = mux %index [%7, %61] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i1>, [<i32>, <i32>] to <i32>
    %result, %index = control_merge [%8, %62]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %23 = cmpi slt, %21, %22 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i32>
    %trueResult, %falseResult = cond_br %23, %22 {handshake.bb = 1 : ui32, handshake.name = "cond_br1"} : <i1>, <i32>
    %trueResult_4, %falseResult_5 = cond_br %23, %21 {handshake.bb = 1 : ui32, handshake.name = "cond_br2"} : <i1>, <i32>
    %trueResult_6, %falseResult_7 = cond_br %23, %result {handshake.bb = 1 : ui32, handshake.name = "cond_br3"} : <i1>, <>
    %trueResult_8, %falseResult_9 = cond_br %23, %17 {handshake.bb = 2 : ui32, handshake.name = "cond_br19"} : <i1>, <>
    %trueResult_10, %falseResult_11 = cond_br %23, %19 {handshake.bb = 2 : ui32, handshake.name = "cond_br20"} : <i1>, <>
    %trueResult_12, %falseResult_13 = cond_br %23, %13 {handshake.bb = 2 : ui32, handshake.name = "cond_br21"} : <i1>, <i32>
    %trueResult_14, %falseResult_15 = cond_br %23, %12 {handshake.bb = 2 : ui32, handshake.name = "cond_br22"} : <i1>, <>
    %trueResult_16, %falseResult_17 = cond_br %23, %11 {handshake.bb = 2 : ui32, handshake.name = "cond_br23"} : <i1>, <i32>
    %trueResult_18, %falseResult_19 = cond_br %23, %14 {handshake.bb = 2 : ui32, handshake.name = "cond_br24"} : <i1>, <i32>
    %trueResult_20, %falseResult_21 = cond_br %23, %10 {handshake.bb = 2 : ui32, handshake.name = "cond_br25"} : <i1>, <i32>
    %trueResult_22, %falseResult_23 = cond_br %23, %9 {handshake.bb = 2 : ui32, handshake.name = "cond_br26"} : <i1>, <i32>
    %trueResult_24, %falseResult_25 = cond_br %23, %16 {handshake.bb = 2 : ui32, handshake.name = "cond_br27"} : <i1>, <>
    %trueResult_26, %falseResult_27 = cond_br %23, %15 {handshake.bb = 2 : ui32, handshake.name = "cond_br28"} : <i1>, <>
    %trueResult_28, %falseResult_29 = cond_br %23, %18 {handshake.bb = 2 : ui32, handshake.name = "cond_br29"} : <i1>, <>
    %24 = merge %trueResult {handshake.bb = 2 : ui32, handshake.name = "merge0"} : <i32>
    %25 = merge %trueResult_4 {handshake.bb = 2 : ui32, handshake.name = "merge1"} : <i32>
    %result_30, %index_31 = control_merge [%trueResult_6]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>] to <>, <i1>
    %26 = constant %result_30 {handshake.bb = 2 : ui32, handshake.name = "constant7", value = 1 : i32} : <>, <i32>
    %27 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %28 = constant %27 {handshake.bb = 2 : ui32, handshake.name = "constant2", value = 1 : i32} : <>, <i32>
    %addressResult, %dataResult = load[%25] %outputs_2 {handshake.bb = 2 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i32>, <i32>, <i32>, <i32>
    %addressResult_32, %dataResult_33 = load[%25] %outputs_0 {handshake.bb = 2 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i32>, <f32>, <i32>, <f32>
    %29 = gate %dataResult, %trueResult_14 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "gate0"} : <i32>, !handshake.control<> to <i32>
    %30 = cmpi ne, %29, %trueResult_12 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi1"} : <i32>
    %31 = cmpi ne, %29, %trueResult_20 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi2"} : <i32>
    %32 = cmpi ne, %29, %trueResult_22 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi3"} : <i32>
    %33 = cmpi ne, %29, %trueResult_16 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi4"} : <i32>
    %34 = cmpi ne, %29, %trueResult_18 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi5"} : <i32>
    %trueResult_34, %falseResult_35 = cond_br %30, %trueResult_10 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br14"} : <i1>, <>
    %trueResult_36, %falseResult_37 = cond_br %31, %trueResult_26 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br15"} : <i1>, <>
    %trueResult_38, %falseResult_39 = cond_br %32, %trueResult_24 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br16"} : <i1>, <>
    %trueResult_40, %falseResult_41 = cond_br %33, %trueResult_8 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br17"} : <i1>, <>
    %trueResult_42, %falseResult_43 = cond_br %34, %trueResult_28 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br18"} : <i1>, <>
    %35 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %36 = mux %30 [%falseResult_35, %35] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux13"} : <i1>, [<>, <>] to <>
    %37 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source2"} : <>
    %38 = mux %31 [%falseResult_37, %37] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux14"} : <i1>, [<>, <>] to <>
    %39 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source3"} : <>
    %40 = mux %32 [%falseResult_39, %39] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux15"} : <i1>, [<>, <>] to <>
    %41 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source4"} : <>
    %42 = mux %33 [%falseResult_41, %41] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux16"} : <i1>, [<>, <>] to <>
    %43 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source5"} : <>
    %44 = mux %34 [%falseResult_43, %43] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux17"} : <i1>, [<>, <>] to <>
    %45 = join %36, %38, %40, %42, %44 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "join0"} : <>
    %46 = gate %dataResult, %45 {handshake.bb = 2 : ui32, handshake.name = "gate1"} : <i32>, !handshake.control<> to <i32>
    %addressResult_44, %dataResult_45 = load[%46] %outputs#0 {handshake.bb = 2 : ui32, handshake.deps = #handshake<deps[["store1", 0, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load5"} : <i32>, <f32>, <i32>, <f32>
    %47 = addf %dataResult_45, %dataResult_33 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "addf0"} : <f32>
    %48 = buffer %dataResult, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer0"} : <i32>
    %49 = init %48 {handshake.bb = 2 : ui32, handshake.name = "init11"} : <i32>
    %50 = init %49 {handshake.bb = 2 : ui32, handshake.name = "init12"} : <i32>
    %51 = init %50 {handshake.bb = 2 : ui32, handshake.name = "init13"} : <i32>
    %52 = init %51 {handshake.bb = 2 : ui32, handshake.name = "init14"} : <i32>
    %53 = buffer %doneResult, bufferType = FIFO_BREAK_NONE, numSlots = 1 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "buffer1"} : <>
    %54 = init %53 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init15"} : <>
    %55 = init %54 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init16"} : <>
    %56 = init %55 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init17"} : <>
    %57 = init %56 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init18"} : <>
    %58 = init %57 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init19"} : <>
    %addressResult_46, %dataResult_47, %doneResult = store[%dataResult] %47 %outputs#1 {handshake.bb = 2 : ui32, handshake.deps = #handshake<deps[["load5", 0, false], ["store1", 0, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store1"} : <i32>, <f32>, <>, <i32>, <f32>, <>
    %59 = addi %25, %28 {handshake.bb = 2 : ui32, handshake.name = "addi0"} : <i32>
    %60 = br %59 {handshake.bb = 2 : ui32, handshake.name = "br5"} : <i32>
    %61 = br %24 {handshake.bb = 2 : ui32, handshake.name = "br6"} : <i32>
    %62 = br %result_30 {handshake.bb = 2 : ui32, handshake.name = "br7"} : <>
    %result_48, %index_49 = control_merge [%falseResult_7]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>] to <>, <i1>
    end {handshake.bb = 3 : ui32, handshake.name = "end0"} %memEnd_3, %memEnd_1, %memEnd, %arg7 : <>, <>, <>, <>
  }
}

