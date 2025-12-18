module {
  handshake.func @histogram(%arg0: memref<1000xi32>, %arg1: memref<1000xf32>, %arg2: memref<1000xf32>, %arg3: !handshake.channel<i32>, %arg4: !handshake.control<>, %arg5: !handshake.control<>, %arg6: !handshake.control<>, %arg7: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["feature", "weight", "hist", "n", "feature_start", "weight_start", "hist_start", "start"], cfg.edges = "[0,1][2,1][1,2,3,cmpi0]", resNames = ["feature_end", "weight_end", "hist_end", "end"]} {
    %outputs:2, %memEnd = mem_controller[%arg2 : memref<1000xf32>] %arg6 (%32, %addressResult_56, %addressResult_58, %dataResult_59) %result_60 {connectedBlocks = [2 : i32], handshake.name = "mem_controller2"} :    (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<f32>) -> (!handshake.channel<f32>, !handshake.control<>)
    %outputs_0, %memEnd_1 = mem_controller[%arg1 : memref<1000xf32>] %arg5 (%addressResult_40) %result_60 {connectedBlocks = [2 : i32], handshake.name = "mem_controller3"} :    (!handshake.channel<i32>) -> !handshake.channel<f32>
    %outputs_2, %memEnd_3 = mem_controller[%arg0 : memref<1000xi32>] %arg4 (%addressResult) %result_60 {connectedBlocks = [2 : i32], handshake.name = "mem_controller4"} :    (!handshake.channel<i32>) -> !handshake.channel<i32>
    %0 = constant %arg7 {handshake.bb = 0 : ui32, handshake.name = "constant1", value = 1000 : i32} : <>, <i32>
    %1 = constant %arg7 {handshake.bb = 0 : ui32, handshake.name = "constant3", value = 1000 : i32} : <>, <i32>
    %2 = constant %arg7 {handshake.bb = 0 : ui32, handshake.name = "constant4", value = 1000 : i32} : <>, <i32>
    %3 = constant %arg7 {handshake.bb = 0 : ui32, handshake.name = "constant5", value = 1000 : i32} : <>, <i32>
    %4 = constant %arg7 {handshake.bb = 0 : ui32, handshake.name = "constant6", value = 1000 : i32} : <>, <i32>
    %5 = constant %arg7 {handshake.bb = 0 : ui32, handshake.name = "constant7", value = 1000 : i32} : <>, <i32>
    %6 = constant %arg7 {handshake.bb = 0 : ui32, handshake.name = "constant8", value = 1000 : i32} : <>, <i32>
    %7 = constant %arg7 {handshake.bb = 0 : ui32, handshake.name = "constant0", value = 0 : i32} : <>, <i32>
    %8 = br %7 {handshake.bb = 0 : ui32, handshake.name = "br2"} : <i32>
    %9 = br %arg3 {handshake.bb = 0 : ui32, handshake.name = "br3"} : <i32>
    %10 = br %arg7 {handshake.bb = 0 : ui32, handshake.name = "br4"} : <>
    %11 = mux %26 [%6, %61] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux2"} : <i1>, [<i32>, <i32>] to <i32>
    %12 = mux %26 [%5, %65] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux3"} : <i1>, [<i32>, <i32>] to <i32>
    %13 = mux %26 [%4, %66] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux4"} : <i1>, [<i32>, <i32>] to <i32>
    %14 = mux %26 [%arg7, %74] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux5"} : <i1>, [<>, <>] to <>
    %15 = mux %26 [%3, %64] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux6"} : <i1>, [<i32>, <i32>] to <i32>
    %16 = mux %26 [%2, %60] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux7"} : <i1>, [<i32>, <i32>] to <i32>
    %17 = mux %26 [%1, %62] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux8"} : <i1>, [<i32>, <i32>] to <i32>
    %18 = mux %26 [%0, %63] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux9"} : <i1>, [<i32>, <i32>] to <i32>
    %19 = mux %26 [%arg7, %72] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux10"} : <i1>, [<>, <>] to <>
    %20 = mux %26 [%arg7, %68] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux11"} : <i1>, [<>, <>] to <>
    %21 = mux %26 [%arg7, %71] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux12"} : <i1>, [<>, <>] to <>
    %22 = mux %26 [%arg7, %69] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux13"} : <i1>, [<>, <>] to <>
    %23 = mux %26 [%arg7, %67] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux14"} : <i1>, [<>, <>] to <>
    %24 = mux %26 [%arg7, %70] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux15"} : <i1>, [<>, <>] to <>
    %25 = mux %26 [%arg7, %73] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux16"} : <i1>, [<>, <>] to <>
    %26 = init %29 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "init0"} : <i1>
    %27 = mux %index [%8, %76] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i32>, <i32>] to <i32>
    %28 = mux %index [%9, %77] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i1>, [<i32>, <i32>] to <i32>
    %result, %index = control_merge [%10, %78]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %29 = cmpi slt, %27, %28 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i32>
    %trueResult, %falseResult = cond_br %29, %28 {handshake.bb = 1 : ui32, handshake.name = "cond_br1"} : <i1>, <i32>
    %trueResult_4, %falseResult_5 = cond_br %29, %27 {handshake.bb = 1 : ui32, handshake.name = "cond_br2"} : <i1>, <i32>
    %trueResult_6, %falseResult_7 = cond_br %29, %result {handshake.bb = 1 : ui32, handshake.name = "cond_br3"} : <i1>, <>
    %trueResult_8, %falseResult_9 = cond_br %29, %25 {handshake.bb = 2 : ui32, handshake.name = "cond_br25"} : <i1>, <>
    %trueResult_10, %falseResult_11 = cond_br %29, %23 {handshake.bb = 2 : ui32, handshake.name = "cond_br26"} : <i1>, <>
    %trueResult_12, %falseResult_13 = cond_br %29, %20 {handshake.bb = 2 : ui32, handshake.name = "cond_br27"} : <i1>, <>
    %trueResult_14, %falseResult_15 = cond_br %29, %18 {handshake.bb = 2 : ui32, handshake.name = "cond_br28"} : <i1>, <i32>
    %trueResult_16, %falseResult_17 = cond_br %29, %15 {handshake.bb = 2 : ui32, handshake.name = "cond_br29"} : <i1>, <i32>
    %trueResult_18, %falseResult_19 = cond_br %29, %24 {handshake.bb = 2 : ui32, handshake.name = "cond_br30"} : <i1>, <>
    %trueResult_20, %falseResult_21 = cond_br %29, %12 {handshake.bb = 2 : ui32, handshake.name = "cond_br31"} : <i1>, <i32>
    %trueResult_22, %falseResult_23 = cond_br %29, %21 {handshake.bb = 2 : ui32, handshake.name = "cond_br32"} : <i1>, <>
    %trueResult_24, %falseResult_25 = cond_br %29, %16 {handshake.bb = 2 : ui32, handshake.name = "cond_br33"} : <i1>, <i32>
    %trueResult_26, %falseResult_27 = cond_br %29, %11 {handshake.bb = 2 : ui32, handshake.name = "cond_br34"} : <i1>, <i32>
    %trueResult_28, %falseResult_29 = cond_br %29, %14 {handshake.bb = 2 : ui32, handshake.name = "cond_br35"} : <i1>, <>
    %trueResult_30, %falseResult_31 = cond_br %29, %19 {handshake.bb = 2 : ui32, handshake.name = "cond_br36"} : <i1>, <>
    %trueResult_32, %falseResult_33 = cond_br %29, %17 {handshake.bb = 2 : ui32, handshake.name = "cond_br37"} : <i1>, <i32>
    %trueResult_34, %falseResult_35 = cond_br %29, %13 {handshake.bb = 2 : ui32, handshake.name = "cond_br38"} : <i1>, <i32>
    %trueResult_36, %falseResult_37 = cond_br %29, %22 {handshake.bb = 2 : ui32, handshake.name = "cond_br39"} : <i1>, <>
    %30 = merge %trueResult {handshake.bb = 2 : ui32, handshake.name = "merge0"} : <i32>
    %31 = merge %trueResult_4 {handshake.bb = 2 : ui32, handshake.name = "merge1"} : <i32>
    %result_38, %index_39 = control_merge [%trueResult_6]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>] to <>, <i1>
    %32 = constant %result_38 {handshake.bb = 2 : ui32, handshake.name = "constant9", value = 1 : i32} : <>, <i32>
    %33 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %34 = constant %33 {handshake.bb = 2 : ui32, handshake.name = "constant2", value = 1 : i32} : <>, <i32>
    %addressResult, %dataResult = load[%31] %outputs_2 {handshake.bb = 2 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i32>, <i32>, <i32>, <i32>
    %addressResult_40, %dataResult_41 = load[%31] %outputs_0 {handshake.bb = 2 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i32>, <f32>, <i32>, <f32>
    %35 = gate %dataResult, %trueResult_28 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "gate0"} : <i32>, !handshake.control<> to <i32>
    %36 = cmpi ne, %35, %trueResult_24 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi1"} : <i32>
    %37 = cmpi ne, %35, %trueResult_26 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi2"} : <i32>
    %38 = cmpi ne, %35, %trueResult_32 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi3"} : <i32>
    %39 = cmpi ne, %35, %trueResult_14 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi4"} : <i32>
    %40 = cmpi ne, %35, %trueResult_16 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi5"} : <i32>
    %41 = cmpi ne, %35, %trueResult_20 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi6"} : <i32>
    %42 = cmpi ne, %35, %trueResult_34 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi7"} : <i32>
    %trueResult_42, %falseResult_43 = cond_br %36, %trueResult_10 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br18"} : <i1>, <>
    %trueResult_44, %falseResult_45 = cond_br %37, %trueResult_12 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br19"} : <i1>, <>
    %trueResult_46, %falseResult_47 = cond_br %38, %trueResult_36 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br20"} : <i1>, <>
    %trueResult_48, %falseResult_49 = cond_br %39, %trueResult_18 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br21"} : <i1>, <>
    %trueResult_50, %falseResult_51 = cond_br %40, %trueResult_22 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br22"} : <i1>, <>
    %trueResult_52, %falseResult_53 = cond_br %41, %trueResult_30 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br23"} : <i1>, <>
    %trueResult_54, %falseResult_55 = cond_br %42, %trueResult_8 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br24"} : <i1>, <>
    %43 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %44 = mux %36 [%falseResult_43, %43] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux17"} : <i1>, [<>, <>] to <>
    %45 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source2"} : <>
    %46 = mux %37 [%falseResult_45, %45] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux18"} : <i1>, [<>, <>] to <>
    %47 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source3"} : <>
    %48 = mux %38 [%falseResult_47, %47] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux19"} : <i1>, [<>, <>] to <>
    %49 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source4"} : <>
    %50 = mux %39 [%falseResult_49, %49] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux20"} : <i1>, [<>, <>] to <>
    %51 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source5"} : <>
    %52 = mux %40 [%falseResult_51, %51] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux21"} : <i1>, [<>, <>] to <>
    %53 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source6"} : <>
    %54 = mux %41 [%falseResult_53, %53] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux22"} : <i1>, [<>, <>] to <>
    %55 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source7"} : <>
    %56 = mux %42 [%falseResult_55, %55] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux23"} : <i1>, [<>, <>] to <>
    %57 = join %44, %46, %48, %50, %52, %54, %56 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "join0"} : <>
    %58 = gate %dataResult, %57 {handshake.bb = 2 : ui32, handshake.name = "gate1"} : <i32>, !handshake.control<> to <i32>
    %addressResult_56, %dataResult_57 = load[%58] %outputs#0 {handshake.bb = 2 : ui32, handshake.deps = #handshake<deps[["store1", 0, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load5"} : <i32>, <f32>, <i32>, <f32>
    %59 = addf %dataResult_57, %dataResult_41 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "addf0"} : <f32>
    %60 = buffer %dataResult, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer0"} : <i32>
    %61 = init %60 {handshake.bb = 2 : ui32, handshake.name = "init15"} : <i32>
    %62 = init %61 {handshake.bb = 2 : ui32, handshake.name = "init16"} : <i32>
    %63 = init %62 {handshake.bb = 2 : ui32, handshake.name = "init17"} : <i32>
    %64 = init %63 {handshake.bb = 2 : ui32, handshake.name = "init18"} : <i32>
    %65 = init %64 {handshake.bb = 2 : ui32, handshake.name = "init19"} : <i32>
    %66 = init %65 {handshake.bb = 2 : ui32, handshake.name = "init20"} : <i32>
    %67 = buffer %doneResult, bufferType = FIFO_BREAK_NONE, numSlots = 1 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "buffer1"} : <>
    %68 = init %67 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init21"} : <>
    %69 = init %68 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init22"} : <>
    %70 = init %69 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init23"} : <>
    %71 = init %70 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init24"} : <>
    %72 = init %71 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init25"} : <>
    %73 = init %72 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init26"} : <>
    %74 = init %73 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init27"} : <>
    %addressResult_58, %dataResult_59, %doneResult = store[%dataResult] %59 %outputs#1 {handshake.bb = 2 : ui32, handshake.deps = #handshake<deps[["load5", 0, false], ["store1", 0, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store1"} : <i32>, <f32>, <>, <i32>, <f32>, <>
    %75 = addi %31, %34 {handshake.bb = 2 : ui32, handshake.name = "addi0"} : <i32>
    %76 = br %75 {handshake.bb = 2 : ui32, handshake.name = "br5"} : <i32>
    %77 = br %30 {handshake.bb = 2 : ui32, handshake.name = "br6"} : <i32>
    %78 = br %result_38 {handshake.bb = 2 : ui32, handshake.name = "br7"} : <>
    %result_60, %index_61 = control_merge [%falseResult_7]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>] to <>, <i1>
    end {handshake.bb = 3 : ui32, handshake.name = "end0"} %memEnd_3, %memEnd_1, %memEnd, %arg7 : <>, <>, <>, <>
  }
}

