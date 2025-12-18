module {
  handshake.func @histogram(%arg0: memref<1000xi32>, %arg1: memref<1000xf32>, %arg2: memref<1000xf32>, %arg3: !handshake.channel<i32>, %arg4: !handshake.control<>, %arg5: !handshake.control<>, %arg6: !handshake.control<>, %arg7: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["feature", "weight", "hist", "n", "feature_start", "weight_start", "hist_start", "start"], cfg.edges = "[0,1][2,1][1,2,3,cmpi0]", resNames = ["feature_end", "weight_end", "hist_end", "end"]} {
    %outputs:2, %memEnd = mem_controller[%arg2 : memref<1000xf32>] %arg6 (%35, %addressResult_62, %addressResult_64, %dataResult_65) %result_66 {connectedBlocks = [2 : i32], handshake.name = "mem_controller2"} :    (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<f32>) -> (!handshake.channel<f32>, !handshake.control<>)
    %outputs_0, %memEnd_1 = mem_controller[%arg1 : memref<1000xf32>] %arg5 (%addressResult_44) %result_66 {connectedBlocks = [2 : i32], handshake.name = "mem_controller3"} :    (!handshake.channel<i32>) -> !handshake.channel<f32>
    %outputs_2, %memEnd_3 = mem_controller[%arg0 : memref<1000xi32>] %arg4 (%addressResult) %result_66 {connectedBlocks = [2 : i32], handshake.name = "mem_controller4"} :    (!handshake.channel<i32>) -> !handshake.channel<i32>
    %0 = constant %arg7 {handshake.bb = 0 : ui32, handshake.name = "constant1", value = 1000 : i32} : <>, <i32>
    %1 = constant %arg7 {handshake.bb = 0 : ui32, handshake.name = "constant3", value = 1000 : i32} : <>, <i32>
    %2 = constant %arg7 {handshake.bb = 0 : ui32, handshake.name = "constant4", value = 1000 : i32} : <>, <i32>
    %3 = constant %arg7 {handshake.bb = 0 : ui32, handshake.name = "constant5", value = 1000 : i32} : <>, <i32>
    %4 = constant %arg7 {handshake.bb = 0 : ui32, handshake.name = "constant6", value = 1000 : i32} : <>, <i32>
    %5 = constant %arg7 {handshake.bb = 0 : ui32, handshake.name = "constant7", value = 1000 : i32} : <>, <i32>
    %6 = constant %arg7 {handshake.bb = 0 : ui32, handshake.name = "constant8", value = 1000 : i32} : <>, <i32>
    %7 = constant %arg7 {handshake.bb = 0 : ui32, handshake.name = "constant9", value = 1000 : i32} : <>, <i32>
    %8 = constant %arg7 {handshake.bb = 0 : ui32, handshake.name = "constant0", value = 0 : i32} : <>, <i32>
    %9 = br %8 {handshake.bb = 0 : ui32, handshake.name = "br2"} : <i32>
    %10 = br %arg3 {handshake.bb = 0 : ui32, handshake.name = "br3"} : <i32>
    %11 = br %arg7 {handshake.bb = 0 : ui32, handshake.name = "br4"} : <>
    %12 = mux %29 [%7, %69] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux2"} : <i1>, [<i32>, <i32>] to <i32>
    %13 = mux %29 [%6, %72] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux3"} : <i1>, [<i32>, <i32>] to <i32>
    %14 = mux %29 [%5, %73] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux4"} : <i1>, [<i32>, <i32>] to <i32>
    %15 = mux %29 [%4, %70] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux5"} : <i1>, [<i32>, <i32>] to <i32>
    %16 = mux %29 [%3, %71] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux6"} : <i1>, [<i32>, <i32>] to <i32>
    %17 = mux %29 [%2, %67] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux7"} : <i1>, [<i32>, <i32>] to <i32>
    %18 = mux %29 [%arg7, %82] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux8"} : <i1>, [<>, <>] to <>
    %19 = mux %29 [%1, %66] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux9"} : <i1>, [<i32>, <i32>] to <i32>
    %20 = mux %29 [%0, %68] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux10"} : <i1>, [<i32>, <i32>] to <i32>
    %21 = mux %29 [%arg7, %76] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux11"} : <i1>, [<>, <>] to <>
    %22 = mux %29 [%arg7, %80] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux12"} : <i1>, [<>, <>] to <>
    %23 = mux %29 [%arg7, %75] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux13"} : <i1>, [<>, <>] to <>
    %24 = mux %29 [%arg7, %81] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux14"} : <i1>, [<>, <>] to <>
    %25 = mux %29 [%arg7, %77] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux15"} : <i1>, [<>, <>] to <>
    %26 = mux %29 [%arg7, %78] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux16"} : <i1>, [<>, <>] to <>
    %27 = mux %29 [%arg7, %74] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux17"} : <i1>, [<>, <>] to <>
    %28 = mux %29 [%arg7, %79] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux18"} : <i1>, [<>, <>] to <>
    %29 = init %32 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "init0"} : <i1>
    %30 = mux %index [%9, %84] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i32>, <i32>] to <i32>
    %31 = mux %index [%10, %85] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i1>, [<i32>, <i32>] to <i32>
    %result, %index = control_merge [%11, %86]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %32 = cmpi slt, %30, %31 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i32>
    %trueResult, %falseResult = cond_br %32, %31 {handshake.bb = 1 : ui32, handshake.name = "cond_br1"} : <i1>, <i32>
    %trueResult_4, %falseResult_5 = cond_br %32, %30 {handshake.bb = 1 : ui32, handshake.name = "cond_br2"} : <i1>, <i32>
    %trueResult_6, %falseResult_7 = cond_br %32, %result {handshake.bb = 1 : ui32, handshake.name = "cond_br3"} : <i1>, <>
    %trueResult_8, %falseResult_9 = cond_br %32, %12 {handshake.bb = 2 : ui32, handshake.name = "cond_br28"} : <i1>, <i32>
    %trueResult_10, %falseResult_11 = cond_br %32, %23 {handshake.bb = 2 : ui32, handshake.name = "cond_br29"} : <i1>, <>
    %trueResult_12, %falseResult_13 = cond_br %32, %22 {handshake.bb = 2 : ui32, handshake.name = "cond_br30"} : <i1>, <>
    %trueResult_14, %falseResult_15 = cond_br %32, %27 {handshake.bb = 2 : ui32, handshake.name = "cond_br31"} : <i1>, <>
    %trueResult_16, %falseResult_17 = cond_br %32, %20 {handshake.bb = 2 : ui32, handshake.name = "cond_br32"} : <i1>, <i32>
    %trueResult_18, %falseResult_19 = cond_br %32, %17 {handshake.bb = 2 : ui32, handshake.name = "cond_br33"} : <i1>, <i32>
    %trueResult_20, %falseResult_21 = cond_br %32, %19 {handshake.bb = 2 : ui32, handshake.name = "cond_br34"} : <i1>, <i32>
    %trueResult_22, %falseResult_23 = cond_br %32, %18 {handshake.bb = 2 : ui32, handshake.name = "cond_br35"} : <i1>, <>
    %trueResult_24, %falseResult_25 = cond_br %32, %28 {handshake.bb = 2 : ui32, handshake.name = "cond_br36"} : <i1>, <>
    %trueResult_26, %falseResult_27 = cond_br %32, %13 {handshake.bb = 2 : ui32, handshake.name = "cond_br37"} : <i1>, <i32>
    %trueResult_28, %falseResult_29 = cond_br %32, %25 {handshake.bb = 2 : ui32, handshake.name = "cond_br38"} : <i1>, <>
    %trueResult_30, %falseResult_31 = cond_br %32, %24 {handshake.bb = 2 : ui32, handshake.name = "cond_br39"} : <i1>, <>
    %trueResult_32, %falseResult_33 = cond_br %32, %26 {handshake.bb = 2 : ui32, handshake.name = "cond_br40"} : <i1>, <>
    %trueResult_34, %falseResult_35 = cond_br %32, %16 {handshake.bb = 2 : ui32, handshake.name = "cond_br41"} : <i1>, <i32>
    %trueResult_36, %falseResult_37 = cond_br %32, %15 {handshake.bb = 2 : ui32, handshake.name = "cond_br42"} : <i1>, <i32>
    %trueResult_38, %falseResult_39 = cond_br %32, %21 {handshake.bb = 2 : ui32, handshake.name = "cond_br43"} : <i1>, <>
    %trueResult_40, %falseResult_41 = cond_br %32, %14 {handshake.bb = 2 : ui32, handshake.name = "cond_br44"} : <i1>, <i32>
    %33 = merge %trueResult {handshake.bb = 2 : ui32, handshake.name = "merge0"} : <i32>
    %34 = merge %trueResult_4 {handshake.bb = 2 : ui32, handshake.name = "merge1"} : <i32>
    %result_42, %index_43 = control_merge [%trueResult_6]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>] to <>, <i1>
    %35 = constant %result_42 {handshake.bb = 2 : ui32, handshake.name = "constant10", value = 1 : i32} : <>, <i32>
    %36 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %37 = constant %36 {handshake.bb = 2 : ui32, handshake.name = "constant2", value = 1 : i32} : <>, <i32>
    %addressResult, %dataResult = load[%34] %outputs_2 {handshake.bb = 2 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i32>, <i32>, <i32>, <i32>
    %addressResult_44, %dataResult_45 = load[%34] %outputs_0 {handshake.bb = 2 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i32>, <f32>, <i32>, <f32>
    %38 = gate %dataResult, %trueResult_22 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "gate0"} : <i32>, !handshake.control<> to <i32>
    %39 = cmpi ne, %38, %trueResult_20 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi1"} : <i32>
    %40 = cmpi ne, %38, %trueResult_18 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi2"} : <i32>
    %41 = cmpi ne, %38, %trueResult_16 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi3"} : <i32>
    %42 = cmpi ne, %38, %trueResult_8 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi4"} : <i32>
    %43 = cmpi ne, %38, %trueResult_36 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi5"} : <i32>
    %44 = cmpi ne, %38, %trueResult_34 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi6"} : <i32>
    %45 = cmpi ne, %38, %trueResult_26 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi7"} : <i32>
    %46 = cmpi ne, %38, %trueResult_40 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi8"} : <i32>
    %trueResult_46, %falseResult_47 = cond_br %39, %trueResult_14 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br20"} : <i1>, <>
    %trueResult_48, %falseResult_49 = cond_br %40, %trueResult_10 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br21"} : <i1>, <>
    %trueResult_50, %falseResult_51 = cond_br %41, %trueResult_38 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br22"} : <i1>, <>
    %trueResult_52, %falseResult_53 = cond_br %42, %trueResult_28 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br23"} : <i1>, <>
    %trueResult_54, %falseResult_55 = cond_br %43, %trueResult_32 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br24"} : <i1>, <>
    %trueResult_56, %falseResult_57 = cond_br %44, %trueResult_24 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br25"} : <i1>, <>
    %trueResult_58, %falseResult_59 = cond_br %45, %trueResult_12 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br26"} : <i1>, <>
    %trueResult_60, %falseResult_61 = cond_br %46, %trueResult_30 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br27"} : <i1>, <>
    %47 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %48 = mux %39 [%falseResult_47, %47] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux19"} : <i1>, [<>, <>] to <>
    %49 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source2"} : <>
    %50 = mux %40 [%falseResult_49, %49] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux20"} : <i1>, [<>, <>] to <>
    %51 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source3"} : <>
    %52 = mux %41 [%falseResult_51, %51] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux21"} : <i1>, [<>, <>] to <>
    %53 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source4"} : <>
    %54 = mux %42 [%falseResult_53, %53] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux22"} : <i1>, [<>, <>] to <>
    %55 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source5"} : <>
    %56 = mux %43 [%falseResult_55, %55] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux23"} : <i1>, [<>, <>] to <>
    %57 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source6"} : <>
    %58 = mux %44 [%falseResult_57, %57] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux24"} : <i1>, [<>, <>] to <>
    %59 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source7"} : <>
    %60 = mux %45 [%falseResult_59, %59] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux25"} : <i1>, [<>, <>] to <>
    %61 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source8"} : <>
    %62 = mux %46 [%falseResult_61, %61] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux26"} : <i1>, [<>, <>] to <>
    %63 = join %48, %50, %52, %54, %56, %58, %60, %62 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "join0"} : <>
    %64 = gate %dataResult, %63 {handshake.bb = 2 : ui32, handshake.name = "gate1"} : <i32>, !handshake.control<> to <i32>
    %addressResult_62, %dataResult_63 = load[%64] %outputs#0 {handshake.bb = 2 : ui32, handshake.deps = #handshake<deps[["store1", 0, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load5"} : <i32>, <f32>, <i32>, <f32>
    %65 = addf %dataResult_63, %dataResult_45 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "addf0"} : <f32>
    %66 = buffer %dataResult, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer0"} : <i32>
    %67 = init %66 {handshake.bb = 2 : ui32, handshake.name = "init17"} : <i32>
    %68 = init %67 {handshake.bb = 2 : ui32, handshake.name = "init18"} : <i32>
    %69 = init %68 {handshake.bb = 2 : ui32, handshake.name = "init19"} : <i32>
    %70 = init %69 {handshake.bb = 2 : ui32, handshake.name = "init20"} : <i32>
    %71 = init %70 {handshake.bb = 2 : ui32, handshake.name = "init21"} : <i32>
    %72 = init %71 {handshake.bb = 2 : ui32, handshake.name = "init22"} : <i32>
    %73 = init %72 {handshake.bb = 2 : ui32, handshake.name = "init23"} : <i32>
    %74 = buffer %doneResult, bufferType = FIFO_BREAK_NONE, numSlots = 1 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "buffer1"} : <>
    %75 = init %74 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init24"} : <>
    %76 = init %75 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init25"} : <>
    %77 = init %76 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init26"} : <>
    %78 = init %77 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init27"} : <>
    %79 = init %78 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init28"} : <>
    %80 = init %79 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init29"} : <>
    %81 = init %80 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init30"} : <>
    %82 = init %81 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init31"} : <>
    %addressResult_64, %dataResult_65, %doneResult = store[%dataResult] %65 %outputs#1 {handshake.bb = 2 : ui32, handshake.deps = #handshake<deps[["load5", 0, false], ["store1", 0, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store1"} : <i32>, <f32>, <>, <i32>, <f32>, <>
    %83 = addi %34, %37 {handshake.bb = 2 : ui32, handshake.name = "addi0"} : <i32>
    %84 = br %83 {handshake.bb = 2 : ui32, handshake.name = "br5"} : <i32>
    %85 = br %33 {handshake.bb = 2 : ui32, handshake.name = "br6"} : <i32>
    %86 = br %result_42 {handshake.bb = 2 : ui32, handshake.name = "br7"} : <>
    %result_66, %index_67 = control_merge [%falseResult_7]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>] to <>, <i1>
    end {handshake.bb = 3 : ui32, handshake.name = "end0"} %memEnd_3, %memEnd_1, %memEnd, %arg7 : <>, <>, <>, <>
  }
}

