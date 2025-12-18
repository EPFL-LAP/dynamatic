module {
  handshake.func @atax(%arg0: memref<400xf32>, %arg1: memref<20xf32>, %arg2: memref<20xf32>, %arg3: memref<20xf32>, %arg4: !handshake.control<>, %arg5: !handshake.control<>, %arg6: !handshake.control<>, %arg7: !handshake.control<>, %arg8: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["A", "x", "y", "tmp", "A_start", "x_start", "y_start", "tmp_start", "start"], cfg.edges = "[0,1][2,2,3,cmpi0][4,1,5,cmpi2][1,2][3,3,4,cmpi1]", resNames = ["A_end", "x_end", "y_end", "tmp_end", "end"]} {
    %outputs:2, %memEnd = mem_controller[%arg3 : memref<20xf32>] %arg7 (%addressResult, %97, %addressResult_78, %dataResult_79) %result_85 {connectedBlocks = [1 : i32, 4 : i32], handshake.name = "mem_controller2"} :    (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<f32>) -> (!handshake.channel<f32>, !handshake.control<>)
    %outputs_0:2, %memEnd_1 = mem_controller[%arg2 : memref<20xf32>] %arg6 (%59, %addressResult_42, %addressResult_46, %dataResult_47) %result_85 {connectedBlocks = [3 : i32], handshake.name = "mem_controller3"} :    (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<f32>) -> (!handshake.channel<f32>, !handshake.control<>)
    %outputs_2, %memEnd_3 = mem_controller[%arg1 : memref<20xf32>] %arg5 (%addressResult_10) %result_85 {connectedBlocks = [2 : i32], handshake.name = "mem_controller4"} :    (!handshake.channel<i32>) -> !handshake.channel<f32>
    %outputs_4:2, %memEnd_5 = mem_controller[%arg0 : memref<400xf32>] %arg4 (%addressResult_8, %addressResult_44) %result_85 {connectedBlocks = [2 : i32, 3 : i32], handshake.name = "mem_controller5"} :    (!handshake.channel<i32>, !handshake.channel<i32>) -> (!handshake.channel<f32>, !handshake.channel<f32>)
    %0 = constant %arg8 {handshake.bb = 0 : ui32, handshake.name = "constant1", value = 1000 : i32} : <>, <i32>
    %1 = constant %arg8 {handshake.bb = 0 : ui32, handshake.name = "constant3", value = 1000 : i32} : <>, <i32>
    %2 = constant %arg8 {handshake.bb = 0 : ui32, handshake.name = "constant18", value = 1000 : i32} : <>, <i32>
    %3 = constant %arg8 {handshake.bb = 0 : ui32, handshake.name = "constant4", value = 0 : i32} : <>, <i32>
    %4 = br %3 {handshake.bb = 0 : ui32, handshake.name = "br2"} : <i32>
    %5 = br %arg8 {handshake.bb = 0 : ui32, handshake.name = "br3"} : <>
    %6 = mux %13 [%2, %trueResult_56] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux7"} : <i1>, [<i32>, <i32>] to <i32>
    %7 = mux %13 [%arg8, %trueResult_68] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux10"} : <i1>, [<>, <>] to <>
    %8 = mux %13 [%1, %trueResult_64] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux11"} : <i1>, [<i32>, <i32>] to <i32>
    %9 = mux %13 [%0, %trueResult_66] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux13"} : <i1>, [<i32>, <i32>] to <i32>
    %10 = mux %13 [%arg8, %trueResult_60] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux15"} : <i1>, [<>, <>] to <>
    %11 = mux %13 [%arg8, %trueResult_58] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux17"} : <i1>, [<>, <>] to <>
    %12 = mux %13 [%arg8, %trueResult_62] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux18"} : <i1>, [<>, <>] to <>
    %13 = init %115 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "init0"} : <i1>
    %14:2 = unbundle %dataResult  {handshake.bb = 1 : ui32, handshake.name = "unbundle1"} : <f32> to _ 
    %15 = mux %index [%4, %trueResult_81] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i32>, <i32>] to <i32>
    %result, %index = control_merge [%5, %trueResult_83]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %16 = constant %result {handshake.bb = 1 : ui32, handshake.name = "constant5", value = 0 : i32} : <>, <i32>
    %17 = buffer %15, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer0"} : <i32>
    %18 = init %17 {handshake.bb = 1 : ui32, handshake.name = "init14"} : <i32>
    %19 = init %18 {handshake.bb = 1 : ui32, handshake.name = "init15"} : <i32>
    %20 = buffer %14#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 1 : ui32, handshake.name = "buffer1"} : <>
    %21 = init %20 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 1 : ui32, handshake.name = "init16"} : <>
    %22 = init %21 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 1 : ui32, handshake.name = "init17"} : <>
    %23 = init %22 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 1 : ui32, handshake.name = "init18"} : <>
    %addressResult, %dataResult = load[%15] %outputs#0 {handshake.bb = 1 : ui32, handshake.deps = #handshake<deps[["store1", 2, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i32>, <f32>, <i32>, <f32>
    %24 = br %16 {handshake.bb = 1 : ui32, handshake.name = "br4"} : <i32>
    %25 = br %dataResult {handshake.bb = 1 : ui32, handshake.name = "br5"} : <f32>
    %26 = br %15 {handshake.bb = 1 : ui32, handshake.name = "br6"} : <i32>
    %27 = br %result {handshake.bb = 1 : ui32, handshake.name = "br7"} : <>
    %28 = mux %index_7 [%24, %trueResult] {handshake.bb = 2 : ui32, handshake.name = "mux1"} : <i1>, [<i32>, <i32>] to <i32>
    %29 = mux %index_7 [%25, %trueResult_12] {handshake.bb = 2 : ui32, handshake.name = "mux2"} : <i1>, [<f32>, <f32>] to <f32>
    %30 = mux %index_7 [%26, %trueResult_14] {handshake.bb = 2 : ui32, handshake.name = "mux3"} : <i1>, [<i32>, <i32>] to <i32>
    %result_6, %index_7 = control_merge [%27, %trueResult_16]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>, <>] to <>, <i1>
    %31 = constant %result_6 {handshake.bb = 2 : ui32, handshake.name = "constant6", value = 0 : i32} : <>, <i32>
    %32 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %33 = constant %32 {handshake.bb = 2 : ui32, handshake.name = "constant7", value = 20 : i32} : <>, <i32>
    %34 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %35 = constant %34 {handshake.bb = 2 : ui32, handshake.name = "constant8", value = 1 : i32} : <>, <i32>
    %36 = source {handshake.bb = 2 : ui32, handshake.name = "source2"} : <>
    %37 = constant %36 {handshake.bb = 2 : ui32, handshake.name = "constant9", value = 4 : i32} : <>, <i32>
    %38 = source {handshake.bb = 2 : ui32, handshake.name = "source3"} : <>
    %39 = constant %38 {handshake.bb = 2 : ui32, handshake.name = "constant10", value = 2 : i32} : <>, <i32>
    %40 = shli %30, %39 {handshake.bb = 2 : ui32, handshake.name = "shli0"} : <i32>
    %41 = shli %30, %37 {handshake.bb = 2 : ui32, handshake.name = "shli1"} : <i32>
    %42 = addi %40, %41 {handshake.bb = 2 : ui32, handshake.name = "addi5"} : <i32>
    %43 = addi %28, %42 {handshake.bb = 2 : ui32, handshake.name = "addi0"} : <i32>
    %addressResult_8, %dataResult_9 = load[%43] %outputs_4#0 {handshake.bb = 2 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i32>, <f32>, <i32>, <f32>
    %addressResult_10, %dataResult_11 = load[%28] %outputs_2 {handshake.bb = 2 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load2"} : <i32>, <f32>, <i32>, <f32>
    %44 = mulf %dataResult_9, %dataResult_11 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "mulf0"} : <f32>
    %45 = addf %29, %44 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "addf0"} : <f32>
    %46 = addi %28, %35 {handshake.bb = 2 : ui32, handshake.name = "addi2"} : <i32>
    %47 = cmpi ult, %46, %33 {handshake.bb = 2 : ui32, handshake.name = "cmpi0"} : <i32>
    %trueResult, %falseResult = cond_br %47, %46 {handshake.bb = 2 : ui32, handshake.name = "cond_br3"} : <i1>, <i32>
    %trueResult_12, %falseResult_13 = cond_br %47, %45 {handshake.bb = 2 : ui32, handshake.name = "cond_br4"} : <i1>, <f32>
    %trueResult_14, %falseResult_15 = cond_br %47, %30 {handshake.bb = 2 : ui32, handshake.name = "cond_br5"} : <i1>, <i32>
    %trueResult_16, %falseResult_17 = cond_br %47, %result_6 {handshake.bb = 2 : ui32, handshake.name = "cond_br6"} : <i1>, <>
    %trueResult_18, %falseResult_19 = cond_br %47, %31 {handshake.bb = 2 : ui32, handshake.name = "cond_br7"} : <i1>, <i32>
    %trueResult_20, %falseResult_21 = cond_br %94, %88 {handshake.bb = 3 : ui32, handshake.name = "cond_br45"} : <i1>, <i32>
    %trueResult_22, %falseResult_23 = cond_br %94, %90 {handshake.bb = 3 : ui32, handshake.name = "cond_br46"} : <i1>, <>
    %trueResult_24, %falseResult_25 = cond_br %94, %87 {handshake.bb = 3 : ui32, handshake.name = "cond_br47"} : <i1>, <i32>
    %trueResult_26, %falseResult_27 = cond_br %94, %86 {handshake.bb = 3 : ui32, handshake.name = "cond_br48"} : <i1>, <i32>
    %trueResult_28, %falseResult_29 = cond_br %94, %89 {handshake.bb = 3 : ui32, handshake.name = "cond_br49"} : <i1>, <>
    %trueResult_30, %falseResult_31 = cond_br %94, %91 {handshake.bb = 3 : ui32, handshake.name = "cond_br50"} : <i1>, <>
    %trueResult_32, %falseResult_33 = cond_br %94, %92 {handshake.bb = 3 : ui32, handshake.name = "cond_br51"} : <i1>, <>
    %48 = mux %55 [%6, %trueResult_20] {ftd.phi, handshake.bb = 3 : ui32, handshake.name = "mux21"} : <i1>, [<i32>, <i32>] to <i32>
    %49 = mux %55 [%7, %trueResult_32] {ftd.phi, handshake.bb = 3 : ui32, handshake.name = "mux22"} : <i1>, [<>, <>] to <>
    %50 = mux %55 [%8, %trueResult_24] {ftd.phi, handshake.bb = 3 : ui32, handshake.name = "mux23"} : <i1>, [<i32>, <i32>] to <i32>
    %51 = mux %55 [%9, %trueResult_26] {ftd.phi, handshake.bb = 3 : ui32, handshake.name = "mux24"} : <i1>, [<i32>, <i32>] to <i32>
    %52 = mux %55 [%10, %trueResult_22] {ftd.phi, handshake.bb = 3 : ui32, handshake.name = "mux25"} : <i1>, [<>, <>] to <>
    %53 = mux %55 [%11, %trueResult_28] {ftd.phi, handshake.bb = 3 : ui32, handshake.name = "mux26"} : <i1>, [<>, <>] to <>
    %54 = mux %55 [%12, %trueResult_30] {ftd.phi, handshake.bb = 3 : ui32, handshake.name = "mux27"} : <i1>, [<>, <>] to <>
    %55 = init %94 {ftd.imerge, handshake.bb = 3 : ui32, handshake.name = "init19"} : <i1>
    %56 = mux %index_35 [%falseResult_19, %trueResult_48] {handshake.bb = 3 : ui32, handshake.name = "mux4"} : <i1>, [<i32>, <i32>] to <i32>
    %57 = mux %index_35 [%falseResult_15, %trueResult_50] {handshake.bb = 3 : ui32, handshake.name = "mux5"} : <i1>, [<i32>, <i32>] to <i32>
    %58 = mux %index_35 [%falseResult_13, %trueResult_52] {handshake.bb = 3 : ui32, handshake.name = "mux6"} : <i1>, [<f32>, <f32>] to <f32>
    %result_34, %index_35 = control_merge [%falseResult_17, %trueResult_54]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>, <>] to <>, <i1>
    %59 = constant %result_34 {handshake.bb = 3 : ui32, handshake.name = "constant19", value = 1 : i32} : <>, <i32>
    %60 = source {handshake.bb = 3 : ui32, handshake.name = "source4"} : <>
    %61 = constant %60 {handshake.bb = 3 : ui32, handshake.name = "constant11", value = 20 : i32} : <>, <i32>
    %62 = source {handshake.bb = 3 : ui32, handshake.name = "source5"} : <>
    %63 = constant %62 {handshake.bb = 3 : ui32, handshake.name = "constant12", value = 1 : i32} : <>, <i32>
    %64 = source {handshake.bb = 3 : ui32, handshake.name = "source6"} : <>
    %65 = constant %64 {handshake.bb = 3 : ui32, handshake.name = "constant13", value = 4 : i32} : <>, <i32>
    %66 = source {handshake.bb = 3 : ui32, handshake.name = "source7"} : <>
    %67 = constant %66 {handshake.bb = 3 : ui32, handshake.name = "constant14", value = 2 : i32} : <>, <i32>
    %68 = gate %56, %49 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 3 : ui32, handshake.name = "gate0"} : <i32>, !handshake.control<> to <i32>
    %69 = cmpi ne, %68, %51 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 3 : ui32, handshake.name = "cmpi3"} : <i32>
    %70 = cmpi ne, %68, %50 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 3 : ui32, handshake.name = "cmpi4"} : <i32>
    %71 = cmpi ne, %68, %48 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 3 : ui32, handshake.name = "cmpi5"} : <i32>
    %trueResult_36, %falseResult_37 = cond_br %69, %53 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 3 : ui32, handshake.name = "cond_br25"} : <i1>, <>
    %trueResult_38, %falseResult_39 = cond_br %70, %52 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 3 : ui32, handshake.name = "cond_br26"} : <i1>, <>
    %trueResult_40, %falseResult_41 = cond_br %71, %54 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 3 : ui32, handshake.name = "cond_br27"} : <i1>, <>
    %72 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 3 : ui32, handshake.name = "source10"} : <>
    %73 = mux %69 [%falseResult_37, %72] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 3 : ui32, handshake.name = "mux28"} : <i1>, [<>, <>] to <>
    %74 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 3 : ui32, handshake.name = "source11"} : <>
    %75 = mux %70 [%falseResult_39, %74] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 3 : ui32, handshake.name = "mux29"} : <i1>, [<>, <>] to <>
    %76 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 3 : ui32, handshake.name = "source12"} : <>
    %77 = mux %71 [%falseResult_41, %76] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 3 : ui32, handshake.name = "mux30"} : <i1>, [<>, <>] to <>
    %78 = join %73, %75, %77 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 3 : ui32, handshake.name = "join0"} : <>
    %79 = gate %56, %78 {handshake.bb = 3 : ui32, handshake.name = "gate1"} : <i32>, !handshake.control<> to <i32>
    %addressResult_42, %dataResult_43 = load[%79] %outputs_0#0 {handshake.bb = 3 : ui32, handshake.deps = #handshake<deps[["store0", 1, false], ["store0", 3, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load3"} : <i32>, <f32>, <i32>, <f32>
    %80 = shli %57, %67 {handshake.bb = 3 : ui32, handshake.name = "shli2"} : <i32>
    %81 = shli %57, %65 {handshake.bb = 3 : ui32, handshake.name = "shli3"} : <i32>
    %82 = addi %80, %81 {handshake.bb = 3 : ui32, handshake.name = "addi6"} : <i32>
    %83 = addi %56, %82 {handshake.bb = 3 : ui32, handshake.name = "addi1"} : <i32>
    %addressResult_44, %dataResult_45 = load[%83] %outputs_4#1 {handshake.bb = 3 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load4"} : <i32>, <f32>, <i32>, <f32>
    %84 = mulf %dataResult_45, %58 {fastmath = #arith.fastmath<none>, handshake.bb = 3 : ui32, handshake.name = "mulf1"} : <f32>
    %85 = addf %dataResult_43, %84 {fastmath = #arith.fastmath<none>, handshake.bb = 3 : ui32, handshake.name = "addf1"} : <f32>
    %86 = buffer %56, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer2"} : <i32>
    %87 = init %86 {handshake.bb = 3 : ui32, handshake.name = "init26"} : <i32>
    %88 = init %87 {handshake.bb = 3 : ui32, handshake.name = "init27"} : <i32>
    %89 = buffer %doneResult, bufferType = FIFO_BREAK_NONE, numSlots = 1 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 3 : ui32, handshake.name = "buffer3"} : <>
    %90 = init %89 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 3 : ui32, handshake.name = "init28"} : <>
    %91 = init %90 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 3 : ui32, handshake.name = "init29"} : <>
    %92 = init %91 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 3 : ui32, handshake.name = "init30"} : <>
    %addressResult_46, %dataResult_47, %doneResult = store[%56] %85 %outputs_0#1 {handshake.bb = 3 : ui32, handshake.deps = #handshake<deps[["load3", 1, false], ["store0", 1, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store0"} : <i32>, <f32>, <>, <i32>, <f32>, <>
    %93 = addi %56, %63 {handshake.bb = 3 : ui32, handshake.name = "addi3"} : <i32>
    %94 = cmpi ult, %93, %61 {handshake.bb = 3 : ui32, handshake.name = "cmpi1"} : <i32>
    %trueResult_48, %falseResult_49 = cond_br %94, %93 {handshake.bb = 3 : ui32, handshake.name = "cond_br8"} : <i1>, <i32>
    %trueResult_50, %falseResult_51 = cond_br %94, %57 {handshake.bb = 3 : ui32, handshake.name = "cond_br9"} : <i1>, <i32>
    %trueResult_52, %falseResult_53 = cond_br %94, %58 {handshake.bb = 3 : ui32, handshake.name = "cond_br10"} : <i1>, <f32>
    %trueResult_54, %falseResult_55 = cond_br %94, %result_34 {handshake.bb = 3 : ui32, handshake.name = "cond_br11"} : <i1>, <>
    %trueResult_56, %falseResult_57 = cond_br %115, %falseResult_21 {handshake.bb = 4 : ui32, handshake.name = "cond_br52"} : <i1>, <i32>
    %trueResult_58, %falseResult_59 = cond_br %115, %falseResult_29 {handshake.bb = 4 : ui32, handshake.name = "cond_br53"} : <i1>, <>
    %trueResult_60, %falseResult_61 = cond_br %115, %falseResult_23 {handshake.bb = 4 : ui32, handshake.name = "cond_br54"} : <i1>, <>
    %trueResult_62, %falseResult_63 = cond_br %115, %falseResult_31 {handshake.bb = 4 : ui32, handshake.name = "cond_br55"} : <i1>, <>
    %trueResult_64, %falseResult_65 = cond_br %115, %falseResult_25 {handshake.bb = 4 : ui32, handshake.name = "cond_br56"} : <i1>, <i32>
    %trueResult_66, %falseResult_67 = cond_br %115, %falseResult_27 {handshake.bb = 4 : ui32, handshake.name = "cond_br57"} : <i1>, <i32>
    %trueResult_68, %falseResult_69 = cond_br %115, %falseResult_33 {handshake.bb = 4 : ui32, handshake.name = "cond_br58"} : <i1>, <>
    %95 = merge %falseResult_51 {handshake.bb = 4 : ui32, handshake.name = "merge0"} : <i32>
    %96 = merge %falseResult_53 {handshake.bb = 4 : ui32, handshake.name = "merge1"} : <f32>
    %result_70, %index_71 = control_merge [%falseResult_55]  {handshake.bb = 4 : ui32, handshake.name = "control_merge3"} : [<>] to <>, <i1>
    %97 = constant %result_70 {handshake.bb = 4 : ui32, handshake.name = "constant20", value = 1 : i32} : <>, <i32>
    %98 = source {handshake.bb = 4 : ui32, handshake.name = "source8"} : <>
    %99 = constant %98 {handshake.bb = 4 : ui32, handshake.name = "constant15", value = 20 : i32} : <>, <i32>
    %100 = source {handshake.bb = 4 : ui32, handshake.name = "source9"} : <>
    %101 = constant %100 {handshake.bb = 4 : ui32, handshake.name = "constant16", value = 1 : i32} : <>, <i32>
    %102 = gate %95, %23 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 4 : ui32, handshake.name = "gate2"} : <i32>, !handshake.control<> to <i32>
    %103 = cmpi ne, %102, %17 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 4 : ui32, handshake.name = "cmpi6"} : <i32>
    %104 = cmpi ne, %102, %18 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 4 : ui32, handshake.name = "cmpi7"} : <i32>
    %105 = cmpi ne, %102, %19 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 4 : ui32, handshake.name = "cmpi8"} : <i32>
    %trueResult_72, %falseResult_73 = cond_br %103, %20 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 4 : ui32, handshake.name = "cond_br42"} : <i1>, <>
    %trueResult_74, %falseResult_75 = cond_br %104, %21 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 4 : ui32, handshake.name = "cond_br43"} : <i1>, <>
    %trueResult_76, %falseResult_77 = cond_br %105, %22 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 4 : ui32, handshake.name = "cond_br44"} : <i1>, <>
    %106 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 4 : ui32, handshake.name = "source13"} : <>
    %107 = mux %103 [%falseResult_73, %106] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 4 : ui32, handshake.name = "mux31"} : <i1>, [<>, <>] to <>
    %108 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 4 : ui32, handshake.name = "source14"} : <>
    %109 = mux %104 [%falseResult_75, %108] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 4 : ui32, handshake.name = "mux32"} : <i1>, [<>, <>] to <>
    %110 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 4 : ui32, handshake.name = "source15"} : <>
    %111 = mux %105 [%falseResult_77, %110] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 4 : ui32, handshake.name = "mux33"} : <i1>, [<>, <>] to <>
    %112 = join %107, %109, %111 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 4 : ui32, handshake.name = "join1"} : <>
    %113 = gate %95, %112 {handshake.bb = 4 : ui32, handshake.name = "gate3"} : <i32>, !handshake.control<> to <i32>
    %addressResult_78, %dataResult_79, %doneResult_80 = store[%113] %96 %outputs#1 {handshake.bb = 4 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store1"} : <i32>, <f32>, <>, <i32>, <f32>, <>
    %114 = addi %95, %101 {handshake.bb = 4 : ui32, handshake.name = "addi4"} : <i32>
    %115 = cmpi ult, %114, %99 {handshake.bb = 4 : ui32, handshake.name = "cmpi2"} : <i32>
    %trueResult_81, %falseResult_82 = cond_br %115, %114 {handshake.bb = 4 : ui32, handshake.name = "cond_br12"} : <i1>, <i32>
    %trueResult_83, %falseResult_84 = cond_br %115, %result_70 {handshake.bb = 4 : ui32, handshake.name = "cond_br13"} : <i1>, <>
    %result_85, %index_86 = control_merge [%falseResult_84]  {handshake.bb = 5 : ui32, handshake.name = "control_merge4"} : [<>] to <>, <i1>
    end {handshake.bb = 5 : ui32, handshake.name = "end0"} %memEnd_5, %memEnd_3, %memEnd_1, %memEnd, %arg8 : <>, <>, <>, <>, <>
  }
}

