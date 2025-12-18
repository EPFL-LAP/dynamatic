module {
  handshake.func @mvt_float(%arg0: memref<900xf32>, %arg1: memref<30xf32>, %arg2: memref<30xf32>, %arg3: memref<30xf32>, %arg4: memref<30xf32>, %arg5: !handshake.control<>, %arg6: !handshake.control<>, %arg7: !handshake.control<>, %arg8: !handshake.control<>, %arg9: !handshake.control<>, %arg10: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["A", "x1", "x2", "y1", "y2", "A_start", "x1_start", "x2_start", "y1_start", "y2_start", "start"], cfg.edges = "[0,1][2,2,3,cmpi0][4,5][6,4,7,cmpi3][1,2][3,1,4,cmpi1][5,5,6,cmpi2]", resNames = ["A_end", "x1_end", "x2_end", "y1_end", "y2_end", "end"]} {
    %outputs, %memEnd = mem_controller[%arg4 : memref<30xf32>] %arg9 (%addressResult_40) %result_61 {connectedBlocks = [5 : i32], handshake.name = "mem_controller3"} :    (!handshake.channel<i32>) -> !handshake.channel<f32>
    %outputs_0, %memEnd_1 = mem_controller[%arg3 : memref<30xf32>] %arg8 (%addressResult_12) %result_61 {connectedBlocks = [2 : i32], handshake.name = "mem_controller4"} :    (!handshake.channel<i32>) -> !handshake.channel<f32>
    %outputs_2:2, %memEnd_3 = mem_controller[%arg2 : memref<30xf32>] %arg7 (%addressResult_34, %69, %addressResult_54, %dataResult_55) %result_61 {connectedBlocks = [4 : i32, 6 : i32], handshake.name = "mem_controller5"} :    (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<f32>) -> (!handshake.channel<f32>, !handshake.control<>)
    %outputs_4:2, %memEnd_5 = mem_controller[%arg1 : memref<30xf32>] %arg6 (%addressResult, %29, %addressResult_24, %dataResult_25) %result_61 {connectedBlocks = [1 : i32, 3 : i32], handshake.name = "mem_controller6"} :    (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<f32>) -> (!handshake.channel<f32>, !handshake.control<>)
    %outputs_6:2, %memEnd_7 = mem_controller[%arg0 : memref<900xf32>] %arg5 (%addressResult_10, %addressResult_38) %result_61 {connectedBlocks = [2 : i32, 5 : i32], handshake.name = "mem_controller7"} :    (!handshake.channel<i32>, !handshake.channel<i32>) -> (!handshake.channel<f32>, !handshake.channel<f32>)
    %0 = constant %arg10 {handshake.bb = 0 : ui32, handshake.name = "constant2", value = 0 : i32} : <>, <i32>
    %1 = br %0 {handshake.bb = 0 : ui32, handshake.name = "br2"} : <i32>
    %2 = br %arg10 {handshake.bb = 0 : ui32, handshake.name = "br4"} : <>
    %3 = init %42 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "init0"} : <i1>
    %4:2 = unbundle %dataResult  {handshake.bb = 1 : ui32, handshake.name = "unbundle1"} : <f32> to _ 
    %5 = mux %index [%1, %trueResult_26] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i32>, <i32>] to <i32>
    %result, %index = control_merge [%2, %trueResult_28]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %6 = constant %result {handshake.bb = 1 : ui32, handshake.name = "constant3", value = 0 : i32} : <>, <i32>
    %7 = buffer %5, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer0"} : <i32>
    %8 = buffer %4#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 1 : ui32, handshake.name = "buffer1"} : <>
    %9 = init %8 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 1 : ui32, handshake.name = "init3"} : <>
    %addressResult, %dataResult = load[%5] %outputs_4#0 {handshake.bb = 1 : ui32, handshake.deps = #handshake<deps[["store0", 2, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i32>, <f32>, <i32>, <f32>
    %10 = br %6 {handshake.bb = 1 : ui32, handshake.name = "br5"} : <i32>
    %11 = br %dataResult {handshake.bb = 1 : ui32, handshake.name = "br6"} : <f32>
    %12 = br %5 {handshake.bb = 1 : ui32, handshake.name = "br7"} : <i32>
    %13 = br %result {handshake.bb = 1 : ui32, handshake.name = "br8"} : <>
    %14 = mux %index_9 [%10, %trueResult] {handshake.bb = 2 : ui32, handshake.name = "mux1"} : <i1>, [<i32>, <i32>] to <i32>
    %15 = mux %index_9 [%11, %trueResult_14] {handshake.bb = 2 : ui32, handshake.name = "mux2"} : <i1>, [<f32>, <f32>] to <f32>
    %16 = mux %index_9 [%12, %trueResult_16] {handshake.bb = 2 : ui32, handshake.name = "mux3"} : <i1>, [<i32>, <i32>] to <i32>
    %result_8, %index_9 = control_merge [%13, %trueResult_18]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>, <>] to <>, <i1>
    %17 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %18 = constant %17 {handshake.bb = 2 : ui32, handshake.name = "constant4", value = 30 : i32} : <>, <i32>
    %19 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %20 = constant %19 {handshake.bb = 2 : ui32, handshake.name = "constant5", value = 1 : i32} : <>, <i32>
    %21 = muli %16, %18 {handshake.bb = 2 : ui32, handshake.name = "muli0"} : <i32>
    %22 = addi %14, %21 {handshake.bb = 2 : ui32, handshake.name = "addi0"} : <i32>
    %addressResult_10, %dataResult_11 = load[%22] %outputs_6#0 {handshake.bb = 2 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i32>, <f32>, <i32>, <f32>
    %addressResult_12, %dataResult_13 = load[%14] %outputs_0 {handshake.bb = 2 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load2"} : <i32>, <f32>, <i32>, <f32>
    %23 = mulf %dataResult_11, %dataResult_13 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "mulf0"} : <f32>
    %24 = addf %15, %23 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "addf0"} : <f32>
    %25 = addi %14, %20 {handshake.bb = 2 : ui32, handshake.name = "addi2"} : <i32>
    %26 = cmpi ult, %25, %18 {handshake.bb = 2 : ui32, handshake.name = "cmpi0"} : <i32>
    %trueResult, %falseResult = cond_br %26, %25 {handshake.bb = 2 : ui32, handshake.name = "cond_br4"} : <i1>, <i32>
    %trueResult_14, %falseResult_15 = cond_br %26, %24 {handshake.bb = 2 : ui32, handshake.name = "cond_br5"} : <i1>, <f32>
    %trueResult_16, %falseResult_17 = cond_br %26, %16 {handshake.bb = 2 : ui32, handshake.name = "cond_br6"} : <i1>, <i32>
    %trueResult_18, %falseResult_19 = cond_br %26, %result_8 {handshake.bb = 2 : ui32, handshake.name = "cond_br7"} : <i1>, <>
    %27 = merge %falseResult_17 {handshake.bb = 3 : ui32, handshake.name = "merge0"} : <i32>
    %28 = merge %falseResult_15 {handshake.bb = 3 : ui32, handshake.name = "merge1"} : <f32>
    %result_20, %index_21 = control_merge [%falseResult_19]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>] to <>, <i1>
    %29 = constant %result_20 {handshake.bb = 3 : ui32, handshake.name = "constant14", value = 1 : i32} : <>, <i32>
    %30 = constant %result_20 {handshake.bb = 3 : ui32, handshake.name = "constant6", value = 0 : i32} : <>, <i32>
    %31 = source {handshake.bb = 3 : ui32, handshake.name = "source2"} : <>
    %32 = constant %31 {handshake.bb = 3 : ui32, handshake.name = "constant7", value = 30 : i32} : <>, <i32>
    %33 = source {handshake.bb = 3 : ui32, handshake.name = "source3"} : <>
    %34 = constant %33 {handshake.bb = 3 : ui32, handshake.name = "constant8", value = 1 : i32} : <>, <i32>
    %35 = gate %27, %9 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 3 : ui32, handshake.name = "gate0"} : <i32>, !handshake.control<> to <i32>
    %36 = cmpi ne, %35, %7 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 3 : ui32, handshake.name = "cmpi4"} : <i32>
    %trueResult_22, %falseResult_23 = cond_br %36, %8 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 3 : ui32, handshake.name = "cond_br3"} : <i1>, <>
    %37 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 3 : ui32, handshake.name = "source8"} : <>
    %38 = mux %36 [%falseResult_23, %37] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 3 : ui32, handshake.name = "mux11"} : <i1>, [<>, <>] to <>
    %39 = join %38 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 3 : ui32, handshake.name = "join0"} : <>
    %40 = gate %27, %39 {handshake.bb = 3 : ui32, handshake.name = "gate1"} : <i32>, !handshake.control<> to <i32>
    %addressResult_24, %dataResult_25, %doneResult = store[%40] %28 %outputs_4#1 {handshake.bb = 3 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store0"} : <i32>, <f32>, <>, <i32>, <f32>, <>
    %41 = addi %27, %34 {handshake.bb = 3 : ui32, handshake.name = "addi3"} : <i32>
    %42 = cmpi ult, %41, %32 {handshake.bb = 3 : ui32, handshake.name = "cmpi1"} : <i32>
    %trueResult_26, %falseResult_27 = cond_br %42, %41 {handshake.bb = 3 : ui32, handshake.name = "cond_br8"} : <i1>, <i32>
    %trueResult_28, %falseResult_29 = cond_br %42, %result_20 {handshake.bb = 3 : ui32, handshake.name = "cond_br9"} : <i1>, <>
    %trueResult_30, %falseResult_31 = cond_br %42, %30 {handshake.bb = 3 : ui32, handshake.name = "cond_br10"} : <i1>, <i32>
    %43 = init %81 {ftd.imerge, handshake.bb = 4 : ui32, handshake.name = "init4"} : <i1>
    %44:2 = unbundle %dataResult_35  {handshake.bb = 4 : ui32, handshake.name = "unbundle3"} : <f32> to _ 
    %45 = mux %index_33 [%falseResult_31, %trueResult_57] {handshake.bb = 4 : ui32, handshake.name = "mux4"} : <i1>, [<i32>, <i32>] to <i32>
    %result_32, %index_33 = control_merge [%falseResult_29, %trueResult_59]  {handshake.bb = 4 : ui32, handshake.name = "control_merge3"} : [<>, <>] to <>, <i1>
    %46 = constant %result_32 {handshake.bb = 4 : ui32, handshake.name = "constant9", value = 0 : i32} : <>, <i32>
    %47 = buffer %45, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer2"} : <i32>
    %48 = buffer %44#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 4 : ui32, handshake.name = "buffer3"} : <>
    %49 = init %48 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 4 : ui32, handshake.name = "init7"} : <>
    %addressResult_34, %dataResult_35 = load[%45] %outputs_2#0 {handshake.bb = 4 : ui32, handshake.deps = #handshake<deps[["store1", 2, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load3"} : <i32>, <f32>, <i32>, <f32>
    %50 = br %46 {handshake.bb = 4 : ui32, handshake.name = "br9"} : <i32>
    %51 = br %dataResult_35 {handshake.bb = 4 : ui32, handshake.name = "br10"} : <f32>
    %52 = br %45 {handshake.bb = 4 : ui32, handshake.name = "br11"} : <i32>
    %53 = br %result_32 {handshake.bb = 4 : ui32, handshake.name = "br12"} : <>
    %54 = mux %index_37 [%50, %trueResult_42] {handshake.bb = 5 : ui32, handshake.name = "mux5"} : <i1>, [<i32>, <i32>] to <i32>
    %55 = mux %index_37 [%51, %trueResult_44] {handshake.bb = 5 : ui32, handshake.name = "mux6"} : <i1>, [<f32>, <f32>] to <f32>
    %56 = mux %index_37 [%52, %trueResult_46] {handshake.bb = 5 : ui32, handshake.name = "mux7"} : <i1>, [<i32>, <i32>] to <i32>
    %result_36, %index_37 = control_merge [%53, %trueResult_48]  {handshake.bb = 5 : ui32, handshake.name = "control_merge4"} : [<>, <>] to <>, <i1>
    %57 = source {handshake.bb = 5 : ui32, handshake.name = "source4"} : <>
    %58 = constant %57 {handshake.bb = 5 : ui32, handshake.name = "constant10", value = 30 : i32} : <>, <i32>
    %59 = source {handshake.bb = 5 : ui32, handshake.name = "source5"} : <>
    %60 = constant %59 {handshake.bb = 5 : ui32, handshake.name = "constant11", value = 1 : i32} : <>, <i32>
    %61 = muli %54, %58 {handshake.bb = 5 : ui32, handshake.name = "muli1"} : <i32>
    %62 = addi %56, %61 {handshake.bb = 5 : ui32, handshake.name = "addi1"} : <i32>
    %addressResult_38, %dataResult_39 = load[%62] %outputs_6#1 {handshake.bb = 5 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load4"} : <i32>, <f32>, <i32>, <f32>
    %addressResult_40, %dataResult_41 = load[%54] %outputs {handshake.bb = 5 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load5"} : <i32>, <f32>, <i32>, <f32>
    %63 = mulf %dataResult_39, %dataResult_41 {fastmath = #arith.fastmath<none>, handshake.bb = 5 : ui32, handshake.name = "mulf1"} : <f32>
    %64 = addf %55, %63 {fastmath = #arith.fastmath<none>, handshake.bb = 5 : ui32, handshake.name = "addf1"} : <f32>
    %65 = addi %54, %60 {handshake.bb = 5 : ui32, handshake.name = "addi4"} : <i32>
    %66 = cmpi ult, %65, %58 {handshake.bb = 5 : ui32, handshake.name = "cmpi2"} : <i32>
    %trueResult_42, %falseResult_43 = cond_br %66, %65 {handshake.bb = 5 : ui32, handshake.name = "cond_br11"} : <i1>, <i32>
    %trueResult_44, %falseResult_45 = cond_br %66, %64 {handshake.bb = 5 : ui32, handshake.name = "cond_br12"} : <i1>, <f32>
    %trueResult_46, %falseResult_47 = cond_br %66, %56 {handshake.bb = 5 : ui32, handshake.name = "cond_br13"} : <i1>, <i32>
    %trueResult_48, %falseResult_49 = cond_br %66, %result_36 {handshake.bb = 5 : ui32, handshake.name = "cond_br14"} : <i1>, <>
    %67 = merge %falseResult_47 {handshake.bb = 6 : ui32, handshake.name = "merge2"} : <i32>
    %68 = merge %falseResult_45 {handshake.bb = 6 : ui32, handshake.name = "merge3"} : <f32>
    %result_50, %index_51 = control_merge [%falseResult_49]  {handshake.bb = 6 : ui32, handshake.name = "control_merge5"} : [<>] to <>, <i1>
    %69 = constant %result_50 {handshake.bb = 6 : ui32, handshake.name = "constant15", value = 1 : i32} : <>, <i32>
    %70 = source {handshake.bb = 6 : ui32, handshake.name = "source6"} : <>
    %71 = constant %70 {handshake.bb = 6 : ui32, handshake.name = "constant12", value = 30 : i32} : <>, <i32>
    %72 = source {handshake.bb = 6 : ui32, handshake.name = "source7"} : <>
    %73 = constant %72 {handshake.bb = 6 : ui32, handshake.name = "constant13", value = 1 : i32} : <>, <i32>
    %74 = gate %67, %49 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 6 : ui32, handshake.name = "gate2"} : <i32>, !handshake.control<> to <i32>
    %75 = cmpi ne, %74, %47 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 6 : ui32, handshake.name = "cmpi5"} : <i32>
    %trueResult_52, %falseResult_53 = cond_br %75, %48 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 6 : ui32, handshake.name = "cond_br20"} : <i1>, <>
    %76 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 6 : ui32, handshake.name = "source9"} : <>
    %77 = mux %75 [%falseResult_53, %76] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 6 : ui32, handshake.name = "mux15"} : <i1>, [<>, <>] to <>
    %78 = join %77 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 6 : ui32, handshake.name = "join1"} : <>
    %79 = gate %67, %78 {handshake.bb = 6 : ui32, handshake.name = "gate3"} : <i32>, !handshake.control<> to <i32>
    %addressResult_54, %dataResult_55, %doneResult_56 = store[%79] %68 %outputs_2#1 {handshake.bb = 6 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store1"} : <i32>, <f32>, <>, <i32>, <f32>, <>
    %80 = addi %67, %73 {handshake.bb = 6 : ui32, handshake.name = "addi5"} : <i32>
    %81 = cmpi ult, %80, %71 {handshake.bb = 6 : ui32, handshake.name = "cmpi3"} : <i32>
    %trueResult_57, %falseResult_58 = cond_br %81, %80 {handshake.bb = 6 : ui32, handshake.name = "cond_br15"} : <i1>, <i32>
    %trueResult_59, %falseResult_60 = cond_br %81, %result_50 {handshake.bb = 6 : ui32, handshake.name = "cond_br16"} : <i1>, <>
    %result_61, %index_62 = control_merge [%falseResult_60]  {handshake.bb = 7 : ui32, handshake.name = "control_merge6"} : [<>] to <>, <i1>
    end {handshake.bb = 7 : ui32, handshake.name = "end0"} %memEnd_7, %memEnd_5, %memEnd_3, %memEnd_1, %memEnd, %arg10 : <>, <>, <>, <>, <>, <>
  }
}

