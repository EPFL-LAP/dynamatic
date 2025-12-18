module {
  handshake.func @triangular(%arg0: memref<10xi32>, %arg1: !handshake.channel<i32>, %arg2: memref<100xi32>, %arg3: !handshake.control<>, %arg4: !handshake.control<>, %arg5: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["x", "n", "a", "x_start", "a_start", "start"], cfg.edges = "[0,1][2,3][4,3][1,2,6,cmpi0][3,4,5,cmpi1][5,1]", resNames = ["x_end", "a_end", "end"]} {
    %outputs:3, %memEnd = mem_controller[%arg2 : memref<100xi32>] %arg4 (%50, %addressResult, %addressResult_60, %addressResult_62, %dataResult_63) %result_66 {connectedBlocks = [4 : i32], handshake.name = "mem_controller2"} :    (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>) -> (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>)
    %outputs_0, %memEnd_1 = mem_controller[%arg0 : memref<10xi32>] %arg3 (%addressResult_52) %result_66 {connectedBlocks = [4 : i32], handshake.name = "mem_controller3"} :    (!handshake.channel<i32>) -> !handshake.channel<i32>
    %0 = constant %arg5 {handshake.bb = 0 : ui32, handshake.name = "constant2", value = 1000 : i32} : <>, <i32>
    %1 = constant %arg5 {handshake.bb = 0 : ui32, handshake.name = "constant3", value = 1000 : i32} : <>, <i32>
    %2 = constant %arg5 {handshake.bb = 0 : ui32, handshake.name = "constant4", value = 1000 : i32} : <>, <i32>
    %3 = constant %arg5 {handshake.bb = 0 : ui32, handshake.name = "constant1", value = 0 : i32} : <>, <i32>
    %4 = br %3 {handshake.bb = 0 : ui32, handshake.name = "br4"} : <i32>
    %5 = br %arg1 {handshake.bb = 0 : ui32, handshake.name = "br5"} : <i32>
    %6 = br %arg5 {handshake.bb = 0 : ui32, handshake.name = "br6"} : <>
    %7 = mux %14 [%2, %falseResult_49] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux7"} : <i1>, [<i32>, <i32>] to <i32>
    %8 = mux %14 [%1, %falseResult_45] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux8"} : <i1>, [<i32>, <i32>] to <i32>
    %9 = mux %14 [%0, %falseResult_41] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux9"} : <i1>, [<i32>, <i32>] to <i32>
    %10 = mux %14 [%arg5, %falseResult_39] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux10"} : <i1>, [<>, <>] to <>
    %11 = mux %14 [%arg5, %falseResult_47] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux11"} : <i1>, [<>, <>] to <>
    %12 = mux %14 [%arg5, %falseResult_43] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux12"} : <i1>, [<>, <>] to <>
    %13 = mux %14 [%arg5, %falseResult_37] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux13"} : <i1>, [<>, <>] to <>
    %14 = init %17 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "init0"} : <i1>
    %15 = mux %index [%4, %121] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i32>, <i32>] to <i32>
    %16 = mux %index [%5, %122] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i1>, [<i32>, <i32>] to <i32>
    %result, %index = control_merge [%6, %123]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %17 = cmpi slt, %15, %16 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i32>
    %trueResult, %falseResult = cond_br %17, %16 {handshake.bb = 1 : ui32, handshake.name = "cond_br2"} : <i1>, <i32>
    %trueResult_2, %falseResult_3 = cond_br %17, %15 {handshake.bb = 1 : ui32, handshake.name = "cond_br3"} : <i1>, <i32>
    %trueResult_4, %falseResult_5 = cond_br %17, %result {handshake.bb = 1 : ui32, handshake.name = "cond_br4"} : <i1>, <>
    %18 = merge %trueResult {handshake.bb = 2 : ui32, handshake.name = "merge0"} : <i32>
    %19 = merge %trueResult_2 {handshake.bb = 2 : ui32, handshake.name = "merge1"} : <i32>
    %result_6, %index_7 = control_merge [%trueResult_4]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>] to <>, <i1>
    %20 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %21 = constant %20 {handshake.bb = 2 : ui32, handshake.name = "constant5", value = -1 : i32} : <>, <i32>
    %22 = constant %result_6 {handshake.bb = 2 : ui32, handshake.name = "constant6", value = 0 : i32} : <>, <i32>
    %23 = subi %18, %19 {handshake.bb = 2 : ui32, handshake.name = "subi1"} : <i32>
    %24 = addi %23, %21 {handshake.bb = 2 : ui32, handshake.name = "addi1"} : <i32>
    %25 = br %22 {handshake.bb = 2 : ui32, handshake.name = "br7"} : <i32>
    %26 = br %18 {handshake.bb = 2 : ui32, handshake.name = "br8"} : <i32>
    %27 = br %19 {handshake.bb = 2 : ui32, handshake.name = "br9"} : <i32>
    %28 = br %23 {handshake.bb = 2 : ui32, handshake.name = "br10"} : <i32>
    %29 = br %24 {handshake.bb = 2 : ui32, handshake.name = "br11"} : <i32>
    %30 = br %result_6 {handshake.bb = 2 : ui32, handshake.name = "br12"} : <>
    %trueResult_8, %falseResult_9 = cond_br %17, %7 {handshake.bb = 3 : ui32, handshake.name = "cond_br33"} : <i1>, <i32>
    %trueResult_10, %falseResult_11 = cond_br %17, %10 {handshake.bb = 3 : ui32, handshake.name = "cond_br34"} : <i1>, <>
    %trueResult_12, %falseResult_13 = cond_br %17, %13 {handshake.bb = 3 : ui32, handshake.name = "cond_br35"} : <i1>, <>
    %trueResult_14, %falseResult_15 = cond_br %17, %12 {handshake.bb = 3 : ui32, handshake.name = "cond_br36"} : <i1>, <>
    %trueResult_16, %falseResult_17 = cond_br %17, %8 {handshake.bb = 3 : ui32, handshake.name = "cond_br37"} : <i1>, <i32>
    %trueResult_18, %falseResult_19 = cond_br %17, %9 {handshake.bb = 3 : ui32, handshake.name = "cond_br38"} : <i1>, <i32>
    %trueResult_20, %falseResult_21 = cond_br %17, %11 {handshake.bb = 3 : ui32, handshake.name = "cond_br39"} : <i1>, <>
    %31 = mux %38 [%trueResult_8, %104] {ftd.phi, handshake.bb = 3 : ui32, handshake.name = "mux14"} : <i1>, [<i32>, <i32>] to <i32>
    %32 = mux %38 [%trueResult_16, %103] {ftd.phi, handshake.bb = 3 : ui32, handshake.name = "mux15"} : <i1>, [<i32>, <i32>] to <i32>
    %33 = mux %38 [%trueResult_18, %102] {ftd.phi, handshake.bb = 3 : ui32, handshake.name = "mux16"} : <i1>, [<i32>, <i32>] to <i32>
    %34 = mux %38 [%trueResult_10, %108] {ftd.phi, handshake.bb = 3 : ui32, handshake.name = "mux17"} : <i1>, [<>, <>] to <>
    %35 = mux %38 [%trueResult_20, %105] {ftd.phi, handshake.bb = 3 : ui32, handshake.name = "mux18"} : <i1>, [<>, <>] to <>
    %36 = mux %38 [%trueResult_14, %106] {ftd.phi, handshake.bb = 3 : ui32, handshake.name = "mux19"} : <i1>, [<>, <>] to <>
    %37 = mux %38 [%trueResult_12, %107] {ftd.phi, handshake.bb = 3 : ui32, handshake.name = "mux20"} : <i1>, [<>, <>] to <>
    %38 = init %44 {ftd.imerge, handshake.bb = 3 : ui32, handshake.name = "init7"} : <i1>
    %39 = mux %index_23 [%25, %110] {handshake.bb = 3 : ui32, handshake.name = "mux2"} : <i1>, [<i32>, <i32>] to <i32>
    %40 = mux %index_23 [%26, %111] {handshake.bb = 3 : ui32, handshake.name = "mux3"} : <i1>, [<i32>, <i32>] to <i32>
    %41 = mux %index_23 [%27, %112] {handshake.bb = 3 : ui32, handshake.name = "mux4"} : <i1>, [<i32>, <i32>] to <i32>
    %42 = mux %index_23 [%28, %113] {handshake.bb = 3 : ui32, handshake.name = "mux5"} : <i1>, [<i32>, <i32>] to <i32>
    %43 = mux %index_23 [%29, %114] {handshake.bb = 3 : ui32, handshake.name = "mux6"} : <i1>, [<i32>, <i32>] to <i32>
    %result_22, %index_23 = control_merge [%30, %115]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>, <>] to <>, <i1>
    %44 = cmpi slt, %39, %43 {handshake.bb = 3 : ui32, handshake.name = "cmpi1"} : <i32>
    %trueResult_24, %falseResult_25 = cond_br %44, %40 {handshake.bb = 3 : ui32, handshake.name = "cond_br5"} : <i1>, <i32>
    %trueResult_26, %falseResult_27 = cond_br %44, %41 {handshake.bb = 3 : ui32, handshake.name = "cond_br6"} : <i1>, <i32>
    %trueResult_28, %falseResult_29 = cond_br %44, %42 {handshake.bb = 3 : ui32, handshake.name = "cond_br7"} : <i1>, <i32>
    %trueResult_30, %falseResult_31 = cond_br %44, %43 {handshake.bb = 3 : ui32, handshake.name = "cond_br8"} : <i1>, <i32>
    %trueResult_32, %falseResult_33 = cond_br %44, %39 {handshake.bb = 3 : ui32, handshake.name = "cond_br9"} : <i1>, <i32>
    %trueResult_34, %falseResult_35 = cond_br %44, %result_22 {handshake.bb = 3 : ui32, handshake.name = "cond_br10"} : <i1>, <>
    %trueResult_36, %falseResult_37 = cond_br %44, %37 {handshake.bb = 4 : ui32, handshake.name = "cond_br40"} : <i1>, <>
    %trueResult_38, %falseResult_39 = cond_br %44, %34 {handshake.bb = 4 : ui32, handshake.name = "cond_br41"} : <i1>, <>
    %trueResult_40, %falseResult_41 = cond_br %44, %33 {handshake.bb = 4 : ui32, handshake.name = "cond_br42"} : <i1>, <i32>
    %trueResult_42, %falseResult_43 = cond_br %44, %36 {handshake.bb = 4 : ui32, handshake.name = "cond_br43"} : <i1>, <>
    %trueResult_44, %falseResult_45 = cond_br %44, %32 {handshake.bb = 4 : ui32, handshake.name = "cond_br44"} : <i1>, <i32>
    %trueResult_46, %falseResult_47 = cond_br %44, %35 {handshake.bb = 4 : ui32, handshake.name = "cond_br45"} : <i1>, <>
    %trueResult_48, %falseResult_49 = cond_br %44, %31 {handshake.bb = 4 : ui32, handshake.name = "cond_br46"} : <i1>, <i32>
    %45 = merge %trueResult_24 {handshake.bb = 4 : ui32, handshake.name = "merge2"} : <i32>
    %46 = merge %trueResult_26 {handshake.bb = 4 : ui32, handshake.name = "merge3"} : <i32>
    %47 = merge %trueResult_28 {handshake.bb = 4 : ui32, handshake.name = "merge4"} : <i32>
    %48 = merge %trueResult_30 {handshake.bb = 4 : ui32, handshake.name = "merge5"} : <i32>
    %49 = merge %trueResult_32 {handshake.bb = 4 : ui32, handshake.name = "merge6"} : <i32>
    %result_50, %index_51 = control_merge [%trueResult_34]  {handshake.bb = 4 : ui32, handshake.name = "control_merge3"} : [<>] to <>, <i1>
    %50 = constant %result_50 {handshake.bb = 4 : ui32, handshake.name = "constant12", value = 1 : i32} : <>, <i32>
    %51 = source {handshake.bb = 4 : ui32, handshake.name = "source1"} : <>
    %52 = constant %51 {handshake.bb = 4 : ui32, handshake.name = "constant7", value = -2 : i32} : <>, <i32>
    %53 = source {handshake.bb = 4 : ui32, handshake.name = "source2"} : <>
    %54 = constant %53 {handshake.bb = 4 : ui32, handshake.name = "constant8", value = -1 : i32} : <>, <i32>
    %55 = source {handshake.bb = 4 : ui32, handshake.name = "source3"} : <>
    %56 = constant %55 {handshake.bb = 4 : ui32, handshake.name = "constant9", value = 1 : i32} : <>, <i32>
    %57 = source {handshake.bb = 4 : ui32, handshake.name = "source4"} : <>
    %58 = constant %57 {handshake.bb = 4 : ui32, handshake.name = "constant10", value = 3 : i32} : <>, <i32>
    %59 = addi %46, %49 {handshake.bb = 4 : ui32, handshake.name = "addi0"} : <i32>
    %60 = xori %59, %54 {handshake.bb = 4 : ui32, handshake.name = "xori0"} : <i32>
    %61 = addi %60, %56 {handshake.bb = 4 : ui32, handshake.name = "addi2"} : <i32>
    %62 = addi %61, %45 {handshake.bb = 4 : ui32, handshake.name = "addi3"} : <i32>
    %63 = addi %62, %52 {handshake.bb = 4 : ui32, handshake.name = "addi4"} : <i32>
    %64 = addi %47, %54 {handshake.bb = 4 : ui32, handshake.name = "addi6"} : <i32>
    %65 = shli %63, %56 {handshake.bb = 4 : ui32, handshake.name = "shli0"} : <i32>
    %66 = shli %63, %58 {handshake.bb = 4 : ui32, handshake.name = "shli1"} : <i32>
    %67 = addi %65, %66 {handshake.bb = 4 : ui32, handshake.name = "addi5"} : <i32>
    %68 = addi %64, %67 {handshake.bb = 4 : ui32, handshake.name = "addi15"} : <i32>
    %addressResult, %dataResult = load[%68] %outputs#0 {handshake.bb = 4 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i32>, <i32>, <i32>, <i32>
    %69 = addi %47, %54 {handshake.bb = 4 : ui32, handshake.name = "addi8"} : <i32>
    %addressResult_52, %dataResult_53 = load[%69] %outputs_0 {handshake.bb = 4 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i32>, <i32>, <i32>, <i32>
    %70 = muli %dataResult, %dataResult_53 {handshake.bb = 4 : ui32, handshake.name = "muli0"} : <i32>
    %71 = addi %46, %49 {handshake.bb = 4 : ui32, handshake.name = "addi7"} : <i32>
    %72 = xori %71, %54 {handshake.bb = 4 : ui32, handshake.name = "xori1"} : <i32>
    %73 = addi %72, %56 {handshake.bb = 4 : ui32, handshake.name = "addi9"} : <i32>
    %74 = addi %73, %45 {handshake.bb = 4 : ui32, handshake.name = "addi10"} : <i32>
    %75 = addi %74, %52 {handshake.bb = 4 : ui32, handshake.name = "addi11"} : <i32>
    %76 = shli %75, %56 {handshake.bb = 4 : ui32, handshake.name = "shli2"} : <i32>
    %77 = shli %75, %58 {handshake.bb = 4 : ui32, handshake.name = "shli3"} : <i32>
    %78 = addi %76, %77 {handshake.bb = 4 : ui32, handshake.name = "addi12"} : <i32>
    %79 = addi %45, %78 {handshake.bb = 4 : ui32, handshake.name = "addi16"} : <i32>
    %80 = gate %79, %trueResult_38 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 4 : ui32, handshake.name = "gate0"} : <i32>, !handshake.control<> to <i32>
    %81 = cmpi ne, %80, %trueResult_40 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 4 : ui32, handshake.name = "cmpi2"} : <i32>
    %82 = cmpi ne, %80, %trueResult_44 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 4 : ui32, handshake.name = "cmpi3"} : <i32>
    %83 = cmpi ne, %80, %trueResult_48 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 4 : ui32, handshake.name = "cmpi4"} : <i32>
    %trueResult_54, %falseResult_55 = cond_br %81, %trueResult_46 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 4 : ui32, handshake.name = "cond_br30"} : <i1>, <>
    %trueResult_56, %falseResult_57 = cond_br %82, %trueResult_42 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 4 : ui32, handshake.name = "cond_br31"} : <i1>, <>
    %trueResult_58, %falseResult_59 = cond_br %83, %trueResult_36 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 4 : ui32, handshake.name = "cond_br32"} : <i1>, <>
    %84 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 4 : ui32, handshake.name = "source6"} : <>
    %85 = mux %81 [%falseResult_55, %84] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 4 : ui32, handshake.name = "mux21"} : <i1>, [<>, <>] to <>
    %86 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 4 : ui32, handshake.name = "source7"} : <>
    %87 = mux %82 [%falseResult_57, %86] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 4 : ui32, handshake.name = "mux22"} : <i1>, [<>, <>] to <>
    %88 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 4 : ui32, handshake.name = "source8"} : <>
    %89 = mux %83 [%falseResult_59, %88] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 4 : ui32, handshake.name = "mux23"} : <i1>, [<>, <>] to <>
    %90 = join %85, %87, %89 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 4 : ui32, handshake.name = "join0"} : <>
    %91 = gate %79, %90 {handshake.bb = 4 : ui32, handshake.name = "gate1"} : <i32>, !handshake.control<> to <i32>
    %addressResult_60, %dataResult_61 = load[%91] %outputs#1 {handshake.bb = 4 : ui32, handshake.deps = #handshake<deps[["store0", 1, false], ["store0", 3, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load2"} : <i32>, <i32>, <i32>, <i32>
    %92 = subi %dataResult_61, %70 {handshake.bb = 4 : ui32, handshake.name = "subi0"} : <i32>
    %93 = addi %46, %49 {handshake.bb = 4 : ui32, handshake.name = "addi20"} : <i32>
    %94 = xori %93, %54 {handshake.bb = 4 : ui32, handshake.name = "xori2"} : <i32>
    %95 = addi %94, %56 {handshake.bb = 4 : ui32, handshake.name = "addi21"} : <i32>
    %96 = addi %95, %45 {handshake.bb = 4 : ui32, handshake.name = "addi13"} : <i32>
    %97 = addi %96, %52 {handshake.bb = 4 : ui32, handshake.name = "addi14"} : <i32>
    %98 = shli %97, %56 {handshake.bb = 4 : ui32, handshake.name = "shli4"} : <i32>
    %99 = shli %97, %58 {handshake.bb = 4 : ui32, handshake.name = "shli5"} : <i32>
    %100 = addi %98, %99 {handshake.bb = 4 : ui32, handshake.name = "addi22"} : <i32>
    %101 = addi %45, %100 {handshake.bb = 4 : ui32, handshake.name = "addi17"} : <i32>
    %102 = buffer %101, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer0"} : <i32>
    %103 = init %102 {handshake.bb = 4 : ui32, handshake.name = "init14"} : <i32>
    %104 = init %103 {handshake.bb = 4 : ui32, handshake.name = "init15"} : <i32>
    %105 = buffer %doneResult, bufferType = FIFO_BREAK_NONE, numSlots = 1 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 4 : ui32, handshake.name = "buffer1"} : <>
    %106 = init %105 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 4 : ui32, handshake.name = "init16"} : <>
    %107 = init %106 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 4 : ui32, handshake.name = "init17"} : <>
    %108 = init %107 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 4 : ui32, handshake.name = "init18"} : <>
    %addressResult_62, %dataResult_63, %doneResult = store[%101] %92 %outputs#2 {handshake.bb = 4 : ui32, handshake.deps = #handshake<deps[["load2", 1, false], ["store0", 1, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store0"} : <i32>, <i32>, <>, <i32>, <i32>, <>
    %109 = addi %49, %56 {handshake.bb = 4 : ui32, handshake.name = "addi18"} : <i32>
    %110 = br %109 {handshake.bb = 4 : ui32, handshake.name = "br13"} : <i32>
    %111 = br %45 {handshake.bb = 4 : ui32, handshake.name = "br14"} : <i32>
    %112 = br %46 {handshake.bb = 4 : ui32, handshake.name = "br15"} : <i32>
    %113 = br %47 {handshake.bb = 4 : ui32, handshake.name = "br16"} : <i32>
    %114 = br %48 {handshake.bb = 4 : ui32, handshake.name = "br17"} : <i32>
    %115 = br %result_50 {handshake.bb = 4 : ui32, handshake.name = "br18"} : <>
    %116 = merge %falseResult_25 {handshake.bb = 5 : ui32, handshake.name = "merge7"} : <i32>
    %117 = merge %falseResult_27 {handshake.bb = 5 : ui32, handshake.name = "merge8"} : <i32>
    %result_64, %index_65 = control_merge [%falseResult_35]  {handshake.bb = 5 : ui32, handshake.name = "control_merge4"} : [<>] to <>, <i1>
    %118 = source {handshake.bb = 5 : ui32, handshake.name = "source5"} : <>
    %119 = constant %118 {handshake.bb = 5 : ui32, handshake.name = "constant11", value = 1 : i32} : <>, <i32>
    %120 = addi %117, %119 {handshake.bb = 5 : ui32, handshake.name = "addi19"} : <i32>
    %121 = br %120 {handshake.bb = 5 : ui32, handshake.name = "br19"} : <i32>
    %122 = br %116 {handshake.bb = 5 : ui32, handshake.name = "br20"} : <i32>
    %123 = br %result_64 {handshake.bb = 5 : ui32, handshake.name = "br21"} : <>
    %result_66, %index_67 = control_merge [%falseResult_5]  {handshake.bb = 6 : ui32, handshake.name = "control_merge5"} : [<>] to <>, <i1>
    end {handshake.bb = 6 : ui32, handshake.name = "end0"} %memEnd_1, %memEnd, %arg5 : <>, <>, <>
  }
}

