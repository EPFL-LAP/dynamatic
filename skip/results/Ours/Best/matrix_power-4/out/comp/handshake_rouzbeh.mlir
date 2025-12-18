module {
  handshake.func @matrix_power(%arg0: memref<400xi32>, %arg1: memref<20xi32>, %arg2: memref<20xi32>, %arg3: memref<20xi32>, %arg4: !handshake.control<>, %arg5: !handshake.control<>, %arg6: !handshake.control<>, %arg7: !handshake.control<>, %arg8: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["mat", "row", "col", "a", "mat_start", "row_start", "col_start", "a_start", "start"], cfg.edges = "[0,1][2,2,3,cmpi0][1,2][3,1,4,cmpi1]", resNames = ["mat_end", "row_end", "col_end", "a_end", "end"]} {
    %outputs, %memEnd = mem_controller[%arg3 : memref<20xi32>] %arg7 (%addressResult_24) %result_82 {connectedBlocks = [2 : i32], handshake.name = "mem_controller3"} :    (!handshake.channel<i32>) -> !handshake.channel<i32>
    %outputs_0, %memEnd_1 = mem_controller[%arg2 : memref<20xi32>] %arg6 (%addressResult_26) %result_82 {connectedBlocks = [2 : i32], handshake.name = "mem_controller4"} :    (!handshake.channel<i32>) -> !handshake.channel<i32>
    %outputs_2, %memEnd_3 = mem_controller[%arg1 : memref<20xi32>] %arg5 (%addressResult) %result_82 {connectedBlocks = [2 : i32], handshake.name = "mem_controller5"} :    (!handshake.channel<i32>) -> !handshake.channel<i32>
    %outputs_4:3, %memEnd_5 = mem_controller[%arg0 : memref<400xi32>] %arg4 (%61, %addressResult_36, %addressResult_46, %addressResult_48, %dataResult_49) %result_82 {connectedBlocks = [2 : i32], handshake.name = "mem_controller6"} :    (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>) -> (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>)
    %0 = constant %arg8 {handshake.bb = 0 : ui32, handshake.name = "constant0", value = 1000 : i32} : <>, <i32>
    %1 = constant %arg8 {handshake.bb = 0 : ui32, handshake.name = "constant1", value = 1000 : i32} : <>, <i32>
    %2 = constant %arg8 {handshake.bb = 0 : ui32, handshake.name = "constant2", value = 1000 : i32} : <>, <i32>
    %3 = constant %arg8 {handshake.bb = 0 : ui32, handshake.name = "constant4", value = 1000 : i32} : <>, <i32>
    %4 = constant %arg8 {handshake.bb = 0 : ui32, handshake.name = "constant5", value = 1000 : i32} : <>, <i32>
    %5 = constant %arg8 {handshake.bb = 0 : ui32, handshake.name = "constant14", value = 1000 : i32} : <>, <i32>
    %6 = constant %arg8 {handshake.bb = 0 : ui32, handshake.name = "constant15", value = 1000 : i32} : <>, <i32>
    %7 = constant %arg8 {handshake.bb = 0 : ui32, handshake.name = "constant16", value = 1000 : i32} : <>, <i32>
    %8 = constant %arg8 {handshake.bb = 0 : ui32, handshake.name = "constant3", value = 1 : i32} : <>, <i32>
    %9 = br %8 {handshake.bb = 0 : ui32, handshake.name = "br2"} : <i32>
    %10 = br %arg8 {handshake.bb = 0 : ui32, handshake.name = "br3"} : <>
    %11 = mux %29 [%7, %trueResult_70] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux4"} : <i1>, [<i32>, <i32>] to <i32>
    %12 = mux %29 [%arg8, %trueResult_58] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux5"} : <i1>, [<>, <>] to <>
    %13 = mux %29 [%arg8, %trueResult_58] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux6"} : <i1>, [<>, <>] to <>
    %14 = mux %29 [%6, %trueResult_66] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux7"} : <i1>, [<i32>, <i32>] to <i32>
    %15 = mux %29 [%5, %trueResult_74] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux8"} : <i1>, [<i32>, <i32>] to <i32>
    %16 = mux %29 [%4, %trueResult_74] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux9"} : <i1>, [<i32>, <i32>] to <i32>
    %17 = mux %29 [%3, %trueResult_66] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux10"} : <i1>, [<i32>, <i32>] to <i32>
    %18 = mux %29 [%2, %trueResult_70] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux11"} : <i1>, [<i32>, <i32>] to <i32>
    %19 = mux %29 [%1, %trueResult_68] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux12"} : <i1>, [<i32>, <i32>] to <i32>
    %20 = mux %29 [%0, %trueResult_68] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux13"} : <i1>, [<i32>, <i32>] to <i32>
    %21 = mux %29 [%arg8, %trueResult_72] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux14"} : <i1>, [<>, <>] to <>
    %22 = mux %29 [%arg8, %trueResult_62] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux15"} : <i1>, [<>, <>] to <>
    %23 = mux %29 [%arg8, %trueResult_64] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux16"} : <i1>, [<>, <>] to <>
    %24 = mux %29 [%arg8, %trueResult_60] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux17"} : <i1>, [<>, <>] to <>
    %25 = mux %29 [%arg8, %trueResult_62] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux18"} : <i1>, [<>, <>] to <>
    %26 = mux %29 [%arg8, %trueResult_60] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux19"} : <i1>, [<>, <>] to <>
    %27 = mux %29 [%arg8, %trueResult_72] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux20"} : <i1>, [<>, <>] to <>
    %28 = mux %29 [%arg8, %trueResult_64] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux21"} : <i1>, [<>, <>] to <>
    %29 = init %131 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "init0"} : <i1>
    %30 = mux %index [%9, %trueResult_78] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i32>, <i32>] to <i32>
    %result, %index = control_merge [%10, %trueResult_80]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %31 = constant %result {handshake.bb = 1 : ui32, handshake.name = "constant6", value = 0 : i32} : <>, <i32>
    %32 = source {handshake.bb = 1 : ui32, handshake.name = "source0"} : <>
    %33 = constant %32 {handshake.bb = 1 : ui32, handshake.name = "constant7", value = -1 : i32} : <>, <i32>
    %34 = addi %30, %33 {handshake.bb = 1 : ui32, handshake.name = "addi0"} : <i32>
    %35 = br %31 {handshake.bb = 1 : ui32, handshake.name = "br4"} : <i32>
    %36 = br %30 {handshake.bb = 1 : ui32, handshake.name = "br5"} : <i32>
    %37 = br %34 {handshake.bb = 1 : ui32, handshake.name = "br6"} : <i32>
    %38 = br %result {handshake.bb = 1 : ui32, handshake.name = "br7"} : <>
    %trueResult, %falseResult = cond_br %124, %121 {handshake.bb = 2 : ui32, handshake.name = "cond_br68"} : <i1>, <>
    %trueResult_6, %falseResult_7 = cond_br %124, %122 {handshake.bb = 2 : ui32, handshake.name = "cond_br69"} : <i1>, <>
    %trueResult_8, %falseResult_9 = cond_br %124, %114 {handshake.bb = 2 : ui32, handshake.name = "cond_br70"} : <i1>, <i32>
    %trueResult_10, %falseResult_11 = cond_br %124, %120 {handshake.bb = 2 : ui32, handshake.name = "cond_br71"} : <i1>, <>
    %trueResult_12, %falseResult_13 = cond_br %124, %117 {handshake.bb = 2 : ui32, handshake.name = "cond_br72"} : <i1>, <i32>
    %trueResult_14, %falseResult_15 = cond_br %124, %118 {handshake.bb = 2 : ui32, handshake.name = "cond_br73"} : <i1>, <>
    %trueResult_16, %falseResult_17 = cond_br %124, %119 {handshake.bb = 2 : ui32, handshake.name = "cond_br74"} : <i1>, <>
    %trueResult_18, %falseResult_19 = cond_br %124, %116 {handshake.bb = 2 : ui32, handshake.name = "cond_br75"} : <i1>, <i32>
    %trueResult_20, %falseResult_21 = cond_br %124, %115 {handshake.bb = 2 : ui32, handshake.name = "cond_br76"} : <i1>, <i32>
    %39 = mux %57 [%11, %trueResult_8] {ftd.phi, handshake.bb = 2 : ui32, handshake.name = "mux22"} : <i1>, [<i32>, <i32>] to <i32>
    %40 = mux %57 [%12, %trueResult_6] {ftd.phi, handshake.bb = 2 : ui32, handshake.name = "mux23"} : <i1>, [<>, <>] to <>
    %41 = mux %57 [%13, %trueResult_6] {ftd.phi, handshake.bb = 2 : ui32, handshake.name = "mux24"} : <i1>, [<>, <>] to <>
    %42 = mux %57 [%14, %trueResult_18] {ftd.phi, handshake.bb = 2 : ui32, handshake.name = "mux25"} : <i1>, [<i32>, <i32>] to <i32>
    %43 = mux %57 [%15, %trueResult_20] {ftd.phi, handshake.bb = 2 : ui32, handshake.name = "mux26"} : <i1>, [<i32>, <i32>] to <i32>
    %44 = mux %57 [%16, %trueResult_20] {ftd.phi, handshake.bb = 2 : ui32, handshake.name = "mux27"} : <i1>, [<i32>, <i32>] to <i32>
    %45 = mux %57 [%17, %trueResult_18] {ftd.phi, handshake.bb = 2 : ui32, handshake.name = "mux28"} : <i1>, [<i32>, <i32>] to <i32>
    %46 = mux %57 [%18, %trueResult_8] {ftd.phi, handshake.bb = 2 : ui32, handshake.name = "mux29"} : <i1>, [<i32>, <i32>] to <i32>
    %47 = mux %57 [%19, %trueResult_12] {ftd.phi, handshake.bb = 2 : ui32, handshake.name = "mux30"} : <i1>, [<i32>, <i32>] to <i32>
    %48 = mux %57 [%20, %trueResult_12] {ftd.phi, handshake.bb = 2 : ui32, handshake.name = "mux31"} : <i1>, [<i32>, <i32>] to <i32>
    %49 = mux %57 [%21, %trueResult_16] {ftd.phi, handshake.bb = 2 : ui32, handshake.name = "mux32"} : <i1>, [<>, <>] to <>
    %50 = mux %57 [%22, %trueResult] {ftd.phi, handshake.bb = 2 : ui32, handshake.name = "mux33"} : <i1>, [<>, <>] to <>
    %51 = mux %57 [%23, %trueResult_14] {ftd.phi, handshake.bb = 2 : ui32, handshake.name = "mux34"} : <i1>, [<>, <>] to <>
    %52 = mux %57 [%24, %trueResult_10] {ftd.phi, handshake.bb = 2 : ui32, handshake.name = "mux35"} : <i1>, [<>, <>] to <>
    %53 = mux %57 [%25, %trueResult] {ftd.phi, handshake.bb = 2 : ui32, handshake.name = "mux36"} : <i1>, [<>, <>] to <>
    %54 = mux %57 [%26, %trueResult_10] {ftd.phi, handshake.bb = 2 : ui32, handshake.name = "mux37"} : <i1>, [<>, <>] to <>
    %55 = mux %57 [%27, %trueResult_16] {ftd.phi, handshake.bb = 2 : ui32, handshake.name = "mux38"} : <i1>, [<>, <>] to <>
    %56 = mux %57 [%28, %trueResult_14] {ftd.phi, handshake.bb = 2 : ui32, handshake.name = "mux39"} : <i1>, [<>, <>] to <>
    %57 = init %124 {ftd.imerge, handshake.bb = 2 : ui32, handshake.name = "init18"} : <i1>
    %58 = mux %index_23 [%35, %trueResult_50] {handshake.bb = 2 : ui32, handshake.name = "mux1"} : <i1>, [<i32>, <i32>] to <i32>
    %59 = mux %index_23 [%36, %trueResult_52] {handshake.bb = 2 : ui32, handshake.name = "mux2"} : <i1>, [<i32>, <i32>] to <i32>
    %60 = mux %index_23 [%37, %trueResult_54] {handshake.bb = 2 : ui32, handshake.name = "mux3"} : <i1>, [<i32>, <i32>] to <i32>
    %result_22, %index_23 = control_merge [%38, %trueResult_56]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>, <>] to <>, <i1>
    %61 = constant %result_22 {handshake.bb = 2 : ui32, handshake.name = "constant17", value = 1 : i32} : <>, <i32>
    %62 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %63 = constant %62 {handshake.bb = 2 : ui32, handshake.name = "constant8", value = 1 : i32} : <>, <i32>
    %64 = source {handshake.bb = 2 : ui32, handshake.name = "source2"} : <>
    %65 = constant %64 {handshake.bb = 2 : ui32, handshake.name = "constant9", value = 20 : i32} : <>, <i32>
    %66 = source {handshake.bb = 2 : ui32, handshake.name = "source3"} : <>
    %67 = constant %66 {handshake.bb = 2 : ui32, handshake.name = "constant10", value = 4 : i32} : <>, <i32>
    %68 = source {handshake.bb = 2 : ui32, handshake.name = "source4"} : <>
    %69 = constant %68 {handshake.bb = 2 : ui32, handshake.name = "constant11", value = 2 : i32} : <>, <i32>
    %addressResult, %dataResult = load[%58] %outputs_2 {handshake.bb = 2 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i32>, <i32>, <i32>, <i32>
    %addressResult_24, %dataResult_25 = load[%58] %outputs {handshake.bb = 2 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i32>, <i32>, <i32>, <i32>
    %addressResult_26, %dataResult_27 = load[%58] %outputs_0 {handshake.bb = 2 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load2"} : <i32>, <i32>, <i32>, <i32>
    %70 = shli %60, %69 {handshake.bb = 2 : ui32, handshake.name = "shli0"} : <i32>
    %71 = shli %60, %67 {handshake.bb = 2 : ui32, handshake.name = "shli1"} : <i32>
    %72 = addi %70, %71 {handshake.bb = 2 : ui32, handshake.name = "addi7"} : <i32>
    %73 = addi %dataResult_27, %72 {handshake.bb = 2 : ui32, handshake.name = "addi2"} : <i32>
    %74 = gate %73, %41 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "gate0"} : <i32>, !handshake.control<> to <i32>
    %75 = cmpi ne, %74, %39 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi2"} : <i32>
    %76 = cmpi ne, %74, %44 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi3"} : <i32>
    %77 = cmpi ne, %74, %42 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi4"} : <i32>
    %78 = cmpi ne, %74, %47 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi5"} : <i32>
    %trueResult_28, %falseResult_29 = cond_br %75, %56 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br42"} : <i1>, <>
    %trueResult_30, %falseResult_31 = cond_br %76, %55 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br43"} : <i1>, <>
    %trueResult_32, %falseResult_33 = cond_br %77, %52 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br44"} : <i1>, <>
    %trueResult_34, %falseResult_35 = cond_br %78, %50 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br45"} : <i1>, <>
    %79 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source7"} : <>
    %80 = mux %75 [%falseResult_29, %79] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux40"} : <i1>, [<>, <>] to <>
    %81 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source8"} : <>
    %82 = mux %76 [%falseResult_31, %81] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux41"} : <i1>, [<>, <>] to <>
    %83 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source9"} : <>
    %84 = mux %77 [%falseResult_33, %83] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux42"} : <i1>, [<>, <>] to <>
    %85 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source10"} : <>
    %86 = mux %78 [%falseResult_35, %85] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux43"} : <i1>, [<>, <>] to <>
    %87 = join %80, %82, %84, %86 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "join0"} : <>
    %88 = gate %73, %87 {handshake.bb = 2 : ui32, handshake.name = "gate1"} : <i32>, !handshake.control<> to <i32>
    %addressResult_36, %dataResult_37 = load[%88] %outputs_4#0 {handshake.bb = 2 : ui32, handshake.deps = #handshake<deps[["store0", 0, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load3"} : <i32>, <i32>, <i32>, <i32>
    %89 = muli %dataResult_25, %dataResult_37 {handshake.bb = 2 : ui32, handshake.name = "muli0"} : <i32>
    %90 = shli %59, %69 {handshake.bb = 2 : ui32, handshake.name = "shli2"} : <i32>
    %91 = shli %59, %67 {handshake.bb = 2 : ui32, handshake.name = "shli3"} : <i32>
    %92 = addi %90, %91 {handshake.bb = 2 : ui32, handshake.name = "addi8"} : <i32>
    %93 = addi %dataResult, %92 {handshake.bb = 2 : ui32, handshake.name = "addi3"} : <i32>
    %94 = gate %93, %40 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "gate2"} : <i32>, !handshake.control<> to <i32>
    %95 = cmpi ne, %94, %46 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi6"} : <i32>
    %96 = cmpi ne, %94, %43 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi7"} : <i32>
    %97 = cmpi ne, %94, %45 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi8"} : <i32>
    %98 = cmpi ne, %94, %48 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi9"} : <i32>
    %trueResult_38, %falseResult_39 = cond_br %95, %51 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br46"} : <i1>, <>
    %trueResult_40, %falseResult_41 = cond_br %96, %49 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br47"} : <i1>, <>
    %trueResult_42, %falseResult_43 = cond_br %97, %54 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br48"} : <i1>, <>
    %trueResult_44, %falseResult_45 = cond_br %98, %53 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br49"} : <i1>, <>
    %99 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source11"} : <>
    %100 = mux %95 [%falseResult_39, %99] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux44"} : <i1>, [<>, <>] to <>
    %101 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source12"} : <>
    %102 = mux %96 [%falseResult_41, %101] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux45"} : <i1>, [<>, <>] to <>
    %103 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source13"} : <>
    %104 = mux %97 [%falseResult_43, %103] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux46"} : <i1>, [<>, <>] to <>
    %105 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source14"} : <>
    %106 = mux %98 [%falseResult_45, %105] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux47"} : <i1>, [<>, <>] to <>
    %107 = join %100, %102, %104, %106 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "join1"} : <>
    %108 = gate %93, %107 {handshake.bb = 2 : ui32, handshake.name = "gate3"} : <i32>, !handshake.control<> to <i32>
    %addressResult_46, %dataResult_47 = load[%108] %outputs_4#1 {handshake.bb = 2 : ui32, handshake.deps = #handshake<deps[["store0", 0, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load4"} : <i32>, <i32>, <i32>, <i32>
    %109 = addi %dataResult_47, %89 {handshake.bb = 2 : ui32, handshake.name = "addi1"} : <i32>
    %110 = shli %59, %69 {handshake.bb = 2 : ui32, handshake.name = "shli4"} : <i32>
    %111 = shli %59, %67 {handshake.bb = 2 : ui32, handshake.name = "shli5"} : <i32>
    %112 = addi %110, %111 {handshake.bb = 2 : ui32, handshake.name = "addi9"} : <i32>
    %113 = addi %dataResult, %112 {handshake.bb = 2 : ui32, handshake.name = "addi4"} : <i32>
    %114 = buffer %113, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer0"} : <i32>
    %115 = init %114 {handshake.bb = 2 : ui32, handshake.name = "init36"} : <i32>
    %116 = init %115 {handshake.bb = 2 : ui32, handshake.name = "init37"} : <i32>
    %117 = init %116 {handshake.bb = 2 : ui32, handshake.name = "init38"} : <i32>
    %118 = buffer %doneResult, bufferType = FIFO_BREAK_NONE, numSlots = 1 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "buffer1"} : <>
    %119 = init %118 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init39"} : <>
    %120 = init %119 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init40"} : <>
    %121 = init %120 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init41"} : <>
    %122 = init %121 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init42"} : <>
    %addressResult_48, %dataResult_49, %doneResult = store[%113] %109 %outputs_4#2 {handshake.bb = 2 : ui32, handshake.deps = #handshake<deps[["load3", 0, false], ["load4", 0, false], ["store0", 0, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store0"} : <i32>, <i32>, <>, <i32>, <i32>, <>
    %123 = addi %58, %63 {handshake.bb = 2 : ui32, handshake.name = "addi5"} : <i32>
    %124 = cmpi ult, %123, %65 {handshake.bb = 2 : ui32, handshake.name = "cmpi0"} : <i32>
    %trueResult_50, %falseResult_51 = cond_br %124, %123 {handshake.bb = 2 : ui32, handshake.name = "cond_br2"} : <i1>, <i32>
    %trueResult_52, %falseResult_53 = cond_br %124, %59 {handshake.bb = 2 : ui32, handshake.name = "cond_br3"} : <i1>, <i32>
    %trueResult_54, %falseResult_55 = cond_br %124, %60 {handshake.bb = 2 : ui32, handshake.name = "cond_br4"} : <i1>, <i32>
    %trueResult_56, %falseResult_57 = cond_br %124, %result_22 {handshake.bb = 2 : ui32, handshake.name = "cond_br5"} : <i1>, <>
    %trueResult_58, %falseResult_59 = cond_br %131, %falseResult_7 {handshake.bb = 3 : ui32, handshake.name = "cond_br77"} : <i1>, <>
    %trueResult_60, %falseResult_61 = cond_br %131, %falseResult_11 {handshake.bb = 3 : ui32, handshake.name = "cond_br78"} : <i1>, <>
    %trueResult_62, %falseResult_63 = cond_br %131, %falseResult {handshake.bb = 3 : ui32, handshake.name = "cond_br79"} : <i1>, <>
    %trueResult_64, %falseResult_65 = cond_br %131, %falseResult_15 {handshake.bb = 3 : ui32, handshake.name = "cond_br80"} : <i1>, <>
    %trueResult_66, %falseResult_67 = cond_br %131, %falseResult_19 {handshake.bb = 3 : ui32, handshake.name = "cond_br81"} : <i1>, <i32>
    %trueResult_68, %falseResult_69 = cond_br %131, %falseResult_13 {handshake.bb = 3 : ui32, handshake.name = "cond_br82"} : <i1>, <i32>
    %trueResult_70, %falseResult_71 = cond_br %131, %falseResult_9 {handshake.bb = 3 : ui32, handshake.name = "cond_br83"} : <i1>, <i32>
    %trueResult_72, %falseResult_73 = cond_br %131, %falseResult_17 {handshake.bb = 3 : ui32, handshake.name = "cond_br84"} : <i1>, <>
    %trueResult_74, %falseResult_75 = cond_br %131, %falseResult_21 {handshake.bb = 3 : ui32, handshake.name = "cond_br85"} : <i1>, <i32>
    %125 = merge %falseResult_53 {handshake.bb = 3 : ui32, handshake.name = "merge0"} : <i32>
    %result_76, %index_77 = control_merge [%falseResult_57]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>] to <>, <i1>
    %126 = source {handshake.bb = 3 : ui32, handshake.name = "source5"} : <>
    %127 = constant %126 {handshake.bb = 3 : ui32, handshake.name = "constant12", value = 1 : i32} : <>, <i32>
    %128 = source {handshake.bb = 3 : ui32, handshake.name = "source6"} : <>
    %129 = constant %128 {handshake.bb = 3 : ui32, handshake.name = "constant13", value = 20 : i32} : <>, <i32>
    %130 = addi %125, %127 {handshake.bb = 3 : ui32, handshake.name = "addi6"} : <i32>
    %131 = cmpi ult, %130, %129 {handshake.bb = 3 : ui32, handshake.name = "cmpi1"} : <i32>
    %trueResult_78, %falseResult_79 = cond_br %131, %130 {handshake.bb = 3 : ui32, handshake.name = "cond_br6"} : <i1>, <i32>
    %trueResult_80, %falseResult_81 = cond_br %131, %result_76 {handshake.bb = 3 : ui32, handshake.name = "cond_br7"} : <i1>, <>
    %result_82, %index_83 = control_merge [%falseResult_81]  {handshake.bb = 4 : ui32, handshake.name = "control_merge3"} : [<>] to <>, <i1>
    end {handshake.bb = 4 : ui32, handshake.name = "end0"} %memEnd_5, %memEnd_3, %memEnd_1, %memEnd, %arg8 : <>, <>, <>, <>, <>
  }
}

