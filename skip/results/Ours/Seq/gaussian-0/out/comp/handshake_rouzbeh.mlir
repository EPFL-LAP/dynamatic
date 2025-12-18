module {
  handshake.func @gaussian(%arg0: memref<20xi32>, %arg1: memref<400xi32>, %arg2: !handshake.control<>, %arg3: !handshake.control<>, %arg4: !handshake.control<>, ...) -> (!handshake.channel<i32>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["c", "a", "c_start", "a_start", "start"], cfg.edges = "[0,1][2,3,5,cmpi2][4,2][1,2][3,3,4,cmpi0][5,1,6,cmpi1]", resNames = ["out0", "c_end", "a_end", "end"]} {
    %outputs:3, %memEnd = mem_controller[%arg1 : memref<400xi32>] %arg3 (%36, %addressResult, %addressResult_24, %addressResult_26, %dataResult_27) %result_54 {connectedBlocks = [3 : i32], handshake.name = "mem_controller1"} :    (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>) -> (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>)
    %outputs_0, %memEnd_1 = mem_controller[%arg0 : memref<20xi32>] %arg2 (%addressResult_22) %result_54 {connectedBlocks = [3 : i32], handshake.name = "mem_controller2"} :    (!handshake.channel<i32>) -> !handshake.channel<i32>
    %0 = constant %arg4 {handshake.bb = 0 : ui32, handshake.name = "constant2", value = 0 : i32} : <>, <i32>
    %1 = constant %arg4 {handshake.bb = 0 : ui32, handshake.name = "constant4", value = 1 : i32} : <>, <i32>
    %2 = br %1 {handshake.bb = 0 : ui32, handshake.name = "br2"} : <i32>
    %3 = br %0 {handshake.bb = 0 : ui32, handshake.name = "br4"} : <i32>
    %4 = br %arg4 {handshake.bb = 0 : ui32, handshake.name = "br5"} : <>
    %5 = mux %7 [%arg4, %trueResult_42] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux10"} : <i1>, [<>, <>] to <>
    %6 = mux %7 [%arg4, %trueResult_44] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux11"} : <i1>, [<>, <>] to <>
    %7 = init %85 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "init0"} : <i1>
    %8 = mux %index [%2, %trueResult_48] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i32>, <i32>] to <i32>
    %9 = mux %index [%3, %trueResult_50] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i1>, [<i32>, <i32>] to <i32>
    %result, %index = control_merge [%4, %trueResult_52]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %10 = source {handshake.bb = 1 : ui32, handshake.name = "source0"} : <>
    %11 = constant %10 {handshake.bb = 1 : ui32, handshake.name = "constant7", value = 1 : i32} : <>, <i32>
    %12 = addi %8, %11 {handshake.bb = 1 : ui32, handshake.name = "addi2"} : <i32>
    %13 = br %12 {handshake.bb = 1 : ui32, handshake.name = "br6"} : <i32>
    %14 = br %9 {handshake.bb = 1 : ui32, handshake.name = "br7"} : <i32>
    %15 = br %8 {handshake.bb = 1 : ui32, handshake.name = "br8"} : <i32>
    %16 = br %result {handshake.bb = 1 : ui32, handshake.name = "br9"} : <>
    %17 = mux %19 [%5, %falseResult_19] {ftd.phi, handshake.bb = 2 : ui32, handshake.name = "mux12"} : <i1>, [<>, <>] to <>
    %18 = mux %19 [%6, %falseResult_19] {ftd.phi, handshake.bb = 2 : ui32, handshake.name = "mux13"} : <i1>, [<>, <>] to <>
    %19 = init %27 {ftd.imerge, handshake.bb = 2 : ui32, handshake.name = "init2"} : <i1>
    %20 = mux %index_3 [%13, %74] {handshake.bb = 2 : ui32, handshake.name = "mux2"} : <i1>, [<i32>, <i32>] to <i32>
    %21 = mux %index_3 [%14, %75] {handshake.bb = 2 : ui32, handshake.name = "mux3"} : <i1>, [<i32>, <i32>] to <i32>
    %22 = mux %index_3 [%15, %76] {handshake.bb = 2 : ui32, handshake.name = "mux4"} : <i1>, [<i32>, <i32>] to <i32>
    %result_2, %index_3 = control_merge [%16, %77]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>, <>] to <>, <i1>
    %23 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %24 = constant %23 {handshake.bb = 2 : ui32, handshake.name = "constant8", value = 19 : i32} : <>, <i32>
    %25 = constant %result_2 {handshake.bb = 2 : ui32, handshake.name = "constant9", value = 1 : i32} : <>, <i32>
    %26 = constant %result_2 {handshake.bb = 2 : ui32, handshake.name = "constant10", value = 1 : i32} : <>, <i32>
    %27 = cmpi ult, %20, %24 {handshake.bb = 2 : ui32, handshake.name = "cmpi2"} : <i32>
    %trueResult, %falseResult = cond_br %27, %25 {handshake.bb = 2 : ui32, handshake.name = "cond_br3"} : <i1>, <i32>
    %trueResult_4, %falseResult_5 = cond_br %27, %26 {handshake.bb = 2 : ui32, handshake.name = "cond_br4"} : <i1>, <i32>
    %trueResult_6, %falseResult_7 = cond_br %27, %21 {handshake.bb = 2 : ui32, handshake.name = "cond_br5"} : <i1>, <i32>
    %trueResult_8, %falseResult_9 = cond_br %27, %22 {handshake.bb = 2 : ui32, handshake.name = "cond_br6"} : <i1>, <i32>
    %trueResult_10, %falseResult_11 = cond_br %27, %20 {handshake.bb = 2 : ui32, handshake.name = "cond_br7"} : <i1>, <i32>
    %trueResult_12, %falseResult_13 = cond_br %27, %result_2 {handshake.bb = 2 : ui32, handshake.name = "cond_br8"} : <i1>, <>
    %trueResult_14, %falseResult_15 = cond_br %27, %18 {handshake.bb = 3 : ui32, handshake.name = "cond_br25"} : <i1>, <>
    %trueResult_16, %falseResult_17 = cond_br %27, %17 {handshake.bb = 3 : ui32, handshake.name = "cond_br26"} : <i1>, <>
    %trueResult_18, %falseResult_19 = cond_br %67, %63 {handshake.bb = 3 : ui32, handshake.name = "cond_br27"} : <i1>, <>
    %28 = mux %30 [%trueResult_16, %trueResult_18] {ftd.phi, handshake.bb = 3 : ui32, handshake.name = "mux14"} : <i1>, [<>, <>] to <>
    %29 = mux %30 [%trueResult_14, %trueResult_18] {ftd.phi, handshake.bb = 3 : ui32, handshake.name = "mux15"} : <i1>, [<>, <>] to <>
    %30 = init %67 {ftd.imerge, handshake.bb = 3 : ui32, handshake.name = "init4"} : <i1>
    %31 = mux %index_21 [%trueResult, %trueResult_28] {handshake.bb = 3 : ui32, handshake.name = "mux5"} : <i1>, [<i32>, <i32>] to <i32>
    %32 = mux %index_21 [%trueResult_4, %trueResult_30] {handshake.bb = 3 : ui32, handshake.name = "mux6"} : <i1>, [<i32>, <i32>] to <i32>
    %33 = mux %index_21 [%trueResult_6, %trueResult_32] {handshake.bb = 3 : ui32, handshake.name = "mux7"} : <i1>, [<i32>, <i32>] to <i32>
    %34 = mux %index_21 [%trueResult_8, %trueResult_34] {handshake.bb = 3 : ui32, handshake.name = "mux8"} : <i1>, [<i32>, <i32>] to <i32>
    %35 = mux %index_21 [%trueResult_10, %trueResult_36] {handshake.bb = 3 : ui32, handshake.name = "mux9"} : <i1>, [<i32>, <i32>] to <i32>
    %result_20, %index_21 = control_merge [%trueResult_12, %trueResult_38]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>, <>] to <>, <i1>
    %36 = constant %result_20 {handshake.bb = 3 : ui32, handshake.name = "constant0", value = 1 : i32} : <>, <i32>
    %37 = source {handshake.bb = 3 : ui32, handshake.name = "source2"} : <>
    %38 = constant %37 {handshake.bb = 3 : ui32, handshake.name = "constant11", value = 20 : i32} : <>, <i32>
    %39 = source {handshake.bb = 3 : ui32, handshake.name = "source3"} : <>
    %40 = constant %39 {handshake.bb = 3 : ui32, handshake.name = "constant12", value = 1 : i32} : <>, <i32>
    %41 = source {handshake.bb = 3 : ui32, handshake.name = "source4"} : <>
    %42 = constant %41 {handshake.bb = 3 : ui32, handshake.name = "constant13", value = 1 : i32} : <>, <i32>
    %43 = source {handshake.bb = 3 : ui32, handshake.name = "source5"} : <>
    %44 = constant %43 {handshake.bb = 3 : ui32, handshake.name = "constant14", value = 4 : i32} : <>, <i32>
    %45 = source {handshake.bb = 3 : ui32, handshake.name = "source6"} : <>
    %46 = constant %45 {handshake.bb = 3 : ui32, handshake.name = "constant15", value = 2 : i32} : <>, <i32>
    %47 = shli %35, %46 {handshake.bb = 3 : ui32, handshake.name = "shli0"} : <i32>
    %48 = shli %35, %44 {handshake.bb = 3 : ui32, handshake.name = "shli1"} : <i32>
    %49 = addi %47, %48 {handshake.bb = 3 : ui32, handshake.name = "addi9"} : <i32>
    %50 = addi %32, %49 {handshake.bb = 3 : ui32, handshake.name = "addi3"} : <i32>
    %51 = gate %50, %29 {handshake.bb = 3 : ui32, handshake.name = "gate0"} : <i32>, !handshake.control<> to <i32>
    %addressResult, %dataResult = load[%51] %outputs#0 {handshake.bb = 3 : ui32, handshake.deps = #handshake<deps[["store0", 0, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i32>, <i32>, <i32>, <i32>
    %addressResult_22, %dataResult_23 = load[%34] %outputs_0 {handshake.bb = 3 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i32>, <i32>, <i32>, <i32>
    %52 = shli %34, %46 {handshake.bb = 3 : ui32, handshake.name = "shli2"} : <i32>
    %53 = shli %34, %44 {handshake.bb = 3 : ui32, handshake.name = "shli3"} : <i32>
    %54 = addi %52, %53 {handshake.bb = 3 : ui32, handshake.name = "addi10"} : <i32>
    %55 = addi %32, %54 {handshake.bb = 3 : ui32, handshake.name = "addi4"} : <i32>
    %56 = gate %55, %28 {handshake.bb = 3 : ui32, handshake.name = "gate1"} : <i32>, !handshake.control<> to <i32>
    %addressResult_24, %dataResult_25 = load[%56] %outputs#1 {handshake.bb = 3 : ui32, handshake.deps = #handshake<deps[["store0", 0, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load2"} : <i32>, <i32>, <i32>, <i32>
    %57 = muli %dataResult_23, %dataResult_25 {handshake.bb = 3 : ui32, handshake.name = "muli0"} : <i32>
    %58 = subi %dataResult, %57 {handshake.bb = 3 : ui32, handshake.name = "subi0"} : <i32>
    %59 = shli %35, %46 {handshake.bb = 3 : ui32, handshake.name = "shli4"} : <i32>
    %60 = shli %35, %44 {handshake.bb = 3 : ui32, handshake.name = "shli5"} : <i32>
    %61 = addi %59, %60 {handshake.bb = 3 : ui32, handshake.name = "addi11"} : <i32>
    %62 = addi %32, %61 {handshake.bb = 3 : ui32, handshake.name = "addi5"} : <i32>
    %63 = buffer %doneResult, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer0"} : <>
    %addressResult_26, %dataResult_27, %doneResult = store[%62] %58 %outputs#2 {handshake.bb = 3 : ui32, handshake.deps = #handshake<deps[["load0", 0, false], ["load2", 0, false], ["store0", 0, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store0"} : <i32>, <i32>, <>, <i32>, <i32>, <>
    %64 = addi %33, %32 {handshake.bb = 3 : ui32, handshake.name = "addi0"} : <i32>
    %65 = addi %32, %42 {handshake.bb = 3 : ui32, handshake.name = "addi1"} : <i32>
    %66 = addi %31, %40 {handshake.bb = 3 : ui32, handshake.name = "addi6"} : <i32>
    %67 = cmpi ult, %66, %38 {handshake.bb = 3 : ui32, handshake.name = "cmpi0"} : <i32>
    %trueResult_28, %falseResult_29 = cond_br %67, %66 {handshake.bb = 3 : ui32, handshake.name = "cond_br9"} : <i1>, <i32>
    %trueResult_30, %falseResult_31 = cond_br %67, %65 {handshake.bb = 3 : ui32, handshake.name = "cond_br10"} : <i1>, <i32>
    %trueResult_32, %falseResult_33 = cond_br %67, %64 {handshake.bb = 3 : ui32, handshake.name = "cond_br11"} : <i1>, <i32>
    %trueResult_34, %falseResult_35 = cond_br %67, %34 {handshake.bb = 3 : ui32, handshake.name = "cond_br12"} : <i1>, <i32>
    %trueResult_36, %falseResult_37 = cond_br %67, %35 {handshake.bb = 3 : ui32, handshake.name = "cond_br13"} : <i1>, <i32>
    %trueResult_38, %falseResult_39 = cond_br %67, %result_20 {handshake.bb = 3 : ui32, handshake.name = "cond_br14"} : <i1>, <>
    %68 = merge %falseResult_35 {handshake.bb = 4 : ui32, handshake.name = "merge0"} : <i32>
    %69 = merge %falseResult_37 {handshake.bb = 4 : ui32, handshake.name = "merge1"} : <i32>
    %70 = merge %falseResult_33 {handshake.bb = 4 : ui32, handshake.name = "merge2"} : <i32>
    %result_40, %index_41 = control_merge [%falseResult_39]  {handshake.bb = 4 : ui32, handshake.name = "control_merge3"} : [<>] to <>, <i1>
    %71 = source {handshake.bb = 4 : ui32, handshake.name = "source7"} : <>
    %72 = constant %71 {handshake.bb = 4 : ui32, handshake.name = "constant16", value = 1 : i32} : <>, <i32>
    %73 = addi %69, %72 {handshake.bb = 4 : ui32, handshake.name = "addi8"} : <i32>
    %74 = br %73 {handshake.bb = 4 : ui32, handshake.name = "br10"} : <i32>
    %75 = br %70 {handshake.bb = 4 : ui32, handshake.name = "br11"} : <i32>
    %76 = br %68 {handshake.bb = 4 : ui32, handshake.name = "br12"} : <i32>
    %77 = br %result_40 {handshake.bb = 4 : ui32, handshake.name = "br13"} : <>
    %trueResult_42, %falseResult_43 = cond_br %85, %falseResult_17 {handshake.bb = 5 : ui32, handshake.name = "cond_br28"} : <i1>, <>
    %trueResult_44, %falseResult_45 = cond_br %85, %falseResult_15 {handshake.bb = 5 : ui32, handshake.name = "cond_br29"} : <i1>, <>
    %78 = merge %falseResult_9 {handshake.bb = 5 : ui32, handshake.name = "merge3"} : <i32>
    %79 = merge %falseResult_7 {handshake.bb = 5 : ui32, handshake.name = "merge4"} : <i32>
    %result_46, %index_47 = control_merge [%falseResult_13]  {handshake.bb = 5 : ui32, handshake.name = "control_merge4"} : [<>] to <>, <i1>
    %80 = source {handshake.bb = 5 : ui32, handshake.name = "source8"} : <>
    %81 = constant %80 {handshake.bb = 5 : ui32, handshake.name = "constant17", value = 19 : i32} : <>, <i32>
    %82 = source {handshake.bb = 5 : ui32, handshake.name = "source9"} : <>
    %83 = constant %82 {handshake.bb = 5 : ui32, handshake.name = "constant18", value = 1 : i32} : <>, <i32>
    %84 = addi %78, %83 {handshake.bb = 5 : ui32, handshake.name = "addi7"} : <i32>
    %85 = cmpi ult, %84, %81 {handshake.bb = 5 : ui32, handshake.name = "cmpi1"} : <i32>
    %trueResult_48, %falseResult_49 = cond_br %85, %84 {handshake.bb = 5 : ui32, handshake.name = "cond_br15"} : <i1>, <i32>
    %trueResult_50, %falseResult_51 = cond_br %85, %79 {handshake.bb = 5 : ui32, handshake.name = "cond_br16"} : <i1>, <i32>
    %trueResult_52, %falseResult_53 = cond_br %85, %result_46 {handshake.bb = 5 : ui32, handshake.name = "cond_br17"} : <i1>, <>
    %86 = merge %falseResult_51 {handshake.bb = 6 : ui32, handshake.name = "merge5"} : <i32>
    %result_54, %index_55 = control_merge [%falseResult_53]  {handshake.bb = 6 : ui32, handshake.name = "control_merge5"} : [<>] to <>, <i1>
    end {handshake.bb = 6 : ui32, handshake.name = "end0"} %86, %memEnd_1, %memEnd, %arg4 : <i32>, <>, <>, <>
  }
}

