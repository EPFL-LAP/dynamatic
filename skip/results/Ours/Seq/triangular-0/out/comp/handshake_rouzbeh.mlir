module {
  handshake.func @triangular(%arg0: memref<10xi32>, %arg1: !handshake.channel<i32>, %arg2: memref<100xi32>, %arg3: !handshake.control<>, %arg4: !handshake.control<>, %arg5: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["x", "n", "a", "x_start", "a_start", "start"], cfg.edges = "[0,1][2,3][4,3][1,2,6,cmpi0][3,4,5,cmpi1][5,1]", resNames = ["x_end", "a_end", "end"]} {
    %outputs:3, %memEnd = mem_controller[%arg2 : memref<100xi32>] %arg4 (%35, %addressResult, %addressResult_30, %addressResult_32, %dataResult_33) %result_36 {connectedBlocks = [4 : i32], handshake.name = "mem_controller2"} :    (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>) -> (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>)
    %outputs_0, %memEnd_1 = mem_controller[%arg0 : memref<10xi32>] %arg3 (%addressResult_28) %result_36 {connectedBlocks = [4 : i32], handshake.name = "mem_controller3"} :    (!handshake.channel<i32>) -> !handshake.channel<i32>
    %0 = constant %arg5 {handshake.bb = 0 : ui32, handshake.name = "constant1", value = 0 : i32} : <>, <i32>
    %1 = br %0 {handshake.bb = 0 : ui32, handshake.name = "br4"} : <i32>
    %2 = br %arg1 {handshake.bb = 0 : ui32, handshake.name = "br5"} : <i32>
    %3 = br %arg5 {handshake.bb = 0 : ui32, handshake.name = "br6"} : <>
    %4 = mux %5 [%arg5, %falseResult_25] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux7"} : <i1>, [<>, <>] to <>
    %5 = init %8 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "init0"} : <i1>
    %6 = mux %index [%1, %89] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i32>, <i32>] to <i32>
    %7 = mux %index [%2, %90] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i1>, [<i32>, <i32>] to <i32>
    %result, %index = control_merge [%3, %91]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %8 = cmpi slt, %6, %7 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i32>
    %trueResult, %falseResult = cond_br %8, %7 {handshake.bb = 1 : ui32, handshake.name = "cond_br2"} : <i1>, <i32>
    %trueResult_2, %falseResult_3 = cond_br %8, %6 {handshake.bb = 1 : ui32, handshake.name = "cond_br3"} : <i1>, <i32>
    %trueResult_4, %falseResult_5 = cond_br %8, %result {handshake.bb = 1 : ui32, handshake.name = "cond_br4"} : <i1>, <>
    %9 = merge %trueResult {handshake.bb = 2 : ui32, handshake.name = "merge0"} : <i32>
    %10 = merge %trueResult_2 {handshake.bb = 2 : ui32, handshake.name = "merge1"} : <i32>
    %result_6, %index_7 = control_merge [%trueResult_4]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>] to <>, <i1>
    %11 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %12 = constant %11 {handshake.bb = 2 : ui32, handshake.name = "constant5", value = -1 : i32} : <>, <i32>
    %13 = constant %result_6 {handshake.bb = 2 : ui32, handshake.name = "constant6", value = 0 : i32} : <>, <i32>
    %14 = subi %9, %10 {handshake.bb = 2 : ui32, handshake.name = "subi1"} : <i32>
    %15 = addi %14, %12 {handshake.bb = 2 : ui32, handshake.name = "addi1"} : <i32>
    %16 = br %13 {handshake.bb = 2 : ui32, handshake.name = "br7"} : <i32>
    %17 = br %9 {handshake.bb = 2 : ui32, handshake.name = "br8"} : <i32>
    %18 = br %10 {handshake.bb = 2 : ui32, handshake.name = "br9"} : <i32>
    %19 = br %14 {handshake.bb = 2 : ui32, handshake.name = "br10"} : <i32>
    %20 = br %15 {handshake.bb = 2 : ui32, handshake.name = "br11"} : <i32>
    %21 = br %result_6 {handshake.bb = 2 : ui32, handshake.name = "br12"} : <>
    %trueResult_8, %falseResult_9 = cond_br %8, %4 {handshake.bb = 3 : ui32, handshake.name = "cond_br12"} : <i1>, <>
    %22 = mux %23 [%trueResult_8, %76] {ftd.phi, handshake.bb = 3 : ui32, handshake.name = "mux8"} : <i1>, [<>, <>] to <>
    %23 = init %29 {ftd.imerge, handshake.bb = 3 : ui32, handshake.name = "init1"} : <i1>
    %24 = mux %index_11 [%16, %78] {handshake.bb = 3 : ui32, handshake.name = "mux2"} : <i1>, [<i32>, <i32>] to <i32>
    %25 = mux %index_11 [%17, %79] {handshake.bb = 3 : ui32, handshake.name = "mux3"} : <i1>, [<i32>, <i32>] to <i32>
    %26 = mux %index_11 [%18, %80] {handshake.bb = 3 : ui32, handshake.name = "mux4"} : <i1>, [<i32>, <i32>] to <i32>
    %27 = mux %index_11 [%19, %81] {handshake.bb = 3 : ui32, handshake.name = "mux5"} : <i1>, [<i32>, <i32>] to <i32>
    %28 = mux %index_11 [%20, %82] {handshake.bb = 3 : ui32, handshake.name = "mux6"} : <i1>, [<i32>, <i32>] to <i32>
    %result_10, %index_11 = control_merge [%21, %83]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>, <>] to <>, <i1>
    %29 = cmpi slt, %24, %28 {handshake.bb = 3 : ui32, handshake.name = "cmpi1"} : <i32>
    %trueResult_12, %falseResult_13 = cond_br %29, %25 {handshake.bb = 3 : ui32, handshake.name = "cond_br5"} : <i1>, <i32>
    %trueResult_14, %falseResult_15 = cond_br %29, %26 {handshake.bb = 3 : ui32, handshake.name = "cond_br6"} : <i1>, <i32>
    %trueResult_16, %falseResult_17 = cond_br %29, %27 {handshake.bb = 3 : ui32, handshake.name = "cond_br7"} : <i1>, <i32>
    %trueResult_18, %falseResult_19 = cond_br %29, %28 {handshake.bb = 3 : ui32, handshake.name = "cond_br8"} : <i1>, <i32>
    %trueResult_20, %falseResult_21 = cond_br %29, %24 {handshake.bb = 3 : ui32, handshake.name = "cond_br9"} : <i1>, <i32>
    %trueResult_22, %falseResult_23 = cond_br %29, %result_10 {handshake.bb = 3 : ui32, handshake.name = "cond_br10"} : <i1>, <>
    %trueResult_24, %falseResult_25 = cond_br %29, %22 {handshake.bb = 4 : ui32, handshake.name = "cond_br13"} : <i1>, <>
    %30 = merge %trueResult_12 {handshake.bb = 4 : ui32, handshake.name = "merge2"} : <i32>
    %31 = merge %trueResult_14 {handshake.bb = 4 : ui32, handshake.name = "merge3"} : <i32>
    %32 = merge %trueResult_16 {handshake.bb = 4 : ui32, handshake.name = "merge4"} : <i32>
    %33 = merge %trueResult_18 {handshake.bb = 4 : ui32, handshake.name = "merge5"} : <i32>
    %34 = merge %trueResult_20 {handshake.bb = 4 : ui32, handshake.name = "merge6"} : <i32>
    %result_26, %index_27 = control_merge [%trueResult_22]  {handshake.bb = 4 : ui32, handshake.name = "control_merge3"} : [<>] to <>, <i1>
    %35 = constant %result_26 {handshake.bb = 4 : ui32, handshake.name = "constant2", value = 1 : i32} : <>, <i32>
    %36 = source {handshake.bb = 4 : ui32, handshake.name = "source1"} : <>
    %37 = constant %36 {handshake.bb = 4 : ui32, handshake.name = "constant7", value = -2 : i32} : <>, <i32>
    %38 = source {handshake.bb = 4 : ui32, handshake.name = "source2"} : <>
    %39 = constant %38 {handshake.bb = 4 : ui32, handshake.name = "constant8", value = -1 : i32} : <>, <i32>
    %40 = source {handshake.bb = 4 : ui32, handshake.name = "source3"} : <>
    %41 = constant %40 {handshake.bb = 4 : ui32, handshake.name = "constant9", value = 1 : i32} : <>, <i32>
    %42 = source {handshake.bb = 4 : ui32, handshake.name = "source4"} : <>
    %43 = constant %42 {handshake.bb = 4 : ui32, handshake.name = "constant10", value = 3 : i32} : <>, <i32>
    %44 = addi %31, %34 {handshake.bb = 4 : ui32, handshake.name = "addi0"} : <i32>
    %45 = xori %44, %39 {handshake.bb = 4 : ui32, handshake.name = "xori0"} : <i32>
    %46 = addi %45, %41 {handshake.bb = 4 : ui32, handshake.name = "addi2"} : <i32>
    %47 = addi %46, %30 {handshake.bb = 4 : ui32, handshake.name = "addi3"} : <i32>
    %48 = addi %47, %37 {handshake.bb = 4 : ui32, handshake.name = "addi4"} : <i32>
    %49 = addi %32, %39 {handshake.bb = 4 : ui32, handshake.name = "addi6"} : <i32>
    %50 = shli %48, %41 {handshake.bb = 4 : ui32, handshake.name = "shli0"} : <i32>
    %51 = shli %48, %43 {handshake.bb = 4 : ui32, handshake.name = "shli1"} : <i32>
    %52 = addi %50, %51 {handshake.bb = 4 : ui32, handshake.name = "addi5"} : <i32>
    %53 = addi %49, %52 {handshake.bb = 4 : ui32, handshake.name = "addi15"} : <i32>
    %addressResult, %dataResult = load[%53] %outputs#0 {handshake.bb = 4 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i32>, <i32>, <i32>, <i32>
    %54 = addi %32, %39 {handshake.bb = 4 : ui32, handshake.name = "addi8"} : <i32>
    %addressResult_28, %dataResult_29 = load[%54] %outputs_0 {handshake.bb = 4 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i32>, <i32>, <i32>, <i32>
    %55 = muli %dataResult, %dataResult_29 {handshake.bb = 4 : ui32, handshake.name = "muli0"} : <i32>
    %56 = addi %31, %34 {handshake.bb = 4 : ui32, handshake.name = "addi7"} : <i32>
    %57 = xori %56, %39 {handshake.bb = 4 : ui32, handshake.name = "xori1"} : <i32>
    %58 = addi %57, %41 {handshake.bb = 4 : ui32, handshake.name = "addi9"} : <i32>
    %59 = addi %58, %30 {handshake.bb = 4 : ui32, handshake.name = "addi10"} : <i32>
    %60 = addi %59, %37 {handshake.bb = 4 : ui32, handshake.name = "addi11"} : <i32>
    %61 = shli %60, %41 {handshake.bb = 4 : ui32, handshake.name = "shli2"} : <i32>
    %62 = shli %60, %43 {handshake.bb = 4 : ui32, handshake.name = "shli3"} : <i32>
    %63 = addi %61, %62 {handshake.bb = 4 : ui32, handshake.name = "addi12"} : <i32>
    %64 = addi %30, %63 {handshake.bb = 4 : ui32, handshake.name = "addi16"} : <i32>
    %65 = gate %64, %trueResult_24 {handshake.bb = 4 : ui32, handshake.name = "gate0"} : <i32>, !handshake.control<> to <i32>
    %addressResult_30, %dataResult_31 = load[%65] %outputs#1 {handshake.bb = 4 : ui32, handshake.deps = #handshake<deps[["store0", 1, false], ["store0", 3, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load2"} : <i32>, <i32>, <i32>, <i32>
    %66 = subi %dataResult_31, %55 {handshake.bb = 4 : ui32, handshake.name = "subi0"} : <i32>
    %67 = addi %31, %34 {handshake.bb = 4 : ui32, handshake.name = "addi20"} : <i32>
    %68 = xori %67, %39 {handshake.bb = 4 : ui32, handshake.name = "xori2"} : <i32>
    %69 = addi %68, %41 {handshake.bb = 4 : ui32, handshake.name = "addi21"} : <i32>
    %70 = addi %69, %30 {handshake.bb = 4 : ui32, handshake.name = "addi13"} : <i32>
    %71 = addi %70, %37 {handshake.bb = 4 : ui32, handshake.name = "addi14"} : <i32>
    %72 = shli %71, %41 {handshake.bb = 4 : ui32, handshake.name = "shli4"} : <i32>
    %73 = shli %71, %43 {handshake.bb = 4 : ui32, handshake.name = "shli5"} : <i32>
    %74 = addi %72, %73 {handshake.bb = 4 : ui32, handshake.name = "addi22"} : <i32>
    %75 = addi %30, %74 {handshake.bb = 4 : ui32, handshake.name = "addi17"} : <i32>
    %76 = buffer %doneResult, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer0"} : <>
    %addressResult_32, %dataResult_33, %doneResult = store[%75] %66 %outputs#2 {handshake.bb = 4 : ui32, handshake.deps = #handshake<deps[["load2", 1, false], ["store0", 1, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store0"} : <i32>, <i32>, <>, <i32>, <i32>, <>
    %77 = addi %34, %41 {handshake.bb = 4 : ui32, handshake.name = "addi18"} : <i32>
    %78 = br %77 {handshake.bb = 4 : ui32, handshake.name = "br13"} : <i32>
    %79 = br %30 {handshake.bb = 4 : ui32, handshake.name = "br14"} : <i32>
    %80 = br %31 {handshake.bb = 4 : ui32, handshake.name = "br15"} : <i32>
    %81 = br %32 {handshake.bb = 4 : ui32, handshake.name = "br16"} : <i32>
    %82 = br %33 {handshake.bb = 4 : ui32, handshake.name = "br17"} : <i32>
    %83 = br %result_26 {handshake.bb = 4 : ui32, handshake.name = "br18"} : <>
    %84 = merge %falseResult_13 {handshake.bb = 5 : ui32, handshake.name = "merge7"} : <i32>
    %85 = merge %falseResult_15 {handshake.bb = 5 : ui32, handshake.name = "merge8"} : <i32>
    %result_34, %index_35 = control_merge [%falseResult_23]  {handshake.bb = 5 : ui32, handshake.name = "control_merge4"} : [<>] to <>, <i1>
    %86 = source {handshake.bb = 5 : ui32, handshake.name = "source5"} : <>
    %87 = constant %86 {handshake.bb = 5 : ui32, handshake.name = "constant11", value = 1 : i32} : <>, <i32>
    %88 = addi %85, %87 {handshake.bb = 5 : ui32, handshake.name = "addi19"} : <i32>
    %89 = br %88 {handshake.bb = 5 : ui32, handshake.name = "br19"} : <i32>
    %90 = br %84 {handshake.bb = 5 : ui32, handshake.name = "br20"} : <i32>
    %91 = br %result_34 {handshake.bb = 5 : ui32, handshake.name = "br21"} : <>
    %result_36, %index_37 = control_merge [%falseResult_5]  {handshake.bb = 6 : ui32, handshake.name = "control_merge5"} : [<>] to <>, <i1>
    end {handshake.bb = 6 : ui32, handshake.name = "end0"} %memEnd_1, %memEnd, %arg5 : <>, <>, <>
  }
}

