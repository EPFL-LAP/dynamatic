module {
  handshake.func @triangular(%arg0: memref<10xi32>, %arg1: !handshake.channel<i32>, %arg2: memref<100xi32>, %arg3: !handshake.control<>, %arg4: !handshake.control<>, %arg5: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["x", "n", "a", "x_start", "a_start", "start"], cfg.edges = "[0,1][2,3][4,3][1,2,6,cmpi0][3,4,5,cmpi1][5,1]", resNames = ["x_end", "a_end", "end"]} {
    %outputs:3, %memEnd = mem_controller[%arg2 : memref<100xi32>] %arg4 (%28, %addressResult, %0#2, %0#3, %0#4) %result_32 {connectedBlocks = [4 : i32], handshake.name = "mem_controller0"} :    (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>) -> (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>)
    %0:5 = lsq[MC] (%result_22, %addressResult_26, %addressResult_28, %dataResult_29, %outputs#1, %outputs#2)  {groupSizes = [2 : i32], handshake.name = "lsq0"} : (!handshake.control<>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>) -> (!handshake.channel<i32>, !handshake.control<>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>)
    %outputs_0, %memEnd_1 = mem_controller[%arg0 : memref<10xi32>] %arg3 (%addressResult_24) %result_32 {connectedBlocks = [4 : i32], handshake.name = "mem_controller1"} :    (!handshake.channel<i32>) -> !handshake.channel<i32>
    %1 = constant %arg5 {handshake.bb = 0 : ui32, handshake.name = "constant1", value = 0 : i32} : <>, <i32>
    %2 = trunci %arg1 {handshake.bb = 0 : ui32, handshake.name = "index_cast0"} : <i32> to <i32>
    %3 = br %1 {handshake.bb = 0 : ui32, handshake.name = "br4"} : <i32>
    %4 = br %2 {handshake.bb = 0 : ui32, handshake.name = "br5"} : <i32>
    %5 = br %arg5 {handshake.bb = 0 : ui32, handshake.name = "br6"} : <>
    %6 = mux %index [%3, %85] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i32>, <i32>] to <i32>
    %7 = mux %index [%4, %86] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i1>, [<i32>, <i32>] to <i32>
    %result, %index = control_merge [%5, %87]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
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
    %22 = mux %index_9 [%16, %74] {handshake.bb = 3 : ui32, handshake.name = "mux2"} : <i1>, [<i32>, <i32>] to <i32>
    %23 = mux %index_9 [%17, %75] {handshake.bb = 3 : ui32, handshake.name = "mux3"} : <i1>, [<i32>, <i32>] to <i32>
    %24 = mux %index_9 [%18, %76] {handshake.bb = 3 : ui32, handshake.name = "mux4"} : <i1>, [<i32>, <i32>] to <i32>
    %25 = mux %index_9 [%19, %77] {handshake.bb = 3 : ui32, handshake.name = "mux5"} : <i1>, [<i32>, <i32>] to <i32>
    %26 = mux %index_9 [%20, %78] {handshake.bb = 3 : ui32, handshake.name = "mux6"} : <i1>, [<i32>, <i32>] to <i32>
    %result_8, %index_9 = control_merge [%21, %79]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>, <>] to <>, <i1>
    %27 = cmpi slt, %22, %26 {handshake.bb = 3 : ui32, handshake.name = "cmpi1"} : <i32>
    %trueResult_10, %falseResult_11 = cond_br %27, %23 {handshake.bb = 3 : ui32, handshake.name = "cond_br5"} : <i1>, <i32>
    %trueResult_12, %falseResult_13 = cond_br %27, %24 {handshake.bb = 3 : ui32, handshake.name = "cond_br6"} : <i1>, <i32>
    %trueResult_14, %falseResult_15 = cond_br %27, %25 {handshake.bb = 3 : ui32, handshake.name = "cond_br7"} : <i1>, <i32>
    %trueResult_16, %falseResult_17 = cond_br %27, %26 {handshake.bb = 3 : ui32, handshake.name = "cond_br8"} : <i1>, <i32>
    %trueResult_18, %falseResult_19 = cond_br %27, %22 {handshake.bb = 3 : ui32, handshake.name = "cond_br9"} : <i1>, <i32>
    %trueResult_20, %falseResult_21 = cond_br %27, %result_8 {handshake.bb = 3 : ui32, handshake.name = "cond_br10"} : <i1>, <>
    %28 = constant %result_22 {handshake.bb = 4 : ui32, handshake.name = "constant0", value = 1 : i32} : <>, <i32>
    %29 = merge %trueResult_10 {handshake.bb = 4 : ui32, handshake.name = "merge2"} : <i32>
    %30 = merge %trueResult_12 {handshake.bb = 4 : ui32, handshake.name = "merge3"} : <i32>
    %31 = merge %trueResult_14 {handshake.bb = 4 : ui32, handshake.name = "merge4"} : <i32>
    %32 = merge %trueResult_16 {handshake.bb = 4 : ui32, handshake.name = "merge5"} : <i32>
    %33 = merge %trueResult_18 {handshake.bb = 4 : ui32, handshake.name = "merge6"} : <i32>
    %result_22, %index_23 = control_merge [%trueResult_20]  {handshake.bb = 4 : ui32, handshake.name = "control_merge3"} : [<>] to <>, <i1>
    %34 = source {handshake.bb = 4 : ui32, handshake.name = "source1"} : <>
    %35 = constant %34 {handshake.bb = 4 : ui32, handshake.name = "constant7", value = -2 : i32} : <>, <i32>
    %36 = source {handshake.bb = 4 : ui32, handshake.name = "source2"} : <>
    %37 = constant %36 {handshake.bb = 4 : ui32, handshake.name = "constant8", value = -1 : i32} : <>, <i32>
    %38 = source {handshake.bb = 4 : ui32, handshake.name = "source3"} : <>
    %39 = constant %38 {handshake.bb = 4 : ui32, handshake.name = "constant9", value = 1 : i32} : <>, <i32>
    %40 = source {handshake.bb = 4 : ui32, handshake.name = "source4"} : <>
    %41 = constant %40 {handshake.bb = 4 : ui32, handshake.name = "constant10", value = 3 : i32} : <>, <i32>
    %42 = addi %30, %33 {handshake.bb = 4 : ui32, handshake.name = "addi0"} : <i32>
    %43 = xori %42, %37 {handshake.bb = 4 : ui32, handshake.name = "xori0"} : <i32>
    %44 = addi %43, %39 {handshake.bb = 4 : ui32, handshake.name = "addi2"} : <i32>
    %45 = addi %44, %29 {handshake.bb = 4 : ui32, handshake.name = "addi3"} : <i32>
    %46 = addi %45, %35 {handshake.bb = 4 : ui32, handshake.name = "addi4"} : <i32>
    %47 = addi %31, %37 {handshake.bb = 4 : ui32, handshake.name = "addi6"} : <i32>
    %48 = shli %46, %39 {handshake.bb = 4 : ui32, handshake.name = "shli0"} : <i32>
    %49 = shli %46, %41 {handshake.bb = 4 : ui32, handshake.name = "shli1"} : <i32>
    %50 = addi %48, %49 {handshake.bb = 4 : ui32, handshake.name = "addi5"} : <i32>
    %51 = addi %47, %50 {handshake.bb = 4 : ui32, handshake.name = "addi15"} : <i32>
    %addressResult, %dataResult = load[%51] %outputs#0 {handshake.bb = 4 : ui32, handshake.name = "load0"} : <i32>, <i32>, <i32>, <i32>
    %52 = addi %31, %37 {handshake.bb = 4 : ui32, handshake.name = "addi8"} : <i32>
    %addressResult_24, %dataResult_25 = load[%52] %outputs_0 {handshake.bb = 4 : ui32, handshake.name = "load1"} : <i32>, <i32>, <i32>, <i32>
    %53 = muli %dataResult, %dataResult_25 {handshake.bb = 4 : ui32, handshake.name = "muli0"} : <i32>
    %54 = addi %30, %33 {handshake.bb = 4 : ui32, handshake.name = "addi7"} : <i32>
    %55 = xori %54, %37 {handshake.bb = 4 : ui32, handshake.name = "xori1"} : <i32>
    %56 = addi %55, %39 {handshake.bb = 4 : ui32, handshake.name = "addi9"} : <i32>
    %57 = addi %56, %29 {handshake.bb = 4 : ui32, handshake.name = "addi10"} : <i32>
    %58 = addi %57, %35 {handshake.bb = 4 : ui32, handshake.name = "addi11"} : <i32>
    %59 = shli %58, %39 {handshake.bb = 4 : ui32, handshake.name = "shli2"} : <i32>
    %60 = shli %58, %41 {handshake.bb = 4 : ui32, handshake.name = "shli3"} : <i32>
    %61 = addi %59, %60 {handshake.bb = 4 : ui32, handshake.name = "addi12"} : <i32>
    %62 = addi %29, %61 {handshake.bb = 4 : ui32, handshake.name = "addi16"} : <i32>
    %addressResult_26, %dataResult_27 = load[%62] %0#0 {handshake.bb = 4 : ui32, handshake.deps = #handshake<deps[["store0", 1, true], ["store0", 3, true]]>, handshake.name = "load2"} : <i32>, <i32>, <i32>, <i32>
    %63 = subi %dataResult_27, %53 {handshake.bb = 4 : ui32, handshake.name = "subi0"} : <i32>
    %64 = addi %30, %33 {handshake.bb = 4 : ui32, handshake.name = "addi20"} : <i32>
    %65 = xori %64, %37 {handshake.bb = 4 : ui32, handshake.name = "xori2"} : <i32>
    %66 = addi %65, %39 {handshake.bb = 4 : ui32, handshake.name = "addi21"} : <i32>
    %67 = addi %66, %29 {handshake.bb = 4 : ui32, handshake.name = "addi13"} : <i32>
    %68 = addi %67, %35 {handshake.bb = 4 : ui32, handshake.name = "addi14"} : <i32>
    %69 = shli %68, %39 {handshake.bb = 4 : ui32, handshake.name = "shli4"} : <i32>
    %70 = shli %68, %41 {handshake.bb = 4 : ui32, handshake.name = "shli5"} : <i32>
    %71 = addi %69, %70 {handshake.bb = 4 : ui32, handshake.name = "addi22"} : <i32>
    %72 = addi %29, %71 {handshake.bb = 4 : ui32, handshake.name = "addi17"} : <i32>
    %addressResult_28, %dataResult_29, %doneResult = store[%72] %63 %0#1 {handshake.bb = 4 : ui32, handshake.deps = #handshake<deps[["load2", 1, true], ["store0", 1, true]]>, handshake.name = "store0"} : <i32>, <i32>, <>, <i32>, <i32>, <>
    %73 = addi %33, %39 {handshake.bb = 4 : ui32, handshake.name = "addi18"} : <i32>
    %74 = br %73 {handshake.bb = 4 : ui32, handshake.name = "br13"} : <i32>
    %75 = br %29 {handshake.bb = 4 : ui32, handshake.name = "br14"} : <i32>
    %76 = br %30 {handshake.bb = 4 : ui32, handshake.name = "br15"} : <i32>
    %77 = br %31 {handshake.bb = 4 : ui32, handshake.name = "br16"} : <i32>
    %78 = br %32 {handshake.bb = 4 : ui32, handshake.name = "br17"} : <i32>
    %79 = br %result_22 {handshake.bb = 4 : ui32, handshake.name = "br18"} : <>
    %80 = merge %falseResult_11 {handshake.bb = 5 : ui32, handshake.name = "merge7"} : <i32>
    %81 = merge %falseResult_13 {handshake.bb = 5 : ui32, handshake.name = "merge8"} : <i32>
    %result_30, %index_31 = control_merge [%falseResult_21]  {handshake.bb = 5 : ui32, handshake.name = "control_merge4"} : [<>] to <>, <i1>
    %82 = source {handshake.bb = 5 : ui32, handshake.name = "source5"} : <>
    %83 = constant %82 {handshake.bb = 5 : ui32, handshake.name = "constant11", value = 1 : i32} : <>, <i32>
    %84 = addi %81, %83 {handshake.bb = 5 : ui32, handshake.name = "addi19"} : <i32>
    %85 = br %84 {handshake.bb = 5 : ui32, handshake.name = "br19"} : <i32>
    %86 = br %80 {handshake.bb = 5 : ui32, handshake.name = "br20"} : <i32>
    %87 = br %result_30 {handshake.bb = 5 : ui32, handshake.name = "br21"} : <>
    %result_32, %index_33 = control_merge [%falseResult_5]  {handshake.bb = 6 : ui32, handshake.name = "control_merge5"} : [<>] to <>, <i1>
    end {handshake.bb = 6 : ui32, handshake.name = "end0"} %memEnd_1, %memEnd, %arg5 : <>, <>, <>
  }
}

