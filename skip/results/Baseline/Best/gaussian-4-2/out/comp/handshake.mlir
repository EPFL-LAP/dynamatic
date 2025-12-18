module {
  handshake.func @gaussian(%arg0: memref<20xi32>, %arg1: memref<400xi32>, %arg2: !handshake.control<>, %arg3: !handshake.control<>, %arg4: !handshake.control<>, ...) -> (!handshake.channel<i32>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["c", "a", "c_start", "a_start", "start"], resNames = ["out0", "c_end", "a_end", "end"]} {
    %0:3 = lsq[%arg1 : memref<400xi32>] (%arg3, %result_12, %addressResult, %addressResult_16, %addressResult_18, %dataResult_19, %result_42)  {groupSizes = [3 : i32], handshake.name = "lsq0"} : (!handshake.control<>, !handshake.control<>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>) -> (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>)
    %outputs, %memEnd = mem_controller[%arg0 : memref<20xi32>] %arg2 (%addressResult_14) %result_42 {connectedBlocks = [3 : i32], handshake.name = "mem_controller0"} :    (!handshake.channel<i32>) -> !handshake.channel<i32>
    %1 = constant %arg4 {handshake.bb = 0 : ui32, handshake.name = "constant2", value = 0 : i32} : <>, <i32>
    %2 = constant %arg4 {handshake.bb = 0 : ui32, handshake.name = "constant4", value = 1 : i32} : <>, <i32>
    %3 = br %2 {handshake.bb = 0 : ui32, handshake.name = "br2"} : <i32>
    %4 = br %1 {handshake.bb = 0 : ui32, handshake.name = "br4"} : <i32>
    %5 = br %arg4 {handshake.bb = 0 : ui32, handshake.name = "br5"} : <>
    %6 = mux %index [%3, %trueResult_36] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i32>, <i32>] to <i32>
    %7 = mux %index [%4, %trueResult_38] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i1>, [<i32>, <i32>] to <i32>
    %result, %index = control_merge [%5, %trueResult_40]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %8 = source {handshake.bb = 1 : ui32, handshake.name = "source0"} : <>
    %9 = constant %8 {handshake.bb = 1 : ui32, handshake.name = "constant7", value = 1 : i32} : <>, <i32>
    %10 = addi %6, %9 {handshake.bb = 1 : ui32, handshake.name = "addi2"} : <i32>
    %11 = br %10 {handshake.bb = 1 : ui32, handshake.name = "br6"} : <i32>
    %12 = br %7 {handshake.bb = 1 : ui32, handshake.name = "br7"} : <i32>
    %13 = br %6 {handshake.bb = 1 : ui32, handshake.name = "br8"} : <i32>
    %14 = br %result {handshake.bb = 1 : ui32, handshake.name = "br9"} : <>
    %15 = mux %index_1 [%11, %63] {handshake.bb = 2 : ui32, handshake.name = "mux2"} : <i1>, [<i32>, <i32>] to <i32>
    %16 = mux %index_1 [%12, %64] {handshake.bb = 2 : ui32, handshake.name = "mux3"} : <i1>, [<i32>, <i32>] to <i32>
    %17 = mux %index_1 [%13, %65] {handshake.bb = 2 : ui32, handshake.name = "mux4"} : <i1>, [<i32>, <i32>] to <i32>
    %result_0, %index_1 = control_merge [%14, %66]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>, <>] to <>, <i1>
    %18 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %19 = constant %18 {handshake.bb = 2 : ui32, handshake.name = "constant8", value = 19 : i32} : <>, <i32>
    %20 = constant %result_0 {handshake.bb = 2 : ui32, handshake.name = "constant9", value = 1 : i32} : <>, <i32>
    %21 = constant %result_0 {handshake.bb = 2 : ui32, handshake.name = "constant10", value = 1 : i32} : <>, <i32>
    %22 = cmpi ult, %15, %19 {handshake.bb = 2 : ui32, handshake.name = "cmpi2"} : <i32>
    %trueResult, %falseResult = cond_br %22, %20 {handshake.bb = 2 : ui32, handshake.name = "cond_br3"} : <i1>, <i32>
    %trueResult_2, %falseResult_3 = cond_br %22, %21 {handshake.bb = 2 : ui32, handshake.name = "cond_br4"} : <i1>, <i32>
    %trueResult_4, %falseResult_5 = cond_br %22, %16 {handshake.bb = 2 : ui32, handshake.name = "cond_br5"} : <i1>, <i32>
    %trueResult_6, %falseResult_7 = cond_br %22, %17 {handshake.bb = 2 : ui32, handshake.name = "cond_br6"} : <i1>, <i32>
    %trueResult_8, %falseResult_9 = cond_br %22, %15 {handshake.bb = 2 : ui32, handshake.name = "cond_br7"} : <i1>, <i32>
    %trueResult_10, %falseResult_11 = cond_br %22, %result_0 {handshake.bb = 2 : ui32, handshake.name = "cond_br8"} : <i1>, <>
    %23 = mux %index_13 [%trueResult, %trueResult_20] {handshake.bb = 3 : ui32, handshake.name = "mux5"} : <i1>, [<i32>, <i32>] to <i32>
    %24 = mux %index_13 [%trueResult_2, %trueResult_22] {handshake.bb = 3 : ui32, handshake.name = "mux6"} : <i1>, [<i32>, <i32>] to <i32>
    %25 = mux %index_13 [%trueResult_4, %trueResult_24] {handshake.bb = 3 : ui32, handshake.name = "mux7"} : <i1>, [<i32>, <i32>] to <i32>
    %26 = mux %index_13 [%trueResult_6, %trueResult_26] {handshake.bb = 3 : ui32, handshake.name = "mux8"} : <i1>, [<i32>, <i32>] to <i32>
    %27 = mux %index_13 [%trueResult_8, %trueResult_28] {handshake.bb = 3 : ui32, handshake.name = "mux9"} : <i1>, [<i32>, <i32>] to <i32>
    %result_12, %index_13 = control_merge [%trueResult_10, %trueResult_30]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>, <>] to <>, <i1>
    %28 = source {handshake.bb = 3 : ui32, handshake.name = "source2"} : <>
    %29 = constant %28 {handshake.bb = 3 : ui32, handshake.name = "constant11", value = 20 : i32} : <>, <i32>
    %30 = source {handshake.bb = 3 : ui32, handshake.name = "source3"} : <>
    %31 = constant %30 {handshake.bb = 3 : ui32, handshake.name = "constant12", value = 1 : i32} : <>, <i32>
    %32 = source {handshake.bb = 3 : ui32, handshake.name = "source4"} : <>
    %33 = constant %32 {handshake.bb = 3 : ui32, handshake.name = "constant13", value = 1 : i32} : <>, <i32>
    %34 = source {handshake.bb = 3 : ui32, handshake.name = "source5"} : <>
    %35 = constant %34 {handshake.bb = 3 : ui32, handshake.name = "constant14", value = 4 : i32} : <>, <i32>
    %36 = source {handshake.bb = 3 : ui32, handshake.name = "source6"} : <>
    %37 = constant %36 {handshake.bb = 3 : ui32, handshake.name = "constant15", value = 2 : i32} : <>, <i32>
    %38 = trunci %24 {handshake.bb = 3 : ui32, handshake.name = "index_cast0"} : <i32> to <i32>
    %39 = shli %27, %37 {handshake.bb = 3 : ui32, handshake.name = "shli0"} : <i32>
    %40 = shli %27, %35 {handshake.bb = 3 : ui32, handshake.name = "shli1"} : <i32>
    %41 = addi %39, %40 {handshake.bb = 3 : ui32, handshake.name = "addi9"} : <i32>
    %42 = addi %38, %41 {handshake.bb = 3 : ui32, handshake.name = "addi3"} : <i32>
    %addressResult, %dataResult = load[%42] %0#0 {handshake.bb = 3 : ui32, handshake.deps = #handshake<deps[["store0", 0]]>, handshake.name = "load0"} : <i32>, <i32>, <i32>, <i32>
    %addressResult_14, %dataResult_15 = load[%26] %outputs {handshake.bb = 3 : ui32, handshake.name = "load1"} : <i32>, <i32>, <i32>, <i32>
    %43 = shli %26, %37 {handshake.bb = 3 : ui32, handshake.name = "shli2"} : <i32>
    %44 = shli %26, %35 {handshake.bb = 3 : ui32, handshake.name = "shli3"} : <i32>
    %45 = addi %43, %44 {handshake.bb = 3 : ui32, handshake.name = "addi10"} : <i32>
    %46 = addi %38, %45 {handshake.bb = 3 : ui32, handshake.name = "addi4"} : <i32>
    %addressResult_16, %dataResult_17 = load[%46] %0#1 {handshake.bb = 3 : ui32, handshake.deps = #handshake<deps[["store0", 0]]>, handshake.name = "load2"} : <i32>, <i32>, <i32>, <i32>
    %47 = muli %dataResult_15, %dataResult_17 {handshake.bb = 3 : ui32, handshake.name = "muli0"} : <i32>
    %48 = subi %dataResult, %47 {handshake.bb = 3 : ui32, handshake.name = "subi0"} : <i32>
    %49 = shli %27, %37 {handshake.bb = 3 : ui32, handshake.name = "shli4"} : <i32>
    %50 = shli %27, %35 {handshake.bb = 3 : ui32, handshake.name = "shli5"} : <i32>
    %51 = addi %49, %50 {handshake.bb = 3 : ui32, handshake.name = "addi11"} : <i32>
    %52 = addi %38, %51 {handshake.bb = 3 : ui32, handshake.name = "addi5"} : <i32>
    %addressResult_18, %dataResult_19 = store[%52] %48 {handshake.bb = 3 : ui32, handshake.deps = #handshake<deps[["load0", 0], ["load2", 0], ["store0", 0]]>, handshake.name = "store0"} : <i32>, <i32>, <i32>, <i32>
    %53 = addi %25, %24 {handshake.bb = 3 : ui32, handshake.name = "addi0"} : <i32>
    %54 = addi %24, %33 {handshake.bb = 3 : ui32, handshake.name = "addi1"} : <i32>
    %55 = addi %23, %31 {handshake.bb = 3 : ui32, handshake.name = "addi6"} : <i32>
    %56 = cmpi ult, %55, %29 {handshake.bb = 3 : ui32, handshake.name = "cmpi0"} : <i32>
    %trueResult_20, %falseResult_21 = cond_br %56, %55 {handshake.bb = 3 : ui32, handshake.name = "cond_br9"} : <i1>, <i32>
    %trueResult_22, %falseResult_23 = cond_br %56, %54 {handshake.bb = 3 : ui32, handshake.name = "cond_br10"} : <i1>, <i32>
    %trueResult_24, %falseResult_25 = cond_br %56, %53 {handshake.bb = 3 : ui32, handshake.name = "cond_br11"} : <i1>, <i32>
    %trueResult_26, %falseResult_27 = cond_br %56, %26 {handshake.bb = 3 : ui32, handshake.name = "cond_br12"} : <i1>, <i32>
    %trueResult_28, %falseResult_29 = cond_br %56, %27 {handshake.bb = 3 : ui32, handshake.name = "cond_br13"} : <i1>, <i32>
    %trueResult_30, %falseResult_31 = cond_br %56, %result_12 {handshake.bb = 3 : ui32, handshake.name = "cond_br14"} : <i1>, <>
    %57 = merge %falseResult_27 {handshake.bb = 4 : ui32, handshake.name = "merge0"} : <i32>
    %58 = merge %falseResult_29 {handshake.bb = 4 : ui32, handshake.name = "merge1"} : <i32>
    %59 = merge %falseResult_25 {handshake.bb = 4 : ui32, handshake.name = "merge2"} : <i32>
    %result_32, %index_33 = control_merge [%falseResult_31]  {handshake.bb = 4 : ui32, handshake.name = "control_merge3"} : [<>] to <>, <i1>
    %60 = source {handshake.bb = 4 : ui32, handshake.name = "source7"} : <>
    %61 = constant %60 {handshake.bb = 4 : ui32, handshake.name = "constant16", value = 1 : i32} : <>, <i32>
    %62 = addi %58, %61 {handshake.bb = 4 : ui32, handshake.name = "addi8"} : <i32>
    %63 = br %62 {handshake.bb = 4 : ui32, handshake.name = "br10"} : <i32>
    %64 = br %59 {handshake.bb = 4 : ui32, handshake.name = "br11"} : <i32>
    %65 = br %57 {handshake.bb = 4 : ui32, handshake.name = "br12"} : <i32>
    %66 = br %result_32 {handshake.bb = 4 : ui32, handshake.name = "br13"} : <>
    %67 = merge %falseResult_7 {handshake.bb = 5 : ui32, handshake.name = "merge3"} : <i32>
    %68 = merge %falseResult_5 {handshake.bb = 5 : ui32, handshake.name = "merge4"} : <i32>
    %result_34, %index_35 = control_merge [%falseResult_11]  {handshake.bb = 5 : ui32, handshake.name = "control_merge4"} : [<>] to <>, <i1>
    %69 = source {handshake.bb = 5 : ui32, handshake.name = "source8"} : <>
    %70 = constant %69 {handshake.bb = 5 : ui32, handshake.name = "constant17", value = 19 : i32} : <>, <i32>
    %71 = source {handshake.bb = 5 : ui32, handshake.name = "source9"} : <>
    %72 = constant %71 {handshake.bb = 5 : ui32, handshake.name = "constant18", value = 1 : i32} : <>, <i32>
    %73 = addi %67, %72 {handshake.bb = 5 : ui32, handshake.name = "addi7"} : <i32>
    %74 = cmpi ult, %73, %70 {handshake.bb = 5 : ui32, handshake.name = "cmpi1"} : <i32>
    %trueResult_36, %falseResult_37 = cond_br %74, %73 {handshake.bb = 5 : ui32, handshake.name = "cond_br15"} : <i1>, <i32>
    %trueResult_38, %falseResult_39 = cond_br %74, %68 {handshake.bb = 5 : ui32, handshake.name = "cond_br16"} : <i1>, <i32>
    %trueResult_40, %falseResult_41 = cond_br %74, %result_34 {handshake.bb = 5 : ui32, handshake.name = "cond_br17"} : <i1>, <>
    %75 = merge %falseResult_39 {handshake.bb = 6 : ui32, handshake.name = "merge5"} : <i32>
    %result_42, %index_43 = control_merge [%falseResult_41]  {handshake.bb = 6 : ui32, handshake.name = "control_merge5"} : [<>] to <>, <i1>
    end {handshake.bb = 6 : ui32, handshake.name = "end0"} %75, %memEnd, %0#2, %arg4 : <i32>, <>, <>, <>
  }
}

