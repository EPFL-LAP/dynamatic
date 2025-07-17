module {
  handshake.func @multiple_exit(%arg0: memref<10xi32>, %arg1: !handshake.control<>, %arg2: !handshake.control<>, ...) -> (!handshake.channel<i32>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["arr", "arr_start", "start"], resNames = ["out0", "arr_end", "end"]} {
    %0:4 = lsq[%arg0 : memref<10xi32>] (%arg1, %result_6, %addressResult, %result_20, %addressResult_22, %result_36, %addressResult_38, %addressResult_40, %dataResult_41, %result_42)  {groupSizes = [1 : i32, 1 : i32, 2 : i32], handshake.name = "lsq0"} : (!handshake.control<>, !handshake.control<>, !handshake.channel<i32>, !handshake.control<>, !handshake.channel<i32>, !handshake.control<>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>) -> (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>)
    %1 = constant %arg2 {handshake.bb = 0 : ui32, handshake.name = "constant0", value = 0 : i32} : <>, <i32>
    %2 = constant %arg2 {handshake.bb = 0 : ui32, handshake.name = "constant6", value = true} : <>, <i1>
    %3 = constant %arg2 {handshake.bb = 0 : ui32, handshake.name = "mlir.undef0", value = 0 : i32} : <>, <i32>
    %4 = br %1 {handshake.bb = 0 : ui32, handshake.name = "br1"} : <i32>
    %5 = br %2 {handshake.bb = 0 : ui32, handshake.name = "br2"} : <i1>
    %6 = br %3 {handshake.bb = 0 : ui32, handshake.name = "br3"} : <i32>
    %7 = br %arg2 {handshake.bb = 0 : ui32, handshake.name = "br4"} : <>
    %8 = mux %index [%4, %falseResult_13, %falseResult_25, %45] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i2>, [<i32>, <i32>, <i32>, <i32>] to <i32>
    %9 = mux %index [%5, %falseResult_9, %falseResult_29, %46] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i2>, [<i1>, <i1>, <i1>, <i1>] to <i1>
    %10 = mux %index [%6, %falseResult_11, %falseResult_31, %47] {handshake.bb = 1 : ui32, handshake.name = "mux2"} : <i2>, [<i32>, <i32>, <i32>, <i32>] to <i32>
    %11 = mux %index [%5, %falseResult_19, %falseResult_35, %48] {handshake.bb = 1 : ui32, handshake.name = "mux3"} : <i2>, [<i1>, <i1>, <i1>, <i1>] to <i1>
    %result, %index = control_merge [%7, %falseResult_17, %falseResult_33, %49]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>, <>, <>] to <>, <i2>
    %12 = source {handshake.bb = 1 : ui32, handshake.name = "source0"} : <>
    %13 = constant %12 {handshake.bb = 1 : ui32, handshake.name = "constant7", value = 10 : i32} : <>, <i32>
    %14 = cmpi slt, %8, %13 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i32>
    %15 = andi %14, %11 {handshake.bb = 1 : ui32, handshake.name = "andi0"} : <i1>
    %trueResult, %falseResult = cond_br %15, %9 {handshake.bb = 1 : ui32, handshake.name = "cond_br3"} : <i1>, <i1>
    %trueResult_0, %falseResult_1 = cond_br %15, %10 {handshake.bb = 1 : ui32, handshake.name = "cond_br4"} : <i1>, <i32>
    %trueResult_2, %falseResult_3 = cond_br %15, %8 {handshake.bb = 1 : ui32, handshake.name = "cond_br5"} : <i1>, <i32>
    %trueResult_4, %falseResult_5 = cond_br %15, %result {handshake.bb = 1 : ui32, handshake.name = "cond_br6"} : <i1>, <>
    %16 = merge %trueResult {handshake.bb = 2 : ui32, handshake.name = "merge0"} : <i1>
    %17 = merge %trueResult_0 {handshake.bb = 2 : ui32, handshake.name = "merge1"} : <i32>
    %18 = merge %trueResult_2 {handshake.bb = 2 : ui32, handshake.name = "merge2"} : <i32>
    %result_6, %index_7 = control_merge [%trueResult_4]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>] to <>, <i1>
    %19 = constant %result_6 {handshake.bb = 2 : ui32, handshake.name = "constant8", value = false} : <>, <i1>
    %20 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %21 = constant %20 {handshake.bb = 2 : ui32, handshake.name = "constant9", value = -1 : i32} : <>, <i32>
    %22 = trunci %18 {handshake.bb = 2 : ui32, handshake.name = "index_cast0"} : <i32> to <i32>
    %addressResult, %dataResult = load[%22] %0#0 {handshake.bb = 2 : ui32, handshake.deps = #handshake<deps[<"store1" (0)>]>, handshake.name = "load3"} : <i32>, <i32>, <i32>, <i32>
    %23 = cmpi ne, %dataResult, %21 {handshake.bb = 2 : ui32, handshake.name = "cmpi1"} : <i32>
    %trueResult_8, %falseResult_9 = cond_br %23, %16 {handshake.bb = 2 : ui32, handshake.name = "cond_br7"} : <i1>, <i1>
    %trueResult_10, %falseResult_11 = cond_br %23, %17 {handshake.bb = 2 : ui32, handshake.name = "cond_br8"} : <i1>, <i32>
    %trueResult_12, %falseResult_13 = cond_br %23, %18 {handshake.bb = 2 : ui32, handshake.name = "cond_br9"} : <i1>, <i32>
    %trueResult_14, %falseResult_15 = cond_br %23, %22 {handshake.bb = 2 : ui32, handshake.name = "cond_br10"} : <i1>, <i32>
    %trueResult_16, %falseResult_17 = cond_br %23, %result_6 {handshake.bb = 2 : ui32, handshake.name = "cond_br11"} : <i1>, <>
    %trueResult_18, %falseResult_19 = cond_br %23, %19 {handshake.bb = 2 : ui32, handshake.name = "cond_br12"} : <i1>, <i1>
    %24 = merge %trueResult_8 {handshake.bb = 3 : ui32, handshake.name = "merge3"} : <i1>
    %25 = merge %trueResult_10 {handshake.bb = 3 : ui32, handshake.name = "merge4"} : <i32>
    %26 = merge %trueResult_12 {handshake.bb = 3 : ui32, handshake.name = "merge5"} : <i32>
    %27 = merge %trueResult_14 {handshake.bb = 3 : ui32, handshake.name = "merge6"} : <i32>
    %result_20, %index_21 = control_merge [%trueResult_16]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>] to <>, <i1>
    %28 = source {handshake.bb = 3 : ui32, handshake.name = "source2"} : <>
    %29 = constant %28 {handshake.bb = 3 : ui32, handshake.name = "constant10", value = 1 : i32} : <>, <i32>
    %30 = source {handshake.bb = 3 : ui32, handshake.name = "source3"} : <>
    %31 = constant %30 {handshake.bb = 3 : ui32, handshake.name = "constant11", value = 0 : i32} : <>, <i32>
    %addressResult_22, %dataResult_23 = load[%27] %0#1 {handshake.bb = 3 : ui32, handshake.deps = #handshake<deps[<"store1" (0)>]>, handshake.name = "load4"} : <i32>, <i32>, <i32>, <i32>
    %32 = cmpi eq, %dataResult_23, %31 {handshake.bb = 3 : ui32, handshake.name = "cmpi2"} : <i32>
    %33 = cmpi ne, %dataResult_23, %31 {handshake.bb = 3 : ui32, handshake.name = "cmpi3"} : <i32>
    %34 = andi %33, %24 {handshake.bb = 3 : ui32, handshake.name = "andi1"} : <i1>
    %35 = select %32[%29, %25] {handshake.bb = 3 : ui32, handshake.name = "select0"} : <i1>, <i32>
    %trueResult_24, %falseResult_25 = cond_br %33, %26 {handshake.bb = 3 : ui32, handshake.name = "cond_br13"} : <i1>, <i32>
    %trueResult_26, %falseResult_27 = cond_br %33, %27 {handshake.bb = 3 : ui32, handshake.name = "cond_br14"} : <i1>, <i32>
    %trueResult_28, %falseResult_29 = cond_br %33, %34 {handshake.bb = 3 : ui32, handshake.name = "cond_br15"} : <i1>, <i1>
    %trueResult_30, %falseResult_31 = cond_br %33, %35 {handshake.bb = 3 : ui32, handshake.name = "cond_br16"} : <i1>, <i32>
    %trueResult_32, %falseResult_33 = cond_br %33, %result_20 {handshake.bb = 3 : ui32, handshake.name = "cond_br17"} : <i1>, <>
    %trueResult_34, %falseResult_35 = cond_br %33, %33 {handshake.bb = 3 : ui32, handshake.name = "cond_br18"} : <i1>, <i1>
    %36 = merge %trueResult_24 {handshake.bb = 4 : ui32, handshake.name = "merge7"} : <i32>
    %37 = merge %trueResult_26 {handshake.bb = 4 : ui32, handshake.name = "merge8"} : <i32>
    %38 = merge %trueResult_28 {handshake.bb = 4 : ui32, handshake.name = "merge9"} : <i1>
    %39 = merge %trueResult_30 {handshake.bb = 4 : ui32, handshake.name = "merge10"} : <i32>
    %result_36, %index_37 = control_merge [%trueResult_32]  {handshake.bb = 4 : ui32, handshake.name = "control_merge3"} : [<>] to <>, <i1>
    %40 = constant %result_36 {handshake.bb = 4 : ui32, handshake.name = "constant12", value = true} : <>, <i1>
    %41 = source {handshake.bb = 4 : ui32, handshake.name = "source4"} : <>
    %42 = constant %41 {handshake.bb = 4 : ui32, handshake.name = "constant13", value = 1 : i32} : <>, <i32>
    %addressResult_38, %dataResult_39 = load[%37] %0#2 {handshake.bb = 4 : ui32, handshake.deps = #handshake<deps[<"store1" (0)>]>, handshake.name = "load5"} : <i32>, <i32>, <i32>, <i32>
    %43 = addi %dataResult_39, %42 {handshake.bb = 4 : ui32, handshake.name = "addi0"} : <i32>
    %addressResult_40, %dataResult_41 = store[%37] %43 {handshake.bb = 4 : ui32, handshake.deps = #handshake<deps[<"load3" (0)>, <"load4" (0)>, <"load5" (0)>, <"store1" (0)>]>, handshake.name = "store1"} : <i32>, <i32>, <i32>, <i32>
    %44 = addi %36, %42 {handshake.bb = 4 : ui32, handshake.name = "addi1"} : <i32>
    %45 = br %44 {handshake.bb = 4 : ui32, handshake.name = "br5"} : <i32>
    %46 = br %38 {handshake.bb = 4 : ui32, handshake.name = "br6"} : <i1>
    %47 = br %39 {handshake.bb = 4 : ui32, handshake.name = "br8"} : <i32>
    %48 = br %40 {handshake.bb = 4 : ui32, handshake.name = "br9"} : <i1>
    %49 = br %result_36 {handshake.bb = 4 : ui32, handshake.name = "br10"} : <>
    %50 = merge %falseResult {handshake.bb = 5 : ui32, handshake.name = "merge11"} : <i1>
    %51 = merge %falseResult_1 {handshake.bb = 5 : ui32, handshake.name = "merge12"} : <i32>
    %result_42, %index_43 = control_merge [%falseResult_5]  {handshake.bb = 5 : ui32, handshake.name = "control_merge4"} : [<>] to <>, <i1>
    %52 = source {handshake.bb = 5 : ui32, handshake.name = "source5"} : <>
    %53 = constant %52 {handshake.bb = 5 : ui32, handshake.name = "constant14", value = 2 : i32} : <>, <i32>
    %54 = select %50[%53, %51] {handshake.bb = 5 : ui32, handshake.name = "select1"} : <i1>, <i32>
    end {handshake.bb = 5 : ui32, handshake.name = "end0"} %54, %0#3, %arg2 : <i32>, <>, <>
  }
}

