module {
  handshake.func @multiple_exit(%arg0: memref<10xi32>, %arg1: !handshake.control<>, %arg2: !handshake.control<>, ...) -> (!handshake.channel<i32>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["arr", "arr_start", "start"], resNames = ["out0", "arr_end", "end"]} {
    %0:4 = lsq[%arg0 : memref<10xi32>] (%arg1, %result_6, %addressResult, %result_20, %addressResult_22, %result_36, %addressResult_38, %addressResult_40, %dataResult_41, %result_42)  {groupSizes = [1 : i32, 1 : i32, 2 : i32], handshake.name = "lsq1"} : (!handshake.control<>, !handshake.control<>, !handshake.channel<i4>, !handshake.control<>, !handshake.channel<i4>, !handshake.control<>, !handshake.channel<i4>, !handshake.channel<i4>, !handshake.channel<i32>, !handshake.control<>) -> (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>)
    %1 = constant %arg2 {handshake.name = "constant1", value = false} : <>, <i1>
    %2 = constant %arg2 {handshake.bb = 0 : ui32, handshake.name = "constant6", value = true} : <>, <i1>
    %3 = br %1 {handshake.bb = 0 : ui32, handshake.name = "br1"} : <i1>
    %4 = extsi %3 {handshake.bb = 0 : ui32, handshake.name = "extsi8"} : <i1> to <i32>
    %5 = br %2 {handshake.bb = 0 : ui32, handshake.name = "br2"} : <i1>
    %6 = br %1 {handshake.bb = 0 : ui32, handshake.name = "br3"} : <i1>
    %7 = extsi %6 {handshake.bb = 0 : ui32, handshake.name = "extsi7"} : <i1> to <i2>
    %8 = br %arg2 {handshake.bb = 0 : ui32, handshake.name = "br4"} : <>
    %9 = mux %index [%4, %falseResult_13, %falseResult_25, %50] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i2>, [<i32>, <i32>, <i32>, <i32>] to <i32>
    %10 = mux %index [%5, %falseResult_9, %falseResult_29, %51] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i2>, [<i1>, <i1>, <i1>, <i1>] to <i1>
    %11 = mux %index [%7, %falseResult_11, %falseResult_31, %52] {handshake.bb = 1 : ui32, handshake.name = "mux2"} : <i2>, [<i2>, <i2>, <i2>, <i2>] to <i2>
    %12 = mux %index [%5, %falseResult_19, %falseResult_35, %53] {handshake.bb = 1 : ui32, handshake.name = "mux3"} : <i2>, [<i1>, <i1>, <i1>, <i1>] to <i1>
    %result, %index = control_merge [%8, %falseResult_17, %falseResult_33, %54]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>, <>, <>] to <>, <i2>
    %13 = source {handshake.bb = 1 : ui32, handshake.name = "source0"} : <>
    %14 = constant %13 {handshake.name = "constant2", value = 10 : i5} : <>, <i5>
    %15 = extsi %14 {handshake.name = "extsi2"} : <i5> to <i32>
    %16 = cmpi slt, %9, %15 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i32>
    %17 = andi %16, %12 {handshake.bb = 1 : ui32, handshake.name = "andi0"} : <i1>
    %trueResult, %falseResult = cond_br %17, %10 {handshake.bb = 1 : ui32, handshake.name = "cond_br3"} : <i1>, <i1>
    %trueResult_0, %falseResult_1 = cond_br %17, %11 {handshake.bb = 1 : ui32, handshake.name = "cond_br4"} : <i1>, <i2>
    %trueResult_2, %falseResult_3 = cond_br %17, %9 {handshake.bb = 1 : ui32, handshake.name = "cond_br5"} : <i1>, <i32>
    %trueResult_4, %falseResult_5 = cond_br %17, %result {handshake.bb = 1 : ui32, handshake.name = "cond_br6"} : <i1>, <>
    %18 = merge %trueResult {handshake.bb = 2 : ui32, handshake.name = "merge0"} : <i1>
    %19 = merge %trueResult_0 {handshake.bb = 2 : ui32, handshake.name = "merge1"} : <i2>
    %20 = merge %trueResult_2 {handshake.bb = 2 : ui32, handshake.name = "merge2"} : <i32>
    %21 = trunci %20 {handshake.bb = 2 : ui32, handshake.name = "trunci0"} : <i32> to <i4>
    %22 = trunci %20 {handshake.bb = 2 : ui32, handshake.name = "trunci1"} : <i32> to <i4>
    %result_6, %index_7 = control_merge [%trueResult_4]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>] to <>, <i1>
    %23 = constant %result_6 {handshake.bb = 2 : ui32, handshake.name = "constant8", value = false} : <>, <i1>
    %24 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %25 = constant %24 {handshake.bb = 2 : ui32, handshake.name = "constant9", value = -1 : i32} : <>, <i32>
    %addressResult, %dataResult = load[%22] %0#0 {handshake.bb = 2 : ui32, handshake.deps = #handshake<deps[<"store1" (0)>]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 0>, handshake.name = "load3"} : <i4>, <i32>, <i4>, <i32>
    %26 = cmpi ne, %dataResult, %25 {handshake.bb = 2 : ui32, handshake.name = "cmpi1"} : <i32>
    %trueResult_8, %falseResult_9 = cond_br %26, %18 {handshake.bb = 2 : ui32, handshake.name = "cond_br7"} : <i1>, <i1>
    %trueResult_10, %falseResult_11 = cond_br %26, %19 {handshake.bb = 2 : ui32, handshake.name = "cond_br8"} : <i1>, <i2>
    %trueResult_12, %falseResult_13 = cond_br %26, %20 {handshake.bb = 2 : ui32, handshake.name = "cond_br9"} : <i1>, <i32>
    %trueResult_14, %falseResult_15 = cond_br %26, %21 {handshake.bb = 2 : ui32, handshake.name = "cond_br10"} : <i1>, <i4>
    %trueResult_16, %falseResult_17 = cond_br %26, %result_6 {handshake.bb = 2 : ui32, handshake.name = "cond_br11"} : <i1>, <>
    %trueResult_18, %falseResult_19 = cond_br %26, %23 {handshake.bb = 2 : ui32, handshake.name = "cond_br12"} : <i1>, <i1>
    %27 = merge %trueResult_8 {handshake.bb = 3 : ui32, handshake.name = "merge3"} : <i1>
    %28 = merge %trueResult_10 {handshake.bb = 3 : ui32, handshake.name = "merge4"} : <i2>
    %29 = merge %trueResult_12 {handshake.bb = 3 : ui32, handshake.name = "merge5"} : <i32>
    %30 = merge %trueResult_14 {handshake.bb = 3 : ui32, handshake.name = "merge6"} : <i4>
    %result_20, %index_21 = control_merge [%trueResult_16]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>] to <>, <i1>
    %31 = source {handshake.bb = 3 : ui32, handshake.name = "source2"} : <>
    %32 = constant %31 {handshake.name = "constant3", value = 1 : i2} : <>, <i2>
    %33 = source {handshake.bb = 3 : ui32, handshake.name = "source3"} : <>
    %34 = constant %33 {handshake.name = "constant4", value = false} : <>, <i1>
    %35 = extsi %34 {handshake.name = "extsi4"} : <i1> to <i32>
    %addressResult_22, %dataResult_23 = load[%30] %0#1 {handshake.bb = 3 : ui32, handshake.deps = #handshake<deps[<"store1" (0)>]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 1>, handshake.name = "load4"} : <i4>, <i32>, <i4>, <i32>
    %36 = cmpi eq, %dataResult_23, %35 {handshake.bb = 3 : ui32, handshake.name = "cmpi2"} : <i32>
    %37 = cmpi ne, %dataResult_23, %35 {handshake.bb = 3 : ui32, handshake.name = "cmpi3"} : <i32>
    %38 = andi %37, %27 {handshake.bb = 3 : ui32, handshake.name = "andi1"} : <i1>
    %39 = select %36[%32, %28] {handshake.bb = 3 : ui32, handshake.name = "select0"} : <i1>, <i2>
    %trueResult_24, %falseResult_25 = cond_br %37, %29 {handshake.bb = 3 : ui32, handshake.name = "cond_br13"} : <i1>, <i32>
    %trueResult_26, %falseResult_27 = cond_br %37, %30 {handshake.bb = 3 : ui32, handshake.name = "cond_br14"} : <i1>, <i4>
    %trueResult_28, %falseResult_29 = cond_br %37, %38 {handshake.bb = 3 : ui32, handshake.name = "cond_br15"} : <i1>, <i1>
    %trueResult_30, %falseResult_31 = cond_br %37, %39 {handshake.bb = 3 : ui32, handshake.name = "cond_br16"} : <i1>, <i2>
    %trueResult_32, %falseResult_33 = cond_br %37, %result_20 {handshake.bb = 3 : ui32, handshake.name = "cond_br17"} : <i1>, <>
    %trueResult_34, %falseResult_35 = cond_br %37, %37 {handshake.bb = 3 : ui32, handshake.name = "cond_br18"} : <i1>, <i1>
    %40 = merge %trueResult_24 {handshake.bb = 4 : ui32, handshake.name = "merge7"} : <i32>
    %41 = merge %trueResult_26 {handshake.bb = 4 : ui32, handshake.name = "merge8"} : <i4>
    %42 = merge %trueResult_28 {handshake.bb = 4 : ui32, handshake.name = "merge9"} : <i1>
    %43 = merge %trueResult_30 {handshake.bb = 4 : ui32, handshake.name = "merge10"} : <i2>
    %result_36, %index_37 = control_merge [%trueResult_32]  {handshake.bb = 4 : ui32, handshake.name = "control_merge3"} : [<>] to <>, <i1>
    %44 = constant %result_36 {handshake.bb = 4 : ui32, handshake.name = "constant12", value = true} : <>, <i1>
    %45 = source {handshake.bb = 4 : ui32, handshake.name = "source4"} : <>
    %46 = constant %45 {handshake.name = "constant5", value = 1 : i2} : <>, <i2>
    %47 = extsi %46 {handshake.name = "extsi5"} : <i2> to <i32>
    %addressResult_38, %dataResult_39 = load[%41] %0#2 {handshake.bb = 4 : ui32, handshake.deps = #handshake<deps[<"store1" (0)>]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 2>, handshake.name = "load5"} : <i4>, <i32>, <i4>, <i32>
    %48 = addi %dataResult_39, %47 {handshake.bb = 4 : ui32, handshake.name = "addi0"} : <i32>
    %addressResult_40, %dataResult_41 = store[%41] %48 {handshake.bb = 4 : ui32, handshake.deps = #handshake<deps[<"load3" (0)>, <"load4" (0)>, <"load5" (0)>, <"store1" (0)>]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 2>, handshake.name = "store1"} : <i4>, <i32>, <i4>, <i32>
    %49 = addi %40, %47 {handshake.bb = 4 : ui32, handshake.name = "addi1"} : <i32>
    %50 = br %49 {handshake.bb = 4 : ui32, handshake.name = "br5"} : <i32>
    %51 = br %42 {handshake.bb = 4 : ui32, handshake.name = "br6"} : <i1>
    %52 = br %43 {handshake.bb = 4 : ui32, handshake.name = "br8"} : <i2>
    %53 = br %44 {handshake.bb = 4 : ui32, handshake.name = "br9"} : <i1>
    %54 = br %result_36 {handshake.bb = 4 : ui32, handshake.name = "br10"} : <>
    %55 = merge %falseResult {handshake.bb = 5 : ui32, handshake.name = "merge11"} : <i1>
    %56 = merge %falseResult_1 {handshake.bb = 5 : ui32, handshake.name = "merge12"} : <i2>
    %57 = extsi %56 {handshake.bb = 5 : ui32, handshake.name = "extsi9"} : <i2> to <i3>
    %result_42, %index_43 = control_merge [%falseResult_5]  {handshake.bb = 5 : ui32, handshake.name = "control_merge4"} : [<>] to <>, <i1>
    %58 = source {handshake.bb = 5 : ui32, handshake.name = "source5"} : <>
    %59 = constant %58 {handshake.name = "constant15", value = 2 : i3} : <>, <i3>
    %60 = select %55[%59, %57] {handshake.bb = 5 : ui32, handshake.name = "select1"} : <i1>, <i3>
    %61 = extsi %60 {handshake.name = "extsi10"} : <i3> to <i32>
    end {handshake.bb = 5 : ui32, handshake.name = "end0"} %61, %0#3, %arg2 : <i32>, <>, <>
  }
}

