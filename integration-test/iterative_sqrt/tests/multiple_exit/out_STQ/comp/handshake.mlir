module {
  handshake.func @multiple_exit(%arg0: memref<10xi32>, %arg1: !handshake.control<>, %arg2: !handshake.control<>, ...) -> (!handshake.channel<i32>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["arr", "arr_start", "start"], cfg.edges = "[0,1][2,3,1,cmpi1][4,1][1,2,5,andi0][3,4,1,cmpi3]", resNames = ["out0", "arr_end", "end"]} {
    %0:4 = lsq[%arg0 : memref<10xi32>] (%arg1, %result_8, %addressResult, %result_16, %addressResult_18, %result_24, %addressResult_26, %addressResult_28, %dataResult_29, %result_30)  {groupSizes = [1 : i32, 1 : i32, 2 : i32], handshake.name = "lsq0"} : (!handshake.control<>, !handshake.control<>, !handshake.channel<i32>, !handshake.control<>, !handshake.channel<i32>, !handshake.control<>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>) -> (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>)
    %1 = source {handshake.bb = 0 : ui32, handshake.name = "source0"} : <>
    %2 = constant %1 {handshake.bb = 0 : ui32, handshake.name = "constant0", value = 0 : i32} : <>, <i32>
    %3 = source {handshake.bb = 0 : ui32, handshake.name = "source1"} : <>
    %4 = constant %3 {handshake.bb = 0 : ui32, handshake.name = "constant6", value = true} : <>, <i1>
    %5 = constant %arg2 {handshake.bb = 0 : ui32, handshake.name = "mlir.undef0", value = 0 : i32} : <>, <i32>
    %6 = br %arg2 {handshake.bb = 0 : ui32, handshake.name = "br1"} : <>
    %trueResult, %falseResult = cond_br %22, %11 {handshake.bb = 1 : ui32, handshake.name = "cond_br25"} : <i1>, <i32>
    %trueResult_0, %falseResult_1 = cond_br %22, %10 {handshake.bb = 1 : ui32, handshake.name = "cond_br26"} : <i1>, <i1>
    %trueResult_2, %falseResult_3 = cond_br %22, %9 {handshake.bb = 1 : ui32, handshake.name = "cond_br27"} : <i1>, <i32>
    %trueResult_4, %falseResult_5 = cond_br %22, %7 {handshake.bb = 1 : ui32, handshake.name = "cond_br28"} : <i1>, <i1>
    %result, %index = control_merge [%6, %falseResult_11, %falseResult_21, %42]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>, <>, <>] to <>, <i2>
    %7 = mux %27 [%24, %8] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i1>, <i1>] to <i1>
    %8 = mux %33 [%33, %37] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i1>, [<i1>, <i1>] to <i1>
    %9 = mux %27 [%16, %35] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux2"} : <i1>, [<i32>, <i32>] to <i32>
    %10 = mux %27 [%17, %34] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux3"} : <i1>, [<i1>, <i1>] to <i1>
    %11 = mux %27 [%18, %12] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux4"} : <i1>, [<i32>, <i32>] to <i32>
    %12 = mux %33 [%18, %41] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux5"} : <i1>, [<i32>, <i32>] to <i32>
    %13 = constant %arg2 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "constant1", value = false} : <>, <i1>
    %14 = mux %15 [%4, %trueResult_4] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux6"} : <i1>, [<i1>, <i1>] to <i1>
    %15 = merge %13, %22 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "merge0"} : <i1>
    %16 = mux %15 [%5, %trueResult_2] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux7"} : <i1>, [<i32>, <i32>] to <i32>
    %17 = mux %15 [%4, %trueResult_0] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux8"} : <i1>, [<i1>, <i1>] to <i1>
    %18 = mux %15 [%2, %trueResult] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux9"} : <i1>, [<i32>, <i32>] to <i32>
    %19 = source {handshake.bb = 1 : ui32, handshake.name = "source2"} : <>
    %20 = constant %19 {handshake.bb = 1 : ui32, handshake.name = "constant7", value = 10 : i32} : <>, <i32>
    %21 = cmpi slt, %18, %20 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i32>
    %22 = andi %21, %14 {handshake.bb = 1 : ui32, handshake.name = "andi0"} : <i1>
    %trueResult_6, %falseResult_7 = cond_br %22, %result {handshake.bb = 1 : ui32, handshake.name = "cond_br9"} : <i1>, <>
    %result_8, %index_9 = control_merge [%trueResult_6]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>] to <>, <i1>
    %23 = source {handshake.bb = 2 : ui32, handshake.name = "source6"} : <>
    %24 = constant %23 {handshake.bb = 2 : ui32, handshake.name = "constant8", value = false} : <>, <i1>
    %25 = source {handshake.bb = 2 : ui32, handshake.name = "source7"} : <>
    %26 = constant %25 {handshake.bb = 2 : ui32, handshake.name = "constant9", value = -1 : i32} : <>, <i32>
    %addressResult, %dataResult = load[%trueResult_22] %0#0 {handshake.bb = 2 : ui32, handshake.deps = #handshake<deps[<"store1" (0)>]>, handshake.name = "load3"} : <i32>, <i32>, <i32>, <i32>
    %27 = cmpi ne, %dataResult, %26 {handshake.bb = 2 : ui32, handshake.name = "cmpi1"} : <i32>
    %trueResult_10, %falseResult_11 = cond_br %27, %result_8 {handshake.bb = 2 : ui32, handshake.name = "cond_br20"} : <i1>, <>
    %trueResult_12, %falseResult_13 = cond_br %22, %16 {handshake.bb = 3 : ui32, handshake.name = "cond_br29"} : <i1>, <i32>
    %trueResult_14, %falseResult_15 = cond_br %22, %17 {handshake.bb = 3 : ui32, handshake.name = "cond_br30"} : <i1>, <i1>
    %result_16, %index_17 = control_merge [%trueResult_10]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>] to <>, <i1>
    %28 = source {handshake.bb = 3 : ui32, handshake.name = "source8"} : <>
    %29 = constant %28 {handshake.bb = 3 : ui32, handshake.name = "constant10", value = 1 : i32} : <>, <i32>
    %30 = source {handshake.bb = 3 : ui32, handshake.name = "source9"} : <>
    %31 = constant %30 {handshake.bb = 3 : ui32, handshake.name = "constant11", value = 0 : i32} : <>, <i32>
    %addressResult_18, %dataResult_19 = load[%trueResult_22] %0#1 {handshake.bb = 3 : ui32, handshake.deps = #handshake<deps[<"store1" (0)>]>, handshake.name = "load4"} : <i32>, <i32>, <i32>, <i32>
    %32 = cmpi eq, %dataResult_19, %31 {handshake.bb = 3 : ui32, handshake.name = "cmpi2"} : <i32>
    %33 = cmpi ne, %dataResult_19, %31 {handshake.bb = 3 : ui32, handshake.name = "cmpi3"} : <i32>
    %34 = andi %33, %trueResult_14 {handshake.bb = 3 : ui32, handshake.name = "andi1"} : <i1>
    %35 = select %32[%29, %trueResult_12] {handshake.bb = 3 : ui32, handshake.name = "select0"} : <i1>, <i32>
    %trueResult_20, %falseResult_21 = cond_br %33, %result_16 {handshake.bb = 3 : ui32, handshake.name = "cond_br23"} : <i1>, <>
    %trueResult_22, %falseResult_23 = cond_br %22, %18 {handshake.bb = 4 : ui32, handshake.name = "cond_br31"} : <i1>, <i32>
    %result_24, %index_25 = control_merge [%trueResult_20]  {handshake.bb = 4 : ui32, handshake.name = "control_merge3"} : [<>] to <>, <i1>
    %36 = source {handshake.bb = 4 : ui32, handshake.name = "source10"} : <>
    %37 = constant %36 {handshake.bb = 4 : ui32, handshake.name = "constant12", value = true} : <>, <i1>
    %38 = source {handshake.bb = 4 : ui32, handshake.name = "source11"} : <>
    %39 = constant %38 {handshake.bb = 4 : ui32, handshake.name = "constant13", value = 1 : i32} : <>, <i32>
    %addressResult_26, %dataResult_27 = load[%trueResult_22] %0#2 {handshake.bb = 4 : ui32, handshake.deps = #handshake<deps[<"store1" (0)>]>, handshake.name = "load5"} : <i32>, <i32>, <i32>, <i32>
    %40 = addi %dataResult_27, %39 {handshake.bb = 4 : ui32, handshake.name = "addi0"} : <i32>
    %addressResult_28, %dataResult_29 = store[%trueResult_22] %40 {handshake.bb = 4 : ui32, handshake.deps = #handshake<deps[<"load3" (0)>, <"load4" (0)>, <"load5" (0)>, <"store1" (0)>]>, handshake.name = "store1"} : <i32>, <i32>, <i32>, <i32>
    %41 = addi %trueResult_22, %39 {handshake.bb = 4 : ui32, handshake.name = "addi1"} : <i32>
    %42 = br %result_24 {handshake.bb = 4 : ui32, handshake.name = "br2"} : <>
    %result_30, %index_31 = control_merge [%falseResult_7]  {handshake.bb = 5 : ui32, handshake.name = "control_merge4"} : [<>] to <>, <i1>
    %43 = source {handshake.bb = 5 : ui32, handshake.name = "source12"} : <>
    %44 = constant %43 {handshake.bb = 5 : ui32, handshake.name = "constant14", value = 2 : i32} : <>, <i32>
    %45 = select %falseResult_15[%44, %falseResult_13] {handshake.bb = 5 : ui32, handshake.name = "select1"} : <i1>, <i32>
    end {handshake.bb = 5 : ui32, handshake.name = "end0"} %45, %0#3, %arg2 : <i32>, <>, <>
  }
}

