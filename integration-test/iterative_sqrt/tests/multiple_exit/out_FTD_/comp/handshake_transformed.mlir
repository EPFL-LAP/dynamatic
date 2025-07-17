module {
  handshake.func @multiple_exit(%arg0: memref<10xi32>, %arg1: !handshake.control<>, %arg2: !handshake.control<>, ...) -> (!handshake.channel<i32>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["arr", "arr_start", "start"], cfg.edges = "[0,1][2,3,1,cmpi1][4,1][1,2,5,andi0][3,4,1,cmpi3]", resNames = ["out0", "arr_end", "end"]} {
    %0:4 = lsq[%arg0 : memref<10xi32>] (%arg1, %result_8, %addressResult, %result_16, %addressResult_18, %result_24, %addressResult_26, %addressResult_28, %dataResult_29, %result_30)  {groupSizes = [1 : i32, 1 : i32, 2 : i32], handshake.name = "lsq1"} : (!handshake.control<>, !handshake.control<>, !handshake.channel<i4>, !handshake.control<>, !handshake.channel<i4>, !handshake.control<>, !handshake.channel<i4>, !handshake.channel<i4>, !handshake.channel<i32>, !handshake.control<>) -> (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>)
    %1 = source {handshake.bb = 0 : ui32, handshake.name = "source0"} : <>
    %2 = constant %1 {handshake.name = "constant2", value = false} : <>, <i1>
    %3 = extsi %2 {handshake.name = "extsi0"} : <i1> to <i32>
    %4 = source {handshake.bb = 0 : ui32, handshake.name = "source1"} : <>
    %5 = constant %4 {handshake.bb = 0 : ui32, handshake.name = "constant6", value = true} : <>, <i1>
    %6 = br %arg2 {handshake.bb = 0 : ui32, handshake.name = "br1"} : <>
    %trueResult, %falseResult = cond_br %24, %11 {handshake.bb = 1 : ui32, handshake.name = "cond_br25"} : <i1>, <i32>
    %trueResult_0, %falseResult_1 = cond_br %24, %10 {handshake.bb = 1 : ui32, handshake.name = "cond_br26"} : <i1>, <i1>
    %trueResult_2, %falseResult_3 = cond_br %24, %9 {handshake.bb = 1 : ui32, handshake.name = "cond_br27"} : <i1>, <i2>
    %trueResult_4, %falseResult_5 = cond_br %24, %7 {handshake.bb = 1 : ui32, handshake.name = "cond_br28"} : <i1>, <i1>
    %result, %index = control_merge [%6, %falseResult_11, %falseResult_21, %51]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>, <>, <>] to <>, <i2>
    %7 = mux %29 [%26, %8] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i1>, <i1>] to <i1>
    %8 = mux %37 [%37, %45] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i1>, [<i1>, <i1>] to <i1>
    %9 = mux %29 [%17, %39] {handshake.bb = 1 : ui32, handshake.name = "mux2"} : <i1>, [<i2>, <i2>] to <i2>
    %10 = mux %29 [%18, %38] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux3"} : <i1>, [<i1>, <i1>] to <i1>
    %11 = mux %29 [%19, %12] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux4"} : <i1>, [<i32>, <i32>] to <i32>
    %12 = mux %37 [%19, %50] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux5"} : <i1>, [<i32>, <i32>] to <i32>
    %13 = constant %arg2 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "constant1", value = false} : <>, <i1>
    %14 = extsi %13 {handshake.bb = 1 : ui32, handshake.name = "extsi7"} : <i1> to <i2>
    %15 = mux %16 [%5, %trueResult_4] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux6"} : <i1>, [<i1>, <i1>] to <i1>
    %16 = merge %13, %24 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "merge0"} : <i1>
    %17 = mux %16 [%14, %trueResult_2] {handshake.bb = 1 : ui32, handshake.name = "mux7"} : <i1>, [<i2>, <i2>] to <i2>
    %18 = mux %16 [%5, %trueResult_0] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux8"} : <i1>, [<i1>, <i1>] to <i1>
    %19 = mux %16 [%3, %trueResult] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux9"} : <i1>, [<i32>, <i32>] to <i32>
    %20 = source {handshake.bb = 1 : ui32, handshake.name = "source2"} : <>
    %21 = constant %20 {handshake.name = "constant3", value = 10 : i5} : <>, <i5>
    %22 = extsi %21 {handshake.name = "extsi2"} : <i5> to <i32>
    %23 = cmpi slt, %19, %22 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i32>
    %24 = andi %23, %15 {handshake.bb = 1 : ui32, handshake.name = "andi0"} : <i1>
    %trueResult_6, %falseResult_7 = cond_br %24, %result {handshake.bb = 1 : ui32, handshake.name = "cond_br9"} : <i1>, <>
    %result_8, %index_9 = control_merge [%trueResult_6]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>] to <>, <i1>
    %25 = source {handshake.bb = 2 : ui32, handshake.name = "source6"} : <>
    %26 = constant %25 {handshake.bb = 2 : ui32, handshake.name = "constant8", value = false} : <>, <i1>
    %27 = source {handshake.bb = 2 : ui32, handshake.name = "source7"} : <>
    %28 = constant %27 {handshake.bb = 2 : ui32, handshake.name = "constant9", value = -1 : i32} : <>, <i32>
    %addressResult, %dataResult = load[%43] %0#0 {handshake.bb = 2 : ui32, handshake.deps = #handshake<deps[<"store1" (0)>]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 0>, handshake.name = "load3"} : <i4>, <i32>, <i4>, <i32>
    %29 = cmpi ne, %dataResult, %28 {handshake.bb = 2 : ui32, handshake.name = "cmpi1"} : <i32>
    %trueResult_10, %falseResult_11 = cond_br %29, %result_8 {handshake.bb = 2 : ui32, handshake.name = "cond_br20"} : <i1>, <>
    %trueResult_12, %falseResult_13 = cond_br %24, %17 {handshake.bb = 3 : ui32, handshake.name = "cond_br29"} : <i1>, <i2>
    %30 = extsi %falseResult_13 {handshake.bb = 3 : ui32, handshake.name = "extsi8"} : <i2> to <i3>
    %trueResult_14, %falseResult_15 = cond_br %24, %18 {handshake.bb = 3 : ui32, handshake.name = "cond_br30"} : <i1>, <i1>
    %result_16, %index_17 = control_merge [%trueResult_10]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>] to <>, <i1>
    %31 = source {handshake.bb = 3 : ui32, handshake.name = "source8"} : <>
    %32 = constant %31 {handshake.name = "constant4", value = 1 : i2} : <>, <i2>
    %33 = source {handshake.bb = 3 : ui32, handshake.name = "source9"} : <>
    %34 = constant %33 {handshake.name = "constant5", value = false} : <>, <i1>
    %35 = extsi %34 {handshake.name = "extsi4"} : <i1> to <i32>
    %addressResult_18, %dataResult_19 = load[%42] %0#1 {handshake.bb = 3 : ui32, handshake.deps = #handshake<deps[<"store1" (0)>]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 1>, handshake.name = "load4"} : <i4>, <i32>, <i4>, <i32>
    %36 = cmpi eq, %dataResult_19, %35 {handshake.bb = 3 : ui32, handshake.name = "cmpi2"} : <i32>
    %37 = cmpi ne, %dataResult_19, %35 {handshake.bb = 3 : ui32, handshake.name = "cmpi3"} : <i32>
    %38 = andi %37, %trueResult_14 {handshake.bb = 3 : ui32, handshake.name = "andi1"} : <i1>
    %39 = select %36[%32, %trueResult_12] {handshake.bb = 3 : ui32, handshake.name = "select0"} : <i1>, <i2>
    %trueResult_20, %falseResult_21 = cond_br %37, %result_16 {handshake.bb = 3 : ui32, handshake.name = "cond_br23"} : <i1>, <>
    %trueResult_22, %falseResult_23 = cond_br %24, %19 {handshake.bb = 4 : ui32, handshake.name = "cond_br31"} : <i1>, <i32>
    %40 = trunci %trueResult_22 {handshake.bb = 4 : ui32, handshake.name = "trunci0"} : <i32> to <i4>
    %41 = trunci %trueResult_22 {handshake.bb = 4 : ui32, handshake.name = "trunci1"} : <i32> to <i4>
    %42 = trunci %trueResult_22 {handshake.bb = 4 : ui32, handshake.name = "trunci2"} : <i32> to <i4>
    %43 = trunci %trueResult_22 {handshake.bb = 4 : ui32, handshake.name = "trunci3"} : <i32> to <i4>
    %result_24, %index_25 = control_merge [%trueResult_20]  {handshake.bb = 4 : ui32, handshake.name = "control_merge3"} : [<>] to <>, <i1>
    %44 = source {handshake.bb = 4 : ui32, handshake.name = "source10"} : <>
    %45 = constant %44 {handshake.bb = 4 : ui32, handshake.name = "constant12", value = true} : <>, <i1>
    %46 = source {handshake.bb = 4 : ui32, handshake.name = "source11"} : <>
    %47 = constant %46 {handshake.name = "constant15", value = 1 : i2} : <>, <i2>
    %48 = extsi %47 {handshake.name = "extsi5"} : <i2> to <i32>
    %addressResult_26, %dataResult_27 = load[%41] %0#2 {handshake.bb = 4 : ui32, handshake.deps = #handshake<deps[<"store1" (0)>]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 2>, handshake.name = "load5"} : <i4>, <i32>, <i4>, <i32>
    %49 = addi %dataResult_27, %48 {handshake.bb = 4 : ui32, handshake.name = "addi0"} : <i32>
    %addressResult_28, %dataResult_29 = store[%40] %49 {handshake.bb = 4 : ui32, handshake.deps = #handshake<deps[<"load3" (0)>, <"load4" (0)>, <"load5" (0)>, <"store1" (0)>]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 2>, handshake.name = "store1"} : <i4>, <i32>, <i4>, <i32>
    %50 = addi %trueResult_22, %48 {handshake.bb = 4 : ui32, handshake.name = "addi1"} : <i32>
    %51 = br %result_24 {handshake.bb = 4 : ui32, handshake.name = "br2"} : <>
    %result_30, %index_31 = control_merge [%falseResult_7]  {handshake.bb = 5 : ui32, handshake.name = "control_merge4"} : [<>] to <>, <i1>
    %52 = source {handshake.bb = 5 : ui32, handshake.name = "source12"} : <>
    %53 = constant %52 {handshake.name = "constant16", value = 2 : i3} : <>, <i3>
    %54 = select %falseResult_15[%53, %30] {handshake.bb = 5 : ui32, handshake.name = "select1"} : <i1>, <i3>
    %55 = extsi %54 {handshake.name = "extsi9"} : <i3> to <i32>
    end {handshake.bb = 5 : ui32, handshake.name = "end0"} %55, %0#3, %arg2 : <i32>, <>, <>
  }
}

