module {
  handshake.func @multiple_exit(%arg0: memref<10xi32>, %arg1: !handshake.control<>, %arg2: !handshake.control<>, ...) -> (!handshake.channel<i32>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["arr", "arr_start", "start"], cfg.edges = "[0,1][2,3,1,cmpi1][4,1][1,2,5,andi0][3,4,1,cmpi3]", resNames = ["out0", "arr_end", "end"]} {
    %0:3 = fork [3] %arg2 {handshake.bb = 0 : ui32, handshake.name = "fork0"} : <>
    %1:4 = lsq[%arg0 : memref<10xi32>] (%arg1, %33#1, %addressResult, %41#1, %addressResult_18, %59#1, %addressResult_26, %addressResult_28, %dataResult_29, %result_30)  {groupSizes = [1 : i32, 1 : i32, 2 : i32], handshake.name = "lsq1"} : (!handshake.control<>, !handshake.control<>, !handshake.channel<i4>, !handshake.control<>, !handshake.channel<i4>, !handshake.control<>, !handshake.channel<i4>, !handshake.channel<i4>, !handshake.channel<i32>, !handshake.control<>) -> (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>)
    %2 = source {handshake.bb = 0 : ui32, handshake.name = "source0"} : <>
    %3 = constant %2 {handshake.bb = 0 : ui32, handshake.name = "constant2", value = false} : <>, <i1>
    %4 = extsi %3 {handshake.bb = 1 : ui32, handshake.name = "extsi0"} : <i1> to <i32>
    %5 = source {handshake.bb = 0 : ui32, handshake.name = "source1"} : <>
    %6 = constant %5 {handshake.bb = 0 : ui32, handshake.name = "constant6", value = true} : <>, <i1>
    %7:2 = fork [2] %6 {handshake.bb = 0 : ui32, handshake.name = "fork1"} : <i1>
    %8 = br %0#2 {handshake.bb = 0 : ui32, handshake.name = "br1"} : <>
    %trueResult, %falseResult = cond_br %32#4, %13 {handshake.bb = 1 : ui32, handshake.name = "cond_br25"} : <i1>, <i32>
    sink %falseResult {handshake.name = "sink0"} : <i32>
    %trueResult_0, %falseResult_1 = cond_br %32#5, %12 {handshake.bb = 1 : ui32, handshake.name = "cond_br26"} : <i1>, <i1>
    sink %falseResult_1 {handshake.name = "sink1"} : <i1>
    %trueResult_2, %falseResult_3 = cond_br %32#6, %11 {handshake.bb = 1 : ui32, handshake.name = "cond_br27"} : <i1>, <i2>
    sink %falseResult_3 {handshake.name = "sink2"} : <i2>
    %trueResult_4, %falseResult_5 = cond_br %32#7, %9 {handshake.bb = 1 : ui32, handshake.name = "cond_br28"} : <i1>, <i1>
    sink %falseResult_5 {handshake.name = "sink3"} : <i1>
    %result, %index = control_merge [%8, %falseResult_11, %falseResult_21, %68]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>, <>, <>] to <>, <i2>
    sink %index {handshake.name = "sink4"} : <i2>
    %9 = mux %39#1 [%35, %10] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i1>, <i1>] to <i1>
    %10 = mux %51#2 [%51#3, %61] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i1>, [<i1>, <i1>] to <i1>
    %11 = mux %39#2 [%22#1, %53] {handshake.bb = 1 : ui32, handshake.name = "mux2"} : <i1>, [<i2>, <i2>] to <i2>
    %12 = mux %39#3 [%24#1, %52] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux3"} : <i1>, [<i1>, <i1>] to <i1>
    %13 = mux %39#4 [%26#2, %14] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux4"} : <i1>, [<i32>, <i32>] to <i32>
    %14 = mux %51#4 [%26#3, %67] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux5"} : <i1>, [<i32>, <i32>] to <i32>
    %15 = constant %0#1 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "constant1", value = false} : <>, <i1>
    %16:2 = fork [2] %15 {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <i1>
    %17 = extsi %16#1 {handshake.bb = 1 : ui32, handshake.name = "extsi7"} : <i1> to <i2>
    %18 = mux %20#3 [%7#1, %trueResult_4] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux6"} : <i1>, [<i1>, <i1>] to <i1>
    %19 = merge %16#0, %32#8 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "merge0"} : <i1>
    %20:4 = fork [4] %19 {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <i1>
    %21 = mux %20#2 [%17, %trueResult_2] {handshake.bb = 1 : ui32, handshake.name = "mux7"} : <i1>, [<i2>, <i2>] to <i2>
    %22:2 = fork [2] %21 {handshake.bb = 1 : ui32, handshake.name = "fork4"} : <i2>
    %23 = mux %20#1 [%7#0, %trueResult_0] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux8"} : <i1>, [<i1>, <i1>] to <i1>
    %24:2 = fork [2] %23 {handshake.bb = 1 : ui32, handshake.name = "fork5"} : <i1>
    %25 = mux %20#0 [%4, %trueResult] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux9"} : <i1>, [<i32>, <i32>] to <i32>
    %26:4 = fork [4] %25 {handshake.bb = 1 : ui32, handshake.name = "fork6"} : <i32>
    %27 = source {handshake.bb = 1 : ui32, handshake.name = "source2"} : <>
    %28 = constant %27 {handshake.bb = 1 : ui32, handshake.name = "constant3", value = 10 : i5} : <>, <i5>
    %29 = extsi %28 {handshake.bb = 1 : ui32, handshake.name = "extsi2"} : <i5> to <i32>
    %30 = cmpi slt, %26#1, %29 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i32>
    %31 = andi %30, %18 {handshake.bb = 1 : ui32, handshake.name = "andi0"} : <i1>
    %32:9 = fork [9] %31 {handshake.bb = 1 : ui32, handshake.name = "fork7"} : <i1>
    %trueResult_6, %falseResult_7 = cond_br %32#3, %result {handshake.bb = 1 : ui32, handshake.name = "cond_br9"} : <i1>, <>
    %result_8, %index_9 = control_merge [%trueResult_6]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>] to <>, <i1>
    sink %index_9 {handshake.name = "sink5"} : <i1>
    %33:2 = lazy_fork [2] %result_8 {handshake.bb = 2 : ui32, handshake.name = "lazy_fork0"} : <>
    %34 = source {handshake.bb = 2 : ui32, handshake.name = "source6"} : <>
    %35 = constant %34 {handshake.bb = 2 : ui32, handshake.name = "constant8", value = false} : <>, <i1>
    %36 = source {handshake.bb = 2 : ui32, handshake.name = "source7"} : <>
    %37 = constant %36 {handshake.bb = 2 : ui32, handshake.name = "constant9", value = -1 : i32} : <>, <i32>
    %addressResult, %dataResult = load[%58] %1#0 {handshake.bb = 2 : ui32, handshake.deps = #handshake<deps[<"store1" (0)>]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 0>, handshake.name = "load3"} : <i4>, <i32>, <i4>, <i32>
    %38 = cmpi ne, %dataResult, %37 {handshake.bb = 2 : ui32, handshake.name = "cmpi1"} : <i32>
    %39:5 = fork [5] %38 {handshake.bb = 2 : ui32, handshake.name = "fork8"} : <i1>
    %trueResult_10, %falseResult_11 = cond_br %39#0, %33#0 {handshake.bb = 2 : ui32, handshake.name = "cond_br20"} : <i1>, <>
    %trueResult_12, %falseResult_13 = cond_br %32#2, %22#0 {handshake.bb = 3 : ui32, handshake.name = "cond_br29"} : <i1>, <i2>
    %40 = extsi %falseResult_13 {handshake.bb = 3 : ui32, handshake.name = "extsi8"} : <i2> to <i3>
    %trueResult_14, %falseResult_15 = cond_br %32#1, %24#0 {handshake.bb = 3 : ui32, handshake.name = "cond_br30"} : <i1>, <i1>
    %result_16, %index_17 = control_merge [%trueResult_10]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>] to <>, <i1>
    sink %index_17 {handshake.name = "sink6"} : <i1>
    %41:2 = lazy_fork [2] %result_16 {handshake.bb = 3 : ui32, handshake.name = "lazy_fork1"} : <>
    %42 = source {handshake.bb = 3 : ui32, handshake.name = "source8"} : <>
    %43 = constant %42 {handshake.bb = 3 : ui32, handshake.name = "constant4", value = 1 : i2} : <>, <i2>
    %44 = source {handshake.bb = 3 : ui32, handshake.name = "source9"} : <>
    %45 = constant %44 {handshake.bb = 3 : ui32, handshake.name = "constant5", value = false} : <>, <i1>
    %46 = extsi %45 {handshake.bb = 3 : ui32, handshake.name = "extsi4"} : <i1> to <i32>
    %47:2 = fork [2] %46 {handshake.bb = 3 : ui32, handshake.name = "fork9"} : <i32>
    %addressResult_18, %dataResult_19 = load[%57] %1#1 {handshake.bb = 3 : ui32, handshake.deps = #handshake<deps[<"store1" (0)>]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 1>, handshake.name = "load4"} : <i4>, <i32>, <i4>, <i32>
    %48:2 = fork [2] %dataResult_19 {handshake.bb = 3 : ui32, handshake.name = "fork10"} : <i32>
    %49 = cmpi eq, %48#1, %47#1 {handshake.bb = 3 : ui32, handshake.name = "cmpi2"} : <i32>
    %50 = cmpi ne, %48#0, %47#0 {handshake.bb = 3 : ui32, handshake.name = "cmpi3"} : <i32>
    %51:5 = fork [5] %50 {handshake.bb = 3 : ui32, handshake.name = "fork11"} : <i1>
    %52 = andi %51#1, %trueResult_14 {handshake.bb = 3 : ui32, handshake.name = "andi1"} : <i1>
    %53 = select %49[%43, %trueResult_12] {handshake.bb = 3 : ui32, handshake.name = "select0"} : <i1>, <i2>
    %trueResult_20, %falseResult_21 = cond_br %51#0, %41#0 {handshake.bb = 3 : ui32, handshake.name = "cond_br23"} : <i1>, <>
    %trueResult_22, %falseResult_23 = cond_br %32#0, %26#0 {handshake.bb = 4 : ui32, handshake.name = "cond_br31"} : <i1>, <i32>
    sink %falseResult_23 {handshake.name = "sink7"} : <i32>
    %54:5 = fork [5] %trueResult_22 {handshake.bb = 4 : ui32, handshake.name = "fork12"} : <i32>
    %55 = trunci %54#4 {handshake.bb = 4 : ui32, handshake.name = "trunci0"} : <i32> to <i4>
    %56 = trunci %54#3 {handshake.bb = 4 : ui32, handshake.name = "trunci1"} : <i32> to <i4>
    %57 = trunci %54#2 {handshake.bb = 4 : ui32, handshake.name = "trunci2"} : <i32> to <i4>
    %58 = trunci %54#1 {handshake.bb = 4 : ui32, handshake.name = "trunci3"} : <i32> to <i4>
    %result_24, %index_25 = control_merge [%trueResult_20]  {handshake.bb = 4 : ui32, handshake.name = "control_merge3"} : [<>] to <>, <i1>
    sink %index_25 {handshake.name = "sink8"} : <i1>
    %59:2 = lazy_fork [2] %result_24 {handshake.bb = 4 : ui32, handshake.name = "lazy_fork2"} : <>
    %60 = source {handshake.bb = 4 : ui32, handshake.name = "source10"} : <>
    %61 = constant %60 {handshake.bb = 4 : ui32, handshake.name = "constant12", value = true} : <>, <i1>
    %62 = source {handshake.bb = 4 : ui32, handshake.name = "source11"} : <>
    %63 = constant %62 {handshake.bb = 4 : ui32, handshake.name = "constant15", value = 1 : i2} : <>, <i2>
    %64 = extsi %63 {handshake.bb = 4 : ui32, handshake.name = "extsi5"} : <i2> to <i32>
    %65:2 = fork [2] %64 {handshake.bb = 4 : ui32, handshake.name = "fork13"} : <i32>
    %addressResult_26, %dataResult_27 = load[%56] %1#2 {handshake.bb = 4 : ui32, handshake.deps = #handshake<deps[<"store1" (0)>]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 2>, handshake.name = "load5"} : <i4>, <i32>, <i4>, <i32>
    %66 = addi %dataResult_27, %65#1 {handshake.bb = 4 : ui32, handshake.name = "addi0"} : <i32>
    %addressResult_28, %dataResult_29 = store[%55] %66 {handshake.bb = 4 : ui32, handshake.deps = #handshake<deps[<"load3" (0)>, <"load4" (0)>, <"load5" (0)>, <"store1" (0)>]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 2>, handshake.name = "store1"} : <i4>, <i32>, <i4>, <i32>
    %67 = addi %54#0, %65#0 {handshake.bb = 4 : ui32, handshake.name = "addi1"} : <i32>
    %68 = br %59#0 {handshake.bb = 4 : ui32, handshake.name = "br2"} : <>
    %result_30, %index_31 = control_merge [%falseResult_7]  {handshake.bb = 5 : ui32, handshake.name = "control_merge4"} : [<>] to <>, <i1>
    sink %index_31 {handshake.name = "sink9"} : <i1>
    %69 = source {handshake.bb = 5 : ui32, handshake.name = "source12"} : <>
    %70 = constant %69 {handshake.bb = 5 : ui32, handshake.name = "constant16", value = 2 : i3} : <>, <i3>
    %71 = select %falseResult_15[%70, %40] {handshake.bb = 5 : ui32, handshake.name = "select1"} : <i1>, <i3>
    %72 = extsi %71 {handshake.bb = 5 : ui32, handshake.name = "extsi9"} : <i3> to <i32>
    end {handshake.bb = 5 : ui32, handshake.name = "end0"} %72, %1#3, %0#0 : <i32>, <>, <>
  }
}

