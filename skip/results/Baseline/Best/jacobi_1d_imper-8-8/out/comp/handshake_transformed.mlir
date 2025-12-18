module {
  handshake.func @jacobi_1d_imper(%arg0: memref<100xi32>, %arg1: memref<100xi32>, %arg2: !handshake.control<>, %arg3: !handshake.control<>, %arg4: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["A", "B", "A_start", "B_start", "start"], resNames = ["A_end", "B_end", "end"]} {
    %0:3 = fork [3] %arg4 {handshake.bb = 0 : ui32, handshake.name = "fork0"} : <>
    %1:2 = lsq[%arg1 : memref<100xi32>] (%arg3, %30#0, %addressResult_6, %dataResult_7, %80#0, %addressResult_16, %114#1)  {groupSizes = [1 : i32, 1 : i32], handshake.name = "lsq2"} : (!handshake.control<>, !handshake.control<>, !handshake.channel<i7>, !handshake.channel<i32>, !handshake.control<>, !handshake.channel<i7>, !handshake.control<>) -> (!handshake.channel<i32>, !handshake.control<>)
    %2:4 = lsq[%arg0 : memref<100xi32>] (%arg2, %30#2, %addressResult, %addressResult_2, %addressResult_4, %80#2, %addressResult_18, %dataResult_19, %114#0)  {groupSizes = [3 : i32, 1 : i32], handshake.name = "lsq3"} : (!handshake.control<>, !handshake.control<>, !handshake.channel<i7>, !handshake.channel<i7>, !handshake.channel<i7>, !handshake.control<>, !handshake.channel<i7>, !handshake.channel<i32>, !handshake.control<>) -> (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>)
    %3 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant0", value = false} : <>, <i1>
    %4 = br %3 {handshake.bb = 0 : ui32, handshake.name = "br2"} : <i1>
    %5 = extsi %4 {handshake.bb = 0 : ui32, handshake.name = "extsi11"} : <i1> to <i3>
    %6 = br %0#2 {handshake.bb = 0 : ui32, handshake.name = "br3"} : <>
    %7 = mux %index [%5, %trueResult_28] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i3>, <i3>] to <i3>
    %result, %index = control_merge [%6, %trueResult_30]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %8:2 = fork [2] %result {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <>
    %9 = constant %8#0 {handshake.bb = 1 : ui32, handshake.name = "constant1", value = 1 : i2} : <>, <i2>
    %10 = br %9 {handshake.bb = 1 : ui32, handshake.name = "br4"} : <i2>
    %11 = extsi %10 {handshake.bb = 1 : ui32, handshake.name = "extsi10"} : <i2> to <i8>
    %12 = br %7 {handshake.bb = 1 : ui32, handshake.name = "br5"} : <i3>
    %13 = br %8#1 {handshake.bb = 1 : ui32, handshake.name = "br6"} : <>
    %14 = mux %29#1 [%11, %trueResult] {handshake.bb = 2 : ui32, handshake.name = "mux1"} : <i1>, [<i8>, <i8>] to <i8>
    %16:5 = fork [5] %14 {handshake.bb = 2 : ui32, handshake.name = "fork2"} : <i8>
    %17 = trunci %16#0 {handshake.bb = 2 : ui32, handshake.name = "trunci0"} : <i8> to <i7>
    %19 = trunci %16#1 {handshake.bb = 2 : ui32, handshake.name = "trunci1"} : <i8> to <i7>
    %21 = extsi %16#4 {handshake.bb = 2 : ui32, handshake.name = "extsi12"} : <i8> to <i9>
    %23 = trunci %16#2 {handshake.bb = 2 : ui32, handshake.name = "trunci2"} : <i8> to <i7>
    %25 = trunci %16#3 {handshake.bb = 2 : ui32, handshake.name = "trunci3"} : <i8> to <i7>
    %27 = mux %29#0 [%12, %trueResult_8] {handshake.bb = 2 : ui32, handshake.name = "mux2"} : <i1>, [<i3>, <i3>] to <i3>
    %result_0, %index_1 = control_merge [%13, %trueResult_10]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>, <>] to <>, <i1>
    %29:2 = fork [2] %index_1 {handshake.bb = 2 : ui32, handshake.name = "fork3"} : <i1>
    %30:4 = lazy_fork [4] %result_0 {handshake.bb = 2 : ui32, handshake.name = "lazy_fork0"} : <>
    %31 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %32 = constant %31 {handshake.bb = 2 : ui32, handshake.name = "constant7", value = -1 : i32} : <>, <i32>
    %33 = trunci %32 {handshake.bb = 2 : ui32, handshake.name = "trunci4"} : <i32> to <i7>
    %34 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %35 = constant %34 {handshake.bb = 2 : ui32, handshake.name = "constant2", value = 99 : i8} : <>, <i8>
    %36 = extsi %35 {handshake.bb = 2 : ui32, handshake.name = "extsi13"} : <i8> to <i9>
    %37 = constant %30#3 {handshake.bb = 2 : ui32, handshake.name = "constant4", value = 1 : i2} : <>, <i2>
    %38:3 = fork [3] %37 {handshake.bb = 2 : ui32, handshake.name = "fork4"} : <i2>
    %39 = extsi %38#0 {handshake.bb = 2 : ui32, handshake.name = "extsi14"} : <i2> to <i7>
    %41 = extsi %38#1 {handshake.bb = 2 : ui32, handshake.name = "extsi15"} : <i2> to <i9>
    %43 = source {handshake.bb = 2 : ui32, handshake.name = "source2"} : <>
    %44 = constant %43 {handshake.bb = 2 : ui32, handshake.name = "constant5", value = 1 : i2} : <>, <i2>
    %45 = extsi %44 {handshake.bb = 2 : ui32, handshake.name = "extsi4"} : <i2> to <i32>
    %46 = addi %17, %33 {handshake.bb = 2 : ui32, handshake.name = "addi2"} : <i7>
    %addressResult, %dataResult = load[%46] %2#0 {handshake.bb = 2 : ui32, handshake.deps = #handshake<deps[["store1", 1], ["store1", 2]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 0>, handshake.name = "load0"} : <i7>, <i32>, <i7>, <i32>
    %addressResult_2, %dataResult_3 = load[%25] %2#1 {handshake.bb = 2 : ui32, handshake.deps = #handshake<deps[["store1", 1], ["store1", 2]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 0>, handshake.name = "load1"} : <i7>, <i32>, <i7>, <i32>
    %47 = addi %dataResult, %dataResult_3 {handshake.bb = 2 : ui32, handshake.name = "addi0"} : <i32>
    %48 = addi %19, %39 {handshake.bb = 2 : ui32, handshake.name = "addi3"} : <i7>
    %addressResult_4, %dataResult_5 = load[%48] %2#2 {handshake.bb = 2 : ui32, handshake.deps = #handshake<deps[["store1", 1], ["store1", 2]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 0>, handshake.name = "load2"} : <i7>, <i32>, <i7>, <i32>
    %49 = addi %47, %dataResult_5 {handshake.bb = 2 : ui32, handshake.name = "addi1"} : <i32>
    %50:2 = fork [2] %49 {handshake.bb = 2 : ui32, handshake.name = "fork5"} : <i32>
    %51 = shli %50#1, %45 {handshake.bb = 2 : ui32, handshake.name = "shli0"} : <i32>
    %53 = addi %50#0, %51 {handshake.bb = 2 : ui32, handshake.name = "addi7"} : <i32>
    %addressResult_6, %dataResult_7 = store[%23] %53 {handshake.bb = 2 : ui32, handshake.deps = #handshake<deps[["store0", 1], ["load3", 1], ["load3", 2]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 0>, handshake.name = "store0"} : <i7>, <i32>, <i7>, <i32>
    %55 = addi %21, %41 {handshake.bb = 2 : ui32, handshake.name = "addi4"} : <i9>
    %56:2 = fork [2] %55 {handshake.bb = 2 : ui32, handshake.name = "fork6"} : <i9>
    %57 = trunci %56#0 {handshake.bb = 2 : ui32, handshake.name = "trunci5"} : <i9> to <i8>
    %59 = cmpi ult, %56#1, %36 {handshake.bb = 2 : ui32, handshake.name = "cmpi0"} : <i9>
    %61:4 = fork [4] %59 {handshake.bb = 2 : ui32, handshake.name = "fork7"} : <i1>
    %trueResult, %falseResult = cond_br %61#0, %57 {handshake.bb = 2 : ui32, handshake.name = "cond_br3"} : <i1>, <i8>
    sink %falseResult {handshake.name = "sink0"} : <i8>
    %trueResult_8, %falseResult_9 = cond_br %61#1, %27 {handshake.bb = 2 : ui32, handshake.name = "cond_br4"} : <i1>, <i3>
    %trueResult_10, %falseResult_11 = cond_br %61#2, %30#1 {handshake.bb = 2 : ui32, handshake.name = "cond_br5"} : <i1>, <>
    %trueResult_12, %falseResult_13 = cond_br %61#3, %38#2 {handshake.bb = 2 : ui32, handshake.name = "cond_br6"} : <i1>, <i2>
    sink %trueResult_12 {handshake.name = "sink1"} : <i2>
    %67 = extsi %falseResult_13 {handshake.bb = 2 : ui32, handshake.name = "extsi9"} : <i2> to <i8>
    %68 = mux %79#1 [%67, %trueResult_20] {handshake.bb = 3 : ui32, handshake.name = "mux3"} : <i1>, [<i8>, <i8>] to <i8>
    %70:3 = fork [3] %68 {handshake.bb = 3 : ui32, handshake.name = "fork8"} : <i8>
    %71 = extsi %70#2 {handshake.bb = 3 : ui32, handshake.name = "extsi16"} : <i8> to <i9>
    %73 = trunci %70#0 {handshake.bb = 3 : ui32, handshake.name = "trunci6"} : <i8> to <i7>
    %75 = trunci %70#1 {handshake.bb = 3 : ui32, handshake.name = "trunci7"} : <i8> to <i7>
    %77 = mux %79#0 [%falseResult_9, %trueResult_22] {handshake.bb = 3 : ui32, handshake.name = "mux4"} : <i1>, [<i3>, <i3>] to <i3>
    %result_14, %index_15 = control_merge [%falseResult_11, %trueResult_24]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>, <>] to <>, <i1>
    %79:2 = fork [2] %index_15 {handshake.bb = 3 : ui32, handshake.name = "fork9"} : <i1>
    %80:3 = lazy_fork [3] %result_14 {handshake.bb = 3 : ui32, handshake.name = "lazy_fork1"} : <>
    %81 = source {handshake.bb = 3 : ui32, handshake.name = "source3"} : <>
    %82 = constant %81 {handshake.bb = 3 : ui32, handshake.name = "constant15", value = 99 : i8} : <>, <i8>
    %83 = extsi %82 {handshake.bb = 3 : ui32, handshake.name = "extsi17"} : <i8> to <i9>
    %84 = source {handshake.bb = 3 : ui32, handshake.name = "source4"} : <>
    %85 = constant %84 {handshake.bb = 3 : ui32, handshake.name = "constant16", value = 1 : i2} : <>, <i2>
    %86 = extsi %85 {handshake.bb = 3 : ui32, handshake.name = "extsi18"} : <i2> to <i9>
    %addressResult_16, %dataResult_17 = load[%75] %1#0 {handshake.bb = 3 : ui32, handshake.deps = #handshake<deps[["store0", 1]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 1>, handshake.name = "load3"} : <i7>, <i32>, <i7>, <i32>
    %addressResult_18, %dataResult_19 = store[%73] %dataResult_17 {handshake.bb = 3 : ui32, handshake.deps = #handshake<deps[["load0", 1], ["load1", 1], ["load2", 1], ["store1", 1]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 1>, handshake.name = "store1"} : <i7>, <i32>, <i7>, <i32>
    %87 = addi %71, %86 {handshake.bb = 3 : ui32, handshake.name = "addi5"} : <i9>
    %88:2 = fork [2] %87 {handshake.bb = 3 : ui32, handshake.name = "fork10"} : <i9>
    %89 = trunci %88#0 {handshake.bb = 3 : ui32, handshake.name = "trunci8"} : <i9> to <i8>
    %91 = cmpi ult, %88#1, %83 {handshake.bb = 3 : ui32, handshake.name = "cmpi1"} : <i9>
    %93:3 = fork [3] %91 {handshake.bb = 3 : ui32, handshake.name = "fork11"} : <i1>
    %trueResult_20, %falseResult_21 = cond_br %93#0, %89 {handshake.bb = 3 : ui32, handshake.name = "cond_br7"} : <i1>, <i8>
    sink %falseResult_21 {handshake.name = "sink2"} : <i8>
    %trueResult_22, %falseResult_23 = cond_br %93#1, %77 {handshake.bb = 3 : ui32, handshake.name = "cond_br8"} : <i1>, <i3>
    %trueResult_24, %falseResult_25 = cond_br %93#2, %80#1 {handshake.bb = 3 : ui32, handshake.name = "cond_br9"} : <i1>, <>
    %97 = merge %falseResult_23 {handshake.bb = 4 : ui32, handshake.name = "merge0"} : <i3>
    %98 = extsi %97 {handshake.bb = 4 : ui32, handshake.name = "extsi19"} : <i3> to <i4>
    %result_26, %index_27 = control_merge [%falseResult_25]  {handshake.bb = 4 : ui32, handshake.name = "control_merge3"} : [<>] to <>, <i1>
    sink %index_27 {handshake.name = "sink3"} : <i1>
    %99 = source {handshake.bb = 4 : ui32, handshake.name = "source5"} : <>
    %100 = constant %99 {handshake.bb = 4 : ui32, handshake.name = "constant17", value = 3 : i3} : <>, <i3>
    %101 = extsi %100 {handshake.bb = 4 : ui32, handshake.name = "extsi20"} : <i3> to <i4>
    %102 = source {handshake.bb = 4 : ui32, handshake.name = "source6"} : <>
    %103 = constant %102 {handshake.bb = 4 : ui32, handshake.name = "constant18", value = 1 : i2} : <>, <i2>
    %104 = extsi %103 {handshake.bb = 4 : ui32, handshake.name = "extsi21"} : <i2> to <i4>
    %105 = addi %98, %104 {handshake.bb = 4 : ui32, handshake.name = "addi6"} : <i4>
    %106:2 = fork [2] %105 {handshake.bb = 4 : ui32, handshake.name = "fork12"} : <i4>
    %107 = trunci %106#0 {handshake.bb = 4 : ui32, handshake.name = "trunci9"} : <i4> to <i3>
    %109 = cmpi ult, %106#1, %101 {handshake.bb = 4 : ui32, handshake.name = "cmpi2"} : <i4>
    %111:2 = fork [2] %109 {handshake.bb = 4 : ui32, handshake.name = "fork13"} : <i1>
    %trueResult_28, %falseResult_29 = cond_br %111#0, %107 {handshake.bb = 4 : ui32, handshake.name = "cond_br10"} : <i1>, <i3>
    sink %falseResult_29 {handshake.name = "sink4"} : <i3>
    %trueResult_30, %falseResult_31 = cond_br %111#1, %result_26 {handshake.bb = 4 : ui32, handshake.name = "cond_br11"} : <i1>, <>
    %result_32, %index_33 = control_merge [%falseResult_31]  {handshake.bb = 5 : ui32, handshake.name = "control_merge4"} : [<>] to <>, <i1>
    sink %index_33 {handshake.name = "sink5"} : <i1>
    %114:2 = fork [2] %result_32 {handshake.bb = 5 : ui32, handshake.name = "fork14"} : <>
    end {handshake.bb = 5 : ui32, handshake.name = "end0"} %2#3, %1#1, %0#1 : <>, <>, <>
  }
}

