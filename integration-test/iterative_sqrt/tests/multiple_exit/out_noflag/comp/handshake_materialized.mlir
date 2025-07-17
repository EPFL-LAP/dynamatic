module {
  handshake.func @multiple_exit(%arg0: memref<10xi32>, %arg1: !handshake.control<>, %arg2: !handshake.control<>, ...) -> (!handshake.channel<i32>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["arr", "arr_start", "start"], resNames = ["out0", "arr_end", "end"]} {
    %0:4 = fork [4] %arg2 {handshake.bb = 0 : ui32, handshake.name = "fork0"} : <>
    %1:4 = lsq[%arg0 : memref<10xi32>] (%arg1, %30#1, %addressResult, %42#1, %addressResult_22, %60#1, %addressResult_38, %addressResult_40, %dataResult_41, %result_42)  {groupSizes = [1 : i32, 1 : i32, 2 : i32], handshake.name = "lsq1"} : (!handshake.control<>, !handshake.control<>, !handshake.channel<i4>, !handshake.control<>, !handshake.channel<i4>, !handshake.control<>, !handshake.channel<i4>, !handshake.channel<i4>, !handshake.channel<i32>, !handshake.control<>) -> (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>)
    %2 = constant %0#3 {handshake.bb = 0 : ui32, handshake.name = "constant1", value = false} : <>, <i1>
    %3:2 = fork [2] %2 {handshake.bb = 0 : ui32, handshake.name = "fork1"} : <i1>
    %4 = constant %0#2 {handshake.bb = 0 : ui32, handshake.name = "constant6", value = true} : <>, <i1>
    %5 = br %3#1 {handshake.bb = 0 : ui32, handshake.name = "br1"} : <i1>
    %6 = extsi %5 {handshake.bb = 0 : ui32, handshake.name = "extsi8"} : <i1> to <i32>
    %7 = br %4 {handshake.bb = 0 : ui32, handshake.name = "br2"} : <i1>
    %8:2 = fork [2] %7 {handshake.bb = 0 : ui32, handshake.name = "fork2"} : <i1>
    %9 = br %3#0 {handshake.bb = 0 : ui32, handshake.name = "br3"} : <i1>
    %10 = extsi %9 {handshake.bb = 0 : ui32, handshake.name = "extsi7"} : <i1> to <i2>
    %11 = br %0#1 {handshake.bb = 0 : ui32, handshake.name = "br4"} : <>
    %12 = mux %17#0 [%6, %falseResult_13, %falseResult_25, %69] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i2>, [<i32>, <i32>, <i32>, <i32>] to <i32>
    %13:2 = fork [2] %12 {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <i32>
    %14 = mux %17#1 [%8#1, %falseResult_9, %falseResult_29, %70] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i2>, [<i1>, <i1>, <i1>, <i1>] to <i1>
    %15 = mux %17#2 [%10, %falseResult_11, %falseResult_31, %71] {handshake.bb = 1 : ui32, handshake.name = "mux2"} : <i2>, [<i2>, <i2>, <i2>, <i2>] to <i2>
    %16 = mux %17#3 [%8#0, %falseResult_19, %falseResult_35, %72] {handshake.bb = 1 : ui32, handshake.name = "mux3"} : <i2>, [<i1>, <i1>, <i1>, <i1>] to <i1>
    %result, %index = control_merge [%11, %falseResult_17, %falseResult_33, %73]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>, <>, <>] to <>, <i2>
    %17:4 = fork [4] %index {handshake.bb = 1 : ui32, handshake.name = "fork4"} : <i2>
    %18 = source {handshake.bb = 1 : ui32, handshake.name = "source0"} : <>
    %19 = constant %18 {handshake.bb = 1 : ui32, handshake.name = "constant2", value = 10 : i5} : <>, <i5>
    %20 = extsi %19 {handshake.bb = 1 : ui32, handshake.name = "extsi2"} : <i5> to <i32>
    %21 = cmpi slt, %13#1, %20 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i32>
    %22 = andi %21, %16 {handshake.bb = 1 : ui32, handshake.name = "andi0"} : <i1>
    %23:4 = fork [4] %22 {handshake.bb = 1 : ui32, handshake.name = "fork5"} : <i1>
    %trueResult, %falseResult = cond_br %23#3, %14 {handshake.bb = 1 : ui32, handshake.name = "cond_br3"} : <i1>, <i1>
    %trueResult_0, %falseResult_1 = cond_br %23#2, %15 {handshake.bb = 1 : ui32, handshake.name = "cond_br4"} : <i1>, <i2>
    %trueResult_2, %falseResult_3 = cond_br %23#1, %13#0 {handshake.bb = 1 : ui32, handshake.name = "cond_br5"} : <i1>, <i32>
    sink %falseResult_3 {handshake.name = "sink0"} : <i32>
    %trueResult_4, %falseResult_5 = cond_br %23#0, %result {handshake.bb = 1 : ui32, handshake.name = "cond_br6"} : <i1>, <>
    %24 = merge %trueResult {handshake.bb = 2 : ui32, handshake.name = "merge0"} : <i1>
    %25 = merge %trueResult_0 {handshake.bb = 2 : ui32, handshake.name = "merge1"} : <i2>
    %26 = merge %trueResult_2 {handshake.bb = 2 : ui32, handshake.name = "merge2"} : <i32>
    %27:3 = fork [3] %26 {handshake.bb = 2 : ui32, handshake.name = "fork6"} : <i32>
    %28 = trunci %27#2 {handshake.bb = 2 : ui32, handshake.name = "trunci0"} : <i32> to <i4>
    %29 = trunci %27#1 {handshake.bb = 2 : ui32, handshake.name = "trunci1"} : <i32> to <i4>
    %result_6, %index_7 = control_merge [%trueResult_4]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>] to <>, <i1>
    sink %index_7 {handshake.name = "sink1"} : <i1>
    %30:3 = lazy_fork [3] %result_6 {handshake.bb = 2 : ui32, handshake.name = "lazy_fork0"} : <>
    %31 = fork [1] %30#2 {handshake.bb = 2 : ui32, handshake.name = "fork7"} : <>
    %32 = constant %31 {handshake.bb = 2 : ui32, handshake.name = "constant8", value = false} : <>, <i1>
    %33 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %34 = constant %33 {handshake.bb = 2 : ui32, handshake.name = "constant9", value = -1 : i32} : <>, <i32>
    %addressResult, %dataResult = load[%29] %1#0 {handshake.bb = 2 : ui32, handshake.deps = #handshake<deps[<"store1" (0)>]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 0>, handshake.name = "load3"} : <i4>, <i32>, <i4>, <i32>
    %35 = cmpi ne, %dataResult, %34 {handshake.bb = 2 : ui32, handshake.name = "cmpi1"} : <i32>
    %36:6 = fork [6] %35 {handshake.bb = 2 : ui32, handshake.name = "fork8"} : <i1>
    %trueResult_8, %falseResult_9 = cond_br %36#5, %24 {handshake.bb = 2 : ui32, handshake.name = "cond_br7"} : <i1>, <i1>
    %trueResult_10, %falseResult_11 = cond_br %36#4, %25 {handshake.bb = 2 : ui32, handshake.name = "cond_br8"} : <i1>, <i2>
    %trueResult_12, %falseResult_13 = cond_br %36#3, %27#0 {handshake.bb = 2 : ui32, handshake.name = "cond_br9"} : <i1>, <i32>
    %trueResult_14, %falseResult_15 = cond_br %36#2, %28 {handshake.bb = 2 : ui32, handshake.name = "cond_br10"} : <i1>, <i4>
    sink %falseResult_15 {handshake.name = "sink2"} : <i4>
    %trueResult_16, %falseResult_17 = cond_br %36#1, %30#0 {handshake.bb = 2 : ui32, handshake.name = "cond_br11"} : <i1>, <>
    %trueResult_18, %falseResult_19 = cond_br %36#0, %32 {handshake.bb = 2 : ui32, handshake.name = "cond_br12"} : <i1>, <i1>
    sink %trueResult_18 {handshake.name = "sink3"} : <i1>
    %37 = merge %trueResult_8 {handshake.bb = 3 : ui32, handshake.name = "merge3"} : <i1>
    %38 = merge %trueResult_10 {handshake.bb = 3 : ui32, handshake.name = "merge4"} : <i2>
    %39 = merge %trueResult_12 {handshake.bb = 3 : ui32, handshake.name = "merge5"} : <i32>
    %40 = merge %trueResult_14 {handshake.bb = 3 : ui32, handshake.name = "merge6"} : <i4>
    %41:2 = fork [2] %40 {handshake.bb = 3 : ui32, handshake.name = "fork9"} : <i4>
    %result_20, %index_21 = control_merge [%trueResult_16]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>] to <>, <i1>
    sink %index_21 {handshake.name = "sink4"} : <i1>
    %42:2 = lazy_fork [2] %result_20 {handshake.bb = 3 : ui32, handshake.name = "lazy_fork1"} : <>
    %43 = source {handshake.bb = 3 : ui32, handshake.name = "source2"} : <>
    %44 = constant %43 {handshake.bb = 3 : ui32, handshake.name = "constant3", value = 1 : i2} : <>, <i2>
    %45 = source {handshake.bb = 3 : ui32, handshake.name = "source3"} : <>
    %46 = constant %45 {handshake.bb = 3 : ui32, handshake.name = "constant4", value = false} : <>, <i1>
    %47 = extsi %46 {handshake.bb = 3 : ui32, handshake.name = "extsi4"} : <i1> to <i32>
    %48:2 = fork [2] %47 {handshake.bb = 3 : ui32, handshake.name = "fork10"} : <i32>
    %addressResult_22, %dataResult_23 = load[%41#1] %1#1 {handshake.bb = 3 : ui32, handshake.deps = #handshake<deps[<"store1" (0)>]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 1>, handshake.name = "load4"} : <i4>, <i32>, <i4>, <i32>
    %49:2 = fork [2] %dataResult_23 {handshake.bb = 3 : ui32, handshake.name = "fork11"} : <i32>
    %50 = cmpi eq, %49#1, %48#1 {handshake.bb = 3 : ui32, handshake.name = "cmpi2"} : <i32>
    %51 = cmpi ne, %49#0, %48#0 {handshake.bb = 3 : ui32, handshake.name = "cmpi3"} : <i32>
    %52:8 = fork [8] %51 {handshake.bb = 3 : ui32, handshake.name = "fork12"} : <i1>
    %53 = andi %52#7, %37 {handshake.bb = 3 : ui32, handshake.name = "andi1"} : <i1>
    %54 = select %50[%44, %38] {handshake.bb = 3 : ui32, handshake.name = "select0"} : <i1>, <i2>
    %trueResult_24, %falseResult_25 = cond_br %52#6, %39 {handshake.bb = 3 : ui32, handshake.name = "cond_br13"} : <i1>, <i32>
    %trueResult_26, %falseResult_27 = cond_br %52#5, %41#0 {handshake.bb = 3 : ui32, handshake.name = "cond_br14"} : <i1>, <i4>
    sink %falseResult_27 {handshake.name = "sink5"} : <i4>
    %trueResult_28, %falseResult_29 = cond_br %52#4, %53 {handshake.bb = 3 : ui32, handshake.name = "cond_br15"} : <i1>, <i1>
    %trueResult_30, %falseResult_31 = cond_br %52#3, %54 {handshake.bb = 3 : ui32, handshake.name = "cond_br16"} : <i1>, <i2>
    %trueResult_32, %falseResult_33 = cond_br %52#2, %42#0 {handshake.bb = 3 : ui32, handshake.name = "cond_br17"} : <i1>, <>
    %trueResult_34, %falseResult_35 = cond_br %52#0, %52#1 {handshake.bb = 3 : ui32, handshake.name = "cond_br18"} : <i1>, <i1>
    sink %trueResult_34 {handshake.name = "sink6"} : <i1>
    %55 = merge %trueResult_24 {handshake.bb = 4 : ui32, handshake.name = "merge7"} : <i32>
    %56 = merge %trueResult_26 {handshake.bb = 4 : ui32, handshake.name = "merge8"} : <i4>
    %57:2 = fork [2] %56 {handshake.bb = 4 : ui32, handshake.name = "fork13"} : <i4>
    %58 = merge %trueResult_28 {handshake.bb = 4 : ui32, handshake.name = "merge9"} : <i1>
    %59 = merge %trueResult_30 {handshake.bb = 4 : ui32, handshake.name = "merge10"} : <i2>
    %result_36, %index_37 = control_merge [%trueResult_32]  {handshake.bb = 4 : ui32, handshake.name = "control_merge3"} : [<>] to <>, <i1>
    sink %index_37 {handshake.name = "sink7"} : <i1>
    %60:3 = lazy_fork [3] %result_36 {handshake.bb = 4 : ui32, handshake.name = "lazy_fork2"} : <>
    %61 = fork [1] %60#2 {handshake.bb = 4 : ui32, handshake.name = "fork14"} : <>
    %62 = constant %61 {handshake.bb = 4 : ui32, handshake.name = "constant12", value = true} : <>, <i1>
    %63 = source {handshake.bb = 4 : ui32, handshake.name = "source4"} : <>
    %64 = constant %63 {handshake.bb = 4 : ui32, handshake.name = "constant5", value = 1 : i2} : <>, <i2>
    %65 = extsi %64 {handshake.bb = 4 : ui32, handshake.name = "extsi5"} : <i2> to <i32>
    %66:2 = fork [2] %65 {handshake.bb = 4 : ui32, handshake.name = "fork15"} : <i32>
    %addressResult_38, %dataResult_39 = load[%57#1] %1#2 {handshake.bb = 4 : ui32, handshake.deps = #handshake<deps[<"store1" (0)>]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 2>, handshake.name = "load5"} : <i4>, <i32>, <i4>, <i32>
    %67 = addi %dataResult_39, %66#1 {handshake.bb = 4 : ui32, handshake.name = "addi0"} : <i32>
    %addressResult_40, %dataResult_41 = store[%57#0] %67 {handshake.bb = 4 : ui32, handshake.deps = #handshake<deps[<"load3" (0)>, <"load4" (0)>, <"load5" (0)>, <"store1" (0)>]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 2>, handshake.name = "store1"} : <i4>, <i32>, <i4>, <i32>
    %68 = addi %55, %66#0 {handshake.bb = 4 : ui32, handshake.name = "addi1"} : <i32>
    %69 = br %68 {handshake.bb = 4 : ui32, handshake.name = "br5"} : <i32>
    %70 = br %58 {handshake.bb = 4 : ui32, handshake.name = "br6"} : <i1>
    %71 = br %59 {handshake.bb = 4 : ui32, handshake.name = "br8"} : <i2>
    %72 = br %62 {handshake.bb = 4 : ui32, handshake.name = "br9"} : <i1>
    %73 = br %60#0 {handshake.bb = 4 : ui32, handshake.name = "br10"} : <>
    %74 = merge %falseResult {handshake.bb = 5 : ui32, handshake.name = "merge11"} : <i1>
    %75 = merge %falseResult_1 {handshake.bb = 5 : ui32, handshake.name = "merge12"} : <i2>
    %76 = extsi %75 {handshake.bb = 5 : ui32, handshake.name = "extsi9"} : <i2> to <i3>
    %result_42, %index_43 = control_merge [%falseResult_5]  {handshake.bb = 5 : ui32, handshake.name = "control_merge4"} : [<>] to <>, <i1>
    sink %index_43 {handshake.name = "sink8"} : <i1>
    %77 = source {handshake.bb = 5 : ui32, handshake.name = "source5"} : <>
    %78 = constant %77 {handshake.bb = 5 : ui32, handshake.name = "constant15", value = 2 : i3} : <>, <i3>
    %79 = select %74[%78, %76] {handshake.bb = 5 : ui32, handshake.name = "select1"} : <i1>, <i3>
    %80 = extsi %79 {handshake.bb = 5 : ui32, handshake.name = "extsi10"} : <i3> to <i32>
    end {handshake.bb = 5 : ui32, handshake.name = "end0"} %80, %1#3, %0#0 : <i32>, <>, <>
  }
}

