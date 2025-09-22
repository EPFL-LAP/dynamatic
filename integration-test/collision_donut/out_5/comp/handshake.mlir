module {
  handshake.func @collision_donut(%arg0: memref<1000xi32>, %arg1: memref<1000xi32>, %arg2: !handshake.control<>, %arg3: !handshake.control<>, %arg4: !handshake.control<>, ...) -> (!handshake.channel<i32>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["x", "y", "x_start", "y_start", "start"], resNames = ["out0", "x_end", "y_end", "end"]} {
    %outputs, %memEnd = mem_controller[%arg1 : memref<1000xi32>] %arg3 (%addressResult_2) %result_28 {connectedBlocks = [1 : i32], handshake.name = "mem_controller0"} :    (!handshake.channel<i32>) -> !handshake.channel<i32>
    %outputs_0, %memEnd_1 = mem_controller[%arg0 : memref<1000xi32>] %arg2 (%addressResult) %result_28 {connectedBlocks = [1 : i32], handshake.name = "mem_controller1"} :    (!handshake.channel<i32>) -> !handshake.channel<i32>
    %0 = constant %arg4 {handshake.bb = 0 : ui32, handshake.name = "constant6", value = 0 : i32} : <>, <i32>
    %1 = br %0 {handshake.bb = 0 : ui32, handshake.name = "br2"} : <i32>
    %2 = br %arg4 {handshake.bb = 0 : ui32, handshake.name = "br3"} : <>
    %3 = mux %index [%1, %trueResult_20] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i32>, <i32>] to <i32>
    %result, %index = control_merge [%2, %trueResult_22]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %4 = source {handshake.bb = 1 : ui32, handshake.name = "source0"} : <>
    %5 = constant %4 {handshake.bb = 1 : ui32, handshake.name = "constant7", value = 4 : i32} : <>, <i32>
    %6 = constant %result {handshake.bb = 1 : ui32, handshake.name = "constant8", value = -1 : i32} : <>, <i32>
    %7 = extui %3 {handshake.bb = 1 : ui32, handshake.name = "extui0"} : <i32> to <i64>
    %8 = trunci %7 {handshake.bb = 1 : ui32, handshake.name = "index_cast0"} : <i64> to <i32>
    %addressResult, %dataResult = load[%8] %outputs_0 {handshake.bb = 1 : ui32, handshake.name = "load0"} : <i32>, <i32>, <i32>, <i32>
    %9 = extui %3 {handshake.bb = 1 : ui32, handshake.name = "extui1"} : <i32> to <i64>
    %10 = trunci %9 {handshake.bb = 1 : ui32, handshake.name = "index_cast1"} : <i64> to <i32>
    %addressResult_2, %dataResult_3 = load[%10] %outputs {handshake.bb = 1 : ui32, handshake.name = "load1"} : <i32>, <i32>, <i32>, <i32>
    %11 = muli %dataResult, %dataResult {handshake.bb = 1 : ui32, handshake.name = "muli0"} : <i32>
    %12 = muli %dataResult_3, %dataResult_3 {handshake.bb = 1 : ui32, handshake.name = "muli1"} : <i32>
    %13 = addi %11, %12 {handshake.bb = 1 : ui32, handshake.name = "addi0"} : <i32>
    %14 = cmpi ult, %13, %5 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i32>
    %trueResult, %falseResult = cond_br %14, %3 {handshake.bb = 1 : ui32, handshake.name = "cond_br3"} : <i1>, <i32>
    %trueResult_4, %falseResult_5 = cond_br %14, %6 {handshake.bb = 1 : ui32, handshake.name = "cond_br4"} : <i1>, <i32>
    %trueResult_6, %falseResult_7 = cond_br %14, %result {handshake.bb = 1 : ui32, handshake.name = "cond_br5"} : <i1>, <>
    %trueResult_8, %falseResult_9 = cond_br %14, %13 {handshake.bb = 1 : ui32, handshake.name = "cond_br6"} : <i1>, <i32>
    %15 = merge %falseResult {handshake.bb = 2 : ui32, handshake.name = "merge0"} : <i32>
    %16 = merge %falseResult_9 {handshake.bb = 2 : ui32, handshake.name = "merge1"} : <i32>
    %result_10, %index_11 = control_merge [%falseResult_7]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>] to <>, <i1>
    %17 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %18 = constant %17 {handshake.bb = 2 : ui32, handshake.name = "constant9", value = 19000 : i32} : <>, <i32>
    %19 = constant %result_10 {handshake.bb = 2 : ui32, handshake.name = "constant10", value = -2 : i32} : <>, <i32>
    %20 = cmpi ugt, %16, %18 {handshake.bb = 2 : ui32, handshake.name = "cmpi1"} : <i32>
    %trueResult_12, %falseResult_13 = cond_br %20, %15 {handshake.bb = 2 : ui32, handshake.name = "cond_br7"} : <i1>, <i32>
    %trueResult_14, %falseResult_15 = cond_br %20, %19 {handshake.bb = 2 : ui32, handshake.name = "cond_br8"} : <i1>, <i32>
    %trueResult_16, %falseResult_17 = cond_br %20, %result_10 {handshake.bb = 2 : ui32, handshake.name = "cond_br9"} : <i1>, <>
    %21 = merge %falseResult_13 {handshake.bb = 3 : ui32, handshake.name = "merge2"} : <i32>
    %result_18, %index_19 = control_merge [%falseResult_17]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>] to <>, <i1>
    %22 = constant %result_18 {handshake.bb = 3 : ui32, handshake.name = "constant11", value = 0 : i32} : <>, <i32>
    %23 = source {handshake.bb = 3 : ui32, handshake.name = "source2"} : <>
    %24 = constant %23 {handshake.bb = 3 : ui32, handshake.name = "constant12", value = 1 : i32} : <>, <i32>
    %25 = source {handshake.bb = 3 : ui32, handshake.name = "source3"} : <>
    %26 = constant %25 {handshake.bb = 3 : ui32, handshake.name = "constant13", value = 1000 : i32} : <>, <i32>
    %27 = addi %21, %24 {handshake.bb = 3 : ui32, handshake.name = "addi1"} : <i32>
    %28 = cmpi ult, %27, %26 {handshake.bb = 3 : ui32, handshake.name = "cmpi2"} : <i32>
    %trueResult_20, %falseResult_21 = cond_br %28, %27 {handshake.bb = 3 : ui32, handshake.name = "cond_br10"} : <i1>, <i32>
    %trueResult_22, %falseResult_23 = cond_br %28, %result_18 {handshake.bb = 3 : ui32, handshake.name = "cond_br11"} : <i1>, <>
    %trueResult_24, %falseResult_25 = cond_br %28, %22 {handshake.bb = 3 : ui32, handshake.name = "cond_br12"} : <i1>, <i32>
    %29 = mux %index_27 [%trueResult, %falseResult_21] {handshake.bb = 4 : ui32, handshake.name = "mux1"} : <i1>, [<i32>, <i32>] to <i32>
    %30 = mux %index_27 [%trueResult_4, %falseResult_25] {handshake.bb = 4 : ui32, handshake.name = "mux2"} : <i1>, [<i32>, <i32>] to <i32>
    %result_26, %index_27 = control_merge [%trueResult_6, %falseResult_23]  {handshake.bb = 4 : ui32, handshake.name = "control_merge3"} : [<>, <>] to <>, <i1>
    %31 = br %29 {handshake.bb = 4 : ui32, handshake.name = "br4"} : <i32>
    %32 = br %30 {handshake.bb = 4 : ui32, handshake.name = "br5"} : <i32>
    %33 = br %result_26 {handshake.bb = 4 : ui32, handshake.name = "br6"} : <>
    %34 = mux %index_29 [%trueResult_12, %31] {handshake.bb = 5 : ui32, handshake.name = "mux3"} : <i1>, [<i32>, <i32>] to <i32>
    %35 = mux %index_29 [%trueResult_14, %32] {handshake.bb = 5 : ui32, handshake.name = "mux4"} : <i1>, [<i32>, <i32>] to <i32>
    %result_28, %index_29 = control_merge [%trueResult_16, %33]  {handshake.bb = 5 : ui32, handshake.name = "control_merge4"} : [<>, <>] to <>, <i1>
    %36 = source {handshake.bb = 5 : ui32, handshake.name = "source4"} : <>
    %37 = constant %36 {handshake.bb = 5 : ui32, handshake.name = "constant14", value = 1 : i32} : <>, <i32>
    %38 = shli %34, %37 {handshake.bb = 5 : ui32, handshake.name = "shli0"} : <i32>
    %39 = andi %38, %35 {handshake.bb = 5 : ui32, handshake.name = "andi0"} : <i32>
    end {handshake.bb = 5 : ui32, handshake.name = "end0"} %39, %memEnd_1, %memEnd, %arg4 : <i32>, <>, <>, <>
  }
}

