module {
  handshake.func @if_convert(%arg0: memref<200xi32>, %arg1: memref<200xi32>, %arg2: !handshake.control<>, %arg3: !handshake.control<>, %arg4: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["a", "b", "a_start", "b_start", "start"], resNames = ["a_end", "b_end", "end"]} {
    %memEnd = mem_controller[%arg1 : memref<200xi32>] %arg3 (%15, %addressResult_9, %dataResult_10) %result_17 {connectedBlocks = [3 : i32], handshake.name = "mem_controller0"} :    (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>) -> ()
    %outputs, %memEnd_0 = mem_controller[%arg0 : memref<200xi32>] %arg2 (%addressResult) %result_17 {connectedBlocks = [3 : i32], handshake.name = "mem_controller1"} :    (!handshake.channel<i32>) -> !handshake.channel<i32>
    %0 = constant %arg4 {handshake.bb = 0 : ui32, handshake.name = "constant3", value = 1 : i32} : <>, <i32>
    %1 = br %0 {handshake.bb = 0 : ui32, handshake.name = "br1"} : <i32>
    %2 = br %arg4 {handshake.bb = 0 : ui32, handshake.name = "br4"} : <>
    %3 = mux %index [%1, %trueResult_13] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i32>, <i32>] to <i32>
    %4 = mux %index [%1, %trueResult_13] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i1>, [<i32>, <i32>] to <i32>
    %result, %index = control_merge [%2, %trueResult_15]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %5 = br %3 {handshake.bb = 1 : ui32, handshake.name = "br5"} : <i32>
    %6 = br %4 {handshake.bb = 1 : ui32, handshake.name = "br6"} : <i32>
    %7 = br %result {handshake.bb = 1 : ui32, handshake.name = "br7"} : <>
    %8 = mux %index_2 [%5, %31] {handshake.bb = 2 : ui32, handshake.name = "mux2"} : <i1>, [<i32>, <i32>] to <i32>
    %9 = mux %index_2 [%6, %32] {handshake.bb = 2 : ui32, handshake.name = "mux3"} : <i1>, [<i32>, <i32>] to <i32>
    %result_1, %index_2 = control_merge [%7, %33]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>, <>] to <>, <i1>
    %10 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %11 = constant %10 {handshake.bb = 2 : ui32, handshake.name = "constant4", value = 199 : i32} : <>, <i32>
    %12 = cmpi slt, %9, %11 {handshake.bb = 2 : ui32, handshake.name = "cmpi0"} : <i32>
    %13 = cmpi eq, %9, %8 {handshake.bb = 2 : ui32, handshake.name = "cmpi1"} : <i32>
    %14 = andi %12, %13 {handshake.bb = 2 : ui32, handshake.name = "andi0"} : <i1>
    %trueResult, %falseResult = cond_br %14, %8 {handshake.bb = 2 : ui32, handshake.name = "cond_br2"} : <i1>, <i32>
    %trueResult_3, %falseResult_4 = cond_br %14, %9 {handshake.bb = 2 : ui32, handshake.name = "cond_br3"} : <i1>, <i32>
    %trueResult_5, %falseResult_6 = cond_br %14, %result_1 {handshake.bb = 2 : ui32, handshake.name = "cond_br4"} : <i1>, <>
    %15 = constant %result_7 {handshake.bb = 3 : ui32, handshake.name = "constant0", value = 1 : i32} : <>, <i32>
    %16 = merge %trueResult {handshake.bb = 3 : ui32, handshake.name = "merge0"} : <i32>
    %17 = merge %trueResult_3 {handshake.bb = 3 : ui32, handshake.name = "merge1"} : <i32>
    %result_7, %index_8 = control_merge [%trueResult_5]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>] to <>, <i1>
    %18 = constant %result_7 {handshake.bb = 3 : ui32, handshake.name = "constant5", value = 1 : i32} : <>, <i32>
    %19 = source {handshake.bb = 3 : ui32, handshake.name = "source1"} : <>
    %20 = constant %19 {handshake.bb = 3 : ui32, handshake.name = "constant6", value = 10000 : i32} : <>, <i32>
    %21 = source {handshake.bb = 3 : ui32, handshake.name = "source2"} : <>
    %22 = constant %21 {handshake.bb = 3 : ui32, handshake.name = "constant7", value = 2 : i32} : <>, <i32>
    %23 = trunci %16 {handshake.bb = 3 : ui32, handshake.name = "index_cast0"} : <i32> to <i32>
    %addressResult, %dataResult = load[%23] %outputs {handshake.bb = 3 : ui32, handshake.name = "load1"} : <i32>, <i32>, <i32>, <i32>
    %24 = muli %16, %dataResult {handshake.bb = 3 : ui32, handshake.name = "muli0"} : <i32>
    %25 = cmpi slt, %24, %20 {handshake.bb = 3 : ui32, handshake.name = "cmpi2"} : <i32>
    %26 = addi %17, %22 {handshake.bb = 3 : ui32, handshake.name = "addi0"} : <i32>
    %27 = addi %17, %18 {handshake.bb = 3 : ui32, handshake.name = "addi1"} : <i32>
    %28 = select %25[%26, %27] {handshake.bb = 3 : ui32, handshake.name = "select1"} : <i1>, <i32>
    %29 = trunci %28 {handshake.bb = 3 : ui32, handshake.name = "index_cast1"} : <i32> to <i32>
    %addressResult_9, %dataResult_10 = store[%29] %18 {handshake.bb = 3 : ui32, handshake.name = "store1"} : <i32>, <i32>, <i32>, <i32>
    %30 = addi %17, %18 {handshake.bb = 3 : ui32, handshake.name = "addi2"} : <i32>
    %31 = br %28 {handshake.bb = 3 : ui32, handshake.name = "br8"} : <i32>
    %32 = br %30 {handshake.bb = 3 : ui32, handshake.name = "br9"} : <i32>
    %33 = br %result_7 {handshake.bb = 3 : ui32, handshake.name = "br10"} : <>
    %34 = merge %falseResult {handshake.bb = 4 : ui32, handshake.name = "merge2"} : <i32>
    %result_11, %index_12 = control_merge [%falseResult_6]  {handshake.bb = 4 : ui32, handshake.name = "control_merge3"} : [<>] to <>, <i1>
    %35 = source {handshake.bb = 4 : ui32, handshake.name = "source3"} : <>
    %36 = constant %35 {handshake.bb = 4 : ui32, handshake.name = "constant8", value = 199 : i32} : <>, <i32>
    %37 = cmpi slt, %34, %36 {handshake.bb = 4 : ui32, handshake.name = "cmpi3"} : <i32>
    %trueResult_13, %falseResult_14 = cond_br %37, %34 {handshake.bb = 4 : ui32, handshake.name = "cond_br5"} : <i1>, <i32>
    %trueResult_15, %falseResult_16 = cond_br %37, %result_11 {handshake.bb = 4 : ui32, handshake.name = "cond_br6"} : <i1>, <>
    %result_17, %index_18 = control_merge [%falseResult_16]  {handshake.bb = 5 : ui32, handshake.name = "control_merge4"} : [<>] to <>, <i1>
    end {handshake.bb = 5 : ui32, handshake.name = "end0"} %memEnd_0, %memEnd, %arg4 : <>, <>, <>
  }
}

