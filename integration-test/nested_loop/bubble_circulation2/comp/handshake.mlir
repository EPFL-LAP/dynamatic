module {
  handshake.func @nested_loop(%arg0: memref<1000xi32>, %arg1: memref<1000xi32>, %arg2: memref<1000xi32>, %arg3: !handshake.control<>, %arg4: !handshake.control<>, %arg5: !handshake.control<>, %arg6: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["a", "b", "c", "a_start", "b_start", "c_start", "start"], resNames = ["a_end", "b_end", "c_end", "end"]} {
    %memEnd = mem_controller[%arg2 : memref<1000xi32>] %arg5 (%13, %addressResult_7, %dataResult_8) %result_21 {connectedBlocks = [2 : i32], handshake.name = "mem_controller0"} :    (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>) -> ()
    %outputs, %memEnd_0 = mem_controller[%arg1 : memref<1000xi32>] %arg4 (%addressResult_5) %result_21 {connectedBlocks = [2 : i32], handshake.name = "mem_controller1"} :    (!handshake.channel<i32>) -> !handshake.channel<i32>
    %outputs_1, %memEnd_2 = mem_controller[%arg0 : memref<1000xi32>] %arg3 (%addressResult) %result_21 {connectedBlocks = [2 : i32], handshake.name = "mem_controller2"} :    (!handshake.channel<i32>) -> !handshake.channel<i32>
    %0 = constant %arg6 {handshake.bb = 0 : ui32, handshake.name = "constant2", value = 0 : i32} : <>, <i32>
    %1 = br %0 {handshake.bb = 0 : ui32, handshake.name = "br2"} : <i32>
    %2 = br %arg6 {handshake.bb = 0 : ui32, handshake.name = "br3"} : <>
    %3 = mux %index [%1, %trueResult_17] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i32>, <i32>] to <i32>
    %result, %index = control_merge [%2, %trueResult_19]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %4 = source {handshake.bb = 1 : ui32, handshake.name = "source0"} : <>
    %5 = constant %4 {handshake.bb = 1 : ui32, handshake.name = "constant7", value = 400 : i32} : <>, <i32>
    %6 = constant %result {handshake.bb = 1 : ui32, handshake.name = "constant8", value = 0 : i32} : <>, <i32>
    %7 = trunci %3 {handshake.bb = 1 : ui32, handshake.name = "index_cast0"} : <i32> to <i32>
    %8 = muli %7, %5 {handshake.bb = 1 : ui32, handshake.name = "muli0"} : <i32>
    %9 = br %6 {handshake.bb = 1 : ui32, handshake.name = "br4"} : <i32>
    %10 = br %3 {handshake.bb = 1 : ui32, handshake.name = "br5"} : <i32>
    %11 = br %8 {handshake.bb = 1 : ui32, handshake.name = "br6"} : <i32>
    %12 = br %result {handshake.bb = 1 : ui32, handshake.name = "br7"} : <>
    %13 = constant %result_3 {handshake.bb = 2 : ui32, handshake.name = "constant0", value = 1 : i32} : <>, <i32>
    %14 = mux %index_4 [%9, %trueResult] {handshake.bb = 2 : ui32, handshake.name = "mux1"} : <i1>, [<i32>, <i32>] to <i32>
    %15 = mux %index_4 [%10, %trueResult_9] {handshake.bb = 2 : ui32, handshake.name = "mux2"} : <i1>, [<i32>, <i32>] to <i32>
    %16 = mux %index_4 [%11, %trueResult_11] {handshake.bb = 2 : ui32, handshake.name = "mux3"} : <i1>, [<i32>, <i32>] to <i32>
    %result_3, %index_4 = control_merge [%12, %trueResult_13]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>, <>] to <>, <i1>
    %17 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %18 = constant %17 {handshake.bb = 2 : ui32, handshake.name = "constant9", value = 1 : i32} : <>, <i32>
    %19 = source {handshake.bb = 2 : ui32, handshake.name = "source2"} : <>
    %20 = constant %19 {handshake.bb = 2 : ui32, handshake.name = "constant10", value = 1000 : i32} : <>, <i32>
    %21 = trunci %14 {handshake.bb = 2 : ui32, handshake.name = "index_cast1"} : <i32> to <i32>
    %addressResult, %dataResult = load[%21] %outputs_1 {handshake.bb = 2 : ui32, handshake.name = "load2"} : <i32>, <i32>, <i32>, <i32>
    %addressResult_5, %dataResult_6 = load[%21] %outputs {handshake.bb = 2 : ui32, handshake.name = "load3"} : <i32>, <i32>, <i32>, <i32>
    %22 = muli %dataResult, %dataResult_6 {handshake.bb = 2 : ui32, handshake.name = "muli1"} : <i32>
    %23 = addi %14, %16 {handshake.bb = 2 : ui32, handshake.name = "addi0"} : <i32>
    %24 = trunci %23 {handshake.bb = 2 : ui32, handshake.name = "index_cast2"} : <i32> to <i32>
    %addressResult_7, %dataResult_8 = store[%24] %22 {handshake.bb = 2 : ui32, handshake.deps = #handshake<deps[<"store1" (0)>]>, handshake.name = "store1"} : <i32>, <i32>, <i32>, <i32>
    %25 = cmpi slt, %22, %20 {handshake.bb = 2 : ui32, handshake.name = "cmpi0"} : <i32>
    %26 = addi %14, %18 {handshake.bb = 2 : ui32, handshake.name = "addi1"} : <i32>
    %trueResult, %falseResult = cond_br %25, %26 {handshake.bb = 2 : ui32, handshake.name = "cond_br2"} : <i1>, <i32>
    %trueResult_9, %falseResult_10 = cond_br %25, %15 {handshake.bb = 2 : ui32, handshake.name = "cond_br3"} : <i1>, <i32>
    %trueResult_11, %falseResult_12 = cond_br %25, %16 {handshake.bb = 2 : ui32, handshake.name = "cond_br4"} : <i1>, <i32>
    %trueResult_13, %falseResult_14 = cond_br %25, %result_3 {handshake.bb = 2 : ui32, handshake.name = "cond_br5"} : <i1>, <>
    %27 = merge %falseResult_10 {handshake.bb = 3 : ui32, handshake.name = "merge0"} : <i32>
    %result_15, %index_16 = control_merge [%falseResult_14]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>] to <>, <i1>
    %28 = source {handshake.bb = 3 : ui32, handshake.name = "source3"} : <>
    %29 = constant %28 {handshake.bb = 3 : ui32, handshake.name = "constant11", value = 2 : i32} : <>, <i32>
    %30 = source {handshake.bb = 3 : ui32, handshake.name = "source4"} : <>
    %31 = constant %30 {handshake.bb = 3 : ui32, handshake.name = "constant12", value = 1 : i32} : <>, <i32>
    %32 = addi %27, %31 {handshake.bb = 3 : ui32, handshake.name = "addi2"} : <i32>
    %33 = cmpi ult, %32, %29 {handshake.bb = 3 : ui32, handshake.name = "cmpi1"} : <i32>
    %trueResult_17, %falseResult_18 = cond_br %33, %32 {handshake.bb = 3 : ui32, handshake.name = "cond_br6"} : <i1>, <i32>
    %trueResult_19, %falseResult_20 = cond_br %33, %result_15 {handshake.bb = 3 : ui32, handshake.name = "cond_br7"} : <i1>, <>
    %result_21, %index_22 = control_merge [%falseResult_20]  {handshake.bb = 4 : ui32, handshake.name = "control_merge3"} : [<>] to <>, <i1>
    end {handshake.bb = 4 : ui32, handshake.name = "end0"} %memEnd_2, %memEnd_0, %memEnd, %arg6 : <>, <>, <>, <>
  }
}

