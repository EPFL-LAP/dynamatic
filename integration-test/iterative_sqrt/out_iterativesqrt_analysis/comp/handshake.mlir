module {
  handshake.func @iterative_sqrt(%arg0: !handshake.channel<i32>, %arg1: !handshake.control<>, ...) -> (!handshake.channel<i32>, !handshake.control<>) attributes {argNames = ["n", "start"], cfg.edges = "[0,1][2,3,1,cmpi2][4,1][1,2,6,andi0][3,4,5,cmpi3][5,1]", resNames = ["out0", "end"]} {
    %0 = source {handshake.bb = 0 : ui32, handshake.name = "source0"} : <>
    %1 = constant %0 {handshake.bb = 0 : ui32, handshake.name = "constant0", value = 0 : i32} : <>, <i32>
    %2 = source {handshake.bb = 0 : ui32, handshake.name = "source1"} : <>
    %3 = constant %2 {handshake.bb = 0 : ui32, handshake.name = "constant3", value = true} : <>, <i1>
    %4 = constant %arg1 {handshake.bb = 0 : ui32, handshake.name = "mlir.undef0", value = 0 : i32} : <>, <i32>
    %5 = br %arg1 {handshake.bb = 0 : ui32, handshake.name = "br2"} : <>
    %trueResult, %falseResult = cond_br %19, %11 {handshake.bb = 1 : ui32, handshake.name = "cond_br35"} : <i1>, <i32>
    %trueResult_0, %falseResult_1 = cond_br %19, %9 {handshake.bb = 1 : ui32, handshake.name = "cond_br36"} : <i1>, <i32>
    %6 = constant %arg1 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "constant1", value = false} : <>, <i1>
    %7 = merge %6, %19 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "merge0"} : <i1>
    %8 = mux %7 [%arg0, %trueResult_8] {ftd.regen, handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i32>, <i32>] to <i32>
    %result, %index = control_merge [%5, %falseResult_17, %33, %37]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>, <>, <>] to <>, <i2>
    %9 = mux %26 [%16, %10] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux3"} : <i1>, [<i32>, <i32>] to <i32>
    %10 = mux %29 [%16, %32] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux4"} : <i1>, [<i32>, <i32>] to <i32>
    %11 = mux %26 [%17, %12] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux5"} : <i1>, [<i32>, <i32>] to <i32>
    %12 = mux %29 [%36, %17] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux6"} : <i1>, [<i32>, <i32>] to <i32>
    %13 = mux %7 [%3, %26] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux7"} : <i1>, [<i1>, <i1>] to <i1>
    %14 = mux %7 [%4, %28] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux8"} : <i1>, [<i32>, <i32>] to <i32>
    %15 = mux %7 [%3, %27] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux9"} : <i1>, [<i1>, <i1>] to <i1>
    %16 = mux %7 [%1, %trueResult_0] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux10"} : <i1>, [<i32>, <i32>] to <i32>
    %17 = mux %7 [%arg0, %trueResult] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux11"} : <i1>, [<i32>, <i32>] to <i32>
    %18 = cmpi sle, %16, %17 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i32>
    %19 = andi %18, %13 {handshake.bb = 1 : ui32, handshake.name = "andi0"} : <i1>
    %trueResult_2, %falseResult_3 = cond_br %19, %result {handshake.bb = 1 : ui32, handshake.name = "cond_br11"} : <i1>, <>
    %trueResult_4, %falseResult_5 = cond_br %19, %14 {handshake.bb = 2 : ui32, handshake.name = "cond_br37"} : <i1>, <i32>
    %trueResult_6, %falseResult_7 = cond_br %19, %15 {handshake.bb = 2 : ui32, handshake.name = "cond_br38"} : <i1>, <i1>
    %trueResult_8, %falseResult_9 = cond_br %19, %8 {handshake.bb = 2 : ui32, handshake.name = "cond_br39"} : <i1>, <i32>
    %trueResult_10, %falseResult_11 = cond_br %19, %17 {handshake.bb = 2 : ui32, handshake.name = "cond_br40"} : <i1>, <i32>
    %trueResult_12, %falseResult_13 = cond_br %19, %16 {handshake.bb = 2 : ui32, handshake.name = "cond_br41"} : <i1>, <i32>
    %result_14, %index_15 = control_merge [%trueResult_2]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>] to <>, <i1>
    %20 = source {handshake.bb = 2 : ui32, handshake.name = "source6"} : <>
    %21 = constant %20 {handshake.bb = 2 : ui32, handshake.name = "constant4", value = 1 : i32} : <>, <i32>
    %22 = addi %trueResult_12, %trueResult_10 {handshake.bb = 2 : ui32, handshake.name = "addi0"} : <i32>
    %23 = shrsi %22, %21 {handshake.bb = 2 : ui32, handshake.name = "shrsi0"} : <i32>
    %24 = muli %23, %23 {handshake.bb = 2 : ui32, handshake.name = "muli0"} : <i32>
    %25 = cmpi eq, %24, %trueResult_8 {handshake.bb = 2 : ui32, handshake.name = "cmpi1"} : <i32>
    %26 = cmpi ne, %24, %trueResult_8 {handshake.bb = 2 : ui32, handshake.name = "cmpi2"} : <i32>
    %27 = andi %26, %trueResult_6 {handshake.bb = 2 : ui32, handshake.name = "andi1"} : <i1>
    %28 = select %25[%23, %trueResult_4] {handshake.bb = 2 : ui32, handshake.name = "select0"} : <i1>, <i32>
    %29 = cmpi slt, %24, %trueResult_8 {handshake.bb = 2 : ui32, handshake.name = "cmpi3"} : <i32>
    %trueResult_16, %falseResult_17 = cond_br %26, %result_14 {handshake.bb = 2 : ui32, handshake.name = "cond_br31"} : <i1>, <>
    %result_18, %index_19 = control_merge [%trueResult_16]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>] to <>, <i1>
    %trueResult_20, %falseResult_21 = cond_br %29, %result_18 {handshake.bb = 3 : ui32, handshake.name = "cond_br32"} : <i1>, <>
    %trueResult_22, %falseResult_23 = cond_br %29, %23 {handshake.bb = 4 : ui32, handshake.name = "cond_br42"} : <i1>, <i32>
    %result_24, %index_25 = control_merge [%trueResult_20]  {handshake.bb = 4 : ui32, handshake.name = "control_merge3"} : [<>] to <>, <i1>
    %30 = source {handshake.bb = 4 : ui32, handshake.name = "source7"} : <>
    %31 = constant %30 {handshake.bb = 4 : ui32, handshake.name = "constant5", value = 1 : i32} : <>, <i32>
    %32 = addi %trueResult_22, %31 {handshake.bb = 4 : ui32, handshake.name = "addi1"} : <i32>
    %33 = br %result_24 {handshake.bb = 4 : ui32, handshake.name = "br3"} : <>
    %result_26, %index_27 = control_merge [%falseResult_21]  {handshake.bb = 5 : ui32, handshake.name = "control_merge4"} : [<>] to <>, <i1>
    %34 = source {handshake.bb = 5 : ui32, handshake.name = "source8"} : <>
    %35 = constant %34 {handshake.bb = 5 : ui32, handshake.name = "constant6", value = -1 : i32} : <>, <i32>
    %36 = addi %falseResult_23, %35 {handshake.bb = 5 : ui32, handshake.name = "addi2"} : <i32>
    %37 = br %result_26 {handshake.bb = 5 : ui32, handshake.name = "br4"} : <>
    %38 = select %falseResult_7[%falseResult_11, %falseResult_5] {handshake.bb = 6 : ui32, handshake.name = "select1"} : <i1>, <i32>
    end {handshake.bb = 6 : ui32, handshake.name = "end0"} %38, %arg1 : <i32>, <>
  }
}

