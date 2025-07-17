module {
  handshake.func @iterative_sqrt(%arg0: !handshake.channel<i32>, %arg1: !handshake.control<>, ...) -> (!handshake.channel<i32>, !handshake.control<>) attributes {argNames = ["n", "start"], cfg.edges = "[0,1][2,3,1,cmpi2][4,1][1,2,6,andi0][3,4,5,cmpi3][5,1]", resNames = ["out0", "end"]} {
    %0 = source {handshake.bb = 0 : ui32, handshake.name = "source0"} : <>
    %1 = constant %0 {handshake.bb = 0 : ui32, handshake.name = "constant0", value = 0 : i32} : <>, <i32>
    %2 = source {handshake.bb = 0 : ui32, handshake.name = "source1"} : <>
    %3 = constant %2 {handshake.bb = 0 : ui32, handshake.name = "constant3", value = true} : <>, <i1>
    %4 = constant %arg1 {handshake.bb = 0 : ui32, handshake.name = "mlir.undef0", value = 0 : i32} : <>, <i32>
    %trueResult, %falseResult = cond_br %18, %10 {handshake.bb = 1 : ui32, handshake.name = "cond_br35"} : <i1>, <i32>
    %trueResult_0, %falseResult_1 = cond_br %18, %8 {handshake.bb = 1 : ui32, handshake.name = "cond_br36"} : <i1>, <i32>
    %5 = constant %arg1 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "constant1", value = false} : <>, <i1>
    %6 = merge %5, %18 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "merge0"} : <i1>
    %7 = mux %6 [%arg0, %trueResult_6] {ftd.regen, handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i32>, <i32>] to <i32>
    %8 = mux %25 [%15, %9] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux3"} : <i1>, [<i32>, <i32>] to <i32>
    %9 = mux %28 [%15, %31] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux4"} : <i1>, [<i32>, <i32>] to <i32>
    %10 = mux %25 [%16, %11] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux5"} : <i1>, [<i32>, <i32>] to <i32>
    %11 = mux %28 [%34, %16] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux6"} : <i1>, [<i32>, <i32>] to <i32>
    %12 = mux %6 [%3, %25] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux7"} : <i1>, [<i1>, <i1>] to <i1>
    %13 = mux %6 [%4, %27] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux8"} : <i1>, [<i32>, <i32>] to <i32>
    %14 = mux %6 [%3, %26] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux9"} : <i1>, [<i1>, <i1>] to <i1>
    %15 = mux %6 [%1, %trueResult_0] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux10"} : <i1>, [<i32>, <i32>] to <i32>
    %16 = mux %6 [%arg0, %trueResult] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux11"} : <i1>, [<i32>, <i32>] to <i32>
    %17 = cmpi sle, %15, %16 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i32>
    %18 = andi %17, %12 {handshake.bb = 1 : ui32, handshake.name = "andi0"} : <i1>
    %trueResult_2, %falseResult_3 = cond_br %18, %13 {handshake.bb = 2 : ui32, handshake.name = "cond_br37"} : <i1>, <i32>
    %trueResult_4, %falseResult_5 = cond_br %18, %14 {handshake.bb = 2 : ui32, handshake.name = "cond_br38"} : <i1>, <i1>
    %trueResult_6, %falseResult_7 = cond_br %18, %7 {handshake.bb = 2 : ui32, handshake.name = "cond_br39"} : <i1>, <i32>
    %trueResult_8, %falseResult_9 = cond_br %18, %16 {handshake.bb = 2 : ui32, handshake.name = "cond_br40"} : <i1>, <i32>
    %trueResult_10, %falseResult_11 = cond_br %18, %15 {handshake.bb = 2 : ui32, handshake.name = "cond_br41"} : <i1>, <i32>
    %19 = source {handshake.bb = 2 : ui32, handshake.name = "source6"} : <>
    %20 = constant %19 {handshake.bb = 2 : ui32, handshake.name = "constant4", value = 1 : i32} : <>, <i32>
    %21 = addi %trueResult_10, %trueResult_8 {handshake.bb = 2 : ui32, handshake.name = "addi0"} : <i32>
    %22 = shrsi %21, %20 {handshake.bb = 2 : ui32, handshake.name = "shrsi0"} : <i32>
    %23 = muli %22, %22 {handshake.bb = 2 : ui32, handshake.name = "muli0"} : <i32>
    %24 = cmpi eq, %23, %trueResult_6 {handshake.bb = 2 : ui32, handshake.name = "cmpi1"} : <i32>
    %25 = cmpi ne, %23, %trueResult_6 {handshake.bb = 2 : ui32, handshake.name = "cmpi2"} : <i32>
    %26 = andi %25, %trueResult_4 {handshake.bb = 2 : ui32, handshake.name = "andi1"} : <i1>
    %27 = select %24[%22, %trueResult_2] {handshake.bb = 2 : ui32, handshake.name = "select0"} : <i1>, <i32>
    %28 = cmpi slt, %23, %trueResult_6 {handshake.bb = 2 : ui32, handshake.name = "cmpi3"} : <i32>
    %trueResult_12, %falseResult_13 = cond_br %28, %22 {handshake.bb = 4 : ui32, handshake.name = "cond_br42"} : <i1>, <i32>
    %29 = source {handshake.bb = 4 : ui32, handshake.name = "source7"} : <>
    %30 = constant %29 {handshake.bb = 4 : ui32, handshake.name = "constant5", value = 1 : i32} : <>, <i32>
    %31 = addi %trueResult_12, %30 {handshake.bb = 4 : ui32, handshake.name = "addi1"} : <i32>
    %32 = source {handshake.bb = 5 : ui32, handshake.name = "source8"} : <>
    %33 = constant %32 {handshake.bb = 5 : ui32, handshake.name = "constant6", value = -1 : i32} : <>, <i32>
    %34 = addi %falseResult_13, %33 {handshake.bb = 5 : ui32, handshake.name = "addi2"} : <i32>
    %35 = select %falseResult_5[%falseResult_9, %falseResult_3] {handshake.bb = 6 : ui32, handshake.name = "select1"} : <i1>, <i32>
    end {handshake.bb = 6 : ui32, handshake.name = "end0"} %35, %arg1 : <i32>, <>
  }
}

