module {
  handshake.func @iterative_sqrt(%arg0: !handshake.channel<i32>, %arg1: !handshake.control<>, ...) -> (!handshake.channel<i32>, !handshake.control<>) attributes {argNames = ["n", "start"], cfg.edges = "[0,1][2,3,1,cmpi2][4,1][1,2,6,andi0][3,4,5,cmpi3][5,1]", resNames = ["out0", "end"]} {
    %0 = source {handshake.bb = 0 : ui32, handshake.name = "source0"} : <>
    %1 = constant %0 {handshake.name = "constant2", value = false} : <>, <i1>
    %2 = extsi %1 {handshake.name = "extsi0"} : <i1> to <i32>
    %3 = source {handshake.bb = 0 : ui32, handshake.name = "source1"} : <>
    %4 = constant %3 {handshake.bb = 0 : ui32, handshake.name = "constant3", value = true} : <>, <i1>
    %trueResult, %falseResult = cond_br %19, %11 {handshake.bb = 1 : ui32, handshake.name = "cond_br35"} : <i1>, <i32>
    %trueResult_0, %falseResult_1 = cond_br %19, %9 {handshake.bb = 1 : ui32, handshake.name = "cond_br36"} : <i1>, <i32>
    %5 = constant %arg1 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "constant1", value = false} : <>, <i1>
    %6 = extsi %5 {handshake.bb = 1 : ui32, handshake.name = "extsi1"} : <i1> to <i32>
    %7 = merge %5, %19 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "merge0"} : <i1>
    %8 = mux %7 [%arg0, %trueResult_6] {ftd.regen, handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i32>, <i32>] to <i32>
    %9 = mux %27 [%16, %10] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux3"} : <i1>, [<i32>, <i32>] to <i32>
    %10 = mux %30 [%16, %34] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux4"} : <i1>, [<i32>, <i32>] to <i32>
    %11 = mux %27 [%17, %12] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux5"} : <i1>, [<i32>, <i32>] to <i32>
    %12 = mux %30 [%37, %17] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux6"} : <i1>, [<i32>, <i32>] to <i32>
    %13 = mux %7 [%4, %27] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux7"} : <i1>, [<i1>, <i1>] to <i1>
    %14 = mux %7 [%6, %29] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux8"} : <i1>, [<i32>, <i32>] to <i32>
    %15 = mux %7 [%4, %28] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux9"} : <i1>, [<i1>, <i1>] to <i1>
    %16 = mux %7 [%2, %trueResult_0] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux10"} : <i1>, [<i32>, <i32>] to <i32>
    %17 = mux %7 [%arg0, %trueResult] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux11"} : <i1>, [<i32>, <i32>] to <i32>
    %18 = cmpi sle, %16, %17 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i32>
    %19 = andi %18, %13 {handshake.bb = 1 : ui32, handshake.name = "andi0"} : <i1>
    %trueResult_2, %falseResult_3 = cond_br %19, %14 {handshake.bb = 2 : ui32, handshake.name = "cond_br37"} : <i1>, <i32>
    %trueResult_4, %falseResult_5 = cond_br %19, %15 {handshake.bb = 2 : ui32, handshake.name = "cond_br38"} : <i1>, <i1>
    %trueResult_6, %falseResult_7 = cond_br %19, %8 {handshake.bb = 2 : ui32, handshake.name = "cond_br39"} : <i1>, <i32>
    %trueResult_8, %falseResult_9 = cond_br %19, %17 {handshake.bb = 2 : ui32, handshake.name = "cond_br40"} : <i1>, <i32>
    %trueResult_10, %falseResult_11 = cond_br %19, %16 {handshake.bb = 2 : ui32, handshake.name = "cond_br41"} : <i1>, <i32>
    %20 = source {handshake.bb = 2 : ui32, handshake.name = "source6"} : <>
    %21 = constant %20 {handshake.name = "constant7", value = 1 : i2} : <>, <i2>
    %22 = extsi %21 {handshake.name = "extsi2"} : <i2> to <i32>
    %23 = addi %trueResult_10, %trueResult_8 {handshake.bb = 2 : ui32, handshake.name = "addi0"} : <i32>
    %24 = shrsi %23, %22 {handshake.bb = 2 : ui32, handshake.name = "shrsi0"} : <i32>
    %25 = muli %24, %24 {handshake.bb = 2 : ui32, handshake.name = "muli0"} : <i32>
    %26 = cmpi eq, %25, %trueResult_6 {handshake.bb = 2 : ui32, handshake.name = "cmpi1"} : <i32>
    %27 = cmpi ne, %25, %trueResult_6 {handshake.bb = 2 : ui32, handshake.name = "cmpi2"} : <i32>
    %28 = andi %27, %trueResult_4 {handshake.bb = 2 : ui32, handshake.name = "andi1"} : <i1>
    %29 = select %26[%24, %trueResult_2] {handshake.bb = 2 : ui32, handshake.name = "select0"} : <i1>, <i32>
    %30 = cmpi slt, %25, %trueResult_6 {handshake.bb = 2 : ui32, handshake.name = "cmpi3"} : <i32>
    %trueResult_12, %falseResult_13 = cond_br %30, %24 {handshake.bb = 4 : ui32, handshake.name = "cond_br42"} : <i1>, <i32>
    %31 = source {handshake.bb = 4 : ui32, handshake.name = "source7"} : <>
    %32 = constant %31 {handshake.name = "constant8", value = 1 : i2} : <>, <i2>
    %33 = extsi %32 {handshake.name = "extsi3"} : <i2> to <i32>
    %34 = addi %trueResult_12, %33 {handshake.bb = 4 : ui32, handshake.name = "addi1"} : <i32>
    %35 = source {handshake.bb = 5 : ui32, handshake.name = "source8"} : <>
    %36 = constant %35 {handshake.bb = 5 : ui32, handshake.name = "constant6", value = -1 : i32} : <>, <i32>
    %37 = addi %falseResult_13, %36 {handshake.bb = 5 : ui32, handshake.name = "addi2"} : <i32>
    %38 = select %falseResult_5[%falseResult_9, %falseResult_3] {handshake.bb = 6 : ui32, handshake.name = "select1"} : <i1>, <i32>
    end {handshake.bb = 6 : ui32, handshake.name = "end0"} %38, %arg1 : <i32>, <>
  }
}

