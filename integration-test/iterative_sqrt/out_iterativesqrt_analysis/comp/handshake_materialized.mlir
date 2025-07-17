module {
  handshake.func @iterative_sqrt(%arg0: !handshake.channel<i32>, %arg1: !handshake.control<>, ...) -> (!handshake.channel<i32>, !handshake.control<>) attributes {argNames = ["n", "start"], cfg.edges = "[0,1][2,3,1,cmpi2][4,1][1,2,6,andi0][3,4,5,cmpi3][5,1]", resNames = ["out0", "end"]} {
    %0:2 = fork [2] %arg1 {handshake.bb = 0 : ui32, handshake.name = "fork0"} : <>
    %1:2 = fork [2] %arg0 {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <i32>
    %2 = source {handshake.bb = 0 : ui32, handshake.name = "source0"} : <>
    %3 = constant %2 {handshake.bb = 0 : ui32, handshake.name = "constant2", value = false} : <>, <i1>
    %4 = extsi %3 {handshake.bb = 1 : ui32, handshake.name = "extsi0"} : <i1> to <i32>
    %5 = source {handshake.bb = 0 : ui32, handshake.name = "source1"} : <>
    %6 = constant %5 {handshake.bb = 0 : ui32, handshake.name = "constant3", value = true} : <>, <i1>
    %7:2 = fork [2] %6 {handshake.bb = 0 : ui32, handshake.name = "fork2"} : <i1>
    %trueResult, %falseResult = cond_br %27#5, %16 {handshake.bb = 1 : ui32, handshake.name = "cond_br35"} : <i1>, <i32>
    sink %falseResult {handshake.name = "sink0"} : <i32>
    %trueResult_0, %falseResult_1 = cond_br %27#6, %14 {handshake.bb = 1 : ui32, handshake.name = "cond_br36"} : <i1>, <i32>
    sink %falseResult_1 {handshake.name = "sink1"} : <i32>
    %8 = constant %0#1 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "constant1", value = false} : <>, <i1>
    %9:2 = fork [2] %8 {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <i1>
    %10 = extsi %9#1 {handshake.bb = 1 : ui32, handshake.name = "extsi1"} : <i1> to <i32>
    %11 = merge %9#0, %27#7 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "merge0"} : <i1>
    %12:6 = fork [6] %11 {handshake.bb = 1 : ui32, handshake.name = "fork4"} : <i1>
    %13 = mux %12#5 [%1#1, %28#3] {ftd.regen, handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i32>, <i32>] to <i32>
    %14 = mux %39#1 [%22#2, %15] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux3"} : <i1>, [<i32>, <i32>] to <i32>
    %15 = mux %43#1 [%22#3, %47] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux4"} : <i1>, [<i32>, <i32>] to <i32>
    %16 = mux %39#2 [%24#2, %17] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux5"} : <i1>, [<i32>, <i32>] to <i32>
    %17 = mux %43#2 [%50, %24#3] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux6"} : <i1>, [<i32>, <i32>] to <i32>
    %18 = mux %12#4 [%7#1, %39#3] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux7"} : <i1>, [<i1>, <i1>] to <i1>
    %19 = mux %12#3 [%10, %41] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux8"} : <i1>, [<i32>, <i32>] to <i32>
    %20 = mux %12#2 [%7#0, %40] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux9"} : <i1>, [<i1>, <i1>] to <i1>
    %21 = mux %12#1 [%4, %trueResult_0] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux10"} : <i1>, [<i32>, <i32>] to <i32>
    %22:4 = fork [4] %21 {handshake.bb = 1 : ui32, handshake.name = "fork5"} : <i32>
    %23 = mux %12#0 [%1#0, %trueResult] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux11"} : <i1>, [<i32>, <i32>] to <i32>
    %24:4 = fork [4] %23 {handshake.bb = 1 : ui32, handshake.name = "fork6"} : <i32>
    %25 = cmpi sle, %22#1, %24#1 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i32>
    %26 = andi %25, %18 {handshake.bb = 1 : ui32, handshake.name = "andi0"} : <i1>
    %27:8 = fork [8] %26 {handshake.bb = 1 : ui32, handshake.name = "fork7"} : <i1>
    %trueResult_2, %falseResult_3 = cond_br %27#4, %19 {handshake.bb = 2 : ui32, handshake.name = "cond_br37"} : <i1>, <i32>
    %trueResult_4, %falseResult_5 = cond_br %27#3, %20 {handshake.bb = 2 : ui32, handshake.name = "cond_br38"} : <i1>, <i1>
    %trueResult_6, %falseResult_7 = cond_br %27#2, %13 {handshake.bb = 2 : ui32, handshake.name = "cond_br39"} : <i1>, <i32>
    sink %falseResult_7 {handshake.name = "sink2"} : <i32>
    %28:4 = fork [4] %trueResult_6 {handshake.bb = 2 : ui32, handshake.name = "fork8"} : <i32>
    %trueResult_8, %falseResult_9 = cond_br %27#1, %24#0 {handshake.bb = 2 : ui32, handshake.name = "cond_br40"} : <i1>, <i32>
    %trueResult_10, %falseResult_11 = cond_br %27#0, %22#0 {handshake.bb = 2 : ui32, handshake.name = "cond_br41"} : <i1>, <i32>
    sink %falseResult_11 {handshake.name = "sink3"} : <i32>
    %29 = source {handshake.bb = 2 : ui32, handshake.name = "source6"} : <>
    %30 = constant %29 {handshake.bb = 2 : ui32, handshake.name = "constant7", value = 1 : i2} : <>, <i2>
    %31 = extsi %30 {handshake.bb = 2 : ui32, handshake.name = "extsi2"} : <i2> to <i32>
    %32 = addi %trueResult_10, %trueResult_8 {handshake.bb = 2 : ui32, handshake.name = "addi0"} : <i32>
    %33 = shrsi %32, %31 {handshake.bb = 2 : ui32, handshake.name = "shrsi0"} : <i32>
    %34:4 = fork [4] %33 {handshake.bb = 2 : ui32, handshake.name = "fork9"} : <i32>
    %35 = muli %34#2, %34#3 {handshake.bb = 2 : ui32, handshake.name = "muli0"} : <i32>
    %36:3 = fork [3] %35 {handshake.bb = 2 : ui32, handshake.name = "fork10"} : <i32>
    %37 = cmpi eq, %36#2, %28#2 {handshake.bb = 2 : ui32, handshake.name = "cmpi1"} : <i32>
    %38 = cmpi ne, %36#1, %28#1 {handshake.bb = 2 : ui32, handshake.name = "cmpi2"} : <i32>
    %39:4 = fork [4] %38 {handshake.bb = 2 : ui32, handshake.name = "fork11"} : <i1>
    %40 = andi %39#0, %trueResult_4 {handshake.bb = 2 : ui32, handshake.name = "andi1"} : <i1>
    %41 = select %37[%34#1, %trueResult_2] {handshake.bb = 2 : ui32, handshake.name = "select0"} : <i1>, <i32>
    %42 = cmpi slt, %36#0, %28#0 {handshake.bb = 2 : ui32, handshake.name = "cmpi3"} : <i32>
    %43:3 = fork [3] %42 {handshake.bb = 2 : ui32, handshake.name = "fork12"} : <i1>
    %trueResult_12, %falseResult_13 = cond_br %43#0, %34#0 {handshake.bb = 4 : ui32, handshake.name = "cond_br42"} : <i1>, <i32>
    %44 = source {handshake.bb = 4 : ui32, handshake.name = "source7"} : <>
    %45 = constant %44 {handshake.bb = 4 : ui32, handshake.name = "constant8", value = 1 : i2} : <>, <i2>
    %46 = extsi %45 {handshake.bb = 4 : ui32, handshake.name = "extsi3"} : <i2> to <i32>
    %47 = addi %trueResult_12, %46 {handshake.bb = 4 : ui32, handshake.name = "addi1"} : <i32>
    %48 = source {handshake.bb = 5 : ui32, handshake.name = "source8"} : <>
    %49 = constant %48 {handshake.bb = 5 : ui32, handshake.name = "constant6", value = -1 : i32} : <>, <i32>
    %50 = addi %falseResult_13, %49 {handshake.bb = 5 : ui32, handshake.name = "addi2"} : <i32>
    %51 = select %falseResult_5[%falseResult_9, %falseResult_3] {handshake.bb = 6 : ui32, handshake.name = "select1"} : <i1>, <i32>
    end {handshake.bb = 6 : ui32, handshake.name = "end0"} %51, %0#0 : <i32>, <>
  }
}

