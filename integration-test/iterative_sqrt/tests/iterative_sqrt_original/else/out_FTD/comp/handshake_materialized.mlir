module {
  handshake.func @iterative_sqrt(%arg0: !handshake.channel<i32>, %arg1: !handshake.control<>, ...) -> (!handshake.channel<i32>, !handshake.control<>) attributes {argNames = ["n", "start"], cfg.edges = "[0,1][2,1,3,cmpi5][4,1][1,2,6,andi0][3,4,5,cmpi6][5,1]", resNames = ["out0", "end"]} {
    %0:2 = fork [2] %arg1 {handshake.bb = 0 : ui32, handshake.name = "fork0"} : <>
    %1:2 = fork [2] %arg0 {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <i32>
    %2 = source {handshake.bb = 0 : ui32, handshake.name = "source0"} : <>
    %3 = constant %2 {handshake.bb = 0 : ui32, handshake.name = "constant2", value = false} : <>, <i1>
    %4 = extsi %3 {handshake.bb = 1 : ui32, handshake.name = "extsi0"} : <i1> to <i32>
    %5 = source {handshake.bb = 0 : ui32, handshake.name = "source1"} : <>
    %6 = constant %5 {handshake.bb = 0 : ui32, handshake.name = "constant3", value = true} : <>, <i1>
    %7:2 = fork [2] %6 {handshake.bb = 0 : ui32, handshake.name = "fork2"} : <i1>
    %trueResult, %falseResult = cond_br %36#6, %17 {handshake.bb = 1 : ui32, handshake.name = "cond_br39"} : <i1>, <i32>
    sink %falseResult {handshake.name = "sink0"} : <i32>
    %trueResult_0, %falseResult_1 = cond_br %36#7, %15 {handshake.bb = 1 : ui32, handshake.name = "cond_br40"} : <i1>, <i32>
    sink %falseResult_1 {handshake.name = "sink1"} : <i32>
    %trueResult_2, %falseResult_3 = cond_br %36#8, %50 {handshake.bb = 1 : ui32, handshake.name = "cond_br41"} : <i1>, <i1>
    %8 = constant %0#1 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "constant1", value = false} : <>, <i1>
    %9:2 = fork [2] %8 {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <i1>
    %10 = extsi %9#1 {handshake.bb = 1 : ui32, handshake.name = "extsi1"} : <i1> to <i32>
    %11 = merge %9#0, %36#9 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "merge0"} : <i1>
    %12:6 = fork [6] %11 {handshake.bb = 1 : ui32, handshake.name = "fork4"} : <i1>
    %13 = mux %12#5 [%1#1, %69#1] {ftd.regen, handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i32>, <i32>] to <i32>
    %14:2 = fork [2] %13 {handshake.bb = 1 : ui32, handshake.name = "fork5"} : <i32>
    %15 = mux %65#1 [%16, %25#3] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux4"} : <i1>, [<i32>, <i32>] to <i32>
    %16 = mux %71#1 [%25#4, %75] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux5"} : <i1>, [<i32>, <i32>] to <i32>
    %17 = mux %65#2 [%18, %27#3] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux6"} : <i1>, [<i32>, <i32>] to <i32>
    %18 = mux %71#2 [%78, %27#4] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux7"} : <i1>, [<i32>, <i32>] to <i32>
    %19 = mux %12#4 [%7#1, %68] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux8"} : <i1>, [<i1>, <i1>] to <i1>
    %20:3 = fork [3] %19 {handshake.bb = 1 : ui32, handshake.name = "fork6"} : <i1>
    %21 = mux %12#3 [%10, %67] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux9"} : <i1>, [<i32>, <i32>] to <i32>
    %22 = mux %12#2 [%7#0, %trueResult_2] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux10"} : <i1>, [<i1>, <i1>] to <i1>
    %23:3 = fork [3] %22 {handshake.bb = 1 : ui32, handshake.name = "fork7"} : <i1>
    %24 = mux %12#1 [%4, %trueResult_0] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux11"} : <i1>, [<i32>, <i32>] to <i32>
    %25:5 = fork [5] %24 {handshake.bb = 1 : ui32, handshake.name = "fork8"} : <i32>
    %26 = mux %12#0 [%1#0, %trueResult] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux12"} : <i1>, [<i32>, <i32>] to <i32>
    %27:5 = fork [5] %26 {handshake.bb = 1 : ui32, handshake.name = "fork9"} : <i32>
    %28 = source {handshake.bb = 1 : ui32, handshake.name = "source2"} : <>
    %29 = constant %28 {handshake.bb = 1 : ui32, handshake.name = "constant4", value = true} : <>, <i1>
    %30:2 = fork [2] %29 {handshake.bb = 1 : ui32, handshake.name = "fork10"} : <i1>
    %31 = source {handshake.bb = 1 : ui32, handshake.name = "source3"} : <>
    %32 = constant %31 {handshake.bb = 1 : ui32, handshake.name = "constant9", value = 1 : i2} : <>, <i2>
    %33 = extsi %32 {handshake.bb = 1 : ui32, handshake.name = "extsi2"} : <i2> to <i32>
    %34 = cmpi sle, %25#2, %27#2 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i32>
    %35 = andi %34, %20#2 {handshake.bb = 1 : ui32, handshake.name = "andi0"} : <i1>
    %36:10 = fork [10] %35 {handshake.bb = 1 : ui32, handshake.name = "fork11"} : <i1>
    %37 = addi %25#1, %27#1 {handshake.bb = 1 : ui32, handshake.name = "addi0"} : <i32>
    %38 = shrsi %37, %33 {handshake.bb = 1 : ui32, handshake.name = "shrsi0"} : <i32>
    %39:2 = fork [2] %38 {handshake.bb = 1 : ui32, handshake.name = "fork12"} : <i32>
    %40 = muli %39#0, %39#1 {handshake.bb = 1 : ui32, handshake.name = "muli0"} : <i32>
    %41 = cmpi ne, %40, %14#1 {handshake.bb = 1 : ui32, handshake.name = "cmpi1"} : <i32>
    %42 = andi %41, %23#2 {handshake.bb = 1 : ui32, handshake.name = "andi1"} : <i1>
    %43 = xori %20#1, %30#1 {handshake.bb = 1 : ui32, handshake.name = "xori0"} : <i1>
    %44 = andi %20#0, %42 {handshake.bb = 1 : ui32, handshake.name = "andi2"} : <i1>
    %45 = andi %43, %23#1 {handshake.bb = 1 : ui32, handshake.name = "andi3"} : <i1>
    %46 = ori %44, %45 {handshake.bb = 1 : ui32, handshake.name = "ori0"} : <i1>
    %47 = xori %36#5, %30#0 {handshake.bb = 1 : ui32, handshake.name = "xori1"} : <i1>
    %48 = andi %36#4, %46 {handshake.bb = 1 : ui32, handshake.name = "andi4"} : <i1>
    %49 = andi %47, %23#0 {handshake.bb = 1 : ui32, handshake.name = "andi5"} : <i1>
    %50 = ori %48, %49 {handshake.bb = 1 : ui32, handshake.name = "ori1"} : <i1>
    %trueResult_4, %falseResult_5 = cond_br %36#3, %21 {handshake.bb = 2 : ui32, handshake.name = "cond_br42"} : <i1>, <i32>
    %trueResult_6, %falseResult_7 = cond_br %36#2, %25#0 {handshake.bb = 2 : ui32, handshake.name = "cond_br43"} : <i1>, <i32>
    sink %falseResult_7 {handshake.name = "sink2"} : <i32>
    %51:3 = fork [3] %trueResult_6 {handshake.bb = 2 : ui32, handshake.name = "fork13"} : <i32>
    %trueResult_8, %falseResult_9 = cond_br %36#1, %27#0 {handshake.bb = 2 : ui32, handshake.name = "cond_br44"} : <i1>, <i32>
    %52:3 = fork [3] %trueResult_8 {handshake.bb = 2 : ui32, handshake.name = "fork14"} : <i32>
    %53 = source {handshake.bb = 2 : ui32, handshake.name = "source8"} : <>
    %54 = constant %53 {handshake.bb = 2 : ui32, handshake.name = "constant10", value = 1 : i2} : <>, <i2>
    %55 = extsi %54 {handshake.bb = 2 : ui32, handshake.name = "extsi3"} : <i2> to <i32>
    %56 = addi %51#2, %52#2 {handshake.bb = 2 : ui32, handshake.name = "addi1"} : <i32>
    %57 = shrsi %56, %55 {handshake.bb = 2 : ui32, handshake.name = "shrsi1"} : <i32>
    %58:4 = fork [4] %57 {handshake.bb = 2 : ui32, handshake.name = "fork15"} : <i32>
    %59 = muli %58#2, %58#3 {handshake.bb = 2 : ui32, handshake.name = "muli1"} : <i32>
    %60:3 = fork [3] %59 {handshake.bb = 2 : ui32, handshake.name = "fork16"} : <i32>
    %61 = cmpi ne, %60#2, %69#2 {handshake.bb = 2 : ui32, handshake.name = "cmpi2"} : <i32>
    %62 = cmpi sle, %51#1, %52#1 {handshake.bb = 2 : ui32, handshake.name = "cmpi3"} : <i32>
    %63 = cmpi sgt, %51#0, %52#0 {handshake.bb = 2 : ui32, handshake.name = "cmpi4"} : <i32>
    %64 = cmpi eq, %60#1, %69#3 {handshake.bb = 2 : ui32, handshake.name = "cmpi5"} : <i32>
    %65:3 = fork [3] %64 {handshake.bb = 2 : ui32, handshake.name = "fork17"} : <i1>
    %66 = andi %62, %61 {handshake.bb = 2 : ui32, handshake.name = "andi6"} : <i1>
    %67 = select %65#0[%58#1, %trueResult_4] {handshake.bb = 2 : ui32, handshake.name = "select0"} : <i1>, <i32>
    %68 = ori %66, %63 {handshake.bb = 2 : ui32, handshake.name = "ori2"} : <i1>
    %trueResult_10, %falseResult_11 = cond_br %36#0, %14#0 {handshake.bb = 3 : ui32, handshake.name = "cond_br45"} : <i1>, <i32>
    sink %falseResult_11 {handshake.name = "sink3"} : <i32>
    %69:4 = fork [4] %trueResult_10 {handshake.bb = 3 : ui32, handshake.name = "fork18"} : <i32>
    %70 = cmpi slt, %60#0, %69#0 {handshake.bb = 3 : ui32, handshake.name = "cmpi6"} : <i32>
    %71:3 = fork [3] %70 {handshake.bb = 3 : ui32, handshake.name = "fork19"} : <i1>
    %trueResult_12, %falseResult_13 = cond_br %71#0, %58#0 {handshake.bb = 4 : ui32, handshake.name = "cond_br46"} : <i1>, <i32>
    %72 = source {handshake.bb = 4 : ui32, handshake.name = "source9"} : <>
    %73 = constant %72 {handshake.bb = 4 : ui32, handshake.name = "constant11", value = 1 : i2} : <>, <i2>
    %74 = extsi %73 {handshake.bb = 4 : ui32, handshake.name = "extsi4"} : <i2> to <i32>
    %75 = addi %trueResult_12, %74 {handshake.bb = 4 : ui32, handshake.name = "addi2"} : <i32>
    %76 = source {handshake.bb = 5 : ui32, handshake.name = "source10"} : <>
    %77 = constant %76 {handshake.bb = 5 : ui32, handshake.name = "constant8", value = -1 : i32} : <>, <i32>
    %78 = addi %falseResult_13, %77 {handshake.bb = 5 : ui32, handshake.name = "addi3"} : <i32>
    %79 = select %falseResult_3[%falseResult_9, %falseResult_5] {handshake.bb = 6 : ui32, handshake.name = "select1"} : <i1>, <i32>
    end {handshake.bb = 6 : ui32, handshake.name = "end0"} %79, %0#0 : <i32>, <>
  }
}

