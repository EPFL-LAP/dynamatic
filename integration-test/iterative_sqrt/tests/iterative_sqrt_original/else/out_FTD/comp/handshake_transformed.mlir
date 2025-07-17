module {
  handshake.func @iterative_sqrt(%arg0: !handshake.channel<i32>, %arg1: !handshake.control<>, ...) -> (!handshake.channel<i32>, !handshake.control<>) attributes {argNames = ["n", "start"], cfg.edges = "[0,1][2,1,3,cmpi5][4,1][1,2,6,andi0][3,4,5,cmpi6][5,1]", resNames = ["out0", "end"]} {
    %0 = source {handshake.bb = 0 : ui32, handshake.name = "source0"} : <>
    %1 = constant %0 {handshake.name = "constant2", value = false} : <>, <i1>
    %2 = extsi %1 {handshake.name = "extsi0"} : <i1> to <i32>
    %3 = source {handshake.bb = 0 : ui32, handshake.name = "source1"} : <>
    %4 = constant %3 {handshake.bb = 0 : ui32, handshake.name = "constant3", value = true} : <>, <i1>
    %trueResult, %falseResult = cond_br %24, %11 {handshake.bb = 1 : ui32, handshake.name = "cond_br39"} : <i1>, <i32>
    %trueResult_0, %falseResult_1 = cond_br %24, %9 {handshake.bb = 1 : ui32, handshake.name = "cond_br40"} : <i1>, <i32>
    %trueResult_2, %falseResult_3 = cond_br %24, %37 {handshake.bb = 1 : ui32, handshake.name = "cond_br41"} : <i1>, <i1>
    %5 = constant %arg1 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "constant1", value = false} : <>, <i1>
    %6 = extsi %5 {handshake.bb = 1 : ui32, handshake.name = "extsi1"} : <i1> to <i32>
    %7 = merge %5, %24 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "merge0"} : <i1>
    %8 = mux %7 [%arg0, %trueResult_10] {ftd.regen, handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i32>, <i32>] to <i32>
    %9 = mux %47 [%10, %16] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux4"} : <i1>, [<i32>, <i32>] to <i32>
    %10 = mux %51 [%16, %55] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux5"} : <i1>, [<i32>, <i32>] to <i32>
    %11 = mux %47 [%12, %17] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux6"} : <i1>, [<i32>, <i32>] to <i32>
    %12 = mux %51 [%58, %17] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux7"} : <i1>, [<i32>, <i32>] to <i32>
    %13 = mux %7 [%4, %50] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux8"} : <i1>, [<i1>, <i1>] to <i1>
    %14 = mux %7 [%6, %49] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux9"} : <i1>, [<i32>, <i32>] to <i32>
    %15 = mux %7 [%4, %trueResult_2] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux10"} : <i1>, [<i1>, <i1>] to <i1>
    %16 = mux %7 [%2, %trueResult_0] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux11"} : <i1>, [<i32>, <i32>] to <i32>
    %17 = mux %7 [%arg0, %trueResult] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux12"} : <i1>, [<i32>, <i32>] to <i32>
    %18 = source {handshake.bb = 1 : ui32, handshake.name = "source2"} : <>
    %19 = constant %18 {handshake.bb = 1 : ui32, handshake.name = "constant4", value = true} : <>, <i1>
    %20 = source {handshake.bb = 1 : ui32, handshake.name = "source3"} : <>
    %21 = constant %20 {handshake.name = "constant9", value = 1 : i2} : <>, <i2>
    %22 = extsi %21 {handshake.name = "extsi2"} : <i2> to <i32>
    %23 = cmpi sle, %16, %17 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i32>
    %24 = andi %23, %13 {handshake.bb = 1 : ui32, handshake.name = "andi0"} : <i1>
    %25 = addi %16, %17 {handshake.bb = 1 : ui32, handshake.name = "addi0"} : <i32>
    %26 = shrsi %25, %22 {handshake.bb = 1 : ui32, handshake.name = "shrsi0"} : <i32>
    %27 = muli %26, %26 {handshake.bb = 1 : ui32, handshake.name = "muli0"} : <i32>
    %28 = cmpi ne, %27, %8 {handshake.bb = 1 : ui32, handshake.name = "cmpi1"} : <i32>
    %29 = andi %28, %15 {handshake.bb = 1 : ui32, handshake.name = "andi1"} : <i1>
    %30 = xori %13, %19 {handshake.bb = 1 : ui32, handshake.name = "xori0"} : <i1>
    %31 = andi %13, %29 {handshake.bb = 1 : ui32, handshake.name = "andi2"} : <i1>
    %32 = andi %30, %15 {handshake.bb = 1 : ui32, handshake.name = "andi3"} : <i1>
    %33 = ori %31, %32 {handshake.bb = 1 : ui32, handshake.name = "ori0"} : <i1>
    %34 = xori %24, %19 {handshake.bb = 1 : ui32, handshake.name = "xori1"} : <i1>
    %35 = andi %24, %33 {handshake.bb = 1 : ui32, handshake.name = "andi4"} : <i1>
    %36 = andi %34, %15 {handshake.bb = 1 : ui32, handshake.name = "andi5"} : <i1>
    %37 = ori %35, %36 {handshake.bb = 1 : ui32, handshake.name = "ori1"} : <i1>
    %trueResult_4, %falseResult_5 = cond_br %24, %14 {handshake.bb = 2 : ui32, handshake.name = "cond_br42"} : <i1>, <i32>
    %trueResult_6, %falseResult_7 = cond_br %24, %16 {handshake.bb = 2 : ui32, handshake.name = "cond_br43"} : <i1>, <i32>
    %trueResult_8, %falseResult_9 = cond_br %24, %17 {handshake.bb = 2 : ui32, handshake.name = "cond_br44"} : <i1>, <i32>
    %38 = source {handshake.bb = 2 : ui32, handshake.name = "source8"} : <>
    %39 = constant %38 {handshake.name = "constant10", value = 1 : i2} : <>, <i2>
    %40 = extsi %39 {handshake.name = "extsi3"} : <i2> to <i32>
    %41 = addi %trueResult_6, %trueResult_8 {handshake.bb = 2 : ui32, handshake.name = "addi1"} : <i32>
    %42 = shrsi %41, %40 {handshake.bb = 2 : ui32, handshake.name = "shrsi1"} : <i32>
    %43 = muli %42, %42 {handshake.bb = 2 : ui32, handshake.name = "muli1"} : <i32>
    %44 = cmpi ne, %43, %trueResult_10 {handshake.bb = 2 : ui32, handshake.name = "cmpi2"} : <i32>
    %45 = cmpi sle, %trueResult_6, %trueResult_8 {handshake.bb = 2 : ui32, handshake.name = "cmpi3"} : <i32>
    %46 = cmpi sgt, %trueResult_6, %trueResult_8 {handshake.bb = 2 : ui32, handshake.name = "cmpi4"} : <i32>
    %47 = cmpi eq, %43, %trueResult_10 {handshake.bb = 2 : ui32, handshake.name = "cmpi5"} : <i32>
    %48 = andi %45, %44 {handshake.bb = 2 : ui32, handshake.name = "andi6"} : <i1>
    %49 = select %47[%42, %trueResult_4] {handshake.bb = 2 : ui32, handshake.name = "select0"} : <i1>, <i32>
    %50 = ori %48, %46 {handshake.bb = 2 : ui32, handshake.name = "ori2"} : <i1>
    %trueResult_10, %falseResult_11 = cond_br %24, %8 {handshake.bb = 3 : ui32, handshake.name = "cond_br45"} : <i1>, <i32>
    %51 = cmpi slt, %43, %trueResult_10 {handshake.bb = 3 : ui32, handshake.name = "cmpi6"} : <i32>
    %trueResult_12, %falseResult_13 = cond_br %51, %42 {handshake.bb = 4 : ui32, handshake.name = "cond_br46"} : <i1>, <i32>
    %52 = source {handshake.bb = 4 : ui32, handshake.name = "source9"} : <>
    %53 = constant %52 {handshake.name = "constant11", value = 1 : i2} : <>, <i2>
    %54 = extsi %53 {handshake.name = "extsi4"} : <i2> to <i32>
    %55 = addi %trueResult_12, %54 {handshake.bb = 4 : ui32, handshake.name = "addi2"} : <i32>
    %56 = source {handshake.bb = 5 : ui32, handshake.name = "source10"} : <>
    %57 = constant %56 {handshake.bb = 5 : ui32, handshake.name = "constant8", value = -1 : i32} : <>, <i32>
    %58 = addi %falseResult_13, %57 {handshake.bb = 5 : ui32, handshake.name = "addi3"} : <i32>
    %59 = select %falseResult_3[%falseResult_9, %falseResult_5] {handshake.bb = 6 : ui32, handshake.name = "select1"} : <i1>, <i32>
    end {handshake.bb = 6 : ui32, handshake.name = "end0"} %59, %arg1 : <i32>, <>
  }
}

