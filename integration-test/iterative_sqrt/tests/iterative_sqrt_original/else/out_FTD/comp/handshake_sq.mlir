module {
  handshake.func @iterative_sqrt(%arg0: !handshake.channel<i32>, %arg1: !handshake.control<>, ...) -> (!handshake.channel<i32>, !handshake.control<>) attributes {argNames = ["n", "start"], cfg.edges = "[0,1][2,1,3,cmpi5][4,1][1,2,6,andi0][3,4,5,cmpi6][5,1]", resNames = ["out0", "end"]} {
    %0 = source {handshake.bb = 0 : ui32, handshake.name = "source0"} : <>
    %1 = constant %0 {handshake.bb = 0 : ui32, handshake.name = "constant0", value = 0 : i32} : <>, <i32>
    %2 = source {handshake.bb = 0 : ui32, handshake.name = "source1"} : <>
    %3 = constant %2 {handshake.bb = 0 : ui32, handshake.name = "constant3", value = true} : <>, <i1>
    %4 = constant %arg1 {handshake.bb = 0 : ui32, handshake.name = "mlir.undef0", value = 0 : i32} : <>, <i32>
    %trueResult, %falseResult = cond_br %22, %10 {handshake.bb = 1 : ui32, handshake.name = "cond_br39"} : <i1>, <i32>
    %trueResult_0, %falseResult_1 = cond_br %22, %8 {handshake.bb = 1 : ui32, handshake.name = "cond_br40"} : <i1>, <i32>
    %trueResult_2, %falseResult_3 = cond_br %22, %35 {handshake.bb = 1 : ui32, handshake.name = "cond_br41"} : <i1>, <i1>
    %5 = constant %arg1 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "constant1", value = false} : <>, <i1>
    %6 = merge %5, %22 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "merge0"} : <i1>
    %7 = mux %6 [%arg0, %trueResult_10] {ftd.regen, handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i32>, <i32>] to <i32>
    %8 = mux %44 [%9, %15] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux4"} : <i1>, [<i32>, <i32>] to <i32>
    %9 = mux %48 [%15, %51] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux5"} : <i1>, [<i32>, <i32>] to <i32>
    %10 = mux %44 [%11, %16] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux6"} : <i1>, [<i32>, <i32>] to <i32>
    %11 = mux %48 [%54, %16] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux7"} : <i1>, [<i32>, <i32>] to <i32>
    %12 = mux %6 [%3, %47] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux8"} : <i1>, [<i1>, <i1>] to <i1>
    %13 = mux %6 [%4, %46] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux9"} : <i1>, [<i32>, <i32>] to <i32>
    %14 = mux %6 [%3, %trueResult_2] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux10"} : <i1>, [<i1>, <i1>] to <i1>
    %15 = mux %6 [%1, %trueResult_0] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux11"} : <i1>, [<i32>, <i32>] to <i32>
    %16 = mux %6 [%arg0, %trueResult] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux12"} : <i1>, [<i32>, <i32>] to <i32>
    %17 = source {handshake.bb = 1 : ui32, handshake.name = "source2"} : <>
    %18 = constant %17 {handshake.bb = 1 : ui32, handshake.name = "constant4", value = true} : <>, <i1>
    %19 = source {handshake.bb = 1 : ui32, handshake.name = "source3"} : <>
    %20 = constant %19 {handshake.bb = 1 : ui32, handshake.name = "constant5", value = 1 : i32} : <>, <i32>
    %21 = cmpi sle, %15, %16 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i32>
    %22 = andi %21, %12 {handshake.bb = 1 : ui32, handshake.name = "andi0"} : <i1>
    %23 = addi %15, %16 {handshake.bb = 1 : ui32, handshake.name = "addi0"} : <i32>
    %24 = shrsi %23, %20 {handshake.bb = 1 : ui32, handshake.name = "shrsi0"} : <i32>
    %25 = muli %24, %24 {handshake.bb = 1 : ui32, handshake.name = "muli0"} : <i32>
    %26 = cmpi ne, %25, %7 {handshake.bb = 1 : ui32, handshake.name = "cmpi1"} : <i32>
    %27 = andi %26, %14 {handshake.bb = 1 : ui32, handshake.name = "andi1"} : <i1>
    %28 = xori %12, %18 {handshake.bb = 1 : ui32, handshake.name = "xori0"} : <i1>
    %29 = andi %12, %27 {handshake.bb = 1 : ui32, handshake.name = "andi2"} : <i1>
    %30 = andi %28, %14 {handshake.bb = 1 : ui32, handshake.name = "andi3"} : <i1>
    %31 = ori %29, %30 {handshake.bb = 1 : ui32, handshake.name = "ori0"} : <i1>
    %32 = xori %22, %18 {handshake.bb = 1 : ui32, handshake.name = "xori1"} : <i1>
    %33 = andi %22, %31 {handshake.bb = 1 : ui32, handshake.name = "andi4"} : <i1>
    %34 = andi %32, %14 {handshake.bb = 1 : ui32, handshake.name = "andi5"} : <i1>
    %35 = ori %33, %34 {handshake.bb = 1 : ui32, handshake.name = "ori1"} : <i1>
    %trueResult_4, %falseResult_5 = cond_br %22, %13 {handshake.bb = 2 : ui32, handshake.name = "cond_br42"} : <i1>, <i32>
    %trueResult_6, %falseResult_7 = cond_br %22, %15 {handshake.bb = 2 : ui32, handshake.name = "cond_br43"} : <i1>, <i32>
    %trueResult_8, %falseResult_9 = cond_br %22, %16 {handshake.bb = 2 : ui32, handshake.name = "cond_br44"} : <i1>, <i32>
    %36 = source {handshake.bb = 2 : ui32, handshake.name = "source8"} : <>
    %37 = constant %36 {handshake.bb = 2 : ui32, handshake.name = "constant6", value = 1 : i32} : <>, <i32>
    %38 = addi %trueResult_6, %trueResult_8 {handshake.bb = 2 : ui32, handshake.name = "addi1"} : <i32>
    %39 = shrsi %38, %37 {handshake.bb = 2 : ui32, handshake.name = "shrsi1"} : <i32>
    %40 = muli %39, %39 {handshake.bb = 2 : ui32, handshake.name = "muli1"} : <i32>
    %41 = cmpi ne, %40, %trueResult_10 {handshake.bb = 2 : ui32, handshake.name = "cmpi2"} : <i32>
    %42 = cmpi sle, %trueResult_6, %trueResult_8 {handshake.bb = 2 : ui32, handshake.name = "cmpi3"} : <i32>
    %43 = cmpi sgt, %trueResult_6, %trueResult_8 {handshake.bb = 2 : ui32, handshake.name = "cmpi4"} : <i32>
    %44 = cmpi eq, %40, %trueResult_10 {handshake.bb = 2 : ui32, handshake.name = "cmpi5"} : <i32>
    %45 = andi %42, %41 {handshake.bb = 2 : ui32, handshake.name = "andi6"} : <i1>
    %46 = select %44[%39, %trueResult_4] {handshake.bb = 2 : ui32, handshake.name = "select0"} : <i1>, <i32>
    %47 = ori %45, %43 {handshake.bb = 2 : ui32, handshake.name = "ori2"} : <i1>
    %trueResult_10, %falseResult_11 = cond_br %22, %7 {handshake.bb = 3 : ui32, handshake.name = "cond_br45"} : <i1>, <i32>
    %48 = cmpi slt, %40, %trueResult_10 {handshake.bb = 3 : ui32, handshake.name = "cmpi6"} : <i32>
    %trueResult_12, %falseResult_13 = cond_br %48, %39 {handshake.bb = 4 : ui32, handshake.name = "cond_br46"} : <i1>, <i32>
    %49 = source {handshake.bb = 4 : ui32, handshake.name = "source9"} : <>
    %50 = constant %49 {handshake.bb = 4 : ui32, handshake.name = "constant7", value = 1 : i32} : <>, <i32>
    %51 = addi %trueResult_12, %50 {handshake.bb = 4 : ui32, handshake.name = "addi2"} : <i32>
    %52 = source {handshake.bb = 5 : ui32, handshake.name = "source10"} : <>
    %53 = constant %52 {handshake.bb = 5 : ui32, handshake.name = "constant8", value = -1 : i32} : <>, <i32>
    %54 = addi %falseResult_13, %53 {handshake.bb = 5 : ui32, handshake.name = "addi3"} : <i32>
    %55 = select %falseResult_3[%falseResult_9, %falseResult_5] {handshake.bb = 6 : ui32, handshake.name = "select1"} : <i1>, <i32>
    end {handshake.bb = 6 : ui32, handshake.name = "end0"} %55, %arg1 : <i32>, <>
  }
}

