module {
  handshake.func @iterative_sqrt(%arg0: !handshake.channel<i32>, %arg1: !handshake.control<>, ...) -> (!handshake.channel<i32>, !handshake.control<>) attributes {argNames = ["n", "start"], cfg.edges = "[0,1][2,1,3,cmpi5][4,1][1,2,6,andi0][3,4,5,cmpi6][5,1]", resNames = ["out0", "end"]} {
    %0 = source {handshake.bb = 0 : ui32, handshake.name = "source0"} : <>
    %1 = constant %0 {handshake.bb = 0 : ui32, handshake.name = "constant0", value = 0 : i32} : <>, <i32>
    %2 = source {handshake.bb = 0 : ui32, handshake.name = "source1"} : <>
    %3 = constant %2 {handshake.bb = 0 : ui32, handshake.name = "constant3", value = true} : <>, <i1>
    %4 = constant %arg1 {handshake.bb = 0 : ui32, handshake.name = "mlir.undef0", value = 0 : i32} : <>, <i32>
    %5 = br %arg1 {handshake.bb = 0 : ui32, handshake.name = "br2"} : <>
    %trueResult, %falseResult = cond_br %23, %11 {handshake.bb = 1 : ui32, handshake.name = "cond_br39"} : <i1>, <i32>
    %trueResult_0, %falseResult_1 = cond_br %23, %9 {handshake.bb = 1 : ui32, handshake.name = "cond_br40"} : <i1>, <i32>
    %trueResult_2, %falseResult_3 = cond_br %23, %36 {handshake.bb = 1 : ui32, handshake.name = "cond_br41"} : <i1>, <i1>
    %6 = constant %arg1 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "constant1", value = false} : <>, <i1>
    %7 = merge %6, %23 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "merge0"} : <i1>
    %8 = mux %7 [%arg0, %trueResult_16] {ftd.regen, handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i32>, <i32>] to <i32>
    %result, %index = control_merge [%5, %trueResult_14, %53, %57]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>, <>, <>] to <>, <i2>
    %9 = mux %45 [%10, %16] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux4"} : <i1>, [<i32>, <i32>] to <i32>
    %10 = mux %49 [%16, %52] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux5"} : <i1>, [<i32>, <i32>] to <i32>
    %11 = mux %45 [%12, %17] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux6"} : <i1>, [<i32>, <i32>] to <i32>
    %12 = mux %49 [%56, %17] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux7"} : <i1>, [<i32>, <i32>] to <i32>
    %13 = mux %7 [%3, %48] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux8"} : <i1>, [<i1>, <i1>] to <i1>
    %14 = mux %7 [%4, %47] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux9"} : <i1>, [<i32>, <i32>] to <i32>
    %15 = mux %7 [%3, %trueResult_2] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux10"} : <i1>, [<i1>, <i1>] to <i1>
    %16 = mux %7 [%1, %trueResult_0] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux11"} : <i1>, [<i32>, <i32>] to <i32>
    %17 = mux %7 [%arg0, %trueResult] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux12"} : <i1>, [<i32>, <i32>] to <i32>
    %18 = source {handshake.bb = 1 : ui32, handshake.name = "source2"} : <>
    %19 = constant %18 {handshake.bb = 1 : ui32, handshake.name = "constant4", value = true} : <>, <i1>
    %20 = source {handshake.bb = 1 : ui32, handshake.name = "source3"} : <>
    %21 = constant %20 {handshake.bb = 1 : ui32, handshake.name = "constant5", value = 1 : i32} : <>, <i32>
    %22 = cmpi sle, %16, %17 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i32>
    %23 = andi %22, %13 {handshake.bb = 1 : ui32, handshake.name = "andi0"} : <i1>
    %24 = addi %16, %17 {handshake.bb = 1 : ui32, handshake.name = "addi0"} : <i32>
    %25 = shrsi %24, %21 {handshake.bb = 1 : ui32, handshake.name = "shrsi0"} : <i32>
    %26 = muli %25, %25 {handshake.bb = 1 : ui32, handshake.name = "muli0"} : <i32>
    %27 = cmpi ne, %26, %8 {handshake.bb = 1 : ui32, handshake.name = "cmpi1"} : <i32>
    %28 = andi %27, %15 {handshake.bb = 1 : ui32, handshake.name = "andi1"} : <i1>
    %29 = xori %13, %19 {handshake.bb = 1 : ui32, handshake.name = "xori0"} : <i1>
    %30 = andi %13, %28 {handshake.bb = 1 : ui32, handshake.name = "andi2"} : <i1>
    %31 = andi %29, %15 {handshake.bb = 1 : ui32, handshake.name = "andi3"} : <i1>
    %32 = ori %30, %31 {handshake.bb = 1 : ui32, handshake.name = "ori0"} : <i1>
    %33 = xori %23, %19 {handshake.bb = 1 : ui32, handshake.name = "xori1"} : <i1>
    %34 = andi %23, %32 {handshake.bb = 1 : ui32, handshake.name = "andi4"} : <i1>
    %35 = andi %33, %15 {handshake.bb = 1 : ui32, handshake.name = "andi5"} : <i1>
    %36 = ori %34, %35 {handshake.bb = 1 : ui32, handshake.name = "ori1"} : <i1>
    %trueResult_4, %falseResult_5 = cond_br %23, %result {handshake.bb = 1 : ui32, handshake.name = "cond_br13"} : <i1>, <>
    %trueResult_6, %falseResult_7 = cond_br %23, %14 {handshake.bb = 2 : ui32, handshake.name = "cond_br42"} : <i1>, <i32>
    %trueResult_8, %falseResult_9 = cond_br %23, %16 {handshake.bb = 2 : ui32, handshake.name = "cond_br43"} : <i1>, <i32>
    %trueResult_10, %falseResult_11 = cond_br %23, %17 {handshake.bb = 2 : ui32, handshake.name = "cond_br44"} : <i1>, <i32>
    %result_12, %index_13 = control_merge [%trueResult_4]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>] to <>, <i1>
    %37 = source {handshake.bb = 2 : ui32, handshake.name = "source8"} : <>
    %38 = constant %37 {handshake.bb = 2 : ui32, handshake.name = "constant6", value = 1 : i32} : <>, <i32>
    %39 = addi %trueResult_8, %trueResult_10 {handshake.bb = 2 : ui32, handshake.name = "addi1"} : <i32>
    %40 = shrsi %39, %38 {handshake.bb = 2 : ui32, handshake.name = "shrsi1"} : <i32>
    %41 = muli %40, %40 {handshake.bb = 2 : ui32, handshake.name = "muli1"} : <i32>
    %42 = cmpi ne, %41, %trueResult_16 {handshake.bb = 2 : ui32, handshake.name = "cmpi2"} : <i32>
    %43 = cmpi sle, %trueResult_8, %trueResult_10 {handshake.bb = 2 : ui32, handshake.name = "cmpi3"} : <i32>
    %44 = cmpi sgt, %trueResult_8, %trueResult_10 {handshake.bb = 2 : ui32, handshake.name = "cmpi4"} : <i32>
    %45 = cmpi eq, %41, %trueResult_16 {handshake.bb = 2 : ui32, handshake.name = "cmpi5"} : <i32>
    %46 = andi %43, %42 {handshake.bb = 2 : ui32, handshake.name = "andi6"} : <i1>
    %47 = select %45[%40, %trueResult_6] {handshake.bb = 2 : ui32, handshake.name = "select0"} : <i1>, <i32>
    %48 = ori %46, %44 {handshake.bb = 2 : ui32, handshake.name = "ori2"} : <i1>
    %trueResult_14, %falseResult_15 = cond_br %45, %result_12 {handshake.bb = 2 : ui32, handshake.name = "cond_br34"} : <i1>, <>
    %trueResult_16, %falseResult_17 = cond_br %23, %8 {handshake.bb = 3 : ui32, handshake.name = "cond_br45"} : <i1>, <i32>
    %result_18, %index_19 = control_merge [%falseResult_15]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>] to <>, <i1>
    %49 = cmpi slt, %41, %trueResult_16 {handshake.bb = 3 : ui32, handshake.name = "cmpi6"} : <i32>
    %trueResult_20, %falseResult_21 = cond_br %49, %result_18 {handshake.bb = 3 : ui32, handshake.name = "cond_br36"} : <i1>, <>
    %trueResult_22, %falseResult_23 = cond_br %49, %40 {handshake.bb = 4 : ui32, handshake.name = "cond_br46"} : <i1>, <i32>
    %result_24, %index_25 = control_merge [%trueResult_20]  {handshake.bb = 4 : ui32, handshake.name = "control_merge3"} : [<>] to <>, <i1>
    %50 = source {handshake.bb = 4 : ui32, handshake.name = "source9"} : <>
    %51 = constant %50 {handshake.bb = 4 : ui32, handshake.name = "constant7", value = 1 : i32} : <>, <i32>
    %52 = addi %trueResult_22, %51 {handshake.bb = 4 : ui32, handshake.name = "addi2"} : <i32>
    %53 = br %result_24 {handshake.bb = 4 : ui32, handshake.name = "br3"} : <>
    %result_26, %index_27 = control_merge [%falseResult_21]  {handshake.bb = 5 : ui32, handshake.name = "control_merge4"} : [<>] to <>, <i1>
    %54 = source {handshake.bb = 5 : ui32, handshake.name = "source10"} : <>
    %55 = constant %54 {handshake.bb = 5 : ui32, handshake.name = "constant8", value = -1 : i32} : <>, <i32>
    %56 = addi %falseResult_23, %55 {handshake.bb = 5 : ui32, handshake.name = "addi3"} : <i32>
    %57 = br %result_26 {handshake.bb = 5 : ui32, handshake.name = "br4"} : <>
    %58 = select %falseResult_3[%falseResult_11, %falseResult_7] {handshake.bb = 6 : ui32, handshake.name = "select1"} : <i1>, <i32>
    end {handshake.bb = 6 : ui32, handshake.name = "end0"} %58, %arg1 : <i32>, <>
  }
}

