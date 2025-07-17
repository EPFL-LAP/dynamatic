module {
  handshake.func @iterative_sqrt(%arg0: !handshake.channel<i32>, %arg1: !handshake.control<>, ...) -> (!handshake.channel<i32>, !handshake.control<>) attributes {argNames = ["n", "start"], resNames = ["out0", "end"]} {
    %0 = constant %arg1 {handshake.bb = 0 : ui32, handshake.name = "constant0", value = 0 : i32} : <>, <i32>
    %1 = constant %arg1 {handshake.bb = 0 : ui32, handshake.name = "constant3", value = true} : <>, <i1>
    %2 = br %arg0 {handshake.bb = 0 : ui32, handshake.name = "br2"} : <i32>
    %3 = br %0 {handshake.bb = 0 : ui32, handshake.name = "br3"} : <i32>
    %4 = br %1 {handshake.bb = 0 : ui32, handshake.name = "br4"} : <i1>
    %5 = br %arg1 {handshake.bb = 0 : ui32, handshake.name = "br5"} : <>
    %6 = mux %index [%2, %trueResult_8, %40, %52] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i2>, [<i32>, <i32>, <i32>, <i32>] to <i32>
    %7 = mux %index [%3, %trueResult_10, %41, %53] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i2>, [<i32>, <i32>, <i32>, <i32>] to <i32>
    %8 = mux %index [%4, %trueResult_12, %42, %54] {handshake.bb = 1 : ui32, handshake.name = "mux2"} : <i2>, [<i1>, <i1>, <i1>, <i1>] to <i1>
    %9 = mux %index [%2, %trueResult_14, %43, %55] {handshake.bb = 1 : ui32, handshake.name = "mux3"} : <i2>, [<i32>, <i32>, <i32>, <i32>] to <i32>
    %result, %index = control_merge [%5, %trueResult_16, %44, %56]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>, <>, <>] to <>, <i2>
    %10 = cmpi sle, %7, %6 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i32>
    %11 = andi %10, %8 {handshake.bb = 1 : ui32, handshake.name = "andi0"} : <i1>
    %trueResult, %falseResult = cond_br %11, %6 {handshake.bb = 1 : ui32, handshake.name = "cond_br3"} : <i1>, <i32>
    %trueResult_0, %falseResult_1 = cond_br %11, %7 {handshake.bb = 1 : ui32, handshake.name = "cond_br4"} : <i1>, <i32>
    %trueResult_2, %falseResult_3 = cond_br %11, %9 {handshake.bb = 1 : ui32, handshake.name = "cond_br5"} : <i1>, <i32>
    %trueResult_4, %falseResult_5 = cond_br %11, %result {handshake.bb = 1 : ui32, handshake.name = "cond_br6"} : <i1>, <>
    %12 = merge %trueResult {handshake.bb = 2 : ui32, handshake.name = "merge0"} : <i32>
    %13 = merge %trueResult_0 {handshake.bb = 2 : ui32, handshake.name = "merge1"} : <i32>
    %14 = merge %trueResult_2 {handshake.bb = 2 : ui32, handshake.name = "merge2"} : <i32>
    %result_6, %index_7 = control_merge [%trueResult_4]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>] to <>, <i1>
    %15 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %16 = constant %15 {handshake.bb = 2 : ui32, handshake.name = "constant4", value = 1 : i32} : <>, <i32>
    %17 = addi %13, %12 {handshake.bb = 2 : ui32, handshake.name = "addi0"} : <i32>
    %18 = shrsi %17, %16 {handshake.bb = 2 : ui32, handshake.name = "shrsi0"} : <i32>
    %19 = muli %18, %18 {handshake.bb = 2 : ui32, handshake.name = "muli0"} : <i32>
    %20 = cmpi ne, %19, %14 {handshake.bb = 2 : ui32, handshake.name = "cmpi1"} : <i32>
    %21 = cmpi sle, %13, %12 {handshake.bb = 2 : ui32, handshake.name = "cmpi2"} : <i32>
    %22 = cmpi sgt, %13, %12 {handshake.bb = 2 : ui32, handshake.name = "cmpi3"} : <i32>
    %23 = andi %21, %20 {handshake.bb = 2 : ui32, handshake.name = "andi1"} : <i1>
    %24 = cmpi eq, %19, %14 {handshake.bb = 2 : ui32, handshake.name = "cmpi4"} : <i32>
    %25 = ori %23, %22 {handshake.bb = 2 : ui32, handshake.name = "ori0"} : <i1>
    %trueResult_8, %falseResult_9 = cond_br %24, %18 {handshake.bb = 2 : ui32, handshake.name = "cond_br7"} : <i1>, <i32>
    %trueResult_10, %falseResult_11 = cond_br %24, %13 {handshake.bb = 2 : ui32, handshake.name = "cond_br8"} : <i1>, <i32>
    %trueResult_12, %falseResult_13 = cond_br %24, %25 {handshake.bb = 2 : ui32, handshake.name = "cond_br9"} : <i1>, <i1>
    %trueResult_14, %falseResult_15 = cond_br %24, %14 {handshake.bb = 2 : ui32, handshake.name = "cond_br10"} : <i1>, <i32>
    %trueResult_16, %falseResult_17 = cond_br %24, %result_6 {handshake.bb = 2 : ui32, handshake.name = "cond_br11"} : <i1>, <>
    %trueResult_18, %falseResult_19 = cond_br %24, %12 {handshake.bb = 2 : ui32, handshake.name = "cond_br12"} : <i1>, <i32>
    %trueResult_20, %falseResult_21 = cond_br %24, %19 {handshake.bb = 2 : ui32, handshake.name = "cond_br13"} : <i1>, <i32>
    %26 = merge %falseResult_15 {handshake.bb = 3 : ui32, handshake.name = "merge3"} : <i32>
    %27 = merge %falseResult_19 {handshake.bb = 3 : ui32, handshake.name = "merge4"} : <i32>
    %28 = merge %falseResult_11 {handshake.bb = 3 : ui32, handshake.name = "merge5"} : <i32>
    %29 = merge %falseResult_9 {handshake.bb = 3 : ui32, handshake.name = "merge6"} : <i32>
    %30 = merge %falseResult_21 {handshake.bb = 3 : ui32, handshake.name = "merge7"} : <i32>
    %31 = merge %falseResult_13 {handshake.bb = 3 : ui32, handshake.name = "merge8"} : <i1>
    %result_22, %index_23 = control_merge [%falseResult_17]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>] to <>, <i1>
    %32 = cmpi slt, %30, %26 {handshake.bb = 3 : ui32, handshake.name = "cmpi5"} : <i32>
    %trueResult_24, %falseResult_25 = cond_br %32, %26 {handshake.bb = 3 : ui32, handshake.name = "cond_br14"} : <i1>, <i32>
    %trueResult_26, %falseResult_27 = cond_br %32, %27 {handshake.bb = 3 : ui32, handshake.name = "cond_br15"} : <i1>, <i32>
    %trueResult_28, %falseResult_29 = cond_br %32, %29 {handshake.bb = 3 : ui32, handshake.name = "cond_br16"} : <i1>, <i32>
    %trueResult_30, %falseResult_31 = cond_br %32, %31 {handshake.bb = 3 : ui32, handshake.name = "cond_br17"} : <i1>, <i1>
    %trueResult_32, %falseResult_33 = cond_br %32, %result_22 {handshake.bb = 3 : ui32, handshake.name = "cond_br18"} : <i1>, <>
    %trueResult_34, %falseResult_35 = cond_br %32, %28 {handshake.bb = 3 : ui32, handshake.name = "cond_br19"} : <i1>, <i32>
    %33 = merge %trueResult_24 {handshake.bb = 4 : ui32, handshake.name = "merge9"} : <i32>
    %34 = merge %trueResult_26 {handshake.bb = 4 : ui32, handshake.name = "merge10"} : <i32>
    %35 = merge %trueResult_28 {handshake.bb = 4 : ui32, handshake.name = "merge11"} : <i32>
    %36 = merge %trueResult_30 {handshake.bb = 4 : ui32, handshake.name = "merge12"} : <i1>
    %result_36, %index_37 = control_merge [%trueResult_32]  {handshake.bb = 4 : ui32, handshake.name = "control_merge3"} : [<>] to <>, <i1>
    %37 = source {handshake.bb = 4 : ui32, handshake.name = "source1"} : <>
    %38 = constant %37 {handshake.bb = 4 : ui32, handshake.name = "constant5", value = 1 : i32} : <>, <i32>
    %39 = addi %35, %38 {handshake.bb = 4 : ui32, handshake.name = "addi1"} : <i32>
    %40 = br %34 {handshake.bb = 4 : ui32, handshake.name = "br6"} : <i32>
    %41 = br %39 {handshake.bb = 4 : ui32, handshake.name = "br8"} : <i32>
    %42 = br %36 {handshake.bb = 4 : ui32, handshake.name = "br9"} : <i1>
    %43 = br %33 {handshake.bb = 4 : ui32, handshake.name = "br10"} : <i32>
    %44 = br %result_36 {handshake.bb = 4 : ui32, handshake.name = "br11"} : <>
    %45 = merge %falseResult_25 {handshake.bb = 5 : ui32, handshake.name = "merge13"} : <i32>
    %46 = merge %falseResult_35 {handshake.bb = 5 : ui32, handshake.name = "merge14"} : <i32>
    %47 = merge %falseResult_29 {handshake.bb = 5 : ui32, handshake.name = "merge15"} : <i32>
    %48 = merge %falseResult_31 {handshake.bb = 5 : ui32, handshake.name = "merge16"} : <i1>
    %result_38, %index_39 = control_merge [%falseResult_33]  {handshake.bb = 5 : ui32, handshake.name = "control_merge4"} : [<>] to <>, <i1>
    %49 = source {handshake.bb = 5 : ui32, handshake.name = "source2"} : <>
    %50 = constant %49 {handshake.bb = 5 : ui32, handshake.name = "constant6", value = -1 : i32} : <>, <i32>
    %51 = addi %47, %50 {handshake.bb = 5 : ui32, handshake.name = "addi2"} : <i32>
    %52 = br %51 {handshake.bb = 5 : ui32, handshake.name = "br12"} : <i32>
    %53 = br %46 {handshake.bb = 5 : ui32, handshake.name = "br13"} : <i32>
    %54 = br %48 {handshake.bb = 5 : ui32, handshake.name = "br14"} : <i1>
    %55 = br %45 {handshake.bb = 5 : ui32, handshake.name = "br15"} : <i32>
    %56 = br %result_38 {handshake.bb = 5 : ui32, handshake.name = "br16"} : <>
    %57 = merge %falseResult {handshake.bb = 6 : ui32, handshake.name = "merge17"} : <i32>
    %result_40, %index_41 = control_merge [%falseResult_5]  {handshake.bb = 6 : ui32, handshake.name = "control_merge5"} : [<>] to <>, <i1>
    end {handshake.bb = 6 : ui32, handshake.name = "end0"} %57, %arg1 : <i32>, <>
  }
}

