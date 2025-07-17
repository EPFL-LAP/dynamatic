module {
  handshake.func @iterative_sqrt(%arg0: !handshake.channel<i32>, %arg1: !handshake.control<>, ...) -> (!handshake.channel<i32>, !handshake.control<>) attributes {argNames = ["n", "start"], resNames = ["out0", "end"]} {
    %0:4 = fork [4] %arg1 {handshake.bb = 0 : ui32, handshake.name = "fork0"} : <>
    %1 = constant %0#3 {handshake.bb = 0 : ui32, handshake.name = "constant1", value = false} : <>, <i1>
    %2 = constant %0#2 {handshake.bb = 0 : ui32, handshake.name = "constant3", value = true} : <>, <i1>
    %3 = br %arg0 {handshake.bb = 0 : ui32, handshake.name = "br2"} : <i32>
    %4:2 = fork [2] %3 {handshake.bb = 0 : ui32, handshake.name = "fork1"} : <i32>
    %5 = br %1 {handshake.bb = 0 : ui32, handshake.name = "br3"} : <i1>
    %6 = extsi %5 {handshake.bb = 0 : ui32, handshake.name = "extsi3"} : <i1> to <i32>
    %7 = br %2 {handshake.bb = 0 : ui32, handshake.name = "br4"} : <i1>
    %8 = br %0#1 {handshake.bb = 0 : ui32, handshake.name = "br5"} : <>
    %9 = mux %15#0 [%4#1, %trueResult_8, %57, %69] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i2>, [<i32>, <i32>, <i32>, <i32>] to <i32>
    %10:2 = fork [2] %9 {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <i32>
    %11 = mux %15#1 [%6, %trueResult_10, %58, %70] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i2>, [<i32>, <i32>, <i32>, <i32>] to <i32>
    %12:2 = fork [2] %11 {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <i32>
    %13 = mux %15#2 [%7, %trueResult_12, %59, %71] {handshake.bb = 1 : ui32, handshake.name = "mux2"} : <i2>, [<i1>, <i1>, <i1>, <i1>] to <i1>
    %14 = mux %15#3 [%4#0, %trueResult_14, %60, %72] {handshake.bb = 1 : ui32, handshake.name = "mux3"} : <i2>, [<i32>, <i32>, <i32>, <i32>] to <i32>
    %result, %index = control_merge [%8, %trueResult_16, %61, %73]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>, <>, <>] to <>, <i2>
    %15:4 = fork [4] %index {handshake.bb = 1 : ui32, handshake.name = "fork4"} : <i2>
    %16 = cmpi sle, %12#1, %10#1 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i32>
    %17 = andi %16, %13 {handshake.bb = 1 : ui32, handshake.name = "andi0"} : <i1>
    %18:4 = fork [4] %17 {handshake.bb = 1 : ui32, handshake.name = "fork5"} : <i1>
    %trueResult, %falseResult = cond_br %18#3, %10#0 {handshake.bb = 1 : ui32, handshake.name = "cond_br3"} : <i1>, <i32>
    %trueResult_0, %falseResult_1 = cond_br %18#2, %12#0 {handshake.bb = 1 : ui32, handshake.name = "cond_br4"} : <i1>, <i32>
    sink %falseResult_1 {handshake.name = "sink0"} : <i32>
    %trueResult_2, %falseResult_3 = cond_br %18#1, %14 {handshake.bb = 1 : ui32, handshake.name = "cond_br5"} : <i1>, <i32>
    sink %falseResult_3 {handshake.name = "sink1"} : <i32>
    %trueResult_4, %falseResult_5 = cond_br %18#0, %result {handshake.bb = 1 : ui32, handshake.name = "cond_br6"} : <i1>, <>
    sink %falseResult_5 {handshake.name = "sink2"} : <>
    %19 = merge %trueResult {handshake.bb = 2 : ui32, handshake.name = "merge0"} : <i32>
    %20:4 = fork [4] %19 {handshake.bb = 2 : ui32, handshake.name = "fork6"} : <i32>
    %21 = merge %trueResult_0 {handshake.bb = 2 : ui32, handshake.name = "merge1"} : <i32>
    %22:4 = fork [4] %21 {handshake.bb = 2 : ui32, handshake.name = "fork7"} : <i32>
    %23 = merge %trueResult_2 {handshake.bb = 2 : ui32, handshake.name = "merge2"} : <i32>
    %24:3 = fork [3] %23 {handshake.bb = 2 : ui32, handshake.name = "fork8"} : <i32>
    %result_6, %index_7 = control_merge [%trueResult_4]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>] to <>, <i1>
    sink %index_7 {handshake.name = "sink3"} : <i1>
    %25 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %26 = constant %25 {handshake.bb = 2 : ui32, handshake.name = "constant2", value = 1 : i2} : <>, <i2>
    %27 = extsi %26 {handshake.bb = 2 : ui32, handshake.name = "extsi1"} : <i2> to <i32>
    %28 = addi %22#3, %20#3 {handshake.bb = 2 : ui32, handshake.name = "addi0"} : <i32>
    %29 = shrsi %28, %27 {handshake.bb = 2 : ui32, handshake.name = "shrsi0"} : <i32>
    %30:3 = fork [3] %29 {handshake.bb = 2 : ui32, handshake.name = "fork9"} : <i32>
    %31 = muli %30#1, %30#2 {handshake.bb = 2 : ui32, handshake.name = "muli0"} : <i32>
    %32:3 = fork [3] %31 {handshake.bb = 2 : ui32, handshake.name = "fork10"} : <i32>
    %33 = cmpi ne, %32#2, %24#2 {handshake.bb = 2 : ui32, handshake.name = "cmpi1"} : <i32>
    %34 = cmpi sle, %22#2, %20#2 {handshake.bb = 2 : ui32, handshake.name = "cmpi2"} : <i32>
    %35 = cmpi sgt, %22#1, %20#1 {handshake.bb = 2 : ui32, handshake.name = "cmpi3"} : <i32>
    %36 = andi %34, %33 {handshake.bb = 2 : ui32, handshake.name = "andi1"} : <i1>
    %37 = cmpi eq, %32#1, %24#1 {handshake.bb = 2 : ui32, handshake.name = "cmpi4"} : <i32>
    %38:7 = fork [7] %37 {handshake.bb = 2 : ui32, handshake.name = "fork11"} : <i1>
    %39 = ori %36, %35 {handshake.bb = 2 : ui32, handshake.name = "ori0"} : <i1>
    %trueResult_8, %falseResult_9 = cond_br %38#6, %30#0 {handshake.bb = 2 : ui32, handshake.name = "cond_br7"} : <i1>, <i32>
    %trueResult_10, %falseResult_11 = cond_br %38#5, %22#0 {handshake.bb = 2 : ui32, handshake.name = "cond_br8"} : <i1>, <i32>
    %trueResult_12, %falseResult_13 = cond_br %38#4, %39 {handshake.bb = 2 : ui32, handshake.name = "cond_br9"} : <i1>, <i1>
    %trueResult_14, %falseResult_15 = cond_br %38#3, %24#0 {handshake.bb = 2 : ui32, handshake.name = "cond_br10"} : <i1>, <i32>
    %trueResult_16, %falseResult_17 = cond_br %38#2, %result_6 {handshake.bb = 2 : ui32, handshake.name = "cond_br11"} : <i1>, <>
    %trueResult_18, %falseResult_19 = cond_br %38#1, %20#0 {handshake.bb = 2 : ui32, handshake.name = "cond_br12"} : <i1>, <i32>
    sink %trueResult_18 {handshake.name = "sink4"} : <i32>
    %trueResult_20, %falseResult_21 = cond_br %38#0, %32#0 {handshake.bb = 2 : ui32, handshake.name = "cond_br13"} : <i1>, <i32>
    sink %trueResult_20 {handshake.name = "sink5"} : <i32>
    %40 = merge %falseResult_15 {handshake.bb = 3 : ui32, handshake.name = "merge3"} : <i32>
    %41:2 = fork [2] %40 {handshake.bb = 3 : ui32, handshake.name = "fork12"} : <i32>
    %42 = merge %falseResult_19 {handshake.bb = 3 : ui32, handshake.name = "merge4"} : <i32>
    %43 = merge %falseResult_11 {handshake.bb = 3 : ui32, handshake.name = "merge5"} : <i32>
    %44 = merge %falseResult_9 {handshake.bb = 3 : ui32, handshake.name = "merge6"} : <i32>
    %45 = merge %falseResult_21 {handshake.bb = 3 : ui32, handshake.name = "merge7"} : <i32>
    %46 = merge %falseResult_13 {handshake.bb = 3 : ui32, handshake.name = "merge8"} : <i1>
    %result_22, %index_23 = control_merge [%falseResult_17]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>] to <>, <i1>
    sink %index_23 {handshake.name = "sink6"} : <i1>
    %47 = cmpi slt, %45, %41#1 {handshake.bb = 3 : ui32, handshake.name = "cmpi5"} : <i32>
    %48:6 = fork [6] %47 {handshake.bb = 3 : ui32, handshake.name = "fork13"} : <i1>
    %trueResult_24, %falseResult_25 = cond_br %48#5, %41#0 {handshake.bb = 3 : ui32, handshake.name = "cond_br14"} : <i1>, <i32>
    %trueResult_26, %falseResult_27 = cond_br %48#4, %42 {handshake.bb = 3 : ui32, handshake.name = "cond_br15"} : <i1>, <i32>
    sink %falseResult_27 {handshake.name = "sink7"} : <i32>
    %trueResult_28, %falseResult_29 = cond_br %48#3, %44 {handshake.bb = 3 : ui32, handshake.name = "cond_br16"} : <i1>, <i32>
    %trueResult_30, %falseResult_31 = cond_br %48#2, %46 {handshake.bb = 3 : ui32, handshake.name = "cond_br17"} : <i1>, <i1>
    %trueResult_32, %falseResult_33 = cond_br %48#1, %result_22 {handshake.bb = 3 : ui32, handshake.name = "cond_br18"} : <i1>, <>
    %trueResult_34, %falseResult_35 = cond_br %48#0, %43 {handshake.bb = 3 : ui32, handshake.name = "cond_br19"} : <i1>, <i32>
    sink %trueResult_34 {handshake.name = "sink8"} : <i32>
    %49 = merge %trueResult_24 {handshake.bb = 4 : ui32, handshake.name = "merge9"} : <i32>
    %50 = merge %trueResult_26 {handshake.bb = 4 : ui32, handshake.name = "merge10"} : <i32>
    %51 = merge %trueResult_28 {handshake.bb = 4 : ui32, handshake.name = "merge11"} : <i32>
    %52 = merge %trueResult_30 {handshake.bb = 4 : ui32, handshake.name = "merge12"} : <i1>
    %result_36, %index_37 = control_merge [%trueResult_32]  {handshake.bb = 4 : ui32, handshake.name = "control_merge3"} : [<>] to <>, <i1>
    sink %index_37 {handshake.name = "sink9"} : <i1>
    %53 = source {handshake.bb = 4 : ui32, handshake.name = "source1"} : <>
    %54 = constant %53 {handshake.bb = 4 : ui32, handshake.name = "constant7", value = 1 : i2} : <>, <i2>
    %55 = extsi %54 {handshake.bb = 4 : ui32, handshake.name = "extsi2"} : <i2> to <i32>
    %56 = addi %51, %55 {handshake.bb = 4 : ui32, handshake.name = "addi1"} : <i32>
    %57 = br %50 {handshake.bb = 4 : ui32, handshake.name = "br6"} : <i32>
    %58 = br %56 {handshake.bb = 4 : ui32, handshake.name = "br8"} : <i32>
    %59 = br %52 {handshake.bb = 4 : ui32, handshake.name = "br9"} : <i1>
    %60 = br %49 {handshake.bb = 4 : ui32, handshake.name = "br10"} : <i32>
    %61 = br %result_36 {handshake.bb = 4 : ui32, handshake.name = "br11"} : <>
    %62 = merge %falseResult_25 {handshake.bb = 5 : ui32, handshake.name = "merge13"} : <i32>
    %63 = merge %falseResult_35 {handshake.bb = 5 : ui32, handshake.name = "merge14"} : <i32>
    %64 = merge %falseResult_29 {handshake.bb = 5 : ui32, handshake.name = "merge15"} : <i32>
    %65 = merge %falseResult_31 {handshake.bb = 5 : ui32, handshake.name = "merge16"} : <i1>
    %result_38, %index_39 = control_merge [%falseResult_33]  {handshake.bb = 5 : ui32, handshake.name = "control_merge4"} : [<>] to <>, <i1>
    sink %index_39 {handshake.name = "sink10"} : <i1>
    %66 = source {handshake.bb = 5 : ui32, handshake.name = "source2"} : <>
    %67 = constant %66 {handshake.bb = 5 : ui32, handshake.name = "constant6", value = -1 : i32} : <>, <i32>
    %68 = addi %64, %67 {handshake.bb = 5 : ui32, handshake.name = "addi2"} : <i32>
    %69 = br %68 {handshake.bb = 5 : ui32, handshake.name = "br12"} : <i32>
    %70 = br %63 {handshake.bb = 5 : ui32, handshake.name = "br13"} : <i32>
    %71 = br %65 {handshake.bb = 5 : ui32, handshake.name = "br14"} : <i1>
    %72 = br %62 {handshake.bb = 5 : ui32, handshake.name = "br15"} : <i32>
    %73 = br %result_38 {handshake.bb = 5 : ui32, handshake.name = "br16"} : <>
    %74 = merge %falseResult {handshake.bb = 6 : ui32, handshake.name = "merge17"} : <i32>
    end {handshake.bb = 6 : ui32, handshake.name = "end0"} %74, %0#0 : <i32>, <>
  }
}

