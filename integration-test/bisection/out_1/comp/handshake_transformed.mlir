module {
  handshake.func @bisection(%arg0: !handshake.channel<f32>, %arg1: !handshake.channel<f32>, %arg2: !handshake.channel<f32>, %arg3: !handshake.control<>, ...) -> (!handshake.channel<f32>, !handshake.control<>) attributes {argNames = ["a", "b", "tol", "start"], resNames = ["out0", "end"]} {
    %0:3 = fork [3] %arg3 {handshake.bb = 0 : ui32, handshake.name = "fork0"} : <>
    %1:3 = fork [3] %arg0 {handshake.bb = 0 : ui32, handshake.name = "fork1"} : <f32>
    %2 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant0", value = false} : <>, <i1>
    %3 = source {handshake.bb = 0 : ui32, handshake.name = "source0"} : <>
    %4 = constant %3 {handshake.bb = 0 : ui32, handshake.name = "constant5", value = -2.000000e+00 : f32} : <>, <f32>
    %5 = mulf %1#1, %1#2 {fastmath = #arith.fastmath<none>, handshake.bb = 0 : ui32, handshake.name = "mulf0"} : <f32>
    %6 = addf %5, %4 {fastmath = #arith.fastmath<none>, handshake.bb = 0 : ui32, handshake.name = "addf0"} : <f32>
    %7 = extsi %2 {handshake.bb = 0 : ui32, handshake.name = "extsi3"} : <i1> to <i8>
    %8 = mux %13#1 [%1#0, %trueResult_66] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<f32>, <f32>] to <f32>
    %9 = mux %13#2 [%arg1, %trueResult_68] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i1>, [<f32>, <f32>] to <f32>
    %10 = mux %13#0 [%7, %trueResult_70] {handshake.bb = 1 : ui32, handshake.name = "mux2"} : <i1>, [<i8>, <i8>] to <i8>
    %11 = mux %13#3 [%6, %trueResult_72] {handshake.bb = 1 : ui32, handshake.name = "mux3"} : <i1>, [<f32>, <f32>] to <f32>
    %12 = mux %13#4 [%arg2, %trueResult_74] {handshake.bb = 1 : ui32, handshake.name = "mux4"} : <i1>, [<f32>, <f32>] to <f32>
    %result, %index = control_merge [%0#2, %trueResult_76]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %13:5 = fork [5] %index {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <i1>
    %14 = mux %22#1 [%8, %trueResult_52] {handshake.bb = 2 : ui32, handshake.name = "mux5"} : <i1>, [<f32>, <f32>] to <f32>
    %15:2 = fork [2] %14 {handshake.bb = 2 : ui32, handshake.name = "fork3"} : <f32>
    %16 = mux %22#2 [%9, %trueResult_54] {handshake.bb = 2 : ui32, handshake.name = "mux6"} : <i1>, [<f32>, <f32>] to <f32>
    %17:2 = fork [2] %16 {handshake.bb = 2 : ui32, handshake.name = "fork4"} : <f32>
    %18 = mux %22#0 [%10, %trueResult_56] {handshake.bb = 2 : ui32, handshake.name = "mux7"} : <i1>, [<i8>, <i8>] to <i8>
    %19 = mux %22#3 [%11, %trueResult_58] {handshake.bb = 2 : ui32, handshake.name = "mux8"} : <i1>, [<f32>, <f32>] to <f32>
    %20 = mux %22#4 [%12, %trueResult_60] {handshake.bb = 2 : ui32, handshake.name = "mux9"} : <i1>, [<f32>, <f32>] to <f32>
    %21:2 = fork [2] %20 {handshake.bb = 2 : ui32, handshake.name = "fork5"} : <f32>
    %result_0, %index_1 = control_merge [%result, %trueResult_62]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>, <>] to <>, <i1>
    %22:5 = fork [5] %index_1 {handshake.bb = 2 : ui32, handshake.name = "fork6"} : <i1>
    %23 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %24 = constant %23 {handshake.bb = 2 : ui32, handshake.name = "constant6", value = -2.000000e+00 : f32} : <>, <f32>
    %25 = source {handshake.bb = 2 : ui32, handshake.name = "source2"} : <>
    %26 = constant %25 {handshake.bb = 2 : ui32, handshake.name = "constant7", value = 5.000000e-01 : f32} : <>, <f32>
    %27 = addf %15#1, %17#1 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "addf1"} : <f32>
    %28 = mulf %27, %26 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "mulf1"} : <f32>
    %29:3 = fork [3] %28 {handshake.bb = 2 : ui32, handshake.name = "fork7"} : <f32>
    %30 = mulf %29#1, %29#2 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "mulf2"} : <f32>
    %31 = addf %30, %24 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "addf2"} : <f32>
    %32:2 = fork [2] %31 {handshake.bb = 2 : ui32, handshake.name = "fork8"} : <f32>
    %33 = absf %32#1 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "absf0"} : <f32> to <f32>
    %34 = cmpf olt, %33, %21#1 {handshake.bb = 2 : ui32, handshake.name = "cmpf0"} : <f32>
    %35:8 = fork [8] %34 {handshake.bb = 2 : ui32, handshake.name = "fork9"} : <i1>
    %trueResult, %falseResult = cond_br %35#7, %29#0 {handshake.bb = 2 : ui32, handshake.name = "cond_br5"} : <i1>, <f32>
    %trueResult_2, %falseResult_3 = cond_br %35#6, %result_0 {handshake.bb = 2 : ui32, handshake.name = "cond_br6"} : <i1>, <>
    %trueResult_4, %falseResult_5 = cond_br %35#5, %21#0 {handshake.bb = 2 : ui32, handshake.name = "cond_br7"} : <i1>, <f32>
    sink %trueResult_4 {handshake.name = "sink0"} : <f32>
    %trueResult_6, %falseResult_7 = cond_br %35#4, %15#0 {handshake.bb = 2 : ui32, handshake.name = "cond_br8"} : <i1>, <f32>
    sink %trueResult_6 {handshake.name = "sink1"} : <f32>
    %trueResult_8, %falseResult_9 = cond_br %35#3, %17#0 {handshake.bb = 2 : ui32, handshake.name = "cond_br9"} : <i1>, <f32>
    sink %trueResult_8 {handshake.name = "sink2"} : <f32>
    %trueResult_10, %falseResult_11 = cond_br %35#0, %18 {handshake.bb = 2 : ui32, handshake.name = "cond_br10"} : <i1>, <i8>
    sink %trueResult_10 {handshake.name = "sink3"} : <i8>
    %trueResult_12, %falseResult_13 = cond_br %35#2, %19 {handshake.bb = 2 : ui32, handshake.name = "cond_br11"} : <i1>, <f32>
    sink %trueResult_12 {handshake.name = "sink4"} : <f32>
    %trueResult_14, %falseResult_15 = cond_br %35#1, %32#0 {handshake.bb = 2 : ui32, handshake.name = "cond_br12"} : <i1>, <f32>
    sink %trueResult_14 {handshake.name = "sink5"} : <f32>
    %36:2 = fork [2] %falseResult_5 {handshake.bb = 3 : ui32, handshake.name = "fork10"} : <f32>
    %37:2 = fork [2] %falseResult_7 {handshake.bb = 3 : ui32, handshake.name = "fork11"} : <f32>
    %38:2 = fork [2] %falseResult_9 {handshake.bb = 3 : ui32, handshake.name = "fork12"} : <f32>
    %39 = source {handshake.bb = 3 : ui32, handshake.name = "source3"} : <>
    %40 = constant %39 {handshake.bb = 3 : ui32, handshake.name = "constant8", value = 5.000000e-01 : f32} : <>, <f32>
    %41 = subf %38#1, %37#1 {fastmath = #arith.fastmath<none>, handshake.bb = 3 : ui32, handshake.name = "subf0"} : <f32>
    %42 = mulf %41, %40 {fastmath = #arith.fastmath<none>, handshake.bb = 3 : ui32, handshake.name = "mulf3"} : <f32>
    %43 = cmpf olt, %42, %36#1 {handshake.bb = 3 : ui32, handshake.name = "cmpf1"} : <f32>
    %44:8 = fork [8] %43 {handshake.bb = 3 : ui32, handshake.name = "fork13"} : <i1>
    %trueResult_16, %falseResult_17 = cond_br %44#7, %falseResult {handshake.bb = 3 : ui32, handshake.name = "cond_br13"} : <i1>, <f32>
    %trueResult_18, %falseResult_19 = cond_br %44#6, %falseResult_3 {handshake.bb = 3 : ui32, handshake.name = "cond_br14"} : <i1>, <>
    %trueResult_20, %falseResult_21 = cond_br %44#5, %36#0 {handshake.bb = 3 : ui32, handshake.name = "cond_br15"} : <i1>, <f32>
    sink %trueResult_20 {handshake.name = "sink7"} : <f32>
    %trueResult_22, %falseResult_23 = cond_br %44#4, %37#0 {handshake.bb = 3 : ui32, handshake.name = "cond_br16"} : <i1>, <f32>
    sink %trueResult_22 {handshake.name = "sink8"} : <f32>
    %trueResult_24, %falseResult_25 = cond_br %44#3, %38#0 {handshake.bb = 3 : ui32, handshake.name = "cond_br17"} : <i1>, <f32>
    sink %trueResult_24 {handshake.name = "sink9"} : <f32>
    %trueResult_26, %falseResult_27 = cond_br %44#0, %falseResult_11 {handshake.bb = 3 : ui32, handshake.name = "cond_br18"} : <i1>, <i8>
    sink %trueResult_26 {handshake.name = "sink10"} : <i8>
    %trueResult_28, %falseResult_29 = cond_br %44#2, %falseResult_13 {handshake.bb = 3 : ui32, handshake.name = "cond_br19"} : <i1>, <f32>
    sink %trueResult_28 {handshake.name = "sink11"} : <f32>
    %trueResult_30, %falseResult_31 = cond_br %44#1, %falseResult_15 {handshake.bb = 3 : ui32, handshake.name = "cond_br20"} : <i1>, <f32>
    sink %trueResult_30 {handshake.name = "sink12"} : <f32>
    %45 = extsi %falseResult_27 {handshake.bb = 4 : ui32, handshake.name = "extsi4"} : <i8> to <i9>
    %46:2 = fork [2] %falseResult_29 {handshake.bb = 4 : ui32, handshake.name = "fork14"} : <f32>
    %47:2 = fork [2] %falseResult_31 {handshake.bb = 4 : ui32, handshake.name = "fork15"} : <f32>
    %48:2 = fork [2] %falseResult_19 {handshake.bb = 4 : ui32, handshake.name = "fork16"} : <>
    %49 = constant %48#1 {handshake.bb = 4 : ui32, handshake.name = "constant9", value = 0.000000e+00 : f32} : <>, <f32>
    %50:2 = fork [2] %49 {handshake.bb = 4 : ui32, handshake.name = "fork17"} : <f32>
    %51 = source {handshake.bb = 4 : ui32, handshake.name = "source4"} : <>
    %52 = constant %51 {handshake.bb = 4 : ui32, handshake.name = "constant1", value = 1 : i2} : <>, <i2>
    %53 = extsi %52 {handshake.bb = 4 : ui32, handshake.name = "extsi5"} : <i2> to <i9>
    %54 = source {handshake.bb = 4 : ui32, handshake.name = "source5"} : <>
    %55 = constant %54 {handshake.bb = 4 : ui32, handshake.name = "constant2", value = 100 : i8} : <>, <i8>
    %56 = extsi %55 {handshake.bb = 4 : ui32, handshake.name = "extsi6"} : <i8> to <i9>
    %57 = mulf %46#1, %47#1 {fastmath = #arith.fastmath<none>, handshake.bb = 4 : ui32, handshake.name = "mulf4"} : <f32>
    %58 = cmpf olt, %57, %50#1 {handshake.bb = 4 : ui32, handshake.name = "cmpf2"} : <f32>
    %59:10 = fork [10] %58 {handshake.bb = 4 : ui32, handshake.name = "fork18"} : <i1>
    %60 = addi %45, %53 {handshake.bb = 4 : ui32, handshake.name = "addi0"} : <i9>
    %61:2 = fork [2] %60 {handshake.bb = 4 : ui32, handshake.name = "fork19"} : <i9>
    %62 = trunci %61#0 {handshake.bb = 4 : ui32, handshake.name = "trunci0"} : <i9> to <i8>
    %63 = cmpi ult, %61#1, %56 {handshake.bb = 4 : ui32, handshake.name = "cmpi0"} : <i9>
    %trueResult_32, %falseResult_33 = cond_br %59#9, %falseResult_21 {handshake.bb = 4 : ui32, handshake.name = "cond_br21"} : <i1>, <f32>
    %trueResult_34, %falseResult_35 = cond_br %59#8, %falseResult_23 {handshake.bb = 4 : ui32, handshake.name = "cond_br22"} : <i1>, <f32>
    sink %falseResult_35 {handshake.name = "sink14"} : <f32>
    %trueResult_36, %falseResult_37 = cond_br %59#7, %46#0 {handshake.bb = 4 : ui32, handshake.name = "cond_br23"} : <i1>, <f32>
    sink %falseResult_37 {handshake.name = "sink15"} : <f32>
    %trueResult_38, %falseResult_39 = cond_br %59#6, %falseResult_17 {handshake.bb = 4 : ui32, handshake.name = "cond_br24"} : <i1>, <f32>
    %trueResult_40, %falseResult_41 = cond_br %59#5, %50#0 {handshake.bb = 4 : ui32, handshake.name = "cond_br25"} : <i1>, <f32>
    %trueResult_42, %falseResult_43 = cond_br %59#0, %62 {handshake.bb = 4 : ui32, handshake.name = "cond_br26"} : <i1>, <i8>
    %trueResult_44, %falseResult_45 = cond_br %59#4, %63 {handshake.bb = 4 : ui32, handshake.name = "cond_br27"} : <i1>, <i1>
    %trueResult_46, %falseResult_47 = cond_br %59#3, %48#0 {handshake.bb = 4 : ui32, handshake.name = "cond_br28"} : <i1>, <>
    %trueResult_48, %falseResult_49 = cond_br %59#2, %falseResult_25 {handshake.bb = 4 : ui32, handshake.name = "cond_br29"} : <i1>, <f32>
    sink %trueResult_48 {handshake.name = "sink16"} : <f32>
    %trueResult_50, %falseResult_51 = cond_br %59#1, %47#0 {handshake.bb = 4 : ui32, handshake.name = "cond_br30"} : <i1>, <f32>
    sink %trueResult_50 {handshake.name = "sink17"} : <f32>
    %64:7 = fork [7] %trueResult_44 {handshake.bb = 5 : ui32, handshake.name = "fork20"} : <i1>
    %trueResult_52, %falseResult_53 = cond_br %64#6, %trueResult_34 {handshake.bb = 5 : ui32, handshake.name = "cond_br31"} : <i1>, <f32>
    sink %falseResult_53 {handshake.name = "sink19"} : <f32>
    %trueResult_54, %falseResult_55 = cond_br %64#5, %trueResult_38 {handshake.bb = 5 : ui32, handshake.name = "cond_br32"} : <i1>, <f32>
    sink %falseResult_55 {handshake.name = "sink20"} : <f32>
    %trueResult_56, %falseResult_57 = cond_br %64#0, %trueResult_42 {handshake.bb = 5 : ui32, handshake.name = "cond_br33"} : <i1>, <i8>
    sink %falseResult_57 {handshake.name = "sink21"} : <i8>
    %trueResult_58, %falseResult_59 = cond_br %64#4, %trueResult_36 {handshake.bb = 5 : ui32, handshake.name = "cond_br34"} : <i1>, <f32>
    sink %falseResult_59 {handshake.name = "sink22"} : <f32>
    %trueResult_60, %falseResult_61 = cond_br %64#3, %trueResult_32 {handshake.bb = 5 : ui32, handshake.name = "cond_br35"} : <i1>, <f32>
    sink %falseResult_61 {handshake.name = "sink23"} : <f32>
    %trueResult_62, %falseResult_63 = cond_br %64#2, %trueResult_46 {handshake.bb = 5 : ui32, handshake.name = "cond_br36"} : <i1>, <>
    %trueResult_64, %falseResult_65 = cond_br %64#1, %trueResult_40 {handshake.bb = 5 : ui32, handshake.name = "cond_br37"} : <i1>, <f32>
    sink %trueResult_64 {handshake.name = "sink24"} : <f32>
    %65:7 = fork [7] %falseResult_45 {handshake.bb = 6 : ui32, handshake.name = "fork21"} : <i1>
    %trueResult_66, %falseResult_67 = cond_br %65#6, %falseResult_39 {handshake.bb = 6 : ui32, handshake.name = "cond_br38"} : <i1>, <f32>
    sink %falseResult_67 {handshake.name = "sink26"} : <f32>
    %trueResult_68, %falseResult_69 = cond_br %65#5, %falseResult_49 {handshake.bb = 6 : ui32, handshake.name = "cond_br39"} : <i1>, <f32>
    sink %falseResult_69 {handshake.name = "sink27"} : <f32>
    %trueResult_70, %falseResult_71 = cond_br %65#0, %falseResult_43 {handshake.bb = 6 : ui32, handshake.name = "cond_br40"} : <i1>, <i8>
    sink %falseResult_71 {handshake.name = "sink28"} : <i8>
    %trueResult_72, %falseResult_73 = cond_br %65#4, %falseResult_51 {handshake.bb = 6 : ui32, handshake.name = "cond_br41"} : <i1>, <f32>
    sink %falseResult_73 {handshake.name = "sink29"} : <f32>
    %trueResult_74, %falseResult_75 = cond_br %65#3, %falseResult_33 {handshake.bb = 6 : ui32, handshake.name = "cond_br42"} : <i1>, <f32>
    sink %falseResult_75 {handshake.name = "sink30"} : <f32>
    %trueResult_76, %falseResult_77 = cond_br %65#2, %falseResult_47 {handshake.bb = 6 : ui32, handshake.name = "cond_br43"} : <i1>, <>
    %trueResult_78, %falseResult_79 = cond_br %65#1, %falseResult_41 {handshake.bb = 6 : ui32, handshake.name = "cond_br44"} : <i1>, <f32>
    sink %trueResult_78 {handshake.name = "sink31"} : <f32>
    %66 = mux %index_81 [%trueResult, %falseResult_65, %falseResult_79] {handshake.bb = 7 : ui32, handshake.name = "mux10"} : <i2>, [<f32>, <f32>, <f32>] to <f32>
    %result_80, %index_81 = control_merge [%trueResult_2, %falseResult_63, %falseResult_77]  {handshake.bb = 7 : ui32, handshake.name = "control_merge6"} : [<>, <>, <>] to <>, <i2>
    %67 = mux %index_83 [%trueResult_16, %66] {handshake.bb = 8 : ui32, handshake.name = "mux11"} : <i1>, [<f32>, <f32>] to <f32>
    %result_82, %index_83 = control_merge [%trueResult_18, %result_80]  {handshake.bb = 8 : ui32, handshake.name = "control_merge7"} : [<>, <>] to <>, <i1>
    sink %result_82 {handshake.name = "sink32"} : <>
    end {handshake.bb = 8 : ui32, handshake.name = "end0"} %67, %0#1 : <f32>, <>
  }
}

