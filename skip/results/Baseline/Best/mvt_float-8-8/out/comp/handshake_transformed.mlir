module {
  handshake.func @mvt_float(%arg0: memref<900xf32>, %arg1: memref<30xf32>, %arg2: memref<30xf32>, %arg3: memref<30xf32>, %arg4: memref<30xf32>, %arg5: !handshake.control<>, %arg6: !handshake.control<>, %arg7: !handshake.control<>, %arg8: !handshake.control<>, %arg9: !handshake.control<>, %arg10: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["A", "x1", "x2", "y1", "y2", "A_start", "x1_start", "x2_start", "y1_start", "y2_start", "start"], resNames = ["A_end", "x1_end", "x2_end", "y1_end", "y2_end", "end"]} {
    %0:3 = fork [3] %arg10 {handshake.bb = 0 : ui32, handshake.name = "fork0"} : <>
    %outputs, %memEnd = mem_controller[%arg4 : memref<30xf32>] %arg9 (%addressResult_34) %168#4 {connectedBlocks = [5 : i32], handshake.name = "mem_controller3"} :    (!handshake.channel<i5>) -> !handshake.channel<f32>
    %outputs_0, %memEnd_1 = mem_controller[%arg3 : memref<30xf32>] %arg8 (%addressResult_8) %168#3 {connectedBlocks = [2 : i32], handshake.name = "mem_controller4"} :    (!handshake.channel<i5>) -> !handshake.channel<f32>
    %1:2 = lsq[%arg2 : memref<30xf32>] (%arg7, %93#0, %addressResult_28, %152#0, %addressResult_46, %dataResult_47, %168#2)  {groupSizes = [1 : i32, 1 : i32], handshake.name = "lsq2"} : (!handshake.control<>, !handshake.control<>, !handshake.channel<i5>, !handshake.control<>, !handshake.channel<i5>, !handshake.channel<f32>, !handshake.control<>) -> (!handshake.channel<f32>, !handshake.control<>)
    %2:2 = lsq[%arg1 : memref<30xf32>] (%arg6, %11#0, %addressResult, %70#0, %addressResult_18, %dataResult_19, %168#1)  {groupSizes = [1 : i32, 1 : i32], handshake.name = "lsq3"} : (!handshake.control<>, !handshake.control<>, !handshake.channel<i5>, !handshake.control<>, !handshake.channel<i5>, !handshake.channel<f32>, !handshake.control<>) -> (!handshake.channel<f32>, !handshake.control<>)
    %outputs_2:2, %memEnd_3 = mem_controller[%arg0 : memref<900xf32>] %arg5 (%addressResult_6, %addressResult_32) %168#0 {connectedBlocks = [2 : i32, 5 : i32], handshake.name = "mem_controller5"} :    (!handshake.channel<i10>, !handshake.channel<i10>) -> (!handshake.channel<f32>, !handshake.channel<f32>)
    %3 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant0", value = false} : <>, <i1>
    %4 = br %3 {handshake.bb = 0 : ui32, handshake.name = "br2"} : <i1>
    %5 = extsi %4 {handshake.bb = 0 : ui32, handshake.name = "extsi15"} : <i1> to <i6>
    %6 = br %0#2 {handshake.bb = 0 : ui32, handshake.name = "br4"} : <>
    %7 = mux %index [%5, %trueResult_20] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i6>, <i6>] to <i6>
    %8:2 = fork [2] %7 {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <i6>
    %9 = trunci %8#0 {handshake.bb = 1 : ui32, handshake.name = "trunci0"} : <i6> to <i5>
    %result, %index = control_merge [%6, %trueResult_22]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %11:3 = lazy_fork [3] %result {handshake.bb = 1 : ui32, handshake.name = "lazy_fork0"} : <>
    %12 = constant %11#2 {handshake.bb = 1 : ui32, handshake.name = "constant1", value = false} : <>, <i1>
    %addressResult, %dataResult = load[%9] %2#0 {handshake.bb = 1 : ui32, handshake.deps = #handshake<deps[["store0", 2]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 0>, handshake.name = "load0"} : <i5>, <f32>, <i5>, <f32>
    %13 = br %12 {handshake.bb = 1 : ui32, handshake.name = "br5"} : <i1>
    %14 = extsi %13 {handshake.bb = 1 : ui32, handshake.name = "extsi14"} : <i1> to <i6>
    %15 = br %dataResult {handshake.bb = 1 : ui32, handshake.name = "br6"} : <f32>
    %16 = br %8#1 {handshake.bb = 1 : ui32, handshake.name = "br7"} : <i6>
    %18 = br %11#1 {handshake.bb = 1 : ui32, handshake.name = "br8"} : <>
    %19 = mux %35#1 [%14, %trueResult] {handshake.bb = 2 : ui32, handshake.name = "mux1"} : <i1>, [<i6>, <i6>] to <i6>
    %21:3 = fork [3] %19 {handshake.bb = 2 : ui32, handshake.name = "fork2"} : <i6>
    %22 = extsi %23 {handshake.bb = 2 : ui32, handshake.name = "extsi16"} : <i6> to <i10>
    %23 = buffer %21#0, bufferType = FIFO_BREAK_NONE, numSlots = 9 {handshake.bb = 2 : ui32, handshake.name = "buffer3"} : <i6>
    %24 = extsi %21#2 {handshake.bb = 2 : ui32, handshake.name = "extsi17"} : <i6> to <i7>
    %26 = trunci %27 {handshake.bb = 2 : ui32, handshake.name = "trunci1"} : <i6> to <i5>
    %27 = buffer %21#1, bufferType = FIFO_BREAK_NONE, numSlots = 9 {handshake.bb = 2 : ui32, handshake.name = "buffer5"} : <i6>
    %28 = mux %29 [%15, %trueResult_10] {handshake.bb = 2 : ui32, handshake.name = "mux2"} : <i1>, [<f32>, <f32>] to <f32>
    %29 = buffer %35#2, bufferType = FIFO_BREAK_NONE, numSlots = 15 {handshake.bb = 2 : ui32, handshake.name = "buffer6"} : <i1>
    %30 = mux %35#0 [%16, %trueResult_12] {handshake.bb = 2 : ui32, handshake.name = "mux3"} : <i1>, [<i6>, <i6>] to <i6>
    %32:2 = fork [2] %30 {handshake.bb = 2 : ui32, handshake.name = "fork3"} : <i6>
    %33 = extsi %34 {handshake.bb = 2 : ui32, handshake.name = "extsi18"} : <i6> to <i12>
    %34 = buffer %32#1, bufferType = FIFO_BREAK_NONE, numSlots = 5 {handshake.bb = 2 : ui32, handshake.name = "buffer8"} : <i6>
    %result_4, %index_5 = control_merge [%18, %trueResult_14]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>, <>] to <>, <i1>
    %35:3 = fork [3] %index_5 {handshake.bb = 2 : ui32, handshake.name = "fork4"} : <i1>
    %36 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %37 = constant %36 {handshake.bb = 2 : ui32, handshake.name = "constant14", value = 30 : i6} : <>, <i6>
    %38:2 = fork [2] %37 {handshake.bb = 2 : ui32, handshake.name = "fork5"} : <i6>
    %39 = extsi %40 {handshake.bb = 2 : ui32, handshake.name = "extsi19"} : <i6> to <i12>
    %40 = buffer %38#0, bufferType = FIFO_BREAK_NONE, numSlots = 5 {handshake.bb = 2 : ui32, handshake.name = "buffer9"} : <i6>
    %41 = extsi %38#1 {handshake.bb = 2 : ui32, handshake.name = "extsi20"} : <i6> to <i7>
    %43 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %44 = constant %43 {handshake.bb = 2 : ui32, handshake.name = "constant15", value = 1 : i2} : <>, <i2>
    %45 = extsi %44 {handshake.bb = 2 : ui32, handshake.name = "extsi21"} : <i2> to <i7>
    %46 = muli %33, %39 {handshake.bb = 2 : ui32, handshake.name = "muli0"} : <i12>
    %47 = trunci %46 {handshake.bb = 2 : ui32, handshake.name = "trunci2"} : <i12> to <i10>
    %48 = addi %22, %47 {handshake.bb = 2 : ui32, handshake.name = "addi0"} : <i10>
    %addressResult_6, %dataResult_7 = load[%48] %outputs_2#0 {handshake.bb = 2 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i10>, <f32>, <i10>, <f32>
    %addressResult_8, %dataResult_9 = load[%26] %outputs_0 {handshake.bb = 2 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load2"} : <i5>, <f32>, <i5>, <f32>
    %49 = mulf %dataResult_7, %dataResult_9 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "mulf0"} : <f32>
    %50 = addf %28, %49 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "addf0"} : <f32>
    %51 = addi %24, %45 {handshake.bb = 2 : ui32, handshake.name = "addi2"} : <i7>
    %52:2 = fork [2] %51 {handshake.bb = 2 : ui32, handshake.name = "fork6"} : <i7>
    %53 = trunci %52#0 {handshake.bb = 2 : ui32, handshake.name = "trunci3"} : <i7> to <i6>
    %55 = cmpi ult, %52#1, %41 {handshake.bb = 2 : ui32, handshake.name = "cmpi0"} : <i7>
    %57:4 = fork [4] %55 {handshake.bb = 2 : ui32, handshake.name = "fork7"} : <i1>
    %trueResult, %falseResult = cond_br %57#0, %53 {handshake.bb = 2 : ui32, handshake.name = "cond_br4"} : <i1>, <i6>
    sink %falseResult {handshake.name = "sink0"} : <i6>
    %trueResult_10, %falseResult_11 = cond_br %59, %50 {handshake.bb = 2 : ui32, handshake.name = "cond_br5"} : <i1>, <f32>
    %59 = buffer %57#2, bufferType = FIFO_BREAK_NONE, numSlots = 15 {handshake.bb = 2 : ui32, handshake.name = "buffer14"} : <i1>
    %trueResult_12, %falseResult_13 = cond_br %57#1, %32#0 {handshake.bb = 2 : ui32, handshake.name = "cond_br6"} : <i1>, <i6>
    %trueResult_14, %falseResult_15 = cond_br %57#3, %result_4 {handshake.bb = 2 : ui32, handshake.name = "cond_br7"} : <i1>, <>
    %63 = merge %falseResult_13 {handshake.bb = 3 : ui32, handshake.name = "merge0"} : <i6>
    %64:2 = fork [2] %63 {handshake.bb = 3 : ui32, handshake.name = "fork8"} : <i6>
    %65 = extsi %64#1 {handshake.bb = 3 : ui32, handshake.name = "extsi22"} : <i6> to <i7>
    %67 = trunci %64#0 {handshake.bb = 3 : ui32, handshake.name = "trunci4"} : <i6> to <i5>
    %69 = merge %falseResult_11 {handshake.bb = 3 : ui32, handshake.name = "merge1"} : <f32>
    %result_16, %index_17 = control_merge [%falseResult_15]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>] to <>, <i1>
    sink %index_17 {handshake.name = "sink1"} : <i1>
    %70:3 = lazy_fork [3] %result_16 {handshake.bb = 3 : ui32, handshake.name = "lazy_fork1"} : <>
    %71 = constant %70#2 {handshake.bb = 3 : ui32, handshake.name = "constant16", value = false} : <>, <i1>
    %72 = source {handshake.bb = 3 : ui32, handshake.name = "source2"} : <>
    %73 = constant %72 {handshake.bb = 3 : ui32, handshake.name = "constant17", value = 30 : i6} : <>, <i6>
    %74 = extsi %73 {handshake.bb = 3 : ui32, handshake.name = "extsi23"} : <i6> to <i7>
    %75 = source {handshake.bb = 3 : ui32, handshake.name = "source3"} : <>
    %76 = constant %75 {handshake.bb = 3 : ui32, handshake.name = "constant18", value = 1 : i2} : <>, <i2>
    %77 = extsi %76 {handshake.bb = 3 : ui32, handshake.name = "extsi24"} : <i2> to <i7>
    %addressResult_18, %dataResult_19 = store[%67] %69 {handshake.bb = 3 : ui32, handshake.mem_interface = #handshake.mem_interface<LSQ: 1>, handshake.name = "store0"} : <i5>, <f32>, <i5>, <f32>
    %78 = addi %65, %77 {handshake.bb = 3 : ui32, handshake.name = "addi3"} : <i7>
    %79:2 = fork [2] %78 {handshake.bb = 3 : ui32, handshake.name = "fork9"} : <i7>
    %80 = trunci %79#0 {handshake.bb = 3 : ui32, handshake.name = "trunci5"} : <i7> to <i6>
    %82 = cmpi ult, %79#1, %74 {handshake.bb = 3 : ui32, handshake.name = "cmpi1"} : <i7>
    %84:3 = fork [3] %82 {handshake.bb = 3 : ui32, handshake.name = "fork10"} : <i1>
    %trueResult_20, %falseResult_21 = cond_br %84#0, %80 {handshake.bb = 3 : ui32, handshake.name = "cond_br8"} : <i1>, <i6>
    sink %falseResult_21 {handshake.name = "sink2"} : <i6>
    %trueResult_22, %falseResult_23 = cond_br %84#1, %70#1 {handshake.bb = 3 : ui32, handshake.name = "cond_br9"} : <i1>, <>
    %trueResult_24, %falseResult_25 = cond_br %84#2, %71 {handshake.bb = 3 : ui32, handshake.name = "cond_br10"} : <i1>, <i1>
    sink %trueResult_24 {handshake.name = "sink3"} : <i1>
    %88 = extsi %falseResult_25 {handshake.bb = 3 : ui32, handshake.name = "extsi13"} : <i1> to <i6>
    %89 = mux %index_27 [%88, %trueResult_48] {handshake.bb = 4 : ui32, handshake.name = "mux4"} : <i1>, [<i6>, <i6>] to <i6>
    %90:2 = fork [2] %89 {handshake.bb = 4 : ui32, handshake.name = "fork11"} : <i6>
    %91 = trunci %90#0 {handshake.bb = 4 : ui32, handshake.name = "trunci6"} : <i6> to <i5>
    %result_26, %index_27 = control_merge [%falseResult_23, %trueResult_50]  {handshake.bb = 4 : ui32, handshake.name = "control_merge3"} : [<>, <>] to <>, <i1>
    %93:3 = lazy_fork [3] %result_26 {handshake.bb = 4 : ui32, handshake.name = "lazy_fork2"} : <>
    %94 = constant %93#2 {handshake.bb = 4 : ui32, handshake.name = "constant19", value = false} : <>, <i1>
    %addressResult_28, %dataResult_29 = load[%91] %1#0 {handshake.bb = 4 : ui32, handshake.deps = #handshake<deps[["store1", 2]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 0>, handshake.name = "load3"} : <i5>, <f32>, <i5>, <f32>
    %95 = br %94 {handshake.bb = 4 : ui32, handshake.name = "br9"} : <i1>
    %96 = extsi %95 {handshake.bb = 4 : ui32, handshake.name = "extsi12"} : <i1> to <i6>
    %97 = br %dataResult_29 {handshake.bb = 4 : ui32, handshake.name = "br10"} : <f32>
    %98 = br %90#1 {handshake.bb = 4 : ui32, handshake.name = "br11"} : <i6>
    %100 = br %93#1 {handshake.bb = 4 : ui32, handshake.name = "br12"} : <>
    %101 = mux %117#1 [%96, %trueResult_36] {handshake.bb = 5 : ui32, handshake.name = "mux5"} : <i1>, [<i6>, <i6>] to <i6>
    %103:3 = fork [3] %101 {handshake.bb = 5 : ui32, handshake.name = "fork12"} : <i6>
    %104 = extsi %103#1 {handshake.bb = 5 : ui32, handshake.name = "extsi25"} : <i6> to <i7>
    %106 = extsi %103#2 {handshake.bb = 5 : ui32, handshake.name = "extsi26"} : <i6> to <i12>
    %108 = trunci %103#0 {handshake.bb = 5 : ui32, handshake.name = "trunci7"} : <i6> to <i5>
    %110 = mux %111 [%97, %trueResult_38] {handshake.bb = 5 : ui32, handshake.name = "mux6"} : <i1>, [<f32>, <f32>] to <f32>
    %111 = buffer %117#2, bufferType = FIFO_BREAK_NONE, numSlots = 3 {handshake.bb = 5 : ui32, handshake.name = "buffer31"} : <i1>
    %112 = mux %117#0 [%98, %trueResult_40] {handshake.bb = 5 : ui32, handshake.name = "mux7"} : <i1>, [<i6>, <i6>] to <i6>
    %114:2 = fork [2] %112 {handshake.bb = 5 : ui32, handshake.name = "fork13"} : <i6>
    %115 = extsi %114#0 {handshake.bb = 5 : ui32, handshake.name = "extsi27"} : <i6> to <i10>
    %result_30, %index_31 = control_merge [%100, %trueResult_42]  {handshake.bb = 5 : ui32, handshake.name = "control_merge4"} : [<>, <>] to <>, <i1>
    %117:3 = fork [3] %index_31 {handshake.bb = 5 : ui32, handshake.name = "fork14"} : <i1>
    %118 = source {handshake.bb = 5 : ui32, handshake.name = "source4"} : <>
    %119 = constant %118 {handshake.bb = 5 : ui32, handshake.name = "constant20", value = 30 : i6} : <>, <i6>
    %120:2 = fork [2] %119 {handshake.bb = 5 : ui32, handshake.name = "fork15"} : <i6>
    %121 = extsi %120#0 {handshake.bb = 5 : ui32, handshake.name = "extsi28"} : <i6> to <i7>
    %123 = extsi %120#1 {handshake.bb = 5 : ui32, handshake.name = "extsi29"} : <i6> to <i12>
    %125 = source {handshake.bb = 5 : ui32, handshake.name = "source5"} : <>
    %126 = constant %125 {handshake.bb = 5 : ui32, handshake.name = "constant21", value = 1 : i2} : <>, <i2>
    %127 = extsi %126 {handshake.bb = 5 : ui32, handshake.name = "extsi30"} : <i2> to <i7>
    %128 = muli %106, %123 {handshake.bb = 5 : ui32, handshake.name = "muli1"} : <i12>
    %129 = trunci %128 {handshake.bb = 5 : ui32, handshake.name = "trunci8"} : <i12> to <i10>
    %130 = addi %115, %129 {handshake.bb = 5 : ui32, handshake.name = "addi1"} : <i10>
    %addressResult_32, %dataResult_33 = load[%130] %outputs_2#1 {handshake.bb = 5 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load4"} : <i10>, <f32>, <i10>, <f32>
    %addressResult_34, %dataResult_35 = load[%108] %outputs {handshake.bb = 5 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load5"} : <i5>, <f32>, <i5>, <f32>
    %131 = mulf %dataResult_33, %dataResult_35 {fastmath = #arith.fastmath<none>, handshake.bb = 5 : ui32, handshake.name = "mulf1"} : <f32>
    %132 = addf %110, %131 {fastmath = #arith.fastmath<none>, handshake.bb = 5 : ui32, handshake.name = "addf1"} : <f32>
    %133 = addi %104, %127 {handshake.bb = 5 : ui32, handshake.name = "addi4"} : <i7>
    %134:2 = fork [2] %133 {handshake.bb = 5 : ui32, handshake.name = "fork16"} : <i7>
    %135 = trunci %134#0 {handshake.bb = 5 : ui32, handshake.name = "trunci9"} : <i7> to <i6>
    %137 = cmpi ult, %134#1, %121 {handshake.bb = 5 : ui32, handshake.name = "cmpi2"} : <i7>
    %139:4 = fork [4] %137 {handshake.bb = 5 : ui32, handshake.name = "fork17"} : <i1>
    %trueResult_36, %falseResult_37 = cond_br %139#0, %135 {handshake.bb = 5 : ui32, handshake.name = "cond_br11"} : <i1>, <i6>
    sink %falseResult_37 {handshake.name = "sink4"} : <i6>
    %trueResult_38, %falseResult_39 = cond_br %141, %132 {handshake.bb = 5 : ui32, handshake.name = "cond_br12"} : <i1>, <f32>
    %141 = buffer %139#2, bufferType = FIFO_BREAK_NONE, numSlots = 4 {handshake.bb = 5 : ui32, handshake.name = "buffer39"} : <i1>
    %trueResult_40, %falseResult_41 = cond_br %142, %114#1 {handshake.bb = 5 : ui32, handshake.name = "cond_br13"} : <i1>, <i6>
    %142 = buffer %139#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 5 : ui32, handshake.name = "buffer40"} : <i1>
    %trueResult_42, %falseResult_43 = cond_br %139#3, %result_30 {handshake.bb = 5 : ui32, handshake.name = "cond_br14"} : <i1>, <>
    %145 = merge %falseResult_41 {handshake.bb = 6 : ui32, handshake.name = "merge2"} : <i6>
    %146:2 = fork [2] %145 {handshake.bb = 6 : ui32, handshake.name = "fork18"} : <i6>
    %147 = extsi %146#1 {handshake.bb = 6 : ui32, handshake.name = "extsi31"} : <i6> to <i7>
    %149 = trunci %146#0 {handshake.bb = 6 : ui32, handshake.name = "trunci10"} : <i6> to <i5>
    %151 = merge %falseResult_39 {handshake.bb = 6 : ui32, handshake.name = "merge3"} : <f32>
    %result_44, %index_45 = control_merge [%falseResult_43]  {handshake.bb = 6 : ui32, handshake.name = "control_merge5"} : [<>] to <>, <i1>
    sink %index_45 {handshake.name = "sink5"} : <i1>
    %152:2 = lazy_fork [2] %result_44 {handshake.bb = 6 : ui32, handshake.name = "lazy_fork3"} : <>
    %153 = source {handshake.bb = 6 : ui32, handshake.name = "source6"} : <>
    %154 = constant %153 {handshake.bb = 6 : ui32, handshake.name = "constant22", value = 30 : i6} : <>, <i6>
    %155 = extsi %154 {handshake.bb = 6 : ui32, handshake.name = "extsi32"} : <i6> to <i7>
    %156 = source {handshake.bb = 6 : ui32, handshake.name = "source7"} : <>
    %157 = constant %156 {handshake.bb = 6 : ui32, handshake.name = "constant23", value = 1 : i2} : <>, <i2>
    %158 = extsi %157 {handshake.bb = 6 : ui32, handshake.name = "extsi33"} : <i2> to <i7>
    %addressResult_46, %dataResult_47 = store[%149] %151 {handshake.bb = 6 : ui32, handshake.mem_interface = #handshake.mem_interface<LSQ: 1>, handshake.name = "store1"} : <i5>, <f32>, <i5>, <f32>
    %159 = addi %147, %158 {handshake.bb = 6 : ui32, handshake.name = "addi5"} : <i7>
    %160:2 = fork [2] %159 {handshake.bb = 6 : ui32, handshake.name = "fork19"} : <i7>
    %161 = trunci %160#0 {handshake.bb = 6 : ui32, handshake.name = "trunci11"} : <i7> to <i6>
    %163 = cmpi ult, %160#1, %155 {handshake.bb = 6 : ui32, handshake.name = "cmpi3"} : <i7>
    %165:2 = fork [2] %163 {handshake.bb = 6 : ui32, handshake.name = "fork20"} : <i1>
    %trueResult_48, %falseResult_49 = cond_br %165#0, %161 {handshake.bb = 6 : ui32, handshake.name = "cond_br15"} : <i1>, <i6>
    sink %falseResult_49 {handshake.name = "sink6"} : <i6>
    %trueResult_50, %falseResult_51 = cond_br %165#1, %152#1 {handshake.bb = 6 : ui32, handshake.name = "cond_br16"} : <i1>, <>
    %result_52, %index_53 = control_merge [%falseResult_51]  {handshake.bb = 7 : ui32, handshake.name = "control_merge6"} : [<>] to <>, <i1>
    sink %index_53 {handshake.name = "sink7"} : <i1>
    %168:5 = fork [5] %result_52 {handshake.bb = 7 : ui32, handshake.name = "fork21"} : <>
    end {handshake.bb = 7 : ui32, handshake.name = "end0"} %memEnd_3, %2#1, %1#1, %memEnd_1, %memEnd, %0#1 : <>, <>, <>, <>, <>, <>
  }
}

