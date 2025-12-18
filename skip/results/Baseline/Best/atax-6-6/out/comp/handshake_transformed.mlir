module {
  handshake.func @atax(%arg0: memref<400xf32>, %arg1: memref<20xf32>, %arg2: memref<20xf32>, %arg3: memref<20xf32>, %arg4: !handshake.control<>, %arg5: !handshake.control<>, %arg6: !handshake.control<>, %arg7: !handshake.control<>, %arg8: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["A", "x", "y", "tmp", "A_start", "x_start", "y_start", "tmp_start", "start"], resNames = ["A_end", "x_end", "y_end", "tmp_end", "end"]} {
    %0:3 = fork [3] %arg8 {handshake.bb = 0 : ui32, handshake.name = "fork0"} : <>
    %1:2 = lsq[%arg3 : memref<20xf32>] (%arg7, %11#0, %addressResult, %140#0, %addressResult_34, %dataResult_35, %156#3)  {groupSizes = [1 : i32, 1 : i32], handshake.name = "lsq2"} : (!handshake.control<>, !handshake.control<>, !handshake.channel<i5>, !handshake.control<>, !handshake.channel<i5>, !handshake.channel<f32>, !handshake.control<>) -> (!handshake.channel<f32>, !handshake.control<>)
    %2:2 = lsq[%arg2 : memref<20xf32>] (%arg6, %96#0, %addressResult_18, %addressResult_22, %dataResult_23, %156#2)  {groupSizes = [2 : i32], handshake.name = "lsq3"} : (!handshake.control<>, !handshake.control<>, !handshake.channel<i5>, !handshake.channel<i5>, !handshake.channel<f32>, !handshake.control<>) -> (!handshake.channel<f32>, !handshake.control<>)
    %outputs, %memEnd = mem_controller[%arg1 : memref<20xf32>] %arg5 (%addressResult_6) %156#1 {connectedBlocks = [2 : i32], handshake.name = "mem_controller2"} :    (!handshake.channel<i5>) -> !handshake.channel<f32>
    %outputs_0:2, %memEnd_1 = mem_controller[%arg0 : memref<400xf32>] %arg4 (%addressResult_4, %addressResult_20) %156#0 {connectedBlocks = [2 : i32, 3 : i32], handshake.name = "mem_controller3"} :    (!handshake.channel<i9>, !handshake.channel<i9>) -> (!handshake.channel<f32>, !handshake.channel<f32>)
    %3 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant0", value = false} : <>, <i1>
    %4 = br %3 {handshake.bb = 0 : ui32, handshake.name = "br2"} : <i1>
    %5 = extsi %4 {handshake.bb = 0 : ui32, handshake.name = "extsi15"} : <i1> to <i6>
    %6 = br %0#2 {handshake.bb = 0 : ui32, handshake.name = "br3"} : <>
    %7 = mux %index [%5, %trueResult_36] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i6>, <i6>] to <i6>
    %8:2 = fork [2] %7 {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <i6>
    %9 = trunci %8#0 {handshake.bb = 1 : ui32, handshake.name = "trunci0"} : <i6> to <i5>
    %result, %index = control_merge [%6, %trueResult_38]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %11:3 = lazy_fork [3] %result {handshake.bb = 1 : ui32, handshake.name = "lazy_fork0"} : <>
    %12 = constant %11#2 {handshake.bb = 1 : ui32, handshake.name = "constant1", value = false} : <>, <i1>
    %addressResult, %dataResult = load[%9] %1#0 {handshake.bb = 1 : ui32, handshake.deps = #handshake<deps[["store1", 2]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 0>, handshake.name = "load0"} : <i5>, <f32>, <i5>, <f32>
    %13 = br %12 {handshake.bb = 1 : ui32, handshake.name = "br4"} : <i1>
    %14 = extsi %13 {handshake.bb = 1 : ui32, handshake.name = "extsi14"} : <i1> to <i6>
    %15 = br %dataResult {handshake.bb = 1 : ui32, handshake.name = "br5"} : <f32>
    %16 = br %17 {handshake.bb = 1 : ui32, handshake.name = "br6"} : <i6>
    %17 = buffer %8#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer1"} : <i6>
    %18 = br %11#1 {handshake.bb = 1 : ui32, handshake.name = "br7"} : <>
    %19 = mux %36#1 [%14, %trueResult] {handshake.bb = 2 : ui32, handshake.name = "mux1"} : <i1>, [<i6>, <i6>] to <i6>
    %21:3 = fork [3] %19 {handshake.bb = 2 : ui32, handshake.name = "fork2"} : <i6>
    %22 = extsi %21#0 {handshake.bb = 2 : ui32, handshake.name = "extsi16"} : <i6> to <i9>
    %24 = extsi %21#2 {handshake.bb = 2 : ui32, handshake.name = "extsi17"} : <i6> to <i7>
    %26 = trunci %21#1 {handshake.bb = 2 : ui32, handshake.name = "trunci1"} : <i6> to <i5>
    %28 = mux %36#2 [%15, %trueResult_8] {handshake.bb = 2 : ui32, handshake.name = "mux2"} : <i1>, [<f32>, <f32>] to <f32>
    %30 = mux %36#0 [%16, %trueResult_10] {handshake.bb = 2 : ui32, handshake.name = "mux3"} : <i1>, [<i6>, <i6>] to <i6>
    %32:2 = fork [2] %30 {handshake.bb = 2 : ui32, handshake.name = "fork3"} : <i6>
    %33 = extsi %32#1 {handshake.bb = 2 : ui32, handshake.name = "extsi18"} : <i6> to <i32>
    %35:2 = fork [2] %33 {handshake.bb = 2 : ui32, handshake.name = "fork4"} : <i32>
    %result_2, %index_3 = control_merge [%18, %trueResult_12]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>, <>] to <>, <i1>
    %36:3 = fork [3] %index_3 {handshake.bb = 2 : ui32, handshake.name = "fork5"} : <i1>
    %37:2 = fork [2] %result_2 {handshake.bb = 2 : ui32, handshake.name = "fork6"} : <>
    %38 = constant %37#0 {handshake.bb = 2 : ui32, handshake.name = "constant2", value = false} : <>, <i1>
    %39 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %40 = constant %39 {handshake.bb = 2 : ui32, handshake.name = "constant3", value = 20 : i6} : <>, <i6>
    %41 = extsi %40 {handshake.bb = 2 : ui32, handshake.name = "extsi19"} : <i6> to <i7>
    %42 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %43 = constant %42 {handshake.bb = 2 : ui32, handshake.name = "constant17", value = 1 : i2} : <>, <i2>
    %44 = extsi %43 {handshake.bb = 2 : ui32, handshake.name = "extsi20"} : <i2> to <i7>
    %45 = source {handshake.bb = 2 : ui32, handshake.name = "source2"} : <>
    %46 = constant %45 {handshake.bb = 2 : ui32, handshake.name = "constant18", value = 4 : i4} : <>, <i4>
    %47 = extsi %46 {handshake.bb = 2 : ui32, handshake.name = "extsi5"} : <i4> to <i32>
    %48 = source {handshake.bb = 2 : ui32, handshake.name = "source3"} : <>
    %49 = constant %48 {handshake.bb = 2 : ui32, handshake.name = "constant19", value = 2 : i3} : <>, <i3>
    %50 = extsi %49 {handshake.bb = 2 : ui32, handshake.name = "extsi6"} : <i3> to <i32>
    %51 = shli %52, %50 {handshake.bb = 2 : ui32, handshake.name = "shli0"} : <i32>
    %52 = buffer %35#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer9"} : <i32>
    %53 = trunci %51 {handshake.bb = 2 : ui32, handshake.name = "trunci2"} : <i32> to <i9>
    %54 = shli %55, %47 {handshake.bb = 2 : ui32, handshake.name = "shli1"} : <i32>
    %55 = buffer %35#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer10"} : <i32>
    %56 = trunci %54 {handshake.bb = 2 : ui32, handshake.name = "trunci3"} : <i32> to <i9>
    %57 = addi %53, %56 {handshake.bb = 2 : ui32, handshake.name = "addi5"} : <i9>
    %58 = addi %22, %57 {handshake.bb = 2 : ui32, handshake.name = "addi0"} : <i9>
    %addressResult_4, %dataResult_5 = load[%58] %outputs_0#0 {handshake.bb = 2 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i9>, <f32>, <i9>, <f32>
    %addressResult_6, %dataResult_7 = load[%26] %outputs {handshake.bb = 2 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load2"} : <i5>, <f32>, <i5>, <f32>
    %59 = mulf %dataResult_5, %dataResult_7 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "mulf0"} : <f32>
    %60 = addf %28, %59 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "addf0"} : <f32>
    %61 = addi %24, %44 {handshake.bb = 2 : ui32, handshake.name = "addi2"} : <i7>
    %62:2 = fork [2] %61 {handshake.bb = 2 : ui32, handshake.name = "fork7"} : <i7>
    %63 = trunci %64 {handshake.bb = 2 : ui32, handshake.name = "trunci4"} : <i7> to <i6>
    %64 = buffer %62#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer11"} : <i7>
    %65 = cmpi ult, %62#1, %41 {handshake.bb = 2 : ui32, handshake.name = "cmpi0"} : <i7>
    %67:5 = fork [5] %65 {handshake.bb = 2 : ui32, handshake.name = "fork8"} : <i1>
    %trueResult, %falseResult = cond_br %68, %63 {handshake.bb = 2 : ui32, handshake.name = "cond_br3"} : <i1>, <i6>
    %68 = buffer %67#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer13"} : <i1>
    sink %falseResult {handshake.name = "sink0"} : <i6>
    %trueResult_8, %falseResult_9 = cond_br %69, %60 {handshake.bb = 2 : ui32, handshake.name = "cond_br4"} : <i1>, <f32>
    %69 = buffer %67#2, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer14"} : <i1>
    %trueResult_10, %falseResult_11 = cond_br %67#1, %32#0 {handshake.bb = 2 : ui32, handshake.name = "cond_br5"} : <i1>, <i6>
    %trueResult_12, %falseResult_13 = cond_br %67#3, %37#1 {handshake.bb = 2 : ui32, handshake.name = "cond_br6"} : <i1>, <>
    %trueResult_14, %falseResult_15 = cond_br %67#4, %38 {handshake.bb = 2 : ui32, handshake.name = "cond_br7"} : <i1>, <i1>
    sink %trueResult_14 {handshake.name = "sink1"} : <i1>
    %74 = extsi %falseResult_15 {handshake.bb = 2 : ui32, handshake.name = "extsi13"} : <i1> to <i6>
    %75 = mux %95#1 [%74, %trueResult_24] {handshake.bb = 3 : ui32, handshake.name = "mux4"} : <i1>, [<i6>, <i6>] to <i6>
    %77:4 = fork [4] %75 {handshake.bb = 3 : ui32, handshake.name = "fork9"} : <i6>
    %78 = extsi %77#0 {handshake.bb = 3 : ui32, handshake.name = "extsi21"} : <i6> to <i9>
    %80 = extsi %77#3 {handshake.bb = 3 : ui32, handshake.name = "extsi22"} : <i6> to <i7>
    %82 = trunci %77#1 {handshake.bb = 3 : ui32, handshake.name = "trunci5"} : <i6> to <i5>
    %84 = trunci %77#2 {handshake.bb = 3 : ui32, handshake.name = "trunci6"} : <i6> to <i5>
    %86 = mux %95#0 [%falseResult_11, %trueResult_26] {handshake.bb = 3 : ui32, handshake.name = "mux5"} : <i1>, [<i6>, <i6>] to <i6>
    %88:2 = fork [2] %86 {handshake.bb = 3 : ui32, handshake.name = "fork10"} : <i6>
    %89 = extsi %88#1 {handshake.bb = 3 : ui32, handshake.name = "extsi23"} : <i6> to <i32>
    %91:2 = fork [2] %89 {handshake.bb = 3 : ui32, handshake.name = "fork11"} : <i32>
    %92 = mux %95#2 [%falseResult_9, %trueResult_28] {handshake.bb = 3 : ui32, handshake.name = "mux6"} : <i1>, [<f32>, <f32>] to <f32>
    %94:2 = fork [2] %92 {handshake.bb = 3 : ui32, handshake.name = "fork12"} : <f32>
    %result_16, %index_17 = control_merge [%falseResult_13, %trueResult_30]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>, <>] to <>, <i1>
    %95:3 = fork [3] %index_17 {handshake.bb = 3 : ui32, handshake.name = "fork13"} : <i1>
    %96:2 = fork [2] %result_16 {handshake.bb = 3 : ui32, handshake.name = "fork14"} : <>
    %97 = source {handshake.bb = 3 : ui32, handshake.name = "source4"} : <>
    %98 = constant %97 {handshake.bb = 3 : ui32, handshake.name = "constant20", value = 20 : i6} : <>, <i6>
    %99 = extsi %98 {handshake.bb = 3 : ui32, handshake.name = "extsi24"} : <i6> to <i7>
    %100 = source {handshake.bb = 3 : ui32, handshake.name = "source5"} : <>
    %101 = constant %100 {handshake.bb = 3 : ui32, handshake.name = "constant21", value = 1 : i2} : <>, <i2>
    %102 = extsi %101 {handshake.bb = 3 : ui32, handshake.name = "extsi25"} : <i2> to <i7>
    %103 = source {handshake.bb = 3 : ui32, handshake.name = "source6"} : <>
    %104 = constant %103 {handshake.bb = 3 : ui32, handshake.name = "constant22", value = 4 : i4} : <>, <i4>
    %105 = extsi %104 {handshake.bb = 3 : ui32, handshake.name = "extsi9"} : <i4> to <i32>
    %106 = source {handshake.bb = 3 : ui32, handshake.name = "source7"} : <>
    %107 = constant %106 {handshake.bb = 3 : ui32, handshake.name = "constant23", value = 2 : i3} : <>, <i3>
    %108 = extsi %107 {handshake.bb = 3 : ui32, handshake.name = "extsi10"} : <i3> to <i32>
    %addressResult_18, %dataResult_19 = load[%84] %2#0 {handshake.bb = 3 : ui32, handshake.deps = #handshake<deps[["store0", 1], ["store0", 3]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 0>, handshake.name = "load3"} : <i5>, <f32>, <i5>, <f32>
    %109 = shli %110, %108 {handshake.bb = 3 : ui32, handshake.name = "shli2"} : <i32>
    %110 = buffer %91#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer27"} : <i32>
    %111 = trunci %109 {handshake.bb = 3 : ui32, handshake.name = "trunci7"} : <i32> to <i9>
    %112 = shli %113, %105 {handshake.bb = 3 : ui32, handshake.name = "shli3"} : <i32>
    %113 = buffer %91#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer28"} : <i32>
    %114 = trunci %112 {handshake.bb = 3 : ui32, handshake.name = "trunci8"} : <i32> to <i9>
    %115 = addi %111, %114 {handshake.bb = 3 : ui32, handshake.name = "addi6"} : <i9>
    %116 = addi %78, %115 {handshake.bb = 3 : ui32, handshake.name = "addi1"} : <i9>
    %addressResult_20, %dataResult_21 = load[%116] %outputs_0#1 {handshake.bb = 3 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load4"} : <i9>, <f32>, <i9>, <f32>
    %117 = mulf %dataResult_21, %118 {fastmath = #arith.fastmath<none>, handshake.bb = 3 : ui32, handshake.name = "mulf1"} : <f32>
    %118 = buffer %94#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer29"} : <f32>
    %119 = addf %dataResult_19, %117 {fastmath = #arith.fastmath<none>, handshake.bb = 3 : ui32, handshake.name = "addf1"} : <f32>
    %addressResult_22, %dataResult_23 = store[%82] %119 {handshake.bb = 3 : ui32, handshake.deps = #handshake<deps[["load3", 1], ["store0", 1]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 0>, handshake.name = "store0"} : <i5>, <f32>, <i5>, <f32>
    %120 = addi %80, %102 {handshake.bb = 3 : ui32, handshake.name = "addi3"} : <i7>
    %121:2 = fork [2] %120 {handshake.bb = 3 : ui32, handshake.name = "fork15"} : <i7>
    %122 = trunci %123 {handshake.bb = 3 : ui32, handshake.name = "trunci9"} : <i7> to <i6>
    %123 = buffer %121#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer30"} : <i7>
    %124 = cmpi ult, %121#1, %99 {handshake.bb = 3 : ui32, handshake.name = "cmpi1"} : <i7>
    %126:4 = fork [4] %124 {handshake.bb = 3 : ui32, handshake.name = "fork16"} : <i1>
    %trueResult_24, %falseResult_25 = cond_br %127, %122 {handshake.bb = 3 : ui32, handshake.name = "cond_br8"} : <i1>, <i6>
    %127 = buffer %126#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer32"} : <i1>
    sink %falseResult_25 {handshake.name = "sink2"} : <i6>
    %trueResult_26, %falseResult_27 = cond_br %128, %88#0 {handshake.bb = 3 : ui32, handshake.name = "cond_br9"} : <i1>, <i6>
    %128 = buffer %126#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer33"} : <i1>
    %trueResult_28, %falseResult_29 = cond_br %130, %131 {handshake.bb = 3 : ui32, handshake.name = "cond_br10"} : <i1>, <f32>
    %130 = buffer %126#2, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer35"} : <i1>
    %131 = buffer %94#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer36"} : <f32>
    %trueResult_30, %falseResult_31 = cond_br %132, %96#1 {handshake.bb = 3 : ui32, handshake.name = "cond_br11"} : <i1>, <>
    %132 = buffer %126#3, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer37"} : <i1>
    %133 = merge %falseResult_27 {handshake.bb = 4 : ui32, handshake.name = "merge0"} : <i6>
    %134:2 = fork [2] %133 {handshake.bb = 4 : ui32, handshake.name = "fork17"} : <i6>
    %135 = extsi %134#1 {handshake.bb = 4 : ui32, handshake.name = "extsi26"} : <i6> to <i7>
    %137 = trunci %134#0 {handshake.bb = 4 : ui32, handshake.name = "trunci10"} : <i6> to <i5>
    %139 = merge %falseResult_29 {handshake.bb = 4 : ui32, handshake.name = "merge1"} : <f32>
    %result_32, %index_33 = control_merge [%falseResult_31]  {handshake.bb = 4 : ui32, handshake.name = "control_merge3"} : [<>] to <>, <i1>
    sink %index_33 {handshake.name = "sink3"} : <i1>
    %140:2 = lazy_fork [2] %result_32 {handshake.bb = 4 : ui32, handshake.name = "lazy_fork1"} : <>
    %141 = source {handshake.bb = 4 : ui32, handshake.name = "source8"} : <>
    %142 = constant %141 {handshake.bb = 4 : ui32, handshake.name = "constant24", value = 20 : i6} : <>, <i6>
    %143 = extsi %142 {handshake.bb = 4 : ui32, handshake.name = "extsi27"} : <i6> to <i7>
    %144 = source {handshake.bb = 4 : ui32, handshake.name = "source9"} : <>
    %145 = constant %144 {handshake.bb = 4 : ui32, handshake.name = "constant25", value = 1 : i2} : <>, <i2>
    %146 = extsi %145 {handshake.bb = 4 : ui32, handshake.name = "extsi28"} : <i2> to <i7>
    %addressResult_34, %dataResult_35 = store[%137] %139 {handshake.bb = 4 : ui32, handshake.mem_interface = #handshake.mem_interface<LSQ: 1>, handshake.name = "store1"} : <i5>, <f32>, <i5>, <f32>
    %147 = addi %135, %146 {handshake.bb = 4 : ui32, handshake.name = "addi4"} : <i7>
    %148:2 = fork [2] %147 {handshake.bb = 4 : ui32, handshake.name = "fork18"} : <i7>
    %149 = trunci %150 {handshake.bb = 4 : ui32, handshake.name = "trunci11"} : <i7> to <i6>
    %150 = buffer %148#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer40"} : <i7>
    %151 = cmpi ult, %148#1, %143 {handshake.bb = 4 : ui32, handshake.name = "cmpi2"} : <i7>
    %153:2 = fork [2] %151 {handshake.bb = 4 : ui32, handshake.name = "fork19"} : <i1>
    %trueResult_36, %falseResult_37 = cond_br %154, %149 {handshake.bb = 4 : ui32, handshake.name = "cond_br12"} : <i1>, <i6>
    %154 = buffer %153#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer42"} : <i1>
    sink %falseResult_37 {handshake.name = "sink4"} : <i6>
    %trueResult_38, %falseResult_39 = cond_br %155, %140#1 {handshake.bb = 4 : ui32, handshake.name = "cond_br13"} : <i1>, <>
    %155 = buffer %153#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer43"} : <i1>
    %result_40, %index_41 = control_merge [%falseResult_39]  {handshake.bb = 5 : ui32, handshake.name = "control_merge4"} : [<>] to <>, <i1>
    sink %index_41 {handshake.name = "sink5"} : <i1>
    %156:4 = fork [4] %result_40 {handshake.bb = 5 : ui32, handshake.name = "fork20"} : <>
    end {handshake.bb = 5 : ui32, handshake.name = "end0"} %memEnd_1, %memEnd, %2#1, %1#1, %0#1 : <>, <>, <>, <>, <>
  }
}

