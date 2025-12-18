module {
  handshake.func @atax(%arg0: memref<400xf32>, %arg1: memref<20xf32>, %arg2: memref<20xf32>, %arg3: memref<20xf32>, %arg4: !handshake.control<>, %arg5: !handshake.control<>, %arg6: !handshake.control<>, %arg7: !handshake.control<>, %arg8: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["A", "x", "y", "tmp", "A_start", "x_start", "y_start", "tmp_start", "start"], cfg.edges = "[0,1][2,2,3,cmpi0][4,1,5,cmpi2][1,2][3,3,4,cmpi1]", resNames = ["A_end", "x_end", "y_end", "tmp_end", "end"]} {
    %0:4 = fork [4] %arg8 {handshake.bb = 0 : ui32, handshake.name = "fork0"} : <>
    %outputs:2, %memEnd = mem_controller[%arg3 : memref<20xf32>] %arg7 (%addressResult, %158, %addressResult_42, %dataResult_43) %176#3 {connectedBlocks = [1 : i32, 4 : i32], handshake.name = "mem_controller2"} :    (!handshake.channel<i5>, !handshake.channel<i32>, !handshake.channel<i5>, !handshake.channel<f32>) -> (!handshake.channel<f32>, !handshake.control<>)
    %outputs_0:2, %memEnd_1 = mem_controller[%arg2 : memref<20xf32>] %arg6 (%108, %addressResult_24, %addressResult_28, %dataResult_29) %176#2 {connectedBlocks = [3 : i32], handshake.name = "mem_controller3"} :    (!handshake.channel<i32>, !handshake.channel<i5>, !handshake.channel<i5>, !handshake.channel<f32>) -> (!handshake.channel<f32>, !handshake.control<>)
    %outputs_2, %memEnd_3 = mem_controller[%arg1 : memref<20xf32>] %arg5 (%addressResult_10) %176#1 {connectedBlocks = [2 : i32], handshake.name = "mem_controller4"} :    (!handshake.channel<i5>) -> !handshake.channel<f32>
    %outputs_4:2, %memEnd_5 = mem_controller[%arg0 : memref<400xf32>] %arg4 (%addressResult_8, %addressResult_26) %176#0 {connectedBlocks = [2 : i32, 3 : i32], handshake.name = "mem_controller5"} :    (!handshake.channel<i9>, !handshake.channel<i9>) -> (!handshake.channel<f32>, !handshake.channel<f32>)
    %1 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant2", value = false} : <>, <i1>
    %2 = br %1 {handshake.bb = 0 : ui32, handshake.name = "br2"} : <i1>
    %3 = extsi %2 {handshake.bb = 0 : ui32, handshake.name = "extsi17"} : <i1> to <i6>
    %4 = br %0#3 {handshake.bb = 0 : ui32, handshake.name = "br3"} : <>
    %5 = mux %6 [%0#2, %trueResult_38] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux8"} : <i1>, [<>, <>] to <>
    %6 = init %173#2 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "init0"} : <i1>
    %8:2 = unbundle %17#0  {handshake.bb = 1 : ui32, handshake.name = "unbundle0"} : <f32> to _ 
    %10 = mux %index [%3, %trueResult_45] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i6>, <i6>] to <i6>
    %11:2 = fork [2] %10 {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <i6>
    %12 = trunci %11#0 {handshake.bb = 1 : ui32, handshake.name = "trunci0"} : <i6> to <i5>
    %result, %index = control_merge [%4, %trueResult_47]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %14:2 = fork [2] %result {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <>
    %15 = constant %14#0 {handshake.bb = 1 : ui32, handshake.name = "constant3", value = false} : <>, <i1>
    %16 = buffer %8#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer0"} : <>
    %addressResult, %dataResult = load[%12] %outputs#0 {handshake.bb = 1 : ui32, handshake.deps = #handshake<deps[["store1", 2, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i5>, <f32>, <i5>, <f32>
    %17:2 = fork [2] %dataResult {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <f32>
    %18 = br %15 {handshake.bb = 1 : ui32, handshake.name = "br4"} : <i1>
    %19 = extsi %18 {handshake.bb = 1 : ui32, handshake.name = "extsi16"} : <i1> to <i6>
    %20 = br %17#1 {handshake.bb = 1 : ui32, handshake.name = "br5"} : <f32>
    %22 = br %11#1 {handshake.bb = 1 : ui32, handshake.name = "br6"} : <i6>
    %24 = br %14#1 {handshake.bb = 1 : ui32, handshake.name = "br7"} : <>
    %25 = mux %42#1 [%19, %trueResult] {handshake.bb = 2 : ui32, handshake.name = "mux1"} : <i1>, [<i6>, <i6>] to <i6>
    %27:3 = fork [3] %25 {handshake.bb = 2 : ui32, handshake.name = "fork4"} : <i6>
    %28 = extsi %27#0 {handshake.bb = 2 : ui32, handshake.name = "extsi18"} : <i6> to <i9>
    %30 = extsi %27#2 {handshake.bb = 2 : ui32, handshake.name = "extsi19"} : <i6> to <i7>
    %32 = trunci %27#1 {handshake.bb = 2 : ui32, handshake.name = "trunci1"} : <i6> to <i5>
    %34 = mux %42#2 [%20, %trueResult_12] {handshake.bb = 2 : ui32, handshake.name = "mux2"} : <i1>, [<f32>, <f32>] to <f32>
    %36 = mux %42#0 [%22, %trueResult_14] {handshake.bb = 2 : ui32, handshake.name = "mux3"} : <i1>, [<i6>, <i6>] to <i6>
    %38:2 = fork [2] %36 {handshake.bb = 2 : ui32, handshake.name = "fork5"} : <i6>
    %39 = extsi %38#1 {handshake.bb = 2 : ui32, handshake.name = "extsi20"} : <i6> to <i32>
    %41:2 = fork [2] %39 {handshake.bb = 2 : ui32, handshake.name = "fork6"} : <i32>
    %result_6, %index_7 = control_merge [%24, %trueResult_16]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>, <>] to <>, <i1>
    %42:3 = fork [3] %index_7 {handshake.bb = 2 : ui32, handshake.name = "fork7"} : <i1>
    %43:2 = fork [2] %result_6 {handshake.bb = 2 : ui32, handshake.name = "fork8"} : <>
    %44 = constant %43#0 {handshake.bb = 2 : ui32, handshake.name = "constant17", value = false} : <>, <i1>
    %45 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %46 = constant %45 {handshake.bb = 2 : ui32, handshake.name = "constant18", value = 20 : i6} : <>, <i6>
    %47 = extsi %46 {handshake.bb = 2 : ui32, handshake.name = "extsi21"} : <i6> to <i7>
    %48 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %49 = constant %48 {handshake.bb = 2 : ui32, handshake.name = "constant19", value = 1 : i2} : <>, <i2>
    %50 = extsi %49 {handshake.bb = 2 : ui32, handshake.name = "extsi22"} : <i2> to <i7>
    %51 = source {handshake.bb = 2 : ui32, handshake.name = "source2"} : <>
    %52 = constant %51 {handshake.bb = 2 : ui32, handshake.name = "constant20", value = 4 : i4} : <>, <i4>
    %53 = extsi %52 {handshake.bb = 2 : ui32, handshake.name = "extsi5"} : <i4> to <i32>
    %54 = source {handshake.bb = 2 : ui32, handshake.name = "source3"} : <>
    %55 = constant %54 {handshake.bb = 2 : ui32, handshake.name = "constant21", value = 2 : i3} : <>, <i3>
    %56 = extsi %55 {handshake.bb = 2 : ui32, handshake.name = "extsi6"} : <i3> to <i32>
    %57 = shli %41#0, %56 {handshake.bb = 2 : ui32, handshake.name = "shli0"} : <i32>
    %59 = trunci %57 {handshake.bb = 2 : ui32, handshake.name = "trunci2"} : <i32> to <i9>
    %60 = shli %41#1, %53 {handshake.bb = 2 : ui32, handshake.name = "shli1"} : <i32>
    %62 = trunci %60 {handshake.bb = 2 : ui32, handshake.name = "trunci3"} : <i32> to <i9>
    %63 = addi %59, %62 {handshake.bb = 2 : ui32, handshake.name = "addi5"} : <i9>
    %64 = addi %28, %63 {handshake.bb = 2 : ui32, handshake.name = "addi0"} : <i9>
    %addressResult_8, %dataResult_9 = load[%64] %outputs_4#0 {handshake.bb = 2 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i9>, <f32>, <i9>, <f32>
    %addressResult_10, %dataResult_11 = load[%32] %outputs_2 {handshake.bb = 2 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load2"} : <i5>, <f32>, <i5>, <f32>
    %65 = mulf %dataResult_9, %dataResult_11 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "mulf0"} : <f32>
    %66 = addf %34, %65 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "addf0"} : <f32>
    %67 = addi %30, %50 {handshake.bb = 2 : ui32, handshake.name = "addi2"} : <i7>
    %68:2 = fork [2] %67 {handshake.bb = 2 : ui32, handshake.name = "fork9"} : <i7>
    %69 = trunci %68#0 {handshake.bb = 2 : ui32, handshake.name = "trunci4"} : <i7> to <i6>
    %71 = cmpi ult, %68#1, %47 {handshake.bb = 2 : ui32, handshake.name = "cmpi0"} : <i7>
    %73:5 = fork [5] %71 {handshake.bb = 2 : ui32, handshake.name = "fork10"} : <i1>
    %trueResult, %falseResult = cond_br %73#0, %69 {handshake.bb = 2 : ui32, handshake.name = "cond_br3"} : <i1>, <i6>
    sink %falseResult {handshake.name = "sink0"} : <i6>
    %trueResult_12, %falseResult_13 = cond_br %75, %66 {handshake.bb = 2 : ui32, handshake.name = "cond_br4"} : <i1>, <f32>
    %75 = buffer %73#2, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer19"} : <i1>
    %trueResult_14, %falseResult_15 = cond_br %73#1, %38#0 {handshake.bb = 2 : ui32, handshake.name = "cond_br5"} : <i1>, <i6>
    %trueResult_16, %falseResult_17 = cond_br %73#3, %43#1 {handshake.bb = 2 : ui32, handshake.name = "cond_br6"} : <i1>, <>
    %trueResult_18, %falseResult_19 = cond_br %73#4, %44 {handshake.bb = 2 : ui32, handshake.name = "cond_br7"} : <i1>, <i1>
    sink %trueResult_18 {handshake.name = "sink1"} : <i1>
    %80 = extsi %falseResult_19 {handshake.bb = 2 : ui32, handshake.name = "extsi15"} : <i1> to <i6>
    %trueResult_20, %falseResult_21 = cond_br %81, %134 {handshake.bb = 3 : ui32, handshake.name = "cond_br15"} : <i1>, <>
    %81 = buffer %141#3, bufferType = FIFO_BREAK_NONE, numSlots = 15 {handshake.bb = 3 : ui32, handshake.name = "buffer24"} : <i1>
    %82 = mux %83 [%5, %trueResult_20] {ftd.phi, handshake.bb = 3 : ui32, handshake.name = "mux9"} : <i1>, [<>, <>] to <>
    %83 = init %84 {ftd.imerge, handshake.bb = 3 : ui32, handshake.name = "init2"} : <i1>
    %84 = buffer %141#2, bufferType = FIFO_BREAK_NONE, numSlots = 15 {handshake.bb = 3 : ui32, handshake.name = "buffer25"} : <i1>
    %85 = mux %105#1 [%80, %trueResult_30] {handshake.bb = 3 : ui32, handshake.name = "mux4"} : <i1>, [<i6>, <i6>] to <i6>
    %87:4 = fork [4] %85 {handshake.bb = 3 : ui32, handshake.name = "fork11"} : <i6>
    %88 = extsi %89 {handshake.bb = 3 : ui32, handshake.name = "extsi23"} : <i6> to <i9>
    %89 = buffer %87#0, bufferType = FIFO_BREAK_NONE, numSlots = 10 {handshake.bb = 3 : ui32, handshake.name = "buffer27"} : <i6>
    %90 = extsi %87#2 {handshake.bb = 3 : ui32, handshake.name = "extsi24"} : <i6> to <i7>
    %92 = extsi %93 {handshake.bb = 3 : ui32, handshake.name = "extsi25"} : <i6> to <i32>
    %93 = buffer %87#3, bufferType = FIFO_BREAK_NONE, numSlots = 15 {handshake.bb = 3 : ui32, handshake.name = "buffer29"} : <i6>
    %94 = trunci %95 {handshake.bb = 3 : ui32, handshake.name = "trunci5"} : <i6> to <i5>
    %95 = buffer %87#1, bufferType = FIFO_BREAK_NONE, numSlots = 15 {handshake.bb = 3 : ui32, handshake.name = "buffer30"} : <i6>
    %96 = mux %105#0 [%falseResult_15, %trueResult_32] {handshake.bb = 3 : ui32, handshake.name = "mux5"} : <i1>, [<i6>, <i6>] to <i6>
    %98:2 = fork [2] %96 {handshake.bb = 3 : ui32, handshake.name = "fork12"} : <i6>
    %99 = extsi %98#1 {handshake.bb = 3 : ui32, handshake.name = "extsi26"} : <i6> to <i32>
    %101:2 = fork [2] %99 {handshake.bb = 3 : ui32, handshake.name = "fork13"} : <i32>
    %102 = mux %103 [%falseResult_13, %trueResult_34] {handshake.bb = 3 : ui32, handshake.name = "mux6"} : <i1>, [<f32>, <f32>] to <f32>
    %103 = buffer %105#2, bufferType = FIFO_BREAK_NONE, numSlots = 11 {handshake.bb = 3 : ui32, handshake.name = "buffer33"} : <i1>
    %104:2 = fork [2] %102 {handshake.bb = 3 : ui32, handshake.name = "fork14"} : <f32>
    %result_22, %index_23 = control_merge [%falseResult_17, %trueResult_36]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>, <>] to <>, <i1>
    %105:3 = fork [3] %index_23 {handshake.bb = 3 : ui32, handshake.name = "fork15"} : <i1>
    %106:2 = fork [2] %result_22 {handshake.bb = 3 : ui32, handshake.name = "fork16"} : <>
    %107 = constant %106#0 {handshake.bb = 3 : ui32, handshake.name = "constant22", value = 1 : i2} : <>, <i2>
    %108 = extsi %107 {handshake.bb = 3 : ui32, handshake.name = "extsi7"} : <i2> to <i32>
    %109 = source {handshake.bb = 3 : ui32, handshake.name = "source4"} : <>
    %110 = constant %109 {handshake.bb = 3 : ui32, handshake.name = "constant23", value = 20 : i6} : <>, <i6>
    %111 = extsi %110 {handshake.bb = 3 : ui32, handshake.name = "extsi27"} : <i6> to <i7>
    %112 = source {handshake.bb = 3 : ui32, handshake.name = "source5"} : <>
    %113 = constant %112 {handshake.bb = 3 : ui32, handshake.name = "constant24", value = 1 : i2} : <>, <i2>
    %114 = extsi %113 {handshake.bb = 3 : ui32, handshake.name = "extsi28"} : <i2> to <i7>
    %115 = source {handshake.bb = 3 : ui32, handshake.name = "source6"} : <>
    %116 = constant %115 {handshake.bb = 3 : ui32, handshake.name = "constant25", value = 4 : i4} : <>, <i4>
    %117 = extsi %116 {handshake.bb = 3 : ui32, handshake.name = "extsi10"} : <i4> to <i32>
    %118 = source {handshake.bb = 3 : ui32, handshake.name = "source7"} : <>
    %119 = constant %118 {handshake.bb = 3 : ui32, handshake.name = "constant26", value = 2 : i3} : <>, <i3>
    %120 = extsi %119 {handshake.bb = 3 : ui32, handshake.name = "extsi11"} : <i3> to <i32>
    %121 = gate %92, %82 {handshake.bb = 3 : ui32, handshake.name = "gate0"} : <i32>, !handshake.control<> to <i32>
    %122 = trunci %121 {handshake.bb = 3 : ui32, handshake.name = "trunci6"} : <i32> to <i5>
    %addressResult_24, %dataResult_25 = load[%122] %outputs_0#0 {handshake.bb = 3 : ui32, handshake.deps = #handshake<deps[["store0", 1, false], ["store0", 3, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load3"} : <i5>, <f32>, <i5>, <f32>
    %123 = shli %124, %120 {handshake.bb = 3 : ui32, handshake.name = "shli2"} : <i32>
    %124 = buffer %101#0, bufferType = FIFO_BREAK_NONE, numSlots = 10 {handshake.bb = 3 : ui32, handshake.name = "buffer34"} : <i32>
    %125 = trunci %123 {handshake.bb = 3 : ui32, handshake.name = "trunci7"} : <i32> to <i9>
    %126 = shli %127, %117 {handshake.bb = 3 : ui32, handshake.name = "shli3"} : <i32>
    %127 = buffer %101#1, bufferType = FIFO_BREAK_NONE, numSlots = 10 {handshake.bb = 3 : ui32, handshake.name = "buffer35"} : <i32>
    %128 = trunci %126 {handshake.bb = 3 : ui32, handshake.name = "trunci8"} : <i32> to <i9>
    %129 = addi %125, %128 {handshake.bb = 3 : ui32, handshake.name = "addi6"} : <i9>
    %130 = addi %88, %129 {handshake.bb = 3 : ui32, handshake.name = "addi1"} : <i9>
    %addressResult_26, %dataResult_27 = load[%130] %outputs_4#1 {handshake.bb = 3 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load4"} : <i9>, <f32>, <i9>, <f32>
    %131 = mulf %dataResult_27, %104#1 {fastmath = #arith.fastmath<none>, handshake.bb = 3 : ui32, handshake.name = "mulf1"} : <f32>
    %133 = addf %dataResult_25, %131 {fastmath = #arith.fastmath<none>, handshake.bb = 3 : ui32, handshake.name = "addf1"} : <f32>
    %134 = buffer %doneResult, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer1"} : <>
    %addressResult_28, %dataResult_29, %doneResult = store[%94] %133 %outputs_0#1 {handshake.bb = 3 : ui32, handshake.deps = #handshake<deps[["load3", 1, false], ["store0", 1, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store0"} : <i5>, <f32>, <>, <i5>, <f32>, <>
    %135 = addi %90, %114 {handshake.bb = 3 : ui32, handshake.name = "addi3"} : <i7>
    %136:2 = fork [2] %135 {handshake.bb = 3 : ui32, handshake.name = "fork17"} : <i7>
    %137 = trunci %136#0 {handshake.bb = 3 : ui32, handshake.name = "trunci9"} : <i7> to <i6>
    %139 = cmpi ult, %136#1, %111 {handshake.bb = 3 : ui32, handshake.name = "cmpi1"} : <i7>
    %141:6 = fork [6] %139 {handshake.bb = 3 : ui32, handshake.name = "fork18"} : <i1>
    %trueResult_30, %falseResult_31 = cond_br %141#0, %137 {handshake.bb = 3 : ui32, handshake.name = "cond_br8"} : <i1>, <i6>
    sink %falseResult_31 {handshake.name = "sink2"} : <i6>
    %trueResult_32, %falseResult_33 = cond_br %141#1, %98#0 {handshake.bb = 3 : ui32, handshake.name = "cond_br9"} : <i1>, <i6>
    %trueResult_34, %falseResult_35 = cond_br %145, %104#0 {handshake.bb = 3 : ui32, handshake.name = "cond_br10"} : <i1>, <f32>
    %145 = buffer %141#4, bufferType = FIFO_BREAK_NONE, numSlots = 12 {handshake.bb = 3 : ui32, handshake.name = "buffer42"} : <i1>
    %trueResult_36, %falseResult_37 = cond_br %141#5, %106#1 {handshake.bb = 3 : ui32, handshake.name = "cond_br11"} : <i1>, <>
    %trueResult_38, %falseResult_39 = cond_br %148, %falseResult_21 {handshake.bb = 4 : ui32, handshake.name = "cond_br16"} : <i1>, <>
    %148 = buffer %173#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer45"} : <i1>
    sink %falseResult_39 {handshake.name = "sink3"} : <>
    %149 = merge %falseResult_33 {handshake.bb = 4 : ui32, handshake.name = "merge0"} : <i6>
    %150:2 = fork [2] %149 {handshake.bb = 4 : ui32, handshake.name = "fork19"} : <i6>
    %151 = extsi %150#0 {handshake.bb = 4 : ui32, handshake.name = "extsi29"} : <i6> to <i7>
    %153 = extsi %150#1 {handshake.bb = 4 : ui32, handshake.name = "extsi30"} : <i6> to <i32>
    %155 = merge %falseResult_35 {handshake.bb = 4 : ui32, handshake.name = "merge1"} : <f32>
    %result_40, %index_41 = control_merge [%falseResult_37]  {handshake.bb = 4 : ui32, handshake.name = "control_merge3"} : [<>] to <>, <i1>
    sink %index_41 {handshake.name = "sink4"} : <i1>
    %156:2 = fork [2] %result_40 {handshake.bb = 4 : ui32, handshake.name = "fork20"} : <>
    %157 = constant %156#0 {handshake.bb = 4 : ui32, handshake.name = "constant27", value = 1 : i2} : <>, <i2>
    %158 = extsi %157 {handshake.bb = 4 : ui32, handshake.name = "extsi12"} : <i2> to <i32>
    %159 = source {handshake.bb = 4 : ui32, handshake.name = "source8"} : <>
    %160 = constant %159 {handshake.bb = 4 : ui32, handshake.name = "constant28", value = 20 : i6} : <>, <i6>
    %161 = extsi %160 {handshake.bb = 4 : ui32, handshake.name = "extsi31"} : <i6> to <i7>
    %162 = source {handshake.bb = 4 : ui32, handshake.name = "source9"} : <>
    %163 = constant %162 {handshake.bb = 4 : ui32, handshake.name = "constant29", value = 1 : i2} : <>, <i2>
    %164 = extsi %163 {handshake.bb = 4 : ui32, handshake.name = "extsi32"} : <i2> to <i7>
    %165 = gate %153, %16 {handshake.bb = 4 : ui32, handshake.name = "gate1"} : <i32>, !handshake.control<> to <i32>
    %166 = trunci %165 {handshake.bb = 4 : ui32, handshake.name = "trunci10"} : <i32> to <i5>
    %addressResult_42, %dataResult_43, %doneResult_44 = store[%166] %155 %outputs#1 {handshake.bb = 4 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store1"} : <i5>, <f32>, <>, <i5>, <f32>, <>
    sink %doneResult_44 {handshake.name = "sink5"} : <>
    %167 = addi %151, %164 {handshake.bb = 4 : ui32, handshake.name = "addi4"} : <i7>
    %168:2 = fork [2] %167 {handshake.bb = 4 : ui32, handshake.name = "fork21"} : <i7>
    %169 = trunci %168#0 {handshake.bb = 4 : ui32, handshake.name = "trunci11"} : <i7> to <i6>
    %171 = cmpi ult, %168#1, %161 {handshake.bb = 4 : ui32, handshake.name = "cmpi2"} : <i7>
    %173:4 = fork [4] %171 {handshake.bb = 4 : ui32, handshake.name = "fork22"} : <i1>
    %trueResult_45, %falseResult_46 = cond_br %173#0, %169 {handshake.bb = 4 : ui32, handshake.name = "cond_br12"} : <i1>, <i6>
    sink %falseResult_46 {handshake.name = "sink6"} : <i6>
    %trueResult_47, %falseResult_48 = cond_br %173#3, %156#1 {handshake.bb = 4 : ui32, handshake.name = "cond_br13"} : <i1>, <>
    %result_49, %index_50 = control_merge [%falseResult_48]  {handshake.bb = 5 : ui32, handshake.name = "control_merge4"} : [<>] to <>, <i1>
    sink %index_50 {handshake.name = "sink7"} : <i1>
    %176:4 = fork [4] %result_49 {handshake.bb = 5 : ui32, handshake.name = "fork23"} : <>
    end {handshake.bb = 5 : ui32, handshake.name = "end0"} %memEnd_5, %memEnd_3, %memEnd_1, %memEnd, %0#1 : <>, <>, <>, <>, <>
  }
}

