module {
  handshake.func @mvt_float(%arg0: memref<900xf32>, %arg1: memref<30xf32>, %arg2: memref<30xf32>, %arg3: memref<30xf32>, %arg4: memref<30xf32>, %arg5: !handshake.control<>, %arg6: !handshake.control<>, %arg7: !handshake.control<>, %arg8: !handshake.control<>, %arg9: !handshake.control<>, %arg10: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["A", "x1", "x2", "y1", "y2", "A_start", "x1_start", "x2_start", "y1_start", "y2_start", "start"], cfg.edges = "[0,1][2,2,3,cmpi0][4,5][6,4,7,cmpi3][1,2][3,1,4,cmpi1][5,5,6,cmpi2]", resNames = ["A_end", "x1_end", "x2_end", "y1_end", "y2_end", "end"]} {
    %0:3 = fork [3] %arg10 {handshake.bb = 0 : ui32, handshake.name = "fork0"} : <>
    %outputs, %memEnd = mem_controller[%arg4 : memref<30xf32>] %arg9 (%addressResult_40) %220#4 {connectedBlocks = [5 : i32], handshake.name = "mem_controller3"} :    (!handshake.channel<i5>) -> !handshake.channel<f32>
    %outputs_0, %memEnd_1 = mem_controller[%arg3 : memref<30xf32>] %arg8 (%addressResult_12) %220#3 {connectedBlocks = [2 : i32], handshake.name = "mem_controller4"} :    (!handshake.channel<i5>) -> !handshake.channel<f32>
    %outputs_2:2, %memEnd_3 = mem_controller[%arg2 : memref<30xf32>] %arg7 (%addressResult_34, %192, %addressResult_54, %dataResult_55) %220#2 {connectedBlocks = [4 : i32, 6 : i32], handshake.name = "mem_controller5"} :    (!handshake.channel<i5>, !handshake.channel<i32>, !handshake.channel<i5>, !handshake.channel<f32>) -> (!handshake.channel<f32>, !handshake.control<>)
    %outputs_4:2, %memEnd_5 = mem_controller[%arg1 : memref<30xf32>] %arg6 (%addressResult, %83, %addressResult_24, %dataResult_25) %220#1 {connectedBlocks = [1 : i32, 3 : i32], handshake.name = "mem_controller6"} :    (!handshake.channel<i5>, !handshake.channel<i32>, !handshake.channel<i5>, !handshake.channel<f32>) -> (!handshake.channel<f32>, !handshake.control<>)
    %outputs_6:2, %memEnd_7 = mem_controller[%arg0 : memref<900xf32>] %arg5 (%addressResult_10, %addressResult_38) %220#0 {connectedBlocks = [2 : i32, 5 : i32], handshake.name = "mem_controller7"} :    (!handshake.channel<i10>, !handshake.channel<i10>) -> (!handshake.channel<f32>, !handshake.channel<f32>)
    %1 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant0", value = false} : <>, <i1>
    %2 = br %1 {handshake.bb = 0 : ui32, handshake.name = "br2"} : <i1>
    %3 = extsi %2 {handshake.bb = 0 : ui32, handshake.name = "extsi17"} : <i1> to <i6>
    %4 = br %0#2 {handshake.bb = 0 : ui32, handshake.name = "br4"} : <>
    %5 = init %109#1 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "init0"} : <i1>
    sink %5 {handshake.name = "sink0"} : <i1>
    %7:2 = unbundle %21#0  {handshake.bb = 1 : ui32, handshake.name = "unbundle1"} : <f32> to _ 
    %9 = mux %index [%3, %trueResult_26] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i6>, <i6>] to <i6>
    %10:3 = fork [3] %9 {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <i6>
    %11 = trunci %10#0 {handshake.bb = 1 : ui32, handshake.name = "trunci0"} : <i6> to <i5>
    %result, %index = control_merge [%4, %trueResult_28]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %13:2 = fork [2] %result {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <>
    %14 = constant %13#0 {handshake.bb = 1 : ui32, handshake.name = "constant1", value = false} : <>, <i1>
    %15 = buffer %10#2, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer0"} : <i6>
    %17 = extsi %15 {handshake.bb = 1 : ui32, handshake.name = "extsi18"} : <i6> to <i32>
    %18 = buffer %7#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 1 : ui32, handshake.name = "buffer1"} : <>
    %19:2 = fork [2] %18 {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <>
    %20 = init %19#1 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 1 : ui32, handshake.name = "init3"} : <>
    %addressResult, %dataResult = load[%11] %outputs_4#0 {handshake.bb = 1 : ui32, handshake.deps = #handshake<deps[["store0", 2, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i5>, <f32>, <i5>, <f32>
    %21:2 = fork [2] %dataResult {handshake.bb = 1 : ui32, handshake.name = "fork4"} : <f32>
    %22 = br %14 {handshake.bb = 1 : ui32, handshake.name = "br5"} : <i1>
    %23 = extsi %22 {handshake.bb = 1 : ui32, handshake.name = "extsi16"} : <i1> to <i6>
    %24 = br %21#1 {handshake.bb = 1 : ui32, handshake.name = "br6"} : <f32>
    %26 = br %10#1 {handshake.bb = 1 : ui32, handshake.name = "br7"} : <i6>
    %28 = br %13#1 {handshake.bb = 1 : ui32, handshake.name = "br8"} : <>
    %29 = mux %45#1 [%23, %trueResult] {handshake.bb = 2 : ui32, handshake.name = "mux1"} : <i1>, [<i6>, <i6>] to <i6>
    %31:3 = fork [3] %29 {handshake.bb = 2 : ui32, handshake.name = "fork5"} : <i6>
    %32 = extsi %33 {handshake.bb = 2 : ui32, handshake.name = "extsi19"} : <i6> to <i10>
    %33 = buffer %31#0, bufferType = FIFO_BREAK_NONE, numSlots = 9 {handshake.bb = 2 : ui32, handshake.name = "buffer11"} : <i6>
    %34 = extsi %31#2 {handshake.bb = 2 : ui32, handshake.name = "extsi20"} : <i6> to <i7>
    %36 = trunci %37 {handshake.bb = 2 : ui32, handshake.name = "trunci1"} : <i6> to <i5>
    %37 = buffer %31#1, bufferType = FIFO_BREAK_NONE, numSlots = 9 {handshake.bb = 2 : ui32, handshake.name = "buffer13"} : <i6>
    %38 = mux %39 [%24, %trueResult_14] {handshake.bb = 2 : ui32, handshake.name = "mux2"} : <i1>, [<f32>, <f32>] to <f32>
    %39 = buffer %45#2, bufferType = FIFO_BREAK_NONE, numSlots = 15 {handshake.bb = 2 : ui32, handshake.name = "buffer14"} : <i1>
    %40 = mux %45#0 [%26, %trueResult_16] {handshake.bb = 2 : ui32, handshake.name = "mux3"} : <i1>, [<i6>, <i6>] to <i6>
    %42:2 = fork [2] %40 {handshake.bb = 2 : ui32, handshake.name = "fork6"} : <i6>
    %43 = extsi %44 {handshake.bb = 2 : ui32, handshake.name = "extsi21"} : <i6> to <i12>
    %44 = buffer %42#1, bufferType = FIFO_BREAK_NONE, numSlots = 5 {handshake.bb = 2 : ui32, handshake.name = "buffer16"} : <i6>
    %result_8, %index_9 = control_merge [%28, %trueResult_18]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>, <>] to <>, <i1>
    %45:3 = fork [3] %index_9 {handshake.bb = 2 : ui32, handshake.name = "fork7"} : <i1>
    %46 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %47 = constant %46 {handshake.bb = 2 : ui32, handshake.name = "constant16", value = 30 : i6} : <>, <i6>
    %48:2 = fork [2] %47 {handshake.bb = 2 : ui32, handshake.name = "fork8"} : <i6>
    %49 = extsi %50 {handshake.bb = 2 : ui32, handshake.name = "extsi22"} : <i6> to <i12>
    %50 = buffer %48#0, bufferType = FIFO_BREAK_NONE, numSlots = 5 {handshake.bb = 2 : ui32, handshake.name = "buffer17"} : <i6>
    %51 = extsi %48#1 {handshake.bb = 2 : ui32, handshake.name = "extsi23"} : <i6> to <i7>
    %53 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %54 = constant %53 {handshake.bb = 2 : ui32, handshake.name = "constant17", value = 1 : i2} : <>, <i2>
    %55 = extsi %54 {handshake.bb = 2 : ui32, handshake.name = "extsi24"} : <i2> to <i7>
    %56 = muli %43, %49 {handshake.bb = 2 : ui32, handshake.name = "muli0"} : <i12>
    %57 = trunci %56 {handshake.bb = 2 : ui32, handshake.name = "trunci2"} : <i12> to <i10>
    %58 = addi %32, %57 {handshake.bb = 2 : ui32, handshake.name = "addi0"} : <i10>
    %addressResult_10, %dataResult_11 = load[%58] %outputs_6#0 {handshake.bb = 2 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i10>, <f32>, <i10>, <f32>
    %addressResult_12, %dataResult_13 = load[%36] %outputs_0 {handshake.bb = 2 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load2"} : <i5>, <f32>, <i5>, <f32>
    %59 = mulf %dataResult_11, %dataResult_13 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "mulf0"} : <f32>
    %60 = addf %38, %59 {fastmath = #arith.fastmath<none>, handshake.bb = 2 : ui32, handshake.name = "addf0"} : <f32>
    %61 = addi %34, %55 {handshake.bb = 2 : ui32, handshake.name = "addi2"} : <i7>
    %62:2 = fork [2] %61 {handshake.bb = 2 : ui32, handshake.name = "fork9"} : <i7>
    %63 = trunci %62#0 {handshake.bb = 2 : ui32, handshake.name = "trunci3"} : <i7> to <i6>
    %65 = cmpi ult, %62#1, %51 {handshake.bb = 2 : ui32, handshake.name = "cmpi0"} : <i7>
    %67:4 = fork [4] %65 {handshake.bb = 2 : ui32, handshake.name = "fork10"} : <i1>
    %trueResult, %falseResult = cond_br %67#0, %63 {handshake.bb = 2 : ui32, handshake.name = "cond_br4"} : <i1>, <i6>
    sink %falseResult {handshake.name = "sink1"} : <i6>
    %trueResult_14, %falseResult_15 = cond_br %69, %60 {handshake.bb = 2 : ui32, handshake.name = "cond_br5"} : <i1>, <f32>
    %69 = buffer %67#2, bufferType = FIFO_BREAK_NONE, numSlots = 15 {handshake.bb = 2 : ui32, handshake.name = "buffer22"} : <i1>
    %trueResult_16, %falseResult_17 = cond_br %67#1, %42#0 {handshake.bb = 2 : ui32, handshake.name = "cond_br6"} : <i1>, <i6>
    %trueResult_18, %falseResult_19 = cond_br %67#3, %result_8 {handshake.bb = 2 : ui32, handshake.name = "cond_br7"} : <i1>, <>
    %73 = merge %falseResult_17 {handshake.bb = 3 : ui32, handshake.name = "merge0"} : <i6>
    %74:2 = fork [2] %73 {handshake.bb = 3 : ui32, handshake.name = "fork11"} : <i6>
    %75 = extsi %74#0 {handshake.bb = 3 : ui32, handshake.name = "extsi25"} : <i6> to <i7>
    %77 = extsi %74#1 {handshake.bb = 3 : ui32, handshake.name = "extsi26"} : <i6> to <i32>
    %79:2 = fork [2] %77 {handshake.bb = 3 : ui32, handshake.name = "fork12"} : <i32>
    %80 = merge %falseResult_15 {handshake.bb = 3 : ui32, handshake.name = "merge1"} : <f32>
    %result_20, %index_21 = control_merge [%falseResult_19]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>] to <>, <i1>
    sink %index_21 {handshake.name = "sink2"} : <i1>
    %81:3 = fork [3] %result_20 {handshake.bb = 3 : ui32, handshake.name = "fork13"} : <>
    %82 = constant %81#1 {handshake.bb = 3 : ui32, handshake.name = "constant18", value = 1 : i2} : <>, <i2>
    %83 = extsi %82 {handshake.bb = 3 : ui32, handshake.name = "extsi4"} : <i2> to <i32>
    %84 = constant %81#0 {handshake.bb = 3 : ui32, handshake.name = "constant19", value = false} : <>, <i1>
    %85 = source {handshake.bb = 3 : ui32, handshake.name = "source2"} : <>
    %86 = constant %85 {handshake.bb = 3 : ui32, handshake.name = "constant20", value = 30 : i6} : <>, <i6>
    %87 = extsi %86 {handshake.bb = 3 : ui32, handshake.name = "extsi27"} : <i6> to <i7>
    %88 = source {handshake.bb = 3 : ui32, handshake.name = "source3"} : <>
    %89 = constant %88 {handshake.bb = 3 : ui32, handshake.name = "constant21", value = 1 : i2} : <>, <i2>
    %90 = extsi %89 {handshake.bb = 3 : ui32, handshake.name = "extsi28"} : <i2> to <i7>
    %91 = gate %79#0, %20 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 3 : ui32, handshake.name = "gate0"} : <i32>, !handshake.control<> to <i32>
    %93 = cmpi ne, %91, %17 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 3 : ui32, handshake.name = "cmpi4"} : <i32>
    %94:2 = fork [2] %93 {handshake.bb = 3 : ui32, handshake.name = "fork14"} : <i1>
    %trueResult_22, %falseResult_23 = cond_br %95, %19#0 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 3 : ui32, handshake.name = "cond_br3"} : <i1>, <>
    %95 = buffer %94#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer29"} : <i1>
    sink %trueResult_22 {handshake.name = "sink3"} : <>
    %96 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 3 : ui32, handshake.name = "source8"} : <>
    %97 = mux %98 [%falseResult_23, %96] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 3 : ui32, handshake.name = "mux11"} : <i1>, [<>, <>] to <>
    %98 = buffer %94#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer30"} : <i1>
    %99 = join %97 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 3 : ui32, handshake.name = "join0"} : <>
    %100 = gate %101, %99 {handshake.bb = 3 : ui32, handshake.name = "gate1"} : <i32>, !handshake.control<> to <i32>
    %101 = buffer %79#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer31"} : <i32>
    %102 = trunci %100 {handshake.bb = 3 : ui32, handshake.name = "trunci4"} : <i32> to <i5>
    %addressResult_24, %dataResult_25, %doneResult = store[%102] %80 %outputs_4#1 {handshake.bb = 3 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store0"} : <i5>, <f32>, <>, <i5>, <f32>, <>
    sink %doneResult {handshake.name = "sink4"} : <>
    %103 = addi %75, %90 {handshake.bb = 3 : ui32, handshake.name = "addi3"} : <i7>
    %104:2 = fork [2] %103 {handshake.bb = 3 : ui32, handshake.name = "fork15"} : <i7>
    %105 = trunci %104#0 {handshake.bb = 3 : ui32, handshake.name = "trunci5"} : <i7> to <i6>
    %107 = cmpi ult, %104#1, %87 {handshake.bb = 3 : ui32, handshake.name = "cmpi1"} : <i7>
    %109:4 = fork [4] %107 {handshake.bb = 3 : ui32, handshake.name = "fork16"} : <i1>
    %trueResult_26, %falseResult_27 = cond_br %109#0, %105 {handshake.bb = 3 : ui32, handshake.name = "cond_br8"} : <i1>, <i6>
    sink %falseResult_27 {handshake.name = "sink5"} : <i6>
    %trueResult_28, %falseResult_29 = cond_br %109#2, %81#2 {handshake.bb = 3 : ui32, handshake.name = "cond_br9"} : <i1>, <>
    %trueResult_30, %falseResult_31 = cond_br %109#3, %84 {handshake.bb = 3 : ui32, handshake.name = "cond_br10"} : <i1>, <i1>
    sink %trueResult_30 {handshake.name = "sink6"} : <i1>
    %113 = extsi %falseResult_31 {handshake.bb = 3 : ui32, handshake.name = "extsi15"} : <i1> to <i6>
    %114 = init %217#1 {ftd.imerge, handshake.bb = 4 : ui32, handshake.name = "init4"} : <i1>
    sink %114 {handshake.name = "sink7"} : <i1>
    %116:2 = unbundle %130#0  {handshake.bb = 4 : ui32, handshake.name = "unbundle3"} : <f32> to _ 
    %118 = mux %index_33 [%113, %trueResult_57] {handshake.bb = 4 : ui32, handshake.name = "mux4"} : <i1>, [<i6>, <i6>] to <i6>
    %119:3 = fork [3] %118 {handshake.bb = 4 : ui32, handshake.name = "fork17"} : <i6>
    %120 = trunci %119#0 {handshake.bb = 4 : ui32, handshake.name = "trunci6"} : <i6> to <i5>
    %result_32, %index_33 = control_merge [%falseResult_29, %trueResult_59]  {handshake.bb = 4 : ui32, handshake.name = "control_merge3"} : [<>, <>] to <>, <i1>
    %122:2 = fork [2] %result_32 {handshake.bb = 4 : ui32, handshake.name = "fork18"} : <>
    %123 = constant %122#0 {handshake.bb = 4 : ui32, handshake.name = "constant22", value = false} : <>, <i1>
    %124 = buffer %119#2, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer2"} : <i6>
    %126 = extsi %124 {handshake.bb = 4 : ui32, handshake.name = "extsi29"} : <i6> to <i32>
    %127 = buffer %116#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 4 : ui32, handshake.name = "buffer3"} : <>
    %128:2 = fork [2] %127 {handshake.bb = 4 : ui32, handshake.name = "fork19"} : <>
    %129 = init %128#1 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 4 : ui32, handshake.name = "init7"} : <>
    %addressResult_34, %dataResult_35 = load[%120] %outputs_2#0 {handshake.bb = 4 : ui32, handshake.deps = #handshake<deps[["store1", 2, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load3"} : <i5>, <f32>, <i5>, <f32>
    %130:2 = fork [2] %dataResult_35 {handshake.bb = 4 : ui32, handshake.name = "fork20"} : <f32>
    %131 = br %123 {handshake.bb = 4 : ui32, handshake.name = "br9"} : <i1>
    %132 = extsi %131 {handshake.bb = 4 : ui32, handshake.name = "extsi14"} : <i1> to <i6>
    %133 = br %130#1 {handshake.bb = 4 : ui32, handshake.name = "br10"} : <f32>
    %135 = br %119#1 {handshake.bb = 4 : ui32, handshake.name = "br11"} : <i6>
    %137 = br %122#1 {handshake.bb = 4 : ui32, handshake.name = "br12"} : <>
    %138 = mux %154#1 [%132, %trueResult_42] {handshake.bb = 5 : ui32, handshake.name = "mux5"} : <i1>, [<i6>, <i6>] to <i6>
    %140:3 = fork [3] %138 {handshake.bb = 5 : ui32, handshake.name = "fork21"} : <i6>
    %141 = extsi %140#1 {handshake.bb = 5 : ui32, handshake.name = "extsi30"} : <i6> to <i7>
    %143 = extsi %140#2 {handshake.bb = 5 : ui32, handshake.name = "extsi31"} : <i6> to <i12>
    %145 = trunci %140#0 {handshake.bb = 5 : ui32, handshake.name = "trunci7"} : <i6> to <i5>
    %147 = mux %148 [%133, %trueResult_44] {handshake.bb = 5 : ui32, handshake.name = "mux6"} : <i1>, [<f32>, <f32>] to <f32>
    %148 = buffer %154#2, bufferType = FIFO_BREAK_NONE, numSlots = 3 {handshake.bb = 5 : ui32, handshake.name = "buffer47"} : <i1>
    %149 = mux %154#0 [%135, %trueResult_46] {handshake.bb = 5 : ui32, handshake.name = "mux7"} : <i1>, [<i6>, <i6>] to <i6>
    %151:2 = fork [2] %149 {handshake.bb = 5 : ui32, handshake.name = "fork22"} : <i6>
    %152 = extsi %151#0 {handshake.bb = 5 : ui32, handshake.name = "extsi32"} : <i6> to <i10>
    %result_36, %index_37 = control_merge [%137, %trueResult_48]  {handshake.bb = 5 : ui32, handshake.name = "control_merge4"} : [<>, <>] to <>, <i1>
    %154:3 = fork [3] %index_37 {handshake.bb = 5 : ui32, handshake.name = "fork23"} : <i1>
    %155 = source {handshake.bb = 5 : ui32, handshake.name = "source4"} : <>
    %156 = constant %155 {handshake.bb = 5 : ui32, handshake.name = "constant23", value = 30 : i6} : <>, <i6>
    %157:2 = fork [2] %156 {handshake.bb = 5 : ui32, handshake.name = "fork24"} : <i6>
    %158 = extsi %157#0 {handshake.bb = 5 : ui32, handshake.name = "extsi33"} : <i6> to <i7>
    %160 = extsi %157#1 {handshake.bb = 5 : ui32, handshake.name = "extsi34"} : <i6> to <i12>
    %162 = source {handshake.bb = 5 : ui32, handshake.name = "source5"} : <>
    %163 = constant %162 {handshake.bb = 5 : ui32, handshake.name = "constant24", value = 1 : i2} : <>, <i2>
    %164 = extsi %163 {handshake.bb = 5 : ui32, handshake.name = "extsi35"} : <i2> to <i7>
    %165 = muli %143, %160 {handshake.bb = 5 : ui32, handshake.name = "muli1"} : <i12>
    %166 = trunci %165 {handshake.bb = 5 : ui32, handshake.name = "trunci8"} : <i12> to <i10>
    %167 = addi %152, %166 {handshake.bb = 5 : ui32, handshake.name = "addi1"} : <i10>
    %addressResult_38, %dataResult_39 = load[%167] %outputs_6#1 {handshake.bb = 5 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load4"} : <i10>, <f32>, <i10>, <f32>
    %addressResult_40, %dataResult_41 = load[%145] %outputs {handshake.bb = 5 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load5"} : <i5>, <f32>, <i5>, <f32>
    %168 = mulf %dataResult_39, %dataResult_41 {fastmath = #arith.fastmath<none>, handshake.bb = 5 : ui32, handshake.name = "mulf1"} : <f32>
    %169 = addf %147, %168 {fastmath = #arith.fastmath<none>, handshake.bb = 5 : ui32, handshake.name = "addf1"} : <f32>
    %170 = addi %141, %164 {handshake.bb = 5 : ui32, handshake.name = "addi4"} : <i7>
    %171:2 = fork [2] %170 {handshake.bb = 5 : ui32, handshake.name = "fork25"} : <i7>
    %172 = trunci %171#0 {handshake.bb = 5 : ui32, handshake.name = "trunci9"} : <i7> to <i6>
    %174 = cmpi ult, %171#1, %158 {handshake.bb = 5 : ui32, handshake.name = "cmpi2"} : <i7>
    %176:4 = fork [4] %174 {handshake.bb = 5 : ui32, handshake.name = "fork26"} : <i1>
    %trueResult_42, %falseResult_43 = cond_br %176#0, %172 {handshake.bb = 5 : ui32, handshake.name = "cond_br11"} : <i1>, <i6>
    sink %falseResult_43 {handshake.name = "sink8"} : <i6>
    %trueResult_44, %falseResult_45 = cond_br %178, %169 {handshake.bb = 5 : ui32, handshake.name = "cond_br12"} : <i1>, <f32>
    %178 = buffer %176#2, bufferType = FIFO_BREAK_NONE, numSlots = 4 {handshake.bb = 5 : ui32, handshake.name = "buffer55"} : <i1>
    %trueResult_46, %falseResult_47 = cond_br %176#1, %151#1 {handshake.bb = 5 : ui32, handshake.name = "cond_br13"} : <i1>, <i6>
    %trueResult_48, %falseResult_49 = cond_br %176#3, %result_36 {handshake.bb = 5 : ui32, handshake.name = "cond_br14"} : <i1>, <>
    %182 = merge %falseResult_47 {handshake.bb = 6 : ui32, handshake.name = "merge2"} : <i6>
    %183:2 = fork [2] %182 {handshake.bb = 6 : ui32, handshake.name = "fork27"} : <i6>
    %184 = extsi %183#0 {handshake.bb = 6 : ui32, handshake.name = "extsi36"} : <i6> to <i7>
    %186 = extsi %183#1 {handshake.bb = 6 : ui32, handshake.name = "extsi37"} : <i6> to <i32>
    %188:2 = fork [2] %186 {handshake.bb = 6 : ui32, handshake.name = "fork28"} : <i32>
    %189 = merge %falseResult_45 {handshake.bb = 6 : ui32, handshake.name = "merge3"} : <f32>
    %result_50, %index_51 = control_merge [%falseResult_49]  {handshake.bb = 6 : ui32, handshake.name = "control_merge5"} : [<>] to <>, <i1>
    sink %index_51 {handshake.name = "sink9"} : <i1>
    %190:2 = fork [2] %result_50 {handshake.bb = 6 : ui32, handshake.name = "fork29"} : <>
    %191 = constant %190#0 {handshake.bb = 6 : ui32, handshake.name = "constant25", value = 1 : i2} : <>, <i2>
    %192 = extsi %191 {handshake.bb = 6 : ui32, handshake.name = "extsi11"} : <i2> to <i32>
    %193 = source {handshake.bb = 6 : ui32, handshake.name = "source6"} : <>
    %194 = constant %193 {handshake.bb = 6 : ui32, handshake.name = "constant26", value = 30 : i6} : <>, <i6>
    %195 = extsi %194 {handshake.bb = 6 : ui32, handshake.name = "extsi38"} : <i6> to <i7>
    %196 = source {handshake.bb = 6 : ui32, handshake.name = "source7"} : <>
    %197 = constant %196 {handshake.bb = 6 : ui32, handshake.name = "constant27", value = 1 : i2} : <>, <i2>
    %198 = extsi %197 {handshake.bb = 6 : ui32, handshake.name = "extsi39"} : <i2> to <i7>
    %199 = gate %188#0, %129 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 6 : ui32, handshake.name = "gate2"} : <i32>, !handshake.control<> to <i32>
    %201 = cmpi ne, %199, %126 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 6 : ui32, handshake.name = "cmpi5"} : <i32>
    %202:2 = fork [2] %201 {handshake.bb = 6 : ui32, handshake.name = "fork30"} : <i1>
    %trueResult_52, %falseResult_53 = cond_br %203, %128#0 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 6 : ui32, handshake.name = "cond_br20"} : <i1>, <>
    %203 = buffer %202#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 6 : ui32, handshake.name = "buffer62"} : <i1>
    sink %trueResult_52 {handshake.name = "sink10"} : <>
    %204 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 6 : ui32, handshake.name = "source9"} : <>
    %205 = mux %206 [%falseResult_53, %204] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 6 : ui32, handshake.name = "mux15"} : <i1>, [<>, <>] to <>
    %206 = buffer %202#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 6 : ui32, handshake.name = "buffer63"} : <i1>
    %207 = join %205 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 6 : ui32, handshake.name = "join1"} : <>
    %208 = gate %209, %207 {handshake.bb = 6 : ui32, handshake.name = "gate3"} : <i32>, !handshake.control<> to <i32>
    %209 = buffer %188#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 6 : ui32, handshake.name = "buffer64"} : <i32>
    %210 = trunci %208 {handshake.bb = 6 : ui32, handshake.name = "trunci10"} : <i32> to <i5>
    %addressResult_54, %dataResult_55, %doneResult_56 = store[%210] %189 %outputs_2#1 {handshake.bb = 6 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store1"} : <i5>, <f32>, <>, <i5>, <f32>, <>
    sink %doneResult_56 {handshake.name = "sink11"} : <>
    %211 = addi %184, %198 {handshake.bb = 6 : ui32, handshake.name = "addi5"} : <i7>
    %212:2 = fork [2] %211 {handshake.bb = 6 : ui32, handshake.name = "fork31"} : <i7>
    %213 = trunci %212#0 {handshake.bb = 6 : ui32, handshake.name = "trunci11"} : <i7> to <i6>
    %215 = cmpi ult, %212#1, %195 {handshake.bb = 6 : ui32, handshake.name = "cmpi3"} : <i7>
    %217:3 = fork [3] %215 {handshake.bb = 6 : ui32, handshake.name = "fork32"} : <i1>
    %trueResult_57, %falseResult_58 = cond_br %217#0, %213 {handshake.bb = 6 : ui32, handshake.name = "cond_br15"} : <i1>, <i6>
    sink %falseResult_58 {handshake.name = "sink12"} : <i6>
    %trueResult_59, %falseResult_60 = cond_br %217#2, %190#1 {handshake.bb = 6 : ui32, handshake.name = "cond_br16"} : <i1>, <>
    %result_61, %index_62 = control_merge [%falseResult_60]  {handshake.bb = 7 : ui32, handshake.name = "control_merge6"} : [<>] to <>, <i1>
    sink %index_62 {handshake.name = "sink13"} : <i1>
    %220:5 = fork [5] %result_61 {handshake.bb = 7 : ui32, handshake.name = "fork33"} : <>
    end {handshake.bb = 7 : ui32, handshake.name = "end0"} %memEnd_7, %memEnd_5, %memEnd_3, %memEnd_1, %memEnd, %0#1 : <>, <>, <>, <>, <>, <>
  }
}

