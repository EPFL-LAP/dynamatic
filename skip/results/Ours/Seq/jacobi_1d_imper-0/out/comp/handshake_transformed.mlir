module {
  handshake.func @jacobi_1d_imper(%arg0: memref<100xi32>, %arg1: memref<100xi32>, %arg2: !handshake.control<>, %arg3: !handshake.control<>, %arg4: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["A", "B", "A_start", "B_start", "start"], cfg.edges = "[0,1][2,2,3,cmpi0][4,1,5,cmpi2][1,2][3,3,4,cmpi1]", resNames = ["A_end", "B_end", "end"]} {
    %0:4 = fork [4] %arg4 {handshake.bb = 0 : ui32, handshake.name = "fork0"} : <>
    %outputs:2, %memEnd = mem_controller[%arg1 : memref<100xi32>] %arg3 (%50, %addressResult_16, %dataResult_17, %addressResult_38) %183#1 {connectedBlocks = [2 : i32, 3 : i32], handshake.name = "mem_controller0"} :    (!handshake.channel<i32>, !handshake.channel<i7>, !handshake.channel<i32>, !handshake.channel<i7>) -> (!handshake.control<>, !handshake.channel<i32>)
    %outputs_0:4, %memEnd_1 = mem_controller[%arg0 : memref<100xi32>] %arg2 (%addressResult, %addressResult_12, %addressResult_14, %138, %addressResult_40, %dataResult_41) %183#0 {connectedBlocks = [2 : i32, 3 : i32], handshake.name = "mem_controller1"} :    (!handshake.channel<i7>, !handshake.channel<i7>, !handshake.channel<i7>, !handshake.channel<i32>, !handshake.channel<i7>, !handshake.channel<i32>) -> (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>)
    %1 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant2", value = false} : <>, <i1>
    %2 = br %1 {handshake.bb = 0 : ui32, handshake.name = "br2"} : <i1>
    %3 = extsi %2 {handshake.bb = 0 : ui32, handshake.name = "extsi13"} : <i1> to <i3>
    %4 = br %0#3 {handshake.bb = 0 : ui32, handshake.name = "br3"} : <>
    %5 = mux %6 [%0#2, %trueResult_49] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux8"} : <i1>, [<>, <>] to <>
    %6 = init %180#2 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "init0"} : <i1>
    %8 = mux %index [%3, %trueResult_53] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i3>, <i3>] to <i3>
    %result, %index = control_merge [%4, %trueResult_55]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %9:2 = fork [2] %result {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <>
    %10 = constant %9#0 {handshake.bb = 1 : ui32, handshake.name = "constant4", value = 1 : i2} : <>, <i2>
    %11 = br %10 {handshake.bb = 1 : ui32, handshake.name = "br4"} : <i2>
    %12 = extsi %11 {handshake.bb = 1 : ui32, handshake.name = "extsi12"} : <i2> to <i8>
    %13 = br %8 {handshake.bb = 1 : ui32, handshake.name = "br5"} : <i3>
    %14 = br %9#1 {handshake.bb = 1 : ui32, handshake.name = "br6"} : <>
    %trueResult, %falseResult = cond_br %96#7, %86 {handshake.bb = 2 : ui32, handshake.name = "cond_br45"} : <i1>, <>
    sink %trueResult {handshake.name = "sink0"} : <>
    %trueResult_2, %falseResult_3 = cond_br %96#6, %65 {handshake.bb = 2 : ui32, handshake.name = "cond_br46"} : <i1>, <>
    sink %trueResult_2 {handshake.name = "sink1"} : <>
    %trueResult_4, %falseResult_5 = cond_br %96#5, %75 {handshake.bb = 2 : ui32, handshake.name = "cond_br47"} : <i1>, <>
    sink %trueResult_4 {handshake.name = "sink2"} : <>
    %trueResult_6, %falseResult_7 = cond_br %96#4, %61 {handshake.bb = 2 : ui32, handshake.name = "cond_br48"} : <i1>, <>
    sink %trueResult_6 {handshake.name = "sink3"} : <>
    %trueResult_8, %falseResult_9 = cond_br %96#3, %23#4 {handshake.bb = 2 : ui32, handshake.name = "cond_br49"} : <i1>, <>
    sink %falseResult_9 {handshake.name = "sink4"} : <>
    %20 = init %96#2 {ftd.imerge, handshake.bb = 2 : ui32, handshake.name = "init8"} : <i1>
    %22 = mux %20 [%5, %trueResult_8] {ftd.regen, handshake.bb = 2 : ui32, handshake.name = "mux13"} : <i1>, [<>, <>] to <>
    %23:5 = fork [5] %22 {handshake.bb = 2 : ui32, handshake.name = "fork2"} : <>
    %24:2 = unbundle %69#0  {handshake.bb = 2 : ui32, handshake.name = "unbundle0"} : <i32> to _ 
    %26:2 = unbundle %64#0  {handshake.bb = 2 : ui32, handshake.name = "unbundle1"} : <i32> to _ 
    %28:2 = unbundle %78#0  {handshake.bb = 2 : ui32, handshake.name = "unbundle2"} : <i32> to _ 
    %30 = mux %42#1 [%12, %trueResult_18] {handshake.bb = 2 : ui32, handshake.name = "mux1"} : <i1>, [<i8>, <i8>] to <i8>
    %32:3 = fork [3] %30 {handshake.bb = 2 : ui32, handshake.name = "fork3"} : <i8>
    %33 = extsi %32#0 {handshake.bb = 2 : ui32, handshake.name = "extsi14"} : <i8> to <i9>
    %35 = extsi %32#1 {handshake.bb = 2 : ui32, handshake.name = "extsi15"} : <i8> to <i9>
    %37 = extsi %32#2 {handshake.bb = 2 : ui32, handshake.name = "extsi16"} : <i8> to <i32>
    %39:3 = fork [3] %37 {handshake.bb = 2 : ui32, handshake.name = "fork4"} : <i32>
    %40 = mux %42#0 [%13, %trueResult_20] {handshake.bb = 2 : ui32, handshake.name = "mux2"} : <i1>, [<i3>, <i3>] to <i3>
    %result_10, %index_11 = control_merge [%14, %trueResult_22]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>, <>] to <>, <i1>
    %42:2 = fork [2] %index_11 {handshake.bb = 2 : ui32, handshake.name = "fork5"} : <i1>
    %43:2 = fork [2] %result_10 {handshake.bb = 2 : ui32, handshake.name = "fork6"} : <>
    %44 = constant %43#0 {handshake.bb = 2 : ui32, handshake.name = "constant5", value = 1 : i2} : <>, <i2>
    %45:4 = fork [4] %44 {handshake.bb = 2 : ui32, handshake.name = "fork7"} : <i2>
    %46 = extsi %45#0 {handshake.bb = 2 : ui32, handshake.name = "extsi17"} : <i2> to <i9>
    %48 = extsi %45#1 {handshake.bb = 2 : ui32, handshake.name = "extsi18"} : <i2> to <i9>
    %50 = extsi %45#3 {handshake.bb = 2 : ui32, handshake.name = "extsi3"} : <i2> to <i32>
    %51 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %52 = constant %51 {handshake.bb = 2 : ui32, handshake.name = "constant7", value = -1 : i32} : <>, <i32>
    %53 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %54 = constant %53 {handshake.bb = 2 : ui32, handshake.name = "constant15", value = 99 : i8} : <>, <i8>
    %55 = extsi %54 {handshake.bb = 2 : ui32, handshake.name = "extsi19"} : <i8> to <i9>
    %56 = source {handshake.bb = 2 : ui32, handshake.name = "source2"} : <>
    %57 = constant %56 {handshake.bb = 2 : ui32, handshake.name = "constant16", value = 1 : i2} : <>, <i2>
    %58 = extsi %57 {handshake.bb = 2 : ui32, handshake.name = "extsi5"} : <i2> to <i32>
    %59 = addi %39#0, %52 {handshake.bb = 2 : ui32, handshake.name = "addi2"} : <i32>
    %61 = buffer %26#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer0"} : <>
    %62 = gate %59, %23#3 {handshake.bb = 2 : ui32, handshake.name = "gate0"} : <i32>, !handshake.control<> to <i32>
    %63 = trunci %62 {handshake.bb = 2 : ui32, handshake.name = "trunci0"} : <i32> to <i7>
    %addressResult, %dataResult = load[%63] %outputs_0#0 {handshake.bb = 2 : ui32, handshake.deps = #handshake<deps[["store1", 1, false], ["store1", 2, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i7>, <i32>, <i7>, <i32>
    %64:2 = fork [2] %dataResult {handshake.bb = 2 : ui32, handshake.name = "fork8"} : <i32>
    %65 = buffer %24#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer1"} : <>
    %66 = gate %39#1, %23#2 {handshake.bb = 2 : ui32, handshake.name = "gate1"} : <i32>, !handshake.control<> to <i32>
    %68 = trunci %66 {handshake.bb = 2 : ui32, handshake.name = "trunci1"} : <i32> to <i7>
    %addressResult_12, %dataResult_13 = load[%68] %outputs_0#1 {handshake.bb = 2 : ui32, handshake.deps = #handshake<deps[["store1", 1, false], ["store1", 2, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i7>, <i32>, <i7>, <i32>
    %69:2 = fork [2] %dataResult_13 {handshake.bb = 2 : ui32, handshake.name = "fork9"} : <i32>
    %70 = addi %64#1, %69#1 {handshake.bb = 2 : ui32, handshake.name = "addi0"} : <i32>
    %73 = addi %35, %48 {handshake.bb = 2 : ui32, handshake.name = "addi3"} : <i9>
    %74 = extsi %73 {handshake.bb = 2 : ui32, handshake.name = "extsi20"} : <i9> to <i32>
    %75 = buffer %28#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer2"} : <>
    %76 = gate %74, %23#1 {handshake.bb = 2 : ui32, handshake.name = "gate2"} : <i32>, !handshake.control<> to <i32>
    %77 = trunci %76 {handshake.bb = 2 : ui32, handshake.name = "trunci2"} : <i32> to <i7>
    %addressResult_14, %dataResult_15 = load[%77] %outputs_0#2 {handshake.bb = 2 : ui32, handshake.deps = #handshake<deps[["store1", 1, false], ["store1", 2, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load2"} : <i7>, <i32>, <i7>, <i32>
    %78:2 = fork [2] %dataResult_15 {handshake.bb = 2 : ui32, handshake.name = "fork10"} : <i32>
    %79 = addi %70, %78#1 {handshake.bb = 2 : ui32, handshake.name = "addi1"} : <i32>
    %81:2 = fork [2] %79 {handshake.bb = 2 : ui32, handshake.name = "fork11"} : <i32>
    %82 = shli %81#1, %58 {handshake.bb = 2 : ui32, handshake.name = "shli0"} : <i32>
    %84 = addi %81#0, %82 {handshake.bb = 2 : ui32, handshake.name = "addi7"} : <i32>
    %86 = buffer %doneResult, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer3"} : <>
    %87 = gate %39#2, %23#0 {handshake.bb = 2 : ui32, handshake.name = "gate3"} : <i32>, !handshake.control<> to <i32>
    %89 = trunci %87 {handshake.bb = 2 : ui32, handshake.name = "trunci3"} : <i32> to <i7>
    %addressResult_16, %dataResult_17, %doneResult = store[%89] %84 %outputs#0 {handshake.bb = 2 : ui32, handshake.deps = #handshake<deps[["store0", 1, false], ["load3", 1, false], ["load3", 2, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store0"} : <i7>, <i32>, <>, <i7>, <i32>, <>
    %90 = addi %33, %46 {handshake.bb = 2 : ui32, handshake.name = "addi4"} : <i9>
    %91:2 = fork [2] %90 {handshake.bb = 2 : ui32, handshake.name = "fork12"} : <i9>
    %92 = trunci %91#0 {handshake.bb = 2 : ui32, handshake.name = "trunci4"} : <i9> to <i8>
    %94 = cmpi ult, %91#1, %55 {handshake.bb = 2 : ui32, handshake.name = "cmpi0"} : <i9>
    %96:10 = fork [10] %94 {handshake.bb = 2 : ui32, handshake.name = "fork13"} : <i1>
    %trueResult_18, %falseResult_19 = cond_br %96#0, %92 {handshake.bb = 2 : ui32, handshake.name = "cond_br3"} : <i1>, <i8>
    sink %falseResult_19 {handshake.name = "sink5"} : <i8>
    %trueResult_20, %falseResult_21 = cond_br %96#1, %40 {handshake.bb = 2 : ui32, handshake.name = "cond_br4"} : <i1>, <i3>
    %trueResult_22, %falseResult_23 = cond_br %96#8, %43#1 {handshake.bb = 2 : ui32, handshake.name = "cond_br5"} : <i1>, <>
    %trueResult_24, %falseResult_25 = cond_br %96#9, %45#2 {handshake.bb = 2 : ui32, handshake.name = "cond_br6"} : <i1>, <i2>
    sink %trueResult_24 {handshake.name = "sink6"} : <i2>
    %102 = extsi %falseResult_25 {handshake.bb = 2 : ui32, handshake.name = "extsi11"} : <i2> to <i8>
    %trueResult_26, %falseResult_27 = cond_br %161#7, %122#1 {handshake.bb = 3 : ui32, handshake.name = "cond_br50"} : <i1>, <>
    sink %falseResult_27 {handshake.name = "sink7"} : <>
    %trueResult_28, %falseResult_29 = cond_br %104, %150 {handshake.bb = 3 : ui32, handshake.name = "cond_br51"} : <i1>, <>
    %104 = buffer %161#6, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 3 : ui32, handshake.name = "buffer39"} : <i1>
    sink %trueResult_28 {handshake.name = "sink8"} : <>
    %trueResult_30, %falseResult_31 = cond_br %105, %116#1 {handshake.bb = 3 : ui32, handshake.name = "cond_br52"} : <i1>, <>
    %105 = buffer %161#5, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer40"} : <i1>
    sink %falseResult_31 {handshake.name = "sink9"} : <>
    %trueResult_32, %falseResult_33 = cond_br %106, %119#1 {handshake.bb = 3 : ui32, handshake.name = "cond_br53"} : <i1>, <>
    %106 = buffer %161#4, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer41"} : <i1>
    sink %falseResult_33 {handshake.name = "sink10"} : <>
    %trueResult_34, %falseResult_35 = cond_br %107, %113#1 {handshake.bb = 3 : ui32, handshake.name = "cond_br54"} : <i1>, <>
    %107 = buffer %161#3, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer42"} : <i1>
    sink %falseResult_35 {handshake.name = "sink11"} : <>
    %108 = init %161#2 {ftd.imerge, handshake.bb = 3 : ui32, handshake.name = "init16"} : <i1>
    %110:4 = fork [4] %108 {handshake.bb = 3 : ui32, handshake.name = "fork14"} : <i1>
    %111 = mux %112 [%falseResult_3, %trueResult_34] {ftd.regen, handshake.bb = 3 : ui32, handshake.name = "mux21"} : <i1>, [<>, <>] to <>
    %112 = buffer %110#3, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer44"} : <i1>
    %113:2 = fork [2] %111 {handshake.bb = 3 : ui32, handshake.name = "fork15"} : <>
    %114 = mux %115 [%falseResult_7, %trueResult_30] {ftd.regen, handshake.bb = 3 : ui32, handshake.name = "mux22"} : <i1>, [<>, <>] to <>
    %115 = buffer %110#2, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer45"} : <i1>
    %116:2 = fork [2] %114 {handshake.bb = 3 : ui32, handshake.name = "fork16"} : <>
    %117 = mux %118 [%falseResult_5, %trueResult_32] {ftd.regen, handshake.bb = 3 : ui32, handshake.name = "mux23"} : <i1>, [<>, <>] to <>
    %118 = buffer %110#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer46"} : <i1>
    %119:2 = fork [2] %117 {handshake.bb = 3 : ui32, handshake.name = "fork17"} : <>
    %120 = mux %121 [%falseResult, %trueResult_26] {ftd.regen, handshake.bb = 3 : ui32, handshake.name = "mux24"} : <i1>, [<>, <>] to <>
    %121 = buffer %110#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer47"} : <i1>
    %122:2 = fork [2] %120 {handshake.bb = 3 : ui32, handshake.name = "fork18"} : <>
    %123:2 = unbundle %149#1  {handshake.bb = 3 : ui32, handshake.name = "unbundle6"} : <i32> to _ 
    %125 = mux %135#1 [%102, %trueResult_43] {handshake.bb = 3 : ui32, handshake.name = "mux3"} : <i1>, [<i8>, <i8>] to <i8>
    %127:2 = fork [2] %125 {handshake.bb = 3 : ui32, handshake.name = "fork19"} : <i8>
    %128 = extsi %127#0 {handshake.bb = 3 : ui32, handshake.name = "extsi21"} : <i8> to <i9>
    %130 = extsi %127#1 {handshake.bb = 3 : ui32, handshake.name = "extsi22"} : <i8> to <i32>
    %132:2 = fork [2] %130 {handshake.bb = 3 : ui32, handshake.name = "fork20"} : <i32>
    %133 = mux %135#0 [%falseResult_21, %trueResult_45] {handshake.bb = 3 : ui32, handshake.name = "mux4"} : <i1>, [<i3>, <i3>] to <i3>
    %result_36, %index_37 = control_merge [%falseResult_23, %trueResult_47]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>, <>] to <>, <i1>
    %135:2 = fork [2] %index_37 {handshake.bb = 3 : ui32, handshake.name = "fork21"} : <i1>
    %136:2 = fork [2] %result_36 {handshake.bb = 3 : ui32, handshake.name = "fork22"} : <>
    %137 = constant %136#0 {handshake.bb = 3 : ui32, handshake.name = "constant17", value = 1 : i2} : <>, <i2>
    %138 = extsi %137 {handshake.bb = 3 : ui32, handshake.name = "extsi6"} : <i2> to <i32>
    %139 = source {handshake.bb = 3 : ui32, handshake.name = "source3"} : <>
    %140 = constant %139 {handshake.bb = 3 : ui32, handshake.name = "constant18", value = 99 : i8} : <>, <i8>
    %141 = extsi %140 {handshake.bb = 3 : ui32, handshake.name = "extsi23"} : <i8> to <i9>
    %142 = source {handshake.bb = 3 : ui32, handshake.name = "source4"} : <>
    %143 = constant %142 {handshake.bb = 3 : ui32, handshake.name = "constant19", value = 1 : i2} : <>, <i2>
    %144 = extsi %143 {handshake.bb = 3 : ui32, handshake.name = "extsi24"} : <i2> to <i9>
    %145 = buffer %123#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer4"} : <>
    sink %145 {handshake.name = "sink12"} : <>
    %146 = gate %147, %122#0 {handshake.bb = 3 : ui32, handshake.name = "gate4"} : <i32>, !handshake.control<> to <i32>
    %147 = buffer %132#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer53"} : <i32>
    %148 = trunci %146 {handshake.bb = 3 : ui32, handshake.name = "trunci5"} : <i32> to <i7>
    %addressResult_38, %dataResult_39 = load[%148] %outputs#1 {handshake.bb = 3 : ui32, handshake.deps = #handshake<deps[["store0", 1, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load3"} : <i7>, <i32>, <i7>, <i32>
    %149:2 = fork [2] %dataResult_39 {handshake.bb = 3 : ui32, handshake.name = "fork23"} : <i32>
    %150 = buffer %doneResult_42, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer5"} : <>
    %151 = gate %152, %119#0, %116#0, %113#0 {handshake.bb = 3 : ui32, handshake.name = "gate5"} : <i32>, !handshake.control<>, !handshake.control<>, !handshake.control<> to <i32>
    %152 = buffer %132#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer54"} : <i32>
    %153 = trunci %151 {handshake.bb = 3 : ui32, handshake.name = "trunci6"} : <i32> to <i7>
    %addressResult_40, %dataResult_41, %doneResult_42 = store[%153] %149#0 %outputs_0#3 {handshake.bb = 3 : ui32, handshake.deps = #handshake<deps[["load0", 1, false], ["load1", 1, false], ["load2", 1, false], ["store1", 1, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store1"} : <i7>, <i32>, <>, <i7>, <i32>, <>
    %155 = addi %128, %144 {handshake.bb = 3 : ui32, handshake.name = "addi5"} : <i9>
    %156:2 = fork [2] %155 {handshake.bb = 3 : ui32, handshake.name = "fork24"} : <i9>
    %157 = trunci %156#0 {handshake.bb = 3 : ui32, handshake.name = "trunci7"} : <i9> to <i8>
    %159 = cmpi ult, %156#1, %141 {handshake.bb = 3 : ui32, handshake.name = "cmpi1"} : <i9>
    %161:9 = fork [9] %159 {handshake.bb = 3 : ui32, handshake.name = "fork25"} : <i1>
    %trueResult_43, %falseResult_44 = cond_br %161#0, %157 {handshake.bb = 3 : ui32, handshake.name = "cond_br7"} : <i1>, <i8>
    sink %falseResult_44 {handshake.name = "sink13"} : <i8>
    %trueResult_45, %falseResult_46 = cond_br %161#1, %133 {handshake.bb = 3 : ui32, handshake.name = "cond_br8"} : <i1>, <i3>
    %trueResult_47, %falseResult_48 = cond_br %164, %136#1 {handshake.bb = 3 : ui32, handshake.name = "cond_br9"} : <i1>, <>
    %164 = buffer %161#8, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer60"} : <i1>
    %trueResult_49, %falseResult_50 = cond_br %180#1, %falseResult_29 {handshake.bb = 4 : ui32, handshake.name = "cond_br55"} : <i1>, <>
    sink %falseResult_50 {handshake.name = "sink14"} : <>
    %166 = merge %falseResult_46 {handshake.bb = 4 : ui32, handshake.name = "merge0"} : <i3>
    %167 = extsi %166 {handshake.bb = 4 : ui32, handshake.name = "extsi25"} : <i3> to <i4>
    %result_51, %index_52 = control_merge [%falseResult_48]  {handshake.bb = 4 : ui32, handshake.name = "control_merge3"} : [<>] to <>, <i1>
    sink %index_52 {handshake.name = "sink15"} : <i1>
    %168 = source {handshake.bb = 4 : ui32, handshake.name = "source5"} : <>
    %169 = constant %168 {handshake.bb = 4 : ui32, handshake.name = "constant20", value = 3 : i3} : <>, <i3>
    %170 = extsi %169 {handshake.bb = 4 : ui32, handshake.name = "extsi26"} : <i3> to <i4>
    %171 = source {handshake.bb = 4 : ui32, handshake.name = "source6"} : <>
    %172 = constant %171 {handshake.bb = 4 : ui32, handshake.name = "constant21", value = 1 : i2} : <>, <i2>
    %173 = extsi %172 {handshake.bb = 4 : ui32, handshake.name = "extsi27"} : <i2> to <i4>
    %174 = addi %167, %173 {handshake.bb = 4 : ui32, handshake.name = "addi6"} : <i4>
    %175:2 = fork [2] %174 {handshake.bb = 4 : ui32, handshake.name = "fork26"} : <i4>
    %176 = trunci %175#0 {handshake.bb = 4 : ui32, handshake.name = "trunci8"} : <i4> to <i3>
    %178 = cmpi ult, %175#1, %170 {handshake.bb = 4 : ui32, handshake.name = "cmpi2"} : <i4>
    %180:4 = fork [4] %178 {handshake.bb = 4 : ui32, handshake.name = "fork27"} : <i1>
    %trueResult_53, %falseResult_54 = cond_br %180#0, %176 {handshake.bb = 4 : ui32, handshake.name = "cond_br10"} : <i1>, <i3>
    sink %falseResult_54 {handshake.name = "sink16"} : <i3>
    %trueResult_55, %falseResult_56 = cond_br %180#3, %result_51 {handshake.bb = 4 : ui32, handshake.name = "cond_br11"} : <i1>, <>
    %result_57, %index_58 = control_merge [%falseResult_56]  {handshake.bb = 5 : ui32, handshake.name = "control_merge4"} : [<>] to <>, <i1>
    sink %index_58 {handshake.name = "sink17"} : <i1>
    %183:2 = fork [2] %result_57 {handshake.bb = 5 : ui32, handshake.name = "fork28"} : <>
    end {handshake.bb = 5 : ui32, handshake.name = "end0"} %memEnd_1, %memEnd, %0#1 : <>, <>, <>
  }
}

