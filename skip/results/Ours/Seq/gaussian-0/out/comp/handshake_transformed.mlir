module {
  handshake.func @gaussian(%arg0: memref<20xi32>, %arg1: memref<400xi32>, %arg2: !handshake.control<>, %arg3: !handshake.control<>, %arg4: !handshake.control<>, ...) -> (!handshake.channel<i32>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["c", "a", "c_start", "a_start", "start"], cfg.edges = "[0,1][2,3,5,cmpi2][4,2][1,2][3,3,4,cmpi0][5,1,6,cmpi1]", resNames = ["out0", "c_end", "a_end", "end"]} {
    %0:6 = fork [6] %arg4 {handshake.bb = 0 : ui32, handshake.name = "fork0"} : <>
    %outputs:3, %memEnd = mem_controller[%arg1 : memref<400xi32>] %arg3 (%107, %addressResult, %addressResult_24, %addressResult_26, %dataResult_27) %214#1 {connectedBlocks = [3 : i32], handshake.name = "mem_controller1"} :    (!handshake.channel<i32>, !handshake.channel<i9>, !handshake.channel<i9>, !handshake.channel<i9>, !handshake.channel<i32>) -> (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>)
    %outputs_0, %memEnd_1 = mem_controller[%arg0 : memref<20xi32>] %arg2 (%addressResult_22) %214#0 {connectedBlocks = [3 : i32], handshake.name = "mem_controller2"} :    (!handshake.channel<i5>) -> !handshake.channel<i32>
    %1 = constant %0#1 {handshake.bb = 0 : ui32, handshake.name = "constant1", value = false} : <>, <i1>
    %2 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant3", value = 1 : i2} : <>, <i2>
    %3 = br %2 {handshake.bb = 0 : ui32, handshake.name = "br2"} : <i2>
    %4 = extsi %3 {handshake.bb = 0 : ui32, handshake.name = "extsi16"} : <i2> to <i6>
    %5 = br %1 {handshake.bb = 0 : ui32, handshake.name = "br4"} : <i1>
    %6 = extsi %5 {handshake.bb = 0 : ui32, handshake.name = "extsi17"} : <i1> to <i32>
    %7 = br %0#5 {handshake.bb = 0 : ui32, handshake.name = "br5"} : <>
    %8 = mux %9 [%0#4, %trueResult_42] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux10"} : <i1>, [<>, <>] to <>
    %9 = buffer %14#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer1"} : <i1>
    %10 = mux %14#1 [%0#3, %trueResult_44] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux11"} : <i1>, [<>, <>] to <>
    %12 = init %209#3 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "init0"} : <i1>
    %14:2 = fork [2] %12 {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <i1>
    %15 = mux %22#0 [%4, %trueResult_48] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i6>, <i6>] to <i6>
    %17:2 = fork [2] %15 {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <i6>
    %18 = extsi %17#1 {handshake.bb = 1 : ui32, handshake.name = "extsi18"} : <i6> to <i7>
    %20 = mux %22#1 [%6, %trueResult_50] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i1>, [<i32>, <i32>] to <i32>
    %result, %index = control_merge [%7, %trueResult_52]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %22:2 = fork [2] %index {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <i1>
    %23 = source {handshake.bb = 1 : ui32, handshake.name = "source0"} : <>
    %24 = constant %23 {handshake.bb = 1 : ui32, handshake.name = "constant5", value = 1 : i2} : <>, <i2>
    %25 = extsi %24 {handshake.bb = 1 : ui32, handshake.name = "extsi19"} : <i2> to <i7>
    %26 = addi %18, %25 {handshake.bb = 1 : ui32, handshake.name = "addi2"} : <i7>
    %27 = br %26 {handshake.bb = 1 : ui32, handshake.name = "br6"} : <i7>
    %28 = br %20 {handshake.bb = 1 : ui32, handshake.name = "br7"} : <i32>
    %29 = br %17#0 {handshake.bb = 1 : ui32, handshake.name = "br8"} : <i6>
    %31 = br %result {handshake.bb = 1 : ui32, handshake.name = "br9"} : <>
    %32 = mux %33 [%8, %71#0] {ftd.phi, handshake.bb = 2 : ui32, handshake.name = "mux12"} : <i1>, [<>, <>] to <>
    %33 = buffer %38#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer8"} : <i1>
    %34 = mux %35 [%10, %71#1] {ftd.phi, handshake.bb = 2 : ui32, handshake.name = "mux13"} : <i1>, [<>, <>] to <>
    %35 = buffer %38#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer9"} : <i1>
    %36 = init %57#2 {ftd.imerge, handshake.bb = 2 : ui32, handshake.name = "init2"} : <i1>
    %38:2 = fork [2] %36 {handshake.bb = 2 : ui32, handshake.name = "fork4"} : <i1>
    %39 = mux %40 [%27, %188] {handshake.bb = 2 : ui32, handshake.name = "mux2"} : <i1>, [<i7>, <i7>] to <i7>
    %40 = buffer %48#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer11"} : <i1>
    %41:2 = fork [2] %39 {handshake.bb = 2 : ui32, handshake.name = "fork5"} : <i7>
    %42 = trunci %41#0 {handshake.bb = 2 : ui32, handshake.name = "trunci0"} : <i7> to <i6>
    %44 = mux %48#2 [%28, %189] {handshake.bb = 2 : ui32, handshake.name = "mux3"} : <i1>, [<i32>, <i32>] to <i32>
    %46 = mux %47 [%29, %190] {handshake.bb = 2 : ui32, handshake.name = "mux4"} : <i1>, [<i6>, <i6>] to <i6>
    %47 = buffer %48#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer14"} : <i1>
    %result_2, %index_3 = control_merge [%31, %191]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>, <>] to <>, <i1>
    %48:3 = fork [3] %index_3 {handshake.bb = 2 : ui32, handshake.name = "fork6"} : <i1>
    %49:2 = fork [2] %result_2 {handshake.bb = 2 : ui32, handshake.name = "fork7"} : <>
    %50 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %51 = constant %50 {handshake.bb = 2 : ui32, handshake.name = "constant6", value = 19 : i6} : <>, <i6>
    %52 = extsi %51 {handshake.bb = 2 : ui32, handshake.name = "extsi20"} : <i6> to <i7>
    %53 = constant %49#0 {handshake.bb = 2 : ui32, handshake.name = "constant19", value = 1 : i2} : <>, <i2>
    %54:2 = fork [2] %53 {handshake.bb = 2 : ui32, handshake.name = "fork8"} : <i2>
    %55 = cmpi ult, %41#1, %52 {handshake.bb = 2 : ui32, handshake.name = "cmpi2"} : <i7>
    %57:9 = fork [9] %55 {handshake.bb = 2 : ui32, handshake.name = "fork9"} : <i1>
    %trueResult, %falseResult = cond_br %57#8, %59 {handshake.bb = 2 : ui32, handshake.name = "cond_br3"} : <i1>, <i2>
    %59 = buffer %54#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer17"} : <i2>
    sink %falseResult {handshake.name = "sink0"} : <i2>
    %60 = extsi %trueResult {handshake.bb = 2 : ui32, handshake.name = "extsi15"} : <i2> to <i6>
    %trueResult_4, %falseResult_5 = cond_br %61, %62 {handshake.bb = 2 : ui32, handshake.name = "cond_br4"} : <i1>, <i2>
    %61 = buffer %57#7, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer18"} : <i1>
    %62 = buffer %54#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer19"} : <i2>
    sink %falseResult_5 {handshake.name = "sink1"} : <i2>
    %63 = extsi %trueResult_4 {handshake.bb = 2 : ui32, handshake.name = "extsi21"} : <i2> to <i32>
    %trueResult_6, %falseResult_7 = cond_br %64, %44 {handshake.bb = 2 : ui32, handshake.name = "cond_br5"} : <i1>, <i32>
    %64 = buffer %57#3, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer20"} : <i1>
    %trueResult_8, %falseResult_9 = cond_br %57#1, %46 {handshake.bb = 2 : ui32, handshake.name = "cond_br6"} : <i1>, <i6>
    %trueResult_10, %falseResult_11 = cond_br %57#0, %42 {handshake.bb = 2 : ui32, handshake.name = "cond_br7"} : <i1>, <i6>
    sink %falseResult_11 {handshake.name = "sink2"} : <i6>
    %trueResult_12, %falseResult_13 = cond_br %67, %49#1 {handshake.bb = 2 : ui32, handshake.name = "cond_br8"} : <i1>, <>
    %67 = buffer %57#4, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer23"} : <i1>
    %trueResult_14, %falseResult_15 = cond_br %68, %34 {handshake.bb = 3 : ui32, handshake.name = "cond_br25"} : <i1>, <>
    %68 = buffer %57#5, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer24"} : <i1>
    %trueResult_16, %falseResult_17 = cond_br %69, %32 {handshake.bb = 3 : ui32, handshake.name = "cond_br26"} : <i1>, <>
    %69 = buffer %57#6, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer25"} : <i1>
    %trueResult_18, %falseResult_19 = cond_br %70, %160 {handshake.bb = 3 : ui32, handshake.name = "cond_br27"} : <i1>, <>
    %70 = buffer %171#4, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer26"} : <i1>
    %71:2 = fork [2] %falseResult_19 {handshake.bb = 3 : ui32, handshake.name = "fork10"} : <>
    %72:2 = fork [2] %trueResult_18 {handshake.bb = 3 : ui32, handshake.name = "fork11"} : <>
    %73 = mux %74 [%trueResult_16, %72#1] {ftd.phi, handshake.bb = 3 : ui32, handshake.name = "mux14"} : <i1>, [<>, <>] to <>
    %74 = buffer %79#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer27"} : <i1>
    %75 = mux %76 [%trueResult_14, %72#0] {ftd.phi, handshake.bb = 3 : ui32, handshake.name = "mux15"} : <i1>, [<>, <>] to <>
    %76 = buffer %79#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer28"} : <i1>
    %77 = init %78 {ftd.imerge, handshake.bb = 3 : ui32, handshake.name = "init4"} : <i1>
    %78 = buffer %171#3, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer29"} : <i1>
    %79:2 = fork [2] %77 {handshake.bb = 3 : ui32, handshake.name = "fork12"} : <i1>
    %80 = mux %104#2 [%60, %trueResult_28] {handshake.bb = 3 : ui32, handshake.name = "mux5"} : <i1>, [<i6>, <i6>] to <i6>
    %82 = extsi %80 {handshake.bb = 3 : ui32, handshake.name = "extsi22"} : <i6> to <i7>
    %83 = mux %84 [%63, %trueResult_30] {handshake.bb = 3 : ui32, handshake.name = "mux6"} : <i1>, [<i32>, <i32>] to <i32>
    %84 = buffer %104#3, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer31"} : <i1>
    %85:5 = fork [5] %83 {handshake.bb = 3 : ui32, handshake.name = "fork13"} : <i32>
    %86 = trunci %85#0 {handshake.bb = 3 : ui32, handshake.name = "trunci1"} : <i32> to <i9>
    %88 = mux %104#4 [%trueResult_6, %trueResult_32] {handshake.bb = 3 : ui32, handshake.name = "mux7"} : <i1>, [<i32>, <i32>] to <i32>
    %90 = mux %104#0 [%trueResult_8, %trueResult_34] {handshake.bb = 3 : ui32, handshake.name = "mux8"} : <i1>, [<i6>, <i6>] to <i6>
    %92:3 = fork [3] %90 {handshake.bb = 3 : ui32, handshake.name = "fork14"} : <i6>
    %93 = extsi %92#2 {handshake.bb = 3 : ui32, handshake.name = "extsi23"} : <i6> to <i32>
    %95:2 = fork [2] %93 {handshake.bb = 3 : ui32, handshake.name = "fork15"} : <i32>
    %96 = trunci %92#0 {handshake.bb = 3 : ui32, handshake.name = "trunci2"} : <i6> to <i5>
    %98 = mux %104#1 [%trueResult_10, %trueResult_36] {handshake.bb = 3 : ui32, handshake.name = "mux9"} : <i1>, [<i6>, <i6>] to <i6>
    %100:2 = fork [2] %98 {handshake.bb = 3 : ui32, handshake.name = "fork16"} : <i6>
    %101 = extsi %100#1 {handshake.bb = 3 : ui32, handshake.name = "extsi24"} : <i6> to <i32>
    %103:4 = fork [4] %101 {handshake.bb = 3 : ui32, handshake.name = "fork17"} : <i32>
    %result_20, %index_21 = control_merge [%trueResult_12, %trueResult_38]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>, <>] to <>, <i1>
    %104:5 = fork [5] %index_21 {handshake.bb = 3 : ui32, handshake.name = "fork18"} : <i1>
    %105:2 = fork [2] %result_20 {handshake.bb = 3 : ui32, handshake.name = "fork19"} : <>
    %106 = constant %105#0 {handshake.bb = 3 : ui32, handshake.name = "constant20", value = 1 : i2} : <>, <i2>
    %107 = extsi %106 {handshake.bb = 3 : ui32, handshake.name = "extsi6"} : <i2> to <i32>
    %108 = source {handshake.bb = 3 : ui32, handshake.name = "source2"} : <>
    %109 = constant %108 {handshake.bb = 3 : ui32, handshake.name = "constant21", value = 20 : i6} : <>, <i6>
    %110 = extsi %109 {handshake.bb = 3 : ui32, handshake.name = "extsi25"} : <i6> to <i7>
    %111 = source {handshake.bb = 3 : ui32, handshake.name = "source3"} : <>
    %112 = constant %111 {handshake.bb = 3 : ui32, handshake.name = "constant22", value = 1 : i2} : <>, <i2>
    %113:2 = fork [2] %112 {handshake.bb = 3 : ui32, handshake.name = "fork20"} : <i2>
    %114 = extsi %113#0 {handshake.bb = 3 : ui32, handshake.name = "extsi26"} : <i2> to <i7>
    %116 = extsi %113#1 {handshake.bb = 3 : ui32, handshake.name = "extsi8"} : <i2> to <i32>
    %118 = source {handshake.bb = 3 : ui32, handshake.name = "source5"} : <>
    %119 = constant %118 {handshake.bb = 3 : ui32, handshake.name = "constant23", value = 4 : i4} : <>, <i4>
    %120 = extsi %119 {handshake.bb = 3 : ui32, handshake.name = "extsi10"} : <i4> to <i32>
    %121:3 = fork [3] %120 {handshake.bb = 3 : ui32, handshake.name = "fork21"} : <i32>
    %122 = source {handshake.bb = 3 : ui32, handshake.name = "source6"} : <>
    %123 = constant %122 {handshake.bb = 3 : ui32, handshake.name = "constant24", value = 2 : i3} : <>, <i3>
    %124 = extsi %123 {handshake.bb = 3 : ui32, handshake.name = "extsi11"} : <i3> to <i32>
    %125:3 = fork [3] %124 {handshake.bb = 3 : ui32, handshake.name = "fork22"} : <i32>
    %126 = shli %103#0, %125#0 {handshake.bb = 3 : ui32, handshake.name = "shli0"} : <i32>
    %129 = shli %131, %121#0 {handshake.bb = 3 : ui32, handshake.name = "shli1"} : <i32>
    %131 = buffer %103#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer44"} : <i32>
    %132 = addi %126, %129 {handshake.bb = 3 : ui32, handshake.name = "addi9"} : <i32>
    %133 = addi %134, %132 {handshake.bb = 3 : ui32, handshake.name = "addi3"} : <i32>
    %134 = buffer %85#4, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer45"} : <i32>
    %135 = gate %133, %75 {handshake.bb = 3 : ui32, handshake.name = "gate0"} : <i32>, !handshake.control<> to <i32>
    %136 = trunci %135 {handshake.bb = 3 : ui32, handshake.name = "trunci3"} : <i32> to <i9>
    %addressResult, %dataResult = load[%136] %outputs#0 {handshake.bb = 3 : ui32, handshake.deps = #handshake<deps[["store0", 0, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i9>, <i32>, <i9>, <i32>
    %addressResult_22, %dataResult_23 = load[%96] %outputs_0 {handshake.bb = 3 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i5>, <i32>, <i5>, <i32>
    %137 = shli %95#0, %125#1 {handshake.bb = 3 : ui32, handshake.name = "shli2"} : <i32>
    %140 = shli %95#1, %121#1 {handshake.bb = 3 : ui32, handshake.name = "shli3"} : <i32>
    %143 = addi %137, %140 {handshake.bb = 3 : ui32, handshake.name = "addi10"} : <i32>
    %144 = addi %85#3, %143 {handshake.bb = 3 : ui32, handshake.name = "addi4"} : <i32>
    %146 = gate %144, %73 {handshake.bb = 3 : ui32, handshake.name = "gate1"} : <i32>, !handshake.control<> to <i32>
    %147 = trunci %146 {handshake.bb = 3 : ui32, handshake.name = "trunci4"} : <i32> to <i9>
    %addressResult_24, %dataResult_25 = load[%147] %outputs#1 {handshake.bb = 3 : ui32, handshake.deps = #handshake<deps[["store0", 0, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load2"} : <i9>, <i32>, <i9>, <i32>
    %148 = muli %dataResult_23, %dataResult_25 {handshake.bb = 3 : ui32, handshake.name = "muli0"} : <i32>
    %149 = subi %dataResult, %148 {handshake.bb = 3 : ui32, handshake.name = "subi0"} : <i32>
    %150 = shli %103#2, %125#2 {handshake.bb = 3 : ui32, handshake.name = "shli4"} : <i32>
    %153 = trunci %150 {handshake.bb = 3 : ui32, handshake.name = "trunci5"} : <i32> to <i9>
    %154 = shli %103#3, %121#2 {handshake.bb = 3 : ui32, handshake.name = "shli5"} : <i32>
    %157 = trunci %154 {handshake.bb = 3 : ui32, handshake.name = "trunci6"} : <i32> to <i9>
    %158 = addi %153, %157 {handshake.bb = 3 : ui32, handshake.name = "addi11"} : <i9>
    %159 = addi %86, %158 {handshake.bb = 3 : ui32, handshake.name = "addi5"} : <i9>
    %160 = buffer %doneResult, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer0"} : <>
    %addressResult_26, %dataResult_27, %doneResult = store[%159] %149 %outputs#2 {handshake.bb = 3 : ui32, handshake.deps = #handshake<deps[["load0", 0, false], ["load2", 0, false], ["store0", 0, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store0"} : <i9>, <i32>, <>, <i9>, <i32>, <>
    %161 = addi %88, %85#2 {handshake.bb = 3 : ui32, handshake.name = "addi0"} : <i32>
    %163 = addi %85#1, %116 {handshake.bb = 3 : ui32, handshake.name = "addi1"} : <i32>
    %165 = addi %82, %114 {handshake.bb = 3 : ui32, handshake.name = "addi6"} : <i7>
    %166:2 = fork [2] %165 {handshake.bb = 3 : ui32, handshake.name = "fork23"} : <i7>
    %167 = trunci %168 {handshake.bb = 3 : ui32, handshake.name = "trunci7"} : <i7> to <i6>
    %168 = buffer %166#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer57"} : <i7>
    %169 = cmpi ult, %170, %110 {handshake.bb = 3 : ui32, handshake.name = "cmpi0"} : <i7>
    %170 = buffer %166#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer58"} : <i7>
    %171:8 = fork [8] %169 {handshake.bb = 3 : ui32, handshake.name = "fork24"} : <i1>
    %trueResult_28, %falseResult_29 = cond_br %171#0, %167 {handshake.bb = 3 : ui32, handshake.name = "cond_br9"} : <i1>, <i6>
    sink %falseResult_29 {handshake.name = "sink3"} : <i6>
    %trueResult_30, %falseResult_31 = cond_br %173, %163 {handshake.bb = 3 : ui32, handshake.name = "cond_br10"} : <i1>, <i32>
    %173 = buffer %171#5, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer60"} : <i1>
    sink %falseResult_31 {handshake.name = "sink4"} : <i32>
    %trueResult_32, %falseResult_33 = cond_br %171#6, %161 {handshake.bb = 3 : ui32, handshake.name = "cond_br11"} : <i1>, <i32>
    %trueResult_34, %falseResult_35 = cond_br %171#1, %92#1 {handshake.bb = 3 : ui32, handshake.name = "cond_br12"} : <i1>, <i6>
    %trueResult_36, %falseResult_37 = cond_br %171#2, %178 {handshake.bb = 3 : ui32, handshake.name = "cond_br13"} : <i1>, <i6>
    %178 = buffer %100#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer65"} : <i6>
    %trueResult_38, %falseResult_39 = cond_br %171#7, %105#1 {handshake.bb = 3 : ui32, handshake.name = "cond_br14"} : <i1>, <>
    %180 = merge %falseResult_35 {handshake.bb = 4 : ui32, handshake.name = "merge0"} : <i6>
    %181 = merge %falseResult_37 {handshake.bb = 4 : ui32, handshake.name = "merge1"} : <i6>
    %182 = extsi %181 {handshake.bb = 4 : ui32, handshake.name = "extsi27"} : <i6> to <i7>
    %183 = merge %falseResult_33 {handshake.bb = 4 : ui32, handshake.name = "merge2"} : <i32>
    %result_40, %index_41 = control_merge [%falseResult_39]  {handshake.bb = 4 : ui32, handshake.name = "control_merge3"} : [<>] to <>, <i1>
    sink %index_41 {handshake.name = "sink5"} : <i1>
    %184 = source {handshake.bb = 4 : ui32, handshake.name = "source7"} : <>
    %185 = constant %184 {handshake.bb = 4 : ui32, handshake.name = "constant25", value = 1 : i2} : <>, <i2>
    %186 = extsi %185 {handshake.bb = 4 : ui32, handshake.name = "extsi28"} : <i2> to <i7>
    %187 = addi %182, %186 {handshake.bb = 4 : ui32, handshake.name = "addi8"} : <i7>
    %188 = br %187 {handshake.bb = 4 : ui32, handshake.name = "br10"} : <i7>
    %189 = br %183 {handshake.bb = 4 : ui32, handshake.name = "br11"} : <i32>
    %190 = br %180 {handshake.bb = 4 : ui32, handshake.name = "br12"} : <i6>
    %191 = br %result_40 {handshake.bb = 4 : ui32, handshake.name = "br13"} : <>
    %trueResult_42, %falseResult_43 = cond_br %192, %falseResult_17 {handshake.bb = 5 : ui32, handshake.name = "cond_br28"} : <i1>, <>
    %192 = buffer %209#2, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 5 : ui32, handshake.name = "buffer67"} : <i1>
    sink %falseResult_43 {handshake.name = "sink6"} : <>
    %trueResult_44, %falseResult_45 = cond_br %193, %falseResult_15 {handshake.bb = 5 : ui32, handshake.name = "cond_br29"} : <i1>, <>
    %193 = buffer %209#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 5 : ui32, handshake.name = "buffer68"} : <i1>
    sink %falseResult_45 {handshake.name = "sink7"} : <>
    %194 = merge %falseResult_9 {handshake.bb = 5 : ui32, handshake.name = "merge3"} : <i6>
    %195 = extsi %194 {handshake.bb = 5 : ui32, handshake.name = "extsi29"} : <i6> to <i7>
    %196 = merge %falseResult_7 {handshake.bb = 5 : ui32, handshake.name = "merge4"} : <i32>
    %result_46, %index_47 = control_merge [%falseResult_13]  {handshake.bb = 5 : ui32, handshake.name = "control_merge4"} : [<>] to <>, <i1>
    sink %index_47 {handshake.name = "sink8"} : <i1>
    %197 = source {handshake.bb = 5 : ui32, handshake.name = "source8"} : <>
    %198 = constant %197 {handshake.bb = 5 : ui32, handshake.name = "constant26", value = 19 : i6} : <>, <i6>
    %199 = extsi %198 {handshake.bb = 5 : ui32, handshake.name = "extsi30"} : <i6> to <i7>
    %200 = source {handshake.bb = 5 : ui32, handshake.name = "source9"} : <>
    %201 = constant %200 {handshake.bb = 5 : ui32, handshake.name = "constant27", value = 1 : i2} : <>, <i2>
    %202 = extsi %201 {handshake.bb = 5 : ui32, handshake.name = "extsi31"} : <i2> to <i7>
    %203 = addi %195, %202 {handshake.bb = 5 : ui32, handshake.name = "addi7"} : <i7>
    %204:2 = fork [2] %203 {handshake.bb = 5 : ui32, handshake.name = "fork25"} : <i7>
    %205 = trunci %206 {handshake.bb = 5 : ui32, handshake.name = "trunci8"} : <i7> to <i6>
    %206 = buffer %204#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 5 : ui32, handshake.name = "buffer69"} : <i7>
    %207 = cmpi ult, %204#1, %199 {handshake.bb = 5 : ui32, handshake.name = "cmpi1"} : <i7>
    %209:6 = fork [6] %207 {handshake.bb = 5 : ui32, handshake.name = "fork26"} : <i1>
    %trueResult_48, %falseResult_49 = cond_br %209#0, %205 {handshake.bb = 5 : ui32, handshake.name = "cond_br15"} : <i1>, <i6>
    sink %falseResult_49 {handshake.name = "sink9"} : <i6>
    %trueResult_50, %falseResult_51 = cond_br %211, %196 {handshake.bb = 5 : ui32, handshake.name = "cond_br16"} : <i1>, <i32>
    %211 = buffer %209#4, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 5 : ui32, handshake.name = "buffer72"} : <i1>
    %trueResult_52, %falseResult_53 = cond_br %212, %result_46 {handshake.bb = 5 : ui32, handshake.name = "cond_br17"} : <i1>, <>
    %212 = buffer %209#5, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 5 : ui32, handshake.name = "buffer73"} : <i1>
    %213 = merge %falseResult_51 {handshake.bb = 6 : ui32, handshake.name = "merge5"} : <i32>
    %result_54, %index_55 = control_merge [%falseResult_53]  {handshake.bb = 6 : ui32, handshake.name = "control_merge5"} : [<>] to <>, <i1>
    sink %index_55 {handshake.name = "sink10"} : <i1>
    %214:2 = fork [2] %result_54 {handshake.bb = 6 : ui32, handshake.name = "fork27"} : <>
    end {handshake.bb = 6 : ui32, handshake.name = "end0"} %213, %memEnd_1, %memEnd, %0#2 : <i32>, <>, <>, <>
  }
}

