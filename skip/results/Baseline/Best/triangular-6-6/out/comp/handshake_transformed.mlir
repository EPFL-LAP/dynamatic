module {
  handshake.func @triangular(%arg0: memref<10xi32>, %arg1: !handshake.channel<i32>, %arg2: memref<100xi32>, %arg3: !handshake.control<>, %arg4: !handshake.control<>, %arg5: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["x", "n", "a", "x_start", "a_start", "start"], resNames = ["x_end", "a_end", "end"]} {
    %0:3 = fork [3] %arg5 {handshake.bb = 0 : ui32, handshake.name = "fork0"} : <>
    %outputs:2, %memEnd = mem_controller[%arg2 : memref<100xi32>] %arg4 (%91, %addressResult, %1#1, %1#2, %1#3) %201#1 {connectedBlocks = [4 : i32], handshake.name = "mem_controller2"} :    (!handshake.channel<i32>, !handshake.channel<i7>, !handshake.channel<i7>, !handshake.channel<i7>, !handshake.channel<i32>) -> (!handshake.channel<i32>, !handshake.channel<i32>)
    %1:4 = lsq[MC] (%89#0, %addressResult_26, %addressResult_28, %dataResult_29, %outputs#1)  {groupSizes = [2 : i32], handshake.name = "lsq1"} : (!handshake.control<>, !handshake.channel<i7>, !handshake.channel<i7>, !handshake.channel<i32>, !handshake.channel<i32>) -> (!handshake.channel<i32>, !handshake.channel<i7>, !handshake.channel<i7>, !handshake.channel<i32>)
    %outputs_0, %memEnd_1 = mem_controller[%arg0 : memref<10xi32>] %arg3 (%addressResult_24) %201#0 {connectedBlocks = [4 : i32], handshake.name = "mem_controller3"} :    (!handshake.channel<i4>) -> !handshake.channel<i32>
    %2 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant3", value = false} : <>, <i1>
    %3 = br %2 {handshake.bb = 0 : ui32, handshake.name = "br4"} : <i1>
    %4 = extsi %3 {handshake.bb = 0 : ui32, handshake.name = "extsi6"} : <i1> to <i32>
    %5 = br %arg1 {handshake.bb = 0 : ui32, handshake.name = "br5"} : <i32>
    %6 = br %0#2 {handshake.bb = 0 : ui32, handshake.name = "br6"} : <>
    %7 = mux %13#0 [%4, %198] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i32>, <i32>] to <i32>
    %9:2 = fork [2] %7 {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <i32>
    %10 = mux %11 [%5, %199] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i1>, [<i32>, <i32>] to <i32>
    %11 = buffer %13#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer1"} : <i1>
    %12:2 = fork [2] %10 {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <i32>
    %result, %index = control_merge [%6, %200]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %13:2 = fork [2] %index {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <i1>
    %14 = cmpi slt, %16, %12#1 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i32>
    %16 = buffer %9#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer3"} : <i32>
    %17:3 = fork [3] %14 {handshake.bb = 1 : ui32, handshake.name = "fork4"} : <i1>
    %trueResult, %falseResult = cond_br %17#2, %12#0 {handshake.bb = 1 : ui32, handshake.name = "cond_br2"} : <i1>, <i32>
    sink %falseResult {handshake.name = "sink0"} : <i32>
    %trueResult_2, %falseResult_3 = cond_br %17#1, %21 {handshake.bb = 1 : ui32, handshake.name = "cond_br3"} : <i1>, <i32>
    %21 = buffer %9#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer7"} : <i32>
    sink %falseResult_3 {handshake.name = "sink1"} : <i32>
    %trueResult_4, %falseResult_5 = cond_br %22, %result {handshake.bb = 1 : ui32, handshake.name = "cond_br4"} : <i1>, <>
    %22 = buffer %17#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer8"} : <i1>
    %23 = merge %trueResult {handshake.bb = 2 : ui32, handshake.name = "merge0"} : <i32>
    %24:2 = fork [2] %23 {handshake.bb = 2 : ui32, handshake.name = "fork5"} : <i32>
    %25 = merge %trueResult_2 {handshake.bb = 2 : ui32, handshake.name = "merge1"} : <i32>
    %26:2 = fork [2] %25 {handshake.bb = 2 : ui32, handshake.name = "fork6"} : <i32>
    %result_6, %index_7 = control_merge [%trueResult_4]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>] to <>, <i1>
    sink %index_7 {handshake.name = "sink2"} : <i1>
    %27:2 = fork [2] %result_6 {handshake.bb = 2 : ui32, handshake.name = "fork7"} : <>
    %28 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %29 = constant %28 {handshake.bb = 2 : ui32, handshake.name = "constant5", value = -1 : i32} : <>, <i32>
    %30 = constant %27#0 {handshake.bb = 2 : ui32, handshake.name = "constant4", value = false} : <>, <i1>
    %31 = subi %24#1, %26#1 {handshake.bb = 2 : ui32, handshake.name = "subi1"} : <i32>
    %34:2 = fork [2] %31 {handshake.bb = 2 : ui32, handshake.name = "fork8"} : <i32>
    %35 = addi %34#1, %29 {handshake.bb = 2 : ui32, handshake.name = "addi1"} : <i32>
    %37 = br %30 {handshake.bb = 2 : ui32, handshake.name = "br7"} : <i1>
    %38 = extsi %37 {handshake.bb = 2 : ui32, handshake.name = "extsi7"} : <i1> to <i32>
    %39 = br %24#0 {handshake.bb = 2 : ui32, handshake.name = "br8"} : <i32>
    %41 = br %26#0 {handshake.bb = 2 : ui32, handshake.name = "br9"} : <i32>
    %43 = br %34#0 {handshake.bb = 2 : ui32, handshake.name = "br10"} : <i32>
    %45 = br %35 {handshake.bb = 2 : ui32, handshake.name = "br11"} : <i32>
    %46 = br %27#1 {handshake.bb = 2 : ui32, handshake.name = "br12"} : <>
    %47 = mux %59#0 [%38, %183] {handshake.bb = 3 : ui32, handshake.name = "mux2"} : <i1>, [<i32>, <i32>] to <i32>
    %49:2 = fork [2] %47 {handshake.bb = 3 : ui32, handshake.name = "fork9"} : <i32>
    %50 = mux %59#1 [%39, %184] {handshake.bb = 3 : ui32, handshake.name = "mux3"} : <i1>, [<i32>, <i32>] to <i32>
    %52 = mux %59#2 [%41, %186] {handshake.bb = 3 : ui32, handshake.name = "mux4"} : <i1>, [<i32>, <i32>] to <i32>
    %54 = mux %59#3 [%43, %188] {handshake.bb = 3 : ui32, handshake.name = "mux5"} : <i1>, [<i32>, <i32>] to <i32>
    %56 = mux %59#4 [%45, %190] {handshake.bb = 3 : ui32, handshake.name = "mux6"} : <i1>, [<i32>, <i32>] to <i32>
    %58:2 = fork [2] %56 {handshake.bb = 3 : ui32, handshake.name = "fork10"} : <i32>
    %result_8, %index_9 = control_merge [%46, %191]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>, <>] to <>, <i1>
    %59:5 = fork [5] %index_9 {handshake.bb = 3 : ui32, handshake.name = "fork11"} : <i1>
    %60 = cmpi slt, %49#1, %58#1 {handshake.bb = 3 : ui32, handshake.name = "cmpi1"} : <i32>
    %63:6 = fork [6] %60 {handshake.bb = 3 : ui32, handshake.name = "fork12"} : <i1>
    %trueResult_10, %falseResult_11 = cond_br %63#5, %50 {handshake.bb = 3 : ui32, handshake.name = "cond_br5"} : <i1>, <i32>
    %trueResult_12, %falseResult_13 = cond_br %63#4, %52 {handshake.bb = 3 : ui32, handshake.name = "cond_br6"} : <i1>, <i32>
    %trueResult_14, %falseResult_15 = cond_br %63#3, %54 {handshake.bb = 3 : ui32, handshake.name = "cond_br7"} : <i1>, <i32>
    sink %falseResult_15 {handshake.name = "sink3"} : <i32>
    %trueResult_16, %falseResult_17 = cond_br %63#2, %58#0 {handshake.bb = 3 : ui32, handshake.name = "cond_br8"} : <i1>, <i32>
    sink %falseResult_17 {handshake.name = "sink4"} : <i32>
    %trueResult_18, %falseResult_19 = cond_br %63#1, %49#0 {handshake.bb = 3 : ui32, handshake.name = "cond_br9"} : <i1>, <i32>
    sink %falseResult_19 {handshake.name = "sink5"} : <i32>
    %trueResult_20, %falseResult_21 = cond_br %63#0, %result_8 {handshake.bb = 3 : ui32, handshake.name = "cond_br10"} : <i1>, <>
    %72 = merge %trueResult_10 {handshake.bb = 4 : ui32, handshake.name = "merge2"} : <i32>
    %73:6 = fork [6] %72 {handshake.bb = 4 : ui32, handshake.name = "fork13"} : <i32>
    %74 = trunci %73#0 {handshake.bb = 4 : ui32, handshake.name = "trunci0"} : <i32> to <i7>
    %76 = trunci %73#1 {handshake.bb = 4 : ui32, handshake.name = "trunci1"} : <i32> to <i7>
    %78 = merge %trueResult_12 {handshake.bb = 4 : ui32, handshake.name = "merge3"} : <i32>
    %79:4 = fork [4] %78 {handshake.bb = 4 : ui32, handshake.name = "fork14"} : <i32>
    %80 = merge %trueResult_14 {handshake.bb = 4 : ui32, handshake.name = "merge4"} : <i32>
    %81:3 = fork [3] %80 {handshake.bb = 4 : ui32, handshake.name = "fork15"} : <i32>
    %82 = trunci %81#0 {handshake.bb = 4 : ui32, handshake.name = "trunci2"} : <i32> to <i7>
    %84 = trunci %81#1 {handshake.bb = 4 : ui32, handshake.name = "trunci3"} : <i32> to <i4>
    %86 = merge %trueResult_16 {handshake.bb = 4 : ui32, handshake.name = "merge5"} : <i32>
    %87 = merge %trueResult_18 {handshake.bb = 4 : ui32, handshake.name = "merge6"} : <i32>
    %88:4 = fork [4] %87 {handshake.bb = 4 : ui32, handshake.name = "fork16"} : <i32>
    %result_22, %index_23 = control_merge [%trueResult_20]  {handshake.bb = 4 : ui32, handshake.name = "control_merge3"} : [<>] to <>, <i1>
    sink %index_23 {handshake.name = "sink6"} : <i1>
    %89:3 = fork [3] %result_22 {handshake.bb = 4 : ui32, handshake.name = "fork17"} : <>
    %90 = constant %89#1 {handshake.bb = 4 : ui32, handshake.name = "constant12", value = 1 : i2} : <>, <i2>
    %91 = extsi %90 {handshake.bb = 4 : ui32, handshake.name = "extsi2"} : <i2> to <i32>
    %92 = source {handshake.bb = 4 : ui32, handshake.name = "source1"} : <>
    %93 = constant %92 {handshake.bb = 4 : ui32, handshake.name = "constant7", value = -2 : i32} : <>, <i32>
    %94:3 = fork [3] %93 {handshake.bb = 4 : ui32, handshake.name = "fork18"} : <i32>
    %95 = source {handshake.bb = 4 : ui32, handshake.name = "source2"} : <>
    %96 = constant %95 {handshake.bb = 4 : ui32, handshake.name = "constant8", value = -1 : i32} : <>, <i32>
    %97:5 = fork [5] %96 {handshake.bb = 4 : ui32, handshake.name = "fork19"} : <i32>
    %98 = trunci %97#0 {handshake.bb = 4 : ui32, handshake.name = "trunci4"} : <i32> to <i7>
    %100 = trunci %97#1 {handshake.bb = 4 : ui32, handshake.name = "trunci5"} : <i32> to <i4>
    %102 = source {handshake.bb = 4 : ui32, handshake.name = "source3"} : <>
    %103 = constant %102 {handshake.bb = 4 : ui32, handshake.name = "constant13", value = 1 : i2} : <>, <i2>
    %104 = extsi %103 {handshake.bb = 4 : ui32, handshake.name = "extsi3"} : <i2> to <i32>
    %105:7 = fork [7] %104 {handshake.bb = 4 : ui32, handshake.name = "fork20"} : <i32>
    %106 = source {handshake.bb = 4 : ui32, handshake.name = "source4"} : <>
    %107 = constant %106 {handshake.bb = 4 : ui32, handshake.name = "constant14", value = 3 : i3} : <>, <i3>
    %108 = extsi %107 {handshake.bb = 4 : ui32, handshake.name = "extsi4"} : <i3> to <i32>
    %109:3 = fork [3] %108 {handshake.bb = 4 : ui32, handshake.name = "fork21"} : <i32>
    %110 = addi %79#3, %88#3 {handshake.bb = 4 : ui32, handshake.name = "addi0"} : <i32>
    %113 = xori %110, %97#4 {handshake.bb = 4 : ui32, handshake.name = "xori0"} : <i32>
    %115 = addi %113, %105#0 {handshake.bb = 4 : ui32, handshake.name = "addi2"} : <i32>
    %117 = addi %115, %73#5 {handshake.bb = 4 : ui32, handshake.name = "addi3"} : <i32>
    %119 = addi %117, %94#2 {handshake.bb = 4 : ui32, handshake.name = "addi4"} : <i32>
    %121:2 = fork [2] %119 {handshake.bb = 4 : ui32, handshake.name = "fork22"} : <i32>
    %122 = addi %82, %98 {handshake.bb = 4 : ui32, handshake.name = "addi6"} : <i7>
    %123 = shli %121#1, %105#1 {handshake.bb = 4 : ui32, handshake.name = "shli0"} : <i32>
    %126 = trunci %123 {handshake.bb = 4 : ui32, handshake.name = "trunci6"} : <i32> to <i7>
    %127 = shli %121#0, %109#0 {handshake.bb = 4 : ui32, handshake.name = "shli1"} : <i32>
    %130 = trunci %127 {handshake.bb = 4 : ui32, handshake.name = "trunci7"} : <i32> to <i7>
    %131 = addi %126, %130 {handshake.bb = 4 : ui32, handshake.name = "addi5"} : <i7>
    %132 = addi %122, %131 {handshake.bb = 4 : ui32, handshake.name = "addi15"} : <i7>
    %addressResult, %dataResult = load[%132] %outputs#0 {handshake.bb = 4 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i7>, <i32>, <i7>, <i32>
    %133 = addi %84, %100 {handshake.bb = 4 : ui32, handshake.name = "addi8"} : <i4>
    %addressResult_24, %dataResult_25 = load[%133] %outputs_0 {handshake.bb = 4 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i4>, <i32>, <i4>, <i32>
    %134 = muli %dataResult, %dataResult_25 {handshake.bb = 4 : ui32, handshake.name = "muli0"} : <i32>
    %135 = addi %79#2, %88#2 {handshake.bb = 4 : ui32, handshake.name = "addi7"} : <i32>
    %138 = xori %135, %97#3 {handshake.bb = 4 : ui32, handshake.name = "xori1"} : <i32>
    %140 = addi %138, %105#2 {handshake.bb = 4 : ui32, handshake.name = "addi9"} : <i32>
    %142 = addi %140, %73#4 {handshake.bb = 4 : ui32, handshake.name = "addi10"} : <i32>
    %144 = addi %142, %94#1 {handshake.bb = 4 : ui32, handshake.name = "addi11"} : <i32>
    %146:2 = fork [2] %144 {handshake.bb = 4 : ui32, handshake.name = "fork23"} : <i32>
    %147 = shli %146#1, %105#3 {handshake.bb = 4 : ui32, handshake.name = "shli2"} : <i32>
    %150 = trunci %147 {handshake.bb = 4 : ui32, handshake.name = "trunci8"} : <i32> to <i7>
    %151 = shli %146#0, %109#1 {handshake.bb = 4 : ui32, handshake.name = "shli3"} : <i32>
    %154 = trunci %151 {handshake.bb = 4 : ui32, handshake.name = "trunci9"} : <i32> to <i7>
    %155 = addi %150, %154 {handshake.bb = 4 : ui32, handshake.name = "addi12"} : <i7>
    %156 = addi %74, %155 {handshake.bb = 4 : ui32, handshake.name = "addi16"} : <i7>
    %addressResult_26, %dataResult_27 = load[%156] %1#0 {handshake.bb = 4 : ui32, handshake.deps = #handshake<deps[["store0", 1], ["store0", 3]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 0>, handshake.name = "load2"} : <i7>, <i32>, <i7>, <i32>
    %157 = subi %dataResult_27, %134 {handshake.bb = 4 : ui32, handshake.name = "subi0"} : <i32>
    %158 = addi %79#1, %88#1 {handshake.bb = 4 : ui32, handshake.name = "addi20"} : <i32>
    %161 = xori %158, %97#2 {handshake.bb = 4 : ui32, handshake.name = "xori2"} : <i32>
    %163 = addi %161, %105#4 {handshake.bb = 4 : ui32, handshake.name = "addi21"} : <i32>
    %165 = addi %163, %73#3 {handshake.bb = 4 : ui32, handshake.name = "addi13"} : <i32>
    %167 = addi %165, %94#0 {handshake.bb = 4 : ui32, handshake.name = "addi14"} : <i32>
    %169:2 = fork [2] %167 {handshake.bb = 4 : ui32, handshake.name = "fork24"} : <i32>
    %170 = shli %169#1, %105#5 {handshake.bb = 4 : ui32, handshake.name = "shli4"} : <i32>
    %173 = trunci %170 {handshake.bb = 4 : ui32, handshake.name = "trunci10"} : <i32> to <i7>
    %174 = shli %169#0, %109#2 {handshake.bb = 4 : ui32, handshake.name = "shli5"} : <i32>
    %177 = trunci %174 {handshake.bb = 4 : ui32, handshake.name = "trunci11"} : <i32> to <i7>
    %178 = addi %173, %177 {handshake.bb = 4 : ui32, handshake.name = "addi22"} : <i7>
    %179 = addi %76, %178 {handshake.bb = 4 : ui32, handshake.name = "addi17"} : <i7>
    %addressResult_28, %dataResult_29 = store[%179] %157 {handshake.bb = 4 : ui32, handshake.deps = #handshake<deps[["load2", 1], ["store0", 1]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 0>, handshake.name = "store0"} : <i7>, <i32>, <i7>, <i32>
    %180 = addi %88#0, %105#6 {handshake.bb = 4 : ui32, handshake.name = "addi18"} : <i32>
    %183 = br %180 {handshake.bb = 4 : ui32, handshake.name = "br13"} : <i32>
    %184 = br %73#2 {handshake.bb = 4 : ui32, handshake.name = "br14"} : <i32>
    %186 = br %79#0 {handshake.bb = 4 : ui32, handshake.name = "br15"} : <i32>
    %188 = br %81#2 {handshake.bb = 4 : ui32, handshake.name = "br16"} : <i32>
    %190 = br %86 {handshake.bb = 4 : ui32, handshake.name = "br17"} : <i32>
    %191 = br %89#2 {handshake.bb = 4 : ui32, handshake.name = "br18"} : <>
    %192 = merge %falseResult_11 {handshake.bb = 5 : ui32, handshake.name = "merge7"} : <i32>
    %193 = merge %falseResult_13 {handshake.bb = 5 : ui32, handshake.name = "merge8"} : <i32>
    %result_30, %index_31 = control_merge [%falseResult_21]  {handshake.bb = 5 : ui32, handshake.name = "control_merge4"} : [<>] to <>, <i1>
    sink %index_31 {handshake.name = "sink7"} : <i1>
    %194 = source {handshake.bb = 5 : ui32, handshake.name = "source5"} : <>
    %195 = constant %194 {handshake.bb = 5 : ui32, handshake.name = "constant15", value = 1 : i2} : <>, <i2>
    %196 = extsi %195 {handshake.bb = 5 : ui32, handshake.name = "extsi5"} : <i2> to <i32>
    %197 = addi %193, %196 {handshake.bb = 5 : ui32, handshake.name = "addi19"} : <i32>
    %198 = br %197 {handshake.bb = 5 : ui32, handshake.name = "br19"} : <i32>
    %199 = br %192 {handshake.bb = 5 : ui32, handshake.name = "br20"} : <i32>
    %200 = br %result_30 {handshake.bb = 5 : ui32, handshake.name = "br21"} : <>
    %result_32, %index_33 = control_merge [%falseResult_5]  {handshake.bb = 6 : ui32, handshake.name = "control_merge5"} : [<>] to <>, <i1>
    sink %index_33 {handshake.name = "sink8"} : <i1>
    %201:2 = fork [2] %result_32 {handshake.bb = 6 : ui32, handshake.name = "fork25"} : <>
    end {handshake.bb = 6 : ui32, handshake.name = "end0"} %memEnd_1, %memEnd, %0#1 : <>, <>, <>
  }
}

