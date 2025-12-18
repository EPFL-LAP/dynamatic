module {
  handshake.func @triangular(%arg0: memref<10xi32>, %arg1: !handshake.channel<i32>, %arg2: memref<100xi32>, %arg3: !handshake.control<>, %arg4: !handshake.control<>, %arg5: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["x", "n", "a", "x_start", "a_start", "start"], cfg.edges = "[0,1][2,3][4,3][1,2,6,cmpi0][3,4,5,cmpi1][5,1]", resNames = ["x_end", "a_end", "end"]} {
    %0:4 = fork [4] %arg5 {handshake.bb = 0 : ui32, handshake.name = "fork0"} : <>
    %outputs:3, %memEnd = mem_controller[%arg2 : memref<100xi32>] %arg4 (%96, %addressResult, %addressResult_30, %addressResult_32, %dataResult_33) %208#1 {connectedBlocks = [4 : i32], handshake.name = "mem_controller2"} :    (!handshake.channel<i32>, !handshake.channel<i7>, !handshake.channel<i7>, !handshake.channel<i7>, !handshake.channel<i32>) -> (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>)
    %outputs_0, %memEnd_1 = mem_controller[%arg0 : memref<10xi32>] %arg3 (%addressResult_28) %208#0 {connectedBlocks = [4 : i32], handshake.name = "mem_controller3"} :    (!handshake.channel<i4>) -> !handshake.channel<i32>
    %1 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant0", value = false} : <>, <i1>
    %2 = br %1 {handshake.bb = 0 : ui32, handshake.name = "br4"} : <i1>
    %3 = extsi %2 {handshake.bb = 0 : ui32, handshake.name = "extsi6"} : <i1> to <i32>
    %4 = br %arg1 {handshake.bb = 0 : ui32, handshake.name = "br5"} : <i32>
    %5 = br %0#3 {handshake.bb = 0 : ui32, handshake.name = "br6"} : <>
    %6 = mux %7 [%0#2, %falseResult_25] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux7"} : <i1>, [<>, <>] to <>
    %7 = init %19#4 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "init0"} : <i1>
    %9 = mux %15#0 [%3, %205] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i32>, <i32>] to <i32>
    %11:2 = fork [2] %9 {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <i32>
    %12 = mux %13 [%4, %206] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i1>, [<i32>, <i32>] to <i32>
    %13 = buffer %15#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer3"} : <i1>
    %14:2 = fork [2] %12 {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <i32>
    %result, %index = control_merge [%5, %207]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %15:2 = fork [2] %index {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <i1>
    %16 = cmpi slt, %18, %14#1 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i32>
    %18 = buffer %11#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer5"} : <i32>
    %19:5 = fork [5] %16 {handshake.bb = 1 : ui32, handshake.name = "fork4"} : <i1>
    %trueResult, %falseResult = cond_br %19#3, %14#0 {handshake.bb = 1 : ui32, handshake.name = "cond_br2"} : <i1>, <i32>
    sink %falseResult {handshake.name = "sink0"} : <i32>
    %trueResult_2, %falseResult_3 = cond_br %19#2, %11#0 {handshake.bb = 1 : ui32, handshake.name = "cond_br3"} : <i1>, <i32>
    sink %falseResult_3 {handshake.name = "sink1"} : <i32>
    %trueResult_4, %falseResult_5 = cond_br %24, %result {handshake.bb = 1 : ui32, handshake.name = "cond_br4"} : <i1>, <>
    %24 = buffer %19#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer10"} : <i1>
    %25 = merge %trueResult {handshake.bb = 2 : ui32, handshake.name = "merge0"} : <i32>
    %26:2 = fork [2] %25 {handshake.bb = 2 : ui32, handshake.name = "fork5"} : <i32>
    %27 = merge %trueResult_2 {handshake.bb = 2 : ui32, handshake.name = "merge1"} : <i32>
    %28:2 = fork [2] %27 {handshake.bb = 2 : ui32, handshake.name = "fork6"} : <i32>
    %result_6, %index_7 = control_merge [%trueResult_4]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>] to <>, <i1>
    sink %index_7 {handshake.name = "sink2"} : <i1>
    %29:2 = fork [2] %result_6 {handshake.bb = 2 : ui32, handshake.name = "fork7"} : <>
    %30 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %31 = constant %30 {handshake.bb = 2 : ui32, handshake.name = "constant5", value = -1 : i32} : <>, <i32>
    %32 = constant %29#0 {handshake.bb = 2 : ui32, handshake.name = "constant3", value = false} : <>, <i1>
    %33 = subi %26#1, %28#1 {handshake.bb = 2 : ui32, handshake.name = "subi1"} : <i32>
    %36:2 = fork [2] %33 {handshake.bb = 2 : ui32, handshake.name = "fork8"} : <i32>
    %37 = addi %36#1, %31 {handshake.bb = 2 : ui32, handshake.name = "addi1"} : <i32>
    %39 = br %32 {handshake.bb = 2 : ui32, handshake.name = "br7"} : <i1>
    %40 = extsi %39 {handshake.bb = 2 : ui32, handshake.name = "extsi7"} : <i1> to <i32>
    %41 = br %26#0 {handshake.bb = 2 : ui32, handshake.name = "br8"} : <i32>
    %43 = br %28#0 {handshake.bb = 2 : ui32, handshake.name = "br9"} : <i32>
    %45 = br %46 {handshake.bb = 2 : ui32, handshake.name = "br10"} : <i32>
    %46 = buffer %36#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer16"} : <i32>
    %47 = br %37 {handshake.bb = 2 : ui32, handshake.name = "br11"} : <i32>
    %48 = br %29#1 {handshake.bb = 2 : ui32, handshake.name = "br12"} : <>
    %trueResult_8, %falseResult_9 = cond_br %19#0, %6 {handshake.bb = 3 : ui32, handshake.name = "cond_br12"} : <i1>, <>
    sink %falseResult_9 {handshake.name = "sink3"} : <>
    %50 = mux %51 [%trueResult_8, %186] {ftd.phi, handshake.bb = 3 : ui32, handshake.name = "mux8"} : <i1>, [<>, <>] to <>
    %51 = init %52 {ftd.imerge, handshake.bb = 3 : ui32, handshake.name = "init1"} : <i1>
    %52 = buffer %69#7, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 3 : ui32, handshake.name = "buffer18"} : <i1>
    %53 = mux %54 [%40, %190] {handshake.bb = 3 : ui32, handshake.name = "mux2"} : <i1>, [<i32>, <i32>] to <i32>
    %54 = buffer %65#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer19"} : <i1>
    %55:2 = fork [2] %53 {handshake.bb = 3 : ui32, handshake.name = "fork9"} : <i32>
    %56 = mux %65#1 [%41, %191] {handshake.bb = 3 : ui32, handshake.name = "mux3"} : <i1>, [<i32>, <i32>] to <i32>
    %58 = mux %65#2 [%43, %193] {handshake.bb = 3 : ui32, handshake.name = "mux4"} : <i1>, [<i32>, <i32>] to <i32>
    %60 = mux %65#3 [%45, %195] {handshake.bb = 3 : ui32, handshake.name = "mux5"} : <i1>, [<i32>, <i32>] to <i32>
    %62 = mux %65#4 [%47, %197] {handshake.bb = 3 : ui32, handshake.name = "mux6"} : <i1>, [<i32>, <i32>] to <i32>
    %64:2 = fork [2] %62 {handshake.bb = 3 : ui32, handshake.name = "fork10"} : <i32>
    %result_10, %index_11 = control_merge [%48, %198]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>, <>] to <>, <i1>
    %65:5 = fork [5] %index_11 {handshake.bb = 3 : ui32, handshake.name = "fork11"} : <i1>
    %66 = cmpi slt, %55#1, %67 {handshake.bb = 3 : ui32, handshake.name = "cmpi1"} : <i32>
    %67 = buffer %64#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer24"} : <i32>
    %69:8 = fork [8] %66 {handshake.bb = 3 : ui32, handshake.name = "fork12"} : <i1>
    %trueResult_12, %falseResult_13 = cond_br %69#6, %56 {handshake.bb = 3 : ui32, handshake.name = "cond_br5"} : <i1>, <i32>
    %trueResult_14, %falseResult_15 = cond_br %69#5, %58 {handshake.bb = 3 : ui32, handshake.name = "cond_br6"} : <i1>, <i32>
    %trueResult_16, %falseResult_17 = cond_br %69#4, %60 {handshake.bb = 3 : ui32, handshake.name = "cond_br7"} : <i1>, <i32>
    sink %falseResult_17 {handshake.name = "sink4"} : <i32>
    %trueResult_18, %falseResult_19 = cond_br %69#3, %64#0 {handshake.bb = 3 : ui32, handshake.name = "cond_br8"} : <i1>, <i32>
    sink %falseResult_19 {handshake.name = "sink5"} : <i32>
    %trueResult_20, %falseResult_21 = cond_br %69#2, %55#0 {handshake.bb = 3 : ui32, handshake.name = "cond_br9"} : <i1>, <i32>
    sink %falseResult_21 {handshake.name = "sink6"} : <i32>
    %trueResult_22, %falseResult_23 = cond_br %69#1, %result_10 {handshake.bb = 3 : ui32, handshake.name = "cond_br10"} : <i1>, <>
    %trueResult_24, %falseResult_25 = cond_br %78, %50 {handshake.bb = 4 : ui32, handshake.name = "cond_br13"} : <i1>, <>
    %78 = buffer %69#0, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 4 : ui32, handshake.name = "buffer34"} : <i1>
    %79 = merge %trueResult_12 {handshake.bb = 4 : ui32, handshake.name = "merge2"} : <i32>
    %80:6 = fork [6] %79 {handshake.bb = 4 : ui32, handshake.name = "fork13"} : <i32>
    %81 = trunci %82 {handshake.bb = 4 : ui32, handshake.name = "trunci0"} : <i32> to <i7>
    %82 = buffer %80#0, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 4 : ui32, handshake.name = "buffer35"} : <i32>
    %83 = merge %trueResult_14 {handshake.bb = 4 : ui32, handshake.name = "merge3"} : <i32>
    %84:4 = fork [4] %83 {handshake.bb = 4 : ui32, handshake.name = "fork14"} : <i32>
    %85 = merge %trueResult_16 {handshake.bb = 4 : ui32, handshake.name = "merge4"} : <i32>
    %86:3 = fork [3] %85 {handshake.bb = 4 : ui32, handshake.name = "fork15"} : <i32>
    %87 = trunci %86#0 {handshake.bb = 4 : ui32, handshake.name = "trunci1"} : <i32> to <i7>
    %89 = trunci %86#1 {handshake.bb = 4 : ui32, handshake.name = "trunci2"} : <i32> to <i4>
    %91 = merge %trueResult_18 {handshake.bb = 4 : ui32, handshake.name = "merge5"} : <i32>
    %92 = merge %trueResult_20 {handshake.bb = 4 : ui32, handshake.name = "merge6"} : <i32>
    %93:4 = fork [4] %92 {handshake.bb = 4 : ui32, handshake.name = "fork16"} : <i32>
    %result_26, %index_27 = control_merge [%trueResult_22]  {handshake.bb = 4 : ui32, handshake.name = "control_merge3"} : [<>] to <>, <i1>
    sink %index_27 {handshake.name = "sink7"} : <i1>
    %94:2 = fork [2] %result_26 {handshake.bb = 4 : ui32, handshake.name = "fork17"} : <>
    %95 = constant %94#0 {handshake.bb = 4 : ui32, handshake.name = "constant4", value = 1 : i2} : <>, <i2>
    %96 = extsi %95 {handshake.bb = 4 : ui32, handshake.name = "extsi2"} : <i2> to <i32>
    %97 = source {handshake.bb = 4 : ui32, handshake.name = "source1"} : <>
    %98 = constant %97 {handshake.bb = 4 : ui32, handshake.name = "constant7", value = -2 : i32} : <>, <i32>
    %99:3 = fork [3] %98 {handshake.bb = 4 : ui32, handshake.name = "fork18"} : <i32>
    %100 = source {handshake.bb = 4 : ui32, handshake.name = "source2"} : <>
    %101 = constant %100 {handshake.bb = 4 : ui32, handshake.name = "constant8", value = -1 : i32} : <>, <i32>
    %102:5 = fork [5] %101 {handshake.bb = 4 : ui32, handshake.name = "fork19"} : <i32>
    %103 = trunci %102#0 {handshake.bb = 4 : ui32, handshake.name = "trunci3"} : <i32> to <i7>
    %105 = trunci %102#1 {handshake.bb = 4 : ui32, handshake.name = "trunci4"} : <i32> to <i4>
    %107 = source {handshake.bb = 4 : ui32, handshake.name = "source3"} : <>
    %108 = constant %107 {handshake.bb = 4 : ui32, handshake.name = "constant12", value = 1 : i2} : <>, <i2>
    %109 = extsi %108 {handshake.bb = 4 : ui32, handshake.name = "extsi3"} : <i2> to <i32>
    %110:7 = fork [7] %109 {handshake.bb = 4 : ui32, handshake.name = "fork20"} : <i32>
    %111 = source {handshake.bb = 4 : ui32, handshake.name = "source4"} : <>
    %112 = constant %111 {handshake.bb = 4 : ui32, handshake.name = "constant13", value = 3 : i3} : <>, <i3>
    %113 = extsi %112 {handshake.bb = 4 : ui32, handshake.name = "extsi4"} : <i3> to <i32>
    %114:3 = fork [3] %113 {handshake.bb = 4 : ui32, handshake.name = "fork21"} : <i32>
    %115 = addi %84#3, %93#3 {handshake.bb = 4 : ui32, handshake.name = "addi0"} : <i32>
    %118 = xori %115, %102#4 {handshake.bb = 4 : ui32, handshake.name = "xori0"} : <i32>
    %120 = addi %118, %110#0 {handshake.bb = 4 : ui32, handshake.name = "addi2"} : <i32>
    %122 = addi %120, %80#5 {handshake.bb = 4 : ui32, handshake.name = "addi3"} : <i32>
    %124 = addi %122, %99#2 {handshake.bb = 4 : ui32, handshake.name = "addi4"} : <i32>
    %126:2 = fork [2] %124 {handshake.bb = 4 : ui32, handshake.name = "fork22"} : <i32>
    %127 = addi %87, %103 {handshake.bb = 4 : ui32, handshake.name = "addi6"} : <i7>
    %128 = shli %126#1, %110#1 {handshake.bb = 4 : ui32, handshake.name = "shli0"} : <i32>
    %131 = trunci %128 {handshake.bb = 4 : ui32, handshake.name = "trunci5"} : <i32> to <i7>
    %132 = shli %126#0, %114#0 {handshake.bb = 4 : ui32, handshake.name = "shli1"} : <i32>
    %135 = trunci %132 {handshake.bb = 4 : ui32, handshake.name = "trunci6"} : <i32> to <i7>
    %136 = addi %131, %135 {handshake.bb = 4 : ui32, handshake.name = "addi5"} : <i7>
    %137 = addi %127, %136 {handshake.bb = 4 : ui32, handshake.name = "addi15"} : <i7>
    %addressResult, %dataResult = load[%137] %outputs#0 {handshake.bb = 4 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i7>, <i32>, <i7>, <i32>
    %138 = addi %89, %105 {handshake.bb = 4 : ui32, handshake.name = "addi8"} : <i4>
    %addressResult_28, %dataResult_29 = load[%138] %outputs_0 {handshake.bb = 4 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i4>, <i32>, <i4>, <i32>
    %139 = muli %dataResult, %dataResult_29 {handshake.bb = 4 : ui32, handshake.name = "muli0"} : <i32>
    %140 = addi %84#2, %93#2 {handshake.bb = 4 : ui32, handshake.name = "addi7"} : <i32>
    %143 = xori %140, %102#3 {handshake.bb = 4 : ui32, handshake.name = "xori1"} : <i32>
    %145 = addi %143, %110#2 {handshake.bb = 4 : ui32, handshake.name = "addi9"} : <i32>
    %147 = addi %145, %148 {handshake.bb = 4 : ui32, handshake.name = "addi10"} : <i32>
    %148 = buffer %80#4, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer54"} : <i32>
    %149 = addi %147, %99#1 {handshake.bb = 4 : ui32, handshake.name = "addi11"} : <i32>
    %151:2 = fork [2] %149 {handshake.bb = 4 : ui32, handshake.name = "fork23"} : <i32>
    %152 = shli %153, %154 {handshake.bb = 4 : ui32, handshake.name = "shli2"} : <i32>
    %153 = buffer %151#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer56"} : <i32>
    %154 = buffer %110#3, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer57"} : <i32>
    %155 = shli %156, %157 {handshake.bb = 4 : ui32, handshake.name = "shli3"} : <i32>
    %156 = buffer %151#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer58"} : <i32>
    %157 = buffer %114#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer59"} : <i32>
    %158 = addi %152, %155 {handshake.bb = 4 : ui32, handshake.name = "addi12"} : <i32>
    %159 = addi %160, %158 {handshake.bb = 4 : ui32, handshake.name = "addi16"} : <i32>
    %160 = buffer %80#3, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer60"} : <i32>
    %161 = gate %159, %trueResult_24 {handshake.bb = 4 : ui32, handshake.name = "gate0"} : <i32>, !handshake.control<> to <i32>
    %162 = trunci %161 {handshake.bb = 4 : ui32, handshake.name = "trunci7"} : <i32> to <i7>
    %addressResult_30, %dataResult_31 = load[%162] %outputs#1 {handshake.bb = 4 : ui32, handshake.deps = #handshake<deps[["store0", 1, false], ["store0", 3, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load2"} : <i7>, <i32>, <i7>, <i32>
    %163 = subi %dataResult_31, %139 {handshake.bb = 4 : ui32, handshake.name = "subi0"} : <i32>
    %164 = addi %84#1, %93#1 {handshake.bb = 4 : ui32, handshake.name = "addi20"} : <i32>
    %167 = xori %164, %102#2 {handshake.bb = 4 : ui32, handshake.name = "xori2"} : <i32>
    %169 = addi %167, %110#4 {handshake.bb = 4 : ui32, handshake.name = "addi21"} : <i32>
    %171 = addi %169, %80#2 {handshake.bb = 4 : ui32, handshake.name = "addi13"} : <i32>
    %173 = addi %171, %174 {handshake.bb = 4 : ui32, handshake.name = "addi14"} : <i32>
    %174 = buffer %99#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer66"} : <i32>
    %175:2 = fork [2] %173 {handshake.bb = 4 : ui32, handshake.name = "fork24"} : <i32>
    %176 = shli %177, %178 {handshake.bb = 4 : ui32, handshake.name = "shli4"} : <i32>
    %177 = buffer %175#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer67"} : <i32>
    %178 = buffer %110#5, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 4 : ui32, handshake.name = "buffer68"} : <i32>
    %179 = trunci %176 {handshake.bb = 4 : ui32, handshake.name = "trunci8"} : <i32> to <i7>
    %180 = shli %181, %182 {handshake.bb = 4 : ui32, handshake.name = "shli5"} : <i32>
    %181 = buffer %175#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer69"} : <i32>
    %182 = buffer %114#2, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 4 : ui32, handshake.name = "buffer70"} : <i32>
    %183 = trunci %180 {handshake.bb = 4 : ui32, handshake.name = "trunci9"} : <i32> to <i7>
    %184 = addi %179, %183 {handshake.bb = 4 : ui32, handshake.name = "addi22"} : <i7>
    %185 = addi %81, %184 {handshake.bb = 4 : ui32, handshake.name = "addi17"} : <i7>
    %186 = buffer %doneResult, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer0"} : <>
    %addressResult_32, %dataResult_33, %doneResult = store[%185] %163 %outputs#2 {handshake.bb = 4 : ui32, handshake.deps = #handshake<deps[["load2", 1, false], ["store0", 1, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store0"} : <i7>, <i32>, <>, <i7>, <i32>, <>
    %187 = addi %93#0, %188 {handshake.bb = 4 : ui32, handshake.name = "addi18"} : <i32>
    %188 = buffer %110#6, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer71"} : <i32>
    %190 = br %187 {handshake.bb = 4 : ui32, handshake.name = "br13"} : <i32>
    %191 = br %80#1 {handshake.bb = 4 : ui32, handshake.name = "br14"} : <i32>
    %193 = br %84#0 {handshake.bb = 4 : ui32, handshake.name = "br15"} : <i32>
    %195 = br %86#2 {handshake.bb = 4 : ui32, handshake.name = "br16"} : <i32>
    %197 = br %91 {handshake.bb = 4 : ui32, handshake.name = "br17"} : <i32>
    %198 = br %94#1 {handshake.bb = 4 : ui32, handshake.name = "br18"} : <>
    %199 = merge %falseResult_13 {handshake.bb = 5 : ui32, handshake.name = "merge7"} : <i32>
    %200 = merge %falseResult_15 {handshake.bb = 5 : ui32, handshake.name = "merge8"} : <i32>
    %result_34, %index_35 = control_merge [%falseResult_23]  {handshake.bb = 5 : ui32, handshake.name = "control_merge4"} : [<>] to <>, <i1>
    sink %index_35 {handshake.name = "sink8"} : <i1>
    %201 = source {handshake.bb = 5 : ui32, handshake.name = "source5"} : <>
    %202 = constant %201 {handshake.bb = 5 : ui32, handshake.name = "constant14", value = 1 : i2} : <>, <i2>
    %203 = extsi %202 {handshake.bb = 5 : ui32, handshake.name = "extsi5"} : <i2> to <i32>
    %204 = addi %200, %203 {handshake.bb = 5 : ui32, handshake.name = "addi19"} : <i32>
    %205 = br %204 {handshake.bb = 5 : ui32, handshake.name = "br19"} : <i32>
    %206 = br %199 {handshake.bb = 5 : ui32, handshake.name = "br20"} : <i32>
    %207 = br %result_34 {handshake.bb = 5 : ui32, handshake.name = "br21"} : <>
    %result_36, %index_37 = control_merge [%falseResult_5]  {handshake.bb = 6 : ui32, handshake.name = "control_merge5"} : [<>] to <>, <i1>
    sink %index_37 {handshake.name = "sink9"} : <i1>
    %208:2 = fork [2] %result_36 {handshake.bb = 6 : ui32, handshake.name = "fork25"} : <>
    end {handshake.bb = 6 : ui32, handshake.name = "end0"} %memEnd_1, %memEnd, %0#1 : <>, <>, <>
  }
}

