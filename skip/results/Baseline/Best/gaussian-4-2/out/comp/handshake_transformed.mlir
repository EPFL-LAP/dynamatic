module {
  handshake.func @gaussian(%arg0: memref<20xi32>, %arg1: memref<400xi32>, %arg2: !handshake.control<>, %arg3: !handshake.control<>, %arg4: !handshake.control<>, ...) -> (!handshake.channel<i32>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["c", "a", "c_start", "a_start", "start"], resNames = ["out0", "c_end", "a_end", "end"]} {
    %0:4 = fork [4] %arg4 {handshake.bb = 0 : ui32, handshake.name = "fork0"} : <>
    %1:3 = lsq[%arg1 : memref<400xi32>] (%arg3, %84#0, %addressResult, %addressResult_16, %addressResult_18, %dataResult_19, %186#1)  {groupSizes = [3 : i32], handshake.name = "lsq1"} : (!handshake.control<>, !handshake.control<>, !handshake.channel<i9>, !handshake.channel<i9>, !handshake.channel<i9>, !handshake.channel<i32>, !handshake.control<>) -> (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>)
    %outputs, %memEnd = mem_controller[%arg0 : memref<20xi32>] %arg2 (%addressResult_14) %186#0 {connectedBlocks = [3 : i32], handshake.name = "mem_controller1"} :    (!handshake.channel<i5>) -> !handshake.channel<i32>
    %2 = constant %0#1 {handshake.bb = 0 : ui32, handshake.name = "constant0", value = false} : <>, <i1>
    %3 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant1", value = 1 : i2} : <>, <i2>
    %4 = br %3 {handshake.bb = 0 : ui32, handshake.name = "br2"} : <i2>
    %5 = extsi %4 {handshake.bb = 0 : ui32, handshake.name = "extsi15"} : <i2> to <i6>
    %6 = br %2 {handshake.bb = 0 : ui32, handshake.name = "br4"} : <i1>
    %7 = extsi %6 {handshake.bb = 0 : ui32, handshake.name = "extsi16"} : <i1> to <i32>
    %8 = br %0#3 {handshake.bb = 0 : ui32, handshake.name = "br5"} : <>
    %9 = mux %16#0 [%5, %trueResult_36] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i6>, <i6>] to <i6>
    %11:2 = fork [2] %9 {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <i6>
    %12 = extsi %11#1 {handshake.bb = 1 : ui32, handshake.name = "extsi17"} : <i6> to <i7>
    %14 = mux %16#1 [%7, %trueResult_38] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i1>, [<i32>, <i32>] to <i32>
    %result, %index = control_merge [%8, %trueResult_40]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %16:2 = fork [2] %index {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <i1>
    %17 = source {handshake.bb = 1 : ui32, handshake.name = "source0"} : <>
    %18 = constant %17 {handshake.bb = 1 : ui32, handshake.name = "constant3", value = 1 : i2} : <>, <i2>
    %19 = extsi %18 {handshake.bb = 1 : ui32, handshake.name = "extsi18"} : <i2> to <i7>
    %20 = addi %12, %19 {handshake.bb = 1 : ui32, handshake.name = "addi2"} : <i7>
    %21 = br %20 {handshake.bb = 1 : ui32, handshake.name = "br6"} : <i7>
    %22 = br %14 {handshake.bb = 1 : ui32, handshake.name = "br7"} : <i32>
    %23 = br %11#0 {handshake.bb = 1 : ui32, handshake.name = "br8"} : <i6>
    %25 = br %result {handshake.bb = 1 : ui32, handshake.name = "br9"} : <>
    %26 = mux %35#1 [%21, %162] {handshake.bb = 2 : ui32, handshake.name = "mux2"} : <i1>, [<i7>, <i7>] to <i7>
    %28:2 = fork [2] %26 {handshake.bb = 2 : ui32, handshake.name = "fork3"} : <i7>
    %29 = trunci %28#0 {handshake.bb = 2 : ui32, handshake.name = "trunci0"} : <i7> to <i6>
    %31 = mux %35#2 [%22, %163] {handshake.bb = 2 : ui32, handshake.name = "mux3"} : <i1>, [<i32>, <i32>] to <i32>
    %33 = mux %35#0 [%23, %164] {handshake.bb = 2 : ui32, handshake.name = "mux4"} : <i1>, [<i6>, <i6>] to <i6>
    %result_0, %index_1 = control_merge [%25, %165]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>, <>] to <>, <i1>
    %35:3 = fork [3] %index_1 {handshake.bb = 2 : ui32, handshake.name = "fork4"} : <i1>
    %36:2 = fork [2] %result_0 {handshake.bb = 2 : ui32, handshake.name = "fork5"} : <>
    %37 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %38 = constant %37 {handshake.bb = 2 : ui32, handshake.name = "constant5", value = 19 : i6} : <>, <i6>
    %39 = extsi %38 {handshake.bb = 2 : ui32, handshake.name = "extsi19"} : <i6> to <i7>
    %40 = constant %36#0 {handshake.bb = 2 : ui32, handshake.name = "constant6", value = 1 : i2} : <>, <i2>
    %41:2 = fork [2] %40 {handshake.bb = 2 : ui32, handshake.name = "fork6"} : <i2>
    %42 = cmpi ult, %28#1, %39 {handshake.bb = 2 : ui32, handshake.name = "cmpi2"} : <i7>
    %44:6 = fork [6] %42 {handshake.bb = 2 : ui32, handshake.name = "fork7"} : <i1>
    %trueResult, %falseResult = cond_br %44#5, %41#0 {handshake.bb = 2 : ui32, handshake.name = "cond_br3"} : <i1>, <i2>
    sink %falseResult {handshake.name = "sink0"} : <i2>
    %47 = extsi %trueResult {handshake.bb = 2 : ui32, handshake.name = "extsi14"} : <i2> to <i6>
    %trueResult_2, %falseResult_3 = cond_br %44#4, %41#1 {handshake.bb = 2 : ui32, handshake.name = "cond_br4"} : <i1>, <i2>
    sink %falseResult_3 {handshake.name = "sink1"} : <i2>
    %50 = extsi %trueResult_2 {handshake.bb = 2 : ui32, handshake.name = "extsi20"} : <i2> to <i32>
    %trueResult_4, %falseResult_5 = cond_br %44#2, %31 {handshake.bb = 2 : ui32, handshake.name = "cond_br5"} : <i1>, <i32>
    %trueResult_6, %falseResult_7 = cond_br %44#1, %33 {handshake.bb = 2 : ui32, handshake.name = "cond_br6"} : <i1>, <i6>
    %trueResult_8, %falseResult_9 = cond_br %44#0, %29 {handshake.bb = 2 : ui32, handshake.name = "cond_br7"} : <i1>, <i6>
    sink %falseResult_9 {handshake.name = "sink2"} : <i6>
    %trueResult_10, %falseResult_11 = cond_br %44#3, %36#1 {handshake.bb = 2 : ui32, handshake.name = "cond_br8"} : <i1>, <>
    %55 = mux %83#2 [%47, %trueResult_20] {handshake.bb = 3 : ui32, handshake.name = "mux5"} : <i1>, [<i6>, <i6>] to <i6>
    %57 = extsi %55 {handshake.bb = 3 : ui32, handshake.name = "extsi21"} : <i6> to <i7>
    %58 = mux %83#3 [%50, %trueResult_22] {handshake.bb = 3 : ui32, handshake.name = "mux6"} : <i1>, [<i32>, <i32>] to <i32>
    %60:5 = fork [5] %58 {handshake.bb = 3 : ui32, handshake.name = "fork8"} : <i32>
    %61 = trunci %60#0 {handshake.bb = 3 : ui32, handshake.name = "trunci1"} : <i32> to <i9>
    %63 = trunci %60#1 {handshake.bb = 3 : ui32, handshake.name = "trunci2"} : <i32> to <i9>
    %65 = trunci %60#2 {handshake.bb = 3 : ui32, handshake.name = "trunci3"} : <i32> to <i9>
    %67 = mux %83#4 [%trueResult_4, %trueResult_24] {handshake.bb = 3 : ui32, handshake.name = "mux7"} : <i1>, [<i32>, <i32>] to <i32>
    %69 = mux %83#0 [%trueResult_6, %trueResult_26] {handshake.bb = 3 : ui32, handshake.name = "mux8"} : <i1>, [<i6>, <i6>] to <i6>
    %71:3 = fork [3] %69 {handshake.bb = 3 : ui32, handshake.name = "fork9"} : <i6>
    %72 = extsi %71#2 {handshake.bb = 3 : ui32, handshake.name = "extsi22"} : <i6> to <i32>
    %74:2 = fork [2] %72 {handshake.bb = 3 : ui32, handshake.name = "fork10"} : <i32>
    %75 = trunci %71#0 {handshake.bb = 3 : ui32, handshake.name = "trunci4"} : <i6> to <i5>
    %77 = mux %83#1 [%trueResult_8, %trueResult_28] {handshake.bb = 3 : ui32, handshake.name = "mux9"} : <i1>, [<i6>, <i6>] to <i6>
    %79:2 = fork [2] %77 {handshake.bb = 3 : ui32, handshake.name = "fork11"} : <i6>
    %80 = extsi %79#1 {handshake.bb = 3 : ui32, handshake.name = "extsi23"} : <i6> to <i32>
    %82:4 = fork [4] %80 {handshake.bb = 3 : ui32, handshake.name = "fork12"} : <i32>
    %result_12, %index_13 = control_merge [%trueResult_10, %trueResult_30]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>, <>] to <>, <i1>
    %83:5 = fork [5] %index_13 {handshake.bb = 3 : ui32, handshake.name = "fork13"} : <i1>
    %84:2 = fork [2] %result_12 {handshake.bb = 3 : ui32, handshake.name = "fork14"} : <>
    %85 = source {handshake.bb = 3 : ui32, handshake.name = "source2"} : <>
    %86 = constant %85 {handshake.bb = 3 : ui32, handshake.name = "constant19", value = 20 : i6} : <>, <i6>
    %87 = extsi %86 {handshake.bb = 3 : ui32, handshake.name = "extsi24"} : <i6> to <i7>
    %88 = source {handshake.bb = 3 : ui32, handshake.name = "source3"} : <>
    %89 = constant %88 {handshake.bb = 3 : ui32, handshake.name = "constant20", value = 1 : i2} : <>, <i2>
    %90:2 = fork [2] %89 {handshake.bb = 3 : ui32, handshake.name = "fork15"} : <i2>
    %91 = extsi %90#0 {handshake.bb = 3 : ui32, handshake.name = "extsi25"} : <i2> to <i7>
    %93 = extsi %90#1 {handshake.bb = 3 : ui32, handshake.name = "extsi7"} : <i2> to <i32>
    %95 = source {handshake.bb = 3 : ui32, handshake.name = "source5"} : <>
    %96 = constant %95 {handshake.bb = 3 : ui32, handshake.name = "constant21", value = 4 : i4} : <>, <i4>
    %97 = extsi %96 {handshake.bb = 3 : ui32, handshake.name = "extsi9"} : <i4> to <i32>
    %98:3 = fork [3] %97 {handshake.bb = 3 : ui32, handshake.name = "fork16"} : <i32>
    %99 = source {handshake.bb = 3 : ui32, handshake.name = "source6"} : <>
    %100 = constant %99 {handshake.bb = 3 : ui32, handshake.name = "constant22", value = 2 : i3} : <>, <i3>
    %101 = extsi %100 {handshake.bb = 3 : ui32, handshake.name = "extsi10"} : <i3> to <i32>
    %102:3 = fork [3] %101 {handshake.bb = 3 : ui32, handshake.name = "fork17"} : <i32>
    %103 = shli %82#0, %102#0 {handshake.bb = 3 : ui32, handshake.name = "shli0"} : <i32>
    %106 = trunci %103 {handshake.bb = 3 : ui32, handshake.name = "trunci5"} : <i32> to <i9>
    %107 = shli %82#1, %98#0 {handshake.bb = 3 : ui32, handshake.name = "shli1"} : <i32>
    %110 = trunci %107 {handshake.bb = 3 : ui32, handshake.name = "trunci6"} : <i32> to <i9>
    %111 = addi %106, %110 {handshake.bb = 3 : ui32, handshake.name = "addi9"} : <i9>
    %112 = addi %61, %111 {handshake.bb = 3 : ui32, handshake.name = "addi3"} : <i9>
    %addressResult, %dataResult = load[%112] %1#0 {handshake.bb = 3 : ui32, handshake.deps = #handshake<deps[["store0", 0]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 0>, handshake.name = "load0"} : <i9>, <i32>, <i9>, <i32>
    %addressResult_14, %dataResult_15 = load[%75] %outputs {handshake.bb = 3 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i5>, <i32>, <i5>, <i32>
    %113 = shli %74#0, %102#1 {handshake.bb = 3 : ui32, handshake.name = "shli2"} : <i32>
    %116 = trunci %113 {handshake.bb = 3 : ui32, handshake.name = "trunci7"} : <i32> to <i9>
    %117 = shli %74#1, %98#1 {handshake.bb = 3 : ui32, handshake.name = "shli3"} : <i32>
    %120 = trunci %117 {handshake.bb = 3 : ui32, handshake.name = "trunci8"} : <i32> to <i9>
    %121 = addi %116, %120 {handshake.bb = 3 : ui32, handshake.name = "addi10"} : <i9>
    %122 = addi %63, %121 {handshake.bb = 3 : ui32, handshake.name = "addi4"} : <i9>
    %addressResult_16, %dataResult_17 = load[%122] %1#1 {handshake.bb = 3 : ui32, handshake.deps = #handshake<deps[["store0", 0]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 0>, handshake.name = "load2"} : <i9>, <i32>, <i9>, <i32>
    %123 = muli %dataResult_15, %dataResult_17 {handshake.bb = 3 : ui32, handshake.name = "muli0"} : <i32>
    %124 = subi %dataResult, %123 {handshake.bb = 3 : ui32, handshake.name = "subi0"} : <i32>
    %125 = shli %82#2, %102#2 {handshake.bb = 3 : ui32, handshake.name = "shli4"} : <i32>
    %128 = trunci %125 {handshake.bb = 3 : ui32, handshake.name = "trunci9"} : <i32> to <i9>
    %129 = shli %82#3, %98#2 {handshake.bb = 3 : ui32, handshake.name = "shli5"} : <i32>
    %132 = trunci %129 {handshake.bb = 3 : ui32, handshake.name = "trunci10"} : <i32> to <i9>
    %133 = addi %128, %132 {handshake.bb = 3 : ui32, handshake.name = "addi11"} : <i9>
    %134 = addi %65, %133 {handshake.bb = 3 : ui32, handshake.name = "addi5"} : <i9>
    %addressResult_18, %dataResult_19 = store[%134] %124 {handshake.bb = 3 : ui32, handshake.deps = #handshake<deps[["load0", 0], ["load2", 0], ["store0", 0]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 0>, handshake.name = "store0"} : <i9>, <i32>, <i9>, <i32>
    %135 = addi %67, %60#4 {handshake.bb = 3 : ui32, handshake.name = "addi0"} : <i32>
    %137 = addi %60#3, %93 {handshake.bb = 3 : ui32, handshake.name = "addi1"} : <i32>
    %139 = addi %57, %91 {handshake.bb = 3 : ui32, handshake.name = "addi6"} : <i7>
    %140:2 = fork [2] %139 {handshake.bb = 3 : ui32, handshake.name = "fork18"} : <i7>
    %141 = trunci %140#0 {handshake.bb = 3 : ui32, handshake.name = "trunci11"} : <i7> to <i6>
    %143 = cmpi ult, %140#1, %87 {handshake.bb = 3 : ui32, handshake.name = "cmpi0"} : <i7>
    %145:6 = fork [6] %143 {handshake.bb = 3 : ui32, handshake.name = "fork19"} : <i1>
    %trueResult_20, %falseResult_21 = cond_br %145#0, %141 {handshake.bb = 3 : ui32, handshake.name = "cond_br9"} : <i1>, <i6>
    sink %falseResult_21 {handshake.name = "sink3"} : <i6>
    %trueResult_22, %falseResult_23 = cond_br %145#3, %137 {handshake.bb = 3 : ui32, handshake.name = "cond_br10"} : <i1>, <i32>
    sink %falseResult_23 {handshake.name = "sink4"} : <i32>
    %trueResult_24, %falseResult_25 = cond_br %145#4, %135 {handshake.bb = 3 : ui32, handshake.name = "cond_br11"} : <i1>, <i32>
    %trueResult_26, %falseResult_27 = cond_br %145#1, %71#1 {handshake.bb = 3 : ui32, handshake.name = "cond_br12"} : <i1>, <i6>
    %trueResult_28, %falseResult_29 = cond_br %145#2, %79#0 {handshake.bb = 3 : ui32, handshake.name = "cond_br13"} : <i1>, <i6>
    %trueResult_30, %falseResult_31 = cond_br %145#5, %84#1 {handshake.bb = 3 : ui32, handshake.name = "cond_br14"} : <i1>, <>
    %154 = merge %falseResult_27 {handshake.bb = 4 : ui32, handshake.name = "merge0"} : <i6>
    %155 = merge %falseResult_29 {handshake.bb = 4 : ui32, handshake.name = "merge1"} : <i6>
    %156 = extsi %155 {handshake.bb = 4 : ui32, handshake.name = "extsi26"} : <i6> to <i7>
    %157 = merge %falseResult_25 {handshake.bb = 4 : ui32, handshake.name = "merge2"} : <i32>
    %result_32, %index_33 = control_merge [%falseResult_31]  {handshake.bb = 4 : ui32, handshake.name = "control_merge3"} : [<>] to <>, <i1>
    sink %index_33 {handshake.name = "sink5"} : <i1>
    %158 = source {handshake.bb = 4 : ui32, handshake.name = "source7"} : <>
    %159 = constant %158 {handshake.bb = 4 : ui32, handshake.name = "constant23", value = 1 : i2} : <>, <i2>
    %160 = extsi %159 {handshake.bb = 4 : ui32, handshake.name = "extsi27"} : <i2> to <i7>
    %161 = addi %156, %160 {handshake.bb = 4 : ui32, handshake.name = "addi8"} : <i7>
    %162 = br %161 {handshake.bb = 4 : ui32, handshake.name = "br10"} : <i7>
    %163 = br %157 {handshake.bb = 4 : ui32, handshake.name = "br11"} : <i32>
    %164 = br %154 {handshake.bb = 4 : ui32, handshake.name = "br12"} : <i6>
    %165 = br %result_32 {handshake.bb = 4 : ui32, handshake.name = "br13"} : <>
    %166 = merge %falseResult_7 {handshake.bb = 5 : ui32, handshake.name = "merge3"} : <i6>
    %167 = extsi %166 {handshake.bb = 5 : ui32, handshake.name = "extsi28"} : <i6> to <i7>
    %168 = merge %falseResult_5 {handshake.bb = 5 : ui32, handshake.name = "merge4"} : <i32>
    %result_34, %index_35 = control_merge [%falseResult_11]  {handshake.bb = 5 : ui32, handshake.name = "control_merge4"} : [<>] to <>, <i1>
    sink %index_35 {handshake.name = "sink6"} : <i1>
    %169 = source {handshake.bb = 5 : ui32, handshake.name = "source8"} : <>
    %170 = constant %169 {handshake.bb = 5 : ui32, handshake.name = "constant24", value = 19 : i6} : <>, <i6>
    %171 = extsi %170 {handshake.bb = 5 : ui32, handshake.name = "extsi29"} : <i6> to <i7>
    %172 = source {handshake.bb = 5 : ui32, handshake.name = "source9"} : <>
    %173 = constant %172 {handshake.bb = 5 : ui32, handshake.name = "constant25", value = 1 : i2} : <>, <i2>
    %174 = extsi %173 {handshake.bb = 5 : ui32, handshake.name = "extsi30"} : <i2> to <i7>
    %175 = addi %167, %174 {handshake.bb = 5 : ui32, handshake.name = "addi7"} : <i7>
    %176:2 = fork [2] %175 {handshake.bb = 5 : ui32, handshake.name = "fork20"} : <i7>
    %177 = trunci %176#0 {handshake.bb = 5 : ui32, handshake.name = "trunci12"} : <i7> to <i6>
    %179 = cmpi ult, %176#1, %171 {handshake.bb = 5 : ui32, handshake.name = "cmpi1"} : <i7>
    %181:3 = fork [3] %179 {handshake.bb = 5 : ui32, handshake.name = "fork21"} : <i1>
    %trueResult_36, %falseResult_37 = cond_br %181#0, %177 {handshake.bb = 5 : ui32, handshake.name = "cond_br15"} : <i1>, <i6>
    sink %falseResult_37 {handshake.name = "sink7"} : <i6>
    %trueResult_38, %falseResult_39 = cond_br %181#1, %168 {handshake.bb = 5 : ui32, handshake.name = "cond_br16"} : <i1>, <i32>
    %trueResult_40, %falseResult_41 = cond_br %181#2, %result_34 {handshake.bb = 5 : ui32, handshake.name = "cond_br17"} : <i1>, <>
    %185 = merge %falseResult_39 {handshake.bb = 6 : ui32, handshake.name = "merge5"} : <i32>
    %result_42, %index_43 = control_merge [%falseResult_41]  {handshake.bb = 6 : ui32, handshake.name = "control_merge5"} : [<>] to <>, <i1>
    sink %index_43 {handshake.name = "sink8"} : <i1>
    %186:2 = fork [2] %result_42 {handshake.bb = 6 : ui32, handshake.name = "fork22"} : <>
    end {handshake.bb = 6 : ui32, handshake.name = "end0"} %185, %memEnd, %1#2, %0#2 : <i32>, <>, <>, <>
  }
}

