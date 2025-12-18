module {
  handshake.func @kernel_3mm(%arg0: memref<100xi32>, %arg1: memref<100xi32>, %arg2: memref<100xi32>, %arg3: memref<100xi32>, %arg4: memref<100xi32>, %arg5: memref<100xi32>, %arg6: memref<100xi32>, %arg7: !handshake.control<>, %arg8: !handshake.control<>, %arg9: !handshake.control<>, %arg10: !handshake.control<>, %arg11: !handshake.control<>, %arg12: !handshake.control<>, %arg13: !handshake.control<>, %arg14: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["A", "B", "C", "D", "E", "F", "G", "A_start", "B_start", "C_start", "D_start", "E_start", "F_start", "G_start", "start"], resNames = ["A_end", "B_end", "C_end", "D_end", "E_end", "F_end", "G_end", "end"]} {
    %0:3 = fork [3] %arg14 {handshake.bb = 0 : ui32, handshake.name = "fork0"} : <>
    %1:2 = lsq[%arg6 : memref<100xi32>] (%arg13, %393#0, %addressResult_84, %dataResult_85, %446#0, %addressResult_92, %addressResult_94, %dataResult_95, %554#6)  {groupSizes = [1 : i32, 2 : i32], handshake.name = "lsq3"} : (!handshake.control<>, !handshake.control<>, !handshake.channel<i7>, !handshake.channel<i32>, !handshake.control<>, !handshake.channel<i7>, !handshake.channel<i7>, !handshake.channel<i32>, !handshake.control<>) -> (!handshake.channel<i32>, !handshake.control<>)
    %2:3 = lsq[%arg5 : memref<100xi32>] (%arg12, %210#0, %addressResult_44, %dataResult_45, %263#0, %addressResult_52, %addressResult_54, %dataResult_55, %447#1, %addressResult_90, %554#5)  {groupSizes = [1 : i32, 2 : i32, 1 : i32], handshake.name = "lsq4"} : (!handshake.control<>, !handshake.control<>, !handshake.channel<i7>, !handshake.channel<i32>, !handshake.control<>, !handshake.channel<i7>, !handshake.channel<i7>, !handshake.channel<i32>, !handshake.control<>, !handshake.channel<i7>, !handshake.control<>) -> (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>)
    %3:3 = lsq[%arg4 : memref<100xi32>] (%arg11, %27#0, %addressResult, %dataResult, %80#0, %addressResult_14, %addressResult_16, %dataResult_17, %447#0, %addressResult_88, %554#4)  {groupSizes = [1 : i32, 2 : i32, 1 : i32], handshake.name = "lsq5"} : (!handshake.control<>, !handshake.control<>, !handshake.channel<i7>, !handshake.channel<i32>, !handshake.control<>, !handshake.channel<i7>, !handshake.channel<i7>, !handshake.channel<i32>, !handshake.control<>, !handshake.channel<i7>, !handshake.control<>) -> (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>)
    %outputs, %memEnd = mem_controller[%arg3 : memref<100xi32>] %arg10 (%addressResult_50) %554#3 {connectedBlocks = [8 : i32], handshake.name = "mem_controller4"} :    (!handshake.channel<i7>) -> !handshake.channel<i32>
    %outputs_0, %memEnd_1 = mem_controller[%arg2 : memref<100xi32>] %arg9 (%addressResult_48) %554#2 {connectedBlocks = [8 : i32], handshake.name = "mem_controller5"} :    (!handshake.channel<i7>) -> !handshake.channel<i32>
    %outputs_2, %memEnd_3 = mem_controller[%arg1 : memref<100xi32>] %arg8 (%addressResult_12) %554#1 {connectedBlocks = [3 : i32], handshake.name = "mem_controller6"} :    (!handshake.channel<i7>) -> !handshake.channel<i32>
    %outputs_4, %memEnd_5 = mem_controller[%arg0 : memref<100xi32>] %arg7 (%addressResult_10) %554#0 {connectedBlocks = [3 : i32], handshake.name = "mem_controller7"} :    (!handshake.channel<i7>) -> !handshake.channel<i32>
    %4 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant0", value = false} : <>, <i1>
    %5 = br %4 {handshake.bb = 0 : ui32, handshake.name = "br3"} : <i1>
    %6 = extsi %5 {handshake.bb = 0 : ui32, handshake.name = "extsi47"} : <i1> to <i5>
    %7 = br %0#2 {handshake.bb = 0 : ui32, handshake.name = "br6"} : <>
    %8 = mux %index [%6, %trueResult_34] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i5>, <i5>] to <i5>
    %result, %index = control_merge [%7, %trueResult_36]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %9:2 = fork [2] %result {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <>
    %10 = constant %9#0 {handshake.bb = 1 : ui32, handshake.name = "constant1", value = false} : <>, <i1>
    %11 = br %10 {handshake.bb = 1 : ui32, handshake.name = "br9"} : <i1>
    %12 = extsi %11 {handshake.bb = 1 : ui32, handshake.name = "extsi46"} : <i1> to <i5>
    %13 = br %8 {handshake.bb = 1 : ui32, handshake.name = "br10"} : <i5>
    %14 = br %9#1 {handshake.bb = 1 : ui32, handshake.name = "br11"} : <>
    %15 = mux %26#1 [%12, %trueResult_26] {handshake.bb = 2 : ui32, handshake.name = "mux1"} : <i1>, [<i5>, <i5>] to <i5>
    %17:2 = fork [2] %15 {handshake.bb = 2 : ui32, handshake.name = "fork2"} : <i5>
    %18 = extsi %17#0 {handshake.bb = 2 : ui32, handshake.name = "extsi48"} : <i5> to <i7>
    %20 = mux %26#0 [%13, %trueResult_28] {handshake.bb = 2 : ui32, handshake.name = "mux2"} : <i1>, [<i5>, <i5>] to <i5>
    %22:2 = fork [2] %20 {handshake.bb = 2 : ui32, handshake.name = "fork3"} : <i5>
    %23 = extsi %22#1 {handshake.bb = 2 : ui32, handshake.name = "extsi49"} : <i5> to <i32>
    %25:2 = fork [2] %23 {handshake.bb = 2 : ui32, handshake.name = "fork4"} : <i32>
    %result_6, %index_7 = control_merge [%14, %trueResult_30]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>, <>] to <>, <i1>
    %26:2 = fork [2] %index_7 {handshake.bb = 2 : ui32, handshake.name = "fork5"} : <i1>
    %27:3 = lazy_fork [3] %result_6 {handshake.bb = 2 : ui32, handshake.name = "lazy_fork0"} : <>
    %28 = constant %27#2 {handshake.bb = 2 : ui32, handshake.name = "constant2", value = false} : <>, <i1>
    %29:2 = fork [2] %28 {handshake.bb = 2 : ui32, handshake.name = "fork6"} : <i1>
    %30 = extsi %29#1 {handshake.bb = 2 : ui32, handshake.name = "extsi3"} : <i1> to <i32>
    %32 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %33 = constant %32 {handshake.bb = 2 : ui32, handshake.name = "constant4", value = 1 : i2} : <>, <i2>
    %34 = extsi %33 {handshake.bb = 2 : ui32, handshake.name = "extsi4"} : <i2> to <i32>
    %35 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %36 = constant %35 {handshake.bb = 2 : ui32, handshake.name = "constant43", value = 3 : i3} : <>, <i3>
    %37 = extsi %36 {handshake.bb = 2 : ui32, handshake.name = "extsi5"} : <i3> to <i32>
    %38 = shli %25#0, %34 {handshake.bb = 2 : ui32, handshake.name = "shli0"} : <i32>
    %40 = trunci %38 {handshake.bb = 2 : ui32, handshake.name = "trunci0"} : <i32> to <i7>
    %41 = shli %25#1, %37 {handshake.bb = 2 : ui32, handshake.name = "shli1"} : <i32>
    %43 = trunci %41 {handshake.bb = 2 : ui32, handshake.name = "trunci1"} : <i32> to <i7>
    %44 = addi %40, %43 {handshake.bb = 2 : ui32, handshake.name = "addi27"} : <i7>
    %45 = addi %18, %44 {handshake.bb = 2 : ui32, handshake.name = "addi3"} : <i7>
    %addressResult, %dataResult = store[%45] %30 {handshake.bb = 2 : ui32, handshake.deps = #handshake<deps[["load2", 3], ["store1", 3], ["load6", 1]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 0>, handshake.name = "store0"} : <i7>, <i32>, <i7>, <i32>
    %46 = br %29#0 {handshake.bb = 2 : ui32, handshake.name = "br12"} : <i1>
    %48 = extsi %46 {handshake.bb = 2 : ui32, handshake.name = "extsi45"} : <i1> to <i5>
    %49 = br %22#0 {handshake.bb = 2 : ui32, handshake.name = "br13"} : <i5>
    %51 = br %17#1 {handshake.bb = 2 : ui32, handshake.name = "br14"} : <i5>
    %53 = br %27#1 {handshake.bb = 2 : ui32, handshake.name = "br15"} : <>
    %54 = mux %79#2 [%48, %trueResult] {handshake.bb = 3 : ui32, handshake.name = "mux3"} : <i1>, [<i5>, <i5>] to <i5>
    %56:3 = fork [3] %54 {handshake.bb = 3 : ui32, handshake.name = "fork7"} : <i5>
    %57 = extsi %56#0 {handshake.bb = 3 : ui32, handshake.name = "extsi50"} : <i5> to <i7>
    %59 = extsi %56#1 {handshake.bb = 3 : ui32, handshake.name = "extsi51"} : <i5> to <i6>
    %61 = extsi %56#2 {handshake.bb = 3 : ui32, handshake.name = "extsi52"} : <i5> to <i32>
    %63:2 = fork [2] %61 {handshake.bb = 3 : ui32, handshake.name = "fork8"} : <i32>
    %64 = mux %79#0 [%49, %trueResult_18] {handshake.bb = 3 : ui32, handshake.name = "mux4"} : <i1>, [<i5>, <i5>] to <i5>
    %66:2 = fork [2] %64 {handshake.bb = 3 : ui32, handshake.name = "fork9"} : <i5>
    %67 = extsi %66#1 {handshake.bb = 3 : ui32, handshake.name = "extsi53"} : <i5> to <i32>
    %69:6 = fork [6] %67 {handshake.bb = 3 : ui32, handshake.name = "fork10"} : <i32>
    %70 = mux %79#1 [%51, %trueResult_20] {handshake.bb = 3 : ui32, handshake.name = "mux5"} : <i1>, [<i5>, <i5>] to <i5>
    %72:4 = fork [4] %70 {handshake.bb = 3 : ui32, handshake.name = "fork11"} : <i5>
    %73 = extsi %72#0 {handshake.bb = 3 : ui32, handshake.name = "extsi54"} : <i5> to <i7>
    %75 = extsi %72#1 {handshake.bb = 3 : ui32, handshake.name = "extsi55"} : <i5> to <i7>
    %77 = extsi %72#2 {handshake.bb = 3 : ui32, handshake.name = "extsi56"} : <i5> to <i7>
    %result_8, %index_9 = control_merge [%53, %trueResult_22]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>, <>] to <>, <i1>
    %79:3 = fork [3] %index_9 {handshake.bb = 3 : ui32, handshake.name = "fork12"} : <i1>
    %80:2 = lazy_fork [2] %result_8 {handshake.bb = 3 : ui32, handshake.name = "lazy_fork1"} : <>
    %81 = source {handshake.bb = 3 : ui32, handshake.name = "source2"} : <>
    %82 = constant %81 {handshake.bb = 3 : ui32, handshake.name = "constant44", value = 10 : i5} : <>, <i5>
    %83 = extsi %82 {handshake.bb = 3 : ui32, handshake.name = "extsi57"} : <i5> to <i6>
    %84 = source {handshake.bb = 3 : ui32, handshake.name = "source3"} : <>
    %85 = constant %84 {handshake.bb = 3 : ui32, handshake.name = "constant45", value = 1 : i2} : <>, <i2>
    %86:2 = fork [2] %85 {handshake.bb = 3 : ui32, handshake.name = "fork13"} : <i2>
    %87 = extsi %86#0 {handshake.bb = 3 : ui32, handshake.name = "extsi58"} : <i2> to <i6>
    %89 = extsi %86#1 {handshake.bb = 3 : ui32, handshake.name = "extsi7"} : <i2> to <i32>
    %91:4 = fork [4] %89 {handshake.bb = 3 : ui32, handshake.name = "fork14"} : <i32>
    %92 = source {handshake.bb = 3 : ui32, handshake.name = "source4"} : <>
    %93 = constant %92 {handshake.bb = 3 : ui32, handshake.name = "constant46", value = 3 : i3} : <>, <i3>
    %94 = extsi %93 {handshake.bb = 3 : ui32, handshake.name = "extsi8"} : <i3> to <i32>
    %95:4 = fork [4] %94 {handshake.bb = 3 : ui32, handshake.name = "fork15"} : <i32>
    %96 = shli %69#0, %91#0 {handshake.bb = 3 : ui32, handshake.name = "shli2"} : <i32>
    %99 = trunci %96 {handshake.bb = 3 : ui32, handshake.name = "trunci2"} : <i32> to <i7>
    %100 = shli %69#1, %95#0 {handshake.bb = 3 : ui32, handshake.name = "shli3"} : <i32>
    %103 = trunci %100 {handshake.bb = 3 : ui32, handshake.name = "trunci3"} : <i32> to <i7>
    %104 = addi %99, %103 {handshake.bb = 3 : ui32, handshake.name = "addi28"} : <i7>
    %105 = addi %57, %104 {handshake.bb = 3 : ui32, handshake.name = "addi4"} : <i7>
    %addressResult_10, %dataResult_11 = load[%105] %outputs_4 {handshake.bb = 3 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i7>, <i32>, <i7>, <i32>
    %106 = shli %63#0, %91#1 {handshake.bb = 3 : ui32, handshake.name = "shli4"} : <i32>
    %109 = trunci %106 {handshake.bb = 3 : ui32, handshake.name = "trunci4"} : <i32> to <i7>
    %110 = shli %63#1, %95#1 {handshake.bb = 3 : ui32, handshake.name = "shli5"} : <i32>
    %113 = trunci %110 {handshake.bb = 3 : ui32, handshake.name = "trunci5"} : <i32> to <i7>
    %114 = addi %109, %113 {handshake.bb = 3 : ui32, handshake.name = "addi29"} : <i7>
    %115 = addi %73, %114 {handshake.bb = 3 : ui32, handshake.name = "addi5"} : <i7>
    %addressResult_12, %dataResult_13 = load[%115] %outputs_2 {handshake.bb = 3 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i7>, <i32>, <i7>, <i32>
    %116 = muli %dataResult_11, %dataResult_13 {handshake.bb = 3 : ui32, handshake.name = "muli0"} : <i32>
    %117 = shli %69#2, %91#2 {handshake.bb = 3 : ui32, handshake.name = "shli6"} : <i32>
    %120 = trunci %117 {handshake.bb = 3 : ui32, handshake.name = "trunci6"} : <i32> to <i7>
    %121 = shli %69#3, %95#2 {handshake.bb = 3 : ui32, handshake.name = "shli7"} : <i32>
    %124 = trunci %121 {handshake.bb = 3 : ui32, handshake.name = "trunci7"} : <i32> to <i7>
    %125 = addi %120, %124 {handshake.bb = 3 : ui32, handshake.name = "addi30"} : <i7>
    %126 = addi %75, %125 {handshake.bb = 3 : ui32, handshake.name = "addi6"} : <i7>
    %addressResult_14, %dataResult_15 = load[%126] %3#0 {handshake.bb = 3 : ui32, handshake.deps = #handshake<deps[["store1", 3], ["store1", 4]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 1>, handshake.name = "load2"} : <i7>, <i32>, <i7>, <i32>
    %127 = addi %dataResult_15, %116 {handshake.bb = 3 : ui32, handshake.name = "addi0"} : <i32>
    %128 = shli %69#4, %91#3 {handshake.bb = 3 : ui32, handshake.name = "shli8"} : <i32>
    %131 = trunci %128 {handshake.bb = 3 : ui32, handshake.name = "trunci8"} : <i32> to <i7>
    %132 = shli %69#5, %95#3 {handshake.bb = 3 : ui32, handshake.name = "shli9"} : <i32>
    %135 = trunci %132 {handshake.bb = 3 : ui32, handshake.name = "trunci9"} : <i32> to <i7>
    %136 = addi %131, %135 {handshake.bb = 3 : ui32, handshake.name = "addi31"} : <i7>
    %137 = addi %77, %136 {handshake.bb = 3 : ui32, handshake.name = "addi7"} : <i7>
    %addressResult_16, %dataResult_17 = store[%137] %127 {handshake.bb = 3 : ui32, handshake.deps = #handshake<deps[["load2", 3], ["store1", 3], ["load6", 1]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 1>, handshake.name = "store1"} : <i7>, <i32>, <i7>, <i32>
    %138 = addi %59, %87 {handshake.bb = 3 : ui32, handshake.name = "addi18"} : <i6>
    %139:2 = fork [2] %138 {handshake.bb = 3 : ui32, handshake.name = "fork16"} : <i6>
    %140 = trunci %139#0 {handshake.bb = 3 : ui32, handshake.name = "trunci10"} : <i6> to <i5>
    %142 = cmpi ult, %139#1, %83 {handshake.bb = 3 : ui32, handshake.name = "cmpi0"} : <i6>
    %144:4 = fork [4] %142 {handshake.bb = 3 : ui32, handshake.name = "fork17"} : <i1>
    %trueResult, %falseResult = cond_br %144#0, %140 {handshake.bb = 3 : ui32, handshake.name = "cond_br9"} : <i1>, <i5>
    sink %falseResult {handshake.name = "sink0"} : <i5>
    %trueResult_18, %falseResult_19 = cond_br %144#1, %66#0 {handshake.bb = 3 : ui32, handshake.name = "cond_br10"} : <i1>, <i5>
    %trueResult_20, %falseResult_21 = cond_br %144#2, %72#3 {handshake.bb = 3 : ui32, handshake.name = "cond_br11"} : <i1>, <i5>
    %trueResult_22, %falseResult_23 = cond_br %144#3, %80#1 {handshake.bb = 3 : ui32, handshake.name = "cond_br12"} : <i1>, <>
    %151 = merge %falseResult_19 {handshake.bb = 4 : ui32, handshake.name = "merge0"} : <i5>
    %152 = merge %falseResult_21 {handshake.bb = 4 : ui32, handshake.name = "merge1"} : <i5>
    %153 = extsi %152 {handshake.bb = 4 : ui32, handshake.name = "extsi59"} : <i5> to <i6>
    %result_24, %index_25 = control_merge [%falseResult_23]  {handshake.bb = 4 : ui32, handshake.name = "control_merge3"} : [<>] to <>, <i1>
    sink %index_25 {handshake.name = "sink1"} : <i1>
    %154 = source {handshake.bb = 4 : ui32, handshake.name = "source5"} : <>
    %155 = constant %154 {handshake.bb = 4 : ui32, handshake.name = "constant47", value = 10 : i5} : <>, <i5>
    %156 = extsi %155 {handshake.bb = 4 : ui32, handshake.name = "extsi60"} : <i5> to <i6>
    %157 = source {handshake.bb = 4 : ui32, handshake.name = "source6"} : <>
    %158 = constant %157 {handshake.bb = 4 : ui32, handshake.name = "constant48", value = 1 : i2} : <>, <i2>
    %159 = extsi %158 {handshake.bb = 4 : ui32, handshake.name = "extsi61"} : <i2> to <i6>
    %160 = addi %153, %159 {handshake.bb = 4 : ui32, handshake.name = "addi19"} : <i6>
    %161:2 = fork [2] %160 {handshake.bb = 4 : ui32, handshake.name = "fork18"} : <i6>
    %162 = trunci %161#0 {handshake.bb = 4 : ui32, handshake.name = "trunci11"} : <i6> to <i5>
    %164 = cmpi ult, %161#1, %156 {handshake.bb = 4 : ui32, handshake.name = "cmpi1"} : <i6>
    %166:3 = fork [3] %164 {handshake.bb = 4 : ui32, handshake.name = "fork19"} : <i1>
    %trueResult_26, %falseResult_27 = cond_br %166#0, %162 {handshake.bb = 4 : ui32, handshake.name = "cond_br13"} : <i1>, <i5>
    sink %falseResult_27 {handshake.name = "sink2"} : <i5>
    %trueResult_28, %falseResult_29 = cond_br %166#1, %151 {handshake.bb = 4 : ui32, handshake.name = "cond_br14"} : <i1>, <i5>
    %trueResult_30, %falseResult_31 = cond_br %166#2, %result_24 {handshake.bb = 4 : ui32, handshake.name = "cond_br15"} : <i1>, <>
    %170 = merge %falseResult_29 {handshake.bb = 5 : ui32, handshake.name = "merge2"} : <i5>
    %171 = extsi %170 {handshake.bb = 5 : ui32, handshake.name = "extsi62"} : <i5> to <i6>
    %result_32, %index_33 = control_merge [%falseResult_31]  {handshake.bb = 5 : ui32, handshake.name = "control_merge4"} : [<>] to <>, <i1>
    sink %index_33 {handshake.name = "sink3"} : <i1>
    %172:2 = fork [2] %result_32 {handshake.bb = 5 : ui32, handshake.name = "fork20"} : <>
    %173 = constant %172#0 {handshake.bb = 5 : ui32, handshake.name = "constant49", value = false} : <>, <i1>
    %174 = source {handshake.bb = 5 : ui32, handshake.name = "source7"} : <>
    %175 = constant %174 {handshake.bb = 5 : ui32, handshake.name = "constant50", value = 10 : i5} : <>, <i5>
    %176 = extsi %175 {handshake.bb = 5 : ui32, handshake.name = "extsi63"} : <i5> to <i6>
    %177 = source {handshake.bb = 5 : ui32, handshake.name = "source8"} : <>
    %178 = constant %177 {handshake.bb = 5 : ui32, handshake.name = "constant51", value = 1 : i2} : <>, <i2>
    %179 = extsi %178 {handshake.bb = 5 : ui32, handshake.name = "extsi64"} : <i2> to <i6>
    %180 = addi %171, %179 {handshake.bb = 5 : ui32, handshake.name = "addi20"} : <i6>
    %181:2 = fork [2] %180 {handshake.bb = 5 : ui32, handshake.name = "fork21"} : <i6>
    %182 = trunci %181#0 {handshake.bb = 5 : ui32, handshake.name = "trunci12"} : <i6> to <i5>
    %184 = cmpi ult, %181#1, %176 {handshake.bb = 5 : ui32, handshake.name = "cmpi2"} : <i6>
    %186:3 = fork [3] %184 {handshake.bb = 5 : ui32, handshake.name = "fork22"} : <i1>
    %trueResult_34, %falseResult_35 = cond_br %186#0, %182 {handshake.bb = 5 : ui32, handshake.name = "cond_br16"} : <i1>, <i5>
    sink %falseResult_35 {handshake.name = "sink4"} : <i5>
    %trueResult_36, %falseResult_37 = cond_br %186#1, %172#1 {handshake.bb = 5 : ui32, handshake.name = "cond_br17"} : <i1>, <>
    %trueResult_38, %falseResult_39 = cond_br %186#2, %173 {handshake.bb = 5 : ui32, handshake.name = "cond_br18"} : <i1>, <i1>
    sink %trueResult_38 {handshake.name = "sink5"} : <i1>
    %190 = extsi %falseResult_39 {handshake.bb = 5 : ui32, handshake.name = "extsi44"} : <i1> to <i5>
    %191 = mux %index_41 [%190, %trueResult_74] {handshake.bb = 6 : ui32, handshake.name = "mux6"} : <i1>, [<i5>, <i5>] to <i5>
    %result_40, %index_41 = control_merge [%falseResult_37, %trueResult_76]  {handshake.bb = 6 : ui32, handshake.name = "control_merge5"} : [<>, <>] to <>, <i1>
    %192:2 = fork [2] %result_40 {handshake.bb = 6 : ui32, handshake.name = "fork23"} : <>
    %193 = constant %192#0 {handshake.bb = 6 : ui32, handshake.name = "constant52", value = false} : <>, <i1>
    %194 = br %193 {handshake.bb = 6 : ui32, handshake.name = "br16"} : <i1>
    %195 = extsi %194 {handshake.bb = 6 : ui32, handshake.name = "extsi43"} : <i1> to <i5>
    %196 = br %191 {handshake.bb = 6 : ui32, handshake.name = "br17"} : <i5>
    %197 = br %192#1 {handshake.bb = 6 : ui32, handshake.name = "br18"} : <>
    %198 = mux %209#1 [%195, %trueResult_66] {handshake.bb = 7 : ui32, handshake.name = "mux7"} : <i1>, [<i5>, <i5>] to <i5>
    %200:2 = fork [2] %198 {handshake.bb = 7 : ui32, handshake.name = "fork24"} : <i5>
    %201 = extsi %200#0 {handshake.bb = 7 : ui32, handshake.name = "extsi65"} : <i5> to <i7>
    %203 = mux %209#0 [%196, %trueResult_68] {handshake.bb = 7 : ui32, handshake.name = "mux8"} : <i1>, [<i5>, <i5>] to <i5>
    %205:2 = fork [2] %203 {handshake.bb = 7 : ui32, handshake.name = "fork25"} : <i5>
    %206 = extsi %205#1 {handshake.bb = 7 : ui32, handshake.name = "extsi66"} : <i5> to <i32>
    %208:2 = fork [2] %206 {handshake.bb = 7 : ui32, handshake.name = "fork26"} : <i32>
    %result_42, %index_43 = control_merge [%197, %trueResult_70]  {handshake.bb = 7 : ui32, handshake.name = "control_merge6"} : [<>, <>] to <>, <i1>
    %209:2 = fork [2] %index_43 {handshake.bb = 7 : ui32, handshake.name = "fork27"} : <i1>
    %210:3 = lazy_fork [3] %result_42 {handshake.bb = 7 : ui32, handshake.name = "lazy_fork2"} : <>
    %211 = constant %210#2 {handshake.bb = 7 : ui32, handshake.name = "constant53", value = false} : <>, <i1>
    %212:2 = fork [2] %211 {handshake.bb = 7 : ui32, handshake.name = "fork28"} : <i1>
    %213 = extsi %212#1 {handshake.bb = 7 : ui32, handshake.name = "extsi16"} : <i1> to <i32>
    %215 = source {handshake.bb = 7 : ui32, handshake.name = "source9"} : <>
    %216 = constant %215 {handshake.bb = 7 : ui32, handshake.name = "constant54", value = 1 : i2} : <>, <i2>
    %217 = extsi %216 {handshake.bb = 7 : ui32, handshake.name = "extsi17"} : <i2> to <i32>
    %218 = source {handshake.bb = 7 : ui32, handshake.name = "source10"} : <>
    %219 = constant %218 {handshake.bb = 7 : ui32, handshake.name = "constant55", value = 3 : i3} : <>, <i3>
    %220 = extsi %219 {handshake.bb = 7 : ui32, handshake.name = "extsi18"} : <i3> to <i32>
    %221 = shli %208#0, %217 {handshake.bb = 7 : ui32, handshake.name = "shli10"} : <i32>
    %223 = trunci %221 {handshake.bb = 7 : ui32, handshake.name = "trunci13"} : <i32> to <i7>
    %224 = shli %208#1, %220 {handshake.bb = 7 : ui32, handshake.name = "shli11"} : <i32>
    %226 = trunci %224 {handshake.bb = 7 : ui32, handshake.name = "trunci14"} : <i32> to <i7>
    %227 = addi %223, %226 {handshake.bb = 7 : ui32, handshake.name = "addi32"} : <i7>
    %228 = addi %201, %227 {handshake.bb = 7 : ui32, handshake.name = "addi8"} : <i7>
    %addressResult_44, %dataResult_45 = store[%228] %213 {handshake.bb = 7 : ui32, handshake.deps = #handshake<deps[["load5", 3], ["store3", 3], ["load7", 1]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 0>, handshake.name = "store2"} : <i7>, <i32>, <i7>, <i32>
    %229 = br %212#0 {handshake.bb = 7 : ui32, handshake.name = "br19"} : <i1>
    %231 = extsi %229 {handshake.bb = 7 : ui32, handshake.name = "extsi42"} : <i1> to <i5>
    %232 = br %205#0 {handshake.bb = 7 : ui32, handshake.name = "br20"} : <i5>
    %234 = br %200#1 {handshake.bb = 7 : ui32, handshake.name = "br21"} : <i5>
    %236 = br %210#1 {handshake.bb = 7 : ui32, handshake.name = "br22"} : <>
    %237 = mux %262#2 [%231, %trueResult_56] {handshake.bb = 8 : ui32, handshake.name = "mux9"} : <i1>, [<i5>, <i5>] to <i5>
    %239:3 = fork [3] %237 {handshake.bb = 8 : ui32, handshake.name = "fork29"} : <i5>
    %240 = extsi %239#0 {handshake.bb = 8 : ui32, handshake.name = "extsi67"} : <i5> to <i7>
    %242 = extsi %239#1 {handshake.bb = 8 : ui32, handshake.name = "extsi68"} : <i5> to <i6>
    %244 = extsi %239#2 {handshake.bb = 8 : ui32, handshake.name = "extsi69"} : <i5> to <i32>
    %246:2 = fork [2] %244 {handshake.bb = 8 : ui32, handshake.name = "fork30"} : <i32>
    %247 = mux %262#0 [%232, %trueResult_58] {handshake.bb = 8 : ui32, handshake.name = "mux10"} : <i1>, [<i5>, <i5>] to <i5>
    %249:2 = fork [2] %247 {handshake.bb = 8 : ui32, handshake.name = "fork31"} : <i5>
    %250 = extsi %249#1 {handshake.bb = 8 : ui32, handshake.name = "extsi70"} : <i5> to <i32>
    %252:6 = fork [6] %250 {handshake.bb = 8 : ui32, handshake.name = "fork32"} : <i32>
    %253 = mux %262#1 [%234, %trueResult_60] {handshake.bb = 8 : ui32, handshake.name = "mux11"} : <i1>, [<i5>, <i5>] to <i5>
    %255:4 = fork [4] %253 {handshake.bb = 8 : ui32, handshake.name = "fork33"} : <i5>
    %256 = extsi %255#0 {handshake.bb = 8 : ui32, handshake.name = "extsi71"} : <i5> to <i7>
    %258 = extsi %255#1 {handshake.bb = 8 : ui32, handshake.name = "extsi72"} : <i5> to <i7>
    %260 = extsi %255#2 {handshake.bb = 8 : ui32, handshake.name = "extsi73"} : <i5> to <i7>
    %result_46, %index_47 = control_merge [%236, %trueResult_62]  {handshake.bb = 8 : ui32, handshake.name = "control_merge7"} : [<>, <>] to <>, <i1>
    %262:3 = fork [3] %index_47 {handshake.bb = 8 : ui32, handshake.name = "fork34"} : <i1>
    %263:2 = lazy_fork [2] %result_46 {handshake.bb = 8 : ui32, handshake.name = "lazy_fork3"} : <>
    %264 = source {handshake.bb = 8 : ui32, handshake.name = "source11"} : <>
    %265 = constant %264 {handshake.bb = 8 : ui32, handshake.name = "constant56", value = 10 : i5} : <>, <i5>
    %266 = extsi %265 {handshake.bb = 8 : ui32, handshake.name = "extsi74"} : <i5> to <i6>
    %267 = source {handshake.bb = 8 : ui32, handshake.name = "source12"} : <>
    %268 = constant %267 {handshake.bb = 8 : ui32, handshake.name = "constant57", value = 1 : i2} : <>, <i2>
    %269:2 = fork [2] %268 {handshake.bb = 8 : ui32, handshake.name = "fork35"} : <i2>
    %270 = extsi %269#0 {handshake.bb = 8 : ui32, handshake.name = "extsi75"} : <i2> to <i6>
    %272 = extsi %269#1 {handshake.bb = 8 : ui32, handshake.name = "extsi20"} : <i2> to <i32>
    %274:4 = fork [4] %272 {handshake.bb = 8 : ui32, handshake.name = "fork36"} : <i32>
    %275 = source {handshake.bb = 8 : ui32, handshake.name = "source13"} : <>
    %276 = constant %275 {handshake.bb = 8 : ui32, handshake.name = "constant58", value = 3 : i3} : <>, <i3>
    %277 = extsi %276 {handshake.bb = 8 : ui32, handshake.name = "extsi21"} : <i3> to <i32>
    %278:4 = fork [4] %277 {handshake.bb = 8 : ui32, handshake.name = "fork37"} : <i32>
    %279 = shli %252#0, %274#0 {handshake.bb = 8 : ui32, handshake.name = "shli12"} : <i32>
    %282 = trunci %279 {handshake.bb = 8 : ui32, handshake.name = "trunci15"} : <i32> to <i7>
    %283 = shli %252#1, %278#0 {handshake.bb = 8 : ui32, handshake.name = "shli13"} : <i32>
    %286 = trunci %283 {handshake.bb = 8 : ui32, handshake.name = "trunci16"} : <i32> to <i7>
    %287 = addi %282, %286 {handshake.bb = 8 : ui32, handshake.name = "addi33"} : <i7>
    %288 = addi %240, %287 {handshake.bb = 8 : ui32, handshake.name = "addi9"} : <i7>
    %addressResult_48, %dataResult_49 = load[%288] %outputs_0 {handshake.bb = 8 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load3"} : <i7>, <i32>, <i7>, <i32>
    %289 = shli %246#0, %274#1 {handshake.bb = 8 : ui32, handshake.name = "shli14"} : <i32>
    %292 = trunci %289 {handshake.bb = 8 : ui32, handshake.name = "trunci17"} : <i32> to <i7>
    %293 = shli %246#1, %278#1 {handshake.bb = 8 : ui32, handshake.name = "shli15"} : <i32>
    %296 = trunci %293 {handshake.bb = 8 : ui32, handshake.name = "trunci18"} : <i32> to <i7>
    %297 = addi %292, %296 {handshake.bb = 8 : ui32, handshake.name = "addi34"} : <i7>
    %298 = addi %256, %297 {handshake.bb = 8 : ui32, handshake.name = "addi10"} : <i7>
    %addressResult_50, %dataResult_51 = load[%298] %outputs {handshake.bb = 8 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load4"} : <i7>, <i32>, <i7>, <i32>
    %299 = muli %dataResult_49, %dataResult_51 {handshake.bb = 8 : ui32, handshake.name = "muli1"} : <i32>
    %300 = shli %252#2, %274#2 {handshake.bb = 8 : ui32, handshake.name = "shli16"} : <i32>
    %303 = trunci %300 {handshake.bb = 8 : ui32, handshake.name = "trunci19"} : <i32> to <i7>
    %304 = shli %252#3, %278#2 {handshake.bb = 8 : ui32, handshake.name = "shli17"} : <i32>
    %307 = trunci %304 {handshake.bb = 8 : ui32, handshake.name = "trunci20"} : <i32> to <i7>
    %308 = addi %303, %307 {handshake.bb = 8 : ui32, handshake.name = "addi35"} : <i7>
    %309 = addi %258, %308 {handshake.bb = 8 : ui32, handshake.name = "addi11"} : <i7>
    %addressResult_52, %dataResult_53 = load[%309] %2#0 {handshake.bb = 8 : ui32, handshake.deps = #handshake<deps[["store3", 3], ["store3", 4]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 1>, handshake.name = "load5"} : <i7>, <i32>, <i7>, <i32>
    %310 = addi %dataResult_53, %299 {handshake.bb = 8 : ui32, handshake.name = "addi1"} : <i32>
    %311 = shli %252#4, %274#3 {handshake.bb = 8 : ui32, handshake.name = "shli18"} : <i32>
    %314 = trunci %311 {handshake.bb = 8 : ui32, handshake.name = "trunci21"} : <i32> to <i7>
    %315 = shli %252#5, %278#3 {handshake.bb = 8 : ui32, handshake.name = "shli19"} : <i32>
    %318 = trunci %315 {handshake.bb = 8 : ui32, handshake.name = "trunci22"} : <i32> to <i7>
    %319 = addi %314, %318 {handshake.bb = 8 : ui32, handshake.name = "addi36"} : <i7>
    %320 = addi %260, %319 {handshake.bb = 8 : ui32, handshake.name = "addi12"} : <i7>
    %addressResult_54, %dataResult_55 = store[%320] %310 {handshake.bb = 8 : ui32, handshake.deps = #handshake<deps[["load5", 3], ["store3", 3], ["load7", 1]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 1>, handshake.name = "store3"} : <i7>, <i32>, <i7>, <i32>
    %321 = addi %242, %270 {handshake.bb = 8 : ui32, handshake.name = "addi21"} : <i6>
    %322:2 = fork [2] %321 {handshake.bb = 8 : ui32, handshake.name = "fork38"} : <i6>
    %323 = trunci %322#0 {handshake.bb = 8 : ui32, handshake.name = "trunci23"} : <i6> to <i5>
    %325 = cmpi ult, %322#1, %266 {handshake.bb = 8 : ui32, handshake.name = "cmpi3"} : <i6>
    %327:4 = fork [4] %325 {handshake.bb = 8 : ui32, handshake.name = "fork39"} : <i1>
    %trueResult_56, %falseResult_57 = cond_br %327#0, %323 {handshake.bb = 8 : ui32, handshake.name = "cond_br19"} : <i1>, <i5>
    sink %falseResult_57 {handshake.name = "sink6"} : <i5>
    %trueResult_58, %falseResult_59 = cond_br %327#1, %249#0 {handshake.bb = 8 : ui32, handshake.name = "cond_br20"} : <i1>, <i5>
    %trueResult_60, %falseResult_61 = cond_br %327#2, %255#3 {handshake.bb = 8 : ui32, handshake.name = "cond_br21"} : <i1>, <i5>
    %trueResult_62, %falseResult_63 = cond_br %327#3, %263#1 {handshake.bb = 8 : ui32, handshake.name = "cond_br22"} : <i1>, <>
    %334 = merge %falseResult_59 {handshake.bb = 9 : ui32, handshake.name = "merge3"} : <i5>
    %335 = merge %falseResult_61 {handshake.bb = 9 : ui32, handshake.name = "merge4"} : <i5>
    %336 = extsi %335 {handshake.bb = 9 : ui32, handshake.name = "extsi76"} : <i5> to <i6>
    %result_64, %index_65 = control_merge [%falseResult_63]  {handshake.bb = 9 : ui32, handshake.name = "control_merge8"} : [<>] to <>, <i1>
    sink %index_65 {handshake.name = "sink7"} : <i1>
    %337 = source {handshake.bb = 9 : ui32, handshake.name = "source14"} : <>
    %338 = constant %337 {handshake.bb = 9 : ui32, handshake.name = "constant59", value = 10 : i5} : <>, <i5>
    %339 = extsi %338 {handshake.bb = 9 : ui32, handshake.name = "extsi77"} : <i5> to <i6>
    %340 = source {handshake.bb = 9 : ui32, handshake.name = "source15"} : <>
    %341 = constant %340 {handshake.bb = 9 : ui32, handshake.name = "constant60", value = 1 : i2} : <>, <i2>
    %342 = extsi %341 {handshake.bb = 9 : ui32, handshake.name = "extsi78"} : <i2> to <i6>
    %343 = addi %336, %342 {handshake.bb = 9 : ui32, handshake.name = "addi22"} : <i6>
    %344:2 = fork [2] %343 {handshake.bb = 9 : ui32, handshake.name = "fork40"} : <i6>
    %345 = trunci %344#0 {handshake.bb = 9 : ui32, handshake.name = "trunci24"} : <i6> to <i5>
    %347 = cmpi ult, %344#1, %339 {handshake.bb = 9 : ui32, handshake.name = "cmpi4"} : <i6>
    %349:3 = fork [3] %347 {handshake.bb = 9 : ui32, handshake.name = "fork41"} : <i1>
    %trueResult_66, %falseResult_67 = cond_br %349#0, %345 {handshake.bb = 9 : ui32, handshake.name = "cond_br23"} : <i1>, <i5>
    sink %falseResult_67 {handshake.name = "sink8"} : <i5>
    %trueResult_68, %falseResult_69 = cond_br %349#1, %334 {handshake.bb = 9 : ui32, handshake.name = "cond_br24"} : <i1>, <i5>
    %trueResult_70, %falseResult_71 = cond_br %349#2, %result_64 {handshake.bb = 9 : ui32, handshake.name = "cond_br25"} : <i1>, <>
    %353 = merge %falseResult_69 {handshake.bb = 10 : ui32, handshake.name = "merge5"} : <i5>
    %354 = extsi %353 {handshake.bb = 10 : ui32, handshake.name = "extsi79"} : <i5> to <i6>
    %result_72, %index_73 = control_merge [%falseResult_71]  {handshake.bb = 10 : ui32, handshake.name = "control_merge9"} : [<>] to <>, <i1>
    sink %index_73 {handshake.name = "sink9"} : <i1>
    %355:2 = fork [2] %result_72 {handshake.bb = 10 : ui32, handshake.name = "fork42"} : <>
    %356 = constant %355#0 {handshake.bb = 10 : ui32, handshake.name = "constant61", value = false} : <>, <i1>
    %357 = source {handshake.bb = 10 : ui32, handshake.name = "source16"} : <>
    %358 = constant %357 {handshake.bb = 10 : ui32, handshake.name = "constant62", value = 10 : i5} : <>, <i5>
    %359 = extsi %358 {handshake.bb = 10 : ui32, handshake.name = "extsi80"} : <i5> to <i6>
    %360 = source {handshake.bb = 10 : ui32, handshake.name = "source17"} : <>
    %361 = constant %360 {handshake.bb = 10 : ui32, handshake.name = "constant63", value = 1 : i2} : <>, <i2>
    %362 = extsi %361 {handshake.bb = 10 : ui32, handshake.name = "extsi81"} : <i2> to <i6>
    %363 = addi %354, %362 {handshake.bb = 10 : ui32, handshake.name = "addi23"} : <i6>
    %364:2 = fork [2] %363 {handshake.bb = 10 : ui32, handshake.name = "fork43"} : <i6>
    %365 = trunci %364#0 {handshake.bb = 10 : ui32, handshake.name = "trunci25"} : <i6> to <i5>
    %367 = cmpi ult, %364#1, %359 {handshake.bb = 10 : ui32, handshake.name = "cmpi5"} : <i6>
    %369:3 = fork [3] %367 {handshake.bb = 10 : ui32, handshake.name = "fork44"} : <i1>
    %trueResult_74, %falseResult_75 = cond_br %369#0, %365 {handshake.bb = 10 : ui32, handshake.name = "cond_br26"} : <i1>, <i5>
    sink %falseResult_75 {handshake.name = "sink10"} : <i5>
    %trueResult_76, %falseResult_77 = cond_br %369#1, %355#1 {handshake.bb = 10 : ui32, handshake.name = "cond_br27"} : <i1>, <>
    %trueResult_78, %falseResult_79 = cond_br %369#2, %356 {handshake.bb = 10 : ui32, handshake.name = "cond_br28"} : <i1>, <i1>
    sink %trueResult_78 {handshake.name = "sink11"} : <i1>
    %373 = extsi %falseResult_79 {handshake.bb = 10 : ui32, handshake.name = "extsi41"} : <i1> to <i5>
    %374 = mux %index_81 [%373, %trueResult_114] {handshake.bb = 11 : ui32, handshake.name = "mux12"} : <i1>, [<i5>, <i5>] to <i5>
    %result_80, %index_81 = control_merge [%falseResult_77, %trueResult_116]  {handshake.bb = 11 : ui32, handshake.name = "control_merge10"} : [<>, <>] to <>, <i1>
    %375:2 = fork [2] %result_80 {handshake.bb = 11 : ui32, handshake.name = "fork45"} : <>
    %376 = constant %375#0 {handshake.bb = 11 : ui32, handshake.name = "constant64", value = false} : <>, <i1>
    %377 = br %376 {handshake.bb = 11 : ui32, handshake.name = "br23"} : <i1>
    %378 = extsi %377 {handshake.bb = 11 : ui32, handshake.name = "extsi40"} : <i1> to <i5>
    %379 = br %374 {handshake.bb = 11 : ui32, handshake.name = "br24"} : <i5>
    %380 = br %375#1 {handshake.bb = 11 : ui32, handshake.name = "br25"} : <>
    %381 = mux %392#1 [%378, %trueResult_106] {handshake.bb = 12 : ui32, handshake.name = "mux13"} : <i1>, [<i5>, <i5>] to <i5>
    %383:2 = fork [2] %381 {handshake.bb = 12 : ui32, handshake.name = "fork46"} : <i5>
    %384 = extsi %383#0 {handshake.bb = 12 : ui32, handshake.name = "extsi82"} : <i5> to <i7>
    %386 = mux %392#0 [%379, %trueResult_108] {handshake.bb = 12 : ui32, handshake.name = "mux14"} : <i1>, [<i5>, <i5>] to <i5>
    %388:2 = fork [2] %386 {handshake.bb = 12 : ui32, handshake.name = "fork47"} : <i5>
    %389 = extsi %388#1 {handshake.bb = 12 : ui32, handshake.name = "extsi83"} : <i5> to <i32>
    %391:2 = fork [2] %389 {handshake.bb = 12 : ui32, handshake.name = "fork48"} : <i32>
    %result_82, %index_83 = control_merge [%380, %trueResult_110]  {handshake.bb = 12 : ui32, handshake.name = "control_merge11"} : [<>, <>] to <>, <i1>
    %392:2 = fork [2] %index_83 {handshake.bb = 12 : ui32, handshake.name = "fork49"} : <i1>
    %393:3 = lazy_fork [3] %result_82 {handshake.bb = 12 : ui32, handshake.name = "lazy_fork4"} : <>
    %394 = constant %393#2 {handshake.bb = 12 : ui32, handshake.name = "constant65", value = false} : <>, <i1>
    %395:2 = fork [2] %394 {handshake.bb = 12 : ui32, handshake.name = "fork50"} : <i1>
    %396 = extsi %395#1 {handshake.bb = 12 : ui32, handshake.name = "extsi29"} : <i1> to <i32>
    %398 = source {handshake.bb = 12 : ui32, handshake.name = "source18"} : <>
    %399 = constant %398 {handshake.bb = 12 : ui32, handshake.name = "constant66", value = 1 : i2} : <>, <i2>
    %400 = extsi %399 {handshake.bb = 12 : ui32, handshake.name = "extsi30"} : <i2> to <i32>
    %401 = source {handshake.bb = 12 : ui32, handshake.name = "source19"} : <>
    %402 = constant %401 {handshake.bb = 12 : ui32, handshake.name = "constant67", value = 3 : i3} : <>, <i3>
    %403 = extsi %402 {handshake.bb = 12 : ui32, handshake.name = "extsi31"} : <i3> to <i32>
    %404 = shli %391#0, %400 {handshake.bb = 12 : ui32, handshake.name = "shli20"} : <i32>
    %406 = trunci %404 {handshake.bb = 12 : ui32, handshake.name = "trunci26"} : <i32> to <i7>
    %407 = shli %391#1, %403 {handshake.bb = 12 : ui32, handshake.name = "shli21"} : <i32>
    %409 = trunci %407 {handshake.bb = 12 : ui32, handshake.name = "trunci27"} : <i32> to <i7>
    %410 = addi %406, %409 {handshake.bb = 12 : ui32, handshake.name = "addi37"} : <i7>
    %411 = addi %384, %410 {handshake.bb = 12 : ui32, handshake.name = "addi13"} : <i7>
    %addressResult_84, %dataResult_85 = store[%411] %396 {handshake.bb = 12 : ui32, handshake.deps = #handshake<deps[["load8", 3], ["store5", 3]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 0>, handshake.name = "store4"} : <i7>, <i32>, <i7>, <i32>
    %412 = br %395#0 {handshake.bb = 12 : ui32, handshake.name = "br26"} : <i1>
    %414 = extsi %412 {handshake.bb = 12 : ui32, handshake.name = "extsi39"} : <i1> to <i5>
    %415 = br %388#0 {handshake.bb = 12 : ui32, handshake.name = "br27"} : <i5>
    %417 = br %383#1 {handshake.bb = 12 : ui32, handshake.name = "br28"} : <i5>
    %419 = br %393#1 {handshake.bb = 12 : ui32, handshake.name = "br29"} : <>
    %420 = mux %445#2 [%414, %trueResult_96] {handshake.bb = 13 : ui32, handshake.name = "mux15"} : <i1>, [<i5>, <i5>] to <i5>
    %422:3 = fork [3] %420 {handshake.bb = 13 : ui32, handshake.name = "fork51"} : <i5>
    %423 = extsi %422#0 {handshake.bb = 13 : ui32, handshake.name = "extsi84"} : <i5> to <i7>
    %425 = extsi %422#1 {handshake.bb = 13 : ui32, handshake.name = "extsi85"} : <i5> to <i6>
    %427 = extsi %422#2 {handshake.bb = 13 : ui32, handshake.name = "extsi86"} : <i5> to <i32>
    %429:2 = fork [2] %427 {handshake.bb = 13 : ui32, handshake.name = "fork52"} : <i32>
    %430 = mux %445#0 [%415, %trueResult_98] {handshake.bb = 13 : ui32, handshake.name = "mux16"} : <i1>, [<i5>, <i5>] to <i5>
    %432:2 = fork [2] %430 {handshake.bb = 13 : ui32, handshake.name = "fork53"} : <i5>
    %433 = extsi %432#1 {handshake.bb = 13 : ui32, handshake.name = "extsi87"} : <i5> to <i32>
    %435:6 = fork [6] %433 {handshake.bb = 13 : ui32, handshake.name = "fork54"} : <i32>
    %436 = mux %445#1 [%417, %trueResult_100] {handshake.bb = 13 : ui32, handshake.name = "mux17"} : <i1>, [<i5>, <i5>] to <i5>
    %438:4 = fork [4] %436 {handshake.bb = 13 : ui32, handshake.name = "fork55"} : <i5>
    %439 = extsi %438#0 {handshake.bb = 13 : ui32, handshake.name = "extsi88"} : <i5> to <i7>
    %441 = extsi %438#1 {handshake.bb = 13 : ui32, handshake.name = "extsi89"} : <i5> to <i7>
    %443 = extsi %438#2 {handshake.bb = 13 : ui32, handshake.name = "extsi90"} : <i5> to <i7>
    %result_86, %index_87 = control_merge [%419, %trueResult_102]  {handshake.bb = 13 : ui32, handshake.name = "control_merge12"} : [<>, <>] to <>, <i1>
    %445:3 = fork [3] %index_87 {handshake.bb = 13 : ui32, handshake.name = "fork56"} : <i1>
    %446:3 = lazy_fork [3] %result_86 {handshake.bb = 13 : ui32, handshake.name = "lazy_fork5"} : <>
    %447:2 = fork [2] %446#2 {handshake.bb = 13 : ui32, handshake.name = "fork57"} : <>
    %448 = source {handshake.bb = 13 : ui32, handshake.name = "source20"} : <>
    %449 = constant %448 {handshake.bb = 13 : ui32, handshake.name = "constant68", value = 10 : i5} : <>, <i5>
    %450 = extsi %449 {handshake.bb = 13 : ui32, handshake.name = "extsi91"} : <i5> to <i6>
    %451 = source {handshake.bb = 13 : ui32, handshake.name = "source21"} : <>
    %452 = constant %451 {handshake.bb = 13 : ui32, handshake.name = "constant69", value = 1 : i2} : <>, <i2>
    %453:2 = fork [2] %452 {handshake.bb = 13 : ui32, handshake.name = "fork58"} : <i2>
    %454 = extsi %453#0 {handshake.bb = 13 : ui32, handshake.name = "extsi92"} : <i2> to <i6>
    %456 = extsi %453#1 {handshake.bb = 13 : ui32, handshake.name = "extsi33"} : <i2> to <i32>
    %458:4 = fork [4] %456 {handshake.bb = 13 : ui32, handshake.name = "fork59"} : <i32>
    %459 = source {handshake.bb = 13 : ui32, handshake.name = "source22"} : <>
    %460 = constant %459 {handshake.bb = 13 : ui32, handshake.name = "constant70", value = 3 : i3} : <>, <i3>
    %461 = extsi %460 {handshake.bb = 13 : ui32, handshake.name = "extsi34"} : <i3> to <i32>
    %462:4 = fork [4] %461 {handshake.bb = 13 : ui32, handshake.name = "fork60"} : <i32>
    %463 = shli %435#0, %458#0 {handshake.bb = 13 : ui32, handshake.name = "shli22"} : <i32>
    %466 = trunci %463 {handshake.bb = 13 : ui32, handshake.name = "trunci28"} : <i32> to <i7>
    %467 = shli %435#1, %462#0 {handshake.bb = 13 : ui32, handshake.name = "shli23"} : <i32>
    %470 = trunci %467 {handshake.bb = 13 : ui32, handshake.name = "trunci29"} : <i32> to <i7>
    %471 = addi %466, %470 {handshake.bb = 13 : ui32, handshake.name = "addi38"} : <i7>
    %472 = addi %423, %471 {handshake.bb = 13 : ui32, handshake.name = "addi14"} : <i7>
    %addressResult_88, %dataResult_89 = load[%472] %3#1 {handshake.bb = 13 : ui32, handshake.mem_interface = #handshake.mem_interface<LSQ: 2>, handshake.name = "load6"} : <i7>, <i32>, <i7>, <i32>
    %473 = shli %429#0, %458#1 {handshake.bb = 13 : ui32, handshake.name = "shli24"} : <i32>
    %476 = trunci %473 {handshake.bb = 13 : ui32, handshake.name = "trunci30"} : <i32> to <i7>
    %477 = shli %429#1, %462#1 {handshake.bb = 13 : ui32, handshake.name = "shli25"} : <i32>
    %480 = trunci %477 {handshake.bb = 13 : ui32, handshake.name = "trunci31"} : <i32> to <i7>
    %481 = addi %476, %480 {handshake.bb = 13 : ui32, handshake.name = "addi39"} : <i7>
    %482 = addi %439, %481 {handshake.bb = 13 : ui32, handshake.name = "addi15"} : <i7>
    %addressResult_90, %dataResult_91 = load[%482] %2#1 {handshake.bb = 13 : ui32, handshake.mem_interface = #handshake.mem_interface<LSQ: 2>, handshake.name = "load7"} : <i7>, <i32>, <i7>, <i32>
    %483 = muli %dataResult_89, %dataResult_91 {handshake.bb = 13 : ui32, handshake.name = "muli2"} : <i32>
    %484 = shli %435#2, %458#2 {handshake.bb = 13 : ui32, handshake.name = "shli26"} : <i32>
    %487 = trunci %484 {handshake.bb = 13 : ui32, handshake.name = "trunci32"} : <i32> to <i7>
    %488 = shli %435#3, %462#2 {handshake.bb = 13 : ui32, handshake.name = "shli27"} : <i32>
    %491 = trunci %488 {handshake.bb = 13 : ui32, handshake.name = "trunci33"} : <i32> to <i7>
    %492 = addi %487, %491 {handshake.bb = 13 : ui32, handshake.name = "addi40"} : <i7>
    %493 = addi %441, %492 {handshake.bb = 13 : ui32, handshake.name = "addi16"} : <i7>
    %addressResult_92, %dataResult_93 = load[%493] %1#0 {handshake.bb = 13 : ui32, handshake.deps = #handshake<deps[["store5", 3], ["store5", 4]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 1>, handshake.name = "load8"} : <i7>, <i32>, <i7>, <i32>
    %494 = addi %dataResult_93, %483 {handshake.bb = 13 : ui32, handshake.name = "addi2"} : <i32>
    %495 = shli %497, %496 {handshake.bb = 13 : ui32, handshake.name = "shli28"} : <i32>
    %496 = buffer %458#3, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 13 : ui32, handshake.name = "buffer146"} : <i32>
    %497 = buffer %435#4, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 13 : ui32, handshake.name = "buffer147"} : <i32>
    %498 = trunci %495 {handshake.bb = 13 : ui32, handshake.name = "trunci34"} : <i32> to <i7>
    %499 = shli %435#5, %462#3 {handshake.bb = 13 : ui32, handshake.name = "shli29"} : <i32>
    %502 = trunci %499 {handshake.bb = 13 : ui32, handshake.name = "trunci35"} : <i32> to <i7>
    %503 = addi %498, %502 {handshake.bb = 13 : ui32, handshake.name = "addi41"} : <i7>
    %504 = addi %443, %503 {handshake.bb = 13 : ui32, handshake.name = "addi17"} : <i7>
    %addressResult_94, %dataResult_95 = store[%504] %494 {handshake.bb = 13 : ui32, handshake.deps = #handshake<deps[["load8", 3], ["store5", 3]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 1>, handshake.name = "store5"} : <i7>, <i32>, <i7>, <i32>
    %505 = addi %425, %454 {handshake.bb = 13 : ui32, handshake.name = "addi24"} : <i6>
    %506:2 = fork [2] %505 {handshake.bb = 13 : ui32, handshake.name = "fork61"} : <i6>
    %507 = trunci %506#0 {handshake.bb = 13 : ui32, handshake.name = "trunci36"} : <i6> to <i5>
    %509 = cmpi ult, %506#1, %450 {handshake.bb = 13 : ui32, handshake.name = "cmpi6"} : <i6>
    %511:4 = fork [4] %509 {handshake.bb = 13 : ui32, handshake.name = "fork62"} : <i1>
    %trueResult_96, %falseResult_97 = cond_br %511#0, %507 {handshake.bb = 13 : ui32, handshake.name = "cond_br29"} : <i1>, <i5>
    sink %falseResult_97 {handshake.name = "sink12"} : <i5>
    %trueResult_98, %falseResult_99 = cond_br %511#1, %432#0 {handshake.bb = 13 : ui32, handshake.name = "cond_br30"} : <i1>, <i5>
    %trueResult_100, %falseResult_101 = cond_br %511#2, %438#3 {handshake.bb = 13 : ui32, handshake.name = "cond_br31"} : <i1>, <i5>
    %trueResult_102, %falseResult_103 = cond_br %511#3, %446#1 {handshake.bb = 13 : ui32, handshake.name = "cond_br32"} : <i1>, <>
    %518 = merge %falseResult_99 {handshake.bb = 14 : ui32, handshake.name = "merge6"} : <i5>
    %519 = merge %falseResult_101 {handshake.bb = 14 : ui32, handshake.name = "merge7"} : <i5>
    %520 = extsi %519 {handshake.bb = 14 : ui32, handshake.name = "extsi93"} : <i5> to <i6>
    %result_104, %index_105 = control_merge [%falseResult_103]  {handshake.bb = 14 : ui32, handshake.name = "control_merge13"} : [<>] to <>, <i1>
    sink %index_105 {handshake.name = "sink13"} : <i1>
    %521 = source {handshake.bb = 14 : ui32, handshake.name = "source23"} : <>
    %522 = constant %521 {handshake.bb = 14 : ui32, handshake.name = "constant71", value = 10 : i5} : <>, <i5>
    %523 = extsi %522 {handshake.bb = 14 : ui32, handshake.name = "extsi94"} : <i5> to <i6>
    %524 = source {handshake.bb = 14 : ui32, handshake.name = "source24"} : <>
    %525 = constant %524 {handshake.bb = 14 : ui32, handshake.name = "constant72", value = 1 : i2} : <>, <i2>
    %526 = extsi %525 {handshake.bb = 14 : ui32, handshake.name = "extsi95"} : <i2> to <i6>
    %527 = addi %520, %526 {handshake.bb = 14 : ui32, handshake.name = "addi25"} : <i6>
    %528:2 = fork [2] %527 {handshake.bb = 14 : ui32, handshake.name = "fork63"} : <i6>
    %529 = trunci %528#0 {handshake.bb = 14 : ui32, handshake.name = "trunci37"} : <i6> to <i5>
    %531 = cmpi ult, %528#1, %523 {handshake.bb = 14 : ui32, handshake.name = "cmpi7"} : <i6>
    %533:3 = fork [3] %531 {handshake.bb = 14 : ui32, handshake.name = "fork64"} : <i1>
    %trueResult_106, %falseResult_107 = cond_br %533#0, %529 {handshake.bb = 14 : ui32, handshake.name = "cond_br33"} : <i1>, <i5>
    sink %falseResult_107 {handshake.name = "sink14"} : <i5>
    %trueResult_108, %falseResult_109 = cond_br %533#1, %518 {handshake.bb = 14 : ui32, handshake.name = "cond_br34"} : <i1>, <i5>
    %trueResult_110, %falseResult_111 = cond_br %533#2, %result_104 {handshake.bb = 14 : ui32, handshake.name = "cond_br35"} : <i1>, <>
    %537 = merge %falseResult_109 {handshake.bb = 15 : ui32, handshake.name = "merge8"} : <i5>
    %538 = extsi %537 {handshake.bb = 15 : ui32, handshake.name = "extsi96"} : <i5> to <i6>
    %result_112, %index_113 = control_merge [%falseResult_111]  {handshake.bb = 15 : ui32, handshake.name = "control_merge14"} : [<>] to <>, <i1>
    sink %index_113 {handshake.name = "sink15"} : <i1>
    %539 = source {handshake.bb = 15 : ui32, handshake.name = "source25"} : <>
    %540 = constant %539 {handshake.bb = 15 : ui32, handshake.name = "constant73", value = 10 : i5} : <>, <i5>
    %541 = extsi %540 {handshake.bb = 15 : ui32, handshake.name = "extsi97"} : <i5> to <i6>
    %542 = source {handshake.bb = 15 : ui32, handshake.name = "source26"} : <>
    %543 = constant %542 {handshake.bb = 15 : ui32, handshake.name = "constant74", value = 1 : i2} : <>, <i2>
    %544 = extsi %543 {handshake.bb = 15 : ui32, handshake.name = "extsi98"} : <i2> to <i6>
    %545 = addi %538, %544 {handshake.bb = 15 : ui32, handshake.name = "addi26"} : <i6>
    %546:2 = fork [2] %545 {handshake.bb = 15 : ui32, handshake.name = "fork65"} : <i6>
    %547 = trunci %548 {handshake.bb = 15 : ui32, handshake.name = "trunci38"} : <i6> to <i5>
    %548 = buffer %546#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 15 : ui32, handshake.name = "buffer163"} : <i6>
    %549 = cmpi ult, %546#1, %541 {handshake.bb = 15 : ui32, handshake.name = "cmpi8"} : <i6>
    %551:2 = fork [2] %549 {handshake.bb = 15 : ui32, handshake.name = "fork66"} : <i1>
    %trueResult_114, %falseResult_115 = cond_br %551#0, %547 {handshake.bb = 15 : ui32, handshake.name = "cond_br36"} : <i1>, <i5>
    sink %falseResult_115 {handshake.name = "sink16"} : <i5>
    %trueResult_116, %falseResult_117 = cond_br %551#1, %result_112 {handshake.bb = 15 : ui32, handshake.name = "cond_br37"} : <i1>, <>
    %result_118, %index_119 = control_merge [%falseResult_117]  {handshake.bb = 16 : ui32, handshake.name = "control_merge15"} : [<>] to <>, <i1>
    sink %index_119 {handshake.name = "sink17"} : <i1>
    %554:7 = fork [7] %result_118 {handshake.bb = 16 : ui32, handshake.name = "fork67"} : <>
    end {handshake.bb = 16 : ui32, handshake.name = "end0"} %memEnd_5, %memEnd_3, %memEnd_1, %memEnd, %3#2, %2#2, %1#1, %0#1 : <>, <>, <>, <>, <>, <>, <>, <>
  }
}

