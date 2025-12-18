module {
  handshake.func @gaussian(%arg0: memref<20xi32>, %arg1: memref<400xi32>, %arg2: !handshake.control<>, %arg3: !handshake.control<>, %arg4: !handshake.control<>, ...) -> (!handshake.channel<i32>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["c", "a", "c_start", "a_start", "start"], cfg.edges = "[0,1][2,3,5,cmpi2][4,2][1,2][3,3,4,cmpi0][5,1,6,cmpi1]", resNames = ["out0", "c_end", "a_end", "end"]} {
    %0:9 = fork [9] %arg4 {handshake.bb = 0 : ui32, handshake.name = "fork0"} : <>
    %outputs:3, %memEnd = mem_controller[%arg1 : memref<400xi32>] %arg3 (%149, %addressResult, %addressResult_40, %addressResult_42, %dataResult_43) %288#1 {connectedBlocks = [3 : i32], handshake.name = "mem_controller1"} :    (!handshake.channel<i32>, !handshake.channel<i9>, !handshake.channel<i9>, !handshake.channel<i9>, !handshake.channel<i32>) -> (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>)
    %outputs_0, %memEnd_1 = mem_controller[%arg0 : memref<20xi32>] %arg2 (%addressResult_36) %288#0 {connectedBlocks = [3 : i32], handshake.name = "mem_controller2"} :    (!handshake.channel<i5>) -> !handshake.channel<i32>
    %1 = constant %0#2 {handshake.bb = 0 : ui32, handshake.name = "constant5", value = 1000 : i11} : <>, <i11>
    %2:2 = fork [2] %1 {handshake.bb = 0 : ui32, handshake.name = "fork1"} : <i11>
    %3 = extsi %2#0 {handshake.bb = 1 : ui32, handshake.name = "extsi0"} : <i11> to <i32>
    %5 = extsi %2#1 {handshake.bb = 1 : ui32, handshake.name = "extsi1"} : <i11> to <i32>
    %7 = constant %0#1 {handshake.bb = 0 : ui32, handshake.name = "constant6", value = false} : <>, <i1>
    %8 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant19", value = 1 : i2} : <>, <i2>
    %9 = br %8 {handshake.bb = 0 : ui32, handshake.name = "br2"} : <i2>
    %10 = extsi %9 {handshake.bb = 0 : ui32, handshake.name = "extsi18"} : <i2> to <i6>
    %11 = br %7 {handshake.bb = 0 : ui32, handshake.name = "br4"} : <i1>
    %12 = extsi %11 {handshake.bb = 0 : ui32, handshake.name = "extsi19"} : <i1> to <i32>
    %13 = br %0#8 {handshake.bb = 0 : ui32, handshake.name = "br5"} : <>
    %14 = mux %28#0 [%0#7, %trueResult_62] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux10"} : <i1>, [<>, <>] to <>
    %16 = mux %28#1 [%3, %trueResult_58] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux11"} : <i1>, [<i32>, <i32>] to <i32>
    %18 = mux %28#2 [%5, %trueResult_60] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux12"} : <i1>, [<i32>, <i32>] to <i32>
    %20 = mux %28#3 [%0#6, %trueResult_68] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux13"} : <i1>, [<>, <>] to <>
    %22 = mux %28#4 [%0#5, %trueResult_66] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux14"} : <i1>, [<>, <>] to <>
    %24 = mux %28#5 [%0#4, %trueResult_64] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux15"} : <i1>, [<>, <>] to <>
    %26 = init %283#7 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "init0"} : <i1>
    %28:6 = fork [6] %26 {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <i1>
    %29 = mux %36#0 [%10, %trueResult_72] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i6>, <i6>] to <i6>
    %31:2 = fork [2] %29 {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <i6>
    %32 = extsi %31#1 {handshake.bb = 1 : ui32, handshake.name = "extsi20"} : <i6> to <i7>
    %34 = mux %36#1 [%12, %trueResult_74] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i1>, [<i32>, <i32>] to <i32>
    %result, %index = control_merge [%13, %trueResult_76]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %36:2 = fork [2] %index {handshake.bb = 1 : ui32, handshake.name = "fork4"} : <i1>
    %37 = source {handshake.bb = 1 : ui32, handshake.name = "source0"} : <>
    %38 = constant %37 {handshake.bb = 1 : ui32, handshake.name = "constant20", value = 1 : i2} : <>, <i2>
    %39 = extsi %38 {handshake.bb = 1 : ui32, handshake.name = "extsi21"} : <i2> to <i7>
    %40 = addi %32, %39 {handshake.bb = 1 : ui32, handshake.name = "addi2"} : <i7>
    %41 = br %40 {handshake.bb = 1 : ui32, handshake.name = "br6"} : <i7>
    %42 = br %34 {handshake.bb = 1 : ui32, handshake.name = "br7"} : <i32>
    %43 = br %31#0 {handshake.bb = 1 : ui32, handshake.name = "br8"} : <i6>
    %45 = br %result {handshake.bb = 1 : ui32, handshake.name = "br9"} : <>
    %46 = mux %62#0 [%14, %105#0] {ftd.phi, handshake.bb = 2 : ui32, handshake.name = "mux16"} : <i1>, [<>, <>] to <>
    %48 = mux %62#1 [%16, %97#0] {ftd.phi, handshake.bb = 2 : ui32, handshake.name = "mux17"} : <i1>, [<i32>, <i32>] to <i32>
    %51 = mux %62#2 [%18, %97#1] {ftd.phi, handshake.bb = 2 : ui32, handshake.name = "mux18"} : <i1>, [<i32>, <i32>] to <i32>
    %54 = mux %62#3 [%20, %105#1] {ftd.phi, handshake.bb = 2 : ui32, handshake.name = "mux19"} : <i1>, [<>, <>] to <>
    %56 = mux %57 [%22, %101#0] {ftd.phi, handshake.bb = 2 : ui32, handshake.name = "mux20"} : <i1>, [<>, <>] to <>
    %57 = buffer %62#4, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer21"} : <i1>
    %58 = mux %59 [%24, %101#1] {ftd.phi, handshake.bb = 2 : ui32, handshake.name = "mux21"} : <i1>, [<>, <>] to <>
    %59 = buffer %62#5, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer22"} : <i1>
    %60 = init %81#2 {ftd.imerge, handshake.bb = 2 : ui32, handshake.name = "init6"} : <i1>
    %62:6 = fork [6] %60 {handshake.bb = 2 : ui32, handshake.name = "fork5"} : <i1>
    %63 = mux %72#1 [%41, %258] {handshake.bb = 2 : ui32, handshake.name = "mux2"} : <i1>, [<i7>, <i7>] to <i7>
    %65:2 = fork [2] %63 {handshake.bb = 2 : ui32, handshake.name = "fork6"} : <i7>
    %66 = trunci %65#0 {handshake.bb = 2 : ui32, handshake.name = "trunci0"} : <i7> to <i6>
    %68 = mux %72#2 [%42, %259] {handshake.bb = 2 : ui32, handshake.name = "mux3"} : <i1>, [<i32>, <i32>] to <i32>
    %70 = mux %72#0 [%43, %260] {handshake.bb = 2 : ui32, handshake.name = "mux4"} : <i1>, [<i6>, <i6>] to <i6>
    %result_2, %index_3 = control_merge [%45, %261]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>, <>] to <>, <i1>
    %72:3 = fork [3] %index_3 {handshake.bb = 2 : ui32, handshake.name = "fork7"} : <i1>
    %73:2 = fork [2] %result_2 {handshake.bb = 2 : ui32, handshake.name = "fork8"} : <>
    %74 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %75 = constant %74 {handshake.bb = 2 : ui32, handshake.name = "constant21", value = 19 : i6} : <>, <i6>
    %76 = extsi %75 {handshake.bb = 2 : ui32, handshake.name = "extsi22"} : <i6> to <i7>
    %77 = constant %73#0 {handshake.bb = 2 : ui32, handshake.name = "constant22", value = 1 : i2} : <>, <i2>
    %78:2 = fork [2] %77 {handshake.bb = 2 : ui32, handshake.name = "fork9"} : <i2>
    %79 = cmpi ult, %65#1, %76 {handshake.bb = 2 : ui32, handshake.name = "cmpi2"} : <i7>
    %81:13 = fork [13] %79 {handshake.bb = 2 : ui32, handshake.name = "fork10"} : <i1>
    %trueResult, %falseResult = cond_br %81#12, %78#0 {handshake.bb = 2 : ui32, handshake.name = "cond_br3"} : <i1>, <i2>
    sink %falseResult {handshake.name = "sink0"} : <i2>
    %84 = extsi %trueResult {handshake.bb = 2 : ui32, handshake.name = "extsi17"} : <i2> to <i6>
    %trueResult_4, %falseResult_5 = cond_br %81#11, %78#1 {handshake.bb = 2 : ui32, handshake.name = "cond_br4"} : <i1>, <i2>
    sink %falseResult_5 {handshake.name = "sink1"} : <i2>
    %87 = extsi %trueResult_4 {handshake.bb = 2 : ui32, handshake.name = "extsi23"} : <i2> to <i32>
    %trueResult_6, %falseResult_7 = cond_br %81#3, %68 {handshake.bb = 2 : ui32, handshake.name = "cond_br5"} : <i1>, <i32>
    %trueResult_8, %falseResult_9 = cond_br %81#1, %70 {handshake.bb = 2 : ui32, handshake.name = "cond_br6"} : <i1>, <i6>
    %trueResult_10, %falseResult_11 = cond_br %81#0, %66 {handshake.bb = 2 : ui32, handshake.name = "cond_br7"} : <i1>, <i6>
    sink %falseResult_11 {handshake.name = "sink2"} : <i6>
    %trueResult_12, %falseResult_13 = cond_br %81#4, %73#1 {handshake.bb = 2 : ui32, handshake.name = "cond_br8"} : <i1>, <>
    %trueResult_14, %falseResult_15 = cond_br %81#5, %48 {handshake.bb = 3 : ui32, handshake.name = "cond_br47"} : <i1>, <i32>
    %trueResult_16, %falseResult_17 = cond_br %81#6, %46 {handshake.bb = 3 : ui32, handshake.name = "cond_br48"} : <i1>, <>
    %trueResult_18, %falseResult_19 = cond_br %81#7, %51 {handshake.bb = 3 : ui32, handshake.name = "cond_br49"} : <i1>, <i32>
    %trueResult_20, %falseResult_21 = cond_br %95, %58 {handshake.bb = 3 : ui32, handshake.name = "cond_br50"} : <i1>, <>
    %95 = buffer %81#8, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer40"} : <i1>
    %trueResult_22, %falseResult_23 = cond_br %241#6, %226 {handshake.bb = 3 : ui32, handshake.name = "cond_br51"} : <i1>, <i32>
    %97:2 = fork [2] %falseResult_23 {handshake.bb = 3 : ui32, handshake.name = "fork11"} : <i32>
    %98:2 = fork [2] %trueResult_22 {handshake.bb = 3 : ui32, handshake.name = "fork12"} : <i32>
    %trueResult_24, %falseResult_25 = cond_br %81#9, %54 {handshake.bb = 3 : ui32, handshake.name = "cond_br52"} : <i1>, <>
    %trueResult_26, %falseResult_27 = cond_br %100, %229#1 {handshake.bb = 3 : ui32, handshake.name = "cond_br53"} : <i1>, <>
    %100 = buffer %241#5, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer43"} : <i1>
    %101:2 = fork [2] %falseResult_27 {handshake.bb = 3 : ui32, handshake.name = "fork13"} : <>
    %102:2 = fork [2] %trueResult_26 {handshake.bb = 3 : ui32, handshake.name = "fork14"} : <>
    %trueResult_28, %falseResult_29 = cond_br %103, %56 {handshake.bb = 3 : ui32, handshake.name = "cond_br54"} : <i1>, <>
    %103 = buffer %81#10, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer44"} : <i1>
    %trueResult_30, %falseResult_31 = cond_br %241#4, %230 {handshake.bb = 3 : ui32, handshake.name = "cond_br55"} : <i1>, <>
    %105:2 = fork [2] %falseResult_31 {handshake.bb = 3 : ui32, handshake.name = "fork15"} : <>
    %106:2 = fork [2] %trueResult_30 {handshake.bb = 3 : ui32, handshake.name = "fork16"} : <>
    %107 = mux %123#0 [%trueResult_16, %106#1] {ftd.phi, handshake.bb = 3 : ui32, handshake.name = "mux22"} : <i1>, [<>, <>] to <>
    %109 = mux %123#1 [%trueResult_14, %98#1] {ftd.phi, handshake.bb = 3 : ui32, handshake.name = "mux23"} : <i1>, [<i32>, <i32>] to <i32>
    %112 = mux %123#2 [%trueResult_18, %98#0] {ftd.phi, handshake.bb = 3 : ui32, handshake.name = "mux24"} : <i1>, [<i32>, <i32>] to <i32>
    %115 = mux %123#3 [%trueResult_24, %106#0] {ftd.phi, handshake.bb = 3 : ui32, handshake.name = "mux25"} : <i1>, [<>, <>] to <>
    %117 = mux %118 [%trueResult_28, %102#1] {ftd.phi, handshake.bb = 3 : ui32, handshake.name = "mux26"} : <i1>, [<>, <>] to <>
    %118 = buffer %123#4, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer52"} : <i1>
    %119 = mux %120 [%trueResult_20, %102#0] {ftd.phi, handshake.bb = 3 : ui32, handshake.name = "mux27"} : <i1>, [<>, <>] to <>
    %120 = buffer %123#5, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer53"} : <i1>
    %121 = init %241#3 {ftd.imerge, handshake.bb = 3 : ui32, handshake.name = "init12"} : <i1>
    %123:6 = fork [6] %121 {handshake.bb = 3 : ui32, handshake.name = "fork17"} : <i1>
    %124 = mux %146#2 [%84, %trueResult_44] {handshake.bb = 3 : ui32, handshake.name = "mux5"} : <i1>, [<i6>, <i6>] to <i6>
    %126 = extsi %124 {handshake.bb = 3 : ui32, handshake.name = "extsi24"} : <i6> to <i7>
    %127 = mux %146#3 [%87, %trueResult_46] {handshake.bb = 3 : ui32, handshake.name = "mux6"} : <i1>, [<i32>, <i32>] to <i32>
    %129:5 = fork [5] %127 {handshake.bb = 3 : ui32, handshake.name = "fork18"} : <i32>
    %130 = mux %146#4 [%trueResult_6, %trueResult_48] {handshake.bb = 3 : ui32, handshake.name = "mux7"} : <i1>, [<i32>, <i32>] to <i32>
    %132 = mux %146#0 [%trueResult_8, %trueResult_50] {handshake.bb = 3 : ui32, handshake.name = "mux8"} : <i1>, [<i6>, <i6>] to <i6>
    %134:3 = fork [3] %132 {handshake.bb = 3 : ui32, handshake.name = "fork19"} : <i6>
    %135 = extsi %134#2 {handshake.bb = 3 : ui32, handshake.name = "extsi25"} : <i6> to <i32>
    %137:2 = fork [2] %135 {handshake.bb = 3 : ui32, handshake.name = "fork20"} : <i32>
    %138 = trunci %134#0 {handshake.bb = 3 : ui32, handshake.name = "trunci1"} : <i6> to <i5>
    %140 = mux %146#1 [%trueResult_10, %trueResult_52] {handshake.bb = 3 : ui32, handshake.name = "mux9"} : <i1>, [<i6>, <i6>] to <i6>
    %142:2 = fork [2] %140 {handshake.bb = 3 : ui32, handshake.name = "fork21"} : <i6>
    %143 = extsi %142#1 {handshake.bb = 3 : ui32, handshake.name = "extsi26"} : <i6> to <i32>
    %145:4 = fork [4] %143 {handshake.bb = 3 : ui32, handshake.name = "fork22"} : <i32>
    %result_32, %index_33 = control_merge [%trueResult_12, %trueResult_54]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>, <>] to <>, <i1>
    %146:5 = fork [5] %index_33 {handshake.bb = 3 : ui32, handshake.name = "fork23"} : <i1>
    %147:2 = fork [2] %result_32 {handshake.bb = 3 : ui32, handshake.name = "fork24"} : <>
    %148 = constant %147#0 {handshake.bb = 3 : ui32, handshake.name = "constant23", value = 1 : i2} : <>, <i2>
    %149 = extsi %148 {handshake.bb = 3 : ui32, handshake.name = "extsi8"} : <i2> to <i32>
    %150 = source {handshake.bb = 3 : ui32, handshake.name = "source2"} : <>
    %151 = constant %150 {handshake.bb = 3 : ui32, handshake.name = "constant24", value = 20 : i6} : <>, <i6>
    %152 = extsi %151 {handshake.bb = 3 : ui32, handshake.name = "extsi27"} : <i6> to <i7>
    %153 = source {handshake.bb = 3 : ui32, handshake.name = "source3"} : <>
    %154 = constant %153 {handshake.bb = 3 : ui32, handshake.name = "constant25", value = 1 : i2} : <>, <i2>
    %155:2 = fork [2] %154 {handshake.bb = 3 : ui32, handshake.name = "fork25"} : <i2>
    %156 = extsi %155#0 {handshake.bb = 3 : ui32, handshake.name = "extsi28"} : <i2> to <i7>
    %158 = extsi %155#1 {handshake.bb = 3 : ui32, handshake.name = "extsi10"} : <i2> to <i32>
    %160 = source {handshake.bb = 3 : ui32, handshake.name = "source5"} : <>
    %161 = constant %160 {handshake.bb = 3 : ui32, handshake.name = "constant26", value = 4 : i4} : <>, <i4>
    %162 = extsi %161 {handshake.bb = 3 : ui32, handshake.name = "extsi12"} : <i4> to <i32>
    %163:3 = fork [3] %162 {handshake.bb = 3 : ui32, handshake.name = "fork26"} : <i32>
    %164 = source {handshake.bb = 3 : ui32, handshake.name = "source6"} : <>
    %165 = constant %164 {handshake.bb = 3 : ui32, handshake.name = "constant27", value = 2 : i3} : <>, <i3>
    %166 = extsi %165 {handshake.bb = 3 : ui32, handshake.name = "extsi13"} : <i3> to <i32>
    %167:3 = fork [3] %166 {handshake.bb = 3 : ui32, handshake.name = "fork27"} : <i32>
    %168 = shli %145#0, %167#0 {handshake.bb = 3 : ui32, handshake.name = "shli0"} : <i32>
    %171 = shli %145#1, %163#0 {handshake.bb = 3 : ui32, handshake.name = "shli1"} : <i32>
    %174 = addi %168, %171 {handshake.bb = 3 : ui32, handshake.name = "addi9"} : <i32>
    %175 = addi %129#4, %174 {handshake.bb = 3 : ui32, handshake.name = "addi3"} : <i32>
    %177:2 = fork [2] %175 {handshake.bb = 3 : ui32, handshake.name = "fork28"} : <i32>
    %178 = gate %177#1, %107 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 3 : ui32, handshake.name = "gate0"} : <i32>, !handshake.control<> to <i32>
    %180 = cmpi ne, %178, %109 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 3 : ui32, handshake.name = "cmpi3"} : <i32>
    %181:2 = fork [2] %180 {handshake.bb = 3 : ui32, handshake.name = "fork29"} : <i1>
    %trueResult_34, %falseResult_35 = cond_br %182, %117 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 3 : ui32, handshake.name = "cond_br39"} : <i1>, <>
    %182 = buffer %181#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer71"} : <i1>
    sink %trueResult_34 {handshake.name = "sink3"} : <>
    %183 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 3 : ui32, handshake.name = "source10"} : <>
    %184 = mux %181#0 [%falseResult_35, %183] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 3 : ui32, handshake.name = "mux28"} : <i1>, [<>, <>] to <>
    %186 = join %184 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 3 : ui32, handshake.name = "join0"} : <>
    %187 = gate %177#0, %186 {handshake.bb = 3 : ui32, handshake.name = "gate1"} : <i32>, !handshake.control<> to <i32>
    %189 = trunci %187 {handshake.bb = 3 : ui32, handshake.name = "trunci2"} : <i32> to <i9>
    %addressResult, %dataResult = load[%189] %outputs#0 {handshake.bb = 3 : ui32, handshake.deps = #handshake<deps[["store0", 0, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i9>, <i32>, <i9>, <i32>
    %addressResult_36, %dataResult_37 = load[%138] %outputs_0 {handshake.bb = 3 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i5>, <i32>, <i5>, <i32>
    %190 = shli %137#0, %167#1 {handshake.bb = 3 : ui32, handshake.name = "shli2"} : <i32>
    %193 = shli %137#1, %163#1 {handshake.bb = 3 : ui32, handshake.name = "shli3"} : <i32>
    %196 = addi %190, %193 {handshake.bb = 3 : ui32, handshake.name = "addi10"} : <i32>
    %197 = addi %129#3, %196 {handshake.bb = 3 : ui32, handshake.name = "addi4"} : <i32>
    %199:2 = fork [2] %197 {handshake.bb = 3 : ui32, handshake.name = "fork30"} : <i32>
    %200 = gate %199#1, %115 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 3 : ui32, handshake.name = "gate2"} : <i32>, !handshake.control<> to <i32>
    %202 = cmpi ne, %200, %112 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 3 : ui32, handshake.name = "cmpi4"} : <i32>
    %203:2 = fork [2] %202 {handshake.bb = 3 : ui32, handshake.name = "fork31"} : <i1>
    %trueResult_38, %falseResult_39 = cond_br %204, %119 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 3 : ui32, handshake.name = "cond_br40"} : <i1>, <>
    %204 = buffer %203#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer80"} : <i1>
    sink %trueResult_38 {handshake.name = "sink4"} : <>
    %205 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 3 : ui32, handshake.name = "source11"} : <>
    %206 = mux %203#0 [%falseResult_39, %205] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 3 : ui32, handshake.name = "mux29"} : <i1>, [<>, <>] to <>
    %208 = join %206 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 3 : ui32, handshake.name = "join1"} : <>
    %209 = gate %199#0, %208 {handshake.bb = 3 : ui32, handshake.name = "gate3"} : <i32>, !handshake.control<> to <i32>
    %211 = trunci %209 {handshake.bb = 3 : ui32, handshake.name = "trunci3"} : <i32> to <i9>
    %addressResult_40, %dataResult_41 = load[%211] %outputs#1 {handshake.bb = 3 : ui32, handshake.deps = #handshake<deps[["store0", 0, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load2"} : <i9>, <i32>, <i9>, <i32>
    %212 = muli %dataResult_37, %dataResult_41 {handshake.bb = 3 : ui32, handshake.name = "muli0"} : <i32>
    %213 = subi %dataResult, %212 {handshake.bb = 3 : ui32, handshake.name = "subi0"} : <i32>
    %214 = shli %145#2, %167#2 {handshake.bb = 3 : ui32, handshake.name = "shli4"} : <i32>
    %217 = shli %145#3, %163#2 {handshake.bb = 3 : ui32, handshake.name = "shli5"} : <i32>
    %220 = addi %214, %217 {handshake.bb = 3 : ui32, handshake.name = "addi11"} : <i32>
    %221 = addi %129#2, %220 {handshake.bb = 3 : ui32, handshake.name = "addi5"} : <i32>
    %223:2 = fork [2] %221 {handshake.bb = 3 : ui32, handshake.name = "fork32"} : <i32>
    %224 = trunci %225 {handshake.bb = 3 : ui32, handshake.name = "trunci4"} : <i32> to <i9>
    %225 = buffer %223#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer88"} : <i32>
    %226 = buffer %223#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer0"} : <i32>
    %228 = buffer %doneResult, bufferType = FIFO_BREAK_NONE, numSlots = 1 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 3 : ui32, handshake.name = "buffer1"} : <>
    %229:2 = fork [2] %228 {handshake.bb = 3 : ui32, handshake.name = "fork33"} : <>
    %230 = init %229#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 3 : ui32, handshake.name = "init18"} : <>
    %addressResult_42, %dataResult_43, %doneResult = store[%224] %213 %outputs#2 {handshake.bb = 3 : ui32, handshake.deps = #handshake<deps[["load0", 0, false], ["load2", 0, false], ["store0", 0, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store0"} : <i9>, <i32>, <>, <i9>, <i32>, <>
    %231 = addi %130, %129#1 {handshake.bb = 3 : ui32, handshake.name = "addi0"} : <i32>
    %233 = addi %129#0, %158 {handshake.bb = 3 : ui32, handshake.name = "addi1"} : <i32>
    %235 = addi %126, %156 {handshake.bb = 3 : ui32, handshake.name = "addi6"} : <i7>
    %236:2 = fork [2] %235 {handshake.bb = 3 : ui32, handshake.name = "fork34"} : <i7>
    %237 = trunci %236#0 {handshake.bb = 3 : ui32, handshake.name = "trunci5"} : <i7> to <i6>
    %239 = cmpi ult, %236#1, %152 {handshake.bb = 3 : ui32, handshake.name = "cmpi0"} : <i7>
    %241:10 = fork [10] %239 {handshake.bb = 3 : ui32, handshake.name = "fork35"} : <i1>
    %trueResult_44, %falseResult_45 = cond_br %241#0, %237 {handshake.bb = 3 : ui32, handshake.name = "cond_br9"} : <i1>, <i6>
    sink %falseResult_45 {handshake.name = "sink5"} : <i6>
    %trueResult_46, %falseResult_47 = cond_br %241#7, %233 {handshake.bb = 3 : ui32, handshake.name = "cond_br10"} : <i1>, <i32>
    sink %falseResult_47 {handshake.name = "sink6"} : <i32>
    %trueResult_48, %falseResult_49 = cond_br %241#8, %231 {handshake.bb = 3 : ui32, handshake.name = "cond_br11"} : <i1>, <i32>
    %trueResult_50, %falseResult_51 = cond_br %241#1, %134#1 {handshake.bb = 3 : ui32, handshake.name = "cond_br12"} : <i1>, <i6>
    %trueResult_52, %falseResult_53 = cond_br %241#2, %142#0 {handshake.bb = 3 : ui32, handshake.name = "cond_br13"} : <i1>, <i6>
    %trueResult_54, %falseResult_55 = cond_br %241#9, %147#1 {handshake.bb = 3 : ui32, handshake.name = "cond_br14"} : <i1>, <>
    %250 = merge %falseResult_51 {handshake.bb = 4 : ui32, handshake.name = "merge0"} : <i6>
    %251 = merge %falseResult_53 {handshake.bb = 4 : ui32, handshake.name = "merge1"} : <i6>
    %252 = extsi %251 {handshake.bb = 4 : ui32, handshake.name = "extsi29"} : <i6> to <i7>
    %253 = merge %falseResult_49 {handshake.bb = 4 : ui32, handshake.name = "merge2"} : <i32>
    %result_56, %index_57 = control_merge [%falseResult_55]  {handshake.bb = 4 : ui32, handshake.name = "control_merge3"} : [<>] to <>, <i1>
    sink %index_57 {handshake.name = "sink7"} : <i1>
    %254 = source {handshake.bb = 4 : ui32, handshake.name = "source7"} : <>
    %255 = constant %254 {handshake.bb = 4 : ui32, handshake.name = "constant28", value = 1 : i2} : <>, <i2>
    %256 = extsi %255 {handshake.bb = 4 : ui32, handshake.name = "extsi30"} : <i2> to <i7>
    %257 = addi %252, %256 {handshake.bb = 4 : ui32, handshake.name = "addi8"} : <i7>
    %258 = br %257 {handshake.bb = 4 : ui32, handshake.name = "br10"} : <i7>
    %259 = br %253 {handshake.bb = 4 : ui32, handshake.name = "br11"} : <i32>
    %260 = br %250 {handshake.bb = 4 : ui32, handshake.name = "br12"} : <i6>
    %261 = br %result_56 {handshake.bb = 4 : ui32, handshake.name = "br13"} : <>
    %trueResult_58, %falseResult_59 = cond_br %283#6, %falseResult_15 {handshake.bb = 5 : ui32, handshake.name = "cond_br56"} : <i1>, <i32>
    sink %falseResult_59 {handshake.name = "sink8"} : <i32>
    %trueResult_60, %falseResult_61 = cond_br %283#5, %falseResult_19 {handshake.bb = 5 : ui32, handshake.name = "cond_br57"} : <i1>, <i32>
    sink %falseResult_61 {handshake.name = "sink9"} : <i32>
    %trueResult_62, %falseResult_63 = cond_br %283#4, %falseResult_17 {handshake.bb = 5 : ui32, handshake.name = "cond_br58"} : <i1>, <>
    sink %falseResult_63 {handshake.name = "sink10"} : <>
    %trueResult_64, %falseResult_65 = cond_br %283#3, %falseResult_21 {handshake.bb = 5 : ui32, handshake.name = "cond_br59"} : <i1>, <>
    sink %falseResult_65 {handshake.name = "sink11"} : <>
    %trueResult_66, %falseResult_67 = cond_br %283#2, %falseResult_29 {handshake.bb = 5 : ui32, handshake.name = "cond_br60"} : <i1>, <>
    sink %falseResult_67 {handshake.name = "sink12"} : <>
    %trueResult_68, %falseResult_69 = cond_br %283#1, %falseResult_25 {handshake.bb = 5 : ui32, handshake.name = "cond_br61"} : <i1>, <>
    sink %falseResult_69 {handshake.name = "sink13"} : <>
    %268 = merge %falseResult_9 {handshake.bb = 5 : ui32, handshake.name = "merge3"} : <i6>
    %269 = extsi %268 {handshake.bb = 5 : ui32, handshake.name = "extsi31"} : <i6> to <i7>
    %270 = merge %falseResult_7 {handshake.bb = 5 : ui32, handshake.name = "merge4"} : <i32>
    %result_70, %index_71 = control_merge [%falseResult_13]  {handshake.bb = 5 : ui32, handshake.name = "control_merge4"} : [<>] to <>, <i1>
    sink %index_71 {handshake.name = "sink14"} : <i1>
    %271 = source {handshake.bb = 5 : ui32, handshake.name = "source8"} : <>
    %272 = constant %271 {handshake.bb = 5 : ui32, handshake.name = "constant29", value = 19 : i6} : <>, <i6>
    %273 = extsi %272 {handshake.bb = 5 : ui32, handshake.name = "extsi32"} : <i6> to <i7>
    %274 = source {handshake.bb = 5 : ui32, handshake.name = "source9"} : <>
    %275 = constant %274 {handshake.bb = 5 : ui32, handshake.name = "constant30", value = 1 : i2} : <>, <i2>
    %276 = extsi %275 {handshake.bb = 5 : ui32, handshake.name = "extsi33"} : <i2> to <i7>
    %277 = addi %269, %276 {handshake.bb = 5 : ui32, handshake.name = "addi7"} : <i7>
    %278:2 = fork [2] %277 {handshake.bb = 5 : ui32, handshake.name = "fork36"} : <i7>
    %279 = trunci %278#0 {handshake.bb = 5 : ui32, handshake.name = "trunci6"} : <i7> to <i6>
    %281 = cmpi ult, %278#1, %273 {handshake.bb = 5 : ui32, handshake.name = "cmpi1"} : <i7>
    %283:10 = fork [10] %281 {handshake.bb = 5 : ui32, handshake.name = "fork37"} : <i1>
    %trueResult_72, %falseResult_73 = cond_br %283#0, %279 {handshake.bb = 5 : ui32, handshake.name = "cond_br15"} : <i1>, <i6>
    sink %falseResult_73 {handshake.name = "sink15"} : <i6>
    %trueResult_74, %falseResult_75 = cond_br %283#8, %270 {handshake.bb = 5 : ui32, handshake.name = "cond_br16"} : <i1>, <i32>
    %trueResult_76, %falseResult_77 = cond_br %283#9, %result_70 {handshake.bb = 5 : ui32, handshake.name = "cond_br17"} : <i1>, <>
    %287 = merge %falseResult_75 {handshake.bb = 6 : ui32, handshake.name = "merge5"} : <i32>
    %result_78, %index_79 = control_merge [%falseResult_77]  {handshake.bb = 6 : ui32, handshake.name = "control_merge5"} : [<>] to <>, <i1>
    sink %index_79 {handshake.name = "sink16"} : <i1>
    %288:2 = fork [2] %result_78 {handshake.bb = 6 : ui32, handshake.name = "fork38"} : <>
    end {handshake.bb = 6 : ui32, handshake.name = "end0"} %287, %memEnd_1, %memEnd, %0#3 : <i32>, <>, <>, <>
  }
}

