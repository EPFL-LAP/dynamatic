module {
  handshake.func @matrix_power(%arg0: memref<400xi32>, %arg1: memref<20xi32>, %arg2: memref<20xi32>, %arg3: memref<20xi32>, %arg4: !handshake.control<>, %arg5: !handshake.control<>, %arg6: !handshake.control<>, %arg7: !handshake.control<>, %arg8: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["mat", "row", "col", "a", "mat_start", "row_start", "col_start", "a_start", "start"], cfg.edges = "[0,1][2,2,3,cmpi0][1,2][3,1,4,cmpi1]", resNames = ["mat_end", "row_end", "col_end", "a_end", "end"]} {
    %0:14 = fork [14] %arg8 {handshake.bb = 0 : ui32, handshake.name = "fork0"} : <>
    %outputs, %memEnd = mem_controller[%arg3 : memref<20xi32>] %arg7 (%addressResult_24) %363#3 {connectedBlocks = [2 : i32], handshake.name = "mem_controller3"} :    (!handshake.channel<i5>) -> !handshake.channel<i32>
    %outputs_0, %memEnd_1 = mem_controller[%arg2 : memref<20xi32>] %arg6 (%addressResult_26) %363#2 {connectedBlocks = [2 : i32], handshake.name = "mem_controller4"} :    (!handshake.channel<i5>) -> !handshake.channel<i32>
    %outputs_2, %memEnd_3 = mem_controller[%arg1 : memref<20xi32>] %arg5 (%addressResult) %363#1 {connectedBlocks = [2 : i32], handshake.name = "mem_controller5"} :    (!handshake.channel<i5>) -> !handshake.channel<i32>
    %outputs_4:3, %memEnd_5 = mem_controller[%arg0 : memref<400xi32>] %arg4 (%176, %addressResult_36, %addressResult_46, %addressResult_48, %dataResult_49) %363#0 {connectedBlocks = [2 : i32], handshake.name = "mem_controller6"} :    (!handshake.channel<i32>, !handshake.channel<i9>, !handshake.channel<i9>, !handshake.channel<i9>, !handshake.channel<i32>) -> (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>)
    %1 = constant %0#1 {handshake.bb = 0 : ui32, handshake.name = "constant18", value = 1000 : i11} : <>, <i11>
    %2:8 = fork [8] %1 {handshake.bb = 0 : ui32, handshake.name = "fork1"} : <i11>
    %3 = extsi %2#0 {handshake.bb = 1 : ui32, handshake.name = "extsi0"} : <i11> to <i32>
    %5 = extsi %2#1 {handshake.bb = 1 : ui32, handshake.name = "extsi1"} : <i11> to <i32>
    %7 = extsi %2#2 {handshake.bb = 1 : ui32, handshake.name = "extsi2"} : <i11> to <i32>
    %9 = extsi %2#3 {handshake.bb = 1 : ui32, handshake.name = "extsi3"} : <i11> to <i32>
    %11 = extsi %2#4 {handshake.bb = 1 : ui32, handshake.name = "extsi4"} : <i11> to <i32>
    %13 = extsi %2#5 {handshake.bb = 1 : ui32, handshake.name = "extsi5"} : <i11> to <i32>
    %15 = extsi %2#6 {handshake.bb = 1 : ui32, handshake.name = "extsi6"} : <i11> to <i32>
    %17 = extsi %2#7 {handshake.bb = 1 : ui32, handshake.name = "extsi7"} : <i11> to <i32>
    %19 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant19", value = 1 : i2} : <>, <i2>
    %20 = br %19 {handshake.bb = 0 : ui32, handshake.name = "br2"} : <i2>
    %21 = extsi %20 {handshake.bb = 0 : ui32, handshake.name = "extsi18"} : <i2> to <i6>
    %22 = br %0#13 {handshake.bb = 0 : ui32, handshake.name = "br3"} : <>
    %23 = mux %69#0 [%3, %341#0] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux4"} : <i1>, [<i32>, <i32>] to <i32>
    %26 = mux %69#1 [%0#12, %329#0] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux5"} : <i1>, [<>, <>] to <>
    %28 = mux %69#2 [%0#11, %329#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux6"} : <i1>, [<>, <>] to <>
    %30 = mux %69#3 [%5, %337#0] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux7"} : <i1>, [<i32>, <i32>] to <i32>
    %33 = mux %69#4 [%7, %345#0] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux8"} : <i1>, [<i32>, <i32>] to <i32>
    %36 = mux %69#5 [%9, %345#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux9"} : <i1>, [<i32>, <i32>] to <i32>
    %39 = mux %69#6 [%11, %337#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux10"} : <i1>, [<i32>, <i32>] to <i32>
    %42 = mux %69#7 [%13, %341#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux11"} : <i1>, [<i32>, <i32>] to <i32>
    %45 = mux %69#8 [%15, %339#0] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux12"} : <i1>, [<i32>, <i32>] to <i32>
    %48 = mux %69#9 [%17, %339#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux13"} : <i1>, [<i32>, <i32>] to <i32>
    %51 = mux %69#10 [%0#10, %343#0] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux14"} : <i1>, [<>, <>] to <>
    %53 = mux %69#11 [%0#9, %333#0] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux15"} : <i1>, [<>, <>] to <>
    %55 = mux %69#12 [%0#8, %335#0] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux16"} : <i1>, [<>, <>] to <>
    %57 = mux %69#13 [%0#7, %331#0] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux17"} : <i1>, [<>, <>] to <>
    %59 = mux %69#14 [%0#6, %333#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux18"} : <i1>, [<>, <>] to <>
    %61 = mux %69#15 [%0#5, %331#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux19"} : <i1>, [<>, <>] to <>
    %63 = mux %69#16 [%0#4, %343#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux20"} : <i1>, [<>, <>] to <>
    %65 = mux %69#17 [%0#3, %335#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux21"} : <i1>, [<>, <>] to <>
    %67 = init %360#10 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "init0"} : <i1>
    %69:18 = fork [18] %67 {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <i1>
    %70 = mux %index [%21, %trueResult_78] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i6>, <i6>] to <i6>
    %71:2 = fork [2] %70 {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <i6>
    %72 = extsi %71#1 {handshake.bb = 1 : ui32, handshake.name = "extsi19"} : <i6> to <i32>
    %result, %index = control_merge [%22, %trueResult_80]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %74:2 = fork [2] %result {handshake.bb = 1 : ui32, handshake.name = "fork4"} : <>
    %75 = constant %74#0 {handshake.bb = 1 : ui32, handshake.name = "constant20", value = false} : <>, <i1>
    %76 = source {handshake.bb = 1 : ui32, handshake.name = "source0"} : <>
    %77 = constant %76 {handshake.bb = 1 : ui32, handshake.name = "constant7", value = -1 : i32} : <>, <i32>
    %78 = addi %72, %77 {handshake.bb = 1 : ui32, handshake.name = "addi0"} : <i32>
    %79 = br %75 {handshake.bb = 1 : ui32, handshake.name = "br4"} : <i1>
    %80 = extsi %79 {handshake.bb = 1 : ui32, handshake.name = "extsi17"} : <i1> to <i6>
    %81 = br %71#0 {handshake.bb = 1 : ui32, handshake.name = "br5"} : <i6>
    %83 = br %78 {handshake.bb = 1 : ui32, handshake.name = "br6"} : <i32>
    %84 = br %74#1 {handshake.bb = 1 : ui32, handshake.name = "br7"} : <>
    %trueResult, %falseResult = cond_br %321#11, %313#1 {handshake.bb = 2 : ui32, handshake.name = "cond_br68"} : <i1>, <>
    %86:2 = fork [2] %trueResult {handshake.bb = 2 : ui32, handshake.name = "fork5"} : <>
    %trueResult_6, %falseResult_7 = cond_br %321#10, %314 {handshake.bb = 2 : ui32, handshake.name = "cond_br69"} : <i1>, <>
    %88:2 = fork [2] %trueResult_6 {handshake.bb = 2 : ui32, handshake.name = "fork6"} : <>
    %trueResult_8, %falseResult_9 = cond_br %321#9, %90 {handshake.bb = 2 : ui32, handshake.name = "cond_br70"} : <i1>, <i32>
    %90 = buffer %297#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer42"} : <i32>
    %91:2 = fork [2] %trueResult_8 {handshake.bb = 2 : ui32, handshake.name = "fork7"} : <i32>
    %trueResult_10, %falseResult_11 = cond_br %92, %311#1 {handshake.bb = 2 : ui32, handshake.name = "cond_br71"} : <i1>, <>
    %92 = buffer %321#8, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer43"} : <i1>
    %93:2 = fork [2] %trueResult_10 {handshake.bb = 2 : ui32, handshake.name = "fork8"} : <>
    %trueResult_12, %falseResult_13 = cond_br %321#7, %304 {handshake.bb = 2 : ui32, handshake.name = "cond_br72"} : <i1>, <i32>
    %95:2 = fork [2] %trueResult_12 {handshake.bb = 2 : ui32, handshake.name = "fork9"} : <i32>
    %trueResult_14, %falseResult_15 = cond_br %96, %307#1 {handshake.bb = 2 : ui32, handshake.name = "cond_br73"} : <i1>, <>
    %96 = buffer %321#6, bufferType = FIFO_BREAK_NONE, numSlots = 3 {handshake.bb = 2 : ui32, handshake.name = "buffer45"} : <i1>
    %97:2 = fork [2] %trueResult_14 {handshake.bb = 2 : ui32, handshake.name = "fork10"} : <>
    %trueResult_16, %falseResult_17 = cond_br %98, %309#1 {handshake.bb = 2 : ui32, handshake.name = "cond_br74"} : <i1>, <>
    %98 = buffer %321#5, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 2 : ui32, handshake.name = "buffer46"} : <i1>
    %99:2 = fork [2] %trueResult_16 {handshake.bb = 2 : ui32, handshake.name = "fork11"} : <>
    %trueResult_18, %falseResult_19 = cond_br %321#4, %303#1 {handshake.bb = 2 : ui32, handshake.name = "cond_br75"} : <i1>, <i32>
    %102:2 = fork [2] %trueResult_18 {handshake.bb = 2 : ui32, handshake.name = "fork12"} : <i32>
    %trueResult_20, %falseResult_21 = cond_br %321#3, %300#1 {handshake.bb = 2 : ui32, handshake.name = "cond_br76"} : <i1>, <i32>
    %105:2 = fork [2] %trueResult_20 {handshake.bb = 2 : ui32, handshake.name = "fork13"} : <i32>
    %106 = mux %152#0 [%23, %91#1] {ftd.phi, handshake.bb = 2 : ui32, handshake.name = "mux22"} : <i1>, [<i32>, <i32>] to <i32>
    %109 = mux %152#1 [%26, %88#1] {ftd.phi, handshake.bb = 2 : ui32, handshake.name = "mux23"} : <i1>, [<>, <>] to <>
    %111 = mux %152#2 [%28, %88#0] {ftd.phi, handshake.bb = 2 : ui32, handshake.name = "mux24"} : <i1>, [<>, <>] to <>
    %113 = mux %152#3 [%30, %102#1] {ftd.phi, handshake.bb = 2 : ui32, handshake.name = "mux25"} : <i1>, [<i32>, <i32>] to <i32>
    %116 = mux %152#4 [%33, %105#1] {ftd.phi, handshake.bb = 2 : ui32, handshake.name = "mux26"} : <i1>, [<i32>, <i32>] to <i32>
    %119 = mux %152#5 [%36, %105#0] {ftd.phi, handshake.bb = 2 : ui32, handshake.name = "mux27"} : <i1>, [<i32>, <i32>] to <i32>
    %122 = mux %152#6 [%39, %102#0] {ftd.phi, handshake.bb = 2 : ui32, handshake.name = "mux28"} : <i1>, [<i32>, <i32>] to <i32>
    %125 = mux %152#7 [%42, %91#0] {ftd.phi, handshake.bb = 2 : ui32, handshake.name = "mux29"} : <i1>, [<i32>, <i32>] to <i32>
    %128 = mux %152#8 [%45, %95#1] {ftd.phi, handshake.bb = 2 : ui32, handshake.name = "mux30"} : <i1>, [<i32>, <i32>] to <i32>
    %131 = mux %152#9 [%48, %95#0] {ftd.phi, handshake.bb = 2 : ui32, handshake.name = "mux31"} : <i1>, [<i32>, <i32>] to <i32>
    %134 = mux %135 [%51, %99#1] {ftd.phi, handshake.bb = 2 : ui32, handshake.name = "mux32"} : <i1>, [<>, <>] to <>
    %135 = buffer %152#10, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer69"} : <i1>
    %136 = mux %152#11 [%53, %86#1] {ftd.phi, handshake.bb = 2 : ui32, handshake.name = "mux33"} : <i1>, [<>, <>] to <>
    %138 = mux %139 [%55, %97#1] {ftd.phi, handshake.bb = 2 : ui32, handshake.name = "mux34"} : <i1>, [<>, <>] to <>
    %139 = buffer %152#12, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 2 : ui32, handshake.name = "buffer71"} : <i1>
    %140 = mux %152#13 [%57, %93#1] {ftd.phi, handshake.bb = 2 : ui32, handshake.name = "mux35"} : <i1>, [<>, <>] to <>
    %142 = mux %152#14 [%59, %86#0] {ftd.phi, handshake.bb = 2 : ui32, handshake.name = "mux36"} : <i1>, [<>, <>] to <>
    %144 = mux %152#15 [%61, %93#0] {ftd.phi, handshake.bb = 2 : ui32, handshake.name = "mux37"} : <i1>, [<>, <>] to <>
    %146 = mux %147 [%63, %99#0] {ftd.phi, handshake.bb = 2 : ui32, handshake.name = "mux38"} : <i1>, [<>, <>] to <>
    %147 = buffer %152#16, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer75"} : <i1>
    %148 = mux %149 [%65, %97#0] {ftd.phi, handshake.bb = 2 : ui32, handshake.name = "mux39"} : <i1>, [<>, <>] to <>
    %149 = buffer %152#17, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 2 : ui32, handshake.name = "buffer76"} : <i1>
    %150 = init %321#2 {ftd.imerge, handshake.bb = 2 : ui32, handshake.name = "init18"} : <i1>
    %152:18 = fork [18] %150 {handshake.bb = 2 : ui32, handshake.name = "fork14"} : <i1>
    %153 = mux %173#1 [%80, %trueResult_50] {handshake.bb = 2 : ui32, handshake.name = "mux1"} : <i1>, [<i6>, <i6>] to <i6>
    %155:4 = fork [4] %153 {handshake.bb = 2 : ui32, handshake.name = "fork15"} : <i6>
    %156 = extsi %155#3 {handshake.bb = 2 : ui32, handshake.name = "extsi20"} : <i6> to <i7>
    %158 = trunci %155#0 {handshake.bb = 2 : ui32, handshake.name = "trunci0"} : <i6> to <i5>
    %160 = trunci %155#1 {handshake.bb = 2 : ui32, handshake.name = "trunci1"} : <i6> to <i5>
    %162 = trunci %155#2 {handshake.bb = 2 : ui32, handshake.name = "trunci2"} : <i6> to <i5>
    %164 = mux %173#0 [%81, %trueResult_52] {handshake.bb = 2 : ui32, handshake.name = "mux2"} : <i1>, [<i6>, <i6>] to <i6>
    %166:2 = fork [2] %164 {handshake.bb = 2 : ui32, handshake.name = "fork16"} : <i6>
    %167 = extsi %166#1 {handshake.bb = 2 : ui32, handshake.name = "extsi21"} : <i6> to <i32>
    %169:4 = fork [4] %167 {handshake.bb = 2 : ui32, handshake.name = "fork17"} : <i32>
    %170 = mux %173#2 [%83, %trueResult_54] {handshake.bb = 2 : ui32, handshake.name = "mux3"} : <i1>, [<i32>, <i32>] to <i32>
    %172:3 = fork [3] %170 {handshake.bb = 2 : ui32, handshake.name = "fork18"} : <i32>
    %result_22, %index_23 = control_merge [%84, %trueResult_56]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>, <>] to <>, <i1>
    %173:3 = fork [3] %index_23 {handshake.bb = 2 : ui32, handshake.name = "fork19"} : <i1>
    %174:2 = fork [2] %result_22 {handshake.bb = 2 : ui32, handshake.name = "fork20"} : <>
    %175 = constant %174#0 {handshake.bb = 2 : ui32, handshake.name = "constant21", value = 1 : i2} : <>, <i2>
    %176 = extsi %175 {handshake.bb = 2 : ui32, handshake.name = "extsi10"} : <i2> to <i32>
    %177 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %178 = constant %177 {handshake.bb = 2 : ui32, handshake.name = "constant22", value = 1 : i2} : <>, <i2>
    %179 = extsi %178 {handshake.bb = 2 : ui32, handshake.name = "extsi22"} : <i2> to <i7>
    %180 = source {handshake.bb = 2 : ui32, handshake.name = "source2"} : <>
    %181 = constant %180 {handshake.bb = 2 : ui32, handshake.name = "constant23", value = 20 : i6} : <>, <i6>
    %182 = extsi %181 {handshake.bb = 2 : ui32, handshake.name = "extsi23"} : <i6> to <i7>
    %183 = source {handshake.bb = 2 : ui32, handshake.name = "source3"} : <>
    %184 = constant %183 {handshake.bb = 2 : ui32, handshake.name = "constant24", value = 4 : i4} : <>, <i4>
    %185 = extsi %184 {handshake.bb = 2 : ui32, handshake.name = "extsi13"} : <i4> to <i32>
    %186:3 = fork [3] %185 {handshake.bb = 2 : ui32, handshake.name = "fork21"} : <i32>
    %187 = source {handshake.bb = 2 : ui32, handshake.name = "source4"} : <>
    %188 = constant %187 {handshake.bb = 2 : ui32, handshake.name = "constant25", value = 2 : i3} : <>, <i3>
    %189 = extsi %188 {handshake.bb = 2 : ui32, handshake.name = "extsi14"} : <i3> to <i32>
    %190:3 = fork [3] %189 {handshake.bb = 2 : ui32, handshake.name = "fork22"} : <i32>
    %addressResult, %dataResult = load[%162] %outputs_2 {handshake.bb = 2 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i5>, <i32>, <i5>, <i32>
    %191:2 = fork [2] %dataResult {handshake.bb = 2 : ui32, handshake.name = "fork23"} : <i32>
    %addressResult_24, %dataResult_25 = load[%160] %outputs {handshake.bb = 2 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i5>, <i32>, <i5>, <i32>
    %addressResult_26, %dataResult_27 = load[%158] %outputs_0 {handshake.bb = 2 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load2"} : <i5>, <i32>, <i5>, <i32>
    %192 = shli %172#2, %190#0 {handshake.bb = 2 : ui32, handshake.name = "shli0"} : <i32>
    %195 = shli %172#1, %186#0 {handshake.bb = 2 : ui32, handshake.name = "shli1"} : <i32>
    %198 = addi %192, %195 {handshake.bb = 2 : ui32, handshake.name = "addi7"} : <i32>
    %199 = addi %dataResult_27, %198 {handshake.bb = 2 : ui32, handshake.name = "addi2"} : <i32>
    %200:2 = fork [2] %199 {handshake.bb = 2 : ui32, handshake.name = "fork24"} : <i32>
    %201 = gate %200#1, %111 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "gate0"} : <i32>, !handshake.control<> to <i32>
    %203:4 = fork [4] %201 {handshake.bb = 2 : ui32, handshake.name = "fork25"} : <i32>
    %204 = cmpi ne, %203#3, %106 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi2"} : <i32>
    %206:2 = fork [2] %204 {handshake.bb = 2 : ui32, handshake.name = "fork26"} : <i1>
    %207 = cmpi ne, %203#2, %119 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi3"} : <i32>
    %209:2 = fork [2] %207 {handshake.bb = 2 : ui32, handshake.name = "fork27"} : <i1>
    %210 = cmpi ne, %203#1, %113 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi4"} : <i32>
    %212:2 = fork [2] %210 {handshake.bb = 2 : ui32, handshake.name = "fork28"} : <i1>
    %213 = cmpi ne, %203#0, %128 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi5"} : <i32>
    %215:2 = fork [2] %213 {handshake.bb = 2 : ui32, handshake.name = "fork29"} : <i1>
    %trueResult_28, %falseResult_29 = cond_br %216, %148 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br42"} : <i1>, <>
    %216 = buffer %206#1, bufferType = FIFO_BREAK_NONE, numSlots = 3 {handshake.bb = 2 : ui32, handshake.name = "buffer95"} : <i1>
    sink %trueResult_28 {handshake.name = "sink0"} : <>
    %trueResult_30, %falseResult_31 = cond_br %217, %146 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br43"} : <i1>, <>
    %217 = buffer %209#1, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 2 : ui32, handshake.name = "buffer96"} : <i1>
    sink %trueResult_30 {handshake.name = "sink1"} : <>
    %trueResult_32, %falseResult_33 = cond_br %218, %140 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br44"} : <i1>, <>
    %218 = buffer %212#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer97"} : <i1>
    sink %trueResult_32 {handshake.name = "sink2"} : <>
    %trueResult_34, %falseResult_35 = cond_br %215#1, %136 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br45"} : <i1>, <>
    sink %trueResult_34 {handshake.name = "sink3"} : <>
    %220 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source7"} : <>
    %221 = mux %206#0 [%falseResult_29, %220] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux40"} : <i1>, [<>, <>] to <>
    %223 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source8"} : <>
    %224 = mux %209#0 [%falseResult_31, %223] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux41"} : <i1>, [<>, <>] to <>
    %226 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source9"} : <>
    %227 = mux %212#0 [%falseResult_33, %226] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux42"} : <i1>, [<>, <>] to <>
    %229 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source10"} : <>
    %230 = mux %215#0 [%falseResult_35, %229] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux43"} : <i1>, [<>, <>] to <>
    %232 = join %221, %224, %227, %230 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "join0"} : <>
    %233 = gate %200#0, %232 {handshake.bb = 2 : ui32, handshake.name = "gate1"} : <i32>, !handshake.control<> to <i32>
    %235 = trunci %233 {handshake.bb = 2 : ui32, handshake.name = "trunci3"} : <i32> to <i9>
    %addressResult_36, %dataResult_37 = load[%235] %outputs_4#0 {handshake.bb = 2 : ui32, handshake.deps = #handshake<deps[["store0", 0, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load3"} : <i9>, <i32>, <i9>, <i32>
    %236 = muli %dataResult_25, %dataResult_37 {handshake.bb = 2 : ui32, handshake.name = "muli0"} : <i32>
    %237 = shli %169#0, %190#1 {handshake.bb = 2 : ui32, handshake.name = "shli2"} : <i32>
    %240 = shli %169#1, %186#1 {handshake.bb = 2 : ui32, handshake.name = "shli3"} : <i32>
    %243 = addi %237, %240 {handshake.bb = 2 : ui32, handshake.name = "addi8"} : <i32>
    %244 = addi %191#0, %243 {handshake.bb = 2 : ui32, handshake.name = "addi3"} : <i32>
    %246:2 = fork [2] %244 {handshake.bb = 2 : ui32, handshake.name = "fork30"} : <i32>
    %247 = gate %246#1, %109 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "gate2"} : <i32>, !handshake.control<> to <i32>
    %249:4 = fork [4] %247 {handshake.bb = 2 : ui32, handshake.name = "fork31"} : <i32>
    %250 = cmpi ne, %249#3, %125 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi6"} : <i32>
    %252:2 = fork [2] %250 {handshake.bb = 2 : ui32, handshake.name = "fork32"} : <i1>
    %253 = cmpi ne, %249#2, %116 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi7"} : <i32>
    %255:2 = fork [2] %253 {handshake.bb = 2 : ui32, handshake.name = "fork33"} : <i1>
    %256 = cmpi ne, %249#1, %122 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi8"} : <i32>
    %258:2 = fork [2] %256 {handshake.bb = 2 : ui32, handshake.name = "fork34"} : <i1>
    %259 = cmpi ne, %249#0, %131 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi9"} : <i32>
    %261:2 = fork [2] %259 {handshake.bb = 2 : ui32, handshake.name = "fork35"} : <i1>
    %trueResult_38, %falseResult_39 = cond_br %262, %138 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br46"} : <i1>, <>
    %262 = buffer %252#1, bufferType = FIFO_BREAK_NONE, numSlots = 3 {handshake.bb = 2 : ui32, handshake.name = "buffer114"} : <i1>
    sink %trueResult_38 {handshake.name = "sink4"} : <>
    %trueResult_40, %falseResult_41 = cond_br %263, %134 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br47"} : <i1>, <>
    %263 = buffer %255#1, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 2 : ui32, handshake.name = "buffer115"} : <i1>
    sink %trueResult_40 {handshake.name = "sink5"} : <>
    %trueResult_42, %falseResult_43 = cond_br %264, %144 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br48"} : <i1>, <>
    %264 = buffer %258#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer116"} : <i1>
    sink %trueResult_42 {handshake.name = "sink6"} : <>
    %trueResult_44, %falseResult_45 = cond_br %265, %142 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br49"} : <i1>, <>
    %265 = buffer %261#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer117"} : <i1>
    sink %trueResult_44 {handshake.name = "sink7"} : <>
    %266 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source11"} : <>
    %267 = mux %268 [%falseResult_39, %266] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux44"} : <i1>, [<>, <>] to <>
    %268 = buffer %252#0, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 2 : ui32, handshake.name = "buffer118"} : <i1>
    %269 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source12"} : <>
    %270 = mux %271 [%falseResult_41, %269] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux45"} : <i1>, [<>, <>] to <>
    %271 = buffer %255#0, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 2 : ui32, handshake.name = "buffer119"} : <i1>
    %272 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source13"} : <>
    %273 = mux %274 [%falseResult_43, %272] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux46"} : <i1>, [<>, <>] to <>
    %274 = buffer %258#0, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 2 : ui32, handshake.name = "buffer120"} : <i1>
    %275 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source14"} : <>
    %276 = mux %277 [%falseResult_45, %275] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux47"} : <i1>, [<>, <>] to <>
    %277 = buffer %261#0, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 2 : ui32, handshake.name = "buffer121"} : <i1>
    %278 = join %267, %270, %273, %276 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "join1"} : <>
    %279 = gate %280, %278 {handshake.bb = 2 : ui32, handshake.name = "gate3"} : <i32>, !handshake.control<> to <i32>
    %280 = buffer %246#0, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 2 : ui32, handshake.name = "buffer122"} : <i32>
    %281 = trunci %279 {handshake.bb = 2 : ui32, handshake.name = "trunci4"} : <i32> to <i9>
    %addressResult_46, %dataResult_47 = load[%281] %outputs_4#1 {handshake.bb = 2 : ui32, handshake.deps = #handshake<deps[["store0", 0, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load4"} : <i9>, <i32>, <i9>, <i32>
    %282 = addi %dataResult_47, %236 {handshake.bb = 2 : ui32, handshake.name = "addi1"} : <i32>
    %283 = shli %169#2, %190#2 {handshake.bb = 2 : ui32, handshake.name = "shli4"} : <i32>
    %286 = shli %288, %186#2 {handshake.bb = 2 : ui32, handshake.name = "shli5"} : <i32>
    %288 = buffer %169#3, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer126"} : <i32>
    %289 = addi %283, %286 {handshake.bb = 2 : ui32, handshake.name = "addi9"} : <i32>
    %290 = addi %291, %289 {handshake.bb = 2 : ui32, handshake.name = "addi4"} : <i32>
    %291 = buffer %191#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer127"} : <i32>
    %292:2 = fork [2] %290 {handshake.bb = 2 : ui32, handshake.name = "fork36"} : <i32>
    %293 = trunci %294 {handshake.bb = 2 : ui32, handshake.name = "trunci5"} : <i32> to <i9>
    %294 = buffer %292#0, bufferType = FIFO_BREAK_NONE, numSlots = 3 {handshake.bb = 2 : ui32, handshake.name = "buffer128"} : <i32>
    %295 = buffer %292#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer0"} : <i32>
    %297:2 = fork [2] %295 {handshake.bb = 2 : ui32, handshake.name = "fork37"} : <i32>
    %298 = init %297#0 {handshake.bb = 2 : ui32, handshake.name = "init36"} : <i32>
    %300:2 = fork [2] %298 {handshake.bb = 2 : ui32, handshake.name = "fork38"} : <i32>
    %301 = init %302 {handshake.bb = 2 : ui32, handshake.name = "init37"} : <i32>
    %302 = buffer %300#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer131"} : <i32>
    %303:2 = fork [2] %301 {handshake.bb = 2 : ui32, handshake.name = "fork39"} : <i32>
    %304 = init %303#0 {handshake.bb = 2 : ui32, handshake.name = "init38"} : <i32>
    %306 = buffer %doneResult, bufferType = FIFO_BREAK_NONE, numSlots = 1 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "buffer1"} : <>
    %307:2 = fork [2] %306 {handshake.bb = 2 : ui32, handshake.name = "fork40"} : <>
    %308 = init %307#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init39"} : <>
    %309:2 = fork [2] %308 {handshake.bb = 2 : ui32, handshake.name = "fork41"} : <>
    %310 = init %309#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init40"} : <>
    %311:2 = fork [2] %310 {handshake.bb = 2 : ui32, handshake.name = "fork42"} : <>
    %312 = init %311#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init41"} : <>
    %313:2 = fork [2] %312 {handshake.bb = 2 : ui32, handshake.name = "fork43"} : <>
    %314 = init %313#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init42"} : <>
    %addressResult_48, %dataResult_49, %doneResult = store[%293] %282 %outputs_4#2 {handshake.bb = 2 : ui32, handshake.deps = #handshake<deps[["load3", 0, false], ["load4", 0, false], ["store0", 0, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store0"} : <i9>, <i32>, <>, <i9>, <i32>, <>
    %315 = addi %156, %179 {handshake.bb = 2 : ui32, handshake.name = "addi5"} : <i7>
    %316:2 = fork [2] %315 {handshake.bb = 2 : ui32, handshake.name = "fork44"} : <i7>
    %317 = trunci %316#0 {handshake.bb = 2 : ui32, handshake.name = "trunci6"} : <i7> to <i6>
    %319 = cmpi ult, %316#1, %182 {handshake.bb = 2 : ui32, handshake.name = "cmpi0"} : <i7>
    %321:14 = fork [14] %319 {handshake.bb = 2 : ui32, handshake.name = "fork45"} : <i1>
    %trueResult_50, %falseResult_51 = cond_br %321#0, %317 {handshake.bb = 2 : ui32, handshake.name = "cond_br2"} : <i1>, <i6>
    sink %falseResult_51 {handshake.name = "sink8"} : <i6>
    %trueResult_52, %falseResult_53 = cond_br %321#1, %324 {handshake.bb = 2 : ui32, handshake.name = "cond_br3"} : <i1>, <i6>
    %324 = buffer %166#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer137"} : <i6>
    %trueResult_54, %falseResult_55 = cond_br %321#12, %326 {handshake.bb = 2 : ui32, handshake.name = "cond_br4"} : <i1>, <i32>
    %326 = buffer %172#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer139"} : <i32>
    sink %falseResult_55 {handshake.name = "sink9"} : <i32>
    %trueResult_56, %falseResult_57 = cond_br %321#13, %174#1 {handshake.bb = 2 : ui32, handshake.name = "cond_br5"} : <i1>, <>
    %trueResult_58, %falseResult_59 = cond_br %360#9, %falseResult_7 {handshake.bb = 3 : ui32, handshake.name = "cond_br77"} : <i1>, <>
    sink %falseResult_59 {handshake.name = "sink10"} : <>
    %329:2 = fork [2] %trueResult_58 {handshake.bb = 3 : ui32, handshake.name = "fork46"} : <>
    %trueResult_60, %falseResult_61 = cond_br %360#8, %falseResult_11 {handshake.bb = 3 : ui32, handshake.name = "cond_br78"} : <i1>, <>
    sink %falseResult_61 {handshake.name = "sink11"} : <>
    %331:2 = fork [2] %trueResult_60 {handshake.bb = 3 : ui32, handshake.name = "fork47"} : <>
    %trueResult_62, %falseResult_63 = cond_br %360#7, %falseResult {handshake.bb = 3 : ui32, handshake.name = "cond_br79"} : <i1>, <>
    sink %falseResult_63 {handshake.name = "sink12"} : <>
    %333:2 = fork [2] %trueResult_62 {handshake.bb = 3 : ui32, handshake.name = "fork48"} : <>
    %trueResult_64, %falseResult_65 = cond_br %334, %falseResult_15 {handshake.bb = 3 : ui32, handshake.name = "cond_br80"} : <i1>, <>
    %334 = buffer %360#6, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer144"} : <i1>
    sink %falseResult_65 {handshake.name = "sink13"} : <>
    %335:2 = fork [2] %trueResult_64 {handshake.bb = 3 : ui32, handshake.name = "fork49"} : <>
    %trueResult_66, %falseResult_67 = cond_br %360#5, %falseResult_19 {handshake.bb = 3 : ui32, handshake.name = "cond_br81"} : <i1>, <i32>
    sink %falseResult_67 {handshake.name = "sink14"} : <i32>
    %337:2 = fork [2] %trueResult_66 {handshake.bb = 3 : ui32, handshake.name = "fork50"} : <i32>
    %trueResult_68, %falseResult_69 = cond_br %360#4, %falseResult_13 {handshake.bb = 3 : ui32, handshake.name = "cond_br82"} : <i1>, <i32>
    sink %falseResult_69 {handshake.name = "sink15"} : <i32>
    %339:2 = fork [2] %trueResult_68 {handshake.bb = 3 : ui32, handshake.name = "fork51"} : <i32>
    %trueResult_70, %falseResult_71 = cond_br %360#3, %falseResult_9 {handshake.bb = 3 : ui32, handshake.name = "cond_br83"} : <i1>, <i32>
    sink %falseResult_71 {handshake.name = "sink16"} : <i32>
    %341:2 = fork [2] %trueResult_70 {handshake.bb = 3 : ui32, handshake.name = "fork52"} : <i32>
    %trueResult_72, %falseResult_73 = cond_br %360#2, %falseResult_17 {handshake.bb = 3 : ui32, handshake.name = "cond_br84"} : <i1>, <>
    sink %falseResult_73 {handshake.name = "sink17"} : <>
    %343:2 = fork [2] %trueResult_72 {handshake.bb = 3 : ui32, handshake.name = "fork53"} : <>
    %trueResult_74, %falseResult_75 = cond_br %360#1, %falseResult_21 {handshake.bb = 3 : ui32, handshake.name = "cond_br85"} : <i1>, <i32>
    sink %falseResult_75 {handshake.name = "sink18"} : <i32>
    %345:2 = fork [2] %trueResult_74 {handshake.bb = 3 : ui32, handshake.name = "fork54"} : <i32>
    %346 = merge %falseResult_53 {handshake.bb = 3 : ui32, handshake.name = "merge0"} : <i6>
    %347 = extsi %346 {handshake.bb = 3 : ui32, handshake.name = "extsi24"} : <i6> to <i7>
    %result_76, %index_77 = control_merge [%falseResult_57]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>] to <>, <i1>
    sink %index_77 {handshake.name = "sink19"} : <i1>
    %348 = source {handshake.bb = 3 : ui32, handshake.name = "source5"} : <>
    %349 = constant %348 {handshake.bb = 3 : ui32, handshake.name = "constant26", value = 1 : i2} : <>, <i2>
    %350 = extsi %349 {handshake.bb = 3 : ui32, handshake.name = "extsi25"} : <i2> to <i7>
    %351 = source {handshake.bb = 3 : ui32, handshake.name = "source6"} : <>
    %352 = constant %351 {handshake.bb = 3 : ui32, handshake.name = "constant27", value = 20 : i6} : <>, <i6>
    %353 = extsi %352 {handshake.bb = 3 : ui32, handshake.name = "extsi26"} : <i6> to <i7>
    %354 = addi %347, %350 {handshake.bb = 3 : ui32, handshake.name = "addi6"} : <i7>
    %355:2 = fork [2] %354 {handshake.bb = 3 : ui32, handshake.name = "fork55"} : <i7>
    %356 = trunci %355#0 {handshake.bb = 3 : ui32, handshake.name = "trunci7"} : <i7> to <i6>
    %358 = cmpi ult, %355#1, %353 {handshake.bb = 3 : ui32, handshake.name = "cmpi1"} : <i7>
    %360:12 = fork [12] %358 {handshake.bb = 3 : ui32, handshake.name = "fork56"} : <i1>
    %trueResult_78, %falseResult_79 = cond_br %360#0, %356 {handshake.bb = 3 : ui32, handshake.name = "cond_br6"} : <i1>, <i6>
    sink %falseResult_79 {handshake.name = "sink20"} : <i6>
    %trueResult_80, %falseResult_81 = cond_br %360#11, %result_76 {handshake.bb = 3 : ui32, handshake.name = "cond_br7"} : <i1>, <>
    %result_82, %index_83 = control_merge [%falseResult_81]  {handshake.bb = 4 : ui32, handshake.name = "control_merge3"} : [<>] to <>, <i1>
    sink %index_83 {handshake.name = "sink21"} : <i1>
    %363:4 = fork [4] %result_82 {handshake.bb = 4 : ui32, handshake.name = "fork57"} : <>
    end {handshake.bb = 4 : ui32, handshake.name = "end0"} %memEnd_5, %memEnd_3, %memEnd_1, %memEnd, %0#2 : <>, <>, <>, <>, <>
  }
}

