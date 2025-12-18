module {
  handshake.func @triangular(%arg0: memref<10xi32>, %arg1: !handshake.channel<i32>, %arg2: memref<100xi32>, %arg3: !handshake.control<>, %arg4: !handshake.control<>, %arg5: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["x", "n", "a", "x_start", "a_start", "start"], cfg.edges = "[0,1][2,3][4,3][1,2,6,cmpi0][3,4,5,cmpi1][5,1]", resNames = ["x_end", "a_end", "end"]} {
    %0:8 = fork [8] %arg5 {handshake.bb = 0 : ui32, handshake.name = "fork0"} : <>
    %outputs:3, %memEnd = mem_controller[%arg2 : memref<100xi32>] %arg4 (%144, %addressResult, %addressResult_60, %addressResult_62, %dataResult_63) %299#1 {connectedBlocks = [4 : i32], handshake.name = "mem_controller2"} :    (!handshake.channel<i32>, !handshake.channel<i7>, !handshake.channel<i7>, !handshake.channel<i7>, !handshake.channel<i32>) -> (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>)
    %outputs_0, %memEnd_1 = mem_controller[%arg0 : memref<10xi32>] %arg3 (%addressResult_52) %299#0 {connectedBlocks = [4 : i32], handshake.name = "mem_controller3"} :    (!handshake.channel<i4>) -> !handshake.channel<i32>
    %1 = constant %0#1 {handshake.bb = 0 : ui32, handshake.name = "constant0", value = 1000 : i11} : <>, <i11>
    %2:3 = fork [3] %1 {handshake.bb = 0 : ui32, handshake.name = "fork1"} : <i11>
    %3 = extsi %2#0 {handshake.bb = 1 : ui32, handshake.name = "extsi0"} : <i11> to <i32>
    %5 = extsi %2#1 {handshake.bb = 1 : ui32, handshake.name = "extsi1"} : <i11> to <i32>
    %7 = extsi %2#2 {handshake.bb = 1 : ui32, handshake.name = "extsi2"} : <i11> to <i32>
    %9 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant13", value = false} : <>, <i1>
    %10 = br %9 {handshake.bb = 0 : ui32, handshake.name = "br4"} : <i1>
    %11 = extsi %10 {handshake.bb = 0 : ui32, handshake.name = "extsi9"} : <i1> to <i32>
    %12 = br %arg1 {handshake.bb = 0 : ui32, handshake.name = "br5"} : <i32>
    %13 = br %0#7 {handshake.bb = 0 : ui32, handshake.name = "br6"} : <>
    %14 = mux %30#0 [%3, %falseResult_49] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux7"} : <i1>, [<i32>, <i32>] to <i32>
    %16 = mux %30#1 [%5, %falseResult_45] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux8"} : <i1>, [<i32>, <i32>] to <i32>
    %18 = mux %30#2 [%7, %falseResult_41] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux9"} : <i1>, [<i32>, <i32>] to <i32>
    %20 = mux %30#3 [%0#6, %falseResult_39] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux10"} : <i1>, [<>, <>] to <>
    %22 = mux %23 [%0#5, %falseResult_47] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux11"} : <i1>, [<>, <>] to <>
    %23 = buffer %30#4, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer9"} : <i1>
    %24 = mux %30#5 [%0#4, %falseResult_43] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux12"} : <i1>, [<>, <>] to <>
    %26 = mux %30#6 [%0#3, %falseResult_37] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux13"} : <i1>, [<>, <>] to <>
    %28 = init %41#10 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "init0"} : <i1>
    %30:7 = fork [7] %28 {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <i1>
    %31 = mux %37#0 [%11, %296] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i32>, <i32>] to <i32>
    %33:2 = fork [2] %31 {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <i32>
    %34 = mux %37#1 [%12, %297] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i1>, [<i32>, <i32>] to <i32>
    %36:2 = fork [2] %34 {handshake.bb = 1 : ui32, handshake.name = "fork4"} : <i32>
    %result, %index = control_merge [%13, %298]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %37:2 = fork [2] %index {handshake.bb = 1 : ui32, handshake.name = "fork5"} : <i1>
    %38 = cmpi slt, %33#1, %36#1 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i32>
    %41:11 = fork [11] %38 {handshake.bb = 1 : ui32, handshake.name = "fork6"} : <i1>
    %trueResult, %falseResult = cond_br %41#9, %36#0 {handshake.bb = 1 : ui32, handshake.name = "cond_br2"} : <i1>, <i32>
    sink %falseResult {handshake.name = "sink0"} : <i32>
    %trueResult_2, %falseResult_3 = cond_br %41#8, %33#0 {handshake.bb = 1 : ui32, handshake.name = "cond_br3"} : <i1>, <i32>
    sink %falseResult_3 {handshake.name = "sink1"} : <i32>
    %trueResult_4, %falseResult_5 = cond_br %41#7, %result {handshake.bb = 1 : ui32, handshake.name = "cond_br4"} : <i1>, <>
    %47 = merge %trueResult {handshake.bb = 2 : ui32, handshake.name = "merge0"} : <i32>
    %48:2 = fork [2] %47 {handshake.bb = 2 : ui32, handshake.name = "fork7"} : <i32>
    %49 = merge %trueResult_2 {handshake.bb = 2 : ui32, handshake.name = "merge1"} : <i32>
    %50:2 = fork [2] %49 {handshake.bb = 2 : ui32, handshake.name = "fork8"} : <i32>
    %result_6, %index_7 = control_merge [%trueResult_4]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>] to <>, <i1>
    sink %index_7 {handshake.name = "sink2"} : <i1>
    %51:2 = fork [2] %result_6 {handshake.bb = 2 : ui32, handshake.name = "fork9"} : <>
    %52 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %53 = constant %52 {handshake.bb = 2 : ui32, handshake.name = "constant5", value = -1 : i32} : <>, <i32>
    %54 = constant %51#0 {handshake.bb = 2 : ui32, handshake.name = "constant14", value = false} : <>, <i1>
    %55 = subi %48#1, %50#1 {handshake.bb = 2 : ui32, handshake.name = "subi1"} : <i32>
    %58:2 = fork [2] %55 {handshake.bb = 2 : ui32, handshake.name = "fork10"} : <i32>
    %59 = addi %58#1, %53 {handshake.bb = 2 : ui32, handshake.name = "addi1"} : <i32>
    %61 = br %54 {handshake.bb = 2 : ui32, handshake.name = "br7"} : <i1>
    %62 = extsi %61 {handshake.bb = 2 : ui32, handshake.name = "extsi10"} : <i1> to <i32>
    %63 = br %48#0 {handshake.bb = 2 : ui32, handshake.name = "br8"} : <i32>
    %65 = br %50#0 {handshake.bb = 2 : ui32, handshake.name = "br9"} : <i32>
    %67 = br %58#0 {handshake.bb = 2 : ui32, handshake.name = "br10"} : <i32>
    %69 = br %59 {handshake.bb = 2 : ui32, handshake.name = "br11"} : <i32>
    %70 = br %51#1 {handshake.bb = 2 : ui32, handshake.name = "br12"} : <>
    %trueResult_8, %falseResult_9 = cond_br %41#6, %14 {handshake.bb = 3 : ui32, handshake.name = "cond_br33"} : <i1>, <i32>
    sink %falseResult_9 {handshake.name = "sink3"} : <i32>
    %trueResult_10, %falseResult_11 = cond_br %72, %20 {handshake.bb = 3 : ui32, handshake.name = "cond_br34"} : <i1>, <>
    %72 = buffer %41#5, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer29"} : <i1>
    sink %falseResult_11 {handshake.name = "sink4"} : <>
    %trueResult_12, %falseResult_13 = cond_br %41#4, %26 {handshake.bb = 3 : ui32, handshake.name = "cond_br35"} : <i1>, <>
    sink %falseResult_13 {handshake.name = "sink5"} : <>
    %trueResult_14, %falseResult_15 = cond_br %41#3, %24 {handshake.bb = 3 : ui32, handshake.name = "cond_br36"} : <i1>, <>
    sink %falseResult_15 {handshake.name = "sink6"} : <>
    %trueResult_16, %falseResult_17 = cond_br %41#2, %16 {handshake.bb = 3 : ui32, handshake.name = "cond_br37"} : <i1>, <i32>
    sink %falseResult_17 {handshake.name = "sink7"} : <i32>
    %trueResult_18, %falseResult_19 = cond_br %41#1, %18 {handshake.bb = 3 : ui32, handshake.name = "cond_br38"} : <i1>, <i32>
    sink %falseResult_19 {handshake.name = "sink8"} : <i32>
    %trueResult_20, %falseResult_21 = cond_br %77, %22 {handshake.bb = 3 : ui32, handshake.name = "cond_br39"} : <i1>, <>
    %77 = buffer %41#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer34"} : <i1>
    sink %falseResult_21 {handshake.name = "sink9"} : <>
    %78 = mux %96#0 [%trueResult_8, %269] {ftd.phi, handshake.bb = 3 : ui32, handshake.name = "mux14"} : <i1>, [<i32>, <i32>] to <i32>
    %80 = mux %96#1 [%trueResult_16, %268#1] {ftd.phi, handshake.bb = 3 : ui32, handshake.name = "mux15"} : <i1>, [<i32>, <i32>] to <i32>
    %83 = mux %96#2 [%trueResult_18, %265#1] {ftd.phi, handshake.bb = 3 : ui32, handshake.name = "mux16"} : <i1>, [<i32>, <i32>] to <i32>
    %86 = mux %96#3 [%trueResult_10, %277] {ftd.phi, handshake.bb = 3 : ui32, handshake.name = "mux17"} : <i1>, [<>, <>] to <>
    %88 = mux %89 [%trueResult_20, %272#1] {ftd.phi, handshake.bb = 3 : ui32, handshake.name = "mux18"} : <i1>, [<>, <>] to <>
    %89 = buffer %96#4, bufferType = FIFO_BREAK_NONE, numSlots = 4 {handshake.bb = 3 : ui32, handshake.name = "buffer41"} : <i1>
    %90 = mux %91 [%trueResult_14, %274#1] {ftd.phi, handshake.bb = 3 : ui32, handshake.name = "mux19"} : <i1>, [<>, <>] to <>
    %91 = buffer %96#5, bufferType = FIFO_BREAK_NONE, numSlots = 3 {handshake.bb = 3 : ui32, handshake.name = "buffer42"} : <i1>
    %92 = mux %93 [%trueResult_12, %276#1] {ftd.phi, handshake.bb = 3 : ui32, handshake.name = "mux20"} : <i1>, [<>, <>] to <>
    %93 = buffer %96#6, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer43"} : <i1>
    %94 = init %95 {ftd.imerge, handshake.bb = 3 : ui32, handshake.name = "init7"} : <i1>
    %95 = buffer %113#13, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer44"} : <i1>
    %96:7 = fork [7] %94 {handshake.bb = 3 : ui32, handshake.name = "fork11"} : <i1>
    %97 = mux %109#0 [%62, %281] {handshake.bb = 3 : ui32, handshake.name = "mux2"} : <i1>, [<i32>, <i32>] to <i32>
    %99:2 = fork [2] %97 {handshake.bb = 3 : ui32, handshake.name = "fork12"} : <i32>
    %100 = mux %109#1 [%63, %282] {handshake.bb = 3 : ui32, handshake.name = "mux3"} : <i1>, [<i32>, <i32>] to <i32>
    %102 = mux %109#2 [%65, %284] {handshake.bb = 3 : ui32, handshake.name = "mux4"} : <i1>, [<i32>, <i32>] to <i32>
    %104 = mux %109#3 [%67, %286] {handshake.bb = 3 : ui32, handshake.name = "mux5"} : <i1>, [<i32>, <i32>] to <i32>
    %106 = mux %109#4 [%69, %288] {handshake.bb = 3 : ui32, handshake.name = "mux6"} : <i1>, [<i32>, <i32>] to <i32>
    %108:2 = fork [2] %106 {handshake.bb = 3 : ui32, handshake.name = "fork13"} : <i32>
    %result_22, %index_23 = control_merge [%70, %289]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>, <>] to <>, <i1>
    %109:5 = fork [5] %index_23 {handshake.bb = 3 : ui32, handshake.name = "fork14"} : <i1>
    %110 = cmpi slt, %99#1, %108#1 {handshake.bb = 3 : ui32, handshake.name = "cmpi1"} : <i32>
    %113:14 = fork [14] %110 {handshake.bb = 3 : ui32, handshake.name = "fork15"} : <i1>
    %trueResult_24, %falseResult_25 = cond_br %113#12, %100 {handshake.bb = 3 : ui32, handshake.name = "cond_br5"} : <i1>, <i32>
    %trueResult_26, %falseResult_27 = cond_br %113#11, %102 {handshake.bb = 3 : ui32, handshake.name = "cond_br6"} : <i1>, <i32>
    %trueResult_28, %falseResult_29 = cond_br %113#10, %104 {handshake.bb = 3 : ui32, handshake.name = "cond_br7"} : <i1>, <i32>
    sink %falseResult_29 {handshake.name = "sink10"} : <i32>
    %trueResult_30, %falseResult_31 = cond_br %113#9, %108#0 {handshake.bb = 3 : ui32, handshake.name = "cond_br8"} : <i1>, <i32>
    sink %falseResult_31 {handshake.name = "sink11"} : <i32>
    %trueResult_32, %falseResult_33 = cond_br %113#8, %120 {handshake.bb = 3 : ui32, handshake.name = "cond_br9"} : <i1>, <i32>
    %120 = buffer %99#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer58"} : <i32>
    sink %falseResult_33 {handshake.name = "sink12"} : <i32>
    %trueResult_34, %falseResult_35 = cond_br %113#7, %result_22 {handshake.bb = 3 : ui32, handshake.name = "cond_br10"} : <i1>, <>
    %trueResult_36, %falseResult_37 = cond_br %122, %92 {handshake.bb = 4 : ui32, handshake.name = "cond_br40"} : <i1>, <>
    %122 = buffer %113#6, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 4 : ui32, handshake.name = "buffer60"} : <i1>
    %trueResult_38, %falseResult_39 = cond_br %123, %86 {handshake.bb = 4 : ui32, handshake.name = "cond_br41"} : <i1>, <>
    %123 = buffer %113#5, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer61"} : <i1>
    %trueResult_40, %falseResult_41 = cond_br %124, %83 {handshake.bb = 4 : ui32, handshake.name = "cond_br42"} : <i1>, <i32>
    %124 = buffer %113#4, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer62"} : <i1>
    %trueResult_42, %falseResult_43 = cond_br %125, %90 {handshake.bb = 4 : ui32, handshake.name = "cond_br43"} : <i1>, <>
    %125 = buffer %113#3, bufferType = FIFO_BREAK_NONE, numSlots = 3 {handshake.bb = 4 : ui32, handshake.name = "buffer63"} : <i1>
    %trueResult_44, %falseResult_45 = cond_br %126, %80 {handshake.bb = 4 : ui32, handshake.name = "cond_br44"} : <i1>, <i32>
    %126 = buffer %113#2, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer64"} : <i1>
    %trueResult_46, %falseResult_47 = cond_br %127, %88 {handshake.bb = 4 : ui32, handshake.name = "cond_br45"} : <i1>, <>
    %127 = buffer %113#1, bufferType = FIFO_BREAK_NONE, numSlots = 4 {handshake.bb = 4 : ui32, handshake.name = "buffer65"} : <i1>
    %trueResult_48, %falseResult_49 = cond_br %128, %78 {handshake.bb = 4 : ui32, handshake.name = "cond_br46"} : <i1>, <i32>
    %128 = buffer %113#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer66"} : <i1>
    %129 = merge %trueResult_24 {handshake.bb = 4 : ui32, handshake.name = "merge2"} : <i32>
    %130:6 = fork [6] %129 {handshake.bb = 4 : ui32, handshake.name = "fork16"} : <i32>
    %131 = merge %trueResult_26 {handshake.bb = 4 : ui32, handshake.name = "merge3"} : <i32>
    %132:4 = fork [4] %131 {handshake.bb = 4 : ui32, handshake.name = "fork17"} : <i32>
    %133 = merge %trueResult_28 {handshake.bb = 4 : ui32, handshake.name = "merge4"} : <i32>
    %134:3 = fork [3] %133 {handshake.bb = 4 : ui32, handshake.name = "fork18"} : <i32>
    %135 = trunci %136 {handshake.bb = 4 : ui32, handshake.name = "trunci0"} : <i32> to <i7>
    %136 = buffer %134#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer67"} : <i32>
    %137 = trunci %134#1 {handshake.bb = 4 : ui32, handshake.name = "trunci1"} : <i32> to <i4>
    %139 = merge %trueResult_30 {handshake.bb = 4 : ui32, handshake.name = "merge5"} : <i32>
    %140 = merge %trueResult_32 {handshake.bb = 4 : ui32, handshake.name = "merge6"} : <i32>
    %141:4 = fork [4] %140 {handshake.bb = 4 : ui32, handshake.name = "fork19"} : <i32>
    %result_50, %index_51 = control_merge [%trueResult_34]  {handshake.bb = 4 : ui32, handshake.name = "control_merge3"} : [<>] to <>, <i1>
    sink %index_51 {handshake.name = "sink13"} : <i1>
    %142:2 = fork [2] %result_50 {handshake.bb = 4 : ui32, handshake.name = "fork20"} : <>
    %143 = constant %142#0 {handshake.bb = 4 : ui32, handshake.name = "constant15", value = 1 : i2} : <>, <i2>
    %144 = extsi %143 {handshake.bb = 4 : ui32, handshake.name = "extsi5"} : <i2> to <i32>
    %145 = source {handshake.bb = 4 : ui32, handshake.name = "source1"} : <>
    %146 = constant %145 {handshake.bb = 4 : ui32, handshake.name = "constant7", value = -2 : i32} : <>, <i32>
    %147:3 = fork [3] %146 {handshake.bb = 4 : ui32, handshake.name = "fork21"} : <i32>
    %148 = source {handshake.bb = 4 : ui32, handshake.name = "source2"} : <>
    %149 = constant %148 {handshake.bb = 4 : ui32, handshake.name = "constant8", value = -1 : i32} : <>, <i32>
    %150:5 = fork [5] %149 {handshake.bb = 4 : ui32, handshake.name = "fork22"} : <i32>
    %151 = trunci %150#0 {handshake.bb = 4 : ui32, handshake.name = "trunci2"} : <i32> to <i7>
    %153 = trunci %150#1 {handshake.bb = 4 : ui32, handshake.name = "trunci3"} : <i32> to <i4>
    %155 = source {handshake.bb = 4 : ui32, handshake.name = "source3"} : <>
    %156 = constant %155 {handshake.bb = 4 : ui32, handshake.name = "constant16", value = 1 : i2} : <>, <i2>
    %157 = extsi %156 {handshake.bb = 4 : ui32, handshake.name = "extsi6"} : <i2> to <i32>
    %158:7 = fork [7] %157 {handshake.bb = 4 : ui32, handshake.name = "fork23"} : <i32>
    %159 = source {handshake.bb = 4 : ui32, handshake.name = "source4"} : <>
    %160 = constant %159 {handshake.bb = 4 : ui32, handshake.name = "constant17", value = 3 : i3} : <>, <i3>
    %161 = extsi %160 {handshake.bb = 4 : ui32, handshake.name = "extsi7"} : <i3> to <i32>
    %162:3 = fork [3] %161 {handshake.bb = 4 : ui32, handshake.name = "fork24"} : <i32>
    %163 = addi %132#3, %141#3 {handshake.bb = 4 : ui32, handshake.name = "addi0"} : <i32>
    %166 = xori %163, %150#4 {handshake.bb = 4 : ui32, handshake.name = "xori0"} : <i32>
    %168 = addi %166, %158#0 {handshake.bb = 4 : ui32, handshake.name = "addi2"} : <i32>
    %170 = addi %168, %171 {handshake.bb = 4 : ui32, handshake.name = "addi3"} : <i32>
    %171 = buffer %130#5, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer75"} : <i32>
    %172 = addi %170, %147#2 {handshake.bb = 4 : ui32, handshake.name = "addi4"} : <i32>
    %174:2 = fork [2] %172 {handshake.bb = 4 : ui32, handshake.name = "fork25"} : <i32>
    %175 = addi %135, %151 {handshake.bb = 4 : ui32, handshake.name = "addi6"} : <i7>
    %176 = shli %174#1, %158#1 {handshake.bb = 4 : ui32, handshake.name = "shli0"} : <i32>
    %179 = trunci %176 {handshake.bb = 4 : ui32, handshake.name = "trunci4"} : <i32> to <i7>
    %180 = shli %174#0, %162#0 {handshake.bb = 4 : ui32, handshake.name = "shli1"} : <i32>
    %183 = trunci %180 {handshake.bb = 4 : ui32, handshake.name = "trunci5"} : <i32> to <i7>
    %184 = addi %179, %183 {handshake.bb = 4 : ui32, handshake.name = "addi5"} : <i7>
    %185 = addi %175, %184 {handshake.bb = 4 : ui32, handshake.name = "addi15"} : <i7>
    %addressResult, %dataResult = load[%185] %outputs#0 {handshake.bb = 4 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i7>, <i32>, <i7>, <i32>
    %186 = addi %137, %153 {handshake.bb = 4 : ui32, handshake.name = "addi8"} : <i4>
    %addressResult_52, %dataResult_53 = load[%186] %outputs_0 {handshake.bb = 4 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i4>, <i32>, <i4>, <i32>
    %187 = muli %dataResult, %dataResult_53 {handshake.bb = 4 : ui32, handshake.name = "muli0"} : <i32>
    %188 = addi %132#2, %141#2 {handshake.bb = 4 : ui32, handshake.name = "addi7"} : <i32>
    %191 = xori %188, %150#3 {handshake.bb = 4 : ui32, handshake.name = "xori1"} : <i32>
    %193 = addi %191, %158#2 {handshake.bb = 4 : ui32, handshake.name = "addi9"} : <i32>
    %195 = addi %193, %130#4 {handshake.bb = 4 : ui32, handshake.name = "addi10"} : <i32>
    %197 = addi %195, %147#1 {handshake.bb = 4 : ui32, handshake.name = "addi11"} : <i32>
    %199:2 = fork [2] %197 {handshake.bb = 4 : ui32, handshake.name = "fork26"} : <i32>
    %200 = shli %199#1, %158#3 {handshake.bb = 4 : ui32, handshake.name = "shli2"} : <i32>
    %203 = shli %199#0, %162#1 {handshake.bb = 4 : ui32, handshake.name = "shli3"} : <i32>
    %206 = addi %200, %203 {handshake.bb = 4 : ui32, handshake.name = "addi12"} : <i32>
    %207 = addi %208, %206 {handshake.bb = 4 : ui32, handshake.name = "addi16"} : <i32>
    %208 = buffer %130#3, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer91"} : <i32>
    %209:2 = fork [2] %207 {handshake.bb = 4 : ui32, handshake.name = "fork27"} : <i32>
    %210 = gate %209#1, %trueResult_38 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 4 : ui32, handshake.name = "gate0"} : <i32>, !handshake.control<> to <i32>
    %212:3 = fork [3] %210 {handshake.bb = 4 : ui32, handshake.name = "fork28"} : <i32>
    %213 = cmpi ne, %212#2, %trueResult_40 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 4 : ui32, handshake.name = "cmpi2"} : <i32>
    %215:2 = fork [2] %213 {handshake.bb = 4 : ui32, handshake.name = "fork29"} : <i1>
    %216 = cmpi ne, %212#1, %trueResult_44 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 4 : ui32, handshake.name = "cmpi3"} : <i32>
    %218:2 = fork [2] %216 {handshake.bb = 4 : ui32, handshake.name = "fork30"} : <i1>
    %219 = cmpi ne, %212#0, %trueResult_48 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 4 : ui32, handshake.name = "cmpi4"} : <i32>
    %221:2 = fork [2] %219 {handshake.bb = 4 : ui32, handshake.name = "fork31"} : <i1>
    %trueResult_54, %falseResult_55 = cond_br %222, %trueResult_46 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 4 : ui32, handshake.name = "cond_br30"} : <i1>, <>
    %222 = buffer %215#1, bufferType = FIFO_BREAK_NONE, numSlots = 4 {handshake.bb = 4 : ui32, handshake.name = "buffer96"} : <i1>
    sink %trueResult_54 {handshake.name = "sink14"} : <>
    %trueResult_56, %falseResult_57 = cond_br %223, %trueResult_42 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 4 : ui32, handshake.name = "cond_br31"} : <i1>, <>
    %223 = buffer %218#1, bufferType = FIFO_BREAK_NONE, numSlots = 3 {handshake.bb = 4 : ui32, handshake.name = "buffer97"} : <i1>
    sink %trueResult_56 {handshake.name = "sink15"} : <>
    %trueResult_58, %falseResult_59 = cond_br %224, %trueResult_36 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 4 : ui32, handshake.name = "cond_br32"} : <i1>, <>
    %224 = buffer %221#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer98"} : <i1>
    sink %trueResult_58 {handshake.name = "sink16"} : <>
    %225 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 4 : ui32, handshake.name = "source6"} : <>
    %226 = mux %227 [%falseResult_55, %225] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 4 : ui32, handshake.name = "mux21"} : <i1>, [<>, <>] to <>
    %227 = buffer %215#0, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 4 : ui32, handshake.name = "buffer99"} : <i1>
    %228 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 4 : ui32, handshake.name = "source7"} : <>
    %229 = mux %230 [%falseResult_57, %228] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 4 : ui32, handshake.name = "mux22"} : <i1>, [<>, <>] to <>
    %230 = buffer %218#0, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 4 : ui32, handshake.name = "buffer100"} : <i1>
    %231 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 4 : ui32, handshake.name = "source8"} : <>
    %232 = mux %233 [%falseResult_59, %231] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 4 : ui32, handshake.name = "mux23"} : <i1>, [<>, <>] to <>
    %233 = buffer %221#0, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 4 : ui32, handshake.name = "buffer101"} : <i1>
    %234 = join %226, %229, %232 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 4 : ui32, handshake.name = "join0"} : <>
    %235 = gate %236, %234 {handshake.bb = 4 : ui32, handshake.name = "gate1"} : <i32>, !handshake.control<> to <i32>
    %236 = buffer %209#0, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 4 : ui32, handshake.name = "buffer102"} : <i32>
    %237 = trunci %235 {handshake.bb = 4 : ui32, handshake.name = "trunci6"} : <i32> to <i7>
    %addressResult_60, %dataResult_61 = load[%237] %outputs#1 {handshake.bb = 4 : ui32, handshake.deps = #handshake<deps[["store0", 1, false], ["store0", 3, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load2"} : <i7>, <i32>, <i7>, <i32>
    %238 = subi %dataResult_61, %187 {handshake.bb = 4 : ui32, handshake.name = "subi0"} : <i32>
    %239 = addi %132#1, %141#1 {handshake.bb = 4 : ui32, handshake.name = "addi20"} : <i32>
    %242 = xori %239, %243 {handshake.bb = 4 : ui32, handshake.name = "xori2"} : <i32>
    %243 = buffer %150#2, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer105"} : <i32>
    %244 = addi %242, %245 {handshake.bb = 4 : ui32, handshake.name = "addi21"} : <i32>
    %245 = buffer %158#4, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer106"} : <i32>
    %246 = addi %244, %130#2 {handshake.bb = 4 : ui32, handshake.name = "addi13"} : <i32>
    %248 = addi %246, %147#0 {handshake.bb = 4 : ui32, handshake.name = "addi14"} : <i32>
    %250:2 = fork [2] %248 {handshake.bb = 4 : ui32, handshake.name = "fork32"} : <i32>
    %251 = shli %250#1, %253 {handshake.bb = 4 : ui32, handshake.name = "shli4"} : <i32>
    %253 = buffer %158#5, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer110"} : <i32>
    %254 = shli %250#0, %256 {handshake.bb = 4 : ui32, handshake.name = "shli5"} : <i32>
    %256 = buffer %162#2, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer112"} : <i32>
    %257 = addi %251, %254 {handshake.bb = 4 : ui32, handshake.name = "addi22"} : <i32>
    %258 = addi %259, %257 {handshake.bb = 4 : ui32, handshake.name = "addi17"} : <i32>
    %259 = buffer %130#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer113"} : <i32>
    %260:2 = fork [2] %258 {handshake.bb = 4 : ui32, handshake.name = "fork33"} : <i32>
    %261 = trunci %262 {handshake.bb = 4 : ui32, handshake.name = "trunci7"} : <i32> to <i7>
    %262 = buffer %260#0, bufferType = FIFO_BREAK_NONE, numSlots = 3 {handshake.bb = 4 : ui32, handshake.name = "buffer114"} : <i32>
    %263 = buffer %260#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer0"} : <i32>
    %265:2 = fork [2] %263 {handshake.bb = 4 : ui32, handshake.name = "fork34"} : <i32>
    %266 = init %265#0 {handshake.bb = 4 : ui32, handshake.name = "init14"} : <i32>
    %268:2 = fork [2] %266 {handshake.bb = 4 : ui32, handshake.name = "fork35"} : <i32>
    %269 = init %270 {handshake.bb = 4 : ui32, handshake.name = "init15"} : <i32>
    %270 = buffer %268#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer117"} : <i32>
    %271 = buffer %doneResult, bufferType = FIFO_BREAK_NONE, numSlots = 1 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 4 : ui32, handshake.name = "buffer1"} : <>
    %272:2 = fork [2] %271 {handshake.bb = 4 : ui32, handshake.name = "fork36"} : <>
    %273 = init %272#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 4 : ui32, handshake.name = "init16"} : <>
    %274:2 = fork [2] %273 {handshake.bb = 4 : ui32, handshake.name = "fork37"} : <>
    %275 = init %274#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 4 : ui32, handshake.name = "init17"} : <>
    %276:2 = fork [2] %275 {handshake.bb = 4 : ui32, handshake.name = "fork38"} : <>
    %277 = init %276#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 4 : ui32, handshake.name = "init18"} : <>
    %addressResult_62, %dataResult_63, %doneResult = store[%261] %238 %outputs#2 {handshake.bb = 4 : ui32, handshake.deps = #handshake<deps[["load2", 1, false], ["store0", 1, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store0"} : <i7>, <i32>, <>, <i7>, <i32>, <>
    %278 = addi %141#0, %279 {handshake.bb = 4 : ui32, handshake.name = "addi18"} : <i32>
    %279 = buffer %158#6, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer118"} : <i32>
    %281 = br %278 {handshake.bb = 4 : ui32, handshake.name = "br13"} : <i32>
    %282 = br %130#0 {handshake.bb = 4 : ui32, handshake.name = "br14"} : <i32>
    %284 = br %132#0 {handshake.bb = 4 : ui32, handshake.name = "br15"} : <i32>
    %286 = br %134#2 {handshake.bb = 4 : ui32, handshake.name = "br16"} : <i32>
    %288 = br %139 {handshake.bb = 4 : ui32, handshake.name = "br17"} : <i32>
    %289 = br %142#1 {handshake.bb = 4 : ui32, handshake.name = "br18"} : <>
    %290 = merge %falseResult_25 {handshake.bb = 5 : ui32, handshake.name = "merge7"} : <i32>
    %291 = merge %falseResult_27 {handshake.bb = 5 : ui32, handshake.name = "merge8"} : <i32>
    %result_64, %index_65 = control_merge [%falseResult_35]  {handshake.bb = 5 : ui32, handshake.name = "control_merge4"} : [<>] to <>, <i1>
    sink %index_65 {handshake.name = "sink17"} : <i1>
    %292 = source {handshake.bb = 5 : ui32, handshake.name = "source5"} : <>
    %293 = constant %292 {handshake.bb = 5 : ui32, handshake.name = "constant18", value = 1 : i2} : <>, <i2>
    %294 = extsi %293 {handshake.bb = 5 : ui32, handshake.name = "extsi8"} : <i2> to <i32>
    %295 = addi %291, %294 {handshake.bb = 5 : ui32, handshake.name = "addi19"} : <i32>
    %296 = br %295 {handshake.bb = 5 : ui32, handshake.name = "br19"} : <i32>
    %297 = br %290 {handshake.bb = 5 : ui32, handshake.name = "br20"} : <i32>
    %298 = br %result_64 {handshake.bb = 5 : ui32, handshake.name = "br21"} : <>
    %result_66, %index_67 = control_merge [%falseResult_5]  {handshake.bb = 6 : ui32, handshake.name = "control_merge5"} : [<>] to <>, <i1>
    sink %index_67 {handshake.name = "sink18"} : <i1>
    %299:2 = fork [2] %result_66 {handshake.bb = 6 : ui32, handshake.name = "fork39"} : <>
    end {handshake.bb = 6 : ui32, handshake.name = "end0"} %memEnd_1, %memEnd, %0#2 : <>, <>, <>
  }
}

