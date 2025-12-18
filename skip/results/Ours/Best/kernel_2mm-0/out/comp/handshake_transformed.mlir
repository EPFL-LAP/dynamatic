module {
  handshake.func @kernel_2mm(%arg0: !handshake.channel<i32>, %arg1: !handshake.channel<i32>, %arg2: memref<100xi32>, %arg3: memref<100xi32>, %arg4: memref<100xi32>, %arg5: memref<100xi32>, %arg6: memref<100xi32>, %arg7: !handshake.control<>, %arg8: !handshake.control<>, %arg9: !handshake.control<>, %arg10: !handshake.control<>, %arg11: !handshake.control<>, %arg12: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["alpha", "beta", "tmp", "A", "B", "C", "D", "tmp_start", "A_start", "B_start", "C_start", "D_start", "start"], cfg.edges = "[0,1][7,8][2,3][9,7,10,cmpi4][4,2,5,cmpi1][6,7][1,2][8,8,9,cmpi3][3,3,4,cmpi0][10,6,11,cmpi5][5,1,6,cmpi2]", resNames = ["tmp_end", "A_end", "B_end", "C_end", "D_end", "end"]} {
    %0:5 = fork [5] %arg12 {handshake.bb = 0 : ui32, handshake.name = "fork0"} : <>
    %outputs:4, %memEnd = mem_controller[%arg6 : memref<100xi32>] %arg11 (%305, %addressResult_71, %addressResult_73, %dataResult_74, %399, %addressResult_92, %addressResult_94, %dataResult_95) %521#4 {connectedBlocks = [7 : i32, 8 : i32], handshake.name = "mem_controller3"} :    (!handshake.channel<i32>, !handshake.channel<i7>, !handshake.channel<i7>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i7>, !handshake.channel<i7>, !handshake.channel<i32>) -> (!handshake.channel<i32>, !handshake.control<>, !handshake.channel<i32>, !handshake.control<>)
    %outputs_0, %memEnd_1 = mem_controller[%arg5 : memref<100xi32>] %arg10 (%addressResult_90) %521#3 {connectedBlocks = [8 : i32], handshake.name = "mem_controller4"} :    (!handshake.channel<i7>) -> !handshake.channel<i32>
    %outputs_2, %memEnd_3 = mem_controller[%arg4 : memref<100xi32>] %arg9 (%addressResult_16) %521#2 {connectedBlocks = [3 : i32], handshake.name = "mem_controller5"} :    (!handshake.channel<i7>) -> !handshake.channel<i32>
    %outputs_4, %memEnd_5 = mem_controller[%arg3 : memref<100xi32>] %arg8 (%addressResult_14) %521#1 {connectedBlocks = [3 : i32], handshake.name = "mem_controller6"} :    (!handshake.channel<i7>) -> !handshake.channel<i32>
    %outputs_6:4, %memEnd_7 = mem_controller[%arg2 : memref<100xi32>] %arg7 (%46, %addressResult, %dataResult, %119, %addressResult_18, %addressResult_20, %dataResult_21, %addressResult_88) %521#0 {connectedBlocks = [2 : i32, 3 : i32, 8 : i32], handshake.name = "mem_controller7"} :    (!handshake.channel<i32>, !handshake.channel<i7>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i7>, !handshake.channel<i7>, !handshake.channel<i32>, !handshake.channel<i7>) -> (!handshake.control<>, !handshake.channel<i32>, !handshake.control<>, !handshake.channel<i32>)
    %1 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant29", value = false} : <>, <i1>
    %2 = br %1 {handshake.bb = 0 : ui32, handshake.name = "br3"} : <i1>
    %3 = extsi %2 {handshake.bb = 0 : ui32, handshake.name = "extsi34"} : <i1> to <i5>
    %4 = br %arg0 {handshake.bb = 0 : ui32, handshake.name = "br6"} : <i32>
    %5 = br %arg1 {handshake.bb = 0 : ui32, handshake.name = "br7"} : <i32>
    %6 = br %0#4 {handshake.bb = 0 : ui32, handshake.name = "br8"} : <>
    %7 = mux %8 [%0#3, %trueResult_53] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux25"} : <i1>, [<>, <>] to <>
    %8 = init %243#3 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "init0"} : <i1>
    %10 = mux %16#0 [%3, %trueResult_57] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i5>, <i5>] to <i5>
    %12 = mux %13 [%4, %trueResult_59] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i1>, [<i32>, <i32>] to <i32>
    %13 = buffer %16#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer7"} : <i1>
    %14 = mux %15 [%5, %trueResult_61] {handshake.bb = 1 : ui32, handshake.name = "mux2"} : <i1>, [<i32>, <i32>] to <i32>
    %15 = buffer %16#2, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer8"} : <i1>
    %result, %index = control_merge [%6, %trueResult_63]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %16:3 = fork [3] %index {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <i1>
    %17:2 = fork [2] %result {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <>
    %18 = constant %17#0 {handshake.bb = 1 : ui32, handshake.name = "constant30", value = false} : <>, <i1>
    %19 = br %18 {handshake.bb = 1 : ui32, handshake.name = "br9"} : <i1>
    %20 = extsi %19 {handshake.bb = 1 : ui32, handshake.name = "extsi33"} : <i1> to <i5>
    %21 = br %12 {handshake.bb = 1 : ui32, handshake.name = "br10"} : <i32>
    %22 = br %14 {handshake.bb = 1 : ui32, handshake.name = "br11"} : <i32>
    %23 = br %10 {handshake.bb = 1 : ui32, handshake.name = "br12"} : <i5>
    %24 = br %17#1 {handshake.bb = 1 : ui32, handshake.name = "br13"} : <>
    %25 = mux %26 [%7, %trueResult_35] {ftd.phi, handshake.bb = 2 : ui32, handshake.name = "mux30"} : <i1>, [<>, <>] to <>
    %26 = init %27 {ftd.imerge, handshake.bb = 2 : ui32, handshake.name = "init5"} : <i1>
    %27 = buffer %217#4, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer9"} : <i1>
    %28 = mux %43#1 [%20, %trueResult_41] {handshake.bb = 2 : ui32, handshake.name = "mux3"} : <i1>, [<i5>, <i5>] to <i5>
    %30:2 = fork [2] %28 {handshake.bb = 2 : ui32, handshake.name = "fork3"} : <i5>
    %31 = extsi %30#0 {handshake.bb = 2 : ui32, handshake.name = "extsi35"} : <i5> to <i7>
    %33 = mux %34 [%21, %trueResult_43] {handshake.bb = 2 : ui32, handshake.name = "mux4"} : <i1>, [<i32>, <i32>] to <i32>
    %34 = buffer %43#2, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer12"} : <i1>
    %35 = mux %36 [%22, %trueResult_45] {handshake.bb = 2 : ui32, handshake.name = "mux5"} : <i1>, [<i32>, <i32>] to <i32>
    %36 = buffer %43#3, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer13"} : <i1>
    %37 = mux %43#0 [%23, %trueResult_47] {handshake.bb = 2 : ui32, handshake.name = "mux6"} : <i1>, [<i5>, <i5>] to <i5>
    %39:2 = fork [2] %37 {handshake.bb = 2 : ui32, handshake.name = "fork4"} : <i5>
    %40 = extsi %39#1 {handshake.bb = 2 : ui32, handshake.name = "extsi36"} : <i5> to <i32>
    %42:2 = fork [2] %40 {handshake.bb = 2 : ui32, handshake.name = "fork5"} : <i32>
    %result_8, %index_9 = control_merge [%24, %trueResult_49]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>, <>] to <>, <i1>
    %43:4 = fork [4] %index_9 {handshake.bb = 2 : ui32, handshake.name = "fork6"} : <i1>
    %44:3 = fork [3] %result_8 {handshake.bb = 2 : ui32, handshake.name = "fork7"} : <>
    %45 = constant %44#1 {handshake.bb = 2 : ui32, handshake.name = "constant31", value = 1 : i2} : <>, <i2>
    %46 = extsi %45 {handshake.bb = 2 : ui32, handshake.name = "extsi2"} : <i2> to <i32>
    %47 = constant %44#0 {handshake.bb = 2 : ui32, handshake.name = "constant32", value = false} : <>, <i1>
    %48:2 = fork [2] %47 {handshake.bb = 2 : ui32, handshake.name = "fork8"} : <i1>
    %49 = extsi %48#1 {handshake.bb = 2 : ui32, handshake.name = "extsi4"} : <i1> to <i32>
    %51 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %52 = constant %51 {handshake.bb = 2 : ui32, handshake.name = "constant33", value = 1 : i2} : <>, <i2>
    %53 = extsi %52 {handshake.bb = 2 : ui32, handshake.name = "extsi5"} : <i2> to <i32>
    %54 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %55 = constant %54 {handshake.bb = 2 : ui32, handshake.name = "constant34", value = 3 : i3} : <>, <i3>
    %56 = extsi %55 {handshake.bb = 2 : ui32, handshake.name = "extsi6"} : <i3> to <i32>
    %57 = shli %42#0, %53 {handshake.bb = 2 : ui32, handshake.name = "shli0"} : <i32>
    %59 = trunci %57 {handshake.bb = 2 : ui32, handshake.name = "trunci0"} : <i32> to <i7>
    %60 = shli %42#1, %56 {handshake.bb = 2 : ui32, handshake.name = "shli1"} : <i32>
    %62 = trunci %60 {handshake.bb = 2 : ui32, handshake.name = "trunci1"} : <i32> to <i7>
    %63 = addi %59, %62 {handshake.bb = 2 : ui32, handshake.name = "addi19"} : <i7>
    %64 = addi %31, %63 {handshake.bb = 2 : ui32, handshake.name = "addi2"} : <i7>
    %65 = buffer %doneResult, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer0"} : <>
    %66:2 = fork [2] %65 {handshake.bb = 2 : ui32, handshake.name = "fork9"} : <>
    %addressResult, %dataResult, %doneResult = store[%64] %49 %outputs_6#0 {handshake.bb = 2 : ui32, handshake.deps = #handshake<deps[["load2", 3, false], ["store1", 3, false], ["load4", 1, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store0"} : <i7>, <i32>, <>, <i7>, <i32>, <>
    %67 = br %48#0 {handshake.bb = 2 : ui32, handshake.name = "br14"} : <i1>
    %69 = extsi %67 {handshake.bb = 2 : ui32, handshake.name = "extsi32"} : <i1> to <i5>
    %70 = br %33 {handshake.bb = 2 : ui32, handshake.name = "br15"} : <i32>
    %71 = br %35 {handshake.bb = 2 : ui32, handshake.name = "br16"} : <i32>
    %72 = br %39#0 {handshake.bb = 2 : ui32, handshake.name = "br17"} : <i5>
    %74 = br %30#1 {handshake.bb = 2 : ui32, handshake.name = "br18"} : <i5>
    %76 = br %44#2 {handshake.bb = 2 : ui32, handshake.name = "br19"} : <>
    %trueResult, %falseResult = cond_br %77, %84#2 {handshake.bb = 3 : ui32, handshake.name = "cond_br83"} : <i1>, <>
    %77 = buffer %188#5, bufferType = FIFO_BREAK_NONE, numSlots = 5 {handshake.bb = 3 : ui32, handshake.name = "buffer22"} : <i1>
    sink %falseResult {handshake.name = "sink0"} : <>
    %trueResult_10, %falseResult_11 = cond_br %78, %179 {handshake.bb = 3 : ui32, handshake.name = "cond_br84"} : <i1>, <>
    %78 = buffer %188#4, bufferType = FIFO_BREAK_NONE, numSlots = 5 {handshake.bb = 3 : ui32, handshake.name = "buffer23"} : <i1>
    %79 = init %188#3 {ftd.imerge, handshake.bb = 3 : ui32, handshake.name = "init10"} : <i1>
    %81:2 = fork [2] %79 {handshake.bb = 3 : ui32, handshake.name = "fork10"} : <i1>
    %82 = mux %83 [%66#1, %trueResult] {ftd.regen, handshake.bb = 3 : ui32, handshake.name = "mux31"} : <i1>, [<>, <>] to <>
    %83 = buffer %81#1, bufferType = FIFO_BREAK_NONE, numSlots = 5 {handshake.bb = 3 : ui32, handshake.name = "buffer25"} : <i1>
    %84:3 = fork [3] %82 {handshake.bb = 3 : ui32, handshake.name = "fork11"} : <>
    %85 = mux %86 [%25, %trueResult_10] {ftd.phi, handshake.bb = 3 : ui32, handshake.name = "mux34"} : <i1>, [<>, <>] to <>
    %86 = buffer %81#0, bufferType = FIFO_BREAK_NONE, numSlots = 5 {handshake.bb = 3 : ui32, handshake.name = "buffer26"} : <i1>
    %87 = mux %116#2 [%69, %trueResult_23] {handshake.bb = 3 : ui32, handshake.name = "mux7"} : <i1>, [<i5>, <i5>] to <i5>
    %89:3 = fork [3] %87 {handshake.bb = 3 : ui32, handshake.name = "fork12"} : <i5>
    %90 = extsi %91 {handshake.bb = 3 : ui32, handshake.name = "extsi37"} : <i5> to <i7>
    %91 = buffer %89#0, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 3 : ui32, handshake.name = "buffer28"} : <i5>
    %92 = extsi %89#1 {handshake.bb = 3 : ui32, handshake.name = "extsi38"} : <i5> to <i6>
    %94 = extsi %89#2 {handshake.bb = 3 : ui32, handshake.name = "extsi39"} : <i5> to <i32>
    %96:2 = fork [2] %94 {handshake.bb = 3 : ui32, handshake.name = "fork13"} : <i32>
    %97 = mux %116#3 [%70, %trueResult_25] {handshake.bb = 3 : ui32, handshake.name = "mux8"} : <i1>, [<i32>, <i32>] to <i32>
    %99:2 = fork [2] %97 {handshake.bb = 3 : ui32, handshake.name = "fork14"} : <i32>
    %100 = mux %116#4 [%71, %trueResult_27] {handshake.bb = 3 : ui32, handshake.name = "mux9"} : <i1>, [<i32>, <i32>] to <i32>
    %102 = mux %116#0 [%72, %trueResult_29] {handshake.bb = 3 : ui32, handshake.name = "mux10"} : <i1>, [<i5>, <i5>] to <i5>
    %104:2 = fork [2] %102 {handshake.bb = 3 : ui32, handshake.name = "fork15"} : <i5>
    %105 = extsi %106 {handshake.bb = 3 : ui32, handshake.name = "extsi40"} : <i5> to <i32>
    %106 = buffer %104#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer34"} : <i5>
    %107:6 = fork [6] %105 {handshake.bb = 3 : ui32, handshake.name = "fork16"} : <i32>
    %108 = mux %109 [%74, %trueResult_31] {handshake.bb = 3 : ui32, handshake.name = "mux11"} : <i1>, [<i5>, <i5>] to <i5>
    %109 = buffer %116#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer35"} : <i1>
    %110:3 = fork [3] %108 {handshake.bb = 3 : ui32, handshake.name = "fork17"} : <i5>
    %111 = extsi %112 {handshake.bb = 3 : ui32, handshake.name = "extsi41"} : <i5> to <i7>
    %112 = buffer %110#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer36"} : <i5>
    %113 = extsi %110#2 {handshake.bb = 3 : ui32, handshake.name = "extsi42"} : <i5> to <i32>
    %115:2 = fork [2] %113 {handshake.bb = 3 : ui32, handshake.name = "fork18"} : <i32>
    %result_12, %index_13 = control_merge [%76, %trueResult_33]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>, <>] to <>, <i1>
    %116:5 = fork [5] %index_13 {handshake.bb = 3 : ui32, handshake.name = "fork19"} : <i1>
    %117:2 = fork [2] %result_12 {handshake.bb = 3 : ui32, handshake.name = "fork20"} : <>
    %118 = constant %117#0 {handshake.bb = 3 : ui32, handshake.name = "constant35", value = 1 : i2} : <>, <i2>
    %119 = extsi %118 {handshake.bb = 3 : ui32, handshake.name = "extsi7"} : <i2> to <i32>
    %120 = source {handshake.bb = 3 : ui32, handshake.name = "source2"} : <>
    %121 = constant %120 {handshake.bb = 3 : ui32, handshake.name = "constant36", value = 10 : i5} : <>, <i5>
    %122 = extsi %121 {handshake.bb = 3 : ui32, handshake.name = "extsi43"} : <i5> to <i6>
    %123 = source {handshake.bb = 3 : ui32, handshake.name = "source3"} : <>
    %124 = constant %123 {handshake.bb = 3 : ui32, handshake.name = "constant37", value = 1 : i2} : <>, <i2>
    %125:2 = fork [2] %124 {handshake.bb = 3 : ui32, handshake.name = "fork21"} : <i2>
    %126 = extsi %127 {handshake.bb = 3 : ui32, handshake.name = "extsi44"} : <i2> to <i6>
    %127 = buffer %125#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer38"} : <i2>
    %128 = extsi %129 {handshake.bb = 3 : ui32, handshake.name = "extsi9"} : <i2> to <i32>
    %129 = buffer %125#1, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 3 : ui32, handshake.name = "buffer39"} : <i2>
    %130:4 = fork [4] %128 {handshake.bb = 3 : ui32, handshake.name = "fork22"} : <i32>
    %131 = source {handshake.bb = 3 : ui32, handshake.name = "source4"} : <>
    %132 = constant %131 {handshake.bb = 3 : ui32, handshake.name = "constant38", value = 3 : i3} : <>, <i3>
    %133 = extsi %132 {handshake.bb = 3 : ui32, handshake.name = "extsi10"} : <i3> to <i32>
    %134:4 = fork [4] %133 {handshake.bb = 3 : ui32, handshake.name = "fork23"} : <i32>
    %135 = shli %137, %130#0 {handshake.bb = 3 : ui32, handshake.name = "shli2"} : <i32>
    %137 = buffer %107#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer41"} : <i32>
    %138 = trunci %135 {handshake.bb = 3 : ui32, handshake.name = "trunci2"} : <i32> to <i7>
    %139 = shli %141, %134#0 {handshake.bb = 3 : ui32, handshake.name = "shli3"} : <i32>
    %141 = buffer %107#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer43"} : <i32>
    %142 = trunci %139 {handshake.bb = 3 : ui32, handshake.name = "trunci3"} : <i32> to <i7>
    %143 = addi %138, %142 {handshake.bb = 3 : ui32, handshake.name = "addi20"} : <i7>
    %144 = addi %90, %143 {handshake.bb = 3 : ui32, handshake.name = "addi3"} : <i7>
    %addressResult_14, %dataResult_15 = load[%144] %outputs_4 {handshake.bb = 3 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i7>, <i32>, <i7>, <i32>
    %145 = muli %146, %dataResult_15 {handshake.bb = 3 : ui32, handshake.name = "muli0"} : <i32>
    %146 = buffer %99#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer44"} : <i32>
    %147 = shli %149, %130#1 {handshake.bb = 3 : ui32, handshake.name = "shli4"} : <i32>
    %149 = buffer %96#0, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 3 : ui32, handshake.name = "buffer46"} : <i32>
    %150 = trunci %147 {handshake.bb = 3 : ui32, handshake.name = "trunci4"} : <i32> to <i7>
    %151 = shli %153, %134#1 {handshake.bb = 3 : ui32, handshake.name = "shli5"} : <i32>
    %153 = buffer %96#1, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 3 : ui32, handshake.name = "buffer48"} : <i32>
    %154 = trunci %151 {handshake.bb = 3 : ui32, handshake.name = "trunci5"} : <i32> to <i7>
    %155 = addi %150, %154 {handshake.bb = 3 : ui32, handshake.name = "addi21"} : <i7>
    %156 = addi %111, %155 {handshake.bb = 3 : ui32, handshake.name = "addi4"} : <i7>
    %addressResult_16, %dataResult_17 = load[%156] %outputs_2 {handshake.bb = 3 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i7>, <i32>, <i7>, <i32>
    %157 = muli %145, %dataResult_17 {handshake.bb = 3 : ui32, handshake.name = "muli1"} : <i32>
    %158 = shli %160, %159 {handshake.bb = 3 : ui32, handshake.name = "shli6"} : <i32>
    %159 = buffer %130#2, bufferType = FIFO_BREAK_NONE, numSlots = 5 {handshake.bb = 3 : ui32, handshake.name = "buffer49"} : <i32>
    %160 = buffer %107#2, bufferType = FIFO_BREAK_NONE, numSlots = 5 {handshake.bb = 3 : ui32, handshake.name = "buffer50"} : <i32>
    %161 = shli %163, %162 {handshake.bb = 3 : ui32, handshake.name = "shli7"} : <i32>
    %162 = buffer %134#2, bufferType = FIFO_BREAK_NONE, numSlots = 5 {handshake.bb = 3 : ui32, handshake.name = "buffer51"} : <i32>
    %163 = buffer %107#3, bufferType = FIFO_BREAK_NONE, numSlots = 5 {handshake.bb = 3 : ui32, handshake.name = "buffer52"} : <i32>
    %164 = addi %158, %161 {handshake.bb = 3 : ui32, handshake.name = "addi22"} : <i32>
    %165 = addi %166, %164 {handshake.bb = 3 : ui32, handshake.name = "addi5"} : <i32>
    %166 = buffer %115#0, bufferType = FIFO_BREAK_NONE, numSlots = 4 {handshake.bb = 3 : ui32, handshake.name = "buffer53"} : <i32>
    %167 = gate %165, %84#1, %85 {handshake.bb = 3 : ui32, handshake.name = "gate0"} : <i32>, !handshake.control<>, !handshake.control<> to <i32>
    %168 = trunci %167 {handshake.bb = 3 : ui32, handshake.name = "trunci6"} : <i32> to <i7>
    %addressResult_18, %dataResult_19 = load[%168] %outputs_6#1 {handshake.bb = 3 : ui32, handshake.deps = #handshake<deps[["store1", 3, false], ["store1", 4, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load2"} : <i7>, <i32>, <i7>, <i32>
    %169 = addi %dataResult_19, %157 {handshake.bb = 3 : ui32, handshake.name = "addi0"} : <i32>
    %170 = shli %172, %171 {handshake.bb = 3 : ui32, handshake.name = "shli8"} : <i32>
    %171 = buffer %130#3, bufferType = FIFO_BREAK_NONE, numSlots = 4 {handshake.bb = 3 : ui32, handshake.name = "buffer54"} : <i32>
    %172 = buffer %107#4, bufferType = FIFO_BREAK_NONE, numSlots = 4 {handshake.bb = 3 : ui32, handshake.name = "buffer55"} : <i32>
    %173 = shli %175, %174 {handshake.bb = 3 : ui32, handshake.name = "shli9"} : <i32>
    %174 = buffer %134#3, bufferType = FIFO_BREAK_NONE, numSlots = 4 {handshake.bb = 3 : ui32, handshake.name = "buffer56"} : <i32>
    %175 = buffer %107#5, bufferType = FIFO_BREAK_NONE, numSlots = 4 {handshake.bb = 3 : ui32, handshake.name = "buffer57"} : <i32>
    %176 = addi %170, %173 {handshake.bb = 3 : ui32, handshake.name = "addi23"} : <i32>
    %177 = addi %178, %176 {handshake.bb = 3 : ui32, handshake.name = "addi6"} : <i32>
    %178 = buffer %115#1, bufferType = FIFO_BREAK_NONE, numSlots = 5 {handshake.bb = 3 : ui32, handshake.name = "buffer58"} : <i32>
    %179 = buffer %doneResult_22, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer1"} : <>
    %180 = gate %177, %84#0 {handshake.bb = 3 : ui32, handshake.name = "gate1"} : <i32>, !handshake.control<> to <i32>
    %181 = trunci %180 {handshake.bb = 3 : ui32, handshake.name = "trunci7"} : <i32> to <i7>
    %addressResult_20, %dataResult_21, %doneResult_22 = store[%181] %169 %outputs_6#2 {handshake.bb = 3 : ui32, handshake.deps = #handshake<deps[["load2", 3, false], ["store1", 3, false], ["load4", 1, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store1"} : <i7>, <i32>, <>, <i7>, <i32>, <>
    %182 = addi %92, %126 {handshake.bb = 3 : ui32, handshake.name = "addi13"} : <i6>
    %183:2 = fork [2] %182 {handshake.bb = 3 : ui32, handshake.name = "fork24"} : <i6>
    %184 = trunci %185 {handshake.bb = 3 : ui32, handshake.name = "trunci8"} : <i6> to <i5>
    %185 = buffer %183#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer59"} : <i6>
    %186 = cmpi ult, %187, %122 {handshake.bb = 3 : ui32, handshake.name = "cmpi0"} : <i6>
    %187 = buffer %183#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer60"} : <i6>
    %188:9 = fork [9] %186 {handshake.bb = 3 : ui32, handshake.name = "fork25"} : <i1>
    %trueResult_23, %falseResult_24 = cond_br %188#0, %184 {handshake.bb = 3 : ui32, handshake.name = "cond_br6"} : <i1>, <i5>
    sink %falseResult_24 {handshake.name = "sink1"} : <i5>
    %trueResult_25, %falseResult_26 = cond_br %190, %191 {handshake.bb = 3 : ui32, handshake.name = "cond_br7"} : <i1>, <i32>
    %190 = buffer %188#6, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer62"} : <i1>
    %191 = buffer %99#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer63"} : <i32>
    %trueResult_27, %falseResult_28 = cond_br %192, %100 {handshake.bb = 3 : ui32, handshake.name = "cond_br8"} : <i1>, <i32>
    %192 = buffer %188#7, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer64"} : <i1>
    %trueResult_29, %falseResult_30 = cond_br %193, %194 {handshake.bb = 3 : ui32, handshake.name = "cond_br9"} : <i1>, <i5>
    %193 = buffer %188#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer65"} : <i1>
    %194 = buffer %104#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer66"} : <i5>
    %trueResult_31, %falseResult_32 = cond_br %188#2, %196 {handshake.bb = 3 : ui32, handshake.name = "cond_br10"} : <i1>, <i5>
    %196 = buffer %110#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer68"} : <i5>
    %trueResult_33, %falseResult_34 = cond_br %197, %117#1 {handshake.bb = 3 : ui32, handshake.name = "cond_br11"} : <i1>, <>
    %197 = buffer %188#8, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer69"} : <i1>
    %trueResult_35, %falseResult_36 = cond_br %198, %falseResult_11 {handshake.bb = 4 : ui32, handshake.name = "cond_br85"} : <i1>, <>
    %198 = buffer %217#3, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer70"} : <i1>
    %trueResult_37, %falseResult_38 = cond_br %199, %66#0 {handshake.bb = 4 : ui32, handshake.name = "cond_br86"} : <i1>, <>
    %199 = buffer %217#2, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer71"} : <i1>
    sink %trueResult_37 {handshake.name = "sink2"} : <>
    %200 = merge %falseResult_26 {handshake.bb = 4 : ui32, handshake.name = "merge0"} : <i32>
    %201 = merge %falseResult_28 {handshake.bb = 4 : ui32, handshake.name = "merge1"} : <i32>
    %202 = merge %falseResult_30 {handshake.bb = 4 : ui32, handshake.name = "merge2"} : <i5>
    %203 = merge %falseResult_32 {handshake.bb = 4 : ui32, handshake.name = "merge3"} : <i5>
    %204 = extsi %203 {handshake.bb = 4 : ui32, handshake.name = "extsi45"} : <i5> to <i6>
    %result_39, %index_40 = control_merge [%falseResult_34]  {handshake.bb = 4 : ui32, handshake.name = "control_merge3"} : [<>] to <>, <i1>
    sink %index_40 {handshake.name = "sink3"} : <i1>
    %205 = source {handshake.bb = 4 : ui32, handshake.name = "source5"} : <>
    %206 = constant %205 {handshake.bb = 4 : ui32, handshake.name = "constant39", value = 10 : i5} : <>, <i5>
    %207 = extsi %206 {handshake.bb = 4 : ui32, handshake.name = "extsi46"} : <i5> to <i6>
    %208 = source {handshake.bb = 4 : ui32, handshake.name = "source6"} : <>
    %209 = constant %208 {handshake.bb = 4 : ui32, handshake.name = "constant40", value = 1 : i2} : <>, <i2>
    %210 = extsi %209 {handshake.bb = 4 : ui32, handshake.name = "extsi47"} : <i2> to <i6>
    %211 = addi %204, %210 {handshake.bb = 4 : ui32, handshake.name = "addi14"} : <i6>
    %212:2 = fork [2] %211 {handshake.bb = 4 : ui32, handshake.name = "fork26"} : <i6>
    %213 = trunci %212#0 {handshake.bb = 4 : ui32, handshake.name = "trunci9"} : <i6> to <i5>
    %215 = cmpi ult, %212#1, %207 {handshake.bb = 4 : ui32, handshake.name = "cmpi1"} : <i6>
    %217:8 = fork [8] %215 {handshake.bb = 4 : ui32, handshake.name = "fork27"} : <i1>
    %trueResult_41, %falseResult_42 = cond_br %217#0, %213 {handshake.bb = 4 : ui32, handshake.name = "cond_br12"} : <i1>, <i5>
    sink %falseResult_42 {handshake.name = "sink4"} : <i5>
    %trueResult_43, %falseResult_44 = cond_br %219, %200 {handshake.bb = 4 : ui32, handshake.name = "cond_br13"} : <i1>, <i32>
    %219 = buffer %217#5, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer75"} : <i1>
    %trueResult_45, %falseResult_46 = cond_br %220, %201 {handshake.bb = 4 : ui32, handshake.name = "cond_br14"} : <i1>, <i32>
    %220 = buffer %217#6, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer76"} : <i1>
    %trueResult_47, %falseResult_48 = cond_br %217#1, %202 {handshake.bb = 4 : ui32, handshake.name = "cond_br15"} : <i1>, <i5>
    %trueResult_49, %falseResult_50 = cond_br %222, %result_39 {handshake.bb = 4 : ui32, handshake.name = "cond_br16"} : <i1>, <>
    %222 = buffer %217#7, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer78"} : <i1>
    %trueResult_51, %falseResult_52 = cond_br %243#2, %falseResult_38 {handshake.bb = 5 : ui32, handshake.name = "cond_br87"} : <i1>, <>
    sink %trueResult_51 {handshake.name = "sink5"} : <>
    %trueResult_53, %falseResult_54 = cond_br %224, %falseResult_36 {handshake.bb = 5 : ui32, handshake.name = "cond_br88"} : <i1>, <>
    %224 = buffer %243#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 5 : ui32, handshake.name = "buffer80"} : <i1>
    %225 = merge %falseResult_44 {handshake.bb = 5 : ui32, handshake.name = "merge4"} : <i32>
    %226 = merge %falseResult_46 {handshake.bb = 5 : ui32, handshake.name = "merge5"} : <i32>
    %227 = merge %falseResult_48 {handshake.bb = 5 : ui32, handshake.name = "merge6"} : <i5>
    %228 = extsi %227 {handshake.bb = 5 : ui32, handshake.name = "extsi48"} : <i5> to <i6>
    %result_55, %index_56 = control_merge [%falseResult_50]  {handshake.bb = 5 : ui32, handshake.name = "control_merge4"} : [<>] to <>, <i1>
    sink %index_56 {handshake.name = "sink6"} : <i1>
    %229:2 = fork [2] %result_55 {handshake.bb = 5 : ui32, handshake.name = "fork28"} : <>
    %230 = constant %229#0 {handshake.bb = 5 : ui32, handshake.name = "constant41", value = false} : <>, <i1>
    %231 = source {handshake.bb = 5 : ui32, handshake.name = "source7"} : <>
    %232 = constant %231 {handshake.bb = 5 : ui32, handshake.name = "constant42", value = 10 : i5} : <>, <i5>
    %233 = extsi %232 {handshake.bb = 5 : ui32, handshake.name = "extsi49"} : <i5> to <i6>
    %234 = source {handshake.bb = 5 : ui32, handshake.name = "source8"} : <>
    %235 = constant %234 {handshake.bb = 5 : ui32, handshake.name = "constant43", value = 1 : i2} : <>, <i2>
    %236 = extsi %235 {handshake.bb = 5 : ui32, handshake.name = "extsi50"} : <i2> to <i6>
    %237 = addi %228, %236 {handshake.bb = 5 : ui32, handshake.name = "addi15"} : <i6>
    %238:2 = fork [2] %237 {handshake.bb = 5 : ui32, handshake.name = "fork29"} : <i6>
    %239 = trunci %238#0 {handshake.bb = 5 : ui32, handshake.name = "trunci10"} : <i6> to <i5>
    %241 = cmpi ult, %238#1, %233 {handshake.bb = 5 : ui32, handshake.name = "cmpi2"} : <i6>
    %243:8 = fork [8] %241 {handshake.bb = 5 : ui32, handshake.name = "fork30"} : <i1>
    %trueResult_57, %falseResult_58 = cond_br %243#0, %239 {handshake.bb = 5 : ui32, handshake.name = "cond_br17"} : <i1>, <i5>
    sink %falseResult_58 {handshake.name = "sink7"} : <i5>
    %trueResult_59, %falseResult_60 = cond_br %243#4, %225 {handshake.bb = 5 : ui32, handshake.name = "cond_br18"} : <i1>, <i32>
    sink %falseResult_60 {handshake.name = "sink8"} : <i32>
    %trueResult_61, %falseResult_62 = cond_br %243#5, %226 {handshake.bb = 5 : ui32, handshake.name = "cond_br19"} : <i1>, <i32>
    %trueResult_63, %falseResult_64 = cond_br %243#6, %229#1 {handshake.bb = 5 : ui32, handshake.name = "cond_br20"} : <i1>, <>
    %trueResult_65, %falseResult_66 = cond_br %243#7, %230 {handshake.bb = 5 : ui32, handshake.name = "cond_br21"} : <i1>, <i1>
    sink %trueResult_65 {handshake.name = "sink9"} : <i1>
    %249 = extsi %falseResult_66 {handshake.bb = 5 : ui32, handshake.name = "extsi31"} : <i1> to <i5>
    %250 = init %517#4 {ftd.imerge, handshake.bb = 6 : ui32, handshake.name = "init14"} : <i1>
    %252:3 = fork [3] %250 {handshake.bb = 6 : ui32, handshake.name = "fork31"} : <i1>
    %253 = mux %254 [%falseResult_54, %trueResult_127] {ftd.regen, handshake.bb = 6 : ui32, handshake.name = "mux35"} : <i1>, [<>, <>] to <>
    %254 = buffer %252#2, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 6 : ui32, handshake.name = "buffer89"} : <i1>
    %255:2 = fork [2] %253 {handshake.bb = 6 : ui32, handshake.name = "fork32"} : <>
    %256 = mux %257 [%falseResult_52, %trueResult_125] {ftd.regen, handshake.bb = 6 : ui32, handshake.name = "mux36"} : <i1>, [<>, <>] to <>
    %257 = buffer %252#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 6 : ui32, handshake.name = "buffer90"} : <i1>
    %258:2 = fork [2] %256 {handshake.bb = 6 : ui32, handshake.name = "fork33"} : <>
    %259 = mux %260 [%0#2, %trueResult_123] {ftd.phi, handshake.bb = 6 : ui32, handshake.name = "mux40"} : <i1>, [<>, <>] to <>
    %260 = buffer %252#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 6 : ui32, handshake.name = "buffer91"} : <i1>
    %261 = mux %265#0 [%249, %trueResult_131] {handshake.bb = 6 : ui32, handshake.name = "mux12"} : <i1>, [<i5>, <i5>] to <i5>
    %263 = mux %265#1 [%falseResult_62, %trueResult_133] {handshake.bb = 6 : ui32, handshake.name = "mux13"} : <i1>, [<i32>, <i32>] to <i32>
    %result_67, %index_68 = control_merge [%falseResult_64, %trueResult_135]  {handshake.bb = 6 : ui32, handshake.name = "control_merge5"} : [<>, <>] to <>, <i1>
    %265:2 = fork [2] %index_68 {handshake.bb = 6 : ui32, handshake.name = "fork34"} : <i1>
    %266:2 = fork [2] %result_67 {handshake.bb = 6 : ui32, handshake.name = "fork35"} : <>
    %267 = constant %266#0 {handshake.bb = 6 : ui32, handshake.name = "constant44", value = false} : <>, <i1>
    %268 = br %267 {handshake.bb = 6 : ui32, handshake.name = "br20"} : <i1>
    %269 = extsi %268 {handshake.bb = 6 : ui32, handshake.name = "extsi30"} : <i1> to <i5>
    %270 = br %263 {handshake.bb = 6 : ui32, handshake.name = "br21"} : <i32>
    %271 = br %261 {handshake.bb = 6 : ui32, handshake.name = "br22"} : <i5>
    %272 = br %266#1 {handshake.bb = 6 : ui32, handshake.name = "br23"} : <>
    %273 = init %494#5 {ftd.imerge, handshake.bb = 7 : ui32, handshake.name = "init20"} : <i1>
    %275:3 = fork [3] %273 {handshake.bb = 7 : ui32, handshake.name = "fork36"} : <i1>
    %276 = mux %277 [%255#1, %trueResult_109] {ftd.regen, handshake.bb = 7 : ui32, handshake.name = "mux41"} : <i1>, [<>, <>] to <>
    %277 = buffer %275#2, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 7 : ui32, handshake.name = "buffer95"} : <i1>
    %278:2 = fork [2] %276 {handshake.bb = 7 : ui32, handshake.name = "fork37"} : <>
    %279 = mux %280 [%258#1, %trueResult_107] {ftd.regen, handshake.bb = 7 : ui32, handshake.name = "mux42"} : <i1>, [<>, <>] to <>
    %280 = buffer %275#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 7 : ui32, handshake.name = "buffer96"} : <i1>
    %281:2 = fork [2] %279 {handshake.bb = 7 : ui32, handshake.name = "fork38"} : <>
    %282 = mux %283 [%259, %trueResult_111] {ftd.phi, handshake.bb = 7 : ui32, handshake.name = "mux46"} : <i1>, [<>, <>] to <>
    %283 = buffer %275#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 7 : ui32, handshake.name = "buffer97"} : <i1>
    %284:2 = unbundle %285  {handshake.bb = 7 : ui32, handshake.name = "unbundle2"} : <i32> to _ 
    %285 = buffer %326#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 7 : ui32, handshake.name = "buffer98"} : <i32>
    %286 = mux %302#1 [%269, %trueResult_115] {handshake.bb = 7 : ui32, handshake.name = "mux14"} : <i1>, [<i5>, <i5>] to <i5>
    %288:3 = fork [3] %286 {handshake.bb = 7 : ui32, handshake.name = "fork39"} : <i5>
    %289 = extsi %288#0 {handshake.bb = 7 : ui32, handshake.name = "extsi51"} : <i5> to <i7>
    %291 = extsi %292 {handshake.bb = 7 : ui32, handshake.name = "extsi52"} : <i5> to <i7>
    %292 = buffer %288#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 7 : ui32, handshake.name = "buffer101"} : <i5>
    %293 = mux %302#2 [%270, %trueResult_117] {handshake.bb = 7 : ui32, handshake.name = "mux15"} : <i1>, [<i32>, <i32>] to <i32>
    %295:2 = fork [2] %293 {handshake.bb = 7 : ui32, handshake.name = "fork40"} : <i32>
    %296 = mux %302#0 [%271, %trueResult_119] {handshake.bb = 7 : ui32, handshake.name = "mux16"} : <i1>, [<i5>, <i5>] to <i5>
    %298:2 = fork [2] %296 {handshake.bb = 7 : ui32, handshake.name = "fork41"} : <i5>
    %299 = extsi %298#1 {handshake.bb = 7 : ui32, handshake.name = "extsi53"} : <i5> to <i32>
    %301:4 = fork [4] %299 {handshake.bb = 7 : ui32, handshake.name = "fork42"} : <i32>
    %result_69, %index_70 = control_merge [%272, %trueResult_121]  {handshake.bb = 7 : ui32, handshake.name = "control_merge6"} : [<>, <>] to <>, <i1>
    %302:3 = fork [3] %index_70 {handshake.bb = 7 : ui32, handshake.name = "fork43"} : <i1>
    %303:3 = fork [3] %result_69 {handshake.bb = 7 : ui32, handshake.name = "fork44"} : <>
    %304 = constant %303#1 {handshake.bb = 7 : ui32, handshake.name = "constant45", value = 1 : i2} : <>, <i2>
    %305 = extsi %304 {handshake.bb = 7 : ui32, handshake.name = "extsi17"} : <i2> to <i32>
    %306 = constant %303#0 {handshake.bb = 7 : ui32, handshake.name = "constant46", value = false} : <>, <i1>
    %307 = source {handshake.bb = 7 : ui32, handshake.name = "source9"} : <>
    %308 = constant %307 {handshake.bb = 7 : ui32, handshake.name = "constant47", value = 1 : i2} : <>, <i2>
    %309 = extsi %308 {handshake.bb = 7 : ui32, handshake.name = "extsi19"} : <i2> to <i32>
    %310:2 = fork [2] %309 {handshake.bb = 7 : ui32, handshake.name = "fork45"} : <i32>
    %311 = source {handshake.bb = 7 : ui32, handshake.name = "source10"} : <>
    %312 = constant %311 {handshake.bb = 7 : ui32, handshake.name = "constant48", value = 3 : i3} : <>, <i3>
    %313 = extsi %312 {handshake.bb = 7 : ui32, handshake.name = "extsi20"} : <i3> to <i32>
    %314:2 = fork [2] %313 {handshake.bb = 7 : ui32, handshake.name = "fork46"} : <i32>
    %315 = shli %301#0, %310#0 {handshake.bb = 7 : ui32, handshake.name = "shli10"} : <i32>
    %318 = trunci %315 {handshake.bb = 7 : ui32, handshake.name = "trunci11"} : <i32> to <i7>
    %319 = shli %301#1, %314#0 {handshake.bb = 7 : ui32, handshake.name = "shli11"} : <i32>
    %322 = trunci %319 {handshake.bb = 7 : ui32, handshake.name = "trunci12"} : <i32> to <i7>
    %323 = addi %318, %322 {handshake.bb = 7 : ui32, handshake.name = "addi24"} : <i7>
    %324 = addi %289, %323 {handshake.bb = 7 : ui32, handshake.name = "addi7"} : <i7>
    %325 = buffer %284#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 7 : ui32, handshake.name = "buffer2"} : <>
    %addressResult_71, %dataResult_72 = load[%324] %outputs#0 {handshake.bb = 7 : ui32, handshake.deps = #handshake<deps[["store2", 3, false], ["store3", 3, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load3"} : <i7>, <i32>, <i7>, <i32>
    %326:2 = fork [2] %dataResult_72 {handshake.bb = 7 : ui32, handshake.name = "fork47"} : <i32>
    %327 = muli %326#1, %329 {handshake.bb = 7 : ui32, handshake.name = "muli2"} : <i32>
    %329 = buffer %295#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 7 : ui32, handshake.name = "buffer110"} : <i32>
    %330 = shli %332, %331 {handshake.bb = 7 : ui32, handshake.name = "shli12"} : <i32>
    %331 = buffer %310#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 7 : ui32, handshake.name = "buffer111"} : <i32>
    %332 = buffer %301#2, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 7 : ui32, handshake.name = "buffer112"} : <i32>
    %333 = trunci %330 {handshake.bb = 7 : ui32, handshake.name = "trunci13"} : <i32> to <i7>
    %334 = shli %301#3, %335 {handshake.bb = 7 : ui32, handshake.name = "shli13"} : <i32>
    %335 = buffer %314#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 7 : ui32, handshake.name = "buffer113"} : <i32>
    %337 = trunci %334 {handshake.bb = 7 : ui32, handshake.name = "trunci14"} : <i32> to <i7>
    %338 = addi %333, %337 {handshake.bb = 7 : ui32, handshake.name = "addi25"} : <i7>
    %339 = addi %291, %338 {handshake.bb = 7 : ui32, handshake.name = "addi8"} : <i7>
    %340 = buffer %doneResult_75, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 7 : ui32, handshake.name = "buffer3"} : <>
    %addressResult_73, %dataResult_74, %doneResult_75 = store[%339] %327 %outputs#1 {handshake.bb = 7 : ui32, handshake.deps = #handshake<deps[["load6", 3, false], ["store3", 3, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store2"} : <i7>, <i32>, <>, <i7>, <i32>, <>
    %341 = br %306 {handshake.bb = 7 : ui32, handshake.name = "br24"} : <i1>
    %342 = extsi %341 {handshake.bb = 7 : ui32, handshake.name = "extsi29"} : <i1> to <i5>
    %343 = br %295#0 {handshake.bb = 7 : ui32, handshake.name = "br25"} : <i32>
    %345 = br %298#0 {handshake.bb = 7 : ui32, handshake.name = "br26"} : <i5>
    %347 = br %288#2 {handshake.bb = 7 : ui32, handshake.name = "br27"} : <i5>
    %349 = br %303#2 {handshake.bb = 7 : ui32, handshake.name = "br28"} : <>
    %trueResult_76, %falseResult_77 = cond_br %350, %363#1 {handshake.bb = 8 : ui32, handshake.name = "cond_br89"} : <i1>, <>
    %350 = buffer %467#8, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 8 : ui32, handshake.name = "buffer118"} : <i1>
    sink %falseResult_77 {handshake.name = "sink10"} : <>
    %trueResult_78, %falseResult_79 = cond_br %351, %369#1 {handshake.bb = 8 : ui32, handshake.name = "cond_br90"} : <i1>, <>
    %351 = buffer %467#7, bufferType = FIFO_BREAK_NONE, numSlots = 4 {handshake.bb = 8 : ui32, handshake.name = "buffer119"} : <i1>
    sink %falseResult_79 {handshake.name = "sink11"} : <>
    %trueResult_80, %falseResult_81 = cond_br %352, %360#2 {handshake.bb = 8 : ui32, handshake.name = "cond_br91"} : <i1>, <>
    %352 = buffer %467#6, bufferType = FIFO_BREAK_NONE, numSlots = 3 {handshake.bb = 8 : ui32, handshake.name = "buffer120"} : <i1>
    sink %falseResult_81 {handshake.name = "sink12"} : <>
    %trueResult_82, %falseResult_83 = cond_br %467#5, %366#1 {handshake.bb = 8 : ui32, handshake.name = "cond_br92"} : <i1>, <>
    sink %falseResult_83 {handshake.name = "sink13"} : <>
    %trueResult_84, %falseResult_85 = cond_br %354, %458 {handshake.bb = 8 : ui32, handshake.name = "cond_br93"} : <i1>, <>
    %354 = buffer %467#4, bufferType = FIFO_BREAK_NONE, numSlots = 4 {handshake.bb = 8 : ui32, handshake.name = "buffer122"} : <i1>
    %355 = init %467#3 {ftd.imerge, handshake.bb = 8 : ui32, handshake.name = "init26"} : <i1>
    %357:5 = fork [5] %355 {handshake.bb = 8 : ui32, handshake.name = "fork48"} : <i1>
    %358 = mux %359 [%340, %trueResult_80] {ftd.regen, handshake.bb = 8 : ui32, handshake.name = "mux47"} : <i1>, [<>, <>] to <>
    %359 = buffer %357#4, bufferType = FIFO_BREAK_NONE, numSlots = 3 {handshake.bb = 8 : ui32, handshake.name = "buffer124"} : <i1>
    %360:3 = fork [3] %358 {handshake.bb = 8 : ui32, handshake.name = "fork49"} : <>
    %361 = mux %362 [%278#1, %trueResult_76] {ftd.regen, handshake.bb = 8 : ui32, handshake.name = "mux48"} : <i1>, [<>, <>] to <>
    %362 = buffer %357#3, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 8 : ui32, handshake.name = "buffer125"} : <i1>
    %363:2 = fork [2] %361 {handshake.bb = 8 : ui32, handshake.name = "fork50"} : <>
    %364 = mux %365 [%281#1, %trueResult_82] {ftd.regen, handshake.bb = 8 : ui32, handshake.name = "mux49"} : <i1>, [<>, <>] to <>
    %365 = buffer %357#2, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 8 : ui32, handshake.name = "buffer126"} : <i1>
    %366:2 = fork [2] %364 {handshake.bb = 8 : ui32, handshake.name = "fork51"} : <>
    %367 = mux %368 [%325, %trueResult_78] {ftd.regen, handshake.bb = 8 : ui32, handshake.name = "mux51"} : <i1>, [<>, <>] to <>
    %368 = buffer %357#1, bufferType = FIFO_BREAK_NONE, numSlots = 3 {handshake.bb = 8 : ui32, handshake.name = "buffer127"} : <i1>
    %369:2 = fork [2] %367 {handshake.bb = 8 : ui32, handshake.name = "fork52"} : <>
    %370 = mux %371 [%282, %trueResult_84] {ftd.phi, handshake.bb = 8 : ui32, handshake.name = "mux52"} : <i1>, [<>, <>] to <>
    %371 = buffer %357#0, bufferType = FIFO_BREAK_NONE, numSlots = 3 {handshake.bb = 8 : ui32, handshake.name = "buffer128"} : <i1>
    %372 = mux %396#2 [%342, %trueResult_97] {handshake.bb = 8 : ui32, handshake.name = "mux17"} : <i1>, [<i5>, <i5>] to <i5>
    %374:2 = fork [2] %372 {handshake.bb = 8 : ui32, handshake.name = "fork53"} : <i5>
    %375 = extsi %374#0 {handshake.bb = 8 : ui32, handshake.name = "extsi54"} : <i5> to <i6>
    %377 = extsi %374#1 {handshake.bb = 8 : ui32, handshake.name = "extsi55"} : <i5> to <i32>
    %379:3 = fork [3] %377 {handshake.bb = 8 : ui32, handshake.name = "fork54"} : <i32>
    %380 = mux %396#3 [%343, %trueResult_99] {handshake.bb = 8 : ui32, handshake.name = "mux18"} : <i1>, [<i32>, <i32>] to <i32>
    %382 = mux %396#0 [%345, %trueResult_101] {handshake.bb = 8 : ui32, handshake.name = "mux19"} : <i1>, [<i5>, <i5>] to <i5>
    %384:2 = fork [2] %382 {handshake.bb = 8 : ui32, handshake.name = "fork55"} : <i5>
    %385 = extsi %386 {handshake.bb = 8 : ui32, handshake.name = "extsi56"} : <i5> to <i32>
    %386 = buffer %384#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 8 : ui32, handshake.name = "buffer134"} : <i5>
    %387:6 = fork [6] %385 {handshake.bb = 8 : ui32, handshake.name = "fork56"} : <i32>
    %388 = mux %396#1 [%347, %trueResult_103] {handshake.bb = 8 : ui32, handshake.name = "mux20"} : <i1>, [<i5>, <i5>] to <i5>
    %390:3 = fork [3] %388 {handshake.bb = 8 : ui32, handshake.name = "fork57"} : <i5>
    %391 = extsi %390#0 {handshake.bb = 8 : ui32, handshake.name = "extsi57"} : <i5> to <i7>
    %393 = extsi %394 {handshake.bb = 8 : ui32, handshake.name = "extsi58"} : <i5> to <i32>
    %394 = buffer %390#2, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 8 : ui32, handshake.name = "buffer137"} : <i5>
    %395:2 = fork [2] %393 {handshake.bb = 8 : ui32, handshake.name = "fork58"} : <i32>
    %result_86, %index_87 = control_merge [%349, %trueResult_105]  {handshake.bb = 8 : ui32, handshake.name = "control_merge7"} : [<>, <>] to <>, <i1>
    %396:4 = fork [4] %index_87 {handshake.bb = 8 : ui32, handshake.name = "fork59"} : <i1>
    %397:2 = fork [2] %result_86 {handshake.bb = 8 : ui32, handshake.name = "fork60"} : <>
    %398 = constant %397#0 {handshake.bb = 8 : ui32, handshake.name = "constant49", value = 1 : i2} : <>, <i2>
    %399 = extsi %398 {handshake.bb = 8 : ui32, handshake.name = "extsi21"} : <i2> to <i32>
    %400 = source {handshake.bb = 8 : ui32, handshake.name = "source11"} : <>
    %401 = constant %400 {handshake.bb = 8 : ui32, handshake.name = "constant50", value = 10 : i5} : <>, <i5>
    %402 = extsi %401 {handshake.bb = 8 : ui32, handshake.name = "extsi59"} : <i5> to <i6>
    %403 = source {handshake.bb = 8 : ui32, handshake.name = "source12"} : <>
    %404 = constant %403 {handshake.bb = 8 : ui32, handshake.name = "constant51", value = 1 : i2} : <>, <i2>
    %405:2 = fork [2] %404 {handshake.bb = 8 : ui32, handshake.name = "fork61"} : <i2>
    %406 = extsi %407 {handshake.bb = 8 : ui32, handshake.name = "extsi60"} : <i2> to <i6>
    %407 = buffer %405#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 8 : ui32, handshake.name = "buffer138"} : <i2>
    %408 = extsi %409 {handshake.bb = 8 : ui32, handshake.name = "extsi23"} : <i2> to <i32>
    %409 = buffer %405#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 8 : ui32, handshake.name = "buffer139"} : <i2>
    %410:4 = fork [4] %408 {handshake.bb = 8 : ui32, handshake.name = "fork62"} : <i32>
    %411 = source {handshake.bb = 8 : ui32, handshake.name = "source13"} : <>
    %412 = constant %411 {handshake.bb = 8 : ui32, handshake.name = "constant52", value = 3 : i3} : <>, <i3>
    %413 = extsi %412 {handshake.bb = 8 : ui32, handshake.name = "extsi24"} : <i3> to <i32>
    %414:4 = fork [4] %413 {handshake.bb = 8 : ui32, handshake.name = "fork63"} : <i32>
    %415 = shli %417, %410#0 {handshake.bb = 8 : ui32, handshake.name = "shli14"} : <i32>
    %417 = buffer %387#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 8 : ui32, handshake.name = "buffer141"} : <i32>
    %418 = shli %420, %414#0 {handshake.bb = 8 : ui32, handshake.name = "shli15"} : <i32>
    %420 = buffer %387#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 8 : ui32, handshake.name = "buffer143"} : <i32>
    %421 = addi %415, %418 {handshake.bb = 8 : ui32, handshake.name = "addi26"} : <i32>
    %422 = addi %423, %421 {handshake.bb = 8 : ui32, handshake.name = "addi9"} : <i32>
    %423 = buffer %379#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 8 : ui32, handshake.name = "buffer144"} : <i32>
    %424 = gate %422, %366#0, %363#0 {handshake.bb = 8 : ui32, handshake.name = "gate2"} : <i32>, !handshake.control<>, !handshake.control<> to <i32>
    %425 = trunci %424 {handshake.bb = 8 : ui32, handshake.name = "trunci15"} : <i32> to <i7>
    %addressResult_88, %dataResult_89 = load[%425] %outputs_6#3 {handshake.bb = 8 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load4"} : <i7>, <i32>, <i7>, <i32>
    %426 = shli %428, %410#1 {handshake.bb = 8 : ui32, handshake.name = "shli16"} : <i32>
    %428 = buffer %379#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 8 : ui32, handshake.name = "buffer146"} : <i32>
    %429 = trunci %426 {handshake.bb = 8 : ui32, handshake.name = "trunci16"} : <i32> to <i7>
    %430 = shli %432, %414#1 {handshake.bb = 8 : ui32, handshake.name = "shli17"} : <i32>
    %432 = buffer %379#2, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 8 : ui32, handshake.name = "buffer148"} : <i32>
    %433 = trunci %430 {handshake.bb = 8 : ui32, handshake.name = "trunci17"} : <i32> to <i7>
    %434 = addi %429, %433 {handshake.bb = 8 : ui32, handshake.name = "addi27"} : <i7>
    %435 = addi %391, %434 {handshake.bb = 8 : ui32, handshake.name = "addi10"} : <i7>
    %addressResult_90, %dataResult_91 = load[%435] %outputs_0 {handshake.bb = 8 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load5"} : <i7>, <i32>, <i7>, <i32>
    %436 = muli %dataResult_89, %dataResult_91 {handshake.bb = 8 : ui32, handshake.name = "muli3"} : <i32>
    %437 = shli %439, %438 {handshake.bb = 8 : ui32, handshake.name = "shli18"} : <i32>
    %438 = buffer %410#2, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 8 : ui32, handshake.name = "buffer149"} : <i32>
    %439 = buffer %387#2, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 8 : ui32, handshake.name = "buffer150"} : <i32>
    %440 = shli %442, %441 {handshake.bb = 8 : ui32, handshake.name = "shli19"} : <i32>
    %441 = buffer %414#2, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 8 : ui32, handshake.name = "buffer151"} : <i32>
    %442 = buffer %387#3, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 8 : ui32, handshake.name = "buffer152"} : <i32>
    %443 = addi %437, %440 {handshake.bb = 8 : ui32, handshake.name = "addi28"} : <i32>
    %444 = addi %445, %443 {handshake.bb = 8 : ui32, handshake.name = "addi11"} : <i32>
    %445 = buffer %395#0, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 8 : ui32, handshake.name = "buffer153"} : <i32>
    %446 = gate %444, %370, %360#1 {handshake.bb = 8 : ui32, handshake.name = "gate3"} : <i32>, !handshake.control<>, !handshake.control<> to <i32>
    %447 = trunci %446 {handshake.bb = 8 : ui32, handshake.name = "trunci18"} : <i32> to <i7>
    %addressResult_92, %dataResult_93 = load[%447] %outputs#2 {handshake.bb = 8 : ui32, handshake.deps = #handshake<deps[["store3", 3, false], ["store3", 4, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load6"} : <i7>, <i32>, <i7>, <i32>
    %448 = addi %dataResult_93, %436 {handshake.bb = 8 : ui32, handshake.name = "addi1"} : <i32>
    %449 = shli %451, %450 {handshake.bb = 8 : ui32, handshake.name = "shli20"} : <i32>
    %450 = buffer %410#3, bufferType = FIFO_BREAK_NONE, numSlots = 3 {handshake.bb = 8 : ui32, handshake.name = "buffer154"} : <i32>
    %451 = buffer %387#4, bufferType = FIFO_BREAK_NONE, numSlots = 3 {handshake.bb = 8 : ui32, handshake.name = "buffer155"} : <i32>
    %452 = shli %454, %453 {handshake.bb = 8 : ui32, handshake.name = "shli21"} : <i32>
    %453 = buffer %414#3, bufferType = FIFO_BREAK_NONE, numSlots = 3 {handshake.bb = 8 : ui32, handshake.name = "buffer156"} : <i32>
    %454 = buffer %387#5, bufferType = FIFO_BREAK_NONE, numSlots = 3 {handshake.bb = 8 : ui32, handshake.name = "buffer157"} : <i32>
    %455 = addi %449, %452 {handshake.bb = 8 : ui32, handshake.name = "addi29"} : <i32>
    %456 = addi %457, %455 {handshake.bb = 8 : ui32, handshake.name = "addi12"} : <i32>
    %457 = buffer %395#1, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 8 : ui32, handshake.name = "buffer158"} : <i32>
    %458 = buffer %doneResult_96, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 8 : ui32, handshake.name = "buffer4"} : <>
    %459 = gate %456, %369#0, %360#0 {handshake.bb = 8 : ui32, handshake.name = "gate4"} : <i32>, !handshake.control<>, !handshake.control<> to <i32>
    %460 = trunci %459 {handshake.bb = 8 : ui32, handshake.name = "trunci19"} : <i32> to <i7>
    %addressResult_94, %dataResult_95, %doneResult_96 = store[%460] %448 %outputs#3 {handshake.bb = 8 : ui32, handshake.deps = #handshake<deps[["load6", 3, false], ["store3", 3, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store3"} : <i7>, <i32>, <>, <i7>, <i32>, <>
    %461 = addi %375, %406 {handshake.bb = 8 : ui32, handshake.name = "addi16"} : <i6>
    %462:2 = fork [2] %461 {handshake.bb = 8 : ui32, handshake.name = "fork64"} : <i6>
    %463 = trunci %464 {handshake.bb = 8 : ui32, handshake.name = "trunci20"} : <i6> to <i5>
    %464 = buffer %462#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 8 : ui32, handshake.name = "buffer159"} : <i6>
    %465 = cmpi ult, %466, %402 {handshake.bb = 8 : ui32, handshake.name = "cmpi3"} : <i6>
    %466 = buffer %462#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 8 : ui32, handshake.name = "buffer160"} : <i6>
    %467:11 = fork [11] %465 {handshake.bb = 8 : ui32, handshake.name = "fork65"} : <i1>
    %trueResult_97, %falseResult_98 = cond_br %467#0, %463 {handshake.bb = 8 : ui32, handshake.name = "cond_br22"} : <i1>, <i5>
    sink %falseResult_98 {handshake.name = "sink14"} : <i5>
    %trueResult_99, %falseResult_100 = cond_br %469, %380 {handshake.bb = 8 : ui32, handshake.name = "cond_br23"} : <i1>, <i32>
    %469 = buffer %467#9, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 8 : ui32, handshake.name = "buffer162"} : <i1>
    %trueResult_101, %falseResult_102 = cond_br %467#1, %471 {handshake.bb = 8 : ui32, handshake.name = "cond_br24"} : <i1>, <i5>
    %471 = buffer %384#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 8 : ui32, handshake.name = "buffer164"} : <i5>
    %trueResult_103, %falseResult_104 = cond_br %467#2, %473 {handshake.bb = 8 : ui32, handshake.name = "cond_br25"} : <i1>, <i5>
    %473 = buffer %390#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 8 : ui32, handshake.name = "buffer166"} : <i5>
    %trueResult_105, %falseResult_106 = cond_br %474, %397#1 {handshake.bb = 8 : ui32, handshake.name = "cond_br26"} : <i1>, <>
    %474 = buffer %467#10, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 8 : ui32, handshake.name = "buffer167"} : <i1>
    %trueResult_107, %falseResult_108 = cond_br %494#4, %281#0 {handshake.bb = 9 : ui32, handshake.name = "cond_br94"} : <i1>, <>
    sink %falseResult_108 {handshake.name = "sink15"} : <>
    %trueResult_109, %falseResult_110 = cond_br %476, %278#0 {handshake.bb = 9 : ui32, handshake.name = "cond_br95"} : <i1>, <>
    %476 = buffer %494#3, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 9 : ui32, handshake.name = "buffer169"} : <i1>
    sink %falseResult_110 {handshake.name = "sink16"} : <>
    %trueResult_111, %falseResult_112 = cond_br %477, %falseResult_85 {handshake.bb = 9 : ui32, handshake.name = "cond_br96"} : <i1>, <>
    %477 = buffer %494#2, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 9 : ui32, handshake.name = "buffer170"} : <i1>
    %478 = merge %falseResult_100 {handshake.bb = 9 : ui32, handshake.name = "merge7"} : <i32>
    %479 = merge %falseResult_102 {handshake.bb = 9 : ui32, handshake.name = "merge8"} : <i5>
    %480 = merge %falseResult_104 {handshake.bb = 9 : ui32, handshake.name = "merge9"} : <i5>
    %481 = extsi %480 {handshake.bb = 9 : ui32, handshake.name = "extsi61"} : <i5> to <i6>
    %result_113, %index_114 = control_merge [%falseResult_106]  {handshake.bb = 9 : ui32, handshake.name = "control_merge8"} : [<>] to <>, <i1>
    sink %index_114 {handshake.name = "sink17"} : <i1>
    %482 = source {handshake.bb = 9 : ui32, handshake.name = "source14"} : <>
    %483 = constant %482 {handshake.bb = 9 : ui32, handshake.name = "constant53", value = 10 : i5} : <>, <i5>
    %484 = extsi %483 {handshake.bb = 9 : ui32, handshake.name = "extsi62"} : <i5> to <i6>
    %485 = source {handshake.bb = 9 : ui32, handshake.name = "source15"} : <>
    %486 = constant %485 {handshake.bb = 9 : ui32, handshake.name = "constant54", value = 1 : i2} : <>, <i2>
    %487 = extsi %486 {handshake.bb = 9 : ui32, handshake.name = "extsi63"} : <i2> to <i6>
    %488 = addi %481, %487 {handshake.bb = 9 : ui32, handshake.name = "addi17"} : <i6>
    %489:2 = fork [2] %488 {handshake.bb = 9 : ui32, handshake.name = "fork66"} : <i6>
    %490 = trunci %489#0 {handshake.bb = 9 : ui32, handshake.name = "trunci21"} : <i6> to <i5>
    %492 = cmpi ult, %489#1, %484 {handshake.bb = 9 : ui32, handshake.name = "cmpi4"} : <i6>
    %494:8 = fork [8] %492 {handshake.bb = 9 : ui32, handshake.name = "fork67"} : <i1>
    %trueResult_115, %falseResult_116 = cond_br %494#0, %490 {handshake.bb = 9 : ui32, handshake.name = "cond_br27"} : <i1>, <i5>
    sink %falseResult_116 {handshake.name = "sink18"} : <i5>
    %trueResult_117, %falseResult_118 = cond_br %496, %478 {handshake.bb = 9 : ui32, handshake.name = "cond_br28"} : <i1>, <i32>
    %496 = buffer %494#6, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 9 : ui32, handshake.name = "buffer174"} : <i1>
    %trueResult_119, %falseResult_120 = cond_br %494#1, %479 {handshake.bb = 9 : ui32, handshake.name = "cond_br29"} : <i1>, <i5>
    %trueResult_121, %falseResult_122 = cond_br %498, %result_113 {handshake.bb = 9 : ui32, handshake.name = "cond_br30"} : <i1>, <>
    %498 = buffer %494#7, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 9 : ui32, handshake.name = "buffer176"} : <i1>
    %trueResult_123, %falseResult_124 = cond_br %499, %falseResult_112 {handshake.bb = 10 : ui32, handshake.name = "cond_br97"} : <i1>, <>
    %499 = buffer %517#3, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 10 : ui32, handshake.name = "buffer177"} : <i1>
    sink %falseResult_124 {handshake.name = "sink19"} : <>
    %trueResult_125, %falseResult_126 = cond_br %517#2, %258#0 {handshake.bb = 10 : ui32, handshake.name = "cond_br98"} : <i1>, <>
    sink %falseResult_126 {handshake.name = "sink20"} : <>
    %trueResult_127, %falseResult_128 = cond_br %517#1, %255#0 {handshake.bb = 10 : ui32, handshake.name = "cond_br99"} : <i1>, <>
    sink %falseResult_128 {handshake.name = "sink21"} : <>
    %502 = merge %falseResult_118 {handshake.bb = 10 : ui32, handshake.name = "merge10"} : <i32>
    %503 = merge %falseResult_120 {handshake.bb = 10 : ui32, handshake.name = "merge11"} : <i5>
    %504 = extsi %503 {handshake.bb = 10 : ui32, handshake.name = "extsi64"} : <i5> to <i6>
    %result_129, %index_130 = control_merge [%falseResult_122]  {handshake.bb = 10 : ui32, handshake.name = "control_merge9"} : [<>] to <>, <i1>
    sink %index_130 {handshake.name = "sink22"} : <i1>
    %505 = source {handshake.bb = 10 : ui32, handshake.name = "source16"} : <>
    %506 = constant %505 {handshake.bb = 10 : ui32, handshake.name = "constant55", value = 10 : i5} : <>, <i5>
    %507 = extsi %506 {handshake.bb = 10 : ui32, handshake.name = "extsi65"} : <i5> to <i6>
    %508 = source {handshake.bb = 10 : ui32, handshake.name = "source17"} : <>
    %509 = constant %508 {handshake.bb = 10 : ui32, handshake.name = "constant56", value = 1 : i2} : <>, <i2>
    %510 = extsi %509 {handshake.bb = 10 : ui32, handshake.name = "extsi66"} : <i2> to <i6>
    %511 = addi %504, %510 {handshake.bb = 10 : ui32, handshake.name = "addi18"} : <i6>
    %512:2 = fork [2] %511 {handshake.bb = 10 : ui32, handshake.name = "fork68"} : <i6>
    %513 = trunci %512#0 {handshake.bb = 10 : ui32, handshake.name = "trunci22"} : <i6> to <i5>
    %515 = cmpi ult, %512#1, %507 {handshake.bb = 10 : ui32, handshake.name = "cmpi5"} : <i6>
    %517:7 = fork [7] %515 {handshake.bb = 10 : ui32, handshake.name = "fork69"} : <i1>
    %trueResult_131, %falseResult_132 = cond_br %517#0, %513 {handshake.bb = 10 : ui32, handshake.name = "cond_br31"} : <i1>, <i5>
    sink %falseResult_132 {handshake.name = "sink23"} : <i5>
    %trueResult_133, %falseResult_134 = cond_br %517#5, %502 {handshake.bb = 10 : ui32, handshake.name = "cond_br32"} : <i1>, <i32>
    sink %falseResult_134 {handshake.name = "sink24"} : <i32>
    %trueResult_135, %falseResult_136 = cond_br %520, %result_129 {handshake.bb = 10 : ui32, handshake.name = "cond_br33"} : <i1>, <>
    %520 = buffer %517#6, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 10 : ui32, handshake.name = "buffer184"} : <i1>
    %result_137, %index_138 = control_merge [%falseResult_136]  {handshake.bb = 11 : ui32, handshake.name = "control_merge10"} : [<>] to <>, <i1>
    sink %index_138 {handshake.name = "sink25"} : <i1>
    %521:5 = fork [5] %result_137 {handshake.bb = 11 : ui32, handshake.name = "fork70"} : <>
    end {handshake.bb = 11 : ui32, handshake.name = "end0"} %memEnd_7, %memEnd_5, %memEnd_3, %memEnd_1, %memEnd, %0#1 : <>, <>, <>, <>, <>, <>
  }
}

