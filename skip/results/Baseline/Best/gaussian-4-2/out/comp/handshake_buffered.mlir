module {
  handshake.func @gaussian(%arg0: memref<20xi32>, %arg1: memref<400xi32>, %arg2: !handshake.control<>, %arg3: !handshake.control<>, %arg4: !handshake.control<>, ...) -> (!handshake.channel<i32>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["c", "a", "c_start", "a_start", "start"], resNames = ["out0", "c_end", "a_end", "end"]} {
    %0:4 = fork [4] %arg4 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork0"} : <>
    %1:3 = lsq[%arg1 : memref<400xi32>] (%arg3, %75#0, %addressResult, %addressResult_16, %addressResult_18, %dataResult_19, %167#1)  {groupSizes = [3 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "5": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "6": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "lsq1"} : (!handshake.control<>, !handshake.control<>, !handshake.channel<i9>, !handshake.channel<i9>, !handshake.channel<i9>, !handshake.channel<i32>, !handshake.control<>) -> (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>)
    %outputs, %memEnd = mem_controller[%arg0 : memref<20xi32>] %arg2 (%addressResult_14) %167#0 {connectedBlocks = [3 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller1"} :    (!handshake.channel<i5>) -> !handshake.channel<i32>
    %2 = constant %0#1 {handshake.bb = 0 : ui32, handshake.name = "constant0", value = false} : <>, <i1>
    %3 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant1", value = 1 : i2} : <>, <i2>
    %4 = br %3 {handshake.bb = 0 : ui32, handshake.name = "br2"} : <i2>
    %5 = extsi %4 {handshake.bb = 0 : ui32, handshake.name = "extsi15"} : <i2> to <i6>
    %6 = br %2 {handshake.bb = 0 : ui32, handshake.name = "br4"} : <i1>
    %7 = extsi %6 {handshake.bb = 0 : ui32, handshake.name = "extsi16"} : <i1> to <i32>
    %8 = br %0#3 {handshake.bb = 0 : ui32, handshake.name = "br5"} : <>
    %9 = mux %16#0 [%5, %trueResult_36] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i6>, <i6>] to <i6>
    %10 = buffer %9, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer0"} : <i6>
    %11 = buffer %10, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer1"} : <i6>
    %12:2 = fork [2] %11 {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <i6>
    %13 = extsi %12#1 {handshake.bb = 1 : ui32, handshake.name = "extsi17"} : <i6> to <i7>
    %14 = mux %16#1 [%7, %trueResult_38] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i1>, [<i32>, <i32>] to <i32>
    %15 = buffer %trueResult_40, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer39"} : <>
    %result, %index = control_merge [%8, %15]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %16:2 = fork [2] %index {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <i1>
    %17 = source {handshake.bb = 1 : ui32, handshake.name = "source0"} : <>
    %18 = constant %17 {handshake.bb = 1 : ui32, handshake.name = "constant3", value = 1 : i2} : <>, <i2>
    %19 = extsi %18 {handshake.bb = 1 : ui32, handshake.name = "extsi18"} : <i2> to <i7>
    %20 = addi %13, %19 {handshake.bb = 1 : ui32, handshake.name = "addi2"} : <i7>
    %21 = br %20 {handshake.bb = 1 : ui32, handshake.name = "br6"} : <i7>
    %22 = buffer %14, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer2"} : <i32>
    %23 = buffer %22, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer3"} : <i32>
    %24 = br %23 {handshake.bb = 1 : ui32, handshake.name = "br7"} : <i32>
    %25 = br %12#0 {handshake.bb = 1 : ui32, handshake.name = "br8"} : <i6>
    %26 = br %result {handshake.bb = 1 : ui32, handshake.name = "br9"} : <>
    %27 = mux %34#1 [%21, %147] {handshake.bb = 2 : ui32, handshake.name = "mux2"} : <i1>, [<i7>, <i7>] to <i7>
    %28 = buffer %27, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer4"} : <i7>
    %29 = buffer %28, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer5"} : <i7>
    %30:2 = fork [2] %29 {handshake.bb = 2 : ui32, handshake.name = "fork3"} : <i7>
    %31 = trunci %30#0 {handshake.bb = 2 : ui32, handshake.name = "trunci0"} : <i7> to <i6>
    %32 = mux %34#2 [%24, %148] {handshake.bb = 2 : ui32, handshake.name = "mux3"} : <i1>, [<i32>, <i32>] to <i32>
    %33 = mux %34#0 [%25, %149] {handshake.bb = 2 : ui32, handshake.name = "mux4"} : <i1>, [<i6>, <i6>] to <i6>
    %result_0, %index_1 = control_merge [%26, %150]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>, <>] to <>, <i1>
    %34:3 = fork [3] %index_1 {handshake.bb = 2 : ui32, handshake.name = "fork4"} : <i1>
    %35:2 = fork [2] %result_0 {handshake.bb = 2 : ui32, handshake.name = "fork5"} : <>
    %36 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %37 = constant %36 {handshake.bb = 2 : ui32, handshake.name = "constant5", value = 19 : i6} : <>, <i6>
    %38 = extsi %37 {handshake.bb = 2 : ui32, handshake.name = "extsi19"} : <i6> to <i7>
    %39 = constant %35#0 {handshake.bb = 2 : ui32, handshake.name = "constant6", value = 1 : i2} : <>, <i2>
    %40:2 = fork [2] %39 {handshake.bb = 2 : ui32, handshake.name = "fork6"} : <i2>
    %41 = cmpi ult, %30#1, %38 {handshake.bb = 2 : ui32, handshake.name = "cmpi2"} : <i7>
    %42 = buffer %41, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer10"} : <i1>
    %43:6 = fork [6] %42 {handshake.bb = 2 : ui32, handshake.name = "fork7"} : <i1>
    %trueResult, %falseResult = cond_br %43#5, %40#0 {handshake.bb = 2 : ui32, handshake.name = "cond_br3"} : <i1>, <i2>
    sink %falseResult {handshake.name = "sink0"} : <i2>
    %44 = extsi %trueResult {handshake.bb = 2 : ui32, handshake.name = "extsi14"} : <i2> to <i6>
    %trueResult_2, %falseResult_3 = cond_br %43#4, %40#1 {handshake.bb = 2 : ui32, handshake.name = "cond_br4"} : <i1>, <i2>
    sink %falseResult_3 {handshake.name = "sink1"} : <i2>
    %45 = extsi %trueResult_2 {handshake.bb = 2 : ui32, handshake.name = "extsi20"} : <i2> to <i32>
    %46 = buffer %32, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer6"} : <i32>
    %47 = buffer %46, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer7"} : <i32>
    %trueResult_4, %falseResult_5 = cond_br %43#2, %47 {handshake.bb = 2 : ui32, handshake.name = "cond_br5"} : <i1>, <i32>
    %48 = buffer %33, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer8"} : <i6>
    %49 = buffer %48, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer9"} : <i6>
    %trueResult_6, %falseResult_7 = cond_br %43#1, %49 {handshake.bb = 2 : ui32, handshake.name = "cond_br6"} : <i1>, <i6>
    %trueResult_8, %falseResult_9 = cond_br %43#0, %31 {handshake.bb = 2 : ui32, handshake.name = "cond_br7"} : <i1>, <i6>
    sink %falseResult_9 {handshake.name = "sink2"} : <i6>
    %trueResult_10, %falseResult_11 = cond_br %43#3, %35#1 {handshake.bb = 2 : ui32, handshake.name = "cond_br8"} : <i1>, <>
    %50 = mux %74#2 [%44, %trueResult_20] {handshake.bb = 3 : ui32, handshake.name = "mux5"} : <i1>, [<i6>, <i6>] to <i6>
    %51 = buffer %50, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer11"} : <i6>
    %52 = extsi %51 {handshake.bb = 3 : ui32, handshake.name = "extsi21"} : <i6> to <i7>
    %53 = mux %74#3 [%45, %trueResult_22] {handshake.bb = 3 : ui32, handshake.name = "mux6"} : <i1>, [<i32>, <i32>] to <i32>
    %54 = buffer %53, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer12"} : <i32>
    %55 = buffer %54, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer13"} : <i32>
    %56:5 = fork [5] %55 {handshake.bb = 3 : ui32, handshake.name = "fork8"} : <i32>
    %57 = trunci %56#0 {handshake.bb = 3 : ui32, handshake.name = "trunci1"} : <i32> to <i9>
    %58 = trunci %56#1 {handshake.bb = 3 : ui32, handshake.name = "trunci2"} : <i32> to <i9>
    %59 = trunci %56#2 {handshake.bb = 3 : ui32, handshake.name = "trunci3"} : <i32> to <i9>
    %60 = mux %74#4 [%trueResult_4, %trueResult_24] {handshake.bb = 3 : ui32, handshake.name = "mux7"} : <i1>, [<i32>, <i32>] to <i32>
    %61 = mux %74#0 [%trueResult_6, %trueResult_26] {handshake.bb = 3 : ui32, handshake.name = "mux8"} : <i1>, [<i6>, <i6>] to <i6>
    %62 = buffer %61, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer15"} : <i6>
    %63 = buffer %62, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer16"} : <i6>
    %64:3 = fork [3] %63 {handshake.bb = 3 : ui32, handshake.name = "fork9"} : <i6>
    %65 = extsi %64#2 {handshake.bb = 3 : ui32, handshake.name = "extsi22"} : <i6> to <i32>
    %66:2 = fork [2] %65 {handshake.bb = 3 : ui32, handshake.name = "fork10"} : <i32>
    %67 = trunci %64#0 {handshake.bb = 3 : ui32, handshake.name = "trunci4"} : <i6> to <i5>
    %68 = mux %74#1 [%trueResult_8, %trueResult_28] {handshake.bb = 3 : ui32, handshake.name = "mux9"} : <i1>, [<i6>, <i6>] to <i6>
    %69 = buffer %68, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer17"} : <i6>
    %70 = buffer %69, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer18"} : <i6>
    %71:2 = fork [2] %70 {handshake.bb = 3 : ui32, handshake.name = "fork11"} : <i6>
    %72 = extsi %71#1 {handshake.bb = 3 : ui32, handshake.name = "extsi23"} : <i6> to <i32>
    %73:4 = fork [4] %72 {handshake.bb = 3 : ui32, handshake.name = "fork12"} : <i32>
    %result_12, %index_13 = control_merge [%trueResult_10, %trueResult_30]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>, <>] to <>, <i1>
    %74:5 = fork [5] %index_13 {handshake.bb = 3 : ui32, handshake.name = "fork13"} : <i1>
    %75:2 = fork [2] %result_12 {handshake.bb = 3 : ui32, handshake.name = "fork14"} : <>
    %76 = source {handshake.bb = 3 : ui32, handshake.name = "source2"} : <>
    %77 = constant %76 {handshake.bb = 3 : ui32, handshake.name = "constant19", value = 20 : i6} : <>, <i6>
    %78 = extsi %77 {handshake.bb = 3 : ui32, handshake.name = "extsi24"} : <i6> to <i7>
    %79 = source {handshake.bb = 3 : ui32, handshake.name = "source3"} : <>
    %80 = constant %79 {handshake.bb = 3 : ui32, handshake.name = "constant20", value = 1 : i2} : <>, <i2>
    %81:2 = fork [2] %80 {handshake.bb = 3 : ui32, handshake.name = "fork15"} : <i2>
    %82 = extsi %81#0 {handshake.bb = 3 : ui32, handshake.name = "extsi25"} : <i2> to <i7>
    %83 = extsi %81#1 {handshake.bb = 3 : ui32, handshake.name = "extsi7"} : <i2> to <i32>
    %84 = source {handshake.bb = 3 : ui32, handshake.name = "source5"} : <>
    %85 = constant %84 {handshake.bb = 3 : ui32, handshake.name = "constant21", value = 4 : i4} : <>, <i4>
    %86 = extsi %85 {handshake.bb = 3 : ui32, handshake.name = "extsi9"} : <i4> to <i32>
    %87:3 = fork [3] %86 {handshake.bb = 3 : ui32, handshake.name = "fork16"} : <i32>
    %88 = source {handshake.bb = 3 : ui32, handshake.name = "source6"} : <>
    %89 = constant %88 {handshake.bb = 3 : ui32, handshake.name = "constant22", value = 2 : i3} : <>, <i3>
    %90 = extsi %89 {handshake.bb = 3 : ui32, handshake.name = "extsi10"} : <i3> to <i32>
    %91:3 = fork [3] %90 {handshake.bb = 3 : ui32, handshake.name = "fork17"} : <i32>
    %92 = shli %73#0, %91#0 {handshake.bb = 3 : ui32, handshake.name = "shli0"} : <i32>
    %93 = buffer %92, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer21"} : <i32>
    %94 = trunci %93 {handshake.bb = 3 : ui32, handshake.name = "trunci5"} : <i32> to <i9>
    %95 = shli %73#1, %87#0 {handshake.bb = 3 : ui32, handshake.name = "shli1"} : <i32>
    %96 = buffer %95, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer22"} : <i32>
    %97 = trunci %96 {handshake.bb = 3 : ui32, handshake.name = "trunci6"} : <i32> to <i9>
    %98 = addi %94, %97 {handshake.bb = 3 : ui32, handshake.name = "addi9"} : <i9>
    %99 = buffer %98, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer23"} : <i9>
    %100 = addi %57, %99 {handshake.bb = 3 : ui32, handshake.name = "addi3"} : <i9>
    %101 = buffer %100, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer24"} : <i9>
    %addressResult, %dataResult = load[%101] %1#0 {handshake.bb = 3 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["store0", 0]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 0>, handshake.name = "load0"} : <i9>, <i32>, <i9>, <i32>
    %addressResult_14, %dataResult_15 = load[%67] %outputs {handshake.bb = 3 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i5>, <i32>, <i5>, <i32>
    %102 = shli %66#0, %91#1 {handshake.bb = 3 : ui32, handshake.name = "shli2"} : <i32>
    %103 = buffer %102, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer25"} : <i32>
    %104 = trunci %103 {handshake.bb = 3 : ui32, handshake.name = "trunci7"} : <i32> to <i9>
    %105 = shli %66#1, %87#1 {handshake.bb = 3 : ui32, handshake.name = "shli3"} : <i32>
    %106 = buffer %105, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer26"} : <i32>
    %107 = trunci %106 {handshake.bb = 3 : ui32, handshake.name = "trunci8"} : <i32> to <i9>
    %108 = addi %104, %107 {handshake.bb = 3 : ui32, handshake.name = "addi10"} : <i9>
    %109 = buffer %108, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer27"} : <i9>
    %110 = addi %58, %109 {handshake.bb = 3 : ui32, handshake.name = "addi4"} : <i9>
    %111 = buffer %110, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer28"} : <i9>
    %addressResult_16, %dataResult_17 = load[%111] %1#1 {handshake.bb = 3 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["store0", 0]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 0>, handshake.name = "load2"} : <i9>, <i32>, <i9>, <i32>
    %112 = muli %dataResult_15, %dataResult_17 {handshake.bb = 3 : ui32, handshake.name = "muli0"} : <i32>
    %113 = subi %dataResult, %112 {handshake.bb = 3 : ui32, handshake.name = "subi0"} : <i32>
    %114 = shli %73#2, %91#2 {handshake.bb = 3 : ui32, handshake.name = "shli4"} : <i32>
    %115 = buffer %114, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer30"} : <i32>
    %116 = trunci %115 {handshake.bb = 3 : ui32, handshake.name = "trunci9"} : <i32> to <i9>
    %117 = shli %73#3, %87#2 {handshake.bb = 3 : ui32, handshake.name = "shli5"} : <i32>
    %118 = buffer %117, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer31"} : <i32>
    %119 = trunci %118 {handshake.bb = 3 : ui32, handshake.name = "trunci10"} : <i32> to <i9>
    %120 = addi %116, %119 {handshake.bb = 3 : ui32, handshake.name = "addi11"} : <i9>
    %121 = buffer %120, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer32"} : <i9>
    %122 = addi %59, %121 {handshake.bb = 3 : ui32, handshake.name = "addi5"} : <i9>
    %123 = buffer %113, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer29"} : <i32>
    %124 = buffer %122, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer33"} : <i9>
    %addressResult_18, %dataResult_19 = store[%124] %123 {handshake.bb = 3 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["load0", 0], ["load2", 0], ["store0", 0]]>, handshake.mem_interface = #handshake.mem_interface<LSQ: 0>, handshake.name = "store0"} : <i9>, <i32>, <i9>, <i32>
    %125 = buffer %60, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer14"} : <i32>
    %126 = addi %125, %56#4 {handshake.bb = 3 : ui32, handshake.name = "addi0"} : <i32>
    %127 = addi %56#3, %83 {handshake.bb = 3 : ui32, handshake.name = "addi1"} : <i32>
    %128 = addi %52, %82 {handshake.bb = 3 : ui32, handshake.name = "addi6"} : <i7>
    %129 = buffer %128, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer35"} : <i7>
    %130:2 = fork [2] %129 {handshake.bb = 3 : ui32, handshake.name = "fork18"} : <i7>
    %131 = trunci %130#0 {handshake.bb = 3 : ui32, handshake.name = "trunci11"} : <i7> to <i6>
    %132 = cmpi ult, %130#1, %78 {handshake.bb = 3 : ui32, handshake.name = "cmpi0"} : <i7>
    %133 = buffer %132, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer36"} : <i1>
    %134:6 = fork [6] %133 {handshake.bb = 3 : ui32, handshake.name = "fork19"} : <i1>
    %trueResult_20, %falseResult_21 = cond_br %134#0, %131 {handshake.bb = 3 : ui32, handshake.name = "cond_br9"} : <i1>, <i6>
    sink %falseResult_21 {handshake.name = "sink3"} : <i6>
    %trueResult_22, %falseResult_23 = cond_br %134#3, %127 {handshake.bb = 3 : ui32, handshake.name = "cond_br10"} : <i1>, <i32>
    sink %falseResult_23 {handshake.name = "sink4"} : <i32>
    %135 = buffer %126, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer34"} : <i32>
    %trueResult_24, %falseResult_25 = cond_br %134#4, %135 {handshake.bb = 3 : ui32, handshake.name = "cond_br11"} : <i1>, <i32>
    %trueResult_26, %falseResult_27 = cond_br %134#1, %64#1 {handshake.bb = 3 : ui32, handshake.name = "cond_br12"} : <i1>, <i6>
    %trueResult_28, %falseResult_29 = cond_br %134#2, %71#0 {handshake.bb = 3 : ui32, handshake.name = "cond_br13"} : <i1>, <i6>
    %136 = buffer %75#1, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer19"} : <>
    %137 = buffer %136, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer20"} : <>
    %trueResult_30, %falseResult_31 = cond_br %134#5, %137 {handshake.bb = 3 : ui32, handshake.bufProps = #handshake<bufProps{"1": [1,inf], [0,inf], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "cond_br14"} : <i1>, <>
    %138 = merge %falseResult_27 {handshake.bb = 4 : ui32, handshake.name = "merge0"} : <i6>
    %139 = merge %falseResult_29 {handshake.bb = 4 : ui32, handshake.name = "merge1"} : <i6>
    %140 = buffer %139, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer37"} : <i6>
    %141 = extsi %140 {handshake.bb = 4 : ui32, handshake.name = "extsi26"} : <i6> to <i7>
    %142 = merge %falseResult_25 {handshake.bb = 4 : ui32, handshake.name = "merge2"} : <i32>
    %result_32, %index_33 = control_merge [%falseResult_31]  {handshake.bb = 4 : ui32, handshake.name = "control_merge3"} : [<>] to <>, <i1>
    sink %index_33 {handshake.name = "sink5"} : <i1>
    %143 = source {handshake.bb = 4 : ui32, handshake.name = "source7"} : <>
    %144 = constant %143 {handshake.bb = 4 : ui32, handshake.name = "constant23", value = 1 : i2} : <>, <i2>
    %145 = extsi %144 {handshake.bb = 4 : ui32, handshake.name = "extsi27"} : <i2> to <i7>
    %146 = addi %141, %145 {handshake.bb = 4 : ui32, handshake.name = "addi8"} : <i7>
    %147 = br %146 {handshake.bb = 4 : ui32, handshake.name = "br10"} : <i7>
    %148 = br %142 {handshake.bb = 4 : ui32, handshake.name = "br11"} : <i32>
    %149 = br %138 {handshake.bb = 4 : ui32, handshake.name = "br12"} : <i6>
    %150 = br %result_32 {handshake.bb = 4 : ui32, handshake.name = "br13"} : <>
    %151 = merge %falseResult_7 {handshake.bb = 5 : ui32, handshake.name = "merge3"} : <i6>
    %152 = extsi %151 {handshake.bb = 5 : ui32, handshake.name = "extsi28"} : <i6> to <i7>
    %153 = merge %falseResult_5 {handshake.bb = 5 : ui32, handshake.name = "merge4"} : <i32>
    %result_34, %index_35 = control_merge [%falseResult_11]  {handshake.bb = 5 : ui32, handshake.name = "control_merge4"} : [<>] to <>, <i1>
    sink %index_35 {handshake.name = "sink6"} : <i1>
    %154 = source {handshake.bb = 5 : ui32, handshake.name = "source8"} : <>
    %155 = constant %154 {handshake.bb = 5 : ui32, handshake.name = "constant24", value = 19 : i6} : <>, <i6>
    %156 = extsi %155 {handshake.bb = 5 : ui32, handshake.name = "extsi29"} : <i6> to <i7>
    %157 = source {handshake.bb = 5 : ui32, handshake.name = "source9"} : <>
    %158 = constant %157 {handshake.bb = 5 : ui32, handshake.name = "constant25", value = 1 : i2} : <>, <i2>
    %159 = extsi %158 {handshake.bb = 5 : ui32, handshake.name = "extsi30"} : <i2> to <i7>
    %160 = addi %152, %159 {handshake.bb = 5 : ui32, handshake.name = "addi7"} : <i7>
    %161 = buffer %160, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 5 : ui32, handshake.name = "buffer38"} : <i7>
    %162:2 = fork [2] %161 {handshake.bb = 5 : ui32, handshake.name = "fork20"} : <i7>
    %163 = trunci %162#0 {handshake.bb = 5 : ui32, handshake.name = "trunci12"} : <i7> to <i6>
    %164 = cmpi ult, %162#1, %156 {handshake.bb = 5 : ui32, handshake.name = "cmpi1"} : <i7>
    %165:3 = fork [3] %164 {handshake.bb = 5 : ui32, handshake.name = "fork21"} : <i1>
    %trueResult_36, %falseResult_37 = cond_br %165#0, %163 {handshake.bb = 5 : ui32, handshake.name = "cond_br15"} : <i1>, <i6>
    sink %falseResult_37 {handshake.name = "sink7"} : <i6>
    %trueResult_38, %falseResult_39 = cond_br %165#1, %153 {handshake.bb = 5 : ui32, handshake.name = "cond_br16"} : <i1>, <i32>
    %trueResult_40, %falseResult_41 = cond_br %165#2, %result_34 {handshake.bb = 5 : ui32, handshake.name = "cond_br17"} : <i1>, <>
    %166 = merge %falseResult_39 {handshake.bb = 6 : ui32, handshake.name = "merge5"} : <i32>
    %result_42, %index_43 = control_merge [%falseResult_41]  {handshake.bb = 6 : ui32, handshake.name = "control_merge5"} : [<>] to <>, <i1>
    sink %index_43 {handshake.name = "sink8"} : <i1>
    %167:2 = fork [2] %result_42 {handshake.bb = 6 : ui32, handshake.name = "fork22"} : <>
    end {handshake.bb = 6 : ui32, handshake.name = "end0"} %166, %memEnd, %1#2, %0#2 : <i32>, <>, <>, <>
  }
}

