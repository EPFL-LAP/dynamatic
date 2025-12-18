module {
  handshake.func @jacobi_1d_imper(%arg0: memref<100xi32>, %arg1: memref<100xi32>, %arg2: !handshake.control<>, %arg3: !handshake.control<>, %arg4: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["A", "B", "A_start", "B_start", "start"], cfg.edges = "[0,1][2,2,3,cmpi0][4,1,5,cmpi2][1,2][3,3,4,cmpi1]", resNames = ["A_end", "B_end", "end"]} {
    %0:4 = fork [4] %arg4 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork0"} : <>
    %outputs:2, %memEnd = mem_controller[%arg1 : memref<100xi32>] %arg3 (%40, %addressResult_16, %dataResult_17, %addressResult_38) %165#1 {connectedBlocks = [2 : i32, 3 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "5": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller0"} :    (!handshake.channel<i32>, !handshake.channel<i7>, !handshake.channel<i32>, !handshake.channel<i7>) -> (!handshake.control<>, !handshake.channel<i32>)
    %outputs_0:4, %memEnd_1 = mem_controller[%arg0 : memref<100xi32>] %arg2 (%addressResult, %addressResult_12, %addressResult_14, %123, %addressResult_40, %dataResult_41) %165#0 {connectedBlocks = [2 : i32, 3 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "5": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "6": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "7": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller1"} :    (!handshake.channel<i7>, !handshake.channel<i7>, !handshake.channel<i7>, !handshake.channel<i32>, !handshake.channel<i7>, !handshake.channel<i32>) -> (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>)
    %1 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant2", value = false} : <>, <i1>
    %2 = br %1 {handshake.bb = 0 : ui32, handshake.name = "br2"} : <i1>
    %3 = extsi %2 {handshake.bb = 0 : ui32, handshake.name = "extsi13"} : <i1> to <i3>
    %4 = br %0#3 {handshake.bb = 0 : ui32, handshake.name = "br3"} : <>
    %5 = mux %6 [%0#2, %trueResult_49] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux8"} : <i1>, [<>, <>] to <>
    %6 = init %164#2 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "init0"} : <i1>
    %7 = mux %index [%3, %trueResult_53] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i3>, <i3>] to <i3>
    %result, %index = control_merge [%4, %trueResult_55]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %8:2 = fork [2] %result {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <>
    %9 = constant %8#0 {handshake.bb = 1 : ui32, handshake.name = "constant4", value = 1 : i2} : <>, <i2>
    %10 = br %9 {handshake.bb = 1 : ui32, handshake.name = "br4"} : <i2>
    %11 = extsi %10 {handshake.bb = 1 : ui32, handshake.name = "extsi12"} : <i2> to <i8>
    %12 = buffer %7, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer7"} : <i3>
    %13 = br %12 {handshake.bb = 1 : ui32, handshake.name = "br5"} : <i3>
    %14 = br %8#1 {handshake.bb = 1 : ui32, handshake.name = "br6"} : <>
    %trueResult, %falseResult = cond_br %81#7, %72 {handshake.bb = 2 : ui32, handshake.name = "cond_br45"} : <i1>, <>
    sink %trueResult {handshake.name = "sink0"} : <>
    %trueResult_2, %falseResult_3 = cond_br %81#6, %54 {handshake.bb = 2 : ui32, handshake.name = "cond_br46"} : <i1>, <>
    sink %trueResult_2 {handshake.name = "sink1"} : <>
    %trueResult_4, %falseResult_5 = cond_br %81#5, %61 {handshake.bb = 2 : ui32, handshake.name = "cond_br47"} : <i1>, <>
    sink %trueResult_4 {handshake.name = "sink2"} : <>
    %trueResult_6, %falseResult_7 = cond_br %81#4, %50 {handshake.bb = 2 : ui32, handshake.name = "cond_br48"} : <i1>, <>
    sink %trueResult_6 {handshake.name = "sink3"} : <>
    %trueResult_8, %falseResult_9 = cond_br %81#3, %20#4 {handshake.bb = 2 : ui32, handshake.name = "cond_br49"} : <i1>, <>
    sink %falseResult_9 {handshake.name = "sink4"} : <>
    %15 = init %81#2 {ftd.imerge, handshake.bb = 2 : ui32, handshake.name = "init8"} : <i1>
    %16 = buffer %5, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer6"} : <>
    %17 = mux %15 [%16, %trueResult_8] {ftd.regen, handshake.bb = 2 : ui32, handshake.name = "mux13"} : <i1>, [<>, <>] to <>
    %18 = buffer %17, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer8"} : <>
    %19 = buffer %18, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer9"} : <>
    %20:5 = fork [5] %19 {handshake.bb = 2 : ui32, handshake.name = "fork2"} : <>
    %21:2 = unbundle %57#0  {handshake.bb = 2 : ui32, handshake.name = "unbundle0"} : <i32> to _ 
    %22:2 = unbundle %53#0  {handshake.bb = 2 : ui32, handshake.name = "unbundle1"} : <i32> to _ 
    %23:2 = unbundle %64#0  {handshake.bb = 2 : ui32, handshake.name = "unbundle2"} : <i32> to _ 
    %24 = mux %33#1 [%11, %trueResult_18] {handshake.bb = 2 : ui32, handshake.name = "mux1"} : <i1>, [<i8>, <i8>] to <i8>
    %25 = buffer %24, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer10"} : <i8>
    %26 = buffer %25, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer11"} : <i8>
    %27:3 = fork [3] %26 {handshake.bb = 2 : ui32, handshake.name = "fork3"} : <i8>
    %28 = extsi %27#0 {handshake.bb = 2 : ui32, handshake.name = "extsi14"} : <i8> to <i9>
    %29 = extsi %27#1 {handshake.bb = 2 : ui32, handshake.name = "extsi15"} : <i8> to <i9>
    %30 = extsi %27#2 {handshake.bb = 2 : ui32, handshake.name = "extsi16"} : <i8> to <i32>
    %31:3 = fork [3] %30 {handshake.bb = 2 : ui32, handshake.name = "fork4"} : <i32>
    %32 = mux %33#0 [%13, %trueResult_20] {handshake.bb = 2 : ui32, handshake.name = "mux2"} : <i1>, [<i3>, <i3>] to <i3>
    %result_10, %index_11 = control_merge [%14, %trueResult_22]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>, <>] to <>, <i1>
    %33:2 = fork [2] %index_11 {handshake.bb = 2 : ui32, handshake.name = "fork5"} : <i1>
    %34 = buffer %result_10, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer14"} : <>
    %35:2 = fork [2] %34 {handshake.bb = 2 : ui32, handshake.name = "fork6"} : <>
    %36 = constant %35#0 {handshake.bb = 2 : ui32, handshake.name = "constant5", value = 1 : i2} : <>, <i2>
    %37:4 = fork [4] %36 {handshake.bb = 2 : ui32, handshake.name = "fork7"} : <i2>
    %38 = extsi %37#0 {handshake.bb = 2 : ui32, handshake.name = "extsi17"} : <i2> to <i9>
    %39 = extsi %37#1 {handshake.bb = 2 : ui32, handshake.name = "extsi18"} : <i2> to <i9>
    %40 = extsi %37#3 {handshake.bb = 2 : ui32, handshake.name = "extsi3"} : <i2> to <i32>
    %41 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %42 = constant %41 {handshake.bb = 2 : ui32, handshake.name = "constant7", value = -1 : i32} : <>, <i32>
    %43 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %44 = constant %43 {handshake.bb = 2 : ui32, handshake.name = "constant15", value = 99 : i8} : <>, <i8>
    %45 = extsi %44 {handshake.bb = 2 : ui32, handshake.name = "extsi19"} : <i8> to <i9>
    %46 = source {handshake.bb = 2 : ui32, handshake.name = "source2"} : <>
    %47 = constant %46 {handshake.bb = 2 : ui32, handshake.name = "constant16", value = 1 : i2} : <>, <i2>
    %48 = extsi %47 {handshake.bb = 2 : ui32, handshake.name = "extsi5"} : <i2> to <i32>
    %49 = addi %31#0, %42 {handshake.bb = 2 : ui32, handshake.name = "addi2"} : <i32>
    %50 = buffer %22#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer0"} : <>
    %51 = gate %49, %20#3 {handshake.bb = 2 : ui32, handshake.name = "gate0"} : <i32>, !handshake.control<> to <i32>
    %52 = trunci %51 {handshake.bb = 2 : ui32, handshake.name = "trunci0"} : <i32> to <i7>
    %addressResult, %dataResult = load[%52] %outputs_0#0 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["store1", 1, false], ["store1", 2, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i7>, <i32>, <i7>, <i32>
    %53:2 = fork [2] %dataResult {handshake.bb = 2 : ui32, handshake.name = "fork8"} : <i32>
    %54 = buffer %21#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer1"} : <>
    %55 = gate %31#1, %20#2 {handshake.bb = 2 : ui32, handshake.name = "gate1"} : <i32>, !handshake.control<> to <i32>
    %56 = trunci %55 {handshake.bb = 2 : ui32, handshake.name = "trunci1"} : <i32> to <i7>
    %addressResult_12, %dataResult_13 = load[%56] %outputs_0#1 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["store1", 1, false], ["store1", 2, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i7>, <i32>, <i7>, <i32>
    %57:2 = fork [2] %dataResult_13 {handshake.bb = 2 : ui32, handshake.name = "fork9"} : <i32>
    %58 = addi %53#1, %57#1 {handshake.bb = 2 : ui32, handshake.name = "addi0"} : <i32>
    %59 = addi %29, %39 {handshake.bb = 2 : ui32, handshake.name = "addi3"} : <i9>
    %60 = extsi %59 {handshake.bb = 2 : ui32, handshake.name = "extsi20"} : <i9> to <i32>
    %61 = buffer %23#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer2"} : <>
    %62 = gate %60, %20#1 {handshake.bb = 2 : ui32, handshake.name = "gate2"} : <i32>, !handshake.control<> to <i32>
    %63 = trunci %62 {handshake.bb = 2 : ui32, handshake.name = "trunci2"} : <i32> to <i7>
    %addressResult_14, %dataResult_15 = load[%63] %outputs_0#2 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["store1", 1, false], ["store1", 2, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load2"} : <i7>, <i32>, <i7>, <i32>
    %64:2 = fork [2] %dataResult_15 {handshake.bb = 2 : ui32, handshake.name = "fork10"} : <i32>
    %65 = buffer %58, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer15"} : <i32>
    %66 = addi %65, %64#1 {handshake.bb = 2 : ui32, handshake.name = "addi1"} : <i32>
    %67 = buffer %66, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer16"} : <i32>
    %68:2 = fork [2] %67 {handshake.bb = 2 : ui32, handshake.name = "fork11"} : <i32>
    %69 = shli %68#1, %48 {handshake.bb = 2 : ui32, handshake.name = "shli0"} : <i32>
    %70 = buffer %69, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer17"} : <i32>
    %71 = addi %68#0, %70 {handshake.bb = 2 : ui32, handshake.name = "addi7"} : <i32>
    %72 = buffer %doneResult, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer3"} : <>
    %73 = gate %31#2, %20#0 {handshake.bb = 2 : ui32, handshake.name = "gate3"} : <i32>, !handshake.control<> to <i32>
    %74 = trunci %73 {handshake.bb = 2 : ui32, handshake.name = "trunci3"} : <i32> to <i7>
    %75 = buffer %71, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer18"} : <i32>
    %addressResult_16, %dataResult_17, %doneResult = store[%74] %75 %outputs#0 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["store0", 1, false], ["load3", 1, false], ["load3", 2, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store0"} : <i7>, <i32>, <>, <i7>, <i32>, <>
    %76 = addi %28, %38 {handshake.bb = 2 : ui32, handshake.name = "addi4"} : <i9>
    %77 = buffer %76, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer19"} : <i9>
    %78:2 = fork [2] %77 {handshake.bb = 2 : ui32, handshake.name = "fork12"} : <i9>
    %79 = trunci %78#0 {handshake.bb = 2 : ui32, handshake.name = "trunci4"} : <i9> to <i8>
    %80 = cmpi ult, %78#1, %45 {handshake.bb = 2 : ui32, handshake.name = "cmpi0"} : <i9>
    %81:10 = fork [10] %80 {handshake.bb = 2 : ui32, handshake.name = "fork13"} : <i1>
    %trueResult_18, %falseResult_19 = cond_br %81#0, %79 {handshake.bb = 2 : ui32, handshake.name = "cond_br3"} : <i1>, <i8>
    sink %falseResult_19 {handshake.name = "sink5"} : <i8>
    %82 = buffer %32, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer12"} : <i3>
    %83 = buffer %82, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer13"} : <i3>
    %trueResult_20, %falseResult_21 = cond_br %81#1, %83 {handshake.bb = 2 : ui32, handshake.name = "cond_br4"} : <i1>, <i3>
    %trueResult_22, %falseResult_23 = cond_br %81#8, %35#1 {handshake.bb = 2 : ui32, handshake.name = "cond_br5"} : <i1>, <>
    %trueResult_24, %falseResult_25 = cond_br %81#9, %37#2 {handshake.bb = 2 : ui32, handshake.name = "cond_br6"} : <i1>, <i2>
    sink %trueResult_24 {handshake.name = "sink6"} : <i2>
    %84 = extsi %falseResult_25 {handshake.bb = 2 : ui32, handshake.name = "extsi11"} : <i2> to <i8>
    %trueResult_26, %falseResult_27 = cond_br %145#7, %110#1 {handshake.bb = 3 : ui32, handshake.name = "cond_br50"} : <i1>, <>
    sink %falseResult_27 {handshake.name = "sink7"} : <>
    %trueResult_28, %falseResult_29 = cond_br %85, %135 {handshake.bb = 3 : ui32, handshake.name = "cond_br51"} : <i1>, <>
    %85 = buffer %145#6, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 3 : ui32, handshake.name = "buffer39"} : <i1>
    sink %trueResult_28 {handshake.name = "sink8"} : <>
    %trueResult_30, %falseResult_31 = cond_br %86, %100#1 {handshake.bb = 3 : ui32, handshake.name = "cond_br52"} : <i1>, <>
    %86 = buffer %145#5, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer40"} : <i1>
    sink %falseResult_31 {handshake.name = "sink9"} : <>
    %trueResult_32, %falseResult_33 = cond_br %87, %105#1 {handshake.bb = 3 : ui32, handshake.name = "cond_br53"} : <i1>, <>
    %87 = buffer %145#4, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer41"} : <i1>
    sink %falseResult_33 {handshake.name = "sink10"} : <>
    %trueResult_34, %falseResult_35 = cond_br %88, %95#1 {handshake.bb = 3 : ui32, handshake.name = "cond_br54"} : <i1>, <>
    %88 = buffer %145#3, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer42"} : <i1>
    sink %falseResult_35 {handshake.name = "sink11"} : <>
    %89 = init %145#2 {ftd.imerge, handshake.bb = 3 : ui32, handshake.name = "init16"} : <i1>
    %90:4 = fork [4] %89 {handshake.bb = 3 : ui32, handshake.name = "fork14"} : <i1>
    %91 = mux %92 [%falseResult_3, %trueResult_34] {ftd.regen, handshake.bb = 3 : ui32, handshake.name = "mux21"} : <i1>, [<>, <>] to <>
    %92 = buffer %90#3, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer44"} : <i1>
    %93 = buffer %91, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer21"} : <>
    %94 = buffer %93, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer22"} : <>
    %95:2 = fork [2] %94 {handshake.bb = 3 : ui32, handshake.name = "fork15"} : <>
    %96 = mux %97 [%falseResult_7, %trueResult_30] {ftd.regen, handshake.bb = 3 : ui32, handshake.name = "mux22"} : <i1>, [<>, <>] to <>
    %97 = buffer %90#2, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer45"} : <i1>
    %98 = buffer %96, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer23"} : <>
    %99 = buffer %98, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer24"} : <>
    %100:2 = fork [2] %99 {handshake.bb = 3 : ui32, handshake.name = "fork16"} : <>
    %101 = mux %102 [%falseResult_5, %trueResult_32] {ftd.regen, handshake.bb = 3 : ui32, handshake.name = "mux23"} : <i1>, [<>, <>] to <>
    %102 = buffer %90#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer46"} : <i1>
    %103 = buffer %101, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer25"} : <>
    %104 = buffer %103, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer26"} : <>
    %105:2 = fork [2] %104 {handshake.bb = 3 : ui32, handshake.name = "fork17"} : <>
    %106 = buffer %trueResult_26, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer20"} : <>
    %107 = mux %108 [%falseResult, %106] {ftd.regen, handshake.bb = 3 : ui32, handshake.name = "mux24"} : <i1>, [<>, <>] to <>
    %108 = buffer %90#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer47"} : <i1>
    %109 = buffer %107, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer27"} : <>
    %110:2 = fork [2] %109 {handshake.bb = 3 : ui32, handshake.name = "fork18"} : <>
    %111:2 = unbundle %134#1  {handshake.bb = 3 : ui32, handshake.name = "unbundle6"} : <i32> to _ 
    %112 = mux %120#1 [%84, %trueResult_43] {handshake.bb = 3 : ui32, handshake.name = "mux3"} : <i1>, [<i8>, <i8>] to <i8>
    %113 = buffer %112, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer28"} : <i8>
    %114 = buffer %113, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer29"} : <i8>
    %115:2 = fork [2] %114 {handshake.bb = 3 : ui32, handshake.name = "fork19"} : <i8>
    %116 = extsi %115#0 {handshake.bb = 3 : ui32, handshake.name = "extsi21"} : <i8> to <i9>
    %117 = extsi %115#1 {handshake.bb = 3 : ui32, handshake.name = "extsi22"} : <i8> to <i32>
    %118:2 = fork [2] %117 {handshake.bb = 3 : ui32, handshake.name = "fork20"} : <i32>
    %119 = mux %120#0 [%falseResult_21, %trueResult_45] {handshake.bb = 3 : ui32, handshake.name = "mux4"} : <i1>, [<i3>, <i3>] to <i3>
    %result_36, %index_37 = control_merge [%falseResult_23, %trueResult_47]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>, <>] to <>, <i1>
    %120:2 = fork [2] %index_37 {handshake.bb = 3 : ui32, handshake.name = "fork21"} : <i1>
    %121:2 = fork [2] %result_36 {handshake.bb = 3 : ui32, handshake.name = "fork22"} : <>
    %122 = constant %121#0 {handshake.bb = 3 : ui32, handshake.name = "constant17", value = 1 : i2} : <>, <i2>
    %123 = extsi %122 {handshake.bb = 3 : ui32, handshake.name = "extsi6"} : <i2> to <i32>
    %124 = source {handshake.bb = 3 : ui32, handshake.name = "source3"} : <>
    %125 = constant %124 {handshake.bb = 3 : ui32, handshake.name = "constant18", value = 99 : i8} : <>, <i8>
    %126 = extsi %125 {handshake.bb = 3 : ui32, handshake.name = "extsi23"} : <i8> to <i9>
    %127 = source {handshake.bb = 3 : ui32, handshake.name = "source4"} : <>
    %128 = constant %127 {handshake.bb = 3 : ui32, handshake.name = "constant19", value = 1 : i2} : <>, <i2>
    %129 = extsi %128 {handshake.bb = 3 : ui32, handshake.name = "extsi24"} : <i2> to <i9>
    %130 = buffer %111#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer4"} : <>
    sink %130 {handshake.name = "sink12"} : <>
    %131 = gate %132, %110#0 {handshake.bb = 3 : ui32, handshake.name = "gate4"} : <i32>, !handshake.control<> to <i32>
    %132 = buffer %118#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer53"} : <i32>
    %133 = trunci %131 {handshake.bb = 3 : ui32, handshake.name = "trunci5"} : <i32> to <i7>
    %addressResult_38, %dataResult_39 = load[%133] %outputs#1 {handshake.bb = 3 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["store0", 1, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load3"} : <i7>, <i32>, <i7>, <i32>
    %134:2 = fork [2] %dataResult_39 {handshake.bb = 3 : ui32, handshake.name = "fork23"} : <i32>
    %135 = buffer %doneResult_42, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer5"} : <>
    %136 = gate %137, %105#0, %100#0, %95#0 {handshake.bb = 3 : ui32, handshake.name = "gate5"} : <i32>, !handshake.control<>, !handshake.control<>, !handshake.control<> to <i32>
    %137 = buffer %118#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer54"} : <i32>
    %138 = trunci %136 {handshake.bb = 3 : ui32, handshake.name = "trunci6"} : <i32> to <i7>
    %addressResult_40, %dataResult_41, %doneResult_42 = store[%138] %134#0 %outputs_0#3 {handshake.bb = 3 : ui32, handshake.bufProps = #handshake<bufProps{"2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["load0", 1, false], ["load1", 1, false], ["load2", 1, false], ["store1", 1, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store1"} : <i7>, <i32>, <>, <i7>, <i32>, <>
    %139 = addi %116, %129 {handshake.bb = 3 : ui32, handshake.name = "addi5"} : <i9>
    %140 = buffer %139, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer33"} : <i9>
    %141:2 = fork [2] %140 {handshake.bb = 3 : ui32, handshake.name = "fork24"} : <i9>
    %142 = trunci %141#0 {handshake.bb = 3 : ui32, handshake.name = "trunci7"} : <i9> to <i8>
    %143 = cmpi ult, %141#1, %126 {handshake.bb = 3 : ui32, handshake.name = "cmpi1"} : <i9>
    %144 = buffer %143, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer34"} : <i1>
    %145:9 = fork [9] %144 {handshake.bb = 3 : ui32, handshake.name = "fork25"} : <i1>
    %trueResult_43, %falseResult_44 = cond_br %145#0, %142 {handshake.bb = 3 : ui32, handshake.name = "cond_br7"} : <i1>, <i8>
    sink %falseResult_44 {handshake.name = "sink13"} : <i8>
    %146 = buffer %119, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer30"} : <i3>
    %147 = buffer %146, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer31"} : <i3>
    %trueResult_45, %falseResult_46 = cond_br %145#1, %147 {handshake.bb = 3 : ui32, handshake.name = "cond_br8"} : <i1>, <i3>
    %148 = buffer %121#1, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer32"} : <>
    %trueResult_47, %falseResult_48 = cond_br %149, %148 {handshake.bb = 3 : ui32, handshake.name = "cond_br9"} : <i1>, <>
    %149 = buffer %145#8, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer60"} : <i1>
    %trueResult_49, %falseResult_50 = cond_br %164#1, %falseResult_29 {handshake.bb = 4 : ui32, handshake.name = "cond_br55"} : <i1>, <>
    sink %falseResult_50 {handshake.name = "sink14"} : <>
    %150 = merge %falseResult_46 {handshake.bb = 4 : ui32, handshake.name = "merge0"} : <i3>
    %151 = extsi %150 {handshake.bb = 4 : ui32, handshake.name = "extsi25"} : <i3> to <i4>
    %result_51, %index_52 = control_merge [%falseResult_48]  {handshake.bb = 4 : ui32, handshake.name = "control_merge3"} : [<>] to <>, <i1>
    sink %index_52 {handshake.name = "sink15"} : <i1>
    %152 = source {handshake.bb = 4 : ui32, handshake.name = "source5"} : <>
    %153 = constant %152 {handshake.bb = 4 : ui32, handshake.name = "constant20", value = 3 : i3} : <>, <i3>
    %154 = extsi %153 {handshake.bb = 4 : ui32, handshake.name = "extsi26"} : <i3> to <i4>
    %155 = source {handshake.bb = 4 : ui32, handshake.name = "source6"} : <>
    %156 = constant %155 {handshake.bb = 4 : ui32, handshake.name = "constant21", value = 1 : i2} : <>, <i2>
    %157 = extsi %156 {handshake.bb = 4 : ui32, handshake.name = "extsi27"} : <i2> to <i4>
    %158 = addi %151, %157 {handshake.bb = 4 : ui32, handshake.name = "addi6"} : <i4>
    %159 = buffer %158, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer35"} : <i4>
    %160:2 = fork [2] %159 {handshake.bb = 4 : ui32, handshake.name = "fork26"} : <i4>
    %161 = trunci %160#0 {handshake.bb = 4 : ui32, handshake.name = "trunci8"} : <i4> to <i3>
    %162 = cmpi ult, %160#1, %154 {handshake.bb = 4 : ui32, handshake.name = "cmpi2"} : <i4>
    %163 = buffer %162, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer36"} : <i1>
    %164:4 = fork [4] %163 {handshake.bb = 4 : ui32, handshake.name = "fork27"} : <i1>
    %trueResult_53, %falseResult_54 = cond_br %164#0, %161 {handshake.bb = 4 : ui32, handshake.name = "cond_br10"} : <i1>, <i3>
    sink %falseResult_54 {handshake.name = "sink16"} : <i3>
    %trueResult_55, %falseResult_56 = cond_br %164#3, %result_51 {handshake.bb = 4 : ui32, handshake.name = "cond_br11"} : <i1>, <>
    %result_57, %index_58 = control_merge [%falseResult_56]  {handshake.bb = 5 : ui32, handshake.name = "control_merge4"} : [<>] to <>, <i1>
    sink %index_58 {handshake.name = "sink17"} : <i1>
    %165:2 = fork [2] %result_57 {handshake.bb = 5 : ui32, handshake.name = "fork28"} : <>
    end {handshake.bb = 5 : ui32, handshake.name = "end0"} %memEnd_1, %memEnd, %0#1 : <>, <>, <>
  }
}

