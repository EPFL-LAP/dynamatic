module {
  handshake.func @jacobi_1d_imper(%arg0: memref<100xi32>, %arg1: memref<100xi32>, %arg2: !handshake.control<>, %arg3: !handshake.control<>, %arg4: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["A", "B", "A_start", "B_start", "start"], cfg.edges = "[0,1][2,2,3,cmpi0][4,1,5,cmpi2][1,2][3,3,4,cmpi1]", resNames = ["A_end", "B_end", "end"]} {
    %0:4 = fork [4] %arg4 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork0"} : <>
    %outputs:2, %memEnd = mem_controller[%arg1 : memref<100xi32>] %arg3 (%35, %addressResult_16, %dataResult_17, %addressResult_38) %159#1 {connectedBlocks = [2 : i32, 3 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "5": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller0"} :    (!handshake.channel<i32>, !handshake.channel<i7>, !handshake.channel<i32>, !handshake.channel<i7>) -> (!handshake.control<>, !handshake.channel<i32>)
    %outputs_0:4, %memEnd_1 = mem_controller[%arg0 : memref<100xi32>] %arg2 (%addressResult, %addressResult_12, %addressResult_14, %118, %addressResult_40, %dataResult_41) %159#0 {connectedBlocks = [2 : i32, 3 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "5": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "6": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "7": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller1"} :    (!handshake.channel<i7>, !handshake.channel<i7>, !handshake.channel<i7>, !handshake.channel<i32>, !handshake.channel<i7>, !handshake.channel<i32>) -> (!handshake.channel<i32>, !handshake.channel<i32>, !handshake.channel<i32>, !handshake.control<>)
    %1 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant2", value = false} : <>, <i1>
    %2 = extsi %1 {handshake.bb = 0 : ui32, handshake.name = "extsi13"} : <i1> to <i3>
    %3 = mux %4 [%0#2, %trueResult_49] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux8"} : <i1>, [<>, <>] to <>
    %4 = init %158#2 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "init0"} : <i1>
    %5 = mux %index [%2, %trueResult_51] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i3>, <i3>] to <i3>
    %result, %index = control_merge [%0#3, %trueResult_53]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %6:2 = fork [2] %result {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <>
    %7 = constant %6#0 {handshake.bb = 1 : ui32, handshake.name = "constant4", value = 1 : i2} : <>, <i2>
    %8 = extsi %7 {handshake.bb = 1 : ui32, handshake.name = "extsi12"} : <i2> to <i8>
    %9 = buffer %5, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer7"} : <i3>
    %trueResult, %falseResult = cond_br %76#7, %67 {handshake.bb = 2 : ui32, handshake.name = "cond_br45"} : <i1>, <>
    sink %trueResult {handshake.name = "sink0"} : <>
    %trueResult_2, %falseResult_3 = cond_br %76#6, %49 {handshake.bb = 2 : ui32, handshake.name = "cond_br46"} : <i1>, <>
    sink %trueResult_2 {handshake.name = "sink1"} : <>
    %trueResult_4, %falseResult_5 = cond_br %76#5, %56 {handshake.bb = 2 : ui32, handshake.name = "cond_br47"} : <i1>, <>
    sink %trueResult_4 {handshake.name = "sink2"} : <>
    %trueResult_6, %falseResult_7 = cond_br %76#4, %45 {handshake.bb = 2 : ui32, handshake.name = "cond_br48"} : <i1>, <>
    sink %trueResult_6 {handshake.name = "sink3"} : <>
    %trueResult_8, %falseResult_9 = cond_br %76#3, %15#4 {handshake.bb = 2 : ui32, handshake.name = "cond_br49"} : <i1>, <>
    sink %falseResult_9 {handshake.name = "sink4"} : <>
    %10 = init %76#2 {ftd.imerge, handshake.bb = 2 : ui32, handshake.name = "init8"} : <i1>
    %11 = buffer %3, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer6"} : <>
    %12 = mux %10 [%11, %trueResult_8] {ftd.regen, handshake.bb = 2 : ui32, handshake.name = "mux13"} : <i1>, [<>, <>] to <>
    %13 = buffer %12, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer8"} : <>
    %14 = buffer %13, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer9"} : <>
    %15:5 = fork [5] %14 {handshake.bb = 2 : ui32, handshake.name = "fork2"} : <>
    %16:2 = unbundle %52#0  {handshake.bb = 2 : ui32, handshake.name = "unbundle0"} : <i32> to _ 
    %17:2 = unbundle %48#0  {handshake.bb = 2 : ui32, handshake.name = "unbundle1"} : <i32> to _ 
    %18:2 = unbundle %59#0  {handshake.bb = 2 : ui32, handshake.name = "unbundle2"} : <i32> to _ 
    %19 = mux %28#1 [%8, %trueResult_18] {handshake.bb = 2 : ui32, handshake.name = "mux1"} : <i1>, [<i8>, <i8>] to <i8>
    %20 = buffer %19, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer10"} : <i8>
    %21 = buffer %20, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer11"} : <i8>
    %22:3 = fork [3] %21 {handshake.bb = 2 : ui32, handshake.name = "fork3"} : <i8>
    %23 = extsi %22#0 {handshake.bb = 2 : ui32, handshake.name = "extsi14"} : <i8> to <i9>
    %24 = extsi %22#1 {handshake.bb = 2 : ui32, handshake.name = "extsi15"} : <i8> to <i9>
    %25 = extsi %22#2 {handshake.bb = 2 : ui32, handshake.name = "extsi16"} : <i8> to <i32>
    %26:3 = fork [3] %25 {handshake.bb = 2 : ui32, handshake.name = "fork4"} : <i32>
    %27 = mux %28#0 [%9, %trueResult_20] {handshake.bb = 2 : ui32, handshake.name = "mux2"} : <i1>, [<i3>, <i3>] to <i3>
    %result_10, %index_11 = control_merge [%6#1, %trueResult_22]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>, <>] to <>, <i1>
    %28:2 = fork [2] %index_11 {handshake.bb = 2 : ui32, handshake.name = "fork5"} : <i1>
    %29 = buffer %result_10, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer14"} : <>
    %30:2 = fork [2] %29 {handshake.bb = 2 : ui32, handshake.name = "fork6"} : <>
    %31 = constant %30#0 {handshake.bb = 2 : ui32, handshake.name = "constant5", value = 1 : i2} : <>, <i2>
    %32:4 = fork [4] %31 {handshake.bb = 2 : ui32, handshake.name = "fork7"} : <i2>
    %33 = extsi %32#0 {handshake.bb = 2 : ui32, handshake.name = "extsi17"} : <i2> to <i9>
    %34 = extsi %32#1 {handshake.bb = 2 : ui32, handshake.name = "extsi18"} : <i2> to <i9>
    %35 = extsi %32#3 {handshake.bb = 2 : ui32, handshake.name = "extsi3"} : <i2> to <i32>
    %36 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %37 = constant %36 {handshake.bb = 2 : ui32, handshake.name = "constant7", value = -1 : i32} : <>, <i32>
    %38 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %39 = constant %38 {handshake.bb = 2 : ui32, handshake.name = "constant15", value = 99 : i8} : <>, <i8>
    %40 = extsi %39 {handshake.bb = 2 : ui32, handshake.name = "extsi19"} : <i8> to <i9>
    %41 = source {handshake.bb = 2 : ui32, handshake.name = "source2"} : <>
    %42 = constant %41 {handshake.bb = 2 : ui32, handshake.name = "constant16", value = 1 : i2} : <>, <i2>
    %43 = extsi %42 {handshake.bb = 2 : ui32, handshake.name = "extsi5"} : <i2> to <i32>
    %44 = addi %26#0, %37 {handshake.bb = 2 : ui32, handshake.name = "addi2"} : <i32>
    %45 = buffer %17#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer0"} : <>
    %46 = gate %44, %15#3 {handshake.bb = 2 : ui32, handshake.name = "gate0"} : <i32>, !handshake.control<> to <i32>
    %47 = trunci %46 {handshake.bb = 2 : ui32, handshake.name = "trunci0"} : <i32> to <i7>
    %addressResult, %dataResult = load[%47] %outputs_0#0 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["store1", 1, false], ["store1", 2, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i7>, <i32>, <i7>, <i32>
    %48:2 = fork [2] %dataResult {handshake.bb = 2 : ui32, handshake.name = "fork8"} : <i32>
    %49 = buffer %16#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer1"} : <>
    %50 = gate %26#1, %15#2 {handshake.bb = 2 : ui32, handshake.name = "gate1"} : <i32>, !handshake.control<> to <i32>
    %51 = trunci %50 {handshake.bb = 2 : ui32, handshake.name = "trunci1"} : <i32> to <i7>
    %addressResult_12, %dataResult_13 = load[%51] %outputs_0#1 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["store1", 1, false], ["store1", 2, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i7>, <i32>, <i7>, <i32>
    %52:2 = fork [2] %dataResult_13 {handshake.bb = 2 : ui32, handshake.name = "fork9"} : <i32>
    %53 = addi %48#1, %52#1 {handshake.bb = 2 : ui32, handshake.name = "addi0"} : <i32>
    %54 = addi %24, %34 {handshake.bb = 2 : ui32, handshake.name = "addi3"} : <i9>
    %55 = extsi %54 {handshake.bb = 2 : ui32, handshake.name = "extsi20"} : <i9> to <i32>
    %56 = buffer %18#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer2"} : <>
    %57 = gate %55, %15#1 {handshake.bb = 2 : ui32, handshake.name = "gate2"} : <i32>, !handshake.control<> to <i32>
    %58 = trunci %57 {handshake.bb = 2 : ui32, handshake.name = "trunci2"} : <i32> to <i7>
    %addressResult_14, %dataResult_15 = load[%58] %outputs_0#2 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["store1", 1, false], ["store1", 2, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load2"} : <i7>, <i32>, <i7>, <i32>
    %59:2 = fork [2] %dataResult_15 {handshake.bb = 2 : ui32, handshake.name = "fork10"} : <i32>
    %60 = buffer %53, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer15"} : <i32>
    %61 = addi %60, %59#1 {handshake.bb = 2 : ui32, handshake.name = "addi1"} : <i32>
    %62 = buffer %61, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer16"} : <i32>
    %63:2 = fork [2] %62 {handshake.bb = 2 : ui32, handshake.name = "fork11"} : <i32>
    %64 = shli %63#1, %43 {handshake.bb = 2 : ui32, handshake.name = "shli0"} : <i32>
    %65 = buffer %64, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer17"} : <i32>
    %66 = addi %63#0, %65 {handshake.bb = 2 : ui32, handshake.name = "addi7"} : <i32>
    %67 = buffer %doneResult, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer3"} : <>
    %68 = gate %26#2, %15#0 {handshake.bb = 2 : ui32, handshake.name = "gate3"} : <i32>, !handshake.control<> to <i32>
    %69 = trunci %68 {handshake.bb = 2 : ui32, handshake.name = "trunci3"} : <i32> to <i7>
    %70 = buffer %66, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer18"} : <i32>
    %addressResult_16, %dataResult_17, %doneResult = store[%69] %70 %outputs#0 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["store0", 1, false], ["load3", 1, false], ["load3", 2, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store0"} : <i7>, <i32>, <>, <i7>, <i32>, <>
    %71 = addi %23, %33 {handshake.bb = 2 : ui32, handshake.name = "addi4"} : <i9>
    %72 = buffer %71, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer19"} : <i9>
    %73:2 = fork [2] %72 {handshake.bb = 2 : ui32, handshake.name = "fork12"} : <i9>
    %74 = trunci %73#0 {handshake.bb = 2 : ui32, handshake.name = "trunci4"} : <i9> to <i8>
    %75 = cmpi ult, %73#1, %40 {handshake.bb = 2 : ui32, handshake.name = "cmpi0"} : <i9>
    %76:10 = fork [10] %75 {handshake.bb = 2 : ui32, handshake.name = "fork13"} : <i1>
    %trueResult_18, %falseResult_19 = cond_br %76#0, %74 {handshake.bb = 2 : ui32, handshake.name = "cond_br3"} : <i1>, <i8>
    sink %falseResult_19 {handshake.name = "sink5"} : <i8>
    %77 = buffer %27, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer12"} : <i3>
    %78 = buffer %77, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer13"} : <i3>
    %trueResult_20, %falseResult_21 = cond_br %76#1, %78 {handshake.bb = 2 : ui32, handshake.name = "cond_br4"} : <i1>, <i3>
    %trueResult_22, %falseResult_23 = cond_br %76#8, %30#1 {handshake.bb = 2 : ui32, handshake.name = "cond_br5"} : <i1>, <>
    %trueResult_24, %falseResult_25 = cond_br %76#9, %32#2 {handshake.bb = 2 : ui32, handshake.name = "cond_br6"} : <i1>, <i2>
    sink %trueResult_24 {handshake.name = "sink6"} : <i2>
    %79 = extsi %falseResult_25 {handshake.bb = 2 : ui32, handshake.name = "extsi11"} : <i2> to <i8>
    %trueResult_26, %falseResult_27 = cond_br %140#7, %105#1 {handshake.bb = 3 : ui32, handshake.name = "cond_br50"} : <i1>, <>
    sink %falseResult_27 {handshake.name = "sink7"} : <>
    %trueResult_28, %falseResult_29 = cond_br %80, %130 {handshake.bb = 3 : ui32, handshake.name = "cond_br51"} : <i1>, <>
    %80 = buffer %140#6, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 3 : ui32, handshake.name = "buffer39"} : <i1>
    sink %trueResult_28 {handshake.name = "sink8"} : <>
    %trueResult_30, %falseResult_31 = cond_br %81, %95#1 {handshake.bb = 3 : ui32, handshake.name = "cond_br52"} : <i1>, <>
    %81 = buffer %140#5, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer40"} : <i1>
    sink %falseResult_31 {handshake.name = "sink9"} : <>
    %trueResult_32, %falseResult_33 = cond_br %82, %100#1 {handshake.bb = 3 : ui32, handshake.name = "cond_br53"} : <i1>, <>
    %82 = buffer %140#4, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer41"} : <i1>
    sink %falseResult_33 {handshake.name = "sink10"} : <>
    %trueResult_34, %falseResult_35 = cond_br %83, %90#1 {handshake.bb = 3 : ui32, handshake.name = "cond_br54"} : <i1>, <>
    %83 = buffer %140#3, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer42"} : <i1>
    sink %falseResult_35 {handshake.name = "sink11"} : <>
    %84 = init %140#2 {ftd.imerge, handshake.bb = 3 : ui32, handshake.name = "init16"} : <i1>
    %85:4 = fork [4] %84 {handshake.bb = 3 : ui32, handshake.name = "fork14"} : <i1>
    %86 = mux %87 [%falseResult_3, %trueResult_34] {ftd.regen, handshake.bb = 3 : ui32, handshake.name = "mux21"} : <i1>, [<>, <>] to <>
    %87 = buffer %85#3, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer44"} : <i1>
    %88 = buffer %86, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer21"} : <>
    %89 = buffer %88, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer22"} : <>
    %90:2 = fork [2] %89 {handshake.bb = 3 : ui32, handshake.name = "fork15"} : <>
    %91 = mux %92 [%falseResult_7, %trueResult_30] {ftd.regen, handshake.bb = 3 : ui32, handshake.name = "mux22"} : <i1>, [<>, <>] to <>
    %92 = buffer %85#2, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer45"} : <i1>
    %93 = buffer %91, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer23"} : <>
    %94 = buffer %93, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer24"} : <>
    %95:2 = fork [2] %94 {handshake.bb = 3 : ui32, handshake.name = "fork16"} : <>
    %96 = mux %97 [%falseResult_5, %trueResult_32] {ftd.regen, handshake.bb = 3 : ui32, handshake.name = "mux23"} : <i1>, [<>, <>] to <>
    %97 = buffer %85#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer46"} : <i1>
    %98 = buffer %96, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer25"} : <>
    %99 = buffer %98, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer26"} : <>
    %100:2 = fork [2] %99 {handshake.bb = 3 : ui32, handshake.name = "fork17"} : <>
    %101 = buffer %trueResult_26, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer20"} : <>
    %102 = mux %103 [%falseResult, %101] {ftd.regen, handshake.bb = 3 : ui32, handshake.name = "mux24"} : <i1>, [<>, <>] to <>
    %103 = buffer %85#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer47"} : <i1>
    %104 = buffer %102, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer27"} : <>
    %105:2 = fork [2] %104 {handshake.bb = 3 : ui32, handshake.name = "fork18"} : <>
    %106:2 = unbundle %129#1  {handshake.bb = 3 : ui32, handshake.name = "unbundle6"} : <i32> to _ 
    %107 = mux %115#1 [%79, %trueResult_43] {handshake.bb = 3 : ui32, handshake.name = "mux3"} : <i1>, [<i8>, <i8>] to <i8>
    %108 = buffer %107, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer28"} : <i8>
    %109 = buffer %108, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer29"} : <i8>
    %110:2 = fork [2] %109 {handshake.bb = 3 : ui32, handshake.name = "fork19"} : <i8>
    %111 = extsi %110#0 {handshake.bb = 3 : ui32, handshake.name = "extsi21"} : <i8> to <i9>
    %112 = extsi %110#1 {handshake.bb = 3 : ui32, handshake.name = "extsi22"} : <i8> to <i32>
    %113:2 = fork [2] %112 {handshake.bb = 3 : ui32, handshake.name = "fork20"} : <i32>
    %114 = mux %115#0 [%falseResult_21, %trueResult_45] {handshake.bb = 3 : ui32, handshake.name = "mux4"} : <i1>, [<i3>, <i3>] to <i3>
    %result_36, %index_37 = control_merge [%falseResult_23, %trueResult_47]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>, <>] to <>, <i1>
    %115:2 = fork [2] %index_37 {handshake.bb = 3 : ui32, handshake.name = "fork21"} : <i1>
    %116:2 = fork [2] %result_36 {handshake.bb = 3 : ui32, handshake.name = "fork22"} : <>
    %117 = constant %116#0 {handshake.bb = 3 : ui32, handshake.name = "constant17", value = 1 : i2} : <>, <i2>
    %118 = extsi %117 {handshake.bb = 3 : ui32, handshake.name = "extsi6"} : <i2> to <i32>
    %119 = source {handshake.bb = 3 : ui32, handshake.name = "source3"} : <>
    %120 = constant %119 {handshake.bb = 3 : ui32, handshake.name = "constant18", value = 99 : i8} : <>, <i8>
    %121 = extsi %120 {handshake.bb = 3 : ui32, handshake.name = "extsi23"} : <i8> to <i9>
    %122 = source {handshake.bb = 3 : ui32, handshake.name = "source4"} : <>
    %123 = constant %122 {handshake.bb = 3 : ui32, handshake.name = "constant19", value = 1 : i2} : <>, <i2>
    %124 = extsi %123 {handshake.bb = 3 : ui32, handshake.name = "extsi24"} : <i2> to <i9>
    %125 = buffer %106#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer4"} : <>
    sink %125 {handshake.name = "sink12"} : <>
    %126 = gate %127, %105#0 {handshake.bb = 3 : ui32, handshake.name = "gate4"} : <i32>, !handshake.control<> to <i32>
    %127 = buffer %113#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer53"} : <i32>
    %128 = trunci %126 {handshake.bb = 3 : ui32, handshake.name = "trunci5"} : <i32> to <i7>
    %addressResult_38, %dataResult_39 = load[%128] %outputs#1 {handshake.bb = 3 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["store0", 1, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load3"} : <i7>, <i32>, <i7>, <i32>
    %129:2 = fork [2] %dataResult_39 {handshake.bb = 3 : ui32, handshake.name = "fork23"} : <i32>
    %130 = buffer %doneResult_42, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer5"} : <>
    %131 = gate %132, %100#0, %95#0, %90#0 {handshake.bb = 3 : ui32, handshake.name = "gate5"} : <i32>, !handshake.control<>, !handshake.control<>, !handshake.control<> to <i32>
    %132 = buffer %113#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer54"} : <i32>
    %133 = trunci %131 {handshake.bb = 3 : ui32, handshake.name = "trunci6"} : <i32> to <i7>
    %addressResult_40, %dataResult_41, %doneResult_42 = store[%133] %129#0 %outputs_0#3 {handshake.bb = 3 : ui32, handshake.bufProps = #handshake<bufProps{"2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["load0", 1, false], ["load1", 1, false], ["load2", 1, false], ["store1", 1, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store1"} : <i7>, <i32>, <>, <i7>, <i32>, <>
    %134 = addi %111, %124 {handshake.bb = 3 : ui32, handshake.name = "addi5"} : <i9>
    %135 = buffer %134, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer33"} : <i9>
    %136:2 = fork [2] %135 {handshake.bb = 3 : ui32, handshake.name = "fork24"} : <i9>
    %137 = trunci %136#0 {handshake.bb = 3 : ui32, handshake.name = "trunci7"} : <i9> to <i8>
    %138 = cmpi ult, %136#1, %121 {handshake.bb = 3 : ui32, handshake.name = "cmpi1"} : <i9>
    %139 = buffer %138, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer34"} : <i1>
    %140:9 = fork [9] %139 {handshake.bb = 3 : ui32, handshake.name = "fork25"} : <i1>
    %trueResult_43, %falseResult_44 = cond_br %140#0, %137 {handshake.bb = 3 : ui32, handshake.name = "cond_br7"} : <i1>, <i8>
    sink %falseResult_44 {handshake.name = "sink13"} : <i8>
    %141 = buffer %114, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer30"} : <i3>
    %142 = buffer %141, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer31"} : <i3>
    %trueResult_45, %falseResult_46 = cond_br %140#1, %142 {handshake.bb = 3 : ui32, handshake.name = "cond_br8"} : <i1>, <i3>
    %143 = buffer %116#1, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer32"} : <>
    %trueResult_47, %falseResult_48 = cond_br %144, %143 {handshake.bb = 3 : ui32, handshake.name = "cond_br9"} : <i1>, <>
    %144 = buffer %140#8, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 3 : ui32, handshake.name = "buffer60"} : <i1>
    %trueResult_49, %falseResult_50 = cond_br %158#1, %falseResult_29 {handshake.bb = 4 : ui32, handshake.name = "cond_br55"} : <i1>, <>
    sink %falseResult_50 {handshake.name = "sink14"} : <>
    %145 = extsi %falseResult_46 {handshake.bb = 4 : ui32, handshake.name = "extsi25"} : <i3> to <i4>
    %146 = source {handshake.bb = 4 : ui32, handshake.name = "source5"} : <>
    %147 = constant %146 {handshake.bb = 4 : ui32, handshake.name = "constant20", value = 3 : i3} : <>, <i3>
    %148 = extsi %147 {handshake.bb = 4 : ui32, handshake.name = "extsi26"} : <i3> to <i4>
    %149 = source {handshake.bb = 4 : ui32, handshake.name = "source6"} : <>
    %150 = constant %149 {handshake.bb = 4 : ui32, handshake.name = "constant21", value = 1 : i2} : <>, <i2>
    %151 = extsi %150 {handshake.bb = 4 : ui32, handshake.name = "extsi27"} : <i2> to <i4>
    %152 = addi %145, %151 {handshake.bb = 4 : ui32, handshake.name = "addi6"} : <i4>
    %153 = buffer %152, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer35"} : <i4>
    %154:2 = fork [2] %153 {handshake.bb = 4 : ui32, handshake.name = "fork26"} : <i4>
    %155 = trunci %154#0 {handshake.bb = 4 : ui32, handshake.name = "trunci8"} : <i4> to <i3>
    %156 = cmpi ult, %154#1, %148 {handshake.bb = 4 : ui32, handshake.name = "cmpi2"} : <i4>
    %157 = buffer %156, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 4 : ui32, handshake.name = "buffer36"} : <i1>
    %158:4 = fork [4] %157 {handshake.bb = 4 : ui32, handshake.name = "fork27"} : <i1>
    %trueResult_51, %falseResult_52 = cond_br %158#0, %155 {handshake.bb = 4 : ui32, handshake.name = "cond_br10"} : <i1>, <i3>
    sink %falseResult_52 {handshake.name = "sink16"} : <i3>
    %trueResult_53, %falseResult_54 = cond_br %158#3, %falseResult_48 {handshake.bb = 4 : ui32, handshake.name = "cond_br11"} : <i1>, <>
    %159:2 = fork [2] %falseResult_54 {handshake.bb = 5 : ui32, handshake.name = "fork28"} : <>
    end {handshake.bb = 5 : ui32, handshake.name = "end0"} %memEnd_1, %memEnd, %0#1 : <>, <>, <>
  }
}

