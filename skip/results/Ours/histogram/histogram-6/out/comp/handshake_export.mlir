module {
  handshake.func @histogram(%arg0: memref<1000xi32>, %arg1: memref<1000xf32>, %arg2: memref<1000xf32>, %arg3: !handshake.channel<i32>, %arg4: !handshake.control<>, %arg5: !handshake.control<>, %arg6: !handshake.control<>, %arg7: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["feature", "weight", "hist", "n", "feature_start", "weight_start", "hist_start", "start"], cfg.edges = "[0,1][2,1][1,2,3,cmpi0]", resNames = ["feature_end", "weight_end", "hist_end", "end"]} {
    %0:11 = fork [11] %arg7 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork0"} : <>
    %outputs:2, %memEnd = mem_controller[%arg2 : memref<1000xf32>] %arg6 (%80, %addressResult_48, %addressResult_50, %dataResult_51) %159#2 {connectedBlocks = [2 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "5": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller2"} :    (!handshake.channel<i32>, !handshake.channel<i10>, !handshake.channel<i10>, !handshake.channel<f32>) -> (!handshake.channel<f32>, !handshake.control<>)
    %outputs_0, %memEnd_1 = mem_controller[%arg1 : memref<1000xf32>] %arg5 (%addressResult_34) %159#1 {connectedBlocks = [2 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller3"} :    (!handshake.channel<i10>) -> !handshake.channel<f32>
    %outputs_2, %memEnd_3 = mem_controller[%arg0 : memref<1000xi32>] %arg4 (%addressResult) %159#0 {connectedBlocks = [2 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller4"} :    (!handshake.channel<i10>) -> !handshake.channel<i32>
    %1 = constant %0#1 {handshake.bb = 0 : ui32, handshake.name = "constant9", value = 1000 : i11} : <>, <i11>
    %2:6 = fork [6] %1 {handshake.bb = 0 : ui32, handshake.name = "fork1"} : <i11>
    %3 = extsi %2#0 {handshake.bb = 1 : ui32, handshake.name = "extsi0"} : <i11> to <i32>
    %4 = extsi %2#1 {handshake.bb = 1 : ui32, handshake.name = "extsi1"} : <i11> to <i32>
    %5 = extsi %2#2 {handshake.bb = 1 : ui32, handshake.name = "extsi2"} : <i11> to <i32>
    %6 = extsi %2#3 {handshake.bb = 1 : ui32, handshake.name = "extsi3"} : <i11> to <i32>
    %7 = extsi %2#4 {handshake.bb = 1 : ui32, handshake.name = "extsi4"} : <i11> to <i32>
    %8 = extsi %2#5 {handshake.bb = 1 : ui32, handshake.name = "extsi5"} : <i11> to <i32>
    %9 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant10", value = false} : <>, <i1>
    %10 = extsi %9 {handshake.bb = 0 : ui32, handshake.name = "extsi9"} : <i1> to <i32>
    %11 = mux %33#0 [%3, %12] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux2"} : <i1>, [<i32>, <i32>] to <i32>
    %12 = buffer %132#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer8"} : <i32>
    %13 = mux %33#1 [%4, %138#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux3"} : <i1>, [<i32>, <i32>] to <i32>
    %14 = mux %33#2 [%5, %135#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux4"} : <i1>, [<i32>, <i32>] to <i32>
    %15 = mux %33#3 [%6, %130#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux5"} : <i1>, [<i32>, <i32>] to <i32>
    %16 = mux %33#4 [%7, %141#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux6"} : <i1>, [<i32>, <i32>] to <i32>
    %17 = mux %33#5 [%0#9, %156] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux7"} : <i1>, [<>, <>] to <>
    %18 = mux %33#6 [%8, %142] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux8"} : <i1>, [<i32>, <i32>] to <i32>
    %19 = mux %20 [%0#8, %155#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux9"} : <i1>, [<>, <>] to <>
    %20 = buffer %33#7, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer20"} : <i1>
    %21 = mux %22 [%0#7, %147#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux10"} : <i1>, [<>, <>] to <>
    %22 = buffer %33#8, bufferType = FIFO_BREAK_NONE, numSlots = 5 {handshake.bb = 1 : ui32, handshake.name = "buffer21"} : <i1>
    %23 = mux %24 [%0#6, %153#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux11"} : <i1>, [<>, <>] to <>
    %24 = buffer %33#9, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 1 : ui32, handshake.name = "buffer22"} : <i1>
    %25 = mux %26 [%0#5, %149#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux12"} : <i1>, [<>, <>] to <>
    %26 = buffer %33#10, bufferType = FIFO_BREAK_NONE, numSlots = 4 {handshake.bb = 1 : ui32, handshake.name = "buffer23"} : <i1>
    %27 = mux %28 [%0#4, %145#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux13"} : <i1>, [<>, <>] to <>
    %28 = buffer %33#11, bufferType = FIFO_BREAK_NONE, numSlots = 6 {handshake.bb = 1 : ui32, handshake.name = "buffer24"} : <i1>
    %29 = mux %30 [%0#3, %151#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux14"} : <i1>, [<>, <>] to <>
    %30 = buffer %33#12, bufferType = FIFO_BREAK_NONE, numSlots = 3 {handshake.bb = 1 : ui32, handshake.name = "buffer25"} : <i1>
    %31 = init %32 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "init0"} : <i1>
    %32 = buffer %46#16, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 1 : ui32, handshake.name = "buffer26"} : <i1>
    %33:13 = fork [13] %31 {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <i1>
    %34 = buffer %157, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer50"} : <i32>
    %35 = mux %43#0 [%10, %34] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i32>, <i32>] to <i32>
    %36 = buffer %35, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer17"} : <i32>
    %37 = buffer %36, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer18"} : <i32>
    %38:2 = fork [2] %37 {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <i32>
    %39 = mux %43#1 [%arg3, %trueResult] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i1>, [<i32>, <i32>] to <i32>
    %40 = buffer %39, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer19"} : <i32>
    %41 = buffer %40, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer27"} : <i32>
    %42:2 = fork [2] %41 {handshake.bb = 1 : ui32, handshake.name = "fork4"} : <i32>
    %result, %index = control_merge [%0#10, %158]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %43:2 = fork [2] %index {handshake.bb = 1 : ui32, handshake.name = "fork5"} : <i1>
    %44 = cmpi slt, %38#1, %42#1 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i32>
    %45 = buffer %44, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer28"} : <i1>
    %46:17 = fork [17] %45 {handshake.bb = 1 : ui32, handshake.name = "fork6"} : <i1>
    %trueResult, %falseResult = cond_br %46#15, %42#0 {handshake.bb = 1 : ui32, handshake.name = "cond_br1"} : <i1>, <i32>
    sink %falseResult {handshake.name = "sink0"} : <i32>
    %trueResult_4, %falseResult_5 = cond_br %46#14, %38#0 {handshake.bb = 1 : ui32, handshake.name = "cond_br2"} : <i1>, <i32>
    sink %falseResult_5 {handshake.name = "sink1"} : <i32>
    %trueResult_6, %falseResult_7 = cond_br %46#13, %result {handshake.bb = 1 : ui32, handshake.name = "cond_br3"} : <i1>, <>
    %47 = buffer %11, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer2"} : <i32>
    %trueResult_8, %falseResult_9 = cond_br %48, %47 {handshake.bb = 2 : ui32, handshake.name = "cond_br22"} : <i1>, <i32>
    %48 = buffer %46#12, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer36"} : <i1>
    sink %falseResult_9 {handshake.name = "sink2"} : <i32>
    %49 = buffer %16, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer6"} : <i32>
    %trueResult_10, %falseResult_11 = cond_br %50, %49 {handshake.bb = 2 : ui32, handshake.name = "cond_br23"} : <i1>, <i32>
    %50 = buffer %46#11, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer37"} : <i1>
    sink %falseResult_11 {handshake.name = "sink3"} : <i32>
    %51 = buffer %29, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer16"} : <>
    %trueResult_12, %falseResult_13 = cond_br %52, %51 {handshake.bb = 2 : ui32, handshake.name = "cond_br24"} : <i1>, <>
    %52 = buffer %46#10, bufferType = FIFO_BREAK_NONE, numSlots = 4 {handshake.bb = 2 : ui32, handshake.name = "buffer38"} : <i1>
    sink %falseResult_13 {handshake.name = "sink4"} : <>
    %53 = buffer %27, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer15"} : <>
    %trueResult_14, %falseResult_15 = cond_br %54, %53 {handshake.bb = 2 : ui32, handshake.name = "cond_br25"} : <i1>, <>
    %54 = buffer %46#9, bufferType = FIFO_BREAK_NONE, numSlots = 7 {handshake.bb = 2 : ui32, handshake.name = "buffer39"} : <i1>
    sink %falseResult_15 {handshake.name = "sink5"} : <>
    %55 = buffer %15, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer5"} : <i32>
    %trueResult_16, %falseResult_17 = cond_br %56, %55 {handshake.bb = 2 : ui32, handshake.name = "cond_br26"} : <i1>, <i32>
    %56 = buffer %46#8, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer40"} : <i1>
    sink %falseResult_17 {handshake.name = "sink6"} : <i32>
    %57 = buffer %18, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer10"} : <i32>
    %trueResult_18, %falseResult_19 = cond_br %58, %57 {handshake.bb = 2 : ui32, handshake.name = "cond_br27"} : <i1>, <i32>
    %58 = buffer %46#7, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer41"} : <i1>
    sink %falseResult_19 {handshake.name = "sink7"} : <i32>
    %59 = buffer %25, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer14"} : <>
    %trueResult_20, %falseResult_21 = cond_br %60, %59 {handshake.bb = 2 : ui32, handshake.name = "cond_br28"} : <i1>, <>
    %60 = buffer %46#6, bufferType = FIFO_BREAK_NONE, numSlots = 5 {handshake.bb = 2 : ui32, handshake.name = "buffer42"} : <i1>
    sink %falseResult_21 {handshake.name = "sink8"} : <>
    %61 = buffer %13, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer3"} : <i32>
    %trueResult_22, %falseResult_23 = cond_br %62, %61 {handshake.bb = 2 : ui32, handshake.name = "cond_br29"} : <i1>, <i32>
    %62 = buffer %46#5, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer43"} : <i1>
    sink %falseResult_23 {handshake.name = "sink9"} : <i32>
    %63 = buffer %23, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer13"} : <>
    %trueResult_24, %falseResult_25 = cond_br %64, %63 {handshake.bb = 2 : ui32, handshake.name = "cond_br30"} : <i1>, <>
    %64 = buffer %46#4, bufferType = FIFO_BREAK_NONE, numSlots = 3 {handshake.bb = 2 : ui32, handshake.name = "buffer44"} : <i1>
    sink %falseResult_25 {handshake.name = "sink10"} : <>
    %65 = buffer %21, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer12"} : <>
    %trueResult_26, %falseResult_27 = cond_br %66, %65 {handshake.bb = 2 : ui32, handshake.name = "cond_br31"} : <i1>, <>
    %66 = buffer %46#3, bufferType = FIFO_BREAK_NONE, numSlots = 6 {handshake.bb = 2 : ui32, handshake.name = "buffer45"} : <i1>
    sink %falseResult_27 {handshake.name = "sink11"} : <>
    %67 = buffer %17, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer7"} : <>
    %68 = buffer %67, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer9"} : <>
    %trueResult_28, %falseResult_29 = cond_br %69, %68 {handshake.bb = 2 : ui32, handshake.name = "cond_br32"} : <i1>, <>
    %69 = buffer %46#2, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer46"} : <i1>
    sink %falseResult_29 {handshake.name = "sink12"} : <>
    %70 = buffer %19, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer11"} : <>
    %trueResult_30, %falseResult_31 = cond_br %71, %70 {handshake.bb = 2 : ui32, handshake.name = "cond_br33"} : <i1>, <>
    %71 = buffer %46#1, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 2 : ui32, handshake.name = "buffer47"} : <i1>
    sink %falseResult_31 {handshake.name = "sink13"} : <>
    %72 = buffer %14, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer4"} : <i32>
    %trueResult_32, %falseResult_33 = cond_br %73, %72 {handshake.bb = 2 : ui32, handshake.name = "cond_br34"} : <i1>, <i32>
    %73 = buffer %46#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer48"} : <i1>
    sink %falseResult_33 {handshake.name = "sink14"} : <i32>
    %74:3 = fork [3] %trueResult_4 {handshake.bb = 2 : ui32, handshake.name = "fork7"} : <i32>
    %75 = trunci %76 {handshake.bb = 2 : ui32, handshake.name = "trunci0"} : <i32> to <i10>
    %76 = buffer %74#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer49"} : <i32>
    %77 = trunci %74#1 {handshake.bb = 2 : ui32, handshake.name = "trunci1"} : <i32> to <i10>
    %78:2 = fork [2] %trueResult_6 {handshake.bb = 2 : ui32, handshake.name = "fork8"} : <>
    %79 = constant %78#0 {handshake.bb = 2 : ui32, handshake.name = "constant11", value = 1 : i2} : <>, <i2>
    %80 = extsi %79 {handshake.bb = 2 : ui32, handshake.name = "extsi7"} : <i2> to <i32>
    %81 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %82 = constant %81 {handshake.bb = 2 : ui32, handshake.name = "constant12", value = 1 : i2} : <>, <i2>
    %83 = extsi %82 {handshake.bb = 2 : ui32, handshake.name = "extsi8"} : <i2> to <i32>
    %addressResult, %dataResult = load[%77] %outputs_2 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i10>, <i32>, <i10>, <i32>
    %84:4 = fork [4] %dataResult {handshake.bb = 2 : ui32, handshake.name = "fork9"} : <i32>
    %85 = trunci %86 {handshake.bb = 2 : ui32, handshake.name = "trunci2"} : <i32> to <i10>
    %86 = buffer %84#0, bufferType = FIFO_BREAK_NONE, numSlots = 6 {handshake.bb = 2 : ui32, handshake.name = "buffer51"} : <i32>
    %addressResult_34, %dataResult_35 = load[%75] %outputs_0 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i10>, <f32>, <i10>, <f32>
    %87 = gate %84#1, %trueResult_28 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "gate0"} : <i32>, !handshake.control<> to <i32>
    %88:6 = fork [6] %87 {handshake.bb = 2 : ui32, handshake.name = "fork10"} : <i32>
    %89 = cmpi ne, %88#5, %trueResult_16 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi1"} : <i32>
    %90:2 = fork [2] %89 {handshake.bb = 2 : ui32, handshake.name = "fork11"} : <i1>
    %91 = cmpi ne, %88#4, %trueResult_8 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi2"} : <i32>
    %92:2 = fork [2] %91 {handshake.bb = 2 : ui32, handshake.name = "fork12"} : <i1>
    %93 = cmpi ne, %88#3, %trueResult_32 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi3"} : <i32>
    %94:2 = fork [2] %93 {handshake.bb = 2 : ui32, handshake.name = "fork13"} : <i1>
    %95 = cmpi ne, %88#2, %trueResult_22 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi4"} : <i32>
    %96:2 = fork [2] %95 {handshake.bb = 2 : ui32, handshake.name = "fork14"} : <i1>
    %97 = cmpi ne, %88#1, %trueResult_10 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi5"} : <i32>
    %98:2 = fork [2] %97 {handshake.bb = 2 : ui32, handshake.name = "fork15"} : <i1>
    %99 = cmpi ne, %88#0, %trueResult_18 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi6"} : <i32>
    %100:2 = fork [2] %99 {handshake.bb = 2 : ui32, handshake.name = "fork16"} : <i1>
    %trueResult_36, %falseResult_37 = cond_br %101, %trueResult_14 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br16"} : <i1>, <>
    %101 = buffer %90#1, bufferType = FIFO_BREAK_NONE, numSlots = 6 {handshake.bb = 2 : ui32, handshake.name = "buffer59"} : <i1>
    sink %trueResult_36 {handshake.name = "sink16"} : <>
    %trueResult_38, %falseResult_39 = cond_br %102, %trueResult_26 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br17"} : <i1>, <>
    %102 = buffer %92#1, bufferType = FIFO_BREAK_NONE, numSlots = 5 {handshake.bb = 2 : ui32, handshake.name = "buffer60"} : <i1>
    sink %trueResult_38 {handshake.name = "sink17"} : <>
    %trueResult_40, %falseResult_41 = cond_br %103, %trueResult_20 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br18"} : <i1>, <>
    %103 = buffer %94#1, bufferType = FIFO_BREAK_NONE, numSlots = 4 {handshake.bb = 2 : ui32, handshake.name = "buffer61"} : <i1>
    sink %trueResult_40 {handshake.name = "sink18"} : <>
    %trueResult_42, %falseResult_43 = cond_br %104, %trueResult_12 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br19"} : <i1>, <>
    %104 = buffer %96#1, bufferType = FIFO_BREAK_NONE, numSlots = 3 {handshake.bb = 2 : ui32, handshake.name = "buffer62"} : <i1>
    sink %trueResult_42 {handshake.name = "sink19"} : <>
    %trueResult_44, %falseResult_45 = cond_br %105, %trueResult_24 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br20"} : <i1>, <>
    %105 = buffer %98#1, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 2 : ui32, handshake.name = "buffer63"} : <i1>
    sink %trueResult_44 {handshake.name = "sink20"} : <>
    %trueResult_46, %falseResult_47 = cond_br %106, %trueResult_30 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br21"} : <i1>, <>
    %106 = buffer %100#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer64"} : <i1>
    sink %trueResult_46 {handshake.name = "sink21"} : <>
    %107 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %108 = mux %90#0 [%falseResult_37, %107] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux15"} : <i1>, [<>, <>] to <>
    %109 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source2"} : <>
    %110 = mux %92#0 [%falseResult_39, %109] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux16"} : <i1>, [<>, <>] to <>
    %111 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source3"} : <>
    %112 = mux %94#0 [%falseResult_41, %111] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux17"} : <i1>, [<>, <>] to <>
    %113 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source4"} : <>
    %114 = mux %96#0 [%falseResult_43, %113] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux18"} : <i1>, [<>, <>] to <>
    %115 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source5"} : <>
    %116 = mux %98#0 [%falseResult_45, %115] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux19"} : <i1>, [<>, <>] to <>
    %117 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source6"} : <>
    %118 = mux %100#0 [%falseResult_47, %117] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux20"} : <i1>, [<>, <>] to <>
    %119 = buffer %108, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer30"} : <>
    %120 = buffer %110, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer31"} : <>
    %121 = buffer %112, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer32"} : <>
    %122 = buffer %114, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer33"} : <>
    %123 = buffer %116, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer34"} : <>
    %124 = buffer %118, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer35"} : <>
    %125 = join %119, %120, %121, %122, %123, %124 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "join0"} : <>
    %126 = gate %84#2, %125 {handshake.bb = 2 : ui32, handshake.name = "gate1"} : <i32>, !handshake.control<> to <i32>
    %127 = trunci %126 {handshake.bb = 2 : ui32, handshake.name = "trunci3"} : <i32> to <i10>
    %addressResult_48, %dataResult_49 = load[%127] %outputs#0 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["store1", 0, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load5"} : <i10>, <f32>, <i10>, <f32>
    %128 = addf %dataResult_49, %dataResult_35 {fastmath = #arith.fastmath<none>, fpu_impl = #handshake<fpu_impl flopoco>, handshake.bb = 2 : ui32, handshake.name = "addf0", internal_delay = "2_922000"} : <f32>
    %129 = buffer %84#3, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer0"} : <i32>
    %130:2 = fork [2] %129 {handshake.bb = 2 : ui32, handshake.name = "fork17"} : <i32>
    %131 = init %130#0 {handshake.bb = 2 : ui32, handshake.name = "init13"} : <i32>
    %132:2 = fork [2] %131 {handshake.bb = 2 : ui32, handshake.name = "fork18"} : <i32>
    %133 = init %134 {handshake.bb = 2 : ui32, handshake.name = "init14"} : <i32>
    %134 = buffer %132#0, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 2 : ui32, handshake.name = "buffer74"} : <i32>
    %135:2 = fork [2] %133 {handshake.bb = 2 : ui32, handshake.name = "fork19"} : <i32>
    %136 = init %137 {handshake.bb = 2 : ui32, handshake.name = "init15"} : <i32>
    %137 = buffer %135#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer75"} : <i32>
    %138:2 = fork [2] %136 {handshake.bb = 2 : ui32, handshake.name = "fork20"} : <i32>
    %139 = init %140 {handshake.bb = 2 : ui32, handshake.name = "init16"} : <i32>
    %140 = buffer %138#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer76"} : <i32>
    %141:2 = fork [2] %139 {handshake.bb = 2 : ui32, handshake.name = "fork21"} : <i32>
    %142 = init %143 {handshake.bb = 2 : ui32, handshake.name = "init17"} : <i32>
    %143 = buffer %141#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer77"} : <i32>
    %144 = buffer %doneResult, bufferType = FIFO_BREAK_NONE, numSlots = 1 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "buffer1"} : <>
    %145:2 = fork [2] %144 {handshake.bb = 2 : ui32, handshake.name = "fork22"} : <>
    %146 = init %145#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init18"} : <>
    %147:2 = fork [2] %146 {handshake.bb = 2 : ui32, handshake.name = "fork23"} : <>
    %148 = init %147#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init19"} : <>
    %149:2 = fork [2] %148 {handshake.bb = 2 : ui32, handshake.name = "fork24"} : <>
    %150 = init %149#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init20"} : <>
    %151:2 = fork [2] %150 {handshake.bb = 2 : ui32, handshake.name = "fork25"} : <>
    %152 = init %151#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init21"} : <>
    %153:2 = fork [2] %152 {handshake.bb = 2 : ui32, handshake.name = "fork26"} : <>
    %154 = init %153#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init22"} : <>
    %155:2 = fork [2] %154 {handshake.bb = 2 : ui32, handshake.name = "fork27"} : <>
    %156 = init %155#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init23"} : <>
    %addressResult_50, %dataResult_51, %doneResult = store[%85] %128 %outputs#1 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["load5", 0, false], ["store1", 0, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store1"} : <i10>, <f32>, <>, <i10>, <f32>, <>
    %157 = addi %74#2, %83 {handshake.bb = 2 : ui32, handshake.name = "addi0"} : <i32>
    %158 = buffer %78#1, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer29"} : <>
    %159:3 = fork [3] %falseResult_7 {handshake.bb = 3 : ui32, handshake.name = "fork28"} : <>
    end {handshake.bb = 3 : ui32, handshake.name = "end0"} %memEnd_3, %memEnd_1, %memEnd, %0#2 : <>, <>, <>, <>
  }
}

