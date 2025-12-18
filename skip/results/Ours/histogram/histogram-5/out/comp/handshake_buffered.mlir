module {
  handshake.func @histogram(%arg0: memref<1000xi32>, %arg1: memref<1000xf32>, %arg2: memref<1000xf32>, %arg3: !handshake.channel<i32>, %arg4: !handshake.control<>, %arg5: !handshake.control<>, %arg6: !handshake.control<>, %arg7: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["feature", "weight", "hist", "n", "feature_start", "weight_start", "hist_start", "start"], cfg.edges = "[0,1][2,1][1,2,3,cmpi0]", resNames = ["feature_end", "weight_end", "hist_end", "end"]} {
    %0:10 = fork [10] %arg7 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork0"} : <>
    %outputs:2, %memEnd = mem_controller[%arg2 : memref<1000xf32>] %arg6 (%76, %addressResult_44, %addressResult_46, %dataResult_47) %147#2 {connectedBlocks = [2 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "5": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller2"} :    (!handshake.channel<i32>, !handshake.channel<i10>, !handshake.channel<i10>, !handshake.channel<f32>) -> (!handshake.channel<f32>, !handshake.control<>)
    %outputs_0, %memEnd_1 = mem_controller[%arg1 : memref<1000xf32>] %arg5 (%addressResult_32) %147#1 {connectedBlocks = [2 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller3"} :    (!handshake.channel<i10>) -> !handshake.channel<f32>
    %outputs_2, %memEnd_3 = mem_controller[%arg0 : memref<1000xi32>] %arg4 (%addressResult) %147#0 {connectedBlocks = [2 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller4"} :    (!handshake.channel<i10>) -> !handshake.channel<i32>
    %1 = constant %0#1 {handshake.bb = 0 : ui32, handshake.name = "constant8", value = 1000 : i11} : <>, <i11>
    %2:5 = fork [5] %1 {handshake.bb = 0 : ui32, handshake.name = "fork1"} : <i11>
    %3 = extsi %2#0 {handshake.bb = 1 : ui32, handshake.name = "extsi0"} : <i11> to <i32>
    %4 = extsi %2#1 {handshake.bb = 1 : ui32, handshake.name = "extsi1"} : <i11> to <i32>
    %5 = extsi %2#2 {handshake.bb = 1 : ui32, handshake.name = "extsi2"} : <i11> to <i32>
    %6 = extsi %2#3 {handshake.bb = 1 : ui32, handshake.name = "extsi3"} : <i11> to <i32>
    %7 = extsi %2#4 {handshake.bb = 1 : ui32, handshake.name = "extsi4"} : <i11> to <i32>
    %8 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant9", value = false} : <>, <i1>
    %9 = br %8 {handshake.bb = 0 : ui32, handshake.name = "br2"} : <i1>
    %10 = extsi %9 {handshake.bb = 0 : ui32, handshake.name = "extsi8"} : <i1> to <i32>
    %11 = br %arg3 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "br3"} : <i32>
    %12 = br %0#9 {handshake.bb = 0 : ui32, handshake.name = "br4"} : <>
    %13 = mux %32#0 [%3, %125#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux2"} : <i1>, [<i32>, <i32>] to <i32>
    %14 = mux %32#1 [%4, %15] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux3"} : <i1>, [<i32>, <i32>] to <i32>
    %15 = buffer %122#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer9"} : <i32>
    %16 = mux %32#2 [%5, %128#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux4"} : <i1>, [<i32>, <i32>] to <i32>
    %17 = mux %32#3 [%0#8, %141] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux5"} : <i1>, [<>, <>] to <>
    %18 = mux %32#4 [%6, %120#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux6"} : <i1>, [<i32>, <i32>] to <i32>
    %19 = mux %32#5 [%7, %129] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux7"} : <i1>, [<i32>, <i32>] to <i32>
    %20 = mux %21 [%0#7, %134#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux8"} : <i1>, [<>, <>] to <>
    %21 = buffer %32#6, bufferType = FIFO_BREAK_NONE, numSlots = 4 {handshake.bb = 1 : ui32, handshake.name = "buffer17"} : <i1>
    %22 = mux %23 [%0#6, %136#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux9"} : <i1>, [<>, <>] to <>
    %23 = buffer %32#7, bufferType = FIFO_BREAK_NONE, numSlots = 3 {handshake.bb = 1 : ui32, handshake.name = "buffer18"} : <i1>
    %24 = mux %25 [%0#5, %138#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux10"} : <i1>, [<>, <>] to <>
    %25 = buffer %32#8, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 1 : ui32, handshake.name = "buffer19"} : <i1>
    %26 = mux %27 [%0#4, %140#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux11"} : <i1>, [<>, <>] to <>
    %27 = buffer %32#9, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer20"} : <i1>
    %28 = mux %29 [%0#3, %132#1] {ftd.phi, handshake.bb = 1 : ui32, handshake.name = "mux12"} : <i1>, [<>, <>] to <>
    %29 = buffer %32#10, bufferType = FIFO_BREAK_NONE, numSlots = 5 {handshake.bb = 1 : ui32, handshake.name = "buffer21"} : <i1>
    %30 = init %31 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "init0"} : <i1>
    %31 = buffer %44#14, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 1 : ui32, handshake.name = "buffer22"} : <i1>
    %32:11 = fork [11] %30 {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <i1>
    %33 = mux %41#0 [%10, %144] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i32>, <i32>] to <i32>
    %34 = buffer %33, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer15"} : <i32>
    %35 = buffer %34, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer16"} : <i32>
    %36:2 = fork [2] %35 {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <i32>
    %37 = mux %41#1 [%11, %145] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i1>, [<i32>, <i32>] to <i32>
    %38 = buffer %37, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer23"} : <i32>
    %39 = buffer %38, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer24"} : <i32>
    %40:2 = fork [2] %39 {handshake.bb = 1 : ui32, handshake.name = "fork4"} : <i32>
    %result, %index = control_merge [%12, %146]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %41:2 = fork [2] %index {handshake.bb = 1 : ui32, handshake.name = "fork5"} : <i1>
    %42 = cmpi slt, %36#1, %40#1 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i32>
    %43 = buffer %42, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer25"} : <i1>
    %44:15 = fork [15] %43 {handshake.bb = 1 : ui32, handshake.name = "fork6"} : <i1>
    %trueResult, %falseResult = cond_br %44#13, %40#0 {handshake.bb = 1 : ui32, handshake.name = "cond_br1"} : <i1>, <i32>
    sink %falseResult {handshake.name = "sink0"} : <i32>
    %trueResult_4, %falseResult_5 = cond_br %44#12, %36#0 {handshake.bb = 1 : ui32, handshake.name = "cond_br2"} : <i1>, <i32>
    sink %falseResult_5 {handshake.name = "sink1"} : <i32>
    %trueResult_6, %falseResult_7 = cond_br %44#11, %result {handshake.bb = 1 : ui32, handshake.name = "cond_br3"} : <i1>, <>
    %45 = buffer %24, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer12"} : <>
    %trueResult_8, %falseResult_9 = cond_br %46, %45 {handshake.bb = 2 : ui32, handshake.name = "cond_br19"} : <i1>, <>
    %46 = buffer %44#10, bufferType = FIFO_BREAK_NONE, numSlots = 3 {handshake.bb = 2 : ui32, handshake.name = "buffer32"} : <i1>
    sink %falseResult_9 {handshake.name = "sink2"} : <>
    %47 = buffer %28, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer14"} : <>
    %trueResult_10, %falseResult_11 = cond_br %48, %47 {handshake.bb = 2 : ui32, handshake.name = "cond_br20"} : <i1>, <>
    %48 = buffer %44#9, bufferType = FIFO_BREAK_NONE, numSlots = 6 {handshake.bb = 2 : ui32, handshake.name = "buffer33"} : <i1>
    sink %falseResult_11 {handshake.name = "sink3"} : <>
    %49 = buffer %18, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer7"} : <i32>
    %trueResult_12, %falseResult_13 = cond_br %50, %49 {handshake.bb = 2 : ui32, handshake.name = "cond_br21"} : <i1>, <i32>
    %50 = buffer %44#8, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer34"} : <i1>
    sink %falseResult_13 {handshake.name = "sink4"} : <i32>
    %51 = buffer %17, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer5"} : <>
    %52 = buffer %51, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer6"} : <>
    %trueResult_14, %falseResult_15 = cond_br %53, %52 {handshake.bb = 2 : ui32, handshake.name = "cond_br22"} : <i1>, <>
    %53 = buffer %44#7, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer35"} : <i1>
    sink %falseResult_15 {handshake.name = "sink5"} : <>
    %54 = buffer %16, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer4"} : <i32>
    %trueResult_16, %falseResult_17 = cond_br %55, %54 {handshake.bb = 2 : ui32, handshake.name = "cond_br23"} : <i1>, <i32>
    %55 = buffer %44#6, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer36"} : <i1>
    sink %falseResult_17 {handshake.name = "sink6"} : <i32>
    %56 = buffer %19, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer8"} : <i32>
    %trueResult_18, %falseResult_19 = cond_br %57, %56 {handshake.bb = 2 : ui32, handshake.name = "cond_br24"} : <i1>, <i32>
    %57 = buffer %44#5, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer37"} : <i1>
    sink %falseResult_19 {handshake.name = "sink7"} : <i32>
    %58 = buffer %14, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer3"} : <i32>
    %trueResult_20, %falseResult_21 = cond_br %59, %58 {handshake.bb = 2 : ui32, handshake.name = "cond_br25"} : <i1>, <i32>
    %59 = buffer %44#4, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer38"} : <i1>
    sink %falseResult_21 {handshake.name = "sink8"} : <i32>
    %60 = buffer %13, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer2"} : <i32>
    %trueResult_22, %falseResult_23 = cond_br %61, %60 {handshake.bb = 2 : ui32, handshake.name = "cond_br26"} : <i1>, <i32>
    %61 = buffer %44#3, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer39"} : <i1>
    sink %falseResult_23 {handshake.name = "sink9"} : <i32>
    %62 = buffer %22, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer11"} : <>
    %trueResult_24, %falseResult_25 = cond_br %63, %62 {handshake.bb = 2 : ui32, handshake.name = "cond_br27"} : <i1>, <>
    %63 = buffer %44#2, bufferType = FIFO_BREAK_NONE, numSlots = 4 {handshake.bb = 2 : ui32, handshake.name = "buffer40"} : <i1>
    sink %falseResult_25 {handshake.name = "sink10"} : <>
    %64 = buffer %20, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer10"} : <>
    %trueResult_26, %falseResult_27 = cond_br %65, %64 {handshake.bb = 2 : ui32, handshake.name = "cond_br28"} : <i1>, <>
    %65 = buffer %44#1, bufferType = FIFO_BREAK_NONE, numSlots = 5 {handshake.bb = 2 : ui32, handshake.name = "buffer41"} : <i1>
    sink %falseResult_27 {handshake.name = "sink11"} : <>
    %66 = buffer %26, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer13"} : <>
    %trueResult_28, %falseResult_29 = cond_br %67, %66 {handshake.bb = 2 : ui32, handshake.name = "cond_br29"} : <i1>, <>
    %67 = buffer %44#0, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 2 : ui32, handshake.name = "buffer42"} : <i1>
    sink %falseResult_29 {handshake.name = "sink12"} : <>
    %68 = merge %trueResult {handshake.bb = 2 : ui32, handshake.name = "merge0"} : <i32>
    %69 = merge %trueResult_4 {handshake.bb = 2 : ui32, handshake.name = "merge1"} : <i32>
    %70:3 = fork [3] %69 {handshake.bb = 2 : ui32, handshake.name = "fork7"} : <i32>
    %71 = trunci %70#0 {handshake.bb = 2 : ui32, handshake.name = "trunci0"} : <i32> to <i10>
    %72 = trunci %70#1 {handshake.bb = 2 : ui32, handshake.name = "trunci1"} : <i32> to <i10>
    %73 = buffer %trueResult_6, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer26"} : <>
    %result_30, %index_31 = control_merge [%73]  {handshake.bb = 2 : ui32, handshake.name = "control_merge1"} : [<>] to <>, <i1>
    sink %index_31 {handshake.name = "sink13"} : <i1>
    %74:2 = fork [2] %result_30 {handshake.bb = 2 : ui32, handshake.name = "fork8"} : <>
    %75 = constant %74#0 {handshake.bb = 2 : ui32, handshake.name = "constant10", value = 1 : i2} : <>, <i2>
    %76 = extsi %75 {handshake.bb = 2 : ui32, handshake.name = "extsi6"} : <i2> to <i32>
    %77 = source {handshake.bb = 2 : ui32, handshake.name = "source0"} : <>
    %78 = constant %77 {handshake.bb = 2 : ui32, handshake.name = "constant11", value = 1 : i2} : <>, <i2>
    %79 = extsi %78 {handshake.bb = 2 : ui32, handshake.name = "extsi7"} : <i2> to <i32>
    %addressResult, %dataResult = load[%72] %outputs_2 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i10>, <i32>, <i10>, <i32>
    %80:4 = fork [4] %dataResult {handshake.bb = 2 : ui32, handshake.name = "fork9"} : <i32>
    %81 = trunci %82 {handshake.bb = 2 : ui32, handshake.name = "trunci2"} : <i32> to <i10>
    %82 = buffer %80#0, bufferType = FIFO_BREAK_NONE, numSlots = 6 {handshake.bb = 2 : ui32, handshake.name = "buffer45"} : <i32>
    %addressResult_32, %dataResult_33 = load[%71] %outputs_0 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i10>, <f32>, <i10>, <f32>
    %83 = gate %80#1, %trueResult_14 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "gate0"} : <i32>, !handshake.control<> to <i32>
    %84:5 = fork [5] %83 {handshake.bb = 2 : ui32, handshake.name = "fork10"} : <i32>
    %85 = cmpi ne, %84#4, %trueResult_12 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi1"} : <i32>
    %86:2 = fork [2] %85 {handshake.bb = 2 : ui32, handshake.name = "fork11"} : <i1>
    %87 = cmpi ne, %84#3, %trueResult_20 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi2"} : <i32>
    %88:2 = fork [2] %87 {handshake.bb = 2 : ui32, handshake.name = "fork12"} : <i1>
    %89 = cmpi ne, %84#2, %trueResult_22 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi3"} : <i32>
    %90:2 = fork [2] %89 {handshake.bb = 2 : ui32, handshake.name = "fork13"} : <i1>
    %91 = cmpi ne, %84#1, %trueResult_16 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi4"} : <i32>
    %92:2 = fork [2] %91 {handshake.bb = 2 : ui32, handshake.name = "fork14"} : <i1>
    %93 = cmpi ne, %84#0, %trueResult_18 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "cmpi5"} : <i32>
    %94:2 = fork [2] %93 {handshake.bb = 2 : ui32, handshake.name = "fork15"} : <i1>
    %trueResult_34, %falseResult_35 = cond_br %95, %trueResult_10 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br14"} : <i1>, <>
    %95 = buffer %86#1, bufferType = FIFO_BREAK_NONE, numSlots = 5 {handshake.bb = 2 : ui32, handshake.name = "buffer52"} : <i1>
    sink %trueResult_34 {handshake.name = "sink14"} : <>
    %trueResult_36, %falseResult_37 = cond_br %96, %trueResult_26 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br15"} : <i1>, <>
    %96 = buffer %88#1, bufferType = FIFO_BREAK_NONE, numSlots = 4 {handshake.bb = 2 : ui32, handshake.name = "buffer53"} : <i1>
    sink %trueResult_36 {handshake.name = "sink15"} : <>
    %trueResult_38, %falseResult_39 = cond_br %97, %trueResult_24 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br16"} : <i1>, <>
    %97 = buffer %90#1, bufferType = FIFO_BREAK_NONE, numSlots = 3 {handshake.bb = 2 : ui32, handshake.name = "buffer54"} : <i1>
    sink %trueResult_38 {handshake.name = "sink16"} : <>
    %trueResult_40, %falseResult_41 = cond_br %98, %trueResult_8 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br17"} : <i1>, <>
    %98 = buffer %92#1, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 2 : ui32, handshake.name = "buffer55"} : <i1>
    sink %trueResult_40 {handshake.name = "sink17"} : <>
    %trueResult_42, %falseResult_43 = cond_br %99, %trueResult_28 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "cond_br18"} : <i1>, <>
    %99 = buffer %94#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer56"} : <i1>
    sink %trueResult_42 {handshake.name = "sink18"} : <>
    %100 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %101 = mux %86#0 [%falseResult_35, %100] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux13"} : <i1>, [<>, <>] to <>
    %102 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source2"} : <>
    %103 = mux %88#0 [%falseResult_37, %102] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux14"} : <i1>, [<>, <>] to <>
    %104 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source3"} : <>
    %105 = mux %90#0 [%falseResult_39, %104] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux15"} : <i1>, [<>, <>] to <>
    %106 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source4"} : <>
    %107 = mux %92#0 [%falseResult_41, %106] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux16"} : <i1>, [<>, <>] to <>
    %108 = source {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "source5"} : <>
    %109 = mux %94#0 [%falseResult_43, %108] {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "mux17"} : <i1>, [<>, <>] to <>
    %110 = buffer %101, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer27"} : <>
    %111 = buffer %103, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer28"} : <>
    %112 = buffer %105, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer29"} : <>
    %113 = buffer %107, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer30"} : <>
    %114 = buffer %109, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer31"} : <>
    %115 = join %110, %111, %112, %113, %114 {drawing = #handshake<drawing["Conditional_Sequentializer"]>, handshake.bb = 2 : ui32, handshake.name = "join0"} : <>
    %116 = gate %80#2, %115 {handshake.bb = 2 : ui32, handshake.name = "gate1"} : <i32>, !handshake.control<> to <i32>
    %117 = trunci %116 {handshake.bb = 2 : ui32, handshake.name = "trunci3"} : <i32> to <i10>
    %addressResult_44, %dataResult_45 = load[%117] %outputs#0 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["store1", 0, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load5"} : <i10>, <f32>, <i10>, <f32>
    %118 = addf %dataResult_45, %dataResult_33 {fastmath = #arith.fastmath<none>, fpu_impl = #handshake<fpu_impl flopoco>, handshake.bb = 2 : ui32, handshake.name = "addf0", internal_delay = "2_922000"} : <f32>
    %119 = buffer %80#3, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer0"} : <i32>
    %120:2 = fork [2] %119 {handshake.bb = 2 : ui32, handshake.name = "fork16"} : <i32>
    %121 = init %120#0 {handshake.bb = 2 : ui32, handshake.name = "init11"} : <i32>
    %122:2 = fork [2] %121 {handshake.bb = 2 : ui32, handshake.name = "fork17"} : <i32>
    %123 = init %124 {handshake.bb = 2 : ui32, handshake.name = "init12"} : <i32>
    %124 = buffer %122#0, bufferType = FIFO_BREAK_NONE, numSlots = 2 {handshake.bb = 2 : ui32, handshake.name = "buffer65"} : <i32>
    %125:2 = fork [2] %123 {handshake.bb = 2 : ui32, handshake.name = "fork18"} : <i32>
    %126 = init %127 {handshake.bb = 2 : ui32, handshake.name = "init13"} : <i32>
    %127 = buffer %125#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer66"} : <i32>
    %128:2 = fork [2] %126 {handshake.bb = 2 : ui32, handshake.name = "fork19"} : <i32>
    %129 = init %130 {handshake.bb = 2 : ui32, handshake.name = "init14"} : <i32>
    %130 = buffer %128#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer67"} : <i32>
    %131 = buffer %doneResult, bufferType = FIFO_BREAK_NONE, numSlots = 1 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "buffer1"} : <>
    %132:2 = fork [2] %131 {handshake.bb = 2 : ui32, handshake.name = "fork20"} : <>
    %133 = init %132#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init15"} : <>
    %134:2 = fork [2] %133 {handshake.bb = 2 : ui32, handshake.name = "fork21"} : <>
    %135 = init %134#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init16"} : <>
    %136:2 = fork [2] %135 {handshake.bb = 2 : ui32, handshake.name = "fork22"} : <>
    %137 = init %136#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init17"} : <>
    %138:2 = fork [2] %137 {handshake.bb = 2 : ui32, handshake.name = "fork23"} : <>
    %139 = init %138#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init18"} : <>
    %140:2 = fork [2] %139 {handshake.bb = 2 : ui32, handshake.name = "fork24"} : <>
    %141 = init %140#0 {drawing = #handshake<drawing["Condition_Generator"]>, handshake.bb = 2 : ui32, handshake.name = "init19"} : <>
    %addressResult_46, %dataResult_47, %doneResult = store[%81] %118 %outputs#1 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.deps = #handshake<deps[["load5", 0, false], ["store1", 0, false]]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store1"} : <i10>, <f32>, <>, <i10>, <f32>, <>
    %142 = addi %70#2, %79 {handshake.bb = 2 : ui32, handshake.name = "addi0"} : <i32>
    %143 = buffer %142, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer43"} : <i32>
    %144 = br %143 {handshake.bb = 2 : ui32, handshake.name = "br5"} : <i32>
    %145 = br %68 {handshake.bb = 2 : ui32, handshake.name = "br6"} : <i32>
    %146 = br %74#1 {handshake.bb = 2 : ui32, handshake.name = "br7"} : <>
    %result_48, %index_49 = control_merge [%falseResult_7]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>] to <>, <i1>
    sink %index_49 {handshake.name = "sink19"} : <i1>
    %147:3 = fork [3] %result_48 {handshake.bb = 3 : ui32, handshake.name = "fork25"} : <>
    end {handshake.bb = 3 : ui32, handshake.name = "end0"} %memEnd_3, %memEnd_1, %memEnd, %0#2 : <>, <>, <>, <>
  }
}

