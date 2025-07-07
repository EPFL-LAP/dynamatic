module {
  handshake.func @nested_loop(%arg0: memref<1000xi32>, %arg1: memref<1000xi32>, %arg2: memref<1000xi32>, %arg3: !handshake.control<>, %arg4: !handshake.control<>, %arg5: !handshake.control<>, %arg6: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["a", "b", "c", "a_start", "b_start", "c_start", "start"], handshake.cfdfcThroughput = #handshake<cfdfcThroughput {"0" = 1.000000e+00 : f64, "1" = 0.16666666666666666 : f64}>, handshake.cfdfcToBBList = #handshake<cfdfcToBBList {"0" = [2 : ui32], "1" = [1 : ui32, 2 : ui32, 3 : ui32]}>, resNames = ["a_end", "b_end", "c_end", "end"]} {
    %0:3 = fork [3] %arg6 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork0"} : <>
    %memEnd = mem_controller[%arg2 : memref<1000xi32>] %arg5 (%46, %addressResult_5, %dataResult_6) %97#2 {connectedBlocks = [2 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller3"} :    (!handshake.channel<i32>, !handshake.channel<i10>, !handshake.channel<i32>) -> ()
    %outputs, %memEnd_0 = mem_controller[%arg1 : memref<1000xi32>] %arg4 (%addressResult_3) %97#1 {connectedBlocks = [2 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller4"} :    (!handshake.channel<i10>) -> !handshake.channel<i32>
    %outputs_1, %memEnd_2 = mem_controller[%arg0 : memref<1000xi32>] %arg3 (%addressResult) %97#0 {connectedBlocks = [2 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller5"} :    (!handshake.channel<i10>) -> !handshake.channel<i32>
    %1 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant3", value = false} : <>, <i1>
    %2 = extsi %1 {handshake.bb = 0 : ui32, handshake.name = "extsi8"} : <i1> to <i2>
    %3 = mux %index [%2, %trueResult] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i2>, <i2>] to <i2>
    %4 = buffer %3 {handshake.bb = 1 : ui32, handshake.name = "buffer0", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i2>
    %5:2 = fork [2] %4 {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <i2>
    %6 = extsi %5#0 {handshake.bb = 1 : ui32, handshake.name = "extsi9"} : <i2> to <i12>
    %result, %index = control_merge [%0#2, %trueResult_7]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %7:2 = fork [2] %result {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <>
    %8 = source {handshake.bb = 1 : ui32, handshake.name = "source0"} : <>
    %9 = constant %8 {handshake.bb = 1 : ui32, handshake.name = "constant4", value = 400 : i10} : <>, <i10>
    %10 = extsi %9 {handshake.bb = 1 : ui32, handshake.name = "extsi10"} : <i10> to <i12>
    %11 = constant %7#0 {handshake.bb = 1 : ui32, handshake.name = "constant5", value = false} : <>, <i1>
    %12 = muli %6, %10 {handshake.bb = 1 : ui32, handshake.name = "muli0", internal_delay = "1_000000"} : <i12>
    %13 = extsi %11 {handshake.bb = 1 : ui32, handshake.name = "extsi11"} : <i1> to <i32>
    %14 = mux %52#3 [%13, %56] {handshake.bb = 2 : ui32, handshake.name = "mux1"} : <i1>, [<i32>, <i32>] to <i32>
    %15 = buffer %14 {handshake.bb = 2 : ui32, handshake.name = "buffer1", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i32>
    %16:4 = fork [4] %15 {handshake.bb = 2 : ui32, handshake.name = "fork3"} : <i32>
    %17 = trunci %16#0 {handshake.bb = 2 : ui32, handshake.name = "trunci0"} : <i32> to <i10>
    %18 = trunci %16#1 {handshake.bb = 2 : ui32, handshake.name = "trunci1"} : <i32> to <i10>
    %19 = trunci %16#2 {handshake.bb = 2 : ui32, handshake.name = "trunci2"} : <i32> to <i10>
    %20 = mux %52#2 [%5#1, %53] {handshake.bb = 2 : ui32, handshake.name = "mux2"} : <i1>, [<i2>, <i2>] to <i2>
    %21 = buffer %52#1 {handshake.bb = 2 : ui32, handshake.name = "buffer12", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_NONE", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 0}>}} : <i1>
    %22 = mux %21 [%12, %57] {handshake.bb = 2 : ui32, handshake.name = "mux3"} : <i1>, [<i12>, <i12>] to <i12>
    %23 = buffer %22 {handshake.bb = 2 : ui32, handshake.name = "buffer5", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i12>
    %24 = buffer %23 {handshake.bb = 2 : ui32, handshake.name = "buffer6", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i12>
    %25:2 = fork [2] %24 {handshake.bb = 2 : ui32, handshake.name = "fork4"} : <i12>
    %26 = trunci %25#0 {handshake.bb = 2 : ui32, handshake.name = "trunci3"} : <i12> to <i10>
    %27 = mux %52#0 [%7#1, %54] {handshake.bb = 2 : ui32, handshake.name = "mux4"} : <i1>, [<>, <>] to <>
    %28 = buffer %27 {handshake.bb = 2 : ui32, handshake.name = "buffer7", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <>
    %29 = buffer %28 {handshake.bb = 2 : ui32, handshake.name = "buffer8", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %30:3 = fork [3] %29 {handshake.bb = 2 : ui32, handshake.name = "fork17"} : <>
    %31 = constant %30#0 {handshake.bb = 2 : ui32, handshake.name = "constant6", value = 1 : i2} : <>, <i2>
    %32 = buffer %31 {handshake.bb = 2 : ui32, handshake.name = "buffer10", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_NONE", NUM_SLOTS = 4 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 0}>}} : <i2>
    %33 = extsi %32 {handshake.bb = 2 : ui32, handshake.name = "extsi3"} : <i2> to <i32>
    %34 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %35 = constant %34 {handshake.bb = 2 : ui32, handshake.name = "constant13", value = 1 : i2} : <>, <i2>
    %36 = extsi %35 {handshake.bb = 2 : ui32, handshake.name = "extsi4"} : <i2> to <i32>
    %37 = source {handshake.bb = 2 : ui32, handshake.name = "source2"} : <>
    %38 = constant %37 {handshake.bb = 2 : ui32, handshake.name = "constant14", value = 1000 : i11} : <>, <i11>
    %39 = extsi %38 {handshake.bb = 2 : ui32, handshake.name = "extsi5"} : <i11> to <i32>
    %addressResult, %dataResult = load[%19] %outputs_1 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load2"} : <i10>, <i32>, <i10>, <i32>
    %addressResult_3, %dataResult_4 = load[%18] %outputs {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load3"} : <i10>, <i32>, <i10>, <i32>
    %40 = muli %dataResult, %dataResult_4 {handshake.bb = 2 : ui32, handshake.name = "muli1", internal_delay = "1_000000"} : <i32>
    %41:2 = fork [2] %40 {handshake.bb = 2 : ui32, handshake.name = "fork7"} : <i32>
    %42 = buffer %17 {handshake.bb = 2 : ui32, handshake.name = "buffer2", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i10>
    %43 = addi %42, %26 {handshake.bb = 2 : ui32, handshake.name = "addi0", internal_delay = "0_000000"} : <i10>
    %addressResult_5, %dataResult_6 = store[%48] %49 {handshake.bb = 2 : ui32, handshake.deps = #handshake<deps[<"store1" (0)>]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store1"} : <i10>, <i32>, <i10>, <i32>
    %44 = cmpi slt, %41#1, %39 {handshake.bb = 2 : ui32, handshake.name = "cmpi0", internal_delay = "0_000000"} : <i32>
    %45 = not %77#0 {handshake.bb = 2 : ui32, handshake.name = "not0"} : <i1>
    %46 = passer %33[%76#3] {handshake.bb = 2 : ui32, handshake.name = "passer0"} : <i32>, <i1>
    %47 = buffer %43 {handshake.bb = 2 : ui32, handshake.name = "buffer11", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_NONE", NUM_SLOTS = 4 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 0}>}} : <i10>
    %48 = passer %47[%76#2] {handshake.bb = 2 : ui32, handshake.name = "passer1"} : <i10>, <i1>
    %49 = passer %41#0[%76#1] {handshake.bb = 2 : ui32, handshake.name = "passer2"} : <i32>, <i1>
    %50 = buffer %60#5 {handshake.bb = 2 : ui32, handshake.name = "buffer14", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_NONE", NUM_SLOTS = 2 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 0}>}} : <i1>
    %51 = init %50 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 2, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "init0"} : <i1>
    %52:4 = fork [4] %51 {handshake.bb = 2 : ui32, handshake.name = "fork18"} : <i1>
    %53 = passer %81#1[%60#4] {handshake.bb = 2 : ui32, handshake.name = "passer3"} : <i2>, <i1>
    %54 = passer %30#2[%60#3] {handshake.bb = 2 : ui32, handshake.name = "passer4"} : <>, <i1>
    %55 = buffer %78 {handshake.bb = 2 : ui32, handshake.name = "buffer22", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i32>
    %56 = passer %55[%60#2] {handshake.bb = 2 : ui32, handshake.name = "passer5"} : <i32>, <i1>
    %57 = passer %25#1[%60#1] {handshake.bb = 2 : ui32, handshake.name = "passer6"} : <i12>, <i1>
    %58 = buffer %62 {handshake.bb = 2 : ui32, handshake.name = "buffer15", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_NONE", NUM_SLOTS = 2 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 0}>}} : <i1>
    %59 = spec_v2_repeating_init %58 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 2, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "spec_v2_repeating_init0"} : <i1>
    %60:6 = fork [6] %59 {handshake.bb = 2 : ui32, handshake.name = "fork19"} : <i1>
    %61 = buffer %64 {handshake.bb = 2 : ui32, handshake.name = "buffer16", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_NONE", NUM_SLOTS = 2 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 0}>}} : <i1>
    %62 = spec_v2_repeating_init %61 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 2, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "spec_v2_repeating_init1"} : <i1>
    %63 = buffer %66 {handshake.bb = 2 : ui32, handshake.name = "buffer17", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_NONE", NUM_SLOTS = 2 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 0}>}} : <i1>
    %64 = spec_v2_repeating_init %63 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 2, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "spec_v2_repeating_init2"} : <i1>
    %65 = buffer %69 {handshake.bb = 2 : ui32, handshake.name = "buffer18", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_NONE", NUM_SLOTS = 2 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 0}>}} : <i1>
    %66 = spec_v2_repeating_init %65 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 2, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "spec_v2_repeating_init3"} : <i1>
    %67 = buffer %70 {handshake.bb = 2 : ui32, handshake.name = "buffer19", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %68 = buffer %67 {handshake.bb = 2 : ui32, handshake.name = "buffer20", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_NONE", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 0}>}} : <i1>
    %69 = spec_v2_repeating_init %68 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 2, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "spec_v2_repeating_init4"} : <i1>
    %70 = passer %77#2[%76#4] {handshake.bb = 2 : ui32, handshake.name = "passer7"} : <i1>, <i1>
    %71 = andi %76#0, %45 {handshake.bb = 2 : ui32, handshake.name = "andi0", internal_delay = "0_000000"} : <i1>
    %72 = buffer %71 {handshake.bb = 2 : ui32, handshake.name = "buffer21", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %73:2 = fork [2] %72 {handshake.bb = 2 : ui32, handshake.name = "fork20"} : <i1>
    %74 = buffer %60#0 {handshake.bb = 2 : ui32, handshake.name = "buffer13", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_NONE", NUM_SLOTS = 4 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 0}>}} : <i1>
    %75 = spec_v2_resolver %77#1, %74 {handshake.bb = 2 : ui32, handshake.name = "spec_v2_resolver0"} : <i1>
    %76:5 = fork [5] %75 {handshake.bb = 2 : ui32, handshake.name = "fork21"} : <i1>
    %77:3 = fork [3] %44 {handshake.bb = 2 : ui32, handshake.name = "fork22"} : <i1>
    %78 = addi %16#3, %36 {handshake.bb = 2 : ui32, handshake.name = "addi1", internal_delay = "0_000000"} : <i32>
    %79 = buffer %20 {handshake.bb = 2 : ui32, handshake.name = "buffer3", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i2>
    %80 = buffer %79 {handshake.bb = 2 : ui32, handshake.name = "buffer4", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i2>
    %81:2 = fork [2] %80 {handshake.bb = 2 : ui32, handshake.name = "fork16"} : <i2>
    %82 = buffer %81#0 {handshake.bb = 2 : ui32, handshake.name = "buffer23", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_NONE", NUM_SLOTS = 5 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 0}>}} : <i2>
    %83 = passer %82[%73#1] {handshake.bb = 2 : ui32, handshake.name = "passer8"} : <i2>, <i1>
    %84 = buffer %30#1 {handshake.bb = 2 : ui32, handshake.name = "buffer9", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_NONE", NUM_SLOTS = 5 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 0}>}} : <>
    %85 = passer %84[%73#0] {handshake.bb = 2 : ui32, handshake.name = "passer9"} : <>, <i1>
    %86 = extsi %83 {handshake.bb = 3 : ui32, handshake.name = "extsi12"} : <i2> to <i3>
    %87 = source {handshake.bb = 3 : ui32, handshake.name = "source3"} : <>
    %88 = constant %87 {handshake.bb = 3 : ui32, handshake.name = "constant15", value = 2 : i3} : <>, <i3>
    %89 = source {handshake.bb = 3 : ui32, handshake.name = "source4"} : <>
    %90 = constant %89 {handshake.bb = 3 : ui32, handshake.name = "constant16", value = 1 : i2} : <>, <i2>
    %91 = extsi %90 {handshake.bb = 3 : ui32, handshake.name = "extsi13"} : <i2> to <i3>
    %92 = addi %86, %91 {handshake.bb = 3 : ui32, handshake.name = "addi2", internal_delay = "0_000000"} : <i3>
    %93:2 = fork [2] %92 {handshake.bb = 3 : ui32, handshake.name = "fork9"} : <i3>
    %94 = trunci %93#0 {handshake.bb = 3 : ui32, handshake.name = "trunci4"} : <i3> to <i2>
    %95 = cmpi ult, %93#1, %88 {handshake.bb = 3 : ui32, handshake.name = "cmpi1", internal_delay = "0_000000"} : <i3>
    %96:2 = fork [2] %95 {handshake.bb = 3 : ui32, handshake.name = "fork10"} : <i1>
    %trueResult, %falseResult = cond_br %96#0, %94 {handshake.bb = 3 : ui32, handshake.name = "cond_br6"} : <i1>, <i2>
    sink %falseResult {handshake.name = "sink3"} : <i2>
    %trueResult_7, %falseResult_8 = cond_br %96#1, %85 {handshake.bb = 3 : ui32, handshake.name = "cond_br7"} : <i1>, <>
    %97:3 = fork [3] %falseResult_8 {handshake.bb = 4 : ui32, handshake.name = "fork11"} : <>
    end {handshake.bb = 4 : ui32, handshake.name = "end0"} %memEnd_2, %memEnd_0, %memEnd, %0#1 : <>, <>, <>, <>
  }
}

