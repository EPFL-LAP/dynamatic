module {
  handshake.func @nested_loop(%arg0: memref<1000xi32>, %arg1: memref<1000xi32>, %arg2: memref<1000xi32>, %arg3: !handshake.control<>, %arg4: !handshake.control<>, %arg5: !handshake.control<>, %arg6: !handshake.control<>, ...) -> (!handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["a", "b", "c", "a_start", "b_start", "c_start", "start"], handshake.cfdfcThroughput = #handshake<cfdfcThroughput {"0" = 1.000000e+00 : f64, "1" = 0.1111111111111111 : f64}>, handshake.cfdfcToBBList = #handshake<cfdfcToBBList {"0" = [2 : ui32], "1" = [2 : ui32, 3 : ui32, 1 : ui32]}>, resNames = ["a_end", "b_end", "c_end", "end"]} {
    %0:3 = fork [3] %arg6 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork0"} : <>
    %memEnd = mem_controller[%arg2 : memref<1000xi32>] %arg5 (%78, %addressResult_5, %dataResult_6) %93#2 {connectedBlocks = [2 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller3"} :    (!handshake.channel<i32>, !handshake.channel<i10>, !handshake.channel<i32>) -> ()
    %outputs, %memEnd_0 = mem_controller[%arg1 : memref<1000xi32>] %arg4 (%addressResult_3) %93#1 {connectedBlocks = [2 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller4"} :    (!handshake.channel<i10>) -> !handshake.channel<i32>
    %outputs_1, %memEnd_2 = mem_controller[%arg0 : memref<1000xi32>] %arg3 (%addressResult) %93#0 {connectedBlocks = [2 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller5"} :    (!handshake.channel<i10>) -> !handshake.channel<i32>
    %1 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant3", value = false} : <>, <i1>
    %2 = extsi %1 {handshake.bb = 0 : ui32, handshake.name = "extsi8"} : <i1> to <i2>
    %3 = mux %index [%2, %trueResult] {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mux0"} : <i1>, [<i2>, <i2>] to <i2>
    %4 = buffer %3 {handshake.bb = 1 : ui32, handshake.name = "buffer1", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i2>
    %5:2 = fork [2] %4 {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <i2>
    %6 = extsi %5#0 {handshake.bb = 1 : ui32, handshake.name = "extsi9"} : <i2> to <i12>
    %result, %index = control_merge [%0#2, %trueResult_7]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %7 = buffer %result {handshake.bb = 1 : ui32, handshake.name = "buffer3", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_NONE", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 0}>}} : <>
    %8:2 = fork [2] %7 {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <>
    %9 = source {handshake.bb = 1 : ui32, handshake.name = "source0"} : <>
    %10 = constant %9 {handshake.bb = 1 : ui32, handshake.name = "constant4", value = 400 : i10} : <>, <i10>
    %11 = extsi %10 {handshake.bb = 1 : ui32, handshake.name = "extsi10"} : <i10> to <i12>
    %12 = constant %8#0 {handshake.bb = 1 : ui32, handshake.name = "constant5", value = false} : <>, <i1>
    %13 = muli %6, %11 {handshake.bb = 1 : ui32, handshake.name = "muli0", internal_delay = "1_000000"} : <i12>
    %14 = extsi %12 {handshake.bb = 1 : ui32, handshake.name = "extsi11"} : <i1> to <i32>
    %15 = mux %47#3 [%14, %66] {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mux1"} : <i1>, [<i32>, <i32>] to <i32>
    %16 = buffer %15 {handshake.bb = 2 : ui32, handshake.name = "buffer4", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i32>
    %17:4 = fork [4] %16 {handshake.bb = 2 : ui32, handshake.name = "fork3"} : <i32>
    %18 = trunci %17#0 {handshake.bb = 2 : ui32, handshake.name = "trunci0"} : <i32> to <i10>
    %19 = trunci %17#1 {handshake.bb = 2 : ui32, handshake.name = "trunci1"} : <i32> to <i10>
    %20 = trunci %17#2 {handshake.bb = 2 : ui32, handshake.name = "trunci2"} : <i32> to <i10>
    %21 = buffer %5#1 {handshake.bb = 2 : ui32, handshake.name = "buffer2", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_NONE", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 0}>}} : <i2>
    %22 = mux %47#2 [%21, %70] {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mux2"} : <i1>, [<i2>, <i2>] to <i2>
    %23 = mux %47#1 [%13, %76] {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mux3"} : <i1>, [<i12>, <i12>] to <i12>
    %24 = buffer %23 {handshake.bb = 2 : ui32, handshake.name = "buffer9", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i12>
    %25 = buffer %24 {handshake.bb = 2 : ui32, handshake.name = "buffer10", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i12>
    %26:2 = fork [2] %25 {handshake.bb = 2 : ui32, handshake.name = "fork19"} : <i12>
    %27 = trunci %26#0 {handshake.bb = 2 : ui32, handshake.name = "trunci3"} : <i12> to <i10>
    %28 = mux %47#0 [%8#1, %79] {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mux4"} : <i1>, [<>, <>] to <>
    %29 = buffer %28 {handshake.bb = 2 : ui32, handshake.name = "buffer11", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <>
    %30 = buffer %29 {handshake.bb = 2 : ui32, handshake.name = "buffer12", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %31:3 = fork [3] %30 {handshake.bb = 2 : ui32, handshake.name = "fork20"} : <>
    %32 = constant %31#0 {handshake.bb = 2 : ui32, handshake.name = "constant6", value = 1 : i2} : <>, <i2>
    %33 = extsi %32 {handshake.bb = 2 : ui32, handshake.name = "extsi3"} : <i2> to <i32>
    %34 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %35 = constant %34 {handshake.bb = 2 : ui32, handshake.name = "constant13", value = 1 : i2} : <>, <i2>
    %36 = extsi %35 {handshake.bb = 2 : ui32, handshake.name = "extsi4"} : <i2> to <i32>
    %37 = source {handshake.bb = 2 : ui32, handshake.name = "source2"} : <>
    %38 = constant %37 {handshake.bb = 2 : ui32, handshake.name = "constant14", value = 1000 : i11} : <>, <i11>
    %39 = extsi %38 {handshake.bb = 2 : ui32, handshake.name = "extsi5"} : <i11> to <i32>
    %addressResult, %dataResult = load[%20] %outputs_1 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load2"} : <i10>, <i32>, <i10>, <i32>
    %addressResult_3, %dataResult_4 = load[%19] %outputs {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load3"} : <i10>, <i32>, <i10>, <i32>
    %40 = muli %dataResult, %dataResult_4 {handshake.bb = 2 : ui32, handshake.name = "muli1", internal_delay = "1_000000"} : <i32>
    %41:2 = fork [2] %40 {handshake.bb = 2 : ui32, handshake.name = "fork7"} : <i32>
    %42 = buffer %18 {handshake.bb = 2 : ui32, handshake.name = "buffer6", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_NONE", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 0}>}} : <i10>
    %43 = addi %42, %27 {handshake.bb = 2 : ui32, handshake.name = "addi0", internal_delay = "0_000000"} : <i10>
    %addressResult_5, %dataResult_6 = store[%75] %65 {handshake.bb = 2 : ui32, handshake.deps = #handshake<deps[<"store1" (0)>]>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store1"} : <i10>, <i32>, <i10>, <i32>
    %44 = cmpi slt, %41#1, %39 {handshake.bb = 2 : ui32, handshake.name = "cmpi0", internal_delay = "0_000000"} : <i32>
    %45 = buffer %54#0 {handshake.bb = 2 : ui32, handshake.name = "buffer18", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_NONE", NUM_SLOTS = 2 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 0}>}} : <i1>
    %46 = init %45 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 2, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "init0", initToken = 0 : ui1} : <i1>
    %47:4 = fork [4] %46 {handshake.bb = 2 : ui32, handshake.name = "fork21"} : <i1>
    %48 = source {handshake.bb = 2 : ui32, handshake.name = "source5", specv2_ignore_buffer = true} : <>
    %49 = constant %48 {handshake.bb = 2 : ui32, handshake.name = "constant0", specv2_ignore_buffer = true, value = true} : <>, <i1>
    %50 = buffer %60 {handshake.bb = 2 : ui32, handshake.name = "buffer0", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}, specv2_buffer_as_sink = true} : <i1>
    %51 = merge %50, %49 {handshake.bb = 2 : ui32, handshake.name = "merge1", specv2_buffer_as_source = true} : <i1>
    %52 = buffer %51 {handshake.bb = 2 : ui32, handshake.name = "buffer16", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_NONE", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 0}>}} : <i1>
    %53 = buffer %52 {handshake.bb = 2 : ui32, handshake.name = "buffer17", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i1>
    %54:6 = fork [6] %53 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,inf], [0,inf], 1, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork22"} : <i1>
    %55 = andi %64, %59#0 {handshake.bb = 2 : ui32, handshake.name = "andi0", internal_delay = "0_000000"} : <i1>
    %56:3 = fork [3] %55 {handshake.bb = 2 : ui32, handshake.name = "fork23"} : <i1>
    %57 = buffer %54#5 {handshake.bb = 2 : ui32, handshake.name = "buffer19", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_NONE", NUM_SLOTS = 4 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 0}>}} : <i1>
    %58 = spec_v2_resolver %61#1, %57 {handshake.bb = 2 : ui32, handshake.name = "spec_v2_resolver0"} : <i1>
    %59:4 = fork [4] %58 {handshake.bb = 2 : ui32, handshake.name = "fork24"} : <i1>
    %60 = passer %61#2[%56#0] {handshake.bb = 2 : ui32, handshake.name = "passer0"} : <i1>, <i1>
    %61:3 = fork [3] %44 {handshake.bb = 2 : ui32, handshake.name = "fork25"} : <i1>
    %62 = buffer %17#3 {handshake.bb = 2 : ui32, handshake.name = "buffer5", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i32>
    %63 = addi %62, %36 {handshake.bb = 2 : ui32, handshake.name = "addi1", internal_delay = "0_000000"} : <i32>
    %64 = not %61#0 {handshake.bb = 2 : ui32, handshake.name = "not0"} : <i1>
    %65 = passer %41#0[%59#2] {handshake.bb = 2 : ui32, handshake.name = "passer1"} : <i32>, <i1>
    %66 = passer %63[%54#3] {handshake.bb = 2 : ui32, handshake.name = "passer2"} : <i32>, <i1>
    %67 = buffer %22 {handshake.bb = 2 : ui32, handshake.name = "buffer7", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i2>
    %68 = buffer %67 {handshake.bb = 2 : ui32, handshake.name = "buffer8", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i2>
    %69:2 = fork [2] %68 {handshake.bb = 2 : ui32, handshake.name = "fork16"} : <i2>
    %70 = passer %69#1[%54#2] {handshake.bb = 2 : ui32, handshake.name = "passer3"} : <i2>, <i1>
    %71 = buffer %69#0 {handshake.bb = 2 : ui32, handshake.name = "buffer20", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i2>
    %72 = buffer %71 {handshake.bb = 2 : ui32, handshake.name = "buffer21", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_NONE", NUM_SLOTS = 3 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 0}>}} : <i2>
    %73 = passer %72[%56#2] {handshake.bb = 2 : ui32, handshake.name = "passer4"} : <i2>, <i1>
    %74 = buffer %43 {handshake.bb = 2 : ui32, handshake.name = "buffer15", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_NONE", NUM_SLOTS = 4 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 0}>}} : <i10>
    %75 = passer %74[%59#3] {handshake.bb = 2 : ui32, handshake.name = "passer5"} : <i10>, <i1>
    %76 = passer %26#1[%54#4] {handshake.bb = 2 : ui32, handshake.name = "passer6"} : <i12>, <i1>
    %77 = buffer %33 {handshake.bb = 2 : ui32, handshake.name = "buffer14", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_NONE", NUM_SLOTS = 4 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 0}>}} : <i32>
    %78 = passer %77[%59#1] {handshake.bb = 2 : ui32, handshake.name = "passer7"} : <i32>, <i1>
    %79 = passer %31#2[%54#1] {handshake.bb = 2 : ui32, handshake.name = "passer8"} : <>, <i1>
    %80 = buffer %31#1 {handshake.bb = 2 : ui32, handshake.name = "buffer13", hw.parameters = {BUFFER_TYPE = "FIFO_BREAK_NONE", NUM_SLOTS = 4 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 0}>}} : <>
    %81 = passer %80[%56#1] {handshake.bb = 2 : ui32, handshake.name = "passer9"} : <>, <i1>
    %82 = extsi %73 {handshake.bb = 3 : ui32, handshake.name = "extsi12"} : <i2> to <i3>
    %83 = source {handshake.bb = 3 : ui32, handshake.name = "source3"} : <>
    %84 = constant %83 {handshake.bb = 3 : ui32, handshake.name = "constant15", value = 2 : i3} : <>, <i3>
    %85 = source {handshake.bb = 3 : ui32, handshake.name = "source4"} : <>
    %86 = constant %85 {handshake.bb = 3 : ui32, handshake.name = "constant16", value = 1 : i2} : <>, <i2>
    %87 = extsi %86 {handshake.bb = 3 : ui32, handshake.name = "extsi13"} : <i2> to <i3>
    %88 = addi %82, %87 {handshake.bb = 3 : ui32, handshake.name = "addi2", internal_delay = "0_000000"} : <i3>
    %89:2 = fork [2] %88 {handshake.bb = 3 : ui32, handshake.name = "fork9"} : <i3>
    %90 = trunci %89#0 {handshake.bb = 3 : ui32, handshake.name = "trunci4"} : <i3> to <i2>
    %91 = cmpi ult, %89#1, %84 {handshake.bb = 3 : ui32, handshake.name = "cmpi1", internal_delay = "0_000000"} : <i3>
    %92:2 = fork [2] %91 {handshake.bb = 3 : ui32, handshake.name = "fork10"} : <i1>
    %trueResult, %falseResult = cond_br %92#0, %90 {handshake.bb = 3 : ui32, handshake.name = "cond_br6"} : <i1>, <i2>
    sink %falseResult {handshake.name = "sink3"} : <i2>
    %trueResult_7, %falseResult_8 = cond_br %92#1, %81 {handshake.bb = 3 : ui32, handshake.name = "cond_br7"} : <i1>, <>
    %93:3 = fork [3] %falseResult_8 {handshake.bb = 4 : ui32, handshake.name = "fork11"} : <>
    end {handshake.bb = 4 : ui32, handshake.name = "end0"} %memEnd_2, %memEnd_0, %memEnd, %0#1 : <>, <>, <>, <>
  }
}

