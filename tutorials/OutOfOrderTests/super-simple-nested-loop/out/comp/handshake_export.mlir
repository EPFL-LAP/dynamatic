module {
  handshake.func @bicg(%arg0: memref<30xi32>, %arg1: !handshake.control<>, %arg2: !handshake.control<>, ...) -> (!handshake.channel<i32>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["q", "q_start", "start"], cfg.edges = "[0,1][2,2,3,cmpi0][1,2][3,1,4,cmpi1]", resNames = ["out0", "q_end", "end"]} {
    %0:5 = fork [5] %arg2 {handshake.bb = 0 : ui32, handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork0"} : <>
    %1 = buffer %0#4 {handshake.name = "buffer20", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %2 = buffer %1 {handshake.name = "buffer21", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %3 = buffer %2 {handshake.name = "buffer22", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %4 = buffer %3 {handshake.name = "buffer23", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %5 = buffer %4 {handshake.name = "buffer24", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %memEnd = mem_controller[%arg0 : memref<30xi32>] %arg1 (%220, %addressResult, %dataResult) %5 {connectedBlocks = [3 : i32], handshake.bufProps = #handshake<bufProps{"0": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00, "1": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00, "2": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00, "3": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00, "4": [0,0], [0,0], 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "mem_controller1"} :    (!handshake.channel<i32>, !handshake.channel<i5>, !handshake.channel<i32>) -> ()
    %6 = buffer %0#3 {handshake.bb = 1 : ui32, handshake.name = "buffer15", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %7 = buffer %6 {handshake.bb = 1 : ui32, handshake.name = "buffer16", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %8 = buffer %7 {handshake.bb = 1 : ui32, handshake.name = "buffer17", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %9 = buffer %8 {handshake.bb = 1 : ui32, handshake.name = "buffer18", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %10 = buffer %9 {handshake.bb = 1 : ui32, handshake.name = "buffer19", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %11 = constant %10 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "constant0", value = false} : <>, <i1>
    %12:2 = fork [2] %11 {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <i1>
    %13 = buffer %12#1 {handshake.bb = 1 : ui32, handshake.name = "buffer30", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i1>
    %14 = buffer %13 {handshake.bb = 1 : ui32, handshake.name = "buffer31", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i1>
    %15 = buffer %14 {handshake.bb = 1 : ui32, handshake.name = "buffer32", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i1>
    %16 = buffer %15 {handshake.bb = 1 : ui32, handshake.name = "buffer33", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i1>
    %17 = buffer %16 {handshake.bb = 1 : ui32, handshake.name = "buffer34", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i1>
    %18 = extsi %17 {handshake.bb = 1 : ui32, handshake.name = "extsi7"} : <i1> to <i6>
    %19 = buffer %12#0 {handshake.bb = 1 : ui32, handshake.name = "buffer25", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i1>
    %20 = buffer %19 {handshake.bb = 1 : ui32, handshake.name = "buffer26", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i1>
    %21 = buffer %20 {handshake.bb = 1 : ui32, handshake.name = "buffer27", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i1>
    %22 = buffer %21 {handshake.bb = 1 : ui32, handshake.name = "buffer28", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i1>
    %23 = buffer %22 {handshake.bb = 1 : ui32, handshake.name = "buffer29", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i1>
    %24 = buffer %254#0 {handshake.bb = 1 : ui32, handshake.name = "buffer196", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i1>
    %25 = buffer %24 {handshake.bb = 1 : ui32, handshake.name = "buffer197", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i1>
    %26 = buffer %25 {handshake.bb = 1 : ui32, handshake.name = "buffer198", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i1>
    %27 = buffer %26 {handshake.bb = 1 : ui32, handshake.name = "buffer199", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i1>
    %28 = buffer %27 {handshake.bb = 1 : ui32, handshake.name = "buffer200", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i1>
    %29 = merge %23, %28 {ftd.imerge, handshake.bb = 1 : ui32, handshake.name = "merge0"} : <i1>
    %30 = buffer %29 {handshake.bb = 1 : ui32, handshake.name = "buffer35", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i1>
    %31:2 = fork [2] %30 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"0": [1,inf], [0,inf], 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.name = "fork2"} : <i1>
    %32 = buffer %0#2 {handshake.bb = 1 : ui32, handshake.name = "buffer10", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %33 = buffer %32 {handshake.bb = 1 : ui32, handshake.name = "buffer11", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %34 = buffer %33 {handshake.bb = 1 : ui32, handshake.name = "buffer12", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %35 = buffer %34 {handshake.bb = 1 : ui32, handshake.name = "buffer13", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %36 = buffer %35 {handshake.bb = 1 : ui32, handshake.name = "buffer14", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %37 = buffer %31#1 {handshake.bb = 1 : ui32, handshake.name = "buffer41", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i1>
    %38 = buffer %37 {handshake.bb = 1 : ui32, handshake.name = "buffer42", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i1>
    %39 = buffer %38 {handshake.bb = 1 : ui32, handshake.name = "buffer43", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i1>
    %40 = buffer %39 {handshake.bb = 1 : ui32, handshake.name = "buffer44", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i1>
    %41 = buffer %40 {handshake.bb = 1 : ui32, handshake.name = "buffer45", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i1>
    %42 = mux %41 [%36, %trueResult_8] {ftd.regen, handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<>, <>] to <>
    %43 = buffer %42 {handshake.bb = 1 : ui32, handshake.name = "buffer46", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <>
    %44 = buffer %43 {handshake.bb = 1 : ui32, handshake.name = "buffer47", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %45 = buffer %44 {handshake.bb = 1 : ui32, handshake.name = "buffer48", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %46 = buffer %45 {handshake.bb = 1 : ui32, handshake.name = "buffer49", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %47 = buffer %46 {handshake.bb = 1 : ui32, handshake.name = "buffer50", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %48 = buffer %47 {handshake.bb = 1 : ui32, handshake.name = "buffer51", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %49:5 = fork [5] %48 {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <>
    %50 = buffer %31#0 {handshake.bb = 1 : ui32, handshake.name = "buffer36", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i1>
    %51 = buffer %50 {handshake.bb = 1 : ui32, handshake.name = "buffer37", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i1>
    %52 = buffer %51 {handshake.bb = 1 : ui32, handshake.name = "buffer38", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i1>
    %53 = buffer %52 {handshake.bb = 1 : ui32, handshake.name = "buffer39", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i1>
    %54 = buffer %53 {handshake.bb = 1 : ui32, handshake.name = "buffer40", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i1>
    %55 = mux %54 [%18, %trueResult_6] {handshake.bb = 1 : ui32, handshake.name = "mux5"} : <i1>, [<i6>, <i6>] to <i6>
    %56 = buffer %55 {handshake.bb = 1 : ui32, handshake.name = "buffer77", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i6>
    %57 = buffer %56 {handshake.bb = 1 : ui32, handshake.name = "buffer78", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i6>
    %58 = buffer %57 {handshake.bb = 1 : ui32, handshake.name = "buffer79", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i6>
    %59 = buffer %58 {handshake.bb = 1 : ui32, handshake.name = "buffer80", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i6>
    %60 = buffer %59 {handshake.bb = 1 : ui32, handshake.name = "buffer81", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i6>
    %61 = buffer %60 {handshake.bb = 1 : ui32, handshake.name = "buffer82", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i6>
    %62:3 = fork [3] %61 {handshake.bb = 1 : ui32, handshake.name = "fork4"} : <i6>
    %63 = buffer %62#2 {handshake.bb = 1 : ui32, handshake.name = "buffer93", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i6>
    %64 = buffer %63 {handshake.bb = 1 : ui32, handshake.name = "buffer94", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i6>
    %65 = buffer %64 {handshake.bb = 1 : ui32, handshake.name = "buffer95", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i6>
    %66 = buffer %65 {handshake.bb = 1 : ui32, handshake.name = "buffer96", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i6>
    %67 = buffer %66 {handshake.bb = 1 : ui32, handshake.name = "buffer97", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i6>
    %68 = extsi %67 {handshake.bb = 1 : ui32, handshake.name = "extsi8"} : <i6> to <i7>
    %69 = buffer %62#1 {handshake.bb = 1 : ui32, handshake.name = "buffer88", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i6>
    %70 = buffer %69 {handshake.bb = 1 : ui32, handshake.name = "buffer89", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i6>
    %71 = buffer %70 {handshake.bb = 1 : ui32, handshake.name = "buffer90", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i6>
    %72 = buffer %71 {handshake.bb = 1 : ui32, handshake.name = "buffer91", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i6>
    %73 = buffer %72 {handshake.bb = 1 : ui32, handshake.name = "buffer92", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i6>
    %74 = extsi %73 {handshake.bb = 1 : ui32, handshake.name = "extsi9"} : <i6> to <i32>
    %75 = buffer %62#0 {handshake.bb = 1 : ui32, handshake.name = "buffer83", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i6>
    %76 = buffer %75 {handshake.bb = 1 : ui32, handshake.name = "buffer84", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i6>
    %77 = buffer %76 {handshake.bb = 1 : ui32, handshake.name = "buffer85", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i6>
    %78 = buffer %77 {handshake.bb = 1 : ui32, handshake.name = "buffer86", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i6>
    %79 = buffer %78 {handshake.bb = 1 : ui32, handshake.name = "buffer87", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i6>
    %80 = trunci %79 {handshake.bb = 1 : ui32, handshake.name = "trunci0"} : <i6> to <i5>
    %81 = buffer %49#4 {handshake.bb = 1 : ui32, handshake.name = "buffer72", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %82 = buffer %81 {handshake.bb = 1 : ui32, handshake.name = "buffer73", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %83 = buffer %82 {handshake.bb = 1 : ui32, handshake.name = "buffer74", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %84 = buffer %83 {handshake.bb = 1 : ui32, handshake.name = "buffer75", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %85 = buffer %84 {handshake.bb = 1 : ui32, handshake.name = "buffer76", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %86 = constant %85 {handshake.bb = 1 : ui32, handshake.name = "constant8", value = false} : <>, <i1>
    %87 = extsi %86 {handshake.bb = 1 : ui32, handshake.name = "extsi10"} : <i1> to <i6>
    %88 = buffer %188#0 {handshake.bb = 2 : ui32, handshake.name = "buffer171", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i1>
    %89 = buffer %88 {handshake.bb = 2 : ui32, handshake.name = "buffer172", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i1>
    %90 = buffer %89 {handshake.bb = 2 : ui32, handshake.name = "buffer173", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i1>
    %91 = buffer %90 {handshake.bb = 2 : ui32, handshake.name = "buffer174", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i1>
    %92 = buffer %91 {handshake.bb = 2 : ui32, handshake.name = "buffer175", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i1>
    %trueResult, %falseResult = cond_br %92, %173 {handshake.bb = 2 : ui32, handshake.name = "cond_br17"} : <i1>, <i32>
    %93:2 = fork [2] %falseResult {handshake.bb = 2 : ui32, handshake.name = "fork5"} : <i32>
    %94 = buffer %188#1 {handshake.bb = 2 : ui32, handshake.name = "buffer176", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i1>
    %95 = buffer %94 {handshake.bb = 2 : ui32, handshake.name = "buffer177", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i1>
    %96 = buffer %95 {handshake.bb = 2 : ui32, handshake.name = "buffer178", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i1>
    %97 = buffer %96 {handshake.bb = 2 : ui32, handshake.name = "buffer179", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i1>
    %98 = buffer %97 {handshake.bb = 2 : ui32, handshake.name = "buffer180", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i1>
    %trueResult_0, %falseResult_1 = cond_br %98, %181 {handshake.bb = 2 : ui32, handshake.name = "cond_br18"} : <i1>, <i6>
    sink %falseResult_1 {handshake.name = "sink0"} : <i6>
    %99 = buffer %121#2 {handshake.bb = 2 : ui32, handshake.name = "buffer134", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %100 = buffer %99 {handshake.bb = 2 : ui32, handshake.name = "buffer135", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %101 = buffer %100 {handshake.bb = 2 : ui32, handshake.name = "buffer136", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %102 = buffer %101 {handshake.bb = 2 : ui32, handshake.name = "buffer137", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %103 = buffer %102 {handshake.bb = 2 : ui32, handshake.name = "buffer138", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %104 = buffer %188#2 {handshake.bb = 2 : ui32, handshake.name = "buffer181", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i1>
    %105 = buffer %104 {handshake.bb = 2 : ui32, handshake.name = "buffer182", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i1>
    %106 = buffer %105 {handshake.bb = 2 : ui32, handshake.name = "buffer183", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i1>
    %107 = buffer %106 {handshake.bb = 2 : ui32, handshake.name = "buffer184", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i1>
    %108 = buffer %107 {handshake.bb = 2 : ui32, handshake.name = "buffer185", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i1>
    %trueResult_2, %falseResult_3 = cond_br %108, %103 {handshake.bb = 2 : ui32, handshake.name = "cond_br19"} : <i1>, <>
    sink %falseResult_3 {handshake.name = "sink1"} : <>
    %109 = buffer %49#3 {handshake.bb = 2 : ui32, handshake.name = "buffer67", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %110 = buffer %109 {handshake.bb = 2 : ui32, handshake.name = "buffer68", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %111 = buffer %110 {handshake.bb = 2 : ui32, handshake.name = "buffer69", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %112 = buffer %111 {handshake.bb = 2 : ui32, handshake.name = "buffer70", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %113 = buffer %112 {handshake.bb = 2 : ui32, handshake.name = "buffer71", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %result, %index = control_merge [%113, %trueResult_2]  {handshake.bb = 2 : ui32, handshake.name = "control_merge0"} : [<>, <>] to <>, <i1>
    %114:2 = fork [2] %index {handshake.bb = 2 : ui32, handshake.name = "fork6"} : <i1>
    %115 = buffer %result {handshake.bb = 2 : ui32, handshake.name = "buffer108", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <>
    %116 = buffer %115 {handshake.bb = 2 : ui32, handshake.name = "buffer109", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %117 = buffer %116 {handshake.bb = 2 : ui32, handshake.name = "buffer110", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %118 = buffer %117 {handshake.bb = 2 : ui32, handshake.name = "buffer111", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %119 = buffer %118 {handshake.bb = 2 : ui32, handshake.name = "buffer112", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %120 = buffer %119 {handshake.bb = 2 : ui32, handshake.name = "buffer113", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %121:3 = fork [3] %120 {handshake.bb = 2 : ui32, handshake.name = "fork7"} : <>
    %122 = buffer %114#1 {handshake.bb = 2 : ui32, handshake.name = "buffer119", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i1>
    %123 = buffer %122 {handshake.bb = 2 : ui32, handshake.name = "buffer120", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i1>
    %124 = buffer %123 {handshake.bb = 2 : ui32, handshake.name = "buffer121", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i1>
    %125 = buffer %124 {handshake.bb = 2 : ui32, handshake.name = "buffer122", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i1>
    %126 = buffer %125 {handshake.bb = 2 : ui32, handshake.name = "buffer123", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i1>
    %127 = mux %126 [%74, %trueResult] {ftd.phi, handshake.bb = 2 : ui32, handshake.name = "mux8"} : <i1>, [<i32>, <i32>] to <i32>
    %128 = buffer %114#0 {handshake.bb = 2 : ui32, handshake.name = "buffer114", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i1>
    %129 = buffer %128 {handshake.bb = 2 : ui32, handshake.name = "buffer115", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i1>
    %130 = buffer %129 {handshake.bb = 2 : ui32, handshake.name = "buffer116", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i1>
    %131 = buffer %130 {handshake.bb = 2 : ui32, handshake.name = "buffer117", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i1>
    %132 = buffer %131 {handshake.bb = 2 : ui32, handshake.name = "buffer118", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i1>
    %133 = mux %132 [%87, %trueResult_0] {handshake.bb = 2 : ui32, handshake.name = "mux9"} : <i1>, [<i6>, <i6>] to <i6>
    %134 = buffer %133 {handshake.bb = 2 : ui32, handshake.name = "buffer145", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i6>
    %135 = buffer %134 {handshake.bb = 2 : ui32, handshake.name = "buffer146", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i6>
    %136 = buffer %135 {handshake.bb = 2 : ui32, handshake.name = "buffer147", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i6>
    %137 = buffer %136 {handshake.bb = 2 : ui32, handshake.name = "buffer148", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i6>
    %138 = buffer %137 {handshake.bb = 2 : ui32, handshake.name = "buffer149", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i6>
    %139 = buffer %138 {handshake.bb = 2 : ui32, handshake.name = "buffer150", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i6>
    %140:2 = fork [2] %139 {handshake.bb = 2 : ui32, handshake.name = "fork8"} : <i6>
    %141 = buffer %140#1 {handshake.bb = 2 : ui32, handshake.name = "buffer156", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i6>
    %142 = buffer %141 {handshake.bb = 2 : ui32, handshake.name = "buffer157", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i6>
    %143 = buffer %142 {handshake.bb = 2 : ui32, handshake.name = "buffer158", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i6>
    %144 = buffer %143 {handshake.bb = 2 : ui32, handshake.name = "buffer159", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i6>
    %145 = buffer %144 {handshake.bb = 2 : ui32, handshake.name = "buffer160", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i6>
    %146 = extsi %145 {handshake.bb = 2 : ui32, handshake.name = "extsi11"} : <i6> to <i7>
    %147 = buffer %140#0 {handshake.bb = 2 : ui32, handshake.name = "buffer151", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i6>
    %148 = buffer %147 {handshake.bb = 2 : ui32, handshake.name = "buffer152", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i6>
    %149 = buffer %148 {handshake.bb = 2 : ui32, handshake.name = "buffer153", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i6>
    %150 = buffer %149 {handshake.bb = 2 : ui32, handshake.name = "buffer154", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i6>
    %151 = buffer %150 {handshake.bb = 2 : ui32, handshake.name = "buffer155", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i6>
    %152 = extsi %151 {handshake.bb = 2 : ui32, handshake.name = "extsi12"} : <i6> to <i32>
    %153 = buffer %121#1 {handshake.bb = 2 : ui32, handshake.name = "buffer129", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %154 = buffer %153 {handshake.bb = 2 : ui32, handshake.name = "buffer130", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %155 = buffer %154 {handshake.bb = 2 : ui32, handshake.name = "buffer131", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %156 = buffer %155 {handshake.bb = 2 : ui32, handshake.name = "buffer132", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %157 = buffer %156 {handshake.bb = 2 : ui32, handshake.name = "buffer133", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %158 = constant %157 {handshake.bb = 2 : ui32, handshake.name = "constant9", value = 30 : i6} : <>, <i6>
    %159 = extsi %158 {handshake.bb = 2 : ui32, handshake.name = "extsi13"} : <i6> to <i7>
    %160 = buffer %121#0 {handshake.bb = 2 : ui32, handshake.name = "buffer124", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %161 = buffer %160 {handshake.bb = 2 : ui32, handshake.name = "buffer125", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %162 = buffer %161 {handshake.bb = 2 : ui32, handshake.name = "buffer126", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %163 = buffer %162 {handshake.bb = 2 : ui32, handshake.name = "buffer127", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %164 = buffer %163 {handshake.bb = 2 : ui32, handshake.name = "buffer128", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %165 = constant %164 {handshake.bb = 2 : ui32, handshake.name = "constant10", value = 1 : i2} : <>, <i2>
    %166 = extsi %165 {handshake.bb = 2 : ui32, handshake.name = "extsi14"} : <i2> to <i7>
    %167 = buffer %127 {handshake.bb = 2 : ui32, handshake.name = "buffer139", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i32>
    %168 = buffer %167 {handshake.bb = 2 : ui32, handshake.name = "buffer140", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i32>
    %169 = buffer %168 {handshake.bb = 2 : ui32, handshake.name = "buffer141", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i32>
    %170 = buffer %169 {handshake.bb = 2 : ui32, handshake.name = "buffer142", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i32>
    %171 = buffer %170 {handshake.bb = 2 : ui32, handshake.name = "buffer143", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i32>
    %172 = buffer %171 {handshake.bb = 2 : ui32, handshake.name = "buffer144", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i32>
    %173 = muli %172, %152 {handshake.bb = 2 : ui32, handshake.name = "muli0"} : <i32>
    %174 = addi %146, %166 {handshake.bb = 2 : ui32, handshake.name = "addi1"} : <i7>
    %175:2 = fork [2] %174 {handshake.bb = 2 : ui32, handshake.name = "fork9"} : <i7>
    %176 = buffer %175#1 {handshake.bb = 2 : ui32, handshake.name = "buffer166", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i7>
    %177 = buffer %176 {handshake.bb = 2 : ui32, handshake.name = "buffer167", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i7>
    %178 = buffer %177 {handshake.bb = 2 : ui32, handshake.name = "buffer168", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i7>
    %179 = buffer %178 {handshake.bb = 2 : ui32, handshake.name = "buffer169", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i7>
    %180 = buffer %179 {handshake.bb = 2 : ui32, handshake.name = "buffer170", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i7>
    %181 = trunci %180 {handshake.bb = 2 : ui32, handshake.name = "trunci1"} : <i7> to <i6>
    %182 = buffer %175#0 {handshake.bb = 2 : ui32, handshake.name = "buffer161", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i7>
    %183 = buffer %182 {handshake.bb = 2 : ui32, handshake.name = "buffer162", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i7>
    %184 = buffer %183 {handshake.bb = 2 : ui32, handshake.name = "buffer163", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i7>
    %185 = buffer %184 {handshake.bb = 2 : ui32, handshake.name = "buffer164", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i7>
    %186 = buffer %185 {handshake.bb = 2 : ui32, handshake.name = "buffer165", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i7>
    %187 = cmpi ult, %186, %159 {handshake.bb = 2 : ui32, handshake.name = "cmpi0"} : <i7>
    %188:3 = fork [3] %187 {handshake.bb = 2 : ui32, handshake.name = "fork10"} : <i1>
    %189 = buffer %93#1 {handshake.bb = 3 : ui32, handshake.name = "buffer103", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i32>
    %190 = buffer %189 {handshake.bb = 3 : ui32, handshake.name = "buffer104", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i32>
    %191 = buffer %190 {handshake.bb = 3 : ui32, handshake.name = "buffer105", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i32>
    %192 = buffer %191 {handshake.bb = 3 : ui32, handshake.name = "buffer106", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i32>
    %193 = buffer %192 {handshake.bb = 3 : ui32, handshake.name = "buffer107", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i32>
    %194 = buffer %254#1 {handshake.bb = 3 : ui32, handshake.name = "buffer201", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i1>
    %195 = buffer %194 {handshake.bb = 3 : ui32, handshake.name = "buffer202", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i1>
    %196 = buffer %195 {handshake.bb = 3 : ui32, handshake.name = "buffer203", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i1>
    %197 = buffer %196 {handshake.bb = 3 : ui32, handshake.name = "buffer204", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i1>
    %198 = buffer %197 {handshake.bb = 3 : ui32, handshake.name = "buffer205", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i1>
    %trueResult_4, %falseResult_5 = cond_br %198, %193 {handshake.bb = 3 : ui32, handshake.name = "cond_br20"} : <i1>, <i32>
    sink %trueResult_4 {handshake.name = "sink2"} : <i32>
    %199 = buffer %254#2 {handshake.bb = 3 : ui32, handshake.name = "buffer206", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i1>
    %200 = buffer %199 {handshake.bb = 3 : ui32, handshake.name = "buffer207", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i1>
    %201 = buffer %200 {handshake.bb = 3 : ui32, handshake.name = "buffer208", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i1>
    %202 = buffer %201 {handshake.bb = 3 : ui32, handshake.name = "buffer209", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i1>
    %203 = buffer %202 {handshake.bb = 3 : ui32, handshake.name = "buffer210", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i1>
    %trueResult_6, %falseResult_7 = cond_br %203, %247 {handshake.bb = 3 : ui32, handshake.name = "cond_br21"} : <i1>, <i6>
    sink %falseResult_7 {handshake.name = "sink3"} : <i6>
    %204 = buffer %49#2 {handshake.bb = 3 : ui32, handshake.name = "buffer62", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %205 = buffer %204 {handshake.bb = 3 : ui32, handshake.name = "buffer63", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %206 = buffer %205 {handshake.bb = 3 : ui32, handshake.name = "buffer64", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %207 = buffer %206 {handshake.bb = 3 : ui32, handshake.name = "buffer65", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %208 = buffer %207 {handshake.bb = 3 : ui32, handshake.name = "buffer66", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %209 = buffer %254#3 {handshake.bb = 3 : ui32, handshake.name = "buffer211", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i1>
    %210 = buffer %209 {handshake.bb = 3 : ui32, handshake.name = "buffer212", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i1>
    %211 = buffer %210 {handshake.bb = 3 : ui32, handshake.name = "buffer213", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i1>
    %212 = buffer %211 {handshake.bb = 3 : ui32, handshake.name = "buffer214", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i1>
    %213 = buffer %212 {handshake.bb = 3 : ui32, handshake.name = "buffer215", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i1>
    %trueResult_8, %falseResult_9 = cond_br %213, %208 {handshake.bb = 3 : ui32, handshake.name = "cond_br22"} : <i1>, <>
    sink %falseResult_9 {handshake.name = "sink4"} : <>
    %214 = buffer %0#1 {handshake.bb = 3 : ui32, handshake.name = "buffer5", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %215 = buffer %214 {handshake.bb = 3 : ui32, handshake.name = "buffer6", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %216 = buffer %215 {handshake.bb = 3 : ui32, handshake.name = "buffer7", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %217 = buffer %216 {handshake.bb = 3 : ui32, handshake.name = "buffer8", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %218 = buffer %217 {handshake.bb = 3 : ui32, handshake.name = "buffer9", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %219 = constant %218 {handshake.bb = 3 : ui32, handshake.name = "constant11", value = 1 : i2} : <>, <i2>
    %220 = extsi %219 {handshake.bb = 3 : ui32, handshake.name = "extsi4"} : <i2> to <i32>
    %221 = buffer %49#1 {handshake.bb = 3 : ui32, handshake.name = "buffer57", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %222 = buffer %221 {handshake.bb = 3 : ui32, handshake.name = "buffer58", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %223 = buffer %222 {handshake.bb = 3 : ui32, handshake.name = "buffer59", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %224 = buffer %223 {handshake.bb = 3 : ui32, handshake.name = "buffer60", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %225 = buffer %224 {handshake.bb = 3 : ui32, handshake.name = "buffer61", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %226 = constant %225 {handshake.bb = 3 : ui32, handshake.name = "constant13", value = 30 : i6} : <>, <i6>
    %227 = extsi %226 {handshake.bb = 3 : ui32, handshake.name = "extsi15"} : <i6> to <i7>
    %228 = buffer %49#0 {handshake.bb = 3 : ui32, handshake.name = "buffer52", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %229 = buffer %228 {handshake.bb = 3 : ui32, handshake.name = "buffer53", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %230 = buffer %229 {handshake.bb = 3 : ui32, handshake.name = "buffer54", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %231 = buffer %230 {handshake.bb = 3 : ui32, handshake.name = "buffer55", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %232 = buffer %231 {handshake.bb = 3 : ui32, handshake.name = "buffer56", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %233 = constant %232 {handshake.bb = 3 : ui32, handshake.name = "constant14", value = 1 : i2} : <>, <i2>
    %234 = extsi %233 {handshake.bb = 3 : ui32, handshake.name = "extsi16"} : <i2> to <i7>
    %235 = buffer %93#0 {handshake.bb = 3 : ui32, handshake.name = "buffer98", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i32>
    %236 = buffer %235 {handshake.bb = 3 : ui32, handshake.name = "buffer99", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i32>
    %237 = buffer %236 {handshake.bb = 3 : ui32, handshake.name = "buffer100", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i32>
    %238 = buffer %237 {handshake.bb = 3 : ui32, handshake.name = "buffer101", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i32>
    %239 = buffer %238 {handshake.bb = 3 : ui32, handshake.name = "buffer102", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i32>
    %addressResult, %dataResult = store[%80] %239 {handshake.bb = 3 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store0"} : <i5>, <i32>, <i5>, <i32>
    %240 = addi %68, %234 {handshake.bb = 3 : ui32, handshake.name = "addi2"} : <i7>
    %241:2 = fork [2] %240 {handshake.bb = 3 : ui32, handshake.name = "fork11"} : <i7>
    %242 = buffer %241#1 {handshake.bb = 3 : ui32, handshake.name = "buffer191", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i7>
    %243 = buffer %242 {handshake.bb = 3 : ui32, handshake.name = "buffer192", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i7>
    %244 = buffer %243 {handshake.bb = 3 : ui32, handshake.name = "buffer193", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i7>
    %245 = buffer %244 {handshake.bb = 3 : ui32, handshake.name = "buffer194", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i7>
    %246 = buffer %245 {handshake.bb = 3 : ui32, handshake.name = "buffer195", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i7>
    %247 = trunci %246 {handshake.bb = 3 : ui32, handshake.name = "trunci2"} : <i7> to <i6>
    %248 = buffer %241#0 {handshake.bb = 3 : ui32, handshake.name = "buffer186", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i7>
    %249 = buffer %248 {handshake.bb = 3 : ui32, handshake.name = "buffer187", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i7>
    %250 = buffer %249 {handshake.bb = 3 : ui32, handshake.name = "buffer188", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i7>
    %251 = buffer %250 {handshake.bb = 3 : ui32, handshake.name = "buffer189", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i7>
    %252 = buffer %251 {handshake.bb = 3 : ui32, handshake.name = "buffer190", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i7>
    %253 = cmpi ult, %252, %227 {handshake.bb = 3 : ui32, handshake.name = "cmpi1"} : <i7>
    %254:4 = fork [4] %253 {handshake.bb = 3 : ui32, handshake.name = "fork12"} : <i1>
    %255 = buffer %0#0 {handshake.bb = 4 : ui32, handshake.name = "buffer0", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %256 = buffer %255 {handshake.bb = 4 : ui32, handshake.name = "buffer1", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %257 = buffer %256 {handshake.bb = 4 : ui32, handshake.name = "buffer2", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %258 = buffer %257 {handshake.bb = 4 : ui32, handshake.name = "buffer3", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    %259 = buffer %258 {handshake.bb = 4 : ui32, handshake.name = "buffer4", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <>
    end {handshake.bb = 4 : ui32, handshake.name = "end0"} %falseResult_5, %memEnd, %259 : <i32>, <>, <>
  }
}

