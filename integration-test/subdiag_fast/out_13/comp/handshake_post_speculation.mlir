module {
  handshake.func @subdiag_fast(%arg0: memref<1000xf32>, %arg1: memref<1000xf32>, %arg2: memref<1000xf32>, %arg3: !handshake.control<>, %arg4: !handshake.control<>, %arg5: !handshake.control<>, %arg6: !handshake.control<>, ...) -> (!handshake.channel<i32>, !handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["d1", "d2", "e", "d1_start", "d2_start", "e_start", "start"], resNames = ["out0", "d1_end", "d2_end", "e_end", "end"]} {
    %0:3 = fork [3] %arg6 {handshake.bb = 0 : ui32, handshake.name = "fork0"} : <>
    %outputs, %memEnd = mem_controller[%arg2 : memref<1000xf32>] %arg5 (%addressResult_6) %87#2 {connectedBlocks = [1 : i32], handshake.name = "mem_controller3"} :    (!handshake.channel<i10>) -> !handshake.channel<f32>
    %outputs_0, %memEnd_1 = mem_controller[%arg1 : memref<1000xf32>] %arg4 (%addressResult_4) %87#1 {connectedBlocks = [1 : i32], handshake.name = "mem_controller4"} :    (!handshake.channel<i10>) -> !handshake.channel<f32>
    %outputs_2, %memEnd_3 = mem_controller[%arg0 : memref<1000xf32>] %arg3 (%addressResult) %87#0 {connectedBlocks = [1 : i32], handshake.name = "mem_controller5"} :    (!handshake.channel<i10>) -> !handshake.channel<f32>
    %1 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant0", value = false} : <>, <i1>
    %2 = extsi %1 {handshake.bb = 0 : ui32, handshake.name = "extsi3"} : <i1> to <i11>
    %3 = mux %70#1 [%2, %80] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i11>, <i11>] to <i11>
    %4 = trunci %79#0 {handshake.bb = 1 : ui32, handshake.name = "trunci0"} : <i11> to <i10>
    %5 = trunci %79#1 {handshake.bb = 1 : ui32, handshake.name = "trunci1"} : <i11> to <i10>
    %6 = trunci %79#2 {handshake.bb = 1 : ui32, handshake.name = "trunci2"} : <i11> to <i10>
    %7 = mux %70#0 [%0#2, %82] {handshake.bb = 1 : ui32, handshake.name = "mux3"} : <i1>, [<>, <>] to <>
    %8 = source {handshake.bb = 1 : ui32, handshake.name = "source0"} : <>
    %9 = constant %8 {handshake.bb = 1 : ui32, handshake.name = "constant4", value = 1.000000e-03 : f32} : <>, <f32>
    %addressResult, %dataResult = load[%6] %outputs_2 {handshake.bb = 1 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i10>, <f32>, <i10>, <f32>
    %addressResult_4, %dataResult_5 = load[%5] %outputs_0 {handshake.bb = 1 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i10>, <f32>, <i10>, <f32>
    %10 = addf %dataResult, %dataResult_5 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "addf0"} : <f32>
    %addressResult_6, %dataResult_7 = load[%4] %outputs {handshake.bb = 1 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load2"} : <i10>, <f32>, <i10>, <f32>
    %11 = mulf %10, %9 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "mulf0"} : <f32>
    %12:3 = fork [3] %13 {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <i1>
    %13 = cmpf ugt, %dataResult_7, %11 {handshake.bb = 1 : ui32, handshake.name = "cmpf0"} : <f32>
    %14 = andi %77, %12#1 {handshake.bb = 1 : ui32, handshake.name = "andi0"} : <i1>
    %15 = not %12#0 {handshake.bb = 1 : ui32, handshake.name = "not0"} : <i1>
    %16 = passer %17[%74#1] {handshake.bb = 1 : ui32, handshake.name = "passer8"} : <i12>, <i1>
    %17 = extsi %79#3 {handshake.bb = 1 : ui32, handshake.name = "extsi4"} : <i11> to <i12>
    %18 = passer %83#2[%74#0] {handshake.bb = 1 : ui32, handshake.name = "passer3"} : <>, <i1>
    %19 = extsi %79#4 {handshake.bb = 1 : ui32, handshake.name = "extsi5"} : <i11> to <i12>
    %20 = source {handshake.bb = 1 : ui32, handshake.name = "source1"} : <>
    %21 = constant %20 {handshake.bb = 1 : ui32, handshake.name = "constant1", value = 1 : i2} : <>, <i2>
    %22 = extsi %21 {handshake.bb = 1 : ui32, handshake.name = "extsi6"} : <i2> to <i12>
    %23 = source {handshake.bb = 1 : ui32, handshake.name = "source2"} : <>
    %24 = constant %23 {handshake.bb = 1 : ui32, handshake.name = "constant2", value = 999 : i11} : <>, <i11>
    %25 = extsi %24 {handshake.bb = 1 : ui32, handshake.name = "extsi7"} : <i11> to <i12>
    %26:3 = fork [3] %27 {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <i12>
    %27 = addi %19, %22 {handshake.bb = 1 : ui32, handshake.name = "addi0"} : <i12>
    %28:2 = fork [2] %29 {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <i1>
    %29 = cmpi ult, %26#0, %25 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i12>
    %30 = passer %31[%72#2] {handshake.bb = 1 : ui32, handshake.name = "passer9"} : <i1>, <i1>
    %31 = andi %12#2, %28#0 {handshake.bb = 1 : ui32, handshake.name = "andi1"} : <i1>
    %32 = spec_v2_repeating_init %30 {handshake.bb = 1 : ui32, handshake.name = "spec_v2_repeating_init0", initToken = 1 : ui1} : <i1>
    %33:2 = fork [2] %32 {handshake.bb = 1 : ui32, handshake.name = "fork4"} : <i1>
    %34 = spec_v2_repeating_init %33#0 {handshake.bb = 1 : ui32, handshake.name = "spec_v2_repeating_init1", initToken = 1 : ui1} : <i1>
    %35:2 = fork [2] %34 {handshake.bb = 1 : ui32, handshake.name = "fork15"} : <i1>
    %36 = andi %33#1, %35#0 {handshake.bb = 1 : ui32, handshake.name = "andi2", specv2_tmp_and = true} : <i1>
    %37 = spec_v2_repeating_init %35#1 {handshake.bb = 1 : ui32, handshake.name = "spec_v2_repeating_init2", initToken = 1 : ui1} : <i1>
    %38:2 = fork [2] %37 {handshake.bb = 1 : ui32, handshake.name = "fork16"} : <i1>
    %39 = andi %36, %38#0 {handshake.bb = 1 : ui32, handshake.name = "andi3", specv2_tmp_and = true} : <i1>
    %40 = spec_v2_repeating_init %38#1 {handshake.bb = 1 : ui32, handshake.name = "spec_v2_repeating_init3", initToken = 1 : ui1} : <i1>
    %41:2 = fork [2] %40 {handshake.bb = 1 : ui32, handshake.name = "fork17"} : <i1>
    %42 = andi %39, %41#0 {handshake.bb = 1 : ui32, handshake.name = "andi4", specv2_tmp_and = true} : <i1>
    %43 = spec_v2_repeating_init %41#1 {handshake.bb = 1 : ui32, handshake.name = "spec_v2_repeating_init4", initToken = 1 : ui1} : <i1>
    %44:2 = fork [2] %43 {handshake.bb = 1 : ui32, handshake.name = "fork18"} : <i1>
    %45 = andi %42, %44#0 {handshake.bb = 1 : ui32, handshake.name = "andi5", specv2_tmp_and = true} : <i1>
    %46 = spec_v2_repeating_init %44#1 {handshake.bb = 1 : ui32, handshake.name = "spec_v2_repeating_init5", initToken = 1 : ui1} : <i1>
    %47:2 = fork [2] %46 {handshake.bb = 1 : ui32, handshake.name = "fork19"} : <i1>
    %48 = andi %45, %47#0 {handshake.bb = 1 : ui32, handshake.name = "andi6", specv2_tmp_and = true} : <i1>
    %49 = spec_v2_repeating_init %47#1 {handshake.bb = 1 : ui32, handshake.name = "spec_v2_repeating_init6", initToken = 1 : ui1} : <i1>
    %50:2 = fork [2] %49 {handshake.bb = 1 : ui32, handshake.name = "fork20"} : <i1>
    %51 = andi %48, %50#0 {handshake.bb = 1 : ui32, handshake.name = "andi7", specv2_tmp_and = true} : <i1>
    %52 = spec_v2_repeating_init %50#1 {handshake.bb = 1 : ui32, handshake.name = "spec_v2_repeating_init7", initToken = 1 : ui1} : <i1>
    %53:2 = fork [2] %52 {handshake.bb = 1 : ui32, handshake.name = "fork21"} : <i1>
    %54 = andi %51, %53#0 {handshake.bb = 1 : ui32, handshake.name = "andi8", specv2_tmp_and = true} : <i1>
    %55 = spec_v2_repeating_init %53#1 {handshake.bb = 1 : ui32, handshake.name = "spec_v2_repeating_init8", initToken = 1 : ui1} : <i1>
    %56:2 = fork [2] %55 {handshake.bb = 1 : ui32, handshake.name = "fork22"} : <i1>
    %57 = andi %54, %56#0 {handshake.bb = 1 : ui32, handshake.name = "andi9", specv2_tmp_and = true} : <i1>
    %58 = spec_v2_repeating_init %56#1 {handshake.bb = 1 : ui32, handshake.name = "spec_v2_repeating_init9", initToken = 1 : ui1} : <i1>
    %59:2 = fork [2] %58 {handshake.bb = 1 : ui32, handshake.name = "fork23"} : <i1>
    %60 = andi %57, %59#0 {handshake.bb = 1 : ui32, handshake.name = "andi10", specv2_tmp_and = true} : <i1>
    %61 = spec_v2_repeating_init %59#1 {handshake.bb = 1 : ui32, handshake.name = "spec_v2_repeating_init10", initToken = 1 : ui1} : <i1>
    %62:2 = fork [2] %61 {handshake.bb = 1 : ui32, handshake.name = "fork24"} : <i1>
    %63 = andi %60, %62#0 {handshake.bb = 1 : ui32, handshake.name = "andi11", specv2_tmp_and = true} : <i1>
    %64 = spec_v2_repeating_init %62#1 {handshake.bb = 1 : ui32, handshake.name = "spec_v2_repeating_init11", initToken = 1 : ui1} : <i1>
    %65:2 = fork [2] %64 {handshake.bb = 1 : ui32, handshake.name = "fork25"} : <i1>
    %66 = andi %63, %65#0 {handshake.bb = 1 : ui32, handshake.name = "andi12", specv2_tmp_and = true} : <i1>
    %67 = spec_v2_repeating_init %65#1 {handshake.bb = 1 : ui32, handshake.name = "spec_v2_repeating_init12", initToken = 1 : ui1} : <i1>
    %68:4 = fork [4] %67 {handshake.bb = 1 : ui32, handshake.name = "fork6"} : <i1>
    %69 = init %68#3 {handshake.bb = 1 : ui32, handshake.name = "init2", initToken = 0 : ui1} : <i1>
    %70:2 = fork [2] %69 {handshake.bb = 1 : ui32, handshake.name = "fork7"} : <i1>
    %71 = andi %66, %68#2 {handshake.bb = 1 : ui32, handshake.name = "andi13", specv2_tmp_and = true} : <i1>
    %72:3 = fork [3] %71 {handshake.bb = 1 : ui32, handshake.name = "fork27"} : <i1>
    %73 = andi %15, %72#0 {handshake.bb = 1 : ui32, handshake.name = "andi14"} : <i1>
    %74:2 = fork [2] %73 {handshake.bb = 1 : ui32, handshake.name = "fork28"} : <i1>
    %75 = andi %14, %72#1 {handshake.bb = 1 : ui32, handshake.name = "andi15"} : <i1>
    %76:2 = fork [2] %75 {handshake.bb = 1 : ui32, handshake.name = "fork29"} : <i1>
    %77 = not %28#1 {handshake.bb = 1 : ui32, handshake.name = "not1"} : <i1>
    %78 = passer %26#1[%76#0] {handshake.bb = 1 : ui32, handshake.name = "passer5"} : <i12>, <i1>
    %79:5 = fork [5] %3 {handshake.bb = 1 : ui32, handshake.name = "fork30"} : <i11>
    %80 = passer %81[%68#0] {handshake.bb = 1 : ui32, handshake.name = "passer10"} : <i11>, <i1>
    %81 = trunci %26#2 {handshake.bb = 1 : ui32, handshake.name = "trunci3"} : <i12> to <i11>
    %82 = passer %83#1[%68#1] {handshake.bb = 1 : ui32, handshake.name = "passer11"} : <>, <i1>
    %83:3 = fork [3] %7 {handshake.bb = 1 : ui32, handshake.name = "fork31"} : <>
    %84 = passer %83#0[%76#1] {handshake.bb = 1 : ui32, handshake.name = "passer7"} : <>, <i1>
    %85 = mux %index [%16, %78] {handshake.bb = 2 : ui32, handshake.name = "mux1"} : <i1>, [<i12>, <i12>] to <i12>
    %86 = extsi %85 {handshake.bb = 2 : ui32, handshake.name = "extsi8"} : <i12> to <i32>
    %result, %index = control_merge [%18, %84]  {handshake.bb = 2 : ui32, handshake.name = "control_merge2"} : [<>, <>] to <>, <i1>
    %87:3 = fork [3] %result {handshake.bb = 2 : ui32, handshake.name = "fork5"} : <>
    end {handshake.bb = 2 : ui32, handshake.name = "end0"} %86, %memEnd_3, %memEnd_1, %memEnd, %0#1 : <i32>, <>, <>, <>, <>
  }
}

