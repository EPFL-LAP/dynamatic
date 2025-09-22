module {
  handshake.func @subdiag_fast(%arg0: memref<1000xf32>, %arg1: memref<1000xf32>, %arg2: memref<1000xf32>, %arg3: !handshake.control<>, %arg4: !handshake.control<>, %arg5: !handshake.control<>, %arg6: !handshake.control<>, ...) -> (!handshake.channel<i32>, !handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["d1", "d2", "e", "d1_start", "d2_start", "e_start", "start"], resNames = ["out0", "d1_end", "d2_end", "e_end", "end"]} {
    %0:3 = fork [3] %arg6 {handshake.bb = 0 : ui32, handshake.name = "fork0"} : <>
    %outputs, %memEnd = mem_controller[%arg2 : memref<1000xf32>] %arg5 (%addressResult_6) %84#2 {connectedBlocks = [1 : i32], handshake.name = "mem_controller3"} :    (!handshake.channel<i10>) -> !handshake.channel<f32>
    %outputs_0, %memEnd_1 = mem_controller[%arg1 : memref<1000xf32>] %arg4 (%addressResult_4) %84#1 {connectedBlocks = [1 : i32], handshake.name = "mem_controller4"} :    (!handshake.channel<i10>) -> !handshake.channel<f32>
    %outputs_2, %memEnd_3 = mem_controller[%arg0 : memref<1000xf32>] %arg3 (%addressResult) %84#0 {connectedBlocks = [1 : i32], handshake.name = "mem_controller5"} :    (!handshake.channel<i10>) -> !handshake.channel<f32>
    %1 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant0", value = false} : <>, <i1>
    %2 = extsi %1 {handshake.bb = 0 : ui32, handshake.name = "extsi3"} : <i1> to <i11>
    %3 = init %67#2 {handshake.bb = 1 : ui32, handshake.name = "init0", initToken = 0 : ui1} : <i1>
    %4 = init %67#1 {handshake.bb = 1 : ui32, handshake.name = "init1", initToken = 0 : ui1} : <i1>
    %5 = mux %3 [%2, %77] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i11>, <i11>] to <i11>
    %6 = trunci %76#0 {handshake.bb = 1 : ui32, handshake.name = "trunci0"} : <i11> to <i10>
    %7 = trunci %76#1 {handshake.bb = 1 : ui32, handshake.name = "trunci1"} : <i11> to <i10>
    %8 = trunci %76#2 {handshake.bb = 1 : ui32, handshake.name = "trunci2"} : <i11> to <i10>
    %9 = mux %4 [%0#2, %79] {handshake.bb = 1 : ui32, handshake.name = "mux3"} : <i1>, [<>, <>] to <>
    %10 = source {handshake.bb = 1 : ui32, handshake.name = "source0"} : <>
    %11 = constant %10 {handshake.bb = 1 : ui32, handshake.name = "constant4", value = 1.000000e-03 : f32} : <>, <f32>
    %addressResult, %dataResult = load[%8] %outputs_2 {handshake.bb = 1 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i10>, <f32>, <i10>, <f32>
    %addressResult_4, %dataResult_5 = load[%7] %outputs_0 {handshake.bb = 1 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i10>, <f32>, <i10>, <f32>
    %12 = addf %dataResult, %dataResult_5 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "addf0"} : <f32>
    %addressResult_6, %dataResult_7 = load[%6] %outputs {handshake.bb = 1 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load2"} : <i10>, <f32>, <i10>, <f32>
    %13 = mulf %12, %11 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "mulf0"} : <f32>
    %14:3 = fork [3] %15 {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <i1>
    %15 = cmpf ugt, %dataResult_7, %13 {handshake.bb = 1 : ui32, handshake.name = "cmpf0"} : <f32>
    %16 = andi %74, %14#1 {handshake.bb = 1 : ui32, handshake.name = "andi0"} : <i1>
    %17 = not %14#0 {handshake.bb = 1 : ui32, handshake.name = "not0"} : <i1>
    %18 = passer %19[%71#1] {handshake.bb = 1 : ui32, handshake.name = "passer8"} : <i12>, <i1>
    %19 = extsi %76#3 {handshake.bb = 1 : ui32, handshake.name = "extsi4"} : <i11> to <i12>
    %20 = passer %80#2[%71#0] {handshake.bb = 1 : ui32, handshake.name = "passer3"} : <>, <i1>
    %21 = extsi %76#4 {handshake.bb = 1 : ui32, handshake.name = "extsi5"} : <i11> to <i12>
    %22 = source {handshake.bb = 1 : ui32, handshake.name = "source1"} : <>
    %23 = constant %22 {handshake.bb = 1 : ui32, handshake.name = "constant1", value = 1 : i2} : <>, <i2>
    %24 = extsi %23 {handshake.bb = 1 : ui32, handshake.name = "extsi6"} : <i2> to <i12>
    %25 = source {handshake.bb = 1 : ui32, handshake.name = "source2"} : <>
    %26 = constant %25 {handshake.bb = 1 : ui32, handshake.name = "constant2", value = 999 : i11} : <>, <i11>
    %27 = extsi %26 {handshake.bb = 1 : ui32, handshake.name = "extsi7"} : <i11> to <i12>
    %28:3 = fork [3] %29 {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <i12>
    %29 = addi %21, %24 {handshake.bb = 1 : ui32, handshake.name = "addi0"} : <i12>
    %30:2 = fork [2] %31 {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <i1>
    %31 = cmpi ult, %28#0, %27 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i12>
    %32 = passer %33[%69#2] {handshake.bb = 1 : ui32, handshake.name = "passer9"} : <i1>, <i1>
    %33 = andi %14#2, %30#0 {handshake.bb = 1 : ui32, handshake.name = "andi1"} : <i1>
    %34 = spec_v2_repeating_init %32 {handshake.bb = 1 : ui32, handshake.name = "spec_v2_repeating_init0", initToken = 1 : ui1} : <i1>
    %35:2 = fork [2] %34 {handshake.bb = 1 : ui32, handshake.name = "fork4"} : <i1>
    %36 = spec_v2_repeating_init %35#0 {handshake.bb = 1 : ui32, handshake.name = "spec_v2_repeating_init1", initToken = 1 : ui1} : <i1>
    %37:2 = fork [2] %36 {handshake.bb = 1 : ui32, handshake.name = "fork15"} : <i1>
    %38 = andi %35#1, %37#0 {handshake.bb = 1 : ui32, handshake.name = "andi2", specv2_tmp_and = true} : <i1>
    %39 = spec_v2_repeating_init %37#1 {handshake.bb = 1 : ui32, handshake.name = "spec_v2_repeating_init2", initToken = 1 : ui1} : <i1>
    %40:2 = fork [2] %39 {handshake.bb = 1 : ui32, handshake.name = "fork16"} : <i1>
    %41 = andi %38, %40#0 {handshake.bb = 1 : ui32, handshake.name = "andi3", specv2_tmp_and = true} : <i1>
    %42 = spec_v2_repeating_init %40#1 {handshake.bb = 1 : ui32, handshake.name = "spec_v2_repeating_init3", initToken = 1 : ui1} : <i1>
    %43:2 = fork [2] %42 {handshake.bb = 1 : ui32, handshake.name = "fork17"} : <i1>
    %44 = andi %41, %43#0 {handshake.bb = 1 : ui32, handshake.name = "andi4", specv2_tmp_and = true} : <i1>
    %45 = spec_v2_repeating_init %43#1 {handshake.bb = 1 : ui32, handshake.name = "spec_v2_repeating_init4", initToken = 1 : ui1} : <i1>
    %46:2 = fork [2] %45 {handshake.bb = 1 : ui32, handshake.name = "fork18"} : <i1>
    %47 = andi %44, %46#0 {handshake.bb = 1 : ui32, handshake.name = "andi5", specv2_tmp_and = true} : <i1>
    %48 = spec_v2_repeating_init %46#1 {handshake.bb = 1 : ui32, handshake.name = "spec_v2_repeating_init5", initToken = 1 : ui1} : <i1>
    %49:2 = fork [2] %48 {handshake.bb = 1 : ui32, handshake.name = "fork19"} : <i1>
    %50 = andi %47, %49#0 {handshake.bb = 1 : ui32, handshake.name = "andi6", specv2_tmp_and = true} : <i1>
    %51 = spec_v2_repeating_init %49#1 {handshake.bb = 1 : ui32, handshake.name = "spec_v2_repeating_init6", initToken = 1 : ui1} : <i1>
    %52:2 = fork [2] %51 {handshake.bb = 1 : ui32, handshake.name = "fork20"} : <i1>
    %53 = andi %50, %52#0 {handshake.bb = 1 : ui32, handshake.name = "andi7", specv2_tmp_and = true} : <i1>
    %54 = spec_v2_repeating_init %52#1 {handshake.bb = 1 : ui32, handshake.name = "spec_v2_repeating_init7", initToken = 1 : ui1} : <i1>
    %55:2 = fork [2] %54 {handshake.bb = 1 : ui32, handshake.name = "fork21"} : <i1>
    %56 = andi %53, %55#0 {handshake.bb = 1 : ui32, handshake.name = "andi8", specv2_tmp_and = true} : <i1>
    %57 = spec_v2_repeating_init %55#1 {handshake.bb = 1 : ui32, handshake.name = "spec_v2_repeating_init8", initToken = 1 : ui1} : <i1>
    %58:2 = fork [2] %57 {handshake.bb = 1 : ui32, handshake.name = "fork22"} : <i1>
    %59 = andi %56, %58#0 {handshake.bb = 1 : ui32, handshake.name = "andi9", specv2_tmp_and = true} : <i1>
    %60 = spec_v2_repeating_init %58#1 {handshake.bb = 1 : ui32, handshake.name = "spec_v2_repeating_init9", initToken = 1 : ui1} : <i1>
    %61:2 = fork [2] %60 {handshake.bb = 1 : ui32, handshake.name = "fork23"} : <i1>
    %62 = andi %59, %61#0 {handshake.bb = 1 : ui32, handshake.name = "andi10", specv2_tmp_and = true} : <i1>
    %63 = spec_v2_repeating_init %61#1 {handshake.bb = 1 : ui32, handshake.name = "spec_v2_repeating_init10", initToken = 1 : ui1} : <i1>
    %64:2 = fork [2] %63 {handshake.bb = 1 : ui32, handshake.name = "fork24"} : <i1>
    %65 = andi %62, %64#0 {handshake.bb = 1 : ui32, handshake.name = "andi11", specv2_tmp_and = true} : <i1>
    %66 = spec_v2_repeating_init %64#1 {handshake.bb = 1 : ui32, handshake.name = "spec_v2_repeating_init11", initToken = 1 : ui1} : <i1>
    %67:5 = fork [5] %66 {handshake.bb = 1 : ui32, handshake.name = "fork25"} : <i1>
    %68 = andi %65, %67#0 {handshake.bb = 1 : ui32, handshake.name = "andi12", specv2_tmp_and = true} : <i1>
    %69:3 = fork [3] %68 {handshake.bb = 1 : ui32, handshake.name = "fork26"} : <i1>
    %70 = andi %17, %69#0 {handshake.bb = 1 : ui32, handshake.name = "andi13"} : <i1>
    %71:2 = fork [2] %70 {handshake.bb = 1 : ui32, handshake.name = "fork27"} : <i1>
    %72 = andi %16, %69#1 {handshake.bb = 1 : ui32, handshake.name = "andi14"} : <i1>
    %73:2 = fork [2] %72 {handshake.bb = 1 : ui32, handshake.name = "fork28"} : <i1>
    %74 = not %30#1 {handshake.bb = 1 : ui32, handshake.name = "not1"} : <i1>
    %75 = passer %28#1[%73#1] {handshake.bb = 1 : ui32, handshake.name = "passer5"} : <i12>, <i1>
    %76:5 = fork [5] %5 {handshake.bb = 1 : ui32, handshake.name = "fork29"} : <i11>
    %77 = passer %78[%67#3] {handshake.bb = 1 : ui32, handshake.name = "passer10"} : <i11>, <i1>
    %78 = trunci %28#2 {handshake.bb = 1 : ui32, handshake.name = "trunci3"} : <i12> to <i11>
    %79 = passer %80#1[%67#4] {handshake.bb = 1 : ui32, handshake.name = "passer11"} : <>, <i1>
    %80:3 = fork [3] %9 {handshake.bb = 1 : ui32, handshake.name = "fork30"} : <>
    %81 = passer %80#0[%73#0] {handshake.bb = 1 : ui32, handshake.name = "passer7"} : <>, <i1>
    %82 = mux %index [%18, %75] {handshake.bb = 2 : ui32, handshake.name = "mux1"} : <i1>, [<i12>, <i12>] to <i12>
    %83 = extsi %82 {handshake.bb = 2 : ui32, handshake.name = "extsi8"} : <i12> to <i32>
    %result, %index = control_merge [%20, %81]  {handshake.bb = 2 : ui32, handshake.name = "control_merge2"} : [<>, <>] to <>, <i1>
    %84:3 = fork [3] %result {handshake.bb = 2 : ui32, handshake.name = "fork5"} : <>
    end {handshake.bb = 2 : ui32, handshake.name = "end0"} %83, %memEnd_3, %memEnd_1, %memEnd, %0#1 : <i32>, <>, <>, <>, <>
  }
}

