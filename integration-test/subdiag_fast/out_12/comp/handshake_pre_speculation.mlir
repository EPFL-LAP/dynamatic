module {
  handshake.func @subdiag_fast(%arg0: memref<1000xf32>, %arg1: memref<1000xf32>, %arg2: memref<1000xf32>, %arg3: !handshake.control<>, %arg4: !handshake.control<>, %arg5: !handshake.control<>, %arg6: !handshake.control<>, ...) -> (!handshake.channel<i32>, !handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["d1", "d2", "e", "d1_start", "d2_start", "e_start", "start"], resNames = ["out0", "d1_end", "d2_end", "e_end", "end"]} {
    %0 = source {handshake.bb = 2 : ui32, handshake.name = "source3"} : <>
    %1 = constant %0 {handshake.bb = 2 : ui32, handshake.name = "constant3", value = false} : <>, <i1>
    %2 = mux %21#0 [%1, %40#0] {handshake.bb = 2 : ui32, handshake.name = "mux2", specv2_loop_cond_mux = true} : <i1>, [<i1>, <i1>] to <i1>
    %3:2 = fork [2] %2 {handshake.bb = 2 : ui32, handshake.name = "fork6"} : <i1>
    %4:3 = fork [3] %arg6 {handshake.bb = 0 : ui32, handshake.name = "fork0"} : <>
    %outputs, %memEnd = mem_controller[%arg2 : memref<1000xf32>] %arg5 (%addressResult_6) %50#2 {connectedBlocks = [1 : i32], handshake.name = "mem_controller3"} :    (!handshake.channel<i10>) -> !handshake.channel<f32>
    %outputs_0, %memEnd_1 = mem_controller[%arg1 : memref<1000xf32>] %arg4 (%addressResult_4) %50#1 {connectedBlocks = [1 : i32], handshake.name = "mem_controller4"} :    (!handshake.channel<i10>) -> !handshake.channel<f32>
    %outputs_2, %memEnd_3 = mem_controller[%arg0 : memref<1000xf32>] %arg3 (%addressResult) %50#0 {connectedBlocks = [1 : i32], handshake.name = "mem_controller5"} :    (!handshake.channel<i10>) -> !handshake.channel<f32>
    %5 = constant %4#0 {handshake.bb = 0 : ui32, handshake.name = "constant0", value = false} : <>, <i1>
    %6 = extsi %5 {handshake.bb = 0 : ui32, handshake.name = "extsi3"} : <i1> to <i11>
    %7 = init %3#1 {handshake.bb = 1 : ui32, handshake.name = "init0", initToken = 0 : ui1} : <i1>
    %8 = init %3#0 {handshake.bb = 1 : ui32, handshake.name = "init1", initToken = 0 : ui1} : <i1>
    %9 = mux %7 [%6, %45] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1>, [<i11>, <i11>] to <i11>
    %10:5 = fork [5] %9 {handshake.bb = 1 : ui32, handshake.name = "fork7"} : <i11>
    %11 = trunci %10#0 {handshake.bb = 1 : ui32, handshake.name = "trunci0"} : <i11> to <i10>
    %12 = trunci %10#1 {handshake.bb = 1 : ui32, handshake.name = "trunci1"} : <i11> to <i10>
    %13 = trunci %10#2 {handshake.bb = 1 : ui32, handshake.name = "trunci2"} : <i11> to <i10>
    %14 = mux %8 [%4#2, %46] {handshake.bb = 1 : ui32, handshake.name = "mux3"} : <i1>, [<>, <>] to <>
    %15:2 = fork [2] %14 {handshake.bb = 1 : ui32, handshake.name = "fork8"} : <>
    %16 = source {handshake.bb = 1 : ui32, handshake.name = "source0"} : <>
    %17 = constant %16 {handshake.bb = 1 : ui32, handshake.name = "constant4", value = 1.000000e-03 : f32} : <>, <f32>
    %addressResult, %dataResult = load[%13] %outputs_2 {handshake.bb = 1 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i10>, <f32>, <i10>, <f32>
    %addressResult_4, %dataResult_5 = load[%12] %outputs_0 {handshake.bb = 1 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i10>, <f32>, <i10>, <f32>
    %18 = addf %dataResult, %dataResult_5 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "addf0"} : <f32>
    %addressResult_6, %dataResult_7 = load[%11] %outputs {handshake.bb = 1 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load2"} : <i10>, <f32>, <i10>, <f32>
    %19 = mulf %18, %17 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "mulf0"} : <f32>
    %20 = cmpf ugt, %dataResult_7, %19 {handshake.bb = 1 : ui32, handshake.name = "cmpf0"} : <f32>
    %21:4 = fork [4] %20 {handshake.bb = 1 : ui32, handshake.name = "fork9"} : <i1>
    %22 = not %21#3 {handshake.bb = 1 : ui32, handshake.name = "not0"} : <i1>
    %23:2 = fork [2] %22 {handshake.bb = 1 : ui32, handshake.name = "fork10"} : <i1>
    %24 = passer %10#4[%21#2] {handshake.bb = 1 : ui32, handshake.name = "passer0"} : <i11>, <i1>
    %25 = passer %10#3[%23#1] {handshake.bb = 1 : ui32, handshake.name = "passer1"} : <i11>, <i1>
    %26 = extsi %25 {handshake.bb = 1 : ui32, handshake.name = "extsi4"} : <i11> to <i12>
    %27 = passer %15#0[%21#1] {handshake.bb = 1 : ui32, handshake.name = "passer2"} : <>, <i1>
    %28:2 = fork [2] %27 {handshake.bb = 1 : ui32, handshake.name = "fork11"} : <>
    %29 = passer %15#1[%23#0] {handshake.bb = 1 : ui32, handshake.name = "passer3"} : <>, <i1>
    %30 = extsi %24 {handshake.bb = 2 : ui32, handshake.name = "extsi5"} : <i11> to <i12>
    %31 = source {handshake.bb = 2 : ui32, handshake.name = "source1"} : <>
    %32 = constant %31 {handshake.bb = 2 : ui32, handshake.name = "constant1", value = 1 : i2} : <>, <i2>
    %33 = extsi %32 {handshake.bb = 2 : ui32, handshake.name = "extsi6"} : <i2> to <i12>
    %34 = source {handshake.bb = 2 : ui32, handshake.name = "source2"} : <>
    %35 = constant %34 {handshake.bb = 2 : ui32, handshake.name = "constant2", value = 999 : i11} : <>, <i11>
    %36 = extsi %35 {handshake.bb = 2 : ui32, handshake.name = "extsi7"} : <i11> to <i12>
    %37 = addi %30, %33 {handshake.bb = 2 : ui32, handshake.name = "addi0"} : <i12>
    %38:3 = fork [3] %37 {handshake.bb = 2 : ui32, handshake.name = "fork12"} : <i12>
    %39 = cmpi ult, %38#0, %36 {handshake.bb = 2 : ui32, handshake.name = "cmpi0"} : <i12>
    %40:4 = fork [4] %39 {handshake.bb = 2 : ui32, handshake.name = "fork13"} : <i1>
    %41 = not %40#3 {handshake.bb = 2 : ui32, handshake.name = "not1"} : <i1>
    %42:2 = fork [2] %41 {handshake.bb = 2 : ui32, handshake.name = "fork14"} : <i1>
    %43 = passer %38#2[%40#2] {handshake.bb = 2 : ui32, handshake.name = "passer4"} : <i12>, <i1>
    %44 = passer %38#1[%42#1] {handshake.bb = 2 : ui32, handshake.name = "passer5"} : <i12>, <i1>
    %45 = trunci %43 {handshake.bb = 2 : ui32, handshake.name = "trunci3"} : <i12> to <i11>
    %46 = passer %28#1[%40#1] {handshake.bb = 2 : ui32, handshake.name = "passer6"} : <>, <i1>
    %47 = passer %28#0[%42#0] {handshake.bb = 2 : ui32, handshake.name = "passer7"} : <>, <i1>
    %48 = mux %index [%26, %44] {handshake.bb = 3 : ui32, handshake.name = "mux1"} : <i1>, [<i12>, <i12>] to <i12>
    %49 = extsi %48 {handshake.bb = 3 : ui32, handshake.name = "extsi8"} : <i12> to <i32>
    %result, %index = control_merge [%29, %47]  {handshake.bb = 3 : ui32, handshake.name = "control_merge2"} : [<>, <>] to <>, <i1>
    %50:3 = fork [3] %result {handshake.bb = 3 : ui32, handshake.name = "fork5"} : <>
    end {handshake.bb = 3 : ui32, handshake.name = "end0"} %49, %memEnd_3, %memEnd_1, %memEnd, %4#1 : <i32>, <>, <>, <>, <>
  }
}

