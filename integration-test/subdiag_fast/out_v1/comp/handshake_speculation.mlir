module {
  handshake.func @subdiag_fast(%arg0: memref<1000xf32>, %arg1: memref<1000xf32>, %arg2: memref<1000xf32>, %arg3: !handshake.control<>, %arg4: !handshake.control<>, %arg5: !handshake.control<>, %arg6: !handshake.control<>, ...) -> (!handshake.channel<i32>, !handshake.control<>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["d1", "d2", "e", "d1_start", "d2_start", "e_start", "start"], resNames = ["out0", "d1_end", "d2_end", "e_end", "end"]} {
    %0:3 = fork [3] %arg6 {handshake.bb = 0 : ui32, handshake.name = "fork0"} : <>
    %1 = non_spec %0#2 {handshake.bb = 1 : ui32, handshake.name = "non_spec0"} : !handshake.control<> to !handshake.control<[spec: i1]>
    %2 = spec_commit[%33#0] %38#2 {handshake.bb = 2 : ui32, handshake.name = "spec_commit0"} : !handshake.control<[spec: i1]>, !handshake.control<>, <i1>
    %outputs, %memEnd = mem_controller[%arg2 : memref<1000xf32>] %arg5 (%addressResult_6) %2 {connectedBlocks = [1 : i32], handshake.name = "mem_controller3"} :    (!handshake.channel<i10>) -> !handshake.channel<f32>
    %3 = spec_commit[%33#1] %38#1 {handshake.bb = 2 : ui32, handshake.name = "spec_commit1"} : !handshake.control<[spec: i1]>, !handshake.control<>, <i1>
    %outputs_0, %memEnd_1 = mem_controller[%arg1 : memref<1000xf32>] %arg4 (%addressResult_4) %3 {connectedBlocks = [1 : i32], handshake.name = "mem_controller4"} :    (!handshake.channel<i10>) -> !handshake.channel<f32>
    %4 = spec_commit[%33#2] %38#0 {handshake.bb = 2 : ui32, handshake.name = "spec_commit2"} : !handshake.control<[spec: i1]>, !handshake.control<>, <i1>
    %outputs_2, %memEnd_3 = mem_controller[%arg0 : memref<1000xf32>] %arg3 (%addressResult) %4 {connectedBlocks = [1 : i32], handshake.name = "mem_controller5"} :    (!handshake.channel<i10>) -> !handshake.channel<f32>
    %5 = constant %0#0 {handshake.bb = 0 : ui32, handshake.name = "constant0", value = false} : <>, <i1>
    %6 = extsi %5 {handshake.bb = 0 : ui32, handshake.name = "extsi3"} : <i1> to <i32>
    %7 = non_spec %6 {handshake.bb = 1 : ui32, handshake.name = "non_spec1"} : !handshake.channel<i32> to !handshake.channel<i32, [spec: i1]>
    %8 = mux %index [%7, %trueResult_12] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1, [spec: i1]>, [<i32, [spec: i1]>, <i32, [spec: i1]>] to <i32, [spec: i1]>
    %9:6 = fork [6] %8 {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <i32, [spec: i1]>
    %10 = trunci %9#0 {handshake.bb = 1 : ui32, handshake.name = "trunci0"} : <i32, [spec: i1]> to <i10, [spec: i1]>
    %11 = trunci %9#1 {handshake.bb = 1 : ui32, handshake.name = "trunci1"} : <i32, [spec: i1]> to <i10, [spec: i1]>
    %12 = trunci %9#2 {handshake.bb = 1 : ui32, handshake.name = "trunci2"} : <i32, [spec: i1]> to <i10, [spec: i1]>
    %result, %index = control_merge [%1, %trueResult_14]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<[spec: i1]>, <[spec: i1]>] to <[spec: i1]>, <i1, [spec: i1]>
    %13:2 = fork [2] %result {handshake.bb = 1 : ui32, handshake.name = "fork5"} : <[spec: i1]>
    %14 = source {handshake.bb = 1 : ui32, handshake.name = "source0"} : <[spec: i1]>
    %15 = constant %14 {handshake.bb = 1 : ui32, handshake.name = "constant1", value = 1 : i2} : <[spec: i1]>, <i2, [spec: i1]>
    %16:2 = fork [2] %15 {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <i2, [spec: i1]>
    %17 = extsi %16#0 {handshake.bb = 1 : ui32, handshake.name = "extsi4"} : <i2, [spec: i1]> to <i10, [spec: i1]>
    %18 = extsi %16#1 {handshake.bb = 1 : ui32, handshake.name = "extsi1"} : <i2, [spec: i1]> to <i32, [spec: i1]>
    %19 = source {handshake.bb = 1 : ui32, handshake.name = "source1"} : <[spec: i1]>
    %20 = constant %19 {handshake.bb = 1 : ui32, handshake.name = "constant5", value = 1.000000e-03 : f32} : <[spec: i1]>, <f32, [spec: i1]>
    %21 = source {handshake.bb = 1 : ui32, handshake.name = "source2"} : <[spec: i1]>
    %22 = constant %21 {handshake.bb = 1 : ui32, handshake.name = "constant2", value = 998 : i11} : <[spec: i1]>, <i11, [spec: i1]>
    %23 = extsi %22 {handshake.bb = 1 : ui32, handshake.name = "extsi2"} : <i11, [spec: i1]> to <i32, [spec: i1]>
    %addressResult, %dataResult = load[%12] %outputs_2 {handshake.bb = 1 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i10, [spec: i1]>, <f32>, <i10>, <f32, [spec: i1]>
    %24 = addi %10, %17 {handshake.bb = 1 : ui32, handshake.name = "addi0"} : <i10, [spec: i1]>
    %addressResult_4, %dataResult_5 = load[%24] %outputs_0 {handshake.bb = 1 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load1"} : <i10, [spec: i1]>, <f32>, <i10>, <f32, [spec: i1]>
    %25 = addf %dataResult, %dataResult_5 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "addf0"} : <f32, [spec: i1]>
    %26 = addi %9#5, %18 {handshake.bb = 1 : ui32, handshake.name = "addi1"} : <i32, [spec: i1]>
    %addressResult_6, %dataResult_7 = load[%11] %outputs {handshake.bb = 1 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load2"} : <i10, [spec: i1]>, <f32>, <i10>, <f32, [spec: i1]>
    %27 = mulf %25, %20 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "mulf0"} : <f32, [spec: i1]>
    %28 = cmpf ugt, %dataResult_7, %27 {handshake.bb = 1 : ui32, handshake.name = "cmpf0"} : <f32, [spec: i1]>
    %29 = cmpi ult, %9#4, %23 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i32, [spec: i1]>
    %30 = andi %29, %28 {handshake.bb = 1 : ui32, handshake.name = "andi0"} : <i1, [spec: i1]>
    %dataOut, %SCSaveCtrl = spec_prebuffer1[%13#1] {fifoDepth = 16 : ui32, handshake.bb = 1 : ui32, handshake.name = "spec_prebuffer10"} : <[spec: i1]>, <i1, [spec: i1]>, <i3>
    %31:3 = fork [3] %SCSaveCtrl {handshake.bb = 1 : ui32, handshake.name = "fork6"} : <i3>
    %32:5 = fork [5] %dataOut {handshake.bb = 1 : ui32, handshake.name = "fork7"} : <i1, [spec: i1]>
    %saveCtrl, %commitCtrl, %SCIsMisspec = spec_prebuffer2 %30 {fifoDepth = 16 : ui32, handshake.bb = 1 : ui32, handshake.name = "spec_prebuffer20"} : <i1, [spec: i1]>, <i1>, <i1>, <i1>
    sink %saveCtrl {handshake.name = "sink2"} : <i1>
    %trueResult, %falseResult = cond_br %34#0, %commitCtrl {handshake.bb = 1 : ui32, handshake.name = "cond_br0"} : <i1>, <i1>
    %33:4 = fork [4] %falseResult {handshake.bb = 1 : ui32, handshake.name = "fork8"} : <i1>
    sink %trueResult {handshake.name = "sink3"} : <i1>
    %trueResult_8, %falseResult_9 = speculating_branch[%32#0] %32#3 {handshake.bb = 1 : ui32, handshake.name = "speculating_branch0"} : !handshake.channel<i1, [spec: i1]>, !handshake.channel<i1, [spec: i1]>, !handshake.channel<i1>, !handshake.channel<i1>
    sink %falseResult_9 {handshake.name = "sink4"} : <i1>
    %34:2 = fork [2] %trueResult_8 {handshake.bb = 1 : ui32, handshake.name = "fork9"} : <i1>
    %trueResult_10, %falseResult_11 = cond_br %SCIsMisspec, %34#1 {handshake.bb = 1 : ui32, handshake.name = "cond_br4"} : <i1>, <i1>
    sink %falseResult_11 {handshake.name = "sink5"} : <i1>
    sink %trueResult_10 {handshake.name = "sink6"} : <i1>
    %35 = spec_save_commit[%31#2] %26 {fifoDepth = 16 : ui32, handshake.bb = 1 : ui32, handshake.name = "spec_save_commit0"} : !handshake.channel<i32, [spec: i1]>, <i3>
    %trueResult_12, %falseResult_13 = cond_br %32#4, %35 {handshake.bb = 1 : ui32, handshake.name = "cond_br1"} : <i1, [spec: i1]>, <i32, [spec: i1]>
    sink %falseResult_13 {handshake.name = "sink0"} : <i32, [spec: i1]>
    %36 = spec_save_commit[%31#1] %13#0 {fifoDepth = 16 : ui32, handshake.bb = 1 : ui32, handshake.name = "spec_save_commit1"} : !handshake.control<[spec: i1]>, <i3>
    %trueResult_14, %falseResult_15 = cond_br %32#2, %36 {handshake.bb = 1 : ui32, handshake.name = "cond_br2"} : <i1, [spec: i1]>, <[spec: i1]>
    %37 = spec_save_commit[%31#0] %9#3 {fifoDepth = 16 : ui32, handshake.bb = 1 : ui32, handshake.name = "spec_save_commit2"} : !handshake.channel<i32, [spec: i1]>, <i3>
    %trueResult_16, %falseResult_17 = cond_br %32#1, %37 {handshake.bb = 1 : ui32, handshake.name = "cond_br3"} : <i1, [spec: i1]>, <i32, [spec: i1]>
    sink %trueResult_16 {handshake.name = "sink1"} : <i32, [spec: i1]>
    %38:3 = fork [3] %falseResult_15 {handshake.bb = 2 : ui32, handshake.name = "fork4"} : <[spec: i1]>
    %39 = spec_commit[%33#3] %falseResult_17 {handshake.bb = 1 : ui32, handshake.name = "spec_commit3"} : !handshake.channel<i32, [spec: i1]>, !handshake.channel<i32>, <i1>
    end {handshake.bb = 2 : ui32, handshake.name = "end0"} %39, %memEnd_3, %memEnd_1, %memEnd, %0#1 : <i32>, <>, <>, <>, <>
  }
}

