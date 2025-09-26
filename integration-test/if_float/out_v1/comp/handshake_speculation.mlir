module {
  handshake.func @if_float(%arg0: !handshake.channel<f32>, %arg1: memref<100xf32>, %arg2: memref<100xf32>, %arg3: !handshake.control<>, %arg4: !handshake.control<>, %arg5: !handshake.control<>, ...) -> (!handshake.channel<f32>, !handshake.control<>, !handshake.control<>, !handshake.control<>) attributes {argNames = ["x0", "a", "minus_trace", "a_start", "minus_trace_start", "start"], resNames = ["out0", "a_end", "minus_trace_end", "end"]} {
    %0 = non_spec %arg0 {handshake.bb = 1 : ui32, handshake.name = "non_spec0"} : !handshake.channel<f32> to !handshake.channel<f32, [spec: i1]>
    %1:3 = fork [3] %arg5 {handshake.bb = 0 : ui32, handshake.name = "fork0"} : <>
    %2 = non_spec %1#2 {handshake.bb = 1 : ui32, handshake.name = "non_spec1"} : !handshake.control<> to !handshake.control<[spec: i1]>
    %3 = spec_commit[%26#0] %64#1 {handshake.bb = 2 : ui32, handshake.name = "spec_commit0"} : !handshake.control<[spec: i1]>, !handshake.control<>, <i1>
    %4 = spec_commit[%27#1] %41 {handshake.bb = 1 : ui32, handshake.name = "spec_commit1"} : !handshake.channel<i32, [spec: i1]>, !handshake.channel<i32>, <i1>
    %memEnd = mem_controller[%arg2 : memref<100xf32>] %arg4 (%4, %addressResult_15, %dataResult_16) %3 {connectedBlocks = [1 : i32], handshake.name = "mem_controller2"} :    (!handshake.channel<i32>, !handshake.channel<i7>, !handshake.channel<f32>) -> ()
    %5 = spec_commit[%26#1] %64#0 {handshake.bb = 2 : ui32, handshake.name = "spec_commit2"} : !handshake.control<[spec: i1]>, !handshake.control<>, <i1>
    %outputs, %memEnd_0 = mem_controller[%arg1 : memref<100xf32>] %arg3 (%addressResult) %5 {connectedBlocks = [1 : i32], handshake.name = "mem_controller3"} :    (!handshake.channel<i7>) -> !handshake.channel<f32>
    %6 = constant %1#0 {handshake.bb = 0 : ui32, handshake.name = "constant2", value = false} : <>, <i1>
    %7 = extsi %6 {handshake.bb = 0 : ui32, handshake.name = "extsi4"} : <i1> to <i8>
    %8 = non_spec %7 {handshake.bb = 1 : ui32, handshake.name = "non_spec2"} : !handshake.channel<i8> to !handshake.channel<i8, [spec: i1]>
    %9 = mux %15#0 [%8, %trueResult_17] {handshake.bb = 1 : ui32, handshake.name = "mux0"} : <i1, [spec: i1]>, [<i8, [spec: i1]>, <i8, [spec: i1]>] to <i8, [spec: i1]>
    %10:2 = fork [2] %9 {handshake.bb = 1 : ui32, handshake.name = "fork1"} : <i8, [spec: i1]>
    %11 = trunci %10#0 {handshake.bb = 1 : ui32, handshake.name = "trunci0"} : <i8, [spec: i1]> to <i7, [spec: i1]>
    %12 = mux %15#1 [%0, %trueResult_19] {handshake.bb = 1 : ui32, handshake.name = "mux1"} : <i1, [spec: i1]>, [<f32, [spec: i1]>, <f32, [spec: i1]>] to <f32, [spec: i1]>
    %13:3 = fork [3] %12 {handshake.bb = 1 : ui32, handshake.name = "fork2"} : <f32, [spec: i1]>
    %result, %index = control_merge [%2, %trueResult_21]  {handshake.bb = 1 : ui32, handshake.name = "control_merge0"} : [<[spec: i1]>, <[spec: i1]>] to <[spec: i1]>, <i1, [spec: i1]>
    %14:2 = fork [2] %result {handshake.bb = 1 : ui32, handshake.name = "fork4"} : <[spec: i1]>
    %15:2 = fork [2] %index {handshake.bb = 1 : ui32, handshake.name = "fork3"} : <i1, [spec: i1]>
    %16 = source {handshake.bb = 1 : ui32, handshake.name = "source0"} : <[spec: i1]>
    %17 = constant %16 {handshake.bb = 1 : ui32, handshake.name = "constant6", value = -0.899999976 : f32} : <[spec: i1]>, <f32, [spec: i1]>
    %18 = source {handshake.bb = 1 : ui32, handshake.name = "source1"} : <[spec: i1]>
    %19 = constant %18 {handshake.bb = 1 : ui32, handshake.name = "constant7", value = 0.000000e+00 : f32} : <[spec: i1]>, <f32, [spec: i1]>
    %addressResult, %dataResult = load[%11] %outputs {handshake.bb = 1 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i7, [spec: i1]>, <f32>, <i7>, <f32, [spec: i1]>
    %20 = mulf %dataResult, %13#2 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "mulf0"} : <f32, [spec: i1]>
    %21 = mulf %13#1, %17 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "mulf1"} : <f32, [spec: i1]>
    %22 = addf %20, %21 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "addf0"} : <f32, [spec: i1]>
    %23 = cmpf ugt, %22, %19 {handshake.bb = 1 : ui32, handshake.name = "cmpf0"} : <f32, [spec: i1]>
    %dataOut, %SCSaveCtrl = spec_prebuffer1[%14#1] {fifoDepth = 30 : ui32, handshake.bb = 1 : ui32, handshake.name = "spec_prebuffer10"} : <[spec: i1]>, <i1, [spec: i1]>, <i3>
    %24:3 = fork [3] %SCSaveCtrl {handshake.bb = 1 : ui32, handshake.name = "fork9"} : <i3>
    %saveCtrl, %commitCtrl, %SCIsMisspec = spec_prebuffer2 %23 {fifoDepth = 30 : ui32, handshake.bb = 1 : ui32, handshake.name = "spec_prebuffer20"} : <i1, [spec: i1]>, <i1>, <i1>, <i1>
    %25:2 = fork [2] %commitCtrl {handshake.bb = 1 : ui32, handshake.name = "fork13"} : <i1>
    sink %saveCtrl {handshake.name = "sink0"} : <i1>
    %trueResult, %falseResult = cond_br %28#0, %25#0 {handshake.bb = 1 : ui32, handshake.name = "cond_br0"} : <i1>, <i1>
    %26:3 = fork [3] %falseResult {handshake.bb = 1 : ui32, handshake.name = "fork15"} : <i1>
    sink %trueResult {handshake.name = "sink1"} : <i1>
    %trueResult_1, %falseResult_2 = speculating_branch[%29#5] %29#6 {handshake.bb = 1 : ui32, handshake.name = "speculating_branch0"} : !handshake.channel<i1, [spec: i1]>, !handshake.channel<i1, [spec: i1]>, !handshake.channel<i1>, !handshake.channel<i1>
    sink %falseResult_2 {handshake.name = "sink3"} : <i1>
    %trueResult_3, %falseResult_4 = cond_br %trueResult_1, %25#1 {handshake.bb = 1 : ui32, handshake.name = "cond_br1"} : <i1>, <i1>
    sink %falseResult_4 {handshake.name = "sink4"} : <i1>
    %27:3 = fork [3] %trueResult_3 {handshake.bb = 1 : ui32, handshake.name = "fork16"} : <i1>
    %trueResult_5, %falseResult_6 = speculating_branch[%63#2] %63#3 {handshake.bb = 1 : ui32, handshake.name = "speculating_branch1"} : !handshake.channel<i1, [spec: i1]>, !handshake.channel<i1, [spec: i1]>, !handshake.channel<i1>, !handshake.channel<i1>
    sink %falseResult_6 {handshake.name = "sink5"} : <i1>
    %28:2 = fork [2] %trueResult_5 {handshake.bb = 1 : ui32, handshake.name = "fork17"} : <i1>
    %trueResult_7, %falseResult_8 = cond_br %SCIsMisspec, %28#1 {handshake.bb = 1 : ui32, handshake.name = "cond_br8"} : <i1>, <i1>
    sink %falseResult_8 {handshake.name = "sink6"} : <i1>
    sink %trueResult_7 {handshake.name = "sink7"} : <i1>
    %29:8 = fork [8] %dataOut {handshake.bb = 1 : ui32, handshake.name = "fork18"} : <i1, [spec: i1]>
    %30 = spec_save_commit[%24#0] %10#1 {fifoDepth = 30 : ui32, handshake.bb = 1 : ui32, handshake.name = "spec_save_commit0"} : !handshake.channel<i8, [spec: i1]>, <i3>
    %trueResult_9, %falseResult_10 = cond_br %29#7, %30 {handshake.bb = 1 : ui32, handshake.name = "cond_br2", specv1_adaptor_inner_loop = true} : <i1, [spec: i1]>, <i8, [spec: i1]>
    %31 = spec_save_commit[%24#1] %13#0 {fifoDepth = 30 : ui32, handshake.bb = 1 : ui32, handshake.name = "spec_save_commit1"} : !handshake.channel<f32, [spec: i1]>, <i3>
    %trueResult_11, %falseResult_12 = cond_br %29#4, %31 {handshake.bb = 1 : ui32, handshake.name = "cond_br3", specv1_adaptor_inner_loop = true} : <i1, [spec: i1]>, <f32, [spec: i1]>
    %32 = spec_save_commit[%24#2] %14#0 {fifoDepth = 30 : ui32, handshake.bb = 1 : ui32, handshake.name = "spec_save_commit2"} : !handshake.control<[spec: i1]>, <i3>
    %trueResult_13, %falseResult_14 = cond_br %29#3, %32 {handshake.bb = 1 : ui32, handshake.name = "cond_br4", specv1_adaptor_inner_loop = true} : <i1, [spec: i1]>, <[spec: i1]>
    %33 = source {handshake.bb = 1 : ui32, handshake.name = "source2"} : <[spec: i1]>
    %34 = constant %33 {handshake.bb = 1 : ui32, handshake.name = "constant8", value = 1.100000e+00 : f32} : <[spec: i1]>, <f32, [spec: i1]>
    %35 = mulf %falseResult_12, %34 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "mulf2"} : <f32, [spec: i1]>
    %36:2 = fork [2] %trueResult_9 {handshake.bb = 1 : ui32, handshake.name = "fork5"} : <i8, [spec: i1]>
    %37 = trunci %36#0 {handshake.bb = 1 : ui32, handshake.name = "trunci1"} : <i8, [spec: i1]> to <i7, [spec: i1]>
    %38:2 = fork [2] %trueResult_11 {handshake.bb = 1 : ui32, handshake.name = "fork6"} : <f32, [spec: i1]>
    %39:2 = fork [2] %trueResult_13 {handshake.bb = 1 : ui32, handshake.name = "fork7"} : <[spec: i1]>
    %40 = constant %39#0 {handshake.bb = 1 : ui32, handshake.name = "constant3", value = 1 : i2} : <[spec: i1]>, <i2, [spec: i1]>
    %41 = extsi %40 {handshake.bb = 1 : ui32, handshake.name = "extsi1"} : <i2, [spec: i1]> to <i32, [spec: i1]>
    %42 = source {handshake.bb = 1 : ui32, handshake.name = "source3"} : <[spec: i1]>
    %43 = constant %42 {handshake.bb = 1 : ui32, handshake.name = "constant9", value = 1.100000e+00 : f32} : <[spec: i1]>, <f32, [spec: i1]>
    %44 = spec_commit[%27#2] %37 {handshake.bb = 1 : ui32, handshake.name = "spec_commit3"} : !handshake.channel<i7, [spec: i1]>, !handshake.channel<i7>, <i1>
    %45 = spec_commit[%27#0] %38#0 {handshake.bb = 1 : ui32, handshake.name = "spec_commit4"} : !handshake.channel<f32, [spec: i1]>, !handshake.channel<f32>, <i1>
    %addressResult_15, %dataResult_16 = store[%44] %45 {handshake.bb = 1 : ui32, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "store0"} : <i7>, <f32>, <i7>, <f32>
    %46 = divf %38#1, %43 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "divf0"} : <f32, [spec: i1]>
    %47 = mux %29#1 [%35, %46] {handshake.bb = 1 : ui32, handshake.name = "mux2", specv1_adaptor_inner_loop = true} : <i1, [spec: i1]>, [<f32, [spec: i1]>, <f32, [spec: i1]>] to <f32, [spec: i1]>
    %48:2 = fork [2] %47 {handshake.bb = 1 : ui32, handshake.name = "fork8"} : <f32, [spec: i1]>
    %49 = mux %29#0 [%falseResult_10, %36#1] {handshake.bb = 1 : ui32, handshake.name = "mux3", specv1_adaptor_inner_loop = true} : <i1, [spec: i1]>, [<i8, [spec: i1]>, <i8, [spec: i1]>] to <i8, [spec: i1]>
    %50 = extsi %49 {handshake.bb = 1 : ui32, handshake.name = "extsi5"} : <i8, [spec: i1]> to <i9, [spec: i1]>
    %51 = mux %29#2 [%falseResult_14, %39#1] {handshake.bb = 1 : ui32, handshake.name = "mux4", specv1_adaptor_inner_loop = true} : <i1, [spec: i1]>, [<[spec: i1]>, <[spec: i1]>] to <[spec: i1]>
    %52 = source {handshake.bb = 1 : ui32, handshake.name = "source4"} : <[spec: i1]>
    %53 = constant %52 {handshake.bb = 1 : ui32, handshake.name = "constant4", value = 1 : i2} : <[spec: i1]>, <i2, [spec: i1]>
    %54 = extsi %53 {handshake.bb = 1 : ui32, handshake.name = "extsi6"} : <i2, [spec: i1]> to <i9, [spec: i1]>
    %55 = source {handshake.bb = 1 : ui32, handshake.name = "source5"} : <[spec: i1]>
    %56 = constant %55 {handshake.bb = 1 : ui32, handshake.name = "constant12", value = 100 : i8} : <[spec: i1]>, <i8, [spec: i1]>
    %57 = extsi %56 {handshake.bb = 1 : ui32, handshake.name = "extsi7"} : <i8, [spec: i1]> to <i9, [spec: i1]>
    %58 = addf %48#0, %48#1 {fastmath = #arith.fastmath<none>, handshake.bb = 1 : ui32, handshake.name = "addf1"} : <f32, [spec: i1]>
    %59 = addi %50, %54 {handshake.bb = 1 : ui32, handshake.name = "addi0"} : <i9, [spec: i1]>
    %60:2 = fork [2] %59 {handshake.bb = 1 : ui32, handshake.name = "fork10"} : <i9, [spec: i1]>
    %61 = trunci %60#0 {handshake.bb = 1 : ui32, handshake.name = "trunci2"} : <i9, [spec: i1]> to <i8, [spec: i1]>
    %62 = cmpi ult, %60#1, %57 {handshake.bb = 1 : ui32, handshake.name = "cmpi0"} : <i9, [spec: i1]>
    %63:5 = fork [5] %62 {handshake.bb = 1 : ui32, handshake.name = "fork19"} : <i1, [spec: i1]>
    %trueResult_17, %falseResult_18 = cond_br %63#4, %61 {handshake.bb = 1 : ui32, handshake.name = "cond_br5"} : <i1, [spec: i1]>, <i8, [spec: i1]>
    sink %falseResult_18 {handshake.name = "sink2"} : <i8, [spec: i1]>
    %trueResult_19, %falseResult_20 = cond_br %63#0, %58 {handshake.bb = 1 : ui32, handshake.name = "cond_br6"} : <i1, [spec: i1]>, <f32, [spec: i1]>
    %trueResult_21, %falseResult_22 = cond_br %63#1, %51 {handshake.bb = 1 : ui32, handshake.name = "cond_br7"} : <i1, [spec: i1]>, <[spec: i1]>
    %64:2 = fork [2] %falseResult_22 {handshake.bb = 2 : ui32, handshake.name = "fork12"} : <[spec: i1]>
    %65 = spec_commit[%26#2] %falseResult_20 {handshake.bb = 1 : ui32, handshake.name = "spec_commit5"} : !handshake.channel<f32, [spec: i1]>, !handshake.channel<f32>, <i1>
    end {handshake.bb = 2 : ui32, handshake.name = "end0"} %65, %memEnd_0, %memEnd, %1#1 : <f32>, <>, <>, <>
  }
}

