module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  handshake.func @bicg(%arg0: memref<900xi32>, %arg1: memref<30xi32>, %arg2: memref<30xi32>, %arg3: memref<30xi32>, %arg4: memref<30xi32>, %arg5: none, ...) -> i32 attributes {argNames = ["in0", "in1", "in2", "in3", "in4", "in5"], bufProps = #handshake<opBufProps{"0": [0,0], [0,0], "1": [0,0], [0,0], "2": [0,0], [0,0], "3": [0,0], [0,0], "4": [0,0], [0,0]}>, llvm.linkage = #llvm.linkage<external>, resNames = ["out0"]} {
    %ldData, %done = mem_controller[%arg4 : memref<30xi32>] (%addressResult_14) {accesses = [[#handshake<AccessType Load>]], bufProps = #handshake<opBufProps{"0": [0,0], [0,0], "1": [0,0], [0,0]}>, id = 4 : i32} : (i32) -> (i32, none)
    %ldData_0, %done_1 = mem_controller[%arg3 : memref<30xi32>] (%addressResult_18) {accesses = [[#handshake<AccessType Load>]], bufProps = #handshake<opBufProps{"0": [0,0], [0,0], "1": [0,0], [0,0]}>, id = 3 : i32} : (i32) -> (i32, none)
    %ldData_2, %done_3 = mem_controller[%arg2 : memref<30xi32>] (%addressResult, %102, %addressResult_26, %dataResult_27) {accesses = [[#handshake<AccessType Load>], [#handshake<AccessType Store>]], bufProps = #handshake<opBufProps{"0": [0,0], [0,0], "1": [0,0], [0,0]}>, id = 2 : i32} : (i32, i32, i32, i32) -> (i32, none)
    %ldData_4, %done_5 = mem_controller[%arg1 : memref<30xi32>] (%38, %addressResult_12, %addressResult_16, %dataResult_17) {accesses = [[#handshake<AccessType Load>, #handshake<AccessType Store>]], bufProps = #handshake<opBufProps{"0": [0,0], [0,0], "1": [0,0], [0,0]}>, id = 1 : i32} : (i32, i32, i32, i32) -> (i32, none)
    %ldData_6, %done_7 = mem_controller[%arg0 : memref<900xi32>] (%addressResult_10) {accesses = [[#handshake<AccessType Load>]], bufProps = #handshake<opBufProps{"0": [0,0], [0,0], "1": [0,0], [0,0]}>, id = 0 : i32} : (i32) -> (i32, none)
    %0:2 = fork [2] %arg5 {bb = 0 : ui32} : none
    %1 = constant %0#1 {bb = 0 : ui32, value = false} : i1
    %2 = arith.extsi %1 {bb = 0 : ui32} : i1 to i6
    %3 = mux %index [%trueResult_28, %2] {bb = 1 : ui32} : i1, i6
    %4 = buffer [1] seq %3 {bb = 1 : ui32} : i6
    %5:2 = fork [2] %4 {bb = 1 : ui32} : i6
    %6 = arith.extsi %5#1 {bb = 1 : ui32} : i6 to i32
    %result, %index = control_merge %trueResult_30, %0#0 {bb = 1 : ui32} : none, i1
    %7 = buffer [1] seq %result {bb = 1 : ui32} : none
    %8:2 = fork [2] %7 {bb = 1 : ui32} : none
    %9 = constant %8#0 {bb = 1 : ui32, value = false} : i1
    %addressResult, %dataResult = d_load[%6] %ldData_2 {bb = 1 : ui32, bufProps = #handshake<opBufProps{"0": [0,0], [0,0]}>} : i32, i32
    %10 = arith.extsi %9 {bb = 1 : ui32} : i1 to i6
    %11 = mux %33#1 [%trueResult, %10] {bb = 2 : ui32} : i1, i6
    %12:3 = fork [3] %11 {bb = 2 : ui32} : i6
    %13 = buffer [3] seq %12#0 {bb = 2 : ui32} : i6
    %14 = arith.extsi %13 {bb = 2 : ui32} : i6 to i13
    %15 = arith.extsi %12#1 {bb = 2 : ui32} : i6 to i7
    %16 = buffer [1] seq %15 {bb = 2 : ui32} : i7
    %17 = arith.extsi %12#2 {bb = 2 : ui32} : i6 to i32
    %18 = buffer [3] fifo %17 {bb = 2 : ui32} : i32
    %19:3 = fork [3] %18 {bb = 2 : ui32} : i32
    %20 = buffer [3] fifo %19#1 {bb = 2 : ui32} : i32
    %21 = buffer [2] fifo %19#0 {bb = 2 : ui32} : i32
    %22 = mux %34 [%trueResult_20, %dataResult] {bb = 2 : ui32} : i1, i32
    %23 = buffer [3] seq %22 {bb = 2 : ui32} : i32
    %24 = mux %33#0 [%trueResult_22, %5#0] {bb = 2 : ui32} : i1, i6
    %25 = buffer [2] seq %24 {bb = 2 : ui32} : i6
    %26:6 = fork [6] %25 {bb = 2 : ui32} : i6
    %27 = arith.extsi %26#1 {bb = 2 : ui32} : i6 to i10
    %28 = arith.extsi %26#2 {bb = 2 : ui32} : i6 to i9
    %29 = arith.extsi %26#3 {bb = 2 : ui32} : i6 to i8
    %30 = arith.extsi %26#4 {bb = 2 : ui32} : i6 to i7
    %31 = arith.extsi %26#5 {bb = 2 : ui32} : i6 to i32
    %32 = buffer [2] fifo %31 {bb = 2 : ui32} : i32
    %result_8, %index_9 = control_merge %trueResult_24, %8#1 {bb = 2 : ui32} : none, i1
    %33:3 = fork [3] %index_9 {bb = 2 : ui32} : i1
    %34 = buffer [5] fifo %33#2 {bb = 2 : ui32} : i1
    %35:2 = fork [2] %result_8 {bb = 2 : ui32} : none
    %36 = buffer [2] seq %35#1 {bb = 2 : ui32} : none
    %37 = constant %35#0 {bb = 2 : ui32, value = 1 : i2} : i2
    %38 = arith.extsi %37 {bb = 2 : ui32, bufProps = #handshake<opBufProps{"0": [0,0], [0,0]}>} : i2 to i32
    %39 = source {bb = 2 : ui32}
    %40 = constant %39 {bb = 2 : ui32, value = 30 : i6} : i6
    %41 = arith.extsi %40 {bb = 2 : ui32} : i6 to i7
    %42 = source {bb = 2 : ui32}
    %43 = constant %42 {bb = 2 : ui32, value = 1 : i2} : i2
    %44:2 = fork [2] %43 {bb = 2 : ui32} : i2
    %45 = buffer [1] fifo %44#0 {bb = 2 : ui32} : i2
    %46 = arith.extui %45 {bb = 2 : ui32} : i2 to i7
    %47 = arith.extsi %44#1 {bb = 2 : ui32} : i2 to i7
    %48 = buffer [1] seq %47 {bb = 2 : ui32} : i7
    %49 = source {bb = 2 : ui32}
    %50 = constant %49 {bb = 2 : ui32, value = 4 : i4} : i4
    %51 = arith.extui %50 {bb = 2 : ui32} : i4 to i10
    %52 = source {bb = 2 : ui32}
    %53 = constant %52 {bb = 2 : ui32, value = 3 : i3} : i3
    %54 = arith.extui %53 {bb = 2 : ui32} : i3 to i9
    %55 = source {bb = 2 : ui32}
    %56 = constant %55 {bb = 2 : ui32, value = 2 : i3} : i3
    %57 = arith.extui %56 {bb = 2 : ui32} : i3 to i8
    %58 = arith.shli %30, %46 {bb = 2 : ui32} : i7
    %59 = buffer [1] seq %58 {bb = 2 : ui32} : i7
    %60 = arith.extsi %59 {bb = 2 : ui32} : i7 to i9
    %61 = arith.shli %29, %57 {bb = 2 : ui32} : i8
    %62 = arith.extsi %61 {bb = 2 : ui32} : i8 to i9
    %63 = buffer [1] seq %62 {bb = 2 : ui32} : i9
    %64 = arith.addi %60, %63 {bb = 2 : ui32} : i9
    %65 = buffer [1] seq %64 {bb = 2 : ui32} : i9
    %66 = arith.extsi %65 {bb = 2 : ui32} : i9 to i12
    %67 = arith.shli %28, %54 {bb = 2 : ui32} : i9
    %68 = buffer [1] seq %67 {bb = 2 : ui32} : i9
    %69 = arith.extsi %68 {bb = 2 : ui32} : i9 to i11
    %70 = arith.shli %27, %51 {bb = 2 : ui32} : i10
    %71 = buffer [1] seq %70 {bb = 2 : ui32} : i10
    %72 = arith.extsi %71 {bb = 2 : ui32} : i10 to i11
    %73 = arith.addi %69, %72 {bb = 2 : ui32} : i11
    %74 = buffer [1] seq %73 {bb = 2 : ui32} : i11
    %75 = arith.extsi %74 {bb = 2 : ui32} : i11 to i12
    %76 = arith.addi %66, %75 {bb = 2 : ui32} : i12
    %77 = buffer [1] seq %76 {bb = 2 : ui32} : i12
    %78 = arith.extsi %77 {bb = 2 : ui32} : i12 to i13
    %79 = arith.addi %14, %78 {bb = 2 : ui32} : i13
    %80 = arith.extsi %79 {bb = 2 : ui32} : i13 to i32
    %addressResult_10, %dataResult_11 = d_load[%80] %ldData_6 {bb = 2 : ui32, bufProps = #handshake<opBufProps{"0": [0,0], [0,0]}>} : i32, i32
    %81:2 = fork [2] %dataResult_11 {bb = 2 : ui32} : i32
    %addressResult_12, %dataResult_13 = d_load[%21] %ldData_4 {bb = 2 : ui32, bufProps = #handshake<opBufProps{"0": [0,0], [0,0]}>} : i32, i32
    %addressResult_14, %dataResult_15 = d_load[%32] %ldData {bb = 2 : ui32, bufProps = #handshake<opBufProps{"0": [0,0], [0,0]}>} : i32, i32
    %82 = arith.muli %dataResult_15, %81#1 {bb = 2 : ui32} : i32
    %83 = arith.addi %dataResult_13, %82 {bb = 2 : ui32} : i32
    %addressResult_16, %dataResult_17 = d_store[%20] %83 {bb = 2 : ui32, bufProps = #handshake<opBufProps{"0": [0,0], [0,0], "1": [0,0], [0,0]}>} : i32, i32
    %addressResult_18, %dataResult_19 = d_load[%19#2] %ldData_0 {bb = 2 : ui32, bufProps = #handshake<opBufProps{"0": [0,0], [0,0]}>} : i32, i32
    %84 = arith.muli %81#0, %dataResult_19 {bb = 2 : ui32} : i32
    %85 = arith.addi %23, %84 {bb = 2 : ui32} : i32
    %86 = arith.addi %16, %48 {bb = 2 : ui32} : i7
    %87:2 = fork [2] %86 {bb = 2 : ui32} : i7
    %88 = arith.trunci %87#0 {bb = 2 : ui32} : i7 to i6
    %89 = buffer [1] seq %88 {bb = 2 : ui32} : i6
    %90 = arith.cmpi ult, %87#1, %41 {bb = 2 : ui32} : i7
    %91 = buffer [1] seq %90 {bb = 2 : ui32} : i1
    %92:4 = fork [4] %91 {bb = 2 : ui32} : i1
    %93 = buffer [5] fifo %92#2 {bb = 2 : ui32} : i1
    %trueResult, %falseResult = cond_br %92#0, %89 {bb = 2 : ui32} : i6
    sink %falseResult : i6
    %trueResult_20, %falseResult_21 = cond_br %93, %85 {bb = 2 : ui32} : i32
    %trueResult_22, %falseResult_23 = cond_br %92#1, %26#0 {bb = 2 : ui32} : i6
    %trueResult_24, %falseResult_25 = cond_br %92#3, %36 {bb = 2 : ui32} : none
    %94:2 = fork [2] %falseResult_23 {bb = 3 : ui32} : i6
    %95 = buffer [3] fifo %94#1 {bb = 3 : ui32} : i6
    %96 = arith.extsi %94#0 {bb = 3 : ui32} : i6 to i7
    %97 = arith.extsi %95 {bb = 3 : ui32} : i6 to i32
    %98:2 = fork [2] %falseResult_21 {bb = 3 : ui32} : i32
    %99:2 = fork [2] %falseResult_25 {bb = 3 : ui32} : none
    %100 = buffer [1] fifo %99#0 {bb = 3 : ui32} : none
    %101 = constant %99#1 {bb = 3 : ui32, value = 1 : i2} : i2
    %102 = arith.extsi %101 {bb = 3 : ui32, bufProps = #handshake<opBufProps{"0": [0,0], [0,0]}>} : i2 to i32
    %103 = source {bb = 3 : ui32}
    %104 = constant %103 {bb = 3 : ui32, value = 30 : i6} : i6
    %105 = arith.extsi %104 {bb = 3 : ui32} : i6 to i7
    %106 = buffer [1] seq %105 {bb = 3 : ui32} : i7
    %107 = source {bb = 3 : ui32}
    %108 = constant %107 {bb = 3 : ui32, value = 1 : i2} : i2
    %109 = arith.extsi %108 {bb = 3 : ui32} : i2 to i7
    %addressResult_26, %dataResult_27 = d_store[%97] %98#1 {bb = 3 : ui32, bufProps = #handshake<opBufProps{"0": [0,0], [0,0], "1": [0,0], [0,0]}>} : i32, i32
    %110 = arith.addi %96, %109 {bb = 3 : ui32} : i7
    %111 = buffer [1] seq %110 {bb = 3 : ui32} : i7
    %112:2 = fork [2] %111 {bb = 3 : ui32} : i7
    %113 = arith.trunci %112#0 {bb = 3 : ui32} : i7 to i6
    %114 = arith.cmpi ult, %112#1, %106 {bb = 3 : ui32} : i7
    %115:3 = fork [3] %114 {bb = 3 : ui32} : i1
    %116 = buffer [2] fifo %115#2 {bb = 3 : ui32} : i1
    %trueResult_28, %falseResult_29 = cond_br %115#0, %113 {bb = 3 : ui32} : i6
    sink %falseResult_29 : i6
    %trueResult_30, %falseResult_31 = cond_br %115#1, %100 {bb = 3 : ui32} : none
    sink %falseResult_31 : none
    %trueResult_32, %falseResult_33 = cond_br %116, %98#0 {bb = 3 : ui32} : i32
    %117 = buffer [1] seq %falseResult_33 {bb = 3 : ui32} : i32
    sink %trueResult_32 : i32
    %118 = d_return {bb = 4 : ui32} %117 : i32
    end {bb = 4 : ui32} %118, %done, %done_1, %done_3, %done_5, %done_7 : i32, none, none, none, none, none
  }
}

