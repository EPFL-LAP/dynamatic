module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  handshake.func @fir(%arg0: memref<1000xi32>, %arg1: memref<1000xi32>, %arg2: none, ...) -> i32 attributes {argNames = ["in0", "in1", "in2"], llvm.linkage = #llvm.linkage<external>, resNames = ["out0"]} {
    %ldData, %done = mem_controller[%arg1 : memref<1000xi32>] (%addressResult) {accesses = [[#handshake<AccessType Load>]], id = 1 : i32} : (index) -> (i32, none)
    %ldData_0, %done_1 = mem_controller[%arg0 : memref<1000xi32>] (%addressResult_2) {accesses = [[#handshake<AccessType Load>]], id = 0 : i32} : (index) -> (i32, none)
    %0 = merge %arg2 {bb = 0 : ui32} : none
    %1 = constant %0 {bb = 0 : ui32, value = 0 : index} : index
    %2 = constant %0 {bb = 0 : ui32, value = 0 : i32} : i32
    %3 = br %1 {bb = 0 : ui32} : index
    %4 = br %2 {bb = 0 : ui32} : i32
    %5 = br %0 {bb = 0 : ui32} : none
    %6 = mux %index [%trueResult, %3] {bb = 1 : ui32} : index, index
    %7 = mux %index [%trueResult_4, %4] {bb = 1 : ui32} : index, i32
    %result, %index = control_merge %trueResult_6, %5 {bb = 1 : ui32} : none, index
    %8 = source {bb = 1 : ui32}
    %9 = constant %8 {bb = 1 : ui32, value = 999 : index} : index
    %10 = source {bb = 1 : ui32}
    %11 = constant %10 {bb = 1 : ui32, value = 1000 : index} : index
    %12 = source {bb = 1 : ui32}
    %13 = constant %12 {bb = 1 : ui32, value = 1 : index} : index
    %addressResult, %dataResult = d_load[%6] %ldData {bb = 1 : ui32} : index, i32
    %14 = arith.subi %9, %6 {bb = 1 : ui32} : index
    %addressResult_2, %dataResult_3 = d_load[%14] %ldData_0 {bb = 1 : ui32} : index, i32
    %15 = arith.muli %dataResult, %dataResult_3 {bb = 1 : ui32} : i32
    %16 = arith.addi %7, %15 {bb = 1 : ui32} : i32
    %17 = arith.addi %6, %13 {bb = 1 : ui32} : index
    %18 = arith.cmpi ult, %17, %11 {bb = 1 : ui32} : index
    %trueResult, %falseResult = cond_br %18, %17 {bb = 1 : ui32} : index
    %trueResult_4, %falseResult_5 = cond_br %18, %16 {bb = 1 : ui32} : i32
    %trueResult_6, %falseResult_7 = cond_br %18, %result {bb = 1 : ui32} : none
    %19 = merge %falseResult_5 {bb = 2 : ui32} : i32
    %result_8, %index_9 = control_merge %falseResult_7 {bb = 2 : ui32} : none, index
    %20 = d_return {bb = 2 : ui32} %19 : i32
    end {bb = 2 : ui32} %20, %done, %done_1 : i32, none, none
  }
}

