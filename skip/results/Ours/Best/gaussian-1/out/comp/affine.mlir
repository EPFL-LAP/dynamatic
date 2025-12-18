#map = affine_map<(d0) -> (d0 + 1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @gaussian(%arg0: memref<20xi32>, %arg1: memref<20x20xi32>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = llvm.mlir.undef : i32
    %1 = affine.for %arg2 = 1 to 19 iter_args(%arg3 = %c0_i32) -> (i32) {
      %2 = affine.for %arg4 = #map(%arg2) to 19 iter_args(%arg5 = %arg3) -> (i32) {
        %3:6 = affine.for %arg6 = 1 to 20 iter_args(%arg7 = %c1_i32, %arg8 = %arg5, %arg9 = %arg5, %arg10 = %0, %arg11 = %c1_i32, %arg12 = %arg5) -> (i32, i32, i32, i32, i32, i32) {
          %4 = arith.index_cast %arg7 : i32 to index
          %5 = memref.load %arg1[%arg4, %4] : memref<20x20xi32>
          %6 = affine.load %arg0[%arg2] : memref<20xi32>
          %7 = memref.load %arg1[%arg2, %4] : memref<20x20xi32>
          %8 = arith.muli %6, %7 : i32
          %9 = arith.subi %5, %8 : i32
          memref.store %9, %arg1[%arg4, %4] : memref<20x20xi32>
          %10 = arith.addi %arg8, %arg7 : i32
          %11 = arith.addi %arg7, %c1_i32 : i32
          affine.yield %11, %10, %10, %0, %11, %10 : i32, i32, i32, i32, i32, i32
        }
        affine.yield %3#2 : i32
      }
      affine.yield %2 : i32
    }
    return %1 : i32
  }
}
