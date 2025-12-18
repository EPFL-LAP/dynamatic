module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @matrix_power(%arg0: memref<20x20xi32>, %arg1: memref<20xi32>, %arg2: memref<20xi32>, %arg3: memref<20xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c-1_i32 = arith.constant -1 : i32
    affine.for %arg4 = 1 to 20 {
      %0 = arith.index_cast %arg4 : index to i32
      %1 = arith.addi %0, %c-1_i32 : i32
      %2 = arith.index_cast %1 : i32 to index
      affine.for %arg5 = 0 to 20 {
        %3 = affine.load %arg1[%arg5] : memref<20xi32>
        %4 = arith.index_cast %3 : i32 to index
        %5 = affine.load %arg3[%arg5] : memref<20xi32>
        %6 = affine.load %arg2[%arg5] : memref<20xi32>
        %7 = arith.index_cast %6 : i32 to index
        %8 = memref.load %arg0[%2, %7] : memref<20x20xi32>
        %9 = arith.muli %5, %8 : i32
        %10 = memref.load %arg0[%arg4, %4] : memref<20x20xi32>
        %11 = arith.addi %10, %9 : i32
        memref.store %11, %arg0[%arg4, %4] : memref<20x20xi32>
      }
    }
    return
  }
}
