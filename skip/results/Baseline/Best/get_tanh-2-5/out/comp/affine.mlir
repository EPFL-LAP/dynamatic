module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @get_tanh(%arg0: memref<1000xf32>, %arg1: memref<1000xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 3.70476198 : f32
    %cst_0 = arith.constant 19.5238094 : f32
    %cst_1 = arith.constant 1.000000e+00 : f32
    affine.for %arg2 = 0 to 1000 {
      %0 = affine.load %arg1[%arg2] : memref<1000xi32>
      %1 = arith.index_cast %0 : i32 to index
      %2 = memref.load %arg0[%1] : memref<1000xf32>
      %3 = arith.cmpf oge, %2, %cst_1 : f32
      %4 = scf.if %3 -> (f32) {
        scf.yield %cst_1 : f32
      } else {
        %5 = arith.mulf %2, %2 : f32
        %6 = arith.addf %5, %cst_0 : f32
        %7 = arith.mulf %6, %2 : f32
        %8 = arith.mulf %7, %2 : f32
        %9 = arith.addf %8, %cst : f32
        %10 = arith.mulf %9, %2 : f32
        scf.yield %10 : f32
      }
      memref.store %4, %arg0[%1] : memref<1000xf32>
    }
    return
  }
}
