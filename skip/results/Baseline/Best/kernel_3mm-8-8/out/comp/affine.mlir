module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_3mm(%arg0: memref<10x10xi32>, %arg1: memref<10x10xi32>, %arg2: memref<10x10xi32>, %arg3: memref<10x10xi32>, %arg4: memref<10x10xi32>, %arg5: memref<10x10xi32>, %arg6: memref<10x10xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    affine.for %arg7 = 0 to 10 {
      affine.for %arg8 = 0 to 10 {
        affine.store %c0_i32, %arg4[%arg7, %arg8] : memref<10x10xi32>
        affine.for %arg9 = 0 to 10 {
          %0 = affine.load %arg0[%arg7, %arg9] : memref<10x10xi32>
          %1 = affine.load %arg1[%arg9, %arg8] : memref<10x10xi32>
          %2 = arith.muli %0, %1 : i32
          %3 = affine.load %arg4[%arg7, %arg8] : memref<10x10xi32>
          %4 = arith.addi %3, %2 : i32
          affine.store %4, %arg4[%arg7, %arg8] : memref<10x10xi32>
        }
      }
    }
    affine.for %arg7 = 0 to 10 {
      affine.for %arg8 = 0 to 10 {
        affine.store %c0_i32, %arg5[%arg7, %arg8] : memref<10x10xi32>
        affine.for %arg9 = 0 to 10 {
          %0 = affine.load %arg2[%arg7, %arg9] : memref<10x10xi32>
          %1 = affine.load %arg3[%arg9, %arg8] : memref<10x10xi32>
          %2 = arith.muli %0, %1 : i32
          %3 = affine.load %arg5[%arg7, %arg8] : memref<10x10xi32>
          %4 = arith.addi %3, %2 : i32
          affine.store %4, %arg5[%arg7, %arg8] : memref<10x10xi32>
        }
      }
    }
    affine.for %arg7 = 0 to 10 {
      affine.for %arg8 = 0 to 10 {
        affine.store %c0_i32, %arg6[%arg7, %arg8] : memref<10x10xi32>
        affine.for %arg9 = 0 to 10 {
          %0 = affine.load %arg4[%arg7, %arg9] : memref<10x10xi32>
          %1 = affine.load %arg5[%arg9, %arg8] : memref<10x10xi32>
          %2 = arith.muli %0, %1 : i32
          %3 = affine.load %arg6[%arg7, %arg8] : memref<10x10xi32>
          %4 = arith.addi %3, %2 : i32
          affine.store %4, %arg6[%arg7, %arg8] : memref<10x10xi32>
        }
      }
    }
    return
  }
}
