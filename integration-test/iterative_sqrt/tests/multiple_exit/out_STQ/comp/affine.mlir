module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @multiple_exit(%arg0: memref<10xi32>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %true = arith.constant true
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %false = arith.constant false
    %c-1_i32 = arith.constant -1 : i32
    %c10_i32 = arith.constant 10 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = llvm.mlir.undef : i32
    %1:3 = scf.while (%arg1 = %c0_i32, %arg2 = %true, %arg3 = %0, %arg4 = %true) : (i32, i1, i32, i1) -> (i1, i32, i32) {
      %3 = arith.cmpi slt, %arg1, %c10_i32 : i32
      %4 = arith.andi %3, %arg4 : i1
      scf.condition(%4) %arg2, %arg3, %arg1 : i1, i32, i32
    } do {
    ^bb0(%arg1: i1, %arg2: i32, %arg3: i32):
      %3 = arith.index_cast %arg3 : i32 to index
      %4 = memref.load %arg0[%3] : memref<10xi32>
      %5 = arith.cmpi ne, %4, %c-1_i32 : i32
      %6:4 = scf.if %5 -> (i1, i32, i1, i32) {
        %7 = memref.load %arg0[%3] : memref<10xi32>
        %8 = arith.cmpi eq, %7, %c0_i32 : i32
        %9 = arith.cmpi ne, %7, %c0_i32 : i32
        %10 = arith.andi %9, %arg1 : i1
        %11 = arith.select %8, %c1_i32, %arg2 : i32
        %12 = scf.if %9 -> (i32) {
          %13 = memref.load %arg0[%3] : memref<10xi32>
          %14 = arith.addi %13, %c1_i32 : i32
          memref.store %14, %arg0[%3] : memref<10xi32>
          %15 = arith.addi %arg3, %c1_i32 : i32
          scf.yield %15 : i32
        } else {
          scf.yield %arg3 : i32
        }
        scf.yield %10, %11, %9, %12 : i1, i32, i1, i32
      } else {
        scf.yield %arg1, %arg2, %false, %arg3 : i1, i32, i1, i32
      }
      scf.yield %6#3, %6#0, %6#1, %6#2 : i32, i1, i32, i1
    }
    %2 = arith.select %1#0, %c2_i32, %1#1 : i32
    return %2 : i32
  }
}
