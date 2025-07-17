module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @multiple_exit(%arg0: memref<10xi32>, %arg1: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
    %true = arith.constant true
    %c1_i32 = arith.constant 1 : i32
    %false = arith.constant false
    %c-1_i32 = arith.constant -1 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = scf.while (%arg2 = %c0_i32, %arg3 = %true) : (i32, i1) -> i32 {
      %1 = arith.cmpi slt, %arg2, %arg1 : i32
      %2 = arith.andi %1, %arg3 : i1
      scf.condition(%2) %arg2 : i32
    } do {
    ^bb0(%arg2: i32):
      %1 = arith.index_cast %arg2 : i32 to index
      %2 = memref.load %arg0[%1] : memref<10xi32>
      %3 = arith.cmpi ne, %2, %c-1_i32 : i32
      %4:2 = scf.if %3 -> (i1, i32) {
        %5 = memref.load %arg0[%1] : memref<10xi32>
        %6 = arith.cmpi ne, %5, %c0_i32 : i32
        %7 = scf.if %6 -> (i32) {
          %8 = memref.load %arg0[%1] : memref<10xi32>
          %9 = arith.addi %8, %c1_i32 : i32
          memref.store %9, %arg0[%1] : memref<10xi32>
          %10 = arith.addi %arg2, %c1_i32 : i32
          scf.yield %10 : i32
        } else {
          scf.yield %arg2 : i32
        }
        scf.yield %6, %7 : i1, i32
      } else {
        scf.yield %false, %arg2 : i1, i32
      }
      scf.yield %4#1, %4#0 : i32, i1
    }
    return
  }
}
