module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @iterative_sqrt(%arg0: i32) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %true = arith.constant true
    %c1_i32 = arith.constant 1 : i32
    %c-1_i32 = arith.constant -1 : i32
    %c0_i32 = arith.constant 0 : i32
    %0:2 = scf.while (%arg1 = %arg0, %arg2 = %c0_i32, %arg3 = %true) : (i32, i32, i1) -> (i32, i32) {
      %1 = arith.cmpi sle, %arg2, %arg1 : i32
      %2 = arith.andi %1, %arg3 : i1
      scf.condition(%2) %arg1, %arg2 : i32, i32
    } do {
    ^bb0(%arg1: i32, %arg2: i32):
      %1 = arith.addi %arg2, %arg1 : i32
      %2 = arith.shrsi %1, %c1_i32 : i32
      %3 = arith.muli %2, %2 : i32
      %4 = arith.cmpi ne, %3, %arg0 : i32
      %5 = arith.cmpi sle, %arg2, %arg1 : i32
      %6 = arith.cmpi sgt, %arg2, %arg1 : i32
      %7 = arith.andi %5, %4 : i1
      %8 = arith.cmpi eq, %3, %arg0 : i32
      %9 = arith.ori %7, %6 : i1
      %10:2 = scf.if %8 -> (i32, i32) {
        scf.yield %2, %arg2 : i32, i32
      } else {
        %11 = arith.cmpi slt, %3, %arg0 : i32
        %12:2 = scf.if %11 -> (i32, i32) {
          %13 = arith.addi %2, %c1_i32 : i32
          scf.yield %arg1, %13 : i32, i32
        } else {
          %13 = arith.addi %2, %c-1_i32 : i32
          scf.yield %13, %arg2 : i32, i32
        }
        scf.yield %12#0, %12#1 : i32, i32
      }
      scf.yield %10#0, %10#1, %9 : i32, i32, i1
    }
    return %0#0 : i32
  }
}
