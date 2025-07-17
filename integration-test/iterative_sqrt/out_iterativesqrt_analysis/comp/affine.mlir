module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @iterative_sqrt(%arg0: i32) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %true = arith.constant true
    %c-1_i32 = arith.constant -1 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = llvm.mlir.undef : i32
    %1:4 = scf.while (%arg1 = %arg0, %arg2 = %c0_i32, %arg3 = %true, %arg4 = %0, %arg5 = %true) : (i32, i32, i1, i32, i1) -> (i32, i1, i32, i32) {
      %3 = arith.cmpi sle, %arg2, %arg1 : i32
      %4 = arith.andi %3, %arg5 : i1
      scf.condition(%4) %arg1, %arg3, %arg4, %arg2 : i32, i1, i32, i32
    } do {
    ^bb0(%arg1: i32, %arg2: i1, %arg3: i32, %arg4: i32):
      %3 = arith.addi %arg4, %arg1 : i32
      %4 = arith.shrsi %3, %c1_i32 : i32
      %5 = arith.muli %4, %4 : i32
      %6 = arith.cmpi eq, %5, %arg0 : i32
      %7 = arith.cmpi ne, %5, %arg0 : i32
      %8 = arith.andi %7, %arg2 : i1
      %9 = arith.select %6, %4, %arg3 : i32
      %10 = arith.cmpi slt, %5, %arg0 : i32
      %11:2 = scf.if %7 -> (i32, i32) {
        %12:2 = scf.if %10 -> (i32, i32) {
          %13 = arith.addi %4, %c1_i32 : i32
          scf.yield %arg1, %13 : i32, i32
        } else {
          %13 = arith.addi %4, %c-1_i32 : i32
          scf.yield %13, %arg4 : i32, i32
        }
        scf.yield %12#0, %12#1 : i32, i32
      } else {
        scf.yield %arg1, %arg4 : i32, i32
      }
      scf.yield %11#0, %11#1, %8, %9, %7 : i32, i32, i1, i32, i1
    }
    %2 = arith.select %1#1, %1#0, %1#2 : i32
    return %2 : i32
  }
}
