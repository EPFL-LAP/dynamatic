module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @iterative_sqrt(%arg0: i32) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %true = arith.constant true
    %c1_i32 = arith.constant 1 : i32
    %c-1_i32 = arith.constant -1 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = llvm.mlir.undef : i32
    %1:4 = scf.while (%arg1 = %arg0, %arg2 = %c0_i32, %arg3 = %true, %arg4 = %0, %arg5 = %true) : (i32, i32, i1, i32, i1) -> (i32, i1, i32, i32) {
      %3 = arith.cmpi sle, %arg2, %arg1 : i32
      %4 = arith.andi %3, %arg5 : i1
      %5 = arith.addi %arg2, %arg1 : i32
      %6 = arith.shrsi %5, %c1_i32 : i32
      %7 = arith.muli %6, %6 : i32
      %8 = arith.cmpi ne, %7, %arg0 : i32
      %9 = arith.andi %8, %arg3 : i1
      %10 = arith.xori %arg5, %true : i1
      %11 = arith.andi %arg5, %9 : i1
      %12 = arith.andi %10, %arg3 : i1
      %13 = arith.ori %11, %12 : i1
      %14 = arith.xori %4, %true : i1
      %15 = arith.andi %4, %13 : i1
      %16 = arith.andi %14, %arg3 : i1
      %17 = arith.ori %15, %16 : i1
      scf.condition(%4) %arg1, %17, %arg4, %arg2 : i32, i1, i32, i32
    } do {
    ^bb0(%arg1: i32, %arg2: i1, %arg3: i32, %arg4: i32):
      %3 = arith.addi %arg4, %arg1 : i32
      %4 = arith.shrsi %3, %c1_i32 : i32
      %5 = arith.muli %4, %4 : i32
      %6 = arith.cmpi ne, %5, %arg0 : i32
      %7 = arith.cmpi sle, %arg4, %arg1 : i32
      %8 = arith.cmpi sgt, %arg4, %arg1 : i32
      %9 = arith.cmpi eq, %5, %arg0 : i32
      %10 = arith.andi %7, %6 : i1
      %11 = arith.select %9, %4, %arg3 : i32
      %12 = arith.ori %10, %8 : i1
      %13:2 = scf.if %9 -> (i32, i32) {
        scf.yield %arg1, %arg4 : i32, i32
      } else {
        %14 = arith.cmpi slt, %5, %arg0 : i32
        %15:2 = scf.if %14 -> (i32, i32) {
          %16 = arith.addi %4, %c1_i32 : i32
          scf.yield %arg1, %16 : i32, i32
        } else {
          %16 = arith.addi %4, %c-1_i32 : i32
          scf.yield %16, %arg4 : i32, i32
        }
        scf.yield %15#0, %15#1 : i32, i32
      }
      scf.yield %13#0, %13#1, %arg2, %11, %12 : i32, i32, i1, i32, i1
    }
    %2 = arith.select %1#1, %1#0, %1#2 : i32
    return %2 : i32
  }
}
