; RUN: %translate-llvm-to-std -o - | FileCheck %s

;--- test.ll

; CHECK: func.func @test(
; CHECK-SAME: %[[VAR0:.*]]: i16
define i32 @test(i16 %var0) {
  ; CHECK: %[[T:.*]] = arith.trunci %[[VAR0]]
  ; CHECK: %[[T2:.*]] = arith.extui %[[T]]
  ; CHECK: return %[[T2]]

  %t = trunc i16 %var0 to i1
  %t2 = call i1 @llvm.bitreverse.i1(i1 %t)
  %t3 = zext i1 %t2 to i32
  ret i32 %t3
}

declare i1 @llvm.bitreverse.i1(i1 noundef)

;--- test.c

int test(short var0);
