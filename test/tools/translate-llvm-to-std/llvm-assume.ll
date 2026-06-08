; RUN: %translate-llvm-to-std -o - | FileCheck %s

;--- test.ll

; CHECK: func.func @test(
define void @test(i16 %var0) {
  %t = trunc i16 %var0 to i1
  call void @llvm.assume(i1 %t)
  ret void
}

declare void @llvm.assume(i1 noundef)

;--- test.c

void test(short var0);
