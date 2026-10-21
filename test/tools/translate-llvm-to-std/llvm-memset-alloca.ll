; RUN: %translate-llvm-to-std -o - | FileCheck %s

; COM: This test checks that a memset on a local array stores into the array

;--- test.ll

; CHECK: func.func @test(
define i8 @test(i8 %var0) {
entry:
; CHECK: %[[ARRAY:.*]] = memref.alloca() : memref<4xi8>
  %arr = alloca [4 x i8], align 1
; CHECK-COUNT-4: memref.store %{{.*}}, %[[ARRAY]][%{{.*}}] : memref<4xi8>
  call void @llvm.memset.p0.i64(ptr align 1 %arr, i8 0, i64 4, i1 false)
  %idx = getelementptr inbounds [4 x i8], ptr %arr, i64 0, i64 2
  %val = load i8, ptr %idx, align 1
  ret i8 %val
}

declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, i1 immarg)

;--- test.c

#include <stdint.h>
int8_t test(int8_t var0);
