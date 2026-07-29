; RUN: %translate-llvm-to-std -o - | FileCheck %s

; COM: This test checks if our importer can handle loading a value from a memory location specified by a select inst

;--- test.ll

; CHECK: func.func @test(
define i8 @test(ptr noundef %var1) #0 {
entry:
  %pred = icmp eq i8 1, 0
  %arrayidx = getelementptr inbounds i8, ptr %var1, i64 1
  %merged = select i1 %pred, ptr %var1, ptr %arrayidx
  %cond.in = load i8, ptr %merged, align 1
  ret i8 %cond.in
}

;--- test.c

#include <stdint.h>
int test(uint8_t var1[4]);
