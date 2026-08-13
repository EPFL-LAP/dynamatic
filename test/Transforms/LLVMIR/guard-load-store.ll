; RUN: opt -S -load-pass-plugin "%shlibdir/GuardLoadStore.so" -passes="guard-load-store" %s | FileCheck %s

; Checking that it does do the replacement for a function that is not main
; NOTE: Normal i32 alignment would be 4, with alignment 1 we can test that it is preserved by default
; CHECK-LABEL: define void @test(
define void @test(ptr noundef %var1) {
  ; CHECK-NOT: load i32
  ; CHECK: %[[RES:.*]] = call i32 @__dyn_guard.load.align1.ptr.to.i32(ptr %var1)
  %cond.in = load i32, ptr %var1, align 1
  
  ; CHECK-NOT: store i32
  ; CHECK: call void @__dyn_guard.store.align1.i32.ptr.to.void(i32 1, ptr %var1)
  store i32 1, ptr %var1, align 1

  ; CHECK: ret void
  ret void
}

; Checking that it doesn't do the replacement for main
; CHECK-LABEL: define i32 @main(
define i32 @main() {
  %x = alloca i32, align 1

  ; CHECK: load i32, ptr %x, align 1
  ; CHECK-NOT: call {{.*}} @__dyn_guard
  %v = load i32, ptr %x, align 1
  
  ; CHECK: store i32 %{{.*}}, ptr %x, align 1
  ; CHECK-NOT: call {{.*}} @__dyn_guard
  store i32 %v, ptr %x, align 1
  
  ; CHECK: ret i32 0
  ret i32 0
}

; Checks for ensuring the function calls are actually created
; CHECK-LABEL: define internal i32 @__dyn_guard.load.align1.ptr.to.i32(ptr %0)
; CHECK-SAME: #[[ATTR:[0-9]+]]
; CHECK-NEXT: entry:
; CHECK-NEXT: %[[LOADED:.*]] = load i32, ptr %0, align 1
; CHECK-NEXT: ret i32 %[[LOADED]]
; CHECK-NOT: call {{.*}} @__dyn_guard

; CHECK-LABEL: define internal void @__dyn_guard.store.align1.i32.ptr.to.void(i32 %0, ptr %1)
; CHECK-SAME: #[[ATTR]]
; CHECK-NEXT: entry:
; CHECK-NEXT: store i32 %0, ptr %1, align 1
; CHECK-NEXT: ret void
; CHECK-NOT: call {{.*}} @__dyn_guard
; CHECK: attributes #[[ATTR]] = { alwaysinline }
