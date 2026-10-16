; RUN: opt -S -load-pass-plugin "%shlibdir/CanonicalizeGEPs.so" -passes="canonicalize-geps" %s | FileCheck %s

; Loads the element of int values[N] at byte offset %i - 1, split into two byte
; GEPs that are not whole elements on their own
; CHECK-LABEL: define i32 @foldByteGEPs(
; CHECK-SAME:    ptr %[[VALUES:.*]], i64 %[[I:.*]])
define i32 @foldByteGEPs(ptr %values, i64 %i) {
  ; CHECK:       %[[SUM:.*]] = add i64 %[[I]], -1
  ; CHECK:       %[[ADDR:.*]] = getelementptr i8, ptr %[[VALUES]], i64 %[[SUM]]
  ; CHECK:       load i32, ptr %[[ADDR]]
  %p = getelementptr i8, ptr %values, i64 %i
  %q = getelementptr i8, ptr %p, i64 -1
  %v = load i32, ptr %q, align 4
  ret i32 %v
}
