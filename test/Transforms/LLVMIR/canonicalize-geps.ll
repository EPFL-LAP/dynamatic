; RUN: opt -S -load-pass-plugin "%shlibdir/CanonicalizeGEPs.so" -passes="canonicalize-geps" %s | FileCheck %s

; Loads mat[i][j] of int mat[10][20], through a chain of two GEPs
; CHECK-LABEL: define i32 @collapse2D(
; CHECK-SAME:    ptr %[[MAT:.*]], i64 %[[I:.*]], i64 %[[J:.*]])
define i32 @collapse2D(ptr %mat, i64 %i, i64 %j) {
  ; CHECK:       %[[MUL:.*]] = mul i64 %[[I]], 20
  ; CHECK:       %[[IDX:.*]] = add i64 %[[MUL]], %[[J]]
  ; CHECK:       %[[ADDR:.*]] = getelementptr i32, ptr %[[MAT]], i64 %[[IDX]]
  ; CHECK-NOT:   getelementptr
  ; CHECK:       load i32, ptr %[[ADDR]]
  %row = getelementptr [10 x [20 x i32]], ptr %mat, i64 0, i64 %i
  %elem = getelementptr [20 x i32], ptr %row, i64 0, i64 %j
  %v = load i32, ptr %elem, align 4
  ret i32 %v
}

; Loads the element of int values[N] at byte offset %i - 1, split into two byte
; GEPs that are not whole elements on their own
; CHECK-LABEL: define i32 @collapseByteGEPs(
; CHECK-SAME:    ptr %[[VALUES:.*]], i64 %[[I:.*]])
define i32 @collapseByteGEPs(ptr %values, i64 %i) {
  ; CHECK:       %[[IDX:.*]] = add i64 %[[I]], -1
  ; CHECK:       %[[ADDR:.*]] = getelementptr i8, ptr %[[VALUES]], i64 %[[IDX]]
  ; CHECK-NOT:   getelementptr
  ; CHECK:       load i32, ptr %[[ADDR]]
  %p = getelementptr i8, ptr %values, i64 %i
  %q = getelementptr i8, ptr %p, i64 -1
  %v = load i32, ptr %q, align 4
  ret i32 %v
}

; Stores to values[i + 1] of int values[N], with the next element reached by a
; byte offset: the offset is a whole number of elements
; CHECK-LABEL: define void @collapseWholeByteOffset(
; CHECK-SAME:    ptr %[[VALUES:.*]], i32 %[[I:.*]], i32 %[[V:.*]])
define void @collapseWholeByteOffset(ptr %values, i32 %i, i32 %v) {
  ; CHECK:       %[[EXT:.*]] = sext i32 %[[I]] to i64
  ; CHECK:       %[[IDX:.*]] = add i64 %[[EXT]], 1
  ; CHECK:       %[[ADDR:.*]] = getelementptr i32, ptr %[[VALUES]], i64 %[[IDX]]
  ; CHECK-NOT:   getelementptr
  ; CHECK:       store i32 %[[V]], ptr %[[ADDR]]
  %elem = getelementptr i32, ptr %values, i32 %i
  %next = getelementptr i8, ptr %elem, i64 4
  store i32 %v, ptr %next, align 4
  ret void
}

; A GEP that is already canonical is kept
; CHECK-LABEL: define i32 @keepCanonical(
; CHECK-SAME:    ptr %[[VALUES:.*]], i64 %[[I:.*]])
define i32 @keepCanonical(ptr %values, i64 %i) {
  ; CHECK:       %[[ADDR:.*]] = getelementptr inbounds i32, ptr %[[VALUES]], i64 %[[I]]
  ; CHECK:       load i32, ptr %[[ADDR]]
  %p = getelementptr inbounds i32, ptr %values, i64 %i
  %v = load i32, ptr %p, align 4
  ret i32 %v
}
