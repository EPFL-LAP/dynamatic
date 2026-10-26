; RUN: opt -S -load-pass-plugin "%shlibdir/ArrayPartition.so" -passes="array-partition" -polly-process-unprofitable %s | FileCheck %s

; int arr[4][8] is partitioned on its second dimension (block, factor 2). Each
; access computes its address with a chain of two GEPs, one for the row and one
; for the column, as recent clang versions emit. The bank and the index in the
; bank must come from the column offset of the second GEP

@.str = private unnamed_addr constant [4 x i8] c"arr\00", align 1
@.str.1 = private unnamed_addr constant [6 x i8] c"block\00", align 1

; __dyn_array_partition(ptr <array name>, i32 <dimension>, i32 <factor>, ptr <style>)
declare void @__dyn_array_partition(ptr noundef, i32 noundef, i32 noundef, ptr noundef)

; CHECK-LABEL: define void @store(
; CHECK-SAME:    i64 %[[I:.*]], i64 %[[J:.*]], i32 %[[V:.*]])
define void @store(i64 %i, i64 %j, i32 %v) {
entry:
  ; CHECK-DAG:   %[[BANK0:.*]] = alloca [4 x [4 x i32]]
  ; CHECK-DAG:   %[[BANK1:.*]] = alloca [4 x [4 x i32]]
  %arr = alloca [4 x [8 x i32]], align 16
  call void @__dyn_array_partition(ptr noundef nonnull @.str, i32 noundef 2, i32 noundef 2, ptr noundef nonnull @.str.1)

  ; arr[i][j] = v
  ; The bank is selected by the column
  ; CHECK:       and i64 %[[J]], 4294967292
  ; Each bank is indexed with the row and the column
  ; CHECK-LABEL: partition.arr.store.0:
  ; CHECK-DAG:   getelementptr {{.*}}ptr %[[BANK0]], {{.*}}%[[I]]
  ; CHECK-DAG:   %[[J]]
  ; CHECK:       store i32 %[[V]]
  ; CHECK-LABEL: partition.arr.store.1:
  ; CHECK-DAG:   getelementptr {{.*}}ptr %[[BANK1]], {{.*}}%[[I]]
  ; CHECK-DAG:   %[[J]]
  ; CHECK:       store i32 %[[V]]
  %row = getelementptr inbounds [32 x i8], ptr %arr, i64 %i
  %col = getelementptr inbounds [4 x i8], ptr %row, i64 %j
  store i32 %v, ptr %col, align 4
  ret void
}

; CHECK-LABEL: define i32 @load(
; CHECK-SAME:    i64 %[[I:.*]], i64 %[[J:.*]])
define i32 @load(i64 %i, i64 %j) {
entry:
  ; CHECK-DAG:   %[[BANK0:.*]] = alloca [4 x [4 x i32]]
  ; CHECK-DAG:   %[[BANK1:.*]] = alloca [4 x [4 x i32]]
  %arr = alloca [4 x [8 x i32]], align 16
  call void @__dyn_array_partition(ptr noundef nonnull @.str, i32 noundef 2, i32 noundef 2, ptr noundef nonnull @.str.1)

  ; return arr[i][j]
  ; The bank is selected by the column
  ; CHECK:       and i64 %[[J]], 4294967292
  ; Each bank is indexed with the row and the column
  ; CHECK-LABEL: partition.arr.load.0:
  ; CHECK-DAG:   getelementptr {{.*}}ptr %[[BANK0]], {{.*}}%[[I]]
  ; CHECK-DAG:   %[[J]]
  ; CHECK:       load i32
  ; CHECK-LABEL: partition.arr.load.1:
  ; CHECK-DAG:   getelementptr {{.*}}ptr %[[BANK1]], {{.*}}%[[I]]
  ; CHECK-DAG:   %[[J]]
  ; CHECK:       load i32
  %row = getelementptr inbounds [32 x i8], ptr %arr, i64 %i
  %col = getelementptr inbounds [4 x i8], ptr %row, i64 %j
  %r = load i32, ptr %col, align 4
  ret i32 %r
}
