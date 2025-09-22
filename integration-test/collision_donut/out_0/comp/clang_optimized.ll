; ModuleID = '/home/shundroid/dynamatic/integration-test/collision_donut/out_0/comp/clang_optnone_removed.ll'
source_filename = "/home/shundroid/dynamatic/integration-test/collision_donut/collision_donut.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline nounwind uwtable
define dso_local i32 @collision_donut(ptr noundef %x, ptr noundef %y) #0 {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.inc
  %i.04 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %idxprom = zext i32 %i.04 to i64
  %arrayidx = getelementptr inbounds i32, ptr %x, i64 %idxprom
  %0 = load i32, ptr %arrayidx, align 4
  %idxprom1 = zext i32 %i.04 to i64
  %arrayidx2 = getelementptr inbounds i32, ptr %y, i64 %idxprom1
  %1 = load i32, ptr %arrayidx2, align 4
  %mul = mul nsw i32 %0, %0
  %mul3 = mul nsw i32 %1, %1
  %add = add nuw nsw i32 %mul, %mul3
  %cmp4 = icmp ult i32 %add, 4
  br i1 %cmp4, label %for.end, label %if.end

if.end:                                           ; preds = %for.body
  %cmp5 = icmp ugt i32 %add, 19000
  br i1 %cmp5, label %for.end, label %for.inc

for.inc:                                          ; preds = %if.end
  %inc = add nuw nsw i32 %i.04, 1
  %cmp = icmp ult i32 %inc, 1000
  br i1 %cmp, label %for.body, label %for.end, !llvm.loop !6

for.end:                                          ; preds = %for.inc, %if.end, %for.body
  %i.03 = phi i32 [ %i.04, %for.body ], [ %i.04, %if.end ], [ %inc, %for.inc ]
  %err.0 = phi i32 [ -1, %for.body ], [ -2, %if.end ], [ 0, %for.inc ]
  %shl = shl nuw i32 %i.03, 1
  %and = and i32 %shl, %err.0
  ret i32 %and
}

; Function Attrs: noinline nounwind uwtable
define dso_local i32 @main() #0 {
entry:
  %x = alloca [1000 x i32], align 16
  %y = alloca [1000 x i32], align 16
  call void @srand(i32 noundef 13) #2
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %j.01 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %call = call i32 @rand() #2
  %rem = srem i32 %call, 100
  %idxprom = zext i32 %j.01 to i64
  %arrayidx = getelementptr inbounds [1000 x i32], ptr %x, i64 0, i64 %idxprom
  store i32 %rem, ptr %arrayidx, align 4
  %call1 = call i32 @rand() #2
  %rem2 = srem i32 %call1, 100
  %idxprom3 = zext i32 %j.01 to i64
  %arrayidx4 = getelementptr inbounds [1000 x i32], ptr %y, i64 0, i64 %idxprom3
  store i32 %rem2, ptr %arrayidx4, align 4
  %inc = add nuw nsw i32 %j.01, 1
  %cmp = icmp ult i32 %inc, 1000
  br i1 %cmp, label %for.body, label %for.end, !llvm.loop !8

for.end:                                          ; preds = %for.body
  %call6 = call i32 @collision_donut(ptr noundef nonnull %x, ptr noundef nonnull %y)
  ret i32 0
}

; Function Attrs: nounwind
declare void @srand(i32 noundef) #1

; Function Attrs: nounwind
declare i32 @rand() #1

attributes #0 = { noinline nounwind uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #2 = { nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 2}
!5 = !{!"clang version 18.0.0 (https://github.com/EPFL-LAP/llvm-project.git b06546b8f001a888f346b38b9f3ae0da11efbff2)"}
!6 = distinct !{!6, !7}
!7 = !{!"llvm.loop.mustprogress"}
!8 = distinct !{!8, !7}
