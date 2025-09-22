; ModuleID = '/home/shundroid/dynamatic/integration-test/subdiag_fast/out_13/comp/clang_optnone_removed.ll'
source_filename = "/home/shundroid/dynamatic/integration-test/subdiag_fast/subdiag_fast.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline nounwind uwtable
define dso_local i32 @subdiag_fast(ptr noundef %d1, ptr noundef %d2, ptr noundef %e) #0 {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.inc
  %i.03 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %idxprom = zext i32 %i.03 to i64
  %arrayidx = getelementptr inbounds float, ptr %d1, i64 %idxprom
  %0 = load float, ptr %arrayidx, align 4
  %idxprom1 = zext i32 %i.03 to i64
  %arrayidx2 = getelementptr inbounds float, ptr %d2, i64 %idxprom1
  %1 = load float, ptr %arrayidx2, align 4
  %add = fadd float %0, %1
  %idxprom3 = zext i32 %i.03 to i64
  %arrayidx4 = getelementptr inbounds float, ptr %e, i64 %idxprom3
  %2 = load float, ptr %arrayidx4, align 4
  %mul = fmul float %add, 0x3F50624DE0000000
  %cmp5 = fcmp ugt float %2, %mul
  br i1 %cmp5, label %for.inc, label %for.end

for.inc:                                          ; preds = %for.body
  %inc = add nuw nsw i32 %i.03, 1
  %cmp = icmp ult i32 %inc, 999
  br i1 %cmp, label %for.body, label %for.end, !llvm.loop !6

for.end:                                          ; preds = %for.inc, %for.body
  %i.02 = phi i32 [ %i.03, %for.body ], [ %inc, %for.inc ]
  ret i32 %i.02
}

; Function Attrs: noinline nounwind uwtable
define dso_local i32 @main() #0 {
entry:
  %d1 = alloca [1000 x float], align 16
  %d2 = alloca [1000 x float], align 16
  %e = alloca [1000 x float], align 16
  call void @srand(i32 noundef 13) #2
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %j.01 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %conv = sitofp i32 %j.01 to float
  %idxprom = zext i32 %j.01 to i64
  %arrayidx = getelementptr inbounds [1000 x float], ptr %d1, i64 0, i64 %idxprom
  store float %conv, ptr %arrayidx, align 4
  %conv1 = sitofp i32 %j.01 to float
  %idxprom2 = zext i32 %j.01 to i64
  %arrayidx3 = getelementptr inbounds [1000 x float], ptr %d2, i64 0, i64 %idxprom2
  store float %conv1, ptr %arrayidx3, align 4
  %conv4 = sitofp i32 %j.01 to float
  %sub = fsub float 3.000000e+02, %conv4
  %mul = fmul float %sub, 0x3F50624DE0000000
  %idxprom5 = zext i32 %j.01 to i64
  %arrayidx6 = getelementptr inbounds [1000 x float], ptr %e, i64 0, i64 %idxprom5
  store float %mul, ptr %arrayidx6, align 4
  %inc = add nuw nsw i32 %j.01, 1
  %cmp = icmp ult i32 %inc, 1000
  br i1 %cmp, label %for.body, label %for.end, !llvm.loop !8

for.end:                                          ; preds = %for.body
  %call = call i32 @subdiag_fast(ptr noundef nonnull %d1, ptr noundef nonnull %d2, ptr noundef nonnull %e)
  ret i32 0
}

; Function Attrs: nounwind
declare void @srand(i32 noundef) #1

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
