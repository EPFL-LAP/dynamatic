; ModuleID = '/home/shundroid/dynamatic/integration-test/subdiag_fast/subdiag_fast.c'
source_filename = "/home/shundroid/dynamatic/integration-test/subdiag_fast/subdiag_fast.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline nounwind  uwtable
define dso_local i32 @subdiag_fast(ptr noundef %d1, ptr noundef %d2, ptr noundef %e) #0 {
entry:
  %d1.addr = alloca ptr, align 8
  %d2.addr = alloca ptr, align 8
  %e.addr = alloca ptr, align 8
  %i = alloca i32, align 4
  %dd = alloca float, align 4
  %x = alloca float, align 4
  store ptr %d1, ptr %d1.addr, align 8
  store ptr %d2, ptr %d2.addr, align 8
  store ptr %e, ptr %e.addr, align 8
  store i32 0, ptr %i, align 4
  store i32 0, ptr %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32, ptr %i, align 4
  %cmp = icmp slt i32 %0, 999
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %1 = load ptr, ptr %d1.addr, align 8
  %2 = load i32, ptr %i, align 4
  %idxprom = sext i32 %2 to i64
  %arrayidx = getelementptr inbounds float, ptr %1, i64 %idxprom
  %3 = load float, ptr %arrayidx, align 4
  %4 = load ptr, ptr %d2.addr, align 8
  %5 = load i32, ptr %i, align 4
  %idxprom1 = sext i32 %5 to i64
  %arrayidx2 = getelementptr inbounds float, ptr %4, i64 %idxprom1
  %6 = load float, ptr %arrayidx2, align 4
  %add = fadd float %3, %6
  store float %add, ptr %dd, align 4
  store float 0x3F50624DE0000000, ptr %x, align 4
  %7 = load ptr, ptr %e.addr, align 8
  %8 = load i32, ptr %i, align 4
  %idxprom3 = sext i32 %8 to i64
  %arrayidx4 = getelementptr inbounds float, ptr %7, i64 %idxprom3
  %9 = load float, ptr %arrayidx4, align 4
  %10 = load float, ptr %x, align 4
  %11 = load float, ptr %dd, align 4
  %mul = fmul float %10, %11
  %cmp5 = fcmp ole float %9, %mul
  br i1 %cmp5, label %if.then, label %if.end

if.then:                                          ; preds = %for.body
  br label %for.end

if.end:                                           ; preds = %for.body
  br label %for.inc

for.inc:                                          ; preds = %if.end
  %12 = load i32, ptr %i, align 4
  %inc = add nsw i32 %12, 1
  store i32 %inc, ptr %i, align 4
  br label %for.cond, !llvm.loop !6

for.end:                                          ; preds = %if.then, %for.cond
  %13 = load i32, ptr %i, align 4
  ret i32 %13
}

; Function Attrs: noinline nounwind  uwtable
define dso_local i32 @main() #0 {
entry:
  %retval = alloca i32, align 4
  %d1 = alloca [1000 x float], align 16
  %d2 = alloca [1000 x float], align 16
  %e = alloca [1000 x float], align 16
  %j = alloca i32, align 4
  store i32 0, ptr %retval, align 4
  call void @srand(i32 noundef 13) #2
  store i32 0, ptr %j, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32, ptr %j, align 4
  %cmp = icmp slt i32 %0, 1000
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %1 = load i32, ptr %j, align 4
  %conv = sitofp i32 %1 to float
  %2 = load i32, ptr %j, align 4
  %idxprom = sext i32 %2 to i64
  %arrayidx = getelementptr inbounds [1000 x float], ptr %d1, i64 0, i64 %idxprom
  store float %conv, ptr %arrayidx, align 4
  %3 = load i32, ptr %j, align 4
  %conv1 = sitofp i32 %3 to float
  %4 = load i32, ptr %j, align 4
  %idxprom2 = sext i32 %4 to i64
  %arrayidx3 = getelementptr inbounds [1000 x float], ptr %d2, i64 0, i64 %idxprom2
  store float %conv1, ptr %arrayidx3, align 4
  %5 = load i32, ptr %j, align 4
  %conv4 = sitofp i32 %5 to float
  %sub = fsub float 3.000000e+02, %conv4
  %mul = fmul float %sub, 0x3F50624DE0000000
  %6 = load i32, ptr %j, align 4
  %idxprom5 = sext i32 %6 to i64
  %arrayidx6 = getelementptr inbounds [1000 x float], ptr %e, i64 0, i64 %idxprom5
  store float %mul, ptr %arrayidx6, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %7 = load i32, ptr %j, align 4
  %inc = add nsw i32 %7, 1
  store i32 %inc, ptr %j, align 4
  br label %for.cond, !llvm.loop !8

for.end:                                          ; preds = %for.cond
  %arraydecay = getelementptr inbounds [1000 x float], ptr %d1, i64 0, i64 0
  %arraydecay7 = getelementptr inbounds [1000 x float], ptr %d2, i64 0, i64 0
  %arraydecay8 = getelementptr inbounds [1000 x float], ptr %e, i64 0, i64 0
  %call = call i32 @subdiag_fast(ptr noundef %arraydecay, ptr noundef %arraydecay7, ptr noundef %arraydecay8)
  ret i32 0
}

; Function Attrs: nounwind
declare void @srand(i32 noundef) #1

attributes #0 = { noinline nounwind  uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
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
