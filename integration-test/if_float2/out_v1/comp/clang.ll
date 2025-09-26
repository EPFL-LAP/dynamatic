; ModuleID = '/home/shundroid/dynamatic/integration-test/if_float2/if_float2.c'
source_filename = "/home/shundroid/dynamatic/integration-test/if_float2/if_float2.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline nounwind optnone uwtable
define dso_local float @if_float2(float noundef %x0, ptr noundef %a, ptr noundef %minus_trace) #0 {
entry:
  %x0.addr = alloca float, align 4
  %a.addr = alloca ptr, align 8
  %minus_trace.addr = alloca ptr, align 8
  %x = alloca float, align 4
  %i = alloca i32, align 4
  store float %x0, ptr %x0.addr, align 4
  store ptr %a, ptr %a.addr, align 8
  store ptr %minus_trace, ptr %minus_trace.addr, align 8
  %0 = load float, ptr %x0.addr, align 4
  store float %0, ptr %x, align 4
  store i32 0, ptr %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %1 = load i32, ptr %i, align 4
  %cmp = icmp slt i32 %1, 100
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %2 = load ptr, ptr %a.addr, align 8
  %3 = load i32, ptr %i, align 4
  %idxprom = sext i32 %3 to i64
  %arrayidx = getelementptr inbounds float, ptr %2, i64 %idxprom
  %4 = load float, ptr %arrayidx, align 4
  %5 = load float, ptr %x, align 4
  %mul = fmul float %4, %5
  %6 = load float, ptr %x, align 4
  %mul1 = fmul float 0xBFE3333340000000, %6
  %add = fadd float %mul, %mul1
  %cmp2 = fcmp ole float %add, 0.000000e+00
  br i1 %cmp2, label %if.then, label %if.else

if.then:                                          ; preds = %for.body
  %7 = load float, ptr %x, align 4
  %add3 = fadd float %7, 3.000000e+00
  store float %add3, ptr %x, align 4
  br label %if.end

if.else:                                          ; preds = %for.body
  %8 = load float, ptr %x, align 4
  %9 = load ptr, ptr %minus_trace.addr, align 8
  %10 = load i32, ptr %i, align 4
  %idxprom4 = sext i32 %10 to i64
  %arrayidx5 = getelementptr inbounds float, ptr %9, i64 %idxprom4
  store float %8, ptr %arrayidx5, align 4
  %11 = load float, ptr %x, align 4
  %add6 = fadd float %11, 1.000000e+00
  store float %add6, ptr %x, align 4
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  %12 = load float, ptr %x, align 4
  %div = fdiv float 1.000000e+00, %12
  store float %div, ptr %x, align 4
  br label %for.inc

for.inc:                                          ; preds = %if.end
  %13 = load i32, ptr %i, align 4
  %inc = add nsw i32 %13, 1
  store i32 %inc, ptr %i, align 4
  br label %for.cond, !llvm.loop !6

for.end:                                          ; preds = %for.cond
  %14 = load float, ptr %x, align 4
  ret float %14
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @main() #0 {
entry:
  %retval = alloca i32, align 4
  %x0 = alloca float, align 4
  %a = alloca [100 x float], align 16
  %minus_trace = alloca [100 x float], align 16
  %j = alloca i32, align 4
  store i32 0, ptr %retval, align 4
  store float 1.000000e+02, ptr %x0, align 4
  call void @srand(i32 noundef 13) #2
  store i32 0, ptr %j, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32, ptr %j, align 4
  %cmp = icmp slt i32 %0, 100
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %call = call i32 @rand() #2
  %rem = srem i32 %call, 100
  %conv = sitofp i32 %rem to float
  %div = fdiv float %conv, 1.000000e+02
  %1 = load i32, ptr %j, align 4
  %idxprom = sext i32 %1 to i64
  %arrayidx = getelementptr inbounds [100 x float], ptr %a, i64 0, i64 %idxprom
  store float %div, ptr %arrayidx, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %2 = load i32, ptr %j, align 4
  %inc = add nsw i32 %2, 1
  store i32 %inc, ptr %j, align 4
  br label %for.cond, !llvm.loop !8

for.end:                                          ; preds = %for.cond
  %3 = load float, ptr %x0, align 4
  %arraydecay = getelementptr inbounds [100 x float], ptr %a, i64 0, i64 0
  %arraydecay1 = getelementptr inbounds [100 x float], ptr %minus_trace, i64 0, i64 0
  %call2 = call float @if_float2(float noundef %3, ptr noundef %arraydecay, ptr noundef %arraydecay1)
  ret i32 0
}

; Function Attrs: nounwind
declare void @srand(i32 noundef) #1

; Function Attrs: nounwind
declare i32 @rand() #1

attributes #0 = { noinline nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
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
