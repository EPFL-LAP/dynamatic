; ModuleID = '/home/shundroid/dynamatic/integration-test/golden_ratio/golden_ratio.c'
source_filename = "/home/shundroid/dynamatic/integration-test/golden_ratio/golden_ratio.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline nounwind optnone uwtable
define dso_local float @golden_ratio(float noundef %x0) #0 {
entry:
  %x0.addr = alloca float, align 4
  %x = alloca float, align 4
  %original_x = alloca float, align 4
  %i = alloca i32, align 4
  %next_x = alloca float, align 4
  store float %x0, ptr %x0.addr, align 4
  %0 = load float, ptr %x0.addr, align 4
  store float %0, ptr %x, align 4
  %1 = load float, ptr %x, align 4
  %div = fdiv float 1.000000e+00, %1
  store float %div, ptr %original_x, align 4
  store i32 0, ptr %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %2 = load i32, ptr %i, align 4
  %cmp = icmp slt i32 %2, 100
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  br label %while.body

while.body:                                       ; preds = %for.body, %if.end
  %3 = load float, ptr %x, align 4
  %4 = load float, ptr %x, align 4
  %5 = load float, ptr %original_x, align 4
  %mul = fmul float %4, %5
  %add = fadd float %3, %mul
  %mul1 = fmul float 5.000000e-01, %add
  store float %mul1, ptr %next_x, align 4
  %6 = load float, ptr %next_x, align 4
  %7 = load float, ptr %x, align 4
  %sub = fsub float %6, %7
  %conv = fpext float %sub to double
  %8 = call double @llvm.fabs.f64(double %conv)
  %cmp2 = fcmp olt double %8, 0x3FB99999A0000000
  br i1 %cmp2, label %if.then, label %if.end

if.then:                                          ; preds = %while.body
  br label %while.end

if.end:                                           ; preds = %while.body
  %9 = load float, ptr %next_x, align 4
  store float %9, ptr %x, align 4
  br label %while.body

while.end:                                        ; preds = %if.then
  %10 = load float, ptr %x, align 4
  %add4 = fadd float %10, 1.000000e+00
  store float %add4, ptr %x, align 4
  %11 = load float, ptr %x, align 4
  %div5 = fdiv float 1.000000e+00, %11
  store float %div5, ptr %original_x, align 4
  br label %for.inc

for.inc:                                          ; preds = %while.end
  %12 = load i32, ptr %i, align 4
  %inc = add nsw i32 %12, 1
  store i32 %inc, ptr %i, align 4
  br label %for.cond, !llvm.loop !6

for.end:                                          ; preds = %for.cond
  %13 = load float, ptr %x, align 4
  ret float %13
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.fabs.f64(double) #1

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @main() #0 {
entry:
  %retval = alloca i32, align 4
  %x0 = alloca float, align 4
  store i32 0, ptr %retval, align 4
  store float 1.000000e+02, ptr %x0, align 4
  %0 = load float, ptr %x0, align 4
  %call = call float @golden_ratio(float noundef %0)
  ret i32 0
}

attributes #0 = { noinline nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

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
