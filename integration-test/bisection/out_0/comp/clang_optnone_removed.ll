; ModuleID = '/home/shundroid/dynamatic/integration-test/bisection/bisection.c'
source_filename = "/home/shundroid/dynamatic/integration-test/bisection/bisection.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline nounwind  uwtable
define dso_local float @bisection(float noundef %a, float noundef %b, float noundef %tol) #0 {
entry:
  %retval = alloca float, align 4
  %a.addr = alloca float, align 4
  %b.addr = alloca float, align 4
  %tol.addr = alloca float, align 4
  %fa = alloca float, align 4
  %i = alloca i32, align 4
  %c = alloca float, align 4
  %fc = alloca float, align 4
  %cond = alloca i8, align 1
  store float %a, ptr %a.addr, align 4
  store float %b, ptr %b.addr, align 4
  store float %tol, ptr %tol.addr, align 4
  %0 = load float, ptr %a.addr, align 4
  %1 = load float, ptr %a.addr, align 4
  %mul = fmul float %0, %1
  %sub = fsub float %mul, 2.000000e+00
  store float %sub, ptr %fa, align 4
  store i32 0, ptr %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %2 = load i32, ptr %i, align 4
  %cmp = icmp slt i32 %2, 100
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %3 = load float, ptr %a.addr, align 4
  %4 = load float, ptr %b.addr, align 4
  %add = fadd float %3, %4
  %mul1 = fmul float 5.000000e-01, %add
  store float %mul1, ptr %c, align 4
  %5 = load float, ptr %c, align 4
  %6 = load float, ptr %c, align 4
  %mul2 = fmul float %5, %6
  %sub3 = fsub float %mul2, 2.000000e+00
  store float %sub3, ptr %fc, align 4
  %7 = load float, ptr %fc, align 4
  %conv = fpext float %7 to double
  %8 = call double @llvm.fabs.f64(double %conv)
  %9 = load float, ptr %tol.addr, align 4
  %conv4 = fpext float %9 to double
  %cmp5 = fcmp olt double %8, %conv4
  br i1 %cmp5, label %if.then, label %lor.lhs.false

lor.lhs.false:                                    ; preds = %for.body
  %10 = load float, ptr %b.addr, align 4
  %11 = load float, ptr %a.addr, align 4
  %sub7 = fsub float %10, %11
  %div = fdiv float %sub7, 2.000000e+00
  %12 = load float, ptr %tol.addr, align 4
  %cmp8 = fcmp olt float %div, %12
  br i1 %cmp8, label %if.then, label %if.end

if.then:                                          ; preds = %lor.lhs.false, %for.body
  %13 = load float, ptr %c, align 4
  store float %13, ptr %retval, align 4
  br label %return

if.end:                                           ; preds = %lor.lhs.false
  %14 = load float, ptr %fa, align 4
  %15 = load float, ptr %fc, align 4
  %mul10 = fmul float %14, %15
  %cmp11 = fcmp olt float %mul10, 0.000000e+00
  %frombool = zext i1 %cmp11 to i8
  store i8 %frombool, ptr %cond, align 1
  %16 = load i8, ptr %cond, align 1
  %tobool = trunc i8 %16 to i1
  br i1 %tobool, label %cond.true, label %cond.false

cond.true:                                        ; preds = %if.end
  %17 = load float, ptr %c, align 4
  br label %cond.end

cond.false:                                       ; preds = %if.end
  %18 = load float, ptr %b.addr, align 4
  br label %cond.end

cond.end:                                         ; preds = %cond.false, %cond.true
  %cond14 = phi float [ %17, %cond.true ], [ %18, %cond.false ]
  store float %cond14, ptr %b.addr, align 4
  %19 = load i8, ptr %cond, align 1
  %tobool15 = trunc i8 %19 to i1
  br i1 %tobool15, label %cond.true17, label %cond.false18

cond.true17:                                      ; preds = %cond.end
  %20 = load float, ptr %a.addr, align 4
  br label %cond.end19

cond.false18:                                     ; preds = %cond.end
  %21 = load float, ptr %c, align 4
  br label %cond.end19

cond.end19:                                       ; preds = %cond.false18, %cond.true17
  %cond20 = phi float [ %20, %cond.true17 ], [ %21, %cond.false18 ]
  store float %cond20, ptr %a.addr, align 4
  %22 = load i8, ptr %cond, align 1
  %tobool21 = trunc i8 %22 to i1
  br i1 %tobool21, label %cond.true23, label %cond.false24

cond.true23:                                      ; preds = %cond.end19
  %23 = load float, ptr %fa, align 4
  br label %cond.end25

cond.false24:                                     ; preds = %cond.end19
  %24 = load float, ptr %fc, align 4
  br label %cond.end25

cond.end25:                                       ; preds = %cond.false24, %cond.true23
  %cond26 = phi float [ %23, %cond.true23 ], [ %24, %cond.false24 ]
  store float %cond26, ptr %fa, align 4
  br label %for.inc

for.inc:                                          ; preds = %cond.end25
  %25 = load i32, ptr %i, align 4
  %inc = add nsw i32 %25, 1
  store i32 %inc, ptr %i, align 4
  br label %for.cond, !llvm.loop !6

for.end:                                          ; preds = %for.cond
  store float 0.000000e+00, ptr %retval, align 4
  br label %return

return:                                           ; preds = %for.end, %if.then
  %26 = load float, ptr %retval, align 4
  ret float %26
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.fabs.f64(double) #1

; Function Attrs: noinline nounwind  uwtable
define dso_local i32 @main() #0 {
entry:
  %retval = alloca i32, align 4
  %a = alloca float, align 4
  %b = alloca float, align 4
  %tol = alloca float, align 4
  store i32 0, ptr %retval, align 4
  store float 0.000000e+00, ptr %a, align 4
  store float 1.000000e+02, ptr %b, align 4
  store float 0x3DDB7CDFE0000000, ptr %tol, align 4
  %0 = load float, ptr %a, align 4
  %1 = load float, ptr %b, align 4
  %2 = load float, ptr %tol, align 4
  %call = call float @bisection(float noundef %0, float noundef %1, float noundef %2)
  ret i32 0
}

attributes #0 = { noinline nounwind  uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
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
