; ModuleID = '/home/shundroid/dynamatic/integration-test/bisection/out_3/comp/clang_optnone_removed.ll'
source_filename = "/home/shundroid/dynamatic/integration-test/bisection/bisection.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline nounwind uwtable
define dso_local float @bisection(float noundef %a, float noundef %b, float noundef %tol) #0 {
entry:
  %mul = fmul float %a, %a
  %sub = fadd float %mul, -2.000000e+00
  br label %for.body

for.body:                                         ; preds = %entry, %if.end
  %a.addr.04 = phi float [ %a, %entry ], [ %cond20, %if.end ]
  %b.addr.03 = phi float [ %b, %entry ], [ %mul1.b.addr.03, %if.end ]
  %i.02 = phi i32 [ 0, %entry ], [ %inc, %if.end ]
  %fa.01 = phi float [ %sub, %entry ], [ %cond26, %if.end ]
  %add = fadd float %a.addr.04, %b.addr.03
  %mul1 = fmul float %add, 5.000000e-01
  %mul2 = fmul float %mul1, %mul1
  %sub3 = fadd float %mul2, -2.000000e+00
  %0 = call float @llvm.fabs.f32(float %sub3)
  %cmp5 = fcmp olt float %0, %tol
  br i1 %cmp5, label %return, label %lor.lhs.false

lor.lhs.false:                                    ; preds = %for.body
  %sub7 = fsub float %b.addr.03, %a.addr.04
  %div = fmul float %sub7, 5.000000e-01
  %cmp8 = fcmp olt float %div, %tol
  br i1 %cmp8, label %return, label %if.end

if.end:                                           ; preds = %lor.lhs.false
  %mul10 = fmul float %fa.01, %sub3
  %cmp11 = fcmp olt float %mul10, 0.000000e+00
  %mul1.b.addr.03 = select i1 %cmp11, float %mul1, float %b.addr.03
  %cond20 = select i1 %cmp11, float %a.addr.04, float %mul1
  %cond26 = select i1 %cmp11, float %fa.01, float %sub3
  %inc = add nuw nsw i32 %i.02, 1
  %cmp = icmp ult i32 %inc, 100
  br i1 %cmp, label %for.body, label %return, !llvm.loop !6

return:                                           ; preds = %if.end, %for.body, %lor.lhs.false
  %retval.0 = phi float [ %mul1, %lor.lhs.false ], [ %mul1, %for.body ], [ 0.000000e+00, %if.end ]
  ret float %retval.0
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.fabs.f64(double) #1

; Function Attrs: noinline nounwind uwtable
define dso_local i32 @main() #0 {
entry:
  %call = call float @bisection(float noundef 0.000000e+00, float noundef 1.000000e+02, float noundef 0x3DDB7CDFE0000000)
  ret i32 0
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.fabs.f32(float) #1

attributes #0 = { noinline nounwind uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
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
