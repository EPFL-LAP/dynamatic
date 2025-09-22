; ModuleID = '/home/shundroid/dynamatic/integration-test/golden_ratio/out_standard/comp/clang_optnone_removed.ll'
source_filename = "/home/shundroid/dynamatic/integration-test/golden_ratio/golden_ratio.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline nounwind uwtable
define dso_local float @golden_ratio(float noundef %x0) #0 {
entry:
  %original_x.01 = fdiv float 1.000000e+00, %x0
  br label %for.body

for.body:                                         ; preds = %entry, %for.inc
  %original_x.04 = phi float [ %original_x.01, %entry ], [ %original_x.0, %for.inc ]
  %i.03 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %x.02 = phi float [ %x0, %entry ], [ %add4, %for.inc ]
  br label %while.body

while.body:                                       ; preds = %while.body, %for.body
  %x.1 = phi float [ %x.02, %for.body ], [ %mul1, %while.body ]
  %mul = fmul float %x.1, %original_x.04
  %add = fadd float %x.1, %mul
  %mul1 = fmul float %add, 5.000000e-01
  %sub = fsub float %mul1, %x.1
  %0 = call float @llvm.fabs.f32(float %sub)
  %cmp2 = fcmp olt float %0, 0x3FB99999A0000000
  br i1 %cmp2, label %for.inc, label %while.body

for.inc:                                          ; preds = %while.body
  %x.1.lcssa = phi float [ %x.1, %while.body ]
  %add4 = fadd float %x.1.lcssa, 1.000000e+00
  %inc = add nuw nsw i32 %i.03, 1
  %original_x.0 = fdiv float 1.000000e+00, %add4
  %cmp = icmp ult i32 %inc, 100
  br i1 %cmp, label %for.body, label %for.end, !llvm.loop !6

for.end:                                          ; preds = %for.inc
  %x.0.lcssa = phi float [ %add4, %for.inc ]
  ret float %x.0.lcssa
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.fabs.f64(double) #1

; Function Attrs: noinline nounwind uwtable
define dso_local i32 @main() #0 {
entry:
  %call = call float @golden_ratio(float noundef 1.000000e+02)
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
