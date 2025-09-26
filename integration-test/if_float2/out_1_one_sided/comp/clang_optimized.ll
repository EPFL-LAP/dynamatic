; ModuleID = '/home/shundroid/dynamatic/integration-test/if_float2/out_1_one_sided/comp/clang_optnone_removed.ll'
source_filename = "/home/shundroid/dynamatic/integration-test/if_float2/if_float2.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline nounwind uwtable
define dso_local float @if_float2(float noundef %x0, ptr noundef %a, ptr noundef %minus_trace) #0 {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.inc
  %i.02 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %x.01 = phi float [ %x0, %entry ], [ %div, %for.inc ]
  %idxprom = zext i32 %i.02 to i64
  %arrayidx = getelementptr inbounds float, ptr %a, i64 %idxprom
  %0 = load float, ptr %arrayidx, align 4
  %mul = fmul float %0, %x.01
  %mul1 = fmul float %x.01, 0xBFECCCCCC0000000
  %add = fadd float %mul, %mul1
  %cmp2 = fcmp ugt float %add, 0.000000e+00
  br i1 %cmp2, label %if.else, label %if.then

if.then:                                          ; preds = %for.body
  %add3 = fadd float %x.01, 3.000000e+00
  br label %for.inc

if.else:                                          ; preds = %for.body
  %idxprom4 = zext i32 %i.02 to i64
  %arrayidx5 = getelementptr inbounds float, ptr %minus_trace, i64 %idxprom4
  store float %x.01, ptr %arrayidx5, align 4
  %add6 = fadd float %x.01, 1.000000e+00
  br label %for.inc

for.inc:                                          ; preds = %if.then, %if.else
  %x.1 = phi float [ %add3, %if.then ], [ %add6, %if.else ]
  %div = fdiv float 1.000000e+00, %x.1
  %inc = add nuw nsw i32 %i.02, 1
  %cmp = icmp ult i32 %inc, 100
  br i1 %cmp, label %for.body, label %for.end, !llvm.loop !6

for.end:                                          ; preds = %for.inc
  %x.0.lcssa = phi float [ %div, %for.inc ]
  ret float %x.0.lcssa
}

; Function Attrs: noinline nounwind uwtable
define dso_local i32 @main() #0 {
entry:
  %a = alloca [100 x float], align 16
  %minus_trace = alloca [100 x float], align 16
  call void @srand(i32 noundef 13) #2
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %j.01 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %call = call i32 @rand() #2
  %rem = srem i32 %call, 100
  %conv = sitofp i32 %rem to float
  %div = fdiv float %conv, 1.000000e+02
  %idxprom = zext i32 %j.01 to i64
  %arrayidx = getelementptr inbounds [100 x float], ptr %a, i64 0, i64 %idxprom
  store float %div, ptr %arrayidx, align 4
  %inc = add nuw nsw i32 %j.01, 1
  %cmp = icmp ult i32 %inc, 100
  br i1 %cmp, label %for.body, label %for.end, !llvm.loop !8

for.end:                                          ; preds = %for.body
  %call2 = call float @if_float2(float noundef 1.000000e+02, ptr noundef nonnull %a, ptr noundef nonnull %minus_trace)
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
