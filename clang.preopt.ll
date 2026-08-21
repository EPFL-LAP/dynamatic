; ModuleID = 'integration-test/matrix/out/comp/clang.opt.ll'
source_filename = "/local/home/crizzi/CI_mapbuf/dynamatic/integration-test/matrix/matrix.c"

; Function Attrs: nounwind uwtable
define dso_local void @matrix(ptr noundef %inA, ptr noundef %inB, ptr noundef %outC) #0 {
entry:
  br label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %for.inc20, %entry
  %i.04 = phi i32 [ 0, %entry ], [ %inc21, %for.inc20 ]
  %idxprom = zext i32 %i.04 to i64
  br label %for.cond4.preheader

for.cond4.preheader:                              ; preds = %for.end, %for.cond1.preheader
  %j.03 = phi i32 [ 0, %for.cond1.preheader ], [ %inc18, %for.end ]
  br label %for.inc

for.inc:                                          ; preds = %for.inc, %for.cond4.preheader
  %k.02 = phi i32 [ 0, %for.cond4.preheader ], [ %inc, %for.inc ]
  %sumMult.01 = phi i32 [ 0, %for.cond4.preheader ], [ %add, %for.inc ]
  %idxprom7 = zext i32 %k.02 to i64
  %arrayidx8 = getelementptr inbounds [30 x i32], ptr %inA, i64 %idxprom, i64 %idxprom7
  %0 = load i32, ptr %arrayidx8, align 4
  %idxprom11 = zext i32 %j.03 to i64
  %arrayidx12 = getelementptr inbounds [30 x i32], ptr %inB, i64 %idxprom7, i64 %idxprom11
  %1 = load i32, ptr %arrayidx12, align 4
  %mul = mul nsw i32 %1, %0
  %add = add nsw i32 %mul, %sumMult.01
  %inc = add i32 %k.02, 1
  %cmp5 = icmp ult i32 %inc, 30
  br i1 %cmp5, label %for.inc, label %for.end, !llvm.loop !6

for.end:                                          ; preds = %for.inc
  %arrayidx16 = getelementptr inbounds [30 x i32], ptr %outC, i64 %idxprom, i64 %idxprom11
  store i32 %add, ptr %arrayidx16, align 4
  %inc18 = add i32 %j.03, 1
  %cmp2 = icmp ult i32 %inc18, 30
  br i1 %cmp2, label %for.cond4.preheader, label %for.inc20, !llvm.loop !9

for.inc20:                                        ; preds = %for.end
  %inc21 = add i32 %i.04, 1
  %cmp = icmp ult i32 %inc21, 30
  br i1 %cmp, label %for.cond1.preheader, label %for.end22, !llvm.loop !10

for.end22:                                        ; preds = %for.inc20
  ret void
}

; Function Attrs: nounwind uwtable
define dso_local i32 @main() #0 {
entry:
  %inA = alloca [30 x [30 x i32]], align 16
  %inB = alloca [30 x [30 x i32]], align 16
  %outC = alloca [30 x [30 x i32]], align 16
  tail call void @srand(i32 noundef 13) #2
  br label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %for.inc6, %entry
  %y.02 = phi i32 [ 0, %entry ], [ %inc7, %for.inc6 ]
  br label %for.body3

for.body3:                                        ; preds = %for.body3, %for.cond1.preheader
  %x.01 = phi i32 [ 0, %for.cond1.preheader ], [ %inc, %for.body3 ]
  %call = tail call i32 @rand() #2
  %rem = srem i32 %call, 10
  %idxprom = zext i32 %y.02 to i64
  %idxprom4 = zext i32 %x.01 to i64
  %arrayidx5 = getelementptr inbounds [30 x [30 x i32]], ptr %inA, i64 0, i64 %idxprom, i64 %idxprom4
  store i32 %rem, ptr %arrayidx5, align 4
  %inc = add nuw nsw i32 %x.01, 1
  %cmp2 = icmp ult i32 %x.01, 29
  br i1 %cmp2, label %for.body3, label %for.inc6, !llvm.loop !11

for.inc6:                                         ; preds = %for.body3
  %inc7 = add nuw nsw i32 %y.02, 1
  %cmp = icmp ult i32 %y.02, 29
  br i1 %cmp, label %for.cond1.preheader, label %for.cond14.preheader.preheader, !llvm.loop !12

for.cond14.preheader.preheader:                   ; preds = %for.inc6
  br label %for.cond14.preheader

for.cond14.preheader:                             ; preds = %for.cond14.preheader.preheader, %for.inc26
  %y9.04 = phi i32 [ %inc27, %for.inc26 ], [ 0, %for.cond14.preheader.preheader ]
  br label %for.body16

for.body16:                                       ; preds = %for.body16, %for.cond14.preheader
  %x13.03 = phi i32 [ 0, %for.cond14.preheader ], [ %inc24, %for.body16 ]
  %call17 = tail call i32 @rand() #2
  %rem18 = srem i32 %call17, 10
  %idxprom19 = zext i32 %y9.04 to i64
  %idxprom21 = zext i32 %x13.03 to i64
  %arrayidx22 = getelementptr inbounds [30 x [30 x i32]], ptr %inB, i64 0, i64 %idxprom19, i64 %idxprom21
  store i32 %rem18, ptr %arrayidx22, align 4
  %inc24 = add nuw nsw i32 %x13.03, 1
  %cmp15 = icmp ult i32 %x13.03, 29
  br i1 %cmp15, label %for.body16, label %for.inc26, !llvm.loop !13

for.inc26:                                        ; preds = %for.body16
  %inc27 = add nuw nsw i32 %y9.04, 1
  %cmp11 = icmp ult i32 %y9.04, 29
  br i1 %cmp11, label %for.cond14.preheader, label %for.end28, !llvm.loop !14

for.end28:                                        ; preds = %for.inc26
  call void @matrix(ptr noundef nonnull %inA, ptr noundef nonnull %inB, ptr noundef nonnull %outC)
  ret i32 0
}

; Function Attrs: nounwind
declare void @srand(i32 noundef) #1

; Function Attrs: nounwind
declare i32 @rand() #1

attributes #0 = { nounwind uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
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
!6 = distinct !{!6, !7, !8}
!7 = !{!"llvm.loop.mustprogress"}
!8 = !{!"llvm.loop.vectorize.enable", i1 true}
!9 = distinct !{!9, !7}
!10 = distinct !{!10, !7}
!11 = distinct !{!11, !7}
!12 = distinct !{!12, !7}
!13 = distinct !{!13, !7}
!14 = distinct !{!14, !7}
