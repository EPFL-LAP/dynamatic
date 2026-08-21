; ModuleID = '/local/home/crizzi/CI_mapbuf/dynamatic/integration-test/matrix/matrix.c'
source_filename = "/local/home/crizzi/CI_mapbuf/dynamatic/integration-test/matrix/matrix.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: readwrite) uwtable
define dso_local void @matrix(ptr nocapture noundef readonly %inA, ptr nocapture noundef readonly %inB, ptr nocapture noundef writeonly %outC) local_unnamed_addr #0 !dbg !8 {
entry:
  br label %for.cond1.preheader, !dbg !12

for.cond1.preheader:                              ; preds = %entry, %for.cond.cleanup3
  %indvars.iv44 = phi i64 [ 0, %entry ], [ %indvars.iv.next45, %for.cond.cleanup3 ]
  br label %scalar.ph, !dbg !13

for.cond.cleanup:                                 ; preds = %for.cond.cleanup3
  ret void, !dbg !14

scalar.ph:                                        ; preds = %for.cond.cleanup7, %for.cond1.preheader
  %indvars.iv40 = phi i64 [ 0, %for.cond1.preheader ], [ %indvars.iv.next41, %for.cond.cleanup7 ]
  br label %for.body8, !dbg !15

for.cond.cleanup3:                                ; preds = %for.cond.cleanup7
  %indvars.iv.next45 = add nuw nsw i64 %indvars.iv44, 1, !dbg !16
  %exitcond47.not = icmp eq i64 %indvars.iv.next45, 30, !dbg !17
  br i1 %exitcond47.not, label %for.cond.cleanup, label %for.cond1.preheader, !dbg !12, !llvm.loop !18

for.cond.cleanup7:                                ; preds = %for.body8
  %arrayidx18 = getelementptr inbounds [30 x i32], ptr %outC, i64 %indvars.iv44, i64 %indvars.iv40, !dbg !21
  store i32 %add.2, ptr %arrayidx18, align 4, !dbg !22, !tbaa !23
  %indvars.iv.next41 = add nuw nsw i64 %indvars.iv40, 1, !dbg !27
  %exitcond43.not = icmp eq i64 %indvars.iv.next41, 30, !dbg !28
  br i1 %exitcond43.not, label %for.cond.cleanup3, label %scalar.ph, !dbg !13, !llvm.loop !29

for.body8:                                        ; preds = %for.body8, %scalar.ph
  %indvars.iv = phi i64 [ 0, %scalar.ph ], [ %indvars.iv.next.2, %for.body8 ]
  %sumMult.035 = phi i32 [ 0, %scalar.ph ], [ %add.2, %for.body8 ]
  %arrayidx10 = getelementptr inbounds [30 x i32], ptr %inA, i64 %indvars.iv44, i64 %indvars.iv, !dbg !31
  %0 = load i32, ptr %arrayidx10, align 4, !dbg !31, !tbaa !23
  %arrayidx14 = getelementptr inbounds [30 x i32], ptr %inB, i64 %indvars.iv, i64 %indvars.iv40, !dbg !32
  %1 = load i32, ptr %arrayidx14, align 4, !dbg !32, !tbaa !23
  %mul = mul nsw i32 %1, %0, !dbg !33
  %add = add nsw i32 %mul, %sumMult.035, !dbg !34
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1, !dbg !35
  %arrayidx10.1 = getelementptr inbounds [30 x i32], ptr %inA, i64 %indvars.iv44, i64 %indvars.iv.next, !dbg !31
  %2 = load i32, ptr %arrayidx10.1, align 4, !dbg !31, !tbaa !23
  %arrayidx14.1 = getelementptr inbounds [30 x i32], ptr %inB, i64 %indvars.iv.next, i64 %indvars.iv40, !dbg !32
  %3 = load i32, ptr %arrayidx14.1, align 4, !dbg !32, !tbaa !23
  %mul.1 = mul nsw i32 %3, %2, !dbg !33
  %add.1 = add nsw i32 %mul.1, %add, !dbg !34
  %indvars.iv.next.1 = add nuw nsw i64 %indvars.iv, 2, !dbg !35
  %arrayidx10.2 = getelementptr inbounds [30 x i32], ptr %inA, i64 %indvars.iv44, i64 %indvars.iv.next.1, !dbg !31
  %4 = load i32, ptr %arrayidx10.2, align 4, !dbg !31, !tbaa !23
  %arrayidx14.2 = getelementptr inbounds [30 x i32], ptr %inB, i64 %indvars.iv.next.1, i64 %indvars.iv40, !dbg !32
  %5 = load i32, ptr %arrayidx14.2, align 4, !dbg !32, !tbaa !23
  %mul.2 = mul nsw i32 %5, %4, !dbg !33
  %add.2 = add nsw i32 %mul.2, %add.1, !dbg !34
  %indvars.iv.next.2 = add nuw nsw i64 %indvars.iv, 3, !dbg !35
  %exitcond.not.2 = icmp eq i64 %indvars.iv.next.2, 30, !dbg !36
  br i1 %exitcond.not.2, label %for.cond.cleanup7, label %for.body8, !dbg !15, !llvm.loop !37
}

; Function Attrs: nounwind uwtable
define dso_local i32 @main() local_unnamed_addr #1 !dbg !41 {
entry:
  tail call void @srand(i32 noundef 13) #3, !dbg !42
  br label %for.cond1.preheader, !dbg !43

for.cond1.preheader:                              ; preds = %entry, %for.cond1.preheader
  %y.043 = phi i32 [ 0, %entry ], [ %inc8, %for.cond1.preheader ]
  %call = tail call i32 @rand() #3, !dbg !44
  %call.1 = tail call i32 @rand() #3, !dbg !44
  %call.2 = tail call i32 @rand() #3, !dbg !44
  %call.3 = tail call i32 @rand() #3, !dbg !44
  %call.4 = tail call i32 @rand() #3, !dbg !44
  %call.5 = tail call i32 @rand() #3, !dbg !44
  %call.6 = tail call i32 @rand() #3, !dbg !44
  %call.7 = tail call i32 @rand() #3, !dbg !44
  %call.8 = tail call i32 @rand() #3, !dbg !44
  %call.9 = tail call i32 @rand() #3, !dbg !44
  %call.10 = tail call i32 @rand() #3, !dbg !44
  %call.11 = tail call i32 @rand() #3, !dbg !44
  %call.12 = tail call i32 @rand() #3, !dbg !44
  %call.13 = tail call i32 @rand() #3, !dbg !44
  %call.14 = tail call i32 @rand() #3, !dbg !44
  %call.15 = tail call i32 @rand() #3, !dbg !44
  %call.16 = tail call i32 @rand() #3, !dbg !44
  %call.17 = tail call i32 @rand() #3, !dbg !44
  %call.18 = tail call i32 @rand() #3, !dbg !44
  %call.19 = tail call i32 @rand() #3, !dbg !44
  %call.20 = tail call i32 @rand() #3, !dbg !44
  %call.21 = tail call i32 @rand() #3, !dbg !44
  %call.22 = tail call i32 @rand() #3, !dbg !44
  %call.23 = tail call i32 @rand() #3, !dbg !44
  %call.24 = tail call i32 @rand() #3, !dbg !44
  %call.25 = tail call i32 @rand() #3, !dbg !44
  %call.26 = tail call i32 @rand() #3, !dbg !44
  %call.27 = tail call i32 @rand() #3, !dbg !44
  %call.28 = tail call i32 @rand() #3, !dbg !44
  %call.29 = tail call i32 @rand() #3, !dbg !44
  %inc8 = add nuw nsw i32 %y.043, 1, !dbg !45
  %exitcond.not = icmp eq i32 %inc8, 30, !dbg !46
  br i1 %exitcond.not, label %for.cond16.preheader, label %for.cond1.preheader, !dbg !43, !llvm.loop !47

for.cond16.preheader:                             ; preds = %for.cond1.preheader, %for.cond16.preheader
  %y10.045 = phi i32 [ %inc30, %for.cond16.preheader ], [ 0, %for.cond1.preheader ]
  %call20 = tail call i32 @rand() #3, !dbg !49
  %call20.1 = tail call i32 @rand() #3, !dbg !49
  %call20.2 = tail call i32 @rand() #3, !dbg !49
  %call20.3 = tail call i32 @rand() #3, !dbg !49
  %call20.4 = tail call i32 @rand() #3, !dbg !49
  %call20.5 = tail call i32 @rand() #3, !dbg !49
  %call20.6 = tail call i32 @rand() #3, !dbg !49
  %call20.7 = tail call i32 @rand() #3, !dbg !49
  %call20.8 = tail call i32 @rand() #3, !dbg !49
  %call20.9 = tail call i32 @rand() #3, !dbg !49
  %call20.10 = tail call i32 @rand() #3, !dbg !49
  %call20.11 = tail call i32 @rand() #3, !dbg !49
  %call20.12 = tail call i32 @rand() #3, !dbg !49
  %call20.13 = tail call i32 @rand() #3, !dbg !49
  %call20.14 = tail call i32 @rand() #3, !dbg !49
  %call20.15 = tail call i32 @rand() #3, !dbg !49
  %call20.16 = tail call i32 @rand() #3, !dbg !49
  %call20.17 = tail call i32 @rand() #3, !dbg !49
  %call20.18 = tail call i32 @rand() #3, !dbg !49
  %call20.19 = tail call i32 @rand() #3, !dbg !49
  %call20.20 = tail call i32 @rand() #3, !dbg !49
  %call20.21 = tail call i32 @rand() #3, !dbg !49
  %call20.22 = tail call i32 @rand() #3, !dbg !49
  %call20.23 = tail call i32 @rand() #3, !dbg !49
  %call20.24 = tail call i32 @rand() #3, !dbg !49
  %call20.25 = tail call i32 @rand() #3, !dbg !49
  %call20.26 = tail call i32 @rand() #3, !dbg !49
  %call20.27 = tail call i32 @rand() #3, !dbg !49
  %call20.28 = tail call i32 @rand() #3, !dbg !49
  %call20.29 = tail call i32 @rand() #3, !dbg !49
  %inc30 = add nuw nsw i32 %y10.045, 1, !dbg !50
  %exitcond47.not = icmp eq i32 %inc30, 30, !dbg !51
  br i1 %exitcond47.not, label %for.cond1.preheader.i.preheader, label %for.cond16.preheader, !dbg !52, !llvm.loop !53

for.cond1.preheader.i.preheader:                  ; preds = %for.cond16.preheader
  ret i32 0, !dbg !55
}

; Function Attrs: nounwind
declare void @srand(i32 noundef) local_unnamed_addr #2

; Function Attrs: nounwind
declare i32 @rand() local_unnamed_addr #2

attributes #0 = { nofree norecurse nosync nounwind memory(argmem: readwrite) uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #2 = { nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #3 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6}
!llvm.ident = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 18.0.0 (https://github.com/EPFL-LAP/llvm-project.git b06546b8f001a888f346b38b9f3ae0da11efbff2)", isOptimized: true, runtimeVersion: 0, emissionKind: NoDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "/local/home/crizzi/CI_mapbuf/dynamatic/integration-test/matrix/matrix.c", directory: "/local/home/crizzi/CI_mapbuf/dynamatic")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{i32 1, !"wchar_size", i32 4}
!4 = !{i32 8, !"PIC Level", i32 2}
!5 = !{i32 7, !"PIE Level", i32 2}
!6 = !{i32 7, !"uwtable", i32 2}
!7 = !{!"clang version 18.0.0 (https://github.com/EPFL-LAP/llvm-project.git b06546b8f001a888f346b38b9f3ae0da11efbff2)"}
!8 = distinct !DISubprogram(name: "matrix", scope: !9, file: !9, line: 11, type: !10, scopeLine: 12, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!9 = !DIFile(filename: "integration-test/matrix/matrix.c", directory: "/local/home/crizzi/CI_mapbuf/dynamatic")
!10 = !DISubroutineType(types: !11)
!11 = !{}
!12 = !DILocation(line: 13, column: 3, scope: !8)
!13 = !DILocation(line: 14, column: 5, scope: !8)
!14 = !DILocation(line: 24, column: 1, scope: !8)
!15 = !DILocation(line: 18, column: 7, scope: !8)
!16 = !DILocation(line: 13, column: 37, scope: !8)
!17 = !DILocation(line: 13, column: 26, scope: !8)
!18 = distinct !{!18, !12, !19, !20}
!19 = !DILocation(line: 23, column: 3, scope: !8)
!20 = !{!"llvm.loop.mustprogress"}
!21 = !DILocation(line: 21, column: 7, scope: !8)
!22 = !DILocation(line: 21, column: 18, scope: !8)
!23 = !{!24, !24, i64 0}
!24 = !{!"int", !25, i64 0}
!25 = !{!"omnipotent char", !26, i64 0}
!26 = !{!"Simple C/C++ TBAA"}
!27 = !DILocation(line: 14, column: 39, scope: !8)
!28 = !DILocation(line: 14, column: 28, scope: !8)
!29 = distinct !{!29, !13, !30, !20}
!30 = !DILocation(line: 22, column: 5, scope: !8)
!31 = !DILocation(line: 19, column: 20, scope: !8)
!32 = !DILocation(line: 19, column: 32, scope: !8)
!33 = !DILocation(line: 19, column: 30, scope: !8)
!34 = !DILocation(line: 19, column: 17, scope: !8)
!35 = !DILocation(line: 18, column: 41, scope: !8)
!36 = !DILocation(line: 18, column: 30, scope: !8)
!37 = distinct !{!37, !15, !38, !20, !39, !40}
!38 = !DILocation(line: 20, column: 7, scope: !8)
!39 = !{!"llvm.loop.unroll.runtime.disable"}
!40 = !{!"llvm.loop.isvectorized", i32 1}
!41 = distinct !DISubprogram(name: "main", scope: !9, file: !9, line: 26, type: !10, scopeLine: 26, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!42 = !DILocation(line: 31, column: 3, scope: !41)
!43 = !DILocation(line: 32, column: 3, scope: !41)
!44 = !DILocation(line: 34, column: 19, scope: !41)
!45 = !DILocation(line: 32, column: 31, scope: !41)
!46 = !DILocation(line: 32, column: 21, scope: !41)
!47 = distinct !{!47, !43, !48, !20}
!48 = !DILocation(line: 36, column: 3, scope: !41)
!49 = !DILocation(line: 39, column: 19, scope: !41)
!50 = !DILocation(line: 37, column: 31, scope: !41)
!51 = !DILocation(line: 37, column: 21, scope: !41)
!52 = !DILocation(line: 37, column: 3, scope: !41)
!53 = distinct !{!53, !52, !54, !20}
!54 = !DILocation(line: 41, column: 3, scope: !41)
!55 = !DILocation(line: 45, column: 1, scope: !41)
