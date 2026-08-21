; ModuleID = 'integration-test/matrix/out/comp/clang.opt.ll'
source_filename = "/local/home/crizzi/CI_mapbuf/dynamatic/integration-test/matrix/matrix.c"

; Function Attrs: nounwind uwtable
define dso_local void @matrix(ptr noundef %inA, ptr noundef %inB, ptr noundef %outC) #0 {
entry:
  br label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %for.inc20, %entry
  %i.04 = phi i32 [ 0, %entry ], [ %inc21, %for.inc20 ]
  %idxprom = zext i32 %i.04 to i64
  %arrayidx8 = getelementptr inbounds [30 x i32], ptr %inA, i64 %idxprom, i64 0
  br label %for.cond4.preheader

for.cond4.preheader:                              ; preds = %for.inc, %for.cond1.preheader
  %j.03 = phi i32 [ 0, %for.cond1.preheader ], [ %inc18, %for.inc ]
  br label %for.inc

for.inc:                                          ; preds = %for.cond4.preheader
  %0 = load i32, ptr %arrayidx8, align 4
  %idxprom11 = zext i32 %j.03 to i64
  %arrayidx12 = getelementptr inbounds [30 x i32], ptr %inB, i64 0, i64 %idxprom11
  %1 = load i32, ptr %arrayidx12, align 4
  %mul = mul nsw i32 %0, %1
  %idxprom.1 = zext i32 %i.04 to i64
  %arrayidx8.1 = getelementptr inbounds [30 x i32], ptr %inA, i64 %idxprom.1, i64 1
  %2 = load i32, ptr %arrayidx8.1, align 4
  %idxprom11.1 = zext i32 %j.03 to i64
  %arrayidx12.1 = getelementptr inbounds [30 x i32], ptr %inB, i64 1, i64 %idxprom11.1
  %3 = load i32, ptr %arrayidx12.1, align 4
  %mul.1 = mul nsw i32 %2, %3
  %add.1 = add nsw i32 %mul, %mul.1
  %idxprom.2 = zext i32 %i.04 to i64
  %arrayidx8.2 = getelementptr inbounds [30 x i32], ptr %inA, i64 %idxprom.2, i64 2
  %4 = load i32, ptr %arrayidx8.2, align 4
  %idxprom11.2 = zext i32 %j.03 to i64
  %arrayidx12.2 = getelementptr inbounds [30 x i32], ptr %inB, i64 2, i64 %idxprom11.2
  %5 = load i32, ptr %arrayidx12.2, align 4
  %mul.2 = mul nsw i32 %4, %5
  %add.2 = add nsw i32 %add.1, %mul.2
  %idxprom.3 = zext i32 %i.04 to i64
  %arrayidx8.3 = getelementptr inbounds [30 x i32], ptr %inA, i64 %idxprom.3, i64 3
  %6 = load i32, ptr %arrayidx8.3, align 4
  %idxprom11.3 = zext i32 %j.03 to i64
  %arrayidx12.3 = getelementptr inbounds [30 x i32], ptr %inB, i64 3, i64 %idxprom11.3
  %7 = load i32, ptr %arrayidx12.3, align 4
  %mul.3 = mul nsw i32 %6, %7
  %add.3 = add nsw i32 %add.2, %mul.3
  %idxprom.4 = zext i32 %i.04 to i64
  %arrayidx8.4 = getelementptr inbounds [30 x i32], ptr %inA, i64 %idxprom.4, i64 4
  %8 = load i32, ptr %arrayidx8.4, align 4
  %idxprom11.4 = zext i32 %j.03 to i64
  %arrayidx12.4 = getelementptr inbounds [30 x i32], ptr %inB, i64 4, i64 %idxprom11.4
  %9 = load i32, ptr %arrayidx12.4, align 4
  %mul.4 = mul nsw i32 %8, %9
  %add.4 = add nsw i32 %add.3, %mul.4
  %idxprom.5 = zext i32 %i.04 to i64
  %arrayidx8.5 = getelementptr inbounds [30 x i32], ptr %inA, i64 %idxprom.5, i64 5
  %10 = load i32, ptr %arrayidx8.5, align 4
  %idxprom11.5 = zext i32 %j.03 to i64
  %arrayidx12.5 = getelementptr inbounds [30 x i32], ptr %inB, i64 5, i64 %idxprom11.5
  %11 = load i32, ptr %arrayidx12.5, align 4
  %mul.5 = mul nsw i32 %10, %11
  %add.5 = add nsw i32 %add.4, %mul.5
  %idxprom.6 = zext i32 %i.04 to i64
  %arrayidx8.6 = getelementptr inbounds [30 x i32], ptr %inA, i64 %idxprom.6, i64 6
  %12 = load i32, ptr %arrayidx8.6, align 4
  %idxprom11.6 = zext i32 %j.03 to i64
  %arrayidx12.6 = getelementptr inbounds [30 x i32], ptr %inB, i64 6, i64 %idxprom11.6
  %13 = load i32, ptr %arrayidx12.6, align 4
  %mul.6 = mul nsw i32 %12, %13
  %add.6 = add nsw i32 %add.5, %mul.6
  %idxprom.7 = zext i32 %i.04 to i64
  %arrayidx8.7 = getelementptr inbounds [30 x i32], ptr %inA, i64 %idxprom.7, i64 7
  %14 = load i32, ptr %arrayidx8.7, align 4
  %idxprom11.7 = zext i32 %j.03 to i64
  %arrayidx12.7 = getelementptr inbounds [30 x i32], ptr %inB, i64 7, i64 %idxprom11.7
  %15 = load i32, ptr %arrayidx12.7, align 4
  %mul.7 = mul nsw i32 %14, %15
  %add.7 = add nsw i32 %add.6, %mul.7
  %idxprom.8 = zext i32 %i.04 to i64
  %arrayidx8.8 = getelementptr inbounds [30 x i32], ptr %inA, i64 %idxprom.8, i64 8
  %16 = load i32, ptr %arrayidx8.8, align 4
  %idxprom11.8 = zext i32 %j.03 to i64
  %arrayidx12.8 = getelementptr inbounds [30 x i32], ptr %inB, i64 8, i64 %idxprom11.8
  %17 = load i32, ptr %arrayidx12.8, align 4
  %mul.8 = mul nsw i32 %16, %17
  %add.8 = add nsw i32 %add.7, %mul.8
  %idxprom.9 = zext i32 %i.04 to i64
  %arrayidx8.9 = getelementptr inbounds [30 x i32], ptr %inA, i64 %idxprom.9, i64 9
  %18 = load i32, ptr %arrayidx8.9, align 4
  %idxprom11.9 = zext i32 %j.03 to i64
  %arrayidx12.9 = getelementptr inbounds [30 x i32], ptr %inB, i64 9, i64 %idxprom11.9
  %19 = load i32, ptr %arrayidx12.9, align 4
  %mul.9 = mul nsw i32 %18, %19
  %add.9 = add nsw i32 %add.8, %mul.9
  %idxprom.10 = zext i32 %i.04 to i64
  %arrayidx8.10 = getelementptr inbounds [30 x i32], ptr %inA, i64 %idxprom.10, i64 10
  %20 = load i32, ptr %arrayidx8.10, align 4
  %idxprom11.10 = zext i32 %j.03 to i64
  %arrayidx12.10 = getelementptr inbounds [30 x i32], ptr %inB, i64 10, i64 %idxprom11.10
  %21 = load i32, ptr %arrayidx12.10, align 4
  %mul.10 = mul nsw i32 %20, %21
  %add.10 = add nsw i32 %add.9, %mul.10
  %idxprom.11 = zext i32 %i.04 to i64
  %arrayidx8.11 = getelementptr inbounds [30 x i32], ptr %inA, i64 %idxprom.11, i64 11
  %22 = load i32, ptr %arrayidx8.11, align 4
  %idxprom11.11 = zext i32 %j.03 to i64
  %arrayidx12.11 = getelementptr inbounds [30 x i32], ptr %inB, i64 11, i64 %idxprom11.11
  %23 = load i32, ptr %arrayidx12.11, align 4
  %mul.11 = mul nsw i32 %22, %23
  %add.11 = add nsw i32 %add.10, %mul.11
  %idxprom.12 = zext i32 %i.04 to i64
  %arrayidx8.12 = getelementptr inbounds [30 x i32], ptr %inA, i64 %idxprom.12, i64 12
  %24 = load i32, ptr %arrayidx8.12, align 4
  %idxprom11.12 = zext i32 %j.03 to i64
  %arrayidx12.12 = getelementptr inbounds [30 x i32], ptr %inB, i64 12, i64 %idxprom11.12
  %25 = load i32, ptr %arrayidx12.12, align 4
  %mul.12 = mul nsw i32 %24, %25
  %add.12 = add nsw i32 %add.11, %mul.12
  %idxprom.13 = zext i32 %i.04 to i64
  %arrayidx8.13 = getelementptr inbounds [30 x i32], ptr %inA, i64 %idxprom.13, i64 13
  %26 = load i32, ptr %arrayidx8.13, align 4
  %idxprom11.13 = zext i32 %j.03 to i64
  %arrayidx12.13 = getelementptr inbounds [30 x i32], ptr %inB, i64 13, i64 %idxprom11.13
  %27 = load i32, ptr %arrayidx12.13, align 4
  %mul.13 = mul nsw i32 %26, %27
  %add.13 = add nsw i32 %add.12, %mul.13
  %idxprom.14 = zext i32 %i.04 to i64
  %arrayidx8.14 = getelementptr inbounds [30 x i32], ptr %inA, i64 %idxprom.14, i64 14
  %28 = load i32, ptr %arrayidx8.14, align 4
  %idxprom11.14 = zext i32 %j.03 to i64
  %arrayidx12.14 = getelementptr inbounds [30 x i32], ptr %inB, i64 14, i64 %idxprom11.14
  %29 = load i32, ptr %arrayidx12.14, align 4
  %mul.14 = mul nsw i32 %28, %29
  %add.14 = add nsw i32 %add.13, %mul.14
  %idxprom.15 = zext i32 %i.04 to i64
  %arrayidx8.15 = getelementptr inbounds [30 x i32], ptr %inA, i64 %idxprom.15, i64 15
  %30 = load i32, ptr %arrayidx8.15, align 4
  %idxprom11.15 = zext i32 %j.03 to i64
  %arrayidx12.15 = getelementptr inbounds [30 x i32], ptr %inB, i64 15, i64 %idxprom11.15
  %31 = load i32, ptr %arrayidx12.15, align 4
  %mul.15 = mul nsw i32 %30, %31
  %add.15 = add nsw i32 %add.14, %mul.15
  %idxprom.16 = zext i32 %i.04 to i64
  %arrayidx8.16 = getelementptr inbounds [30 x i32], ptr %inA, i64 %idxprom.16, i64 16
  %32 = load i32, ptr %arrayidx8.16, align 4
  %idxprom11.16 = zext i32 %j.03 to i64
  %arrayidx12.16 = getelementptr inbounds [30 x i32], ptr %inB, i64 16, i64 %idxprom11.16
  %33 = load i32, ptr %arrayidx12.16, align 4
  %mul.16 = mul nsw i32 %32, %33
  %add.16 = add nsw i32 %add.15, %mul.16
  %idxprom.17 = zext i32 %i.04 to i64
  %arrayidx8.17 = getelementptr inbounds [30 x i32], ptr %inA, i64 %idxprom.17, i64 17
  %34 = load i32, ptr %arrayidx8.17, align 4
  %idxprom11.17 = zext i32 %j.03 to i64
  %arrayidx12.17 = getelementptr inbounds [30 x i32], ptr %inB, i64 17, i64 %idxprom11.17
  %35 = load i32, ptr %arrayidx12.17, align 4
  %mul.17 = mul nsw i32 %34, %35
  %add.17 = add nsw i32 %add.16, %mul.17
  %idxprom.18 = zext i32 %i.04 to i64
  %arrayidx8.18 = getelementptr inbounds [30 x i32], ptr %inA, i64 %idxprom.18, i64 18
  %36 = load i32, ptr %arrayidx8.18, align 4
  %idxprom11.18 = zext i32 %j.03 to i64
  %arrayidx12.18 = getelementptr inbounds [30 x i32], ptr %inB, i64 18, i64 %idxprom11.18
  %37 = load i32, ptr %arrayidx12.18, align 4
  %mul.18 = mul nsw i32 %36, %37
  %add.18 = add nsw i32 %add.17, %mul.18
  %idxprom.19 = zext i32 %i.04 to i64
  %arrayidx8.19 = getelementptr inbounds [30 x i32], ptr %inA, i64 %idxprom.19, i64 19
  %38 = load i32, ptr %arrayidx8.19, align 4
  %idxprom11.19 = zext i32 %j.03 to i64
  %arrayidx12.19 = getelementptr inbounds [30 x i32], ptr %inB, i64 19, i64 %idxprom11.19
  %39 = load i32, ptr %arrayidx12.19, align 4
  %mul.19 = mul nsw i32 %38, %39
  %add.19 = add nsw i32 %add.18, %mul.19
  %idxprom.20 = zext i32 %i.04 to i64
  %arrayidx8.20 = getelementptr inbounds [30 x i32], ptr %inA, i64 %idxprom.20, i64 20
  %40 = load i32, ptr %arrayidx8.20, align 4
  %idxprom11.20 = zext i32 %j.03 to i64
  %arrayidx12.20 = getelementptr inbounds [30 x i32], ptr %inB, i64 20, i64 %idxprom11.20
  %41 = load i32, ptr %arrayidx12.20, align 4
  %mul.20 = mul nsw i32 %40, %41
  %add.20 = add nsw i32 %add.19, %mul.20
  %idxprom.21 = zext i32 %i.04 to i64
  %arrayidx8.21 = getelementptr inbounds [30 x i32], ptr %inA, i64 %idxprom.21, i64 21
  %42 = load i32, ptr %arrayidx8.21, align 4
  %idxprom11.21 = zext i32 %j.03 to i64
  %arrayidx12.21 = getelementptr inbounds [30 x i32], ptr %inB, i64 21, i64 %idxprom11.21
  %43 = load i32, ptr %arrayidx12.21, align 4
  %mul.21 = mul nsw i32 %42, %43
  %add.21 = add nsw i32 %add.20, %mul.21
  %idxprom.22 = zext i32 %i.04 to i64
  %arrayidx8.22 = getelementptr inbounds [30 x i32], ptr %inA, i64 %idxprom.22, i64 22
  %44 = load i32, ptr %arrayidx8.22, align 4
  %idxprom11.22 = zext i32 %j.03 to i64
  %arrayidx12.22 = getelementptr inbounds [30 x i32], ptr %inB, i64 22, i64 %idxprom11.22
  %45 = load i32, ptr %arrayidx12.22, align 4
  %mul.22 = mul nsw i32 %44, %45
  %add.22 = add nsw i32 %add.21, %mul.22
  %idxprom.23 = zext i32 %i.04 to i64
  %arrayidx8.23 = getelementptr inbounds [30 x i32], ptr %inA, i64 %idxprom.23, i64 23
  %46 = load i32, ptr %arrayidx8.23, align 4
  %idxprom11.23 = zext i32 %j.03 to i64
  %arrayidx12.23 = getelementptr inbounds [30 x i32], ptr %inB, i64 23, i64 %idxprom11.23
  %47 = load i32, ptr %arrayidx12.23, align 4
  %mul.23 = mul nsw i32 %46, %47
  %add.23 = add nsw i32 %add.22, %mul.23
  %idxprom.24 = zext i32 %i.04 to i64
  %arrayidx8.24 = getelementptr inbounds [30 x i32], ptr %inA, i64 %idxprom.24, i64 24
  %48 = load i32, ptr %arrayidx8.24, align 4
  %idxprom11.24 = zext i32 %j.03 to i64
  %arrayidx12.24 = getelementptr inbounds [30 x i32], ptr %inB, i64 24, i64 %idxprom11.24
  %49 = load i32, ptr %arrayidx12.24, align 4
  %mul.24 = mul nsw i32 %48, %49
  %add.24 = add nsw i32 %add.23, %mul.24
  %idxprom.25 = zext i32 %i.04 to i64
  %arrayidx8.25 = getelementptr inbounds [30 x i32], ptr %inA, i64 %idxprom.25, i64 25
  %50 = load i32, ptr %arrayidx8.25, align 4
  %idxprom11.25 = zext i32 %j.03 to i64
  %arrayidx12.25 = getelementptr inbounds [30 x i32], ptr %inB, i64 25, i64 %idxprom11.25
  %51 = load i32, ptr %arrayidx12.25, align 4
  %mul.25 = mul nsw i32 %50, %51
  %add.25 = add nsw i32 %add.24, %mul.25
  %idxprom.26 = zext i32 %i.04 to i64
  %arrayidx8.26 = getelementptr inbounds [30 x i32], ptr %inA, i64 %idxprom.26, i64 26
  %52 = load i32, ptr %arrayidx8.26, align 4
  %idxprom11.26 = zext i32 %j.03 to i64
  %arrayidx12.26 = getelementptr inbounds [30 x i32], ptr %inB, i64 26, i64 %idxprom11.26
  %53 = load i32, ptr %arrayidx12.26, align 4
  %mul.26 = mul nsw i32 %52, %53
  %add.26 = add nsw i32 %add.25, %mul.26
  %idxprom.27 = zext i32 %i.04 to i64
  %arrayidx8.27 = getelementptr inbounds [30 x i32], ptr %inA, i64 %idxprom.27, i64 27
  %54 = load i32, ptr %arrayidx8.27, align 4
  %idxprom11.27 = zext i32 %j.03 to i64
  %arrayidx12.27 = getelementptr inbounds [30 x i32], ptr %inB, i64 27, i64 %idxprom11.27
  %55 = load i32, ptr %arrayidx12.27, align 4
  %mul.27 = mul nsw i32 %54, %55
  %add.27 = add nsw i32 %add.26, %mul.27
  %idxprom.28 = zext i32 %i.04 to i64
  %arrayidx8.28 = getelementptr inbounds [30 x i32], ptr %inA, i64 %idxprom.28, i64 28
  %56 = load i32, ptr %arrayidx8.28, align 4
  %idxprom11.28 = zext i32 %j.03 to i64
  %arrayidx12.28 = getelementptr inbounds [30 x i32], ptr %inB, i64 28, i64 %idxprom11.28
  %57 = load i32, ptr %arrayidx12.28, align 4
  %mul.28 = mul nsw i32 %56, %57
  %add.28 = add nsw i32 %add.27, %mul.28
  %idxprom.29 = zext i32 %i.04 to i64
  %arrayidx8.29 = getelementptr inbounds [30 x i32], ptr %inA, i64 %idxprom.29, i64 29
  %58 = load i32, ptr %arrayidx8.29, align 4
  %idxprom11.29 = zext i32 %j.03 to i64
  %arrayidx12.29 = getelementptr inbounds [30 x i32], ptr %inB, i64 29, i64 %idxprom11.29
  %59 = load i32, ptr %arrayidx12.29, align 4
  %mul.29 = mul nsw i32 %58, %59
  %add.29 = add nsw i32 %add.28, %mul.29
  %idxprom13 = zext i32 %i.04 to i64
  %idxprom15 = zext i32 %j.03 to i64
  %arrayidx16 = getelementptr inbounds [30 x i32], ptr %outC, i64 %idxprom13, i64 %idxprom15
  store i32 %add.29, ptr %arrayidx16, align 4
  %inc18 = add i32 %j.03, 1
  %cmp2 = icmp ult i32 %inc18, 30
  br i1 %cmp2, label %for.cond4.preheader, label %for.inc20, !llvm.loop !6

for.inc20:                                        ; preds = %for.inc
  %inc21 = add i32 %i.04, 1
  %cmp = icmp ult i32 %inc21, 30
  br i1 %cmp, label %for.cond1.preheader, label %for.end22, !llvm.loop !8

for.end22:                                        ; preds = %for.inc20
  ret void
}

; Function Attrs: nounwind uwtable
define dso_local i32 @main() #0 {
entry:
  %inA = alloca [30 x [30 x i32]], align 16
  %inB = alloca [30 x [30 x i32]], align 16
  %outC = alloca [30 x [30 x i32]], align 16
  call void @srand(i32 noundef 13) #2
  br label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %for.inc6, %entry
  %y.02 = phi i32 [ 0, %entry ], [ %inc7, %for.inc6 ]
  br label %for.body3

for.body3:                                        ; preds = %for.body3, %for.cond1.preheader
  %x.01 = phi i32 [ 0, %for.cond1.preheader ], [ %inc, %for.body3 ]
  %call = call i32 @rand() #2
  %rem = srem i32 %call, 10
  %idxprom = zext i32 %y.02 to i64
  %idxprom4 = zext i32 %x.01 to i64
  %arrayidx5 = getelementptr inbounds [30 x [30 x i32]], ptr %inA, i64 0, i64 %idxprom, i64 %idxprom4
  store i32 %rem, ptr %arrayidx5, align 4
  %inc = add nuw nsw i32 %x.01, 1
  %cmp2 = icmp ult i32 %inc, 30
  br i1 %cmp2, label %for.body3, label %for.inc6, !llvm.loop !9

for.inc6:                                         ; preds = %for.body3
  %inc7 = add nuw nsw i32 %y.02, 1
  %cmp = icmp ult i32 %inc7, 30
  br i1 %cmp, label %for.cond1.preheader, label %for.cond14.preheader.preheader, !llvm.loop !10

for.cond14.preheader.preheader:                   ; preds = %for.inc6
  br label %for.cond14.preheader

for.cond14.preheader:                             ; preds = %for.cond14.preheader.preheader, %for.inc26
  %y9.04 = phi i32 [ %inc27, %for.inc26 ], [ 0, %for.cond14.preheader.preheader ]
  br label %for.body16

for.body16:                                       ; preds = %for.body16, %for.cond14.preheader
  %x13.03 = phi i32 [ 0, %for.cond14.preheader ], [ %inc24, %for.body16 ]
  %call17 = call i32 @rand() #2
  %rem18 = srem i32 %call17, 10
  %idxprom19 = zext i32 %y9.04 to i64
  %idxprom21 = zext i32 %x13.03 to i64
  %arrayidx22 = getelementptr inbounds [30 x [30 x i32]], ptr %inB, i64 0, i64 %idxprom19, i64 %idxprom21
  store i32 %rem18, ptr %arrayidx22, align 4
  %inc24 = add nuw nsw i32 %x13.03, 1
  %cmp15 = icmp ult i32 %inc24, 30
  br i1 %cmp15, label %for.body16, label %for.inc26, !llvm.loop !11

for.inc26:                                        ; preds = %for.body16
  %inc27 = add nuw nsw i32 %y9.04, 1
  %cmp11 = icmp ult i32 %inc27, 30
  br i1 %cmp11, label %for.cond14.preheader, label %for.end28, !llvm.loop !12

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
!6 = distinct !{!6, !7}
!7 = !{!"llvm.loop.mustprogress"}
!8 = distinct !{!8, !7}
!9 = distinct !{!9, !7}
!10 = distinct !{!10, !7}
!11 = distinct !{!11, !7}
!12 = distinct !{!12, !7}
