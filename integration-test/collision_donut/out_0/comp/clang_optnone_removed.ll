; ModuleID = '/home/shundroid/dynamatic/integration-test/collision_donut/collision_donut.c'
source_filename = "/home/shundroid/dynamatic/integration-test/collision_donut/collision_donut.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline nounwind  uwtable
define dso_local i32 @collision_donut(ptr noundef %x, ptr noundef %y) #0 {
entry:
  %x.addr = alloca ptr, align 8
  %y.addr = alloca ptr, align 8
  %err = alloca i32, align 4
  %i = alloca i32, align 4
  %xi = alloca i32, align 4
  %yi = alloca i32, align 4
  %distance_2 = alloca i32, align 4
  store ptr %x, ptr %x.addr, align 8
  store ptr %y, ptr %y.addr, align 8
  store i32 0, ptr %err, align 4
  store i32 0, ptr %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32, ptr %i, align 4
  %cmp = icmp slt i32 %0, 1000
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %1 = load ptr, ptr %x.addr, align 8
  %2 = load i32, ptr %i, align 4
  %idxprom = sext i32 %2 to i64
  %arrayidx = getelementptr inbounds i32, ptr %1, i64 %idxprom
  %3 = load i32, ptr %arrayidx, align 4
  store i32 %3, ptr %xi, align 4
  %4 = load ptr, ptr %y.addr, align 8
  %5 = load i32, ptr %i, align 4
  %idxprom1 = sext i32 %5 to i64
  %arrayidx2 = getelementptr inbounds i32, ptr %4, i64 %idxprom1
  %6 = load i32, ptr %arrayidx2, align 4
  store i32 %6, ptr %yi, align 4
  %7 = load i32, ptr %xi, align 4
  %8 = load i32, ptr %xi, align 4
  %mul = mul nsw i32 %7, %8
  %9 = load i32, ptr %yi, align 4
  %10 = load i32, ptr %yi, align 4
  %mul3 = mul nsw i32 %9, %10
  %add = add nsw i32 %mul, %mul3
  store i32 %add, ptr %distance_2, align 4
  %11 = load i32, ptr %distance_2, align 4
  %cmp4 = icmp slt i32 %11, 4
  br i1 %cmp4, label %if.then, label %if.end

if.then:                                          ; preds = %for.body
  store i32 -1, ptr %err, align 4
  br label %for.end

if.end:                                           ; preds = %for.body
  %12 = load i32, ptr %distance_2, align 4
  %cmp5 = icmp sgt i32 %12, 19000
  br i1 %cmp5, label %if.then6, label %if.end7

if.then6:                                         ; preds = %if.end
  store i32 -2, ptr %err, align 4
  br label %for.end

if.end7:                                          ; preds = %if.end
  br label %for.inc

for.inc:                                          ; preds = %if.end7
  %13 = load i32, ptr %i, align 4
  %inc = add nsw i32 %13, 1
  store i32 %inc, ptr %i, align 4
  br label %for.cond, !llvm.loop !6

for.end:                                          ; preds = %if.then6, %if.then, %for.cond
  %14 = load i32, ptr %i, align 4
  %shl = shl i32 %14, 1
  %15 = load i32, ptr %err, align 4
  %and = and i32 %shl, %15
  ret i32 %and
}

; Function Attrs: noinline nounwind  uwtable
define dso_local i32 @main() #0 {
entry:
  %retval = alloca i32, align 4
  %x = alloca [1000 x i32], align 16
  %y = alloca [1000 x i32], align 16
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
  %call = call i32 @rand() #2
  %rem = srem i32 %call, 100
  %1 = load i32, ptr %j, align 4
  %idxprom = sext i32 %1 to i64
  %arrayidx = getelementptr inbounds [1000 x i32], ptr %x, i64 0, i64 %idxprom
  store i32 %rem, ptr %arrayidx, align 4
  %call1 = call i32 @rand() #2
  %rem2 = srem i32 %call1, 100
  %2 = load i32, ptr %j, align 4
  %idxprom3 = sext i32 %2 to i64
  %arrayidx4 = getelementptr inbounds [1000 x i32], ptr %y, i64 0, i64 %idxprom3
  store i32 %rem2, ptr %arrayidx4, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %3 = load i32, ptr %j, align 4
  %inc = add nsw i32 %3, 1
  store i32 %inc, ptr %j, align 4
  br label %for.cond, !llvm.loop !8

for.end:                                          ; preds = %for.cond
  %arraydecay = getelementptr inbounds [1000 x i32], ptr %x, i64 0, i64 0
  %arraydecay5 = getelementptr inbounds [1000 x i32], ptr %y, i64 0, i64 0
  %call6 = call i32 @collision_donut(ptr noundef %arraydecay, ptr noundef %arraydecay5)
  %4 = load i32, ptr %retval, align 4
  ret i32 %4
}

; Function Attrs: nounwind
declare void @srand(i32 noundef) #1

; Function Attrs: nounwind
declare i32 @rand() #1

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
