#loop_annotation = #llvm.loop_annotation<mustProgress = true>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>} {
  llvm.func @collision_donut(%arg0: !llvm.ptr {llvm.noundef}, %arg1: !llvm.ptr {llvm.noundef}) -> i32 attributes {passthrough = ["noinline", "nounwind", ["uwtable", "2"], ["frame-pointer", "all"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    %0 = llvm.mlir.constant(0 : i32) : i32
    %1 = llvm.mlir.constant(4 : i32) : i32
    %2 = llvm.mlir.constant(-1 : i32) : i32
    %3 = llvm.mlir.constant(19000 : i32) : i32
    %4 = llvm.mlir.constant(-2 : i32) : i32
    %5 = llvm.mlir.constant(1 : i32) : i32
    %6 = llvm.mlir.constant(1000 : i32) : i32
    llvm.br ^bb1(%0 : i32)
  ^bb1(%7: i32):  // 2 preds: ^bb0, ^bb3
    %8 = llvm.zext %7 : i32 to i64
    %9 = llvm.getelementptr inbounds %arg0[%8] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %10 = llvm.load %9 {alignment = 4 : i64} : !llvm.ptr -> i32
    %11 = llvm.zext %7 : i32 to i64
    %12 = llvm.getelementptr inbounds %arg1[%11] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %13 = llvm.load %12 {alignment = 4 : i64} : !llvm.ptr -> i32
    %14 = llvm.mul %10, %10  : i32
    %15 = llvm.mul %13, %13  : i32
    %16 = llvm.add %14, %15  : i32
    %17 = llvm.icmp "ult" %16, %1 : i32
    llvm.cond_br %17, ^bb4(%7, %2 : i32, i32), ^bb2
  ^bb2:  // pred: ^bb1
    %18 = llvm.icmp "ugt" %16, %3 : i32
    llvm.cond_br %18, ^bb4(%7, %4 : i32, i32), ^bb3
  ^bb3:  // pred: ^bb2
    %19 = llvm.add %7, %5  : i32
    %20 = llvm.icmp "ult" %19, %6 : i32
    llvm.cond_br %20, ^bb1(%19 : i32), ^bb4(%19, %0 : i32, i32) {loop_annotation = #loop_annotation}
  ^bb4(%21: i32, %22: i32):  // 3 preds: ^bb1, ^bb2, ^bb3
    %23 = llvm.shl %21, %5  : i32
    %24 = llvm.and %23, %22  : i32
    llvm.return %24 : i32
  }
  llvm.func @main() -> i32 attributes {passthrough = ["noinline", "nounwind", ["uwtable", "2"], ["frame-pointer", "all"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(13 : i32) : i32
    %2 = llvm.mlir.constant(0 : i32) : i32
    %3 = llvm.mlir.constant(100 : i32) : i32
    %4 = llvm.mlir.constant(0 : i64) : i64
    %5 = llvm.mlir.constant(1000 : i32) : i32
    %6 = llvm.alloca %0 x !llvm.array<1000 x i32> {alignment = 16 : i64} : (i32) -> !llvm.ptr
    %7 = llvm.alloca %0 x !llvm.array<1000 x i32> {alignment = 16 : i64} : (i32) -> !llvm.ptr
    llvm.call @srand(%1) : (i32) -> ()
    llvm.br ^bb1(%2 : i32)
  ^bb1(%8: i32):  // 2 preds: ^bb0, ^bb1
    %9 = llvm.call @rand() : () -> i32
    %10 = llvm.srem %9, %3  : i32
    %11 = llvm.zext %8 : i32 to i64
    %12 = llvm.getelementptr inbounds %6[%4, %11] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.array<1000 x i32>
    llvm.store %10, %12 {alignment = 4 : i64} : i32, !llvm.ptr
    %13 = llvm.call @rand() : () -> i32
    %14 = llvm.srem %13, %3  : i32
    %15 = llvm.zext %8 : i32 to i64
    %16 = llvm.getelementptr inbounds %7[%4, %15] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.array<1000 x i32>
    llvm.store %14, %16 {alignment = 4 : i64} : i32, !llvm.ptr
    %17 = llvm.add %8, %0  : i32
    %18 = llvm.icmp "ult" %17, %5 : i32
    llvm.cond_br %18, ^bb1(%17 : i32), ^bb2 {loop_annotation = #loop_annotation}
  ^bb2:  // pred: ^bb1
    %19 = llvm.call @collision_donut(%6, %7) : (!llvm.ptr, !llvm.ptr) -> i32
    llvm.return %2 : i32
  }
  llvm.func @srand(i32 {llvm.noundef}) attributes {passthrough = ["nounwind", ["frame-pointer", "all"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]}
  llvm.func @rand() -> i32 attributes {passthrough = ["nounwind", ["frame-pointer", "all"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]}
}
