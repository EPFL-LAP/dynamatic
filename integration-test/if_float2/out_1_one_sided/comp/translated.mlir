#loop_annotation = #llvm.loop_annotation<mustProgress = true>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>} {
  llvm.func @if_float2(%arg0: f32 {llvm.noundef}, %arg1: !llvm.ptr {llvm.noundef}, %arg2: !llvm.ptr {llvm.noundef}) -> f32 attributes {passthrough = ["noinline", "nounwind", ["uwtable", "2"], ["frame-pointer", "all"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    %0 = llvm.mlir.constant(0 : i32) : i32
    %1 = llvm.mlir.constant(-0.899999976 : f32) : f32
    %2 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %3 = llvm.mlir.constant(3.000000e+00 : f32) : f32
    %4 = llvm.mlir.constant(1.000000e+00 : f32) : f32
    %5 = llvm.mlir.constant(1 : i32) : i32
    %6 = llvm.mlir.constant(100 : i32) : i32
    llvm.br ^bb1(%0, %arg0 : i32, f32)
  ^bb1(%7: i32, %8: f32):  // 2 preds: ^bb0, ^bb4
    %9 = llvm.zext %7 : i32 to i64
    %10 = llvm.getelementptr inbounds %arg1[%9] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %11 = llvm.load %10 {alignment = 4 : i64} : !llvm.ptr -> f32
    %12 = llvm.fmul %11, %8  : f32
    %13 = llvm.fmul %8, %1  : f32
    %14 = llvm.fadd %12, %13  : f32
    %15 = llvm.fcmp "ugt" %14, %2 : f32
    llvm.cond_br %15, ^bb3, ^bb2
  ^bb2:  // pred: ^bb1
    %16 = llvm.fadd %8, %3  : f32
    llvm.br ^bb4(%16 : f32)
  ^bb3:  // pred: ^bb1
    %17 = llvm.zext %7 : i32 to i64
    %18 = llvm.getelementptr inbounds %arg2[%17] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %8, %18 {alignment = 4 : i64} : f32, !llvm.ptr
    %19 = llvm.fadd %8, %4  : f32
    llvm.br ^bb4(%19 : f32)
  ^bb4(%20: f32):  // 2 preds: ^bb2, ^bb3
    %21 = llvm.fdiv %4, %20  : f32
    %22 = llvm.add %7, %5  : i32
    %23 = llvm.icmp "ult" %22, %6 : i32
    llvm.cond_br %23, ^bb1(%22, %21 : i32, f32), ^bb5(%21 : f32) {loop_annotation = #loop_annotation}
  ^bb5(%24: f32):  // pred: ^bb4
    llvm.return %24 : f32
  }
  llvm.func @main() -> i32 attributes {passthrough = ["noinline", "nounwind", ["uwtable", "2"], ["frame-pointer", "all"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(13 : i32) : i32
    %2 = llvm.mlir.constant(0 : i32) : i32
    %3 = llvm.mlir.constant(100 : i32) : i32
    %4 = llvm.mlir.constant(1.000000e+02 : f32) : f32
    %5 = llvm.mlir.constant(0 : i64) : i64
    %6 = llvm.alloca %0 x !llvm.array<100 x f32> {alignment = 16 : i64} : (i32) -> !llvm.ptr
    %7 = llvm.alloca %0 x !llvm.array<100 x f32> {alignment = 16 : i64} : (i32) -> !llvm.ptr
    llvm.call @srand(%1) : (i32) -> ()
    llvm.br ^bb1(%2 : i32)
  ^bb1(%8: i32):  // 2 preds: ^bb0, ^bb1
    %9 = llvm.call @rand() : () -> i32
    %10 = llvm.srem %9, %3  : i32
    %11 = llvm.sitofp %10 : i32 to f32
    %12 = llvm.fdiv %11, %4  : f32
    %13 = llvm.zext %8 : i32 to i64
    %14 = llvm.getelementptr inbounds %6[%5, %13] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.array<100 x f32>
    llvm.store %12, %14 {alignment = 4 : i64} : f32, !llvm.ptr
    %15 = llvm.add %8, %0  : i32
    %16 = llvm.icmp "ult" %15, %3 : i32
    llvm.cond_br %16, ^bb1(%15 : i32), ^bb2 {loop_annotation = #loop_annotation}
  ^bb2:  // pred: ^bb1
    %17 = llvm.call @if_float2(%4, %6, %7) : (f32, !llvm.ptr, !llvm.ptr) -> f32
    llvm.return %2 : i32
  }
  llvm.func @srand(i32 {llvm.noundef}) attributes {passthrough = ["nounwind", ["frame-pointer", "all"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]}
  llvm.func @rand() -> i32 attributes {passthrough = ["nounwind", ["frame-pointer", "all"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]}
}
