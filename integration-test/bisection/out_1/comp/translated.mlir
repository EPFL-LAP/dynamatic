#loop_annotation = #llvm.loop_annotation<mustProgress = true>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>} {
  llvm.func @bisection(%arg0: f32 {llvm.noundef}, %arg1: f32 {llvm.noundef}, %arg2: f32 {llvm.noundef}) -> f32 attributes {passthrough = ["noinline", "nounwind", ["uwtable", "2"], ["frame-pointer", "all"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    %0 = llvm.mlir.constant(-2.000000e+00 : f32) : f32
    %1 = llvm.mlir.constant(0 : i32) : i32
    %2 = llvm.mlir.constant(5.000000e-01 : f32) : f32
    %3 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %4 = llvm.mlir.constant(1 : i32) : i32
    %5 = llvm.mlir.constant(100 : i32) : i32
    %6 = llvm.fmul %arg0, %arg0  : f32
    %7 = llvm.fadd %6, %0  : f32
    llvm.br ^bb1(%arg0, %arg1, %1, %7 : f32, f32, i32, f32)
  ^bb1(%8: f32, %9: f32, %10: i32, %11: f32):  // 2 preds: ^bb0, ^bb3
    %12 = llvm.fadd %8, %9  : f32
    %13 = llvm.fmul %12, %2  : f32
    %14 = llvm.fmul %13, %13  : f32
    %15 = llvm.fadd %14, %0  : f32
    %16 = llvm.intr.fabs(%15)  : (f32) -> f32
    %17 = llvm.fcmp "olt" %16, %arg2 : f32
    llvm.cond_br %17, ^bb4(%13 : f32), ^bb2
  ^bb2:  // pred: ^bb1
    %18 = llvm.fsub %9, %8  : f32
    %19 = llvm.fmul %18, %2  : f32
    %20 = llvm.fcmp "olt" %19, %arg2 : f32
    llvm.cond_br %20, ^bb4(%13 : f32), ^bb3
  ^bb3:  // pred: ^bb2
    %21 = llvm.fmul %11, %15  : f32
    %22 = llvm.fcmp "olt" %21, %3 : f32
    %23 = llvm.select %22, %13, %9 : i1, f32
    %24 = llvm.select %22, %8, %13 : i1, f32
    %25 = llvm.select %22, %11, %15 : i1, f32
    %26 = llvm.add %10, %4  : i32
    %27 = llvm.icmp "ult" %26, %5 : i32
    llvm.cond_br %27, ^bb1(%24, %23, %26, %25 : f32, f32, i32, f32), ^bb4(%3 : f32) {loop_annotation = #loop_annotation}
  ^bb4(%28: f32):  // 3 preds: ^bb1, ^bb2, ^bb3
    llvm.return %28 : f32
  }
  llvm.func @main() -> i32 attributes {passthrough = ["noinline", "nounwind", ["uwtable", "2"], ["frame-pointer", "all"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    %0 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %1 = llvm.mlir.constant(1.000000e+02 : f32) : f32
    %2 = llvm.mlir.constant(1.000000e-10 : f32) : f32
    %3 = llvm.mlir.constant(0 : i32) : i32
    %4 = llvm.call @bisection(%0, %1, %2) : (f32, f32, f32) -> f32
    llvm.return %3 : i32
  }
}
