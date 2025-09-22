#loop_annotation = #llvm.loop_annotation<mustProgress = true>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>} {
  llvm.func @golden_ratio(%arg0: f32 {llvm.noundef}, %arg1: f32 {llvm.noundef}) -> f32 attributes {passthrough = ["noinline", "nounwind", ["uwtable", "2"], ["frame-pointer", "all"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    %0 = llvm.mlir.constant(0 : i32) : i32
    %1 = llvm.mlir.constant(5.000000e-01 : f32) : f32
    %2 = llvm.mlir.constant(1.000000e-01 : f32) : f32
    %3 = llvm.mlir.constant(1.000000e+00 : f32) : f32
    %4 = llvm.mlir.constant(1 : i32) : i32
    %5 = llvm.mlir.constant(100 : i32) : i32
    llvm.br ^bb1(%0, %arg1, %arg0 : i32, f32, f32)
  ^bb1(%6: i32, %7: f32, %8: f32):  // 2 preds: ^bb0, ^bb3
    llvm.br ^bb2(%8 : f32)
  ^bb2(%9: f32):  // 2 preds: ^bb1, ^bb2
    %10 = llvm.fmul %9, %7  : f32
    %11 = llvm.fadd %9, %10  : f32
    %12 = llvm.fmul %11, %1  : f32
    %13 = llvm.fsub %12, %9  : f32
    %14 = llvm.intr.fabs(%13)  : (f32) -> f32
    %15 = llvm.fcmp "olt" %14, %2 : f32
    llvm.cond_br %15, ^bb3(%9 : f32), ^bb2(%12 : f32)
  ^bb3(%16: f32):  // pred: ^bb2
    %17 = llvm.fadd %16, %3  : f32
    %18 = llvm.fdiv %3, %17  : f32
    %19 = llvm.add %6, %4  : i32
    %20 = llvm.icmp "ult" %19, %5 : i32
    llvm.cond_br %20, ^bb1(%19, %18, %17 : i32, f32, f32), ^bb4(%17 : f32) {loop_annotation = #loop_annotation}
  ^bb4(%21: f32):  // pred: ^bb3
    llvm.return %21 : f32
  }
  llvm.func @main() -> i32 attributes {passthrough = ["noinline", "nounwind", ["uwtable", "2"], ["frame-pointer", "all"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    %0 = llvm.mlir.constant(1.000000e+02 : f32) : f32
    %1 = llvm.mlir.constant(0.00999999977 : f32) : f32
    %2 = llvm.mlir.constant(0 : i32) : i32
    %3 = llvm.call @golden_ratio(%0, %1) : (f32, f32) -> f32
    llvm.return %2 : i32
  }
}
