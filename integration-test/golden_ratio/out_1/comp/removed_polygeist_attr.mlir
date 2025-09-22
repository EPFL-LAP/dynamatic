#loop_annotation = #llvm.loop_annotation<mustProgress = true>
module {
  llvm.func @golden_ratio(%arg0: f32 {llvm.noundef}, %arg1: f32 {llvm.noundef}) -> f32 attributes {handshake.name = "func0", passthrough = ["noinline", "nounwind", ["uwtable", "2"], ["frame-pointer", "all"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    %0 = llvm.mlir.constant(0 : i32) {handshake.name = "mlir.constant0"} : i32
    %1 = llvm.mlir.constant(5.000000e-01 : f32) {handshake.name = "mlir.constant1"} : f32
    %2 = llvm.mlir.constant(1.000000e-01 : f32) {handshake.name = "mlir.constant2"} : f32
    %3 = llvm.mlir.constant(1.000000e+00 : f32) {handshake.name = "mlir.constant3"} : f32
    %4 = llvm.mlir.constant(1 : i32) {handshake.name = "mlir.constant4"} : i32
    %5 = llvm.mlir.constant(100 : i32) {handshake.name = "mlir.constant5"} : i32
    llvm.br ^bb1(%0, %arg1, %arg0 : i32, f32, f32) {handshake.name = "br0"}
  ^bb1(%6: i32, %7: f32, %8: f32):  // 2 preds: ^bb0, ^bb3
    llvm.br ^bb2(%8 : f32) {handshake.name = "br1"}
  ^bb2(%9: f32):  // 2 preds: ^bb1, ^bb2
    %10 = llvm.fmul %9, %7  {handshake.name = "fmul0"} : f32
    %11 = llvm.fadd %9, %10  {handshake.name = "fadd0"} : f32
    %12 = llvm.fmul %11, %1  {handshake.name = "fmul1"} : f32
    %13 = llvm.fsub %12, %9  {handshake.name = "fsub0"} : f32
    %14 = llvm.intr.fabs(%13)  {handshake.name = "intr.fabs0"} : (f32) -> f32
    %15 = llvm.fcmp "olt" %14, %2 {handshake.name = "fcmp0"} : f32
    llvm.cond_br %15, ^bb3(%9 : f32), ^bb2(%12 : f32) {handshake.name = "cond_br0"}
  ^bb3(%16: f32):  // pred: ^bb2
    %17 = llvm.fadd %16, %3  {handshake.name = "fadd1"} : f32
    %18 = llvm.fdiv %3, %17  {handshake.name = "fdiv0"} : f32
    %19 = llvm.add %6, %4  {handshake.name = "add0"} : i32
    %20 = llvm.icmp "ult" %19, %5 {handshake.name = "icmp0"} : i32
    llvm.cond_br %20, ^bb1(%19, %18, %17 : i32, f32, f32), ^bb4(%17 : f32) {handshake.name = "cond_br1", loop_annotation = #loop_annotation}
  ^bb4(%21: f32):  // pred: ^bb3
    llvm.return {handshake.name = "return0"} %21 : f32
  }
}

