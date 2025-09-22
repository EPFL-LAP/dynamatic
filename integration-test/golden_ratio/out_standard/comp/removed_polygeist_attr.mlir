#loop_annotation = #llvm.loop_annotation<mustProgress = true>
module {
  llvm.func @golden_ratio(%arg0: f32 {llvm.noundef}) -> f32 attributes {handshake.name = "func0", passthrough = ["noinline", "nounwind", ["uwtable", "2"], ["frame-pointer", "all"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    %0 = llvm.mlir.constant(1.000000e+00 : f32) {handshake.name = "mlir.constant0"} : f32
    %1 = llvm.mlir.constant(0 : i32) {handshake.name = "mlir.constant1"} : i32
    %2 = llvm.mlir.constant(5.000000e-01 : f32) {handshake.name = "mlir.constant2"} : f32
    %3 = llvm.mlir.constant(1.000000e-01 : f32) {handshake.name = "mlir.constant3"} : f32
    %4 = llvm.mlir.constant(1 : i32) {handshake.name = "mlir.constant4"} : i32
    %5 = llvm.mlir.constant(100 : i32) {handshake.name = "mlir.constant5"} : i32
    %6 = llvm.fdiv %0, %arg0  {handshake.name = "fdiv0"} : f32
    llvm.br ^bb1(%6, %1, %arg0 : f32, i32, f32) {handshake.name = "br0"}
  ^bb1(%7: f32, %8: i32, %9: f32):  // 2 preds: ^bb0, ^bb3
    llvm.br ^bb2(%9 : f32) {handshake.name = "br1"}
  ^bb2(%10: f32):  // 2 preds: ^bb1, ^bb2
    %11 = llvm.fmul %10, %7  {handshake.name = "fmul0"} : f32
    %12 = llvm.fadd %10, %11  {handshake.name = "fadd0"} : f32
    %13 = llvm.fmul %12, %2  {handshake.name = "fmul1"} : f32
    %14 = llvm.fsub %13, %10  {handshake.name = "fsub0"} : f32
    %15 = llvm.intr.fabs(%14)  {handshake.name = "intr.fabs0"} : (f32) -> f32
    %16 = llvm.fcmp "olt" %15, %3 {handshake.name = "fcmp0"} : f32
    llvm.cond_br %16, ^bb3(%10 : f32), ^bb2(%13 : f32) {handshake.name = "cond_br0"}
  ^bb3(%17: f32):  // pred: ^bb2
    %18 = llvm.fadd %17, %0  {handshake.name = "fadd1"} : f32
    %19 = llvm.add %8, %4  {handshake.name = "add0"} : i32
    %20 = llvm.fdiv %0, %18  {handshake.name = "fdiv1"} : f32
    %21 = llvm.icmp "ult" %19, %5 {handshake.name = "icmp0"} : i32
    llvm.cond_br %21, ^bb1(%20, %19, %18 : f32, i32, f32), ^bb4(%18 : f32) {handshake.name = "cond_br1", loop_annotation = #loop_annotation}
  ^bb4(%22: f32):  // pred: ^bb3
    llvm.return {handshake.name = "return0"} %22 : f32
  }
}

