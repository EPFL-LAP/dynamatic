#loop_annotation = #llvm.loop_annotation<mustProgress = true>
module {
  llvm.func @bisection(%arg0: f32 {llvm.noundef}, %arg1: f32 {llvm.noundef}, %arg2: f32 {llvm.noundef}) -> f32 attributes {handshake.name = "func0", passthrough = ["noinline", "nounwind", ["uwtable", "2"], ["frame-pointer", "all"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    %0 = llvm.mlir.constant(-2.000000e+00 : f32) {handshake.name = "mlir.constant0"} : f32
    %1 = llvm.mlir.constant(0 : i32) {handshake.name = "mlir.constant1"} : i32
    %2 = llvm.mlir.constant(5.000000e-01 : f32) {handshake.name = "mlir.constant2"} : f32
    %3 = llvm.mlir.constant(0.000000e+00 : f32) {handshake.name = "mlir.constant3"} : f32
    %4 = llvm.mlir.constant(1 : i32) {handshake.name = "mlir.constant4"} : i32
    %5 = llvm.mlir.constant(100 : i32) {handshake.name = "mlir.constant5"} : i32
    %6 = llvm.fmul %arg0, %arg0  {handshake.name = "fmul0"} : f32
    %7 = llvm.fadd %6, %0  {handshake.name = "fadd0"} : f32
    llvm.br ^bb1(%arg0, %arg1, %1, %7 : f32, f32, i32, f32) {handshake.name = "br0"}
  ^bb1(%8: f32, %9: f32, %10: i32, %11: f32):  // 2 preds: ^bb0, ^bb3
    %12 = llvm.fadd %8, %9  {handshake.name = "fadd1"} : f32
    %13 = llvm.fmul %12, %2  {handshake.name = "fmul1"} : f32
    %14 = llvm.fmul %13, %13  {handshake.name = "fmul2"} : f32
    %15 = llvm.fadd %14, %0  {handshake.name = "fadd2"} : f32
    %16 = llvm.intr.fabs(%15)  {handshake.name = "intr.fabs0"} : (f32) -> f32
    %17 = llvm.fcmp "olt" %16, %arg2 {handshake.name = "fcmp0"} : f32
    llvm.cond_br %17, ^bb4(%13 : f32), ^bb2 {handshake.name = "cond_br0"}
  ^bb2:  // pred: ^bb1
    %18 = llvm.fsub %9, %8  {handshake.name = "fsub0"} : f32
    %19 = llvm.fmul %18, %2  {handshake.name = "fmul3"} : f32
    %20 = llvm.fcmp "olt" %19, %arg2 {handshake.name = "fcmp1"} : f32
    llvm.cond_br %20, ^bb4(%13 : f32), ^bb3 {handshake.name = "cond_br1"}
  ^bb3:  // pred: ^bb2
    %21 = llvm.fmul %11, %15  {handshake.name = "fmul4"} : f32
    %22 = llvm.fcmp "olt" %21, %3 {handshake.name = "fcmp2"} : f32
    %23 = llvm.select %22, %13, %9 {handshake.name = "select0"} : i1, f32
    %24 = llvm.select %22, %8, %13 {handshake.name = "select1"} : i1, f32
    %25 = llvm.select %22, %11, %15 {handshake.name = "select2"} : i1, f32
    %26 = llvm.add %10, %4  {handshake.name = "add0"} : i32
    %27 = llvm.icmp "ult" %26, %5 {handshake.name = "icmp0"} : i32
    llvm.cond_br %27, ^bb1(%24, %23, %26, %25 : f32, f32, i32, f32), ^bb4(%3 : f32) {handshake.name = "cond_br2", loop_annotation = #loop_annotation}
  ^bb4(%28: f32):  // 3 preds: ^bb1, ^bb2, ^bb3
    llvm.return {handshake.name = "return0"} %28 : f32
  }
}

