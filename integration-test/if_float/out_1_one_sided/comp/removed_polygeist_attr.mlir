#loop_annotation = #llvm.loop_annotation<mustProgress = true>
module {
  llvm.func @if_float(%arg0: f32 {llvm.noundef}, %arg1: !llvm.ptr {llvm.noundef}, %arg2: !llvm.ptr {llvm.noundef}) -> f32 attributes {handshake.name = "func0", passthrough = ["noinline", "nounwind", ["uwtable", "2"], ["frame-pointer", "all"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    %0 = llvm.mlir.constant(0 : i32) {handshake.name = "mlir.constant0"} : i32
    %1 = llvm.mlir.constant(-0.899999976 : f32) {handshake.name = "mlir.constant1"} : f32
    %2 = llvm.mlir.constant(0.000000e+00 : f32) {handshake.name = "mlir.constant2"} : f32
    %3 = llvm.mlir.constant(1.100000e+00 : f32) {handshake.name = "mlir.constant3"} : f32
    %4 = llvm.mlir.constant(1 : i32) {handshake.name = "mlir.constant4"} : i32
    %5 = llvm.mlir.constant(100 : i32) {handshake.name = "mlir.constant5"} : i32
    llvm.br ^bb1(%0, %arg0 : i32, f32) {handshake.name = "br0"}
  ^bb1(%6: i32, %7: f32):  // 2 preds: ^bb0, ^bb4
    %8 = llvm.zext %6 {handshake.name = "zext0"} : i32 to i64
    %9 = llvm.getelementptr inbounds %arg1[%8] {handshake.name = "getelementptr0"} : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %10 = llvm.load %9 {alignment = 4 : i64, handshake.name = "load0"} : !llvm.ptr -> f32
    %11 = llvm.fmul %10, %7  {handshake.name = "fmul0"} : f32
    %12 = llvm.fmul %7, %1  {handshake.name = "fmul1"} : f32
    %13 = llvm.fadd %11, %12  {handshake.name = "fadd0"} : f32
    %14 = llvm.fcmp "ugt" %13, %2 {handshake.name = "fcmp0"} : f32
    llvm.cond_br %14, ^bb3, ^bb2 {handshake.name = "cond_br0"}
  ^bb2:  // pred: ^bb1
    %15 = llvm.fmul %7, %3  {handshake.name = "fmul2"} : f32
    llvm.br ^bb4(%15 : f32) {handshake.name = "br1"}
  ^bb3:  // pred: ^bb1
    %16 = llvm.zext %6 {handshake.name = "zext1"} : i32 to i64
    %17 = llvm.getelementptr inbounds %arg2[%16] {handshake.name = "getelementptr1"} : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %7, %17 {alignment = 4 : i64, handshake.name = "store0"} : f32, !llvm.ptr
    %18 = llvm.fdiv %7, %3  {handshake.name = "fdiv0"} : f32
    llvm.br ^bb4(%18 : f32) {handshake.name = "br2"}
  ^bb4(%19: f32):  // 2 preds: ^bb2, ^bb3
    %20 = llvm.fadd %19, %19  {handshake.name = "fadd1"} : f32
    %21 = llvm.add %6, %4  {handshake.name = "add0"} : i32
    %22 = llvm.icmp "ult" %21, %5 {handshake.name = "icmp0"} : i32
    llvm.cond_br %22, ^bb1(%21, %20 : i32, f32), ^bb5(%20 : f32) {handshake.name = "cond_br1", loop_annotation = #loop_annotation}
  ^bb5(%23: f32):  // pred: ^bb4
    llvm.return {handshake.name = "return0"} %23 : f32
  }
}

