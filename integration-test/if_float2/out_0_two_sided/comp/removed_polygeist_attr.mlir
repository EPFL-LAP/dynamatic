#loop_annotation = #llvm.loop_annotation<mustProgress = true>
module {
  llvm.func @if_float2(%arg0: f32 {llvm.noundef}, %arg1: !llvm.ptr {llvm.noundef}, %arg2: !llvm.ptr {llvm.noundef}) -> f32 attributes {handshake.name = "func0", passthrough = ["noinline", "nounwind", ["uwtable", "2"], ["frame-pointer", "all"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    %0 = llvm.mlir.constant(0 : i32) {handshake.name = "mlir.constant0"} : i32
    %1 = llvm.mlir.constant(-0.899999976 : f32) {handshake.name = "mlir.constant1"} : f32
    %2 = llvm.mlir.constant(0.000000e+00 : f32) {handshake.name = "mlir.constant2"} : f32
    %3 = llvm.mlir.constant(3.000000e+00 : f32) {handshake.name = "mlir.constant3"} : f32
    %4 = llvm.mlir.constant(1.000000e+00 : f32) {handshake.name = "mlir.constant4"} : f32
    %5 = llvm.mlir.constant(1 : i32) {handshake.name = "mlir.constant5"} : i32
    %6 = llvm.mlir.constant(100 : i32) {handshake.name = "mlir.constant6"} : i32
    llvm.br ^bb1(%0, %arg0 : i32, f32) {handshake.name = "br0"}
  ^bb1(%7: i32, %8: f32):  // 2 preds: ^bb0, ^bb4
    %9 = llvm.zext %7 {handshake.name = "zext0"} : i32 to i64
    %10 = llvm.getelementptr inbounds %arg1[%9] {handshake.name = "getelementptr0"} : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %11 = llvm.load %10 {alignment = 4 : i64, handshake.name = "load0"} : !llvm.ptr -> f32
    %12 = llvm.fmul %11, %8  {handshake.name = "fmul0"} : f32
    %13 = llvm.fmul %8, %1  {handshake.name = "fmul1"} : f32
    %14 = llvm.fadd %12, %13  {handshake.name = "fadd0"} : f32
    %15 = llvm.fcmp "ugt" %14, %2 {handshake.name = "fcmp0"} : f32
    llvm.cond_br %15, ^bb3, ^bb2 {handshake.name = "cond_br0"}
  ^bb2:  // pred: ^bb1
    %16 = llvm.fadd %8, %3  {handshake.name = "fadd1"} : f32
    llvm.br ^bb4(%16 : f32) {handshake.name = "br1"}
  ^bb3:  // pred: ^bb1
    %17 = llvm.zext %7 {handshake.name = "zext1"} : i32 to i64
    %18 = llvm.getelementptr inbounds %arg2[%17] {handshake.name = "getelementptr1"} : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %8, %18 {alignment = 4 : i64, handshake.name = "store0"} : f32, !llvm.ptr
    %19 = llvm.fadd %8, %4  {handshake.name = "fadd2"} : f32
    llvm.br ^bb4(%19 : f32) {handshake.name = "br2"}
  ^bb4(%20: f32):  // 2 preds: ^bb2, ^bb3
    %21 = llvm.fdiv %4, %20  {handshake.name = "fdiv0"} : f32
    %22 = llvm.add %7, %5  {handshake.name = "add0"} : i32
    %23 = llvm.icmp "ult" %22, %6 {handshake.name = "icmp0"} : i32
    llvm.cond_br %23, ^bb1(%22, %21 : i32, f32), ^bb5(%21 : f32) {handshake.name = "cond_br1", loop_annotation = #loop_annotation}
  ^bb5(%24: f32):  // pred: ^bb4
    llvm.return {handshake.name = "return0"} %24 : f32
  }
}

