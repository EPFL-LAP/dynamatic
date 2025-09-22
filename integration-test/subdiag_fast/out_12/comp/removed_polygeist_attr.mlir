#loop_annotation = #llvm.loop_annotation<mustProgress = true>
module {
  llvm.func @subdiag_fast(%arg0: !llvm.ptr {llvm.noundef}, %arg1: !llvm.ptr {llvm.noundef}, %arg2: !llvm.ptr {llvm.noundef}) -> i32 attributes {handshake.name = "func0", passthrough = ["noinline", "nounwind", ["uwtable", "2"], ["frame-pointer", "all"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    %0 = llvm.mlir.constant(0 : i32) {handshake.name = "mlir.constant0"} : i32
    %1 = llvm.mlir.constant(1.000000e-03 : f32) {handshake.name = "mlir.constant1"} : f32
    %2 = llvm.mlir.constant(1 : i32) {handshake.name = "mlir.constant2"} : i32
    %3 = llvm.mlir.constant(999 : i32) {handshake.name = "mlir.constant3"} : i32
    llvm.br ^bb1(%0 : i32) {handshake.name = "br0"}
  ^bb1(%4: i32):  // 2 preds: ^bb0, ^bb2
    %5 = llvm.zext %4 {handshake.name = "zext0"} : i32 to i64
    %6 = llvm.getelementptr inbounds %arg0[%5] {handshake.name = "getelementptr0"} : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %7 = llvm.load %6 {alignment = 4 : i64, handshake.name = "load0"} : !llvm.ptr -> f32
    %8 = llvm.zext %4 {handshake.name = "zext1"} : i32 to i64
    %9 = llvm.getelementptr inbounds %arg1[%8] {handshake.name = "getelementptr1"} : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %10 = llvm.load %9 {alignment = 4 : i64, handshake.name = "load1"} : !llvm.ptr -> f32
    %11 = llvm.fadd %7, %10  {handshake.name = "fadd0"} : f32
    %12 = llvm.zext %4 {handshake.name = "zext2"} : i32 to i64
    %13 = llvm.getelementptr inbounds %arg2[%12] {handshake.name = "getelementptr2"} : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %14 = llvm.load %13 {alignment = 4 : i64, handshake.name = "load2"} : !llvm.ptr -> f32
    %15 = llvm.fmul %11, %1  {handshake.name = "fmul0"} : f32
    %16 = llvm.fcmp "ugt" %14, %15 {handshake.name = "fcmp0"} : f32
    llvm.cond_br %16, ^bb2, ^bb3(%4 : i32) {handshake.name = "cond_br0"}
  ^bb2:  // pred: ^bb1
    %17 = llvm.add %4, %2  {handshake.name = "add0"} : i32
    %18 = llvm.icmp "ult" %17, %3 {handshake.name = "icmp0"} : i32
    llvm.cond_br %18, ^bb1(%17 : i32), ^bb3(%17 : i32) {handshake.name = "cond_br1", loop_annotation = #loop_annotation}
  ^bb3(%19: i32):  // 2 preds: ^bb1, ^bb2
    llvm.return {handshake.name = "return0"} %19 : i32
  }
}

