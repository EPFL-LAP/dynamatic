#loop_annotation = #llvm.loop_annotation<mustProgress = true>
module {
  llvm.func @subdiag_fast(%arg0: !llvm.ptr {llvm.noundef}, %arg1: !llvm.ptr {llvm.noundef}, %arg2: !llvm.ptr {llvm.noundef}) -> i32 attributes {handshake.name = "func0", passthrough = ["noinline", "nounwind", ["uwtable", "2"], ["frame-pointer", "all"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    %0 = llvm.mlir.constant(0 : i32) {handshake.name = "mlir.constant0"} : i32
    %1 = llvm.mlir.constant(1 : i32) {handshake.name = "mlir.constant1"} : i32
    %2 = llvm.mlir.constant(1.000000e-03 : f32) {handshake.name = "mlir.constant2"} : f32
    %3 = llvm.mlir.constant(998 : i32) {handshake.name = "mlir.constant3"} : i32
    %4 = llvm.mlir.constant(false) {handshake.name = "mlir.constant4"} : i1
    llvm.br ^bb1(%0 : i32) {handshake.name = "br0"}
  ^bb1(%5: i32):  // 2 preds: ^bb0, ^bb1
    %6 = llvm.zext %5 {handshake.name = "zext0"} : i32 to i64
    %7 = llvm.getelementptr inbounds %arg0[%6] {handshake.name = "getelementptr0"} : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %8 = llvm.load %7 {alignment = 4 : i64, handshake.name = "load0"} : !llvm.ptr -> f32
    %9 = llvm.add %5, %1  {handshake.name = "add0"} : i32
    %10 = llvm.zext %9 {handshake.name = "zext1"} : i32 to i64
    %11 = llvm.getelementptr inbounds %arg1[%10] {handshake.name = "getelementptr1"} : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %12 = llvm.load %11 {alignment = 4 : i64, handshake.name = "load1"} : !llvm.ptr -> f32
    %13 = llvm.fadd %8, %12  {handshake.name = "fadd0"} : f32
    %14 = llvm.add %5, %1  {handshake.name = "add1"} : i32
    %15 = llvm.zext %5 {handshake.name = "zext2"} : i32 to i64
    %16 = llvm.getelementptr inbounds %arg2[%15] {handshake.name = "getelementptr2"} : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %17 = llvm.load %16 {alignment = 4 : i64, handshake.name = "load2"} : !llvm.ptr -> f32
    %18 = llvm.fmul %13, %2  {handshake.name = "fmul0"} : f32
    %19 = llvm.fcmp "ugt" %17, %18 {handshake.name = "fcmp0"} : f32
    %20 = llvm.icmp "ult" %5, %3 {handshake.name = "icmp0"} : i32
    %21 = llvm.select %20, %19, %4 {handshake.name = "select0"} : i1, i1
    llvm.cond_br %21, ^bb1(%14 : i32), ^bb2(%5 : i32) {handshake.name = "cond_br0", loop_annotation = #loop_annotation}
  ^bb2(%22: i32):  // pred: ^bb1
    llvm.return {handshake.name = "return0"} %22 : i32
  }
}

