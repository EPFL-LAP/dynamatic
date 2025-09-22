#loop_annotation = #llvm.loop_annotation<mustProgress = true>
module {
  llvm.func @collision_donut(%arg0: !llvm.ptr {llvm.noundef}, %arg1: !llvm.ptr {llvm.noundef}) -> i32 attributes {handshake.name = "func0", passthrough = ["noinline", "nounwind", ["uwtable", "2"], ["frame-pointer", "all"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    %0 = llvm.mlir.constant(0 : i32) {handshake.name = "mlir.constant0"} : i32
    %1 = llvm.mlir.constant(4 : i32) {handshake.name = "mlir.constant1"} : i32
    %2 = llvm.mlir.constant(-1 : i32) {handshake.name = "mlir.constant2"} : i32
    %3 = llvm.mlir.constant(19000 : i32) {handshake.name = "mlir.constant3"} : i32
    %4 = llvm.mlir.constant(-2 : i32) {handshake.name = "mlir.constant4"} : i32
    %5 = llvm.mlir.constant(1 : i32) {handshake.name = "mlir.constant5"} : i32
    %6 = llvm.mlir.constant(1000 : i32) {handshake.name = "mlir.constant6"} : i32
    llvm.br ^bb1(%0 : i32) {handshake.name = "br0"}
  ^bb1(%7: i32):  // 2 preds: ^bb0, ^bb3
    %8 = llvm.zext %7 {handshake.name = "zext0"} : i32 to i64
    %9 = llvm.getelementptr inbounds %arg0[%8] {handshake.name = "getelementptr0"} : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %10 = llvm.load %9 {alignment = 4 : i64, handshake.name = "load0"} : !llvm.ptr -> i32
    %11 = llvm.zext %7 {handshake.name = "zext1"} : i32 to i64
    %12 = llvm.getelementptr inbounds %arg1[%11] {handshake.name = "getelementptr1"} : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %13 = llvm.load %12 {alignment = 4 : i64, handshake.name = "load1"} : !llvm.ptr -> i32
    %14 = llvm.mul %10, %10  {handshake.name = "mul0"} : i32
    %15 = llvm.mul %13, %13  {handshake.name = "mul1"} : i32
    %16 = llvm.add %14, %15  {handshake.name = "add0"} : i32
    %17 = llvm.icmp "ult" %16, %1 {handshake.name = "icmp0"} : i32
    llvm.cond_br %17, ^bb4(%7, %2 : i32, i32), ^bb2 {handshake.name = "cond_br0"}
  ^bb2:  // pred: ^bb1
    %18 = llvm.icmp "ugt" %16, %3 {handshake.name = "icmp1"} : i32
    llvm.cond_br %18, ^bb4(%7, %4 : i32, i32), ^bb3 {handshake.name = "cond_br1"}
  ^bb3:  // pred: ^bb2
    %19 = llvm.add %7, %5  {handshake.name = "add1"} : i32
    %20 = llvm.icmp "ult" %19, %6 {handshake.name = "icmp2"} : i32
    llvm.cond_br %20, ^bb1(%19 : i32), ^bb4(%19, %0 : i32, i32) {handshake.name = "cond_br2", loop_annotation = #loop_annotation}
  ^bb4(%21: i32, %22: i32):  // 3 preds: ^bb1, ^bb2, ^bb3
    %23 = llvm.shl %21, %5  {handshake.name = "shl0"} : i32
    %24 = llvm.and %23, %22  {handshake.name = "and0"} : i32
    llvm.return {handshake.name = "return0"} %24 : i32
  }
}

