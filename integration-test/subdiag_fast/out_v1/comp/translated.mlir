#loop_annotation = #llvm.loop_annotation<mustProgress = true>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>} {
  llvm.func @subdiag_fast(%arg0: !llvm.ptr {llvm.noundef}, %arg1: !llvm.ptr {llvm.noundef}, %arg2: !llvm.ptr {llvm.noundef}) -> i32 attributes {passthrough = ["noinline", "nounwind", ["uwtable", "2"], ["frame-pointer", "all"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    %0 = llvm.mlir.constant(0 : i32) : i32
    %1 = llvm.mlir.constant(1 : i32) : i32
    %2 = llvm.mlir.constant(1.000000e-03 : f32) : f32
    %3 = llvm.mlir.constant(998 : i32) : i32
    %4 = llvm.mlir.constant(false) : i1
    llvm.br ^bb1(%0 : i32)
  ^bb1(%5: i32):  // 2 preds: ^bb0, ^bb1
    %6 = llvm.zext %5 : i32 to i64
    %7 = llvm.getelementptr inbounds %arg0[%6] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %8 = llvm.load %7 {alignment = 4 : i64} : !llvm.ptr -> f32
    %9 = llvm.add %5, %1  : i32
    %10 = llvm.zext %9 : i32 to i64
    %11 = llvm.getelementptr inbounds %arg1[%10] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %12 = llvm.load %11 {alignment = 4 : i64} : !llvm.ptr -> f32
    %13 = llvm.fadd %8, %12  : f32
    %14 = llvm.add %5, %1  : i32
    %15 = llvm.zext %5 : i32 to i64
    %16 = llvm.getelementptr inbounds %arg2[%15] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %17 = llvm.load %16 {alignment = 4 : i64} : !llvm.ptr -> f32
    %18 = llvm.fmul %13, %2  : f32
    %19 = llvm.fcmp "ugt" %17, %18 : f32
    %20 = llvm.icmp "ult" %5, %3 : i32
    %21 = llvm.select %20, %19, %4 : i1, i1
    llvm.cond_br %21, ^bb1(%14 : i32), ^bb2(%5 : i32) {loop_annotation = #loop_annotation}
  ^bb2(%22: i32):  // pred: ^bb1
    llvm.return %22 : i32
  }
  llvm.func @main() -> i32 attributes {passthrough = ["noinline", "nounwind", ["uwtable", "2"], ["frame-pointer", "all"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(13 : i32) : i32
    %2 = llvm.mlir.constant(0 : i32) : i32
    %3 = llvm.mlir.constant(0 : i64) : i64
    %4 = llvm.mlir.constant(3.000000e+02 : f32) : f32
    %5 = llvm.mlir.constant(1.000000e-03 : f32) : f32
    %6 = llvm.mlir.constant(1000 : i32) : i32
    %7 = llvm.alloca %0 x !llvm.array<1000 x f32> {alignment = 16 : i64} : (i32) -> !llvm.ptr
    %8 = llvm.alloca %0 x !llvm.array<1000 x f32> {alignment = 16 : i64} : (i32) -> !llvm.ptr
    %9 = llvm.alloca %0 x !llvm.array<1000 x f32> {alignment = 16 : i64} : (i32) -> !llvm.ptr
    llvm.call @srand(%1) : (i32) -> ()
    llvm.br ^bb1(%2 : i32)
  ^bb1(%10: i32):  // 2 preds: ^bb0, ^bb1
    %11 = llvm.sitofp %10 : i32 to f32
    %12 = llvm.zext %10 : i32 to i64
    %13 = llvm.getelementptr inbounds %7[%3, %12] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.array<1000 x f32>
    llvm.store %11, %13 {alignment = 4 : i64} : f32, !llvm.ptr
    %14 = llvm.sitofp %10 : i32 to f32
    %15 = llvm.zext %10 : i32 to i64
    %16 = llvm.getelementptr inbounds %8[%3, %15] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.array<1000 x f32>
    llvm.store %14, %16 {alignment = 4 : i64} : f32, !llvm.ptr
    %17 = llvm.sitofp %10 : i32 to f32
    %18 = llvm.fsub %4, %17  : f32
    %19 = llvm.fmul %18, %5  : f32
    %20 = llvm.zext %10 : i32 to i64
    %21 = llvm.getelementptr inbounds %9[%3, %20] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.array<1000 x f32>
    llvm.store %19, %21 {alignment = 4 : i64} : f32, !llvm.ptr
    %22 = llvm.add %10, %0  : i32
    %23 = llvm.icmp "ult" %22, %6 : i32
    llvm.cond_br %23, ^bb1(%22 : i32), ^bb2 {loop_annotation = #loop_annotation}
  ^bb2:  // pred: ^bb1
    %24 = llvm.call @subdiag_fast(%7, %8, %9) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
    llvm.return %2 : i32
  }
  llvm.func @srand(i32 {llvm.noundef}) attributes {passthrough = ["nounwind", ["frame-pointer", "all"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]}
}
