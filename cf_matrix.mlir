module {
  func.func @matrix(%arg0: memref<30x30xi32>, %arg1: memref<30x30xi32>, %arg2: memref<30x30xi32>) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c30_i32 = arith.constant 30 : i32
    cf.br ^bb1(%c0_i32 : i32)
  ^bb1(%0: i32):  // 2 preds: ^bb0, ^bb5
    cf.br ^bb2(%c0_i32 : i32)
  ^bb2(%1: i32):  // 2 preds: ^bb1, ^bb4
    cf.br ^bb3(%c0_i32, %c0_i32 : i32, i32)
  ^bb3(%2: i32, %3: i32):  // 2 preds: ^bb2, ^bb3
    %4 = arith.extui %0 : i32 to i64
    %5 = arith.extui %2 : i32 to i64
    %6 = arith.index_cast %4 : i64 to index
    %7 = arith.index_cast %5 : i64 to index
    %8 = memref.load %arg0[%6, %7] {handshake.name = "load0"} : memref<30x30xi32>
    %9 = arith.extui %2 : i32 to i64
    %10 = arith.extui %1 : i32 to i64
    %11 = arith.index_cast %9 : i64 to index
    %12 = arith.index_cast %10 : i64 to index
    %13 = memref.load %arg1[%11, %12] {handshake.name = "load1"} : memref<30x30xi32>
    %14 = arith.muli %8, %13 : i32
    %15 = arith.addi %3, %14 : i32
    %16 = arith.addi %2, %c1_i32 : i32
    %17 = arith.cmpi ult, %16, %c30_i32 : i32
    cf.cond_br %17, ^bb3(%16, %15 : i32, i32), ^bb4(%15 : i32)
  ^bb4(%18: i32):  // pred: ^bb3
    %19 = arith.extui %0 : i32 to i64
    %20 = arith.extui %1 : i32 to i64
    %21 = arith.index_cast %19 : i64 to index
    %22 = arith.index_cast %20 : i64 to index
    memref.store %18, %arg2[%21, %22] {handshake.name = "store2"} : memref<30x30xi32>
    %23 = arith.addi %1, %c1_i32 : i32
    %24 = arith.cmpi ult, %23, %c30_i32 : i32
    cf.cond_br %24, ^bb2(%23 : i32), ^bb5
  ^bb5:  // pred: ^bb4
    %25 = arith.addi %0, %c1_i32 : i32
    %26 = vector.transfer_read %arg1[%11, %12], %c0_i32 {handshake.name = "transfer_read0", in_bounds = [true]} : memref<30x30xi32>, vector<1xi32>
    %27 = vector.extract %26[0] : vector<1xi32> 
    %28 = arith.cmpi ult, %27, %c30_i32 : i32
    cf.cond_br %28, ^bb1(%25 : i32), ^bb6
  ^bb6:  // pred: ^bb5
    return
  }
}
