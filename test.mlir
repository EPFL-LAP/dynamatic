module {
  func.func @mat_mul(%arg0: memref<512x32xi32> {handshake.arg_name = "A"}, %arg1: memref<32x512xi32> {handshake.arg_name = "B"}, %arg2: memref<512x512xi32> {handshake.arg_name = "C"}) {
	%c1_i32 = arith.constant {handshake.name = "constant0"} 8 : i32
	%c2_i32 = arith.constant {handshake.name = "constant1"} 8 : i32
	%0 = call @fpsa_mat_mul(%arg0, %arg1, %arg2, %c1_i32, %c2_i32) {handshake.name = "call0"} : (memref<512x32xi32>, memref<32x512xi32>, memref<512x512xi32>, i32, i32) -> i32
	return {handshake.name = "return0"}
  }
 func.func private @fpsa_mat_mul(memref<512x32xi32> {handshake.arg_name = "M"}, memref<32x512xi32> {handshake.arg_name = "N"}, memref<512x512xi32> {handshake.arg_name = "P"}, i32 {handshake.arg_name = "O"}, i32 {handshake.arg_name = "Q"}) -> i32
}

