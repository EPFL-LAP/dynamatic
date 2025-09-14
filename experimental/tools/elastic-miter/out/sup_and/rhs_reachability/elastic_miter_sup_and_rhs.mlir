module {
  handshake.func @sup_and_rhs(%arg0: !handshake.channel<i1>, %arg1: !handshake.channel<i1>, %arg2: !handshake.channel<i1>, ...) -> !handshake.channel<i1> attributes {argNames = ["A_in", "B_in", "Cond"], resNames = ["C_out"]} {
    %0 = andi %arg0, %arg1 {handshake.bb = 1 : ui32, handshake.name = "andi"} : <i1>
    %1 = passer %0[%arg2] {handshake.bb = 1 : ui32, handshake.name = "passer1"} : <i1>, <i1>
    end {handshake.bb = 1 : ui32, handshake.name = "end0"} %1 : <i1>
  }
}
