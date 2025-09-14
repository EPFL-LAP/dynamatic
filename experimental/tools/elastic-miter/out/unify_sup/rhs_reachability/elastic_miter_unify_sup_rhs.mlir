module {
  handshake.func @unify_sup_rhs(%arg0: !handshake.channel<i1>, %arg1: !handshake.channel<i1>, %arg2: !handshake.channel<i1>, ...) -> !handshake.channel<i1> attributes {argNames = ["A_in", "Cond1", "Cond2"], resNames = ["B_out"]} {
    %0 = andi %arg1, %arg2 {handshake.bb = 1 : ui32, handshake.name = "andi"} : <i1>
    %1 = passer %arg0[%0] {handshake.bb = 1 : ui32, handshake.name = "passer"} : <i1>, <i1>
    end {handshake.bb = 1 : ui32, handshake.name = "end0"} %1 : <i1>
  }
}
