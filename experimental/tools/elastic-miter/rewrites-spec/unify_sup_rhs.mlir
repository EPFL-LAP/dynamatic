module {
  handshake.func @unify_sup_rhs(%a: !handshake.channel<i1>, %cond1: !handshake.channel<i1>, %cond2: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>) attributes {argNames = ["A_in", "Cond1", "Cond2"], resNames = ["B_out"]} {
    %cond_and = andi %cond1, %cond2 {handshake.bb = 1 : ui32, handshake.name = "andi"} : <i1>
    %b = passer %a [%cond_and] {handshake.bb = 1 : ui32, handshake.name = "passer"} : <i1>, <i1>

    end {handshake.bb = 1 : ui32, handshake.name = "end0"} %b : <i1>
  }
}
