module {
  handshake.func @sup_gamma_rhs(%a1: !handshake.channel<i1>, %a2: !handshake.channel<i1>, %c1: !handshake.channel<i1>, %c2: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>) attributes {argNames = ["a1", "a2", "c1", "c2"], resNames = ["b"]} {
    %b = mux %c2 [%a2, %a1] {handshake.name = "mux"} : <i1>, [<i1>, <i1>] to <i1>
    %b_passer = passer %b [%c1] {handshake.name = "passer"} : <i1>, <i1>
    end {handshake.name = "end0"} %b_passer : <i1>
  }
}
