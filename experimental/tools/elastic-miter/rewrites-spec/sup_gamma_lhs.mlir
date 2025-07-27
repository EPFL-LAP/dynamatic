module {
  handshake.func @sup_gamma_lhs(%a1: !handshake.channel<i1>, %a2: !handshake.channel<i1>, %c1: !handshake.channel<i1>, %c2: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>) attributes {argNames = ["a1", "a2", "c1", "c2"], resNames = ["b"]} {
    %c12_1 = passer %c1 [%c2] {handshake.name = "passer1"} : <i1>, <i1>
    %c2_not = not %c2 {handshake.name = "not"} : <i1>
    %c12_2 = passer %c1 [%c2_not] {handshake.name = "passer2"} : <i1>, <i1>
    %c21 = passer %c2 [%c1] {handshake.name = "passer3"} : <i1>, <i1>
    %a1_passer = passer %a1 [%c12_1] {handshake.name = "passer_a1"} : <i1>, <i1>
    %a2_passer = passer %a2 [%c12_2] {handshake.name = "passer_a2"} : <i1>, <i1>
    %b = mux %c21 [%a2_passer, %a1_passer] {handshake.name = "mux"} : <i1>, [<i1>, <i1>] to <i1>
    end {handshake.name = "end0"} %b : <i1>
  }
}
