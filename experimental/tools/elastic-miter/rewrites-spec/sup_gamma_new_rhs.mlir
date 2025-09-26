module {
  handshake.func @sup_gamma_new_rhs(%a1: !handshake.channel<i1>, %a2: !handshake.channel<i1>, %c1: !handshake.channel<i1>, %c2: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>) attributes {argNames = ["a1", "a2", "c1", "c2"], resNames = ["b"]} {
    %c2_forked:3 = fork [3] %c2 {handshake.name = "c2_fork"} : <i1>
    %a1_passed = passer %a1 [%c2_forked#0] {handshake.name = "a1_passer"} : <i1>, <i1>
    %c2_not = not %c2_forked#1 {handshake.name = "not"} : <i1>
    %a2_passed = passer %a2 [%c2_not] {handshake.name = "a2_passer"} : <i1>, <i1>
    %b = mux %c2_forked#2 [%a2_passed, %a1_passed] {handshake.name = "mux"} : <i1>, [<i1>, <i1>] to <i1>
    %b_passed = passer %b [%c1] {handshake.name = "b_passer"} : <i1>, <i1>
    end {handshake.name = "end0"} %b_passed : <i1>
  }
}
